"""喂食任务 PPO 训练 + 评估(论文版 v3).

v3 相对 v2 的改动:
  1. --seed:PPO 和环境都接种子。深度 RL 种子间方差极大,单次训练说明不了问题。
     论文至少需要 3-5 个 seed,报告 mean ± std。
  2. VecNormalize:观测归一化。你的观测里位置约 0.5(m)、关节约 100(deg)、
     力 0-10(N),量纲差两个数量级,MLP 第一层会被大数值维度主导,
     而"位置"恰恰是任务目标。这很可能是 v1 卡在 0.4m 的真正原因。
     归一化统计量会随模型一起保存,评估时必须加载同一份,否则结果无效。
  3. tensorboard_log:出学习曲线(论文 Figure)。
  4. CheckpointCallback:每 N 步存档。10 小时训练无中间存档风险太大。
  5. 评估报告 mean ± std,并导出 CSV/JSON 供论文表格直接引用。
  6. 评估逐 episode 记录,不再只保留最后一局的 info。
  7. device 默认 cpu:MlpPolicy 网络很小,GPU 搬运开销大于收益;
     你的瓶颈是 Unity 仿真(约 4 fps),不是网络前向。

运行:
  conda activate rcareworld
  cd ~/research/my_feeding_project

  # 冒烟测试(先跑这个,10 分钟内确认不报错)
  python train_feeding_ppo_paper.py --mode train --shield 0 --steps 3000 \
      --seed 0 --out smoke_test

  # 正式训练(每个条件跑多个 seed)
  python train_feeding_ppo_paper.py --mode train --shield 0 --steps 150000 \
      --seed 0 --out ppo_noshield_s0
  python train_feeding_ppo_paper.py --mode train --shield 1 --steps 150000 \
      --seed 0 --out ppo_shield_s0

  # 评估
  python train_feeding_ppo_paper.py --mode eval --shield 0 \
      --model ppo_noshield_s0 --episodes 20 --seed 100

  # 学习曲线
  tensorboard --logdir ./tb/
"""

import argparse
import csv
import json
import os

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from feeding_task_env import FeedingTaskEnv, check_thresholds


# 论文表格里要报告的指标,以及它们的显示名
METRIC_KEYS = [
    ("success",              "成功率 (Success Rate)",          "%"),
    ("avg_force",            "平均接触力 (Avg Force)",          "N"),
    ("max_force",            "最大接触力 (Max Force)",          "N"),
    ("impulse",              "力-时间积分 (Impulse)",           "N·step"),
    ("force_violations",     "舒适违规步数 (>6N)",              "steps"),
    ("highforce_violations", "危险违规步数 (>10N)",             "steps"),
    ("shield_triggers",      "护盾触发次数 (Shield Triggers)",  "steps"),
    ("min_distance",         "最近距离 (Min Distance)",         "m"),
    ("final_distance",       "结束距离 (Final Distance)",       "m"),
    ("ep_length",            "回合长度 (Episode Length)",       "steps"),
]


def vecnorm_path(base_path):
    """归一化统计量的存放路径,与模型同名不同后缀。"""
    return f"{base_path}_vecnorm.pkl"


def make_env(use_shield, graphics=False, seed=None, check=False):
    """返回一份"造环境的配方"(闭包),而不是环境本身。

    SB3 需要 callable,以便在需要时(多进程并行)重复构造环境。
    """
    def _f():
        env = FeedingTaskEnv(
            use_shield=use_shield, graphics=graphics, seed=seed, check=check
        )
        return Monitor(env)
    return _f


# ============================== 训练 ==============================
def train(use_shield, total_steps, out_path, seed=0, graphics=False,
          device="cpu", ckpt_freq=10000, tb_dir="./tb"):
    tag = "shield" if use_shield else "noshield"

    venv = DummyVecEnv([make_env(use_shield, graphics=graphics, seed=seed, check=True)])
    # 观测归一化。norm_reward=False:奖励量纲本身有物理含义(距离/力),
    # 归一化会让不同条件间的 reward 不可比,也会干扰安全惩罚的相对权重。
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", venv,
        verbose=1,
        device=device,
        seed=seed,
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        tensorboard_log=tb_dir,
        # 以下为 SB3 默认值,显式写出便于论文 Implementation Details 直接引用
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    ckpt_dir = f"{out_path}_ckpt"
    callback = CheckpointCallback(
        save_freq=ckpt_freq, save_path=ckpt_dir,
        name_prefix="ckpt", save_vecnormalize=True,
    )

    print(f"\n训练开始: shield={use_shield}, seed={seed}, steps={total_steps}")
    print(f"  存档目录: {ckpt_dir}/  (每 {ckpt_freq} 步)")
    print(f"  tensorboard: {tb_dir}/\n")

    model.learn(
        total_timesteps=total_steps,
        tb_log_name=f"{tag}_seed{seed}",
        callback=callback,
    )

    model.save(out_path)
    venv.save(vecnorm_path(out_path))      # 必须保存,否则评估无法复现
    print(f"\n训练完成")
    print(f"  模型:       {out_path}.zip")
    print(f"  归一化统计: {vecnorm_path(out_path)}")
    venv.close()


# ============================== 评估 ==============================
def evaluate(use_shield, model_path, episodes, seed=100, graphics=False,
             out_csv=None):
    check_thresholds(verbose=True)

    venv = DummyVecEnv([make_env(use_shield, graphics=graphics, check=False)])

    vn_path = vecnorm_path(model_path)
    if os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
        venv.training = False              # 评估时冻结统计量,不再更新
        venv.norm_reward = False
        print(f"[已加载归一化统计] {vn_path}")
    else:
        print(f"[警告] 未找到 {vn_path}。")
        print("       若该模型是带 VecNormalize 训练的,评估结果无效。")

    model = PPO.load(model_path, device="cpu")

    # 一次性接种子:后续每次 auto-reset 都会推进随机数流,
    # 于是各 episode 初始状态不同,但整体序列可复现。
    venv.seed(seed)
    obs = venv.reset()

    all_metrics = []
    ep = 0
    safety_counter = 0
    max_iters = episodes * 5000            # 防死循环

    while ep < episodes and safety_counter < max_iters:
        safety_counter += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)

        if dones[0]:
            m = infos[0].get("episode_metrics", {})
            all_metrics.append(m)
            print(
                f"ep {ep:2d}: success={int(m.get('success', 0))} "
                f"min_dist={m.get('min_distance', float('nan')):.4f}m "
                f"avg_F={m.get('avg_force', 0):.2f}N "
                f"max_F={m.get('max_force', 0):.2f}N "
                f"viol>10N={m.get('highforce_violations', 0)} "
                f"len={m.get('ep_length', 0)}"
            )
            ep += 1
            # VecEnv 已自动 reset,obs 就是下一局的首帧,不需要手动 reset

    venv.close()

    # ---------- 汇总: mean ± std ----------
    def stat(key):
        vals = [m.get(key, 0) for m in all_metrics if key in m]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    label = "有护盾 (Shielded)" if use_shield else "无护盾 (Baseline)"
    print("\n" + "=" * 64)
    print(f"评估汇总: {label} | {len(all_metrics)} episodes | eval_seed={seed}")
    print("=" * 64)

    rows = []
    for key, name, unit in METRIC_KEYS:
        mean, std = stat(key)
        if key == "success":
            mean, std = mean * 100.0, std * 100.0
        rows.append({"metric": key, "name": name, "unit": unit,
                     "mean": mean, "std": std})
        print(f"{name:<32}: {mean:8.3f} ± {std:6.3f} {unit}")

    # time_to_success 只在成功局有意义,单独统计
    tts = [m.get("time_to_success", -1) for m in all_metrics
           if m.get("time_to_success", -1) > 0]
    if tts:
        print(f"{'成功耗时 (Time to Success)':<32}: "
              f"{np.mean(tts):8.3f} ± {np.std(tts):6.3f} steps  "
              f"(仅 {len(tts)}/{len(all_metrics)} 成功局)")
    else:
        print(f"{'成功耗时 (Time to Success)':<32}: 无成功局")
    print("=" * 64)

    # 方差为 0 的自检:提醒随机化是否真的生效
    md_mean, md_std = stat("min_distance")
    if md_std < 1e-9:
        print("\n[警告] min_distance 方差为 0,说明所有 episode 完全相同。")
        print("       检查 feeding_task_env.py 里 INIT_JOINT_NOISE 是否 > 0。")

    # ---------- 导出 ----------
    if out_csv is None:
        out_csv = f"eval_{'shield' if use_shield else 'noshield'}_{os.path.basename(model_path)}"
    with open(out_csv + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "name", "unit", "mean", "std"])
        w.writeheader()
        w.writerows(rows)
    with open(out_csv + "_raw.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已导出: {out_csv}.csv")
    print(f"逐局原始数据: {out_csv}_raw.json  (做统计检验用)")


# ============================== 入口 ==============================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="喂食任务 PPO 训练/评估 (论文版 v3)")
    p.add_argument("--mode", choices=["train", "eval", "check"], required=True,
                   help="check = 只做阈值自洽检查,不启动 Unity")
    p.add_argument("--shield", type=int, choices=[0, 1], default=1)
    p.add_argument("--steps", type=int, default=150000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0,
                   help="训练种子;评估时用作 eval 种子(建议与训练种子不同)")
    p.add_argument("--out", type=str, default="ppo_model")
    p.add_argument("--model", type=str, default="ppo_model")
    p.add_argument("--graphics", action="store_true", help="开画面(会显著拖慢)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--ckpt-freq", type=int, default=10000)
    p.add_argument("--tb", type=str, default="./tb")
    p.add_argument("--out-csv", type=str, default=None)
    args = p.parse_args()

    use_shield = bool(args.shield)

    if args.mode == "check":
        check_thresholds(verbose=True)
    elif args.mode == "train":
        train(use_shield, args.steps, args.out, seed=args.seed,
              graphics=args.graphics, device=args.device,
              ckpt_freq=args.ckpt_freq, tb_dir=args.tb)
    else:
        evaluate(use_shield, args.model, args.episodes, seed=args.seed,
                 graphics=args.graphics, out_csv=args.out_csv)
