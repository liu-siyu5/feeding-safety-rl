"""可达性排查 v2 —— 勺尖到底能靠嘴多近?

为什么需要这个
--------------
策略停在 0.41 m,而上一版粗糙搜索(360 次)找到 0.123 m。
两个数字都不足以下结论:
  - 0.41 vs 0.123 说明策略远未发挥硬件能力 -> 优化问题
  - 0.123 vs 0.03 说明可能不可达,但上一版搜索质量太差,不能作数

上一版的两个缺陷,本版已修:
  1. sigma 坍缩: sigma = std + 3.0,前两轮 elite 一集中就塌到 3 度,
     之后 9 轮原地打转 -> 本版加 sigma 下限,并在停滞时重新扩张
  2. 采样先验偏心: mu = zeros(7),只搜了零位姿附近
     -> 本版阶段 1 在整个关节空间均匀采样

结论判据
--------
  best < 0.03 m         可达。问题在学习算法 -> 继续调 RL
  0.03 < best < 0.06 m  勉强可达,处于工作空间边缘 -> 考虑放宽阈值或移近基座
  best > 0.10 m         不可达。这是场景配置问题,改算法无效

运行
    conda activate rcareworld
    cd ~/research/my_feeding_project

    # 快速试跑(约 4 分钟,先确认能跑通)
    python check_reachability.py --uniform 300 --cem-iters 5 --pop 20

    # 正式(约 35-45 分钟)
    python check_reachability.py

    # 后台跑
    nohup python check_reachability.py > reach.log 2>&1 &
    tail -f reach.log
"""

import argparse
import json
import time

import numpy as np

from feeding_env import FeedingEnv
from kinematics import Gen3Kinematics

try:
    from feeding_task_env import (
        N_JOINTS, JOINT_LOW, JOINT_HIGH, SUCCESS_DIST,
        MOUTH_RADIUS, STIFFNESS_K, D_DANGER,
    )
except ImportError:      # 独立运行时的兜底默认值
    N_JOINTS, JOINT_LOW, JOINT_HIGH = 7, -180.0, 180.0
    SUCCESS_DIST, MOUTH_RADIUS, STIFFNESS_K, D_DANGER = 0.03, 0.04, 500.0, 0.02


class Probe:
    """封装一次 '设关节角 -> 读勺尖距离' 的评估。"""

    def __init__(self, graphics=False, settle=15):
        self.env = FeedingEnv(graphics=graphics)
        self.env.step()
        # Same time scale as feeding_task_env; verified not to change trajectories.
        self.env.SetTimeScale(10.0)
        self.env.step(5)
        self.robot = self.env.get_robot()
        self.mouth = self.env.get_mouth()
        self.env.step()
        self.settle = settle
        self.n_evals = 0
        self.kin = Gen3Kinematics()

        self.init_q = np.array(
            self.robot.data.get("joint_positions", [0.0] * N_JOINTS), dtype=float)
        self.mouth_pos = np.array(
            self.mouth.data.get("position", [0, 0, 0]), dtype=float)

    def __call__(self, q):
        q = np.clip(np.asarray(q, dtype=float), JOINT_LOW, JOINT_HIGH)
        self.robot.SetJointPosition(joint_positions=q.tolist())
        self.env.step(self.settle)
        # Spoon bowl from FK of the measured joints; grasp_point_position is the
        # gripper fingertip centre, ~13 cm short of the bowl.
        q_meas = np.array(self.robot.data.get("joint_positions", [0.0] * N_JOINTS),
                          dtype=float)
        sp = self.kin.fk_tip_unity(q_meas)
        self.n_evals += 1
        return float(np.linalg.norm(sp - self.mouth_pos)), sp

    def close(self):
        self.env.close()


def phase_uniform(probe, n, rng, report_every=100):
    """阶段 1: 全关节空间均匀采样,给出可达性的全局图像。"""
    print(f"\n[阶段 1] 全空间均匀采样 {n} 次")
    print(f"  采样范围: [{JOINT_LOW}, {JOINT_HIGH}] deg,各关节独立均匀")
    dists, best_d, best_q = [], float("inf"), None
    t0 = time.time()
    for i in range(n):
        q = rng.uniform(JOINT_LOW, JOINT_HIGH, size=N_JOINTS)
        d, _ = probe(q)
        dists.append(d)
        if d < best_d:
            best_d, best_q = d, q.copy()
        if (i + 1) % report_every == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (n - i - 1)
            print(f"  {i+1:5d}/{n}  最优 = {best_d:.4f} m   "
                  f"已用 {el/60:.1f} min, 预计还需 {eta/60:.1f} min")
    return np.array(dists), best_d, best_q


def phase_cem(probe, seeds, rng, iters, pop, elite_frac,
              sigma0, sigma_floor, stall_patience):
    """阶段 2: 从多个起点做 CEM,带 sigma 下限与停滞重扩张。"""
    print(f"\n[阶段 2] CEM 局部精化 —— {len(seeds)} 个起点 x {iters} 轮 x {pop} 样本")
    print(f"  sigma: 初始 {sigma0} deg, 下限 {sigma_floor} deg "
          f"(上一版无下限,导致第 2 轮即坍缩)")

    n_elite = max(2, int(pop * elite_frac))
    g_best_d, g_best_q = float("inf"), None

    for si, q0 in enumerate(seeds):
        mu = np.asarray(q0, dtype=float).copy()
        sigma = np.full(N_JOINTS, float(sigma0))
        best_d, best_q = float("inf"), None
        stall = 0
        print(f"\n  -- 起点 {si+1}/{len(seeds)} (初始距离 {probe(mu)[0]:.4f} m)")

        for it in range(iters):
            pop_q = np.clip(rng.normal(mu, sigma, size=(pop, N_JOINTS)),
                            JOINT_LOW, JOINT_HIGH)
            ds = np.array([probe(q)[0] for q in pop_q])
            order = np.argsort(ds)
            elite = pop_q[order[:n_elite]]

            improved = ds[order[0]] < best_d - 1e-5
            if improved:
                best_d, best_q = float(ds[order[0]]), pop_q[order[0]].copy()
                stall = 0
            else:
                stall += 1

            mu = elite.mean(axis=0)
            # 关键修复: sigma 下限,防止坍缩
            sigma = np.maximum(elite.std(axis=0), sigma_floor)
            # 停滞时重新扩张,跳出局部极小
            if stall >= stall_patience:
                sigma = np.maximum(sigma * 4.0, sigma0 * 0.5)
                stall = 0
                tag = "  <- 停滞,扩张 sigma"
            else:
                tag = ""
            print(f"     iter {it:2d}  best={best_d:.4f} m  "
                  f"sigma_mean={sigma.mean():5.1f} deg{tag}")

        if best_d < g_best_d:
            g_best_d, g_best_q = best_d, best_q

    return g_best_d, g_best_q


def main(a):
    rng = np.random.default_rng(a.seed)
    probe = Probe(graphics=a.graphics, settle=a.settle)

    print("=" * 70)
    print("可达性排查 v2")
    print("=" * 70)
    print(f"关节数        : {N_JOINTS}")
    print(f"关节限位      : [{JOINT_LOW}, {JOINT_HIGH}] deg")
    print(f"嘴部位置      : {np.round(probe.mouth_pos, 4)}")
    print(f"成功阈值      : {SUCCESS_DIST} m")
    d0, sp0 = probe(probe.init_q)
    print(f"初始姿态距离  : {d0:.4f} m  (勺尖 {np.round(sp0,4)})")
    total = a.uniform + len(range(a.restarts)) * a.cem_iters * a.pop
    print(f"计划评估次数  : {total}  (阶段1 {a.uniform} + 阶段2 "
          f"{a.restarts}x{a.cem_iters}x{a.pop})")
    print("=" * 70)

    t_start = time.time()

    # ---------- 阶段 1 ----------
    dists, u_best_d, u_best_q = phase_uniform(probe, a.uniform, rng)
    print(f"\n  均匀采样结果:")
    print(f"    最小 {dists.min():.4f} m | 中位 {np.median(dists):.4f} m | "
          f"最大 {dists.max():.4f} m")
    for th in (0.30, 0.20, 0.10, 0.05, SUCCESS_DIST):
        c = int((dists < th).sum())
        print(f"    < {th:.2f} m : {c:5d} 次 ({c/len(dists)*100:5.2f}%)")

    # 取最好的若干个作为 CEM 起点
    k = min(a.restarts, len(dists))
    seeds_idx = np.argsort(dists)[:k]
    seeds = []
    rng2 = np.random.default_rng(a.seed + 1)
    for i in seeds_idx:
        # 重采样一个与该距离相近的点作为起点(已知 dists[i],但未存 q)
        pass
    # 为节省内存,阶段1未保存全部 q;用最优点 + 随机扰动构造起点
    seeds = [u_best_q]
    for _ in range(k - 1):
        seeds.append(np.clip(u_best_q + rng2.normal(0, a.sigma0, N_JOINTS),
                             JOINT_LOW, JOINT_HIGH))

    # ---------- 阶段 2 ----------
    c_best_d, c_best_q = phase_cem(
        probe, seeds, rng, a.cem_iters, a.pop, a.elite,
        a.sigma0, a.sigma_floor, a.stall)

    best_d = min(u_best_d, c_best_d)
    best_q = c_best_q if c_best_d <= u_best_d else u_best_q
    f_at_best = STIFFNESS_K * max(0.0, MOUTH_RADIUS - best_d)
    elapsed = time.time() - t_start

    # ---------- 结论 ----------
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"评估总次数       : {probe.n_evals}")
    print(f"耗时             : {elapsed/60:.1f} min")
    print(f"阶段1 (均匀) 最优: {u_best_d:.4f} m")
    print(f"阶段2 (CEM)  最优: {c_best_d:.4f} m")
    print(f"可达最小距离     : {best_d:.4f} m   <<< 关键")
    print(f"该点接触力       : {f_at_best:.2f} N")
    print(f"最优关节角       : {np.round(best_q, 1).tolist()}")
    print(f"成功阈值         : {SUCCESS_DIST} m")
    print(f"策略当前停滞于   : ~0.41 m  (参考)")
    print()

    if best_d < SUCCESS_DIST:
        verdict = "REACHABLE"
        print(">>> 可达。目标在工作空间内。")
        print("    结论: 这是优化问题,不是几何问题。")
        print(f"    策略停在 0.41 m,而硬件能到 {best_d:.3f} m —— "
              f"差 {0.41/max(best_d,1e-6):.1f} 倍。")
        print("    下一步: 检查 v3 观测归一化后的 min_distance 与 std,")
        print("            并核对动作预算 (200 步 x 5 deg) 是否够用。")
    elif best_d < SUCCESS_DIST * 2:
        verdict = "MARGINAL"
        print(">>> 勉强可达。目标处于工作空间边缘。")
        print("    策略需要极精确的控制才能成功,成功率会很低且方差大。")
        print(f"    建议: 将 SUCCESS_DIST 放宽至 {best_d*1.3:.3f} m 左右,")
        print("          或把机械臂基座移近嘴部。")
    else:
        verdict = "UNREACHABLE"
        print(">>> 不可达。这是场景配置问题,不是算法问题。")
        print(f"    {probe.n_evals} 次评估的最优结果仍差 "
              f"{best_d - SUCCESS_DIST:.3f} m。")
        print("    排查顺序:")
        print("      1. 机械臂基座与嘴的距离是否超出臂展")
        print("      2. 关节限位是否设得过严")
        print("      3. 是否有碰撞体阻挡(桌面、餐盘、自身连杆)")
        print("      4. 嘴部坐标读数是否正确")
        print("    在修正场景之前,任何算法改进都无效。")
    print("=" * 70)

    out = {
        "verdict": verdict,
        "best_distance_m": best_d,
        "best_joints_deg": np.round(best_q, 4).tolist(),
        "force_at_best_N": f_at_best,
        "success_threshold_m": SUCCESS_DIST,
        "uniform_best_m": float(u_best_d),
        "cem_best_m": float(c_best_d),
        "uniform_min_m": float(dists.min()),
        "uniform_median_m": float(np.median(dists)),
        "n_evaluations": probe.n_evals,
        "elapsed_min": elapsed / 60.0,
        "init_pose_distance_m": d0,
        "mouth_position": probe.mouth_pos.tolist(),
        "seed": a.seed,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {a.out}")

    probe.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="机械臂可达性排查 v2")
    p.add_argument("--uniform", type=int, default=2500, help="阶段1 均匀采样次数")
    p.add_argument("--restarts", type=int, default=4, help="阶段2 CEM 起点个数")
    p.add_argument("--cem-iters", type=int, default=15, help="每个起点的迭代轮数")
    p.add_argument("--pop", type=int, default=40, help="每轮样本数")
    p.add_argument("--elite", type=float, default=0.2, help="精英比例")
    p.add_argument("--sigma0", type=float, default=45.0, help="初始 sigma (deg)")
    p.add_argument("--sigma-floor", type=float, default=2.0,
                   help="sigma 下限 (deg) —— 防坍缩的关键")
    p.add_argument("--stall", type=int, default=3, help="停滞几轮后扩张 sigma")
    p.add_argument("--settle", type=int, default=15, help="每次评估的物理 tick 数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--graphics", action="store_true")
    p.add_argument("--out", type=str, default="reachability_result.json")
    args = p.parse_args()
    main(args)
