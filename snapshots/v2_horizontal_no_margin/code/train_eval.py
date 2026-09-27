"""Training and evaluation for the four-condition comparison.

Conditions
    C1            unconstrained
    C2 (4 lambdas) reward penalty, swept logarithmically over 0.005 - 5.0
    C3            reactive shield
    C4            predictive shield (proposed)

With 5 seeds per configuration this is 7 x 5 = 35 runs.

Usage
    # one run
    python train_eval.py train --condition C4 --seed 0 --steps 1000000

    # one C2 run at a given penalty weight
    python train_eval.py train --condition C2 --lam 0.05 --seed 0 --steps 1000000

    # evaluate a trained policy over 100 episodes
    python train_eval.py eval --condition C4 --seed 0 --episodes 100

    # print every command needed for the full 35-run grid
    python train_eval.py plan

    # aggregate all finished evaluations into the results table and trade-off data
    python train_eval.py report

Observations and rewards are normalised with VecNormalize (VECNORM_KWARGS). The
statistics are saved as models/<tag>_vecnormalize.pkl and loaded, frozen, by eval.
"""

import argparse
import json
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from feeding_task_env import FeedingTaskEnv


# ------------------------------------------------------------------ config

# Hyperparameters, identical across all conditions and all penalty weights.
PPO_KWARGS = dict(
    policy="MlpPolicy",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    policy_kwargs=dict(net_arch=[64, 64]),
)

# Observation and reward normalisation, identical across all runs. The raw 27-D
# observation mixes joint angles of tens of degrees with a tip-to-mouth vector of
# a few centimetres; on raw inputs 100k-step PPO stayed ~15 cm from the mouth
# (0 % success), normalised it reached 50 % (C1) and 75 % (C4) (2026-09-25).
VECNORM_KWARGS = dict(norm_obs=True, norm_reward=True, clip_obs=10.0,
                      gamma=PPO_KWARGS["gamma"])

TOTAL_TIMESTEPS = 1_000_000
SEEDS = [0, 1, 2, 3, 4]
# Four penalty weights, logarithmically spaced over three orders of magnitude.
C2_LAMBDAS = [0.005, 0.05, 0.5, 5.0]

MODEL_DIR = "models"
RESULT_DIR = "results"


def run_port(condition, seed, lam=None, base=5100):
    """A deterministic, unique port per configuration.

    35 runs may share a machine or a cluster node; each needs its own socket.
    """
    lam_idx = 0 if lam is None else (C2_LAMBDAS.index(lam) + 1
                                     if lam in C2_LAMBDAS else 9)
    cond_idx = {"C1": 0, "C2": 1, "C3": 2, "C4": 3}[condition.upper()]
    return base + cond_idx * 100 + lam_idx * 10 + seed


def run_tag(condition, seed, lam=None):
    c = condition.upper()
    if c == "C2":
        return f"C2_lam{lam:g}_seed{seed}"
    return f"{c}_seed{seed}"


# ------------------------------------------------------------------ train

def train(condition, seed, lam=None, steps=TOTAL_TIMESTEPS, graphics=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    tag = run_tag(condition, seed, lam)
    print(f"=== training {tag} for {steps} steps ===")

    port = run_port(condition, seed, lam)
    print(f"    port {port}")
    task = FeedingTaskEnv(condition=condition, lam=lam, graphics=graphics,
                          seed=seed, check=True, port=port)
    env = VecNormalize(DummyVecEnv([lambda: Monitor(task)]), **VECNORM_KWARGS)
    model = PPO(env=env, seed=seed, verbose=1, device="cpu", **PPO_KWARGS)
    model.learn(total_timesteps=steps)

    path = os.path.join(MODEL_DIR, tag)
    # Explicit suffix: SB3 only adds ".zip" when the name has no suffix, and C2
    # tags like "C2_lam0.05_seed0" already have one (".05_seed0").
    model.save(path + ".zip")
    env.save(path + "_vecnormalize.pkl")
    print(f"saved {path}.zip and {path}_vecnormalize.pkl")
    env.close()


# ------------------------------------------------------------------ evaluate

def evaluate(condition, seed, lam=None, episodes=100, graphics=False):
    os.makedirs(RESULT_DIR, exist_ok=True)
    tag = run_tag(condition, seed, lam)
    model_path = os.path.join(MODEL_DIR, tag)
    if not os.path.exists(model_path + ".zip"):
        raise FileNotFoundError(f"missing model: {model_path}.zip")
    stats_path = model_path + "_vecnormalize.pkl"
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"missing normalisation statistics: {stats_path}")

    env = FeedingTaskEnv(condition=condition, lam=lam, graphics=graphics,
                         seed=1000 + seed, check=False,
                         port=run_port(condition, seed, lam) + 1000)
    # Normalise observations exactly as in training, with the statistics frozen.
    norm = VecNormalize.load(stats_path, DummyVecEnv([lambda: env]))
    norm.training = False
    model = PPO.load(model_path + ".zip")

    rows = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        last = {}
        while not done:
            action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last = info
        m = last.get("episode_metrics", {})
        rows.append(m)
        if (ep + 1) % 10 == 0:
            print(f"  {ep+1}/{episodes} episodes")
    env.close()

    summary = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in rows[0]}
    summary["condition"] = condition.upper()
    summary["lam"] = lam
    summary["seed"] = seed
    summary["episodes"] = episodes

    out = os.path.join(RESULT_DIR, tag + ".json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "episodes": rows}, f, indent=2)

    print("\n" + "=" * 62)
    print(f"{tag}  ({episodes} episodes)")
    print("=" * 62)
    print(f"  success rate            : {summary['success']*100:6.1f} %")
    print(f"  held (ignoring safety)  : {summary['held']*100:6.1f} %")
    print(f"  reached (ignoring safety): {summary['reached']*100:6.1f} %")
    print(f"  peak contact force      : {summary['peak_force']:6.2f} N")
    print(f"  mean contact force      : {summary['mean_force']:6.2f} N")
    print(f"  comfort violation rate  : {summary['comfort_violation_rate']*100:6.2f} %")
    print(f"  danger violation rate   : {summary['danger_violation_rate']*100:6.2f} %")
    print(f"  force impulse           : {summary['force_impulse']:6.3f} N*s")
    print(f"  shield rate             : {summary['shield_rate']*100:6.2f} %")
    print(f"  mean correction         : {summary['mean_correction']:6.4f}")
    print(f"  min distance            : {summary['min_distance']:6.4f} m")
    print("=" * 62)
    print(f"saved {out}")
    return summary


# ------------------------------------------------------------------ plan

def plan(steps=TOTAL_TIMESTEPS, episodes=100):
    """Print every command for the full grid, in an order suitable for a job array."""
    lines = []
    for seed in SEEDS:
        for cond in ["C1", "C3", "C4"]:
            lines.append(f"python train_eval.py train --condition {cond} "
                         f"--seed {seed} --steps {steps}")
        for lam in C2_LAMBDAS:
            lines.append(f"python train_eval.py train --condition C2 --lam {lam} "
                         f"--seed {seed} --steps {steps}")
    print(f"# {len(lines)} training runs")
    print("\n".join(lines))
    print()
    ev = []
    for seed in SEEDS:
        for cond in ["C1", "C3", "C4"]:
            ev.append(f"python train_eval.py eval --condition {cond} "
                      f"--seed {seed} --episodes {episodes}")
        for lam in C2_LAMBDAS:
            ev.append(f"python train_eval.py eval --condition C2 --lam {lam} "
                      f"--seed {seed} --episodes {episodes}")
    print(f"# {len(ev)} evaluation runs")
    print("\n".join(ev))


# ------------------------------------------------------------------ report

def report():
    """Aggregate finished evaluations: per-condition table plus trade-off points."""
    if not os.path.isdir(RESULT_DIR):
        print("no results directory yet")
        return

    groups = {}
    for fn in sorted(os.listdir(RESULT_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(RESULT_DIR, fn)) as f:
            data = json.load(f)
        # results/ also holds diagnostics (calibrate_pose.json, ...) and the
        # trade-off file this function writes; only evaluations have a summary.
        if not (isinstance(data, dict) and "condition" in data.get("summary", {})):
            continue
        s = data["summary"]
        key = (s["condition"] if s["condition"] != "C2"
               else f"C2_lam{s['lam']:g}")
        groups.setdefault(key, []).append(s)

    if not groups:
        print("no evaluation results found")
        return

    def ms(rows, k):
        v = [r[k] for r in rows]
        return float(np.mean(v)), float(np.std(v))

    print("\nMean +- std across seeds\n")
    header = (f"{'condition':<16}{'seeds':>6}{'success %':>12}"
              f"{'peak F (N)':>14}{'danger %':>12}{'impulse':>12}{'shield %':>12}")
    print(header)
    print("-" * len(header))
    tradeoff = []
    for key in sorted(groups):
        rows = groups[key]
        sm, ss = ms(rows, "success")
        pm, ps = ms(rows, "peak_force")
        dm, ds = ms(rows, "danger_violation_rate")
        im, _ = ms(rows, "force_impulse")
        hm, _ = ms(rows, "shield_rate")
        print(f"{key:<16}{len(rows):>6}"
              f"{sm*100:>8.1f}+-{ss*100:<3.1f}"
              f"{pm:>9.2f}+-{ps:<4.2f}"
              f"{dm*100:>8.2f}+-{ds*100:<3.2f}"
              f"{im:>12.3f}{hm*100:>12.2f}")
        tradeoff.append({"condition": key, "success": sm, "success_std": ss,
                         "peak_force": pm, "peak_force_std": ps})

    out = os.path.join(RESULT_DIR, "tradeoff.json")
    with open(out, "w") as f:
        json.dump(tradeoff, f, indent=2)
    print(f"\ntrade-off data written to {out}")
    print("Plot: success rate on x, peak contact force on y. The C2 lambdas trace a "
          "curve; C1, C3 and C4 are single points. A shielded condition beats the "
          "penalty class only if it lies outside that curve.")


# ------------------------------------------------------------------ cli

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--condition", required=True, choices=["C1", "C2", "C3", "C4"])
    t.add_argument("--lam", type=float, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    t.add_argument("--graphics", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--condition", required=True, choices=["C1", "C2", "C3", "C4"])
    e.add_argument("--lam", type=float, default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--graphics", action="store_true")

    pl = sub.add_parser("plan")
    pl.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    pl.add_argument("--episodes", type=int, default=100)

    sub.add_parser("report")

    a = p.parse_args()
    if a.cmd == "train":
        train(a.condition, a.seed, a.lam, a.steps, a.graphics)
    elif a.cmd == "eval":
        evaluate(a.condition, a.seed, a.lam, a.episodes, a.graphics)
    elif a.cmd == "plan":
        plan(a.steps, a.episodes)
    elif a.cmd == "report":
        report()
