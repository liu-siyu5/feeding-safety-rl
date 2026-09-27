"""Short PPO run to check that the constrained task is learnable.

Same PPO hyperparameters and environment as train_eval.py, but it saves to
models/short/ and results/short/ so it never overwrites the paper runs, and it
logs every finished episode's metrics while training.

Observations are normalised (VecNormalize) by default: the raw 27-D vector
mixes joint angles of tens of degrees with a tip-to-mouth vector of a few
centimetres, which the policy barely sees at raw scale. --no-normalize disables
it (train_eval.py always normalises, with the same VECNORM_KWARGS). The
statistics are saved next to the model.

Usage
    python train_short.py --condition C1 --steps 100000
    python train_short.py --condition C4 --steps 100000
"""

import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from feeding_task_env import FeedingTaskEnv
from train_eval import PPO_KWARGS, VECNORM_KWARGS, run_port


def summarise(rows):
    """Mean of the headline metrics over a list of episode_metrics dicts."""
    def mean(k):
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.mean(vals)) if vals else float("nan")
    keys = ("success", "held", "near_mouth", "min_distance", "approach_at_min_dist",
            "tilt_at_min_dist", "pitch_at_min_dist", "centre_offset_at_min_dist",
            "peak_force", "danger_violation_rate", "ep_length")
    return {k: mean(k) for k in keys}


def format_summary(s):
    return (f"success {s['success'] * 100:5.1f}% | held {s['held'] * 100:5.1f}% | "
            f"within 3 cm {s['near_mouth'] * 100:5.1f}% | "
            f"min dist {s['min_distance']:.3f} m | approach {s['approach_at_min_dist']:5.1f} deg | "
            f"tilt {s['tilt_at_min_dist']:5.1f} deg | pitch {s['pitch_at_min_dist']:+5.1f} deg | off centre "
            f"{s['centre_offset_at_min_dist'] * 100:4.1f} cm | peak F {s['peak_force']:4.1f} N | "
            f"length {s['ep_length']:5.1f}")


class EpisodeLog(BaseCallback):
    """Append each finished episode to a JSON-lines file; print rolling means."""

    def __init__(self, path, every=50):
        super().__init__()
        self.path, self.every, self.rows = path, every, []

    def _on_step(self):
        for info in self.locals["infos"]:
            m = info.get("episode_metrics")
            if m is None:
                continue
            m = dict(m, timesteps=self.num_timesteps)
            self.rows.append(m)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(m) + "\n")
            if len(self.rows) % self.every == 0:
                print(f"[{self.num_timesteps:7d} steps, {len(self.rows):5d} episodes] last {self.every}: "
                      + format_summary(summarise(self.rows[-self.every:])), flush=True)
        return True


def evaluate(model, env, episodes, venv=None):
    """Deterministic rollouts on the raw env; observations normalised like in training."""
    rows = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            policy_obs = venv.normalize_obs(obs) if venv is not None else obs
            action, _ = model.predict(policy_obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        rows.append(info["episode_metrics"])
    return rows


def main(a):
    os.makedirs("models/short", exist_ok=True)
    os.makedirs("results/short", exist_ok=True)
    lam_tag = f"_lam{a.lam:g}" if a.condition == "C2" else ""
    rate_tag = f"_rate{a.barrier_rate:g}" if a.barrier_rate is not None else ""
    tag = f"{a.condition}{lam_tag}{rate_tag}_seed{a.seed}_{a.steps // 1000}k"
    log_path = f"results/short/{tag}_episodes.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path)

    # Offset keeps these ports clear of train_eval.py's train (+0) and eval (+1000) ports.
    port = a.port if a.port is not None else run_port(a.condition, a.seed, a.lam) + 2000
    task = FeedingTaskEnv(condition=a.condition, lam=a.lam, seed=a.seed, check=False, port=port)
    if a.barrier_rate is not None:
        task.mechanism.barrier_rate = a.barrier_rate   # C4 only (checked in the parser)
    venv = None
    if a.normalize:
        venv = VecNormalize(DummyVecEnv([lambda: Monitor(task)]), **VECNORM_KWARGS)
        train_env = venv
    else:
        train_env = Monitor(task)
    model = PPO(env=train_env, seed=a.seed, verbose=0, device="cpu", **PPO_KWARGS)

    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=EpisodeLog(log_path))
    # Explicit suffix: SB3 only adds ".zip" when the name has no suffix, and a
    # tag like "rate0.5" already has one (".5_seed0_100k").
    model.save(f"models/short/{tag}.zip")
    if venv is not None:
        venv.save(f"models/short/{tag}_vecnormalize.pkl")
        venv.training = False      # freeze the statistics for evaluation
    print(f"trained {a.steps} steps in {(time.time() - t0) / 60:.1f} min -> models/short/{tag}.zip")

    rows = evaluate(model, task, a.eval_episodes, venv)
    summary = summarise(rows)
    print(f"evaluation, {a.eval_episodes} deterministic episodes: " + format_summary(summary))
    with open(f"results/short/{tag}_eval.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "episodes": rows}, fh, indent=2)
    task.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Short PPO run on the constrained feeding task")
    p.add_argument("--condition", required=True, choices=["C1", "C2", "C3", "C4"])
    p.add_argument("--lam", type=float, default=None, help="penalty weight, required for C2")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                   help="normalise observations and rewards with VecNormalize")
    p.add_argument("--barrier-rate", type=float, default=None,
                   help="C4 only: fraction of the remaining margin the tip may close per step "
                        "(default: C4PredictiveShield's); added to the run tag")
    p.add_argument("--port", type=int, default=None,
                   help="simulator port, for running variants of one configuration side by side")
    args = p.parse_args()
    if args.barrier_rate is not None and args.condition != "C4":
        p.error("--barrier-rate applies to C4 only")
    main(args)
