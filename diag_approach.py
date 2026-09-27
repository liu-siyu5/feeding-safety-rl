"""Diagnose the feasible approach direction to the mouth.

Starting from q_reach (grasp point 0.0137 m from the mouth), retreat along the
joint-space interpolation path toward the zero pose,
    q(f) = f * q_reach,   f = 1.00, 0.95, 0.90, 0.85, 0.80, 0.70,
and record the spoon tip in the Unity world frame at each checkpoint. From
these points we infer the direction from which the spoon reaches the mouth.

Spoon tip = the bowl (kinematics.fk_tip_unity of the measured joints). Runs
before 2026-09-25 used the gripper grasp point, ~13 cm short of the bowl; at
q_reach the bowl is 12.5 cm from the mouth, inside the head.

Why this way: the face orientation cannot be derived from the human model
(SMPL-X bones are not readable, and using the body-local Z axis as an IK
target already failed). The path itself is the only trustworthy evidence.

Three passes are reported with the same analysis:
  ideal   : URDF FK of the commanded joints (no simulator, no contact)
  forward : 0 -> 1.0 in the simulator, checkpoints logged on the way in
  retreat : 1.0 -> 0.70 in the simulator (the originally planned measurement)
If forward and retreat disagree, the arm is being held by contact on one of
them (large tracking error flags it) and that pass should not be trusted.

Procedure
  1. Position control (stiffness 10000, damping 100).
  2. Move in small increments of f so the arm never jumps.
  3. Settle at each checkpoint; log tip, distance to mouth, tracking error.

Run
    conda activate rcareworld
    cd ~/research/my_feeding_project
    python diag_approach.py              # headless
    python diag_approach.py --graphics   # watch it
"""

import argparse
import json

import numpy as np

from feeding_env import FeedingEnv
from kinematics import Gen3Kinematics


N_JOINTS = 7
Q_REACH = np.array([38.3, -33.4, -99.0, -45.0, -1.4, -40.5, 22.2])
CHECKPOINTS = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70]

POS_STIFFNESS = 10000.0
POS_DAMPING = 100.0
TIME_SCALE = 10.0

# Tracking error above this (deg) means the arm is probably held by contact.
TRACK_WARN_DEG = 3.0

# Revolute joints with finite limits (deg). Joints 0/2/4/6 are continuous.
FINITE_LIMITS = {1: 138.1, 3: 152.4, 5: 127.8}


# ============================== geometry helpers ==============================

def wrap_deg(a):
    """Wrap angles to [-180, 180) so continuous joints compare correctly."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))))


def describe_direction(d):
    """Elevation / azimuth of a Unity-frame direction (Y is up).

    elevation: angle above the horizontal plane (+ = moving upward)
    azimuth  : heading in the XZ plane, atan2(x, z), 0 deg = +Z, 90 deg = +X
    """
    d = unit(d)
    elev = float(np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(d[0], d[2])))
    return elev, azim


# ============================== simulator rig ==============================

class Rig:
    """Thin wrapper: set joint target under position control, read state."""

    def __init__(self, graphics, kin, port=None):
        self.kin = kin
        kw = {"graphics": graphics}
        if port is not None:
            kw["port"] = int(port)
        self.env = FeedingEnv(**kw)
        self.env.step()
        self.env.SetTimeScale(TIME_SCALE)
        self.env.step(5)
        self.robot = self.env.get_robot()
        self.mouth = self.env.get_mouth()
        self.env.step()
        self.robot.SetJointStiffness([POS_STIFFNESS] * N_JOINTS)
        self.robot.SetJointDamping([POS_DAMPING] * N_JOINTS)
        self.env.step(5)

    def q(self):
        return np.array(self.robot.data.get("joint_positions", [0.0] * N_JOINTS), float)

    def tip(self):
        """Spoon bowl, from FK of the measured joints (see kinematics.SPOON_TIP_IN_GRASP)."""
        return self.kin.fk_tip_unity(self.q())

    def grasp(self):
        """Gripper grasp point as reported by the simulator."""
        return np.array(self.robot.data.get("grasp_point_position", [0, 0, 0]), float)

    def mouth_pos(self):
        return np.array(self.mouth.data.get("position", [0, 0, 0]), float)

    def command(self, q_cmd, ticks):
        self.robot.SetJointPosition(joint_positions=np.asarray(q_cmd, float).tolist())
        self.env.step(ticks)

    def settle(self, q_cmd, max_ticks, tol_deg=0.05, chunk=5):
        """Hold q_cmd until joints stop moving (or max_ticks). Returns ticks used."""
        self.command(q_cmd, chunk)
        used, prev = chunk, self.q()
        while used < max_ticks:
            self.command(q_cmd, chunk)
            used += chunk
            cur = self.q()
            if np.max(np.abs(wrap_deg(cur - prev))) < tol_deg:
                break
            prev = cur
        return used

    def close(self):
        self.env.close()


def record(rig, kin, f, q_cmd, settle_ticks):
    q_meas = rig.q()
    track_err = np.abs(wrap_deg(q_meas - q_cmd))
    tip = rig.tip()
    mouth = rig.mouth_pos()
    # Frame check: FK grasp point vs the simulator's own grasp point readout.
    fk_err = float(np.linalg.norm(kin.fk_grasp_unity(q_meas) - rig.grasp()))
    return {
        "f": f,
        "q_cmd": q_cmd.tolist(),
        "q_meas": q_meas.tolist(),
        "track_err_deg": track_err.tolist(),
        "track_err_max_deg": float(track_err.max()),
        "tip": tip.tolist(),
        "mouth": mouth.tolist(),
        "dist_m": float(np.linalg.norm(tip - mouth)),
        "fk_err_mm": fk_err * 1000.0,
        "settle_ticks": settle_ticks,
    }


def run_pass(rig, kin, f_start, checkpoints, a):
    """Walk f from f_start through the checkpoints in the given order.

    Returns (checkpoint records, fine samples [(f, tip)] between checkpoints).
    """
    rows, fine = [], []
    f_cur = f_start
    for f_target in checkpoints:
        n_sub = max(1, int(round(abs(f_cur - f_target) / a.df)))
        for k in range(1, n_sub + 1):
            f = f_cur + (f_target - f_cur) * k / n_sub
            rig.command(f * Q_REACH, a.substep_ticks)
            if rows:  # only log fine samples once inside the checkpoint range
                fine.append((f, rig.tip().tolist()))
        q_cmd = f_target * Q_REACH
        used = rig.settle(q_cmd, a.checkpoint_ticks)
        rows.append(record(rig, kin, f_target, q_cmd, used))
        fine.append((f_target, rows[-1]["tip"]))
        f_cur = f_target
    return rows, fine


def ideal_pass(kin, mouth, df):
    """Same path through URDF FK of the commanded joints (no contact, no lag)."""
    rows = []
    for f in CHECKPOINTS:
        q_cmd = f * Q_REACH
        tip = kin.fk_tip_unity(q_cmd)
        rows.append({"f": f, "q_cmd": q_cmd.tolist(), "q_meas": q_cmd.tolist(),
                     "track_err_deg": [0.0] * N_JOINTS, "track_err_max_deg": 0.0,
                     "tip": tip.tolist(), "mouth": mouth.tolist(),
                     "dist_m": float(np.linalg.norm(tip - mouth)),
                     "fk_err_mm": 0.0, "settle_ticks": 0})
    fs = np.arange(min(CHECKPOINTS), max(CHECKPOINTS) + 1e-9, df)
    fine = [(float(f), kin.fk_tip_unity(f * Q_REACH).tolist()) for f in fs]
    return rows, fine


# ============================== analysis / report ==============================

def analyse(title, rows, fine, base_position):
    """Print checkpoints, approach direction and segments; return a summary dict."""
    rows = sorted(rows, key=lambda r: -r["f"])          # near (f=1) -> far
    mouth = np.array(rows[0]["mouth"])
    tips = np.array([r["tip"] for r in rows])

    print("\n" + "#" * 100)
    print(f"# {title}")
    print("#" * 100)

    # ---------- 1. checkpoints ----------
    print("1. Checkpoints (Unity world frame, Y up)")
    print(f"   mouth = {np.round(mouth, 4).tolist()}")
    print(f"{'f':>5} | {'tip x':>8} {'tip y':>8} {'tip z':>8} | {'dist m':>7} | "
          f"{'trk max':>7} | {'FK err mm':>9} | per-joint tracking error (deg)")
    print("-" * 100)
    n_flag = 0
    for r in rows:
        t = r["tip"]
        errs = " ".join(f"{e:4.1f}" for e in r["track_err_deg"])
        flag = " !" if r["track_err_max_deg"] > TRACK_WARN_DEG else ""
        n_flag += bool(flag)
        print(f"{r['f']:5.2f} | {t[0]:8.4f} {t[1]:8.4f} {t[2]:8.4f} | {r['dist_m']:7.4f} | "
              f"{r['track_err_max_deg']:7.2f} | {r['fk_err_mm']:9.2f} | {errs}{flag}")
    if n_flag:
        print(f"   ! {n_flag} checkpoint(s) exceed {TRACK_WARN_DEG} deg tracking error: "
              "arm likely held by contact; directions below are unreliable.")

    # ---------- 2. approach direction ----------
    i_far = int(np.argmax([r["dist_m"] for r in rows]))
    p_far, p_near = tips[i_far], tips[0]
    approach = unit(mouth - p_far)          # farthest point -> mouth
    travel = unit(p_near - p_far)           # actual tip travel, far -> near
    residual = mouth - p_near
    base_to_mouth = unit(mouth - base_position)

    print("\n2. Approach direction")
    for name, v in [("far point -> mouth (approach)", approach),
                    ("far tip -> near tip (travel)", travel),
                    ("reference: base -> mouth", base_to_mouth)]:
        el, az = describe_direction(v)
        print(f"   {name:30s}: {str(np.round(v, 4).tolist()):32s} "
              f"elev {el:+6.1f} deg, azim {az:+7.1f} deg")
    print(f"   farthest checkpoint           : f = {rows[i_far]['f']:.2f}, "
          f"dist {rows[i_far]['dist_m']:.4f} m")
    print(f"   angle(approach, travel)       : {angle_deg(approach, travel):6.1f} deg")
    print(f"   angle(approach, base->mouth)  : {angle_deg(approach, base_to_mouth):6.1f} deg")
    print(f"   residual mouth - tip @ f=1.0  : {np.round(residual, 4).tolist()} "
          f"(|r| = {np.linalg.norm(residual):.4f} m)")
    print("   ('approach' points toward the mouth; the spoon comes from the -approach side)")

    # ---------- 3. segments ----------
    fine_f = np.array([p[0] for p in fine])
    fine_p = np.array([p[1] for p in fine])
    print("\n3. Segments (far -> near)")
    print(f"{'segment':>13} | {'chord m':>8} | {'arc m':>8} | {'arc/chord':>9} | "
          f"{'direction (unit)':>26} | {'elev':>6} {'azim':>7} | {'turn':>6} | {'vs appr':>7}")
    print("-" * 100)
    segs, prev_dir, total_arc = [], None, 0.0
    for i in range(len(rows) - 1, 0, -1):
        f_a, f_b = rows[i]["f"], rows[i - 1]["f"]
        d = tips[i - 1] - tips[i]
        chord = float(np.linalg.norm(d))
        # arc length from the fine samples between the two checkpoints, ordered by f
        mask = (fine_f >= f_a - 1e-9) & (fine_f <= f_b + 1e-9)
        pts = fine_p[mask][np.argsort(fine_f[mask], kind="stable")]
        arc = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else chord
        total_arc += arc
        u = unit(d)
        el, az = describe_direction(u)
        turn = angle_deg(prev_dir, u) if prev_dir is not None else float("nan")
        vs_app = angle_deg(u, approach)
        print(f"{f_a:4.2f} -> {f_b:4.2f} | {chord:8.4f} | {arc:8.4f} | "
              f"{arc / max(chord, 1e-9):9.3f} | {str(np.round(u, 3).tolist()):>26} | "
              f"{el:+6.1f} {az:+7.1f} | {turn:6.1f} | {vs_app:7.1f}")
        segs.append({"from_f": f_a, "to_f": f_b, "chord_m": chord, "arc_m": arc,
                     "dir": u.tolist(), "elevation_deg": el, "azimuth_deg": az,
                     "turn_from_prev_deg": turn, "angle_to_approach_deg": vs_app})
        prev_dir = u
    total_chord = float(np.linalg.norm(p_near - p_far))
    print("-" * 100)
    print(f"   total chord {total_chord:.4f} m, total arc {total_arc:.4f} m, "
          f"arc/chord = {total_arc / max(total_chord, 1e-9):.3f} (1.00 = straight)")

    return {"checkpoints": rows, "approach_dir": approach.tolist(),
            "travel_dir": travel.tolist(), "residual_at_f1": residual.tolist(),
            "segments": segs, "total_chord_m": total_chord, "total_arc_m": total_arc,
            "n_tracking_flags": n_flag,
            "fine_samples": [{"f": f, "tip": p} for f, p in fine]}


def main(a):
    for j, lim in FINITE_LIMITS.items():
        assert abs(Q_REACH[j]) <= lim, f"joint {j} target outside ±{lim} deg"

    kin = Gen3Kinematics(a.urdf)

    rig = Rig(a.graphics, kin, a.port)
    try:
        rig.settle(np.zeros(N_JOINTS), a.checkpoint_ticks)
        # forward: 0 -> 0.70 -> ... -> 1.00 (the verified-clear direction)
        fwd_rows, fwd_fine = run_pass(rig, kin, 0.0, CHECKPOINTS[::-1], a)
        # retreat: 1.00 -> ... -> 0.70
        ret_rows, ret_fine = run_pass(rig, kin, 1.0, CHECKPOINTS, a)
    finally:
        rig.close()

    mouth = np.array(fwd_rows[0]["mouth"])
    out = {
        "q_reach_deg": Q_REACH.tolist(),
        "mouth": mouth.tolist(),
        "ideal": analyse("IDEAL (URDF FK of commanded joints)",
                         *ideal_pass(kin, mouth, a.df), kin.base_position),
        "forward": analyse("FORWARD (simulator, 0 -> 1.0)",
                           fwd_rows, fwd_fine, kin.base_position),
        "retreat": analyse("RETREAT (simulator, 1.0 -> 0.70)",
                           ret_rows, ret_fine, kin.base_position),
    }

    # ---------- cross-pass comparison ----------
    print("\n" + "#" * 100)
    print("# Comparison")
    print("#" * 100)
    names = ["ideal", "forward", "retreat"]
    print(f"   {'':8s} " + " ".join(f"{n:>10s}" for n in names))
    for i, f in enumerate(CHECKPOINTS):
        ds = [out[n]["checkpoints"][i]["dist_m"] for n in names]
        print(f"   f={f:4.2f}  " + " ".join(f"{d:10.4f}" for d in ds) + "   (dist to mouth, m)")
    for n1, n2 in [("ideal", "forward"), ("ideal", "retreat"), ("forward", "retreat")]:
        print(f"   angle(approach {n1}, {n2}) = "
              f"{angle_deg(np.array(out[n1]['approach_dir']), np.array(out[n2]['approach_dir'])):.1f} deg")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Diagnose the feasible spoon approach direction")
    p.add_argument("--df", type=float, default=0.01,
                   help="interpolation increment in f between commands")
    p.add_argument("--substep-ticks", type=int, default=3,
                   help="physics ticks per increment (3 = one 0.06 s control step)")
    p.add_argument("--checkpoint-ticks", type=int, default=100,
                   help="max physics ticks to settle at each checkpoint")
    p.add_argument("--urdf", type=str, default="gen3_kinematics_only.urdf")
    p.add_argument("--graphics", action="store_true")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--out", type=str, default="results/diag_approach.json")
    main(p.parse_args())
