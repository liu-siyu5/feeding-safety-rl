"""Visual calibration of the two references needed for the feeding constraints.

CLAUDE.md, open problems 1-2: the approach direction and the spoon attitude
need constraints, but two reference quantities are unknown and cannot be
derived from the model:

  (a) which way the face points in the Unity world frame (skeleton unreadable,
      body-local Z failed as an IK target, the kinematic spoon teleports
      through geometry under SetPosition);
  (b) which spoon local axis is the bowl's concave normal.

This script poses the arm and holds still so a human can read both off the
screen. Every probe pose is commanded with SetJointPosition and verified by
reading grasp_point_position back.

Stops (numbering is fixed so screen observations can be matched to the log):
  #1      spoon in free space, held 8 s; prints the spoon's local axes
  #2-#7   grasp point 12 cm from the mouth along +X, -X, +Y, -Y, +Z, -Z,
          held 5 s each; skipped, with the reason printed, if IK has no
          solution or the arm does not actually get there

Each probe tries two IK strategies and shows the first that is verified:
  posture   position-only; redundancy pulled toward q_reach; hub -> target
  pointing  tool axis aimed at the mouth; hub -> 10 cm further out -> target
Position-only solutions all inherit q_reach's arm configuration, which comes
at the mouth from behind the ear, so on their own they fail for reasons that
say nothing about the face. Aiming the tool axis gives a radial approach.

Measured while writing this script (headless):
  - local_to_world_matrix arrives as a 4x4 row-major array: its columns are
    the local axes scaled by (0.2, 0.3, 0.15), its translation equals
    data['position'], and the normalised columns match data['quaternion'].
  - The grasp frame's z axis (flange -> grasp point) equals the spoon's local
    -z axis. The spoon's origin is 8 cm beyond grasp_point_position along it.
    Probes deliberately position the grasp point, not the spoon tip.

Result (2026-09-25, user watching stops #1, #3, #6): face = world +Z; concave
normal = spoon local +y; the bowl is ~13 cm beyond the grasp point, now
kinematics.SPOON_TIP_IN_GRASP.
  - On the path q = f * q_reach the arm touches the human (collision pairs
    with ids 2322 / 2333) at f = 0.70 and at most poses beyond; f <= 0.60 is
    contact-free with 0.04 deg tracking error. Every attempt starts from
    f = 0.60, reached with SetJointPositionDirectly, so a stuck attempt
    cannot contaminate the next.
  - The mouth is near the edge of the workspace. With the tool aimed at the
    mouth the wrist-pitch joint sits 0.3175 m behind the grasp point, so for
    +X / -X / +Y / -Z it would have to be 0.84-1.21 m from the shoulder, past
    the arm's 0.735 m shoulder-to-wrist reach; only +Z and -Y (0.68 / 0.63 m)
    have pointing IK solutions.
  - Near the mouth the joints settle 1-2 deg off the command even without
    contact (at the -X probe the error did not shrink when the command was
    offset to compensate), so a verified grasp point is typically 1-3 cm
    from its IK target.

Run
    conda activate rcareworld
    cd ~/research/my_feeding_project
    python calibrate_pose.py                                 # graphics
    python calibrate_pose.py --hold-scale 2                  # hold twice as long
    python calibrate_pose.py --no-graphics --hold-scale 0    # headless dry run
"""

import argparse
import json
import sys
import time

import numpy as np

from feeding_env import FeedingEnv
from kinematics import Gen3Kinematics, R_URDF_TO_UNITY


N_JOINTS = 7
SPOON_ID = 114514
# SMPL-X articulation objects that show up in collision pairs (2333 is
# FeedingEnv._person_id; 2322 is a second articulation named 'SMPLX-male').
HUMAN_IDS = (2322, 2333)
Q_REACH = np.array([38.3, -33.4, -99.0, -45.0, -1.4, -40.5, 22.2])   # deg

POS_STIFFNESS = 10000.0
POS_DAMPING = 100.0

# Finite joint limits (deg); joints 0/2/4/6 are continuous.
JOINT_LIMIT_DEG = np.array([np.inf, 138.1, np.inf, 152.4, np.inf, 127.8, np.inf])

PROBE_RADIUS_M = 0.12
PRE_APPROACH_M = 0.10        # pointing strategy: extra distance before the radial approach
# Order fixes the stop numbers (#2 = +X ... #7 = -Z). Also used to label
# arbitrary vectors by their nearest signed world axis.
DIRECTIONS = [
    ("+X", np.array([1.0, 0.0, 0.0])), ("-X", np.array([-1.0, 0.0, 0.0])),
    ("+Y", np.array([0.0, 1.0, 0.0])), ("-Y", np.array([0.0, -1.0, 0.0])),
    ("+Z", np.array([0.0, 0.0, 1.0])), ("-Z", np.array([0.0, 0.0, -1.0])),
]
UP = np.array([0.0, 1.0, 0.0])       # Unity world up

STRATEGIES = [
    ("posture", "position-only IK, redundancy pulled toward q_reach; hub -> target"),
    ("pointing", "tool axis aimed at the mouth; hub -> 10 cm further out -> target"),
]

RULE = "=" * 88


# ============================== geometry helpers ==============================

def unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(u), unit(v)), -1.0, 1.0))))


def nearest_world_axis(v):
    """Label of the signed world axis closest to v, and the angle to it (deg)."""
    label, axis = max(DIRECTIONS, key=lambda d: float(np.dot(unit(v), d[1])))
    return label, angle_deg(v, axis)


def quat_to_matrix(quat):
    """Rotation matrix of a Unity quaternion given as [x, y, z, w]."""
    x, y, z, w = unit(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def spoon_frame(spoon_data):
    """Spoon local axes in the world frame, from local_to_world_matrix.

    Returns (axes, scale, checks): axes[:, i] is local axis i as a world unit
    vector, scale[i] the column norm that was divided out.
    """
    M = np.asarray(spoon_data["local_to_world_matrix"], dtype=float).reshape(4, 4)
    pos = np.asarray(spoon_data["position"], dtype=float)
    # Layout check: the translation column must equal the reported position.
    # Measured row-major; the transpose branch only guards a platform change.
    if np.linalg.norm(M[:3, 3] - pos) > 1e-3:
        if np.linalg.norm(M.T[:3, 3] - pos) < 1e-3:
            M = M.T
        else:
            raise RuntimeError("local_to_world_matrix translation does not match "
                               "the spoon position under either layout")
    A = M[:3, :3]
    scale = np.linalg.norm(A, axis=0)
    axes = A / scale
    checks = {
        "translation_err_m": float(np.linalg.norm(M[:3, 3] - pos)),
        "orthonormality_err": float(np.max(np.abs(axes.T @ axes - np.eye(3)))),
        "det": float(np.linalg.det(axes)),
        "quaternion_err": float(np.max(np.abs(axes - quat_to_matrix(spoon_data["quaternion"])))),
    }
    return axes, scale, checks


# ============================== inverse kinematics ==============================

def solve_ik(kin, target, q_rest, seeds, axis=None, tol_m=1e-4, tol_axis_deg=0.5,
             iters=800, damping=0.02, null_gain=0.05, axis_weight=0.3, max_step_deg=2.0):
    """Damped-least-squares IK for the grasp point.

    target: grasp-point position, Unity world frame (m).
    axis:   optional tool-axis direction, Unity world frame. When given, the
            grasp frame's z axis (flange -> grasp point, = spoon local -z) is
            aligned with it as well; roll about it stays free.

    Solved in the URDF frame: R_URDF_TO_UNITY is a reflection (det -1), which
    flips the sign of angular quantities such as the rotation error below.
    The null-space term pulls the redundant DOF toward q_rest. Returns the
    converged solution closest to q_rest (max-abs joint difference), or None.
    """
    R = R_URDF_TO_UNITY
    p_des = R.T @ (np.asarray(target, dtype=float) - kin.base_position)
    a_des = None if axis is None else R.T @ unit(axis)
    cos_tol = np.cos(np.deg2rad(tol_axis_deg))

    def errors(q):
        T = kin.robot.fkine(np.deg2rad(q))
        e_p = p_des - T.t
        z = T.R[:, 2]
        ok = np.linalg.norm(e_p) < tol_m and (a_des is None or np.dot(z, a_des) > cos_tol)
        return e_p, z, ok

    best, best_dev = None, np.inf
    for q0 in seeds:
        q = np.array(q0, dtype=float)
        for _ in range(iters):
            e_p, z, ok = errors(q)
            if ok:
                break
            J = kin.robot.jacob0(np.deg2rad(q))          # 6 x 7, [linear; angular], URDF frame
            if a_des is None:
                Jt, e = J[:3], e_p
            else:
                Jt = np.vstack([J[:3], axis_weight * J[3:]])
                e = np.r_[e_p, axis_weight * np.cross(z, a_des)]
            J_pinv = Jt.T @ np.linalg.inv(Jt @ Jt.T + damping ** 2 * np.eye(len(e)))
            null = (np.eye(N_JOINTS) - J_pinv @ Jt) @ (null_gain * np.deg2rad(q_rest - q))
            dq = np.rad2deg(J_pinv @ e + null)
            peak = float(np.max(np.abs(dq)))
            if peak > max_step_deg:
                dq *= max_step_deg / peak
            q = np.clip(q + dq, -JOINT_LIMIT_DEG, JOINT_LIMIT_DEG)
        if not errors(q)[2]:
            continue
        dev = float(np.max(np.abs(q - q_rest)))
        if dev < best_dev:
            best, best_dev = q, dev
    return best


# IK seeds along the reachable path; the pointing strategy adds random ones.
PATH_SEEDS = [f * Q_REACH for f in (1.0, 0.9, 0.8, 0.7)]


def random_seeds(n, rng):
    """Random IK seeds inside the joint limits (continuous joints: +-180 deg)."""
    span = np.where(np.isfinite(JOINT_LIMIT_DEG), 0.8 * JOINT_LIMIT_DEG, 180.0)
    return [rng.uniform(-span, span) for _ in range(n)]


# ============================== simulator rig ==============================

class Rig:
    """Position-controlled arm plus the readouts this calibration needs."""

    def __init__(self, graphics, time_scale, port=None):
        kw = {"graphics": graphics}
        if port is not None:
            kw["port"] = int(port)
        self.env = FeedingEnv(**kw)
        self.env.step()
        self.env.SetTimeScale(time_scale)
        self.env.step(5)
        self.robot = self.env.get_robot()
        self.mouth = self.env.get_mouth()
        self.spoon = self.env.GetAttr(SPOON_ID)
        self.env.step()
        self.robot.SetJointStiffness([POS_STIFFNESS] * N_JOINTS)
        self.robot.SetJointDamping([POS_DAMPING] * N_JOINTS)
        self.env.step(5)

    # ---- readers
    def q(self):
        return np.array(self.robot.data.get("joint_positions", [0.0] * N_JOINTS), float)

    def tip(self):
        return np.array(self.robot.data.get("grasp_point_position", [0, 0, 0]), float)

    def mouth_pos(self):
        return np.array(self.mouth.data.get("position", [0, 0, 0]), float)

    def contacts(self):
        """Current collision pairs as (id_a, id_b). Costs one physics tick."""
        self.env.GetCurrentCollisionPairs()
        self.env.step()
        return [tuple(int(i) for i in p) for p in self.env.data.get("collision_pairs", [])]

    def describe(self, obj_id):
        attr = self.env.attrs.get(obj_id)
        name = attr.data.get("name") if attr is not None and isinstance(attr.data, dict) else None
        return f"{name}({obj_id})" if name else f"?({obj_id})"

    # ---- motion
    def command(self, q_cmd, ticks):
        self.robot.SetJointPosition(joint_positions=np.asarray(q_cmd, float).tolist())
        self.env.step(ticks)

    def settle(self, q_cmd, max_ticks, tol_deg=0.05, chunk=5):
        """Hold q_cmd until the joints stop moving (or max_ticks)."""
        self.command(q_cmd, chunk)
        used, prev = chunk, self.q()
        while used < max_ticks:
            self.command(q_cmd, chunk)
            used += chunk
            cur = self.q()
            if np.max(np.abs(cur - prev)) < tol_deg:
                break
            prev = cur

    def teleport(self, q_cmd, ticks=30):
        """Jump to q_cmd without physics, then hold it with the PD drive.

        Only used for poses measured to be contact-free. grasp_point_position
        lags one tick behind a teleport, hence the settling ticks.
        """
        q_cmd = np.asarray(q_cmd, float).tolist()
        self.robot.SetJointPositionDirectly(q_cmd)
        self.robot.SetJointPosition(q_cmd)
        self.env.step(ticks)

    def move(self, q_from, q_to, step_deg, ticks_per_step):
        """Joint-space interpolation, at most step_deg per increment."""
        q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
        n = max(1, int(np.ceil(np.max(np.abs(q_to - q_from)) / step_deg)))
        for k in range(1, n + 1):
            self.command(q_from + (q_to - q_from) * k / n, ticks_per_step)

    def hold(self, seconds):
        """Keep stepping (so the window stays live) for `seconds` of wall time."""
        if seconds <= 0:
            return
        print(f"   HOLDING {seconds:.0f} s: ", end="", flush=True)
        t_end, shown = time.time() + seconds, None
        while (left := t_end - time.time()) > 0:
            s = int(np.ceil(left))
            if s != shown:
                print(f"{s} ", end="", flush=True)
                shown = s
            self.env.step(1)
        print("done")

    def close(self):
        self.env.close()


# ============================== stops ==============================

def banner(stop, title):
    print("\n" + RULE)
    print(f"#{stop}  {title}")
    print(RULE)


def format_contacts(rig, pairs):
    if not pairs:
        return "none"
    return ", ".join(f"{rig.describe(a)} <-> {rig.describe(b)}" for a, b in pairs)


def touches_human(pairs):
    return any(h in p for p in pairs for h in HUMAN_IDS)


def stop_spoon_axes(rig, q_view, hold_s, settle_ticks):
    banner(1, "SPOON AXES  (question b: which way does the concave side face?)")
    rig.teleport(q_view)
    rig.settle(q_view, settle_ticks)

    axes, scale, checks = spoon_frame(rig.spoon.data)
    spoon_pos = np.array(rig.spoon.data["position"], float)
    tip, mouth = rig.tip(), rig.mouth_pos()
    to_mouth = unit(mouth - spoon_pos)

    print(f"   pose: q = {np.round(q_view, 1).tolist()} deg")
    print(f"   spoon origin: {np.round(spoon_pos, 4).tolist()}   "
          f"grasp point: {np.round(tip, 4).tolist()}")
    print(f"   scale (column norms): {np.round(scale, 4).tolist()}   (expected 0.2 / 0.3 / 0.15)")
    print(f"   checks: translation {checks['translation_err_m'] * 1000:.2f} mm | "
          f"orthonormality {checks['orthonormality_err']:.1e} | det {checks['det']:+.4f} | "
          f"vs quaternion {checks['quaternion_err']:.1e}")
    print()
    print(f"   {'local axis':10s} | {'world direction (unit)':26s} | {'nearest world axis':18s} | "
          f"{'angle to up':>11s} | {'angle to mouth':>14s}")
    print("   " + "-" * 84)
    for i, name in enumerate("xyz"):
        for sign in (+1, -1):
            v = sign * axes[:, i]
            label, off = nearest_world_axis(v)
            print(f"   {('+' if sign > 0 else '-') + name:10s} | "
                  f"{str(np.round(v, 3).tolist()):26s} | {label + f' ({off:4.1f} deg)':18s} | "
                  f"{angle_deg(v, UP):9.1f}   | {angle_deg(v, to_mouth):12.1f}")
    print("   ('angle to up' = to world +Y; 'angle to mouth' = to the spoon-origin -> mouth direction)")

    # Where the positioned point sits on the spoon, in the spoon's own axes.
    rel = axes.T @ (tip - spoon_pos)                      # metres along unit local axes
    out_dir = axes.T @ unit(spoon_pos - tip)               # grasp point -> spoon origin
    k = int(np.argmax(np.abs(out_dir)))
    print()
    print(f"   grasp point relative to spoon origin (m, along local x/y/z): "
          f"{np.round(rel, 4).tolist()}  |d| = {np.linalg.norm(rel):.3f} m")
    print(f"   -> from the grasp point, the spoon origin lies along local "
          f"{'+' if out_dir[k] > 0 else '-'}{'xyz'[k]} "
          f"({angle_deg(out_dir, np.eye(3)[k] * np.sign(out_dir[k])):.1f} deg off)")
    print("   CHECK ON SCREEN: stops #2-#7 position the GRASP POINT. Is it at the bowl,")
    print("   or is the bowl further out? (grasp-point marker requested; may not render)")

    pairs = rig.contacts()
    print(f"   contacts: {format_contacts(rig, pairs)}")
    print("   >>> LOOK NOW: which world direction does the concave side of the bowl face?")
    rig.hold(hold_s)
    return {
        "q_deg": q_view.tolist(),
        "local_to_world_matrix": np.asarray(rig.spoon.data["local_to_world_matrix"], float).tolist(),
        "axes_world": {n: axes[:, i].tolist() for i, n in enumerate("xyz")},
        "scale": scale.tolist(),
        "checks": checks,
        "spoon_origin": spoon_pos.tolist(),
        "grasp_point": tip.tolist(),
        "grasp_point_in_spoon_axes_m": rel.tolist(),
        "contacts": pairs,
    }


def measure(rig, target, q_goal, mouth, direction):
    """Read the arm back after a move and compare with the target."""
    q_meas, tip = rig.q(), rig.tip()
    track = np.abs(q_meas - q_goal)
    offset = tip - mouth
    return {
        "q_meas_deg": q_meas.tolist(),
        "tip": tip.tolist(),
        "err_m": float(np.linalg.norm(tip - target)),
        "offset_from_mouth_m": float(np.linalg.norm(offset)),
        "offset_angle_deg": angle_deg(offset, direction),
        "track_err_max_deg": float(track.max()),
        "track_err_joint": int(np.argmax(track)),
        "contacts": rig.contacts(),
    }


def print_measurement(rig, m, label, indent="      "):
    print(f"{indent}grasp point {np.round(m['tip'], 4).tolist()}: error {m['err_m'] * 100:.1f} cm | "
          f"{m['offset_from_mouth_m'] * 100:.1f} cm from mouth, {m['offset_angle_deg']:.1f} deg "
          f"from {label} | tracking {m['track_err_max_deg']:.1f} deg (joint {m['track_err_joint']})")
    print(f"{indent}contacts: {format_contacts(rig, m['contacts'])}")


def attempt(rig, kin, strategy, target, direction, label, q_hub, extra_seeds, a):
    """Run one IK strategy for one probe. Returns a record with 'status'."""
    rec = {"strategy": strategy}
    if strategy == "posture":
        q_t = solve_ik(kin, target, Q_REACH, PATH_SEEDS)
        q_p = None
    else:
        axis = -direction                                  # tool aimed at the mouth
        q_t = solve_ik(kin, target, Q_REACH, PATH_SEEDS + extra_seeds, axis=axis)
        q_p = None if q_t is None else solve_ik(
            kin, target + PRE_APPROACH_M * direction, q_t, [q_t], axis=axis)
    if q_t is None or (strategy == "pointing" and q_p is None):
        rec["status"] = "no_ik"
        print("      IK: no solution" + (" with the tool axis aimed at the mouth"
                                         if strategy == "pointing" else ""))
        return rec
    rec["q_ik_deg"] = q_t.tolist()
    print(f"      IK: {np.round(q_t, 1).tolist()} deg (max |q - q_reach| "
          f"{np.max(np.abs(q_t - Q_REACH)):.1f} deg)")

    rig.teleport(q_hub)
    q_from = q_hub
    if q_p is not None:
        pre_target = target + PRE_APPROACH_M * direction
        rig.move(q_hub, q_p, a.step_deg, a.substep_ticks)
        rig.settle(q_p, a.settle_ticks)
        m_pre = measure(rig, pre_target, q_p, rig.mouth_pos(), direction)
        rec["pre_approach"] = m_pre
        print(f"      pre-approach ({(PROBE_RADIUS_M + PRE_APPROACH_M) * 100:.0f} cm out):")
        print_measurement(rig, m_pre, label, indent="        ")
        if m_pre["err_m"] > a.reach_tol:
            rec["status"] = "pre_not_reached"
            print("      -> pre-approach not reached; radial approach not attempted")
            return rec
        q_from = q_p

    rig.move(q_from, q_t, a.step_deg, a.substep_ticks)
    rig.settle(q_t, a.settle_ticks)
    m = measure(rig, target, q_t, rig.mouth_pos(), direction)
    rec.update(m)
    rec["status"] = "reached" if m["err_m"] <= a.reach_tol else "not_reached"
    return rec


def stop_direction(rig, kin, stop, label, direction, mouth, q_hub, extra_seeds, a, hold_s):
    banner(stop, f"DIRECTION {label}  (question a: air in front of the face, or inside the head?)")
    target = mouth + PROBE_RADIUS_M * direction
    print(f"   target = mouth + {PROBE_RADIUS_M * 100:.0f} cm along {label}: "
          f"{np.round(target, 4).tolist()}   (tolerance {a.reach_tol * 100:.1f} cm)")
    rec = {"stop": stop, "direction": label, "target": target.tolist(), "attempts": []}

    shown = None
    for k, (strategy, desc) in enumerate(STRATEGIES, start=1):
        print(f"   attempt {k}/{len(STRATEGIES)} [{strategy}: {desc}]")
        att = attempt(rig, kin, strategy, target, direction, label, q_hub, extra_seeds, a)
        rec["attempts"].append(att)
        if "err_m" in att:
            if "pre_approach" in att:
                print(f"      final ({PROBE_RADIUS_M * 100:.0f} cm out):")
            print_measurement(rig, att, label,
                              indent="        " if "pre_approach" in att else "      ")
            print(f"      -> {'REACHED' if att['status'] == 'reached' else 'not reached'}")
        if att["status"] == "reached":
            shown = att
            break

    if shown is None:
        rec["status"] = ("skipped_no_ik" if all(t["status"] == "no_ik" for t in rec["attempts"])
                         else "skipped_not_reached")
        reasons = []
        for t in rec["attempts"]:
            if t["status"] == "no_ik":
                reasons.append(f"[{t['strategy']}] no IK solution")
            elif t["status"] == "pre_not_reached":
                reasons.append(f"[{t['strategy']}] pre-approach missed by "
                               f"{t['pre_approach']['err_m'] * 100:.1f} cm")
            else:
                reasons.append(f"[{t['strategy']}] stopped {t['err_m'] * 100:.1f} cm from the target, "
                               f"{t['offset_from_mouth_m'] * 100:.1f} cm from the mouth, "
                               f"{t['offset_angle_deg']:.0f} deg off {label}"
                               + (", touching the human" if touches_human(t["contacts"]) else ""))
        print(f"   SKIPPED #{stop} {label}: " + "; ".join(reasons) + ".")
        return rec

    rec["status"] = "shown"
    rec["shown_by"] = shown["strategy"]
    print(f"   >>> LOOK NOW: grasp point is {shown['offset_from_mouth_m'] * 100:.0f} cm from the mouth, "
          f"{shown['offset_angle_deg']:.0f} deg off {label} (strategy: {shown['strategy']})")
    rig.hold(hold_s)
    return rec


# ============================== main ==============================

def main(a):
    sys.stdout.reconfigure(line_buffering=True)   # show progress live when piped
    kin = Gen3Kinematics(a.urdf)
    q_view = a.view_f * Q_REACH
    q_hub = a.hub_f * Q_REACH
    extra_seeds = random_seeds(12, np.random.default_rng(0))
    wanted = set(a.only) if a.only else set(range(1, len(DIRECTIONS) + 2))

    print(RULE)
    print("Pose calibration. Stops:")
    if 1 in wanted:
        print("   #1  spoon axes (concave side)")
    for i, (label, _) in enumerate(DIRECTIONS):
        if i + 2 in wanted:
            print(f"   #{i + 2}  grasp point {PROBE_RADIUS_M * 100:.0f} cm from the mouth along {label}")
    print(f"Holds: {8 * a.hold_scale:.0f} s at #1, {5 * a.hold_scale:.0f} s at #2-#7. "
          f"Every attempt starts from f = {a.hub_f:.2f} on the reachable path.")
    print("Failed attempts are shown moving but not held; only verified poses are held.")
    print(RULE)

    rig = Rig(not a.no_graphics, a.time_scale, a.port)
    out = {"q_reach_deg": Q_REACH.tolist(), "hub_f": a.hub_f, "view_f": a.view_f,
           "probe_radius_m": PROBE_RADIUS_M, "pre_approach_m": PRE_APPROACH_M,
           "reach_tol_m": a.reach_tol}
    try:
        if not a.no_marker:
            rig.env.DebugGraspPoint(True)
            rig.env.step()
        mouth = rig.mouth_pos()
        out["mouth"] = mouth.tolist()
        print(f"mouth = {np.round(mouth, 4).tolist()}")

        if 1 in wanted:
            out["spoon_axes"] = stop_spoon_axes(rig, q_view, 8 * a.hold_scale, a.settle_ticks)
        out["probes"] = [
            stop_direction(rig, kin, i + 2, label, d, mouth, q_hub, extra_seeds, a, 5 * a.hold_scale)
            for i, (label, d) in enumerate(DIRECTIONS) if i + 2 in wanted
        ]
    finally:
        rig.close()

    # ---- summary
    print("\n" + RULE)
    print("SUMMARY - please report what you saw at each stop")
    print(RULE)
    if 1 in wanted:
        print("   #1  spoon axes: which world direction does the concave side face?")
    for r in out["probes"]:
        if r["status"] == "shown":
            s = next(t for t in r["attempts"] if t["status"] == "reached")
            note = (f"shown [{s['strategy']}], {s['offset_from_mouth_m'] * 100:.0f} cm out, "
                    f"{s['offset_angle_deg']:.0f} deg off -> air in front of the face / inside the head?")
            touched = touches_human(s["contacts"])
        else:
            note = "SKIPPED: no IK solution" if r["status"] == "skipped_no_ik" else "SKIPPED: not reached"
            touched = any(touches_human(t.get("contacts", [])) for t in r["attempts"])
        print(f"   #{r['stop']}  {r['direction']}: {note}{' [touching human]' if touched else ''}")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visual calibration of face direction and spoon axes")
    p.add_argument("--hold-scale", type=float, default=1.0,
                   help="multiplier on the 8 s / 5 s holds (0 = no holds)")
    p.add_argument("--view-f", type=float, default=0.50,
                   help="stop #1 pose, as f on q = f * q_reach (contact-free for f <= 0.60)")
    p.add_argument("--hub-f", type=float, default=0.60,
                   help="start pose of every attempt, as f on q = f * q_reach")
    # 4 cm keeps a probe within asin(4/12) = 19.5 deg of its own axis, i.e.
    # unambiguously nearer it than any neighbouring axis (90 deg away).
    p.add_argument("--reach-tol", type=float, default=0.04,
                   help="max grasp-point error (m) for a probe to count as reached")
    p.add_argument("--step-deg", type=float, default=1.0,
                   help="max joint change (deg) per interpolation increment")
    p.add_argument("--substep-ticks", type=int, default=3,
                   help="physics ticks per increment (3 = one 0.06 s control step)")
    p.add_argument("--settle-ticks", type=int, default=120,
                   help="max physics ticks to settle at each pose")
    p.add_argument("--time-scale", type=float, default=10.0,
                   help="simulator time scale; holds are wall-clock regardless")
    p.add_argument("--only", type=int, nargs="+", default=None,
                   help="run only these stop numbers, e.g. --only 1 3 6")
    p.add_argument("--no-graphics", action="store_true")
    p.add_argument("--no-marker", action="store_true",
                   help="do not request the grasp-point debug marker")
    p.add_argument("--urdf", type=str, default="gen3_kinematics_only.urdf")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--out", type=str, default="results/calibrate_pose.json")
    main(p.parse_args())
