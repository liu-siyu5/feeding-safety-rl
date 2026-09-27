"""The four safety mechanisms compared in the paper (C1-C4).

All four share the same environment, task, contact model, learning algorithm
and evaluation criteria; the mechanism is the only variable.

  C1  Unconstrained     - base reward, no action filtering. Lower bound.
  C2  Reward penalty    - base reward minus lambda * max(0, F - F_c). Soft constraint.
  C3  Reactive shield   - once measured F > F_d, replace the action with a retraction
                          along the approach direction until F < F_c. Post-collision.
  C4  Predictive shield - predict the force at the position the proposed action drives
                          the arm to (including motion already on its way: the drive's
                          lead and the joints' momentum) using the analytic Jacobian,
                          and slow the whole action down (same direction) just enough
                          that the predicted force stays under F_d, with the tip closing
                          at most a fixed fraction of its remaining margin per step.
                          Pre-collision. This is the proposed mechanism.

Every mechanism exposes the same interface:

    filtered_action, info = mechanism.filter(action, ctx)
    bonus_or_penalty      = mechanism.reward_term(ctx)

where `ctx` carries the current state needed to make the decision.
"""

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------- context

@dataclass
class SafetyContext:
    """State handed to a safety mechanism at one control step."""
    q_deg: np.ndarray          # current joint angles (deg), shape (n,)
    tip_pos: np.ndarray        # spoon tip position, Unity world frame (m)
    mouth_pos: np.ndarray      # mouth centre, Unity world frame (m)
    force: float               # current contact force (N)
    dt: float                  # control interval (s)
    vel_scale: float           # command -> rad/s conversion for the action
    stiffness_k: float         # penalty contact stiffness (N/m)
    mouth_radius: float        # mouth sphere radius (m)
    comfort_force: float       # F_c (N)
    danger_force: float        # F_d (N)
    # Position target minus current joint angles (deg), shape (n,): motion the
    # position drive still owes. None (treated as zero) under velocity actuation.
    target_lead_deg: np.ndarray = None
    # Current joint velocities (deg/s), shape (n,). None = treated as zero.
    joint_vel_deg: np.ndarray = None
    # Unit vector pointing straight out of the mouth (the face's forward axis,
    # horizontal). When set, C3 and C4 retreat along it and C4 measures its
    # margin as the depth in front of the mouth along it. None = away from the
    # mouth point (the behaviour before 2026-09-26).
    retreat_dir: np.ndarray = None


# ---------------------------------------------------------------- base class

class SafetyMechanism:
    name = "base"

    def reset(self):
        """Clear any per-episode internal state."""
        pass

    def filter(self, action, ctx):
        """Return (possibly modified action, info dict)."""
        return action, {"intervened": False, "correction": 0.0}

    def reward_term(self, ctx):
        """Additional reward contributed by this mechanism (0 unless it shapes reward)."""
        return 0.0


# ---------------------------------------------------------------- C1

class C1Unconstrained(SafetyMechanism):
    """No constraint at all. Establishes what the task alone induces."""
    name = "C1_unconstrained"


# ---------------------------------------------------------------- C2

class C2RewardPenalty(SafetyMechanism):
    """Force penalty added to the reward, proportional to excess over F_c.

    r = r_base - lambda * max(0, F - F_c)

    lambda is swept over several values; the condition is reported as a
    trade-off curve rather than as a single point.
    """
    name = "C2_reward_penalty"

    def __init__(self, lam: float):
        self.lam = float(lam)

    @property
    def label(self):
        return f"C2_lam{self.lam:g}"

    def reward_term(self, ctx):
        return -self.lam * max(0.0, ctx.force - ctx.comfort_force)


# ---------------------------------------------------------------- C3

class C3ReactiveShield(SafetyMechanism):
    """Reactive shield acting on measured force (post-collision).

    Once the measured force exceeds F_d, the commanded action is replaced by a
    retraction that withdraws the tip straight out of the mouth (ctx.retreat_dir,
    horizontal, as a feeder withdraws a spoon without spilling; before
    2026-09-26 along tip minus mouth point) at a fixed rate. Retraction continues
    until the measured force falls below F_c, after which control returns to the
    policy.

    If retraction fails to reduce the force within `max_retract_steps` steps, the
    shield releases control back to the policy and records a failure, so that a
    stuck retraction cannot deadlock the episode.
    """
    name = "C3_reactive_shield"

    def __init__(self, retract_rate: float = 1.0, max_retract_steps: int = 50):
        self.retract_rate = float(retract_rate)
        self.max_retract_steps = int(max_retract_steps)
        self.reset()

    def reset(self):
        self.retracting = False
        self.retract_counter = 0
        self.retract_failures = 0

    def filter(self, action, ctx):
        # Enter retraction when the danger threshold is crossed.
        if not self.retracting and ctx.force > ctx.danger_force:
            self.retracting = True
            self.retract_counter = 0

        if not self.retracting:
            return action, {"intervened": False, "correction": 0.0}

        # Leave retraction once the force is back inside the comfort band.
        if ctx.force < ctx.comfort_force:
            self.retracting = False
            self.retract_counter = 0
            return action, {"intervened": False, "correction": 0.0}

        # Give up if retraction is not working, rather than deadlocking.
        self.retract_counter += 1
        if self.retract_counter > self.max_retract_steps:
            self.retracting = False
            self.retract_counter = 0
            self.retract_failures += 1
            return action, {"intervened": False, "correction": 0.0,
                            "retract_timeout": True}

        # Retract: move the tip away from the mouth in Cartesian space, mapped to
        # joint space through the Jacobian pseudo-inverse.
        away = (np.asarray(ctx.retreat_dir, dtype=np.float64) if ctx.retreat_dir is not None
                else ctx.tip_pos - ctx.mouth_pos)
        norm = np.linalg.norm(away)
        if norm < 1e-9:
            # Degenerate: tip exactly at the mouth centre. Reverse the action instead.
            retract_action = -np.asarray(action, dtype=np.float64)
        else:
            direction = away / norm
            v_cart = self.retract_rate * direction          # desired tip velocity (m/s)
            J = ctx.jacobian                                 # 3 x n, m per rad
            q_dot = np.linalg.pinv(J) @ v_cart               # rad/s
            retract_action = q_dot / ctx.vel_scale           # back to command units

        correction = float(np.linalg.norm(retract_action - np.asarray(action)))
        return retract_action, {"intervened": True, "correction": correction}


# ---------------------------------------------------------------- C4

class C4PredictiveShield(SafetyMechanism):
    """Predictive shield acting on predicted force (pre-collision). Proposed method.

    Prediction. The joints track a position target, and a proposed action a moves
    that target by a * vel_scale * dt. Two further motions are already on their
    way and are added: the target leads the joints by `lead`
    (ctx.target_lead_deg, in rad below; the tracking lag while moving, more once
    the spoon has been held up, e.g. at the lips), and the joints carry their
    current velocity qd (ctx.joint_vel_deg) through the interval. So

        dp = J(q) @ (lead + qd * dt + a * vel_scale * dt)

    with J the analytic positional Jacobian of the spoon tip. The predicted
    penetration depth into the mouth sphere follows from the predicted tip
    position, and the predicted force from the penalty contact model F = k*d.

    Barrier rate. Stopping the tip exactly at d_min in one step (the original
    constraint, barrier_rate = 1) leaves no margin: any prediction error crosses
    the limit, and a fast arm cannot stop within one step. Instead the tip may
    close at most `barrier_rate` of its remaining margin h = dist - d_min per
    step (a discrete-time control barrier function), h_next >= (1 - rate) * h,
    so it slows down as it nears the limit and far from it is not restricted.

    Measured 2026-09-26 (policies trained with C4, 20 fixed start states):
    predicting only the new increment J(q) @ (a * vel_scale * dt), the tip, held
    up at the lips, jumped past d_min in one step in 7 of 20 episodes (up to
    11.8 N); adding the lead, 0 of 20 for that policy, but a policy retrained
    with it rushed in (15 cm in ~5 steps) and the arm's momentum carried the tip
    past d_min in 6 of 20; adding the velocity term alone, still 6 of 20; with
    it and barrier_rate = 0.5, 1 of 20 (same policy, not retrained). Under
    velocity actuation there is no target and the lead is zero.

    Constraint. Requiring the predicted force to stay under F_d is, after
    linearising the distance about the current tip position, a single half-space
    constraint in action space:

        let u   = (p_tip - p_mouth) / ||p_tip - p_mouth||     (unit outward normal)
            d_min = R - F_d / k                                (min allowed distance)
            c   = u^T J (lead + qd * dt)                       (motion on its way)
        the predicted distance along u is
            dist_pred ~ ||p_tip - p_mouth|| + u . dp
                      = dist + c + (u^T J vel_scale dt) a
        the constraint dist_pred - d_min >= (1 - rate) * (dist - d_min) becomes
            n^T a >= b,   n = (J^T u) * vel_scale * dt,  b = -rate * (dist - d_min) - c

    Retreat direction. With ctx.retreat_dir = f set (the face's forward axis),
    u above is f and dist is the depth in front of the mouth along it,
    forward = (p_tip - p_mouth) . f, so moving up, down or sideways does not
    count as moving away, and a back-off goes straight out, horizontally, the
    way a feeder withdraws a spoon without spilling. Since dist >= forward,
    keeping forward >= d_min still keeps the modelled force under F_d. The check
    is skipped when the tip is more than R off the centre line (then dist >= R
    and the force is zero anyway). Without retreat_dir, u is the direction from
    the mouth point, as before 2026-09-26. Reason for the change: in the paper
    grid one C4 seed parked with the bowl against the upper lip; "away from the
    mouth point" pointed forward and up into the lip, the lip deflected the arm
    and the bowl slid 1-2 mm further in per step (7 % success, 19 % of steps
    over 10 N). With the horizontal margin the same trained policy reached 87 %
    (30 episodes, not retrained).

    Correction. A violating action is scaled as a whole,

        a_safe = s * a,   s = b / (n^T a)  in [0, 1)     (when b <= 0)

    so every joint slows by the same factor and the tip keeps its direction,
    stopping on the limit. If even a = 0 violates (b > 0: the motion already on
    its way would cross the limit), the smallest joint motion that moves the tip
    straight along u by b, a_safe = J^+ (b u) / (vel_scale dt) clipped to the
    action bounds, backs the tip off. (Before 2026-09-26 the back-off was
    (b / ||n||^2) * n, which moves the tip along J J^T u rather than along u.)

    Scaling replaced the minimum-norm projection a + ((b - n^T a) / ||n||^2) * n
    (2026-09-26). The projection removes only the part of the motion that closes
    on the mouth and keeps the rest, so the tip slid sideways along the limit.
    With a policy that feeds on its own (C1: 9 of 20 successes, 11 of 20 over
    10 N), C4 with the projection gave 0 of 20 successes (tip pushed 3-6 cm off
    the centre line); with scaling 16 of 20 (barrier_rate 1) or 8 of 20 (0.5),
    none over 10 N.

    Cost. One Jacobian evaluation, one constraint evaluation and one closed-form
    correction per control step. No optimisation problem is solved and no
    reachable set is computed. The Jacobian is analytic (from the robot URDF),
    so no finite-difference kinematics queries are needed.
    """
    name = "C4_predictive_shield"

    def __init__(self, action_low=-1.0, action_high=1.0, barrier_rate: float = 0.5):
        self.action_low = action_low
        self.action_high = action_high
        # Fraction of the remaining margin to d_min the tip may close per step
        # (1 = allowed to reach d_min in one step).
        self.barrier_rate = float(barrier_rate)

    def filter(self, action, ctx):
        a = np.asarray(action, dtype=np.float64).copy()
        a_proposed = a.copy()

        rel = ctx.tip_pos - ctx.mouth_pos
        if ctx.retreat_dir is not None:
            # Margin measured horizontally: depth in front of the mouth.
            u = np.asarray(ctx.retreat_dir, dtype=np.float64)   # unit, out of the mouth
            dist = float(rel @ u)
            if np.linalg.norm(rel - dist * u) >= ctx.mouth_radius:
                # More than R off the centre line: the force ball cannot be reached.
                return a, {"intervened": False, "correction": 0.0}
        else:
            dist = float(np.linalg.norm(rel))
            if dist < 1e-9:
                # Degenerate: no well-defined outward direction. Fall back to braking.
                return np.zeros_like(a), {"intervened": True,
                                          "correction": float(np.linalg.norm(a))}
            u = rel / dist                               # unit outward normal
        # Minimum distance at which the force stays under F_d.
        d_min = ctx.mouth_radius - ctx.danger_force / ctx.stiffness_k

        # If the mouth sphere is already deeper than the danger threshold allows,
        # d_min may be below zero; clamp so the constraint stays meaningful.
        d_min = max(d_min, 0.0)

        J = ctx.jacobian                                 # 3 x n, m per rad
        n_vec = (J.T @ u) * ctx.vel_scale * ctx.dt       # shape (n,), metres per unit action
        # Distance change already on its way (m): the lead the drive still owes
        # (target ahead of the joints) and the joints' current velocity over dt.
        on_its_way = 0.0
        if ctx.target_lead_deg is not None:
            on_its_way += float(u @ J @ np.deg2rad(ctx.target_lead_deg))
        if ctx.joint_vel_deg is not None:
            on_its_way += float(u @ J @ np.deg2rad(ctx.joint_vel_deg)) * ctx.dt
        b = -self.barrier_rate * (dist - d_min) - on_its_way

        lhs = float(n_vec @ a)
        if lhs >= b - 1e-12:
            return a, {"intervened": False, "correction": 0.0}   # constraint satisfied
        if b <= 0.0:
            # Closing too fast (lhs < b <= 0): slow the whole action down.
            a = (b / lhs) * a
        else:
            # Stopping is not enough: back the tip off straight along u by b.
            dq = np.linalg.pinv(J) @ (b * u)             # rad over one interval
            a = np.clip(dq / (ctx.vel_scale * ctx.dt), self.action_low, self.action_high)

        correction = float(np.linalg.norm(a - a_proposed))
        return a, {"intervened": True, "correction": correction}


# ---------------------------------------------------------------- factory

def make_mechanism(condition: str, lam: float = None, action_low=-1.0, action_high=1.0):
    """Build a mechanism from a condition name: 'C1', 'C2', 'C3' or 'C4'."""
    c = condition.upper()
    if c == "C1":
        return C1Unconstrained()
    if c == "C2":
        if lam is None:
            raise ValueError("C2 requires a penalty weight lambda.")
        return C2RewardPenalty(lam)
    if c == "C3":
        return C3ReactiveShield()
    if c == "C4":
        return C4PredictiveShield(action_low=action_low, action_high=action_high)
    raise ValueError(f"Unknown condition: {condition}")
