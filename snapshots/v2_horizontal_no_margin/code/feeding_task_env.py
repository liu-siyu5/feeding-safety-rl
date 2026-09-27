"""Robot-assisted feeding task environment (paper version).

One task, one contact model, one reward, one set of metrics. The safety
mechanism (C1-C4) is injected and is the only thing that differs between
conditions.

Verified platform facts this environment relies on:
  - Joint velocity control works only once the position drive is released:
        SetJointStiffness([0]*7) and SetJointDamping([1000]*7)
    but even then joints do not move as commanded. Measured 2026-09-25 at the
    start pose: commanding a single joint moves it 8-290 % of the commanded
    amount and drags the others by up to 26 deg; joints 5 and 6 barely respond.
  - The default actuation therefore integrates the commanded joint speed into
    position targets tracked by the stiff position drive (stiffness 10000,
    damping 100): 94-112 % of the commanded motion, <= 2.4 deg on the other
    joints at full speed, no drift at zero command. actuation="velocity" keeps
    the old behaviour for comparison with earlier runs.
  - Command scale: deg/s ~ 11.3 * command, |command| <= 5 (the calibration of
    the velocity mode, linear in that range; kept so an action means the same
    joint speed in both modes).
  - The simulator's robot.data['grasp_point_position'] is the gripper fingertip
    centre, ~13 cm short of the spoon bowl. The spoon tip used here is the bowl:
    URDF FK of the measured joints plus the fixed offset
    kinematics.SPOON_TIP_IN_GRASP (FK matches the simulator to < 1.2 mm).
  - Scene ids: robot 315893, mouth 9999, food 19024, human 2333.

Contact model. The mouth region is a sphere of radius R centred on the mouth
target. Contact is resolved with a penalty model F = k * d, where d is the
penetration depth of the spoon tip into the sphere and k the contact stiffness.
Damping is not modelled.

Feeding-pose constraints. A feed counts only if the spoon enters the mouth from
the front with the bowl level: the tool axis (gripper -> bowl) within
MAX_APPROACH_ANGLE_DEG of -FACE_FORWARD and the concave normal within
MAX_TILT_DEG of world up, and the bowl in front of the mouth within
MAX_CENTRE_OFFSET_M of the mouth's centre line (the line through the mouth
along FACE_FORWARD), so touching the lips from below or the side does not count.
The bowl must also be level or tipped down toward the mouth, never tipped up
toward the handle (pitch within [-MAX_PITCH_DOWN_DEG, MAX_PITCH_UP_DEG]), or the
food would slide off the back of the bowl.
The spoon tip is the front of the bowl, which is the part that enters the mouth.

Hold. Arriving in the feeding pose is not enough: the spoon must stay in it for
HOLD_STEPS consecutive control steps (HOLD_TIME_S) while the person takes the
food, and the force must never exceed F_d. The episode ends when the hold is
complete.
The references were measured with calibrate_pose.py (face along world +Z;
concave normal = spoon local +y); the limits are our own choice.

Face collisions are physical. The Unity build (FeedingPlayer_v2) has colliders
on the head, cheeks, jaw and neck with an opening at the mouth, and the spoon
is attached to the arm as a dynamic body (FeedingEnv.attach_spoon), so the face
stops the spoon instead of being penetrated. Measured 2026-09-25: along the
centre line the bowl front reaches the mouth point; nose, chin and cheeks block
the spoon elsewhere. (This replaces a Python keep-out box used before, whose
guessed face dimensions let the spoon into the nose.)

Thresholds. Comfort F_c = 6 N, danger F_d = 10 N. F_d follows the force limit
used in Assistive Gym; F_c is our own choice and is not drawn from a published
limit.
"""

import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from feeding_env import FeedingEnv
from kinematics import Gen3Kinematics
from safety import SafetyContext, make_mechanism


# ============================== parameters ==============================

# Contact model
STIFFNESS_K = 500.0        # contact stiffness k (N/m)
MOUTH_RADIUS = 0.04        # mouth sphere radius R (m)
COMFORT_FORCE = 6.0        # F_c (N)
DANGER_FORCE = 10.0        # F_d (N)

# Task
N_JOINTS = 7
MAX_STEPS = 200            # 200 steps at 20 Hz = 10 s of simulated time
SUCCESS_DIST = 0.03        # success when the tip is within 3 cm of the mouth

# Feeding-pose constraints (references measured 2026-09-25 with calibrate_pose.py;
# the angle limits are our own choice)
FACE_FORWARD = np.array([0.0, 0.0, 1.0])   # world direction the face points
WORLD_UP = np.array([0.0, 1.0, 0.0])
MAX_APPROACH_ANGLE_DEG = 30.0  # tool axis vs straight into the face (-FACE_FORWARD)
MAX_TILT_DEG = 30.0            # concave normal vs world up
MAX_CENTRE_OFFSET_M = 0.015    # bowl distance from the mouth's centre line (m)
# Spoon pitch = elevation of the tool axis (gripper -> bowl); + = bowl higher than
# the gripper, which tips the food toward the handle. Before this rule (added
# 2026-09-26 at the user's request) episodes ended up to 19 deg tip-up, pressed
# against the lips; the start pose is level.
MAX_PITCH_UP_DEG = 2.0         # tolerance above level
MAX_PITCH_DOWN_DEG = 20.0      # bowl below the gripper by at most this


# Base reward weights
W_DIST = 1.0               # w_d
W_SUCCESS = 10.0           # w_s
W_ORIENT = 0.5             # w_o, weight of the orientation error e_o in [0, 2]
# w_c, weight of the tip's distance from the centre line (m). At 1.0 the policy
# (pre-feed start, 100k steps) parked ~7 cm out and ~3 cm off the line.
W_CENTRE = 10.0
W_PITCH = 2.0              # w_p, weight of the pitch outside its limits (rad)
D0_DIST = 1.077           # nominal base-to-mouth distance (m), reward offset d0

# Control
CONTROL_DT = 0.06          # 20 Hz control interval (s)
SIM_STEPS_PER_CONTROL = 3   # physics ticks per control step (dt = 3 * 0.02 s)
TIME_SCALE = 10.0          # simulation runs 10x real time; verified to leave trajectories bit-identical
MAX_JOINT_VEL_CMD = 5.0    # command magnitude at |action| = 1 (inside the linear range)
CMD_TO_DEG_PER_S = 11.3    # measured: deg/s per unit of command

# Hold phase (added 2026-09-26). After arriving in the feeding pose the spoon
# must stay in it for HOLD_STEPS consecutive control steps, the time the person
# needs to take the food; leaving resets the count. Success = hold completed and
# the force never above F_d. Before, the episode ended on arrival, so a violation
# on arrival was never followed by a step in which the reactive shield (C3)
# could act, and C3 behaved exactly like C1.
HOLD_TIME_S = 0.5
HOLD_STEPS = int(round(HOLD_TIME_S / CONTROL_DT))     # 8 steps = 0.48 s

# Joint limits (deg). Joints reported as [0, 0] by the platform are treated as
# unconstrained rather than locked, since they demonstrably move.
JOINT_LIMIT_FALLBACK = 1e6   # continuous-rotation joints: effectively unlimited

# Episode start randomisation (deg). Zero would make every episode identical,
# so evaluation across episodes would carry no variance.
INIT_JOINT_NOISE = 8.0

# Start pose: on the joint-space path from home to Q_FRONT, a verified feeding
# pose (bowl 2.5 cm from the mouth, from the front, concave up). Along that path
# the bowl distance and both angles fall monotonically, tracking error stays
# <= 0.45 deg, and the arm touches the human only lightly at f = 0.90-0.95.
# f = 0.5 starts the bowl 0.46 m from the mouth, close to the previous start.
Q_FRONT = np.array([45.3, 15.1, -78.1, -46.6, 36.3, -61.3, 52.9])
START_F = 0.5

# Default start ("pre_feed"): bowl front 15 cm in front of the mouth on the
# centre line, tool axis -Z, concave up, i.e. the pose a feeding robot reaches
# by motion planning before the final bite transfer, which is what is learned
# here. With +-3 deg joint noise the front starts 12-19 cm in front and within
# 4.7 cm of the centre line; 12 of 12 resets were contact-free with the spoon
# attached to < 1 mm. start="path" restores START_F * Q_FRONT with +-8 deg noise.
Q_PRE_FEED = np.array([13.7, 28.7, -57.7, -49.0, 37.1, -79.1, 51.5])
INIT_JOINT_NOISE_PRE_FEED = 3.0

# Drive gains that enable velocity control (actuation="velocity")
VEL_MODE_STIFFNESS = 0.0
VEL_MODE_DAMPING = 1000.0

# Drive gains for position-target actuation (the default)
POS_MODE_STIFFNESS = 10000.0
POS_MODE_DAMPING = 100.0
# Anti-windup: the position target may lead the measured joints by at most this
# much (deg), so a blocked joint cannot build up a large jump. At full speed the
# target leads by ~6 deg (3.4 deg per step plus ~2.4 deg tracking lag). C4
# includes the lead in its prediction (SafetyContext.target_lead_deg).
TARGET_LEAD_MAX_DEG = 10.0

# Reset (position actuation). SetJointPositionDirectly keeps the joint
# velocities, so the arm is braked (damping only) before the teleport and then
# held by the position drive. Without the brake the previous episode leaked into
# the next: up to 3 deg off the start pose and 66 deg/s still moving.
RESET_BRAKE_TICKS = 10
RESET_SETTLE_TICKS = 30


def check_thresholds(verbose=True):
    """Check that the success threshold and the safety thresholds are consistent.

    If the force at the success distance already exceeds F_c, then every success
    is also a violation and the comparison between conditions becomes meaningless.
    """
    f_success = STIFFNESS_K * max(0.0, MOUTH_RADIUS - SUCCESS_DIST)
    ok = f_success < COMFORT_FORCE
    if verbose:
        print("-" * 62)
        print("[threshold consistency]")
        print(f"  success distance      = {SUCCESS_DIST:.3f} m")
        print(f"  force at success      = {f_success:.2f} N")
        print(f"  comfort / danger      = {COMFORT_FORCE:.1f} N / {DANGER_FORCE:.1f} N")
        if f_success >= DANGER_FORCE:
            print("  >>> success would always be a danger violation. Fix before running.")
        elif f_success >= COMFORT_FORCE:
            print("  >>> success would always be a comfort violation. Consider a larger "
                  "success distance.")
        else:
            print("  >>> OK: the success point lies inside the safe band.")
        print("-" * 62)
    return ok, f_success


# ============================== environment ==============================

class FeedingTaskEnv(gym.Env):
    """Bite-transfer task with a pluggable safety mechanism.

    :param condition: 'C1', 'C2', 'C3' or 'C4'
    :param lam: penalty weight, required for C2
    :param graphics: render the simulator window
    :param urdf_path: kinematics URDF (mesh-stripped)
    :param actuation: 'position' (default) or 'velocity' (pre-2026-09-25 behaviour)
    :param start: 'pre_feed' (default, bowl 15 cm in front of the mouth) or 'path'
    """

    metadata = {"render_modes": []}

    def __init__(self, condition="C1", lam=None, graphics=False,
                 max_steps=MAX_STEPS, seed=None,
                 urdf_path="gen3_kinematics_only.urdf",
                 init_joint_noise=None, check=True, port=None,
                 actuation="position", start="pre_feed"):
        super().__init__()
        if actuation not in ("position", "velocity"):
            raise ValueError(f"unknown actuation: {actuation}")
        if start not in ("pre_feed", "path"):
            raise ValueError(f"unknown start: {start}")
        self.actuation = actuation
        self.start = start
        if init_joint_noise is None:
            init_joint_noise = INIT_JOINT_NOISE_PRE_FEED if start == "pre_feed" else INIT_JOINT_NOISE

        self.condition = condition.upper()
        self.lam = lam
        self.max_steps = max_steps
        self.init_joint_noise = float(init_joint_noise)
        self.step_count = 0

        if check:
            ok, f_succ = check_thresholds(verbose=True)
            if not ok:
                warnings.warn(
                    f"Force at the success distance is {f_succ:.2f} N, at or above "
                    f"the comfort threshold {COMFORT_FORCE} N. Success and violation "
                    "coincide; results will not be interpretable.",
                    RuntimeWarning,
                )

        if seed is not None:
            super().reset(seed=seed)

        # Analytic kinematics (URDF), used by C4 and for diagnostics.
        self.kin = Gen3Kinematics(urdf_path)

        # Simulator
        # An explicit port lets several runs share one machine; without it every
        # process binds the same default and all but the first fail with EADDRINUSE.
        env_kwargs = {"graphics": graphics}
        if port is not None:
            env_kwargs["port"] = int(port)
        self.env = FeedingEnv(**env_kwargs)
        self.env.step()
        # Run the simulator faster than real time. Verified to leave trajectories
        # unchanged to within floating-point equality, so this only affects wall
        # clock cost, not the physics.
        self.env.SetTimeScale(TIME_SCALE)
        self.env.step(5)
        self.robot = self.env.get_robot()
        self.mouth = self.env.get_mouth()
        self.env.step()

        if self.actuation == "velocity":
            # Release the position drive so that velocity commands take effect.
            self._enable_velocity_control()
        else:
            self._enable_position_control()
        self.q_target = None
        # Dynamic spoon fixed to the end effector, so the face colliders stop it.
        self.spoon = self.env.attach_spoon()

        # Start in front of the mouth (Q_PRE_FEED) or on the path to the frontal
        # feeding pose (START_F). The start before 2026-09-25,
        # 0.3 * [38.3, -33.4, -99.0, -45.0, -1.4, -40.5, 22.2], led to a pose with
        # the gripper at the mouth and the bowl inside the head.
        self.init_joints = Q_PRE_FEED.copy() if start == "pre_feed" else START_F * Q_FRONT

        # Joint limits from the platform, with a fallback for joints it reports as [0, 0].
        lo = np.array(self.robot.data.get("joint_lower_limit", [0.0] * N_JOINTS), float)
        hi = np.array(self.robot.data.get("joint_upper_limit", [0.0] * N_JOINTS), float)
        degenerate = (np.abs(hi - lo) < 1e-6)
        lo = np.where(degenerate, -JOINT_LIMIT_FALLBACK, lo)
        hi = np.where(degenerate, JOINT_LIMIT_FALLBACK, hi)
        self.joint_low, self.joint_high = lo, hi

        # Action: normalised joint velocity commands.
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(N_JOINTS,), dtype=np.float32)

        # Observation: joint positions (7), joint velocities (7), tip position (3),
        # tip-to-mouth vector (3), contact force (1), tool axis (3),
        # concave normal (3) = 27.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(27,), dtype=np.float32)

        # Safety mechanism
        self.mechanism = make_mechanism(self.condition, lam=self.lam,
                                        action_low=-1.0, action_high=1.0)

        self._reset_episode_stats()

    # -------------------------------------------------- platform helpers

    def _enable_velocity_control(self):
        self.robot.SetJointStiffness([VEL_MODE_STIFFNESS] * N_JOINTS)
        self.robot.SetJointDamping([VEL_MODE_DAMPING] * N_JOINTS)
        self.env.step(5)

    def _enable_position_control(self):
        self.robot.SetJointStiffness([POS_MODE_STIFFNESS] * N_JOINTS)
        self.robot.SetJointDamping([POS_MODE_DAMPING] * N_JOINTS)
        self.env.step(5)

    # -------------------------------------------------- readers

    def _joints(self):
        return np.array(self.robot.data.get("joint_positions", [0.0] * N_JOINTS),
                        dtype=np.float64)

    def _joint_vels(self):
        return np.array(self.robot.data.get("joint_velocities", [0.0] * N_JOINTS),
                        dtype=np.float64)

    def _tip_pos(self):
        # Spoon bowl, not grasp_point_position (see module docstring). Same model
        # as the Jacobian that C3 / C4 use.
        return self.kin.fk_tip_unity(self._joints())

    def _spoon_pose(self):
        """(tip, tool axis, concave normal), Unity world frame."""
        return self.kin.spoon_pose_unity(self._joints())

    def _mouth_pos(self):
        return np.array(self.mouth.data.get("position", [0, 0, 0]), dtype=np.float64)

    @staticmethod
    def _centre_offset(tip, mouth):
        """(distance from the mouth's centre line, signed distance in front of the mouth), m."""
        rel = tip - mouth
        forward = float(rel @ FACE_FORWARD)
        return float(np.linalg.norm(rel - forward * FACE_FORWARD)), forward

    @staticmethod
    def _pose_angles(axis, concave):
        """(approach angle, tilt) in degrees; (0, 0) = straight in from the front, level."""
        approach = np.degrees(np.arccos(np.clip(axis @ -FACE_FORWARD, -1.0, 1.0)))
        tilt = np.degrees(np.arccos(np.clip(concave @ WORLD_UP, -1.0, 1.0)))
        return float(approach), float(tilt)

    @staticmethod
    def _pitch(axis):
        """Elevation of the tool axis in degrees; + = bowl higher than the gripper."""
        return float(np.degrees(np.arcsin(np.clip(axis @ WORLD_UP, -1.0, 1.0))))

    # -------------------------------------------------- contact model

    @staticmethod
    def _penetration(tip, mouth):
        return max(0.0, MOUTH_RADIUS - float(np.linalg.norm(tip - mouth)))

    def _contact_force(self, tip, mouth):
        return STIFFNESS_K * self._penetration(tip, mouth)

    # -------------------------------------------------- observation

    def _get_obs(self):
        q = self._joints()
        qd = self._joint_vels()
        tip, axis, concave = self._spoon_pose()
        mouth = self._mouth_pos()
        force = self._contact_force(tip, mouth)
        return np.concatenate([q, qd, tip, mouth - tip, [force],
                               axis, concave]).astype(np.float32)

    # -------------------------------------------------- episode statistics

    def _reset_episode_stats(self):
        self.ep_force_sum = 0.0
        self.ep_max_force = 0.0
        self.ep_comfort_violations = 0
        self.ep_danger_violations = 0
        self.ep_shield_steps = 0
        self.ep_correction_sum = 0.0
        self.ep_reached = False          # in the feeding pose at least once
        self.ep_held = False             # hold completed
        self.hold_count = 0              # consecutive steps in the feeding pose
        self.hold_best = 0               # longest such run this episode
        self.ep_danger_free = True
        self.ep_min_dist = float("inf")
        self.ep_angles_at_min_dist = (float("nan"),) * 3     # approach, tilt, pitch
        self.ep_centre_at_min_dist = (float("nan"), float("nan"))
        self.ep_final_angles = (float("nan"),) * 3
        self.ep_near_mouth = False       # within SUCCESS_DIST, whatever the pose
        self.ep_path_length = 0.0
        self.ep_time_to_success = -1
        self._prev_tip = None

    # -------------------------------------------------- gym interface

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.mechanism.reset()

        # Place the arm at a randomised start pose using position control,
        # then hand back to velocity control for the episode itself.
        start = self.init_joints.copy()
        if self.init_joint_noise > 0:
            start = start + self.np_random.uniform(
                -self.init_joint_noise, self.init_joint_noise, size=N_JOINTS)
        start = np.clip(start, self.joint_low, self.joint_high)

        if self.actuation == "position":
            # Brake, teleport, hold: every episode starts at `start`, at rest,
            # whatever the previous one did (the start region is contact-free).
            self.robot.SetJointVelocity([0.0] * N_JOINTS)
            self.robot.SetJointStiffness([VEL_MODE_STIFFNESS] * N_JOINTS)
            self.robot.SetJointDamping([VEL_MODE_DAMPING] * N_JOINTS)
            self.env.step(RESET_BRAKE_TICKS)
            self.robot.SetJointPositionDirectly(start.tolist())
            self.robot.SetJointStiffness([POS_MODE_STIFFNESS] * N_JOINTS)
            self.robot.SetJointDamping([POS_MODE_DAMPING] * N_JOINTS)
            self.robot.SetJointPosition(start.tolist())
            self.env.step(RESET_SETTLE_TICKS)
        else:
            self._enable_position_control()
            self.robot.SetJointPosition(start.tolist())
            self.env.step(30)
            self._enable_velocity_control()
        self.q_target = start.copy()

        self._reset_episode_stats()
        self._prev_tip = self._tip_pos()
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        q = self._joints()
        qd = self._joint_vels()
        tip = self._tip_pos()
        mouth = self._mouth_pos()
        force_before = self._contact_force(tip, mouth)

        # Analytic Jacobian at the current configuration, in the Unity world frame.
        J = self.kin.jacobian_unity(q)

        ctx = SafetyContext(
            q_deg=q, tip_pos=tip, mouth_pos=mouth, force=force_before,
            dt=CONTROL_DT, vel_scale=np.deg2rad(MAX_JOINT_VEL_CMD * CMD_TO_DEG_PER_S),
            stiffness_k=STIFFNESS_K, mouth_radius=MOUTH_RADIUS,
            comfort_force=COMFORT_FORCE, danger_force=DANGER_FORCE,
            # Motion already on its way: the lead the position drive still owes
            # and the current joint velocity. C4 predicts from both.
            target_lead_deg=(self.q_target - q if self.actuation == "position"
                             else np.zeros(N_JOINTS)),
            joint_vel_deg=qd,
            # Retreats (C3, C4) go straight out of the mouth, horizontally.
            retreat_dir=FACE_FORWARD,
        )
        ctx.jacobian = J   # attached dynamically; used by C3 and C4

        # ---- safety mechanism acts on the action ----
        exec_action, sinfo = self.mechanism.filter(action, ctx)
        exec_action = np.clip(np.asarray(exec_action, dtype=np.float64), -1.0, 1.0)
        if sinfo.get("intervened", False):
            self.ep_shield_steps += 1
            self.ep_correction_sum += sinfo.get("correction", 0.0)

        # ---- execute ----
        cmd = exec_action * MAX_JOINT_VEL_CMD
        if self.actuation == "velocity":
            self.robot.SetJointVelocity(cmd.tolist())
        else:
            # Integrate the commanded joint speed into the position target.
            dq = cmd * CMD_TO_DEG_PER_S * CONTROL_DT
            self.q_target = np.clip(self.q_target + dq,
                                    q - TARGET_LEAD_MAX_DEG, q + TARGET_LEAD_MAX_DEG)
            self.q_target = np.clip(self.q_target, self.joint_low, self.joint_high)
            self.robot.SetJointPosition(self.q_target.tolist())
        self.env.step(SIM_STEPS_PER_CONTROL)

        # ---- read the resulting state ----
        tip_new, axis_new, concave_new = self._spoon_pose()
        mouth_new = self._mouth_pos()
        dist = float(np.linalg.norm(tip_new - mouth_new))
        force = self._contact_force(tip_new, mouth_new)
        approach, tilt = self._pose_angles(axis_new, concave_new)
        pitch = self._pitch(axis_new)
        off_centre, forward = self._centre_offset(tip_new, mouth_new)

        # ---- statistics ----
        self.ep_force_sum += force
        self.ep_max_force = max(self.ep_max_force, force)
        if dist < self.ep_min_dist:
            self.ep_min_dist = dist
            self.ep_angles_at_min_dist = (approach, tilt, pitch)
            self.ep_centre_at_min_dist = (off_centre, forward)
        self.ep_final_angles = (approach, tilt, pitch)
        self.ep_near_mouth = self.ep_near_mouth or dist < SUCCESS_DIST
        if self._prev_tip is not None:
            self.ep_path_length += float(np.linalg.norm(tip_new - self._prev_tip))
        self._prev_tip = tip_new
        if force > COMFORT_FORCE:
            self.ep_comfort_violations += 1
        if force > DANGER_FORCE:
            self.ep_danger_violations += 1
            self.ep_danger_free = False

        # ---- reward ----
        # r = -w_d * dist - w_c * off_centre - w_o * e_o - w_p * e_p + hold bonus,
        # with e_o = (1 - cos approach) / 2 + (1 - cos tilt) / 2 in [0, 2] and
        # e_p the pitch outside [-MAX_PITCH_DOWN_DEG, MAX_PITCH_UP_DEG] (rad).
        # The feeding pose needs the distance, the angle and pitch limits and the
        # tip centred on the mouth's centre line; the face colliders keep the
        # spoon out of the face, so a sideways or through-the-face approach
        # cannot reach it.
        e_orient = 0.5 * ((1.0 - np.cos(np.deg2rad(approach)))
                          + (1.0 - np.cos(np.deg2rad(tilt))))
        e_pitch = np.deg2rad(max(0.0, pitch - MAX_PITCH_UP_DEG)
                             + max(0.0, -pitch - MAX_PITCH_DOWN_DEG))
        reward = (-W_DIST * dist - W_CENTRE * off_centre - W_ORIENT * e_orient
                  - W_PITCH * e_pitch)
        reached = (dist < SUCCESS_DIST and approach <= MAX_APPROACH_ANGLE_DEG
                   and tilt <= MAX_TILT_DEG and off_centre <= MAX_CENTRE_OFFSET_M
                   and -MAX_PITCH_DOWN_DEG <= pitch <= MAX_PITCH_UP_DEG)
        self.hold_count = self.hold_count + 1 if reached else 0
        self.ep_reached = self.ep_reached or reached
        # Hold bonus: w_s paid in HOLD_STEPS equal parts, one for each step the
        # consecutive hold passes its best so far this episode. Leaving and coming
        # back earns nothing until the old best is passed, so the total is at most
        # w_s and moving in and out of the pose does not pay.
        if self.hold_count > self.hold_best:
            reward += W_SUCCESS / HOLD_STEPS
            self.hold_best = self.hold_count
        held = self.hold_count >= HOLD_STEPS
        if held and not self.ep_held:
            self.ep_held = True
            self.ep_time_to_success = self.step_count
        # Mechanism-specific reward term (non-zero only for C2).
        reward += self.mechanism.reward_term(
            SafetyContext(q_deg=q, tip_pos=tip_new, mouth_pos=mouth_new, force=force,
                          dt=CONTROL_DT, vel_scale=ctx.vel_scale,
                          stiffness_k=STIFFNESS_K, mouth_radius=MOUTH_RADIUS,
                          comfort_force=COMFORT_FORCE, danger_force=DANGER_FORCE)
        )

        # ---- termination ----
        joints_now = self._joints()
        joint_violation = bool(np.any(joints_now < self.joint_low - 1e-6) or
                               np.any(joints_now > self.joint_high + 1e-6))
        # A joint at its limit stops there; it does not end the episode. The
        # violation is still recorded as a metric. The episode ends when the
        # hold is complete.
        terminated = bool(held)
        truncated = self.step_count >= self.max_steps

        info = {"distance": dist, "contact_force": force, "reached": reached,
                "hold_steps": self.hold_count,
                "approach_deg": approach, "tilt_deg": tilt, "pitch_deg": pitch,
                "centre_offset": off_centre, "forward": forward}
        if terminated or truncated:
            info["episode_metrics"] = self._episode_metrics(joint_violation)
        return self._get_obs(), float(reward), terminated, truncated, info

    def _episode_metrics(self, joint_violation=False):
        n = max(self.step_count, 1)
        return {
            # task performance
            "success": float(self.ep_held and self.ep_danger_free),
            "held": float(self.ep_held),          # hold completed, ignoring safety
            "reached": float(self.ep_reached),    # in the feeding pose at least once
            "hold_best": self.hold_best,          # longest consecutive hold (steps)
            "time_to_success": self.ep_time_to_success,   # step the hold completed
            "path_length": self.ep_path_length,
            # safety
            "peak_force": self.ep_max_force,
            "mean_force": self.ep_force_sum / n,
            "comfort_violation_rate": self.ep_comfort_violations / n,
            "danger_violation_rate": self.ep_danger_violations / n,
            "force_impulse": self.ep_force_sum * CONTROL_DT,   # N*s
            # intervention (meaningful for C3 and C4)
            "shield_rate": self.ep_shield_steps / n,
            "mean_correction": (self.ep_correction_sum / self.ep_shield_steps
                                if self.ep_shield_steps > 0 else 0.0),
            # diagnostics
            "min_distance": self.ep_min_dist,
            "near_mouth": float(self.ep_near_mouth),
            "approach_at_min_dist": self.ep_angles_at_min_dist[0],
            "tilt_at_min_dist": self.ep_angles_at_min_dist[1],
            "pitch_at_min_dist": self.ep_angles_at_min_dist[2],
            "centre_offset_at_min_dist": self.ep_centre_at_min_dist[0],
            "forward_at_min_dist": self.ep_centre_at_min_dist[1],
            "final_approach": self.ep_final_angles[0],
            "final_tilt": self.ep_final_angles[1],
            "final_pitch": self.ep_final_angles[2],
            "ep_length": self.step_count,
            "joint_violation": float(joint_violation),
        }

    def close(self):
        self.env.close()


if __name__ == "__main__":
    check_thresholds(verbose=True)
