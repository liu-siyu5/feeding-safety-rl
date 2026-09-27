"""Analytic kinematics for the Kinova Gen3, loaded from URDF.

Provides forward kinematics and the analytic Jacobian of the spoon tip (the
bowl), expressed in the Unity world frame. The tip is a fixed offset from the
URDF's @GraspPoint_Link, which is the gripper's fingertip centre, not the spoon.

Verified facts this module is built on:
  - URDF: gen3_kinematics_only.urdf (mesh references stripped; kinematics intact)
  - 7 revolute joints
  - URDF -> Unity frame mapping (validated to < 1.2 mm over several poses):
        Unity_X = -URDF_Y
        Unity_Y =  URDF_Z
        Unity_Z =  URDF_X
    This mapping is a reflection (det = -1), so angular quantities must be
    handled in the URDF frame and only positions / linear velocities mapped.
  - @GraspPoint_Link matches the simulator's robot.data['grasp_point_position'].
  - Jacobian evaluation cost ~0.07 ms (one FK for the tip offset included), so
    it can be called every control step.
"""

import os
import numpy as np
import roboticstoolbox as rtb


# URDF -> Unity world frame rotation (validated empirically).
R_URDF_TO_UNITY = np.array(
    [[0.0, -1.0, 0.0],
     [0.0,  0.0, 1.0],
     [1.0,  0.0, 0.0]],
    dtype=np.float64,
)

DEFAULT_URDF = "gen3_kinematics_only.urdf"

# Spoon tip = the FRONT of the bowl, relative to @GraspPoint_Link, in that
# link's frame (m). The link's z axis is the tool axis (flange -> fingertips),
# which is also the spoon's local -z. From the bowl Box Collider the user fitted
# to the spoon mesh in the Unity scene (2026-09-25): spoon-local centre
# (-0.0028, 0.0608, -0.63), size z 0.64, spoon scale (0.2, 0.3, 0.15), spoon
# origin at (0, -0.0089, 0.0802) in this frame -> bowl front at 22.3 cm along
# the axis, 0.93 cm toward the concave side; bowl centre at 17.5 cm. The earlier
# visual estimate (13 cm) was 9 cm short of the front.
SPOON_TIP_IN_GRASP = np.array([0.0, 0.0093, 0.2227])

# Concave normal of the spoon bowl, in the @GraspPoint_Link frame. Measured
# 2026-09-25: the spoon's local +y (the concave side, seen on screen with
# calibrate_pose.py) equals the grasp frame's +y to within 0.05 deg.
SPOON_CONCAVE_IN_GRASP = np.array([0.0, 1.0, 0.0])


def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


class Gen3Kinematics:
    """Analytic FK / Jacobian for the Gen3 spoon tip in the Unity world frame."""

    def __init__(self, urdf_path: str = DEFAULT_URDF, base_position=None,
                 tip_in_grasp=SPOON_TIP_IN_GRASP):
        path = os.path.abspath(urdf_path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"URDF not found: {path}. Generate it by stripping mesh tags "
                "from GEN3_URDF_V12.urdf."
            )
        self.robot = rtb.ERobot.URDF(path)
        self.n = self.robot.n
        # Robot base position in the Unity world frame (measured to be the origin,
        # but kept explicit so a relocated robot does not silently break the math).
        self.base_position = (np.zeros(3) if base_position is None
                              else np.asarray(base_position, dtype=np.float64))
        self.tip_in_grasp = np.asarray(tip_in_grasp, dtype=np.float64)

    # ---------- Forward kinematics ----------
    def fk_grasp_unity(self, q_deg) -> np.ndarray:
        """Gripper grasp point (@GraspPoint_Link) in the Unity world frame.

        This is the point the simulator reports as grasp_point_position.
        """
        q = np.deg2rad(np.asarray(q_deg, dtype=np.float64))
        p_urdf = self.robot.fkine(q).t
        return R_URDF_TO_UNITY @ p_urdf + self.base_position

    def fk_tip_unity(self, q_deg) -> np.ndarray:
        """Spoon tip (bowl) position in the Unity world frame, joint angles in degrees."""
        q = np.deg2rad(np.asarray(q_deg, dtype=np.float64))
        T = self.robot.fkine(q)
        p_urdf = T.t + T.R @ self.tip_in_grasp
        return R_URDF_TO_UNITY @ p_urdf + self.base_position

    def spoon_pose_unity(self, q_deg):
        """Spoon tip, tool axis and concave normal in the Unity world frame.

        The tool axis points from the gripper toward the bowl (spoon local -z).
        Both directions are unit vectors; directions map with R_URDF_TO_UNITY
        like positions. One FK evaluation.
        """
        q = np.deg2rad(np.asarray(q_deg, dtype=np.float64))
        T = self.robot.fkine(q)
        tip = R_URDF_TO_UNITY @ (T.t + T.R @ self.tip_in_grasp) + self.base_position
        axis = R_URDF_TO_UNITY @ T.R[:, 2]
        concave = R_URDF_TO_UNITY @ (T.R @ SPOON_CONCAVE_IN_GRASP)
        return tip, axis, concave

    # ---------- Jacobian ----------
    def jacobian_unity(self, q_deg) -> np.ndarray:
        """Positional Jacobian (3 x n) of the spoon tip, in the Unity world frame.

        Units: metres per radian of joint rotation.
        """
        q = np.deg2rad(np.asarray(q_deg, dtype=np.float64))
        J = self.robot.jacob0(q)                         # 6 x n at the grasp point, URDF frame
        r = self.robot.fkine(q).R @ self.tip_in_grasp    # grasp point -> tip, URDF frame
        J_tip = J[:3, :] - _skew(r) @ J[3:, :]           # v_tip = v + w x r
        return R_URDF_TO_UNITY @ J_tip                   # 3 x n, Unity frame
