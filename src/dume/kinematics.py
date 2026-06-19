"""Kinematics for the SO-101: a thin, typed wrapper over lerobot's placo-backed solver.

This is the only module that knows about lerobot's ``RobotKinematics``. It exposes just
what the controller needs: ``fk`` (joints -> end-effector pose) and ``ik`` (current joints +
target pose -> joints), both in the joint order reported by the URDF.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from dume import _placo_fix


@contextmanager
def _silence_native_stdio():
    """Mute C-level stdout/stderr (fd 1/2) within the block.

    placo's RobotWrapper prints a benign "self collisions in neutral position" notice from
    its compiled extension when it loads the URDF (adjacent links share collision geometry at
    their joint, so they always overlap). We don't use placo's collision avoidance, so this is
    noise; suppress it at construction without hiding real Python errors (those surface after).
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])

# Default URDF vendored in the repo; the SO-101 end-effector frame.
DEFAULT_URDF = str(Path(__file__).resolve().parents[2] / "urdf" / "so101_new_calib.urdf")
DEFAULT_EE_FRAME = "gripper_frame_link"


class Kinematics:
    """Forward/inverse kinematics for the SO-101 arm.

    Joint values are in **degrees**, in the order ``joint_names`` (URDF order:
    shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper).
    Poses are 4x4 homogeneous transforms in the arm's base frame.
    """

    def __init__(
        self,
        urdf_path: str = DEFAULT_URDF,
        ee_frame: str = DEFAULT_EE_FRAME,
        *,
        joint_names: list[str] | None = None,
        ik_iterations: int = 6,
    ) -> None:
        _placo_fix.ensure_placo_importable()
        from lerobot.model.kinematics import RobotKinematics

        if not Path(urdf_path).exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self._urdf_path = urdf_path
        with _silence_native_stdio():
            self._kin = RobotKinematics(urdf_path, target_frame_name=ee_frame, joint_names=joint_names)
        self.joint_names: list[str] = list(self._kin.joint_names)
        self.ik_iterations = ik_iterations

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def joint_limits_deg(self) -> np.ndarray:
        """(n_joints, 2) array of [lower, upper] limits in degrees, in ``joint_names`` order.

        Parsed from the URDF (revolute ``<limit>`` is radians). Joints without a limit get
        ``[-180, 180]``.
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(self._urdf_path)
        limits: dict[str, tuple[float, float]] = {}
        for joint in tree.getroot().findall("joint"):
            name = joint.get("name")
            lim = joint.find("limit")
            if name and lim is not None and lim.get("lower") and lim.get("upper"):
                limits[name] = (float(lim.get("lower")), float(lim.get("upper")))
        out = []
        for name in self.joint_names:
            lo, hi = limits.get(name, (-np.pi, np.pi))
            out.append([np.rad2deg(lo), np.rad2deg(hi)])
        return np.array(out, dtype=float)

    def fk(self, joints_deg) -> np.ndarray:
        """Forward kinematics: joint angles (deg) -> 4x4 end-effector pose."""
        return np.asarray(self._kin.forward_kinematics(np.asarray(joints_deg, dtype=float)))

    def manipulability(self, joints_deg) -> float:
        """Yoshikawa position manipulability ``sqrt(det(J Jᵀ))`` at ``joints_deg``.

        ``J`` is the 3×n position Jacobian (finite-differenced). Near 0 means the arm is near a
        positional singularity (fully extended/folded) where motion gets ill-conditioned.
        """
        q = np.asarray(joints_deg, dtype=float)
        base = self.fk(q)[:3, 3]
        n = len(q)
        J = np.empty((3, n))
        step = 0.5
        for i in range(n):
            dq = np.zeros(n)
            dq[i] = step
            J[:, i] = (self.fk(q + dq)[:3, 3] - base) / np.deg2rad(step)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def ik(
        self,
        current_joints_deg,
        target_pose,
        *,
        position_weight: float = 1.0,
        orientation_weight: float = 0.02,
    ) -> np.ndarray:
        """Inverse kinematics: solve for joints (deg) reaching ``target_pose``.

        The placo solver takes one Gauss-Newton step per call, so we iterate a few times
        seeded from the running solution for convergence. ``current_joints_deg`` is the
        seed (use the live measured joints for continuity).
        """
        q = np.asarray(current_joints_deg, dtype=float)
        target = np.asarray(target_pose, dtype=float)
        for _ in range(self.ik_iterations):
            q = self._kin.inverse_kinematics(
                q, target, position_weight=position_weight, orientation_weight=orientation_weight
            )
        return q

    def ik_position_dls(
        self,
        current_joints_deg,
        target_position,
        *,
        damping: float = 0.05,
        iters: int = 12,
        step_deg: float = 0.5,
    ) -> np.ndarray:
        """Damped least-squares IK for *position only*, over this instance's joint set.

        Purpose-built for the velocity-jog wrist-pivot solve (pan/lift/elbow): far smoother and
        faster than the general placo solver, with explicit singularity damping so the arm
        steadies near rank-deficiency instead of jittering. The Jacobian is computed by finite
        differences of ``fk`` position (no placo IK call), so the result is deterministic and
        seed-stable. ``current_joints_deg`` seeds the iteration — pass the internal commanded
        reference, never raw measured joints, for continuity.

        Update rule per step: ``dq = J^T (J J^T + lambda^2 I)^{-1} e``, with ``J`` in metres/radian
        and ``e`` the position error (metres). Returns joints in degrees.
        """
        q = np.asarray(current_joints_deg, dtype=float).copy()
        target = np.asarray(target_position, dtype=float)
        n = len(q)
        lam2 = float(damping) ** 2
        step_rad = np.deg2rad(step_deg)
        eye = np.eye(3)
        for _ in range(iters):
            pos = self.fk(q)[:3, 3]
            e = target - pos
            if np.linalg.norm(e) < 1e-5:
                break
            J = np.empty((3, n))
            for i in range(n):
                dq = np.zeros(n)
                dq[i] = step_deg
                J[:, i] = (self.fk(q + dq)[:3, 3] - pos) / step_rad
            dtheta = J.T @ np.linalg.solve(J @ J.T + lam2 * eye, e)  # radians
            q = q + np.rad2deg(dtheta)
        return q
