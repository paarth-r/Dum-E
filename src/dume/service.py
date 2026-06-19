"""``DumeArm`` — the public, reusable API. Other services import *this*.

It owns the kinematics, the arm I/O, and a controller, and exposes intent-level methods
(``get_pose``, ``goto``, ``follow_path``, ``jog``, ``home``, ``set_gripper``) that work the
same whether driving real hardware or a simulation (``dry_run=True``). The Xbox CLI is just
one consumer of this class.
"""

from __future__ import annotations

import time

import numpy as np

from dume import geometry as g
from dume.arm import ArmIO, SimArm, SO101Arm
from dume.config import ControllerConfig
from dume.controller import Controller
from dume.input_xbox import Command
from dume.kinematics import Kinematics
from dume.poses import HOME_JOINTS, PoseStore


class DumeArm:
    """High-level handle on the SO-101. Use as a context manager."""

    def __init__(
        self,
        config: ControllerConfig | None = None,
        *,
        dry_run: bool = False,
        arm: ArmIO | None = None,
        poses: PoseStore | None = None,
    ):
        self.config = config or ControllerConfig()
        self.kin = Kinematics(self.config.urdf_path, self.config.ee_frame)
        if arm is not None:
            self.arm = arm
        elif dry_run:
            self.arm = SimArm(initial_joints=HOME_JOINTS)
        else:
            self.arm = SO101Arm(self.config.port, self.config.robot_id)
        self.controller = Controller(
            self.config, self.arm, self.kin, poses or PoseStore()
        )

    # ---- lifecycle ----
    def __enter__(self) -> "DumeArm":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def connect(self) -> None:
        self.controller.start()

    def disconnect(self) -> None:
        self.controller.stop()

    # ---- queries ----
    def get_joints(self) -> np.ndarray:
        return self.arm.read_joints()

    def get_pose(self) -> np.ndarray:
        """Current end-effector pose (4x4)."""
        return self.kin.fk(self.arm.read_joints())

    def get_xyzrpy(self) -> np.ndarray:
        return g.pose_to_xyzrpy(self.get_pose())

    # ---- motion (plan-then-solve) ----
    def goto(
        self,
        pose_or_xyzrpy,
        *,
        wait: bool = True,
        timeout: float = 15.0,
        honor_orientation: bool = False,
    ):
        """Move the gripper to an absolute pose (4x4 or ``[x,y,z,roll,pitch,yaw]``).

        By default this is position-priority (exact XYZ, natural orientation), which is what
        the 5-DOF SO-101 can actually do. Set ``honor_orientation=True`` to weight the
        requested orientation heavily instead (at the cost of position accuracy).
        """
        arr = np.asarray(pose_or_xyzrpy, dtype=float)
        goal = arr if arr.shape == (4, 4) else g.xyzrpy_to_pose(arr)
        ori_w = self.config.ik_orientation_lock_weight if honor_orientation else None
        self.controller._goto_pose(goal, self.arm.read_joints(), orientation_weight=ori_w)
        if wait:
            self._drive_until_idle(timeout)
        return self.get_pose()

    def follow_path(self, poses, *, wait: bool = True, timeout_per_segment: float = 15.0):
        """Visit a sequence of poses in order (each planned + solved through the pipeline)."""
        for p in poses:
            self.goto(p, wait=wait, timeout=timeout_per_segment)
        return self.get_pose()

    def home(self, *, wait: bool = True, timeout: float = 15.0):
        """Joint-space return to the neutral config captured at connect (exact, no IK)."""
        target = self.controller.home_joints
        if target is None:
            target = HOME_JOINTS
        self.controller._joint_target = target.copy()
        if wait:
            deadline = time.perf_counter() + timeout
            while self.controller._joint_target is not None and time.perf_counter() < deadline:
                self.controller.step(Command())
                if isinstance(self.arm, SO101Arm):
                    time.sleep(self.config.dt)
        return self.get_pose()

    def goto_joints(self, joints, *, wait: bool = True, timeout: float = 15.0):
        """Joint-space move to an exact configuration (length-6, in ``arm.MOTOR_ORDER``).

        Straight-line in joint space (no IK), so it reproduces a captured pose exactly. The
        sixth element drives the gripper. Used to send the arm to its saved start pose.
        """
        target = np.asarray(joints, dtype=float)
        self.controller.gripper_cmd = float(target[5])
        self.controller._joint_target = target.copy()
        if wait:
            deadline = time.perf_counter() + timeout
            while self.controller._joint_target is not None and time.perf_counter() < deadline:
                self.controller.step(Command())
                if isinstance(self.arm, SO101Arm):
                    time.sleep(self.config.dt)
        return self.get_pose()

    def jog(self, lin=(0, 0, 0), wrist_pitch=0.0, wrist_roll=0.0, gripper: float = 0.0, *, ticks: int = 1):
        """Apply a jog for ``ticks`` ticks: ``lin`` (normalised XYZ in [-1,1]) moves the wrist
        pivot; ``wrist_pitch``/``wrist_roll`` jog the wrist joints; ``gripper`` opens/closes."""
        cmd = Command(
            lin=np.asarray(lin, float),
            wrist_pitch=float(wrist_pitch),
            wrist_roll=float(wrist_roll),
            gripper=gripper,
        )
        tel = None
        for _ in range(ticks):
            tel = self.controller.step(cmd)
        return tel

    def set_gripper(self, value_0_100: float, *, settle_ticks: int = 25):
        """Drive the gripper toward an absolute 0..100 opening and let it settle."""
        target = float(np.clip(value_0_100, self.config.gripper_closed, self.config.gripper_open))
        for _ in range(settle_ticks):
            direction = np.sign(target - self.controller.gripper_cmd)
            self.controller.step(Command(gripper=float(direction)))
            if abs(self.controller.gripper_cmd - target) < 1.0:
                break

    def run_teleop(self, poll, on_tick=None) -> None:
        """Hand the loop to a Command source (e.g. the Xbox controller)."""
        self.controller.run(poll, on_tick=on_tick)

    # ---- internals ----
    def _drive_until_idle(self, timeout: float, *, tol_mm: float = 2.0) -> None:
        """Tick (holding the goal) until the trajectory is consumed *and* the arm settles.

        Settling matters because the per-tick slew limiter can still be catching up to the
        goal after the trajectory timer expires; stopping then would leave residual error.
        """
        idle = Command()
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            tel = self.controller.step(idle)
            settled = self.controller._traj is None and tel.tracking_pos_err_mm < tol_mm
            if isinstance(self.arm, SO101Arm):
                time.sleep(self.config.dt)
            if settled:
                break
