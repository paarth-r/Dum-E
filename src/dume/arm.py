"""Arm I/O — the only modules that touch (or simulate) the hardware.

``ArmIO`` is the interface the controller depends on. ``SO101Arm`` drives the real arm via
lerobot's ``SOFollower``; ``SimArm`` is a kinematic stand-in for ``--dry-run`` that simply
adopts commanded joints, so the full control pipeline (input -> plan -> IK -> command) can be
exercised without moving (or owning) hardware.

Joint vectors are length-6, in URDF order:
``[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]``.
The first five are degrees; the sixth (gripper) is normalised 0..100.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

MOTOR_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@runtime_checkable
class ArmIO(Protocol):
    name: str

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_calibrated(self) -> bool: ...
    def read_joints(self) -> np.ndarray: ...
    def write_joints(self, joints) -> None: ...


class SO101Arm:
    """Real SO-101 follower via lerobot's ``SOFollower``."""

    name = "so101"

    def __init__(self, port: str, robot_id: str, *, disable_torque_on_disconnect: bool = True):
        self.port = port
        self.robot_id = robot_id
        self._disable_torque = disable_torque_on_disconnect
        self._robot = None

    def connect(self) -> None:
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        cfg = SOFollowerRobotConfig(
            port=self.port,
            id=self.robot_id,
            use_degrees=True,
            disable_torque_on_disconnect=self._disable_torque,
        )
        self._robot = SOFollower(cfg)
        # calibrate=False: never silently launch the interactive calibration routine;
        # we check is_calibrated() explicitly and tell the user to run `dume calibrate`.
        self._robot.connect(calibrate=False)

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None

    def is_calibrated(self) -> bool:
        return bool(self._robot and self._robot.is_calibrated)

    def read_joints(self) -> np.ndarray:
        obs = self._robot.get_observation()
        return np.array([obs[f"{m}.pos"] for m in MOTOR_ORDER], dtype=float)

    def write_joints(self, joints) -> None:
        joints = np.asarray(joints, dtype=float)
        action = {f"{m}.pos": float(joints[i]) for i, m in enumerate(MOTOR_ORDER)}
        self._robot.send_action(action)


class SimArm:
    """Kinematic simulation: adopts commanded joints immediately. Powers ``--dry-run``."""

    name = "sim"

    def __init__(self, initial_joints=None):
        self._joints = (
            np.array([0.0, -20.0, 20.0, 0.0, 0.0, 50.0], dtype=float)
            if initial_joints is None
            else np.asarray(initial_joints, dtype=float).copy()
        )
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_calibrated(self) -> bool:
        return True

    def read_joints(self) -> np.ndarray:
        return self._joints.copy()

    def write_joints(self, joints) -> None:
        self._joints = np.asarray(joints, dtype=float).copy()
