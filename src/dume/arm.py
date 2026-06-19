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

import glob
import os
from typing import Callable, Protocol, runtime_checkable

import numpy as np

# macOS exposes the SO-101's USB-serial bridge as /dev/cu.usbmodem<serial>. The trailing serial
# can change across reflashes/ports, so we glob rather than hard-code the suffix.
USBMODEM_GLOB = "/dev/cu.usbmodem*"


def resolve_serial_port(
    preferred: str,
    *,
    exists: Callable[[str], bool] = os.path.exists,
    candidates: Callable[[], list[str]] | None = None,
) -> str:
    """Return a usable serial port, so ``dume run`` works without hand-editing the config.

    If ``preferred`` is present, use it verbatim. Otherwise glob ``/dev/cu.usbmodem*``: a single
    match is used automatically; zero or several raise a helpful error rather than guessing.
    ``exists``/``candidates`` are injectable for testing.
    """
    if exists(preferred):
        return preferred
    found = sorted(candidates() if candidates is not None else glob.glob(USBMODEM_GLOB))
    if len(found) == 1:
        return found[0]
    if not found:
        raise RuntimeError(
            f"Serial port {preferred!r} not found and no {USBMODEM_GLOB} device is connected. "
            "Plug in the arm, or run `dume find-port`."
        )
    raise RuntimeError(
        f"Serial port {preferred!r} not found and multiple candidates exist: {found}. "
        "Pass the right one with --port."
    )


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
    def relax(self) -> None: ...


class SO101Arm:
    """Real SO-101 follower via lerobot's ``SOFollower``."""

    name = "so101"

    def __init__(
        self,
        port: str,
        robot_id: str,
        *,
        disable_torque_on_disconnect: bool = True,
        gripper_servo_p: int | None = None,
    ):
        self.port = port
        self.robot_id = robot_id
        self._disable_torque = disable_torque_on_disconnect
        self._gripper_servo_p = gripper_servo_p
        self._robot = None

    def connect(self) -> None:
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots.so_follower.so_follower import SOFollower

        # Auto-resolve so a changed usbmodem suffix doesn't require editing the config.
        self.port = resolve_serial_port(self.port)
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
        # Bump the gripper's position-loop P after lerobot's configure() (which sets all motors to
        # 16) so the gripper tracks the trigger snappily. Gripper only; arm joints stay gentle.
        if self._gripper_servo_p is not None:
            self._robot.bus.write("P_Coefficient", "gripper", int(self._gripper_servo_p))

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

    def relax(self) -> None:
        """Cut motor torque so the arm can be moved by hand (e.g. to capture a pose)."""
        self._robot.bus.disable_torque()


class SimArm:
    """Kinematic simulation: adopts commanded joints immediately. Powers ``--dry-run``.

    ``servo_noise_deg`` injects zero-mean Gaussian noise into ``read_joints`` for the five arm
    joints (not the gripper), modelling the quantised/noisy feedback real Feetech servos report.
    Used to reproduce hardware teleop jitter offline and verify the controller's internal
    commanded-reference (``q_ref``) ignores it. Default 0.0 keeps the sim exact.
    """

    name = "sim"

    def __init__(self, initial_joints=None, *, servo_noise_deg: float = 0.0, seed: int = 0):
        self._joints = (
            np.array([0.0, -20.0, 20.0, 0.0, 0.0, 50.0], dtype=float)
            if initial_joints is None
            else np.asarray(initial_joints, dtype=float).copy()
        )
        self._connected = False
        self.servo_noise_deg = float(servo_noise_deg)
        self._rng = np.random.default_rng(seed)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_calibrated(self) -> bool:
        return True

    def read_joints(self) -> np.ndarray:
        q = self._joints.copy()
        if self.servo_noise_deg > 0.0:
            q[:5] += self._rng.normal(0.0, self.servo_noise_deg, size=5)
        return q

    def write_joints(self, joints) -> None:
        self._joints = np.asarray(joints, dtype=float).copy()

    def relax(self) -> None:
        """No-op in simulation — there's no torque to disable."""
