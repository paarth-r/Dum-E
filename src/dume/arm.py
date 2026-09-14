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
import logging
import os
from typing import Callable, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

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


def assert_zero_drive_modes(calibration) -> None:
    """Refuse a calibration that would silently invert a joint's reported load.

    lerobot applies ``drive_mode`` inversion inside ``_normalize``, which never runs for
    ``Present_Load`` (the register is raw). Every motor is ``drive_mode: 0`` today, so servo load
    sign matches the URDF joint direction; a future recalibration that sets 1 on any motor would
    flip that joint's force sign with no error anywhere. Fail loudly at connect instead.
    """
    inverted = [m for m, c in calibration.items() if int(getattr(c, "drive_mode", 0)) != 0]
    if inverted:
        raise RuntimeError(
            f"Motors {inverted} have drive_mode != 0; force sensing assumes drive_mode 0 on every "
            "motor (Present_Load is read raw and would be sign-inverted). Recalibrate with "
            "`dume calibrate` without inverting, or teach dume.arm.read_loads about drive_mode."
        )


@runtime_checkable
class ArmIO(Protocol):
    name: str

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_calibrated(self) -> bool: ...
    def read_joints(self) -> np.ndarray: ...
    def read_loads(self) -> np.ndarray: ...
    def write_joints(self, joints) -> None: ...
    def relax(self) -> None: ...
    def engage(self) -> None: ...
    def relax_gripper(self) -> None: ...
    def engage_gripper(self) -> None: ...


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
        assert_zero_drive_modes(self._robot.calibration)
        # Bump the gripper's position-loop P after lerobot's configure() (which sets all motors to
        # 16) so the gripper tracks the trigger snappily. Gripper only; arm joints stay gentle.
        if self._gripper_servo_p is not None:
            self._robot.bus.write("P_Coefficient", "gripper", int(self._gripper_servo_p))
        # Clear the gripper's profile-velocity cap (0 = move at full speed). lerobot never writes
        # this register, so whatever value is stored in the servo silently throttles the jaw; the
        # SQUEEZE command path is an absolute 1:1 trigger map, so any lag here is pure servo.
        self._robot.bus.write("Goal_Velocity", "gripper", 0)

    def disconnect(self) -> None:
        """Release the robot, never raising over the top of a live exception.

        lerobot's ``disconnect`` writes ``Torque_Enable = 0`` to every motor, which fails if
        the bus has already gone dark — exactly the situation a fault leaves behind. Letting
        that propagate replaces the *real* exception with a ConnectionError from the shutdown
        path, which is how a genuine fault was lost on 2026-07-25. The failure is logged and
        swallowed, and ``_robot`` is cleared either way so a retry can't reuse a dead handle.
        """
        robot, self._robot = self._robot, None
        if robot is None:
            return
        try:
            robot.disconnect()
        except Exception as exc:  # noqa: BLE001 — cleanup must not mask the original error
            logger.warning("Arm disconnect failed (torque may still be enabled): %s", exc)

    def is_calibrated(self) -> bool:
        return bool(self._robot and self._robot.is_calibrated)

    def read_joints(self) -> np.ndarray:
        obs = self._robot.get_observation()
        return np.array([obs[f"{m}.pos"] for m in MOTOR_ORDER], dtype=float)

    def read_loads(self) -> np.ndarray:
        """Per-motor ``Present_Load`` in raw signed servo units (~+-1000 full scale), MOTOR_ORDER.

        This is the PWM duty the servo's position loop is applying; at zero velocity it is
        proportional to torque. Read raw (``normalize=False``): the register is absent from
        lerobot's ``normalized_data`` and its 10-bit sign-magnitude encoding is already decoded
        by the bus. Meaningless while torque is off (reads 0).
        """
        vals = self._robot.bus.sync_read("Present_Load", MOTOR_ORDER, normalize=False)
        return np.array([vals[m] for m in MOTOR_ORDER], dtype=float)

    def read_voltage(self) -> float:
        """Supply rail in volts (``Present_Voltage`` is in 0.1 V units), read from motor 1."""
        vals = self._robot.bus.sync_read("Present_Voltage", MOTOR_ORDER[:1], normalize=False)
        return float(vals[MOTOR_ORDER[0]]) / 10.0

    def write_joints(self, joints) -> None:
        joints = np.asarray(joints, dtype=float)
        action = {f"{m}.pos": float(joints[i]) for i, m in enumerate(MOTOR_ORDER)}
        self._robot.send_action(action)

    def relax(self) -> None:
        """Cut motor torque so the arm can be moved by hand (e.g. to capture a pose)."""
        self._robot.bus.disable_torque()

    def engage(self) -> None:
        """Restore motor torque after ``relax`` (the arm holds wherever it is)."""
        self._robot.bus.enable_torque()

    def relax_gripper(self) -> None:
        """Cut torque on the gripper motor only — the jaw goes limp, the arm keeps holding.
        Goal positions streamed while torque is off are ignored by the servo."""
        self._robot.bus.disable_torque("gripper")

    def engage_gripper(self) -> None:
        """Restore gripper torque; the jaw resumes chasing the streamed goal position."""
        self._robot.bus.enable_torque("gripper")


class SimArm:
    """Kinematic simulation: adopts commanded joints immediately. Powers ``--dry-run``.

    ``servo_noise_deg`` injects zero-mean Gaussian noise into ``read_joints`` for the five arm
    joints (not the gripper), modelling the quantised/noisy feedback real Feetech servos report.
    Used to reproduce hardware teleop jitter offline and verify the controller's internal
    commanded-reference (``q_ref``) ignores it. Default 0.0 keeps the sim exact.

    ``load_source`` (joints -> length-6 array) synthesises ``read_loads`` so the force estimator
    and grasp logic run under ``--dry-run``; default is all zeros (no load, like torque off).
    """

    name = "sim"

    def __init__(
        self,
        initial_joints=None,
        *,
        servo_noise_deg: float = 0.0,
        seed: int = 0,
        load_source: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self._joints = (
            np.array([0.0, -20.0, 20.0, 0.0, 0.0, 50.0], dtype=float)
            if initial_joints is None
            else np.asarray(initial_joints, dtype=float).copy()
        )
        self._connected = False
        self.servo_noise_deg = float(servo_noise_deg)
        self._rng = np.random.default_rng(seed)
        self._load_source = load_source

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

    def read_loads(self) -> np.ndarray:
        if self._load_source is None:
            return np.zeros(6)
        return np.asarray(self._load_source(self._joints.copy()), dtype=float)

    def write_joints(self, joints) -> None:
        self._joints = np.asarray(joints, dtype=float).copy()

    def engage(self) -> None:
        """No-op in simulation — there's no torque to restore."""

    def relax(self) -> None:
        """No-op in simulation — there's no torque to disable."""

    def relax_gripper(self) -> None:
        """No-op in simulation."""

    def engage_gripper(self) -> None:
        """No-op in simulation."""
