"""All tunable parameters in one place: feel, safety limits, and the Xbox button map.

Other services can construct a :class:`ControllerConfig`, tweak fields, and hand it to
``DumeArm`` — nothing else needs editing to change behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from dume.kinematics import DEFAULT_EE_FRAME, DEFAULT_URDF

DEFAULT_PORT = "/dev/cu.usbmodem58FA0818281"


class ControlMode(str, Enum):
    VELOCITY = "velocity"  # Cartesian velocity jog from the sticks
    POSE = "pose"  # hold/track an absolute target pose (goto, presets)


@dataclass
class AxisBox:
    """An axis-aligned workspace box in the base frame (metres)."""

    x: tuple[float, float] = (0.06, 0.42)
    y: tuple[float, float] = (-0.34, 0.34)
    z: tuple[float, float] = (0.02, 0.42)

    def clamp(self, position):
        import numpy as np

        p = np.asarray(position, dtype=float).copy()
        p[0] = min(max(p[0], self.x[0]), self.x[1])
        p[1] = min(max(p[1], self.y[0]), self.y[1])
        p[2] = min(max(p[2], self.z[0]), self.z[1])
        return p


@dataclass
class XboxMap:
    """pygame joystick axis/button indices (SDL2 Xbox layout) and input shaping.

    Bindings used: sticks (position), triggers (gripper), A (mode toggle), and the D-pad
    (pitch/roll while held). The D-pad reports as buttons on this controller (0 hats), so
    these are button indices. Indices vary by OS/driver; run ``dume axes`` to verify.
    Trigger axes rest near -1 and read +1 fully pressed; we normalise to [0, 1].
    """

    axis_left_x: int = 0  # +right
    axis_left_y: int = 1  # +down (we invert)
    axis_right_x: int = 2  # +right
    axis_right_y: int = 3  # +down (we invert)
    axis_lt: int = 4  # left trigger
    axis_rt: int = 5  # right trigger

    btn_a: int = 0  # gripper -> full close (setpoint)
    btn_b: int = 1  # toggle velocity/pose (freeze) mode
    btn_y: int = 3  # gripper -> full open (setpoint)

    # D-pad (held): pitch = wrist_flex, roll = wrist_roll. SDL Xbox layout = buttons 11-14.
    btn_dpad_up: int = 11  # pitch up
    btn_dpad_down: int = 12  # pitch down
    btn_dpad_left: int = 13  # roll left
    btn_dpad_right: int = 14  # roll right

    deadzone: float = 0.08
    expo: float = 0.6  # 0 = linear, 1 = full cubic
    trigger_deadzone: float = 0.05


@dataclass
class ControllerConfig:
    # Hardware
    port: str = DEFAULT_PORT
    robot_id: str = "so101_follower"
    urdf_path: str = DEFAULT_URDF
    ee_frame: str = DEFAULT_EE_FRAME

    # Loop
    loop_hz: float = 50.0

    # Velocity jog feel
    max_linear_vel: float = 0.12  # m/s at full stick (positions the wrist pivot)
    wrist_speed: float = 80.0  # deg/s for D-pad wrist_flex (pitch) / wrist_roll (roll) jog
    max_angular_vel: float = 1.2  # rad/s — used by programmatic goto orientation only
    vel_ema_alpha: float = 0.35  # velocity low-pass (0..1, higher = snappier)

    # Smoothing / safety
    joint_slew_deg: float = 6.0  # max commanded joint change per tick (per joint, velocity cap)
    joint_jerk_deg: float = 2.0  # max change in per-tick joint velocity (accel cap; smooths reversals)
    workspace: AxisBox = field(default_factory=AxisBox)

    # Velocity-jog position IK: damped least-squares over pan/lift/elbow (smoother + faster than
    # the general placo solver, with explicit singularity damping). Higher damping = steadier
    # near singularities, slower tracking.
    dls_damping: float = 0.05

    # IK weights. The SO-101 is a 5-DOF arm, so it cannot satisfy an arbitrary 6-DOF pose;
    # control is position-led with best-effort orientation.
    ik_position_weight: float = 1.0
    ik_orientation_weight: float = 0.02  # velocity jog: wrist responds, doesn't wander
    ik_goto_orientation_weight: float = 0.0  # goto/pose: exact position, natural orientation
    ik_orientation_lock_weight: float = 1.0  # when orientation lock / honor_orientation is on

    # Gripper (the gripper motor is normalised 0..100, not degrees)
    gripper_open: float = 95.0
    gripper_closed: float = 5.0
    gripper_speed: float = 200.0  # units/s (0..100 scale) at full trigger

    # Planner (goto / pose moves)
    plan_max_linear_vel: float = 0.10  # m/s
    plan_max_linear_acc: float = 0.30  # m/s^2
    plan_max_angular_vel: float = 1.0  # rad/s
    plan_max_angular_acc: float = 3.0  # rad/s^2

    # Input
    xbox: XboxMap = field(default_factory=XboxMap)

    @property
    def dt(self) -> float:
        return 1.0 / self.loop_hz
