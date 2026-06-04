"""The control core: turns a stream of :class:`Command`s into smooth, safe joint motion.

``step()`` advances exactly one tick and is pure with respect to time (no sleeping), so it can
be driven by the real loop, by ``--dry-run`` over a ``SimArm``, or by tests with synthetic
input. ``run()`` wraps ``step`` in a fixed-rate loop with safe shutdown.

Velocity jog (live): sticks move the wrist pivot via position-only IK over pan/lift/elbow,
the D-pad jogs wrist_flex/wrist_roll directly, triggers drive the gripper — fully decoupled,
so orientation never fights position on this 5-DOF arm.

Pose mode (goto / programmatic): the *plan-then-solve* path — plan a Cartesian trajectory,
sample it, full-pose IK seeded from measured joints, slew-limit, clamp, send.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dume import geometry as g
from dume.arm import ArmIO
from dume.config import ControllerConfig, ControlMode
from dume.input_xbox import Command
from dume.kinematics import Kinematics
from dume.planning import StraightLinePlanner, Trajectory
from dume.poses import PoseStore


@dataclass
class Telemetry:
    mode: ControlMode
    target_xyzrpy: np.ndarray
    joints_sent: np.ndarray
    gripper: float
    orientation_lock: bool
    tracking_pos_err_mm: float
    trajectory_active: bool


@dataclass
class Controller:
    config: ControllerConfig
    arm: ArmIO
    kin: Kinematics
    poses: PoseStore = field(default_factory=PoseStore)

    def __post_init__(self):
        c = self.config
        self.planner = StraightLinePlanner(
            max_linear_vel=c.plan_max_linear_vel,
            max_linear_acc=c.plan_max_linear_acc,
            max_angular_vel=c.plan_max_angular_vel,
            max_angular_acc=c.plan_max_angular_acc,
        )
        # Position kinematics: pan/lift/elbow position the WRIST PIVOT (wrist_link origin,
        # invariant to wrist_flex/roll). The wrist joints are then jogged directly, so the
        # D-pad pitches/rolls the gripper without the IK fighting to hold the tip.
        self.kin_pos = Kinematics(
            self.config.urdf_path,
            ee_frame="wrist_link",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex"],
        )
        self.joint_limits = self.kin.joint_limits_deg()
        self.mode = ControlMode.VELOCITY
        self.orientation_lock = False
        self._filt_lin = np.zeros(3)
        self._traj: Trajectory | None = None
        self._traj_t = 0.0
        self._pose_ori_w = self.config.ik_goto_orientation_weight
        self._joint_target: np.ndarray | None = None  # active joint-space move (e.g. Home)
        self.home_joints: np.ndarray | None = None  # neutral config, captured at start()
        self._pivot_target: np.ndarray | None = None  # integrated wrist-pivot position (m)
        self.wrist_flex_cmd = 0.0  # user-owned pitch joint (deg)
        self.wrist_roll_cmd = 0.0  # user-owned roll joint (deg)
        self.target_pose: np.ndarray | None = None
        self.gripper_cmd = 0.0

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self.arm.connect()
        self._sync_to_joints(self.arm.read_joints())
        # Home == the configuration the arm is in at startup (the neutral pose).
        self.home_joints = self.arm.read_joints().copy()

    def _sync_to_joints(self, q: np.ndarray) -> None:
        """Re-seat all targets from a measured joint vector (start / mode switch / re-zero)."""
        self._pivot_target = self.kin_pos.fk(q[:3])[:3, 3].copy()
        self.wrist_flex_cmd = float(q[3])
        self.wrist_roll_cmd = float(q[4])
        self.gripper_cmd = float(q[5])
        self._filt_lin = np.zeros(3)
        self.target_pose = self.kin.fk(q)

    def stop(self) -> None:
        self.arm.disconnect()

    # ---- one tick ----------------------------------------------------------
    def step(self, cmd: Command, dt: float | None = None) -> Telemetry:
        dt = self.config.dt if dt is None else dt
        q_cur = self.arm.read_joints()
        if self._pivot_target is None:
            self._sync_to_joints(q_cur)

        self._handle_buttons(cmd, q_cur)

        # Gripper integrates independently of mode.
        self.gripper_cmd = float(
            np.clip(
                self.gripper_cmd + cmd.gripper * self.config.gripper_speed * dt,
                self.config.gripper_closed,
                self.config.gripper_open,
            )
        )

        if self._joint_target is not None:
            # Joint-space move (Home): drive joints straight to the target, no IK, so it
            # returns to the exact neutral configuration instead of an IK-chosen one.
            q_send = self._advance_joint_move(q_cur)
        elif self.mode is ControlMode.VELOCITY:
            q_send = self._velocity_jog(cmd, q_cur, dt)
        else:  # POSE: track a planned trajectory (goto), else hold (freeze)
            self._advance_trajectory(dt)
            q_send = self._solve_and_limit(q_cur)
        self.arm.write_joints(q_send)

        achieved = self.kin.fk(q_send)
        err_mm = float(np.linalg.norm(g.position_of(achieved) - g.position_of(self.target_pose)) * 1000)
        return Telemetry(
            mode=self.mode,
            target_xyzrpy=g.pose_to_xyzrpy(self.target_pose),
            joints_sent=q_send,
            gripper=self.gripper_cmd,
            orientation_lock=self.orientation_lock,
            tracking_pos_err_mm=err_mm,
            trajectory_active=self._traj is not None or self._joint_target is not None,
        )

    # ---- pieces ------------------------------------------------------------
    def _handle_buttons(self, cmd: Command, q_cur: np.ndarray) -> None:
        if cmd.toggle_mode:
            self.mode = ControlMode.POSE if self.mode is ControlMode.VELOCITY else ControlMode.VELOCITY
            self._traj = None
            self._sync_to_joints(q_cur)  # re-seat targets so neither mode lurches

    def _advance_joint_move(self, q_cur: np.ndarray) -> np.ndarray:
        """Slew the 5 arm joints straight toward ``self._joint_target``; clear when arrived."""
        target = self._joint_target
        delta = np.clip(target[:5] - q_cur[:5], -self.config.joint_slew_deg, self.config.joint_slew_deg)
        q_send = q_cur.copy().astype(float)
        q_send[:5] = np.clip(q_cur[:5] + delta, self.joint_limits[:5, 0], self.joint_limits[:5, 1])
        q_send[5] = self.gripper_cmd
        if np.max(np.abs(target[:5] - q_send[:5])) < 0.5:  # arrived (deg)
            self._joint_target = None
            self.target_pose = self.kin.fk(q_send)  # resync so velocity jog resumes cleanly
        return q_send

    def _goto_pose(
        self, goal_pose: np.ndarray, q_cur: np.ndarray, orientation_weight: float | None = None
    ) -> None:
        """Switch to pose mode and plan a trajectory from the current pose to ``goal_pose``.

        ``orientation_weight`` controls how hard IK honours the requested orientation during
        the move; ``None`` uses the position-priority default (``ik_goto_orientation_weight``).
        """
        start = self.kin.fk(q_cur)
        self.mode = ControlMode.POSE
        self._traj = self.planner.plan(start, goal_pose)
        self._traj_t = 0.0
        self._pose_ori_w = (
            self.config.ik_goto_orientation_weight if orientation_weight is None else orientation_weight
        )
        self.target_pose = self._traj.sample(0.0)

    def _velocity_jog(self, cmd: Command, q_cur: np.ndarray, dt: float) -> np.ndarray:
        """Live jog: sticks move the wrist pivot (IK over pan/lift/elbow); D-pad jogs the
        wrist joints directly. Fully decoupled, so pitch/roll never fight position."""
        c = self.config
        # Wrist pivot position: EMA-smoothed velocity integrated into the target, clamped.
        desired_lin = cmd.lin * c.max_linear_vel
        a = c.vel_ema_alpha
        self._filt_lin = a * desired_lin + (1 - a) * self._filt_lin
        self._pivot_target = c.workspace.clamp(self._pivot_target + self._filt_lin * dt)
        pivot_pose = g.make_transform(self._pivot_target, np.eye(3))
        q_pos = self.kin_pos.ik(q_cur[:3], pivot_pose, orientation_weight=0.0)  # position-only

        # Wrist joints jogged directly (D-pad), rate-limited by wrist_speed, clamped to limits.
        self.wrist_flex_cmd = float(
            np.clip(
                self.wrist_flex_cmd + cmd.wrist_pitch * c.wrist_speed * dt,
                self.joint_limits[3, 0],
                self.joint_limits[3, 1],
            )
        )
        self.wrist_roll_cmd = float(
            np.clip(
                self.wrist_roll_cmd + cmd.wrist_roll * c.wrist_speed * dt,
                self.joint_limits[4, 0],
                self.joint_limits[4, 1],
            )
        )

        q_send = q_cur.copy().astype(float)
        delta = np.clip(q_pos[:3] - q_cur[:3], -c.joint_slew_deg, c.joint_slew_deg)
        q_send[:3] = np.clip(q_cur[:3] + delta, self.joint_limits[:3, 0], self.joint_limits[:3, 1])
        q_send[3] = self.wrist_flex_cmd
        q_send[4] = self.wrist_roll_cmd
        q_send[5] = self.gripper_cmd
        self.target_pose = self.kin.fk(q_send)  # tip pose, for telemetry
        return q_send

    def _advance_trajectory(self, dt: float) -> None:
        if self._traj is None:
            return  # holding current target
        self._traj_t += dt
        self.target_pose = self._traj.sample(self._traj_t)
        if self._traj_t >= self._traj.duration:
            self.target_pose = self._traj.sample(self._traj.duration)
            self._traj = None  # arrived; hold

    def _solve_and_limit(self, q_cur: np.ndarray) -> np.ndarray:
        c = self.config
        if self.orientation_lock:
            ori_w = c.ik_orientation_lock_weight
        elif self.mode is ControlMode.POSE:
            ori_w = self._pose_ori_w  # position-priority by default for goto/pose moves
        else:
            ori_w = c.ik_orientation_weight
        q_goal = self.kin.ik(
            q_cur, self.target_pose, position_weight=c.ik_position_weight, orientation_weight=ori_w
        )
        q_send = q_cur.copy().astype(float)
        # Slew-rate limit the 5 arm joints (continuity); gripper is already rate-limited.
        delta = np.clip(q_goal[:5] - q_cur[:5], -c.joint_slew_deg, c.joint_slew_deg)
        q_send[:5] = q_cur[:5] + delta
        q_send[5] = self.gripper_cmd
        # Joint-limit clamp for the arm joints.
        q_send[:5] = np.clip(q_send[:5], self.joint_limits[:5, 0], self.joint_limits[:5, 1])
        return q_send

    # ---- real-time loop ----------------------------------------------------
    def run(self, poll, on_tick=None) -> None:
        """Fixed-rate loop. ``poll`` returns a Command each tick; ``on_tick`` gets Telemetry."""
        period = self.config.dt
        try:
            while True:
                t0 = time.perf_counter()
                tel = self.step(poll())
                if on_tick is not None:
                    on_tick(tel)
                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
