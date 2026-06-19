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
from dume.config import ControllerConfig, ControlMode, GripperMode
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
    min_joint_margin_deg: float  # smallest distance from any arm joint to its limit
    margin_joint: str  # which joint owns that margin
    near_singular: bool  # position manipulability below the configured floor


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
        self.gripper_mode = self.config.gripper_mode_default
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
        # Internal *commanded* joint reference. Control integrates and slews from this, NOT from
        # the measured joints, so noisy servo feedback never leaks into the command stream (the
        # root cause of teleop jitter). Re-synced to measured only on start / mode-switch / re-zero.
        self.q_ref: np.ndarray | None = None
        self._prev_vel = np.zeros(6)  # last commanded per-tick joint velocity (deg/tick), for jerk limit

    # ---- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self.arm.connect()
        self._sync_to_joints(self.arm.read_joints())
        # Home == the configuration the arm is in at startup (the neutral pose).
        self.home_joints = self.arm.read_joints().copy()

    def _sync_to_joints(self, q: np.ndarray) -> None:
        """Re-seat all targets from a measured joint vector (start / mode switch / re-zero)."""
        q = np.asarray(q, dtype=float)
        self._pivot_target = self.kin_pos.fk(q[:3])[:3, 3].copy()
        self.wrist_flex_cmd = float(q[3])
        self.wrist_roll_cmd = float(q[4])
        self.gripper_cmd = float(q[5])
        self._filt_lin = np.zeros(3)
        self.target_pose = self.kin.fk(q)
        self.q_ref = q.copy()  # commanded reference starts at the measured config
        self._prev_vel = np.zeros(6)

    def stop(self) -> None:
        self.arm.disconnect()

    # ---- one tick ----------------------------------------------------------
    def step(self, cmd: Command, dt: float | None = None) -> Telemetry:
        dt = self.config.dt if dt is None else dt
        q_cur = self.arm.read_joints()
        if self._pivot_target is None or self.q_ref is None:
            self._sync_to_joints(q_cur)

        self._handle_buttons(cmd, q_cur)

        # Gripper, interpreted per gripper_mode (independent of the arm motion mode). Skipped
        # during a scripted joint move (home / start / goto_joints), which commands its own
        # gripper value and must not be fought by the triggers mid-move.
        c = self.config
        if self._joint_target is None:
            if self.gripper_mode is GripperMode.SQUEEZE:
                # RT position IS the openness: released -> closed (default), squeezed -> open.
                self.gripper_cmd = c.gripper_closed + cmd.rt * (c.gripper_open - c.gripper_closed)
            else:  # RATE: LT opens, RT closes, integrated.
                self.gripper_cmd = float(
                    np.clip(
                        self.gripper_cmd + (cmd.lt - cmd.rt) * c.gripper_speed * dt,
                        c.gripper_closed,
                        c.gripper_open,
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
        self.q_ref = q_send.copy()  # advance the commanded reference (open-loop, noise-free)

        achieved = self.kin.fk(q_send)
        err_mm = float(np.linalg.norm(g.position_of(achieved) - g.position_of(self.target_pose)) * 1000)
        margins = np.minimum(
            q_send[:5] - self.joint_limits[:5, 0], self.joint_limits[:5, 1] - q_send[:5]
        )
        j = int(np.argmin(margins))
        manip = self.kin_pos.manipulability(q_send[:3])
        return Telemetry(
            mode=self.mode,
            target_xyzrpy=g.pose_to_xyzrpy(self.target_pose),
            joints_sent=q_send,
            gripper=self.gripper_cmd,
            orientation_lock=self.orientation_lock,
            tracking_pos_err_mm=err_mm,
            trajectory_active=self._traj is not None or self._joint_target is not None,
            min_joint_margin_deg=float(margins[j]),
            margin_joint=self.kin.joint_names[j],
            near_singular=manip < self.config.manipulability_floor,
        )

    # ---- pieces ------------------------------------------------------------
    def _handle_buttons(self, cmd: Command, q_cur: np.ndarray) -> None:
        if cmd.toggle_mode:
            self.mode = ControlMode.POSE if self.mode is ControlMode.VELOCITY else ControlMode.VELOCITY
            self._traj = None
            self._sync_to_joints(q_cur)  # re-seat targets so neither mode lurches
        if cmd.gripper_mode_toggle:
            self.gripper_mode = (
                GripperMode.RATE if self.gripper_mode is GripperMode.SQUEEZE else GripperMode.SQUEEZE
            )

    def _advance_joint_move(self, q_cur: np.ndarray) -> np.ndarray:
        """Slew the 5 arm joints straight toward ``self._joint_target``; clear when arrived."""
        target = self._joint_target
        ref = self.q_ref  # slew from the commanded reference, not noisy feedback
        delta = np.clip(target[:5] - ref[:5], -self.config.joint_slew_deg, self.config.joint_slew_deg)
        q_send = ref.copy().astype(float)
        q_send[:5] = np.clip(ref[:5] + delta, self.joint_limits[:5, 0], self.joint_limits[:5, 1])
        q_send[5] = self.gripper_cmd
        if np.max(np.abs(target[:5] - q_send[:5])) < 0.5:  # arrived (deg)
            self._joint_target = None
            # Re-seat ALL velocity-jog state (pivot/wrist/filters), not just target_pose, or the
            # next zero-command tick would jog back toward the pre-move pose's stale pivot target.
            self._sync_to_joints(q_send)
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
        ref = self.q_ref  # slew/seed from the commanded reference, never the noisy measurement
        # Wrist pivot position: EMA-smoothed velocity integrated into the target, clamped to the
        # workspace, then leashed to the achieved pivot so it can't wind up far past reach.
        desired_lin = cmd.lin * c.max_linear_vel
        a = c.vel_ema_alpha
        self._filt_lin = a * desired_lin + (1 - a) * self._filt_lin
        candidate = c.workspace.clamp(self._pivot_target + self._filt_lin * dt)
        achieved_pivot = self.kin_pos.fk(ref[:3])[:3, 3]  # where the arm actually is now
        lead = candidate - achieved_pivot
        dist = float(np.linalg.norm(lead))
        if dist > c.pivot_leash_m:  # anti-lockout: bound how far the target leads reality
            candidate = achieved_pivot + lead * (c.pivot_leash_m / dist)
        self._pivot_target = candidate
        # Damped least-squares position IK over pan/lift/elbow, seeded from the reference.
        q_pos = self.kin_pos.ik_position_dls(ref[:3], self._pivot_target, damping=c.dls_damping)

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

        # Desired per-tick joint deltas (arm joints 0..4) relative to the reference.
        delta = np.zeros(6)
        delta[:3] = q_pos[:3] - ref[:3]
        delta[3] = self.wrist_flex_cmd - ref[3]
        delta[4] = self.wrist_roll_cmd - ref[4]
        delta[:5] = self._limit_velocity_and_jerk(delta[:5])

        q_send = ref.copy().astype(float)
        q_send[:5] = np.clip(ref[:5] + delta[:5], self.joint_limits[:5, 0], self.joint_limits[:5, 1])
        q_send[5] = self.gripper_cmd
        self.target_pose = self.kin.fk(q_send)  # tip pose, for telemetry
        return q_send

    def _limit_velocity_and_jerk(self, delta5: np.ndarray) -> np.ndarray:
        """Clamp the 5 arm-joint per-tick deltas to the slew (velocity) cap, then limit how fast
        that velocity may change (jerk cap). Limiting acceleration is what kills the buzz from
        instant direction reversals. Updates the stored previous velocity."""
        c = self.config
        v = np.clip(delta5, -c.joint_slew_deg, c.joint_slew_deg)
        dv = np.clip(v - self._prev_vel[:5], -c.joint_jerk_deg, c.joint_jerk_deg)
        v = self._prev_vel[:5] + dv
        self._prev_vel[:5] = v
        return v

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
        ref = self.q_ref  # seed + slew from the commanded reference, not noisy feedback
        q_goal = self.kin.ik(
            ref, self.target_pose, position_weight=c.ik_position_weight, orientation_weight=ori_w
        )
        q_send = ref.copy().astype(float)
        # Slew-rate limit the 5 arm joints (continuity); gripper is already rate-limited.
        delta = np.clip(q_goal[:5] - ref[:5], -c.joint_slew_deg, c.joint_slew_deg)
        q_send[:5] = ref[:5] + delta
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
