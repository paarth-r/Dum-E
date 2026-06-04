# SO-101 Inverse-Kinematics Controller — Design

**Date:** 2026-06-02
**Status:** Approved, implementing
**Hardware:** Single LeRobot SO-101 follower arm on `/dev/cu.usbmodem58FA0818281`, driven by a wireless Xbox controller.

## Goal

A smooth, intuitive end-effector controller for the SO-101: jog the gripper in Cartesian
space with the Xbox controller, or send it to absolute poses. Built as a reusable library
(`dume`) so other services can call `goto`/`follow_path`/`jog` directly — the CLI is just one
consumer.

## Approach

Use lerobot's proven surfaces for the hard parts and own the control loop:

- **Motor I/O:** lerobot `SO101Follower` (Feetech bus) — connect, read joint positions (deg),
  write goal positions, gripper.
- **Kinematics:** lerobot `RobotKinematics(urdf_path, target_frame_name="gripper_frame_link")`
  — `forward_kinematics(joints_deg) -> 4x4 pose`, `inverse_kinematics(cur_joints, target_pose,
  position_weight, orientation_weight) -> joints_deg`, backed by Placo. macOS arm64 wheels exist.
- **Our code:** a fixed-rate (50 Hz) control loop with input shaping, a plan-then-solve motion
  pipeline, smoothing, and safety clamps.

Rejected: lerobot's built-in `*FollowerEndEffector` teleop (SO-101 variant incomplete, hides the
loop we need to tune); the community `lerobot-kinematics` repo (extra dep, no upside).

## Plan-then-solve pipeline

Motion is never sent straight to IK. Every commanded motion goes:

```
goal/target EE pose
   -> Planner.plan(start_pose, goal_pose, constraints) -> Trajectory   (Cartesian path)
   -> sample Trajectory at loop rate -> EE pose at time t
   -> Kinematics.ik(current_joints, pose_t) -> joint goals
   -> slew-rate limit -> motors
```

This makes the path an explicit, inspectable object generated *before* solving — the seam where
future obstacle-avoidance or contact-compliant planners drop in by implementing the same
`Planner` interface. The default planner is a straight-line Cartesian path: linear interpolation
on position, SLERP on orientation, with trapezoidal time scaling (velocity/accel limits) for
smooth start/stop.

Velocity jog reuses the same EE→IK→slew tail but feeds it a continuously integrated moving
target rather than a precomputed trajectory (a path planner mid-jog would add latency); the
planner path is used for `goto`/pose moves.

## Modules (each: one purpose, well-defined interface, independently testable)

- `config.py` — `ControllerConfig` dataclass: loop rate, deadzones, expo, max linear/angular
  velocity, EMA factor, joint slew limit, IK weights, workspace bounding box, button map, port,
  URDF path, calibration id. One place to tune feel.
- `geometry.py` — pure pose helpers: build/decompose 4x4 transforms, position+quaternion <->
  matrix, SLERP, pose distance. No hardware. (scipy `Rotation`.)
- `kinematics.py` — `Kinematics` wraps lerobot `RobotKinematics`: `fk(joints)`, `ik(cur, pose)`.
- `planning.py` — `Planner` protocol + `Trajectory` (sample(t) -> pose, duration).
  `StraightLinePlanner` implements lerp/SLERP + trapezoidal timing.
- `arm.py` — `Arm`: thin SO101Follower wrapper. connect/disconnect, `read_joints()`,
  `write_joints(goal)`, gripper, context manager. The only module that touches hardware.
- `input_xbox.py` — `XboxController` (pygame): poll -> `Command` (lin_vel xyz, ang_vel rpy,
  gripper delta, mode/home/save/recall/stop flags). Deadzone + cubic expo applied here.
- `poses.py` — named pose store (Home + presets), JSON-backed, save/recall.
- `controller.py` — the 50 Hz loop tying it together; velocity + pose modes; smoothing + safety.
- `service.py` — `DumeArm` facade: the public, hardware-or-dryrun API other services import
  (`get_pose`, `jog(cmd)`, `goto(pose|xyzrpy)`, `follow_path(poses)`, `home`, context manager).
- `cli.py` — `find-port | calibrate | run | run --dry-run | goto ...`; a thin consumer of the above.

## Smoothness

- Stick: deadzone -> cubic expo -> scale to max linear (m/s) & angular (rad/s) velocity.
- EMA low-pass on velocity command (jitter); slew-rate limit on joint goals (IK snap).
- IK seeded from current joints each tick (continuity); low orientation weight by default so the
  arm tracks position without fighting itself (toggleable orientation lock).
- On start / mode switch: target pose := current measured pose (no lurch).

## Safety

- Workspace bounding box clamp on target XYZ; joint-limit + max-step-per-tick clamp; gripper
  limits. Both-bumpers / Stop = hold. Ctrl-C = safe disconnect. `--dry-run` prints commanded
  poses/joints without moving motors (validate IK + feel before live).

## Control mapping (Xbox, config-driven defaults)

- Left stick: gripper X / Y (base plane). Right stick Y: Z. Right stick X + bumpers: wrist
  roll / pitch. LT / RT: gripper close / open. A: toggle velocity/pose mode. Y: Home.
  X: save preset. B: recall preset. Start: re-zero target to current. Back: orientation lock.

## Pose mode

Pad recalls named poses (Home + saved presets). CLI `dume goto X Y Z ROLL PITCH YAW` sends an
exact absolute pose through the planner. Both use the same plan-then-solve pipeline.

## Testing

- `geometry`: matrix<->pos/quat round-trip, SLERP endpoints.
- `kinematics`: FK then IK returns ~original joints (real URDF, no hardware).
- `planning`: trajectory endpoints exact, monotonic timing, velocity/accel within limits,
  straight-line position interpolation.
- `input`: deadzone/expo math, flag edges.
- `controller`: dry-run loop over synthetic input — commanded joint steps bounded (continuous),
  targets stay in workspace + joint limits.
- Manual hardware bring-up checklist (pair controller, find-port, calibrate, dry-run, live).

## Repo layout

```
pyproject.toml      urdf/        tests/
src/dume/{config,geometry,kinematics,planning,arm,input_xbox,poses,controller,service,cli}.py
```
