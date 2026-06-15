# dum-e Perception + Learning Scaffolding + Interactive Sim — Design

**Date:** 2026-06-15
**Status:** Approved, implementing
**Builds on:** `2026-06-02-so101-ik-controller-design.md` (the IK controller this scaffolds on top of).

## Goal

Lay down *real-where-possible* scaffolding for three future subsystems — flown-extrinsics
depth/grasp, episode recording, and a diffusion learning policy — plus an **interactive,
PyBullet-rendered, Xbox-driven simulator** that doubles as a teleop-feel test rig. Everything
that does not strictly require physical hardware or recorded data is implemented for real and
unit-tested against `SimArm` today. The hardware/data-bound pieces are explicit stubs with
clear "needs hardware/data" errors, ready to fill in when the arm and demonstrations arrive.

This is deliberately a *foundation* commit: define the module boundaries, data contracts, and
the sim harness now, while the design is fresh, so future work writes implementations against
fixed interfaces instead of re-deriving them.

### Context & motivating vision

The long-term aim is a generalizable, precise manipulation policy in the spirit of TidyBot++
(Wu, Bohg, et al.): the policy is the easy part — the moat is high-quality demonstration data
and a teleop interface good enough to collect it. dum-e's existing Xbox+IK controller *is* that
interface. The flown-extrinsics grasp idea is a separate, classical-geometry subsystem that
leans on dum-e's superpower (it already knows where its own hand is via FK), so it needs no
calibrated stereo rig. These are independent subsystems sharing one foundation; this spec
scaffolds all of them plus the sim they will be developed and tested in.

## Non-goals (deferred to later specs)

- Real Arducam image capture (`ArduCamSource`).
- Real diffusion model training / weight loading.
- `T_cam_mount` extrinsic calibration on hardware (a best-guess constant ships now).
- "Pre-cached diffusion path for known geometry → zero-shot grasp" — parked as a research
  question; revisit once recording + flown-stereo + policy exist.
- Rendered-camera realism tuning, domain randomization, sim-to-real transfer.

## Decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Scaffolding depth | Real where possible, stub only hardware/data | Farms green squares now, testable in sim |
| Episode storage | Custom `dume` interface + real `LocalBackend` | Decoupled from lerobot in case the arm/framework changes |
| Learning policy | lerobot `DiffusionPolicy` via an adapter, for now | Use lerobot today; swapping later is a localized change |
| Sim camera fidelity | Geometric detections + depth | Deterministic, unit-testable; realized via PyBullet buffers |
| Viewer backend | PyBullet (interactive GUI) | Turnkey URDF load + free RGB-D + segmentation; physics not used for control |
| Kinematics authority | placo (unchanged) | PyBullet is renderer/world only; arm set via `resetJointState` |

## Architecture overview

```
Xbox  ->  Controller  ->  SimArm (ArmIO)
                |              |
                |          SimRenderer (PyBullet GUI: resetJointState each tick)
                |              |
                |          SimScene (objects as PyBullet bodies)
                |              |
                v          SimCamera (CameraSource: getCameraImage @ FK pose)
          EpisodeRecorder <----+----> flown_stereo (two-view -> 3D -> grasp)
                |
          DatasetBackend (LocalBackend now; to_lerobot() adapter later)
                |
          Policy (ScriptedPolicy now; LeRobotDiffusionPolicy stub)
```

The arm is driven kinematically: the `Controller` and `XboxInput` from the existing IK
controller are reused unchanged in their roles; PyBullet only mirrors the commanded joint
state for display and renders the camera. placo remains the FK/IK authority.

## Modules (extend the existing flat `dume` package)

### `camera.py` — pure types, placo-only
- `CameraIntrinsics`: `fx, fy, cx, cy, width, height` (distortion deferred).
- `CameraFrame`: `rgb: np.ndarray|None`, `depth: np.ndarray|None`, `t: float`, `pose: 4x4`.
- `Detections`: `ids: list[int]`, `pixels: np.ndarray (N,2)`, optional `depths: np.ndarray (N,)`.
- `CameraSource` Protocol: `capture() -> CameraFrame`, `detect() -> Detections`.
- `T_cam_mount`: config constant, transform from EE/gripper frame to camera optical frame
  (best-guess now; calibrate on hardware later).
- `camera_pose_from_fk(kin, joints) -> 4x4`: `kin.fk(joints) @ T_cam_mount`. **The core trick.**

### `sim_world.py` — PyBullet-backed sim + camera (real)
- `SimScene`: declarative object specs (shape primitive or mesh, world pose, id); loads them as
  PyBullet bodies. Static by default; physics not stepped for the arm.
- `SimRenderer`: connects PyBullet (GUI or DIRECT), loads the SO-101 URDF, `set_joints(q)` via
  `resetJointState` each tick. Exposes the loaded scene.
- `SimCamera` (implements `CameraSource`): given a `camera_pose_from_fk` result, calls
  `getCameraImage` → RGB + depth buffer (ground-truth depth) + segmentation buffer
  (per-object masks → `Detections` centroids). Deterministic.

### `flown_stereo.py` — flown-extrinsics depth + grasp (real)
- `relative_pose(T_world_camA, T_world_camB) -> 4x4`: baseline from FK, no rig calibration.
- `triangulate(detsA, detsB, T_A, T_B, K) -> points_3d`: two-view triangulation of matched
  detections using known camera poses.
- `propose_grasp(points_3d) -> grasp_pose (4x4)`: heuristic grasp pose from object geometry.
- Execution reuses the existing `goto` / plan-then-solve path (real, runs in `SimArm`).

### `dataset.py` — episode recording (real, storage decoupled)
- `Observation`: `joints (6)`, `ee_pose (4x4)`, `detections: Detections|None`,
  `depth: np.ndarray|None`, `image: np.ndarray|None` (None in geometric sim), `t: float`.
  Extensible named channels so hardware adds RGB without breaking the schema.
- `Action`: `joints_target (6)` (+ gripper) — matches `ArmIO.write_joints`; no new control path.
- `Step`: `(observation, action)`. `Episode`: ordered steps + metadata.
- `EpisodeRecorder`: assembles `Step`s from `ArmIO` + `CameraSource` during the control loop.
- `DatasetBackend` Protocol: `write(episode)`, `read(id) -> Episode`, `list()`.
- `LocalBackend` (real): episodes on disk (npz/parquet arrays + JSON sidecar metadata).
- `to_lerobot(episode_or_dataset)`: adapter that exports to lerobot's training format. Keeps
  storage ours while letting lerobot's trainer consume the data later.

### `policy.py` — learning interface (interface real, training stubbed)
- `Policy` Protocol: `select_action(observation) -> Action`, `load(path)`, `train(dataset)`.
- `ScriptedPolicy` (real): deterministic action generator used to drive the controller in tests.
- `LeRobotDiffusionPolicy` (adapter): wraps lerobot `DiffusionPolicy`. `train()`/`load()` raise
  `NotImplementedError("needs recorded data + hardware")` — the one genuinely data-bound stub.

### `controller.py` / `config.py` — control smoothing rework (real)
From the IK-jitter diagnosis: the loop chases noisy measured joints (`read_joints()` is used as
both IK seed and slew base), which on hardware injects servo noise into commands.
- Maintain an internal **commanded** joint reference `q_ref`; integrate and slew from `q_ref`,
  re-syncing to measured joints only on start / mode-switch / re-zero.
- Replace the generic placo IK in `_velocity_jog` with a **damped least-squares (DLS)** position
  Jacobian solver for the 3-DOF wrist-pivot (smoother, faster, explicit singularity damping).
- Add a **jerk limit** (acceleration clamp) on top of the existing velocity/slew clamp.
- `SimArm` gains optional synthetic servo-noise injection to reproduce hardware jitter offline
  and prove the fix without the robot.
- New config knobs: `dls_damping`, `joint_jerk_deg`, `sim_servo_noise_deg` (default 0).

### `cli.py` — `dume sim` (real)
New subcommand: build `Controller(arm=SimArm)` + `XboxInput` + `SimRenderer`, run the existing
fixed-rate loop, render each tick via the existing `on_tick` hook. Headed and drivable today.
Optional flags to spawn a scene and enable the camera/recorder.

## Real vs stubbed

**Real + unit-tested in sim now:** sim render/scene/camera (projection + depth via PyBullet),
extrinsics + triangulation + grasp proposal + execution-via-goto, dataset record→write→read,
policy interface + `ScriptedPolicy`, control-smoothing rework, `dume sim` harness.

**Stubbed (hardware/data only), each raising an explicit error and `@pytest.mark.skip(reason="hardware")`:**
`ArduCamSource` real capture, `LeRobotDiffusionPolicy.train/load`, `T_cam_mount` calibrated value.

## Testing

- `test_camera.py` — `camera_pose_from_fk` correctness; intrinsics/projection round-trip.
- `test_sim_world.py` — scene loads; `SimCamera` depth matches known object distance; segmentation
  → detections centroids within tolerance. (PyBullet in DIRECT mode for headless CI.)
- `test_flown_stereo.py` — place a known object, snapshot from two FK poses, triangulate →
  recovered 3D within tolerance; `relative_pose` correctness.
- `test_dataset.py` — record a synthetic episode, write, read back, assert fidelity; backend
  round-trip.
- `test_policy.py` — `ScriptedPolicy` drives the controller in sim for N steps; recorder captures
  a matching episode.
- `test_control_smoothing.py` — jitter regression: `SimArm` + injected servo noise; assert
  `q_ref`-based command variance is below threshold and the legacy feedback-seeded path is not.

PyBullet tests use DIRECT (headless) mode so they run in CI; the GUI is only for `dume sim`.

## Dependencies

Add `pybullet` (interactive sim + rendering/world). lerobot stays for the kinematics and the
future `DiffusionPolicy`. No additional renderer is introduced beyond PyBullet.

## Open questions / future specs

- `T_cam_mount` calibration procedure (hand-eye) — its own spec once the mount STL exists.
- lerobot dataset format mapping details in `to_lerobot()` — finalize when first real episodes
  are recorded.
- Whether grasp execution wants a dedicated motion path vs reusing `goto` — revisit after
  feeling it in sim.
