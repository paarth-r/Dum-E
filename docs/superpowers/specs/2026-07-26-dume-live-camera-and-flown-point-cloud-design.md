# Live end-effector camera and flown-extrinsics point cloud

Status: approved design, not yet implemented
Date: 2026-07-26

## Goal

Drive the physical SO-101 to a sequence of poses while streaming its end-effector Arducam live,
and build a 3D point cloud of what it sees — accumulated as the arm moves, expressed in the
**arm base frame**.

This is the first work that runs the perception stack on real hardware. Everything in
`flown_stereo.py` has so far only ever seen `SimCamera`'s fabricated detections.

## Background: what already exists

- `flown_stereo.triangulate()` — DLT triangulation from two 3x4 projection matrices. Correct and
  tested. Takes **matched pixel correspondences**.
- `flown_stereo.relative_pose()`, `propose_grasp()` — usable as-is.
- `camera.camera_pose_from_fk()` — `kin.fk(joints) @ T_CAM_MOUNT`, the flown-extrinsics trick.
- `urdf/so101_new_calib.urdf` now carries a hand-authored `camera_optical_link` (added
  2026-07-26) giving the mount transform as a real URDF frame.
- `SimCamera` in `sim_world.py` — the geometric stand-in used by all current tests.

## The gap this closes

The only correspondence producer today is `triangulate_detections()`, which pairs points **by
object id**. Those ids exist because `SimCamera` invents them. Real imagery has no ids, so the
correspondence problem — the actual hard part of turning photographs into 3D — is entirely
unbuilt. Stage 2 below is that piece.

## Non-goals

- **Metric accuracy.** Intrinsics are a placeholder 70 deg FOV guess (`arducam.py:23`), so
  triangulated depth is scaled by however wrong `fx` is. The cloud will be structurally correct
  and dimensionally wrong. Deliberate, per decision on 2026-07-26; a ChArUco intrinsic
  calibration is separate future work and makes this same code metric with no rewrite.
- **Dense reconstruction.** Sparse ORB features only. Dense StereoSGBM needs rectified pairs,
  which arbitrary arm motion does not give without a constrained scan pattern. Later upgrade.
- **Hand-eye calibration.** The mount transform stays the analytic CAD-derived value. A wrong
  mount shifts the cloud rigidly; wrong intrinsics distort its shape. Only the latter is fatal
  to structure, and neither is being fixed here.
- **Object detection.** `ArduCamSource.detect()` returns empty. No detector on real frames yet.

## Frames and conventions

One rule, stated once: **the point cloud lives in the arm base frame, origin at the base.**

This is not a transform anyone has to apply. `kin.fk()` returns poses in the base frame, and
`triangulate()` returns points in whatever frame its camera poses were expressed in. Feed it FK
poses and the output is already base-framed. Do not re-base it downstream.

Optical frame stays OpenCV convention (+z forward, +x right, +y down), per `camera.py`.

### Camera pose comes from MEASURED joints, never the commanded reference

Every camera pose fed to triangulation must be:

```python
kin.fk(arm.read_joints()) @ mount     # measured — correct
```

and never:

```python
kin.fk(controller.q_ref) @ mount      # commanded — WRONG for perception
kin.fk(telemetry.joints_sent) @ mount # same mistake via telemetry
```

This deliberately contradicts the control path, and the contradiction is correct. The controller
tracks an internal commanded reference `q_ref` and ignores servo feedback on purpose — the
TidyBot++ lesson, documented at `controller.py:84` — because feeding noisy measurements back into
control produces jitter. That is the right call **for control**.

It is the wrong call for **perception**, because `q_ref` is where the arm was *told* to go, not
where it *is*. The two differ by real, systematic amounts: gravity sag alone put the arm several
degrees off its commanded pose on 2026-07-25, and a proportional servo loop holding an arm
horizontal has nonzero steady-state error by construction. Triangulating with commanded poses
bakes that droop into the geometry as a pose error that is *correlated across views* — which is
the worst kind, because it does not average out. It bends the whole cloud.

Note `Telemetry.joints_sent` is the **commanded** vector (`controller.py:160` sets it from
`q_send`), so the `on_tick` hook is not a valid pose source. Call `arm.read_joints()` explicitly.

**Averaging at keyframes.** Measured joints carry servo quantisation noise, which is why the
control path avoids them. Since `scan` already dwells at each stop, average several reads while
the arm is stationary and use the mean for that keyframe's pose. This gets measurement accuracy
without the jitter that made `q_ref` necessary for control — the dwell is what makes averaging
valid, and it is only valid because the arm is not moving.

## Stage 1 — capture, motion, live view

### 1.1 `ArduCamSource.capture()` (`src/dume/arducam.py`)

Replace the stub with a real UVC grab.

**Device selection by resolution probe, not index.** macOS enumeration order is unstable and the
machine has three cameras (built-in FaceTime 1920x1080, the Arducam 1280x800, and an iPhone
Continuity camera). Probe indices 0..7, open each, read the reported frame size, and take the
first that matches 1280x800. Cache the winning index on the instance. Raise a clear error naming
what was found if nothing matches, rather than silently streaming the laptop's webcam — that
failure mode would produce a plausible-looking and completely meaningless point cloud.

The sensor is mono (OV9281). Note `arducam.py:3` currently says "OV9781", a typo; the device
reports `Arducam OV9281 USB Camera`. Fix the docstring.

OpenCV will hand back a 3-channel BGR frame regardless; convert to single-channel grey once at
capture and store that. `CameraFrame.rgb` keeps its existing meaning for the sim path.

### 1.2 Threaded capture (`src/dume/arducam.py`)

The control loop runs at `config.dt`. A blocking USB read inside it would stall the arm — this
is the same class of problem as reading noisy servo feedback in the control path.

So: a background thread does `cap.read()` in a tight loop and writes into a single-slot "latest
frame" holder under a lock. `capture()` returns the latest frame without blocking. Consumers get
whatever is freshest and never wait on I/O. Dropped frames are fine and expected; we are not
recording video.

### 1.3 `src/dume/liveview.py` (new)

Small module owning the `cv2.imshow` window. Takes a frame-provider callable and an optional
overlay callable, so it is testable without a camera and without a window (the render step is
pure: frame in, annotated frame out; only `show()` touches the GUI).

Overlay draws the current FK camera pose as text. Keypress handling: `q` closes, and in `scan`,
any key aborts motion.

### 1.4 `dume scan` (`src/dume/cli.py`)

New subcommand. Visits saved setpoints in sequence via `DumeArm.goto_joints`, streaming the live
view throughout and pausing briefly at each stop.

```
dume scan                       # visit every setpoint in the joint store
dume scan --poses a b c         # visit named setpoints in order
dume scan --dwell 1.0           # seconds held at each stop
dume scan --slew 3.0            # deg/tick cap, lower than the 6.0 default
```

Safety, given the arm faulted once on 2026-07-25 with no root cause found and scripted motion is
precisely the case where nobody's hand is on it:

- Default to a reduced slew rate.
- Any keypress in the view window aborts the remaining moves.
- Wrap the per-move call so a mid-scan hardware fault stops the scan and reports the **original**
  exception rather than dying in cleanup (see 1.6).

### 1.5 Live view in `dume run` (`src/dume/cli.py`)

`run` gains `--view`. The controller already exposes an `on_tick` hook (used today by
`_status_printer`); the view render hangs off that, so no new loop and no threading in the
control path. Composes with the existing status printer rather than replacing it.

### 1.6 Guard `SO101Arm.disconnect()` (`src/dume/arm.py:118`)

Not cosmetic, and in scope because stage 1 adds scripted motion. Today `self._robot.disconnect()`
is unguarded, so a failure during cleanup raises **over the top of** whatever real exception was
propagating. That is exactly what happened on 2026-07-25: a genuine fault was buried under a
`ConnectionError` from the shutdown path and the original traceback was lost.

Catch and log on the disconnect path so the real exception survives.

## Stage 2 — live sparse point cloud

### 2.1 `src/dume/pointcloud.py` (new)

Pure geometry and matching. No hardware, no GUI, no PyBullet — same discipline as
`flown_stereo.py`, so it is fully testable on synthetic frames.

**Keyframes.** Not every frame is useful: triangulation from a near-zero baseline is
ill-conditioned and produces garbage at huge range. Accept a new keyframe only once the camera
has moved more than a threshold (default 20 mm) from the last one. This is the single most
important quality knob.

**Matching.** ORB detect + describe, `BFMatcher` with Hamming distance and a ratio test, between
the new keyframe and the previous one.

**Triangulation.** Feed matched pixels and the two FK camera poses straight into the existing
`flown_stereo.triangulate()`. No new triangulation math. Poses are FK of **measured, dwell-averaged
joints** — see "Camera pose comes from MEASURED joints" above; this is not negotiable, and a
keyframe captured while the arm is still moving must be discarded rather than posed from `q_ref`.

**Filtering** — this determines whether the cloud is a shape or a fog:

1. *Cheirality*: reject points behind either camera (`z <= 0` in either optical frame).
2. *Reprojection*: project the solved point back into both views, reject above a pixel threshold.
3. *Range gate*: reject points nearer than a few cm or beyond a max range; unconstrained DLT
   happily returns points at infinity from noisy matches.

**Accumulation.** Growing `(N, 3)` array in the base frame, with a cap and voxel-grid
downsampling so an unbounded scan does not exhaust memory.

### 2.2 `src/dume/cloudview.py` (new)

PyBullet GUI as the 3D viewer — already a hard dependency, so nothing new to install.

Loads the URDF and sets it to live joint angles each tick, then draws the cloud with
`addUserDebugPoints`. Seeing the arm rendered in the same frame as the cloud makes "is this in
the right place" answerable at a glance, which is most of what we want from it right now.

Debug points must be re-issued in batches rather than one call per point; per-point calls are far
too slow at cloud scale.

### 2.3 Wiring

`dume scan --cloud` runs the scan while feeding keyframes to the cloud builder and rendering
live. The cloud can be written out as a plain `.npy` of base-frame points at exit.

## Error handling

- **No Arducam found** — raise naming the cameras that *were* found. Never silently fall back to
  another camera.
- **Camera disappears mid-run** — the capture thread marks itself dead; `capture()` raises rather
  than returning a stale frame forever. Motion continues; only the view stops.
- **Hardware fault mid-scan** — abort remaining moves, surface the original exception (1.6).
- **Too few matches between keyframes** — skip the pair, keep scanning. Textureless scenes are
  expected and are not an error.

## Testing

All new geometry is testable without hardware, matching the existing pattern.

- `tests/test_pointcloud.py` — synthetic scene: generate 3D points, project into two known
  camera poses with a known K, and confirm triangulation recovers the originals. Cover each
  filter (behind-camera, high reprojection, out-of-range) and keyframe threshold logic.
- `tests/test_liveview.py` — overlay rendering as a pure function; no window.
- `tests/test_arducam.py` — device probe against a fake enumerator: picks 1280x800, raises with a
  useful message when absent, does not pick the 1920x1080 device.
- **Measured-not-commanded pose** — regression test with a fake arm whose `read_joints()` returns
  a config deliberately offset from the commanded one (simulating sag). Assert the keyframe pose
  matches FK of the *measured* joints. Without this, a future refactor that "helpfully" reuses
  `telemetry.joints_sent` would silently bend every cloud with no visible failure.
- Hardware paths (`capture`, `cloudview`) stay stubs in tests, as `arducam.py` does today.

Existing suite is 110 passing and must stay green.

## Staging

Stage 1 ships and gets verified on hardware **before** stage 2 begins. First real frames will
tell us things this design is guessing at — exposure, focus, whether the claw occludes the view,
how much texture the scene actually has — and those answers should inform stage 2 rather than be
assumed by it.

## Known-wrong values carried forward

Recorded so they are not mistaken for measurements:

- `arducam.py` intrinsics: placeholder 70 deg FOV. Not calibrated.
- `camera_optical_joint` z = -0.0550 m: provisional, from a scaled CAD screenshot. Constrained to
  roughly -0.095..-0.045 (the flat mounting face on the claw underside).
- `camera.py:39` `T_CAM_MOUNT` still holds the superseded `[0, 0, -0.02]` and disagrees with the
  URDF. Reconcile to a single source of truth during stage 1.
- Optical origin sits on the **sensor centre**, not the lens projection centre; these differ by
  roughly the focal length, a few mm.
