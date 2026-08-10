# dume

Smooth, intuitive **inverse-kinematics controller** for the [LeRobot](https://github.com/huggingface/lerobot) **SO-101** arm, driven by an Xbox controller or keyboard — built as a reusable library so other services can call `goto` / `follow_path` / `jog` directly. Now with an interactive PyBullet **simulator** and scaffolding for **perception + imitation learning**, aimed at generalizable, precise manipulation in the spirit of [TidyBot++](https://tidybot2.github.io/).

## Highlights

- **Plan-then-solve:** every motion generates an explicit Cartesian path (straight-line + SLERP, trapezoidal timing), *then* solves IK along it — the seam for future obstacle-avoidance / contact-compliant planners.
- **Connected feel (no jitter):** the loop tracks an internal *commanded reference* (`q_ref`) instead of noisy servo feedback — the TidyBot++ lesson — with a damped-least-squares jog solver, deadzone/expo shaping, and velocity + jerk limits.
- **Two modes:** real-time Cartesian velocity jog, and absolute `goto` / named-pose recall.
- **Interactive sim:** `dume sim` — PyBullet GUI, Xbox/keyboard control, a grabbable object, a live end-effector camera, and OnShape-style mouse navigation.
- **Perception + learning scaffolding:** end-effector-camera "flown-extrinsics" stereo depth (calibration-free, from forward kinematics), episode recording, and a diffusion-policy interface.
- **Captured start pose:** hand-pose the arm (torque off) and save its joints; `run` slews there on launch.
- **Safe:** workspace bounding box, joint + step limits, `--dry-run` (no motor motion).
- **Modular & tested:** `DumeArm` facade is the public API; the CLI is just one consumer; ~90 tests.

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/dume find-port            # confirm the serial port
.venv/bin/dume calibrate            # one-time SO-101 calibration
.venv/bin/dume save-pose            # hand-pose the arm, hit Enter to save the start pose
.venv/bin/dume run --dry-run        # validate IK + feel, no motion
.venv/bin/dume run                  # live control (slews to the saved start pose first)
```

## Saving poses

`save-pose` cuts motor torque so you can move the arm by hand, then snapshots all six motor
positions to `~/.dume/joint_poses.json` (readable `{motor: value}` JSON) when you press Enter:

```bash
.venv/bin/dume save-pose                 # save as "start" (what `run` launches into)
.venv/bin/dume save-pose --name pickup   # save any other named setpoint
.venv/bin/dume save-pose --no-relax      # keep torque on; snapshot the held pose as-is
```

`run` moves to the `start` pose on launch. Override or skip it:

```bash
.venv/bin/dume run --start-pose pickup   # start at a different saved pose
.venv/bin/dume run --no-start-pose       # start from wherever the arm already is
```

`start` is just one named setpoint — you can save and recall as many as you like. See
[`docs/setpoints.md`](docs/setpoints.md) for full setpoint usage and flags.

## Simulation (`dume sim`)

A PyBullet harness that runs the exact control stack over a kinematic arm — no hardware needed.
It doubles as the teleop-feel test rig and the cockpit for recording demonstrations.

```bash
./run_sim.sh                 # keyboard + demo scene + end-effector camera
.venv/bin/dume sim --keyboard --scene --camera
.venv/bin/dume sim --noise 0.5    # inject servo-feedback noise to feel the smoothing
```

- **Drive:** keyboard (`WASD` = X/Y, `R`/`F` = Z, arrows = wrist, `O`/`C` = gripper, `M` = mode, hold `Shift` = turbo) or an Xbox pad (hold `RB` = turbo).
- **Grab:** a dynamic box rests on the ground; close the gripper near it to pick it up, open to drop it.
- **Camera:** a live end-effector RGB/depth feed with object detection.
- **Navigate:** OnShape-style — left-drag orbit, `Ctrl`+left-drag pan, scroll to zoom.

## End-effector camera (`dume scan`, `dume run --view`)

The claw carries an **Arducam UC-844** (1280×800 global-shutter mono, OV9281). Its mount frame
lives in the URDF as `camera_optical_link`, so FK walks straight to the camera.

```bash
.venv/bin/dume view                    # camera only, no arm — use this to focus the lens
.venv/bin/dume run --view              # teleop with the live camera feed
.venv/bin/dume scan                    # visit every saved setpoint, streaming the feed
.venv/bin/dume scan --poses a b c      # visit specific setpoints, in order
.venv/bin/dume scan --save ~/scans/01  # also write each stop's frame + measured pose
```

### Point cloud (`dume cloud`)

```bash
.venv/bin/dume cloud                   # sweep, triangulate, live 3D view, save to ~/scans/cloud.npy
.venv/bin/dume cloud --stops 11 --pan 36   # wider sweep, more viewpoints
.venv/bin/dume cloud --no-3d           # skip the PyBullet window
```

Sweeps `shoulder_pan` around wherever the arm currently is, capturing a posed frame at each
stop, matching ORB features between successive keyframes, and triangulating them through
`flown_stereo`. The cloud accumulates live in the **arm base frame** — origin at the base —
and renders in a PyBullet window alongside the arm itself, so you can see whether the geometry
lands where it should.

The sweep runs one way across the arc, which means every stop is approached from the same
direction; that keeps gear backlash loaded consistently instead of flipping sign mid-scan.

Points are rejected on three independent grounds: behind either camera, high reprojection
error (the main defence against bad matches), and out of range. Frames closer together than
`--min-baseline` are skipped entirely, because near-parallel rays triangulate noise into
enormous distances.

**The cloud is non-metric.** Triangulated depth scales directly with focal length, and the
intrinsics are still a guess, so the structure is right and the scale is not. Do not measure
anything off it until a real calibration lands.

`view` touches no hardware but the camera and stays open until you press `q`. It overlays a
live focus readout — Laplacian variance and ORB keypoint count — so setting the M12 lens is
hill-climbing on a number rather than squinting at a blurry picture. Turn the barrel until
both stop rising; it also tracks the best value seen so you can tell when you've gone past it.
Feature count is the one that matters: sparse reconstruction needs hundreds of keypoints per
frame, and an out-of-focus lens yields single digits.

`scan` walks the arm through saved setpoints (see [`docs/setpoints.md`](docs/setpoints.md)),
pausing at each to capture a frame paired with the camera's pose in the **arm base frame**.
Any keypress in the view window aborts the whole scan, and it moves at a gentler slew than
teleop by default since nobody's hand is on the arm.

Two things worth knowing before trusting anything measured from these frames:

- **The camera is picked by resolution, not device index.** macOS enumeration order is not
  stable, and choosing the wrong camera is silent — frames arrive and look fine while meaning
  nothing. A machine with no 1280×800 device gets an error naming what *was* found.
- **Poses come from measured joints, never commanded ones.** The controller deliberately
  ignores servo feedback to avoid jitter, which is right for control and wrong for perception:
  `q_ref` is where the arm was *told* to go, and gravity sag puts it degrees away. Since that
  error is correlated across views it does not average out. `scan` averages several measured
  reads during the dwell instead, which is valid only because the arm is stationary.

**Intrinsics are still an uncalibrated placeholder** (`ArduCamSource.calibrated` is `False`).
Anything metric derived from these frames is scaled by however wrong the guessed focal length
is. The mount translation is CAD-derived rather than hand-eye calibrated, and its position
along the reach is provisional.

## Perception & learning (scaffolding)

Foundations toward learned, generalizable grasping. Everything hardware/data-independent is real
and tested; hardware/data-bound pieces are explicit stubs.

- **Flown-extrinsics depth** (`flown_stereo.py`): the arm knows where its own hand is, so an
  end-effector camera (Arducam UC-844, 1280×800 global-shutter mono) snapped from two poses forms
  a known-baseline stereo pair with *no* rig calibration → triangulate → propose a grasp.
- **Episode recording** (`dataset.py`): synchronized observation/action episodes in a custom,
  framework-agnostic format, with a `to_lerobot()` export for training.
- **Policy interface** (`policy.py`): a `Policy` protocol with a scripted policy for tests and a
  lerobot `DiffusionPolicy` adapter, pending recorded demonstrations.

See `docs/superpowers/specs/` for the designs.
