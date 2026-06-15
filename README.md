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

- **Drive:** keyboard (`WASD` = X/Y, `R`/`F` = Z, arrows = wrist, `O`/`C` = gripper, `M` = mode) or an Xbox pad.
- **Grab:** a dynamic box rests on the ground; close the gripper near it to pick it up, open to drop it.
- **Camera:** a live end-effector RGB/depth feed with object detection.
- **Navigate:** OnShape-style — left-drag orbit, `Ctrl`+left-drag pan, scroll to zoom.

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
