# dume

Smooth, intuitive **inverse-kinematics controller** for the [LeRobot](https://github.com/huggingface/lerobot) **SO-101** arm, driven by an Xbox controller or keyboard — built as a reusable library so other services can call `goto` / `follow_path` / `jog` directly. Now with an interactive PyBullet **simulator**, servo-load **force sensing**, and scaffolding for **imitation learning**, aimed at generalizable, precise manipulation in the spirit of [TidyBot++](https://tidybot2.github.io/).

## Highlights

- **Plan-then-solve:** every motion generates an explicit Cartesian path (straight-line + SLERP, trapezoidal timing), *then* solves IK along it — the seam for future obstacle-avoidance / contact-compliant planners.
- **Connected feel (no jitter):** the loop tracks an internal *commanded reference* (`q_ref`) instead of noisy servo feedback — the TidyBot++ lesson — with a damped-least-squares jog solver, deadzone/expo shaping, and velocity + jerk limits.
- **Two modes:** real-time Cartesian velocity jog, and absolute `goto` / named-pose recall.
- **Interactive sim:** `dume sim` — PyBullet GUI, Xbox/keyboard control, a grabbable object, and OnShape-style mouse navigation.
- **Force sensing:** `dume feel` — servo load minus modelled gravity, per joint, no extra hardware.
- **Learning scaffolding:** episode recording and a diffusion-policy interface. Perception is moving to a fixed desk camera that localises the robot in the camera frame (not built yet).
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

## Macros (`dume record`)

Record a whole motion by hand and replay it during teleop. `record` (or `./record-macro.sh`)
prompts for a name and a digit keybind (0-9), counts down 3-2-1, cuts torque so the arm goes
limp, and samples the measured joints at 50 Hz while you demonstrate the motion — hit space to
stop the take. Macros persist in `~/.dume/macros.json`.

```bash
./record-macro.sh                        # prompts: name, keybind, 3-2-1, record until space
.venv/bin/dume run                       # press the digit to play it back
```

During `run`, pressing a bound digit moves the arm to the macro's start pose, replays the
recording in its original timing, then holds the end pose and hands you back the sticks —
space aborts a macro mid-flight and holds wherever it is.

Outside a macro, space toggles a **limp gripper**: torque is cut on the jaw only so you can
position it (or the object) by hand while the arm keeps holding; space re-grips, and starting
a macro re-grips automatically. In squeeze mode the jaw then returns to the trigger position.

## Force sensing (`dume feel`)

The servos have no torque sensor, but each STS3215 reports `Present_Load` — the PWM duty its
position loop is applying, which at rest is proportional to torque. `dume feel` reads it on
every joint, subtracts the torque gravity demands at the measured pose (from the URDF's CAD
link masses, via placo), and shows the residual: a hand on the arm, a collision, a held object.

```bash
.venv/bin/dume feel                          # live table: angle / load / gravity / resid / ext
.venv/bin/dume feel --log sweep.csv --seconds 5   # record samples at this pose for calibration
.venv/bin/dume feel --dry-run                # sim self-check: residual must read exactly 0
```

The arm holds wherever it is while `feel` runs (load reads 0 with torque off), so push on a
link and the joints upstream of it light up. Everything is in **raw servo load units** until
`--scale` (N*m per unit) is fitted: log a grid of static poses with `--log`, regress load
against the `grav_*` columns, and the slope is the scale. The status line's Hz is the achieved
loop rate *with* the extra load read on the bus — check it before hanging anything on this
signal at 50 Hz. Design and the follow-ons (grasp sensing, collision detection, compliance,
zero-g) live in `docs/superpowers/specs/2026-09-08-force-sensing-design.md`.

## Simulation (`dume sim`)

A PyBullet harness that runs the exact control stack over a kinematic arm — no hardware needed.
It doubles as the teleop-feel test rig and the cockpit for recording demonstrations.

```bash
./run_sim.sh                 # keyboard + demo scene
.venv/bin/dume sim --keyboard --scene
.venv/bin/dume sim --noise 0.5    # inject servo-feedback noise to feel the smoothing
```

- **Drive:** keyboard (`WASD` = X/Y, `R`/`F` = Z, arrows = wrist, `O`/`C` = gripper, `M` = mode, hold `Shift` = turbo) or an Xbox pad (hold `RB` = turbo).
- **Grab:** a dynamic box rests on the ground; close the gripper near it to pick it up, open to drop it.
- **Navigate:** OnShape-style — left-drag orbit, `Ctrl`+left-drag pan, scroll to zoom.

## Perception & learning (scaffolding)

Foundations toward learned, generalizable grasping. Everything hardware/data-independent is real
and tested; hardware/data-bound pieces are explicit stubs.

- **Perception (next):** a fixed camera over the desk, with the robot localised in the camera
  frame. The earlier end-effector camera and flown-stereo point-cloud stack was removed on
  2026-09-14 (see git history before that date if you need it).
- **Episode recording** (`dataset.py`): synchronized observation/action episodes in a custom,
  framework-agnostic format, with a `to_lerobot()` export for training.
- **Policy interface** (`policy.py`): a `Policy` protocol with a scripted policy for tests and a
  lerobot `DiffusionPolicy` adapter, pending recorded demonstrations.

See `docs/superpowers/specs/` for the designs.
