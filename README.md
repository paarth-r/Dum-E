# Dum-E

A smooth inverse-kinematics controller for the [LeRobot SO-101](https://github.com/huggingface/lerobot)
arm, driven from an Xbox pad or keyboard, with a physics simulator, hand-guided macros, and
servo-load force sensing. Built as a library first: `DumeArm` is the API, the CLI is one consumer.
Named after DUM-E, Tony Stark's clumsy robot arm.

<!-- GIF goes here: `dume run` on hardware or `./run_sim.sh`. Drop it at docs/dume.gif and use
![Dum-E teleop](docs/dume.gif) -->

## What it does

- **Plan-then-solve motion.** Every move generates an explicit Cartesian path (straight line +
  SLERP, trapezoidal timing) and solves IK along it. That seam is where obstacle avoidance and
  contact-aware planning will plug in.
- **No jitter.** The loop tracks an internal commanded reference instead of noisy servo
  feedback (the [TidyBot++](https://tidybot2.github.io/) lesson), with a damped-least-squares
  jog solver, deadzone/expo shaping, and velocity + jerk limits. Near full extension a
  nullspace posture bias keeps the arm from folding into a singularity and locking up.
- **Two modes.** Real-time Cartesian velocity jog, and absolute `goto` / named-pose recall.
- **Hand-guided macros.** Cut torque, move the arm by hand, replay it from a digit key.
- **Force sensing without sensors.** `dume feel` reads servo load, subtracts modelled gravity
  from the URDF's CAD masses, and shows what is left: a hand, a collision, a held object.
- **Physics sim.** `dume sim` runs the identical control stack over a PyBullet arm with real
  contacts and a grabbable box.
- **Safe by default.** Workspace box, joint and step limits, `--dry-run` moves nothing.

## What you need

- A LeRobot **SO-101 follower** arm (Feetech STS3215 servos) on USB, calibrated once with
  `dume calibrate`.
- An **Xbox controller** (wireless Series X pad tested) or just the keyboard for the sim.
- **macOS on Apple silicon** is the tested platform. Linux should work but is unverified.
- **Python 3.12+** (lerobot 0.5 requires it) and [uv](https://github.com/astral-sh/uv).

## Install

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q     # 143 tests, no hardware needed
```

Two platform quirks, both handled or documented:

- **placo's macOS wheel is broken** (0.9.23 links `liburdfdom_*.4.0` but ships `6.0.0`).
  `dume` symlinks the sonames itself on first import (`src/dume/_placo_fix.py`); nothing to do.
- **pybullet has no macOS arm64 wheel**, so pip builds it from source. On recent macOS SDKs the
  build fails in `zutil.h` on a `#define fdopen` that clashes with the system `stdio.h`; remove
  that define and build with `CFLAGS="-std=gnu17 -Wno-deprecated-non-prototype"`. The compiled
  module lives in `.venv`, so a fresh venv repeats the build.

lerobot ships no URDF, so the SO-ARM100 `so101_new_calib.urdf` and meshes are vendored in `urdf/`.

## Quick start

```bash
.venv/bin/dume find-port            # confirm the serial port
.venv/bin/dume calibrate            # one-time SO-101 calibration (wraps lerobot)
.venv/bin/dume save-pose            # hand-pose the arm, hit Enter to save the start pose
.venv/bin/dume run --dry-run        # validate IK + feel, no motion
.venv/bin/dume run                  # live control (slews to the saved start pose first)
```

## Controls

Xbox, in `dume run` and `dume sim`. Button indices are SDL's Xbox layout; if a pad maps
differently, `dume axes` shows live indices and `XboxMap` in `config.py` is the single place
to change them.

| Input | Action |
|---|---|
| Left stick | X / Y (positions the wrist pivot) |
| Right stick Y, or L3 / R3 click | Z up / down |
| D-pad up / down | wrist pitch (`wrist_flex`) |
| D-pad left / right | wrist roll (`wrist_roll`) |
| RT | gripper: **squeeze** mode maps trigger travel 1:1 to jaw opening |
| LT / RT | gripper in **rate** mode: LT opens, RT closes, integrated over time |
| X | toggle gripper mode (squeeze / rate) |
| B | toggle velocity jog / freeze (pose hold) |
| RB (hold) | turbo, 2.5x speed |
| 0-9 keys | play the macro bound to that digit (space aborts) |
| Space | toggle a limp gripper (torque off on the jaw only) |
| Ctrl-C | quit; torque is released on disconnect |

Keyboard, in `dume sim --keyboard` (focus the PyBullet window): `WASD` X/Y, `R`/`F` Z,
arrows wrist pitch/roll, `O`/`C` gripper open/close, `[`/`]` snap closed/open, `M` mode,
hold `Shift` for turbo.

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

## Architecture

```
input_xbox / input_keyboard  -->  Command  -->  controller  -->  arm (ArmIO)
                                                  |  ^
                               planning, kinematics, forces
```

| Module | Owns |
|---|---|
| `service.py` | `DumeArm`, the public facade: `goto`, `follow_path`, `jog`, `home`, `goto_joints`, `run_teleop` |
| `controller.py` | the 50 Hz control core: commanded reference, jog solver, slew/jerk limits, joint moves |
| `kinematics.py` | FK / IK / Jacobian / gravity torques over lerobot's placo solver, plus the DLS jog solver |
| `planning.py` | Cartesian trajectories with trapezoidal timing |
| `arm.py` | `ArmIO` protocol; `SO101Arm` (hardware via lerobot) and `SimArm` (kinematic stand-in) |
| `sim_world.py` | PyBullet scene, renderer, `PyBulletArm` (physics-backed `ArmIO`), GUI navigation |
| `forces.py` | `ForceEstimator`: load minus gravity, filtered, mapped to an end-effector wrench |
| `macros.py`, `poses.py` | hand-guided recordings and named joint setpoints, persisted under `~/.dume/` |
| `input_xbox.py`, `input_keyboard.py` | pad and keyboard into a shared `Command` |
| `config.py` | every tunable: feel, limits, gripper, the Xbox map |
| `dataset.py`, `policy.py` | episode recording and the policy interface for imitation learning |
| `geometry.py` | pure 4x4 pose math |
| `cli.py` | the `dume` command |

Joint vectors are length 6 in URDF order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper`. The first five are degrees; the gripper is normalised 0-100.

## Status

| Area | State |
|---|---|
| Teleop, goto, named poses, macros | live on hardware |
| Physics sim with grasping | live |
| Force sensing (`dume feel`) | built and verified in sim; hardware bus timing and load calibration pending |
| Grasp sensing, collision detection, compliance, zero-g | designed, not built ([design](docs/superpowers/specs/2026-09-08-force-sensing-design.md)) |
| Perception | next up: a fixed camera over the desk with the robot localised in the camera frame. The earlier end-effector camera stack was removed on 2026-09-14 |
| Imitation learning | recording format and policy interface exist; diffusion-policy training is a stub |

Design notes for each feature live in `docs/superpowers/specs/`.

## License

[MIT](LICENSE).
