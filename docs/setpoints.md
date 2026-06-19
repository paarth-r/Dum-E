# Setpoint motor control

`dume` can capture and recall **named joint configurations** — the exact position of all six
motors (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`). The first
five are degrees; the gripper is normalised `0..100`.

These are *joint-space* setpoints (raw motor positions), distinct from the end-effector
Cartesian poses used by `goto`. A setpoint reproduces an exact arm configuration; it does not
depend on IK.

## Where they live

All setpoints are stored together in one human-readable JSON file:

```
~/.dume/joint_poses.json
```

```json
{
  "start": {
    "shoulder_pan": 0.0,
    "shoulder_lift": -20.0,
    "elbow_flex": 20.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 50.0
  },
  "pickup": { "...": 0.0 }
}
```

The file is safe to hand-edit. Bare 6-element lists (in motor order) are also accepted on load,
but `save-pose` always writes the named-dict form.

`start` is special only by convention: it's the setpoint `dume run` launches into. Every other
name is just another setpoint.

## Capturing a setpoint — `dume save-pose`

Connects to the arm, **cuts motor torque so you can move it by hand**, then waits. Pose the arm,
press **Enter**, and all six motor positions are read and saved.

```bash
dume save-pose                 # save as "start" (the pose `run` launches into)
dume save-pose --name pickup   # save an arbitrary named setpoint
dume save-pose --name drop     # ...add as many as you like; same file
```

### Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--name NAME` | `start` | Name to save the setpoint under (overwrites if it exists). |
| `--file PATH` | `~/.dume/joint_poses.json` | Which JSON store to write to. |
| `--no-relax` | _(torque off)_ | Keep motor torque **on** and snapshot the currently held position as-is (e.g. after teleop) instead of freeing the arm for hand-posing. |
| `--port PORT` | from `ControllerConfig` | Serial port override. |
| `--id ID` | from `ControllerConfig` | Robot id override. |

Press **Ctrl-C** (or send EOF) at the prompt to cancel without saving.

## Recalling a setpoint at launch — `dume run`

`run` moves to a saved setpoint before handing control to the gamepad.

```bash
dume run                       # slew to "start", then teleop
dume run --start-pose pickup   # slew to a different saved setpoint first
dume run --no-start-pose       # skip it; start from wherever the arm already is
```

### Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--start-pose NAME` | `start` | Which saved setpoint to move to on launch. |
| `--start-file PATH` | `~/.dume/joint_poses.json` | Which JSON store to read from. |
| `--no-start-pose` | _(move to start)_ | Don't move to any setpoint; begin from the current position. |
| `--dry-run` | _(live)_ | Simulate the full pipeline with no motor motion. |

If the requested setpoint isn't in the store, `run` prints a note and starts from the current
position. The chosen setpoint also becomes "home" for the session.

> On real hardware, `run` powers on torque (holding the current pose) and then slews to the
> setpoint at the configured `joint_slew_deg` rate — smooth, but it starts moving immediately.

## Using setpoints from code

The CLI is a thin consumer; the same store is available programmatically:

```python
import numpy as np
from dume.poses import JointPoseStore
from dume.service import DumeArm

store = JointPoseStore()                 # ~/.dume/joint_poses.json
print(store.names())                     # ['drop', 'pickup', 'start']

with DumeArm() as arm:                   # dry_run=True for simulation
    arm.goto_joints(store.get("pickup")) # exact joint-space move (restores gripper)
    arm.goto_joints(store.get("drop"))

# Save a config you computed yourself:
store.set("stow", np.array([0, -90, 90, 0, 0, 5], dtype=float))
```

`JointPoseStore` methods: `names()`, `has(name)`, `get(name)` (returns a length-6 array in
motor order), `set(name, joints)` (validates length and persists). `DumeArm.goto_joints(joints)`
drives the five arm joints straight to the target and restores the gripper from the sixth value.
