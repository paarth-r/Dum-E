# Physical grasp simulation — design

Date: 2026-06-30
Status: approved (design), pending implementation plan

## Context

`dume sim` exists to run the **real control stack** against a surrogate arm and show how it
behaves the way it will IRL / in prod. Today it falls short of that in two ways:

1. The sim arm is **kinematic** — teleported each tick via `resetJointState`. It instantly adopts
   commanded joints, so `read_joints` never reflects reality (e.g. a jaw stalled against an
   object). The controller's feedback path is therefore not exercised faithfully.
2. Grasping is a **magnet hack**: a fixed constraint fires when the gripper origin is within 7 cm
   of the box *centre* — an isotropic sphere far larger than the 5 cm box, so the box snaps to the
   gripper from a distance with no jaws, friction, or contact involved.

We want the sim to grasp the way hardware would: jaws physically close on the box, friction at the
contact points holds it, opening drops it. Scope for now is **real-ish** — believable contact
behaviour with basic tuning; occasional slip/tip/drop is acceptable. Reliable pick-and-place and
high-fidelity contact tuning are deferred to the grasp-learning phase ("path to C" below).

## Goals

- The control stack (IK, `q_ref` smoothing, leash, posture/limit-aware solve, gripper logic) runs
  **unchanged** against a physics-backed arm, exactly as it runs against `SO101Arm`.
- `read_joints` reports **physically-achieved** joint positions, including a jaw stalled on a box.
- Grasp is **emergent**: bounded gripper motor force presses the jaws on the box; lateral friction
  at the contacts holds it; opening the jaws drops it. No constraint, no magnet.
- Physics knobs centralized so reaching high fidelity (C) is tuning, not re-architecture.
- One sim path: `dume sim` always runs on the physics arm (gravity + ground always on); a scene
  just adds a box.

## Non-goals (this phase)

- Reliable, slip-free pick-and-place (that is C).
- Contact-solver / friction-model tuning beyond what "real-ish" needs.
- Changing hardware (`SO101Arm`) or the controller.
- Changing `--dry-run` / headless behaviour: `SimArm` (kinematic, instant-adopt, fast,
  deterministic) stays as-is for dry-run feel-checks and the headless controller tests.

## Architecture

The faithful surrogate is a new `ArmIO` implementation the controller actually drives — not a
display mirror. `DumeArm` already accepts an injected `arm`, which is the seam we use.

```
SimRenderer      owns the PyBullet client: loads the arm URDF, ground plane, gravity, scene
                 objects, friction, solver params. Renders the GUI. (the "world")
PyBulletArm      ArmIO over that client+body: write = motor control + advance sim; read =
(new)            getJointState. (the "robot I/O" — a hardware surrogate)
DumeArm(arm=...) the public stack, unchanged, driving PyBulletArm instead of SO101Arm/SimArm
```

`cmd_sim` wires them: build `SimRenderer(gui=True, dynamic=True)`, wrap it in `PyBulletArm`, build
`DumeArm(arm=pybullet_arm)`, load the scene, connect, slew to the start pose, `run_teleop`. The GUI
shows the same body the motors move, so there is no separate mirror to keep in sync.

### `PyBulletArm` (new, in `sim_world.py`)

Implements the `dume.arm.ArmIO` protocol against a `SimRenderer`:

- `write_joints(q)` — `setJointMotorControl2(POSITION_CONTROL)` for the five arm joints (stiff
  force, so the displayed arm tracks the command closely) and the gripper joint (**bounded** force
  — the core grasp knob: enough to hold, not enough to explode the box). Then advance the sim by
  one control `dt` (`round(dt / sim_timestep)` ≈ 5 substeps at 240 Hz). This mirrors hardware:
  send an action, time elapses.
- `read_joints()` — `getJointState` actual positions for the five arm joints (rad→deg) and the
  gripper joint mapped back to the 0..100 motor units `SO101Arm` reports. Gripper map is the
  identity deg↔units used today (`set_joints` already treats the gripper value as degrees), so a
  jaw stalled at a wider angle reads as a higher gripper value — feedback reflects reality.
- `connect()` — hold the loaded pose: read current joint positions and set motor targets to them
  so the arm does not sag under gravity before control takes over.
- `relax()` — drop motor forces to ~0 (torque-off equivalent; the arm then sags under gravity).
- `disconnect()` — no-op (the renderer owns the client lifecycle).
- `is_calibrated()` → `True`; `name = "pybullet"`.

Advancing physics inside `write_joints` keeps the controller agnostic: `step()` calls
`write_joints` once per tick, so one control tick advances physics by exactly `dt` (faithful 1:1).
`controller.run()` still paces the loop to `loop_hz` in wall-clock.

### `GraspParams` (new dataclass, in `sim_world.py`)

Single home for every physics knob, so C is "turn the knobs + add stability tricks":

- `arm_motor_force` — position-control force for the five arm joints (stiff tracking).
- `gripper_motor_force` — bounded jaw force (the grasp knob).
- `jaw_friction`, `box_friction` — lateral friction on the jaw links and box.
- `sim_timestep` (default 1/240), `substeps_per_tick` (derived from `dt`), `solver_iterations`.

`SimRenderer(dynamic=True, grasp=GraspParams())` applies `solver_iterations` via
`setPhysicsEngineParameter`, sets `jaw_friction` on the three jaw links
(`gripper_link`, `gripper_frame_link`, `moving_jaw_so101_v1_link`) at load, and `box_friction` on
dynamic scene objects.

### `SimRenderer` changes

- Always `dynamic=True` for `dume sim` (gravity + ground always on). Self-collision stays off
  (only arm↔box and box↔ground collide).
- `set_joints` (teleport) is retained — still used by the constraint-utility grasp test and as a
  reset primitive — but is no longer how the sim arm moves during teleop.
- New `contact_points(body_a, body_b)` helper wrapping `getContactPoints`, returning count and
  summed normal force, for the readout and as the data hook for C.
- `attach` / `release` / `holding` (constraint magnet) stay as a utility (a "perfect grasp"
  baseline useful for the learning phase) but are removed from the sim loop.

### `cmd_sim` changes

- Build `SimRenderer` + `PyBulletArm` + `DumeArm(arm=...)`; drive via the normal control loop (no
  `set_joints` mirror, no `drive`-vs-teleport branch).
- Delete the distance-sphere magnet grab/release block.
- Status line shows the contact readout: contact count, summed normal force, and box height — the
  tangible signal that the grip is engaging.

## Data flow (one tick, grasping)

```
poll() -> Command
controller.step():
  read_joints()  <- getJointState (physical, e.g. jaw stalled on box)
  ... IK / q_ref / posture / gripper logic (unchanged) ...
  write_joints(q_send):
     setJointMotorControl2 arm joints (stiff) + gripper joint (bounded force)
     advance sim ~5 substeps   # jaws press, contacts + friction update, box held or slips
on_tick(tel): render nav + print contacts/force/box-height
```

## Faithfulness rationale

The whole point of the change: because `write` drives motors and `read` returns `getJointState`,
the controller sees the same imperfect feedback it will see on hardware. A jaw that cannot close
because a box is in the way stalls and reads stalled; the box is held only by real friction; the
arm shows real contact compliance. The sim now demonstrates prod behaviour rather than a kinematic
idealization.

## Path to C (higher fidelity, later)

No architectural change — only:
- Tune `GraspParams` (friction model incl. spinning/rolling friction, `gripper_motor_force`,
  `solver_iterations`, contact damping/stiffness, box mass).
- Optionally add finger compliance / soft contact.
- Use `contact_points` as the substrate for grasp-quality metrics and demo recording.

## Testing

Deterministic where physics allows; reliable pick-and-place is explicitly **not** asserted.

- Keep `test_sim_grasp.py` (constraint-utility `attach`/`release`) unchanged.
- New `PyBulletArm` tests:
  - implements `ArmIO` (duck-type/`isinstance` of the protocol).
  - write→read roundtrip with no obstacle: commanding a joint vector and writing repeatedly
    converges `read_joints` to it within tolerance.
  - gripper-units roundtrip with no obstacle: write 50 → read ≈ 50.
  - `relax()` drops motor force (joints sag under gravity → `read_joints` changes).
  - **stall faithfulness** (the key behavioural test): with a box clamped between the jaws,
    commanding the jaw fully closed leaves `read_joints` gripper value short of the command — the
    jaw stalled on the box.
  - contact smoke: jaws closed on a between-the-jaws box → `contact_points(arm, box)` non-empty.
- Manual sim verification for feel (grasp, lift, slip, drop).

## Risks

- **PyBullet grasp stability.** Objects can slip/jitter/eject. Mitigation: scope is "real-ish";
  basic friction + adequate solver iterations; accept slip. `GraspParams` isolates the tuning.
- **Arm sag before control engages.** Mitigated by holding the loaded pose on `connect()`.
- **Blocking-move loops** (`goto`, `home`, `_drive_until_idle`) currently sleep only for
  `SO101Arm`; with `PyBulletArm` they iterate without sleeping, stepping physics fast — acceptable
  (physics still advances one `dt` per write). Confirm start-pose slew looks right in the GUI.
- **`sim_world.py` size.** Adding `PyBulletArm` + `GraspParams` grows an already ~526-line file;
  split into a `pybullet_arm.py` module if it gets unwieldy.
