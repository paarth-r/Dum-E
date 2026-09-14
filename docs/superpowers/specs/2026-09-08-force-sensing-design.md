# Force sensing on dumE

Date: 2026-09-08
Status: design approved. Note 2026-09-14: the end-effector camera was removed from the arm
and the URDF, so the `g_payload` term in Calibration is no longer needed — the CAD masses are
the whole model. SP1 implemented 2026-09-14 (`dume feel`, `forces.py`, `read_loads`;
hardware bus-timing measurement and the calibration sweep still to run). SP2 next.

## Goal

Give the arm a sense of touch. Concretely: know when it has hit something, know when it
is holding something and how big and how firmly, and eventually let it be pushed around
by hand as if weightless.

This document specifies the first two sub-projects. The later three are sequenced at the
end but not designed here — each gets its own spec, informed by measurements from the one
before it.

## Why this is possible without new hardware

The SO-101 carries no torque sensors and no series-elastic element. Every number below
comes from registers the STS3215 servos already publish and that `dume` does not yet read:

- `Present_Load` (addr 60, 2 bytes, sign-magnitude 10-bit) — the PWM duty the servo's
  internal position loop is applying. At zero velocity this is proportional to torque.
- `Present_Current` (addr 69) — reserved; not used by this design.
- `Present_Voltage` (addr 62) — supply rail, needed to normalise duty (see Calibration).

Two existing facts make the rest work. First, the arm is hand-backdrivable with torque
off — that is what makes `dume record` possible — so the gearbox is not self-locking and
gravity is the only thing fighting a human hand. Second, `urdf/so101_new_calib.urdf`
carries real CAD-derived inertials for nine links, and placo's `RobotWrapper` (already a
dependency via `dume.kinematics`) exposes `static_gravity_compensation_torques`. The
gravity model needed to interpret load readings therefore already exists in the repo.

## Ordering rationale

Detection is cheap; compliance is expensive. A collision is a *step change* in load, and
gearbox stiction — the dominant noise source — is slowly varying, so a change detector
resolves a hit well below the absolute noise floor. Compliance is the hard one, because
there you must know the absolute residual continuously in order to know which way to
yield, and stiction sits directly on top of that signal.

The gripper is cheaper still, because it needs no dynamics model at all and offers two
independent signals (position error and load) where the arm offers one.

## Units

All signals are carried in **raw servo load units** (signed, roughly ±1000 full scale).
Every consumer multiplies by a per-joint `scale` constant that defaults to `1.0`, so the
conversion seam exists from day one and SI units drop in later without touching any
downstream code. Nothing in SP1-SP4 depends on `scale` being correct — see Calibration.

## SP1 — Load plumbing, force estimator, `dume feel`

### `ArmIO` gains one method

```python
def read_loads(self) -> np.ndarray: ...  # length-6, signed, raw units, MOTOR_ORDER
```

`SO101Arm.read_loads` issues `bus.sync_read("Present_Load", MOTOR_ORDER)`. The register is
absent from lerobot's `normalized_data`, so values arrive raw, and `_decode_sign` already
unpacks the 10-bit sign-magnitude encoding declared in
`lerobot/motors/feetech/tables.py`.

**Sign convention.** Every motor currently has `drive_mode: 0` in
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`, so the
servo's load sign already matches the URDF joint direction. lerobot applies `drive_mode`
inversion inside `_normalize`, which does not run for `Present_Load` — meaning a future
recalibration that sets `drive_mode: 1` on any motor would silently invert that joint's
reported force. `connect()` therefore asserts all-zero drive modes and raises with a clear
message if that ever changes. This is a correctness guard, not a limitation.

`SimArm.read_loads` returns zeros by default and accepts an injectable load source so
tests and the grasp simulation can feed synthetic loads.

### `src/dume/forces.py` — pure, no I/O

- `gravity_torques(kin, q) -> np.ndarray` — placo's
  `static_gravity_compensation_torques` at measured joints, in N·m.
- `ForceEstimator` — holds per-joint `scale` and `friction_floor`, plus a low-pass filter.
  Returns per-joint external torque (measured load minus modelled gravity) and a Cartesian
  wrench at the gripper via the transposed-Jacobian pseudo-inverse.

Both are pure functions of their inputs, so the whole estimator tests against synthetic
load sequences with no hardware and no simulator.

### `dume feel`

A live readout at loop rate: one row per joint showing angle, raw load, modelled gravity,
and residual, plus a gripper block showing the SP2 grasp state. It is simultaneously the
user-facing deliverable and the instrument used to characterise the arm.

### Bus timing

This adds a third bus transaction per 50 Hz tick, on top of the existing position read and
goal write. Measure the achieved loop rate with load reads enabled before wiring any
consumer. If the budget is exceeded, read loads on alternate ticks (25 Hz), which is ample
for every consumer in this document. Do not guess — `dume run` already reports effective
loop Hz in its status line.

## Calibration — why CAD is the reference, and what it cannot tell us

`scale` converts load units to N·m. The naive way to find it is to hang a known mass. We
do not need to, because the arm's own CAD masses are a known reference signal:

```
load_raw(q) = a · tau_cad(q) + b · g_payload(q) + c
```

Sweeping a grid of static poses and regressing measured load against modelled gravity
yields `a` (the inverse scale), `b` (the unmodelled payload mass), and `c` (a bias). The
payload term matters: `urdf/so101_new_calib.urdf` gives the camera mount link
`mass="1e-9"` while a real Arducam UC-844 with lens and cable hangs at the claw — the
longest moment arm on the robot, so it is the dominant model error. It is separably
identifiable from `scale` because it multiplies a different known function of pose than
the link masses do.

Three properties make this well conditioned:

1. **All six joints are the same STS3215.** `scale` is a motor constant, not a per-joint
   property, so it is one shared parameter fitted from the joints with strong gravity
   signal (shoulder_lift, elbow_flex, wrist_flex) and applied to all. This also rescues
   `wrist_roll`, whose axis lies roughly along the forearm and whose gravity torque is
   near zero at every pose — it could never identify on its own.
2. **Samples must be static.** `Present_Load` is PWM duty, proportional to torque only at
   zero velocity. Move to a pose, let it settle, then sample.
3. **Supply voltage must be logged.** Duty maps to current through the rail, which sags
   under load, so `Present_Voltage` is recorded alongside every sample and normalised out.

**Stiction falls out of the same sweep for free.** Visiting each pose from above and from
below produces a hysteresis band rather than a curve. The band's centre is the gravity
fit; the band's *width* is the stiction estimate — which is precisely the number that
decides whether SP4 compliance will feel like zero-g or like shoving a stiff arm. One
sweep therefore produces the SI constants and the go/no-go signal for the hardest
sub-project.

**The known limitation.** A *systematic* CAD error is invisible to this fit. If the
printed parts are denser than modelled, or a motor variant is heavier than the URDF says,
the regression absorbs the entire error into `scale` while remaining perfectly
self-consistent. Regressing CAD against itself cannot detect this. CAD therefore buys a
self-consistent calibration, not a verified one.

The check is cheap and is not a procedure: weigh one object on a kitchen scale, hang it at
one pose, compare predicted against measured. It runs once, before SP5, where a biased
scale would mean commanding the wrong duty and dropping the arm. Everything before SP5
works in raw units and is unaffected.

## SP2 — Grasp sensing and crush guard

The gripper needs no dynamics model and offers two independent signals:

- **Commanded versus measured jaw position.** In SQUEEZE mode the trigger maps 1:1 to a
  goal position. A jaw that stalls short of its goal has something between the fingers,
  and the measured position *is* the object's width.
- **`Present_Load` while stalled.** How hard it is squeezing, independent of object size.

Together they separate a wide object held gently from a small object held hard, which
neither signal can do alone.

### `GraspState`

Fields: `object_present`, `width_mm`, `squeeze`, `slipping`.

Stall detection requires **both** conditions simultaneously — jaw position error beyond a
threshold for N consecutive ticks *and* load above a floor. Position error alone is just a
jaw still slewing toward its goal; load alone is just acceleration. Requiring both is what
makes the detector robust, and it is the core thing the tests must pin down.

`slipping` is width increasing while the trigger still commands closed.

### Width calibration

The gripper joint is revolute (`urdf/so101_new_calib.urdf:383`, -10° to 100°) but reports
in normalised 0-100 units, so opening is not linear in the reported value. Fit
`width_mm(jaw_units)` by closing on a handful of objects of known width. Stored in
`~/.dume/` alongside poses and macros.

### Crush guard

In the controller's gripper path: when `squeeze` exceeds `gripper_crush_limit`, latch the
commanded jaw position where it tripped and refuse to advance further closed. Backing the
trigger off past the latch clears it. Implemented as a floor on the commanded position, so
SQUEEZE mode's 1:1 trigger feel is completely untouched until the guard actually engages.

The guard must not fight the existing limp-gripper toggle (`89bdd0a`): with gripper torque
off, load readings are meaningless, so the guard is disabled whenever the jaw is limp.

### Testing

The estimator and grasp detector are pure functions over synthetic `(goal, measured,
load)` sequences. `SimArm` gains a virtual-object load model — an object of width W that
produces stall and rising load once the jaw commands below W — so the entire grasp path
including the crush guard runs under `--dry-run` with no hardware.

## Sequenced follow-ons (not designed here)

- **SP3 — Arm collision detection.** Step detection on the gravity-subtracted residual,
  wired into `run`, `goto`, and macro replay. Depends on SP1 only.
- **SP4 — Compliance mode.** Joint-space admittance: `q_cmd[j] += k[j] · tau_ext[j] · dt`,
  deadbanded, through the existing slew limiter and joint clamps. Push a link and that
  link yields; release and the servo's own position loop locks the pose in hardware for
  free. No IK in the loop, so no singularity risk with hands on the robot. Cartesian
  wrench is used only for display and thresholds, never to drive motion — a Cartesian
  admittance law would only let the arm be dragged by the gripper and would fight a push
  mid-link. Viability gated on the stiction width measured in SP1.
- **SP5 — PWM float.** True zero-g via `Operating_Mode = 2` with `Goal_Velocity` as duty,
  feeding forward gravity plus friction. Requires the verified `scale` from Calibration.
  Highest risk: it bypasses the servo position loop entirely, so a model error or a
  stalled loop drops the arm.

## Out of scope

- Force channels in recorded macros and datasets. Deferred until there is a reason.
- Force-controlled grasping (trigger commands squeeze force rather than jaw position).
  The crush guard is the bounded subset that delivers safety without a closed loop over a
  noisy signal through a stiff gearbox.
