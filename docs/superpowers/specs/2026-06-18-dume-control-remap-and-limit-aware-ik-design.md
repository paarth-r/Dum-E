# Dume control remap + limit-aware IK — design

Date: 2026-06-18
Status: approved-pending-review

## Problem

Driving the SO-101 under Xbox teleop has three rough edges Paarth called out:

1. **Gripper is two-handed and fiddly.** It's a rate control split across LT (open) / RT (close)
   plus A/Y full-open/close snaps. He wants one analog motion: squeeze the right trigger to close.
2. **Vertical (Z) is only on the right-stick Y axis.** He wants stick-*clicks* as an additional,
   discrete way to nudge up/down without leaving the sticks.
3. **The arm locks up.** Driving a Cartesian direction until a position joint (elbow_flex, the one
   before wrist_flex) hits full ROM leaves the arm pinned and unable to reverse — it has to "unwind"
   before it responds again. The arm should respect its own joint limits so it never pins.

Out of scope for this iteration (deferred, revisit after this lands):
- Body-frame / view-relative left-stick translation (left stick stays base-relative X/Y).
- Moving the wrist off the D-pad (the right stick stays on Z, so no analog stick is free for it).
- Hardware: re-tightening joints with Loctite + recalibration (waiting on Loctite).

## 1. Gripper: two modes, toggled by X

A `GripperMode` enum on the controller: `SQUEEZE` (default) and `RATE`. The **X** button rising edge
toggles between them. The input layer only reports raw trigger positions; the control core
interprets them per the active mode, so neither input source (Xbox, keyboard) needs to know the mode.

- **SQUEEZE (default):** RT position *is* the jaw openness, absolute and continuous.
  `gripper_cmd = gripper_open - rt * (gripper_open - gripper_closed)`.
  RT released (`rt=0`) -> fully open (95); RT fully depressed (`rt=1`) -> fully closed (5).
- **RATE (legacy feel):** `gripper_cmd += (lt - rt) * gripper_speed * dt`, clamped to
  `[gripper_closed, gripper_open]`. LT opens, RT closes.

`Command` changes: drop `gripper` (rate), `gripper_open_set`, `gripper_close_set`; add
`lt: float = 0.0`, `rt: float = 0.0` (both absolute, [0,1]) and `gripper_mode_toggle: bool = False`.
This frees the **A** and **Y** buttons (no more snap setpoints).

Keyboard adapter: open/close keys map to `rt = 0.0 / 1.0` (momentary), so SQUEEZE works there too
(hold close = closed); the mode toggle is also exposed.

## 2. Z (up/down): right-stick Y + stick clicks

Unchanged: right-stick Y -> Z velocity (analog). **Added:** L3 (left-stick click) -> Z up,
R3 (right-stick click) -> Z down, at a fixed rate while held.

Implemented purely in the input layer: the Z component of `Command.lin` becomes
`clip(right_stick_z + l3 - r3, -1, 1)`, where `l3`/`r3` are 1.0 while their stick is clicked. The
control core's velocity jog is unchanged — it still just consumes `cmd.lin`.

`XboxMap` adds `btn_l3`, `btn_r3` (SDL indices vary by driver — defaults are best-guess and MUST be
confirmed with `dume axes` before trusting them).

## 3. Button map (summary)

| Input | Action | Change |
|---|---|---|
| Left stick | X/Y plane translate | unchanged (base-relative) |
| Right stick Y | Z up/down | unchanged |
| L3 / R3 click | Z up / Z down | new |
| D-pad | wrist pitch / roll | unchanged |
| RT (+ LT in RATE mode) | gripper | reworked (see 1) |
| X | gripper mode toggle (SQUEEZE/RATE) | new |
| B | velocity / freeze (pose) mode toggle | unchanged |
| A, Y | (freed) | removed |

## 4. Limit-aware IK / anti-lockout

Root cause of the lockout: in `_velocity_jog`, `_pivot_target` integrates the desired Cartesian
velocity every tick, but the solved joints `q_send` get clipped to joint limits *afterward*. When a
position joint (pan/lift/elbow) saturates, `_pivot_target` keeps marching past the reachable
envelope, so it decouples from where the arm actually is. Reversing the stick then only chips away at
that accumulated overshoot before the arm visibly moves — the "lock."

**Fix — back-project the target onto the achieved configuration.** At the end of `_velocity_jog`,
after clipping `q_send`, set `self._pivot_target = self.kin_pos.fk(q_send[:3])[:3, 3]`. The target can
then never lead the arm past its reachable envelope: when elbow_flex hits its limit, the target stops
at that boundary, and a reversing command moves inward on the very next tick. This is anti-windup by
back-projection — standard, deterministic, and it makes the arm "aware of its own physical limits"
because the commanded target is always a configuration the joints can actually hold.

Wrist joints (wrist_flex / wrist_roll) already integrate-then-clamp each tick, so they don't wind up;
no change needed there.

**Self-awareness readout.** Surface, per tick (reusing `awareness.py` where possible), in telemetry
and the live `dume run` status line:
- minimum joint-limit margin (deg) across the arm joints, and which joint,
- a near-singular flag when manipulability drops below a threshold.

This is reporting only — the back-projection is what prevents the lockout. (Optional, not required
for this iteration: scale commanded velocity down as manipulability falls. The existing DLS damping
already keeps the solver from blowing up near singularities, so leave velocity scaling out unless
testing shows residual jumpiness.)

## Testing

All verifiable in sim over `SimArm` (no hardware):

- **Gripper SQUEEZE:** rt=0 -> gripper at `gripper_open`; rt=1 -> `gripper_closed`; rt=0.5 -> midpoint.
- **Gripper RATE:** holding rt closes over time; holding lt opens; clamped at the limits.
- **Mode toggle:** X rising edge flips `gripper_mode`; the same trigger input produces the
  mode-appropriate result before vs after the toggle.
- **Z clicks:** L3 raises Z, R3 lowers Z, and they sum with right-stick Y; clamped to [-1, 1].
- **Anti-lockout (regression):** drive a constant Cartesian velocity until a position joint
  saturates; assert the arm pins at the limit, then a reversed command moves it *immediately* (within
  a tick or two), and `_pivot_target` never exceeds the reachable envelope.
- **Awareness readout:** telemetry reports a shrinking joint-limit margin as a joint approaches its
  limit, and the near-singular flag trips in a known near-singular configuration.

## Files touched

- `src/dume/config.py` — `XboxMap.btn_l3/btn_r3`, `btn_x`; any gripper-mode default.
- `src/dume/input_xbox.py` — `Command` field rework; poll() builds `lt`/`rt`/`l3`/`r3`/`gripper_mode_toggle` and folds clicks into `lin[2]`.
- `src/dume/input_keyboard.py` — map open/close keys to `rt`, expose mode toggle.
- `src/dume/controller.py` — `GripperMode`, gripper interpretation in `step`/`_handle_buttons`, X toggle, pivot back-projection in `_velocity_jog`, awareness fields in `Telemetry`.
- `src/dume/cli.py` — updated `dume run` help text; awareness in the status line.
- `tests/` — gripper modes, mode toggle, Z clicks, anti-lockout regression, awareness readout.
