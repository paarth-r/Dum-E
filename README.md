# dume

Smooth, intuitive **inverse-kinematics controller** for the [LeRobot](https://github.com/huggingface/lerobot) **SO-101** arm, driven by an Xbox controller — built as a reusable library so other services can call `goto` / `follow_path` / `jog` directly.

## Highlights

- **Plan-then-solve:** every motion generates an explicit Cartesian path (straight-line + SLERP, trapezoidal timing), *then* solves IK along it — the seam for future obstacle-avoidance / contact-compliant planners.
- **Smooth feel:** deadzone + expo input shaping, EMA velocity smoothing, joint slew-rate limiting, IK seeded from current pose for continuity.
- **Two modes:** real-time Cartesian velocity jog, and absolute `goto` / named-pose recall.
- **Safe:** workspace bounding box, joint + step limits, `--dry-run` (no motor motion).
- **Modular:** `DumeArm` facade is the public API; the CLI is just one consumer.

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/dume find-port            # confirm the serial port
.venv/bin/dume calibrate            # one-time SO-101 calibration
.venv/bin/dume run --dry-run        # validate IK + feel, no motion
.venv/bin/dume run                  # live control
```

See `docs/superpowers/specs/` for the design.
