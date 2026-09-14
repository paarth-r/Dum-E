"""Force estimation from servo load — pure, no I/O.

The SO-101 has no torque sensors. What it has is ``Present_Load`` on every STS3215: the PWM duty
the servo's position loop is applying, which at zero velocity is proportional to torque. Subtract
the torque gravity demands at the measured pose (from the URDF's CAD masses, via
``Kinematics.gravity_torques``) and what remains is external load: a hand, a collision, a held
object.

Units: everything the estimator returns is in **raw servo load units** (signed, ~+-1000 full
scale) unless the name says otherwise. ``scale`` is the one conversion constant (N*m per raw
unit); it defaults to ``1.0`` so the seam exists from day one and a calibrated value drops in
without touching any consumer. Only the Cartesian wrench is multiplied through, so with a real
``scale`` it reads in N and N*m.

See docs/superpowers/specs/2026-09-08-force-sensing-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_ARM = 5  # joints that carry a Cartesian wrench; the gripper jaw is its own signal


@dataclass
class ForceReading:
    q_deg: np.ndarray  # joints the reading was taken at
    load: np.ndarray  # raw measured load, as read
    gravity_load: np.ndarray  # modelled gravity, in raw load units (tau_g / scale)
    residual: np.ndarray  # load - gravity_load, unfiltered
    tau_ext: np.ndarray  # filtered residual with the friction floor removed (raw units)
    wrench: np.ndarray  # [Fx, Fy, Fz, Tx, Ty, Tz] at the end-effector (N, N*m when scale is real)


class ForceEstimator:
    """Turn ``(q, load)`` samples into external joint torque and an end-effector wrench.

    ``scale``: N*m per raw load unit, scalar or per-joint. ``friction_floor``: deadband on the
    filtered residual (raw units), scalar or per-joint — gearbox stiction sits inside it.
    ``alpha``: EMA coefficient on the residual (1.0 = no filtering). The filter seeds on the first
    sample so a reading is honest from tick one instead of ramping up from zero.
    """

    def __init__(self, kin, *, scale=1.0, friction_floor=0.0, alpha: float = 0.3):
        self.kin = kin
        n = kin.n_joints
        self.scale = np.broadcast_to(np.asarray(scale, dtype=float), (n,)).copy()
        self.friction_floor = np.broadcast_to(np.asarray(friction_floor, dtype=float), (n,)).copy()
        self.alpha = float(alpha)
        self._filtered: np.ndarray | None = None

    def reset(self) -> None:
        self._filtered = None

    def gravity_load(self, q_deg) -> np.ndarray:
        """Modelled gravity at ``q_deg`` expressed in raw load units."""
        return self.kin.gravity_torques(q_deg) / self.scale

    def update(self, q_deg, load) -> ForceReading:
        q = np.asarray(q_deg, dtype=float)
        load = np.asarray(load, dtype=float)
        g_load = self.gravity_load(q)
        residual = load - g_load
        if self._filtered is None:
            self._filtered = residual.copy()
        else:
            self._filtered = self.alpha * residual + (1.0 - self.alpha) * self._filtered
        f = self._filtered
        tau_ext = np.sign(f) * np.maximum(np.abs(f) - self.friction_floor, 0.0)
        # tau = J^T F  ->  F = pinv(J^T) tau, arm joints only, in SI via scale.
        J = np.asarray(self.kin.jacobian(q))[:, :N_ARM]
        wrench = np.linalg.pinv(J.T) @ (tau_ext[:N_ARM] * self.scale[:N_ARM])
        return ForceReading(
            q_deg=q, load=load, gravity_load=g_load, residual=residual, tau_ext=tau_ext, wrench=wrench
        )


# ---- `dume feel` readout + calibration logging ------------------------------------------

def format_feel(joint_names, reading: ForceReading, *, voltage: float | None, hz: float) -> str:
    """Multi-line block for the ``dume feel`` readout: one row per joint plus the wrench."""
    lines = [
        f"{'joint':14} {'angle':>7} {'load':>7} {'gravity':>8} {'resid':>7} {'ext':>7}",
    ]
    for i, name in enumerate(joint_names):
        lines.append(
            f"{name:14} {reading.q_deg[i]:7.1f} {reading.load[i]:7.0f} {reading.gravity_load[i]:8.1f} "
            f"{reading.residual[i]:7.1f} {reading.tau_ext[i]:7.1f}"
        )
    fx, fy, fz, tx, ty, tz = reading.wrench
    volts = f"{voltage:.1f} V" if voltage is not None else "-- V"
    lines.append(
        f"wrench  F=({fx:+.2f},{fy:+.2f},{fz:+.2f})  T=({tx:+.3f},{ty:+.3f},{tz:+.3f})   "
        f"{volts}   {hz:4.0f} Hz"
    )
    return "\n".join(lines)


class LoadLogger:
    """CSV of ``(t, q, load, gravity, voltage)`` samples — the input the calibration sweep fits."""

    def __init__(self, path, joint_names):
        self.path = path
        self.names = list(joint_names)
        self._f = None
        self._w = None

    def __enter__(self) -> "LoadLogger":
        import csv

        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(
            ["t"]
            + [f"q_{n}" for n in self.names]
            + [f"load_{n}" for n in self.names]
            + [f"grav_{n}" for n in self.names]
            + ["voltage"]
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def write(self, t: float, reading: ForceReading, *, voltage: float | None) -> None:
        self._w.writerow(
            [t]
            + [float(v) for v in reading.q_deg]
            + [float(v) for v in reading.load]
            + [float(v) for v in reading.gravity_load]
            + ["" if voltage is None else voltage]
        )
