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
