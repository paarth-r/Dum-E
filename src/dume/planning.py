"""Cartesian path planning: turn (start pose, goal pose) into a sampleable Trajectory.

This is the *plan-then-solve* seam. The controller never sends a goal straight to IK; it
asks a ``Planner`` for a ``Trajectory`` of end-effector poses, then solves IK along it.
Today the only planner is a straight line; a future obstacle-avoiding or contact-compliant
planner just needs to implement ``Planner.plan`` and return a ``Trajectory``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dume import geometry as g


@dataclass
class Trajectory:
    """A time-parameterised end-effector path. ``sample(t)`` returns a 4x4 pose.

    ``poses`` is precomputed for inspection (obstacle checks, visualisation); ``sample``
    interpolates the underlying start/goal with a trapezoidal time-scaling so motion eases
    in and out instead of snapping.
    """

    start_pose: np.ndarray
    goal_pose: np.ndarray
    duration: float
    ramp_frac: float = 0.25

    def s_of_t(self, t: float) -> float:
        """Normalised progress in [0, 1] under a symmetric trapezoidal velocity profile."""
        T = self.duration
        if T <= 0:
            return 1.0
        t = min(max(t, 0.0), T)
        ta = self.ramp_frac * T  # accel (and decel) duration
        if ta <= 0:
            return t / T
        vp = 1.0 / (T - ta)  # peak normalised velocity (area under profile == 1)
        a = vp / ta
        if t < ta:
            return 0.5 * a * t * t
        if t <= T - ta:
            return 0.5 * a * ta * ta + vp * (t - ta)
        te = T - t
        return 1.0 - 0.5 * a * te * te

    def sample(self, t: float) -> np.ndarray:
        return g.interpolate_pose(self.start_pose, self.goal_pose, self.s_of_t(t))

    def waypoints(self, n: int = 50) -> list[np.ndarray]:
        if self.duration <= 0 or n <= 1:
            return [self.goal_pose.copy()]
        return [self.sample(self.duration * i / (n - 1)) for i in range(n)]

    @property
    def is_done_at(self):
        return self.duration


class Planner(Protocol):
    def plan(self, start_pose: np.ndarray, goal_pose: np.ndarray) -> Trajectory: ...


@dataclass
class StraightLinePlanner:
    """Straight line in position (lerp) + SLERP in orientation, trapezoidal timing.

    Duration is chosen so neither the translational nor rotational velocity/acceleration
    limits are exceeded, given the trapezoidal ramp.
    """

    max_linear_vel: float = 0.10
    max_linear_acc: float = 0.30
    max_angular_vel: float = 1.0
    max_angular_acc: float = 3.0
    ramp_frac: float = 0.25
    min_duration: float = 0.15

    def _axis_duration(self, dist: float, vmax: float, amax: float) -> float:
        if dist <= 1e-9:
            return 0.0
        r = self.ramp_frac
        # velocity limit: peak vel = dist / (T*(1-r)) <= vmax
        t_v = dist / (vmax * (1.0 - r))
        # accel limit: peak accel = dist / (T^2 * r * (1-r)) <= amax
        t_a = math.sqrt(dist / (amax * r * (1.0 - r)))
        return max(t_v, t_a)

    def plan(self, start_pose: np.ndarray, goal_pose: np.ndarray) -> Trajectory:
        start = np.asarray(start_pose, dtype=float)
        goal = np.asarray(goal_pose, dtype=float)
        d_pos = g.position_distance(start, goal)
        d_ang = g.rotation_angle(start, goal)
        duration = max(
            self._axis_duration(d_pos, self.max_linear_vel, self.max_linear_acc),
            self._axis_duration(d_ang, self.max_angular_vel, self.max_angular_acc),
            self.min_duration,
        )
        return Trajectory(start, goal, duration, ramp_frac=self.ramp_frac)
