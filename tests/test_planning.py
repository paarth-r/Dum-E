import numpy as np

from dume import geometry as g
from dume.planning import StraightLinePlanner, Trajectory


def _pose(xyz, rpy=(0, 0, 0)):
    return g.transform_from_pos_rpy(xyz, rpy)


def test_trajectory_endpoints_exact():
    start = _pose([0.2, 0, 0.2])
    goal = _pose([0.3, 0.1, 0.25], [0.2, 0, 0.3])
    traj = StraightLinePlanner().plan(start, goal)
    assert np.allclose(traj.sample(0.0), start)
    assert np.allclose(traj.sample(traj.duration), goal)
    # past the end stays at goal
    assert np.allclose(traj.sample(traj.duration + 5), goal)


def test_s_monotonic_and_bounded():
    traj = Trajectory(_pose([0, 0, 0]), _pose([1, 0, 0]), duration=2.0)
    ts = np.linspace(0, 2.0, 200)
    s = [traj.s_of_t(t) for t in ts]
    assert s[0] == 0.0
    assert abs(s[-1] - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in s)
    assert all(b >= a - 1e-12 for a, b in zip(s, s[1:]))  # non-decreasing


def test_velocity_starts_and_ends_near_zero():
    traj = Trajectory(_pose([0, 0, 0]), _pose([1, 0, 0]), duration=2.0)
    dt = 1e-3
    v_start = (traj.s_of_t(dt) - traj.s_of_t(0)) / dt
    v_end = (traj.s_of_t(2.0) - traj.s_of_t(2.0 - dt)) / dt
    assert v_start < 0.05
    assert v_end < 0.05


def test_velocity_within_limits():
    planner = StraightLinePlanner(max_linear_vel=0.1, max_linear_acc=0.3)
    start = _pose([0.1, 0, 0.2])
    goal = _pose([0.4, 0, 0.2])  # 0.3 m straight line
    traj = planner.plan(start, goal)
    dt = 1e-3
    max_v = 0.0
    t = 0.0
    while t < traj.duration:
        p0 = g.position_of(traj.sample(t))
        p1 = g.position_of(traj.sample(t + dt))
        max_v = max(max_v, np.linalg.norm(p1 - p0) / dt)
        t += dt
    assert max_v <= 0.1 * 1.05  # within 5% of the limit


def test_zero_distance_uses_min_duration():
    p = _pose([0.2, 0, 0.2])
    traj = StraightLinePlanner(min_duration=0.2).plan(p, p)
    assert traj.duration == 0.2
    assert np.allclose(traj.sample(0.0), p)
    assert np.allclose(traj.sample(traj.duration), p)


def test_waypoints_count_and_ends():
    start = _pose([0.2, 0, 0.2])
    goal = _pose([0.3, 0, 0.2])
    traj = StraightLinePlanner().plan(start, goal)
    wps = traj.waypoints(10)
    assert len(wps) == 10
    assert np.allclose(wps[0], start)
    assert np.allclose(wps[-1], goal)
