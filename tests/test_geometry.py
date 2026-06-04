import numpy as np
from scipy.spatial.transform import Rotation

from dume import geometry as g


def test_pos_rpy_roundtrip():
    pos = [0.1, -0.2, 0.3]
    rpy = [0.3, -0.5, 1.1]
    T = g.transform_from_pos_rpy(pos, rpy)
    assert np.allclose(g.position_of(T), pos)
    assert np.allclose(g.rpy_of(T), rpy)


def test_xyzrpy_roundtrip():
    v = np.array([0.2, 0.1, 0.25, 0.2, -0.3, 0.8])
    T = g.xyzrpy_to_pose(v)
    assert np.allclose(g.pose_to_xyzrpy(T), v)


def test_interpolate_endpoints():
    T0 = g.transform_from_pos_rpy([0, 0, 0], [0, 0, 0])
    T1 = g.transform_from_pos_rpy([1, 2, 3], [0.5, -0.5, 1.0])
    assert np.allclose(g.interpolate_pose(T0, T1, 0.0), T0)
    assert np.allclose(g.interpolate_pose(T0, T1, 1.0), T1)


def test_interpolate_midpoint_position_is_linear():
    T0 = g.transform_from_pos_rpy([0, 0, 0], [0, 0, 0])
    T1 = g.transform_from_pos_rpy([1, 2, 4], [0, 0, 0])
    mid = g.interpolate_pose(T0, T1, 0.5)
    assert np.allclose(g.position_of(mid), [0.5, 1.0, 2.0])


def test_interpolate_clamps_s():
    T0 = g.transform_from_pos_rpy([0, 0, 0], [0, 0, 0])
    T1 = g.transform_from_pos_rpy([1, 0, 0], [0, 0, 0])
    assert np.allclose(g.interpolate_pose(T0, T1, -5), T0)
    assert np.allclose(g.interpolate_pose(T0, T1, 5), T1)


def test_pose_error_zero_when_equal():
    T = g.transform_from_pos_rpy([0.3, 0.1, 0.2], [0.1, 0.2, 0.3])
    dp, ang = g.pose_error(T, T)
    assert np.allclose(dp, 0)
    assert ang < 1e-9


def test_rotation_angle_known():
    T0 = g.transform_from_pos_rpy([0, 0, 0], [0, 0, 0])
    T1 = g.transform_from_pos_rpy([0, 0, 0], [0, 0, np.pi / 2])
    assert abs(g.rotation_angle(T0, T1) - np.pi / 2) < 1e-9
