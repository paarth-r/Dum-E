"""``ForceEstimator``: measured load minus modelled gravity, filtered, deadbanded, and mapped to
a Cartesian wrench. Pure functions over synthetic sequences — no hardware, no placo."""

import numpy as np
import pytest

from dume.forces import ForceEstimator

TAU_G = np.array([0.0, -0.5, -0.4, -0.1, 0.0, 0.002])  # N*m, a plausible pose


class _Kin:
    """Fixed gravity + a Jacobian whose first five columns are the identity's, so a joint torque
    vector maps 1:1 onto [Fx, Fy, Fz, Tx, Ty]."""

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    n_joints = 6

    def gravity_torques(self, q):
        return TAU_G.copy()

    def jacobian(self, q):
        return np.eye(6)


def _est(**kw):
    return ForceEstimator(_Kin(), **kw)


def test_gravity_alone_reads_as_zero_external_torque():
    est = _est(scale=0.001)  # 1 raw unit = 1 mN*m
    r = est.update(np.zeros(6), TAU_G / 0.001)
    assert np.allclose(r.tau_ext, 0.0)
    assert np.allclose(r.wrench, 0.0)


def test_gravity_load_is_model_over_scale():
    est = _est(scale=0.002)
    r = est.update(np.zeros(6), np.zeros(6))
    assert np.allclose(r.gravity_load, TAU_G / 0.002)


def test_external_torque_is_the_residual_in_raw_units():
    est = _est(scale=0.001, alpha=1.0)
    r = est.update(np.zeros(6), TAU_G / 0.001 + np.array([0, 100, 0, 0, 0, 0]))
    assert r.tau_ext[1] == pytest.approx(100.0)
    assert np.allclose(np.delete(r.tau_ext, 1), 0.0)


def test_friction_floor_deadbands_symmetrically():
    est = _est(scale=1.0, friction_floor=10.0, alpha=1.0)
    small = est.update(np.zeros(6), TAU_G + np.array([5, -5, 0, 0, 0, 0]))
    assert np.allclose(small.tau_ext, 0.0)
    est.reset()
    big = est.update(np.zeros(6), TAU_G + np.array([15, -15, 0, 0, 0, 0]))
    assert big.tau_ext[0] == pytest.approx(5.0)
    assert big.tau_ext[1] == pytest.approx(-5.0)


def test_friction_floor_can_be_per_joint():
    est = _est(scale=1.0, friction_floor=[0, 10, 0, 0, 0, 0], alpha=1.0)
    r = est.update(np.zeros(6), TAU_G + np.array([5, 5, 0, 0, 0, 0]))
    assert r.tau_ext[0] == pytest.approx(5.0)
    assert r.tau_ext[1] == pytest.approx(0.0)


def test_filter_seeds_on_first_sample_then_lowpasses():
    est = _est(scale=1.0, alpha=0.5)
    first = est.update(np.zeros(6), TAU_G + 8.0)
    assert np.allclose(first.tau_ext, 8.0)  # no ramp-up from zero on the first tick
    second = est.update(np.zeros(6), TAU_G + 0.0)
    assert np.allclose(second.tau_ext, 4.0)
    third = est.update(np.zeros(6), TAU_G + 0.0)
    assert np.allclose(third.tau_ext, 2.0)


def test_wrench_maps_arm_torques_through_the_jacobian():
    est = _est(scale=0.5, alpha=1.0)
    r = est.update(np.zeros(6), TAU_G / 0.5 + np.array([2, 4, 6, 0, 0, 0]))
    # tau = J^T F with J = I (first five columns) -> F = scale * tau_ext, gripper excluded.
    assert r.wrench.shape == (6,)
    assert np.allclose(r.wrench, [1, 2, 3, 0, 0, 0])


def test_wrench_ignores_the_gripper_joint():
    est = _est(scale=1.0, alpha=1.0)
    r = est.update(np.zeros(6), TAU_G + np.array([0, 0, 0, 0, 0, 500]))
    assert r.tau_ext[5] == pytest.approx(500.0)
    assert np.allclose(r.wrench, 0.0)


def test_reset_forgets_filter_state():
    est = _est(scale=1.0, alpha=0.5)
    est.update(np.zeros(6), TAU_G + 8.0)
    est.reset()
    r = est.update(np.zeros(6), TAU_G + 2.0)
    assert np.allclose(r.tau_ext, 2.0)


def test_estimator_reads_zero_external_torque_holding_in_physics_sim():
    """End-to-end gravity feedback: real URDF model, PyBullet applied torques (N*m, scale=1),
    arm holding still under gravity -> the residual is model error only, well under the shoulder's
    own gravity torque, and pushing on nothing yields ~no wrench."""
    from dume.kinematics import Kinematics
    from dume.poses import HOME_JOINTS
    from dume.sim_world import PyBulletArm, SimRenderer

    kin = Kinematics()
    est = ForceEstimator(kin, scale=1.0, alpha=1.0)
    with SimRenderer(gui=False, dynamic=True) as r:
        arm = PyBulletArm(r)
        arm.connect()
        for _ in range(40):
            arm.write_joints(HOME_JOINTS)
        q = arm.read_joints()
        reading = est.update(q, arm.read_loads())
    assert abs(reading.gravity_load[1]) > 0.2
    assert np.all(np.abs(reading.tau_ext[:5]) < 0.05 * max(np.abs(reading.gravity_load[:5])) + 0.01)
    assert np.linalg.norm(reading.wrench[:3]) < 0.5  # newtons of phantom force
