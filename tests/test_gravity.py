"""Gravity model + Jacobian seam on ``Kinematics`` (SP1 of the force-sensing design)."""

import numpy as np
import pytest

from dume.kinematics import Kinematics

POSES = [
    np.array([0, 0, 0, 0, 0, 0.0]),
    np.array([20, 45, -30, 15, 40, 30.0]),
    np.array([-35, -60, 70, -25, -80, 60.0]),
]


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


def _potential_energy(kin, q_deg):
    """U = m_total * g * z_com, from placo's own CoM (test-only access to the wrapper)."""
    robot = kin._kin.robot
    for name, val in zip(kin.joint_names, np.deg2rad(q_deg)):
        robot.set_joint(name, float(val))
    robot.update_kinematics()
    return float(robot.total_mass() * 9.81 * np.asarray(robot.com_world())[2])


@pytest.mark.parametrize("q", POSES)
def test_gravity_torques_match_potential_energy_gradient(kin, q):
    """tau_g[j] = dU/dq_j (N*m per rad): the torque the motor must apply to hold the pose."""
    tau = kin.gravity_torques(q)
    assert tau.shape == (kin.n_joints,)
    h = 1e-4
    for j in range(kin.n_joints):
        dq = np.zeros(kin.n_joints)
        dq[j] = np.rad2deg(h)
        fd = (_potential_energy(kin, q + dq) - _potential_energy(kin, q - dq)) / (2 * h)
        assert tau[j] == pytest.approx(fd, abs=2e-3), kin.joint_names[j]


def test_gravity_torques_vertical_axes_carry_no_load(kin):
    tau = kin.gravity_torques(POSES[1])
    assert abs(tau[0]) < 1e-3  # shoulder_pan: vertical axis
    # wrist_roll: axis runs along the forearm, so only the jaw's off-axis offset loads it —
    # a few mN*m, two orders below the shoulder.
    assert abs(tau[4]) < 1e-2


def test_gravity_torques_have_real_magnitude(kin):
    # ~0.4 kg of arm hanging off the shoulder: the lift joint must see tenths of a N*m.
    assert abs(kin.gravity_torques(POSES[0])[1]) > 0.2


@pytest.mark.parametrize("q", POSES)
def test_jacobian_matches_finite_difference_fk(kin, q):
    J = kin.jacobian(q)
    assert J.shape == (6, kin.n_joints)
    h = 0.05
    for j in range(kin.n_joints):
        dq = np.zeros(kin.n_joints)
        dq[j] = h
        fd = (kin.fk(q + dq)[:3, 3] - kin.fk(q - dq)[:3, 3]) / (2 * np.deg2rad(h))
        assert np.allclose(J[:3, j], fd, atol=1e-5), kin.joint_names[j]
