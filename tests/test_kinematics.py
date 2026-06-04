import numpy as np
import pytest

from dume import geometry as g
from dume.kinematics import Kinematics


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


def test_joint_names(kin):
    assert kin.joint_names[:5] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]


def test_fk_returns_4x4(kin):
    T = kin.fk(np.zeros(kin.n_joints))
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0, 0, 0, 1])


@pytest.mark.parametrize(
    "q",
    [
        np.array([0, 0, 0, 0, 0, 0.0]),
        np.array([15, -20, 25, 10, -30, 0.0]),
        np.array([-30, 10, -15, -20, 40, 0.0]),
    ],
)
def test_ik_recovers_fk_pose(kin, q):
    target = kin.fk(q)
    seed = q + np.array([8, -7, 6, 5, -6, 0.0])
    sol = kin.ik(seed, target, orientation_weight=1.0)
    achieved = kin.fk(sol)
    dp, ang = g.pose_error(achieved, target)
    assert np.linalg.norm(dp) < 1e-3  # < 1 mm
    assert ang < np.deg2rad(1.0)  # < 1 deg


def test_ik_preserves_gripper(kin):
    q = np.array([10, -10, 10, 5, 0, 42.0])
    target = kin.fk(q)
    sol = kin.ik(q, target)
    assert abs(sol[5] - 42.0) < 1e-6
