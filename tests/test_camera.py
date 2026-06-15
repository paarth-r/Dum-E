"""Camera geometry: intrinsics, the FK-derived camera pose, and projection round-trips.

No PyBullet here — pure pinhole math. The flown-stereo and sim-camera code build on these.
"""

import numpy as np
import pytest

from dume.camera import (
    T_CAM_MOUNT,
    CameraIntrinsics,
    camera_pose_from_fk,
    project_points,
    world_to_camera,
)
from dume.kinematics import Kinematics
from dume.poses import HOME_JOINTS


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


def test_intrinsics_from_fov_centre_and_K():
    intr = CameraIntrinsics.from_fov(640, 480, fov_y_deg=60.0)
    assert intr.cx == 320 and intr.cy == 240
    K = intr.K
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(intr.fx)
    assert K[2, 2] == 1.0


def test_camera_pose_from_fk_composes_mount(kin):
    """The camera pose is exactly the gripper FK times the fixed mount transform."""
    pose = camera_pose_from_fk(kin, HOME_JOINTS)
    expected = kin.fk(HOME_JOINTS) @ T_CAM_MOUNT
    assert np.allclose(pose, expected)
    assert pose.shape == (4, 4)


def test_world_to_camera_then_project_round_trips():
    """A point placed at a known camera-frame location projects to the expected pixel."""
    intr = CameraIntrinsics.from_fov(640, 480, 60.0)
    # Camera at origin looking down +z (optical frame == world frame here).
    cam_pose = np.eye(4)
    # A point 2 m in front, offset so it lands off-centre.
    p_world = np.array([[0.1, -0.05, 2.0]])
    p_cam = world_to_camera(p_world, cam_pose)
    assert np.allclose(p_cam, p_world)  # identity pose
    uv = project_points(p_cam, intr.K)
    # u = cx + fx*x/z, v = cy + fy*y/z
    expect_u = intr.cx + intr.fx * 0.1 / 2.0
    expect_v = intr.cy + intr.fy * (-0.05) / 2.0
    assert uv[0, 0] == pytest.approx(expect_u)
    assert uv[0, 1] == pytest.approx(expect_v)


def test_world_to_camera_handles_translation_and_rotation():
    """Round-trip a point through a non-trivial camera pose."""
    from dume import geometry as g

    cam_pose = g.transform_from_pos_rpy([0.3, -0.1, 0.5], [0.2, -0.3, 0.1])
    p_world = np.array([[0.5, 0.2, 0.9], [0.1, 0.0, 0.4]])
    p_cam = world_to_camera(p_world, cam_pose)
    # Bring back to world: p_world = R @ p_cam + t
    back = (cam_pose[:3, :3] @ p_cam.T).T + cam_pose[:3, 3]
    assert np.allclose(back, p_world)
