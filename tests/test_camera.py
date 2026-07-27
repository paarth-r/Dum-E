"""Camera geometry: intrinsics, the FK-derived camera pose, and projection round-trips.

No PyBullet here — pure pinhole math. The flown-stereo and sim-camera code build on these.
"""

import numpy as np
import pytest

from dume.camera import (
    CAMERA_FRAME,
    T_CAM_MOUNT,
    CameraIntrinsics,
    camera_pose_from_fk,
    mount_from_urdf,
    project_points,
    world_to_camera,
)
from dume.kinematics import Kinematics
from dume.poses import HOME_JOINTS


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


def test_mount_comes_from_the_urdf_camera_frame():
    """T_CAM_MOUNT is read from the URDF, not hardcoded — one source of truth.

    A hardcoded copy silently diverges from camera_optical_link the moment the frame is
    re-measured, and a wrong mount is invisible: the cloud just sits in the wrong place.
    """
    assert np.allclose(T_CAM_MOUNT, mount_from_urdf())


def test_mount_matches_fk_through_the_urdf_chain(kin):
    """FK walked to camera_optical_link equals gripper FK composed with the mount.

    This is the property the flown-extrinsics trick depends on; if the URDF joint and the
    mount transform ever disagree, triangulation is quietly wrong.
    """
    kcam = Kinematics(ee_frame=CAMERA_FRAME, joint_names=kin.joint_names)
    assert np.allclose(kcam.fk(HOME_JOINTS), kin.fk(HOME_JOINTS) @ T_CAM_MOUNT, atol=1e-9)


def test_mount_from_urdf_raises_clearly_when_frame_absent(tmp_path):
    """Re-exporting the URDF from onshape-to-robot drops the hand-authored camera frame.

    Failing loudly beats falling back to a guess: a silently wrong extrinsic produces a
    plausible-looking cloud in the wrong place.
    """
    stub = tmp_path / "no_camera.urdf"
    stub.write_text('<?xml version="1.0"?><robot name="x"><link name="base_link"/></robot>')
    with pytest.raises(ValueError, match=CAMERA_FRAME):
        mount_from_urdf(stub)


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
