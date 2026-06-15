"""Tests for dume.sim_world — PyBullet-backed kinematic simulator.

All tests use DIRECT (headless) mode so they run in CI without a display.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dume.camera import CameraIntrinsics
from dume import geometry as g
from dume.poses import HOME_JOINTS
from dume.sim_world import SceneObject, SimCamera, SimRenderer, SimScene


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_looking_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build a 4x4 camera pose (OpenCV optical frame) placed at *eye* looking at *target*.

    +z points from eye toward target, +y points down in world, +x = +z cross +y_down.
    """
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)

    # World "down" as +y in optical convention.
    world_down = np.array([0.0, 0.0, -1.0])  # -Z world is "down" to something above the table

    # Ensure non-degenerate: if forward is parallel to world_down, perturb.
    if abs(np.dot(fwd, world_down)) > 0.99:
        world_down = np.array([0.0, 1.0, 0.0])

    right = np.cross(fwd, world_down)
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)  # +y in optical frame = -up

    R = np.stack([right, down, fwd], axis=1)  # columns: x_cam, y_cam, z_cam
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return T


# ---------------------------------------------------------------------------
# Test 1: renderer loads URDF and accepts joint commands
# ---------------------------------------------------------------------------

def test_renderer_loads_and_sets_joints():
    """SimRenderer loads the SO-101 URDF and maps at least 5 of 6 motor joints."""
    renderer = SimRenderer(gui=False)
    try:
        # set_joints should not raise, even with all 6 DOF
        renderer.set_joints(HOME_JOINTS)
        found = renderer.joint_indices
        # The URDF must expose at least 5 of the 6 MOTOR_ORDER joints.
        assert len(found) >= 5, (
            f"Expected >=5 joints mapped, got {len(found)}: {list(found.keys())}"
        )
    finally:
        renderer.disconnect()


# ---------------------------------------------------------------------------
# Test 2: depth image matches a known distance to a box
# ---------------------------------------------------------------------------

def test_camera_depth_matches_known_distance():
    """The minimum rendered depth to a box placed D metres away is within 10 % of D.

    The arm URDF is also present in the scene (SimRenderer always loads it).  We use
    the segmentation buffer to read depth *specifically on the box pixels* so arm
    geometry does not interfere with the measurement.
    """
    BOX_HALF = 0.08  # 8 cm half-extent — large enough to cover many pixels
    BOX_POS = np.array([0.0, 0.0, 1.0])   # 1 m along camera +Z
    EYE = np.array([0.0, 0.0, -1.0])       # camera 1 m behind origin, looking +Z

    # Distance from eye to the *front face* of the box (nearest surface).
    D_to_front = float(np.linalg.norm(BOX_POS - EYE)) - BOX_HALF  # 2.0 - 0.08 = 1.92 m

    scene = SimScene()
    scene.add(SceneObject(
        name="test_box",
        shape="box",
        half_extents=[BOX_HALF, BOX_HALF, BOX_HALF],
        position=BOX_POS.tolist(),
        rgba=[1.0, 0.0, 0.0, 1.0],
    ))

    # Camera at EYE looking toward BOX_POS along +Z world.
    cam_pose = _camera_looking_at(EYE, BOX_POS)

    intrinsics = CameraIntrinsics.from_fov(320, 240, fov_y_deg=60.0)

    renderer = SimRenderer(gui=False)
    try:
        renderer.load_scene(scene)
        box_body_id = renderer.scene_bodies["test_box"]
        cam = SimCamera(renderer, intrinsics, pose_provider=lambda: cam_pose)
        frame = cam.capture()

        assert frame.depth is not None, "capture() returned None depth"
        assert frame.rgb is not None, "capture() returned None rgb"
        assert frame.depth.shape == (240, 320), f"unexpected depth shape {frame.depth.shape}"

        # Use the cached segmentation to isolate pixels that belong to the box.
        seg = cam._last_seg
        assert seg is not None, "No segmentation buffer cached after capture()"
        box_mask = seg == box_body_id
        assert np.any(box_mask), "Box not visible in segmentation — check camera pose"

        # Minimum depth on the box front face should match D_to_front within 10 %.
        min_depth = float(frame.depth[box_mask].min())
        tol = 0.10
        assert abs(min_depth - D_to_front) < tol * D_to_front, (
            f"Depth {min_depth:.3f} m deviates >10 % from expected {D_to_front:.3f} m"
        )
    finally:
        renderer.disconnect()


# ---------------------------------------------------------------------------
# Test 3: detect() returns a detection for the visible object
# ---------------------------------------------------------------------------

def test_detect_returns_object():
    """detect() finds the visible box, pixel is in-bounds, depth is within 15 % of GT.

    Camera is placed behind the arm's origin so arm geometry does not occlude the box.
    """
    BOX_HALF = 0.10
    BOX_POS = np.array([0.0, 0.0, 1.5])
    EYE = np.array([0.0, 0.0, -1.0])     # 1 m behind origin, arm is between 0 and ~0.3 m
    D_to_front = float(np.linalg.norm(BOX_POS - EYE)) - BOX_HALF  # 2.5 - 0.10 = 2.40 m

    scene = SimScene()
    scene.add(SceneObject(
        name="detect_box",
        shape="box",
        half_extents=[BOX_HALF, BOX_HALF, BOX_HALF],
        position=BOX_POS.tolist(),
        rgba=[0.0, 1.0, 0.0, 1.0],
    ))

    cam_pose = _camera_looking_at(EYE, BOX_POS)
    intrinsics = CameraIntrinsics.from_fov(320, 240, fov_y_deg=60.0)

    renderer = SimRenderer(gui=False)
    try:
        renderer.load_scene(scene)
        cam = SimCamera(renderer, intrinsics, pose_provider=lambda: cam_pose)

        dets = cam.detect()

        assert len(dets) >= 1, "detect() returned no detections"

        # Check that the first detection pixel is within image bounds.
        u, v = dets.pixels[0]
        W, H = intrinsics.width, intrinsics.height
        assert 0 <= u < W, f"centroid u={u:.1f} out of [0, {W})"
        assert 0 <= v < H, f"centroid v={v:.1f} out of [0, {H})"

        # Median depth should be within 15 % of the box-centre distance (not front face,
        # since detect() uses the median over the whole visible region including centre).
        D_to_centre = float(np.linalg.norm(BOX_POS - EYE))  # 2.5 m
        det_depth = float(dets.depths[0])
        tol = 0.15
        assert abs(det_depth - D_to_centre) < tol * D_to_centre, (
            f"Detection depth {det_depth:.3f} m deviates >15 % from expected centre "
            f"distance {D_to_centre:.3f} m"
        )
    finally:
        renderer.disconnect()
