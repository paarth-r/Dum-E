"""Tests for dume.flown_stereo — synthetic two-view geometry, no sim/PyBullet.

All camera poses are constructed analytically so that +z (optical forward)
points toward the scene origin.  The tests exercise the full pipeline:
relative pose → triangulation → detection matching → grasp proposal.
"""

from __future__ import annotations

import numpy as np
import pytest

from dume.camera import CameraIntrinsics, Detections, project_points, world_to_camera
from dume.geometry import make_transform
from dume.flown_stereo import (
    Grasp,
    propose_grasp,
    relative_pose,
    triangulate,
    triangulate_detections,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_K() -> np.ndarray:
    return CameraIntrinsics.from_fov(640, 480, fov_y_deg=60.0).K


def _look_at_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Build a camera-in-world pose (4x4) such that the camera sits at ``eye``
    and its +z optical axis points toward ``target``.

    +y is chosen to be world-up (or world-z-up rotated so that optical +y
    points downward in world space, i.e. world -z).  We construct a right-
    handed optical frame:
        z_cam = normalize(target - eye)        (forward)
        x_cam = normalize(z_cam × world_up)    (right)
        y_cam = z_cam × x_cam                  (down in image)
    where world_up = [0, 0, 1].  The rotation matrix R_wc has columns
    [x_cam, y_cam, z_cam] (camera-frame axes expressed in world).
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    z_cam = target - eye
    z_cam /= np.linalg.norm(z_cam)

    world_up = np.array([0.0, 0.0, 1.0])
    # Handle degenerate case where z_cam is parallel to world_up.
    if abs(np.dot(z_cam, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    x_cam = np.cross(z_cam, world_up)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)   # already unit length

    R_wc = np.column_stack([x_cam, y_cam, z_cam])   # (3, 3)
    return make_transform(eye, R_wc)


# Two camera positions looking at the origin, separated by ~0.3 m baseline.
_EYE_A = np.array([0.0, 0.3, 0.5])
_EYE_B = np.array([0.3, 0.3, 0.5])
_TARGET = np.array([0.0, 0.0, 0.0])

_T_A = _look_at_pose(_EYE_A, _TARGET)
_T_B = _look_at_pose(_EYE_B, _TARGET)

# A few known world points in front of both cameras (roughly around the origin,
# well within the field of view of both).
_WORLD_POINTS = np.array([
    [0.00,  0.00, 0.00],
    [0.05,  0.02, 0.01],
    [-0.03, 0.04, 0.02],
    [0.01, -0.02, 0.03],
], dtype=float)


def _project_both(world_pts: np.ndarray, K: np.ndarray):
    """Return pixel arrays (N,2) for camA and camB."""
    cam_a = world_to_camera(world_pts, _T_A)
    cam_b = world_to_camera(world_pts, _T_B)
    pix_a = project_points(cam_a, K)
    pix_b = project_points(cam_b, K)
    return pix_a, pix_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRelativePose:
    def test_relative_pose_identity(self):
        """relative_pose(T, T) must be the 4x4 identity."""
        result = relative_pose(_T_A, _T_A)
        assert np.allclose(result, np.eye(4), atol=1e-10)

    def test_relative_pose_translation(self):
        """Two poses differing only by a known translation → recovered correctly."""
        t_offset = np.array([0.1, -0.05, 0.2])
        T_base = make_transform([0.0, 0.0, 1.0], np.eye(3))
        T_shifted = make_transform([0.0, 0.0, 1.0] + t_offset, np.eye(3))

        result = relative_pose(T_base, T_shifted)

        # Rotation part should be identity.
        assert np.allclose(result[:3, :3], np.eye(3), atol=1e-10)
        # Translation should equal the offset.
        assert np.allclose(result[:3, 3], t_offset, atol=1e-10)

    def test_relative_pose_inverse_consistency(self):
        """relative_pose(A,B) and relative_pose(B,A) must be inverses."""
        T_AB = relative_pose(_T_A, _T_B)
        T_BA = relative_pose(_T_B, _T_A)
        assert np.allclose(T_AB @ T_BA, np.eye(4), atol=1e-10)


class TestTriangulate:
    def test_triangulate_recovers_known_points(self):
        """Noiseless projection then triangulation must recover the original points."""
        K = make_K()
        pix_a, pix_b = _project_both(_WORLD_POINTS, K)
        recovered = triangulate(pix_a, pix_b, _T_A, _T_B, K)
        assert recovered.shape == _WORLD_POINTS.shape
        assert np.allclose(recovered, _WORLD_POINTS, atol=1e-3), (
            f"Max error: {np.abs(recovered - _WORLD_POINTS).max():.6f} m"
        )

    def test_triangulate_robust_to_small_noise(self):
        """0.5 px gaussian pixel noise should produce < 1 cm world error."""
        rng = np.random.default_rng(42)
        K = make_K()
        pix_a, pix_b = _project_both(_WORLD_POINTS, K)
        noise_a = rng.normal(0, 0.5, pix_a.shape)
        noise_b = rng.normal(0, 0.5, pix_b.shape)
        recovered = triangulate(pix_a + noise_a, pix_b + noise_b, _T_A, _T_B, K)
        max_err = float(np.abs(recovered - _WORLD_POINTS).max())
        assert max_err < 0.01, f"Max error with noise: {max_err:.4f} m (expected < 0.01)"

    def test_triangulate_single_point(self):
        """Single-point triangulation must work (no off-by-one in loop)."""
        K = make_K()
        pt = np.array([[0.02, -0.01, 0.03]])
        pix_a, pix_b = _project_both(pt, K)
        recovered = triangulate(pix_a, pix_b, _T_A, _T_B, K)
        assert np.allclose(recovered, pt, atol=1e-3)


class TestTriangulateDetections:
    def test_matches_by_id_and_order(self):
        """Detection rows can be in any order; matching must be by id."""
        K = make_K()
        pix_a, pix_b = _project_both(_WORLD_POINTS, K)

        # Build detections with ids 10, 20, 30, 40.
        ids_all = [10, 20, 30, 40]
        detsA = Detections(ids=ids_all.copy(), pixels=pix_a.copy())

        # Shuffle B's order and drop id=30 to test partial overlap.
        order_b = [3, 0, 1]     # ids 40, 10, 20 (skip 30)
        detsB = Detections(
            ids=[ids_all[i] for i in order_b],
            pixels=pix_b[order_b],
        )

        shared_ids, pts = triangulate_detections(detsA, detsB, _T_A, _T_B, K)

        assert shared_ids == [10, 20, 40]
        # Verify each recovered point matches the original world point.
        for out_id, pt in zip(shared_ids, pts):
            idx = ids_all.index(out_id)
            assert np.allclose(pt, _WORLD_POINTS[idx], atol=1e-3), (
                f"id={out_id}: expected {_WORLD_POINTS[idx]}, got {pt}"
            )

    def test_empty_overlap_returns_empty(self):
        """No shared ids → empty result."""
        K = make_K()
        pix_a, pix_b = _project_both(_WORLD_POINTS[:2], K)
        detsA = Detections(ids=[1, 2], pixels=pix_a)
        detsB = Detections(ids=[3, 4], pixels=pix_b)
        shared_ids, pts = triangulate_detections(detsA, detsB, _T_A, _T_B, K)
        assert shared_ids == []
        assert pts.shape == (0, 3)


class TestProposeGrasp:
    def test_centroid_and_basics(self):
        """Grasp position ≈ centroid; width > 0; approach is unit norm."""
        pts = np.array([
            [0.1, 0.2, 0.05],
            [0.12, 0.21, 0.04],
            [0.09, 0.19, 0.06],
            [0.11, 0.20, 0.05],
        ])
        g = propose_grasp(pts)
        expected_centroid = pts.mean(axis=0)
        assert np.allclose(g.position, expected_centroid, atol=1e-9)
        assert g.width > 0.0
        assert abs(np.linalg.norm(g.approach) - 1.0) < 1e-9

    def test_flat_cluster_approaches_from_above(self):
        """Flat (z-thin) cluster → approach should be [0, 0, -1]."""
        rng = np.random.default_rng(7)
        xy = rng.uniform(-0.05, 0.05, (20, 2))
        z = rng.uniform(0.0, 0.002, (20, 1))     # very thin in z
        pts = np.hstack([xy, z])
        g = propose_grasp(pts)
        assert np.allclose(g.approach, [0.0, 0.0, -1.0], atol=1e-9)

    def test_tall_cluster_approaches_horizontally(self):
        """Tall narrow cluster → approach is horizontal (z-component ≈ 0)."""
        rng = np.random.default_rng(13)
        # Very tall (z extent 0.5 m) and narrow (0.005 m in x and y).
        x = rng.uniform(-0.002, 0.002, (30, 1))
        y = rng.uniform(-0.002, 0.002, (30, 1))
        z = np.linspace(0.0, 0.5, 30)[:, None]
        pts = np.hstack([x, y, z])
        g = propose_grasp(pts)
        # Approach should be roughly horizontal (z component close to 0).
        assert abs(g.approach[2]) < 0.15, (
            f"Expected horizontal approach, got {g.approach}"
        )
        assert abs(np.linalg.norm(g.approach) - 1.0) < 1e-9

    def test_single_point(self):
        """Single-point cloud should not crash and return sensible defaults."""
        g = propose_grasp(np.array([[0.1, 0.2, 0.3]]))
        assert np.allclose(g.position, [0.1, 0.2, 0.3])
        assert abs(np.linalg.norm(g.approach) - 1.0) < 1e-9
        assert g.width > 0.0
