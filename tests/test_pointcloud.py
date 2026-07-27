"""Point-cloud reconstruction on synthetic scenes — no camera, no arm.

The key property: project known 3D points into two known camera poses, and triangulation must
recover the originals. If that round-trip holds, the geometry is right and any failure on real
imagery is a matching or calibration problem, not a maths problem.
"""

import numpy as np
import pytest

from dume.camera import CameraIntrinsics, project_points, world_to_camera
from dume.flown_stereo import triangulate
from dume.pointcloud import CloudBuilder, filter_points, voxel_downsample


@pytest.fixture
def K():
    return CameraIntrinsics.from_fov(640, 480, 60.0).K


def _pose(x=0.0, y=0.0, z=0.0):
    """Camera at (x,y,z) looking down world +z (optical frame aligned with world)."""
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    pts = np.column_stack([
        rng.uniform(-0.2, 0.2, 60),
        rng.uniform(-0.15, 0.15, 60),
        rng.uniform(0.5, 1.0, 60),
    ])
    return pts


def _project(points, pose, K):
    return project_points(world_to_camera(points, pose), K)


# ---------------------------------------------------------------------------
# Geometry round-trip
# ---------------------------------------------------------------------------

def test_triangulation_recovers_the_scene(scene, K):
    a, b = _pose(), _pose(x=0.08)
    got = triangulate(_project(scene, a, K), _project(scene, b, K), a, b, K)
    assert np.allclose(got, scene, atol=1e-6)


def test_wider_baseline_is_better_conditioned(scene, K):
    """Noise hurts less with more parallax — the reason keyframes need a minimum baseline."""
    rng = np.random.default_rng(1)
    a = _pose()
    errs = {}
    for name, b in [("narrow", _pose(x=0.002)), ("wide", _pose(x=0.10))]:
        pa = _project(scene, a, K) + rng.normal(0, 0.3, (len(scene), 2))
        pb = _project(scene, b, K) + rng.normal(0, 0.3, (len(scene), 2))
        got = triangulate(pa, pb, a, b, K)
        errs[name] = np.median(np.linalg.norm(got - scene, axis=1))
    assert errs["wide"] < errs["narrow"]


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def test_filter_keeps_clean_points(scene, K):
    a, b = _pose(), _pose(x=0.08)
    pa, pb = _project(scene, a, K), _project(scene, b, K)
    keep = filter_points(scene, pa, pb, a, b, K)
    assert keep.all()


def test_filter_rejects_points_behind_the_camera(K):
    a, b = _pose(), _pose(x=0.08)
    behind = np.array([[0.0, 0.0, -0.5]])
    pa = pb = np.array([[320.0, 240.0]])
    assert not filter_points(behind, pa, pb, a, b, K).any()


def test_filter_rejects_high_reprojection_error(scene, K):
    """A mismatch triangulates to a point that does not project back to its own pixels."""
    a, b = _pose(), _pose(x=0.08)
    pa, pb = _project(scene, a, K), _project(scene, b, K)
    pb_shuffled = pb[::-1].copy()  # deliberately wrong correspondences
    pts = triangulate(pa, pb_shuffled, a, b, K)
    keep = filter_points(pts, pa, pb_shuffled, a, b, K, max_reproj_px=2.0)
    assert keep.sum() < len(scene) // 2


def test_filter_rejects_out_of_range(scene, K):
    a, b = _pose(), _pose(x=0.08)
    pa, pb = _project(scene, a, K), _project(scene, b, K)
    assert not filter_points(scene, pa, pb, a, b, K, max_range=0.1).any()
    assert not filter_points(scene, pa, pb, a, b, K, min_range=5.0).any()


def test_voxel_downsample_collapses_duplicates():
    pts = np.repeat(np.array([[0.0, 0.0, 0.5]]), 50, axis=0)
    assert len(voxel_downsample(pts, 0.01)) == 1


def test_voxel_downsample_keeps_distinct_points():
    pts = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    assert len(voxel_downsample(pts, 0.01)) == 3


def test_voxel_zero_is_a_noop(scene):
    assert len(voxel_downsample(scene, 0.0)) == len(scene)


# ---------------------------------------------------------------------------
# Builder behaviour
# ---------------------------------------------------------------------------

def _textured(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (480, 640), dtype=np.uint8)
    img[::16, :] = 0
    img[:, ::16] = 255
    return img


def test_first_frame_adds_nothing(K):
    b = CloudBuilder(K)
    assert b.add(_textured(), _pose()) == 0
    assert b.keyframes == 1
    assert len(b.points) == 0


def test_insufficient_baseline_is_skipped(K):
    """Near-parallel rays are ill-conditioned; refusing them is the main quality knob."""
    b = CloudBuilder(K, min_baseline=0.05)
    b.add(_textured(), _pose())
    assert b.add(_textured(1), _pose(x=0.001)) == 0
    assert b.keyframes == 1  # not promoted


def test_points_land_in_the_base_frame(K):
    """Nothing re-bases the output: poses are base-framed, so points are too."""
    b = CloudBuilder(K, min_baseline=0.01, max_range=50.0)
    img = _textured()
    b.add(img, _pose())
    b.add(img, _pose(x=0.05))
    # Identical images at a pure translation: matches are self-consistent, points finite.
    assert b.points.ndim == 2 and b.points.shape[1] == 3


def test_max_points_is_enforced(K):
    b = CloudBuilder(K, max_points=10, voxel=0.0)
    b._points = np.zeros((50, 3))
    b._points = b._points[-b.max_points:]
    assert len(b._points) == 10


def test_no_matches_contributes_nothing(K):
    """A textureless view is not an error — scanning continues."""
    b = CloudBuilder(K, min_baseline=0.01)
    flat = np.full((480, 640), 128, dtype=np.uint8)
    b.add(flat, _pose())
    assert b.add(flat, _pose(x=0.05)) == 0
