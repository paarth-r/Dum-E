"""Sparse point cloud from flown extrinsics: match features across views, triangulate, filter.

The missing half of ``flown_stereo``. That module triangulates *given* correspondences; the only
producer of those today matches by object id, which exists solely because ``SimCamera`` invents
them. Real imagery has no ids, so this module supplies correspondence: ORB features matched
between successive keyframes.

Pure numpy + cv2 — no hardware, no GUI, no PyBullet — so it is testable on synthetic scenes.

Output points are in the **arm base frame**. Nothing here re-bases them: ``triangulate`` returns
points in whatever frame the camera poses were expressed in, and those poses come from FK, which
is base-framed. Zero is the base of the arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dume import geometry as g
from dume.camera import project_points, world_to_camera
from dume.flown_stereo import triangulate


@dataclass
class Keyframe:
    """One posed view, with its features already extracted."""

    pose: np.ndarray  # 4x4 camera-in-base
    keypoints: tuple
    descriptors: np.ndarray | None


def detect_features(image: np.ndarray, max_features: int = 3000) -> Keyframe:
    """ORB keypoints + descriptors. ORB because it is fast, free, and binary-matchable."""
    import cv2

    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp, desc = orb.detectAndCompute(img, None)
    return Keyframe(pose=np.eye(4), keypoints=tuple(kp), descriptors=desc)


def match_features(desc_a, desc_b, ratio: float = 0.75) -> list[tuple[int, int]]:
    """Lowe-ratio-tested mutual matches, returned as ``(index_a, index_b)`` pairs.

    The ratio test is what keeps this honest: a descriptor whose best match is barely better
    than its second-best is ambiguous, and an ambiguous match triangulates to a confident,
    completely wrong 3D point. Rejecting them costs density and buys correctness.
    """
    import cv2

    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)
    out = []
    for group in pairs:
        if len(group) < 2:
            continue
        best, second = group
        if best.distance < ratio * second.distance:
            out.append((best.queryIdx, best.trainIdx))
    return out


def filter_points(
    points: np.ndarray,
    pix_a: np.ndarray,
    pix_b: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    K: np.ndarray,
    *,
    max_reproj_px: float = 3.0,
    min_range: float = 0.03,
    max_range: float = 2.0,
) -> np.ndarray:
    """Boolean mask of points worth keeping.

    Three independent rejections, each catching a different way DLT lies:

    1. **Cheirality** — a point behind either camera is geometrically impossible; unconstrained
       DLT will happily return one from a bad match.
    2. **Reprojection** — project the solution back into both views; a good point lands on the
       pixels that produced it. This is the main filter, and it catches mismatches directly.
    3. **Range gate** — near-parallel rays from a noisy match triangulate to enormous distances.
       Without this, a handful of bad matches dominate the cloud's extent and wreck any view.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    cam_a = world_to_camera(points, pose_a)
    cam_b = world_to_camera(points, pose_b)
    in_front = (cam_a[:, 2] > 1e-6) & (cam_b[:, 2] > 1e-6)

    keep = in_front.copy()
    if keep.any():
        err_a = np.full(len(points), np.inf)
        err_b = np.full(len(points), np.inf)
        idx = np.flatnonzero(in_front)
        err_a[idx] = np.linalg.norm(project_points(cam_a[idx], K) - pix_a[idx], axis=1)
        err_b[idx] = np.linalg.norm(project_points(cam_b[idx], K) - pix_b[idx], axis=1)
        keep &= (err_a < max_reproj_px) & (err_b < max_reproj_px)

    depth = np.linalg.norm(cam_a, axis=1)
    keep &= (depth > min_range) & (depth < max_range)
    return keep


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    """One point per occupied voxel — bounds memory and evens out density."""
    if voxel <= 0 or len(points) == 0:
        return points
    keys = np.floor(np.asarray(points) / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


class CloudBuilder:
    """Accumulates a base-frame point cloud from posed frames, one keyframe pair at a time.

    Frames are only promoted to keyframes once the camera has moved far enough. Triangulation
    from a near-zero baseline is ill-conditioned: the two rays are nearly parallel, so small
    pixel noise moves the intersection enormously. Feeding every frame in would not densify the
    cloud, it would flood it with garbage at huge range. ``min_baseline`` is the single most
    important quality knob here.
    """

    def __init__(
        self,
        K: np.ndarray,
        *,
        min_baseline: float = 0.02,
        ratio: float = 0.75,
        max_reproj_px: float = 3.0,
        min_range: float = 0.03,
        max_range: float = 2.0,
        voxel: float = 0.003,
        max_points: int = 300_000,
        max_features: int = 3000,
    ):
        self.K = np.asarray(K, dtype=float)
        self.min_baseline = float(min_baseline)
        self.ratio = float(ratio)
        self.max_reproj_px = float(max_reproj_px)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.voxel = float(voxel)
        self.max_points = int(max_points)
        self.max_features = int(max_features)
        self._last: Keyframe | None = None
        self._points = np.zeros((0, 3), dtype=float)
        self.keyframes = 0
        self.last_matches = 0
        self.last_kept = 0

    @property
    def points(self) -> np.ndarray:
        """(N, 3) accumulated points, in the arm base frame."""
        return self._points

    def add(self, image: np.ndarray, pose: np.ndarray) -> int:
        """Offer a posed frame. Returns how many points it contributed (0 if skipped)."""
        pose = np.asarray(pose, dtype=float)
        kf = detect_features(image, self.max_features)
        kf.pose = pose

        if self._last is None:
            self._last, self.keyframes = kf, 1
            return 0

        baseline = g.position_distance(self._last.pose, pose)
        if baseline < self.min_baseline:
            return 0  # too little parallax to triangulate anything trustworthy

        added = self._triangulate_against(self._last, kf)
        self._last, self.keyframes = kf, self.keyframes + 1
        return added

    def _triangulate_against(self, a: Keyframe, b: Keyframe) -> int:
        matches = match_features(a.descriptors, b.descriptors, self.ratio)
        self.last_matches = len(matches)
        self.last_kept = 0
        if not matches:
            return 0

        pix_a = np.array([a.keypoints[i].pt for i, _ in matches], dtype=float)
        pix_b = np.array([b.keypoints[j].pt for _, j in matches], dtype=float)
        pts = triangulate(pix_a, pix_b, a.pose, b.pose, self.K)

        keep = filter_points(
            pts, pix_a, pix_b, a.pose, b.pose, self.K,
            max_reproj_px=self.max_reproj_px,
            min_range=self.min_range,
            max_range=self.max_range,
        )
        kept = pts[keep]
        self.last_kept = len(kept)
        if not len(kept):
            return 0

        self._points = voxel_downsample(np.vstack([self._points, kept]), self.voxel)
        if len(self._points) > self.max_points:
            self._points = self._points[-self.max_points:]
        return len(kept)
