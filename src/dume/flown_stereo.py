"""Flown-extrinsics depth and grasp core.

The SO-101 arm moves its end-effector-mounted camera to two configurations.
Forward kinematics gives both camera poses (camera-in-world 4x4) with a known
baseline — no rig calibration needed. This module triangulates object geometry
from the two views and proposes a grasp.

All geometry is pure numpy/scipy — no PyBullet, no hardware, fully testable.

Conventions (match camera.py):
- Poses are 4x4 homogeneous transforms: T[:3,:3] = rotation, T[:3,3] = position.
- Camera optical frame: +z forward, +x right, +y down (OpenCV).
- Pixels are (u, v) = (column, row), origin top-left.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 1. Relative pose
# ---------------------------------------------------------------------------

def relative_pose(T_world_camA: np.ndarray, T_world_camB: np.ndarray) -> np.ndarray:
    """Return the pose of camB expressed in camA's frame.

    ``result = inv(T_world_camA) @ T_world_camB``

    This is exactly the "baseline" you get for free from FK: snap two arm
    configurations, compute both camera-in-world poses, and their relative
    transform is a fully calibrated extrinsic.

    Parameters
    ----------
    T_world_camA, T_world_camB:
        4x4 camera-in-world poses (from FK; see :func:`camera.camera_pose_from_fk`).

    Returns
    -------
    (4, 4) pose of camB in camA's frame.
    """
    T_A = np.asarray(T_world_camA, dtype=float)
    T_B = np.asarray(T_world_camB, dtype=float)
    return np.linalg.inv(T_A) @ T_B


# ---------------------------------------------------------------------------
# 2. Triangulation (DLT)
# ---------------------------------------------------------------------------

def _build_projection(K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Build the 3x4 projection matrix P = K @ [R_cw | t_cw].

    The world-to-camera extrinsic is derived from the camera-in-world pose:
      R_cw = R_wc.T
      t_cw = -R_wc.T @ t_wc
    which matches how ``world_to_camera`` in camera.py works.
    """
    R_wc = T_world_cam[:3, :3]
    t_wc = T_world_cam[:3, 3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    Rt = np.hstack([R_cw, t_cw[:, None]])   # (3, 4)
    return K @ Rt                            # (3, 4)


def triangulate(
    pixels_a: np.ndarray,
    pixels_b: np.ndarray,
    T_world_camA: np.ndarray,
    T_world_camB: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Triangulate matched pixel correspondences to world 3D points (DLT).

    For each correspondence (u_a, v_a) <-> (u_b, v_b), we build a 4x4
    homogeneous linear system from the two camera projection matrices and solve
    it via SVD.  The 3D point X in world coordinates satisfies:

        λ_a [u_a, v_a, 1]^T  =  P_a @ X_h
        λ_b [u_b, v_b, 1]^T  =  P_b @ X_h

    Cross-multiplying each equation with the image point eliminates the scale
    and yields two independent linear constraints per view → four rows total.
    The solution is the right-singular vector corresponding to the smallest
    singular value; we then dehomogenize.

    Parameters
    ----------
    pixels_a, pixels_b : (N, 2) float arrays of (u, v) correspondences.
    T_world_camA, T_world_camB : 4x4 camera-in-world poses.
    K : 3x3 intrinsic matrix (same for both views).

    Returns
    -------
    (N, 3) world points.
    """
    pix_a = np.asarray(pixels_a, dtype=float).reshape(-1, 2)
    pix_b = np.asarray(pixels_b, dtype=float).reshape(-1, 2)
    N = pix_a.shape[0]

    P_a = _build_projection(K, T_world_camA)  # (3, 4)
    P_b = _build_projection(K, T_world_camB)  # (3, 4)

    points = np.empty((N, 3), dtype=float)
    for i in range(N):
        u_a, v_a = pix_a[i]
        u_b, v_b = pix_b[i]

        # Each row: x * P[2,:] - P[0,:] = 0, etc.
        A = np.array([
            u_a * P_a[2, :] - P_a[0, :],
            v_a * P_a[2, :] - P_a[1, :],
            u_b * P_b[2, :] - P_b[0, :],
            v_b * P_b[2, :] - P_b[1, :],
        ])  # (4, 4)

        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]          # homogeneous world point
        points[i] = X_h[:3] / X_h[3]

    return points


# ---------------------------------------------------------------------------
# 3. Triangulate matched detections
# ---------------------------------------------------------------------------

def triangulate_detections(
    detsA: object,
    detsB: object,
    T_world_camA: np.ndarray,
    T_world_camB: np.ndarray,
    K: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    """Triangulate matched detections (by object id) across two views.

    Finds the intersection of ``detsA.ids`` and ``detsB.ids``, aligns the
    corresponding pixel rows, and calls :func:`triangulate`.

    Parameters
    ----------
    detsA, detsB : :class:`camera.Detections`-compatible objects.
        Must have ``.ids`` (list[int]) and ``.pixels`` ((N,2) ndarray).
    T_world_camA, T_world_camB : 4x4 camera-in-world poses.
    K : 3x3 intrinsic matrix.

    Returns
    -------
    ids : list[int]
        Matched object ids, in ascending order.
    points : (M, 3) ndarray
        World coordinates, one row per id.
    """
    ids_a: list[int] = list(detsA.ids)
    ids_b: list[int] = list(detsB.ids)
    pix_a = np.asarray(detsA.pixels, dtype=float)
    pix_b = np.asarray(detsB.pixels, dtype=float)

    shared = sorted(set(ids_a) & set(ids_b))
    if not shared:
        return [], np.zeros((0, 3), dtype=float)

    idx_a = [ids_a.index(id_) for id_ in shared]
    idx_b = [ids_b.index(id_) for id_ in shared]

    matched_a = pix_a[idx_a]
    matched_b = pix_b[idx_b]

    pts = triangulate(matched_a, matched_b, T_world_camA, T_world_camB, K)
    return shared, pts


# ---------------------------------------------------------------------------
# 4. Grasp dataclass
# ---------------------------------------------------------------------------

@dataclass
class Grasp:
    """Proposed grasp for a detected object cluster.

    Attributes
    ----------
    position : (3,) ndarray
        Grasp target in world coordinates (centroid of the point cluster).
    approach : (3,) ndarray
        Unit vector in the direction the gripper advances before closing.
        Defined as the vector from the gripper "far" side toward the object.
    width : float
        Gripper opening in metres (extent of the object along its minor axis).
    """
    position: np.ndarray
    approach: np.ndarray
    width: float


# ---------------------------------------------------------------------------
# 5. Heuristic grasp proposal
# ---------------------------------------------------------------------------

def propose_grasp(points_world: np.ndarray) -> Grasp:
    """Propose a grasp from a small point cloud cluster.

    Algorithm
    ---------
    1. Centroid → ``position``.
    2. PCA of the centred cloud (eigendecomposition of the 3x3 covariance).
       Eigenvectors are sorted by eigenvalue (ascending), so:
         - ``axes[0]`` = minor axis (smallest spread) — gripper closes across this.
         - ``axes[2]`` = major axis (largest spread).
    3. Width = 2 * standard deviation along the minor axis (covers ~95 % of
       the extent under a rough Gaussian assumption; clamped to ≥ 1 mm).
    4. Approach direction:
       - Default: straight down from above, ``[0, 0, -1]``.
         (Assumes the world z-axis points up and the arm approaches vertically.)
       - Override: if the cluster's height extent (z range) exceeds the width
         along the horizontal minor axis, approach from the side along the
         horizontal component of the minor axis (so the gripper closes along
         the cluster's narrowest horizontal direction).

    Parameters
    ----------
    points_world : (N, 3) ndarray, N ≥ 1.

    Returns
    -------
    :class:`Grasp`
    """
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    centroid = pts.mean(axis=0)
    position = centroid.copy()

    if len(pts) == 1:
        # Degenerate: single point, use defaults.
        return Grasp(position=position, approach=np.array([0.0, 0.0, -1.0]), width=0.01)

    centred = pts - centroid
    cov = centred.T @ centred / max(len(pts) - 1, 1)           # (3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)            # ascending eigenvalues
    # eigenvectors columns are eigenvectors; eigenvectors[:, i] = i-th eigenvec
    minor_axis = eigenvectors[:, 0]   # smallest spread
    # major_axis = eigenvectors[:, 2]  # largest spread

    # Width: 2-sigma along minor axis.
    projections = centred @ minor_axis
    sigma_minor = float(np.std(projections))
    width = max(2.0 * sigma_minor, 1e-3)

    # Horizontal spread: extent of the cluster in the xy-plane (world z up).
    xy_extents = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
    horiz_extent = float(xy_extents.max())

    # Height extent (z range).
    z_extent = float(pts[:, 2].max() - pts[:, 2].min())

    # Decide approach direction.
    # A cluster is "tall" only when its vertical extent meaningfully exceeds its
    # horizontal spread — not merely its minor-axis width, which can be tiny even
    # for flat pancake shapes.  Require z_extent > horiz_extent as a stricter guard.
    horiz_minor = minor_axis.copy()
    horiz_minor[2] = 0.0                     # flatten to horizontal
    horiz_minor_norm = np.linalg.norm(horiz_minor)

    if z_extent > horiz_extent and z_extent > width and horiz_minor_norm > 1e-6:
        # Tall, narrow cluster → approach horizontally along the minor axis.
        approach = horiz_minor / horiz_minor_norm
    else:
        # Flat/wide cluster → approach from above.
        approach = np.array([0.0, 0.0, -1.0])

    return Grasp(position=position, approach=approach, width=width)
