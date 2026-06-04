"""Pure pose math — no hardware, no lerobot. Poses are 4x4 homogeneous transforms.

Conventions:
- A pose ``T`` is a (4, 4) float array; ``T[:3, 3]`` is position (metres),
  ``T[:3, :3]`` is rotation.
- RPY is XYZ-extrinsic roll/pitch/yaw in radians (scipy ``Rotation`` "xyz").
- Quaternions are ``[x, y, z, w]`` (scipy convention).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(position, rotation) -> np.ndarray:
    """Build a 4x4 transform from a 3-vector position and a 3x3 rotation matrix."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=float)
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


def transform_from_pos_rpy(position, rpy_rad) -> np.ndarray:
    """Build a 4x4 transform from position and XYZ-extrinsic roll/pitch/yaw (radians)."""
    rot = Rotation.from_euler("xyz", np.asarray(rpy_rad, dtype=float)).as_matrix()
    return make_transform(position, rot)


def position_of(T) -> np.ndarray:
    return np.asarray(T, dtype=float)[:3, 3].copy()


def rotation_of(T) -> np.ndarray:
    return np.asarray(T, dtype=float)[:3, :3].copy()


def rpy_of(T) -> np.ndarray:
    """Roll/pitch/yaw (radians, XYZ-extrinsic) of a transform's rotation."""
    return Rotation.from_matrix(rotation_of(T)).as_euler("xyz")


def quat_of(T) -> np.ndarray:
    """Quaternion ``[x, y, z, w]`` of a transform's rotation."""
    return Rotation.from_matrix(rotation_of(T)).as_quat()


def pose_to_xyzrpy(T) -> np.ndarray:
    """Flatten a pose to ``[x, y, z, roll, pitch, yaw]`` (m, radians)."""
    return np.concatenate([position_of(T), rpy_of(T)])


def xyzrpy_to_pose(xyzrpy) -> np.ndarray:
    v = np.asarray(xyzrpy, dtype=float)
    return transform_from_pos_rpy(v[:3], v[3:])


def interpolate_pose(T0, T1, s: float) -> np.ndarray:
    """Interpolate between two poses: linear on position, SLERP on rotation.

    ``s`` is clamped to ``[0, 1]``; ``s=0`` returns ``T0``, ``s=1`` returns ``T1``.
    """
    s = float(np.clip(s, 0.0, 1.0))
    p = (1.0 - s) * position_of(T0) + s * position_of(T1)
    r0 = Rotation.from_matrix(rotation_of(T0))
    r1 = Rotation.from_matrix(rotation_of(T1))
    # SLERP via relative rotation scaled by s (robust, no Slerp object needed).
    rel = (r0.inv() * r1).as_rotvec()
    rot = (r0 * Rotation.from_rotvec(rel * s)).as_matrix()
    return make_transform(p, rot)


def pose_error(T_current, T_target) -> tuple[np.ndarray, float]:
    """Return (position error vector ``target - current``, rotation angle error in rad)."""
    dp = position_of(T_target) - position_of(T_current)
    r_cur = Rotation.from_matrix(rotation_of(T_current))
    r_tgt = Rotation.from_matrix(rotation_of(T_target))
    angle = float(np.linalg.norm((r_cur.inv() * r_tgt).as_rotvec()))
    return dp, angle


def position_distance(T0, T1) -> float:
    return float(np.linalg.norm(position_of(T1) - position_of(T0)))


def rotation_angle(T0, T1) -> float:
    """Geodesic angle (radians) between the rotations of two poses."""
    r0 = Rotation.from_matrix(rotation_of(T0))
    r1 = Rotation.from_matrix(rotation_of(T1))
    return float(np.linalg.norm((r0.inv() * r1).as_rotvec()))
