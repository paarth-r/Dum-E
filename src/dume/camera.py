"""Camera data types and the end-effector-mounted-camera geometry.

This module is pure geometry — no PyBullet, no hardware. It defines the observation types a
:class:`CameraSource` produces and the one piece of arm-specific knowledge that makes the
"flown extrinsics" idea work: where the camera sits relative to the gripper.

``camera_pose_from_fk`` is the crux. The SO-101 already knows where its own hand is (via FK),
so multiplying the gripper pose by a fixed mount transform gives the camera's world pose with
no rig calibration. Snapshot from two arm configurations and you have a known-baseline stereo
pair (see :mod:`dume.flown_stereo`).

Conventions:
- Poses are 4x4 homogeneous transforms in the arm base frame (see :mod:`dume.geometry`).
- The camera optical frame is OpenCV convention: +z forward (into the scene), +x right, +y down.
- Pixels are ``(u, v)`` = (column, row), origin top-left.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from dume import geometry as g

# ---------------------------------------------------------------------------
# Mount extrinsic (gripper_frame_link -> camera optical frame).
#
# Read from the URDF's `camera_optical_link` rather than hardcoded here, so the frame has
# exactly one definition. A duplicated constant diverges the moment the mount is re-measured,
# and the divergence is invisible: nothing crashes, the point cloud just lands in the wrong
# place. See that link in urdf/so101_new_calib.urdf for how the numbers were derived.
#
# The path is computed locally rather than imported from `kinematics`, which would drag placo
# and lerobot into this otherwise-pure-geometry module.
# ---------------------------------------------------------------------------
DEFAULT_URDF = Path(__file__).resolve().parents[2] / "urdf" / "so101_new_calib.urdf"
CAMERA_FRAME = "camera_optical_link"


def mount_from_urdf(urdf_path=DEFAULT_URDF, child: str = CAMERA_FRAME) -> np.ndarray:
    """Return the 4x4 transform of the fixed joint whose child link is ``child``.

    Raises rather than falling back to a default: the camera frame is hand-authored, and
    re-running onshape-to-robot silently drops it. A missing frame must be loud, because a
    guessed extrinsic produces a plausible-looking reconstruction that is simply wrong.
    """
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        link = joint.find("child")
        if link is None or link.get("link") != child:
            continue
        origin = joint.find("origin")
        xyz = (origin.get("xyz") if origin is not None else None) or "0 0 0"
        rpy = (origin.get("rpy") if origin is not None else None) or "0 0 0"
        return g.transform_from_pos_rpy(
            [float(v) for v in xyz.split()], [float(v) for v in rpy.split()]
        )
    raise ValueError(
        f"No fixed joint with child link {child!r} in {urdf_path}. The end-effector camera "
        "frame is hand-authored; re-exporting the URDF from onshape-to-robot removes it."
    )


T_CAM_MOUNT: np.ndarray = mount_from_urdf()


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics. ``K`` is the standard 3x3 calibration matrix."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fov_y_deg: float = 60.0) -> "CameraIntrinsics":
        """Build intrinsics from a vertical field of view (matches PyBullet's projection)."""
        fy = (height / 2.0) / np.tan(np.deg2rad(fov_y_deg) / 2.0)
        fx = fy  # square pixels
        return cls(fx=fx, fy=fy, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float
        )


@dataclass
class Detections:
    """Objects seen in a frame. Parallel arrays, one row per detection."""

    ids: list[int] = field(default_factory=list)
    pixels: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (N, 2) u,v
    depths: np.ndarray = field(default_factory=lambda: np.zeros((0,)))  # (N,) metres, optional

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class CameraFrame:
    """One capture: optional RGB + depth, the camera pose it was taken from, and a timestamp."""

    pose: np.ndarray  # 4x4 camera-in-base
    rgb: np.ndarray | None = None  # (H, W, 3) uint8
    depth: np.ndarray | None = None  # (H, W) float metres
    t: float = 0.0


@runtime_checkable
class CameraSource(Protocol):
    """Anything that yields camera frames + detections (sim, or a real Arducam later)."""

    intrinsics: CameraIntrinsics

    def capture(self) -> CameraFrame: ...
    def detect(self) -> Detections: ...


def camera_pose_from_fk(kin, joints_deg, mount: np.ndarray = T_CAM_MOUNT) -> np.ndarray:
    """World pose (4x4) of the camera optical frame for a given arm configuration.

    ``kin`` is a :class:`dume.kinematics.Kinematics`; ``joints_deg`` the 6-vector. This is the
    whole flown-extrinsics trick: the camera pose is derived from forward kinematics, so two
    snapshots from two configurations form a calibrated stereo pair.
    """
    return kin.fk(joints_deg) @ mount


def project_points(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D points (in the camera optical frame) to pixels via the pinhole model.

    ``points_cam`` is (N, 3). Returns (N, 2) ``(u, v)``. Points must have positive z (in front).
    """
    pts = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    z = pts[:, 2]
    uv = (K @ pts.T).T
    return uv[:, :2] / z[:, None]


def world_to_camera(points_world: np.ndarray, cam_pose: np.ndarray) -> np.ndarray:
    """Transform world points (N, 3) into the camera optical frame given the camera pose."""
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    R = cam_pose[:3, :3]
    t = cam_pose[:3, 3]
    return (R.T @ (pts - t).T).T
