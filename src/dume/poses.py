"""Named poses (presets), persisted as JSON.

Two flavours, two files:

* :class:`PoseStore` — end-effector poses as ``[x, y, z, roll, pitch, yaw]`` (metres,
  radians). Used by pose mode (pad save/recall) and the CLI.
* :class:`JointPoseStore` — raw joint configurations (one entry per motor, in
  :data:`dume.arm.MOTOR_ORDER`). Used for the startup pose and any other named setpoints
  captured directly off the hardware.

Both store human-readable, editable JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dume import geometry as g
from dume.arm import MOTOR_ORDER

DEFAULT_STORE = Path.home() / ".dume" / "poses.json"
DEFAULT_JOINT_STORE = Path.home() / ".dume" / "joint_poses.json"

# A stable, well-conditioned "ready" joint configuration (degrees; gripper 0..100).
HOME_JOINTS = np.array([0.0, -20.0, 20.0, 0.0, 0.0, 50.0], dtype=float)


class PoseStore:
    def __init__(self, path: Path | str = DEFAULT_STORE):
        self.path = Path(path)
        self._poses: dict[str, np.ndarray] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._poses = {k: np.asarray(v, dtype=float) for k, v in data.items()}

    def save_to_disk(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.tolist() for k, v in self._poses.items()}
        self.path.write_text(json.dumps(data, indent=2))

    def names(self) -> list[str]:
        return sorted(self._poses)

    def has(self, name: str) -> bool:
        return name in self._poses

    def get_pose(self, name: str) -> np.ndarray:
        """Return the named pose as a 4x4 transform."""
        return g.xyzrpy_to_pose(self._poses[name])

    def set_pose(self, name: str, pose) -> None:
        """Store a 4x4 transform under ``name`` and persist."""
        self._poses[name] = g.pose_to_xyzrpy(pose)
        self.save_to_disk()


class JointPoseStore:
    """Named joint configurations (degrees + gripper 0..100), keyed by name and persisted.

    Each entry is a ``{motor: value}`` object in :data:`dume.arm.MOTOR_ORDER`, so the file
    stays readable and hand-editable. The default name is ``"start"`` (the pose ``dume run``
    moves to at startup); any other name is just another setpoint.
    """

    DEFAULT_NAME = "start"

    def __init__(self, path: Path | str = DEFAULT_JOINT_STORE):
        self.path = Path(path)
        self._poses: dict[str, np.ndarray] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._poses = {k: self._to_vec(v) for k, v in data.items()}

    @staticmethod
    def _to_vec(entry) -> np.ndarray:
        """Accept either a ``{motor: value}`` dict or a bare 6-list."""
        if isinstance(entry, dict):
            return np.array([float(entry[m]) for m in MOTOR_ORDER], dtype=float)
        return np.asarray(entry, dtype=float)

    def save_to_disk(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            name: {m: float(vec[i]) for i, m in enumerate(MOTOR_ORDER)}
            for name, vec in self._poses.items()
        }
        self.path.write_text(json.dumps(data, indent=2))

    def names(self) -> list[str]:
        return sorted(self._poses)

    def has(self, name: str) -> bool:
        return name in self._poses

    def get(self, name: str) -> np.ndarray:
        """Return the named joint vector (length 6, in ``MOTOR_ORDER``)."""
        return self._poses[name].copy()

    def set(self, name: str, joints) -> None:
        """Store a length-6 joint vector under ``name`` and persist."""
        vec = np.asarray(joints, dtype=float)
        if vec.shape != (len(MOTOR_ORDER),):
            raise ValueError(f"expected {len(MOTOR_ORDER)} joints, got shape {vec.shape}")
        self._poses[name] = vec.copy()
        self.save_to_disk()
