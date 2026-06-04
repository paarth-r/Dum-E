"""Named end-effector poses (presets), persisted as JSON.

Poses are stored as ``[x, y, z, roll, pitch, yaw]`` (metres, radians) so the file is
human-readable and editable. Used by pose mode (pad save/recall) and the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dume import geometry as g

DEFAULT_STORE = Path.home() / ".dume" / "poses.json"

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
