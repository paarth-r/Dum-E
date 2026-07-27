"""3D viewer for the accumulated point cloud, using PyBullet's GUI.

PyBullet rather than open3d or matplotlib because it is already a hard dependency — nothing
new to install — and because it can render the *arm* in the same frame as the cloud. Seeing
the robot alongside the points makes "is this cloud in the right place" answerable at a
glance, which is most of what we want from a viewer at this stage.

The world here is the arm base frame, matching the cloud. Zero is the base of the arm.
"""

from __future__ import annotations

import numpy as np

#: addUserDebugPoints is slow per call; batch rather than looping per point.
_BATCH = 8000


def height_colours(points: np.ndarray) -> np.ndarray:
    """Colour by height so structure reads without shading. Blue = low, red = high."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros((0, 3))
    z = pts[:, 2]
    lo, hi = float(z.min()), float(z.max())
    t = np.zeros_like(z) if hi - lo < 1e-9 else (z - lo) / (hi - lo)
    return np.column_stack([t, np.zeros_like(t) + 0.25, 1.0 - t])


class CloudView:
    """PyBullet GUI showing the arm at live joint angles plus the accumulated cloud."""

    def __init__(self, urdf_path: str, joint_names: list[str], *, point_size: float = 3.0):
        import pybullet as p
        import pybullet_data

        self.p = p
        self.point_size = point_size
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # Base frame at the origin, fixed — the cloud is expressed in exactly this frame.
        self.robot = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True)
        self._joint_index = {}
        for i in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, i)[1].decode()
            self._joint_index[name] = i
        self.joint_names = list(joint_names)
        self._items: list[int] = []
        p.resetDebugVisualizerCamera(0.9, 50, -30, [0.2, 0.0, 0.15])

    def set_joints(self, joints_deg) -> None:
        """Pose the rendered arm. Feed MEASURED joints so the render matches reality."""
        for name, value in zip(self.joint_names, np.asarray(joints_deg, dtype=float)):
            idx = self._joint_index.get(name)
            if idx is not None:
                self.p.resetJointState(self.robot, idx, np.deg2rad(value))

    def set_points(self, points: np.ndarray) -> None:
        """Replace the drawn cloud. Cheaper to clear and redraw than to diff."""
        for item in self._items:
            self.p.removeUserDebugItem(item)
        self._items = []
        pts = np.asarray(points, dtype=float)
        if not len(pts):
            return
        cols = height_colours(pts)
        for start in range(0, len(pts), _BATCH):
            chunk = pts[start:start + _BATCH]
            self._items.append(
                self.p.addUserDebugPoints(
                    chunk.tolist(),
                    cols[start:start + _BATCH].tolist(),
                    pointSize=self.point_size,
                )
            )

    def step(self) -> None:
        """Keep the GUI responsive without advancing any physics (there is none here)."""
        self.p.getCameraImage  # no-op attribute touch; GUI pumps on its own thread

    def close(self) -> None:
        try:
            self.p.disconnect(self.client)
        except Exception:  # noqa: BLE001 — viewer teardown must never mask a real error
            pass

    def __enter__(self) -> "CloudView":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
