"""PyBullet-backed kinematic simulator and synthetic camera for the SO-101.

Three layers::

    SceneObject / SimScene  — pure data; describe what is in the world.
    SimRenderer             — owns a PyBullet client; loads the arm URDF and spawns
                              SceneObjects as static rigid bodies.
    SimCamera               — implements CameraSource; renders RGB + depth from the
                              current arm pose and returns Detections via segmentation.

All classes operate in PyBullet DIRECT (headless) mode by default so they work in CI.

Coordinate conventions follow :mod:`dume.camera`:
- Arm base frame for world coordinates.
- Camera optical frame: +z forward, +x right, +y down (OpenCV).

Usage example::

    renderer = SimRenderer()
    scene = SimScene()
    scene.add(SceneObject("target", "box", half_extents=[0.02,0.02,0.02],
                          position=[0.3, 0.0, 0.2]))
    renderer.load_scene(scene)

    intrinsics = CameraIntrinsics.from_fov(320, 240, fov_y_deg=60.0)
    cam = SimCamera(renderer, intrinsics, pose_provider=lambda: np.eye(4))
    frame = cam.capture()
    dets  = cam.detect()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pybullet as p

from dume.arm import MOTOR_ORDER
from dume.camera import CameraFrame, CameraIntrinsics, Detections
from dume.kinematics import DEFAULT_URDF

# ---------------------------------------------------------------------------
# Scene description
# ---------------------------------------------------------------------------

@dataclass
class SceneObject:
    """Declarative description of one rigid body in the simulation world.

    Parameters
    ----------
    name:
        Unique human-readable identifier; used to index detections.
    shape:
        ``"box"`` or ``"sphere"``.
    half_extents:
        For box: ``[hx, hy, hz]`` in metres.  Ignored for sphere.
    radius:
        For sphere: radius in metres.  Ignored for box.
    position:
        World-frame ``[x, y, z]`` in metres.
    rgba:
        ``[r, g, b, a]`` colour, each in ``[0, 1]``.  Defaults to opaque red.
    """

    name: str
    shape: str  # "box" | "sphere"
    half_extents: list[float] | np.ndarray = field(default_factory=lambda: [0.05, 0.05, 0.05])
    radius: float = 0.05
    position: list[float] | np.ndarray = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rgba: list[float] | np.ndarray = field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0])
    mass: float = 0.0  # 0 = static prop; >0 = dynamic (falls under gravity, grabbable)


class SimScene:
    """Ordered list of :class:`SceneObject` descriptions.  No PyBullet state."""

    def __init__(self) -> None:
        self._objects: list[SceneObject] = []

    def add(self, obj: SceneObject) -> None:
        """Append *obj* to the scene.  Raises if the name is already taken."""
        if any(o.name == obj.name for o in self._objects):
            raise ValueError(f"SceneObject name {obj.name!r} already in scene")
        self._objects.append(obj)

    @property
    def objects(self) -> list[SceneObject]:
        return list(self._objects)

    def __len__(self) -> int:
        return len(self._objects)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class SimRenderer:
    """Owns a PyBullet physics client and exposes the SO-101 arm + scene objects.

    Parameters
    ----------
    urdf_path:
        Path to the arm URDF.  Defaults to the vendored :data:`dume.kinematics.DEFAULT_URDF`.
    gui:
        If ``True`` open a PyBullet GUI window.  Use only interactively; tests must use
        the default ``False`` (DIRECT / headless).
    dynamic:
        If ``True`` enable gravity, a ground plane, and physics stepping so objects with mass
        fall and can be grasped (see :meth:`attach`).  Default ``False`` keeps the pure
        kinematic behaviour (arm teleported, props static) that the tests rely on.
    """

    def __init__(self, urdf_path: str = DEFAULT_URDF, gui: bool = False, dynamic: bool = False) -> None:
        self._client: int = p.connect(p.GUI if gui else p.DIRECT)
        self.dynamic = dynamic
        self._plane: int | None = None
        self._grasp_constraint: int | None = None
        if dynamic:
            p.setGravity(0, 0, -9.81, physicsClientId=self._client)
            # Procedural ground plane (no pybullet_data dependency — the source build ships none).
            plane_col = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=self._client)
            self._plane = p.createMultiBody(0, plane_col, physicsClientId=self._client)
            p.changeDynamics(self._plane, -1, lateralFriction=1.0, physicsClientId=self._client)
        else:
            p.setGravity(0, 0, 0, physicsClientId=self._client)
        self._arm_body: int = p.loadURDF(
            urdf_path,
            useFixedBase=True,
            physicsClientId=self._client,
        )
        # Build motor_name -> joint_index mapping (only for joints in MOTOR_ORDER).
        self._joint_indices: dict[str, int] = {}
        n = p.getNumJoints(self._arm_body, physicsClientId=self._client)
        for i in range(n):
            info = p.getJointInfo(self._arm_body, i, physicsClientId=self._client)
            name: str = info[1].decode()
            if name in MOTOR_ORDER:
                self._joint_indices[name] = i

        # scene body tracking
        self._scene_bodies: dict[str, int] = {}   # object name -> pybullet body id
        self._body_to_idx: dict[int, int] = {}    # pybullet body id -> scene object index

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> int:
        return self._client

    @property
    def arm_body(self) -> int:
        return self._arm_body

    @property
    def joint_indices(self) -> dict[str, int]:
        """Motor name -> PyBullet joint index, for joints found in the URDF."""
        return dict(self._joint_indices)

    @property
    def scene_bodies(self) -> dict[str, int]:
        """Object name -> PyBullet body id for loaded scene objects."""
        return dict(self._scene_bodies)

    @property
    def body_to_idx(self) -> dict[int, int]:
        """PyBullet body id -> scene object index (position in :class:`SimScene`)."""
        return dict(self._body_to_idx)

    # ------------------------------------------------------------------
    # Arm control
    # ------------------------------------------------------------------

    def set_joints(self, joints_deg: np.ndarray) -> None:
        """Teleport arm joints to *joints_deg* (length-6, degrees, in MOTOR_ORDER).

        Joints not found in the URDF are silently skipped so that the gripper
        joint (which may be absent in some URDF variants) does not cause errors.
        """
        arr = np.asarray(joints_deg, dtype=float)
        for i, name in enumerate(MOTOR_ORDER):
            idx = self._joint_indices.get(name)
            if idx is not None:
                p.resetJointState(
                    self._arm_body, idx, np.deg2rad(arr[i]),
                    physicsClientId=self._client,
                )

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def load_scene(self, scene: SimScene) -> dict[str, int]:
        """Instantiate every :class:`SceneObject` in *scene* as a static PyBullet body.

        Returns the name -> body_id mapping.  Calling this multiple times is
        allowed; previously loaded bodies are NOT removed.
        """
        for obj_idx, obj in enumerate(scene.objects):
            pos = list(np.asarray(obj.position, dtype=float))
            rgba = list(np.asarray(obj.rgba, dtype=float))

            if obj.shape == "box":
                he = list(np.asarray(obj.half_extents, dtype=float))
                col_id = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=he, physicsClientId=self._client
                )
                vis_id = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=he, rgbaColor=rgba,
                    physicsClientId=self._client,
                )
            elif obj.shape == "sphere":
                col_id = p.createCollisionShape(
                    p.GEOM_SPHERE, radius=obj.radius, physicsClientId=self._client
                )
                vis_id = p.createVisualShape(
                    p.GEOM_SPHERE, radius=obj.radius, rgbaColor=rgba,
                    physicsClientId=self._client,
                )
            else:
                raise ValueError(f"Unknown shape {obj.shape!r}; expected 'box' or 'sphere'")

            body_id = p.createMultiBody(
                baseMass=obj.mass,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=pos,
                physicsClientId=self._client,
            )
            if obj.mass > 0:
                # Friction so a grasped/resting object behaves sensibly.
                p.changeDynamics(body_id, -1, lateralFriction=1.0, physicsClientId=self._client)
            self._scene_bodies[obj.name] = body_id
            self._body_to_idx[body_id] = obj_idx

        return dict(self._scene_bodies)

    # ------------------------------------------------------------------
    # Physics + grasp (only meaningful when dynamic=True)
    # ------------------------------------------------------------------

    def step_physics(self) -> None:
        """Advance one physics step if dynamic; no-op otherwise."""
        if self.dynamic:
            p.stepSimulation(physicsClientId=self._client)

    def link_index(self, link_name: str) -> int:
        """PyBullet link index whose child link is *link_name*; -1 (base) if not found."""
        for i in range(p.getNumJoints(self._arm_body, physicsClientId=self._client)):
            info = p.getJointInfo(self._arm_body, i, physicsClientId=self._client)
            if info[12].decode() == link_name:
                return i
        return -1

    def _link_world_pose(self, link_index: int):
        if link_index < 0:
            return p.getBasePositionAndOrientation(self._arm_body, physicsClientId=self._client)
        ls = p.getLinkState(self._arm_body, link_index, physicsClientId=self._client)
        return ls[4], ls[5]  # worldLinkFramePosition, worldLinkFrameOrientation

    def attach(self, body_id: int, link_name: str = "gripper_frame_link") -> None:
        """Rigidly attach *body_id* to a gripper link at its current relative pose ("magnet"
        grasp). Stable with the kinematic (teleported) arm — no friction tuning needed. No-op
        if already holding something."""
        if self._grasp_constraint is not None:
            return
        li = self.link_index(link_name)
        lp, lo = self._link_world_pose(li)
        bp, bo = p.getBasePositionAndOrientation(body_id, physicsClientId=self._client)
        inv_p, inv_o = p.invertTransform(lp, lo)
        rel_p, rel_o = p.multiplyTransforms(inv_p, inv_o, bp, bo)  # box pose in link frame
        self._grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self._arm_body,
            parentLinkIndex=li,
            childBodyUniqueId=body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=list(rel_p),
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=list(rel_o),
            physicsClientId=self._client,
        )

    def release(self) -> None:
        """Release any grasped object (it then falls under gravity). No-op if not holding."""
        if self._grasp_constraint is not None:
            p.removeConstraint(self._grasp_constraint, physicsClientId=self._client)
            self._grasp_constraint = None

    @property
    def holding(self) -> bool:
        return self._grasp_constraint is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Disconnect from the PyBullet physics server."""
        try:
            p.disconnect(self._client)
        except Exception:
            pass

    def __enter__(self) -> "SimRenderer":
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

# Depth linearisation constants (metres).
_NEAR: float = 0.01
_FAR: float = 5.0


def _view_matrix_from_pose(cam_pose: np.ndarray) -> list[float]:
    """Build a PyBullet view matrix from a 4x4 camera pose (OpenCV convention).

    The camera optical frame has +z forward and +y down.
    ``target = eye + forward``, ``up = -T[:3,1]`` (negate because +y is down).
    """
    eye = cam_pose[:3, 3]
    forward = cam_pose[:3, 2]
    target = eye + forward
    up = -cam_pose[:3, 1]  # +y is down in optical frame; PyBullet wants "up" vector
    return list(p.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist(),
    ))


def _proj_matrix_from_intrinsics(intrinsics: CameraIntrinsics) -> list[float]:
    """Build a PyBullet projection matrix from :class:`CameraIntrinsics`."""
    fov_y_rad = 2.0 * math.atan((intrinsics.height / 2.0) / intrinsics.fy)
    fov_y_deg = math.degrees(fov_y_rad)
    aspect = intrinsics.width / intrinsics.height
    return list(p.computeProjectionMatrixFOV(
        fov=fov_y_deg,
        aspect=aspect,
        nearVal=_NEAR,
        farVal=_FAR,
    ))


def _depth_buffer_to_metres(depth_buf: np.ndarray) -> np.ndarray:
    """Convert PyBullet's normalised depth buffer to metric depth (metres).

    PyBullet stores ``(far/depth_linear - 1) / (far/near - 1)`` so the inverse is::

        depth_m = far * near / (far - (far - near) * depth_buf)
    """
    return _FAR * _NEAR / (_FAR - (_FAR - _NEAR) * depth_buf)


class OrbitCameraNav:
    """OnShape-style GUI camera: left-drag orbits, Ctrl+left-drag pans, wheel zooms.

    Authoritative — each :meth:`update` resets the debug camera from its own accumulated
    yaw/pitch/target, so it wins over PyBullet's built-in mouse handling. Distance is read live
    each tick, so the built-in scroll-wheel zoom is preserved. Mouse buttons come from
    ``getMouseEvents`` (independent of the keyboard event queue); the caller passes whether Ctrl
    is held (read from the shared keyboard events, since ``getKeyboardEvents`` consumes on read).
    """

    _MOVE, _BUTTON, _LEFT = 1, 2, 0  # PyBullet mouse eventType / buttonIndex

    def __init__(self, renderer: "SimRenderer", rotate_gain: float = 0.3, pan_gain: float = 0.002):
        self._client = renderer.client
        ci = p.getDebugVisualizerCamera(physicsClientId=self._client)
        self.yaw, self.pitch = float(ci[8]), float(ci[9])
        self.target = list(ci[11])
        self.rotate_gain, self.pan_gain = rotate_gain, pan_gain
        self._drag = False
        self._px = self._py = 0.0

    def update(self, ctrl: bool) -> None:
        ci = p.getDebugVisualizerCamera(physicsClientId=self._client)
        dist = float(ci[10])
        forward = np.asarray(ci[5], dtype=float)
        up = np.asarray(ci[4], dtype=float)
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) or 1.0)
        upn = up / (np.linalg.norm(up) or 1.0)
        for e in p.getMouseEvents(physicsClientId=self._client):
            if e[0] == self._BUTTON and e[3] == self._LEFT:
                if e[4] & p.KEY_WAS_TRIGGERED:
                    self._drag = True
                    self._px, self._py = e[1], e[2]
                elif e[4] & p.KEY_WAS_RELEASED:
                    self._drag = False
            elif e[0] == self._MOVE and self._drag:
                dx, dy = e[1] - self._px, e[2] - self._py
                self._px, self._py = e[1], e[2]
                if ctrl:  # pan: slide the look-at target in the camera plane
                    s = self.pan_gain * max(dist, 0.1)
                    self.target = (np.asarray(self.target) - right * dx * s + upn * dy * s).tolist()
                else:  # orbit
                    self.yaw += dx * self.rotate_gain
                    self.pitch = float(np.clip(self.pitch - dy * self.rotate_gain, -89.0, 89.0))
        p.resetDebugVisualizerCamera(dist, self.yaw, self.pitch, self.target, physicsClientId=self._client)


class SimCamera:
    """Synthetic camera that renders from the current arm pose via PyBullet.

    Implements :class:`dume.camera.CameraSource`.

    Parameters
    ----------
    renderer:
        A connected :class:`SimRenderer`.
    intrinsics:
        Pinhole camera parameters (width, height, focal lengths).
    pose_provider:
        Zero-argument callable that returns the 4x4 camera-in-world pose.
        Typically ``lambda: camera_pose_from_fk(kin, arm.read_joints())``.
    hardware:
        Use the GPU renderer (``ER_BULLET_HARDWARE_OPENGL``) instead of the CPU software
        rasterizer (``ER_TINY_RENDERER``). The software path is *very* slow — keep ``hardware``
        on in a GUI session; tests in DIRECT mode use software for reliability.
    """

    def __init__(
        self,
        renderer: SimRenderer,
        intrinsics: CameraIntrinsics,
        pose_provider: Callable[[], np.ndarray],
        hardware: bool = False,
    ) -> None:
        self._renderer = renderer
        self.intrinsics = intrinsics
        self._pose_provider = pose_provider
        self._render_flag = p.ER_BULLET_HARDWARE_OPENGL if hardware else p.ER_TINY_RENDERER
        # Cache last seg buffer for detect() to avoid double-render.
        self._last_seg: np.ndarray | None = None
        self._last_depth_m: np.ndarray | None = None

    # ------------------------------------------------------------------
    # CameraSource protocol
    # ------------------------------------------------------------------

    def capture(self) -> CameraFrame:
        """Render a frame from the current camera pose.

        Returns a :class:`~dume.camera.CameraFrame` with:

        - ``rgb``: ``(H, W, 3)`` uint8 array.
        - ``depth``: ``(H, W)`` float32 array in metres.
        - ``pose``: the 4x4 camera pose used.
        - ``t``: 0.0 (no clock in sim).
        """
        cam_pose = np.asarray(self._pose_provider(), dtype=float)
        view = _view_matrix_from_pose(cam_pose)
        proj = _proj_matrix_from_intrinsics(self.intrinsics)

        W, H = self.intrinsics.width, self.intrinsics.height
        _, _, rgb_raw, depth_raw, seg_raw = p.getCameraImage(
            width=W,
            height=H,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=self._render_flag,
            physicsClientId=self._renderer.client,
        )

        rgb = np.array(rgb_raw, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
        depth_buf = np.array(depth_raw, dtype=np.float32).reshape(H, W)
        depth_m = _depth_buffer_to_metres(depth_buf).astype(np.float32)
        seg = np.array(seg_raw, dtype=np.int32).reshape(H, W)

        # Cache for detect().
        self._last_seg = seg
        self._last_depth_m = depth_m

        return CameraFrame(pose=cam_pose, rgb=rgb, depth=depth_m, t=0.0)

    def detect(self) -> Detections:
        """Return detections for scene objects visible in the last captured frame.

        Calls :meth:`capture` internally if no frame has been captured yet.
        Segmentation body ids are mapped back to stable scene-object indices.
        """
        if self._last_seg is None or self._last_depth_m is None:
            self.capture()

        seg: np.ndarray = self._last_seg  # type: ignore[assignment]
        depth_m: np.ndarray = self._last_depth_m  # type: ignore[assignment]
        body_to_idx = self._renderer.body_to_idx

        ids: list[int] = []
        pixels_list: list[tuple[float, float]] = []
        depths_list: list[float] = []

        H, W = seg.shape
        for body_id, obj_idx in body_to_idx.items():
            mask = seg == body_id
            if not np.any(mask):
                continue
            rows, cols = np.where(mask)
            u = float(np.mean(cols))
            v = float(np.mean(rows))
            med_depth = float(np.median(depth_m[mask]))
            ids.append(obj_idx)
            pixels_list.append((u, v))
            depths_list.append(med_depth)

        if ids:
            pixels = np.array(pixels_list, dtype=float)
            depths = np.array(depths_list, dtype=float)
        else:
            pixels = np.zeros((0, 2), dtype=float)
            depths = np.zeros((0,), dtype=float)

        return Detections(ids=ids, pixels=pixels, depths=depths)
