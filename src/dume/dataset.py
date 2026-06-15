"""Episode recording and dataset persistence for the dume arm controller.

Captures the observation -> action trajectory needed for imitation learning. The design
is deliberately hardware-agnostic: the caller assembles :class:`Observation` and
:class:`Action` values; this module handles structuring them into :class:`Episode` objects
and persisting them to disk via :class:`LocalBackend`.

On-disk layout (LocalBackend)
------------------------------
``<root>/<episode_id>/``
  ``arrays.npz``   — all numeric arrays stacked across timesteps:
                      ``joints``        (T, 6)
                      ``ee_pose``       (T, 4, 4)
                      ``action_joints`` (T, 6)
                      ``depth``         (T, H, W)  — present only when depth was recorded
                      ``image``         (T, H, W, C) — present only when image was recorded
  ``meta.json``    — episode metadata dict (task, id, created, step count, etc.)

Round-trip fidelity: joints, ee_pose, and action_joints always round-trip exactly
(float64 preserved). depth/image are stored when present; None-valued steps get a NaN
plane inserted so array dimensions stay consistent, but the LocalBackend raises
``ValueError`` if steps are mixed (some with depth, some without) — record consistently
or leave depth=None throughout.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from dume.camera import CameraSource, Detections


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """One snapshot of the world: arm state + optional camera data.

    Parameters
    ----------
    joints:
        Length-6 joint vector (degrees, except element 5 which is gripper 0..100).
    ee_pose:
        4x4 homogeneous transform of the end-effector in the arm base frame.
    detections:
        Object detections from the wrist camera, if available.
    depth:
        (H, W) depth map in metres from the wrist camera, if available.
    image:
        (H, W, 3) uint8 RGB image from the wrist camera, if available.
    t:
        Timestamp in seconds (monotonic clock of the recording host).
    """

    joints: np.ndarray          # (6,)
    ee_pose: np.ndarray         # (4, 4)
    detections: Detections | None = None
    depth: np.ndarray | None = None     # (H, W) float32/float64
    image: np.ndarray | None = None     # (H, W, 3) uint8
    t: float = 0.0


@dataclass
class Action:
    """Target joint configuration sent to the arm for this timestep.

    Parameters
    ----------
    joints_target:
        Length-6 target joint vector consumed directly by ``ArmIO.write_joints``.
        Elements 0-4 are degrees; element 5 is gripper 0..100.
    """

    joints_target: np.ndarray   # (6,)


@dataclass
class Step:
    """One (observation, action) pair in a trajectory."""

    observation: Observation
    action: Action


@dataclass
class Episode:
    """A complete demonstration trajectory.

    Parameters
    ----------
    steps:
        Ordered list of :class:`Step` objects.
    metadata:
        Arbitrary key-value store; conventionally includes ``"task"`` (str),
        ``"created"`` (ISO-8601 string), and ``"id"`` (UUID hex).
    """

    steps: list[Step] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# Observation builder (convenience helper — not I/O)
# ---------------------------------------------------------------------------


def observe(
    arm,
    kin,
    camera: CameraSource | None = None,
    t: float = 0.0,
) -> Observation:
    """Build an :class:`Observation` by querying ``arm`` and optionally ``camera``.

    Parameters
    ----------
    arm:
        Any object satisfying the ``ArmIO`` protocol (``read_joints()->np.ndarray``).
    kin:
        A ``Kinematics`` instance; ``fk(joints_deg)->4x4`` is called once.
    camera:
        Optional :class:`~dume.camera.CameraSource`. When provided, ``capture()`` and
        ``detect()`` are both called and their outputs stored in the Observation.
    t:
        Timestamp in seconds to tag this snapshot.

    Returns
    -------
    Observation
        Fully populated observation; depth/image/detections are None when ``camera`` is None.
    """
    joints = np.asarray(arm.read_joints(), dtype=float)
    ee_pose = np.asarray(kin.fk(joints), dtype=float)

    depth: np.ndarray | None = None
    image: np.ndarray | None = None
    detections: Detections | None = None

    if camera is not None:
        frame = camera.capture()
        depth = frame.depth
        image = frame.rgb
        detections = camera.detect()

    return Observation(
        joints=joints,
        ee_pose=ee_pose,
        detections=detections,
        depth=depth,
        image=image,
        t=t,
    )


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class EpisodeRecorder:
    """Accumulates (observation, action) pairs into an :class:`Episode`.

    Usage::

        rec = EpisodeRecorder()
        rec.start({"task": "pick_cube"})
        for _ in range(N):
            obs = observe(arm, kin)
            act = Action(joints_target=policy.select_action(obs).joints_target)
            rec.record(obs, act)
        episode = rec.finish()

    The recorder is pure-Python — no hardware I/O, no filesystem writes.
    """

    def __init__(self) -> None:
        self._episode: Episode | None = None

    def start(self, metadata: dict | None = None) -> None:
        """Begin a new episode, clearing any previous in-progress state."""
        now = datetime.now(tz=timezone.utc).isoformat()
        base_meta = {"id": uuid.uuid4().hex, "created": now}
        if metadata:
            base_meta.update(metadata)
        self._episode = Episode(steps=[], metadata=base_meta)

    def record(self, observation: Observation, action: Action) -> None:
        """Append one (observation, action) step to the in-progress episode."""
        if self._episode is None:
            raise RuntimeError("call start() before record()")
        self._episode.steps.append(Step(observation=observation, action=action))

    def finish(self) -> Episode:
        """Return the completed episode and reset internal state."""
        if self._episode is None:
            raise RuntimeError("call start() before finish()")
        episode = self._episode
        self._episode = None
        return episode

    # Convenience: expose observe() as a staticmethod so callers can do rec.observe(...)
    observe = staticmethod(observe)


# ---------------------------------------------------------------------------
# DatasetBackend Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DatasetBackend(Protocol):
    """Abstract persistence layer for :class:`Episode` objects."""

    def write(self, episode: Episode) -> str:
        """Persist ``episode`` and return its id string."""
        ...

    def read(self, episode_id: str) -> Episode:
        """Reconstruct an :class:`Episode` from storage by its id."""
        ...

    def list_ids(self) -> list[str]:
        """Return all stored episode ids."""
        ...


# ---------------------------------------------------------------------------
# LocalBackend — real filesystem persistence
# ---------------------------------------------------------------------------


class LocalBackend:
    """Stores episodes as ``.npz`` + ``.json`` pairs under ``root``.

    Parameters
    ----------
    root:
        Base directory for episode storage. Defaults to ``~/.dume/episodes``.
        Each episode occupies ``<root>/<episode_id>/``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = Path.home() / ".dume" / "episodes"
        self.root = Path(root)

    def _episode_dir(self, episode_id: str) -> Path:
        return self.root / episode_id

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    def write(self, episode: Episode) -> str:
        """Persist ``episode`` to ``<root>/<id>/``. Returns the episode id."""
        meta = dict(episode.metadata)
        ep_id: str = meta.get("id") or uuid.uuid4().hex
        meta["id"] = ep_id
        meta["n_steps"] = len(episode.steps)

        ep_dir = self._episode_dir(ep_id)
        ep_dir.mkdir(parents=True, exist_ok=True)

        # ---- stack arrays -----------------------------------------------
        steps = episode.steps
        if not steps:
            # Empty episode: write empty arrays
            np.savez(
                ep_dir / "arrays.npz",
                joints=np.zeros((0, 6), dtype=float),
                ee_pose=np.zeros((0, 4, 4), dtype=float),
                action_joints=np.zeros((0, 6), dtype=float),
            )
        else:
            joints = np.stack([s.observation.joints for s in steps]).astype(float)
            ee_pose = np.stack([s.observation.ee_pose for s in steps]).astype(float)
            action_joints = np.stack([s.action.joints_target for s in steps]).astype(float)

            arrays: dict[str, np.ndarray] = {
                "joints": joints,
                "ee_pose": ee_pose,
                "action_joints": action_joints,
            }

            # Optional depth
            has_depth = [s.observation.depth is not None for s in steps]
            if any(has_depth):
                if not all(has_depth):
                    raise ValueError(
                        "Mixed depth: some steps have depth, some don't. "
                        "Record consistently or leave depth=None throughout."
                    )
                arrays["depth"] = np.stack(
                    [s.observation.depth for s in steps]
                ).astype(float)

            # Optional image
            has_image = [s.observation.image is not None for s in steps]
            if any(has_image):
                if not all(has_image):
                    raise ValueError(
                        "Mixed image: some steps have image, some don't. "
                        "Record consistently or leave image=None throughout."
                    )
                arrays["image"] = np.stack([s.observation.image for s in steps])

            np.savez(ep_dir / "arrays.npz", **arrays)

        # ---- metadata sidecar -------------------------------------------
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return ep_id

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def read(self, episode_id: str) -> Episode:
        """Reconstruct an :class:`Episode` from disk."""
        ep_dir = self._episode_dir(episode_id)
        if not ep_dir.exists():
            raise FileNotFoundError(f"Episode not found: {episode_id!r} (looked in {ep_dir})")

        meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
        data = np.load(ep_dir / "arrays.npz", allow_pickle=False)

        joints_all = data["joints"]          # (T, 6)
        ee_pose_all = data["ee_pose"]        # (T, 4, 4)
        action_joints_all = data["action_joints"]  # (T, 6)

        depth_all = data["depth"] if "depth" in data else None
        image_all = data["image"] if "image" in data else None

        n_steps = len(joints_all)
        steps: list[Step] = []
        for i in range(n_steps):
            obs = Observation(
                joints=joints_all[i],
                ee_pose=ee_pose_all[i],
                depth=depth_all[i] if depth_all is not None else None,
                image=image_all[i] if image_all is not None else None,
                detections=None,   # not serialised; reconstructed from sensor on replay
                t=0.0,
            )
            act = Action(joints_target=action_joints_all[i])
            steps.append(Step(observation=obs, action=act))

        return Episode(steps=steps, metadata=meta)

    # ------------------------------------------------------------------
    # list_ids
    # ------------------------------------------------------------------

    def list_ids(self) -> list[str]:
        """Return ids of all stored episodes (subdirectory names with a meta.json)."""
        if not self.root.exists():
            return []
        return sorted(
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        )


# ---------------------------------------------------------------------------
# LeRobot export stub
# ---------------------------------------------------------------------------


def to_lerobot(episode_or_episodes, repo_id: str, root=None):
    """Export dume episodes to a LeRobotDataset for training lerobot's DiffusionPolicy.

    This function will convert one or more :class:`Episode` objects (or a LocalBackend)
    into the LeRobot dataset format expected by ``lerobot.scripts.push_dataset_to_hub``
    / ``LeRobotDataset``. It will handle stacking observations, writing the Parquet
    episode tables, computing stats, and optionally pushing to the Hugging Face Hub
    under ``repo_id``.

    Parameters
    ----------
    episode_or_episodes:
        A single :class:`Episode`, a list thereof, or a :class:`LocalBackend` instance
        whose stored episodes should all be exported.
    repo_id:
        Hugging Face repo id, e.g. ``"paarth-r/so101-pick-cube"``.
    root:
        Optional local path to materialise the dataset before pushing.

    Raises
    ------
    NotImplementedError
        Always — this export path is not yet implemented. It will be wired up once the
        first real demonstrations are recorded and ``lerobot`` integration is validated
        end-to-end.
    """
    raise NotImplementedError(
        "to_lerobot: lerobot dataset export lands when first real episodes are recorded"
    )
