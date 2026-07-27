"""Scripted scan: walk the arm through saved setpoints, capturing a posed frame at each stop.

This is the first code that pairs real imagery with real arm geometry, so the pose convention
matters more here than anywhere else:

**Camera poses come from MEASURED joints, never the commanded reference.** The controller
tracks an internal commanded reference ``q_ref`` and ignores servo feedback on purpose (see
``controller.py``) — correct for control, because feeding noisy measurements back in produces
jitter. It is wrong for perception. ``q_ref`` is where the arm was *told* to go; gravity sag
and a proportional loop's steady-state error put the real arm degrees away from it. That error
is *correlated across views*, so it does not average out — it bends a reconstruction rather
than blurring it. Note ``Telemetry.joints_sent`` is also the commanded vector, so the tick hook
is not a valid pose source either.

Measured joints are noisy, which is exactly why control avoids them. The fix is the dwell:
while the arm is stationary at a stop, average several reads. Averaging is only valid because
nothing is moving, which is why a frame captured mid-motion is discarded rather than posed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from dume.camera import T_CAM_MOUNT


@dataclass
class ScanStop:
    """One captured stop: the setpoint name, the measured joints, the camera pose, the frame."""

    name: str
    joints: np.ndarray  # measured, dwell-averaged (deg; gripper 0..100)
    pose: np.ndarray  # 4x4 camera-in-base
    frame: object  # dume.camera.CameraFrame


def averaged_joints(arm, samples: int = 5, *, sleep=time.sleep, delay: float = 0.01) -> np.ndarray:
    """Mean of several measured joint reads.

    Valid **only while the arm is stationary**. Averaging a moving arm smears position across
    the motion and produces a pose that matches no frame at all.
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    reads = []
    for i in range(samples):
        reads.append(np.asarray(arm.read_joints(), dtype=float))
        if i + 1 < samples:
            sleep(delay)
    return np.mean(reads, axis=0)


def measured_camera_pose(kin, arm, *, samples: int = 5, mount=None, sleep=time.sleep) -> np.ndarray:
    """Camera-in-base pose from dwell-averaged **measured** joints.

    This is the only sanctioned way to pose a real frame. See the module docstring for why
    ``q_ref`` is not an acceptable substitute.
    """
    mount = T_CAM_MOUNT if mount is None else mount
    joints = averaged_joints(arm, samples, sleep=sleep)
    return kin.fk(joints) @ mount, joints


class InMemoryStore:
    """Store-shaped adapter for generated setpoints, so ``run_scan`` needs no special case."""

    def __init__(self, mapping):
        self._m = {k: np.asarray(v, dtype=float) for k, v in mapping.items()}

    def has(self, name) -> bool:
        return name in self._m

    def get(self, name) -> np.ndarray:
        return self._m[name].copy()

    def names(self) -> list[str]:
        return list(self._m)


def sweep_setpoints(base_joints, n: int = 7, pan_deg: float = 24.0):
    """Generate ``n`` configurations fanning shoulder_pan around ``base_joints``.

    Triangulation needs parallax, and parallax needs the camera to actually move between views.
    Panning the base sweeps the camera along an arc — a large lateral baseline for a small
    joint excursion, and it keeps the scene broadly in frame because the camera rotates far
    less than it translates relative to nearby objects.

    Returns ``(names, InMemoryStore)`` ordered so the sweep runs one way across the arc, which
    also means every stop is approached from the same direction — the standard trick for
    keeping gear backlash consistent instead of flipping sign mid-scan.
    """
    base = np.asarray(base_joints, dtype=float)
    if n < 2:
        raise ValueError("a sweep needs at least 2 viewpoints to triangulate anything")
    offsets = np.linspace(-pan_deg / 2.0, pan_deg / 2.0, n)
    mapping = {}
    names = []
    for i, off in enumerate(offsets):
        q = base.copy()
        q[0] = base[0] + off  # shoulder_pan
        name = f"sweep{i:02d}"
        mapping[name] = q
        names.append(name)
    return names, InMemoryStore(mapping)


def run_scan(
    dume_arm,
    camera,
    names,
    store,
    *,
    dwell: float = 0.6,
    samples: int = 5,
    on_tick=None,
    sleep=time.sleep,
):
    """Visit each named setpoint, yielding a :class:`ScanStop` per stop.

    ``on_tick`` is forwarded to :meth:`DumeArm.goto_joints` and called every control tick while
    moving; returning ``False`` aborts the whole scan (not just the current move), so a single
    keypress stops the arm rather than merely skipping ahead to the next setpoint.

    A generator on purpose: the caller decides what to do with each stop (render it, feed it to
    a cloud builder, write it out) without this module knowing about any of them.
    """
    for name in names:
        if not store.has(name):
            raise KeyError(f"No saved setpoint named {name!r}. Known: {store.names()}")

    aborted = False

    def tick():
        nonlocal aborted
        if on_tick is None:
            return None
        if on_tick() is False:
            aborted = True
            return False
        return None

    for name in names:
        dume_arm.goto_joints(store.get(name), on_tick=tick)
        if aborted:
            return
        # Settle, then measure. The dwell is what makes averaging measured joints valid.
        sleep(dwell)
        pose, joints = measured_camera_pose(dume_arm.kin, dume_arm.arm, samples=samples, sleep=sleep)
        frame = camera.capture()
        frame.pose = pose  # authoritative: measured, not whatever the source guessed
        yield ScanStop(name=name, joints=joints, pose=pose, frame=frame)
