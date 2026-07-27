"""Real Arducam capture — the hardware-bound :class:`CameraSource`.

The physical camera is the end-effector-mounted **Arducam UC-844: 1280x800 monochrome,
OV9281 global-shutter sensor** (global shutter matters — no rolling-shutter skew while the arm
moves between flown-stereo snapshots; mono means frames are single-channel).

Two decisions here are load-bearing:

**The device is chosen by resolution, not by index.** macOS enumeration order is not stable and
this machine carries three cameras (built-in 1920x1080, the Arducam 1280x800, an iPhone
Continuity camera). A wrong index is the dangerous failure mode: nothing raises, frames keep
arriving, and every reconstruction built on them is quietly meaningless. Better to fail loudly.

**Capture is threaded.** A blocking USB read inside the control loop would stall the arm — the
same class of mistake as feeding noisy servo feedback into the control path. A background thread
owns the device and keeps a single latest-frame slot; ``capture()`` never waits on I/O. Dropped
frames are expected and fine; this is not a recorder.

Intrinsics remain an uncalibrated placeholder (see ``calibrated``). Triangulated depth scales
directly with focal length, so anything measured from these frames is structurally right and
dimensionally wrong until a real ChArUco calibration lands.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from dume.camera import CameraFrame, CameraIntrinsics, Detections

# Arducam UC-844 native sensor resolution (OV9281). Known; focal/centre are NOT — they need
# real intrinsic calibration. The FOV here is a rough placeholder only so the type is usable
# before calibration; do not trust fx/fy/cx/cy until measured.
ARDUCAM_WIDTH = 1280
ARDUCAM_HEIGHT = 800
_PLACEHOLDER_FOV_Y_DEG = 70.0  # TODO: replace with calibrated intrinsics

# cv2.CAP_PROP_FRAME_WIDTH / _HEIGHT, inlined so device probing needs no cv2 import.
_PROP_WIDTH = 3
_PROP_HEIGHT = 4


def _cv2_opener(index: int):
    """Default opener: a real cv2.VideoCapture. Imported lazily to keep import cost off tests."""
    import cv2

    return cv2.VideoCapture(index)


def find_camera_index(
    width: int = ARDUCAM_WIDTH,
    height: int = ARDUCAM_HEIGHT,
    *,
    opener=None,
    max_index: int = 8,
) -> int:
    """Return the index of the first camera reporting exactly ``width`` x ``height``.

    Reads each device's *native* mode rather than requesting a resolution first: most webcams
    accept a ``set()`` and snap to their nearest supported mode, which would let the built-in
    camera masquerade as the Arducam. The native size is the honest discriminator.

    Raises naming every device found, so a failure says what the machine actually has.
    """
    opener = opener or _cv2_opener
    seen: list[str] = []
    for index in range(max_index):
        cap = opener(index)
        if cap is None:
            continue
        try:
            if not cap.isOpened():
                continue
            found = (int(cap.get(_PROP_WIDTH)), int(cap.get(_PROP_HEIGHT)))
            seen.append(f"index {index}: {found[0]}x{found[1]}")
            if found == (width, height):
                return index
        finally:
            # Always release: a probe that leaks a handle makes the real open fail.
            cap.release()
    raise RuntimeError(
        f"No camera reporting {width}x{height} (the Arducam UC-844). Found: "
        + ("; ".join(seen) if seen else "no cameras at all")
        + ". Check the USB connection, or pass an explicit device index."
    )


class _Grabber:
    """Background reader holding only the most recent frame.

    Deliberately keeps a slot, not a queue: consumers want the freshest view, and a queue would
    grow without bound whenever the control loop runs slower than the camera.
    """

    def __init__(self, cap):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stamp: float = 0.0
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="arducam-grabber")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ok, frame = self._cap.read()
            except BaseException as exc:  # noqa: BLE001 — surfaced to the caller via latest()
                with self._lock:
                    self._error = exc
                return
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._stamp = time.time()

    def latest(self) -> tuple[np.ndarray, float]:
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"Camera read failed: {self._error}") from self._error
            frame, stamp = self._frame, self._stamp
        if frame is None:
            # First call can outrun the thread's first read; wait briefly rather than
            # returning a fake frame that would silently enter the reconstruction.
            deadline = time.time() + 2.0
            while time.time() < deadline:
                time.sleep(0.002)
                with self._lock:
                    frame, stamp = self._frame, self._stamp
                if frame is not None:
                    break
            else:
                raise RuntimeError("Camera produced no frames within 2 s of opening.")
        return frame, stamp

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def _to_grey(frame: np.ndarray) -> np.ndarray:
    """Collapse to single channel. The sensor is mono; cv2 hands back BGR regardless."""
    if frame.ndim == 2:
        return frame
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class ArduCamSource:
    """End-effector Arducam UC-844 as a :class:`dume.camera.CameraSource`.

    ``pose_provider`` supplies the camera-in-base pose for each frame. This class deliberately
    cannot compute its own pose — it holds no arm reference — which forces the
    measured-versus-commanded joint decision out to the wiring site where it is visible.
    Perception must use ``kin.fk(arm.read_joints()) @ mount``: the commanded reference ``q_ref``
    is where the arm was *told* to go, and gravity sag makes that differ by degrees in a way
    that is correlated across views, so it bends a reconstruction rather than averaging out.
    """

    def __init__(
        self,
        device: int | None = None,
        intrinsics: CameraIntrinsics | None = None,
        *,
        opener=None,
        pose_provider=None,
    ):
        self.device = device
        self._opener = opener or _cv2_opener
        self._pose_provider = pose_provider
        # Native 1280x800; focal/centre are uncalibrated placeholders until measured.
        self.intrinsics = intrinsics or CameraIntrinsics.from_fov(
            ARDUCAM_WIDTH, ARDUCAM_HEIGHT, _PLACEHOLDER_FOV_Y_DEG
        )
        #: False until real intrinsics are supplied. Anything metric read off these frames is
        #: scaled by however wrong the placeholder focal length is.
        self.calibrated = intrinsics is not None
        self._cap = None
        self._grabber: _Grabber | None = None

    # ---- lifecycle ----
    def open(self) -> "ArduCamSource":
        if self._grabber is not None:
            return self
        if self.device is None:
            self.device = find_camera_index(opener=self._opener)
        cap = self._opener(self.device)
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self.device}.")
        self._cap = cap
        self._grabber = _Grabber(cap)
        return self

    def close(self) -> None:
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "ArduCamSource":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- CameraSource ----
    def capture(self) -> CameraFrame:
        if self._grabber is None:
            raise RuntimeError("Camera is not open — call open() (or use it as a context manager).")
        frame, stamp = self._grabber.latest()
        pose = np.eye(4) if self._pose_provider is None else np.asarray(
            self._pose_provider(), dtype=float
        )
        return CameraFrame(pose=pose, rgb=_to_grey(frame), depth=None, t=stamp)

    def detect(self) -> Detections:
        """No detector runs on real frames yet — empty rather than raising.

        ``triangulate_detections`` matches by object id, which only exists because ``SimCamera``
        invents it. Real correspondence is feature matching (see the point-cloud work), not this.
        """
        return Detections()
