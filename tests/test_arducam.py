"""Arducam device selection and non-blocking capture.

The device is chosen by *resolution*, not index: this machine has three cameras (built-in
1920x1080, the Arducam 1280x800, an iPhone Continuity camera) and macOS enumeration order is
not stable. Picking the wrong one is the dangerous failure — nothing raises, frames arrive,
and every reconstruction built from them is meaningless.

Everything here injects a fake opener, so no camera and no cv2 GUI is involved.
"""

import numpy as np
import pytest

from dume.arducam import (
    ARDUCAM_HEIGHT,
    ARDUCAM_WIDTH,
    ArduCamSource,
    find_camera_index,
)


class _FakeCapture:
    """Minimal stand-in for cv2.VideoCapture."""

    def __init__(self, width, height, *, opened=True, frames=None):
        self._w, self._h = width, height
        self._opened = opened
        self._frames = frames
        self.released = False

    def isOpened(self):
        return self._opened

    def get(self, prop):
        return {3: float(self._w), 4: float(self._h)}.get(prop, 0.0)

    def set(self, prop, value):
        return True

    def read(self):
        if self._frames is None:
            return True, np.zeros((self._h, self._w, 3), dtype=np.uint8)
        return True, self._frames()

    def release(self):
        self.released = True


def _opener(spec):
    """Build an opener from {index: (w, h)}; missing indices come back unopened."""

    def open_index(i):
        if i not in spec:
            return _FakeCapture(0, 0, opened=False)
        return _FakeCapture(*spec[i])

    return open_index


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def test_picks_the_1280x800_device():
    idx = find_camera_index(opener=_opener({0: (1920, 1080), 1: (1280, 800)}), max_index=4)
    assert idx == 1


def test_skips_the_builtin_webcam():
    """The laptop camera must never win — it is index 0 and would silently be chosen."""
    idx = find_camera_index(opener=_opener({0: (1920, 1080), 2: (1280, 800)}), max_index=4)
    assert idx == 2


def test_raises_naming_what_was_found_when_absent():
    """A clear error beats streaming the wrong camera."""
    with pytest.raises(RuntimeError) as err:
        find_camera_index(opener=_opener({0: (1920, 1080)}), max_index=3)
    msg = str(err.value)
    assert "1280x800" in msg
    assert "1920x1080" in msg  # reports what it did see, so the user can tell what happened


def test_releases_every_probed_device():
    """Probing must not leave capture handles open, or the real open will fail."""
    opened = []

    def opener(i):
        cap = _FakeCapture(*({0: (1920, 1080), 1: (1280, 800)}.get(i, (0, 0))), opened=i in (0, 1))
        opened.append(cap)
        return cap

    find_camera_index(opener=opener, max_index=3)
    assert all(c.released for c in opened)


def test_explicit_device_index_skips_probing():
    """Passing device= bypasses the probe entirely (escape hatch for odd setups)."""
    src = ArduCamSource(device=3)
    assert src.device == 3


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def test_capture_returns_single_channel_frame():
    """The OV9281 is mono; cv2 hands back 3-channel BGR regardless. Convert once."""
    src = ArduCamSource(device=0, opener=_opener({0: (ARDUCAM_WIDTH, ARDUCAM_HEIGHT)}))
    src.open()
    try:
        frame = src.capture()
    finally:
        src.close()
    assert frame.rgb.ndim == 2
    assert frame.rgb.shape == (ARDUCAM_HEIGHT, ARDUCAM_WIDTH)


def test_capture_uses_the_pose_provider():
    """Pose comes from the caller — which must supply MEASURED joints, not commanded.

    ArduCamSource deliberately cannot compute its own pose: it has no arm reference, so the
    measured-vs-commanded decision is forced to the wiring site where it is visible.
    """
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    src = ArduCamSource(
        device=0,
        opener=_opener({0: (ARDUCAM_WIDTH, ARDUCAM_HEIGHT)}),
        pose_provider=lambda: pose,
    )
    src.open()
    try:
        assert np.allclose(src.capture().pose, pose)
    finally:
        src.close()


def test_capture_without_pose_provider_is_identity():
    src = ArduCamSource(device=0, opener=_opener({0: (ARDUCAM_WIDTH, ARDUCAM_HEIGHT)}))
    src.open()
    try:
        assert np.allclose(src.capture().pose, np.eye(4))
    finally:
        src.close()


def test_capture_before_open_raises():
    src = ArduCamSource(device=0, opener=_opener({0: (ARDUCAM_WIDTH, ARDUCAM_HEIGHT)}))
    with pytest.raises(RuntimeError, match="open"):
        src.capture()


def test_capture_does_not_block_on_the_device():
    """The grabber thread owns the device; capture() returns the freshest frame.

    A blocking USB read inside the control loop would stall the arm — the same class of
    problem as reading noisy servo feedback in the control path.
    """
    reads = {"n": 0}

    def frames():
        reads["n"] += 1
        return np.full((ARDUCAM_HEIGHT, ARDUCAM_WIDTH, 3), reads["n"] % 256, dtype=np.uint8)

    cap = _FakeCapture(ARDUCAM_WIDTH, ARDUCAM_HEIGHT, frames=frames)
    src = ArduCamSource(device=0, opener=lambda i: cap)
    src.open()
    try:
        for _ in range(5):
            src.capture()
        # Many device reads happened in the thread, independent of how often we called capture.
        assert reads["n"] >= 5
    finally:
        src.close()


def test_close_releases_the_device():
    cap = _FakeCapture(ARDUCAM_WIDTH, ARDUCAM_HEIGHT)
    src = ArduCamSource(device=0, opener=lambda i: cap)
    src.open()
    src.close()
    assert cap.released


def test_context_manager_closes():
    cap = _FakeCapture(ARDUCAM_WIDTH, ARDUCAM_HEIGHT)
    with ArduCamSource(device=0, opener=lambda i: cap) as src:
        src.capture()
    assert cap.released


def test_detect_returns_empty_not_an_error():
    """No detector on real frames yet; empty is honest and keeps the protocol usable."""
    src = ArduCamSource(device=0, opener=_opener({0: (ARDUCAM_WIDTH, ARDUCAM_HEIGHT)}))
    assert len(src.detect()) == 0


def test_intrinsics_are_flagged_uncalibrated():
    """Placeholder intrinsics must stay obviously placeholder until a real calibration."""
    src = ArduCamSource(device=0)
    assert src.intrinsics.width == ARDUCAM_WIDTH
    assert src.intrinsics.height == ARDUCAM_HEIGHT
    assert src.calibrated is False


def test_is_a_camera_source():
    from dume.camera import CameraSource

    assert isinstance(ArduCamSource(device=0), CameraSource)
