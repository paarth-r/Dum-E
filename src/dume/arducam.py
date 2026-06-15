"""Real Arducam capture — the hardware-bound :class:`CameraSource` (stub for now).

The physical camera is the end-effector-mounted **Arducam UC-844: 1280x800 monochrome,
OV9781 global-shutter sensor** (global shutter matters — no rolling-shutter skew while the arm
moves between the two flown-stereo snapshots; mono means captured frames are single-channel).

This is the only camera backend that needs the physical camera. It is intentionally a stub
until the mount STL exists and calibration is done; the geometric :class:`dume.sim_world.SimCamera`
stands in for all testing today. When the hardware lands, implement ``capture`` (grab a frame
off the UVC device at native 1280x800) and ``detect``, and replace the placeholder intrinsics
below with values from real chessboard/ChArUco calibration, plus calibrate ``dume.camera.T_CAM_MOUNT``.
"""

from __future__ import annotations

from dume.camera import CameraFrame, CameraIntrinsics, Detections

# Arducam UC-844 native sensor resolution (OV9781). Known; focal/centre are NOT — they need
# real intrinsic calibration. The FOV here is a rough placeholder only so the type is usable
# before calibration; do not trust fx/fy/cx/cy until measured.
ARDUCAM_WIDTH = 1280
ARDUCAM_HEIGHT = 800
_PLACEHOLDER_FOV_Y_DEG = 70.0  # TODO: replace with calibrated intrinsics

_NEEDS_HARDWARE = (
    "ArduCamSource needs the physical end-effector Arducam UC-844 (mount + intrinsic/hand-eye "
    "calibration pending). Use dume.sim_world.SimCamera for sim/testing."
)


class ArduCamSource:
    """End-effector Arducam UC-844 as a :class:`dume.camera.CameraSource`. Not yet implemented."""

    def __init__(self, device: int = 0, intrinsics: CameraIntrinsics | None = None):
        self.device = device
        # Native 1280x800; focal/centre are uncalibrated placeholders until measured.
        self.intrinsics = intrinsics or CameraIntrinsics.from_fov(
            ARDUCAM_WIDTH, ARDUCAM_HEIGHT, _PLACEHOLDER_FOV_Y_DEG
        )

    def capture(self) -> CameraFrame:
        raise NotImplementedError(_NEEDS_HARDWARE)

    def detect(self) -> Detections:
        raise NotImplementedError(_NEEDS_HARDWARE)
