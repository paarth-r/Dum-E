"""Real Arducam capture — the hardware-bound :class:`CameraSource` (stub for now).

This is the only camera backend that needs the physical end-effector-mounted Arducam. It is
intentionally a stub until the camera mount STL exists and hand-eye calibration is done; the
geometric :class:`dume.sim_world.SimCamera` stands in for all testing today. When the hardware
lands, implement ``capture`` (grab a frame off the UVC device) and ``detect`` (run whatever
detector we settle on), and calibrate ``intrinsics`` + ``dume.camera.T_CAM_MOUNT``.
"""

from __future__ import annotations

from dume.camera import CameraFrame, CameraIntrinsics, Detections

_NEEDS_HARDWARE = (
    "ArduCamSource needs the physical end-effector Arducam (mount + hand-eye calibration "
    "pending). Use dume.sim_world.SimCamera for sim/testing."
)


class ArduCamSource:
    """End-effector Arducam as a :class:`dume.camera.CameraSource`. Not yet implemented."""

    def __init__(self, device: int = 0, intrinsics: CameraIntrinsics | None = None):
        self.device = device
        # Placeholder intrinsics until calibrated; lets callers wire the type without a device.
        self.intrinsics = intrinsics or CameraIntrinsics.from_fov(640, 480, 60.0)

    def capture(self) -> CameraFrame:
        raise NotImplementedError(_NEEDS_HARDWARE)

    def detect(self) -> Detections:
        raise NotImplementedError(_NEEDS_HARDWARE)
