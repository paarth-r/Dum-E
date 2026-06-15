"""Dum-E (``dume``) — smooth, intuitive inverse-kinematics control for the LeRobot SO-101 arm.

Named after DUM-E, Tony Stark's robotic arm in Iron Man.

Public API lives in :mod:`dume.service` (the ``DumeArm`` facade). Lower-level pieces
(`geometry`, `planning`, `kinematics`, `arm`, `controller`, `input_xbox`) are importable
on their own so other services can compose them.
"""

from dume.camera import (
    CameraFrame,
    CameraIntrinsics,
    CameraSource,
    Detections,
    camera_pose_from_fk,
)
from dume.config import ControllerConfig
from dume.dataset import (
    Action,
    DatasetBackend,
    Episode,
    EpisodeRecorder,
    LocalBackend,
    Observation,
    Step,
)
from dume.flown_stereo import Grasp, propose_grasp, relative_pose, triangulate
from dume.policy import LeRobotDiffusionPolicy, Policy, ScriptedPolicy

# Note: dume.sim_world (PyBullet) and dume.arducam are imported on demand, not here — sim_world
# pulls in a heavy renderer and arducam is a hardware stub.

__all__ = [
    "ControllerConfig",
    "CameraFrame",
    "CameraIntrinsics",
    "CameraSource",
    "Detections",
    "camera_pose_from_fk",
    "Observation",
    "Action",
    "Step",
    "Episode",
    "EpisodeRecorder",
    "DatasetBackend",
    "LocalBackend",
    "Grasp",
    "propose_grasp",
    "relative_pose",
    "triangulate",
    "Policy",
    "ScriptedPolicy",
    "LeRobotDiffusionPolicy",
]
__version__ = "0.1.0"
