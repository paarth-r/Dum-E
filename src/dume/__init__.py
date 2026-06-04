"""dume — smooth, intuitive inverse-kinematics control for the LeRobot SO-101 arm.

Public API lives in :mod:`dume.service` (the ``DumeArm`` facade). Lower-level pieces
(`geometry`, `planning`, `kinematics`, `arm`, `controller`, `input_xbox`) are importable
on their own so other services can compose them.
"""

from dume.config import ControllerConfig

__all__ = ["ControllerConfig"]
__version__ = "0.1.0"
