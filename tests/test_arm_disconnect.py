"""Disconnect must not raise over the top of a live exception.

On 2026-07-25 a real hardware fault was lost because cleanup failed while the original
exception was propagating: the traceback that reached the terminal was a ConnectionError
from ``disable_torque`` on the shutdown path, and the fault that actually caused it was
never seen. Cleanup that can throw makes every failure look like the same failure.
"""

import numpy as np
import pytest

from dume.arm import SO101Arm


class _ExplodingRobot:
    """Stands in for lerobot's SOFollower with a bus that has already gone dark."""

    def __init__(self):
        self.disconnect_attempted = False

    def disconnect(self):
        self.disconnect_attempted = True
        raise ConnectionError("Failed to write 'Torque_Enable' on id_=1: no status packet")


def _armed() -> tuple[SO101Arm, _ExplodingRobot]:
    arm = SO101Arm("/dev/null", "test")
    robot = _ExplodingRobot()
    arm._robot = robot
    return arm, robot


def test_disconnect_swallows_bus_failure():
    """A dead bus during shutdown must not become the exception the user sees."""
    arm, robot = _armed()
    arm.disconnect()  # must not raise
    assert robot.disconnect_attempted


def test_disconnect_clears_robot_even_when_it_fails():
    """State is reset regardless, so a retry doesn't reuse a half-dead handle."""
    arm, _ = _armed()
    arm.disconnect()
    assert arm._robot is None
    arm.disconnect()  # idempotent, still no raise


def test_original_exception_survives_failing_cleanup():
    """The real failure reaches the caller instead of being masked by cleanup.

    This is the regression: previously the ConnectionError from disconnect() replaced
    whatever was actually being handled.
    """
    arm, _ = _armed()
    with pytest.raises(RuntimeError, match="the real fault"):
        try:
            raise RuntimeError("the real fault")
        finally:
            arm.disconnect()


def test_disconnect_with_no_robot_is_a_noop():
    arm = SO101Arm("/dev/null", "test")
    arm.disconnect()
    assert arm._robot is None


def test_relax_still_propagates():
    """Only the shutdown path is guarded — a failed relax must still be visible.

    save-pose cuts torque so the arm can be hand-posed; silently failing there would drop
    the arm's weight onto the user with no warning.
    """

    class _Bus:
        def disable_torque(self):
            raise ConnectionError("bus down")

    class _Robot:
        bus = _Bus()

    arm = SO101Arm("/dev/null", "test")
    arm._robot = _Robot()
    with pytest.raises(ConnectionError):
        arm.relax()


def test_read_joints_unaffected():
    """Sanity: guarding disconnect doesn't touch the normal read path."""

    class _Robot:
        def get_observation(self):
            return {f"{m}.pos": float(i) for i, m in enumerate(
                ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
            )}

    arm = SO101Arm("/dev/null", "test")
    arm._robot = _Robot()
    assert np.allclose(arm.read_joints(), [0, 1, 2, 3, 4, 5])
