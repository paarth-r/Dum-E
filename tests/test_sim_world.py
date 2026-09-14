"""Tests for dume.sim_world — PyBullet-backed kinematic simulator.

All tests use DIRECT (headless) mode so they run in CI without a display.
"""

from __future__ import annotations

from dume.poses import HOME_JOINTS
from dume.sim_world import SimRenderer


# ---------------------------------------------------------------------------
# Test 1: renderer loads URDF and accepts joint commands
# ---------------------------------------------------------------------------

def test_renderer_loads_and_sets_joints():
    """SimRenderer loads the SO-101 URDF and maps at least 5 of 6 motor joints."""
    renderer = SimRenderer(gui=False)
    try:
        # set_joints should not raise, even with all 6 DOF
        renderer.set_joints(HOME_JOINTS)
        found = renderer.joint_indices
        # The URDF must expose at least 5 of the 6 MOTOR_ORDER joints.
        assert len(found) >= 5, (
            f"Expected >=5 joints mapped, got {len(found)}: {list(found.keys())}"
        )
    finally:
        renderer.disconnect()
