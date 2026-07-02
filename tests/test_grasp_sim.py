"""Physical grasp sim: GraspParams physics setup + the PyBulletArm faithful ArmIO.

Headless (DIRECT) + dynamic. These assert the physics substrate (friction/solver knobs land on
the real bodies) and that the control stack's hardware surrogate reports *physical* feedback —
including a jaw stalled on a box. Reliable pick-and-place is intentionally NOT asserted (that's the
later high-fidelity phase); see the design spec.
"""

import numpy as np
import pybullet as p
import pytest

from dume.arm import ArmIO
from dume.poses import HOME_JOINTS
from dume.sim_world import GraspParams, PyBulletArm, SceneObject, SimRenderer, SimScene


def _clamp_static_box(r, arm):
    """Move to HOME with the gripper open, drop a fixed (mass-0) box at the jaw, close onto it.

    A static box is a clean, deterministic stand-in for "the jaw is physically blocked": the
    gripper motor cannot close through it, so it exercises the stall/contact path without the
    positioning flakiness of a falling dynamic box. Returns the box body id."""
    q = HOME_JOINTS.copy()
    q[5] = 95.0  # open
    for _ in range(40):
        arm.write_joints(q)
    gfl = r.link_index("gripper_frame_link")
    pos = list(p.getLinkState(r.arm_body, gfl, physicsClientId=r.client)[4])
    s = SimScene()
    s.add(SceneObject("obstacle", "box", half_extents=[0.03, 0.03, 0.03], position=pos, mass=0.0))
    r.load_scene(s)
    q[5] = 5.0  # command fully closed
    for _ in range(60):
        arm.write_joints(q)
    return r.scene_bodies["obstacle"]


def test_grasp_params_friction_lands_on_jaw_links():
    grasp = GraspParams(jaw_friction=2.0)
    with SimRenderer(gui=False, dynamic=True, grasp=grasp) as r:
        jaw = r.link_index("moving_jaw_so101_v1_link")
        fric = p.getDynamicsInfo(r.arm_body, jaw, physicsClientId=r.client)[1]
        assert fric == 2.0


def test_pybullet_arm_satisfies_armio():
    with SimRenderer(gui=False, dynamic=True) as r:
        arm = PyBulletArm(r)
        assert isinstance(arm, ArmIO)
        assert arm.is_calibrated() is True


def test_write_read_tracks_command_without_obstacle():
    """Motor-driven arm converges read_joints to the commanded joints in free space."""
    with SimRenderer(gui=False, dynamic=True) as r:
        arm = PyBulletArm(r)
        arm.connect()
        target = np.array([10.0, -15.0, 25.0, 5.0, -5.0, 50.0])
        for _ in range(60):
            arm.write_joints(target)
        assert np.max(np.abs(arm.read_joints()[:5] - target[:5])) < 3.0


def test_gripper_units_roundtrip():
    """Gripper is reported in the same 0..100 units SO101Arm uses; tracks in free space."""
    with SimRenderer(gui=False, dynamic=True) as r:
        arm = PyBulletArm(r)
        arm.connect()
        q = HOME_JOINTS.copy()
        q[5] = 50.0
        for _ in range(60):
            arm.write_joints(q)
        assert arm.read_joints()[5] == pytest.approx(50.0, abs=5.0)


def test_relax_lets_arm_sag_under_gravity():
    """read_joints reflects physics, not the last command: cutting torque lets gravity move it."""
    with SimRenderer(gui=False, dynamic=True) as r:
        arm = PyBulletArm(r)
        arm.connect()
        held = HOME_JOINTS.copy()
        for _ in range(30):
            arm.write_joints(held)  # motors hold the pose
        q_held = arm.read_joints()
        arm.relax()  # torque off
        for _ in range(240):
            r.step_physics()
        q_sagged = arm.read_joints()
        assert np.max(np.abs(q_sagged[:5] - q_held[:5])) > 5.0


def test_gripper_read_stalls_on_obstacle():
    """The faithfulness property: a jaw that can't close through an object reads short of the
    commanded closed value — exactly what a real servo feedback would report."""
    with SimRenderer(gui=False, dynamic=True) as r:
        arm = PyBulletArm(r)
        arm.connect()
        _clamp_static_box(r, arm)
        assert arm.read_joints()[5] > 8.0  # commanded 5.0; stalled open by the box


def test_jaws_report_contact_on_obstacle():
    """Closing on an object produces real contact points (the tangible grip signal / C data hook)."""
    with SimRenderer(gui=False, dynamic=True) as r:
        arm = PyBulletArm(r)
        arm.connect()
        box = _clamp_static_box(r, arm)
        count, force = r.contact_points(r.arm_body, box)
        assert count > 0
        assert force > 0.0
