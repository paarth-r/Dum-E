"""Dynamic sim: objects with mass fall, and the magnet grasp holds/releases them.

Headless (DIRECT) + dynamic=True. The grasp is a fixed constraint to a gripper link, so while
held the box-to-gripper distance is preserved rigidly regardless of arm motion; releasing lets
it fall under gravity.
"""

import numpy as np
import pybullet as p
import pytest

from dume.poses import HOME_JOINTS
from dume.sim_world import SceneObject, SimRenderer, SimScene


def _box_pos(r, bid):
    return np.asarray(p.getBasePositionAndOrientation(bid, physicsClientId=r.client)[0])


def test_dynamic_box_falls_to_plane():
    with SimRenderer(gui=False, dynamic=True) as r:
        s = SimScene()
        s.add(SceneObject("b", "box", half_extents=[0.025, 0.025, 0.025], position=[0.3, 0.0, 0.5], mass=0.1))
        r.load_scene(s)
        bid = r.scene_bodies["b"]
        for _ in range(480):
            r.step_physics()
        z = _box_pos(r, bid)[2]
        assert z < 0.5  # it fell
        assert z == pytest.approx(0.025, abs=0.02)  # resting on the plane at half-height


def test_grasp_holds_then_release_drops():
    with SimRenderer(gui=False, dynamic=True) as r:
        s = SimScene()
        s.add(SceneObject("b", "box", half_extents=[0.02, 0.02, 0.02], position=[0.25, 0.0, 0.10], mass=0.05))
        r.load_scene(s)
        bid = r.scene_bodies["b"]
        r.set_joints(HOME_JOINTS)

        r.attach(bid)
        assert r.holding
        gli = r.link_index("gripper_frame_link")
        link0 = np.asarray(r._link_world_pose(gli)[0])
        d0 = np.linalg.norm(_box_pos(r, bid) - link0)

        # Move the arm; a rigid grasp must preserve the box-to-gripper distance.
        q = HOME_JOINTS.copy()
        q[1] += 25.0  # shoulder_lift — raises the gripper
        r.set_joints(q)
        for _ in range(60):
            r.step_physics()
        link1 = np.asarray(r._link_world_pose(gli)[0])
        d1 = np.linalg.norm(_box_pos(r, bid) - link1)
        assert d1 == pytest.approx(d0, abs=0.02)  # box tracked the gripper

        # Release: the box falls and settles on the ground plane (at its half-extent height).
        r.release()
        assert not r.holding
        for _ in range(600):
            r.step_physics()
        assert _box_pos(r, bid)[2] == pytest.approx(0.02, abs=0.015)
