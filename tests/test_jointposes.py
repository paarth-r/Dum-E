"""JointPoseStore round-trips, and DumeArm.goto_joints reaching a saved config (SimArm)."""

import json

import numpy as np
import pytest

from dume.arm import MOTOR_ORDER, SimArm
from dume.input_xbox import Command
from dume.poses import HOME_JOINTS, JointPoseStore
from dume.service import DumeArm


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "joint_poses.json"
    store = JointPoseStore(path)
    joints = np.array([10.0, -15.0, 25.0, 5.0, -30.0, 40.0])
    store.set("start", joints)

    reloaded = JointPoseStore(path)
    assert reloaded.has("start")
    assert np.allclose(reloaded.get("start"), joints)


def test_file_is_readable_motor_dict(tmp_path):
    path = tmp_path / "joint_poses.json"
    store = JointPoseStore(path)
    joints = np.arange(6, dtype=float)
    store.set("pickup", joints)

    data = json.loads(path.read_text())
    assert set(data["pickup"]) == set(MOTOR_ORDER)
    assert data["pickup"]["shoulder_pan"] == 0.0
    assert data["pickup"]["gripper"] == 5.0


def test_accepts_bare_list_entries(tmp_path):
    path = tmp_path / "joint_poses.json"
    path.write_text(json.dumps({"start": [1, 2, 3, 4, 5, 6]}))
    store = JointPoseStore(path)
    assert np.allclose(store.get("start"), [1, 2, 3, 4, 5, 6])


def test_set_rejects_wrong_length(tmp_path):
    store = JointPoseStore(tmp_path / "joint_poses.json")
    with pytest.raises(ValueError):
        store.set("bad", [1, 2, 3])


def test_multiple_named_setpoints(tmp_path):
    store = JointPoseStore(tmp_path / "joint_poses.json")
    store.set("start", np.zeros(6))
    store.set("pickup", np.ones(6))
    assert store.names() == ["pickup", "start"]


def test_goto_joints_reaches_target():
    target = np.array([12.0, -25.0, 30.0, 8.0, -10.0, 70.0])
    with DumeArm(dry_run=True) as arm:
        arm.goto_joints(target, wait=True)
        reached = arm.get_joints()
    # 5 arm joints driven straight to target; gripper restored from element 6.
    assert np.allclose(reached[:5], target[:5], atol=0.5)
    assert reached[5] == pytest.approx(target[5])


def test_run_start_pose_wiring(tmp_path):
    """Replicate cmd_run's launch block: a saved 'start' pose is reached and becomes home."""
    store = JointPoseStore(tmp_path / "joint_poses.json")
    start = np.array([8.9, -2.5, 5.7, 21.6, 2.0, 30.0])
    store.set("start", start)

    with DumeArm(dry_run=True) as arm:
        assert store.has("start")
        arm.goto_joints(store.get("start"))
        arm.controller.home_joints = store.get("start")  # 'home' returns here
        reached = arm.get_joints()

    assert np.allclose(reached[:5], start[:5], atol=0.5)  # arm joints exact
    assert np.allclose(arm.controller.home_joints, start)  # home now points at start


def test_start_pose_holds_when_velocity_jog_resumes():
    """Regression: after a joint move (start pose / home), idle velocity-jog must NOT drift
    back to the pre-move config. The bug was stale velocity-jog state (_pivot_target etc.)
    left pointing at the startup pose, so the first zero-command tick dragged the arm back."""
    start = np.array([25.0, -10.0, 15.0, 18.0, 5.0, 30.0])
    with DumeArm(dry_run=True) as arm:  # SimArm boots at HOME_JOINTS (the "curled" pose)
        arm.goto_joints(start, wait=True)
        for _ in range(100):  # simulate teleop idle ticks (VELOCITY mode, no input)
            arm.controller.step(Command())
        held = arm.get_joints()
    assert np.allclose(held[:5], start[:5], atol=1.0), held


def test_sim_relax_is_noop():
    SimArm(initial_joints=HOME_JOINTS.copy()).relax()  # must not raise
