"""Controller behaviour over a SimArm — no hardware, no pygame.

These assert the safety/smoothness contracts: bounded joint steps (continuity), workspace
clamping, joint-limit clamping, and that velocity jog + goto actually move the end-effector
the right way.
"""

import numpy as np
import pytest

from dume import geometry as g
from dume.arm import SimArm
from dume.config import ControllerConfig, ControlMode
from dume.controller import Controller
from dume.input_xbox import Command
from dume.kinematics import Kinematics
from dume.poses import HOME_JOINTS, PoseStore


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


@pytest.fixture
def controller(kin, tmp_path):
    cfg = ControllerConfig()
    arm = SimArm(initial_joints=HOME_JOINTS.copy())
    ctl = Controller(cfg, arm, kin, PoseStore(tmp_path / "poses.json"))
    ctl.start()
    return ctl


def test_zero_command_holds_still(controller):
    start = controller.arm.read_joints().copy()
    for _ in range(20):
        controller.step(Command())
    assert np.allclose(controller.arm.read_joints(), start, atol=1e-6)


def test_velocity_jog_moves_forward_in_x(controller):
    p0 = g.position_of(controller.kin.fk(controller.arm.read_joints()))
    for _ in range(40):
        controller.step(Command(lin=np.array([1.0, 0.0, 0.0])))  # +X
    p1 = g.position_of(controller.kin.fk(controller.arm.read_joints()))
    assert p1[0] - p0[0] > 0.01  # moved forward at least 1 cm


def test_joint_steps_are_bounded(controller):
    cfg = controller.config
    prev = controller.arm.read_joints().copy()
    # Slam full command; per-tick joint change must stay within the slew limit.
    for _ in range(60):
        controller.step(Command(lin=np.array([1.0, 1.0, 1.0]), wrist_pitch=1.0, wrist_roll=1.0))
        now = controller.arm.read_joints()
        assert np.all(np.abs(now[:5] - prev[:5]) <= cfg.joint_slew_deg + 1e-6)
        prev = now.copy()


def test_target_stays_in_workspace(controller):
    cfg = controller.config
    for _ in range(300):  # push hard toward +X +Y +Z for a long time
        controller.step(Command(lin=np.array([1.0, 1.0, 1.0])))
    tx, ty, tz = controller._pivot_target  # the integrated wrist-pivot target is what's clamped
    assert cfg.workspace.x[0] - 1e-6 <= tx <= cfg.workspace.x[1] + 1e-6
    assert cfg.workspace.y[0] - 1e-6 <= ty <= cfg.workspace.y[1] + 1e-6
    assert cfg.workspace.z[0] - 1e-6 <= tz <= cfg.workspace.z[1] + 1e-6


def test_joints_stay_within_limits(controller):
    lim = controller.joint_limits
    for _ in range(300):
        controller.step(Command(lin=np.array([1.0, -1.0, 1.0]), wrist_pitch=-1.0, wrist_roll=1.0))
        q = controller.arm.read_joints()
        assert np.all(q[:5] >= lim[:5, 0] - 1e-6)
        assert np.all(q[:5] <= lim[:5, 1] + 1e-6)


def test_dpad_pitch_jogs_wrist_flex_without_moving_pivot(controller):
    pivot0 = controller._pivot_target.copy()
    wf0 = controller.arm.read_joints()[3]
    for _ in range(30):
        controller.step(Command(wrist_pitch=1.0))  # D-pad up
    q = controller.arm.read_joints()
    assert q[3] - wf0 > 5.0  # wrist_flex moved
    assert np.allclose(controller._pivot_target, pivot0, atol=1e-9)  # pivot held


def test_dpad_roll_jogs_wrist_roll(controller):
    wr0 = controller.arm.read_joints()[4]
    for _ in range(30):
        controller.step(Command(wrist_roll=1.0))  # D-pad right
    assert controller.arm.read_joints()[4] - wr0 > 5.0


def test_home_returns_to_start_config(controller):
    start = controller.home_joints.copy()
    for _ in range(40):
        controller.step(Command(lin=np.array([1.0, 1.0, -1.0]), wrist_pitch=1.0))
    controller._joint_target = controller.home_joints.copy()
    for _ in range(300):
        controller.step(Command())
        if controller._joint_target is None:
            break
    assert np.max(np.abs(controller.arm.read_joints()[:5] - start[:5])) < 0.6


def test_gripper_opens_and_closes(controller):
    cfg = controller.config
    for _ in range(80):
        controller.step(Command(gripper=1.0))  # open
    assert controller.gripper_cmd >= cfg.gripper_open - 1.0
    for _ in range(120):
        controller.step(Command(gripper=-1.0))  # close
    assert controller.gripper_cmd <= cfg.gripper_closed + 1.0


def test_goto_reaches_pose(controller):
    target = controller.kin.fk(np.array([20.0, -25.0, 30.0, 10.0, -15.0, 50.0]))
    controller._goto_pose(target, controller.arm.read_joints())
    for _ in range(400):
        controller.step(Command())
        if controller._traj is None:
            break
    achieved = controller.kin.fk(controller.arm.read_joints())
    err = np.linalg.norm(g.position_of(achieved) - g.position_of(target))
    assert err < 5e-3  # within 5 mm


def test_mode_toggle(controller):
    assert controller.mode is ControlMode.VELOCITY
    controller.step(Command(toggle_mode=True))
    assert controller.mode is ControlMode.POSE
    controller.step(Command(toggle_mode=True))
    assert controller.mode is ControlMode.VELOCITY
