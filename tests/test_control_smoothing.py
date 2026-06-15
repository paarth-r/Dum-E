"""Teleop-smoothness contracts for the velocity-jog rework.

The fix: the controller integrates an internal *commanded* reference (``q_ref``) and slews from
it, so noisy servo feedback never reaches the command stream; plus a damped-least-squares
position solver and a jerk (acceleration) limit. These tests pin that behaviour using a
``SimArm`` with injected servo noise to stand in for real Feetech feedback.
"""

import numpy as np
import pytest

from dume import geometry as g
from dume.arm import SimArm
from dume.config import ControllerConfig
from dume.controller import Controller
from dume.input_xbox import Command
from dume.kinematics import Kinematics
from dume.poses import HOME_JOINTS, PoseStore


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


def _controller(kin, tmp_path, *, noise=0.0):
    cfg = ControllerConfig()
    arm = SimArm(initial_joints=HOME_JOINTS.copy(), servo_noise_deg=noise, seed=1)
    ctl = Controller(cfg, arm, kin, PoseStore(tmp_path / "poses.json"))
    ctl.start()
    return ctl


def test_dls_position_ik_recovers_target(kin):
    """The DLS solver drives the wrist pivot to a reachable position (within 1 mm)."""
    kin_pos = Kinematics(joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex"])
    q_true = np.array([15.0, -25.0, 30.0])
    target = kin_pos.fk(q_true)[:3, 3]
    q_solved = kin_pos.ik_position_dls(np.zeros(3), target, damping=0.02, iters=40)
    achieved = kin_pos.fk(q_solved)[:3, 3]
    assert np.linalg.norm(achieved - target) < 1e-3


def test_commands_ignore_servo_noise(kin, tmp_path):
    """Holding zero command, the *commanded* joints barely move despite 0.5 deg read noise —
    proof the loop no longer chases measured feedback."""
    ctl = _controller(kin, tmp_path, noise=0.5)
    sent = np.array([ctl.step(Command()).joints_sent.copy() for _ in range(60)])
    assert np.max(np.std(sent[:, :5], axis=0)) < 0.05


def test_velocity_jog_moves_forward_under_noise(kin, tmp_path):
    """Even with noisy feedback the jog still tracks +X cleanly."""
    ctl = _controller(kin, tmp_path, noise=0.5)
    p0 = g.position_of(ctl.kin.fk(ctl.q_ref))
    for _ in range(40):
        ctl.step(Command(lin=np.array([1.0, 0.0, 0.0])))
    p1 = g.position_of(ctl.kin.fk(ctl.q_ref))
    assert p1[0] - p0[0] > 0.01


def test_jerk_limit_bounds_acceleration(kin, tmp_path):
    """Per-tick joint velocity changes by no more than the jerk cap, even on a slammed command."""
    ctl = _controller(kin, tmp_path, noise=0.0)
    cfg = ctl.config
    prev_q = ctl.q_ref[:5].copy()
    prev_v = np.zeros(5)
    for _ in range(40):
        q = ctl.step(Command(lin=np.array([1.0, 1.0, 1.0]), wrist_pitch=1.0, wrist_roll=1.0)).joints_sent
        v = q[:5] - prev_q
        assert np.all(np.abs(v - prev_v) <= cfg.joint_jerk_deg + 1e-6)
        prev_v = v
        prev_q = q[:5].copy()
