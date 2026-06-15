"""Tests for dume.policy — ScriptedPolicy and LeRobotDiffusionPolicy stub.

Hardware/weight-dependent paths are marked with @pytest.mark.skip so CI stays green.
The NotImplementedError tests for the stub DO run (they assert the raise, which is
the current correct behaviour).
"""

from __future__ import annotations

import numpy as np
import pytest

from dume.arm import SimArm
from dume.dataset import Action, EpisodeRecorder, Observation, observe
from dume.kinematics import Kinematics
from dume.policy import LeRobotDiffusionPolicy, Policy, ScriptedPolicy


# ---------------------------------------------------------------------------
# Module-scoped Kinematics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(kin) -> Observation:
    """Return a minimal valid Observation using a fresh SimArm."""
    arm = SimArm()
    joints = arm.read_joints()
    ee_pose = kin.fk(joints)
    return Observation(joints=joints, ee_pose=ee_pose, t=0.0)


def _fixed_action() -> Action:
    return Action(joints_target=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 50.0]))


# ---------------------------------------------------------------------------
# test_scripted_policy_fixed_action
# ---------------------------------------------------------------------------


def test_scripted_policy_fixed_action(kin):
    """ScriptedPolicy with a fixed Action always returns that exact action."""
    fixed = _fixed_action()
    policy = ScriptedPolicy(fixed)

    obs = _make_obs(kin)
    result = policy.select_action(obs)

    np.testing.assert_array_equal(result.joints_target, fixed.joints_target)

    # Called twice — same result each time
    result2 = policy.select_action(obs)
    np.testing.assert_array_equal(result2.joints_target, fixed.joints_target)


# ---------------------------------------------------------------------------
# test_scripted_policy_callable
# ---------------------------------------------------------------------------


def test_scripted_policy_callable(kin):
    """ScriptedPolicy with a callable fn uses the observation to compute the action."""

    def fn(obs: Observation) -> Action:
        # Return joints shifted by +10 for first 3, unchanged for rest
        target = obs.joints.copy()
        target[:3] += 10.0
        return Action(joints_target=target)

    arm = SimArm(initial_joints=[5.0, -10.0, 20.0, 0.0, 0.0, 50.0])
    joints = arm.read_joints()
    obs = Observation(joints=joints, ee_pose=kin.fk(joints), t=0.0)

    policy = ScriptedPolicy(fn)
    result = policy.select_action(obs)

    expected = joints.copy()
    expected[:3] += 10.0
    np.testing.assert_allclose(result.joints_target, expected)


# ---------------------------------------------------------------------------
# test_scripted_policy_drives_recorder
# ---------------------------------------------------------------------------


def test_scripted_policy_drives_recorder(kin):
    """obs->action->record loop produces exactly N steps in the finished episode."""
    N = 5
    arm = SimArm()
    # Policy: always return current joints as target (hold-in-place)
    policy = ScriptedPolicy(lambda obs: Action(joints_target=obs.joints.copy()))

    rec = EpisodeRecorder()
    rec.start({"task": "policy_drive_test"})

    for i in range(N):
        obs = observe(arm, kin, t=float(i) * 0.1)
        act = policy.select_action(obs)
        rec.record(obs, act)

    episode = rec.finish()
    assert len(episode) == N

    # Spot-check: action should equal the joints we read (hold-in-place policy)
    for step in episode.steps:
        np.testing.assert_allclose(
            step.action.joints_target,
            step.observation.joints,
        )

    # load and train are no-ops — shouldn't raise
    policy.load("dummy_path")
    policy.train(None)


# ---------------------------------------------------------------------------
# test_lerobot_policy_stub_raises
# ---------------------------------------------------------------------------


def test_lerobot_policy_select_action_raises():
    """LeRobotDiffusionPolicy.select_action raises NotImplementedError (no weights)."""
    policy = LeRobotDiffusionPolicy(n_action_steps=8, horizon=16)

    arm = SimArm()
    joints = arm.read_joints()
    # Build a minimal Observation inline — no kin needed for this test
    ee_pose = np.eye(4)
    obs = Observation(joints=joints, ee_pose=ee_pose, t=0.0)

    with pytest.raises(NotImplementedError, match="load\\(\\)"):
        policy.select_action(obs)


def test_lerobot_policy_load_raises():
    """LeRobotDiffusionPolicy.load raises NotImplementedError."""
    policy = LeRobotDiffusionPolicy()
    with pytest.raises(NotImplementedError):
        policy.load("/some/path")


def test_lerobot_policy_train_raises():
    """LeRobotDiffusionPolicy.train raises NotImplementedError."""
    policy = LeRobotDiffusionPolicy()
    with pytest.raises(NotImplementedError):
        policy.train(None)


# ---------------------------------------------------------------------------
# Protocol structural subtype checks
# ---------------------------------------------------------------------------


def test_scripted_policy_satisfies_protocol():
    """ScriptedPolicy is structurally a Policy (runtime_checkable)."""
    assert isinstance(ScriptedPolicy(_fixed_action()), Policy)


def test_lerobot_policy_satisfies_protocol():
    """LeRobotDiffusionPolicy is structurally a Policy (runtime_checkable)."""
    assert isinstance(LeRobotDiffusionPolicy(), Policy)
