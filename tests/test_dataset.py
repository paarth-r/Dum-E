"""Tests for dume.dataset — episode recording and LocalBackend persistence.

All tests use SimArm + Kinematics (no real hardware) and pytest tmp_path for I/O.
The to_lerobot stub test simply asserts the expected NotImplementedError.
"""

from __future__ import annotations

import numpy as np
import pytest

from dume.arm import SimArm
from dume.dataset import (
    Action,
    Episode,
    EpisodeRecorder,
    LocalBackend,
    Observation,
    Step,
    observe,
    to_lerobot,
)
from dume.kinematics import Kinematics


# ---------------------------------------------------------------------------
# Module-scoped Kinematics (loading the URDF takes ~0.5 s)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


# ---------------------------------------------------------------------------
# test_recorder_assembles_episode
# ---------------------------------------------------------------------------


def test_recorder_assembles_episode(kin):
    """EpisodeRecorder + observe() builds an Episode with the correct structure."""
    arm = SimArm()

    # Set up some deterministic joint positions to track
    joint_sequences = [
        np.array([10.0, -20.0, 30.0, -10.0, 5.0, 50.0]),
        np.array([15.0, -25.0, 35.0, -15.0, 10.0, 60.0]),
        np.array([20.0, -30.0, 40.0, -20.0, 15.0, 70.0]),
    ]
    N = len(joint_sequences)

    rec = EpisodeRecorder()
    rec.start({"task": "test_pick"})

    for joints in joint_sequences:
        arm.write_joints(joints)
        obs = observe(arm, kin, camera=None, t=0.1)
        act = Action(joints_target=joints + 1.0)   # trivial: shift by 1
        rec.record(obs, act)

    episode = rec.finish()

    # structural checks
    assert len(episode) == N
    assert isinstance(episode.steps[0], Step)
    assert episode.metadata["task"] == "test_pick"
    assert "id" in episode.metadata
    assert "created" in episode.metadata

    # joint values in Observation should match what we wrote
    for i, joints in enumerate(joint_sequences):
        np.testing.assert_allclose(episode.steps[i].observation.joints, joints)

    # action targets
    for i, joints in enumerate(joint_sequences):
        np.testing.assert_allclose(episode.steps[i].action.joints_target, joints + 1.0)

    # ee_pose is 4x4
    assert episode.steps[0].observation.ee_pose.shape == (4, 4)

    # camera fields are None (we passed camera=None)
    assert episode.steps[0].observation.depth is None
    assert episode.steps[0].observation.image is None
    assert episode.steps[0].observation.detections is None


# ---------------------------------------------------------------------------
# test_local_backend_roundtrip
# ---------------------------------------------------------------------------


def test_local_backend_roundtrip(kin, tmp_path):
    """LocalBackend write+read round-trips joints, ee_pose, action, and metadata."""
    arm = SimArm()
    rec = EpisodeRecorder()
    rec.start({"task": "roundtrip_test"})

    steps_data = [
        np.array([0.0, -10.0, 20.0, -5.0, 2.0, 50.0]),
        np.array([5.0, -15.0, 25.0, -10.0, 7.0, 55.0]),
    ]
    for i, joints in enumerate(steps_data):
        arm.write_joints(joints)
        obs = observe(arm, kin, t=float(i))
        act = Action(joints_target=joints * 0.9)
        rec.record(obs, act)

    original = rec.finish()
    backend = LocalBackend(root=tmp_path / "episodes")

    ep_id = backend.write(original)
    assert isinstance(ep_id, str) and len(ep_id) > 0

    restored = backend.read(ep_id)

    # step count
    assert len(restored) == len(original)

    # numeric round-trip
    for i in range(len(original)):
        np.testing.assert_allclose(
            restored.steps[i].observation.joints,
            original.steps[i].observation.joints,
        )
        np.testing.assert_allclose(
            restored.steps[i].observation.ee_pose,
            original.steps[i].observation.ee_pose,
        )
        np.testing.assert_allclose(
            restored.steps[i].action.joints_target,
            original.steps[i].action.joints_target,
        )

    # metadata preserved
    assert restored.metadata["task"] == "roundtrip_test"
    assert restored.metadata["id"] == ep_id

    # no depth/image (we didn't record any)
    assert restored.steps[0].observation.depth is None
    assert restored.steps[0].observation.image is None

    # list_ids returns the written id
    assert ep_id in backend.list_ids()


# ---------------------------------------------------------------------------
# test_local_backend_with_depth
# ---------------------------------------------------------------------------


def test_local_backend_with_depth(kin, tmp_path):
    """Depth arrays survive a write+read round-trip via LocalBackend."""
    arm = SimArm()

    H, W = 8, 12   # small synthetic depth map
    depth_a = np.random.default_rng(0).uniform(0.3, 1.5, size=(H, W)).astype(np.float32)
    depth_b = np.random.default_rng(1).uniform(0.3, 1.5, size=(H, W)).astype(np.float32)

    joints_a = np.array([0.0, -10.0, 20.0, -5.0, 2.0, 50.0])
    joints_b = np.array([5.0, -15.0, 25.0, -10.0, 7.0, 55.0])

    arm.write_joints(joints_a)
    obs_a = Observation(
        joints=arm.read_joints(),
        ee_pose=kin.fk(arm.read_joints()),
        depth=depth_a,
        t=0.0,
    )
    act_a = Action(joints_target=joints_a)

    arm.write_joints(joints_b)
    obs_b = Observation(
        joints=arm.read_joints(),
        ee_pose=kin.fk(arm.read_joints()),
        depth=depth_b,
        t=0.1,
    )
    act_b = Action(joints_target=joints_b)

    episode = Episode(
        steps=[Step(obs_a, act_a), Step(obs_b, act_b)],
        metadata={"task": "depth_test"},
    )

    backend = LocalBackend(root=tmp_path / "episodes")
    ep_id = backend.write(episode)
    restored = backend.read(ep_id)

    assert restored.steps[0].observation.depth is not None
    np.testing.assert_allclose(
        restored.steps[0].observation.depth, depth_a, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        restored.steps[1].observation.depth, depth_b, rtol=1e-5, atol=1e-6
    )


# ---------------------------------------------------------------------------
# test_to_lerobot_stub_raises
# ---------------------------------------------------------------------------


def test_to_lerobot_stub_raises():
    """to_lerobot() raises NotImplementedError — it is a pending stub."""
    dummy_episode = Episode(steps=[], metadata={"id": "test"})
    with pytest.raises(NotImplementedError):
        to_lerobot(dummy_episode, repo_id="paarth-r/test")
