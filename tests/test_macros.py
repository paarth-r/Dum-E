"""Macro record/playback — store round-trip, replay fidelity, and clean teleop resume."""

import numpy as np
import pytest

from dume.input_xbox import Command
from dume.macros import Macro, MacroStore, play_macro
from dume.service import DumeArm


@pytest.fixture
def dume(tmp_path):
    from dume.poses import PoseStore

    arm = DumeArm(dry_run=True, poses=PoseStore(tmp_path / "poses.json"))
    arm.connect()
    yield arm
    arm.disconnect()


def _recorded_macro(dume, ticks: int = 30) -> Macro:
    """Capture measured joints while the arm moves, exactly as dume record does. The sim
    arm is jogged to generate the motion a hand would; recording only reads joints."""
    frames = []
    for _ in range(ticks):
        dume.controller.step(Command(lin=np.array([0.0, 1.0, 0.0]), rt=1.0))
        frames.append(dume.get_joints().copy())
    return Macro(name="test", key="3", dt=dume.config.dt, frames=np.array(frames))


def test_store_round_trip(tmp_path):
    store = MacroStore(tmp_path / "macros.json")
    m = Macro(name="wave", key="7", dt=0.02, frames=np.arange(18, dtype=float).reshape(3, 6))
    store.set(m)
    reloaded = MacroStore(tmp_path / "macros.json").get("7")
    assert reloaded.name == "wave"
    assert reloaded.dt == pytest.approx(0.02)
    assert np.allclose(reloaded.frames, m.frames)


def test_store_rejects_bad_key(tmp_path):
    store = MacroStore(tmp_path / "macros.json")
    with pytest.raises(ValueError):
        store.set(Macro(name="x", key="a", dt=0.02, frames=np.zeros((2, 6))))


def test_store_get_missing_returns_none(tmp_path):
    assert MacroStore(tmp_path / "macros.json").get("5") is None


def test_playback_reaches_end_pose_from_elsewhere(dume):
    macro = _recorded_macro(dume)
    # Wander somewhere else so playback must first return to the macro's start frame.
    for _ in range(30):
        dume.controller.step(Command(lin=np.array([0.0, -1.0, 0.5])))
    play_macro(dume, macro)
    assert np.allclose(dume.get_joints()[:5], macro.frames[-1][:5], atol=0.6)


def test_playback_resumes_teleop_without_jump(dume):
    macro = _recorded_macro(dume)
    for _ in range(20):
        dume.controller.step(Command(lin=np.array([0.0, -1.0, 0.0])))
    play_macro(dume, macro)
    end = dume.get_joints().copy()
    for _ in range(5):
        dume.controller.step(Command(rt=1.0))  # hold trigger as recorded; sticks centred
    settled = dume.get_joints().copy()
    for _ in range(20):
        dume.controller.step(Command(rt=1.0))
    # A sub-degree one-tick settle (pivot re-seats through the IK) is fine; jogging back
    # toward stale pre-macro state is not, and neither is ongoing drift.
    assert np.allclose(settled[:5], end[:5], atol=1.0)
    assert np.allclose(dume.get_joints()[:5], settled[:5], atol=1e-6)


def test_playback_abort_holds_and_resyncs(dume):
    macro = _recorded_macro(dume)
    ticks = iter(range(1000))
    play_macro(dume, macro, abort=lambda: next(ticks) > 5)  # abort a few frames in
    held = dume.get_joints().copy()
    for _ in range(10):
        dume.controller.step(Command(rt=1.0))
    assert np.allclose(dume.get_joints()[:5], held[:5], atol=0.6)


def test_trim_idle_cuts_still_head_and_tail():
    from dume.macros import trim_idle

    still = np.tile(np.array([10.0, 20, 30, 0, 0, 50]), (60, 1))
    moving = still[0] + np.linspace(0, 15, 40)[:, None] * np.array([1, 0.5, 0, 0, 0, 0])
    frames = np.vstack([still, moving, np.tile(moving[-1], (80, 1))])
    trimmed = trim_idle(frames, margin=5)
    # Head: at most the margin of still frames survives; tail likewise.
    assert len(trimmed) <= 40 + 2 * 5 + 2
    assert np.allclose(trimmed[-1], moving[-1])
    # The motion itself is intact.
    assert np.allclose(trimmed[0], still[0])


def test_trim_idle_tolerates_servo_noise():
    from dume.macros import trim_idle

    rng = np.random.default_rng(0)
    base = np.tile(np.array([10.0, 20, 30, 0, 0, 50]), (100, 1))
    noisy_still = base + rng.normal(0, 0.05, size=base.shape)  # feedback jitter, no motion
    assert len(trim_idle(noisy_still)) < 20  # jitter alone must not count as motion


def test_trim_idle_all_still_keeps_a_frame():
    from dume.macros import trim_idle

    frames = np.tile(np.arange(6, dtype=float), (30, 1))
    trimmed = trim_idle(frames)
    assert len(trimmed) >= 1


def test_out_of_limit_macro_still_plays_and_returns_teleop(dume):
    """Hand-recorded frames can sit past the software joint limits (torque was off). The
    goto-start phase must not wedge on an unreachable target, and teleop must come back."""
    lim = dume.controller.joint_limits
    frames = np.tile(dume.get_joints().copy(), (20, 1))
    frames[:, 1] = np.linspace(lim[1, 0] - 10.0, lim[1, 0] - 8.0, 20)  # beyond the limit
    macro = Macro(name="pushed", key="9", dt=dume.config.dt, frames=frames)
    play_macro(dume, macro)
    assert dume.controller._joint_target is None  # teleop branch must be live again
    before = dume.get_joints().copy()
    for _ in range(30):
        dume.controller.step(Command(lin=np.array([0.0, 1.0, 0.0]), rt=1.0))
    assert not np.allclose(dume.get_joints()[:5], before[:5], atol=0.5)  # sticks work


def test_goto_joints_clamps_target_and_arrives(dume):
    lim = dume.controller.joint_limits
    target = dume.get_joints().copy()
    target[1] = lim[1, 0] - 10.0  # 10 deg past the shoulder_lift limit
    dume.goto_joints(target, timeout=10.0)
    assert dume.controller._joint_target is None  # arrived at the clamped target, no timeout
    assert dume.get_joints()[1] == pytest.approx(lim[1, 0], abs=0.6)
