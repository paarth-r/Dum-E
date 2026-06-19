"""Keyboard -> Command mapping. No GUI: PyBullet's getKeyboardEvents is monkeypatched."""

import numpy as np

import dume.input_keyboard as ik
from dume.input_keyboard import KeyboardController


def _poll_with(monkeypatch, events):
    monkeypatch.setattr(ik.p, "getKeyboardEvents", lambda **_: events)
    return KeyboardController(client=0).poll()


def test_wasd_maps_to_xy(monkeypatch):
    down = ik.p.KEY_IS_DOWN
    assert _poll_with(monkeypatch, {ord("w"): down}).lin[0] == 1.0
    assert _poll_with(monkeypatch, {ord("s"): down}).lin[0] == -1.0
    assert _poll_with(monkeypatch, {ord("a"): down}).lin[1] == 1.0
    assert _poll_with(monkeypatch, {ord("d"): down}).lin[1] == -1.0


def test_rf_maps_to_z(monkeypatch):
    down = ik.p.KEY_IS_DOWN
    assert _poll_with(monkeypatch, {ord("r"): down}).lin[2] == 1.0
    assert _poll_with(monkeypatch, {ord("f"): down}).lin[2] == -1.0


def test_arrows_map_to_wrist(monkeypatch):
    down = ik.p.KEY_IS_DOWN
    assert _poll_with(monkeypatch, {ik.p.B3G_DOWN_ARROW: down}).wrist_pitch == 1.0
    assert _poll_with(monkeypatch, {ik.p.B3G_UP_ARROW: down}).wrist_pitch == -1.0
    assert _poll_with(monkeypatch, {ik.p.B3G_LEFT_ARROW: down}).wrist_roll == 1.0
    assert _poll_with(monkeypatch, {ik.p.B3G_RIGHT_ARROW: down}).wrist_roll == -1.0


def test_gripper_hold_keys(monkeypatch):
    down = ik.p.KEY_IS_DOWN
    assert _poll_with(monkeypatch, {ord("o"): down}).lt == 1.0  # open
    assert _poll_with(monkeypatch, {ord("c"): down}).rt == 1.0  # close


def test_tap_keys_mode_and_gripper_mode(monkeypatch):
    trig = ik.p.KEY_WAS_TRIGGERED
    assert _poll_with(monkeypatch, {ord("m"): trig}).toggle_mode is True
    assert _poll_with(monkeypatch, {ord("g"): trig}).gripper_mode_toggle is True


def test_idle_is_zero(monkeypatch):
    cmd = _poll_with(monkeypatch, {})
    assert np.allclose(cmd.lin, 0) and cmd.wrist_pitch == 0 and cmd.lt == 0 and cmd.rt == 0
    assert cmd.toggle_mode is False and cmd.gripper_mode_toggle is False
