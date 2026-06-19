"""Keyboard control for the PyBullet sim — a drop-in alternative to the Xbox pad.

Produces the same :class:`dume.input_xbox.Command` that the controller consumes, read from the
PyBullet GUI window's keyboard events (``getKeyboardEvents``), so no extra dependency and the
window that shows the arm is the one that captures keys. Held keys jog; tapped keys toggle.

Keymap::

    W / S      move +X / -X        (forward / back)
    A / D      move +Y / -Y        (left / right)
    R / F      move +Z / -Z        (up / down)
    Up / Down  wrist pitch         (wrist_flex)
    Left/Right wrist roll          (wrist_roll)
    O / C      gripper open / close (lt/rt; in SQUEEZE, hold C = closed, release = open)
    G          toggle gripper mode (squeeze / rate) (tap)
    M          toggle velocity / freeze mode (tap)
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from dume.input_xbox import Command


class KeyboardController:
    """Polls the PyBullet GUI keyboard and emits :class:`Command`s. ``poll`` matches the
    XboxController interface so it can be handed straight to ``DumeArm.run_teleop``."""

    name = "keyboard"

    def __init__(self, client: int):
        self._client = client
        # Last raw keyboard-event dict from poll(). getKeyboardEvents() consumes events on read,
        # so other consumers in the same tick (e.g. the camera nav's Ctrl check) read this instead.
        self.last_keys: dict[int, int] = {}

    def connect(self) -> None:  # parity with XboxController
        pass

    def disconnect(self) -> None:
        pass

    def poll(self) -> Command:
        keys = p.getKeyboardEvents(physicsClientId=self._client)
        self.last_keys = keys

        def down(code: int) -> bool:
            return bool(keys.get(code, 0) & p.KEY_IS_DOWN)

        def tapped(code: int) -> bool:
            return bool(keys.get(code, 0) & p.KEY_WAS_TRIGGERED)

        lin = np.zeros(3)
        lin[0] = (1.0 if down(ord("w")) else 0.0) - (1.0 if down(ord("s")) else 0.0)
        lin[1] = (1.0 if down(ord("a")) else 0.0) - (1.0 if down(ord("d")) else 0.0)
        lin[2] = (1.0 if down(ord("r")) else 0.0) - (1.0 if down(ord("f")) else 0.0)

        # D-pad parity: down = +pitch, up = -pitch; left = +roll, right = -roll.
        wrist_pitch = (1.0 if down(p.B3G_DOWN_ARROW) else 0.0) - (1.0 if down(p.B3G_UP_ARROW) else 0.0)
        wrist_roll = (1.0 if down(p.B3G_LEFT_ARROW) else 0.0) - (1.0 if down(p.B3G_RIGHT_ARROW) else 0.0)

        # Triggers as absolute positions: O = open (lt), C = close (rt). Works in both gripper
        # modes (SQUEEZE: rt is absolute, so hold C = closed, release = open; RATE: integrated).
        lt = 1.0 if down(ord("o")) else 0.0
        rt = 1.0 if down(ord("c")) else 0.0

        return Command(
            lin=lin,
            wrist_pitch=wrist_pitch,
            wrist_roll=wrist_roll,
            lt=lt,
            rt=rt,
            toggle_mode=tapped(ord("m")),
            gripper_mode_toggle=tapped(ord("g")),
        )
