"""Xbox controller input via pygame, shaped into a generic :class:`Command`.

Input shaping (deadzone + cubic expo) lives here; velocity *scaling* and integration live in
the controller. Buttons are edge-triggered (a flag is True only on the press that started it)
so one push = one action.

Semantic mapping (the only bindings):
- Left stick    -> base-plane translation (up = +X forward, left/right = +/-Y)
- Right stick Y -> Z (up = +Z)
- D-pad up/down -> wrist pitch (wrist_flex), while held
- D-pad left/right -> wrist roll (wrist_roll), while held
- LT / RT       -> gripper close / open (proportional)
- A             -> toggle velocity / pose (freeze) mode
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dume.config import XboxMap


@dataclass
class Command:
    """One control sample. Stick axes are shaped, normalised to [-1, 1]; wrist is -1/0/+1."""

    lin: np.ndarray = field(default_factory=lambda: np.zeros(3))  # x, y, z
    wrist_pitch: float = 0.0  # signed wrist_flex jog (D-pad down = +, up = -)
    wrist_roll: float = 0.0  # signed wrist_roll jog (D-pad left = +, right = -)
    gripper: float = 0.0  # + open, - close
    toggle_mode: bool = False


def apply_deadzone(v: float, dz: float) -> float:
    if abs(v) <= dz:
        return 0.0
    return (v - np.sign(v) * dz) / (1.0 - dz)


def apply_expo(v: float, expo: float) -> float:
    """Blend linear and cubic: expo=0 -> linear, expo=1 -> pure cubic. Smoother near centre."""
    return (1.0 - expo) * v + expo * v**3


def shape_axis(v: float, dz: float, expo: float) -> float:
    return apply_expo(apply_deadzone(v, dz), expo)


class XboxController:
    """Polls a pygame joystick and returns a shaped :class:`Command` each tick."""

    def __init__(self, mapping: XboxMap | None = None, joystick_index: int = 0):
        self.map = mapping or XboxMap()
        self.joystick_index = joystick_index
        self._js = None
        self._prev_buttons: dict[int, bool] = {}

    def connect(self) -> None:
        import os

        # We never open a window; force SDL's dummy video driver so init can't block or
        # pop up a window in a headless terminal.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        pygame.display.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No joystick detected. Pair the Xbox controller (System Settings > Bluetooth) "
                "and confirm with `dume axes`."
            )
        self._js = pygame.joystick.Joystick(self.joystick_index)
        self._js.init()

    @property
    def name(self) -> str:
        return self._js.get_name() if self._js else "<disconnected>"

    def _axis(self, idx: int) -> float:
        try:
            return float(self._js.get_axis(idx))
        except Exception:
            return 0.0

    def _button(self, idx: int) -> bool:
        try:
            return bool(self._js.get_button(idx))
        except Exception:
            return False

    def _rising(self, idx: int) -> bool:
        cur = self._button(idx)
        prev = self._prev_buttons.get(idx, False)
        self._prev_buttons[idx] = cur
        return cur and not prev

    def _trigger(self, idx: int) -> float:
        # Triggers rest near -1, full press near +1 -> normalise to [0, 1].
        n = (self._axis(idx) + 1.0) / 2.0
        n = float(np.clip(n, 0.0, 1.0))
        return 0.0 if n < self.map.trigger_deadzone else n

    def poll(self) -> Command:
        import pygame

        pygame.event.pump()
        m = self.map
        dz, ex = m.deadzone, m.expo

        lin = np.array(
            [
                -shape_axis(self._axis(m.axis_left_y), dz, ex),  # up = +X forward
                -shape_axis(self._axis(m.axis_left_x), dz, ex),  # left = +Y
                -shape_axis(self._axis(m.axis_right_y), dz, ex),  # up = +Z
            ]
        )
        # D-pad held (continuous), not edge-triggered. Signs match observed arm direction.
        wrist_pitch = (1.0 if self._button(m.btn_dpad_down) else 0.0) - (
            1.0 if self._button(m.btn_dpad_up) else 0.0
        )
        wrist_roll = (1.0 if self._button(m.btn_dpad_left) else 0.0) - (
            1.0 if self._button(m.btn_dpad_right) else 0.0
        )
        gripper = self._trigger(m.axis_rt) - self._trigger(m.axis_lt)  # + open, - close

        return Command(
            lin=lin,
            wrist_pitch=wrist_pitch,
            wrist_roll=wrist_roll,
            gripper=gripper,
            toggle_mode=self._rising(m.btn_a),
        )

    def disconnect(self) -> None:
        try:
            import pygame

            if self._js:
                self._js.quit()
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
