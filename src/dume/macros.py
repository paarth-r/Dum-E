"""Hand-guided macros: record the limp arm's measured joints, replay them tick-for-tick.

Recording is kinesthetic: torque is cut (``arm.relax()``) and the measured 6-vector is
sampled every tick while the user moves the arm by hand, so replaying the frames at the
recorded period reproduces the demonstrated motion in the same time frame. Playback
joint-moves to the first frame (same machinery as Home), streams the frames, then
re-seats the velocity-jog state on the last frame so teleop resumes with no jump.
Macros live in ``~/.dume/macros.json`` keyed by a single digit 0-9.
"""

from __future__ import annotations

import json
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MACRO_STORE = Path.home() / ".dume" / "macros.json"


@dataclass
class Macro:
    name: str
    key: str  # single digit "0".."9"
    dt: float  # tick period the frames were recorded at
    frames: np.ndarray  # (N, 6) commanded joints, arm.MOTOR_ORDER


class MacroStore:
    """Digit-keyed macro persistence, same shape as PoseStore: load-all, mutate, write-all."""

    def __init__(self, path: Path | str = DEFAULT_MACRO_STORE):
        self.path = Path(path)
        self._macros: dict[str, Macro] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._macros = {
                k: Macro(name=v["name"], key=k, dt=float(v["dt"]), frames=np.asarray(v["frames"], dtype=float))
                for k, v in data.items()
            }

    def save_to_disk(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            k: {"name": m.name, "dt": m.dt, "frames": m.frames.tolist()}
            for k, m in sorted(self._macros.items())
        }
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, key: str) -> Macro | None:
        return self._macros.get(key)

    def set(self, macro: Macro) -> None:
        if len(macro.key) != 1 or macro.key not in "0123456789":
            raise ValueError(f"Macro key must be a single digit 0-9, got {macro.key!r}")
        self._macros[macro.key] = macro
        self.save_to_disk()

    def items(self) -> list[tuple[str, Macro]]:
        return sorted(self._macros.items())


class RawKeys:
    """Non-blocking single-key reads from the controlling terminal, raw mode while open.

    ``get()`` returns one pressed character or None. Degrades to a no-op when stdin is not
    a TTY (tests, pipes) so the teleop loop never has to care.
    """

    def __init__(self):
        self._fd = None
        self._saved = None

    def __enter__(self) -> "RawKeys":
        if sys.stdin.isatty():
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # cbreak, not raw: keep Ctrl-C working
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._fd = None

    def get(self) -> str | None:
        if self._fd is None:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if ready else None


def trim_idle(frames: np.ndarray, *, threshold_deg: float = 0.3, margin: int = 5) -> np.ndarray:
    """Drop still frames from both ends of a recording: the hand travelling to the arm after
    the countdown, and the reach for the spacebar at the end — both replay as dead air
    otherwise. A frame is "moving" when any joint changes more than ``threshold_deg`` from
    the previous tick (well above servo feedback jitter); ``margin`` frames of lead-in and
    tail are kept so the motion doesn't start abruptly."""
    frames = np.asarray(frames)
    if len(frames) < 2:
        return frames
    delta = np.abs(np.diff(frames, axis=0)).max(axis=1)
    moving = np.where(delta > threshold_deg)[0]
    if len(moving) == 0:
        return frames[:1]  # a pure hold: keep the pose, drop the wait
    start = max(0, int(moving[0]) - margin)
    end = min(len(frames), int(moving[-1]) + 2 + margin)
    return frames[start:end]


def play_macro(dume, macro: Macro, *, abort=None) -> bool:
    """Go to the macro's start frame, replay it in recorded time, hold the end pose.

    ``abort`` is polled once per tick (both phases); returning True stops the motion where
    it is. Either way the controller state is re-seated on the final commanded frame, so
    the caller can hand control straight back to teleop with no jump. Returns True if the
    macro ran to completion.
    """
    from dume.arm import SO101Arm

    frames = macro.frames
    ctl = dume.controller
    ok = True

    dume.goto_joints(frames[0], on_tick=(lambda: not abort()) if abort else None)
    if abort and ctl._joint_target is None and not np.allclose(ctl.q_ref[:5], frames[0][:5], atol=1.0):
        ok = False  # goto was aborted mid-move

    if ok:
        period = macro.dt
        for frame in frames[1:]:
            t0 = time.perf_counter()
            if abort is not None and abort():
                ok = False
                break
            dume.arm.write_joints(frame)
            ctl.q_ref = frame.copy()
            if isinstance(dume.arm, SO101Arm):
                sleep = period - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    # Hold wherever we ended (completion or abort): gripper + all velocity-jog state.
    ctl.gripper_cmd = float(ctl.q_ref[5])
    ctl._sync_to_joints(ctl.q_ref.copy())
    return ok
