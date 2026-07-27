"""Live camera window for the end-effector Arducam.

Split so the interesting part is testable: :func:`annotate` is pure (frame in, annotated
frame out, no GUI), and :class:`LiveView` is the thin shell that actually touches cv2's
window functions. Tests exercise the former and never open a window.

Kept deliberately small — this is a viewfinder, not a UI.
"""

from __future__ import annotations

import numpy as np

from dume import geometry as g

_FONT_SCALE = 0.5
_LINE_H = 22
_MARGIN = 10


def pose_lines(pose: np.ndarray | None) -> list[str]:
    """Human-readable camera pose, in the arm base frame — the frame the cloud lives in."""
    if pose is None:
        return ["camera pose: unknown"]
    xyz = g.position_of(pose) * 1000.0
    rpy = np.rad2deg(g.rpy_of(pose))
    return [
        f"cam xyz  {xyz[0]:7.1f} {xyz[1]:7.1f} {xyz[2]:7.1f}  mm (base frame)",
        f"cam rpy  {rpy[0]:7.1f} {rpy[1]:7.1f} {rpy[2]:7.1f}  deg",
    ]


def annotate(frame: np.ndarray, lines: list[str] | None = None) -> np.ndarray:
    """Return a BGR copy of ``frame`` with ``lines`` drawn top-left.

    Accepts single-channel (the Arducam is mono) or BGR. Never mutates the input — the same
    frame object is handed to every consumer by the grabber's latest-frame slot, so drawing
    in place would corrupt what the point-cloud path sees.
    """
    import cv2

    img = np.asarray(frame)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img.copy()

    for i, text in enumerate(lines or []):
        y = _MARGIN + _LINE_H * (i + 1)
        # Dark outline first so text stays readable against a blown-out background.
        cv2.putText(img, text, (_MARGIN, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (_MARGIN, y), cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


class LiveView:
    """A named cv2 window. ``show`` returns the pressed key (-1 if none).

    ``scale`` shrinks the 1280x800 frame for display only; it never touches the data used
    for geometry.
    """

    def __init__(self, window: str = "dume camera", scale: float = 0.6):
        self.window = window
        self.scale = float(scale)
        self._open = False

    def show(self, frame: np.ndarray, lines: list[str] | None = None) -> int:
        import cv2

        img = annotate(frame, lines)
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_AREA)
        cv2.imshow(self.window, img)
        self._open = True
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        if not self._open:
            return
        import cv2

        cv2.destroyWindow(self.window)
        cv2.waitKey(1)  # let the GUI event loop actually tear the window down
        self._open = False

    def __enter__(self) -> "LiveView":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
