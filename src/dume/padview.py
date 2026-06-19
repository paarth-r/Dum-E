"""ASCII rendering of the Xbox controller state for ``dume axes``.

Pure functions: state in (stick/trigger floats in their natural ranges, button flags), a
printable string out. No pygame and no cursor control — the caller owns the live redraw.
Stick ``y`` is passed as pygame reports it (+down), so it maps straight to screen rows.
"""

from __future__ import annotations

REVERSE = "\033[7m"  # highlight an "on" cell
DIM = "\033[2m"
RESET = "\033[0m"

_W, _H = 13, 7  # inner grid (cols, rows) of each stick box


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


def _stick_box(x: float, y: float, label: str) -> list[str]:
    """A bordered grid with a ``●`` at the stick position and a ``+`` at centre."""
    cx, cy = (_W - 1) / 2, (_H - 1) / 2
    col = _clamp_int(cx + x * cx, 0, _W - 1)
    row = _clamp_int(cy + y * cy, 0, _H - 1)
    ccol, crow = round(cx), round(cy)

    lines = [label, "┌" + "─" * _W + "┐"]
    for r in range(_H):
        cells = []
        for c in range(_W):
            if r == row and c == col:
                cells.append("●")
            elif r == crow and c == ccol:
                cells.append("+")
            else:
                cells.append(" ")
        lines.append("│" + "".join(cells) + "│")
    lines.append("└" + "─" * _W + "┘")
    lines.append(f" x={x:+.2f} y={y:+.2f}")
    return lines


def _beside(left: list[str], right: list[str], gap: str = "   ") -> list[str]:
    """Lay two equal-height-ish blocks side by side (no ANSI in either, so ``len`` aligns)."""
    h = max(len(left), len(right))
    left = left + [""] * (h - len(left))
    right = right + [""] * (h - len(right))
    w = max(len(s) for s in left)
    return [l.ljust(w) + gap + r for l, r in zip(left, right)]


def _bar(v: float, width: int = 12) -> str:
    n = _clamp_int(v * width, 0, width)
    return "█" * n + "░" * (width - n)


def _button_table(buttons: dict[str, bool]) -> list[str]:
    """A single-row table; each pressed cell is shown in reverse video, blank otherwise."""
    labels = list(buttons)
    widths = [len(l) + 2 for l in labels]  # one space of padding each side
    top = "┌" + "┬".join("─" * w for w in widths) + "┐"
    bot = "└" + "┴".join("─" * w for w in widths) + "┘"
    cells = []
    for label, w in zip(labels, widths):
        text = label.center(w)
        cells.append(f"{REVERSE}{text}{RESET}" if buttons[label] else text)
    return [top, "│" + "│".join(cells) + "│", bot]


def render_pad(
    lx: float,
    ly: float,
    rx: float,
    ry: float,
    lt: float,
    rt: float,
    buttons: dict[str, bool],
) -> str:
    """Full controller view: two stick boxes, the trigger bars, then the button table."""
    sticks = _beside(_stick_box(lx, ly, "Left stick"), _stick_box(rx, ry, "Right stick"))
    triggers = f"Triggers   LT [{_bar(lt)}] {lt:4.2f}    RT [{_bar(rt)}] {rt:4.2f}"
    return "\n".join(sticks + ["", triggers, "", *_button_table(buttons)])
