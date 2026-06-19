from dume.padview import REVERSE, _stick_box, render_pad


def _dot_pos(lines: list[str]) -> tuple[int, int]:
    """Return (row, col) of ``●`` within the grid rows of a stick box."""
    grid = lines[2:-2]  # drop label, top border, bottom border, coord line
    for r, line in enumerate(grid):
        c = line.find("●")
        if c != -1:
            return r, c - 1  # subtract the left "│" border
    raise AssertionError("no dot found")


def test_centered_stick_dot_at_center():
    r, c = _dot_pos(_stick_box(0.0, 0.0, "L"))
    assert (r, c) == (3, 6)  # centre of a 13x7 grid


def test_full_deflection_corners():
    assert _dot_pos(_stick_box(1.0, 1.0, "L")) == (6, 12)  # bottom-right
    assert _dot_pos(_stick_box(-1.0, -1.0, "L")) == (0, 0)  # top-left


def test_button_on_is_highlighted_off_is_not():
    out = render_pad(0, 0, 0, 0, 0, 0, {"A": True, "D-Up": False})
    assert f"{REVERSE} A " in out  # pressed cell uses reverse video
    assert f"{REVERSE} D-Up" not in out  # released cell does not


def test_render_has_both_sticks_and_triggers():
    out = render_pad(0, 0, 0, 0, 0.5, 0.0, {"A": False})
    assert "Left stick" in out and "Right stick" in out
    assert "LT [" in out and "RT [" in out
