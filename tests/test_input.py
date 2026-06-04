import numpy as np

from dume.input_xbox import apply_deadzone, apply_expo, shape_axis


def test_deadzone_zeros_small():
    assert apply_deadzone(0.05, 0.1) == 0.0
    assert apply_deadzone(-0.05, 0.1) == 0.0


def test_deadzone_rescales_to_full_range():
    assert abs(apply_deadzone(1.0, 0.1) - 1.0) < 1e-9
    assert abs(apply_deadzone(-1.0, 0.1) + 1.0) < 1e-9
    # just past the deadzone -> near zero
    assert abs(apply_deadzone(0.1 + 1e-6, 0.1)) < 1e-3


def test_expo_endpoints_and_center():
    assert apply_expo(0.0, 0.6) == 0.0
    assert abs(apply_expo(1.0, 0.6) - 1.0) < 1e-9
    assert abs(apply_expo(-1.0, 0.6) + 1.0) < 1e-9


def test_expo_softens_midrange():
    # with expo, mid-stick output is below linear (finer control near center)
    assert apply_expo(0.5, 0.6) < 0.5


def test_shape_axis_monotonic():
    xs = np.linspace(-1, 1, 50)
    ys = [shape_axis(x, 0.08, 0.6) for x in xs]
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:]))
