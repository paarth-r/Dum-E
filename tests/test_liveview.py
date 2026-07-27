"""Live-view overlay rendering — pure, so no window and no camera is involved."""

import numpy as np

from dume import geometry as g
from dume.arducam import ARDUCAM_HEIGHT, ARDUCAM_WIDTH
from dume.liveview import annotate, pose_lines


def _grey(h=ARDUCAM_HEIGHT, w=ARDUCAM_WIDTH, value=120):
    return np.full((h, w), value, dtype=np.uint8)


def test_annotate_returns_bgr_same_size():
    out = annotate(_grey(), ["hello"])
    assert out.shape == (ARDUCAM_HEIGHT, ARDUCAM_WIDTH, 3)
    assert out.dtype == np.uint8


def test_annotate_does_not_mutate_the_input():
    """The grabber hands the same frame object to every consumer.

    Drawing in place would corrupt exactly the pixels the point-cloud path feature-matches on,
    and the corruption would be text — high-contrast, highly matchable, and completely fake.
    """
    frame = _grey()
    before = frame.copy()
    annotate(frame, ["overlay text"])
    assert np.array_equal(frame, before)


def test_annotate_actually_draws():
    plain = annotate(_grey(), [])
    marked = annotate(_grey(), ["some text"])
    assert not np.array_equal(plain, marked)


def test_annotate_accepts_bgr_input():
    bgr = np.full((64, 64, 3), 30, dtype=np.uint8)
    out = annotate(bgr, ["x"])
    assert out.shape == (64, 64, 3)


def test_annotate_with_no_lines_is_just_a_colour_conversion():
    out = annotate(_grey(8, 8, value=77), None)
    assert out.shape == (8, 8, 3)
    assert (out == 77).all()


def test_pose_lines_reports_millimetres_in_base_frame():
    pose = g.transform_from_pos_rpy([0.1, -0.2, 0.3], [0.0, 0.0, 0.0])
    lines = pose_lines(pose)
    assert any("100.0" in ln and "-200.0" in ln and "300.0" in ln for ln in lines)
    assert any("base frame" in ln for ln in lines)


def test_pose_lines_handles_missing_pose():
    assert pose_lines(None) == ["camera pose: unknown"]
