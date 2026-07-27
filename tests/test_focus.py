"""Focus metrics: blurring an image must move both numbers the right way."""

import numpy as np
import pytest

from dume.focus import FEATURES_ENOUGH, feature_count, focus_lines, sharpness


@pytest.fixture
def textured():
    """A high-frequency checker pattern — plenty for both metrics to bite on."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    img[::8, :] = 0
    img[:, ::8] = 255
    return img


def _blur(img, k=15):
    import cv2

    return cv2.GaussianBlur(img, (k, k), 0)


def test_blurring_lowers_sharpness(textured):
    assert sharpness(_blur(textured)) < sharpness(textured)


def test_blurring_lowers_feature_count(textured):
    """The metric that actually predicts whether reconstruction can work."""
    assert feature_count(_blur(textured)) < feature_count(textured)


def test_flat_image_scores_near_zero():
    assert sharpness(np.full((64, 64), 128, dtype=np.uint8)) == pytest.approx(0.0, abs=1e-6)


def test_metrics_accept_bgr(textured):
    import cv2

    bgr = cv2.cvtColor(textured, cv2.COLOR_GRAY2BGR)
    assert sharpness(bgr) == pytest.approx(sharpness(textured))
    assert feature_count(bgr) == feature_count(textured)


def test_feature_count_respects_the_cap(textured):
    assert feature_count(textured, max_features=50) <= 50


def test_focus_lines_flag_a_featureless_image():
    flat = np.full((240, 320), 100, dtype=np.uint8)
    assert any("TOO SOFT" in ln for ln in focus_lines(flat))


def test_focus_lines_pass_a_textured_image(textured):
    assert any("USABLE" in ln for ln in focus_lines(textured))


def test_verdict_ignores_sharpness(textured, monkeypatch):
    """The usable/not call must come from feature count alone.

    Regression: an absolute sharpness threshold called a genuinely focused frame "too soft"
    (a low-contrast room scored 21.7) while passing a badly out-of-focus close-up (79.5).
    Laplacian variance tracks scene contrast as much as focus, so it cannot gate anything.
    Driving it to zero here must not change the verdict.
    """
    import dume.focus as focus

    assert feature_count(textured) >= FEATURES_ENOUGH
    monkeypatch.setattr(focus, "sharpness", lambda _f: 0.0)
    assert any("USABLE" in ln for ln in focus_lines(textured))
