"""Focus metrics — turn "does this look sharp?" into a number you can maximise.

The UC-844's M12 lens focuses by rotating the barrel, which is a blind adjustment: small
turns change the image subtly and it is easy to walk past the optimum. These give the live
viewer something to display so focusing becomes hill-climbing on a scalar.

Both metrics are pure functions of a frame, so they are testable on synthetic images.
"""

from __future__ import annotations

import numpy as np

#: Laplacian variance below this reads as blurry; a well-focused textured scene runs far above.
SHARP_ENOUGH = 300.0
#: ORB yield below this makes sparse reconstruction hopeless — there is nothing to match.
FEATURES_ENOUGH = 300


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian — the standard no-reference focus measure.

    A blurred image has little high-frequency content, so its second derivative is small
    everywhere and its variance collapses. Scene-dependent (a blank wall scores low however
    sharp the lens), which is fine for *relative* comparison while turning a focus ring.
    """
    import cv2

    img = np.asarray(frame)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def feature_count(frame: np.ndarray, max_features: int = 2000) -> int:
    """Number of ORB keypoints found, capped at ``max_features``.

    This is the metric that actually predicts whether reconstruction can work: it counts the
    things the matcher will try to match. A frame scoring 2 here cannot produce a point cloud
    no matter how good the matching code is.
    """
    import cv2

    img = np.asarray(frame)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return len(cv2.ORB_create(nfeatures=max_features).detect(img, None))


def focus_lines(frame: np.ndarray) -> list[str]:
    """Overlay text reporting both metrics and whether they clear the usable thresholds."""
    sharp = sharpness(frame)
    feats = feature_count(frame)
    verdict = "OK" if sharp >= SHARP_ENOUGH and feats >= FEATURES_ENOUGH else "TOO SOFT"
    return [
        f"sharpness {sharp:7.1f}  (want >{SHARP_ENOUGH:.0f})",
        f"features  {feats:7d}  (want >{FEATURES_ENOUGH})",
        f"focus: {verdict}",
    ]
