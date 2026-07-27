"""Focus metrics — turn "does this look sharp?" into a number you can maximise.

The UC-844's M12 lens focuses by rotating the barrel, which is a blind adjustment: small
turns change the image subtly and it is easy to walk past the optimum. These give the live
viewer something to display so focusing becomes hill-climbing on a scalar.

Both metrics are pure functions of a frame, so they are testable on synthetic images.
"""

from __future__ import annotations

import numpy as np

#: ORB yield below this makes sparse reconstruction hopeless — there is nothing to match.
#: This is the *only* absolute threshold here, because it measures the thing we actually need:
#: how many points the matcher will have to work with.
FEATURES_ENOUGH = 300


def sharpness(frame: np.ndarray) -> float:
    """Variance of the Laplacian — a *relative* focus signal only.

    A blurred image has little high-frequency content, so its second derivative is small
    everywhere and its variance collapses. But the value depends as much on the scene as on
    the lens: a well-focused wall of flat paint scores lower than a badly-focused close-up of
    a keyboard. Only compare it against itself, on one fixed scene, while turning the barrel.

    Never threshold it. A sharp frame of a low-contrast room measured 21.7 here while a
    genuinely out-of-focus desk shot measured 79.5 — an absolute cutoff gets that backwards.
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
    """Overlay text: feature count decides usable/not, sharpness is shown for hill-climbing."""
    feats = feature_count(frame)
    return [
        f"features  {feats:7d}  (need >{FEATURES_ENOUGH} to reconstruct)",
        f"sharpness {sharpness(frame):7.1f}  (relative only — maximise, don't compare)",
        f"focus: {'USABLE' if feats >= FEATURES_ENOUGH else 'TOO SOFT'}",
    ]
