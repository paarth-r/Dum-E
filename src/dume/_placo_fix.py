"""Self-healing fix for placo's broken macOS wheel (placo 0.9.23).

The macOS arm64 wheel's ``placo.so`` links ``liburdfdom_{sensor,model,world}.4.0.dylib``
but bundles the ``6.0.0`` versions, so import fails with a dlopen error. The urdfdom
parser ABI is stable across these versions (verified by exact FK/IK round-trip), so we
symlink the missing ``4.0`` sonames to the bundled ``6.0.0`` libs. Idempotent; only acts
on macOS when the mismatch is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LIBS = ("liburdfdom_sensor", "liburdfdom_model", "liburdfdom_world")
_MISSING_SONAME = "4.0"
_BUNDLED_SONAME = "6.0.0"


def ensure_placo_importable() -> None:
    """Create the urdfdom soname symlinks placo needs, if they're missing (macOS only)."""
    if sys.platform != "darwin":
        return
    try:
        import cmeel  # noqa: F401  -- placo bundles its deps under cmeel.prefix

        prefix = Path(cmeel.__file__).resolve()
    except Exception:
        # Fall back to locating cmeel.prefix next to the placo install.
        prefix = None

    lib_dir = _find_cmeel_lib_dir(prefix)
    if lib_dir is None:
        return
    for name in _LIBS:
        target = lib_dir / f"{name}.{_BUNDLED_SONAME}.dylib"
        link = lib_dir / f"{name}.{_MISSING_SONAME}.dylib"
        if target.exists() and not link.exists():
            try:
                link.symlink_to(target.name)
            except OSError:
                pass


def _find_cmeel_lib_dir(cmeel_prefix_module) -> Path | None:
    candidates = []
    if cmeel_prefix_module is not None:
        candidates.append(cmeel_prefix_module.parent / "cmeel.prefix" / "lib")
    for entry in sys.path:
        p = Path(entry)
        candidates.append(p / "cmeel.prefix" / "lib")
    for c in candidates:
        if c.is_dir() and any(c.glob("liburdfdom_*.dylib")):
            return c
    return None
