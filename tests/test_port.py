"""Serial-port auto-resolution: ``dume run`` should find the arm without manual config edits.

The resolver is pure and dependency-injected (``exists`` + ``candidates``) so these run with no
hardware and no real ``/dev`` lookups.
"""

import pytest

from dume.arm import resolve_serial_port


def test_preferred_port_used_when_present():
    # The configured port exists -> use it verbatim, never glob.
    got = resolve_serial_port(
        "/dev/cu.usbmodemAAA",
        exists=lambda p: True,
        candidates=lambda: ["/dev/cu.usbmodemZZZ"],  # ignored
    )
    assert got == "/dev/cu.usbmodemAAA"


def test_falls_back_to_single_candidate():
    # Configured port is gone (e.g. suffix changed), exactly one usbmodem present -> use it.
    got = resolve_serial_port(
        "/dev/cu.usbmodemOLD",
        exists=lambda p: False,
        candidates=lambda: ["/dev/cu.usbmodemNEW"],
    )
    assert got == "/dev/cu.usbmodemNEW"


def test_no_candidates_raises_helpful_error():
    with pytest.raises(RuntimeError, match="find-port"):
        resolve_serial_port(
            "/dev/cu.usbmodemX",
            exists=lambda p: False,
            candidates=lambda: [],
        )


def test_ambiguous_candidates_raise():
    with pytest.raises(RuntimeError, match="multiple"):
        resolve_serial_port(
            "/dev/cu.usbmodemX",
            exists=lambda p: False,
            candidates=lambda: ["/dev/cu.usbmodemA", "/dev/cu.usbmodemB"],
        )
