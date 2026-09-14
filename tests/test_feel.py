"""``dume feel``: the live load/gravity/residual readout and the CSV sample logger."""

import csv

import numpy as np

from dume.cli import build_parser
from dume.forces import ForceReading, LoadLogger, format_feel

NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _reading():
    return ForceReading(
        q_deg=np.array([1.5, -20.0, 20.0, 0.0, 0.0, 50.0]),
        load=np.array([3, -412, 300, -50, 2, 8.0]),
        gravity_load=np.array([0, -395, 452, -116, 0, 2.0]),
        residual=np.array([3, -17, -152, 66, 2, 6.0]),
        tau_ext=np.array([0, -7, -142, 56, 0, 0.0]),
        wrench=np.array([0.1, -0.2, 1.5, 0, 0, 0.0]),
    )


def test_format_feel_has_one_row_per_joint_with_all_four_columns():
    text = format_feel(NAMES, _reading(), voltage=12.3, hz=48.0)
    rows = [line for line in text.splitlines() if any(line.startswith(n) for n in NAMES)]
    assert len(rows) == 6
    lift = next(r for r in rows if r.startswith("shoulder_lift"))
    for token in ("-20.0", "-412", "-395", "-17", "-7"):
        assert token in lift
    assert "12.3" in text and "48" in text


def test_format_feel_shows_the_wrench():
    text = format_feel(NAMES, _reading(), voltage=None, hz=50.0)
    assert "1.50" in text  # Fz


def test_load_logger_writes_one_csv_row_per_sample(tmp_path):
    path = tmp_path / "loads.csv"
    with LoadLogger(path, NAMES) as log:
        log.write(0.02, _reading(), voltage=12.3)
        log.write(0.04, _reading(), voltage=None)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["t"] == "0.02"
    assert rows[0]["q_shoulder_lift"] == "-20.0"
    assert rows[0]["load_shoulder_lift"] == "-412.0"
    assert rows[0]["grav_shoulder_lift"] == "-395.0"
    assert rows[0]["voltage"] == "12.3"
    assert rows[1]["voltage"] == ""


def test_parser_accepts_feel():
    args = build_parser().parse_args(["feel", "--dry-run", "--log", "x.csv", "--seconds", "2.5"])
    assert args.command == "feel" and args.dry_run and args.log == "x.csv"
    assert args.seconds == 2.5
    assert build_parser().parse_args(["feel"]).seconds is None


def test_feel_dry_run_exits_after_seconds_and_reads_zero_residual(tmp_path):
    """Dry-run feeds the sim modelled gravity as its load, so the residual is exactly zero and the
    logged gravity column is non-zero — a self-check that the pipeline is wired end to end."""
    import csv

    from dume.cli import main

    log = tmp_path / "feel.csv"
    rc = main(["feel", "--dry-run", "--seconds", "0.3", "--log", str(log), "--every", "100"])
    assert rc == 0
    rows = list(csv.DictReader(open(log)))
    assert 5 <= len(rows) <= 30
    r = rows[-1]
    assert float(r["grav_shoulder_lift"]) != 0.0
    assert float(r["load_shoulder_lift"]) == float(r["grav_shoulder_lift"])
