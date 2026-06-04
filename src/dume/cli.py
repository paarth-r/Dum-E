"""``dume`` command-line entry point — a thin consumer of :class:`dume.service.DumeArm`.

    dume find-port              discover the arm's serial port (wraps lerobot)
    dume calibrate              one-time SO-101 calibration (wraps lerobot)
    dume axes                   print live controller axes/buttons (verify mapping)
    dume run [--dry-run]        Xbox teleoperation (velocity jog + pose mode)
    dume goto X Y Z R P Y       move to an absolute pose (metres, radians)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dume.config import ControllerConfig


def _lerobot_script(name: str) -> str:
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.exists() else (shutil.which(name) or name)


def cmd_find_port(args) -> int:
    return subprocess.call([_lerobot_script("lerobot-find-port")])


def cmd_calibrate(args) -> int:
    cfg = ControllerConfig()
    return subprocess.call(
        [
            _lerobot_script("lerobot-calibrate"),
            "--robot.type=so101_follower",
            f"--robot.port={args.port or cfg.port}",
            f"--robot.id={args.id or cfg.robot_id}",
        ]
    )


def cmd_axes(args) -> int:
    from dume.input_xbox import XboxController

    xb = XboxController()
    xb.connect()
    print(f"Connected: {xb.name}. Move sticks/triggers/buttons. Ctrl-C to quit.")
    import pygame

    try:
        while True:
            pygame.event.pump()
            js = xb._js
            axes = [round(js.get_axis(i), 2) for i in range(js.get_numaxes())]
            btns = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
            print(f"axes={axes}  buttons_down={btns}        ", end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        xb.disconnect()
    return 0


def _status_printer(every: int = 5):
    state = {"i": 0}

    def on_tick(tel):
        state["i"] += 1
        if state["i"] % every:
            return
        x, y, z = tel.target_xyzrpy[:3]
        lock = "LOCK" if tel.orientation_lock else "free"
        traj = "traj" if tel.trajectory_active else "hold"
        print(
            f"[{tel.mode.value:8}] target xyz=({x:+.3f},{y:+.3f},{z:+.3f}) "
            f"grip={tel.gripper:5.1f} ori={lock} {traj} err={tel.tracking_pos_err_mm:5.1f}mm   ",
            end="\r",
            flush=True,
        )

    return on_tick


def cmd_run(args) -> int:
    from dume.input_xbox import XboxController
    from dume.service import DumeArm

    with DumeArm(dry_run=args.dry_run) as arm:
        if not args.dry_run and not arm.arm.is_calibrated():
            print("Arm is not calibrated. Run `dume calibrate` first.", file=sys.stderr)
            return 2
        print(f"Mode: {'DRY-RUN (no motion)' if args.dry_run else 'LIVE'}")
        if args.dry_run:
            print("No hardware will move. Use this to feel out IK + smoothing.")
        xb = XboxController(mapping=arm.config.xbox)
        xb.connect()
        print(f"Controller: {xb.name}")
        print(
            "Left stick: X/Y  |  Right stick: Z  |  D-pad up/down: pitch  |  D-pad L/R: roll\n"
            "LT/RT: gripper  |  A: velocity/freeze mode  |  Ctrl-C: quit"
        )
        try:
            arm.run_teleop(xb.poll, on_tick=_status_printer())
        finally:
            xb.disconnect()
            print("\nStopped.")
    return 0


def cmd_goto(args) -> int:
    from dume.service import DumeArm

    target = np.array(args.pose, dtype=float)
    with DumeArm(dry_run=args.dry_run) as arm:
        if not args.dry_run and not arm.arm.is_calibrated():
            print("Arm is not calibrated. Run `dume calibrate` first.", file=sys.stderr)
            return 2
        print("start xyzrpy:", np.round(arm.get_xyzrpy(), 4))
        arm.goto(target, wait=True)
        print("final xyzrpy:", np.round(arm.get_xyzrpy(), 4))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dume", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("find-port", help="discover the arm's serial port")

    pc = sub.add_parser("calibrate", help="one-time SO-101 calibration")
    pc.add_argument("--port")
    pc.add_argument("--id")

    sub.add_parser("axes", help="print live controller axes/buttons")

    pr = sub.add_parser("run", help="Xbox teleoperation")
    pr.add_argument("--dry-run", action="store_true", help="no motor motion (simulation)")

    pg = sub.add_parser("goto", help="move to an absolute pose")
    pg.add_argument("pose", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    pg.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "find-port": cmd_find_port,
        "calibrate": cmd_calibrate,
        "axes": cmd_axes,
        "run": cmd_run,
        "goto": cmd_goto,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
