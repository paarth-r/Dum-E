"""``dume`` command-line entry point — a thin consumer of :class:`dume.service.DumeArm`.

    dume find-port              discover the arm's serial port (wraps lerobot)
    dume calibrate              one-time SO-101 calibration (wraps lerobot)
    dume axes                   print live controller axes/buttons (verify mapping)
    dume save-pose [--name N]   hand-pose the arm, hit Enter to save its joints (default: start)
    dume run [--dry-run]        Xbox teleoperation (velocity jog + pose mode)
    dume record                 record a hand-guided macro onto a digit key (played in `run`)
    dume goto X Y Z R P Y       move to an absolute pose (metres, radians)
    dume feel [--log F.csv]     live per-joint load / modelled gravity / residual readout
"""

from __future__ import annotations

import argparse
import contextlib
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
    from dume.config import XboxMap
    from dume.input_xbox import XboxController
    from dume.padview import render_pad

    xb = XboxController()
    xb.connect()
    m = XboxMap()
    print(f"Connected: {xb.name}. Move sticks/triggers/buttons. Ctrl-C to quit.\n")

    import pygame

    def trig(idx: int) -> float:
        return float(np.clip((xb._axis(idx) + 1.0) / 2.0, 0.0, 1.0))

    sys.stdout.write("\033[?25l")  # hide cursor while we repaint in place
    first = True
    try:
        while True:
            pygame.event.pump()
            buttons = {
                "X:gmode": xb._button(m.btn_x),
                "B:mode": xb._button(m.btn_b),
                "L3:Zup": xb._button(m.btn_l3),
                "R3:Zdn": xb._button(m.btn_r3),
                "D-Up": xb._button(m.btn_dpad_up),
                "D-Down": xb._button(m.btn_dpad_down),
                "D-Left": xb._button(m.btn_dpad_left),
                "D-Right": xb._button(m.btn_dpad_right),
            }
            frame = render_pad(
                xb._axis(m.axis_left_x),
                xb._axis(m.axis_left_y),
                xb._axis(m.axis_right_x),
                xb._axis(m.axis_right_y),
                trig(m.axis_lt),
                trig(m.axis_rt),
                buttons,
            )
            lines = frame.split("\n")
            if not first:
                sys.stdout.write(f"\033[{len(lines)}A")  # back up to the top of the block
            sys.stdout.write("\r" + "\n".join(line + "\033[K" for line in lines) + "\n")
            sys.stdout.flush()
            first = False
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\033[?25h\n")  # restore cursor
        sys.stdout.flush()
        xb.disconnect()
    return 0


def _status_printer(every: int = 5):
    state = {"i": 0, "t": None, "hz": 0.0}

    def on_tick(tel):
        # Effective loop rate (EMA-smoothed) — if this sits well below 50 Hz, stepped motion is a
        # timing/serial-latency problem, not the control pipeline.
        now = time.perf_counter()
        if state["t"] is not None:
            inst = 1.0 / max(now - state["t"], 1e-6)
            state["hz"] = inst if state["hz"] == 0 else 0.9 * state["hz"] + 0.1 * inst
        state["t"] = now
        state["i"] += 1
        if state["i"] % every:
            return
        x, y, z = tel.target_xyzrpy[:3]
        lock = "LOCK" if tel.orientation_lock else "free"
        traj = "traj" if tel.trajectory_active else "hold"
        sing = " SINGULAR" if tel.near_singular else ""
        print(
            f"[{tel.mode.value:8}] xyz=({x:+.3f},{y:+.3f},{z:+.3f}) "
            f"grip={tel.gripper:5.1f} {traj} err={tel.tracking_pos_err_mm:4.0f}mm "
            f"margin={tel.min_joint_margin_deg:4.0f}°({tel.margin_joint}) {state['hz']:4.0f}Hz{sing}   ",
            end="\r",
            flush=True,
        )

    return on_tick


def cmd_save_pose(args) -> int:
    from dume.arm import MOTOR_ORDER, SO101Arm
    from dume.poses import DEFAULT_JOINT_STORE, JointPoseStore

    cfg = ControllerConfig()
    arm = SO101Arm(args.port or cfg.port, args.id or cfg.robot_id)
    arm.connect()
    try:
        if not arm.is_calibrated():
            print("Arm is not calibrated. Run `dume calibrate` first.", file=sys.stderr)
            return 2
        if args.relax:
            arm.relax()
            print("Torque disabled — move the arm by hand to the pose you want.")
        else:
            print("Torque held — pose the arm however you like (e.g. via teleop).")
        try:
            input(f"Press Enter to save '{args.name}' (Ctrl-C to cancel)... ")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled — nothing saved.")
            return 1
        joints = arm.read_joints()
    finally:
        arm.disconnect()

    store = JointPoseStore(args.file or DEFAULT_JOINT_STORE)
    store.set(args.name, joints)
    pretty = ", ".join(f"{m}={joints[i]:.1f}" for i, m in enumerate(MOTOR_ORDER))
    print(f"Saved '{args.name}' to {store.path}\n  {pretty}")
    return 0


def cmd_record(args) -> int:
    """Record a hand-guided macro: name it, bind it to a digit, count down, then the arm
    goes limp and samples its measured joints every tick until space stops the take."""
    from dume.macros import Macro, MacroStore, RawKeys, trim_idle
    from dume.service import DumeArm

    store = MacroStore(args.file) if args.file else MacroStore()
    name = input("Macro name: ").strip()
    if not name:
        print("A macro needs a name.", file=sys.stderr)
        return 2
    while True:
        key = input("Keybind (0-9): ").strip()
        if len(key) == 1 and key.isdigit():
            break
        print("Pick a single digit 0-9.")
    existing = store.get(key)
    if existing and input(f"Key {key} holds '{existing.name}'. Overwrite? [y/N] ").lower() != "y":
        return 0

    with DumeArm() as arm:
        if not arm.arm.is_calibrated():
            print("Arm is not calibrated. Run `dume calibrate` first.", file=sys.stderr)
            return 2
        for n in (3, 2, 1):
            print(f"Recording in {n}...")
            time.sleep(1.0)
        arm.arm.relax()
        print("RECORDING — move the arm by hand. Space stops the take.")
        dt = arm.config.dt
        frames = []
        with RawKeys() as keys:
            while keys.get() != " ":
                t0 = time.perf_counter()
                frames.append(arm.arm.read_joints().copy())
                sleep = dt - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        arm.arm.engage()
        arm.arm.write_joints(frames[-1])  # hold where the hand left it
        arm.controller._sync_to_joints(frames[-1].copy())
        kept = trim_idle(np.array(frames))
        store.set(Macro(name=name, key=key, dt=dt, frames=kept))
        cut = (len(frames) - len(kept)) * dt
        print(f"Saved '{name}' to key {key}: {len(kept)} frames, {len(kept) * dt:.1f}s"
              + (f" ({cut:.1f}s of idle head/tail trimmed)." if cut > 0.05 else "."))
    return 0


def cmd_run(args) -> int:
    from dume.input_xbox import XboxController
    from dume.macros import MacroStore, RawKeys, play_macro
    from dume.poses import DEFAULT_JOINT_STORE, JointPoseStore
    from dume.service import DumeArm

    with DumeArm(dry_run=args.dry_run) as arm:
        if not args.dry_run and not arm.arm.is_calibrated():
            print("Arm is not calibrated. Run `dume calibrate` first.", file=sys.stderr)
            return 2
        print(f"Mode: {'DRY-RUN (no motion)' if args.dry_run else 'LIVE'}")
        if args.dry_run:
            print("No hardware will move. Use this to feel out IK + smoothing.")
        if not args.no_start_pose:
            store = JointPoseStore(args.start_file or DEFAULT_JOINT_STORE)
            if store.has(args.start_pose):
                print(f"Moving to saved start pose '{args.start_pose}'...")
                arm.goto_joints(store.get(args.start_pose))
                arm.controller.home_joints = store.get(args.start_pose)  # 'home' returns here
            else:
                print(f"No saved '{args.start_pose}' pose — starting from current position.")
        xb = XboxController(mapping=arm.config.xbox)
        xb.connect()
        print(f"Controller: {xb.name}")
        macros = MacroStore(args.macro_file) if args.macro_file else MacroStore()
        print(
            "Left stick: X/Y  |  Right stick: Z (L3 up / R3 down)  |  D-pad: pitch (U/D), roll (L/R)\n"
            "RT: gripper (squeeze)  |  X: gripper mode (squeeze/rate)  |  B: velocity/freeze  |  Ctrl-C: quit"
        )
        if macros.items():
            bound = "  ".join(f"[{k}] {m.name}" for k, m in macros.items())
            print(f"Macros: {bound}  (press the digit to play; space aborts mid-macro)")
        print("Space: toggle a limp gripper (torque off — position the jaw by hand; space or a macro re-grips)")
        try:
            with contextlib.ExitStack() as stack:
                on_tick = _status_printer()
                keys = stack.enter_context(RawKeys())
                gripper_limp = False

                def engage_gripper_if_limp():
                    nonlocal gripper_limp
                    if gripper_limp:
                        arm.arm.engage_gripper()
                        gripper_limp = False

                def poll():
                    nonlocal gripper_limp
                    k = keys.get()
                    if k == " ":
                        if gripper_limp:
                            engage_gripper_if_limp()
                            print("\nGripper re-engaged (following the trigger).")
                        else:
                            arm.arm.relax_gripper()
                            gripper_limp = True
                            print("\nGripper limp — space re-grips.")
                    elif k is not None and k.isdigit():
                        macro = macros.get(k)
                        if macro is not None:
                            engage_gripper_if_limp()
                            print(f"\nMacro '{macro.name}' [{k}] — space aborts.")
                            done = play_macro(arm, macro, abort=lambda: keys.get() == " ")
                            print("Macro done." if done else "Macro aborted — holding here.")
                    return xb.poll()
                arm.run_teleop(poll, on_tick=on_tick)
        finally:
            xb.disconnect()
            print("\nStopped.")
    return 0


def cmd_sim(args) -> int:
    """Interactive PyBullet sim: drive the SO-101 in a 3D window with the Xbox controller.

    One faithful path — the control stack drives a *physics-backed* arm (``PyBulletArm``: motors +
    real contacts), so the sim shows how the code behaves IRL. ``--scene`` spawns a graspable box
    (jaws close, friction holds it, opening drops it — fully emergent, no magnet).
    """
    import pybullet as pb

    from dume.input_xbox import XboxController
    from dume.poses import HOME_JOINTS
    from dume.service import DumeArm
    from dume.sim_world import (
        OrbitCameraNav, PyBulletArm, SceneObject, SimRenderer, SimScene,
    )

    cfg = ControllerConfig()
    has_scene = args.scene
    # Physics always on (gravity + ground); a scene just adds a box. The control stack drives the
    # PyBulletArm directly, so the GUI shows the physical arm — no separate commanded-joint mirror.
    renderer = SimRenderer(urdf_path=cfg.urdf_path, gui=True, dynamic=True)
    renderer.set_joints(HOME_JOINTS)  # start posed at HOME; connect() then holds it with motors
    arm = DumeArm(config=cfg, arm=PyBulletArm(renderer, dt=cfg.dt))
    arm.connect()
    if args.noise > 0:
        print("--noise is ignored in the physics sim (feedback is real); use `run --dry-run` for it.")
    # Lighten the GUI: no shadows, no side panels, no extra software renderer pass. Also kill
    # PyBullet's built-in keyboard shortcuts: they bind w=wireframe, a=AABB, g=grid, which collide
    # with our WASD/G control keys (pressing W to drive forward would flip the view to wireframe).
    # getKeyboardEvents still delivers the keys to our KeyboardController; only PyBullet's own
    # handling is suppressed.
    for flag in (
        pb.COV_ENABLE_SHADOWS,
        pb.COV_ENABLE_GUI,
        pb.COV_ENABLE_TINY_RENDERER,
        pb.COV_ENABLE_KEYBOARD_SHORTCUTS,
    ):
        pb.configureDebugVisualizer(flag, 0, physicsClientId=renderer.client)
    nav = OrbitCameraNav(renderer)  # OnShape-style: left-drag orbit, Ctrl+left-drag pan, wheel zoom
    CTRL_KEY = getattr(pb, "B3G_CONTROL", None)

    box_id = None
    if has_scene:
        scene = SimScene()
        # Dynamic (mass > 0) so it rests on the ground plane and can be grabbed.
        scene.add(SceneObject("target", "box", half_extents=[0.025, 0.025, 0.025],
                              position=[0.28, 0.0, 0.05], rgba=[0.1, 0.6, 1.0, 1.0], mass=0.05))
        renderer.load_scene(scene)
        box_id = renderer.scene_bodies["target"]

    def on_tick(tel):
        # The physics arm moves itself — motors are stepped inside PyBulletArm.write_joints, so
        # there's no commanded-joint mirror to apply here. OnShape-style camera: reuse the keyboard
        # controller's last events (getKeyboardEvents consumes on read) for the Ctrl check; fall
        # back to a fresh read under an Xbox pad.
        keys = getattr(source, "last_keys", None)
        if keys is None:
            keys = pb.getKeyboardEvents(physicsClientId=renderer.client)
        ctrl = CTRL_KEY is not None and bool(keys.get(CTRL_KEY, 0) & pb.KEY_IS_DOWN)
        nav.update(ctrl)
        if box_id is not None:  # grip readout: real contact points + normal force, and box height
            n, f = renderer.contact_points(renderer.arm_body, box_id)
            bz = pb.getBasePositionAndOrientation(box_id, physicsClientId=renderer.client)[0][2]
            print(f"grip: contacts={n} force={f:5.1f}N  box_z={bz:+.3f}m   ", end="\r", flush=True)

    from dume.input_keyboard import KeyboardController

    _KB_HELP = (
        "Keyboard (focus the PyBullet window): WASD = X/Y, R/F = Z up/down, "
        "arrows = wrist pitch/roll, O/C = gripper open/close, [ / ] = snap closed/open, "
        "M = velocity/freeze, Ctrl-C in this terminal = quit"
    )
    _PAD_HELP = (
        "Left stick: X/Y | Right stick: Z | D-pad: wrist pitch/roll | LT/RT: gripper | "
        "X: gripper mode | B: velocity/freeze | RB: turbo | Ctrl-C: quit"
    )
    _MOUSE_HELP = "Camera: drag = orbit, Ctrl+drag = pan, scroll = zoom"

    if args.keyboard:
        source = KeyboardController(renderer.client)
        source.connect()
        print("Input: keyboard.\n" + _KB_HELP)
    else:
        try:
            source = XboxController(mapping=arm.config.xbox)
            source.connect()
            print(f"Controller: {source.name}\n" + _PAD_HELP)
        except Exception as exc:  # no pad attached — fall back to keyboard
            source = KeyboardController(renderer.client)
            source.connect()
            print(f"No pad ({exc}); using keyboard.\n" + _KB_HELP)

    print(_MOUSE_HELP)
    try:
        arm.run_teleop(source.poll, on_tick=on_tick)
    except KeyboardInterrupt:
        pass
    finally:
        source.disconnect()
        renderer.disconnect()
        arm.disconnect()
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


def cmd_feel(args) -> int:
    """Live force readout: raw servo load, modelled gravity, and their residual, per joint.

    The arm holds wherever it is (torque on — load is meaningless with torque off), so pushing
    on a link shows up as residual on the joints upstream of it. ``--log`` records every sample
    for the calibration sweep that fits ``scale`` (see the force-sensing design doc). The status
    line's Hz is the achieved rate *with* the extra load read on the bus.
    """
    from dume.arm import SimArm
    from dume.forces import ForceEstimator, LoadLogger, format_feel
    from dume.kinematics import Kinematics
    from dume.poses import HOME_JOINTS
    from dume.service import DumeArm

    sim = None
    if args.dry_run:
        # A perfect servo holding a perfect model: load == modelled gravity, residual == 0.
        # Non-zero residual in dry-run therefore means the pipeline itself is broken.
        model = Kinematics()
        sim = SimArm(initial_joints=HOME_JOINTS, load_source=lambda q: model.gravity_torques(q))
    with DumeArm(dry_run=args.dry_run, arm=sim) as arm:
        if not args.dry_run and not arm.arm.is_calibrated():
            print("Arm is not calibrated. Run `dume calibrate` first.", file=sys.stderr)
            return 2
        est = ForceEstimator(arm.kin, scale=args.scale, friction_floor=args.floor, alpha=args.alpha)
        read_voltage = getattr(arm.arm, "read_voltage", None)
        names = arm.kin.joint_names
        n_lines = len(names) + 2
        print(f"Mode: {'DRY-RUN (synthetic load = modelled gravity; residual must read 0)' if args.dry_run else 'LIVE'}")
        print("Holding position. Push on the arm to see residual load. Ctrl-C to quit.")
        if args.log:
            print(f"Logging samples to {args.log}")
        period = 1.0 / args.hz
        with contextlib.ExitStack() as stack:
            log = stack.enter_context(LoadLogger(args.log, names)) if args.log else None
            voltage = None
            hz = 0.0
            t_start = t_prev = time.perf_counter()
            tick = 0
            print("\n" * (n_lines - 1), end="")
            try:
                while args.seconds is None or time.perf_counter() - t_start < args.seconds:
                    t0 = time.perf_counter()
                    q = arm.arm.read_joints()
                    reading = est.update(q, arm.arm.read_loads())
                    if read_voltage is not None and tick % 25 == 0:
                        voltage = read_voltage()
                    if tick > 0:  # first interval is just setup time, not a loop period
                        inst = 1.0 / max(t0 - t_prev, 1e-6)
                        hz = inst if tick == 1 else 0.9 * hz + 0.1 * inst
                    t_prev = t0
                    if log is not None:
                        log.write(round(t0 - t_start, 4), reading, voltage=voltage)
                    if tick % args.every == 0:
                        sys.stdout.write(f"\033[{n_lines}F\033[J")
                        print(format_feel(names, reading, voltage=voltage, hz=hz), flush=True)
                    tick += 1
                    sleep = period - (time.perf_counter() - t0)
                    if sleep > 0:
                        time.sleep(sleep)
            except KeyboardInterrupt:
                pass
    print("Stopped.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dume", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("find-port", help="discover the arm's serial port")

    pc = sub.add_parser("calibrate", help="one-time SO-101 calibration")
    pc.add_argument("--port")
    pc.add_argument("--id")

    sub.add_parser("axes", help="print live controller axes/buttons")

    ps = sub.add_parser("save-pose", help="hand-pose the arm and save its joints to JSON")
    ps.add_argument("--name", default="start", help="name to save under (default: start)")
    ps.add_argument("--file", help="JSON store path (default: ~/.dume/joint_poses.json)")
    ps.add_argument("--port")
    ps.add_argument("--id")
    ps.add_argument(
        "--no-relax",
        dest="relax",
        action="store_false",
        help="keep motor torque on (don't free the arm for hand-posing)",
    )

    pr = sub.add_parser("run", help="Xbox teleoperation")
    pr.add_argument("--dry-run", action="store_true", help="no motor motion (simulation)")
    pr.add_argument("--start-pose", default="start", help="saved joint pose to start at (default: start)")
    pr.add_argument("--start-file", help="JSON store path (default: ~/.dume/joint_poses.json)")
    pr.add_argument("--no-start-pose", action="store_true", help="don't move to a start pose on launch")
    pr.add_argument("--macro-file", help="macro store path (default: ~/.dume/macros.json)")

    prc = sub.add_parser("record", help="record a hand-guided macro onto a digit key (0-9)")
    prc.add_argument("--file", help="macro store path (default: ~/.dume/macros.json)")

    pg = sub.add_parser("goto", help="move to an absolute pose")
    pg.add_argument("pose", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    pg.add_argument("--dry-run", action="store_true")

    pf = sub.add_parser("feel", help="live per-joint load / gravity / residual readout")
    pf.add_argument("--dry-run", action="store_true", help="simulated arm (loads read as zero)")
    pf.add_argument("--log", help="CSV path to record every sample (for the calibration sweep)")
    pf.add_argument("--hz", type=float, default=50.0, help="sample rate to attempt (default: 50)")
    pf.add_argument("--seconds", type=float, help="stop after this long (default: run until Ctrl-C)")
    pf.add_argument("--every", type=int, default=5, help="redraw the table every N samples")
    pf.add_argument("--scale", type=float, default=1.0,
                    help="N*m per raw load unit (default 1.0 = uncalibrated, raw units)")
    pf.add_argument("--floor", type=float, default=0.0, help="friction deadband, raw units")
    pf.add_argument("--alpha", type=float, default=0.3, help="residual low-pass (1 = none)")

    psim = sub.add_parser("sim", help="interactive PyBullet sim, Xbox-driven")
    psim.add_argument("--noise", type=float, default=0.0,
                      help="inject N deg servo-feedback noise to feel the smoothing (default 0)")
    psim.add_argument("--scene", action="store_true", help="spawn a demo target object")
    psim.add_argument("--keyboard", action="store_true",
                      help="drive with the keyboard instead of an Xbox pad (also the no-pad fallback)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "find-port": cmd_find_port,
        "calibrate": cmd_calibrate,
        "axes": cmd_axes,
        "save-pose": cmd_save_pose,
        "run": cmd_run,
        "record": cmd_record,
        "goto": cmd_goto,
        "sim": cmd_sim,
        "feel": cmd_feel,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
