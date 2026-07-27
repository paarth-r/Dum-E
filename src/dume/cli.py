"""``dume`` command-line entry point — a thin consumer of :class:`dume.service.DumeArm`.

    dume find-port              discover the arm's serial port (wraps lerobot)
    dume calibrate              one-time SO-101 calibration (wraps lerobot)
    dume axes                   print live controller axes/buttons (verify mapping)
    dume save-pose [--name N]   hand-pose the arm, hit Enter to save its joints (default: start)
    dume run [--dry-run]        Xbox teleoperation (velocity jog + pose mode); --view adds camera
    dume view                   live camera view only (no arm) — use this to focus the lens
    dume scan [--poses A B]     walk saved setpoints, streaming the end-effector camera
    dume goto X Y Z R P Y       move to an absolute pose (metres, radians)
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


def _camera_tick(view, camera, extra_lines=None):
    """Tick hook that renders the live feed and reports a keypress as an abort.

    Deliberately does NOT read joints to compute a pose: ``controller.step`` already reads the
    bus every tick, and a second read here would double serial traffic in the control loop.
    Poses are attached at stops, where the arm is stationary and reads can be averaged.
    """

    def tick(*_):
        frame = camera.capture()
        key = view.show(frame.rgb, list(extra_lines or []))
        return False if key != 255 else None  # 255 == no key (waitKey -1 masked)

    return tick


def cmd_view(args) -> int:
    """Camera-only live viewer. No arm, no motion — stays open until you quit.

    Exists because focusing the M12 lens is a blind adjustment: it needs a window that simply
    persists while you turn the barrel. ``scan`` renders during motion and at stops, so it is
    the wrong tool for this — with ``--dry-run`` the sim arm teleports and there is nothing to
    render at all.
    """
    from dume.arducam import ArduCamSource
    from dume.focus import focus_lines
    from dume.liveview import LiveView

    with ArduCamSource(device=args.device) as camera, LiveView("dume view") as view:
        print(f"Camera on device {camera.device} "
              f"({camera.intrinsics.width}x{camera.intrinsics.height}, intrinsics UNCALIBRATED)")
        print("Turn the lens barrel to maximise the numbers. 'q' or Esc to quit.")
        best = 0.0
        while True:
            frame = camera.capture()
            lines = [] if args.no_metrics else focus_lines(frame.rgb)
            if lines:
                # Track the best seen so you can tell you've walked past the optimum.
                from dume.focus import sharpness

                best = max(best, sharpness(frame.rgb))
                lines.append(f"best seen {best:7.1f}")
            key = view.show(frame.rgb, lines)
            if key in (ord("q"), 27):
                break
    print("Closed.")
    return 0


def cmd_scan(args) -> int:
    """Walk the arm through saved setpoints, streaming the end-effector camera."""
    from dume.arducam import ArduCamSource
    from dume.liveview import LiveView, pose_lines
    from dume.poses import DEFAULT_JOINT_STORE, JointPoseStore
    from dume.scan import run_scan
    from dume.service import DumeArm

    store = JointPoseStore(args.file or DEFAULT_JOINT_STORE)
    names = args.poses or store.names()
    if not names:
        print(f"No saved setpoints in {store.path}. Capture one with `dume save-pose`.",
              file=sys.stderr)
        return 2
    missing = [n for n in names if not store.has(n)]
    if missing:
        print(f"Unknown setpoint(s) {missing}. Known: {store.names()}", file=sys.stderr)
        return 2

    out_dir = None
    if args.save:
        out_dir = Path(args.save).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

    with DumeArm(dry_run=args.dry_run) as arm:
        if not args.dry_run and not arm.arm.is_calibrated():
            print("Arm is not calibrated. Run `dume calibrate` first.", file=sys.stderr)
            return 2
        # Scripted motion moves with nobody's hand on the arm; default to a gentler cap.
        arm.config.joint_slew_deg = args.slew
        print(f"Visiting {len(names)} setpoint(s): {', '.join(names)}")
        print(f"Slew {args.slew} deg/tick, dwell {args.dwell}s. Any key in the view aborts.")

        with ArduCamSource() as camera, LiveView("dume scan") as view:
            print(f"Camera on device {camera.device} ({camera.intrinsics.width}x"
                  f"{camera.intrinsics.height}, intrinsics UNCALIBRATED)")
            tick = _camera_tick(view, camera, ["scanning — any key aborts"])
            stops = 0
            for stop in run_scan(arm, camera, names, store, dwell=args.dwell,
                                 samples=args.samples, on_tick=tick):
                stops += 1
                xyz = stop.pose[:3, 3] * 1000
                print(f"  [{stop.name}] camera at ({xyz[0]:7.1f},{xyz[1]:7.1f},{xyz[2]:7.1f}) mm")
                view.show(stop.frame.rgb, [f"stop: {stop.name}"] + pose_lines(stop.pose))
                if out_dir is not None:
                    _write_stop(out_dir, stop)
            print(f"Captured {stops} of {len(names)} stop(s)"
                  + (f" to {out_dir}" if out_dir else "")
                  + ("" if stops == len(names) else " — aborted early."))
    return 0


def _write_stop(out_dir: Path, stop) -> None:
    """Persist one stop: the frame, plus the measured pose and joints beside it."""
    import json

    import cv2

    cv2.imwrite(str(out_dir / f"{stop.name}.png"), stop.frame.rgb)
    (out_dir / f"{stop.name}.json").write_text(
        json.dumps(
            {
                "name": stop.name,
                "joints_measured_deg": stop.joints.tolist(),
                "camera_pose_in_base": stop.pose.tolist(),
                "note": "pose is FK of MEASURED joints; intrinsics are uncalibrated",
            },
            indent=2,
        )
    )


def cmd_run(args) -> int:
    from dume.input_xbox import XboxController
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
        print(
            "Left stick: X/Y  |  Right stick: Z (L3 up / R3 down)  |  D-pad: pitch (U/D), roll (L/R)\n"
            "RT: gripper (squeeze)  |  X: gripper mode (squeeze/rate)  |  B: velocity/freeze  |  Ctrl-C: quit"
        )
        try:
            with contextlib.ExitStack() as stack:
                on_tick = _status_printer()
                if args.view:
                    from dume.arducam import ArduCamSource
                    from dume.liveview import LiveView

                    camera = stack.enter_context(ArduCamSource())
                    view = stack.enter_context(LiveView("dume camera"))
                    print(f"Camera on device {camera.device} (intrinsics UNCALIBRATED)")
                    status = on_tick

                    def on_tick(tel):  # noqa: F811 — compose, don't replace the status line
                        status(tel)
                        view.show(camera.capture().rgb, [])

                arm.run_teleop(xb.poll, on_tick=on_tick)
        finally:
            xb.disconnect()
            print("\nStopped.")
    return 0


def cmd_sim(args) -> int:
    """Interactive PyBullet sim: drive the SO-101 in a 3D window with the Xbox controller.

    One faithful path — the control stack drives a *physics-backed* arm (``PyBulletArm``: motors +
    real contacts), so the sim shows how the code behaves IRL. ``--scene`` spawns a graspable box
    (jaws close, friction holds it, opening drops it — fully emergent, no magnet); with ``--camera``
    the end-effector camera's live detections print each tick.
    """
    import pybullet as pb

    from dume.camera import CameraIntrinsics, camera_pose_from_fk
    from dume.input_xbox import XboxController
    from dume.poses import HOME_JOINTS
    from dume.service import DumeArm
    from dume.sim_world import (
        OrbitCameraNav, PyBulletArm, SceneObject, SimCamera, SimRenderer, SimScene,
    )

    cfg = ControllerConfig()
    has_scene = args.scene or args.camera
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

    cam = None
    box_id = None
    if has_scene:
        scene = SimScene()
        # Dynamic (mass > 0) so it rests on the ground plane and can be grabbed.
        scene.add(SceneObject("target", "box", half_extents=[0.025, 0.025, 0.025],
                              position=[0.28, 0.0, 0.05], rgba=[0.1, 0.6, 1.0, 1.0], mass=0.05))
        renderer.load_scene(scene)
        box_id = renderer.scene_bodies["target"]
        if args.camera:
            # Barebones vision: small frame, GPU renderer, throttled — see CAM_EVERY below.
            intr = CameraIntrinsics.from_fov(160, 120, fov_y_deg=60.0)
            cam = SimCamera(renderer, intr, lambda: camera_pose_from_fk(arm.kin, arm.get_joints()),
                            hardware=True)

    # Optional live camera feed window (separate from the 3D view).
    cv2 = None
    if cam is not None:
        try:
            import cv2 as _cv2
            cv2 = _cv2
            cv2.namedWindow("dume EE camera", cv2.WINDOW_NORMAL)
        except Exception:
            cv2 = None

    state = {"i": 0}
    CAM_EVERY = 5  # render the camera every 5th control tick (~10 Hz vs the 50 Hz loop)

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
        state["i"] += 1
        if box_id is not None:  # grip readout: real contact points + normal force, and box height
            n, f = renderer.contact_points(renderer.arm_body, box_id)
            bz = pb.getBasePositionAndOrientation(box_id, physicsClientId=renderer.client)[0][2]
            print(f"grip: contacts={n} force={f:5.1f}N  box_z={bz:+.3f}m   ", end="\r", flush=True)
        if cam is not None and state["i"] % CAM_EVERY == 0:  # throttled — vision is the slow part
            frame = cam.capture()  # re-render from the CURRENT EE pose
            if cv2 is not None:
                cv2.imshow("dume EE camera", frame.rgb[:, :, ::-1])  # RGB->BGR
                cv2.waitKey(1)

    from dume.input_keyboard import KeyboardController

    _KB_HELP = (
        "Keyboard (focus the PyBullet window): WASD = X/Y, R/F = Z up/down, "
        "arrows = wrist pitch/roll, O/C = gripper open/close, [ / ] = snap closed/open, "
        "M = velocity/freeze, Ctrl-C in this terminal = quit"
    )
    _PAD_HELP = (
        "Left stick: X/Y | Right stick: Z | D-pad: wrist pitch/roll | LT/RT: gripper | "
        "A/Y: close/open | B: velocity/freeze | Ctrl-C: quit"
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
        if cv2 is not None:
            cv2.destroyAllWindows()
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
    pr.add_argument("--view", action="store_true", help="show the end-effector camera feed")

    pv = sub.add_parser("view", help="live end-effector camera view (no arm) — for focusing")
    pv.add_argument("--device", type=int, help="explicit camera index (default: probe for 1280x800)")
    pv.add_argument("--no-metrics", action="store_true", help="hide the focus readout")

    psc = sub.add_parser("scan", help="walk saved setpoints, streaming the end-effector camera")
    psc.add_argument("--poses", nargs="+", help="setpoint names to visit (default: all saved)")
    psc.add_argument("--file", help="JSON store path (default: ~/.dume/joint_poses.json)")
    psc.add_argument("--dwell", type=float, default=0.6,
                     help="seconds held at each stop before measuring (default: 0.6)")
    psc.add_argument("--samples", type=int, default=5,
                     help="measured joint reads averaged at each stop (default: 5)")
    psc.add_argument("--slew", type=float, default=3.0,
                     help="deg/tick joint cap; lower than teleop's 6.0 since nobody's hand is "
                          "on the arm (default: 3.0)")
    psc.add_argument("--save", help="directory to write each stop's frame + measured pose")
    psc.add_argument("--dry-run", action="store_true", help="no motor motion (simulation)")

    pg = sub.add_parser("goto", help="move to an absolute pose")
    pg.add_argument("pose", nargs=6, type=float, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"))
    pg.add_argument("--dry-run", action="store_true")

    psim = sub.add_parser("sim", help="interactive PyBullet sim, Xbox-driven")
    psim.add_argument("--noise", type=float, default=0.0,
                      help="inject N deg servo-feedback noise to feel the smoothing (default 0)")
    psim.add_argument("--scene", action="store_true", help="spawn a demo target object")
    psim.add_argument("--camera", action="store_true",
                      help="attach the end-effector camera and print live detections (implies --scene)")
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
        "view": cmd_view,
        "scan": cmd_scan,
        "goto": cmd_goto,
        "sim": cmd_sim,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
