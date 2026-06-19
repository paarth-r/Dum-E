"""``dume`` command-line entry point — a thin consumer of :class:`dume.service.DumeArm`.

    dume find-port              discover the arm's serial port (wraps lerobot)
    dume calibrate              one-time SO-101 calibration (wraps lerobot)
    dume axes                   print live controller axes/buttons (verify mapping)
    dume save-pose [--name N]   hand-pose the arm, hit Enter to save its joints (default: start)
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
    state = {"i": 0}

    def on_tick(tel):
        state["i"] += 1
        if state["i"] % every:
            return
        x, y, z = tel.target_xyzrpy[:3]
        lock = "LOCK" if tel.orientation_lock else "free"
        traj = "traj" if tel.trajectory_active else "hold"
        sing = " SINGULAR" if tel.near_singular else ""
        print(
            f"[{tel.mode.value:8}] target xyz=({x:+.3f},{y:+.3f},{z:+.3f}) "
            f"grip={tel.gripper:5.1f} ori={lock} {traj} err={tel.tracking_pos_err_mm:5.1f}mm "
            f"margin={tel.min_joint_margin_deg:4.0f}° ({tel.margin_joint}){sing}   ",
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
            arm.run_teleop(xb.poll, on_tick=_status_printer())
        finally:
            xb.disconnect()
            print("\nStopped.")
    return 0


def cmd_sim(args) -> int:
    """Interactive PyBullet sim: drive the SO-101 in a 3D window with the Xbox controller.

    Pure kinematic mirror — the controller runs exactly as on hardware (over a SimArm), and the
    renderer reflects each commanded joint vector. ``--noise`` injects synthetic servo feedback
    noise so you can feel that the q_ref smoothing keeps motion clean. ``--scene`` spawns a demo
    object; with ``--camera`` the end-effector camera's live detections print each tick.
    """
    import numpy as _np
    import pybullet as pb

    from dume.camera import CameraIntrinsics, camera_pose_from_fk
    from dume.input_xbox import XboxController
    from dume.service import DumeArm
    from dume.sim_world import OrbitCameraNav, SceneObject, SimCamera, SimRenderer, SimScene

    arm = DumeArm(dry_run=True)
    arm.connect()
    if args.noise > 0:
        arm.arm.servo_noise_deg = float(args.noise)
        print(f"Injecting {args.noise} deg servo-feedback noise (the smoothing should absorb it).")

    has_scene = args.scene or args.camera
    renderer = SimRenderer(urdf_path=arm.config.urdf_path, gui=True, dynamic=has_scene)
    renderer.set_joints(arm.get_joints())
    # Lighten the GUI: no shadows, no side panels, no extra software renderer pass.
    for flag in (pb.COV_ENABLE_SHADOWS, pb.COV_ENABLE_GUI, pb.COV_ENABLE_TINY_RENDERER):
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

    state = {"i": 0, "holding": False}
    GRASP_DIST, CLOSE_T, OPEN_T = 0.07, 40.0, 60.0  # m / gripper units
    CAM_EVERY = 5  # render the camera every 5th control tick (~10 Hz vs the 50 Hz loop)

    def on_tick(tel):
        renderer.set_joints(tel.joints_sent)  # mirror the commanded config into the 3D view
        # OnShape-style camera. Reuse the keyboard controller's last events (getKeyboardEvents
        # consumes on read) for the Ctrl check; fall back to a fresh read under an Xbox pad.
        keys = getattr(source, "last_keys", None)
        if keys is None:
            keys = pb.getKeyboardEvents(physicsClientId=renderer.client)
        ctrl = CTRL_KEY is not None and bool(keys.get(CTRL_KEY, 0) & pb.KEY_IS_DOWN)
        nav.update(ctrl)
        if has_scene:
            # Magnet grasp: close the gripper near the box to pick it up; open to drop it.
            ee = arm.kin.fk(tel.joints_sent)[:3, 3]
            bpos, _ = pb.getBasePositionAndOrientation(box_id, physicsClientId=renderer.client)
            dist = float(_np.linalg.norm(ee - _np.asarray(bpos)))
            if not state["holding"] and tel.gripper < CLOSE_T and dist < GRASP_DIST:
                renderer.attach(box_id); state["holding"] = True
            elif state["holding"] and tel.gripper > OPEN_T:
                renderer.release(); state["holding"] = False
            renderer.step_physics()
        state["i"] += 1
        if cam is not None and state["i"] % CAM_EVERY == 0:  # throttled — vision is the slow part
            frame = cam.capture()  # re-render from the CURRENT EE pose
            if cv2 is not None:
                cv2.imshow("dume EE camera", frame.rgb[:, :, ::-1])  # RGB->BGR
                cv2.waitKey(1)
            dets = cam.detect()
            grab = " [HOLDING]" if state["holding"] else ""
            print(f"camera sees {len(dets)} object(s)" + (
                f" nearest~{_np.min(dets.depths):.3f}m" if len(dets) else "") + grab,
                end="\r", flush=True)

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
        "goto": cmd_goto,
        "sim": cmd_sim,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
