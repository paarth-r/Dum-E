"""End-to-end integration (headless): controller -> sim render -> EE camera -> recorder.

This is the pipeline `dume sim` drives, minus the GUI and the Xbox pad. It proves the pieces
built across the perception/learning scaffolding compose: the smoothed controller commands a
SimArm, the renderer mirrors the commanded config, the end-effector camera renders depth from
the FK-derived pose, and an episode of (observation, action) steps is recorded.
"""

import numpy as np
import pytest

from dume.camera import CameraIntrinsics, camera_pose_from_fk
from dume.dataset import Action, EpisodeRecorder, observe
from dume.input_xbox import Command
from dume.service import DumeArm
from dume.sim_world import SceneObject, SimCamera, SimRenderer, SimScene


@pytest.fixture
def rig():
    arm = DumeArm(dry_run=True)
    arm.connect()
    renderer = SimRenderer(urdf_path=arm.config.urdf_path, gui=False)
    scene = SimScene()
    scene.add(SceneObject("target", "box", half_extents=[0.03, 0.03, 0.03], position=[0.30, 0.0, 0.10]))
    renderer.load_scene(scene)
    cam = SimCamera(
        renderer,
        CameraIntrinsics.from_fov(160, 120, 60.0),
        lambda: camera_pose_from_fk(arm.kin, arm.get_joints()),
    )
    try:
        yield arm, renderer, cam
    finally:
        renderer.disconnect()
        arm.disconnect()


def test_teleop_render_camera_record_pipeline(rig):
    arm, renderer, cam = rig
    rec = EpisodeRecorder()
    rec.start({"task": "integration-smoke"})

    n = 10
    for _ in range(n):
        tel = arm.controller.step(Command(lin=np.array([1.0, 0.0, 0.2])))
        renderer.set_joints(tel.joints_sent)  # mirror commanded config (what dume sim does)
        obs = observe(arm.arm, arm.kin, camera=cam)
        rec.record(obs, Action(joints_target=tel.joints_sent.copy()))

    episode = rec.finish()
    assert len(episode) == n
    # The EE camera produced metric depth of the right shape on each step.
    assert episode.steps[0].observation.depth is not None
    assert episode.steps[0].observation.depth.shape == (120, 160)
    # Actions are the commanded joints (length-6, includes gripper).
    assert episode.steps[-1].action.joints_target.shape == (6,)


def test_camera_aimed_at_box_detects_it(rig):
    """Aim the camera straight at the demo box (bypassing FK) and confirm a detection."""
    arm, renderer, cam = rig
    from dume import geometry as g

    # Place the camera at x=0.70 looking back toward -x at the box (x=0.30). Optical +z must
    # point along world -x, which is Ry(-pi/2): R[:,2] = [sin(-pi/2),0,cos(-pi/2)] = [-1,0,0].
    cam._pose_provider = lambda: g.transform_from_pos_rpy([0.70, 0.0, 0.10], [0.0, -np.pi / 2, 0.0])
    cam.capture()
    dets = cam.detect()
    assert len(dets) >= 1
    assert np.all(dets.depths > 0)
