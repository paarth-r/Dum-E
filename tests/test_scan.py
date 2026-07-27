"""Scan sequencing and — the load-bearing part — where camera poses come from.

The regression these guard: a future refactor that "helpfully" poses frames from the
controller's commanded reference (q_ref, or Telemetry.joints_sent) would still run, still
produce frames, and silently bend every reconstruction. Nothing would fail visibly.
"""

import numpy as np
import pytest

from dume.camera import T_CAM_MOUNT, CameraFrame
from dume.kinematics import Kinematics
from dume.poses import HOME_JOINTS, JointPoseStore
from dume.scan import averaged_joints, measured_camera_pose, run_scan


class _FakeArmIO:
    """Reports joints that differ from what was commanded — i.e. an arm that sags."""

    def __init__(self, measured, noise=0.0, seed=0):
        self.measured = np.asarray(measured, dtype=float)
        self.commanded = None
        self._rng = np.random.default_rng(seed)
        self._noise = noise

    def read_joints(self):
        q = self.measured.copy()
        if self._noise:
            q[:5] += self._rng.normal(0, self._noise, 5)
        return q


class _FakeDumeArm:
    def __init__(self, kin, arm):
        self.kin = kin
        self.arm = arm
        self.visited = []

    def goto_joints(self, joints, *, on_tick=None, **kw):
        self.arm.commanded = np.asarray(joints, dtype=float)
        self.visited.append(np.asarray(joints, dtype=float))
        for _ in range(3):
            if on_tick is not None and on_tick() is False:
                return
        return None


class _FakeCamera:
    def __init__(self):
        self.captures = 0

    def capture(self):
        self.captures += 1
        return CameraFrame(pose=np.eye(4), rgb=np.zeros((4, 4), dtype=np.uint8))


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


@pytest.fixture
def store(tmp_path):
    s = JointPoseStore(tmp_path / "joint_poses.json")
    s.set("a", HOME_JOINTS)
    s.set("b", HOME_JOINTS + np.array([10.0, 5.0, -5.0, 0.0, 0.0, 0.0]))
    return s


# ---------------------------------------------------------------------------
# Pose provenance — the reason this module exists
# ---------------------------------------------------------------------------

def test_pose_uses_measured_joints_not_commanded(kin, store):
    """The arm is commanded to 'a' but physically sits elsewhere (sag). Pose must follow reality."""
    sagged = HOME_JOINTS + np.array([0.0, -6.0, 3.0, 0.0, 0.0, 0.0])
    arm = _FakeArmIO(measured=sagged)
    stops = list(run_scan(_FakeDumeArm(kin, arm), _FakeCamera(), ["a"], store, dwell=0, sleep=lambda s: None))

    expected_measured = kin.fk(sagged) @ T_CAM_MOUNT
    expected_commanded = kin.fk(store.get("a")) @ T_CAM_MOUNT

    assert np.allclose(stops[0].pose, expected_measured)
    assert not np.allclose(stops[0].pose, expected_commanded)  # the bug this guards


def test_frame_pose_is_overwritten_with_the_measured_pose(kin, store):
    """Whatever the camera source guessed is discarded; the scan's measured pose wins."""
    arm = _FakeArmIO(measured=HOME_JOINTS)
    cam = _FakeCamera()
    stops = list(run_scan(_FakeDumeArm(kin, arm), cam, ["a"], store, dwell=0, sleep=lambda s: None))
    assert not np.allclose(stops[0].frame.pose, np.eye(4))
    assert np.allclose(stops[0].frame.pose, stops[0].pose)


def test_averaging_reduces_measurement_noise():
    """Servo quantisation is why control avoids measured joints; the dwell lets us average it out."""
    truth = HOME_JOINTS
    noisy = _FakeArmIO(measured=truth, noise=0.5, seed=3)
    single = noisy.read_joints()
    averaged = averaged_joints(_FakeArmIO(measured=truth, noise=0.5, seed=3), 40, sleep=lambda s: None)
    assert np.linalg.norm(averaged[:5] - truth[:5]) < np.linalg.norm(single[:5] - truth[:5])


def test_averaged_joints_requires_at_least_one_sample():
    with pytest.raises(ValueError):
        averaged_joints(_FakeArmIO(HOME_JOINTS), 0)


def test_measured_camera_pose_composes_the_mount(kin):
    arm = _FakeArmIO(measured=HOME_JOINTS)
    pose, joints = measured_camera_pose(kin, arm, samples=1, sleep=lambda s: None)
    assert np.allclose(pose, kin.fk(HOME_JOINTS) @ T_CAM_MOUNT)
    assert np.allclose(joints, HOME_JOINTS)


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------

def test_visits_every_setpoint_in_order(kin, store):
    arm = _FakeArmIO(measured=HOME_JOINTS)
    fake = _FakeDumeArm(kin, arm)
    stops = list(run_scan(fake, _FakeCamera(), ["a", "b"], store, dwell=0, sleep=lambda s: None))
    assert [s.name for s in stops] == ["a", "b"]
    assert np.allclose(fake.visited[0], store.get("a"))
    assert np.allclose(fake.visited[1], store.get("b"))


def test_one_capture_per_stop(kin, store):
    cam = _FakeCamera()
    list(run_scan(_FakeDumeArm(kin, _FakeArmIO(HOME_JOINTS)), cam, ["a", "b"], store,
                  dwell=0, sleep=lambda s: None))
    assert cam.captures == 2


def test_unknown_setpoint_raises_before_moving(kin, store):
    """Validate the whole list up front — discovering a typo mid-scan means the arm has
    already moved somewhere on a plan that was never going to complete."""
    fake = _FakeDumeArm(kin, _FakeArmIO(HOME_JOINTS))
    with pytest.raises(KeyError, match="nope"):
        list(run_scan(fake, _FakeCamera(), ["a", "nope"], store, dwell=0, sleep=lambda s: None))
    assert fake.visited == []


def test_abort_stops_the_whole_scan_not_just_one_move(kin, store):
    """One keypress must stop the arm, not skip to the next setpoint."""
    fake = _FakeDumeArm(kin, _FakeArmIO(HOME_JOINTS))
    stops = list(run_scan(fake, _FakeCamera(), ["a", "b"], store,
                          dwell=0, sleep=lambda s: None, on_tick=lambda: False))
    assert stops == []
    assert len(fake.visited) == 1  # aborted during the first move, never started the second
