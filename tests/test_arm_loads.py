"""``read_loads`` on every ArmIO, plus the drive_mode sign guard (force-sensing SP1)."""

import numpy as np
import pytest

from dume.arm import MOTOR_ORDER, SimArm, SO101Arm, assert_zero_drive_modes


class _Bus:
    def __init__(self, load, voltage=120):
        self.load, self.voltage = load, voltage
        self.calls = []

    def sync_read(self, name, motors=None, *, normalize=True):
        self.calls.append((name, list(motors), normalize))
        if name == "Present_Load":
            return {m: self.load[m] for m in motors}
        if name == "Present_Voltage":
            return {m: self.voltage for m in motors}
        raise AssertionError(name)


class _Robot:
    def __init__(self, bus):
        self.bus = bus


def test_sim_arm_loads_default_to_zero():
    arm = SimArm()
    loads = arm.read_loads()
    assert loads.shape == (6,)
    assert np.all(loads == 0.0)


def test_sim_arm_loads_come_from_injected_source():
    arm = SimArm(load_source=lambda q: np.arange(6.0) * 10 + q[1])
    arm.write_joints([0, -20, 20, 0, 0, 50])
    assert np.allclose(arm.read_loads(), [-20, -10, 0, 10, 20, 30])


def test_so101_read_loads_returns_raw_signed_units_in_motor_order():
    load = {m: v for m, v in zip(MOTOR_ORDER, [5, -420, 310, -12, 0, 77])}
    bus = _Bus(load)
    arm = SO101Arm("/dev/null", "test")
    arm._robot = _Robot(bus)
    assert np.array_equal(arm.read_loads(), [5, -420, 310, -12, 0, 77])
    # Raw: Present_Load is not in lerobot's normalized_data, so normalisation must be off.
    assert bus.calls == [("Present_Load", MOTOR_ORDER, False)]


def test_so101_read_voltage_is_volts():
    bus = _Bus({}, voltage=123)  # register unit is 0.1 V
    arm = SO101Arm("/dev/null", "test")
    arm._robot = _Robot(bus)
    assert arm.read_voltage() == pytest.approx(12.3)


def _calib(drive_modes):
    class C:
        def __init__(self, dm):
            self.drive_mode = dm
    return {m: C(dm) for m, dm in zip(MOTOR_ORDER, drive_modes)}


def test_drive_mode_guard_passes_on_all_zero():
    assert_zero_drive_modes(_calib([0] * 6))


def test_drive_mode_guard_names_the_inverted_motor():
    with pytest.raises(RuntimeError, match="elbow_flex"):
        assert_zero_drive_modes(_calib([0, 0, 1, 0, 0, 0]))


def test_pybullet_arm_loads_hold_gravity():
    """Holding a pose under physics, the applied motor torque on shoulder_lift is non-zero and
    has the sign of the modelled gravity torque — the sim is a truth source for the estimator."""
    from dume.kinematics import Kinematics
    from dume.poses import HOME_JOINTS
    from dume.sim_world import PyBulletArm, SimRenderer

    renderer = SimRenderer(gui=False, dynamic=True)  # gravity is only on in dynamic mode
    try:
        arm = PyBulletArm(renderer)
        arm.connect()
        for _ in range(25):
            arm.write_joints(HOME_JOINTS)
        loads = arm.read_loads()
        tau_g = Kinematics().gravity_torques(arm.read_joints())
        assert loads.shape == (6,)
        assert abs(loads[1]) > 0.05
        assert np.sign(loads[1]) == np.sign(tau_g[1])
    finally:
        renderer.disconnect()
