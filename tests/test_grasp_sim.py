"""Physical grasp sim: GraspParams physics setup + the PyBulletArm faithful ArmIO.

Headless (DIRECT) + dynamic. These assert the physics substrate (friction/solver knobs land on
the real bodies) and that the control stack's hardware surrogate reports *physical* feedback —
including a jaw stalled on a box. Reliable pick-and-place is intentionally NOT asserted (that's the
later high-fidelity phase); see the design spec.
"""

import numpy as np
import pybullet as p

from dume.sim_world import GraspParams, SimRenderer


def test_grasp_params_friction_lands_on_jaw_links():
    grasp = GraspParams(jaw_friction=2.0)
    with SimRenderer(gui=False, dynamic=True, grasp=grasp) as r:
        jaw = r.link_index("moving_jaw_so101_v1_link")
        fric = p.getDynamicsInfo(r.arm_body, jaw, physicsClientId=r.client)[1]
        assert fric == 2.0
