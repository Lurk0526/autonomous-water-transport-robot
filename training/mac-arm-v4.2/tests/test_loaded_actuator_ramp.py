"""Targeted regression for the measured 1 ms setpoint discontinuity."""
import unittest

import numpy as np

from arm_training.repaired_env import RepairedBowlEnv


class LoadedActuatorRampTests(unittest.TestCase):
    def setUp(self):
        self.env = RepairedBowlEnv.__new__(RepairedBowlEnv)
        self.env.phase_index = 2
        self.env.support_lock_q = None
        self.env.actuator_target = None
        self.env.ramp_goal = None
        self.env.ramp_start = None
        self.env.ramp_elapsed = 0.
        self.env.cfg = {'arm': {'max_joint_velocity_rad_s': .7}}
        self.env.dt = .05
        self.env.physics_dt = .001

    def test_loaded_ramp_reaches_the_original_endpoint_without_a_step_jump(self):
        env = self.env
        prior = env._actuator_position_target(np.zeros(5)).copy()
        env.phase_index = 3    # lift
        requested = np.array([0., -.004432, .026622, -.022192, 0.])
        for tick in range(50):
            target = env._actuator_position_target(requested).copy()
            np.testing.assert_allclose(target, requested * ((tick + 1) / 50), atol=1e-15)
            self.assertLessEqual(np.max(np.abs(target - prior)), .0007 + 1e-15)
            prior = target
        np.testing.assert_allclose(target, requested, atol=1e-15)

    def test_touchdown_uses_the_support_lock_immediately(self):
        env = self.env
        env.phase_index = 5    # lower
        env._actuator_position_target(np.array([0., .1, .2, .3, .4]))
        env._actuator_position_target(np.array([0., .09, .21, .32, .4]))
        # The support handler sets ctrl[:5] to actual settled q and clears the queue.
        env.support_lock_q = np.array([0., .1, .18, .32, .4])
        np.testing.assert_array_equal(env._actuator_position_target(env.support_lock_q), env.support_lock_q)
        self.assertIsNone(env.ramp_goal)
