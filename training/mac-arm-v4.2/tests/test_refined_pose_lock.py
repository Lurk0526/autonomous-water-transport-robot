import unittest

import numpy as np

from arm_training.precision_ik import PrecisionIK
from arm_training.repaired_env import RepairedBowlEnv


class RefinedPoseLockTests(unittest.TestCase):
    def test_refinement_repairs_the_reported_lift_with_the_same_height(self):
        env = RepairedBowlEnv()
        env.reset(seed=112034)
        stale_ik = PrecisionIK(env.model)
        stale_ik.orientation_target = env.ik.orientation_target.copy()
        initial_grasp = env.targets[2].copy()
        initial_carry = env.targets[3].copy()
        for _ in range(80):
            command = env.command.copy()
            _, _, done, _, info = env.step(np.zeros(3))
            self.assertFalse(done, info['failure_reason'])
            if env.target_refinement is not None:
                break
        self.assertEqual(env.phase, 'approach')
        self.assertFalse(env.grip_contact_confirmed)
        self.assertIsNotNone(env.target_refinement)
        self.assertTrue(env.target_refinement['full_pose_refit'])
        max_step = env.cfg['arm']['max_joint_velocity_rad_s'] * env.dt
        self.assertLessEqual(np.max(np.abs(env.command - command)), max_step + 1e-10)

        correction = env.target_refinement['correction_m']
        np.testing.assert_allclose(env.targets[2], initial_grasp + correction)
        np.testing.assert_allclose(env.targets[3], initial_carry + correction)
        np.testing.assert_allclose(env.targets[3] - env.targets[2], [0., 0., .06])
        self.assertFalse(stale_ik.solve(env.targets[3], seed=env.home_ik_seed)[1])
        lock = env.ik.orientation_target.copy()
        np.testing.assert_allclose(lock[:, 1], [0., 0., 1.], atol=.003)
        q = env.home_ik_seed
        for point in np.linspace(env.targets[2], env.targets[3], 51):
            q, possible = env.ik.solve(point, seed=q, iterations=35)
            self.assertTrue(possible, point)
        np.testing.assert_array_equal(env.ik.orientation_target, lock)

    def test_frozen_wrist_target_bypasses_simulated_refinement(self):
        env = RepairedBowlEnv()
        measured = np.array([.2, 0., .35])
        env.reset(seed=112034, options={
            'parking': [0., 0., 0.],
            'target_estimate_base_m': measured,
            'observation_source': 'frozen_wrist_estimate',
        })
        lock = env.ik.orientation_target.copy()
        for _ in range(90):
            _, _, done, _, info = env.step(np.zeros(3))
            self.assertFalse(done, info['failure_reason'])
            if env.phase == 'close':
                break
        self.assertEqual(env.phase, 'close')
        self.assertIsNone(env.target_refinement)
        np.testing.assert_array_equal(env.estimate, measured)
        np.testing.assert_array_equal(env.ik.orientation_target, lock)


if __name__ == '__main__':
    unittest.main()
