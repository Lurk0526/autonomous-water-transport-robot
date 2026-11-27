"""Regression tests for full-pose carry residual feasibility."""
import unittest
from unittest.mock import patch

import numpy as np

from arm_training.repaired_env import RepairedBowlEnv


class CarryResidualTests(unittest.TestCase):
    def setUp(self):
        self.env = RepairedBowlEnv()
        _, info = self.env.reset(seed=112000, options={
            'parking': [0., 0., 0.],
            'target_estimate_base_m': [.2, 0., .35],
            'observation_source': 'frozen_wrist_estimate',
        })
        self.assertTrue(info['reachable'])
        self.env.phase_index = 4
        self.env.current_limit = self.env.cfg['simulation']['phase_residual_limits_m'][4]
        self.env.phase_start_limit = self.env.current_limit
        self.env.nominal = self.env.targets[4].copy()
        self.env.start_target = self.env.nominal.copy()
        self.action = np.array([1., -1., 1.])
        self.max_change = self.env.cfg['simulation']['residual_rate_m_s'] * self.env.dt

    def test_unreachable_request_uses_reachable_rate_limited_residual(self):
        env = self.env
        previous = np.zeros(3)
        requested = self.action * self.max_change
        seed = env.command.copy()
        self.assertFalse(env.ik.solve(env.nominal + requested, seed=seed)[1])
        orientation = env.ik.orientation_target.copy()
        env.residual = requested.copy()
        env.waypoint = env.nominal + requested

        q, possible = env._solve_waypoint(seed, previous)

        self.assertTrue(possible)
        self.assertIsNone(env.ik_failure)
        self.assertEqual(env.residual_projection_count, 1)
        sample = env.last_residual_projection
        self.assertEqual(sample['phase'], 'hold')
        np.testing.assert_allclose(sample['requested_residual_m'], requested)
        np.testing.assert_allclose(sample['applied_residual_m'], env.residual)
        self.assertLess(np.linalg.norm(env.residual), np.linalg.norm(requested))
        self.assertLessEqual(np.max(np.abs(env.residual - previous)), self.max_change + 1e-12)
        np.testing.assert_array_equal(env.ik.orientation_target, orientation)
        details = env.ik.rejection_details(env.waypoint, q)
        self.assertLess(details['position_error_m'], details['position_tolerance_m'])
        self.assertLess(details['orientation_error'], details['orientation_tolerance'])
        self.assertTrue(np.all(q >= env.ranges[:, 0] + .01))
        self.assertTrue(np.all(q <= env.ranges[:, 1] - .01))

    def test_failed_nominal_terminates_before_residual_or_joint_command(self):
        env = self.env
        command = env.command.copy()
        env.residual = self.action * self.max_change
        with patch.object(env.ik, 'solve', return_value=(command + .01, False)) as solver:
            _, _, done, _, info = env.step(self.action)

        self.assertTrue(done)
        self.assertEqual(info['failure_reason'], 'waypoint_unreachable')
        self.assertEqual(info['ik_failure']['target_kind'], 'nominal')
        np.testing.assert_allclose(info['ik_failure']['target_m'], env.nominal)
        np.testing.assert_allclose(info['ik_failure']['requested_residual_m'],
                                   self.action * env.current_limit)
        self.assertIsNone(info['ik_failure']['applied_residual_m'])
        self.assertEqual(solver.call_count, 1)
        np.testing.assert_array_equal(solver.call_args.args[0], env.nominal)
        np.testing.assert_array_equal(env.command, command)
        np.testing.assert_array_equal(env.data.ctrl[:5], command)
        self.assertEqual(env.data.time, 0.)

    def test_slew_exhaustion_terminates_without_using_out_of_rate_zero(self):
        env = self.env
        command = env.command.copy()
        env.residual = self.action * env.current_limit
        requests = []

        def solve(point, **_kwargs):
            requests.append(point.copy())
            return command.copy(), len(requests) == 1

        with patch.object(env.ik, 'solve', side_effect=solve):
            _, _, done, _, info = env.step(self.action)

        self.assertTrue(done)
        self.assertEqual(info['failure_reason'], 'waypoint_unreachable')
        self.assertEqual(info['ik_failure']['target_kind'], 'residual_within_slew_limit')
        self.assertIsNone(info['ik_failure']['applied_residual_m'])
        self.assertEqual(info['synchronous_control']['residual_projection_count'], 0)
        self.assertGreater(len(requests), 2)
        for point in requests[1:]:
            residual = point - env.nominal
            self.assertGreater(np.linalg.norm(residual), 0.)
            self.assertLessEqual(np.max(np.abs(residual - self.action * env.current_limit)),
                                 self.max_change + 1e-12)
        np.testing.assert_array_equal(env.command, command)
        self.assertEqual(env.data.time, 0.)


if __name__ == '__main__':
    unittest.main()
