import copy
import unittest

from arm_training.cli import validate_paired_case


class PairingRegressionTests(unittest.TestCase):
    def test_allow_roundoff_but_reject_changed_scenario_or_shape(self):
        a = {'seed':7,'parking_xy_yaw':[.123,0.,0.],'scenario':{'initial_target_error_m':[.001,0,0]}}
        b = copy.deepcopy(a)
        b['parking_xy_yaw'][0] += 1e-14
        validate_paired_case(a, b)
        b['scenario']['initial_target_error_m'][0] += .0001
        with self.assertRaises(ValueError):
            validate_paired_case(a, b)
        b['scenario']['initial_target_error_m'] = .001
        with self.assertRaises(ValueError):
            validate_paired_case(a, b)
