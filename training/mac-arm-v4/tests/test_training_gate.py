"""Evidence-admission regressions, to be executed on the receiving Mac."""
import copy
from collections import Counter
import hashlib
import json
import unittest

from arm_training.training_gate import build_training_gate


TEST_LOG = 'test_evidence (tests.test_guard.GuardTests) ... ok\n\nRan 12 tests in 0.410s\n\nOK\n'


def evidence():
    config = {'task': {'joint_tracking_limit_rad': .25, 'gate_tracking_rad': .06,
                       'contact_supervisor_mode': 'foam_required'},
              'gripper': {'closure_limit_tolerance_m': .001,
                          'closure_sync_tolerance_m': .003}}
    provenance = {'package_generation': 'mac-arm-v4', 'config': config,
                  'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  'source_sha256': 'a' * 64, 'model_sha256': 'b' * 64}
    baseline = {'episodes': 100, 'successes': 100, 'success_rate': 1., 'failures': {},
        'provenance': copy.deepcopy(provenance),
        'trials': [{'seed': 30000 + index, 'is_success': True, 'failure_reason': None,
                    'first_collision': None, 'first_physics_anomaly': None,
                    'guard_events': [], 'q_tracking_error_rad': .01} for index in range(100)]}
    mobile = {'episodes': 20, 'learned_vision': True,
        'vision_backend': 'ultralytics_yolo_segmenter', 'provenance': copy.deepcopy(provenance),
        'trials': [{'seed': 60000 + index, 'recognition_success': True,
                    'parking_success': True, 'precise_localization_success': True,
                    'raw_grasp_result': {'first_collision': None, 'first_physics_anomaly': None,
                                         'guard_events': []}} for index in range(20)]}
    expected = {key: provenance[key] for key in ('package_generation', 'config_sha256', 'source_sha256', 'model_sha256')}
    return baseline, mobile, expected


def gate(baseline=None, mobile=None, expected=None, log=TEST_LOG):
    first, second, third = evidence()
    return build_training_gate(first if baseline is None else baseline,
                               second if mobile is None else mobile, log,
                               expected_provenance=third if expected is None else expected)


def recount(baseline):
    baseline['episodes'] = len(baseline['trials'])
    baseline['successes'] = sum(trial['is_success'] for trial in baseline['trials'])
    baseline['success_rate'] = baseline['successes'] / max(1, baseline['episodes'])
    baseline['failures'] = dict(Counter(trial['failure_reason'] for trial in baseline['trials'] if trial['failure_reason']))


class TrainingGateTests(unittest.TestCase):
    def test_complete_evidence_allows_only_bounded_component_training(self):
        result = gate()
        self.assertTrue(result['arm_residual_training_allowed'])
        self.assertTrue(result['mobile_learning_allowed'])
        self.assertFalse(result['end_to_end_policy_training_allowed'])
        self.assertFalse(result['hardware_safety_certification'])
        self.assertEqual(result['training_schedule_if_allowed'], [20000, 100000])

    def test_smoke_three_of_three_never_qualifies(self):
        baseline, mobile, expected = evidence()
        baseline['trials'] = baseline['trials'][:3]
        recount(baseline)
        result = gate(baseline, mobile, expected)
        self.assertFalse(result['arm_residual_training_allowed'])
        self.assertIn('baseline_episode_count_insufficient', {item['code'] for item in result['reasons']})

    def test_ninety_five_percent_threshold_and_full_denominator(self):
        baseline, mobile, expected = evidence()
        for trial in baseline['trials'][:5]:
            trial.update(is_success=False, failure_reason='grasp_contact_missing')
        recount(baseline)
        self.assertTrue(gate(baseline, mobile, expected)['arm_residual_training_allowed'])
        baseline['trials'][5].update(is_success=False, failure_reason='grasp_contact_missing')
        recount(baseline)
        self.assertFalse(gate(baseline, mobile, expected)['arm_residual_training_allowed'])

    def test_complete_tests_log_required_and_failures_cannot_be_hidden(self):
        for log in ['', 'OK\n', TEST_LOG + '\nFAILED (failures=1)\n',
                    TEST_LOG + '\nERROR: test_guard\n',
                    'Ran 12 tests in 0.4s\nOK (skipped=12)\n']:
            with self.subTest(log=log):
                result = gate(log=log)
                self.assertFalse(result['arm_residual_training_allowed'])
                self.assertFalse(result['checks']['tests']['valid'])

    def test_missing_new_guard_fields_do_not_mean_no_anomalies(self):
        baseline, mobile, expected = evidence()
        del baseline['trials'][0]['first_physics_anomaly']
        result = gate(baseline, mobile, expected)
        self.assertFalse(result['arm_residual_training_allowed'])
        self.assertFalse(result['checks']['baseline']['guard_evidence_complete'])

    def test_serious_guard_events_block_even_one_hundred_percent_success(self):
        for reason in ['hard_contact_before_foam', 'closure_joint_limit',
                       'closure_coupling_limit', 'physics_warning_or_nonfinite',
                       'unexpected_collision', 'joint_tracking_limit', 'table_collision']:
            baseline, mobile, expected = evidence()
            baseline['trials'][0]['guard_events'] = [{'kind': reason, 'time_s': .1}]
            result = gate(baseline, mobile, expected)
            with self.subTest(reason=reason):
                self.assertEqual(result['checks']['baseline']['success_rate_from_trials'], 1.)
                self.assertFalse(result['arm_residual_training_allowed'])
                self.assertGreater(result['checks']['baseline']['serious_event_count'], 0)

    def test_safe_contact_stop_is_not_itself_a_fault(self):
        baseline, mobile, expected = evidence()
        baseline['trials'][0]['guard_events'] = [
            {'kind': 'gripper_contact_stop', 'reason': 'bilateral_contact'},
            {'kind': 'support_stop_descent', 'support_n': 6.4}]
        self.assertTrue(gate(baseline, mobile, expected)['arm_residual_training_allowed'])

    def test_relaxing_tracking_or_closure_thresholds_is_rejected(self):
        for section, key, value in [('task', 'joint_tracking_limit_rad', .35),
                                    ('gripper', 'closure_limit_tolerance_m', .01)]:
            baseline, mobile, expected = evidence()
            config = baseline['provenance']['config']
            config[section][key] = value
            digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            baseline['provenance']['config_sha256'] = digest
            mobile['provenance'] = copy.deepcopy(baseline['provenance'])
            expected['config_sha256'] = digest
            result = gate(baseline, mobile, expected)
            self.assertFalse(result['arm_residual_training_allowed'])
            self.assertIn('guard_threshold_missing_or_relaxed', {item['code'] for item in result['reasons']})

    def test_hash_mismatch_old_generation_and_embedded_config_tampering(self):
        for change in ['source', 'generation', 'config']:
            baseline, mobile, expected = evidence()
            if change == 'source':
                baseline['provenance']['source_sha256'] = 'c' * 64
            elif change == 'generation':
                baseline['provenance']['package_generation'] = 'mac-arm-v2'
            else:
                baseline['provenance']['config']['task']['gate_tracking_rad'] = .05
            result = gate(baseline, mobile, expected)
            self.assertFalse(result['arm_residual_training_allowed'])

    def test_summary_and_repeated_seeds_cannot_inflate_sample(self):
        baseline, mobile, expected = evidence()
        baseline['trials'][1]['seed'] = baseline['trials'][0]['seed']
        self.assertFalse(gate(baseline, mobile, expected)['arm_residual_training_allowed'])
        baseline, mobile, expected = evidence()
        baseline['successes'] = 101
        self.assertFalse(gate(baseline, mobile, expected)['arm_residual_training_allowed'])

    def test_contract_backend_or_missing_mobile_trials_cannot_admit_mobile_learning(self):
        baseline, mobile, expected = evidence()
        mobile.update(learned_vision=False, vision_backend='contract_backend')
        result = gate(baseline, mobile, expected)
        self.assertTrue(result['arm_residual_training_allowed'])
        self.assertFalse(result['mobile_learning_allowed'])
        mobile = {}
        result = gate(baseline, mobile, expected)
        self.assertTrue(result['arm_residual_training_allowed'])
        self.assertFalse(result['mobile_learning_allowed'])

    def test_mobile_ninety_five_percent_each_and_per_episode_backend(self):
        baseline, mobile, expected = evidence()
        mobile['trials'][0]['recognition_success'] = False
        self.assertTrue(gate(baseline, mobile, expected)['mobile_learning_allowed'])
        mobile['trials'][1]['recognition_success'] = False
        self.assertFalse(gate(baseline, mobile, expected)['mobile_learning_allowed'])
        baseline, mobile, expected = evidence()
        mobile['trials'][0]['vision_backend'] = 'contract_backend'
        self.assertFalse(gate(baseline, mobile, expected)['mobile_learning_allowed'])

    def test_nested_mobile_grasp_anomaly_does_not_get_lost(self):
        baseline, mobile, expected = evidence()
        mobile['trials'][0]['raw_grasp_result']['first_physics_anomaly'] = {
            'kind': 'closure_coupling_limit', 'closure_m': [0., .02, 0., 0.]}
        result = gate(baseline, mobile, expected)
        self.assertTrue(result['arm_residual_training_allowed'])
        self.assertFalse(result['mobile_learning_allowed'])
        self.assertGreater(result['checks']['mobile']['serious_event_count'], 0)

    def test_platform_localization_alone_cannot_pass_full_grasp_gate(self):
        baseline,mobile,expected=evidence()
        mobile.update(pipeline='observed_platform_drive_wrist_physical_grasp_v4',localization_only=False)
        for trial in mobile['trials']:trial['is_success']=True
        self.assertTrue(gate(baseline,mobile,expected)['mobile_learning_allowed'])
        for trial in mobile['trials'][:2]:trial['is_success']=False
        self.assertFalse(gate(baseline,mobile,expected)['mobile_learning_allowed'])
        for trial in mobile['trials']:trial['is_success']=True
        mobile['localization_only']=True
        self.assertFalse(gate(baseline,mobile,expected)['mobile_learning_allowed'])

    def test_nonfinite_or_malformed_evidence_fails_closed_and_report_is_json_serializable(self):
        baseline, mobile, expected = evidence()
        baseline['trials'][0]['q_tracking_error_rad'] = float('nan')
        baseline['trials'][0]['first_physics_anomaly'] = {'kind': 'nonfinite', 'q': float('inf')}
        result = gate(baseline, mobile, expected)
        self.assertFalse(result['arm_residual_training_allowed'])
        json.dumps(result, allow_nan=False)
        baseline['provenance'] = None
        mobile['vision_backend'] = {'unexpected': 'object'}
        result = gate(baseline, mobile, expected)
        self.assertFalse(result['arm_residual_training_allowed'])
        self.assertFalse(result['mobile_learning_allowed'])


if __name__ == '__main__':
    unittest.main()
