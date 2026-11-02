"""Adversarial accounting tests; no physics/success evidence is generated here."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim' / 'bowl_transport'))
from reward_audit import RewardAudit, RewardConfig, EpisodeMetrics, compare_episodes, run_attacks


class RewardAccounting(unittest.TestCase):
    def test_attack_report(self):
        report = run_attacks()
        self.assertTrue(report['all_passed'], report)

    def test_continuous_time_integration_and_potential_telescope(self):
        results = []
        for dt in (.1, .05, .02):
            audit = RewardAudit(300, RewardConfig(horizon_s=10), max_spill_g=5)
            for i in range(round(4 / dt)):
                result = audit.advance(dt, progress=(i % 3) / 2,
                                       motion=.3, risk_proxy=.7, intervention=.1,
                                       cumulative_loss_g=0,
                                       outcome='success' if i == round(4 / dt) - 1 else None)
            self.assertEqual(audit.potential, 0)
            results.append(result['candidate_discounted_return'])
        for value in results:
            self.assertAlmostEqual(value, results[0], places=11)

    def test_irreversible_loss_and_no_refunds(self):
        audit = RewardAudit(300)
        first = audit.advance(.1, cumulative_loss_g=10)
        second = audit.advance(.1, cumulative_loss_g=10)
        self.assertLess(first['loss_charge'], 0)
        self.assertEqual(second['loss_charge'], 0)
        with self.assertRaises(ValueError):
            audit.advance(.1, cumulative_loss_g=9)
        with self.assertRaises(ValueError):
            audit.advance(.1, cumulative_loss_g=301)

    def test_missing_evidence_stays_unknown(self):
        audit = RewardAudit(300)
        self.assertIsNone(audit.advance(.1)['candidate_reward'])
        result = audit.advance(.1, cumulative_loss_g=0, outcome='success')
        self.assertIsNone(result['candidate_discounted_return'])
        self.assertFalse(result['evidence_complete'])

    def test_timeout_is_terminal_failure_and_cannot_extend(self):
        audit = RewardAudit(300, RewardConfig(horizon_s=1))
        result = audit.advance(1, progress=1, cumulative_loss_g=0)
        self.assertEqual(result['terminal'], 'timeout')
        self.assertTrue(result['terminated'])
        self.assertFalse(result['truncated'])
        self.assertLess(result['candidate_discounted_return'], 0)
        self.assertEqual(audit.potential, 0)
        with self.assertRaises(RuntimeError):
            audit.advance(.1, outcome='success', cumulative_loss_g=0)

    def test_failure_separated_from_worst_valid_success(self):
        cfg = RewardConfig(horizon_s=1)
        failure = RewardAudit(300, cfg).advance(1, cumulative_loss_g=0, outcome='failure')
        worst = RewardAudit(300, cfg, initial_progress=1, max_spill_g=5).advance(
            1, cumulative_loss_g=5, motion=1, risk_proxy=1, intervention=1, outcome='success')
        self.assertLess(failure['candidate_discounted_return'], 0)
        self.assertGreater(worst['candidate_discounted_return'], 0)
        lost_all = RewardAudit(300, cfg, max_spill_g=5).advance(
            1, cumulative_loss_g=300, outcome='success')
        self.assertEqual(lost_all['terminal'], 'failure')
        self.assertLess(lost_all['candidate_discounted_return'], 0)

    def test_success_requires_known_acceptance_limit_and_measurements(self):
        result=RewardAudit(300).advance(.1,cumulative_loss_g=0,outcome='success')
        self.assertEqual(result['terminal'],'unverified_success')
        self.assertIsNone(result['candidate_discounted_return'])

    def test_postponing_identical_water_loss_gives_no_reward_advantage(self):
        returns=[]
        for loss_at in (0,9):
            a=RewardAudit(300,RewardConfig(horizon_s=10),max_spill_g=5)
            for i in range(10):
                r=a.advance(.1,cumulative_loss_g=2 if i>=loss_at else 0,
                            outcome='success' if i==9 else None)
            returns.append(r['candidate_discounted_return'])
        self.assertAlmostEqual(*returns,places=12)

    def test_spill_limit_cannot_be_compensated_by_success_bonus(self):
        audit = RewardAudit(300, max_spill_g=5)
        result = audit.advance(.1, cumulative_loss_g=6, outcome='success')
        self.assertEqual(result['terminal'], 'failure')
        self.assertLess(result['candidate_discounted_return'], 0)

    def test_invalid_inputs_and_bad_config_are_rejected(self):
        for args in ({'dt':float('nan')}, {'dt':-1}, {'dt':1, 'motion':1.1},
                     {'dt':1, 'risk_proxy':float('inf')}, {'dt':1, 'progress':-1}):
            with self.assertRaises(ValueError):
                RewardAudit(300).advance(**args)
        with self.assertRaises(ValueError):
            RewardConfig(soft_budget=10)
        with self.assertRaises(ValueError):
            RewardAudit(300, RewardConfig(horizon_s=1)).advance(2, outcome='success')


class IndependentEvaluation(unittest.TestCase):
    def setUp(self):
        self.base = EpisodeMetrics('seed_1', 'synthetic', 300, 350, 650,
                                   'to_rim', 'success', 1, 10)

    def test_failure_precedes_spill_precedes_time(self):
        for bad in (replace(self.base, outcome='failure', escaped_water_g=0, duration_s=1),
                    replace(self.base, hard_failure=True, escaped_water_g=0, duration_s=1),
                    replace(self.base, escaped_water_g=2, duration_s=1),
                    replace(self.base, duration_s=11)):
            self.assertEqual(compare_episodes(self.base, bad)['preferred'], 'a')

    def test_unknown_or_unmatched_evidence_cannot_be_ranked(self):
        for other in (replace(self.base, initial_water_g=None),
                      replace(self.base, empty_bowl_g=None),
                      replace(self.base, escaped_water_g=None),
                      replace(self.base, filled_total_g=600),
                      replace(self.base, initial_water_g=250, filled_total_g=600),
                      replace(self.base, scenario_id='seed_2'),
                      replace(self.base, source='measured'),
                      replace(self.base, fill_condition='low')):
            with self.assertRaises(ValueError):
                compare_episodes(self.base, other)

    def test_unspecified_spill_threshold_does_not_claim_acceptance(self):
        result = compare_episodes(self.base, replace(self.base, duration_s=11))
        self.assertEqual(result['spill_acceptance'], 'undetermined_no_threshold')
        result = compare_episodes(self.base, replace(self.base, escaped_water_g=10), max_spill_g=5)
        self.assertEqual(result['b_key'][0], 1)
        self.assertEqual(result['a_key'][0], 0)


if __name__ == '__main__':
    unittest.main()
