import json
import unittest
from unittest.mock import patch
import numpy as np

from arm_training.control import synchronized_step,checkpoint_key
from arm_training.env import BowlEnv
from arm_training.rewards import Reward
from arm_training.cli import paired_compare,episode


class RegressionTests(unittest.TestCase):
    def test_joint_rate_limit_preserves_coordinated_pitch(self):
        current=np.array([0.,.1,-.2,.1,0.])
        target=current+np.array([0.,.2,-.3,.1,0.])
        nxt=synchronized_step(current,target,.035)
        self.assertAlmostEqual(sum(nxt[1:4]),sum(current[1:4]))
        self.assertLessEqual(np.max(np.abs(nxt-current)),.035+1e-12)
        self.assertGreater(abs(sum(np.clip(target-current,-.035,.035))),.01)

    def test_reward_discount_matches_ppo(self):
        for gamma in [.99,.995,.999]:
            r=Reward(gamma=gamma,control_dt=.05)
            _,parts=r.step(dt=.05,distance_before=.025,distance_after=.025,tilt=0,residual=[0]*3,previous_residual=[0]*3)
            self.assertAlmostEqual(parts['shaping'],5*(-.5*gamma+.5))

    def test_residual_does_not_move_nominal_reward_reference(self):
        a,b=BowlEnv(),BowlEnv();a.reset(seed=3);b.reset(seed=3)
        for e in [a,b]:e.phase_index=1;e.phase_time=.7
        a.step([1,1,1]);b.step([-1,-1,-1])
        np.testing.assert_array_equal(a.nominal,b.nominal)
        self.assertGreater(np.linalg.norm(a.waypoint-b.waypoint),0.)

    def test_first_collision_is_preserved_and_serializable(self):
        e=BowlEnv();e.reset(seed=1)
        event={'phase':'settle','time_s':.001,'geoms':['test_finger','platform'],'force_n':1.}
        with patch.object(e,'_collision_event',return_value=event):
            _,_,done,_,info=e.step(np.zeros(3))
        self.assertTrue(done);self.assertEqual(info['failure_reason'],'unexpected_collision')
        self.assertEqual(info['first_collision'],event);json.dumps(info)

    def test_placement_residual_target_is_disabled_after_lowering(self):
        e=BowlEnv();e.reset(seed=1);e.phase_index=6;e.phase_time=.4
        e.step([1,1,1]);np.testing.assert_allclose(e.residual,0.)

    def test_fixed_clock_does_not_allow_unsupported_release(self):
        e=BowlEnv();e.reset(seed=30003);e.phase_index=6
        e.phase_time=e.cfg['task']['durations_s'][6]-.05
        original=e._evidence
        def unsupported():
            v=original();v['support_n']=0.;return v
        with patch.object(e,'_evidence',side_effect=unsupported):
            _,_,done,_,_=e.step(np.zeros(3))
        self.assertFalse(done);self.assertEqual(e.phase,'support')
        self.assertEqual(e.support_time,0.)

    def test_full_episode_json_and_stable_success(self):
        e=BowlEnv();r=episode(e,seed=30003,options={'trace':True});json.dumps(r);json.dumps(e.rows)
        self.assertTrue(r['is_success'])
        self.assertGreaterEqual(r['place_stable_s'],e.cfg['task']['place_stable_s']-1e-8)
        self.assertTrue(r['hold_verified'])

    def test_fixed_scenario_is_not_changed_by_episode_length(self):
        a,b=BowlEnv(),BowlEnv();a.reset(seed=30001);b.reset(seed=30001)
        for _ in range(7):a._observation()
        _,ia=a.reset(seed=30002);_,ib=b.reset(seed=30002)
        self.assertEqual(ia['scenario'],ib['scenario'])
        self.assertEqual(ia['parking_xy_yaw'],ib['parking_xy_yaw'])

    def test_checkpoint_selection_uses_success_not_return(self):
        a={'successes':8,'failures':{},'trials':[{'place_error_m':.02,'return':999.}]}
        b={'successes':9,'failures':{},'trials':[{'place_error_m':.03,'return':-999.}]}
        self.assertGreater(checkpoint_key(b),checkpoint_key(a))

    def test_paired_comparison_and_seed_mismatch(self):
        def result(flags):return {'trials':[{'seed':i,'parking_xy_yaw':[0,0,0],'is_success':v} for i,v in enumerate(flags)]}
        p=paired_compare(result([1,0,0,1]),result([0,1,1,1]))
        self.assertEqual(p['candidate_only_success'],2);self.assertEqual(p['reference_only_success'],1)
        self.assertEqual(p['exact_two_sided_p'],1.)
        with self.assertRaises(ValueError):paired_compare(result([1]),result([1,0]))


if __name__=='__main__':unittest.main()
