import copy
import unittest
import numpy as np

from arm_training.model import config,make_model
from arm_training.env import BowlEnv
from arm_training.parking import world_to_base
from arm_training.rewards import Reward


class CoreTests(unittest.TestCase):
    def test_parking_translation_and_rotation(self):
        np.testing.assert_allclose(world_to_base([1,2,.35],[1,1,np.pi/2]),[1,0,.35],atol=1e-10)

    def test_goal_changes_with_parking(self):
        env=BowlEnv();env.reset(seed=1,options={'parking':[0,0,0]});a=env.place.copy()
        env.reset(seed=1,options={'parking':[.02,0,.2]});self.assertGreater(np.linalg.norm(a-env.place),.02)

    def test_same_seed_reproducible_and_pose_changes_across_seeds(self):
        env=BowlEnv();a,ia=env.reset(seed=11);b,ib=env.reset(seed=11)
        np.testing.assert_array_equal(a,b);self.assertEqual(ia['parking_xy_yaw'],ib['parking_xy_yaw'])
        _,ic=env.reset(seed=12);self.assertNotEqual(ia['parking_xy_yaw'],ic['parking_xy_yaw'])

    def test_single_actuator_four_coupled_fingers_free_bowl(self):
        model,_,_=make_model(config(),visual=False)
        self.assertEqual(model.nu,6);self.assertEqual(model.neq,3)
        self.assertEqual(model.joint('bowl_free').type[0],0)
        self.assertEqual(model.body_mass[model.body('bowl').id],.65)

    def test_camera_support_does_not_rotate_with_004(self):
        import mujoco
        model,_,_=make_model(config(),visual=False);data=mujoco.MjData(model)
        site=model.site('camera_mount_reference').id;mujoco.mj_forward(model,data)
        before=data.site_xpos[site].copy();data.qpos[int(model.joint('tool_roll').qposadr[0])]=.5
        mujoco.mj_forward(model,data);np.testing.assert_allclose(data.site_xpos[site],before,atol=1e-10)

    def test_unreachable_has_explicit_reposition_result(self):
        env=BowlEnv();env.reset(seed=1,options={'parking':[2,0,0]})
        _,_,done,_,info=env.step(np.zeros(3));self.assertTrue(done)
        self.assertEqual(info['failure_reason'],'unreachable_reposition_required')

    def test_invalid_sensor_terminates(self):
        env=BowlEnv();env.reset(seed=1,options={'parking':[0,0,0],'invalid_sensor':True})
        _,_,done,_,info=env.step(np.zeros(3));self.assertTrue(done)
        self.assertEqual(info['failure_reason'],'invalid_target_observation')

    def test_zero_force_and_no_grip_cannot_succeed(self):
        cfg=config();cfg['sensor']['initial_error_std_m']=0;cfg['arm']['actuator_delay_s']=[0,0]
        for options in [{'no_grip':True},{'force_n':0.}]:
            env=BowlEnv(cfg);env.reset(seed=1,options={'parking':[0,0,0],**options})
            for _ in range(500):
                _,_,done,_,info=env.step(np.zeros(3))
                if done:break
            self.assertTrue(done);self.assertFalse(info['is_success'])

    def test_trial_and_hold_awards_are_one_shot(self):
        reward=Reward();kwargs=dict(dt=.05,distance_before=.01,distance_after=.01,tilt=0,residual=[0,0,0],previous_residual=[0,0,0])
        _,a=reward.step(**kwargs,trial_lift=True,hold_verified=True)
        _,b=reward.step(**kwargs,trial_lift=True,hold_verified=True)
        self.assertEqual(a['trial_lift'],10);self.assertEqual(a['hold_verified'],20)
        self.assertNotIn('trial_lift',b);self.assertNotIn('hold_verified',b)
        _,c=reward.step(**kwargs,failure=True)
        self.assertEqual(c['failure'],-200);self.assertNotIn('complete',c)
        with self.assertRaises(RuntimeError):reward.step(**kwargs)

    def test_complete_requires_verified_hold(self):
        with self.assertRaises(ValueError):Reward().step(dt=.05,distance_before=0,distance_after=0,tilt=0,residual=[0]*3,previous_residual=[0]*3,complete=True)

    def test_bounded_observation_and_nan_action_rejected(self):
        env=BowlEnv();obs,_=env.reset(seed=10);self.assertTrue(env.observation_space.contains(obs))
        with self.assertRaises(ValueError):env.step([np.nan,0,0])

if __name__=='__main__':unittest.main()
