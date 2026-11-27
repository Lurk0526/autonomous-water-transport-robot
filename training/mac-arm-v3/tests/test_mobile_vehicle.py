"""Receiving-Mac unit checks for the forward-only kinematic vehicle surrogate.

These do not establish real steering, tyre dynamics, odometry or controller gains.
"""
import copy
import math
import platform
import unittest

import numpy as np


MAC=platform.system()=='Darwin' and platform.machine() in ('arm64','aarch64')


@unittest.skipUnless(MAC,'Execute vehicle checks on the receiving Apple Silicon Mac')
class VehicleTests(unittest.TestCase):
    def setUp(self):
        import yaml
        from arm_training.model import ROOT
        from arm_training.mobile_vehicle import VehicleState,advance,PathFollower
        self.State,self.advance,self.Follower=VehicleState,advance,PathFollower
        self.cfg=yaml.safe_load((ROOT/'config/mobile.yaml').read_text())

    def straight_path(self,length=1.,spacing=.02,yaw=0.):
        return {'samples':[{'x_m':float(s),'y_m':0.,'yaw_rad':yaw,'s_m':float(s)}
            for s in np.linspace(0,length,int(round(length/spacing))+1)]}

    def test_stationary_vehicle_never_spins_from_steering(self):
        state=self.State(x_m=.2,y_m=-.3,yaw_rad=.4)
        samples=self.advance(state,0.,self.cfg['vehicle']['max_steer_rad'],.5,self.cfg)
        np.testing.assert_allclose(state.pose,[.2,-.3,.4],atol=1e-12)
        self.assertEqual(state.speed_m_s,0.);self.assertEqual(state.yaw_rate_rad_s,0.)
        self.assertTrue(all(np.allclose(p,[.2,-.3,.4]) for p in samples))

    def test_substeps_obey_speed_acceleration_and_steering_bounds(self):
        state=self.State();dt=.053;before=np.array(state.pose)
        samples=self.advance(state,10.,10.,dt,self.cfg);h=dt/len(samples)
        self.assertEqual(len(samples),math.ceil(dt/.01))
        self.assertLessEqual(h,.01);self.assertLessEqual(state.speed_m_s,self.cfg['drive']['max_speed_m_s'])
        self.assertLessEqual(state.speed_m_s,self.cfg['drive']['max_acceleration_m_s2']*dt+1e-12)
        self.assertLessEqual(state.steering_rad,self.cfg['drive']['steering_rate_rad_s']*dt+1e-12)
        self.assertLessEqual(state.steering_rad,self.cfg['vehicle']['max_steer_rad'])
        for point in samples:
            point=np.array(point);distance=np.linalg.norm(point[:2]-before[:2])
            self.assertLessEqual(distance,self.cfg['drive']['max_speed_m_s']*h+1e-12)
            before=point

    def test_zero_dt_does_not_modify_any_state(self):
        state=self.State(x_m=.1,speed_m_s=.02,steering_rad=.1);before=state.record()
        self.assertEqual(self.advance(state,.04,.2,0.,self.cfg),[]);self.assertEqual(state.record(),before)

    def test_negative_commands_cannot_request_reverse_or_excess_steer(self):
        state=self.State();self.advance(state,-2.,-10.,.2,self.cfg)
        self.assertEqual(state.speed_m_s,0.);self.assertGreaterEqual(state.steering_rad,-self.cfg['vehicle']['max_steer_rad'])
        np.testing.assert_allclose(state.pose,[0,0,0],atol=1e-12)

    def test_tiny_time_constants_do_not_overshoot_targets(self):
        cfg=copy.deepcopy(self.cfg);cfg['drive']['speed_time_constant_s']=.00001;cfg['drive']['steering_time_constant_s']=.00001
        state=self.State();self.advance(state,.001,.001,.05,cfg)
        self.assertLessEqual(state.speed_m_s,.001);self.assertGreaterEqual(state.speed_m_s,0.)
        self.assertLessEqual(state.steering_rad,.001);self.assertGreaterEqual(state.steering_rad,0.)

    def test_stop_command_monotonically_brakes_to_exact_rest(self):
        state=self.State(speed_m_s=.04,steering_rad=.2)
        previous=state.speed_m_s
        for _ in range(40):
            self.advance(state,0.,0.,.05,self.cfg)
            self.assertLessEqual(state.speed_m_s,previous+1e-12)
            self.assertGreaterEqual(state.acceleration_m_s2,-self.cfg['drive']['max_deceleration_m_s2']-1e-12)
            previous=state.speed_m_s
        self.assertEqual(state.speed_m_s,0.);self.assertEqual(state.yaw_rate_rad_s,0.)
        stopped=state.pose.copy();self.advance(state,0.,.4,.5,self.cfg)
        np.testing.assert_allclose(state.pose,stopped,atol=1e-12)

    def test_forward_motion_yaw_rate_matches_bicycle_kinematics(self):
        state=self.State(speed_m_s=.03,steering_rad=.2)
        self.advance(state,.03,.2,.1,self.cfg)
        expected=.03*math.tan(.2)/self.cfg['vehicle']['wheelbase_m']
        self.assertAlmostEqual(state.yaw_rate_rad_s,expected)
        self.assertAlmostEqual(state.yaw_rad,expected*.1)
        self.assertGreater(state.x_m,0.)

    def test_invalid_dynamics_input_is_rejected(self):
        for dt in (-.1,float('nan')):
            with self.assertRaises(ValueError):self.advance(self.State(),0.,0.,dt,self.cfg)
        with self.assertRaises(ValueError):self.advance(self.State(),float('nan'),0.,.1,self.cfg)
        with self.assertRaises(ValueError):self.advance(self.State(speed_m_s=-.1),0.,0.,.1,self.cfg)
        cfg=copy.deepcopy(self.cfg);cfg['vehicle']['wheelbase_m']=0.
        with self.assertRaises(ValueError):self.advance(self.State(),0.,0.,.1,cfg)

    def test_odometry_declares_simulation_source_and_finite_clock(self):
        row=self.State(x_m=.1).odometry(1.2)
        self.assertTrue(row['valid']);self.assertEqual(row['frame'],'odom')
        self.assertIn('simulated',row['source']);self.assertEqual(row['xy_yaw'],[.1,0,0])
        with self.assertRaises(ValueError):self.State().odometry(float('inf'))

    def test_path_follower_bounded_forward_command(self):
        follower=self.Follower(self.straight_path(),self.cfg)
        speed,steer=follower.command(self.State())
        self.assertGreater(speed,0.);self.assertLessEqual(speed,self.cfg['drive']['max_speed_m_s'])
        self.assertLessEqual(abs(steer),self.cfg['vehicle']['max_steer_rad'])

    def test_valid_terminal_pose_commands_stop(self):
        follower=self.Follower(self.straight_path(),self.cfg);follower.index=len(follower.samples)-1
        speed,_=follower.command(self.State(x_m=1.))
        self.assertEqual(speed,0.);self.assertIsNone(follower.failure_reason)

    def test_missed_terminal_heading_latches_stop_without_creeping(self):
        follower=self.Follower(self.straight_path(),self.cfg);follower.index=len(follower.samples)-1
        state=self.State(x_m=1.,yaw_rad=.3)
        self.assertEqual(follower.command(state),(0.,0.))
        self.assertEqual(follower.failure_reason,'forward_path_terminal_pose_mismatch')
        stopped=state.pose.copy()
        for _ in range(5):
            speed,steer=follower.command(state);self.assertEqual((speed,steer),(0.,0.))
            self.advance(state,speed,steer,.05,self.cfg)
        np.testing.assert_allclose(state.pose,stopped,atol=1e-12)

    def test_failed_follower_stays_failed_even_after_pose_changes(self):
        follower=self.Follower(self.straight_path(),self.cfg);follower.failure_reason='forced_failure'
        self.assertEqual(follower.command(self.State()),(0.,0.))
        self.assertEqual(follower.command(self.State(x_m=.7,y_m=.2)),(0.,0.))

    def test_empty_and_nonmonotonic_paths_rejected(self):
        with self.assertRaises(ValueError):self.Follower({'samples':[]},self.cfg)
        path=self.straight_path();path['samples'][2]['s_m']=-1.
        with self.assertRaises(ValueError):self.Follower(path,self.cfg)

    def test_progress_search_cannot_jump_entire_path(self):
        follower=self.Follower(self.straight_path(length=4.),self.cfg)
        window=self.cfg['drive']['projection_search_samples']
        follower.command(self.State(x_m=3.))
        self.assertLess(follower.index,window)


if __name__=='__main__':unittest.main()
