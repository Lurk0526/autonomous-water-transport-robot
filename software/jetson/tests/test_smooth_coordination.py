"""Early pitch handoff and loaded-axis streaming preserve stopping guards."""
import time
import unittest
from unittest.mock import patch
from vision.bowl_follow import BoundedArm,FollowPolicy,FollowSettings
from tests.test_bowl_follow import FakeArm,calibrated_fixture,detection


class SmoothTests(unittest.TestCase):
    def test_observed_loaded_rebound_is_rebased_but_larger_motion_still_faults(self):
        class Transport(FakeArm):
            def move(self,targets,duration_ms,allowed_axes=None):super().move(targets,duration_ms)
        for actual,allowed in ((1841,True),(1860,False)):
            config=calibrated_fixture();config.update(tracking_axes=[0,2,3],max_step_p=65,
                allow_settling_drift=True,bounded_settling_feedback=True,bounded_stop_feedback=True,
                adaptive_bias_tolerance_p=25,settling_allowance_p={'2':25},session_window_p=90,
                session_travel_p=1000)
            config['pose_reference']['2']=1827
            config['axes']['2']=dict(min_p=1600,max_p=2000,direction=-1)
            transport=Transport();transport.positions[2]=1827
            with patch('vision.bowl_follow.time.monotonic',return_value=10):
                arm=BoundedArm(transport,config,10)
                self.assertEqual(arm.tick(dict(state='move',delta_p={2:-55},source_stamp_unix=time.time()),10),'command_sent')
                self.assertEqual(arm.tick(dict(state='centered',delta_p={}),10.1),'centered')
            transport.positions[2]=actual
            with patch('vision.bowl_follow.time.monotonic',return_value=20):
                result=arm.tick(dict(state='move',delta_p={2:-10},source_stamp_unix=time.time()),20)
            self.assertEqual(result=='command_sent',allowed)

    def test_elbow_compensation_does_not_consume_its_useful_negative_motion(self):
        settings=FollowSettings(jacobian=[[2,0,0],[0,-1.1,1.15]],joint_axes=(0,2,3),
            bounded_allocation=True,stable_frames=2,max_step=100,joint_weights=[1,1.35,1],
            smooth_coordination=True,adaptive_gain=True,gain_near=.55,gain_far=.92,
            deadband_compensation=((15,13),(-7,45),(25,-11)),
            command_step_limits={'0':100,'2':65,'3':100},negative_step_limits={'2':100},
            displacement_bounds=([-80,-55,-100],[85,65,75]),period=.08)
        policy=FollowPolicy(settings);start=time.time()
        for i in range(2):plan=policy.update(detection(start+i*.1,(160,35)),start+i*.1,i*.1)
        self.assertLess(plan['delta_p'].get(2,0),-65)
        self.assertLessEqual(abs(plan['delta_p'][2]+45),65)
        self.assertGreater(plan['delta_p'].get(3,0),25)

    def test_larger_compensation_command_cannot_exceed_actual_motion_cap(self):
        config=calibrated_fixture()
        config.update(tracking_axes=[0,2,3],max_step_p=100,
            command_step_limits_p={'2':65},negative_command_step_limits_p={'2':100},
            actual_step_limits_p={'2':65})
        config['axes']['2']=dict(min_p=1300,max_p=1700,direction=-1)
        transport=FakeArm();now=time.monotonic();arm=BoundedArm(transport,config,now)
        result=arm.tick(dict(state='move',delta_p={2:-100},source_stamp_unix=time.time()),now)
        self.assertIn('actual_step_limit',result);self.assertEqual(transport.commands,[])

    def test_elbow_joins_before_the_wrist_exhausts_its_remaining_travel(self):
        settings=FollowSettings(jacobian=[[2,0,0],[0,-1.1,1.15]],joint_axes=(0,2,3),
            bounded_allocation=True,stable_frames=2,max_step=100,joint_weights=[1,8,1],
            handoff_weights=[1,2,1],smooth_coordination=True,
            deadband_compensation=((15,13),(-7,42),(25,-11)),period=.08)
        policy=FollowPolicy(settings);start=time.time()
        for i,headroom in enumerate((75,75,60,45,30)):
            settings.displacement_bounds=([-80,-23,-100],[85,65,headroom])
            plan=policy.update(detection(start+i*.1,(160,50)),start+i*.1,i*.1)
        self.assertLess(plan['delta_p'].get(2,0),-42)
        self.assertGreater(plan['delta_p'].get(3,0),25)
        stop=policy.update(detection(start+.5,(160,120)),start+.5,.5)
        self.assertFalse(stop['delta_p'])
        stop=policy.update(detection(start+.6,(160,120)),start+.6,.6)
        self.assertEqual(stop['state'],'centered');self.assertIsNone(policy.previous_demand)

    def test_loaded_streaming_requires_real_progress_and_can_stop_immediately(self):
        class Transport(FakeArm):
            def move(self,targets,duration_ms,allowed_axes=None):
                self.allowed=allowed_axes
                super().move(targets,duration_ms)
        config=calibrated_fixture()
        config.update(tracking_axes=[0,2,3],max_step_p=100,continuous_follow=True,
                      coordinated_retarget=True,retarget_period_s=.22,duration_ms=500,
                      session_travel_p=12000,position_windows_p={'0':200,'2':200,'3':200})
        config['axes']['2']=dict(min_p=1300,max_p=1700,direction=-1)
        for entry in config['axes'].values():entry.update(min_p=1300,max_p=1700)
        now=time.monotonic();transport=Transport();arm=BoundedArm(transport,config,now)
        plan=dict(state='move',delta_p={2:60},source_stamp_unix=time.time(),center_px=(160,60),target_px=(160,120))
        self.assertEqual(arm.tick(plan,now),'command_sent')
        self.assertEqual(arm.tick(plan,now+.25),'moving')
        transport.positions[2]=1520
        self.assertEqual(arm.tick(plan,now+.45),'command_retargeted')
        self.assertEqual(transport.allowed,(0,2,3))
        self.assertEqual(arm.tick(dict(state='no_target',delta_p={}),now+.5),'no_target')
        self.assertIsNone(arm.pending)
