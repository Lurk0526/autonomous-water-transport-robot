"""Reproduce the effective upper wrist limit and verify bounded upward control."""
import time
import unittest
from vision.tracking_profile import apply_tracking_profile
from vision.bowl_follow import BoundedArm,FollowPolicy,FollowSettings
from tests.test_bowl_follow import FakeArm,detection


def fixture():
    pose={'0':1654,'1':1379,'2':1460,'3':779,'4':1492,'5':1508}
    bounds={'0':(1404,1884),'1':(1235,1415),'2':(1424,1604),'3':(629,809),'4':(1408,1588)}
    offsets={'0':dict(positive=16,negative=14),'1':dict(positive=36,negative=-7),
             '2':dict(positive=-6,negative=36),'3':dict(positive=30,negative=-8),
             '4':dict(positive=9,negative=8)}
    return dict(camera_rigid_on_arm=True,directions_and_limits_verified=True,serial_open_reset_resolved=True,
        perception_validated=True,port='/test/not-a-real-port',pose_reference=pose,tracking_axes=[0,1,2,3,4],
        max_step_p=65,session_travel_p=12000,session_window_p=90,feedback_deadband_p=offsets,
        axes={a:dict(min_p=lo,max_p=hi,direction=-1) for a,(lo,hi) in bounds.items()})


def arm(config):
    transport=FakeArm();transport.positions={int(a):p for a,p in config['pose_reference'].items()}
    return BoundedArm(transport,config,time.monotonic()),transport


class UpwardTrackingTests(unittest.TestCase):
    def test_expansion_opens_effective_wrist_range_without_expanding_loaded_axes(self):
        config=fixture();old,_=arm(config);_,high=old.displacement_bounds()
        self.assertEqual(high[1],0);self.assertEqual(high[3],0)
        self.assertEqual(old.displacement_bounds()[0][2],0)
        profile=apply_tracking_profile(config);new,_=arm(profile)
        self.assertEqual(new.displacement_bounds()[1][3],70)
        self.assertEqual(profile['axes']['1'],config['axes']['1'])
        self.assertEqual(profile['axes']['2'],config['axes']['2'])
        self.assertEqual(profile['axes']['3']['min_p'],629)

    def test_bowl_above_center_gets_visible_upward_correction(self):
        config=apply_tracking_profile(fixture());controller,_=arm(config)
        offsets=config['feedback_deadband_p']
        settings=FollowSettings(jacobian=[[1.6,.02,-.09,.05,0],[-.12,.88,-1.07,1.26,0]],
            joint_axes=(0,1,2,3,4),bounded_allocation=True,adaptive_gain=True,stable_frames=2,
            displacement_bounds=controller.displacement_bounds(),max_step=85,
            deadband_compensation=tuple((offsets[str(a)]['positive'],offsets[str(a)]['negative']) for a in range(5)),
            command_step_limits=config['command_step_limits_p'],joint_weights=[1,8,8,1,1],
            gain_near=config['gain_near'],gain_far=config['gain_far'])
        policy=FollowPolicy(settings);start=time.time()
        for i in range(2):out=policy.update(detection(start+i*.05,(160,60)),start+i*.05,i*.05)
        self.assertEqual(out['state'],'move')
        self.assertGreater(out['delta_p'][3],65)
        self.assertLessEqual(out['delta_p'][3],85)

    def test_loaded_joint_command_limit_is_enforced(self):
        controller,transport=arm(apply_tracking_profile(fixture()))
        result=controller.tick(dict(state='move',delta_p={1:46},source_stamp_unix=time.time()),time.monotonic())
        self.assertIn('invalid_step',result);self.assertEqual(transport.commands,[])

    def test_minimum_servo_step_cannot_overrun_two_remaining_units(self):
        settings=FollowSettings(jacobian=[[1,0],[0,1]],joint_axes=(0,3),bounded_allocation=True,
            stable_frames=2,max_step=100,deadband_compensation=(0,30),
            displacement_bounds=([0,0],[0,2]))
        policy=FollowPolicy(settings);start=time.time()
        for i in range(2):out=policy.update(detection(start+i*.05,(160,60)),start+i*.05,i*.05)
        self.assertEqual(out['delta_p'],{});self.assertEqual(out['state'],'limited')

    def test_primary_joint_limit_relaxes_secondary_priority(self):
        settings=FollowSettings(jacobian=[[1.6,.02,-.09,.05,0],[-.12,.88,-1.07,1.26,0]],
            joint_axes=(0,1,2,3,4),bounded_allocation=True,stable_frames=2,max_step=85,
            displacement_bounds=([-40,-40,-40,0,0],[40,40,40,0,0]),
            deadband_compensation=(0,)*5,joint_weights=[1,20,20,1,1],handoff_weights=[1,2,2,1,1])
        policy=FollowPolicy(settings);start=time.time()
        for i in range(2):out=policy.update(detection(start+i*.05,(160,60)),start+i*.05,i*.05)
        self.assertTrue(out['limit_handoff']);self.assertTrue({1,2}&set(out['delta_p']))

    def test_measured_elbow_has_useful_displacement_at_wrist_limit(self):
        config=fixture()
        config['local_calibration']={'axis_amplitudes':{'2':50}}
        profile=apply_tracking_profile(config)
        self.assertEqual(profile['command_step_limits_p']['2'],65)
        controller,_=arm(profile)
        # Move the elbow away from the lower boundary to reproduce a new pose
        # with a measured available negative range.
        controller.positions[2]=1530
        low,_=controller.displacement_bounds()
        self.assertLessEqual(low[2],-20)
