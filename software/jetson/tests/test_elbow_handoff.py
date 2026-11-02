"""At the effective wrist upper stop, the measured elbow must still lift view."""
import time
import unittest
import numpy as np
from vision.bowl_follow import BoundedArm,FollowPolicy,FollowSettings
from vision.tracking_profile import apply_tracking_profile,apply_measured_elbow_extension
from tests.test_bowl_follow import FakeArm,detection


class ElbowHandoffTests(unittest.TestCase):
    def extension_fixture(self):
        profile=dict(tracking_axes=[0,2,3],axes={'2':dict(min_p=1679,max_p=1859)},
                     pose_reference={'2':1740},operator_target_lock={'weights_sha256':'a'*64})
        report=dict(result='MEASURED_EXTENSION_PASS',motion_enabled=True,finished_unix=100,
            target={'weights_sha256':'a'*64},minimum_command_p=1600,minimum_feedback_p=1643,
            steps=[dict(before_p=1659,target_p=1600,after_p=1643,before_center=[160,76],
                        after_center=[160,96],image_response_px_per_p=1.25)])
        return profile,report

    def test_extension_only_applies_actually_exercised_absolute_target(self):
        profile,report=self.extension_fixture()
        updated=apply_measured_elbow_extension(profile,report,now=101)
        self.assertEqual(updated['axes']['2'],dict(min_p=1600,max_p=1859))
        self.assertEqual(profile['axes']['2']['min_p'],1679)
        self.assertEqual(updated['position_windows_p']['2'],140)

    def test_failed_stale_or_inconsistent_range_probe_never_expands_limits(self):
        for kind in ('failed','stale','endpoint','response','model'):
            with self.subTest(kind=kind):
                profile,report=self.extension_fixture()
                if kind=='failed':report['result']='ABORTED'
                if kind=='stale':report['finished_unix']=-2000
                if kind=='endpoint':report['minimum_command_p']=1500
                if kind=='response':report['steps'][0]['image_response_px_per_p']=1.1
                if kind=='model':report['target']['weights_sha256']='b'*64
                with self.assertRaises(ValueError):apply_measured_elbow_extension(profile,report,now=101)

    def test_upward_demand_uses_loaded_elbow_when_wrist_cannot_advance(self):
        positions={0:1528,1:1331,2:1769,3:1036,4:1473,5:1508}
        bias={'0':dict(positive=16,negative=15),'2':dict(positive=-12,negative=45),
              '3':dict(positive=40,negative=-6)}
        config=dict(camera_rigid_on_arm=True,directions_and_limits_verified=True,
            serial_open_reset_resolved=True,perception_validated=True,port='/test/not-a-real-port',
            pose_reference={str(a):p for a,p in positions.items()},tracking_axes=[0,2,3],
            axes={'0':dict(min_p=1288,max_p=1768,direction=-1),
                  '2':dict(min_p=1679,max_p=1859,direction=-1),
                  '3':dict(min_p=806,max_p=1076,direction=-1)},
            feedback_deadband_p=bias,session_travel_p=12000,session_window_p=90,
            local_calibration={'axis_amplitudes':{'2':50}})
        config=apply_tracking_profile(config)
        transport=FakeArm();transport.positions=positions.copy()
        controller=BoundedArm(transport,config,time.monotonic())
        low,high=controller.displacement_bounds()
        self.assertEqual(high[2],0)
        self.assertEqual(low[1],-20)
        matrix=np.array([[1.6,-.14,0],[-.01,-1.2,.9]])
        policy=FollowPolicy(FollowSettings(jacobian=matrix,joint_axes=(0,2,3),
            bounded_allocation=True,adaptive_gain=True,stable_frames=2,max_step=100,
            displacement_bounds=(low,high),joint_weights=[1,8,1],handoff_weights=[1,2,1],
            deadband_compensation=((16,15),(-12,45),(40,-6)),
            command_step_limits=config['command_step_limits_p']))
        start=time.time()
        for i in range(2):out=policy.update(detection(start+i*.05,(160,55)),start+i*.05,i*.05)
        self.assertEqual(out['state'],'move')
        self.assertNotIn(3,out['delta_p'])
        self.assertLessEqual(out['delta_p'][2],-48)
        elbow_actual=out['delta_p'][2]+45
        self.assertLess(elbow_actual,0)
        self.assertGreater((matrix[:,1]*elbow_actual)[1],0)
