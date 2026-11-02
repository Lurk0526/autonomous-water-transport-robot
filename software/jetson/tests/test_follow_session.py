"""Explicit resume validates fresh target and preserves verified motion limits."""
import copy
import time
import unittest
from vision.follow_session import PoseCalibrationRequired, resume_config
from tests.test_bowl_follow import calibrated_fixture
from vision.tracking_profile import apply_tracking_profile


class ResumeTests(unittest.TestCase):
    def test_selected_model_does_not_bypass_absolute_start_range(self):
        old,positions,target=self.fixture()
        old.update(tracking_axes=[0,2,3],startup_pose_policy='measure_current_pose')
        old['axes']['2']=dict(min_p=1437,max_p=2447,direction=-1)
        old=apply_tracking_profile(old)
        positions.update({2:1229,3:590})
        with self.assertRaisesRegex(ValueError,'002.*1229.*003.*590'):
            resume_config(old,positions,target,time.time(),accept_current_pose=True)

    def fixture(self):
        old=calibrated_fixture()
        old['operator_target_lock']={'weights_sha256':'verified-model','stamp_unix':1}
        target=dict(center=[160,120],bbox=[100,80,220,160],min_confidence=.95,
                    weights_sha256='verified-model',last_stamp=time.time())
        return old,{a:1500 for a in range(6)},target

    def test_resume_uses_current_positions_without_expanding_absolute_limits(self):
        old,positions,target=self.fixture();before=copy.deepcopy(old)
        positions[0]=1520
        new=resume_config(old,positions,target,time.time())
        self.assertEqual(new['pose_reference']['0'],1520)
        self.assertEqual(new['axes'],old['axes'])
        self.assertEqual(new['operator_target_lock']['center'],[160,120])
        self.assertEqual(old,before)

    def test_unsafe_or_unverified_session_cannot_be_enabled(self):
        for kind in ('outside','wrong_model','stale','no_feedback','invalid_profile'):
            with self.subTest(kind=kind):
                old,positions,target=self.fixture()
                if kind=='outside':positions[0]=1600
                if kind=='wrong_model':target['weights_sha256']='other-model'
                if kind=='stale':target['last_stamp']-=10
                if kind=='no_feedback':positions.pop(5)
                if kind=='invalid_profile':old['directions_and_limits_verified']=False
                with self.assertRaises(ValueError):resume_config(old,positions,target,time.time())

    def test_changed_camera_pose_requests_automatic_measurement(self):
        for axis in (1,2,4):
            old,positions,target=self.fixture()
            positions[axis]+=100
            with self.assertRaises(PoseCalibrationRequired):
                resume_config(old,positions,target,time.time())

    def test_new_gripper_opening_is_held_at_its_current_position(self):
        old,positions,target=self.fixture();positions[5]=1700
        new=resume_config(old,positions,target,time.time())
        self.assertEqual(new['pose_reference']['5'],1700)
        self.assertEqual(new['axes'],old['axes'])

    def test_wide_travel_does_not_silently_reuse_a_distant_wrist_fit(self):
        old,positions,target=self.fixture()
        old.update(tracking_axes=[0,2,3],startup_pose_policy='measure_current_pose')
        old['axes']['2']=dict(min_p=1460,max_p=1540,direction=-1)
        old=apply_tracking_profile(old);positions[3]=1700
        with self.assertRaises(PoseCalibrationRequired):
            resume_config(old,positions,target,time.time())
