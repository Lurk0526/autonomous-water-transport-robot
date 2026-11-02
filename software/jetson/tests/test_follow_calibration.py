"""Local direction measurements and bounded visual-control planning."""
import copy
import time
import unittest

import numpy as np
from vision.follow_calibration import target_measurement, stable_summary, estimate_jacobian, recent_target_samples, measurement_anchor
from vision.bowl_follow import FollowPolicy, FollowSettings, validate_motion_config
from vision.follow_calibration import segmented_move
from tests.test_bowl_follow import detection, calibrated_fixture


def trials():
    result=[]
    for axis,column in [(0,[-.45,.03]),(3,[.02,.5])]:
        for dp in (22,-22):
            result.append({'axis':axis,'actual_delta_p':dp,
                           'before':{'center':[160,120]},
                           'after':{'center':(np.array([160,120])+np.array(column)*dp).tolist()}})
    return result


class CalibrationTests(unittest.TestCase):
    def test_loaded_return_replans_each_step_from_actual_position(self):
        position=[2414];commands=[]
        def move(goal,final):
            self.assertLessEqual(abs(goal-position[0]),65)
            commands.append(goal);position[0]=goal+43
            return position[0]
        actual=segmented_move(lambda:position[0],move,2292)
        self.assertGreater(len(commands),2)
        self.assertEqual(actual,2335)

    def test_unresponsive_return_does_not_repeat_unbounded_commands(self):
        calls=[]
        def no_motion(goal,final):calls.append(goal);return 2400
        with self.assertRaisesRegex(RuntimeError,'no measured progress'):
            segmented_move(lambda:2400,no_motion,2300)
        self.assertEqual(len(calls),1)

    def test_resume_preserves_original_anchor_instead_of_interrupted_pose(self):
        original={str(a):1500 for a in range(6)};original['3']=925
        restart=dict(original);restart['3']=902
        report={'anchor':original,'initial_pose':restart}
        chosen=measurement_anchor(report)
        self.assertEqual(chosen['3'],925)
        chosen['3']=880
        self.assertEqual(original['3'],925)

    def test_legacy_initial_pose_remains_supported_but_incomplete_pose_fails(self):
        pose={str(a):1500 for a in range(6)}
        self.assertEqual(measurement_anchor({'initial_pose':pose}),pose)
        with self.assertRaises(ValueError):measurement_anchor({'initial_pose':{'0':1500}})
    def test_slower_frames_keep_recent_seven_instead_of_timing_out_on_old_ninth(self):
        samples=[dict(stamp=i*.1,center=[160,120],area=3000,confidence=.95,bbox=[125,95,195,145])
                 for i in range(15)]
        recent=recent_target_samples(samples)
        self.assertGreaterEqual(len(recent),7)
        self.assertLessEqual(recent[-1]['stamp']-recent[0]['stamp'],.7)
        self.assertGreaterEqual(stable_summary(recent)['frames'],7)
    def test_fresh_complete_target_is_required(self):
        now=time.time();record=detection(now, (160,120))
        self.assertIsNotNone(target_measurement(record,now+.05))
        self.assertIsNone(target_measurement(record,now+1))
        record['detections'][0]['bbox_xyxy_px'][0]=0
        self.assertIsNone(target_measurement(record,now+.05))

    def test_stability_rejects_moving_target(self):
        now=time.time()
        samples=[target_measurement(detection(now+i*.05,(160+i*.1,120)),now+i*.05+.02) for i in range(7)]
        self.assertEqual(stable_summary(samples)['frames'],7)
        samples[-1]['center']=[180,120]
        with self.assertRaises(ValueError):stable_summary(samples)

    def test_both_directions_measure_actual_image_servo_response(self):
        matrix,condition=estimate_jacobian(trials())
        np.testing.assert_allclose(matrix,[[-.45,.02],[.03,.5]],atol=1e-9)
        self.assertLess(condition,2)
        with self.assertRaises(ValueError):estimate_jacobian(trials()[:2])
        conflicting=trials();conflicting[1]['after']['center']=[150,120]
        with self.assertRaises(ValueError):estimate_jacobian(conflicting)

    def test_measured_jacobian_drives_the_correct_coupled_direction(self):
        matrix,_=estimate_jacobian(trials())
        policy=FollowPolicy(FollowSettings(jacobian=matrix,max_step=20,min_step=12,deadband_compensation=(8,8)))
        now=time.time()
        for i in range(5):
            out=policy.update(detection(now+i*.05,(220,170)),now+i*.05+.02,i*.05)
        self.assertEqual(out['state'],'move')
        self.assertGreater(out['delta_p'][0],0)
        self.assertLess(out['delta_p'][3],0)
        self.assertTrue(all(0<abs(v)<=20 for v in out['delta_p'].values()))

    def test_trial_requires_fresh_target_and_recomputable_calibration(self):
        matrix,_=estimate_jacobian(trials())
        config=calibrated_fixture();config['perception_validated']=False
        config.update(supervised_trial=True,max_step_p=20,min_step_p=12,
                      local_calibration={'result':'LOCAL_JACOBIAN_PASS','jacobian':matrix,'trials':trials()},
                      operator_target_lock={'center':[160,120],'stamp_unix':time.time(),
                      'weights_sha256':'c402769dabe8f46fd69e01f76054622b15677862fd34e1825618fedc4b9073a6'})
        validate_motion_config(config)
        stale=copy.deepcopy(config);stale['operator_target_lock']['stamp_unix']-=120
        with self.assertRaises(ValueError):validate_motion_config(stale)
        fabricated=copy.deepcopy(config);fabricated['local_calibration']['trials']=[]
        with self.assertRaises(ValueError):validate_motion_config(fabricated)
        mismatch=copy.deepcopy(config);mismatch['local_calibration']['jacobian']=[[.5,0],[0,.5]]
        with self.assertRaises(ValueError):validate_motion_config(mismatch)


if __name__=='__main__':unittest.main()
