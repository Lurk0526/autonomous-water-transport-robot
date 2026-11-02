"""Changed starts measure a fresh response without reusing old pose geometry."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from tools import prepare_follow_trial as preparation
from tests.test_bowl_follow import calibrated_fixture
from tests.test_follow_calibration import trials
from vision.follow_calibration import estimate_jacobian
from vision.bowl_follow import BoundedArm,FollowPolicy,FollowSettings,validate_motion_config
from vision.tracking_profile import apply_tracking_profile
from tests.test_bowl_follow import FakeArm,detection
from tests.test_pose_response import measured


class CurrentPoseTests(unittest.TestCase):
    def test_outside_verified_range_reports_all_joints_before_waiting_for_a_bowl(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'work/bowl-follow/current-trial.json'
            path.parent.mkdir(parents=True)
            old=calibrated_fixture()
            old.update(tracking_axes=[0,2,3],startup_pose_policy='measure_current_pose')
            old['axes']['2']=dict(min_p=1437,max_p=2447,direction=-1)
            path.write_text(json.dumps(old));original=path.read_bytes()
            arm=Mock();arm.positions.return_value={0:1547,1:1366,2:1229,3:590,4:1495,5:1533}
            feed=Mock()
            with patch.object(preparation,'ROOT',root),patch.object(preparation,'rclpy'), \
                 patch.object(preparation,'ArmSerial',return_value=arm), \
                 patch.object(preparation,'TargetFeed',return_value=feed), \
                 patch.object(preparation,'prepare_current_pose') as probe:
                with self.assertRaisesRegex(ValueError,'002.*1229.*1437.*003.*590.*650'):
                    preparation.prepare_resume()
            feed.current.assert_not_called();probe.assert_not_called()
            arm.move.assert_not_called();arm.close.assert_called_once()
            feed.node.destroy_node.assert_called_once()
            self.assertEqual(path.read_bytes(),original)

    def test_clean_work_directory_bootstraps_v1_without_a_saved_start_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config').mkdir()
            preset=calibrated_fixture();preset.pop('pose_reference')
            preset['operator_target_lock']={'weights_sha256':'verified-model'}
            (root/'config/bowl-follow.v1.json').write_text(json.dumps(preset))
            model=measured({str(a):1500 for a in range(6)})
            (root/'config/pose-response-library.v1.json').write_text(json.dumps(
                {'models':[{'source':'v1-saved-model','calibration':model}]}))
            arm=Mock();arm.positions.return_value={a:1500 for a in range(6)}
            arm.positions.return_value.update({1:1300,2:2000,3:1100})
            feed=Mock();feed.current.return_value=dict(center=[160,120],bbox=[100,80,220,160],
                min_confidence=.95,last_stamp=time.time(),weights_sha256='verified-model')
            with patch.object(preparation,'ROOT',root),patch.object(preparation,'rclpy'), \
                 patch.object(preparation,'ArmSerial',return_value=arm), \
                 patch.object(preparation,'TargetFeed',return_value=feed), \
                 patch.object(preparation,'prepare_current_pose') as probe:
                preparation.prepare_resume()
            probe.assert_not_called();arm.move.assert_not_called()
            saved=json.loads((root/'work/bowl-follow/current-trial.json').read_text())
            self.assertEqual(saved['profile_version'],'1.0.0')
            self.assertEqual(saved['pose_reference']['1'],1300)
            self.assertEqual(saved['pose_reference']['2'],2000)

    def test_changed_pose_restarts_from_saved_model_without_motion_probing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'work/bowl-follow/current-trial.json';path.parent.mkdir(parents=True)
            old=calibrated_fixture();old.update(tracking_axes=[0,2,3],
                local_calibration=measured({str(a):1500 for a in range(6)}),
                operator_target_lock={'weights_sha256':'verified-model'})
            path.write_text(json.dumps(old))
            arm=Mock();arm.positions.return_value={a:1500 for a in range(6)}
            arm.positions.return_value.update({1:1300,2:2050})
            feed=Mock();feed.current.return_value=dict(center=[160,120],bbox=[100,80,220,160],
                min_confidence=.95,last_stamp=time.time(),weights_sha256='verified-model')
            with patch.object(preparation,'ROOT',root),patch.object(preparation,'rclpy'), \
                 patch.object(preparation,'ArmSerial',return_value=arm), \
                 patch.object(preparation,'TargetFeed',return_value=feed), \
                 patch.object(preparation,'prepare_current_pose') as probe:
                preparation.prepare_resume()
            probe.assert_not_called();arm.move.assert_not_called()
            saved=json.loads(path.read_text())
            self.assertEqual(saved['pose_reference']['1'],1300)
            self.assertEqual(saved['pose_reference']['2'],2050)
            self.assertEqual(saved['pose_response_selection']['source'],'current-profile')
    def test_invalid_bounds_are_rejected_before_reference_arithmetic(self):
        for value in (None,'bad'):
            config=calibrated_fixture();config['axes']['0']['min_p']=value
            with self.assertRaises(ValueError):validate_motion_config(config)

    def test_feedback_at_command_boundary_can_start_inward_without_expanding_commands(self):
        config=calibrated_fixture()
        config.update(tracking_axes=[0,2,3],startup_pose_policy='measure_current_pose',
                      local_calibration={'axis_amplitudes':{'2':60}},feedback_tolerance_p=10)
        config['pose_reference']['2']=2449
        config['axes']['2']=dict(min_p=1600,max_p=2447,direction=-1)
        config=apply_tracking_profile(config)
        validate_motion_config(config)
        transport=FakeArm();transport.positions={int(a):p for a,p in config['pose_reference'].items()}
        arm=BoundedArm(transport,config,time.monotonic())
        self.assertEqual(arm.displacement_bounds()[1][1],0)
        self.assertLess(arm.displacement_bounds()[0][1],0)
        self.assertEqual(config['axes']['2']['max_p'],2447)
        config['pose_reference']['2']=2460
        with self.assertRaises(ValueError):validate_motion_config(config)

    def test_delayed_settling_must_stay_inside_last_commanded_trajectory(self):
        for actual,allowed in ((1504,True),(1510,False)):
            with self.subTest(actual=actual):
                config=calibrated_fixture()
                config.update(max_step_p=30,allow_settling_drift=True,adaptive_bias_tolerance_p=25,
                              bounded_settling_feedback=True,fresh_frame_after_feedback=True)
                transport=FakeArm()
                with patch('vision.bowl_follow.time.monotonic',return_value=10):
                    arm=BoundedArm(transport,config,10)
                    arm.tick(dict(state='move',delta_p={0:-10},source_stamp_unix=time.time()),10)
                transport.positions[0]=1490
                with patch('vision.bowl_follow.time.monotonic',return_value=11):
                    arm.tick(dict(state='rate_limited',delta_p={}),11)
                transport.positions[0]=actual
                with patch('vision.bowl_follow.time.monotonic',return_value=20):
                    result=arm.tick(dict(state='move',delta_p={0:-3},source_stamp_unix=time.time()),20)
                self.assertEqual(result=='command_sent',allowed)

    def test_elbow_remains_available_at_the_observed_wrist_upper_stop(self):
        config=calibrated_fixture()
        config.update(tracking_axes=[0,2,3],startup_pose_policy='measure_current_pose',
                      local_calibration={'axis_amplitudes':{'2':60}})
        config['pose_reference'].update({'2':2395,'3':1775})
        config['axes']['2']=dict(min_p=2305,max_p=2485,direction=-1)
        config=apply_tracking_profile(config)
        config['feedback_deadband_p']={'0':dict(positive=16,negative=11),
            '2':dict(positive=-9,negative=46),'3':dict(positive=28,negative=-4)}
        transport=FakeArm();transport.positions={int(a):p for a,p in config['pose_reference'].items()}
        arm=BoundedArm(transport,config,time.monotonic());arm.positions[2]=2343
        low,high=arm.displacement_bounds()
        self.assertEqual(high[2],0)
        self.assertLessEqual(low[1],-19)
        policy=FollowPolicy(FollowSettings(jacobian=[[1.638,-.777,.097],[-.111,-.331,.739]],
            joint_axes=(0,2,3),bounded_allocation=True,stable_frames=2,max_step=100,
            displacement_bounds=(low,high),joint_weights=[1,8,1],
            command_step_limits=config['command_step_limits_p'],
            deadband_compensation=((16,11),(-9,46),(28,-4))))
        now=time.time()
        for i in range(2):plan=policy.update(detection(now+i*.05,(162,80)),now+i*.05,i*.05)
        self.assertLess(plan['delta_p'].get(2,0),-46)

    def test_changed_shoulder_releases_old_connection_before_auto_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'work/bowl-follow/current-trial.json'
            path.parent.mkdir(parents=True)
            old=calibrated_fixture();old['operator_target_lock']={'weights_sha256':'verified-model'}
            path.write_text(json.dumps(old));original=path.read_bytes()
            arm=Mock();arm.positions.return_value={a:1500 for a in range(6)}
            arm.positions.return_value[1]=1600
            feed=Mock();feed.current.return_value=dict(center=[160,120],bbox=[100,80,220,160],
                min_confidence=.95,last_stamp=time.time(),weights_sha256='verified-model')
            events=[];arm.close.side_effect=lambda:events.append('close')
            with patch.object(preparation,'ROOT',root), \
                 patch.object(preparation,'apply_tracking_profile',side_effect=lambda r:r), \
                 patch.object(preparation,'ArmSerial',return_value=arm), \
                 patch.object(preparation,'TargetFeed',return_value=feed), \
                 patch.object(preparation,'rclpy'), \
                 patch.object(preparation,'prepare_current_pose',side_effect=lambda:events.append('measure')):
                preparation.prepare_resume()
            self.assertEqual(events,['close','measure'])
            self.assertEqual(path.read_bytes(),original)
            arm.move.assert_not_called()
            feed.node.destroy_node.assert_called_once()

    def test_fresh_measurement_includes_elbow_and_failure_keeps_profile(self):
        with patch.object(preparation.subprocess,'run',return_value=Mock(returncode=1)) as run, \
             patch.object(preparation,'prepare_from_calibration') as build:
            with self.assertRaisesRegex(RuntimeError,'原跟随配置已保留'):
                preparation.prepare_current_pose()
        command=run.call_args.args[0]
        self.assertEqual(command[command.index('--axes')+1:command.index('--amplitude-p')],['0','2','3'])
        build.assert_not_called()

    def test_new_measured_profile_uses_arbitrary_held_pose_without_old_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'work/bowl-follow';folder.mkdir(parents=True)
            positions={0:1510,1:1491,2:2050,3:1490,4:1492,5:1541}
            samples=trials()
            for dp in (22,-22):
                samples.append(dict(axis=2,actual_delta_p=dp,before={'center':[160,120]},
                                    after={'center':[160-.1*dp,120-1.1*dp]}))
            matrix,condition=estimate_jacobian(samples,(0,2,3))
            for row in samples:
                row['actual_after_p']=positions[row['axis']]+row['actual_delta_p']
                row['target_p']=row['actual_after_p']
            calibration=dict(result='LOCAL_JACOBIAN_PASS',events=[{'unix':time.time()}],
                anchor={str(a):p for a,p in positions.items()},
                final_pose={str(a):p for a,p in positions.items()},
                axes=[0,2,3],axis_amplitudes={'0':30,'2':60,'3':60},amplitude_p=60,
                trials=samples,jacobian=matrix,condition=condition)
            (folder/'last-calibration.json').write_text(json.dumps(calibration))
            arm=Mock();arm.positions.return_value=positions
            feed=Mock();feed.current.return_value=dict(center=[160,120],bbox=[100,80,220,160],
                min_confidence=.95,last_stamp=time.time())
            with patch.object(preparation,'ROOT',root), \
                 patch.object(preparation,'ArmSerial',return_value=arm), \
                 patch.object(preparation,'TargetFeed',return_value=feed), \
                 patch.object(preparation,'rclpy'):
                preparation.prepare_from_calibration()
            config=json.loads((folder/'current-trial.json').read_text())
            self.assertEqual(config['pose_reference'],calibration['anchor'])
            self.assertEqual(config['tracking_axes'],[0,2,3])
            self.assertEqual(config['joint_weights'],[1,1.35,1])
            self.assertEqual(config['axes']['2']['min_p'],1437)
            self.assertEqual(config['axes']['2']['max_p'],2447)
            self.assertEqual(config['command_step_limits_p']['2'],65)
            self.assertLess(config['axes']['3']['min_p'],1490)
            self.assertGreater(config['axes']['3']['max_p'],1490)
            self.assertEqual(config['startup_pose_policy'],'measure_current_pose')

    def test_observed_upward_stop_is_not_recreated_by_a_local_fit_window(self):
        config=calibrated_fixture()
        config.update(tracking_axes=[0,3],startup_pose_policy='measure_current_pose')
        config['pose_reference']['3']=1493
        config=apply_tracking_profile(config)
        config['feedback_deadband_p']={'0':dict(positive=11,negative=12),
                                     '3':dict(positive=28,negative=-1)}
        transport=FakeArm();transport.positions={int(a):p for a,p in config['pose_reference'].items()}
        arm=BoundedArm(transport,config,time.monotonic());arm.positions[3]=1552
        self.assertEqual(config['axes']['3']['max_p'],1800)
        self.assertGreater(arm.displacement_bounds()[1][1],30)
        settings=FollowSettings(jacobian=[[1.77,.035],[-.005,1.197]],joint_axes=(0,3),
            bounded_allocation=True,stable_frames=2,max_step=100,
            displacement_bounds=arm.displacement_bounds(),
            deadband_compensation=((11,12),(28,-1)))
        policy=FollowPolicy(settings);now=time.time()
        for i in range(2):
            plan=policy.update(detection(now+i*.05,(162,80)),now+i*.05,i*.05)
        self.assertGreater(plan['delta_p'].get(3,0),28)

    def test_full_validated_envelope_does_not_accept_protocol_wide_limits(self):
        config=calibrated_fixture()
        config.update(tracking_axes=[0,3],startup_pose_policy='measure_current_pose')
        config['pose_reference']['3']=700
        config=apply_tracking_profile(config)
        validate_motion_config(config)
        self.assertEqual(config['position_windows_p']['3'],1100)
        config['axes']['3']['max_p']=2500
        with self.assertRaises(ValueError):validate_motion_config(config)


if __name__=='__main__':
    unittest.main()
