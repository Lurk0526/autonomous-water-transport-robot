"""Full-range starts, current-pose adaptation and endpoint calibration; no hardware."""
import contextlib
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from tools import prepare_follow_trial as preparation
from tools import calibrate_bowl_follow as calibration
from vision.bowl_follow import BoundedArm, validate_motion_config
from vision.follow_session import resume_config
from vision.motion_limits import FULL_RANGE_POLICY, validate_current_pose
from vision.tracking_profile import apply_tracking_profile
from tests.test_bowl_follow import calibrated_fixture, FakeArm
from tests.test_pose_response import measured


def full_config():
    config=calibrated_fixture()
    config.update(tracking_axes=[0,2,3],startup_pose_policy='measure_current_pose',
                  motion_range_policy=FULL_RANGE_POLICY,
                  operator_target_lock={'weights_sha256':'verified-model'})
    return apply_tracking_profile(config)


def target():
    return dict(center=[160,120],bbox=[100,80,220,160],min_confidence=.95,
                weights_sha256='verified-model',last_stamp=time.time())


class FullRangeStartTests(unittest.TestCase):
    def test_all_six_joint_endpoints_start_at_measured_positions_without_homing(self):
        for axis in range(6):
            for endpoint in (500,2500):
                with self.subTest(axis=axis,endpoint=endpoint):
                    config=full_config();positions={a:1500 for a in range(6)}
                    positions[axis]=endpoint
                    updated=resume_config(config,positions,target(),time.time(),accept_current_pose=True)
                    transport=FakeArm();transport.positions=dict(positions)
                    arm=BoundedArm(transport,updated,time.monotonic())
                    self.assertEqual(arm.positions,positions)
                    self.assertEqual(transport.commands,[])
                    for active in (0,2,3):
                        self.assertEqual(updated['axes'][str(active)]['min_p'],500)
                        self.assertEqual(updated['axes'][str(active)]['max_p'],2500)

    def test_reported_elbow_and_wrist_pose_is_accepted_but_protocol_overrun_is_rejected(self):
        config=full_config();positions={0:1547,1:1366,2:1229,3:590,4:1495,5:1533}
        validate_current_pose(positions,[0,2,3],config=config)
        updated=resume_config(config,positions,target(),time.time(),accept_current_pose=True)
        validate_motion_config(updated)
        for axis in range(6):
            for outside in (499,2501):
                changed=dict(positions);changed[axis]=outside
                with self.assertRaises(ValueError):
                    resume_config(config,changed,target(),time.time(),accept_current_pose=True)

    def test_endpoint_planning_only_has_inward_headroom(self):
        config=full_config();config['pose_reference'].update({'0':500,'2':500,'3':2500})
        config=apply_tracking_profile(config)
        transport=FakeArm();transport.positions={int(a):p for a,p in config['pose_reference'].items()}
        arm=BoundedArm(transport,config,time.monotonic())
        low,high=arm.displacement_bounds()
        self.assertEqual(low[:2],[0,0])
        self.assertEqual(high[2],0)
        self.assertGreater(high[0],0);self.assertGreater(high[1],0);self.assertLess(low[2],0)

    def test_restart_migrates_saved_limits_and_adapts_distant_models_in_place(self):
        for distant in (False,True):
            with self.subTest(distant=distant),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'config').mkdir()
                (root/'config/bowl-follow.runtime.json').write_text(json.dumps(
                    dict(profile_version='1.1.0',motion_range_policy=FULL_RANGE_POLICY)))
                path=root/'work/bowl-follow/current-trial.json';path.parent.mkdir(parents=True)
                positions={0:1547,1:1366,2:1229,3:590,4:1495,5:1533}
                pose={str(a):1500 for a in range(6)} if distant else {str(a):p for a,p in positions.items()}
                old=full_config();old.pop('motion_range_policy')
                old.update(local_calibration=measured(pose))
                old=apply_tracking_profile(old)
                path.write_text(json.dumps(old));original=path.read_bytes()
                arm=Mock();arm.positions.return_value=positions
                feed=Mock();feed.current.side_effect=target
                events=[];arm.close.side_effect=lambda:events.append('close')
                with patch.object(preparation,'ROOT',root),patch.object(preparation,'rclpy'), \
                     patch.object(preparation,'ArmSerial',return_value=arm), \
                     patch.object(preparation,'TargetFeed',return_value=feed), \
                     patch.object(preparation,'prepare_current_pose',side_effect=lambda:events.append('adapt')):
                    preparation.prepare_resume()
                arm.move.assert_not_called()
                if distant:
                    self.assertEqual(events,['close','adapt'])
                    self.assertEqual(path.read_bytes(),original)
                else:
                    self.assertEqual(events,['close'])
                    saved=json.loads(path.read_text())
                    self.assertEqual(saved['motion_range_policy'],FULL_RANGE_POLICY)
                    self.assertEqual(saved['pose_reference'],{str(a):p for a,p in positions.items()})
                    self.assertEqual(saved['axes']['2']['min_p'],500)
                    self.assertEqual(saved['axes']['3']['max_p'],2500)

    def test_calibration_at_each_protocol_edge_measures_both_directions_inward(self):
        for endpoint in (500,2500):
            with self.subTest(endpoint=endpoint),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'config').mkdir()
                (root/'config/bowl-follow.runtime.json').write_text(json.dumps(
                    dict(motion_range_policy=FULL_RANGE_POLICY)))
                vendor=root/'vendor';vendor.write_bytes(b'offline fixture only')
                positions={a:endpoint for a in range(6)};commands=[];reply=[b'']
                arm=Mock();arm.position.side_effect=lambda axis:positions[axis]
                def send(command):
                    motion=re.fullmatch(r'#(\d{3})P(\d{4})T(\d{4})!',command)
                    if motion:
                        axis,goal,_=map(int,motion.groups())
                        self.assertTrue(500<=goal<=2500)
                        self.assertLessEqual(abs(goal-positions[axis]),65)
                        commands.append((axis,goal));positions[axis]=goal
                    elif command.endswith('PVER!'):
                        reply[0]=(command[:4]+'@ ZL-ZServo_AD_CBM V2.1.16STG!').encode()
                def read(_):
                    data=reply[0];reply[0]=b'';return data
                arm._send.side_effect=send;arm._read_for.side_effect=read
                feed=Mock()
                feed.sample.side_effect=lambda:dict(center=[160+2*(positions[0]-endpoint),
                    120-(positions[2]-endpoint)+(positions[3]-endpoint)])
                ticks=itertools.count(0,.2)
                argv=['calibrate_bowl_follow.py','--execute','--axes','0','2','3',
                      '--amplitude-p','60','--axis-amplitudes','0:30,2:60','--negative-amplitudes','3:25']
                with patch.object(calibration,'ROOT',root),patch.object(calibration,'rclpy'), \
                     patch.object(calibration,'VENDOR_ACTIONS',vendor), \
                     patch.object(calibration,'VENDOR_SHA256',hashlib.sha256(vendor.read_bytes()).hexdigest()), \
                     patch.object(calibration,'ArmSerial',return_value=arm), \
                     patch.object(calibration,'TargetFeed',return_value=feed), \
                     patch.object(calibration.time,'monotonic',side_effect=lambda:next(ticks)), \
                     patch.object(calibration.time,'sleep'),patch('sys.argv',argv), \
                     contextlib.redirect_stdout(io.StringIO()):
                    calibration.main()
                report=json.loads((root/'work/bowl-follow/last-calibration.json').read_text())
                self.assertEqual(report['result'],'LOCAL_JACOBIAN_PASS')
                self.assertEqual(positions,{a:endpoint for a in range(6)})
                self.assertEqual({a for a,_ in commands},{0,2,3})
                for axis in (0,2,3):
                    self.assertEqual({1 if row['actual_delta_p']>0 else -1
                                      for row in report['trials'] if row['axis']==axis},{-1,1})


if __name__=='__main__':unittest.main()
