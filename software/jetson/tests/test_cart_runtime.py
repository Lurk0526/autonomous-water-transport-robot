"""Input, provenance, localization and calibration checks, without hardware I/O."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from cart.parameters import apply_measurements
from cart.feedback import FeedbackMonitor,replay_jsonl
from cart.localization import PlanarLocalization
from cart.calibration import estimate_braking
from cart.rgbd_audit import nearest_index,timing

ROOT=Path(__file__).resolve().parents[1]


def cfg():
    return json.loads((ROOT/'config/cart-integration.json').read_text())


def packet(kind='odometry',t=1.0,**changes):
    p={'kind':kind,'stamp_s':t,'source':'synthetic_fixture'}
    p.update({
        'odometry':dict(frame='odom',child_frame='base_link',x_m=0.,y_m=0.,yaw_rad=0.,speed_m_s=0.,yaw_rate_rad_s=0.,pose_sigma_m=.01,yaw_sigma_rad=.02,covariance_known=True),
        'imu':dict(frame='base_link',yaw_rate_rad_s=0.,yaw_rate_variance=.0004),
        'steering':dict(measured=True,steer_rad=0.),
        'perception':dict(valid=True,fusion_valid=True,valid_until_s=t+.5),
        'obstacles':dict(frame='base_link',mount_ready=True,valid=True,valid_until_s=t+.5,point_count=0,points_xyz_m=[])
    }[kind])
    p.update(changes);return p


class ParameterProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.base=json.loads((ROOT/'sim/line_free_cart/config.json').read_text())
        self.registry=json.loads((ROOT/'config/cart-vehicle-measurements.json').read_text())

    def test_missing_measurements_do_not_become_real_vehicle_values(self):
        result,report=apply_measurements(self.base,self.registry,ROOT)
        self.assertEqual(result['vehicle'],self.base['vehicle'])
        self.assertIsNone(result['measured_vehicle_parameters'])
        self.assertFalse(report['all_required_measurements_present'])

    def test_supplied_measurement_changes_only_its_parameter_and_preserves_source(self):
        before=deepcopy(self.base)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'measurement.txt';p.write_text('Test fixture evidence, not a real vehicle measurement')
            self.registry['parameters']['wheelbase_m']=dict(value=.41,unit='m',status='measured',evidence={'path':'measurement.txt','sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            result,report=apply_measurements(self.base,self.registry,d)
        self.assertEqual(result['vehicle']['wheelbase_m'],.41)
        self.assertEqual(self.base,before)
        self.assertIn('wheelbase_m',report['applied'])
        self.assertFalse(report['all_required_measurements_present'])

    def test_synthetic_status_wrong_units_and_nonfinite_numbers_are_rejected(self):
        for changes in [dict(value=.5,unit='m',status='synthetic'),dict(value=500,unit='mm',status='measured'),dict(value=float('nan'),unit='m',status='measured'),dict(value=True,unit='m',status='measured')]:
            with self.subTest(changes=changes):
                self.registry['parameters']['wheelbase_m']=changes
                result,report=apply_measurements(self.base,self.registry,ROOT)
                self.assertIn('wheelbase_m',report['rejected'])
                self.assertEqual(result['vehicle']['wheelbase_m'],self.base['vehicle']['wheelbase_m'])

    def test_changed_evidence_does_not_silently_reuse_old_measurement(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence';p.write_text('original')
            entry=dict(value=.4,unit='m',status='measured',evidence={'path':'evidence','sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            p.write_text('changed')
            self.registry['parameters']['wheelbase_m']=entry
            _,report=apply_measurements(self.base,self.registry,d)
            self.assertEqual(report['rejected']['wheelbase_m'],'evidence_hash_mismatch')


class FeedbackContractTests(unittest.TestCase):
    def test_all_synthetic_inputs_are_distinguished_from_real_chassis_feedback(self):
        f=FeedbackMonitor(cfg())
        for kind in ['odometry','imu','steering','perception','obstacles']:
            self.assertTrue(f.ingest(packet(kind),1.01))
        self.assertTrue(f.snapshot(1.02)['all_required_inputs_valid'])
        self.assertFalse(f.snapshot(1.02)['real_chassis_feedback_seen'])

    def test_covariance_zero_unknown_negative_and_truthy_string_are_rejected(self):
        for changes in [dict(covariance_known=False),dict(covariance_known='true'),dict(pose_sigma_m=-1),dict(pose_sigma_m=1),dict(yaw_sigma_rad=float('inf'))]:
            with self.subTest(changes=changes):
                f=FeedbackMonitor(cfg());self.assertFalse(f.ingest(packet(**changes),1.01))

    def test_future_stale_and_nonfinite_timestamps_are_rejected(self):
        for t in [2.,0.,float('nan')]:
            with self.subTest(t=t):
                self.assertFalse(FeedbackMonitor(cfg()).ingest(packet(t=t),1.0))

    def test_frame_mismatch_prevents_camera_frame_odometry(self):
        self.assertFalse(FeedbackMonitor(cfg()).ingest(packet(frame='camera_depth_optical_frame'),1.01))

    def test_replayed_sequence_and_timestamp_cannot_restore_freshness(self):
        f=FeedbackMonitor(cfg());self.assertTrue(f.ingest(packet(sequence=4),1.01))
        self.assertFalse(f.ingest(packet(t=1.1,sequence=3),1.11))
        self.assertFalse(f.snapshot(1.12)['inputs']['odometry']['fresh_and_valid'])
        self.assertFalse(f.ingest(packet(sequence=5),1.12))

    def test_position_jump_is_detected_against_allowed_motion(self):
        f=FeedbackMonitor(cfg());f.ingest(packet(),1.)
        self.assertFalse(f.ingest(packet(t=1.05,x_m=1),1.06))
        self.assertIn('localization_jump',f.latched)

    def test_frozen_pose_while_speed_reports_motion_is_detected(self):
        f=FeedbackMonitor(cfg())
        for i in range(15):f.ingest(packet(t=1+i*.1,speed_m_s=.5),1+i*.1)
        self.assertIn('odometry_motion_inconsistent',f.latched)

    def test_commanded_steering_cannot_be_used_as_measured_feedback(self):
        self.assertFalse(FeedbackMonitor(cfg()).ingest(packet('steering',measured=False),1.01))

    def test_empty_obstacle_cloud_is_unknown_without_valid_mount(self):
        for change in [dict(mount_ready=False),dict(valid=False),dict(frame='camera_depth_optical_frame')]:
            with self.subTest(change=change):
                self.assertFalse(FeedbackMonitor(cfg()).ingest(packet('obstacles',**change),1.01))

    def test_obstacle_nan_geometry_is_rejected(self):
        self.assertFalse(FeedbackMonitor(cfg()).ingest(packet('obstacles',point_count=1,points_xyz_m=[[0,0,float('nan')]]),1.01))

    def test_perception_validity_cannot_be_extended_indefinitely(self):
        self.assertFalse(FeedbackMonitor(cfg()).ingest(packet('perception',valid_until_s=100),1.01))

    def test_snapshot_ages_inputs_even_when_nothing_new_arrives(self):
        f=FeedbackMonitor(cfg());f.ingest(packet(),1.01)
        self.assertFalse(f.snapshot(1.5)['inputs']['odometry']['fresh_and_valid'])

    def test_imu_wheel_disagreement_and_time_misalignment_are_separate(self):
        f=FeedbackMonitor(cfg());f.ingest(packet(yaw_rate_rad_s=.3),1.)
        f.ingest(packet('imu'),1.)
        self.assertTrue(f.snapshot(1.)['wheel_imu_yaw_disagreement'])
        g=FeedbackMonitor(cfg());g.ingest(packet(t=1.),1.2);g.ingest(packet('imu',t=1.2),1.2)
        self.assertFalse(g.snapshot(1.2)['imu_odom_time_aligned'])
        self.assertFalse(g.snapshot(1.2)['wheel_imu_yaw_disagreement'])

    def test_bad_json_does_not_abort_the_remaining_recording(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.jsonl'
            p.write_text('{broken}\n'+json.dumps(dict(packet(),received_s=1.01))+'\n')
            r=replay_jsonl(p,cfg())
        self.assertEqual(r['counts']['rejected'],1)
        self.assertTrue(r['inputs']['odometry']['received'])


class LocalizationTests(unittest.TestCase):
    def test_wheel_gyro_integration_matches_analytic_circle(self):
        e=PlanarLocalization();e.predict(0,.3,.2)
        for i in range(1,201):self.assertTrue(e.predict(i*.02,.3,.2))
        self.assertAlmostEqual(e.pose[0],1.5*math.sin(.8),delta=2e-6)
        self.assertAlmostEqual(e.pose[1],1.5*(1-math.cos(.8)),delta=2e-6)
        self.assertEqual(e.snapshot()['origin'],'arbitrary_odometry_origin')

    def test_uncertainty_grows_without_absolute_updates(self):
        e=PlanarLocalization();before=e.snapshot()['pose_sigma_m'];e.predict(0,.5,0)
        for i in range(1,1001):e.predict(i*.02,.5,0)
        self.assertGreater(e.snapshot()['pose_sigma_m'],before)

    def test_absolute_update_reduces_uncertainty_and_preserves_covariance_symmetry(self):
        e=PlanarLocalization();e.predict(0,.2,0)
        for i in range(1,101):e.predict(i*.02,.2,0)
        before=e.snapshot()['pose_sigma_m']
        self.assertTrue(e.anchor(2,[.41,.01,.01],.01,.01))
        self.assertLess(e.snapshot()['pose_sigma_m'],before)
        for i in range(3):
            self.assertGreater(e.cov[i][i],0)
            for j in range(3):self.assertAlmostEqual(e.cov[i][j],e.cov[j][i],places=12)

    def test_old_repeated_and_outlier_anchors_are_rejected(self):
        e=PlanarLocalization();e.predict(1,0,0)
        self.assertFalse(e.anchor(.8,[0,0,0],.01,.01))
        self.assertFalse(e.anchor(1,[3,0,0],.01,.01))
        self.assertTrue(e.anchor(1,[0,0,0],.01,.01))
        self.assertFalse(e.anchor(1,[0,0,0],.01,.01))

    def test_timestamp_regression_gap_and_nonfinite_inputs_are_rejected(self):
        e=PlanarLocalization();e.predict(1,0,0)
        for t,v in [(.9,0),(1.5,0),(1.05,float('nan'))]:self.assertFalse(e.predict(t,v,0))
        self.assertEqual(e.pose,[0,0,0])


class RecordedDataChecks(unittest.TestCase):
    def test_pairing_uses_nearest_timestamp_at_both_ends(self):
        self.assertEqual(nearest_index([1,2,3],.5),0)
        self.assertEqual(nearest_index([1,2,3],3.5),2)
        self.assertEqual(nearest_index([1,2,3],1.8),1)
        self.assertIsNone(nearest_index([],1))

    def test_duplicate_frame_stamp_and_gap_are_reported(self):
        r=timing([1,1.03,1.03,2])
        self.assertEqual(r['non_increasing'],1)
        self.assertAlmostEqual(r['max_gap_s'],.97)

    def test_braking_fit_recovers_known_fixture_deceleration_without_applying_it(self):
        rows=[]
        for i in range(61):
            t=i*.05
            rows.append(dict(t_s=t,command_speed_m_s=.5 if t<1 else 0.,measured_speed_m_s=max(0,.5-.4*max(0,t-1.2))))
        r=estimate_braking(rows,'synthetic_fixture')
        self.assertAlmostEqual(r['minimum_trial_decel_m_s2'],.4,places=8)
        self.assertGreaterEqual(r['maximum_observed_response_delay_s'],.2)
        self.assertFalse(r['enough_repeated_trials'])
        self.assertFalse(r['measurements_automatically_applied'])

    def test_braking_fit_rejects_large_gaps_and_time_reversal(self):
        rows=[dict(t_s=t,command_speed_m_s=.5,measured_speed_m_s=.5) for t in [0,.05,.6,.65]]
        with self.assertRaises(ValueError):estimate_braking(rows)


if __name__=='__main__':unittest.main()
