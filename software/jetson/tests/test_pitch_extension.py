"""Only completed coordinated measurements can enlarge wrist command limits."""
import copy
import unittest
from vision.tracking_profile import apply_measured_pitch_extension


class PitchExtensionTests(unittest.TestCase):
    def fixture(self):
        config=dict(tracking_axes=[0,2,3],local_calibration={'jacobian':[[1.6,-.14,.03],[0,-1.25,1.15]]},
            axes={'2':dict(min_p=1600,max_p=1859),'3':dict(min_p=835,max_p=1076)},
            pose_reference={'3':945},operator_target_lock={'weights_sha256':'a'*64})
        report=dict(result='MEASURED_PITCH_EXTENSION_PASS',motion_enabled=True,finished_unix=100,
            target={'weights_sha256':'a'*64},maximum_wrist_command_p=1185,
            steps=[dict(before_p={'2':1779,'3':1120},target_p={'2':1798,'3':1185},
                        after_p={'2':1810,'3':1156},before_center=[173,85],after_center=[172,86],
                        observed_image_shift_px=[-1,1],predicted_image_shift_px=[-3.26,2.65])])
        return config,report

    def test_only_measured_wrist_upper_endpoint_changes(self):
        config,report=self.fixture()
        updated=apply_measured_pitch_extension(config,report,now=101)
        self.assertEqual(updated['axes']['3'],dict(min_p=835,max_p=1185))
        self.assertEqual(updated['axes']['2'],config['axes']['2'])
        self.assertEqual(config['axes']['3']['max_p'],1076)

    def test_rejected_image_or_unmeasured_endpoint_never_expands(self):
        for kind in ('failed','endpoint','image','oversize','stale'):
            with self.subTest(kind=kind):
                config,report=self.fixture()
                if kind=='failed':report['result']='ABORTED'
                if kind=='endpoint':report['maximum_wrist_command_p']=1250
                if kind=='image':report['steps'][0]['observed_image_shift_px']=[0,35]
                if kind=='oversize':report['steps'][0]['target_p']['3']=1250
                if kind=='stale':report['finished_unix']=-2000
                with self.assertRaises(ValueError):apply_measured_pitch_extension(config,report,now=101)

    def test_verified_prefix_needs_explicit_bounds_stop_and_provenance(self):
        config,report=self.fixture();report['result']='MEASURED_PITCH_PREFIX_PASS'
        with self.assertRaises(ValueError):apply_measured_pitch_extension(config,report,now=101)
        report.update(source_stop_error='RuntimeError: Coordinated pitch probe would exceed bounded range',
                      completion_scope='completed_steps_only',source_report_sha256='b'*64)
        updated=apply_measured_pitch_extension(config,report,now=101)
        self.assertEqual(updated['axes']['3']['max_p'],1185)
