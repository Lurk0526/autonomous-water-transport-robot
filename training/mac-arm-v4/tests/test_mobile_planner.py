"""Pure planner contract checks. Run on the Mac; no device or MuJoCo use."""
import copy
import math
import unittest

import numpy as np

from arm_training.mobile_planner import (MobileCoordinator, compose_pose,
    default_mobile_config, dubins_paths, footprint_clear, plan_parking,
    point_in_model, screen_grasp_at_parking, target_validation, wrap)


def target(stamp=1., source='main_rgbd', position=(.6, 0., .35), sigma=.003):
    return {'valid': True, 'frame': 'odom', 'stamp_s': stamp,
            'valid_until_s': stamp + .5, 'target_id': 'bowl-1',
            'source': source, 'position_m': list(position),
            'covariance_m2': (np.eye(3) * sigma ** 2).tolist()}


def table():
    return {'frame': 'odom', 'center_xy_m': [.6, 0.], 'half_size_m': [.18, .18],
            'yaw_rad': 0., 'bottom_height_m': 0., 'top_height_m': .35}


def odom(stamp=1., pose=(-1., 0., 0.), speed=0., yaw_rate=0.):
    return {'valid': True, 'frame': 'odom', 'stamp_s': stamp,
            'xy_yaw': list(pose), 'speed_m_s': speed, 'yaw_rate_rad_s': yaw_rate}


def fake_ik(point, seed=None):
    return np.zeros(5), True


class MobilePlannerTests(unittest.TestCase):
    def test_target_record_rejects_stale_covariance_and_wrong_frame(self):
        self.assertEqual(target_validation(target(), 1.1), (True, None))
        self.assertFalse(target_validation(target(), 2.)[0])
        bad = target(); bad['frame'] = 'camera_optical_frame'
        self.assertFalse(target_validation(bad, 1.1)[0])
        bad = target(); bad['covariance_m2'][0][0] = -.1
        self.assertFalse(target_validation(bad, 1.1)[0])
        bad = target(); bad['valid_until_s'] = 100.
        self.assertFalse(target_validation(bad, 1.1)[0])

    def test_mount_transform_preserves_model_z_reference(self):
        pose = compose_pose([1., 2., math.pi / 2], [.3, 0., 0.])
        np.testing.assert_allclose(pose, [1., 2.3, math.pi / 2], atol=1e-12)
        np.testing.assert_allclose(point_in_model([1., 2.5, .35], pose), [.2, 0., .35], atol=1e-12)

    def test_table_is_solid_volume_and_full_vehicle_front_is_checked(self):
        vehicle = default_mobile_config()['vehicle']
        # rear axle itself is outside the table, but the vehicle nose intersects.
        self.assertLess(.2, .6 - .18)
        clear, reason = footprint_clear([.2, 0., 0.], table(), vehicle)
        self.assertFalse(clear)
        self.assertEqual(reason, 'vehicle_footprint_intersects_solid_platform')
        with self.assertRaises(KeyError):
            bad = table(); del bad['bottom_height_m']
            footprint_clear([0., 0., 0.], bad, vehicle)

    def test_dubins_endpoint_curvature_sampling_and_no_spin(self):
        cfg = default_mobile_config()
        far_table = table(); far_table['center_xy_m'] = [20., 20.]
        start, end = [-1., 0., .2], [1., .3, .5]
        paths, rejected = dubins_paths(start, end, cfg['vehicle'], far_table, maximum_length_m=15.)
        self.assertTrue(paths, rejected)
        bound = math.tan(cfg['vehicle']['max_steer_rad']) / cfg['vehicle']['wheelbase_m']
        for path in paths:
            final = path['samples'][-1]
            np.testing.assert_allclose([final['x_m'], final['y_m']], end[:2], atol=1e-6)
            self.assertLess(abs(wrap(final['yaw_rad'] - end[2])), 1e-6)
            for before, after in zip(path['samples'], path['samples'][1:]):
                ds = after['s_m'] - before['s_m']
                self.assertGreaterEqual(ds, 0.)
                self.assertLessEqual(ds, cfg['planning']['path_sample_m'] + 1e-12)
                self.assertLessEqual(abs(after['curvature_m_inv']), bound + 1e-12)
                self.assertLessEqual(abs(wrap(after['yaw_rad'] - before['yaw_rad'])), bound * ds + 1e-9)

    def test_path_rejects_initial_obstacle_overlap(self):
        cfg = default_mobile_config()
        far_table = table(); far_table['center_xy_m'] = [20., 20.]
        paths, reasons = dubins_paths([-1., 0., 0.], [1., 0., 0.], cfg['vehicle'], far_table,
            obstacles=[{'x_m': -1., 'y_m': 0., 'radius_m': .2}])
        self.assertFalse(paths)
        self.assertTrue(any(item['reason'] == 'vehicle_footprint_intersects_obstacle' for item in reasons))

    def test_parking_uses_mount_checks_whole_arm_path_and_reports_scope(self):
        cfg = default_mobile_config()
        cfg['planning']['heading_offsets_deg'] = [0.]
        calls = []
        def geometry(q, point, phase, platform):
            calls.append((phase, point.copy()))
            self.assertEqual(platform['frame'], 'arm_model')
            return {'valid': True, 'clearance_m': .02}
        report = plan_parking(target(), table(), odom(), cfg, fake_ik, geometry_check=geometry, now_s=1.1)
        self.assertTrue(report['valid'], report)
        selected = report['selected']
        self.assertFalse(report['hardware_output_enabled'])
        self.assertTrue(selected['arm']['full_arm_geometry_checked'])
        self.assertGreater(len(selected['joint_path']), 4)
        self.assertEqual({item['phase'] for item in selected['joint_path']}, {'home', 'grasp', 'lift', 'lower'})
        np.testing.assert_allclose(compose_pose(selected['parking_xy_yaw'], cfg['arm']['mount_xy_yaw_m_rad']), selected['arm_model_parking_xy_yaw'])
        self.assertAlmostEqual(selected['score'], sum(selected['score_components'].values()))
        self.assertTrue(calls)
        for sample in selected['path']['samples']:
            self.assertTrue(footprint_clear([sample['x_m'], sample['y_m'], sample['yaw_rad']], table(), cfg['vehicle'])[0])

    def test_intermediate_arm_rejection_is_not_hidden_by_reachable_endpoints(self):
        cfg = default_mobile_config()
        def geometry(q, point, phase, platform):
            return {'valid': not (phase == 'lift' and .40 < point[2] < .42), 'reason': 'intermediate_table_collision'}
        report = plan_parking(target(), table(), odom(), cfg, fake_ik, geometry_check=geometry, now_s=1.1)
        self.assertFalse(report['valid'])
        self.assertTrue(any('intermediate_table_collision' in item.get('rejection_reason', '') for item in report['candidates']))

    def test_unreachable_and_camera_fov_are_distinct_rejections(self):
        cfg = default_mobile_config()
        bad_ik = lambda point, seed=None: (np.zeros(5), False)
        unreachable = plan_parking(target(), table(), odom(), cfg, bad_ik, now_s=1.1)
        self.assertFalse(unreachable['valid'])
        self.assertTrue(any('arm_ik_unreachable' in item.get('rejection_reason', '') for item in unreachable['candidates']))
        cfg['camera']['fixed_pose_in_model'][0][3] = 10.
        hidden = plan_parking(target(), table(), odom(), cfg, fake_ik, now_s=1.1)
        self.assertFalse(hidden['valid'])
        self.assertTrue(any('camera_fov' in item.get('rejection_reason', '') for item in hidden['candidates']))

    def test_precise_recheck_keeps_current_parking_and_rejects_missing_wrist_source(self):
        cfg = default_mobile_config()
        parked = [.08, 0., 0.]
        report = screen_grasp_at_parking(target(source='wrist'), table(), parked, cfg, fake_ik, now_s=1.1)
        self.assertTrue(report['valid'], report)
        self.assertEqual(report['parking_xy_yaw'], parked)
        self.assertFalse(report['hardware_output_enabled'])
        report = screen_grasp_at_parking(target(), table(), parked, cfg, fake_ik, now_s=1.1)
        self.assertFalse(report['valid'])
        self.assertEqual(report['reason'], 'target_source_mismatch')

    def test_coordinator_requires_two_cameras_continuous_stop_and_arm_recheck(self):
        cfg = default_mobile_config(); flow = MobileCoordinator(cfg)
        result = flow.update(1., odom(), main_target=target())
        self.assertTrue(result['plan_requested'])
        plan = {'valid': True, 'target_id': 'bowl-1', 'selected': {'parking_xy_yaw': [0., 0., 0.]}}
        flow.select_plan(plan)
        for tick in range(7):
            stamp = 1.1 + tick * .1
            result = flow.update(stamp, odom(stamp, pose=(0., 0., 0.)), main_target=target(stamp))
        self.assertEqual(result['state'], 'acquire_wrist')
        result = flow.update(1.8, odom(1.8, pose=(0., 0., 0.)), wrist_target=target(1.8, source='wrist'))
        self.assertFalse(result['grasp_ready'])
        self.assertEqual(result['reason'], 'precise_target_arm_recheck_required')
        result = flow.update(1.9, odom(1.9, pose=(0., 0., 0.)), wrist_target=target(1.9, source='wrist'), precise_grasp_feasible=True)
        self.assertTrue(result['grasp_ready'])
        result = flow.update(2., odom(2., pose=(0., 0., 0.), speed=.02), wrist_target=target(2., source='wrist'), precise_grasp_feasible=True)
        self.assertEqual(result['state'], 'stop')
        self.assertFalse(result['hardware_output_enabled'])

    def test_stop_evidence_resets_when_vehicle_moves_and_stale_fault_latches(self):
        cfg = default_mobile_config(); flow = MobileCoordinator(cfg)
        flow.select_plan({'valid': True, 'target_id': 'bowl-1', 'selected': {'parking_xy_yaw': [0., 0., 0.]}})
        for stamp, speed in [(1., 0.), (1.1, 0.), (1.2, .03), (1.3, 0.), (1.4, 0.)]:
            result = flow.update(stamp, odom(stamp, pose=(0., 0., 0.), speed=speed), main_target=target(stamp))
        self.assertEqual(result['state'], 'stop_confirm')
        result = flow.update(2., odom(1.4, pose=(0., 0., 0.)), main_target=target(2.))
        self.assertEqual(result['state'], 'stop')
        next_result = flow.update(2.1, odom(2.1), main_target=target(2.1))
        self.assertEqual(next_result['state'], 'stop')


if __name__ == '__main__':
    unittest.main()
