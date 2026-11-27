"""Bounded mobile-manipulation planning and intent coordination, without I/O.

All vehicle dimensions, optical poses, and controller limits in the defaults are
SIMULATION ASSUMPTIONS.  This module opens no device and publishes no ROS topic.
It ranks a finite parking set; it does not claim global optimality.  Forward-only
Dubins candidates obey a front-steered bicycle's curvature bound.  They do not
include steering-rate dynamics, reversing, or arbitrary-world navigation.
"""
from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np


TAU = 2.0 * math.pi


def wrap(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def default_mobile_config():
    """Return independent defaults, explicitly unmeasured simulation settings."""
    return {
        'status': 'simulation_assumptions_not_measured_vehicle',
        'vehicle': {
            'wheelbase_m': .22, 'width_m': .20, 'front_overhang_m': .04,
            'rear_overhang_m': .04, 'height_m': .20, 'max_steer_rad': .60,
            'footprint_margin_m': .025,
        },
        'planning': {
            'relative_target_x_m': [.16, .20, .24],
            'relative_target_y_m': [-.04, 0., .04],
            'heading_offsets_deg': [-25., 0., 25.],
            'path_sample_m': .02, 'max_path_length_m': 8.,
            'arm_sample_m': .01, 'max_joint_step_rad': .4,
            'minimum_joint_margin_rad': .015, 'minimum_tcp_clearance_m': .006,
            'target_max_sigma_m': .025, 'max_source_age_s': .5,
            'max_future_s': .02, 'table_target_height_tolerance_m': .03,
            'obstacles': [],
            'score_weights': {'path_length': 1., 'turning': .15,
                              'joint_margin': .15, 'clearance': .05,
                              'centering': .2},
        },
        'arm': {
            # rear axle -> model x/y/yaw; model's z origin remains as in BowlEnv.
            'mount_xy_yaw_m_rad': [.32, 0., 0.],
            'joint_limits_rad': [[-2.3, 2.3]] * 5,
            'grasp_height_above_table_m': .035,
            'approach_clearance_m': .06, 'lift_m': .06,
            'lower_offset_m': -.0015, 'tcp_envelope_radius_m': .012,
        },
        'camera': {
            'status': 'synthetic_observation_pose_not_hand_eye_calibration',
            # Optical x right, y down, z forward. Synthetic downward view.
            'fixed_pose_in_model': [[0., 1., 0., .20], [1., 0., 0., 0.],
                                    [0., 0., -1., .65], [0., 0., 0., 1.]],
            'horizontal_fov_deg': 80., 'vertical_fov_deg': 65.,
            'near_m': .08, 'far_m': 2., 'edge_margin_fraction': .1,
            'check_phases': ['home', 'grasp'],
        },
        'coordinator': {
            'odometry_max_age_s': .25, 'target_max_age_s': .5,
            'main_max_sigma_m': .025, 'wrist_max_sigma_m': .008,
            'coarse_replan_distance_m': .04,
            'parking_position_tolerance_m': .025,
            'parking_yaw_tolerance_rad': .08, 'stop_speed_m_s': .01,
            'stop_yaw_rate_rad_s': .03, 'stop_dwell_s': .5,
            'maximum_sample_gap_s': .2, 'wrist_after_stop_required': True,
        },
    }


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('invalid_' + name)
    return result


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('invalid_' + name)
    return value


def target_validation(target, now_s, max_age_s=.5, max_sigma_m=.025,
                      source=None, future_s=.02):
    """Validate a bowl-bottom centre in odom; unknown covariance is rejected."""
    try:
        if target.get('valid') is not True or target.get('frame') != 'odom':
            return False, 'target_invalid_or_wrong_frame'
        if not isinstance(target.get('target_id'), str) or not target['target_id']:
            return False, 'target_id_missing'
        if source is not None and target.get('source') != source:
            return False, 'target_source_mismatch'
        _array(target['position_m'], (3,), 'target_position')
        cov = _array(target['covariance_m2'], (3, 3), 'target_covariance')
        if not np.allclose(cov, cov.T, atol=1e-10):
            return False, 'target_covariance_not_symmetric'
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues.min() < -1e-12 or math.sqrt(max(0., float(eigenvalues.max()))) > max_sigma_m:
            return False, 'target_covariance_out_of_bounds'
        stamp, expiry = float(target['stamp_s']), float(target['valid_until_s'])
        if not all(math.isfinite(v) for v in (now_s, stamp, expiry)):
            return False, 'target_timestamp_invalid'
        if not -future_s <= now_s - stamp <= max_age_s or not stamp <= expiry or expiry < now_s:
            return False, 'target_stale_or_future'
        if expiry - stamp > max_age_s + 1e-9:
            return False, 'target_validity_horizon_too_long'
        return True, None
    except (KeyError, ValueError, TypeError, AttributeError):
        return False, 'target_record_invalid'


def _validate_config(cfg):
    vehicle, p, arm, camera = (cfg[k] for k in ('vehicle', 'planning', 'arm', 'camera'))
    for key in ('wheelbase_m', 'width_m', 'height_m', 'max_steer_rad'):
        _positive(vehicle[key], key)
    if vehicle['max_steer_rad'] >= math.pi / 2:
        raise ValueError('steering_must_be_below_pi_over_two')
    for key in ('front_overhang_m', 'rear_overhang_m', 'footprint_margin_m'):
        if not math.isfinite(float(vehicle[key])) or vehicle[key] < 0:
            raise ValueError('invalid_' + key)
    for key in ('path_sample_m', 'arm_sample_m', 'max_path_length_m',
                'max_joint_step_rad', 'target_max_sigma_m', 'max_source_age_s'):
        _positive(p[key], key)
    _array(arm['mount_xy_yaw_m_rad'], (3,), 'arm_mount')
    ranges = np.asarray(arm['joint_limits_rad'], dtype=float)
    if ranges.ndim != 2 or ranges.shape[1] != 2 or not np.isfinite(ranges).all() or np.any(ranges[:, 1] <= ranges[:, 0]):
        raise ValueError('invalid_joint_limits')
    for key in ('relative_target_x_m', 'relative_target_y_m', 'heading_offsets_deg'):
        values = np.asarray(p[key], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError('invalid_' + key)
    if not 0 < camera['horizontal_fov_deg'] < 179 or not 0 < camera['vertical_fov_deg'] < 179:
        raise ValueError('invalid_camera_fov')
    if not 0 <= camera['edge_margin_fraction'] < .5 or not 0 < camera['near_m'] < camera['far_m']:
        raise ValueError('invalid_camera_range_or_margin')
    return ranges


def compose_pose(base_xy_yaw, local_xy_yaw):
    base, local = np.asarray(base_xy_yaw), np.asarray(local_xy_yaw)
    c, s = math.cos(base[2]), math.sin(base[2])
    return np.array([base[0] + c * local[0] - s * local[1],
                     base[1] + s * local[0] + c * local[1], wrap(base[2] + local[2])])


def point_in_model(point_odom, model_pose):
    point, pose = np.asarray(point_odom), np.asarray(model_pose)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    d = point[:2] - pose[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], point[2]])


def _rectangle(center, half, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    return np.array([center + rotation @ np.array([x, y])
                     for x, y in ((-half[0], -half[1]), (half[0], -half[1]),
                                  (half[0], half[1]), (-half[0], half[1]))])


def _vehicle_rectangle(pose, vehicle, margin=0.):
    front = vehicle['wheelbase_m'] + vehicle['front_overhang_m']
    rear = vehicle['rear_overhang_m']
    center = compose_pose(pose, [(front - rear) / 2, 0., 0.])[:2]
    return _rectangle(center, [(front + rear) / 2 + margin,
                                vehicle['width_m'] / 2 + margin], pose[2])


def _rectangles_overlap(first, second):
    for rectangle in (first, second):
        for a, b in zip(rectangle, np.roll(rectangle, -1, axis=0)):
            edge = b - a
            axis = np.array([-edge[1], edge[0]])
            one, two = first @ axis, second @ axis
            if one.max() < two.min() or two.max() < one.min():
                return False
    return True


def _circle_rectangle_collision(pose, obstacle, vehicle, margin):
    point = point_in_model([obstacle['x_m'], obstacle['y_m'], 0.], pose)
    front = vehicle['wheelbase_m'] + vehicle['front_overhang_m'] + margin
    rear = -vehicle['rear_overhang_m'] - margin
    half = vehicle['width_m'] / 2 + margin
    dx, dy = max(rear - point[0], 0., point[0] - front), max(abs(point[1]) - half, 0.)
    return math.hypot(dx, dy) <= obstacle['radius_m']


def footprint_clear(pose, table, vehicle, obstacles=(), additional_margin_m=0.):
    """Test full rectangular vehicle against a solid table and circular obstacles."""
    margin = vehicle['footprint_margin_m'] + additional_margin_m
    # A table's underside/legs are conservatively its full rectangular volume.
    if table['bottom_height_m'] <= vehicle['height_m'] and table['top_height_m'] >= 0:
        solid = _rectangle(np.asarray(table['center_xy_m']), table['half_size_m'], table['yaw_rad'])
        if _rectangles_overlap(_vehicle_rectangle(pose, vehicle, margin), solid):
            return False, 'vehicle_footprint_intersects_solid_platform'
    for obstacle in obstacles:
        if _circle_rectangle_collision(pose, obstacle, vehicle, margin):
            return False, 'vehicle_footprint_intersects_obstacle'
    return True, None


def _arc_length(center, start, end, turn, radius):
    one = math.atan2(start[1] - center[1], start[0] - center[0])
    two = math.atan2(end[1] - center[1], end[0] - center[0])
    angle = (turn * (two - one)) % TAU
    if angle > TAU - 1e-10:
        angle = 0.
    return radius * angle


def _dubins_segments(start, goal, radius):
    """Construct the six Dubins families geometrically; lengths are physical metres."""
    p, q = start[:2], goal[:2]
    left0 = np.array([-math.sin(start[2]), math.cos(start[2])])
    left1 = np.array([-math.sin(goal[2]), math.cos(goal[2])])
    for first in (1, -1):
        c1 = p + first * radius * left0
        for last in (1, -1):
            c3 = q + last * radius * left1
            delta = c3 - c1
            distance = float(np.linalg.norm(delta))
            if distance < 1e-10:
                continue
            side = (last - first) * radius
            if abs(side) > distance + 1e-10:
                continue
            heading = math.atan2(delta[1], delta[0]) - math.asin(np.clip(side / distance, -1., 1.))
            left = np.array([-math.sin(heading), math.cos(heading)])
            tangent1, tangent2 = c1 - first * radius * left, c3 - last * radius * left
            straight = float((tangent2 - tangent1) @ np.array([math.cos(heading), math.sin(heading)]))
            if straight < -1e-8:
                continue
            name = ('L' if first > 0 else 'R') + 'S' + ('L' if last > 0 else 'R')
            yield name, [(first / radius, _arc_length(c1, p, tangent1, first, radius)),
                         (0., max(0., straight)),
                         (last / radius, _arc_length(c3, tangent2, q, last, radius))]
        c3 = q + first * radius * left1
        delta = c3 - c1
        distance = float(np.linalg.norm(delta))
        if 1e-10 < distance <= 4 * radius + 1e-10:
            midpoint = (c1 + c3) / 2
            normal = np.array([-delta[1], delta[0]]) / distance
            height = math.sqrt(max(0., 4 * radius * radius - distance * distance / 4))
            for side in (-1., 1.):
                middle = midpoint + side * height * normal
                tangent1, tangent2 = (c1 + middle) / 2, (c3 + middle) / 2
                name = 'LRL' if first > 0 else 'RLR'
                yield name, [(first / radius, _arc_length(c1, p, tangent1, first, radius)),
                             (-first / radius, _arc_length(middle, tangent1, tangent2, -first, radius)),
                             (first / radius, _arc_length(c3, tangent2, q, first, radius))]


def _integrate(pose, curvature, distance):
    x, y, yaw = pose
    end_yaw = yaw + curvature * distance
    if abs(curvature) < 1e-12:
        return np.array([x + distance * math.cos(yaw), y + distance * math.sin(yaw), yaw])
    return np.array([x + (math.sin(end_yaw) - math.sin(yaw)) / curvature,
                     y + (math.cos(yaw) - math.cos(end_yaw)) / curvature, wrap(end_yaw)])


def dubins_paths(start_xy_yaw, goal_xy_yaw, vehicle, table, *, obstacles=(),
                  sample_m=.02, maximum_length_m=8.):
    """Return sampled feasible forward paths and explicit rejected family reasons.

    The sampling inflation bounds any footprint corner's displacement between a
    sample and its nearest path point for the ideal bounded-curvature bicycle.
    This is geometry screening, not a dynamic tracking or braking guarantee.
    """
    start = _array(start_xy_yaw, (3,), 'start_pose')
    goal = _array(goal_xy_yaw, (3,), 'goal_pose')
    sample_m = _positive(sample_m, 'path_sample_m')
    radius = vehicle['wheelbase_m'] / math.tan(vehicle['max_steer_rad'])
    corner_radius = math.hypot(max(vehicle['rear_overhang_m'], vehicle['wheelbase_m'] + vehicle['front_overhang_m']), vehicle['width_m'] / 2)
    swept_margin = sample_m / 2 * (1. + corner_radius / radius)
    feasible, rejected = [], []
    segments = list(_dubins_segments(start, goal, radius))
    if np.linalg.norm(start[:2] - goal[:2]) < 1e-9 and abs(wrap(start[2] - goal[2])) < 1e-9:
        segments.insert(0, ('stationary', []))
    for family, pieces in segments:
        length = sum(piece[1] for piece in pieces)
        if length > maximum_length_m:
            rejected.append({'family': family, 'reason': 'path_length_limit', 'length_m': length})
            continue
        pose, travelled = start.copy(), 0.
        samples = [{'x_m': float(pose[0]), 'y_m': float(pose[1]), 'yaw_rad': float(pose[2]), 'curvature_m_inv': 0., 's_m': 0.}]
        reason = None
        for curvature, distance in pieces:
            count = max(1, int(math.ceil(distance / sample_m)))
            for _ in range(count):
                ds = distance / count
                pose = _integrate(pose, curvature, ds)
                travelled += ds
                samples.append({'x_m': float(pose[0]), 'y_m': float(pose[1]), 'yaw_rad': float(pose[2]),
                                'curvature_m_inv': float(curvature), 's_m': travelled})
        if np.linalg.norm(pose[:2] - goal[:2]) > 1e-6 or abs(wrap(pose[2] - goal[2])) > 1e-6:
            rejected.append({'family': family, 'reason': 'analytic_endpoint_mismatch'})
            continue
        for sample in samples:
            clear, reason = footprint_clear([sample['x_m'], sample['y_m'], sample['yaw_rad']],
                                             table, vehicle, obstacles, swept_margin)
            if not clear:
                break
        if reason:
            rejected.append({'family': family, 'reason': reason, 's_m': sample['s_m']})
            continue
        feasible.append({'family': family, 'length_m': length,
                         'turning_rad': sum(abs(k) * d for k, d in pieces),
                         'minimum_turn_radius_m': radius,
                         'sampling_inflation_m': swept_margin,
                         'segments': [{'curvature_m_inv': k, 'length_m': d} for k, d in pieces],
                         'samples': samples})
    feasible.sort(key=lambda path: (path['length_m'], path['turning_rad']))
    return feasible, rejected


def _table_validate(table):
    if table.get('frame') != 'odom':
        raise ValueError('table_frame_must_be_odom')
    _array(table['center_xy_m'], (2,), 'table_center')
    half = _array(table['half_size_m'], (2,), 'table_half_size')
    if np.any(half <= 0) or not math.isfinite(float(table['yaw_rad'])):
        raise ValueError('invalid_table_geometry')
    if not math.isfinite(float(table['bottom_height_m'])) or not math.isfinite(float(table['top_height_m'])) or table['bottom_height_m'] >= table['top_height_m']:
        raise ValueError('table_must_have_positive_solid_volume')


def _camera_visible(point, pose, camera):
    transform = _array(pose, (4, 4), 'camera_pose')
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)
            or not np.allclose(transform[3], [0., 0., 0., 1.])):
        return False, 'camera_rotation_invalid', math.inf
    optical = rotation.T @ (np.asarray(point) - transform[:3, 3])
    x, y, z = optical
    if not camera['near_m'] <= z <= camera['far_m']:
        return False, 'target_outside_camera_depth_range', math.inf
    usable = 1. - 2 * camera['edge_margin_fraction']
    horizontal = abs(x / z) / (math.tan(math.radians(camera['horizontal_fov_deg']) / 2) * usable)
    vertical = abs(y / z) / (math.tan(math.radians(camera['vertical_fov_deg']) / 2) * usable)
    return max(horizontal, vertical) <= 1., 'target_outside_camera_fov', max(horizontal, vertical)


def _arm_screen(target_model, model_pose, table, cfg, ik_solver, geometry_check, camera_pose_fn):
    p, arm = cfg['planning'], cfg['arm']
    grasp = np.asarray(target_model) + [0., 0., arm['grasp_height_above_table_m']]
    waypoints = [('home', grasp + [0., 0., arm['approach_clearance_m']]),
                 ('grasp', grasp), ('lift', grasp + [0., 0., arm['lift_m']]),
                 ('lower', grasp + [0., 0., arm['lower_offset_m']])]
    ranges = np.asarray(arm['joint_limits_rad'])
    table_model = copy.deepcopy(table)
    table_model['frame'] = 'arm_model'
    table_model['center_xy_m'] = point_in_model([*table['center_xy_m'], 0.], model_pose)[:2].tolist()
    table_model['yaw_rad'] = wrap(table['yaw_rad'] - model_pose[2])
    minimum_margin, minimum_clearance, camera_error = math.inf, math.inf, 0.
    joint_path, seed, previous = [], None, None
    for phase, destination in waypoints:
        if previous is None:
            points = [destination]
        else:
            count = max(1, int(math.ceil(float(np.linalg.norm(destination - previous)) / p['arm_sample_m'])))
            points = [previous + (destination - previous) * index / count for index in range(1, count + 1)]
        for point in points:
            q, ok = ik_solver(point, seed=seed)
            q = _array(q, (len(ranges),), 'ik_joints')
            if not ok:
                return None, 'arm_ik_unreachable_' + phase
            margin = float(np.minimum(q - ranges[:, 0], ranges[:, 1] - q).min())
            if margin < p['minimum_joint_margin_rad']:
                return None, 'joint_margin_insufficient_' + phase
            if seed is not None and float(np.abs(q - seed).max()) > p['max_joint_step_rad']:
                return None, 'joint_branch_discontinuity_' + phase
            clearance = float(point[2] - arm['tcp_envelope_radius_m'] - table['top_height_m'])
            if clearance < p['minimum_tcp_clearance_m']:
                return None, 'tcp_envelope_below_table_clearance_' + phase
            if geometry_check is not None:
                evidence = geometry_check(q.copy(), point.copy(), phase, table_model)
                if isinstance(evidence, dict):
                    if evidence.get('valid') is not True:
                        return None, 'arm_geometry_' + str(evidence.get('reason', phase))
                    measured = evidence.get('clearance_m')
                    if measured is not None:
                        if not math.isfinite(float(measured)) or measured < 0:
                            return None, 'arm_geometry_clearance_invalid'
                        clearance = min(clearance, float(measured))
                elif evidence is not True:
                    return None, 'arm_geometry_rejected_' + phase
            if phase in cfg['camera']['check_phases']:
                camera_pose = camera_pose_fn(q.copy()) if camera_pose_fn is not None else cfg['camera']['fixed_pose_in_model']
                visible, reason, error = _camera_visible(target_model, camera_pose, cfg['camera'])
                if not visible:
                    return None, reason + '_' + phase
                camera_error = max(camera_error, float(error))
            joint_path.append({'phase': phase, 'point_model_m': point.tolist(), 'q_rad': q.tolist()})
            minimum_margin, minimum_clearance = min(minimum_margin, margin), min(minimum_clearance, clearance)
            seed = q.copy()
        previous = destination
    return {'waypoints': [{'phase': phase, 'point_model_m': point.tolist()} for phase, point in waypoints],
            'joint_path': joint_path, 'minimum_joint_margin_rad': minimum_margin,
            'minimum_clearance_m': minimum_clearance, 'camera_center_error': camera_error,
            'full_arm_geometry_checked': geometry_check is not None,
            'camera_pose_source': 'joint_pose_callback' if camera_pose_fn is not None else 'synthetic_fixed_pose_assumption'}, None


def plan_parking(target, table, odometry, cfg, ik_solver: Callable, *,
                 geometry_check=None, camera_pose_fn=None, now_s=None):
    """Rank a finite parking set and forward paths, returning all rejection reasons.

    ``ik_solver(point_model_m, seed=None) -> (q, possible)`` is dependency-injected.
    ``geometry_check(q, point, phase, table_in_model)`` must check complete arm/table
    geometry if that stronger claim is required.  Without it only joint and TCP
    envelope clearance is screened.  The model z origin matches the v2 BowlEnv;
    do not subtract its existing base_height_m a second time.
    """
    output = {'schema_version': 1, 'valid': False, 'selected': None, 'candidates': [],
              'rejections': [], 'hardware_output_enabled': False,
              'scope': 'finite_parking_set_and_forward_Dubins_candidates_not_global_optimum',
              'assumptions': ['unmeasured_simulation_vehicle_dimensions_and_control_limits',
                              'solid_rectangular_platform_and_circular_obstacles',
                              'ideal_bicycle_curvature_no_steering_rate_or_braking_proof',
                              'geometry_screening_does_not_prove_dynamic_grasp_success'],
              'full_arm_geometry_checked': geometry_check is not None}
    try:
        _validate_config(cfg)
        _table_validate(table)
        if odometry.get('valid') is not True or odometry.get('frame') != 'odom':
            raise ValueError('odometry_invalid_or_wrong_frame')
        start = _array(odometry['xy_yaw'], (3,), 'odometry_pose')
        now_s = float(odometry['stamp_s'] if now_s is None else now_s)
        age = now_s - float(odometry['stamp_s'])
        if not math.isfinite(now_s) or not -cfg['planning']['max_future_s'] <= age <= cfg['coordinator']['odometry_max_age_s']:
            raise ValueError('odometry_stale_or_future')
        good, reason = target_validation(target, now_s, cfg['planning']['max_source_age_s'], cfg['planning']['target_max_sigma_m'], future_s=cfg['planning']['max_future_s'])
        if not good:
            raise ValueError(reason)
        position = np.asarray(target['position_m'], dtype=float)
        if abs(position[2] - table['top_height_m']) > cfg['planning']['table_target_height_tolerance_m']:
            raise ValueError('target_bottom_height_disagrees_with_platform')
        local_target = point_in_model(position, [*table['center_xy_m'], table['yaw_rad']])
        if np.any(np.abs(local_target[:2]) > np.asarray(table['half_size_m'])):
            raise ValueError('target_outside_platform_top')
        for obstacle in cfg['planning']['obstacles']:
            for key in ('x_m', 'y_m', 'radius_m'):
                if not math.isfinite(float(obstacle[key])):
                    raise ValueError('invalid_obstacle')
            _positive(obstacle['radius_m'], 'obstacle_radius')
        heading = math.atan2(position[1] - start[1], position[0] - start[0])
        mount = np.asarray(cfg['arm']['mount_xy_yaw_m_rad'])
        candidate_id = 0
        for offset in cfg['planning']['heading_offsets_deg']:
            yaw = wrap(heading + math.radians(offset))
            for rx in cfg['planning']['relative_target_x_m']:
                for ry in cfg['planning']['relative_target_y_m']:
                    model_yaw = wrap(yaw + mount[2])
                    c, s = math.cos(model_yaw), math.sin(model_yaw)
                    model_xy = position[:2] - np.array([c * rx - s * ry, s * rx + c * ry])
                    cy, sy = math.cos(yaw), math.sin(yaw)
                    base_xy = model_xy - np.array([cy * mount[0] - sy * mount[1], sy * mount[0] + cy * mount[1]])
                    parking = np.array([*base_xy, yaw])
                    model_pose = compose_pose(parking, mount)
                    candidate = {'id': candidate_id, 'parking_xy_yaw': parking.tolist(),
                                 'arm_model_parking_xy_yaw': model_pose.tolist(), 'valid': False}
                    candidate_id += 1
                    clear, reason = footprint_clear(parking, table, cfg['vehicle'], cfg['planning']['obstacles'])
                    arm_report = None
                    if clear:
                        arm_report, reason = _arm_screen(point_in_model(position, model_pose), model_pose, table, cfg, ik_solver, geometry_check, camera_pose_fn)
                    if reason:
                        candidate['rejection_reason'] = reason
                    else:
                        paths, rejected = dubins_paths(start, parking, cfg['vehicle'], table,
                            obstacles=cfg['planning']['obstacles'], sample_m=cfg['planning']['path_sample_m'],
                            maximum_length_m=cfg['planning']['max_path_length_m'])
                        candidate['path_rejections'] = rejected
                        if not paths:
                            candidate['rejection_reason'] = 'no_collision_free_forward_Dubins_candidate'
                        else:
                            weights = cfg['planning']['score_weights']
                            for path in paths:
                                path['path_score'] = weights['path_length'] * path['length_m'] + weights['turning'] * path['turning_rad']
                            path = min(paths, key=lambda item: item['path_score'])
                            components = {'path_length': weights['path_length'] * path['length_m'],
                                'turning': weights['turning'] * path['turning_rad'],
                                'joint_margin': weights['joint_margin'] / max(.01, arm_report['minimum_joint_margin_rad']),
                                'clearance': weights['clearance'] / max(.005, arm_report['minimum_clearance_m']),
                                'centering': weights['centering'] * arm_report['camera_center_error']}
                            candidate.update(valid=True, score=sum(components.values()), score_components=components,
                                             arm=arm_report, arm_waypoints=arm_report['waypoints'],
                                             joint_path=arm_report['joint_path'], path=path, paths=paths,
                                             feasible_path_families=[{'family': item['family'], 'length_m': item['length_m']} for item in paths])
                    output['candidates'].append(candidate)
        feasible = [item for item in output['candidates'] if item['valid']]
        if feasible:
            selected = min(feasible, key=lambda item: (item['score'], item['id']))
            output.update(valid=True, selected=copy.deepcopy(selected), target_id=target['target_id'],
                          target_stamp_s=target['stamp_s'], target_source=target.get('source'),
                          target_position_m=position.tolist())
        else:
            output['rejections'].append('no_feasible_parking_candidate_in_configured_set')
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        output['rejections'].append(str(error))
    return output


def screen_grasp_at_parking(target, table, parking_xy_yaw, cfg, ik_solver, *,
                            geometry_check=None, camera_pose_fn=None, now_s=None):
    """Re-screen the wrist target at the current parked pose without moving it."""
    result = {'valid': False, 'reason': None, 'arm': None,
              'hardware_output_enabled': False}
    try:
        _validate_config(cfg)
        _table_validate(table)
        parking = _array(parking_xy_yaw, (3,), 'parking_pose')
        now_s = float(target['stamp_s'] if now_s is None else now_s)
        good, reason = target_validation(target, now_s, cfg['coordinator']['target_max_age_s'],
                                         cfg['coordinator']['wrist_max_sigma_m'], source='wrist')
        if not good:
            raise ValueError(reason)
        clear, reason = footprint_clear(parking, table, cfg['vehicle'], cfg['planning']['obstacles'])
        if not clear:
            raise ValueError(reason)
        position = np.asarray(target['position_m'], dtype=float)
        if abs(position[2] - table['top_height_m']) > cfg['planning']['table_target_height_tolerance_m']:
            raise ValueError('target_bottom_height_disagrees_with_platform')
        table_point = point_in_model(position, [*table['center_xy_m'], table['yaw_rad']])
        if np.any(np.abs(table_point[:2]) > np.asarray(table['half_size_m'])):
            raise ValueError('target_outside_platform_top')
        model_pose = compose_pose(parking, cfg['arm']['mount_xy_yaw_m_rad'])
        arm_report, reason = _arm_screen(point_in_model(position, model_pose), model_pose,
                                         table, cfg, ik_solver, geometry_check, camera_pose_fn)
        result.update(valid=arm_report is not None, reason=reason, arm=arm_report,
                      target_id=target['target_id'], target_stamp_s=target['stamp_s'],
                      parking_xy_yaw=parking.tolist(), arm_model_parking_xy_yaw=model_pose.tolist())
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        result['reason'] = str(error)
    return result


class MobileCoordinator:
    """Main-camera acquisition -> approach -> stop -> wrist reacquire -> ready.

    Returns intents only.  The external simulation/controller must follow the
    path and supply measured feedback.  A fault latches STOP until reset().
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.state, self.reason = 'acquire_main', None
        self.plan, self.target_id = None, None
        self.last_stamp, self.quiet_since, self.stopped_at = None, None, None
        self.wrist_reference = None
        self.coarse_position = None

    def select_plan(self, plan):
        if plan.get('valid') is not True or not plan.get('selected') or not plan.get('target_id'):
            raise ValueError('cannot_select_invalid_mobile_plan')
        self.plan, self.target_id = copy.deepcopy(plan['selected']), plan['target_id']
        self.coarse_position = copy.deepcopy(plan.get('target_position_m'))
        self.state, self.reason = 'approach', None
        self.quiet_since = self.stopped_at = None

    def _stop(self, reason):
        self.state, self.reason = 'stop', reason
        return self.snapshot()

    def snapshot(self):
        moving = self.state == 'approach'
        return {'state': self.state, 'reason': self.reason,
                'base_intent': 'follow_forward_path' if moving else 'hold_stationary',
                'arm_intent': 'grasp_ready' if self.state == 'grasp_ready' else 'hold_observation_pose',
                'vision_intent': 'wrist_precise_localization' if self.state in ('acquire_wrist', 'grasp_ready') else 'main_camera_coarse_localization',
                'hardware_output_enabled': False, 'grasp_ready': self.state == 'grasp_ready',
                'precise_target_requires_arm_recheck': self.state == 'acquire_wrist',
                'parking_xy_yaw': None if self.plan is None else self.plan['parking_xy_yaw'],
                'wrist_target': copy.deepcopy(self.wrist_reference),
                'stopped_at_s': self.stopped_at}

    def update(self, now_s, odometry, *, main_target=None, wrist_target=None,
               obstacle_clear=True, arm_stowed=True, precise_grasp_feasible=False):
        if self.state == 'stop':
            return self.snapshot()
        gates = self.cfg['coordinator']
        try:
            if odometry.get('valid') is not True or odometry.get('frame') != 'odom':
                return self._stop('odometry_invalid_or_wrong_frame')
            pose = _array(odometry['xy_yaw'], (3,), 'odometry_pose')
            stamp = float(odometry['stamp_s'])
            speed, angular = float(odometry['speed_m_s']), float(odometry['yaw_rate_rad_s'])
            if not all(math.isfinite(v) for v in (now_s, stamp, speed, angular)) or not -.02 <= now_s - stamp <= gates['odometry_max_age_s']:
                return self._stop('odometry_stale_or_future')
            if self.last_stamp is not None:
                if stamp <= self.last_stamp:
                    return self._stop('odometry_nonmonotonic')
                if stamp - self.last_stamp > gates['maximum_sample_gap_s']:
                    return self._stop('odometry_sampling_gap')
            self.last_stamp = stamp
            if obstacle_clear is not True:
                return self._stop('approach_path_obstructed_or_unknown')
            if self.state == 'acquire_main':
                good, reason = target_validation(main_target or {}, now_s, gates['target_max_age_s'], gates['main_max_sigma_m'], source='main_rgbd')
                result = self.snapshot()
                result.update(plan_requested=good, reason=reason)
                return result
            goal = np.asarray(self.plan['parking_xy_yaw'])
            arrived = np.linalg.norm(pose[:2] - goal[:2]) <= gates['parking_position_tolerance_m'] and abs(wrap(pose[2] - goal[2])) <= gates['parking_yaw_tolerance_rad']
            quiet = abs(speed) <= gates['stop_speed_m_s'] and abs(angular) <= gates['stop_yaw_rate_rad_s']
            if self.state == 'approach':
                if arm_stowed is not True:
                    return self._stop('arm_not_in_validated_navigation_pose')
                good, reason = target_validation(main_target or {}, now_s, gates['target_max_age_s'], gates['main_max_sigma_m'], source='main_rgbd')
                if not good:
                    return self._stop(reason)
                if main_target['target_id'] != self.target_id:
                    return self._stop('coarse_target_identity_changed')
                if self.coarse_position is not None and np.linalg.norm(np.asarray(main_target['position_m']) - self.coarse_position) > gates['coarse_replan_distance_m']:
                    return self._stop('coarse_target_moved_replan_required')
                if arrived:
                    self.state = 'stop_confirm'
            if self.state == 'stop_confirm':
                if not arrived:
                    return self._stop('parking_pose_drift_during_stop')
                if quiet:
                    self.quiet_since = stamp if self.quiet_since is None else self.quiet_since
                    if stamp - self.quiet_since >= gates['stop_dwell_s']:
                        self.state, self.stopped_at = 'acquire_wrist', stamp
                else:
                    self.quiet_since = None
            elif self.state in ('acquire_wrist', 'grasp_ready'):
                if not arrived or not quiet:
                    return self._stop('base_moved_during_wrist_localization_or_grasp')
                good, reason = target_validation(wrist_target or {}, now_s, gates['target_max_age_s'], gates['wrist_max_sigma_m'], source='wrist')
                if not good:
                    if self.state == 'grasp_ready':
                        return self._stop(reason)
                    result = self.snapshot()
                    result['reason'] = reason
                    return result
                if wrist_target['target_id'] != self.target_id:
                    return self._stop('precise_target_identity_changed')
                if gates['wrist_after_stop_required'] and wrist_target['stamp_s'] < self.stopped_at:
                    result = self.snapshot()
                    result['reason'] = 'wrist_observation_predates_confirmed_stop'
                    return result
                self.wrist_reference = copy.deepcopy(wrist_target)
                # A fresh precise detection can move the target outside the old
                # coarse plan's safe arm path.  The caller must re-screen it.
                if precise_grasp_feasible is not True:
                    if self.state == 'grasp_ready':
                        return self._stop('precise_arm_path_no_longer_verified')
                    result = self.snapshot()
                    result['reason'] = 'precise_target_arm_recheck_required'
                    return result
                self.state = 'grasp_ready'
            return self.snapshot()
        except (KeyError, ValueError, TypeError, AttributeError):
            return self._stop('invalid_mobile_observation')
