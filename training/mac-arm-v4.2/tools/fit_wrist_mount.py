"""Estimate a fixed wrist-camera mount correction from measured board views.

Input JSON is an array of views. Each view has a unique id, split ("fit" or
"check"), points_odom_m, pixels (undistorted xy), K, and
T_nominal_optical_to_odom from the measured arm posture and nominal mount.
Points must lie on a horizontal, independently surveyed plane.

The output delta_optical is right-multiplied onto the nominal transform.
The check-plane errors quantify this data set and its tested volume.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def correction(parameters):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    matrix[:3, 3] = parameters[3:]
    return matrix


def unpack(views):
    records = []
    identifiers = set()
    for view in views:
        view_id = view['id']
        if view_id in identifiers:
            raise ValueError('Duplicate view id')
        identifiers.add(view_id)
        k = np.asarray(view['K'], dtype=float)
        nominal = np.asarray(view['T_nominal_optical_to_odom'], dtype=float)
        points = np.asarray(view['points_odom_m'], dtype=float)
        pixels = np.asarray(view['pixels'], dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 6
                or pixels.shape != (len(points), 2) or k.shape != (3, 3)
                or nominal.shape != (4, 4)
                or not all(np.isfinite(a).all() for a in (points, pixels, k, nominal))):
            raise ValueError(f'Invalid calibration observations in view {view_id}')
        rotation = nominal[:3, :3]
        if (not np.allclose(nominal[3], [0, 0, 0, 1])
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(rotation), 1)
                or not np.allclose(k[2], [0, 0, 1]) or min(k[0, 0], k[1, 1]) <= 0):
            raise ValueError(f'Invalid intrinsics or transform in view {view_id}')
        board_xy = points[:, :2]
        if np.ptp(points[:, 2]) > .0001 or np.linalg.matrix_rank(board_xy - board_xy.mean(0)) < 2:
            raise ValueError(f'View {view_id} requires a horizontal plane with noncollinear landmarks')
        if np.min((points - nominal[:3, 3]) @ rotation[:, 2]) <= .05:
            raise ValueError(f'Board in view {view_id} is too close to the camera or behind it')
        if view['split'] not in ('fit', 'check'):
            raise ValueError(f'Unknown split for view {view_id}')
        records.append((view['split'], k, nominal, points, pixels))
    return records


def reprojection_errors(record, delta):
    _, k, nominal, points, pixels = record
    calibrated = nominal @ delta
    optical = (points - calibrated[:3, 3]) @ calibrated[:3, :3]
    projected = optical @ k.T
    return projected[:, :2] / np.maximum(projected[:, 2, None], 1e-6) - pixels


def summarize(records, delta):
    ground_errors = []
    image_errors = []
    p95_per_view = []
    for record in records:
        _, k, nominal, points, pixels = record
        calibrated = nominal @ delta
        optical_rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(k).T
        rays = optical_rays @ calibrated[:3, :3].T
        if np.any(np.abs(rays[:, 2]) < .05):
            raise ValueError('Rays too close to parallel with surveyed plane')
        distance = (points[:, 2] - calibrated[2, 3]) / rays[:, 2]
        if np.any(distance <= 0):
            raise ValueError('Board intersects estimated rays behind the camera')
        estimated = calibrated[:3, 3] + rays * distance[:, None]
        error_mm = 1000 * np.linalg.norm(estimated[:, :2] - points[:, :2], axis=1)
        ground_errors.extend(error_mm)
        image_errors.extend(np.linalg.norm(reprojection_errors(record, delta), axis=1))
        p95_per_view.append(float(np.quantile(error_mm, .95)))
    return {
        'views': len(records), 'points': len(ground_errors),
        'rms_pixels': float(np.sqrt(np.mean(np.square(image_errors)))),
        'median_xy_mm': float(np.median(ground_errors)),
        'p95_xy_mm': float(np.quantile(ground_errors, .95)),
        'worst_view_p95_xy_mm': max(p95_per_view),
        'max_xy_mm': max(ground_errors),
    }


def fit(views):
    data = unpack(views)
    fitting = [record for record in data if record[0] == 'fit']
    checking = [record for record in data if record[0] == 'check']
    if len(fitting) < 3 or len(checking) < 2:
        raise ValueError('Supply at least three fit poses and two separate check poses')
    for i, (_, _, pose, _, _) in enumerate(data):
        for _, _, earlier, _, _ in data[:i]:
            distance = np.linalg.norm(pose[:3, 3] - earlier[:3, 3])
            angle = Rotation.from_matrix(pose[:3, :3].T @ earlier[:3, :3]).magnitude()
            if distance < .005 and angle < .02:
                raise ValueError('Calibration needs distinct camera poses')
    bounds = np.r_[np.repeat(.16, 3), np.repeat(.025, 3)]

    def residuals(parameters):
        delta = correction(parameters)
        return np.concatenate([reprojection_errors(row, delta).ravel() for row in fitting])

    solution = least_squares(residuals, np.zeros(6), bounds=(-bounds, bounds),
                             x_scale='jac', loss='soft_l1', f_scale=.75)
    delta = correction(solution.x)
    singular_values = np.linalg.svd(solution.jac * bounds, compute_uv=False)
    condition = float(singular_values[0] / max(1e-12, singular_values[-1]))
    before = summarize(checking, np.eye(4))
    after = summarize(checking, delta)
    usable = bool(solution.success and np.linalg.matrix_rank(solution.jac) == 6
                  and condition < 500 and not np.any(solution.active_mask)
                  and after['worst_view_p95_xy_mm'] <= 1. and after['max_xy_mm'] <= 2.)
    return {
        'delta_optical': delta.tolist(),
        'parameter_order': 'rotvec radians, translation metres; right-multiply T_nominal',
        'fit': summarize(fitting, delta),
        'check_before': before,
        'check_after': after,
        'scaled_jac_condition': condition,
        'usable_in_tested_volume': usable,
        'criteria': 'per-check-view 95th percentile <=1 mm and maximum <=2 mm; provisional engineering limits',
        'scope': 'Surveyed stationary horizontal planes, undistorted intrinsics, fixed mount, accurate nominal kinematics',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('views', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = fit(json.loads(args.views.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['check_after'])
    raise SystemExit(0 if result['usable_in_tested_volume'] else 2)
