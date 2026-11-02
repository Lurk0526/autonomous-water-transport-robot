#!/usr/bin/env python3
"""Observe perception outputs, save previews and check current/stale behavior."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.road_geometry import image_array


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=15)
    p.add_argument('--prefix', default='/road_perception')
    p.add_argument('--output', type=Path, default=ROOT / 'work/road_perception/check')
    p.add_argument('--expect-stale', action='store_true')
    p.add_argument('--sim-time', action='store_true')
    p.add_argument('--clock-topic', default='/clock')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    ros_args = ['--ros-args', '-p', 'use_sim_time:=true', '-r', '/clock:=' + a.clock_topic] if a.sim_time else None
    rclpy.init(args=ros_args)
    node = rclpy.create_node('road_perception_check')
    latest, counts, stamps, statuses, received = {}, {}, {}, [], {}

    def receive(key, message):
        latest[key] = message
        received[key] = time.monotonic()
        counts[key] = counts.get(key, 0) + 1
        if key == 'status':
            statuses.append(json.loads(message.data))
        else:
            value = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
            stamps.setdefault(key, []).append(value)

    prefix = a.prefix.rstrip('/')
    subscriptions = [node.create_subscription(Image, prefix + '/' + name, lambda m, k=name: receive(k, m), qos_profile_sensor_data)
                     for name in ['mask', 'confidence', 'overlay', 'aligned_depth', 'depth_preview', 'paved_candidate']]
    subscriptions += [node.create_subscription(PointCloud2, prefix + '/' + name, lambda m, k=name: receive(k, m), qos_profile_sensor_data)
                      for name in ['labeled_points', 'obstacle_points']]
    subscriptions.append(node.create_subscription(String, prefix + '/status', lambda m: receive('status', m), 1))
    begin = time.monotonic()
    while time.monotonic() - begin < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    errors = []
    status = statuses[-1] if statuses else {}
    report = {'counts': counts, 'last_status': status, 'expect_stale': a.expect_stale, 'duration_seconds': a.seconds}
    report['valid_status_count'] = sum(bool(s.get('valid')) for s in statuses)
    report['fusion_status_count'] = sum(bool(s.get('fusion_valid')) for s in statuses)
    if 'status' in received and time.monotonic()-received['status'] > 2:
        errors.append('Diagnostic status stopped arriving')
    if status and not a.expect_stale and not -0.1 <= node.get_clock().now().nanoseconds*1e-9-status.get('status_stamp', 0) <= 2:
        errors.append('Diagnostic status timestamp is not current')
    for key in ['mask', 'confidence', 'overlay', 'aligned_depth', 'depth_preview', 'paved_candidate', 'labeled_points', 'status']:
        if key not in latest:
            errors.append('No ' + key + ' output')
    if not status:
        errors.append('No diagnostic status')
    elif a.expect_stale:
        if status.get('valid') or status.get('fusion_valid'):
            errors.append('Old input still marked valid')
        if not report['fusion_status_count']:
            errors.append('No valid RGB/depth output preceded the stale-input state')
    elif not status.get('valid') or not status.get('fusion_valid'):
        errors.append('Current RGB/depth not valid: ' + status.get('fusion_reason', ''))
    if status.get('drive_output_enabled') is not False:
        errors.append('Unexpected drive-output state')
    if status.get('mount_ready'):
        errors.append('Bench configuration unexpectedly reports mounted geometry')
    if 'mask' in latest:
        mask = image_array(latest['mask'])
        report['mask_classes'] = np.unique(mask).tolist()
        if a.expect_stale and np.any(mask != 255):
            errors.append('Stale semantic mask not cleared to unknown')
        if not a.expect_stale and (mask > 6).any():
            errors.append('Unexpected semantic label')
        cv2.imwrite(str(a.output / 'mask.png'), mask)
    for key in ['overlay', 'depth_preview', 'aligned_depth', 'paved_candidate', 'confidence']:
        if key not in latest:
            continue
        value = image_array(latest[key])
        if key in ['overlay', 'depth_preview']:
            cv2.imwrite(str(a.output / (key + '.jpg')), cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
            if key == 'depth_preview' and a.expect_stale and value.any():
                errors.append('Stale depth preview not cleared')
        elif key == 'aligned_depth':
            np.save(a.output / 'aligned-depth-metres.npy', value)
            valid = np.isfinite(value) & (value > 0)
            report['aligned_valid_fraction'] = float(valid.mean())
            view = cv2.applyColorMap(np.uint8(np.clip(np.nan_to_num(value) / 5 * 255, 0, 255)), cv2.COLORMAP_TURBO)
            view[~valid] = 0
            cv2.imwrite(str(a.output / 'aligned-depth.jpg'), view)
            if a.expect_stale and valid.any():
                errors.append('Stale aligned depth not cleared')
            if not a.expect_stale and not valid.any():
                errors.append('No aligned depth samples')
        elif key == 'confidence':
            if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1.0001:
                errors.append('Invalid confidence values')
            if a.expect_stale and value.any():
                errors.append('Stale confidence not cleared')
        else:
            cv2.imwrite(str(a.output / 'paved-candidate.png'), value)
            if a.expect_stale and value.any():
                errors.append('Stale candidate mask not cleared')
    if a.expect_stale and 'labeled_points' in latest and latest['labeled_points'].width:
        errors.append('Stale point cloud not cleared')
    if 'labeled_points' in latest:
        cloud = latest['labeled_points']
        report['cloud'] = {'points': cloud.width, 'frame': cloud.header.frame_id, 'point_step': cloud.point_step,
                           'fields': [f.name for f in cloud.fields]}
    report['output_rates_hz'] = {k: round((len(v) - 1) / (v[-1] - v[0]), 2)
                                 for k, v in stamps.items() if len(v) > 1 and v[-1] > v[0]}
    report['errors'], report['passed'] = errors, not errors
    (a.output / 'health.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
