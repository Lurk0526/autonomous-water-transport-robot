#!/usr/bin/python3
"""Check real ROS camera streams, calibration, TF, freshness and obstacle scan."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2
from tf2_ros import Buffer, TransformListener


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=15.0)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'work/gemini')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node('gemini_camera_check')
    counts, stamps, latest, received = {}, {}, {}, {}
    buffer = Buffer()
    listener = TransformListener(buffer, node)

    def receive(key, msg):
        counts[key] = counts.get(key, 0) + 1
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        stamps.setdefault(key, []).append(stamp)
        latest[key] = msg
        received[key] = time.time()

    streams = {
        'rgb': (Image, '/camera/color/image_raw'),
        'depth': (Image, '/camera/depth/image_raw'),
        'rectified_depth': (Image, '/camera/depth/image_rect_raw'),
        'rgb_info': (CameraInfo, '/camera/color/camera_info'),
        'depth_info': (CameraInfo, '/camera/depth/camera_info'),
        'points': (PointCloud2, '/camera/depth/points'),
        'scan': (LaserScan, '/camera/scan'),
    }
    subs = [node.create_subscription(cls, topic, lambda msg, k=key: receive(k, msg),
                                     qos_profile_sensor_data)
            for key, (cls, topic) in streams.items()]
    start = time.monotonic()
    while time.monotonic() - start < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    errors, report = [], {'streams': {}}
    for key, (_, topic) in streams.items():
        seq = stamps.get(key, [])
        hz = (len(seq) - 1) / (seq[-1] - seq[0]) if len(seq) > 1 and seq[-1] > seq[0] else 0
        msg = latest.get(key)
        report['streams'][key] = {'topic': topic, 'messages': counts.get(key, 0),
                                  'hz': round(hz, 2), 'frame': msg.header.frame_id if msg else None}
        if len(seq) < 3:
            errors.append(f'{key}: missing or insufficient messages')
        elif hz < 10:
            errors.append(f'{key}: rate below 10 Hz ({hz:.2f})')
        if seq and (not 0 <= time.time() - seq[-1] < 2.0 or time.time() - received[key] >= 2.0):
            errors.append(f'{key}: stale or future timestamp')
    for key in ('rgb', 'depth'):
        image, info = latest.get(key), latest.get(key + '_info')
        if image is None or info is None:
            continue
        report[key + '_calibration'] = {
            'image_size': [image.width, image.height], 'info_size': [info.width, info.height],
            'encoding': image.encoding, 'K': list(info.k), 'D': list(info.d),
            'matching_timestamps': len(set(stamps[key]) & set(stamps[key + '_info'])),
        }
        if (image.width, image.height) != (info.width, info.height) or info.k[0] <= 0 or info.k[4] <= 0:
            errors.append(f'{key}: invalid calibration or size mismatch')
        if image.header.frame_id != info.header.frame_id or not info.header.frame_id:
            errors.append(f'{key}: camera-info frame mismatch')
        if not set(stamps[key]) & set(stamps[key + '_info']):
            errors.append(f'{key}: image and camera-info timestamps do not match')
    if 'depth' in latest:
        d = latest['depth']
        depth = np.ndarray((d.height, d.width), dtype='<u2', buffer=bytes(d.data), strides=(d.step, 2)).copy()
        valid = depth > 0
        report['depth_measurements'] = {'valid_fraction': float(valid.mean()),
                                      'median_m': float(np.median(depth[valid]) / 1000) if valid.any() else None}
        if not valid.any():
            errors.append('depth: all pixels invalid; check scene and camera distance')
        cv2.imwrite(str(args.output / 'depth_mm.png'), depth)
        depth_view = cv2.applyColorMap(np.uint8(np.clip(depth / 5000 * 255, 0, 255)), cv2.COLORMAP_TURBO)
        depth_view[~valid] = 0
        cv2.imwrite(str(args.output / 'depth_preview.jpg'), depth_view)
    if 'rectified_depth' in latest:
        d = latest['rectified_depth']
        rectified = np.ndarray((d.height, d.width), dtype='<u2', buffer=bytes(d.data), strides=(d.step, 2)).copy()
        valid = rectified > 0
        report['rectified_depth_measurements'] = {'valid_fraction': float(valid.mean()),
                                                  'range_0_3_to_5_m_fraction': float(((rectified >= 300) & (rectified <= 5000)).mean()),
                                                  'median_m': float(np.median(rectified[valid]) / 1000) if valid.any() else None}
        cv2.imwrite(str(args.output / 'rectified_depth_mm.png'), rectified)
    if 'rgb' in latest:
        m = latest['rgb']
        rgb = np.ndarray((m.height, m.width, 3), dtype=np.uint8, buffer=bytes(m.data), strides=(m.step, 3, 1))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if m.encoding == 'rgb8' else rgb
        cv2.imwrite(str(args.output / 'rgb.jpg'), bgr)
    if 'scan' in latest:
        scan = latest['scan']
        values = np.asarray(scan.ranges)
        finite = values[np.isfinite(values)]
        report['scan_measurements'] = {'bins': len(values), 'finite_bins': len(finite),
                                       'min_m': float(finite.min()) if finite.size else None,
                                       'fov_degrees': float(np.degrees(scan.angle_max - scan.angle_min))}
        if not finite.size:
            errors.append('scan: no obstacle returns within configured range')
    try:
        tf = buffer.lookup_transform('camera_link', 'camera_depth_optical_frame', rclpy.time.Time())
        report['camera_tf'] = {'ok': True, 'parent': tf.header.frame_id, 'child': tf.child_frame_id}
    except Exception as exc:
        report['camera_tf'] = {'ok': False, 'error': str(exc)}
        errors.append('camera optical TF is missing')
    report['errors'] = errors
    report['passed'] = not errors
    result = json.dumps(report, ensure_ascii=False, indent=2)
    (args.output / 'health.json').write_text(result + '\n')
    print(result)
    node.destroy_node()
    rclpy.shutdown()
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
