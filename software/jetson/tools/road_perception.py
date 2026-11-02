#!/usr/bin/env python3
"""Latest-frame GPU semantic inference and calibrated RGB/depth association.

Only image, point cloud and diagnostic publishers are created. Installation
geometry remains disabled until an actual mount TF and ground height exist.
"""
import argparse
from collections import deque
import copy
import json
import os
import signal
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformListener
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.road_geometry import (Calibration, associate_depth, depth_metres, height_obstacles,
                                  image_array, is_fresh, nearest_stamp, rotation_matrix)
from vision.road_inference import RoadPredictor
from vision.road_model import LABELS


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def save_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def image_message(value, encoding, header):
    value = np.ascontiguousarray(value)
    message = Image()
    message.header = copy.deepcopy(header)
    message.height, message.width = value.shape[:2]
    message.encoding = encoding
    message.is_bigendian = False
    message.step = value.strides[0]
    message.data = value.tobytes()
    return message


def point_message(xyz, labels, confidence, header, palette, max_points=50000):
    count = len(xyz)
    stride = max(1, int(np.ceil(count / max_points)))
    xyz, labels, confidence = xyz[::stride], labels[::stride], confidence[::stride]
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4'),
                      ('semantic_id', 'u1'), ('padding', 'u1', 3), ('confidence', '<f4')])
    values = np.zeros(len(xyz), dtype=dtype)
    if len(xyz):
        values['x'], values['y'], values['z'] = xyz.T
        colors = palette[np.minimum(labels, 6)].copy()
        colors[labels == 255] = [125, 125, 125]
        colors = colors.astype(np.uint32)
        values['rgb'] = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
        values['semantic_id'], values['confidence'] = labels, confidence
    message = PointCloud2()
    message.header = copy.deepcopy(header)
    message.height, message.width = 1, len(values)
    message.is_dense, message.is_bigendian = True, False
    message.point_step, message.row_step = dtype.itemsize, len(values) * dtype.itemsize
    message.fields = [PointField(name=n, offset=o, datatype=d, count=1) for n, o, d in
                      [('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('rgb', 12, 7), ('semantic_id', 16, 2), ('confidence', 20, 7)]]
    message.data = values.tobytes()
    return message


def calibration(message, rectified=False):
    if message is None:
        raise ValueError('CameraInfo not received')
    if rectified:
        if not np.allclose(np.array(message.r).reshape(3, 3), np.eye(3), atol=1e-6):
            raise ValueError('Non-identity depth rectification rotation is unsupported')
        k = np.array(message.p).reshape(3, 4)[:, :3]
        d = []
    else:
        k, d = message.k, message.d
    return Calibration(message.width, message.height, k, d, message.header.frame_id,
                       message.distortion_model or 'plumb_bob').validate()


def transform_matrix(message):
    t = np.eye(4)
    r, v = message.transform.rotation, message.transform.translation
    t[:3, :3] = rotation_matrix([r.x, r.y, r.z, r.w])
    t[:3, 3] = [v.x, v.y, v.z]
    return t


class RoadNode(Node):
    def __init__(self, config):
        super().__init__('road_perception')
        self.cfg = config
        if not 0 < config['max_rate_hz'] <= 30 or not 0 < config['max_age_seconds'] <= 5:
            raise ValueError('Invalid rate/age configuration')
        if not 0 <= config['confidence_min'] <= 1 or not 0 < config['sync_slop_seconds'] <= 0.2:
            raise ValueError('Invalid confidence or synchronization configuration')
        if config['mount']['enabled'] and config['mount']['ground_z_m'] is None:
            raise ValueError('Mount geometry requires measured ground_z_m')
        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        self.predictor = RoadPredictor(resolve(config['deployment']), config['backend'], config['input_rotation_degrees'])
        self.palette = np.array(LABELS['palette'], dtype=np.uint8)
        self.latest = None
        self.depth_cache = deque(maxlen=45)
        self.color_info = self.depth_info = None
        self.previous_clock = 0.0
        self.last_result = self.last_input_wall = 0.0
        self.next_process_wall = 0.0
        self.last_source_stamp = 0.0
        self.last_shape = (480, 640)
        self.last_frame = 'camera_color_optical_frame'
        self.invalidated = True
        self.last_invalidation_wall = 0.0
        self.processed = self.replaced = self.errors = 0
        self.rgb_valid = self.fusion_valid = self.mount_ready = False
        self.reason = 'Waiting for RGB image'
        self.fusion_reason = 'Waiting for synchronized depth/calibration/TF'
        self.statistics = {}
        self.runtime_dir = resolve(config['runtime_dir'])
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_written = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        camera = config['camera_prefix'].rstrip('/')
        output = config['output_prefix'].rstrip('/')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.subs = [
            self.create_subscription(Image, camera + '/color/image_raw', self.receive_rgb, qos),
            self.create_subscription(Image, camera + '/depth/image_rect_raw', self.receive_depth, qos),
            self.create_subscription(CameraInfo, camera + '/color/camera_info', self.receive_color_info, qos),
            self.create_subscription(CameraInfo, camera + '/depth/camera_info', self.receive_depth_info, qos),
        ]
        self.images = {name: self.create_publisher(Image, output + '/' + name, qos) for name in
                       ['mask', 'confidence', 'overlay', 'aligned_depth', 'depth_preview', 'paved_candidate']}
        self.cloud = self.create_publisher(PointCloud2, output + '/labeled_points', qos)
        self.obstacles = self.create_publisher(PointCloud2, output + '/obstacle_points', qos)
        self.status = self.create_publisher(String, output + '/status', 1)
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        # Poll lightly; rate-limit actual inference below. A slow frame must not
        # consume the next full inference period while callbacks refill latest.
        self.timer = self.create_timer(min(0.02, 1 / config['max_rate_hz']), self.process, clock=self.steady_clock)
        self.watchdog = self.create_timer(0.5, self.health, clock=self.steady_clock)
        self.run_info = {'pid': os.getpid(), 'started_wall_time': time.time(), 'model_sha256': self.predictor.manifest['model_sha256'], 'backend': self.predictor.backend,
                         'fallback_reason': self.predictor.fallback_reason, 'torch': torch.__version__,
                         'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0), 'config': config,
                         'note': 'Camera-relative perception. Paved candidate is not permission to drive.'}
        save_json(self.runtime_dir / 'runtime.json', self.run_info)
        self.get_logger().info(f"Road model ready: {self.predictor.backend}, rotation={config['input_rotation_degrees']}°, mount_enabled={config['mount']['enabled']}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def receive_rgb(self, message):
        if self.latest is not None:
            self.replaced += 1
        self.latest = message
        self.last_input_wall = time.monotonic()

    def receive_depth(self, message):
        self.depth_cache.append((stamp(message), message))

    def receive_color_info(self, message):
        self.color_info = message

    def receive_depth_info(self, message):
        self.depth_info = message

    def empty_clouds(self, header):
        empty = point_message(np.empty((0, 3)), np.empty(0, np.uint8), np.empty(0, np.float32), header, self.palette)
        self.cloud.publish(empty)
        self.obstacles.publish(empty)

    def clear_fusion(self, header):
        self.images['aligned_depth'].publish(image_message(np.full(self.last_shape, np.nan, np.float32), '32FC1', header))
        self.images['paved_candidate'].publish(image_message(np.zeros(self.last_shape, np.uint8), 'mono8', header))
        self.images['depth_preview'].publish(image_message(np.zeros((*self.last_shape, 3), np.uint8), 'rgb8', header))
        self.empty_clouds(header)

    def invalidate(self, reason):
        self.reason = reason
        self.rgb_valid = self.fusion_valid = self.mount_ready = False
        if not self.invalidated or time.monotonic() - self.last_invalidation_wall >= 1:
            header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.last_frame)
            self.images['mask'].publish(image_message(np.full(self.last_shape, 255, np.uint8), 'mono8', header))
            self.images['confidence'].publish(image_message(np.zeros(self.last_shape, np.float32), '32FC1', header))
            blank = np.zeros((*self.last_shape, 3), np.uint8)
            cv2.putText(blank, 'NO FRESH VALID INPUT', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 80, 60), 2)
            self.images['overlay'].publish(image_message(blank, 'rgb8', header))
            self.clear_fusion(header)
            self.last_invalidation_wall = time.monotonic()
        self.invalidated = True

    def process(self):
        if time.monotonic() < self.next_process_wall:
            return
        now = self.now()
        if now < self.previous_clock - 0.1:
            self.latest = None
            self.depth_cache.clear()
            self.invalidate('ROS clock moved backwards; input buffers cleared')
        self.previous_clock = now
        message, self.latest = self.latest, None
        if message is None:
            return
        self.next_process_wall = time.monotonic() + 1 / self.cfg['max_rate_hz']
        start = time.perf_counter()
        try:
            source_stamp = stamp(message)
            if not is_fresh(now, source_stamp, self.cfg['max_age_seconds']):
                self.invalidate('RGB source timestamp is stale or in the future')
                return
            if message.encoding not in ('rgb8', 'bgr8') or not message.header.frame_id:
                raise ValueError('RGB image encoding/frame is invalid')
            rgb = image_array(message)
            self.last_shape, self.last_frame = rgb.shape[:2], message.header.frame_id
            labels, confidence, inference_ms = self.predictor.predict(rgb)
            if not is_fresh(self.now(), source_stamp, self.cfg['max_age_seconds']):
                self.invalidate('Inference finished after the source age limit')
                return
            colors = self.palette[labels].copy()
            colors[confidence < self.cfg['confidence_min']] = [125, 125, 125]
            overlay = np.rint(rgb * 0.55 + colors * 0.45).astype(np.uint8)
            header = message.header
            self.images['mask'].publish(image_message(labels, 'mono8', header))
            self.images['confidence'].publish(image_message(confidence, '32FC1', header))
            self.images['overlay'].publish(image_message(overlay, 'rgb8', header))
            self.rgb_valid, self.invalidated = True, False
            self.reason = 'RGB inference current'
            self.fusion_valid = self.mount_ready = False
            try:
                self.fuse(message, labels, confidence)
                self.fusion_valid = True
                self.fusion_reason = 'Factory camera geometry associated; field alignment remains to be measured'
            except Exception as exc:
                self.fusion_reason = str(exc)
                self.statistics.update({'aligned_valid_fraction': 0.0, 'depth_points': 0,
                                        'associated_points': 0, 'candidate_pixels': 0, 'rgb_depth_delta_ms': None})
                self.clear_fusion(header)
            self.processed += 1
            self.last_result = time.monotonic()
            self.last_source_stamp = source_stamp
            self.statistics.update({'inference_ms': round(inference_ms, 2),
                                    'callback_ms': round((time.perf_counter() - start) * 1000, 2),
                                    'source_age_ms': round((self.now() - source_stamp) * 1000, 2)})
        except Exception as exc:
            self.errors += 1
            self.invalidate(str(exc))
            if self.errors <= 3 or self.errors % 50 == 0:
                self.get_logger().error(f'Frame rejected: {exc}')

    def fuse(self, rgb, labels, confidence):
        pair = nearest_stamp(self.depth_cache, stamp(rgb), self.cfg['sync_slop_seconds'])
        if pair is None:
            raise ValueError('No depth frame within synchronization tolerance')
        depth_stamp, depth_msg = pair
        if not is_fresh(self.now(), depth_stamp, self.cfg['max_age_seconds'] + self.cfg['sync_slop_seconds']):
            raise ValueError('Depth source is stale')
        cc, dc = calibration(self.color_info), calibration(self.depth_info, rectified=True)
        if cc.frame != rgb.header.frame_id or dc.frame != depth_msg.header.frame_id:
            raise ValueError('Image and CameraInfo frame mismatch')
        tf = self.tf_buffer.lookup_transform(cc.frame, dc.frame, Time.from_msg(depth_msg.header.stamp))
        matrix = transform_matrix(tf)
        depth = depth_metres(depth_msg)
        opts = self.cfg['depth']
        result = associate_depth(depth, dc, cc, matrix, labels, confidence,
                                 opts['min_m'], opts['max_m'], self.cfg['confidence_min'], opts['occlusion_tolerance_m'])
        if not len(result['xyz']) or not np.isfinite(result['aligned_depth']).any():
            raise ValueError('No valid depth projects into the RGB image')
        candidate = result['candidate']
        # Candidate borders are conservative; missing depth is never filled in.
        road_interior = cv2.erode(((labels == 0) & (confidence >= self.cfg['confidence_min'])).astype(np.uint8), np.ones((3, 3), np.uint8))
        candidate[road_interior == 0] = 0
        self.images['aligned_depth'].publish(image_message(result['aligned_depth'], '32FC1', rgb.header))
        valid_depth = np.isfinite(result['aligned_depth'])
        scaled = np.uint8(np.clip((np.nan_to_num(result['aligned_depth']) - opts['min_m']) / (opts['max_m'] - opts['min_m']) * 255, 0, 255))
        preview = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        preview[~valid_depth] = 0
        self.images['depth_preview'].publish(image_message(preview, 'rgb8', rgb.header))
        self.images['paved_candidate'].publish(image_message(candidate, 'mono8', rgb.header))
        cloud_header = copy.deepcopy(depth_msg.header)
        self.cloud.publish(point_message(result['xyz'], result['labels'], result['confidence'], cloud_header,
                                         self.palette, opts['cloud_max_points']))
        mount = self.cfg['mount']
        self.obstacles.publish(point_message(np.empty((0, 3)), np.empty(0, np.uint8), np.empty(0, np.float32), cloud_header, self.palette))
        if mount['enabled']:
            try:
                base_tf = self.tf_buffer.lookup_transform(mount['target_frame'], dc.frame, Time.from_msg(depth_msg.header.stamp))
                base, obstacle, _ = height_obstacles(result['xyz'], transform_matrix(base_tf), mount['ground_z_m'],
                                                     mount['min_obstacle_height_m'], mount['max_obstacle_height_m'], mount['ground_tolerance_m'])
                cloud_header.frame_id = mount['target_frame']
                self.obstacles.publish(point_message(base[obstacle], result['labels'][obstacle], result['confidence'][obstacle], cloud_header,
                                                     self.palette, opts['cloud_max_points']))
                self.mount_ready = True
            except Exception as exc:
                self.statistics['mount_reason'] = str(exc)
        valid = np.isfinite(result['aligned_depth'])
        self.statistics.update({'rgb_depth_delta_ms': round(abs(stamp(rgb) - depth_stamp) * 1000, 2),
                                'aligned_valid_fraction': float(valid.mean()), 'depth_points': len(result['xyz']),
                                'candidate_pixels': int(np.count_nonzero(candidate)),
                                'associated_points': int(np.count_nonzero(result['labels'] != 255))})
        if not self.calibration_written:
            details = {'color_K': cc.k.tolist(), 'color_D': cc.d.tolist(), 'color_size': [cc.width, cc.height],
                       'depth_rectified_K': dc.k.tolist(), 'depth_size': [dc.width, dc.height],
                       'depth_frame': dc.frame, 'color_frame': cc.frame, 'depth_to_color_metres': matrix.tolist(),
                       'depth_encoding': depth_msg.encoding, 'note': 'Factory calibration, not a measured robot mount or field depth-accuracy certificate'}
            save_json(self.runtime_dir / 'camera-calibration.json', details)
            self.calibration_written = True

    def health(self):
        fresh_wall = time.monotonic() - self.last_result <= self.cfg['max_age_seconds']
        if not fresh_wall or not is_fresh(self.now(), self.last_source_stamp, self.cfg['max_age_seconds']):
            self.invalidate('Waiting for fresh RGB; old results invalidated')
        value = {'valid': self.rgb_valid, 'fusion_valid': self.fusion_valid, 'mount_ready': self.mount_ready,
                 'reason': self.reason, 'fusion_reason': self.fusion_reason, 'backend': self.predictor.backend,
                 'model_sha256': self.predictor.manifest['model_sha256'], 'processed_frames': self.processed,
                 'replaced_input_frames': self.replaced, 'errors': self.errors,
                 'source_stamp': self.last_source_stamp, 'status_stamp': self.now(),
                 'output_valid_until_stamp': self.last_source_stamp + self.cfg['max_age_seconds'],
                 'statistics': self.statistics, 'drive_output_enabled': False, 'field_validated': False,
                 'candidate_semantics': 'Confident paved appearance with measured depth, not a drivable-space decision',
                 'mount_reason': 'Not installed/configured' if not self.cfg['mount']['enabled'] else self.statistics.get('mount_reason', '')}
        message = String(data=json.dumps(value, allow_nan=False))
        self.status.publish(message)
        save_json(self.runtime_dir / 'status.json', value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/road-perception.yaml')
    parser.add_argument('--backend', choices=['auto', 'torch', 'tensorrt'])
    parser.add_argument('--rotation', type=int, choices=[0, 90, 180, 270])
    parser.add_argument('--camera-prefix')
    parser.add_argument('--output-prefix')
    parser.add_argument('--runtime-dir')
    args, ros_args = parser.parse_known_args()
    config = yaml.safe_load(args.config.read_text())
    for key in ['backend', 'camera_prefix', 'output_prefix', 'runtime_dir']:
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.rotation is not None:
        config['input_rotation_degrees'] = args.rotation
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    node = None
    try:
        node = RoadNode(config)
        while rclpy.ok() and not stopping:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.invalidate('Road perception stopped')
                node.health()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
