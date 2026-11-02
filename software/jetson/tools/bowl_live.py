#!/usr/bin/env python3
"""Subscribe to the single OpenMV capture process and publish segmentation."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from vision.bowl_detector import BowlDetector


class BowlNode(Node):
    def __init__(self, args):
        super().__init__('target_bowl_segmentation')
        self.args = args
        self.detector = BowlDetector(args.weights, args.imgsz, args.conf, args.device)
        self.weights_sha256 = hashlib.sha256(args.weights.read_bytes()).hexdigest()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.input = self.create_subscription(Image, args.topic, self.on_frame, qos)
        self.overlay = self.create_publisher(Image, '/openmv/bowl/overlay', qos)
        self.results = self.create_publisher(String, '/openmv/bowl/detections', 1)
        self.timer = self.create_timer(0.5, self.watchdog)
        self.last_frame = None
        self.frames = 0
        self.times = deque(maxlen=100)
        self.last_save = 0
        self.previous_present = None
        self.last_event_save = 0
        self.event_count = len(list((self.args.output / 'events').glob('*.json')))
        self.last_record = {'state': 'waiting_for_camera', 'detections': [], 'valid': False}
        self.args.output.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f'Model ready on {args.device}; input={args.topic}; conf={args.conf}; imgsz={args.imgsz}')

    def emit(self, record):
        value = String()
        value.data = json.dumps(record, ensure_ascii=False)
        self.results.publish(value)
        report = self.args.output / 'status.json'
        tmp = report.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(report)

    def on_frame(self, msg):
        age = (self.get_clock().now().nanoseconds - (msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)) / 1e9
        if not 0 <= age <= self.args.max_age:
            return
        if msg.encoding not in ('rgb8', 'bgr8') or msg.step < msg.width * 3:
            self.get_logger().error(f'Unsupported image: {msg.encoding}, step={msg.step}')
            return
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        rgb = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding == 'bgr8':
            rgb = rgb[:, :, ::-1]
        rgb = np.ascontiguousarray(rgb)
        overlay, detections, elapsed = self.detector.infer(rgb)
        now = time.monotonic()
        source_age = age + elapsed / 1000
        if source_age > self.args.max_age:
            return
        self.last_frame = now
        self.frames += 1
        self.times.append(now)
        fps = (len(self.times)-1) / max(self.times[-1]-self.times[0], 0.001)
        record = {'state': 'live', 'valid': True, 'frames': self.frames,
                  'stamp': {'sec': msg.header.stamp.sec, 'nanosec': msg.header.stamp.nanosec},
                  'updated_unix': time.time(), 'source_topic': self.args.topic,
                  'frame_id': msg.header.frame_id, 'width': msg.width, 'height': msg.height,
                  'inference_ms': elapsed, 'fps': fps, 'source_age_ms': source_age * 1000,
                  'confidence_threshold': self.args.conf, 'imgsz': self.args.imgsz,
                  'input_layout': 'square_letterbox', 'rect': False,
                  'weights_sha256': self.weights_sha256,
                  'coordinate_system': 'image_pixels_only', 'detections': detections}
        record.update(self.detector.runtime_info)
        output = Image()
        output.header = msg.header
        output.height, output.width = msg.height, msg.width
        output.encoding = 'rgb8'
        output.step = msg.width * 3
        output.data = overlay.tobytes()
        self.overlay.publish(output)
        self.last_record = record
        self.emit(record)
        present = bool(detections)
        if (self.event_count < self.args.max_events and present != self.previous_present
                and (now - self.last_event_save > 1 or self.previous_present is None)):
            event_dir = self.args.output / 'events'
            event_dir.mkdir(exist_ok=True)
            prefix = event_dir / f"{msg.header.stamp.sec}-{msg.header.stamp.nanosec:09d}-{'target' if present else 'background'}"
            PILImage.fromarray(rgb).save(prefix.with_suffix('.raw.png'))
            PILImage.fromarray(overlay).save(prefix.with_suffix('.overlay.png'))
            prefix.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
            self.previous_present = present
            self.last_event_save = now
            self.event_count += 1
        if now - self.last_save > 5:
            PILImage.fromarray(rgb).save(self.args.output / 'latest-raw.png')
            PILImage.fromarray(overlay).save(self.args.output / 'latest-overlay.png')
            (self.args.output / 'latest-sample.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
            self.last_save = now
            self.get_logger().info(f'{self.frames} frames; {fps:.1f} FPS; {elapsed:.1f} ms; targets={len(detections)}')

    def watchdog(self):
        if self.last_frame is None or time.monotonic() - self.last_frame > self.args.max_age:
            record = dict(self.last_record, state='waiting_for_camera' if self.last_frame is None else 'stale',
                          valid=False, detections=[], fps=0, updated_unix=time.time())
            self.emit(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'work/models/target-bowl-20261021/best.pt')
    parser.add_argument('--topic', default='/openmv/image_raw')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-age', type=float, default=1.0)
    parser.add_argument('--max-events', type=int, default=200, help='Bound saved scene-transition examples')
    parser.add_argument('--output', type=Path, default=ROOT / 'work/bowl-live')
    args = parser.parse_args()
    if not args.weights.is_file() or not 0 < args.conf < 1 or args.max_age <= 0:
        parser.error('Check weights, confidence threshold and max-age')
    rclpy.init()
    node = None
    try:
        node = BowlNode(args)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.emit({'state': 'stopped', 'valid': False, 'detections': [], 'updated_unix': time.time()})
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
