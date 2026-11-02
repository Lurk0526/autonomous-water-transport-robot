#!/usr/bin/env python3
"""Exercise missing TF/depth, stale frames, recovery and dropout in an isolated ROS domain."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.road_geometry import image_array


def main():
    output = ROOT/'work/road_perception/fault-tests'
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = '84'
    env['ROAD_PERCEPTION_LOCK'] = str(output/'test.lock')
    os.environ['ROS_DOMAIN_ID'] = '84'
    rclpy.init()
    node = rclpy.create_node('road_fault_test_source')
    pubs = {name: node.create_publisher(kind, '/camera/'+name, qos_profile_sensor_data) for name, kind in
            [('color/image_raw', Image), ('depth/image_rect_raw', Image), ('color/camera_info', CameraInfo), ('depth/camera_info', CameraInfo)]}
    latest = {}
    def receive(name, message):
        latest[name] = json.loads(message.data) if name == 'status' else message
    subs = [node.create_subscription(kind, '/road_perception/'+name, lambda m, k=name: receive(k, m),
                                     1 if name == 'status' else qos_profile_sensor_data) for name, kind in
            [('status', String), ('mask', Image), ('confidence', Image), ('aligned_depth', Image), ('paved_candidate', Image), ('labeled_points', PointCloud2)]]
    broadcaster = StaticTransformBroadcaster(node)
    log = (output/'perception.log').open('w')
    process = subprocess.Popen([str(ROOT/'start_road_perception.sh'), '--rotation', '0', '--runtime-dir', str(output/'runtime')],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    stages = []
    def send(depth_mm=1000, stale=False):
        current = node.get_clock().now().to_msg()
        if stale:
            current.sec -= 5
        for color in [True, False]:
            frame = 'test_color_optical' if color else 'test_depth_optical'
            w, h = 640, 480 if color else 400
            info = CameraInfo()
            info.header.stamp, info.header.frame_id = current, frame
            info.width, info.height = w, h
            info.distortion_model, info.d = 'plumb_bob', [0.0]*5
            info.k = [500., 0., 320., 0., 500., h/2, 0., 0., 1.]
            info.r = np.eye(3).ravel().tolist()
            info.p = [500.,0.,320.,0.,0.,500.,h/2,0.,0.,0.,1.,0.]
            image = Image()
            image.header = info.header
            image.width, image.height = w, h
            image.encoding = 'rgb8' if color else '16UC1'
            image.is_bigendian = False
            image.step = w*(3 if color else 2)
            image.data = (np.full((h,w,3), 128, np.uint8) if color else np.full((h,w), depth_mm, np.uint16)).tobytes()
            key = 'color' if color else 'depth'
            pubs[key+'/camera_info'].publish(info)
            pubs[key+('/image_raw' if color else '/image_rect_raw')].publish(image)
    def stage(name, predicate, depth_mm=1000, stale=False, inputs=True):
        start = time.monotonic()
        while time.monotonic()-start < 2.5:
            if process.poll() is not None:
                raise RuntimeError('Perception node exited during '+name)
            if inputs:
                send(depth_mm, stale)
            rclpy.spin_once(node, timeout_sec=0.02)
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0)
            time.sleep(0.03)
        status = latest.get('status', {})
        passed = bool(predicate(status))
        stages.append({'name': name, 'passed': passed, 'status': status})
        print(name, 'PASS' if passed else 'FAIL', flush=True)
    try:
        deadline = time.monotonic()+40
        ready = output/'runtime/runtime.json'
        launch_time = time.time()
        while time.monotonic()<deadline and not (ready.exists() and ready.stat().st_mtime >= launch_time-0.2):
            if process.poll() is not None:
                raise RuntimeError('Fault test node failed to start')
            rclpy.spin_once(node, timeout_sec=0.1)
        stage('missing_tf_keeps_rgb_but_blocks_fusion', lambda s: s.get('valid') and not s.get('fusion_valid'))
        tf = TransformStamped()
        tf.header.stamp = node.get_clock().now().to_msg()
        tf.header.frame_id, tf.child_frame_id = 'test_color_optical', 'test_depth_optical'
        tf.transform.rotation.w = 1.
        broadcaster.sendTransform(tf)
        stage('all_zero_depth_is_unknown', lambda s: s.get('valid') and not s.get('fusion_valid'), depth_mm=0)
        stage('fresh_data_recovers', lambda s: s.get('valid') and s.get('fusion_valid'))
        stage('stale_rgb_is_rejected', lambda s: not s.get('valid') and not s.get('fusion_valid'), stale=True)
        stage('recovery_after_stale', lambda s: s.get('valid') and s.get('fusion_valid'))
        stage('dropout_clears_outputs', lambda s: not s.get('valid') and not s.get('fusion_valid'), inputs=False)
        checks = {'mask_unknown': 'mask' in latest and bool(np.all(image_array(latest['mask']) == 255)),
                  'confidence_zero': 'confidence' in latest and not image_array(latest['confidence']).any(),
                  'candidate_empty': 'paved_candidate' in latest and not image_array(latest['paved_candidate']).any(),
                  'cloud_empty': 'labeled_points' in latest and latest['labeled_points'].width == 0}
        report = {'passed': all(s['passed'] for s in stages) and all(checks.values()),
                  'stages': stages, 'dropout_outputs': checks, 'domain_id': 84,
                  'note': 'Synthetic images/TF only; no live camera or controller topics in this isolated domain'}
        (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'passed': report['passed'], 'dropout_outputs': checks}), flush=True)
        return not report['passed']
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=15)
        log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
