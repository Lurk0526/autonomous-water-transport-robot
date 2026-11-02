#!/usr/bin/env python3
"""Check real Isaac GPU RGB→BGR conversion on a dedicated ROS domain.

Run start_isaac_vision.sh in ROS_DOMAIN_ID=89 before this script.
Synthetic images here are deliberately isolated from the live-camera GUI.
"""
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    context = Context()
    rclpy.init(context=context, domain_id=89)
    node = rclpy.create_node('verify_isaac_openmv', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    received = []
    subscription = node.create_subscription(Image, '/isaac/openmv/image_bgr',
                                            received.append, qos_profile_sensor_data)
    publisher = node.create_publisher(Image, '/openmv/image_raw', 2)
    rows, cols = np.indices((240, 320))
    rgb = np.stack((cols % 256, rows % 256, (cols + rows) % 256), axis=-1).astype(np.uint8)
    expected = rgb[:, :, ::-1]
    checked = 0
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline and checked < 3:
            message = Image()
            message.header.stamp = node.get_clock().now().to_msg()
            message.header.frame_id = 'synthetic_test_optical_frame'
            message.height, message.width = 240, 320
            message.encoding = 'rgb8'
            message.step = 960
            message.data = rgb.tobytes()
            publisher.publish(message)
            executor.spin_once(timeout_sec=0.15)
            for result in received:
                assert result.header.frame_id == 'synthetic_test_optical_frame'
                assert result.encoding == 'bgr8', result.encoding
                assert (result.width, result.height) == (320, 240)
                actual_rows = np.frombuffer(result.data, dtype=np.uint8).reshape(result.height, result.step)
                actual = actual_rows[:, :result.width * 3].reshape(240, 320, 3)
                np.testing.assert_array_equal(actual, expected)
                checked += 1
            received.clear()
        assert checked >= 3, f'Only received {checked} converted frames'
        report = {'result': 'PASS', 'domain_id': 89, 'input': 'synthetic RGB8 320x240',
                  'output': 'BGR8 320x240', 'pixel_check': 'exact RGB channel reversal',
                  'frames_checked': checked, 'real_camera_verified': False}
        path = Path(__file__).resolve().parents[1] / 'work/isaac/gpu-conversion-check.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False))
    finally:
        executor.shutdown()
        node.destroy_node()
        context.shutdown()


if __name__ == '__main__':
    main()
