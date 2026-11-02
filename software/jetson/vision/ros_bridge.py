"""ROS 2 sensor_msgs bridge, with no joint command interfaces."""
import json
import time

import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String


class ImageBridge:
    def __init__(self):
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('jibot_openmv', context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        image_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.images = self.node.create_publisher(Image, '/openmv/image_raw', image_qos)
        self.infos = self.node.create_publisher(CameraInfo, '/openmv/camera_info', image_qos)
        self.status = self.node.create_publisher(String, '/openmv/status', 10)
        result_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.latest_overlay = None
        self.detection_status = None
        self.overlay_sub = self.node.create_subscription(
            Image, '/openmv/bowl/overlay', self.on_overlay, result_qos)
        self.detections_sub = self.node.create_subscription(
            String, '/openmv/bowl/detections', self.on_detections, 1)
        self.isaac_frames = 0
        self.isaac_first = None
        self.isaac_last = None
        self.isaac_sub = self.node.create_subscription(
            Image, '/isaac/openmv/image_bgr', self.on_isaac, result_qos)
        self.follow_status = None
        self.follow_sub = self.node.create_subscription(
            String, '/openmv/bowl/follow_status', self.on_follow_status, 1)
        self.follow_stop = self.node.create_publisher(String, '/openmv/bowl/stop_request', 1)
        self.count = 0
        self.last_status = 0

    def on_overlay(self, message):
        if message.encoding != 'rgb8' or message.step < message.width * 3:
            return
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        rgb = rows[:, :message.width * 3].reshape(message.height, message.width, 3).copy()
        self.latest_overlay = (rgb, stamp)

    def on_detections(self, message):
        self.detection_status = (json.loads(message.data), time.monotonic())

    def on_isaac(self, message):
        if message.encoding == 'bgr8':
            self.isaac_last = time.monotonic()
            if self.isaac_first is None:
                self.isaac_first = self.isaac_last
            self.isaac_frames += 1

    def on_follow_status(self, message):
        self.follow_status = (json.loads(message.data), time.monotonic())

    def request_follow_stop(self):
        message = String()
        message.data = 'operator_stop'
        self.follow_stop.publish(message)

    def fresh_overlay(self):
        if self.latest_overlay is None:
            return None
        rgb, stamp = self.latest_overlay
        age = self.node.get_clock().now().nanoseconds / 1e9 - stamp
        return rgb if 0 <= age <= 1 else None

    def publish(self, rgb):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        height, width = rgb.shape[:2]
        message = Image()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = 'openmv_optical_frame'
        message.height, message.width = height, width
        message.encoding = 'rgb8'
        message.step = width * 3
        message.data = rgb.tobytes()
        self.images.publish(message)
        # CameraInfo's all-zero K explicitly denotes an uncalibrated camera.
        info = CameraInfo()
        info.header = message.header
        info.height, info.width = height, width
        self.infos.publish(info)
        self.count += 1

    def tick(self, camera_state):
        # Drain several ready subscriptions without blocking the Qt event loop.
        for _ in range(4):
            self.executor.spin_once(timeout_sec=0)
        if time.monotonic() - self.last_status > 1:
            value = String()
            value.data = json.dumps({'camera': camera_state, 'frames_published': self.count,
                                     'calibrated': False, 'depth_available': False,
                                     'stamp_source': 'host_receive_time'}, ensure_ascii=False)
            self.status.publish(value)
            self.last_status = time.monotonic()

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.shutdown()
