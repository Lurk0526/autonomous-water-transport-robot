#!/usr/bin/env python3
"""Validate ROS packages and local-only DDS using synthetic test messages."""
import importlib
import json
import os
from pathlib import Path
import time

# Applies only to this test process. Do not discover or send to other machines.
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ["ROS_LOG_DIR"] = str(Path(__file__).resolve().parents[1] / "work" / "ros-logs")

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from ament_index_python.packages import get_packages_with_prefixes, get_package_share_directory
from std_msgs.msg import String


def main():
    names = ["rclpy", "cv_bridge", "image_transport", "tf2_ros", "tf2_tools",
             "robot_state_publisher", "joint_state_publisher", "rosbag2_transport",
             "moveit_ros_move_group", "moveit_servo", "moveit_py", "moveit_configs_utils",
             "controller_manager", "joint_trajectory_controller", "joint_state_broadcaster",
             "gripper_controllers", "ackermann_steering_controller", "nav2_bringup",
             "nav2_mppi_controller", "robot_localization", "ackermann_msgs",
             "orbbec_camera", "orbbec_camera_msgs", "image_pipeline", "camera_calibration",
             "v4l2_camera", "usb_cam", "vision_msgs"]
    packages = get_packages_with_prefixes()
    missing = [name for name in names if name not in packages]
    assert not missing, "Missing ROS packages: " + str(missing)
    print(json.dumps(dict(package_index="PASS", checked_packages=names)), flush=True)
    for name in ["moveit.core.robot_model", "moveit.planning", "moveit_configs_utils",
                 "controller_manager", "cv_bridge", "tf2_ros"]:
        importlib.import_module(name)
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.action import MoveGroup
    from orbbec_camera_msgs.msg import Extrinsics
    for message in [FollowJointTrajectory.Goal, MoveGroup.Goal, Extrinsics]:
        message.__class__.__import_type_support__()
        assert message.__class__._TYPE_SUPPORT is not None
    print(json.dumps(dict(python_and_native_interfaces="PASS")), flush=True)
    camera_share = Path(get_package_share_directory("orbbec_camera"))
    launches = sorted(p.name for p in (camera_share / "launch").glob("*gemini*"))
    print(json.dumps(dict(orbbec_launch_files=launches, camera_opened=False)), flush=True)
    context = Context()
    executor = None
    tx = rx = None
    try:
        # A dedicated domain, unique topic, local discovery; no robot topics used.
        rclpy.init(context=context, domain_id=88)
        tx = rclpy.create_node("jetson_env_check_tx", context=context, enable_rosout=False)
        rx = rclpy.create_node("jetson_env_check_rx", context=context, enable_rosout=False)
        received = []
        topic = "/jetson_env_check_{}".format(os.getpid())
        subscription = rx.create_subscription(String, topic, lambda msg: received.append(msg.data), 10)
        publisher = tx.create_publisher(String, topic, 10)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(tx)
        executor.add_node(rx)
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline and not received:
            publisher.publish(String(data="environment-check"))
            executor.spin_once(timeout_sec=0.1)
        assert received and received[0] == "environment-check", "Local DDS round trip timed out"
        print(json.dumps(dict(local_dds_pubsub="PASS", domain=88,
                              discovery="LOCALHOST", hardware_accessed=False)), flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
        for node in [tx, rx]:
            if node is not None:
                node.destroy_node()
        if context.ok():
            context.shutdown()


if __name__ == "__main__":
    main()
