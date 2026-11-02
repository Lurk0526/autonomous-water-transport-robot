"""Gemini MAX RGB-D and camera-relative obstacle scan; no motor commands."""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    config = str(root / 'config' / 'gemini-max.yaml')
    args = [
        DeclareLaunchArgument('publish_mount_tf', default_value='false',
                              description='Enable only after measuring camera pose on robot'),
        DeclareLaunchArgument('rviz', default_value='false'),
    ]
    for name in ('mount_x', 'mount_y', 'mount_z', 'mount_roll', 'mount_pitch', 'mount_yaw'):
        args.append(DeclareLaunchArgument(name, default_value='0.0'))
    return LaunchDescription(args + [
        Node(package='gemini_camera', executable='gemini_camera_node',
             namespace='camera', name='camera', parameters=[config], output='screen',
             respawn=True, respawn_delay=3.0),
        Node(package='v4l2_camera', executable='v4l2_camera_node',
             namespace='camera/color', name='uvc_color',
             parameters=[config, {'camera_info_url':
                 'file://' + str(root / 'config' / 'gemini-max-color-calibration.yaml')}],
             output='screen', respawn=True, respawn_delay=3.0),
        Node(package='image_proc', executable='rectify_node',
             namespace='camera/depth', name='rectify_depth',
             parameters=[{'interpolation': 0}],
             remappings=[('image', 'image_raw'), ('image_rect', 'image_rect_raw')]),
        Node(package='depthimage_to_laserscan', executable='depthimage_to_laserscan_node',
             namespace='camera', name='depth_to_scan', parameters=[config],
             remappings=[('depth', '/camera/depth/image_rect_raw'),
                         ('depth_camera_info', '/camera/depth/camera_info'),
                         ('scan', '/camera/scan')], output='screen'),
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='base_to_main_camera', condition=IfCondition(LaunchConfiguration('publish_mount_tf')),
             arguments=[
                 '--x', LaunchConfiguration('mount_x'), '--y', LaunchConfiguration('mount_y'),
                 '--z', LaunchConfiguration('mount_z'), '--roll', LaunchConfiguration('mount_roll'),
                 '--pitch', LaunchConfiguration('mount_pitch'), '--yaw', LaunchConfiguration('mount_yaw'),
                 '--frame-id', 'base_link', '--child-frame-id', 'camera_link']),
        Node(package='rviz2', executable='rviz2',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', str(root / 'config' / 'gemini-max.rviz')]),
    ])
