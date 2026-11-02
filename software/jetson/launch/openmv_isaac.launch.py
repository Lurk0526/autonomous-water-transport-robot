"""GPU color conversion for OpenMV images using Isaac ROS 4.6."""
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='openmv_isaac_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                ComposableNode(
                    name='openmv_color_convert',
                    package='isaac_ros_image_proc',
                    plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
                    parameters=[{'encoding_desired': 'bgr8',
                                 'image_width': 320, 'image_height': 240}],
                    remappings=[('image_raw', '/openmv/image_raw'),
                                ('image', '/isaac/openmv/image_bgr')],
                ),
            ],
            output='screen',
        ),
    ])
