#!/usr/bin/env bash
set -e
task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
source "$task_project_root/config/ros-transport.sh"
export ROS_HOME="$task_project_root/work/ros"
export ROS_LOG_DIR="$ROS_HOME/log"
mkdir -p "$ROS_LOG_DIR"
exec ros2 launch "$task_project_root/launch/openmv_isaac.launch.py" "$@"
