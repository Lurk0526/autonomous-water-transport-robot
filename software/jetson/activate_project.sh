# Source this file: source ./activate_project.sh
# ROS must be loaded before the project Python environment.
_jetson_activation_nounset=false
case "$-" in
    *u*) _jetson_activation_nounset=true; set +u ;;
esac
_jetson_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if declare -F deactivate >/dev/null; then
    deactivate
fi
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi
source "${_jetson_project_root}/.venv/bin/activate"
source "${_jetson_project_root}/config/ros-transport.sh"
export YOLO_CONFIG_DIR="${_jetson_project_root}/work/ultralytics"
export YOLO_AUTOINSTALL=false
export MPLCONFIGDIR="${_jetson_project_root}/work/matplotlib"
export ROS_HOME="${_jetson_project_root}/work/ros"
export ROS_LOG_DIR="${ROS_HOME}/log"
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$ROS_HOME" "$ROS_LOG_DIR"
unset _jetson_project_root
if "$_jetson_activation_nounset"; then
    set -u
fi
unset _jetson_activation_nounset
