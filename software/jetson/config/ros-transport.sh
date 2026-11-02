# Shared by project Python and the system-Python Isaac launcher.
# Camera, detector and tracker communicate locally even when Wi-Fi changes.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
if [ "$RMW_IMPLEMENTATION" = rmw_cyclonedds_cpp ] && [ -z "${CYCLONEDDS_URI:-}" ]; then
    _jibot_ros_config_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    export CYCLONEDDS_URI="file://${_jibot_ros_config_dir}/cyclonedds-local.xml"
    unset _jibot_ros_config_dir
fi
