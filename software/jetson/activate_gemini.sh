# Source this file to use the Gemini MAX camera and ROS 2 tools.
_gemini_nounset=false
case "$-" in *u*) _gemini_nounset=true; set +u ;; esac
source /opt/ros/jazzy/setup.bash
source /home/nvidia/gemini_ws/install/local_setup.bash
_gemini_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$_gemini_root/config/cyclonedds-gemini.xml"
unset _gemini_root
if "$_gemini_nounset"; then set -u; fi
unset _gemini_nounset
