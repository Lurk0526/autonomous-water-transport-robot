#!/usr/bin/env bash
set -eo pipefail
GEMINI_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$GEMINI_PROJECT_DIR/activate_gemini.sh"
mkdir -p "$GEMINI_PROJECT_DIR/work/gemini"
exec 9>"$GEMINI_PROJECT_DIR/work/gemini/camera.lock"
if ! flock -n 9; then
    echo 'Gemini MAX 已经启动。查看状态：systemctl --user status gemini-max' >&2
    exit 1
fi
# The UVC node stays alive without publishing if started before udev creates
# this device. Wait for it before launching so login autostart can recover.
if [[ ! -c /dev/gemini-color || ! -r /dev/gemini-color || ! -w /dev/gemini-color ]]; then
    echo '等待 Gemini MAX 彩色接口 /dev/gemini-color 就绪……' >&2
    while [[ ! -c /dev/gemini-color || ! -r /dev/gemini-color || ! -w /dev/gemini-color ]]; do
        sleep 1
    done
fi
cd /home/nvidia/gemini_ws
exec /opt/ros/jazzy/bin/ros2 launch "$GEMINI_PROJECT_DIR/launch/gemini_max.launch.py" "$@"
