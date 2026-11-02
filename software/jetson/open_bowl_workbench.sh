#!/usr/bin/env bash
# Desktop launcher opens/focuses vision only; motor start is an explicit GUI action.
set -e
task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_project_root"
mkdir -p work/openmv
exec >> work/openmv/desktop-launch.log 2>&1
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
./start_bowl_vision.sh start
exec ./start_vision_gui.sh --activate-only
