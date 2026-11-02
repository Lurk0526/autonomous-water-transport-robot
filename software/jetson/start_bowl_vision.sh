#!/usr/bin/env bash
# Session services only; no boot-time autostart and no robot control.
set -euo pipefail
task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-start}" in
  stop)
    systemctl --user stop jibot-openmv-gui jibot-bowl-detector jibot-isaac-vision
    exit 0 ;;
  status)
    systemctl --user --no-pager status jibot-openmv-gui jibot-bowl-detector jibot-isaac-vision
    exit $? ;;
  start) ;;
  *) echo "Usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac
start_unit() {
    local unit="$1"
    shift
    if ! systemctl --user is-active --quiet "$unit"; then
        systemd-run --user --collect --unit="$unit" \
            --property="WorkingDirectory=$task_project_root" "$@"
    fi
}
start_unit jibot-bowl-detector "$task_project_root/start_bowl_detection.sh"
start_unit jibot-isaac-vision "$task_project_root/start_isaac_vision.sh"
start_unit jibot-openmv-gui /usr/bin/env "DISPLAY=${DISPLAY:-:1}" \
    "XAUTHORITY=${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}" "$task_project_root/start_vision_gui.sh"
