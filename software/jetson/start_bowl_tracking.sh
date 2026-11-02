#!/usr/bin/env bash
# Start the calibrated image tracker; a separate flag requests physical calibration.
set -euo pipefail
task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_project_root"
case "${1:-start}" in
  --stop|stop)
    systemctl --user stop jibot-bowl-follow
    exit 0 ;;
  --calibrate) task_calibrate=true ;;
  --restart) task_calibrate=false ;;
  start) task_calibrate=false ;;
  *) echo "Usage: $0 [start|--restart|--calibrate|--stop]" >&2; exit 2 ;;
esac
if systemctl --user is-active --quiet jibot-bowl-follow; then
    if ! "$task_calibrate" && [ "${1:-start}" != '--restart' ]; then
        if "$task_project_root/.venv/bin/python" -c 'import json,time,sys; r=json.load(open("work/bowl-follow/status.json")); sys.exit(not (r.get("motion_enabled") is True and 0<=time.time()-r.get("updated_unix",0)<=1.5))'; then
            echo '机械臂跟随正在运行；可用界面停止按钮或 --stop 停止。'
            exit 0
        fi
    fi
    systemctl --user stop jibot-bowl-follow
fi
./start_bowl_vision.sh start
source ./activate_project.sh
if "$task_calibrate"; then
    python tools/prepare_follow_trial.py --adapt-current-pose
else
    python tools/prepare_follow_trial.py --reuse-profile
fi
systemd-run --user --collect --unit=jibot-bowl-follow \
    --property="WorkingDirectory=$task_project_root" \
    "$task_project_root/start_bowl_follow.sh" --enable-motion \
    --config "$task_project_root/work/bowl-follow/current-trial.json"
