#!/usr/bin/env bash
set -eo pipefail
ROAD_CONTROL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROAD_CONTROL_ROOT"
action="${1:-status}"
if [ "$#" -gt 0 ]; then shift; fi
case "$action" in
  start)
    if systemctl --user is-active --quiet water-cart-road-perception; then
      echo '道路感知已运行。'
    else
      systemd-run --user --unit=water-cart-road-perception --collect \
        --property=Restart=on-failure --property=RestartSec=3 \
        --property=WorkingDirectory="$ROAD_CONTROL_ROOT" \
        "$ROAD_CONTROL_ROOT/start_road_perception.sh" "$@"
    fi
    /usr/bin/python3 tools/wait_road_ready.py
    ;;
  restart)
    systemctl --user stop water-cart-road-perception || true
    exec "$ROAD_CONTROL_ROOT/road_perception.sh" start "$@"
    ;;
  stop) systemctl --user stop water-cart-road-perception ;;
  status)
    systemctl --user --no-pager status water-cart-road-perception || true
    /usr/bin/python3 tools/show_road_status.py
    ;;
  check)
    source ./activate_gemini.sh
    exec /usr/bin/python3 tools/check_road_perception.py "$@"
    ;;
  record)
    source ./activate_gemini.sh
    exec /usr/bin/python3 tools/record_road_session.py "$@"
    ;;
  replay)
    source ./activate_gemini.sh
    exec /usr/bin/python3 tools/replay_road_session.py "$@"
    ;;
  rviz)
    source ./activate_gemini.sh
    mkdir -p work/road_perception
    exec 9>work/road_perception/rviz.lock
    if ! flock -n 9; then echo '道路感知 RViz 已打开。'; exit 0; fi
    exec rviz2 -d "$ROAD_CONTROL_ROOT/config/road-perception.rviz"
    ;;
  *) echo '用法: ./road_perception.sh start|restart|stop|status|check|rviz|record|replay' >&2; exit 2 ;;
esac
