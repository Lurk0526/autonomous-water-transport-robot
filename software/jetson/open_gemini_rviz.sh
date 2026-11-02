#!/usr/bin/env bash
set -eo pipefail
GEMINI_GUI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$GEMINI_GUI_ROOT/activate_gemini.sh"
# VNC has its own session bus; systemd's user manager uses /run/user/<uid>.
# Override only systemctl's environment so RViz keeps the current desktop session.
GEMINI_USER_RUNTIME="/run/user/$(id -u)"
if [[ -S "$GEMINI_USER_RUNTIME/bus" ]]; then
    env XDG_RUNTIME_DIR="$GEMINI_USER_RUNTIME" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=$GEMINI_USER_RUNTIME/bus" \
        systemctl --user start gemini-max
else
    systemctl --user start gemini-max
fi
mkdir -p "$GEMINI_GUI_ROOT/work/gemini"
exec 8>"$GEMINI_GUI_ROOT/work/gemini/rviz.lock"
if ! flock -n 8; then
    if command -v notify-send >/dev/null; then notify-send 'Gemini MAX' 'RViz 已打开，请切换到 RViz 窗口。'; fi
    exit 0
fi
export QT_QPA_PLATFORM=xcb
# OpenCV Python wheels may set Qt plugin paths unsuitable for the system RViz.
unset QT_QPA_PLATFORM_PLUGIN_PATH QT_PLUGIN_PATH
exec /opt/ros/jazzy/bin/rviz2 -d "$GEMINI_GUI_ROOT/config/gemini-max.rviz" >> "$GEMINI_GUI_ROOT/work/gemini/rviz.log" 2>&1
