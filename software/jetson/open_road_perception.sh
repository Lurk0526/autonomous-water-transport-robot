#!/usr/bin/env bash
set -eo pipefail
ROAD_GUI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROAD_GUI_ROOT"
./road_perception.sh start
exec ./road_perception.sh rviz
