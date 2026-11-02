#!/usr/bin/env bash
set -e
task_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_project_root"
source ./activate_project.sh
exec python tools/openmv_gui.py "$@"
