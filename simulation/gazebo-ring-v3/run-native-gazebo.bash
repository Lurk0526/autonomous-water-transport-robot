#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")/gazebo-ring-v1"

sim_root=$(cat mac-gazebo-env.txt)
installer_pid=${1:-}
run_stamp=$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)
status_file="$PWD/mac-native-status.txt"

status() {
  printf '%s %s\n' "$(TZ=Asia/Shanghai date '+%F %T %Z')" "$*" | tee "$status_file"
}

if [[ -n "$installer_pid" ]]; then
  status '等待 Gazebo 隔离环境安装'
  while ps -p "$installer_pid" -o command= | grep -F "$sim_root/bin/micromamba" | grep -q ' create '; do
    sleep 10
  done
fi

gazebo_env() {
  "$sim_root/bin/micromamba" --root-prefix "$sim_root/store" run -p "$sim_root/env" \
    env CMAKE_PREFIX_PATH="$sim_root/env" XDG_CACHE_HOME="$sim_root/cache" \
    GZ_IP=127.0.0.1 GZ_RENDERING_RESOURCE_PATH="$sim_root/env/share/gz/gz-rendering8" "$@"
}

if [[ ! -x "$sim_root/env/bin/gz" || ! -x "$sim_root/env/bin/cmake" ]] ||
   ! gazebo_env gz sim --version | tee mac-native-version.txt ||
   ! grep -q 'version 8\.' mac-native-version.txt; then
  status '安装或 Gazebo 8.x 版本检查未通过；未启动烟测'
  exit 1
fi

selected_engine=
for engine in ogre2 ogre; do
  smoke_output="outputs/smoke-mac-$run_stamp-$engine"
  status "运行 $engine 动力学与 RGB-D 烟测：$smoke_output"
  if gazebo_env .venv/bin/python -u -m ring_training.gazebo_run smoke --seed 0 \
      --render-engine "$engine" --output "$smoke_output"; then
    selected_engine=$engine
    break
  fi
done

if [[ -z "$selected_engine" ]]; then
  status '两种渲染引擎的烟测均未通过；未启动采集'
  exit 2
fi

run_output="outputs/mac-native-$run_stamp"
status "烟测通过（${selected_engine}）；启动 16 圈采集、训练和 4 圈闭环：$run_output"
if gazebo_env bash Mac训练.command --collect --episodes 16 --start-seed 100 \
    --eval-episodes 4 --render-engine "$selected_engine" --output "$run_output"; then
  status "完整流程成功：$run_output"
else
  exit_code=$?
  status "完整流程未通过（退出码 ${exit_code}）：$run_output"
  exit "$exit_code"
fi
