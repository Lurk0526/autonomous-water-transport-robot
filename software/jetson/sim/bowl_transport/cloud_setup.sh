#!/usr/bin/env bash
# Run only on the rented Ubuntu RTX workstation. Does not modify this Jetson.
set -euo pipefail
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo 'This setup targets an Ubuntu x86_64 RTX rental workstation.' >&2
  exit 2
fi
if ! command -v uv >/dev/null; then
  echo 'Install uv using https://docs.astral.sh/uv/getting-started/installation/ then rerun.' >&2
  exit 2
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
task_lab_dir="${1:-$HOME/IsaacLab-bowl}"
task_lab_commit='961760bcb91e3af48e7d116292e57e8ecaefc31e'
if [[ ! -e "$task_lab_dir" ]]; then
  git clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git "$task_lab_dir"
fi
if [[ ! -d "$task_lab_dir/.git" ]]; then
  echo "Existing destination is not an IsaacLab checkout: $task_lab_dir" >&2
  exit 2
fi
if [[ -n "$(git -C "$task_lab_dir" status --porcelain)" ]]; then
  echo 'Existing checkout has changes; choose a fresh installation directory.' >&2
  exit 2
fi
git -C "$task_lab_dir" checkout --detach "$task_lab_commit"
cd "$task_lab_dir"
uv sync --extra isaacsim --extra rsl-rl
uv run --extra isaacsim --extra rsl-rl python -c 'from importlib.metadata import version; print("Isaac Sim", version("isaacsim")); print("RSL RL", version("rsl-rl-lib"))'
echo "Environment prepared at $task_lab_dir. Follow RUN.md for the fixture and official Cartpole smoke check."
