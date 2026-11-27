#!/usr/bin/env bash
set -euo pipefail
arm_package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$arm_package_dir"
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash setup.sh first.' >&2; exit 2; fi
export PYTHONPATH="$arm_package_dir${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$arm_package_dir/outputs/matplotlib"
export YOLO_CONFIG_DIR="$arm_package_dir/outputs/ultralytics"
export YOLO_AUTOINSTALL=false
export YOLO_OFFLINE=true
export YOLO_VERBOSE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
arm_python=.venv/bin/python
for arm_arg in "$@"; do
  if [[ "$arm_arg" == gui && "$(uname -s)" == Darwin ]]; then arm_python=.venv/bin/mjpython; break; fi
done
exec "$arm_python" -m arm_training.mobile_cli "$@"
