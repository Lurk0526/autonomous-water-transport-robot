#!/usr/bin/env bash
set -euo pipefail
arm_package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$arm_package_dir"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then echo 'Run on the receiving native arm64 Mac.' >&2; exit 2; fi
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash setup.sh first.' >&2; exit 2; fi
export PYTHONPATH="$arm_package_dir${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$arm_package_dir/outputs/matplotlib"
export YOLO_CONFIG_DIR="$arm_package_dir/outputs/ultralytics"
export YOLO_AUTOINSTALL=false YOLO_OFFLINE=true YOLO_VERBOSE=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
exec .venv/bin/python -m arm_training.mac_workflow "$@"
