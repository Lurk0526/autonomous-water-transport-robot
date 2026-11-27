#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c 'import cv2,numpy,onnxruntime,PIL' >/dev/null 2>&1; then
  bash setup_mac.sh
fi
exec .venv/bin/python tools/mac_training.py "$@"
