#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ "$(uname -s)/$(uname -m)" != Darwin/arm64 ]]; then
  echo "此离线环境供 Apple Silicon Mac 原生终端使用。" >&2
  exit 2
fi
shasum -a 256 -c SHA256SUMS.txt > /dev/null
if [[ ! -x runtime/python/bin/python3 ]]; then
  mkdir -p runtime
  tar -xzf payload/python-macos-arm64.tar.gz -C runtime
fi
runtime/python/bin/python3 -c 'import platform,sys; assert platform.machine()=="arm64" and sys.version_info[:2]==(3,12)'
runtime/python/bin/python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links payload/wheels --require-hashes -r requirements-mac.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import cv2,numpy,onnxruntime,PIL; print("Mac 环境就绪", numpy.__version__, onnxruntime.__version__, onnxruntime.get_available_providers())'
