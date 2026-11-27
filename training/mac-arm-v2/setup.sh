#!/usr/bin/env bash
set -euo pipefail
arm_package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$arm_package_dir"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'This offline installer requires native Apple Silicon macOS (arm64). Do not run under Rosetta.' >&2
  exit 2
fi
if [[ ! -f SHA256SUMS.txt ]]; then
  echo 'Missing SHA256SUMS.txt; use the complete released offline archive.' >&2
  exit 2
fi
shasum -a 256 -c SHA256SUMS.txt > /dev/null
mkdir -p outputs runtime
if [[ ! -x runtime/python/bin/python3 ]]; then
  tar -xzf payload/python-macos-arm64.tar.gz -C runtime
fi
runtime/python/bin/python3 -c 'import platform,sys; assert platform.machine()=="arm64"; assert sys.version_info[:2]==(3,12); print(sys.version)'
runtime/python/bin/python3 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links payload/wheels --require-hashes -r requirements-mac.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m tensorboard.main --version
bash run.sh inspect
echo 'Installed offline. Next: bash run.sh gui --cad-only --experimental-model'
echo 'Read the model limitations and Mac调参与仿真指南.md before training.'
