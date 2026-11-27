#!/usr/bin/env bash
set -euo pipefail
arm_package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$arm_package_dir"
if [[ -x .venv/bin/python ]]; then arm_python=.venv/bin/python
elif [[ -x runtime/python/bin/python3 ]]; then arm_python=runtime/python/bin/python3
else echo 'No project Python; preserve installation logs and package them manually.' >&2; exit 2; fi
exec "$arm_python" tools/collect_return.py "$arm_package_dir" "$@"
