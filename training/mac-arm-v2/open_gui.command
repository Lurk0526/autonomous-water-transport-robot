#!/usr/bin/env bash
set -euo pipefail
arm_package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$arm_package_dir/run.sh" gui --cad-only --experimental-model
