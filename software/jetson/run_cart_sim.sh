#!/usr/bin/env bash
set -euo pipefail
cart_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$cart_project_dir"
exec python3 -m sim.line_free_cart.run "$@"
