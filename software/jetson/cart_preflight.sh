#!/usr/bin/env bash
set -eo pipefail
cart_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$cart_project_dir"
source ./activate_gemini.sh
exec /usr/bin/python3 -m cart.preflight "$@"
