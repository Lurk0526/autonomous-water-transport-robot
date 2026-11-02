#!/usr/bin/env bash
set -eo pipefail
ROAD_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROAD_PROJECT"
source ./activate_project.sh
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec python tools/train_road_model.py "$@"
