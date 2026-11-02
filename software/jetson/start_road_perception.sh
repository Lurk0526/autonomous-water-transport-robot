#!/usr/bin/env bash
set -eo pipefail
ROAD_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ROAD_PROJECT_DIR/activate_project.sh"
source "$ROAD_PROJECT_DIR/activate_gemini.sh"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
cd "$ROAD_PROJECT_DIR"
mkdir -p work/road_perception
# A separate lock can be used for isolated bag replay.
exec 9>"${ROAD_PERCEPTION_LOCK:-$ROAD_PROJECT_DIR/work/road_perception/live.lock}"
if ! flock -n 9; then
    echo '道路识别已运行；请先检查现有实例。' >&2
    exit 1
fi
exec python tools/road_perception.py "$@"
