#!/usr/bin/env bash
set -e
task_annotation_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_image_folder="${1:-$task_annotation_root/datasets/target_bowl/source}"
if [ "$#" -gt 0 ]; then shift; fi
cd "$task_annotation_root"
mkdir -p "$task_image_folder"
exec "$task_annotation_root/.venv-labelme/bin/labelme" \
  "$task_image_folder" --config "$task_annotation_root/annotation/labelme-bowl.yaml" "$@"
