#!/usr/bin/env python3
"""Install a verified training checkpoint without modifying the training bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/nvidia/Desktop/water-cart-road-4060-20261029'))
    parser.add_argument('--name', default='road-4060-v1')
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError('Model name must be a directory name')
    run = args.source / 'work/road_training/runs' / args.name
    labels = args.source / 'config/road-labels.json'
    result = json.loads((run / 'result.json').read_text())
    config = json.loads((run / 'best/model/config.json').read_text())
    names = json.loads(labels.read_text())['names']
    if result['status'] != 'completed' or result['smoke_only']:
        raise ValueError('A completed formal run is required')
    if sha(labels) != result['label_sha256'] or sha(labels) != sha(ROOT / 'config/road-labels.json'):
        raise ValueError('Training/deployment label mappings differ')
    if list(config['id2label'].values()) != names or config['num_channels'] != 3:
        raise ValueError('Unexpected checkpoint classes or input channels')
    for key in ['train_manifest', 'val_manifest']:
        if sha(args.source / result['config'][key]) != result[key + '_sha256']:
            raise ValueError(f'{key} changed since training')
    parent = ROOT / 'work/models/road'
    target = parent / args.name
    model_sha = sha(run / 'best/model/model.safetensors')
    if target.exists():
        manifest = json.loads((target / 'manifest.json').read_text())
        if manifest['model_sha256'] != model_sha or sha(target / 'model/model.safetensors') != model_sha:
            raise FileExistsError('Existing deployment differs; use another name')
        print(target)
        return
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.install-road-', dir=parent))
    try:
        shutil.copytree(run / 'best/model', staging / 'model')
        shutil.copy2(labels, staging / 'labels.json')
        evidence = staging / 'training-evidence'
        evidence.mkdir()
        for name in ['run.json', 'result.json', 'history.jsonl']:
            shutil.copy2(run / name, evidence / name)
        review = ROOT / 'work/road-4060-review-20261030'
        for name in ['test-evaluation.json', 'training-audit.json', 'focused-summary.json']:
            if (review / name).exists():
                shutil.copy2(review / name, evidence / name)
        manifest = {
            'schema_version': 1, 'name': args.name, 'checkpoint': 'best',
            'model_sha256': model_sha, 'config_sha256': sha(staging / 'model/config.json'),
            'labels_sha256': sha(staging / 'labels.json'), 'classes': names,
            'input_size': result['config']['size'], 'source': str(run / 'best/model'),
            'validation_selection_score': result['best_validation_selection_score'],
            'preprocess': 'RGB, ImageNet normalization, aspect-preserving letterbox',
            'postprocess': 'bilinear logits, remove letterbox padding, restore native pixel coordinates',
            'training_completed': True, 'field_validated': False,
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps({'installed': str(target), 'model_sha256': model_sha}, indent=2))


if __name__ == '__main__':
    main()
