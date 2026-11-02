#!/usr/bin/env python3
"""Portable single-GPU YOLO11 instance-segmentation training and inference."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare_data(data_file, runtime_dir=None):
    """Rebase a transferred YOLO dataset without modifying its original YAML."""
    import yaml
    data_file = Path(data_file).resolve()
    if not data_file.is_file():
        raise ValueError(f'Dataset YAML missing: {data_file}. Export annotations first.')
    data = yaml.safe_load(data_file.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('Dataset YAML must be a mapping')
    names = data.get('names')
    if names not in ({0: 'target_bowl'}, {'0': 'target_bowl'}, ['target_bowl']):
        raise ValueError('Expected the single target_bowl class')
    dataset_root = data_file.parent
    for split in ('train', 'val', 'test'):
        expected = f'images/{split}'
        if data.get(split) != expected or not (dataset_root / expected).is_dir():
            raise ValueError(f'Expected {expected} next to data.yaml; use the bundled exporter')
        if not any(p.is_file() for p in (dataset_root / expected).iterdir()):
            raise ValueError(f'Empty image split: {split}')
    data['path'] = dataset_root.as_posix()
    data.pop('download', None)
    destination = Path(runtime_dir or ROOT / 'work/data-configs')
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / 'target-bowl.yaml'
    target.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return target


def default_pretrained():
    for candidate in [ROOT / 'pretrained/yolo11n-seg.pt', ROOT / 'work/models/yolo11n-seg.pt']:
        if candidate.is_file():
            return candidate
    return ROOT / 'pretrained/yolo11n-seg.pt'


def latest_weights():
    matches = list((ROOT / 'runs/segment').glob('*/weights/best.pt'))
    if not matches:
        raise ValueError('No trained best.pt found. Pass --weights explicitly.')
    return max(matches, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'validate', 'predict', 'export'])
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets/target_bowl/yolo/data.yaml')
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', default='0', help='0 for NVIDIA GPU; cpu for CPU')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--name', default='target-bowl-v1')
    parser.add_argument('--project', type=Path, default=ROOT / 'runs/segment')
    parser.add_argument('--source', help='Image, image folder or video for predict')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'work/ultralytics'))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    from ultralytics import YOLO
    import torch

    if args.device != 'cpu' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. Run check_training_env.py or pass --device cpu explicitly.')
    try:
        weights = args.weights or (default_pretrained() if args.action == 'train' else latest_weights())
        if not weights.is_file():
            raise ValueError(f'Weights not found: {weights}')
        model = YOLO(str(weights))
        if model.task != 'segment':
            raise ValueError('Expected a YOLO instance segmentation model')
        if args.action == 'train':
            data = prepare_data(args.data)
            if args.epochs < 1 or args.batch < 1:
                raise ValueError('epochs and batch must be positive')
            # Disable auto AMP test/model downloads; native FP32 is the first baseline.
            model.train(data=str(data), epochs=args.epochs, batch=args.batch,
                        imgsz=args.imgsz, workers=args.workers, device=args.device,
                        project=str(args.project.resolve()), name=args.name,
                        pretrained=True, amp=False, seed=42, plots=not args.no_plots,
                        patience=20, cache=False, exist_ok=False)
            directory = Path(model.trainer.save_dir)
            record = {'save_dir': str(directory), 'best_weights': str(directory / 'weights/best.pt'),
                      'data': str(args.data.resolve()), 'task': 'segment', 'class': 'target_bowl'}
            (ROOT / 'work').mkdir(parents=True, exist_ok=True)
            (ROOT / 'work/last-training.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(record, ensure_ascii=False, indent=2))
        elif args.action == 'validate':
            metrics = model.val(data=str(prepare_data(args.data)), split=args.split,
                                imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                                device=args.device, project=str(args.project.resolve()),
                                name='evaluation-' + args.split, plots=not args.no_plots)
            print(json.dumps(metrics.results_dict, indent=2))
        elif args.action == 'predict':
            if not args.source:
                raise ValueError('Pass --source IMAGE_OR_FOLDER_OR_VIDEO')
            model.predict(source=args.source, device=args.device, imgsz=args.imgsz,
                          conf=args.conf, save=True, project=str(args.project.resolve()), name='predict')
        else:
            # Transfer ONNX/weights to Jetson; TensorRT engines are built on target.
            print(model.export(format='onnx', imgsz=args.imgsz, device='cpu',
                               opset=17, dynamic=False, simplify=False))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
