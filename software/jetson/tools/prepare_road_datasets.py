#!/usr/bin/python3
"""Audit public images and labels, map labels, and build video-disjoint manifests."""
import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=['camvid', 'rugd'], default=['camvid', 'rugd'])
    parser.add_argument('--annotations-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config/road-labels.json').read_text(encoding='utf-8'))
    out = ROOT / 'datasets/public_road/prepared'
    out.mkdir(parents=True, exist_ok=True)
    jobs, rgb_lut = [], None
    if 'camvid' in args.datasets:
        base = ROOT / 'datasets/public_road/camvid/SegNet-Tutorial-master/CamVid'
        for original_split in ('train', 'val', 'test'):
            for image in sorted((base / original_split).glob('*.png')):
                seq = image.stem.split('_')[0]
                split = 'test' if seq in cfg['camvid_test_sequences'] else 'val' if seq in cfg['camvid_val_sequences'] else 'train'
                jobs.append(('camvid', seq, split, original_split, image, base / (original_split + 'annot') / image.name))
    if 'rugd' in args.datasets:
        base = ROOT / 'datasets/public_road/rugd'
        annotations = base / 'RUGD_annotations'
        rgb_lut = np.full(1 << 24, 255, dtype=np.uint8)
        known_colors = set()
        for line in (annotations / 'RUGD_annotation-colormap.txt').read_text(encoding='utf-8').splitlines():
            index, name, r, g, b = line.split()
            packed = (int(r) << 16) | (int(g) << 8) | int(b)
            rgb_lut[packed] = cfg['rugd_id_to_target'][index]
            known_colors.add(packed)
        for label in sorted(annotations.glob('*/*.png')):
            seq = label.parent.name
            split = 'val' if seq in cfg['rugd_val_sequences'] else 'test' if seq in cfg['rugd_test_sequences'] else 'train'
            image = base / 'RUGD_frames-with-annotations' / seq / label.name
            jobs.append(('rugd', seq, split, split, image, label))
    if not jobs:
        raise RuntimeError('No dataset files found')
    camvid_lut = np.full(256, 255, dtype=np.uint8)
    for key, value in cfg['camvid_id_to_target'].items():
        camvid_lut[int(key)] = value

    def prepare(job):
        dataset, seq, split, original, image_path, label_path = job
        with Image.open(label_path) as source:
            size = source.size
            if dataset == 'camvid':
                native = np.asarray(source)
                if native.ndim != 2 or not set(np.unique(native)).issubset(set(range(12))):
                    raise ValueError(f'Invalid CamVid class IDs: {label_path}')
                mapped = camvid_lut[native]
            else:
                native = np.asarray(source.convert('RGB'), dtype=np.uint32)
                packed = (native[..., 0] << 16) | (native[..., 1] << 8) | native[..., 2]
                unexpected = set(np.unique(packed)) - known_colors
                if unexpected:
                    raise ValueError(f'Unrecognized annotation colors: {label_path}: {unexpected}')
                mapped = rgb_lut[packed]
        digest = None
        if not args.annotations_only:
            with Image.open(image_path) as image:
                if image.size != size:
                    raise ValueError(f'Image-mask size mismatch: {image_path}')
                image.verify()
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        mask_path = out / 'masks' / dataset / seq / label_path.name
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mapped).save(mask_path)
        hist = np.bincount(mapped.ravel(), minlength=256)
        record = {'dataset': dataset, 'sequence': seq, 'split': split, 'original_split': original,
                  'image': str(image_path.relative_to(ROOT)), 'mask': str(mask_path.relative_to(ROOT)),
                  'width': size[0], 'height': size[1], 'sha256': digest,
                  'class_pixels': hist[:len(cfg['names'])].tolist(), 'ignored_pixels': int(hist[255])}
        return record

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, record in enumerate(pool.map(prepare, jobs), 1):
            records.append(record)
            if i % 500 == 0:
                print(f'Validated {i}/{len(jobs)} labels/images', flush=True)
    sequence_splits, hash_splits = defaultdict(set), defaultdict(set)
    for r in records:
        sequence_splits[(r['dataset'], r['sequence'])].add(r['split'])
        if r['sha256']:
            hash_splits[r['sha256']].add(r['split'])
    assert all(len(s) == 1 for s in sequence_splits.values()), 'Source video crosses splits'
    assert all(len(s) == 1 for s in hash_splits.values()), 'Identical image crosses splits'
    stats = {'datasets': args.datasets, 'annotations_only': args.annotations_only,
             'total_images': len(records), 'class_names': cfg['names'], 'splits': {},
             'video_disjoint': True, 'cross_split_duplicate_images': 0,
             'label_config_sha256': hashlib.sha256((ROOT / 'config/road-labels.json').read_bytes()).hexdigest()}
    suffix = '-'.join(args.datasets)
    for split in ('train', 'val', 'test'):
        subset = [r for r in records if r['split'] == split]
        pixels = np.sum([r['class_pixels'] for r in subset], axis=0).astype(np.int64)
        stats['splits'][split] = {'images': len(subset), 'datasets': dict(Counter(r['dataset'] for r in subset)),
                                  'sequences': sorted({r['dataset'] + '/' + r['sequence'] for r in subset}),
                                  'class_pixels': pixels.tolist(),
                                  'missing_classes': [cfg['names'][i] for i, n in enumerate(pixels) if n == 0]}
        if not args.annotations_only:
            (out / f'{suffix}-{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in subset), encoding='utf-8')
            if len(args.datasets) > 1:
                for name in args.datasets:
                    (out / f'{name}-{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in subset if r['dataset'] == name), encoding='utf-8')
    (out / f'{suffix}-audit.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
