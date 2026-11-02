#!/usr/bin/env python3
"""Validate reviewed Labelme polygons and export YOLO instance segmentation.

Source layout: source/{train,val,test}/<capture-session>/*.{png,jpg,json}.
Empty JSON is accepted only when image flag negative=true is explicit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from PIL import Image

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
SPLITS = ('train', 'val', 'test')


def validate_polygon(points, width, height):
    points = [(float(x), float(y)) for x, y in points]
    if len(points) > 1 and points[-1] == points[0]:
        points.pop()
    if len(set(points)) < 3 or len(set(points)) != len(points):
        raise ValueError('多边形至少需要三个不同顶点，且不能重复顶点')
    for x, y in points:
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
            raise ValueError('轮廓坐标无效或超出图像范围')
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
    if abs(area) < 1e-3:
        raise ValueError('轮廓面积为零')

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersects(a, b, c, d):
        if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
            return False
        if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
            return False
        return cross(a, b, c) * cross(a, b, d) <= 0 and cross(c, d, a) * cross(c, d, b) <= 0

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if j == i + 1 or (i == 0 and j == len(points) - 1):
                continue
            if intersects(points[i], points[(i + 1) % len(points)], points[j], points[(j + 1) % len(points)]):
                raise ValueError('轮廓存在自交，请修正交叉的边')
    return '0 ' + ' '.join(f'{value:.8f}' for x, y in points for value in (x / width, y / height))


def inspect_source(source):
    source = source.resolve()
    images = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError('没有找到待标注图像')
    rows, errors, hashes, sessions = [], [], {}, {}
    for image in images:
        try:
            relative = image.relative_to(source)
            split = relative.parts[0]
            if split not in SPLITS or len(relative.parts) < 2:
                raise ValueError('图片必须位于 source/train、source/val 或 source/test 下')
            if len(relative.parts) > 2:
                session = relative.parts[1]
                if session in sessions and sessions[session] != split:
                    raise ValueError(f'同一拍摄批次 {session} 出现在不同数据划分中')
                sessions[session] = split
            annotation = image.with_suffix('.json')
            if not annotation.exists():
                raise ValueError('尚无 JSON 标注；未标注图不能自动当成负样本')
            value = json.loads(annotation.read_text(encoding='utf-8-sig'))
            image_path = value.get('imagePath')
            if not image_path or (annotation.parent / image_path).resolve() != image.resolve():
                raise ValueError('JSON 的 imagePath 与原图不一致')
            with Image.open(image) as actual:
                width, height = actual.size
            if (value.get('imageWidth'), value.get('imageHeight')) != (width, height):
                raise ValueError('JSON 图像尺寸与原图不一致')
            shapes = value.get('shapes')
            if not isinstance(shapes, list):
                raise ValueError('缺少 shapes 标注列表')
            negative = value.get('flags', {}).get('negative') is True
            if not shapes and not negative:
                raise ValueError('空标注需勾选 negative 并保存，确认无目标碗')
            if shapes and negative:
                raise ValueError('同时存在目标轮廓和 negative 标记，请核实')
            labels, groups = [], set()
            for shape in shapes:
                if shape.get('label') != 'target_bowl' or shape.get('shape_type') != 'polygon':
                    raise ValueError('只接受 target_bowl 类别的 polygon 多边形')
                group = shape.get('group_id')
                if group is not None:
                    if group in groups:
                        raise ValueError('同一实例有多个分离轮廓；请先核实为单个多边形')
                    groups.add(group)
                labels.append(validate_polygon(shape['points'], width, height))
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes and hashes[digest] != split:
                raise ValueError('相同图片内容出现在不同划分中，存在数据泄漏')
            hashes[digest] = split
            name_hash = hashlib.sha256(relative.as_posix().encode('utf-8')).hexdigest()[:12]
            name = f'{image.stem}-{name_hash}'
            rows.append({'source': image, 'relative': relative.as_posix(), 'split': split,
                         'name': name, 'suffix': image.suffix.lower(), 'labels': labels,
                         'negative': negative, 'sha256': digest})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(f'{image.relative_to(source)}: {exc}')
    for split in SPLITS:
        in_split = [r for r in rows if r['split'] == split]
        if not in_split or not any(r['labels'] for r in in_split):
            errors.append(f'{split}: 至少需要一张已标注的目标碗图片')
    if errors:
        raise ValueError('\n'.join(errors[:30]) + (f'\n另有 {len(errors)-30} 项问题' if len(errors) > 30 else ''))
    return rows


def export_dataset(source, output, check_only=False):
    rows = inspect_source(source)
    summary = {split: {'images': sum(r['split'] == split for r in rows),
                       'negative': sum(r['split'] == split and r['negative'] for r in rows)} for split in SPLITS}
    if check_only:
        return summary
    output = output.resolve()
    if output == source.resolve() or source.resolve() in output.parents:
        raise ValueError('导出目录应放在 source 外，避免混入原始标注数据')
    if output.exists():
        raise ValueError(f'输出目录已存在，为保留旧版本请换一个目录：{output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.bowl-export-', dir=output.parent))
    try:
        manifest = []
        for row in rows:
            image_dir = staging / 'images' / row['split']
            label_dir = staging / 'labels' / row['split']
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row['source'], image_dir / (row['name'] + row['suffix']))
            (label_dir / (row['name'] + '.txt')).write_text('\n'.join(row['labels']) + ('\n' if row['labels'] else ''), encoding='utf-8')
            manifest.append({k: v for k, v in row.items() if k not in ('source', 'labels')})
        yaml = f'path: {json.dumps(str(output), ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: target_bowl\n'
        (staging / 'data.yaml').write_text(yaml, encoding='utf-8')
        (staging / 'manifest.json').write_text(json.dumps({'summary': summary, 'images': manifest}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root / 'datasets/target_bowl/source')
    parser.add_argument('--output', type=Path, default=root / 'datasets/target_bowl/yolo')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(export_dataset(args.source, args.output, args.check_only), ensure_ascii=False, indent=2))
    except ValueError as exc:
        parser.exit(1, f'尚不能导出：\n{exc}\n')


if __name__ == '__main__':
    main()
