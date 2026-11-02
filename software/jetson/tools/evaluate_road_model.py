#!/usr/bin/env python3
"""Evaluate a selected checkpoint on a requested held-out manifest."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.road_model import RoadDataset, evaluate, load_model, resolve


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--height', type=int, default=384)
    p.add_argument('--width', type=int, default=512)
    a = p.parse_args()
    torch.set_num_threads(2)
    data = RoadDataset(a.manifest, (a.height, a.width))
    model = load_model(a.model).cuda().eval()
    result = evaluate(model, DataLoader(data, batch_size=1, num_workers=0), torch.device('cuda'), True)
    result.update({'model': a.model, 'manifest': a.manifest, 'images': len(data),
                   'size': [a.height, a.width], 'deployment_ready': False})
    output = resolve(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
