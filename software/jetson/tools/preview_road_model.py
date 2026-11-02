#!/usr/bin/env python3
"""Render public validation images and GPU inference timings without controlling a robot."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.road_model import LABELS, ROOT, letterbox, tensor_image, load_model, resolve


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='work/models/road/segformer-b0-cityscapes')
    p.add_argument('--manifest', default='datasets/public_road/prepared/camvid-val.jsonl')
    p.add_argument('--output', default='work/road_training/preview-pretrained')
    p.add_argument('--count', type=int, default=4)
    a = p.parse_args()
    out = resolve(a.output); out.mkdir(parents=True, exist_ok=True)
    model = load_model(a.model, task_head=False).cuda().eval()
    records = [json.loads(l) for l in resolve(a.manifest).read_text(encoding='utf-8').splitlines()]
    indexes = np.linspace(0, len(records) - 1, min(a.count, len(records)), dtype=int)
    timings, rows = [], []
    cityscapes = model.config.num_labels == 19
    # For the pretrained model, preserve its native 19-class task.
    city_palette = np.array([[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
                             [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
                             [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],[0,60,100],
                             [0,80,100],[0,0,230],[119,11,32]], dtype=np.uint8)
    palette = city_palette if cityscapes else np.array(LABELS['palette'], dtype=np.uint8)
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 384, 512, device='cuda')
        for _ in range(3):
            with torch.autocast('cuda'): model(dummy)
        for index in indexes:
            record = records[index]
            image = Image.open(resolve(record['image'])).convert('RGB')
            resized, _, crop = letterbox(image, None, (384, 512))
            tensor = tensor_image(resized).unsqueeze(0).cuda()
            torch.cuda.synchronize(); start = time.perf_counter()
            with torch.autocast('cuda'): logits = model(tensor).logits
            torch.cuda.synchronize(); timings.append((time.perf_counter() - start) * 1000)
            logits = F.interpolate(logits.float(), size=(384, 512), mode='bilinear', align_corners=False)
            mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            x, y, w, h = crop
            mask = np.asarray(Image.fromarray(mask[y:y+h, x:x+w]).resize(image.size, Image.Resampling.NEAREST))
            overlay = Image.blend(image, Image.fromarray(palette[mask]), 0.45)
            pair = Image.new('RGB', (image.width * 2, image.height + 24), '#161c25')
            pair.paste(image, (0, 24)); pair.paste(overlay, (image.width, 24))
            ImageDraw.Draw(pair).text((8, 5), f'{record["dataset"]}/{record["sequence"]}  |  RGB  ->  semantic prediction (prototype)', fill='white')
            pair.save(out / (Path(record['image']).stem + '.jpg'))
            rows.append(pair.resize((960, round(pair.height * 960 / pair.width))))
        board = Image.new('RGB', (960, sum(r.height for r in rows)), '#161c25')
        offset = 0
        for row in rows: board.paste(row, (0, offset)); offset += row.height
        board.save(out / 'contact-sheet.jpg')
    report = {'model': a.model, 'model_classes': model.config.id2label, 'input_size': [384,512],
              'forward_ms': timings, 'median_forward_ms': float(np.median(timings)),
              'timing_scope': 'GPU model forward only; excludes ROS, preprocessing, transfer and postprocessing',
              'validation_images': len(rows), 'deployment_ready': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
