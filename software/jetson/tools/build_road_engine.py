#!/usr/bin/env python3
"""Build an Orin-local TensorRT engine and compare real-image logits to PyTorch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import tensorrt as trt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.road_inference import TrtLogits
from vision.road_model import letterbox, load_model, tensor_image, resolve


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment', type=Path, default=ROOT / 'work/models/road/road-4060-v1')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--count', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    deployment = args.deployment
    manifest = json.loads((deployment / 'manifest.json').read_text())
    accelerator = deployment / 'accelerator'
    onnx = accelerator / 'model.onnx'
    export = json.loads(Path(str(onnx) + '.json').read_text())
    if not export['passed'] or sha(deployment / 'model/model.safetensors') != manifest['model_sha256']:
        raise ValueError('Verified model and ONNX export required')
    if export.get('model_sha256') != manifest['model_sha256'] or export.get('onnx_sha256') != sha(onnx):
        raise ValueError('ONNX export provenance mismatch; export the pinned model again')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser_onnx = trt.OnnxParser(network, logger)
    if not parser_onnx.parse(onnx.read_bytes()):
        raise RuntimeError('\n'.join(str(parser_onnx.get_error(i)) for i in range(parser_onnx.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 * 1024 * 1024)
    config.builder_optimization_level = 1
    if not args.fp32:
        config.set_flag(trt.BuilderFlag.FP16)
    start = time.monotonic()
    print('Building TensorRT', trt.__version__, 'FP32' if args.fp32 else 'FP16', flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT build failed')
    plan = accelerator / 'model.plan'
    temporary = accelerator / 'model.plan.pending'
    temporary.write_bytes(bytes(serialized))
    engine = TrtLogits(temporary, [1, 3, *manifest['input_size']])
    model = load_model(deployment / 'model').cuda().eval()
    records = [json.loads(line) for line in (ROOT / 'datasets/public_road/prepared/camvid-rugd-val.jsonl').read_text().splitlines() if line.strip()]
    indexes = np.linspace(0, len(records) - 1, min(args.count, len(records)), dtype=int)
    comparisons = []
    with torch.inference_mode():
        for index in indexes:
            record = records[index]
            image = Image.open(resolve(record['image'])).convert('RGB')
            padded, _, _ = letterbox(image, None, manifest['input_size'])
            tensor = tensor_image(padded).unsqueeze(0).cuda()
            reference = model(tensor).logits.float()
            actual = engine(tensor)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(actual).all())
            error = (reference - actual).abs()
            a = F.interpolate(actual, size=manifest['input_size'], mode='bilinear', align_corners=False).argmax(1)
            b = F.interpolate(reference, size=manifest['input_size'], mode='bilinear', align_corners=False).argmax(1)
            item = {'image': record['image'], 'finite': finite, 'max_abs_logit_error': float(error.max()),
                    'mean_abs_logit_error': float(error.mean()), 'pixel_argmax_agreement': float((a == b).float().mean())}
            comparisons.append(item)
            print('Compare', int(index), 'agreement', item['pixel_argmax_agreement'], flush=True)
        sample = tensor
        for _ in range(5):
            engine(sample)
        torch.cuda.synchronize()
        timings = []
        for _ in range(30):
            begin = time.perf_counter()
            engine(sample)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - begin) * 1000)
    passed = all(r['finite'] and r['pixel_argmax_agreement'] >= 0.995 and r['mean_abs_logit_error'] <= 0.03 for r in comparisons)
    report = {'passed': passed, 'model_sha256': manifest['model_sha256'], 'onnx_sha256': sha(onnx),
              'engine_sha256': sha(temporary), 'tensorrt': trt.__version__, 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(0), 'device_capability': list(torch.cuda.get_device_capability(0)),
              'precision': 'FP32' if args.fp32 else 'FP16', 'workspace_mb': 256,
              'build_seconds': time.monotonic() - start, 'comparisons': comparisons,
              'acceptance': 'Every sampled validation image: finite logits, pixel argmax agreement >= 99.5%, mean absolute logit error <= 0.03',
              'median_forward_ms': float(np.median(timings)), 'timing_scope': 'Model forward only, batch 1; excludes camera/ROS/preprocess/fusion',
              'field_validated': False}
    Path(str(plan) + '.json').write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise RuntimeError('TensorRT numerical verification failed; pending engine not activated')
    temporary.replace(plan)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
