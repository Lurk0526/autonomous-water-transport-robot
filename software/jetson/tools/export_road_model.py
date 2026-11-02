#!/usr/bin/env python3
"""Export fixed-size FP32 ONNX and check model outputs against PyTorch."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.road_model import LABELS, load_model, resolve


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, images):
        return self.model(images).logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--height', type=int, default=384)
    p.add_argument('--width', type=int, default=512)
    a = p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(42)
    model = LogitsOnly(load_model(a.model)).cpu().eval()
    sample = torch.randn(1, 3, a.height, a.width)
    output = resolve(a.output); output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        reference = model(sample).numpy()
        torch.onnx.export(model, sample, str(output), opset_version=17, dynamo=False,
                          input_names=['images'], output_names=['logits'], do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(output)))
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), opts, providers=['CPUExecutionProvider'])
    actual = session.run(None, {'images': sample.numpy()})[0]
    passed = bool(np.allclose(reference, actual, atol=1e-4, rtol=1e-3))
    report = {'model': a.model, 'onnx': str(output), 'shape': list(sample.shape),
              'model_sha256': hashlib.sha256((resolve(a.model) / 'model.safetensors').read_bytes()).hexdigest(),
              'onnx_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'output_shape': list(actual.shape), 'classes': LABELS['names'], 'passed': passed,
              'max_abs_logit_error': float(np.max(np.abs(reference - actual))),
              'argmax_agreement': float(np.mean(reference.argmax(1) == actual.argmax(1))),
              'preprocess': 'RGB / 255, ImageNet normalization, aspect-preserving letterbox; see vision/road_model.py',
              'postprocess': 'bilinear resize logits to letterbox size; remove padding; resize to original image; argmax',
              'deployment_ready': False}
    Path(str(output) + '.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    if not passed: raise RuntimeError('ONNX outputs differ from PyTorch beyond tolerance')


if __name__ == '__main__': main()
