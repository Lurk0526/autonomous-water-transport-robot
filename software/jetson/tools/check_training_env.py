#!/usr/bin/env python3
"""Check Windows/desktop training dependencies without reading camera or serial."""
import argparse
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    import torch
    import torchvision
    import ultralytics
    from torchvision.ops import nms

    report = {'platform': platform.platform(), 'python': sys.version.split()[0],
              'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'ultralytics': ultralytics.__version__, 'cuda_runtime': torch.version.cuda,
              'cuda_available': torch.cuda.is_available()}
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Check NVIDIA driver and install the GPU PyTorch wheel. ' + json.dumps(report))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda'):
        props = torch.cuda.get_device_properties(0)
        report.update(gpu=props.name, vram_gib=round(props.total_memory / 1024**3, 2),
                      compute_capability=list(torch.cuda.get_device_capability(0)))
    a = torch.randn(64, 64, device=device)
    assert torch.isfinite(a @ a).all().item()
    conv = torch.nn.Conv2d(3, 4, 3).to(device)
    conv(torch.randn(2, 3, 16, 16, device=device)).square().mean().backward()
    kept = nms(torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device),
               torch.tensor([.9, .8], device=device), .5)
    assert kept.tolist() == [0]
    report.update(tensor_conv_backward_nms='PASS', result='PASS')
    output = Path(__file__).resolve().parents[1] / 'work/training-env.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
