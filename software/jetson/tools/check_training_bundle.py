#!/usr/bin/env python3
"""Check portable training bundle hashes, split isolation and optional CUDA execution."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full', action='store_true', help='Verify the SHA256 of every packaged file')
    p.add_argument('--gpu', action='store_true', help='Run a CUDA operation in the active Python environment')
    a = p.parse_args()
    manifest = json.loads((ROOT / 'BUNDLE_FILES.json').read_text(encoding='utf-8'))
    errors = []
    for i, item in enumerate(manifest['files'], 1):
        path = ROOT / item['path']
        if not path.is_file() or path.stat().st_size != item['bytes']:
            errors.append('Missing file or wrong size: ' + item['path']); continue
        if a.full:
            with path.open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
            if digest != item['sha256']: errors.append('SHA256 mismatch: ' + item['path'])
        if a.full and i % 2000 == 0: print(f'Checked {i}/{len(manifest["files"])} files', flush=True)
    sequences, hashes, counts = {}, {}, {}
    for split in ('train', 'val', 'test'):
        records = [json.loads(l) for l in (ROOT / f'datasets/public_road/prepared/camvid-rugd-{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        counts[split] = len(records)
        for r in records:
            key = (r['dataset'], r['sequence'])
            if sequences.setdefault(key, split) != split: errors.append('Video appears in multiple splits: ' + str(key))
            if hashes.setdefault(r['sha256'], split) != split: errors.append('Image appears in multiple splits')
            for field in ('image', 'mask'):
                target = (ROOT / r[field]).resolve()
                if not target.is_relative_to(ROOT) or not target.is_file(): errors.append('Broken dataset path: ' + r[field])
    result = {'files': len(manifest['files']), 'full_hashes_checked': a.full, 'splits': counts, 'errors': errors}
    if a.gpu:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable: check NVIDIA driver and CUDA PyTorch install')
        tensor = torch.ones((32, 32), device='cuda')
        value = float((tensor @ tensor)[0, 0]); torch.cuda.synchronize()
        if value != 32: raise RuntimeError('CUDA matrix operation failed')
        prop = torch.cuda.get_device_properties(0)
        result['gpu'] = {'name': prop.name, 'memory_gb': round(prop.total_memory / 2**30, 2),
                         'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'operation_passed': True}
    result['passed'] = not errors
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
