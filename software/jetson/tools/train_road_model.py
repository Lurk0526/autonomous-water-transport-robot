#!/usr/bin/env python3
"""Bounded, reproducible SegFormer training with validation-only model selection."""
import argparse
import math
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.road_model import LABELS, ROOT, RoadDataset, evaluate, load_model, resolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(2)
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by this configuration')
    amp = cfg['amp'] and device.type == 'cuda'
    out = resolve(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists() and args.resume is None:
        raise RuntimeError(f'{out} already has a finished run; select a new output directory or explicitly resume')
    train = RoadDataset(cfg['train_manifest'], cfg['size'], True, cfg.get('max_train_samples'))
    val = RoadDataset(cfg['val_manifest'], cfg['size'], False, cfg.get('max_val_samples'))
    if any(r['split'] != 'train' for r in train.records) or any(r['split'] != 'val' for r in val.records):
        raise ValueError('Training/selection manifest uses an incorrect split')
    train_sequences = {(r['dataset'], r['sequence']) for r in train.records}
    val_sequences = {(r['dataset'], r['sequence']) for r in val.records}
    if train_sequences & val_sequences:
        raise ValueError('Video leakage between training and validation')
    generator = torch.Generator().manual_seed(cfg['seed'])
    sampler = None
    if cfg.get('dataset_sampling_weights'):
        counts = {name: sum(r['dataset'] == name for r in train.records) for name in cfg['dataset_sampling_weights']}
        sample_weights = [cfg['dataset_sampling_weights'][r['dataset']] / counts[r['dataset']] for r in train.records]
        sampler = WeightedRandomSampler(sample_weights, len(train), replacement=True, generator=generator)
    train_loader = DataLoader(train, batch_size=cfg['batch'], shuffle=sampler is None, sampler=sampler,
                              num_workers=cfg['workers'], generator=generator)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=cfg['workers'])
    initial = args.resume / 'model' if args.resume else cfg['initial_model']
    model = load_model(initial).to(device)
    if cfg['freeze_encoder']:
        model.segformer.requires_grad_(False)
    groups = [
        {'params': [p for p in model.segformer.parameters() if p.requires_grad], 'lr': cfg['encoder_lr']},
        {'params': model.decode_head.parameters(), 'lr': cfg['head_lr']},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    planned_steps = min(cfg.get('max_steps', len(train_loader) * cfg['epochs']), len(train_loader) * cfg['epochs'])
    whole_epochs, tail_steps = divmod(planned_steps, len(train_loader))
    total_updates = max(1, whole_epochs * math.ceil(len(train_loader) / cfg['accumulation'])
                        + math.ceil(tail_steps / cfg['accumulation']))
    warmup_updates = min(100, max(1, total_updates // 20))
    base_lrs = [cfg['encoder_lr'], cfg['head_lr']]
    scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=1024.0)
    hist = np.sum([r['class_pixels'] for r in train.records], axis=0).astype(float)
    freq = hist / max(hist.sum(), 1)
    weights = np.clip(1 / np.sqrt(np.maximum(freq, 1e-4)), 0.5, 4.0)
    weights /= weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    epoch_start, steps, updates, best, losses = 0, 0, 0, -1.0, []
    if args.resume:
        previous_run = args.resume.parent / 'run.json'
        if previous_run.is_file() and json.loads(previous_run.read_text(encoding='utf-8'))['config'] != cfg:
            raise ValueError('Resume configuration differs from the saved run; start a new output directory')
        state = torch.load(args.resume / 'trainer.pt', map_location='cpu', weights_only=False)
        if state['label_sha256'] != hashlib.sha256((ROOT / 'config/road-labels.json').read_bytes()).hexdigest():
            raise ValueError('Label mapping changed since the checkpoint')
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        epoch_start, steps, updates, best = state['epoch'] + 1, state['steps'], state['updates'], state['best']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); generator.set_state(state['loader_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
    metadata = {'config': cfg, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'train_images': len(train), 'val_images': len(val), 'class_weights': weights.cpu().tolist(),
                'label_sha256': hashlib.sha256((ROOT / 'config/road-labels.json').read_bytes()).hexdigest(),
                'train_manifest_sha256': hashlib.sha256(resolve(cfg['train_manifest']).read_bytes()).hexdigest(),
                'val_manifest_sha256': hashlib.sha256(resolve(cfg['val_manifest']).read_bytes()).hexdigest(),
                'status': 'running', 'smoke_only': cfg.get('smoke_only', False)}
    (out / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    started = time.monotonic()
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    skipped_updates, consecutive_overflows = 0, 0

    def apply_update(pending):
        nonlocal updates, skipped_updates, consecutive_overflows
        scaler.unscale_(optimizer)
        if pending != cfg['accumulation']:
            for parameter in model.parameters():
                if parameter.grad is not None: parameter.grad.mul_(cfg['accumulation'] / pending)
        gradients_finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        if not gradients_finite:
            consecutive_overflows += 1
            if not amp or scaler.get_scale() <= 1 or consecutive_overflows > 8:
                raise RuntimeError('Persistent non-finite gradients after AMP scale reduction')
            # GradScaler skips this update and reduces its scale; do not clip Inf/NaN gradients.
            previous = scaler.get_scale()
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            skipped_updates += 1
            print(json.dumps({'amp_overflow_skipped': True, 'scale_before': previous,
                              'scale_after': scaler.get_scale(), 'step': steps}), flush=True)
            return
        consecutive_overflows = 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        updates += 1

    def set_learning_rates():
        warmup = min(1.0, (updates + 1) / warmup_updates)
        decay = max(0.0, 1 - updates / total_updates) ** 0.9
        for group, base in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base * warmup * decay

    def save_checkpoint(name, epoch):
        target = out / name
        model.save_pretrained(target / 'model', safe_serialization=True)
        torch.save({'epoch': epoch, 'steps': steps, 'updates': updates, 'best': best,
                    'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'label_sha256': metadata['label_sha256'], 'python_rng': random.getstate(),
                    'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                    'loader_rng': generator.get_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, target / 'trainer.pt')

    for epoch in range(epoch_start, cfg['epochs']):
        model.train()
        if cfg['freeze_encoder']: model.segformer.eval()
        # Small Jetson batches need frozen BatchNorm statistics in the decode head.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for images, labels in train_loader:
            if cfg.get('max_steps') and steps >= cfg['max_steps']: break
            images, labels = images.to(device), labels.to(device)
            if not (labels != 255).any(): continue
            set_learning_rates()
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images).logits
                logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                loss = F.cross_entropy(logits.float(), labels, weight=weights, ignore_index=255)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            scaler.scale(loss / cfg['accumulation']).backward()
            pending += 1; steps += 1; losses.append(float(loss.detach()))
            if pending == cfg['accumulation']:
                apply_update(pending)
                pending = 0
            if steps % cfg['log_every'] == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': steps, 'loss': losses[-1]}), flush=True)
        if pending:
            apply_update(pending)
        score = evaluate(model, val_loader, device, amp)
        # Keep RUGD's larger validation set from hiding poor paved-road performance.
        dataset_scores = {}
        if len({r['dataset'] for r in val.records}) > 1:
            for name in sorted({r['dataset'] for r in val.records}):
                indexes = [i for i, r in enumerate(val.records) if r['dataset'] == name]
                subset_loader = DataLoader(torch.utils.data.Subset(val, indexes), batch_size=1, num_workers=cfg['workers'])
                dataset_scores[name] = evaluate(model, subset_loader, device, amp)
        selection_score = float(np.mean([s['mIoU'] for s in dataset_scores.values()])) if dataset_scores else score['mIoU']
        item = {'epoch': epoch + 1, 'steps': steps, 'train_loss': float(np.mean(losses[-len(train_loader):])),
                'validation': score, 'validation_per_dataset': dataset_scores, 'selection_score': selection_score}
        with (out / 'history.jsonl').open('a') as f: f.write(json.dumps(item) + '\n')
        if selection_score > best:
            best = selection_score; save_checkpoint('best', epoch)
        save_checkpoint('last', epoch)
        print(json.dumps(item), flush=True)
        if cfg.get('max_steps') and steps >= cfg['max_steps']: break
    if not losses and not args.resume: raise RuntimeError('No training steps performed')
    if updates == 0: raise RuntimeError('No successful optimizer updates performed')
    result = {**metadata, 'status': 'completed', 'steps': steps, 'optimizer_updates': updates,
              'elapsed_seconds': time.monotonic() - started, 'best_validation_selection_score': best,
              'elapsed_scope': 'current invocation; earlier completed epochs remain in history.jsonl',
              'resumed_from': str(args.resume) if args.resume else None,
              'amp_skipped_updates_this_invocation': skipped_updates, 'final_amp_scale': scaler.get_scale(),
              'first_loss': losses[0] if losses else None, 'last_loss': losses[-1] if losses else None,
              'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else None,
              'deployment_ready': False}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
