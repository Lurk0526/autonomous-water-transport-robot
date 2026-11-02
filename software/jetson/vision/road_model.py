"""Semantic appearance model for the outdoor water-cart project."""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((ROOT / 'config/road-labels.json').read_text(encoding='utf-8'))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def letterbox(image, mask, size):
    """Preserve aspect ratio; label padding is always ignored."""
    height, width = size
    ratio = min(width / image.width, height / image.height)
    dims = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    xy = ((width - dims[0]) // 2, (height - dims[1]) // 2)
    canvas = Image.new('RGB', (width, height), tuple((MEAN * 255).astype(int)))
    canvas.paste(image.resize(dims, Image.Resampling.BILINEAR), xy)
    target = None
    if mask is not None:
        target = Image.new('L', (width, height), 255)
        target.paste(mask.resize(dims, Image.Resampling.NEAREST), xy)
    return canvas, target, (xy[0], xy[1], dims[0], dims[1])


def tensor_image(image):
    array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class RoadDataset(Dataset):
    def __init__(self, manifest, size, augment=False, limit=None):
        self.records = [json.loads(line) for line in resolve(manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
        if limit:
            # Deterministic evenly spaced smoke samples, rather than adjacent video frames.
            indexes = np.linspace(0, len(self.records) - 1, min(limit, len(self.records)), dtype=int)
            self.records = [self.records[i] for i in indexes]
        if not self.records:
            raise ValueError('Empty dataset')
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(resolve(record['image'])) as source:
            image = source.convert('RGB')
        with Image.open(resolve(record['mask'])) as source:
            mask = source.convert('L')
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for enhancer, span in ((ImageEnhance.Brightness, 0.3), (ImageEnhance.Contrast, 0.2), (ImageEnhance.Color, 0.2)):
                image = enhancer(image).enhance(random.uniform(1 - span, 1 + span))
            if random.random() < 0.15:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.8)))
            if random.random() < 0.15:
                image = image.filter(ImageFilter.Kernel((3, 3), [0, 0, 0, 1, 1, 1, 0, 0, 0], scale=3))
        image, mask, _ = letterbox(image, mask, self.size)
        return tensor_image(image), torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())


def load_model(path, task_head=True):
    model = SegformerForSemanticSegmentation.from_pretrained(resolve(path), local_files_only=True)
    if task_head and model.config.num_labels != len(LABELS['names']):
        if model.config.num_labels != 19:
            raise ValueError('Expected the pinned 19-class Cityscapes initialization')
        old = model.decode_head.classifier
        new = torch.nn.Conv2d(old.in_channels, len(LABELS['names']), 1)
        torch.nn.init.normal_(new.weight, std=model.config.initializer_range)
        torch.nn.init.zeros_(new.bias)
        with torch.no_grad():
            for dst, sources in enumerate(LABELS['cityscapes_groups_for_initialization']):
                if sources:
                    new.weight[dst].copy_(old.weight[sources].mean(0))
                    new.bias[dst].copy_(old.bias[sources].mean(0))
        model.decode_head.classifier = new
        model.config.id2label = dict(enumerate(LABELS['names']))
        model.config.label2id = {name: i for i, name in enumerate(LABELS['names'])}
        model.config.num_labels = len(LABELS['names'])
    if task_head and list(model.config.id2label.values()) != LABELS['names']:
        raise ValueError('Checkpoint class order does not match road-labels.json')
    return model


def update_confusion(confusion, prediction, target):
    valid = target != 255
    if valid.any():
        n = confusion.shape[0]
        confusion += torch.bincount((target[valid] * n + prediction[valid]).long(), minlength=n * n).reshape(n, n)


def metrics(confusion):
    matrix = np.asarray(confusion, dtype=np.float64)
    actual, predicted = matrix.sum(1), matrix.sum(0)
    diagonal = matrix.diagonal()
    union = actual + predicted - diagonal
    iou = np.divide(diagonal, union, out=np.full_like(union, np.nan), where=union > 0)
    present = actual > 0
    return {
        'mIoU': float(np.nanmean(iou)),
        'pixel_accuracy': float(diagonal.sum() / max(matrix.sum(), 1)),
        'ground_truth_present_class_mIoU': float(np.mean(iou[present])) if present.any() else None,
        'per_class_IoU': {name: float(iou[i]) if np.isfinite(iou[i]) else None for i, name in enumerate(LABELS['names'])},
        'paved_precision': float(diagonal[0] / max(predicted[0], 1)),
        'paved_recall': float(diagonal[0] / max(actual[0], 1)),
        'non_paved_predicted_paved_rate': float(matrix[1:, 0].sum() / max(actual[1:].sum(), 1)),
        'ground_truth_pixels': actual.astype(int).tolist(), 'confusion_matrix': matrix.astype(int).tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, device, amp):
    model.eval()
    confusion = torch.zeros(len(LABELS['names']), len(LABELS['names']), dtype=torch.int64, device=device)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images).logits
        logits = F.interpolate(logits.float(), size=labels.shape[-2:], mode='bilinear', align_corners=False)
        update_confusion(confusion, logits.argmax(1), labels)
    return metrics(confusion.cpu().numpy())
