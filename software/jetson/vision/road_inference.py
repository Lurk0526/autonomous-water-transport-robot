"""Pinned road model inference, preserving the original camera pixel coordinates."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from vision.road_model import LABELS, letterbox, load_model, tensor_image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrtLogits:
    def __init__(self, path, shape):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT deserialization failed')
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.shape = tuple(shape)
        if tuple(self.engine.get_tensor_shape('images')) != self.shape:
            raise ValueError('TensorRT input shape mismatch')
        if self.engine.get_tensor_dtype('images') != trt.float32 or self.engine.get_tensor_dtype('logits') != trt.float32:
            raise ValueError('Expected FP32 engine input/output buffers')
        self.output = torch.empty(tuple(self.engine.get_tensor_shape('logits')), dtype=torch.float32, device='cuda')

    def __call__(self, images):
        images = images.contiguous()
        if tuple(images.shape) != self.shape or images.dtype != torch.float32:
            raise ValueError('Invalid TensorRT input')
        if not self.context.set_tensor_address('images', images.data_ptr()) or not self.context.set_tensor_address('logits', self.output.data_ptr()):
            raise RuntimeError('TensorRT tensor binding failed')
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        images.record_stream(self.stream)
        self.output.record_stream(self.stream)
        caller.wait_stream(self.stream)
        return self.output


class RoadPredictor:
    def __init__(self, deployment, backend='auto', rotation=0):
        self.path = Path(deployment)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        checks = [('model/model.safetensors', 'model_sha256'), ('model/config.json', 'config_sha256'), ('labels.json', 'labels_sha256')]
        for file, key in checks:
            if sha(self.path / file) != self.manifest[key]:
                raise ValueError(f'Deployment checksum mismatch: {file}')
        if self.manifest['classes'] != LABELS['names']:
            raise ValueError('Model and runtime label order mismatch')
        if rotation not in (0, 90, 180, 270):
            raise ValueError('Input rotation must be 0, 90, 180 or 270')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; launch with GPU device access')
        if backend not in ('auto', 'torch', 'tensorrt'):
            raise ValueError('Unknown inference backend')
        self.rotation = rotation // 90
        self.size = self.manifest['input_size']
        self.backend = 'torch'
        self.fallback_reason = None
        plan = self.path / 'accelerator/model.plan'
        report = self.path / 'accelerator/model.plan.json'
        if backend != 'torch' and plan.exists() and report.exists():
            try:
                info = json.loads(report.read_text())
                import tensorrt as trt
                if not info['passed'] or info['model_sha256'] != self.manifest['model_sha256'] or info['engine_sha256'] != sha(plan):
                    raise ValueError('TensorRT evidence/hash mismatch')
                if info['tensorrt'] != trt.__version__ or info['device_capability'] != list(torch.cuda.get_device_capability(0)) or info['device'] != torch.cuda.get_device_name(0):
                    raise ValueError('TensorRT engine belongs to another runtime/device')
                self.model = TrtLogits(plan, [1, 3, *self.size])
                self.backend = 'tensorrt'
            except Exception as exc:
                if backend == 'tensorrt':
                    raise
                self.fallback_reason = str(exc)
        elif backend == 'tensorrt':
            raise FileNotFoundError('A verified TensorRT engine is required')
        if self.backend == 'torch':
            self.model = load_model(self.path / 'model').cuda().eval()
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, *self.size, device='cuda')
            for _ in range(3):
                self.logits(dummy)
        torch.cuda.synchronize()

    def logits(self, tensor):
        if self.backend == 'tensorrt':
            return self.model(tensor)
        with torch.autocast('cuda'):
            return self.model(tensor).logits.float()

    @torch.inference_mode()
    def predict(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected RGB8 image')
        start = time.perf_counter()
        canonical = np.ascontiguousarray(np.rot90(rgb, self.rotation))
        image = Image.fromarray(canonical)
        padded, _, (x, y, w, h) = letterbox(image, None, self.size)
        tensor = tensor_image(padded).unsqueeze(0).cuda()
        logits = self.logits(tensor).float()
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError('Non-finite model logits')
        logits = F.interpolate(logits, size=self.size, mode='bilinear', align_corners=False)
        logits = F.interpolate(logits[:, :, y:y+h, x:x+w], size=canonical.shape[:2], mode='bilinear', align_corners=False)
        confidence, labels = logits.softmax(1).max(1)
        labels = np.rot90(labels[0].cpu().numpy().astype(np.uint8), -self.rotation).copy()
        confidence = np.rot90(confidence[0].cpu().numpy(), -self.rotation).copy()
        return labels, confidence, (time.perf_counter() - start) * 1000
