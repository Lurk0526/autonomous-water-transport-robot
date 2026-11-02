"""Target-bowl segmentation and image-coordinate measurements only."""
import time

import cv2
import numpy as np


class BowlDetector:
    def __init__(self, weights, imgsz=640, conf=0.5, device='0'):
        import torch
        from ultralytics import YOLO

        if device != 'cpu' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
        self.model = YOLO(str(weights))
        if self.model.task != 'segment' or self.model.names != {0: 'target_bowl'}:
            raise ValueError('Expected a single-class target_bowl segmentation model')
        self.options = dict(imgsz=imgsz, conf=conf, device=device,
                            rect=False, retina_masks=True, verbose=False)
        self.infer(np.zeros((240, 320, 3), dtype=np.uint8))
        # Ultralytics deep-copies the source model when constructing its actual
        # predictor. Inspect the executing backend rather than the CPU source.
        parameter=next(self.model.predictor.model.model.parameters())
        self.runtime_info={'inference_backend':'pytorch_cuda' if parameter.is_cuda else 'pytorch_cpu',
                           'inference_device':str(parameter.device),
                           'inference_precision':str(parameter.dtype),'model_format':'pytorch'}

    def infer(self, rgb):
        # Ultralytics NumPy inputs use BGR; the camera/ROS input uses RGB.
        start = time.monotonic()
        result = self.model.predict(np.ascontiguousarray(rgb[:, :, ::-1]), **self.options)[0]
        overlay = rgb.copy()
        detections = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            for box, mask in zip(boxes, masks):
                if not np.isfinite(box).all() or not np.isfinite(mask).all():
                    raise ValueError('Non-finite model output')
                binary = (mask > 0.5).astype(np.uint8)
                if binary.shape != rgb.shape[:2]:
                    raise ValueError('Segmentation mask size does not match the source frame')
                moments = cv2.moments(binary, binaryImage=True)
                if not moments['m00']:
                    continue
                center = [moments['m10'] / moments['m00'], moments['m01'] / moments['m00']]
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                selected = binary.astype(bool)
                overlay[selected] = (overlay[selected] * 0.65 + np.array([30, 220, 100]) * 0.35).astype(np.uint8)
                cv2.drawContours(overlay, contours, -1, (30, 255, 120), 1)
                cx, cy = map(round, center)
                cv2.drawMarker(overlay, (cx, cy), (255, 80, 50), cv2.MARKER_CROSS, 12, 1)
                x1, y1, x2, y2, confidence, _ = box.tolist()
                cv2.rectangle(overlay, (round(x1), round(y1)), (round(x2), round(y2)), (30, 255, 120), 1)
                cv2.putText(overlay, f'bowl {confidence:.2f} ({cx},{cy})',
                            (max(0, round(x1)), max(13, round(y1)-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 255, 120), 1)
                detections.append({'class': 'target_bowl', 'confidence': confidence,
                                   'bbox_xyxy_px': [x1, y1, x2, y2],
                                   'mask_centroid_px': center, 'mask_area_px': int(moments['m00']),
                                   'contours_px': [c.reshape(-1, 2).tolist() for c in contours]})
        if not detections:
            cv2.putText(overlay, 'No target bowl', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 190, 60), 1)
        return overlay, detections, (time.monotonic() - start) * 1000
