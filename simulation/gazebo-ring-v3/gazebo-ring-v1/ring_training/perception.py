"""RGB-D features for the ring policy and its offline data extractor."""

import hashlib
from pathlib import Path

import numpy as np


MODELS = Path(__file__).resolve().parents[1] / "models"
ROAD_MODEL = MODELS / "road-4060-v1.onnx"
CONE_MODEL = MODELS / "traffic-cone.onnx"
FEATURE_VERSION = "ring-rgbd-v3"
ROAD_CLASSES = (
    "paved_ground", "unpaved_ground", "vegetation", "water", "rigid_obstacle",
    "person_or_vehicle", "sky",
)
ROW_EDGES = (0.16, 0.36, 0.54, 0.72, 1.0)
CORRIDOR_ROWS = (0.46, 0.49, 0.52, 0.55, 0.59, 0.64, 0.72, 0.80, 0.85)
CORRIDOR_FIELDS = ("left_line_x", "left_line_visible", "right_line_x", "right_line_visible",
                   "center_x", "width", "pair_visible")
COLUMNS = 7
DEPTH_RANGE_M = 24.0
CELL_FIELDS = (*ROAD_CLASSES, "depth_p20", "depth_median", "depth_valid", "orange",
               "yellow", "white", "blue", "cyan", "magenta", "brick", "cone_score", "cone_depth")
FEATURE_NAMES = tuple(
    f"row{row}_col{col}_{field}"
    for row in range(len(ROW_EDGES) - 1)
    for col in range(COLUMNS)
    for field in CELL_FIELDS
) + tuple(
    f"row{row}_{field}"
    for row in range(len(ROW_EDGES) - 1)
    for field in ("yellow_center_x", "yellow_fraction")
) + tuple(
    f"scan{row}_{field}" for row in range(len(CORRIDOR_ROWS)) for field in CORRIDOR_FIELDS
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _corridor_features(yellow, value, paved):
    """Locate the visible left and right yellow boundaries on camera scanlines."""
    height, width = yellow.shape
    result = np.zeros((len(CORRIDOR_ROWS), len(CORRIDOR_FIELDS)), dtype=np.float32)
    max_gap = max(1, round(width * 0.008))
    sample_width = max(2, round(width * 0.045))
    min_separation = max(2, round(width * 0.06))
    for index, fraction in enumerate(CORRIDOR_ROWS):
        row = min(round(height * fraction), height - 1)
        runs = np.flatnonzero(yellow[row])
        if not runs.size:
            continue
        paint = np.split(runs, np.flatnonzero(np.diff(runs) > max_gap) + 1)
        left = float(paint[0].mean())
        right = float(paint[-1].mean())
        if right - left >= min_separation:
            result[index] = (2 * (left + 0.5) / width - 1, 1,
                             2 * (right + 0.5) / width - 1, 1,
                             (left + right + 1) / width - 1, (right - left) / width, 1)
            continue

        # A single line is assigned using the surface on each side of its painted pixels.
        run = max(paint, key=len)
        left_end, right_start = max(0, run[0] - 1), min(width, run[-1] + 2)
        before = np.s_[max(0, row - 1):row + 2, max(0, left_end - sample_width):left_end]
        after = np.s_[max(0, row - 1):row + 2, right_start:right_start + sample_width]
        x = 2 * (float(run.mean()) + 0.5) / width - 1
        if value[before].size < 3 or value[after].size < 3:
            paved_to_right = x < 0
        else:
            light_difference = float(np.median(value[before])) - float(np.median(value[after]))
            paved_difference = float(paved[after].mean() - paved[before].mean())
            if abs(light_difference) >= 25:
                paved_to_right = light_difference > 0
            elif abs(paved_difference) >= 0.25:
                paved_to_right = paved_difference > 0
            elif abs(x) >= 0.35:
                paved_to_right = x < 0
            else:
                continue
        offset = 0 if paved_to_right else 2
        result[index, offset:offset + 2] = x, 1
    return result.ravel()


class Perception:
    def __init__(self, road_model=ROAD_MODEL, cone_model=CONE_MODEL, *, threads=2, depth_max_m=DEPTH_RANGE_M):
        import cv2
        import onnxruntime as ort
        from PIL import Image

        if threads < 1:
            raise ValueError("threads must be positive")
        if not 0 < depth_max_m <= DEPTH_RANGE_M:
            raise ValueError(f"depth_max_m must be in (0, {DEPTH_RANGE_M}] meters")
        self.cv2 = cv2
        self.image = Image
        self.depth_max_m = float(depth_max_m)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        providers = ["CPUExecutionProvider"]
        self.model_hashes = (_digest(road_model), _digest(cone_model))
        self.road = ort.InferenceSession(str(road_model), options, providers=providers)
        self.cone = ort.InferenceSession(str(cone_model), options, providers=providers)
        self.road_input, self.road_size = self._input(self.road)
        self.cone_input, self.cone_size = self._input(self.cone)

    @staticmethod
    def _input(session):
        model_input, = session.get_inputs()
        batch, channels, height, width = model_input.shape
        if (batch != 1 or channels != 3 or model_input.type != "tensor(float)"
                or not all(isinstance(size, int) and size > 0 for size in (height, width))):
            raise ValueError("Expected a fixed-size NCHW FP32 RGB ONNX input")
        return model_input.name, (height, width)

    @staticmethod
    def _letterbox_geometry(rgb, target_size):
        height, width = rgb.shape[:2]
        target_h, target_w = target_size
        ratio = min(target_w / width, target_h / height)
        resized_w = max(1, round(width * ratio))
        resized_h = max(1, round(height * ratio))
        return (target_w - resized_w) // 2, (target_h - resized_h) // 2, resized_w, resized_h

    def road_mask(self, rgb):
        """Restore model logits to camera pixel coordinates before labeling."""
        x, y, width, height = self._letterbox_geometry(rgb, self.road_size)
        pad_color = tuple((IMAGENET_MEAN * 255).astype(int))
        canvas = self.image.new("RGB", self.road_size[::-1], pad_color)
        resized = self.image.fromarray(rgb).resize((width, height), self.image.Resampling.BILINEAR)
        canvas.paste(resized, (x, y))
        normalized = (np.asarray(canvas, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        logits, = self.road.run(None, {self.road_input: tensor})
        if logits.ndim != 4 or logits.shape[:2] != (1, len(ROAD_CLASSES)) or not np.isfinite(logits).all():
            raise ValueError("Invalid road ONNX output, expected seven finite logit maps")
        padded = self.cv2.resize(logits[0].transpose(1, 2, 0), self.road_size[::-1])
        crop = padded[y:y + height, x:x + width]
        original = self.cv2.resize(crop, rgb.shape[1::-1])
        return original.argmax(2).astype(np.uint8)

    def cone_boxes(self, rgb):
        """Return YOLO cone detections as x1, y1, x2, y2, confidence in camera pixels."""
        x, y, width, height = self._letterbox_geometry(rgb, self.cone_size)
        padded = np.full((*self.cone_size, 3), 114, dtype=np.uint8)
        padded[y:y + height, x:x + width] = self.cv2.resize(rgb, (width, height))
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        raw, = self.cone.run(None, {self.cone_input: tensor})
        if raw.ndim != 3 or raw.shape[:2] != (1, 5) or not np.isfinite(raw).all():
            raise ValueError("Invalid cone ONNX output, expected raw single-class YOLO boxes")
        confidences = raw[0, 4]
        keep = confidences > 0.25
        candidates = raw[0, :4].T[keep]
        if not len(candidates):
            return np.empty((0, 5), dtype=np.float32)
        scores = confidences[keep]
        candidates[:, :2] -= candidates[:, 2:] / 2
        indices = self.cv2.dnn.NMSBoxes(candidates.tolist(), scores.tolist(), 0.25, 0.7)
        selected = np.asarray(indices, dtype=int).reshape(-1)[:300]
        boxes = candidates[selected]
        boxes[:, 2:] += boxes[:, :2]
        source_h, source_w = rgb.shape[:2]
        boxes[:, 0::2] = np.clip((boxes[:, 0::2] - x) * source_w / width, 0, source_w)
        boxes[:, 1::2] = np.clip((boxes[:, 1::2] - y) * source_h / height, 0, source_h)
        visible = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        return np.column_stack((boxes[visible], scores[selected][visible]))

    def features(self, rgb, depth=None):
        """Encode an RGB8 frame and optional registered optical-Z depth in meters."""
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 16:
            raise ValueError("Expected an HxWx3 RGB8 image with each side at least 16 pixels")
        if depth is not None and (depth.shape != rgb.shape[:2] or depth.dtype.kind != "f"):
            raise ValueError("Expected registered HxW floating-point depth in meters")
        labels = self.road_mask(rgb)
        boxes = self.cone_boxes(rgb)
        hue, saturation, value = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2HSV).transpose(2, 0, 1)
        orange = (hue <= 19) & (saturation >= 130) & (value >= 135)
        yellow = (hue >= 20) & (hue <= 39) & (saturation >= 95) & (value >= 130)
        white = (saturation <= 50) & (value >= 165)
        saturated = (saturation >= 115) & (value >= 120)
        blue = (hue >= 108) & (hue <= 136) & saturated
        cyan = (hue >= 82) & (hue < 108) & saturated
        magenta = (hue >= 137) & (hue <= 170) & saturated
        brick = ((hue <= 18) | (hue >= 172)) & (saturation >= 33) & (saturation < 135) & (value >= 45)
        colors = (orange, yellow, white, blue, cyan, magenta, brick)

        height, width = rgb.shape[:2]
        y_edges = np.rint(np.asarray(ROW_EDGES) * height).astype(int)
        x_edges = np.rint(np.linspace(0, width, COLUMNS + 1)).astype(int)
        shape = (len(y_edges) - 1, len(x_edges) - 1)
        cone_grid = np.zeros((*shape, 2), dtype=np.float32)
        if depth is None:
            depth = np.zeros((height, width), dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0) & (depth <= self.depth_max_m)
        for x1, y1, x2, y2, confidence in boxes:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            row = int(np.searchsorted(y_edges, center_y, side="right")) - 1
            col = int(np.searchsorted(x_edges, center_x, side="right")) - 1
            if 0 <= row < shape[0] and 0 <= col < shape[1] and confidence > cone_grid[row, col, 0]:
                margin = (x2 - x1) * 0.2
                patch = np.s_[int(center_y):int(np.ceil(y2)), int(x1 + margin):int(np.ceil(x2 - margin))]
                ranges = depth[patch][valid[patch]]
                distance = np.percentile(ranges, 25) / DEPTH_RANGE_M if len(ranges) else 0.0
                cone_grid[row, col] = confidence, distance

        features = []
        lane_features = []
        for row, (top, bottom) in enumerate(zip(y_edges[:-1], y_edges[1:])):
            row_yellow = yellow[top:bottom]
            _, positions = np.nonzero(row_yellow)
            lane_features.extend((2 * (positions.mean() + 0.5) / width - 1 if len(positions) else 0.0,
                                  row_yellow.mean()))
            for col, (left, right) in enumerate(zip(x_edges[:-1], x_edges[1:])):
                cell = np.s_[top:bottom, left:right]
                counts = np.bincount(labels[cell].ravel(), minlength=len(ROAD_CLASSES))
                fractions = counts / counts.sum()
                distance_samples = depth[cell][valid[cell]]
                ranges = (np.percentile(distance_samples, (20, 50)) / DEPTH_RANGE_M
                          if len(distance_samples) else (0.0, 0.0))
                color_occupancy = (mask[cell].mean() for mask in colors)
                features.extend((*fractions, *ranges, valid[cell].mean(), *color_occupancy, *cone_grid[row, col]))
        corridor = _corridor_features(yellow, value, labels == 0)
        return np.concatenate((np.asarray([*features, *lane_features], dtype=np.float32), corridor))
