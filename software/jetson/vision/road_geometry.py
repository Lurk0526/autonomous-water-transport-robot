"""Camera-frame RGB/depth association; no robot control or assumed mount pose."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Calibration:
    width: int
    height: int
    k: np.ndarray
    d: np.ndarray
    frame: str
    model: str = 'plumb_bob'

    def validate(self):
        self.k = np.asarray(self.k, dtype=np.float64).reshape(3, 3)
        self.d = np.asarray(self.d, dtype=np.float64).reshape(-1)
        if self.width <= 0 or self.height <= 0 or not self.frame:
            raise ValueError('Missing camera size/frame')
        if not np.isfinite(self.k).all() or self.k[0, 0] <= 0 or self.k[1, 1] <= 0:
            raise ValueError('Uncalibrated camera intrinsics')
        if not np.isfinite(self.d).all() or self.model not in ('plumb_bob', 'rational_polynomial'):
            raise ValueError('Unsupported camera distortion model')
        if self.d.size not in (0, 4, 5, 8, 12, 14):
            raise ValueError('Unsupported distortion coefficient count')
        return self


def rotation_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError('Invalid transform quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def valid_transform(transform):
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError('Invalid rigid transform')
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError('Transform rotation is not a proper rotation')
    if not np.allclose(t[3], [0, 0, 0, 1]):
        raise ValueError('Invalid transform final row')
    return t


def image_array(message):
    """Decode supported ROS image encodings, including row padding and endianness."""
    types = {'rgb8': (np.uint8, 3), 'bgr8': (np.uint8, 3), 'mono8': (np.uint8, 1),
             '16UC1': (np.uint16, 1), '32FC1': (np.float32, 1)}
    if message.encoding not in types:
        raise ValueError(f'Unsupported encoding {message.encoding}')
    kind, channels = types[message.encoding]
    dtype = np.dtype(kind).newbyteorder('>' if message.is_bigendian else '<')
    h, w, step = message.height, message.width, message.step
    if not (0 < h <= 4096 and 0 < w <= 4096) or step < w * channels * dtype.itemsize:
        raise ValueError('Invalid image dimensions or row stride')
    if len(message.data) < h * step or h * step > 64 * 1024 * 1024:
        raise ValueError('Invalid image buffer size')
    shape = (h, w, channels) if channels > 1 else (h, w)
    strides = (step, channels * dtype.itemsize, dtype.itemsize) if channels > 1 else (step, dtype.itemsize)
    value = np.ndarray(shape, dtype=dtype, buffer=bytes(message.data), strides=strides).astype(kind, copy=True)
    if message.encoding == 'bgr8':
        value = value[:, :, ::-1].copy()
    return value


def depth_metres(message):
    value = image_array(message)
    if message.encoding == '16UC1':
        return value.astype(np.float32) * 0.001
    if message.encoding == '32FC1':
        return value
    raise ValueError('Depth must use 16UC1 millimetres or 32FC1 metres')


def is_fresh(now, stamp, limit, future_tolerance=0.05):
    return np.isfinite(now) and np.isfinite(stamp) and stamp > 0 and -future_tolerance <= now - stamp <= limit


def nearest_stamp(items, stamp, tolerance):
    if not items:
        return None
    item = min(items, key=lambda x: abs(x[0] - stamp))
    return item if abs(item[0] - stamp) <= tolerance else None


def associate_depth(depth, depth_camera, color_camera, depth_to_color, labels, confidence,
                    min_m=0.3, max_m=5.0, confidence_min=0.65, occlusion_m=0.02):
    """Backproject rectified depth, transform, distort/project into raw color, z-buffer.

    Returned XYZ points remain in the original depth optical frame. Unknown or
    occluded semantic associations use 255. Invalid depth stays NaN, never free.
    """
    dc, cc = depth_camera.validate(), color_camera.validate()
    t = valid_transform(depth_to_color)
    if np.linalg.norm(t[:3, 3]) > 2:
        raise ValueError('Camera-internal baseline exceeds 2 m; check transform units')
    if depth.shape != (dc.height, dc.width) or labels.shape != (cc.height, cc.width) or confidence.shape != labels.shape:
        raise ValueError('Camera calibration/image dimensions differ')
    if not 0 < min_m < max_m or not 0 <= confidence_min <= 1:
        raise ValueError('Invalid depth range or confidence threshold')
    if np.any(np.abs(dc.d) > 1e-12):
        raise ValueError('Depth input must be rectified')
    v, u = np.nonzero(np.isfinite(depth) & (depth >= min_m) & (depth <= max_m))
    z = depth[v, u].astype(np.float64)
    xyz = np.column_stack(((u-dc.k[0, 2])*z/dc.k[0, 0], (v-dc.k[1, 2])*z/dc.k[1, 1], z))
    mapped = xyz @ t[:3, :3].T + t[:3, 3]
    in_front = mapped[:, 2] > 1e-6
    uv = np.full((len(xyz), 2), -1, dtype=np.int32)
    if in_front.any():
        front = mapped[in_front]
        if cc.d.size <= 8:
            # Brown/rational projection without allocating OpenCV's unused Nx15 Jacobian.
            x, y = front[:, 0] / front[:, 2], front[:, 1] / front[:, 2]
            coeff = np.zeros(8)
            coeff[:cc.d.size] = cc.d
            k1, k2, p1, p2, k3, k4, k5, k6 = coeff
            r2 = x*x + y*y
            radial = (1 + r2*(k1 + r2*(k2 + r2*k3))) / (1 + r2*(k4 + r2*(k5 + r2*k6)))
            xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
            yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
            projected = np.column_stack((cc.k[0, 0]*xd + cc.k[0, 2], cc.k[1, 1]*yd + cc.k[1, 2]))
        else:
            pixels, _ = cv2.projectPoints(front, np.zeros(3), np.zeros(3), cc.k, cc.d)
            projected = pixels.reshape(-1, 2)
        finite = np.isfinite(projected).all(axis=1) & (np.abs(projected).max(axis=1) < 1e7)
        indexes = np.flatnonzero(in_front)[finite]
        uv[indexes] = np.rint(projected[finite]).astype(np.int32)
    inside = in_front & (uv[:, 0] >= 0) & (uv[:, 0] < cc.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cc.height)
    selected = np.flatnonzero(inside)
    flat = np.full(cc.width * cc.height, np.inf, dtype=np.float32)
    raster = uv[selected, 1] * cc.width + uv[selected, 0]
    np.minimum.at(flat, raster, mapped[selected, 2])
    visible = np.zeros(len(xyz), dtype=bool)
    visible[selected] = mapped[selected, 2] <= flat[raster] + occlusion_m
    semantic = np.full(len(xyz), 255, dtype=np.uint8)
    certainty = np.zeros(len(xyz), dtype=np.float32)
    selected = np.flatnonzero(visible)
    if len(selected):
        certainty[selected] = confidence[uv[selected, 1], uv[selected, 0]]
        known = selected[(certainty[selected] >= confidence_min) & np.isfinite(certainty[selected])]
        source_labels = labels[uv[known, 1], uv[known, 0]]
        semantic[known] = np.where(source_labels < 7, source_labels, 255)
    flat[~np.isfinite(flat)] = np.nan
    aligned = flat.reshape(cc.height, cc.width)
    candidate = ((labels == 0) & (confidence >= confidence_min) & np.isfinite(aligned)).astype(np.uint8) * 255
    return {'xyz': xyz.astype(np.float32), 'labels': semantic, 'confidence': certainty,
            'aligned_depth': aligned, 'candidate': candidate, 'visible': visible, 'uv': uv}


def height_obstacles(points, camera_to_base, ground_z_m, min_height=0.10, max_height=1.5, ground_tolerance=0.06):
    """Prepared height filter; requires measured TF and ground height at runtime."""
    if ground_z_m is None or not np.isfinite(ground_z_m):
        raise ValueError('Measured ground height is required')
    if not 0 <= ground_tolerance < min_height < max_height:
        raise ValueError('Invalid height bands')
    t = valid_transform(camera_to_base)
    base = np.asarray(points) @ t[:3, :3].T + t[:3, 3]
    height = base[:, 2] - ground_z_m
    finite = np.isfinite(base).all(axis=1)
    return base.astype(np.float32), finite & (height >= min_height) & (height <= max_height), finite & (np.abs(height) <= ground_tolerance)
