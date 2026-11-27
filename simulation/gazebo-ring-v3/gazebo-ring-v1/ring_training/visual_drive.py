"""Calibrated RGB-D reference controller for the boundary-line course."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


CONTROLLER_VERSION = "camera-geometry-serial-v5"
GUIDED_POLICY_WEIGHT = 0.08
GUIDED_POLICY_MAX_AGE_S = 1.0


@dataclass
class ConeTrack:
    distance: float
    lateral: float
    last_seen: float
    observations: int = 1
    bypass_side: int | None = None


class VisionGuide:
    """Estimate the local road and obstacles from the simulated camera alone."""

    def __init__(
        self, *, width: int, height: int, hfov: float, camera_x: float, camera_y: float,
        camera_z: float, pitch: float, yellow_half_width: float, body_half_width: float,
        body_front: float, cone_half_width: float, bypass_offset: float,
    ):
        focal = width / (2 * math.tan(hfov / 2))
        image_y, image_x = np.indices((height, width), dtype=np.float32)
        image_x = (image_x - (width - 1) / 2) / focal
        image_y = (image_y - (height - 1) / 2) / focal
        self.ray_x = math.cos(pitch) - image_y * math.sin(pitch)
        self.ray_y = -image_x
        self.ray_z = -math.sin(pitch) - image_y * math.cos(pitch)
        self.origin = camera_x, camera_y, camera_z
        self.half_width = yellow_half_width
        self.bypass_offset = bypass_offset
        self.body_half_width = body_half_width
        self.body_front = body_front
        self.cone_half_width = cone_half_width
        self.road = np.array([0.0, 0.0, 0.0])
        self.seen_lane: float | None = None
        self.time: float | None = None
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.side = 0
        self.offset = 0.0
        self.cones = []
        self.obstacle_speed_limit = math.inf
        self.speed_state = "cruise"
        self.exit_remaining: float | None = None
        self.brick_last_seen = -math.inf
        self.stats: dict[str, int] = {
            "frames": 0, "lane_frames": 0, "cone_frames": 0, "blue_trigger_frames": 0,
            "magenta_trigger_frames": 0, "left_avoidance_frames": 0, "right_avoidance_frames": 0,
            "blocked_observation_frames": 0,
            "brick_slow_frames": 0,
        }

    def road_y(self, x):
        return np.polyval(self.road, x)

    def update_lane(self, x, y) -> bool:
        # Use consistent longitudinal slices so a close boundary contributes once
        # per range bin, even when it occupies most of the camera pixels.
        longitudinal = []
        lateral = []
        weights = []
        for start in np.arange(1.20, 8.4, 0.58):
            strip = (x >= start) & (x < start + 0.58)
            if np.count_nonzero(strip) < 3:
                continue
            values = np.sort(y[strip])
            groups = np.split(values, np.flatnonzero(np.diff(values) > 0.36 * self.half_width) + 1)
            peaks = [float(np.median(part)) for part in groups if part.size >= 3]
            if not peaks:
                continue
            distance = float(np.median(x[strip]))
            prediction = float(self.road_y(distance))
            slope = 2 * self.road[0] * distance + self.road[1]
            half = self.half_width * math.sqrt(1 + min(abs(slope), 0.70) ** 2)
            paired = [(p + q) / 2 for p, q in zip(peaks, peaks[1:])
                      if 1.48 * half <= q - p <= 2.52 * half]
            if paired:
                center = min(paired, key=lambda value: abs(prediction - value))
                reliability = 2.0
            else:
                options = [peak + sign * half for peak in peaks for sign in (-1, 1)]
                center = min(options, key=lambda value: abs(prediction - value))
                reliability = 1.0
            if self.seen_lane is not None and abs(center - prediction) > 0.52:
                continue
            longitudinal.append(distance)
            lateral.append(center)
            weights.append(reliability / (1 + 0.16 * distance))
        if len(longitudinal) < 3:
            return False

        x_fit = np.asarray(longitudinal)
        y_fit = np.asarray(lateral)
        weight = np.asarray(weights)
        if x_fit[-1] - x_fit[0] < 1.4:
            return False
        # Penalize implausible curvature on short or partially obscured rows.
        design = np.column_stack((x_fit * x_fit, x_fit, np.ones(len(x_fit))))
        fit = np.linalg.lstsq(
            np.vstack((design * weight[:, None], [4.5, 0.0, 0.0])),
            np.append(y_fit * weight, 4.5 * self.road[0]), rcond=None,
        )[0]
        if self.seen_lane is not None:
            fit = 0.84 * fit + 0.16 * self.road
        self.road = fit
        return True

    @staticmethod
    def cone_observations(mask, x, y) -> np.ndarray:
        count, components, statistics, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8,
        )
        observations = []
        for item in range(1, count):
            left, top, width, height, area = statistics[item]
            if area < 3:
                continue
            window = np.s_[top:top + height, left:left + width]
            pixels = components[window] == item
            observations.append((
                float(np.median(x[window][pixels])),
                float(np.median(y[window][pixels])),
            ))
        return np.asarray(observations, dtype=float).reshape(-1, 2)

    def update_avoidance(self, cones, now: float, elapsed: float, lookahead: float) -> None:
        # Observations close in depth are normally the coloured strips of one cone.
        travel = self.speed * elapsed
        for track in self.cones:
            track.distance -= travel
        observed = []
        for distance, y in sorted(cones.tolist()):
            lateral = float(y - self.road_y(distance))
            if distance > 9.2 or abs(lateral) > self.half_width + self.cone_half_width:
                continue
            if observed and distance - observed[-1][0] < 0.48:
                continue
            observed.append((distance, lateral))
        for distance, lateral in observed:
            matches = [track for track in self.cones if abs(track.distance - distance) < 0.77
                       and track.last_seen != now]
            if matches:
                closest = min(matches, key=lambda track: abs(track.distance - distance))
                closest.distance = 0.55 * distance + 0.45 * closest.distance
                closest.lateral = 0.32 * lateral + 0.68 * closest.lateral
                closest.observations += 1
                closest.last_seen = now
            else:
                self.cones.append(ConeTrack(distance, lateral, now))
        self.cones = [track for track in self.cones if track.distance > -4.4 and
                      (track.bypass_side is not None or now - track.last_seen <= 0.42)]
        self.cones.sort(key=lambda track: track.distance)
        self.side = 0
        self.obstacle_speed_limit = math.inf
        self.offset = 0.0
        active = [track for track in self.cones if track.observations >= 2]
        if not active or active[0].distance > 8.2:
            return
        knots = [(active[0].distance - 5.8, 0.0)]
        required_separation = self.body_half_width + self.cone_half_width + 0.025
        for track in active:
            distance, lateral = track.distance, track.lateral
            if track.bypass_side is None:
                if abs(lateral) + self.bypass_offset >= required_separation:
                    track.bypass_side = -1 if lateral > 0 else 1
                elif (abs(lateral) < 0.20 and distance < 5.7) or distance < 3.0:
                    track.bypass_side = 0
            if not track.bypass_side:
                if distance >= -0.32:
                    clear_space = max(0.0, distance - self.body_front - self.cone_half_width - 0.59)
                    self.obstacle_speed_limit = min(2.12, math.sqrt(2.4 * clear_space))
                    if track.bypass_side == 0:
                        self.stats["blocked_observation_frames"] += 1
                    break
                continue
            entry, exit = distance - 1.22, distance + 0.34
            if entry <= knots[-1][0]:
                if distance > 0 and knots[-1][1] * track.bypass_side < 0:
                    self.obstacle_speed_limit = min(self.obstacle_speed_limit, 1.05)
                continue
            offset = track.bypass_side * self.bypass_offset
            knots.extend(((entry, offset), (exit, offset)))
            if distance > -2.5:
                self.obstacle_speed_limit = min(self.obstacle_speed_limit, 2.12)
        if len(knots) == 1:
            return
        knots.append((knots[-1][0] + 3.5, 0.0))
        for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
            if x0 <= lookahead <= x1:
                t = (lookahead - x0) / (x1 - x0)
                self.offset = y0 + (y1 - y0) * t * t * (3 - 2 * t)
                break
        if abs(self.offset) > 0.06:
            self.side = 1 if self.offset > 0 else -1
            self.stats["left_avoidance_frames" if self.side > 0 else "right_avoidance_frames"] += 1

    @staticmethod
    def marker_distance(mask, x) -> float | None:
        return float(np.percentile(x[mask], 25)) if np.count_nonzero(mask) >= 25 else None

    def command(self, stamp: int, rgb: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
        now = stamp * 1e-9
        elapsed = min(0.35, max(0.0, now - self.time)) if self.time is not None else 0.10
        self.time = now
        self.stats["frames"] += 1
        valid = np.isfinite(depth) & (depth > 0) & (depth <= 24)
        distance = np.where(valid, depth, 0.0)
        x = self.origin[0] + self.ray_x * distance
        y = self.origin[1] + self.ray_y * distance
        z = self.origin[2] + self.ray_z * distance
        hue, saturation, value = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).transpose(2, 0, 1)
        strong = valid & (x > 1.05) & (saturation >= 100) & (value >= 78)
        yellow = strong & (hue >= 22) & (hue <= 36) & (abs(z) < 0.30) & (abs(y) < 8)
        if self.update_lane(x[yellow], y[yellow]):
            self.seen_lane = now
            self.stats["lane_frames"] += 1

        road_surface = (valid & (x >= 1.3) & (x <= 6.1) &
                        (abs(y - self.road_y(x)) < self.half_width) & (abs(z) < 0.15))
        red_brown = (hue <= 22) & (saturation >= 38) & (value >= 35)
        ground_pixels = np.count_nonzero(road_surface)
        if ground_pixels >= 100 and np.count_nonzero(road_surface & red_brown) > 0.16 * ground_pixels:
            self.brick_last_seen = now

        orange = strong & (hue >= 8) & (hue <= 19) & (z > 0.07) & (z < 0.59) & (abs(y) < 7)
        cones = self.cone_observations(orange, x, y)
        if len(cones):
            self.stats["cone_frames"] += 1
        lookahead = max(1.18, min(4.6, 0.74 + 0.39 * self.speed))
        self.update_avoidance(cones, now, elapsed, lookahead)

        # The marker colors are evaluated in the local drivable corridor.
        marks = strong & (abs(y - self.road_y(x)) < 4.0) & (z >= -0.10) & (z < 1.65)
        blue = self.marker_distance(marks & (hue >= 114) & (hue <= 128), x)
        if self.speed_state == "cruise" and blue is not None and blue < 8.8 and abs(self.road[0]) < 0.035:
            self.speed_state = "fast"
            self.stats["blue_trigger_frames"] += 1
        if self.speed_state == "fast":
            if self.exit_remaining is not None:
                self.exit_remaining -= elapsed * self.speed
            magenta = self.marker_distance(marks & (hue >= 143) & (hue <= 158), x)
            if magenta is not None and magenta < 6.0:
                self.exit_remaining = magenta
            if self.exit_remaining is not None and self.exit_remaining < 0.70:
                self.speed_state = "done"
                self.stats["magenta_trigger_frames"] += 1

        target_speed = 9.85 if self.speed_state == "fast" else 3.35
        target_speed = min(target_speed, self.obstacle_speed_limit)
        if now - self.brick_last_seen < 1.25:
            target_speed = min(target_speed, 1.5)
            self.stats["brick_slow_frames"] += 1
        missing_lane = self.seen_lane is None or now - self.seen_lane > 0.45
        if missing_lane:
            target_speed = 0.0
        change = target_speed - self.speed
        self.speed += max(-3.6 * elapsed, min(2.3 * elapsed, change))
        lateral_target = float(self.road_y(lookahead)) + self.offset
        curvature = 2 * lateral_target / (lookahead**2 + lateral_target**2)
        requested_yaw = max(-1.25, min(1.25, max(0.65, self.speed) * curvature))
        self.yaw_rate = 0.70 * requested_yaw + 0.30 * self.yaw_rate
        return target_speed, self.yaw_rate


class GuidedPolicy:
    """Keep geometric control responsive while ONNX inference runs separately."""

    def __init__(self, guide: VisionGuide, learned_worker):
        self.guide = guide
        self.learned_worker = learned_worker
        self.learned_action = None
        self.stats = {
            "learned_inferences_received": 0,
            "learned_residual_applied_frames": 0,
            "lane_loss_stop_frames": 0,
            "maximum_applied_learned_age_sim_s": 0.0,
            "maximum_speed_residual_mps": 0.0,
            "maximum_yaw_residual_rad_s": 0.0,
        }

    def command(self, stamp: int, rgb: np.ndarray, depth: np.ndarray) -> tuple[float, float]:
        speed, yaw = self.guide.command(stamp, rgb, depth)
        inferred = self.learned_worker.take()
        if inferred is not None:
            self.learned_action = inferred
            self.stats["learned_inferences_received"] += 1
        self.learned_worker.submit((stamp, rgb, depth))

        now = stamp * 1e-9
        if self.guide.seen_lane is None or now - self.guide.seen_lane > 0.45:
            self.stats["lane_loss_stop_frames"] += 1
            return 0.0, 0.0
        if self.learned_action is None or speed < 1.0:
            return speed, yaw
        learned_stamp, learned_speed, learned_yaw, _ = self.learned_action
        age = (stamp - learned_stamp) * 1e-9
        if age > GUIDED_POLICY_MAX_AGE_S:
            return speed, yaw

        # The geometric channel retains speed authority throughout the marked
        # acceleration and timing region and while the cart passes obstacles.
        fast = self.guide.speed_state == "fast"
        bypass = bool(self.guide.side) or abs(self.guide.offset) > 0.10
        on_brick = now - self.guide.brick_last_seen < 1.25
        speed_limit = 0.0 if fast or bypass or on_brick or self.guide.obstacle_speed_limit < math.inf else 0.12
        yaw_limit = 0.010 if fast else (0.008 if bypass else 0.014)
        speed_residual = max(-speed_limit, min(speed_limit, GUIDED_POLICY_WEIGHT * (learned_speed - speed)))
        yaw_residual = max(-yaw_limit, min(yaw_limit, GUIDED_POLICY_WEIGHT * (learned_yaw - yaw)))
        if speed_residual != 0 or yaw_residual != 0:
            self.stats["learned_residual_applied_frames"] += 1
            self.stats["maximum_applied_learned_age_sim_s"] = max(
                self.stats["maximum_applied_learned_age_sim_s"], age,
            )
            self.stats["maximum_speed_residual_mps"] = max(
                self.stats["maximum_speed_residual_mps"], abs(speed_residual),
            )
            self.stats["maximum_yaw_residual_rad_s"] = max(
                self.stats["maximum_yaw_residual_rad_s"], abs(yaw_residual),
            )
        return speed + speed_residual, yaw + yaw_residual
