"""Run a Gazebo Harmonic episode, score ground truth, or collect expert RGB-D samples."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid

import cv2
import numpy as np

from . import world


TRACK_RADIUS = world.TURN_RADIUS
STRAIGHT_HALF_LENGTH = world.HALF_STRAIGHT
HALF_ARC_LENGTH = world.HALF_TURN_LENGTH
MIN_SPEED_MPS = world.SPEED_LIMIT_MPS
DEFAULT_OFFSET = world.LANE_OFFSET
GROUND_CONTACT_MAX_AGE_NS = 12_000_000
POSE_STRUCT = struct.Struct("<Q28d")
GROUND_STRUCT = struct.Struct("<QB")
IMAGE_STRUCT = struct.Struct("<QIIII")
PACKET_STRUCT = struct.Struct("<cI")
TRACE_FIELDS = (
    "sim_time", "x", "y", "z", "yaw_rad", "station_m", "offset_m",
    "speed_mps", "adjacent_pose_speed_mps", "yaw_rate_rad_s", "segment",
    "target_speed_mps", "target_yaw_rate_rad_s", "grounded_wheels",
    "inner_yellow_touch", "outer_yellow_touch",
)


def angle_difference(value: float, reference: float) -> float:
    return math.atan2(math.sin(value - reference), math.cos(value - reference))


def station_difference(value: float, reference: float) -> float:
    half = world.TRACK_LENGTH / 2
    return (value - reference + half) % world.TRACK_LENGTH - half


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def project(x: float, y: float) -> tuple[float, float]:
    """Project world coordinates onto the stadium centerline."""
    straight_length = world.STRAIGHT_LENGTH
    if x > STRAIGHT_HALF_LENGTH:
        dx = x - STRAIGHT_HALF_LENGTH
        angle = math.atan2(y, dx) + math.pi / 2
        return straight_length + angle * TRACK_RADIUS, TRACK_RADIUS - math.hypot(dx, y)
    if x < -STRAIGHT_HALF_LENGTH:
        dx = x + STRAIGHT_HALF_LENGTH
        angle = math.atan2(y, dx) % math.tau - math.pi / 2
        return 2 * straight_length + HALF_ARC_LENGTH + angle * TRACK_RADIUS, TRACK_RADIUS - math.hypot(dx, y)
    if y <= 0:
        return x + STRAIGHT_HALF_LENGTH, y + TRACK_RADIUS
    return straight_length + HALF_ARC_LENGTH + STRAIGHT_HALF_LENGTH - x, TRACK_RADIUS - y


def segment_at(station: float) -> str:
    if world.BRICK_START_S <= station < world.BRICK_END_S:
        return "brick"
    if world.SPEED_START_S <= station < world.SPEED_END_S:
        return "speed"
    if world.BYPASS_START_S <= station <= world.BYPASS_END_S:
        return "cones"
    if 88 <= station <= 88 + HALF_ARC_LENGTH or station >= 176 + HALF_ARC_LENGTH:
        return "curve"
    return "straight"


@dataclass(frozen=True)
class Pose:
    stamp: int
    x: float
    y: float
    z: float
    yaw: float

    @property
    def time(self) -> float:
        return self.stamp * 1e-9


@dataclass(frozen=True)
class Motion:
    pose: Pose
    station: float
    offset: float
    speed: float
    yaw_rate: float
    adjacent_speed: float


def tire_road_span(
    center_x: float, center_y: float, axle_x: float, axle_y: float, axle_z: float,
) -> tuple[float, float, float]:
    """Project both ends of the cylindrical tire's lower support line."""
    rolling_vertical = math.hypot(axle_x, axle_y)
    if rolling_vertical > 1e-8:
        radial_shift = world.WHEEL_RADIUS * axle_z / rolling_vertical
        center_x += radial_shift * axle_x
        center_y += radial_shift * axle_y
    half_width = world.WHEEL_WIDTH / 2
    station, _ = project(center_x, center_y)
    _, side_a = project(center_x - half_width * axle_x, center_y - half_width * axle_y)
    _, side_b = project(center_x + half_width * axle_x, center_y + half_width * axle_y)
    return station, min(side_a, side_b), max(side_a, side_b)


def footprint(pose: Pose) -> list[tuple[float, float]]:
    cos_yaw, sin_yaw = math.cos(pose.yaw), math.sin(pose.yaw)
    return [
        (pose.x + x * cos_yaw - y * sin_yaw, pose.y + x * sin_yaw + y * cos_yaw)
        for x in (-abs(world.BODY_REAR), world.BODY_FRONT)
        for y in (-world.BODY_HALF_WIDTH, world.BODY_HALF_WIDTH)
    ]


def overlaps_cone(pose: Pose, cone_x: float, cone_y: float) -> bool:
    """Test the measured body footprint against the cone's ground footprint."""
    cos_yaw, sin_yaw = math.cos(pose.yaw), math.sin(pose.yaw)
    half_length = (world.BODY_FRONT + abs(world.BODY_REAR)) / 2
    half_width = world.BODY_HALF_WIDTH
    center = world.BODY_FRONT - half_length
    dx = cone_x - (pose.x + center * cos_yaw)
    dy = cone_y - (pose.y + center * sin_yaw)
    cone_half = world.CONE_BASE / 2
    abs_cos, abs_sin = abs(cos_yaw), abs(sin_yaw)
    cone_projection = cone_half * (abs_cos + abs_sin)
    return (
        abs(dx) <= half_length * abs_cos + half_width * abs_sin + cone_half
        and abs(dy) <= half_length * abs_sin + half_width * abs_cos + cone_half
        and abs(dx * cos_yaw + dy * sin_yaw) <= half_length + cone_projection
        and abs(-dx * sin_yaw + dy * cos_yaw) <= half_width + cone_projection
    )


class Scoring:
    def __init__(self, cones: tuple[tuple[float, float], ...]):
        self.cones = cones
        self.history: deque[Motion] = deque(maxlen=180)
        self.gates: dict[str, float] = {}
        self.contacts: dict[str, float] = {}
        self.cone_overlaps: dict[int, float] = {}
        self.first_offtrack: float | None = None
        self.first_wrong_way: float | None = None
        self.speeds: list[float] = []
        self.adjacent_zone_speeds: list[float] = []
        self.max_pose_gap = 0.0
        self.speed_zone_max_pose_gap = 0.0
        self.brick_distance = 0.0
        self.yellow_touches: dict[str, dict] = {}
        self.first_yellow_violation: dict | None = None
        self.grounded_samples = dict.fromkeys(world.WHEEL_LINKS, 0)
        self.reverse_progress = 0.0
        self.lap_distance = 0.0
        self.total_distance = 0.0
        self.stats_position_samples = 0
        self.max_speed = 0.0
        self.first_motion: Motion | None = None

    @property
    def finished(self) -> bool:
        return "finish" in self.gates

    def update(self, pose: Pose) -> Motion:
        station, offset = project(pose.x, pose.y)
        previous = self.history[-1] if self.history else None
        speed = yaw_rate = adjacent_speed = 0.0
        pose_gap = 0.0
        if previous:
            pose_gap = pose.time - previous.pose.time
            adjacent_speed = math.hypot(pose.x - previous.pose.x, pose.y - previous.pose.y) / pose_gap
            self.max_pose_gap = max(self.max_pose_gap, pose_gap)
            baseline = next(
                (motion.pose for motion in reversed(self.history) if pose.time - motion.pose.time >= 0.10),
                self.history[0].pose,
            )
            elapsed = pose.time - baseline.time
            if elapsed > 0:
                speed = math.hypot(pose.x - baseline.x, pose.y - baseline.y) / elapsed
                yaw_rate = angle_difference(pose.yaw, baseline.yaw) / elapsed

        state = Motion(pose, station, offset, speed, yaw_rate, adjacent_speed)
        self.history.append(state)
        self.max_speed = max(self.max_speed, speed)
        self.stats_position_samples += 1
        if not self.first_motion:
            self.first_motion = state
        if previous:
            self.total_distance += math.hypot(pose.x - previous.pose.x, pose.y - previous.pose.y)

        corners = footprint(pose)
        if self.first_offtrack is None and any(abs(project(x, y)[1]) > world.ROAD_HALF_WIDTH for x, y in corners):
            self.first_offtrack = pose.time
        for index, (cone_x, cone_y) in enumerate(self.cones):
            if index not in self.cone_overlaps and overlaps_cone(pose, cone_x, cone_y):
                self.cone_overlaps[index] = pose.time

        if not previous:
            return state

        progress = station_difference(station, previous.station)
        if "start" in self.gates:
            self.reverse_progress = max(0.0, self.reverse_progress - progress)
            if self.reverse_progress >= 0.25 and self.first_wrong_way is None:
                self.first_wrong_way = pose.time

        if self.crosses(previous, state, world.START_S):
            if "start" not in self.gates:
                self.gates["start"] = self.crossing_time(previous, state, world.START_S)
                progress = station_difference(station, world.START_S)
            elif "speed_exit" in self.gates and self.lap_distance >= world.TRACK_LENGTH - 1.0:
                self.gates.setdefault("finish", self.crossing_time(previous, state, world.START_S))
                self.lap_distance += station_difference(world.START_S, previous.station)

        if "start" not in self.gates or self.finished:
            return state

        self.lap_distance += progress
        sections = (
            ("brick_entry", "start", world.BRICK_START_S),
            ("brick_exit", "brick_entry", world.BRICK_END_S),
            ("speed_entry", "brick_exit", world.SPEED_START_S),
            ("speed_exit", "speed_entry", world.SPEED_END_S),
        )
        for gate, required, location in sections:
            if gate not in self.gates and required in self.gates and self.crosses(previous, state, location):
                self.gates[gate] = self.crossing_time(previous, state, location)

        if progress > 0 and "brick_entry" in self.gates:
            self.brick_distance += max(
                0.0, min(station, world.BRICK_END_S) - max(previous.station, world.BRICK_START_S)
            ) if previous.station <= station else 0.0

        if "speed_entry" in self.gates:
            crossed_entry = self.crosses(previous, state, world.SPEED_START_S)
            crossed_exit = self.crosses(previous, state, world.SPEED_END_S)
            inside_zone = world.SPEED_START_S <= station < world.SPEED_END_S
            if crossed_entry or crossed_exit or inside_zone:
                self.adjacent_zone_speeds.append(adjacent_speed)
                self.speed_zone_max_pose_gap = max(self.speed_zone_max_pose_gap, pose_gap)
            if crossed_entry:
                self.speeds.append(self.crossing_speed(previous, state, world.SPEED_START_S))
            if inside_zone:
                self.speeds.append(speed)
            if crossed_exit:
                self.speeds.append(self.crossing_speed(previous, state, world.SPEED_END_S))
        return state

    def score_wheels(self, pose: Pose, wheels: tuple, grounded: tuple[bool, ...]) -> set[str]:
        touches = set()
        inner_band = world.YELLOW_LINE_OFFSET - world.YELLOW_LINE_WIDTH / 2
        outer_band = world.YELLOW_LINE_OFFSET + world.YELLOW_LINE_WIDTH / 2
        for index, is_grounded in enumerate(grounded):
            if not is_grounded:
                continue
            wheel = world.WHEEL_LINKS[index]
            self.grounded_samples[wheel] += 1
            x, y, _z, axle_x, axle_y, axle_z = wheels[index * 6:(index + 1) * 6]
            station, minimum, maximum = tire_road_span(x, y, axle_x, axle_y, axle_z)
            legal_bypass = bool(self.cones) and world.BYPASS_START_S <= station <= world.BYPASS_END_S
            if self.first_offtrack is None and max(abs(minimum), abs(maximum)) > world.ROAD_HALF_WIDTH:
                self.first_offtrack = pose.time
            for side, low, high in (("inner", minimum, maximum), ("outer", -maximum, -minimum)):
                if high < inner_band:
                    continue
                event = {
                    "sim_time_s": pose.time, "station_m": station, "wheel": wheel,
                    "side": side, "in_cone_bypass_zone": legal_bypass,
                }
                if not legal_bypass and self.first_yellow_violation is None:
                    self.first_yellow_violation = event
                if low <= outer_band:
                    touches.add(side)
                    summary = self.yellow_touches.setdefault(side, {
                        "first_touch": event, "wheels": [], "samples": 0,
                        "allowed_samples": 0, "violating_samples": 0,
                    })
                    if wheel not in summary["wheels"]:
                        summary["wheels"].append(wheel)
                    summary["samples"] += 1
                    summary["allowed_samples" if legal_bypass else "violating_samples"] += 1
        return touches

    def crosses(self, before: Motion, after: Motion, gate: float) -> bool:
        advance = station_difference(after.station, before.station)
        distance = (gate - before.station) % world.TRACK_LENGTH
        if not 0 < distance <= advance:
            return False
        fraction = self.crossing_fraction(before, after, gate)
        x = before.pose.x + fraction * (after.pose.x - before.pose.x)
        y = before.pose.y + fraction * (after.pose.y - before.pose.y)
        _, offset = project(x, y)
        return abs(offset) <= world.ROAD_HALF_WIDTH

    @staticmethod
    def crossing_fraction(before: Motion, after: Motion, gate: float) -> float:
        return station_difference(gate, before.station) / station_difference(after.station, before.station)

    def crossing_time(self, before: Motion, after: Motion, gate: float) -> float:
        fraction = self.crossing_fraction(before, after, gate)
        return before.pose.time + fraction * (after.pose.time - before.pose.time)

    def crossing_speed(self, before: Motion, after: Motion, gate: float) -> float:
        fraction = self.crossing_fraction(before, after, gate)
        return before.speed + fraction * (after.speed - before.speed)

    def result(self) -> dict:
        brick = self.brick_distance >= world.BRICK_END_S - world.BRICK_START_S - 0.15 and "brick_exit" in self.gates
        minimum_speed = min(self.speeds, default=None)
        minimum_adjacent_speed = min(self.adjacent_zone_speeds, default=None)
        zone_gate_time = self.gates["speed_exit"] - self.gates["speed_entry"] if "speed_exit" in self.gates else None
        gate_average_speed = (world.SPEED_END_S - world.SPEED_START_S) / zone_gate_time if zone_gate_time else None
        speed_pass = (
            gate_average_speed is not None
            and minimum_speed is not None and minimum_speed >= MIN_SPEED_MPS
            and minimum_adjacent_speed is not None and minimum_adjacent_speed >= MIN_SPEED_MPS
        )
        measured_span = self.history[-1].pose.time - self.first_motion.pose.time if self.first_motion else 0.0
        gate_x, gate_y, gate_yaw = world.centreline(world.START_S)
        failures = []
        if not self.finished:
            failures.append("lap_not_completed")
        if self.contacts or self.cone_overlaps:
            failures.append("collision")
        if self.first_offtrack is not None:
            failures.append("pavement_boundary_crossed")
        if self.first_wrong_way is not None:
            failures.append("wrong_direction")
        if self.first_yellow_violation is not None:
            failures.append("yellow_boundary_violation")
        if not all(self.grounded_samples.values()):
            failures.append("wheel_contact_truth_unavailable")
        if not brick:
            failures.append("brick_segment_not_completed")
        if not speed_pass:
            failures.append("speed_zone_below_30_kmh" if minimum_speed is not None else "speed_zone_not_crossed")
        return {
            "passed": not failures,
            "failures": failures,
            "completed_laps": int(self.finished),
            "crossings_sim_time_s": self.gates,
            "lap_time_s": self.gates["finish"] - self.gates["start"] if self.finished else None,
            "position_samples": self.stats_position_samples,
            "lap_progress_m": self.lap_distance,
            "total_distance_m": self.total_distance,
            "collision_contacts": self.contacts,
            "collision_footprint_overlaps": self.cone_overlaps,
            "first_offtrack_sim_time": self.first_offtrack,
            "first_wrong_way_sim_time": self.first_wrong_way,
            "yellow_line_touches": self.yellow_touches,
            "first_yellow_boundary_violation": self.first_yellow_violation,
            "tire_grounded_pose_samples": self.grounded_samples,
            "tire_ground_contact_max_stamp_difference_s": GROUND_CONTACT_MAX_AGE_NS * 1e-9,
            "start_finish_gate": {
                "station_m": world.START_S, "center_x_m": gate_x, "center_y_m": gate_y,
                "heading_rad": gate_yaw, "width_m": 2 * world.ROAD_HALF_WIDTH,
                "vehicle_reference": "rear axle center",
            },
            "brick_segment_completed": brick,
            "brick_progress_m": self.brick_distance,
            "speed_zone_passed": speed_pass,
            "speed_zone_samples": len(self.speeds),
            "speed_zone_min_kmh": None if minimum_speed is None else minimum_speed * 3.6,
            "speed_zone_max_kmh": max(self.speeds) * 3.6 if self.speeds else None,
            "speed_zone_adjacent_pose_min_kmh": None if minimum_adjacent_speed is None else minimum_adjacent_speed * 3.6,
            "speed_zone_gate_average_kmh": None if gate_average_speed is None else gate_average_speed * 3.6,
            "speed_zone_max_pose_gap_sim_s": self.speed_zone_max_pose_gap or None,
            "max_speed_kmh": self.max_speed * 3.6,
            "world_pose_topic": world.POSE_TOPIC,
            "world_pose_expected_hz": 1 / world.MIN_STEP,
            "world_pose_observed_hz": (self.stats_position_samples - 1) / measured_span if measured_span else None,
            "world_pose_max_gap_sim_s": self.max_pose_gap or None,
            "speed_source": "Planar Gazebo model displacement divided by simulated elapsed time",
            "speed_zone_min_window_sim_s": 0.10,
        }


def smooth_step(value: float) -> float:
    value = clamp(value, 0, 1)
    return value * value * (3 - 2 * value)


def expert_path(cones: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    offsets = [(world.BYPASS_START_S, 0.0)]
    for station, offset in cones:
        if not offset:
            break
        bypass = -math.copysign(world.BYPASS_OFFSET, offset)
        offsets.extend(((station - 1.7, bypass), (station + 0.65, bypass)))
    offsets.append((world.BYPASS_END_S, 0.0))
    return tuple(offsets)


def expert(state: Motion, path: tuple[tuple[float, float], ...],
           blocked_station: float | None) -> tuple[float, float]:
    station = state.station
    speed = 3.5
    if segment_at(station) == "curve" or 74 <= station <= 88:
        speed = 3.0
    if len(path) > 2 and world.BYPASS_START_S - 2 <= station <= world.BYPASS_END_S:
        speed = 2.35
    if world.BRICK_START_S - 4 <= station <= world.BRICK_END_S + 1.1:
        speed = 1.5

    # Accelerate along the straight before the first speed-zone gate, then brake before the next bend.
    if world.SPEED_APPROACH_S <= station <= world.SPEED_END_S + 2:
        speed = 9.9
    if world.SPEED_END_S + 2 < station <= world.SPEED_END_S + 19:
        speed = max(3.0, 9.9 - (station - world.SPEED_END_S - 2) * 0.54)

    if blocked_station is not None and station < world.BYPASS_END_S:
        clear_distance = max(0.0, blocked_station - station - world.BODY_FRONT - world.CONE_BASE / 2 - 0.48)
        speed = min(speed, math.sqrt(2 * 1.4 * clear_distance))
        if clear_distance < 0.25:
            speed = min(speed, 1.6 * clear_distance)

    lookahead = clamp(0.67 + 0.35 * state.speed, 1.15, 4.4)
    target_station = (station + lookahead) % world.TRACK_LENGTH
    target_offset = DEFAULT_OFFSET
    for (s0, offset0), (s1, offset1) in zip(path, path[1:]):
        if s0 <= target_station <= s1:
            target_offset = offset0 + (offset1 - offset0) * smooth_step((target_station - s0) / (s1 - s0))
            break
    target_x, target_y, _ = world.centreline(target_station, target_offset)
    dx, dy = target_x - state.pose.x, target_y - state.pose.y
    heading_error = angle_difference(math.atan2(dy, dx), state.pose.yaw)
    curvature = 2 * math.sin(heading_error) / max(0.7, math.hypot(dx, dy))
    yaw_rate = clamp(curvature * max(0.65, state.speed), -1.65, 1.65)
    return speed, yaw_rate


class GazeboTransport:
    def __init__(self, executable: Path, environment: dict[str, str], log):
        self.process = subprocess.Popen(
            [str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
            env=environment,
        )
        self.events: queue.Queue = queue.Queue(maxsize=140)
        self.stopping = threading.Event()
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def enqueue(self, event: tuple) -> None:
        while not self.stopping.is_set():
            try:
                self.events.put(event, timeout=0.2)
                return
            except queue.Full:
                continue

    def read(self) -> None:
        assert self.process.stdout is not None
        stream = self.process.stdout
        try:
            while not self.stopping.is_set():
                packet = stream.read(PACKET_STRUCT.size)
                if not packet:
                    break
                kind, size = PACKET_STRUCT.unpack(packet)
                if size > 64 * 1024 * 1024:
                    raise RuntimeError("Gazebo transport returned an invalid packet length")
                data = stream.read(size)
                if len(data) != size:
                    raise RuntimeError("Gazebo transport packet was truncated")
                self.enqueue((kind, data))
        except (OSError, RuntimeError, struct.error) as error:
            self.enqueue((b"E", str(error)))
        finally:
            self.enqueue((b"E", "Gazebo transport stream closed"))

    def command(self, speed: float, yaw_rate: float) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(f"{speed:.7f} {yaw_rate:.7f}\n".encode("ascii"))
        self.process.stdin.flush()

    def close(self) -> None:
        self.stopping.set()
        stop_process(self.process)
        self.reader.join(timeout=2)
        for stream in (self.process.stdout, self.process.stdin):
            if stream is not None:
                stream.close()


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def executable_bridge(override: Path | None, log_path: Path) -> Path:
    if override:
        if not override.is_file() or not os.access(override, os.X_OK):
            raise RuntimeError(f"Gazebo transport bridge is missing or isn't executable: {override}")
        return override.resolve()

    source = Path(__file__).resolve().with_name("native_transport")
    content = b"".join((source / name).read_bytes() for name in ("CMakeLists.txt", "main.cc"))
    digest = hashlib.sha256(content).hexdigest()[:12]
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    build = cache_root / "ring_training" / f"gz_harmonic_transport_{digest}"
    build.mkdir(parents=True, exist_ok=True)
    executable = build / "ring_gazebo_transport"
    if not shutil.which("cmake"):
        raise RuntimeError("CMake is required to build the Harmonic transport bridge")
    with (build / ".build-lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not executable.is_file():
            with log_path.open("w") as log:
                for command in (
                    ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(build), "--parallel", "2"],
                ):
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                    if result.returncode:
                        raise RuntimeError(f"Harmonic transport bridge build failed. See {log_path}")
    return executable


def image_packet(data: bytes, depth: bool) -> tuple[int, np.ndarray]:
    stamp, width, height, step, encoding = IMAGE_STRUCT.unpack_from(data)
    true_row_bytes = width * (4 if depth else 3)
    if step < true_row_bytes or len(data) != IMAGE_STRUCT.size + step * height:
        raise RuntimeError("Gazebo camera packet had an inconsistent size")
    rows = np.frombuffer(data, dtype=np.uint8, offset=IMAGE_STRUCT.size).reshape(height, step)
    pixels = np.ascontiguousarray(rows[:, :true_row_bytes])
    if depth and encoding == 3:
        return stamp, pixels.view("<f4").reshape(height, width)
    if not depth and encoding in (1, 2):
        rgb = pixels.reshape(height, width, 3)
        return stamp, rgb if encoding == 1 else cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    sensor = "depth" if depth else "RGB"
    raise RuntimeError(f"Gazebo {sensor} camera used an unsupported encoding: {encoding}")


class CameraPairs:
    def __init__(self):
        self.rgb: dict[int, np.ndarray] = {}
        self.depth: dict[int, np.ndarray] = {}
        self.count = 0
        self.usable_count = 0
        self.rejected_count = 0
        self.latest_usable_stamp = -1
        self.finite_depth_pixels = 0
        self.color_std = 0.0
        self.surface_visible_fraction = 0.0

    def update(self, kind: bytes, data: bytes) -> tuple[int, np.ndarray, np.ndarray] | None:
        is_depth = kind == b"D"
        destination, other = (self.depth, self.rgb) if is_depth else (self.rgb, self.depth)
        stamp, image = image_packet(data, is_depth)
        destination[stamp] = image
        if len(destination) > 12:
            del destination[min(destination)]
        if stamp not in other:
            return None
        rgb, depth = self.rgb.pop(stamp), self.depth.pop(stamp)
        if rgb.shape[:2] != depth.shape:
            raise RuntimeError("Gazebo RGB and depth frames differ in resolution")
        self.count += 1
        valid_depth = np.isfinite(depth) & (depth > 0)
        depth_count = int(valid_depth.sum())
        self.finite_depth_pixels = max(self.finite_depth_pixels, depth_count)
        # Disregard sky and warm-up frames where the renderer has a depth buffer
        # but has not shaded the road or obstacles yet.
        lower = slice(rgb.shape[0] // 2, None)
        surface = valid_depth[lower]
        surface_pixels = rgb[lower][surface]
        visible_fraction = (
            float(np.mean(surface_pixels.max(axis=1) > 8)) if len(surface_pixels) else 0.0
        )
        surface_variation = (
            float(surface_pixels.astype(np.float32).std(axis=0).max()) if len(surface_pixels) else 0.0
        )
        self.surface_visible_fraction = max(self.surface_visible_fraction, visible_fraction)
        self.color_std = max(self.color_std, surface_variation)
        if (
            stamp <= self.latest_usable_stamp
            or depth_count < depth.size * 0.05
            or visible_fraction < 0.35
            or surface_variation < 2
        ):
            self.rejected_count += 1
            return None
        self.usable_count += 1
        self.latest_usable_stamp = stamp
        return stamp, rgb, depth


class VisualWorker:
    """Keep only the latest RGB-D observation while inference is in progress."""

    def __init__(self, compute_action):
        self.compute_action = compute_action
        self.wakeup = threading.Condition()
        self.observation = None
        self.ready = None
        self.error = None
        self.stopping = False
        self.dropped_inputs = 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, observation: tuple) -> None:
        with self.wakeup:
            if self.observation is not None:
                self.dropped_inputs += 1
            self.observation = observation
            self.wakeup.notify()

    def take(self):
        with self.wakeup:
            if self.error is not None:
                raise RuntimeError(f"Visual controller failed: {self.error}")
            ready = self.ready
            self.ready = None
            return ready

    def close(self) -> None:
        with self.wakeup:
            self.stopping = True
            self.wakeup.notify()
        self.thread.join(timeout=4)

    def run(self) -> None:
        while True:
            with self.wakeup:
                self.wakeup.wait_for(lambda: self.stopping or self.observation is not None)
                if self.stopping:
                    return
                stamp, rgb, depth = self.observation
                self.observation = None
            try:
                start = time.monotonic()
                speed, yaw_rate = self.compute_action(stamp, rgb, depth)
                if not math.isfinite(speed) or not math.isfinite(yaw_rate):
                    raise RuntimeError("Visual controller returned a non-finite driving command")
                with self.wakeup:
                    self.ready = (stamp, float(speed), float(yaw_rate), time.monotonic() - start)
            except Exception as error:
                with self.wakeup:
                    self.error = error
                return


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_sample(output: Path, manifest, count: int, observation: tuple, state: Motion, action: tuple) -> None:
    stamp, rgb, depth = observation
    filename = f"{count:06d}"
    rgb_path = f"rgb/{filename}.png"
    depth_path = f"depth/{filename}.npy"
    if not cv2.imwrite(str(output / rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError(f"Couldn't write RGB frame: {rgb_path}")
    np.save(output / depth_path, depth, allow_pickle=False)
    record = {
        "sim_time": stamp * 1e-9,
        "rgb": rgb_path,
        "depth": depth_path,
        "speed_mps": state.speed,
        "yaw_rate_rad_s": state.yaw_rate,
        "target_speed_mps": action[0],
        "target_yaw_rate_rad_s": action[1],
        "source": "gazebo_expert",
        "segment": segment_at(state.station),
        "truth_time_offset_s": (state.pose.stamp - stamp) * 1e-9,
    }
    manifest.write(json.dumps(record) + "\n")


def run_episode(args: argparse.Namespace, output: Path) -> dict:
    mode = args.mode
    render = not args.no_render
    controller = args.controller if mode == "run" else ("straight" if mode == "smoke" else "expert")
    is_visual = controller in ("policy", "vision-guide", "guided-policy")
    uses_learning = controller in ("policy", "guided-policy")
    uses_guide = controller in ("vision-guide", "guided-policy")
    guide = None
    guided = None
    compute_action = None
    layout = "clear" if args.no_cones else world.resolve_layout(args.seed, args.layout)
    placements = world.cone_stations(args.seed, layout, args.cone_count)
    cones = tuple(world.centreline(station, offset)[:2] for station, offset in placements)
    demonstration = expert_path(placements)
    blocked_station = next((station for station, offset in placements if not offset), None)
    scorer = Scoring(cones)
    camera = CameraPairs()
    run_id = uuid.uuid4().hex
    details = {
        "schema_version": 1,
        "world_schema": world.WORLD_SCHEMA,
        "episode_id": run_id,
        "source": "gazebo_expert" if mode == "record" else "gazebo",
        "simulator": "Gazebo Harmonic",
        "mode": mode,
        "controller": controller,
        "seed": args.seed,
        "layout": layout,
        "requested_layout": args.layout,
        "rgb_encoding": "RGB8 PNG",
        "depth_encoding": "Float32 NPY in metres",
        "action_fields": ["target_speed_mps", "target_yaw_rate_rad_s"],
        "camera_hz": args.camera_hz if render else None,
        "camera_intrinsics": {
            "width": args.width, "height": args.height, "horizontal_fov_rad": world.CAMERA_HFOV,
            "depth_near_m": world.CAMERA_DEPTH_NEAR, "depth_far_m": world.CAMERA_DEPTH_FAR,
        } if render else None,
        "sample_hz": args.sample_hz if mode == "record" else None,
        "cones": list(cones),
        "cone_stations_m_and_offsets_m": placements,
        "cone_count": len(cones),
        "spawn_offset_m": args.spawn_offset,
        "spawn_yaw_error_rad": args.spawn_yaw_error,
        "render_engine": args.render_engine if render else None,
    }
    stop_reason = None
    runtime_error = None
    samples = commands = 0
    latest_stamp = None
    previous_action = (0.0, 0.0)
    first_command_time = None
    worker = None
    policy_dropped_actions = 0
    policy_inference_wall_times: list[float] = []
    policy_command_delays: list[float] = []
    missing_sensors = False
    version = None

    try:
        if not shutil.which(args.gz):
            raise RuntimeError(f"Gazebo executable {args.gz!r} is unavailable. Install Gazebo Harmonic or source Jazzy.")
        version_result = subprocess.run([args.gz, "sim", "--version"], capture_output=True, text=True, timeout=10)
        if version_result.returncode or "version 8." not in version_result.stdout:
            raise RuntimeError("Gazebo Harmonic Sim 8.x is required")
        version = version_result.stdout.splitlines()[0]
        bridge = executable_bridge(args.bridge, output / "bridge-build.log")
        if uses_learning:
            from .learning import Policy
            from .perception import Perception
            policy = Policy.load(args.policy)
            perception = Perception(
                road_model=args.road_model, cone_model=args.cone_model, threads=args.threads,
                depth_max_m=policy.depth_max_m,
            )
            policy = Policy.load(args.policy, perception=perception)

            def infer_action(_stamp, rgb, depth):
                return policy.predict(perception.features(rgb, depth))

            compute_action = infer_action
            details["policy_file"] = str(args.policy.resolve())
            details["policy_sha256"] = hashlib.sha256(args.policy.read_bytes()).hexdigest()
            details["policy_max_observation_age_sim_s"] = args.max_policy_age
            details["controller_family"] = "learned_rgbd_policy"
        if uses_guide:
            from .visual_drive import CONTROLLER_VERSION, VisionGuide

            guide = VisionGuide(
                width=args.width, height=args.height, hfov=world.CAMERA_HFOV,
                camera_x=world.CAMERA_X, camera_y=world.CAMERA_Y, camera_z=world.CAMERA_Z,
                pitch=world.CAMERA_PITCH, yellow_half_width=world.YELLOW_LINE_OFFSET,
                body_half_width=world.BODY_HALF_WIDTH,
                body_front=world.BODY_FRONT, cone_half_width=world.CONE_BASE / 2,
                bypass_offset=0.90 * world.BYPASS_OFFSET,
            )
            compute_action = guide.command
            details["controller_family"] = "rgbd_geometry_baseline"
            details["vision_guide_version"] = CONTROLLER_VERSION
            details["vision_guide_max_observation_age_sim_s"] = args.max_policy_age
        if controller == "guided-policy":
            from .visual_drive import GUIDED_POLICY_MAX_AGE_S, GUIDED_POLICY_WEIGHT

            details["controller_family"] = "geometry_guided_learned_rgbd_policy"
            details["guided_policy_weight"] = GUIDED_POLICY_WEIGHT
            details["guided_policy_learned_max_age_sim_s"] = GUIDED_POLICY_MAX_AGE_S
        if mode == "run" and args.speed_cap_mps is not None:
            details["configured_speed_cap_mps"] = args.speed_cap_mps

        sdf = output / "ring.sdf"
        world.write_world(
            sdf, seed=args.seed, cones=layout != "clear", layout=layout, cone_count=args.cone_count,
            width=args.width, height=args.height,
            camera_hz=args.camera_hz, render=render, spawn_offset=args.spawn_offset,
            spawn_yaw_error=args.spawn_yaw_error, render_engine=args.render_engine, realtime_factor=args.rtf,
        )
        details["world_sha256"] = hashlib.sha256(sdf.read_bytes()).hexdigest()

        environment = os.environ.copy()
        environment["GZ_PARTITION"] = f"gazebo_ring_{run_id}"
        command = [args.gz, "sim", "-s", "-r", "-v", "3"]
        if render and sys.platform.startswith("linux") and args.render_engine == "ogre2":
            command.append("--headless-rendering")
        command.append(str(sdf))

        with ExitStack() as stack:
            sim_log = stack.enter_context((output / "gazebo.log").open("w"))
            bridge_log = stack.enter_context((output / "transport.log").open("w"))
            trace_stream = stack.enter_context((output / "truth_trace.csv").open("w", newline="", encoding="utf-8"))
            trace = csv.writer(trace_stream)
            trace.writerow(TRACE_FIELDS)
            manifest = None
            if mode == "record":
                (output / "rgb").mkdir()
                (output / "depth").mkdir()
                manifest = stack.enter_context((output / "samples.jsonl").open("w", encoding="utf-8"))

            if controller == "guided-policy":
                from .visual_drive import GuidedPolicy

                learned_worker = VisualWorker(infer_action)
                stack.callback(learned_worker.close)
                guided = GuidedPolicy(guide, learned_worker)
                compute_action = guided.command
            if compute_action is not None:
                worker = VisualWorker(compute_action)
                stack.callback(worker.close)
            transport = GazeboTransport(bridge, environment, bridge_log)
            stack.callback(transport.close)
            sim = subprocess.Popen(command, stdout=sim_log, stderr=subprocess.STDOUT, env=environment)
            stack.callback(stop_process, sim)
            startup_deadline = time.monotonic() + args.startup_timeout
            last_event = time.monotonic()
            last_pose_event = last_event
            last_pair_event = last_event
            last_command = -math.inf
            next_sample = None
            pending: deque = deque(maxlen=6)
            ground_stamps = [-1] * len(world.WHEEL_LINKS)

            while True:
                if sim.poll() is not None:
                    raise RuntimeError(f"Gazebo exited with code {sim.returncode}. See gazebo.log")
                if transport.process.poll() is not None:
                    raise RuntimeError(f"Gazebo transport exited with code {transport.process.returncode}. See transport.log")
                wall_time = time.monotonic()
                if first_command_time is None and wall_time >= startup_deadline:
                    missing_sensors = render and not camera.usable_count
                    raise RuntimeError(
                        "Timed out waiting for a shaded Gazebo RGB-D frame and world pose; see gazebo.log and transport.log"
                        if render else "Timed out waiting for Gazebo world pose; see gazebo.log and transport.log"
                    )
                if first_command_time is not None and (
                    max(wall_time - last_event, wall_time - last_pose_event) >= 15
                    or (render and wall_time - last_pair_event >= 15)
                ):
                    raise RuntimeError("Gazebo stopped producing world pose or paired sensor updates")
                if worker is not None:
                    ready_action = worker.take()
                    if ready_action is not None:
                        stamp, speed, yaw_rate, inference_wall_time = ready_action
                        policy_inference_wall_times.append(inference_wall_time)
                        if latest_stamp is not None:
                            lag = max(0.0, (latest_stamp - stamp) * 1e-9)
                            if lag > args.max_policy_age:
                                policy_dropped_actions += 1
                            else:
                                previous_action = (
                                    clamp(speed, 0, min(11.0, args.speed_cap_mps or 11.0)),
                                    clamp(yaw_rate, -1.65, 1.65),
                                )
                                transport.command(*previous_action)
                                commands += 1
                                policy_command_delays.append(lag)
                                last_command = latest_stamp * 1e-9
                                if first_command_time is None:
                                    first_command_time = last_command
                    if first_command_time is not None and latest_stamp * 1e-9 - last_command > 2 * args.max_policy_age:
                        raise RuntimeError("No fresh visual driving command within the configured simulated age limit")
                try:
                    kind, data = transport.events.get(timeout=0.2)
                except queue.Empty:
                    continue
                last_event = time.monotonic()
                if kind == b"E":
                    raise RuntimeError(f"{data}; see transport.log")
                if kind == b"G":
                    stamp, mask = GROUND_STRUCT.unpack(data)
                    for index in range(len(ground_stamps)):
                        if mask & (1 << index):
                            ground_stamps[index] = max(ground_stamps[index], stamp)
                elif kind == b"P":
                    stamp, x, y, z, yaw, *wheels = POSE_STRUCT.unpack(data)
                    pose = Pose(stamp, x, y, z, yaw)
                    if latest_stamp is not None and latest_stamp >= pose.stamp:
                        continue
                    latest_stamp = pose.stamp
                    last_pose_event = last_event
                    state = scorer.update(pose)
                    grounded = tuple(
                        ground >= 0 and abs(ground - stamp) <= GROUND_CONTACT_MAX_AGE_NS
                        for ground in ground_stamps
                    )
                    touches = scorer.score_wheels(pose, wheels, grounded)
                    trace.writerow([
                        pose.time, pose.x, pose.y, pose.z, pose.yaw, state.station, state.offset, state.speed,
                        state.adjacent_speed, state.yaw_rate, segment_at(state.station), *previous_action,
                        sum(grounded), "inner" in touches, "outer" in touches,
                    ])
                    ready = camera.usable_count > 0 if render else True
                    if ready and not is_visual and pose.time - last_command >= 0.025:
                        if mode == "smoke":
                            action = (2.3, 0.0)
                        elif controller == "straight":
                            action = (3.5, 0.0)
                        else:
                            action = expert(state, demonstration, blocked_station)
                        if mode == "run" and args.speed_cap_mps is not None:
                            action = min(args.speed_cap_mps, action[0]), action[1]
                        transport.command(*action)
                        previous_action = action
                        last_command = pose.time
                        commands += 1
                        if first_command_time is None:
                            first_command_time = pose.time
                elif kind in (b"R", b"D"):
                    pair = camera.update(kind, data)
                    if pair is not None:
                        last_pair_event = last_event
                        if worker is not None:
                            worker.submit(pair)
                        if manifest is not None:
                            pending.append(pair)
                elif kind == b"C":
                    contact_stamp, = struct.unpack_from("<Q", data)
                    name = data[8:].decode("utf-8", errors="replace")
                    scorer.contacts.setdefault(name, contact_stamp * 1e-9)

                while latest_stamp is not None and pending and pending[0][0] <= latest_stamp:
                    observation = pending.popleft()
                    stamp = observation[0]
                    frame_time = stamp * 1e-9
                    state = min(scorer.history, key=lambda sample: abs(sample.pose.stamp - stamp))
                    if abs(state.pose.stamp - stamp) > 50_000_000:
                        continue
                    if first_command_time is not None and (
                        next_sample is None or frame_time >= next_sample - 0.001
                    ):
                        action = expert(state, demonstration, blocked_station)
                        record_sample(output, manifest, samples, observation, state, action)
                        samples += 1
                        period = 1.0 / args.sample_hz
                        next_sample = max((next_sample or frame_time) + period, frame_time + period - 1 / args.camera_hz)

                if first_command_time is None or latest_stamp is None:
                    continue
                duration = latest_stamp * 1e-9 - first_command_time
                if mode != "smoke" and scorer.finished:
                    elapsed_to_gate = scorer.gates["finish"] - first_command_time
                    stop_reason = "finish_gate" if elapsed_to_gate <= args.duration else "duration_reached"
                    break
                if mode == "run" and (scorer.contacts or scorer.cone_overlaps or scorer.first_offtrack is not None):
                    stop_reason = "disqualifying_event"
                    break
                if duration >= args.duration:
                    stop_reason = "duration_reached"
                    break
            transport.command(0, 0)
    except (ImportError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        runtime_error = str(error)
        stop_reason = "runtime_error"
    except KeyboardInterrupt:
        runtime_error = "interrupted"
        stop_reason = "interrupted"

    result = scorer.result()
    if mode == "smoke":
        displaced = scorer.total_distance >= 1.0 and scorer.max_speed >= 0.6
        valid_camera = camera.usable_count >= 2
        failures = []
        if not displaced:
            failures.append("no_measured_dynamics")
        if render and not valid_camera:
            failures.append("no_usable_gazebo_rgbd")
        if not all(scorer.grounded_samples.values()):
            failures.append("wheel_contact_truth_unavailable")
        result["passed"] = not failures
        result["failures"] = failures
        result["completed_laps"] = 0

    if mode == "record" and samples == 0:
        result["passed"] = False
        result["failures"].append("no_expert_samples")
    if mode != "smoke" and stop_reason == "duration_reached":
        result["passed"] = False
        result["failures"].append("finish_gate_deadline_exceeded")
    if guided is not None and not guided.stats["learned_residual_applied_frames"]:
        result["passed"] = False
        result["failures"].append("guided_policy_learned_residual_not_applied")
    if runtime_error:
        result["passed"] = False
        result["failures"].append("missing_gazebo_rgbd" if missing_sensors else "runtime_error")
    result.update({
        "mode": mode,
        "world_schema": world.WORLD_SCHEMA,
        "controller": controller,
        "seed": args.seed,
        "layout": layout,
        "cone_count": len(cones),
        "last_measured_speed_kmh": scorer.history[-1].speed * 3.6 if scorer.history else None,
        "stop_reason": stop_reason,
        "driving_deadline_sim_s": args.duration,
        "runtime_error": runtime_error,
        "gazebo_version": version,
        "rgbd_pairs": camera.count,
        "rgbd_usable_pairs": camera.usable_count,
        "rgbd_rejected_pairs": camera.rejected_count,
        "depth_valid_pixels_max": camera.finite_depth_pixels,
        "rgb_surface_color_std_max": camera.color_std,
        "rgb_surface_visible_fraction_max": camera.surface_visible_fraction,
        "expert_samples": samples,
        "commands_sent": commands,
        "elapsed_driving_sim_time_s": (
            latest_stamp * 1e-9 - first_command_time
            if latest_stamp is not None and first_command_time is not None else None
        ),
    })
    if is_visual:
        prefix = controller.replace("-", "_")
        result.update({
            "controller_family": details.get("controller_family"),
            f"{prefix}_dropped_observations": worker.dropped_inputs if worker is not None else 0,
            f"{prefix}_rejected_stale_actions": policy_dropped_actions,
            f"{prefix}_observation_age_max_sim_s": max(policy_command_delays, default=None),
            f"{prefix}_inference_mean_wall_s": (
                sum(policy_inference_wall_times) / len(policy_inference_wall_times)
                if policy_inference_wall_times else None
            ),
        })
    if uses_learning:
        result["policy_sha256"] = details.get("policy_sha256")
    if guide is not None:
        result["vision_guide_version"] = details["vision_guide_version"]
        result["vision_guide_observations"] = guide.stats
    if guided is not None:
        result["guided_policy_observations"] = guided.stats
        result["guided_policy_learned_dropped_observations"] = guided.learned_worker.dropped_inputs
        result["guided_policy_weight"] = details["guided_policy_weight"]
    details.update({"samples": samples, "passed": result["passed"], "result": "result.json"})
    write_json(output / "metadata.json", details)
    write_json(output / "result.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = argparse.ArgumentParser(description=__doc__)
    modes = root.add_subparsers(dest="mode", required=True)
    for mode, help_text in (
        ("record", "run the Gazebo expert, collect RGB-D and labels, and score the episode"),
        ("run", "run and score a closed-loop controller"),
        ("smoke", "check Gazebo dynamics and RGB-D over a brief straight-line run"),
    ):
        parser = modes.add_parser(mode, help=help_text)
        parser.add_argument("--output", type=Path, required=True, help="new directory for this episode")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--duration", type=float, default=4.0 if mode == "smoke" else 100.0, help="maximum driving sim seconds")
        parser.add_argument("--startup-timeout", type=float, default=60.0, help="maximum startup wall seconds")
        parser.add_argument("--camera-hz", type=float, default=10.0)
        parser.add_argument("--width", type=int, default=320)
        parser.add_argument("--height", type=int, default=240)
        parser.add_argument("--spawn-offset", type=float, default=DEFAULT_OFFSET)
        parser.add_argument("--spawn-yaw-error", type=float, default=0.0, help="initial yaw error in radians")
        parser.add_argument("--rtf", type=float, help="target real-time factor (defaults to 0.12 for policy and 1 otherwise)")
        parser.add_argument("--layout", choices=world.LAYOUTS, default=world.DEFAULT_LAYOUT)
        parser.add_argument("--cone-count", type=int, choices=(3, 4, 5))
        parser.add_argument("--no-cones", action="store_true")
        parser.add_argument("--render-engine", choices=("ogre2", "ogre"), default="ogre2")
        parser.add_argument("--gz", default="gz", help="Gazebo Harmonic CLI")
        parser.add_argument("--bridge", type=Path, help="use a prebuilt native Gazebo transport executable")
        if mode == "record":
            parser.add_argument("--sample-hz", type=float, default=4.0)
        else:
            parser.add_argument("--no-render", action="store_true", help="skip camera sensors for a dynamics-only run")
        parser.set_defaults(no_render=False, sample_hz=4.0)
        if mode == "run":
            parser.add_argument(
                "--controller", choices=("expert", "policy", "vision-guide", "guided-policy", "straight"),
                default="expert",
            )
            parser.add_argument("--policy", type=Path)
            defaults = Path(__file__).resolve().parents[1] / "models"
            parser.add_argument("--road-model", type=Path, default=defaults / "road-4060-v1.onnx")
            parser.add_argument("--cone-model", type=Path, default=defaults / "traffic-cone.onnx")
            parser.add_argument("--threads", type=int, default=2)
            parser.add_argument("--speed-cap-mps", type=float, help="optional target speed cap for a control trial")
            parser.add_argument("--max-policy-age", type=float, default=0.35, help="maximum policy observation age in simulated seconds")
    args = root.parse_args(argv)
    if args.rtf is None:
        args.rtf = 0.12 if args.mode == "run" and args.controller == "policy" else 1.0
    if not all(math.isfinite(value) and value > 0 for value in (
        args.duration, args.startup_timeout, args.camera_hz, args.sample_hz, args.rtf,
    )):
        root.error("duration, rates and startup timeout must be finite and positive")
    if args.width <= 0 or args.height <= 0:
        root.error("camera dimensions must be positive")
    if args.mode == "record" and args.sample_hz > args.camera_hz:
        root.error("--sample-hz cannot exceed --camera-hz")
    if args.mode == "run" and args.controller in ("policy", "guided-policy") and (args.policy is None or args.no_render):
        root.error(f"--controller {args.controller} requires --policy and enabled camera rendering")
    if args.mode == "run" and args.controller == "vision-guide" and args.no_render:
        root.error("--controller vision-guide requires camera rendering")
    if args.mode == "run":
        for name, value in (("--speed-cap-mps", args.speed_cap_mps), ("--max-policy-age", args.max_policy_age)):
            if value is not None and (not math.isfinite(value) or value <= 0):
                root.error(f"{name} must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    try:
        output.mkdir(parents=True)
    except FileExistsError:
        print(f"Choose a new episode directory: {output}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"Couldn't create the episode directory: {error}", file=sys.stderr)
        return 2
    result = run_episode(args, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
