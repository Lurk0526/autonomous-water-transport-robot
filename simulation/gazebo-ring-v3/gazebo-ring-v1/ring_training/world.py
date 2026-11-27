"""Generate the Gazebo Harmonic world and its local meshes.

The stadium centreline is counter-clockwise. The rear axle is the vehicle
reference point and positive lateral offsets point to the vehicle's left.
Nominal vehicle geometry comes from the final cart CAD. Dynamic parameters and
camera placement are simulation assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import struct
from typing import Sequence
import xml.etree.ElementTree as ET
import zlib


WORLD_NAME = "ring"
WORLD_SCHEMA = "ring-cad-narrow-serial-v3"
MODEL_NAME = "cart"
VEHICLE_ASSETS = Path(__file__).resolve().parents[1] / "vehicle_assets"
VEHICLE_CAD = json.loads((VEHICLE_ASSETS / "cad-measurements.json").read_text(encoding="utf-8"))
STRAIGHT_LENGTH = 88.0
HALF_STRAIGHT = STRAIGHT_LENGTH / 2
TURN_RADIUS = 12.0
HALF_TURN_LENGTH = math.pi * TURN_RADIUS
TRACK_LENGTH = 2 * (STRAIGHT_LENGTH + HALF_TURN_LENGTH)
ROAD_HALF_WIDTH = 0.72
LANE_OFFSET = 0.0
YELLOW_LINE_OFFSET = 0.53
YELLOW_LINE_WIDTH = 0.06
START_S = 6.0
SPAWN_S = START_S - 0.8
BYPASS_START_S = 9.0
BYPASS_END_S = 49.0
BYPASS_OFFSET = 0.252
BRICK_START_S = 53.0
BRICK_END_S = 67.0
SPEED_START_S = STRAIGHT_LENGTH + HALF_TURN_LENGTH + 34.0
SPEED_END_S = SPEED_START_S + 20.0
SPEED_APPROACH_S = SPEED_START_S - 23.0
SPEED_LIMIT_MPS = 30.0 / 3.6

WHEELBASE = VEHICLE_CAD["wheelbase_m"]
WHEEL_TRACK = VEHICLE_CAD["nominal_track_m"]
WHEEL_RADIUS = VEHICLE_CAD["wheel_radius_m"]
WHEEL_WIDTH = VEHICLE_CAD["wheel_width_m"]
BODY_FRONT = VEHICLE_CAD["scoring_envelope"]["front_from_rear_axle_m"]
BODY_REAR = VEHICLE_CAD["scoring_envelope"]["rear_from_rear_axle_m"]
BODY_HALF_WIDTH = VEHICLE_CAD["scoring_envelope"]["body_half_width_m"]
MAX_STEER_RAD = 0.48
MIN_STEP = 0.004
CONE_BASE = 0.34
CONE_HEIGHT = 0.53
TRAINING_LAYOUTS = ("left", "right", "slalom", "clear")
LAYOUTS = (*TRAINING_LAYOUTS, "blocked", "mixed")
DEFAULT_LAYOUT = "slalom"

CAMERA_X = 0.68
CAMERA_Y = 0.0
CAMERA_Z = 0.55
CAMERA_PITCH = 0.10
CAMERA_HFOV = 1.40
CAMERA_DEPTH_NEAR = 0.10
CAMERA_DEPTH_FAR = 24.0
CAMERA_FRAME = "cart_camera_optical"
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_CAMERA_HZ = 15.0

CMD_TOPIC = "/model/cart/cmd_vel"
POSE_TOPIC = "/model/cart/pose"
ODOM_TOPIC = "/model/cart/odometry"
CAMERA_TOPIC = "/cart/camera"
IMU_TOPIC = "/cart/imu"
CONTACT_TOPIC = "/cart/contacts"
WHEEL_LINKS = ("rear_left_wheel", "rear_right_wheel", "front_left_wheel", "front_right_wheel")


def centreline(s: float, offset: float = 0.0) -> tuple[float, float, float]:
    """Return world x, y and tangent heading at route station s and offset."""
    s %= TRACK_LENGTH
    radius = TURN_RADIUS - offset
    if s < STRAIGHT_LENGTH:
        return s - HALF_STRAIGHT, -radius, 0.0
    s -= STRAIGHT_LENGTH
    if s < HALF_TURN_LENGTH:
        angle = s / TURN_RADIUS
        return HALF_STRAIGHT + radius * math.sin(angle), -radius * math.cos(angle), angle
    s -= HALF_TURN_LENGTH
    if s < STRAIGHT_LENGTH:
        return HALF_STRAIGHT - s, radius, math.pi
    angle = (s - STRAIGHT_LENGTH) / TURN_RADIUS
    return -HALF_STRAIGHT - radius * math.sin(angle), radius * math.cos(angle), math.pi + angle


def resolve_layout(seed: int, layout: str) -> str:
    """Resolve the requested obstacle layout into one of the concrete courses."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown cone layout: {layout}")
    return TRAINING_LAYOUTS[seed % len(TRAINING_LAYOUTS)] if layout == "mixed" else layout


def cone_stations(seed: int = 0, layout: str = DEFAULT_LAYOUT,
                  count: int | None = None) -> tuple[tuple[float, float], ...]:
    """Return the ordered (station, lateral offset) for individual cones."""
    layout = resolve_layout(seed, layout)
    if count is not None and count not in (3, 4, 5):
        raise ValueError("cone count must be 3, 4 or 5")
    if layout == "clear":
        return ()
    rng = random.Random(seed)
    count = count if count is not None else (4 if seed == 0 else 3 + seed % 3)
    stations = []
    first = 20.0 + rng.uniform(-0.65, 0.65)
    for index in range(count):
        station = first + 5.2 * index + rng.uniform(-0.22, 0.22)
        side = -1 if layout == "left" else 1
        if layout in ("slalom", "blocked"):
            side = 1 if (index + seed) % 2 else -1
        offset = side * rng.uniform(0.413, 0.451)
        if layout == "blocked" and index == 1:
            offset = 0.0
        stations.append((station, offset))
    return tuple(stations)


def cone_positions(seed: int = 0, layout: str = DEFAULT_LAYOUT,
                   count: int | None = None) -> tuple[tuple[float, float], ...]:
    return tuple(centreline(station, offset)[:2] for station, offset in cone_stations(seed, layout, count))


def _format(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, (tuple, list)):
        return " ".join(_format(item) for item in value)
    return str(value)


def _element(parent: ET.Element, tag: str, value: object | None = None, **attrs: str) -> ET.Element:
    element = ET.SubElement(parent, tag, attrs)
    if value is not None:
        element.text = _format(value)
    return element


def _plugin(parent: ET.Element, filename: str, name: str, **settings: object) -> ET.Element:
    plugin = _element(parent, "plugin", filename=f"gz-sim-{filename}-system", name=f"gz::sim::systems::{name}")
    for key, value in settings.items():
        _element(plugin, key, value)
    return plugin


def _model(world: ET.Element, name: str, pose: Sequence[float] | None = None, *, static: bool = True) -> ET.Element:
    model = _element(world, "model", name=name)
    _element(model, "static", static)
    if pose is not None:
        _element(model, "pose", pose)
    return model


def _geometry(parent: ET.Element, kind: str, **parts: object) -> None:
    shape = _element(_element(parent, "geometry"), kind)
    for name, value in parts.items():
        _element(shape, name, value)


def _shape(
    link: ET.Element,
    name: str,
    kind: str,
    pose: Sequence[float],
    *,
    color: Sequence[float] | None = None,
    texture: Path | None = None,
    friction: float | None = None,
    **parts: object,
) -> ET.Element:
    shape = _element(link, "visual" if color is not None else "collision", name=name)
    _element(shape, "pose", pose)
    _geometry(shape, kind, **parts)
    if color is not None:
        material = _element(shape, "material")
        _element(material, "ambient", color)
        _element(material, "diffuse", color)
        _element(material, "specular", (0.12, 0.12, 0.12, 1))
        if texture is not None:
            metal = _element(_element(material, "pbr"), "metal")
            _element(metal, "albedo_map", texture)
            _element(metal, "metalness", 0)
            _element(metal, "roughness", 0.96)
    if friction is not None:
        friction_element = _element(_element(shape, "surface"), "friction")
        ode = _element(friction_element, "ode")
        _element(ode, "mu", friction)
        _element(ode, "mu2", friction)
    return shape


def _box(link: ET.Element, name: str, xyz: Sequence[float], size: Sequence[float], **kwargs: object) -> ET.Element:
    return _shape(link, name, "box", (*xyz, 0, 0, 0), size=size, **kwargs)


class _Mesh:
    """Build a small Wavefront mesh with explicit normals and optional UVs."""

    def __init__(self) -> None:
        self.vertices: list[tuple[float, float, float]] = []
        self.uvs: list[tuple[float, float]] = []
        self.faces: list[tuple[int, ...]] = []

    def face(self, vertices: Sequence[tuple[float, float, float]], uv: Sequence[tuple[float, float]] | None = None) -> None:
        index = len(self.vertices) + 1
        self.vertices.extend(vertices)
        self.uvs.extend(uv if uv is not None else [(0.0, 0.0)] * len(vertices))
        for i in range(1, len(vertices) - 1):
            self.faces.append((index, index + i, index + i + 1))

    def write(self, path: Path, *, flat_up: bool = False) -> Path:
        with path.open("w", encoding="utf-8") as file:
            file.write(f"o {path.stem}\n")
            file.writelines(f"v {_format(vertex)}\n" for vertex in self.vertices)
            file.writelines(f"vt {_format(uv)}\n" for uv in self.uvs)
            if flat_up:
                file.write("vn 0 0 1\n")
            else:
                for a, b, c in self.faces:
                    p, q, r = (self.vertices[i - 1] for i in (a, b, c))
                    u = tuple(y - x for x, y in zip(p, q))
                    v = tuple(y - x for x, y in zip(p, r))
                    cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
                    magnitude = math.sqrt(sum(component * component for component in cross))
                    file.write(f"vn {_format(tuple(component / magnitude for component in cross))}\n")
            for normal, face in enumerate(self.faces, 1):
                file.write("f " + " ".join(f"{i}/{i}/{1 if flat_up else normal}" for i in face) + "\n")
        return path


def _strip(mesh: _Mesh, d0: float, d1: float, z: float, start: float = 0.0, end: float = TRACK_LENGTH) -> None:
    """Tessellate straight/curved strips without seams at segment boundaries."""
    breaks = (start, *(s for s in (STRAIGHT_LENGTH, STRAIGHT_LENGTH + HALF_TURN_LENGTH,
                                   2 * STRAIGHT_LENGTH + HALF_TURN_LENGTH) if start < s < end), end)
    for a, b in zip(breaks, breaks[1:]):
        samples = math.ceil((b - a) / 0.65)
        for i in range(samples):
            s0 = a + (b - a) * i / samples
            s1 = a + (b - a) * (i + 1) / samples
            points = ((s0, d0), (s1, d0), (s1, d1), (s0, d1))
            mesh.face([(*centreline(s, d)[:2], z) for s, d in points], [(s / 2.4, d / 2.4) for s, d in points])


def _asphalt_texture(path: Path) -> Path:
    """Write a repeatable fine asphalt texture using only the standard library."""
    rng = random.Random(510)
    size = 128
    coarse = [[rng.uniform(-6, 6) for _ in range(16)] for _ in range(16)]
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            ax, tx = divmod(x, 8)
            ay, ty = divmod(y, 8)
            tx /= 8
            ty /= 8
            a = coarse[ay][ax] * (1 - tx) + coarse[ay][(ax + 1) % 16] * tx
            b = coarse[(ay + 1) % 16][ax] * (1 - tx) + coarse[(ay + 1) % 16][(ax + 1) % 16] * tx
            base = int(57 + a * (1 - ty) + b * ty + rng.gauss(0, 7))
            if rng.random() < 0.042:
                base += rng.randint(9, 22)
            rows.extend((base, base + 1, base + 3))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return path


def _road(world: ET.Element, assets: Path) -> None:
    ground = _element(_model(world, "ground"), "link", name="surface")
    _shape(ground, "ground_collision", "plane", (0,) * 6, normal=(0, 0, 1), size=(260, 140), friction=1.1)
    _shape(ground, "verge", "plane", (0, 0, -0.01, 0, 0, 0), normal=(0, 0, 1),
           size=(260, 140), color=(0.24, 0.32, 0.20, 1))

    road = _element(_model(world, "paved_road"), "link", name="surface")
    asphalt = _Mesh()
    _strip(asphalt, -ROAD_HALF_WIDTH, ROAD_HALF_WIDTH, 0.002)
    _shape(road, "asphalt", "mesh", (0,) * 6, uri=asphalt.write(assets / "asphalt.obj", flat_up=True),
           color=(0.19, 0.20, 0.22, 1), texture=_asphalt_texture(assets / "asphalt.png"))

    yellow = _Mesh()
    for offset in (-YELLOW_LINE_OFFSET, YELLOW_LINE_OFFSET):
        _strip(yellow, offset - YELLOW_LINE_WIDTH / 2, offset + YELLOW_LINE_WIDTH / 2, 0.006)
    _shape(road, "inner_and_outer_yellow_boundaries", "mesh", (0,) * 6,
           uri=yellow.write(assets / "yellow_boundaries.obj", flat_up=True), color=(0.95, 0.75, 0.06, 1))
    boundaries = _Mesh()
    for offset in (-ROAD_HALF_WIDTH + 0.045, ROAD_HALF_WIDTH - 0.045):
        _strip(boundaries, offset - 0.034, offset + 0.034, 0.005)
    _shape(road, "white_road_edges", "mesh", (0,) * 6, uri=boundaries.write(assets / "edges.obj", flat_up=True),
           color=(0.79, 0.79, 0.73, 1))

    for name, s, color in (
        ("start_finish", START_S, (0.90, 0.90, 0.85, 1)),
        ("speed_approach", SPEED_APPROACH_S, (0.11, 0.12, 0.90, 1)),
        ("speed_entry", SPEED_START_S, (0.02, 0.75, 0.90, 1)),
        ("speed_exit", SPEED_END_S, (0.78, 0.04, 0.80, 1)),
    ):
        x, y, _ = centreline(s)
        for index in range(-2, 3):
            _box(road, f"{name}_{index+2}", (x, y + index * ROAD_HALF_WIDTH / 2.7, 0.009),
                 (0.18, ROAD_HALF_WIDTH / 4.5, 0.004), color=color)
        for side in (-1, 1):
            post_y = y + side * (ROAD_HALF_WIDTH + 0.40)
            _box(road, f"{name}_post_{side}", (x, post_y, 0.53), (0.14, 0.14, 1.06), color=color)
            if name != "start_finish":
                _box(road, f"{name}_sign_{side}", (x, post_y, 1.18), (0.08, 0.60, 0.64), color=color)

    speed_markers = _Mesh()
    for offset in (-ROAD_HALF_WIDTH + 0.075, ROAD_HALF_WIDTH - 0.075):
        _strip(speed_markers, offset - 0.018, offset + 0.018, 0.008, SPEED_START_S, SPEED_END_S)
    _shape(road, "speed_zone_edges", "mesh", (0,) * 6,
           uri=speed_markers.write(assets / "speed_zone_edges.obj", flat_up=True), color=(0.02, 0.68, 0.85, 1))


def _bricks(world: ET.Element, assets: Path, rng: random.Random) -> None:
    link = _element(_model(world, "brick_rough_section"), "link", name="pavers")
    x0, y0, _ = centreline(BRICK_START_S)
    x1, _, _ = centreline(BRICK_END_S)
    size = (x1 - x0, 2 * ROAD_HALF_WIDTH, 0.006)
    _box(link, "mortar_visual", ((x0 + x1) / 2, y0, 0.002), size, color=(0.30, 0.29, 0.28, 1))
    rows, columns = 42, 7
    brick_length, brick_width = size[0] / rows, size[1] / columns
    palette = ((0.43, 0.27, 0.22, 1), (0.52, 0.32, 0.25, 1), (0.56, 0.36, 0.29, 1),
               (0.48, 0.34, 0.30, 1), (0.39, 0.29, 0.27, 1))
    pavers = [_Mesh() for _ in palette]
    brick_paint = _Mesh()
    for row in range(rows):
        x = x0 + (row + 0.5) * brick_length
        base_height = rng.uniform(0.007, 0.016)
        if row in (8, 19, 34):
            base_height += 0.004
        h0 = base_height + rng.uniform(-0.0015, 0.0015)
        h1 = base_height + rng.uniform(-0.0015, 0.0015)
        if row in (0, rows - 1):
            h0 *= 0.55
            h1 *= 0.55
        # Each lane bears weight on a separate physical ridge so the axle
        # encounters different heights as it crosses the centreline.
        for sign, height in ((-1, h0), (1, h1)):
            _box(link, f"ridge_{row}_{sign}", (x, y0 + sign * ROAD_HALF_WIDTH / 2, height / 2),
                 (brick_length - 0.006, ROAD_HALF_WIDTH, height), friction=0.88)
        stagger = brick_width / 2 if row % 2 else 0.0
        for column in range(-1, columns):
            left = max(-ROAD_HALF_WIDTH, -ROAD_HALF_WIDTH + column * brick_width + stagger) + 0.007
            right = min(ROAD_HALF_WIDTH, -ROAD_HALF_WIDTH + (column + 1) * brick_width + stagger) - 0.007
            if left >= right:
                continue
            near, far = x - brick_length / 2 + 0.007, x + brick_length / 2 - 0.007
            height = max(0.006, h0 if (left + right) / 2 < 0 else h1)
            mesh = rng.choice(pavers)
            mesh.face(((near, y0 + left, height), (far, y0 + left, height),
                       (far, y0 + right, height), (near, y0 + right, height)))
            for offset in (-YELLOW_LINE_OFFSET, YELLOW_LINE_OFFSET):
                paint_left, paint_right = max(left, offset - YELLOW_LINE_WIDTH / 2), min(right, offset + YELLOW_LINE_WIDTH / 2)
                if paint_left < paint_right:
                    z = height + 0.001
                    brick_paint.face(((near, y0 + paint_left, z), (far, y0 + paint_left, z),
                                      (far, y0 + paint_right, z), (near, y0 + paint_right, z)))
    for i, (mesh, color) in enumerate(zip(pavers, palette)):
        _shape(link, f"brick_tops_{i}", "mesh", (0,) * 6,
               uri=mesh.write(assets / f"brick_tops_{i}.obj", flat_up=True), color=color)
    _shape(link, "yellow_boundaries_over_pavers", "mesh", (0,) * 6,
           uri=brick_paint.write(assets / "brick_yellow_paint.obj", flat_up=True), color=(0.91, 0.69, 0.05, 1))


def _cone_shell(path: Path, z0: float, z1: float) -> Path:
    def radius(z: float) -> float:
        ratio = (z - 0.035) / (CONE_HEIGHT - 0.035)
        return 0.139 * (1 - ratio) + 0.028 * ratio

    mesh = _Mesh()
    for i in range(32):
        angle0, angle1 = math.tau * i / 32, math.tau * (i + 1) / 32
        mesh.face(tuple(
            (radius(z) * math.cos(angle), radius(z) * math.sin(angle), z)
            for z, angle in ((z0, angle0), (z0, angle1), (z1, angle1), (z1, angle0))
        ))
        if z1 == CONE_HEIGHT:
            mesh.face(((0, 0, z1), (radius(z1) * math.cos(angle0), radius(z1) * math.sin(angle0), z1),
                       (radius(z1) * math.cos(angle1), radius(z1) * math.sin(angle1), z1)))
    return mesh.write(path)


def _cones(world: ET.Element, assets: Path, positions: Sequence[tuple[float, float]]) -> None:
    colors = ((0.98, 0.27, 0.015, 1), (0.95, 0.91, 0.79, 1), (0.98, 0.22, 0.012, 1),
              (0.96, 0.92, 0.82, 1), (0.98, 0.25, 0.012, 1))
    bounds = (0.035, 0.19, 0.245, 0.315, 0.375, CONE_HEIGHT)
    shells = [_cone_shell(assets / f"cone_{i}.obj", a, b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    for index, (x, y) in enumerate(positions):
        cone = _model(world, f"traffic_cone_{index}", (x, y, 0, 0, 0, 0))
        link = _element(cone, "link", name="shell")
        _box(link, "base", (0, 0, 0.018), (CONE_BASE, CONE_BASE, 0.036))
        _box(link, "rubber_base", (0, 0, 0.018), (CONE_BASE, CONE_BASE, 0.036), color=(0.085, 0.057, 0.034, 1))
        # The square footprint is exact. Cylindrical tiers approximate the tapered shell.
        for j, (z, length, radius) in enumerate(((0.121, 0.17, 0.131), (0.28, 0.15, 0.093), (0.4375, 0.185, 0.056))):
            _shape(link, f"taper_{j}", "cylinder", (0, 0, z, 0, 0, 0), radius=radius, length=length)
        for j, (shell, color) in enumerate(zip(shells, colors)):
            _shape(link, f"reflective_shell_{j}", "mesh", (0,) * 6, uri=shell, color=color)


def _inertial(link: ET.Element, mass: float, diagonal: Sequence[float], pose: Sequence[float] = (0,) * 6) -> None:
    inertial = _element(link, "inertial")
    _element(inertial, "mass", mass)
    _element(inertial, "pose", pose)
    inertia = _element(inertial, "inertia")
    for key, value in zip(("ixx", "iyy", "izz", "ixy", "ixz", "iyz"), (*diagonal, 0, 0, 0)):
        _element(inertia, key, value)


def _joint(
    model: ET.Element,
    name: str,
    parent: str,
    child: str,
    direction: Sequence[float],
    limit: float,
    effort: float,
    velocity: float,
    damping: float = 0.025,
) -> None:
    joint = _element(model, "joint", name=name, type="revolute")
    _element(joint, "parent", parent)
    _element(joint, "child", child)
    axis = _element(joint, "axis")
    _element(axis, "xyz", direction)
    limits = _element(axis, "limit")
    _element(limits, "lower", -limit)
    _element(limits, "upper", limit)
    _element(limits, "effort", effort)
    _element(limits, "velocity", velocity)
    _element(_element(axis, "dynamics"), "damping", damping)


def _contact_sensor(link: ET.Element, collision: str, topic: str) -> None:
    sensor = _element(link, "sensor", name="contact_sensor", type="contact")
    _element(sensor, "topic", topic)
    _element(sensor, "always_on", True)
    _element(sensor, "update_rate", 1 / MIN_STEP)
    contact = _element(sensor, "contact")
    _element(contact, "collision", collision)
    _element(contact, "topic", topic)


def _camera(base: ET.Element, width: int, height: int, hz: float) -> None:
    sensor = _element(base, "sensor", name="rgbd", type="rgbd_camera")
    _element(sensor, "pose", (CAMERA_X, CAMERA_Y, CAMERA_Z, 0, CAMERA_PITCH, 0))
    _element(sensor, "always_on", True)
    _element(sensor, "update_rate", hz)
    _element(sensor, "topic", CAMERA_TOPIC)
    camera = _element(sensor, "camera", name="cart_rgbd")
    _element(camera, "horizontal_fov", CAMERA_HFOV)
    _element(camera, "optical_frame_id", CAMERA_FRAME)
    image = _element(camera, "image")
    _element(image, "width", width)
    _element(image, "height", height)
    _element(image, "format", "R8G8B8")
    clip = _element(camera, "clip")
    _element(clip, "near", 0.07)
    _element(clip, "far", 75)
    depth_clip = _element(_element(camera, "depth_camera"), "clip")
    _element(depth_clip, "near", CAMERA_DEPTH_NEAR)
    _element(depth_clip, "far", CAMERA_DEPTH_FAR)


def _vehicle(
    world: ET.Element,
    assets: Path,
    spawn_offset: float,
    spawn_yaw_error: float,
    render: bool,
    width: int,
    height: int,
    hz: float,
) -> None:
    x, y, yaw = centreline(SPAWN_S, spawn_offset)
    model = _model(world, MODEL_NAME, (x, y, 0.012, 0, 0, yaw + spawn_yaw_error), static=False)
    model.set("canonical_link", "base_link")
    base = _element(model, "link", name="base_link")
    _inertial(base, 21.0, (0.91, 1.64, 1.96), (0.26, 0, 0.29, 0, 0, 0))
    _box(base, "body_collision", (0.29, 0, 0.279), (0.658, 0.37, 0.17))
    def visual_mesh(filename):
        path = assets / filename
        shutil.copyfile(VEHICLE_ASSETS / filename, path)
        return path

    for name, color in (
        ("frame", (0.23, 0.38, 0.48, 1)), ("deck", (0.60, 0.64, 0.67, 1)),
        ("battery", (0.10, 0.12, 0.15, 1)), ("computer", (0.16, 0.26, 0.18, 1)),
    ):
        _shape(base, f"cad_{name}_visual", "mesh", (0,) * 6,
               uri=visual_mesh(f"cad-{name}.stl"), color=color)
    _box(base, "camera_mount", (CAMERA_X - 0.045, 0, 0.497), (0.09, 0.09, 0.085), color=(0.13, 0.14, 0.15, 1))
    _box(base, "camera_housing", (CAMERA_X - 0.013, 0, CAMERA_Z - 0.006), (0.06, 0.14, 0.055),
         color=(0.11, 0.11, 0.13, 1))
    imu = _element(base, "sensor", name="imu", type="imu")
    _element(imu, "always_on", True)
    _element(imu, "update_rate", 125)
    _element(imu, "topic", IMU_TOPIC)
    _contact_sensor(base, "body_collision", CONTACT_TOPIC)
    if render:
        _camera(base, width, height, hz)

    tire_visual = visual_mesh("cad-wheel.stl")
    for axle in ("rear", "front"):
        for side, sign in (("left", 1), ("right", -1)):
            origin = (WHEELBASE if axle == "front" else 0, sign * WHEEL_TRACK / 2, WHEEL_RADIUS)
            parent = "base_link"
            if axle == "front":
                parent = f"{side}_knuckle"
                knuckle = _element(model, "link", name=parent)
                _element(knuckle, "pose", (*origin, 0, 0, 0))
                _inertial(knuckle, 0.30, (0.003, 0.003, 0.003))
                _joint(model, f"{side}_steering_joint", "base_link", parent, (0, 0, 1), 0.62, 15.0, 2.2)
            name = f"{axle}_{side}_wheel"
            wheel = _element(model, "link", name=name)
            _element(wheel, "pose", (*origin, 0, 0, 0))
            _inertial(wheel, 0.91, (0.0057, 0.0096, 0.0057))
            _shape(wheel, "tire_collision", "cylinder", (0, 0, 0, math.pi / 2, 0, 0),
                   radius=WHEEL_RADIUS, length=WHEEL_WIDTH, friction=1.15)
            _shape(wheel, "cad_wheel_visual", "mesh", (0, 0, 0, 0, 0, 0 if sign > 0 else math.pi),
                   uri=tire_visual, color=(0.075, 0.080, 0.090, 1))
            _joint(model, f"{name}_joint", parent, name, (0, 1, 0), 1e16,
                   23.0 if axle == "rear" else 0.0, 105.0, damping=0.015 if axle == "rear" else 0.004)
            _contact_sensor(wheel, "tire_collision", f"{CONTACT_TOPIC}/{name}")

    _plugin(
        model, "ackermann-steering", "AckermannSteering",
        left_joint="rear_left_wheel_joint", right_joint="rear_right_wheel_joint",
        left_steering_joint="left_steering_joint", right_steering_joint="right_steering_joint",
        wheel_radius=WHEEL_RADIUS, wheel_base=WHEELBASE, wheel_separation=WHEEL_TRACK, kingpin_width=WHEEL_TRACK,
        steering_limit=MAX_STEER_RAD, steer_p_gain=13.0, min_velocity=-2.0, max_velocity=12.0,
        min_acceleration=-3.6, max_acceleration=2.3, topic=CMD_TOPIC,
        odom_topic="/model/cart/wheel_odometry", tf_topic="/model/cart/wheel_tf",
    )
    _plugin(
        model, "odometry-publisher", "OdometryPublisher",
        dimensions=3, odom_frame="world", robot_base_frame="cart/base_link",
        odom_topic=ODOM_TOPIC, tf_topic="/model/cart/odometry_tf", odom_publish_frequency=1 / MIN_STEP,
    )
    _plugin(
        model, "pose-publisher", "PosePublisher",
        publish_model_pose=True, publish_link_pose=True, use_pose_vector_msg=True, update_frequency=-1,
    )


def write_world(
    path: str | Path,
    *,
    seed: int = 0,
    cones: bool = True,
    layout: str = DEFAULT_LAYOUT,
    cone_count: int | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    camera_hz: float = DEFAULT_CAMERA_HZ,
    render: bool = True,
    spawn_offset: float = LANE_OFFSET,
    spawn_yaw_error: float = 0.0,
    render_engine: str = "ogre2",
    realtime_factor: float = 1.0,
) -> Path:
    """Write a standalone SDF with adjacent assets and return its path."""
    path = Path(path).resolve()
    if width < 1 or height < 1 or not math.isfinite(camera_hz) or not 0 < camera_hz <= 1 / MIN_STEP:
        raise ValueError("invalid RGB-D resolution or frequency")
    if not math.isfinite(realtime_factor) or realtime_factor <= 0:
        raise ValueError("realtime_factor must be positive")
    if not abs(spawn_offset) <= ROAD_HALF_WIDTH - BODY_HALF_WIDTH or not math.isfinite(spawn_yaw_error):
        raise ValueError("invalid spawn pose")
    if render_engine not in ("ogre", "ogre2"):
        raise ValueError("render_engine must be ogre or ogre2")
    positions = cone_positions(seed, layout, cone_count) if cones else ()
    assets = path.parent / f"{path.stem}_assets"
    assets.mkdir(parents=True, exist_ok=True)

    root = ET.Element("sdf", {"version": "1.10"})
    world = _element(root, "world", name=WORLD_NAME)
    _element(world, "gravity", (0, 0, -9.81))
    physics = _element(world, "physics", name="4ms", type="dart")
    _element(physics, "max_step_size", MIN_STEP)
    _element(physics, "real_time_factor", realtime_factor)
    _element(physics, "real_time_update_rate", realtime_factor / MIN_STEP)
    for filename, class_name in (
        ("physics", "Physics"), ("user-commands", "UserCommands"), ("scene-broadcaster", "SceneBroadcaster"),
        ("imu", "Imu"), ("contact", "Contact"),
    ):
        _plugin(world, filename, class_name)
    if render:
        _plugin(world, "sensors", "Sensors", render_engine=render_engine)

    scene = _element(world, "scene")
    _element(scene, "ambient", (0.55, 0.55, 0.55, 1))
    _element(scene, "background", (0.60, 0.72, 0.82, 1))
    _element(scene, "grid", False)
    _element(scene, "shadows", True)
    rng = random.Random(seed)
    sun = _element(world, "light", name="sun", type="directional")
    _element(sun, "pose", (0, 0, 45, 0, 0, 0))
    _element(sun, "cast_shadows", True)
    illumination = rng.uniform(0.78, 0.96)
    _element(sun, "diffuse", (illumination, illumination, illumination * 0.97, 1))
    _element(sun, "direction", (-0.38, rng.uniform(-0.55, 0.25), -0.90))
    _road(world, assets)
    _bricks(world, assets, rng)
    if positions:
        _cones(world, assets, positions)
    _vehicle(world, assets, spawn_offset, spawn_yaw_error, render, width, height, camera_hz)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=LAYOUTS, default=DEFAULT_LAYOUT)
    parser.add_argument("--cone-count", choices=(3, 4, 5), type=int)
    parser.add_argument("--no-cones", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--camera-hz", type=float, default=DEFAULT_CAMERA_HZ)
    parser.add_argument("--spawn-offset", type=float, default=LANE_OFFSET)
    parser.add_argument("--spawn-yaw-error", type=float, default=0.0)
    parser.add_argument("--render-engine", choices=("ogre", "ogre2"), default="ogre2")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    args = parser.parse_args()
    print(write_world(
        args.output, seed=args.seed, cones=not args.no_cones, layout=args.layout, cone_count=args.cone_count,
        render=not args.no_render,
        width=args.width, height=args.height, camera_hz=args.camera_hz, spawn_offset=args.spawn_offset,
        spawn_yaw_error=args.spawn_yaw_error, render_engine=args.render_engine, realtime_factor=args.realtime_factor,
    ))


if __name__ == "__main__":
    main()
