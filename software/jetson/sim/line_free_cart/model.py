"""Planar bicycle + bounded steering + hypothetical longitudinal actuator.

The pose origin is the rear-axle centre. This is NOT a calibrated dynamics model.
"""
from dataclasses import dataclass, replace
import math


def clamp(x, low, high):
    return max(low, min(high, x))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    steer: float = 0.0
    a: float = 0.0


@dataclass(frozen=True)
class Command:
    speed: float = 0.0
    steer: float = 0.0
    emergency: bool = False
    reason: str = "normal"


@dataclass(frozen=True)
class Obstacle:
    name: str
    x: float
    y: float
    radius: float
    start_s: float = 0.0
    end_s: float = math.inf
    kind: str = "obstacle"

    def active(self, t):
        return self.start_s <= t < self.end_s

    def serializable(self):
        result = dict(vars(self))
        if math.isinf(self.end_s):
            result["end_s"] = None
        return result


def footprint_corners(state, vehicle, margin=0.0):
    left = vehicle["width_m"] / 2 + margin
    front = vehicle["wheelbase_m"] + vehicle["front_overhang_m"] + margin
    rear = -vehicle["rear_overhang_m"] - margin
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    return [(state.x + c*x - s*y, state.y + s*x + c*y)
            for x, y in ((rear, -left), (front, -left), (front, left), (rear, left))]


def obstacle_clearance(state, obstacle, vehicle, margin=0.0):
    """Circle-to-oriented-rectangle distance; <=0 means contact/overlap."""
    dx, dy = obstacle.x-state.x, obstacle.y-state.y
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    local_x, local_y = c*dx+s*dy, -s*dx+c*dy
    x0 = -vehicle["rear_overhang_m"]-margin
    x1 = vehicle["wheelbase_m"]+vehicle["front_overhang_m"]+margin
    half_width = vehicle["width_m"]/2+margin
    outside_x = max(x0-local_x, 0.0, local_x-x1)
    outside_y = max(abs(local_y)-half_width, 0.0)
    return math.hypot(outside_x, outside_y)-obstacle.radius


class Vehicle:
    def __init__(self, config, state=None):
        self.cfg = config
        self.state = replace(state) if state else State()

    def advance(self, command, dt):
        if dt <= 0 or not math.isfinite(dt):
            raise ValueError("positive finite timestep required")
        if not all(math.isfinite(v) for v in (command.speed, command.steer)):
            raise ValueError("nonfinite command")
        p, q = self.cfg, self.state
        target = clamp(command.speed, 0.0, p["max_speed_m_s"])
        desired_steer = clamp(command.steer, -p["max_steer_rad"], p["max_steer_rad"])
        steer_change = clamp((desired_steer-q.steer)/p["steer_time_constant_s"],
                             -p["steer_rate_rad_s"], p["steer_rate_rad_s"])*dt
        next_steer = q.steer+steer_change
        if command.emergency:
            next_a = -p["emergency_decel_m_s2"] if q.v > 0 else 0.0
        else:
            omega = p["speed_response_rad_s"]
            jerk = clamp(omega*omega*(target-q.v)-2*omega*q.a,
                         -p["max_jerk_m_s3"], p["max_jerk_m_s3"])
            next_a = clamp(q.a+jerk*dt, -p["max_decel_m_s2"], p["max_accel_m_s2"])
        next_v = max(0.0, q.v+next_a*dt)
        actual_a = (next_v-q.v)/dt
        # Substeps also provide collision samples, rather than checking endpoints only.
        samples = []
        count = max(1, math.ceil(dt/0.01))
        h = dt/count
        old_v, old_steer = q.v, q.steer
        for i in range(count):
            u = (i+0.5)/count
            v = old_v+(next_v-old_v)*u
            steer = old_steer+(next_steer-old_steer)*u
            yaw_rate = v*math.tan(steer)/p["wheelbase_m"]
            q.x += v*math.cos(q.yaw+yaw_rate*h/2)*h
            q.y += v*math.sin(q.yaw+yaw_rate*h/2)*h
            q.yaw = wrap(q.yaw+yaw_rate*h)
            samples.append(replace(q))
        q.v, q.steer, q.a = next_v, next_steer, actual_a
        return samples


class Route:
    def __init__(self, points, half_width=0.90):
        if len(points) < 2 or half_width <= 0:
            raise ValueError("route needs points and a positive corridor width")
        self.points = [tuple(map(float, p)) for p in points]
        if not all(math.isfinite(v) for p in self.points for v in p):
            raise ValueError("nonfinite route")
        self.half_width = half_width
        self.s = [0.0]
        self.headings = []
        for a, b in zip(self.points, self.points[1:]):
            d = math.dist(a, b)
            if d < 1e-8:
                raise ValueError("duplicate route point")
            self.s.append(self.s[-1]+d)
            self.headings.append(math.atan2(b[1]-a[1], b[0]-a[0]))
        self.curvatures = [0.0]*len(self.points)
        for i in range(1, len(self.points)-1):
            ds = (self.s[i+1]-self.s[i-1])/2
            self.curvatures[i] = wrap(self.headings[i]-self.headings[i-1])/ds
        self.curvatures[0] = self.curvatures[1]
        self.curvatures[-1] = self.curvatures[-2]

    @property
    def length(self):
        return self.s[-1]

    def at(self, distance):
        from bisect import bisect_right
        distance = clamp(distance, 0.0, self.length)
        i = min(len(self.points)-2, max(0, bisect_right(self.s, distance)-1))
        u = (distance-self.s[i])/(self.s[i+1]-self.s[i])
        a, b = self.points[i], self.points[i+1]
        return (a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1]), self.headings[i])

    def project(self, x, y, hint=None, backward=1.0, forward=1.5):
        from bisect import bisect_left, bisect_right
        lo, hi = 0, len(self.points)-1
        if hint is not None:
            lo = max(0, bisect_left(self.s, hint-backward)-1)
            hi = min(hi, bisect_right(self.s, hint+forward))
        best = (math.inf, 0.0, 0.0)
        for i in range(lo, hi):
            ax, ay = self.points[i]
            bx, by = self.points[i+1]
            dx, dy = bx-ax, by-ay
            u = clamp(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy), 0.0, 1.0)
            px, py = ax+u*dx, ay+u*dy
            distance = math.hypot(x-px, y-py)
            if distance < best[0]:
                sign = 1.0 if dx*(y-py)-dy*(x-px) >= 0 else -1.0
                best = distance, self.s[i]+u*(self.s[i+1]-self.s[i]), sign*distance
        return best[1], best[2]

    def max_curvature(self, start=0.0, end=math.inf):
        return max((abs(k) for s, k in zip(self.s, self.curvatures)
                    if start <= s <= end), default=0.0)

    def corridor_error(self,x,y,hint=None):
        """Road continues tangentially beyond route endpoints; goals are not walls."""
        s,error=self.project(x,y,hint)
        if s<=1e-8 or s>=self.length-1e-8:
            px,py=self.points[0] if s<=1e-8 else self.points[-1]
            heading=self.headings[0] if s<=1e-8 else self.headings[-1]
            return -(x-px)*math.sin(heading)+(y-py)*math.cos(heading)
        return error
