"""Pure Pursuit, synthetic delayed sensors, footprint stop monitor and rollout."""
from collections import deque
from dataclasses import dataclass, replace
import math
import random

from .model import Command, Obstacle, Route, State, Vehicle, clamp, footprint_corners, obstacle_clearance, wrap


@dataclass
class Experiment:
    name: str
    label: str
    route: Route
    expected: str = "arrived"
    initial_offset_m: float = 0.12
    initial_yaw_error_rad: float = 0.03
    localization_delay_s: float = 0.05
    obstacle_delay_s: float = 0.05
    command_delay_s: float = 0.05
    position_noise_m: float = 0.0
    yaw_noise_rad: float = 0.0
    seed: int = 7
    obstacles: tuple = ()
    fault: str = ""
    fault_at_s: float = 4.0
    stop_on_block: bool = False
    max_time_s: float = 100.0
    initial_state: State | None = None
    notes: str = ""
    replan: bool = False
    plant_overrides: dict | None = None
    drift_y_m_s: float = 0.0
    reported_sigma_growth_m_s: float = 0.0
    fault_end_s: float = math.inf


class NavigationSimulation:
    def __init__(self, config, experiment):
        self.config, self.case = config, experiment
        self.vcfg, self.ncfg = config["vehicle"], config["navigation"]
        self.dt = self.ncfg["dt_s"]
        x, y, yaw = experiment.route.at(0)
        state = experiment.initial_state or State(
            x-math.sin(yaw)*experiment.initial_offset_m,
            y+math.cos(yaw)*experiment.initial_offset_m,
            yaw+experiment.initial_yaw_error_rad)
        self.plant_cfg = dict(self.vcfg, **(experiment.plant_overrides or {}))
        self.vehicle = Vehicle(self.plant_cfg, state)
        self.reference_route = experiment.route
        self.planning = {"attempted": False, "accepted": False}
        self.planning_signatures = set()
        self.rng = random.Random(experiment.seed)
        self.t = self.progress = 0.0
        self.road_progress = self.truth_road_progress = 0.0
        self.pose_packets, self.obstacle_packets, self.commands = deque(), deque(), deque()
        self.pose = self.observation = None
        self.applied_command = Command()
        self.previous_pose = None
        self.pose_history = deque()
        self.frozen_pose = None
        self.latched_fault = ""
        self.blocked_by = ""
        self.clear_time = self.stopped_time = 0.0
        self.collision = False
        self.left_corridor = False
        self.min_clearance = math.inf
        self.rows, self.events = [], []
        self.last_mode = None
        radius_limit = self.vcfg["wheelbase_m"]/math.tan(self.vcfg["max_steer_rad"])
        self.infeasible = experiment.route.max_curvature() > 1/radius_limit+1e-6
        if self.infeasible:
            self.latched_fault = "infeasible_route"

    def _sample_sensors(self):
        q, c, t = self.vehicle.state, self.case, self.t
        fault_active = c.fault_at_s-1e-9 <= t < c.fault_end_s
        if not (fault_active and c.fault == "localization_dropout"):
            stamp = min(t, c.fault_at_s) if c.fault == "frozen_timestamp" and fault_active else t
            pose = State(q.x+self.rng.gauss(0, c.position_noise_m),
                         q.y+self.rng.gauss(0, c.position_noise_m)+c.drift_y_m_s*t,
                         wrap(q.yaw+self.rng.gauss(0, c.yaw_noise_rad)), q.v, q.steer, q.a)
            if fault_active and c.fault == "future_timestamp":
                stamp += 1.0
            if fault_active and c.fault == "frozen_pose":
                if self.frozen_pose is None:
                    self.frozen_pose = replace(pose)
                pose.x, pose.y, pose.yaw = self.frozen_pose.x, self.frozen_pose.y, self.frozen_pose.yaw
            if fault_active and c.fault == "pose_jump":
                pose.x += 1.5
            if fault_active and c.fault == "nonfinite_pose":
                pose.x = math.nan
            self.pose_packets.append((t+c.localization_delay_s, stamp, pose))
        if not (fault_active and c.fault == "obstacle_dropout"):
            visible = tuple(o for o in c.obstacles if o.active(t) and
                            math.hypot(o.x-q.x, o.y-q.y) <= self.ncfg["sensor_range_m"] and
                            (o.x-q.x)*math.cos(q.yaw)+(o.y-q.y)*math.sin(q.yaw) > -0.6)
            self.obstacle_packets.append((t+c.obstacle_delay_s, t, visible))
        while self.pose_packets and self.pose_packets[0][0] <= t+1e-9:
            _, stamp, pose = self.pose_packets.popleft()
            self.pose = stamp, pose
        while self.obstacle_packets and self.obstacle_packets[0][0] <= t+1e-9:
            _, stamp, obstacles = self.obstacle_packets.popleft()
            self.observation = stamp, obstacles

    def _input_fault(self):
        n, c = self.ncfg, self.case
        if self.pose is None or self.observation is None:
            return "waiting_inputs" if self.t <= n["startup_grace_s"] else "missing_inputs"
        stamp, pose = self.pose
        if not all(math.isfinite(v) for v in vars(pose).values()):
            return "nonfinite_pose"
        if stamp-self.t > .05 or self.observation[0]-self.t > .05:
            return "future_timestamp"
        if self.t-stamp > n["localization_max_age_s"]+1e-9:
            return "stale_localization"
        if self.t-self.observation[0] > n["obstacle_max_age_s"]+1e-9:
            return "stale_obstacles"
        if c.position_noise_m+c.reported_sigma_growth_m_s*self.t > n["max_pose_sigma_m"]:
            return "localization_uncertain"
        if self.previous_pose and stamp > self.previous_pose[0]+1e-9:
            old_stamp, old = self.previous_pose
            gap = stamp-old_stamp
            max_step = self.vcfg["max_speed_m_s"]*gap+6*c.position_noise_m+0.08
            max_angle = (self.vcfg["max_speed_m_s"]*math.tan(self.vcfg["max_steer_rad"])
                         /self.vcfg["wheelbase_m"]*gap+6*c.yaw_noise_rad+0.08)
            if math.hypot(pose.x-old.x, pose.y-old.y) > max_step or abs(wrap(pose.yaw-old.yaw)) > max_angle:
                return "localization_jump"
        if self.previous_pose is None or stamp > self.previous_pose[0]+1e-9:
            self.pose_history.append((stamp, replace(pose)))
            while self.pose_history and stamp-self.pose_history[0][0] > 1.05:
                self.pose_history.popleft()
            if len(self.pose_history) >= 2 and stamp-self.pose_history[0][0] >= .90:
                first_stamp, first = self.pose_history[0]
                predicted = sum((b.v+a.v)/2*(tb-ta) for (ta,a),(tb,b) in zip(self.pose_history,list(self.pose_history)[1:]))
                measured = math.hypot(pose.x-first.x,pose.y-first.y)
                if predicted-measured > .24+6*c.position_noise_m:
                    return "localization_inconsistent"
        self.previous_pose = stamp, replace(pose)
        return ""

    def _obstacle_distance(self, pose, progress):
        """Evaluate the intended path's swept rectangle using ONLY delivered detections.

        This is an idealized local occupancy input, not a rendered vision pipeline.
        The truth-world collision checker is deliberately separate.
        """
        nearest, name = math.inf, ""
        uncertainty = 3*self.case.position_noise_m
        margin = self.ncfg["footprint_margin_m"]+uncertainty
        obstacles = self.observation[1]
        for o in obstacles:
            if obstacle_clearance(pose, o, self.vcfg, margin) <= 0:
                return 0.0, o.name
        if not obstacles:
            return nearest, name
        distance = 0.0
        while distance <= self.ncfg["sensor_range_m"]:
            x, y, yaw = self.reference_route.at(progress+distance)
            candidate = State(x, y, yaw)
            for o in obstacles:
                if obstacle_clearance(candidate, o, self.vcfg, margin) <= 0:
                    return distance, o.name
            if progress+distance >= self.reference_route.length:
                break
            distance += 0.06
        return nearest, name

    def _decide(self, motion_enabled, external_hold):
        self._sample_sensors()
        fault = self._input_fault()
        if fault and fault != "waiting_inputs":
            self.latched_fault = self.latched_fault or fault
        if external_hold:
            self.latched_fault = self.latched_fault or external_hold
        if self.latched_fault:
            return Command(0, self.vehicle.state.steer, True, self.latched_fault), math.inf
        if fault:
            return Command(0, 0, True, "waiting_inputs"), math.inf
        _, pose = self.pose
        route, n, v = self.reference_route, self.ncfg, self.vcfg
        progress, error = route.project(pose.x, pose.y, self.progress)
        self.progress = progress
        self.road_progress = self.case.route.project(pose.x,pose.y,self.road_progress)[0]
        # Account for the estimated vehicle footprint, rather than centre alone.
        margin = n["footprint_margin_m"]+3*self.case.position_noise_m
        if any(abs(self.case.route.corridor_error(x,y,self.road_progress))+margin > self.case.route.half_width
               for x, y in footprint_corners(pose, v)):
            self.latched_fault = "outside_corridor"
            return Command(0, pose.steer, True, self.latched_fault), math.inf
        distance, obstacle_name = self._obstacle_distance(pose, progress)
        signature = tuple((o.name, o.x, o.y, o.radius) for o in self.observation[1])
        if self.case.replan and obstacle_name and signature not in self.planning_signatures:
            from .planning import corridor_detour
            self.planning_signatures.add(signature)
            candidate, self.planning = corridor_detour(
                self.case.route, pose, self.observation[1], v, n, self.road_progress)
            self.planning["attempt_count"] = len(self.planning_signatures)
            if candidate is not None:
                self.reference_route = route = candidate
                progress, error = route.project(pose.x, pose.y)
                self.progress = progress
                distance, obstacle_name = self._obstacle_distance(pose, progress)
        speed_bound = min(v["max_speed_m_s"], pose.v+v["max_accel_m_s2"]*
                          (max(0.0, self.t-self.pose[0])+n["stop_latency_budget_s"]))
        stop_distance = (speed_bound*n["stop_latency_budget_s"]
                         +speed_bound**2/(2*v["emergency_decel_m_s2"])+n["stop_buffer_m"])
        if distance <= stop_distance:
            self.blocked_by = obstacle_name
        if self.blocked_by:
            still_blocked = bool(obstacle_name) and distance < 2.0
            self.clear_time = 0.0 if still_blocked else self.clear_time+self.dt
            if self.clear_time < n["clear_dwell_s"]:
                return Command(0, pose.steer, True, "obstacle_stop"), distance
            self.blocked_by = ""
            self.clear_time = 0.0
        if not motion_enabled:
            return Command(0, pose.steer, False, "stationary_task"), distance
        lookahead = n["lookahead_base_m"]+n["lookahead_speed_gain_s"]*pose.v
        tx, ty, _ = route.at(progress+lookahead)
        dx, dy = tx-pose.x, ty-pose.y
        local_y = -math.sin(pose.yaw)*dx+math.cos(pose.yaw)*dy
        curvature = 2*local_y/max(dx*dx+dy*dy, 0.04)
        steer = clamp(math.atan(v["wheelbase_m"]*curvature), -v["max_steer_rad"], v["max_steer_rad"])
        remaining = max(0.0, route.length-progress)
        max_curvature = max(abs(curvature), route.max_curvature(progress, progress+1.3), 1e-6)
        speed = min(v["max_speed_m_s"], math.sqrt(n["lateral_accel_limit_m_s2"]/max_curvature),
                    math.sqrt(2*v["max_decel_m_s2"]*remaining), 0.20*remaining)
        return Command(speed, steer), distance

    def step(self, motion_enabled=True, external_hold=""):
        old_state = replace(self.vehicle.state)
        command, obstacle_distance = self._decide(motion_enabled, external_hold)
        self.commands.append((self.t+self.case.command_delay_s, command))
        while self.commands and self.commands[0][0] <= self.t+1e-9:
            _, self.applied_command = self.commands.popleft()
        samples = self.vehicle.advance(self.applied_command, self.dt)
        route = self.reference_route
        true_progress, error = route.project(self.vehicle.state.x, self.vehicle.state.y, self.progress)
        self.truth_road_progress = self.case.route.project(self.vehicle.state.x,self.vehicle.state.y,self.truth_road_progress)[0]
        corridor_clearance = math.inf
        for i, sample in enumerate(samples):
            sample_t = self.t+(i+1)*self.dt/len(samples)
            for o in self.case.obstacles:
                if o.active(sample_t):
                    clearance = obstacle_clearance(sample, o, self.plant_cfg)
                    self.min_clearance = min(self.min_clearance, clearance)
                    self.collision |= clearance <= 0
            # This geometric corridor check uses ground truth, independent of noisy localization.
            for x, y in footprint_corners(sample, self.plant_cfg):
                clearance = self.case.route.half_width-abs(self.case.route.corridor_error(x,y,self.truth_road_progress))
                corridor_clearance = min(corridor_clearance, clearance)
            self.left_corridor |= corridor_clearance < -1e-8
        self.t = round(self.t+self.dt, 9)
        q = self.vehicle.state
        mode = self.latched_fault or command.reason
        if mode != self.last_mode:
            self.events.append({"t_s": self.t, "mode": mode})
            self.last_mode = mode
        if mode == "obstacle_stop" and q.v <= self.ncfg["stopped_speed_m_s"]:
            self.stopped_time += self.dt
        else:
            self.stopped_time = 0.0
        row = {
            "t_s": self.t, "x_m": q.x, "y_m": q.y, "yaw_rad": q.yaw,
            "speed_m_s": q.v, "accel_m_s2": q.a,
            "jerk_m_s3": (q.a-old_state.a)/self.dt,
            "steer_rad": q.steer, "steer_rate_rad_s": (q.steer-old_state.steer)/self.dt,
            "lateral_accel_m_s2": q.v*q.v*math.tan(q.steer)/self.plant_cfg["wheelbase_m"],
            "cross_track_m": error, "progress_m": true_progress,
            "corridor_clearance_m": corridor_clearance,
            "obstacle_distance_m": obstacle_distance if math.isfinite(obstacle_distance) else None,
            "command_speed_m_s": command.speed, "command_steer_rad": command.steer,
            "emergency": self.applied_command.emergency,
            "pose_age_s": round(self.t-self.pose[0], 9) if self.pose else None,
            "obstacle_age_s": round(self.t-self.observation[0], 9) if self.observation else None,
            "mode": mode, "collision": self.collision, "left_corridor": self.left_corridor,
        }
        self.rows.append(row)
        return row

    @property
    def arrived(self):
        q = self.vehicle.state
        return (self.reference_route.length-self.reference_route.project(q.x, q.y, self.progress)[0]
                <= self.ncfg["goal_tolerance_m"]
                and math.dist((q.x, q.y), self.case.route.points[-1]) <= self.ncfg["goal_tolerance_m"]
                and q.v <= self.ncfg["stopped_speed_m_s"] and abs(q.a) <= 0.02)

    def run(self):
        outcome = "timeout"
        while self.t < self.case.max_time_s:
            self.step()
            if self.collision or self.left_corridor:
                outcome = "collision" if self.collision else "left_corridor"
                break
            if self.latched_fault and self.vehicle.state.v <= self.ncfg["stopped_speed_m_s"]:
                outcome = self.latched_fault
                break
            if self.case.stop_on_block and self.stopped_time >= 0.60:
                outcome = "obstacle_stop"
                break
            if self.arrived:
                outcome = "arrived"
                break
        return self.result(outcome)

    def result(self, outcome):
        rows, q = self.rows, self.vehicle.state
        normal = [r for i, r in enumerate(rows) if not r["emergency"] and
                  (i == 0 or not rows[i-1]["emergency"])]
        metrics = {
            "duration_s": self.t,
            "rms_cross_track_m": math.sqrt(sum(r["cross_track_m"]**2 for r in rows)/max(1, len(rows))),
            "max_cross_track_m": max((abs(r["cross_track_m"]) for r in rows), default=0),
            "goal_error_m": math.dist((q.x, q.y), self.case.route.points[-1]),
            "max_speed_m_s": max((r["speed_m_s"] for r in rows), default=0),
            "max_normal_accel_m_s2": max((abs(r["accel_m_s2"]) for r in normal), default=0),
            "max_normal_jerk_m_s3": max((abs(r["jerk_m_s3"]) for r in normal), default=0),
            "max_steer_rate_rad_s": max((abs(r["steer_rate_rad_s"]) for r in rows), default=0),
            "max_lateral_accel_m_s2": max((abs(r["lateral_accel_m_s2"]) for r in rows), default=0),
            "min_obstacle_clearance_m": self.min_clearance if math.isfinite(self.min_clearance) else None,
            "min_corridor_clearance_m": min((r["corridor_clearance_m"] for r in rows), default=0),
            "final_speed_m_s": q.v,
        }
        return {
            "name": self.case.name, "label": self.case.label,
            "expected_outcome": self.case.expected, "outcome": outcome,
            "expectation_met": outcome == self.case.expected and not self.collision and not self.left_corridor,
            "task_completed": outcome == "arrived", "collision_free": not self.collision,
            "inside_corridor": not self.left_corridor, "metrics": metrics,
            "seed": self.case.seed, "notes": self.case.notes,
            "simulation_dt_s":self.dt,
            "localization_delay_s": self.case.localization_delay_s,
            "command_delay_s": self.case.command_delay_s,
            "position_noise_m": self.case.position_noise_m,
            "plant_overrides": self.case.plant_overrides,
            "route": self.case.route.points, "half_width_m": self.case.route.half_width,
            "reference_route": self.reference_route.points, "planning": self.planning,
            "obstacles": [o.serializable() for o in self.case.obstacles],
            "events": self.events, "rows": rows,
        }
