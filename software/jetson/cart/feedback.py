"""Timestamp/frame/sequence checks for decoded, explicitly sourced feedback.

No serial protocol is guessed and no commands are sent. JSON Lines is an
integration/log format, not a claim about the existing STM32 firmware protocol.
"""
from collections import Counter, deque
import json
import math


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class FeedbackMonitor:
    def __init__(self, config):
        self.cfg = config
        self.last = {}
        self.valid = {}
        self.counts = Counter()
        self.errors = []
        self.latched = set()
        self.motion_history=deque()

    def reject(self, kind, reason):
        self.valid[kind] = False
        self.counts["rejected"] += 1
        self.latched.add(reason)
        self.errors.append({"kind": kind, "reason": reason})
        self.errors = self.errors[-100:]
        return False

    def ingest(self, packet, now):
        if not isinstance(packet, dict):
            return self.reject("unknown", "not_an_object")
        kind = packet.get("kind", "unknown")
        if kind not in {"odometry", "imu", "steering", "perception", "obstacles"}:
            return self.reject("unknown", "unknown_packet_kind")
        stamp = packet.get("stamp_s")
        self.counts[kind] += 1
        if not finite(now) or not finite(stamp) or stamp < 0:
            return self.reject(kind, "invalid_timestamp")
        age = now-stamp
        maximum = self.cfg["checks"]["perception_max_age_s"] if kind in {"perception", "obstacles"} else self.cfg["checks"]["imu_max_age_s"] if kind=="imu" else self.cfg["checks"]["feedback_max_age_s"]
        if age < -self.cfg["checks"]["max_future_s"] or age > maximum:
            return self.reject(kind, "future_timestamp" if age < 0 else "stale_packet")
        previous = self.last.get(kind)
        if previous and stamp <= previous["stamp_s"]:
            return self.reject(kind, "non_monotonic_timestamp")
        if packet.get("source") not in {"ros_live", "recorded_real", "synthetic_fixture"}:
            return self.reject(kind, "source_not_declared")
        if "sequence" in packet:
            sequence = packet["sequence"]
            if type(sequence) is not int or sequence < 0:
                return self.reject(kind, "invalid_sequence")
            if previous and "sequence" in previous and sequence <= previous["sequence"]:
                return self.reject(kind, "sequence_reset_or_replay")
        ros = self.cfg["ros"]
        if kind == "odometry":
            if packet.get("frame") != ros["odometry_frame"] or packet.get("child_frame") != ros["body_frame"]:
                return self.reject(kind, "odometry_frame_mismatch")
            for key in ("x_m", "y_m", "yaw_rad", "speed_m_s", "yaw_rate_rad_s", "pose_sigma_m", "yaw_sigma_rad"):
                if not finite(packet.get(key)):
                    return self.reject(kind, "nonfinite_or_missing_"+key)
            if packet["pose_sigma_m"] < 0 or packet["yaw_sigma_rad"] < 0:
                return self.reject(kind, "negative_covariance")
            if packet.get("covariance_known") is not True:
                return self.reject(kind, "covariance_unknown")
            if packet["pose_sigma_m"] > self.cfg["checks"]["max_pose_sigma_m"] or packet["yaw_sigma_rad"] > self.cfg["checks"]["max_yaw_sigma_rad"]:
                return self.reject(kind, "localization_uncertain")
            if abs(packet["speed_m_s"]) > self.cfg["checks"]["max_speed_m_s"]:
                return self.reject(kind, "speed_out_of_range")
            if abs(packet["yaw_rate_rad_s"]) > self.cfg["checks"]["max_yaw_rate_rad_s"]:
                return self.reject(kind, "yaw_rate_out_of_range")
            if previous:
                dt = stamp-previous["stamp_s"]
                distance = math.hypot(packet["x_m"]-previous["x_m"], packet["y_m"]-previous["y_m"])
                bound = self.cfg["checks"]["max_speed_m_s"]*dt+self.cfg["checks"]["max_localization_step_slack_m"]
                if distance > bound:
                    return self.reject(kind, "localization_jump")
        elif kind == "imu":
            if packet.get("frame") != ros["body_frame"]:
                return self.reject(kind, "imu_requires_measured_body_transform")
            if not finite(packet.get("yaw_rate_rad_s")) or not finite(packet.get("yaw_rate_variance")) or packet["yaw_rate_variance"] <= 0:
                return self.reject(kind, "imu_measurement_or_covariance_invalid")
        elif kind == "steering":
            if packet.get("measured") is not True or not finite(packet.get("steer_rad")):
                return self.reject(kind, "steering_command_is_not_feedback")
        elif kind == "perception":
            if packet.get("valid") is not True or packet.get("fusion_valid") is not True:
                return self.reject(kind, "perception_invalid")
            if not finite(packet.get("valid_until_s")) or packet["valid_until_s"] < now:
                return self.reject(kind, "perception_expired")
            if packet["valid_until_s"]>stamp+self.cfg["checks"]["perception_max_age_s"]+.001:
                return self.reject(kind,"invalid_validity_horizon")
        elif kind == "obstacles":
            if packet.get("frame") != ros["body_frame"] or packet.get("mount_ready") is not True:
                return self.reject(kind, "obstacles_require_measured_mount")
            if packet.get("valid") is not True:
                return self.reject(kind, "obstacles_invalid_empty_is_not_free")
            if not finite(packet.get("valid_until_s")) or packet["valid_until_s"] < now:
                return self.reject(kind, "obstacles_expired")
            if packet["valid_until_s"]>stamp+self.cfg["checks"]["perception_max_age_s"]+.001:
                return self.reject(kind,"invalid_validity_horizon")
            points=packet.get("points_xyz_m",[])
            if type(packet.get("point_count")) is not int or packet["point_count"]<0 or not isinstance(points,list) or any(not isinstance(p,list) or len(p)!=3 or not all(finite(v) for v in p) for p in points):
                return self.reject(kind,"invalid_obstacle_geometry")
        if kind=="odometry":
            self.motion_history.append(packet)
            while self.motion_history and stamp-self.motion_history[0]["stamp_s"]>1.05:
                self.motion_history.popleft()
            history=list(self.motion_history)
            if len(history)>1 and stamp-history[0]["stamp_s"]>=.9:
                expected=sum((b["speed_m_s"]+a["speed_m_s"])/2*(b["stamp_s"]-a["stamp_s"]) for a,b in zip(history,history[1:]))
                measured=math.hypot(packet["x_m"]-history[0]["x_m"],packet["y_m"]-history[0]["y_m"])
                if abs(expected)-measured>.25+4*packet["pose_sigma_m"]:
                    return self.reject(kind,"odometry_motion_inconsistent")
        self.last[kind] = dict(packet)
        self.valid[kind] = True
        return True

    def snapshot(self, now):
        result = {}
        for kind in ("odometry", "imu", "steering", "perception", "obstacles"):
            packet = self.last.get(kind)
            age = now-packet["stamp_s"] if packet else None
            maximum = self.cfg["checks"]["perception_max_age_s"] if kind in {"perception", "obstacles"} else self.cfg["checks"]["imu_max_age_s"] if kind=="imu" else self.cfg["checks"]["feedback_max_age_s"]
            fresh = bool(packet and -self.cfg["checks"]["max_future_s"] <= age <= maximum and self.valid.get(kind))
            if kind in {"perception", "obstacles"} and packet:
                fresh = fresh and packet.get("valid_until_s", -math.inf) >= now
            result[kind] = {"received": packet is not None, "fresh_and_valid": fresh,
                            "age_s": age, "source": packet.get("source") if packet else None}
        slip = False
        synchronized=bool(all(result[k]["fresh_and_valid"] for k in ("odometry","imu")) and
                          abs(self.last["odometry"]["stamp_s"]-self.last["imu"]["stamp_s"])<=self.cfg["checks"].get("imu_odom_sync_slop_s",.06))
        if synchronized:
            slip = abs(self.last["odometry"]["yaw_rate_rad_s"]-self.last["imu"]["yaw_rate_rad_s"]) > self.cfg["checks"]["slip_yaw_rate_difference_rad_s"]
        return {"inputs": result, "all_required_inputs_valid": all(r["fresh_and_valid"] for r in result.values()) and synchronized and not slip,
                "imu_odom_time_aligned":synchronized,
                "wheel_imu_yaw_disagreement": slip, "faults_latched": sorted(self.latched),
                "counts": dict(self.counts), "errors": self.errors,
                "motion_output_enabled": False,
                "real_chassis_feedback_seen": any(self.last.get(k, {}).get("source") == "ros_live" for k in ("odometry", "imu", "steering")),
                "real_perception_seen": any(self.last.get(k, {}).get("source") == "ros_live" for k in ("perception", "obstacles"))}


def replay_jsonl(path, config):
    monitor = FeedbackMonitor(config)
    now = 0.0
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                packet = json.loads(line)
            except (ValueError, TypeError):
                monitor.reject("unknown", "invalid_json")
                continue
            receipt = packet.get("received_s") if isinstance(packet, dict) else None
            if finite(receipt):
                now = max(now, receipt)
            monitor.ingest(packet, now)
    return monitor.snapshot(now)
