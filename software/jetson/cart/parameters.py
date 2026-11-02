"""Auditable parameter replacement, preserving unknown and synthetic provenance."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path


UNITS = {"wheelbase_m": "m", "width_m": "m", "front_overhang_m": "m", "rear_overhang_m": "m",
         "max_steer_rad": "rad", "steer_rate_rad_s": "rad/s", "steer_time_constant_s": "s",
         "max_accel_m_s2": "m/s^2", "max_decel_m_s2": "m/s^2", "emergency_decel_m_s2": "m/s^2",
         "stop_latency_budget_s": "s"}


def apply_measurements(base, register, evidence_root):
    result = deepcopy(base)
    applied, missing, rejected = {}, [], {}
    for key, unit in UNITS.items():
        entry = register.get("parameters", {}).get(key, {})
        value = entry.get("value")
        if value is None:
            missing.append(key)
            continue
        reason = ""
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value==0 and key not in {'front_overhang_m','rear_overhang_m'}):
            reason = "not_positive_finite"
        elif entry.get("unit") != unit:
            reason = "wrong_unit"
        elif key == "max_steer_rad" and value >= math.pi/2:
            reason = "steering_angle_not_physical"
        elif entry.get("status") not in {"measured", "calibrated"}:
            reason = "not_measured_or_calibrated"
        else:
            evidence = entry.get("evidence") or {}
            path = evidence.get("path")
            if not path or not evidence.get("sha256"):
                reason = "missing_evidence_reference"
            else:
                target = Path(path)
                if not target.is_absolute():
                    target = Path(evidence_root)/target
                if not target.is_file():
                    reason = "evidence_file_missing"
                elif hashlib.sha256(target.read_bytes()).hexdigest() != evidence["sha256"]:
                    reason = "evidence_hash_mismatch"
        if reason:
            rejected[key] = reason
            continue
        group = "navigation" if key == "stop_latency_budget_s" else "vehicle"
        result[group][key] = float(value)
        applied[key] = {"value": float(value), "unit": unit, "status": entry["status"],
                        "evidence": entry["evidence"]}
    complete = not missing and not rejected
    result["measured_vehicle_parameters"] = applied or None
    result["parameter_provenance"] = {
        "kind": "partial_measured_overlay" if applied else "simulation_assumptions_only",
        "applied": applied, "missing": missing, "rejected": rejected,
        "all_required_measurements_present": complete,
        "note": "文件/单位/哈希通过不等于已独立核验实测精度；软件速度与jerk设定仍需整车验收。"}
    return result, {"applied": applied, "missing": missing, "rejected": rejected,
                    "all_required_measurements_present": complete}
