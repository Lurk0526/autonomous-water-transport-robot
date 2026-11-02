"""A self-contained offline playback page; all full-rate data also lives in CSV."""
import json
from pathlib import Path


def write_report(path, summary, cases, speed):
    payload_cases = []
    for case in cases:
        rows = case["rows"]
        # Include transition frames and final frame as well as 5 Hz display samples.
        indices = set(range(0, len(rows), 4)) | {len(rows)-1}
        indices |= {i for i in range(1, len(rows))
                    if rows[i].get("mode") != rows[i-1].get("mode") or
                    rows[i].get("phase") != rows[i-1].get("phase")}
        keys = ("t_s", "x_m", "y_m", "yaw_rad", "speed_m_s", "accel_m_s2", "jerk_m_s3",
                "steer_rad", "cross_track_m", "mode", "phase", "emergency")
        slim_rows = [{k: r[k] for k in keys if k in r} for i, r in enumerate(rows) if i in indices]
        payload_cases.append(dict(case, rows=slim_rows))
    profiles = [dict(p, rows=p["rows"][::4]+[p["rows"][-1]]) for p in speed["profiles"]]
    payload = {"summary": summary, "cases": payload_cases, "speed": dict(speed, profiles=profiles)}
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = (Path(__file__).with_name("report_template.html")).read_text(encoding="utf-8")
    path.write_text(template.replace("__SIMULATION_PAYLOAD__", text), encoding="utf-8")
