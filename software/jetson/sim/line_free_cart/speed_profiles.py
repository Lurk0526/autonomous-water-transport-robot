"""Equal-distance, equal-duration trajectory-shaping comparison, without liquid."""
import math


def compare(distance=5.0, duration=20.0, dt=.02):
    """Both velocity schedules have mean speed D/T and zero end speed.

    The reference is a triangular velocity profile; the alternative uses raised
    cosine velocity. Report both benefits and costs; neither proves reduced spill.
    """
    profiles = []
    for name, label in (("triangle", "三角速度基线"), ("cosine", "余弦平滑速度")):
        rows, x, old_a = [], 0.0, 0.0
        n = round(duration/dt)
        for i in range(n+1):
            t = duration*i/n
            u = t/duration
            if name == "triangle":
                speed = 4*distance/duration*min(u, 1-u)
                accel = (4*distance/duration**2)*(1 if u < .5 else -1) if 0 < u < 1 else 0.0
            else:
                speed = distance/duration*(1-math.cos(2*math.pi*u))
                accel = 2*math.pi*distance/duration**2*math.sin(2*math.pi*u)
            if i:
                x += (rows[-1]["speed_m_s"]+speed)*dt/2
            jerk = (accel-old_a)/dt if i else 0.0
            old_a = accel
            rows.append(dict(t_s=t, x_m=x, speed_m_s=speed, accel_m_s2=accel, jerk_m_s3=jerk))
        profiles.append({"name": name, "label": label, "rows": rows,
                         "distance_m": x, "duration_s": duration,
                         "peak_speed_m_s": max(r["speed_m_s"] for r in rows),
                         "peak_accel_m_s2": max(abs(r["accel_m_s2"]) for r in rows),
                         "peak_sampled_jerk_m_s3": max(abs(r["jerk_m_s3"]) for r in rows)})
    return {"kind": "trajectory_only_not_fluid", "profiles": profiles,
            "same_distance_and_duration": all(abs(p["distance_m"]-distance) < 1e-6 for p in profiles),
            "spill_mass_g": None,
            "note": "三角速度在加速度切换处的连续时间 jerk 不有界；表中是给定采样周期下的差分。余弦速度降低切换冲击但峰值加速度更高，不能直接判定洒水更少。"}
