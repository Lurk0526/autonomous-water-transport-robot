"""Read-only ROS adapters; exports decoded feedback and diagnostics, never motion."""
import argparse
import json
import math
from pathlib import Path
import time

from .feedback import FeedbackMonitor


ROOT = Path(__file__).resolve().parents[1]


def stamp(header):
    return header.stamp.sec+header.stamp.nanosec*1e-9


def main():
    import numpy as np
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu, JointState, PointCloud2, Image
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    from std_msgs.msg import String
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--config", type=Path, default=ROOT/"config/cart-integration.json")
    parser.add_argument("--output", type=Path, default=ROOT/"work/cart_integration/monitor")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 600:
        raise ValueError("seconds must be in 1..600")
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    monitor = FeedbackMonitor(cfg)
    rclpy.init()
    node = rclpy.create_node("cart_input_monitor")
    latest_status = {}
    callback_errors = []
    point_samples = []
    raw_counts,raw_stamps,raw_depth_fractions={},{},[]
    stream = (args.output/"feedback.jsonl").open("w", encoding="utf-8")

    def receive(packet):
        now = node.get_clock().now().nanoseconds*1e-9
        packet.update(source="ros_live", received_s=now)
        accepted = monitor.ingest(packet, now)
        # Invalid values are represented by null in logs, never nonstandard NaN.
        packet = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in packet.items()}
        stream.write(json.dumps(dict(packet, monitor_accepted=accepted), allow_nan=False)+"\n")

    def raw_image(kind,msg):
        raw_counts[kind]=raw_counts.get(kind,0)+1
        raw_stamps[kind]=stamp(msg.header)
        if kind=="depth":
            from vision.road_geometry import depth_metres
            values=depth_metres(msg)
            raw_depth_fractions.append(float((np.isfinite(values)&(values>=.3)&(values<=5)).mean()))

    def guarded(callback):
        def wrapped(msg):
            try:
                callback(msg)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
                callback_errors.append(str(error))
                del callback_errors[:-30]
        return wrapped

    def odometry(msg):
        q = msg.pose.pose.orientation
        values = np.array([q.x, q.y, q.z, q.w])
        valid_q = np.isfinite(values).all() and abs(float(np.linalg.norm(values))-1) < .01
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)) if valid_q else math.nan
        cov = np.array(msg.pose.covariance).reshape(6, 6)[np.ix_([0, 1, 5], [0, 1, 5])]
        known = bool(np.isfinite(cov).all() and np.allclose(cov, cov.T) and
                     np.linalg.eigvalsh(cov).min() > 0)
        receive(dict(kind="odometry", stamp_s=stamp(msg.header), frame=msg.header.frame_id,
                     child_frame=msg.child_frame_id, x_m=msg.pose.pose.position.x,
                     y_m=msg.pose.pose.position.y, yaw_rad=yaw, speed_m_s=msg.twist.twist.linear.x,
                     yaw_rate_rad_s=msg.twist.twist.angular.z,
                     covariance_known=known,
                     pose_sigma_m=math.sqrt(max(0, float(np.linalg.eigvalsh(cov[:2, :2]).max()))) if known else math.nan,
                     yaw_sigma_rad=math.sqrt(cov[2, 2]) if known else math.nan))

    def imu(msg):
        receive(dict(kind="imu", stamp_s=stamp(msg.header), frame=msg.header.frame_id,
                     yaw_rate_rad_s=msg.angular_velocity.z, yaw_rate_variance=msg.angular_velocity_covariance[8]))

    def steering(msg):
        name = cfg["ros"].get("steering_joint_name")
        if cfg["ros"].get("steering_is_measured_feedback") is not True or not name:
            raise ValueError("measured steering joint mapping not configured")
        index = msg.name.index(name)
        receive(dict(kind="steering", stamp_s=stamp(msg.header), measured=True, steer_rad=msg.position[index]))

    def status(msg):
        nonlocal latest_status
        latest_status = json.loads(msg.data)
        receive(dict(kind="perception", stamp_s=latest_status["status_stamp"],
                     valid=latest_status.get("valid") is True,
                     fusion_valid=latest_status.get("fusion_valid") is True,
                     mount_ready=latest_status.get("mount_ready") is True,
                     valid_until_s=latest_status.get("output_valid_until_stamp")))

    def obstacles(msg):
        now = node.get_clock().now().nanoseconds*1e-9
        synchronized = abs(stamp(msg.header)-latest_status.get("source_stamp", -1)) <= .12
        current = (latest_status.get("fusion_valid") is True and synchronized and
                   0 <= now-latest_status.get("status_stamp", -1) <= 1)
        values = read_points_numpy(msg, field_names=["x", "y", "z"], skip_nans=True)
        if len(values):
            step = max(1, math.ceil(len(values)/500))
            point_samples[:] = values[::step].tolist()
        else:
            point_samples.clear()
        receive(dict(kind="obstacles", stamp_s=stamp(msg.header), frame=msg.header.frame_id,
                     valid=current, mount_ready=latest_status.get("mount_ready") is True,
                     valid_until_s=latest_status.get("output_valid_until_stamp"),
                     point_count=len(values), points_xyz_m=point_samples))

    ros = cfg["ros"]
    subscriptions = [
        node.create_subscription(Odometry, ros["odometry_topic"], guarded(odometry), qos_profile_sensor_data),
        node.create_subscription(Imu, ros["imu_topic"], guarded(imu), qos_profile_sensor_data),
        node.create_subscription(String, ros["perception_status_topic"], guarded(status), 1),
        node.create_subscription(PointCloud2, ros["obstacle_points_topic"], guarded(obstacles), qos_profile_sensor_data),
        node.create_subscription(Image,"/camera/color/image_raw",guarded(lambda msg:raw_image("rgb",msg)),qos_profile_sensor_data),
        node.create_subscription(Image,"/camera/depth/image_rect_raw",guarded(lambda msg:raw_image("depth",msg)),qos_profile_sensor_data),
    ]
    if ros["measured_steering_topic"]:
        subscriptions.append(node.create_subscription(JointState, ros["measured_steering_topic"], guarded(steering), qos_profile_sensor_data))
    begin = time.monotonic()
    try:
        while time.monotonic()-begin < args.seconds:
            rclpy.spin_once(node, timeout_sec=.05)
        now = node.get_clock().now().nanoseconds*1e-9
        report = monitor.snapshot(now)
        report.update(duration_s=time.monotonic()-begin, graph_topics=dict(node.get_topic_names_and_types()),
                      publisher_counts={topic: node.count_publishers(topic) for key, topic in ros.items()
                                        if key.endswith("_topic") and isinstance(topic,str)},
                      callback_errors=callback_errors, latest_perception_status=latest_status,
                      obstacle_sample_xyz_m=point_samples, subscriptions_only=True,
                      hardware_commands_sent=False,
                      rear_axle_frame_semantics_confirmed=ros.get("odom_frame_semantics") == "rear_axle_center_confirmed")
        report["camera_streams"]={kind:{"frames":raw_counts.get(kind,0),
            "age_s":now-raw_stamps[kind] if kind in raw_stamps else None,
            "fresh":kind in raw_stamps and -.05<=now-raw_stamps[kind]<=.75} for kind in ("rgb","depth")}
        report["raw_depth_valid_fraction_mean"]=float(np.mean(raw_depth_fractions)) if raw_depth_fractions else None
        (args.output/"health.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps({k: report[k] for k in ["inputs", "all_required_inputs_valid", "faults_latched", "callback_errors", "hardware_commands_sent"]}, ensure_ascii=False, indent=2))
    finally:
        stream.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0 if not callback_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
