#!/usr/bin/env python3
"""Preview image-based bowl following. Serial is opened only with --enable-motion."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.bowl_follow import BoundedArm, FollowPolicy, FollowSettings, validate_motion_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-motion', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/bowl-follow.example.json')
    parser.add_argument('--replay', type=Path, help='Offline JSONL: detection records; serial always disabled')
    parser.add_argument('--output', type=Path, default=ROOT / 'work/bowl-follow/status.json')
    args = parser.parse_args()
    if args.replay and args.enable_motion:
        parser.error('Replay cannot drive hardware')
    settings = FollowSettings()
    config = None
    if args.enable_motion:
        config = json.loads(args.config.read_text())
        try:
            validate_motion_config(config)
        except (ValueError, TypeError, KeyError) as exc:
            parser.error(str(exc))
        settings.x_direction = config['axes']['0']['direction']
        settings.y_direction = config['axes']['3']['direction']
        if config.get('supervised_trial'):
            settings.joint_axes = tuple(config.get('tracking_axes', [0,3]))
            settings.damped_solver = len(settings.joint_axes) > 2
            settings.jacobian = config['local_calibration']['jacobian']
            settings.max_step = config['max_step_p']
            settings.min_step = config.get('min_step_p', 12)
            settings.deadband_compensation = tuple(config.get('deadband_compensation_p', [8, 8]))
            settings.period = config.get('planning_period_s', config['duration_ms']/1000+.1)
            settings.bounded_allocation = config.get('bounded_allocation', False)
            settings.smooth_coordination = config.get('smooth_coordination', False)
            settings.responsive_tracking = config.get('responsive_tracking', False)
            settings.adaptive_gain = settings.bounded_allocation
            settings.joint_weights = config.get('joint_weights')
            settings.gain_near = config.get('gain_near',.4)
            settings.gain_far = config.get('gain_far',.8)
            settings.gain_scale_px = config.get('gain_scale_px',30)
            settings.command_step_limits = config.get('command_step_limits_p')
            settings.negative_step_limits = config.get('negative_command_step_limits_p')
            settings.handoff_weights = config.get('handoff_weights') if config.get('relax_priority_at_limits') else None
            settings.hold_hysteresis = config.get('hold_hysteresis', 1.0)
            settings.stable_frames = 2
            settings.alpha = config.get('filter_alpha', .65)
            settings.confidence = config.get('tracking_confidence', .85)
            settings.locked_confidence = config.get('locked_tracking_confidence')
            settings.confidence_grace_s = config.get('confidence_grace_s',.35)
            settings.allow_clipped = config.get('allow_clipped_tracking', False)
            settings.deadband_x = config.get('center_tolerance_px', 4)/160
            settings.deadband_y = config.get('center_tolerance_px', 4)/120
            settings.target_pixel = None
            settings.position_gain = .65
    policy = FollowPolicy(settings)
    if args.replay:
        count = 0
        for line in args.replay.read_text().splitlines():
            record = json.loads(line)
            stamp = record['stamp']['sec'] + record['stamp']['nanosec'] / 1e9
            plan = policy.update(record, stamp + 0.05, stamp)
            print(json.dumps(dict(plan, motion_enabled=False), ensure_ascii=False))
            count += 1
        print(f'Replayed {count} frames; no serial port opened', file=sys.stderr)
        return

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    from vision.follow_worker import FollowWorker

    class FollowNode(Node):
        def __init__(self):
            super().__init__('target_bowl_follow')
            factory = None
            if args.enable_motion:
                from vision.arm_serial import ArmSerial
                factory = lambda: ArmSerial(config['port'])
            self.worker = FollowWorker(settings, config, factory)
            self.sub = self.create_subscription(String, '/openmv/bowl/detections', self.receive, 1)
            self.pub = self.create_publisher(String, '/openmv/bowl/follow_status', 1)
            self.stop_sub = self.create_subscription(String, '/openmv/bowl/stop_request',
                                                     lambda msg: self.worker.request_stop(), 1)
            self.timer = self.create_timer(.05, self.tick)
            self.last_log = None
            self.last_write = 0
            args.output.parent.mkdir(parents=True, exist_ok=True)
            self.worker.start()
            self.get_logger().info('Control worker started; camera and serial use separate threads')

        def receive(self, msg):
            try:
                record = json.loads(msg.data)
                if not isinstance(record, dict):
                    raise ValueError('Detection record must be an object')
                self.worker.submit(record)
            except (ValueError, TypeError):
                self.worker.submit({'valid': False, 'state': 'invalid_message'})

        def tick(self):
            plan = self.worker.snapshot()
            state = (plan['state'], plan.get('actuator_state'))
            if state != self.last_log:
                self.get_logger().info(json.dumps(plan, ensure_ascii=False))
                self.last_log = state
            if time.monotonic()-self.last_write >= .1:
                value = String(); value.data = json.dumps(plan, ensure_ascii=False)
                self.pub.publish(value)
                tmp = args.output.with_suffix('.tmp')
                tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
                tmp.replace(args.output)
                self.last_write = time.monotonic()

        def close(self):
            self.worker.close()
            status = self.worker.snapshot()
            status.update(state='stopped', motion_enabled=False, updated_unix=time.time())
            args.output.write_text(json.dumps(status, ensure_ascii=False, indent=2))

    rclpy.init()
    node = None
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        node = FollowNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
