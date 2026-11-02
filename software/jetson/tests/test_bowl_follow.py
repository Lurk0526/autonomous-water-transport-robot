"""Offline control checks; these tests never open a device."""
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from vision.bowl_follow import FollowPolicy, BoundedArm, validate_motion_config
from vision.arm_serial import ArmSerial


def detection(stamp, center=(220, 120), confidence=0.95):
    sec = int(stamp)
    cx, cy = center
    return {'state': 'live', 'valid': True, 'stamp': {'sec': sec, 'nanosec': round((stamp-sec)*1e9)},
            'width': 320, 'height': 240, 'detections': [
                {'class': 'target_bowl', 'confidence': confidence, 'mask_centroid_px': list(center),
                 'bbox_xyxy_px': [cx-35, cy-25, cx+35, cy+25], 'mask_area_px': 3000}]}


def calibrated_fixture():
    # Test fixture only. These are not physical robot limits.
    return {'camera_rigid_on_arm': True, 'directions_and_limits_verified': True,
            'serial_open_reset_resolved': True, 'perception_validated': True,
            'port': '/test/not-a-real-port', 'pose_reference': {str(i): 1500 for i in range(6)},
            'axes': {'0': {'min_p': 1460, 'max_p': 1540, 'direction': -1},
                     '3': {'min_p': 1460, 'max_p': 1540, 'direction': -1}}}


class FakeArm:
    def __init__(self):
        self.positions = {i: 1500 for i in range(6)}
        self.commands = []
        self.stops = 0

    def position(self, axis):
        return self.positions[axis]

    def move(self, targets, duration_ms):
        self.commands.append((targets, duration_ms))

    def stop(self):
        self.stops += 1


class FollowPolicyTests(unittest.TestCase):
    def sequence(self, center, frames=5, policy=None):
        policy = policy or FollowPolicy()
        start = time.time()
        outputs = [policy.update(detection(start+i*.05, center), start+i*.05+.03, i*.05)
                   for i in range(frames)]
        return policy, outputs

    def test_requires_distinct_stable_frames_and_small_steps(self):
        policy, outputs = self.sequence((220, 170))
        self.assertTrue(all(not p['delta_p'] for p in outputs[:4]))
        self.assertEqual(outputs[-1]['state'], 'move')
        self.assertEqual(set(outputs[-1]['delta_p']), {0, 3})
        self.assertTrue(all(-3 <= p < 0 for p in outputs[-1]['delta_p'].values()))

    def test_center_deadband_holds(self):
        _, outputs = self.sequence((170, 128))
        self.assertEqual(outputs[-1]['state'], 'centered')
        self.assertEqual(outputs[-1]['delta_p'], {})

    def test_duplicate_stale_low_confidence_multiple_and_nan_never_move(self):
        for kind in ('duplicate', 'stale', 'low', 'multiple', 'nan', 'clipped'):
            with self.subTest(kind=kind):
                p = FollowPolicy(); start = time.time()
                for i in range(10):
                    r = detection(start if kind == 'duplicate' else start+i*.05)
                    if kind == 'low': r['detections'][0]['confidence'] = .8
                    if kind == 'multiple': r['detections'] *= 2
                    if kind == 'nan': r['detections'][0]['mask_centroid_px'][0] = float('nan')
                    if kind == 'clipped': r['detections'][0]['bbox_xyxy_px'][0] = 0
                    now = start+i*.05+(2 if kind == 'stale' else .02)
                    output = p.update(r, now, i*.05)
                    self.assertFalse(output['delta_p'])

    def test_lost_or_jump_reacquires_instead_of_reusing_target(self):
        p, _ = self.sequence((220, 120))
        now = time.time()+1
        lost = p.update({'state': 'stale', 'valid': False}, now, 1)
        self.assertFalse(lost['delta_p'])
        output = p.update(detection(now, (100, 120)), now+.02, 1.1)
        self.assertEqual(output['state'], 'acquiring')
        output = p.update(detection(now+.05, (220, 120)), now+.07, 1.2)
        self.assertEqual(output['stable_frames'], 1)

    def test_example_config_cannot_enable_hardware(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'config/bowl-follow.example.json').read_text())
        with self.assertRaises(ValueError):
            validate_motion_config(config)

    def test_simulated_closed_loop_centers_without_axis_runaway(self):
        policy = FollowPolicy(); cx, cy = 230., 165.; start=time.time()
        moves = 0
        for i in range(500):
            t = start+i*.05
            plan = policy.update(detection(t, (cx, cy)), t+.02, i*.05)
            # Simulated calibrated camera Jacobian: lower P moves target toward center.
            cx += plan['delta_p'].get(0, 0)*2
            cy += plan['delta_p'].get(3, 0)*2
            moves += bool(plan['delta_p'])
        self.assertLessEqual(abs(cx-160), 12)
        self.assertLessEqual(abs(cy-120), 10)
        self.assertGreater(moves, 0)
        self.assertEqual(plan['state'], 'centered')


class BoundedArmTests(unittest.TestCase):
    def plan(self, **changes):
        value = {'state': 'move', 'delta_p': {0: -3}, 'source_stamp_unix': time.time()}
        value.update(changes)
        return value

    def test_startup_checks_all_joints_without_homing(self):
        t = FakeArm(); controller = BoundedArm(t, calibrated_fixture(), time.monotonic())
        self.assertEqual(t.commands, [])
        t2 = FakeArm(); t2.positions[2] = 1520
        with self.assertRaises(ValueError): BoundedArm(t2, calibrated_fixture(), time.monotonic())
        self.assertEqual(t2.commands, [])
        self.assertEqual(t2.stops, 1)

    def test_loss_stops_inflight_motion_without_return_to_center(self):
        t = FakeArm(); now=time.monotonic(); c=BoundedArm(t, calibrated_fixture(), now)
        self.assertEqual(c.tick(self.plan(), now), 'command_sent')
        self.assertEqual(c.tick({'state': 'stale', 'delta_p': {}}, now+.1), 'stale')
        self.assertEqual(len(t.commands), 1)
        self.assertEqual(t.stops, 1)

    def test_feedback_mismatch_latches_fault(self):
        t=FakeArm(); now=time.monotonic(); c=BoundedArm(t, calibrated_fixture(), now)
        c.tick(self.plan(), now); t.positions[0]=1480
        self.assertIn('feedback_mismatch', c.tick(self.plan(state='rate_limited',delta_p={}), now+1))
        c.tick(self.plan(), now+2)
        self.assertEqual(len(t.commands), 1)

    def test_soft_limit_or_stale_plan_never_sent(self):
        for kind in ('limit', 'stale', 'invalid_axis', 'large_step', 'travel', 'timeout'):
            with self.subTest(kind=kind):
                t=FakeArm(); now=time.monotonic(); c=BoundedArm(t, calibrated_fixture(), now)
                p=self.plan()
                if kind=='limit': c.config['axes']['0']['min_p']=1499
                if kind=='stale': p['source_stamp_unix']-=1
                if kind=='invalid_axis': p['delta_p']={1:1}
                if kind=='large_step': p['delta_p']={0:10}
                if kind=='travel': c.travel[0]=39
                if kind=='timeout': now+=61
                c.tick(p, now)
                self.assertFalse(t.commands)


class ArmSerialTests(unittest.TestCase):
    def test_openmv_port_is_rejected_before_opening_anything(self):
        port = SimpleNamespace(device='/dev/ttyACM0', vid=0x1209, pid=0xABD1)
        with patch('vision.arm_serial.list_ports.comports', return_value=[port]), \
                patch('vision.arm_serial.serial.Serial') as opening:
            with self.assertRaises(ValueError): ArmSerial('/dev/ttyACM0')
            opening.assert_not_called()

    def test_query_echo_is_not_position_feedback(self):
        from unittest.mock import MagicMock
        t=ArmSerial.__new__(ArmSerial)
        t.serial=MagicMock()
        t.serial.in_waiting=0
        t.serial.write.side_effect=lambda data:len(data)
        t.serial.read.return_value=b'#000PRAD!'
        # Fast-forward the two bounded query attempts; echoed PRAD is never accepted.
        with patch('vision.arm_serial.time.monotonic', side_effect=[0,0,0,1,2,2,2,3]):
            with self.assertRaises(RuntimeError): t.position(0)

    def test_complete_measured_reply_is_accepted_and_movement_is_restricted(self):
        from unittest.mock import MagicMock
        t=ArmSerial.__new__(ArmSerial)
        t.serial=MagicMock(); t.serial.in_waiting=0
        t.serial.write.side_effect=lambda data:len(data)
        t.serial.read.return_value=b'#003PRAD!#003P1497!'
        self.assertEqual(t.position(3),1497)
        with self.assertRaises(ValueError): t.move({1:1500},400)
        with self.assertRaises(ValueError): t.move({0:1500},0)
        t.move({0:1497,3:1503},400)
        self.assertEqual(t.serial.write.call_args.args[0], b'{#000P1497T0400!#003P1503T0400!}')


if __name__ == '__main__':
    unittest.main()
