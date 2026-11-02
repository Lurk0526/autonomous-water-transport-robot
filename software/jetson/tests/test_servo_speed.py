"""Control quality, stopped-target holding, and non-blocking camera ingestion."""
import threading
import time
import unittest
from unittest.mock import MagicMock
import numpy as np

from vision.servo_allocation import allocate_correction
from vision.bowl_follow import FollowPolicy, FollowSettings, BoundedArm
from vision.follow_worker import FollowWorker
from tests.test_bowl_follow import detection, FakeArm, calibrated_fixture
from vision.target_association import select_locked_target
from vision.arm_serial import ArmSerial


class AllocationTests(unittest.TestCase):
    def test_joint_at_limit_leaves_correction_to_available_joint(self):
        matrix = [[1, 1, 0], [0, 0, 1]]
        result = allocate_correction(matrix, [40, 20], [0, -40, -40], [0, 40, 40])
        self.assertEqual(result[0], 0)
        self.assertLess(result[1], -25)
        self.assertLess(result[2], -10)

    def test_zero_response_gripper_and_infeasible_direction_hold(self):
        result = allocate_correction([[1, 0, 0], [0, 1, 0]], [50, 50], [0, 0, -60], [60, 60, 60])
        np.testing.assert_allclose(result, 0, atol=1e-8)

    def test_bounds_respond_to_signed_directional_bias(self):
        config = calibrated_fixture()
        config['feedback_deadband_p'] = {'0': {'positive': 17, 'negative': 16},
                                          '3': {'positive': 25, 'negative': -4}}
        config['max_step_p'] = 65
        arm = BoundedArm(FakeArm(), config, time.monotonic())
        low, high = arm.displacement_bounds()
        self.assertEqual(high, [23, 15])
        self.assertEqual(low, [-24, -40])  # includes cumulative travel bound

    def test_invalid_or_nonfinite_matrix_cannot_produce_commands(self):
        for matrix in ([[float('nan'), 0], [0, 1]], [[1, 0, 0]]):
            with self.assertRaises(ValueError):
                allocate_correction(matrix, [40, 20], [-40, -40], [40, 40])


class TargetAssociationTests(unittest.TestCase):
    def test_low_confidence_extra_detection_does_not_lose_locked_bowl(self):
        record=detection(time.time(),(160,120))
        record['detections']+=detection(time.time(),(240,120),.55)['detections']
        chosen=select_locked_target(record,[160,120],[125,95,195,145],.75)
        self.assertEqual(len(chosen['detections']),1)
        self.assertEqual(chosen['raw_detection_count'],2)

    def test_duplicate_masks_for_same_bowl_are_associated(self):
        record=detection(time.time(),(160,120))
        record['detections']+=detection(time.time(),(162,121),.9)['detections']
        chosen=select_locked_target(record,[160,120],[125,95,195,145],.75)
        self.assertEqual(len(chosen['detections']),1)

    def test_crossing_high_confidence_candidates_remain_ambiguous(self):
        record=detection(time.time(),(146,120))
        record['detections']+=detection(time.time(),(174,120))['detections']
        chosen=select_locked_target(record,[160,120],[125,95,195,145],.75)
        self.assertEqual(len(chosen['detections']),2)


class TrackingTests(unittest.TestCase):
    def settings(self):
        return FollowSettings(jacobian=[[1.2, .03, -.02, 0, 0], [-.08, .9, -.85, .87, 0]],
            joint_axes=(0, 1, 2, 3, 4), bounded_allocation=True, adaptive_gain=True,
            joint_weights=[1, 2, 2, 1, 1], max_step=65,
            deadband_compensation=(0,)*5, stable_frames=2, alpha=.8, period=.1,
            deadband_x=6/160, deadband_y=6/120, hold_hysteresis=1.5)

    def test_stopped_bowl_converges_and_jitter_does_not_restart_motion(self):
        settings = self.settings(); policy = FollowPolicy(settings)
        matrix = np.asarray(settings.jacobian); center = np.array([220., 165.])
        start = time.time(); moves = 0
        for i in range(140):
            t = start+i*.05
            plan = policy.update(detection(t, center), t+.02, i*.05)
            actual = [plan['delta_p'].get(axis, 0) for axis in range(5)]
            center += matrix @ actual; moves += bool(plan['delta_p'])
        self.assertLess(np.linalg.norm(center-[160,120]), 9)
        self.assertGreater(moves, 0)
        for i in range(20):
            t = start+7+i*.05
            plan = policy.update(detection(t, (162+i%2, 121-i%2)), t+.02, 7+i*.05)
            self.assertEqual(plan['state'], 'centered')
            self.assertFalse(plan['delta_p'])
        # Renewed target movement exits the larger hold band immediately.
        t = start+8.05
        plan = policy.update(detection(t, (176,120)), t+.02, 8.05)
        self.assertEqual(plan['state'], 'move')

    def test_current_center_stops_without_waiting_for_filter_tail(self):
        policy = FollowPolicy(self.settings()); start = time.time()
        for i in range(4):
            policy.update(detection(start+i*.05, (190,140)), start+i*.05, i*.05)
        out = policy.update(detection(start+.2, (161,121)), start+.2, .2)
        self.assertEqual(out['state'], 'centered')
        self.assertFalse(out['delta_p'])

    def test_no_available_displacement_reports_limit_instead_of_centered(self):
        settings = self.settings(); settings.displacement_bounds=([0]*5, [0]*5)
        policy=FollowPolicy(settings); start=time.time()
        for i in range(3):
            out=policy.update(detection(start+i*.05),start+i*.05,i*.05)
        self.assertEqual(out['state'], 'limited')
        self.assertFalse(out['delta_p'])

    def test_centered_plan_interrupts_inflight_motion(self):
        fake=FakeArm(); now=time.monotonic(); arm=BoundedArm(fake, calibrated_fixture(),now)
        arm.tick(dict(state='move',delta_p={0:-3},source_stamp_unix=time.time()),now)
        self.assertEqual(arm.tick(dict(state='centered',delta_p={}),now+.05),'centered')
        self.assertEqual(fake.stops,1)

    def test_deliberate_stop_uses_local_envelope_but_rejects_escape(self):
        for measured,valid in [(1515,True),(1555,False)]:
            config=calibrated_fixture();config.update(bounded_stop_feedback=True,max_step_p=40)
            fake=FakeArm();now=time.monotonic();arm=BoundedArm(fake,config,now)
            arm.tick(dict(state='move',delta_p={0:-10},source_stamp_unix=time.time()),now)
            fake.positions[0]=measured
            result=arm.tick(dict(state='no_target',delta_p={}),now+.1)
            self.assertEqual(result=='no_target',valid)

    def test_selective_stop_leaves_loaded_held_joints_untouched(self):
        class Transport(FakeArm):
            def stop_axes(self,axes):self.selected=list(axes)
        fake=Transport();config=calibrated_fixture();config['selective_stop']=True
        now=time.monotonic();arm=BoundedArm(fake,config,now)
        arm.tick(dict(state='move',delta_p={0:-3},source_stamp_unix=time.time()),now)
        arm.tick(dict(state='centered',delta_p={}),now+.1)
        self.assertEqual(fake.selected,[0]);self.assertEqual(fake.stops,0)

    def test_selective_stop_uses_documented_controller_commands(self):
        transport=ArmSerial.__new__(ArmSerial);transport._send=MagicMock()
        transport.stop_axes([3,0])
        self.assertEqual([c.args[0] for c in transport._send.call_args_list],['$DST:0!','$DST:3!'])

    def test_feedback_is_split_and_requires_replanning_afterwards(self):
        fake=FakeArm(); config=calibrated_fixture(); config['fresh_frame_after_feedback']=True
        now=time.monotonic(); arm=BoundedArm(fake,config,now)
        arm.tick(dict(state='move',delta_p={0:-3,3:3},source_stamp_unix=time.time()),now)
        fake.positions.update({0:1497,3:1503})
        waiting=dict(state='rate_limited',delta_p={})
        self.assertEqual(arm.tick(waiting,now+1),'verifying_feedback')
        self.assertEqual(arm.tick(waiting,now+1.02),'feedback_verified')
        self.assertEqual(len(fake.commands),1)
        self.assertIsNone(arm.pending)

    def test_small_unresponsive_motion_learns_bounded_bias(self):
        fake=FakeArm(); config=calibrated_fixture()
        config.update(adaptive_servo_bias=True,fresh_frame_after_feedback=True,
            feedback_deadband_p={'0':dict(positive=0,negative=0),'3':dict(positive=0,negative=0)})
        now=time.monotonic(); arm=BoundedArm(fake,config,now)
        arm.tick(dict(state='move',delta_p={0:3},source_stamp_unix=time.time()),now)
        arm.tick(dict(state='rate_limited',delta_p={}),now+1)
        self.assertEqual(arm.feedback_deadband['0']['positive'],3)

    def test_observed_motion_does_not_inflate_bias(self):
        fake=FakeArm(); config=calibrated_fixture()
        config.update(adaptive_servo_bias=True,fresh_frame_after_feedback=True,
            feedback_deadband_p={'0':dict(positive=0,negative=0),'3':dict(positive=0,negative=0)})
        now=time.monotonic(); arm=BoundedArm(fake,config,now)
        arm.tick(dict(state='move',delta_p={0:3},source_stamp_unix=time.time()),now)
        fake.positions[0]=1503
        arm.tick(dict(state='rate_limited',delta_p={}),now+1)
        self.assertEqual(arm.feedback_deadband['0']['positive'],0)

    def test_stopped_camera_never_reuses_last_motion(self):
        class Transport(FakeArm):
            def close(self):self.stop()
        transport=Transport(); settings=FollowSettings(stable_frames=1)
        worker=FollowWorker(settings,calibrated_fixture(),lambda:transport)
        worker.start()
        try:
            worker.submit(detection(time.time()))
            deadline=time.monotonic()+1.5
            while time.monotonic()<deadline:
                if worker.snapshot().get('state')=='stale':break
                time.sleep(.02)
            self.assertEqual(worker.snapshot()['state'],'stale')
            self.assertEqual(len(transport.commands),1)
            self.assertGreater(transport.stops,0)
        finally:worker.close()

    def test_camera_submit_does_not_wait_for_slow_position_query(self):
        gate=threading.Event(); entered=threading.Event()
        class SlowTransport(FakeArm):
            def position(self, axis):
                entered.set(); gate.wait(2)
                return super().position(axis)
            def close(self):
                self.stop()
        worker=FollowWorker(FollowSettings(),calibrated_fixture(),SlowTransport)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            start=time.monotonic()
            for i in range(100):worker.submit(detection(time.time(),(160+i*.01,120)))
            self.assertLess(time.monotonic()-start,.1)
            self.assertTrue(worker.thread.is_alive())
        finally:
            gate.set(); worker.close()

    def test_operator_stop_remains_latched_in_preview(self):
        worker=FollowWorker(FollowSettings(stable_frames=1)); worker.start()
        try:
            worker.request_stop()
            deadline=time.monotonic()+1
            while time.monotonic()<deadline:
                worker.submit(detection(time.time()))
                if worker.snapshot()['state']=='operator_stopped':break
                time.sleep(.02)
            self.assertEqual(worker.snapshot()['state'],'operator_stopped')
            self.assertFalse(worker.snapshot()['delta_p'])
        finally:worker.close()
