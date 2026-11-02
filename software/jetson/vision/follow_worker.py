"""Serial-owned control thread: camera callbacks never wait for arm feedback."""
import copy
import math
import threading
import time

from vision.bowl_follow import BoundedArm, FollowPolicy
from vision.target_association import select_locked_target,VisibleTargetReacquisition


class FollowWorker:
    def __init__(self, settings, config=None, transport_factory=None):
        self.settings, self.config = settings, config
        self.transport_factory = transport_factory
        self.lock = threading.Lock()
        self.quit = threading.Event()
        self.operator_stop = threading.Event()
        self.latest = None
        self.status = {'state': 'waiting_for_camera', 'motion_enabled': False}
        self.thread = threading.Thread(target=self.run, name='bowl-servo', daemon=True)

    def submit(self, record):
        with self.lock:
            self.latest = record

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.status)

    def request_stop(self):
        self.operator_stop.set()

    def start(self):
        self.thread.start()

    def close(self):
        self.quit.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Arm control thread did not close')

    def run(self):
        arm = transport = None
        policy = FollowPolicy(self.settings)
        config = self.config or {}
        initial = config.get('operator_target_lock', {})
        target_center = initial.get('center')
        target_bbox = initial.get('bbox')
        reacquirer=VisibleTargetReacquisition() if config.get('reacquire_visible_target') else None
        reacquisition_count=0
        plan = {'state': 'waiting_for_camera', 'delta_p': {}}
        command_count = 0
        passive_index = 0
        next_passive = 0
        try:
            if self.transport_factory:
                transport = self.transport_factory()
                arm = BoundedArm(transport, config, time.monotonic())
            while not self.quit.is_set():
                with self.lock:
                    record = self.latest
                now, mono = time.time(), time.monotonic()
                if arm and self.settings.bounded_allocation:
                    self.settings.displacement_bounds = arm.displacement_bounds()
                    offsets = arm.feedback_deadband or {str(a):dict(positive=0,negative=0) for a in arm.axes}
                    self.settings.deadband_compensation = tuple(
                        (offsets[str(a)]['positive'],offsets[str(a)]['negative'])
                        for a in arm.axes)
                if record is not None:
                    raw_record=record
                    record=select_locked_target(raw_record,target_center,target_bbox,self.settings.confidence)
                    if reacquirer and target_center is not None:
                        try:
                            item=record['detections'][0];bbox=item['bbox_xyxy_px'];reference=initial['bbox']
                            ratio=((bbox[2]-bbox[0])*(bbox[3]-bbox[1]) /
                                   ((reference[2]-reference[0])*(reference[3]-reference[1])))
                            associated=(len(record['detections'])==1
                                        and math.dist(item['mask_centroid_px'],target_center)<=80
                                        and .35<=ratio<=3)
                        except (KeyError,IndexError,TypeError,ValueError,ZeroDivisionError):associated=False
                        if associated:reacquirer.reset()
                        else:
                            recovered=reacquirer.consider(raw_record,initial.get('weights_sha256'),now)
                            if recovered:
                                record=recovered;item=record['detections'][0]
                                target_center=list(item['mask_centroid_px']);target_bbox=list(item['bbox_xyxy_px'])
                                initial=dict(initial,bbox=list(target_bbox))
                                policy.reject('waiting_for_target');reacquirer.reset()
                                reacquisition_count+=1
                    candidate = policy.update(record, now, mono)
                    if candidate['state'] != 'duplicate':
                        plan = candidate
                        if target_center is not None and plan.get('center_px'):
                            item = record['detections'][0]
                            bbox = item['bbox_xyxy_px']
                            reference = initial['bbox']
                            ratio = ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]) /
                                     ((reference[2]-reference[0])*(reference[3]-reference[1])))
                            if (record.get('weights_sha256') != initial.get('weights_sha256')
                                    or math.dist(plan['center_px'], target_center) > 80
                                    or not .35 <= ratio <= 3):
                                plan = policy.reject('outside_locked_target')
                            else:
                                target_center = plan['center_px']
                                target_bbox = bbox
                if self.operator_stop.is_set():
                    plan = policy.reject('operator_stopped')
                elif not 0 <= now-plan.get('source_stamp_unix', 0) <= self.settings.max_age:
                    if plan['state'] not in ('waiting_for_camera', 'no_target', 'outside_locked_target',
                                             'low_confidence', 'ambiguous', 'invalid_coordinates',
                                             'invalid_size', 'clipped_target'):
                        plan = policy.reject('stale')
                if arm:
                    result = arm.tick(plan, mono)
                    command_delta = dict(plan.get('delta_p', {})) if result in ('command_sent','command_retargeted') else {}
                    if result in ('command_sent','command_retargeted'):
                        command_count += 1
                    if result in ('command_sent','command_retargeted','feedback_verified','direction_changed','coordination_changed'):
                        plan = dict(plan, state='rate_limited', delta_p={})
                    # Only query held joints when the robot is stopped. A query
                    # during motion must not delay arrival/loss stop decisions.
                    passive = [axis for axis in range(6) if axis not in arm.axes]
                    if passive and not arm.pending and not arm.fault and mono >= next_passive:
                        axis = passive[passive_index % len(passive)]
                        actual = transport.position(axis)
                        if abs(actual-arm.positions[axis]) > 12:
                            result = arm.fail(f'unexpected_held_joint_motion:{axis}')
                        else:
                            arm.positions[axis] = actual
                        passive_index += 1
                        next_passive = time.monotonic()+2
                    status = dict(plan, actuator_state=result,
                                  joint_positions_p=dict(arm.positions),
                                  tracking_axes=list(arm.axes),
                                  enabled_axes=config.get('enabled_axes', list(arm.axes)),
                                  session_remaining_s=max(0, arm.deadline-mono),
                                  command_count=command_count, command_delta_p=command_delta,
                                  retarget_count=arm.retarget_count,
                                  travel_p=dict(arm.travel),
                                  servo_bias_p=copy.deepcopy(arm.feedback_deadband),
                                  motion_envelopes_p=dict(arm.motion_envelopes),
                                  motion_enabled=not arm.fault and not self.operator_stop.is_set())
                    low,high=arm.displacement_bounds()
                    status['joint_headroom_p']={str(a):dict(negative=round(-lo,1),positive=round(hi,1))
                                                for a,lo,hi in zip(arm.axes,low,high)}
                else:
                    status = dict(plan, motion_enabled=False)
                status.update(updated_unix=time.time(), supervised_trial=bool(config.get('supervised_trial')),
                              target_reacquisitions=reacquisition_count,
                              detection_count=len(record.get('detections',[])) if record else 0,
                              raw_detection_count=record.get('raw_detection_count',len(record.get('detections',[]))) if record else 0,
                              camera_age_ms=(now-plan['source_stamp_unix'])*1000
                              if plan.get('source_stamp_unix') else None)
                with self.lock:
                    self.status = status
                if arm and arm.fault:
                    break
                self.quit.wait(.02)
        except Exception as exc:
            if arm:
                arm.fail(str(exc))
            with self.lock:
                self.status = dict(state='fault', actuator_state=f'fault:{exc}',
                                   motion_enabled=False, updated_unix=time.time())
        finally:
            if transport:
                transport.close()
