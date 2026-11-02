"""Image-based pointing controller; no conversion from pixels to millimetres."""
from dataclasses import dataclass
import math
import time
import numpy as np


@dataclass
class FollowSettings:
    confidence: float = 0.85
    stable_frames: int = 5
    max_age: float = 0.35
    deadband_x: float = 0.075  # ±12 px at 320 px width
    deadband_y: float = 1 / 12  # ±10 px at 240 px height
    alpha: float = 0.35
    max_step: int = 3
    period: float = 0.5
    x_direction: int = -1  # Factory example; requires physical calibration.
    y_direction: int = -1
    jacobian: object = None
    position_gain: float = 0.5
    min_step: int = 1
    deadband_compensation: tuple = (0, 0)
    target_pixel: object = None
    joint_axes: tuple = (0, 3)
    damped_solver: bool = False
    allow_clipped: bool = False
    bounded_allocation: bool = False
    displacement_bounds: object = None
    joint_weights: object = None
    adaptive_gain: bool = False
    hold_hysteresis: float = 1.0
    gain_near: float = .4
    gain_far: float = .8
    gain_scale_px: float = 30
    command_step_limits: object = None
    handoff_weights: object = None
    locked_confidence: object = None
    confidence_grace_s: float = .35
    smooth_coordination: bool = False
    negative_step_limits: object = None
    responsive_tracking: bool = False


class FollowPolicy:
    def __init__(self, settings=None):
        self.settings = settings or FollowSettings()
        self.reset()
        self.last_stamp = None
        self.next_step = 0

    def reset(self):
        self.streak = 0
        self.filtered = None
        self.previous = None
        self.holding = False
        self.trusted_target = None
        self.previous_demand = None

    def reject(self, reason):
        self.reset()
        return {'state': reason, 'delta_p': {}, 'stable_frames': 0}

    def update(self, record, now, monotonic):
        s = self.settings
        try:
            if record.get('valid') is not True or record.get('state') != 'live':
                return self.reject('no_target')
            stamp = record['stamp']['sec'] + record['stamp']['nanosec'] / 1e9
            if not math.isfinite(stamp) or not 0 <= now - stamp <= s.max_age:
                return self.reject('stale')
            width, height = float(record['width']), float(record['height'])
            targets = record['detections']
            if len(targets) != 1:
                return self.reject('ambiguous' if targets else 'no_target')
            target = targets[0]
            cx, cy = map(float, target['mask_centroid_px'])
            x1, y1, x2, y2 = map(float, target['bbox_xyxy_px'])
            confidence, area = float(target['confidence']), float(target['mask_area_px'])
            if not all(map(math.isfinite, [width, height, cx, cy, x1, y1, x2, y2, confidence, area])):
                return self.reject('invalid_coordinates')
            if width <= 0 or height <= 0 or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                return self.reject('invalid_coordinates')
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                return self.reject('invalid_coordinates')
            continuing=False
            if self.trusted_target is not None and s.locked_confidence is not None:
                from vision.target_association import overlap
                trusted=self.trusted_target
                old_area=(trusted['bbox'][2]-trusted['bbox'][0])*(trusted['bbox'][3]-trusted['bbox'][1])
                new_area=(x2-x1)*(y2-y1)
                continuity_distance=60 if s.responsive_tracking and confidence>=s.confidence else 40
                continuing=(0<=stamp-trusted['stamp']<=s.confidence_grace_s
                    and math.dist((cx,cy),trusted['center'])<=continuity_distance
                    and .5<=new_area/old_area<=2
                    and overlap([x1,y1,x2,y2],trusted['bbox'])>=.3)
            threshold=s.locked_confidence if continuing else s.confidence
            if target['class'] != 'target_bowl' or not threshold <= confidence <= 1:
                return self.reject('low_confidence')
            if not 0.02 <= area / (width * height) <= 0.65:
                return self.reject('invalid_size')
            if not s.allow_clipped and min(x1 / width, y1 / height, (width-x2) / width, (height-y2) / height) < 0.015:
                return self.reject('clipped_target')
            aim_x,aim_y = s.target_pixel if s.target_pixel is not None else (width/2,height/2)
            error = ((cx - aim_x) / (width / 2), (cy - aim_y) / (height / 2))
        except (KeyError, ValueError, TypeError, OverflowError, AttributeError):
            return self.reject('invalid_coordinates')
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return {'state': 'duplicate', 'delta_p': {}, 'stable_frames': self.streak}
        gap_limit=.35 if s.responsive_tracking else .2
        gap = self.last_stamp is not None and stamp - self.last_stamp > gap_limit
        self.last_stamp = stamp
        jump = self.previous is not None and math.dist(error, self.previous) > 0.25
        if s.responsive_tracking and continuing:
            jump=False
        if gap or jump:
            self.reset()
        self.previous = error
        self.streak += 1
        if confidence>=s.confidence and self.streak>=s.stable_frames:
            self.trusted_target=dict(stamp=stamp,center=[cx,cy],bbox=[x1,y1,x2,y2])
        self.filtered = error if self.filtered is None else tuple(
            s.alpha * e + (1-s.alpha) * old for e, old in zip(error, self.filtered))
        output = {'state': 'acquiring', 'delta_p': {}, 'stable_frames': self.streak,
                  'center_px': [cx, cy], 'target_px': [aim_x, aim_y],
                  'error_normalized': list(self.filtered), 'confidence': confidence,
                  'source_stamp_unix': stamp,
                  'locked_continuity': bool(continuing and confidence<s.confidence),
                  'coordinate_system': 'image_pixels_only'}
        if self.streak < s.stable_frames:
            return output
        dx, dy = self.filtered
        # Stop against the current observation, avoiding EMA tail after arrival.
        threshold = s.hold_hysteresis if self.holding else 1.0
        if abs(error[0]) <= s.deadband_x*threshold and abs(error[1]) <= s.deadband_y*threshold:
            self.holding = True
            self.filtered = error
            self.previous_demand = None
            output['state'] = 'centered'
            return output
        self.holding = False
        if monotonic < self.next_step:
            output['state'] = 'rate_limited'
            return output
        deltas = {}
        if s.jacobian is not None:
            # Columns are measured image shifts per actual P on joints 000/003.
            error_px = np.array([dx * width/2, dy * height/2])
            matrix = np.asarray(s.jacobian)
            if s.bounded_allocation:
                from vision.servo_allocation import allocate_correction
                low, high = (s.displacement_bounds if s.displacement_bounds is not None
                             else ([-s.max_step]*len(s.joint_axes), [s.max_step]*len(s.joint_axes)))
                gain = (s.gain_near + (s.gain_far-s.gain_near)*(1-math.exp(-np.linalg.norm(error_px)/s.gain_scale_px))
                        if s.adaptive_gain else s.position_gain)
                weights=s.joint_weights
                if s.smooth_coordination and 2 in s.joint_axes and 3 in s.joint_axes:
                    wrist=s.joint_axes.index(3);elbow=s.joint_axes.index(2)
                    weights=list(weights or [1]*len(s.joint_axes))
                    sign=-1 if error_px[1]*matrix[1,wrist]>0 else 1
                    available=(-low[wrist] if sign<0 else high[wrist])
                    bias=s.deadband_compensation[wrist]
                    bias=bias[0 if sign>0 else 1] if isinstance(bias,(list,tuple)) else bias
                    capacity=max(1,s.max_step-bias)
                    blend=max(0,1-min(1,available/capacity))**2
                    elbow_weight=weights[elbow] if s.responsive_tracking else 1.35
                    weights[elbow]=elbow_weight*(1-blend*.35/1.35)
                    weights[wrist]=1+1.5*blend
                    output['coordination_blend']=round(blend,3)
                demand = allocate_correction(matrix, error_px, low, high, gain=gain,
                                             weights=weights)
                residual=np.linalg.norm(matrix@demand+gain*error_px)
                if s.handoff_weights is not None and residual>max(3,.25*gain*np.linalg.norm(error_px)):
                    handoff=allocate_correction(matrix,error_px,low,high,gain=gain,weights=s.handoff_weights)
                    if np.linalg.norm(matrix@handoff+gain*error_px)<residual:
                        demand=handoff;output['limit_handoff']=True
                if s.smooth_coordination:
                    if self.previous_demand is not None:
                        smoothing=.8 if s.responsive_tracking else .65
                        blended=smoothing*demand+(1-smoothing)*self.previous_demand
                        blended=np.clip(blended,low,high)
                        if np.dot(error_px,matrix@blended)<0:
                            demand=blended
                    self.previous_demand=np.asarray(demand).copy()
                output['gain'] = gain
            elif s.damped_solver:
                demand = matrix.T @ np.linalg.solve(matrix @ matrix.T + .02*np.eye(2), -s.position_gain * error_px)
                # Scale the entire vector together so saturation preserves direction.
                peak = max(abs(demand), default=0)
                if peak > s.max_step - 35:
                    demand *= max(1, s.max_step - 35) / peak
            else:
                demand = np.linalg.solve(matrix, -s.position_gain * error_px)
            for index, axis in enumerate(s.joint_axes):
                if abs(demand[index]) < (1 if s.bounded_allocation else 2 if s.damped_solver else 1):
                    continue
                compensation=s.deadband_compensation[index]
                if isinstance(compensation,(list,tuple)):
                    compensation=compensation[0 if demand[index]>0 else 1]
                if s.smooth_coordination and axis in (1,2) and abs(demand[index])<max(1,1-compensation):
                    # A negative static bias makes a tiny command produce a
                    # much larger real move. Leave sub-resolution work to the
                    # other camera joints instead of snapping the loaded arm.
                    continue
                minimum=max(s.min_step,math.ceil(compensation+3))
                limit=min(s.max_step,(s.command_step_limits or {}).get(str(axis),s.max_step))
                if demand[index]<0:
                    limit=min(s.max_step,(s.negative_step_limits or {}).get(str(axis),limit))
                step = min(limit,max(minimum,round(abs(demand[index]) + compensation)))
                if s.bounded_allocation and s.displacement_bounds is not None:
                    signed_actual=(step-compensation)*(1 if demand[index]>0 else -1)
                    if not low[index]-.01<=signed_actual<=high[index]+.01:
                        continue
                deltas[axis] = int(step if demand[index] > 0 else -step)
            if s.bounded_allocation and deltas:
                actual = []
                for index, axis in enumerate(s.joint_axes):
                    command = deltas.get(axis, 0)
                    compensation = s.deadband_compensation[index]
                    if isinstance(compensation, (list, tuple)):
                        compensation = compensation[0 if command > 0 else 1]
                    actual.append(command - math.copysign(1, command)*compensation if command else 0)
                # Rounding and directional bias must still reduce the image error.
                if np.linalg.norm(error_px + matrix @ actual) >= np.linalg.norm(error_px):
                    deltas = {}
                output['predicted_image_shift_px'] = (matrix @ actual).tolist()
        else:
            for axis, e, deadband, direction in [(0, dx, s.deadband_x, s.x_direction),
                                                (3, dy, s.deadband_y, s.y_direction)]:
                if abs(e) > deadband:
                    step = max(1, min(s.max_step, round(abs(e) * 6)))
                    deltas[axis] = direction * (step if e > 0 else -step)
        output.update(state='move' if deltas else 'limited', delta_p=deltas)
        if deltas:
            self.next_step = monotonic + s.period
        return output


def validate_motion_config(config):
    for flag in ('camera_rigid_on_arm', 'directions_and_limits_verified',
                 'serial_open_reset_resolved'):
        if config.get(flag) is not True:
            raise ValueError(f'Motion blocked: {flag} must be verified')
    if config.get('supervised_trial') is True:
        calibration = config.get('local_calibration', {})
        matrix = np.asarray(calibration.get('jacobian'), dtype=float)
        if (matrix.shape != (2, len(config.get('tracking_axes', [0,3]))) or not np.isfinite(matrix).all()
                or np.linalg.cond(matrix) > 10 or calibration.get('result') != 'LOCAL_JACOBIAN_PASS'):
            raise ValueError('Measured local image/servo calibration required for supervised trial')
        from vision.follow_calibration import estimate_jacobian
        measured, _ = estimate_jacobian(calibration.get('trials', []), tuple(config.get('tracking_axes', [0,3])))
        if not np.allclose(matrix, measured, rtol=.01, atol=.001):
            raise ValueError('Trial Jacobian does not match the measured calibration records')
        lock = config.get('operator_target_lock', {})
        center = np.asarray(lock.get('center'), dtype=float)
        if (center.shape != (2,) or not np.isfinite(center).all()
                or not (0 <= center[0] < 320 and 0 <= center[1] < 240)
                or not 0 <= time.time()-lock.get('stamp_unix', 0) <= 60
                or lock.get('weights_sha256') != 'c402769dabe8f46fd69e01f76054622b15677862fd34e1825618fedc4b9073a6'):
            raise ValueError('Fresh operator target lock required for supervised trial')
    elif config.get('perception_validated') is not True:
        raise ValueError('Motion blocked: perception_validated must be verified')
    if set(config.get('pose_reference', {})) != {str(i) for i in range(6)}:
        raise ValueError('Measured pose_reference required for all six joints')
    reference = config['pose_reference']
    if any(type(v) is not int or not 500 <= v <= 2500 for v in reference.values()):
        raise ValueError('Pose references must be measured integer P values')
    tracking_axes = config.get('tracking_axes', [0,3])
    if (not tracking_axes or len(set(tracking_axes)) != len(tracking_axes)
            or not set(tracking_axes) <= set(range(5))):
        raise ValueError('Invalid camera tracking axes')
    for key, default, low, high in [('planning_period_s', .5, .05, 1),
            ('filter_alpha', .65, .1, 1), ('hold_hysteresis', 1, 1, 2),
            ('center_tolerance_px', 4, 3, 12), ('settle_seconds', .1, .02, .5),
            ('tracking_confidence', .85, .5, 1),('retarget_period_s',.4,.18,.6),
            ('retarget_min_error_px',30,8,120),('locked_tracking_confidence',.75,.5,1),
            ('confidence_grace_s',.35,.1,.5)]:
        value=config.get(key,default)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
            raise ValueError(f'Invalid controller setting: {key}')
    if config.get('locked_tracking_confidence',config.get('tracking_confidence',.85))>config.get('tracking_confidence',.85):
        raise ValueError('Locked confidence cannot exceed acquisition confidence')
    for key in ('joint_weights','handoff_weights'):
        weights=np.asarray(config.get(key,[1]*len(tracking_axes)),dtype=float)
        if weights.shape!=(len(tracking_axes),) or not np.isfinite(weights).all() or np.any(weights<=0):
            raise ValueError('Invalid joint weights')
    windows=config.get('position_windows_p',{})
    max_window=500
    if config.get('startup_pose_policy')=='measure_current_pose':
        from vision.motion_limits import pose_envelopes,pose_max_window
        if tracking_axes not in ([0,3],[0,2,3]):
            raise ValueError('Current-pose profile requires measured base/wrist and optional elbow')
        for axis,(low,high) in pose_envelopes(config).items():
            if axis not in tracking_axes:
                continue
            entry=config['axes'][str(axis)]
            if (entry['min_p'],entry['max_p'])!=(low,high):
                raise ValueError('Current-pose profile must match its selected joint ranges')
        max_window=pose_max_window(config)
    if any(str(axis) not in set(map(str,tracking_axes)) or type(value) is not int or not 1<=value<=max_window
           for axis,value in windows.items()):
        raise ValueError('Invalid per-joint position window')
    for axis,value in config.get('command_step_limits_p',{}).items():
        if str(axis) not in set(map(str,tracking_axes)) or type(value) is not int or not 1<=value<=100:
            raise ValueError('Invalid per-joint command limit')
    for key in ('negative_command_step_limits_p','actual_step_limits_p','settling_allowance_p'):
        for axis,value in config.get(key,{}).items():
            maximum=25 if key=='settling_allowance_p' else 100
            if str(axis) not in set(map(str,tracking_axes)) or type(value) is not int or not 1<=value<=maximum:
                raise ValueError(f'Invalid per-joint limit: {key}')
    near,far=config.get('gain_near',.4),config.get('gain_far',.8)
    if not 0<near<=far<=1 or not 1<=config.get('gain_scale_px',30)<=120:
        raise ValueError('Invalid adaptive image gain')
    for axis in map(str, tracking_axes):
        entry = config['axes'][axis]
        low, high = entry['min_p'], entry['max_p']
        if type(low) is not int or type(high) is not int:
            raise ValueError(f'Invalid calibrated bounds/direction for joint {axis}')
        margin=config.get('feedback_tolerance_p',8) if config.get('startup_pose_policy')=='measure_current_pose' else 0
        if type(margin) is not int or not 0<=margin<=12:
            raise ValueError('Invalid bounded actuator setting: feedback_tolerance_p')
        reference_ok=(low-margin<=reference[axis]<=high+margin if margin else low<reference[axis]<high)
        if (type(low) is not int or type(high) is not int or
                not 500<=low<high<=2500 or not reference_ok or entry['direction'] not in (-1, 1)):
            raise ValueError(f'Invalid calibrated bounds/direction for joint {axis}')
    if not isinstance(config.get('port'), str) or not config['port']:
        raise ValueError('Explicit CH340 port required')
    for key, default, low, high in [('max_step_p', 3, 1, 100), ('duration_ms', 400, 400, 1000),
                                   ('session_travel_p', 40, 1, 12000), ('session_window_p', 40, 1, 90),
                                   ('feedback_tolerance_p', 8, 1, 12)]:
        value = config.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'Invalid bounded actuator setting: {key}')
    if config.get('supervised_trial'):
        if not 1 <= config.get('min_step_p', 12) <= config.get('max_step_p', 3):
            raise ValueError('Trial minimum step exceeds the bounded actuator step')
        if config.get('feedback_deadband_p'):
            for axis in map(str, tracking_axes):
                for direction in ('positive','negative'):
                    value=config['feedback_deadband_p'][axis][direction]
                    if type(value) is not int or not -60<=value<=60:
                        raise ValueError('Invalid measured directional servo deadband')


class BoundedArm:
    """No homing/reconnect; absolute moves computed from measured feedback."""
    def __init__(self, transport, config, now):
        validate_motion_config(config)
        self.transport, self.config = transport, config
        self.positions = {}
        self.axes = tuple(config.get('tracking_axes', [0,3]))
        self.travel = {i: 0 for i in self.axes}
        self.pending = None
        self.ready_at = now
        session = config.get('session_seconds', 60)
        if type(session) is not int or not 1 <= session <= 1800:
            raise ValueError('Session duration must be 1 through 1800 seconds')
        self.deadline = now + session
        self.fault = None
        self.max_step = config.get('max_step_p', 3)
        self.duration_ms = config.get('duration_ms', 400)
        self.travel_limit = config.get('session_travel_p', 40)
        self.position_window = config.get('session_window_p', 40)
        self.position_windows = config.get('position_windows_p', {})
        self.feedback_tolerance = config.get('feedback_tolerance_p', 8)
        self.feedback_deadband = config.get('feedback_deadband_p', {})
        self.pending_expected = {}
        self.pending_origin = {}
        self.position_stamps = {}
        self.pending_verified = set()
        self.last_command_at = {}
        self.motion_envelopes = {}
        self.stationary_attempts = {}
        self.last_stop_at = -math.inf
        self.next_retarget_at = now
        self.retarget_count = 0
        # Startup never sends a target pose. All joints must match the calibrated fixture pose.
        try:
            for axis in range(6):
                actual = transport.position(axis)
                reference = config['pose_reference'][str(axis)]
                if abs(actual - reference) > 8:
                    raise ValueError(f'Joint {axis} differs from calibrated startup pose')
                self.positions[axis] = actual
                self.position_stamps[axis] = time.monotonic()
        except Exception:
            transport.stop()
            raise

    def halt(self, reason):
        if self.pending is not None:
            if self.config.get('selective_stop') and hasattr(self.transport,'stop_axes'):
                self.transport.stop_axes(self.pending)
            else:
                self.transport.stop()
            self.last_stop_at = time.monotonic()
            try:
                for axis,target in self.pending.items():
                    actual=self.transport.position(axis)
                    expected = self.pending_expected.get(axis, target)
                    if self.config.get('bounded_stop_feedback'):
                        bounds=self.config['axes'][str(axis)]
                        reference=self.config['pose_reference'][str(axis)]
                        origin=self.pending_origin[axis]
                        valid=(bounds['min_p']-self.feedback_tolerance<=actual<=bounds['max_p']+self.feedback_tolerance
                               and abs(actual-reference)<=self.window(axis)+self.feedback_tolerance
                               and abs(actual-origin)<=self.actual_step_limit(axis)+self.feedback_tolerance)
                    else:
                        valid=min(self.positions[axis],target,expected)-self.feedback_tolerance<=actual<=max(self.positions[axis],target,expected)+self.feedback_tolerance
                    if not valid:
                        return self.fail(f'unexpected_position_after_stop:axis={axis},actual={actual},target={target}')
                    if axis not in self.pending_verified:
                        self.record_travel(axis,actual)
                        self.learn_interrupted_deadband(axis,actual)
                    self.positions[axis]=actual
                    self.position_stamps[axis]=time.monotonic()
            except Exception as exc:
                return self.fail(str(exc))
        self.pending = None
        return reason

    def fail(self, reason):
        self.fault = reason
        try:
            self.transport.stop()
        except Exception as exc:
            self.fault += f'; stop_failed:{exc}'
        self.pending = None
        return 'fault:' + self.fault

    def window(self, axis):
        return self.position_windows.get(str(axis),self.position_window)

    def step_limit(self,axis,direction=0):
        positive=self.config.get('command_step_limits_p',{}).get(str(axis),self.max_step)
        negative=self.config.get('negative_command_step_limits_p',{}).get(str(axis),positive)
        return min(self.max_step,negative if direction<0 else positive if direction>0 else max(positive,negative))

    def actual_step_limit(self,axis):
        return min(self.max_step,self.config.get('actual_step_limits_p',{}).get(str(axis),self.step_limit(axis)))

    def envelope_margin(self,axis):
        return max(self.feedback_tolerance,self.config.get('settling_allowance_p',{}).get(str(axis),0))

    def record_travel(self,axis,actual):
        reserved=abs(self.pending_expected[axis]-self.pending_origin[axis])
        self.travel[axis]=max(0,self.travel[axis]-reserved)+abs(actual-self.pending_origin[axis])

    def learn_interrupted_deadband(self,axis,actual):
        """Accumulate short stopped attempts only after feedback proves no travel."""
        if not (self.config.get('responsive_tracking') and self.config.get('adaptive_servo_bias')
                and str(axis) in self.feedback_deadband):
            return
        origin=self.pending_origin[axis]
        direction=1 if self.pending[axis]>origin else -1
        key=(axis,direction);now=time.monotonic()
        if actual!=origin or direction*(self.pending_expected[axis]-origin)<2:
            self.stationary_attempts.pop(key,None)
            return
        dwell,last=self.stationary_attempts.get(key,(0.,-math.inf))
        if now-last>1.5:dwell=0.
        duration=self.duration_ms/1000
        dwell+=min(duration,max(0,now-self.last_command_at.get(axis,now)))
        if dwell>=duration:
            label='positive' if direction>0 else 'negative'
            self.feedback_deadband[str(axis)][label]=min(60,self.feedback_deadband[str(axis)][label]+3)
            dwell=0.
        self.stationary_attempts[key]=(dwell,now)

    def try_retarget(self,plan,now):
        """Extend a moving primary-axis trajectory using fresh measured progress.

        Never retarget stalled joints, loaded axes, small errors, or stale images.
        Keep a pending trajectory so centering/loss can always interrupt it.
        """
        if not self.config.get('continuous_follow') or now<self.next_retarget_at or plan['state']!='move':
            return None
        axes=set(plan.get('delta_p',{}))
        allowed={0,2,3} if self.config.get('coordinated_retarget') else {0,3}
        if not axes or not axes<=allowed:
            return None
        center,target=plan.get('center_px'),plan.get('target_px')
        if center is None or target is None or math.dist(center,target)<self.config.get('retarget_min_error_px',30):
            return None
        if not 0<=time.time()-plan.get('source_stamp_unix',0)<=.35:
            return None
        if axes!=set(self.pending) and not self.config.get('responsive_retarget'):
            return None
        self.next_retarget_at=now+.15
        if axes!=set(self.pending):
            if self.config.get('responsive_retarget'):
                # Release the old allocation using measured stop feedback, then
                # let a new frame redistribute work instead of waiting its tail.
                return self.halt('coordination_changed')
        if any((self.pending[a]-self.pending_origin[a])*plan['delta_p'][a]<0 for a in axes):
            return self.halt('direction_changed')
        targets={};expected_targets={};positions={};travels={};extended=False
        for axis in sorted(axes):
            actual=self.transport.position(axis)
            bounds=self.config['axes'][str(axis)]
            reference=self.config['pose_reference'][str(axis)]
            if (not bounds['min_p']-self.feedback_tolerance<=actual<=bounds['max_p']+self.feedback_tolerance
                    or abs(actual-reference)>self.window(axis)+self.feedback_tolerance):
                return self.fail(f'retarget_position_out_of_bounds:{axis}')
            direction=1 if self.pending[axis]>self.pending_origin[axis] else -1
            if abs(actual-self.pending_origin[axis])>self.actual_step_limit(axis)+self.feedback_tolerance:
                return self.fail(f'unexpected_retarget_motion:{axis}')
            self.positions[axis]=actual;self.position_stamps[axis]=time.monotonic()
            if direction*(actual-self.pending_origin[axis])<3:
                return None
            delta=plan['delta_p'][axis]
            if type(delta) is not int or not 0<abs(delta)<=self.step_limit(axis,1 if delta>0 else -1):
                return self.fail('invalid_retarget_step')
            goal=max(bounds['min_p'],min(bounds['max_p'],actual+delta))
            offset=self.feedback_deadband.get(str(axis),dict(positive=0,negative=0))
            expected=goal-direction*offset['positive' if direction>0 else 'negative']
            if abs(expected-actual)>self.actual_step_limit(axis):
                return None
            if ((direction*(goal-self.pending[axis])<0 and not self.config.get('coordinated_retarget')) or direction*(expected-actual)<=1
                    or abs(expected-reference)>self.window(axis)):
                return None
            reserved=abs(self.pending_expected[axis]-self.pending_origin[axis])
            travels[axis]=(max(0,self.travel[axis]-reserved)
                           +abs(actual-self.pending_origin[axis])+abs(expected-actual))
            if travels[axis]>self.travel_limit:
                return None
            extended |= (abs(goal-self.pending[axis])>=6 if self.config.get('coordinated_retarget')
                         else direction*(goal-self.pending[axis])>=6)
            targets[axis]=goal;expected_targets[axis]=expected;positions[axis]=actual
        if not extended or not 0<=time.time()-plan.get('source_stamp_unix',0)<=.35:
            return None
        if set(targets)<={0,3}:
            self.transport.move(targets,duration_ms=self.duration_ms)
        else:
            self.transport.move(targets,duration_ms=self.duration_ms,allowed_axes=self.axes)
        self.pending=targets;self.pending_expected=expected_targets;self.pending_origin=positions
        self.pending_verified=set();self.retarget_count+=1
        self.ready_at=time.monotonic()+self.duration_ms/1000+self.config.get('settle_seconds',.1)
        self.next_retarget_at=time.monotonic()+self.config.get('retarget_period_s',.4)
        for axis in axes:
            self.last_command_at[axis]=time.monotonic()
            self.travel[axis]=travels[axis]
            margin=self.envelope_margin(axis)
            self.motion_envelopes[axis]=(min(positions[axis],targets[axis],expected_targets[axis])-margin,
                                         max(positions[axis],targets[axis],expected_targets[axis])+margin)
        return 'command_retargeted'

    def displacement_bounds(self):
        """Actual displacement available after command bias and local limits."""
        lower, upper = [], []
        for axis in self.axes:
            actual = self.positions[axis]
            bounds = self.config['axes'][str(axis)]
            reference = self.config['pose_reference'][str(axis)]
            offset = self.feedback_deadband.get(str(axis), {'positive': 0, 'negative': 0})
            remaining = max(0, self.travel_limit-self.travel[axis])
            low_target = max(bounds['min_p'], reference-self.window(axis))
            high_target = min(bounds['max_p'], reference+self.window(axis))
            lower.append(-max(0, min(self.actual_step_limit(axis),self.step_limit(axis,-1)-offset['negative'],
                actual-bounds['min_p']-offset['negative'], actual-low_target, remaining)))
            upper.append(max(0, min(self.actual_step_limit(axis),self.step_limit(axis,1)-offset['positive'],
                bounds['max_p']-actual-offset['positive'], high_target-actual, remaining)))
        return lower, upper

    def tick(self, plan, now):
        if self.fault:
            return 'fault:' + self.fault
        if now > self.deadline:
            return self.fail('session_timeout')
        state = plan['state']
        if state not in ('move', 'rate_limited'):
            return self.halt(state)
        try:
            if self.pending is not None:
                if now < self.ready_at:
                    result=self.try_retarget(plan,now)
                    if result is not None:return result
                    return 'moving'
                for axis, target in self.pending.items():
                    if axis in self.pending_verified:
                        continue
                    actual = self.transport.position(axis)
                    expected=self.pending_expected.get(axis,target)
                    if abs(actual - expected) > self.feedback_tolerance:
                        origin=self.pending_origin[axis]
                        direction=1 if target>origin else -1
                        bounds=self.config['axes'][str(axis)]
                        reference=self.config['pose_reference'][str(axis)]
                        adaptive_ok=(self.config.get('adaptive_servo_bias')
                            and abs(actual-expected)<=self.config.get('adaptive_bias_tolerance_p',self.feedback_tolerance)
                            and -60<=direction*(target-actual)<=60
                            and direction*(actual-origin)>=-self.feedback_tolerance
                            and abs(actual-origin)<=self.actual_step_limit(axis)+self.feedback_tolerance
                            and bounds['min_p']-self.feedback_tolerance<=actual<=bounds['max_p']+self.feedback_tolerance
                            and abs(actual-reference)<=self.window(axis)+self.feedback_tolerance)
                        if not adaptive_ok:
                            return self.fail(f'position_feedback_mismatch:axis={axis},actual={actual},expected={expected},target={target}')
                    bounds = self.config['axes'][str(axis)]
                    if not bounds['min_p']-self.feedback_tolerance <= actual <= bounds['max_p']+self.feedback_tolerance:
                        return self.fail('position_out_of_bounds')
                    if self.config.get('adaptive_servo_bias') and self.feedback_deadband:
                        origin = self.pending_origin[axis]
                        direction = 1 if target > origin else -1
                        label = 'positive' if direction > 0 else 'negative'
                        old = self.feedback_deadband[str(axis)][label]
                        measured_bias = direction*(target-actual)
                        if -60 <= measured_bias <= 60 and direction*(actual-origin) >= -2:
                            learned = round(.5*old+.5*measured_bias)
                            if direction*(expected-origin) >= 2 and direction*(actual-origin) < 1:
                                learned = max(learned, old+3)
                            self.feedback_deadband[str(axis)][label] = min(60,max(-60,learned))
                        self.stationary_attempts.pop((axis,direction),None)
                    self.record_travel(axis,actual)
                    self.positions[axis] = actual
                    self.position_stamps[axis] = time.monotonic()
                    self.pending_verified.add(axis)
                    if (self.config.get('fresh_frame_after_feedback') and
                            len(self.pending_verified) < len(self.pending)):
                        return 'verifying_feedback'
                self.pending = None
                # Let the worker read the newest camera observation after blocking
                # feedback, instead of acting on the pre-query image.
                if self.config.get('fresh_frame_after_feedback'):
                    return 'feedback_verified'
            if state != 'move':
                return 'ready'
            targets = {}
            expected_targets = {}
            for axis, delta in plan['delta_p'].items():
                axis = int(axis)
                if axis not in self.axes or type(delta) is not int or not 0 < abs(delta) <= self.step_limit(axis,1 if delta>0 else -1):
                    return self.fail('invalid_step')
                bounds = self.config['axes'][str(axis)]
                if (self.config.get('fresh_frame_after_feedback') and
                        time.monotonic()-self.position_stamps.get(axis, 0) < .5):
                    actual = self.positions[axis]
                else:
                    actual = self.transport.position(axis)
                    self.position_stamps[axis] = time.monotonic()
                if abs(actual - self.positions[axis]) > 8:
                    recent=max(self.last_stop_at,self.last_command_at.get(axis,-math.inf))
                    reference=self.config['pose_reference'][str(axis)]
                    envelope=self.motion_envelopes.get(axis)
                    known_settling=(self.config.get('bounded_settling_feedback') and envelope is not None
                                    and envelope[0]<=actual<=envelope[1])
                    settling=(self.config.get('allow_settling_drift')
                        and (time.monotonic()-recent<=2 or known_settling)
                        and abs(actual-self.positions[axis])<=self.config.get('adaptive_bias_tolerance_p',8)
                        and bounds['min_p']-self.feedback_tolerance<=actual<=bounds['max_p']+self.feedback_tolerance
                        and abs(actual-reference)<=self.window(axis)+self.feedback_tolerance)
                    if not settling:
                        return self.fail(f'unexpected_joint_motion:axis={axis},actual={actual},previous={self.positions[axis]}')
                    self.positions[axis]=actual
                target = actual + delta
                expected=target
                if self.config.get('supervised_trial') and self.feedback_deadband:
                    target=max(bounds['min_p'],min(bounds['max_p'],target))
                    if target == actual:
                        continue
                    direction=1 if target>actual else -1
                    offsets=self.feedback_deadband[str(axis)]
                    offset=offsets['positive' if direction>0 else 'negative']
                    expected=target-direction*offset
                    if direction*(expected-actual)<=1:
                        continue
                elif not bounds['min_p'] <= target <= bounds['max_p']:
                    return self.fail('soft_limit')
                if abs(expected-actual)>self.actual_step_limit(axis):
                    return self.fail('actual_step_limit')
                if abs(expected - self.config['pose_reference'][str(axis)]) > self.window(axis):
                    return self.fail('session_position_window')
                if self.travel[axis] + abs(expected - actual) > self.travel_limit:
                    return self.fail('session_travel_limit')
                targets[axis] = target
                expected_targets[axis] = expected
            if targets:
                if not 0 <= time.time() - plan.get('source_stamp_unix', 0) <= 0.35:
                    return self.halt('stale_before_send')
                if self.axes == (0,3):
                    self.transport.move(targets, duration_ms=self.duration_ms)
                else:
                    self.transport.move(targets, duration_ms=self.duration_ms, allowed_axes=self.axes)
                self.pending = targets
                for axis in targets:
                    self.last_command_at[axis]=time.monotonic()
                self.pending_origin = {axis:self.positions[axis] for axis in targets}
                for axis in targets:
                    margin=self.envelope_margin(axis)
                    self.motion_envelopes[axis]=(min(self.positions[axis],targets[axis],expected_targets[axis])-margin,
                                                 max(self.positions[axis],targets[axis],expected_targets[axis])+margin)
                self.pending_verified = set()
                self.pending_expected = expected_targets
                self.ready_at = time.monotonic() + self.duration_ms/1000 + self.config.get('settle_seconds', .1)
                self.next_retarget_at=time.monotonic()+self.config.get('retarget_period_s',.4)
                for axis, expected in expected_targets.items():
                    self.travel[axis] += abs(expected - self.positions[axis])
                return 'command_sent'
            return 'ready'
        except Exception as exc:
            return self.fail(str(exc))
