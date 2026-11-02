"""Fail-closed OFFLINE supervisor; emits requests, never hardware commands."""
from dataclasses import dataclass, fields
import math

PHASES = ('APPROACH_PICK', 'PICK_WAIT', 'DESCEND', 'CLOSE', 'LIFT', 'RETRACT',
          'CARRY', 'PREPLACE', 'PLACE_WAIT', 'LOWER', 'OPEN', 'RETREAT', 'DONE')
CONTACT_PHASES = {'DESCEND','CLOSE','LIFT','RETRACT','LOWER','OPEN','RETREAT'}
PAYLOAD_PHASES = {'LIFT','RETRACT','CARRY','PREPLACE','PLACE_WAIT','LOWER'}

@dataclass
class Observation:
    sample_time_s: float = math.nan
    speed: float = math.nan
    angular_speed: float = math.nan
    acceleration: float = math.nan
    telemetry_age: float = math.inf
    path_age: float = math.inf
    evidence_age: float = math.inf
    target_age: float = math.inf
    pose_error_m: float = math.inf
    pose_sigma_m: float = math.inf
    tracking_error_rad: float = math.inf
    tilt_rad: float = math.nan
    slip_m: float = math.inf
    pose_valid: bool = False
    arm_settled: bool = False
    path_clear: bool = False
    phase_done: bool = False
    grasp_verified: bool = False
    support_verified: bool = False
    release_verified: bool = False
    carry_secure: bool = False
    preplace_window: bool = False
    lift_clearance_verified: bool = False
    destination_ready: bool = False
    bowl_resting_verified: bool = False
    base_hold_confirmed: bool = False
    overlap_allowed: bool = False
    braking_clear: bool = False
    kinematics_ok: bool = False
    load_ok: bool = False

class Supervisor:
    def __init__(self, thresholds):
        self.g = thresholds
        if any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in thresholds.values()):
            raise ValueError('gate thresholds must be positive finite values')
        self.phase = PHASES[0]
        self.elapsed = self.stable = 0.0
        self.stable_samples = 0
        self.last_sample = None
        self.hold_reason = ''

    def quiet(self,o):
        return (abs(o.speed)<=self.g['linear_speed_m_s'] and
                abs(o.angular_speed)<=self.g['angular_speed_rad_s'] and
                abs(o.acceleration)<=self.g['acceleration_m_s2'])

    def target_ok(self,o):
        return (o.pose_valid and o.target_age<=self.g['target_age_s'] and
                o.pose_error_m<=self.g['pose_error_m'] and o.pose_sigma_m<=self.g['pose_sigma_m'])

    def overlap_ok(self,o):
        return (o.overlap_allowed and o.braking_clear and o.path_clear and
                abs(o.speed)<=self.g['overlap_speed_m_s'] and
                abs(o.angular_speed)<=self.g['overlap_angular_speed_rad_s'] and
                abs(o.acceleration)<=self.g['overlap_acceleration_m_s2'] and
                abs(o.tilt_rad)<=self.g['overlap_tilt_rad'])

    def update(self,o,dt):
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('dt must be finite and positive')
        if self.phase in {'DONE','HOLD'}:return self.phase
        for f in fields(o):
            val=getattr(o,f.name)
            if f.type is bool:
                if type(val) is not bool:return self.hold('non-boolean evidence')
            elif type(val) not in (int,float) or not math.isfinite(val):
                return self.hold('invalid observation')
        nonnegative=('sample_time_s','telemetry_age','path_age','evidence_age','target_age',
                     'pose_error_m','pose_sigma_m','tracking_error_rad','slip_m')
        if any(getattr(o,n)<0 for n in nonnegative):return self.hold('negative age or error')
        if dt>self.g['max_sample_interval_s']:return self.hold('sampling gap')
        if self.last_sample is not None:
            delta=o.sample_time_s-self.last_sample
            if delta<=0 or abs(delta-dt)>1e-6:return self.hold('timestamp discontinuity')
        self.last_sample=o.sample_time_s
        self.elapsed+=dt
        if self.elapsed>self.g['phase_timeout_s']:return self.hold('phase timeout')
        if o.telemetry_age>self.g['telemetry_age_s']:return self.hold('stale telemetry')
        if o.path_age>self.g['path_age_s'] or not o.path_clear:return self.hold('path unavailable or obstructed')
        if not o.kinematics_ok or not o.load_ok:return self.hold('kinematic or load limit')
        if o.tracking_error_rad>self.g['tracking_error_rad']:return self.hold('tracking error')
        p=self.phase
        if p in CONTACT_PHASES and (not self.quiet(o) or not o.base_hold_confirmed):
            return self.hold('base not held stationary')
        if p in {'DESCEND','CLOSE','LOWER'} and not self.target_ok(o):
            return self.hold('contact target unavailable or uncertain')
        if p=='CLOSE' and (o.carry_secure or o.grasp_verified):
            if o.slip_m>self.g['slip_m'] or abs(o.tilt_rad)>self.g['carrying_tilt_rad']:
                return self.hold('grasp confirmation inconsistent with slip or tilt')
        if p in PAYLOAD_PHASES:
            if not o.carry_secure or o.evidence_age>self.g['evidence_age_s']:
                return self.hold('payload not verified secure')
            if o.slip_m>self.g['slip_m'] or abs(o.tilt_rad)>self.g['carrying_tilt_rad']:
                return self.hold('payload slip or tilt')
        if p in {'OPEN','RETREAT'}:
            if not o.support_verified or not o.bowl_resting_verified or o.evidence_age>self.g['evidence_age_s']:
                return self.hold('support lost or not independently verified')
        if p=='PREPLACE':
            if not self.target_ok(o) or not o.destination_ready:
                return self.hold('preplace target lost')
            if not (self.quiet(o) and o.base_hold_confirmed) and not self.overlap_ok(o):
                return self.hold('moving preplace outside validated envelope')
        ready=(self.quiet(o) and o.base_hold_confirmed and self.target_ok(o) and o.arm_settled)
        fresh=o.evidence_age<=self.g['evidence_age_s']
        if p in {'PICK_WAIT','PLACE_WAIT'}:
            condition=ready and (p!='PLACE_WAIT' or o.destination_ready);dwell=self.g['settle_s']
        elif p=='CLOSE':
            condition=o.phase_done and o.grasp_verified and o.carry_secure and fresh;dwell=self.g['evidence_dwell_s']
        elif p=='LOWER':
            condition=o.phase_done and o.support_verified and o.bowl_resting_verified and o.destination_ready and fresh;dwell=self.g['evidence_dwell_s']
        elif p=='OPEN':
            condition=o.phase_done and o.release_verified and fresh;dwell=self.g['evidence_dwell_s']
        elif p=='RETREAT':
            condition=o.phase_done and o.release_verified and o.arm_settled and fresh;dwell=self.g['evidence_dwell_s']
        else:condition=False;dwell=0.
        if condition:
            # The first sample establishes the interval start; it does not prove a past interval.
            self.stable+=dt if self.stable_samples else 0.
            self.stable_samples+=1
        else:self.stable=0.;self.stable_samples=0
        if dwell:go=self.stable+1e-9>=dwell and self.stable_samples>=self.g['min_evidence_samples']
        elif p=='APPROACH_PICK':go=o.phase_done and ((self.quiet(o) and o.base_hold_confirmed) or self.overlap_ok(o))
        elif p=='LIFT':go=o.phase_done and o.lift_clearance_verified
        elif p=='CARRY':go=o.preplace_window and o.destination_ready and self.target_ok(o) and ((self.quiet(o) and o.base_hold_confirmed) or self.overlap_ok(o))
        else:go=o.phase_done
        if go:
            self.phase=PHASES[PHASES.index(p)+1]
            self.elapsed=self.stable=0.;self.stable_samples=0
        return self.phase

    def requests(self,o=None):
        """Intent contract only: a real controller must enforce these requests."""
        if self.phase=='HOLD':
            return {'base':'controlled_stop','arm':'load_preserving_recovery',
                    'gripper':'retain_current_command','allow_next_phase':False}
        valid=(o is not None and all(type(getattr(o,f.name)) is bool if f.type is bool else
               type(getattr(o,f.name)) in (int,float) and math.isfinite(getattr(o,f.name))
               for f in fields(o)))
        if valid:
            valid=all(getattr(o,n)>=0 for n in ('sample_time_s','telemetry_age','path_age',
                       'evidence_age','target_age','pose_error_m','pose_sigma_m','tracking_error_rad','slip_m'))
        prepare=(valid and self.phase in {'APPROACH_PICK','PREPLACE'} and
                 self.overlap_ok(o) and self.target_ok(o) and
                 o.telemetry_age<=self.g['telemetry_age_s'] and o.path_age<=self.g['path_age_s'] and
                 o.kinematics_ok and o.load_ok and o.tracking_error_rad<=self.g['tracking_error_rad'] and
                 (self.phase!='PREPLACE' or (o.destination_ready and o.carry_secure and
                  o.evidence_age<=self.g['evidence_age_s'] and o.slip_m<=self.g['slip_m'])))
        return {'base':'hold_stationary' if self.phase in CONTACT_PHASES|{'PICK_WAIT','PLACE_WAIT','DONE'} else 'bounded_motion',
                'arm':'prepare_allowed' if prepare else 'stationary_only_preparation',
                'gripper':'supervised_only','allow_next_phase':self.phase!='DONE'}

    def hold(self,reason):
        self.phase,self.hold_reason='HOLD',reason
        return self.phase


def edge_rise_proxy(radius,acceleration_xy,tilt_rad,slosh_amplitude=0.0):
    """Legacy bound: slosh_amplitude must be an independent RESIDUAL bound.

    Do not add an absolute forced slosh response; use screening.py instead.
    No spill-mass inference is possible from this small-tilt bound.
    """
    if not all(math.isfinite(x) for x in (radius,acceleration_xy,tilt_rad,slosh_amplitude)):
        raise ValueError('non-finite input')
    if radius<=0 or abs(tilt_rad)>=math.pi/4:raise ValueError('proxy outside small-tilt domain')
    return radius*(abs(acceleration_xy)/9.81+abs(math.tan(tilt_rad)))+abs(slosh_amplitude)
