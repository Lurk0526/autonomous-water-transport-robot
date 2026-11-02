"""Tracking tuning for the verified Jibot1 arm/camera installation."""
import copy
import math
import time
from vision.motion_limits import pose_envelopes


def apply_measured_elbow_extension(previous, report, now=None):
    """Only extend to absolute commands actually exercised with image feedback."""
    now=time.time() if now is None else now
    weights=report.get('target',{}).get('weights_sha256')
    if (report.get('result')!='MEASURED_EXTENSION_PASS' or report.get('motion_enabled') is not True
            or not 0<=now-report.get('finished_unix',0)<=1800
            or 2 not in previous.get('tracking_axes',[])
            or not isinstance(weights,str) or len(weights)!=64
            or weights!=previous.get('operator_target_lock',{}).get('weights_sha256')):
        raise ValueError('Fresh, matching measured elbow extension required')
    steps=report.get('steps',[])
    if not steps:
        raise ValueError('No measured elbow extension steps')
    for step in steps:
        before,goal,after=(step[k] for k in ('before_p','target_p','after_p'))
        response=step['image_response_px_per_p']
        first,last=step['before_center'],step['after_center']
        if (any(type(p) is not int for p in (before,goal,after))
                or not 1600<=goal<=1900 or not 20<=before-goal<=65
                or not goal-10<=after<=before+5 or not 5<=before-after<=75
                or not math.isfinite(response) or not .3<=response<=2.5
                or len(first)!=2 or len(last)!=2 or not all(math.isfinite(v) for v in [*first,*last])):
            raise ValueError('Inconsistent measured elbow extension step')
        if not math.isclose((last[1]-first[1])/(before-after),response,rel_tol=1e-6,abs_tol=1e-6):
            raise ValueError('Extension image response does not match observations')
    minimum=min(s['target_p'] for s in steps)
    if minimum!=report.get('minimum_command_p') or min(s['after_p'] for s in steps)!=report.get('minimum_feedback_p'):
        raise ValueError('Extension endpoint does not match measurements')
    config=copy.deepcopy(previous)
    config['axes']['2']['min_p']=min(config['axes']['2']['min_p'],minimum)
    config.setdefault('position_windows_p',{})['2']=max(config.get('position_windows_p',{}).get('2',90),
        abs(config['pose_reference']['2']-config['axes']['2']['min_p']))
    config.setdefault('measured_range_extensions',{})['2']=copy.deepcopy(report)
    return config


def apply_measured_pitch_extension(previous, report, now=None):
    """Extend wrist command range only from completed coordinated probes."""
    import numpy as np
    now=time.time() if now is None else now
    weights=report.get('target',{}).get('weights_sha256')
    axes=previous.get('tracking_axes',[])
    allowed_result=report.get('result')=='MEASURED_PITCH_EXTENSION_PASS'
    if report.get('result')=='MEASURED_PITCH_PREFIX_PASS':
        allowed_result=(report.get('source_stop_error')=='RuntimeError: Coordinated pitch probe would exceed bounded range'
                        and report.get('completion_scope')=='completed_steps_only'
                        and isinstance(report.get('source_report_sha256'),str)
                        and len(report['source_report_sha256'])==64)
    if (not allowed_result
            or report.get('motion_enabled') is not True
            or not 0<=now-report.get('finished_unix',0)<=1800
            or 2 not in axes or 3 not in axes or not isinstance(weights,str) or len(weights)!=64
            or weights!=previous.get('operator_target_lock',{}).get('weights_sha256')):
        raise ValueError('Fresh matching coordinated pitch measurements required')
    matrix=np.asarray(previous['local_calibration']['jacobian'])
    steps=report.get('steps',[])
    if not steps:raise ValueError('No measured coordinated steps')
    for step in steps:
        start,goal,end=step['before_p'],step['target_p'],step['after_p']
        if (any(type(d.get(a)) is not int for d in (start,goal,end) for a in ('2','3'))
                or not 0<goal['3']-start['3']<=65 or not 0<goal['2']-start['2']<=65
                or not 1000<=start['3']<=1175 or not goal['3']<=1200
                or not 5<=end['3']-start['3']<=75 or not 5<=end['2']-start['2']<=65
                or not previous['axes']['2']['min_p']<=goal['2']<=previous['axes']['2']['max_p']):
            raise ValueError('Coordinated joint step outside measured bounds')
        first,last=np.asarray(step['before_center']),np.asarray(step['after_center'])
        if first.shape!=(2,) or last.shape!=(2,) or not np.isfinite([first,last]).all():
            raise ValueError('Invalid coordinated image measurement')
        observed=last-first
        predicted=matrix[:,axes.index(2)]*(end['2']-start['2'])+matrix[:,axes.index(3)]*(end['3']-start['3'])
        if (np.linalg.norm(observed)>40 or abs(observed[1]-predicted[1])>22
                or not np.allclose(observed,step['observed_image_shift_px'])
                or not np.allclose(predicted,step['predicted_image_shift_px'])):
            raise ValueError('Coordinated image response does not match measurements')
    maximum=max(s['target_p']['3'] for s in steps)
    if maximum!=report.get('maximum_wrist_command_p'):
        raise ValueError('Claimed pitch endpoint does not match measurements')
    config=copy.deepcopy(previous)
    config['axes']['3']['max_p']=max(config['axes']['3']['max_p'],maximum)
    config.setdefault('position_windows_p',{})['3']=max(config.get('position_windows_p',{}).get('3',90),
        abs(config['pose_reference']['3']-maximum))
    config.setdefault('measured_range_extensions',{})['3']=copy.deepcopy(report)
    return config


def apply_tracking_profile(previous):
    config=copy.deepcopy(previous)
    axes=config.get('tracking_axes',[0,3])
    config.update(max_step_p=100,gain_near=.55,gain_far=.92,gain_scale_px=30,
                  relax_priority_at_limits=True,continuous_follow=True,
                  retarget_period_s=.4,retarget_min_error_px=30)
    config.update(locked_tracking_confidence=.6,confidence_grace_s=.35)
    config['command_step_limits_p']={str(a):100 if a in (0,3) else 30 if a==4 else 45 for a in axes}
    calibration=config.get('local_calibration',{})
    elbow_span=calibration.get('axis_amplitudes',{}).get('2',0)
    if 2 in axes and elbow_span>=45:
        # The measured loaded elbow needs ~35 P just to overcome negative bias.
        # Give it useful displacement when the wrist saturates, inside soft limits.
        config['command_step_limits_p']['2']=65
    config['handoff_weights']=[2 if a in (1,2) else 1 for a in axes]
    if 3 in axes:
        # P1076 commands / P1051 feedback were physically exercised on this wrist.
        # Expand only upward toward that measured region; retain the lower bound.
        config['axes']['3']['max_p']=max(config['axes']['3']['max_p'],1076)
        config.setdefault('position_windows_p',{})['3']=400
    if config.get('startup_pose_policy')=='measure_current_pose':
        # A local camera fit does not define a new mechanical travel stop.
        # Startup, tracking and calibration use the same selected ranges.
        for axis,(low,high) in pose_envelopes(config).items():
            if axis not in axes:
                continue
            config['axes'].setdefault(str(axis),{'direction':-1}).update(min_p=low,max_p=high)
            reference=config['pose_reference'][str(axis)]
            config.setdefault('position_windows_p',{})[str(axis)]=max(abs(reference-low),abs(high-reference))
        if axes==[0,2,3]:
            config['joint_weights']=[1,1.35,1]
            config['negative_command_step_limits_p']={'2':100}
            config['actual_step_limits_p']={'2':65}
            config['settling_allowance_p']={'2':25}
        config['bounded_settling_feedback']=True
        config.update(smooth_coordination=True,coordinated_retarget=True,
                      retarget_period_s=.22,planning_period_s=.08)
    config['tracking_profile']='continuous-fast-20261022'
    response=config.get('response_profile')
    if response not in (None,'responsive-v1'):
        raise ValueError(f'Unknown follow response profile: {response}')
    if response=='responsive-v1':
        config.update(duration_ms=400,settle_seconds=.06,planning_period_s=.06,
                      retarget_period_s=.2,retarget_min_error_px=12,
                      center_tolerance_px=4,hold_hysteresis=1.2,filter_alpha=.9,
                      gain_near=.72,gain_far=.98,gain_scale_px=22,
                      responsive_tracking=True,responsive_retarget=True,
                      reacquire_visible_target=True)
        if axes==[0,2,3]:
            config['joint_weights']=[1,1.15,1]
        config['tracking_profile']='responsive-full-range-20261025'
    return config
