"""Uncalibrated delay/lag/deadband/slew stimuli; not identified Jibot dynamics."""
import math

def simulate(command_hz=20.,delay_s=.2,tau_s=.25,slew_rad_s=.6,deadband_rad=.01):
    dt=.005;duration=3.;q=0.;target=0.;pending=[];next_command=0.;errors=[];samples=[]
    for i in range(round(duration/dt)):
        t=i*dt;desired=.2 if t>=.2 else 0.
        if t+1e-9>=next_command:
            pending.append((t+delay_s,desired));next_command+=1/command_hz
        while pending and pending[0][0]<=t+1e-9:target=pending.pop(0)[1]
        error=target-q
        vel=0. if abs(error)<=deadband_rad else max(-slew_rad_s,min(slew_rad_s,error/tau_s))
        q+=vel*dt;errors.append(abs(desired-q));samples.append((t,q,vel))
    return {'max_error_rad':max(errors),'final_error_rad':errors[-1],'samples':samples}

def audit():
    a=simulate(20.);b=simulate(100.)
    checks={
        'no_response_before_delayed_command':all(abs(q)<1e-12 for t,q,v in a['samples'] if t<.4-1e-9),
        'command_frequency_does_not_remove_delay':all(abs(q)<1e-12 for t,q,v in b['samples'] if t<.4-1e-9),
        'slew_limit_enforced':all(abs(v)<=.6+1e-12 for t,q,v in a['samples']),
        'deadband_leaves_nonzero_tracking_error':.005<a['final_error_rad']<=.011,
    }
    return {'kind':'synthetic_actuator_interface_screening','checks':checks,
            'all_checks_passed':all(checks.values()),
            'assumptions':{'delay_s':.2,'lag_tau_s':.25,'slew_rad_s':.6,'deadband_rad':.01},
            'max_tracking_error_rad_20hz':a['max_error_rad'],
            'max_tracking_error_rad_100hz':b['max_error_rad'],
            'hardware_identification_completed':False,
            'warning':'These values are test stimuli, not measured servo parameters. Command rate is not mechanical tracking bandwidth.'}
