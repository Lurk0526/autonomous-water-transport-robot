"""Small-amplitude single-mode numerical screening; no robot/contact/overflow physics.

q is the TOTAL signed first-mode edge response, not a residual to add to r*a/g.
Natural frequency/damping are assumptions; results cannot estimate spilled grams.
"""
import json
import math
from pathlib import Path


def quintic(t, duration, distance=1.):
    if duration<=0:raise ValueError('duration must be positive')
    u=max(0.,min(1.,t/duration))
    p=distance*(10*u**3-15*u**4+6*u**5)
    v=distance*(30*u*u-60*u**3+30*u**4)/duration
    a=distance*(60*u-180*u*u+120*u**3)/duration**2
    j=distance*(60-360*u+360*u*u)/duration**3 if 0<=t<=duration else 0.
    return p,v,a,j


def response(forcing, duration, dt, frequency_hz, damping, radius=.078):
    if not all(math.isfinite(v) and v>0 for v in (duration,dt,frequency_hz,damping,radius)):
        raise ValueError('positive finite model parameters required')
    if dt*frequency_hz>.1:raise ValueError('time step too coarse for screening mode')
    q=v=0.;peak=0.;rows=[];omega=2*math.pi*frequency_hz
    n=math.ceil(duration/dt);h=duration/n
    def rhs(t,q,v):
        ax,az,tilt=forcing(t)
        if not all(math.isfinite(x) for x in (ax,az,tilt)) or 9.81+az<4.905 or abs(tilt)>.2:
            raise ValueError('outside reduced-model domain')
        target=radius*(ax/(9.81+az)-math.tan(tilt))
        return v,omega**2*(target-q)-2*damping*omega*v
    for i in range(n):
        t=i*h
        k1=rhs(t,q,v);k2=rhs(t+h/2,q+h*k1[0]/2,v+h*k1[1]/2)
        k3=rhs(t+h/2,q+h*k2[0]/2,v+h*k2[1]/2);k4=rhs(t+h,q+h*k3[0],v+h*k3[1])
        q+=h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
        v+=h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
        peak=max(peak,abs(q));rows.append((t+h,q))
    return {'peak_edge_response_m':peak,'final_edge_response_m':q,'rows':rows}


def sweep():
    results=[]
    # Same base braking profile and same arm displacement/duration; timing differs.
    # All numbers below are test stimuli, NOT measured robot trajectories.
    for freq in [.6,1.2,2.4,3.5]:
        for damping in [.05,.12,.3]:
            for arm_duration in [1.,2.,4.]:
                pair={}
                for name,start in [('simultaneous',0.),('after_braking',2.5)]:
                    def force(t):
                        ax=-quintic(t,2.,.2)[1]+quintic(t-start,arm_duration,-.12)[2]
                        az=quintic(t-start,arm_duration,-.12)[2]
                        return ax,az,0.
                    r=response(force,12.,.01,freq,damping)
                    pair[name]=r['peak_edge_response_m']*1000
                results.append({'frequency_hz':freq,'damping':damping,'arm_duration_s':arm_duration,
                                'simultaneous_mm':pair['simultaneous'],'after_braking_mm':pair['after_braking'],
                                'simultaneous_worse':pair['simultaneous']>pair['after_braking']})
    force=lambda t:(.2*math.sin(2*math.pi*.7*t),.04*math.sin(t),.005*math.sin(3*t))
    convergence={str(dt):response(force,8.,dt,2.4,.12)['peak_edge_response_m'] for dt in [.02,.01,.005]}
    diff=abs(convergence['0.01']-convergence['0.005'])
    constant=response(lambda t:(.1,0.,0.),15.,.005,2.4,.3)
    expected=.078*.1/9.81
    # Exact effective-gravity compensation in this idealized scalar model.
    compensated=response(lambda t:(.1,0.,math.atan(.1/9.81)),5.,.005,2.4,.12)
    checks={
        'zero_input_stays_zero':response(lambda t:(0.,0.,0.),5.,.01,2.4,.12)['peak_edge_response_m']==0.,
        'constant_input_not_double_counted':abs(constant['final_edge_response_m']-expected)<1e-7,
        'time_step_convergence':diff<1e-5,
        'signed_tilt_compensation_not_penalized_twice':compensated['peak_edge_response_m']<1e-12,
        'quintic_acceleration_endpoints_zero':all(abs(quintic(t,2.,.12)[2])<1e-12 for t in [0,2]),
        'overlap_not_assumed_better':any(r['simultaneous_worse'] for r in results),
    }
    return {'kind':'uncalibrated_single_mode_sensitivity_screening',
            'checks':checks,'all_checks_passed':all(checks.values()),'scenarios':results,
            'overlap_worse_count':sum(r['simultaneous_worse'] for r in results),'scenario_count':len(results),
            'dt_convergence_m':convergence,'fine_step_difference_m':diff,
            'spilled_mass_g':None,'robot_dynamics_validated':False,
            'warning':'Total response may exceed zero freeboard; overflow and mass loss are not modeled. No runtime tilt-compensation recommendation follows from this idealized model.'}

if __name__=='__main__':
    out=Path(__file__).parent/'output/screening.json';out.parent.mkdir(exist_ok=True)
    r=sweep();out.write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='scenarios'},indent=2))
