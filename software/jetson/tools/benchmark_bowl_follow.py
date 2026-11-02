#!/usr/bin/env python3
"""Offline delayed-camera/finite-duration servo simulation. Never opens serial."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from vision.bowl_follow import FollowPolicy,FollowSettings


def simulate(policy_type, settings, seed=9):
    matrix=np.array([[1.2,.005],[-.08,.87]])
    rng=np.random.default_rng(seed)
    position=np.zeros(2);destination=None;motion_start=None;origin=None
    offset=np.array([[17,16],[25,4]])
    dt=.01;camera_period=.06;camera_latency=.05;motion_duration=.4
    camera_next=0.;queue=[];latest=None;baseline=time.time();commands=0
    available_at=0.;errors=[];trace=[];hold_commands=0
    policy=policy_type(settings)
    for i in range(1801):
        t=i*dt
        if destination is not None:
            fraction=min(1,(t-motion_start)/motion_duration)
            position=origin+(destination-origin)*fraction
            if fraction>=1:destination=None
        # A bowl starts off-centre, moves left during 6..8 seconds, then stops.
        external=np.array([-min(2,max(0,t-6))*25.,0.])
        center=np.array([225.,165.])+external+matrix@position
        if t>=camera_next:
            observed=center+rng.normal(0,.45,2);cx,cy=observed
            stamp=baseline+t;sec=int(stamp)
            record=dict(state='live',valid=True,stamp=dict(sec=sec,nanosec=round((stamp-sec)*1e9)),
                width=320,height=240,detections=[dict(**{'class':'target_bowl'},confidence=.95,
                mask_centroid_px=observed.tolist(),bbox_xyxy_px=[cx-30,cy-20,cx+30,cy+20],mask_area_px=2400)])
            queue.append((t+camera_latency,record));camera_next=t+camera_period
        if queue and queue[0][0]<=t:
            _,latest=queue.pop(0)
            if getattr(settings,'bounded_allocation',False):
                low=[];high=[]
                for axis in range(2):
                    low.append(-max(0,min(settings.max_step-offset[axis,1],position[axis]+90-offset[axis,1])))
                    high.append(max(0,min(settings.max_step-offset[axis,0],90-position[axis]-offset[axis,0])))
                settings.displacement_bounds=(low,high)
            plan=policy.update(latest,baseline+t,t)
            if plan['state']=='centered':
                if destination is not None:
                    destination=None;available_at=t+.05
            elif plan['state']=='move' and destination is None and t>=available_at:
                delta=np.array([plan['delta_p'].get(0,0),plan['delta_p'].get(3,0)],float)
                for j in range(2):
                    if delta[j]:delta[j]-=np.sign(delta[j])*offset[j,0 if delta[j]>0 else 1]
                origin=position.copy();destination=position+delta;motion_start=t
                available_at=t+motion_duration+.05;commands+=1
                if t>=12:hold_commands+=1
        error=float(np.linalg.norm(center-[160,120]));errors.append(error)
        if i%10==0:trace.append(dict(t=round(t,2),error_px=error,center=center.tolist()))
    def settled(after,before):
        for i in range(int(after/dt),int((before-.6)/dt)):
            if max(errors[i:i+60])<=9:return round(i*dt-after,2)
        return None
    return dict(initial_settle_s=settled(0,6),after_stop_settle_s=settled(8,18),
        moving_rms_error_px=float(np.sqrt(np.mean(np.square(errors[600:800])))),
        final_error_px=errors[-1],commands=commands,commands_after_12s=hold_commands,trace=trace)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'work/control-speed-20261022/before/bowl_follow.py')
    p.add_argument('--output',type=Path,default=ROOT/'work/control-speed-20261022/simulation.json')
    args=p.parse_args()
    spec=importlib.util.spec_from_file_location('before_follow',args.baseline)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    common=dict(jacobian=[[1.2,.005],[-.08,.87]],stable_frames=2,
                deadband_compensation=((17,16),(25,4)),min_step=1,position_gain=.65,
                deadband_x=6/160,deadband_y=6/120)
    old=module.FollowSettings(**common,max_step=40,alpha=.65,period=.5)
    new=FollowSettings(**common,max_step=65,alpha=.8,period=.1,bounded_allocation=True,
                       adaptive_gain=True,hold_hysteresis=1.5)
    result=dict(kind='simulation_only_no_hardware',model='fixed local image Jacobian; 50ms camera latency; 400ms interpolation; 450ms actuator availability',
                baseline=simulate(module.FollowPolicy,old),optimized=simulate(FollowPolicy,new))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:({a:b for a,b in v.items() if a!='trace'} if isinstance(v,dict) else v) for k,v in result.items()},indent=2))


if __name__=='__main__':main()
