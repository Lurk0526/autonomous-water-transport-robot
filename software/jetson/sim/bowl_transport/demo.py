"""Mocked-observation phase rehearsal. No robotics or fluid performance claims."""
import argparse
import csv
import html
import json
from pathlib import Path
from flow import Supervisor,Observation,PHASES
from screening import quintic
HERE=Path(__file__).resolve().parent
LABELS=['接近并准备','停车稳定检查','下降接近','闭爪并持续验证','抬离平台','平滑回收',
        '携碗行驶并制动','准备放置（默认停车）','停车定位检查','接触并验证承托',
        '持续承托下松爪','撤离与稳定复查','完成']

def synthetic_observation(t,**changes):
    """Deliberately synthetic evidence; never use this helper on hardware."""
    vals=dict(sample_time_s=t,speed=0.,angular_speed=0.,acceleration=0.,telemetry_age=.01,
              path_age=.01,evidence_age=.01,target_age=.05,pose_error_m=.001,pose_sigma_m=.001,
              tracking_error_rad=.001,tilt_rad=0.,slip_m=0.,pose_valid=True,arm_settled=True,
              path_clear=True,phase_done=False,grasp_verified=True,support_verified=True,
              release_verified=True,carry_secure=True,lift_clearance_verified=True,
              destination_ready=True,bowl_resting_verified=True,base_hold_confirmed=True,
              overlap_allowed=False,braking_clear=True,kinematics_ok=True,load_ok=True)
    vals.update(changes);return Observation(**vals)

def run(config,out):
    f=Supervisor(config['initial_gate_thresholds_require_calibration'])
    cfg=config['preview_assumptions_not_measurements']
    durations=dict(zip(['APPROACH_PICK','DESCEND','CLOSE','LIFT','RETRACT','CARRY','PREPLACE','LOWER','OPEN','RETREAT'],[4.,1.2,1.,1.5,2.,7.,3.,1.5,.8,1.]))
    dt=.02;t=0.;rows=[];events=[(0.,f.phase)];vlast=0.
    while f.phase not in {'DONE','HOLD'} and t<120:
        p=f.phase;duration=durations.get(p,1.);u=min(1.,f.elapsed/duration)
        v=0.
        if p=='APPROACH_PICK':v=.20*(1-quintic(f.elapsed,4.)[0])
        if p=='CARRY':v=.2*(quintic(f.elapsed,3.)[0]-quintic(f.elapsed-4.,3.)[0])
        a=(v-vlast)/dt if rows else 0.;vlast=v
        done=f.elapsed>=duration
        obs=synthetic_observation(t,speed=v,acceleration=a,phase_done=done,
                                  preplace_window=(p=='CARRY' and done),
                                  base_hold_confirmed=abs(v)<.001)
        # All contact/retention signals here are fabricated to exercise normal control flow.
        if p=='CLOSE':obs.grasp_verified=done
        if p=='LOWER':obs.support_verified=obs.bowl_resting_verified=done
        if p=='OPEN':obs.release_verified=done
        h=cfg['platform_height_m'];s=quintic(f.elapsed,duration)[0]
        if p=='LIFT':h+=.06*s
        elif p=='RETRACT':h=(h+.06)*(1-s)+cfg['carry_bowl_bottom_height_m']*s
        elif p=='CARRY':h=cfg['carry_bowl_bottom_height_m']
        elif p=='PREPLACE':h=cfg['carry_bowl_bottom_height_m']*(1-s)+(h+.06)*s
        elif p=='PLACE_WAIT':h+=.06
        elif p=='LOWER':h+=.06*(1-s)
        new=f.update(obs,dt)
        rows.append(dict(t_s=round(t,3),phase=p,speed_m_s=v,acceleration_m_s2=a,bowl_bottom_m=h))
        t=round(t+dt,6)
        if new!=p:events.append((round(t,3),new))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'timeline.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'kind':'mocked_observation_rehearsal_not_robot_or_fluid_simulation','final_phase':f.phase,
             'hold_reason':f.hold_reason,'illustrative_duration_s':round(t,2),'moving_preparation_default':False,
             'spilled_water_mass_g':None,'physical_grasp_success':None,'events':events}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    labels=dict(zip(PHASES,LABELS))
    tr=''.join(f'<tr><td>{ts:.2f}</td><td>{html.escape(p)}</td><td>{labels.get(p,p)}</td></tr>' for ts,p in events)
    page='<meta charset="utf-8"><title>动作顺序预演</title><style>body{font:16px system-ui;max-width:960px;margin:40px auto}td,th{padding:10px;border-bottom:1px solid #ddd}</style><h1>动作顺序预演</h1><p>此页面全部采用合成成功信号，仅检查正常状态转换。没有机器人动力学或物理水。时间不代表实机性能，默认不在行驶中展开。</p><p><a href="report.html">返回本机审计报告</a></p><table><tr><th>示例时间/s</th><th>状态</th><th>含义</th></tr>'+tr+'</table>'
    (out/'timeline.html').write_text(page,encoding='utf8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=HERE/'scenario.json');p.add_argument('--out',type=Path,default=HERE/'output');a=p.parse_args()
    print(json.dumps(run(json.loads(a.config.read_text()),a.out.resolve()),ensure_ascii=False,indent=2))
