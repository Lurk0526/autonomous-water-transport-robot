"""Fit candidate braking parameters from timestamped command + measured-speed logs.

No motions are requested. Estimates do not automatically become deployment limits.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
from .feedback import finite


def estimate_braking(rows, source_kind="unconfirmed"):
    if len(rows)<4:
        raise ValueError("insufficient samples")
    keys=("t_s","command_speed_m_s","measured_speed_m_s")
    if any(not all(finite(r.get(k)) for k in keys) for r in rows):
        raise ValueError("missing/nonfinite calibration samples")
    if any(b['t_s']<=a['t_s'] or b['t_s']-a['t_s']>.25 for a,b in zip(rows,rows[1:])):
        raise ValueError("timestamp reversal or excessive sampling gap")
    trials=[]
    for i in range(1,len(rows)):
        if rows[i-1]['command_speed_m_s']<=.05 or rows[i]['command_speed_m_s']>.005:
            continue
        v0=rows[i]['measured_speed_m_s']
        if v0<=.10:
            continue
        segment=[]
        for r in rows[i:]:
            if r['command_speed_m_s']>.005:
                break
            segment.append(r)
            if r['measured_speed_m_s']<=.005:
                break
        if len(segment)<4 or segment[-1]['measured_speed_m_s']>.005:
            continue
        onset=next((r for r in segment if r['measured_speed_m_s']<=v0-.02),None)
        decels=[(a['measured_speed_m_s']-b['measured_speed_m_s'])/(b['t_s']-a['t_s'])
                for a,b in zip(segment,segment[1:]) if .10*v0<=b['measured_speed_m_s']<=.90*v0]
        decels=[d for d in decels if d>0]
        if onset is None or len(decels)<3:
            continue
        distance=sum((a['measured_speed_m_s']+b['measured_speed_m_s'])/2*(b['t_s']-a['t_s'])
                     for a,b in zip(segment,segment[1:]))
        trials.append({'command_time_s':rows[i]['t_s'],'initial_speed_m_s':v0,
                       'response_delay_upper_sample_s':onset['t_s']-rows[i]['t_s'],
                       'median_decel_m_s2':statistics.median(decels),
                       'stop_duration_s':segment[-1]['t_s']-rows[i]['t_s'],
                       'wheel_speed_integrated_stop_distance_m':distance})
    return {'kind':'braking_candidate_fit','source_kind':source_kind,'trials':trials,
            'minimum_trial_decel_m_s2':min((t['median_decel_m_s2'] for t in trials),default=None),
            'maximum_observed_response_delay_s':max((t['response_delay_upper_sample_s'] for t in trials),default=None),
            'sample_interval_max_s':max(b['t_s']-a['t_s'] for a,b in zip(rows,rows[1:])),
            'enough_repeated_trials':len(trials)>=3,'measurements_automatically_applied':False,
            'note':'编码器积分距离受打滑影响；这不是外部真值制动距离。响应延迟不包含未记录的相机/网络时延，不能直接当完整停车预算。'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-kind',choices=['recorded_real','synthetic_fixture','unconfirmed'],default='unconfirmed')
    a=p.parse_args()
    with a.csv.open() as f:
        rows=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(f)]
    report=estimate_braking(rows,a.source_kind)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['trials'] else 3


if __name__=='__main__':
    raise SystemExit(main())
