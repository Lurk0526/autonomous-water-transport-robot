"""Finite full-orientation reachability probe; no physics, bowl truth, or hardware.

Inspired by inverse-reachability base placement: determine a useful arm region
before choosing where to park. This grid is a diagnostic, not a global map or a
claim of dynamically feasible grasp success.
"""
import argparse,json
from pathlib import Path
import numpy as np
from arm_training.model import config
from arm_training.mobile_scene import ArmScreen
import yaml


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 c=config();root=Path(__file__).resolve().parents[1];mobile=yaml.safe_load((root/'config/mobile.yaml').read_text());screen=ArmScreen(c,mobile);rows=[]
 for x in [.18,.20,.21,.215,.22,.225,.23,.24,.25,.26,.27]:
  for y in [-.04,0.,.04]:
   seed=None;poses=[]
   for z in [.46,.40,.46,.3985]:
    q,ok=screen.solve_grasp([x,y,z],seed=seed);poses.append({'tcp_m':[x,y,z],'possible':bool(ok),'q_rad':q.tolist()})
    if not ok:break
    seed=q
   rows.append({'xy_m':[x,y],'all_waypoints_reachable':len(poses)==4 and all(p['possible'] for p in poses),'poses':poses})
 result={'scope':'Finite kinematic grid only; collision, dynamics and camera confirmation still required for parking',
  'height_m':.35,'target_offsets_and_parking_errors_are_not_measured':True,'trials':rows}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print('reachable',sum(r['all_waypoints_reachable'] for r in rows),'/',len(rows))


if __name__=='__main__':main()
