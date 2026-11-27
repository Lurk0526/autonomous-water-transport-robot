"""Inspection, contact baseline, PPO training/resume, evaluation and native GUI."""
import argparse
import copy
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .model import ROOT,config


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def episode(env,policy=None,*,seed=42,options=None):
    obs,info=env.reset(seed=seed,options=options);total=0.;parts=Counter()
    while True:
        action=np.zeros(3) if policy is None else policy.predict(obs,deterministic=True)[0]
        obs,reward,terminated,truncated,info=env.step(action);total+=reward
        parts.update(info.get('reward_parts',{}))
        if terminated or truncated:break
    return {**info,'return':float(total),'reward_totals':dict(parts)}


def provenance(cfg):
    digest=hashlib.sha256()
    for source in sorted((ROOT/'arm_training').glob('*.py')):
        digest.update(source.name.encode());digest.update(source.read_bytes())
    return {'config':cfg,'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        'model_sha256':hashlib.sha256((ROOT/'assets/candidate-model.json').read_bytes()).hexdigest(),
        'source_sha256':digest.hexdigest(),
        'platform':platform.platform(),'machine':platform.machine(),'hardware_ready':False,
        'scope':'User STEP visual geometry; candidate arm grouping and single-command four-finger radial contact envelope. Not validated MG23 linkage or hardware dynamics.'}


def permit(args,cfg):
    if not getattr(args,'experimental_model',False):
        raise SystemExit('真实网格已转换，但关节分组、MG23 联动和泡沫参数尚未标定。请先运行 inspect / baseline 并阅读模型限制；仅做候选模型实验时追加 --experimental-model。不会导出实机运动指令。')


def evaluate(env,policy,count,seed):
    results=[]
    for i in range(count):
        results.append(episode(env,policy,seed=seed+i))
        if (i+1)%20==0:print(f'EVAL {i+1}/{count} success={sum(r["is_success"] for r in results)}',flush=True)
    reachable=[r for r in results if r['reachable']];wins=sum(r['is_success'] for r in results);n=len(results)
    # Wilson interval, rather than treating a finite sample as a population guarantee.
    z=1.96;phat=wins/max(n,1);den=1+z*z/max(n,1)
    center=(phat+z*z/(2*max(n,1)))/den
    half=z*np.sqrt(phat*(1-phat)/max(n,1)+z*z/(4*max(n,1)**2))/den
    return {'episodes':n,'successes':wins,'success_rate':wins/max(n,1),'success_rate_wilson95':[max(0.,center-half),min(1.,center+half)],
        'reachable_episodes':len(reachable),'reachable_fraction':len(reachable)/max(n,1),
        'success_rate_among_reachable':sum(r['is_success'] for r in reachable)/max(len(reachable),1),
        'failures':dict(Counter(r['failure_reason'] for r in results if r['failure_reason'])),
        'passes_provisional_simulation_goal':n>=200 and wins/n>=.95,
        'hardware_ready':False,'trials':results}


def policy_load(path):
    if path is None:return None
    from stable_baselines3 import PPO
    return PPO.load(path,device='cpu')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'config/train.yaml'))
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inspect');p.add_argument('--output',type=Path,default=ROOT/'outputs/inspection.json')
    p=sub.add_parser('baseline');p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=10000);p.add_argument('--output',type=Path,default=ROOT/'outputs/baseline')
    for cmd in ['train','evaluate','gui','search','workspace']:
        p=sub.add_parser(cmd);p.add_argument('--experimental-model',action='store_true')
        p.add_argument('--output',type=Path,default=ROOT/'outputs'/cmd)
        p.add_argument('--seed',type=int,default=42 if cmd=='train' else 10000)
        if cmd in ['evaluate','gui']:p.add_argument('--policy',type=Path)
        if cmd=='train':
            p.add_argument('--steps',type=int);p.add_argument('--envs',type=int);p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true')
        if cmd=='evaluate':p.add_argument('--episodes',type=int,default=200)
        if cmd=='gui':p.add_argument('--duration',type=float);p.add_argument('--cad-only',action='store_true')
    args=parser.parse_args();cfg=config(args.config)
    if args.command=='inspect':
        import mujoco
        from .model import make_model
        m,_,cad=make_model(cfg,visual=True)
        result={**provenance(cfg),'model_status':cad['status'],'dimensions':{'nq':m.nq,'nu':m.nu,'geoms':m.ngeom,'meshes':m.nmesh},
            'cad_parts':len(cad['parts']),'cad_occurrences':len(cad['occurrences']),
            'all_brep_valid':all(x['brep_valid'] for x in cad['parts']),
            'parking_ranges_are_measured':False,'camera_rendering_or_hand_eye_calibration_included':False}
        write_json(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    if args.command!='baseline':permit(args,cfg)
    from .env import BowlEnv
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command in ['baseline','evaluate']:
        if args.episodes<=0:parser.error('episodes must be positive')
        policy=policy_load(getattr(args,'policy',None));env=BowlEnv(cfg)
        start=time.perf_counter();result=evaluate(env,policy,args.episodes,args.seed)
        result['wall_s']=time.perf_counter()-start;result['provenance']=provenance(cfg)
        write_json(args.output/'evaluation.json',result)
        if env.frames:
            write_json(args.output/'replay.json',env.frames)
            write_json(args.output/'replay-state.json',{'parking':env.parking.tolist(),'place':env.place.tolist()})
        print(json.dumps({k:v for k,v in result.items() if k not in ['trials','provenance']},ensure_ascii=False,indent=2))
        return 0
    if args.command=='workspace':
        env=BowlEnv(cfg);trials=[]
        for x in [-.08,-.04,0.,.04,.08]:
            for y in [-.08,-.04,0.,.04,.08]:
                for yaw in [-45,-20,0,20,45]:
                    _,info=env.reset(seed=args.seed,options={'parking':[x,y,np.deg2rad(yaw)]})
                    trials.append({'parking':[x,y,yaw],'reachable':info['reachable']})
        write_json(args.output/'workspace.json',{'ranges_are_not_measurements':True,'provenance':provenance(cfg),'trials':trials})
        print('Workspace poses',len(trials),'reachable',sum(t['reachable'] for t in trials));return 0
    if args.command=='search':
        results=[]
        for height in [.025,.035,.045]:
            for close in [.025,.032,.038,.045]:
                trial=copy.deepcopy(cfg);trial['fixture']['grasp_height_above_table_m']=height;trial['gripper']['close_m']=close
                env=BowlEnv(trial)
                score=evaluate(env,None,5,args.seed)
                results.append({'height_m':height,'close_m':close,**{k:v for k,v in score.items() if k!='trials'}})
                print(height,close,score['success_rate'],score['failures'],flush=True)
        write_json(args.output/'contact-search.json',{'provenance':provenance(cfg),'candidates':results,'automatically_applied':False});return 0
    if args.command=='gui':
        import mujoco
        import mujoco.viewer
        if platform.system()=='Darwin' and not os.environ.get('MUJOCO_PYTHON_BIN_PATH'):
            print('macOS 原生 passive viewer 请通过 ./run.sh gui 启动（使用 mjpython）。',flush=True)
        env=BowlEnv(cfg,visual=True);policy=policy_load(args.policy)
        if args.cad_only:
            env.reset(seed=args.seed,options={'parking':[0,0,0]});env.data.qpos[env.qids]=0;mujoco.mj_forward(env.model,env.data)
            frames=[{'t':0.,'qpos':env.data.qpos.tolist(),'qvel':env.data.qvel.tolist(),'ctrl':env.data.ctrl.tolist()}]
        else:
            info=episode(env,policy,seed=args.seed,options={'parking':[0,0,0]})
            print('Candidate episode result:',info['is_success'],info['failure_reason'],flush=True);frames=env.frames
            if not frames:
                frames=[{'t':0.,'qpos':env.data.qpos.tolist(),'qvel':env.data.qvel.tolist(),'ctrl':env.data.ctrl.tolist()}]
        index=0;state={'paused':False,'restart':False,'quit':False}
        def key(k):
            if k==32:state['paused']=not state['paused']
            elif k==82:state['restart']=True
            elif k==81:state['quit']=True
        with mujoco.viewer.launch_passive(env.model,env.data,key_callback=key,show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:]=[.15,0,.45];viewer.cam.distance=1.1;viewer.cam.elevation=-25;viewer.cam.azimuth=135
            print('GUI_READY — CAD geometry / uncalibrated joints and contact. Space pause; R replay; Q quit.',flush=True)
            started=last=time.perf_counter()
            while viewer.is_running() and not state['quit']:
                now=time.perf_counter()
                if args.duration and now-started>=args.duration:break
                if state['restart']:index=0;state['restart']=False
                if not state['paused'] and now-last>=cfg['simulation']['control_dt_s']:
                    frame=frames[index];index=(index+1)%len(frames);last=now
                    with viewer.lock():
                        env.data.qpos[:]=frame['qpos'];env.data.qvel[:]=frame['qvel'];env.data.ctrl[:]=frame['ctrl'];mujoco.mj_forward(env.model,env.data)
                viewer.sync();time.sleep(.01)
        return 0
    if args.command=='train':
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback,CallbackList,BaseCallback
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        n=args.envs or cfg['ppo']['n_envs'];steps=args.steps or cfg['ppo']['total_timesteps']
        if n not in range(1,9) or steps<=0:parser.error('Use 1–8 environments and positive steps')
        if not args.resume and (args.output/'final.zip').exists():raise SystemExit('此目录已有 final.zip，请换输出目录或使用 --resume。')
        def factory(rank):
            def make():
                env=Monitor(BowlEnv(cfg),filename=str(args.output/f'env-{rank}.csv'))
                env.reset(seed=args.seed+rank);return env
            return make
        venv=DummyVecEnv([factory(0)]) if n==1 else SubprocVecEnv([factory(i) for i in range(n)],start_method='spawn')
        eval_env=Monitor(BowlEnv(cfg));eval_env.reset(seed=1000000)
        if args.resume:
            metadata_path=args.resume.parent/'run.json'
            if not metadata_path.exists():raise SystemExit('Checkpoint is missing run.json; cannot verify model/config compatibility')
            old=json.loads(metadata_path.read_text());current=provenance(cfg)
            if any(old['provenance'].get(key)!=current[key] for key in ['config_sha256','model_sha256','source_sha256']):
                raise SystemExit('Resume requires the same config and model. Start a new run for changed parameters.')
            model=PPO.load(args.resume,env=venv,device='cpu',tensorboard_log=str(args.output/'tensorboard'))
        else:
            model=PPO('MlpPolicy',venv,device='cpu',learning_rate=cfg['ppo']['learning_rate'],n_steps=64 if args.smoke else cfg['ppo']['n_steps'],
                batch_size=cfg['ppo']['batch_size'],n_epochs=2 if args.smoke else cfg['ppo']['n_epochs'],gamma=cfg['ppo']['gamma'],gae_lambda=cfg['ppo']['gae_lambda'],
                policy_kwargs={'net_arch':cfg['ppo']['hidden_sizes'],'log_std_init':cfg['ppo']['log_std_init']},seed=args.seed,verbose=1,tensorboard_log=str(args.output/'tensorboard'))
        write_json(args.output/'run.json',{'provenance':provenance(cfg),'seed':args.seed,'envs':n,'additional_steps':steps,'resume':str(args.resume) if args.resume else None,'smoke_only':args.smoke})
        write_json(args.output/'config-used.json',cfg)
        class Metrics(BaseCallback):
            def _on_step(self):
                for info in self.locals['infos']:
                    if 'episode' in info:
                        self.logger.record('task/success',float(info.get('is_success',False)))
                        self.logger.record('task/reachable',float(info.get('reachable',False)))
                        self.logger.record('task/place_error_m',info.get('place_error_m',0.))
                return True
        freq=max(64 if args.smoke else cfg['ppo']['checkpoint_frequency']//n,1)
        checkpoints=CheckpointCallback(freq,str(args.output),name_prefix='checkpoint')
        callback=CallbackList([Metrics(),checkpoints,EvalCallback(eval_env,best_model_save_path=str(args.output),log_path=str(args.output/'eval'),
            eval_freq=max(freq,256),n_eval_episodes=3 if args.smoke else 20,deterministic=True)])
        try:model.learn(total_timesteps=steps,reset_num_timesteps=not bool(args.resume),callback=callback,progress_bar=False)
        except KeyboardInterrupt:print('Interrupted; preserving current policy.',flush=True)
        finally:
            model.save(args.output/'final');venv.close();eval_env.close()
        print('Saved',args.output/'final.zip','— smoke or training completion does not imply grasp acceptance.');return 0


if __name__=='__main__':raise SystemExit(main())
