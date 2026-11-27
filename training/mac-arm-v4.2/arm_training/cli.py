"""Inspection, contact baseline, PPO training/resume, evaluation and native GUI."""
import argparse
import copy
from collections import Counter
import hashlib
from . import __version__
from .fingerprints import model_fingerprint
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .model import ROOT,config
from .control import checkpoint_key
from .device import device_record,resolve_device,synchronize


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
    return {'seed':seed,**info,'return':float(total),'reward_totals':dict(parts)}


def provenance(cfg):
    digest=hashlib.sha256()
    for source in sorted((ROOT/'arm_training').glob('*.py')):
        digest.update(source.name.encode());digest.update(source.read_bytes())
    return {'package_generation':'mac-arm-v4','package_version':__version__,'config':cfg,'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
        'model_sha256':model_fingerprint(ROOT),
        'source_sha256':digest.hexdigest(),
        'platform':platform.platform(),'machine':platform.machine(),'hardware_ready':False,
        'scope':'User STEP visual geometry; CAD-pin closed-chain topology, one common rack, passive finger hinges and baked convex collision parts. Physical actuator, foam and camera parameters remain unmeasured.'}


def permit(args,cfg):
    if not getattr(args,'experimental_model',False):
        raise SystemExit('真实网格已转换，但关节分组、MG23 联动和泡沫参数尚未标定。请先运行 inspect / baseline 并阅读模型限制；仅做候选模型实验时追加 --experimental-model。不会导出实机运动指令。')


def evaluate(env,policy,count,seed):
    results=[]
    for i in range(count):
        results.append(episode(env,policy,seed=seed+i))
        if (i+1)%20==0:print(f'EVAL {i+1}/{count} success={sum(r["is_success"] for r in results)}',flush=True)
    return summarize_trials(results)


def summarize_trials(results):
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


def validate_paired_case(row, original):
    """Check matched numeric scenario fields, allowing platform round-off only."""
    def check(a, b, field):
        left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all() or not np.allclose(left, right, rtol=0, atol=1e-11):
            raise ValueError(f"Seed {row['seed']} has different {field}")
    check(row['parking_xy_yaw'], original['parking_xy_yaw'], 'parking')
    if ('scenario' in row) != ('scenario' in original):
        raise ValueError('One record is missing its scenario')
    if 'scenario' in row:
        if row['scenario'].keys() != original['scenario'].keys():
            raise ValueError('Paired scenarios have different fields')
        for key, value in row['scenario'].items():
            check(value, original['scenario'][key], key)


def paired_compare(reference,candidate):
    """Exact two-sided paired sign/McNemar test on discordant binary outcomes."""
    import math
    a={r['seed']:r for r in reference['trials']};b={r['seed']:r for r in candidate['trials']}
    if not a or len(a)!=len(reference['trials']) or len(b)!=len(candidate['trials']):raise ValueError('Nonempty unique seeds required')
    if set(a)!=set(b):raise ValueError('Paired evaluation requires exactly the same seeds')
    for seed in a:
        validate_paired_case(b[seed], a[seed])
    gains=sum(not a[s]['is_success'] and b[s]['is_success'] for s in a)
    losses=sum(a[s]['is_success'] and not b[s]['is_success'] for s in a)
    n=gains+losses
    from scipy.stats import binomtest
    p=float(binomtest(gains,n,.5).pvalue) if n else 1.
    return {'paired_episodes':len(a),'candidate_only_success':gains,'reference_only_success':losses,
        'success_rate_difference':(gains-losses)/len(a),'exact_two_sided_p':p,
        'interpretation':'Exploratory development comparison unless both results are from untouched holdout scenarios.'}


def policy_load(path,cfg=None,device='cpu'):
    selected=resolve_device(device)
    if path is None:return None
    path=Path(path)
    metadata=path.parent/'run.json'
    if not metadata.exists():raise ValueError('Missing run.json next to policy')
    old=json.loads(metadata.read_text())['provenance']
    if old.get('package_generation')!='mac-arm-v4':
        raise ValueError('v4 只加载 v4 策略；v1/v2/v3 策略请保留在对应旧包中，v4 从头训练。')
    if cfg is not None:
        now=provenance(cfg)
        if any(old.get(k)!=now[k] for k in ['config_sha256','model_sha256','source_sha256']):
            raise ValueError('Policy/config/source mismatch. Use the exact v4 source/config that created this policy.')
    from stable_baselines3 import PPO
    return PPO.load(path,device=selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'config/train.yaml'))
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inspect');p.add_argument('--output',type=Path,default=ROOT/'outputs/inspection.json')
    p=sub.add_parser('baseline');p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=10000);p.add_argument('--output',type=Path,default=ROOT/'outputs/baseline')
    p=sub.add_parser('compare');p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('diagnose');p.add_argument('--seed',type=int,default=30000);p.add_argument('--policy',type=Path);p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--output',type=Path,default=ROOT/'outputs/diagnose')
    p=sub.add_parser('benchmark',help='Only execute on the receiving Apple Silicon Mac; no grasp-quality evaluation')
    p.add_argument('--experimental-model',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'outputs/benchmark')
    p.add_argument('--seed',type=int,default=70000);p.add_argument('--envs',type=int,nargs='+',default=[2,4,6])
    p.add_argument('--rollout-samples',type=int,default=1536);p.add_argument('--updates',type=int,default=8)
    p.add_argument('--warmup-updates',type=int,default=2);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--compare-mps',action='store_true',help='Compare explicit MPS with the fastest measured CPU environment count; fail if unavailable')
    for cmd in ['train','evaluate','gui','search','workspace']:
        p=sub.add_parser(cmd);p.add_argument('--experimental-model',action='store_true')
        p.add_argument('--output',type=Path,default=ROOT/'outputs'/cmd)
        p.add_argument('--seed',type=int,default=42 if cmd=='train' else 10000)
        if cmd in ['evaluate','gui']:p.add_argument('--policy',type=Path)
        if cmd in ['train','evaluate','gui']:p.add_argument('--device',choices=['cpu','mps'],default='cpu')
        if cmd=='train':
            p.add_argument('--steps',type=int);p.add_argument('--envs',type=int);p.add_argument('--resume',type=Path);p.add_argument('--smoke',action='store_true')
        if cmd=='evaluate':p.add_argument('--episodes',type=int,default=200)
        if cmd=='search':p.add_argument('--episodes',type=int,default=50)
        if cmd=='gui':p.add_argument('--duration',type=float);p.add_argument('--cad-only',action='store_true')
    args=parser.parse_args();cfg=config(args.config)
    # Freeze provenance before execution. If edited during a run, never attribute the result to the new files.
    run_provenance=provenance(cfg)
    if args.command=='compare':
        result=paired_compare(json.loads(args.reference.read_text()),json.loads(args.candidate.read_text()))
        write_json(args.output,result);print(json.dumps(result,indent=2));return 0
    if args.command=='inspect':
        import mujoco
        from .model import make_model
        m,_,cad=make_model(cfg,visual=True)
        result={**provenance(cfg),'model_status':cad['status'],'dimensions':{'nq':m.nq,'nu':m.nu,'geoms':m.ngeom,'meshes':m.nmesh},
            'cad_parts':len(cad['parts']),'cad_occurrences':len(cad['occurrences']),
            'all_brep_valid':all(x['brep_valid'] for x in cad['parts']),
            'parking_ranges_are_measured':False,'camera_rendering_or_hand_eye_calibration_included':False}
        write_json(args.output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return 0
    if args.command not in ['baseline','diagnose']:permit(args,cfg)
    if args.command=='benchmark':
        from .benchmark import run_benchmark
        return run_benchmark(cfg,args,run_provenance,write_json)
    from .repaired_env import RepairedBowlEnv as BowlEnv
    args.output.mkdir(parents=True,exist_ok=True)
    if args.command in ['baseline','evaluate']:
        if args.episodes<=0:parser.error('episodes must be positive')
        requested=getattr(args,'device','cpu')
        policy=policy_load(getattr(args,'policy',None),cfg,requested);env=BowlEnv(cfg)
        start=time.perf_counter();result=evaluate(env,policy,args.episodes,args.seed)
        result['wall_s']=time.perf_counter()-start;result['provenance']=run_provenance
        result['execution']=device_record(requested,policy.device if policy is not None else None)
        write_json(args.output/'evaluation.json',result)
        if env.frames:
            write_json(args.output/'replay.json',env.frames)
            write_json(args.output/'replay-state.json',{'parking':env.parking.tolist(),'place':env.place.tolist()})
        print(json.dumps({k:v for k,v in result.items() if k not in ['trials','provenance']},ensure_ascii=False,indent=2))
        return 0
    if args.command=='diagnose':
        policy=policy_load(args.policy,cfg,args.device);env=BowlEnv(cfg);result=episode(env,policy,seed=args.seed,options={'trace':True})
        write_json(args.output/'episode.json',{'provenance':provenance(cfg),'execution':device_record(args.device,policy.device if policy is not None else None),**result})
        write_json(args.output/'trace.json',env.rows);write_json(args.output/'replay.json',env.frames)
        print(json.dumps(result,ensure_ascii=False,indent=2));return 0
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
        for height in [.048,.050,.052]:
            for close in [.006,.007,.008]:
                trial=copy.deepcopy(cfg);trial['fixture']['grasp_height_above_table_m']=height;trial['gripper']['close_m']=close
                env=BowlEnv(trial)
                if args.episodes<=0:parser.error('episodes must be positive')
                score=evaluate(env,None,args.episodes,args.seed)
                write_json(args.output/f'height-{height:.3f}-close-{close:.3f}.json',{'provenance':provenance(trial),**score})
                results.append({'height_m':height,'close_m':close,**{k:v for k,v in score.items() if k!='trials'}})
                print(height,close,score['success_rate'],score['failures'],flush=True)
        write_json(args.output/'contact-search.json',{'provenance':provenance(cfg),'candidates':results,'automatically_applied':False});return 0
    if args.command=='gui':
        import mujoco
        import mujoco.viewer
        if platform.system()=='Darwin' and not os.environ.get('MUJOCO_PYTHON_BIN_PATH'):
            print('macOS 原生 passive viewer 请通过 ./run.sh gui 启动（使用 mjpython）。',flush=True)
        policy=policy_load(args.policy,cfg,args.device);env=BowlEnv(cfg,visual=True)
        write_json(args.output/'gui-session.json',{'provenance':run_provenance,'seed':args.seed,
            'policy':str(args.policy) if args.policy else None,'cad_only':args.cad_only,
            'execution':device_record(args.device,policy.device if policy is not None else None)})
        if args.cad_only:
            env.reset(seed=args.seed,options={'parking':[0,0,0]});env.data.qpos[env.qids]=0;mujoco.mj_forward(env.model,env.data)
            frames=[{'t':0.,'qpos':env.data.qpos.tolist(),'qvel':env.data.qvel.tolist(),'ctrl':env.data.ctrl.tolist()}]
        else:
            info=episode(env,policy,seed=args.seed,options={'trace':True})
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
        from .benchmark import TimedPPO
        from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import CheckpointCallback,CallbackList,BaseCallback
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        selected_device=resolve_device(args.device);initialization_started=time.perf_counter()
        n=args.envs if args.envs is not None else cfg['ppo']['n_envs']
        steps=args.steps if args.steps is not None else cfg['ppo']['total_timesteps']
        if n not in range(1,9) or steps<=0:parser.error('Use 1–8 environments and positive steps')
        if (args.output/'run.json').exists():raise SystemExit('此目录已有 run.json；请换新的输出目录，续训也使用新目录以保留旧记录。')
        if args.resume:
            metadata_path=args.resume.parent/'run.json'
            if not metadata_path.exists():raise SystemExit('Checkpoint is missing run.json; cannot verify model/config compatibility')
            old=json.loads(metadata_path.read_text())
            if old['provenance'].get('package_generation')!='mac-arm-v4':
                raise SystemExit('v4 不续训 v1/v2/v3 策略；请在 v4 新建训练。')
            if any(old['provenance'].get(key)!=run_provenance[key] for key in ['config_sha256','model_sha256','source_sha256']):
                raise SystemExit('Resume requires the same config, model and source. Start a new run for changed parameters.')
            if old.get('smoke_only',False)!=args.smoke:raise SystemExit('Smoke and formal training runs cannot be resumed into each other.')
        def factory(rank):
            def make():
                env=Monitor(BowlEnv(cfg),filename=str(args.output/f'env-{rank}.csv'))
                env.reset(seed=args.seed+rank);return env
            return make
        venv=DummyVecEnv([factory(0)]) if n==1 else SubprocVecEnv([factory(i) for i in range(n)],start_method='spawn')
        eval_env=BowlEnv(cfg)
        if args.resume:
            model=TimedPPO.load(args.resume,env=venv,device=selected_device,tensorboard_log=str(args.output/'tensorboard'))
            model.reset_performance()
        else:
            model=TimedPPO('MlpPolicy',venv,device=selected_device,learning_rate=cfg['ppo']['learning_rate'],n_steps=64 if args.smoke else cfg['ppo']['n_steps'],
                batch_size=cfg['ppo']['batch_size'],n_epochs=2 if args.smoke else cfg['ppo']['n_epochs'],gamma=cfg['ppo']['gamma'],gae_lambda=cfg['ppo']['gae_lambda'],
                policy_kwargs={'net_arch':cfg['ppo']['hidden_sizes'],'log_std_init':cfg['ppo']['log_std_init']},target_kl=cfg['ppo']['target_kl'],seed=args.seed,verbose=1,tensorboard_log=str(args.output/'tensorboard'))
        synchronize(model.device)
        run_record={'provenance':run_provenance,'seed':args.seed,'envs':n,'additional_steps':steps,
            'resume':str(args.resume) if args.resume else None,'smoke_only':args.smoke,
            'status':'initializing',
            'execution':{**device_record(args.device,model.device),'parallel_environments':n,
                'vector_env':'DummyVecEnv' if n==1 else 'SubprocVecEnv','start_method':None if n==1 else 'spawn',
                'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()},
            'initialization_s':time.perf_counter()-initialization_started,'evaluation_s':0.}
        write_json(args.output/'run.json',run_record)
        write_json(args.output/'config-used.json',cfg)
        class Metrics(BaseCallback):
            def __init__(self):
                super().__init__();self.completed=0;self.failures=Counter();self.events=Counter();self.stop_reason=None
            def report(self):
                return {'completed_training_episodes':self.completed,'failures':dict(self.failures),
                    'guard_events':dict(self.events),'stopped_for_physics_anomaly':self.stop_reason,
                    'scope':'Training-rollout faults are retained separately from fixed development evaluations.'}
            def _on_step(self):
                for info in self.locals['infos']:
                    if 'episode' in info:
                        self.completed+=1;reason=info.get('failure_reason')
                        if reason:self.failures[reason]+=1
                        self.events.update(e.get('kind','unknown') for e in info.get('guard_events',[]))
                        self.logger.record('task/success',float(info.get('is_success',False)))
                        self.logger.record('task/reachable',float(info.get('reachable',False)))
                        self.logger.record('task/place_error_m',info.get('place_error_m',0.))
                        if reason and (reason.startswith('closure_') or 'nonfinite' in reason or 'physics_warning' in reason):
                            self.stop_reason=reason;write_json(args.output/'first-training-physics-anomaly.json',info)
                        write_json(args.output/'rollout-guard-summary.json',self.report())
                return self.stop_reason is None
        eval_count=3 if args.smoke else cfg['ppo']['evaluation_episodes']
        eval_seed=cfg['ppo']['evaluation_seed']
        evaluation_started=time.perf_counter();reference=evaluate(eval_env,None,eval_count,eval_seed)
        run_record['evaluation_s']+=time.perf_counter()-evaluation_started
        write_json(args.output/'development-baseline.json',{'provenance':run_provenance,**reference})
        class FixedSuccessEvaluation(BaseCallback):
            def __init__(self):
                super().__init__();self.best=None;self.last_eval=-1
            def evaluate_now(self):
                synchronize(self.model.device);evaluation_started=time.perf_counter()
                result=evaluate(eval_env,self.model,eval_count,eval_seed)
                synchronize(self.model.device);run_record['evaluation_s']+=time.perf_counter()-evaluation_started
                result.update({'timesteps':self.num_timesteps,'development_only':True,
                    'paired_vs_baseline':paired_compare(reference,result),'provenance':run_provenance,
                    'execution':device_record(args.device,self.model.device)})
                write_json(args.output/'eval'/f'{self.num_timesteps}.json',result)
                self.logger.record('eval/success_rate',result['success_rate'])
                self.logger.record('eval/mean_reward',float(np.mean([r['return'] for r in result['trials']])))
                key=checkpoint_key(result)
                if self.best is None or key>self.best:
                    self.best=key;self.model.save(args.output/'best_model');write_json(args.output/'best-evaluation.json',result)
                    paired=result['paired_vs_baseline']
                    write_json(args.output/'selection.json',{'recommended_controller':'baseline',
                        'best_policy_development_success_rate':result['success_rate'],
                        'baseline_development_success_rate':reference['success_rate'],
                        'policy_has_development_evidence_of_improvement':paired['success_rate_difference']>0 and paired['exact_two_sided_p']<.05,
                        'reason':'Keep the baseline until a fresh holdout verifies policy improvement and MG23 contact/motion is calibrated.',
                        'hardware_ready':False})
                self.last_eval=self.num_timesteps
                print(f'FIXED_DEV step={self.num_timesteps} success={result["successes"]}/{eval_count}',flush=True)
            def _on_step(self):
                if self.n_calls%eval_freq==0:self.evaluate_now()
                return True
            def _on_training_end(self):
                if not metrics.stop_reason and self.last_eval!=self.num_timesteps:self.evaluate_now()
        freq=max(64 if args.smoke else cfg['ppo']['checkpoint_frequency']//n,1)
        eval_freq=max(256 if args.smoke else cfg['ppo']['evaluation_frequency']//n,1)
        checkpoints=CheckpointCallback(freq,str(args.output),name_prefix='checkpoint')
        metrics=Metrics();callback=CallbackList([metrics,checkpoints,FixedSuccessEvaluation()])
        training_started=time.perf_counter()
        run_record['status']='running';write_json(args.output/'run.json',run_record)
        try:
            model.learn(total_timesteps=steps,reset_num_timesteps=not bool(args.resume),callback=callback,progress_bar=False)
            run_record['status']='stopped_physics_anomaly' if metrics.stop_reason else 'completed'
        except KeyboardInterrupt:
            run_record['status']='interrupted';print('Interrupted; preserving current policy.',flush=True)
        except BaseException as exc:
            run_record['status']='failed';run_record['error']=f'{type(exc).__name__}: {exc}';raise
        finally:
            synchronize(model.device)
            run_record.update({'learning_wall_s':time.perf_counter()-training_started,'actual_total_timesteps':model.num_timesteps,
                'performance':model.performance,
                'performance_note':'Rollout timing includes callbacks and periodic evaluations; benchmark disables these to isolate throughput.'})
            write_json(args.output/'run.json',run_record)
            write_json(args.output/'rollout-guard-summary.json',metrics.report())
            model.save(args.output/'final');venv.close();eval_env.close()
            write_json(args.output/'source-integrity.json',{'unchanged_during_run':provenance(cfg)==run_provenance})
        print('Saved',args.output/'final.zip','— smoke or training completion does not imply grasp acceptance.')
        return 130 if run_record['status']=='interrupted' else (2 if metrics.stop_reason else 0)


if __name__=='__main__':raise SystemExit(main())
