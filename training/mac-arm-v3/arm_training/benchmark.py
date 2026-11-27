"""Mac-only throughput experiments. They do not establish grasp quality."""
import copy
import os
import platform
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from .device import device_record,resolve_device,synchronize


class TimedPPO(PPO):
    """Measure SB3 sampling and updates separately, including device completion."""
    def __init__(self,*args,**kwargs):
        self.reset_performance()
        super().__init__(*args,**kwargs)

    def reset_performance(self):
        self.performance={'rollouts':[],'updates':[]}

    def collect_rollouts(self,*args,**kwargs):
        synchronize(self.device);started=time.perf_counter();before=self.num_timesteps
        result=super().collect_rollouts(*args,**kwargs)
        synchronize(self.device)
        self.performance['rollouts'].append({'wall_s':time.perf_counter()-started,
            'samples':self.num_timesteps-before,'total_timesteps':self.num_timesteps})
        return result

    def train(self):
        synchronize(self.device);started=time.perf_counter()
        result=super().train()
        synchronize(self.device)
        self.performance['updates'].append({'wall_s':time.perf_counter()-started,
            'total_timesteps':self.num_timesteps})
        return result


def run_benchmark(cfg,args,provenance,write_json):
    """Execute only on the receiving Mac, in fresh independent trial environments."""
    if platform.system()!='Darwin' or platform.machine() not in ('arm64','aarch64'):
        raise RuntimeError('benchmark 仅在接收训练包的 Apple Silicon Mac 上执行；本机不做仿真或训练测速。')
    if args.updates<=args.warmup_updates or args.warmup_updates<0:
        raise ValueError('updates must exceed nonnegative warmup-updates')
    if args.repeats<1 or args.rollout_samples<64:
        raise ValueError('Positive repeats and at least 64 rollout samples are required')
    env_counts=list(dict.fromkeys(args.envs))
    if any(n not in range(1,9) or args.rollout_samples%n for n in env_counts):
        raise ValueError('Use 1–8 environments; rollout-samples must be divisible by every environment count')
    # Resolve explicit MPS before any simulations. An unavailable device is not silently skipped.
    resolve_device('cpu')
    if args.compare_mps:resolve_device('mps')
    if args.output.exists() and (args.output/'benchmark.json').exists():
        raise ValueError('benchmark.json already exists; use a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    from .env import BowlEnv
    from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    trial_cfg=copy.deepcopy(cfg)
    result={'purpose':'Throughput only; no reward-based device selection or grasp acceptance.',
        'provenance':provenance,'rollout_samples':args.rollout_samples,'updates':args.updates,
        'warmup_updates':args.warmup_updates,'repeats':args.repeats,
        'requested_timesteps_per_trial':args.rollout_samples*args.updates,
        'gui':False,'fixed_evaluation':False,'tensorboard':False,'trials':[],
        'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'changes_training_default':False}

    def one_trial(requested,n,repeat):
        device=resolve_device(requested);seed=args.seed+repeat
        def factory(rank):
            def make():
                env=Monitor(BowlEnv(trial_cfg));env.reset(seed=seed+rank);return env
            return make
        started=time.perf_counter();venv=None
        try:
            venv=DummyVecEnv([factory(0)]) if n==1 else SubprocVecEnv([factory(i) for i in range(n)],start_method='spawn')
            model=TimedPPO('MlpPolicy',venv,device=device,learning_rate=trial_cfg['ppo']['learning_rate'],
                n_steps=args.rollout_samples//n,batch_size=64,n_epochs=trial_cfg['ppo']['n_epochs'],
                gamma=trial_cfg['ppo']['gamma'],gae_lambda=trial_cfg['ppo']['gae_lambda'],
                policy_kwargs={'net_arch':trial_cfg['ppo']['hidden_sizes'],'log_std_init':trial_cfg['ppo']['log_std_init']},
                target_kl=trial_cfg['ppo']['target_kl'],seed=seed,verbose=0,tensorboard_log=None)
            # No output formats means no stdout/CSV/TensorBoard I/O inside the measured loops.
            model.set_logger(configure(folder=None,format_strings=[]))
            synchronize(device);initialization_s=time.perf_counter()-started
            started=time.perf_counter()
            model.learn(total_timesteps=args.rollout_samples*args.updates,progress_bar=False)
            synchronize(device);learning_wall_s=time.perf_counter()-started
            rollouts=model.performance['rollouts'];updates=model.performance['updates']
            measured_rollouts=rollouts[args.warmup_updates:];measured_updates=updates[args.warmup_updates:]
            samples=sum(row['samples'] for row in measured_rollouts)
            sampling_s=sum(row['wall_s'] for row in measured_rollouts)
            update_s=sum(row['wall_s'] for row in measured_updates)
            return {'repeat':repeat,'seed':seed,'envs':n,'n_steps':model.n_steps,'batch_size':64,
                'device':device_record(requested,model.device),'initialization_s':initialization_s,
                'learning_wall_s_including_warmup':learning_wall_s,'actual_timesteps':model.num_timesteps,
                'measured_samples':samples,'sampling_s':sampling_s,'update_s':update_s,
                'sampling_samples_per_s':samples/max(sampling_s,1e-12),
                'sampling_plus_updates_samples_per_s':samples/max(sampling_s+update_s,1e-12),
                'measured_rollout_count':len(measured_rollouts),'measured_update_count':len(measured_updates),
                'all_rollouts':rollouts,'all_updates':updates}
        finally:
            if venv is not None:venv.close()

    def save():write_json(args.output/'benchmark.json',result)
    save()
    try:
        for n in env_counts:
            for repeat in range(args.repeats):
                row=one_trial('cpu',n,repeat);result['trials'].append(row);save()
                print(f'BENCH cpu envs={n} repeat={repeat} sampling+update={row["sampling_plus_updates_samples_per_s"]:.1f} samples/s',flush=True)
        medians={n:float(np.median([r['sampling_plus_updates_samples_per_s'] for r in result['trials'] if r['envs']==n])) for n in env_counts}
        fastest=max(medians,key=medians.get)
        result['cpu_median_samples_per_s']={str(k):v for k,v in medians.items()}
        result['fastest_measured_cpu_envs']=fastest
        if args.compare_mps:
            for repeat in range(args.repeats):
                row=one_trial('mps',fastest,repeat);result['trials'].append(row);save()
                print(f'BENCH mps envs={fastest} repeat={repeat} sampling+update={row["sampling_plus_updates_samples_per_s"]:.1f} samples/s',flush=True)
            mps=float(np.median([r['sampling_plus_updates_samples_per_s'] for r in result['trials'] if r['device']['requested_policy_device']=='mps']))
            result['mps_to_matching_cpu_throughput_ratio']=mps/medians[fastest]
        result['completed']=True;save()
    except BaseException as exc:
        result['completed']=False;result['error']=f'{type(exc).__name__}: {exc}';save();raise
    print('Saved',args.output/'benchmark.json','— choose devices from measured speed and stability; this is not a success-rate comparison.',flush=True)
    return 0
