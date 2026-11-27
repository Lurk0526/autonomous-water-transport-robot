"""Paired held-out comparison using the original v4.2 physics and fixed checkpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
from datetime import datetime

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))

EXPECTED = {
    'package_version':'0.4.2',
    'source_sha256':'a3f4c562b8f683ef28bab3c4aa2b1036f01b6c33fa3fdcbf656a772395d82ed7',
    'model_sha256':'9f15bb7c008a886461f88e7f544b6c98c82b8d595e5688ea1e7f9c977cf90733',
    'config_sha256':'bcf242d0f6a29ab2f713b881f169bdc275582ea78775b209ced0e9576ca8678e',
}
CHECKPOINT_SHA256 = '9c20cf468bd360035e39001c59ba2a970923129a47dee804bc3ed2ce9fba04a2'
SEEDS = tuple(range(114000, 114200))


def require_provenance(value):
    for key, expected in EXPECTED.items():
        if value.get(key) != expected:
            raise ValueError(f'{key} does not match the archived Mac v4.2 run')


def validate(run):
    from arm_training.cli import provenance
    from arm_training.model import ROOT, config
    if ROOT.resolve() != PROJECT.resolve():
        raise ValueError('Run from the original mac-arm-v4.2 directory')
    cfg = config()
    now = provenance(cfg)
    require_provenance(now)
    checkpoint = run/'train-stage2-to100k/best_model.zip'
    with checkpoint.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('best_model.zip does not match the frozen 101472-step checkpoint')
    previous = json.loads((checkpoint.parent/'run.json').read_text())
    require_provenance(previous['provenance'])
    if previous['status'] != 'completed':
        raise ValueError('The original training run is incomplete')
    return cfg, now, checkpoint


def run_chunk(cfg, checkpoint, seeds, folder):
    import torch
    from arm_training.cli import episode, policy_load, write_json
    from arm_training.repaired_env import RepairedBowlEnv

    torch.set_num_threads(1)
    policy = policy_load(checkpoint,cfg,'cpu') if checkpoint else None
    env = RepairedBowlEnv(cfg)
    try:
        for seed in seeds:
            row = episode(env,policy,seed=seed)
            temp = folder/f'{seed}.tmp'
            write_json(temp,row)
            temp.replace(folder/f'{seed}.json')
    finally:
        env.close()
    return len(seeds)


def evaluate(cfg, provenance_value, checkpoint, workers, output):
    from arm_training.cli import summarize_trials, write_json
    trials = output/'trials'
    trials.mkdir(parents=True,exist_ok=True)
    # Distribute independent fixed seeds. The same checkpoint is loaded per worker.
    chunks = [SEEDS[i::workers] for i in range(workers)]
    complete = 0
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for done in as_completed([pool.submit(run_chunk,cfg,checkpoint,chunk,trials) for chunk in chunks]):
            complete += done.result()
            print(f'{output.name}: {complete}/{len(SEEDS)} trials saved',flush=True)
    rows = [json.loads((trials/f'{seed}.json').read_text()) for seed in SEEDS]
    if [row['seed'] for row in rows] != list(SEEDS):
        raise ValueError('Trial IDs differ from the sealed range')
    result = summarize_trials(rows)
    result.update(provenance=provenance_value,policy='PPO' if checkpoint else 'zero_residual',
                  checkpoint_sha256=CHECKPOINT_SHA256 if checkpoint else None)
    write_json(output/'evaluation.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=PROJECT/'outputs/mac-v4.2-check-20261120-212802')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, choices=range(1,9), default=6)
    parser.add_argument('--check-only', action='store_true', help='Verify files without loading the checkpoint or running any episodes')
    args = parser.parse_args()
    cfg, bound, checkpoint = validate(args.run.resolve())
    if args.check_only:
        print('Original v4.2 and frozen checkpoint match. Zero episodes run.')
        return
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise SystemExit('Use the original native arm64 macOS environment for the blind comparison')

    from arm_training.cli import paired_compare, write_json
    from arm_training.training_gate import _serious_events
    output = (args.output or PROJECT/'outputs'/('mac-v4.2-holdout-'+datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    print(output,flush=True)
    write_json(output/'holdout-plan.json',{'seeds':list(SEEDS),'provenance':bound,'ppo_timesteps':101472,
                                          'ppo_sha256':CHECKPOINT_SHA256,'workers':args.workers,
                                          'device':'cpu','primary_metric':'full_pick_hold_place'})
    baseline = evaluate(cfg,bound,None,args.workers,output/'baseline')
    validate(args.run.resolve())
    ppo = evaluate(cfg,bound,checkpoint,args.workers,output/'ppo-101472')
    validate(args.run.resolve())
    paired = paired_compare(baseline,ppo)
    paired.update(baseline_successes=baseline['successes'],ppo_successes=ppo['successes'],
                  baseline_serious_events=_serious_events(baseline['trials']),
                  ppo_serious_events=_serious_events(ppo['trials']),
                  baseline_wilson95=baseline['success_rate_wilson95'],
                  ppo_wilson95=ppo['success_rate_wilson95'])
    paired['ppo_improvement_on_this_holdout'] = bool(
        paired['success_rate_difference']>0 and paired['exact_two_sided_p']<.05 and not paired['ppo_serious_events'])
    paired['interpretation'] = 'Paired independent simulation results for the frozen v4.2 rule controller and checkpoint.'
    write_json(output/'paired-holdout.json',paired)
    print(json.dumps({k:v for k,v in paired.items() if not k.endswith('serious_events')},indent=2),flush=True)


if __name__ == '__main__':
    main()
