"""Run a fixed set of physical baseline trials and retain completed trials on interruption."""
import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .cli import episode, provenance, summarize_trials, validate_paired_case, write_json
from .model import config
from .repaired_env import RepairedBowlEnv


def run_group(seeds, cfg, output):
    env = RepairedBowlEnv(cfg)
    try:
        for seed in seeds:
            row = episode(env, seed=seed)
            temporary = output / f'{seed}.tmp'
            write_json(temporary, row)
            temporary.replace(output / f'{seed}.json')
    finally:
        env.close()
    return len(seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--case-file', type=Path)
    selector.add_argument('--seed', type=int)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = {}
    if args.case_file:
        if args.episodes is not None:
            parser.error('--episodes applies only to a consecutive seed range')
        cases = json.loads(args.case_file.read_text())
        seeds = [case['seed'] for case in cases]
        reference = {case['seed']: case for case in cases}
    else:
        if args.episodes is None or args.episodes <= 0:
            parser.error('A positive --episodes is required with --seed')
        seeds = list(range(args.seed, args.seed + args.episodes))
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        parser.error('Each trial must have a distinct nonnegative integer seed')

    args.output.mkdir(parents=True, exist_ok=False)
    records = args.output / 'trials'
    records.mkdir()
    cfg = config()
    initial = provenance(cfg)
    started = time.monotonic()
    workers = min(len(seeds), args.workers)
    write_json(args.output / 'plan.json', {'seeds': seeds, 'workers': workers, 'provenance': initial})
    chunks = [seeds[i:i + 5] for i in range(0, len(seeds), 5)] if len(seeds) >= workers * 5 else [[s] for s in seeds]
    completed = 0
    if workers == 1:
        for chunk in chunks:
            completed += run_group(chunk, cfg, records)
            print(f'{completed}/{len(seeds)} trials saved', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(run_group, chunk, cfg, records) for chunk in chunks]
            for future in as_completed(futures):
                completed += future.result()
                print(f'{completed}/{len(seeds)} trials saved', flush=True)

    if provenance(config()) != initial:
        raise RuntimeError('Source, model or configuration changed during evaluation')
    rows = [json.loads((records / f'{seed}.json').read_text()) for seed in seeds]
    if [row['seed'] for row in rows] != seeds:
        raise RuntimeError('Trial record IDs do not match the planned seeds')
    if reference:
        for row in rows:
            validate_paired_case(row, reference[row['seed']])
    result = summarize_trials(rows)
    result.update(provenance=initial, workers=workers, wall_s=time.monotonic()-started,
                  study='paired_failure_regression' if reference else 'consecutive_seed_baseline')
    write_json(args.output / 'evaluation.json', result)
    print(f"{result['successes']}/{result['episodes']} complete. Failures: {result['failures']}", flush=True)


if __name__ == '__main__':
    main()
