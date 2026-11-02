#!/usr/bin/env python3
"""Replay perception inputs in a separate ROS domain, preserving the live camera."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('--domain', type=int, default=83)
    parser.add_argument('--rate', type=float, default=1)
    parser.add_argument('--verify', action='store_true', help='Also verify valid outputs followed by invalidation after the bag ends')
    parser.add_argument('--rviz', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'work/road_perception/replay')
    a = parser.parse_args()
    bag = a.bag.resolve()
    if not (bag/'metadata.yaml').is_file():
        raise FileNotFoundError('Not a rosbag2 directory')
    if not 0 < a.rate <= 2 or not 1 <= a.domain <= 200:
        raise ValueError('Replay rate must be (0,2], domain 1..200')
    if str(a.domain) == os.environ.get('ROS_DOMAIN_ID', '0'):
        raise ValueError('Replay must use a different domain from the live camera')
    a.output.mkdir(parents=True, exist_ok=True)
    metadata = yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    duration = metadata['duration']['nanoseconds'] / 1e9
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(a.domain)
    env['ROAD_PERCEPTION_LOCK'] = str(a.output.resolve()/'replay.lock')
    node = player = checker = viewer = None
    logs = []
    def launch(command, name):
        file = (a.output/(name+'.log')).open('w')
        logs.append(file)
        return subprocess.Popen(command, env=env, cwd=ROOT, stdout=file, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        node = launch([str(ROOT/'start_road_perception.sh'), '--runtime-dir', str(a.output.resolve()/'runtime'),
                       '--ros-args', '-p', 'use_sim_time:=true'], 'perception')
        ready = a.output/'runtime/runtime.json'
        started = time.time()
        deadline = time.monotonic()+40
        while time.monotonic() < deadline:
            if node.poll() is not None:
                raise RuntimeError('Replay perception exited; see perception.log')
            if ready.exists() and ready.stat().st_mtime >= started-0.1:
                break
            time.sleep(0.2)
        else:
            raise TimeoutError('Replay perception did not initialize')
        if a.rviz:
            viewer = launch(['ros2', 'run', 'rviz2', 'rviz2', '-d', str(ROOT/'config/road-perception.rviz'),
                             '--ros-args', '-p', 'use_sim_time:=true'], 'rviz')
        if a.verify:
            check_seconds = max(10, duration/a.rate+8)
            checker = launch(['/usr/bin/python3', str(ROOT/'tools/check_road_perception.py'), '--seconds', str(check_seconds),
                              '--expect-stale', '--sim-time', '--output', str(a.output.resolve()/'check')], 'check')
            time.sleep(1)
        print(f'Replaying in isolated ROS_DOMAIN_ID={a.domain}; live camera domain unchanged', flush=True)
        player = launch(['ros2', 'bag', 'play', str(bag), '--rate', str(a.rate), '--clock', '40', '--delay', '1',
                         '--qos-profile-overrides-path', str(ROOT/'config/road-bag-qos.yaml')], 'player')
        while player.poll() is None:
            if node.poll() is not None:
                raise RuntimeError('Replay perception stopped unexpectedly')
            time.sleep(0.2)
        if player.returncode:
            raise RuntimeError('Bag playback failed; see player.log')
        if checker:
            checker.wait(timeout=max(15, duration/a.rate+12))
            report = json.loads((a.output/'check/health.json').read_text())
            print(json.dumps({'passed': report['passed'], 'valid_status_count': report['valid_status_count'],
                              'last_status_valid': report['last_status']['valid'], 'output': str(a.output)}, indent=2), flush=True)
            if checker.returncode:
                raise RuntimeError('Replay/stale-input checks failed; see check/health.json')
        elif viewer:
            print('Playback ended; press Ctrl+C to close the isolated replay session.', flush=True)
            viewer.wait()
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        for process in [player, checker, viewer, node]:
            stop(process)
        for file in logs:
            file.close()


if __name__ == '__main__':
    raise SystemExit(main())
