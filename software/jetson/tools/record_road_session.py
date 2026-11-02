#!/usr/bin/env python3
"""Record bounded raw perception inputs with calibration/TF and provenance."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ['/camera/color/image_raw', '/camera/depth/image_rect_raw',
          '/camera/color/camera_info', '/camera/depth/camera_info', '/tf_static']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=15)
    p.add_argument('--output', type=Path)
    p.add_argument('--extra-topic', action='append', default=[], help='Additional read-only audit topics')
    p.add_argument('--purpose', default='Bench sensor capture; not a labeled field-validation dataset')
    args = p.parse_args()
    if not 1 <= args.seconds <= 600:
        raise ValueError('Record duration must be 1..600 seconds')
    folder = args.output or ROOT / 'work/road_perception/recordings' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    if folder.exists():
        raise FileExistsError('Recording directory already exists')
    folder.parent.mkdir(parents=True, exist_ok=True)
    expected = args.seconds * 30 * (640*480*3 + 640*400*2) * 1.3
    if shutil.disk_usage(folder.parent).free < expected + 1024**3:
        raise RuntimeError('Insufficient free disk for bounded RGB/depth recording')
    topics = list(dict.fromkeys(TOPICS+args.extra_topic))
    command = ['ros2', 'bag', 'record', '--storage', 'sqlite3', '--output', str(folder),
               '--max-bag-size', str(256*1024**2), '--max-cache-size', str(64*1024**2),
               '--qos-profile-overrides-path', str(ROOT/'config/road-bag-qos.yaml'), '--topics', *topics]
    print('Recording', args.seconds, 'seconds into', folder, flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    start = time.monotonic()
    try:
        while process.poll() is None and time.monotonic() - start < args.seconds:
            if shutil.disk_usage(folder.parent).free < 512*1024**2:
                raise RuntimeError('Recording stopped: less than 512 MiB free')
            time.sleep(0.2)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
    counts = {}
    for db in sorted(folder.glob('*.db3')):
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
            for name, count in connection.execute('SELECT topics.name, COUNT(messages.id) FROM topics LEFT JOIN messages ON topics.id=messages.topic_id GROUP BY topics.name'):
                counts[name] = counts.get(name, 0) + count
    report = {'topics': counts, 'requested_seconds': args.seconds, 'elapsed_seconds': time.monotonic()-start,
              'domain_id': os.environ.get('ROS_DOMAIN_ID', '0'), 'passed': all(counts.get(t, 0) > 0 for t in TOPICS),
              'purpose': args.purpose, 'required_topics': TOPICS, 'requested_topics': topics,
              'missing_extra_topics': [t for t in args.extra_topic if counts.get(t,0)==0]}
    if folder.exists():
        (folder/'session.json').write_text(json.dumps(report, indent=2)+'\n')
        shutil.copy2(ROOT/'config/road-perception.yaml', folder/'road-perception.yaml')
        shutil.copy2(ROOT/'work/models/road/road-4060-v1/manifest.json', folder/'model-manifest.json')
        info = subprocess.run(['ros2','bag','info',str(folder)], capture_output=True, text=True)
        (folder/'bag-info.txt').write_text(info.stdout+info.stderr)
    print(json.dumps(report, indent=2), flush=True)
    return not report['passed']


if __name__ == '__main__':
    raise SystemExit(main())
