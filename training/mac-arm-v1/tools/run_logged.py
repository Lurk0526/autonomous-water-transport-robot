"""Run one local experiment and retain its command, log, timing and exit code."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required after --')
    record_path = args.record.resolve()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = record_path.with_suffix('.log')
    cache_path = Path.cwd() / 'outputs' / '.cache'
    cache_path.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, XDG_CACHE_HOME=str(cache_path), PYTHONUNBUFFERED='1')
    record = {
        'command': command,
        'cwd': str(Path.cwd()),
        'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status': 'running',
        'log': str(log_path),
        'environment_overrides': {'XDG_CACHE_HOME': str(cache_path), 'PYTHONUNBUFFERED': '1'},
    }
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    started = time.perf_counter()
    with log_path.open('w') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, env=env)
        for line in child.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        returncode = child.wait()
    record.update(
        ended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        wall_seconds=time.perf_counter() - started,
        exit_code=returncode,
        status='completed' if returncode == 0 else 'failed',
    )
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    raise SystemExit(returncode)


if __name__ == '__main__':
    main()
