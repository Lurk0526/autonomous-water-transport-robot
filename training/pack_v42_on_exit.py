import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def write_status(destination, status, **details):
    state = {'status': status, 'updated_at': datetime.now().astimezone().isoformat(), **details}
    temp = destination / 'packaging-status.tmp'
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temp.replace(destination / 'packaging-status.json')
    print(json.dumps(state, ensure_ascii=False), flush=True)


def supervisor_alive(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'command='],
                            capture_output=True, text=True, check=False)
    fields = result.stdout.strip().split(maxsplit=1)
    return len(fields) == 2 and not fields[0].startswith('Z') and 'Mac执行本轮.sh --large-if-ready' in fields[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    parser.add_argument('run', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('supervisor_pid', type=int)
    args = parser.parse_args()
    package = args.package.resolve()
    run = args.run.resolve()
    destination = args.destination.resolve()
    if run.parent != package / 'outputs' or not run.is_dir():
        parser.error('Select the existing output directory of this package')
    destination.mkdir(parents=True, exist_ok=True)
    exit_file = package / 'Mac训练工作流退出码.txt'
    write_status(destination, 'waiting_for_workflow_exit', run=str(run), supervisor_pid=args.supervisor_pid)
    while not exit_file.is_file() and supervisor_alive(args.supervisor_pid):
        time.sleep(15)
    if not exit_file.is_file():
        time.sleep(10)
    exit_code = exit_file.read_text().strip() if exit_file.is_file() else None
    try:
        if list(destination.glob('*.zip')):
            raise RuntimeError('A ZIP already exists in the dedicated destination; refusing a duplicate')
        if exit_code is not None:
            (run / 'Mac训练工作流退出码.txt').write_text(exit_code + '\n')
        decision = read_json(run / 'training-decision.json')
        phase_notes = []
        for label in ('train-stage1-20k', 'train-stage2-to100k'):
            phase = read_json(run / label / 'run.json')
            if phase:
                phase_notes.append(f'- {label}：{phase.get("status", "未记录")}，累计步数 {phase.get("actual_total_timesteps", "未记录")}。')
        summary = [
            '# Mac v4.2 自动回传记录',
            '',
            f'- 执行目录：`{run.name}`。',
            f'- 工作流退出码：{exit_code if exit_code is not None else "未写入；按中断证据打包"}。',
            f'- 训练决策：{decision.get("reason", "见包内 training-decision.json；缺失则未生成")}。',
            *phase_notes,
            '- 包含已保存的仿真、训练和验收证据。未执行实机测试或 MuJoCo GUI 人工验收。',
        ]
        note = destination / 'Mac回传说明.md'
        note.write_text('\n'.join(summary) + '\n')
        write_status(destination, 'packaging', run=str(run), workflow_exit_code=exit_code)
        subprocess.run([sys.executable, str(package / 'tools' / 'collect_return.py'), str(package),
                        '--run', str(run), '--note', str(note), '--destination', str(destination)],
                       cwd=package, check=True)
        archive, = destination.glob('*.zip')
        checksum = archive.with_suffix('.zip.sha256')
        subprocess.run(['shasum', '-a', '256', '-c', checksum.name], cwd=destination, check=True)
        write_status(destination, 'verified', archive=str(archive), checksum=str(checksum),
                     archive_bytes=archive.stat().st_size, workflow_exit_code=exit_code)
    except Exception as error:
        write_status(destination, 'packaging_failed', error=str(error), run=str(run), workflow_exit_code=exit_code)
        raise


if __name__ == '__main__':
    main()
