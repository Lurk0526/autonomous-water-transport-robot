#!/bin/bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/mac-arm-v4.2"
exec "$package_dir/.venv/bin/python" -u - "$package_dir" "$@" <<'PY'
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time


def read(path):
    try:
        return path.read_text()
    except OSError:
        return ''


def record(path):
    try:
        return json.loads(read(path))
    except ValueError:
        return {}


def bar(label, count, target):
    if count is None:
        return f'{label}  [{"░" * 30}]  等待首条进度 / {target:,}'
    fraction = min(count / max(target, 1), 1)
    width = int(30 * fraction)
    return f'{label}  [{"█" * width}{"░" * (30 - width)}]  {count:,}/{target:,}  {fraction:5.1%}'


def render():
    log = read(package / 'Mac训练工作流日志.log')
    stages = re.findall(r'^([\w-]+): (/.+/logs/[^\n]+\.log)$', log, re.M)
    if not stages:
        return ['等待工作流记录当前阶段…']
    stage, filename = stages[-1]
    path = Path(filename)
    run = path.parent.parent
    contents = read(path)
    lines = [f'当前阶段：{stage}']
    if stage in {'train-stage1-20k', 'train-stage2-to100k'}:
        state = record(run / stage / 'run.json')
        status = state.get('status', 'initializing')
        evaluations = list(re.finditer(r'^EVAL (\d+)/(\d+) success=(\d+)', contents, re.M))
        if status == 'initializing':
            count, target = map(int, evaluations[-1].groups()[:2]) if evaluations else (None, 100)
            lines = [bar('训练前固定开发集复评', count, target), '尚未更新 PPO 权重。复评每 20 例写一次进度。']
        else:
            start = record(run / 'stage2-budget.json').get('resumed_final_checkpoint_steps', 0) if stage == 'train-stage2-to100k' else 0
            steps = [int(value) for value in re.findall(r'\|\s*total_timesteps\s*\|\s*(\d+)\s*\|', contents)]
            checkpoints = (run / stage).glob('checkpoint_*_steps.zip')
            steps.extend(int(checkpoint.stem.split('_')[-2]) for checkpoint in checkpoints)
            if 'actual_total_timesteps' in state:
                steps.append(state['actual_total_timesteps'])
            done = max(0, max(steps, default=start) - start)
            lines = [bar('PPO 本阶段步数', done, state.get('additional_steps', 80000 if start else 20000))]
            fixed = contents.rfind('FIXED_DEV step=')
            rollout = contents.rfind('total_timesteps')
            if len(evaluations) > 5 and evaluations[-1].start() > max(fixed, rollout):
                count, target, successes = map(int, evaluations[-1].groups())
                lines.append(f'策略评估 {count}/{target}，已成功 {successes} 例；评估时采样暂停。')
            lines.append(f'状态：{status} · 设备：{state.get("execution", {}).get("actual_policy_device", "待记录")}')
    else:
        progress = re.findall(r'^EVAL (\d+)/(\d+) success=(\d+)', contents, re.M)
        if progress:
            count, total, _ = map(int, progress[-1])
            lines.append(bar('评估场景', count, total))
    decision = record(run / 'training-decision.json')
    exit_code = read(package / 'Mac训练工作流退出码.txt').strip()
    if exit_code and decision:
        lines.append(f'工作流已退出（代码 {exit_code}）。详见 {run.name}/training-decision.json。')
    lines.extend(['', f'日志：{path.name}  ·  {datetime.now():%H:%M:%S}', 'Ctrl+C 只关闭进度窗口，不影响训练。'])
    return lines


package = Path(sys.argv[1])
if '--once' in sys.argv[2:]:
    print('\n'.join(render()))
    raise SystemExit
print('\x1b]0;AstraGO v4.2 训练进度\x07', end='')
try:
    while True:
        print('\x1b[H\x1b[J' + 'AstraGO v4.2 · 训练进度\n\n' + '\n'.join(render()), flush=True)
        time.sleep(2)
except KeyboardInterrupt:
    print('\n已关闭进度监视。')
PY
