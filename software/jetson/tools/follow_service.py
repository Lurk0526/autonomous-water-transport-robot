#!/usr/bin/env python3
"""Cancellable GUI start/stop controller. Opening the GUI never starts motors."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
UNIT='jibot-bowl-follow'


def stop_service():
    active=subprocess.run(['systemctl','--user','show',UNIT,'-p','LoadState','--value'],
                          capture_output=True,text=True,timeout=5)
    if active.stdout.strip()=='not-found':return
    subprocess.run(['systemctl','--user','stop',UNIT],check=True,timeout=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','stop'])
    args=parser.parse_args()
    if args.action=='stop':
        stop_service();print('已停止，点击绿色按钮即可重新开启。',flush=True);return 0
    child=None
    def cancelled(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,cancelled)
    signal.signal(signal.SIGINT,cancelled)
    try:
        print('正在检查目标碗、控制板和当前位置……',flush=True)
        started=time.time()
        child=subprocess.Popen([str(ROOT/'start_bowl_tracking.sh'),'--restart'],
                               cwd=ROOT,start_new_session=True)
        result=child.wait()
        if result:raise RuntimeError('未能恢复跟踪；请查看上方具体原因。')
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            try:status=json.loads((ROOT/'work/bowl-follow/status.json').read_text())
            except (OSError,ValueError):status={}
            if status.get('updated_unix',0)>=started:
                if status.get('motion_enabled'):
                    print('跟踪已开启；再次点击“停止跟踪”即可暂停。',flush=True);return 0
                if str(status.get('actuator_state','')).startswith('fault:'):
                    raise RuntimeError(status['actuator_state'])
            time.sleep(.1)
        raise RuntimeError('启动超时，未收到机械臂控制节点确认。')
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
        stop_service()
        print('启动已取消，机械臂保持停止。',flush=True);return 2
    except Exception as exc:
        print(f'开启失败：{exc}',flush=True)
        stop_service();return 1


if __name__=='__main__':raise SystemExit(main())
