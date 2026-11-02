#!/usr/bin/env python3
"""Wait for this service instance, rather than reading an old status file."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
deadline = time.monotonic()+45
while time.monotonic()<deadline:
    try:
        pid = int(subprocess.check_output(['systemctl','--user','show','water-cart-road-perception','--property=MainPID','--value'], text=True).strip())
        runtime = json.loads((ROOT/'work/road_perception/live/runtime.json').read_text())
        file = ROOT/'work/road_perception/live/status.json'
        status = json.loads(file.read_text())
        if pid > 0 and runtime.get('pid') == pid and 0 <= time.time()-file.stat().st_mtime < 2:
            if status.get('valid') and status.get('fusion_valid'):
                print(f"道路感知已就绪：{status['backend']}，RGB 与深度均有效。")
            else:
                print(f"感知节点已启动：{status['backend']}；RGB有效={status.get('valid')}，深度融合有效={status.get('fusion_valid')}。可打开RViz检查画面。")
            raise SystemExit(0)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError):
        pass
    time.sleep(0.3)
raise SystemExit('服务未在45秒内就绪。请运行 ./road_perception.sh status 查看原因；不要重复启动相机。')
