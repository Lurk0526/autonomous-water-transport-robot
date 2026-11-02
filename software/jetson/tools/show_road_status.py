#!/usr/bin/env python3
"""Display live service state without presenting an old JSON snapshot as current."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'work/road_perception/live'
try:
    result = json.loads((folder/'status.json').read_text())
    runtime = json.loads((folder/'runtime.json').read_text())
    age = time.time()-(folder/'status.json').stat().st_mtime
except (FileNotFoundError, json.JSONDecodeError):
    result, runtime, age = {}, {}, None
state = subprocess.run(['systemctl','--user','is-active','water-cart-road-perception'], capture_output=True, text=True).stdout.strip()
pid = subprocess.run(['systemctl','--user','show','water-cart-road-perception','--property=MainPID','--value'], capture_output=True, text=True).stdout.strip()
live = state == 'active' and pid.isdigit() and int(pid) == runtime.get('pid') and age is not None and 0 <= age < 2
result.update({'service_state': state, 'snapshot_age_seconds': round(age, 2) if age is not None else None, 'live_snapshot': live})
if not live:
    result.update({'valid': False, 'fusion_valid': False, 'reason': 'No fresh status from the current service instance'})
print(json.dumps(result, ensure_ascii=False, indent=2))
