#!/usr/bin/env python3
"""One slow wrist-pitch adjustment from the confirmed near-home pose."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.arm_serial import ArmSerial
from tools.arm_small_test import VENDOR_ACTIONS, VENDOR_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'work/arm-small-test' / f'nod-{time.strftime("%Y%m%d-%H%M%S")}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {'axis': 3, 'delta_p': -60, 'duration_ms': 3000,
              'execute': args.execute, 'events': [], 'motion_sent': False}
    arm = None

    def event(kind, **values):
        row = dict(kind=kind, unix=time.time(), **values)
        report['events'].append(row)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)

    try:
        if hashlib.sha256(VENDOR_ACTIONS.read_bytes()).hexdigest() != VENDOR_SHA256:
            raise RuntimeError('Manufacturer position-protocol evidence differs')
        arm = ArmSerial('/dev/ttyUSB0')
        event('connection', **arm.connection_report)
        arm.stop()
        time.sleep(.15)
        first = {str(i): arm.position(i) for i in range(6)}
        time.sleep(.2)
        before = {str(i): arm.position(i) for i in range(6)}
        if any(abs(first[k]-before[k]) > 3 for k in before):
            raise RuntimeError('Arm is not stationary')
        if any(abs(v-1500) > 80 for v in before.values()):
            raise RuntimeError('Nod commissioning requires the confirmed near-home pose')
        arm._send('#003PVER!')
        identity = arm._read_for(.3)
        arm._check_received(identity)
        if b'#003@ ZL-ZServo_AD_CBM V2.1.16STG!' not in identity:
            raise RuntimeError('Unrecognised wrist-pitch servo')
        baseline = before['3']
        target = baseline - 60
        report.update(before=before, baseline_p=baseline, target_p=target,
                      vendor_sha256=VENDOR_SHA256, mode_query_confirmed=False)
        event('ready', baseline_p=baseline, target_p=target, identity=identity.decode(errors='replace'))
        if not args.execute:
            report['result'] = 'probe_only_no_motion'
            return
        command = f'#003P{target:04d}T3000!'
        event('sending_nod', command=command)
        arm._send(command)
        report['motion_sent'] = True
        deadline = time.monotonic() + 3.3
        while time.monotonic() < deadline:
            incoming = arm._read_for(min(.05, max(0, deadline-time.monotonic())))
            if incoming:
                arm._check_received(incoming)
                event('motion_receive', raw=incoming.decode(errors='replace'))
        after = {str(i): arm.position(i) for i in range(6)}
        arm.stop()
        event('stopped_at_new_pose', positions=after)
        if abs(after['3']-baseline) > 80:
            raise RuntimeError('Wrist exceeded the commissioning travel window')
        if any(abs(after[str(i)]-before[str(i)]) > 5 for i in range(6) if i != 3):
            raise RuntimeError('Uncommanded joint movement detected')
        report['after'] = after
        if baseline-after['3'] >= 5:
            report['result'] = 'pitch_changed_requires_visual_direction_confirmation'
        else:
            report['result'] = 'no_measured_pitch_change'
    except BaseException as exc:
        report.update(result='aborted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if arm:
            try:
                arm.close()
                report['stop_sent_and_port_closed'] = True
            except Exception as exc:
                report['stop_error'] = str(exc)
        report['finished_unix'] = time.time()
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
