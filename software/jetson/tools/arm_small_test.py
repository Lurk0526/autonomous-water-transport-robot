#!/usr/bin/env python3
"""One bounded relative base-joint test; does not enable visual following."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.arm_serial import ArmSerial


VENDOR_ACTIONS = Path('/home/nvidia/Desktop/002-开源机械臂Jibot1（STM32）/004-软件工具/04-上位机软件zide/机械臂动作组（磁编码舵机）-爪子版-241217.ini')
VENDOR_SHA256 = '9d8cb501fb3ab64aaad7d41025d18598c292acf1f9610993adc7b6a1d60f98a1'


def resolve_position_protocol(mode_reply, version_reply, vendor_file=VENDOR_ACTIONS, axis=0):
    """Select an identified protocol, never treat an echo as a measured mode.

    The CBM test profile is intentionally limited to the exact installed version
    and the manufacturer's supplied magnetic-encoder position command file.
    It is a single-axis commissioning profile, not automatic-follow calibration.
    """
    if axis not in (0, 4):
        raise RuntimeError('Only base 000 and previously tested wrist rotation 004 are supported')
    prefix = f'#{axis:03d}'.encode('ascii')
    modes = re.findall(prefix + rb'PMOD([1-8])!', mode_reply)
    if modes:
        mode = int(modes[-1])
        if mode > 4:
            raise RuntimeError('Motor/continuous-rotation mode is not allowed')
        return {'protocol': 'legacy_position_mode', 'mode_confirmed': True,
                'mode': mode, 'base_test_envelope_p': [510, 2490]}
    if mode_reply.strip() not in (b'', prefix + b'PMOD!'):
        raise RuntimeError('Unexpected mode response; no motion allowed')
    if prefix + b'@ ZL-ZServo_AD_CBM V2.1.16STG!' not in version_reply:
        raise RuntimeError('Unrecognised servo version; no position test profile')
    data = vendor_file.read_bytes()
    if hashlib.sha256(data).hexdigest() != VENDOR_SHA256:
        raise RuntimeError('Manufacturer position-protocol evidence changed or missing')
    commands = re.findall(rb'#000P(\d{4})T(\d{4})!', data)
    if not commands or {int(p) for p, _ in commands} != {1000, 1500, 2000}:
        raise RuntimeError('Invalid manufacturer magnetic-encoder position examples')
    return {'protocol': 'ZL-ZServo_AD_CBM_V2.1.16STG_vendor_position_commands',
            'mode_confirmed': False, 'mode_query': 'unsupported_or_no_mode_response',
            'vendor_file': str(vendor_file), 'vendor_sha256': VENDOR_SHA256,
            'base_test_envelope_p': [1010, 1990] if axis == 0 else [1480, 1580],
            'single_axis_commissioning_only': True, 'axis': axis}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='/dev/ttyUSB0')
    parser.add_argument('--execute', action='store_true', help='Otherwise probe only, never move')
    parser.add_argument('--delta', type=int, default=10)
    parser.add_argument('--axis', type=int, choices=(0, 4), default=0)
    args = parser.parse_args()
    if not 1 <= abs(args.delta) <= (10 if args.axis == 0 else 30):
        parser.error('Relative offset limit: 10 P for base 000; 30 P for wrist rotation 004')
    axis = args.axis
    directory = ROOT / 'work/arm-small-test'
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / f'{datetime.now():%Y%m%d-%H%M%S}.json'
    report = {'started_unix': time.time(), 'port': args.port, 'axis': axis,
              'relative_delta_p': args.delta, 'duration_ms': 2000,
              'execute': args.execute, 'events': [], 'result': 'starting'}
    arm = None

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    def event(kind, **details):
        record = dict(kind=kind, unix=time.time(), **details)
        report['events'].append(record)
        save()
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def monitor(baseline, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            # Keep receiving reset/external-command indications. The factory
            # UART bridge echoes servo replies back onto the bus; continuous
            # PRAD requests are deliberately isolated from the timed move.
            incoming = arm._read_for(min(0.05, max(0, end-time.monotonic())))
            if incoming:
                arm._check_received(incoming)
                event('motion_receive', raw=incoming.decode('ascii', errors='replace'))
        value = arm.position(axis)
        event('measured_endpoint', p=value)
        if abs(value-baseline) > (20 if axis == 0 else 40):
            raise RuntimeError('Measured motion exceeded the bounded test window')
        return value

    try:
        save()
        arm = ArmSerial(args.port)
        event('connected_without_reset_detected', **arm.connection_report)
        arm.stop()
        reply = arm._read_for(0.15)
        arm._check_received(reply)
        event('initial_stop', raw=reply.decode('ascii', errors='replace'))
        before = {str(axis): arm.position(axis) for axis in range(6)}
        time.sleep(0.2)
        again = {str(axis): arm.position(axis) for axis in range(6)}
        event('startup_positions', first=before, second=again)
        if any(abs(before[k]-again[k]) > 3 for k in before):
            raise RuntimeError('Joints are not stationary; test aborted')
        arm.serial.reset_input_buffer()
        arm._send(f'#{axis:03d}PMOD!')
        mode_reply = arm._read_for(0.3)
        arm._check_received(mode_reply)
        arm._send(f'#{axis:03d}PVER!')
        version_reply = arm._read_for(0.3)
        arm._check_received(version_reply)
        profile = resolve_position_protocol(mode_reply, version_reply, axis=axis)
        event('position_protocol_identified', mode_raw=mode_reply.decode('ascii', errors='replace'),
              version_raw=version_reply.decode('ascii', errors='replace'), **profile)
        if not args.execute:
            report['result'] = 'probe_passed_no_motion'
            return
        baseline = again[str(axis)]
        target = baseline + args.delta
        low, high = profile['base_test_envelope_p']
        if not low <= baseline <= high or not low <= target <= high:
            raise RuntimeError('Relative test is outside the identified profile test envelope')
        # This one-shot calibration path deliberately does not loosen the follower's gates.
        command = f'#{axis:03d}P{target:04d}T2000!'
        event('sending_relative_test', command=command, baseline_p=baseline)
        arm._send(command)
        outward = monitor(baseline, 2.3)
        arm.stop()
        event('outward_stopped', measured_p=outward)
        for other in range(6):
            if other != axis and abs(arm.position(other)-again[str(other)]) > 5:
                raise RuntimeError(f'Uncommanded joint {other} moved; return aborted')
        command = f'#{axis:03d}P{baseline:04d}T2000!'
        event('sending_return', command=command)
        arm._send(command)
        returned = monitor(baseline, 2.3)
        arm.stop()
        event('return_stopped', measured_p=returned)
        report.update(result='commands_completed_requires_visual_confirmation', baseline_p=baseline,
                      outward_measured_p=outward, return_measured_p=returned,
                      return_within_8p=abs(returned-baseline) <= 8,
                      outward_measured_change_p=outward-baseline,
                      outward_feedback_changed_in_requested_direction=(outward-baseline)*args.delta >= 3*abs(args.delta),
                      position_protocol=profile)
        if not report['outward_feedback_changed_in_requested_direction']:
            report['result'] = 'no_measured_outward_motion'
        elif not report['return_within_8p']:
            report['result'] = 'return_outside_tolerance'
    except BaseException as exc:
        report.update(result='aborted', error=f'{type(exc).__name__}: {exc}')
        if hasattr(exc, 'connection_report'):
            report['connection_report'] = exc.connection_report
            print(json.dumps(exc.connection_report, ensure_ascii=False), flush=True)
        raise
    finally:
        if arm:
            try:
                arm.close()
                event('stop_sent_and_port_closed')
            except Exception as exc:
                report['stop_error'] = str(exc)
        report['finished_unix'] = time.time()
        save()
        print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
