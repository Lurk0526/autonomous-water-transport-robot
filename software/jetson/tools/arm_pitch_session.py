#!/usr/bin/env python3
"""Staged wrist-pitch positioning with a fresh camera snapshot after each step."""
import argparse
import hashlib
import json
from pathlib import Path
import select
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image as PILImage
import rclpy
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from vision.arm_serial import ArmSerial
from tools.arm_small_test import VENDOR_ACTIONS, VENDOR_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    destination=parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--target-p', type=int)
    destination.add_argument('--lower-by-p', type=int)
    parser.add_argument('--duration-ms', type=int, default=2000)
    args = parser.parse_args()
    if args.target_p is not None and not 850 <= args.target_p <= 1500:
        parser.error('Wrist-pitch target must stay inside the limited commissioning range')
    if args.lower_by_p is not None and not 1 <= args.lower_by_p <= 500:
        parser.error('Relative lowering must be between 1 and 500 P')
    if not 1000 <= args.duration_ms <= 3000:
        parser.error('Each pitch stage must take between 1000 and 3000 ms')
    directory = ROOT / 'work/arm-small-test' / f'pitch-{time.strftime("%Y%m%d-%H%M%S")}'
    directory.mkdir(parents=True)
    report = {'target_p': args.target_p, 'events': [], 'purpose': 'user_requested_staged_head_pitch'}
    arm = None
    node = None

    def event(kind, **values):
        row = dict(kind=kind, unix=time.time(), **values)
        report['events'].append(row)
        (directory/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)

    rclpy.init()
    node = rclpy.create_node('arm_pitch_camera_check')
    frames = []
    sub = node.create_subscription(Image, '/openmv/image_raw', lambda m: frames.append(m), qos_profile_sensor_data)

    def snapshot(name):
        after = time.time()
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            while frames:
                msg = frames.pop()
                frames.clear()
                stamp = msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
                if stamp < after or msg.encoding != 'rgb8':
                    continue
                rows=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)
                rgb=rows[:,:msg.width*3].reshape(msg.height,msg.width,3).copy()
                path=directory/f'{name}.png'
                PILImage.fromarray(rgb).save(path)
                return str(path)
        raise RuntimeError('Fresh camera image unavailable; further pitch steps inhibited')

    try:
        if hashlib.sha256(VENDOR_ACTIONS.read_bytes()).hexdigest()!=VENDOR_SHA256:
            raise RuntimeError('Manufacturer position command evidence differs')
        arm=ArmSerial('/dev/ttyUSB0')
        arm.stop();time.sleep(.15)
        initial={str(i):arm.position(i) for i in range(6)}
        time.sleep(.2)
        again={str(i):arm.position(i) for i in range(6)}
        report['measured_start_positions']=again
        event('measured_start_positions', positions=again)
        if any(abs(initial[k]-again[k])>3 for k in initial):
            raise RuntimeError('Arm not stationary')
        if any(abs(again[str(i)]-1500)>80 for i in (0,1,2,4,5)):
            raise RuntimeError('Other joints are outside the confirmed near-home fixture pose')
        current=again['3']; baseline=current
        target_p=args.target_p if args.target_p is not None else baseline-args.lower_by_p
        if not 850 <= target_p <= 1500 or not 0 <= baseline-target_p <= 500:
            raise RuntimeError('Requested pitch exceeds 500 P total or is in the wrong direction')
        arm._send('#003PVER!');identity=arm._read_for(.3);arm._check_received(identity)
        if b'#003@ ZL-ZServo_AD_CBM V2.1.16STG!' not in identity:
            raise RuntimeError('Unrecognised wrist-pitch servo')
        report.update(initial=again, baseline_p=baseline, target_p=target_p, connection=arm.connection_report)
        event('ready', current_p=current, snapshot=snapshot('before'), directory=str(directory))
        stages=0
        last_input=time.monotonic()
        while time.monotonic()-last_input < 90:
            ready,_,_=select.select([sys.stdin],[],[],.05)
            if not ready:
                incoming=arm._read_for(.01)
                if incoming:arm._check_received(incoming)
                continue
            command=sys.stdin.readline().strip()
            last_input=time.monotonic()
            if command in ('finish','stop',''):
                break
            if command!='step':
                event('command_rejected', command=command)
                continue
            measured=arm.position(3)
            if abs(measured-current)>5:
                raise RuntimeError('Wrist moved unexpectedly between stages')
            current=measured
            if current-target_p<=10:
                event('target_reached', current_p=current)
                continue
            if stages>=6:
                raise RuntimeError('Maximum number of pitch stages reached')
            target=max(target_p,current-100)
            event('moving_one_stage', current_p=current, target_p=target, duration_ms=args.duration_ms)
            arm._send(f'#003P{target:04d}T{args.duration_ms:04d}!')
            deadline=time.monotonic()+args.duration_ms/1000+0.3
            while time.monotonic()<deadline:
                incoming=arm._read_for(min(.05,max(0,deadline-time.monotonic())))
                if incoming:arm._check_received(incoming)
            measured=arm.position(3)
            arm.stop()
            if not target-8<=measured<=current+3 or abs(measured-target)>16:
                raise RuntimeError(f'Pitch endpoint differs from target: {measured} vs {target}')
            for i in (0,1,2,4,5):
                if abs(arm.position(i)-again[str(i)])>6:
                    raise RuntimeError(f'Uncommanded joint {i} changed')
            current=measured;stages+=1
            event('stage_stopped', stage=stages, current_p=current,
                  total_measured_delta_p=current-baseline, snapshot=snapshot(f'stage-{stages}'))
        report.update(result='stopped_at_measured_pose', final_p=current)
    except BaseException as exc:
        report.update(result='aborted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if arm:
            arm.close()
            report['stop_sent_and_port_closed']=True
        if node:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        report['finished_unix']=time.time()
        (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(f'Report: {directory}/report.json',flush=True)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
    main()
