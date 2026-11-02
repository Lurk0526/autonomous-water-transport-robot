#!/usr/bin/env python3
"""Measure extra wrist pitch while the verified elbow compensates the image."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import rclpy
from vision.arm_serial import ArmSerial
from tools.calibrate_bowl_follow import TargetFeed,FrameFeed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/'work/bowl-follow/current-trial.json').read_text())
    status=json.loads((ROOT/'work/bowl-follow/status.json').read_text())
    axes=config['tracking_axes'];matrix=np.asarray(config['local_calibration']['jacobian'])
    elbow,wrist=matrix[:,axes.index(2)],matrix[:,axes.index(3)]
    if not elbow[1]<-.3 or not wrist[1]>.3:
        raise RuntimeError('Verified opposite elbow/wrist responses required')
    offsets=status.get('servo_bias_p',config['feedback_deadband_p'])
    report=dict(result='STARTED',steps=[],motion_enabled=args.execute)
    path=ROOT/'work/speed-validation-current/pitch-extension.json'
    def save():path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    arm=None;feed=None;frames=None;rclpy.init()
    try:
        feed=TargetFeed(allow_clipped=True);frames=FrameFeed()
        frames.frame();before=feed.current();report['target']=before
        report['probe_watchdog']='fresh_camera_during_bounded_pose_probe; unique_target_before_and_after'
        save()
        if not args.execute:
            report['result']='TARGET_VISIBLE_NO_MOTION';return
        arm=ArmSerial(config['port']);reference={a:arm.position(a) for a in range(6)}
        report['initial_pose']={str(a):p for a,p in reference.items()}
        if not 1000<=reference[3]<=1175 or not 1620<=reference[2]<=1820:
            raise RuntimeError('Pose differs from observed pitch saturation')
        for index in range(3):
            p2,p3=arm.position(2),arm.position(3)
            goal3=min(1200,p3+65)
            demand3=goal3-p3-offsets['3']['positive']
            if demand3<8:
                break
            demand2=round(-wrist[1]*demand3/elbow[1])
            goal2=p2+demand2+offsets['2']['positive']
            limits=config['axes']['2']
            if (goal3>1200 or not limits['min_p']<=goal2<=limits['max_p']
                    or not 0<goal2-p2<=65 or not 5<=demand2<=50):
                if report['steps']:
                    report['completion_scope']='completed_steps_only; next step was not sent because it failed command bounds'
                    break
                raise RuntimeError('Coordinated pitch probe would exceed bounded range')
            frames.frame();end=time.monotonic()+.95
            arm.move({2:goal2,3:goal3},600,allowed_axes=[2,3])
            while time.monotonic()<end:
                rclpy.spin_once(feed.node,timeout_sec=.02)
                rclpy.spin_once(frames.node,timeout_sec=.01)
                message=frames.latest
                age=(time.time()-message.header.stamp.sec-message.header.stamp.nanosec/1e9
                     if message is not None else float('inf'))
                if not 0<=age<=.35:
                    arm.stop_axes([2,3]);raise RuntimeError('Camera frames lost during coordinated probe')
                feed.records=[]
                data=arm._read_for(.02)
                if data:arm._check_received(data)
            q2,q3=arm.position(2),arm.position(3);arm.stop_axes([2,3])
            if (not 5<=q3-p3<=75 or not 5<=q2-p2<=65
                    or not p3-10<=q3<=goal3+10 or abs(q2-(p2+demand2))>15
                    or not limits['min_p']-10<=q2<=limits['max_p']+10):
                raise RuntimeError('Measured joint motion disagrees with coordinated probe')
            held={a:arm.position(a) for a in (0,1,4,5)}
            if any(abs(p-reference[a])>10 for a,p in held.items()):
                raise RuntimeError('Unexpected held-joint motion')
            after=feed.current()
            predicted=elbow*(q2-p2)+wrist*(q3-p3)
            observed=np.asarray(after['center'])-np.asarray(before['center'])
            if np.linalg.norm(observed)>40 or abs(observed[1]-predicted[1])>22:
                report['rejected_step']=dict(before_p={'2':p2,'3':p3},target_p={'2':goal2,'3':goal3},
                    after_p={'2':q2,'3':q3},before_center=before['center'],after_center=after['center'],
                    predicted_image_shift_px=predicted.tolist(),observed_image_shift_px=observed.tolist())
                raise RuntimeError('Image is inconsistent with coordinated joint motion')
            row=dict(index=index,before_p={'2':p2,'3':p3},target_p={'2':goal2,'3':goal3},
                     after_p={'2':q2,'3':q3},before_center=before['center'],after_center=after['center'],
                     predicted_image_shift_px=predicted.tolist(),observed_image_shift_px=observed.tolist())
            report['steps'].append(row);save();print(json.dumps(row,ensure_ascii=False),flush=True)
            before=after
            if goal3==1200:
                break
        if not report['steps']:
            raise RuntimeError('No additional wrist range was measured')
        report.update(result='MEASURED_PITCH_EXTENSION_PASS',finished_unix=time.time(),
                      maximum_wrist_command_p=max(s['target_p']['3'] for s in report['steps']))
    except BaseException as exc:
        report.update(result='ABORTED',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if arm:arm.close()
        if feed:feed.node.destroy_node()
        if frames:frames.node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        save()


if __name__=='__main__':main()
