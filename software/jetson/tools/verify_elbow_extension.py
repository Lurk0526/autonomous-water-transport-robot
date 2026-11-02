#!/usr/bin/env python3
"""Supervised incremental elbow-range probe from a visible, fixed bowl.

Each target is at most 65 P from actual feedback, within P1600..1900.
The resulting envelope is applied only after measured image/servo agreement.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import rclpy
from vision.arm_serial import ArmSerial
from tools.calibrate_bowl_follow import TargetFeed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--minimum-feedback-p',type=int,help='Measured elbow endpoint to verify, P1650..1750')
    args=parser.parse_args()
    if args.minimum_feedback_p is not None and not 1650<=args.minimum_feedback_p<=1750:
        parser.error('Requested measured endpoint must stay between P1650 and P1750')
    directory=ROOT/'work/speed-validation-current'
    directory.mkdir(parents=True,exist_ok=True)
    report={'steps':[],'motion_enabled':args.execute,'result':'STARTED'}
    path=directory/'elbow-extension.json';arm=None;feed=None
    def save():path.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    rclpy.init()
    try:
        feed=TargetFeed(allow_clipped=True);initial=feed.current()
        report['target']=initial;save()
        if not args.execute:
            report['result']='TARGET_VISIBLE_NO_MOTION';return
        arm=ArmSerial('/dev/ttyUSB0')
        reference={i:arm.position(i) for i in range(6)}
        report['pose_before']={str(a):p for a,p in reference.items()}
        if not 1650<=reference[2]<=1760:
            raise RuntimeError('Elbow pose differs from the observed upper tracking stop')
        before=feed.current();last_good=time.monotonic()
        if args.minimum_feedback_p is not None and before['center'][1]>95:
            # Free vertical image margin before measuring a longer elbow range.
            # This is a separate wrist setup move, excluded from elbow response.
            wrist=arm.position(3)
            delta=-min(45,max(10,round((before['center'][1]-80)/1.15)-9))
            target=wrist+delta
            if not 806<=target<=1076:
                raise RuntimeError('Wrist setup would leave its tested range')
            arm.move({3:target},600);time.sleep(.95)
            measured=arm.position(3);arm.stop_axes([3])
            if not target-15<=measured<=wrist+5:
                raise RuntimeError('Wrist setup feedback mismatch')
            reference[3]=measured
            report['wrist_setup']=dict(before_p=wrist,target_p=target,after_p=measured)
            before=feed.current();last_good=time.monotonic();save()
        for index in range(4):
            actual=arm.position(2)
            if (args.minimum_feedback_p is None and before['center'][1]>=114
                    or args.minimum_feedback_p is not None and actual<=args.minimum_feedback_p):
                break
            if before['center'][1]>=185:
                raise RuntimeError('Not enough remaining image margin for extension probe')
            target=max(1600,actual-65)
            if not 20<=actual-target<=65 or not 1600<=target<=1900:
                raise RuntimeError('Probe would leave bounded extension range')
            arm.move({2:target},600,allowed_axes=[2]);end=time.monotonic()+.95
            while time.monotonic()<end:
                rclpy.spin_once(feed.node,timeout_sec=.02)
                if feed.records:
                    last_good=time.monotonic();feed.records=[]
                if time.monotonic()-last_good>.35:
                    arm.stop_axes([2]);raise RuntimeError('Bowl observation lost during range probe')
                incoming=arm._read_for(.02)
                if incoming:arm._check_received(incoming)
            measured=arm.position(2);arm.stop_axes([2])
            if not target-10<=measured<=actual+5 or not 5<=actual-measured<=75:
                raise RuntimeError('Unexpected measured elbow response during extension')
            others={a:arm.position(a) for a in range(6) if a!=2}
            if any(abs(p-reference[a])>10 for a,p in others.items()):
                raise RuntimeError('Held joint changed during extension')
            after=feed.current()
            shift=after['center'][1]-before['center'][1]
            sensitivity=shift/(actual-measured)
            if not .3<=sensitivity<=2.5:
                raise RuntimeError('Image did not follow measured upward correction')
            row=dict(index=index,before_p=actual,target_p=target,after_p=measured,
                     before_center=before['center'],after_center=after['center'],
                     image_response_px_per_p=sensitivity,stamp_unix=time.time())
            report['steps'].append(row);save();print(json.dumps(row,ensure_ascii=False),flush=True)
            before=after;last_good=time.monotonic()
        if not report['steps']:
            raise RuntimeError('Bowl is already below the test point; no extension measured')
        if args.minimum_feedback_p is not None and report['steps'][-1]['after_p']>args.minimum_feedback_p:
            raise RuntimeError('Requested elbow endpoint was not reached')
        report.update(result='MEASURED_EXTENSION_PASS',finished_unix=time.time(),
                      minimum_command_p=min(s['target_p'] for s in report['steps']),
                      minimum_feedback_p=min(s['after_p'] for s in report['steps']))
    except BaseException as exc:
        report.update(result='ABORTED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if arm:arm.close()
        if feed:feed.node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        save()


if __name__=='__main__':main()
