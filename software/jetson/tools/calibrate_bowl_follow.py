#!/usr/bin/env python3
"""Measure local target-image response within ±30 P on joints 000 and 003."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import signal
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import rclpy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
import numpy as np
from PIL import Image as PILImage
from vision.arm_serial import ArmSerial
from vision.follow_calibration import target_measurement,stable_summary,estimate_jacobian,registered_center_shift,recent_target_samples,measurement_anchor,segmented_move
from tools.arm_small_test import VENDOR_ACTIONS,VENDOR_SHA256
from vision.target_association import select_locked_target
from vision.motion_limits import pose_envelopes, runtime_profile, validate_current_pose


class TargetFeed:
    def __init__(self,allow_clipped=False):
        self.allow_clipped=allow_clipped
        self.locked_sample=None
        self.node=rclpy.create_node('bowl_follow_direction_calibration')
        self.records=[]
        self.sub=self.node.create_subscription(String,'/openmv/bowl/detections',self.receive,1)

    def receive(self,msg):
        try:
            record=json.loads(msg.data)
            # A low-score background proposal must not veto an otherwise unique
            # high-confidence calibration bowl. Multiple trusted bowls still fail.
            detections=record.get('detections',[])
            record=dict(record,detections=[d for d in detections
                        if isinstance(d,dict) and isinstance(d.get('confidence'),(int,float))
                        and d['confidence']>=.85])
            if self.locked_sample:
                record=select_locked_target(record,self.locked_sample['center'],self.locked_sample['bbox'],.85)
            sample=target_measurement(record,time.time(),allow_clipped=self.allow_clipped)
            if sample:
                self.locked_sample=sample
                sample['weights_sha256']=record.get('weights_sha256')
                self.records.append(sample)
        except ValueError:pass

    def sample(self,timeout=12):
        after=time.time();deadline=time.monotonic()+timeout
        self.records=[];samples=[];seen=set()
        while time.monotonic()<deadline:
            rclpy.spin_once(self.node,timeout_sec=.05)
            for sample in self.records:
                if sample['stamp']>=after and sample['stamp'] not in seen:
                    samples.append(sample);seen.add(sample['stamp'])
            self.records=[]
            samples=recent_target_samples(samples)
            if len(samples)>=7 and samples[-1]['stamp']-samples[0]['stamp']<=.7:
                try:return stable_summary(samples)
                except ValueError:pass
        raise RuntimeError('No stable, complete, high-confidence target bowl; no further motion')

    def current(self,timeout=15):
        after=time.time();deadline=time.monotonic()+timeout;self.records=[]
        while time.monotonic()<deadline:
            rclpy.spin_once(self.node,timeout_sec=.05)
            if self.records and self.records[-1]['stamp']>=after:
                s=self.records[-1]
                return {'center':s['center'],'bbox':s['bbox'],'area':s['area'],
                        'min_confidence':s['confidence'],'last_stamp':s['stamp'],
                        'weights_sha256':s.get('weights_sha256')}
        raise RuntimeError('No fresh, complete, high-confidence target to lock')


class FrameFeed:
    def __init__(self):
        self.node=rclpy.create_node('bowl_camera_registration')
        self.latest=None
        self.sub=self.node.create_subscription(Image,'/openmv/image_raw',self.receive,qos_profile_sensor_data)
    def receive(self,msg):self.latest=msg
    def frame(self):
        after=time.time();deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            rclpy.spin_once(self.node,timeout_sec=.05)
            msg=self.latest
            if msg and msg.header.stamp.sec+msg.header.stamp.nanosec/1e9>=after and msg.encoding=='rgb8':
                rows=np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.step)
                return rows[:,:msg.width*3].reshape(msg.height,msg.width,3).copy()
        raise RuntimeError('Fresh camera frame missing')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--allow-clipped',action='store_true',help='Permit an edge-clipped, stable locked bowl')
    p.add_argument('--amplitude-p',type=int,default=30)
    p.add_argument('--image-registration',action='store_true')
    p.add_argument('--axes', type=int, nargs='+', default=[0,3])
    p.add_argument('--axis-amplitudes', default='', help='Overrides, e.g. 0:30,1:30,2:45')
    p.add_argument('--resume-report', type=Path, help='Reuse completed double-direction axes at the same pose')
    p.add_argument('--negative-amplitudes', default='', help='Smaller negative probes to keep the bowl in view')
    args=p.parse_args()
    range_config=runtime_profile(ROOT)
    envelopes=pose_envelopes(range_config)
    def interrupt(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    if not 30<=args.amplitude_p<=60:p.error('Calibration amplitude must stay between 30 and 60 P')
    if len(set(args.axes)) != len(args.axes) or not set(args.axes) <= set(range(5)):
        p.error('Choose unique camera joints 0 through 4; gripper holds position')
    amplitude=args.amplitude_p
    amplitudes={axis:amplitude for axis in args.axes}
    if args.axis_amplitudes:
        for item in args.axis_amplitudes.split(','):
            axis,value=map(int,item.split(':'))
            if axis not in amplitudes or not 10<=value<=60:
                p.error('Axis amplitude overrides must be 10 through 60 P on enabled joints')
            amplitudes[axis]=value
    negative_amplitudes=dict(amplitudes)
    if args.negative_amplitudes:
        for item in args.negative_amplitudes.split(','):
            axis,value=map(int,item.split(':'))
            if axis not in amplitudes or not 10<=value<=60:
                p.error('Negative probes must be 10 through 60 P on enabled joints')
            negative_amplitudes[axis]=value
    directory=ROOT/'work/bowl-follow'/f'calibration-{time.strftime("%Y%m%d-%H%M%S")}'
    directory.mkdir(parents=True)
    report={'trials':[],'events':[],'motion_enabled':args.execute,'axes':args.axes,
            'motion_range_policy':range_config.get('motion_range_policy','commissioned'),
            'joint_ranges_p':envelopes,
            'positive_amplitudes':dict(amplitudes),'negative_amplitudes':dict(negative_amplitudes),
            'axis_amplitudes':{a:max(amplitudes[a],negative_amplitudes[a]) for a in args.axes}}
    arm=None;feed=None;frames=None
    def event(kind,**values):
        row=dict(kind=kind,unix=time.time(),**copy.deepcopy(values));report['events'].append(row)
        (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(row,ensure_ascii=False),flush=True)
    def move_endpoint(axis,target,require_response=True):
        amplitude=max(amplitudes[axis],negative_amplitudes[axis])
        current=arm.position(axis)
        if abs(target-current)>80 or not anchor[str(axis)]-amplitude<=target<=anchor[str(axis)]+amplitude:
            raise RuntimeError('Calibration move exceeds local bounds')
        limits=envelopes
        if not 500<=target<=2500:
            raise RuntimeError('Calibration target outside servo protocol range')
        if axis in limits and not limits[axis][0]<=target<=limits[axis][1]:
            raise RuntimeError(f'Joint {axis:03d} probe would cross the commissioning envelope')
        arm._send(f'#{axis:03d}P{target:04d}T0600!')
        end=time.monotonic()+.9
        while time.monotonic()<end:
            incoming=arm._read_for(min(.05,max(0,end-time.monotonic())))
            if incoming:arm._check_received(incoming)
        actual=arm.position(axis);arm.stop_axes([axis])
        event('calibration_endpoint', axis=axis, command_target_p=target, measured_p=actual,
              measured_before_p=current, target_error_p=actual-target)
        if abs(actual-anchor[str(axis)])>amplitude+20:
            raise RuntimeError('Calibration actual position exceeded its local window')
        if not min(current,target)-20<=actual<=max(current,target)+20:
            raise RuntimeError('Measured calibration motion left the requested path')
        if require_response and abs(actual-current)<5:
            raise RuntimeError('Calibration joint did not respond')
        for other in range(6):
            if other!=axis and abs(arm.position(other)-reference[str(other)])>8:
                raise RuntimeError(f'Unexpected movement on joint {other}')
        return actual
    def move_to(axis,target,require_response=True):
        amplitude=max(amplitudes[axis],negative_amplitudes[axis])
        if not anchor[str(axis)]-amplitude<=target<=anchor[str(axis)]+amplitude:
            raise RuntimeError('Calibration move exceeds local bounds')
        return segmented_move(lambda:arm.position(axis),
            lambda goal,final:move_endpoint(axis,goal,require_response and final),target)
    rclpy.init()
    try:
        feed=TargetFeed(allow_clipped=args.allow_clipped)
        if args.image_registration:
            frames=FrameFeed();initial_frame=frames.frame()
            PILImage.fromarray(initial_frame).save(directory/'scene-before.png')
            event('scene_registration_ready')
        else:
            initial=feed.sample();report['initial_target']=initial
            event('target_stable',**initial)
        if not args.execute:
            report['result']='TARGET_SAMPLE_PASS_NO_MOTION';return
        if hashlib.sha256(VENDOR_ACTIONS.read_bytes()).hexdigest()!=VENDOR_SHA256:
            raise RuntimeError('Manufacturer position examples differ')
        arm=ArmSerial('/dev/ttyUSB0');arm.stop();time.sleep(.15)
        reference={str(i):arm.position(i) for i in range(6)};anchor=dict(reference)
        report['initial_pose']=dict(reference)
        completed=set()
        if args.resume_report:
            previous=json.loads(args.resume_report.read_text())
            if not 0<=time.time()-previous.get('finished_unix',0)<=1800:
                raise RuntimeError('Resume measurements must be from the last 30 minutes')
            previous_anchor=measurement_anchor(previous)
            pose_delta={str(i):reference[str(i)]-previous_anchor[str(i)] for i in range(6)}
            if any(abs(pose_delta[str(i)])>(min(30,max(amplitudes[i],negative_amplitudes[i])) if i in amplitudes else 10)
                   for i in range(6)):
                raise RuntimeError('Resume pose is outside the measured local neighborhood')
            report['resume_pose_delta_p']=pose_delta
            for axis in args.axes:
                rows=[t for t in previous['trials'] if t['axis']==axis and abs(t['actual_delta_p'])>=5]
                report['trials'].extend(rows)
                if rows:
                    anchor[str(axis)]=previous_anchor[str(axis)]
                if {1,-1} <= {1 if t['actual_delta_p']>0 else -1 for t in rows}:
                    completed.add(axis)
            event('resumed_axes',axes=sorted(completed),source=str(args.resume_report))
        report['anchor']=dict(anchor)
        validate_current_pose({int(a):p for a,p in anchor.items()},args.axes,
                              tolerance_p=0,config=range_config)
        for axis in args.axes:
            arm._send(f'#{axis:03d}PVER!');reply=arm._read_for(.3);arm._check_received(reply)
            if f'#{axis:03d}@ ZL-ZServo_AD_CBM V2.1.16STG!'.encode() not in reply:
                raise RuntimeError('Unrecognised calibration servo')
        event('pose_reference',positions=reference)
        for axis in args.axes:
            if axis in completed:
                continue
            amplitude=max(amplitudes[axis],negative_amplitudes[axis])
            low,high=envelopes.get(axis,(500,2500))
            near_edge=(anchor[str(axis)]+amplitudes[axis]>high or anchor[str(axis)]-negative_amplitudes[axis]<low)
            signs=(-1,1) if anchor[str(axis)]+amplitudes[axis]>high else (1,-1)
            for sign in signs:
                if any(t['axis']==axis and t['actual_delta_p']*sign>0 for t in report['trials']):
                    continue
                if args.image_registration:before_frame=frames.frame()
                else:before=feed.sample()
                p_before=arm.position(axis)
                target=anchor[str(axis)]+sign*(amplitudes[axis] if sign>0 else negative_amplitudes[axis])
                # A stopped loaded joint can settle past the anchor-relative
                # probe. Ensure the requested probe still goes in its named
                # direction from the current measured position.
                probe=amplitudes[axis] if sign>0 else negative_amplitudes[axis]
                target=max(target,p_before+probe) if sign>0 else min(target,p_before-probe)
                target=max(anchor[str(axis)]-amplitude,min(anchor[str(axis)]+amplitude,target))
                target=max(low,min(high,target))
                if sign*(target-p_before)<10:
                    raise RuntimeError('Not enough local range for the requested calibration direction')
                actual=move_to(axis,target)
                reference[str(axis)]=actual
                registration=None
                if args.image_registration:
                    after_frame=frames.frame()
                    PILImage.fromarray(before_frame).save(directory/f'axis-{axis}-{sign}-before.png')
                    PILImage.fromarray(after_frame).save(directory/f'axis-{axis}-{sign}-after.png')
                    registration=registered_center_shift(before_frame,after_frame)
                    before=registration['before'];after=registration['after']
                else:after=feed.sample()
                trial={'axis':axis,'target_p':target,'actual_before_p':p_before,
                       'actual_after_p':actual,'actual_delta_p':actual-p_before,
                       'before':before,'after':after}
                if registration:trial['registration']=registration
                report['trials'].append(trial);event('trial',**trial)
                # Restore to the measured anchor before measuring the other direction.
                # The measured static offset changes sign with motion direction.
                # Compensate the return using this measured endpoint residual.
                return_target=anchor[str(axis)]
                restored=move_to(axis,return_target,require_response=False)
                for _ in range(3):
                    if abs(restored-anchor[str(axis)])<=10:
                        break
                    corrected=max(low,anchor[str(axis)]-amplitude,min(high,anchor[str(axis)]+amplitude,anchor[str(axis)]-(restored-return_target)))
                    restored=move_to(axis,corrected,require_response=False)
                    return_target=corrected
                reference[str(axis)]=restored
                event('restored',axis=axis,actual_p=restored)
                if abs(restored-anchor[str(axis)])>10:
                    raise RuntimeError('Calibration could not restore its measured local anchor')
                if (axis==2 or near_edge) and not args.image_registration and abs(restored-actual)>=5:
                    returned=feed.sample()
                    trial={'axis':axis,'target_p':return_target,'actual_before_p':actual,
                           'actual_after_p':restored,'actual_delta_p':restored-actual,
                           'before':after,'after':returned,'measurement_source':'bounded_return_to_anchor'}
                    report['trials'].append(trial);event('trial',**trial)
        jacobian,condition=estimate_jacobian(report['trials'], tuple(args.axes))
        final_pose={str(i):arm.position(i) for i in range(6)}
        report.update(result='LOCAL_JACOBIAN_PASS',jacobian=jacobian,condition=condition,
                      anchor=anchor,final_pose=final_pose,amplitude_p=max(*amplitudes.values(),*negative_amplitudes.values()),
                      axis_amplitudes={a:max(amplitudes[a],negative_amplitudes[a]) for a in args.axes},
                      positive_amplitudes=amplitudes,negative_amplitudes=negative_amplitudes,axes=args.axes,
                      measurement_source='registered_scene_center' if args.image_registration else 'stationary_target_center',
                      operator_target=feed.current() if args.image_registration else feed.sample())
        event('calibration_pass',jacobian=jacobian,condition=condition,final_pose=final_pose)
        (ROOT/'work/bowl-follow/last-calibration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    except BaseException as exc:
        report.update(result='ABORTED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if arm:arm.close()
        if feed:feed.node.destroy_node()
        if frames:frames.node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        report['finished_unix']=time.time()
        (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(f'Report: {directory}/report.json',flush=True)


if __name__=='__main__':main()
