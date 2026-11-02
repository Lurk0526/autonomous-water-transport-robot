#!/usr/bin/env python3
"""Prepare a fresh, measured configuration for bounded multi-joint tracking."""
import json
import argparse
from pathlib import Path
import sys
import subprocess
import math
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import rclpy
from tools.calibrate_bowl_follow import TargetFeed
from vision.arm_serial import ArmSerial
from vision.bowl_follow import validate_motion_config
from vision.follow_session import PoseCalibrationRequired, resume_config
from vision.tracking_profile import apply_tracking_profile
from vision.pose_response import select_pose_response,response_offsets
from vision.motion_limits import validate_current_pose, runtime_profile, pose_envelopes, FULL_RANGE_POLICY


def prepare_resume():
    path=ROOT/'work/bowl-follow/current-trial.json'
    preset=path if path.exists() else ROOT/'config/bowl-follow.v1.json'
    previous=json.loads(preset.read_text())
    previous.update(runtime_profile(ROOT))
    if 'pose_reference' in previous:
        previous=apply_tracking_profile(previous)
    previous.setdefault('profile_version','1.0.0')
    rclpy.init();feed=TargetFeed(allow_clipped=True);arm=None
    recalibrate=False
    try:
        arm=ArmSerial(previous['port']);arm.stop()
        positions=arm.positions()
        if previous.get('startup_pose_policy')=='measure_current_pose':
            validate_current_pose(positions,previous.get('tracking_axes',[0,3]),
                                  tolerance_p=previous.get('feedback_tolerance_p',8),config=previous)
        target=feed.current()
        reports=[('current-profile',previous.get('local_calibration',{}))]
        frozen=ROOT/'config/pose-response-library.v1.json'
        if frozen.exists():
            library=json.loads(frozen.read_text())
            reports.extend((row['source'],row['calibration']) for row in library['models'])
        for saved in (ROOT/'work/bowl-follow').glob('calibration-*/report.json'):
            try:reports.append((str(saved),json.loads(saved.read_text())))
            except (OSError,ValueError):continue
        try:
            measured,selection=select_pose_response(reports,positions)
        except ValueError:
            measured=None
        needs_current_measurement=(previous.get('motion_range_policy')==FULL_RANGE_POLICY
                                   and (measured is None or selection['distance']>1.0))
        if needs_current_measurement:
            measured=None
        if measured is not None:
            previous.update(local_calibration=measured,tracking_axes=[0,2,3],
                            startup_pose_policy='measure_current_pose',pose_response_selection=selection)
            previous['feedback_deadband_p']=response_offsets(measured)
            previous['deadband_compensation_p']=[[previous['feedback_deadband_p'][str(a)][d]
                for d in ('positive','negative')] for a in (0,2,3)]
            previous['pose_reference']={str(a):p for a,p in positions.items()}
            previous=apply_tracking_profile(previous)
        try:
            if needs_current_measurement:
                raise PoseCalibrationRequired('当前姿态与已有模型差异较大，正在原地适配跟随方向')
            if 'pose_reference' not in previous:
                raise PoseCalibrationRequired('尚无可用姿态模型，准备首次测量')
            config=resume_config(previous,positions,target,time.time(),accept_current_pose=measured is not None)
        except PoseCalibrationRequired as exc:
            print(str(exc),flush=True)
            recalibrate=True
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(config,ensure_ascii=False,indent=2));temp.replace(path)
            print('已读取当前位置并选择实测姿态模型，可直接跟踪，无需恢复旧姿态或重新探测。',flush=True)
    finally:
        if arm:arm.close()
        feed.node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    if recalibrate:
        prepare_current_pose()


def prepare_current_pose():
    """Measure the base, elbow and wrist at this pose before enabling them."""
    print('正在自动适配当前位置：底座、肘部和腕部将小幅测量，请暂时保持碗静止。',flush=True)
    result=subprocess.run([sys.executable,str(ROOT/'tools/calibrate_bowl_follow.py'),
        '--execute','--allow-clipped','--axes','0','2','3','--amplitude-p','60',
        '--axis-amplitudes','0:30,2:60','--negative-amplitudes','3:25'],cwd=ROOT)
    if result.returncode:
        raise RuntimeError('当前位置方向测量未完成；请保持碗在画面内且静止后重试，原跟随配置已保留')
    prepare_from_calibration()


def prepare_from_calibration():
    runtime=runtime_profile(ROOT)
    calibration=json.loads((ROOT/'work/bowl-follow/last-calibration.json').read_text())
    if calibration['result']!='LOCAL_JACOBIAN_PASS' or time.time()-calibration['events'][-1]['unix']>1800:
        raise RuntimeError('A current successful local calibration is required')
    rclpy.init();feed=TargetFeed(allow_clipped=True);arm=None
    try:
        arm=ArmSerial('/dev/ttyUSB0');arm.stop()
        reference={str(i):p for i,p in arm.positions().items()}
        tracking_axes=calibration.get('axes', [0,3])
        if (tracking_axes==[0,2,3] and math.hypot(*(row[1] for row in calibration['jacobian']))<.08
                and runtime.get('motion_range_policy')!=FULL_RANGE_POLICY):
            raise RuntimeError('肘部图像响应不足，未启用未经确认的协同运动')
        amplitudes={str(a):calibration.get('axis_amplitudes', {}).get(str(a),calibration['amplitude_p'])
                    for a in tracking_axes}
        for axis in range(6):
            if axis in tracking_axes:
                allowed=240 if axis==0 else amplitudes[str(axis)]+10
                good=abs(reference[str(axis)]-calibration['anchor'][str(axis)])<=allowed
            else:
                good=abs(reference[str(axis)]-calibration['final_pose'][str(axis)])<=8
            if not good:
                raise RuntimeError('Arm has moved since the local calibration')
        offsets={}
        for axis in tracking_axes:
            offsets[str(axis)]={}
            for sign,label in ((1,'positive'),(-1,'negative')):
                row=next(t for t in calibration['trials'] if t['axis']==axis and t['actual_delta_p']*sign>0)
                error=row['actual_after_p']-row['target_p']
                offsets[str(axis)][label]=int(min(60,max(-60,-sign*error)))
        target=feed.current()
        amplitude=calibration['amplitude_p']
        config={'camera_rigid_on_arm':True,'directions_and_limits_verified':True,
                'serial_open_reset_resolved':True,'perception_validated':False,'supervised_trial':True,
                'port':'/dev/ttyUSB0','pose_reference':reference,'local_calibration':calibration,
                'operator_target_lock':{'center':target['center'],'bbox':target['bbox'],
                    'stamp_unix':time.time(),'weights_sha256':'c402769dabe8f46fd69e01f76054622b15677862fd34e1825618fedc4b9073a6'},
                'tracking_axes':tracking_axes,'enabled_axes':list(range(6)),
                'bounded_allocation':True,'fresh_frame_after_feedback':True,'adaptive_servo_bias':True,
                'bounded_stop_feedback':True,
                'adaptive_bias_tolerance_p':25,
                'allow_settling_drift':True,
                'selective_stop':True,
                'position_windows_p':{'0':240},
                'planning_period_s':.1,'filter_alpha':.8,'hold_hysteresis':1.5,
                'center_tolerance_px':6,'settle_seconds':.2,
                'joint_weights':[8 if a in (1,2) else 1 for a in tracking_axes],
                'session_seconds':900,'allow_clipped_tracking':True,'tracking_confidence':.75,'max_step_p':65,'min_step_p':1,'duration_ms':500,'session_travel_p':12000,
                'session_window_p':90,'feedback_tolerance_p':10,'feedback_deadband_p':offsets,
                'deadband_compensation_p':[[offsets[str(a)]['positive'],offsets[str(a)]['negative']] for a in tracking_axes],
                'axes':{str(axis):{'min_p':calibration['anchor'][str(axis)]-min(90,amplitudes[str(axis)]*2),
                                   'max_p':calibration['anchor'][str(axis)]+min(90,amplitudes[str(axis)]*2),'direction':-1} for axis in tracking_axes}}
        if 0 in tracking_axes:
            anchor=calibration['anchor']['0']
            low,high=pose_envelopes(runtime)[0]
            config['axes']['0'].update(min_p=max(low,anchor-240),max_p=min(high,anchor+240))
        for axis in tracking_axes:
            limits=config['axes'][str(axis)]
            config['position_windows_p'][str(axis)]=min(500,max(config['position_windows_p'].get(str(axis),90),
                abs(reference[str(axis)]-limits['min_p']),abs(reference[str(axis)]-limits['max_p'])))
        if tracking_axes in ([0,3],[0,2,3]):
            config['startup_pose_policy']='measure_current_pose'
        config.update(runtime)
        config=apply_tracking_profile(config)
        validate_motion_config(config)
        path=ROOT/'work/bowl-follow/current-trial.json'
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(config,ensure_ascii=False,indent=2));temp.replace(path)
        print(json.dumps({'configuration':str(path),'reference':reference,'target':target,
                          'deadband_p':offsets,'joint_axes':tracking_axes,'duration_ms':config['duration_ms']},ensure_ascii=False),flush=True)
    finally:
        if arm:arm.close()
        feed.node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--reuse-profile',action='store_true')
    group.add_argument('--adapt-current-pose',action='store_true')
    args=parser.parse_args()
    if args.reuse_profile:
        prepare_resume()
    elif args.adapt_current_pose:
        prepare_current_pose()
    else:
        prepare_from_calibration()


if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError,KeyError,OSError) as exc:
        message=str(exc)
        if 'No fresh, complete, high-confidence target' in message:
            message='未识别到清晰的目标碗，请将碗放入画面后重新点击开启'
        elif 'Explicit port is not' in message:
            message='未找到机械臂控制板 USB，请检查连接后重试'
        print(f'无法准备跟踪：{message}',file=sys.stderr,flush=True)
        raise SystemExit(1)
