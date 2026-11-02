"""Refresh a manually requested session inside the previously verified limits."""
import copy
import math
from vision.bowl_follow import validate_motion_config
from vision.motion_limits import pose_max_window, validate_current_pose


class PoseCalibrationRequired(ValueError):
    """The current pose needs a fresh local camera response measurement."""


def resume_config(previous, positions, target, now, *, accept_current_pose=False):
    config=copy.deepcopy(previous)
    if target.get('weights_sha256') != config.get('operator_target_lock',{}).get('weights_sha256'):
        raise ValueError('当前识别模型与跟随标定不一致，请重新标定')
    confidence=target.get('min_confidence',0)
    if not 0<=now-target.get('last_stamp',0)<=.35 or not math.isfinite(confidence) or not .85<=confidence<=1:
        raise ValueError('请将目标碗放入画面，等待清晰识别后再开启')
    center=target.get('center',[]);bbox=target.get('bbox',[])
    if (len(center)!=2 or len(bbox)!=4 or not all(math.isfinite(v) for v in [*center,*bbox])
            or not 0<=bbox[0]<bbox[2]<=320 or not 0<=bbox[1]<bbox[3]<=240
            or not bbox[0]<=center[0]<=bbox[2] or not bbox[1]<=center[1]<=bbox[3]):
        raise ValueError('目标碗坐标无效，请等待新画面')
    if set(positions)!=set(range(6)) or any(type(p) is not int or not 500<=p<=2500 for p in positions.values()):
        raise ValueError('无法取得六路关节位置，请检查控制板连接')
    axes=config.get('tracking_axes',[0,3])
    if config.get('startup_pose_policy')=='measure_current_pose':
        validate_current_pose(positions,axes,tolerance_p=config.get('feedback_tolerance_p',8),config=config)
    if not accept_current_pose and config.get('startup_pose_policy')=='measure_current_pose':
        if axes==[0,3]:
            raise PoseCalibrationRequired('恢复底座、肘部和腕部三轴协同，自动测量当前姿态')
        measured=config.get('local_calibration',{}).get('final_pose',config['pose_reference'])
        if abs(positions[3]-measured['3'])>90:
            raise PoseCalibrationRequired('腕部姿态已离开原图像测量邻域，自动适配当前位置')
    for axis,p in (() if accept_current_pose else positions.items()):
        if axis in axes:
            bounds=config['axes'][str(axis)]
            margin=config.get('feedback_tolerance_p',8) if config.get('startup_pose_policy')=='measure_current_pose' else 0
            within=(bounds['min_p']-margin<=p<=bounds['max_p']+margin if margin else bounds['min_p']<p<bounds['max_p'])
            if not within:
                raise PoseCalibrationRequired(f'关节 {axis:03d} 已离开旧标定范围，自动适配当前位置')
        elif axis != 5 and abs(p-config['pose_reference'][str(axis)])>12:
            raise PoseCalibrationRequired(f'关节 {axis:03d} 姿态已改变，自动适配当前位置')
    config['pose_reference']={str(a):p for a,p in positions.items()}
    config['operator_target_lock']=dict(center=list(center),bbox=list(bbox),stamp_unix=now,
                                       weights_sha256=target['weights_sha256'])
    windows=config.setdefault('position_windows_p',{})
    max_window=pose_max_window(config) if config.get('startup_pose_policy')=='measure_current_pose' else 500
    for axis in axes:
        bounds=config['axes'][str(axis)];p=positions[axis]
        windows[str(axis)]=min(max_window,max(windows.get(str(axis),config.get('session_window_p',40)),
                                    abs(p-bounds['min_p']),abs(p-bounds['max_p'])))
    validate_motion_config(config)
    return config
