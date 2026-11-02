"""Select a measured response for the current geometry without homing/probing."""
import copy
import math
import numpy as np
from vision.follow_calibration import estimate_jacobian


def select_pose_response(reports, positions):
    candidates=[]
    for source,report in reports:
        try:
            if report.get('result')!='LOCAL_JACOBIAN_PASS' or report.get('axes')!=[0,2,3]:
                continue
            pose=report.get('final_pose',report.get('anchor'))
            if set(pose)!={str(a) for a in range(6)}:
                continue
            matrix,_=estimate_jacobian(report['trials'],(0,2,3))
            if not np.allclose(matrix,report['jacobian'],rtol=.01,atol=.001):
                continue
            # Yaw is controlled in the camera frame. Shoulder/elbow/wrist and
            # camera roll determine which saved geometry best matches startup.
            scale={1:120,2:180,3:180,4:40}
            distance=math.sqrt(sum(((positions[a]-pose[str(a)])/s)**2 for a,s in scale.items()))
            if math.isfinite(distance):
                candidates.append((distance,source,report))
        except (ValueError,TypeError,KeyError,ArithmeticError):
            continue
    if not candidates:
        raise ValueError('尚无有效的三轴姿态模型，需要先完成一次当前姿态测量')
    distance,source,report=min(candidates,key=lambda item:item[0])
    return copy.deepcopy(report),dict(source=source,distance=round(distance,4))


def response_offsets(report):
    result={}
    for axis in report['axes']:
        result[str(axis)]={}
        for sign,label in ((1,'positive'),(-1,'negative')):
            row=next(t for t in report['trials'] if t['axis']==axis and t['actual_delta_p']*sign>0)
            result[str(axis)][label]=int(min(60,max(-60,-sign*(row['actual_after_p']-row['target_p']))))
    return result
