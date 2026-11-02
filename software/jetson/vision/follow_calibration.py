"""Fresh, stable target samples and a measured local image/servo Jacobian."""
import math
import time

import numpy as np
import cv2


def segmented_move(read_position, move_endpoint, target, max_step=65, max_segments=8):
    """Recompute bounded waypoints from feedback when loaded servos undershoot."""
    for _ in range(max_segments):
        current=read_position()
        if abs(target-current)<=max_step:
            return move_endpoint(target,True)
        waypoint=current+(max_step if target>current else -max_step)
        measured=move_endpoint(waypoint,False)
        if abs(target-measured)>=abs(target-current)-2:
            raise RuntimeError('Calibration return made no measured progress')
    raise RuntimeError('Calibration return exceeded its bounded segment count')


def registered_center_shift(before_rgb, after_rgb):
    """Measure a camera-induced image shift using distributed scene features.

    Ignore the central working area so hand movement around the bowl does not
    define the camera calibration. Return the registered image center, with
    inlier/spread checks; this is a scene measurement, not a bowl detection.
    """
    first=cv2.cvtColor(before_rgb,cv2.COLOR_RGB2GRAY)
    second=cv2.cvtColor(after_rgb,cv2.COLOR_RGB2GRAY)
    if first.shape!=(240,320) or second.shape!=first.shape:
        raise ValueError('Registration requires 320x240 camera images')
    mask=np.full(first.shape,255,np.uint8)
    mask[120:240,40:280]=0
    # Descriptor matches tolerate the larger rotations and displacement of
    # shoulder/elbow probes. Both images restrict features to the same scene region.
    detector=cv2.SIFT_create(nfeatures=1200,contrastThreshold=.012,edgeThreshold=15)
    keys_a,desc_a=detector.detectAndCompute(first,mask)
    keys_b,desc_b=detector.detectAndCompute(second,mask)
    if desc_a is None or desc_b is None:
        raise ValueError('Not enough distributed scene features')
    pairs=cv2.BFMatcher().knnMatch(desc_a,desc_b,k=2)
    matches=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance < .72*pair[1].distance]
    if len(matches)<18:
        raise ValueError('Scene feature correspondence failed')
    a=np.float32([keys_a[m.queryIdx].pt for m in matches])
    b=np.float32([keys_b[m.trainIdx].pt for m in matches])
    matrix,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=2000,confidence=.99)
    if matrix is None or inliers is None:raise ValueError('Scene registration failed')
    use=inliers.ravel()>0;count=int(use.sum())
    if count<15 or count/len(a)<.55:
        raise ValueError('Inconsistent scene motion')
    span=np.ptp(a[use],axis=0)
    if span[0]<100 and span[1]<65:
        raise ValueError('Registration is confined to a moving foreground region')
    center=(matrix@np.array([160.,120.,1.])).tolist()
    errors=np.linalg.norm((a[use]@matrix[:,:2].T+matrix[:,2])-b[use],axis=1)
    if float(np.median(errors))>1.5:raise ValueError('Scene registration residual too large')
    return {'before':{'center':[160.,120.]},'after':{'center':center},
            'measurement_source':'registered_scene_center','inliers':count,
            'matched_points':len(a),'median_residual_px':float(np.median(errors)),
            'feature_span_px':span.tolist(),'affine':matrix.tolist()}


def target_measurement(record, now, confidence=.85, allow_clipped=False):
    try:
        stamp=record['stamp']['sec']+record['stamp']['nanosec']/1e9
        if not record['valid'] or record['state']!='live' or not 0<=now-stamp<=.35:
            return None
        if (record['width'],record['height'])!=(320,240) or len(record['detections'])!=1:
            return None
        item=record['detections'][0]
        x,y=map(float,item['mask_centroid_px'])
        x1,y1,x2,y2=map(float,item['bbox_xyxy_px'])
        conf=float(item['confidence']);area=float(item['mask_area_px'])
        if not all(map(math.isfinite,[stamp,x,y,x1,y1,x2,y2,conf,area])):
            return None
        if item['class']!='target_bowl' or not confidence<=conf<=1 or not .02<=area/(320*240)<=.65:
            return None
        margin_x,margin_y=(0,0) if allow_clipped else (5,4)
        if not (margin_x<=x1<x2<=320-margin_x and margin_y<=y1<y2<=240-margin_y and x1<=x<=x2 and y1<=y<=y2):
            return None
        return {'stamp':stamp,'center':[x,y],'confidence':conf,'area':area,'bbox':[x1,y1,x2,y2]}
    except (KeyError,TypeError,ValueError,AttributeError,OverflowError):
        return None


def stable_summary(samples):
    if len(samples)<7:
        raise ValueError('At least seven independent target frames are required')
    centers=np.array([s['center'] for s in samples],dtype=float)
    median=np.median(centers,axis=0)
    jitter=float(np.max(np.linalg.norm(centers-median,axis=1)))
    areas=np.array([s['area'] for s in samples],dtype=float)
    if jitter>3 or areas.max()/areas.min()>1.15:
        raise ValueError('Target is not stationary enough for direction calibration')
    return {'center':median.tolist(),'jitter_px':jitter,'area':float(np.median(areas)),
            'min_confidence':min(s['confidence'] for s in samples),'frames':len(samples),
            'first_stamp':samples[0]['stamp'],'last_stamp':samples[-1]['stamp'],
            'bbox':np.median([s['bbox'] for s in samples],axis=0).tolist()}


def recent_target_samples(samples, max_span=.7, max_count=9):
    """Keep recent independent measurements even when frame rate drops.

    Nine frames at 10 FPS span .8 s, while seven recent frames span .6 s.
    An old ninth frame must not veto a valid seven-frame calibration window.
    """
    if not samples:
        return []
    cutoff=samples[-1]['stamp']-max_span
    return [s for s in samples if s['stamp']>=cutoff][-max_count:]


def measurement_anchor(report):
    """A resumed run keeps original measurement anchors, not its restart pose."""
    anchor=report.get('anchor') or report.get('initial_pose')
    if anchor is None:
        anchor=next((e['positions'] for e in report.get('events',[])
                     if e.get('kind')=='pose_reference'),None)
    if (not isinstance(anchor,dict) or set(anchor)!={str(a) for a in range(6)}
            or any(type(value) is not int or not 500<=value<=2500 for value in anchor.values())):
        raise ValueError('Missing measured six-joint calibration anchor')
    return dict(anchor)


def estimate_jacobian(trials, axes=(0, 3)):
    """Fit each column from measured P changes, not commanded P changes."""
    columns=[]
    for axis in axes:
        rows=[t for t in trials if t['axis']==axis]
        if len(rows)<2 or not {1,-1}<={1 if t['actual_delta_p']>0 else -1 for t in rows}:
            raise ValueError(f'Both motion directions must be measured for joint {axis}')
        slopes=[]
        for row in rows:
            dp=float(row['actual_delta_p'])
            displacement=np.array(row['after']['center'])-np.array(row['before']['center'])
            if abs(dp)<5 or (axes == (0, 3) and np.linalg.norm(displacement)<3):
                raise ValueError('Joint or image displacement too small to establish direction')
            slopes.append(displacement/dp)
        a,b=slopes[:2]
        if axes != (0, 3) and max(np.linalg.norm(a), np.linalg.norm(b)) < .08:
            columns.append(np.zeros(2))
            continue
        if float(np.dot(a,b))<=0 or np.linalg.norm(a-b)>max(np.linalg.norm(a),np.linalg.norm(b))*.7:
            raise ValueError('Opposite direction trials disagree; target may have moved')
        columns.append(np.mean(slopes,axis=0))
    jacobian=np.column_stack(columns)
    condition=float(np.linalg.cond(jacobian))
    if not np.isfinite(jacobian).all() or not condition<=10:
        raise ValueError('Image/servo response is singular or inconsistent')
    if np.max(np.abs(jacobian))>5 or (axes == (0, 3) and np.min(np.linalg.norm(jacobian,axis=0))<.03):
        raise ValueError('Implausible local image/servo sensitivity')
    return jacobian.tolist(),condition
