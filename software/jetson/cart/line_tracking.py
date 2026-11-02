"""Black/white/yellow line observations and explicit primary/backup selection.

Image errors are pixel-normalized observations, not metric vehicle commands.
"""
import math


def detect_line(rgb,color='auto_black_white_yellow'):
    import cv2
    import numpy as np
    image=np.asarray(rgb)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('Expected uint8 RGB image')
    h,w=image.shape[:2]
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    masks={
        'black':cv2.inRange(hsv,np.array([0,0,0]),np.array([179,255,65])),
        'white':cv2.inRange(hsv,np.array([0,0,185]),np.array([179,75,255])),
        'yellow':cv2.inRange(hsv,np.array([15,85,85]),np.array([40,255,255]))}
    colors=list(masks) if color=='auto_black_white_yellow' else [color]
    if any(c not in masks for c in colors):raise ValueError('Unsupported line color')
    candidates=[]
    for candidate_color in colors:
        mask=cv2.morphologyEx(masks[candidate_color],cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        centers=[];wide=0;ambiguous=False
        ys=np.linspace(.91*h,.45*h,13).astype(int)
        previous=None
        for band,y in enumerate(ys):
            row=np.count_nonzero(mask[max(0,y-2):min(h,y+3)],axis=0)>=3
            transitions=np.diff(np.concatenate([[False],row,[False]]).astype(int))
            runs=list(zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)))
            if any(b-a>.3*w for a,b in runs):wide+=1
            runs=[(a,b) for a,b in runs if max(3,.005*w)<=b-a<=.14*w]
            if previous is not None:runs=[(a,b) for a,b in runs if abs((a+b)/2-previous)<.18*w]
            if len(runs)>1:
                ambiguous=True;continue
            if len(runs)==1:
                center=sum(runs[0])/2
                centers.append((float(center),float(y),band));previous=center
        valid=False;reason='ambiguous_line_candidates' if ambiguous else 'no_continuous_line';confidence=0.;error=heading=0.;path=[]
        if len(centers)>=8 and any(b<=1 for _,_,b in centers):
            if wide:
                reason='junction_or_wide_marking'
            elif ambiguous:
                reason='ambiguous_line_candidates'
            else:
                x=np.array([p[0] for p in centers]);y=np.array([p[1] for p in centers])
                polynomial=np.polyfit(y/h,x/w,2)
                fitted=np.polyval(polynomial,y/h)*w
                rms=float(np.sqrt(np.mean((x-fitted)**2)))
                confidence=len(centers)/len(ys)*max(0,1-rms/(.025*w))
                near=float(np.polyval(polynomial,.90)*w)
                far=float(np.polyval(polynomial,.48)*w)
                error=(near-w/2)/(w/2)
                heading=math.atan2(far-near,.42*h)
                valid=confidence>=.75 and abs(error)<.95
                reason='line_observed' if valid else 'low_line_confidence'
                path=[[float(np.polyval(polynomial,float(v)/h)*w),float(v)] for v in ys]
        candidates.append({'valid':valid,'reason':reason,'color':candidate_color,'confidence':confidence,
                           'lateral_error_normalized':error,'heading_image_rad':heading,'path_pixels':path,
                           'support_bands':len(centers),'wide_bands':wide,'mask':mask})
    accepted=sorted([r for r in candidates if r['valid']],key=lambda r:r['confidence'],reverse=True)
    if len(accepted)>1 and accepted[0]['confidence']-accepted[1]['confidence']<.15:
        result=dict(accepted[0],valid=False,reason='ambiguous_line_colors',confidence=0.)
    elif accepted:
        result=accepted[0]
    else:
        result=max(candidates,key=lambda r:r['support_bands'])
    return dict(result,candidates=[{k:v for k,v in c.items() if k!='mask'} for c in candidates],
                output_kind='image_line_observation',metric_vehicle_pose=False,hardware_commands_sent=False)


def line_to_ground(path_pixels,camera_k,camera_d,camera_to_base,ground_z_m,mount_verified=False):
    """Intersect calibrated optical rays with an explicitly supplied ground plane."""
    import cv2
    import numpy as np
    if mount_verified is not True:
        return {'valid':False,'reason':'measured_camera_mount_required','points_xy_m':[]}
    from vision.road_geometry import valid_transform
    t=valid_transform(camera_to_base)
    pixels=np.asarray(path_pixels,dtype=float).reshape(-1,1,2)
    if len(pixels)<5 or not np.isfinite(pixels).all() or not np.isfinite(ground_z_m):
        return {'valid':False,'reason':'invalid_line_or_ground_plane','points_xy_m':[]}
    normalized=cv2.undistortPoints(pixels,np.asarray(camera_k,dtype=float),np.asarray(camera_d,dtype=float)).reshape(-1,2)
    rays=np.column_stack([normalized,np.ones(len(normalized))])@t[:3,:3].T
    points=[]
    for ray in rays:
        if abs(ray[2])<1e-8:continue
        scale=(ground_z_m-t[2,3])/ray[2]
        point=t[:3,3]+scale*ray
        if scale>0 and .05<=point[0]<=5 and np.isfinite(point).all():points.append(point[:2].tolist())
    points.sort(key=lambda p:p[0])
    return {'valid':len(points)>=5,'reason':'calibrated_flat_ground_projection' if len(points)>=5 else 'insufficient_ground_intersections',
            'points_xy_m':points,'terrain_assumption':'locally_flat_ground_requires_field_validation'}


class NavigationMode:
    def __init__(self,config):
        self.cfg=config;self.mode='WAIT_LINE';self.reason='waiting_for_primary_line'
        self.last_now=None;self.started=None;self.last_good=None;self.good_frames=0
        self.last_line_stamp=None
        self.latched=False

    def update(self,now,line,stamp_s,base_ready=False,obstacles_ready=False,backup_ready=False,route_choice_ready=True):
        if self.started is None:self.started=now
        if self.latched:return self.mode
        if type(now) not in (int,float) or not math.isfinite(now) or self.last_now is not None and now<=self.last_now:
            return self.stop('clock_discontinuity')
        self.last_now=now
        if base_ready is not True or obstacles_ready is not True:return self.stop('base_or_obstacles_unavailable')
        if line.get('reason') in {'junction_or_wide_marking','ambiguous_line_candidates','ambiguous_line_colors'}:
            # Selecting a branch requires a separate route/marker decision. This
            # baseline deliberately does not guess which painted line to follow.
            return self.stop('junction_or_line_ambiguity')
        if not route_choice_ready:return self.stop('route_choice_unavailable')
        fresh=(type(stamp_s) in (int,float) and math.isfinite(stamp_s)
               and -.05<=now-stamp_s<=self.cfg['line_max_age_s'])
        confidence=line.get('confidence')
        good=fresh and line.get('valid') is True and type(confidence) in (int,float) and math.isfinite(confidence) and self.cfg['minimum_confidence']<=confidence<=1
        if good:
            if self.last_line_stamp is not None and stamp_s<=self.last_line_stamp:
                return self.mode
            self.good_frames=self.good_frames+1 if self.last_line_stamp is not None and stamp_s-self.last_line_stamp<=self.cfg['line_max_age_s'] else 1
            self.last_line_stamp=stamp_s;self.last_good=stamp_s
            if self.good_frames>=self.cfg['stable_line_frames']:
                self.mode,self.reason='LINE','track_line_primary'
            elif self.mode=='BACKUP' and backup_ready is not True:
                return self.stop('backup_became_unavailable')
            return self.mode
        self.good_frames=0
        if self.last_good is None:
            if now-self.started<=self.cfg['startup_line_grace_s']:
                self.mode='WAIT_LINE';return self.mode
            if backup_ready is True:
                self.mode,self.reason='BACKUP','no_line_backup_explicitly_available';return self.mode
            return self.stop('line_unavailable')
        if now-self.last_good<=self.cfg['lost_line_slow_window_s']:
            self.mode,self.reason='LINE_LOST_SLOW','brief_line_loss';return self.mode
        if backup_ready is True:
            self.mode,self.reason='BACKUP','line_lost_backup_valid';return self.mode
        return self.stop('line_loss_stop')

    def stop(self,reason):
        self.mode,self.reason,self.latched='STOP',reason,True
        return self.mode
