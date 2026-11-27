"""Bowl masks and calibrated optical geometry, without robot or camera I/O.

Image arrays are RGB uint8. Depth is REGISTERED optical-z depth in METRES.
Positions are bowl-bottom centres in odom. No simulator state is accepted.
An unmeasured camera transform is usable only as an explicit simulation
assumption; the returned covariance is an engineering estimate, not a calibrated
statistical guarantee. Real OpenMV intrinsics/extrinsics remain to be measured.
"""
from __future__ import annotations

from pathlib import Path
import math
import time

import numpy as np

from .device import resolve_device,synchronize


class Segmenter:
    """Reuse the bundled single-class YOLO11n-seg; construction does no inference."""
    def __init__(self,weights,device='cpu',conf=.5,imgsz=640):
        weights=Path(weights)
        if not weights.is_file():raise FileNotFoundError(weights)
        if not 0<float(conf)<1 or not isinstance(imgsz,int) or imgsz<32:
            raise ValueError('Invalid confidence/image size')
        self.device=resolve_device(device);self.conf=float(conf);self.imgsz=imgsz
        from ultralytics import YOLO
        self.model=YOLO(str(weights.resolve()),task='segment')
        self.model.to(self.device)
        names=self.model.names
        names=list(names.values()) if isinstance(names,dict) else list(names)
        if names!=['target_bowl']:
            raise ValueError('Expected the bundled single-class target_bowl segmentation weights')
        if getattr(self.model,'task',None)!='segment':raise ValueError('Segmentation weights required')
        self.last_inference_s=None

    def infer(self,rgb):
        image=np.asarray(rgb)
        if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
            raise ValueError('infer expects HWC RGB uint8')
        if min(image.shape[:2])<2:raise ValueError('Image is too small')
        # Ultralytics ndarray sources use BGR, whereas this API deliberately uses RGB.
        synchronize(self.device);started=time.perf_counter()
        results=self.model.predict(source=np.ascontiguousarray(image[...,::-1]),device=str(self.device),
            conf=self.conf,imgsz=self.imgsz,retina_masks=True,verbose=False,save=False,stream=False)
        synchronize(self.device);self.last_inference_s=time.perf_counter()-started
        actual=next(self.model.predictor.model.model.parameters()).device
        if actual.type!=self.device.type:
            raise RuntimeError(f'Network device changed from {self.device} to {actual}; refusing silent fallback')
        if len(results)!=1:raise RuntimeError('Expected one result for one image')
        result=results[0]
        if result.boxes is None or not len(result.boxes):return []
        if result.masks is None:raise RuntimeError('Detector returned boxes without segmentation masks')
        masks=result.masks.data.detach().cpu().numpy()
        boxes=result.boxes.xyxy.detach().cpu().numpy()
        confidences=result.boxes.conf.detach().cpu().numpy()
        classes=result.boxes.cls.detach().cpu().numpy()
        if masks.shape!=(len(boxes),*image.shape[:2]):
            raise RuntimeError('Segmentation masks must be aligned with the original RGB/depth image')
        detections=[]
        for mask,box,confidence,class_id in zip(masks,boxes,confidences,classes):
            if int(class_id)!=0:continue
            binary=mask>.5;ys,xs=np.nonzero(binary)
            if not len(xs):continue
            detections.append({'mask':binary,'confidence':float(confidence),
                'bbox':[float(v) for v in box],'center':[float(np.mean(xs)),float(np.mean(ys))],
                'class_id':0,'class_name':'target_bowl','network_device':str(actual)})
        return detections


def infer_tabletop(segmenter,rgb,depth,k,transform,table_z,bowl_height,settings):
    """Propose image crops from registered depth; only YOLO can supply a bowl mask."""
    import cv2
    h,w=rgb.shape[:2]
    if depth.shape!=(h,w):raise ValueError('RGB and optical depth must be registered')
    scale=float(settings['crop_scale']);gamma=float(settings['gamma']);budget=int(settings['maximum_crops'])
    if not (1<=scale<=4 and .25<=gamma<=2 and 1<=budget<=16):raise ValueError('Invalid tabletop crop settings')
    yy,xx=np.ogrid[:h,:w];vertical=transform[2,2]+(xx-k[0,2])/k[0,0]*transform[2,0]+(yy-k[1,2])/k[1,1]*transform[2,1]
    z=transform[2,3]+depth*vertical
    visible=(np.isfinite(depth)&(depth>.08)&(depth<3.)&(z>table_z+.006)&(z<table_z+bowl_height+.008)).astype(np.uint8)
    stats=cv2.connectedComponentsWithStats(visible,connectivity=8)[2][1:]
    components=[s for s in stats if s[4]>100 and min(s[2:4])>8]
    rects=[(0,0,w,h)]
    if len(components)>budget:return [],{'reason':'too_many_depth_proposals','crops':[],'inference_s':0.}
    scales=[scale]
    if 'context_scale' in settings:
        context=float(settings['context_scale'])
        if not scale<context<=4:raise ValueError('Invalid context crop scale')
        scales.append(context)
    for x,y,width,height,_ in components:
        for factor in scales:
            half=factor*max(width,height)/2;cx=x+width/2;cy=y+height/2
            rect=(max(0,int(cx-half)),max(0,int(cy-half)),min(w,int(cx+half)),min(h,int(cy+half)))
            if rect not in rects:rects.append(rect)
    return _infer_crops(segmenter,rgb,rects,gamma)


def infer_wrist(segmenter,rgb,k,transform,prior,bowl,padding):
    """The parked main estimate predicts a search window, never the wrist mask or pose."""
    h,w=rgb.shape[:2];rects=[(0,0,w,h)];radius=float(bowl['rim_radius_m']);height=float(bowl['height_m'])
    if not 1<=padding<=3:raise ValueError('Invalid wrist search padding')
    corners=np.asarray(prior)+np.array([[x,y,z] for x in [-radius,radius] for y in [-radius,radius] for z in [0.,height]])
    optical=(corners-transform[:3,3])@transform[:3,:3]
    if np.isfinite(optical).all() and np.all(optical[:,2]>.01):
        points=optical@k.T;uv=points[:,:2]/points[:,2:];center=(uv.min(0)+uv.max(0))/2;half=np.ptp(uv,axis=0)*padding/2
        low=np.clip(np.floor(center-half),[0,0],[w,h]).astype(int);high=np.clip(np.ceil(center+half),[0,0],[w,h]).astype(int)
        rect=tuple(int(v) for v in (*low,*high))
        if np.min(high-low)>=16 and rect not in rects:rects.append(rect)
    return _infer_crops(segmenter,rgb,rects,1.)


def _infer_crops(segmenter,rgb,rects,gamma):
    import cv2
    h,w=rgb.shape[:2]
    lut=np.uint8(255*(np.arange(256)/255.)**gamma);found=[];seconds=0.
    for idx,(x0,y0,x1,y1) in enumerate(rects):
        image=rgb[y0:y1,x0:x1];image=cv2.LUT(image,lut) if idx and gamma!=1. else image
        predictions=segmenter.infer(image);seconds+=segmenter.last_inference_s
        for pred in predictions:
            mask=np.zeros((h,w),bool);mask[y0:y1,x0:x1]=pred['mask']
            found.append({**pred,'mask':mask,'bbox':(np.array(pred['bbox'])+[x0,y0,x0,y0]).tolist(),
                'center':(np.array(pred['center'])+[x0,y0]).tolist(),'image_crop_xyxy':[x0,y0,x1,y1]})
    unique=[]
    for pred in sorted(found,key=lambda p:-p['confidence']):
        if not any(np.count_nonzero(pred['mask'] & kept['mask'])>.5*np.count_nonzero(pred['mask'] | kept['mask']) for kept in unique):unique.append(pred)
    return unique,{'crops':[[int(n) for n in r] for r in rects],'gamma_on_crops':gamma,'inference_s':seconds}


def default_perception_config():
    """Unmeasured simulation thresholds; callers must carry a stable target ID."""
    return {'min_confidence':.5,'validity_s':.35,'maximum_future_s':.02,
        'minimum_mask_pixels':100,'depth_min_m':.08,'depth_max_m':3.,
        'depth_kind':'optical_z','edge_band_fraction':.12,'maximum_fit_points':2048,
        'minimum_depth_points':60,'minimum_sidewall_height_m':.008,
        'rim_exclusion_height_m':.004,'minimum_sidewall_height_span_m':.01,
        'depth_inlier_tolerance_m':.005,'rim_inlier_tolerance_m':.003,
        'minimum_inlier_fraction':.60,'minimum_depth_arc_deg':80.,
        'minimum_rim_arc_deg':150.,'maximum_fit_condition':40.,
        'minimum_ray_plane_cosine':.12,'ransac_iterations':192,
        'pixel_sigma_px':1.,'depth_sigma_m':.003,'calibration_sigma_m':.003,
        'bowl_radius_sigma_m':.002,'table_height_sigma_m':.002,
        'depth_xy_floor_m':.003,'wrist_xy_floor_m':.002,
        'calibration_status':'simulation_only_unmeasured_camera',
        'target_id':None,'now_s':None}


def _finite_array(value,shape,name):
    result=np.asarray(value,dtype=float)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError('invalid_'+name)
    return result


def _profile(value):
    if isinstance(value,dict):
        samples=value['radial_profile_m']
        height=float(value.get('height_m',np.asarray(samples)[:,1].max()))
        rim=float(value.get('rim_radius_m',np.asarray(samples)[-1,0]))
    else:
        samples=value;height=float(np.asarray(samples)[:,1].max());rim=float(np.asarray(samples)[-1,0])
    samples=np.asarray(samples,dtype=float)
    if samples.ndim!=2 or samples.shape[1]!=2 or len(samples)<2 or not np.isfinite(samples).all():
        raise ValueError('invalid_bowl_profile')
    if np.any(samples[:,0]<=0) or np.any(np.diff(samples[:,1])<=0) or abs(samples[0,1])>1e-8:
        raise ValueError('invalid_bowl_profile')
    if not 0<height<.5 or not .005<rim<.5 or abs(samples[-1,1]-height)>.001 or abs(samples[-1,0]-rim)>.001:
        raise ValueError('inconsistent_bowl_profile')
    return samples,height,rim


def _contour(mask):
    import cv2
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:raise ValueError('mask_contour_missing')
    largest=max(contours,key=cv2.contourArea)
    filled=np.zeros(mask.shape,dtype=np.uint8);cv2.drawContours(filled,[largest],-1,1,thickness=cv2.FILLED)
    if np.count_nonzero(mask & (filled==0))>max(10,.1*np.count_nonzero(mask)):
        raise ValueError('fragmented_mask')
    boundary=largest[:,0,:].astype(float)
    h,w=mask.shape
    if np.any(boundary[:,0]<=1) or np.any(boundary[:,0]>=w-2) or np.any(boundary[:,1]<=1) or np.any(boundary[:,1]>=h-2):
        raise ValueError('clipped_mask')
    return boundary,filled.astype(bool)


def _rays(pixels,K,T):
    rays=np.column_stack((pixels,np.ones(len(pixels))))@np.linalg.inv(K).T
    return rays,rays@T[:3,:3].T


def _angular_coverage(points,center):
    angles=np.sort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))
    if len(angles)<3:return 0.
    gap=np.diff(np.r_[angles,angles[0]+2*math.pi]).max()
    return float(2*math.pi-gap)


def _circle_candidates(a,b,ra,rb):
    delta=b-a;distance=float(np.linalg.norm(delta))
    if distance<.01 or distance>ra+rb or distance<abs(ra-rb):return []
    along=(ra*ra-rb*rb+distance*distance)/(2*distance)
    height2=ra*ra-along*along
    if height2<0:return []
    direction=delta/distance;middle=a+along*direction
    perpendicular=np.array([-direction[1],direction[0]])*math.sqrt(max(0.,height2))
    return [middle+perpendicular,middle-perpendicular]


def _fit_center(xy,radii,tolerance,cfg,minimum_arc_deg):
    """Known-radius/profile RANSAC, followed by robust metric centre fitting."""
    from scipy.optimize import least_squares
    xy=np.asarray(xy,dtype=float);radii=np.asarray(radii,dtype=float)
    if len(xy)<20:raise ValueError('insufficient_fit_points')
    if len(xy)>cfg['maximum_fit_points']:
        indices=np.linspace(0,len(xy)-1,cfg['maximum_fit_points']).astype(int)
        xy,radii=xy[indices],radii[indices]
    rng=np.random.default_rng(0);best=None;best_score=(-1,-math.inf)
    for _ in range(int(cfg['ransac_iterations'])):
        i,j=rng.choice(len(xy),2,replace=False)
        for candidate in _circle_candidates(xy[i],xy[j],radii[i],radii[j]):
            errors=np.abs(np.linalg.norm(xy-candidate,axis=1)-radii)
            inliers=errors<=tolerance;count=int(inliers.sum())
            score=(count,-float(np.median(errors[inliers])) if count else -math.inf)
            if score>best_score:best=(candidate,inliers);best_score=score
    if best is None:raise ValueError('profile_ransac_failed')
    center,inliers=best
    for _ in range(3):
        if inliers.sum()<20:raise ValueError('insufficient_profile_inliers')
        selected=xy[inliers];expected=radii[inliers]
        fit=least_squares(lambda c:np.linalg.norm(selected-c,axis=1)-expected,center,
            loss='soft_l1',f_scale=tolerance/2,max_nfev=100)
        if not fit.success or not np.isfinite(fit.x).all():raise ValueError('profile_refinement_failed')
        center=fit.x;errors=np.linalg.norm(xy-center,axis=1)-radii
        inliers=np.abs(errors)<=tolerance
    fraction=float(np.mean(inliers))
    if fraction<cfg['minimum_inlier_fraction']:raise ValueError('profile_inlier_fraction_low')
    vectors=xy[inliers]-center
    lengths=np.linalg.norm(vectors,axis=1)
    J=vectors/np.maximum(lengths[:,None],1e-9)
    information=J.T@J/len(J);eigenvalues=np.linalg.eigvalsh(information)
    condition=float(eigenvalues[-1]/max(eigenvalues[0],1e-12))
    if condition>cfg['maximum_fit_condition']:raise ValueError('profile_fit_ill_conditioned')
    arc=math.degrees(_angular_coverage(xy[inliers],center))
    if arc<minimum_arc_deg:raise ValueError('visible_arc_too_small')
    residual=float(np.sqrt(np.mean(errors[inliers]**2)))
    if residual>tolerance*.8:raise ValueError('profile_residual_too_large')
    # Dense neighbouring pixels are correlated. Cap the effective sample count.
    sigma=max(.001,residual)
    covariance=np.linalg.inv(information)*(sigma*sigma/min(32,int(inliers.sum())))
    return center,covariance,{'fit_points':len(xy),'inliers':int(inliers.sum()),
        'inlier_fraction':fraction,'residual_rms_m':residual,'visible_arc_deg':arc,
        'fit_condition':condition}


def _sidewall_points(mask,depth,K,T,table,profile,height,cfg):
    import cv2
    if cfg['depth_kind']!='optical_z':raise ValueError('depth_must_be_registered_optical_z_metres')
    depth=np.asarray(depth,dtype=float)
    if depth.shape!=mask.shape:raise ValueError('rgb_depth_shape_mismatch')
    distances=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    ys,xs=np.nonzero(mask)
    band=max(2.,min(np.ptp(xs)+1,np.ptp(ys)+1)*cfg['edge_band_fraction'])
    edge=mask & (distances>=1.5) & (distances<=band)
    valid=edge & np.isfinite(depth) & (depth>=cfg['depth_min_m']) & (depth<=cfg['depth_max_m'])
    ys,xs=np.nonzero(valid)
    if not len(xs):raise ValueError('depth_sidewall_insufficient')
    rays,_=_rays(np.column_stack((xs,ys)),K,T)
    points=(rays*depth[ys,xs,None])@T[:3,:3].T+T[:3,3]
    heights=points[:,2]-table
    keep=(heights>=cfg['minimum_sidewall_height_m']) & (heights<=height-cfg['rim_exclusion_height_m'])
    points,heights=points[keep],heights[keep]
    if len(points)<cfg['minimum_depth_points']:raise ValueError('depth_sidewall_insufficient')
    if np.ptp(heights)<cfg['minimum_sidewall_height_span_m']:raise ValueError('depth_sidewall_height_span_insufficient')
    radii=np.interp(heights,profile[:,1],profile[:,0])
    return points,radii


def estimate_target(detection,K,T_camera_optical_to_odom,table_top_m,bowl_profile,
                    stamp_s,source,depth_m=None,config=None):
    """Return a JSON-safe target or an explicit invalid record.

    `config.target_id` or `detection.target_id` must identify the SAME bowl across
    both cameras. Assigning identity/multiple-detection association is the caller's
    responsibility. `config.now_s` uses the same clock as `stamp_s`; by default
    estimation is evaluated at capture time. No wall-clock/simulation-clock mixing.
    Wrist mode requires the bowl to rest upright on the known horizontal table.
    It must not be used to estimate a freely moving/lifted bowl from one mask.
    """
    cfg=default_perception_config();cfg.update(config or {})
    output={'valid':False,'reason':None,'frame':'odom','source':str(source),
        'target_id':None,'stamp_s':None,'valid_until_s':None,'position_m':None,
        'covariance_m2':None,'hardware_ready':False,
        'quality':{'calibration_status':str(cfg['calibration_status']),
            'covariance_status':'engineering_assumption_not_empirically_calibrated',
            'uses_simulator_truth':False}}
    try:
        stamp=float(stamp_s);now=stamp if cfg['now_s'] is None else float(cfg['now_s'])
        ttl=float(cfg['validity_s'])
        if not all(math.isfinite(v) for v in (stamp,now,ttl)) or not 0<ttl<=.5:raise ValueError('invalid_timestamp_or_validity')
        output['stamp_s']=stamp;output['valid_until_s']=stamp+ttl
        if now>stamp+ttl:raise ValueError('target_expired')
        if now<stamp-float(cfg['maximum_future_s']):raise ValueError('target_from_future')
        if source not in ('main_rgbd','wrist'):raise ValueError('unknown_camera_source')
        if detection is None:raise ValueError('no_detection')
        if detection.get('valid') is False:raise ValueError('detection_invalid')
        if detection.get('class_name')!='target_bowl' or int(detection.get('class_id',0))!=0:
            raise ValueError('wrong_class')
        confidence=float(detection['confidence'])
        if not math.isfinite(confidence) or not cfg['min_confidence']<=confidence<=1:raise ValueError('low_confidence')
        target_id=detection.get('target_id') or cfg['target_id']
        if not isinstance(target_id,str) or not target_id:raise ValueError('target_id_missing')
        output['target_id']=target_id
        mask=np.asarray(detection['mask'])
        if mask.dtype!=np.bool_ or mask.ndim!=2 or np.count_nonzero(mask)<cfg['minimum_mask_pixels']:
            raise ValueError('invalid_or_small_mask')
        boundary,component=_contour(mask)
        K=_finite_array(K,(3,3),'intrinsics');T=_finite_array(T_camera_optical_to_odom,(4,4),'camera_transform')
        if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1],atol=1e-8):raise ValueError('invalid_intrinsics')
        if not np.allclose(T[3],[0,0,0,1],atol=1e-8) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-6):
            raise ValueError('camera_transform_not_rigid')
        table=float(table_top_m)
        if not math.isfinite(table):raise ValueError('invalid_table_height')
        profile,height,rim=_profile(bowl_profile)
        if source=='main_rgbd':
            if depth_m is None:raise ValueError('registered_depth_required')
            points,radii=_sidewall_points(mask & component,depth_m,K,T,table,profile,height,cfg)
            center,cov,quality=_fit_center(points[:,:2],radii,cfg['depth_inlier_tolerance_m'],cfg,cfg['minimum_depth_arc_deg'])
            floor=cfg['depth_xy_floor_m']
            range_m=float(np.max(np.linalg.norm(points-T[:3,3],axis=1)))
            profile_slope=float(np.max(np.abs(np.diff(profile[:,0])/np.diff(profile[:,1]))))
            measurement_variance=(range_m*cfg['pixel_sigma_px']/min(K[0,0],K[1,1]))**2
            measurement_variance+=cfg['depth_sigma_m']**2*(1+profile_slope**2)
            quality.update(method='edge_sidewall_profile_rgbd',sidewall_points=len(points),
                sidewall_height_span_m=float(np.ptp(points[:,2])),depth_kind='registered_optical_z_metres')
        else:
            if depth_m is not None:raise ValueError('wrist_mode_is_monocular_do_not_supply_depth')
            _,rays=_rays(boundary,K,T);norm=np.linalg.norm(rays,axis=1)
            if np.any(np.abs(rays[:,2])/norm<cfg['minimum_ray_plane_cosine']):raise ValueError('rim_plane_near_parallel')
            distances=(table+height-T[2,3])/rays[:,2]
            if np.any(distances<cfg['depth_min_m']) or np.any(distances>cfg['depth_max_m']):raise ValueError('rim_plane_behind_camera_or_out_of_range')
            points=T[:3,3]+distances[:,None]*rays
            center,cov,quality=_fit_center(points[:,:2],np.full(len(points),rim),cfg['rim_inlier_tolerance_m'],cfg,cfg['minimum_rim_arc_deg'])
            floor=cfg['wrist_xy_floor_m']
            range_m=float(np.max(np.linalg.norm(points-T[:3,3],axis=1)))
            plane_sensitivity=float(np.max(np.linalg.norm(rays[:,:2],axis=1)/np.abs(rays[:,2])))
            measurement_variance=(range_m*cfg['pixel_sigma_px']/min(K[0,0],K[1,1]))**2
            measurement_variance+=(cfg['table_height_sigma_m']*plane_sensitivity)**2
            quality.update(method='known_rim_plane_radius_monocular',rim_plane_z_m=table+height,
                rim_radius_m=rim,assumption='upright_bowl_resting_on_known_horizontal_table')
        uncertainty=float(floor)**2+float(cfg['calibration_sigma_m'])**2+float(cfg['bowl_radius_sigma_m'])**2+measurement_variance
        cov=cov+np.eye(2)*uncertainty
        covariance=np.zeros((3,3));covariance[:2,:2]=cov;covariance[2,2]=float(cfg['table_height_sigma_m'])**2
        if not np.isfinite(covariance).all() or np.min(np.linalg.eigvalsh(covariance))<0:raise ValueError('covariance_invalid')
        output.update(valid=True,reason=None,position_m=[float(center[0]),float(center[1]),table],covariance_m2=covariance.tolist())
        output['quality'].update(quality,confidence=confidence,mask_pixels=int(np.count_nonzero(mask)),
            measurement_xy_sigma_assumption_m=float(math.sqrt(measurement_variance)))
    except (ValueError,TypeError,KeyError,IndexError,np.linalg.LinAlgError) as exc:
        output.update(valid=False,reason=str(exc) or type(exc).__name__)
    return output
