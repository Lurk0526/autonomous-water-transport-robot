"""Detect a platform from registered metric depth; no object labels or scene API.

There is no trained platform network in this package. Horizontal-plane geometry
locates a support surface; task-size/orientation priors bound its hidden edges.
Those priors are explicit assumptions, never presented as measured geometry.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np


def default_platform_config():
    return {'height_range_m':[.25,.45], 'height_bin_m':.004,
        'plane_tolerance_m':.003, 'normal_vertical_min':.97,
        'minimum_support_pixels':300,'minimum_visible_extent_m':[.18,.18],
        'maximum_platform_size_m':[.4,.4],
        'orientation_prior_yaw_rad':0.,'orientation_uncertainty_rad':0.,
        'prior_status':'simulation_task_size_and_alignment_assumption_not_measurement',
        'depth_range_m':[.08,3.],'calibration_sigma_m':.003,
        'validity_s':.5,'maximum_frame_age_s':.5,'maximum_future_s':.02,
        'footprint_extra_margin_m':.005}


def _arr(value,shape,name):
    array=np.asarray(value,dtype=float)
    if array.shape!=shape or not np.isfinite(array).all():raise ValueError('invalid_'+name)
    return array


def _settings(config):
    cfg=default_platform_config();cfg.update(config or {})
    for key in ('height_range_m','depth_range_m'):
        bounds=_arr(cfg[key],(2,),key)
        if bounds[0]>=bounds[1]:raise ValueError('invalid_'+key)
    for key in ('minimum_visible_extent_m','maximum_platform_size_m'):
        if np.any(_arr(cfg[key],(2,),key)<=0):raise ValueError('invalid_'+key)
    if cfg['orientation_prior_yaw_rad'] is None:
        # Without an orientation prior use the diagonal bound, never assume an
        # axis-aligned platform just because the simulated example is aligned.
        cfg['orientation_uncertainty_rad']=math.pi
    elif not math.isfinite(float(cfg['orientation_prior_yaw_rad'])):raise ValueError('invalid_orientation_prior')
    else:
        angle=float(cfg['orientation_prior_yaw_rad']);cfg['orientation_prior_yaw_rad']=math.atan2(math.sin(angle),math.cos(angle))
    if not 0<=float(cfg['orientation_uncertainty_rad'])<=math.pi:raise ValueError('invalid_orientation_uncertainty')
    if not 0<cfg['normal_vertical_min']<1 or not 0<cfg['validity_s']<=.5:raise ValueError('invalid_platform_threshold')
    if type(cfg['minimum_support_pixels']) is not int or cfg['minimum_support_pixels']<20:raise ValueError('invalid_support_pixels')
    for key in ('height_bin_m','plane_tolerance_m','calibration_sigma_m'):
        if not math.isfinite(float(cfg[key])) or cfg[key]<=0:raise ValueError('invalid_'+key)
    if not math.isfinite(float(cfg['footprint_extra_margin_m'])) or cfg['footprint_extra_margin_m']<0:raise ValueError('invalid_footprint_margin')
    return cfg


def _maximum_axis_extent(cfg):
    size=np.asarray(cfg['maximum_platform_size_m'],float)
    if cfg['orientation_prior_yaw_rad'] is None or cfg['orientation_uncertainty_rad']>=math.pi/2:
        return np.repeat(np.linalg.norm(size),2)
    yaw=float(cfg['orientation_prior_yaw_rad']);uncertainty=float(cfg['orientation_uncertainty_rad'])
    # Include all stationary points of width*cos(yaw)+height*sin(yaw), as well
    # as the interval endpoints. Sampling an angle grid would under-bound it.
    angles=[yaw-uncertainty,yaw+uncertainty]
    for index in range(-6,7):
        angles.extend([index*math.pi/2,index*math.pi/2+math.atan2(size[1],size[0]),
                       index*math.pi/2+math.atan2(size[0],size[1])])
    allowed=[a for a in angles if yaw-uncertainty-1e-12<=a<=yaw+uncertainty+1e-12]
    c,s=np.abs(np.cos(allowed)),np.abs(np.sin(allowed))
    return np.array([np.max(size[0]*c+size[1]*s),np.max(size[0]*s+size[1]*c)])


def detect_platform(depth_m,K,T_optical_to_odom,stamp_s,*,config=None,now_s=None,
                    return_mask=False):
    """Return a support-plane estimate with conservative hidden-edge bounds.

    Inputs contain no table centre, simulated geom ID, bowl position or mask.
    The returned table is an axis-aligned conservative enclosure, not an exact
    reconstruction. Its z interval conservatively occupies ground to top.
    ``valid`` means a plausible bounded support surface was found, not that every
    edge was seen or that a learned semantic platform classifier was validated.
    """
    result={'valid':False,'reason':None,'kind':'platform','source':'main_rgbd_platform',
        'frame':'odom','stamp_s':None,'valid_until_s':None,'center_xy_m':None,
        'covariance_m2':None,'conservative_table':None,'uses_simulator_truth':False,
        'learned_platform_detector':False,'hardware_ready':False}
    selected_mask=None
    try:
        cfg=_settings(config);depth=np.asarray(depth_m,dtype=float)
        if depth.ndim!=2 or min(depth.shape)<8:raise ValueError('invalid_registered_depth')
        k=_arr(K,(3,3),'intrinsics');tf=_arr(T_optical_to_odom,(4,4),'optical_transform')
        if k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):raise ValueError('invalid_intrinsics')
        if not np.allclose(tf[3],[0,0,0,1]) or not np.allclose(tf[:3,:3].T@tf[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(tf[:3,:3]),1.,atol=1e-6):raise ValueError('invalid_rigid_transform')
        stamp=float(stamp_s);now=stamp if now_s is None else float(now_s)
        if not all(math.isfinite(v) for v in (stamp,now)) or not -cfg['maximum_future_s']<=now-stamp<=cfg['maximum_frame_age_s']:raise ValueError('stale_or_future_platform_frame')
        result.update(stamp_s=stamp,valid_until_s=stamp+cfg['validity_s'])
        v,u=np.mgrid[:depth.shape[0],:depth.shape[1]]
        rays=np.stack([(u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1],np.ones_like(u)],axis=-1)
        xyz=(rays*depth[...,None])@tf[:3,:3].T+tf[:3,3]
        valid=np.isfinite(depth)&(depth>=cfg['depth_range_m'][0])&(depth<=cfg['depth_range_m'][1])
        dx=xyz[1:-1,2:]-xyz[1:-1,:-2];dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
        normal=np.cross(dx,dy);length=np.linalg.norm(normal,axis=-1)
        horizontal=np.zeros(depth.shape,bool)
        horizontal[1:-1,1:-1]=(np.abs(normal[...,2])/np.maximum(length,1e-12)>=cfg['normal_vertical_min'])&(length>1e-10)
        allowed=valid&horizontal&(xyz[...,2]>=cfg['height_range_m'][0])&(xyz[...,2]<=cfg['height_range_m'][1])
        heights=xyz[...,2][allowed]
        if len(heights)<cfg['minimum_support_pixels']:raise ValueError('insufficient_horizontal_depth_support')
        bins=np.arange(cfg['height_range_m'][0],cfg['height_range_m'][1]+cfg['height_bin_m'],cfg['height_bin_m'])
        histogram,edges=np.histogram(heights,bins=bins);proposals=[]
        maximum_extent=_maximum_axis_extent(cfg)
        for peak in np.argsort(histogram)[-6:][::-1]:
            if histogram[peak]<cfg['minimum_support_pixels']:continue
            level=(edges[peak]+edges[peak+1])/2
            near=allowed&(np.abs(xyz[...,2]-level)<=cfg['height_bin_m'])
            if np.count_nonzero(near)<cfg['minimum_support_pixels']:continue
            top=float(np.median(xyz[...,2][near]))
            support=(allowed&(np.abs(xyz[...,2]-top)<=cfg['plane_tolerance_m'])).astype(np.uint8)
            count,labels,stats,_=cv2.connectedComponentsWithStats(support,connectivity=8)
            for label in range(1,count):
                if stats[label,4]<cfg['minimum_support_pixels']:continue
                mask=labels==label;points=xyz[mask];xy=points[:,:2]
                low,high=xy.min(0),xy.max(0);extent=high-low
                if np.any(extent<cfg['minimum_visible_extent_m']) or np.any(extent>maximum_extent+.01):continue
                center_min,center_max=high-maximum_extent/2,low+maximum_extent/2
                if np.any(center_min>center_max):continue
                center=(center_min+center_max)/2;radius=(center_max-center_min)/2
                _,_,vt=np.linalg.svd(points-points.mean(0),full_matrices=False)
                rms=float(np.sqrt(np.mean(((points-points.mean(0))@vt[-1])**2)))
                if rms>cfg['plane_tolerance_m']*.75:continue
                margin=cfg['footprint_extra_margin_m']+cfg['calibration_sigma_m']*3
                conservative_half=maximum_extent/2+radius+margin
                covariance=np.diag(radius**2/3+cfg['calibration_sigma_m']**2)
                clipped=bool(np.any(mask[[0,-1]]) or np.any(mask[:,[0,-1]]))
                candidate={'valid':True,'reason':None,'center_xy_m':center.tolist(),
                    'top_height_m':top,'covariance_m2':covariance.tolist(),
                    'center_bounds_xy_m':{'minimum':center_min.tolist(),'maximum':center_max.tolist()},
                    'observed_bounds_xy_m':{'minimum':low.tolist(),'maximum':high.tolist()},
                    'conservative_table':{'frame':'odom','center_xy_m':center.tolist(),
                        'half_size_m':conservative_half.tolist(),'yaw_rad':0.,
                        'bottom_height_m':0.,'top_height_m':top,
                        'geometry_source':'depth_surface_plus_explicit_maximum_size_and_orientation_prior',
                        'full_footprint_measured':False},
                    'quality':{'method':'registered_depth_horizontal_plane_connected_surface',
                        'plane_rms_m':rms,'support_pixels':int(stats[label,4]),
                        'visible_extent_m':extent.tolist(),'surface_touches_image_edge':clipped,
                        'complete_platform_edges_verified':False,
                        'hidden_edges_bounded_by_prior':True,
                        'maximum_platform_size_m':list(cfg['maximum_platform_size_m']),
                        'maximum_axis_extent_m':maximum_extent.tolist(),
                        'orientation_prior_yaw_rad':cfg['orientation_prior_yaw_rad'],
                        'orientation_uncertainty_rad':cfg['orientation_uncertainty_rad'],
                        'height_prior_m':list(cfg['height_range_m']),'prior_status':cfg['prior_status']}}
                proposals.append((stats[label,4],candidate,mask))
        if not proposals:raise ValueError('no_bounded_horizontal_platform_surface')
        _,candidate,selected_mask=max(proposals,key=lambda item:item[0]);result.update(candidate)
    except (ValueError,TypeError,KeyError,np.linalg.LinAlgError) as error:
        result.update(valid=False,reason=str(error))
    return (result,selected_mask) if return_mask else result


def platform_search_prior(platform,*,target_id='platform-search',validity_s=.5):
    """Explicit exploration centre, NEVER a detected bowl or a grasp target."""
    if platform.get('valid') is not True or platform.get('kind')!='platform':raise ValueError('valid_platform_required')
    cov=np.zeros((3,3));cov[:2,:2]=platform['covariance_m2'];cov[2,2]=.003**2
    return {'valid':True,'kind':'platform_center_search_prior','is_bowl_detection':False,
        'source':'platform_search_prior','frame':'odom','target_id':target_id,
        'stamp_s':platform['stamp_s'],'valid_until_s':min(platform['valid_until_s'],platform['stamp_s']+validity_s),
        'position_m':[*platform['center_xy_m'],platform['top_height_m']],
        'covariance_m2':cov.tolist(),'assumption':'Bowl is somewhere on this support platform; search from its bounded estimated centre before any grasp.'}


def fuse_static_platform(observations,*,config=None):
    """Intersect hidden-edge centre bounds from actual successive observations.

    A view losing a near edge must not drag a static platform's mapped centre
    toward its far edge. Association is limited to overlapping world bounds and
    agreeing heights; no scene identifier or true platform position is used.
    """
    cfg=_settings(config);valid=[row for row in observations if row.get('valid') is True]
    if not valid:return {'valid':False,'kind':'platform','reason':'no_valid_platform_observation'}
    result=copy.deepcopy(valid[-1])
    low=np.max([row['center_bounds_xy_m']['minimum'] for row in valid],axis=0)
    high=np.min([row['center_bounds_xy_m']['maximum'] for row in valid],axis=0)
    heights=np.array([row['top_height_m'] for row in valid])
    if np.any(low>high) or np.ptp(heights)>.01:
        result.update(valid=False,reason='static_platform_observations_inconsistent');return result
    center=(low+high)/2;radius=(high-low)/2;maximum=_maximum_axis_extent(cfg)
    half=maximum/2+radius+cfg['footprint_extra_margin_m']+cfg['calibration_sigma_m']*3
    result.update(center_xy_m=center.tolist(),top_height_m=float(np.median(heights)),
        center_bounds_xy_m={'minimum':low.tolist(),'maximum':high.tolist()},
        covariance_m2=np.diag(radius**2/3+cfg['calibration_sigma_m']**2).tolist())
    result['conservative_table'].update(center_xy_m=center.tolist(),half_size_m=half.tolist(),top_height_m=result['top_height_m'])
    result['quality'].update(static_map_assumption=True,associated_depth_observations=len(valid),
        fusion='intersection_of_observed_center_bounds_no_scene_truth')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='NPZ containing depth_m, K, T_optical_to_odom; no labels')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    data=np.load(args.input,allow_pickle=False)
    result=detect_platform(data['depth_m'],data['K'],data['T_optical_to_odom'],float(data['stamp_s']) if 'stamp_s' in data else 0.)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['valid'] else 2


if __name__=='__main__':raise SystemExit(main())
