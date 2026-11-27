"""Dual learned-camera discovery -> forward parking -> wrist localization -> pick.

Run only on the Mac for verification. There is no hardware publisher/serial I/O.
Ground-truth object pose initializes the scene and computes metrics only; learned
mode never substitutes it for a failed detection or geometric estimate.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import yaml

from .model import ROOT,JOINTS,config
from .cli import write_json,provenance,policy_load
from .mobile_planner import (MobileCoordinator,plan_parking,screen_grasp_at_parking,
    footprint_clear,compose_pose,point_in_model,wrap)
from .mobile_scene import MobileScene,ArmScreen
from .mobile_vehicle import VehicleState,PathFollower,advance
from .perception import Segmenter,estimate_target
from .repaired_env import RepairedBowlEnv as BowlEnv
from .platform_perception import detect_platform,platform_search_prior,fuse_static_platform


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def platform_observation(scene,stamp_s,cfg,folder,serial):
    """Main camera finds the platform; it does not require or invent a bowl mask."""
    rgb,depth,k,t=scene.render('main_rgbd',with_depth=True)
    platform,mask=detect_platform(depth,k,t,stamp_s,config=cfg.get('platform_perception'),return_mask=True)
    folder.mkdir(parents=True,exist_ok=True);prefix=folder/f'{serial:04d}-platform'
    Image.fromarray(rgb).save(str(prefix)+'-rgb.png')
    if mask is not None:
        Image.fromarray((mask*255).astype(np.uint8)).save(str(prefix)+'-depth-plane-mask.png')
        overlay=rgb.copy();overlay[mask]=(overlay[mask]*.5+np.array([20,220,120])*.5).astype(np.uint8)
        Image.fromarray(overlay).save(str(prefix)+'-overlay.png')
    np.savez_compressed(str(prefix)+'-inputs.npz',depth_m=depth,K=k,T_optical_to_odom=t,stamp_s=stamp_s)
    record={'source':'main_rgbd_platform','stamp_s':stamp_s,'platform':platform,
        'learned_bowl_detection_performed':False,'uses_simulator_truth':False,
        'intrinsics':k.tolist(),'T_optical_to_odom':t.tolist()}
    write_json(Path(str(prefix)+'.json'),record)
    return platform,record


def _table_in_model(table,scene):
    local=copy.deepcopy(table)
    local['center_xy_m']=point_in_model([*table['center_xy_m'],0.],scene.arm_pose)[:2].tolist()
    local['yaw_rad']=wrap(table['yaw_rad']-scene.arm_pose[2]);local['frame']='arm_model'
    return local


def move_observation_joints(scene,goal,screen,platform,arm_cfg,*,stamp_s=0.,frames=None):
    """Rate-bounded kinematic search movement, with each pose collision screened.

    This is camera-view development, not calibrated actuator dynamics. No camera
    rotation is injected: optics move only with the actual model arm joints.
    """
    actual=scene.data.qpos[scene.ids[:5]].copy();goal=np.asarray(goal,float)
    max_step=arm_cfg['arm']['max_joint_velocity_rad_s']*.05
    count=max(1,int(np.ceil(np.max(np.abs(goal-actual))/max_step)))
    local=_table_in_model(platform['conservative_table'],scene)
    path=[]
    for index in range(1,count+1):
        q=actual+(goal-actual)*index/count
        evidence=screen.geometry_check(q,np.zeros(3),'wrist_search',local)
        if not evidence['valid']:
            return {'valid':False,'reason':'wrist_search_transition_collision','evidence':evidence,
                'path':path,'end_stamp_s':stamp_s}
        scene.set_arm_joints(q);stamp_s+=.05
        path.append({'stamp_s':stamp_s,'q_rad':q.tolist()})
        if frames is not None:frames.append(scene.snapshot(stamp_s,'wrist_search_motion'))
    return {'valid':True,'reason':None,'path':path,'end_stamp_s':stamp_s,
        'movement_model':'kinematic_joint_rate_limit_not_calibrated_servo_response'}


def search_wrist_views(scene,platform,segmenter,cfg,arm_cfg,folder,*,stamp_s=0.,screen=None):
    """Bounded real-mask search over whole-arm postures around a seen platform."""
    if platform.get('valid') is not True:raise ValueError('Valid platform observation required')
    folder.mkdir(parents=True,exist_ok=True);screen=screen or ArmScreen(arm_cfg,cfg)
    views=scene.wrist_search_candidates(platform);settings=cfg.get('wrist_search',{})
    budget=int(settings.get('maximum_views',5));required=int(settings.get('stable_frames',3))
    if not 1<=budget<=10 or not 1<=required<=5:raise ValueError('Invalid bounded wrist-search budget')
    current_stamp=float(stamp_s);records=[];frames=[];selected=None
    for view in views[:budget]:
        candidate={'view':view,'valid':False,'frames':[]}
        if not view['valid']:
            candidate['reason']='search_pose_ik_unreachable';records.append(candidate);continue
        movement=move_observation_joints(scene,view['q_rad'],screen,platform,arm_cfg,stamp_s=current_stamp,frames=frames)
        current_stamp=movement['end_stamp_s'];candidate['movement']=movement
        if not movement['valid']:
            candidate['reason']=movement['reason'];records.append(candidate);continue
        estimates=[]
        for frame_index in range(required):
            current_stamp+=.1
            rgb,_,k,t=scene.render('wrist',with_depth=False);detections=segmenter.infer(rgb)
            prefix=folder/f"view-{view['view_id']:02d}-frame-{frame_index:02d}"
            Image.fromarray(rgb).save(str(prefix)+'-rgb.png')
            candidates=[]
            for detection_index,detection in enumerate(detections):
                Image.fromarray((detection['mask']*255).astype(np.uint8)).save(str(prefix)+f'-mask-{detection_index}.png')
                target=estimate_target(detection,k,t,platform['top_height_m'],arm_cfg['bowl'],current_stamp,'wrist',
                    config={**cfg.get('perception',{}),'target_id':'bowl-1','now_s':current_stamp,
                        'table_height_sigma_m':max(.002,cfg.get('platform_perception',{}).get('calibration_sigma_m',.003)),
                        'calibration_status':'simulation_camera_extrinsics_not_measured_real_mount'})
                candidates.append({'confidence':detection['confidence'],'bbox':detection['bbox'],
                    'mask_pixels':int(detection['mask'].sum()),'target':target})
            valid=[item for item in candidates if item['target']['valid']]
            record={'stamp_s':current_stamp,'view_id':view['view_id'],'detections':candidates,
                'intrinsics':k.tolist(),'T_optical_to_odom':t.tolist(),'inference_s':segmenter.last_inference_s,
                'learned_vision':True,'mask_backend':'bundled_target_bowl_yolo11n_seg',
                'no_truth_fallback':True}
            write_json(Path(str(prefix)+'.json'),record);candidate['frames'].append(record)
            if len(valid)!=1:break
            estimates.append(valid[0]['target'])
        if len(estimates)==required:
            positions=np.array([item['position_m'] for item in estimates])
            spread=float(np.max(np.linalg.norm(positions-np.median(positions,axis=0),axis=1)))
            if spread<=settings.get('maximum_frame_shift_m',.008):
                target=copy.deepcopy(estimates[-1]);target['position_m']=np.median(positions,axis=0).tolist()
                target['quality']['stable_frames']=required;target['quality']['whole_arm_search_view_id']=view['view_id']
                target['quality']['uses_platform_truth']=False
                confidence=float(target['quality']['confidence']);residual=float(target['quality']['residual_rms_m'])
                score=confidence-10*residual-.05*float(np.trace(np.asarray(target['covariance_m2'])))
                candidate.update(valid=True,target=target,score=score,frame_spread_m=spread)
                selected=copy.deepcopy(candidate);records.append(candidate)
                # Stop at the first stable geometrically valid learned detection;
                # no claim that its view is a global optimum or fully unoccluded.
                break
        candidate['reason']='learned_mask_or_precise_pose_not_verified';records.append(candidate)
    report={'valid':selected is not None,'selected':selected,'views':records,
        'end_stamp_s':current_stamp,'replay_frames':frames,'uses_truth_fallback':False,
        'search_scope':'bounded_arm_postures_first_stable_valid_real_mask_not_global_best',
        'camera_mount_status':'fixed_configurable_simulation_mount_no_assumed_real_gimbal',
        'platform_map_assumption':'Platform is static in odom during this bounded camera-only search; captured platform frame is not relabelled as a fresh frame.',
        'target':selected['target'] if selected is not None else None}
    write_json(folder/'wrist-search.json',report)
    return report


def run_camera_bench(arm_cfg,mobile_cfg,segmenter,seed,output):
    """Small platform -> wrist-view probe; base is a declared fixed test fixture."""
    rng=np.random.default_rng(seed);cfg=copy.deepcopy(mobile_cfg);r=cfg['scenario']
    center=np.array([rng.uniform(*r['platform_world_x_m']),rng.uniform(*r['platform_world_y_m'])])
    bowl=np.r_[center+rng.uniform(*r['bowl_offset_on_platform_m'],size=2),arm_cfg['fixture']['top_height_m']]
    table={'frame':'odom','center_xy_m':center.tolist(),'half_size_m':r['platform_half_size_m'],
        'yaw_rad':0.,'top_height_m':float(bowl[2]),'bottom_height_m':0.}
    # Fixture placement is an experiment input, never a localization output.
    pose=cfg.get('camera_bench',{}).get('parked_vehicle_xy_yaw',[.48,0.,0.])
    scene=MobileScene(arm_cfg,cfg);output.mkdir(parents=True,exist_ok=True)
    result={'seed':seed,'valid':False,'scope':'fixed_parked_camera_bench_no_driving_no_grasp',
        'platform_localized':False,'wrist_localized':False,'uses_truth_fallback':False,
        'truth_for_metrics_only':{'platform':table,'bowl_position_m':bowl.tolist()},
        'parked_test_fixture_xy_yaw':pose,'configuration':cfg,'provenance':provenance(arm_cfg)}
    try:
        initial_pose=cfg.get('camera_bench',{}).get('main_observation_vehicle_xy_yaw',[0.,0.,0.])
        scene.set_scenario(table,bowl,initial_pose)
        platform,_=platform_observation(scene,0.,cfg,output/'main-platform',0)
        result['platform']=platform;result['platform_localized']=platform['valid']
        if not platform['valid']:
            result['reason']='main_platform_detection_failed';return result
        # This camera-only fixture explicitly omits autonomous approach. Never
        # count the fixture relocation as a planned/controlled parking success.
        scene.set_vehicle(pose)
        result['fixture_relocation_not_vehicle_control']={'from':initial_pose,'to':pose,
            'parking_success_claimed':False}
        search=search_wrist_views(scene,platform,segmenter,cfg,arm_cfg,output/'wrist-search',stamp_s=.1)
        result.update(valid=search['valid'],wrist_localized=search['valid'],wrist_search=search)
        if search['valid']:
            result['wrist_position_error_metric_m']=float(np.linalg.norm(np.asarray(search['target']['position_m'])-bowl))
            handoff={'valid':True,'scope':'camera-derived_grasp_input_at_declared_fixed_parked_fixture',
                'target':search['target'],'observation_q_rad':search['selected']['view']['q_rad'],
                'parked_vehicle_xy_yaw':list(pose),'arm_model_xy_yaw':scene.arm_pose.tolist(),
                'target_model_m':point_in_model(search['target']['position_m'],scene.arm_pose).tolist(),
                'conservative_platform':platform['conservative_table'],
                'end_stamp_s':search['end_stamp_s'],'source':'real_target_bowl_segmenter_and_monocular_rim_fit_on_rendered_image',
                'uses_truth_fallback':False,'autonomous_parking_verified':False,'hardware_ready':False}
            result['grasp_handoff']=handoff;write_json(output/'localized-grasp-handoff.json',handoff)
        else:result['reason']='bounded_wrist_search_exhausted'
        result['platform_center_error_metric_m']=float(np.linalg.norm(np.asarray(platform['center_xy_m'])-center))
        return result
    finally:
        write_json(output/'camera-bench.json',result);scene.close()


def run_platform_episode(arm_cfg,mobile_cfg,segmenter,seed,output,*,localization_only=False):
    """Observed support -> conservative parking -> bicycle -> wrist search -> grasp.

    Only depth observations construct navigation geometry. Scene truth is used
    to initialize the physical environment, check actual collisions, and score.
    No platform-centre prior is counted as a bowl detection.
    """
    rng=np.random.default_rng(seed);cfg=copy.deepcopy(mobile_cfg);r=cfg['scenario']
    center=np.array([rng.uniform(*r['platform_world_x_m']),rng.uniform(*r['platform_world_y_m'])])
    bowl=np.r_[center+rng.uniform(*r['bowl_offset_on_platform_m'],size=2),arm_cfg['fixture']['top_height_m']]
    truth_table={'frame':'odom','center_xy_m':center.tolist(),'half_size_m':r['platform_half_size_m'],
        'yaw_rad':0.,'top_height_m':float(bowl[2]),'bottom_height_m':0.}
    state=VehicleState(x_m=rng.uniform(*r['start_x_m']),y_m=rng.uniform(*r['start_y_m']),yaw_rad=math.radians(rng.uniform(*r['start_yaw_deg'])))
    flow=MobileCoordinator(cfg);scene=MobileScene(arm_cfg,cfg);screen=ArmScreen(arm_cfg,cfg)
    output.mkdir(parents=True,exist_ok=True);rows=[];frames=[];platforms=[];follower=None
    t=0.;dt=cfg['runtime']['control_dt_s'];next_vision=0.;platform=None;prior=None;measured_table=None
    committed_center=None;map_refinements=0;plan_time=0.
    result={'seed':seed,'is_success':False,'failure_reason':None,'platform_localized':False,
        'parking_selected':False,'parked':False,'wrist_localized':False,'grasp_started':False,
        'learned_vision':True,'main_bowl_detection_claimed':False,'localization_only':localization_only,
        'hardware_ready':False,'uses_truth_fallback':False,'configuration':cfg,'provenance':provenance(arm_cfg),
        'truth_for_environment_and_metrics_only':{'table':truth_table,'bowl_position_m':bowl.tolist()}}
    def make_route(target,map_table,now):
        nonlocal measured_table
        measured_table=map_table
        def solve(point,seed=None):return screen.solve_grasp(point,seed=seed)
        report=plan_parking(target,measured_table,state.odometry(now),cfg,solve,
            geometry_check=screen.geometry_check,camera_pose_fn=scene.wrist_camera_pose_in_model,now_s=now)
        report['planning_target_semantics']='Observed platform-centre search prior; bowl has not been detected.'
        safe=[]
        for candidate in sorted((item for item in report['candidates'] if item.get('valid')),key=lambda item:item['score']):
            rejected=None
            for sample in candidate['path']['samples']:
                check=navigation_arm_clear(screen,scene.navigation_q,[sample['x_m'],sample['y_m'],sample['yaw_rad']],measured_table,cfg)
                if not check['valid']:rejected=check;break
            candidate['navigation_arm_screen']={'valid':rejected is None,'rejection':rejected}
            if rejected is None:safe.append(candidate)
        report['valid']=bool(safe);report['selected']=safe[0] if safe else None
        return report
    try:
        scene.set_scenario(truth_table,bowl,state.pose);scene.set_arm_joints(scene.navigation_q)
        while t<cfg['runtime']['maximum_approach_s']:
            if flow.state in ('acquire_main','approach','stop_confirm') and t+1e-9>=next_vision:
                observed,_=platform_observation(scene,t,cfg,output/'main-platform',len(platforms));platforms.append(observed)
                next_vision=t+1/cfg['runtime']['vision_hz']
                if observed['valid']:
                    platform=fuse_static_platform(platforms,config=cfg.get('platform_perception'))
                    if not platform['valid']:
                        result['failure_reason']='platform_depth_observations_inconsistent';break
                    prior=platform_search_prior(platform,target_id='station-1')
                    result['platform_localized']=True
                    if measured_table is not None:measured_table=copy.deepcopy(platform['conservative_table'])
                    # Refining the static platform map is permitted before parking;
                    # never relabel the old coarse plan as a precise bowl target.
                    changed=0. if committed_center is None else float(np.linalg.norm(np.asarray(prior['position_m'])-committed_center))
                    sigma=float(np.sqrt(np.max(np.linalg.eigvalsh(prior['covariance_m2']))))
                    if flow.state=='approach' and map_refinements<2 and t-plan_time>=.8 and changed>.025 and sigma<.035:
                        refined=make_route(prior,copy.deepcopy(platform['conservative_table']),t)
                        write_json(output/f'platform-map-refinement-{map_refinements+1}.json',refined)
                        if refined['valid']:
                            flow.select_plan(refined);follower=PathFollower(refined['selected']['path'],cfg)
                            committed_center=np.asarray(prior['position_m']);plan_time=t;map_refinements+=1
            if flow.state=='acquire_main':
                intent=flow.update(t,state.odometry(t),main_target=prior)
                if intent.get('plan_requested'):
                    measured_table=copy.deepcopy(platform['conservative_table'])
                    report=make_route(prior,measured_table,t)
                    safe=[report['selected']] if report['valid'] else []
                    write_json(output/'platform-parking-plan.json',report)
                    committed_center=np.asarray(prior['position_m']);plan_time=t
                    if not safe:result['failure_reason']='no_feasible_parking_for_observed_platform_bounds';break
                    flow.select_plan(report);follower=PathFollower(report['selected']['path'],cfg);result['parking_selected']=True
                    intent=flow.snapshot()
                elif t>=cfg['runtime']['main_acquisition_timeout_s']:
                    result['failure_reason']='main_platform_acquisition_failed';break
            else:
                # Obstacles/clearance used by the controller come from the
                # observed conservative geometry, not the scene's true box.
                clear,_=footprint_clear(state.pose,measured_table,cfg['vehicle'],cfg['planning']['obstacles'])
                arm_clear=navigation_arm_clear(screen,scene.navigation_q,state.pose,measured_table,cfg)
                intent=flow.update(t,state.odometry(t),main_target=prior,
                    obstacle_clear=clear,arm_stowed=arm_clear['valid'])
            rows.append({'t_s':t,'vehicle':state.record(),'intent':intent,'platform_prior':prior})
            frames.append(scene.snapshot(t,'platform:'+flow.state))
            if flow.state=='stop':result['failure_reason']=flow.reason;break
            if flow.state=='acquire_wrist':
                result['parked']=True
                if platform is None:result['failure_reason']='platform_map_missing';break
                search=search_wrist_views(scene,platform,segmenter,cfg,arm_cfg,output/'wrist-search',stamp_s=t,screen=screen)
                result['wrist_search']=search;frames.extend(search['replay_frames'])
                # The simulated base stays at the confirmed parked state while
                # bounded arm motion runs. Supply continuous quiet telemetry.
                for sample_time in np.arange(t+dt,search['end_stamp_s']-dt/2,dt):
                    flow.update(float(sample_time),state.odometry(float(sample_time)),main_target=prior)
                t=search['end_stamp_s']
                if not search['valid']:result['failure_reason']='bounded_wrist_search_exhausted';break
                target=copy.deepcopy(search['target']);target['station_id']='station-1'
                target['station_association']='estimated_platform_bounds'
                point=np.asarray(target['position_m'][:2]);half=np.asarray(measured_table['half_size_m'])
                if np.any(np.abs(point-np.asarray(measured_table['center_xy_m']))>half):
                    result['failure_reason']='wrist_bowl_outside_estimated_station_bounds';break
                result['wrist_localized']=True
                def solve(point,seed=None):return screen.solve_grasp(point,seed=seed)
                wrist_check=screen_grasp_at_parking(target,measured_table,state.pose,cfg,solve,
                    geometry_check=screen.geometry_check,camera_pose_fn=scene.wrist_camera_pose_in_model,now_s=t)
                write_json(output/'wrist-grasp-screen.json',wrist_check)
                if not wrist_check['valid']:
                    result['failure_reason']='wrist_target_requires_bounded_reparking_or_arm_replan'
                    result['reparking_request']={'requested':True,'executed':False,'maximum_retries':1,'reason':wrist_check['reason']};break
                intent=flow.update(t,state.odometry(t),main_target=prior,wrist_target=target,precise_grasp_feasible=True)
                if not intent['grasp_ready']:result['failure_reason']=intent.get('reason') or 'grasp_gate_not_ready';break
                local_target=point_in_model(target['position_m'],scene.arm_pose)
                handoff={'valid':True,'target':target,'target_model_m':local_target.tolist(),
                    'observation_q_rad':search['selected']['view']['q_rad'],'parked_vehicle_xy_yaw':state.pose,
                    'arm_model_xy_yaw':scene.arm_pose.tolist(),'conservative_platform':measured_table,
                    'autonomous_parking_verified_in_simulation':True,'uses_truth_fallback':False,'hardware_ready':False}
                result['grasp_handoff']=handoff;write_json(output/'localized-grasp-handoff.json',handoff)
                result['localization_pipeline_success']=True
                if localization_only:
                    result['localization_pipeline_success']=True;result['failure_reason']=None;break
                result['grasp_started']=True
                # These values define the physical scene, not the controller's
                # measured target. The grasp estimate is exclusively from wrist.
                pick_cfg=copy.deepcopy(arm_cfg)
                pick_cfg['fixture']['grasp_height_above_table_m']=cfg['arm']['grasp_height_above_table_m']
                pick_cfg['fixture']['bowl_world_xy_m']=bowl[:2].tolist()
                pick_cfg['fixture']['platform_world_xy_m']=truth_table['center_xy_m'];pick_cfg['fixture']['half_size_m']=truth_table['half_size_m']
                arm=BowlEnv(pick_cfg);obs,info=arm.reset(seed=seed,options={'parking':scene.arm_pose,
                    'target_estimate_base_m':local_target,'observation_source':'frozen_wrist_estimate','trace':True})
                arm.home_ik_seed=arm.command.copy()
                for name,idx in zip(scene.joint_names,scene.ids):
                    arm.data.qpos[int(arm.model.joint(name).qposadr[0])]=scene.data.qpos[idx]
                q=scene.data.qpos[scene.ids[:5]].copy();arm.command=q.copy();arm.desired_q=q.copy()
                arm.data.ctrl[:5]=q;arm.commands.clear();arm.commands.append((0.,q.copy()))
                import mujoco
                mujoco.mj_forward(arm.model,arm.data)
                transit=float(np.max(np.abs(np.asarray(wrist_check['arm']['joint_path'][0]['q_rad'])-q)))/arm_cfg['arm']['max_joint_velocity_rad_s']
                arm.cfg['task']['durations_s'][0]+=transit
                arm.previous_distance=float(np.linalg.norm(arm.data.site_xpos[arm.site]-arm.nominal))
                result['grasp_used_config']=copy.deepcopy(arm.cfg)
                while True:
                    _,_,done,_,info=arm.step(np.zeros(3));scene.apply_grasp_state(arm)
                    frames.append(scene.snapshot(t+arm.elapsed,'grasp:'+info['phase']))
                    if done:break
                result.update(is_success=bool(info['is_success']),failure_reason=info['failure_reason'],grasp_result=info)
                write_json(output/'grasp-trace.json',arm.rows);break
            speed,steer=follower.command(state) if flow.state=='approach' else (0.,0.)
            if follower is not None and follower.failure_reason:result['failure_reason']=follower.failure_reason;break
            samples=advance(state,speed,steer,dt,cfg)
            physical_collision=False
            for pose in samples:
                # Actual-scene collision is an independent stop/metric, not a
                # replacement geometry source for navigation or target fitting.
                actual_clear,_=footprint_clear(pose,truth_table,cfg['vehicle'],cfg['planning']['obstacles'])
                if not actual_clear:physical_collision=True;break
            if physical_collision:result['failure_reason']='actual_environment_vehicle_collision';break
            scene.set_vehicle(state.pose);t=round(t+dt,10)
        else:result['failure_reason']='platform_approach_timeout'
    except Exception as error:
        import traceback
        result.update(failure_reason='runtime_error',exception=repr(error),traceback=traceback.format_exc())
    finally:
        result['simulation_time_s']=t;result['final_vehicle_xy_yaw']=state.pose
        result['platform_observation_count']=len(platforms);result['bounded_map_replans']=map_refinements
        result['recognition_success']=result['platform_localized'] and result['wrist_localized']
        result['parking_success']=result['parked'];result['precise_localization_success']=result['wrist_localized']
        result['localization_pipeline_success']=bool(result.get('localization_pipeline_success',False))
        write_json(output/'platform-episode.json',result);write_json(output/'mobile-trace.json',rows)
        write_json(output/'mobile-replay.json',frames);scene.close()
    return result


def camera_observation(scene,segmenter,source,stamp_s,cfg,arm_cfg,folder,serial):
    rgb,depth,k,t=scene.render(source,with_depth=source=='main_rgbd')
    detections=segmenter.infer(rgb);targets=[];candidate_info=[]
    perception_cfg={**cfg['perception'],'target_id':'bowl-1','now_s':stamp_s,
        'calibration_status':'simulation_only_unmeasured_camera'}
    for detection in detections:
        target=estimate_target(detection,k,t,scene.table['top_height_m'],arm_cfg['bowl'],stamp_s,source,
            depth_m=depth if source=='main_rgbd' else None,config=perception_cfg)
        targets.append(target);candidate_info.append({**{k:v for k,v in detection.items() if k!='mask'},'target':target})
    good=[(d,target) for d,target in zip(detections,targets) if target['valid']]
    valid=len(good)==1
    target=copy.deepcopy(good[0][1]) if valid else {'valid':False,'source':source,'stamp_s':stamp_s,'frame':'odom',
        'reason':'ambiguous_multiple_bowls' if len(good)>1 else 'no_valid_learned_bowl_pose','target_id':'bowl-1'}
    # Save exact input evidence for every frame; no identity/segmentation truth is supplied to YOLO.
    folder.mkdir(parents=True,exist_ok=True);prefix=folder/f'{serial:04d}-{source}'
    save_image=serial%cfg['runtime'].get('save_rgb_every_n_frames',5)==0 or not valid
    if save_image:
        Image.fromarray(rgb).save(str(prefix)+'-rgb.png')
        for i,d in enumerate(detections):Image.fromarray((d['mask']*255).astype(np.uint8)).save(str(prefix)+f'-mask-{i}.png')
    if depth is not None and (serial%cfg['runtime']['save_depth_every_n_frames']==0 or not valid):
        np.save(str(prefix)+'-depth-m.npy',depth)
    record={'source':source,'stamp_s':stamp_s,'intrinsics':k.tolist(),'T_optical_to_odom':t.tolist(),
        'inference_s':segmenter.last_inference_s,'detections':candidate_info,'selected_target':target,
        'calibration_status':'simulation_only_not_real_camera_calibration','learned_vision':True}
    write_json(Path(str(prefix)+'.json'),record)
    return target,record


def smooth_target(target,history,required,max_shift):
    if not target.get('valid'):history.clear();return None
    position=np.asarray(target['position_m'])
    if history and np.linalg.norm(position-np.asarray(history[-1]['position_m']))>max_shift:history.clear()
    history.append(copy.deepcopy(target));history[:]=history[-required:]
    if len(history)<required:return None
    out=copy.deepcopy(target);out['position_m']=np.median([t['position_m'] for t in history],axis=0).tolist()
    # Do not divide shared calibration uncertainty by the number of frames.
    out['covariance_m2']=np.max(np.stack([t['covariance_m2'] for t in history]),axis=0).tolist()
    out['quality']['stable_frames']=len(history)
    return out


def navigation_arm_clear(screen,q_observation,vehicle_pose,table,mobile):
    model_pose=compose_pose(vehicle_pose,mobile['arm']['mount_xy_yaw_m_rad'])
    table_local=copy.deepcopy(table);table_local['center_xy_m']=point_in_model([*table['center_xy_m'],0.],model_pose)[:2].tolist()
    table_local['yaw_rad']=wrap(table['yaw_rad']-model_pose[2])
    return screen.geometry_check(q_observation,np.array([0.,0.,0.]),'navigation',table_local)


def run_mobile_episode(arm_cfg,mobile_cfg,segmenter,seed,output,policy=None):
    rng=np.random.default_rng(seed);cfg=copy.deepcopy(mobile_cfg)
    r=cfg['scenario'];table_center=np.array([rng.uniform(*r['platform_world_x_m']),rng.uniform(*r['platform_world_y_m'])])
    target_truth=np.r_[table_center+rng.uniform(*r['bowl_offset_on_platform_m'],size=2),arm_cfg['fixture']['top_height_m']]
    table={'frame':'odom','center_xy_m':table_center.tolist(),'half_size_m':r['platform_half_size_m'],
        'yaw_rad':0.,'top_height_m':float(target_truth[2]),'bottom_height_m':0.}
    state=VehicleState(x_m=rng.uniform(*r['start_x_m']),y_m=rng.uniform(*r['start_y_m']),yaw_rad=math.radians(rng.uniform(*r['start_yaw_deg'])))
    scene=None;flow=MobileCoordinator(cfg);follower=None
    output.mkdir(parents=True,exist_ok=True);rows=[];frames=[];vision_rows=[]
    t=0.;dt=cfg['runtime']['control_dt_s'];next_vision=0.;serial=0
    main_target=None;wrist_target=None;main_history=[];wrist_history=[];selected=None;wrist_check=None
    result={'seed':seed,'is_success':False,'failure_reason':None,'learned_vision':True,
        'main_recognized':False,'parking_selected':False,'parked':False,'wrist_localized':False,'grasp_started':False,
        'hardware_ready':False,'vehicle_dynamics':'unmeasured_forward_bicycle_surrogate','physics':'MuJoCo_arm_after_stationary_handoff',
        'grasp_controller':'zero_residual_baseline' if policy is None else 'optional_v3_residual_policy',
        'grasp_visual_feedback':'stationary_wrist_estimate_frozen_during_grasp_not_continuous_tracking',
        'scene_truth_for_initialization_and_metrics_only':{'bowl_world_m':target_truth.tolist(),'start_pose':state.pose},
        'weights_sha256':digest(ROOT/cfg['vision_model']['weights']),'configuration':cfg,'provenance':provenance(arm_cfg)}
    episode_started=time.perf_counter();wrist_stage_started=None
    try:
        scene=MobileScene(arm_cfg,cfg);scene.set_scenario(table,target_truth,state.pose);screen=ArmScreen(arm_cfg,cfg)
        def solve_arm(point,seed=None):return screen.ik.solve(point,seed=scene.observation_q if seed is None else seed)
        def transition_clear(joint_path,pose):
            if not joint_path:return False
            end=np.asarray(joint_path[0]['q_rad']);start=scene.observation_q
            for u in np.linspace(0.,1.,max(2,int(np.ceil(np.max(np.abs(end-start))/.05))+1)):
                if not navigation_arm_clear(screen,start+(end-start)*u,pose,table,cfg)['valid']:return False
            return True
        while t<cfg['runtime']['maximum_approach_s']:
            source='wrist' if flow.state in ('acquire_wrist','grasp_ready') else 'main_rgbd'
            if t+1e-9>=next_vision:
                target,record=camera_observation(scene,segmenter,source,t,cfg,arm_cfg,output/'vision',serial)
                serial+=1;next_vision=t+1/cfg['runtime']['vision_hz'];vision_rows.append(record)
                if source=='main_rgbd':
                    stable=smooth_target(target,main_history,cfg['runtime']['main_stable_frames'],cfg['runtime']['main_max_frame_shift_m'])
                    if stable is not None:main_target=stable;result['main_recognized']=True
                else:
                    stable=smooth_target(target,wrist_history,cfg['runtime']['wrist_stable_frames'],cfg['runtime']['wrist_max_frame_shift_m'])
                    if stable is not None:wrist_target=stable;result['wrist_localized']=True
            if flow.state=='acquire_main':
                intent=flow.update(t,state.odometry(t),main_target=main_target)
                if intent.get('plan_requested'):
                    report=plan_parking(main_target,table,state.odometry(t),cfg,solve_arm,
                        geometry_check=screen.geometry_check,camera_pose_fn=scene.wrist_camera_pose_in_model,now_s=t)
                    # Screen the fixed observation arm along the complete selected navigation path.
                    viable=sorted((c for c in report['candidates'] if c.get('valid')),key=lambda c:(c['score'],c['id']))
                    safe=[]
                    for candidate in viable:
                        bad=None
                        for p in candidate['path']['samples']:
                            check=navigation_arm_clear(screen,scene.observation_q,[p['x_m'],p['y_m'],p['yaw_rad']],table,cfg)
                            if not check['valid']:bad=check;break
                        if bad is None and not transition_clear(candidate['joint_path'],candidate['parking_xy_yaw']):bad={'valid':False,'reason':'observation_to_pregrasp_transition_collision'}
                        candidate['navigation_arm_screen']={'valid':bad is None,'rejection':bad}
                        if bad is None:safe.append(candidate)
                    if safe:report['selected']=safe[0];report['valid']=True
                    else:report['selected']=None;report['valid']=False;report['rejections'].append('navigation_arm_screen_rejected_all_candidates')
                    write_json(output/'parking-plan.json',report)
                    if not report['valid']:
                        result['failure_reason']='no_feasible_vision_guided_parking';break
                    flow.select_plan(report);selected=report['selected'];follower=PathFollower(selected['path'],cfg);result['parking_selected']=True
                    intent=flow.snapshot()
                elif t>=cfg['runtime']['main_acquisition_timeout_s']:
                    result['failure_reason']='main_vision_acquisition_failed';break
            else:
                precise=False
                if flow.state=='acquire_wrist':
                    if wrist_stage_started is None:wrist_stage_started=t;result['parked']=True
                    if wrist_target and wrist_target.get('valid'):
                        wrist_check=screen_grasp_at_parking(wrist_target,table,state.pose,cfg,solve_arm,
                            geometry_check=screen.geometry_check,camera_pose_fn=scene.wrist_camera_pose_in_model,now_s=t)
                        write_json(output/'wrist-grasp-screen.json',wrist_check);precise=bool(wrist_check['valid'] and transition_clear(wrist_check.get('arm',{}).get('joint_path',[]),state.pose) and state.speed_m_s==0. and state.yaw_rate_rad_s==0.)
                    if t-wrist_stage_started>cfg['runtime']['wrist_acquisition_timeout_s']:
                        result['failure_reason']='wrist_vision_or_precise_grasp_screen_failed';break
                clear,reason=footprint_clear(state.pose,table,cfg['vehicle'],cfg['planning']['obstacles'])
                arm_clear=navigation_arm_clear(screen,scene.observation_q,state.pose,table,cfg)
                intent=flow.update(t,state.odometry(t),main_target=main_target,wrist_target=wrist_target,
                    obstacle_clear=clear,arm_stowed=arm_clear['valid'],precise_grasp_feasible=precise)
            rows.append({'t_s':t,'vehicle':state.record(),'intent':intent,'main_target':main_target,'wrist_target':wrist_target})
            frames.append(scene.snapshot(t,flow.state))
            if flow.state=='stop':result['failure_reason']=flow.reason;break
            if intent['grasp_ready']:
                result['parked']=True;result['grasp_started']=True
                actual_arm=compose_pose(state.pose,cfg['arm']['mount_xy_yaw_m_rad'])
                local_target=point_in_model(wrist_target['position_m'],actual_arm)
                pick_cfg=copy.deepcopy(arm_cfg)
                pick_cfg['fixture']['grasp_height_above_table_m']=cfg['arm']['grasp_height_above_table_m']
                pick_cfg['fixture']['bowl_world_xy_m']=target_truth[:2].tolist()
                pick_cfg['fixture']['platform_world_xy_m']=table['center_xy_m'];pick_cfg['fixture']['half_size_m']=table['half_size_m'];pick_cfg['sensor']['observation_noise_std_m']=0.
                arm=BowlEnv(pick_cfg);obs,info=arm.reset(seed=seed,options={'parking':actual_arm,
                    'target_estimate_base_m':local_target,'target_estimate_age_s':max(0.,t-wrist_target['stamp_s']),'observation_source':'frozen_wrist_estimate','trace':True})
                # Preserve the observation arm state at handoff. Move to pre-grasp in place
                # using the existing coordinated, bounded controller; do not teleport q.
                arm.data.qpos[arm.qids]=scene.observation_q;arm.command=scene.observation_q.copy();arm.desired_q=scene.observation_q.copy()
                arm.data.ctrl[:5]=scene.observation_q;arm.commands.clear();arm.commands.append((0.,scene.observation_q.copy()))
                import mujoco
                mujoco.mj_forward(arm.model,arm.data)
                arm.previous_distance=float(np.linalg.norm(arm.data.site_xpos[arm.site]-arm.nominal))
                obs=arm._observation()
                result['parking_brake_confirmed']=state.speed_m_s==0. and state.yaw_rate_rad_s==0.
                while True:
                    action=np.zeros(3) if policy is None else policy.predict(obs,deterministic=True)[0]
                    obs,_,done,_,info=arm.step(action);scene.apply_grasp_state(arm)
                    frames.append(scene.snapshot(t+arm.elapsed,'grasp:'+info['phase']))
                    if done:break
                result.update(is_success=bool(info['is_success']),failure_reason=info['failure_reason'],grasp_result=info,
                    wrist_position_error_metric_m=float(np.linalg.norm(np.asarray(wrist_target['position_m'])-target_truth)))
                write_json(output/'grasp-trace.json',arm.rows);write_json(output/'grasp-episode.json',info)
                break
            speed,steer=(follower.command(state) if flow.state=='approach' else (0.,0.))
            if follower is not None and follower.failure_reason:
                result['failure_reason']=follower.failure_reason;break
            samples=advance(state,speed,steer,dt,cfg)
            collision=None
            for pose in samples:
                clear,reason=footprint_clear(pose,table,cfg['vehicle'],cfg['planning']['obstacles'])
                if not clear:collision=reason;break
                check=navigation_arm_clear(screen,scene.observation_q,pose,table,cfg)
                if not check['valid']:collision=check['reason'];break
            if collision:result['failure_reason']='vehicle_or_navigation_arm_collision:'+str(collision);break
            scene.set_vehicle(state.pose);t=round(t+dt,10)
        else:result['failure_reason']='mobile_approach_timeout'
    except Exception as error:
        # Store the exception as a failed episode; never replace it with scene truth.
        import traceback
        result['failure_reason']='runtime_error';result['exception']=repr(error);result['traceback']=traceback.format_exc()
    finally:
        result['wall_s']=time.perf_counter()-episode_started;result['approach_sim_s']=t
        result['vision_inference_s']=sum(float(r['inference_s'] or 0.) for r in vision_rows)
        result['vision_frames']=len(vision_rows);result['last_main_target']=main_target;result['last_wrist_target']=wrist_target
        result['final_vehicle_xy_yaw']=state.pose
        result['recognition_success']=result['main_recognized'];result['parking_success']=result['parked'];result['precise_localization_success']=result['wrist_localized']
        write_json(output/'mobile-episode.json',result);write_json(output/'mobile-trace.json',rows);write_json(output/'mobile-replay.json',frames)
        write_json(output/'scene.json',{'table':table,'bowl_world_m':target_truth.tolist(),'initial_vehicle_xy_yaw':frames[0]['vehicle_xy_yaw'] if frames else state.pose})
        if scene is not None:scene.close()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'config/train.yaml')
    p.add_argument('--mobile-config',type=Path,default=ROOT/'config/mobile.yaml')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('evaluate');q.add_argument('--episodes',type=int,default=3);q.add_argument('--seed',type=int,default=70000)
    q.add_argument('--vision-device',choices=['cpu','mps'],default='cpu');q.add_argument('--policy',type=Path);q.add_argument('--policy-device',choices=['cpu','mps'],default='cpu');q.add_argument('--output',type=Path,default=ROOT/'outputs/mobile')
    q=sub.add_parser('gui');q.add_argument('--episode',type=Path,required=True);q.add_argument('--duration',type=float,default=60.)
    q=sub.add_parser('camera-bench',help='Platform geometry + real learned wrist search at an explicit parked fixture; no drive/grasp success claim')
    q.add_argument('--episodes',type=int,default=3);q.add_argument('--seed',type=int,default=71000)
    q.add_argument('--vision-device',choices=['cpu','mps'],default='cpu')
    q.add_argument('--output',type=Path,default=ROOT/'outputs/camera-bench')
    q=sub.add_parser('platform-evaluate',help='Depth platform -> bicycle approach -> stopped real-mask wrist search -> optional physical grasp')
    q.add_argument('--episodes',type=int,default=3);q.add_argument('--seed',type=int,default=72000)
    q.add_argument('--vision-device',choices=['cpu','mps'],default='cpu')
    q.add_argument('--localization-only',action='store_true',help='Stop after actual driving and wrist localization; do not count this as grasp success')
    q.add_argument('--output',type=Path,default=ROOT/'outputs/platform-mobile')
    a=p.parse_args();cfg=config(a.config);mobile=yaml.safe_load(a.mobile_config.read_text())
    if a.command=='platform-evaluate':
        if not 1<=a.episodes<=20:p.error('Use 1–20 bounded platform-pipeline episodes')
        if (a.output/'platform-evaluation.json').exists():raise SystemExit('Use a fresh platform-pipeline output directory')
        if a.vision_device=='cpu':
            import torch
            torch.set_num_threads(int(mobile.get('runtime',{}).get('vision_cpu_threads',2)))
        segmenter=Segmenter(ROOT/mobile['vision_model']['weights'],a.vision_device,mobile['vision_model']['confidence'],mobile['vision_model']['imgsz'])
        results=[run_platform_episode(cfg,mobile,segmenter,a.seed+index,a.output/f'episode-{a.seed+index}',localization_only=a.localization_only) for index in range(a.episodes)]
        summary={'episodes':len(results),'successes':sum(row['is_success'] for row in results),
            'success_rate':sum(row['is_success'] for row in results)/len(results),
            'localization_pipeline_successes':sum(row['localization_pipeline_success'] for row in results),
            'stage_success_rates':{key:sum(bool(row[key]) for row in results)/len(results) for key in
                ('platform_localized','parking_selected','parked','wrist_localized','grasp_started')},
            'failures':dict(Counter(row['failure_reason'] for row in results if row['failure_reason'])),
            'localization_only':a.localization_only,'vision_backend':'ultralytics_yolo_segmenter','learned_vision':True,
            'main_platform_backend':'registered_depth_plane_with_explicit_size_prior_not_semantic_network',
            'pipeline':'observed_platform_drive_wrist_physical_grasp_v4',
            'vision_device':str(segmenter.device), 'weights_sha256':digest(ROOT/mobile['vision_model']['weights']),
            'mobile_config_sha256':digest(a.mobile_config),
            'configuration':mobile,'provenance':provenance(cfg),'trials':results,'hardware_ready':False}
        write_json(a.output/'platform-evaluation.json',summary)
        print(json.dumps({key:value for key,value in summary.items() if key not in ('configuration','provenance','trials')},indent=2))
        passed=summary['localization_pipeline_successes']==a.episodes if a.localization_only else summary['successes']==a.episodes
        return 0 if passed else 2
    if a.command=='camera-bench':
        if not 1<=a.episodes<=20:p.error('Use 1–20 bounded camera-only probe episodes')
        if (a.output/'camera-bench-summary.json').exists():raise SystemExit('Use a fresh camera output directory')
        if a.vision_device=='cpu':
            import torch
            torch.set_num_threads(int(mobile.get('runtime',{}).get('vision_cpu_threads',2)))
        segmenter=Segmenter(ROOT/mobile['vision_model']['weights'],a.vision_device,mobile['vision_model']['confidence'],mobile['vision_model']['imgsz'])
        results=[run_camera_bench(cfg,mobile,segmenter,a.seed+index,a.output/f'episode-{a.seed+index}') for index in range(a.episodes)]
        summary={'scope':'camera_only_fixed_parked_test_fixture_not_autonomous_approach_or_grasp',
            'episodes':len(results),'platform_localized':sum(row['platform_localized'] for row in results),
            'wrist_localized':sum(row['wrist_localized'] for row in results),
            'vision_backend':'bundled_target_bowl_yolo11n_seg','truth_fallback_used':False,
            'trials':results,'provenance':provenance(cfg),'configuration':mobile,'hardware_ready':False}
        write_json(a.output/'camera-bench-summary.json',summary)
        print(json.dumps({key:value for key,value in summary.items() if key not in ('trials','provenance','configuration')},indent=2))
        return 0 if summary['wrist_localized']==a.episodes else 2
    if a.command=='gui':
        import mujoco.viewer
        scene_meta=json.loads((a.episode/'scene.json').read_text());frames=json.loads((a.episode/'mobile-replay.json').read_text())
        if not frames:raise SystemExit('No replay frames; inspect mobile-episode.json for the initialization failure')
        result=json.loads((a.episode/'mobile-episode.json').read_text());mobile=result['configuration'];cfg=result['provenance']['config']
        scene=MobileScene(cfg,mobile);scene.set_scenario(scene_meta['table'],scene_meta['bowl_world_m'],scene_meta['initial_vehicle_xy_yaw'])
        with mujoco.viewer.launch_passive(scene.model,scene.data) as viewer:
            viewer.cam.lookat[:]=[.6,0,.4];viewer.cam.distance=2.;viewer.cam.azimuth=140;viewer.cam.elevation=-25
            started=time.perf_counter();i=0
            while viewer.is_running() and time.perf_counter()-started<a.duration:
                with viewer.lock():scene.apply_snapshot(frames[i]);i=(i+1)%len(frames)
                viewer.sync();time.sleep(.05)
        scene.close();return 0
    if not 0<a.episodes<=200:p.error('Use 1–200 episodes and separate output folders')
    if (a.output/'mobile-evaluation.json').exists():raise SystemExit('Use a fresh output directory')
    policy=policy_load(a.policy,cfg,a.policy_device) if a.policy else None
    weights=ROOT/mobile['vision_model']['weights'];segmenter=Segmenter(weights,a.vision_device,mobile['vision_model']['confidence'],mobile['vision_model']['imgsz'])
    results=[]
    for i in range(a.episodes):
        r=run_mobile_episode(cfg,mobile,segmenter,a.seed+i,a.output/f'episode-{a.seed+i}',policy=policy);results.append(r)
        print(f'MOBILE {i+1}/{a.episodes} success={r["is_success"]} reason={r["failure_reason"]}',flush=True)
    metrics={name:sum(bool(r[name]) for r in results)/len(results) for name in ['main_recognized','parking_selected','parked','wrist_localized','grasp_started']}
    summary={'episodes':len(results),'successes':sum(r['is_success'] for r in results),
        'success_rate':sum(r['is_success'] for r in results)/len(results),'stage_success_rates':metrics,
        'failures':dict(Counter(r['failure_reason'] for r in results if r['failure_reason'])),'learned_vision':True,
        'perception_backend':'bundled_target_bowl_yolo11n_seg','vision_backend':'ultralytics_yolo_segmenter','vision_device':a.vision_device,'weights_sha256':digest(weights),
        'configuration':mobile,'mobile_config_sha256':digest(a.mobile_config),'provenance':provenance(cfg),
        'policy':str(a.policy) if a.policy else None,'policy_device':a.policy_device,'hardware_ready':False,'local_execution_claim':'To be run and verified on Mac; not validated by packaging host',
        'trials':[{k:v for k,v in r.items() if k not in ('configuration','provenance')} for r in results]}
    write_json(a.output/'mobile-evaluation.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('configuration','provenance','trials')},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
