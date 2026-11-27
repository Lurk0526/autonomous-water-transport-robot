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
from .perception import Segmenter,estimate_target,infer_tabletop,infer_wrist
from .env import BowlEnv


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def camera_observation(scene,segmenter,source,stamp_s,cfg,arm_cfg,folder,serial,coarse_hint=None):
    rgb,depth,k,t=scene.render(source,with_depth=source=='main_rgbd')
    proposal=None;settings=cfg['vision_model'].get('tabletop_crops')
    if source=='main_rgbd' and settings:
        detections,proposal=infer_tabletop(segmenter,rgb,depth,k,t,scene.table['top_height_m'],arm_cfg['bowl']['height_m'],settings)
    elif source=='wrist' and coarse_hint is not None and cfg['vision_model'].get('wrist_search_padding'):
        detections,proposal=infer_wrist(segmenter,rgb,k,t,coarse_hint,arm_cfg['bowl'],cfg['vision_model']['wrist_search_padding'])
    else:detections=segmenter.infer(rgb)
    targets=[];candidate_info=[]
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
        'inference_s':proposal['inference_s'] if proposal else segmenter.last_inference_s,'image_proposals':proposal,'detections':candidate_info,'selected_target':target,
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
        actual_q=scene.navigation_q.copy();scene.set_navigation_joints(actual_q)
        wrist_q=scene.observation_q.copy();aim_initialized=False
        def aim_at(position,pose):
            wrist=cfg['render_cameras']['wrist']
            if 'observation_xy_offset_m' not in wrist:return scene.observation_q.copy(),True
            relative=point_in_model(position,compose_pose(pose,cfg['arm']['mount_xy_yaw_m_rad']))
            return screen.ik.solve(relative+np.r_[wrist['observation_xy_offset_m'],wrist['observation_tcp_above_plane_m']],seed=scene.observation_q)
        def solve_arm(point,seed=None):
            solution,ok=screen.ik.solve(point,seed=wrist_q if seed is None else seed)
            if not ok and seed is None:return screen.ik.solve(point)
            return solution,ok
        def transition_clear(joint_path,pose,start=None):
            if not joint_path:return False
            end=np.asarray(joint_path[0]['q_rad']);start=wrist_q if start is None else start
            for u in np.linspace(0.,1.,max(2,int(np.ceil(np.max(np.abs(end-start))/.05))+1)):
                if not navigation_arm_clear(screen,start+(end-start)*u,pose,table,cfg)['valid']:return False
            return True
        while t<cfg['runtime']['maximum_approach_s']:
            source='wrist' if flow.state in ('acquire_wrist','grasp_ready') else 'main_rgbd'
            if source=='wrist' and not aim_initialized:
                wrist_q,ok=aim_at(main_target['position_m'],state.pose)
                if not ok or not transition_clear([{'q_rad':wrist_q}],state.pose,actual_q):
                    result['failure_reason']='stow_to_wrist_pose_unreachable_or_blocked';break
                aim_initialized=True
            wrist_ready=np.array_equal(actual_q,wrist_q)
            if source=='wrist' and not wrist_ready and state.speed_m_s==0. and state.yaw_rate_rad_s==0.:
                delta=wrist_q-actual_q;step=arm_cfg['arm']['max_joint_velocity_rad_s']*dt
                new_q=actual_q+delta*min(1.,step/max(1e-12,np.max(np.abs(delta))))
                if np.max(np.abs(delta))<=step:new_q=wrist_q.copy()
                if not navigation_arm_clear(screen,new_q,state.pose,table,cfg)['valid']:
                    result['failure_reason']='stow_to_wrist_pose_collision';break
                actual_q=new_q;scene.set_navigation_joints(actual_q);wrist_ready=np.array_equal(actual_q,wrist_q)
            if t+1e-9>=next_vision and (source!='wrist' or wrist_ready):
                hint=main_target['position_m'] if source=='wrist' and main_target else None
                target,record=camera_observation(scene,segmenter,source,t,cfg,arm_cfg,output/'vision',serial,coarse_hint=hint)
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
                            check=navigation_arm_clear(screen,scene.navigation_q,[p['x_m'],p['y_m'],p['yaw_rad']],table,cfg)
                            if not check['valid']:bad=check;break
                        aim,ok=aim_at(main_target['position_m'],candidate['parking_xy_yaw'])
                        if bad is None and (not ok or not transition_clear([{'q_rad':aim}],candidate['parking_xy_yaw'],scene.navigation_q)):bad={'valid':False,'reason':'stow_to_wrist_pose_collision'}
                        if bad is None and not transition_clear(candidate['joint_path'],candidate['parking_xy_yaw'],aim):bad={'valid':False,'reason':'observation_to_pregrasp_transition_collision'}
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
                    result['parked']=True
                    if wrist_ready and wrist_stage_started is None:wrist_stage_started=t
                    if wrist_target and wrist_target.get('valid'):
                        wrist_check=screen_grasp_at_parking(wrist_target,table,state.pose,cfg,solve_arm,
                            geometry_check=screen.geometry_check,camera_pose_fn=scene.wrist_camera_pose_in_model,now_s=t)
                        if not wrist_check['valid'] and wrist_check['reason'].startswith(('arm_ik_unreachable','joint_branch_discontinuity','joint_margin_insufficient')):
                            alternate=screen_grasp_at_parking(wrist_target,table,state.pose,cfg,screen.ik.solve,
                                geometry_check=screen.geometry_check,camera_pose_fn=scene.wrist_camera_pose_in_model,now_s=t)
                            if alternate['valid']:wrist_check=alternate
                        write_json(output/'wrist-grasp-screen.json',wrist_check);precise=bool(wrist_check['valid'] and transition_clear(wrist_check.get('arm',{}).get('joint_path',[]),state.pose) and state.speed_m_s==0. and state.yaw_rate_rad_s==0.)
                    if wrist_stage_started is not None and t-wrist_stage_started>cfg['runtime']['wrist_acquisition_timeout_s']:
                        result['failure_reason']='wrist_vision_or_precise_grasp_screen_failed';break
                clear,reason=footprint_clear(state.pose,table,cfg['vehicle'],cfg['planning']['obstacles'])
                arm_clear=navigation_arm_clear(screen,actual_q,state.pose,table,cfg)
                intent=flow.update(t,state.odometry(t),main_target=main_target,wrist_target=wrist_target,
                    obstacle_clear=clear,arm_stowed=arm_clear['valid'],precise_grasp_feasible=precise)
            rows.append({'t_s':t,'vehicle':state.record(),'intent':intent,'main_target':main_target,'wrist_target':wrist_target})
            frames.append(scene.snapshot(t,flow.state))
            if flow.state=='stop':result['failure_reason']=flow.reason;break
            if intent['grasp_ready']:
                result['parked']=True;result['grasp_started']=True
                actual_arm=compose_pose(state.pose,cfg['arm']['mount_xy_yaw_m_rad'])
                local_target=point_in_model(wrist_target['position_m'],actual_arm)
                pick_cfg=copy.deepcopy(arm_cfg);pick_cfg['fixture']['bowl_world_xy_m']=target_truth[:2].tolist()
                pick_cfg['fixture']['platform_world_xy_m']=table['center_xy_m'];pick_cfg['fixture']['half_size_m']=table['half_size_m'];pick_cfg['sensor']['observation_noise_std_m']=0.
                screened_home=wrist_check['arm']['joint_path'][0]['q_rad']
                transit=float(np.max(np.abs(np.asarray(screened_home)-actual_q)))/pick_cfg['arm']['max_joint_velocity_rad_s']
                pick_cfg['task']['durations_s'][0]+=transit;result['screened_pregrasp_transit_s']=transit
                arm=BowlEnv(pick_cfg);obs,info=arm.reset(seed=seed,options={'parking':actual_arm,
                    'target_estimate_base_m':local_target,'target_estimate_age_s':max(0.,t-wrist_target['stamp_s']),'observation_source':'frozen_wrist_estimate','trace':True,
                    'screened_home_q':screened_home})
                # Preserve the observation arm state at handoff. Move to pre-grasp in place
                # using the existing coordinated, bounded controller; do not teleport q.
                arm.data.qpos[arm.qids]=actual_q;arm.command=actual_q.copy();arm.desired_q=actual_q.copy()
                arm.data.ctrl[:5]=actual_q;arm.commands.clear();arm.commands.append((0.,actual_q.copy()))
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
                check=navigation_arm_clear(screen,actual_q,pose,table,cfg)
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
    a=p.parse_args();cfg=config(a.config);mobile=yaml.safe_load(a.mobile_config.read_text())
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
