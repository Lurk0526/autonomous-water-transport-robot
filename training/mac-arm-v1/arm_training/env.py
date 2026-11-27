"""Target-conditioned pick/hold/place task with parking changes and actual contact evidence."""
from collections import deque
import copy
import math

import gymnasium as gym
import mujoco
import numpy as np

from .model import JOINTS,IK,config,make_model
from .parking import sample_parking,world_to_base
from .rewards import Reward

PHASES=['settle','approach','close','lift','hold','lower','support','release','retreat','settle_place']


class BowlEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,cfg=None,*,visual=False):
        self.cfg=copy.deepcopy(cfg) if isinstance(cfg,dict) else config(cfg)
        self.model,self.xml,self.cad=make_model(self.cfg,visual=visual);self.data=mujoco.MjData(self.model)
        self.ik=IK(self.model);self.action_space=gym.spaces.Box(-1.,1.,shape=(3,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-5.,5.,shape=(35,),dtype=np.float32)
        self.qids=np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.dofs=np.array([self.model.joint(j).dofadr[0] for j in JOINTS])
        self.ranges=np.array([self.model.joint(j).range for j in JOINTS])
        self.body=self.model.body('bowl').id;self.site=self.model.site('tcp').id
        self.free_q=int(self.model.joint('bowl_free').qposadr[0]);self.free_dof=int(self.model.joint('bowl_free').dofadr[0])
        self.platform=self.model.geom('platform').id;self.platform_mocap=int(self.model.body('platform_body').mocapid[0])
        self.bowl_geoms=set(np.where(self.model.geom_bodyid==self.body)[0])
        self.robot_bodies={self.model.body(j).id for j in JOINTS}
        self.finger_bodies=[self.model.body(f'finger_{i}').id for i in range(4)]
        self.foam_geoms=[i for i in range(self.model.ngeom) if (self.model.geom(i).name or '').endswith('_foam')]
        axes=np.array(self.cad['finger_closure_axes'])
        self.opposed=[(i,j) for i in range(4) for j in range(i+1,4) if np.dot(axes[i],axes[j])<-.5]
        self.dt=self.cfg['simulation']['control_dt_s'];self.nsub=round(self.dt/self.cfg['simulation']['dt_s'])
        self.reference_inertia=self.model.body_inertia[self.body].copy()
        self._force=np.zeros(6);self._done=True

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {};self.options=options
        mujoco.mj_resetData(self.model,self.data)
        self.reward_model=Reward();self.phase_index=0;self.phase_time=0.;self.elapsed=0.;self.reason=None
        self.previous_action=np.zeros(3);self.residual=np.zeros(3);self.hold_time=0.;self.support_time=0.;self.contact_lost_time=0.
        self.trial_verified=False;self.hold_verified=False;self._done=False;self.rows=[];self.frames=[]
        self.parking=np.asarray(options['parking'],dtype=float) if 'parking' in options else sample_parking(self.np_random,self.cfg['parking'])
        bowlworld=np.array([*self.cfg['fixture']['bowl_world_xy_m'],self.cfg['fixture']['top_height_m']])
        self.place=world_to_base(bowlworld,self.parking);self.top=float(self.place[2])
        self.data.mocap_pos[self.platform_mocap]=[self.place[0],self.place[1],self.top/2]
        yaw=-self.parking[2];self.data.mocap_quat[self.platform_mocap]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.data.qpos[self.free_q:self.free_q+7]=[*self.place,1,0,0,0]
        self.estimate=self.place+self.np_random.normal(0,self.cfg['sensor']['initial_error_std_m'],3)
        if 'target_estimate_offset_m' in options:self.estimate+=np.asarray(options['target_estimate_offset_m'])
        grasp=self.estimate.copy();grasp[2]+=self.cfg['fixture']['grasp_height_above_table_m']
        home=grasp+[0,0,self.cfg['task']['approach_clearance_m']]
        carry=grasp+[0,0,self.cfg['task']['lift_m']]
        lower=grasp-[0,0,.0015]
        self.targets=[home,grasp,grasp,carry,carry,lower,lower,lower,home,home]
        qhome,ok=self.ik.solve(home)
        for target in [grasp,carry,lower]:
            _,possible=self.ik.solve(target,seed=qhome)
            ok=ok and possible
        self.reachable=bool(ok)
        self.command=qhome;self.desired_q=qhome.copy();self.data.qpos[self.qids]=qhome;self.data.ctrl[:5]=qhome
        self.commands=deque([(0.,qhome.copy())]);self.sensor_history=deque([(0.,self.place.copy())])
        self.delay=float(self.np_random.uniform(*self.cfg['arm']['actuator_delay_s']))
        friction=float(options.get('friction',self.np_random.uniform(*self.cfg['gripper']['foam_friction'])))
        for i in self.foam_geoms:self.model.geom_friction[i,0]=friction
        mass=float(options.get('mass_kg',self.cfg['bowl']['total_mass_kg']))
        self.model.body_mass[self.body]=mass;self.model.body_inertia[self.body]=self.reference_inertia*(mass/self.cfg['bowl']['total_mass_kg'])
        if 'force_n' in options:
            force=max(float(options['force_n']),1e-6);self.model.actuator_forcerange[5]=[-force,force]
        else:self.model.actuator_forcerange[5]=[-self.cfg['gripper']['force_limit_n'],self.cfg['gripper']['force_limit_n']]
        mujoco.mj_setConst(self.model,self.data)
        # mj_setConst restores qpos0; set the episode state only after constants are updated.
        self.data.mocap_pos[self.platform_mocap]=[self.place[0],self.place[1],self.top/2]
        self.data.mocap_quat[self.platform_mocap]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.data.qpos[self.free_q:self.free_q+7]=[*self.place,1,0,0,0]
        self.data.qpos[self.qids]=qhome;self.data.ctrl[:5]=qhome
        mujoco.mj_forward(self.model,self.data)
        self.start_target=home.copy();self.start_grip=0.;self.waypoint=home.copy();self.previous_distance=float(np.linalg.norm(self.data.site_xpos[self.site]-self.waypoint))
        info=self._info(False)
        return self._observation(),info

    @property
    def phase(self):return PHASES[self.phase_index]

    def _observation(self):
        if not self.sensor_history:observed=self.place.copy();age=0.
        else:
            observed=self.sensor_history[0][1];age=max(0.,self.elapsed-self.sensor_history[0][0])
        observed=observed+self.np_random.normal(0,self.cfg['sensor']['observation_noise_std_m'],3)
        ee=self.data.site_xpos[self.site];phase=np.eye(len(PHASES))[self.phase_index]
        q=self.data.qpos[self.qids]/np.pi;qd=self.data.qvel[self.dofs]/2.
        opening=self.data.qpos[int(self.model.joint('closure_0').qposadr[0])]/.055
        obs=np.r_[q,qd,opening,(observed-ee)/.3,(self.estimate-ee)/.3,(self.waypoint-ee)/.2,
                  self.previous_action,phase,not self.options.get('invalid_sensor',False),min(age,.5)/.5]
        return np.clip(obs,-5,5).astype(np.float32)

    def _evidence(self):
        force_by_finger=np.zeros(4);support=0.;collision=0.
        for i in range(self.data.ncon):
            c=self.data.contact[i];g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(self.model.geom_bodyid[g1]),int(self.model.geom_bodyid[g2])
            mujoco.mj_contactForce(self.model,self.data,i,self._force);normal=max(float(self._force[0]),0.)
            if g1 in self.bowl_geoms or g2 in self.bowl_geoms:
                other=g2 if g1 in self.bowl_geoms else g1;ob=int(self.model.geom_bodyid[other])
                if other==self.platform:
                    if abs(c.frame[2])>.9 and c.pos[2]>=self.top-.003:support+=normal*abs(float(c.frame[2]))
                elif ob in self.finger_bodies:force_by_finger[self.finger_bodies.index(ob)]+=normal
                elif normal>.5:collision+=normal
            elif normal>.5:
                both_fingers=b1 in self.finger_bodies and b2 in self.finger_bodies
                robot=bool({b1,b2}&(self.robot_bodies|set(self.finger_bodies)))
                if robot and not both_fingers:collision+=normal
        matrix=self.data.xmat[self.body].reshape(3,3);radial=np.hypot(matrix[2,0],matrix[2,1])
        bottom=self.data.xpos[self.body,2]+min(z*matrix[2,2]-r*radial for r,z in self.cfg['bowl']['radial_profile_m'])
        tilt=float(np.arccos(np.clip(matrix[2,2],-1,1)))
        bilateral=any(min(force_by_finger[i],force_by_finger[j])>=self.cfg['task']['bilateral_force_n'] for i,j in self.opposed)
        return {'clearance_m':float(bottom-self.top),'tilt_rad':tilt,'finger_forces_n':force_by_finger.tolist(),
                'bilateral':bilateral,'support_n':support,'collision_n':collision}

    def _info(self,success):
        return {'is_success':bool(success),'failure_reason':self.reason,'phase':self.phase,
                'reachable':self.reachable,'parking_xy_yaw':self.parking.tolist(),
                'place_error_m':float(np.linalg.norm(self.data.xpos[self.body,:2]-self.place[:2])),
                'elapsed_s':self.elapsed,'model_status':self.cfg['model_status'],'hardware_ready':False,'fluid_simulated':False}

    def step(self,action):
        if self._done:raise RuntimeError('Call reset after termination')
        action=np.asarray(action,dtype=float)
        if action.shape!=(3,) or not np.isfinite(action).all():raise ValueError('Expected three finite residual actions')
        action=np.clip(action,-1,1)
        failure=None;success=False
        if not self.reachable:failure='unreachable_reposition_required'
        if self.options.get('invalid_sensor'):failure='invalid_target_observation'
        if failure:
            self.reason=failure;self._done=True;self.elapsed+=self.dt
            reward,parts=self.reward_model.step(dt=self.dt,distance_before=0,distance_after=0,tilt=0,
                residual=action,previous_residual=self.previous_action,failure=True)
            return self._observation(),reward,True,False,{**self._info(False),**self._evidence(),'reward_parts':parts}
        u=min(1.,(self.phase_time+self.dt)/self.cfg['task']['durations_s'][self.phase_index]);s=u**3*(10+u*(-15+6*u))
        nominal=self.start_target+(self.targets[self.phase_index]-self.start_target)*s
        target_residual=action*self.cfg['simulation']['residual_limit_m']
        max_change=self.cfg['simulation']['residual_rate_m_s']*self.dt
        self.residual+=np.clip(target_residual-self.residual,-max_change,max_change)
        self.waypoint=nominal+self.residual
        q,possible=self.ik.solve(self.waypoint,seed=self.command,iterations=35)
        if not possible and failure is None:failure='waypoint_unreachable'
        max_q=self.cfg['arm']['max_joint_velocity_rad_s']*self.dt
        self.command+=np.clip(q-self.command,-max_q,max_q)
        self.commands.append((self.elapsed,self.command.copy()))
        while len(self.commands)>1 and self.commands[1][0]<=self.elapsed-self.delay:self.commands.popleft()
        self.desired_q=self.commands[0][1]
        closing=self.phase_index in [2,3,4,5,6]
        target_grip=self.cfg['gripper']['close_m'] if closing else 0.
        if self.options.get('no_grip'):target_grip=0.
        grip=self.start_grip+(target_grip-self.start_grip)*s
        self.data.ctrl[:5]=self.desired_q;self.data.ctrl[5]=grip
        if failure is None:
            for _ in range(self.nsub):mujoco.mj_step(self.model,self.data)
            mujoco.mj_forward(self.model,self.data)
        self.elapsed+=self.dt;self.phase_time+=self.dt
        evidence=self._evidence();weight=float(self.model.body_mass[self.body])*9.81
        if np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all():failure='physics_warning_or_nonfinite'
        if evidence['collision_n']>.5 and failure is None:failure='unexpected_collision'
        qactual=self.data.qpos[self.qids]
        if np.any(qactual<self.ranges[:,0]-.02) or np.any(qactual>self.ranges[:,1]+.02):failure='joint_limit'
        if np.max(np.abs(qactual-self.desired_q))>self.cfg['task']['joint_tracking_limit_rad']:failure='joint_tracking_limit'
        if self.data.xpos[self.body,2]<self.top-.03:failure='dropped_bowl'
        if self.phase in ['lift','hold'] and evidence['bilateral'] and evidence['clearance_m']>.01:self.trial_verified=True
        holding=evidence['bilateral'] and evidence['clearance_m']>=self.cfg['task']['minimum_hold_clearance_m'] and evidence['tilt_rad']<=self.cfg['task']['tilt_limit_rad']
        if self.phase=='hold':
            self.hold_time=self.hold_time+self.dt if holding else 0.
            if self.hold_time>=self.cfg['task']['hold_s']:self.hold_verified=True
        supported=evidence['support_n']>=weight*.5 and abs(self.data.xpos[self.body,2]-self.top)<.005
        if self.phase=='support':self.support_time=self.support_time+self.dt if supported else 0.
        ending=self.phase_time+1e-8>=self.cfg['task']['durations_s'][self.phase_index]
        if ending and failure is None:
            if self.phase=='close' and not evidence['bilateral']:failure='grasp_contact_missing'
            elif self.phase=='hold' and not self.hold_verified:failure='lift_hold_not_verified'
            elif self.phase=='support' and self.support_time<.2:failure='support_missing_release_inhibited'
            elif self.phase=='settle_place':
                stable=np.linalg.norm(self.data.qvel[self.free_dof:self.free_dof+3])<.02
                success=bool(self.hold_verified and supported and evidence['support_n']>=weight*.8 and max(evidence['finger_forces_n'])<.1 and self._info(False)['place_error_m']<=self.cfg['task']['place_tolerance_m'] and stable)
                if not success:failure='final_placement_not_verified'
        if self.elapsed>=self.cfg['task']['timeout_s'] and not success:failure='task_timeout'
        self.reason=failure
        distance=float(np.linalg.norm(self.data.site_xpos[self.site]-self.waypoint))
        reward,parts=self.reward_model.step(dt=self.dt,distance_before=self.previous_distance,distance_after=distance,
            tilt=evidence['tilt_rad'],residual=action,previous_residual=self.previous_action,
            trial_lift=self.trial_verified,hold_verified=self.hold_verified,complete=success,failure=failure is not None)
        self.previous_distance=distance;self.previous_action=action.copy()
        self.sensor_history.append((self.elapsed,self.data.xpos[self.body].copy()))
        while len(self.sensor_history)>1 and self.sensor_history[1][0]<=self.elapsed-self.cfg['sensor']['latency_s']:self.sensor_history.popleft()
        info={**self._info(success),**evidence,'reward_parts':parts}
        self.rows.append({k:v for k,v in info.items() if not isinstance(v,(dict,list))})
        self.frames.append({'t':float(self.data.time),'phase':self.phase,'qpos':self.data.qpos.tolist(),'qvel':self.data.qvel.tolist(),'ctrl':self.data.ctrl.tolist()})
        if failure or success:self._done=True
        elif ending:
            self.start_target=self.targets[self.phase_index].copy();self.start_grip=target_grip;self.phase_index+=1;self.phase_time=0.
        return self._observation(),float(reward),self._done,False,info
