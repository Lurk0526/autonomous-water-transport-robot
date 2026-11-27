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
from .control import synchronized_step,smoothstep

PHASES=['settle','approach','close','lift','hold','lower','support','release','retreat','settle_place']


def supported_from_evidence(evidence,weight,linear_speed,angular_speed,task,*,settled=True):
    """Use the lowest bowl surface, not the pose origin, to identify support.

    The unsettled form is a stop-on-touch guard. Stable phase transitions use
    the stricter velocity checks as well. This is simulator supervision; it is
    not an assertion that the real robot has a contact or force sensor.
    """
    contact=(evidence['support_n']>=weight*.5
        and abs(evidence['clearance_m'])<=task.get('support_clearance_tolerance_m',.003)
        and evidence['tilt_rad']<=task['tilt_limit_rad'])
    return bool(contact and (not settled or (
        linear_speed<=task.get('support_linear_speed_m_s',.02)
        and angular_speed<=task.get('support_angular_speed_rad_s',.1))))


def closure_anomaly(positions,ranges,*,limit_tolerance=.001,sync_tolerance=.003):
    """Return a separate fault for slide limits or a broken four-finger coupling."""
    positions=np.asarray(positions,dtype=float);ranges=np.asarray(ranges,dtype=float)
    if not np.isfinite(positions).all():return 'closure_nonfinite'
    if np.any(positions<ranges[:,0]-limit_tolerance) or np.any(positions>ranges[:,1]+limit_tolerance):
        return 'closure_joint_limit'
    if np.ptp(positions)>sync_tolerance:return 'closure_coupling_limit'
    return None


class BowlEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,cfg=None,*,visual=False):
        self.cfg=copy.deepcopy(cfg) if isinstance(cfg,dict) else config(cfg)
        self.model,self.xml,self.cad=make_model(self.cfg,visual=visual);self.data=mujoco.MjData(self.model)
        self.ik=IK(self.model);self.action_space=gym.spaces.Box(-1.,1.,shape=(3,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-5.,5.,shape=(45,),dtype=np.float32)
        self.qids=np.array([self.model.joint(j).qposadr[0] for j in JOINTS])
        self.dofs=np.array([self.model.joint(j).dofadr[0] for j in JOINTS])
        self.ranges=np.array([self.model.joint(j).range for j in JOINTS])
        self.closure_qids=np.array([self.model.joint(f'closure_{i}').qposadr[0] for i in range(4)])
        self.closure_dofs=np.array([self.model.joint(f'closure_{i}').dofadr[0] for i in range(4)])
        self.closure_ranges=np.array([self.model.joint(f'closure_{i}').range for i in range(4)])
        self.body=self.model.body('bowl').id;self.site=self.model.site('tcp').id
        self.free_q=int(self.model.joint('bowl_free').qposadr[0]);self.free_dof=int(self.model.joint('bowl_free').dofadr[0])
        self.platform=self.model.geom('platform').id;self.platform_mocap=int(self.model.body('platform_body').mocapid[0])
        self.bowl_geoms=set(np.where(self.model.geom_bodyid==self.body)[0])
        self.robot_bodies={self.model.body(j).id for j in JOINTS}
        self.finger_bodies=[self.model.body(f'finger_{i}').id for i in range(4)]
        self.foam_geoms=[i for i in range(self.model.ngeom) if (self.model.geom(i).name or '').endswith('_foam')]
        self.robot_geoms=set(np.where(np.isin(self.model.geom_bodyid,list(self.robot_bodies)+self.finger_bodies))[0])
        self.forbidden_contact=np.zeros((self.model.ngeom,self.model.ngeom),dtype=bool)
        for g1 in range(self.model.ngeom):
            for g2 in range(self.model.ngeom):
                b1,b2=int(self.model.geom_bodyid[g1]),int(self.model.geom_bodyid[g2])
                allowed=(b1 in self.finger_bodies and b2 in self.finger_bodies) or (g1 in self.bowl_geoms and b2 in self.finger_bodies) or (g2 in self.bowl_geoms and b1 in self.finger_bodies)
                self.forbidden_contact[g1,g2]=bool({g1,g2}&self.robot_geoms) and not allowed
        axes=np.array(self.cad['finger_closure_axes'])
        self.opposed=[(i,j) for i in range(4) for j in range(i+1,4) if np.dot(axes[i],axes[j])<-.5]
        self.dt=self.cfg['simulation']['control_dt_s'];self.nsub=round(self.dt/self.cfg['simulation']['dt_s'])
        self.physics_dt=float(self.cfg['simulation']['dt_s'])
        self.contact_mode=self.cfg['task'].get('contact_supervisor_mode','foam_required')
        if self.contact_mode not in ('foam_required','any_contact_diagnostic'):
            raise ValueError('Unknown contact supervisor mode')
        preload=float(self.cfg['gripper'].get('contact_preload_m',0.))
        if not 0<=preload<=.002:raise ValueError('Contact preload must be an explicit value between 0 and 2 mm')
        window=float(self.cfg['simulation'].get('diagnostic_window_s',.25))
        if not 0<window<=2:raise ValueError('Diagnostic window must be between zero and 2 seconds')
        self.microstep_window_size=max(1,math.ceil(window/self.physics_dt))
        self.reference_inertia=self.model.body_inertia[self.body].copy()
        self._force=np.zeros(6);self._done=True

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {};self.options=options
        mujoco.mj_resetData(self.model,self.data)
        self.reward_model=Reward(gamma=self.cfg['ppo']['gamma'],control_dt=self.dt)
        self.phase_index=0;self.phase_time=0.;self.elapsed=0.;self.reason=None
        self.previous_action=np.zeros(3);self.residual=np.zeros(3);self.hold_time=0.;self.support_time=0.;self.contact_lost_time=0.
        self.trial_verified=False;self.hold_verified=False;self._done=False;self.rows=[];self.frames=[]
        self.gate_time=0.;self.place_stable_time=0.;self.phase_summary=[];self.first_collision=None
        self.first_physics_anomaly=None;self.guard_events=[]
        self.microsteps=deque(maxlen=self.microstep_window_size)
        self.grip_hold_command=None;self.grip_stop_reason=None;self.grip_stop_time=None
        self.grip_contact_time=0.;self.grip_contact_confirmed=False;self.grip_hard_only_time=0.;self.grip_single_time=0.
        self.support_lock_q=None;self.support_lock_target=None
        self.phase_start_limit=0.;self.current_limit=0.;self.nominal=np.zeros(3)
        self.parking=np.asarray(options['parking'],dtype=float) if 'parking' in options else sample_parking(self.np_random,self.cfg['parking'])
        bowlworld=np.array([*self.cfg['fixture']['bowl_world_xy_m'],self.cfg['fixture']['top_height_m']])
        self.place=world_to_base(bowlworld,self.parking);self.top=float(self.place[2])
        platformworld=np.array([*self.cfg['fixture'].get('platform_world_xy_m',self.cfg['fixture']['bowl_world_xy_m']),self.top])
        self.platform_place=world_to_base(platformworld,self.parking)
        self.data.mocap_pos[self.platform_mocap]=[self.platform_place[0],self.platform_place[1],self.top/2]
        yaw=-self.parking[2];self.data.mocap_quat[self.platform_mocap]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.data.qpos[self.free_q:self.free_q+7]=[*self.place,1,0,0,0]
        self.estimate=self.place+self.np_random.normal(0,self.cfg['sensor']['initial_error_std_m'],3)
        if 'target_estimate_offset_m' in options:self.estimate+=np.asarray(options['target_estimate_offset_m'])
        if 'target_estimate_base_m' in options:
            value=np.asarray(options['target_estimate_base_m'],dtype=float)
            if value.shape!=(3,) or not np.isfinite(value).all():raise ValueError('External target must be a finite base-frame 3-vector')
            self.estimate=value.copy()
        self.observation_source=options.get('observation_source','simulated_noisy_pose')
        if self.observation_source not in ('simulated_noisy_pose','frozen_wrist_estimate'):raise ValueError('Unknown observation source')
        if self.observation_source=='frozen_wrist_estimate' and 'target_estimate_base_m' not in options:
            raise ValueError('Frozen wrist observations require an explicit measured target; no truth fallback')
        self.estimate_initial_age=float(options.get('target_estimate_age_s',0.))
        if not np.isfinite(self.estimate_initial_age) or self.estimate_initial_age<0:raise ValueError('Invalid external estimate age')
        grasp=self.estimate.copy();grasp[2]+=self.cfg['fixture']['grasp_height_above_table_m']
        home=grasp+[0,0,self.cfg['task']['approach_clearance_m']]
        carry=grasp+[0,0,self.cfg['task']['lift_m']]
        lower=grasp-[0,0,.0015]
        self.targets=[home,grasp,grasp,carry,carry,lower,lower,lower,home,home]
        qhome,ok=self.ik.solve(home)
        self.screened_home_q=None
        if 'screened_home_q' in options:
            candidate=np.asarray(options['screened_home_q'],dtype=float)
            if candidate.shape!=(5,) or not np.isfinite(candidate).all() or np.any((candidate<self.ranges[:,0])|(candidate>self.ranges[:,1])):raise ValueError('Invalid screened home joints')
            check=self.ik.data;check.qpos[self.qids]=candidate;mujoco.mj_forward(self.model,check)
            axis=check.site_xmat[self.site].reshape(3,3)[:,1]
            if np.linalg.norm(check.site_xpos[self.site]-home)>.005 or np.linalg.norm(axis-[0,0,1])>.08:raise ValueError('Screened joints do not reach the grasp home pose')
            self.screened_home_q=candidate.copy();qhome=candidate.copy();ok=True
        for target in [grasp,carry,lower]:
            _,possible=self.ik.solve(target,seed=qhome)
            ok=ok and possible
        self.reachable=bool(ok)
        self.command=qhome;self.desired_q=qhome.copy();self.data.qpos[self.qids]=qhome;self.data.ctrl[:5]=qhome
        self.commands=deque([(0.,qhome.copy())]);self.sensor_history=deque([(-self.estimate_initial_age,self.estimate.copy() if self.observation_source=='frozen_wrist_estimate' else self.place.copy())])
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
        self.data.mocap_pos[self.platform_mocap]=[self.platform_place[0],self.platform_place[1],self.top/2]
        self.data.mocap_quat[self.platform_mocap]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.data.qpos[self.free_q:self.free_q+7]=[*self.place,1,0,0,0]
        self.data.qpos[self.qids]=qhome;self.data.ctrl[:5]=qhome
        mujoco.mj_forward(self.model,self.data)
        self.start_target=home.copy();self.start_grip=0.;self.waypoint=home.copy();self.nominal=home.copy()
        self.previous_distance=float(np.linalg.norm(self.data.site_xpos[self.site]-self.nominal))
        info=self._info(False)
        return self._observation(),info

    @property
    def phase(self):return PHASES[self.phase_index]

    def _observation(self):
        if not self.sensor_history:
            if self.observation_source=='frozen_wrist_estimate':raise RuntimeError('Lost frozen visual measurement')
            observed=self.place.copy();age=0.
        else:
            observed=self.sensor_history[0][1];age=max(0.,self.elapsed-self.sensor_history[0][0])
        observed=observed+self.np_random.normal(0,self.cfg['sensor']['observation_noise_std_m'],3)
        ee=self.data.site_xpos[self.site];phase=np.eye(len(PHASES))[self.phase_index]
        q=self.data.qpos[self.qids]/np.pi;qd=self.data.qvel[self.dofs]/2.
        opening=self.data.qpos[int(self.model.joint('closure_0').qposadr[0])]/.055
        duration=self.cfg['task']['durations_s'][self.phase_index]
        obs=np.r_[q,qd,opening,(observed-ee)/.3,(self.estimate-ee)/.3,(self.nominal-ee)/.2,
                  self.previous_action,phase,not self.options.get('invalid_sensor',False),min(age,.5)/.5,
                  self.residual/self.cfg['simulation']['residual_limit_m'],
                  (self.command-self.data.qpos[self.qids])/np.pi,
                  min(self.phase_time/duration,2.),self.elapsed/self.cfg['task']['timeout_s']]
        return np.clip(obs,-5,5).astype(np.float32)

    def _evidence(self,*,details=True):
        force_by_finger=np.zeros(4);foam_forces=np.zeros(4);support=0.;collision=0.;contacts=[]
        max_penetration=0.;peak_contact=None;peak_gripper_contact=None
        for i in range(self.data.ncon):
            c=self.data.contact[i];g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(self.model.geom_bodyid[g1]),int(self.model.geom_bodyid[g2])
            mujoco.mj_contactForce(self.model,self.data,i,self._force);normal=max(float(self._force[0]),0.)
            max_penetration=max(max_penetration,max(0.,-float(c.dist)))
            if peak_contact is None or normal>peak_contact['force_n']:
                peak_contact={'geoms':[self.model.geom(g1).name,self.model.geom(g2).name],
                    'force_n':normal,'penetration_m':max(0.,-float(c.dist))}
            if details and normal>.1:contacts.append({'geoms':[self.model.geom(g1).name,self.model.geom(g2).name],
                'force_n':normal,'position_m':c.pos.tolist()})
            if g1 in self.bowl_geoms or g2 in self.bowl_geoms:
                other=g2 if g1 in self.bowl_geoms else g1;ob=int(self.model.geom_bodyid[other])
                if other==self.platform:
                    if abs(c.frame[2])>.9 and c.pos[2]>=self.top-.003:support+=normal*abs(float(c.frame[2]))
                elif ob in self.finger_bodies:
                    finger=self.finger_bodies.index(ob);force_by_finger[finger]+=normal
                    if other in self.foam_geoms:foam_forces[finger]+=normal
                    if peak_gripper_contact is None or normal>peak_gripper_contact['force_n']:
                        peak_gripper_contact={'geoms':[self.model.geom(g1).name,self.model.geom(g2).name],
                            'force_n':normal,'penetration_m':max(0.,-float(c.dist)),
                            'foam':other in self.foam_geoms}
                elif normal>.5:collision+=normal
            elif normal>.5:
                both_fingers=b1 in self.finger_bodies and b2 in self.finger_bodies
                robot=bool({b1,b2}&(self.robot_bodies|set(self.finger_bodies)))
                if robot and not both_fingers:collision+=normal
        matrix=self.data.xmat[self.body].reshape(3,3);radial=np.hypot(matrix[2,0],matrix[2,1])
        bottom=self.data.xpos[self.body,2]+min(z*matrix[2,2]-r*radial for r,z in self.cfg['bowl']['radial_profile_m'])
        tilt=float(np.arccos(np.clip(matrix[2,2],-1,1)))
        bilateral=any(min(force_by_finger[i],force_by_finger[j])>=self.cfg['task']['bilateral_force_n'] for i,j in self.opposed)
        foam_bilateral=any(min(foam_forces[i],foam_forces[j])>=self.cfg['task']['bilateral_force_n'] for i,j in self.opposed)
        return {'clearance_m':float(bottom-self.top),'tilt_rad':tilt,'finger_forces_n':force_by_finger.tolist(),
                'bilateral':bilateral,'foam_bilateral':foam_bilateral,'foam_forces_n':foam_forces.tolist(),
                'support_n':support,'collision_n':collision,'contacts':contacts,
                'max_penetration_m':max_penetration,'peak_contact':peak_contact,
                'peak_gripper_contact':peak_gripper_contact}

    def _collision_event(self):
        """Inspect each physics substep; keep the first disallowed contact, not just the last frame."""
        pairs=self.data.contact.geom[:self.data.ncon]
        candidates=np.flatnonzero(self.forbidden_contact[pairs[:,0],pairs[:,1]])
        for i in candidates:
            c=self.data.contact[i];g1,g2=int(c.geom1),int(c.geom2)
            mujoco.mj_contactForce(self.model,self.data,i,self._force)
            if self._force[0]>.5:
                return {'phase':self.phase,'time_s':float(self.data.time),
                    'geoms':[self.model.geom(g1).name,self.model.geom(g2).name],
                    'force_n':float(self._force[0]),'position_m':c.pos.tolist()}
        return None

    def _guard_event(self,kind,**values):
        event={'kind':kind,'phase':self.phase,'time_s':float(self.data.time),**values}
        self.guard_events.append(event)
        return event

    def _contact_accepted(self,evidence):
        foam_required=self.cfg['task']['require_foam_grasp'] or self.contact_mode=='foam_required'
        return bool(evidence['foam_bilateral'] if foam_required else evidence['bilateral'])

    def _pause_gripper(self,reason):
        if self.grip_hold_command is not None:return
        actual=float(np.min(self.data.qpos[self.closure_qids]))
        preload=float(self.cfg['gripper'].get('contact_preload_m',0.))
        # Never retain a large blocked position error as an implicit preload.
        self.grip_hold_command=float(np.clip(min(float(self.data.ctrl[5]),actual+preload),
            *self.closure_ranges[0]))
        self.grip_stop_reason=reason;self.grip_stop_time=float(self.data.time)
        self._guard_event('gripper_contact_stop',reason=reason,
            closure_m=self.data.qpos[self.closure_qids].tolist(),
            previous_command_m=float(self.data.ctrl[5]),hold_command_m=self.grip_hold_command)
        self.data.ctrl[5]=self.grip_hold_command

    def _stop_on_support(self,evidence):
        if self.support_lock_q is not None or self.phase!='lower':return
        if not supported_from_evidence(evidence,float(self.model.body_mass[self.body])*9.81,
                0.,0.,self.cfg['task'],settled=False):return
        # Cancel queued downward commands as well as the current Cartesian path.
        # A pending delayed target must not resume the descent on the next step.
        self.support_lock_q=self.data.qpos[self.qids].copy()
        self.support_lock_target=self.data.site_xpos[self.site].copy()
        self.command=self.support_lock_q.copy();self.desired_q=self.support_lock_q.copy()
        self.commands=deque([(float(self.data.time),self.support_lock_q.copy())])
        self.data.ctrl[:5]=self.support_lock_q
        self.residual[:]=0.;self.current_limit=0.;self.phase_start_limit=0.
        for index in (5,6,7):self.targets[index]=self.support_lock_target.copy()
        self.nominal=self.support_lock_target.copy();self.waypoint=self.nominal.copy()
        self._guard_event('support_stop_descent',support_n=float(evidence['support_n']),
            clearance_m=float(evidence['clearance_m']),q_hold=self.support_lock_q.tolist())

    def _contact_guard(self,evidence):
        """Simulator-only safety supervisor. Its force evidence is not an observation."""
        if self.phase not in ('close','lift','hold','lower','support') or self.options.get('no_grip'):
            return None
        accepted=self._contact_accepted(evidence)
        foam_required=self.cfg['task']['require_foam_grasp'] or self.contact_mode=='foam_required'
        forces=np.asarray(evidence['finger_forces_n']);foam=np.asarray(evidence['foam_forces_n'])
        threshold=float(self.cfg['task']['bilateral_force_n'])
        hard_contact=bool(np.any(np.maximum(0.,forces-foam)>=threshold))
        single_contact=bool(np.max(forces)>=self.cfg['gripper'].get('single_contact_stop_force_n',.5)
            and not evidence['bilateral'])
        if accepted:self._pause_gripper('opposed_contact')
        elif foam_required and hard_contact:self._pause_gripper('hard_contact_before_foam')
        elif single_contact:self._pause_gripper('single_contact_blocked')
        if self.grip_hold_command is None:return None
        self.data.ctrl[5]=self.grip_hold_command
        self.grip_contact_time=self.grip_contact_time+self.physics_dt if accepted else 0.
        self.grip_hard_only_time=self.grip_hard_only_time+self.physics_dt if hard_contact and not evidence['foam_bilateral'] else 0.
        self.grip_single_time=self.grip_single_time+self.physics_dt if single_contact else 0.
        if self.grip_contact_time+1e-10>=self.cfg['task'].get('contact_confirmation_s',.05):
            if not self.grip_contact_confirmed:self._guard_event('gripper_contact_confirmed')
            self.grip_contact_confirmed=True
        if self.grip_contact_confirmed:
            if foam_required and self.grip_hard_only_time>=self.cfg['task'].get('hard_contact_timeout_s',.15):
                return 'hard_contact_after_foam_loss'
            if self.grip_single_time>=self.cfg['task'].get('single_contact_timeout_s',.15):
                return 'gripper_single_contact_after_loss'
        waiting=float(self.data.time)-self.grip_stop_time
        if not self.grip_contact_confirmed:
            if self.grip_stop_reason=='hard_contact_before_foam':
                if waiting>=self.cfg['task'].get('hard_contact_timeout_s',.15):return 'hard_contact_before_foam'
            elif self.grip_stop_reason=='single_contact_blocked':
                if waiting>=self.cfg['task'].get('single_contact_timeout_s',.15):return 'gripper_single_contact_blocked'
            elif waiting>=self.cfg['task'].get('single_contact_timeout_s',.15):
                return 'gripper_contact_not_sustained'
        return None

    def _physics_guard(self):
        if np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return 'physics_warning_or_nonfinite'
        fault=closure_anomaly(self.data.qpos[self.closure_qids],self.closure_ranges,
            limit_tolerance=self.cfg['gripper'].get('closure_limit_tolerance_m',.001),
            sync_tolerance=self.cfg['gripper'].get('closure_sync_tolerance_m',.003))
        if fault:return fault
        actual=self.data.qpos[self.qids]
        if np.any(actual<self.ranges[:,0]-.02) or np.any(actual>self.ranges[:,1]+.02):return 'joint_limit'
        if np.max(np.abs(actual-self.desired_q))>self.cfg['task']['joint_tracking_limit_rad']:return 'joint_tracking_limit'
        return None

    def _microstep_snapshot(self,evidence):
        # Bounded ring, including ordinary steps before the first abnormal event.
        return {'time_s':float(self.data.time),'phase':self.phase,
            'q_actual':self.data.qpos[self.qids].tolist(),'q_desired':self.desired_q.tolist(),
            'q_velocity':self.data.qvel[self.dofs].tolist(),
            'closure_m':self.data.qpos[self.closure_qids].tolist(),
            'closure_velocity_m_s':self.data.qvel[self.closure_dofs].tolist(),
            'control':self.data.ctrl.tolist(),'actuator_force':self.data.actuator_force.tolist(),
            'maximum_acceleration':float(np.max(np.abs(self.data.qacc))),
            'finger_forces_n':evidence['finger_forces_n'],'foam_forces_n':evidence['foam_forces_n'],
            'support_n':evidence['support_n'],'clearance_m':evidence['clearance_m'],
            'tilt_rad':evidence['tilt_rad'],'max_penetration_m':evidence['max_penetration_m'],
            'peak_contact':evidence['peak_contact'],'peak_gripper_contact':evidence['peak_gripper_contact']}

    def _info(self,success):
        return {'is_success':bool(success),'failure_reason':self.reason,'phase':self.phase,
                'reachable':self.reachable,'parking_xy_yaw':self.parking.tolist(),
                'place_error_m':float(np.linalg.norm(self.data.xpos[self.body,:2]-self.place[:2])),
                'elapsed_s':self.elapsed,'external_target_age_s':self.elapsed+self.estimate_initial_age if self.observation_source=='frozen_wrist_estimate' else None,'visual_tracking_during_grasp':False,'observation_source':self.observation_source,'model_status':self.cfg['model_status'],'hardware_ready':False,'fluid_simulated':False,
                'first_collision':self.first_collision,'first_physics_anomaly':self.first_physics_anomaly,
                'guard_events':list(self.guard_events),'phase_summary':list(self.phase_summary),
                'contact_supervisor':{'mode':self.contact_mode,'uses_simulator_truth':True,
                    'real_contact_sensor_validated':False,'gripper_stop_reason':self.grip_stop_reason,
                    'gripper_contact_confirmed':self.grip_contact_confirmed,
                    'support_descent_locked':self.support_lock_q is not None},
                'microstep_failure_window':list(self.microsteps) if self.reason else [],
                'q_tracking_error_rad':float(np.max(np.abs(self.data.qpos[self.qids]-self.desired_q))),
                'q_target_error_rad':float(np.max(np.abs(self.data.qpos[self.qids]-self.command))),
                'hold_verified':self.hold_verified,'hold_streak_s':self.hold_time,
                'place_stable_s':self.place_stable_time,'scenario':{'delay_s':self.delay,
                'initial_target_error_m':(self.estimate-self.place).tolist(),
                'foam_friction':float(self.model.geom_friction[self.foam_geoms[0],0])}}

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
        u=min(1.,(self.phase_time+self.dt)/self.cfg['task']['durations_s'][self.phase_index]);s=smoothstep(u)
        nominal=self.start_target+(self.targets[self.phase_index]-self.start_target)*s
        self.nominal=nominal
        limit=self.cfg['simulation']['phase_residual_limits_m'][self.phase_index]
        self.current_limit=self.phase_start_limit+(limit-self.phase_start_limit)*s
        target_residual=action*self.current_limit
        max_change=self.cfg['simulation']['residual_rate_m_s']*self.dt
        self.residual+=np.clip(target_residual-self.residual,-max_change,max_change)
        self.waypoint=nominal+self.residual
        if self.support_lock_q is not None and self.phase in ('lower','support','release'):
            self.nominal=self.support_lock_target.copy();self.waypoint=self.nominal.copy()
            self.residual[:]=0.;self.current_limit=0.
            q=self.support_lock_q.copy();self.command=q.copy();self.desired_q=q.copy()
            self.commands=deque([(self.elapsed,q.copy())])
        else:
            if self.phase=='settle' and self.screened_home_q is not None:q,possible=self.screened_home_q,True
            else:q,possible=self.ik.solve(self.waypoint,seed=self.command,iterations=35)
            if not possible and failure is None:failure='waypoint_unreachable'
            max_q=self.cfg['arm']['max_joint_velocity_rad_s']*self.dt
            self.command=synchronized_step(self.command,q,max_q)
            self.commands.append((self.elapsed,self.command.copy()))
            while len(self.commands)>1 and self.commands[1][0]<=self.elapsed-self.delay:self.commands.popleft()
            self.desired_q=self.commands[0][1]
        closing=self.phase_index in [2,3,4,5,6]
        target_grip=self.cfg['gripper']['close_m'] if closing else 0.
        if self.options.get('no_grip'):target_grip=0.
        grip=self.start_grip+(target_grip-self.start_grip)*s
        if closing and self.grip_hold_command is not None:grip=self.grip_hold_command
        self.data.ctrl[:5]=self.desired_q;self.data.ctrl[5]=grip
        elapsed_step=0.
        if failure is None:
            for _ in range(self.nsub):
                mujoco.mj_step(self.model,self.data)
                elapsed_step+=self.physics_dt
                sub_evidence=self._evidence(details=False)
                self.microsteps.append(self._microstep_snapshot(sub_evidence))
                anomaly=self._physics_guard()
                event=self._collision_event()
                if event:
                    if self.first_collision is None:self.first_collision=event
                    if anomaly is None:anomaly='unexpected_collision'
                if anomaly is None:
                    self._stop_on_support(sub_evidence)
                    if self.support_lock_q is not None and self.phase in ('lower','support','release'):
                        q=self.support_lock_q.copy()
                    anomaly=self._contact_guard(sub_evidence)
                if anomaly:
                    if self.first_physics_anomaly is None:
                        self.first_physics_anomaly=self._guard_event(anomaly,
                            closure_m=self.data.qpos[self.closure_qids].tolist(),
                            q_tracking_error_rad=float(np.max(np.abs(self.data.qpos[self.qids]-self.desired_q))))
                    failure=anomaly
                    # Preserve the offending target for diagnostic tracking error.
                    stop_q=self.data.qpos[self.qids].copy()
                    self.commands=deque([(float(self.data.time),stop_q.copy())])
                    self.data.ctrl[:5]=stop_q
                    self.data.ctrl[5]=float(np.clip(self.data.qpos[self.closure_qids[0]],*self.closure_ranges[0]))
                    break
            mujoco.mj_forward(self.model,self.data)
        self.elapsed+=elapsed_step;self.phase_time+=elapsed_step
        evidence=self._evidence();weight=float(self.model.body_mass[self.body])*9.81
        if failure is None and (np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all()):failure='physics_warning_or_nonfinite'
        if evidence['collision_n']>.5 and failure is None:failure='unexpected_collision'
        qactual=self.data.qpos[self.qids]
        if failure is None and (np.any(qactual<self.ranges[:,0]-.02) or np.any(qactual>self.ranges[:,1]+.02)):failure='joint_limit'
        if failure is None and np.max(np.abs(qactual-self.desired_q))>self.cfg['task']['joint_tracking_limit_rad']:failure='joint_tracking_limit'
        if failure is None and self.data.xpos[self.body,2]<self.top-.03:failure='dropped_bowl'
        contact=self._contact_accepted(evidence)
        if self.phase in ['lift','hold'] and contact and evidence['clearance_m']>.01:self.trial_verified=True
        holding=contact and evidence['clearance_m']>=self.cfg['task']['minimum_hold_clearance_m'] and evidence['tilt_rad']<=self.cfg['task']['tilt_limit_rad']
        if self.phase=='hold':
            self.hold_time=self.hold_time+elapsed_step if holding else 0.
            if self.hold_time>=self.cfg['task']['hold_s']:self.hold_verified=True
        supported=supported_from_evidence(evidence,weight,
            float(np.linalg.norm(self.data.qvel[self.free_dof:self.free_dof+3])),
            float(np.linalg.norm(self.data.qvel[self.free_dof+3:self.free_dof+6])),self.cfg['task'])
        if self.phase=='support':self.support_time=self.support_time+elapsed_step if supported else 0.
        tracking=np.max(np.abs(qactual-q))<=self.cfg['task']['gate_tracking_rad']
        stopped=np.max(np.abs(self.data.qvel[self.dofs]))<=self.cfg['task']['gate_joint_speed_rad_s']
        ready=bool(tracking and stopped)
        if self.phase=='close':ready=ready and contact and self.grip_contact_confirmed
        if self.phase=='lift':ready=ready and holding
        if self.phase=='hold':ready=ready and self.hold_verified and holding
        if self.phase in ['lower','support']:ready=ready and supported
        if self.phase=='support':ready=ready and self.support_time>=self.cfg['task']['support_stable_s']
        if self.phase=='release':ready=ready and supported and max(evidence['finger_forces_n'])<.1
        stable=bool(self.hold_verified and supported and evidence['support_n']>=weight*.8 and max(evidence['finger_forces_n'])<.1
            and np.linalg.norm(self.data.xpos[self.body,:2]-self.place[:2])<=self.cfg['task']['place_tolerance_m']
            and np.linalg.norm(self.data.qvel[self.free_dof:self.free_dof+3])<.02
            and np.linalg.norm(self.data.qvel[self.free_dof+3:self.free_dof+6])<.1
            and evidence['tilt_rad']<=self.cfg['task']['tilt_limit_rad'])
        if self.phase=='settle_place':
            self.place_stable_time=self.place_stable_time+elapsed_step if stable else 0.
            ready=ready and self.place_stable_time+1e-8>=self.cfg['task']['place_stable_s']
        duration=self.cfg['task']['durations_s'][self.phase_index]
        motion_finished=self.phase_time+1e-8>=duration
        self.gate_time=self.gate_time+elapsed_step if ready and motion_finished else 0.
        ending=self.gate_time+1e-8>=self.cfg['task']['gate_stable_s']
        if failure is None and motion_finished and not ending and self.phase_time>=duration+self.cfg['task']['gate_wait_s']:
            failure={'close':'grasp_contact_missing','lift':'lift_clearance_not_verified','hold':'lift_hold_not_verified',
                'lower':'support_missing_after_lower','support':'support_missing_release_inhibited',
                'release':'release_not_verified','settle_place':'final_placement_not_verified'}.get(self.phase,'motion_not_settled')
            if not tracking or not stopped:failure='motion_not_settled'
        if ending and failure is None and self.phase=='settle_place':success=True
        if failure is None and self.elapsed>=self.cfg['task']['timeout_s'] and not success:failure='task_timeout'
        self.reason=failure
        # The action cannot move its own reward target. Use the nominal task reference.
        distance=float(np.linalg.norm(self.data.site_xpos[self.site]-self.nominal))
        reward,parts=self.reward_model.step(dt=self.dt,distance_before=self.previous_distance,distance_after=distance,
            tilt=evidence['tilt_rad'],residual=action,previous_residual=self.previous_action,
            trial_lift=self.trial_verified,hold_verified=self.hold_verified,complete=success,failure=failure is not None)
        self.previous_distance=distance;self.previous_action=action.copy()
        if self.observation_source=='simulated_noisy_pose':self.sensor_history.append((self.elapsed,self.data.xpos[self.body].copy()))
        while len(self.sensor_history)>1 and self.sensor_history[1][0]<=self.elapsed-self.cfg['sensor']['latency_s']:self.sensor_history.popleft()
        if ending or failure:self.phase_summary.append({'phase':self.phase,'elapsed_s':self.phase_time,'ready':bool(ready),'tracking':bool(tracking),'stopped':bool(stopped),'contact':bool(contact),'supported':bool(supported)})
        info={**self._info(success),**evidence,'reward_parts':parts}
        if self.options.get('trace'):
            self.rows.append({**info,'nominal_m':self.nominal.tolist(),'residual_m':self.residual.tolist(),
                'q_desired':self.desired_q.tolist(),'q_actual':qactual.tolist()})
            self.frames.append({'t':float(self.data.time),'phase':self.phase,'qpos':self.data.qpos.tolist(),'qvel':self.data.qvel.tolist(),'ctrl':self.data.ctrl.tolist()})
        if failure or success:self._done=True
        elif ending:
            self.start_target=self.targets[self.phase_index].copy();self.start_grip=float(self.data.ctrl[5])
            self.phase_start_limit=self.current_limit;self.phase_index+=1;self.phase_time=0.;self.gate_time=0.
        return self._observation(),float(reward),self._done,False,info
