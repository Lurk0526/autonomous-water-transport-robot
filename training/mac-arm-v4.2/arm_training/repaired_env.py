"""Physical one-rack contact environment, preserving the pick/hold/place criteria."""
import copy
from collections import deque
import mujoco
import numpy as np
from .env import BowlEnv
from .control import synchronized_step
from .compliant_control import SynchronousGraspController
from .linked_gripper import loop_residuals
from .precision_ik import PrecisionIK


class RepairedBowlEnv(BowlEnv):
    def reset(self,*,seed=None,options=None):
        self.servo_integral=np.zeros(5);self.contact_control=SynchronousGraspController(self.cfg)
        self.last_contact_command=None;self.maximum_loop_error_m=0.;self.lower_search_m=0.;self.lower_search_reference=None;self.support_unload_command=None
        self.target_measurements=deque(maxlen=24);self.target_refinement=None
        self.residual_projection_count = 0
        self.last_residual_projection = None
        self.actuator_target = None
        self.ramp_start = None
        self.ramp_goal = None
        self.ramp_elapsed = 0.
        if self.cfg.get('position_servo',{}).get('retain_tool_orientation',False):self.ik=PrecisionIK(self.model)
        obs,info=super().reset(seed=seed,options=options)
        if isinstance(self.ik,PrecisionIK):
            if self.cfg.get('position_servo',{}).get('retain_tool_orientation')=='full_pose':
                self.ik.orientation_target=self.data.site_xmat[self.site].reshape(3,3).copy()
            for target in self.targets:
                _,possible=self.ik.solve(target,seed=self.command)
                self.reachable=self.reachable and possible
        info['reachable'] = bool(self.reachable)
        return obs,info

    def _actuator_position_target(self, desired):
        """Spread a delayed 20 Hz setpoint change across the physical steps."""
        if self.phase not in ('lift', 'hold', 'lower') or self.support_lock_q is not None:
            self.ramp_goal = None
            self.actuator_target = desired.copy()
            return self.actuator_target
        if self.ramp_goal is None or not np.array_equal(desired, self.ramp_goal):
            self.ramp_goal = desired.copy()
            self.ramp_start = self.actuator_target.copy() if self.actuator_target is not None else desired.copy()
            self.ramp_elapsed = 0.
        self.ramp_elapsed = min(self.dt, self.ramp_elapsed + self.physics_dt)
        fraction = self.ramp_elapsed / self.dt
        target = self.ramp_start + fraction * (self.ramp_goal - self.ramp_start)
        rate_limit = self.cfg['arm']['max_joint_velocity_rad_s'] * self.physics_dt
        previous = self.actuator_target if self.actuator_target is not None else self.ramp_start
        self.actuator_target = synchronized_step(previous, target, rate_limit)
        return self.actuator_target

    def _physics_step(self):
        desired=self.data.ctrl[:5].copy();jaw=float(self.data.ctrl[5])
        target = self._actuator_position_target(desired)
        self.data.ctrl[:5] = target
        servo=self.cfg.get('position_servo',{})
        if servo.get('integral_enabled',True):
            error=target-self.data.qpos[self.qids]
            self.servo_integral+=servo.get('integral_gain_s_inv',2.)*error*self.physics_dt
            limit=servo.get('integral_limit_rad',.04)
            self.servo_integral=np.clip(self.servo_integral,-limit,limit)
            self.data.ctrl[:5]=np.clip(target+self.servo_integral,self.ranges[:,0],self.ranges[:,1])
        if self.last_contact_command is not None and self.phase in ('close','lift','hold','lower','support','release','retreat','settle_place'):
            self.data.ctrl[5]=self.last_contact_command
        mujoco.mj_step(self.model,self.data)
        # Compensation changes the position target, never adds arbitrary body forces.
        self.data.ctrl[:5]=desired;self.data.ctrl[5]=jaw

    def step(self,action):
        if self.observation_source=='simulated_noisy_pose' and self.phase=='approach' and self.target_refinement is None:
            duration=self.cfg['task']['durations_s'][1]
            if self.phase_time>=max(0.,duration-.5) and len(self.target_measurements)>=16:
                self._refine_approach_target()
        if self.phase=='lower' and self.support_lock_q is None:
            if self.lower_search_reference is None:self.lower_search_reference=self.targets[5].copy()
            duration=self.cfg['task']['durations_s'][5]
            if self.phase_time>=duration:
                settings=self.cfg.get('placement',{})
                self.lower_search_m=min(settings.get('maximum_search_depth_m',.008),
                    self.lower_search_m+settings.get('support_search_speed_m_s',.002)*self.dt)
                self.targets[5]=self.lower_search_reference-np.array([0.,0.,self.lower_search_m])
        output=super().step(action)
        if self.observation_source=='simulated_noisy_pose' and self.phase in ('settle','approach'):
            self.target_measurements.append(self.data.site_xpos[self.site]+np.asarray(output[0][11:14])*.3)
        return output

    def _solve_waypoint(self, seed, previous_residual):
        if (self.phase not in ('lift', 'hold', 'lower') or not isinstance(self.ik, PrecisionIK)
                or self.ik.orientation_target is None):
            return super()._solve_waypoint(seed, previous_residual)

        requested = self.residual
        nominal_q, nominal_possible = self.ik.solve(self.nominal, seed=seed, iterations=35)
        if not nominal_possible:
            self.ik_failure = self.ik.rejection_details(self.nominal, nominal_q)
            self.ik_failure.update(target_kind='nominal', requested_residual_m=requested.tolist(),
                                   applied_residual_m=None)
            return nominal_q, False
        if not np.any(requested):
            return nominal_q, True
        q, possible = super()._solve_waypoint(seed, previous_residual)
        if possible:
            return q, True

        # Backtrack toward zero within the same per-axis slew limit as the
        # requested action. Both endpoints satisfy the slew limit, so all
        # interpolated candidates do as well.
        max_change = self.cfg['simulation']['residual_rate_m_s'] * self.dt
        anchor = np.clip(np.zeros(3), previous_residual - max_change, previous_residual + max_change)
        candidates = [anchor + f * (requested - anchor) for f in (.5, .25, .125, 0.)]
        # A moving nominal waypoint can make the backtracking segment
        # infeasible even when the previous residual is still admissible.
        if np.max(np.abs(previous_residual)) <= self.current_limit:
            candidates.append(previous_residual)
        for applied in candidates:
            if np.array_equal(applied, requested):
                continue
            if np.any(applied):
                candidate_q, accepted = self.ik.solve(self.nominal + applied, seed=seed, iterations=35)
            else:
                candidate_q, accepted = nominal_q, True
            if accepted:
                self.residual = applied
                self.waypoint = self.nominal + applied
                self.residual_projection_count += 1
                self.last_residual_projection = {
                    'phase': self.phase,
                    'time_s': float(self.elapsed),
                    'requested_residual_m': requested.tolist(),
                    'applied_residual_m': applied.tolist(),
                    'requested_ik_error': self.ik.rejection_details(self.nominal + requested, q),
                }
                return candidate_q, True
        self.ik_failure = self.ik.rejection_details(self.waypoint, q)
        self.ik_failure.update(target_kind='residual_within_slew_limit',
                               nominal_target_m=self.nominal.tolist(),
                               requested_residual_m=requested.tolist(), applied_residual_m=None)
        return q, False

    def _refine_approach_target(self):
        samples = np.asarray(self.target_measurements)
        measured = np.mean(samples, axis=0)
        spread = np.std(samples, axis=0)
        delta = measured - self.estimate
        if np.linalg.norm(delta) > .01 or np.max(spread) > .004:
            self.options['invalid_sensor'] = True
            return

        targets = [self.targets[0], *(target + delta for target in self.targets[1:])]
        ik = self.ik
        refit = False
        if isinstance(ik, PrecisionIK) and ik.orientation_target is not None:
            previous_orientation = ik.orientation_target
            # Check the corrected grasp, carry, lower and retreat endpoints
            # against the current pose lock while the jaws are still open.
            endpoints = [targets[index] for index in (1, 3, 5, 8)]
            if not all(ik.solve(point, seed=self.home_ik_seed)[1] for point in endpoints):
                ik.orientation_target = None
                q, refit = ik.solve(targets[3], seed=self.home_ik_seed)
                if refit:
                    # Anchor the new full orientation at the refined lift
                    # endpoint, then retain it throughout the grasp and place.
                    ik.data.qpos[ik.qids] = q
                    ik._forward()
                    ik.orientation_target = ik.data.site_xmat[ik.site].reshape(3, 3).copy()
                    refit = all(ik.solve(point, seed=q)[1] for point in endpoints)
                if not refit:
                    ik.orientation_target = previous_orientation
                    self.reachable = False
                    self.target_refinement = {
                        'samples': len(samples), 'correction_m': delta.tolist(),
                        'sample_std_m': spread.tolist(), 'accepted': False,
                        'reason': 'no_verified_refined_grasp_path',
                        'source': 'fresh_noisy_pose_observation_contract_not_RGB_validation',
                    }
                    return

        self.targets = targets
        self.start_target = self.start_target + delta
        self.estimate = measured
        self.target_refinement = {
            'samples': len(samples), 'correction_m': delta.tolist(),
            'sample_std_m': spread.tolist(), 'full_pose_refit': refit, 'accepted': True,
            'source': 'fresh_noisy_pose_observation_contract_not_RGB_validation',
        }

    def _stop_on_support(self,evidence):
        if self.support_lock_q is not None or self.phase not in ('lower','support'):return
        settings=self.cfg.get('placement',{})
        if not (evidence['support_n']>=settings.get('touch_stop_force_n',.5)
                and abs(evidence['clearance_m'])<=self.cfg['task'].get('support_clearance_tolerance_m',.003)
                and evidence['tilt_rad']<=self.cfg['task']['tilt_limit_rad']):return
        self.support_lock_q=self.data.qpos[self.qids].copy()
        self.support_lock_target=self.data.site_xpos[self.site].copy()
        self.command=self.support_lock_q.copy();self.desired_q=self.support_lock_q.copy()
        self.commands=deque([(float(self.data.time),self.support_lock_q.copy())])
        self.data.ctrl[:5]=self.support_lock_q
        self.residual[:]=0.;self.current_limit=0.;self.phase_start_limit=0.
        for index in (5,6,7):self.targets[index]=self.support_lock_target.copy()
        self.nominal=self.support_lock_target.copy();self.waypoint=self.nominal.copy()
        self.servo_integral[:]=0.
        # Touch stops descent; only the separate sustained load gate permits release.
        self.support_unload_command=float(self.data.qpos[self.closure_qids[0]])
        self.last_contact_command=self.support_unload_command
        self.contact_control.command=self.support_unload_command
        self.data.ctrl[5]=self.support_unload_command
        self._guard_event('support_stop_descent',support_n=float(evidence['support_n']),
            clearance_m=float(evidence['clearance_m']),q_hold=self.support_lock_q.tolist(),
            stop_kind='first_real_support_touch_not_release_acceptance')

    def _contact_guard(self,evidence):
        if self.options.get('no_grip'):return None
        hard=np.maximum(np.asarray(evidence['finger_forces_n'])-np.asarray(evidence['foam_forces_n']),0.)
        control_phase=self.phase
        requested=self.cfg['gripper']['close_m'] if self.phase in ('close','lift','hold','lower','support') else self.cfg['gripper'].get('open_m',0.)
        if self.phase in ('lower','support') and self.support_lock_q is not None:
            if self.support_unload_command is None:self.support_unload_command=float(self.last_contact_command)
            settings=self.cfg.get('placement',{})
            if evidence['support_n']<float(self.model.body_mass[self.body])*9.81*.85:
                self.support_unload_command=max(self.cfg['gripper'].get('open_m',0.),
                    self.support_unload_command-settings.get('unload_open_rate_m_s',.0005)*self.physics_dt)
            requested=self.support_unload_command;control_phase='support_unload'
        row=self.contact_control.update(control_phase,self.physics_dt,
            self.data.qpos[self.closure_qids[0]],requested,evidence['foam_forces_n'],hard,
            self.opposed,actuator_force_n=float(self.data.actuator_force[5]))
        self.last_contact_command=row['command_m'];self.data.ctrl[5]=row['command_m']
        self.grip_contact_confirmed=row['confirmed']
        self.grip_stop_reason=row['state'];self.grip_hold_command=None
        self.grip_contact_time=self.contact_control.opposed_elapsed
        if row['failure']:return row['failure']
        return None

    def _physics_guard(self):
        fault=super()._physics_guard()
        if fault:return fault
        if self.cad.get('linked_gripper'):
            residuals=loop_residuals(self.model,self.data)
            worst=max((r['separation_m'] for r in residuals),default=0.)
            self.maximum_loop_error_m=max(self.maximum_loop_error_m,worst)
            if worst>self.cfg.get('linked_gripper',{}).get('maximum_loop_error_m',.001):
                return 'closed_chain_loop_error'
        return None

    def _info(self,success):
        value=super()._info(success)
        if hasattr(self,'contact_control'):
            value['synchronous_control']={'state':self.contact_control.state,
                'events':copy.deepcopy(self.contact_control.events),'common_command_m':self.last_contact_command,
                'target_refinement':copy.deepcopy(self.target_refinement),'lower_support_search_m':self.lower_search_m,'independent_finger_actuators':0,'maximum_loop_error_m':self.maximum_loop_error_m,
                'residual_projection_count':self.residual_projection_count,
                'last_residual_projection':copy.deepcopy(self.last_residual_projection),
                'force_inputs':'simulation_supervision_not_real_force_sensing'}
        return value
