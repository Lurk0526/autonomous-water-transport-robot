"""One-rack compliant grasp supervisor; force inputs are simulation evidence.

This module has no physics engine, hardware transport, target-pose input or
per-finger output. It converts measured/simulated contact evidence to one bounded
position command. The defaults are engineering assumptions, not identified
MG995/MG23 force or foam parameters. Hard contact is never classified as a valid
substitute for foam support.
"""
from __future__ import annotations

import math
import numpy as np


DEFAULTS={
    'rack_min_m':0.,'rack_max_m':.009,
    'contact_force_n':.1,'contact_confirmation_s':.1,
    'confirmation_total_foam_n':8.,'confirmation_opposed_finger_n':.5,
    'foam_estimate_filter_s':.02,
    'close_rate_m_s':.0015,'open_rate_m_s':.003,
    'single_extra_stroke_m':.0012,'single_contact_timeout_s':.8,
    'maximum_contact_extra_stroke_m':.0025,
    'foam_total_target_n':16.,'foam_single_limit_n':8.,'foam_total_limit_n':25.,
    'foam_force_deadband_n':.5,'force_position_gain_m_n_s':.00015,
    'contact_loss_timeout_s':.20,
    'hard_micro_peak_n':.5,'hard_micro_total_n':1.,'hard_allowed_duration_s':.04,
    'hard_emergency_peak_n':2.,'hard_emergency_total_n':4.,
    'rack_force_target_n':20.,'rack_force_limit_n':40.,
    'rack_position_error_limit_m':.003,
    'allow_bounded_mixed_contact':False,'mixed_hard_peak_n':4.,'mixed_hard_total_n':8.,'mixed_hard_fraction':.35,
}


class SynchronousGraspController:
    """Seek -> compliant single contact -> confirmed opposed hold, one input.

    ``rack_command`` is the caller's open-loop requested common stroke. Initial
    seeking does not exceed it. Once foam has touched, a bounded common preload
    may exceed that request by at most ``maximum_contact_extra_stroke_m`` from
    the measured first-contact stroke. Thus a request must be sent only in an
    explicitly enabled closing phase. All forces remain a simulation supervisor
    assumption until the real robot has a validated force/current estimator.

    A failure latches until ``reset``. ``recenter_hint`` gives a finger index,
    never a world direction: transform actual pad/contact kinematics externally.
    """
    CLOSING_PHASES={'close','lift','hold','lower','support'}

    def __init__(self,cfg=None):
        supplied=cfg or {}
        self.cfg={**DEFAULTS,**supplied.get('compliant_grasp',supplied)}
        for key,value in self.cfg.items():
            if key in DEFAULTS and (not math.isfinite(float(value)) or (float(value)<0 and key!='rack_min_m')):
                raise ValueError('Invalid compliant grasp setting '+key)
        c=self.cfg
        if c['rack_max_m']<=c['rack_min_m']:raise ValueError('Invalid rack stroke range')
        for key in ('close_rate_m_s','open_rate_m_s','contact_confirmation_s',
                    'single_contact_timeout_s','force_position_gain_m_n_s','hard_allowed_duration_s'):
            if c[key]<=0:raise ValueError('Positive controller setting required: '+key)
        if not c['contact_force_n']<c['foam_single_limit_n']<c['foam_total_limit_n']:
            raise ValueError('Inconsistent foam force bounds')
        if not c['foam_total_target_n']<c['foam_total_limit_n']:
            raise ValueError('Foam force target must remain below its stop limit')
        if not 0<c['confirmation_total_foam_n']<c['foam_total_limit_n']:
            raise ValueError('Confirmation load must remain within the foam force bounds')
        if not c['contact_force_n']<=c['confirmation_opposed_finger_n']<c['foam_single_limit_n']:
            raise ValueError('Invalid opposed-load confirmation threshold')
        if c['hard_micro_peak_n']>=c['hard_emergency_peak_n'] or c['hard_micro_total_n']>=c['hard_emergency_total_n']:
            raise ValueError('Hard contact emergency bounds must exceed micro-contact bounds')
        self.reset()

    def reset(self):
        self.state='open';self.command=None;self.elapsed=0.;self.failure=None
        self.first_contact_stroke=None;self.first_contact_digit=None;self.single_elapsed=0.
        self.opposed_elapsed=0.;self.confirmed=False;self.contact_lost_elapsed=0.
        self.hard_elapsed=0.;self.hard_only_elapsed=0.;self.peak_foam=0.;self.peak_hard=0.
        self.events=[];self.recenter_hint=None
        self.filtered_foam=None

    def _event(self,kind,**fields):
        self.events.append({'time_s':self.elapsed,'kind':kind,**fields})

    def _fail(self,reason,actual,recenter=False):
        if self.failure is None:
            self.failure=reason;self.state='stopped'
            # Unload a blocked position error; no residual close target survives.
            self.command=float(np.clip(min(self.command,float(actual)),
                self.cfg['rack_min_m'],self.cfg['rack_max_m']))
            if recenter and self.first_contact_digit is not None:
                self.recenter_hint={'action':'open_all_then_recenter_whole_gripper',
                    'toward_first_contact_digit':self.first_contact_digit,
                    'frame':'finger_index_requires_actual_pad_kinematics',
                    'independent_finger_motion':False}
            self._event('stop',reason=reason)

    def _output(self,force_target=0.):
        return {'command_m':float(self.command),'force_target_n':float(force_target),
            'confirmed':bool(self.confirmed),'failure':self.failure,
            'recenter_hint':self.recenter_hint,'state':self.state,
            'events':list(self.events),'uses_simulated_force':True,
            'real_force_estimator_validated':False,'independent_finger_commands':False}

    def update(self,phase,dt,rack_actual,rack_command,foam_forces,hard_forces,
               opposed_pairs,actuator_force_n=None):
        dt=float(dt);actual=float(rack_actual);requested=float(rack_command);c=self.cfg
        if not math.isfinite(dt) or dt<=0 or dt>.1:raise ValueError('Controller dt must be in (0, 0.1] seconds')
        if not math.isfinite(actual) or not math.isfinite(requested):raise ValueError('Nonfinite rack position')
        foam=np.asarray(foam_forces,dtype=float);hard=np.asarray(hard_forces,dtype=float)
        if foam.shape!=(4,) or hard.shape!=(4,) or not np.isfinite(foam).all() or not np.isfinite(hard).all():
            raise ValueError('Four finite foam and hard contact force magnitudes required')
        if np.min(foam)<-1e-8 or np.min(hard)<-1e-8:raise ValueError('Contact magnitudes must be nonnegative')
        foam=np.maximum(foam,0.);hard=np.maximum(hard,0.)
        pairs=[tuple(pair) for pair in opposed_pairs]
        if not pairs or any(len(pair)!=2 or any(not isinstance(i,(int,np.integer)) or not 0<=i<4 for i in pair) or pair[0]==pair[1] for pair in pairs):
            raise ValueError('Valid opposed finger index pairs required')
        if actuator_force_n is not None and not math.isfinite(float(actuator_force_n)):
            raise ValueError('Nonfinite actuator force estimate')
        self.elapsed+=dt
        if self.command is None:self.command=float(np.clip(actual,c['rack_min_m'],c['rack_max_m']))
        if self.failure:return self._output()
        if actual<c['rack_min_m']-.001 or actual>c['rack_max_m']+.001:
            self._fail('common_rack_out_of_range',actual);return self._output()
        peak=float(np.max(foam));total=float(np.sum(foam));hard_peak=float(np.max(hard));hard_total=float(np.sum(hard))
        self.peak_foam=max(self.peak_foam,peak);self.peak_hard=max(self.peak_hard,hard_peak)
        # Micro hard contacts get a finite duration, not unlimited low-force use.
        hard_present=hard_peak>=c['contact_force_n']
        self.hard_elapsed=self.hard_elapsed+dt if hard_present else 0.
        self.hard_only_elapsed=self.hard_only_elapsed+dt if hard_present and peak<c['contact_force_n'] else 0.
        if c['allow_bounded_mixed_contact']:
            if hard_peak>c['mixed_hard_peak_n'] or hard_total>c['mixed_hard_total_n']:
                self._fail('mixed_hard_contact_force_limit',actual);return self._output()
            fraction=hard_total/max(total+hard_total,1e-12)
            if self.hard_elapsed>=c['hard_allowed_duration_s'] and (not self.confirmed or fraction>c['mixed_hard_fraction']):
                self._fail('hard_contact_not_foam_dominant',actual);return self._output()
        else:
            if hard_peak>=c['hard_emergency_peak_n'] or hard_total>=c['hard_emergency_total_n']:
                self._fail('hard_contact_peak_limit',actual);return self._output()
            if hard_peak>c['hard_micro_peak_n'] or hard_total>c['hard_micro_total_n']:
                self._fail('hard_contact_above_micro_tolerance',actual);return self._output()
            if self.hard_elapsed+1e-12>=c['hard_allowed_duration_s']:
                self._fail('sustained_hard_contact',actual);return self._output()
        if peak>=c['foam_single_limit_n'] or total>=c['foam_total_limit_n']:
            self._fail('foam_contact_force_limit',actual,recenter=not self.confirmed);return self._output()
        if actuator_force_n is not None and abs(float(actuator_force_n))>c['rack_force_limit_n']:
            self._fail('rack_force_limit',actual);return self._output()
        if phase not in self.CLOSING_PHASES:
            if phase in ('release','retreat','settle_place'):
                self.state='releasing'
                self.command=max(c['rack_min_m'],self.command-c['open_rate_m_s']*dt)
            else:
                self.state='open'
                self.command=float(np.clip(requested,c['rack_min_m'],c['rack_max_m']))
            return self._output()
        contacted=peak>=c['contact_force_n']
        opposed=any(min(float(foam[i]),float(foam[j]))>=c['contact_force_n'] for i,j in pairs)
        # Filter only the non-safety force-regulation estimate. Every stop bound,
        # contact confirmation and loss timer below uses the current RAW sample.
        # An absent raw contact is cleared immediately rather than carried over
        # as a fictitious filtered contact.
        if self.filtered_foam is None:self.filtered_foam=foam.copy()
        else:
            tau=float(c['foam_estimate_filter_s']);alpha=1. if tau<=0 else 1.-math.exp(-dt/tau)
            self.filtered_foam+=alpha*(foam-self.filtered_foam)
            self.filtered_foam[foam<c['contact_force_n']]=0.
        loaded_opposed=bool(total>=c['confirmation_total_foam_n'] and any(
            min(float(foam[i]),float(foam[j]))>=c['confirmation_opposed_finger_n'] for i,j in pairs))
        if contacted and self.first_contact_stroke is None:
            self.first_contact_stroke=actual;self.first_contact_digit=int(np.argmax(foam));self.single_elapsed=0.
            self._event('first_foam_contact',digit=self.first_contact_digit,rack_m=actual)
        self.opposed_elapsed=self.opposed_elapsed+dt if loaded_opposed else 0.
        if self.opposed_elapsed+1e-12>=c['contact_confirmation_s'] and not self.confirmed:
            self.confirmed=True;self._event('opposed_contact_confirmed',rack_m=actual,
                total_raw_foam_n=total,required_total_foam_n=c['confirmation_total_foam_n'],
                required_opposed_finger_n=c['confirmation_opposed_finger_n'])
        if self.first_contact_stroke is None:
            self.state='seeking'
            goal=float(np.clip(requested,c['rack_min_m'],c['rack_max_m']))
            self.command+=float(np.clip(goal-self.command,-c['open_rate_m_s']*dt,c['close_rate_m_s']*dt))
            return self._output()
        ceiling=min(c['rack_max_m'],self.first_contact_stroke+c['maximum_contact_extra_stroke_m'],
            actual+c['rack_position_error_limit_m'])
        if not self.confirmed:
            self.state='establishing_opposed_contact'
            # This is a unilateral-contact timer, not a hidden three-second
            # deadline for building the full load after both sides have touched.
            # The enclosing close phase still provides its finite global timeout.
            self.single_elapsed=self.single_elapsed+dt if not opposed else 0.
            if not opposed:ceiling=min(ceiling,self.first_contact_stroke+c['single_extra_stroke_m'])
            if self.single_elapsed>=c['single_contact_timeout_s'] or (
                    self.command>=ceiling-1e-9 and not opposed and self.single_elapsed>.1):
                self._fail('opposed_foam_missing_within_compliant_stroke',actual,recenter=True)
                return self._output()
            # Weak opposed contact is not a grasp. Continue bounded common
            # preload while confirming enough actual load, instead of freezing
            # forever at the first 0.1 N threshold crossing.
            error=c['foam_total_target_n']-float(np.sum(self.filtered_foam))
            rate=float(np.clip(c['force_position_gain_m_n_s']*error,
                -c['open_rate_m_s'],c['close_rate_m_s']))
            if actuator_force_n is not None and abs(float(actuator_force_n))>=c['rack_force_limit_n']:
                rate=min(0.,rate)
            self.command=min(ceiling,self.command+rate*dt)
            self.command=max(c['rack_min_m'],min(self.command,ceiling))
            return self._output(c['rack_force_target_n'])
        self.state='force_holding'
        self.contact_lost_elapsed=self.contact_lost_elapsed+dt if not opposed else 0.
        if self.contact_lost_elapsed>=c['contact_loss_timeout_s']:
            self._fail('opposed_foam_contact_lost',actual);return self._output()
        error=c['foam_total_target_n']-float(np.sum(self.filtered_foam))
        if abs(error)<c['foam_force_deadband_n']:error=0.
        increment=float(np.clip(c['force_position_gain_m_n_s']*error*dt,
            -c['open_rate_m_s']*dt,c['close_rate_m_s']*dt))
        # Actuator load can stop more preload, never justify exceeding pad bounds.
        if actuator_force_n is not None and abs(float(actuator_force_n))>=c['rack_force_limit_n']:
            increment=min(0.,increment)
        self.command=float(np.clip(self.command+increment,c['rack_min_m'],ceiling))
        return self._output(c['rack_force_target_n'])
