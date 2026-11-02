"""Timed response and full-range handoff checks using a simulated servo bus."""
import itertools
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tools import prepare_follow_trial as preparation
from vision.bowl_follow import BoundedArm, FollowPolicy, FollowSettings
from vision.follow_calibration import estimate_jacobian
from vision.tracking_profile import apply_tracking_profile
from tests.test_bowl_follow import FakeArm, detection
from tests.test_full_range_start import full_config
from tests.test_pose_response import measured


class ThreeAxisArm(FakeArm):
    def move(self,targets,duration_ms,allowed_axes=None):
        super().move(targets,duration_ms)


def response_config(responsive=True):
    config=full_config()
    config.update(filter_alpha=.8,center_tolerance_px=6,hold_hysteresis=1.5,duration_ms=500,
                  settle_seconds=.2,supervised_trial=True,local_calibration=measured({str(a):1500 for a in range(6)}),
                  operator_target_lock={'center':[160,120],'stamp_unix':time.time(),
                    'weights_sha256':'c402769dabe8f46fd69e01f76054622b15677862fd34e1825618fedc4b9073a6'},
                  feedback_deadband_p={'0':dict(positive=16,negative=14),
                    '2':dict(positive=-6,negative=27),'3':dict(positive=27,negative=-5)},
                  selective_stop=True,bounded_stop_feedback=True,fresh_frame_after_feedback=True,
                  session_travel_p=12000)
    if responsive:config['response_profile']='responsive-v1'
    return apply_tracking_profile(config)


def response_settings(config):
    return FollowSettings(jacobian=config['local_calibration']['jacobian'],joint_axes=(0,2,3),
        bounded_allocation=True,adaptive_gain=True,smooth_coordination=True,
        responsive_tracking=config.get('responsive_tracking',False),stable_frames=2,
        max_step=config['max_step_p'],min_step=1,period=config['planning_period_s'],
        alpha=config['filter_alpha'],deadband_x=config['center_tolerance_px']/160,
        deadband_y=config['center_tolerance_px']/120,hold_hysteresis=config['hold_hysteresis'],
        gain_near=config['gain_near'],gain_far=config['gain_far'],gain_scale_px=config['gain_scale_px'],
        joint_weights=config['joint_weights'],handoff_weights=config['handoff_weights'],
        command_step_limits=config['command_step_limits_p'],
        negative_step_limits=config['negative_command_step_limits_p'],
        deadband_compensation=tuple((config['feedback_deadband_p'][str(a)]['positive'],
                                    config['feedback_deadband_p'][str(a)]['negative']) for a in (0,2,3)))


def simulate_response(responsive=True,offset=(60,35),seconds=6):
    """20 FPS input, 60 ms image delay, directional bias, and 40 ms queries."""
    clock=[0.];epoch=time.time();config=response_config(responsive)
    class ServoBus:
        def __init__(self):
            self.history={a:[(0.,1500.,1500.,0.)] for a in range(6)}
            self.commands=[];self.stops=0
        def pose(self,at):
            result=[]
            for axis in range(6):
                row=next(row for row in reversed(self.history[axis]) if row[0]<=at)
                start,origin,goal,duration=row
                fraction=min(1,max(0,(at-start)/duration)) if duration else 1
                result.append(origin+(goal-origin)*fraction)
            return np.array(result)
        def position(self,axis):
            clock[0]+=.04
            return round(self.pose(clock[0])[axis])
        def move(self,targets,duration_ms,allowed_axes=None):
            pose=self.pose(clock[0]);self.commands.append((clock[0],dict(targets)))
            for axis,goal in targets.items():
                sign=1 if goal>pose[axis] else -1
                bias=config['feedback_deadband_p'][str(axis)]['positive' if sign>0 else 'negative']
                self.history[axis].append((clock[0],pose[axis],goal-sign*bias,duration_ms/1000))
        def stop_axes(self,axes):
            self.stops+=1;pose=self.pose(clock[0])
            for axis in axes:self.history[axis].append((clock[0],pose[axis],pose[axis],0.))
        def stop(self):self.stop_axes(range(6))
    bus=ServoBus();settings=response_settings(config);policy=FollowPolicy(settings)
    with patch('vision.bowl_follow.time.monotonic',side_effect=lambda:clock[0]), \
         patch('vision.bowl_follow.time.time',side_effect=lambda:epoch+clock[0]):
        config['operator_target_lock']['stamp_unix']=epoch
        arm=BoundedArm(bus,config,clock[0]);start=clock[0];next_frame=0.;record=None
        plan=dict(state='waiting_for_camera',delta_p={});first_move=None;settled_at=None;dwell=None
        while clock[0]-start<seconds:
            elapsed=clock[0]-start
            if elapsed>=next_frame:
                captured=max(start,clock[0]-.06)
                error=np.asarray(offset if captured-start>=.3 else (0,0),dtype=float)
                center=np.array([160.,120.])+error+np.asarray(settings.jacobian)@(bus.pose(captured)[[0,2,3]]-1500)
                record=detection(epoch+captured,center);next_frame=elapsed+.05
            settings.displacement_bounds=arm.displacement_bounds()
            candidate=policy.update(record,epoch+clock[0],clock[0])
            if candidate['state']!='duplicate':plan=candidate
            result=arm.tick(plan,clock[0])
            if first_move is None and bus.commands:first_move=bus.commands[0][0]-start-.3
            if result in ('command_sent','command_retargeted','feedback_verified','direction_changed','coordination_changed'):
                plan=dict(plan,state='rate_limited',delta_p={})
            actual_error=np.asarray(offset)+np.asarray(settings.jacobian)@(bus.pose(clock[0])[[0,2,3]]-1500)
            if elapsed>.3 and np.max(np.abs(actual_error))<=8 and plan['state']=='centered':
                if dwell is None:dwell=elapsed
                elif elapsed-dwell>=.2:
                    settled_at=dwell-.3;break
            else:dwell=None
            if arm.fault:raise AssertionError(arm.fault)
            clock[0]+=.02
        return dict(first_motion_s=first_move,settled_s=settled_at,commands=len(bus.commands),
                    final_error_px=actual_error.tolist(),stops=bus.stops,retargets=arm.retarget_count)


class ResponsiveTests(unittest.TestCase):
    def test_short_unresponsive_attempts_learn_bounded_bias_but_real_progress_does_not(self):
        for moves in (False,True):
            with self.subTest(moves=moves):
                config=response_config();config['adaptive_servo_bias']=True
                transport=ThreeAxisArm();clock=[10.]
                with patch('vision.bowl_follow.time.monotonic',side_effect=lambda:clock[0]):
                    arm=BoundedArm(transport,config,clock[0])
                    original=arm.feedback_deadband['0']['negative']
                    for i in range(3):
                        clock[0]=10+i*.25
                        plan=dict(state='move',delta_p={0:-17},source_stamp_unix=time.time())
                        self.assertEqual(arm.tick(plan,clock[0]),'command_sent')
                        if moves:transport.positions[0]-=2
                        clock[0]+=.15
                        self.assertEqual(arm.tick(dict(state='centered',delta_p={}),clock[0]),'centered')
                    self.assertEqual(arm.feedback_deadband['0']['negative'],original if moves else original+3)
                    self.assertIsNone(arm.fault)

    def test_widely_spaced_interruptions_do_not_accumulate_deadband(self):
        config=response_config();config['adaptive_servo_bias']=True;clock=[10.];transport=ThreeAxisArm()
        with patch('vision.bowl_follow.time.monotonic',side_effect=lambda:clock[0]):
            arm=BoundedArm(transport,config,clock[0]);original=arm.feedback_deadband['0']['negative']
            for i in range(4):
                clock[0]=10+i*2
                self.assertEqual(arm.tick(dict(state='move',delta_p={0:-17},source_stamp_unix=time.time()),clock[0]),'command_sent')
                clock[0]+=.15;arm.tick(dict(state='centered',delta_p={}),clock[0])
            self.assertEqual(arm.feedback_deadband['0']['negative'],original)

    def test_small_bowl_movement_leaves_hold_band_without_chasing_one_pixel_noise(self):
        legacy=simulate_response(False,(7,0),seconds=2)
        responsive=simulate_response(True,(7,0),seconds=2)
        self.assertEqual(legacy['commands'],0)
        self.assertGreater(responsive['commands'],0)
        self.assertLess(responsive['first_motion_s'],.2)
        config=response_config();settings=response_settings(config);policy=FollowPolicy(settings)
        now=time.time()
        for i in range(25):
            plan=policy.update(detection(now+i*.05,(160+i%2,120-i%2)),now+i*.05,.05*i)
            self.assertFalse(plan['delta_p'])

    def test_shorter_motion_pipeline_reduces_simulated_settling_time(self):
        for offset in ((60,35),(0,50),(-60,-35)):
            with self.subTest(offset=offset):
                legacy=simulate_response(False,offset);responsive=simulate_response(True,offset)
                self.assertIsNotNone(legacy['settled_s'],legacy)
                self.assertIsNotNone(responsive['settled_s'],responsive)
                self.assertLess(responsive['settled_s'],legacy['settled_s']*.85,(legacy,responsive))

    def test_axis_handoff_releases_old_motion_and_checks_feedback_before_replanning(self):
        config=response_config();transport=FakeArm()
        class Transport(FakeArm):
            def move(self,targets,duration_ms,allowed_axes=None):super().move(targets,duration_ms)
        for bad_feedback in (False,True):
            with self.subTest(bad_feedback=bad_feedback):
                transport=Transport();now=time.monotonic();arm=BoundedArm(transport,config,now)
                initial=dict(state='move',delta_p={0:40,3:40},source_stamp_unix=time.time(),
                             center_px=[190,140],target_px=[160,120])
                self.assertEqual(arm.tick(initial,now),'command_sent')
                transport.positions.update({0:1510,3:1508 if not bad_feedback else 1700})
                new=dict(initial,delta_p={0:30},center_px=[180,120])
                result=arm.tick(new,now+.25)
                if bad_feedback:self.assertTrue(result.startswith('fault:'),result)
                else:self.assertEqual(result,'coordination_changed')
                self.assertIsNone(arm.pending)
                self.assertEqual(len(transport.commands),1)
                self.assertGreater(transport.stops,0)

    def test_full_range_corner_and_camera_rotation_handoff_stays_inside_limits(self):
        for endpoints in itertools.product((500,2500),repeat=6):
            for angle in (0,math.pi/2,math.pi,3*math.pi/2):
                with self.subTest(endpoints=endpoints,angle=angle):
                    config=response_config();config['pose_reference']={str(a):p for a,p in enumerate(endpoints)}
                    config=apply_tracking_profile(config);transport=FakeArm();transport.positions=dict(enumerate(endpoints))
                    arm=BoundedArm(transport,config,time.monotonic());settings=response_settings(config)
                    rotation=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
                    matrix=rotation@np.asarray(settings.jacobian);settings.jacobian=matrix
                    settings.displacement_bounds=arm.displacement_bounds()
                    feasible=np.array([20 if endpoints[a]==500 else -20 for a in (0,2,3)],dtype=float)
                    feasible[2]*=.5;error=-matrix@feasible
                    policy=FollowPolicy(settings);stamp=time.time()
                    for i in range(2):
                        plan=policy.update(detection(stamp+i*.08,np.array([160,120])+error),stamp+i*.08,.08*i)
                    self.assertTrue(plan['delta_p'],plan)
                    actual=[]
                    for a in (0,2,3):
                        command=plan['delta_p'].get(a,0)
                        self.assertTrue(500<=endpoints[a]+command<=2500)
                        bias=config['feedback_deadband_p'][str(a)]['positive' if command>0 else 'negative']
                        actual.append(command-(1 if command>0 else -1)*bias if command else 0)
                    self.assertLess(np.linalg.norm(error+matrix@actual),np.linalg.norm(error))

    def test_unobservable_elbow_does_not_block_other_two_observable_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config').mkdir();folder=root/'work/bowl-follow';folder.mkdir(parents=True)
            (root/'config/bowl-follow.runtime.json').write_text(json.dumps(
                dict(motion_range_policy='operator_confirmed_full_range',response_profile='responsive-v1')))
            pose={str(a):1500 for a in range(6)};report=measured(pose)
            for row in report['trials']:
                if row['axis']==2:row['after']['center']=list(row['before']['center'])
            report['jacobian'],report['condition']=estimate_jacobian(report['trials'],(0,2,3))
            report.update(anchor=pose,events=[{'unix':time.time()}],axis_amplitudes={'0':30,'2':60,'3':60},amplitude_p=60)
            (folder/'last-calibration.json').write_text(json.dumps(report))
            arm=Mock();arm.positions.return_value={int(a):p for a,p in pose.items()}
            feed=Mock();feed.current.return_value=dict(center=[160,120],bbox=[100,80,220,160])
            with patch.object(preparation,'ROOT',root),patch.object(preparation,'rclpy'), \
                 patch.object(preparation,'ArmSerial',return_value=arm),patch.object(preparation,'TargetFeed',return_value=feed):
                preparation.prepare_from_calibration()
            config=json.loads((folder/'current-trial.json').read_text());settings=response_settings(config)
            self.assertEqual(settings.jacobian[1][1],0)
            policy=FollowPolicy(settings);stamp=time.time()
            for i in range(2):plan=policy.update(detection(stamp+i*.08,(190,140)),stamp+i*.08,.08*i)
            self.assertNotIn(2,plan['delta_p']);self.assertTrue({0,3}<=set(plan['delta_p']))


if __name__=='__main__':unittest.main()
