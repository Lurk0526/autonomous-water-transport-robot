"""Supervisor contract checks; these tests do not run MuJoCo dynamics.

The Mac handoff runs the full suite. These cases check the observed v2 failure
conditions without asserting an unmeasured improvement in grasp success.
"""
from collections import deque
from types import SimpleNamespace
import unittest

import numpy as np

from arm_training.env import BowlEnv,closure_anomaly,supported_from_evidence


def evidence(**overrides):
    value={'support_n':6.3765,'clearance_m':0.,'tilt_rad':.01,
        'bilateral':False,'foam_bilateral':False,
        'finger_forces_n':[0.,0.,0.,0.],'foam_forces_n':[0.,0.,0.,0.]}
    value.update(overrides)
    return value


def supervisor_stub(phase=2):
    env=BowlEnv.__new__(BowlEnv)
    env.cfg={'task':{'tilt_limit_rad':.2,'bilateral_force_n':.1,'require_foam_grasp':False},
        'gripper':{}}
    env.phase_index=phase;env.physics_dt=.01;env.contact_mode='foam_required';env.options={}
    env.closure_qids=np.arange(5,9);env.closure_ranges=np.array([[0.,.055]]*4)
    env.qids=np.arange(5);env.body=0;env.site=0
    env.data=SimpleNamespace(time=9.,qpos=np.r_[np.arange(5)*.1,[.009]*4],
        ctrl=np.r_[np.ones(5),.032],site_xpos=np.array([[.2,.01,.39]]))
    env.model=SimpleNamespace(body_mass=np.array([.65]))
    env.guard_events=[];env.grip_hold_command=None;env.grip_stop_reason=None;env.grip_stop_time=None
    env.grip_contact_time=0.;env.grip_contact_confirmed=False
    env.grip_hard_only_time=0.;env.grip_single_time=0.
    env.support_lock_q=None;env.support_lock_target=None
    env.command=np.ones(5);env.desired_q=np.ones(5)
    env.commands=deque([(8.8,np.ones(5)),(8.9,np.ones(5)*2)])
    env.targets=[np.ones(3)*index for index in range(10)]
    env.residual=np.ones(3)*.002;env.current_limit=.002;env.phase_start_limit=.002
    return env


class SupportEvidenceTests(unittest.TestCase):
    def test_tilted_bowl_uses_surface_clearance_not_origin(self):
        # Mac seed 40010: body origin was +5.409 mm but the bottom was touching.
        actual=evidence(support_n=18.7977,clearance_m=-8.85e-6,tilt_rad=.18159)
        self.assertTrue(supported_from_evidence(actual,6.3765,.001,.001,{'tilt_limit_rad':.2}))

    def test_support_needs_contact_geometry_force_and_stillness(self):
        task={'tilt_limit_rad':.2}
        for values in ({'support_n':2.}, {'clearance_m':.004}, {'tilt_rad':.21}):
            with self.subTest(values=values):
                self.assertFalse(supported_from_evidence(evidence(**values),6.3765,0.,0.,task))
        self.assertFalse(supported_from_evidence(evidence(),6.3765,.03,0.,task))
        self.assertFalse(supported_from_evidence(evidence(),6.3765,0.,.2,task))
        self.assertTrue(supported_from_evidence(evidence(),6.3765,.1,.2,task,settled=False))

    def test_support_halts_descent_and_discards_delayed_commands(self):
        env=supervisor_stub(phase=5);actual=env.data.qpos[env.qids].copy()
        env._stop_on_support(evidence())
        np.testing.assert_array_equal(env.data.ctrl[:5],actual)
        np.testing.assert_array_equal(env.command,actual)
        np.testing.assert_array_equal(env.desired_q,actual)
        self.assertEqual(len(env.commands),1)
        np.testing.assert_array_equal(env.commands[0][1],actual)
        for index in (5,6,7):np.testing.assert_array_equal(env.targets[index],env.data.site_xpos[0])
        np.testing.assert_array_equal(env.residual,np.zeros(3))
        # A second contact cannot overwrite the original stop pose with a lower pose.
        env.data.qpos[:5]+=1.
        env._stop_on_support(evidence())
        np.testing.assert_array_equal(env.support_lock_q,actual)
        self.assertEqual(len(env.guard_events),1)

    def test_airborne_contact_does_not_lock_descent(self):
        env=supervisor_stub(phase=5)
        env._stop_on_support(evidence(support_n=0.,clearance_m=.02))
        self.assertIsNone(env.support_lock_q)
        self.assertEqual(len(env.commands),2)


class ClosureGuardTests(unittest.TestCase):
    def test_mac_negative_slides_fail_independently_of_arm_tracking(self):
        bounds=np.array([[0.,.055]]*4)
        for minimum in (-.01110,-.01122,-.01483):
            self.assertEqual(closure_anomaly([minimum,.009,.009,.009],bounds),'closure_joint_limit')
        self.assertIsNone(closure_anomaly([-.00004,0.,0.,0.],bounds))

    def test_coupling_and_nonfinite_have_distinct_faults(self):
        bounds=np.array([[0.,.055]]*4)
        self.assertEqual(closure_anomaly([.01,.01,.014,.01],bounds),'closure_coupling_limit')
        self.assertEqual(closure_anomaly([float('nan'),0.,0.,0.],bounds),'closure_nonfinite')

    def test_hard_contact_stops_at_actual_opening_and_times_out(self):
        env=supervisor_stub()
        hard=evidence(bilateral=True,finger_forces_n=[2.,0.,2.,0.])
        self.assertIsNone(env._contact_guard(hard))
        self.assertAlmostEqual(env.data.ctrl[5],.009)
        self.assertEqual(env.grip_stop_reason,'hard_contact_before_foam')
        self.assertFalse(env.grip_contact_confirmed)
        env.data.time+=.151
        self.assertEqual(env._contact_guard(evidence()),'hard_contact_before_foam')
        self.assertAlmostEqual(env.data.ctrl[5],.009)

    def test_single_foam_contact_cannot_keep_pushing(self):
        env=supervisor_stub()
        single=evidence(finger_forces_n=[1.,0.,0.,0.],foam_forces_n=[1.,0.,0.,0.])
        self.assertIsNone(env._contact_guard(single))
        self.assertAlmostEqual(env.data.ctrl[5],.009)
        env.data.time+=.151
        self.assertEqual(env._contact_guard(single),'gripper_single_contact_blocked')

    def test_stop_uses_least_closed_finger_not_only_driven_finger(self):
        env=supervisor_stub();env.data.qpos[5:9]=[.010,.010,.008,.010]
        hard=evidence(bilateral=True,finger_forces_n=[2.,0.,2.,0.])
        env._contact_guard(hard)
        self.assertAlmostEqual(env.data.ctrl[5],.008)

    def test_opposed_foam_needs_continuous_confirmation(self):
        env=supervisor_stub()
        foam=evidence(bilateral=True,foam_bilateral=True,
            finger_forces_n=[1.,0.,1.,0.],foam_forces_n=[1.,0.,1.,0.])
        for _ in range(4):
            self.assertIsNone(env._contact_guard(foam));env.data.time+=env.physics_dt
        self.assertFalse(env.grip_contact_confirmed)
        env._contact_guard(evidence());env.data.time+=env.physics_dt
        self.assertEqual(env.grip_contact_time,0.)
        for _ in range(5):
            self.assertIsNone(env._contact_guard(foam));env.data.time+=env.physics_dt
        self.assertTrue(env.grip_contact_confirmed)
        self.assertAlmostEqual(env.data.ctrl[5],.009)

    def test_diagnostic_mode_still_stops_hard_contact(self):
        env=supervisor_stub();env.contact_mode='any_contact_diagnostic'
        hard=evidence(bilateral=True,finger_forces_n=[2.,0.,2.,0.])
        self.assertIsNone(env._contact_guard(hard))
        self.assertAlmostEqual(env.data.ctrl[5],.009)
        self.assertEqual(env.grip_stop_reason,'opposed_contact')

    def test_foam_loss_cannot_turn_into_sustained_hard_push(self):
        env=supervisor_stub()
        foam=evidence(bilateral=True,foam_bilateral=True,
            finger_forces_n=[1.,0.,1.,0.],foam_forces_n=[1.,0.,1.,0.])
        for _ in range(5):
            env._contact_guard(foam);env.data.time+=env.physics_dt
        self.assertTrue(env.grip_contact_confirmed)
        hard=evidence(bilateral=True,finger_forces_n=[1.,0.,1.,0.])
        reason=None
        for _ in range(16):
            reason=env._contact_guard(hard);env.data.time+=env.physics_dt
        self.assertEqual(reason,'hard_contact_after_foam_loss')
        self.assertAlmostEqual(env.data.ctrl[5],.009)


if __name__=='__main__':unittest.main()
