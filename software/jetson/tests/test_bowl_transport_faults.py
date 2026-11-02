"""Adversarial supervisor checks with synthetic sensor evidence, no hardware I/O.

These establish software rejection behavior only. Fixture booleans are NOT physical
proof that the robot can grasp, hold or support a bowl.
"""
import dataclasses
import json
import math
from pathlib import Path
import sys
import unittest

SIM = Path(__file__).resolve().parents[1] / 'sim' / 'bowl_transport'
sys.path.insert(0, str(SIM))
from flow import Observation, Supervisor, PHASES, CONTACT_PHASES, PAYLOAD_PHASES


def nominal_observation(t, **changes):
    values = dict(sample_time_s=t, speed=0., angular_speed=0., acceleration=0.,
                  telemetry_age=.01, path_age=.01, evidence_age=.01, target_age=.01,
                  pose_error_m=.001, pose_sigma_m=.001, tracking_error_rad=.001,
                  tilt_rad=0., slip_m=0., pose_valid=True, arm_settled=True,
                  path_clear=True, phase_done=True, grasp_verified=True,
                  support_verified=True, release_verified=True, carry_secure=True,
                  preplace_window=True, lift_clearance_verified=True,
                  destination_ready=True, bowl_resting_verified=True,
                  base_hold_confirmed=True, overlap_allowed=False, braking_clear=True,
                  kinematics_ok=True, load_ok=True)
    values.update(changes)
    return Observation(**values)


class SupervisorFaults(unittest.TestCase):
    def setUp(self):
        self.g = json.loads((SIM / 'scenario.json').read_text())['initial_gate_thresholds_require_calibration']

    def supervisor(self, phase):
        f = Supervisor(dict(self.g)); f.phase = phase
        return f

    def step(self, f, dt=.02, **changes):
        t = (f.last_sample or 0.) + dt
        return f.update(nominal_observation(t, **changes), dt)

    def run_for(self, f, count, **changes):
        for _ in range(count):
            self.step(f, **changes)
        return f.phase

    def test_missing_observation_fails_closed(self):
        f = self.supervisor('APPROACH_PICK')
        self.assertEqual(f.update(Observation(), .02), 'HOLD')

    def test_all_boolean_evidence_rejects_truthy_strings_and_integers(self):
        for field in dataclasses.fields(Observation):
            if field.type is bool:
                for value in ('false', 1):
                    with self.subTest(field=field.name, value=value):
                        f = self.supervisor('PICK_WAIT')
                        self.assertEqual(self.step(f, **{field.name: value}), 'HOLD')

    def test_all_numeric_observations_reject_nonfinite_values(self):
        for field in dataclasses.fields(Observation):
            if field.type is not bool:
                for value in (math.nan, math.inf):
                    with self.subTest(field=field.name, value=value):
                        f = self.supervisor('PICK_WAIT')
                        self.assertEqual(self.step(f, **{field.name: value}), 'HOLD')

    def test_replayed_or_discontinuous_timestamp_cannot_build_stability(self):
        for next_time in (.02, .01, .05):
            with self.subTest(next_time=next_time):
                f = self.supervisor('PICK_WAIT')
                f.update(nominal_observation(.02), .02)
                self.assertEqual(f.update(nominal_observation(next_time), .02), 'HOLD')

    def test_one_long_sample_cannot_satisfy_settle(self):
        f = self.supervisor('PICK_WAIT')
        self.assertEqual(self.step(f, dt=.5), 'HOLD')
        self.assertEqual(f.hold_reason, 'sampling gap')

    def test_negative_age_or_error_rejected(self):
        for field in ('sample_time_s','telemetry_age','path_age','evidence_age',
                      'target_age','pose_error_m','pose_sigma_m','tracking_error_rad','slip_m'):
            with self.subTest(field=field):
                f = self.supervisor('PICK_WAIT')
                self.assertEqual(self.step(f, **{field: -.01}), 'HOLD')

    def test_path_obstruction_and_stale_path_stop_every_active_phase(self):
        for phase in PHASES[:-1]:
            for fault in ({'path_clear':False}, {'path_age':1.}):
                with self.subTest(phase=phase, fault=fault):
                    f = self.supervisor(phase)
                    self.assertEqual(self.step(f, **fault), 'HOLD')

    def test_stale_telemetry_and_mechanical_violations_stop_every_active_phase(self):
        faults = ({'telemetry_age':1.}, {'kinematics_ok':False}, {'load_ok':False},
                  {'tracking_error_rad':self.g['tracking_error_rad']*2})
        for phase in PHASES[:-1]:
            for fault in faults:
                with self.subTest(phase=phase, fault=fault):
                    f = self.supervisor(phase)
                    self.assertEqual(self.step(f, **fault), 'HOLD')

    def test_lost_or_uncertain_contact_target_cannot_advance(self):
        faults = ({'pose_valid':False}, {'target_age':99.},
                  {'pose_error_m':.02}, {'pose_sigma_m':.02})
        for phase in ('DESCEND','CLOSE','LOWER'):
            for fault in faults:
                with self.subTest(phase=phase, fault=fault):
                    f = self.supervisor(phase)
                    self.assertEqual(self.step(f, **fault), 'HOLD')

    def test_drop_slip_tilt_and_stale_payload_stop_all_loaded_phases(self):
        faults = ({'carry_secure':False}, {'evidence_age':1.},
                  {'slip_m':.02}, {'tilt_rad':.2})
        for phase in PAYLOAD_PHASES:
            for fault in faults:
                with self.subTest(phase=phase, fault=fault):
                    f = self.supervisor(phase)
                    self.assertEqual(self.step(f, **fault), 'HOLD')

    def test_unconfirmed_brake_or_motion_during_contact_latches_hold(self):
        for phase in CONTACT_PHASES:
            for fault in ({'base_hold_confirmed':False}, {'speed':.03},
                          {'angular_speed':.1}, {'acceleration':.2}):
                with self.subTest(phase=phase, fault=fault):
                    f = self.supervisor(phase)
                    self.assertEqual(self.step(f, **fault), 'HOLD')
                    self.assertEqual(self.step(f), 'HOLD')
                    self.assertFalse(f.requests()['allow_next_phase'])
                    self.assertEqual(f.requests()['gripper'], 'retain_current_command')

    def test_waiting_for_brake_is_allowed_but_cannot_advance(self):
        for phase in ('PICK_WAIT','PLACE_WAIT'):
            f = self.supervisor(phase)
            self.assertEqual(self.run_for(f, 50, base_hold_confirmed=False), phase)
            self.assertEqual(self.run_for(f, 26), 'DESCEND' if phase=='PICK_WAIT' else 'LOWER')

    def test_speed_noise_resets_the_continuous_settle_interval(self):
        for phase in ('PICK_WAIT','PLACE_WAIT'):
            f = self.supervisor(phase)
            for index in range(100):
                self.step(f, speed=.006 if index%10==9 else .004)
            self.assertEqual(f.phase, phase)
            self.run_for(f, 25)
            self.assertEqual(f.phase, phase)
            self.step(f)
            self.assertEqual(f.phase, 'DESCEND' if phase=='PICK_WAIT' else 'LOWER')

    def test_lift_completion_requires_platform_clearance(self):
        f = self.supervisor('LIFT')
        self.assertEqual(self.run_for(f, 50, lift_clearance_verified=False), 'LIFT')
        self.assertEqual(self.step(f), 'RETRACT')

    def test_single_grasp_or_support_or_release_sample_does_not_advance(self):
        for phase in ('CLOSE','LOWER','OPEN','RETREAT'):
            with self.subTest(phase=phase):
                f = self.supervisor(phase)
                self.assertEqual(self.step(f), phase)

    def test_confirmation_glitch_restarts_the_full_evidence_dwell(self):
        for phase, signal in (('CLOSE','grasp_verified'),('LOWER','support_verified'),
                              ('OPEN','release_verified'),('RETREAT','release_verified')):
            with self.subTest(phase=phase):
                f = self.supervisor(phase)
                self.run_for(f, 10)
                self.step(f, **{signal:False})
                self.run_for(f, 13)
                self.assertEqual(f.phase, phase)
                self.step(f)
                self.assertEqual(f.phase, PHASES[PHASES.index(phase)+1])

    def test_release_and_retreat_keep_checking_independent_support(self):
        for phase in ('OPEN','RETREAT'):
            for fault in ({'support_verified':False}, {'bowl_resting_verified':False},
                          {'evidence_age':1.}):
                with self.subTest(phase=phase, fault=fault):
                    f = self.supervisor(phase)
                    self.assertEqual(self.step(f, **fault), 'HOLD')

    def test_absent_support_cannot_be_replaced_by_elapsed_time(self):
        f = self.supervisor('LOWER')
        self.assertEqual(self.run_for(f, 50, support_verified=False), 'LOWER')

    def test_destination_not_ready_blocks_preparation_and_release(self):
        for phase in ('CARRY','PLACE_WAIT','LOWER'):
            with self.subTest(phase=phase):
                f = self.supervisor(phase)
                self.assertEqual(self.run_for(f, 50, destination_ready=False), phase)

    def test_moving_preparation_requires_opt_in_and_full_envelope(self):
        f = self.supervisor('CARRY')
        self.assertEqual(self.step(f, speed=.04, base_hold_confirmed=False), 'CARRY')
        self.assertEqual(self.step(f), 'PREPLACE')
        faults = ({'overlap_allowed':False}, {'braking_clear':False}, {'speed':.12},
                  {'angular_speed':.05}, {'acceleration':.08}, {'tilt_rad':.025})
        for fault in faults:
            with self.subTest(fault=fault):
                f = self.supervisor('PREPLACE')
                values = dict(speed=.04, base_hold_confirmed=False, overlap_allowed=True)
                values.update(fault)
                self.assertEqual(self.step(f, **values), 'HOLD')
        f = self.supervisor('PREPLACE')
        self.assertEqual(self.step(f, speed=.04, base_hold_confirmed=False,
                                   overlap_allowed=True), 'PLACE_WAIT')

    def test_preplace_stops_if_target_or_destination_becomes_unavailable(self):
        for fault in ({'pose_valid':False}, {'target_age':1.}, {'pose_sigma_m':.02},
                      {'pose_error_m':.02}, {'destination_ready':False}):
            with self.subTest(fault=fault):
                f = self.supervisor('PREPLACE')
                self.assertEqual(self.step(f, speed=.04, base_hold_confirmed=False,
                                           overlap_allowed=True, **fault), 'HOLD')

    def test_close_cannot_authorize_lift_when_payload_is_already_slipping_or_tilted(self):
        for fault in ({'slip_m':.02}, {'tilt_rad':.2}):
            for confirmed in ({'grasp_verified':True, 'carry_secure':False},
                              {'grasp_verified':False, 'carry_secure':True}):
                with self.subTest(fault=fault, confirmed=confirmed):
                    f = self.supervisor('CLOSE')
                    self.assertEqual(self.step(f, **fault, **confirmed), 'HOLD')

    def test_preparation_requests_require_fresh_valid_phase_appropriate_evidence(self):
        for phase in ('APPROACH_PICK','PREPLACE'):
            with self.subTest(phase=phase):
                f = self.supervisor(phase)
                good = nominal_observation(.02, speed=.04, base_hold_confirmed=False,
                                           overlap_allowed=True)
                self.assertEqual(f.requests(good)['arm'], 'prepare_allowed')
                self.assertNotEqual(f.requests(Observation())['arm'], 'prepare_allowed')
                faults = ({'pose_valid':False}, {'telemetry_age':1.}, {'path_age':1.},
                          {'path_clear':False}, {'kinematics_ok':False}, {'load_ok':False},
                          {'tracking_error_rad':.2}, {'overlap_allowed':False},
                          {'braking_clear':False}, {'target_age':1.})
                if phase == 'PREPLACE': faults += ({'destination_ready':False},)
                for fault in faults:
                    with self.subTest(phase=phase, fault=fault):
                        self.assertNotEqual(f.requests(dataclasses.replace(good, **fault))['arm'],
                                            'prepare_allowed')
                f.hold('test fault')
                self.assertNotEqual(f.requests(good)['arm'], 'prepare_allowed')
        for phase in set(PHASES)-{'APPROACH_PICK','PREPLACE'}:
            with self.subTest(phase=phase):
                f = self.supervisor(phase)
                self.assertNotEqual(f.requests(nominal_observation(.02, overlap_allowed=True))['arm'],
                                    'prepare_allowed')

    def test_failed_grasp_eventually_times_out_and_cannot_resume_silently(self):
        f = self.supervisor('CLOSE'); f.g['phase_timeout_s'] = .12
        self.run_for(f, 10, grasp_verified=False)
        self.assertEqual(f.phase, 'HOLD')
        self.assertEqual(f.hold_reason, 'phase timeout')
        self.assertEqual(self.step(f), 'HOLD')

    def test_preparation_request_rejects_malformed_or_negative_evidence(self):
        f=self.supervisor('PREPLACE')
        good=nominal_observation(.02,overlap_allowed=True)
        for fault in ({'path_age':-.1},{'target_age':-.1},{'carry_secure':False},
                      {'evidence_age':1.},{'overlap_allowed':'false'},{'slip_m':.02}):
            with self.subTest(fault=fault):
                self.assertNotEqual(f.requests(dataclasses.replace(good,**fault))['arm'],'prepare_allowed')

    def test_nominal_synthetic_evidence_visits_all_phases_in_order(self):
        f = self.supervisor('APPROACH_PICK'); visited = [f.phase]
        for _ in range(300):
            phase = self.step(f)
            if phase != visited[-1]: visited.append(phase)
            if phase in ('DONE','HOLD'): break
        self.assertEqual(visited, list(PHASES))
        self.assertFalse(f.requests()['allow_next_phase'])


if __name__ == '__main__':
    unittest.main()
