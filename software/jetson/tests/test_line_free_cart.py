"""Independent geometry, closed-loop, delay, stop and mission-contract checks."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

from sim.line_free_cart.model import Command, Obstacle, State, Vehicle, obstacle_clearance
from sim.line_free_cart.navigation import Experiment, NavigationSimulation
from sim.line_free_cart.scenarios import cases, straight
from sim.line_free_cart.mission import run_mission
from sim.line_free_cart.speed_profiles import compare


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT/"sim/line_free_cart/config.json").read_text())


class GeometryAndActuatorTests(unittest.TestCase):
    def test_constant_speed_turn_matches_independent_analytic_circle(self):
        cfg = configuration()["vehicle"]
        v, steer, duration = .3, .2, 4.0
        vehicle = Vehicle(cfg, State(v=v, steer=steer))
        for _ in range(200):
            vehicle.advance(Command(v, steer), .02)
        radius = cfg["wheelbase_m"]/math.tan(steer)
        angle = v*duration/radius
        self.assertAlmostEqual(vehicle.state.x, radius*math.sin(angle), delta=2e-6)
        self.assertAlmostEqual(vehicle.state.y, radius*(1-math.cos(angle)), delta=2e-6)
        self.assertAlmostEqual(vehicle.state.yaw, angle, places=8)

    def test_front_overhang_and_margin_are_included_in_collision_check(self):
        cfg = configuration()["vehicle"]
        self.assertLess(obstacle_clearance(State(), Obstacle("front", .43, 0, .03), cfg), 0)
        side = Obstacle("side", .1, .21, .02)
        self.assertGreater(obstacle_clearance(State(), side, cfg), 0)
        self.assertLess(obstacle_clearance(State(), side, cfg, .05), 0)

    def test_rotating_vehicle_and_obstacle_preserves_clearance(self):
        cfg = configuration()["vehicle"]
        o, angle = Obstacle("test", .7, .2, .1), 1.3
        transformed = Obstacle("test", math.cos(angle)*o.x-math.sin(angle)*o.y,
                               math.sin(angle)*o.x+math.cos(angle)*o.y, o.radius)
        self.assertAlmostEqual(obstacle_clearance(State(), o, cfg),
                               obstacle_clearance(State(yaw=angle), transformed, cfg), places=12)

    def test_saturated_steering_respects_angle_and_rate(self):
        cfg = configuration()["vehicle"]
        vehicle = Vehicle(cfg)
        previous = 0.0
        for _ in range(200):
            vehicle.advance(Command(.3, 100), .02)
            self.assertLessEqual(abs(vehicle.state.steer), cfg["max_steer_rad"]+1e-9)
            self.assertLessEqual(abs(vehicle.state.steer-previous)/.02, cfg["steer_rate_rad_s"]+1e-9)
            previous = vehicle.state.steer

    def test_nonfinite_commands_and_invalid_step_are_rejected(self):
        vehicle = Vehicle(configuration()["vehicle"])
        for command, dt in ((Command(math.nan, 0), .02), (Command(), 0), (Command(), math.nan)):
            with self.assertRaises(ValueError):
                vehicle.advance(command, dt)

    def test_collision_sampling_includes_intermediate_vehicle_positions(self):
        cfg = configuration()["vehicle"]
        vehicle = Vehicle(cfg, State(v=.5))
        samples = vehicle.advance(Command(.5, 0), 1.0)
        self.assertGreaterEqual(len(samples), 100)
        self.assertTrue(any(.20 < s.x < .30 for s in samples))

    def test_goal_is_not_a_wall_for_front_overhang_clearance(self):
        road=straight(3,half_width=.48)
        self.assertAlmostEqual(road.corridor_error(3.45,0,3),0)
        self.assertGreater(abs(road.corridor_error(3.45,.60,3)),road.half_width)


class ClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = configuration()
        cls.results = {c.name: NavigationSimulation(cls.cfg, c).run() for c in cases()}

    def test_straight_left_right_and_s_routes_finish_without_collision(self):
        for name in ("straight", "left", "right", "s_curve", "noise_delay"):
            with self.subTest(name=name):
                result = self.results[name]
                self.assertTrue(result["expectation_met"], result["outcome"])
                self.assertLess(result["metrics"]["rms_cross_track_m"], .10)
                self.assertLess(result["metrics"]["goal_error_m"], .10)

    def test_nominal_motion_obeys_actuator_acceleration_jerk_and_steering_limits(self):
        vehicle = self.cfg["vehicle"]
        for name in ("straight", "left", "right", "s_curve", "noise_delay"):
            with self.subTest(name=name):
                for row in self.results[name]["rows"]:
                    self.assertLessEqual(row["accel_m_s2"], vehicle["max_accel_m_s2"]+1e-8)
                    self.assertGreaterEqual(row["accel_m_s2"], -vehicle["max_decel_m_s2"]-1e-8)
                    self.assertLessEqual(abs(row["jerk_m_s3"]), vehicle["max_jerk_m_s3"]+1e-8)
                    self.assertLessEqual(abs(row["steer_rate_rad_s"]), vehicle["steer_rate_rad_s"]+1e-8)

    def test_obstacle_cases_stop_before_contact_and_do_not_claim_task_completion(self):
        for name in ("persistent_obstacle", "sudden_obstacle", "unknown_area", "blocked_corridor"):
            with self.subTest(name=name):
                result = self.results[name]
                self.assertTrue(result["expectation_met"], result["outcome"])
                self.assertFalse(result["task_completed"])
                self.assertGreater(result["metrics"]["min_obstacle_clearance_m"], .08)
                self.assertEqual(result["metrics"]["final_speed_m_s"], 0)

    def test_temporary_obstacle_requires_clear_dwell_before_resuming(self):
        result = self.results["temporary_obstacle"]
        self.assertTrue(result["expectation_met"])
        stops = [e for e in result["events"] if e["mode"] == "obstacle_stop"]
        self.assertTrue(stops)
        resumes = [e for e in result["events"] if e["mode"] == "normal" and e["t_s"] > stops[0]["t_s"]]
        self.assertTrue(resumes)
        self.assertGreaterEqual(resumes[0]["t_s"], 13+self.cfg["navigation"]["clear_dwell_s"])

    def test_roadside_obstacle_does_not_block_the_swept_vehicle(self):
        result = self.results["offpath_obstacle"]
        self.assertTrue(result["expectation_met"])
        self.assertNotIn("obstacle_stop", [e["mode"] for e in result["events"]])

    def test_detour_avoids_obstacle_without_moving_the_road_boundaries(self):
        for name in ("detour", "detour_side"):
            with self.subTest(name=name):
                result = self.results[name]
                self.assertTrue(result["expectation_met"], result["outcome"])
                self.assertTrue(result["planning"]["accepted"])
                self.assertGreater(result["metrics"]["min_obstacle_clearance_m"], .08)
                self.assertGreater(result["metrics"]["min_corridor_clearance_m"], 0)
                self.assertTrue(all(y == 0 for _, y in result["route"]))
                self.assertGreater(max(abs(y) for _, y in result["reference_route"]), .5)
        self.assertLess(self.results["detour_side"]["planning"]["lateral_offset_m"], 0)

    def test_no_feasible_detour_falls_back_to_stop(self):
        result = self.results["detour_rejected"]
        self.assertTrue(result["expectation_met"])
        self.assertTrue(result["planning"]["attempted"])
        self.assertFalse(result["planning"]["accepted"])
        self.assertFalse(result["task_completed"])

    def test_all_extended_route_and_error_scenarios_have_explicit_expected_outcomes(self):
        for case in cases()[22:]:
            with self.subTest(case=case.name):
                self.assertTrue(self.results[case.name]['expectation_met'],self.results[case.name]['outcome'])

    def test_crossing_loop_does_not_skip_to_goal_at_the_initial_position(self):
        result=self.results['crossing_loop']
        self.assertTrue(result['task_completed'])
        self.assertGreater(result['metrics']['duration_s'],40)
        self.assertGreater(result['rows'][-1]['progress_m'],20)

    def test_curved_corridor_detours_remain_inside_original_road(self):
        for name in ['curved_detour','s_detour']:
            with self.subTest(name=name):
                result=self.results[name]
                self.assertTrue(result['planning']['accepted'])
                self.assertGreater(result['metrics']['min_obstacle_clearance_m'],.08)
                self.assertGreater(result['metrics']['min_corridor_clearance_m'],0)
        self.assertFalse(self.results['curved_no_space']['planning']['accepted'])

    def test_sensor_recovery_does_not_automatically_clear_a_latched_stop(self):
        c=Experiment('recover','恢复观测',straight(),fault='localization_dropout',fault_at_s=2,fault_end_s=3)
        sim=NavigationSimulation(self.cfg,c)
        for _ in range(100):sim.step()
        self.assertEqual(sim.latched_fault,'stale_localization')
        self.assertLess(sim.rows[-1]['pose_age_s'],.2)
        self.assertEqual(sim.vehicle.state.v,0)

    def test_stale_frozen_jumped_and_nonfinite_inputs_latch_a_real_stop(self):
        for name in ("localization_dropout", "obstacle_dropout", "frozen_timestamp", "pose_jump", "nonfinite_pose"):
            with self.subTest(name=name):
                result = self.results[name]
                self.assertTrue(result["expectation_met"], result["outcome"])
                self.assertLessEqual(result["events"][-1]["t_s"], 4.6)
                self.assertLess(result["metrics"]["final_speed_m_s"], .0051)
                self.assertLess(result["metrics"]["duration_s"], 6.0)
                self.assertFalse(result["task_completed"])

    def test_unreachable_turn_and_uncertain_or_outside_pose_prevent_departure(self):
        for name in ("tight_turn", "uncertain_pose", "outside_corridor"):
            with self.subTest(name=name):
                result = self.results[name]
                self.assertTrue(result["expectation_met"])
                self.assertEqual(result["metrics"]["max_speed_m_s"], 0)

    def test_fixed_seed_reproduces_delayed_noisy_rollout(self):
        case = next(c for c in cases() if c.name == "noise_delay")
        again = NavigationSimulation(self.cfg, case).run()
        self.assertEqual(self.results["noise_delay"]["rows"], again["rows"])

    def test_halved_timestep_preserves_straight_route_result(self):
        cfg = deepcopy(self.cfg)
        cfg["navigation"]["dt_s"] /= 2
        result = NavigationSimulation(cfg, next(c for c in cases() if c.name == "straight")).run()
        self.assertEqual(result["outcome"], "arrived")
        self.assertAlmostEqual(result["metrics"]["goal_error_m"],
                               self.results["straight"]["metrics"]["goal_error_m"], delta=.015)

    def test_ground_truth_collision_is_not_hidden_by_stale_or_missing_detection(self):
        # Negative control: obstacle materializes overlapping the vehicle. No
        # causal controller can avoid this; world truth must report collision.
        case = Experiment("negative_intrusion", "车体内突现障碍反例", straight(),
                          obstacles=(Obstacle("intrusion", 1.3, 0, .20, 4),))
        result = NavigationSimulation(self.cfg, case).run()
        self.assertEqual(result["outcome"], "collision")
        self.assertFalse(result["collision_free"])
        self.assertFalse(result["expectation_met"])


class MissionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {fault: run_mission(configuration(), fault)
                       for fault in ("", "missing_grasp", "missing_support", "payload_loss")}

    def test_base_drives_to_both_stations_and_remains_stopped_during_contact(self):
        result = self.results[""]
        self.assertTrue(result["expectation_met"], result["hold_reason"])
        self.assertEqual(result["final_phase"], "DONE")
        self.assertGreater(result["rows"][-1]["x_m"], 7.9)
        self.assertEqual(result["contact_motion_violations"], 0)
        self.assertEqual(result["release_without_support"], 0)
        self.assertFalse(result["physical_grasp_validated"])
        self.assertIsNone(result["spill_mass_g"])

    def test_elapsed_grasp_time_cannot_replace_confirmation(self):
        result = self.results["missing_grasp"]
        self.assertTrue(result["expectation_met"])
        self.assertNotIn("LIFT", [e["phase"] for e in result["events"]])
        self.assertTrue(result["final_gripper_closed"])

    def test_missing_support_prevents_open_command(self):
        result = self.results["missing_support"]
        self.assertTrue(result["expectation_met"])
        self.assertNotIn("OPEN", [e["phase"] for e in result["events"]])
        self.assertFalse(any(r["release_evidence"] for r in result["rows"]))

    def test_payload_loss_latches_hold_then_applies_vehicle_braking(self):
        result = self.results["payload_loss"]
        self.assertTrue(result["expectation_met"])
        hold = next(e for e in result["events"] if e["phase"] == "HOLD")
        self.assertLess(result["final_speed_m_s"], .0051)
        self.assertGreater(result["duration_s"], hold["t_s"])
        self.assertTrue(result["final_gripper_closed"])


class TrajectoryComparisonTests(unittest.TestCase):
    def test_comparison_controls_distance_duration_and_reports_acceleration_tradeoff(self):
        result = compare()
        a, b = result["profiles"]
        self.assertTrue(result["same_distance_and_duration"])
        self.assertEqual(a["duration_s"], b["duration_s"])
        self.assertAlmostEqual(a["distance_m"], b["distance_m"], places=7)
        self.assertGreater(b["peak_accel_m_s2"], a["peak_accel_m_s2"])
        self.assertLess(b["peak_sampled_jerk_m_s3"], a["peak_sampled_jerk_m_s3"])
        self.assertIsNone(result["spill_mass_g"])


if __name__ == "__main__":
    unittest.main()
