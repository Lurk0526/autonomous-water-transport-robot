"""Existing bowl supervisor + running bicycle vehicle + explicit mock arm I/O.

Arm acknowledgements and target/retention evidence are synthetic fixtures. They
exercise the interface contract; they are never interpreted as a physical grasp.
"""
import math
from pathlib import Path
import json

from sim.bowl_transport.flow import CONTACT_PHASES, Observation, Supervisor
from .navigation import Experiment, NavigationSimulation
from .scenarios import straight


DURATIONS = {"DESCEND": .8, "CLOSE": .8, "LIFT": 1.0, "RETRACT": 1.0,
             "PREPLACE": .8, "LOWER": 1.0, "OPEN": .6, "RETREAT": .8}


def run_mission(config, fault=""):
    root = Path(__file__).resolve().parents[2]
    thresholds = json.loads((root/"sim/bowl_transport/scenario.json").read_text())[
        "initial_gate_thresholds_require_calibration"]
    thresholds = dict(thresholds, phase_timeout_s=120.0)
    # Synthetic travel gets 120 s, mock contact stages 8 s. These timeouts apply
    # only here; the existing hardware-related configuration is not modified.
    supervisor = Supervisor(thresholds)
    nav = NavigationSimulation(config, Experiment("pick_leg", "接近取碗点", straight(3.0),
                                                  initial_offset_m=.08,
                                                  localization_delay_s=0, obstacle_delay_s=0))
    dt = config["navigation"]["dt_s"]
    t = 0.0
    events = [{"t_s": 0.0, "phase": supervisor.phase}]
    rows = []
    closed = grasp = support = released = False
    contact_motion_violations = release_without_support = 0
    carry_started = False
    hold_start = None
    while t < 180:
        phase = supervisor.phase
        supervisor.g["phase_timeout_s"] = 120.0 if phase in {"APPROACH_PICK", "CARRY"} else 8.0
        if phase == "CARRY" and not carry_started:
            nav = NavigationSimulation(config, Experiment("place_leg", "携碗到放置点", straight(5.0, 3.0),
                                                          initial_state=nav.vehicle.state,
                                                          localization_delay_s=0, obstacle_delay_s=0))
            carry_started = True
        is_motion_phase = phase in {"APPROACH_PICK", "CARRY"}
        # The actual supervisor intent is enforced by the simulated actuator.
        intents = supervisor.requests()
        row = nav.step(motion_enabled=is_motion_phase and intents["base"] == "bounded_motion",
                       external_hold="mission_hold" if phase == "HOLD" else "")
        t = round(t+dt, 9)
        arm_done = supervisor.elapsed >= DURATIONS.get(phase, 0.0)
        if phase == "CLOSE" and arm_done:
            closed = True
            grasp = fault != "missing_grasp"
        if phase == "LOWER" and arm_done and fault != "missing_support":
            support = True
        if phase == "OPEN" and arm_done:
            if support:
                closed, released, grasp = False, True, False
            else:
                release_without_support += 1
        payload_lost = fault == "payload_loss" and phase == "CARRY" and supervisor.elapsed >= 2.0
        if payload_lost:
            grasp = False
        quiet = (row["speed_m_s"] <= supervisor.g["linear_speed_m_s"]
                 and abs(row["accel_m_s2"]) <= supervisor.g["acceleration_m_s2"])
        angular_speed = row["speed_m_s"]*math.tan(row["steer_rad"])/config["vehicle"]["wheelbase_m"]
        if phase in CONTACT_PHASES and (not quiet or abs(angular_speed) > supervisor.g["angular_speed_rad_s"]):
            contact_motion_violations += 1
        phase_done = nav.arrived if is_motion_phase else arm_done
        obs = Observation(
            sample_time_s=t, speed=row["speed_m_s"], angular_speed=angular_speed,
            acceleration=row["accel_m_s2"],
            telemetry_age=row["pose_age_s"] if row["pose_age_s"] is not None else math.nan,
            path_age=row["obstacle_age_s"] if row["obstacle_age_s"] is not None else math.nan,
            evidence_age=.01, target_age=.01,
            pose_error_m=.001, pose_sigma_m=.001, tracking_error_rad=.001,
            tilt_rad=0.0, slip_m=0.0, pose_valid=True, arm_settled=not is_motion_phase,
            path_clear=not bool(nav.latched_fault), phase_done=phase_done,
            grasp_verified=grasp, support_verified=support, release_verified=released,
            carry_secure=grasp, preplace_window=(phase == "CARRY" and nav.arrived),
            lift_clearance_verified=phase == "LIFT" and arm_done,
            destination_ready=True, bowl_resting_verified=support,
            base_hold_confirmed=quiet, overlap_allowed=False, braking_clear=True,
            kinematics_ok=True, load_ok=True)
        previous_phase = phase
        supervisor.update(obs, dt)
        if supervisor.phase != previous_phase:
            events.append({"t_s": t, "phase": supervisor.phase, "reason": supervisor.hold_reason})
        row = dict(row, t_s=t, phase=supervisor.phase, gripper_closed=closed,
                   grasp_evidence=grasp, support_evidence=support,
                   release_evidence=released, arm_feedback="synthetic_mock",
                   base_intent=supervisor.requests()["base"])
        rows.append(row)
        if supervisor.phase == "DONE":
            break
        if supervisor.phase == "HOLD":
            if hold_start is None:
                hold_start = t
            if t-hold_start >= 1.0 and row["speed_m_s"] <= .005:
                break
        if nav.collision or nav.left_corridor:
            supervisor.hold("simulated collision or corridor departure")
    expected = "DONE" if not fault else "HOLD"
    expected_reason = {"missing_grasp": "phase timeout", "missing_support": "phase timeout",
                       "payload_loss": "payload not verified secure"}.get(fault, "")
    ok = (supervisor.phase == expected and (not fault or supervisor.hold_reason == expected_reason)
          and not contact_motion_violations and not release_without_support
          and not nav.collision and not nav.left_corridor and rows[-1]["speed_m_s"] <= .005
          and (not fault or closed))
    return {
        "name": "mission_"+(fault or "nominal"),
        "label": {"": "整套取放流程", "missing_grasp": "抓取反馈缺失",
                  "missing_support": "放置承托反馈缺失", "payload_loss": "运输中夹持反馈丢失"}[fault],
        "final_phase": supervisor.phase, "hold_reason": supervisor.hold_reason,
        "expectation_met": ok, "task_completed": supervisor.phase == "DONE",
        "duration_s": t, "contact_motion_violations": contact_motion_violations,
        "release_without_support": release_without_support,
        "final_gripper_closed": closed, "final_speed_m_s": rows[-1]["speed_m_s"],
        "physical_grasp_validated": False, "spill_mass_g": None,
        "base_feedback": "simulated_bicycle_state", "arm_feedback": "synthetic_mock",
        "events": events, "rows": rows,
        "route": straight(8.0).points, "half_width_m": .9, "obstacles": [],
    }
