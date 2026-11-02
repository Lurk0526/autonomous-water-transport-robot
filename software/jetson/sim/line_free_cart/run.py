"""Reproducible CPU batch runner, metrics, traces, tests and offline report."""
import argparse
from datetime import datetime, timedelta, timezone
import csv
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest

from .navigation import Experiment, NavigationSimulation
from .model import Obstacle
from .scenarios import cases, straight, sweep_cases
from .mission import run_mission
from .speed_profiles import compare


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_configuration(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    for group in ("vehicle", "navigation"):
        for key, value in config[group].items():
            if type(value) not in (int, float) or not 0 <= value < float("inf") or (value==0 and key not in {'front_overhang_m','rear_overhang_m'}):
                raise ValueError(f"{group}.{key} must be positive and finite")
    if config["navigation"]["dt_s"] > .1:
        raise ValueError("dt_s must be <= 0.1 for the supervisor and sampling model")
    return config


def write_trace(out, result):
    rows = result["rows"]
    path = out/"traces"/(result["name"]+".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path.relative_to(out))


def without_rows(result):
    return {k: v for k, v in result.items() if k != "rows"}


def run(out, config_path, quick=False, run_tests=True,backup_only=False,measurements_path=None):
    from .report import write_report
    started = time.monotonic()
    config = load_configuration(config_path)
    if measurements_path:
        from cart.parameters import apply_measurements
        config,_=apply_measurements(config,json.loads(measurements_path.read_text()),ROOT)
    out.mkdir(parents=True, exist_ok=True)
    primary=[]
    if not backup_only:
        from .line_primary import run_line_cases
        primary_config=deepcopy(config)
        primary_config['navigation']['dt_s']=.1
        primary=run_line_cases(primary_config)
        for result in primary:
            result['trace_csv']=write_trace(out,result)
            print(f"[循迹主方案] {result['name']}: {result['outcome']} / expected={result['expectation_met']}",flush=True)
    navigation = []
    for case in cases():
        result = NavigationSimulation(config, case).run()
        result["trace_csv"] = write_trace(out, result)
        navigation.append(result)
        print(f"[导航] {case.name}: {result['outcome']} / expected={result['expectation_met']}", flush=True)
    missions = []
    for fault in ("", "missing_grasp", "missing_support", "payload_loss"):
        result = run_mission(config, fault)
        result["trace_csv"] = write_trace(out, result)
        missions.append(result)
        print(f"[流程] {result['name']}: {result['final_phase']} / expected={result['expectation_met']}", flush=True)
    sweep = []
    if not quick:
        for case in sweep_cases():
            result = NavigationSimulation(config, case).run()
            # Full-rate evidence retained for each seed, not just summary values.
            result["trace_csv"] = write_trace(out, result)
            sweep.append(without_rows(result))
        print(f"[敏感性] {sum(r['expectation_met'] for r in sweep)}/{len(sweep)} reached expected outcome", flush=True)
    negative = NavigationSimulation(config, Experiment(
        "negative_intrusion", "反例：障碍在车体内部突然出现", straight(),
        obstacles=(Obstacle("intrusion", 1.3, 0, .20, 4),),
        notes="该反例应发生碰撞。它验证碰撞检查没有被停止命令或理想检测结果掩盖，不计入成功导航场景。"
    )).run()
    negative["trace_csv"] = write_trace(out, negative)
    tests = {"run": False, "passed": None, "methods": 0}
    if run_tests:
        stream = io.StringIO()
        patterns=['test_line_free_cart.py','test_cart_runtime.py']+([] if backup_only else ['test_line_primary.py','test_cart_vision.py'])
        suite=unittest.TestSuite(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=pattern) for pattern in patterns)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        (out/"tests.txt").write_text(stream.getvalue(), encoding="utf-8")
        tests = {"run": True, "passed": result.wasSuccessful(), "methods": result.testsRun,
                 "failures": len(result.failures), "errors": len(result.errors),'skipped':len(result.skipped)}
        print(f"[测试] {tests}", flush=True)
    speed = compare()
    for profile in speed["profiles"]:
        profile["trace_csv"] = write_trace(out, dict(profile, name="profile_"+profile["name"]))
    passed = (all(r["expectation_met"] for r in primary+navigation+missions)
              and all(r["expectation_met"] for r in sweep)
              and negative["outcome"] == "collision" and speed["same_distance_and_duration"]
              and (not run_tests or tests["passed"]))
    sources = sorted(HERE.glob("*.py"))+sorted(HERE.glob("*.html"))+sorted((ROOT/'cart').glob('*.py'))+[config_path,
               ROOT/"tests/test_line_free_cart.py", ROOT/"sim/bowl_transport/flow.py",
               ROOT/"sim/bowl_transport/scenario.json",ROOT/'tests/test_cart_runtime.py',ROOT/'tests/test_cart_vision.py',ROOT/'tests/test_line_primary.py',
               ROOT/'config/cart-line-tracking.json',ROOT/'config/cart-integration.json',ROOT/'vision/road_geometry.py']
    if measurements_path:sources.append(measurements_path)
    sources+=sorted((ROOT/'config').glob('cart-*.json'))
    report = {
        "schema_version": 1, "kind": "synthetic_planar_kinematic_and_mock_interface_simulation",
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "platform": {"python": platform.python_version(), "architecture": platform.machine()},
        "all_requested_local_checks_passed": passed,
        "full_sensitivity_sweep_run": not quick,
        "navigation_strategy":{'primary':'track_line_black_white_yellow','backup':'line_free_route_following'},
        "primary_line": [without_rows(r) for r in primary],
        "primary_line_suite_run":not backup_only,
        "primary_control_dt_s":.1,
        "rendered_line_rgb":not backup_only,"physics_rgbd_rendered":False,
        "gpu_needed_for_this_suite": False,
        "isaac_sim_run": False, "isaac_lab_training_run": False,
        "hardware_io": False, "physical_grasp_validated": False,
        "spill_mass_g": None, "vehicle_parameters_measured":bool(config.get('parameter_provenance',{}).get('all_required_measurements_present')) if isinstance(config.get('parameter_provenance'),dict) else False,
        "config": config,
        "navigation": [without_rows(r) for r in navigation],
        "missions": [without_rows(r) for r in missions],
        "sensitivity": sweep,
        "negative_control": without_rows(negative),
        "speed_comparison": dict(speed, profiles=[without_rows(p) for p in speed["profiles"]]),
        "tests": tests,
        "source_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "wall_time_s": round(time.monotonic()-started, 3),
        "not_validated": ["真实定位/建图", "相机图像与RGB-D识别", "轮胎、悬架和路面动力学",
                          "三维机械臂碰撞、夹持与承载", "物理液体和真实洒水量", "任意场地的全局规划与动态绕障"],
    }
    integration_path=out/'integration/readiness.json'
    report['integration_snapshot']=json.loads(integration_path.read_text()) if integration_path.exists() else None
    (out/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out/"config.snapshot.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out/"report.html", report, primary+navigation+missions+[negative], speed)
    print(f"Report: {out/'report.html'}", flush=True)
    print(f"ALL_LOCAL_CHECKS_PASSED={passed}; wall_time_s={report['wall_time_s']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"output")
    parser.add_argument("--config", type=Path, default=HERE/"config.json")
    parser.add_argument("--quick", action="store_true", help="omit the 27-case sensitivity sweep (labelled in report)")
    parser.add_argument("--skip-tests", action="store_true", help="omit regression tests (labelled in report)")
    parser.add_argument('--backup-only',action='store_true',help='Only the standard-library line-free backup suite')
    parser.add_argument('--measurements',type=Path,default=ROOT/'config/cart-vehicle-measurements.json')
    args = parser.parse_args()
    report = run(args.out.resolve(), args.config.resolve(), args.quick, not args.skip_tests,args.backup_only,args.measurements.resolve())
    return 0 if report["all_requested_local_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
