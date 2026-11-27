"""Train from the shipped camera features or collect new Gazebo demonstrations."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="collect expert RGB-D sessions with Gazebo first")
    parser.add_argument("--closed-loop", action="store_true", help="evaluate predictions in new Gazebo sessions")
    parser.add_argument("--episodes", type=int, default=16, help="number of new expert episodes across side, slalom and unobstructed layouts")
    parser.add_argument("--start-seed", type=int, default=100, help="first independent expert episode")
    parser.add_argument("--eval-episodes", type=int, default=4, help="number of new closed-loop episodes")
    parser.add_argument("--eval-seed", type=int, default=1000000, help="first seed for the closed-loop check")
    parser.add_argument("--controller", choices=("policy", "guided-policy", "vision-guide"),
                        default="guided-policy", help="controller used for the independent Gazebo check")
    parser.add_argument("--render-engine", choices=("ogre2", "ogre"), default="ogre2")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--features", type=Path, default=ROOT / "starter_features")
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    if opts.epochs < 1 or opts.eval_episodes < 1 or min(opts.start_seed, opts.eval_seed) < 0:
        parser.error("epochs and eval-episodes must be positive, start-seed must be nonnegative")
    if opts.collect and opts.episodes < 3:
        parser.error("Collect at least three distinct seeds for the train, validation and test groups.")
    if opts.collect and (opts.start_seed < opts.eval_seed + opts.eval_episodes
                         and opts.eval_seed < opts.start_seed + group_count(opts.episodes)):
        parser.error("Choose disjoint demonstration and closed-loop seed ranges.")
    return opts


def group_count(episodes):
    return max(3, (episodes + 3) // 4)


def main():
    opts = arguments()
    if (opts.collect or opts.closed_loop) and shutil.which("gz") is None:
        raise SystemExit("采集和闭环需要 Gazebo Harmonic。安装方法见 README.md。")
    output = (opts.output or ROOT / "outputs" / datetime.now().strftime("mac-%Y%m%d-%H%M%S-%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(key, "2")
    stages = []
    summary = output / "workflow.json"

    def step(title, module, *args, allowed_codes=(0,)):
        command = [sys.executable, "-u", "-m", "ring_training." + module, *map(str, args)]
        print(f"\n{title}", flush=True)
        record = {"stage": title, "command": command, "log": f"{len(stages)+1:02}.log"}
        stages.append(record)
        summary.write_text(json.dumps(stages, indent=2, ensure_ascii=False))
        with (output / record["log"]).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
            finally:
                if process.poll() is None:
                    process.wait()
        record["return_code"] = process.returncode
        summary.write_text(json.dumps(stages, indent=2, ensure_ascii=False))
        if process.returncode not in allowed_codes:
            raise SystemExit(f"{title} 退出码 {process.returncode}。日志 {output / record['log']}")
        return process.returncode

    features = opts.features.resolve()
    if opts.collect:
        sessions = output / "episodes"
        features = output / "features"
        groups = group_count(opts.episodes)
        for index in range(opts.episodes):
            seed = opts.start_seed + index % groups
            layout = ("left", "right", "slalom", "clear")[(index // groups + index % groups) % 4]
            rng = random.Random(seed)
            offset, yaw = rng.uniform(-0.037, 0.037), rng.uniform(-0.022, 0.022)
            step(f"采集 Gazebo 专家示范 {layout} seed {seed}", "gazebo_run", "record", "--seed", seed,
                 "--layout", layout, "--sample-hz", 2.5, "--camera-hz", 8,
                 "--spawn-offset", offset, "--spawn-yaw-error", yaw,
                 "--render-engine", opts.render_engine,
                 "--output", sessions / f"{layout}-{seed}")
        step("提取 RGB-D 感知特征", "learning", "extract", "--input", sessions, "--output", features,
             "--sample-step", 2, "--threads", 1, "--workers", 2)
    if not features.is_dir() or not any(features.rglob("*.npz")):
        raise SystemExit(f"找不到逐场训练特征 {features}")
    model = output / "training"
    step("按独立场次训练及验证", "learning", "train", "--input", features,
         "--output", model, "--epochs", opts.epochs)
    step("在保留场次复核预测", "learning", "evaluate", "--input", features,
         "--policy", model / "policy.npz")
    if opts.closed_loop or opts.collect:
        with (model / "metrics.json").open() as stream:
            training = json.load(stream)
        used = {group for groups in training["scenario_groups"].values() for group in groups}
        scores = []
        for seed in range(opts.eval_seed, opts.eval_seed + opts.eval_episodes):
            if f"seed:{seed}" in used:
                raise SystemExit(f"种子 {seed} 已用于示范，请更换 --eval-seed。")
            run_dir = output / "closed-loop" / f"seed-{seed}"
            checkpoint = [] if opts.controller == "vision-guide" else ["--policy", model / "policy.npz"]
            camera_hz = 8 if opts.controller == "policy" else 12
            code = step(f"Gazebo {opts.controller} 闭环 seed {seed}", "gazebo_run", "run",
                        "--controller", opts.controller, *checkpoint, "--seed", seed, "--layout", "mixed",
                        "--camera-hz", camera_hz, "--render-engine", opts.render_engine,
                        "--output", run_dir, allowed_codes=(0, 2))
            result_file = run_dir / "result.json"
            if not result_file.exists():
                raise SystemExit(f"Gazebo 未能生成评估记录。查看 {run_dir} 与本轮日志。")
            score = json.loads(result_file.read_text())
            if score.get("runtime_error") or bool(score["passed"]) != (code == 0):
                raise SystemExit(f"Gazebo 评估运行异常，请检查 {result_file}")
            scores.append({"seed": seed, "controller": score["controller"], "layout": score["layout"],
                           "passed": score["passed"], "failures": score["failures"],
                           "speed_zone_min_kmh": score["speed_zone_adjacent_pose_min_kmh"]})
        (output / "closed-loop-results.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False))
        print(f"闭环合格 {sum(s['passed'] for s in scores)}/{len(scores)} 圈。")
    print(f"\n本轮模型、日志及评估保存在 {output}", flush=True)
    if (opts.closed_loop or opts.collect) and any(not score["passed"] for score in scores):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
