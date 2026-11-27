"""Extract teacher demonstrations, fit a CPU behavior-cloning policy and score held-out episodes."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .perception import CONE_MODEL, DEPTH_RANGE_M, FEATURE_NAMES, FEATURE_VERSION, ROAD_MODEL, Perception
from .world import WORLD_SCHEMA


ACTION_NAMES = ("target_speed_mps", "target_yaw_rate_rad_s")
SPLITS = ("train", "validation", "test")
POLICY_FORMAT_VERSION = 2
TEACHER_SOURCES = {"gazebo_expert", "expert", "teacher"}
SPARSE_COLOR_SUFFIXES = ("_orange", "_blue", "_cyan", "_magenta")
SPARSE_COLOR_COLUMNS = [index for index, name in enumerate(FEATURE_NAMES) if name.endswith(SPARSE_COLOR_SUFFIXES)]


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        with temporary.open("wb") as file:
            np.savez_compressed(file, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def extract(input_path, output, *, road_model=ROAD_MODEL, cone_model=CONE_MODEL, threads=2,
            workers=1, sample_step=1, max_samples_per_episode=None, depth_max_m=DEPTH_RANGE_M):
    """Read teacher samples.jsonl files and save one portable feature archive per episode."""
    import cv2

    root = Path(input_path).resolve()
    sources = sorted(root.rglob("samples.jsonl")) if root.is_dir() else [root]
    if not sources or any(path.name != "samples.jsonl" or not path.is_file() for path in sources):
        raise ValueError(f"Expected a rollout directory or samples.jsonl under {root}")
    if min(sample_step, workers) < 1 or (max_samples_per_episode is not None and max_samples_per_episode < 1):
        raise ValueError("workers, sample-step and max-samples-per-episode must be positive")
    recordings = []
    for source in sources:
        metadata_path = source.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("world_schema") != WORLD_SCHEMA:
            raise ValueError(f"Expected world_schema {WORLD_SCHEMA!r} in {metadata_path}")
        recordings.append((source, metadata))
    perception = Perception(road_model, cone_model, threads=threads, depth_max_m=depth_max_m)
    seen = set()
    results = []
    for source, metadata in recordings:
        folder = source.parent
        episode_id = str(metadata.get("episode_id", folder.name))
        if episode_id in seen:
            raise ValueError(f"Duplicate episode_id {episode_id!r} in {folder}")
        seen.add(episode_id)
        seed = metadata.get("seed")
        group = str(f"seed:{seed}" if seed is not None else (metadata.get("scenario_group") or episode_id))
        records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        source_default = metadata.get("source", metadata.get("mode"))
        teacher = [item for item in records if item.get("source", source_default) in TEACHER_SOURCES]
        if not teacher:
            print(f"Skipping {episode_id}: no teacher samples", flush=True)
            continue
        events = teacher[::sample_step]
        if max_samples_per_episode is not None and len(events) > max_samples_per_episode:
            indices = np.linspace(0, len(events) - 1, max_samples_per_episode, dtype=int)
            events = [events[index] for index in indices]

        def read_features(event):
            image_path = folder / event["rgb"]
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read RGB frame {image_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = np.load(folder / event["depth"], allow_pickle=False) if event.get("depth") else None
            return perception.features(rgb, depth)

        if workers == 1:
            features = list(map(read_features, events))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                features = list(pool.map(read_features, events))
        actions = [[event[name] for name in ACTION_NAMES] for event in events]
        times = [event["sim_time"] for event in events]
        relative = folder.relative_to(root) if root.is_dir() else Path()
        if not relative.parts:
            relative = Path(folder.name)
        target = Path(output) / relative / "features.npz"
        arrays = _validate_samples(features, actions, episode_id)
        if not np.isfinite(times).all():
            raise ValueError(f"Non-finite timestamps in {source}")
        _save_npz(target, features=arrays[0], actions=arrays[1], timestamps=np.asarray(times, np.float64),
                  episode_id=episode_id, scenario_group=group, feature_version=FEATURE_VERSION,
                  feature_names=FEATURE_NAMES, action_names=ACTION_NAMES, model_hashes=perception.model_hashes,
                  depth_max_m=perception.depth_max_m, world_schema=WORLD_SCHEMA,
                  layout=str(metadata.get("layout") or "unknown"))
        print(f"{episode_id}: {len(events)} teacher samples -> {target}", flush=True)
        results.append(target)
    if not results:
        raise ValueError(f"No teacher episodes in {root}")
    return results


@dataclass(frozen=True)
class Episode:
    episode_id: str
    scenario_group: str
    features: np.ndarray
    actions: np.ndarray
    layout: str


def _validate_samples(features, actions, episode_id):
    features = np.asarray(features, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if (features.ndim != 2 or min(features.shape) < 1 or actions.shape != (len(features), len(ACTION_NAMES))
            or not np.isfinite(features).all() or not np.isfinite(actions).all()):
        raise ValueError(f"{episode_id}: expected finite features (N, D) and actions (N, 2)")
    return features, actions


def load_episodes(input_path):
    root = Path(input_path)
    paths = sorted(root.rglob("features.npz")) if root.is_dir() else [root]
    if root.is_dir() and not paths:
        paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise ValueError(f"No feature archives under {root}")
    episodes = []
    provenance = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            features, actions = _validate_samples(archive["features"], archive["actions"], path)
            version = str(archive["feature_version"].item()) if "feature_version" in archive else ""
            world_schema = str(archive["world_schema"].item()) if "world_schema" in archive else ""
            if (world_schema != WORLD_SCHEMA or version != FEATURE_VERSION or features.shape[1] != len(FEATURE_NAMES)
                    or "feature_names" not in archive or tuple(archive["feature_names"].tolist()) != FEATURE_NAMES):
                raise ValueError(f"Expected {WORLD_SCHEMA!r} / {FEATURE_VERSION!r} features in {path}")
            if "action_names" in archive and tuple(archive["action_names"].tolist()) != ACTION_NAMES:
                raise ValueError(f"Action schema mismatch in {path}")
            hashes = tuple(archive["model_hashes"].tolist()) if "model_hashes" in archive else ()
            depth_max_m = float(archive["depth_max_m"].item()) if "depth_max_m" in archive else DEPTH_RANGE_M
            if not 0 < depth_max_m <= DEPTH_RANGE_M:
                raise ValueError(f"Invalid camera depth limit in {path}")
            provenance.add((version, features.shape[1], hashes, depth_max_m))
            if "episode_ids" in archive:
                ids = archive["episode_ids"].astype(str)
                groups = archive["scenario_groups"].astype(str) if "scenario_groups" in archive else ids
                if ids.shape != (len(features),) or groups.shape != ids.shape:
                    raise ValueError(f"{path}: expected one episode_id and scenario_group per row")
                for episode_id in np.unique(ids):
                    rows = ids == episode_id
                    scenario_groups = np.unique(groups[rows])
                    if len(scenario_groups) != 1:
                        raise ValueError(f"{path}: {episode_id!r} has multiple scenario groups")
                    episodes.append(Episode(str(episode_id), str(scenario_groups[0]), features[rows], actions[rows], "unknown"))
            else:
                episode_id = str(archive["episode_id"].item()) if "episode_id" in archive else path.stem
                group = str(archive["scenario_group"].item()) if "scenario_group" in archive else episode_id
                layout = str(archive["layout"].item()) if "layout" in archive else "unknown"
                episodes.append(Episode(episode_id, group, features, actions, layout))
    ids = [episode.episode_id for episode in episodes]
    if any(not value.strip() for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Episode IDs must be nonempty and unique across archives")
    if len(provenance) != 1:
        raise ValueError("Feature archives use different model weights, depth limits or feature schemas")
    version, _, hashes, depth_max_m = provenance.pop()
    return sorted(episodes, key=lambda episode: episode.episode_id), version, hashes, depth_max_m


def _transform_features(features):
    # Small distant objects and paint occupy only a few pixels of their cell.
    transformed = features.copy()
    sparse = features[:, SPARSE_COLOR_COLUMNS]
    transformed[:, SPARSE_COLOR_COLUMNS] = np.sqrt(np.clip(sparse, 0, 1))
    return transformed


@dataclass(frozen=True)
class Policy:
    """A normalized linear policy or one-hidden-layer MLP serialized as numeric NPZ arrays."""

    model: str
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    weight: np.ndarray
    bias: np.ndarray
    input_weight: np.ndarray
    input_bias: np.ndarray
    feature_version: str = ""
    model_hashes: tuple = ()
    depth_max_m: float = DEPTH_RANGE_M
    split_ids: dict | None = None
    split_groups: dict | None = None

    def predict_batch(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != len(self.feature_mean) or not np.isfinite(features).all():
            raise ValueError(f"Expected finite policy features of shape (N, {len(self.feature_mean)})")
        bounded = np.clip((_transform_features(features) - self.feature_mean) / self.feature_scale, -8, 8)
        if self.model == "mlp":
            bounded = np.tanh(bounded @ self.input_weight + self.input_bias)
        normalized_actions = bounded @ self.weight + self.bias
        actions = normalized_actions * self.action_scale + self.action_mean
        return np.clip(actions, self.action_min, self.action_max)

    def predict(self, features):
        """Return target speed in m/s and target yaw rate in rad/s for one frame."""
        speed, yaw_rate = self.predict_batch(np.asarray(features)[None])[0]
        return float(speed), float(yaw_rate)

    def save(self, path):
        arrays = {key: value for key, value in vars(self).items() if isinstance(value, np.ndarray)}
        _save_npz(path, **arrays, format_version=POLICY_FORMAT_VERSION, model=self.model,
                  feature_version=self.feature_version, model_hashes=np.asarray(self.model_hashes, dtype=str),
                  depth_max_m=self.depth_max_m, action_names=ACTION_NAMES, split_ids=json.dumps(self.split_ids or {}),
                  split_groups=json.dumps(self.split_groups or {}), world_schema=WORLD_SCHEMA)

    @classmethod
    def load(cls, path, *, perception=None):
        with np.load(path, allow_pickle=False) as data:
            if (data["format_version"].item() != POLICY_FORMAT_VERSION or data["world_schema"].item() != WORLD_SCHEMA
                    or tuple(data["action_names"]) != ACTION_NAMES):
                raise ValueError(f"Incompatible policy format, course, or action schema in {path}")
            keys = ("feature_mean", "feature_scale", "action_mean", "action_scale", "action_min", "action_max",
                    "weight", "bias", "input_weight", "input_bias")
            arrays = {key: data[key].astype(np.float32) for key in keys}
            policy = cls(str(data["model"].item()), **arrays, feature_version=str(data["feature_version"].item()),
                         model_hashes=tuple(data["model_hashes"].tolist()),
                         depth_max_m=float(data["depth_max_m"].item()) if "depth_max_m" in data else DEPTH_RANGE_M,
                         split_ids=json.loads(data["split_ids"].item()),
                         split_groups=json.loads(data["split_groups"].item()))
        dimension = policy.feature_mean.size
        hidden = policy.input_bias.size
        if (policy.model not in {"constant", "ridge", "mlp"} or not dimension
                or policy.feature_mean.shape != (dimension,) or policy.feature_scale.shape != (dimension,)
                or any(arrays[key].shape != (2,) for key in ("action_mean", "action_scale", "action_min", "action_max", "bias"))
                or policy.input_weight.shape != (dimension, hidden) or policy.input_bias.shape != (hidden,)
                or policy.weight.shape != (hidden if policy.model == "mlp" else dimension, 2)
                or policy.feature_version != FEATURE_VERSION or dimension != len(FEATURE_NAMES)
                or not 0 < policy.depth_max_m <= DEPTH_RANGE_M
                or not all(np.isfinite(value).all() for value in arrays.values())
                or (policy.feature_scale <= 0).any() or (policy.action_scale <= 0).any()
                or (policy.action_min > policy.action_max).any()):
            raise ValueError(f"Invalid model parameters in {path}")
        if perception is not None and (dimension != len(FEATURE_NAMES)
                                      or (policy.model_hashes and policy.model_hashes != perception.model_hashes)
                                      or policy.depth_max_m != perception.depth_max_m):
            raise ValueError(f"Perception features, depth limit or model weights do not match policy {path}")
        return policy


def _split_episodes(episodes, seed):
    groups = sorted({episode.scenario_group for episode in episodes})
    if len(groups) < 3:
        raise ValueError("Need at least three independent scenario groups for train, validation and test")
    np.random.default_rng(seed).shuffle(groups)
    holdout = max(1, round(0.2 * len(groups)))
    assignments = dict(zip(SPLITS, (groups[2 * holdout:], groups[:holdout], groups[holdout:2 * holdout])))
    return {split: [episode for episode in episodes if episode.scenario_group in members]
            for split, members in assignments.items()}


def _metrics(policy, episodes):
    per_episode = []
    sample_count = 0
    total_abs = np.zeros(2, dtype=np.float64)
    total_squared = np.zeros(2, dtype=np.float64)
    for episode in episodes:
        errors = np.asarray(policy.predict_batch(episode.features) - episode.actions, dtype=np.float64)
        absolute = np.abs(errors)
        squared = errors ** 2
        per_episode.append({"episode_id": episode.episode_id, "scenario_group": episode.scenario_group,
                            "layout": episode.layout, "samples": len(errors), "mae": absolute.mean(0).tolist(),
                            "rmse": np.sqrt(squared.mean(0)).tolist()})
        sample_count += len(errors)
        total_abs += absolute.sum(0)
        total_squared += squared.sum(0)
    macro_mae = np.mean([result["mae"] for result in per_episode], axis=0)
    return {"samples": sample_count, "episode_count": len(per_episode), "mae": (total_abs / sample_count).tolist(),
            "rmse": np.sqrt(total_squared / sample_count).tolist(), "episode_macro_mae": macro_mae.tolist(),
            "episode_macro_normalized_mae": float(np.mean(macro_mae / policy.action_scale)), "episodes": per_episode}


def _fit_mlp(features, actions, validation, params, *, seed, epochs, hidden, batch_size):
    rng = np.random.default_rng(seed)
    dimension = features.shape[1]
    layers = [rng.normal(0, np.sqrt(1 / dimension), (dimension, hidden)).astype(np.float32),
              np.zeros(hidden, np.float32), rng.normal(0, np.sqrt(1 / hidden), (hidden, 2)).astype(np.float32),
              np.zeros(2, np.float32)]
    first = [np.zeros_like(layer) for layer in layers]
    second = [np.zeros_like(layer) for layer in layers]
    best, best_loss, best_epoch, step = None, float("inf"), 0, 0
    for epoch in range(1, epochs + 1):
        selection = rng.permutation(len(features))
        for start in range(0, len(selection), batch_size):
            indexes = selection[start:start + batch_size]
            x, y = features[indexes], actions[indexes]
            w1, b1, w2, b2 = layers
            activation = np.tanh(x @ w1 + b1)
            residual = activation @ w2 + b2 - y
            output_gradient = (2 / residual.size) * residual
            hidden_gradient = (output_gradient @ w2.T) * (1 - activation ** 2)
            gradients = (x.T @ hidden_gradient + 0.0001 * w1, hidden_gradient.sum(0),
                         activation.T @ output_gradient + 0.0001 * w2, output_gradient.sum(0))
            step += 1
            for layer, avg, square, gradient in zip(layers, first, second, gradients):
                avg *= 0.9
                avg += 0.1 * gradient
                square *= 0.999
                square += 0.001 * gradient ** 2
                layer -= (0.002 / (1 - 0.9 ** step)) * avg / (np.sqrt(square / (1 - 0.999 ** step)) + 1e-8)
        candidate = Policy("mlp", **params, input_weight=w1, input_bias=b1, weight=w2, bias=b2)
        score = _metrics(candidate, validation)["episode_macro_normalized_mae"]
        if score < best_loss - 1e-6:
            best_loss, best_epoch = score, epoch
            best = Policy("mlp", **params, input_weight=w1.copy(), input_bias=b1.copy(), weight=w2.copy(), bias=b2.copy())
        if epoch - best_epoch >= 18:
            break
    return best, best_epoch, epoch


def train(input_path, output, *, seed=42, epochs=120, hidden=64, batch_size=128, model="auto"):
    """Split episodes before normalization, fit baselines and select a policy using validation."""
    if seed < 0 or min(epochs, hidden, batch_size) < 1 or model not in {"auto", "mlp", "ridge"}:
        raise ValueError("Invalid training parameters")
    episodes, version, hashes, depth_max_m = load_episodes(input_path)
    splits = _split_episodes(episodes, seed)
    x = _transform_features(np.concatenate([episode.features for episode in splits["train"]]))
    y = np.concatenate([episode.actions for episode in splits["train"]])
    params = dict(feature_mean=x.mean(0), feature_scale=np.maximum(x.std(0), np.float32(0.02)),
                  action_mean=y.mean(0), action_scale=np.maximum(y.std(0), np.array([0.10, 0.15], np.float32)),
                  action_min=y.min(0), action_max=y.max(0), feature_version=version, model_hashes=hashes,
                  depth_max_m=depth_max_m,
                  split_ids={name: [episode.episode_id for episode in values] for name, values in splits.items()},
                  split_groups={name: sorted({episode.scenario_group for episode in values})
                                for name, values in splits.items()})
    features = np.clip((x - params["feature_mean"]) / params["feature_scale"], -8, 8)
    actions = (y - params["action_mean"]) / params["action_scale"]
    dimension = features.shape[1]
    empty_hidden = dict(input_weight=np.empty((dimension, 0), np.float32), input_bias=np.empty(0, np.float32))
    constant = Policy("constant", **params, **empty_hidden, weight=np.zeros((dimension, 2), np.float32),
                      bias=np.zeros(2, np.float32))
    centered = np.asarray(features - features.mean(0), dtype=np.float64)
    covariance = centered.T @ centered / len(centered)
    regularized = covariance + 0.05 * np.eye(dimension)
    weight = np.linalg.solve(regularized, centered.T @ actions / len(centered)).astype(np.float32)
    ridge = Policy("ridge", **params, **empty_hidden, weight=weight, bias=actions.mean(0) - features.mean(0) @ weight)
    mlp, best_epoch, epochs_run = _fit_mlp(features, actions, splits["validation"], params, seed=seed,
                                          epochs=epochs, hidden=hidden, batch_size=batch_size)
    policies = {"constant": constant, "ridge": ridge, "mlp": mlp}
    validation = {name: _metrics(policy, splits["validation"]) for name, policy in policies.items()}
    selected = min(("ridge", "mlp"), key=lambda name: validation[name]["episode_macro_normalized_mae"]) if model == "auto" else model
    scores = {name: {"train": _metrics(policy, splits["train"]), "validation": validation[name],
                     "test": _metrics(policy, splits["test"])} for name, policy in policies.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, policy in policies.items():
        policy.save(output / f"{name}.npz")
    policies[selected].save(output / "policy.npz")
    report = {"action_names": ACTION_NAMES, "world_schema": WORLD_SCHEMA, "feature_version": version,
              "model_hashes": hashes, "depth_max_m": depth_max_m,
              "feature_preprocessing": {"square_root_color_occupancy": [name[1:] for name in SPARSE_COLOR_SUFFIXES]},
              "selected_policy": selected, "selection": model, "selection_metric": "validation episode-macro normalized MAE",
              "split_seed": seed, "splits": params["split_ids"], "scenario_groups": params["split_groups"],
              "action_normalization_scale": params["action_scale"].tolist(),
              "mlp_best_epoch": best_epoch, "mlp_epochs_run": epochs_run, "models": scores}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def evaluate(input_path, policy_path, split="test"):
    episodes, version, hashes, depth_max_m = load_episodes(input_path)
    policy = Policy.load(policy_path)
    if version != policy.feature_version or hashes != policy.model_hashes or depth_max_m != policy.depth_max_m:
        raise ValueError("Evaluation features do not match the policy model weights, depth limit or feature schema")
    known_groups = {group for values in policy.split_groups.values() for group in values}
    requested = (*SPLITS, "external") if split == "all" else (split,)
    scores = {}
    for name in requested:
        if name == "external":
            known_ids = {episode_id for values in policy.split_ids.values() for episode_id in values}
            selected = [episode for episode in episodes if episode.scenario_group not in known_groups
                        and episode.episode_id not in known_ids]
        else:
            selected = [episode for episode in episodes if episode.episode_id in policy.split_ids.get(name, ())
                        and episode.scenario_group in policy.split_groups.get(name, ())]
        if selected:
            scores[name] = _metrics(policy, selected)
    if not scores:
        raise ValueError(f"No episodes matching the policy's {split!r} split")
    return {"model": policy.model, "action_names": ACTION_NAMES, "world_schema": WORLD_SCHEMA, "splits": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract", help="Cache camera features from teacher rollout recordings")
    extraction.add_argument("--road-model", type=Path, default=ROAD_MODEL)
    extraction.add_argument("--cone-model", type=Path, default=CONE_MODEL)
    extraction.add_argument("--threads", type=int, default=2)
    extraction.add_argument("--workers", type=int, default=1, help="number of frames inferred concurrently")
    extraction.add_argument("--sample-step", type=int, default=1)
    extraction.add_argument("--max-samples-per-episode", type=int)
    extraction.add_argument("--depth-max-m", type=float, default=DEPTH_RANGE_M)
    fitting = commands.add_parser("train", help="Fit a policy and two baselines on disjoint episode groups")
    fitting.add_argument("--seed", type=int, default=42)
    fitting.add_argument("--epochs", type=int, default=120)
    fitting.add_argument("--hidden", type=int, default=64)
    fitting.add_argument("--batch-size", type=int, default=128)
    fitting.add_argument("--model", choices=("auto", "mlp", "ridge"), default="auto")
    scoring = commands.add_parser("evaluate", help="Score an existing policy on saved features")
    scoring.add_argument("--policy", type=Path, required=True)
    scoring.add_argument("--split", choices=(*SPLITS, "external", "all"), default="test")
    for command in commands.choices.values():
        command.add_argument("--input", type=Path, required=True)
    for command in (extraction, fitting):
        command.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    input_path = args.pop("input")
    if command == "extract":
        extract(input_path, **args)
    elif command == "train":
        report = train(input_path, **args)
        selected = report["selected_policy"]
        print(json.dumps({"selected_policy": selected, "test_mae": report["models"][selected]["test"]["mae"],
                          "metrics": str(args["output"] / "metrics.json")}, ensure_ascii=False))
    else:
        print(json.dumps(evaluate(input_path, args.pop("policy"), **args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
