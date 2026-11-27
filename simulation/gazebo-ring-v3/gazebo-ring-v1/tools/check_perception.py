"""Replay three recorded Gazebo frames against the feature extraction used for training."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ring_training.perception import FEATURE_VERSION, Perception
from ring_training.world import WORLD_SCHEMA


def main():
    pipeline = Perception()
    with np.load(ROOT/'validation/rgbd-parity.npz',allow_pickle=False) as z:
        if (tuple(pipeline.model_hashes) != tuple(z['model_hashes']) or
            z['feature_version'].item() != FEATURE_VERSION or z['world_schema'].item() != WORLD_SCHEMA):
            raise ValueError('Observation or model version mismatch')
        reconstructed = np.stack([pipeline.features(rgb,depth) for rgb,depth in zip(z['rgb'],z['depth'],strict=True)])
        difference = np.abs(reconstructed-z['features'])
        result = {"cases":z['sessions'].tolist(), "feature_dimension":int(difference.shape[1]),
                  "max_abs_feature_difference":float(difference.max()),
                  "mean_abs_feature_difference":float(difference.mean()),
                  "passed":bool(np.isfinite(difference).all() and difference.max()<0.025 and difference.mean()<0.001)}
        print(json.dumps(result,indent=2))
        if not result["passed"]:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
