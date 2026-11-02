#!/usr/bin/env python3
"""Check rotation round trips and startup rejection of corrupted deployment data."""
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.road_inference import RoadPredictor


def main():
    deployment = ROOT/'work/models/road/road-4060-v1'
    predictor = RoadPredictor(deployment, 'tensorrt', 0)
    rgb = np.asarray(Image.open(ROOT/'work/road_perception/setup/camera/rgb.jpg').convert('RGB'))
    results = []
    for k in [1, 2, 3]:
        predictor.rotation = k
        labels, confidence, _ = predictor.predict(rgb)
        predictor.rotation = 0
        reference, reference_confidence, _ = predictor.predict(np.ascontiguousarray(np.rot90(rgb, k)))
        expected_labels, expected_confidence = np.rot90(reference, -k), np.rot90(reference_confidence, -k)
        results.append({'rotation_degrees': k*90, 'same_native_shape': labels.shape == rgb.shape[:2],
                        'label_agreement': float((labels == expected_labels).mean()),
                        'max_confidence_error': float(np.abs(confidence-expected_confidence).max())})
    with tempfile.TemporaryDirectory(prefix='road-integrity-', dir=ROOT/'work/road_perception') as folder:
        target = Path(folder)
        (target/'model').mkdir()
        (target/'model/model.safetensors').symlink_to(deployment/'model/model.safetensors')
        shutil.copy2(deployment/'model/config.json', target/'model/config.json')
        shutil.copy2(deployment/'manifest.json', target/'manifest.json')
        (target/'labels.json').write_text('{"tampered":true}')
        rejected = False
        try:
            RoadPredictor(target)
        except ValueError as exc:
            rejected = 'checksum mismatch' in str(exc)
    report = {'rotation_round_trips': results, 'tampered_labels_rejected': rejected,
              'passed': rejected and all(r['same_native_shape'] and r['label_agreement'] == 1 and r['max_confidence_error'] < 1e-5 for r in results)}
    output = ROOT/'work/road_perception/deployment-check.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return not report['passed']


if __name__ == '__main__':
    raise SystemExit(main())
