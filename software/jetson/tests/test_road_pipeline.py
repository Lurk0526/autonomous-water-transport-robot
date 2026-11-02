"""Checks for label semantics, geometry and validation leakage."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.road_model import LABELS, ROOT, letterbox, metrics


class RoadPipelineTests(unittest.TestCase):
    def test_water_and_vegetation_are_distinct_from_pavement(self):
        for native in ('3', '4', '6', '19'):
            self.assertNotEqual(LABELS['rugd_id_to_target'][native], 0)
        self.assertEqual(LABELS['rugd_id_to_target']['0'], 255)
        self.assertEqual(LABELS['camvid_id_to_target']['11'], 255)

    def test_letterbox_preserves_aspect_and_ignores_padding(self):
        image = Image.new('RGB', (200, 100), 'white')
        mask = Image.fromarray(np.tile(np.array([0, 3], np.uint8), (100, 100)))
        resized, label, box = letterbox(image, mask, (256, 320))
        self.assertEqual(box, (0, 48, 320, 160))
        pixels = np.array(label)
        self.assertTrue((pixels[:48] == 255).all())
        self.assertTrue((pixels[208:] == 255).all())
        self.assertEqual(set(np.unique(pixels[48:208])), {0, 3})
        self.assertEqual(resized.size, (320, 256))

    def test_unsafe_pavement_predictions_are_counted(self):
        confusion = np.eye(7, dtype=np.int64) * 100
        confusion[3, 0] = 100
        report = metrics(confusion)
        self.assertAlmostEqual(report['paved_precision'], 0.5)
        self.assertAlmostEqual(report['non_paved_predicted_paved_rate'], 100 / 700)
        self.assertAlmostEqual(report['per_class_IoU']['water'], 0.5)

    def test_missing_class_is_not_reported_as_perfect(self):
        confusion = np.zeros((7, 7), np.int64)
        confusion[0, 0] = 100
        report = metrics(confusion)
        self.assertIsNone(report['per_class_IoU']['water'])
        self.assertEqual(report['ground_truth_present_class_mIoU'], 1.0)

    @unittest.skipUnless(
        all((ROOT / f'datasets/public_road/prepared/camvid-{split}.jsonl').is_file()
            for split in ('train', 'val', 'test')),
        'Full road dataset is stored separately; restore it to run leakage checks')
    def test_manifests_do_not_leak_videos_or_images(self):
        seen_sequences, seen_hashes = {}, {}
        for split in ('train', 'val', 'test'):
            path = ROOT / f'datasets/public_road/prepared/camvid-{split}.jsonl'
            for line in path.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                key = (record['dataset'], record['sequence'])
                self.assertEqual(seen_sequences.setdefault(key, split), split)
                self.assertEqual(seen_hashes.setdefault(record['sha256'], split), split)
                self.assertTrue((ROOT / record['image']).is_file())
                self.assertTrue((ROOT / record['mask']).is_file())


if __name__ == '__main__': unittest.main()
