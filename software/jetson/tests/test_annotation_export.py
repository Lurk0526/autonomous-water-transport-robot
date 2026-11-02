"""Checks for label normalization, explicit negatives, leakage and preservation."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.export_bowl_dataset import export_dataset, inspect_source, validate_polygon


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.images = {}
        for i, split in enumerate(('train', 'val', 'test')):
            image = self.source / split / (split + '-session') / 'bowl.png'
            image.parent.mkdir(parents=True)
            Image.new('RGB', (100, 80), (i * 80, 10, 20)).save(image)
            self.images[split] = image
            value = {'imagePath': image.name, 'imageWidth': 100, 'imageHeight': 80,
                     'flags': {}, 'shapes': [{'label': 'target_bowl', 'shape_type': 'polygon',
                                             'points': [[10, 8], [90, 8], [90, 72], [10, 72]]}]}
            image.with_suffix('.json').write_text(json.dumps(value))

    def tearDown(self):
        self.temp.cleanup()

    def change(self, split, edit):
        path = self.images[split].with_suffix('.json')
        value = json.loads(path.read_text())
        edit(value)
        path.write_text(json.dumps(value))

    def test_export_normalizes_and_keeps_same_named_images(self):
        output = self.root / 'yolo'
        summary = export_dataset(self.source, output)
        self.assertEqual(summary['train']['images'], 1)
        labels = list((output / 'labels').rglob('*.txt'))
        self.assertEqual(len(labels), 3)
        self.assertEqual(len({p.name for p in labels}), 3)
        for label in labels:
            numbers = list(map(float, label.read_text().split()))
            self.assertEqual(numbers, [0, .1, .1, .9, .1, .9, .9, .1, .9])
        self.assertIn('0: target_bowl', (output / 'data.yaml').read_text())
        for image in self.images.values():
            self.assertTrue(image.exists())
            self.assertTrue(image.with_suffix('.json').exists())

    def test_unreviewed_image_not_treated_as_background(self):
        image = self.source / 'train' / 'extra.png'
        Image.new('RGB', (10, 10)).save(image)
        with self.assertRaisesRegex(ValueError, '尚无 JSON'):
            export_dataset(self.source, self.root / 'yolo')
        self.assertFalse((self.root / 'yolo').exists())

    def test_explicit_negative_exported_as_empty_label(self):
        image = self.source / 'train' / 'empty.png'
        Image.new('RGB', (10, 10)).save(image)
        value = {'imagePath': image.name, 'imageWidth': 10, 'imageHeight': 10,
                 'flags': {'negative': True}, 'shapes': []}
        image.with_suffix('.json').write_text(json.dumps(value))
        output = self.root / 'yolo'
        summary = export_dataset(self.source, output)
        self.assertEqual(summary['train']['negative'], 1)
        negative_labels = [p for p in (output / 'labels/train').glob('empty-*.txt')]
        self.assertEqual(negative_labels[0].read_text(), '')
        value['flags'] = {}
        image.with_suffix('.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, '空标注需勾选'):
            inspect_source(self.source)

    def test_duplicate_across_splits_rejected(self):
        self.images['val'].write_bytes(self.images['train'].read_bytes())
        with self.assertRaisesRegex(ValueError, '数据泄漏'):
            inspect_source(self.source)

    def test_bad_geometry_dimensions_and_labels(self):
        with self.assertRaises(ValueError):
            validate_polygon([[10, 10], [90, 70], [90, 10], [10, 70], [50, 75]], 100, 80)
        self.change('train', lambda value: value.update(imageWidth=101))
        with self.assertRaisesRegex(ValueError, '尺寸'):
            inspect_source(self.source)
        self.change('train', lambda value: value.update(imageWidth=100))
        self.change('train', lambda value: value['shapes'][0].update(label='bowl'))
        with self.assertRaisesRegex(ValueError, '只接受 target_bowl'):
            inspect_source(self.source)

    def test_existing_output_preserved(self):
        output = self.root / 'yolo'
        output.mkdir()
        marker = output / 'important.txt'
        marker.write_text('keep')
        with self.assertRaisesRegex(ValueError, '输出目录已存在'):
            export_dataset(self.source, output)
        self.assertEqual(marker.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
