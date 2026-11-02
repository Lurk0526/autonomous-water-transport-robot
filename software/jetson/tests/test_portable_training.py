"""Portable dataset paths and Windows UTF-8 source annotations."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from tools.bowl_model import prepare_data
from tools.export_bowl_dataset import inspect_source


class PortableTests(unittest.TestCase):
    def test_dataset_rebased_after_transfer_without_changing_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = {'path': 'C:/old-machine/dataset', 'train': 'images/train',
                    'val': 'images/val', 'test': 'images/test', 'names': {0: 'target_bowl'}}
            source = root / 'dataset/data.yaml'
            for split in ('train', 'val', 'test'):
                destination = root / 'dataset/images' / split
                destination.mkdir(parents=True)
                Image.new('RGB', (16, 16)).save(destination / 'a.png')
            source.write_text(yaml.safe_dump(data), encoding='utf-8-sig')
            original = source.read_bytes()
            result = prepare_data(source, root / 'runtime')
            runtime = yaml.safe_load(result.read_text(encoding='utf-8'))
            self.assertEqual(runtime['path'], source.parent.resolve().as_posix())
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(runtime['test'], 'images/test')

    def test_wrong_class_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'data.yaml'
            source.write_text('names: [bowl]\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'single target_bowl'):
                prepare_data(source)

    def test_unicode_image_names_and_bom_annotations(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            for i, split in enumerate(('train', 'val', 'test')):
                image = source / split / '黑色碗.png'
                image.parent.mkdir(parents=True)
                Image.new('RGB', (16, 16), (i * 70, 30, 40)).save(image)
                annotation = {'imagePath': image.name, 'imageWidth': 16, 'imageHeight': 16,
                              'flags': {}, 'shapes': [{'shape_type': 'polygon', 'label': 'target_bowl',
                                                       'points': [[1, 1], [15, 1], [15, 15], [1, 15]]}]}
                image.with_suffix('.json').write_text(json.dumps(annotation, ensure_ascii=False), encoding='utf-8-sig')
            rows = inspect_source(source)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all('黑色碗' in row['relative'] for row in rows))


if __name__ == '__main__':
    unittest.main()
