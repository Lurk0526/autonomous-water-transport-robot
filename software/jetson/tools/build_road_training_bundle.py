#!/usr/bin/python3
"""Build a standalone RTX 4060 bundle, with data and portable relative paths."""
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'water-cart-road-4060-20261029'
STAGE = ROOT / 'work/road_training' / NAME
OUTPUT = ROOT / 'deliverables/road-training-4060-20261029'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if STAGE.exists(): raise RuntimeError('Staging directory already exists; preserve it before rebuilding')
    STAGE.mkdir(parents=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    included = {}
    known_hashes = {}

    def add(relative, dest=None):
        source = ROOT / relative
        target_relative = Path(dest or relative)
        target = STAGE / target_relative
        if target_relative.as_posix() in included: return
        if not source.is_file(): raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target_relative.parts[0] in ('reports', 'tools', 'vision', 'tests', 'config', 'docs') or source.stat().st_size < 1024 * 1024:
            # Snapshot small/editable files; only immutable bulk assets use hardlinks.
            shutil.copy2(source, target)
        else:
            try: os.link(source, target)
            except OSError: shutil.copy2(source, target)
        included[target_relative.as_posix()] = target

    code = ['vision/road_model.py', 'tools/train_road_model.py', 'tools/preview_road_model.py',
            'tools/evaluate_road_model.py', 'tools/export_road_model.py', 'tools/prepare_road_datasets.py',
            'tools/check_training_bundle.py', 'tests/test_road_pipeline.py',
            'config/road-labels.json', 'config/road-train-4060.yaml', 'config/road-train-4060-smoke.yaml',
            'config/road-train-warmup.yaml', 'requirements-road-4060.txt',
            'INSTALL_4060_WINDOWS.bat', 'install_4060_linux.sh', 'docs/road-training-plan.md',
            'docs/ros2-first-steps.md', 'docs/RTX4060训练指南.md']
    for path in code: add(path)
    add('docs/RTX4060训练指南.md', 'RTX4060训练指南.md')
    (STAGE / 'vision/__init__.py').write_text('"""Portable outdoor road training package."""\n')
    included['vision/__init__.py'] = STAGE / 'vision/__init__.py'
    for prefix in ('camvid', 'rugd', 'camvid-rugd'):
        audit = f'datasets/public_road/prepared/{prefix}-audit.json'
        add(audit)
        for split in ('train', 'val', 'test'):
            manifest = f'datasets/public_road/prepared/{prefix}-{split}.jsonl'
            if (ROOT / manifest).is_file(): add(manifest)
    for split in ('train', 'val', 'test'):
        manifest = ROOT / f'datasets/public_road/prepared/camvid-rugd-{split}.jsonl'
        for line in manifest.read_text().splitlines():
            record = json.loads(line)
            add(record['image']); add(record['mask'])
            known_hashes[record['image']] = record['sha256']
    # Include original annotations too, so users can re-audit/remap the public data.
    for folder in [ROOT / 'datasets/public_road/rugd/RUGD_annotations',
                   ROOT / 'datasets/public_road/camvid/SegNet-Tutorial-master/CamVid']:
        for file in folder.rglob('*'):
            if file.is_file(): add(file.relative_to(ROOT))
    for relative in ['work/models/road/segformer-b0-cityscapes',
                     'work/road_training/runs/camvid-warmup/best/model']:
        for file in (ROOT / relative).iterdir():
            if file.is_file() and file.name != 'pytorch_model.bin': add(file.relative_to(ROOT))
    for path in ['work/models/road/yolo11n.pt', 'work/models/yolo11n-seg.pt',
                 'work/models/target-bowl-20261021/best.pt']:
        add(path, 'reference_models/' + Path(path).parent.name + '/' + Path(path).name)
    reports = ['camvid-warmup-test.json', 'pipeline-tests.log', '4060-config-smoke.log',
               'rugd-extraction.json', 'assets-manifest.json', 'python-freeze.txt']
    for path in reports:
        source = ROOT / 'work/road_training' / path
        if source.is_file(): add(source.relative_to(ROOT), 'reports/' + path)
    add('work/road_training/runs/camvid-warmup/result.json', 'reports/camvid-warmup-result.json')
    add('work/road_training/runs/camvid-warmup/history.jsonl', 'reports/camvid-warmup-history.jsonl')
    add('work/road_training/runs/road-4060-smoke/result.json', 'reports/4060-config-smoke-result.json')
    for folder in ['preview-pretrained', 'preview-warmup']:
        for file in (ROOT / 'work/road_training' / folder).glob('*'):
            if file.is_file(): add(file.relative_to(ROOT), 'reports/' + folder + '/' + file.name)
    for file in (ROOT / 'work/road_training/sources').glob('*'):
        if file.is_file(): add(file.relative_to(ROOT), 'sources/' + file.name)
    for path in ['camvid_readme.txt', 'segformer_license.txt']:
        if not (STAGE / 'sources' / path).is_file(): raise RuntimeError('Missing source/license record: ' + path)
    add('work/road_training/downloads/RUGD_frames-with-annotations.zip.receipt.json', 'sources/rugd-download-receipt.json')
    source_doc = '''# Sources and use conditions\n\nThis is a local research/educational training bundle for the user's water-cart project.\n\n- SegFormer: https://github.com/NVlabs/SegFormer ; https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024 ; revision 21b3847fae21ddee674abd31129307b6a1235bd9. NVIDIA non-commercial research/evaluation license is saved in sources/segformer_license.txt. Original LFS weight SHA256 verified: 027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c. Safetensors is a local serialization of these weights.\n- CamVid: https://github.com/alexgkendall/SegNet-Tutorial ; commit bb68b64e72cb004b3ed927fee0f5624dd8097498. Repository states CC BY-NC 4.0 personal/research use; see sources/camvid_readme.txt. Original data authors retain their rights.\n- RUGD: GitHub discovery https://github.com/dataset-ninja/rugd/blob/main/DOWNLOAD.md ; author origin http://rugd.vision/ ; HTTPS transport to the same object via https://s3.amazonaws.com/rugd.vision/data/RUGD_frames-with-annotations.zip (matched object size and ETag). The original website does not give an explicit general-purpose license grant. Preserve attribution and original terms; public download does not imply unrestricted commercial use or redistribution. Author paper is saved in sources/rugd_paper.pdf.\n- YOLO11 reference weights: https://github.com/ultralytics/assets/releases/tag/v8.3.0 ; Ultralytics license terms apply. These are optional reference models; road training uses SegFormer. Existing target-bowl weights belong to this project and were copied without retraining.\n\nThis bundle contains real public images/annotations and pretrained models. The five-epoch CamVid warmup and short training checks are not a site-specific driving certification. See reports for validation scope.\n\nDataset hashes are observed local integrity records. RUGD ZIP CRC was verified while extracting all members. They are not represented as an independently published author SHA256.\n'''
    (STAGE / 'SOURCES.md').write_text(source_doc); included['SOURCES.md'] = STAGE / 'SOURCES.md'
    readme = '# RTX 4060 training bundle\n\nStart with [RTX4060训练指南.md](RTX4060训练指南.md).\n\nWindows: Python 3.11, then INSTALL_4060_WINDOWS.bat.\nLinux: Python 3.11, then bash install_4060_linux.sh.\n\nAfter environment setup:\n\n```bash\npython tools/check_training_bundle.py --full --gpu\npython tools/train_road_model.py --config config/road-train-4060-smoke.yaml\npython tools/train_road_model.py --config config/road-train-4060.yaml\n```\n\nNo ROS 2, Jetson drivers, ARM64 environment, or TensorRT engine is required for training on the PC.\n'
    (STAGE / 'README.md').write_text(readme); included['README.md'] = STAGE / 'README.md'
    file_list = []
    for i, (relative, file) in enumerate(sorted(included.items()), 1):
        file_list.append({'path': relative, 'bytes': file.stat().st_size,
                          'sha256': known_hashes.get(relative) or sha(file)})
        if i % 3000 == 0: print('Manifest files', i, flush=True)
    (STAGE / 'BUNDLE_FILES.json').write_text(json.dumps({'name': NAME, 'files': file_list}, indent=2) + '\n')
    for filename in ['RTX4060训练指南.md', 'README.md', 'SOURCES.md']:
        shutil.copy2(STAGE / filename, OUTPUT / filename)
    archive = OUTPUT / (NAME + '.zip')
    temp = Path(str(archive) + '.part')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for i, file in enumerate(sorted(STAGE.rglob('*')), 1):
            if file.is_file():
                bundle.write(file, NAME + '/' + file.relative_to(STAGE).as_posix())
            if i % 5000 == 0: print('Packaged entries', i, flush=True)
    temp.replace(archive)
    digest = sha(archive)
    (OUTPUT / 'SHA256SUMS.txt').write_text(digest + '  ' + archive.name + '\n')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': digest, 'files': len(file_list) + 1}, indent=2), flush=True)


if __name__ == '__main__': main()
