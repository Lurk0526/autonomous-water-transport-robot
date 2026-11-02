#!/usr/bin/env python3
"""Build a source-and-bootstrap Windows ZIP, excluding private data and venvs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'Output exists: {output}')
    needed = ['labelme-7.8.1-py3-none-any.whl', 'labelme-7.8.1.tar.gz',
              'ultralytics-8.4.178-py3-none-any.whl', 'ultralytics-8.4.178.tar.gz']
    for name in needed:
        if not (args.third_party_dir / name).is_file():
            raise ValueError(f'Missing upstream distribution: {name}')
    if not args.model.is_file():
        raise ValueError('Pretrained model is missing')
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = ROOT / 'work/windows-package'
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as folder:
        package = Path(folder) / 'jibot-bowl-windows'
        package.mkdir()
        for directory in ('annotation', 'docs', 'launch', 'tests', 'tools', 'vision', 'windows'):
            for source in (ROOT / directory).rglob('*'):
                if not source.is_file() or '__pycache__' in source.parts:
                    continue
                if source.suffix.lower() not in {'.py', '.md', '.txt', '.yaml', '.yml', '.cmd', '.sh'}:
                    continue
                destination = package / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        for source in ROOT.iterdir():
            if source.is_file() and (source.suffix in ('.md', '.sh') or
                                     source.name.startswith(('requirements-', 'constraints-')) or
                                     source.name in ('.gitignore', 'ros-packages.txt')):
                shutil.copy2(source, package / source.name)
        # Normalize CMD line endings, leaving Python/Markdown as UTF-8.
        for cmd in (package / 'windows').glob('*.cmd'):
            text = cmd.read_text(encoding='utf-8').replace('\r\n', '\n')
            cmd.write_bytes(text.replace('\n', '\r\n').encode('ascii'))
        (package / 'START_HERE_WINDOWS.txt').write_text(
            (ROOT / 'README_WINDOWS.md').read_text(encoding='utf-8'), encoding='utf-8-sig')
        for split in ('train', 'val', 'test'):
            path = package / 'datasets/target_bowl/source' / split
            path.mkdir(parents=True)
            (path / 'PUT_IMAGES_AND_JSON_HERE.txt').write_text(
                'Put complete capture sessions here, with original images and matching Labelme JSON.\n'
                'Keep adjacent video frames in the same split.\n', encoding='utf-8')
        (package / 'pretrained').mkdir()
        shutil.copy2(args.model, package / 'pretrained/yolo11n-seg.pt')
        third_party = package / 'third_party'
        third_party.mkdir()
        for name in needed:
            shutil.copy2(args.third_party_dir / name, third_party / name)
        for name in ('labelme-7.8.1.tar.gz', 'ultralytics-8.4.178.tar.gz'):
            with tarfile.open(third_party / name) as archive:
                prefix = name.removesuffix('.tar.gz')
                license_data = archive.extractfile(prefix + '/LICENSE').read()
                (third_party / (prefix + '-LICENSE.txt')).write_bytes(license_data)
        notice = ('Upstream distributions are included unchanged.\n'
                  'Labelme 7.8.1: https://pypi.org/project/labelme/7.8.1/ (GPL-3.0).\n'
                  'Ultralytics 8.4.178: https://pypi.org/project/ultralytics/8.4.178/ (AGPL-3.0).\n'
                  'YOLO11 pretrained weights: https://github.com/ultralytics/assets/releases/tag/v8.3.0\n'
                  'Keep the corresponding license notices when redistributing.\n'
                  'Full Windows platform dependencies are downloaded by the setup scripts.\n')
        (third_party / 'THIRD_PARTY_NOTICES.txt').write_text(notice, encoding='utf-8')
        reference = package / 'reference_firmware'
        reference.mkdir()
        if args.firmware:
            shutil.copy2(args.firmware, reference / args.firmware.name)
        examples = Path('/home/nvidia/Desktop/002-开源机械臂Jibot1（STM32）/003-源码例程/06-视觉OpenMV拓展')
        if examples.exists():
            for source in examples.rglob('*.py'):
                destination = reference / 'OpenMV' / source.relative_to(examples)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        if (ROOT / 'work/openmv/main-before-streaming.py').is_file():
            shutil.copy2(ROOT / 'work/openmv/main-before-streaming.py', reference / 'OpenMV-main-original.py')
        (reference / 'README.txt').write_text(
            'Manufacturer source archive and OpenMV scripts supplied with the robot.\n'
            'Reference only. No firmware flashing is required for Windows training.\n', encoding='utf-8')
        (package / 'PACKAGE_VERIFICATION.json').write_text(json.dumps({
            'date': '2026-10-19', 'target': 'Windows 11 x64, Python 3.12, RTX 4060 8GB',
            'host': 'Jetson Linux aarch64', 'windows_runtime_tested': False,
            'python_checks': '16 unit tests passed',
            'synthetic_pipeline': 'Labelme export, 1 CPU epoch, independent test validation, predict and ONNX export',
            'pretrained_model_sha256': sha256(args.model),
            'real_target_bowl_model_trained': False,
            'raw_images_and_annotations_included': False,
            'online_install_required': True,
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        manifest = {p.relative_to(package).as_posix(): sha256(p)
                    for p in sorted(package.rglob('*')) if p.is_file()}
        (package / 'MANIFEST_SHA256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        pending = output.with_suffix('.partial.zip')
        try:
            with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for file in sorted(package.rglob('*')):
                    if file.is_file():
                        archive.write(file, file.relative_to(package.parent).as_posix())
            with zipfile.ZipFile(pending) as archive:
                assert archive.testzip() is None
                for name in archive.namelist():
                    assert not name.startswith('/') and '..' not in Path(name).parts
                    assert '/.venv' not in name and '/__pycache__/' not in name
                for name, checksum in manifest.items():
                    assert hashlib.sha256(archive.read('jibot-bowl-windows/' + name)).hexdigest() == checksum
            pending.rename(output)
        finally:
            if pending.exists():
                pending.unlink()
    checksum = sha256(output)
    output.with_suffix('.zip.sha256').write_text(f'{checksum}  {output.name}\n', encoding='ascii')
    print(json.dumps({'file': str(output), 'bytes': output.stat().st_size,
                      'files': len(manifest) + 1, 'sha256': checksum}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party-dir', type=Path, default=ROOT / 'work/windows-downloads')
    parser.add_argument('--model', type=Path, default=ROOT / 'work/models/yolo11n-seg.pt')
    parser.add_argument('--firmware', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'deliverables/Jibot-Bowl-Windows-20261019.zip')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
