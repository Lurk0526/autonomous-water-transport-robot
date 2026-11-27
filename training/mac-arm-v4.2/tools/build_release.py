"""Assemble the v4.2 offline Mac zip using the checksum-verified v4 runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FOLDER = 'mac-arm-v4.2'
NAME = 'Mac机械臂-v4.2-负载轨迹与腕摄复测.zip'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def project_files():
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if any(part in {'__pycache__', '.venv', 'outputs', 'runtime', 'payload', 'source'} for part in relative.parts):
            continue
        if path.is_file() and not path.is_symlink() and path.suffix not in {'.pyc', '.pyo'}:
            yield relative.as_posix(), path


def build(base, destination):
    if (destination / NAME).exists():
        raise FileExistsError(destination / NAME)
    source = dict(project_files())
    # Reuse the earlier SHA-256 inventory. Verify every incorporated byte.
    base_manifest = json.loads((base / 'payload-manifest.json').read_text())
    for row in base_manifest:
        name = row['path']
        if not (name.startswith(('payload/', 'docs/dependency-notices/', 'source/bowl-reference/'))
                or name in {'source/机械臂带爪子.STEP','source/source-manifest.json','source/arm-camera-mount.jpg'}):
            continue
        path = base / name
        if not path.is_file() or sha256(path) != row['sha256']:
            raise ValueError(f'v4 asset differs from its release inventory: {name}')
        if name in source:
            raise ValueError(f'Duplicate source asset: {name}')
        source[name] = path
    if sum(name.endswith('.whl') for name in source) != 60 or 'payload/python-macos-arm64.tar.gz' not in source:
        raise ValueError('The complete v4 Mac runtime and all 60 wheels are required')
    if sha256(ROOT/'requirements-mac.lock.txt') != sha256(base/'requirements-mac.lock.txt'):
        raise ValueError('Dependency lock differs from the v4 offline environment')

    destination.mkdir(parents=True, exist_ok=True)
    manifest = [{'path':name, 'bytes':path.stat().st_size, 'sha256':sha256(path)} for name,path in sorted(source.items())]
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    checksums = {row['path']:row['sha256'] for row in manifest}
    checksums['payload-manifest.json'] = hashlib.sha256(manifest_bytes).hexdigest()
    sums_bytes = ''.join(f'{digest}  {name}\n' for name,digest in checksums.items()).encode()
    checksums['SHA256SUMS.txt'] = hashlib.sha256(sums_bytes).hexdigest()
    with tempfile.TemporaryDirectory(dir=destination) as temp:
        archive = Path(temp) / NAME
        with ZipFile(archive, 'x', ZIP_DEFLATED, compresslevel=5, allowZip64=True) as z:
            for name,path in sorted(source.items()):
                compression = ZIP_STORED if path.suffix in {'.whl','.zip','.gz','.png','.jpg'} else ZIP_DEFLATED
                z.write(path, FOLDER + '/' + name, compress_type=compression)
            z.writestr(FOLDER+'/payload-manifest.json', manifest_bytes)
            z.writestr(FOLDER+'/SHA256SUMS.txt', sums_bytes)
        with ZipFile(archive) as z:
            if len(z.infolist()) != len(checksums):
                raise ValueError('Unexpected archive member count')
            for name, digest in checksums.items():
                # Reading each member to EOF also verifies the ZIP CRC.
                with z.open(FOLDER+'/'+name) as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                        raise ValueError(f'Archive verification failed for {name}')
        digest = sha256(archive)
        size = archive.stat().st_size
        os.link(archive, destination/NAME)
    (destination/(NAME+'.sha256')).write_text(f'{digest}  {NAME}\n')
    proof = {'bytes':size, 'sha256':digest, 'verified_zip_members':len(checksums), 'wheel_count':60}
    (destination/'archive-verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base.resolve(), args.destination.resolve()), indent=2))
