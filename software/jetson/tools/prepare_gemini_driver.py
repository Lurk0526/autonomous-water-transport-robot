#!/usr/bin/python3
"""Rebuild the isolated Jazzy driver from the supplied Gemini MAX archive."""
import argparse
import hashlib
import json
import stat
import subprocess
import zipfile
from pathlib import Path


def transformed_source(relative, data):
    if Path(relative).suffix not in ('.cpp', '.h', '.hpp', '.xml', '.txt', '.py', '.sh', '.cmake'):
        return data
    try:
        text = data.decode()
    except UnicodeDecodeError:
        return data
    text = text.replace('orbbec_camera_msgs', 'gemini_camera_msgs').replace('orbbec_camera', 'gemini_camera')
    text = text.replace('cv_bridge/cv_bridge.h', 'cv_bridge/cv_bridge.hpp')
    text = text.replace('set(CMAKE_BUILD_TYPE "Debug")', 'set(CMAKE_BUILD_TYPE "Release")')
    return text.replace('-Wpedantic -Werror', '-Wpedantic').encode()


def renamed_path(relative):
    names = {'orbbec_camera_msgs': 'gemini_camera_msgs', 'orbbec_camera': 'gemini_camera'}
    return '/'.join(names.get(part, part) for part in Path(relative).parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path,
                        default=Path('/home/nvidia/Desktop/Gemini MAX深度相机/ROS2/ROS2源码/orbbec_ws.zip'))
    parser.add_argument('--workspace', type=Path, default=Path('/home/nvidia/gemini_ws'))
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if not str(workspace).isascii() or any(c.isspace() for c in str(workspace)):
        parser.error('ROSIDL requires an ASCII workspace path without spaces')
    target = workspace / 'src/vendor_gemini_legacy'
    if target.exists():
        parser.error(f'{target} already exists; preserve it and select a new workspace to reproduce')
    prefix = 'orbbec_ws/src/OrbbecSDK_ROS2/'
    with zipfile.ZipFile(args.archive) as archive:
        for entry in archive.infolist():
            if not entry.filename.startswith(prefix) or entry.is_dir():
                continue
            relative = entry.filename[len(prefix):]
            if not relative.startswith(('orbbec_camera/', 'orbbec_camera_msgs/')):
                continue
            if '/SDK/lib/' in relative and '/SDK/lib/arm64/' not in relative:
                continue
            if '/SDK/lib/arm64/libudev' in relative or '/.git/' in relative:
                continue
            if '..' in Path(relative).parts:
                raise ValueError('Unexpected archive path')
            output = target / renamed_path(relative)
            output.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(entry)
            if stat.S_ISLNK(entry.external_attr >> 16):
                output.symlink_to(data.decode())
            else:
                output.write_bytes(transformed_source(relative, data))
    patch = Path(__file__).resolve().parents[1] / 'drivers/gemini-max-jazzy.patch'
    subprocess.run(['patch', '-p1', '-i', str(patch)], cwd=target, check=True)
    with args.archive.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    provenance = {'archive': str(args.archive), 'sha256': digest, 'sdk': '1.5.8 arm64',
                  'packages': ['gemini_camera', 'gemini_camera_msgs'], 'patch': str(patch)}
    (workspace / 'source-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if args.build:
        subprocess.run(['bash', '-c',
                        'source /opt/ros/jazzy/setup.bash && '
                        'MAKEFLAGS=-j2 colcon build --packages-select gemini_camera_msgs gemini_camera '
                        '--executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF'],
                       cwd=workspace, check=True)
    print(workspace)


if __name__ == '__main__':
    main()
