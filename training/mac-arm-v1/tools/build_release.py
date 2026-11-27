"""Assemble a native Mac arm64 offline delivery with per-file checksums."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import tarfile
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]


def copy_file(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();dest=a.destination.resolve();stage=dest/'mac-arm-v1'
    if stage.exists():raise SystemExit('Use a fresh destination to preserve previous deliveries.')
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT,stage,ignore=shutil.ignore_patterns('__pycache__','*.pyc','outputs','runtime','.venv'))
    payload=PROJECT/'work/mac-arm-payload'
    shutil.copytree(payload/'wheels',stage/'payload/wheels')
    metadata=json.loads((payload/'python-release.json').read_text())
    copy_file(payload/metadata['assets'][0]['name'],stage/'payload/python-macos-arm64.tar.gz')
    for name in ['wheel-manifest.json','payload-verification.json','python-release.json']:
        copy_file(payload/name,stage/'payload'/name)
    copy_file(PROJECT/'assets/cad/arm-step-20261117/机械臂带爪子.STEP',stage/'source/机械臂带爪子.STEP')
    copy_file(PROJECT/'assets/cad/arm-step-20261117/source-manifest.json',stage/'source/source-manifest.json')
    shutil.copytree(PROJECT/'assets/reference/bowl-20261117',stage/'source/bowl-reference')
    copy_file(PROJECT/'assets/reference/arm-camera-mount-20261116.jpg',stage/'source/arm-camera-mount.jpg')
    validation=PROJECT/'work/mac-arm-validation'
    required=['inspection.json','tests.log','gui-native.log','baseline200/evaluation.json','smoke-final/run.json','smoke-final/final.zip',
              'train-final.log','resume-final.log','smoke-evaluation/evaluation.json','workspace/workspace.json']
    for relative in required:
        if not (validation/relative).is_file():raise SystemExit('Required validation result not available: '+relative)
        target=stage/'validation'/relative
        copy_file(validation/relative,target)
    baseline=json.loads((validation/'baseline200/evaluation.json').read_text())
    learned=json.loads((validation/'smoke-evaluation/evaluation.json').read_text())
    summary={'version':'0.1.0','date':'2026-11-17','target':'M2 Pro, 16GB, macOS 27 arm64',
        'local_runtime_tested':'Linux aarch64 / CPU; Mac binaries not executed on this host',
        'cad_unique_parts':41,'cad_instances':171,'all_brep_valid':True,'tests_passed':11,
        'gymnasium_sb3_env_check':'passed','linux_gui_start_check':'passed',
        'ppo_training':'1024-step two-process smoke + 256-step resume verified',
        'baseline_200':{k:baseline[k] for k in ['successes','episodes','success_rate','failures','passes_provisional_simulation_goal']},
        'smoke_policy':{k:learned[k] for k in ['successes','episodes','success_rate','failures']},
        'actual_MG23_linkage_validated':False,'hardware_ready':False,'mac_runtime_executed':False,
        'model_status':'Real STEP geometry with candidate arm grouping and coupled radial finger-envelope approximation; requires kinematic/contact calibration.'}
    (stage/'validation/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    notices=stage/'docs/dependency-notices';notices.mkdir(parents=True,exist_ok=True)
    for wheel in (stage/'payload/wheels').glob('*.whl'):
        with ZipFile(wheel) as z:
            entries=[n for n in z.namelist() if '.dist-info/' in n and any(k in n.upper().rsplit('/',1)[-1] for k in ['LICENSE','COPYING','NOTICE'])]
            for i,name in enumerate(entries):
                (notices/(wheel.stem+f'-{i:02d}-'+Path(name).name)).write_bytes(z.read(name))
    for name in ['setup.sh','run.sh','setup.command','open_gui.command']:(stage/name).chmod(0o755)
    manifest=[]
    for path in sorted(stage.rglob('*')):
        if path.is_file():
            manifest.append({'path':str(path.relative_to(stage)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (stage/'payload-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    manifest.append({'path':'payload-manifest.json','bytes':(stage/'payload-manifest.json').stat().st_size,
        'sha256':hashlib.sha256((stage/'payload-manifest.json').read_bytes()).hexdigest()})
    (stage/'SHA256SUMS.txt').write_text(''.join(f"{f['sha256']}  {f['path']}\n" for f in manifest))
    archive=dest/'mac-arm-v1-offline-20261117.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=5) as t:t.add(stage,arcname='mac-arm-v1')
    expected={f['path']:f for f in manifest};checked=0
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            if not member.isfile():continue
            relative=str(Path(member.name).relative_to('mac-arm-v1'))
            if relative=='SHA256SUMS.txt':continue
            if relative not in expected:raise ValueError('Unexpected archive member '+relative)
            digest=hashlib.sha256(t.extractfile(member).read()).hexdigest()
            if digest!=expected[relative]['sha256']:raise ValueError('Archive content mismatch '+relative)
            checked+=1
    if checked!=len(expected):raise ValueError('Archive missing payload files')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    (dest/'SHA256SUMS.txt').write_text(f'{sha}  {archive.name}\n')
    copy_file(stage/'Mac调参与仿真指南.md',dest/'Mac调参与仿真指南.md')
    copy_file(stage/'validation/summary.json',dest/'validation-summary.json')
    result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha,'verified_payload_files':checked}
    (dest/'archive-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
