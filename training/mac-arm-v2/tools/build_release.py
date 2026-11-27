"""Build and verify the v0.2 native Mac arm64 offline package."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    dest=a.destination.resolve();stage=dest/'mac-arm-v2'
    if stage.exists():raise SystemExit('Use a fresh destination; previous delivery is preserved')
    summary=ROOT/'validation/summary.json'
    if not summary.exists():raise SystemExit('Missing final validation summary')
    v=json.loads(summary.read_text());assert v['version']=='0.2.0' and v['checks_complete']
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT,stage,ignore=shutil.ignore_patterns('__pycache__','*.pyc','outputs','runtime','.venv'))
    payload=PROJECT/'work/mac-arm-payload-v2';assert not json.loads((payload/'payload-verification.json').read_text())['errors']
    shutil.copytree(payload/'wheels',stage/'payload/wheels')
    metadata=json.loads((payload/'python-release.json').read_text())
    copy(payload/metadata['assets'][0]['name'],stage/'payload/python-macos-arm64.tar.gz')
    for name in ['wheel-manifest.json','payload-verification.json','python-release.json','setuptools-official.json']:
        copy(payload/name,stage/'payload'/name)
    for name in ['机械臂带爪子.STEP','source-manifest.json']:
        copy(PROJECT/'assets/cad/arm-step-20261117'/name,stage/'source'/name)
    shutil.copytree(PROJECT/'assets/reference/bowl-20261117',stage/'source/bowl-reference')
    copy(PROJECT/'assets/reference/arm-camera-mount-20261116.jpg',stage/'source/arm-camera-mount.jpg')
    validation=PROJECT/'work/mac-arm-v2-validation'
    required=['tests-final.log','env-check.log','gui-native.log','tensorboard-version.log','ablation-method.json',
        'controller-comparison.json','reference200/evaluation.json','sync200/evaluation.json','final-baseline200/evaluation.json',
        'smoke-final/run.json','smoke-final/final.zip','smoke-final/best-evaluation.json','smoke-final/selection.json','smoke-final/source-integrity.json',
        'smoke-final.log','resume-final/run.json','resume-final/final.zip','resume-final/source-integrity.json','resume-final.log',
        'nominal-diagnose/episode.json','nominal-diagnose/trace.json']
    for name in required:
        if not (validation/name).is_file():raise SystemExit('Missing validation file '+name)
        copy(validation/name,stage/'validation'/name)
    notices=stage/'docs/dependency-notices';notices.mkdir(parents=True,exist_ok=True)
    for wheel in (stage/'payload/wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as z:
            names=[n for n in z.namelist() if '.dist-info/' in n and any(k in Path(n).name.upper() for k in ['LICENSE','COPYING','NOTICE'])]
            for i,name in enumerate(names):(notices/(wheel.stem+f'-{i:02d}-'+Path(name).name)).write_bytes(z.read(name))
    for name in ['setup.sh','run.sh','setup.command','open_gui.command']:(stage/name).chmod(0o755)
    records=[{'path':str(p.relative_to(stage)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(stage.rglob('*')) if p.is_file()]
    (stage/'payload-manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    records.append({'path':'payload-manifest.json','bytes':(stage/'payload-manifest.json').stat().st_size,'sha256':sha(stage/'payload-manifest.json')})
    (stage/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in records))
    archive=dest/'mac-arm-v2-offline-20261117.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=5) as t:t.add(stage,arcname='mac-arm-v2')
    expected={r['path']:r for r in records};checked=set()
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            if not member.isfile():continue
            rel=str(Path(member.name).relative_to('mac-arm-v2'))
            if rel=='SHA256SUMS.txt':continue
            assert rel in expected and rel not in checked,rel
            assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==expected[rel]['sha256'],rel
            checked.add(rel)
    assert checked==set(expected)
    result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'verified_payload_files':len(checked)}
    (dest/'archive-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (dest/'SHA256SUMS.txt').write_text(result['sha256']+'  '+archive.name+'\n')
    for name in ['Mac调参与仿真指南.md','validation/调整验证报告.md','validation/summary.json']:
        copy(stage/name,dest/Path(name).name)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
