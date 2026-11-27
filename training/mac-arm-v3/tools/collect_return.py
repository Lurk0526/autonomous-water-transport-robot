#!/usr/bin/env python3
"""Collect only this robot experiment's files into a portable feedback ZIP."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

EXCLUDE_DIRS={'.venv','runtime','payload','__pycache__','.git','.DS_Store'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def files(root):
    if not root.is_dir():return
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not path.is_file():continue
        if any(p in EXCLUDE_DIRS for p in path.relative_to(root).parts):continue
        if path.name.endswith(('.pyc','.pyo')) or path.name=='.env' or path.name.startswith('.env.'):continue
        yield path


def command(argv):
    try:
        r=subprocess.run(argv,capture_output=True,text=True,timeout=15)
        return {'exit_code':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as e:return {'error':str(e)}


def system_info():
    versions={}
    for name in ['mujoco','torch','stable-baselines3','gymnasium','numpy','scipy','tensorboard','setuptools','ultralytics','torchvision','opencv-python']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    result={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'packages':versions}
    if platform.system()=='Darwin':
        result['macos']=command(['sw_vers'])
        result['hardware']=command(['sysctl','hw.model','hw.memsize','hw.ncpu'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('project',type=Path,help='Extracted mac-arm-v3 directory')
    p.add_argument('--run',type=Path,action='append',help='Specific output directory to return; can be repeated')
    p.add_argument('--log-dir',type=Path,action='append',default=[])
    p.add_argument('--note',type=Path,help='Filled Markdown experiment report')
    p.add_argument('--destination',type=Path,help='Output folder; default sibling Mac回传')
    a=p.parse_args();root=a.project.expanduser().resolve()
    if not (root/'arm_training/env.py').is_file():p.error('Select the actual extracted mac-arm-v3 directory')
    destination=(a.destination.expanduser().resolve() if a.destination else root.parent/'Mac回传')
    destination.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')
    archive=destination/f'Mac机械臂v3-回传-{stamp}.zip'
    if archive.exists():p.error('Archive already exists; wait one second or choose another destination')
    runs=[r.expanduser().resolve() for r in a.run] if a.run else sorted((root/'outputs').glob('mac-v3-check-*'))
    if not runs:runs=[root/'outputs']
    selected={};missing=[]
    def add(path,name):
        if not path.is_symlink() and path.is_file():selected[name]=path
    for dirname in ['arm_training','config','tests','tools']:
        for f in files(root/dirname):add(f,'project/'+str(f.relative_to(root)))
    for name in ['requirements.txt','requirements-mac.lock.txt','README.md','Mac调参与仿真指南.md','Mac视觉接近与训练v3指南.md','本轮需要回传的文件.md','SHA256SUMS.txt','payload-manifest.json']:
        if (root/name).is_file():add(root/name,'project/'+name)
    for name in ['assets/candidate-model.json','assets/cad/cad-manifest.json','assets/vision/manifest.json']:
        if (root/name).is_file():add(root/name,'project/'+name)
    evaluation_summaries=[]
    for i,run in enumerate(runs):
        if not run.is_dir():missing.append(str(run));continue
        label=f'run-{i+1:02d}-{run.name}'
        for f in files(run):add(f,'results/'+label+'/'+str(f.relative_to(run)))
        for f in sorted([*run.rglob('evaluation.json'),*run.rglob('mobile-evaluation.json')]):
            if f.is_symlink():continue
            try:
                v=json.loads(f.read_text())
                evaluation_summaries.append({'path':str(f.relative_to(root)) if f.is_relative_to(root) else f.name,
                    **{k:v.get(k) for k in ['episodes','successes','success_rate','success_rate_wilson95','failures','reachable_episodes','wall_s','stage_success_rates','learned_vision','vision_device']},
                    'provenance':{k:v.get('provenance',{}).get(k) for k in ['package_generation','source_sha256','config_sha256','model_sha256']}})
            except (OSError,ValueError) as e:missing.append(f'{f.name}: {e}')
    for i,logs in enumerate(a.log_dir):
        logs=logs.expanduser().resolve()
        if not logs.is_dir():missing.append(str(logs));continue
        for f in files(logs):add(f,f'logs/log-{i+1:02d}-{logs.name}/'+str(f.relative_to(logs)))
    note=a.note.expanduser().resolve() if a.note else root/'Mac回传说明.md'
    if note.is_file():add(note,'Mac回传说明.md')
    else:missing.append('Mac回传说明.md 尚未填写；请在最终发送前补充 GUI 观察、改动、失败现象。')
    changes=[]
    manifest=root/'payload-manifest.json'
    if manifest.is_file():
        original=json.loads(manifest.read_text())
        for row in original:
            rel=row['path'];path=root/rel
            if not (rel.startswith(('arm_training/','config/','assets/','tools/','tests/')) or rel in ['requirements.txt','requirements-mac.lock.txt']):continue
            if not path.is_file():changes.append({'path':rel,'status':'missing'});continue
            current=digest(path)
            if current!=row['sha256']:
                changes.append({'path':rel,'status':'modified','original_sha256':row['sha256'],'actual_sha256':current})
                add(path,'project/'+rel)
    info={'created_at':datetime.now().astimezone().isoformat(),'system':system_info(),
        'evaluation_summaries':evaluation_summaries,'changed_original_files':changes,'missing_or_notes':missing,
        'scope':'Robot experiment source/config/results only. No virtualenv, portable runtime, wheel payload or unchanged CAD meshes.',
        'hardware_ready':False}
    index=[]
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for name,path in sorted(selected.items()):
            z.write(path,name);index.append({'path':name,'bytes':path.stat().st_size,'sha256':digest(path)})
        z.writestr('回传索引.json',json.dumps(info,ensure_ascii=False,indent=2)+'\n')
        z.writestr('文件清单.json',json.dumps(index,ensure_ascii=False,indent=2)+'\n')
        if not note.is_file():
            z.writestr('Mac回传说明-待补充.md','# 请补充\n\nGUI 能否正常显示：未知\n模型/夹爪现象：待填写\n相对原版改动：见回传索引.json\n失败与未完成项目：请填写\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in index:
            h=hashlib.sha256()
            with z.open(row['path']) as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==row['sha256']
    checksum=digest(archive)
    archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':checksum,'collected_files':len(index),'notes':missing},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
