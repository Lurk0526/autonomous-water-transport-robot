"""Package v3 for the Mac. Does not run tests, inference, simulation or training."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)


def main():
    dest=PROJECT/'releases/mac-arm-v3-20261117';stage=dest/'Mac机械臂-v3交接包-20261117';pkg=stage/'mac-arm-v3'
    if stage.exists():raise SystemExit('Preserve the existing delivery; choose a fresh release version')
    # Parse code as data only; do not import project modules or execute tests.
    python_paths=sorted(ROOT.rglob('*.py'))
    for p in python_paths:
        if '__pycache__' not in p.parts:ast.parse(p.read_text(),filename=str(p))
    shells=[p for p in ROOT.iterdir() if p.suffix in ('.sh','.command')]
    for p in shells:subprocess.run(['bash','-n',str(p)],check=True)
    shutil.copytree(ROOT,pkg,ignore=shutil.ignore_patterns('__pycache__','*.pyc','outputs','runtime','.venv'))
    payload=PROJECT/'work/mac-arm-payload-v3'
    proof=json.loads((payload/'payload-verification.json').read_text());assert not proof['errors']
    shutil.copytree(payload/'wheels',pkg/'payload/wheels')
    python=json.loads((payload/'python-release.json').read_text())
    copy(payload/python['assets'][0]['name'],pkg/'payload/python-macos-arm64.tar.gz')
    for n in ['python-release.json','wheel-manifest.json','payload-verification.json','base-version-lock.json','dependency-extension-summary.json']:
        copy(payload/n,pkg/'payload'/n)
    for n in ['机械臂带爪子.STEP','source-manifest.json']:copy(PROJECT/'assets/cad/arm-step-20261117'/n,pkg/'source'/n)
    shutil.copytree(PROJECT/'assets/reference/bowl-20261117',pkg/'source/bowl-reference')
    copy(PROJECT/'assets/reference/arm-camera-mount-20261116.jpg',pkg/'source/arm-camera-mount.jpg')
    original=Path('/media/nvidia/TU100/Mac机械臂v2-回传-20261117-211433.zip')
    copy(original,pkg/'source/mac-v2-feedback/Mac机械臂v2-回传-20261117-211433.zip')
    copy(original.with_suffix('.zip.sha256'),pkg/'source/mac-v2-feedback/Mac机械臂v2-回传-20261117-211433.zip.sha256')
    summary={'version':'0.3.0','package_status':'configured_for_Mac_execution_not_effect_validated',
        'user_requirement':'No local simulation, training, inference, benchmark or test execution; run on receiving Mac.',
        'local_runtime_tests_executed':False,'local_inference_executed':False,'local_simulation_executed':False,
        'local_training_or_benchmark_executed':False,'test_suite_status':'written_but_not_executed_for_v3',
        'static_ast_parsed_files':len(python_paths),'shell_syntax_checked_files':len(shells),
        'dependencies':{'wheels':60,'old_base_versions_preserved':42,'applicable_dependency_edges':96,'official_PyPI_hashes_and_zip_crc_verified':True},
        'prior_Mac_v2_evidence':{'baseline_dev':'93/100','baseline_replication':'179/200','is_v3_result':False,'internal_file_records_verified':82},
        'dual_camera_scene':'YOLO predictions and measured synthetic optical geometry; known platform map; no ground-truth pose fallback',
        'vehicle':'unmeasured front-steered planar surrogate, stationary handoff to MuJoCo arm',
        'camera_calibration':'simulation_assumptions_not_real_camera_extrinsics',
        'gripper':'unvalidated_radial_proxy_with_new_contact_supervision','hardware_ready':False,'fluid_simulated':False}
    (pkg/'validation/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    notices=pkg/'docs/dependency-notices';notices.mkdir(parents=True,exist_ok=True)
    for wheel in (pkg/'payload/wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as z:
            for i,n in enumerate([n for n in z.namelist() if '.dist-info/' in n and any(k in Path(n).name.upper() for k in ['LICENSE','COPYING','NOTICE'])]):
                (notices/(wheel.stem+f'-{i:02d}-'+Path(n).name)).write_bytes(z.read(n))
    for p in pkg.iterdir():
        if p.suffix in ('.sh','.command'):p.chmod(0o755)
    manifest=[{'path':str(p.relative_to(pkg)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(pkg.rglob('*')) if p.is_file()]
    (pkg/'payload-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    manifest.append({'path':'payload-manifest.json','bytes':(pkg/'payload-manifest.json').stat().st_size,'sha256':sha(pkg/'payload-manifest.json')})
    (pkg/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in manifest))
    for name in ['Mac视觉接近与训练v3指南.md','本轮需要回传的文件.md']:
        copy(pkg/name,stage/name)
    (stage/'00-先读我.md').write_text('''# Mac v0.3 交接包

完整环境已经放在 mac-arm-v3 子目录，不依赖旧包，不需要另外下载依赖。

1. 整个 ZIP 传到 Mac 并解压到新目录，保留旧 v1/v2。
2. 把《本轮需要回传的文件.md》交给 Mac 上的助手。
3. 在 mac-arm-v3 子目录终端执行：

```bash
bash setup.sh > Mac安装日志.log 2>&1
bash Mac执行本轮.sh --large-if-ready
```

先运行 Mac 检查与双相机接近，再依据结果决定是否2万步/累计约10万步训练。未准入会明确停止并保留原始证据。MPS经Mac实测选择；物理仍CPU。

结束或报错后填写 Mac回传说明.md，运行 bash 打包回传.sh，把生成的ZIP和.sha256传回。安装日志复制进本轮实验的logs目录。

本轮交付主机没有运行测试、识别、仿真、训练或测速。新效果待Mac执行；上一轮179/200只是Mac v2历史结果。
''')
    outer=[{'path':str(p.relative_to(stage)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(stage.rglob('*')) if p.is_file()]
    (stage/'交接包文件清单.json').write_text(json.dumps(outer,ensure_ascii=False,indent=2)+'\n')
    archive=dest/'Mac机械臂-v3视觉接近与训练-20261117.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,stage.name+'/'+str(p.relative_to(stage)),compress_type=zipfile.ZIP_STORED if p.suffix in ('.whl','.gz','.zip') else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in outer:
            h=hashlib.sha256()
            with z.open(stage.name+'/'+row['path']) as f:
                for b in iter(lambda:f.read(1048576),b''):h.update(b)
            assert h.hexdigest()==row['sha256'],row['path']
    checksum=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n')
    result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':checksum,'archive_files_verified':len(outer)+1,
        'local_effect_execution_performed':False,'summary':summary}
    (dest/'archive-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='summary'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
