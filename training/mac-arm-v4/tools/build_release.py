"""Build a self-contained Apple Silicon release; preserve existing releases."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--label',default='20261119');args=parser.parse_args()
    if not args.label.replace('-','').isalnum():raise SystemExit('Use an alphanumeric release label')
    dest=PROJECT/('releases/mac-arm-v4-'+args.label)
    stage=dest/('Mac机械臂-v4交接包-'+args.label);pkg=stage/'mac-arm-v4'
    archive=dest/('Mac机械臂-v4修复与视觉抓放-'+args.label+'.zip')
    if stage.exists() or archive.exists():raise SystemExit('Preserve the existing delivery; choose another --label')
    summary=json.loads((ROOT/'validation/summary.json').read_text())
    if not summary.get('local_tests_passed'):raise SystemExit('Missing completed local test evidence')
    paths=[p for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts]
    for p in paths:ast.parse(p.read_text(),filename=str(p))
    shells=[p for p in ROOT.iterdir() if p.suffix in ('.sh','.command')]
    for p in shells:subprocess.run(['bash','-n',str(p)],check=True)
    shutil.copytree(ROOT,pkg,ignore=shutil.ignore_patterns('__pycache__','*.pyc','outputs','runtime','.venv'))
    payload=PROJECT/'work/mac-arm-payload-v3'
    from verify_payload import verify
    verification=verify(payload,ROOT/'requirements-mac.lock.txt')
    shutil.copytree(payload/'wheels',pkg/'payload/wheels')
    runtime=json.loads((payload/'python-release.json').read_text())
    copy(payload/runtime['assets'][0]['name'],pkg/'payload/python-macos-arm64.tar.gz')
    for name in ['python-release.json','wheel-manifest.json','base-version-lock.json','dependency-extension-summary.json']:
        copy(payload/name,pkg/'payload'/name)
    (pkg/'payload/payload-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    for name in ['机械臂带爪子.STEP','source-manifest.json']:
        copy(PROJECT/'assets/cad/arm-step-20261117'/name,pkg/'source'/name)
    shutil.copytree(PROJECT/'assets/reference/bowl-20261117',pkg/'source/bowl-reference')
    copy(PROJECT/'assets/reference/arm-camera-mount-20261116.jpg',pkg/'source/arm-camera-mount.jpg')
    copy(Path('同步四爪抓水碗-算法与复测.zip'),pkg/'source/用户回传-同步四爪抓水碗-算法与复测.zip')
    notices=pkg/'docs/dependency-notices';notices.mkdir(parents=True,exist_ok=True)
    for wheel in (pkg/'payload/wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as z:
            names=[name for name in z.namelist() if '.dist-info/' in name and
                any(key in Path(name).name.upper() for key in ['LICENSE','COPYING','NOTICE'])]
            for i,name in enumerate(names):(notices/(wheel.stem+f'-{i:02d}-'+Path(name).name)).write_bytes(z.read(name))
    for p in pkg.iterdir():
        if p.suffix in ('.sh','.command'):p.chmod(0o755)
    manifest=[{'path':str(p.relative_to(pkg)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(pkg.rglob('*')) if p.is_file()]
    (pkg/'payload-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    manifest.append({'path':'payload-manifest.json','bytes':(pkg/'payload-manifest.json').stat().st_size,'sha256':sha(pkg/'payload-manifest.json')})
    (pkg/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in manifest))
    for name in ['Mac训练与视觉抓放v4指南.md','本轮需要回传的文件.md']:copy(pkg/name,stage/name)
    copy(pkg/'validation/本地修复与验证报告.md',stage/'本地修复与验证报告.md')
    (stage/'00-先读我.md').write_text('''# Mac v0.4完整离线交接包

整个包解压到Mac新目录，保留旧v1/v2/v3。环境、MuJoCo GUI、CAD、分割模型与原始本地验证证据均在mac-arm-v4内。

进入mac-arm-v4后运行：

```bash
bash setup.sh > Mac安装日志.log 2>&1
bash Mac执行本轮.sh --large-if-ready
```

先验收新场景、测CPU/MPS，再依据门槛启动2万步/累计约10万步。可以去掉--large-if-ready只验收。完整说明见《Mac训练与视觉抓放v4指南.md》。本轮按用户授权做了本地小规模迭代，结果和限制见《本地修复与验证报告.md》；未证明实机95%成功率。

把《本轮需要回传的文件.md》交给Mac助手。无论成功或失败都填写Mac回传说明.md，运行bash 打包回传.sh，传回生成的.zip及.sha256。没有真实机器人命令接口。
''')
    outer=[{'path':str(p.relative_to(stage)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(stage.rglob('*')) if p.is_file()]
    (stage/'交接包文件清单.json').write_text(json.dumps(outer,ensure_ascii=False,indent=2)+'\n')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():z.write(p,stage.name+'/'+str(p.relative_to(stage)),compress_type=zipfile.ZIP_STORED if p.suffix in ('.whl','.gz','.zip') else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in outer:
            digest=hashlib.sha256()
            with z.open(stage.name+'/'+row['path']) as f:
                for block in iter(lambda:f.read(1048576),b''):digest.update(block)
            assert digest.hexdigest()==row['sha256'],row['path']
    checksum=sha(archive);archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n')
    result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':checksum,
        'archive_files_verified':len(outer)+1,'static_python_files':len(paths),'shell_scripts_checked':len(shells),
        'dependencies':{key:value for key,value in verification.items() if key!='dependencies'},'validation':summary}
    (dest/'archive-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('dependencies','validation')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
