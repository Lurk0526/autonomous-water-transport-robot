"""Package local reviewed sources and CAD; no environment or private runtime files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[2]
VERSION='bowl-transport-v0.2'

def build():
    root=ROOT;files=[]
    for folder in ('assets/cad','sim/bowl_transport'):
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    files.extend((root/'tests').glob('test_bowl_transport*.py'))
    files.extend(root/p for p in ('docs/bowl-transport-review-v0.2.md','docs/bowl-transport-simulation-20261023.md'))
    entries=[(str(p.relative_to(root)),p.read_bytes()) for p in sorted(set(files))]
    previews=root/'work/cad-inspection-20261023/packaged'
    if previews.exists():entries.extend(('assets/cad/previews/'+p.name,p.read_bytes()) for p in sorted(previews.glob('*.png')))
    entries.append(('README.md',('''# 满水碗搬运方案与本机审计 v0.2

当前以 docs/bowl-transport-review-v0.2.md 为准。
运行说明：sim/bowl_transport/RUN.md。

在此目录运行：

```bash
python3 sim/bowl_transport/preflight.py
```

打开 sim/bowl_transport/output/report.html。
本机检查通过只证明逻辑、数值和奖励记账；实际机器人与液体尚未验证。
当前 TRAINING_READY=false，建议暂不租长时训练服务器。
''').encode()))
    manifest={'version':VERSION,'files':[{'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()} for n,b in entries]}
    dest=root/'releases'/f'{VERSION}.zip';dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in entries:z.writestr(VERSION+'/'+name,data)
        z.writestr(VERSION+'/bundle-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for f in manifest['files']:
            assert hashlib.sha256(z.read(VERSION+'/'+f['path'])).hexdigest()==f['sha256']
    print(json.dumps({'path':str(dest),'files':len(entries)+1,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
if __name__=='__main__':build()
