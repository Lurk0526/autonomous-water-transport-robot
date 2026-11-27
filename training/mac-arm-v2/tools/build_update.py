"""Create a small offline update that builds a sibling v2 directory from verified v1 assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

INSTALLER='''import argparse,hashlib,json,os,shutil,sys
from pathlib import Path
p=argparse.ArgumentParser(description="Verify old immutable payload and create a separate mac-arm-v2 directory")
p.add_argument("old",type=Path);p.add_argument("--destination",type=Path)
a=p.parse_args();root=Path(__file__).resolve().parent;old=a.old.expanduser().resolve()
dest=a.destination.expanduser().resolve() if a.destination else old.parent/"mac-arm-v2"
if not (old/"assets/candidate-model.json").is_file():sys.exit("Select the extracted mac-arm-v1 directory")
if dest.exists():sys.exit("Destination already exists; preserve it and select a fresh --destination")
records=json.loads((root/"update-manifest.json").read_text());sources=[]
for row in records:
    rel=Path(row["path"])
    if rel.is_absolute() or ".." in rel.parts:sys.exit("Invalid manifest path")
    source=root/"files"/rel
    if not source.is_file():source=old/rel
    if not source.is_file():sys.exit("Missing old payload: "+str(rel)+"; use the full v2 archive")
    if hashlib.sha256(source.read_bytes()).hexdigest()!=row["sha256"]:sys.exit("Hash mismatch: "+str(rel)+"; old files were changed or this update was corrupted")
    sources.append((source,rel))
temp=dest.with_name(dest.name+"-building-"+str(os.getpid()))
if temp.exists():sys.exit("Temporary destination already exists")
try:
    temp.mkdir(parents=True)
    for source,rel in sources:
        target=temp/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    temp.rename(dest)
except BaseException:
    if temp.exists():shutil.rmtree(temp)
    raise
print("Verified and created:",dest)
print("Next: cd into that directory and run bash setup.sh. Old training outputs are preserved.")
'''
LAUNCHER='''#!/usr/bin/env bash
set -euo pipefail
update_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash apply_update.sh /path/to/mac-arm-v1' >&2
  exit 2
fi
old_arm_dir="$1"
shift
if [[ -x "$old_arm_dir/runtime/python/bin/python3" ]]; then
  arm_update_python="$old_arm_dir/runtime/python/bin/python3"
elif [[ -x "$old_arm_dir/.venv/bin/python" ]]; then
  arm_update_python="$old_arm_dir/.venv/bin/python"
else
  echo 'Old package Python not found; use the full v2 archive.' >&2
  exit 2
fi
exec "$arm_update_python" "$update_dir/apply_update.py" "$old_arm_dir" "$@"
'''


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('full_stage',type=Path);p.add_argument('old_stage',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();dest=a.destination;stage=dest/'mac-arm-v2-update'
    if stage.exists():raise SystemExit('Use a fresh update destination')
    stage.mkdir(parents=True);records=[];reused=0
    for path in sorted(a.full_stage.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(a.full_stage);digest=sha(path)
        records.append({'path':str(rel),'sha256':digest,'bytes':path.stat().st_size})
        old=a.old_stage/rel
        reuse=str(rel).startswith(('assets/','payload/','source/','docs/dependency-notices/')) and old.is_file() and sha(old)==digest
        if reuse:reused+=1;continue
        target=stage/'files'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    (stage/'update-manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    (stage/'apply_update.py').write_text(INSTALLER);(stage/'apply_update.sh').write_text(LAUNCHER);(stage/'apply_update.sh').chmod(0o755)
    (stage/'使用方法.md').write_text('# v0.2 离线更新\n\n在更新目录终端执行：\n\n```bash\nbash apply_update.sh "/你的路径/mac-arm-v1"\ncd "/你的路径/mac-arm-v2"\nbash setup.sh\n```\n\n验证旧包内未改变的网格、原始资料和运行时，复制到独立 v2 目录，再应用修复。不修改 v1，不复制旧 outputs、.venv 或 runtime。已有 v2 时停止，可用 --destination 指定新目录。旧文件被修改或不完整时使用完整 v2 归档。\n')
    archive=dest/'mac-arm-v2-update-20261117.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=5) as t:t.add(stage,arcname='mac-arm-v2-update')
    result={'archive':str(archive.resolve()),'sha256':sha(archive),'bytes':archive.stat().st_size,'files_verified':len(records),'reused_files':reused}
    (dest/'update-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (dest/'SHA256SUMS.txt').open('a') as f:f.write(result['sha256']+'  '+archive.name+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
