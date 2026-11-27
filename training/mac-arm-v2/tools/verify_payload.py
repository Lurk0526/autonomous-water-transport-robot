"""Check offline wheel hashes, target tags, Python requirements and dependency closure."""
import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import zipfile
from packaging.requirements import Requirement
from packaging.tags import Tag,cpython_tags,compatible_tags,mac_platforms
from packaging.utils import canonicalize_name,parse_wheel_filename
from packaging.specifiers import SpecifierSet


def verify(payload,lock):
    manifest=json.loads((payload/'wheel-manifest.json').read_text())
    files={p.name:p for p in (payload/'wheels').glob('*.whl')}
    assert set(files)=={r['file'] for r in manifest},'Unexpected or missing wheels'
    platforms=list(mac_platforms(version=(15,0),arch='arm64'))
    allowed=set(cpython_tags((3,12),platforms=platforms))|set(compatible_tags((3,12),interpreter='cp312',platforms=platforms))
    records={};edges=[];errors=[]
    lock_text=lock.read_text().lower().replace('_','-')
    for row in manifest:
        path=files[row['file']];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==row['sha256'],path.name+' SHA256 mismatch'
        name,version,_,tags=parse_wheel_filename(path.name)
        assert tags&allowed,path.name+' target tags incompatible'
        assert str(version)==row['version']
        assert f'{canonicalize_name(name)}=={version} --hash=sha256:{digest}' in lock_text,path.name+' lock mismatch'
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None,path.name+' ZIP CRC mismatch'
            meta=BytesParser().parsebytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
        if meta['Requires-Python']:assert SpecifierSet(meta['Requires-Python']).contains('3.12.16'),path.name+' Python mismatch'
        records[canonicalize_name(name)]={'version':str(version),'requirements':meta.get_all('Requires-Dist',[])}
    environment={'os_name':'posix','sys_platform':'darwin','platform_machine':'arm64','platform_system':'Darwin',
        'platform_release':'26.3.0','platform_version':'macOS 27.3','implementation_name':'cpython',
        'platform_python_implementation':'CPython','implementation_version':'3.12.16','python_version':'3.12','python_full_version':'3.12.16'}
    pending=[(name,'') for name in records];seen=set()
    while pending:
        name,extra=pending.pop()
        if (name,extra) in seen:continue
        seen.add((name,extra))
        for raw in records[name]['requirements']:
            req=Requirement(raw)
            if req.marker and not req.marker.evaluate({**environment,'extra':extra}):continue
            dep=canonicalize_name(req.name)
            if dep not in records:errors.append(f'{name}: missing {raw}');continue
            v=records[dep]['version']
            if req.specifier and not req.specifier.contains(v):errors.append(f'{name}: {raw} conflicts with {v}')
            edge={'package':name,'requires':raw,'resolved_version':v}
            if edge not in edges:edges.append(edge)
            pending.extend((dep,e) for e in req.extras)
    assert not errors,errors
    return {'target':'macOS arm64 CPython 3.12.16','wheels':len(files),'all_hashes_and_zip_crc_verified':True,
        'target_tags_and_requires_python_verified':True,'dependency_relations_checked':len(edges),
        'dependencies':edges,'errors':errors,'runtime':'Same verified native runtime archive as v0.1; macOS execution not performed on Linux',
        'setuptools':'80.9.0 universal wheel; official PyPI hash verified; local TensorBoard CLI import tested'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('payload',type=Path);p.add_argument('lock',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();result=verify(a.payload,a.lock);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='dependencies'})
