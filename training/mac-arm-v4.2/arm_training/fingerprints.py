"""Fingerprint actual collision assets, not only CAD instance labels."""
import hashlib
import json
from pathlib import Path


def model_fingerprint(root):
    root=Path(root);paths=[root/'assets/candidate-model.json',root/'assets/cad/cad-manifest.json',root/'assets/cad/collision-parts.json']
    cad=json.loads(paths[0].read_text())
    paths += [root/'assets/cad/meshes'/part['mesh'] for part in cad['parts']]
    result=hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():raise FileNotFoundError('Missing model asset: '+str(path))
        result.update(str(path.relative_to(root)).encode());result.update(path.read_bytes())
    return result.hexdigest()
