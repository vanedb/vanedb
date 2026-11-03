"""Check every retained raw archive member against the public content index."""
from pathlib import Path
import hashlib,json,tarfile
D=Path(__file__).parent;index=json.loads((D/'archive-index.json').read_text());expected={x['path']:x for x in index['files']};assert len(expected)==len(index['files'])
with tarfile.open(D/index['archive'],'r:gz') as t:
 members=t.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
 for m in members:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  assert m.size==expected[m.name]['size'];f=t.extractfile(m);assert f is not None
  assert hashlib.file_digest(f,'sha256').hexdigest()==expected[m.name]['sha256'],m.name
print(f'PASS: {len(expected)} archived raw files match their size and SHA-256 records')
