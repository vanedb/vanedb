"""Package completed campaign evidence; run only after all measurements stop."""
from pathlib import Path
import gzip,hashlib,io,json,tarfile
O=Path(__file__).parent;D=Path('/Users/anton/code/codex/vanedb-020-perf-evidence-20261101/docs/release/evidence/2026-11-02-performance')
phases=('probe','criterion','index-confirm','cosine','l2','android-cosine','android-l2')
for phase in phases:
 m=json.loads((O/(phase+'-campaign.json')).read_text());assert m['status'] in ('completed','blocked','failed'),phase
 g=json.loads((O/(phase+'-indexer-pause.json')).read_text());assert g['resumed'],phase
assert 'ended' in json.loads((O/'android-emulator-lifecycle.json').read_text())
D.mkdir(parents=True,exist_ok=True);assert not list(D.iterdir()),'Never overwrite published evidence'
exclude={'evidence-pr-before-update.json','package_evidence.py'}
plain=[];raw=[]
for p in sorted(O.iterdir()):
 if p.name in exclude or p.name.startswith('evidence-pr-'):continue
 if p.is_file():
  if p.suffix in ('.json','.py','.md','.ini') and '-v1.' not in p.name and '-v2.' not in p.name:plain.append(p)
  else:raw.append(p)
 elif p.is_dir() and (p.name.startswith('pass-') or p.name.startswith('index-confirm-')):
  raw.extend(sorted(q for q in p.rglob('*') if q.is_file()))
original=[];entries=[];redactions=[]
def content(p):
 assert not p.is_symlink(),p
 data=p.read_bytes();original.append(f'{hashlib.sha256(data).hexdigest()}  {p.relative_to(O)}')
 if p.name in ('android-emulator.log','android-equivalent-emulator.log'):
  lines=data.decode().splitlines(keepends=True);count=0
  for i,line in enumerate(lines):
   if 'adb public key' in line.lower() or 'adb.pubkey' in line.lower():lines[i]='[REDACTED host ADB public-key line]\n';count+=1
  data=''.join(lines).encode();redactions.append(dict(file=p.name,redacted_lines=count))
 return data
for p in plain:
 data=content(p);(D/p.name).write_bytes(data)
with (D/'raw-rounds-and-load.tar.gz').open('wb') as out:
 with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as z:
  with tarfile.open(fileobj=z,mode='w') as t:
   for p in raw:
    name=p.relative_to(O).as_posix();data=content(p);info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';t.addfile(info,io.BytesIO(data));entries.append(dict(path=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
(D/'archive-index.json').write_text(json.dumps(dict(archive='raw-rounds-and-load.tar.gz',files=entries,redactions=redactions),indent=2)+'\n')
(D/'local-original-artifacts.sha256').write_text('\n'.join(sorted(original))+'\n')
(D/'package_evidence.py').write_bytes(Path(__file__).read_bytes())
print('Packaged',len(plain),'plain files and',len(raw),'raw files; archive bytes',(D/'raw-rounds-and-load.tar.gz').stat().st_size)
print('Generate SHA256SUMS after README and final validation records are added.')
