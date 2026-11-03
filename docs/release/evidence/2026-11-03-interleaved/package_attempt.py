from pathlib import Path
import hashlib,json,tarfile,io,shutil
O=Path(__file__).parent
D=Path('/Users/anton/code/codex/020-perf-acceptance-20261103T091447Z')
E=Path('/Users/anton/code/codex/vanedb-020-perf-evidence-20261101/docs/release/evidence/2026-11-03-interleaved')
assert json.loads((O/'criterion-campaign.json').read_text())['status']=='completed'
assert json.loads((O/'criterion-indexer-pause.json').read_text())['resumed'] is True
assert (O/'criterion-analysis.json').is_file() and (O/'post-run-validation.json').is_file()
assert not E.exists(), 'Never overwrite a public attempt'
E.mkdir()
shutil.copytree(O,D,dirs_exist_ok=True)
files=sorted(p for p in D.rglob('*') if p.is_file());assert all(not p.is_symlink() for p in files)
index=[]
with tarfile.open(E/'raw-results.tar.gz','w:gz') as t:
 for p in files:
  data=p.read_bytes(); name=str(p.relative_to(D));index.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0;t.addfile(info,io.BytesIO(data))
(E/'raw-file-index.json').write_text(json.dumps(index,indent=2)+'\n')
for name in ['criterion-campaign.json','criterion-analysis.json','criterion-indexer-pause.json','initial-preflight.json','attempt-provenance.json','post-run-validation.json','analyze_criterion.py','run_native.py','pause_indexer.py','finalize_attempt.py','package_attempt.py']:
 shutil.copy2(D/name,E/name)
with tarfile.open(E/'raw-results.tar.gz') as t:
 members=t.getmembers();assert len(members)==len(index) and len({m.name for m in members})==len(index)
 for m,row in zip(members,index):
  assert m.isfile() and m.name==row['path'] and not m.name.startswith('/') and '..' not in Path(m.name).parts
  data=t.extractfile(m).read();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
print(json.dumps({'raw_files':len(index),'raw_bytes':sum(x['bytes'] for x in index),'archive_bytes':(E/'raw-results.tar.gz').stat().st_size,'archive_verified':True,'public_directory':str(E)},indent=2))
