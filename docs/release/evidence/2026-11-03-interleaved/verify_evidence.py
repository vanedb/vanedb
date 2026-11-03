from pathlib import Path
import hashlib,json,tarfile,tempfile,subprocess
E=Path(__file__).parent
index=json.loads((E/'raw-file-index.json').read_text())
with tarfile.open(E/'raw-results.tar.gz') as t:
 members=t.getmembers()
 assert len(members)==len(index)==1418
 assert len({m.name for m in members})==len(members)
 for m,row in zip(members,index):
  assert m.isfile() and m.name==row['path']
  assert not m.name.startswith('/') and '..' not in Path(m.name).parts
  data=t.extractfile(m).read()
  assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 with tempfile.TemporaryDirectory(prefix='vanedb-interleaved-verify-') as d:
  t.extractall(d,filter='data')
  subprocess.run(['python3',str(Path(d)/'analyze_criterion.py')],check=True,capture_output=True,text=True)
  assert (Path(d)/'criterion-analysis.json').read_bytes()==(E/'criterion-analysis.json').read_bytes()
  v=json.loads((Path(d)/'post-run-validation.json').read_text())
  assert all(x['passed'] for x in v['hashes'].values()) and len(v['hashes'])==14
  assert v['indexer_guardian']['resumed'] is True
  for name,expected in v['controllers'].items():
   assert hashlib.sha256((Path(d)/name).read_bytes()).hexdigest()==expected,name
m=json.loads((E/'criterion-campaign.json').read_text())
a=json.loads((E/'criterion-analysis.json').read_text())
assert m['status']=='completed' and len(m['events'])==24 and all(e['exit_code']==0 for e in m['events'])
assert a['host']['load_flags']==14 and a['host']['measured_samples']==359
assert not a['qualifying'] and not a['host']['screen_passed']
assert all(not q['complete_block_screen_passed'] for q in a['target_quality'].values())
print('PASS: all raw files, final controller hashes, independently regenerated analysis, complete planned execution and nonqualifying verdict verified.')
