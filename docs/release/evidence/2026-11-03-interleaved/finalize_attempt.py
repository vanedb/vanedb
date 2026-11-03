from pathlib import Path
import hashlib,json,subprocess,datetime,difflib
O=Path(__file__).parent
P=Path('/Users/anton/code/codex/020-perf-acceptance-20261101')
S=Path('/Users/anton/code/codex/vanedb-020-perf-20261101')
A=Path('/Users/anton/code/codex/vanedb-020-perf-baseline-20261101')
OLD=Path('/Users/anton/code/codex/020-perf-acceptance-20261102')
M=json.loads((O/'criterion-campaign.json').read_text())
G=json.loads((O/'criterion-indexer-pause.json').read_text())
assert M['status']=='completed', M['status']
assert G.get('resumed') is True
print('guardian',json.dumps(G))
def command(args,cwd=None):
 p=subprocess.run(args,cwd=cwd,text=True,capture_output=True)
 return {'command':args,'cwd':str(cwd) if cwd else None,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
checks={}
expected=json.loads((P/'binary-hashes.json').read_text())
expected[str(S/'bench/compare/fixtures/embeddings.vnef')]='4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d'
expected[str(P/'graph.vndb')]='3672206a43d866b598b2abac71babf71181f7d459a2d73687fc8928d389b1cbc'
for p,e in expected.items():
 actual=sha(p);checks[p]={'expected':e,'actual':actual,'passed':actual==e}
assert all(x['passed'] for x in checks.values())
sources=[]
for path,head in [(S,M['candidate']),(A,M['baseline'])]:
 rev=command(['git','rev-parse','HEAD'],path); status=command(['git','status','--porcelain','--untracked-files=no'],path)
 assert rev['stdout'].strip()==head and status['stdout']==''
 sources.append({'revision':rev,'status':status})
pids=[M['controller_pid']]+[e['pid'] for e in M['events']]
proc=command(['ps','-p',','.join(map(str,pids)),'-o','pid=,ppid=,stat=,lstart=,comm='])
assert proc['stdout'].strip()==''
target=command(['ps','-p',str(G['target_pid']),'-o','uid=,lstart=,stat=,comm='])
state=command(['ps','-p',str(G['target_pid']),'-o','stat='])
assert state['exit_code']==0 and 'T' not in state['stdout']
for p in O.glob('*.py'):
 old=OLD/p.name
 if old.exists():(O/(p.name+'.diff')).write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),p.read_text().splitlines(True),fromfile=str(old),tofile=str(p))))
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hashes':checks,'sources':sources,'owned_process_check':proc,'indexer_guardian':G,'indexer_current_state':target,'power':command(['pmset','-g','batt']),'storage':command(['df','-h',str(O)]),'controllers':{p.name:sha(p) for p in O.glob('*.py')},'decision':'No further full competitor campaigns on this repeatedly nonqualifying host. Existing November2 complete raw campaigns retained. No host-screen bypass; no numerical performance PASS inferred.'}
(O/'post-run-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('Verified',len(checks),'binary/fixture/graph identities and source heads; all owned campaign processes exited.')
