"""Sequential, prebuilt-only benchmarks; never turn load/power flags into PASS."""
from pathlib import Path
import datetime, hashlib, json, os, signal, subprocess, sys, time
R=Path('/Users/anton/code/codex');O=R/'020-perf-acceptance-20261103T080155Z';P=R/'020-perf-acceptance-20261101';S=R/'vanedb-020-perf-20261101';A=R/'vanedb-020-perf-baseline-20261101'
PHASE=sys.argv[1] if len(sys.argv)>1 else 'probe'; assert PHASE in ('probe','criterion','index-confirm','cosine','l2')
E=os.environ.copy();E['PATH']='/Users/anton/.rustup/toolchains/stable-aarch64-apple-darwin/bin:'+E['PATH'];E['RUSTC']='/Users/anton/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc';E.pop('CARGO_TARGET_DIR',None)
M={'phase':PHASE,'reservation':'User on 2026-11-03: Mac is quiet now, can you do it?','candidate':'e1761dcf4258011b446a361dc529da082ea2fc94','baseline':'51cb2d871538985aad933a13ebe50c13a0031dfa','status':'preflight','events':[],'host_samples_file':PHASE+'-host.jsonl','sampling':'5-second ps snapshots; excluding this controller, ps and current benchmark process descendants only; no background application exclusions','screen':'max background CPU <=50%, sum <=150%, AC power; all observations retained; samples are not continuous isolation proof'}
manifest=O/(PHASE+'-campaign.json');assert not manifest.exists(),'Never overwrite an attempt'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():manifest.write_text(json.dumps(M,indent=2)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evaluate(rows,excluded):
 v=[r['cpu'] for r in rows if r['pid'] not in excluded and r['executable']!='ps'];return {'background_max':max(v,default=0),'background_sum':sum(v),'quiet':max(v,default=0)<=50 and sum(v)<=150}
# Boundary checks include exact-threshold acceptance and background exclusion.
assert evaluate([{'pid':1,'cpu':50,'executable':'app'}],set())['quiet']
assert not evaluate([{'pid':1,'cpu':50.1,'executable':'app'}],set())['quiet']
assert not evaluate([{'pid':i,'cpu':40,'executable':'app'} for i in range(4)],set())['quiet']
assert evaluate([{'pid':1,'cpu':900,'executable':'bench'}],{1})['quiet']
def sample(stage,pid=None):
 raw=subprocess.check_output(['ps','-axo','pid=,ppid=,pcpu=,comm='],text=True);rows=[]
 for line in raw.splitlines():
  x=line.split(None,3)
  if len(x)==4:rows.append({'pid':int(x[0]),'ppid':int(x[1]),'cpu':float(x[2]),'executable':Path(x[3]).name})
 excluded={os.getpid()};desc={pid} if pid else set()
 while True:
  nxt=desc|{r['pid'] for r in rows if r['ppid'] in desc}
  if nxt==desc:break
  desc=nxt
 excluded|=desc
 power=subprocess.check_output(['pmset','-g','batt'],text=True)
 rec={'utc':now(),'stage':stage,'excluded_pids':sorted(excluded),'processes':[r for r in rows if r['cpu']>0],'zero_cpu_processes_omitted':sum(r['cpu']==0 for r in rows),'power':power,'ac_power':"'AC Power'" in power,**evaluate(rows,excluded)}
 with (O/(PHASE+'-host.jsonl')).open('a') as f:f.write(json.dumps(rec)+'\n')
 return rec
def preflight(stage):
 consecutive=0;deadline=time.monotonic()+600
 while time.monotonic()<deadline:
  s=sample(stage);consecutive=consecutive+1 if s['quiet'] and s['ac_power'] else 0
  print(now(),stage,'AC',s['ac_power'],'max',s['background_max'],'sum',round(s['background_sum'],1),'quiet_streak',consecutive,flush=True)
  if not s['ac_power']:raise RuntimeError('AC power required; stopped before measurement')
  if consecutive>=3:return
  time.sleep(5)
 raise RuntimeError('Quiet host did not settle within ten minutes; no bypass')
def run(name,args,cwd=S,env=E):
 preflight(name+'-preflight')
 event={'name':name,'command':list(map(str,args)),'cwd':str(cwd),'started':now(),'load_flags':0,'power_flags':0};M['events'].append(event);M['status']='running';save()
 with (O/(name+'.stdout.log')).open('w') as out,(O/(name+'.stderr.log')).open('w') as err:
  p=subprocess.Popen(args,cwd=cwd,env=env,stdout=out,stderr=err,start_new_session=True);event['pid']=p.pid;save()
  try:
   while True:
    s=sample(name,p.pid);event['load_flags']+=not s['quiet'];event['power_flags']+=not s['ac_power'];save()
    if not s['ac_power']:raise RuntimeError('AC lost during measurement; attempt not qualifying')
    try:code=p.wait(timeout=5);break
    except subprocess.TimeoutExpired:pass
  except BaseException:
   if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait()
   raise
 event.update(exit_code=code,ended=now(),host_screen_passed=event['load_flags']==event['power_flags']==0);save();print(now(),name,'finished',code,'load_flags',event['load_flags'],flush=True)
 if code:raise RuntimeError(name+' exited '+str(code))
M['started']=now();save();awake=None
try:
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()==M['candidate']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=A,text=True).strip()==M['baseline']
 for tree in (A,S):assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=tree,text=True)==''
 hashes=json.loads((P/'binary-hashes.json').read_text())
 for path,expected in hashes.items():assert digest(path)==expected,path
 assert digest(S/'bench/compare/fixtures/embeddings.vnef')=='4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d'
 assert digest(P/'graph.vndb')=='3672206a43d866b598b2abac71babf71181f7d459a2d73687fc8928d389b1cbc'
 subprocess.run(['bash','bench/compare/scripts/refuse_shared_runner.sh'],cwd=S,env=E,check=True)
 M['binary_hashes']=hashes;M['hash_verification']='PASS';save()
 awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())])
 if PHASE=='probe':
  for i,(arm,order) in enumerate([('A','normal'),('B','normal'),('B','reverse'),('A','reverse')],1):run(f'probe-{i}-{arm}-{order}',[str(P/('probe-'+arm)/'target/release/abi-probe'),'measure',str(P/'graph.vndb'),order])
 elif PHASE=='index-confirm':
  for i,arm in enumerate(['A','B','B','A','B','A','A','B'],1):
   name=f'index-confirm-{i}-{arm}';binary=next(p for p in hashes if '/target-'+arm+'/release/deps/index-' in p);env=E.copy();env['CARGO_TARGET_DIR']=str(O/name);run(name,[binary,'--bench','index_search','--noplot'],A if arm=='A' else S,env)
 elif PHASE=='criterion':
  for i,arm in enumerate(['A','B','A','B','A','B'],1):
   for target in ['distance','store','index','disk']:
    name=f'pass-{i}-{arm}-{target}';binary=next(p for p in hashes if '/target-'+arm+'/release/deps/'+target+'-' in p);env=E.copy();env['CARGO_TARGET_DIR']=str(O/f'pass-{i}-{arm}');run(name,[binary,'--bench','--noplot'],A if arm=='A' else S,env)
 else:
  env=E.copy();env.update(VANEDB_COMPARE_HW='apple-m4-pro',VANEDB_COMPARE_DEDICATED='1')
  run('compare-'+PHASE,[str(S/'bench/compare/target/release/compare'),'run','--fixture',str(S/'bench/compare/fixtures/embeddings.vnef'),'--metric',PHASE,'--rounds','4','--k','10','--ef','16,32,50,100','--m','16','--ef-construction','200','--seed','42','--markdown','--json-out',str(O/(PHASE+'.json')),'--out-dir',str(O/(PHASE+'-indices'))],env=env)
 M['status']='completed';M['completed']=now();M['host_screen_passed']=all(e['host_screen_passed'] for e in M['events']);save()
except BaseException as e:
 M.update(status='blocked' if isinstance(e,RuntimeError) else 'failed',error=repr(e),stopped=now());save();raise
finally:
 if awake:awake.terminate()
