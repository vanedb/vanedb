"""Full ARM64 emulator campaigns using prebuilt, hash-verified candidate binaries."""
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess,sys,time
R=Path('/Users/anton/code/codex');O=R/'020-perf-acceptance-20261103T080155Z';P=R/'020-perf-acceptance-20261101';S=R/'vanedb-020-perf-20261101';SDK=R/'020-android-benchmark-sdk'
metric=sys.argv[1];assert metric in ('cosine','l2');phase='android-'+metric;manifest=O/(phase+'-campaign.json');assert not manifest.exists(),'Never overwrite attempts'
adb=[str(SDK/'platform-tools/adb'),'-s','emulator-5580'];base='/data/local/tmp/vanedb-020-perf-20261103T080155Z';dest='/data/local/tmp/vanedb-020-perf-20261103T080155Z';emulator_pid=json.loads((O/'android-emulator-lifecycle.json').read_text())['pid']
M={'phase':phase,'candidate':'e1761dcf4258011b446a361dc529da082ea2fc94','status':'setup','controller_pid':os.getpid(),'emulator_pid':emulator_pid,'avd':'vanedb_020_acceptance_20261102_api35_arm64','hardware':'ARM64 emulator, API 35, HVF, 4 vCPUs, 8192 MiB, 4 KiB guest pages; not a physical device','reservation':'User reserved Mac on 2026-11-03','commands':[],'load_flags':0,'guest_load_flags':0,'power_flags':0,'sampling':'Host every 5 seconds; guest /proc/stat and readable per-process stat approximately every 30 seconds during campaign and every 5 seconds in preflight. Host excludes controller, ps, task emulator tree and current adb tree only. Guest aggregate busy minus measured compare process ticks includes all other guest activity, including sampler overhead. Snapshots do not prove continuous isolation.','screen':'Host max background <=50%, sum <=150%, AC; guest aggregate background <=50% of one virtual CPU and visible individual <=50%; three consecutive quiet preflight samples; no load-failure discard'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():manifest.write_text(json.dumps(M,indent=2)+'\n')
def run(label,args):
 p=subprocess.run(args,text=True,capture_output=True);M['commands'].append(dict(name=label,command=args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr));save()
 if p.returncode:raise RuntimeError(label+': '+p.stderr)
 return p.stdout
last_guest=None
def guest():
 global last_guest
 # Shell builtins avoid hundreds of cat subprocesses. Unreadable proc rows are explicitly counted.
 script='cat /proc/stat; echo UPTIME; cat /proc/uptime; echo PROCESSES; denied=0; for f in /proc/[0-9]*/stat; do if read -r line < "$f" 2>/dev/null; then echo "$line"; else denied=$((denied+1)); fi; done; echo DENIED=$denied; echo TARGET; cat '+dest+'/'+metric+'.pid 2>/dev/null; true'
 p=subprocess.run(adb+['shell',script],capture_output=True,text=True);assert p.returncode==0,p.stderr
 raw=p.stdout;cpu=next(l for l in raw.splitlines() if l.startswith('cpu '));ticks=list(map(int,cpu.split()[1:]));total=sum(ticks[:8]);busy=total-ticks[3]-ticks[4]
 proc={};section=raw.split('PROCESSES\n',1)[1].split('TARGET\n',1);target=int(section[1].strip()) if section[1].strip().isdigit() else None
 for l in section[0].splitlines():
  if not l or not l[0].isdigit() or ')' not in l:continue
  a=l.index('(');b=l.rindex(')');pid=int(l[:a]);x=l[b+2:].split();proc[pid]={'name':l[a+1:b],'start':int(x[19]),'ticks':int(x[11])+int(x[12])}
 rec={'utc':now(),'raw':raw,'stderr':p.stderr,'target_pid':target,'quiet':False,'baseline_only':last_guest is None}
 current={'total':total,'busy':busy,'proc':proc}
 if last_guest:
  dt=total-last_guest['total'];assert dt>0;loads=[];target_ticks=0
  for pid,v in proc.items():
   old=last_guest['proc'].get(pid)
   if old and old['start']==v['start']:
    delta=v['ticks']-old['ticks'];assert delta>=0
    if pid==target:target_ticks=delta
    elif delta:loads.append({'pid':pid,'name':v['name'],'cpu':400*delta/dt})
  background=400*(busy-last_guest['busy']-target_ticks)/dt
  rec.update(background_sum=background,background_max=max((x['cpu'] for x in loads),default=0),visible_process_loads=loads,total_tick_delta=dt,target_tick_delta=target_ticks)
  rec['quiet']=-1<=background<=50 and rec['background_max']<=50
 last_guest=current
 with (O/(phase+'-guest.jsonl')).open('a') as f:f.write(json.dumps(rec)+'\n')
 return rec
def host(stage,adb_pid=None):
 raw=subprocess.check_output(['ps','-axo','pid=,ppid=,pcpu=,comm='],text=True);rows=[]
 for l in raw.splitlines():
  x=l.split(None,3)
  if len(x)==4:rows.append(dict(pid=int(x[0]),ppid=int(x[1]),cpu=float(x[2]),executable=Path(x[3]).name))
 excluded={os.getpid(),emulator_pid};excluded.update([adb_pid] if adb_pid else [])
 while True:
  nxt=excluded|{r['pid'] for r in rows if r['ppid'] in excluded}
  if nxt==excluded:break
  excluded=nxt
 v=[r['cpu'] for r in rows if r['pid'] not in excluded and r['executable']!='ps'];power=subprocess.check_output(['pmset','-g','batt'],text=True)
 rec=dict(utc=now(),stage=stage,controller_pid=os.getpid(),excluded_pids=sorted(excluded),processes=[r for r in rows if r['cpu']>0],zero_cpu_processes_omitted=sum(r['cpu']==0 for r in rows),power=power,ac_power="'AC Power'" in power,background_max=max(v,default=0),background_sum=sum(v),quiet=max(v,default=0)<=50 and sum(v)<=150)
 with (O/(phase+'-host.jsonl')).open('a') as f:f.write(json.dumps(rec)+'\n')
 return rec
awake=None;p=None
try:
 M['started']=now();save()
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=S,text=True).strip()==M['candidate']
 assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=S,text=True)==''
 run('shared-host-check',['bash',str(S/'bench/compare/scripts/refuse_shared_runner.sh')])
 identity=run('emulator-identity',['ps','-p',str(emulator_pid),'-o','lstart=,comm=']);assert 'qemu-system-aarch64-headless' in identity;M['emulator_identity']=identity
 assert run('boot',adb+['shell','getprop sys.boot_completed']).strip()=='1'
 facts=run('guest-facts',adb+['shell','uname -a; getprop ro.product.cpu.abi; getprop ro.build.version.sdk; getconf PAGESIZE; getconf CLK_TCK; nproc; getprop ro.kernel.qemu; head -4 /proc/meminfo; df -k /data']);assert '\narm64-v8a\n35\n4096\n100\n4\n1\n' in facts
 disk=run('capacity',adb+['shell','df -k /data']);available=int(disk.splitlines()[-1].split()[3])*1024;required=4100000000 if metric=='cosine' else 6800000000;assert available>=required,(available,required)
 hashes=run('reused-device-hashes',adb+['shell','cd '+base+' && sha256sum compare libc++_shared.so embeddings.vnef SHA256SUMS metadata.json'])
 paths={'compare':S/'bench/compare/target/aarch64-linux-android/release/compare','libc++_shared.so':SDK/'ndk/26.1.10909125/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so'}
 for n in ('embeddings.vnef','SHA256SUMS','metadata.json'):paths[n]=S/'bench/compare/fixtures'/n
 expected=json.loads((P/'binary-hashes.json').read_text());assert hashlib.sha256(paths['compare'].read_bytes()).hexdigest()==expected[str(paths['compare'])]
 for n,f in paths.items():
  with f.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
  assert sha+'  '+n in hashes,n
 run('new-output-directory',adb+['shell','mkdir -p '+dest+' && test ! -e '+dest+'/'+metric+'.json && test ! -e '+dest+'/'+metric+'-indices'])
 awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())]);M['status']='preflight';save();deadline=time.monotonic()+600;streak=0
 while time.monotonic()<deadline:
  h=host('preflight');g=guest();assert h['ac_power'],'AC required';streak=streak+1 if h['quiet'] and g['quiet'] else 0
  print(now(),'preflight',h['background_max'],round(h['background_sum'],1),'guest',g.get('background_sum'),'streak',streak,flush=True)
  if streak>=3:break
  time.sleep(5)
 else:raise RuntimeError('Host/guest failed to settle within ten minutes; no bypass')
 command='cd '+base+' && echo $$ > '+dest+'/'+metric+'.pid && exec env LD_LIBRARY_PATH='+base+' VANEDB_COMPARE_HW=android-arm64-emulator VANEDB_COMPARE_DEDICATED=1 ./compare run --fixture embeddings.vnef --metric '+metric+' --rounds 4 --k 10 --ef 16,32,50,100 --m 16 --ef-construction 200 --seed 42 --out-dir '+dest+'/'+metric+'-indices --json-out '+dest+'/'+metric+'.json'
 M.update(status='running',measurement_started=now(),measurement_command=adb+['shell',command]);save()
 with (O/(phase+'.stdout.log')).open('w') as out,(O/(phase+'.stderr.log')).open('w') as err:
  p=subprocess.Popen(adb+['shell',command],stdout=out,stderr=err,start_new_session=True);M['adb_pid']=p.pid;save();time.sleep(1);last_guest=None;initial_guest=guest();assert initial_guest['target_pid'] is not None;nextguest=time.monotonic()+30
  while True:
   h=host('campaign',p.pid);M['load_flags']+=not h['quiet'];M['power_flags']+=not h['ac_power']
   if not h['ac_power']:raise RuntimeError('AC lost during measurement; nonqualifying')
   if time.monotonic()>=nextguest:
    g=guest();M['guest_load_flags']+=not g['quiet'];nextguest=time.monotonic()+30
   save()
   try:code=p.wait(timeout=5);break
   except subprocess.TimeoutExpired:pass
 M.update(exit_code=code,measurement_ended=now());save();assert code==0,code
 run('pull-result',adb+['pull',dest+'/'+metric+'.json',str(O/(phase+'.json'))])
 # Only this attempt's disposable indices are removed, after hashing each.
 listing=run('index-hashes',adb+['shell','cd '+dest+'/'+metric+'-indices && sha256sum *.idx && du -k *.idx'])
 M['index_disposition']='Task-generated disposable indices hashed then removed; existing guest files untouched'
 run('remove-own-indices',adb+['shell','rm -f '+dest+'/'+metric+'-indices/*-'+metric+'-r*.idx && rmdir '+dest+'/'+metric+'-indices'])
 run('remaining-capacity',adb+['shell','df -k /data'])
 M.update(status='completed',completed=now(),host_screen_passed=M['load_flags']==M['power_flags']==0,guest_screen_passed=M['guest_load_flags']==0);save();print(json.dumps({k:M[k] for k in ('status','load_flags','guest_load_flags','power_flags')}),flush=True)
except BaseException as e:
 M.update(status='blocked',error=repr(e),stopped=now());save()
 if p and p.poll() is None:
  subprocess.run(adb+['shell','if test -f '+dest+'/'+metric+'.pid; then p=$(cat '+dest+'/'+metric+'.pid); test "$(cat /proc/$p/comm)" = compare && kill -TERM "$p"; fi'],capture_output=True)
  p.terminate();p.wait()
 raise
finally:
 if awake:awake.terminate()
