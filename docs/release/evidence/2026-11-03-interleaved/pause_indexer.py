from pathlib import Path
import subprocess,os,signal,time,json,datetime,sys
O=Path('/private/tmp/vanedb-020-perf-20261103T091447Z');phase=sys.argv[1] if len(sys.argv)>1 else 'probe'
for _ in range(40):
 if (O/(phase+'-host.jsonl')).exists():break
 time.sleep(.25)
sample=json.loads((O/(phase+'-host.jsonl')).read_text().splitlines()[-1]);controller=sample['controller_pid']
rows=[line.split(None,2) for line in subprocess.check_output(['ps','-axo','pid=,uid=,comm='],text=True).splitlines()]
targets=[int(row[0]) for row in rows if len(row)==3 and int(row[1])==os.getuid() and row[2].endswith('/corespotlightd')]
assert len(targets)==1,targets
target=targets[0]
original_state=subprocess.check_output(['ps','-p',str(target),'-o','stat='],text=True).strip();assert 'T' not in original_state,'Do not resume a process already stopped by somebody else'
info=subprocess.check_output(['ps','-p',str(target),'-o','uid=,lstart=,comm='],text=True).strip();assert int(info.split()[0])==os.getuid() and info.endswith('/corespotlightd')
controller_info=subprocess.check_output(['ps','-p',str(controller),'-o','lstart=,comm='],text=True).strip();assert 'Python' in controller_info or 'python' in controller_info
record={'target_pid':target,'original_state':original_state,'target_identity':info,'controller_pid':controller,'controller_identity':controller_info,'method':'SIGSTOP then SIGCONT of the user-owned corespotlightd; no measurement exclusion','started':datetime.datetime.now(datetime.timezone.utc).isoformat()};p=O/(phase+'-indexer-pause.json');assert not p.exists(),'Never overwrite isolation evidence'
def save():p.write_text(json.dumps(record,indent=2)+'\n')
def identity(pid,fmt):
 r=subprocess.run(['ps','-p',str(pid),'-o',fmt],capture_output=True,text=True);return r.stdout.strip() if r.returncode==0 else None
save()
def interrupted(signum, frame):raise KeyboardInterrupt
signal.signal(signal.SIGTERM, interrupted)
try:
 os.kill(target,signal.SIGSTOP);record['suspended']=True;save();deadline=time.monotonic()+10800
 while time.monotonic()<deadline and identity(controller,'lstart=,comm=')==controller_info:time.sleep(2)
finally:
 if identity(target,'uid=,lstart=,comm=')==info:
  os.kill(target,signal.SIGCONT);record['resumed']=True
 else:record['resumed']=False;record['reason']='Original process exited; no signal sent to any reused PID'
 record['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
