from pathlib import Path
import subprocess,json,time,hashlib
R=Path('/Users/anton/code/codex');O=R/'020-perf-acceptance-20261102';S=R/'vanedb-020-perf-20261101';sdk=R/'020-android-benchmark-sdk';adb=[str(sdk/'platform-tools/adb'),'-s','emulator-5580'];dest='/data/local/tmp/vanedb-020-perf-20261102';records=[]
def run(label,args):
 p=subprocess.run(args,capture_output=True,text=True);records.append(dict(name=label,command=args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr));(O/'android-setup.json').write_text(json.dumps(records,indent=2)+'\n');assert p.returncode==0,(label,p.stderr);return p.stdout
for i in range(120):
 p=subprocess.run(adb+['shell','getprop sys.boot_completed'],text=True,capture_output=True)
 if p.returncode==0 and p.stdout.strip()=='1':break
 time.sleep(5)
else:raise RuntimeError('Boot timeout')
run('mkdir',adb+['shell','mkdir -p '+dest]);files={'compare':S/'bench/compare/target/aarch64-linux-android/release/compare','libc++_shared.so':sdk/'ndk/26.1.10909125/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so'}
for n in ('embeddings.vnef','metadata.json','SHA256SUMS'):files[n]=S/'bench/compare/fixtures'/n
for n,p in files.items():run('push-'+n,adb+['push',str(p),dest+'/'+n])
run('chmod-and-sync',adb+['shell','chmod 700 '+dest+'/compare && sync'])
hashes=run('hashes',adb+['shell','cd '+dest+' && sha256sum compare libc++_shared.so embeddings.vnef metadata.json SHA256SUMS'])
for n,p in files.items():
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()+'  '+n in hashes
run('capacity',adb+['shell','df -k /data']);print('Setup and all hashes PASS',flush=True)
