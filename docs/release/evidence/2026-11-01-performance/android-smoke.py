from pathlib import Path
import subprocess,json,hashlib,datetime
R=Path('/Users/anton/code/codex');S=R/'vanedb-020-perf-20261101';O=R/'020-perf-acceptance-20261101';sdk=R/'020-android-benchmark-sdk';adb=[str(sdk/'platform-tools/adb'),'-s','emulator-5580'];dest='/data/local/tmp/vanedb-020-perf-20261101';records=[]
def run(label,args):
 p=subprocess.run(args,text=True,capture_output=True);records.append({'name':label,'command':args,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr});(O/'android-setup.json').write_text(json.dumps(records,indent=2)+'\n');print(label,p.returncode,flush=True)
 if p.returncode:raise RuntimeError(label)
 return p.stdout
run('mkdir',adb+['shell','mkdir','-p',dest])
files={'compare':S/'bench/compare/target/aarch64-linux-android/release/compare','libc++_shared.so':sdk/'ndk/26.1.10909125/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so'}
for name in ['embeddings.vnef','smoke.vnef','SHA256SUMS','metadata.json']:files[name]=S/'bench/compare/fixtures'/name
for name,p in files.items():run('push-'+name,adb+['push',str(p),dest+'/'+name])
run('chmod',adb+['shell','chmod','700',dest+'/compare'])
run('device-facts',adb+['shell','uname -a; getprop ro.product.cpu.abi; getprop ro.build.version.sdk; getconf PAGESIZE; cat /proc/meminfo | head -4; getprop ro.kernel.qemu'])
hashes=run('device-hashes',adb+['shell','cd '+dest+' && sha256sum compare libc++_shared.so embeddings.vnef smoke.vnef SHA256SUMS metadata.json'])
for name,p in files.items():assert hashlib.sha256(p.read_bytes()).hexdigest()+'  '+name in hashes
for metric in ['cosine','l2']:
 command='cd '+dest+' && LD_LIBRARY_PATH='+dest+' VANEDB_COMPARE_HW=android-arm64-emulator ./compare run --fixture smoke.vnef --allow-smoke --rounds 2 --max-queries 4 --metric '+metric+' --out-dir '+dest+'/smoke-out --json-out '+dest+'/smoke-'+metric+'.json'
 run('smoke-'+metric,adb+['shell',command]);run('pull-'+metric,adb+['pull',dest+'/smoke-'+metric+'.json',str(O/('android-smoke-'+metric+'.json'))])
run('stop-task-avd',adb+['emu','kill'])
