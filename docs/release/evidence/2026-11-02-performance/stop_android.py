"""Verify final guest artifacts, sync, and stop only this task's emulator."""
from pathlib import Path
import subprocess,json,hashlib,datetime
O=Path(__file__).parent;R=O.parent;S=R/'vanedb-020-perf-20261101';SDK=R/'020-android-benchmark-sdk';adb=[str(SDK/'platform-tools/adb'),'-s','emulator-5580'];dest='/data/local/tmp/vanedb-020-perf-20261102';records=[];output=O/'android-shutdown.json';assert not output.exists()
for metric in ('cosine','l2'):assert json.loads((O/('android-'+metric+'-campaign.json')).read_text())['status']=='completed'
life=json.loads((O/'android-emulator-lifecycle.json').read_text());m=json.loads((O/'android-l2-campaign.json').read_text())
def run(name,args):
 p=subprocess.run(args,text=True,capture_output=True);records.append(dict(name=name,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr));output.write_text(json.dumps(records,indent=2)+'\n');assert p.returncode==0,(name,p.stderr);return p.stdout
assert run('emulator-identity',['ps','-p',str(life['pid']),'-o','lstart=,comm='])==m['emulator_identity']
assert 'vanedb_020_acceptance_20261102_api35_arm64' in run('avd-name',adb+['emu','avd','name'])
hashes=run('final-guest-hashes-and-sync',adb+['shell','sync && cd '+dest+' && sha256sum cosine.json l2.json compare libc++_shared.so embeddings.vnef SHA256SUMS metadata.json'])
paths={'cosine.json':O/'android-cosine.json','l2.json':O/'android-l2.json','compare':S/'bench/compare/target/aarch64-linux-android/release/compare','libc++_shared.so':SDK/'ndk/26.1.10909125/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so'}
for n in ('embeddings.vnef','SHA256SUMS','metadata.json'):paths[n]=S/'bench/compare/fixtures'/n
for name,p in paths.items():
 with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()+'  '+name in hashes,name
run('final-guest-capacity',adb+['shell','df -k /data'])
run('stop-task-avd',adb+['emu','kill'])
print('PASS: final guest/host hashes match; synced and stopped the task-owned AVD')
