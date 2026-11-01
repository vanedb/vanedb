import pathlib, subprocess, os, json, datetime, hashlib
R=pathlib.Path('/Users/anton/code/codex'); O=R/'020-perf-acceptance-20261101'; S=R/'vanedb-020-perf-20261101'
T=pathlib.Path('/Users/anton/.rustup/toolchains/stable-aarch64-apple-darwin/bin'); E=os.environ.copy(); E['PATH']=str(T)+':'+E['PATH']; E['RUSTC']=str(T/'rustc'); E.pop('CARGO_TARGET_DIR',None)
records=json.loads((O/'preparation.json').read_text())
def run(label,args,env=E,cwd=S):
 rec={'name':label,'command':list(map(str,args)),'cwd':str(cwd),'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}; records.append(rec)
 with (O/(label+'.log')).open('w') as log:
  p=subprocess.run(args,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
 rec['exit_code']=p.returncode; rec['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat(); (O/'preparation.json').write_text(json.dumps(records,indent=2)+'\n');print(label,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
run('test-compare',[str(T/'cargo'),'test','--locked','--offline','--manifest-path','bench/compare/Cargo.toml'])
run('build-compare',[str(T/'cargo'),'build','--release','--locked','--offline','--manifest-path','bench/compare/Cargo.toml'])
run('smoke-compare',[str(S/'bench/compare/target/release/compare'),'run','--fixture','bench/compare/fixtures/smoke.vnef','--allow-smoke','--rounds','2','--max-queries','4','--json-out',str(O/'smoke.json'),'--out-dir',str(O/'smoke-out')])
ndk=R/'020-android-benchmark-sdk/ndk/26.1.10909125'; b=ndk/'toolchains/llvm/prebuilt/darwin-x86_64/bin'; cc=b/'aarch64-linux-android24-clang'; ae=E.copy();ae.update(ANDROID_NDK_HOME=str(ndk),CC_aarch64_linux_android=str(cc),CXX_aarch64_linux_android=str(cc)+'++',AR_aarch64_linux_android=str(b/'llvm-ar'),CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=str(cc))
run('build-android',[str(T/'cargo'),'build','--release','--locked','--offline','--manifest-path','bench/compare/Cargo.toml','--target','aarch64-linux-android','-j','4'],ae)
run('check-android-elf',['python3','scripts/check_android_elf.py','bench/compare/target/aarch64-linux-android/release/compare'])
