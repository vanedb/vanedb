"""Run arithmetic and source checks only after every measurement has ended."""
from pathlib import Path
import subprocess,json,hashlib,datetime,os,sys
O=Path(__file__).parent;R=O.parent;S=R/'vanedb-020-perf-20261101';A=R/'vanedb-020-perf-baseline-20261101';E=R/'vanedb-020-perf-evidence-20261101';P=R/'020-perf-acceptance-20261101';records=[]
for phase in ('probe','criterion','index-confirm','cosine','l2','android-cosine','android-l2'):assert json.loads((O/(phase+'-campaign.json')).read_text())['status']=='completed',phase
assert 'ended' in json.loads((O/'android-emulator-lifecycle.json').read_text())
def run(name,args,cwd=O,env=None):
 p=subprocess.run(args,cwd=cwd,env=env,text=True,capture_output=True);records.append(dict(name=name,command=args,cwd=str(cwd),exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr));(O/'final-validation.json').write_text(json.dumps(records,indent=2)+'\n');assert p.returncode==0,(name,p.stderr);return p.stdout
for tree,sha in ((S,'e1761dcf4258011b446a361dc529da082ea2fc94'),(A,'51cb2d871538985aad933a13ebe50c13a0031dfa')):
 assert run('source-head-'+tree.name,['git','rev-parse','HEAD'],tree).strip()==sha
 assert run('source-clean-'+tree.name,['git','status','--porcelain','--untracked-files=no'],tree)==''
run('preserved-original-checkout',['git','status','--porcelain'],R/'vanedb')
hashes=json.loads((P/'binary-hashes.json').read_text())
for name,sha in hashes.items():
 with Path(name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha,name
for path,sha in ((S/'bench/compare/fixtures/embeddings.vnef','4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d'),(P/'graph.vndb','3672206a43d866b598b2abac71babf71181f7d459a2d73687fc8928d389b1cbc')):
 with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha
records.append(dict(name='final-binary-fixture-graph-hashes',status='PASS',binary_hashes=hashes));(O/'final-validation.json').write_text(json.dumps(records,indent=2)+'\n')
run('probe-arithmetic',[sys.executable,str(O/'verify_probe.py')])
for stem in ('criterion','index'):
 name='criterion-analysis.json' if stem=='criterion' else 'index-confirm-analysis.json';before=json.loads((O/name).read_text());run(stem+'-arithmetic',[sys.executable,str(O/('analyze_'+stem+'.py'))]);assert json.loads((O/name).read_text())==before
for phase in ('cosine','l2','android-cosine','android-l2'):
 log=phase+'.stderr.log' if phase.startswith('android-') else 'compare-'+phase+'.stderr.log'
 run(phase+'-raw-round-arithmetic',[sys.executable,str(O/'verify_compare.py'),str(O/(phase+'.json')),str(O/log),str(O/(phase+'-verification.json'))])
 rendered=run(phase+'-repository-renderer',[sys.executable,str(S/'bench/compare/scripts/render_comparison_md.py'),str(O/(phase+'.json'))]);out=O/(phase+'-rendered.md')
 if out.exists():assert out.read_text()==rendered,phase
 else:out.write_text(rendered)
run('quality-counter-reconciliation',[sys.executable,str(O/'summarize_quality.py')])
env=os.environ.copy();env['PATH']='/Users/anton/.rustup/toolchains/stable-aarch64-apple-darwin/bin:'+env['PATH']
for name,args in [('python',[sys.executable,'--version']),('rustc',['rustc','-Vv']),('cargo',['cargo','--version']),('clang',['clang','--version']),('cmake',['cmake','--version']),('git',['git','--version']),('gh',['gh','--version'])]:run('tool-version-'+name,args,env=env)
run('workspace-format',['cargo','fmt','--all','--','--check'],E,env)
run('bench-format',['cargo','fmt','--manifest-path','bench/Cargo.toml','--all','--','--check'],E,env)
for name,args in [('utc',['date','-u']),('architecture',['uname','-m']),('os',['sw_vers']),('hardware',['sysctl','hw.model','machdep.cpu.brand_string','hw.physicalcpu','hw.logicalcpu','hw.memsize','kern.hv_vmm_present']),('power',['pmset','-g','batt']),('thermal',['pmset','-g','therm']),('swap',['sysctl','vm.swapusage']),('disk',['df','-k','.'])]:run('host-final-'+name,args)
run('indexer-final-state',['ps','-p','1075','-o','uid=,lstart=,state=,comm='])
print('PASS: source identities, all binary/fixture hashes, raw arithmetic, renderer, load counters and formatting')
