"""Recompute the retained probe analysis without changing the original record."""
from pathlib import Path
import json,statistics,math
O=Path(__file__).parent;r=json.loads((O/'probe-analysis.json').read_text());reference=None;medians={'native':{'A':[],'B':[]},'capi':{'A':[],'B':[]}}
def eq(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
for run in r['runs']:
 rows=[json.loads(l) for l in (O/(run['name']+'.stdout.log')).read_text().splitlines()];checks=[x for x in rows if 'check_query' in x];assert len(checks)==32
 if reference is None:reference=checks
 assert checks==reference and rows[-1]=={'verified':True}
 for path in ('native','capi'):
  rounds=[x for x in rows if x.get('path')==path];assert len(rounds)==10;assert sorted(x['round'] for x in rounds)==list(range(10));assert all(x['warmup']==(x['round']<2) for x in rounds)
  v=[x['nanoseconds']/(x['sweeps']*x['queries_per_sweep']) for x in rounds if not x['warmup']];assert v==run['samples_ns'][path];m=statistics.median(v);eq(m,run['median_ns'][path]);medians[path][run['arm']].append(m)
for path,arms in medians.items():
 summary=r['summary'][path]
 for arm,values in arms.items():
  assert values==summary['process_medians_ns'][arm];eq(statistics.median(values),summary['arm_medians_ns'][arm]);eq((max(values)-min(values))/min(values),summary['spread'][arm])
 eq(statistics.median(arms['B'])/statistics.median(arms['A'])-1,summary['change'])
print('PASS: all raw probe rounds, exact outputs, medians, spreads and deltas reconcile')
