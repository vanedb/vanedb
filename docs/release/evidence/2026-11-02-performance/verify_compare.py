from pathlib import Path
import json,statistics,math,sys
report,log,out=map(Path,sys.argv[1:]);r=json.loads(report.read_text());raw=[json.loads(l.removeprefix('RAW_ROUNDS ')) for l in log.read_text().splitlines() if l.startswith('RAW_ROUNDS ')]
expected={'vanedb','usearch','hnswlib','instant-distance','hnsw_rs'}|({'sqlite-vec'} if r['metric']=='l2' else set())
assert {x['engine'] for x in raw}=={x['engine'] for x in r['results']}==expected
assert len(raw)==len(expected)
assert (r['fixture_n_docs'],r['fixture_n_queries'],r['queries_measured'],r['fixture_dim'],r['k'])==(100000,1000,1000,768,10)
assert r['fixture_sha256']=='4676911521f13dc6592d102cf3b4fcace85c6b62b3011e9bd57557dfc9c01f2d' and r['fixture_role']=='publish'
assert r['params']=={'m':16,'ef_construction':200,'seed':42}
def eq(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
def spread(v):return (max(v)-min(v))/min(v)
for x in r['results']:
 a=next(v for v in raw if v['engine']==x['engine']);assert x['rounds']==len(a['build_secs'])==4;eq(statistics.median(a['build_secs']),x['build_secs_median']);eq(spread(a['build_secs']),x['build_secs_spread']);assert x['delete_ok'] is not False
 if x['engine']=='vanedb':assert x['delete_ok'] is True
 efs={50} if x['engine'] in ('instant-distance','sqlite-vec') else {16,32,50,100}
 assert {v['ef'] for v in x['latency_ns_by_ef']}=={v['ef'] for v in x['recall_at_k_by_ef']}==efs
 assert set(a['latency_ns_by_ef'])==set(a['recall_at_k_by_ef'])=={str(v) for v in efs}
 for row in x['latency_ns_by_ef']:
  v=a['latency_ns_by_ef'][str(row['ef'])];assert len(v)==4;eq(statistics.median(v),row['median_ns']);eq(spread(v),row['spread'])
 for row in x['recall_at_k_by_ef']:
  v=a['recall_at_k_by_ef'][str(row['ef'])];assert len(v)==4;eq(statistics.median(v),row['recall_median'])
result={'status':'PASS','scope':'Report completeness and raw-round arithmetic only; host qualification is separate','metric':r['metric'],'engines':sorted(expected),'rounds':4,'queries':1000,'host_os':r['host_os'],'host_arch':r['host_arch'],'shared_runner':r['shared_runner'],'dedicated_attested':r['dedicated_attested']};out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
