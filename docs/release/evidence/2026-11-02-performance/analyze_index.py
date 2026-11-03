from pathlib import Path
import json,statistics
O=Path(__file__).parent;m=json.loads((O/'index-confirm-campaign.json').read_text());runs=[]
for e in m['events']:
 values={}
 for p in (O/e['name']/'criterion').glob('**/new/benchmark.json'):
  b=json.loads(p.read_text());v=json.loads(p.with_name('estimates.json').read_text());values[b['full_id']]=(v.get('slope') or v['mean'])['point_estimate']
 assert set(values)=={'index_search/rs','index_search/cpp'};runs.append({'name':e['name'],'arm':e['name'][-1],'values_ns':values,'host_screen_passed':e['host_screen_passed'],'load_flags':e['load_flags']})
def summary(rs):
 out={}
 for key in ['index_search/rs','index_search/cpp']:
  arms={a:[r['values_ns'][key] for r in rs if r['arm']==a] for a in ['A','B']};med={a:statistics.median(v) for a,v in arms.items()};spread={a:(max(v)-min(v))/min(v) for a,v in arms.items()};delta=med['B']/med['A']-1;out[key]={'samples_ns':arms,'median_ns':med,'spread':spread,'delta':delta,'exceeds_floor_and_spreads':abs(delta)>max(.03,*spread.values())}
 return {'host_screen_passed':all(r['host_screen_passed'] for r in rs),'rows':out,'normalized_delta':(1+out['index_search/rs']['delta'])/(1+out['index_search/cpp']['delta'])-1}
r={'runs':runs,'all_eight':summary(runs),'planned_abba_block':summary(runs[:4]),'planned_baab_block':summary(runs[4:]),'scope':'Focused index_search confirmation; separate fixed-graph native/CAPI probe retained independently'};(O/'index-confirm-analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['runs','scope']},indent=2))
