from pathlib import Path
import json,statistics
O=Path(__file__).parent;m=json.loads((O/'criterion-campaign.json').read_text());assert m['status']=='completed';passes=[]
for i,arm in enumerate(['A','B','A','B','A','B'],1):
 rows={}
 for p in (O/f'pass-{i}-{arm}/criterion').glob('**/new/benchmark.json'):
  b=json.loads(p.read_text());e=json.loads(p.with_name('estimates.json').read_text());sample=json.loads(p.with_name('sample.json').read_text());estimate=e.get('slope') or e['mean'];rows[b['full_id']]={'point_estimate_ns':estimate['point_estimate'],'sample_count':len(sample['times'])}
 assert len(rows)==28,(i,len(rows));passes.append({'pass':i,'arm':arm,'rows':rows})
keys=set(passes[0]['rows']);assert all(set(p['rows'])==keys for p in passes)
rows={}
for key in sorted(keys):
 samples={a:[p['rows'][key]['point_estimate_ns'] for p in passes if p['arm']==a] for a in ['A','B']};medians={a:statistics.median(v) for a,v in samples.items()};spreads={a:(max(v)-min(v))/min(v) for a,v in samples.items()};delta=medians['B']/medians['A']-1;rows[key]={'samples_ns':samples,'median_ns':medians,'spread':spreads,'delta':delta,'exceeds_floor_and_both_spreads':abs(delta)>max(.03,*spreads.values())}
ratios={}
for k in rows:
 if k.endswith('/rs'):
  cpp=k[:-3]+'/cpp';r=rows[k];c=rows[cpp];ratios[k[:-3]]={'candidate_rs_over_cpp':r['median_ns']['B']/c['median_ns']['B'],'rust_delta':r['delta'],'cpp_delta':c['delta'],'ratio_of_ratios_delta':(r['median_ns']['B']/r['median_ns']['A'])/(c['median_ns']['B']/c['median_ns']['A'])-1}
host=[json.loads(l) for l in (O/'criterion-host.jsonl').read_text().splitlines()];measured=[s for s in host if not s['stage'].endswith('preflight')]
result={'candidate':m['candidate'],'baseline':m['baseline'],'method':'Median of three per-pass Criterion slope point estimates (mean when slope absent); no timing rounds excluded','passes':passes,'rows':rows,'ratios':ratios,'host':{'measured_samples':len(measured),'load_flags':sum(not s['quiet'] for s in measured),'power_flags':sum(not s['ac_power'] for s in measured),'all_events_completed':len(m['events'])==24 and all(e['exit_code']==0 for e in m['events']),'screen_passed':m['host_screen_passed']},'qualifying':False,'qualification_reason':'Recorded background-load screen failures; retain as diagnostic evidence, not an acceptance PASS'}
(O/'criterion-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'host':result['host'],'ratios':ratios},indent=2))

quality={}
for target in ['distance','store','index','disk']:
 events=[e for e in m['events'] if e['name'].endswith('-'+target)]
 samples=[s for s in measured if s['stage'].endswith('-'+target)]
 quality[target]={'planned_runs':6,'completed_runs':sum(e.get('exit_code')==0 for e in events),'load_flags':sum(not s['quiet'] for s in samples),'samples':len(samples),'power_flags':sum(not s['ac_power'] for s in samples),'complete_block_screen_passed':len(events)==6 and all(e.get('exit_code')==0 and e['host_screen_passed'] for e in events)}
result['target_quality']=quality
result['flagged_samples']=[{'utc':s['utc'],'stage':s['stage'],'background_max':s['background_max'],'background_sum':s['background_sum'],'offenders':[x for x in s['processes'] if x['pid'] not in s['excluded_pids'] and x['cpu']>50]} for s in measured if not s['quiet']]
result['scope']='Pinned e1761dc candidate only; not final shipping source. Full preplanned ABABAB, every round retained. No target block or individual round promoted to acceptance on a subset.'
(O/'criterion-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'target_quality':quality},indent=2))
