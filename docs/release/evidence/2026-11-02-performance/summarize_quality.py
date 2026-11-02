"""Reconcile phase counters to retained host/guest samples and non-overlap."""
from pathlib import Path
import datetime,json
O=Path(__file__).parent;phases=['probe','criterion','index-confirm','cosine','l2','android-cosine','android-l2'];result={};intervals=[]
for phase in phases:
 m=json.loads((O/(phase+'-campaign.json')).read_text());assert m['status']=='completed',phase
 rows=[json.loads(l) for l in (O/(phase+'-host.jsonl')).read_text().splitlines()]
 if phase.startswith('android-'):
  measured=[x for x in rows if x['stage']=='campaign'];start=m['measurement_started'];end=m['measurement_ended'];flags=m['load_flags'];power=m['power_flags']
 else:
  measured=[x for x in rows if not x['stage'].endswith('preflight')];start=m['events'][0]['started'];end=m['events'][-1]['ended'];flags=sum(x['load_flags'] for x in m['events']);power=sum(x['power_flags'] for x in m['events']);assert all(x['exit_code']==0 for x in m['events'])
 assert flags==sum(not x['quiet'] for x in measured) and power==sum(not x['ac_power'] for x in measured)
 for x in rows:
  v=[p['cpu'] for p in x['processes'] if p['pid'] not in x['excluded_pids'] and p['executable']!='ps'];assert abs(sum(v)-x['background_sum'])<1e-8;assert max(v,default=0)==x['background_max'];assert x['quiet']==(max(v,default=0)<=50 and sum(v)<=150);assert x['ac_power']==("'AC Power'" in x['power'])
 r=dict(host_process_phase_samples=len(measured),host_load_flags=flags,power_flags=power,host_screen_passed=flags==power==0,preflight_samples=len(rows)-len(measured),preflight_load_flags=sum(not x['quiet'] for x in rows if x not in measured),start=start,end=end,duration_seconds=(datetime.datetime.fromisoformat(end)-datetime.datetime.fromisoformat(start)).total_seconds())
 if phase.startswith('android-'):
  guest=[json.loads(l) for l in (O/(phase+'-guest.jsonl')).read_text().splitlines()];g=[x for x in guest if start<=x['utc']<=end and not x['baseline_only']];assert sum(not x['quiet'] for x in g)==m['guest_load_flags'];r.update(guest_interval_samples=len(g),guest_load_flags=m['guest_load_flags'],guest_screen_passed=m['guest_screen_passed'])
 guard=json.loads((O/(phase+'-indexer-pause.json')).read_text());assert guard['resumed'];r['indexer_restored']=True;result[phase]=r;intervals.append((start,end,phase))
ordered=sorted(intervals)
for first,second in zip(ordered,ordered[1:]):assert first[1]<=second[0],(first,second)
result['method']='Counts apply to complete benchmark process phases, including setup/warmup; preflight failures are separately retained. Snapshot screens do not prove continuous isolation. Guest sampling is interval-based, approximately 30 seconds; its non-target busy CPU includes benchmark-driven kernel/interrupt work and does not by itself prove competing guest activity. Every campaign measurement interval is sequential and non-overlapping.'
(O/'quality-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
