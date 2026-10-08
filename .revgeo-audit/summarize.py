import json,random,statistics
from pathlib import Path
root=Path('/tmp/revgeo-audit-evidence')
r=json.loads((root/'results.json').read_text())
assert len(r['browsers'])==3
report={'seed':20261008,'bootstrap_resamples':10000,'confidence':0.99,'unit':'ms per workload operation','metrics':[]}
for browser,data in r['browsers'].items():
 assert data['complete']
 assert set(data['samples'])==set(r['contract']['changedWorkloads'])
 for workload,(before,after) in data['samples'].items():
  assert len(before)==len(after)==40
  b,a=map(statistics.median,[before,after])
  deltas=[y-x for x,y in zip(before,after)]
  rng=random.Random(20261008)
  boots=sorted(statistics.median(rng.choices(deltas,k=40)) for _ in range(10000))
  upper=boots[9899];allowed=b*.05+.05
  report['metrics'].append({'browser':browser,'workload':workload,'baseline_median':b,'final_median':a,'baseline_p95':sorted(before)[37],'final_p95':sorted(after)[37],'paired_median_delta':statistics.median(deltas),'delta_upper_99':upper,'allowed_delta':allowed,'pass':upper<=allowed})
report['checks']=sum(len(d['checks']) for d in r['browsers'].values())
report['total_gates']=len(report['metrics'])
report['passed_gates']=sum(m['pass'] for m in report['metrics'])
(root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print('AUDIT_SUMMARY '+json.dumps(report,separators=(',',':')))
if report['passed_gates']!=report['total_gates']:raise SystemExit('protected performance gate failed')
