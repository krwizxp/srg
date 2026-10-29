import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
OLD = ROOT / 'evidence/perf4-rust'
DIRECTORY = WORK / 'primary-runs'
RESULT = WORK / 'primary-qualification.json'
DIRECTORY.mkdir(exist_ok=False)
plan = json.loads((WORK / 'qualification-plan.json').read_text())
baseline_build = next(row for row in json.loads((OLD / 'build.json').read_text()) if row['variant'] == 'baseline')
command = baseline_build['command']
with (WORK / 'formal-probe-build.log').open('wb') as output:
    compilation = subprocess.run(command,cwd=WORK/'formal-probe',stdout=output,stderr=subprocess.STDOUT)
assert compilation.returncode == 0
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(WORK/'formal-probe/src/korea_boundary/hash.rs') == plan['candidate_hash_sha256']
report = {'plan_sha256':digest(WORK/'qualification-plan.json'),'compile_command':command,
          'compile_cwd':str((WORK/'formal-probe').relative_to(ROOT)),'compile_exit_code':0,
          'baseline_compile_command_identical':True,
          'source_sha256':plan['candidate_hash_sha256'],
          'binaries':{'before':{'path':str((OLD/'baseline/probe').relative_to(ROOT)),'sha256':digest(OLD/'baseline/probe')},
                      'after':{'path':str((WORK/'formal-probe/probe').relative_to(ROOT)),'sha256':digest(WORK/'formal-probe/probe')}},
          'runs':[],'passed':False,'screen_samples_included':0}
assert report['binaries']['before']['sha256'] == baseline_build['probe_sha256']
expected = (OLD/'expected-10000.txt').read_bytes()
for iteration in range(1,6):
    for variant in ['before','after'] if iteration%2 else ['after','before']:
        label=f'{variant}-{iteration}'
        metric=DIRECTORY/f'{label}.metrics.json'
        output=DIRECTORY/f'{label}.txt'
        errors=DIRECTORY/f'{label}.stderr'
        binary=ROOT/report['binaries'][variant]['path']
        with (OLD/'corpus-10000.txt').open('rb') as source,output.open('wb') as out,errors.open('wb') as err:
            cpu_before=Path('/sys/fs/cgroup/cpu.stat').read_text()
            result=subprocess.run([str(OLD/'measure'),str(metric),str(binary)],stdin=source,stdout=out,stderr=err,cwd=ROOT)
            cpu_after=Path('/sys/fs/cgroup/cpu.stat').read_text()
        raw=output.read_bytes()
        row=json.loads(metric.read_text())
        row.update({'variant':variant,'iteration':iteration,'binary_sha256':report['binaries'][variant]['sha256'],
                    'all_coordinate_bits_and_supp_exact':raw==expected,'output_sha256':hashlib.sha256(raw).hexdigest(),
                    'cpu_stat_before':cpu_before,'cpu_stat_after':cpu_after,'stderr_bytes':errors.stat().st_size})
        report['runs'].append(row)
        RESULT.write_text(json.dumps(report,indent=2)+'\n')
        assert result.returncode==0 and raw==expected and row['stderr_bytes']==0
        print(json.dumps({k:row[k] for k in ['variant','iteration','wall_seconds','max_rss_kib']}),flush=True)
before=[row['wall_seconds'] for row in report['runs'] if row['variant']=='before']
after=[row['wall_seconds'] for row in report['runs'] if row['variant']=='after']
logs=[math.log(b/a) for a,b in zip(before,after)]
mean=statistics.mean(logs)
half=2.7764451051977987*statistics.stdev(logs)/math.sqrt(5)
faster=sum(b<a for a,b in zip(before,after))
report['comparison']={'before_seconds':before,'after_seconds':after,'faster_pairs':faster,
                      'before_median_seconds':statistics.median(before),'after_median_seconds':statistics.median(after),
                      'median_reduction_percent':100*(1-statistics.median(after)/statistics.median(before)),
                      'geometric_mean_ratio':math.exp(mean),'paired_log_time_95_percent_t_interval':[math.exp(mean-half),math.exp(mean+half)]}
report['passed']=faster>=4 and math.exp(mean+half)<1
report['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
RESULT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'comparison':report['comparison']},indent=2),flush=True)
if not report['passed']:
    raise SystemExit(2)
