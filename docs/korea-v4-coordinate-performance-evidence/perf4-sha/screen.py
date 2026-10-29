import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
OLD = ROOT / 'evidence/perf4-rust'
PLAN = WORK / 'screen-plan.json'
RESULT = WORK / 'screen.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cpu_stat():
    return Path('/sys/fs/cgroup/cpu.stat').read_text()


def save():
    RESULT.write_text(json.dumps(report, indent=2) + '\n')


if '--run' not in sys.argv:
    raise SystemExit('Prepared only: pass --run with exclusive CPU GO.')
assert not RESULT.exists()
candidate_sha = digest(WORK / 'candidate.rs')
assert candidate_sha == '90b580b4ff5c70529b7127a47742246141d39112add1efd361a9b10be092d8ef'
clippy = {}
for target in ['host', 'windows', 'intel-mac', 'arm-mac']:
    record = json.loads((WORK / f'clippy-final-{target}.json').read_text())
    assert record['exit_code'] == 0 and record['source_sha256'] == candidate_sha and record['diagnostic_count'] == 323
    clippy[target] = {'exit_code':0, 'diagnostic_count':323, 'evidence':f'clippy-final-{target}.json'}
build = json.loads((WORK / 'build.json').read_text())
assert all(row['exit_code'] == 0 for row in build['commands'])
assert build['source_sha256']['candidate.rs'] == candidate_sha
for path in (ROOT / 'evidence/perf4/before/src').rglob('*'):
    if path.is_file():
        name = path.relative_to(ROOT / 'evidence/perf4/before')
        if str(name) == 'src/korea_boundary/hash.rs':
            assert digest(WORK / 'probe' / name) == candidate_sha
        elif str(name) != 'src/main.rs':
            assert path.read_bytes() == (WORK / 'probe' / name).read_bytes()
assert (WORK / 'probe/src/main.rs').read_bytes() == (OLD / 'baseline/src/main.rs').read_bytes()
plan = {'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'candidate_source_sha256':candidate_sha, 'baseline_hash_sha256':digest(WORK / 'baseline.rs'),
        'candidate_scope':'SHA rounds-only acceleration, baseline inflater unchanged; independent of provisional bits+Adler',
        'counts':{'correctness':'All97 data files plus1034 known/boundary/random messages and4096 arbitrary state-block records; both hardware dispatch and forced scalar compared to unchanged scalar, and digests to Python hashlib',
                  'resident_sha_diagnostic':'Exactly one baseline then one candidate, all97 original files, resident input bytes; excludes file I/O and vector validation',
                  'whole_fixed10000_smoke':'Exactly one baseline then one candidate fresh process with identical num/supp corpus; output bits and supp counts must equal existing oracle'},
        'status':'One-pair screen only, not formal qualification',
        'no_additional_variants_or_repeats':True,
        'baseline_inflater_sha256':digest(WORK / 'probe/src/boundary_inflate.rs'),
        'all_other_non_main_product_modules_match_frozen_baseline':True,
        'clippy':clippy}
assert not PLAN.exists()
PLAN.write_text(json.dumps(plan,indent=2)+'\n')
report = {'plan_sha256':digest(PLAN),'source_sha256':candidate_sha,
          'validation':None,'resident_sha':[],'whole_fixed10000':[],
          'compiler':build['compiler'],'all_four_target_clippy':clippy,
          'source_binary_link':build,'script_sha256':digest(Path(__file__)),
          'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip()}
save()
validation = subprocess.run([str(WORK/'bench'),'validate',str(WORK/'inputs.tsv'),str(WORK/'states.bin')],capture_output=True,text=True)
(WORK/'validation.stdout').write_text(validation.stdout)
(WORK/'validation.stderr').write_text(validation.stderr)
assert validation.returncode == 0, validation.stderr
report['validation'] = json.loads(validation.stdout)
assert report['validation']['runtime_sha'] and report['validation']['runtime_sse2']
save()
for mode in ['baseline','candidate']:
    before = cpu_stat()
    result = subprocess.run([str(WORK/'bench'),mode,str(WORK/'inputs.tsv')],capture_output=True,text=True)
    after = cpu_stat()
    assert result.returncode == 0, result.stderr
    row = json.loads(result.stdout)
    row.update({'cpu_stat_before':before,'cpu_stat_after':after,'binary_sha256':digest(WORK/'bench')})
    report['resident_sha'].append(row)
    save()
    print(json.dumps(row),flush=True)
expected = (OLD/'expected-10000.txt').read_bytes()
for mode,binary in [('baseline',OLD/'baseline/probe'),('candidate',WORK/'probe/probe')]:
    metrics = WORK/f'whole-{mode}.metrics.json'
    output = WORK/f'whole-{mode}.txt'
    stderr = WORK/f'whole-{mode}.stderr'
    with (OLD/'corpus-10000.txt').open('rb') as source, output.open('wb') as out, stderr.open('wb') as err:
        before = cpu_stat()
        result = subprocess.run([str(OLD/'measure'),str(metrics),str(binary)],stdin=source,stdout=out,stderr=err,cwd=ROOT)
        after = cpu_stat()
    actual = output.read_bytes()
    assert result.returncode == 0 and actual == expected and stderr.stat().st_size == 0
    row=json.loads(metrics.read_text())
    row.update({'variant':mode,'cpu_stat_before':before,'cpu_stat_after':after,
                'all_coordinate_bits_and_supp_exact':True,'output_sha256':hashlib.sha256(actual).hexdigest(),
                'binary_sha256':digest(binary)})
    report['whole_fixed10000'].append(row)
    save()
    print(json.dumps(row),flush=True)
report['resident_sha_time_reduction_percent'] = 100*(1-report['resident_sha'][1]['nanoseconds']/report['resident_sha'][0]['nanoseconds'])
report['whole_fixed10000_time_reduction_percent'] = 100*(1-report['whole_fixed10000'][1]['wall_seconds']/report['whole_fixed10000'][0]['wall_seconds'])
report['qualification'] = 'Not run; independent correctness and one-pair screen only'
report['product_files_modified'] = []
report['source_unchanged_after_screen'] = digest(WORK/'candidate.rs') == candidate_sha
report['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
save()
print(json.dumps({key:report[key] for key in ['validation','resident_sha_time_reduction_percent','whole_fixed10000_time_reduction_percent','qualification','source_unchanged_after_screen']},indent=2),flush=True)
