import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
FINAL = WORK / 'final-native'
PREVIOUS = ROOT / 'evidence/perf2-native-512k'
OUTPUT = FINAL / 'fixed-runs'
RESULT = FINAL / 'fixed-probe-validation.json'
CRITICAL = 2.0638985616280205
OLD = ROOT / 'evidence/perf4-rust'
VARIANTS = {'before': OLD / 'baseline', 'after': WORK / 'formal-probe'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save():
    RESULT.write_text(json.dumps(report, indent=2) + '\n')


def paired(before, after):
    logs = [math.log(b / a) for a, b in zip(before, after)]
    deltas = [b - a for a, b in zip(before, after)]
    log_mean = statistics.mean(logs)
    log_half = CRITICAL * statistics.stdev(logs) / math.sqrt(len(logs))
    delta_mean = statistics.mean(deltas)
    delta_half = CRITICAL * statistics.stdev(deltas) / math.sqrt(len(deltas))
    tolerance = max(0.001, statistics.median(before) * 0.03)
    return {'pairs': len(logs), 'faster_pairs': sum(b < a for a, b in zip(before, after)),
            'before_median_seconds': statistics.median(before),
            'after_median_seconds': statistics.median(after),
            'geometric_mean_ratio': math.exp(log_mean),
            'log_time_ratio_two_sided_95_percent_t_interval': [math.exp(log_mean - log_half), math.exp(log_mean + log_half)],
            'paired_mean_delta_seconds': delta_mean,
            'paired_delta_two_sided_95_percent_t_interval_seconds': [delta_mean - delta_half, delta_mean + delta_half],
            'cold_allowance_seconds': tolerance,
            'cold_noninferiority_upper_delta_within_allowance': delta_mean + delta_half <= tolerance}


def run(variant, count, iteration):
    label = f'{variant}-{count}-{iteration}'
    metrics_path = OUTPUT / (label + '.metrics.json')
    output_path = OUTPUT / (label + '.txt')
    stderr_path = OUTPUT / (label + '.stderr')
    binary = VARIANTS[variant] / 'probe'
    with (FINAL / f'corpus-{count}.txt').open('rb') as source, output_path.open('wb') as out, stderr_path.open('wb') as err:
        cpu_before = Path('/sys/fs/cgroup/cpu.stat').read_text()
        result = subprocess.run([str(FINAL / 'native_cli_measure'), str(metrics_path), str(binary)],
                                stdin=source, stdout=out, stderr=err, cwd=ROOT)
    raw = output_path.read_bytes()
    expected = (FINAL / f'expected-{count}.txt').read_bytes()
    row = json.loads(metrics_path.read_text())
    row.update({'cpu_stat_before': cpu_before, 'cpu_stat_after': Path('/sys/fs/cgroup/cpu.stat').read_text(),
                'variant': variant, 'count': count, 'iteration': iteration,
                'output_lines': len(raw.splitlines()), 'output_sha256': hashlib.sha256(raw).hexdigest(),
                'expected_sha256': hashlib.sha256(expected).hexdigest(),
                'all_coordinate_bits_and_supp_exact': raw == expected,
                'binary_sha256': report['probes'][variant]['sha256'],
                'stderr_bytes': stderr_path.stat().st_size})
    report['runs'].append(row)
    save()
    if result.returncode or raw != expected or row['stderr_bytes']:
        raise AssertionError(label + ' failed exact comparison')
    print(json.dumps(row), flush=True)


if '--run' not in sys.argv:
    raise SystemExit('Prepared only: pass --run after the root CPU GO.')
OUTPUT.mkdir(exist_ok=False)
report = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'method': {'fixed_cold': 'Twenty-five alternating fresh-process pairs; identical first num64 and supplemental words from the original 100000 corpus',
                     'large_equivalence': 'One final 100000-output process; binary64 latitude/longitude and exact supplemental count compared byte-for-byte to existing independent expected output',
                     'timing': 'Native C CLOCK_MONOTONIC fork+exec through wait4; no concurrent benchmark; Linux wait4 child peak RSS',
                     'disk_cache': 'Not evicted; identities read before timing; no discarded first run',
                     'scope': 'Coordinate module probe, not the full shipping CLI; actual shipping CLI is checked separately',
                     'acceptance': 'Paired absolute-time 95% t upper difference <= max(1 ms, 3% of baseline median); log-ratio interval also reported',
                     'sampling': 'One declared twenty-five-pair set; no retries or post-result source changes'},
          'probes': {}, 'corpora': {}, 'runs': [], 'passed': False}
build = json.loads((FINAL / 'build-validation.json').read_text())
baseline_build = json.loads((OLD / 'build.json').read_text())
baseline_identity = next(row for row in baseline_build if row['variant'] == 'baseline')['probe_sha256']
candidate_identity = json.loads((WORK / 'primary-qualification.json').read_text())['binaries']['after']['sha256']
for variant, target in VARIANTS.items():
    source = ROOT / 'evidence/perf4/before' if variant == 'before' else WORK / 'product'
    matching = {}
    for path in (source / 'src').rglob('*'):
        if path.is_file() and path.name != 'main.rs':
            name = path.relative_to(source)
            expected = digest(path)
            assert digest(target / name) == expected, str(name)
            matching[str(name)] = expected
    binary_sha = digest(target / 'probe')
    assert binary_sha == (baseline_identity if variant == 'before' else candidate_identity)
    report['probes'][variant] = {'path': str((target / 'probe').relative_to(ROOT)), 'sha256': binary_sha,
                                'bytes': (target / 'probe').stat().st_size,
                                'matching_product_non_main_modules_sha256': matching,
                                'driver_sha256': digest(target / 'src/main.rs'),
                                'driver_substitutions': ['Entry point prints coordinate bits and supplemental count', 'Unused time error and coordinate struct are local harness stubs']}
assert report['probes']['before']['driver_sha256'] == report['probes']['after']['driver_sha256']
for count in [1, 100000]:
    for prefix in ['corpus', 'expected']:
        if count == 1:
            with (PREVIOUS / f'{prefix}-100000.txt').open('rb') as stream:
                raw = stream.readline()
        else:
            raw = (PREVIOUS / f'{prefix}-{count}.txt').read_bytes()
        destination = FINAL / f'{prefix}-{count}.txt'
        destination.write_bytes(raw)
        report['corpora'][f'{prefix}-{count}'] = {'bytes': len(raw), 'lines': len(raw.splitlines()),
                                                 'sha256': hashlib.sha256(raw).hexdigest()}
assert report['corpora']['expected-100000']['sha256'] == '31c1a4b2ad83e9204bb4eb502ea3e06b578a1692d58b50449ca16c2ee01214be'
save()
for iteration in range(1, 26):
    for variant in ('before', 'after') if iteration % 2 else ('after', 'before'):
        run(variant, 1, iteration)
report['cold_comparison'] = paired(*[[row['wall_seconds'] for row in report['runs'] if row['variant'] == variant and row['count'] == 1] for variant in ['before', 'after']])
save()
run('after', 100000, 1)
report['passed'] = all(row['exit_code'] == 0 and row['all_coordinate_bits_and_supp_exact'] and row['stderr_bytes'] == 0 for row in report['runs'])
report['cold_noninferiority_passed'] = report['cold_comparison']['cold_noninferiority_upper_delta_within_allowance']
report['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
save()
print(json.dumps({'passed': report['passed'], 'cold_comparison': report['cold_comparison']}, indent=2), flush=True)
