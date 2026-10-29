import hashlib
import json
import math
import os
import pathlib
import platform
import pty
import re
import select
import shutil
import signal
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile

ROOT = pathlib.Path('/workspace/scratch/20ca01e7d960')
WORK = ROOT / 'evidence/perf4-sha/final-native'
BASE = WORK / 'cli-runs'
MEASURE = WORK / 'native_cli_measure'
RESULT = WORK / 'native-cli-validation.json'
VARIANTS = ('before', 'after')
if '--run' not in sys.argv or '--after-sha' not in sys.argv:
    raise SystemExit('Prepared only: pass --run --after-sha <frozen final ELF SHA256> after the source freeze.')
EXPECTED_AFTER_SHA = sys.argv[sys.argv.index('--after-sha') + 1]
BINARY_PATHS = {'before': WORK / 'binaries/before',
                'after': WORK / 'binaries/after'}
EXPECTED_BINARY_SHA = {'before': '3d12310ba41743e7a667021beeec861004ff73fe3a64aa2e30935573f527b50e',
                       'after': EXPECTED_AFTER_SHA}
PACKAGER = WORK / 'package_artifact'
LABELS = ('64비트 난수', '2진수', '8진수', '16진수', 'Hex 코드', '바이트 배열',
          '6자리 숫자 비밀번호', '8자리 비밀번호', '로또 번호', '일본 로또 7 번호',
          '유로밀리언 번호', '한글 음절 4글자', '대한민국 위경도', '세계 위경도',
          'NMS 은하 번호', 'NMS 포탈 주소', 'NMS 은하 좌표')
report = {'schema': 1, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'method': {'inputs': ['generate 10000'], 'repetitions': 25,
                     'order': 'alternating previous-v4/final, reversed each repetition',
                     'baseline': 'current adopted perf4 baseline, ELF 3d12310ba41743e7a667021beeec861004ff73fe3a64aa2e30935573f527b50e',
                     'timing': 'fresh process; native C CLOCK_MONOTONIC around fork+execv through wait4',
                     'memory': 'Linux wait4 ru_maxrss per native child, KiB; small native runner prevents Python parent RSS inheritance',
                     'disk_cache': 'not evicted; first run retained',
                     'rng': 'hardware RNG, no deterministic seed CLI; values are not paired',
                     'isolation': 'identical immutable binary copies, fresh empty cwd per run',
                     'limits': ['isolated team benchmark slot, but host noise remains; actual CLI1 has different random inputs and is not an improvement claim',
                                'hardware supplemental words and exact read count are not exposed by CLI']},
          'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                          'cpu_count': os.cpu_count(), 'affinity': sorted(os.sched_getaffinity(0)),
                          'cpu_max': pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                          'memory_max': pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip()},
          'binaries': {}, 'runs': [], 'contracts': [], 'packages': {}, 'errors': []}
record_sets = []


def save():
    RESULT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    with pathlib.Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resource_record(usage):
    return {'max_rss_kib': usage.ru_maxrss, 'user_seconds': usage.ru_utime,
            'system_seconds': usage.ru_stime, 'major_page_faults': usage.ru_majflt,
            'minor_page_faults': usage.ru_minflt,
            'voluntary_context_switches': usage.ru_nvcsw,
            'involuntary_context_switches': usage.ru_nivcsw}


def wait_process(process, start, timeout=300):
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return {'exit_code': process.returncode, 'wall_seconds': time.perf_counter() - start,
                    **resource_record(usage)}
        if time.perf_counter() - start > timeout:
            process.kill()
            pid, status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(status)
            return {'exit_code': process.returncode, 'wall_seconds': time.perf_counter() - start,
                    'timeout': True, **resource_record(usage)}
        time.sleep(0.002)


def execute(binary, cwd, arguments, input_bytes=b'', env=None):
    cwd.mkdir(parents=True, exist_ok=False)
    stdout_path, stderr_path = cwd / 'stdout.log', cwd / 'stderr.log'
    with stdout_path.open('wb') as out, stderr_path.open('wb') as err:
        cpu_before = pathlib.Path('/sys/fs/cgroup/cpu.stat').read_text()
        start = time.perf_counter()
        process = subprocess.Popen([str(MEASURE), str(cwd / 'resource.json'), str(binary), *arguments], cwd=cwd,
                                   stdin=subprocess.PIPE, stdout=out, stderr=err, env=env)
        process.stdin.write(input_bytes)
        process.stdin.close()
        wrapper_result = wait_process(process, start)
        resource_path = cwd / 'resource.json'
        result = json.loads(resource_path.read_text()) if resource_path.exists() else wrapper_result
    result.update({'cpu_stat_before': cpu_before, 'cpu_stat_after': pathlib.Path('/sys/fs/cgroup/cpu.stat').read_text(),
                   'cwd': str(cwd.relative_to(ROOT)), 'arguments': arguments,
                   'stdin_hex': input_bytes.hex(), 'stdout': str(stdout_path.relative_to(ROOT)),
                   'stderr': str(stderr_path.relative_to(ROOT)),
                   'stderr_text': stderr_path.read_text()})
    return result


def parse_output(cwd, expected=None):
    path = cwd / 'random_data.txt'
    raw = path.read_bytes()
    if not raw.startswith(b'\xef\xbb\xbf'):
        raise AssertionError('missing UTF-8 BOM')
    text = raw[3:].decode('utf-8')
    lines = text.splitlines()
    if len(lines) % 17:
        raise AssertionError(f'incomplete record: {len(lines)} lines')
    values, coordinates = [], []
    for start in range(0, len(lines), 17):
        record = lines[start:start + 17]
        if not all(line.startswith(label + ': ') for line, label in zip(record, LABELS)):
            raise AssertionError(f'record labels changed at {start // 17}')
        match = re.fullmatch(r'64비트 난수: (\d+) \(유부호 정수: (-?\d+)\)', record[0])
        if not match:
            raise AssertionError('invalid num64 output')
        value, signed = map(int, match.groups())
        if not 0 <= value < 2 ** 64 or signed != (value if value < 2 ** 63 else value - 2 ** 64):
            raise AssertionError('num64 range/signed value mismatch')
        latitude, longitude = map(float, record[12].split(': ', 1)[1].split(', '))
        if not all(map(math.isfinite, [latitude, longitude])):
            raise AssertionError('nonfinite coordinate')
        if not re.fullmatch(r'NMS 은하 좌표: [0-9A-F]{4}(?::[0-9A-F]{4}){3}', record[-1]):
            raise AssertionError('incomplete final field')
        values.append(value)
        coordinates.append((longitude, latitude))
    if expected is not None and len(values) != expected:
        raise AssertionError(f'{len(values)} records != expected {expected}')
    if text and not text.endswith('\n'):
        raise AssertionError('no final newline')
    result = {'output_path': str(path.relative_to(ROOT)), 'output_bytes': len(raw),
              'output_sha256': hashlib.sha256(raw).hexdigest(), 'records': len(values),
              'complete_records': True, 'utf8_bom': True}
    return result, values, coordinates


def observe(result, variant, cwd, expected=None):
    try:
        parsed, values, coordinates = parse_output(cwd, expected)
        result.update(parsed)
        record_sets.append((result, variant, values, coordinates))
    except Exception as error:
        result['parse_error'] = str(error)
        report['errors'].append({'cwd': str(cwd), 'error': str(error)})
    return result


def terminal_cancel(variant, binary):
    cwd = BASE / (variant + '-menu-cancel')
    cwd.mkdir()
    master, slave = pty.openpty()
    start = time.perf_counter()
    process = subprocess.Popen([str(MEASURE), str(cwd / 'resource.json'), str(binary)], cwd=cwd, stdin=slave, stdout=slave,
                               stderr=slave, start_new_session=True)
    os.close(slave)
    transcript = bytearray()
    consumed = 0

    def until(marker, timeout=60):
        nonlocal consumed
        target = marker.encode()
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            found = transcript.find(target, consumed)
            if found >= 0:
                consumed = found + len(target)
                return
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    chunk = b''
                if not chunk:
                    raise RuntimeError('PTY closed before ' + marker)
                transcript.extend(chunk)
        raise TimeoutError('waiting for ' + marker)

    try:
        until('선택해 주세요: ')
        os.write(master, b'6\n')
        until('선택해 주세요: ')
        os.write(master, b'4\n')
        until('생성할 데이터 개수를 입력해 주세요: ')
        os.write(master, b'1000000\n')
        until('생성 중 Enter를 누르면 작업을 중단합니다.')
        time.sleep(0.15)
        os.write(master, b'\n')
        until('Enter 입력으로 작업을 중단했습니다.')
        until('선택해 주세요: ')
        os.write(master, b'0\n')
        wait_process(process, start, 90)
        result = json.loads((cwd / 'resource.json').read_text())
        result.update({'variant': variant, 'case': 'menu_enter_cancel', 'requested_records': 1000000,
                       'enter_delay_after_prompt_seconds': 0.15,
                       'stdin_sequence': ['6', '4', '1000000', 'Enter', '0']})
    except Exception as error:
        process.kill()
        result = wait_process(process, start, 90)
        result.update({'variant': variant, 'case': 'menu_enter_cancel', 'error': str(error)})
        report['errors'].append({'case': 'menu_enter_cancel', 'variant': variant, 'error': str(error)})
    finally:
        os.close(master)
    (cwd / 'transcript.log').write_bytes(transcript)
    result['transcript'] = str((cwd / 'transcript.log').relative_to(ROOT))
    observe(result, variant, cwd)
    result['cancelled_before_full_count'] = result.get('records', 1000000) < 1000000
    match = re.search(r'총 (\d+)건이 random_data.txt에 기록되었습니다', transcript.decode(errors='replace'))
    result['reported_completed_records'] = int(match.group(1)) if match else None
    result['cancelled_count_matches_file'] = result['reported_completed_records'] == result.get('records')
    return result


BASE.mkdir(parents=True, exist_ok=False)
cpuinfo = pathlib.Path('/proc/cpuinfo').read_text()
report['environment']['hardware_rng_flags'] = [flag for flag in ('rdrand', 'rdseed') if flag in cpuinfo.split()]
for variant in VARIANTS:
    original = BINARY_PATHS[variant]
    assert digest(original) == EXPECTED_BINARY_SHA[variant], 'Frozen ELF identity mismatch'
    snapshot = BASE / 'binaries' / variant / 'release/srg'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(original.read_bytes())
    snapshot.chmod(0o755)
    report['binaries'][variant] = {'original': str(original), 'snapshot': str(snapshot),
                                   'bytes': snapshot.stat().st_size, 'sha256': digest(snapshot),
                                   'original_mtime_ns': original.stat().st_mtime_ns,
                                   'snapshot_matches_original': digest(original) == digest(snapshot)}
save()
for count in (10000,):
    for iteration in range(1, 26):
        for variant in VARIANTS if iteration % 2 else reversed(VARIANTS):
            binary = pathlib.Path(report['binaries'][variant]['snapshot'])
            cwd = BASE / f'{variant}-{count}-{iteration}'
            result = execute(binary, cwd, ['generate', str(count)])
            result.update({'variant': variant, 'count': count, 'iteration': iteration})
            observe(result, variant, cwd, count)
            report['runs'].append(result)
            save()
            print(json.dumps({k: result.get(k) for k in ('variant', 'count', 'iteration', 'exit_code', 'wall_seconds', 'max_rss_kib', 'records')}), flush=True)
fixed_command = [sys.executable, str(WORK.parent / 'final_fixed.py'), '--run']
with (WORK / 'fixed-probe-validation.log').open('wb') as output:
    fixed_result = subprocess.run(fixed_command, stdout=output, stderr=subprocess.STDOUT, cwd=ROOT)
report['fixed_probe_validation'] = {'command': fixed_command, 'exit_code': fixed_result.returncode,
                                    'report': str((WORK / 'fixed-probe-validation.json').relative_to(ROOT))}
if fixed_result.returncode:
    report['errors'].append({'case': 'fixed_probe_validation', 'error': 'fixed source or output equivalence failed'})
    save()
    raise SystemExit(fixed_result.returncode)
fixed_report = json.loads((WORK / 'fixed-probe-validation.json').read_text())
report['fixed_probe_validation'].update({'passed': fixed_report['passed'],
                                        'cold_noninferiority_passed': fixed_report['cold_noninferiority_passed'],
                                        'cold_comparison': fixed_report['cold_comparison']})
save()
print(json.dumps({'fixed_probe_validation': report['fixed_probe_validation']}), flush=True)
for variant in VARIANTS:
    binary = pathlib.Path(report['binaries'][variant]['snapshot'])
    packager = PACKAGER
    cwd = BASE / (variant + '-package')
    env = dict(os.environ, CARGO_TARGET_DIR=str(binary.parent.parent))
    result = execute(packager, cwd, ['srg-linux-x64'], env=env)
    artifact = cwd / 'artifacts/srg-linux-x64.tar'
    with tarfile.open(artifact) as archive:
        members = archive.getmembers()
        member = members[0]
        with archive.extractfile(member) as stream:
            embedded_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        tar_contract = {'members': len(members), 'name': member.name, 'size': member.size,
                        'mode_octal': oct(member.mode), 'uid': member.uid, 'gid': member.gid,
                        'mtime': member.mtime, 'embedded_sha256': embedded_sha,
                        'matches_binary': embedded_sha == report['binaries'][variant]['sha256']}
    zip_path = cwd / 'artifacts/srg-linux-x64.zip'
    info = zipfile.ZipInfo('srg-linux-x64', (1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = (0o100755 << 16)
    info.compress_type = zipfile.ZIP_DEFLATED
    info._compresslevel = 9
    with zipfile.ZipFile(zip_path, 'w') as archive:
        with binary.open('rb') as source, archive.open(info, 'w') as destination:
            shutil.copyfileobj(source, destination)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open('srg-linux-x64') as stream:
            zip_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    result.update({'official_linux_format': 'uncompressed deterministic USTAR TAR',
                   'packager': str(packager), 'tar_path': str(artifact.relative_to(ROOT)),
                   'tar_bytes': artifact.stat().st_size, 'tar_sha256': digest(artifact),
                   'tar_contract': tar_contract,
                   'supplementary_zip_method': 'Python zipfile DEFLATE level 9; fixed metadata; same ELF entry',
                   'zip_path': str(zip_path.relative_to(ROOT)), 'zip_bytes': zip_path.stat().st_size,
                   'zip_sha256': digest(zip_path), 'zip_embedded_sha256': zip_sha,
                   'zip_matches_binary': zip_sha == report['binaries'][variant]['sha256']})
    report['packages'][variant] = result
    save()
    print('packaging complete: ' + variant, flush=True)

for variant in VARIANTS:
    binary = pathlib.Path(report['binaries'][variant]['snapshot'])
    for case, input_bytes, expected in [('menu_eof', b'', 1), ('manual_supp_eof', b'6\n7\n0\n', 0)]:
        cwd = BASE / f'{variant}-{case}'
        result = execute(binary, cwd, [], input_bytes)
        result.update({'variant': variant, 'case': case})
        observe(result, variant, cwd, expected)
        result['manual_supp_prompt_observed'] = 'supp 값 #' in (cwd / 'stdout.log').read_text()
        report['contracts'].append(result)
        save()
    for manual_case in json.loads((WORK / 'manual-fixtures/manifest.json').read_text())['cases']:
        case = manual_case['name']
        cwd = BASE / f'{variant}-{case}'
        input_bytes = (WORK / 'manual-fixtures' / (case + '.input.txt')).read_bytes()
        result = execute(binary, cwd, [], input_bytes)
        result.update({'variant': variant, 'case': case, 'expected_exit_code': manual_case['exit_code']})
        observe(result, variant, cwd, manual_case['atomic_record_count'])
        stdout = (cwd / 'stdout.log').read_text()
        result['supp_prompt_count'] = stdout.count('supp 값 #')
        result['korea_prompt_count'] = stdout.count('대한민국 좌표 재추출')
        result['manual_record_bytes_exact'] = (cwd / 'random_data.txt').read_bytes() == (WORK / 'manual-fixtures' / (case + '.expected.txt')).read_bytes()
        result['manual_contract_passed'] = (result['exit_code'] == manual_case['exit_code']
            and result['supp_prompt_count'] == manual_case['supp_prompt_count']
            and result['korea_prompt_count'] == manual_case['korea_prompt_count']
            and result['manual_record_bytes_exact'])
        if not result['manual_contract_passed']:
            report['errors'].append({'variant': variant, 'case': case, 'error': 'manual contract changed'})
        report['contracts'].append(result)
        save()
    standalone = BASE / (variant + '-standalone')
    standalone.mkdir()
    copied = standalone / 'srg'
    copied.write_bytes(binary.read_bytes())
    copied.chmod(0o755)
    cwd = standalone / 'empty-cwd'
    result = execute(copied, cwd, ['generate', '1'])
    result.update({'variant': variant, 'case': 'standalone_empty_cwd', 'adjacent_files_before_run': ['srg'],
                   'binary_sha256_matches': digest(copied) == report['binaries'][variant]['sha256']})
    observe(result, variant, cwd, 1)
    report['contracts'].append(result)
    report['contracts'].append(terminal_cancel(variant, binary))
    save()
    print('contracts complete: ' + variant, flush=True)
import numpy as np
import shapely

oracle_path = ROOT / 'evidence/source-mask.wkb'
oracle_sha = digest(oracle_path)
assert oracle_sha == '4e7802454d55937eac2ac0a25fb9ed4434196963b5d848441ce8c08c3f857baf'
mask = shapely.from_wkb(oracle_path.read_bytes())
shapely.prepare(mask)
west, south, east, north = mask.bounds
eccentricity_squared = 0.006694380022900787
eccentricity = math.sqrt(eccentricity_squared)


def area_integral(latitude):
    sine = np.sin(np.deg2rad(latitude))
    return sine / (2 * (1 - eccentricity_squared * sine * sine)) + np.arctanh(eccentricity * sine) / (2 * eccentricity)


totals = {variant: {'records_checked': 0, 'outside': 0} for variant in VARIANTS}
auto_supp = {'initial_candidates_outside': 0, 'outside_initial_candidates_replaced': 0,
             'initial_candidates_inside': 0, 'inside_initial_candidates_preserved': 0,
             'coordinate_tolerance_degrees': 1e-12,
             'oracle': 'closed-form GRS80 authalic integral with 52-step bisection; GEOS source covers',
             'supplemental_count_observable': False}
for result, variant, values, coordinates in record_sets:
    points = np.asarray(coordinates, dtype=np.float64).reshape((-1, 2))
    covered = shapely.covers(mask, shapely.points(points))
    outside_indices = np.flatnonzero(~covered)
    result['geos_outside'] = int(len(outside_indices))
    result['geos_checked'] = len(values)
    result['outside_examples'] = [{'record_index': int(i), 'longitude': float(points[i, 0]), 'latitude': float(points[i, 1]), 'num64': str(values[i])} for i in outside_indices[:10]]
    totals[variant]['records_checked'] += len(values)
    totals[variant]['outside'] += len(outside_indices)
    if variant == VARIANTS[1] and result.get('count') is not None:
        numbers = np.asarray(values, dtype=np.uint64)
        vertical = ((numbers >> 32).astype(np.float64) + 0.5) / 4294967296.0
        horizontal = ((numbers & 0xffffffff).astype(np.float64) + 0.5) / 4294967296.0
        target = area_integral(south) + vertical * (area_integral(north) - area_integral(south))
        low, high = np.full(len(values), south), np.full(len(values), north)
        for _ in range(52):
            middle = low + (high - low) * 0.5
            below = area_integral(middle) < target
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        proposed = np.column_stack((west + horizontal * (east - west), low + (high - low) * 0.5))
        initial_covered = shapely.covers(mask, shapely.points(proposed))
        same = np.all(np.abs(proposed - points) <= 1e-12, axis=1)
        initial_outside = int(np.count_nonzero(~initial_covered))
        replaced = int(np.count_nonzero((~initial_covered) & (~same)))
        initial_inside = int(np.count_nonzero(initial_covered))
        preserved = int(np.count_nonzero(initial_covered & same))
        result['automatic_supplemental'] = {'initial_outside': initial_outside, 'outside_replaced': replaced,
                                           'initial_inside': initial_inside, 'inside_preserved': preserved}
        auto_supp['initial_candidates_outside'] += initial_outside
        auto_supp['outside_initial_candidates_replaced'] += replaced
        auto_supp['initial_candidates_inside'] += initial_inside
        auto_supp['inside_initial_candidates_preserved'] += preserved
report['geography'] = {'oracle_path': str(oracle_path.relative_to(ROOT)), 'oracle_sha256': oracle_sha,
                        'source_shp_sha256': '5060f88241e2d80b87f052b4b83296ef59cd7f4ee066a9330bc2ea580bc55428',
                        'shapely': shapely.__version__, 'geos': shapely.geos_version_string,
                        'predicate': 'GEOS covers on full original float64 polygon set, including holes and boundary',
                        'variants': totals, 'candidate_automatic_supplemental': auto_supp}
report['summary'] = {}
for variant in VARIANTS:
    summary = {}
    for count in (10000,):
        runs = [run for run in report['runs'] if run['variant'] == variant and run['count'] == count]
        summary[str(count)] = {key: {'values': [run[key] for run in runs], 'median': statistics.median(run[key] for run in runs),
                                     'minimum': min(run[key] for run in runs), 'maximum': max(run[key] for run in runs)}
                              for key in ('wall_seconds', 'max_rss_kib', 'user_seconds', 'system_seconds')}
        summary[str(count)]['all_successful'] = all(run['exit_code'] == 0 and run.get('records') == count for run in runs)
    report['summary'][variant] = summary
    source = pathlib.Path(report['binaries'][variant]['original'])
    report['binaries'][variant]['original_sha256_at_end'] = digest(source)
    report['binaries'][variant]['original_unchanged'] = digest(source) == report['binaries'][variant]['sha256']
report['paired_comparison'] = {}
for count in (10000,):
    before = [row['wall_seconds'] for row in report['runs'] if row['variant'] == 'before' and row['count'] == count]
    after = [row['wall_seconds'] for row in report['runs'] if row['variant'] == 'after' and row['count'] == count]
    logs = [math.log(b / a) for a, b in zip(before, after)]
    mean = statistics.mean(logs)
    half = 2.0638985616280205 * statistics.stdev(logs) / math.sqrt(25)
    upper = math.exp(mean + half)
    report['paired_comparison'][str(count)] = {
        'pairs': 25, 'faster_pairs': sum(b < a for a, b in zip(before, after)),
        'geometric_mean_ratio': math.exp(mean),
        'log_time_ratio_two_sided_95_percent_t_interval': [math.exp(mean - half), upper],
        'median_reduction_percent': 100 * (1 - statistics.median(after) / statistics.median(before)),
        'hardware_rng_values_paired': False,
        'use': 'Actual CLI throughput noninferiority' if count == 10000 else 'Live RNG smoke; no improvement claim',
        'bulk_noninferiority_upper_ratio_1_03': upper <= 1.03 if count == 10000 else None}
report['measurement_source_sha256'] = digest(pathlib.Path(__file__))
report['cold_noninferiority_passed'] = fixed_report['cold_noninferiority_passed']
report['other_bulk_noninferiority_passed'] = report['paired_comparison']['10000']['bulk_noninferiority_upper_ratio_1_03']
report['passed'] = (not report['errors'] and fixed_report['passed']
    and all(run['exit_code'] == 0 and run.get('records') == run['count'] and run.get('geos_outside') == 0 for run in report['runs'])
    and all(case['exit_code'] == case.get('expected_exit_code', 0) and case.get('geos_outside') == 0 for case in report['contracts'])
    and all(case.get('cancelled_count_matches_file', True) and case.get('cancelled_before_full_count', True) for case in report['contracts'])
    and all(value['original_unchanged'] for value in report['binaries'].values())
    and all(value['zip_matches_binary'] and value['tar_contract']['matches_binary'] for value in report['packages'].values())
    and auto_supp['initial_candidates_outside'] == auto_supp['outside_initial_candidates_replaced']
    and auto_supp['initial_candidates_inside'] == auto_supp['inside_initial_candidates_preserved'])
report['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
save()
print(json.dumps({'summary': report['summary'], 'geography': report['geography'], 'errors': report['errors']}, ensure_ascii=False, indent=2), flush=True)

if not report['passed']:
    raise SystemExit(1)
