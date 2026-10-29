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
FROZEN = ROOT / 'evidence/perf4/before'
PREVIOUS = ROOT / 'evidence/perf2-native-512k'
PROFILE = ROOT / 'evidence/perf2-rust/profile'
TOOLCHAIN = ROOT / 'toolchains/rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def copy_sources(target):
    for source in (FROZEN / 'src').rglob('*'):
        if source.is_file():
            write(target / source.relative_to(FROZEN), source.read_bytes())
    for name in ['Cargo.toml', 'Cargo.lock']:
        write(target / name, (FROZEN / name).read_bytes())
    if not (target / 'data').exists():
        (target / 'data').symlink_to(FROZEN / 'data', target_is_directory=True)


def prepare():
    original = (FROZEN / 'src/boundary_inflate.rs').read_text()
    anchor = '    fn read(&mut self, count: u32) -> Result<u32> {\n'
    assert original.count(anchor) == 1
    fast = '''        if count <= 16
            && let Some(bytes) = self
                .bytes
                .get(self.position.strict_div(8)..)
                .and_then(|tail| tail.first_chunk::<4>())
        {
            let value = (u32::from_le_bytes(*bytes) >> (self.position & 7))
                & ((1_u32 << count).strict_sub(1));
            self.position = self.position.strict_add(count as usize);
            return Ok(value);
        }
'''
    for variant in ['baseline', 'bits-read', 'profile']:
        target = WORK / variant
        copy_sources(target)
        write(target / 'src/main.rs', (PREVIOUS / 'probe-after/src/main.rs').read_bytes())
    (WORK / 'bits-read/src/boundary_inflate.rs').write_text(original.replace(anchor, anchor + fast))
    subprocess.run([str(TOOLCHAIN / 'rustfmt'), '--edition', '2024', str(WORK / 'bits-read/src/boundary_inflate.rs')], check=True)
    assert (FROZEN / 'src/korea_boundary.rs').read_bytes() == (ROOT / 'evidence/perf2-rust/baseline/src/korea_boundary.rs').read_bytes()
    for name in ['main.rs', 'metrics.rs', 'korea_boundary.rs']:
        write(WORK / 'profile/src' / name, (PROFILE / 'src' / name).read_bytes())
    for name in ['corpus-10000.txt']:
        write(WORK / name, (PREVIOUS / name).read_bytes())
    expected = (PREVIOUS / 'expected-100000.txt').read_bytes().splitlines(keepends=True)[:10000]
    write(WORK / 'expected-10000.txt', b''.join(expected))
    measure = WORK / 'measure'
    write(measure, (PREVIOUS / 'native_cli_measure').read_bytes())
    measure.chmod(0o755)
    write(WORK / 'measure.c', (ROOT / 'evidence/perf2-native/native_cli_measure.c').read_bytes())
    identity = {
        'baseline': 'Current adopted Rust, JS quick9 and original 512 KiB data checkpoint frozen in evidence/perf4/before',
        'candidate': 'Safe four-byte load for Bits.read counts up to 16; original bit loop retained at input tail and other counts',
        'acceptance_plan': '../perf4/plan.json',
        'new_runtime_allocations': 0,
        'new_retained_state_bytes': 0,
        'cache_or_math_or_data_changed': False,
        'compiler': subprocess.check_output([str(TOOLCHAIN / 'rustc'), '-Vv'], text=True),
        'source_sha256': {
            variant: {name: digest(WORK / variant / name) for name in ['src/boundary_inflate.rs', 'src/korea_boundary.rs', 'src/korea_boundary/hash.rs', 'src/geo_uniform.rs', 'src/korea_boundary/data.rs']}
            for variant in ['baseline', 'bits-read', 'profile']
        },
        'corpus': {'file': 'corpus-10000.txt', 'sha256': digest(WORK / 'corpus-10000.txt'), 'records': 10000},
        'expected': {'file': 'expected-10000.txt', 'sha256': digest(WORK / 'expected-10000.txt'), 'basis': 'First 10000 rows of unchanged independent 100000-row oracle'},
        'profile_scope': 'Existing diagnostic counters transplanted only after byte-identity check of uninstrumented korea_boundary.rs; timings are inclusive and instrumented, not qualification samples',
    }
    (WORK / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps({'prepared': True, 'candidate_sha256': identity['source_sha256']['bits-read']['src/boundary_inflate.rs']}))


def compile_probes():
    records = []
    for variant in ['baseline', 'bits-read', 'profile']:
        target = WORK / variant
        command = [str(TOOLCHAIN / 'rustc'), '--edition=2024', '--crate-name', 'srg_perf4_probe', '-C', 'opt-level=3', '-C', 'codegen-units=1', '-C', 'lto=fat', '-C', 'panic=abort', '-C', 'strip=symbols', 'src/main.rs', '-o', 'probe']
        started = time.perf_counter()
        with (target / 'build.log').open('wb') as output:
            result = subprocess.run(command, cwd=target, stdout=output, stderr=subprocess.STDOUT)
        record = {'variant': variant, 'command': command, 'exit_code': result.returncode, 'compile_seconds': time.perf_counter() - started}
        records.append(record)
        print(json.dumps(record), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        record.update({'probe_sha256': digest(target / 'probe'), 'probe_bytes': (target / 'probe').stat().st_size})
    (WORK / 'build.json').write_text(json.dumps(records, indent=2) + '\n')


def prepare_other():
    original = (FROZEN / 'src/boundary_inflate.rs').read_text()
    adler = original.replace('first = first.strict_add(u32::from(*byte));', 'first = first.wrapping_add(u32::from(*byte));').replace('second = second.strict_add(first);', 'second = second.wrapping_add(first);')
    assert adler != original
    anchor = '''                    for _ in 0..length {
                        let source = written
'''
    assert original.count(anchor) == 1
    fast = '''                    if let Some(source_start) = written.checked_sub(backward)
                        && let Some(end) = written.checked_add(length)
                        && end <= output.len()
                    {
                        if backward == 1 {
                            let source = *output
                                .get(source_start)
                                .ok_or("경계 압축 참조 위치 오류")?;
                            output
                                .get_mut(written..end)
                                .ok_or("경계 압축 출력 한도 오류")?
                                .fill(source);
                            written = end;
                            continue;
                        }
                        if backward >= length {
                            output.copy_within(
                                source_start..source_start.strict_add(length),
                                written,
                            );
                            written = end;
                            continue;
                        }
                    }
'''
    lz = original.replace(anchor, fast + anchor)
    for name, source in [('adler', adler), ('lz-copy', lz)]:
        target = WORK / name
        copy_sources(target)
        write(target / 'src/main.rs', (PREVIOUS / 'probe-after/src/main.rs').read_bytes())
        (target / 'src/boundary_inflate.rs').write_text(source)
    report = {
        'baseline_inflater_sha256': digest(FROZEN / 'src/boundary_inflate.rs'),
        'adler': {
            'status': 'prepared_not_compiled_or_measured',
            'source_sha256': digest(WORK / 'adler/src/boundary_inflate.rs'),
            'changed_statements': 2,
            'chunk_length_bound': 1024,
            'chunk_start_first_and_second_max': 65520,
            'first_max': 65520 + 1024 * 255,
            'second_max': 65520 + 1024 * 65520 + 255 * 1024 * 1025 // 2,
            'u32_max': 2**32 - 1,
            'proof': 'Every byte is in 0..255. Both accumulators are reduced modulo 65521 at each 1024-byte chunk boundary. Therefore neither addition can overflow u32 even for arbitrary corrupted decompressed bytes; wrapping_add and strict_add have identical values. Chunking, reductions and checksum comparison remain unchanged.',
            'retained_state_delta_bytes': 0,
        },
        'lz-copy': {
            'status': 'prepared_not_compiled_or_measured',
            'source_sha256': digest(WORK / 'lz-copy/src/boundary_inflate.rs'),
            'fast_cases': ['distance=1 repeated byte fill', 'distance>=length nonoverlapping source copy'],
            'proof': 'Distance codes are validated before this branch and give a positive distance. checked_sub validates an existing prefix source; checked_add and end<=output.len validate the entire destination. Distance=1 repeats the already-written byte. Distance>=length means source_end<=written, so copy_within uses only previously produced bytes. Every other case, including any incomplete output capacity or invalid source, executes the original byte loop unchanged, preserving error priority and partial writes.',
            'partial_output_verification': 'Compare baseline and candidate full output buffers and error strings on all existing bounded malformed fixtures, using a nonzero initial output sentinel. Full valid page oracle remains separate.',
            'retained_state_delta_bytes': 0,
        },
        'data_math_cache_dependencies_profiles_changed': False,
        'candidate_relationship': 'Both candidates are independently based on the current adopted inflater, not the unproven Bits.read candidate',
    }
    (WORK / 'other-candidates.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def smoke():
    records = []
    for variant in ['baseline', 'bits-read', 'profile']:
        target = WORK / variant
        metrics = target / 'smoke.metrics.json'
        command = [str(WORK / 'measure'), str(metrics), str(target / 'probe')]
        with (WORK / 'corpus-10000.txt').open('rb') as source, (target / 'smoke.output').open('wb') as output, (target / 'smoke.stderr').open('wb') as errors:
            result = subprocess.run(command, stdin=source, stdout=output, stderr=errors, cwd=target)
        row = json.loads(metrics.read_text())
        row.update({'variant': variant, 'command': command, 'output_sha256': digest(target / 'smoke.output'), 'expected_equal': (target / 'smoke.output').read_bytes() == (WORK / 'expected-10000.txt').read_bytes()})
        if variant == 'profile':
            row['profile'] = json.loads((target / 'smoke.stderr').read_text().strip())
        records.append(row)
        print(json.dumps(row), flush=True)
        if result.returncode or not row['expected_equal']:
            raise SystemExit('smoke failed')
    (WORK / 'smoke.json').write_text(json.dumps({'qualification': False, 'runs_per_variant': 1, 'records': records}, indent=2) + '\n')


def prepare_combined():
    target = WORK / 'bits-adler'
    copy_sources(target)
    write(target / 'src/main.rs', (PREVIOUS / 'probe-after/src/main.rs').read_bytes())
    bits_source = (WORK / 'bits-read/src/boundary_inflate.rs').read_text()
    source = bits_source.replace('first = first.strict_add(u32::from(*byte));', 'first = first.wrapping_add(u32::from(*byte));').replace('second = second.strict_add(first);', 'second = second.wrapping_add(first);')
    assert source != bits_source
    (target / 'src/boundary_inflate.rs').write_text(source)
    before = (FROZEN / 'src/boundary_inflate.rs').read_text()
    report = {
        'status': 'prepared_not_compiled_or_measured',
        'candidate': 'bits-adler',
        'source_sha256': digest(target / 'src/boundary_inflate.rs'),
        'baseline_source_sha256': digest(FROZEN / 'src/boundary_inflate.rs'),
        'frozen_bits_candidate_sha256': digest(WORK / 'bits-read/src/boundary_inflate.rs'),
        'additional_change_from_bits_candidate': 'Exactly two checksum strict_add operations changed to wrapping_add; no count=0 specialization or other reader variant',
        'cost_model': 'The independent components reduce per-bit extra-field reads and per-byte impossible overflow branches within the same repeatedly executed inflater. No cache, memory layout, retained state, data, geometry, coordinate arithmetic or dependency change.',
        'bytes_before': len(before.encode()), 'bytes_after': len(source.encode()),
        'nonblank_lines_before': sum(bool(line.strip()) for line in before.splitlines()),
        'nonblank_lines_after': sum(bool(line.strip()) for line in source.splitlines()),
        'new_allocations': 0, 'new_retained_state_bytes': 0,
        'independent_screens': ['smoke.json', 'other-screen.json'],
        'independent_correctness': ['bits-read-inflate-validation.json', 'other-state-validation.json'],
        'bits_standalone_qualification': 'Five of five faster, but 95% paired log-time upper ratio 1.0008991615011613; retained as unproven, not relabeled passed',
        'qualification_plan': 'Exactly one set of five alternating pairs against the current adopted baseline; at least four faster and paired log-time 95% t upper ratio below 1. No new bit-reader variants or extra repetitions if this combined candidate fails.',
        'other_paths': 'If qualified, actual shipping CLI, fixed-input fresh-process cold and affected validation remain separate checks under ../perf4/plan.json',
    }
    (WORK / 'combined-plan.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def qualify():
    candidate_variant = sys.argv[2] if len(sys.argv) > 2 else 'bits-read'
    if candidate_variant == 'bits-adler':
        plan = json.loads((WORK / 'combined-plan.json').read_text())
        build = json.loads((WORK / candidate_variant / 'probe-build.json').read_text())
        clippy = json.loads((WORK / 'combined-clippy.json').read_text())
        assert digest(WORK / candidate_variant / 'src/boundary_inflate.rs') == plan['qualification_source_sha256'] == build['source_sha256']
        assert build['source_sha256'] == clippy['source_sha256'] and clippy['exit_code'] == 0
        assert digest(WORK / candidate_variant / 'probe') == build['probe_sha256']
    records = []
    for repetition in range(1, 6):
        order = ['baseline', candidate_variant] if repetition % 2 else [candidate_variant, 'baseline']
        for variant in order:
            target = WORK / variant
            prefix = target / f'pair-{repetition}'
            command = [str(WORK / 'measure'), str(prefix.with_suffix('.metrics.json')), str(target / 'probe')]
            with (WORK / 'corpus-10000.txt').open('rb') as source, prefix.with_suffix('.output').open('wb') as output, prefix.with_suffix('.stderr').open('wb') as errors:
                result = subprocess.run(command, stdin=source, stdout=output, stderr=errors, cwd=target)
            row = json.loads(prefix.with_suffix('.metrics.json').read_text())
            row.update({'variant': variant, 'repetition': repetition, 'output_sha256': digest(prefix.with_suffix('.output')), 'expected_equal': prefix.with_suffix('.output').read_bytes() == (WORK / 'expected-10000.txt').read_bytes()})
            records.append(row)
            print(json.dumps(row), flush=True)
            if result.returncode or not row['expected_equal']:
                raise SystemExit('qualification behavior failed')
    before = {r['repetition']: r for r in records if r['variant'] == 'baseline'}
    after = {r['repetition']: r for r in records if r['variant'] == candidate_variant}
    paired = [math.log(after[i]['wall_seconds'] / before[i]['wall_seconds']) for i in before]
    mean = statistics.mean(paired)
    half = 2.7764451051977987 * statistics.stdev(paired) / math.sqrt(5)
    faster = sum(after[i]['wall_seconds'] < before[i]['wall_seconds'] for i in before)
    summary = {
        'candidate': candidate_variant,
        'source_sha256': digest(WORK / candidate_variant / 'src/boundary_inflate.rs'),
        'primary_workload': 'fixed 10000 coordinates, isolated subprocess, same direct probe and shipping optimization flags',
        'samples': 5,
        'extra_warmup_runs': 0,
        'all_output_bytes_exact': True,
        'faster_pairs': faster,
        'paired_log_time_ratio': {'geometric_mean': math.exp(mean), 'confidence_95_percent': [math.exp(mean - half), math.exp(mean + half)], 'student_t_df': 4, 'student_t_critical': 2.7764451051977987},
        'primary_performance_criterion_met': faster >= 4 and math.exp(mean + half) < 1,
        'medians': {variant: {key: statistics.median(r[key] for r in records if r['variant'] == variant) for key in ['wall_seconds', 'max_rss_kib', 'user_seconds', 'system_seconds']} for variant in ['baseline', candidate_variant]},
        'records': records,
        'remaining': 'Full inflater/error validation; shipping CLI, deterministic cold, platform lint/tests and root integration remain separate gates',
    }
    (WORK / f'{candidate_variant}-qualification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


def validate_inflate():
    target = WORK / 'bits-read'
    source = ROOT / 'evidence/perf2-rust/inflate_probe.rs'
    write(target / 'src/inflate_probe.rs', source.read_bytes())
    command = [str(TOOLCHAIN / 'rustc'), '--edition=2024', '--crate-name', 'srg_perf4_inflate_probe', '-C', 'opt-level=3', '-C', 'codegen-units=1', '-C', 'lto=fat', '-C', 'panic=abort', '-C', 'strip=symbols', 'src/inflate_probe.rs', '-o', 'inflate-probe']
    with (target / 'inflate-build.log').open('wb') as output:
        result = subprocess.run(command, cwd=target, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.returncode)
    oracle = ROOT / 'evidence/boundary-inflate-oracle.bin'
    invalid = ROOT / 'evidence/perf2-rust/inflate-errors.bin'
    command = [str(target / 'inflate-probe'), str(oracle), str(invalid)]
    with (target / 'inflate-errors.output').open('wb') as output, (target / 'inflate-validation.stderr').open('wb') as errors:
        result = subprocess.run(command, cwd=target, stdout=output, stderr=errors)
    expected = ROOT / 'evidence/perf2-native/final-inflate-errors.txt'
    equal = (target / 'inflate-errors.output').read_bytes() == expected.read_bytes()
    report = {
        'candidate_source_sha256': digest(target / 'src/boundary_inflate.rs'),
        'oracle_file': str(oracle.relative_to(ROOT)), 'oracle_sha256': digest(oracle),
        'error_fixture_file': str(invalid.relative_to(ROOT)), 'error_fixture_sha256': digest(invalid),
        'command': command, 'exit_code': result.returncode,
        'error_strings_and_order_exact': equal,
        'error_output_sha256': digest(target / 'inflate-errors.output'),
        'counts': json.loads((target / 'inflate-validation.stderr').read_text().strip()) if result.returncode == 0 else None,
        'valid_fixtures': 10, 'invalid_fixtures': 1499,
        'new_retained_state_bytes': 0,
    }
    (WORK / 'bits-read-inflate-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if result.returncode or not equal:
        raise SystemExit('inflate qualification failed')


if __name__ == '__main__':
    {'prepare': prepare, 'prepare-other': prepare_other, 'prepare-combined': prepare_combined, 'compile': compile_probes, 'smoke': smoke, 'qualify': qualify, 'inflate': validate_inflate}[sys.argv[1]]()
