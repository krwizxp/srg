import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
FROZEN = ROOT / 'evidence/perf4/before'
OLD = ROOT / 'evidence/perf4-rust'
TOOLCHAIN = ROOT / 'toolchains/rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)

SOURCE = WORK / 'product'
TARGET = ROOT / 'build/perf4-sha-final'
FINAL = WORK / 'final-native'
FINAL.mkdir(parents=True, exist_ok=True)
environment = os.environ.copy()
environment['PATH'] = str(TOOLCHAIN) + os.pathsep + environment.get('PATH', '')
environment['RUSTC'] = str(TOOLCHAIN / 'rustc')
environment['CARGO_HOME'] = str(OLD / 'cargo-home')
environment['CARGO_TERM_COLOR'] = 'never'
source_manifest = json.loads((FINAL / 'source-manifest.json').read_text())
source_hashes = source_manifest['source_sha256']
assert all(digest(SOURCE / name) == value for name, value in source_hashes.items())
for name in ['Cargo.toml', 'Cargo.lock', 'clippy.toml', 'examples/package_artifact.rs']:
    source_hashes[name] = digest(SOURCE / name)
assert source_hashes['src/korea_boundary/hash.rs'] == '90b580b4ff5c70529b7127a47742246141d39112add1efd361a9b10be092d8ef'
report = {'source_manifest_sha256': digest(FINAL / 'source-manifest.json'), 'clippy': source_manifest['clippy'], 'source_sha256': source_hashes, 'toolchain': subprocess.check_output([str(TOOLCHAIN / 'rustc'), '-Vv'], text=True), 'commands': [], 'shipping_performance_measurements_started': False, 'primary_qualification_already_completed': True}
for label, verb, selection in [('release-build', 'build', ['--bin', 'srg']), ('release-tests', 'test', ['--bin', 'srg']), ('packager-build', 'build', ['--example', 'package_artifact'])]:
    command = [str(TOOLCHAIN / 'cargo'), verb, '--release', '--frozen', *selection, '--target', 'x86_64-unknown-linux-gnu', '--target-dir', str(TARGET)]
    start = time.perf_counter()
    with (FINAL / f'{label}.log').open('wb') as output:
        result = subprocess.run(command, cwd=SOURCE, env=environment, stdout=output, stderr=subprocess.STDOUT)
    row = {'label': label, 'command': command, 'exit_code': result.returncode, 'build_or_test_elapsed_seconds_not_benchmark': time.perf_counter() - start}
    report['commands'].append(row)
    (FINAL / 'build-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(row), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
test_log = (FINAL / 'release-tests.log').read_text()
report['release_tests_passed'] = int(re.search(r'test result: ok\. (\d+) passed;', test_log).group(1))
before = ROOT / 'build/perf2-final-512k/x86_64-unknown-linux-gnu/release/srg'
after = TARGET / 'x86_64-unknown-linux-gnu/release/srg'
assert digest(before) == '3d12310ba41743e7a667021beeec861004ff73fe3a64aa2e30935573f527b50e'
report['binaries'] = {}
for name, source in [('before', before), ('after', after)]:
    path = FINAL / 'binaries' / name
    write(path, source.read_bytes())
    path.chmod(0o755)
    report['binaries'][name] = {'source': str(source.relative_to(ROOT)), 'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': digest(path)}
write(FINAL / 'package_artifact', (TARGET / 'x86_64-unknown-linux-gnu/release/examples/package_artifact').read_bytes())
(FINAL / 'package_artifact').chmod(0o755)
write(FINAL / 'native_cli_measure', (OLD / 'measure').read_bytes())
(FINAL / 'native_cli_measure').chmod(0o755)
for source in (ROOT / 'evidence/perf2-native-512k/manual-fixtures').rglob('*'):
    if source.is_file():
        write(FINAL / 'manual-fixtures' / source.relative_to(ROOT / 'evidence/perf2-native-512k/manual-fixtures'), source.read_bytes())
report['source_frozen_unchanged'] = all(digest(SOURCE / name) == value for name, value in source_hashes.items())
assert report['source_frozen_unchanged']
report['measurement_runner_sha256'] = digest(FINAL / 'native_cli_measure')
report['packager_sha256'] = digest(FINAL / 'package_artifact')
report['extra_rustflags'] = {name: environment.get(name) for name in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS']}
(FINAL / 'build-validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ['source_sha256', 'commands']}), flush=True)
