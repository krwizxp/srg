import hashlib
import json
from pathlib import Path
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
OLD = ROOT / 'evidence/perf4-rust'
FROZEN = ROOT / 'evidence/perf4/before'
FINAL = WORK / 'final-native'
FINAL.mkdir(exist_ok=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


candidate = digest(WORK / 'candidate.rs')
assert candidate == '90b580b4ff5c70529b7127a47742246141d39112add1efd361a9b10be092d8ef'
formal_probe = WORK / 'formal-probe'
for source in (WORK / 'probe/src').rglob('*'):
    if source.is_file():
        write(formal_probe / source.relative_to(WORK / 'probe'), source.read_bytes())
(formal_probe / 'data').symlink_to(FROZEN / 'data', target_is_directory=True)
for name in ['rust-toolchain.toml', '.cargo/config', '.cargo/config.toml']:
    source = ROOT / 'srg' / name
    if source.exists():
        write(WORK / 'product' / name, source.read_bytes())
source_hashes = {}
for directory in ['src', 'examples']:
    for source in (WORK / 'product' / directory).rglob('*'):
        if source.is_file():
            source_hashes[str(source.relative_to(WORK / 'product'))] = digest(source)
for name in ['Cargo.toml', 'Cargo.lock', 'clippy.toml', 'rust-toolchain.toml', '.cargo/config', '.cargo/config.toml']:
    source = WORK / 'product' / name
    if source.exists():
        source_hashes[name] = digest(source)
changed = [name for name, sha in source_hashes.items() if (FROZEN / name).exists() and digest(FROZEN / name) != sha]
assert changed == ['src/korea_boundary/hash.rs']
lint = []
for label in ['host', 'windows', 'intel-mac', 'arm-mac']:
    path = WORK / f'clippy-final-{label}.json'
    item = json.loads(path.read_text())
    assert item['source_sha256'] == candidate and item['exit_code'] == 0 and item['diagnostic_count'] == 323
    lint.append({'target':item['command'][item['command'].index('--target') + 1],
                 'command':item['command'], 'cwd':str((WORK / 'product').relative_to(ROOT)),
                 'exit_code':0,'diagnostics':323,'bin_warnings':109,'test_warnings':214,
                 'record':str(path.relative_to(ROOT)), 'record_sha256':digest(path),
                 'stdout_sha256':digest(WORK / f'clippy-final-{label}.stdout.jsonl'),
                 'stderr_sha256':digest(WORK / f'clippy-final-{label}.stderr')})
manifest = {'source_root':str((WORK / 'product').relative_to(ROOT)), 'source_sha256':source_hashes,
            'changed_vs_frozen_baseline':changed, 'clippy':lint,
            'clippy_sources_unchanged_since_execution':True,
            'toolchain_selection':{'compiler':'1.99.1','direct_toolchain_binaries':True,'RUSTC':'Explicit pinned1.99.1 rustc path',
                                  'repository_toolchain_file':'Original stable/minimal manifest copied for source completeness; explicit cargo/rustc selection used in all builds and Clippy, so this file did not select a different compiler'},
            'input_data':'Original SHA-pinned97 data files; product and probes symlink the immutable perf4/before data directory'}
(FINAL / 'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
plan = {'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'parent_plan_sha256':digest(ROOT / 'evidence/perf4/plan.json'),
        'source_manifest_sha256':digest(FINAL / 'source-manifest.json'),
        'candidate_hash_sha256':candidate,'baseline_hash_sha256':digest(WORK / 'baseline.rs'),
        'baseline_inflater_sha256':digest(FROZEN / 'src/boundary_inflate.rs'),
        'candidate':'SHA-only; provisional bits+Adler excluded',
        'primary':{'fixed10000_pairs':5,'criterion':'At least4/5 faster and paired log-time two-sided95% t interval upper ratio <1'},
        'protected':{'actual_shipping_cli10000_pairs':25,'criterion':'Paired log-time two-sided95% t interval upper ratio <=1.03',
                     'fixed_cold1_pairs':25,'cold_criterion':'Paired absolute-time two-sided95% t interval upper delta <=max(1ms,3% of baseline median)'},
        'sample_selection':'Independent new formal blocks; no screen, former candidate, or previous supplemental samples included; no exclusions or repeats',
        'order':'Each block alternates before/after for odd pair, after/before for even pair; exclusive CPU slot',
        'stages':['Matched-flags probe compile','Primary5 pairs; stop if criterion is not met','Shipping release build and17 tests','Actual CLI25 pairs','Fixed cold25 pairs and100k exact','Official package and existing CLI/manual/error/cancel/GEOS checks'],
        'screen_configuration_note':'Screen coordinate probe used a different crate-name; preserve it as diagnostic only. Formal coordinate probe uses the exact baseline compile argument list in a new directory.',
        'normal_cli_stderr':'Preserve RDSEED informational text; not an error when existing exit/output contracts pass',
        'platform_limits':'Hardware dispatch is measured on SHA-enabled Linux x86_64 only. Forced scalar is functional validation, not measured unsupported hardware. Cross-target Clippy does not replace native Windows/macOS execution.',
        'stop':'One predeclared qualification only. No extra samples after fixed counts; stop and report if the primary criterion fails or a protected path establishes regression.',
        'source_changes_during_measurement':'None allowed; stop and notify parent if necessary'}
assert not (WORK / 'qualification-plan.json').exists()
(WORK / 'qualification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
for source in (OLD / 'final-native/manual-fixtures').rglob('*'):
    if source.is_file():
        write(FINAL / 'manual-fixtures' / source.relative_to(OLD / 'final-native/manual-fixtures'), source.read_bytes())
write(FINAL / 'native_cli_measure', (OLD / 'measure').read_bytes())
(FINAL / 'native_cli_measure').chmod(0o755)
print(json.dumps({'candidate_sha256':candidate,'source_files':len(source_hashes),'changed':changed,'plan_sha256':digest(WORK / 'qualification-plan.json')},indent=2))
