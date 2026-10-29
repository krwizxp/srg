from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import zipfile

root = Path('/workspace/scratch/20ca01e7d960')
repo = root / 'srg'
work = root / 'evidence/perf4'
if '--run' not in sys.argv:
    raise SystemExit('Prepared only; --run verifies the integrated SHA-only source.')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

plan = json.loads((work / 'plan.json').read_text())
allowed = {'README.md', 'docs/korea-v4-format.md', 'src/korea_boundary/hash.rs'}
compared = []
changes = []
for entry in plan['baseline_files']:
    path = repo / entry['path']
    actual = digest(path)
    same = actual == entry['sha256']
    compared.append({**entry, 'current_sha256': actual, 'current_bytes': path.stat().st_size,
                     'unchanged': same})
    if not same:
        assert entry['path'] in allowed, entry['path']
        changes.append(entry['path'])
assert 'src/korea_boundary/hash.rs' in changes
assert digest(repo / 'src/korea_boundary/hash.rs') == '90b580b4ff5c70529b7127a47742246141d39112add1efd361a9b10be092d8ef'
assert digest(repo / 'src/boundary_inflate.rs') == '3d5914e2265978f2085dc436739d85c6f8b069525fb0f15545fc52f0c8112dd5'

product = root / 'evidence/perf4-sha/product'
inputs = sorted({str(path.relative_to(repo)) for directory in ['src', 'examples']
                 for path in (repo / directory).rglob('*.rs')})
inputs += [name for name in ['Cargo.toml', 'Cargo.lock', 'clippy.toml',
                             'rust-toolchain.toml', 'rustfmt.toml', '.rustfmt.toml']
           if (repo / name).is_file()]
validated_inputs = []
for relative in inputs:
    path = repo / relative
    assert path.read_bytes() == (product / relative).read_bytes(), relative
    validated_inputs.append({'path': relative, 'bytes': path.stat().st_size,
                             'sha256': digest(path)})

archives = [
    ('srg-korea-v4-review.zip', 49920572,
     'ab4e26cef17289c77a3dae89d2ecebc64a5c489335585d1b11060206608b4363'),
    ('srg-korea-v4-optimized-review.zip', 50160735,
     '0df9f0243964f40c54d93bc25bbd16f714aedc5c9e597120cb6a9b5fd20ac688'),
    ('srg-korea-v4-js-adopted-review.zip', 50210667,
     'd5bd27ad1666e62c7395d888dbdb5dddb55d978945631c77313e715dd39e39b4'),
]
historical_archives = []
for name, size, expected in archives:
    path = root / name
    assert path.stat().st_size == size and digest(path) == expected, name
    historical_archives.append({'path': name, 'bytes': size, 'sha256': expected,
                                'unchanged': True})

with zipfile.ZipFile(root / 'srg-korea-v4-js-adopted-review.zip') as archive:
    previous_manifest = json.loads(archive.read('REVIEW.json'))
current_hashes = {item['path']: item['current_sha256'] for item in compared}
previous_snapshot_changes = []
for entry in previous_manifest['files']:
    relative = entry['path'].removeprefix('srg/')
    path = repo / relative
    assert path.is_file(), relative
    actual = current_hashes.get(relative)
    if actual is None:
        actual = digest(path)
    if actual != entry['sha256']:
        assert relative in allowed, relative
        previous_snapshot_changes.append(relative)
assert set(previous_snapshot_changes) == allowed

head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
assert head == plan['git_head']
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=repo)
subprocess.run(['git', 'diff', '--check'], cwd=repo, check=True)
assert not (repo / 'data/korea-boundary-v3').exists()
bins = [item for item in compared if item['path'].endswith('.bin')
        and item['path'].startswith('data/korea-boundary-v4/')]
assert len(bins) == 97 and sum(item['current_bytes'] for item in bins) == 50276630
assert all(item['unchanged'] for item in bins)
report = {
    'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'baseline_commit': head,
    'baseline_compared_files': compared,
    'changes_within_frozen_baseline': changes,
    'sha_only_integrated': True,
    'provisional_inflater_rolled_back_to_adopted_baseline': True,
    'validated_isolated_build_inputs_equal_integrated_product': validated_inputs,
    'binary_data_files': len(bins),
    'binary_data_bytes': sum(item['current_bytes'] for item in bins),
    'all_data_hashes_unchanged': True,
    'historical_archives': historical_archives,
    'previous_snapshot_compared_files': len(previous_manifest['files']),
    'previous_snapshot_changes': previous_snapshot_changes,
    'all_prior_reports_and_evidence_preserved': True,
    'javascript_and_html_unchanged': True,
    'active_ci_and_build_policy_unchanged': True,
    'git_head_unchanged': True,
    'git_index_empty': True,
    'git_diff_check': 'passed',
    'remote_writes': [],
}
(work / 'final-identity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'baseline_files': len(compared), 'changes': changes,
                  'validated_build_inputs': len(validated_inputs), 'data_files': len(bins),
                  'historical_archives_unchanged': len(historical_archives)}, ensure_ascii=False))
