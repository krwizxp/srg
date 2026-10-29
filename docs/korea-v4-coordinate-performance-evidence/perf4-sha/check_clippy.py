import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
FROZEN = ROOT / 'evidence/perf4/before'
TOOLCHAIN = ROOT / 'toolchains/rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin'
label = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 else 'x86_64-unknown-linux-gnu'
product = WORK / 'product'
environment = os.environ.copy()
environment['PATH'] = str(TOOLCHAIN) + os.pathsep + environment.get('PATH', '')
environment['RUSTC'] = str(TOOLCHAIN / 'rustc')
environment['CARGO_HOME'] = str(ROOT / 'evidence/perf4-rust/cargo-home')
command = [str(TOOLCHAIN / 'cargo'), 'clippy', '--all-targets', '--target', target, '--frozen', '--manifest-path', str(product / 'Cargo.toml'), '--target-dir', str(WORK / 'clippy-target'), '--message-format=json']
with (WORK / f'{label}.stdout.jsonl').open('wb') as output, (WORK / f'{label}.stderr').open('wb') as errors:
    result = subprocess.run(command, cwd=product, env=environment, stdout=output, stderr=errors)
messages = []
candidate_messages = []
for line in (WORK / f'{label}.stdout.jsonl').read_text().splitlines():
    value = json.loads(line)
    if value.get('reason') != 'compiler-message':
        continue
    message = value['message']
    messages.append(message)
    if any(span['file_name'].endswith('src/korea_boundary/hash.rs') for span in message.get('spans', [])):
        candidate_messages.append(message)
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = {'command': command, 'exit_code': result.returncode,
          'source_sha256': digest(product / 'src/korea_boundary/hash.rs'),
          'policy_sha256': {name: digest(product / name) for name in ['Cargo.toml', 'Cargo.lock', 'clippy.toml']},
          'baseline_policy_exact': all((product / name).read_bytes() == (FROZEN / name).read_bytes() for name in ['Cargo.toml', 'Cargo.lock', 'clippy.toml']),
          'diagnostic_count': len(messages), 'hash_diagnostics': candidate_messages}
(WORK / f'{label}.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'hash_diagnostics'}), flush=True)
for message in candidate_messages:
    print(json.dumps({'level': message['level'], 'code': message.get('code'), 'message': message['message'], 'primary_spans': [{'line': span['line_start'], 'column': span['column_start'], 'text': span['text']} for span in message['spans'] if span['is_primary']]}), flush=True)
raise SystemExit(result.returncode)
