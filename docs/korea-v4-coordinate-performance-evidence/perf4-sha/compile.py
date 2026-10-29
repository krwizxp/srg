import hashlib
import json
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
TOOLCHAIN = ROOT / 'toolchains/rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin'
flags = ['--edition=2024', '-C', 'opt-level=3', '-C', 'codegen-units=1', '-C', 'lto=fat', '-C', 'panic=abort', '-C', 'strip=symbols']
subprocess.run([str(TOOLCHAIN / 'rustfmt'), '--edition', '2024', str(WORK / 'bench.rs')], check=True)
report = {'compiler':subprocess.check_output([str(TOOLCHAIN / 'rustc'), '-Vv'], text=True),
          'commands': [], 'source_sha256':{name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in ['baseline.rs','candidate.rs','forced_scalar.rs','bench.rs']}}
for label, cwd, source, output in [('sha-bench', WORK, 'bench.rs', 'bench'), ('coordinate-probe', WORK / 'probe', 'src/main.rs', 'probe')]:
    command = [str(TOOLCHAIN / 'rustc'), *flags, '--crate-name', 'srg_perf4_sha', source, '-o', output]
    with (WORK / f'{label}-build.log').open('wb') as log:
        start = time.perf_counter()
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    row = {'label':label,'command':command,'cwd':str(cwd.relative_to(ROOT)),'exit_code':result.returncode,'compile_seconds_not_benchmark':time.perf_counter()-start}
    if not result.returncode:
        path = cwd / output
        row.update({'binary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'binary_bytes':path.stat().st_size})
    report['commands'].append(row)
    (WORK / 'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
assert hashlib.sha256((WORK/'candidate.rs').read_bytes()).hexdigest() == report['source_sha256']['candidate.rs']
assert (WORK/'candidate.rs').read_bytes() == (WORK/'probe/src/korea_boundary/hash.rs').read_bytes()
assert (WORK/'probe/src/boundary_inflate.rs').read_bytes() == (ROOT/'evidence/perf4/before/src/boundary_inflate.rs').read_bytes()
