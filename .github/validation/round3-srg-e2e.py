import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time

import round3_verify as validate

root = Path.cwd()
work = Path(os.environ["RUNNER_TEMP"]) / "goal-round3-srg"
work.mkdir()
source = work / "source"
source.mkdir()
archive = subprocess.check_output(["git", "archive", sys.argv[1]])
with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
    contents.extractall(source, filter="data")
injection = '''
fn goal_round3_output(count: &std::ffi::OsStr) -> Result<()> {
    let count: usize = count.to_str().unwrap().parse().unwrap();
    assert!(count <= 4096);
    let mut file = OutputFile::try_from(Path::new(FILE_NAME))?;
    let mut state = 0x9e3779b97f4a7c15_u64;
    let mut numbers = 20260920_u64;
    for index in 0..count {
        let num_64 = match index {
            0 => 0,
            1 => 1,
            2 => u64::MAX,
            _ => {
                numbers = numbers.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                numbers
            }
        };
        let mut next = |_reason: &'static str| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Ok(state)
        };
        let data = random_data::RandomDataSet { num_64, ..Default::default() }.populate(&mut next)?;
        file.persist_and_print(&data)?;
    }
    Ok(())
}
'''
binaries = []
for phase in ("before", "after"):
    if phase == "after":
        for name in subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0"):
            if name and not name.startswith(".github/"):
                destination = source / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, destination)
    main = source / "src/main.rs"
    text = main.read_text()
    marker = "fn main() -> Result<()> {"
    assert text.count(marker) == 1
    main.write_text(text.replace(marker, marker + '\n    if let Some(count) = env::var_os("GOAL_FIXED_OUTPUT_COUNT") { return goal_round3_output(&count); }') + injection)
    validate.run([os.environ.get("GOAL_CARGO", "cargo"), "+1.98.1", "build", "--release", "--frozen"],
                 cwd=source, env={**os.environ, "CARGO_TARGET_DIR": str(work / "target")})
    binary = work / phase
    shutil.copy2(work / "target/release/srg", binary)
    binaries.append(binary)

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
directories = [work / "run-before", work / "run-after"]
for directory in directories:
    directory.mkdir()

def execute(side, count, measured=False):
    directory = directories[side]
    output = directory / "random_data.txt"
    output.unlink(missing_ok=True)
    args = [str(binaries[side])]
    usage = directory / "usage.txt"
    if measured:
        args = ["/usr/bin/time", "-f", "%U %S %M", "-o", str(usage), *args]
    started = time.perf_counter_ns()
    result = subprocess.run(args, cwd=directory, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            env={**os.environ, "GOAL_FIXED_OUTPUT_COUNT": str(count)}, timeout=30)
    elapsed = time.perf_counter_ns() - started
    assert result.returncode == 0, result.stderr.decode()
    assert output.stat().st_mode & 0o077 == 0
    return elapsed, int(usage.read_text().split()[2]) if measured else None

for side in (0, 1):
    execute(side, 3)
before, after = [(directory / "random_data.txt").read_bytes() for directory in directories]
assert before == after and after.startswith(b"\xef\xbb\xbf")
lines = after[3:].decode().splitlines()
assert len(lines) == 51
reference = "\n".join(lines)
(work / "expected-browser.txt").write_text(reference)
print("SRG_REFERENCE", hashlib.sha256(after).hexdigest(), flush=True)

for count, repeats in ((1, 16), (4096, 1)):
    for warm in (0, 2):
        samples, rss = [[], []], [[], []]
        for pair in range(validate.PAIRS):
            outputs = [None, None]
            for side in ([0, 1] if pair % 2 == 0 else [1, 0]):
                for _ in range(warm):
                    execute(side, count)
                elapsed, peak = 0, 0
                for _ in range(repeats):
                    duration, resident = execute(side, count, True)
                    elapsed += duration
                    peak = max(peak, resident)
                samples[side].append(elapsed)
                rss[side].append(peak)
                outputs[side] = (directories[side] / "random_data.txt").read_bytes()
            assert outputs[0] == outputs[1]
        upper, memory_upper = validate.paired_bounds(*samples), validate.paired_bounds(*rss)
        print("SRG_E2E " + json.dumps({"count": count, "repeats_per_sample": repeats, "warmup_runs": warm,
              "pairs": validate.PAIRS, "before_ns": samples[0], "after_ns": samples[1],
              "median_ns": list(map(statistics.median, samples)), "p95_ns": [sorted(x)[29] for x in samples],
              "upper95_ratio": upper, "rss_kib": rss, "rss_upper95": memory_upper,
              "limit": validate.LIMIT, "pass": upper <= validate.LIMIT and memory_upper <= validate.LIMIT,
              "scope": "actual release product copy; fixed entropy; populate, format, secured output file and process included"}), flush=True)
        assert upper <= validate.LIMIT and memory_upper <= validate.LIMIT

for scenario in ("locked", "bad-bom", "symlink"):
    observed = []
    for side in (0, 1):
        directory = work / f"reject-{scenario}-{side}"
        directory.mkdir()
        output = directory / "random_data.txt"
        original = b"bad" if scenario == "bad-bom" else b"\xef\xbb\xbf"
        lock = None
        if scenario == "symlink":
            target = directory / "protected.txt"
            target.write_bytes(original)
            output.symlink_to(target)
        else:
            output.write_bytes(original)
            os.chmod(output, 0o600)
            if scenario == "locked":
                lock = output.open("rb")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([str(binaries[side])], cwd=directory, capture_output=True,
                                env={**os.environ, "GOAL_FIXED_OUTPUT_COUNT": "1"}, timeout=30)
        assert result.returncode != 0 and output.read_bytes() == original
        observed.append((result.returncode, result.stdout, result.stderr))
        if lock:
            lock.close()
    assert observed[0] == observed[1]
    print("SRG_REJECTION", scenario, "equivalent; protected bytes unchanged", flush=True)
