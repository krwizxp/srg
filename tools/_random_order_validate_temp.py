import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback


parser = argparse.ArgumentParser()
parser.add_argument("--target", required=True)
parser.add_argument("--binary-name", required=True)
parser.add_argument("--baseline", required=True)
parser.add_argument("--toolchain", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--live-hardware", action="store_true")
args = parser.parse_args()
repository = Path.cwd().resolve()
evidence = Path(args.out).resolve()
evidence.mkdir(parents=True, exist_ok=True)
work = evidence.parent / "random-order-native-work"
work.mkdir(parents=True, exist_ok=True)
baseline_root = work / "baseline"
versions = {"baseline": baseline_root, "candidate": repository}
version_checks = ["format", "clippy", "release_tests", "release_build", "package_roundtrip", "cli_help", "cli_version", "manual_conversion", "historical_probe"]
if args.live_hardware:
    version_checks.append("live_hardware_generate_100")
required = ["toolchain_identity", "native_architecture", "baseline_checkout", "dependency_resolution"]
required += [f"{version}.{name}" for version in versions for name in version_checks]
required += ["clippy_no_new_warnings", "help_exact", "version_exact", "manual_record_exact", "historical_samples_exact", "historical_counts_exact", "release_size_limit", "source_preservation"]
checks = {name: {"status": "pending"} for name in required}
metadata = {"baseline": args.baseline, "target": args.target, "toolchain": args.toolchain, "required_checks": required, "platform": platform.platform(), "machine": platform.machine(), "python": sys.version, "runner_image": {name: os.environ.get(name) for name in ["ImageOS", "ImageVersion", "RUNNER_ARCH", "RUNNER_OS"]}, "versions": {}}
command_sequence = 0


def write_json(filename, value):
    (evidence / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def persist():
    write_json("checks.json", checks)
    write_json("metadata.json", metadata)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check(name, action):
    started = time.monotonic()
    try:
        detail = action()
        checks[name] = {"status": "passed", "detail": detail, "elapsed_seconds": time.monotonic() - started}
    except Exception as error:
        checks[name] = {"status": "failed", "error": str(error), "exception": type(error).__name__, "elapsed_seconds": time.monotonic() - started}
        (evidence / (name + ".exception.txt")).write_text(traceback.format_exc(), encoding="utf-8")
    persist()
    print(json.dumps({"check": name, **checks[name]}, ensure_ascii=False), flush=True)


def digest(filename):
    result = hashlib.sha256()
    with Path(filename).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def execute(label, command, cwd, env=None, stdin=None, timeout=600):
    global command_sequence
    command_sequence += 1
    prefix = f"{command_sequence:03d}-{label}"
    effective_env = os.environ.copy()
    effective_env["CARGO_TERM_COLOR"] = "never"
    if env:
        effective_env.update({name: str(value) for name, value in env.items()})
    started = time.monotonic()
    try:
        result = subprocess.run([str(part) for part in command], cwd=cwd, env=effective_env, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as error:
        for suffix, data in [("stdout", error.stdout), ("stderr", error.stderr)]:
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            (evidence / f"{prefix}.{suffix}.log").write_text(data or "", encoding="utf-8")
        write_json(f"{prefix}.command.json", {"command": list(map(str, command)), "cwd": str(cwd), "timeout": timeout, "timed_out": True})
        raise
    (evidence / f"{prefix}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (evidence / f"{prefix}.stderr.log").write_text(result.stderr, encoding="utf-8")
    write_json(f"{prefix}.command.json", {"command": list(map(str, command)), "cwd": str(cwd), "returncode": result.returncode, "elapsed_seconds": time.monotonic() - started, "timeout": timeout})
    require(result.returncode == 0, f"Command failed ({result.returncode}): {label}; see {prefix} logs")
    return result


def snapshot(root):
    paths = sorted(root.glob("src/**/*.rs"))
    paths += [root / name for name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "revgeo.html", "korea_boundary.js"]]
    paths += sorted((root / "data" / "korea-boundary-v4").glob("*.bin"))
    return {str(filename.relative_to(root)).replace("\\", "/"): digest(filename) for filename in paths}


def toolchain_identity():
    result = execute("rustc-version", ["rustc", "+" + args.toolchain, "-Vv"], repository)
    metadata["rustc_vv"] = result.stdout
    require(re.search(r"^release: " + re.escape(args.toolchain) + r"$", result.stdout, re.MULTILINE), "Exact Rust version mismatch")
    require(re.search(r"^host: " + re.escape(args.target) + r"$", result.stdout, re.MULTILINE), "Rust host differs from native target")
    metadata["candidate_commit"] = execute("candidate-revision", ["git", "rev-parse", "HEAD"], repository).stdout.strip()
    metadata["candidate_tree"] = execute("candidate-tree", ["git", "rev-parse", "HEAD^{tree}"], repository).stdout.strip()
    return {"version": args.toolchain, "host": args.target}


def native_architecture():
    actual = platform.machine().lower()
    expected = "arm64" if args.target.startswith("aarch64") else "x64"
    normalized = "arm64" if actual in ["arm64", "aarch64"] else "x64" if actual in ["amd64", "x86_64"] else actual
    require(normalized == expected, f"Native architecture mismatch: {actual} versus {expected}")
    if os.name == "nt":
        host = json.loads((evidence / "windows-host.json").read_text(encoding="utf-8-sig"))
        require("Windows Server 2025" in host["Caption"], "Windows Server 2025 required")
        require(host["OSArchitecture"] == host["ProcessArchitecture"] == "X64", "Native OS/process x64 required")
        require(host["Is64BitOperatingSystem"] and host["Is64BitProcess"], "64-bit Windows OS/process required")
        metadata["windows_host"] = host
    return {"native_machine": actual, "expected": expected}


def baseline_checkout():
    execute("verify-baseline-object", ["git", "cat-file", "-e", args.baseline + "^{commit}"], repository)
    require(not baseline_root.exists(), "Task baseline worktree path already exists")
    execute("baseline-worktree", ["git", "worktree", "add", "--detach", baseline_root, args.baseline], repository)
    actual = execute("baseline-revision", ["git", "rev-parse", "HEAD"], baseline_root).stdout.strip()
    require(actual == args.baseline, "Baseline checkout revision mismatch")
    for version, root in versions.items():
        metadata["versions"][version] = {"source_hashes": snapshot(root), "lock_sha256": digest(root / "Cargo.lock")}
    return {"revision": actual}


def dependency_resolution():
    left = metadata["versions"]["baseline"]["lock_sha256"]
    right = metadata["versions"]["candidate"]["lock_sha256"]
    require(left == right, "Cargo.lock changed")
    require(digest(baseline_root / "Cargo.toml") == digest(repository / "Cargo.toml"), "Cargo.toml changed")
    return {"Cargo.lock_sha256": left}


def parse_warnings(result, root):
    messages = Counter()
    errors = []
    for line in result.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("reason") != "compiler-message":
            continue
        message = value["message"]
        if message["level"] == "error":
            errors.append(message["message"])
        if message["level"] != "warning":
            continue
        spans = message.get("spans", [])
        primary = next((span for span in spans if span.get("is_primary")), spans[0] if spans else {})
        filename = primary.get("file_name", "<none>").replace("\\", "/")
        root_prefix = str(root).replace("\\", "/") + "/"
        if filename.startswith(root_prefix):
            filename = filename[len(root_prefix):]
        code = (message.get("code") or {}).get("code", "<none>")
        messages[(filename, code, message["message"])] += 1
    require(not errors, "Clippy emitted compiler errors")
    return messages


def cargo_env(version, target_suffix=""):
    directory = work / "targets" / version
    if target_suffix:
        directory = directory / target_suffix
    return {"CARGO_TARGET_DIR": directory}


def cargo(version, subcommand, flags, label=None, cwd=None, env=None, timeout=600):
    root = versions[version]
    return execute(label or f"{version}-{subcommand}", ["cargo", "+" + args.toolchain, subcommand, *flags], cwd or root, env or cargo_env(version), timeout=timeout)


def format_check(version):
    cargo(version, "fmt", ["--all", "--", "--check"])
    return "cargo fmt --all -- --check"


def clippy_check(version):
    result = cargo(version, "clippy", ["--all-targets", "--frozen", "--target", args.target, "--message-format", "json"])
    warnings = parse_warnings(result, versions[version])
    rows = [{"file": key[0], "code": key[1], "message": key[2], "count": number} for key, number in sorted(warnings.items())]
    write_json(f"{version}-clippy-normalized.json", rows)
    metadata["versions"][version]["warnings"] = rows
    return {"warnings": sum(warnings.values()), "unique_diagnostics": len(warnings), "policy": "unchanged project policy; no -D warnings"}


def release_tests(version):
    result = cargo(version, "test", ["--release", "--frozen", "--target", args.target], timeout=900)
    require("test result: ok." in result.stdout, "No successful release test result")
    return {"summaries": [line for line in result.stdout.splitlines() if line.startswith("test result:")]}


def binary_path(version):
    return work / "targets" / version / args.target / "release" / ("srg.exe" if os.name == "nt" else "srg")


def release_build(version):
    cargo(version, "build", ["--release", "--frozen", "--bin", "srg", "--target", args.target], timeout=900)
    binary = binary_path(version)
    value = {"bytes": binary.stat().st_size, "sha256": digest(binary)}
    metadata["versions"][version]["binary"] = value
    return value


def streams_equal(left, right):
    while True:
        first = left.read(1024 * 1024)
        second = right.read(1024 * 1024)
        if first != second:
            return False
        if not first:
            return True


def package_roundtrip(version):
    package_dir = work / "packages" / version
    package_dir.mkdir(parents=True, exist_ok=True)
    cargo(version, "run", ["--frozen", "--manifest-path", versions[version] / "Cargo.toml", "--example", "package_artifact", "--", args.binary_name], label=f"{version}-official-package", cwd=package_dir, env=cargo_env(version, args.target))
    artifact = package_dir / "artifacts" / (args.binary_name + (".exe" if os.name == "nt" else ".tar"))
    binary = binary_path(version)
    if os.name == "nt":
        with artifact.open("rb") as packed, binary.open("rb") as original:
            require(streams_equal(packed, original), "Packaged executable differs from release binary")
    else:
        with tarfile.open(artifact, "r:") as archive:
            entries = archive.getmembers()
            require(len(entries) == 1, "Expected exactly one TAR entry")
            entry = entries[0]
            require(entry.name == args.binary_name and entry.isfile() and entry.mode == 0o755, "TAR entry identity or executable mode mismatch")
            require(entry.size == binary.stat().st_size and entry.mtime == 0, "TAR size or timestamp mismatch")
            with archive.extractfile(entry) as packed, binary.open("rb") as original:
                require(streams_equal(packed, original), "Independent TAR round-trip differs from release binary")
    value = {"bytes": artifact.stat().st_size, "sha256": digest(artifact), "independent_reader": "byte-for-byte" if os.name == "nt" else "Python tarfile and byte-for-byte"}
    metadata["versions"][version]["package"] = value
    require(digest(binary) == metadata["versions"][version]["binary"]["sha256"], "Release binary changed during packaging")
    return value


def cli_text(version, argument):
    directory = work / "cli" / version / argument.removeprefix("--")
    directory.mkdir(parents=True, exist_ok=True)
    result = execute(version + argument, [binary_path(version), argument], directory, timeout=90)
    require(result.stdout.strip(), f"No output from {argument}")
    metadata["versions"][version][argument] = result.stdout
    return {"stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(), "stderr_bytes": len(result.stderr.encode())}


def extract_records(filename):
    lines = filename.read_text(encoding="utf-8-sig").splitlines()
    records = []
    for index, line in enumerate(lines):
        if line.startswith("64비트 난수: "):
            record = lines[index:index + 17]
            require(len(record) == 17 and record[-1].startswith("NMS 은하 좌표: "), "Invalid 17-line TXT record")
            records.append("\n".join(record) + "\n")
    require(records, "No records found in generated TXT")
    return records


def manual_conversion(version):
    directory = work / "cli" / version / "manual"
    directory.mkdir(parents=True, exist_ok=True)
    text = "7\n3246858695411730098\n8468297219712610898\n0\n"
    result = execute(f"{version}-manual-menu", [binary_path(version)], directory, stdin=text, timeout=90)
    records = extract_records(directory / "random_data.txt")
    manual = records[-1]
    require(manual.startswith("64비트 난수: 3246858695411730098 "), "Last record is not the requested manual num64")
    require("supp 값 #1 입력" in result.stdout and "supp 값 #2 입력" not in result.stdout, "Manual supplemental prompt count changed")
    (evidence / f"{version}-manual-record.txt").write_text(manual, encoding="utf-8")
    metadata["versions"][version]["manual_record"] = manual
    return {"records_including_optional_startup_hardware": len(records), "manual_record_sha256": hashlib.sha256(manual.encode()).hexdigest(), "manual_supplements": 1}


def live_hardware(version):
    directory = work / "cli" / version / "live-hardware"
    directory.mkdir(parents=True, exist_ok=True)
    execute(f"{version}-live-generate-100", [binary_path(version), "generate", "100"], directory, stdin="", timeout=90)
    records = extract_records(directory / "random_data.txt")
    require(len(records) == 100, "Real hardware CLI did not save exactly 100 complete records")
    return {"records": len(records), "file_bytes": (directory / "random_data.txt").stat().st_size, "file_sha256": digest(directory / "random_data.txt")}


def historical_probe(version):
    source = versions[version]
    target = work / "probes" / version
    require(not target.exists(), "Probe copy unexpectedly exists")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "target", "artifacts", "node_modules", "random-order-native-evidence"))
    probe = repository / "tools" / "_random_order_probe_temp.rs"
    destination = target / "tools" / "_random_order_probe_temp.rs"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(probe, destination)
    test_module = target / "src" / "korea_boundary" / "tests.rs"
    with test_module.open("a", encoding="utf-8", newline="\n") as output:
        output.write('\ninclude!("../../tools/_random_order_probe_temp.rs");\n')
    result = execute(f"{version}-historical-probe", ["cargo", "+" + args.toolchain, "test", "--release", "--frozen", "--target", args.target, "--manifest-path", target / "Cargo.toml", "--bin", "srg", "verify_unified_shared_stream_ten_thousand", "--", "--nocapture", "--test-threads=1"], target, {"CARGO_TARGET_DIR": work / "targets" / ("probe-" + version)}, timeout=900)
    samples = []
    metric = None
    for line in result.stdout.splitlines():
        marker = "SRG_UNIFIED_SAMPLE\t"
        if marker in line:
            sample = line.split(marker, 1)[1]
            require(len(sample.split("\t")) == 21, "Probe sample column count changed")
            samples.append(sample)
        if "SRG_UNIFIED_METRIC " in line:
            metric = {name: int(value) for name, value in re.findall(r"([a-z_]+)=(\d+)", line.split("SRG_UNIFIED_METRIC ", 1)[1])}
    require(len(samples) == 128, "Expected 128 historical samples")
    expected = {"records": 10000, "supp_total": 14355, "supp_zero": 479, "supp_max": 14, "checksum": 9880379199121600078}
    require(metric is not None and all(metric.get(name) == value for name, value in expected.items()), "Historical probe counters differ from the frozen corpus")
    (evidence / f"{version}-historical-fixtures.tsv").write_text("\n".join(samples) + "\n", encoding="utf-8")
    write_json(f"{version}-historical-metrics.json", metric)
    metadata["versions"][version]["probe_metrics"] = metric
    metadata["versions"][version]["probe_sample_sha256"] = digest(evidence / f"{version}-historical-fixtures.tsv")
    return {"samples": len(samples), "fields": len(samples) * 21, "metrics": metric, "timing_claim": "none; correctness probe only"}


def equality(key):
    first = metadata["versions"]["baseline"][key]
    second = metadata["versions"]["candidate"][key]
    require(first == second, f"Baseline/candidate mismatch: {key}")
    return "exact equality"


def no_new_warnings():
    counts = {}
    for version in versions:
        counts[version] = Counter({(row["file"], row["code"], row["message"]): row["count"] for row in metadata["versions"][version]["warnings"]})
    added = counts["candidate"] - counts["baseline"]
    write_json("clippy-added-diagnostics.json", [{"file": key[0], "code": key[1], "message": key[2], "count": number} for key, number in sorted(added.items())])
    require(not added, "Candidate added project Clippy warning diagnostics")
    return {"baseline": sum(counts["baseline"].values()), "candidate": sum(counts["candidate"].values()), "added": 0}


def historical_counts():
    names = ["records", "supp_total", "supp_zero", "supp_max", "checksum"]
    first = metadata["versions"]["baseline"]["probe_metrics"]
    second = metadata["versions"]["candidate"]["probe_metrics"]
    require(all(first[name] == second[name] for name in names), "Historical metrics differ")
    return {name: first[name] for name in names}


def release_size():
    result = {}
    for kind in ["binary", "package"]:
        first = metadata["versions"]["baseline"][kind]["bytes"]
        second = metadata["versions"]["candidate"][kind]["bytes"]
        require(second <= first * 1.001, f"{kind} increased more than the predeclared 0.1 percent budget")
        result[kind] = {"baseline": first, "candidate": second, "delta": second - first, "percent": 100 * (second / first - 1)}
    return result


def source_preservation():
    for version, root in versions.items():
        require(snapshot(root) == metadata["versions"][version]["source_hashes"], f"{version} source/asset bytes changed during validation")
    return "All recorded product, source, dependency, and boundary hashes preserved"


persist()
check("toolchain_identity", toolchain_identity)
check("native_architecture", native_architecture)
check("baseline_checkout", baseline_checkout)
check("dependency_resolution", dependency_resolution)
for version in versions:
    for name, function in [("format", format_check), ("clippy", clippy_check), ("release_tests", release_tests), ("release_build", release_build), ("package_roundtrip", package_roundtrip), ("cli_help", lambda value: cli_text(value, "--help")), ("cli_version", lambda value: cli_text(value, "--version")), ("manual_conversion", manual_conversion), ("historical_probe", historical_probe)]:
        check(f"{version}.{name}", lambda version=version, function=function: function(version))
    if args.live_hardware:
        check(f"{version}.live_hardware_generate_100", lambda version=version: live_hardware(version))
for name, function in [("clippy_no_new_warnings", no_new_warnings), ("help_exact", lambda: equality("--help")), ("version_exact", lambda: equality("--version")), ("manual_record_exact", lambda: equality("manual_record")), ("historical_samples_exact", lambda: equality("probe_sample_sha256")), ("historical_counts_exact", historical_counts), ("release_size_limit", release_size), ("source_preservation", source_preservation)]:
    check(name, function)
passed = sum(value["status"] == "passed" for value in checks.values())
summary = {"passed": passed, "required": len(required), "failed": [name for name, result in checks.items() if result["status"] != "passed"], "complete": passed == len(required)}
write_json("summary.json", summary)
print(json.dumps(summary), flush=True)
sys.exit(0 if summary["complete"] else 1)
