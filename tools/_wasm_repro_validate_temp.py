import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--toolchain", default="1.99.1")
args = parser.parse_args()
root = Path.cwd().resolve()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
environment = os.environ.copy()
environment["CARGO_TARGET_DIR"] = str(out.parent / "coordinate-wasm-repro-target")
environment["CARGO_TERM_COLOR"] = "never"
names = ["toolchain_identity", "shared_source_format", "wasm_clippy_policy", "reproducible_build",
         "delivery_binding_and_size", "scalar_abi_contracts", "shared_source_identity", "source_preservation"]
checks = {name: {"status": "pending"} for name in names}
sequence = 0

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot():
    paths = sorted(root.glob("src/**/*.rs"))
    paths += [root / name for name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "revgeo.html",
              "korea_boundary.js", "coordinates_wasm.js", "coordinates.wasm", "README.md", "tools/build_coordinates_wasm.py"]]
    return {str(path.relative_to(root)): digest(path) for path in paths}

before = snapshot()
(out / "source-before.json").write_text(json.dumps(before, indent=2) + "\n")

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def run(label, command, timeout=600):
    global sequence
    sequence += 1
    stem = f"{sequence:03d}-{label}"
    started = time.monotonic()
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    (out / (stem + ".stdout.log")).write_text(result.stdout)
    (out / (stem + ".stderr.log")).write_text(result.stderr)
    (out / (stem + ".command.json")).write_text(json.dumps({"command": command, "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
    require(result.returncode == 0, label + " failed; see " + stem)
    return result

def toolchain_identity():
    result = run("rustc", ["rustc", "+" + args.toolchain, "-Vv"])
    require(re.search(r"^release: " + re.escape(args.toolchain) + "$", result.stdout, re.MULTILINE), "Compiler version mismatch")
    return result.stdout

def format_source():
    run("format", ["cargo", "+" + args.toolchain, "fmt", "--all", "--", "--check"])
    return "All Rust source format checked"

def wasm_clippy():
    result = run("wasm-clippy", ["cargo", "+" + args.toolchain, "clippy", "--frozen", "--release",
        "--target", "wasm32-unknown-unknown", "--lib", "--message-format", "json"])
    warnings = Counter()
    for line in result.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("reason") != "compiler-message":
            continue
        message = value["message"]
        require(message["level"] != "error", "WASM compiler emitted an error")
        if message["level"] != "warning":
            continue
        spans = message.get("spans", [])
        span = next((item for item in spans if item.get("is_primary")), spans[0] if spans else {})
        filename = span.get("file_name", "<none>").replace("\\", "/")
        prefix = str(root).replace("\\", "/") + "/"
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
        warnings[(filename, (message.get("code") or {}).get("code", "<none>"), message["message"])] += 1
    specification = json.loads((root / "tools/_wasm_reviewed_diagnostics.json").read_text())
    reviewed = Counter({(row["file"], row["code"], row["message"]): row["count"] for row in specification["wasm_warnings"]})
    def rows(values):
        return [{"file": key[0], "code": key[1], "message": key[2], "count": count} for key, count in sorted(values.items())]
    (out / "wasm-clippy-normalized.json").write_text(json.dumps(rows(warnings), indent=2) + "\n")
    (out / "wasm-clippy-unexpected.json").write_text(json.dumps(rows(warnings - reviewed), indent=2) + "\n")
    require(not (warnings - reviewed), "Unexpected WASM Clippy diagnostics outside frozen review")
    return {"warnings": sum(warnings.values()), "unexpected": 0, "policy": "Project lint levels unchanged; no warning suppression"}

def reproducible_build():
    run("canonical-build", [sys.executable, "tools/build_coordinates_wasm.py", "--toolchain", args.toolchain])
    actual = digest(root / "coordinates.wasm")
    require(actual == before["coordinates.wasm"], "Pinned compiler build differs from tracked WASM artifact")
    require(digest(root / "coordinates_wasm.js") == before["coordinates_wasm.js"], "Build changed loader binding")
    return {"wasm_sha256": actual, "reproduction": "Clean independent hosted runner, canonical build helper, same pinned release profile"}

def delivery():
    source = (root / "coordinates_wasm.js").read_text()
    match = re.search(r'^const WASM_SHA256 = "([0-9a-f]{64})";', source, re.MULTILINE)
    require(match and match[1] == digest(root / "coordinates.wasm"), "Loader does not bind artifact SHA-256")
    sizes = {name: (root / name).stat().st_size for name in ["coordinates.wasm", "coordinates_wasm.js"]}
    require(sum(sizes.values()) <= 262144, "Combined module and loader exceeds predeclared byte budget")
    return {"sizes": sizes, "combined_bytes": sum(sizes.values()), "maximum_bytes": 262144}

def abi_contracts():
    result = run("wasm-abi", ["node", "tools/_wasm_abi_check_temp.mjs", str(root / "coordinates.wasm")])
    (out / "wasm-abi-result.json").write_text(result.stdout)
    return {"result": result.stdout.strip()}

def shared_identity():
    modules = ["coordinate_core", "geo_uniform", "boundary_inflate"]
    for entry in ["src/main.rs", "src/lib.rs"]:
        source = (root / entry).read_text()
        for module in modules:
            require("mod " + module + ";" in source, entry + " does not compile shared module " + module)
    return {name: digest(root / ("src/" + name + ".rs")) for name in modules}

def source_preservation():
    after = snapshot()
    (out / "source-after.json").write_text(json.dumps(after, indent=2) + "\n")
    require(after == before, "Source or shipped bytes changed during reproduction")
    return "All recorded product source and delivered bytes unchanged"

for name, function in zip(names, [toolchain_identity, format_source, wasm_clippy, reproducible_build,
                                delivery, abi_contracts, shared_identity, source_preservation]):
    started = time.monotonic()
    try:
        detail = function()
        checks[name] = {"status": "passed", "detail": detail}
    except Exception as error:
        checks[name] = {"status": "failed", "error": str(error)}
        (out / (name + ".exception.txt")).write_text(traceback.format_exc())
    checks[name]["elapsed_seconds"] = time.monotonic() - started
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"check": name, **checks[name]}), flush=True)
passed = sum(value["status"] == "passed" for value in checks.values())
summary = {"passed": passed, "required": len(names), "complete": passed == len(names)}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
raise SystemExit(0 if summary["complete"] else 1)
