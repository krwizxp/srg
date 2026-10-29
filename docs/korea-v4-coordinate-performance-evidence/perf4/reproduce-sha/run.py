import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import struct
import subprocess
import sys
import tempfile

SOURCE_HASHES = {'bench.rs': 'dcee3e8c4ef29cb92064bcc71c98e56c26c7f15c130998f7031da4b96ef95fb8', 'baseline.rs': '4be2e4e2396bcf9bdf43ac6a3e8296f2fd77930c966897186a634df2d5b6c1e6', 'candidate.rs': '90b580b4ff5c70529b7127a47742246141d39112add1efd361a9b10be092d8ef', 'forced_scalar.rs': '7d7ba0e9d0d8f684138a3c4d5d143cd134ad914077242553b7b9792651575b59', 'prepare_inputs.py': '6054d5640f250208845627543c9a3b94754f75829345188663f9dc12aad748fa', 'inputs.json': 'adfdf6946ec7de3ab9ad35cece7d70c0f7790992c3a36d94709dca6dd7635b71'}
MANIFEST_SHA256 = "037796f19af191f3b0ae19f6d9ba270004fe7e36e03e981ae9d1a5135187a2c6"
NORMALIZED_INPUT_SHA256 = "4cbdd29cc8dc2dc5251a21f6774298b133276179f63c64d2964b38052ceffc1f"
RUSTC_RELEASE = "1.99.1"
RUSTC_COMMIT = "58cf8f9c35b355199727627d07ae34a4e2846d67"
FLAGS = ["--edition=2024", "-C", "opt-level=3", "-C", "codegen-units=1", "-C", "lto=fat", "-C", "panic=abort", "-C", "strip=symbols"]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_sources():
    base = Path(__file__).resolve().parent
    result = {}
    for name, expected in SOURCE_HASHES.items():
        raw = (base / name).read_bytes()
        require(digest(raw) == expected, f"Bundled source checksum mismatch: {name}")
        result[name] = raw
    return result


def create_inputs(WORK, DATA):
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding="utf-8"))
    rows = []
    for entry in manifest['files']:
        path = DATA / entry['file']
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
        rows.append(['asset', entry['file'], str(path), entry['sha256']])
    vectors = {
        'empty': b'',
        'abc': b'abc',
        'fips_448_bits': b'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq',
        'million_a': b'a' * 1000000,
    }
    known = {
        'empty': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
        'abc': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
        'fips_448_bits': '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
        'million_a': 'cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0',
    }
    for length in range(258):
        vectors[f'zeros_{length}'] = bytes(length)
        vectors[f'ones_{length}'] = b'\xff' * length
        vectors[f'binary_{length}'] = bytes((index * 73 + 19) % 256 for index in range(length))
    rng = random.Random(20261029)
    random_lengths = []
    for index in range(256):
        length = rng.randrange(131073)
        random_lengths.append(length)
        vectors[f'random_{index}_{length}'] = rng.randbytes(length)
    (WORK / 'vectors').mkdir(exist_ok=True)
    for name, raw in vectors.items():
        sha = hashlib.sha256(raw).hexdigest()
        if name in known:
            assert sha == known[name]
        path = WORK / 'vectors' / f'{name}.bin'
        path.write_bytes(raw)
        rows.append(['vector', name, str(path), sha])
    states = []
    special = [[0] * 8, [2**32 - 1] * 8, [0xAAAAAAAA, 0x55555555] * 4,
               [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]]
    for index in range(4096):
        state = special[index] if index < len(special) else [rng.getrandbits(32) for _ in range(8)]
        states.append(struct.pack('<8I', *state) + rng.randbytes(64))
    (WORK / 'states.bin').write_bytes(b''.join(states))
    (WORK / 'inputs.tsv').write_text(''.join('\t'.join(row) + '\n' for row in rows), encoding='utf-8', newline='\n')
    source = (WORK / 'candidate.rs').read_text(encoding="utf-8")
    assert source.count('if is_x86_feature_detected!("sha") {') == 1
    (WORK / 'forced_scalar.rs').write_text(source.replace('if is_x86_feature_detected!("sha") {', 'if false {'), encoding='utf-8', newline='\n')
    report = {'asset_files': len(manifest['files']), 'asset_bytes': sum(row['bytes'] for row in manifest['files']),
              'known_vectors': known, 'hashlib_vectors': len(vectors),
              'padding_length_cases': {'inclusive_range': [0, 257], 'byte_patterns': ['all zero', 'all ff', 'i*73+19 mod256']},
              'random_seed': 20261029, 'random_lengths': random_lengths,
              'arbitrary_initial_state_compression_cases': 4096,
              'forced_scalar_change': 'Exactly one runtime feature condition changed from is_x86_feature_detected!("sha") to false; no product test switch',
              'file_sha256': {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in ['baseline.rs', 'candidate.rs', 'forced_scalar.rs', 'inputs.tsv', 'states.bin']},
              'expected_digest_source': 'Python hashlib.sha256, with four published known results independently asserted'}
    (WORK / 'inputs.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate the frozen SHA-only candidate on x86_64; no performance measurements.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--rustc", default="rustc")
    parser.add_argument("--rustc-arg", action="append", default=[], help="Optional Rustup selector: +1.99.1. Other compiler flags are fixed.")
    args = parser.parse_args()
    require(sys.version_info >= (3, 10), "Python 3.10 or newer is required.")
    require(platform.machine().lower() in {"x86_64", "amd64"}, "This diagnostic harness supports x86_64 hosts only.")
    require(args.rustc_arg in ([], ["+1.99.1"]), "Only the optional +1.99.1 Rustup selector is accepted; release flags are fixed.")
    sources = verified_sources()
    expected = json.loads(sources["inputs.json"])
    repo = args.repo.resolve(strict=True)
    require(not any(value in str(repo) for value in "\t\r\n"), "The repository path cannot contain tab or newline characters.")
    data = repo / "data/korea-boundary-v4"
    require(digest((data / "manifest.json").read_bytes()) == MANIFEST_SHA256, "Repository boundary manifest differs from the frozen input.")
    require(digest((repo / "src/korea_boundary/hash.rs").read_bytes()) == SOURCE_HASHES["candidate.rs"], "Repository hash.rs differs from the validated SHA-only candidate.")
    compiler_path = shutil.which(args.rustc)
    require(compiler_path is not None, "The requested rustc executable was not found.")
    prefix = [str(Path(compiler_path).absolute()), *args.rustc_arg]
    compiler = subprocess.run([*prefix, "-Vv"], check=True, capture_output=True, text=True, encoding="utf-8").stdout
    compiler_fields = dict(line.split(": ", 1) for line in compiler.splitlines() if ": " in line)
    require(compiler_fields.get("release") == RUSTC_RELEASE, "Use the pinned Rust 1.99.1 toolchain.")
    require(compiler_fields.get("commit-hash") == RUSTC_COMMIT, "Rust compiler commit differs from the verified cycle.")
    require(compiler_fields.get("host", "").startswith("x86_64-"), "Use a native x86_64 Rust compiler.")
    with tempfile.TemporaryDirectory(prefix="srg-sha-validate-") as temporary:
        work = Path(temporary)
        require(not any(value in str(work) for value in "\t\r\n"), "The temporary directory cannot contain tab or newline characters.")
        for name in ["bench.rs", "baseline.rs", "candidate.rs", "forced_scalar.rs"]:
            (work / name).write_bytes(sources[name])
        generated = create_inputs(work, data)
        for name in ["baseline.rs", "candidate.rs", "forced_scalar.rs", "bench.rs"]:
            require(digest((work / name).read_bytes()) == SOURCE_HASHES[name], f"Temporary source checksum mismatch: {name}")
        for name in ["asset_files", "asset_bytes", "known_vectors", "hashlib_vectors", "padding_length_cases", "random_seed", "random_lengths", "arbitrary_initial_state_compression_cases"]:
            require(generated[name] == expected[name], f"Regenerated input metadata differs: {name}")
        require(generated["file_sha256"]["states.bin"] == expected["file_sha256"]["states.bin"], "Regenerated initial-state corpus checksum differs.")
        rows = [line.split("\t") for line in (work / "inputs.tsv").read_text(encoding="utf-8").splitlines()]
        require(all(len(row) == 4 for row in rows), "Unexpected input row layout.")
        normalized = "".join("\t".join([row[0], row[1], row[3]]) + "\n" for row in rows).encode("utf-8")
        require(digest(normalized) == NORMALIZED_INPUT_SHA256, "Regenerated input names/order/digests differ from the original corpus.")
        executable = "validate.exe" if sys.platform == "win32" else "validate"
        command = [*prefix, *FLAGS, "--crate-name", "srg_perf4_sha", "bench.rs", "-o", executable]
        build = subprocess.run(command, cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if build.returncode:
            sys.stderr.write(build.stdout + build.stderr)
            raise ValueError(f"rustc exited with status {build.returncode}.")
        run = subprocess.run([str(work / executable), "validate", str(work / "inputs.tsv"), str(work / "states.bin")], cwd=work, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if run.returncode:
            sys.stderr.write(run.stdout + run.stderr)
            raise ValueError(f"Validation exited with status {run.returncode}.")
        result = json.loads(run.stdout)
        require(result.get("mode") == "validate", "Unexpected harness mode.")
        for name, value in {"asset_files": 97, "vectors": 1034, "arbitrary_state_blocks": 4096, "digest_mismatches": 0, "compression_mismatches": 0}.items():
            require(result.get(name) == value, f"Unexpected validation result: {name}")
        require(type(result.get("runtime_sha")) is bool and type(result.get("runtime_sse2")) is bool, "Missing runtime feature detection result.")
        report = {
            "status": "passed",
            "scope": "SHA-only diagnostic validation, not a performance measurement or the product platform acceptance matrix",
            "supported_host_architecture": "x86_64",
            "compiler": compiler,
            "python": platform.python_version(),
            "source_sha256": SOURCE_HASHES,
            "runner_sha256": digest(Path(__file__).read_bytes()),
            "repository_candidate_sha256": SOURCE_HASHES["candidate.rs"],
            "manifest_sha256": MANIFEST_SHA256,
            "normalized_input_sha256": NORMALIZED_INPUT_SHA256,
            "state_corpus_sha256": generated["file_sha256"]["states.bin"],
            "compile_command": command,
            "compiler_stdout": build.stdout,
            "compiler_stderr": build.stderr,
            "validation_stderr": run.stderr,
            "validation": result,
            "accelerated_path_executed": result["runtime_sha"],
            "coverage_note": "SHA-NI and forced scalar paths both executed." if result["runtime_sha"] else "SHA is unavailable: validation passed via scalar dispatch; this is not accelerated-path execution evidence.",
            "performance_measurements": 0,
            "temporary_inputs_removed_on_exit": True,
            "repository_files_modified": False,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, AssertionError, subprocess.CalledProcessError) as error:
        print(f"SHA validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
