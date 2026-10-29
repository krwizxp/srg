import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--toolchain", default="stable")
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
subprocess.run([
    "cargo", "+" + args.toolchain, "build", "--locked", "--release",
    "--target", "wasm32-unknown-unknown", "--lib",
], cwd=root, check=True)
target = Path(os.environ.get("CARGO_TARGET_DIR", root / "target"))
if not target.is_absolute():
    target = root / target
binary = target / "wasm32-unknown-unknown/release/coordinates.wasm"
payload = binary.read_bytes()
if len(payload) > 262144:
    raise SystemExit("Coordinate WASM exceeds 256 KiB")
digest = hashlib.sha256(payload).hexdigest()
loader = root / "coordinates_wasm.js"
source, replacements = re.subn(
    r'const WASM_SHA256 = "[0-9a-f]{64}";',
    'const WASM_SHA256 = "' + digest + '";',
    loader.read_text(),
)
if replacements != 1:
    raise SystemExit("Expected one coordinate WASM digest in the loader")
shutil.copyfile(binary, root / "coordinates.wasm")
loader.write_text(source)
print(f"coordinates.wasm: {len(payload)} bytes; SHA-256 {digest}")
