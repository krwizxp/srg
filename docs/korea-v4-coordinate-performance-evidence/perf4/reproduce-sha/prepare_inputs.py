import hashlib
import json
from pathlib import Path
import random
import struct

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
DATA = ROOT / 'evidence/perf4/before/data/korea-boundary-v4'
manifest = json.loads((DATA / 'manifest.json').read_text())
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
(WORK / 'inputs.tsv').write_text(''.join('\t'.join(row) + '\n' for row in rows))
source = (WORK / 'candidate.rs').read_text()
assert source.count('if is_x86_feature_detected!("sha") {') == 1
(WORK / 'forced_scalar.rs').write_text(source.replace('if is_x86_feature_detected!("sha") {', 'if false {'))
report = {'asset_files': len(manifest['files']), 'asset_bytes': sum(row['bytes'] for row in manifest['files']),
          'known_vectors': known, 'hashlib_vectors': len(vectors),
          'padding_length_cases': {'inclusive_range': [0, 257], 'byte_patterns': ['all zero', 'all ff', 'i*73+19 mod256']},
          'random_seed': 20261029, 'random_lengths': random_lengths,
          'arbitrary_initial_state_compression_cases': 4096,
          'forced_scalar_change': 'Exactly one runtime feature condition changed from is_x86_feature_detected!("sha") to false; no product test switch',
          'file_sha256': {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in ['baseline.rs', 'candidate.rs', 'forced_scalar.rs', 'inputs.tsv', 'states.bin']},
          'expected_digest_source': 'Python hashlib.sha256, with four published known results independently asserted'}
(WORK / 'inputs.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in ['random_lengths', 'known_vectors']}, indent=2))
