import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
FROZEN = ROOT / 'evidence/perf4/before'
TOOLCHAIN = ROOT / 'toolchains/rustup/toolchains/1.99.1-x86_64-unknown-linux-gnu/bin'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


source = (FROZEN / 'src/korea_boundary/hash.rs').read_text()
assert source == (ROOT / 'srg/src/korea_boundary/hash.rs').read_text()
candidate = source.replace('use crate::diagnostic::Result;\n', '''#[cfg(target_arch = "x86_64")]
use std::is_x86_feature_detected;

use crate::diagnostic::Result;
''')
anchor = '    let [\n        mut aa,'
guard = '''    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sha") {
        unsafe { compress_sha(state, &words) };
        return;
    }
'''
assert candidate.count(anchor) == 1
candidate = candidate.replace(anchor, guard + anchor)
accelerated = '''#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha")]
fn compress_sha(state: &mut [u32; 8], words: &[u32; 64]) {
    use core::arch::x86_64::{
        _mm_cvtsi128_si32, _mm_set_epi32, _mm_sha256rnds2_epu32, _mm_shuffle_epi32,
    };

    let [aa, bb, cc, dd, ee, ff, gg, hh] = *state;
    let mut abef = _mm_set_epi32(aa.cast_signed(), bb.cast_signed(), ee.cast_signed(), ff.cast_signed());
    let mut cdgh = _mm_set_epi32(cc.cast_signed(), dd.cast_signed(), gg.cast_signed(), hh.cast_signed());
    for (block, constants) in words.as_chunks::<4>().0.iter().zip(ROUND.as_chunks::<4>().0) {
        let [word0, word1, word2, word3] = *block;
        let [constant0, constant1, constant2, constant3] = *constants;
        let combined = _mm_set_epi32(
            word3.wrapping_add(constant3).cast_signed(),
            word2.wrapping_add(constant2).cast_signed(),
            word1.wrapping_add(constant1).cast_signed(),
            word0.wrapping_add(constant0).cast_signed(),
        );
        cdgh = _mm_sha256rnds2_epu32(cdgh, abef, combined);
        abef = _mm_sha256rnds2_epu32(abef, cdgh, _mm_shuffle_epi32::<0x0E>(combined));
    }
    let values = [
        _mm_cvtsi128_si32(_mm_shuffle_epi32::<0xFF>(abef)).cast_unsigned(),
        _mm_cvtsi128_si32(_mm_shuffle_epi32::<0xAA>(abef)).cast_unsigned(),
        _mm_cvtsi128_si32(_mm_shuffle_epi32::<0xFF>(cdgh)).cast_unsigned(),
        _mm_cvtsi128_si32(_mm_shuffle_epi32::<0xAA>(cdgh)).cast_unsigned(),
        _mm_cvtsi128_si32(_mm_shuffle_epi32::<0x55>(abef)).cast_unsigned(),
        _mm_cvtsi128_si32(abef).cast_unsigned(),
        _mm_cvtsi128_si32(_mm_shuffle_epi32::<0x55>(cdgh)).cast_unsigned(),
        _mm_cvtsi128_si32(cdgh).cast_unsigned(),
    ];
    for (destination, value) in state.iter_mut().zip(values) {
        *destination = destination.wrapping_add(value);
    }
}

'''
candidate = candidate.replace('pub(super) fn digest(', accelerated + 'pub(super) fn digest(')
write(WORK / 'baseline.rs', source.encode())
write(WORK / 'candidate.rs', candidate.encode())
subprocess.run([str(TOOLCHAIN / 'rustfmt'), '--edition', '2024', str(WORK / 'candidate.rs')], check=True)
candidate_path = WORK / 'candidate.rs'
for name in ['product', 'probe']:
    target = WORK / name
    for path in (FROZEN / 'src').rglob('*'):
        if path.is_file():
            write(target / path.relative_to(FROZEN), path.read_bytes())
    for item in ['Cargo.toml', 'Cargo.lock', 'clippy.toml', 'README.md', 'LICENSE']:
        if (FROZEN / item).exists():
            write(target / item, (FROZEN / item).read_bytes())
    for path in (FROZEN / 'examples').rglob('*'):
        if path.is_file():
            write(target / path.relative_to(FROZEN), path.read_bytes())
    if not (target / 'data').exists():
        (target / 'data').symlink_to(FROZEN / 'data', target_is_directory=True)
    write(target / 'src/korea_boundary/hash.rs', candidate_path.read_bytes())
write(WORK / 'probe/src/main.rs', (ROOT / 'evidence/perf4-rust/baseline/src/main.rs').read_bytes())
report = {'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'baseline_source_sha256': digest(WORK / 'baseline.rs'),
          'candidate_pre_safety_comment_sha256': digest(candidate_path),
          'whole_probe_baseline': 'Original adopted perf4/before inflater; provisional bits+Adler is not included',
          'dispatch': 'Only x86_64 runtime SHA feature detection enters target_feature(sha); x86_64 SSE2 is baseline; other targets retain exact scalar rounds',
          'unchanged': ['64-word schedule', 'round constants', 'scalar rounds and feedforward', 'padding', 'length error', 'output byte order', 'all coordinate math/data/cache'],
          'added': ['one runtime feature guard per block', 'one target_feature SHA rounds-only function', 'SSE2 register set/shuffle/extract and SHA rounds2; no pointer loads/stores'],
          'new_heap_allocations': 0, 'new_dependencies': 0,
          'docs': ['https://doc.rust-lang.org/core/arch/x86_64/fn._mm_sha256rnds2_epu32.html', 'https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html'],
          'qualification_status': 'Preparation only; first confirm enabled undocumented_unsafe_blocks deny, then add its required concrete safety line and freeze formatted source'}
(WORK / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
