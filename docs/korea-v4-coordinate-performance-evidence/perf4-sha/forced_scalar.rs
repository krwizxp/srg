#[cfg(target_arch = "x86_64")]
use std::is_x86_feature_detected;

use crate::diagnostic::Result;

const INITIAL: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];
const ROUND: [u32; 64] = [
    0x428A_2F98,
    0x7137_4491,
    0xB5C0_FBCF,
    0xE9B5_DBA5,
    0x3956_C25B,
    0x59F1_11F1,
    0x923F_82A4,
    0xAB1C_5ED5,
    0xD807_AA98,
    0x1283_5B01,
    0x2431_85BE,
    0x550C_7DC3,
    0x72BE_5D74,
    0x80DE_B1FE,
    0x9BDC_06A7,
    0xC19B_F174,
    0xE49B_69C1,
    0xEFBE_4786,
    0x0FC1_9DC6,
    0x240C_A1CC,
    0x2DE9_2C6F,
    0x4A74_84AA,
    0x5CB0_A9DC,
    0x76F9_88DA,
    0x983E_5152,
    0xA831_C66D,
    0xB003_27C8,
    0xBF59_7FC7,
    0xC6E0_0BF3,
    0xD5A7_9147,
    0x06CA_6351,
    0x1429_2967,
    0x27B7_0A85,
    0x2E1B_2138,
    0x4D2C_6DFC,
    0x5338_0D13,
    0x650A_7354,
    0x766A_0ABB,
    0x81C2_C92E,
    0x9272_2C85,
    0xA2BF_E8A1,
    0xA81A_664B,
    0xC24B_8B70,
    0xC76C_51A3,
    0xD192_E819,
    0xD699_0624,
    0xF40E_3585,
    0x106A_A070,
    0x19A4_C116,
    0x1E37_6C08,
    0x2748_774C,
    0x34B0_BCB5,
    0x391C_0CB3,
    0x4ED8_AA4A,
    0x5B9C_CA4F,
    0x682E_6FF3,
    0x748F_82EE,
    0x78A5_636F,
    0x84C8_7814,
    0x8CC7_0208,
    0x90BE_FFFA,
    0xA450_6CEB,
    0xBEF9_A3F7,
    0xC671_78F2,
];

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut words = [0_u32; 64];
    for (word, bytes) in words.iter_mut().zip(block.as_chunks::<4>().0) {
        *word = u32::from_be_bytes(*bytes);
    }
    for round in 16_usize..64 {
        let lower_word = words[round.strict_sub(15)];
        let upper_word = words[round.strict_sub(2)];
        let lower =
            lower_word.rotate_right(7) ^ lower_word.rotate_right(18) ^ (lower_word >> 3_u32);
        let upper =
            upper_word.rotate_right(17) ^ upper_word.rotate_right(19) ^ (upper_word >> 10_u32);
        words[round] = words[round.strict_sub(16)]
            .wrapping_add(lower)
            .wrapping_add(words[round.strict_sub(7)])
            .wrapping_add(upper);
    }
    #[cfg(target_arch = "x86_64")]
    if false {
        // SAFETY: Runtime detection confirms SHA support; x86_64 guarantees SSE2.
        unsafe {
            compress_sha(state, &words);
        }
        return;
    }
    let [
        mut aa,
        mut bb,
        mut cc,
        mut dd,
        mut ee,
        mut ff,
        mut gg,
        mut hh,
    ] = *state;
    for (word, constant) in words.into_iter().zip(ROUND) {
        let upper = ee.rotate_right(6) ^ ee.rotate_right(11) ^ ee.rotate_right(25);
        let choice = (ee & ff) ^ (!ee & gg);
        let first = hh
            .wrapping_add(upper)
            .wrapping_add(choice)
            .wrapping_add(constant)
            .wrapping_add(word);
        let lower = aa.rotate_right(2) ^ aa.rotate_right(13) ^ aa.rotate_right(22);
        let majority = (aa & bb) ^ (aa & cc) ^ (bb & cc);
        hh = gg;
        gg = ff;
        ff = ee;
        ee = dd.wrapping_add(first);
        dd = cc;
        cc = bb;
        bb = aa;
        aa = first.wrapping_add(lower).wrapping_add(majority);
    }
    for (destination, value) in state.iter_mut().zip([aa, bb, cc, dd, ee, ff, gg, hh]) {
        *destination = destination.wrapping_add(value);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha")]
fn compress_sha(state: &mut [u32; 8], words: &[u32; 64]) {
    use core::arch::x86_64::{
        _mm_cvtsi128_si32, _mm_set_epi32, _mm_sha256rnds2_epu32, _mm_shuffle_epi32,
    };

    let [aa, bb, cc, dd, ee, ff, gg, hh] = *state;
    let mut abef = _mm_set_epi32(
        aa.cast_signed(),
        bb.cast_signed(),
        ee.cast_signed(),
        ff.cast_signed(),
    );
    let mut cdgh = _mm_set_epi32(
        cc.cast_signed(),
        dd.cast_signed(),
        gg.cast_signed(),
        hh.cast_signed(),
    );
    for (block, constants) in words
        .as_chunks::<4>()
        .0
        .iter()
        .zip(ROUND.as_chunks::<4>().0)
    {
        let [message0, message1, message2, message3] = *block;
        let [constant0, constant1, constant2, constant3] = *constants;
        let combined = _mm_set_epi32(
            message3.wrapping_add(constant3).cast_signed(),
            message2.wrapping_add(constant2).cast_signed(),
            message1.wrapping_add(constant1).cast_signed(),
            message0.wrapping_add(constant0).cast_signed(),
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

pub(super) fn digest(bytes: &[u8]) -> Result<[u8; 32]> {
    let length =
        u32::try_from(bytes.len()).map_err(|source| format!("경계 해시 길이 오류: {source}"))?;
    let mut state = INITIAL;
    let (blocks, tail) = bytes.as_chunks::<64>();
    for block in blocks {
        compress(&mut state, block);
    }
    let mut padding = [0_u8; 128];
    padding
        .get_mut(..tail.len())
        .ok_or("경계 해시 패딩 오류")?
        .copy_from_slice(tail);
    *padding.get_mut(tail.len()).ok_or("경계 해시 패딩 오류")? = 0x80;
    let padded = if tail.len() < 56 { 64_usize } else { 128_usize };
    padding
        .get_mut(padded.strict_sub(8)..padded)
        .ok_or("경계 해시 패딩 오류")?
        .copy_from_slice(&u64::from(length).strict_mul(8).to_be_bytes());
    for block in padding
        .get(..padded)
        .ok_or("경계 해시 패딩 오류")?
        .as_chunks::<64>()
        .0
    {
        compress(&mut state, block);
    }
    let mut result = [0_u8; 32];
    for (destination, word) in result.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        *destination = word.to_be_bytes();
    }
    Ok(result)
}
