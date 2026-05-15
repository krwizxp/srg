#[cfg(target_arch = "x86_64")]
use super::BYTE_BITS;
#[cfg(target_arch = "x86_64")]
use super::hardware_rng::get_hardware_random;
use super::{
    ASCII_PRINTABLE_LEN, ASCII_PRINTABLE_START, EURO_LUCKY_MODULUS, EURO_MAIN_MODULUS,
    EURO_MILLIONS_LUCKY_COUNT, EURO_MILLIONS_MAIN_COUNT, HANGUL_BASE_CODE_POINT, HANGUL_SHIFTS,
    HANGUL_SYLLABLE_COUNT, HANGUL_SYLLABLE_MAX, HANGUL_SYLLABLE_MODULUS,
    INPUT_BYTE_MAX_FOR_EURO_MAIN, INPUT_BYTE_MAX_FOR_LOTTO, INPUT_BYTE_MAX_FOR_LOTTO7,
    INPUT_BYTE_MAX_FOR_LUCKY_STAR, INPUT_BYTE_MAX_FOR_PASSWORD, LOTTO_COUNT, LOTTO_COUNT_U8,
    LOTTO_MODULUS, LOTTO7_COUNT, LOTTO7_MODULUS, NMS_COORD_MASK, NMS_GLYPH_COUNT,
    NMS_GLYPH_NUM_SHIFTS, NMS_GLYPH_PREFIX_COUNT, NMS_PLANET_MAX_VALUE, NMS_PLANET_MODULUS,
    NMS_SOLAR_SYSTEM_MAX_VALUE, NMS_SOLAR_SYSTEM_MODULUS, PASSWORD_BYTE_LEN, PASSWORD_BYTE_LEN_U8,
    Result, U32_MAX_INV,
    numeric::{low_u8_from_u32, low_u8_from_u64, low_u16_from_u64},
    random_util::{
        checked_add_one_u8, checked_add_one_u64, checked_add_one_usize, galaxy_coord,
        glyph_from_low_nibble, split_u64_to_u32_pair,
    },
};
use core::{char::from_u32, ops::Mul as _};
use std::io::Error as IoError;
#[derive(Default)]
pub struct RandomDataSet {
    pub euro_lucky_next_idx: usize,
    pub euro_main_next_idx: usize,
    pub euro_millions_lucky_stars: [u8; EURO_MILLIONS_LUCKY_COUNT],
    pub euro_millions_main_numbers: [u8; EURO_MILLIONS_MAIN_COUNT],
    pub galaxy_x: u16,
    pub galaxy_y: u16,
    pub galaxy_z: u16,
    pub glyph_string: [char; NMS_GLYPH_COUNT],
    pub hangul_syllables: [char; HANGUL_SYLLABLE_COUNT],
    pub kor_coords: (f64, f64),
    pub lotto7_next_idx: usize,
    pub lotto7_numbers: [u8; LOTTO7_COUNT],
    pub lotto_next_idx: usize,
    pub lotto_numbers: [u8; LOTTO_COUNT],
    pub nms_portal_xxx: u16,
    pub nms_portal_yy: u8,
    pub nms_portal_zzz: u16,
    pub num_64: u64,
    pub numeric_password: u32,
    pub numeric_password_digits: u8,
    pub password: [u8; PASSWORD_BYTE_LEN],
    pub password_len: u8,
    pub planet_number: u8,
    pub seen_euro_millions_lucky: u64,
    pub seen_euro_millions_main: u64,
    pub seen_lotto: u64,
    pub seen_lotto7: u64,
    pub solar_system_index: u16,
    pub world_coords: (f64, f64),
}
impl RandomDataSet {
    const fn is_complete(&self) -> bool {
        self.numeric_password_digits >= LOTTO_COUNT_U8
            && self.lotto_next_idx >= LOTTO_COUNT
            && self.lotto7_next_idx >= LOTTO7_COUNT
            && self.password_len >= PASSWORD_BYTE_LEN_U8
            && self.euro_main_next_idx >= EURO_MILLIONS_MAIN_COUNT
    }
}
#[derive(Clone, Copy)]
pub struct RandomBitBuffer {
    bits_remaining: u8,
    value: u64,
}
impl RandomBitBuffer {
    pub const fn bits_remaining(self) -> u8 {
        self.bits_remaining
    }
    pub fn consume_bits(&mut self, bits: u8, reason: &'static str) -> Result<u64> {
        let Some(shift) = self.bits_remaining.checked_sub(bits) else {
            return Err(format!(
                "보완 난수 비트 수가 부족합니다. (remaining={}, required={bits}, reason={reason})",
                self.bits_remaining
            )
            .into());
        };
        self.bits_remaining = shift;
        Ok(self.value >> shift)
    }
    pub const fn new(value: u64, bits_remaining: u8) -> Self {
        Self {
            bits_remaining,
            value,
        }
    }
    pub const fn value(self) -> u64 {
        self.value
    }
}
pub type SupplementalProvider<'a> = dyn FnMut(&'static str) -> Result<RandomBitBuffer> + 'a;
struct RandomDataBuilder<'a, 'b> {
    data: RandomDataSet,
    next_supp: &'a mut SupplementalProvider<'b>,
    num: u64,
    supplemental: Option<RandomBitBuffer>,
}
impl RandomDataBuilder<'_, '_> {
    fn build(mut self) -> Result<RandomDataSet> {
        self.fill_required_fields()?;
        self.fill_lucky_stars()?;
        self.fill_hangul_syllables()?;
        self.fill_coords();
        self.fill_nms_fields()?;
        Ok(self.data)
    }
    fn fill_coords(&mut self) {
        let (upper_32_bits, lower_32_bits) = split_u64_to_u32_pair(self.num);
        let upper_ratio = f64::from(upper_32_bits).mul(U32_MAX_INV);
        let lower_ratio = f64::from(lower_32_bits).mul(U32_MAX_INV);
        self.data.kor_coords = (
            5.504_167_f64.mul_add(upper_ratio, 33.112_500),
            7.263_056_f64.mul_add(lower_ratio, 124.609_722),
        );
        self.data.world_coords = (
            180.0_f64.mul_add(upper_ratio, -90.0),
            360.0_f64.mul_add(lower_ratio, -180.0),
        );
        self.data.nms_portal_yy = low_u8_from_u32(upper_32_bits);
        self.data.nms_portal_zzz = low_u16_from_u64((self.num >> 20) & NMS_COORD_MASK);
        self.data.nms_portal_xxx = low_u16_from_u64((self.num >> 8) & NMS_COORD_MASK);
        self.data.galaxy_x = galaxy_coord::<0x801, 0x7FF>(self.data.nms_portal_xxx);
        self.data.galaxy_y = galaxy_coord::<0x81, 0x7F>(u16::from(self.data.nms_portal_yy));
        self.data.galaxy_z = galaxy_coord::<0x801, 0x7FF>(self.data.nms_portal_zzz);
    }
    fn fill_hangul_syllables(&mut self) -> Result<()> {
        let mut hangul = ['\0'; HANGUL_SYLLABLE_COUNT];
        for (slot, shift) in hangul.iter_mut().zip(HANGUL_SHIFTS) {
            let mut syllable_index = u32::from(low_u16_from_u64(self.num >> shift));
            while syllable_index > HANGUL_SYLLABLE_MAX {
                if self.supplemental.is_none() {
                    self.supplemental = Some(self.next_supplemental("한글 음절 보완")?);
                }
                let Some(supplemental) = self.supplemental else {
                    return Err("한글 음절 보완 상태 불일치".into());
                };
                let supp_value = supplemental.value();
                let candidate_value = HANGUL_SHIFTS
                    .into_iter()
                    .map(|supp_shift| u32::from(low_u16_from_u64(supp_value >> supp_shift)))
                    .find(|value| *value <= HANGUL_SYLLABLE_MAX);
                if let Some(candidate) = candidate_value {
                    syllable_index = candidate;
                } else {
                    self.supplemental = Some(self.next_supplemental("한글 음절 보완 재시도")?);
                }
            }
            let Some(code_point) = HANGUL_BASE_CODE_POINT
                .checked_add(syllable_index.rem_euclid(HANGUL_SYLLABLE_MODULUS))
            else {
                return Err("한글 음절 코드포인트 계산 실패".into());
            };
            *slot = from_u32(code_point).ok_or("한글 음절 변환 실패")?;
        }
        self.data.hangul_syllables = hangul;
        Ok(())
    }
    fn fill_lucky_stars(&mut self) -> Result<()> {
        let lucky_star_base = if let Some(supp) = self.supplemental.as_ref() {
            supp.value()
        } else {
            self.num
        };
        let mut lucky_star_source = lucky_star_base.reverse_bits();
        'lucky_star_loop: loop {
            for byte in lucky_star_source.to_be_bytes() {
                if byte > INPUT_BYTE_MAX_FOR_LUCKY_STAR {
                    continue;
                }
                if process_lotto_numbers(
                    byte,
                    EURO_LUCKY_MODULUS,
                    &mut self.data.euro_millions_lucky_stars,
                    &mut self.data.seen_euro_millions_lucky,
                    &mut self.data.euro_lucky_next_idx,
                ) && self.data.euro_lucky_next_idx >= EURO_MILLIONS_LUCKY_COUNT
                {
                    break 'lucky_star_loop;
                }
            }
            lucky_star_source = self
                .next_supplemental("유로밀리언 럭키 스타 보완")?
                .value()
                .reverse_bits();
        }
        Ok(())
    }
    fn fill_nms_fields(&mut self) -> Result<()> {
        let planet_number_base = extract_valid_bits_for_nms::<4>(
            self.num,
            &[52, 4, 0],
            NMS_PLANET_MAX_VALUE,
            "NMS 행성 번호 보완",
            &mut self.supplemental,
            self.next_supp,
        )?
        .rem_euclid(NMS_PLANET_MODULUS);
        let planet_number = checked_add_one_u64(planet_number_base, "NMS 행성 번호 계산 실패")?;
        self.data.planet_number = low_u8_from_u64(planet_number);
        let solar_system_index_base = extract_valid_bits_for_nms::<12>(
            self.num,
            &[40],
            NMS_SOLAR_SYSTEM_MAX_VALUE,
            "NMS 태양계 번호 보완",
            &mut self.supplemental,
            self.next_supp,
        )?
        .rem_euclid(NMS_SOLAR_SYSTEM_MODULUS);
        let solar_system_index =
            checked_add_one_u64(solar_system_index_base, "NMS 태양계 번호 계산 실패")?;
        self.data.solar_system_index = low_u16_from_u64(solar_system_index);
        let Some((prefix_glyphs, suffix_glyphs)) = self
            .data
            .glyph_string
            .split_first_chunk_mut::<NMS_GLYPH_PREFIX_COUNT>()
        else {
            return Err(IoError::other("NMS glyph prefix 길이가 올바르지 않습니다.").into());
        };
        for (slot, nibble_source) in prefix_glyphs.iter_mut().zip([
            u64::from(self.data.planet_number),
            u64::from(self.data.solar_system_index >> 8_u32),
            u64::from(self.data.solar_system_index >> 4_u32),
            u64::from(self.data.solar_system_index),
        ]) {
            let Some(glyph) = glyph_from_low_nibble(nibble_source) else {
                continue;
            };
            *slot = glyph;
        }
        for (slot, shift) in suffix_glyphs.iter_mut().zip(NMS_GLYPH_NUM_SHIFTS) {
            let Some(glyph) = glyph_from_low_nibble(self.num >> shift) else {
                continue;
            };
            *slot = glyph;
        }
        Ok(())
    }
    fn fill_required_fields(&mut self) -> Result<()> {
        fill_data_fields_from_u64(self.num, &mut self.data);
        while !self.data.is_complete() {
            let new_supp = self.next_supplemental("기본 필드 보완")?;
            fill_data_fields_from_u64(new_supp.value(), &mut self.data);
        }
        Ok(())
    }
    fn next_supplemental(&mut self, reason: &'static str) -> Result<RandomBitBuffer> {
        let supplemental = (self.next_supp)(reason)?;
        self.supplemental = Some(supplemental);
        Ok(supplemental)
    }
}
pub fn generate_random_data_from_num(
    num: u64,
    next_supp: &mut SupplementalProvider<'_>,
) -> Result<RandomDataSet> {
    RandomDataBuilder {
        data: RandomDataSet {
            num_64: num,
            ..Default::default()
        },
        next_supp,
        num,
        supplemental: None,
    }
    .build()
}
#[cfg(target_arch = "x86_64")]
pub fn generate_random_data() -> Result<RandomDataSet> {
    let num = get_hardware_random()?;
    let mut next_supp =
        |_reason: &'static str| Ok(RandomBitBuffer::new(get_hardware_random()?, BYTE_BITS));
    generate_random_data_from_num(num, &mut next_supp)
}
fn fill_data_fields_from_u64(value: u64, data: &mut RandomDataSet) {
    for byte in value.to_be_bytes() {
        if byte > INPUT_BYTE_MAX_FOR_EURO_MAIN {
            continue;
        }
        if data.numeric_password_digits < LOTTO_COUNT_U8 && {
            let digit = u32::from(byte.rem_euclid(10));
            let Some(next_password) = data
                .numeric_password
                .checked_mul(10)
                .and_then(|current_password| current_password.checked_add(digit))
            else {
                return;
            };
            data.numeric_password = next_password;
            let Some(next_digit_count) = checked_add_one_u8(data.numeric_password_digits) else {
                return;
            };
            data.numeric_password_digits = next_digit_count;
            next_digit_count >= LOTTO_COUNT_U8 && data.is_complete()
        } {
            return;
        }
        let euro_main_added = data.euro_main_next_idx < EURO_MILLIONS_MAIN_COUNT
            && process_lotto_numbers(
                byte,
                EURO_MAIN_MODULUS,
                &mut data.euro_millions_main_numbers,
                &mut data.seen_euro_millions_main,
                &mut data.euro_main_next_idx,
            );
        if euro_main_added
            && data.euro_main_next_idx >= EURO_MILLIONS_MAIN_COUNT
            && data.is_complete()
        {
            return;
        }
        if byte > INPUT_BYTE_MAX_FOR_LOTTO {
            continue;
        }
        let lotto_added = data.lotto_next_idx < LOTTO_COUNT
            && process_lotto_numbers(
                byte,
                LOTTO_MODULUS,
                &mut data.lotto_numbers,
                &mut data.seen_lotto,
                &mut data.lotto_next_idx,
            );
        if lotto_added && data.lotto_next_idx >= LOTTO_COUNT && data.is_complete() {
            return;
        }
        if byte > INPUT_BYTE_MAX_FOR_LOTTO7 {
            continue;
        }
        let lotto7_number_added = data.lotto7_next_idx < LOTTO7_COUNT
            && process_lotto_numbers(
                byte,
                LOTTO7_MODULUS,
                &mut data.lotto7_numbers,
                &mut data.seen_lotto7,
                &mut data.lotto7_next_idx,
            );
        if lotto7_number_added && data.lotto7_next_idx >= LOTTO7_COUNT && data.is_complete() {
            return;
        }
        if byte > INPUT_BYTE_MAX_FOR_PASSWORD {
            continue;
        }
        if data.password_len < PASSWORD_BYTE_LEN_U8
            && let Some(password_byte) = byte
                .rem_euclid(ASCII_PRINTABLE_LEN)
                .checked_add(ASCII_PRINTABLE_START)
            && let Some(slot) = data.password.get_mut(usize::from(data.password_len))
        {
            *slot = password_byte;
            let Some(next_password_len) = checked_add_one_u8(data.password_len) else {
                return;
            };
            data.password_len = next_password_len;
            if next_password_len >= PASSWORD_BYTE_LEN_U8 && data.is_complete() {
                return;
            }
        }
    }
}
fn process_lotto_numbers(
    byte: u8,
    modulus: u8,
    numbers: &mut [u8],
    seen: &mut u64,
    next_idx: &mut usize,
) -> bool {
    if *next_idx >= numbers.len() {
        return false;
    }
    let Some(number) = byte.checked_rem(modulus).and_then(checked_add_one_u8) else {
        return false;
    };
    let mask = 1_u64 << number;
    if (*seen & mask) != 0 {
        return false;
    }
    let Some(slot) = numbers.get_mut(*next_idx) else {
        return false;
    };
    *slot = number;
    *seen |= mask;
    let Some(next_index) = checked_add_one_usize(*next_idx) else {
        return false;
    };
    *next_idx = next_index;
    if *next_idx == numbers.len() {
        numbers.sort_unstable();
    }
    true
}
fn extract_valid_bits_for_nms<const BITS: u8>(
    num: u64,
    shifts: &[u8],
    max_value: u64,
    reason: &'static str,
    supplemental: &mut Option<RandomBitBuffer>,
    next_supp: &mut SupplementalProvider<'_>,
) -> Result<u64> {
    let bit_count = BITS.min(64);
    let mask = match bit_count {
        0 => 0,
        64 => u64::MAX,
        _ => 1_u64
            .checked_shl(u32::from(bit_count))
            .map_or(0, |shifted| shifted.saturating_sub(1)),
    };
    for &shift in shifts {
        let extracted_value = (num >> shift) & mask;
        if extracted_value <= max_value {
            return Ok(extracted_value);
        }
    }
    loop {
        let need_new = supplemental
            .as_ref()
            .is_none_or(|supp| supp.bits_remaining() < BITS);
        if need_new {
            *supplemental = Some(next_supp(reason)?);
        }
        let Some(supp) = supplemental.as_mut() else {
            return Err("보완 난수 상태 불일치".into());
        };
        let extracted = supp.consume_bits(BITS, reason)? & mask;
        if extracted <= max_value {
            return Ok(extracted);
        }
    }
}
