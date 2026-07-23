use crate::{
    diagnostic::Result,
    numeric::{low_u8_from_u32, low_u8_from_u64, low_u16_from_u64},
};
cfg_select! {
    target_arch = "x86_64" => {
        use super::hardware_rng::HardwareRng;
        use crate::diagnostic::AppError;
    }
    _ => {}
}
use core::{num::NonZeroU8, ops::Mul as NumericMul};
const ASCII_PRINTABLE_LEN: u8 = 94;
const ASCII_PRINTABLE_START: u8 = 33;
const BYTE_BITS: u8 = 64;
const EURO_LUCKY_MODULUS: NonZeroU8 = NonZeroU8::MIN.saturating_add(11);
const EURO_MAIN_MODULUS: NonZeroU8 = NonZeroU8::MIN.saturating_add(49);
const EURO_MILLIONS_LUCKY_COUNT: usize = 2;
const EURO_MILLIONS_MAIN_COUNT: usize = 5;
const HANGUL_BASE_CODE_POINT: u32 = 0xAC00;
const HANGUL_SYLLABLE_COUNT: usize = 4;
const HANGUL_SYLLABLE_MAX: u32 = 55_859;
const HANGUL_SYLLABLE_MODULUS: u32 = 11_172;
const HANGUL_SHIFTS: [u32; HANGUL_SYLLABLE_COUNT] = [48_u32, 32, 16, 0];
const INPUT_BYTE_MAX_FOR_EURO_MAIN: u8 = 249;
const INPUT_BYTE_MAX_FOR_LOTTO: u8 = 224;
const INPUT_BYTE_MAX_FOR_LOTTO7: u8 = 221;
const INPUT_BYTE_MAX_FOR_LUCKY_STAR: u8 = 251;
const INPUT_BYTE_MAX_FOR_PASSWORD: u8 = 187;
const LOTTO_MODULUS: NonZeroU8 = NonZeroU8::MIN.saturating_add(44);
const LOTTO7_COUNT: usize = 7;
const LOTTO7_MODULUS: NonZeroU8 = NonZeroU8::MIN.saturating_add(36);
const LOTTO_COUNT: usize = 6;
const LOTTO_COUNT_U8: u8 = 6;
const NIBBLE_MASK_U64: u64 = 0xF;
const NMS_COORD_MASK: u64 = 0x0FFF;
const NMS_GLYPH_COUNT: usize = 12;
const NMS_PLANET_MAX_VALUE: u64 = 11;
const NMS_PLANET_FIELD: (u8, u64) = (4, 0x0f);
const NMS_PLANET_MODULUS: u64 = 6;
const NMS_SOLAR_SYSTEM_MAX_VALUE: u64 = 3834;
const NMS_SOLAR_SYSTEM_FIELD: (u8, u64) = (12, 0x0fff);
const NMS_SOLAR_SYSTEM_MODULUS: u64 = 767;
const PASSWORD_BYTE_LEN: usize = 8;
const PASSWORD_BYTE_LEN_U8: u8 = 8;
const SUPPLEMENTAL_RETRY_LIMIT: usize = 1024;
const U32_MAX_INV: f64 = 1.0 / 4_294_967_295.0;
#[derive(Default)]
pub(super) struct Coordinates {
    pub latitude: f64,
    pub longitude: f64,
}
#[derive(Default)]
pub(super) struct RandomDataSet {
    pub euro_millions_lucky_stars: [u8; EURO_MILLIONS_LUCKY_COUNT],
    pub euro_millions_main_numbers: [u8; EURO_MILLIONS_MAIN_COUNT],
    pub galaxy_x: u16,
    pub galaxy_y: u16,
    pub galaxy_z: u16,
    pub glyph_string: [char; NMS_GLYPH_COUNT],
    pub hangul_syllables: [char; HANGUL_SYLLABLE_COUNT],
    pub kor_coords: Coordinates,
    pub lotto7_numbers: [u8; LOTTO7_COUNT],
    pub lotto_numbers: [u8; LOTTO_COUNT],
    pub nms_portal_xxx: u16,
    pub nms_portal_yy: u8,
    pub nms_portal_zzz: u16,
    pub num_64: u64,
    pub numeric_password: u32,
    pub password: [u8; PASSWORD_BYTE_LEN],
    pub planet_number: u8,
    pub solar_system_index: u16,
    pub world_coords: Coordinates,
}
#[derive(Clone, Copy)]
struct RandomBitBuffer {
    bits_remaining: u8,
    value: u64,
}
struct UniqueNumbers<const N: usize> {
    len: usize,
    seen: u64,
    values: [u8; N],
}
struct RandomDataBuildState<'provider_ref, F>
where
    F: FnMut(&'static str) -> Result<u64>,
{
    data: RandomDataSet,
    euro_lucky: UniqueNumbers<EURO_MILLIONS_LUCKY_COUNT>,
    euro_main: UniqueNumbers<EURO_MILLIONS_MAIN_COUNT>,
    lotto: UniqueNumbers<LOTTO_COUNT>,
    lotto7: UniqueNumbers<LOTTO7_COUNT>,
    next_supp: &'provider_ref mut F,
    numeric_password_digits: u8,
    password_len: u8,
    supplemental: RandomBitBuffer,
}
impl RandomDataSet {
    pub(super) fn populate<F>(self, next_supp: &mut F) -> Result<Self>
    where
        F: FnMut(&'static str) -> Result<u64>,
    {
        let mut state = RandomDataBuildState {
            data: self,
            euro_lucky: UniqueNumbers::new(),
            euro_main: UniqueNumbers::new(),
            lotto: UniqueNumbers::new(),
            lotto7: UniqueNumbers::new(),
            next_supp,
            numeric_password_digits: 0,
            password_len: 0,
            supplemental: RandomBitBuffer {
                bits_remaining: 0,
                value: 0,
            },
        };
        state.fill_required_fields()?;
        state.fill_lucky_stars()?;
        state.fill_hangul_syllables()?;
        state.fill_coords();
        state.fill_nms_fields()?;
        state.data.euro_millions_lucky_stars = state.euro_lucky.values;
        state.data.euro_millions_main_numbers = state.euro_main.values;
        state.data.lotto_numbers = state.lotto.values;
        state.data.lotto7_numbers = state.lotto7.values;
        Ok(state.data)
    }
}
impl<const N: usize> UniqueNumbers<N> {
    const fn is_full(&self) -> bool {
        self.len >= N
    }
    const fn new() -> Self {
        Self {
            len: 0,
            seen: 0,
            values: [0; N],
        }
    }
    fn push(&mut self, byte: u8, modulus: NonZeroU8) {
        if self.is_full() {
            return;
        }
        let number = NonZeroU8::MIN
            .saturating_add(byte.rem_euclid(modulus.get()))
            .get();
        let mask = 1_u64 << number;
        if (self.seen & mask) != 0 {
            return;
        }
        let Some(slot) = self.values.get_mut(self.len) else {
            return;
        };
        *slot = number;
        self.seen |= mask;
        self.len = self.len.wrapping_add(1);
        if self.is_full() {
            self.values.sort_unstable();
        }
    }
}
impl<F> RandomDataBuildState<'_, F>
where
    F: FnMut(&'static str) -> Result<u64>,
{
    fn fill_coords(&mut self) {
        let [b0, b1, b2, b3, b4, b5, b6, b7] = self.data.num_64.to_be_bytes();
        let upper_32_bits = u32::from_be_bytes([b0, b1, b2, b3]);
        let lower_32_bits = u32::from_be_bytes([b4, b5, b6, b7]);
        let upper_ratio = NumericMul::mul(f64::from(upper_32_bits), U32_MAX_INV);
        let lower_ratio = NumericMul::mul(f64::from(lower_32_bits), U32_MAX_INV);
        self.data.kor_coords = Coordinates {
            latitude: 5.504_167_f64.mul_add(upper_ratio, 33.112_500),
            longitude: 7.263_056_f64.mul_add(lower_ratio, 124.609_722),
        };
        self.data.world_coords = Coordinates {
            latitude: 180.0_f64.mul_add(upper_ratio, -90.0),
            longitude: 360.0_f64.mul_add(lower_ratio, -180.0),
        };
        self.data.nms_portal_yy = low_u8_from_u32(upper_32_bits);
        self.data.nms_portal_zzz = low_u16_from_u64((self.data.num_64 >> 20) & NMS_COORD_MASK);
        self.data.nms_portal_xxx = low_u16_from_u64((self.data.num_64 >> 8) & NMS_COORD_MASK);
        self.data.galaxy_x = galaxy_coord::<0x801, 0x7FF>(self.data.nms_portal_xxx);
        self.data.galaxy_y = galaxy_coord::<0x81, 0x7F>(u16::from(self.data.nms_portal_yy));
        self.data.galaxy_z = galaxy_coord::<0x801, 0x7FF>(self.data.nms_portal_zzz);
    }
    fn fill_hangul_syllables(&mut self) -> Result<()> {
        let mut hangul = ['\0'; HANGUL_SYLLABLE_COUNT];
        for (slot, shift) in hangul.iter_mut().zip(HANGUL_SHIFTS) {
            let mut syllable_index = u32::from(low_u16_from_u64(self.data.num_64 >> shift));
            for attempts_remaining in (0..SUPPLEMENTAL_RETRY_LIMIT).rev() {
                if syllable_index <= HANGUL_SYLLABLE_MAX {
                    break;
                }
                let supp_value = if self.supplemental.bits_remaining == 0 {
                    self.next_supplemental("한글 음절 보완")?
                } else {
                    self.supplemental.value
                };
                let candidate_value = HANGUL_SHIFTS.into_iter().find_map(|supp_shift| {
                    let candidate = u32::from(low_u16_from_u64(supp_value >> supp_shift));
                    (candidate <= HANGUL_SYLLABLE_MAX).then_some(candidate)
                });
                if let Some(candidate) = candidate_value {
                    syllable_index = candidate;
                } else {
                    if attempts_remaining == 0 {
                        break;
                    }
                    self.next_supplemental("한글 음절 보완 재시도")?;
                }
            }
            if syllable_index > HANGUL_SYLLABLE_MAX {
                return Err("한글 음절 보완 난수 시도 횟수를 초과했습니다.".into());
            }
            let Some(code_point) = HANGUL_BASE_CODE_POINT
                .checked_add(syllable_index.rem_euclid(HANGUL_SYLLABLE_MODULUS))
            else {
                return Err("한글 음절 코드포인트 계산 실패".into());
            };
            *slot = char::from_u32(code_point).ok_or("한글 음절 변환 실패")?;
        }
        self.data.hangul_syllables = hangul;
        Ok(())
    }
    fn fill_lucky_stars(&mut self) -> Result<()> {
        let lucky_star_base = if self.supplemental.bits_remaining == 0 {
            self.data.num_64
        } else {
            self.supplemental.value
        };
        let mut lucky_star_source = lucky_star_base.reverse_bits();
        for attempts_remaining in (0..SUPPLEMENTAL_RETRY_LIMIT).rev() {
            for byte in lucky_star_source.to_be_bytes() {
                if byte > INPUT_BYTE_MAX_FOR_LUCKY_STAR {
                    continue;
                }
                self.euro_lucky.push(byte, EURO_LUCKY_MODULUS);
                if self.euro_lucky.is_full() {
                    return Ok(());
                }
            }
            if attempts_remaining == 0 {
                break;
            }
            lucky_star_source = self
                .next_supplemental("유로밀리언 럭키 스타 보완")?
                .reverse_bits();
        }
        Err("유로밀리언 럭키 스타 보완 난수 시도 횟수를 초과했습니다.".into())
    }
    fn fill_nms_fields(&mut self) -> Result<()> {
        let num = self.data.num_64;
        let planet_number_base = extract_valid_bits_for_nms(
            num,
            &[52, 4, 0],
            NMS_PLANET_FIELD,
            NMS_PLANET_MAX_VALUE,
            "NMS 행성 번호 보완",
            &mut self.supplemental,
            self.next_supp,
        )?
        .rem_euclid(NMS_PLANET_MODULUS);
        let planet_number = planet_number_base.saturating_add(1);
        self.data.planet_number = low_u8_from_u64(planet_number);
        let solar_system_index_base = extract_valid_bits_for_nms(
            num,
            &[40],
            NMS_SOLAR_SYSTEM_FIELD,
            NMS_SOLAR_SYSTEM_MAX_VALUE,
            "NMS 태양계 번호 보완",
            &mut self.supplemental,
            self.next_supp,
        )?
        .rem_euclid(NMS_SOLAR_SYSTEM_MODULUS);
        let solar_system_index = solar_system_index_base.saturating_add(1);
        self.data.solar_system_index = low_u16_from_u64(solar_system_index);
        let glyph_sources = [
            u64::from(self.data.planet_number),
            u64::from(self.data.solar_system_index >> 8_u32),
            u64::from(self.data.solar_system_index >> 4_u32),
            u64::from(self.data.solar_system_index),
            num >> 36_u32,
            num >> 32_u32,
            num >> 28_u32,
            num >> 24_u32,
            num >> 20_u32,
            num >> 16_u32,
            num >> 12_u32,
            num >> 8_u32,
        ];
        for (slot, nibble_source) in self.data.glyph_string.iter_mut().zip(glyph_sources) {
            *slot = match low_u8_from_u64(nibble_source & NIBBLE_MASK_U64) {
                0x0 => '🌅',
                0x1 => '🐦',
                0x2 => '👫',
                0x3 => '🦕',
                0x4 => '🌘',
                0x5 => '🎈',
                0x6 => '⛵',
                0x7 => '🕷',
                0x8 => '🦋',
                0x9 => '🌀',
                0xA => '🧊',
                0xB => '🐟',
                0xC => '⛺',
                0xD => '🚀',
                0xE => '🌳',
                _ => '🔯',
            };
        }
        Ok(())
    }
    fn fill_required_fields(&mut self) -> Result<()> {
        self.fill_required_fields_from_u64(self.data.num_64);
        for _ in 0..SUPPLEMENTAL_RETRY_LIMIT {
            if self.is_complete() {
                return Ok(());
            }
            let new_supp = self.next_supplemental("기본 필드 보완")?;
            self.fill_required_fields_from_u64(new_supp);
        }
        Err("기본 필드 보완 난수 시도 횟수를 초과했습니다.".into())
    }
    fn fill_required_fields_from_u64(&mut self, value: u64) {
        for byte in value.to_be_bytes() {
            if byte > INPUT_BYTE_MAX_FOR_EURO_MAIN {
                continue;
            }
            if self.numeric_password_digits < LOTTO_COUNT_U8 {
                let digit = u32::from(byte.rem_euclid(10));
                self.data.numeric_password = self
                    .data
                    .numeric_password
                    .saturating_mul(10)
                    .saturating_add(digit);
                self.numeric_password_digits = self.numeric_password_digits.saturating_add(1);
            }
            if !self.euro_main.is_full() {
                self.euro_main.push(byte, EURO_MAIN_MODULUS);
            }
            if byte <= INPUT_BYTE_MAX_FOR_LOTTO {
                if !self.lotto.is_full() {
                    self.lotto.push(byte, LOTTO_MODULUS);
                }
                if byte <= INPUT_BYTE_MAX_FOR_LOTTO7 {
                    if !self.lotto7.is_full() {
                        self.lotto7.push(byte, LOTTO7_MODULUS);
                    }
                    if byte <= INPUT_BYTE_MAX_FOR_PASSWORD
                        && self.password_len < PASSWORD_BYTE_LEN_U8
                        && let Some(slot) =
                            self.data.password.get_mut(usize::from(self.password_len))
                    {
                        *slot = byte
                            .rem_euclid(ASCII_PRINTABLE_LEN)
                            .saturating_add(ASCII_PRINTABLE_START);
                        self.password_len = self.password_len.saturating_add(1);
                    }
                }
            }
            if self.is_complete() {
                return;
            }
        }
    }
    const fn is_complete(&self) -> bool {
        self.numeric_password_digits >= LOTTO_COUNT_U8
            && self.lotto.is_full()
            && self.lotto7.is_full()
            && self.password_len >= PASSWORD_BYTE_LEN_U8
            && self.euro_main.is_full()
    }
    fn next_supplemental(&mut self, reason: &'static str) -> Result<u64> {
        let value = (self.next_supp)(reason)?;
        self.supplemental = RandomBitBuffer {
            bits_remaining: BYTE_BITS,
            value,
        };
        Ok(value)
    }
}
const fn galaxy_coord<const SUB: u16, const ADD: u16>(value: u16) -> u16 {
    if value >= SUB {
        return value.saturating_sub(SUB);
    }
    value.saturating_add(ADD)
}
cfg_select! {
    target_arch = "x86_64" => {
pub(super) fn generate_random_data_with_rng(rng: &HardwareRng) -> Result<RandomDataSet> {
            let num = rng.next_u64()?;
            let mut next_supp = |reason: &'static str| -> Result<u64> {
                rng.next_u64().map_err(|source| AppError::context(reason, source))
            };
            RandomDataSet {
                num_64: num,
                ..Default::default()
            }
            .populate(&mut next_supp)
        }
    }
    _ => {}
}
fn extract_valid_bits_for_nms(
    num: u64,
    shifts: &[u8],
    bit_field: (u8, u64),
    max_value: u64,
    reason: &'static str,
    supplemental: &mut RandomBitBuffer,
    next_supp: &mut impl FnMut(&'static str) -> Result<u64>,
) -> Result<u64> {
    let (bits, mask) = bit_field;
    if let Some(extracted_value) = shifts.iter().find_map(|&shift| {
        let value = (num >> shift) & mask;
        (value <= max_value).then_some(value)
    }) {
        return Ok(extracted_value);
    }
    for _ in 0..SUPPLEMENTAL_RETRY_LIMIT {
        if supplemental.bits_remaining < bits {
            *supplemental = RandomBitBuffer {
                bits_remaining: BYTE_BITS,
                value: next_supp(reason)?,
            };
        }
        let shift = supplemental.bits_remaining.abs_diff(bits);
        supplemental.bits_remaining = shift;
        let extracted = (supplemental.value >> shift) & mask;
        if extracted <= max_value {
            return Ok(extracted);
        }
    }
    Err("NMS 보완 난수 시도 횟수를 초과했습니다.".into())
}
