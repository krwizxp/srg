#[cfg(target_arch = "x86_64")]
use super::hardware_rng::HardwareRng;
use crate::{
    coordinate_core::{Coordinates, world_coordinates},
    diagnostic::{AppError, Result},
    korea_boundary,
    numeric::{low_u8_from_u32, low_u8_from_u64, low_u16_from_u64},
};
use core::array;
use std::process;
const ASCII_PRINTABLE_LEN: u8 = 94;
const ASCII_PRINTABLE_START: u8 = 33;
const EURO_LUCKY_MODULUS: u8 = 12;
const EURO_MAIN_MODULUS: u8 = 50;
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
const LOTTO_MODULUS: u8 = 45;
const LOTTO7_COUNT: usize = 7;
const LOTTO7_MODULUS: u8 = 37;
const LOTTO_COUNT: usize = 6;
const LOTTO_COUNT_U8: u8 = 6;
const NMS_COORD_MASK: u64 = 0x0FFF;
const NMS_PLANET_MAX_VALUE: u64 = 11;
const NMS_PLANET_MASK: u64 = 0x0f;
const NMS_PLANET_SHIFTS: [u8; 3] = [52, 4, 0];
const NMS_PLANET_SUPP_SHIFTS: [u8; 4] = [60, 56, 52, 48];
const NMS_PLANET_MODULUS: u64 = 6;
const NMS_SOLAR_SYSTEM_MAX_VALUE: u64 = 3834;
const NMS_SOLAR_SYSTEM_MASK: u64 = 0x0fff;
const NMS_SOLAR_SYSTEM_SHIFTS: [u8; 1] = [40];
const NMS_SOLAR_SYSTEM_SUPP_SHIFTS: [u8; 4] = [36, 24, 12, 0];
const NMS_SOLAR_SYSTEM_MODULUS: u64 = 767;
const PASSWORD_BYTE_LEN: usize = 8;
const SUPPLEMENTAL_RETRY_LIMIT: usize = 1024;
#[derive(Default)]
pub(super) struct RandomDataSet {
    pub euro_millions_lucky_stars: [u8; EURO_MILLIONS_LUCKY_COUNT],
    pub euro_millions_main_numbers: [u8; EURO_MILLIONS_MAIN_COUNT],
    pub galaxy_x: u16,
    pub galaxy_y: u16,
    pub galaxy_z: u16,
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
struct UniqueNumbers<const N: usize> {
    count: u8,
    seen: u64,
}
struct RandomDataBuildState<'cache> {
    korea: korea_boundary::SampleState<'cache>,
    data: RandomDataSet,
    euro_lucky: UniqueNumbers<EURO_MILLIONS_LUCKY_COUNT>,
    euro_main: UniqueNumbers<EURO_MILLIONS_MAIN_COUNT>,
    lotto: UniqueNumbers<LOTTO_COUNT>,
    lotto7: UniqueNumbers<LOTTO7_COUNT>,
    numeric_password_digits: u8,
    password_len: u8,
}
impl RandomDataSet {
    pub(super) fn populate<F>(
        self,
        next_supp: &mut F,
        boundary_cache: Option<&mut korea_boundary::BoundaryCache>,
    ) -> Result<Self>
    where
        F: FnMut(&'static str) -> Result<u64>,
    {
        let initial = self.num_64;
        let mut local_cache = korea_boundary::BoundaryCache::default();
        let mut fields = RandomDataBuildState {
            korea: korea_boundary::SampleState::new(
                initial,
                boundary_cache.unwrap_or(&mut local_cache),
            )?,
            data: self,
            euro_lucky: UniqueNumbers::new(),
            euro_main: UniqueNumbers::new(),
            lotto: UniqueNumbers::new(),
            lotto7: UniqueNumbers::new(),
            numeric_password_digits: 0,
            password_len: 0,
        };
        fields.fill_coords();
        fields.offer_word(initial, true)?;
        for _ in 0..SUPPLEMENTAL_RETRY_LIMIT {
            if fields.is_complete() {
                break;
            }
            fields.offer_word(next_supp("미완성 필드 보완")?, false)?;
        }
        fields.finish()
    }
}
impl<const N: usize> UniqueNumbers<N> {
    fn into_values(self) -> [u8; N] {
        let mut seen = self.seen;
        array::from_fn(|_| {
            let number = low_u8_from_u32(seen.trailing_zeros());
            seen ^= seen.isolate_lowest_one();
            number
        })
    }
    fn is_full(&self) -> bool {
        usize::from(self.count) == N
    }
    const fn new() -> Self {
        Self { count: 0, seen: 0 }
    }
    fn push(&mut self, byte: u8, modulus: u8) -> bool {
        if self.is_full() {
            return true;
        }
        let mask = 1_u64 << byte.rem_euclid(modulus).strict_add(1);
        if (self.seen & mask) != 0 {
            return false;
        }
        self.seen |= mask;
        self.count = self.count.strict_add(1);
        self.is_full()
    }
}
impl RandomDataBuildState<'_> {
    fn is_basic_complete(&self) -> bool {
        usize::from(self.password_len) == PASSWORD_BYTE_LEN
            && self.lotto7.is_full()
            && self.lotto.is_full()
            && self.euro_main.is_full()
    }
    fn is_complete(&self) -> bool {
        self.is_basic_complete()
            && self.korea.is_complete()
            && self.data.hangul_syllables.iter().all(|&c| c != '\0')
            && self.data.planet_number != 0
            && self.data.solar_system_index != 0
            && self.euro_lucky.is_full()
    }
    fn check_complete(&self) -> Result<()> {
        if !self.is_basic_complete() {
            return Err("기본 필드 보완 난수 시도 횟수를 초과했습니다.".into());
        }
        if !self.euro_lucky.is_full() {
            return Err("유로밀리언 럭키 스타 보완 난수 시도 횟수를 초과했습니다.".into());
        }
        if self.data.hangul_syllables.contains(&'\0') {
            return Err("한글 음절 보완 난수 시도 횟수를 초과했습니다.".into());
        }
        if self.data.planet_number == 0 || self.data.solar_system_index == 0 {
            return Err("NMS 보완 난수 시도 횟수를 초과했습니다.".into());
        }
        Ok(())
    }
    fn offer_word(&mut self, value: u64, initial: bool) -> Result<()> {
        self.fill_required_fields_from_u64(value);
        self.fill_hangul_from_u64(value, initial);
        self.fill_nms_from_u64(value, initial);
        self.fill_lucky_stars_from_u64(value);
        if !initial {
            self.korea.offer_supplement(value)?;
        }
        Ok(())
    }
    fn finish(self) -> Result<RandomDataSet> {
        self.check_complete()?;
        let mut data = self.data;
        data.kor_coords = self.korea.finish()?;
        data.euro_millions_lucky_stars = self.euro_lucky.into_values();
        data.euro_millions_main_numbers = self.euro_main.into_values();
        data.lotto_numbers = self.lotto.into_values();
        data.lotto7_numbers = self.lotto7.into_values();
        Ok(data)
    }
    fn fill_coords(&mut self) {
        self.data.world_coords = world_coordinates(self.data.num_64);
        self.data.nms_portal_yy = low_u8_from_u64(self.data.num_64 >> 32);
        self.data.nms_portal_zzz = low_u16_from_u64((self.data.num_64 >> 20) & NMS_COORD_MASK);
        self.data.nms_portal_xxx = low_u16_from_u64((self.data.num_64 >> 8) & NMS_COORD_MASK);
        self.data.galaxy_x = galaxy_coord::<0x7FF, 0xFFF>(self.data.nms_portal_xxx);
        self.data.galaxy_y = galaxy_coord::<0x7F, 0xFF>(u16::from(self.data.nms_portal_yy));
        self.data.galaxy_z = galaxy_coord::<0x7FF, 0xFFF>(self.data.nms_portal_zzz);
    }
    fn fill_lucky_stars_from_u64(&mut self, value: u64) {
        if self.euro_lucky.is_full() {
            return;
        }
        for byte in value.reverse_bits().to_be_bytes() {
            if byte <= INPUT_BYTE_MAX_FOR_LUCKY_STAR
                && self.euro_lucky.push(byte, EURO_LUCKY_MODULUS)
            {
                return;
            }
        }
    }
    fn fill_hangul_from_u64(&mut self, value: u64, initial: bool) {
        for (position, slot) in self.data.hangul_syllables.iter_mut().enumerate() {
            if *slot != '\0' {
                continue;
            }
            let shifts = HANGUL_SHIFTS.into_iter().cycle().skip(position);
            let limit = if initial { 1 } else { HANGUL_SYLLABLE_COUNT };
            let index = shifts.take(limit).find_map(|shift| {
                let candidate = u32::from(low_u16_from_u64(value >> shift));
                (candidate <= HANGUL_SYLLABLE_MAX).then_some(candidate)
            });
            if let Some(index) = index {
                let point =
                    HANGUL_BASE_CODE_POINT.strict_add(index.rem_euclid(HANGUL_SYLLABLE_MODULUS));
                *slot = char::from_u32(point).unwrap_or_else(|| process::abort());
            }
        }
    }
    fn fill_nms_from_u64(&mut self, value: u64, initial: bool) {
        if self.data.planet_number == 0 {
            let shifts = if initial {
                &NMS_PLANET_SHIFTS[..]
            } else {
                &NMS_PLANET_SUPP_SHIFTS[..]
            };
            if let Some(candidate) =
                nms_candidate(value, shifts, NMS_PLANET_MASK, NMS_PLANET_MAX_VALUE)
            {
                self.data.planet_number =
                    low_u8_from_u64(candidate.rem_euclid(NMS_PLANET_MODULUS).strict_add(1));
            }
        }
        if self.data.solar_system_index == 0 {
            let shifts = if initial {
                &NMS_SOLAR_SYSTEM_SHIFTS[..]
            } else {
                &NMS_SOLAR_SYSTEM_SUPP_SHIFTS[..]
            };
            if let Some(candidate) = nms_candidate(
                value,
                shifts,
                NMS_SOLAR_SYSTEM_MASK,
                NMS_SOLAR_SYSTEM_MAX_VALUE,
            ) {
                self.data.solar_system_index =
                    low_u16_from_u64(candidate.rem_euclid(NMS_SOLAR_SYSTEM_MODULUS).strict_add(1));
            }
        }
    }
    fn fill_required_fields_from_u64(&mut self, value: u64) {
        if self.is_basic_complete() {
            return;
        }
        for byte in value.to_be_bytes() {
            if byte > INPUT_BYTE_MAX_FOR_EURO_MAIN {
                continue;
            }
            if self.numeric_password_digits < LOTTO_COUNT_U8 {
                let digit = u32::from(byte.rem_euclid(10));
                self.data.numeric_password =
                    self.data.numeric_password.strict_mul(10).strict_add(digit);
                self.numeric_password_digits = self.numeric_password_digits.strict_add(1);
            }
            self.euro_main.push(byte, EURO_MAIN_MODULUS);
            if byte <= INPUT_BYTE_MAX_FOR_LOTTO {
                self.lotto.push(byte, LOTTO_MODULUS);
                if byte <= INPUT_BYTE_MAX_FOR_LOTTO7 {
                    self.lotto7.push(byte, LOTTO7_MODULUS);
                    if byte <= INPUT_BYTE_MAX_FOR_PASSWORD
                        && let Some(slot) =
                            self.data.password.get_mut(usize::from(self.password_len))
                    {
                        *slot = byte
                            .rem_euclid(ASCII_PRINTABLE_LEN)
                            .strict_add(ASCII_PRINTABLE_START);
                        self.password_len = self.password_len.strict_add(1);
                    }
                }
            }
            if self.is_basic_complete() {
                return;
            }
        }
    }
}
fn nms_candidate(value: u64, shifts: &[u8], mask: u64, max: u64) -> Option<u64> {
    shifts.iter().find_map(|&shift| {
        let candidate = (value >> shift) & mask;
        (candidate <= max).then_some(candidate)
    })
}
const fn galaxy_coord<const OFFSET: u16, const MASK: u16>(value: u16) -> u16 {
    // Callers supply 8-bit or 12-bit fields; shift the origin within that field.
    value.strict_add(OFFSET) & MASK
}
#[cfg(target_arch = "x86_64")]
pub(super) fn generate_random_data_with_rng(
    rng: &HardwareRng,
    boundary_cache: Option<&mut korea_boundary::BoundaryCache>,
) -> Result<RandomDataSet> {
    let mut next_supp = |reason: &'static str| -> Result<u64> {
        rng.next_u64()
            .map_err(|source| AppError::context(reason, source))
    };
    RandomDataSet {
        num_64: rng.next_u64()?,
        ..Default::default()
    }
    .populate(&mut next_supp, boundary_cache)
}
