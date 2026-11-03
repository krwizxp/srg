#[cfg(target_arch = "x86_64")]
use super::{diagnostic::AppError, hardware_rng::HardwareRng};
use crate::{
    diagnostic::Result,
    korea_boundary,
    numeric::{low_u8_from_u32, low_u8_from_u64, low_u16_from_u64, low_u32_from_u64},
};
use std::process;
const ASCII_PRINTABLE_LEN: u64 = 94;
const ASCII_PRINTABLE_START: u8 = 33;
const BINOMIAL_VALUES: [[u32; 8]; 51] = {
    let mut values = [[0; 8]; 51];
    let mut rows: &mut [[u32; 8]] = &mut values;
    let mut total = 0_u32;
    while let Some((row, remaining_rows)) = rows.split_first_mut() {
        let mut cells: &mut [u32] = row;
        let mut choose = 0_u32;
        while let Some((cell, remaining_cells)) = cells.split_first_mut() {
            *cell = binomial(total, choose);
            choose = choose.strict_add(1);
            cells = remaining_cells;
        }
        total = total.strict_add(1);
        rows = remaining_rows;
    }
    values
};
const EURO_LUCKY_COMB: u64 = binomial(EURO_LUCKY_MAX, 2) as u64;
const EURO_LUCKY_MAX: u32 = 12;
const EURO_MAIN_COMB: u64 = binomial(EURO_MAIN_MAX, 5) as u64;
const EURO_MAIN_MAX: u32 = 50;
const EURO_MILLIONS_LUCKY_COUNT: usize = 2;
const EURO_MILLIONS_MAIN_COUNT: usize = 5;
const EURO_PIN_NMS_LIMIT: u64 = u64::MAX
    .div_euclid(EURO_PIN_NMS_MODULUS)
    .strict_mul(EURO_PIN_NMS_MODULUS);
const EURO_PIN_NMS_MODULUS: u64 = EURO_MAIN_COMB
    .strict_mul(NUMERIC_PASSWORD_MODULUS)
    .strict_mul(NMS_SOLAR_SYSTEM_MODULUS);
const HANGUL_BASE_CODE_POINT: u32 = 0xAC00;
const HANGUL_QUAD_LIMIT: u64 = u64::MAX
    .div_euclid(HANGUL_QUAD_MODULUS)
    .strict_mul(HANGUL_QUAD_MODULUS);
const HANGUL_QUAD_MODULUS: u64 = HANGUL_SYLLABLE_MODULUS.strict_pow(4);
const HANGUL_SYLLABLE_COUNT: usize = 4;
const HANGUL_SYLLABLE_MODULUS: u64 = 11_172;
const LOTTO_BUNDLE_LIMIT: u64 = u64::MAX
    .div_euclid(LOTTO_BUNDLE_MODULUS)
    .strict_mul(LOTTO_BUNDLE_MODULUS);
const LOTTO_BUNDLE_MODULUS: u64 = LOTTO_COMB
    .strict_mul(LOTTO7_COMB)
    .strict_mul(EURO_LUCKY_COMB)
    .strict_mul(NMS_PLANET_MODULUS);
const LOTTO_COMB: u64 = binomial(LOTTO_MAX, 6) as u64;
const LOTTO_COUNT: usize = 6;
const LOTTO_MAX: u32 = 45;
const LOTTO7_COMB: u64 = binomial(LOTTO7_MAX, 7) as u64;
const LOTTO7_COUNT: usize = 7;
const LOTTO7_MAX: u32 = 37;
const MIX_MUL_A: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_MUL_B: u64 = 0x94D0_49BB_1331_11EB;
const MIX_SALT_EURO: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_SALT_HANGUL: u64 = 0x3C6E_F372_FE94_F82A;
const MIX_SALT_LOTTO: u64 = 0xD1B5_4A32_D192_ED03;
const MIX_SALT_PASSWORD: u64 = 0xA5A5_A5A5_5A5A_5A5A;
const NMS_COORD_MASK: u64 = 0x0FFF;
const NMS_PLANET_MODULUS: u64 = 6;
const NMS_SOLAR_SYSTEM_MODULUS: u64 = 767;
const NUMERIC_PASSWORD_MODULUS: u64 = 1_000_000;
const PASSWORD_BYTE_LEN: usize = 8;
const PASSWORD_OCTET_LIMIT: u64 = u64::MAX
    .div_euclid(PASSWORD_OCTET_MODULUS)
    .strict_mul(PASSWORD_OCTET_MODULUS);
const PASSWORD_OCTET_MODULUS: u64 = ASCII_PRINTABLE_LEN.strict_pow(8);
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
struct RandomDataBuildState<'cache> {
    data: RandomDataSet,
    korea: korea_boundary::SampleState<'cache>,
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
            korea: korea_boundary::SampleState::try_from((
                initial,
                boundary_cache.unwrap_or(&mut local_cache),
            ))?,
            data: self,
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
impl RandomDataBuildState<'_> {
    fn check_complete(&self) -> Result<()> {
        if !self.is_basic_complete() {
            return Err("기본 필드 보완 난수 시도 횟수를 초과했습니다.".into());
        }
        if self.data.hangul_syllables.contains(&'\0') {
            return Err("한글 음절 보완 난수 시도 횟수를 초과했습니다.".into());
        }
        Ok(())
    }
    fn fill_coords(&mut self) {
        let [b0, b1, b2, b3, b4, b5, b6, b7] = self.data.num_64.to_be_bytes();
        let upper_32_bits = u32::from_be_bytes([b0, b1, b2, b3]);
        let lower_32_bits = u32::from_be_bytes([b4, b5, b6, b7]);
        let upper_ratio = f64::from(upper_32_bits).algebraic_mul(U32_MAX_INV);
        let lower_ratio = f64::from(lower_32_bits).algebraic_mul(U32_MAX_INV);
        self.data.world_coords = Coordinates {
            latitude: 180.0_f64.mul_add(upper_ratio, -90.0),
            longitude: 360.0_f64.mul_add(lower_ratio, -180.0),
        };
        self.data.nms_portal_yy = low_u8_from_u32(upper_32_bits);
        self.data.nms_portal_zzz = low_u16_from_u64((self.data.num_64 >> 20) & NMS_COORD_MASK);
        self.data.nms_portal_xxx = low_u16_from_u64((self.data.num_64 >> 8) & NMS_COORD_MASK);
        self.data.galaxy_x = galaxy_coord::<0x7FF, 0xFFF>(self.data.nms_portal_xxx);
        self.data.galaxy_y = galaxy_coord::<0x7F, 0xFF>(u16::from(self.data.nms_portal_yy));
        self.data.galaxy_z = galaxy_coord::<0x7FF, 0xFFF>(self.data.nms_portal_zzz);
    }
    fn fill_euro_pin_nms_from_u64(&mut self, value: u64) {
        if self.data.solar_system_index != 0 {
            return;
        }
        let mixed = mix_word(value, MIX_SALT_EURO);
        if mixed >= EURO_PIN_NMS_LIMIT {
            return;
        }
        let rem = mixed.rem_euclid(EURO_PIN_NMS_MODULUS);
        let solar_rem = rem.rem_euclid(NMS_SOLAR_SYSTEM_MODULUS);
        let rem_pin_euro = rem.div_euclid(NMS_SOLAR_SYSTEM_MODULUS);
        let pin_rem = rem_pin_euro.rem_euclid(NUMERIC_PASSWORD_MODULUS);
        let euro_rank = low_u32_from_u64(rem_pin_euro.div_euclid(NUMERIC_PASSWORD_MODULUS));
        self.data.solar_system_index = low_u16_from_u64(solar_rem.strict_add(1));
        self.data.numeric_password = low_u32_from_u64(pin_rem);
        self.data.euro_millions_main_numbers =
            unrank_combination::<EURO_MILLIONS_MAIN_COUNT>(euro_rank, EURO_MAIN_MAX);
    }
    fn fill_hangul_from_u64(&mut self, value: u64) {
        if !self.data.hangul_syllables.contains(&'\0') {
            return;
        }
        let mixed = mix_word(value, MIX_SALT_HANGUL);
        if mixed >= HANGUL_QUAD_LIMIT {
            return;
        }
        let mut rem = mixed.rem_euclid(HANGUL_QUAD_MODULUS);
        for slot in &mut self.data.hangul_syllables {
            let offset = low_u32_from_u64(rem.rem_euclid(HANGUL_SYLLABLE_MODULUS));
            rem = rem.div_euclid(HANGUL_SYLLABLE_MODULUS);
            let point = HANGUL_BASE_CODE_POINT.strict_add(offset);
            *slot = char::from_u32(point).unwrap_or_else(|| process::abort());
        }
    }
    fn fill_lotto_bundle_from_u64(&mut self, value: u64) {
        if self.data.planet_number != 0 {
            return;
        }
        let mixed = mix_word(value, MIX_SALT_LOTTO);
        if mixed >= LOTTO_BUNDLE_LIMIT {
            return;
        }
        let rem = mixed.rem_euclid(LOTTO_BUNDLE_MODULUS);
        let planet_rem = rem.rem_euclid(NMS_PLANET_MODULUS);
        let rem_after_planet = rem.div_euclid(NMS_PLANET_MODULUS);
        let lucky_rank = low_u32_from_u64(rem_after_planet.rem_euclid(EURO_LUCKY_COMB));
        let rem_lottos = rem_after_planet.div_euclid(EURO_LUCKY_COMB);
        let korea_lotto_rank = low_u32_from_u64(rem_lottos.rem_euclid(LOTTO_COMB));
        let japan_lotto_rank = low_u32_from_u64(rem_lottos.div_euclid(LOTTO_COMB));
        self.data.planet_number = low_u8_from_u64(planet_rem.strict_add(1));
        self.data.euro_millions_lucky_stars =
            unrank_combination::<EURO_MILLIONS_LUCKY_COUNT>(lucky_rank, EURO_LUCKY_MAX);
        self.data.lotto_numbers = unrank_combination::<LOTTO_COUNT>(korea_lotto_rank, LOTTO_MAX);
        self.data.lotto7_numbers = unrank_combination::<LOTTO7_COUNT>(japan_lotto_rank, LOTTO7_MAX);
    }
    fn fill_password_from_u64(&mut self, value: u64) {
        if !self.data.password.contains(&0) {
            return;
        }
        let mixed = mix_word(value, MIX_SALT_PASSWORD);
        if mixed >= PASSWORD_OCTET_LIMIT {
            return;
        }
        let mut rem = mixed.rem_euclid(PASSWORD_OCTET_MODULUS);
        for slot in &mut self.data.password {
            let offset = low_u8_from_u64(rem.rem_euclid(ASCII_PRINTABLE_LEN));
            rem = rem.div_euclid(ASCII_PRINTABLE_LEN);
            *slot = ASCII_PRINTABLE_START.strict_add(offset);
        }
    }
    fn finish(self) -> Result<RandomDataSet> {
        self.check_complete()?;
        let mut data = self.data;
        data.kor_coords = self.korea.finish()?;
        Ok(data)
    }
    fn is_basic_complete(&self) -> bool {
        !self.data.password.contains(&0)
            && self.data.planet_number != 0
            && self.data.solar_system_index != 0
    }
    fn is_complete(&self) -> bool {
        self.is_basic_complete()
            && !self.data.hangul_syllables.contains(&'\0')
            && self.korea.is_complete()
    }
    fn offer_word(&mut self, value: u64, initial: bool) -> Result<()> {
        self.fill_lotto_bundle_from_u64(value);
        self.fill_euro_pin_nms_from_u64(value);
        self.fill_hangul_from_u64(value);
        self.fill_password_from_u64(value);
        if !initial {
            self.korea.offer_supplement(value)?;
        }
        Ok(())
    }
}
const fn binomial(total: u32, choose: u32) -> u32 {
    if choose > total {
        return 0;
    }
    let mut result = 1_u32;
    let mut step = 1_u32;
    while step <= choose {
        let factor = total.strict_sub(choose).strict_add(step);
        result = result.strict_mul(factor).div_euclid(step);
        step = step.strict_add(1);
    }
    result
}
fn combination_count(total: u32, choose: usize) -> u32 {
    BINOMIAL_VALUES
        .get(usize::try_from(total).unwrap_or_else(|_| process::abort()))
        .and_then(|row| row.get(choose))
        .copied()
        .unwrap_or_else(|| process::abort())
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
const fn mix_word(value: u64, salt: u64) -> u64 {
    let mut state = value.wrapping_add(salt);
    state = (state ^ (state >> 30_u32)).wrapping_mul(MIX_MUL_A);
    state = (state ^ (state >> 27_u32)).wrapping_mul(MIX_MUL_B);
    state ^ (state >> 31_u32)
}
fn unrank_combination<const K: usize>(mut rank: u32, max_n: u32) -> [u8; K] {
    let mut out = [0_u8; K];
    let mut upper = max_n;
    for slot_num in (1..=K).rev() {
        let choose = u32::try_from(slot_num).unwrap_or_else(|_| process::abort());
        let mut low = choose.strict_sub(1);
        let mut high = upper;
        while low.strict_add(1) < high {
            let mid = low.strict_add(high.strict_sub(low).div_euclid(2));
            if combination_count(mid, slot_num) <= rank {
                low = mid;
            } else {
                high = mid;
            }
        }
        rank = rank.strict_sub(combination_count(low, slot_num));
        let slot = out
            .get_mut(slot_num.strict_sub(1))
            .unwrap_or_else(|| process::abort());
        *slot = low_u8_from_u32(low.strict_add(1));
        upper = low;
    }
    out
}
