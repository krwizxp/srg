#[cfg(target_arch = "x86_64")]
use super::{diagnostic::AppError, hardware_rng::HardwareRng};
use crate::{
    diagnostic::Result,
    korea_boundary,
    numeric::{
        low_u8_from_u32, low_u8_from_u64, low_u16_from_u64, low_u32_from_u64, split_u64_be_u32,
    },
};
use std::process;
const ASCII_PRINTABLE_LEN: u64 = 94;
const ASCII_PRINTABLE_START: u8 = 33;
const BINOMIAL_VALUES: [[u32; 50]; 6] = {
    let mut values = [[0; 50]; 6];
    let mut rows: &mut [[u32; 50]] = &mut values;
    let mut choose = 2_u32;
    while let Some((row, remaining_rows)) = rows.split_first_mut() {
        let mut cells: &mut [u32] = row;
        let mut total = 0_u32;
        while let Some((cell, remaining_cells)) = cells.split_first_mut() {
            *cell = binomial(total, choose);
            total = total.strict_add(1);
            cells = remaining_cells;
        }
        choose = choose.strict_add(1);
        rows = remaining_rows;
    }
    values
};
const EURO_LUCKY_COMB: u64 = binomial(EURO_LUCKY_MAX as u32, 2) as u64;
const EURO_LUCKY_MAX: u8 = 12;
const EURO_MAIN_COMB: u64 = binomial(EURO_MAIN_MAX as u32, 5) as u64;
const EURO_MAIN_MAX: u8 = 50;
const EURO_MILLIONS_LUCKY_COUNT: usize = 2;
const EURO_MILLIONS_MAIN_COUNT: usize = 5;
const EURO_PIN_NMS_MODULUS: u64 = EURO_MAIN_COMB
    .strict_mul(NUMERIC_PASSWORD_MODULUS)
    .strict_mul(NMS_SOLAR_SYSTEM_MODULUS);
const HANGUL_BASE_CODE_POINT: u16 = 0xAC00;
const HANGUL_QUAD_MODULUS: u64 = HANGUL_SYLLABLE_MODULUS.strict_pow(4);
const HANGUL_SYLLABLE_COUNT: usize = 4;
const HANGUL_SYLLABLE_MODULUS: u64 = 11_172;
const LOTTO_BUNDLE_MODULUS: u64 = LOTTO_COMB
    .strict_mul(LOTTO7_COMB)
    .strict_mul(EURO_LUCKY_COMB)
    .strict_mul(NMS_PLANET_MODULUS);
const LOTTO_COMB: u64 = binomial(LOTTO_MAX as u32, 6) as u64;
const LOTTO_COUNT: usize = 6;
const LOTTO_MAX: u8 = 45;
const LOTTO7_COMB: u64 = binomial(LOTTO7_MAX as u32, 7) as u64;
const LOTTO7_COUNT: usize = 7;
const LOTTO7_MAX: u8 = 37;
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
    pub hangul_syllables: [u16; HANGUL_SYLLABLE_COUNT],
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
        boundary_cache: &mut korea_boundary::BoundaryCache,
    ) -> Result<Self>
    where
        F: FnMut(&'static str) -> Result<u64>,
    {
        let initial = self.num_64;
        let mut fields = RandomDataBuildState {
            korea: korea_boundary::SampleState::try_from((initial, boundary_cache))?,
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
        self.is_basic_complete()
            .ok_or("기본 필드 보완 난수 시도 횟수를 초과했습니다.")?;
        (!matches!(self.data.hangul_syllables, [0, ..]))
            .ok_or("한글 음절 보완 난수 시도 횟수를 초과했습니다.")
            .map_err(Into::into)
    }
    fn fill_coords(&mut self) {
        let (upper_32_bits, lower_32_bits) = split_u64_be_u32(self.data.num_64);
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
        let Some(rem) = mix_bounded::<EURO_PIN_NMS_MODULUS>(value, MIX_SALT_EURO) else {
            return;
        };
        let solar_rem = rem.rem_euclid(NMS_SOLAR_SYSTEM_MODULUS);
        let rem_pin_euro = rem.div_euclid(NMS_SOLAR_SYSTEM_MODULUS);
        let pin_rem = rem_pin_euro.rem_euclid(NUMERIC_PASSWORD_MODULUS);
        let euro_rank = low_u32_from_u64(rem_pin_euro.div_euclid(NUMERIC_PASSWORD_MODULUS));
        self.data.solar_system_index = low_u16_from_u64(solar_rem.strict_add(1));
        self.data.numeric_password = low_u32_from_u64(pin_rem);
        unrank_combination_into(
            euro_rank,
            EURO_MAIN_MAX,
            &mut self.data.euro_millions_main_numbers,
        );
    }
    fn fill_hangul_from_u64(&mut self, value: u64) {
        if !matches!(self.data.hangul_syllables, [0, ..]) {
            return;
        }
        let Some(rem) = mix_bounded::<HANGUL_QUAD_MODULUS>(value, MIX_SALT_HANGUL) else {
            return;
        };
        let pair_mod = HANGUL_SYLLABLE_MODULUS.strict_mul(HANGUL_SYLLABLE_MODULUS);
        let lo = low_u32_from_u64(rem.rem_euclid(pair_mod));
        let hi = low_u32_from_u64(rem.div_euclid(pair_mod));
        let base_mod = low_u32_from_u64(HANGUL_SYLLABLE_MODULUS);
        let offsets = [
            lo.rem_euclid(base_mod),
            lo.div_euclid(base_mod),
            hi.rem_euclid(base_mod),
            hi.div_euclid(base_mod),
        ];
        for (slot, offset) in self.data.hangul_syllables.iter_mut().zip(offsets) {
            *slot = HANGUL_BASE_CODE_POINT.strict_add(low_u16_from_u64(u64::from(offset)));
        }
    }
    fn fill_lotto_bundle_from_u64(&mut self, value: u64) {
        if self.data.planet_number != 0 {
            return;
        }
        let Some(rem) = mix_bounded::<LOTTO_BUNDLE_MODULUS>(value, MIX_SALT_LOTTO) else {
            return;
        };
        let planet_rem = rem.rem_euclid(NMS_PLANET_MODULUS);
        let rem_after_planet = rem.div_euclid(NMS_PLANET_MODULUS);
        let lucky_rank = low_u32_from_u64(rem_after_planet.rem_euclid(EURO_LUCKY_COMB));
        let rem_lottos = rem_after_planet.div_euclid(EURO_LUCKY_COMB);
        let korea_lotto_rank = low_u32_from_u64(rem_lottos.rem_euclid(LOTTO_COMB));
        let japan_lotto_rank = low_u32_from_u64(rem_lottos.div_euclid(LOTTO_COMB));
        self.data.planet_number = low_u8_from_u64(planet_rem.strict_add(1));
        unrank_combination_into(
            lucky_rank,
            EURO_LUCKY_MAX,
            &mut self.data.euro_millions_lucky_stars,
        );
        unrank_combination_into(korea_lotto_rank, LOTTO_MAX, &mut self.data.lotto_numbers);
        unrank_combination_into(japan_lotto_rank, LOTTO7_MAX, &mut self.data.lotto7_numbers);
    }
    fn fill_password_from_u64(&mut self, value: u64) {
        if !matches!(self.data.password, [0, ..]) {
            return;
        }
        let Some(rem) = mix_bounded::<PASSWORD_OCTET_MODULUS>(value, MIX_SALT_PASSWORD) else {
            return;
        };
        let quad_mod = const { ASCII_PRINTABLE_LEN.strict_pow(4) };
        let mut lo = low_u32_from_u64(rem.rem_euclid(quad_mod));
        let mut hi = low_u32_from_u64(rem.div_euclid(quad_mod));
        let base_mod = low_u32_from_u64(ASCII_PRINTABLE_LEN);
        let (lo_slots, hi_slots) = self.data.password.split_at_mut(4);
        for (lo_slot, hi_slot) in lo_slots.iter_mut().zip(hi_slots) {
            *lo_slot = ASCII_PRINTABLE_START.strict_add(low_u8_from_u32(lo.rem_euclid(base_mod)));
            *hi_slot = ASCII_PRINTABLE_START.strict_add(low_u8_from_u32(hi.rem_euclid(base_mod)));
            lo = lo.div_euclid(base_mod);
            hi = hi.div_euclid(base_mod);
        }
    }
    fn finish(self) -> Result<RandomDataSet> {
        self.check_complete()?;
        let mut data = self.data;
        data.kor_coords = self.korea.finish()?;
        Ok(data)
    }
    const fn is_basic_complete(&self) -> bool {
        !matches!(self.data.password, [0, ..])
            && self.data.planet_number != 0
            && self.data.solar_system_index != 0
    }
    const fn is_complete(&self) -> bool {
        self.is_basic_complete()
            && !matches!(self.data.hangul_syllables, [0, ..])
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

const fn galaxy_coord<const OFFSET: u16, const MASK: u16>(value: u16) -> u16 {
    // Callers supply 8-bit or 12-bit fields; shift the origin within that field.
    value.strict_add(OFFSET) & MASK
}
#[cfg(target_arch = "x86_64")]
pub(super) fn generate_random_data_with_rng(
    rng: &HardwareRng,
    boundary_cache: &mut korea_boundary::BoundaryCache,
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
const fn mix_bounded<const MODULUS: u64>(value: u64, salt: u64) -> Option<u64> {
    let limit = const { u64::MAX.div_euclid(MODULUS).strict_mul(MODULUS) };
    let mut state = value.wrapping_add(salt);
    state = (state ^ (state >> 30_u32)).wrapping_mul(MIX_MUL_A);
    state = (state ^ (state >> 27_u32)).wrapping_mul(MIX_MUL_B);
    let mixed = state ^ (state >> 31_u32);
    if mixed >= limit {
        None
    } else {
        Some(mixed.rem_euclid(MODULUS))
    }
}
fn unrank_combination_into(mut rank: u32, max_n: u8, out: &mut [u8]) {
    let Some((first_slot, tail)) = out.split_first_mut() else {
        return;
    };
    let rows = BINOMIAL_VALUES
        .get(..tail.len())
        .unwrap_or_else(|| process::abort());
    let mut upper = usize::from(max_n);
    for (tail_idx, (slot, values)) in tail.iter_mut().zip(rows).enumerate().rev() {
        let lower = tail_idx.strict_add(1);
        let row = values.get(lower..upper).unwrap_or_else(|| process::abort());
        let offset = row.partition_point(|&comb| comb <= rank).strict_sub(1);
        let combinations = *row.get(offset).unwrap_or_else(|| process::abort());
        rank = rank.strict_sub(combinations);
        let value = lower.strict_add(offset);
        let [low, ..] = value.to_le_bytes();
        *slot = low.strict_add(1);
        upper = value;
    }
    *first_slot = low_u8_from_u32(rank.strict_add(1));
}
