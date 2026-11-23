#[cfg(target_arch = "x86_64")]
use super::{diagnostic::AppError, hardware_rng::HardwareRng};
use crate::{
    diagnostic::{Result, or_abort},
    korea_boundary,
    numeric::{
        low_u8_from_u32, low_u8_from_u64, low_u16_from_u64, low_u32_from_u64, split_u64_be_u32,
    },
};
const ASCII_PRINTABLE_LEN: u64 = 94;
const BINOMIAL_VALUES: [[u32; 50]; 6] = {
    let mut values = [[0; 50]; 6];
    let (mut rows, mut choose) = (values.as_mut_slice(), 2_u32);
    while let Some((row, rem_rows)) = rows.split_first_mut() {
        let (mut cells, mut total) = (row.as_mut_slice(), 0_u32);
        while let Some((cell, rem_cells)) = cells.split_first_mut() {
            *cell = binomial(total, choose);
            total = total.strict_add(1);
            cells = rem_cells;
        }
        choose = choose.strict_add(1);
        rows = rem_rows;
    }
    values
};
const EURO_LUCKY_COMB: u64 = binomial(12, 2) as u64;
const EURO_PIN_NMS_MODULUS: u64 = (binomial(50, 5) as u64)
    .strict_mul(1_000_000)
    .strict_mul(767);
const HANGUL_QUAD_MODULUS: u64 = 11_172_u64.strict_pow(4);
const LOTTO_BUNDLE_MODULUS: u64 = LOTTO_COMB
    .strict_mul(binomial(37, 7) as u64)
    .strict_mul(EURO_LUCKY_COMB)
    .strict_mul(6);
const LOTTO_COMB: u64 = binomial(45, 6) as u64;
const PASSWORD_OCTET_MODULUS: u64 = ASCII_PRINTABLE_LEN.strict_pow(8);
#[derive(Default)]
pub(super) struct Coordinates {
    pub latitude: f64,
    pub longitude: f64,
}
struct RandomDataBuildState<'cache> {
    data: RandomDataSet,
    korea: korea_boundary::SampleState<'cache>,
}
#[derive(Default)]
pub(super) struct RandomDataSet {
    pub euro_millions_lucky_stars: [u8; 2],
    pub euro_millions_main_numbers: [u8; 5],
    pub galaxy_x: u16,
    pub galaxy_y: u16,
    pub galaxy_z: u16,
    pub hangul_syllables: [u16; 4],
    pub kor_coords: Coordinates,
    pub lotto7_numbers: [u8; 7],
    pub lotto_numbers: [u8; 6],
    pub nms_portal_xxx: u16,
    pub nms_portal_yy: u8,
    pub nms_portal_zzz: u16,
    pub num_64: u64,
    pub numeric_password: u32,
    pub password: [u8; 8],
    pub planet_number: u8,
    pub solar_system_index: u16,
    pub world_coords: Coordinates,
}
impl RandomDataBuildState<'_> {
    fn fill_coords(&mut self) {
        let (hi, lo) = split_u64_be_u32(self.data.num_64);
        let inv = 1.0_f64 / 4_294_967_295.0_f64;
        self.data.world_coords = Coordinates {
            latitude: 180.0_f64.mul_add(f64::from(hi).algebraic_mul(inv), -90.0),
            longitude: 360.0_f64.mul_add(f64::from(lo).algebraic_mul(inv), -180.0),
        };
        self.data.nms_portal_yy = low_u8_from_u32(hi);
        self.data.nms_portal_zzz = low_u16_from_u64((self.data.num_64 >> 20) & 0x0FFF);
        self.data.nms_portal_xxx = low_u16_from_u64((self.data.num_64 >> 8) & 0x0FFF);
        self.data.galaxy_x = galaxy_coord::<0x7FF, 0xFFF>(self.data.nms_portal_xxx);
        self.data.galaxy_y = galaxy_coord::<0x7F, 0xFF>(u16::from(self.data.nms_portal_yy));
        self.data.galaxy_z = galaxy_coord::<0x7FF, 0xFFF>(self.data.nms_portal_zzz);
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
        if self.data.planet_number == 0
            && let Some(rem) = mix_bounded::<LOTTO_BUNDLE_MODULUS>(value, 0xD1B5_4A32_D192_ED03)
        {
            self.data.planet_number = low_u8_from_u64(rem.rem_euclid(6).strict_add(1));
            let after_planet = rem.div_euclid(6);
            let lottos = after_planet.div_euclid(EURO_LUCKY_COMB);
            unrank_combination_into(
                low_u32_from_u64(after_planet.rem_euclid(EURO_LUCKY_COMB)),
                12,
                &mut self.data.euro_millions_lucky_stars,
            );
            unrank_combination_into(
                low_u32_from_u64(lottos.rem_euclid(LOTTO_COMB)),
                45,
                &mut self.data.lotto_numbers,
            );
            unrank_combination_into(
                low_u32_from_u64(lottos.div_euclid(LOTTO_COMB)),
                37,
                &mut self.data.lotto7_numbers,
            );
        }
        if self.data.solar_system_index == 0
            && let Some(rem) = mix_bounded::<EURO_PIN_NMS_MODULUS>(value, 0x9E37_79B9_7F4A_7C15)
        {
            self.data.solar_system_index = low_u16_from_u64(rem.rem_euclid(767).strict_add(1));
            let pin_euro = rem.div_euclid(767);
            self.data.numeric_password = low_u32_from_u64(pin_euro.rem_euclid(1_000_000));
            unrank_combination_into(
                low_u32_from_u64(pin_euro.div_euclid(1_000_000)),
                50,
                &mut self.data.euro_millions_main_numbers,
            );
        }
        if matches!(self.data.hangul_syllables, [0, ..])
            && let Some(rem) = mix_bounded::<HANGUL_QUAD_MODULUS>(value, 0x3C6E_F372_FE94_F82A)
        {
            let pair_mod = 11_172_u64.strict_mul(11_172);
            let (lo, hi) = (
                low_u32_from_u64(rem.rem_euclid(pair_mod)),
                low_u32_from_u64(rem.div_euclid(pair_mod)),
            );
            let offsets = [
                lo.rem_euclid(11_172),
                lo.div_euclid(11_172),
                hi.rem_euclid(11_172),
                hi.div_euclid(11_172),
            ];
            for (slot, offset) in self.data.hangul_syllables.iter_mut().zip(offsets) {
                *slot = 0xAC00_u16.strict_add(low_u16_from_u64(u64::from(offset)));
            }
        }
        if matches!(self.data.password, [0, ..])
            && let Some(rem) = mix_bounded::<PASSWORD_OCTET_MODULUS>(value, 0xA5A5_A5A5_5A5A_5A5A)
        {
            let quad_mod = const { ASCII_PRINTABLE_LEN.strict_pow(4) };
            let (mut lo, mut hi) = (
                low_u32_from_u64(rem.rem_euclid(quad_mod)),
                low_u32_from_u64(rem.div_euclid(quad_mod)),
            );
            let (lo_slots, hi_slots) = self.data.password.split_at_mut(4);
            for (lo_slot, hi_slot) in lo_slots.iter_mut().zip(hi_slots) {
                *lo_slot = 33_u8.strict_add(low_u8_from_u32(lo.rem_euclid(94)));
                *hi_slot = 33_u8.strict_add(low_u8_from_u32(hi.rem_euclid(94)));
                lo = lo.div_euclid(94);
                hi = hi.div_euclid(94);
            }
        }
        if !initial {
            self.korea.offer_supplement(value)?;
        }
        Ok(())
    }
}
impl RandomDataSet {
    pub(super) fn populate<NextSupp>(
        self,
        mut next_supp: NextSupp,
        boundary_cache: &mut korea_boundary::BoundaryCache,
    ) -> Result<Self>
    where
        NextSupp: FnMut(&'static str) -> Result<u64>,
    {
        let initial = self.num_64;
        let mut fields = RandomDataBuildState {
            korea: korea_boundary::SampleState::try_from((initial, boundary_cache))?,
            data: self,
        };
        fields.fill_coords();
        fields.offer_word(initial, true)?;
        for _ in 0..1024_usize {
            if fields.is_complete() {
                break;
            }
            fields.offer_word(next_supp("미완성 필드 보완")?, false)?;
        }
        fields
            .is_basic_complete()
            .ok_or("기본 필드 보완 난수 시도 횟수를 초과했습니다.")?;
        (!matches!(fields.data.hangul_syllables, [0, ..]))
            .ok_or("한글 음절 보완 난수 시도 횟수를 초과했습니다.")?;
        fields.data.kor_coords = fields.korea.finish()?;
        Ok(fields.data)
    }
}
const fn binomial(total: u32, choose: u32) -> u32 {
    if choose > total {
        return 0;
    }
    let (mut result, mut step) = (1_u32, 1_u32);
    while step <= choose {
        result = result
            .strict_mul(total.strict_sub(choose).strict_add(step))
            .div_euclid(step);
        step = step.strict_add(1);
    }
    result
}
const fn galaxy_coord<const OFFSET: u16, const MASK: u16>(value: u16) -> u16 {
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
    state = (state ^ (state >> 30_u32)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = (state ^ (state >> 27_u32)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let mixed = state ^ (state >> 31_u32);
    if mixed >= limit {
        None
    } else {
        Some(mixed.rem_euclid(MODULUS))
    }
}
#[inline]
fn unrank_combination_into(mut rank: u32, max_n: u8, out: &mut [u8]) {
    let Some((first_slot, tail)) = out.split_first_mut() else {
        return;
    };
    let rows = or_abort(BINOMIAL_VALUES.get(..tail.len()));
    let mut upper = usize::from(max_n);
    for (tail_idx, (slot, values)) in tail.iter_mut().zip(rows).enumerate().rev() {
        let lower = tail_idx.strict_add(1);
        let row = or_abort(values.get(lower..upper));
        let offset = row.partition_point(|&comb| comb <= rank).strict_sub(1);
        rank = rank.strict_sub(*or_abort(row.get(offset)));
        let value = lower.strict_add(offset);
        let [low, ..] = value.to_le_bytes();
        *slot = low.strict_add(1);
        upper = value;
    }
    *first_slot = low_u8_from_u32(rank.strict_add(1));
}
