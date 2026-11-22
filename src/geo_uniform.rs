use crate::{
    diagnostic::{AppError, Result},
    numeric::split_u64_be_u32,
    random_data::Coordinates,
};
const DEGREES: f64 = 0.017_453_292_519_943_295_f64;
const ECCENTRICITY_SQUARED: f64 = 0.006_694_380_022_900_787_f64;
#[derive(Clone, Copy)]
pub(super) struct BoundaryAreaBounds {
    area_span: f64,
    east: f64,
    north: f64,
    south: f64,
    south_area: f64,
    west: f64,
}
impl TryFrom<[f64; 4]> for BoundaryAreaBounds {
    type Error = AppError;
    fn try_from(bounds: [f64; 4]) -> Result<Self> {
        let [west, south, east, north] = bounds;
        (bounds.into_iter().all(f64::is_finite)
            && west >= -180.0_f64
            && east <= 180.0_f64
            && west < east
            && south >= 32.0_f64
            && north <= 40.0_f64
            && south < north)
            .ok_or("대한민국 등면적 좌표 범위 오류")?;
        let south_area = cumulative_area_and_derivative(south).0;
        let area_span = cumulative_area_and_derivative(north).0 - south_area;
        Ok(Self {
            area_span,
            east,
            north,
            south,
            south_area,
            west,
        })
    }
}
impl TryFrom<(u64, BoundaryAreaBounds)> for Coordinates {
    type Error = AppError;
    fn try_from((value, bounds): (u64, BoundaryAreaBounds)) -> Result<Self> {
        let (hi, lo) = split_u64_be_u32(value);
        let vertical = (f64::from(hi) + 0.5_f64) / 4_294_967_296.0_f64;
        let horizontal = (f64::from(lo) + 0.5_f64) / 4_294_967_296.0_f64;
        let area_offset = vertical * bounds.area_span;
        let target = bounds.south_area + area_offset;
        let latitude_offset = (bounds.north - bounds.south) * vertical;
        let mut latitude = bounds.south + latitude_offset;
        for _ in 0_usize..3_usize {
            let (area, derivative) = cumulative_area_and_derivative(latitude);
            latitude -= (area - target) / derivative;
        }
        let longitude_offset = horizontal * (bounds.east - bounds.west);
        let longitude = bounds.west + longitude_offset;
        (latitude > bounds.south
            && latitude < bounds.north
            && longitude > bounds.west
            && longitude < bounds.east)
            .ok_or("대한민국 등면적 좌표 역산 오류")?;
        Ok(Self {
            latitude,
            longitude,
        })
    }
}
fn cumulative_area_and_derivative(latitude: f64) -> (f64, f64) {
    let delta = (latitude - 36.0_f64) * DEGREES;
    let square = delta * delta;
    let sine_delta = delta
        * eval_poly(
            square,
            -2.505_210_838_544_172e-8_f64,
            &[
                2.755_731_922_398_589_3e-6_f64,
                -0.000_198_412_698_412_698_4_f64,
                0.008_333_333_333_333_333_f64,
                -0.166_666_666_666_666_66_f64,
                1.0_f64,
            ],
        );
    let cosine_delta = eval_poly(
        square,
        2.087_675_698_786_81e-9_f64,
        &[
            -2.755_731_922_398_589e-7_f64,
            2.480_158_730_158_73e-5_f64,
            -0.001_388_888_888_888_889_f64,
            0.041_666_666_666_666_664_f64,
            -0.5_f64,
            1.0_f64,
        ],
    );
    let cosine_part = 0.587_785_252_292_473_1_f64 * cosine_delta;
    let sine_part = 0.809_016_994_374_947_5_f64 * sine_delta;
    let sine = cosine_part + sine_part;
    let square_eccentric_sine = ECCENTRICITY_SQUARED * sine * sine;
    let polynomial = eval_poly(
        square_eccentric_sine,
        9.0_f64 / 17.0_f64,
        &[
            8.0_f64 / 15.0_f64,
            7.0_f64 / 13.0_f64,
            6.0_f64 / 11.0_f64,
            5.0_f64 / 9.0_f64,
            4.0_f64 / 7.0_f64,
            3.0_f64 / 5.0_f64,
            2.0_f64 / 3.0_f64,
            1.0_f64,
        ],
    );
    let longitude_cosine_term = 0.809_016_994_374_947_5_f64 * cosine_delta;
    let longitude_sine_term = 0.587_785_252_292_473_1_f64 * sine_delta;
    let cosine = longitude_cosine_term - longitude_sine_term;
    let denominator = 1.0_f64 - square_eccentric_sine;
    let numerator = DEGREES * cosine;
    let denominator_squared = denominator * denominator;
    (sine * polynomial, numerator / denominator_squared)
}
fn eval_poly(point: f64, initial: f64, coefficients: &[f64]) -> f64 {
    coefficients.iter().fold(initial, |acc, &coeff| {
        let term = point * acc;
        coeff + term
    })
}
