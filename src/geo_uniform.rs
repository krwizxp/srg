use crate::{
    diagnostic::{AppError, Result},
    random_data::Coordinates,
};
const DEGREES: f64 = 0.017_453_292_519_943_295_f64;
const ECCENTRICITY_SQUARED: f64 = 0.006_694_380_022_900_787_f64;
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum Orientation {
    Negative,
    Positive,
    Unordered,
    Zero,
}
impl TryFrom<(u64, [f64; 4])> for Coordinates {
    type Error = AppError;
    fn try_from((value, bounds): (u64, [f64; 4])) -> Result<Self> {
        let [west, south, east, north] = bounds;
        (bounds.into_iter().all(f64::is_finite)
            && west >= -180.0_f64
            && east <= 180.0_f64
            && west < east
            && south >= 32.0_f64
            && north <= 40.0_f64
            && south < north)
            .ok_or("대한민국 등면적 좌표 범위 오류")?;
        let [b0, b1, b2, b3, b4, b5, b6, b7] = value.to_be_bytes();
        let vertical =
            (f64::from(u32::from_be_bytes([b0, b1, b2, b3])) + 0.5_f64) / 4_294_967_296.0_f64;
        let horizontal =
            (f64::from(u32::from_be_bytes([b4, b5, b6, b7])) + 0.5_f64) / 4_294_967_296.0_f64;
        let south_area = cumulative_area_and_derivative(south).0;
        let area_span = cumulative_area_and_derivative(north).0 - south_area;
        let area_offset = vertical * area_span;
        let target = south_area + area_offset;
        let latitude_offset = (north - south) * vertical;
        let mut latitude = south + latitude_offset;
        for _ in 0_usize..3_usize {
            let (area, derivative) = cumulative_area_and_derivative(latitude);
            latitude -= (area - target) / derivative;
        }
        let longitude_offset = horizontal * (east - west);
        let longitude = west + longitude_offset;
        (latitude > south && latitude < north && longitude > west && longitude < east)
            .ok_or("대한민국 등면적 좌표 역산 오류")?;
        Ok(Self {
            latitude,
            longitude,
        })
    }
}
impl TryFrom<([f64; 2], [f64; 2], [f64; 2])> for Orientation {
    type Error = AppError;
    fn try_from((start, end, point): ([f64; 2], [f64; 2], [f64; 2])) -> Result<Self> {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let px = point[0] - start[0];
        let py = point[1] - start[1];
        let left = dx * py;
        let right = dy * px;
        let approximate = left - right;
        let determinant: f64 =
            if approximate.abs() > 3.330_669_073_875_471_6e-16_f64 * (left.abs() + right.abs()) {
                approximate
            } else {
                let mut expansion = [0.0_f64; 4];
                let mut length = 0_usize;
                for mut value in [
                    product_error(dx, py, left),
                    left,
                    -product_error(dy, px, right),
                    -right,
                ] {
                    let mut count = 0_usize;
                    let previous = expansion;
                    for entry in previous.into_iter().take(length) {
                        let total = value + entry;
                        let virtual_value = total - value;
                        let error = (value - (total - virtual_value)) + (entry - virtual_value);
                        if error != 0.0_f64 {
                            *expansion
                                .get_mut(count)
                                .ok_or("대한민국 경계 좌표 누산 범위 오류")? = error;
                            count = count.strict_add(1);
                        }
                        value = total;
                    }
                    if value != 0.0_f64 || count == 0 {
                        *expansion
                            .get_mut(count)
                            .ok_or("대한민국 경계 좌표 누산 범위 오류")? = value;
                        count = count.strict_add(1);
                    }
                    length = count;
                }
                expansion.iter().take(length).sum()
            };
        Ok(if determinant < 0.0_f64 {
            Self::Negative
        } else if determinant > 0.0_f64 {
            Self::Positive
        } else if determinant == 0.0_f64 {
            Self::Zero
        } else {
            Self::Unordered
        })
    }
}
fn cumulative_area_and_derivative(latitude: f64) -> (f64, f64) {
    let delta = (latitude - 36.0_f64) * DEGREES;
    let square = delta * delta;
    let mut sine_polynomial = -2.505_210_838_544_172e-8_f64;
    for coefficient in [
        2.755_731_922_398_589_3e-6_f64,
        -0.000_198_412_698_412_698_4_f64,
        0.008_333_333_333_333_333_f64,
        -0.166_666_666_666_666_66_f64,
        1.0_f64,
    ] {
        let term = square * sine_polynomial;
        sine_polynomial = coefficient + term;
    }
    let sine_delta = delta * sine_polynomial;
    let mut cosine_polynomial = 2.087_675_698_786_81e-9_f64;
    for coefficient in [
        -2.755_731_922_398_589e-7_f64,
        2.480_158_730_158_73e-5_f64,
        -0.001_388_888_888_888_889_f64,
        0.041_666_666_666_666_664_f64,
        -0.5_f64,
        1.0_f64,
    ] {
        let term = square * cosine_polynomial;
        cosine_polynomial = coefficient + term;
    }
    let cosine_delta = cosine_polynomial;
    let cosine_part = 0.587_785_252_292_473_1_f64 * cosine_delta;
    let sine_part = 0.809_016_994_374_947_5_f64 * sine_delta;
    let sine = cosine_part + sine_part;
    let square_eccentric_sine = ECCENTRICITY_SQUARED * sine * sine;
    let mut polynomial = 9.0_f64 / 17.0_f64;
    for coefficient in [
        8.0_f64 / 15.0_f64,
        7.0_f64 / 13.0_f64,
        6.0_f64 / 11.0_f64,
        5.0_f64 / 9.0_f64,
        4.0_f64 / 7.0_f64,
        3.0_f64 / 5.0_f64,
        2.0_f64 / 3.0_f64,
        1.0_f64,
    ] {
        let contribution = square_eccentric_sine * polynomial;
        polynomial = coefficient + contribution;
    }
    let longitude_cosine_term = 0.809_016_994_374_947_5_f64 * cosine_delta;
    let longitude_sine_term = 0.587_785_252_292_473_1_f64 * sine_delta;
    let cosine = longitude_cosine_term - longitude_sine_term;
    let denominator = 1.0_f64 - square_eccentric_sine;
    let numerator = DEGREES * cosine;
    let denominator_squared = denominator * denominator;
    (sine * polynomial, numerator / denominator_squared)
}
fn product_error(left: f64, right: f64, product: f64) -> f64 {
    let split_left = 134_217_729.0_f64 * left;
    let ah = split_left - (split_left - left);
    let al = left - ah;
    let split_right = 134_217_729.0_f64 * right;
    let bh = split_right - (split_right - right);
    let bl = right - bh;
    let high_product = ah * bh;
    let high_error = high_product - product;
    let cross_left = ah * bl;
    let cross_right = al * bh;
    let low_product = al * bl;
    ((high_error + cross_left) + cross_right) + low_product
}
