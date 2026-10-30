use crate::{diagnostic::Result, random_data::Coordinates};
const DEGREES: f64 = 0.017_453_292_519_943_295;
const ECCENTRICITY_SQUARED: f64 = 0.006_694_380_022_900_787;
fn product_error(a: f64, b: f64, product: f64) -> f64 {
    let split = 134_217_729.0_f64 * a;
    let ah = split - (split - a);
    let al = a - ah;
    let split = 134_217_729.0_f64 * b;
    let bh = split - (split - b);
    let bl = b - bh;
    ((ah * bh - product) + ah * bl + al * bh) + al * bl
}
pub(super) fn orientation(start: [f64; 2], end: [f64; 2], point: [f64; 2]) -> f64 {
    let a = end[0] - start[0];
    let b = end[1] - start[1];
    let c = point[0] - start[0];
    let d = point[1] - start[1];
    let left = a * d;
    let right = b * c;
    let approximate = left - right;
    if approximate.abs() > 3.330_669_073_875_471_6e-16 * (left.abs() + right.abs()) {
        return approximate;
    }
    let mut expansion = [0.0_f64; 4];
    let mut length = 0_usize;
    for mut value in [
        product_error(a, d, left),
        left,
        -product_error(b, c, right),
        -right,
    ] {
        let mut count = 0_usize;
        let previous = expansion;
        for entry in previous.into_iter().take(length) {
            let total = value + entry;
            let virtual_value = total - value;
            let error = (value - (total - virtual_value)) + (entry - virtual_value);
            if error != 0.0 {
                expansion[count] = error;
                count = count.strict_add(1);
            }
            value = total;
        }
        if value != 0.0 || count == 0 {
            expansion[count] = value;
            count = count.strict_add(1);
        }
        length = count;
    }
    expansion.iter().take(length).sum()
}
fn cumulative_area_and_derivative(latitude: f64) -> (f64, f64) {
    let delta = (latitude - 36.0) * DEGREES;
    let square = delta * delta;
    let sine_delta = delta
        * (1.0
            + square
                * (-0.166_666_666_666_666_66
                    + square
                        * (0.008_333_333_333_333_333
                            + square
                                * (-0.000_198_412_698_412_698_4
                                    + square
                                        * (2.755_731_922_398_589_3e-6
                                            + square * -2.505_210_838_544_172e-8)))));
    let cosine_delta = 1.0
        + square
            * (-0.5
                + square
                    * (0.041_666_666_666_666_664
                        + square
                            * (-0.001_388_888_888_888_889
                                + square
                                    * (2.480_158_730_158_73e-5
                                        + square
                                            * (-2.755_731_922_398_589e-7
                                                + square * 2.087_675_698_786_81e-9)))));
    let sine = 0.587_785_252_292_473_1 * cosine_delta + 0.809_016_994_374_947_5 * sine_delta;
    let square_eccentric_sine = ECCENTRICITY_SQUARED * sine * sine;
    let mut polynomial = 9.0 / 17.0;
    for coefficient in [
        8.0 / 15.0,
        7.0 / 13.0,
        6.0 / 11.0,
        5.0 / 9.0,
        4.0 / 7.0,
        3.0 / 5.0,
        2.0 / 3.0,
        1.0,
    ] {
        polynomial = coefficient + square_eccentric_sine * polynomial;
    }
    let cosine = 0.809_016_994_374_947_5 * cosine_delta - 0.587_785_252_292_473_1 * sine_delta;
    let denominator = 1.0 - square_eccentric_sine;
    (
        sine * polynomial,
        DEGREES * cosine / (denominator * denominator),
    )
}
pub(super) fn candidate(value: u64, bounds: [f64; 4]) -> Result<Coordinates> {
    let [west, south, east, north] = bounds;
    (bounds.into_iter().all(f64::is_finite)
        && west >= -180.0
        && east <= 180.0
        && west < east
        && south >= 32.0
        && north <= 40.0
        && south < north)
        .ok_or("대한민국 등면적 좌표 범위 오류")?;
    let [b0, b1, b2, b3, b4, b5, b6, b7] = value.to_be_bytes();
    let vertical = (f64::from(u32::from_be_bytes([b0, b1, b2, b3])) + 0.5) / 4_294_967_296.0;
    let horizontal = (f64::from(u32::from_be_bytes([b4, b5, b6, b7])) + 0.5) / 4_294_967_296.0;
    let south_area = cumulative_area_and_derivative(south).0;
    let target = south_area + vertical * (cumulative_area_and_derivative(north).0 - south_area);
    let mut latitude = south + (north - south) * vertical;
    for _ in 0..3 {
        let (area, derivative) = cumulative_area_and_derivative(latitude);
        latitude -= (area - target) / derivative;
    }
    let longitude = west + horizontal * (east - west);
    (latitude > south && latitude < north && longitude > west && longitude < east)
        .ok_or("대한민국 등면적 좌표 역산 오류")?;
    Ok(Coordinates {
        latitude,
        longitude,
    })
}
