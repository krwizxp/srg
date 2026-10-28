use crate::{diagnostic::Result, random_data::Coordinates};
#[cfg(test)]
mod tests;
const DEGREES: f64 = 0.017_453_292_519_943_295;
const ECCENTRICITY_SQUARED: f64 = 0.006_694_380_022_900_787;
const SURFACE_SCALE: f64 = 12_309_072_078.888_908;
fn product_error(a: f64, b: f64, product: f64) -> f64 {
    let split = 134_217_729.0_f64 * a;
    let ah = split - (split - a);
    let al = a - ah;
    let split = 134_217_729.0_f64 * b;
    let bh = split - (split - b);
    let bl = b - bh;
    ((ah * bh - product) + ah * bl + al * bh) + al * bl
}
fn determinant(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let left = a * d;
    let right = b * c;
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
fn density(latitude: f64) -> f64 {
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
    let cosine = 0.809_016_994_374_947_5 * cosine_delta - 0.587_785_252_292_473_1 * sine_delta;
    let denominator = 1.0 - ECCENTRICITY_SQUARED * sine * sine;
    cosine / (denominator * denominator)
}
fn float_from_u64(value: u64) -> f64 {
    let [b0, b1, b2, b3, b4, b5, b6, b7] = value.to_be_bytes();
    f64::from(u32::from_be_bytes([b0, b1, b2, b3])).mul_add(
        4_294_967_296.0,
        f64::from(u32::from_be_bytes([b4, b5, b6, b7])),
    )
}
fn unit(value: u64) -> f64 {
    (float_from_u64(value >> 12_u32) + 0.5) / 4_503_599_627_370_496.0
}
pub(super) fn propose(
    vertices: &[[f64; 2]; 3],
    weight: u64,
    words: [u64; 3],
) -> Result<Option<Coordinates>> {
    let [a, b, c] = *vertices;
    let det = determinant(b[1] - a[1], b[0] - a[0], c[1] - a[1], c[0] - a[0]);
    let area = det.abs() * 0.5;
    (area > 0.0 && area.is_finite() && weight > 0).ok_or("대한민국 삼각형 면적 오류")?;
    let mut u = unit(words[0]);
    let mut v = unit(words[1]);
    if u + v > 1.0 {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    let latitude = (a[0] + u * (b[0] - a[0])) + v * (c[0] - a[0]);
    let longitude = (a[1] + u * (b[1] - a[1])) + v * (c[1] - a[1]);
    for (start, end) in [(a, b), (b, c), (c, a)] {
        let side = determinant(
            end[1] - start[1],
            end[0] - start[0],
            longitude - start[1],
            latitude - start[0],
        );
        if (det > 0.0 && side < 0.0) || (det < 0.0 && side > 0.0) {
            return Ok(None);
        }
    }
    let probability =
        ((area * SURFACE_SCALE) * density(latitude)) * 1_000_000.0 / float_from_u64(weight);
    (probability.is_finite() && probability > 0.0 && probability <= 1.0)
        .ok_or("대한민국 면적 수락 확률 오류")?;
    if unit(words[2]) < probability {
        Ok(Some(Coordinates {
            latitude,
            longitude,
        }))
    } else {
        Ok(None)
    }
}
