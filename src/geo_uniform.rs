use crate::{diagnostic::Result, random_data::Coordinates};
#[cfg(test)]
mod tests;
const DEGREES: f64 = 0.017_453_292_519_943_295;
const ECCENTRICITY_SQUARED: f64 = 0.006_694_380_022_900_787;
const QUADRATURE: [(f64, f64); 4] = [
    (0.069_431_844_202_973_71, 0.173_927_422_568_726_92),
    (0.330_009_478_207_571_87, 0.326_072_577_431_273_05),
    (0.669_990_521_792_428_1, 0.326_072_577_431_273_05),
    (0.930_568_155_797_026_2, 0.173_927_422_568_726_92),
];
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
fn triangular_density(start: f64, delta: f64) -> f64 {
    let mut total = 0.0;
    for (position, weight) in QUADRATURE {
        total += (2.0 * position * weight) * density(start + delta * position);
    }
    total
}
fn residual_position(rank: u64, count: u64) -> [f64; 2] {
    let vertical = ((float_from_u64(rank) + 0.5) / float_from_u64(count)).min(1.0_f64.next_down());
    let reversed = rank.reverse_bits() ^ 0x9E37_79B9_7F4A_7C15_u64;
    let horizontal = (float_from_u64(reversed >> 12_u32) + 0.5) / 4_503_599_627_370_496.0;
    [vertical, horizontal]
}
fn inside(vertices: &[[f64; 2]; 3], point: [f64; 2], orientation: f64) -> bool {
    let [a, b, c] = *vertices;
    [(a, b), (b, c), (c, a)].into_iter().all(|(start, end)| {
        let side = determinant(
            end[1] - start[1],
            end[0] - start[0],
            point[1] - start[1],
            point[0] - start[0],
        );
        if orientation > 0.0 {
            side >= 0.0
        } else {
            side <= 0.0
        }
    })
}
pub(super) fn sample(vertices: &[[f64; 2]; 3], rank: u64, count: u64) -> Result<Coordinates> {
    let [a, b, c] = *vertices;
    let orientation = determinant(b[1] - a[1], b[0] - a[0], c[1] - a[1], c[0] - a[0]);
    (orientation != 0.0 && orientation.is_finite() && count > 0 && rank < count)
        .ok_or("대한민국 삼각형 면적 또는 입력 구간 오류")?;
    let mut ordered = *vertices;
    ordered.sort_by(|left, right| left[0].total_cmp(&right[0]));
    let [south, middle, north] = ordered;
    let span = north[0] - south[0];
    (span > 0.0).ok_or("대한민국 삼각형 위도 오류")?;
    let fraction = (middle[0] - south[0]) / span;
    let crossing = [middle[0], south[1] + fraction * (north[1] - south[1])];
    let lower_mass = fraction * triangular_density(south[0], middle[0] - south[0]);
    let upper_mass = (1.0 - fraction) * triangular_density(north[0], middle[0] - north[0]);
    let [vertical, horizontal] = residual_position(rank, count);
    let target = vertical * (lower_mass + upper_mass);
    let (apex, unit) = if target < lower_mass {
        (south, target / lower_mass)
    } else {
        (north, 1.0 - (target - lower_mass) / upper_mass)
    };
    let delta = middle[0] - apex[0];
    let mean = triangular_density(apex[0], delta);
    let mut distance = unit.clamp(0.0, 1.0).sqrt();
    for _ in 0..4 {
        if distance == 0.0 {
            break;
        }
        let mass = distance * distance * triangular_density(apex[0], delta * distance);
        distance -= (mass - unit * mean) / (2.0 * distance * density(apex[0] + delta * distance));
    }
    let edge = [
        apex[0] + distance * (middle[0] - apex[0]),
        apex[1] + distance * (middle[1] - apex[1]),
    ];
    let other = apex[1] + distance * (crossing[1] - apex[1]);
    let mut point = [edge[0], edge[1] + horizontal * (other - edge[1])];
    let center = [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0];
    for _ in 0..8 {
        if inside(vertices, point, orientation) {
            return Ok(Coordinates {
                latitude: point[0],
                longitude: point[1],
            });
        }
        for (coordinate, destination) in point.iter_mut().zip(center) {
            *coordinate = if *coordinate < destination {
                coordinate.next_up()
            } else {
                coordinate.next_down()
            };
        }
    }
    let nearest = vertices
        .iter()
        .min_by(|left, right| {
            let square = |vertex: &&[f64; 2]| {
                (vertex[0] - point[0]).powi(2) + (vertex[1] - point[1]).powi(2)
            };
            square(left).total_cmp(&square(right))
        })
        .ok_or("대한민국 삼각형 정점 오류")?;
    Ok(Coordinates {
        latitude: nearest[0],
        longitude: nearest[1],
    })
}
