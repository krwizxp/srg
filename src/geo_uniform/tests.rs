use super::{density, determinant, residual_position, sample};
use crate::diagnostic::Result;
#[test]
fn jacobian_matches_independent_trigonometry() {
    for step in 0_u32..=6000 {
        let phi = (33.0_f64 + f64::from(step) / 1000.0).to_radians();
        let denominator = 1.0_f64 - 0.006_694_380_022_900_787_f64 * phi.sin().powi(2);
        let expected = phi.cos() / denominator.powi(2);
        assert!(
            (density(phi.to_degrees()) / expected - 1.0_f64).abs() < 1e-14_f64,
            "Area Jacobian differs from the GRS80 reference"
        );
    }
}
#[test]
fn residual_positions_are_interior_and_orientation_survives() {
    for count in 1_u64..=257 {
        for rank in 0..count {
            let [vertical, horizontal] = residual_position(rank, count);
            assert!(
                (0.0..1.0).contains(&vertical) && vertical > 0.0,
                "Latitude stratum is not interior"
            );
            assert!(
                (0.0..1.0).contains(&horizontal) && horizontal > 0.0,
                "Longitude stratum is not interior"
            );
        }
    }
    assert!(
        determinant(1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64.next_up()) > 0.0_f64,
        "Positive orientation was lost"
    );
    assert!(
        determinant(1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64.next_down()) < 0.0_f64,
        "Negative orientation was lost"
    );
}
#[test]
fn invalid_geometry_cannot_produce_a_fallback() -> Result<()> {
    let vertices = [[35.0_f64, 127.0_f64]; 3];
    assert!(
        sample(&vertices, 0, 1).is_err(),
        "A zero-area triangle must fail"
    );
    Ok(())
}
#[test]
fn smallest_triangle_does_not_fold_distinct_inputs() -> Result<()> {
    let vertices = [
        [34.501_653_444_045_03_f64, 127.5_f64],
        [34.502_102_822_576_27_f64, 127.5_f64],
        [34.502_102_822_576_26_f64, 127.500_000_000_000_01_f64],
    ];
    let mut previous = vertices[0][0];
    for rank in 0_u64..6 {
        let point = sample(&vertices, rank, 6)?;
        assert!(
            point.latitude > previous,
            "Distinct residual inputs collapsed to the same latitude"
        );
        previous = point.latitude;
    }
    Ok(())
}
#[test]
fn residual_positions_remain_open_above_binary64_integer_precision() {
    for count in [
        (1_u64 << 52_u32).strict_add(1),
        (1_u64 << 53_u32).strict_add(1),
        1_u64 << 63_u32,
        u64::MAX,
    ] {
        for rank in [0, 1, count.div_euclid(2), count.strict_sub(1)] {
            let [vertical, horizontal] = residual_position(rank, count);
            assert!(
                vertical > 0.0 && vertical < 1.0 && horizontal > 0.0 && horizontal < 1.0,
                "Residual mapping reached an endpoint"
            );
        }
    }
}
