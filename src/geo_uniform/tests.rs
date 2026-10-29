use super::{candidate, cumulative_area, orientation};
use crate::diagnostic::Result;
const BOUNDS: [f64; 4] = [124.0, 32.0, 132.0, 40.0];
fn reference_area(latitude: f64) -> f64 {
    let sine = latitude.to_radians().sin();
    let eccentricity_squared = 0.006_694_380_022_900_787_f64;
    let eccentricity = eccentricity_squared.sqrt();
    sine / (2.0 * (1.0 - eccentricity_squared * sine * sine))
        + (eccentricity * sine).atanh() / (2.0 * eccentricity)
}
#[test]
fn cumulative_area_matches_independent_grs80_formula() {
    for step in 0_u32..=8000 {
        let latitude = 32.0 + f64::from(step) / 1000.0;
        assert!(
            (cumulative_area(latitude) - reference_area(latitude)).abs() < 5e-16,
            "Cumulative area differs from the GRS80 closed form"
        );
        if step > 0 && step < 8000 {
            let derivative = (cumulative_area(latitude + 0.0001)
                - cumulative_area(latitude - 0.0001))
                / 0.0002_f64.to_radians();
            let phi = latitude.to_radians();
            let denominator = 1.0 - 0.006_694_380_022_900_787 * phi.sin().powi(2);
            let expected = phi.cos() / denominator.powi(2);
            assert!(
                (derivative / expected - 1.0).abs() < 1e-9,
                "Cumulative area derivative differs from the GRS80 Jacobian"
            );
        }
    }
}
#[test]
fn full_u64_extremes_produce_open_independent_axes() -> Result<()> {
    for (value, latitude_bits, longitude_bits) in [
        (0, 0x4040_0000_0001_E872_u64, 0x405F_0000_0001_0000_u64),
        (1, 0x4040_0000_0001_E872, 0x405F_0000_0003_0000),
        (
            u64::from(u32::MAX),
            0x4040_0000_0001_E872,
            0x4060_7FFF_FFFF_8000,
        ),
        (
            1_u64 << 32_u32,
            0x4040_0000_0005_B960,
            0x405F_0000_0001_0000,
        ),
        (
            u64::MAX.strict_sub(1),
            0x4043_FFFF_FFFD_E436,
            0x4060_7FFF_FFFE_8000,
        ),
        (u64::MAX, 0x4043_FFFF_FFFD_E436, 0x4060_7FFF_FFFF_8000),
    ] {
        let point = candidate(value, BOUNDS)?;
        assert_eq!(
            (point.latitude.to_bits(), point.longitude.to_bits()),
            (latitude_bits, longitude_bits),
            "Binary64 candidate contract changed for {value}"
        );
        assert!(
            point.latitude > BOUNDS[1]
                && point.latitude < BOUNDS[3]
                && point.longitude > BOUNDS[0]
                && point.longitude < BOUNDS[2],
            "A candidate reached a rectangle boundary"
        );
    }
    let origin = candidate(0, BOUNDS)?;
    let horizontal = candidate(1, BOUNDS)?;
    let vertical = candidate(1_u64 << 32_u32, BOUNDS)?;
    assert_eq!(
        origin.latitude.to_bits(),
        horizontal.latitude.to_bits(),
        "Low bits changed latitude"
    );
    assert_eq!(
        origin.longitude.to_bits(),
        vertical.longitude.to_bits(),
        "High bits changed longitude"
    );
    assert!(
        horizontal.longitude > origin.longitude && vertical.latitude > origin.latitude,
        "Distinct adjacent indices collapsed"
    );
    Ok(())
}
#[test]
fn inverse_area_is_monotone_and_matches_cell_centers() -> Result<()> {
    let south_area = reference_area(BOUNDS[1]);
    let area_span = reference_area(BOUNDS[3]) - south_area;
    let mut previous = BOUNDS[1];
    for index in 0_u32..=65_535 {
        let axis = index.strict_mul(65_537);
        let point = candidate(u64::from(axis) << 32_u32, BOUNDS)?;
        let expected = (f64::from(axis) + 0.5) / 4_294_967_296.0;
        let actual = (reference_area(point.latitude) - south_area) / area_span;
        assert!(point.latitude > previous, "The inverse is not monotone");
        assert!(
            (actual - expected).abs() < 5e-15,
            "The latitude is not at its equal-area cell center"
        );
        previous = point.latitude;
    }
    Ok(())
}
#[test]
fn orientation_retains_boundary_and_near_collinear_signs() {
    let unit = 2.0_f64.powi(-40);
    let rank = 1_073_741_824.0_f64;
    let origin = [127.0, 36.0];
    let end = [127.0 + rank * unit, 36.0 + (rank + 1.0) * unit];
    let point = [127.0 + (rank - 1.0) * unit, 36.0 + rank * unit];
    assert!(
        orientation(origin, end, point) > 0.0,
        "Exact positive orientation was rounded to zero"
    );
    assert!(
        orientation(origin, point, end) < 0.0,
        "Exact negative orientation was rounded to zero"
    );
    assert_eq!(
        orientation(origin, end, end),
        0.0,
        "A vertex must be on its edge"
    );
    assert_eq!(
        orientation([124.0, 32.0], [132.0, 40.0], [128.0, 36.0]),
        0.0,
        "A collinear point must be on its edge"
    );
}
#[test]
fn invalid_bounds_cannot_produce_a_candidate() {
    for bounds in [
        [124.0, 32.0, 124.0, 40.0],
        [124.0, 40.0, 132.0, 32.0],
        [124.0, 31.0, 132.0, 40.0],
        [124.0, 32.0, 181.0, 40.0],
        [f64::NAN, 32.0, 132.0, 40.0],
        [124.0, 32.0, f64::INFINITY, 40.0],
    ] {
        assert!(
            candidate(0, bounds).is_err(),
            "Invalid bounds produced a coordinate"
        );
    }
}
