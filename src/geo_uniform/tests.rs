use super::{density, determinant, propose, unit};
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
fn variates_are_open_and_near_collinear_signs_survive() {
    assert!(
        unit(0) > 0.0_f64 && unit(u64::MAX) < 1.0_f64,
        "Variates must exclude both endpoints"
    );
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
        propose(&vertices, 1, [0; 3]).is_err(),
        "A zero-area triangle must fail"
    );
    Ok(())
}
