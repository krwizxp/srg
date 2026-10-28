use super::sample;
use crate::diagnostic::Result;
#[test]
fn islands_and_border_region_remain_reachable() -> Result<()> {
    let cases = [
        (
            1712366629353805_u64,
            33.1_f64,
            33.123_f64,
            126.26_f64,
            126.275_f64,
        ),
        (
            13885604326397940595_u64,
            37.24_f64,
            37.245_f64,
            131.86_f64,
            131.868_f64,
        ),
        (
            15082953419755466149_u64,
            37.65_f64,
            37.7_f64,
            125.67_f64,
            125.74_f64,
        ),
        (
            16270525529238665242_u64,
            37.9_f64,
            38.0_f64,
            124.6_f64,
            124.74_f64,
        ),
        (
            16276029567443348942_u64,
            37.8_f64,
            37.86_f64,
            124.68_f64,
            124.74_f64,
        ),
        (
            18304039865364792788_u64,
            38.55_f64,
            38.615_f64,
            128.32_f64,
            128.39_f64,
        ),
        (
            18435746224877771189_u64,
            37.238_f64,
            37.242_f64,
            131.868_f64,
            131.874_f64,
        ),
    ];
    for (value, south, north, west, east) in cases {
        let coordinate = sample(value)?;
        assert!(
            (south..=north).contains(&coordinate.latitude)
                && (west..=east).contains(&coordinate.longitude),
            "Region fixture is outside its independently checked bounds"
        );
    }
    Ok(())
}
