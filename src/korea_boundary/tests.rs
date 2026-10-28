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
        let coordinate = sample(value, None)?;
        assert!(
            (south..=north).contains(&coordinate.latitude)
                && (west..=east).contains(&coordinate.longitude),
            "Region fixture is outside its independently checked bounds"
        );
    }
    Ok(())
}
#[test]
fn cached_records_preserve_fields_and_supplement_consumption() -> Result<()> {
    use crate::{
        output::{OutputTarget, format_data_into_buffer},
        random_data::RandomDataSet,
    };
    let mut cache = super::BoundaryCache::default();
    let mut input = 20_261_028_u64;
    for iteration in 0_u32..1027 {
        input = input.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let value = match iteration {
            0 => 0,
            1 => 1,
            2 => u64::MAX,
            _ => input ^ input.rotate_left(23),
        };
        let render = |cache: Option<&mut super::BoundaryCache>| -> Result<(Vec<u8>, usize)> {
            let mut seed = value;
            let mut calls = 0_usize;
            let mut next = |_reason: &'static str| -> Result<u64> {
                calls = calls.strict_add(1);
                seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut value = seed;
                value = (value ^ (value >> 30_u32)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                value = (value ^ (value >> 27_u32)).wrapping_mul(0x94D0_49BB_1331_11EB);
                Ok(value ^ (value >> 31_u32))
            };
            let record = RandomDataSet {
                num_64: value,
                ..Default::default()
            }
            .populate(&mut next, cache)?;
            let mut buffer = [0_u8; crate::BUFFER_SIZE];
            let length = format_data_into_buffer(&record, &mut buffer, OutputTarget::File);
            Ok((
                buffer.get(..length).ok_or("검증 출력 길이 오류")?.to_vec(),
                calls,
            ))
        };
        (render(None)? == render(Some(&mut cache))?)
            .ok_or("캐시 사용 시 출력 또는 보완 난수 소비가 달라졌습니다.")?;
    }
    Ok(())
}
