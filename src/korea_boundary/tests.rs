use super::sample;
use crate::diagnostic::Result;
#[test]
fn islands_and_border_region_remain_reachable() -> Result<()> {
    let cases = [
        (
            1_712_113_391_928_591_u64,
            33.1_f64,
            33.123_f64,
            126.26_f64,
            126.275_f64,
        ),
        (
            13_885_403_883_042_727_121_u64,
            37.24_f64,
            37.245_f64,
            131.86_f64,
            131.868_f64,
        ),
        (
            15_082_769_092_612_884_037_u64,
            37.65_f64,
            37.7_f64,
            125.67_f64,
            125.74_f64,
        ),
        (
            16_270_336_713_927_052_590_u64,
            37.9_f64,
            38.0_f64,
            124.6_f64,
            124.74_f64,
        ),
        (
            16_275_840_567_791_587_860_u64,
            37.8_f64,
            37.86_f64,
            124.68_f64,
            124.74_f64,
        ),
        (
            18_304_038_466_927_502_203_u64,
            38.55_f64,
            38.615_f64,
            128.32_f64,
            128.39_f64,
        ),
        (
            18_435_746_070_798_239_545_u64,
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
    for iteration in 0_u32..1_027 {
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
            let mut next = |reason: &'static str| -> Result<u64> {
                assert!(
                    !reason.starts_with("대한민국"),
                    "Domestic coordinates consumed supplemental entropy"
                );
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
#[test]
fn all_u64_endpoints_are_valid_without_an_entropy_provider() -> Result<()> {
    for value in [0, 1, u64::MAX.strict_sub(1), u64::MAX] {
        let point = sample(value, None)?;
        assert!(
            (33.0..39.0).contains(&point.latitude),
            "Latitude is outside Korea bounds"
        );
        assert!(
            (124.0..132.0).contains(&point.longitude),
            "Longitude is outside Korea bounds"
        );
    }
    Ok(())
}
#[test]
fn single_word_mapping_matches_cross_language_fixtures() -> Result<()> {
    let cases = [
        [
            0_u64,
            4_629_875_594_377_501_582_u64,
            4_638_581_400_102_764_471_u64,
        ],
        [
            1_u64,
            4_629_875_594_377_627_106_u64,
            4_638_581_400_102_626_171_u64,
        ],
        [
            18_446_744_073_709_551_615_u64,
            4_630_615_749_709_440_204_u64,
            4_638_722_744_866_156_479_u64,
        ],
        [
            1_712_113_391_928_591_u64,
            4_629_857_553_612_834_479_u64,
            4_638_585_762_187_892_353_u64,
        ],
        [
            13_885_403_883_042_727_121_u64,
            4_630_438_086_207_700_329_u64,
            4_638_843_572_429_526_086_u64,
        ],
        [
            18_435_746_070_798_239_545_u64,
            4_630_437_722_771_756_223_u64,
            4_638_843_710_896_310_596_u64,
        ],
    ];
    for [value, latitude, longitude] in cases {
        let point = sample(value, None)?;
        assert_eq!(
            [point.latitude.to_bits(), point.longitude.to_bits()],
            [latitude, longitude],
            "Single-word mapping changed"
        );
    }
    Ok(())
}
