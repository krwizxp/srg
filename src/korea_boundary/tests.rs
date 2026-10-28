use super::sample;
use crate::diagnostic::Result;
#[test]
fn islands_and_border_region_remain_reachable() -> Result<()> {
    let cases = [
        (
            9_362_441_748_123_u64,
            33.1_f64,
            33.123_f64,
            126.26_f64,
            126.275_f64,
        ),
        (
            76_025_477_720_643_476_u64,
            37.24_f64,
            37.245_f64,
            131.86_f64,
            131.868_f64,
        ),
        (
            82_582_034_390_531_324_u64,
            37.65_f64,
            37.7_f64,
            125.67_f64,
            125.74_f64,
        ),
        (
            89_085_194_964_887_242_u64,
            37.9_f64,
            38.0_f64,
            124.6_f64,
            124.74_f64,
        ),
        (
            89_115_296_921_869_377_u64,
            37.8_f64,
            37.86_f64,
            124.68_f64,
            124.74_f64,
        ),
        (
            100_221_274_423_774_957_u64,
            38.55_f64,
            38.615_f64,
            128.32_f64,
            128.39_f64,
        ),
        (
            100_941_457_114_691_630_u64,
            37.238_f64,
            37.242_f64,
            131.868_f64,
            131.874_f64,
        ),
    ];
    for (value, south, north, west, east) in cases {
        let coordinate = sample(value, None, &mut |_reason| Ok(0x4000_0000_0000_0000_u64))?;
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
#[test]
fn bounded_selection_and_entropy_failures_propagate() -> Result<()> {
    let mut requests = Vec::new();
    let result = sample(u64::MAX, None, &mut |reason| {
        requests.push(reason);
        Err("fixture exhausted".into())
    });
    assert!(
        result.is_err(),
        "Upper-tail selection must request fresh entropy"
    );
    assert_eq!(
        requests,
        ["대한민국 좌표 선택"],
        "The biased tail must be discarded before selecting a triangle"
    );
    let mut calls = 0_usize;
    let result = sample(9_362_441_748_123_u64, None, &mut |_reason| {
        calls = calls.strict_add(1);
        if calls == 3 {
            Err("fixture exhausted".into())
        } else {
            Ok(0x4000_0000_0000_0000_u64)
        }
    });
    assert!(result.is_err(), "Acceptance entropy is mandatory");
    assert_eq!(calls, 3, "No fallback may replace missing entropy");
    Ok(())
}
#[test]
fn rejection_has_a_bounded_failure_path() -> Result<()> {
    let mut calls = 0_usize;
    let result = sample(0, None, &mut |_reason| {
        let value = if calls.rem_euclid(4) == 2 {
            u64::MAX
        } else {
            0
        };
        calls = calls.strict_add(1);
        Ok(value)
    });
    assert!(
        result.is_err(),
        "Repeated rejection must fail instead of changing the target distribution"
    );
    assert_eq!(calls, 4096, "The retry limit must remain bounded");
    Ok(())
}

#[test]
fn uniform_v2_mapping_matches_cross_language_fixtures() -> Result<()> {
    let cases = [
        [
            10_727_398_953_992_791_243_u64,
            17_900_708_356_184_149_765_u64,
            3_u64,
            4_630_168_837_161_070_190_u64,
            4_638_698_482_775_343_857_u64,
        ],
        [
            7_087_192_844_234_247_593_u64,
            1_382_785_774_782_151_078_u64,
            3_u64,
            4_630_140_776_311_155_181_u64,
            4_638_742_710_156_609_048_u64,
        ],
        [
            3_728_682_670_307_502_467_u64,
            18_029_258_626_894_305_223_u64,
            3_u64,
            4_630_529_940_568_030_972_u64,
            4_638_708_880_918_254_851_u64,
        ],
        [
            9_960_039_744_275_199_692_u64,
            935_757_162_582_791_704_u64,
            3_u64,
            4_630_373_784_477_722_133_u64,
            4_638_596_881_566_767_484_u64,
        ],
        [
            17_733_815_040_234_978_873_u64,
            11_316_897_598_213_655_946_u64,
            3_u64,
            4_630_352_744_616_736_617_u64,
            4_638_715_472_067_773_464_u64,
        ],
        [
            389_525_306_419_595_668_u64,
            7_424_617_860_516_494_856_u64,
            3_u64,
            4_630_493_015_455_683_431_u64,
            4_638_708_428_661_678_519_u64,
        ],
        [
            2_634_856_691_471_878_960_u64,
            12_884_406_893_425_578_858_u64,
            3_u64,
            4_630_119_320_236_497_784_u64,
            4_638_667_208_340_042_250_u64,
        ],
        [
            8_543_309_311_135_394_936_u64,
            16_752_959_716_500_816_721_u64,
            3_u64,
            4_630_335_065_854_559_333_u64,
            4_638_724_953_427_301_844_u64,
        ],
        [
            14_625_120_864_271_834_030_u64,
            606_302_795_864_688_160_u64,
            3_u64,
            4_630_452_813_606_792_333_u64,
            4_638_728_216_386_492_998_u64,
        ],
        [
            5_158_475_398_815_792_914_u64,
            17_613_996_502_734_376_334_u64,
            3_u64,
            4_630_097_678_141_590_533_u64,
            4_638_623_982_653_596_032_u64,
        ],
        [
            17_013_617_386_603_485_178_u64,
            16_563_535_064_098_572_119_u64,
            3_u64,
            4_630_294_522_292_900_764_u64,
            4_638_741_834_518_810_195_u64,
        ],
        [
            12_155_121_884_830_108_539_u64,
            11_455_356_392_163_731_441_u64,
            3_u64,
            4_630_229_276_296_491_578_u64,
            4_638_677_818_121_164_526_u64,
        ],
    ];
    for [value, mut state, expected_calls, latitude, longitude] in cases {
        let mut calls = 0_u64;
        let point = sample(value, None, &mut |_reason| {
            calls = calls.strict_add(1);
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut word = state;
            word = (word ^ (word >> 30_u32)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            word = (word ^ (word >> 27_u32)).wrapping_mul(0x94D0_49BB_1331_11EB);
            Ok(word ^ (word >> 31_u32))
        })?;
        assert_eq!(
            [calls, point.latitude.to_bits(), point.longitude.to_bits()],
            [expected_calls, latitude, longitude],
            "Uniform v2 mapping or entropy consumption changed"
        );
    }
    Ok(())
}
