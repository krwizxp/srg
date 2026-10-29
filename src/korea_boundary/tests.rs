use super::{
    BOUNDARY, BoundaryCache, BoundaryIndex, DATA, INDEX_SHA256, KOREA_SUPP_RETRY_LIMIT, PARTS,
    hash, remap_tile, sample,
};
use crate::{diagnostic::Result, geo_uniform::candidate, random_data::Coordinates};
const SEOUL: u64 = 15_033_626_022_739_801_206;
const WEST_SEA: u64 = 8_160_136_085_698_478_266;
const DOKDO: u64 = 13_962_275_557_818_425_490;
fn encode_tile_candidate(original: u64) -> Result<u64> {
    let vertical = original >> 32;
    let horizontal = original & 0xffff_ffff;
    let tile = ((vertical >> 23) << 9) | (horizontal >> 23);
    let index = BOUNDARY.as_ref().map_err(|error| format!("{error}"))?;
    let rank = index
        .sampler_tiles
        .binary_search(&u32::try_from(tile).map_err(|_| "Tile index overflow")?)
        .map_err(|_| "Fixture point omitted from tile cover")?;
    let lower = ((vertical & 0x7f_ffff) << 23) | (horizontal & 0x7f_ffff);
    Ok(((u64::try_from(rank).map_err(|_| "Tile rank overflow")? * 2) << 46) | lower)
}
#[test]
fn tile_mapping_preserves_original_centers() -> Result<()> {
    for original in [
        SEOUL,
        DOKDO,
        13_965_754_644_534_248_813,
        19_555_846_072_170_497,
        16_289_122_984_369_905_278,
    ] {
        let encoded = encode_tile_candidate(original)?;
        (remap_tile(
            encoded,
            BOUNDARY.as_ref().map_err(|error| format!("{error}"))?,
        )? == Some(original))
        .ok_or("Mapped grid point changed")?;
        (remap_tile(
            encoded | (1_u64 << 46),
            BOUNDARY.as_ref().map_err(|error| format!("{error}"))?,
        )? == Some(original))
        .ok_or("Second tile selection alias changed")?;
    }
    (remap_tile(
        u64::MAX,
        BOUNDARY.as_ref().map_err(|error| format!("{error}"))?,
    )?
    .is_none())
    .ok_or("Rejected selector was accepted")?;
    Ok(())
}
#[test]
fn hash_matches_independent_sha256_vectors() -> Result<()> {
    let cases: [(&[u8], &str); 3] = [
        (
            b"",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        ),
    ];
    for (input, expected) in cases {
        let actual = hash::digest(input)?
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        (actual == expected).ok_or("SHA-256 reference mismatch")?;
    }
    (hash::digest(DATA)? == INDEX_SHA256).ok_or("Embedded index digest mismatch")?;
    Ok(())
}
#[test]
fn original_geographic_probes_keep_covers_semantics() -> Result<()> {
    let index = BOUNDARY.as_ref().map_err(|error| format!("{error}"))?;
    let mut cache = BoundaryCache::default();
    let cases = [
        (126.978_f64, 37.5665_f64, true),
        (131.869_555_6, 37.240_777_8, true),
        (131.865, 37.241_833_3, true),
        (126.267, 33.118, true),
        (124.667, 37.95, true),
        (124.705, 37.825, true),
        (125.704, 37.665, true),
        (126.488, 37.746, true),
        (128.369, 38.585, true),
        (125.754, 39.039, false),
        (126.554, 37.971, false),
        (129.325, 34.205, false),
        (125.6, 35.5, false),
        (127.983_284_812_195_34, 38.302_453_273_850_496, true),
        (126.883_667_474_483_89, 38.109_516_351_129_066, false),
    ];
    for (longitude, latitude, expected) in cases {
        let point = Coordinates {
            latitude,
            longitude,
        };
        (index.covers(&point, &mut cache)? == expected)
            .ok_or("Original geographic probe changed")?;
    }
    Ok(())
}
#[test]
fn independent_grid_candidates_reach_mainland_and_islands() -> Result<()> {
    let cases = [
        (
            SEOUL,
            37.566_499_999_469_514_f64,
            126.977_999_999_834_62_f64,
        ),
        (DOKDO, 37.240_777_800_018_755, 131.869_555_599_714_91),
        (
            13_965_754_644_534_248_813,
            37.241_833_300_061_08,
            131.864_999_999_568_54,
        ),
        (
            19_555_846_072_170_497,
            33.118_000_000_556_97,
            126.266_999_999_916_93,
        ),
        (
            16_289_122_984_369_905_278,
            37.950_000_000_368_67,
            124.667_000_000_356_65,
        ),
        (
            15_880_603_076_929_503_247,
            37.825_000_000_389_69,
            124.705_000_000_563_93,
        ),
        (
            15_356_704_255_244_364_305,
            37.665_000_000_116_684,
            125.703_999_999_827_52,
        ),
        (
            18_353_816_716_112_344_413,
            38.584_999_999_565_12,
            128.369_000_000_791_46,
        ),
    ];
    for (value, latitude, longitude) in cases {
        let mut denied =
            |_| -> Result<u64> { Err("Accepted initial candidate requested entropy".into()) };
        let point = sample(encode_tile_candidate(value)?, &mut denied, None)?;
        ((point.latitude - latitude).abs() < 1e-12_f64
            && (point.longitude - longitude).abs() < 1e-12_f64)
            .ok_or("Independent high-precision candidate mismatch")?;
    }
    Ok(())
}
#[test]
fn rejection_consumes_one_new_complete_word_each_time() -> Result<()> {
    let accepted = encode_tile_candidate(DOKDO)?;
    for rejected in [0_usize, 1, 2, 3, 1024] {
        let mut calls = 0_usize;
        let mut provider = |reason: &'static str| -> Result<u64> {
            (reason == "대한민국 좌표 재추출").ok_or("Supplemental reason changed")?;
            calls = calls.strict_add(1);
            Ok(if calls == rejected {
                accepted
            } else {
                u64::MAX
            })
        };
        let initial = if rejected == 0 { accepted } else { u64::MAX };
        let actual = sample(initial, &mut provider, None)?;
        (calls == rejected).ok_or("Unexpected supplemental read count")?;
        ((actual.latitude - 37.240_777_800_018_755_f64).abs() < 1e-12_f64
            && (actual.longitude - 131.869_555_599_714_91_f64).abs() < 1e-12_f64)
            .ok_or("Supplemental candidate reused an earlier coordinate axis")?;
    }
    Ok(())
}
#[test]
fn retry_limit_and_provider_failures_are_errors() -> Result<()> {
    let mut calls = 0_usize;
    let mut repeated = |_| -> Result<u64> {
        calls = calls.strict_add(1);
        Ok(u64::MAX)
    };
    let failure = match sample(u64::MAX, &mut repeated, None) {
        Ok(_) => return Err("Rejected candidates became a coordinate".into()),
        Err(error) => error,
    };
    (calls == KOREA_SUPP_RETRY_LIMIT && failure.to_string().contains("시도 횟수 초과"))
        .ok_or("Supplemental retry bound changed")?;
    let mut unavailable = |_| -> Result<u64> { Err("fixture entropy source failed".into()) };
    let failure = match sample(u64::MAX, &mut unavailable, None) {
        Ok(_) => return Err("Provider failure became a coordinate".into()),
        Err(error) => error,
    };
    let message = failure.to_string();
    (message.contains("대한민국 좌표 보완 난수 공급 실패")
        && message.contains("fixture entropy source failed"))
    .ok_or("Supplemental provider context was lost")?;
    Ok(())
}
#[test]
fn corrupt_index_and_parts_cannot_be_interpreted_as_outside() -> Result<()> {
    let index = BoundaryIndex::parse(DATA, PARTS.len())?;
    for length in [0_usize, 7, 127, DATA.len().strict_sub(1)] {
        BoundaryIndex::parse(DATA.get(..length).ok_or("Fixture slice")?, PARTS.len())
            .is_err()
            .ok_or("Truncated root index accepted")?;
    }
    for offset in [0_usize, 8, 12, 16, 20, 36, 40, 44, 128] {
        let mut corrupted = DATA.to_vec();
        *corrupted.get_mut(offset).ok_or("Fixture offset")? ^= 0xff;
        BoundaryIndex::parse(&corrupted, PARTS.len())
            .is_err()
            .ok_or("Malformed root index accepted")?;
    }
    let part = index.parts.first().ok_or("Missing part metadata")?;
    let bytes = PARTS.first().ok_or("Missing part bytes")?;
    for offset in [0_usize, 8, 16, 24, 64, bytes.len().strict_sub(1)] {
        let mut corrupted = bytes.to_vec();
        *corrupted.get_mut(offset).ok_or("Fixture offset")? ^= 1;
        index
            .validate_part(part, &corrupted)
            .is_err()
            .ok_or("Corrupted part accepted")?;
    }
    Ok(())
}
#[test]
fn caches_preserve_all_fields_and_entropy_consumption() -> Result<()> {
    use crate::{
        output::{OutputTarget, format_data_into_buffer},
        random_data::RandomDataSet,
    };
    let mut cache = BoundaryCache::default();
    for value in [
        0_u64,
        1,
        u64::MAX,
        WEST_SEA,
        SEOUL,
        DOKDO,
        0x1234_5678_9ABC_DEF0,
    ] {
        let render = |storage: Option<&mut BoundaryCache>| -> Result<(Vec<u8>, Vec<&'static str>)> {
            let mut seed = 20_261_029_u64;
            let mut reasons = Vec::new();
            let mut next = |reason: &'static str| -> Result<u64> {
                reasons.push(reason);
                seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut word = seed;
                word = (word ^ (word >> 30_u32)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                word = (word ^ (word >> 27_u32)).wrapping_mul(0x94D0_49BB_1331_11EB);
                Ok(word ^ (word >> 31_u32))
            };
            let record = RandomDataSet {
                num_64: value,
                ..Default::default()
            }
            .populate(&mut next, storage)?;
            (record.num_64 == value).ok_or("Original num_64 changed")?;
            let mut buffer = [0_u8; crate::BUFFER_SIZE];
            let length = format_data_into_buffer(&record, &mut buffer, OutputTarget::File);
            Ok((
                buffer.get(..length).ok_or("Output length")?.to_vec(),
                reasons,
            ))
        };
        (render(None)? == render(Some(&mut cache))?)
            .ok_or("Cache changed a record or entropy consumption")?;
    }
    (cache.pages.len() <= super::CACHE_PAGES && cache.bytes <= super::CACHE_BYTES)
        .ok_or("Boundary cache exceeded its memory budget")?;
    Ok(())
}
#[test]
fn all_bit_extremes_are_finite_and_rejection_can_recover() -> Result<()> {
    let index = BOUNDARY.as_ref().map_err(|error| format!("{error}"))?;
    for value in [
        0,
        1,
        u64::from(u32::MAX),
        1_u64 << 32,
        u64::MAX.strict_sub(1),
        u64::MAX,
    ] {
        let point = candidate(value, index.bounds)?;
        (point.latitude.is_finite() && point.longitude.is_finite())
            .ok_or("Nonfinite extreme candidate")?;
        let mut provider = |_| -> Result<u64> { encode_tile_candidate(SEOUL) };
        let accepted = sample(value, &mut provider, None)?;
        index
            .covers(&accepted, &mut BoundaryCache::default())?
            .ok_or("Extreme input escaped the mask")?;
    }
    Ok(())
}
