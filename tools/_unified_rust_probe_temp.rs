#[test]
fn verify_unified_shared_stream_ten_thousand() -> Result<()> {
    use crate::random_data::RandomDataSet;
    fn next_word(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
    let started = std::time::Instant::now();
    let mut cache = BoundaryCache::default();
    let mut total_supp = 0_usize;
    let mut zero_supp = 0_usize;
    let mut max_supp = 0_usize;
    let mut checksum = 0_u64;
    for row in 0..10_000_usize {
        let id = u64::try_from(row).map_err(|_| "Invalid row index")?;
        let mut state = 0xd1b5_4a32_d192_ed03_u64 ^ id;
        let num = next_word(&mut state);
        let mut used = 0_usize;
        let mut provider = |reason: &'static str| -> Result<u64> {
            (reason == "미완성 필드 보완")
                .ok_or("Supplement was requested outside the shared completion loop")?;
            used += 1;
            Ok(next_word(&mut state))
        };
        let record = RandomDataSet {
            num_64: num,
            ..Default::default()
        }
        .populate(&mut provider, Some(&mut cache))?;
        (used <= 1024).ok_or("Shared supplement hard cap exceeded")?;
        (record.kor_coords.latitude.is_finite() && record.kor_coords.longitude.is_finite())
            .ok_or("Invalid output coordinate")?;
        total_supp += used;
        zero_supp += usize::from(used == 0);
        max_supp = max_supp.max(used);
        checksum ^= num.rotate_left(u32::try_from(row % 63).map_err(|_| "Invalid rotation")?);
        if row < 128 {
            let columns = vec![
                num.to_string(),
                used.to_string(),
                record.numeric_password.to_string(),
                record.password.iter().map(|&byte| char::from(byte)).collect::<String>(),
                record.lotto_numbers.map(|v| v.to_string()).join("-"),
                record.lotto7_numbers.map(|v| v.to_string()).join("-"),
                record.euro_millions_main_numbers.map(|v| v.to_string()).join("-"),
                record.euro_millions_lucky_stars.map(|v| v.to_string()).join("-"),
                record.hangul_syllables.iter().copied().collect::<String>(),
                record.planet_number.to_string(),
                record.solar_system_index.to_string(),
                record.kor_coords.latitude.to_bits().to_string(),
                record.kor_coords.longitude.to_bits().to_string(),
                record.world_coords.latitude.to_bits().to_string(),
                record.world_coords.longitude.to_bits().to_string(),
                record.nms_portal_xxx.to_string(),
                record.nms_portal_yy.to_string(),
                record.nms_portal_zzz.to_string(),
                record.galaxy_x.to_string(),
                record.galaxy_y.to_string(),
                record.galaxy_z.to_string(),
            ];
            println!("SRG_UNIFIED_SAMPLE\t{}", columns.join("\t"));
        }
        std::hint::black_box(record);
    }
    println!(
        "SRG_UNIFIED_METRIC records=10000 supp_total={total_supp} supp_zero={zero_supp} supp_max={max_supp} checksum={checksum} elapsed_ms={}",
        started.elapsed().as_millis()
    );
    Ok(())
}
