
#[test]
fn measure_unified_supplements_ten_thousand() -> Result<()> {
    use crate::random_data::RandomDataSet;
    let mut total = 0_usize;
    let mut korea = 0_usize;
    let mut first_with_none = 0_usize;
    let mut cache = BoundaryCache::default();
    let mut examples = Vec::new();
    let started = std::time::Instant::now();
    let mut saved_checksum = 0_u64;
    for record in 0..10_000_usize {
        let mut state = 0xD1B5_4A32_D192_ED03_u64 ^ u64::try_from(record).map_err(|_| "record index")?;
        let mut next_word = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            value ^ (value >> 31)
        };
        let num = next_word();
        let mut consumed = 0_usize;
        let mut next_supp = |reason: &'static str| -> Result<u64> {
            let value = next_word();
            consumed += 1;
            if reason == "대한민국 좌표 재추출" {
                korea += 1;
            }
            Ok(value)
        };
        let output = RandomDataSet { num_64: num, ..Default::default() }
            .populate(&mut next_supp, Some(&mut cache))?;
        if !output.kor_coords.latitude.is_finite() || !output.kor_coords.longitude.is_finite() {
            return Err("Invalid coordinate".into());
        }
        first_with_none += usize::from(consumed == 0);
        total += consumed;
        saved_checksum ^= output.num_64.rotate_left(u32::try_from(record % 63).map_err(|_| "rotate index")?);
        if record < 16 {
            examples.push(format!("{num},{consumed},{},{},{},{},{},{},{}",
                output.kor_coords.latitude,output.kor_coords.longitude,
                output.numeric_password,output.password.iter().map(|c|char::from(*c)).collect::<String>(),
                output.lotto_numbers.map(|v|v.to_string()).join("-"),
                output.lotto7_numbers.map(|v|v.to_string()).join("-"),
                output.euro_millions_main_numbers.map(|v|v.to_string()).join("-")));
        }
    }
    println!("SRG_INTEGRATED_METRICS records=10000 supp_total={total} supp_fresh_korea={korea} supp_zero={first_with_none} elapsed_ms={} checksum={saved_checksum}", started.elapsed().as_millis());
    for item in examples { println!("SRG_INTEGRATED_SAMPLE {item}"); }
    Ok(())
}
