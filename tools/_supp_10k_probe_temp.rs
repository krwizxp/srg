#[cfg(target_arch = "x86_64")]
#[test]
fn measure_supp_for_ten_thousand_real_random_records() -> crate::diagnostic::Result<()> {
    use crate::{
        hardware_rng::{HardwareRandomSource, HardwareRng},
        korea_boundary::BoundaryCache,
        random_data::RandomDataSet,
    };
    use std::{collections::BTreeMap, hint::black_box, time::Instant};

    const RECORDS: usize = 10_000;
    const KOREA_REASON: &str = "\u{b300}\u{d55c}\u{bbfc}\u{ad6d} \u{c88c}\u{d45c} \u{c7ac}\u{cd94}\u{cd9c}";
    let rng = HardwareRng::new();
    let describe_source = |source: HardwareRandomSource| match source {
        HardwareRandomSource::RdSeed => "rdseed",
        HardwareRandomSource::RdRand => "rdrand",
        HardwareRandomSource::None => "none",
    };
    let initial_source = describe_source(rng.source());
    if initial_source == "none" {
        return Err("CPU hardware RNG is unavailable on measurement runner".into());
    }
    let started = Instant::now();
    let mut cache = BoundaryCache::default();
    let mut reason_counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut all_supp = 0_usize;
    let mut korea_supp = 0_usize;
    let mut korea_zero = 0_usize;
    let mut max_korea = 0_usize;
    let mut max_all = 0_usize;
    let mut korea_each = Vec::with_capacity(RECORDS);
    let mut total_each = Vec::with_capacity(RECORDS);
    let mut xor_checksum = 0_u64;
    for row in 0..RECORDS {
        let num64 = rng.next_u64()?;
        let mut used_supp = 0_usize;
        let mut used_korea = 0_usize;
        let mut next_supp = |reason: &'static str| -> crate::diagnostic::Result<u64> {
            let value = rng.next_u64()?;
            used_supp += 1;
            if reason == KOREA_REASON {
                used_korea += 1;
            }
            *reason_counts.entry(reason).or_insert(0) += 1;
            Ok(value)
        };
        let generated = RandomDataSet {
            num_64: num64,
            ..Default::default()
        }
        .populate(&mut next_supp, Some(&mut cache))?;
        if generated.num_64 != num64 || !generated.kor_coords.latitude.is_finite() {
            return Err("Generated record changed input or produced invalid latitude".into());
        }
        if row % 1000 == 0 {
            eprintln!("SRG_SUPP_PROGRESS completed={} supp_all={} supp_korea={}", row + 1, all_supp + used_supp, korea_supp + used_korea);
        }
        xor_checksum ^= num64.rotate_left((row % 63) as u32);
        black_box(generated);
        all_supp += used_supp;
        korea_supp += used_korea;
        korea_zero += usize::from(used_korea == 0);
        max_korea = max_korea.max(used_korea);
        max_all = max_all.max(used_supp);
        korea_each.push(used_korea);
        total_each.push(used_supp);
    }
    korea_each.sort_unstable();
    total_each.sort_unstable();
    let reasons = reason_counts
        .iter()
        .map(|(reason, count)| format!("\"{reason}\":{count}"))
        .collect::<Vec<_>>()
        .join(",");
    let elapsed = started.elapsed();
    let final_source = describe_source(rng.source());
    println!(
        "SRG_SUPP_10000_JSON={{\"records\":{RECORDS},\"rng_initial\":\"{initial_source}\",\"rng_final\":\"{final_source}\",\"initial_64bit_words\":{RECORDS},\"supp_all\":{all_supp},\"supp_korea\":{korea_supp},\"supp_other\":{},\"korea_zero\":{korea_zero},\"korea_max\":{max_korea},\"all_max\":{max_all},\"korea_p50\":{},\"korea_p95\":{},\"korea_p99\":{},\"all_p50\":{},\"all_p95\":{},\"all_p99\":{},\"total_hardware_u64\":{},\"elapsed_ms\":{},\"checksum_xor\":{xor_checksum},\"by_reason\":{{{reasons}}}}}",
        all_supp - korea_supp,
        korea_each[(RECORDS * 50 - 1) / 100],
        korea_each[(RECORDS * 95 - 1) / 100],
        korea_each[(RECORDS * 99 - 1) / 100],
        total_each[(RECORDS * 50 - 1) / 100],
        total_each[(RECORDS * 95 - 1) / 100],
        total_each[(RECORDS * 99 - 1) / 100],
        RECORDS + all_supp,
        elapsed.as_millis(),
    );
    Ok(())
}
