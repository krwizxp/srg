extern crate alloc;
mod diagnostic {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
    pub struct AppError;
    impl AppError {
        pub fn context(message: &'static str, cause: impl std::error::Error + Send + Sync + 'static) -> Box<dyn std::error::Error + Send + Sync> {
            Box::new(std::io::Error::other(format!("{message}: {cause}")))
        }
    }
}
mod boundary_inflate;
mod geo_uniform;
mod korea_boundary;
mod hardware_rng;
mod numeric;
mod random_data;

use std::fmt::Write as _;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 10_000;
const REPS: usize = 5;
const SEED: u64 = 0x1234_5678_9abc_def0;
const SALT: u64 = 0xa5a5_a5a5_a5a5_a5a5;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
fn samples() -> Vec<u64> {
    let mut state = SEED;
    (0..N).map(|_| splitmix(&mut state)).collect()
}
fn coords(words: &[u64], cache: &mut korea_boundary::BoundaryCache, capture: bool) -> diagnostic::Result<(u64, usize, String)> {
    let mut checksum = 0_u64;
    let mut supplementary = 0_usize;
    let mut output = if capture {String::with_capacity(N * 42)} else {String::new()};
    for &word in words {
        let mut supp = word ^ SALT;
        let mut sampler = korea_boundary::SampleState::new(word, cache)?;
        let mut reads = 0_usize;
        while !sampler.is_complete() {
            sampler.offer_supplement(splitmix(&mut supp))?;
            reads += 1;
        }
        let point = sampler.finish()?;
        let latitude_bits = point.latitude.to_bits();
        let longitude_bits = point.longitude.to_bits();
        checksum = checksum.wrapping_add(latitude_bits ^ longitude_bits ^ u64::try_from(reads)?);
        supplementary += reads;
        if capture {
            writeln!(output, "{latitude_bits:016x} {longitude_bits:016x} {reads}")?;
        }
    }
    Ok((black_box(checksum), supplementary, output))
}
fn full(words: &[u64], cache: &mut korea_boundary::BoundaryCache, capture: bool) -> diagnostic::Result<(u64, usize, String)> {
    let mut checksum = 0_u64;
    let mut supplementary = 0_usize;
    let mut output = if capture {String::with_capacity(N * 88)} else {String::new()};
    for &word in words {
        let mut supp = word ^ SALT;
        let mut reads = 0_usize;
        let dataset = random_data::RandomDataSet {
            num_64: word,
            ..Default::default()
        }.populate(&mut |_| {
            reads += 1;
            Ok(splitmix(&mut supp))
        }, Some(cache))?;
        let lat = dataset.kor_coords.latitude.to_bits();
        let lon = dataset.kor_coords.longitude.to_bits();
        let wlat = dataset.world_coords.latitude.to_bits();
        let wlon = dataset.world_coords.longitude.to_bits();
        checksum = checksum.wrapping_add(lat ^ lon ^ wlat ^ wlon ^ u64::try_from(reads)?);
        supplementary += reads;
        if capture {
            writeln!(output, "{lat:016x} {lon:016x} {wlat:016x} {wlon:016x} {reads}")?;
        }
    }
    Ok((black_box(checksum), supplementary, output))
}
fn percentile(values: &[f64], frac: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered[((ordered.len() - 1) as f64 * frac).round() as usize]
}
fn measure(
    words: &[u64],
    cache: &mut korea_boundary::BoundaryCache,
    execute: impl Fn(&[u64], &mut korea_boundary::BoundaryCache, bool) -> diagnostic::Result<(u64, usize, String)>,
    output_path: &str,
    name: &str,
) -> diagnostic::Result<()> {
    let start = Instant::now();
    let (expected, supplemental, lines) = execute(words, cache, true)?;
    let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
    std::fs::write(output_path, lines)?;
    let (warmup, _, _) = execute(words, cache, false)?;
    (warmup == expected).then_some(()).ok_or("warmup checksum differs")?;
    let mut timings = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t0 = Instant::now();
        let (result, _, _) = execute(words, cache, false)?;
        (result == expected).then_some(()).ok_or("repeated checksum differs")?;
        timings.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    println!("RUST_BENCH {{\"kind\":\"{name}\",\"n\":{N},\"repetitions\":{REPS},\"coldMs\":{cold_ms:.6},\"medianMs\":{:.6},\"minMs\":{:.6},\"maxMs\":{:.6},\"supplementalReads\":{supplemental},\"timesMs\":{:?},\"checksum\":\"{expected:016x}\"}}",
        percentile(&timings, 0.5), percentile(&timings, 0.0), percentile(&timings, 1.0), timings);
    Ok(())
}
fn main() -> diagnostic::Result<()> {
    let words = samples();
    println!("RUST_ENV {{\"cpu\":\"{}\",\"parallelism\":{},\"target\":\"{}\",\"compiler\":\"{}\",\"features\":\"native release-matched\"}}",
        std::env::var("RUNNER_ARCH").unwrap_or_default(),
        std::thread::available_parallelism()?.get(),
        std::env::consts::ARCH,
        option_env!("RUSTC_VERSION").unwrap_or("rustc+1.99.1"));
    let mut coord_cache = korea_boundary::BoundaryCache::default();
    measure(&words, &mut coord_cache, coords, "validation/bench10000_rust_coords.txt", "coords")?;
    let mut full_cache = korea_boundary::BoundaryCache::default();
    measure(&words, &mut full_cache, full, "validation/bench10000_rust_full.txt", "full")?;
    Ok(())
}
