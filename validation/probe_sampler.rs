extern crate alloc;
mod diagnostic {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
}
mod random_data {
    pub struct Coordinates {
        pub latitude: f64,
        pub longitude: f64,
    }
}
#[path = "../src/boundary_inflate.rs"]
mod boundary_inflate;
#[path = "../src/geo_uniform.rs"]
mod geo_uniform;
#[path = "../src/korea_boundary.rs"]
mod korea_boundary;

use std::io::{self, Write as _};

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn main() -> diagnostic::Result<()> {
    let mut writer = io::BufWriter::new(io::stdout().lock());
    let mut seed = 0x123456789abcdef0_u64;
    let mut cache = korea_boundary::BoundaryCache::default();
    for _ in 0..750 {
        let word = splitmix(&mut seed);
        let mut supplemental = word ^ 0xa5a5a5a5a5a5a5a5;
        let mut sampler = korea_boundary::SampleState::new(word, &mut cache)?;
        let mut attempts = 0usize;
        while !sampler.is_complete() {
            sampler.offer_supplement(splitmix(&mut supplemental))?;
            attempts += 1;
        }
        let point = sampler.finish()?;
        writeln!(
            writer,
            "{:016x} {:016x} {}",
            point.latitude.to_bits(),
            point.longitude.to_bits(),
            attempts
        )?;
    }
    writer.flush()?;
    Ok(())
}
