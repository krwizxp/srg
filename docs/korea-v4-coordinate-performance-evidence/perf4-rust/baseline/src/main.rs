extern crate alloc;
mod time {
    #[derive(Debug)]
    pub struct TimeError;
    impl std::fmt::Display for TimeError {
        fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {output.write_str("unused")}
    }
    impl std::error::Error for TimeError {}
}
mod diagnostic;
mod random_data {pub struct Coordinates {pub latitude: f64, pub longitude: f64}}
mod geo_uniform;
mod boundary_inflate;
mod korea_boundary;
use std::io::{self, BufRead, Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cache = korea_boundary::BoundaryCache::default();
    let mut output = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let text = line?;
        let mut fields = text.split_whitespace();
        if fields.next() != Some("sample") {return Err("command".into());}
        let value = fields.next().ok_or("num")?.parse()?;
        let mut count = 0_usize;
        let mut supply = |reason: &'static str| -> diagnostic::Result<u64> {
            (reason == "대한민국 좌표 재추출").ok_or("supp reason changed")?;
            count += 1;
            fields.next().ok_or("supply exhausted")?.parse().map_err(|error| format!("{error}").into())
        };
        match korea_boundary::sample(value, &mut supply, Some(&mut cache)) {
            Ok(point) => writeln!(output, "{} {} {}", point.latitude.to_bits(), point.longitude.to_bits(), count)?,
            Err(error) => writeln!(output, "ERROR {} {}", count, error)?,
        }
    }
    output.flush()?;
    Ok(())
}
