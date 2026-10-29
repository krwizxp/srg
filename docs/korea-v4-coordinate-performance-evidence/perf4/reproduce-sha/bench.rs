mod diagnostic {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
}
mod baseline {
    include!("baseline.rs");
    pub(super) fn state_block(mut state: [u32; 8], block: &[u8; 64]) -> [u32; 8] {
        compress(&mut state, block);
        state
    }
}
mod candidate {
    include!("candidate.rs");
    pub(super) fn state_block(mut state: [u32; 8], block: &[u8; 64]) -> [u32; 8] {
        compress(&mut state, block);
        state
    }
}
mod forced_scalar {
    include!("forced_scalar.rs");
    pub(super) fn state_block(mut state: [u32; 8], block: &[u8; 64]) -> [u32; 8] {
        compress(&mut state, block);
        state
    }
}
use std::hint::black_box;
use std::time::Instant;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn main() -> diagnostic::Result<()> {
    let arguments = std::env::args().collect::<Vec<_>>();
    let mode = arguments.get(1).ok_or("mode")?.as_str();
    let input = std::fs::read_to_string(arguments.get(2).ok_or("input")?)?;
    let mut assets = Vec::new();
    let mut vectors = 0_usize;
    let digest = match mode {
        "baseline" | "validate" => baseline::digest,
        "candidate" => candidate::digest,
        "forced_scalar" => forced_scalar::digest,
        _ => return Err("mode".into()),
    };
    for line in input.lines() {
        let fields = line.split('\t').collect::<Vec<_>>();
        (fields.len() == 4).ok_or("input row")?;
        let bytes = std::fs::read(fields[2])?;
        if mode == "validate" {
            let expected = fields[3];
            let original = baseline::digest(black_box(&bytes))?;
            let accelerated = candidate::digest(black_box(&bytes))?;
            let fallback = forced_scalar::digest(black_box(&bytes))?;
            (hex(&original) == expected && original == accelerated && original == fallback)
                .ok_or_else(|| format!("digest mismatch: {}", fields[1]))?;
        }
        if fields[0] == "vector" {
            (hex(&digest(black_box(&bytes))?) == fields[3])
                .ok_or_else(|| format!("vector mismatch: {}", fields[1]))?;
            vectors += 1;
        } else {
            assets.push((bytes, fields[3].to_string()));
        }
    }
    if mode == "validate" {
        let bytes = std::fs::read(arguments.get(3).ok_or("states")?)?;
        (bytes.len() % 96 == 0).ok_or("state record length")?;
        for (index, record) in bytes.chunks_exact(96).enumerate() {
            let mut state = [0_u32; 8];
            for (word, raw) in state.iter_mut().zip(record[..32].as_chunks::<4>().0) {
                *word = u32::from_le_bytes(*raw);
            }
            let block: &[u8; 64] = record[32..].try_into()?;
            let expected = baseline::state_block(black_box(state), black_box(block));
            (candidate::state_block(black_box(state), black_box(block)) == expected
                && forced_scalar::state_block(black_box(state), black_box(block)) == expected)
                .ok_or_else(|| format!("compression mismatch: {index}"))?;
        }
        println!(
            "{{\"mode\":\"validate\",\"asset_files\":{},\"vectors\":{},\"arbitrary_state_blocks\":{},\"digest_mismatches\":0,\"compression_mismatches\":0,\"runtime_sha\":{},\"runtime_sse2\":{}}}",
            assets.len(),
            vectors,
            bytes.len() / 96,
            std::is_x86_feature_detected!("sha"),
            std::is_x86_feature_detected!("sse2")
        );
        return Ok(());
    }
    let mut output = Vec::with_capacity(assets.len());
    let start = Instant::now();
    for (bytes, _) in &assets {
        output.push(digest(black_box(bytes))?);
    }
    let nanoseconds = start.elapsed().as_nanos();
    let mut byte_count = 0_usize;
    for ((bytes, expected), actual) in assets.iter().zip(&output) {
        (hex(actual) == *expected).ok_or("asset SHA-256 mismatch")?;
        byte_count += bytes.len();
    }
    println!(
        "{{\"mode\":\"{}\",\"asset_files\":{},\"asset_bytes\":{},\"vectors\":{},\"nanoseconds\":{},\"megabytes_per_second\":{},\"digest_mismatches\":0}}",
        mode,
        assets.len(),
        byte_count,
        vectors,
        nanoseconds,
        byte_count as f64 * 1000.0 / nanoseconds as f64
    );
    Ok(())
}
