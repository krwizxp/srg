use std::{env, process, time::Instant};

fn arguments() -> u32 {
    let mut args = env::args().skip(1);
    match (args.next(), args.next()) {
        (None, None) => 16,
        (Some(raw), None) => match raw.parse::<u32>() {
            Ok(bits @ (4 | 8 | 16)) => bits,
            _ => process::exit(2),
        },
        _ => process::exit(2),
    }
}

fn main() {
    let bits = arguments();
    let n = 1_u32 << bits;
    let started = Instant::now();
    let mut visited = 0_u64;
    let mut accepted = 0_u64;
    let mut mainland = 0_u64;
    let mut island_a = 0_u64;
    let mut island_b = 0_u64;
    let mut checksum = 0_u64;
    let mut square_checksum = 0_u64;
    let mut xor_checksum = 0_u64;

    for y in 0..n {
        let cy = 2 * y + 1;
        let prefix = u64::from(y) * u64::from(n);
        for x in 0..n {
            let cx = 2 * x + 1;
            let word = prefix + u64::from(x);
            let is_mainland = cx <= 3 * n / 2
                && cy <= 3 * n / 2
                && !(cx > n / 2 && cx < n && cy > n / 2 && cy < n);
            let is_island_a = cx >= 7 * n / 4 && cy >= 3 * n / 2;
            let is_island_b = cx >= 7 * n / 4
                && cx <= 15 * n / 8
                && cy >= n / 4
                && cy <= 3 * n / 8;
            let inside = is_mainland || is_island_a || is_island_b;
            visited += 1;
            accepted += u64::from(inside);
            mainland += u64::from(is_mainland);
            island_a += u64::from(is_island_a);
            island_b += u64::from(is_island_b);
            let selected = if inside { word } else { 0 };
            checksum = checksum.wrapping_add(selected);
            square_checksum = square_checksum.wrapping_add(selected.wrapping_mul(selected));
            xor_checksum ^= selected;
        }
    }

    let elapsed_seconds = started.elapsed().as_secs_f64();
    println!(
        r#"{{"axis_bits":{},"word_bits":{},"visited_candidates":{},"accepted_candidates":{},"rejected_candidates":{},"component_counts":[{},{},{}],"accepted_word_sum_mod_2_64":"{}","accepted_word_square_sum_mod_2_64":"{}","accepted_word_xor":"{}","elapsed_seconds":{:.9}}}"#,
        bits,
        bits * 2,
        visited,
        accepted,
        visited - accepted,
        mainland,
        island_a,
        island_b,
        checksum,
        square_checksum,
        xor_checksum,
        elapsed_seconds
    );
    if visited != u64::from(n) * u64::from(n) || mainland + island_a + island_b != accepted {
        process::exit(3);
    }
}
