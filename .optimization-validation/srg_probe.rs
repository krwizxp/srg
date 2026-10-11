fn next_fixed(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn main() -> Result<()> {
    let mode = env::args().nth(1).unwrap_or_default();

    if mode == "generate" {
        let start = std::time::Instant::now();
        let mut state = 20261011_u64;
        for _ in 0..262144 {
            let number = next_fixed(&mut state);
            let mut supplemental = number ^ 0xa0761d6478bd642f;
            std::hint::black_box(random_data::RandomDataSet { num_64: number, ..Default::default() }
                .populate(&mut |_| Ok(next_fixed(&mut supplemental)))?);
        }
        println!("{} {}", start.elapsed().as_nanos(), state);
        return Ok(());
    }
    let seeds = [0, 1, 9, 10, 99, 100, 255, 256, u64::MAX, 1 << 63];
    let mut records = Vec::with_capacity(4096);
    let mut state = 20261011_u64;
    for index in 0..4096 {
        let number = seeds.get(index).copied().unwrap_or_else(|| next_fixed(&mut state));
        let mut supplemental = number ^ 0xa0761d6478bd642f;
        let data = random_data::RandomDataSet { num_64: number, ..Default::default() }
            .populate(&mut |_| Ok(next_fixed(&mut supplemental)))?;
        records.push(data);
    }
    let mut buffer = [0; BUFFER_SIZE];
    if mode == "dump" {
        let mut out = io::stdout().lock();
        for data in &records {
            for target in [output::OutputTarget::File, output::OutputTarget::Console] {
                let len = output::format_data_into_buffer(data, &mut buffer, target);
                out.write_all(&buffer[..len])?;
                out.write_all(b"\0")?;
            }
        }
    } else {
        let start = std::time::Instant::now();
        let mut total = 0_usize;
        for _ in 0..64 {
            for data in &records {
                let len = output::format_data_into_buffer(std::hint::black_box(data), &mut buffer, output::OutputTarget::File);
                std::hint::black_box(&buffer[..len]);
                total = total.wrapping_add(len);
            }
        }
        println!("{} {}", start.elapsed().as_nanos(), total);
    }
    Ok(())
}
