
fn main() {
    use std::hint::black_box;
    use std::time::Instant;
    let mut seed = 0x123456789abcdef0_u64;
    let mut next = |_: &'static str| -> Result<u64> {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        Ok(seed)
    };
    let mut samples = Vec::new();
    for i in 0..4096_u64 {
        let value = match i { 0 => 0, 1 => u64::MAX, 2 => i64::MAX as u64, _ => next("").unwrap() };
        samples.push(random_data::RandomDataSet {num_64:value, ..Default::default()}.populate(&mut next).unwrap());
    }
    let mut buffer = [0; BUFFER_SIZE];
    if env::args().any(|x|x=="dump") {
        for sample in &samples {
            for target in [output::OutputTarget::File, output::OutputTarget::Console] {
                let n = output::format_data_into_buffer(sample,&mut buffer,target);
                io::stdout().write_all(&buffer[..n]).unwrap();
            }
        }
        return;
    }
    for _ in 0..4 { for sample in &samples { black_box(output::format_data_into_buffer(black_box(sample),black_box(&mut buffer),output::OutputTarget::File)); } }
    let start=Instant::now();
    for _ in 0..120 { for sample in &samples { black_box(output::format_data_into_buffer(black_box(sample),black_box(&mut buffer),output::OutputTarget::File)); } }
    println!("{}",start.elapsed().as_nanos());
}
