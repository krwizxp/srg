mod diagnostic {
    pub type Result<T> = core::result::Result<T, Box<dyn core::error::Error>>;
}
mod hardware_rng {
    pub struct HardwareRng(pub u64);
    impl HardwareRng {
        pub fn next_u64(&self) -> crate::diagnostic::Result<u64> { Ok(self.0) }
    }
}
#[path = "../srg/src/random_number.rs"] mod random_number;
fn main() {
    if std::env::args().any(|arg| arg=="bench") {
        let mut output=Vec::with_capacity(2048);
        let start=std::time::Instant::now();
        for seed in 0..100_000_u64 {
            output.clear();
            random_number::generate_random_float(-100.0,-0.1,0,&mut output,&hardware_rng::HardwareRng(std::hint::black_box(seed ^ 0x123456789ABCDEF0))).unwrap();
            std::hint::black_box(&output);
        }
        println!("{}",start.elapsed().as_nanos());return;
    }
    let mut count = 0;
    for (min, max) in [(-1e308, -1e300), (-1e100, -1e90), (-100.0, -0.1), (0.0, 1.0), (1e-300, 1e-290), (-0.0, 0.0), (2.0, 2.0)] {
        for seed in [0, 1, u64::MAX / 2, u64::MAX - 1024, u64::MAX] {
            let mut output = Vec::new();
            random_number::generate_random_float(min, max, 0, &mut output, &hardware_rng::HardwareRng(seed)).unwrap();
            let text = String::from_utf8(output).unwrap();
            let result: f64 = text.rsplit_once(": ").unwrap().1.trim().parse().unwrap();
            if !(min..=max).contains(&result) { println!("OUT_OF_RANGE min={min:e} max={max:e} seed={seed} result={result:e}"); count += 1; }
        }
    }
    println!("out_of_range={count}");
}
