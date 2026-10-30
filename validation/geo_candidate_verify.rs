mod diagnostic {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
}
mod random_data {
    pub struct Coordinates {
        pub latitude: f64,
        pub longitude: f64,
    }
}
#[path = "../src/geo_uniform.rs"]
mod geo_uniform;
fn next_word(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}
fn main() {
    let mut state = 0x123456789abcdef0_u64;
    let bounds = [124.60971767856728, 33.11235575873201, 131.87278314710602, 38.61370930167485];
    for _ in 0..25000 {
        let word = next_word(&mut state);
        let coords = geo_uniform::candidate(word, bounds).unwrap();
        println!("{:016x} {:016x}", coords.latitude.to_bits(), coords.longitude.to_bits());
    }
    for word in [0, u64::MAX, 0xffff_ffff, 0xffff_ffff_0000_0000, 0x8000_0000_8000_0000] {
        let coords = geo_uniform::candidate(word, bounds).unwrap();
        println!("{:016x} {:016x}", coords.latitude.to_bits(), coords.longitude.to_bits());
    }
}
