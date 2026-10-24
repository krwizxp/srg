use crate::diagnostic::Result;
use std::cell::Cell;
pub struct HardwareRng {
    pub calls: Cell<usize>,
    state: Cell<u64>,
    values: Vec<u64>,
    failure: usize,
    constant: Option<u64>,
}
impl HardwareRng {
    pub fn new() -> Self {
        Self {
            calls: Cell::new(0),
            state: Cell::new(0x123456789abcdef0),
            values: std::env::var("MOCK_VALUES").ok().map_or_else(Vec::new, |s| s.split(',').filter(|x| !x.is_empty()).map(|x| u64::from_str_radix(x,16).unwrap()).collect()),
            failure: std::env::var("MOCK_FAIL").ok().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX),
            constant: std::env::var("MOCK_CONSTANT").ok().map(|s| u64::from_str_radix(&s,16).unwrap()),
        }
    }
    pub fn next_u64(&self) -> Result<u64> {
        let index = self.calls.get();
        self.calls.set(index+1);
        if index==self.failure {return Err("injected entropy failure".into());}
        if let Some(value)=self.constant {return Ok(value);}
        if let Some(&value)=self.values.get(index) {return Ok(value);}
        let value=self.state.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state.set(value);
        Ok(value)
    }
}
