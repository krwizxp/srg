use core::fmt::{self, NumBuffer};
use std::process;
pub(super) struct ByteCursor<'buffer> {
    initial_len: usize,
    remaining: &'buffer mut [u8],
}
impl<'buffer> ByteCursor<'buffer> {
    pub(super) const fn new(remaining: &'buffer mut [u8]) -> Self {
        Self {
            initial_len: remaining.len(),
            remaining,
        }
    }
    pub(super) fn take(&mut self, len: usize) -> &mut [u8] {
        self.remaining
            .split_off_mut(..len)
            .unwrap_or_else(|| process::abort())
    }
    pub(super) fn take_array<const N: usize>(&mut self) -> &mut [u8; N] {
        self.take(N)
            .as_mut_array()
            .unwrap_or_else(|| process::abort())
    }
    pub(super) fn write_byte(&mut self, byte: u8) {
        self.write_bytes(&[byte]);
    }
    pub(super) fn write_bytes(&mut self, bytes: &[u8]) {
        self.take(bytes.len()).copy_from_slice(bytes);
    }
    pub(super) fn write_format(&mut self, args: fmt::Arguments<'_>) {
        if fmt::Write::write_fmt(self, args).is_err() {
            process::abort();
        }
    }
    pub(super) fn write_u32_dec(&mut self, value: u32) {
        let mut buffer = NumBuffer::new();
        self.write_bytes(value.format_into(&mut buffer).as_bytes());
    }
    pub(super) fn write_u64_dec(&mut self, value: u64) {
        let mut buffer = NumBuffer::new();
        self.write_bytes(value.format_into(&mut buffer).as_bytes());
    }
    pub(super) const fn written_len(self) -> usize {
        self.initial_len.strict_sub(self.remaining.len())
    }
}
impl fmt::Write for ByteCursor<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_bytes(s.as_bytes());
        Ok(())
    }
}
pub(super) const fn two_digits(value: u8) -> [u8; 2] {
    [
        digit_byte(value.div_euclid(10)),
        digit_byte(value.rem_euclid(10)),
    ]
}
pub(super) const fn digit_byte(value: u8) -> u8 {
    b'0'.strict_add(value)
}
