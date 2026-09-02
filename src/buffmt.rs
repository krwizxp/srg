use core::fmt::{self, NumBuffer};
use std::process;
static TWO_DIGITS: &[u8; 200] = b"00010203040506070809101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899";
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
pub(super) fn two_digits(value: u8) -> [u8; 2] {
    let start = usize::from(value).strict_mul(2);
    *TWO_DIGITS
        .get(start..)
        .and_then(|tail| tail.first_chunk())
        .unwrap_or_else(|| process::abort())
}
pub(super) const fn digit_byte(value: u8) -> u8 {
    b'0'.strict_add(value)
}
