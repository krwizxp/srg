use core::{
    fmt::{self, NumBuffer},
    range::Range,
};
use std::process;
pub(super) struct ByteCursor<'buffer> {
    buf: &'buffer mut [u8],
    pos: usize,
}
impl<'buffer> ByteCursor<'buffer> {
    pub(super) const fn new(buf: &'buffer mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    pub(super) fn take(&mut self, len: usize) -> &mut [u8] {
        let start = self.pos;
        let end = start.strict_add(len);
        let slice = self
            .buf
            .get_mut(Range { start, end })
            .unwrap_or_else(|| process::abort());
        self.pos = end;
        slice
    }
    pub(super) fn take_array<const N: usize>(&mut self) -> &mut [u8; N] {
        self.take(N).try_into().unwrap_or_else(|_| process::abort())
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
    pub(super) const fn written_slice(&self) -> &[u8] {
        self.buf.split_at(self.pos).0
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
