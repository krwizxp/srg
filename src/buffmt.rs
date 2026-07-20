use crate::numeric::low_u8_from_u32;
use core::{fmt, range::Range};
use std::io;
const TWO_DIGIT_WIDTH: usize = 2;
const U32_DEC_BUF_LEN: usize = 10;
const U32_THREE_DIGIT_THRESHOLD: u32 = 100;
const U32_TWO_DIGIT_THRESHOLD: u32 = 10;
const DIGITS: [u8; 10] = *b"0123456789";
pub(super) struct ByteCursor<'buffer> {
    buf: &'buffer mut [u8],
    pos: usize,
}
impl<'buffer> ByteCursor<'buffer> {
    pub(super) const fn new(buf: &'buffer mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    pub(super) fn take(&mut self, len: usize) -> io::Result<&mut [u8]> {
        let start = self.pos;
        let end = start.checked_add(len).ok_or_else(write_zero_err)?;
        let slice = self
            .buf
            .get_mut(Range { start, end })
            .ok_or_else(write_zero_err)?;
        self.pos = end;
        Ok(slice)
    }
    pub(super) fn write_byte(&mut self, byte: u8) -> io::Result<()> {
        self.write_bytes(&[byte])
    }
    pub(super) fn write_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.take(bytes.len())?.copy_from_slice(bytes);
        Ok(())
    }
    pub(super) fn write_u32_dec(&mut self, mut value: u32) -> io::Result<()> {
        let mut buffer = [0_u8; U32_DEC_BUF_LEN];
        let mut index = buffer.len();
        while value >= U32_THREE_DIGIT_THRESHOLD {
            let remainder =
                usize::from(low_u8_from_u32(value.rem_euclid(U32_THREE_DIGIT_THRESHOLD)));
            value = value.div_euclid(U32_THREE_DIGIT_THRESHOLD);
            index = index
                .checked_sub(TWO_DIGIT_WIDTH)
                .ok_or_else(write_zero_err)?;
            let digits = buffer
                .get_mut(index..)
                .and_then(|tail| tail.first_chunk_mut::<TWO_DIGIT_WIDTH>())
                .ok_or_else(write_zero_err)?;
            copy_two_digits(digits, remainder)?;
        }
        if value >= U32_TWO_DIGIT_THRESHOLD {
            index = index
                .checked_sub(TWO_DIGIT_WIDTH)
                .ok_or_else(write_zero_err)?;
            let digits = buffer
                .get_mut(index..)
                .and_then(|tail| tail.first_chunk_mut::<TWO_DIGIT_WIDTH>())
                .ok_or_else(write_zero_err)?;
            copy_two_digits(digits, usize::from(low_u8_from_u32(value)))?;
        } else {
            index = index.checked_sub(1).ok_or_else(write_zero_err)?;
            let slot = buffer.get_mut(index).ok_or_else(write_zero_err)?;
            *slot = digit_byte(usize::from(low_u8_from_u32(value)))?;
        }
        self.write_bytes(buffer.get(index..).ok_or_else(write_zero_err)?)
    }
    pub(super) fn written_slice(&self) -> io::Result<&[u8]> {
        self.buf.get(..self.pos).ok_or_else(write_zero_err)
    }
}
impl fmt::Write for ByteCursor<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.write_bytes(s.as_bytes()).is_ok() {
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}
pub(super) fn copy_two_digits(target: &mut [u8; 2], value: usize) -> io::Result<()> {
    *target = [
        digit_byte(value.div_euclid(10))?,
        digit_byte(value.rem_euclid(10))?,
    ];
    Ok(())
}
pub(super) fn digit_byte(index: usize) -> io::Result<u8> {
    DIGITS.get(index).copied().ok_or_else(write_zero_err)
}
#[inline(never)]
#[cold]
pub(super) fn write_zero_err() -> io::Error {
    io::Error::new(io::ErrorKind::WriteZero, "failed to write whole buffer")
}
