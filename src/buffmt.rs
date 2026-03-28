use alloc::fmt;
use std::{io, sync::LazyLock};
const DECIMAL_BASE_U8: u8 = 10;
const SINGLE_BYTE_WIDTH: usize = 1;
const TWO_DIGIT_TABLE_LEN: usize = 100;
const UTF8_MAX_CHAR_LEN: usize = 4;
pub const DIGITS: [u8; 10] = *b"0123456789";
pub static TWO_DIGITS: LazyLock<[[u8; 2]; TWO_DIGIT_TABLE_LEN]> = LazyLock::new(|| {
    let mut table = [[0_u8; 2]; TWO_DIGIT_TABLE_LEN];
    for (slot, value) in table.iter_mut().zip(0_u8..) {
        let Some(&tens) = DIGITS.get(usize::from(value.div_euclid(DECIMAL_BASE_U8))) else {
            continue;
        };
        let Some(&ones) = DIGITS.get(usize::from(value.rem_euclid(DECIMAL_BASE_U8))) else {
            continue;
        };
        *slot = [tens, ones];
    }
    table
});
pub struct ByteCursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}
impl<'a> ByteCursor<'a> {
    pub const fn new(buf: &'a mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    pub const fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
    pub fn take(&mut self, len: usize) -> io::Result<&mut [u8]> {
        if self.remaining() < len {
            return Err(write_zero_err());
        }
        let start = self.pos;
        let end = checked_cursor_end(start, len)?;
        self.pos = end;
        self.buf.get_mut(start..end).ok_or_else(write_zero_err)
    }
    pub fn write_byte(&mut self, byte: u8) -> io::Result<()> {
        if self.remaining() < SINGLE_BYTE_WIDTH {
            return Err(write_zero_err());
        }
        *(self.buf.get_mut(self.pos).ok_or_else(write_zero_err))? = byte;
        self.pos = checked_cursor_end(self.pos, SINGLE_BYTE_WIDTH)?;
        Ok(())
    }
    pub fn write_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        let len = bytes.len();
        if self.remaining() < len {
            return Err(write_zero_err());
        }
        let end = checked_cursor_end(self.pos, len)?;
        self.buf
            .get_mut(self.pos..end)
            .ok_or_else(write_zero_err)?
            .copy_from_slice(bytes);
        self.pos = end;
        Ok(())
    }
    pub const fn written_len(&self) -> usize {
        self.pos
    }
    pub fn written_slice(&self) -> io::Result<&[u8]> {
        self.buf.get(..self.pos).ok_or_else(write_zero_err)
    }
}
impl fmt::Write for ByteCursor<'_> {
    fn write_char(&mut self, c: char) -> fmt::Result {
        let mut encoded = [0_u8; UTF8_MAX_CHAR_LEN];
        self.write_str(c.encode_utf8(&mut encoded))
    }
    fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> fmt::Result {
        fmt::write(self, args)
    }
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_bytes(s.as_bytes())
            .map_err(|_write_err| fmt::Error)
    }
}
pub fn copy_two_digits(target: &mut [u8], value: usize) -> io::Result<()> {
    if target.len() != 2 {
        return Err(write_zero_err());
    }
    target.copy_from_slice(TWO_DIGITS.get(value).ok_or_else(write_zero_err)?);
    Ok(())
}
pub fn digit_byte(index: usize) -> io::Result<u8> {
    DIGITS.get(index).copied().ok_or_else(write_zero_err)
}
#[inline(never)]
#[cold]
pub fn write_zero_err() -> io::Error {
    io::Error::new(io::ErrorKind::WriteZero, "failed to write whole buffer")
}
fn checked_cursor_end(start: usize, len: usize) -> io::Result<usize> {
    start.checked_add(len).ok_or_else(write_zero_err)
}
