use alloc::fmt;
use core::range::Range;
use std::io;
const SINGLE_BYTE_WIDTH: usize = 1;
const TWO_DIGIT_TABLE_LEN: usize = 100;
const UTF8_MAX_CHAR_LEN: usize = 4;
pub const DIGITS: [u8; 10] = *b"0123456789";
const TWO_DIGITS: [[u8; 2]; TWO_DIGIT_TABLE_LEN] = [
    *b"00", *b"01", *b"02", *b"03", *b"04", *b"05", *b"06", *b"07", *b"08", *b"09", *b"10", *b"11",
    *b"12", *b"13", *b"14", *b"15", *b"16", *b"17", *b"18", *b"19", *b"20", *b"21", *b"22", *b"23",
    *b"24", *b"25", *b"26", *b"27", *b"28", *b"29", *b"30", *b"31", *b"32", *b"33", *b"34", *b"35",
    *b"36", *b"37", *b"38", *b"39", *b"40", *b"41", *b"42", *b"43", *b"44", *b"45", *b"46", *b"47",
    *b"48", *b"49", *b"50", *b"51", *b"52", *b"53", *b"54", *b"55", *b"56", *b"57", *b"58", *b"59",
    *b"60", *b"61", *b"62", *b"63", *b"64", *b"65", *b"66", *b"67", *b"68", *b"69", *b"70", *b"71",
    *b"72", *b"73", *b"74", *b"75", *b"76", *b"77", *b"78", *b"79", *b"80", *b"81", *b"82", *b"83",
    *b"84", *b"85", *b"86", *b"87", *b"88", *b"89", *b"90", *b"91", *b"92", *b"93", *b"94", *b"95",
    *b"96", *b"97", *b"98", *b"99",
];
pub struct ByteCursor<'buffer> {
    buf: &'buffer mut [u8],
    pos: usize,
}
impl<'buffer> ByteCursor<'buffer> {
    pub const fn new(buf: &'buffer mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    pub const fn remaining(&self) -> usize {
        match self.buf.len().checked_sub(self.pos) {
            Some(remaining) => remaining,
            None => 0,
        }
    }
    pub fn take(&mut self, len: usize) -> io::Result<&mut [u8]> {
        if self.remaining() < len {
            return Err(write_zero_err());
        }
        let start = self.pos;
        let end = start.checked_add(len).ok_or_else(write_zero_err)?;
        self.pos = end;
        self.buf
            .get_mut(Range { start, end })
            .ok_or_else(write_zero_err)
    }
    pub fn write_byte(&mut self, byte: u8) -> io::Result<()> {
        let Some(slot) = self.take(SINGLE_BYTE_WIDTH)?.first_mut() else {
            return Err(write_zero_err());
        };
        *slot = byte;
        Ok(())
    }
    pub fn write_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.take(bytes.len())?.copy_from_slice(bytes);
        Ok(())
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
        if self.write_bytes(s.as_bytes()).is_ok() {
            Ok(())
        } else {
            Err(fmt::Error)
        }
    }
}
pub fn copy_two_digits(target: &mut [u8; 2], value: usize) -> io::Result<()> {
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
