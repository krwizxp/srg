use std::{io, sync::LazyLock};
pub const DIGITS: [u8; 10] = *b"0123456789";
pub static TWO_DIGITS: LazyLock<[[u8; 2]; 100]> = LazyLock::new(|| {
    let mut table = [[0_u8; 2]; 100];
    for (slot, value) in table.iter_mut().zip(0_u8..) {
        let Some(&tens) = DIGITS.get(usize::from(value.div_euclid(10))) else {
            continue;
        };
        let Some(&ones) = DIGITS.get(usize::from(value.rem_euclid(10))) else {
            continue;
        };
        *slot = [tens, ones];
    }
    table
});
pub struct ByteCursor<'buffer> {
    buf: &'buffer mut [u8],
    pos: usize,
}
impl<'buffer> ByteCursor<'buffer> {
    pub fn advance_by(&mut self, len: usize) -> io::Result<()> {
        self.pos = (checked_cursor_end(self.pos, len))?;
        Ok(())
    }
    pub const fn new(buf: &'buffer mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    pub const fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
    pub fn tail_mut(&mut self) -> &mut [u8] {
        let split_pos = self.pos.min(self.buf.len());
        let (_, tail) = self.buf.split_at_mut(split_pos);
        tail
    }
    pub fn take(&mut self, len: usize) -> io::Result<&mut [u8]> {
        if self.remaining() < len {
            return Err(write_zero_err());
        }
        let start = self.pos;
        let end = (checked_cursor_end(start, len))?;
        self.pos = end;
        self.buf.get_mut(start..end).ok_or_else(write_zero_err)
    }
    pub fn write_byte(&mut self, byte: u8) -> io::Result<()> {
        if self.remaining() < 1 {
            return Err(write_zero_err());
        }
        *(self.buf.get_mut(self.pos).ok_or_else(write_zero_err))? = byte;
        self.pos = (checked_cursor_end(self.pos, 1))?;
        Ok(())
    }
    pub fn write_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        let len = bytes.len();
        if self.remaining() < len {
            return Err(write_zero_err());
        }
        let end = (checked_cursor_end(self.pos, len))?;
        (self.buf.get_mut(self.pos..end).ok_or_else(write_zero_err))?.copy_from_slice(bytes);
        self.pos = end;
        Ok(())
    }
    pub const fn written_len(&self) -> usize {
        self.pos
    }
}
#[inline(never)]
#[cold]
pub fn write_zero_err() -> io::Error {
    io::Error::new(io::ErrorKind::WriteZero, "failed to write whole buffer")
}
fn checked_cursor_end(start: usize, len: usize) -> io::Result<usize> {
    start.checked_add(len).ok_or_else(write_zero_err)
}
