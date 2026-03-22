use std::{io, sync::LazyLock};

pub const DIGITS: [u8; 10] = *b"0123456789";

fn make_two_digits_table() -> [[u8; 2]; 100] {
    let mut table = [[0_u8; 2]; 100];
    for (slot, value) in table.iter_mut().zip(0_u8..) {
        *slot = [b'0' + value / 10, b'0' + value % 10];
    }
    table
}

pub static TWO_DIGITS: LazyLock<[[u8; 2]; 100]> = LazyLock::new(make_two_digits_table);

#[inline(never)]
#[cold]
pub fn write_zero_err() -> io::Error {
    io::Error::new(io::ErrorKind::WriteZero, "failed to write whole buffer")
}

pub struct ByteCursor<'buffer> {
    buf: &'buffer mut [u8],
    pos: usize,
}

impl<'buffer> ByteCursor<'buffer> {
    pub const fn new(buf: &'buffer mut [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    pub const fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    pub const fn written_len(&self) -> usize {
        self.pos
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        let len = bytes.len();
        if self.remaining() < len {
            return Err(write_zero_err());
        }
        let end = self.pos + len;
        self.buf
            .get_mut(self.pos..end)
            .ok_or_else(write_zero_err)?
            .copy_from_slice(bytes);
        self.pos = end;
        Ok(())
    }

    pub fn write_byte(&mut self, b: u8) -> io::Result<()> {
        if self.remaining() < 1 {
            return Err(write_zero_err());
        }
        *self.buf.get_mut(self.pos).ok_or_else(write_zero_err)? = b;
        self.pos += 1;
        Ok(())
    }

    pub fn take(&mut self, len: usize) -> io::Result<&mut [u8]> {
        if self.remaining() < len {
            return Err(write_zero_err());
        }
        let start = self.pos;
        let end = start + len;
        self.pos = end;
        self.buf.get_mut(start..end).ok_or_else(write_zero_err)
    }

    pub fn tail_mut(&mut self) -> &mut [u8] {
        let split_pos = self.pos.min(self.buf.len());
        let (_, tail) = self.buf.split_at_mut(split_pos);
        tail
    }

    pub const fn advance_by(&mut self, len: usize) {
        self.pos += len;
    }
}
