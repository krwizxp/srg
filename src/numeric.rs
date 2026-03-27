pub const fn low_u8_from_u32(value: u32) -> u8 {
    let mut out = 0_u8;
    if value & 0x01 != 0 {
        out |= 0x01;
    }
    if value & 0x02 != 0 {
        out |= 0x02;
    }
    if value & 0x04 != 0 {
        out |= 0x04;
    }
    if value & 0x08 != 0 {
        out |= 0x08;
    }
    if value & 0x10 != 0 {
        out |= 0x10;
    }
    if value & 0x20 != 0 {
        out |= 0x20;
    }
    if value & 0x40 != 0 {
        out |= 0x40;
    }
    if value & 0x80 != 0 {
        out |= 0x80;
    }
    out
}
pub const fn low_u8_from_u64(value: u64) -> u8 {
    let mut out = 0_u8;
    if value & 0x01 != 0 {
        out |= 0x01;
    }
    if value & 0x02 != 0 {
        out |= 0x02;
    }
    if value & 0x04 != 0 {
        out |= 0x04;
    }
    if value & 0x08 != 0 {
        out |= 0x08;
    }
    if value & 0x10 != 0 {
        out |= 0x10;
    }
    if value & 0x20 != 0 {
        out |= 0x20;
    }
    if value & 0x40 != 0 {
        out |= 0x40;
    }
    if value & 0x80 != 0 {
        out |= 0x80;
    }
    out
}
pub const fn low_u8_from_u128(value: u128) -> u8 {
    let mut out = 0_u8;
    if value & 0x01 != 0 {
        out |= 0x01;
    }
    if value & 0x02 != 0 {
        out |= 0x02;
    }
    if value & 0x04 != 0 {
        out |= 0x04;
    }
    if value & 0x08 != 0 {
        out |= 0x08;
    }
    if value & 0x10 != 0 {
        out |= 0x10;
    }
    if value & 0x20 != 0 {
        out |= 0x20;
    }
    if value & 0x40 != 0 {
        out |= 0x40;
    }
    if value & 0x80 != 0 {
        out |= 0x80;
    }
    out
}
pub const fn low_u16_from_u64(value: u64) -> u16 {
    let mut out = 0_u16;
    if value & 0x0001 != 0 {
        out |= 0x0001;
    }
    if value & 0x0002 != 0 {
        out |= 0x0002;
    }
    if value & 0x0004 != 0 {
        out |= 0x0004;
    }
    if value & 0x0008 != 0 {
        out |= 0x0008;
    }
    if value & 0x0010 != 0 {
        out |= 0x0010;
    }
    if value & 0x0020 != 0 {
        out |= 0x0020;
    }
    if value & 0x0040 != 0 {
        out |= 0x0040;
    }
    if value & 0x0080 != 0 {
        out |= 0x0080;
    }
    if value & 0x0100 != 0 {
        out |= 0x0100;
    }
    if value & 0x0200 != 0 {
        out |= 0x0200;
    }
    if value & 0x0400 != 0 {
        out |= 0x0400;
    }
    if value & 0x0800 != 0 {
        out |= 0x0800;
    }
    if value & 0x1000 != 0 {
        out |= 0x1000;
    }
    if value & 0x2000 != 0 {
        out |= 0x2000;
    }
    if value & 0x4000 != 0 {
        out |= 0x4000;
    }
    if value & 0x8000 != 0 {
        out |= 0x8000;
    }
    out
}
