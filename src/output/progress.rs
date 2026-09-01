use super::buf_write_u8_dec;
use crate::{
    buffmt::ByteCursor,
    diagnostic::Result,
    numeric::{low_u8_from_u128, u128_from_usize},
};
use core::{fmt::NumBuffer, time::Duration};
use std::io::Write as IoWrite;
const BAR_WIDTH: usize = 10;
const DECI_PER_MINUTE: u128 = 600;
const DECI_PER_SECOND: u128 = 10;
const ELAPSED_MILLIS_PER_DECI: u128 = 100;
const INVALID_TIME: &[u8; 7] = b"--:--.-";
const MAX_TIME_MINUTES: u128 = 99;
const PERCENT_SCALE: usize = 100;
const PERCENT_SCALE_U128: u128 = 100;
const SECONDS_PER_MINUTE_U128: u128 = 60;
const TIME_BUF_LEN: usize = 7;
pub(crate) struct ProgressBuffers {
    elapsed: [u8; TIME_BUF_LEN],
    eta: [u8; TIME_BUF_LEN],
    line: [u8; super::PROGRESS_LINE_BUF_LEN],
}
impl ProgressBuffers {
    pub(crate) const fn new() -> Self {
        Self {
            elapsed: [0_u8; TIME_BUF_LEN],
            eta: [0_u8; TIME_BUF_LEN],
            line: [0_u8; super::PROGRESS_LINE_BUF_LEN],
        }
    }
    pub(crate) fn print(
        &mut self,
        out: &mut dyn IoWrite,
        completed: usize,
        total: usize,
        elapsed: Duration,
    ) -> Result<()> {
        let elapsed_millis = elapsed.as_millis();
        let elapsed_deci = elapsed_millis.div_euclid(ELAPSED_MILLIS_PER_DECI);
        let eta_deci = if total == 0 || completed == total {
            Some(0)
        } else if completed == 0 {
            None
        } else {
            let completed_scaled = u128_from_usize(completed).strict_mul(PERCENT_SCALE_U128);
            let remaining_wide = u128_from_usize(total.strict_sub(completed));
            let eta_numerator = elapsed_millis.strict_mul(remaining_wide);
            Some(eta_numerator.div_euclid(completed_scaled))
        };
        format_time_into(Some(elapsed_deci), &mut self.elapsed);
        format_time_into(eta_deci, &mut self.eta);
        let percent_value = if total == 0 {
            PERCENT_SCALE
        } else {
            completed
                .strict_mul(PERCENT_SCALE)
                .div_euclid(total)
                .min(PERCENT_SCALE)
        };
        let filled = percent_value.div_euclid(PERCENT_SCALE.div_euclid(BAR_WIDTH));
        let [percent, ..] = percent_value.to_le_bytes();
        let mut cur = ByteCursor::new(&mut self.line);
        cur.write_byte(b'\r');
        cur.write_byte(b'[');
        for _ in 0..filled {
            cur.write_bytes("█".as_bytes());
        }
        for _ in filled..BAR_WIDTH {
            cur.write_byte(b' ');
        }
        cur.write_byte(b']');
        cur.write_byte(b' ');
        if percent < 100 {
            cur.write_byte(b' ');
        }
        if percent < 10 {
            cur.write_byte(b' ');
        }
        buf_write_u8_dec(&mut cur, percent);
        cur.write_byte(b'%');
        cur.write_bytes(b" (");
        let mut count_buffer = NumBuffer::new();
        cur.write_bytes(completed.format_into(&mut count_buffer).as_bytes());
        cur.write_byte(b'/');
        cur.write_bytes(total.format_into(&mut count_buffer).as_bytes());
        cur.write_bytes(") | 소요: ".as_bytes());
        cur.write_bytes(&self.elapsed);
        cur.write_bytes(b" | ETA: ");
        cur.write_bytes(&self.eta);
        cur.write_bytes(b" \x1b[K");
        let written_len = cur.written_len();
        IoWrite::write_all(out, self.line.split_at(written_len).0)?;
        IoWrite::flush(out).map_err(Into::into)
    }
}
fn format_time_into(deci_seconds: Option<u128>, buf: &mut [u8; TIME_BUF_LEN]) {
    let Some(deci) = deci_seconds else {
        *buf = *INVALID_TIME;
        return;
    };
    let minutes = low_u8_from_u128((deci.div_euclid(DECI_PER_MINUTE)).min(MAX_TIME_MINUTES));
    let sec_whole = low_u8_from_u128(
        deci.div_euclid(DECI_PER_SECOND)
            .rem_euclid(SECONDS_PER_MINUTE_U128),
    );
    let tenths = low_u8_from_u128(deci.rem_euclid(DECI_PER_SECOND));
    *buf = [
        b'0'.strict_add(minutes.div_euclid(10)),
        b'0'.strict_add(minutes.rem_euclid(10)),
        b':',
        b'0'.strict_add(sec_whole.div_euclid(10)),
        b'0'.strict_add(sec_whole.rem_euclid(10)),
        b'.',
        b'0'.strict_add(tenths),
    ];
}
