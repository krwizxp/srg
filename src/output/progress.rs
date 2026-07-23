use super::{buf_write_u8_dec, u8_dec_len};
use crate::{
    buffmt::ByteCursor,
    diagnostic::Result,
    numeric::{low_u8_from_u128, low_u8_from_usize, u128_from_usize},
    IS_TERMINAL,
};
use core::{fmt::Write as _, time::Duration};
use std::io::Write as IoWrite;
const BAR_WIDTH: usize = 10;
const DECI_PER_MINUTE: u128 = 600;
const DECI_PER_SECOND: u128 = 10;
const ELAPSED_MILLIS_PER_DECI: u128 = 100;
const INVALID_TIME: &[u8; 7] = b"--:--.-";
const MAX_TIME_MINUTES: u128 = 99;
const PERCENT_SCALE: usize = 100;
const PERCENT_SCALE_U128: u128 = 100;
const PERCENT_WIDTH: usize = 3;
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
        if !*IS_TERMINAL {
            return Ok(());
        }
        let elapsed_millis = elapsed.as_millis();
        let elapsed_deci = elapsed_millis.div_euclid(ELAPSED_MILLIS_PER_DECI);
        let eta_deci = if total == 0 || completed >= total {
            Some(0)
        } else if completed == 0 {
            None
        } else {
            let remaining = total.saturating_sub(completed);
            let completed_scaled = u128_from_usize(completed)
                .saturating_mul(PERCENT_SCALE_U128);
            let remaining_wide = u128_from_usize(remaining);
            let eta_numerator = elapsed_millis
                .checked_mul(remaining_wide)
                .ok_or("ETA 분자 계산 실패")?;
            Some(eta_numerator.div_euclid(completed_scaled))
        };
        format_time_into(Some(elapsed_deci), &mut self.elapsed);
        format_time_into(eta_deci, &mut self.eta);
        let filled_value = scaled_progress_value(completed, total, BAR_WIDTH, BAR_WIDTH);
        let filled = usize::from(low_u8_from_usize(filled_value.min(BAR_WIDTH)));
        let percent_value =
            scaled_progress_value(completed, total, PERCENT_SCALE, PERCENT_SCALE);
        let percent = low_u8_from_usize(percent_value.min(PERCENT_SCALE));
        let mut cur = ByteCursor::new(&mut self.line);
        cur.write_byte(b'\r')?;
        cur.write_byte(b'[')?;
        for _ in 0..filled {
            cur.write_bytes("█".as_bytes())?;
        }
        for _ in filled..BAR_WIDTH {
            cur.write_byte(b' ')?;
        }
        cur.write_byte(b']')?;
        cur.write_byte(b' ')?;
        let padding = PERCENT_WIDTH.saturating_sub(u8_dec_len(percent));
        for _ in 0..padding {
            cur.write_byte(b' ')?;
        }
        buf_write_u8_dec(&mut cur, percent)?;
        cur.write_byte(b'%')?;
        cur.write_bytes(b" (")?;
        write!(&mut cur, "{completed}/{total}")?;
        cur.write_bytes(") | 소요: ".as_bytes())?;
        cur.write_bytes(&self.elapsed)?;
        cur.write_bytes(b" | ETA: ")?;
        cur.write_bytes(&self.eta)?;
        cur.write_bytes(b" \x1b[K")?;
        IoWrite::write_all(out, cur.written_slice()?)?;
        IoWrite::flush(out)?;
        Ok(())
    }
}
fn format_time_into(deci_seconds: Option<u128>, buf: &mut [u8; TIME_BUF_LEN]) {
    let Some(deci) = deci_seconds else {
        *buf = *INVALID_TIME;
        return;
    };
    let minutes = low_u8_from_u128(
        (deci.div_euclid(DECI_PER_MINUTE)).min(MAX_TIME_MINUTES),
    );
    let sec_whole = low_u8_from_u128(
        deci.div_euclid(DECI_PER_SECOND)
            .rem_euclid(SECONDS_PER_MINUTE_U128),
    );
    let tenths = low_u8_from_u128(deci.rem_euclid(DECI_PER_SECOND));
    *buf = [
        b'0'.saturating_add(minutes.div_euclid(10)),
        b'0'.saturating_add(minutes.rem_euclid(10)),
        b':',
        b'0'.saturating_add(sec_whole.div_euclid(10)),
        b'0'.saturating_add(sec_whole.rem_euclid(10)),
        b'.',
        b'0'.saturating_add(tenths),
    ];
}
const fn scaled_progress_value(
    completed: usize,
    total: usize,
    scale: usize,
    zero_total_value: usize,
) -> usize {
    if total == 0 {
        return zero_total_value;
    }
    completed.saturating_mul(scale).div_euclid(total)
}
