use super::{buf_write_u8_dec, buf_write_u64_dec, prefix_slice, u8_dec_len};
use crate::{
    buffmt::{ByteCursor, copy_two_digits, digit_byte},
    diagnostic::{AppError, Result},
    numeric::low_u8_from_u64,
    numeric::low_u8_from_u128,
    IS_TERMINAL,
};
use core::time::Duration;
use std::io::Write as IoWrite;
const BAR_WIDTH: usize = 10;
const BAR_WIDTH_U64: u64 = 10;
const DECI_PER_MINUTE: u128 = 600;
const DECI_PER_SECOND: u128 = 10;
const ELAPSED_MILLIS_PER_DECI: u128 = 100;
const INVALID_TIME: &[u8; 7] = b"--:--.-";
const MAX_TIME_MINUTES: u128 = 99;
const PERCENT_SCALE_U64: u64 = 100;
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
        completed: u64,
        total: u64,
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
            let completed_scaled = u128::from(completed)
                .checked_mul(u128::from(PERCENT_SCALE_U64))
                .ok_or("ETA 분모 계산 실패")?;
            let eta_numerator = elapsed_millis
                .checked_mul(u128::from(remaining))
                .ok_or("ETA 분자 계산 실패")?;
            Some(eta_numerator.div_euclid(completed_scaled))
        };
        let elapsed_len = format_time_into(Some(elapsed_deci), &mut self.elapsed)?;
        let eta_len = format_time_into(eta_deci, &mut self.eta)?;
        let filled_u64 = scaled_progress_value(
            completed,
            total,
            BAR_WIDTH_U64,
            BAR_WIDTH_U64,
            "진행 막대 계산 실패",
        )?;
        let filled = usize::from(low_u8_from_u64(filled_u64.min(BAR_WIDTH_U64)));
        let percent_u64 = scaled_progress_value(
            completed,
            total,
            PERCENT_SCALE_U64,
            PERCENT_SCALE_U64,
            "진행률 계산 실패",
        )?;
        let percent = low_u8_from_u64(percent_u64.min(PERCENT_SCALE_U64));
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
        buf_write_u64_dec(&mut cur, completed)?;
        cur.write_byte(b'/')?;
        buf_write_u64_dec(&mut cur, total)?;
        cur.write_bytes(") | 소요: ".as_bytes())?;
        cur.write_bytes(prefix_slice(&self.elapsed, elapsed_len)?)?;
        cur.write_bytes(b" | ETA: ")?;
        cur.write_bytes(prefix_slice(&self.eta, eta_len)?)?;
        cur.write_bytes(b" \x1b[K")?;
        IoWrite::write_all(out, cur.written_slice()?)?;
        IoWrite::flush(out)?;
        Ok(())
    }
}
fn format_time_into(deci_seconds: Option<u128>, buf: &mut [u8; TIME_BUF_LEN]) -> Result<usize> {
    let Some(deci) = deci_seconds else {
        *buf = *INVALID_TIME;
        return Ok(TIME_BUF_LEN);
    };
    let minutes = usize::from(low_u8_from_u128(
        (deci.div_euclid(DECI_PER_MINUTE)).min(MAX_TIME_MINUTES),
    ));
    let sec_whole = usize::from(low_u8_from_u128(
        deci.div_euclid(DECI_PER_SECOND)
            .rem_euclid(SECONDS_PER_MINUTE_U128),
    ));
    let tenths = usize::from(low_u8_from_u128(deci.rem_euclid(DECI_PER_SECOND)));
    let mut minute_digits = [0_u8; 2];
    copy_two_digits(&mut minute_digits, minutes)?;
    let [minute_tens_value, minute_ones_value] = minute_digits;
    let mut second_digits = [0_u8; 2];
    copy_two_digits(&mut second_digits, sec_whole)?;
    let [second_tens_value, second_ones_value] = second_digits;
    *buf = [
        minute_tens_value,
        minute_ones_value,
        b':',
        second_tens_value,
        second_ones_value,
        b'.',
        digit_byte(tenths)?,
    ];
    Ok(TIME_BUF_LEN)
}
fn scaled_progress_value(
    completed: u64,
    total: u64,
    scale: u64,
    zero_total_value: u64,
    err_msg: &'static str,
) -> Result<u64> {
    if total == 0 {
        return Ok(zero_total_value);
    }
    let scaled = completed
        .checked_mul(scale)
        .ok_or_else(|| AppError::message(err_msg))?;
    Ok(scaled.div_euclid(total))
}
