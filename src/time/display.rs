use super::{
    CivilDate, KST_OFFSET_SECS, Result, ServerTime, TimeError,
    http_date::civil_from_days,
    util::{blend_weighted_nanos, parse_result_with_context},
};
use crate::{
    buffmt::{
        ByteCursor, copy_two_digits as buffmt_copy_two_digits, digit_byte as buffmt_digit_byte,
        write_zero_err,
    },
    numeric::low_u8_from_u32,
};
use core::{range::Range, time::Duration};
use std::{
    io::Result as IoResult,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
const DAY_SECONDS_I64: i64 = 86_400;
const DAYS_PER_WEEK_I64: i64 = 7;
pub(super) const DISPLAY_LINE_BUF_LEN: usize = 80;
const DAY_OF_WEEK_KO: [&str; 7] = ["일", "월", "화", "수", "목", "금", "토"];
const FOUR_DIGIT_WIDTH: usize = 4;
const HOUR_SECONDS_I64: i64 = 3_600;
const MINUTE_SECONDS_I64: i64 = 60;
const THREE_DIGIT_WIDTH: usize = 3;
const TWO_DIGIT_WIDTH: usize = 2;
const U32_DEC_BUF_LEN: usize = 10;
const U32_FOUR_DIGIT_THRESHOLD: u32 = 10_000;
const U32_NEGATIVE_YEAR_SHORT_THRESHOLD: u32 = 1_000;
const U32_THREE_DIGIT_THRESHOLD: u32 = 100;
const U32_TWO_DIGIT_THRESHOLD: u32 = 10;
const UNIX_EPOCH_WEEKDAY_OFFSET_I64: i64 = 4;
struct DisplayableTime {
    day_of_month: u32,
    day_of_week_str: &'static str,
    hour: u32,
    millis: u32,
    minute: u32,
    month: u32,
    second: u32,
    year: i32,
}
pub(super) struct SliceCursor<'buffer> {
    pub inner: ByteCursor<'buffer>,
}
impl SliceCursor<'_> {
    fn checked_sub_index(value: usize, amount: usize) -> IoResult<usize> {
        value.checked_sub(amount).ok_or_else(write_zero_err)
    }
    fn copy_two_digits(target: &mut [u8; TWO_DIGIT_WIDTH], value: usize) -> IoResult<()> {
        buffmt_copy_two_digits(target, value)
    }
    fn digit_byte(index: usize) -> IoResult<u8> {
        buffmt_digit_byte(index)
    }
    fn range_two_digits(slice: &mut [u8], start: usize) -> IoResult<&mut [u8; TWO_DIGIT_WIDTH]> {
        let end = start
            .checked_add(TWO_DIGIT_WIDTH)
            .ok_or_else(write_zero_err)?;
        slice
            .get_mut(Range { start, end })
            .and_then(|digits| digits.first_chunk_mut::<TWO_DIGIT_WIDTH>())
            .ok_or_else(write_zero_err)
    }
    fn write_byte(&mut self, byte: u8) -> IoResult<()> {
        self.inner.write_byte(byte)
    }
    pub(super) fn write_bytes(&mut self, bytes: &[u8]) -> IoResult<()> {
        self.inner.write_bytes(bytes)
    }
    fn write_u32_2digits(&mut self, value: u32) -> IoResult<()> {
        let idx = usize::from(low_u8_from_u32(value));
        let digits_slice = self.inner.take(TWO_DIGIT_WIDTH)?;
        let Some(digits) = digits_slice.first_chunk_mut::<TWO_DIGIT_WIDTH>() else {
            return Err(write_zero_err());
        };
        Self::copy_two_digits(digits, idx)?;
        Ok(())
    }
    fn write_u32_3digits(&mut self, value: u32) -> IoResult<()> {
        let hundreds = usize::from(low_u8_from_u32(value.div_euclid(U32_THREE_DIGIT_THRESHOLD)));
        let rem = usize::from(low_u8_from_u32(value.rem_euclid(U32_THREE_DIGIT_THRESHOLD)));
        let head = self.inner.take(THREE_DIGIT_WIDTH)?;
        let Some((digit_slot, remaining_tail)) = head.split_first_mut() else {
            return Err(write_zero_err());
        };
        *digit_slot = Self::digit_byte(hundreds)?;
        let Some(remaining_digits) = remaining_tail.first_chunk_mut::<TWO_DIGIT_WIDTH>() else {
            return Err(write_zero_err());
        };
        Self::copy_two_digits(remaining_digits, rem)?;
        Ok(())
    }
    fn write_u32_dec(&mut self, mut n: u32) -> IoResult<()> {
        let mut tmp = [0_u8; U32_DEC_BUF_LEN];
        let mut i = tmp.len();
        while n >= U32_THREE_DIGIT_THRESHOLD {
            let rem = usize::from(low_u8_from_u32(n.rem_euclid(U32_THREE_DIGIT_THRESHOLD)));
            n /= U32_THREE_DIGIT_THRESHOLD;
            i = Self::checked_sub_index(i, 2)?;
            Self::copy_two_digits(Self::range_two_digits(&mut tmp, i)?, rem)?;
        }
        if n >= U32_TWO_DIGIT_THRESHOLD {
            let rem = usize::from(low_u8_from_u32(n));
            i = Self::checked_sub_index(i, TWO_DIGIT_WIDTH)?;
            Self::copy_two_digits(Self::range_two_digits(&mut tmp, i)?, rem)?;
        } else {
            i = Self::checked_sub_index(i, 1)?;
            let digit = usize::from(low_u8_from_u32(n));
            *tmp.get_mut(i).ok_or_else(write_zero_err)? = Self::digit_byte(digit)?;
        }
        let Some(suffix) = tmp.get(i..) else {
            return Err(write_zero_err());
        };
        self.write_bytes(suffix)
    }
    fn write_year_padded4(&mut self, year: i32) -> IoResult<()> {
        if year >= 0_i32 {
            let year_value = year.cast_unsigned();
            if year_value < U32_FOUR_DIGIT_THRESHOLD {
                let hi = usize::from(low_u8_from_u32(
                    year_value.div_euclid(U32_THREE_DIGIT_THRESHOLD),
                ));
                let lo = usize::from(low_u8_from_u32(
                    year_value.rem_euclid(U32_THREE_DIGIT_THRESHOLD),
                ));
                let head = self.inner.take(FOUR_DIGIT_WIDTH)?;
                let Some((hi_digits, lo_tail)) = head.split_first_chunk_mut::<TWO_DIGIT_WIDTH>()
                else {
                    return Err(write_zero_err());
                };
                let Some(lo_digits) = lo_tail.first_chunk_mut::<TWO_DIGIT_WIDTH>() else {
                    return Err(write_zero_err());
                };
                Self::copy_two_digits(hi_digits, hi)?;
                Self::copy_two_digits(lo_digits, lo)?;
                return Ok(());
            }
            return self.write_u32_dec(year_value);
        }
        self.write_byte(b'-')?;
        let abs = year.unsigned_abs();
        if abs < U32_NEGATIVE_YEAR_SHORT_THRESHOLD {
            let hundreds = usize::from(low_u8_from_u32(abs.div_euclid(U32_THREE_DIGIT_THRESHOLD)));
            let rem = usize::from(low_u8_from_u32(abs.rem_euclid(U32_THREE_DIGIT_THRESHOLD)));
            let head = self.inner.take(THREE_DIGIT_WIDTH)?;
            let Some((digit_slot, remaining_tail)) = head.split_first_mut() else {
                return Err(write_zero_err());
            };
            *digit_slot = Self::digit_byte(hundreds)?;
            let Some(remaining_digits) = remaining_tail.first_chunk_mut::<TWO_DIGIT_WIDTH>() else {
                return Err(write_zero_err());
            };
            Self::copy_two_digits(remaining_digits, rem)?;
            return Ok(());
        }
        self.write_u32_dec(abs)
    }
    pub(super) fn written_slice(&self) -> IoResult<&[u8]> {
        self.inner.written_slice()
    }
}
impl ServerTime {
    fn calculate_display_time_at(&self, now: Instant) -> Result<DisplayableTime> {
        let current_time = self.current_server_time_at(now);
        let since_epoch = current_time.duration_since(UNIX_EPOCH)?;
        let total_seconds = parse_result_with_context(
            i64::try_from(since_epoch.as_secs()),
            "초 계산 중 범위 오류",
        )?;
        let total_seconds_kst = total_seconds
            .checked_add(KST_OFFSET_SECS)
            .ok_or_else(|| TimeError::parse("초 계산 중 범위 오류"))?;
        let millis = since_epoch.subsec_millis();
        let days_since_epoch = total_seconds_kst.div_euclid(DAY_SECONDS_I64);
        let day_of_week_num = days_since_epoch
            .checked_add(UNIX_EPOCH_WEEKDAY_OFFSET_I64)
            .ok_or_else(|| TimeError::parse("요일 계산 중 범위 오류"))?
            .rem_euclid(DAYS_PER_WEEK_I64);
        let day_of_week_idx =
            parse_result_with_context(usize::try_from(day_of_week_num), "요일 계산 중 범위 오류")?;
        let day_of_week_str = DAY_OF_WEEK_KO
            .get(day_of_week_idx)
            .copied()
            .ok_or_else(|| TimeError::parse("요일 계산 중 범위 오류"))?;
        let sec_of_day = total_seconds_kst.rem_euclid(DAY_SECONDS_I64);
        let hour = parse_result_with_context(
            u32::try_from(sec_of_day.div_euclid(HOUR_SECONDS_I64)),
            "시 계산 중 범위 오류",
        )?;
        let minute = parse_result_with_context(
            u32::try_from(
                sec_of_day
                    .rem_euclid(HOUR_SECONDS_I64)
                    .div_euclid(MINUTE_SECONDS_I64),
            ),
            "분 계산 중 범위 오류",
        )?;
        let second = parse_result_with_context(
            u32::try_from(sec_of_day.rem_euclid(MINUTE_SECONDS_I64)),
            "초 계산 중 범위 오류",
        )?;
        let day_index =
            parse_result_with_context(i32::try_from(days_since_epoch), "일자 계산 중 범위 오류")?;
        let CivilDate {
            day: day_of_month,
            month,
            year,
        } = civil_from_days(day_index).ok_or_else(|| TimeError::parse("일자 계산 중 범위 오류"))?;
        Ok(DisplayableTime {
            day_of_month,
            day_of_week_str,
            hour,
            millis,
            minute,
            month,
            second,
            year,
        })
    }
    pub(super) fn current_server_time_at(&self, now: Instant) -> SystemTime {
        let elapsed_since_anchor = now.duration_since(self.anchor_instant);
        let Some(server_time) = self.anchor_time.checked_add(elapsed_since_anchor) else {
            return self.anchor_time;
        };
        server_time
    }
    pub(super) fn recalibrate_with_rtt(&self, new_rtt: Duration) -> Self {
        let smoothed_rtt_nanos =
            blend_weighted_nanos(self.baseline_rtt.as_nanos(), new_rtt.as_nanos(), 3, 10);
        let smoothed_rtt = Duration::from_nanos_u128(smoothed_rtt_nanos);
        Self {
            anchor_time: self.anchor_time,
            anchor_instant: self.anchor_instant,
            baseline_rtt: smoothed_rtt,
        }
    }
    pub(super) fn write_current_display_time_buf_at(
        &self,
        cur: &mut SliceCursor<'_>,
        now: Instant,
    ) -> Result<()> {
        let dt = self.calculate_display_time_at(now)?;
        cur.write_year_padded4(dt.year).map_err(TimeError::from)?;
        cur.write_byte(b'-').map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.month).map_err(TimeError::from)?;
        cur.write_byte(b'-').map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.day_of_month)
            .map_err(TimeError::from)?;
        cur.write_byte(b'(').map_err(TimeError::from)?;
        cur.write_bytes(dt.day_of_week_str.as_bytes())
            .map_err(TimeError::from)?;
        cur.write_bytes(b") ").map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.hour).map_err(TimeError::from)?;
        cur.write_byte(b':').map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.minute).map_err(TimeError::from)?;
        cur.write_byte(b':').map_err(TimeError::from)?;
        cur.write_u32_2digits(dt.second).map_err(TimeError::from)?;
        cur.write_byte(b'.').map_err(TimeError::from)?;
        cur.write_u32_3digits(dt.millis).map_err(TimeError::from)?;
        Ok(())
    }
}
