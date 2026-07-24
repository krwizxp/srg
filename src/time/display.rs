use super::{
    CivilDate, KST_OFFSET_SECS, Result, ServerTime, TimeError,
    http_date::civil_from_days,
    util::{blend_rtt, parse_result_with_context},
};
use crate::{
    buffmt::{ByteCursor, digit_byte, two_digits},
    numeric::low_u8_from_u32,
};
use core::time::Duration;
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
const U32_FOUR_DIGIT_THRESHOLD: u32 = 10_000;
const U32_NEGATIVE_YEAR_SHORT_THRESHOLD: u32 = 1_000;
const U32_THREE_DIGIT_THRESHOLD: u32 = 100;
const UNIX_EPOCH_WEEKDAY_OFFSET_I64: i64 = 4;
impl ByteCursor<'_> {
    fn write_u32_2digits(&mut self, value: u32) -> IoResult<()> {
        let idx = usize::from(low_u8_from_u32(value));
        *self.take_array::<TWO_DIGIT_WIDTH>()? = two_digits(idx)?;
        Ok(())
    }
    fn write_u32_3digits(&mut self, value: u32) -> IoResult<()> {
        let hundreds = usize::from(low_u8_from_u32(value.div_euclid(U32_THREE_DIGIT_THRESHOLD)));
        let rem = usize::from(low_u8_from_u32(value.rem_euclid(U32_THREE_DIGIT_THRESHOLD)));
        let [tens, ones] = two_digits(rem)?;
        *self.take_array::<THREE_DIGIT_WIDTH>()? = [digit_byte(hundreds)?, tens, ones];
        Ok(())
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
                let [h0, h1] = two_digits(hi)?;
                let [l0, l1] = two_digits(lo)?;
                *self.take_array::<FOUR_DIGIT_WIDTH>()? = [h0, h1, l0, l1];
                return Ok(());
            }
            return self.write_u32_dec(year_value);
        }
        self.write_byte(b'-')?;
        let abs = year.unsigned_abs();
        if abs < U32_NEGATIVE_YEAR_SHORT_THRESHOLD {
            let hundreds = usize::from(low_u8_from_u32(abs.div_euclid(U32_THREE_DIGIT_THRESHOLD)));
            let rem = usize::from(low_u8_from_u32(abs.rem_euclid(U32_THREE_DIGIT_THRESHOLD)));
            let [tens, ones] = two_digits(rem)?;
            *self.take_array::<THREE_DIGIT_WIDTH>()? = [digit_byte(hundreds)?, tens, ones];
            return Ok(());
        }
        self.write_u32_dec(abs)
    }
}
impl ServerTime {
    pub(super) fn current_server_time_at(&self, now: Instant) -> SystemTime {
        let elapsed_since_anchor = now.saturating_duration_since(self.anchor_instant);
        let Some(server_time) = self.anchor_time.checked_add(elapsed_since_anchor) else {
            return self.anchor_time;
        };
        server_time
    }
    pub(super) const fn recalibrate_with_rtt(&self, new_rtt: Duration) -> Self {
        let smoothed_rtt = blend_rtt::<3>(self.baseline_rtt, new_rtt);
        Self {
            anchor_time: self.anchor_time,
            anchor_instant: self.anchor_instant,
            baseline_rtt: smoothed_rtt,
        }
    }
    pub(super) fn write_current_display_time_buf_at(
        &self,
        cur: &mut ByteCursor<'_>,
        now: Instant,
    ) -> Result<()> {
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
        } = civil_from_days(day_index);
        cur.write_year_padded4(year)?;
        cur.write_byte(b'-')?;
        cur.write_u32_2digits(month)?;
        cur.write_byte(b'-')?;
        cur.write_u32_2digits(day_of_month)?;
        cur.write_byte(b'(')?;
        cur.write_bytes(day_of_week_str.as_bytes())?;
        cur.write_bytes(b") ")?;
        cur.write_u32_2digits(hour)?;
        cur.write_byte(b':')?;
        cur.write_u32_2digits(minute)?;
        cur.write_byte(b':')?;
        cur.write_u32_2digits(second)?;
        cur.write_byte(b'.')?;
        cur.write_u32_3digits(millis)?;
        Ok(())
    }
}
