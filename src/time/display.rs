use super::{
    CivilDate, KST_OFFSET_SECS, Result, ServerTime, TimeError,
    http_date::civil_from_days,
    util::{DAYS_PER_WEEK_I64, SECS_PER_DAY_I64, hms_from_second_of_day},
};
use crate::{
    buffmt::{ByteCursor, digit_byte, two_digits},
    numeric::{low_u8_from_u32, low_u32_from_u64},
};
use core::fmt::NumBuffer;
use std::{
    process,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
pub(super) const DISPLAY_LINE_BUF_LEN: usize = 80;
const DAY_OF_WEEK_KO: &[[u8; 3]] = "일월화수목금토".as_bytes().as_chunks::<3>().0;
const U32_FOUR_DIGIT_THRESHOLD: u32 = 10_000;
const U32_THREE_DIGIT_THRESHOLD: u32 = 100;
const UNIX_EPOCH_WEEKDAY_OFFSET_I64: i64 = 4;
impl ByteCursor<'_> {
    fn write_u32_2digits(&mut self, value: u32) {
        *self.take_array::<2>() = two_digits(low_u8_from_u32(value));
    }
    fn write_u32_3digits(&mut self, value: u32) {
        let hundreds = low_u8_from_u32(value.div_euclid(U32_THREE_DIGIT_THRESHOLD));
        let [tens, ones] = two_digits(low_u8_from_u32(value.rem_euclid(U32_THREE_DIGIT_THRESHOLD)));
        *self.take_array::<3>() = [digit_byte(hundreds), tens, ones];
    }
    fn write_year_padded4(&mut self, year: i32) {
        let year_value = year.cast_unsigned();
        if year_value < U32_FOUR_DIGIT_THRESHOLD {
            let hi = low_u8_from_u32(year_value.div_euclid(U32_THREE_DIGIT_THRESHOLD));
            let lo = low_u8_from_u32(year_value.rem_euclid(U32_THREE_DIGIT_THRESHOLD));
            let [h0, h1] = two_digits(hi);
            let [l0, l1] = two_digits(lo);
            *self.take_array::<4>() = [h0, h1, l0, l1];
            return;
        }
        let mut buffer = NumBuffer::new();
        self.write_bytes(year_value.format_into(&mut buffer).as_bytes());
    }
}
impl ServerTime {
    pub(super) fn current_server_time_at(&self, now: Instant) -> Result<SystemTime> {
        let elapsed_since_anchor = now
            .checked_duration_since(self.anchor_instant)
            .ok_or("현재 시각이 서버 시간 기준 시각보다 앞섭니다.")?;
        self.anchor_time
            .checked_add(elapsed_since_anchor)
            .ok_or_else(|| TimeError::parse("현재 서버 시간 계산 범위를 초과했습니다."))
    }
    pub(super) fn write_current_display_time_buf_at(
        &self,
        cur: &mut ByteCursor<'_>,
        now: Instant,
    ) -> Result<()> {
        let current_time = self.current_server_time_at(now)?;
        let since_epoch = current_time.duration_since(UNIX_EPOCH)?;
        let total_seconds =
            i64::try_from(since_epoch.as_secs()).unwrap_or_else(|_| process::abort());
        let total_seconds_kst = total_seconds.strict_add(KST_OFFSET_SECS);
        let millis = since_epoch.subsec_millis();
        let days_since_epoch = total_seconds_kst.div_euclid(SECS_PER_DAY_I64);
        let day_of_week_num = days_since_epoch
            .strict_add(UNIX_EPOCH_WEEKDAY_OFFSET_I64)
            .rem_euclid(DAYS_PER_WEEK_I64);
        let day_of_week_idx = usize::try_from(day_of_week_num).unwrap_or_else(|_| process::abort());
        let day_of_week_bytes = DAY_OF_WEEK_KO
            .get(day_of_week_idx)
            .copied()
            .unwrap_or_else(|| process::abort());
        let sec_of_day = low_u32_from_u64(
            total_seconds_kst
                .rem_euclid(SECS_PER_DAY_I64)
                .cast_unsigned(),
        );
        let (hour, minute, second) = hms_from_second_of_day(sec_of_day);
        let day_index = i32::try_from(days_since_epoch).unwrap_or_else(|_| process::abort());
        let CivilDate {
            day: day_of_month,
            month,
            year,
        } = civil_from_days(day_index);
        cur.write_year_padded4(year);
        cur.write_byte(b'-');
        cur.write_u32_2digits(month);
        cur.write_byte(b'-');
        cur.write_u32_2digits(day_of_month);
        cur.write_byte(b'(');
        *cur.take_array::<3>() = day_of_week_bytes;
        cur.write_bytes(b") ");
        cur.write_u32_2digits(hour);
        cur.write_byte(b':');
        cur.write_u32_2digits(minute);
        cur.write_byte(b':');
        cur.write_u32_2digits(second);
        cur.write_byte(b'.');
        cur.write_u32_3digits(millis);
        Ok(())
    }
}
