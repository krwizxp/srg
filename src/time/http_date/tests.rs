use super::{HttpDate, TimeError};
use core::{str::FromStr as _, time::Duration};
use std::time::UNIX_EPOCH;

#[test]
fn date_formats_preserve_the_same_instant() -> Result<(), TimeError> {
    let expected = UNIX_EPOCH
        .checked_add(Duration::from_secs(784_111_777))
        .ok_or_else(|| TimeError::parse("Test timestamp overflow."))?;
    for raw in [
        "Sun, 06 Nov 1994 08:49:37 GMT",
        "Sunday, 06-Nov-94 08:49:37 GMT",
        "Sun Nov  6 08:49:37 1994",
        "Sun Nov 06 08:49:37 1994",
    ] {
        (HttpDate::from_str(raw)?.0 == expected)
            .ok_or_else(|| TimeError::parse(format!("HTTP date changed its instant: {raw}")))?;
    }
    Ok(())
}

#[test]
fn invalid_date_components_remain_rejected() -> Result<(), TimeError> {
    for raw in [
        "Sun, 6 Nov 1994 08:49:37 GMT",
        "Sun, 06 Nov 1994 08:49:37 UTC",
        "Sun, 06 Nov 1994 08:49:37 GMT trailing",
        "Mon, 06 Nov 1994 08:49:37 GMT",
        "Sun, 06 Nov 1994 24:49:37 GMT",
        "Sun, 06 Nov 1994 08:60:37 GMT",
        "Sun Nov 006 08:49:37 1994",
        "Sun Nov 6 08:49:37 1994 trailing",
        "Sun Nov 6 08:49:37 +994",
        "Sun XXX 6 08:49:37 1994",
        "Sun Nov 6 8:49:37 1994",
    ] {
        if HttpDate::from_str(raw).is_ok() {
            return Err(TimeError::parse("Malformed HTTP date was accepted."));
        }
    }
    Ok(())
}
