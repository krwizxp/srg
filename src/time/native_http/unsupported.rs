use super::{HeadResponse, ParseHttpDate, Result, error};
#[derive(Default)]
pub(super) struct Client;
impl Client {
    pub(super) fn fetch_head(
        &mut self,
        _url: &str,
        context: &str,
        _parse_http_date: ParseHttpDate,
    ) -> Result<HeadResponse> {
        Err(error(
            context,
            "이 플랫폼은 외부 크레이트 없이 native HTTP를 지원하지 않습니다.",
        ))
    }
}
