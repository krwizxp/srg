use super::{HeadResponse, Result, error};
pub(super) const CLIENT: Client = Client;
pub(super) struct Client;
impl Client {
    pub(super) fn fetch_head(&self, _url: &str, context: &str) -> Result<HeadResponse> {
        Err(error(
            context,
            "이 플랫폼은 외부 크레이트 없이 native HTTP를 지원하지 않습니다.",
        ))
    }
}
