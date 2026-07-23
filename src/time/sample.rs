use super::{NetworkContext, Result, TimeSample, UrlScheme};
pub(super) fn fetch_server_time_sample(net_ctx: &mut NetworkContext) -> Result<TimeSample> {
    let context = match net_ctx.host.scheme {
        UrlScheme::Http => "HTTP",
        UrlScheme::Https => "HTTPS",
    };
    net_ctx.native_http.fetch_head(&net_ctx.host, context)
}
