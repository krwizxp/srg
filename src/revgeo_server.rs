use crate::diagnostic::Result;
#[cfg(target_arch = "x86_64")]
use crate::hardware_rng::{HardwareRandomSource, HardwareRng};
use core::time::Duration;
use std::{
    io::{self, Read as _, Write},
    net::{Ipv4Addr, TcpListener, TcpStream},
};

const MAX_COUNT: usize = 10_000;
const MAX_HEADER_BYTES: usize = 8192;
const BACKEND_MARKER: &str = "<meta name=\"srg-rng-backend\" content=\"/api/random\">";

struct WebRng {
    #[cfg(target_arch = "x86_64")]
    hardware: HardwareRng,
}

impl WebRng {
    fn new() -> Self {
        Self {
            #[cfg(target_arch = "x86_64")]
            hardware: HardwareRng::new(),
        }
    }

    fn generate(&self, count: usize) -> Result<String> {
        let mut values = Vec::with_capacity(count);
        #[cfg(target_arch = "x86_64")]
        let source = {
            let initial = self.hardware.source();
            for _ in 0..count {
                values.push(self.hardware.next_u64()?.to_string());
            }
            match (initial, self.hardware.source()) {
                (HardwareRandomSource::RdSeed, HardwareRandomSource::RdSeed) => "rdseed",
                (HardwareRandomSource::RdSeed, HardwareRandomSource::RdRand) => "rdseed+rdrand",
                (_, HardwareRandomSource::RdRand) => "rdrand",
                _ => return Err("RDSEED/RDRAND를 사용할 수 없습니다.".into()),
            }
        };
        #[cfg(not(target_arch = "x86_64"))]
        let source = {
            let mut bytes = vec![0_u8; count.strict_mul(8)];
            crate::system_rng::fill(&mut bytes)?;
            for chunk in bytes.chunks_exact(8) {
                let mut value = [0_u8; 8];
                value.copy_from_slice(chunk);
                values.push(u64::from_le_bytes(value).to_string());
            }
            "os-csprng"
        };
        // Decimal strings preserve all 64 bits when decoded by JavaScript.
        Ok(format!("{{\"source\":\"{source}\",\"values\":[\"{}\"]}}", values.join("\",\"")))
    }
}

pub(super) fn serve(port: u16, out: &mut dyn Write, err: &mut dyn Write) -> Result<()> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))?;
    let authority = listener.local_addr()?.to_string();
    let rng = WebRng::new();
    writeln!(out, "http://{authority}/revgeo.html\n브라우저에서 위 주소를 여세요. 종료: Ctrl+C")?;
    out.flush()?;
    for connection in listener.incoming() {
        let mut stream = connection?;
        if let Err(error) = handle_connection(&mut stream, &authority, &rng) {
            writeln!(err, "revgeo 요청 실패: {error}")?;
        }
    }
    Ok(())
}

fn respond(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) -> io::Result<()> {
    write!(stream, "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nCross-Origin-Resource-Policy: same-origin\r\nConnection: close\r\n\r\n{body}", body.len())
}

fn handle_connection(stream: &mut TcpStream, authority: &str, rng: &WebRng) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let mut header = Vec::new();
    // Read only the header. Both memory and read time are bounded.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !header.ends_with(b"\r\n\r\n") {
        if header.len() >= MAX_HEADER_BYTES || std::time::Instant::now() >= deadline {
            return respond(stream, "431 Request Header Fields Too Large", "text/plain", "요청 헤더가 너무 큽니다.").map_err(Into::into);
        }
        let mut byte = [0_u8; 1];
        stream.read_exact(&mut byte)?;
        header.push(byte[0]);
    }
    let Ok(header_text) = core::str::from_utf8(&header) else {
        return respond(stream, "400 Bad Request", "text/plain", "잘못된 요청입니다.").map_err(Into::into);
    };
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut request = request_line.split_whitespace();
    let method = request.next().unwrap_or_default();
    let path = request.next().unwrap_or_default();
    if request.next() != Some("HTTP/1.1") || request.next().is_some() {
        return respond(stream, "400 Bad Request", "text/plain", "HTTP/1.1 요청이 필요합니다.").map_err(Into::into);
    }
    let mut host = None;
    let mut origin = None;
    let mut length = None;
    let mut invalid = false;
    for line in lines.filter(|line| !line.is_empty()) {
        let Some((name, value)) = line.split_once(':') else { invalid = true; break; };
        let value = value.trim();
        if name.eq_ignore_ascii_case("host") {
            invalid |= host.replace(value).is_some();
        } else if name.eq_ignore_ascii_case("origin") {
            invalid |= origin.replace(value).is_some();
        } else if name.eq_ignore_ascii_case("content-length") {
            invalid |= length.replace(value).is_some();
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            invalid = true;
        }
    }
    // Exact Host blocks DNS rebinding; POST additionally requires our own Origin.
    let expected_origin = format!("http://{authority}");
    if invalid || host != Some(authority) || origin.is_some_and(|value| value != expected_origin) {
        return respond(stream, "403 Forbidden", "text/plain", "로컬 SRG 페이지에서 요청해 주세요.").map_err(Into::into);
    }
    if path == "/api/random" {
        if method != "POST" || origin != Some(expected_origin.as_str()) {
            return respond(stream, "403 Forbidden", "text/plain", "같은 출처의 POST 요청이 필요합니다.").map_err(Into::into);
        }
        let body_length = length.and_then(|value| value.parse::<usize>().ok());
        let Some(body_length @ 1..=5) = body_length else {
            return respond(stream, "400 Bad Request", "text/plain", "생성 개수는 1~10000이어야 합니다.").map_err(Into::into);
        };
        let mut body = vec![0_u8; body_length];
        stream.read_exact(&mut body)?;
        let count = core::str::from_utf8(&body).ok().filter(|text| text.bytes().all(|byte| byte.is_ascii_digit())).and_then(|text| text.parse::<usize>().ok());
        let Some(count @ 1..=MAX_COUNT) = count else {
            return respond(stream, "400 Bad Request", "text/plain", "생성 개수는 1~10000이어야 합니다.").map_err(Into::into);
        };
        return match rng.generate(count) {
            Ok(body) => respond(stream, "200 OK", "application/json; charset=utf-8", &body).map_err(Into::into),
            Err(error) => respond(stream, "503 Service Unavailable", "text/plain; charset=utf-8", &error.to_string()).map_err(Into::into),
        };
    }
    if method != "GET" {
        return respond(stream, "405 Method Not Allowed", "text/plain", "GET 요청이 필요합니다.").map_err(Into::into);
    }
    let (kind, body) = match path {
        "/" | "/revgeo.html" => ("text/html; charset=utf-8", include_str!("../revgeo.html").replacen("</head>", &format!("{BACKEND_MARKER}</head>"), 1)),
        "/icon.svg" => ("image/svg+xml", include_str!("../icon.svg").to_owned()),
        "/site.webmanifest" => ("application/manifest+json", include_str!("../site.webmanifest").to_owned()),
        _ => return respond(stream, "404 Not Found", "text/plain", "Not Found").map_err(Into::into),
    };
    respond(stream, "200 OK", kind, &body).map_err(Into::into)
}

#[cfg(test)]
#[path = "revgeo_server_tests.rs"]
mod tests;
