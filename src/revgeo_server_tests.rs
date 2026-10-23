use super::{BACKEND_MARKER, WebRng, handle_connection};
use crate::diagnostic::Result;
use std::{io::{Read as _, Write as _}, net::{Ipv4Addr, TcpListener, TcpStream}, thread};

fn request(method: &str, path: &str, headers: &str, body: &str) -> Result<String> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    let address = listener.local_addr()?;
    let authority = address.to_string();
    let headers = headers.replace("$HOST", &authority);
    let worker = thread::spawn(move || {
        let (mut stream, _) = listener.accept()?;
        handle_connection(&mut stream, &authority, &WebRng::new())
    });
    let mut stream = TcpStream::connect(address)?;
    write!(stream, "{method} {path} HTTP/1.1\r\n{headers}\r\n{body}")?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    worker.join().map_err(|_| "요청 스레드 실패")??;
    Ok(response)
}

#[test]
fn serves_bundled_page_and_only_known_assets() -> Result<()> {
    let page = request("GET", "/revgeo.html", "Host: $HOST\r\n", "")?;
    assert!(page.starts_with("HTTP/1.1 200"), "page must load");
    assert!(page.contains(BACKEND_MARKER), "local page must discover its backend");
    assert!(page.contains("Cache-Control: no-store"), "page must not be cached");
    let response = request("GET", "/../Cargo.toml", "Host: $HOST\r\n", "")?;
    assert!(response.starts_with("HTTP/1.1 404"), "unknown files must stay private");
    Ok(())
}

#[test]
fn rejects_foreign_hosts_and_origins() -> Result<()> {
    for headers in [
        "Host: attacker.example\r\nOrigin: http://$HOST\r\nContent-Length: 1\r\n",
        "Host: $HOST\r\nOrigin: https://attacker.example\r\nContent-Length: 1\r\n",
        "Host: $HOST\r\nContent-Length: 1\r\n",
        "Host: $HOST\r\nHost: attacker.example\r\nOrigin: http://$HOST\r\nContent-Length: 1\r\n",
    ] {
        let response = request("POST", "/api/random", headers, "")?;
        assert!(response.starts_with("HTTP/1.1 403"), "foreign request must be forbidden");
        assert!(!response.contains("\"values\""), "forbidden requests must not generate entropy");
    }
    Ok(())
}

#[test]
fn rejects_invalid_counts_and_request_framing() -> Result<()> {
    for body in ["0", "10001", "-1", "1.5", "1e3", ""] {
        let response = request("POST", "/api/random", &format!("Host: $HOST\r\nOrigin: http://$HOST\r\nContent-Length: {}\r\n", body.len()), body)?;
        assert!(response.starts_with("HTTP/1.1 400"), "invalid count must be rejected: {body}");
    }
    let response = request("POST", "/api/random", "Host: $HOST\r\nOrigin: http://$HOST\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n", "")?;
    assert!(response.starts_with("HTTP/1.1 403"), "ambiguous framing must be rejected");
    Ok(())
}

#[test]
fn generates_exact_count_as_decimal_strings() -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    if WebRng::new().hardware.source() == crate::hardware_rng::HardwareRandomSource::None {
        let response = request("POST", "/api/random", "Host: $HOST\r\nOrigin: http://$HOST\r\nContent-Length: 1\r\n", "3")?;
        assert!(response.starts_with("HTTP/1.1 503"), "unsupported hardware must fail explicitly");
        return Ok(());
    }
    let response = request("POST", "/api/random", "Host: $HOST\r\nOrigin: http://$HOST\r\nContent-Length: 1\r\n", "3")?;
    assert!(response.starts_with("HTTP/1.1 200"), "entropy request must succeed");
    let (_, body) = response.split_once("\r\n\r\n").ok_or("응답 본문 누락")?;
    let (_, values) = body.split_once("\"values\":[\"").ok_or("난수 목록 누락")?;
    let values = values.strip_suffix("\"]}").ok_or("난수 목록 형식 오류")?;
    let tokens: Vec<_> = values.split("\",\"").collect();
    assert_eq!(tokens.len(), 3, "requested count must match response count");
    for value in tokens {
        assert!(value.parse::<u64>().is_ok(), "all values must be full-width u64 strings");
    }
    Ok(())
}
