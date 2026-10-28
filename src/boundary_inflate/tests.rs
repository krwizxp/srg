use super::inflate;
use crate::diagnostic::Result;
#[test]
fn decompresses_python_zlib_fixtures() -> Result<()> {
    let cases: [(&[u8], Vec<u8>); 3] = [
        (include_bytes!("tests/stored.zlib"), (0_u8..128).collect()),
        (
            include_bytes!("tests/fixed.zlib"),
            b"SRG boundary".repeat(1000),
        ),
        (
            include_bytes!("tests/dynamic.zlib"),
            b"coordinate boundary sphere triangle area latitude longitude ".repeat(180),
        ),
    ];
    for (compressed, expected) in cases {
        let mut output = [0_u8; 15_876];
        let count = inflate(compressed, &mut output)?;
        assert_eq!(
            output.get(..count),
            Some(expected.as_slice()),
            "Python zlib fixture differs"
        );
    }
    Ok(())
}
#[test]
fn rejects_truncation_checksum_and_output_overflow() {
    let source = include_bytes!("tests/dynamic.zlib");
    let mut output = [0_u8; 15_876];
    for length in 0..source.len() {
        assert!(
            matches!(inflate(source.split_at(length).0, &mut output), Err(_)),
            "Truncated stream accepted"
        );
    }
    let mut corrupted = *source;
    if let Some(last) = corrupted.last_mut() {
        *last ^= 1;
    }
    assert!(
        matches!(inflate(&corrupted, &mut output), Err(_)),
        "Corrupt checksum accepted"
    );
    assert!(
        matches!(inflate(source, &mut [0_u8; 8]), Err(_)),
        "Output limit ignored"
    );
}
