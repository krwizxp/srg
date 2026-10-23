use crate::diagnostic::Result;

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
unsafe extern "C" {
    fn getentropy(buffer: *mut u8, length: usize) -> i32;
}

#[cfg(windows)]
#[link(name = "bcrypt")]
unsafe extern "system" {
    #[link_name = "BCryptGenRandom"]
    fn bcrypt_gen_random(algorithm: *mut core::ffi::c_void, buffer: *mut u8, length: u32, flags: u32) -> i32;
}

#[cfg(any(target_os = "linux", target_vendor = "apple"))]
pub(super) fn fill(bytes: &mut [u8]) -> Result<()> {
    // getentropy accepts at most 256 bytes per call and waits for OS entropy.
    for chunk in bytes.chunks_mut(256) {
        // SAFETY: chunk is writable for chunk.len() bytes, at most 256.
        if unsafe { getentropy(chunk.as_mut_ptr(), chunk.len()) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}

#[cfg(windows)]
pub(super) fn fill(bytes: &mut [u8]) -> Result<()> {
    let length = u32::try_from(bytes.len()).map_err(|_| "난수 요청이 너무 큽니다.")?;
    // SAFETY: the buffer is writable for length bytes. A null provider is
    // required with BCRYPT_USE_SYSTEM_PREFERRED_RNG (0x00000002).
    let status = unsafe { bcrypt_gen_random(core::ptr::null_mut(), bytes.as_mut_ptr(), length, 2) };
    if status < 0 {
        return Err(format!("BCryptGenRandom 실패: {status}").into());
    }
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_vendor = "apple", windows)))]
pub(super) fn fill(_bytes: &mut [u8]) -> Result<()> {
    Err("이 운영체제의 암호화 난수 API는 지원하지 않습니다.".into())
}

#[cfg(test)]
#[path = "system_rng_tests.rs"]
mod tests;
