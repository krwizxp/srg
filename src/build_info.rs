pub const APP_NAME: &str = env!("CARGO_PKG_NAME");
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const BUILD_GIT_SHA: &str = env!("BUILD_GIT_SHA");
pub const BUILD_PROFILE: &str = env!("BUILD_PROFILE");
pub const BUILD_RUSTC: &str = env!("BUILD_RUSTC");
pub const BUILD_TARGET: &str = env!("BUILD_TARGET");
pub const RNG_BACKEND: &str = if cfg!(target_arch = "x86_64") {
    "RDSEED/RDRAND CPU feature detection"
} else {
    "hardware RNG disabled on non-x86_64"
};
