pub(super) const APP_NAME: &str = env!("CARGO_PKG_NAME");
pub(super) const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
pub(super) const BUILD_GIT_DIRTY: &str = env!("BUILD_GIT_DIRTY");
pub(super) const BUILD_GIT_SHA: &str = env!("BUILD_GIT_SHA");
pub(super) const BUILD_PROFILE: &str = env!("BUILD_PROFILE");
pub(super) const BUILD_RUSTC: &str = env!("BUILD_RUSTC");
pub(super) const BUILD_TARGET: &str = env!("BUILD_TARGET");
pub(super) const RNG_BACKEND: &str = cfg_select! {
    target_arch = "x86_64" => {
        "RDSEED/RDRAND CPU feature detection"
    }
    _ => {
        "hardware RNG disabled on non-x86_64"
    }
};
