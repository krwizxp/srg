use super::Input;
#[link(name = "user32")]
unsafe extern "system" {
    pub(super) fn SendInput(c_inputs: u32, p_inputs: *const Input, cb_size: i32) -> u32;
}
