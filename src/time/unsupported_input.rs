use super::NativeInputSendStatus;
use std::io;
#[derive(Clone, Copy)]
pub(super) enum InputAction {
    F5Press,
    MouseClick,
}
pub(super) struct PreparedInput;
impl PreparedInput {
    pub(super) const EMPTY: Self = Self;
    pub(super) fn prepare(
        &mut self,
        _action: Option<InputAction>,
        _err: &mut dyn io::Write,
    ) {
        *self = Self;
    }
    pub(super) const fn reset(&mut self) {
        *self = Self;
    }
    pub(super) fn send(
        &mut self,
        _action: InputAction,
        _err: &mut dyn io::Write,
    ) -> NativeInputSendStatus {
        *self = Self;
        NativeInputSendStatus::FailedBeforeSend
    }
}
