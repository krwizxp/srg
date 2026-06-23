use crate::write_line_best_effort;
use super::NativeInputSendStatus;
use core::mem::size_of;
use std::io::Write;
mod sys {
    use super::Input;
    #[link(name = "user32")]
    unsafe extern "system" {
        pub(super) fn SendInput(c_inputs: u32, p_inputs: *const Input, cb_size: i32) -> u32;
    }
}
const INPUT_MOUSE: u32 = 0;
const INPUT_KEYBOARD: u32 = 1;
const KEYEVENTF_KEYDOWN: u32 = 0;
const MOUSEEVENTF_LEFTDOWN: u32 = 0x0002;
const MOUSEEVENTF_LEFTUP: u32 = 0x0004;
const KEYEVENTF_KEYUP: u32 = 0x0002;
const VK_F5: u16 = 0x74;
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct MouseInput {
    dx: i32,
    dy: i32,
    mouse_data: u32,
    dw_flags: u32,
    time: u32,
    dw_extra_info: usize,
}
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct KeybdInput {
    w_vk: u16,
    w_scan: u16,
    dw_flags: u32,
    time: u32,
    dw_extra_info: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
union InputUnion {
    mi: MouseInput,
    ki: KeybdInput,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct Input {
    r#type: u32,
    union: InputUnion,
}
cfg_select! {
    target_pointer_width = "64" => {
        const _: () = assert!(size_of::<Input>() == 40, "Windows INPUT x64 size mismatch");
    }
    target_pointer_width = "32" => {
        const _: () = assert!(size_of::<Input>() == 28, "Windows INPUT x86 size mismatch");
    }
    _ => {}
}
#[derive(Clone, Copy)]
pub(super) enum InputAction {
    F5Press,
    MouseClick,
}
pub(super) struct PreparedInput;
impl PreparedInput {
    pub(super) const EMPTY: Self = Self;
    pub(super) fn prepare(&mut self, _action: Option<InputAction>, _err: &mut dyn Write) {
        *self = Self;
    }
    pub(super) const fn reset(&mut self) {
        *self = Self;
    }
    pub(super) fn send(
        &mut self,
        action: InputAction,
        err: &mut dyn Write,
    ) -> NativeInputSendStatus {
        *self = Self;
        match action {
            InputAction::MouseClick => {
                let release = mouse_input(MOUSEEVENTF_LEFTUP);
                let inputs = [
                    mouse_input(MOUSEEVENTF_LEFTDOWN),
                    release,
                ];
                send_input_events(&inputs, Some(&release), err)
            }
            InputAction::F5Press => {
                let release = keyboard_input(VK_F5, KEYEVENTF_KEYUP);
                let inputs = [
                    keyboard_input(VK_F5, KEYEVENTF_KEYDOWN),
                    release,
                ];
                send_input_events(&inputs, Some(&release), err)
            }
        }
    }
}
fn keyboard_input(w_vk: u16, dw_flags: u32) -> Input {
    Input {
        r#type: INPUT_KEYBOARD,
        union: InputUnion {
            ki: KeybdInput {
                w_vk,
                dw_flags,
                ..Default::default()
            },
        },
    }
}
fn mouse_input(dw_flags: u32) -> Input {
    Input {
        r#type: INPUT_MOUSE,
        union: InputUnion {
            mi: MouseInput {
                dw_flags,
                ..Default::default()
            },
        },
    }
}
fn send_input_events(
    inputs: &[Input],
    release_input: Option<&Input>,
    err: &mut dyn Write,
) -> NativeInputSendStatus {
    let Ok(input_count) = u32::try_from(inputs.len()) else {
        write_line_best_effort(err, format_args!("[경고] Windows 입력 이벤트 수 변환 실패"));
        return NativeInputSendStatus::FailedBeforeSend;
    };
    let Ok(input_size) = i32::try_from(size_of::<Input>()) else {
        write_line_best_effort(
            err,
            format_args!("[경고] Windows 입력 이벤트 크기 변환 실패"),
        );
        return NativeInputSendStatus::FailedBeforeSend;
    };
    // SAFETY: `inputs.as_ptr()` stays valid for the call and `cb_size` matches the Rust `Input` representation.
    let sent = unsafe { sys::SendInput(input_count, inputs.as_ptr(), input_size) };
    if sent == input_count {
        return NativeInputSendStatus::Sent;
    }
    if sent != input_count {
        write_line_best_effort(
            err,
            format_args!("[경고] Windows 입력 이벤트 전송 실패: 요청 {input_count}, 전송 {sent}"),
        );
        if sent == 1
            && let Some(release) = release_input
        {
            // SAFETY: `release` is a valid one-element INPUT pointer and `input_size` matches the Rust representation.
            let release_sent = unsafe { sys::SendInput(1, release, input_size) };
            if release_sent != 1 {
                write_line_best_effort(
                    err,
                    format_args!("[경고] Windows 입력 release 이벤트 전송 실패"),
                );
            }
        }
    }
    if sent == 0 {
        NativeInputSendStatus::FailedBeforeSend
    } else {
        NativeInputSendStatus::PartialOrUnknown
    }
}
