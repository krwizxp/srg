use crate::write_line_best_effort;
use super::{NativeInputSendStatus, TriggerAction};
use core::mem::{align_of, offset_of, size_of};
use std::io::Write;
macro_rules! assert_ffi_layout {
    ($type:ty, $size:expr, $align:expr, $($field:ident: $offset:expr),+ $(,)?) => {
        const _: () = assert!(
            size_of::<$type>() == $size
                && align_of::<$type>() == $align
                $(&& offset_of!($type, $field) == $offset)+,
            concat!(stringify!($type), " ABI layout mismatch")
        );
    };
}
const INPUT_MOUSE: u32 = 0;
const INPUT_KEYBOARD: u32 = 1;
const KEYEVENTF_KEYDOWN: u32 = 0;
const MOUSEEVENTF_LEFTDOWN: u32 = 0x0002;
const MOUSEEVENTF_LEFTUP: u32 = 0x0004;
const KEYEVENTF_KEYUP: u32 = 0x0002;
const VK_F5: u16 = 0x74;
const INPUT_EVENT_COUNT: u32 = 2;
cfg_select! {
    target_pointer_width = "64" => {
        const INPUT_SIZE: i32 = 40;
    }
    target_pointer_width = "32" => {
        const INPUT_SIZE: i32 = 28;
    }
    _ => {}
}
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
#[link(name = "user32")]
unsafe extern "system" {
    fn SendInput(c_inputs: u32, p_inputs: *const Input, cb_size: i32) -> u32;
}
cfg_select! {
    target_pointer_width = "64" => {
        assert_ffi_layout!(
            MouseInput, 32, 8,
            dx: 0, dy: 4, mouse_data: 8, dw_flags: 12, time: 16, dw_extra_info: 24,
        );
        assert_ffi_layout!(
            KeybdInput, 24, 8,
            w_vk: 0, w_scan: 2, dw_flags: 4, time: 8, dw_extra_info: 16,
        );
        assert_ffi_layout!(Input, 40, 8, r#type: 0, union: 8);
    }
    target_pointer_width = "32" => {
        assert_ffi_layout!(
            MouseInput, 24, 4,
            dx: 0, dy: 4, mouse_data: 8, dw_flags: 12, time: 16, dw_extra_info: 20,
        );
        assert_ffi_layout!(
            KeybdInput, 16, 4,
            w_vk: 0, w_scan: 2, dw_flags: 4, time: 8, dw_extra_info: 12,
        );
        assert_ffi_layout!(Input, 28, 4, r#type: 0, union: 4);
    }
    _ => {}
}
impl TriggerAction {
    pub(super) fn send(self, err: &mut dyn Write) -> NativeInputSendStatus {
        match self {
            Self::LeftClick => {
                let release = mouse_input(MOUSEEVENTF_LEFTUP);
                let inputs = [mouse_input(MOUSEEVENTF_LEFTDOWN), release];
                send_input_events(&inputs, &release, err)
            }
            Self::F5Press => {
                let release = keyboard_input(VK_F5, KEYEVENTF_KEYUP);
                let inputs = [keyboard_input(VK_F5, KEYEVENTF_KEYDOWN), release];
                send_input_events(&inputs, &release, err)
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
    inputs: &[Input; 2],
    release_input: &Input,
    err: &mut dyn Write,
) -> NativeInputSendStatus {
    // SAFETY: `inputs.as_ptr()` stays valid for the call and `cb_size` matches the Rust `Input` representation.
    let sent = unsafe { SendInput(INPUT_EVENT_COUNT, inputs.as_ptr(), INPUT_SIZE) };
    if sent == INPUT_EVENT_COUNT {
        return NativeInputSendStatus::Sent;
    }
    write_line_best_effort(
        err,
        format_args!(
            "[경고] Windows 입력 이벤트 전송 실패: 요청 {INPUT_EVENT_COUNT}, 전송 {sent}"
        ),
    );
    if sent == 1 {
        // SAFETY: `release` is a valid one-element INPUT pointer and `INPUT_SIZE` matches the Rust representation.
        let release_sent = unsafe { SendInput(1, release_input, INPUT_SIZE) };
        if release_sent != 1 {
            write_line_best_effort(
                err,
                format_args!("[경고] Windows 입력 release 이벤트 전송 실패"),
            );
        }
    }
    if sent == 0 {
        NativeInputSendStatus::FailedBeforeSend
    } else {
        NativeInputSendStatus::PartialOrUnknown
    }
}
