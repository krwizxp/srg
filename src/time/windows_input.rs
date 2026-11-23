use super::{NativeInputSendStatus, TriggerAction};
use crate::write_line_best_effort;
use core::mem::offset_of;
use std::io::Write;
#[link(name = "user32")]
unsafe extern "system" {
    fn SendInput(c_inputs: u32, p_inputs: *const Input, cb_size: i32) -> u32;
}
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
assert_ffi_layout!(
    MouseInput, 32, 8,
    dx: 0, dy: 4, mouse_data: 8, dw_flags: 12, time: 16, dw_extra_info: 24,
);
assert_ffi_layout!(
    KeybdInput, 24, 8,
    w_vk: 0, w_scan: 2, dw_flags: 4, time: 8, dw_extra_info: 16,
);
assert_ffi_layout!(Input, 40, 8, r#type: 0, union: 8);
impl TriggerAction {
    pub(super) fn send(self, err: &mut dyn Write) -> NativeInputSendStatus {
        let inputs = match self {
            Self::LeftClick => [0x0002_u32, 0x0004].map(|dw_flags| Input {
                r#type: 0,
                union: InputUnion {
                    mi: MouseInput {
                        dw_flags,
                        ..Default::default()
                    },
                },
            }),
            Self::F5Press => [0_u32, 0x0002].map(|dw_flags| Input {
                r#type: 1,
                union: InputUnion {
                    ki: KeybdInput {
                        w_vk: 0x74,
                        dw_flags,
                        ..Default::default()
                    },
                },
            }),
        };
        // SAFETY: `inputs.as_ptr()` stays valid for the call and `cb_size` matches the Rust `Input` representation.
        let sent = unsafe { SendInput(2, inputs.as_ptr(), 40) };
        if sent == 2 {
            return NativeInputSendStatus::Sent;
        }
        write_line_best_effort(
            err,
            format_args!("[경고] Windows 입력 이벤트 전송 실패: 요청 2, 전송 {sent}"),
        );
        if sent == 1 {
            let [_press, release_input] = inputs;
            // SAFETY: `release_input` is a valid one-element INPUT pointer and `40` matches the Rust representation.
            if unsafe { SendInput(1, &raw const release_input, 40) } != 1 {
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
}
