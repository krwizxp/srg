use super::{HighResTimerGuard, NativeInputSendStatus, TriggerAction};
use crate::write_line_best_effort;
use core::{ffi::c_void, mem::offset_of, ptr::null, time::Duration};
use std::{
    io::{self, Write},
    thread::sleep,
    time::Instant,
};
#[link(name = "user32")]
unsafe extern "system" {
    fn SendInput(c_inputs: u32, p_inputs: *const Input, cb_size: i32) -> u32;
}
unsafe extern "system" {
    #[link_name = "CloseHandle"]
    fn close_handle(h_object: *mut c_void) -> i32;
    #[link_name = "CreateWaitableTimerExW"]
    pub(super) fn create_waitable_timer_ex_w(
        attrs: *const c_void,
        name: *const u16,
        flags: u32,
        access: u32,
    ) -> *mut c_void;
    #[link_name = "SetWaitableTimerEx"]
    fn set_waitable_timer_ex(
        timer: *mut c_void,
        due_time: *const i64,
        period: i32,
        routine: Option<unsafe extern "system" fn(*const c_void, u32, u32)>,
        arg: *const c_void,
        wake_ctx: *const c_void,
        tolerable_delay: u32,
    ) -> i32;
    #[link_name = "WaitForSingleObject"]
    fn wait_for_single_object(handle: *mut c_void, milliseconds: u32) -> u32;
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
#[derive(Copy, Clone)]
struct Input {
    r#type: u32,
    union: InputUnion,
}
#[repr(C)]
#[derive(Copy, Clone)]
union InputUnion {
    ki: KeybdInput,
    mi: MouseInput,
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
#[derive(Copy, Clone, Default)]
struct MouseInput {
    dx: i32,
    dy: i32,
    mouse_data: u32,
    dw_flags: u32,
    time: u32,
    dw_extra_info: usize,
}
assert_ffi_layout!(MouseInput, 32, 8, dx: 0, dy: 4, mouse_data: 8, dw_flags: 12, time: 16, dw_extra_info: 24);
assert_ffi_layout!(KeybdInput, 24, 8, w_vk: 0, w_scan: 2, dw_flags: 4, time: 8, dw_extra_info: 16);
assert_ffi_layout!(Input, 40, 8, r#type: 0, union: 8);
impl Drop for HighResTimerGuard {
    fn drop(&mut self) {
        // SAFETY: handle was returned by CreateWaitableTimerExW and is closed once here.
        unsafe {
            close_handle(self.handle.as_ptr());
        }
    }
}
impl HighResTimerGuard {
    pub(super) fn sleep(&self, duration: Duration) -> io::Result<()> {
        let started_at = Instant::now();
        (|| -> io::Result<()> {
            let due_time = i64::try_from(duration.as_nanos().div_ceil(100).max(1))
                .map_err(|src| io::Error::new(io::ErrorKind::InvalidInput, src))?
                .wrapping_neg();
            // SAFETY: handle is a live waitable timer handle and due_time points to a valid relative due-time.
            let set_ok = unsafe {
                set_waitable_timer_ex(
                    self.handle.as_ptr(),
                    &raw const due_time,
                    0,
                    None,
                    null(),
                    null(),
                    0,
                )
            };
            (set_ok != 0_i32).ok_or_else(|| {
                io::Error::other(format!(
                    "SetWaitableTimerEx 실패: {}",
                    io::Error::last_os_error()
                ))
            })?;
            // SAFETY: handle remains valid while waiting.
            let wait_status = unsafe { wait_for_single_object(self.handle.as_ptr(), u32::MAX) };
            (wait_status == 0_u32).ok_or_else(|| {
                io::Error::other(format!(
                    "WaitForSingleObject 상태 {wait_status:#010x}: {}",
                    io::Error::last_os_error()
                ))
            })
        })()
        .inspect_err(|_| sleep(duration.saturating_sub(started_at.elapsed())))
    }
}
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
        // SAFETY: `inputs.as_ptr()` stays valid for the call and `cb_size` matches `Input`.
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
            // SAFETY: `release_input` is a valid one-element INPUT pointer and `40` matches `Input`.
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
