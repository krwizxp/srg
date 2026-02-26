const INPUT_MOUSE: u32 = 0;
const INPUT_KEYBOARD: u32 = 1;
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
impl Input {
    const fn mouse(mi: MouseInput) -> Self {
        Self {
            r#type: INPUT_MOUSE,
            union: InputUnion { mi },
        }
    }
    const fn keyboard(ki: KeybdInput) -> Self {
        Self {
            r#type: INPUT_KEYBOARD,
            union: InputUnion { ki },
        }
    }
}
#[cfg(target_pointer_width = "64")]
const _: [(); 40] = [(); std::mem::size_of::<Input>()];
#[cfg(target_pointer_width = "32")]
const _: [(); 28] = [(); std::mem::size_of::<Input>()];
#[link(name = "user32")]
unsafe extern "system" {
    fn SendInput(c_inputs: u32, p_inputs: *const Input, cb_size: i32) -> u32;
}
fn send_input_events(inputs: &[Input]) {
    unsafe {
        let Ok(input_count) = u32::try_from(inputs.len()) else {
            eprintln!("[경고] Windows 입력 이벤트 수 변환 실패");
            return;
        };
        let Ok(input_size) = i32::try_from(std::mem::size_of::<Input>()) else {
            eprintln!("[경고] Windows 입력 이벤트 크기 변환 실패");
            return;
        };
        let sent = SendInput(input_count, inputs.as_ptr(), input_size);
        if sent != input_count {
            eprintln!("[경고] Windows 입력 이벤트 전송 실패: 요청 {input_count}, 전송 {sent}");
        }
    }
}
pub fn send_mouse_click() {
    let inputs = [
        Input::mouse(MouseInput {
            dw_flags: MOUSEEVENTF_LEFTDOWN,
            ..Default::default()
        }),
        Input::mouse(MouseInput {
            dw_flags: MOUSEEVENTF_LEFTUP,
            ..Default::default()
        }),
    ];
    send_input_events(&inputs);
}
pub fn send_f5_press() {
    let inputs = [
        Input::keyboard(KeybdInput {
            w_vk: VK_F5,
            ..Default::default()
        }),
        Input::keyboard(KeybdInput {
            w_vk: VK_F5,
            dw_flags: KEYEVENTF_KEYUP,
            ..Default::default()
        }),
    ];
    send_input_events(&inputs);
}
