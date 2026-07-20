use super::c_void;
unsafe extern "system" {
    #[link_name = "CloseHandle"]
    pub(super) fn close_handle(h_object: *mut c_void) -> i32;
    #[link_name = "CreateWaitableTimerExW"]
    pub(in crate::time) fn create_waitable_timer_ex_w(
        timer_attributes: *const c_void,
        timer_name: *const u16,
        flags: u32,
        desired_access: u32,
    ) -> *mut c_void;
    #[link_name = "SetWaitableTimerEx"]
    pub(super) fn set_waitable_timer_ex(
        timer: *mut c_void,
        due_time: *const i64,
        period: i32,
        completion_routine: Option<unsafe extern "system" fn(*const c_void, u32, u32)>,
        arg_to_completion_routine: *const c_void,
        wake_context: *const c_void,
        tolerable_delay: u32,
    ) -> i32;
    #[link_name = "WaitForSingleObject"]
    pub(super) fn wait_for_single_object(handle: *mut c_void, milliseconds: u32) -> u32;
}
