use std::ffi::{c_char, c_void};

const APIR_BACKEND_INITIALIZE_SUCCESS: u32 = 0;
const APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED: u32 = 6;
const APIR_BACKEND_INITIALIZE_ALREADY_INITED: u32 = 7;
const APIR_BACKEND_INITIALIZE_NO_DEVICE: u32 = 8;
const APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED: u32 = 9;

#[repr(C)]
struct GgmlBackendReg {
    api_version: i32,
    iface: GgmlBackendRegI,
    context: *mut c_void,
}

#[repr(C)]
struct GgmlBackendRegI {
    get_name: Option<unsafe extern "C" fn(*mut GgmlBackendReg) -> *const c_char>,
    get_device_count: Option<unsafe extern "C" fn(*mut GgmlBackendReg) -> usize>,
    get_device: Option<unsafe extern "C" fn(*mut GgmlBackendReg, usize) -> *mut GgmlBackendDevice>,
    get_proc_address: Option<unsafe extern "C" fn(*mut GgmlBackendReg, *const c_char) -> *mut c_void>,
}

#[repr(C)]
struct GgmlBackendDevice {
    iface: GgmlBackendDeviceI,
    reg: *mut GgmlBackendReg,
    context: *mut c_void,
}

#[repr(C)]
struct GgmlBackendDeviceI {
    get_name: Option<unsafe extern "C" fn(*mut GgmlBackendDevice) -> *const c_char>,
    get_description: Option<unsafe extern "C" fn(*mut GgmlBackendDevice) -> *const c_char>,
    get_memory: Option<unsafe extern "C" fn(*mut GgmlBackendDevice, *mut usize, *mut usize)>,
    get_type: Option<unsafe extern "C" fn(*mut GgmlBackendDevice) -> i32>,
    get_props: Option<unsafe extern "C" fn(*mut GgmlBackendDevice, *mut c_void)>,
    init_backend: Option<unsafe extern "C" fn(*mut GgmlBackendDevice, *const c_char) -> *mut GgmlBackend>,
}

#[repr(C)]
struct GgmlBackend {
    _private: [u8; 0],
}

type BackendRegFn = unsafe extern "C" fn() -> *mut GgmlBackendReg;

#[no_mangle]
pub static mut reg: *mut c_void = std::ptr::null_mut();
#[no_mangle]
pub static mut dev: *mut c_void = std::ptr::null_mut();
#[no_mangle]
pub static mut bck: *mut c_void = std::ptr::null_mut();

#[no_mangle]
pub static mut timer_start: u64 = 0;
#[no_mangle]
pub static mut timer_total: u64 = 0;
#[no_mangle]
pub static mut timer_count: u64 = 0;

unsafe fn initialize_from_fn(backend_reg_fn: BackendRegFn) -> u32 {
    if !reg.is_null() {
        return APIR_BACKEND_INITIALIZE_ALREADY_INITED;
    }

    reg = backend_reg_fn().cast();
    if reg.is_null() {
        return APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    let reg_ptr = reg.cast::<GgmlBackendReg>();

    let Some(get_device_count) = (*reg_ptr).iface.get_device_count else {
        return APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    };
    if get_device_count(reg_ptr) == 0 {
        return APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    let Some(get_device) = (*reg_ptr).iface.get_device else {
        return APIR_BACKEND_INITIALIZE_NO_DEVICE;
    };
    dev = get_device(reg_ptr, 0).cast();
    if dev.is_null() {
        return APIR_BACKEND_INITIALIZE_NO_DEVICE;
    }

    let dev_ptr = dev.cast::<GgmlBackendDevice>();

    let Some(init_backend) = (*dev_ptr).iface.init_backend else {
        return APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    };
    bck = init_backend(dev_ptr, std::ptr::null()).cast();
    if bck.is_null() {
        return APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;
    }

    APIR_BACKEND_INITIALIZE_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn backend_dispatch_initialize(ggml_backend_reg_fct_p: *mut c_void) -> u32 {
    if ggml_backend_reg_fct_p.is_null() {
        return APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;
    }

    let backend_reg_fn = std::mem::transmute::<*mut c_void, BackendRegFn>(ggml_backend_reg_fct_p);
    initialize_from_fn(backend_reg_fn)
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn fake_device_count(_reg: *mut GgmlBackendReg) -> usize {
        1
    }

    unsafe extern "C" fn fake_get_device(_reg: *mut GgmlBackendReg, _index: usize) -> *mut GgmlBackendDevice {
        &raw mut FAKE_DEV
    }

    unsafe extern "C" fn fake_init_backend(
        _dev: *mut GgmlBackendDevice,
        _params: *const c_char,
    ) -> *mut GgmlBackend {
        &raw mut FAKE_BACKEND
    }

    static mut FAKE_BACKEND: GgmlBackend = GgmlBackend { _private: [] };
    static mut FAKE_DEV: GgmlBackendDevice = GgmlBackendDevice {
        iface: GgmlBackendDeviceI {
            get_name: None,
            get_description: None,
            get_memory: None,
            get_type: None,
            get_props: None,
            init_backend: Some(fake_init_backend),
        },
        reg: std::ptr::null_mut(),
        context: std::ptr::null_mut(),
    };
    static mut FAKE_REG: GgmlBackendReg = GgmlBackendReg {
        api_version: 2,
        iface: GgmlBackendRegI {
            get_name: None,
            get_device_count: Some(fake_device_count),
            get_device: Some(fake_get_device),
            get_proc_address: None,
        },
        context: std::ptr::null_mut(),
    };

    unsafe extern "C" fn fake_reg() -> *mut GgmlBackendReg {
        &raw mut FAKE_REG
    }

    #[test]
    fn initializes_backend_once() {
        unsafe {
            reg = std::ptr::null_mut();
            dev = std::ptr::null_mut();
            bck = std::ptr::null_mut();

            assert_eq!(initialize_from_fn(fake_reg), APIR_BACKEND_INITIALIZE_SUCCESS);
            assert!(!reg.is_null());
            assert!(!dev.is_null());
            assert!(!bck.is_null());
            assert_eq!(initialize_from_fn(fake_reg), APIR_BACKEND_INITIALIZE_ALREADY_INITED);
        }
    }
}
