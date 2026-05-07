use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;

#[cfg(unix)]
mod platform {
    use super::*;

    const RTLD_NOW: c_int = 2;
    const RTLD_LOCAL: c_int = 0;

    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
    }

    pub unsafe fn load(path: *const c_char) -> *mut c_void {
        dlopen(path, RTLD_NOW | RTLD_LOCAL)
    }

    pub unsafe fn sym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
        dlsym(handle, name)
    }

    pub unsafe fn error() -> *const c_char {
        let err = dlerror();
        if err.is_null() {
            c"".as_ptr()
        } else {
            err
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::*;

    const SEM_FAILCRITICALERRORS: u32 = 0x0001;

    #[link(name = "kernel32")]
    extern "system" {
        fn SetErrorMode(mode: u32) -> u32;
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    pub unsafe fn load(path: *const c_char) -> *mut c_void {
        let Ok(path) = std::ffi::CStr::from_ptr(path).to_str() else {
            return ptr::null_mut();
        };
        let mut wide: Vec<u16> = path.encode_utf16().collect();
        wide.push(0);

        let old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
        SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
        let handle = LoadLibraryW(wide.as_ptr());
        SetErrorMode(old_mode);
        handle
    }

    pub unsafe fn sym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
        let old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
        SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
        let symbol = GetProcAddress(handle, name);
        SetErrorMode(old_mode);
        symbol
    }

    pub unsafe fn error() -> *const c_char {
        c"".as_ptr()
    }
}

#[no_mangle]
pub unsafe extern "C" fn ggml_backend_dl_load_library_rust(path: *const c_char) -> *mut c_void {
    if path.is_null() {
        return ptr::null_mut();
    }
    platform::load(path)
}

#[no_mangle]
pub unsafe extern "C" fn ggml_backend_dl_get_sym_rust(
    handle: *mut c_void,
    name: *const c_char,
) -> *mut c_void {
    if handle.is_null() || name.is_null() {
        return ptr::null_mut();
    }
    platform::sym(handle, name)
}

#[no_mangle]
pub unsafe extern "C" fn ggml_backend_dl_error_rust() -> *const c_char {
    platform::error()
}

#[no_mangle]
pub unsafe extern "C" fn ggml_backend_striequals_rust(a: *const c_char, b: *const c_char) -> c_int {
    if a.is_null() || b.is_null() {
        return 0;
    }

    let a = CStr::from_ptr(a).to_bytes();
    let b = CStr::from_ptr(b).to_bytes();
    if a.len() != b.len() {
        return 0;
    }

    a.iter()
        .zip(b.iter())
        .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase()) as c_int
}

#[no_mangle]
pub extern "C" fn ggml_backend_current_exe_dir_rust() -> *mut c_char {
    current_exe_dir()
        .and_then(|path| CString::new(path).ok())
        .map(CString::into_raw)
        .unwrap_or(ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn ggml_backend_cstring_free_rust(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

#[no_mangle]
pub extern "C" fn ggml_backend_filename_prefix_rust() -> *const c_char {
    #[cfg(windows)]
    {
        c"ggml-".as_ptr()
    }
    #[cfg(not(windows))]
    {
        c"libggml-".as_ptr()
    }
}

#[no_mangle]
pub extern "C" fn ggml_backend_filename_extension_rust() -> *const c_char {
    #[cfg(windows)]
    {
        c".dll".as_ptr()
    }
    #[cfg(not(windows))]
    {
        c".so".as_ptr()
    }
}

fn current_exe_dir() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    Some(dir.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn nulls_are_rejected() {
        unsafe {
            assert!(ggml_backend_dl_load_library_rust(ptr::null()).is_null());
            assert!(ggml_backend_dl_get_sym_rust(ptr::null_mut(), c"missing".as_ptr()).is_null());
            assert!(ggml_backend_dl_get_sym_rust(ptr::NonNull::<c_void>::dangling().as_ptr(), ptr::null()).is_null());
            assert!(!ggml_backend_dl_error_rust().is_null());
            assert_eq!(ggml_backend_striequals_rust(ptr::null(), c"cpu".as_ptr()), 0);
            assert_eq!(ggml_backend_striequals_rust(c"cpu".as_ptr(), ptr::null()), 0);
        }
    }

    #[test]
    fn compares_ascii_names_case_insensitively() {
        unsafe {
            assert_eq!(ggml_backend_striequals_rust(c"CPU".as_ptr(), c"cpu".as_ptr()), 1);
            assert_eq!(ggml_backend_striequals_rust(c"CuDa".as_ptr(), c"cuda".as_ptr()), 1);
            assert_eq!(ggml_backend_striequals_rust(c"cpu".as_ptr(), c"cpux".as_ptr()), 0);
            assert_eq!(ggml_backend_striequals_rust(c"metal".as_ptr(), c"vulkan".as_ptr()), 0);
        }
    }

    #[test]
    fn exposes_current_executable_directory() {
        unsafe {
            let dir_ptr = ggml_backend_current_exe_dir_rust();
            assert!(!dir_ptr.is_null());
            let dir = CStr::from_ptr(dir_ptr).to_string_lossy().into_owned();
            assert!(!dir.is_empty());
            assert!(std::path::Path::new(&dir).is_dir());
            ggml_backend_cstring_free_rust(dir_ptr);
        }
    }

    #[test]
    fn exposes_backend_filename_parts() {
        unsafe {
            #[cfg(windows)]
            {
                assert_eq!(CStr::from_ptr(ggml_backend_filename_prefix_rust()).to_bytes(), b"ggml-");
                assert_eq!(CStr::from_ptr(ggml_backend_filename_extension_rust()).to_bytes(), b".dll");
            }
            #[cfg(not(windows))]
            {
                assert_eq!(CStr::from_ptr(ggml_backend_filename_prefix_rust()).to_bytes(), b"libggml-");
                assert_eq!(CStr::from_ptr(ggml_backend_filename_extension_rust()).to_bytes(), b".so");
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn can_resolve_process_symbol() {
        unsafe {
            let handle = ggml_backend_dl_load_library_rust(c"".as_ptr());
            assert!(!handle.is_null());
            let symbol = ggml_backend_dl_get_sym_rust(handle, c"printf".as_ptr());
            assert!(!symbol.is_null());
        }
    }
}
