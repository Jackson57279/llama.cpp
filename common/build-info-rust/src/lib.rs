use std::os::raw::{c_char, c_int};

#[no_mangle]
pub extern "C" fn llama_build_number() -> c_int {
    env!("LLAMA_BUILD_NUMBER").parse::<c_int>().unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn llama_commit() -> *const c_char {
    concat!(env!("LLAMA_BUILD_COMMIT"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn llama_compiler() -> *const c_char {
    concat!(env!("BUILD_COMPILER"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn llama_build_target() -> *const c_char {
    concat!(env!("BUILD_TARGET"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn llama_build_info() -> *const c_char {
    concat!(env!("LLAMA_BUILD_INFO"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn llama_print_build_info() {
    eprintln!(
        "llama_print_build_info: build = {} ({})",
        llama_build_number(),
        env!("LLAMA_BUILD_COMMIT")
    );
    eprintln!(
        "llama_print_build_info: built with {} for {}",
        env!("BUILD_COMPILER"),
        env!("BUILD_TARGET")
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    unsafe fn c_string(ptr: *const c_char) -> String {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }

    #[test]
    fn exposes_configured_strings() {
        assert!(!unsafe { c_string(llama_commit()) }.is_empty());
        assert!(!unsafe { c_string(llama_compiler()) }.is_empty());
        assert!(!unsafe { c_string(llama_build_target()) }.is_empty());
        assert!(unsafe { c_string(llama_build_info()) }.starts_with('b'));
    }

    #[test]
    fn build_number_is_parseable() {
        assert_eq!(
            llama_build_number(),
            env!("LLAMA_BUILD_NUMBER").parse::<c_int>().unwrap_or(0)
        );
    }
}
