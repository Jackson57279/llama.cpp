use std::slice;

#[repr(C)]
pub struct JinjaString {
    data: *mut u8,
    len: usize,
}

#[no_mangle]
pub unsafe extern "C" fn llama_jinja_string_free(value: JinjaString) {
    if !value.data.is_null() {
        drop(Vec::from_raw_parts(value.data, value.len, value.len));
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_jinja_string_is_uppercase(data: *const u8, len: usize) -> bool {
    bytes(data, len)
        .iter()
        .all(|&byte| !byte.is_ascii_lowercase())
}

#[no_mangle]
pub unsafe extern "C" fn llama_jinja_string_is_lowercase(data: *const u8, len: usize) -> bool {
    bytes(data, len)
        .iter()
        .all(|&byte| !byte.is_ascii_uppercase())
}

#[no_mangle]
pub unsafe extern "C" fn llama_jinja_string_uppercase(data: *const u8, len: usize) -> JinjaString {
    into_ffi(bytes(data, len).iter().map(|byte| byte.to_ascii_uppercase()).collect())
}

#[no_mangle]
pub unsafe extern "C" fn llama_jinja_string_lowercase(data: *const u8, len: usize) -> JinjaString {
    into_ffi(bytes(data, len).iter().map(|byte| byte.to_ascii_lowercase()).collect())
}

#[no_mangle]
pub unsafe extern "C" fn llama_jinja_string_capitalize(data: *const u8, len: usize) -> JinjaString {
    let input = bytes(data, len);
    if input.is_empty() {
        return into_ffi(Vec::new());
    }

    let mut result = Vec::with_capacity(input.len());
    result.push(input[0].to_ascii_uppercase());
    result.extend(input[1..].iter().map(|byte| byte.to_ascii_lowercase()));
    into_ffi(result)
}

#[no_mangle]
pub unsafe extern "C" fn llama_jinja_string_titlecase(data: *const u8, len: usize) -> JinjaString {
    let mut capitalize_next = true;
    let mut result = Vec::with_capacity(len);
    for &byte in bytes(data, len) {
        if byte.is_ascii_whitespace() {
            capitalize_next = true;
            result.push(byte);
        } else if capitalize_next {
            result.push(byte.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(byte.to_ascii_lowercase());
        }
    }
    into_ffi(result)
}

#[no_mangle]
pub unsafe extern "C" fn llama_jinja_string_strip(
    data: *const u8,
    len: usize,
    left: bool,
    right: bool,
    chars: *const u8,
    chars_len: usize,
    has_chars: bool,
) -> JinjaString {
    let input = bytes(data, len);
    let chars = if has_chars {
        Some(bytes(chars, chars_len))
    } else {
        None
    };

    let mut start = 0usize;
    let mut end = input.len();
    if left {
        while start < end && match_char(input[start], chars) {
            start += 1;
        }
    }
    if right {
        while end > start && match_char(input[end - 1], chars) {
            end -= 1;
        }
    }

    into_ffi(input[start..end].to_vec())
}

unsafe fn bytes<'a>(data: *const u8, len: usize) -> &'a [u8] {
    if data.is_null() && len != 0 {
        &[]
    } else {
        slice::from_raw_parts(data, len)
    }
}

fn match_char(byte: u8, chars: Option<&[u8]>) -> bool {
    if let Some(chars) = chars {
        chars.contains(&byte)
    } else {
        byte.is_ascii_whitespace()
    }
}

fn into_ffi(mut bytes: Vec<u8>) -> JinjaString {
    let len = bytes.len();
    let data = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    JinjaString { data, len }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn owned(value: JinjaString) -> String {
        let data = slice::from_raw_parts(value.data, value.len).to_vec();
        llama_jinja_string_free(value);
        String::from_utf8(data).unwrap()
    }

    #[test]
    fn case_transforms_match_cpp_ascii_behavior() {
        assert_eq!(unsafe { owned(llama_jinja_string_uppercase(b"aBc!".as_ptr(), 4)) }, "ABC!");
        assert_eq!(unsafe { owned(llama_jinja_string_lowercase(b"aBc!".as_ptr(), 4)) }, "abc!");
        assert_eq!(unsafe { owned(llama_jinja_string_capitalize(b"hELLO".as_ptr(), 5)) }, "Hello");
        assert_eq!(unsafe { owned(llama_jinja_string_titlecase(b"hi THERE".as_ptr(), 8)) }, "Hi There");
    }

    #[test]
    fn strips_default_and_custom_chars() {
        assert_eq!(
            unsafe { owned(llama_jinja_string_strip(b"  hi \n".as_ptr(), 6, true, true, std::ptr::null(), 0, false)) },
            "hi"
        );
        assert_eq!(
            unsafe { owned(llama_jinja_string_strip(b"..hi..".as_ptr(), 6, true, true, b".".as_ptr(), 1, true)) },
            "hi"
        );
    }
}
