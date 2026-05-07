use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BOUNDARY_CHARS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
static BOUNDARY_COUNTER: AtomicU64 = AtomicU64::new(0);

#[no_mangle]
pub unsafe extern "C" fn llama_server_base64_encode(
    data: *const u8,
    len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    if data.is_null() && len != 0 {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null_mut();
    }

    let input = unsafe { slice::from_raw_parts(data, len) };
    let mut output = encode(input).into_boxed_slice();
    let ptr = output.as_mut_ptr();
    let len = output.len();
    std::mem::forget(output);

    if !out_len.is_null() {
        unsafe {
            *out_len = len;
        }
    }

    ptr
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_base64_free(data: *mut u8, len: usize) {
    if !data.is_null() {
        unsafe {
            drop(Vec::from_raw_parts(data, len, len));
        }
    }
}

pub fn encode(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len().div_ceil(3) * 4);

    for chunk in input.chunks(3) {
        let i0 = chunk[0];
        let i1 = *chunk.get(1).unwrap_or(&0);
        let i2 = *chunk.get(2).unwrap_or(&0);

        output.push(ALPHABET[(i0 >> 2) as usize]);
        output.push(ALPHABET[(((i0 & 0x03) << 4) | (i1 >> 4)) as usize]);

        if chunk.len() > 1 {
            output.push(ALPHABET[(((i1 & 0x0f) << 2) | (i2 >> 6)) as usize]);
        } else {
            output.push(b'=');
        }

        if chunk.len() > 2 {
            output.push(ALPHABET[(i2 & 0x3f) as usize]);
        } else {
            output.push(b'=');
        }
    }

    output
}

#[no_mangle]
pub extern "C" fn llama_server_model_status_to_string_rust(status: i32) -> *const std::ffi::c_char {
    match status {
        0 => b"unloaded\0".as_ptr().cast(),
        1 => b"loading\0".as_ptr().cast(),
        2 => b"loaded\0".as_ptr().cast(),
        3 => b"sleeping\0".as_ptr().cast(),
        _ => b"unknown\0".as_ptr().cast(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_model_status_from_string_rust(data: *const u8, len: usize) -> i32 {
    let Some(input) = ffi_bytes(data, len) else {
        return -1;
    };
    match input {
        b"unloaded" => 0,
        b"loading" => 1,
        b"loaded" => 2,
        b"sleeping" => 3,
        _ => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_header_name_is_rust(
    data: *const u8,
    len: usize,
    expected: *const std::ffi::c_char,
) -> bool {
    let Some(input) = ffi_bytes(data, len) else {
        return false;
    };
    if expected.is_null() {
        return false;
    }
    let expected = unsafe { std::ffi::CStr::from_ptr(expected) }.to_bytes();
    ascii_eq_ignore_case(input, expected)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_should_strip_proxy_header_rust(data: *const u8, len: usize) -> bool {
    let Some(input) = ffi_bytes(data, len) else {
        return false;
    };

    if ascii_eq_ignore_case(input, b"server")
        || ascii_eq_ignore_case(input, b"transfer-encoding")
        || ascii_eq_ignore_case(input, b"content-length")
        || ascii_eq_ignore_case(input, b"keep-alive")
    {
        return true;
    }

    ascii_starts_with_ignore_case(input, b"access-control-")
}

#[no_mangle]
pub extern "C" fn llama_server_generate_multipart_boundary_rust(out_len: *mut usize) -> *mut u8 {
    let mut output = b"----llama-cpp-proxy-".to_vec();
    let seed = boundary_seed();
    let mut state = seed ^ BOUNDARY_COUNTER.fetch_add(1, Ordering::Relaxed);
    for _ in 0..16 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        output.push(BOUNDARY_CHARS[(state % BOUNDARY_CHARS.len() as u64) as usize]);
    }

    let mut output = output.into_boxed_slice();
    let ptr = output.as_mut_ptr();
    let len = output.len();
    std::mem::forget(output);

    if !out_len.is_null() {
        unsafe {
            *out_len = len;
        }
    }
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_fnv_hash_rust(data: *const u8, len: usize) -> u64 {
    let Some(input) = ffi_bytes(data, len) else {
        return 0;
    };
    fnv_hash(input)
}

fn ffi_bytes<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    if data.is_null() && len != 0 {
        return None;
    }
    Some(unsafe { slice::from_raw_parts(data, len) })
}

fn ascii_eq_ignore_case(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.eq_ignore_ascii_case(b))
}

fn ascii_starts_with_ignore_case(value: &[u8], prefix: &[u8]) -> bool {
    value.len() >= prefix.len() && ascii_eq_ignore_case(&value[..prefix.len()], prefix)
}

fn boundary_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
}

fn fnv_hash(input: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in input {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(input: &[u8]) -> String {
        String::from_utf8(encode(input)).unwrap()
    }

    #[test]
    fn encodes_standard_padding_cases() {
        assert_eq!(encoded(b""), "");
        assert_eq!(encoded(b"f"), "Zg==");
        assert_eq!(encoded(b"fo"), "Zm8=");
        assert_eq!(encoded(b"foo"), "Zm9v");
        assert_eq!(encoded(b"hello world"), "aGVsbG8gd29ybGQ=");
    }

    #[test]
    fn maps_model_statuses() {
        unsafe {
            assert_eq!(llama_server_model_status_from_string_rust(b"unloaded".as_ptr(), 8), 0);
            assert_eq!(llama_server_model_status_from_string_rust(b"loading".as_ptr(), 7), 1);
            assert_eq!(llama_server_model_status_from_string_rust(b"loaded".as_ptr(), 6), 2);
            assert_eq!(llama_server_model_status_from_string_rust(b"sleeping".as_ptr(), 8), 3);
            assert_eq!(llama_server_model_status_from_string_rust(b"bad".as_ptr(), 3), -1);
        }
    }

    #[test]
    fn compares_and_filters_header_names() {
        unsafe {
            assert!(llama_server_header_name_is_rust(
                b"Content-Type".as_ptr(),
                12,
                b"content-type\0".as_ptr().cast()
            ));
            assert!(llama_server_should_strip_proxy_header_rust(b"Server".as_ptr(), 6));
            assert!(llama_server_should_strip_proxy_header_rust(
                b"Access-Control-Allow-Origin".as_ptr(),
                27
            ));
            assert!(!llama_server_should_strip_proxy_header_rust(
                b"X-Request-Id".as_ptr(),
                12
            ));
        }
    }

    #[test]
    fn generates_proxy_boundaries() {
        let mut len = 0usize;
        let ptr = llama_server_generate_multipart_boundary_rust(&mut len);
        assert!(!ptr.is_null());
        let boundary = unsafe {
            let bytes = slice::from_raw_parts(ptr, len);
            let boundary = String::from_utf8(bytes.to_vec()).unwrap();
            llama_server_base64_free(ptr, len);
            boundary
        };
        assert!(boundary.starts_with("----llama-cpp-proxy-"));
        assert_eq!(boundary.len(), "----llama-cpp-proxy-".len() + 16);
    }

    #[test]
    fn computes_fnv1a_hashes() {
        unsafe {
            assert_eq!(llama_server_fnv_hash_rust(b"".as_ptr(), 0), 14695981039346656037);
            assert_eq!(llama_server_fnv_hash_rust(b"hello".as_ptr(), 5), 11831194018420276491);
            assert_eq!(llama_server_fnv_hash_rust(std::ptr::null(), 1), 0);
        }
    }
}
