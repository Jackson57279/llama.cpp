use std::sync::{Mutex, MutexGuard};

static CRITICAL_SECTION: Mutex<()> = Mutex::new(());
static mut CRITICAL_SECTION_GUARD: Option<MutexGuard<'static, ()>> = None;

#[no_mangle]
pub extern "C" fn ggml_critical_section_start() {
    let guard = CRITICAL_SECTION
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    unsafe {
        CRITICAL_SECTION_GUARD = Some(guard);
    }
}

#[no_mangle]
pub extern "C" fn ggml_critical_section_end() {
    unsafe {
        CRITICAL_SECTION_GUARD = None;
    }
}

#[no_mangle]
pub extern "C" fn ggml_op_is_empty_rust(op: i32) -> bool {
    matches!(op, 0 | 36 | 37 | 38 | 39)
}

#[no_mangle]
pub extern "C" fn ggml_bitset_size_rust(n: usize) -> usize {
    (n + 31) >> 5
}

#[no_mangle]
pub extern "C" fn ggml_aligned_offset_rust(buffer: *const std::ffi::c_void, offset: usize, alignment: usize) -> usize {
    debug_assert!(alignment != 0 && alignment.is_power_of_two());
    let address = buffer as usize + offset;
    let align = (alignment - (address % alignment)) % alignment;
    offset + align
}

#[no_mangle]
pub unsafe extern "C" fn ggml_get_node_buffer_id_rust(node_buffer_ids: *const i32, index: i32) -> i32 {
    if node_buffer_ids.is_null() {
        return 0;
    }
    unsafe { *node_buffer_ids.add(index as usize) }
}

#[no_mangle]
pub extern "C" fn ggml_buffer_address_less_rust(
    a_chunk: i32,
    a_offset: usize,
    b_chunk: i32,
    b_offset: usize,
) -> bool {
    if a_chunk != b_chunk {
        return a_chunk < b_chunk;
    }
    a_offset < b_offset
}

#[no_mangle]
pub extern "C" fn ggml_isinf_fp16_rust(value: u16) -> bool {
    (value & 0x7c00) == 0x7c00 && (value & 0x03ff) == 0
}

#[no_mangle]
pub extern "C" fn ggml_isnan_fp16_rust(value: u16) -> bool {
    (value & 0x7c00) == 0x7c00 && (value & 0x03ff) != 0
}

#[no_mangle]
pub extern "C" fn ggml_is_invalid_e8m0_rust(value: u8) -> bool {
    value == 0xff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locks_and_unlocks() {
        ggml_critical_section_start();
        ggml_critical_section_end();
    }

    #[test]
    fn identifies_empty_ops() {
        assert!(ggml_op_is_empty_rust(0));
        assert!(ggml_op_is_empty_rust(36));
        assert!(ggml_op_is_empty_rust(37));
        assert!(ggml_op_is_empty_rust(38));
        assert!(ggml_op_is_empty_rust(39));
        assert!(!ggml_op_is_empty_rust(1));
    }

    #[test]
    fn computes_bitset_word_count() {
        assert_eq!(ggml_bitset_size_rust(0), 0);
        assert_eq!(ggml_bitset_size_rust(1), 1);
        assert_eq!(ggml_bitset_size_rust(32), 1);
        assert_eq!(ggml_bitset_size_rust(33), 2);
    }

    #[test]
    fn computes_aligned_offsets() {
        assert_eq!(ggml_aligned_offset_rust(std::ptr::null(), 0, 32), 0);
        assert_eq!(ggml_aligned_offset_rust(std::ptr::null(), 1, 32), 32);
        assert_eq!(ggml_aligned_offset_rust(std::ptr::null(), 64, 32), 64);

        let data = [0u8; 64];
        let base = data.as_ptr() as usize;
        let expected = (16 - ((base + 3) % 16)) % 16 + 3;
        assert_eq!(ggml_aligned_offset_rust(data.as_ptr().cast(), 3, 16), expected);
    }

    #[test]
    fn reads_node_buffer_ids_with_default() {
        let ids = [7, 11, 13];
        unsafe {
            assert_eq!(ggml_get_node_buffer_id_rust(std::ptr::null(), 2), 0);
            assert_eq!(ggml_get_node_buffer_id_rust(ids.as_ptr(), 0), 7);
            assert_eq!(ggml_get_node_buffer_id_rust(ids.as_ptr(), 2), 13);
        }
    }

    #[test]
    fn compares_buffer_addresses() {
        assert!(ggml_buffer_address_less_rust(0, 99, 1, 0));
        assert!(ggml_buffer_address_less_rust(1, 4, 1, 5));
        assert!(!ggml_buffer_address_less_rust(2, 0, 1, 99));
        assert!(!ggml_buffer_address_less_rust(1, 5, 1, 5));
    }

    #[test]
    fn classifies_fp16_and_e8m0_values() {
        assert!(ggml_isinf_fp16_rust(0x7c00));
        assert!(ggml_isinf_fp16_rust(0xfc00));
        assert!(!ggml_isinf_fp16_rust(0x7c01));
        assert!(!ggml_isinf_fp16_rust(0x3c00));

        assert!(ggml_isnan_fp16_rust(0x7c01));
        assert!(ggml_isnan_fp16_rust(0x7fff));
        assert!(!ggml_isnan_fp16_rust(0x7c00));
        assert!(!ggml_isnan_fp16_rust(0x3c00));

        assert!(ggml_is_invalid_e8m0_rust(0xff));
        assert!(!ggml_is_invalid_e8m0_rust(0xfe));
    }
}
