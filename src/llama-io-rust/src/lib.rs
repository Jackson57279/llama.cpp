use std::ffi::c_void;

type WriteCallback = unsafe extern "C" fn(user_data: *mut c_void, src: *const c_void, size: usize);
type ReadCallback = unsafe extern "C" fn(user_data: *mut c_void, size: usize) -> *const u8;
type ReadToCallback = unsafe extern "C" fn(user_data: *mut c_void, dst: *mut c_void, size: usize);
type AssignStringCallback =
    unsafe extern "C" fn(user_data: *mut c_void, src: *const u8, size: usize);

#[no_mangle]
pub unsafe extern "C" fn llama_io_write_string_rust(
    user_data: *mut c_void,
    write_cb: WriteCallback,
    src: *const u8,
    size: usize,
) {
    let str_size = size as u32;
    unsafe {
        write_cb(
            user_data,
            (&str_size as *const u32).cast::<c_void>(),
            std::mem::size_of::<u32>(),
        );
    }

    if size != 0 {
        unsafe {
            write_cb(user_data, src.cast::<c_void>(), size);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_io_read_string_rust(
    user_data: *mut c_void,
    read_to_cb: ReadToCallback,
    read_cb: ReadCallback,
    output_data: *mut c_void,
    assign_cb: AssignStringCallback,
) {
    let mut str_size = 0_u32;
    unsafe {
        read_to_cb(
            user_data,
            (&mut str_size as *mut u32).cast::<c_void>(),
            std::mem::size_of::<u32>(),
        );
    }

    let data = if str_size == 0 {
        std::ptr::null()
    } else {
        unsafe { read_cb(user_data, str_size as usize) }
    };

    unsafe {
        assign_cb(output_data, data, str_size as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct WriteHarness {
        bytes: Vec<u8>,
    }

    unsafe extern "C" fn write_cb(user_data: *mut c_void, src: *const c_void, size: usize) {
        let harness = unsafe { &mut *(user_data as *mut WriteHarness) };
        let bytes = unsafe { std::slice::from_raw_parts(src.cast::<u8>(), size) };
        harness.bytes.extend_from_slice(bytes);
    }

    #[test]
    fn write_string_prefixes_little_endian_u32_length() {
        let mut harness = WriteHarness::default();

        unsafe {
            llama_io_write_string_rust(
                (&mut harness as *mut WriteHarness).cast::<c_void>(),
                write_cb,
                b"abc".as_ptr(),
                3,
            );
        }

        assert_eq!(harness.bytes, [3, 0, 0, 0, b'a', b'b', b'c']);
    }
}
