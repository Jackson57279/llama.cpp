use std::ffi::{c_char, c_int, CStr};
use std::slice;
use std::time::{SystemTime, UNIX_EPOCH};

#[repr(C)]
pub struct Utf8ParseResult {
    codepoint: u32,
    bytes_consumed: usize,
    status: i32,
}

#[repr(C)]
pub struct UnicodeString {
    data: *mut u8,
    len: usize,
}

#[repr(C)]
pub struct StringView {
    data: *const u8,
    len: usize,
}

#[repr(C)]
pub struct StringList {
    data: *mut UnicodeString,
    len: usize,
}

#[repr(C)]
pub struct GgufSplitInfo {
    prefix: UnicodeString,
    tag: UnicodeString,
    index: c_int,
    count: c_int,
}

const SUCCESS: i32 = 0;
const INCOMPLETE: i32 = 1;
const INVALID: i32 = 2;

#[no_mangle]
pub extern "C" fn llama_common_utf8_sequence_length(first_byte: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(first_byte >> 4) as usize]
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_utf8_is_complete(data: *const u8, len: usize) -> bool {
    if data.is_null() && len != 0 {
        return false;
    }
    utf8_is_complete(slice::from_raw_parts(data, len))
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_parse_utf8_codepoint(
    data: *const u8,
    len: usize,
    offset: usize,
) -> Utf8ParseResult {
    if data.is_null() && len != 0 {
        return result(INVALID, 0, 0);
    }
    parse_utf8_codepoint(slice::from_raw_parts(data, len), offset)
}

#[no_mangle]
pub extern "C" fn llama_common_unicode_cpt_to_utf8(cpt: u32) -> UnicodeString {
    match unicode_cpt_to_utf8(cpt) {
        Some(bytes) => into_ffi(bytes),
        None => UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        },
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_unicode_cpts_to_utf8_rust(
    data: *const u32,
    len: usize,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    unicode_cpts_to_utf8(slice::from_raw_parts(data, len))
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_token_type_to_string_rust(token_type: c_int) -> *const c_char {
    match token_type {
        0 => c"eof".as_ptr(),
        1 => c"text".as_ptr(),
        2 => c"numeric_literal".as_ptr(),
        3 => c"string_literal".as_ptr(),
        4 => c"identifier".as_ptr(),
        5 => c"equals".as_ptr(),
        6 => c"open_paren".as_ptr(),
        7 => c"close_paren".as_ptr(),
        8 => c"open_statement".as_ptr(),
        9 => c"close_statement".as_ptr(),
        10 => c"open_expression".as_ptr(),
        11 => c"close_expression".as_ptr(),
        12 => c"open_square_bracket".as_ptr(),
        13 => c"close_square_bracket".as_ptr(),
        14 => c"open_curly_bracket".as_ptr(),
        15 => c"close_curly_bracket".as_ptr(),
        16 => c"comma".as_ptr(),
        17 => c"dot".as_ptr(),
        18 => c"colon".as_ptr(),
        19 => c"pipe".as_ptr(),
        20 => c"call_operator".as_ptr(),
        21 => c"additive_binary_operator".as_ptr(),
        22 => c"multiplicative_binary_operator".as_ptr(),
        23 => c"comparison_binary_operator".as_ptr(),
        24 => c"unary_operator".as_ptr(),
        25 => c"comment".as_ptr(),
        _ => c"unknown".as_ptr(),
    }
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_is_word_rust(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_is_integer_rust(c: u8) -> bool {
    c.is_ascii_digit()
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_http_show_masked_url_rust(
    scheme: *const u8,
    scheme_len: usize,
    has_user: bool,
    host: *const u8,
    host_len: usize,
    path: *const u8,
    path_len: usize,
) -> UnicodeString {
    let Some(scheme) = byte_slice(scheme, scheme_len) else {
        return into_ffi(Vec::new());
    };
    let Some(host) = byte_slice(host, host_len) else {
        return into_ffi(Vec::new());
    };
    let Some(path) = byte_slice(path, path_len) else {
        return into_ffi(Vec::new());
    };
    http_show_masked_url(scheme, has_user, host, path)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_peak_source_rust(
    source: *const u8,
    source_len: usize,
    pos: usize,
    max_peak_chars: usize,
) -> UnicodeString {
    let Some(source) = byte_slice(source, source_len) else {
        return into_ffi(b"(no source available)".to_vec());
    };
    peak_source(source, pos, max_peak_chars)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_fmt_error_with_source_rust(
    tag: *const u8,
    tag_len: usize,
    msg: *const u8,
    msg_len: usize,
    source: *const u8,
    source_len: usize,
    pos: usize,
) -> UnicodeString {
    let Some(tag) = byte_slice(tag, tag_len) else {
        return into_ffi(Vec::new());
    };
    let Some(msg) = byte_slice(msg, msg_len) else {
        return into_ffi(Vec::new());
    };
    let Some(source) = byte_slice(source, source_len) else {
        return into_ffi(Vec::new());
    };
    fmt_error_with_source(tag, msg, source, pos)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_unicode_string_free(value: UnicodeString) {
    if !value.data.is_null() {
        drop(Vec::from_raw_parts(value.data, value.len, value.len));
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_repeat_rust(
    data: *const u8,
    len: usize,
    n: usize,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if len == 0 {
        return string_repeat(&[], n);
    }
    string_repeat(slice::from_raw_parts(data, len), n)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_int_vec_to_string_rust(
    data: *const i32,
    len: usize,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if len == 0 {
        return int_vec_to_string(&[]);
    }
    int_vec_to_string(slice::from_raw_parts(data, len))
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_trim_rust(
    data: *const u8,
    len: usize,
    mode: c_int,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if len == 0 {
        return into_ffi(Vec::new());
    }
    string_trim(slice::from_raw_parts(data, len), mode)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_prefix_len_rust(
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
) -> usize {
    let Some((left, right)) = byte_pair(left, left_len, right, right_len) else {
        return 0;
    };
    common_prefix_len(left, right)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_suffix_len_rust(
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
) -> usize {
    let Some((left, right)) = byte_pair(left, left_len, right, right_len) else {
        return 0;
    };
    common_suffix_len(left, right)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_replace_all_rust(
    data: *const u8,
    len: usize,
    search: *const u8,
    search_len: usize,
    replace: *const u8,
    replace_len: usize,
) -> UnicodeString {
    let Some((data, search)) = byte_pair(data, len, search, search_len) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    if replace.is_null() && replace_len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let replace = if replace_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(replace, replace_len)
    };
    string_replace_all(data, search, replace)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_regex_escape_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if len == 0 {
        return into_ffi(Vec::new());
    }
    regex_escape(slice::from_raw_parts(data, len))
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_join_rust(
    values: *const StringView,
    values_len: usize,
    separator: *const u8,
    separator_len: usize,
) -> UnicodeString {
    if values.is_null() && values_len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if separator.is_null() && separator_len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }

    let values = if values_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(values, values_len)
    };
    let separator = if separator_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(separator, separator_len)
    };
    string_join(values, separator)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_split_rust(
    data: *const u8,
    len: usize,
    delimiter: *const u8,
    delimiter_len: usize,
) -> StringList {
    let Some((data, delimiter)) = byte_pair(data, len, delimiter, delimiter_len) else {
        return StringList {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    string_split(data, delimiter)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_parse_csv_row_rust(
    data: *const u8,
    len: usize,
) -> StringList {
    if data.is_null() && len != 0 {
        return StringList {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if len == 0 {
        return into_string_list(vec![Vec::new()]);
    }
    parse_csv_row(slice::from_raw_parts(data, len))
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_trim_trailing_space_view_rust(
    data: *const u8,
    len: usize,
    max: c_int,
) -> StringView {
    let Some(data) = byte_slice(data, len) else {
        return StringView {
            data: std::ptr::null(),
            len: 0,
        };
    };
    trim_trailing_space_view(data, max)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_trim_leading_space_view_rust(
    data: *const u8,
    len: usize,
    max: c_int,
) -> StringView {
    let Some(data) = byte_slice(data, len) else {
        return StringView {
            data: std::ptr::null(),
            len: 0,
        };
    };
    trim_leading_space_view(data, max)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_process_escapes_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if len == 0 {
        return into_ffi(Vec::new());
    }
    string_process_escapes(slice::from_raw_parts(data, len))
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_clean_file_name_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if len == 0 {
        return into_ffi(Vec::new());
    }
    clean_file_name(slice::from_raw_parts(data, len))
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_diff_rust(
    last: *const u8,
    last_len: usize,
    current: *const u8,
    current_len: usize,
    status: *mut c_int,
) -> UnicodeString {
    if !status.is_null() {
        *status = -1;
    }
    let Some((last, current)) = byte_pair(last, last_len, current, current_len) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };

    let (result, result_status) = string_diff(last, current);
    if !status.is_null() {
        *status = result_status;
    }
    result
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_starts_with_rust(
    data: *const u8,
    len: usize,
    prefix: *const u8,
    prefix_len: usize,
) -> bool {
    let Some((data, prefix)) = byte_pair(data, len, prefix, prefix_len) else {
        return false;
    };
    data.starts_with(prefix)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_ends_with_rust(
    data: *const u8,
    len: usize,
    suffix: *const u8,
    suffix_len: usize,
) -> bool {
    let Some((data, suffix)) = byte_pair(data, len, suffix, suffix_len) else {
        return false;
    };
    data.ends_with(suffix)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_bytes_equal_rust(
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
) -> bool {
    let Some((left, right)) = byte_pair(left, left_len, right, right_len) else {
        return false;
    };
    left == right
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_find_partial_stop_rust(
    data: *const u8,
    len: usize,
    stop: *const u8,
    stop_len: usize,
) -> usize {
    let Some((data, stop)) = byte_pair(data, len, stop, stop_len) else {
        return usize::MAX;
    };
    string_find_partial_stop(data, stop)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_remove_suffix_len_rust(
    data: *const u8,
    len: usize,
    suffix: *const u8,
    suffix_len: usize,
) -> usize {
    let Some((data, suffix)) = byte_pair(data, len, suffix, suffix_len) else {
        return usize::MAX;
    };
    if data.ends_with(suffix) {
        data.len() - suffix.len()
    } else {
        usize::MAX
    }
}

fn string_find_partial_stop(data: &[u8], stop: &[u8]) -> usize {
    if !data.is_empty() && !stop.is_empty() {
        let max_len = data.len().min(stop.len());
        let last_char = data[data.len() - 1];
        for len in (1..=max_len).rev() {
            if stop[len - 1] == last_char && data.ends_with(&stop[..len]) {
                return data.len() - len;
            }
        }
    }
    usize::MAX
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_glob_match_rust(
    pattern: *const u8,
    pattern_len: usize,
    str: *const u8,
    str_len: usize,
) -> bool {
    let Some((pattern, str)) = byte_pair(pattern, pattern_len, str, str_len) else {
        return false;
    };
    glob_match(c_string_bytes(pattern), c_string_bytes(str))
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_fs_validate_filename_rust(
    filename: *const u8,
    filename_len: usize,
    allow_subdirs: bool,
) -> bool {
    if filename.is_null() && filename_len != 0 {
        return false;
    }
    let filename = if filename_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(filename, filename_len)
    };
    fs_validate_filename(filename, allow_subdirs)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_lstrip_chars_rust(
    data: *const u8,
    len: usize,
    chars: *const c_char,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if chars.is_null() {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let data = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(data, len)
    };
    string_lstrip_chars(data, CStr::from_ptr(chars).to_bytes())
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_rstrip_chars_rust(
    data: *const u8,
    len: usize,
    chars: *const c_char,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    if chars.is_null() {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let data = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(data, len)
    };
    string_rstrip_chars(data, CStr::from_ptr(chars).to_bytes())
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_break_str_into_lines_rust(
    data: *const u8,
    len: usize,
    max_char_per_line: usize,
) -> StringList {
    if data.is_null() && len != 0 {
        return StringList {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let data = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(data, len)
    };
    break_str_into_lines(data, max_char_per_line)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_rm_leading_dashes_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let data = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(data, len)
    };
    rm_leading_dashes(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_until_common_prefix_rust(
    full: *const u8,
    full_len: usize,
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
) -> UnicodeString {
    let Some((full, left)) = byte_pair(full, full_len, left, left_len) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    if right.is_null() && right_len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let right = if right_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(right, right_len)
    };
    until_common_prefix(full, left, right)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_after_common_suffix_rust(
    full: *const u8,
    full_len: usize,
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
) -> UnicodeString {
    let Some((full, left)) = byte_pair(full, full_len, left, left_len) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    if right.is_null() && right_len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let right = if right_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(right, right_len)
    };
    after_common_suffix(full, left, right)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_json_ensure_ascii_preserving_format_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    if data.is_null() && len != 0 {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    }
    let data = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(data, len)
    };
    json_ensure_ascii_preserving_format(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_normalize_quotes_to_json_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    let Some(data) = byte_slice(data, len) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    normalize_quotes_to_json(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_string_list_free(value: StringList) {
    if !value.data.is_null() {
        let items = Vec::from_raw_parts(value.data, value.len, value.len);
        for item in items {
            llama_common_unicode_string_free(item);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_ascii_case_equal(
    a: *const c_char,
    b: *const c_char,
) -> c_int {
    ascii_case_equal(a, b) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_opt_get_optimizer_rust(name: *const c_char) -> c_int {
    if ascii_case_equal(name, c"adamw".as_ptr()) {
        0
    } else if ascii_case_equal(name, c"sgd".as_ptr()) {
        1
    } else {
        2
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_arg_is_truthy_rust(value: *const c_char) -> c_int {
    matches_cstr(value, &[b"on", b"enabled", b"true", b"1"]) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_arg_is_falsey_rust(value: *const c_char) -> c_int {
    matches_cstr(value, &[b"off", b"disabled", b"false", b"0"]) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_arg_is_autoy_rust(value: *const c_char) -> c_int {
    matches_cstr(value, &[b"auto", b"-1"]) as c_int
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_parse_bool_value_rust(value: *const c_char) -> c_int {
    if matches_cstr(value, &[b"on", b"enabled", b"true", b"1"]) {
        1
    } else if matches_cstr(value, &[b"off", b"disabled", b"false", b"0"]) {
        0
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_parse_bool_arg_rust(
    neg_args: *const StringView,
    neg_args_len: usize,
    key: *const u8,
    key_len: usize,
    value: *const u8,
    value_len: usize,
) -> UnicodeString {
    if (neg_args.is_null() && neg_args_len != 0) || (key.is_null() && key_len != 0) {
        return into_ffi(Vec::new());
    }
    let Some(value) = byte_slice(value, value_len) else {
        return into_ffi(Vec::new());
    };
    let neg_args = if neg_args_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(neg_args, neg_args_len)
    };
    let key = if key_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(key, key_len)
    };
    parse_bool_arg(neg_args, key, value)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_read_file_rust(
    path: *const u8,
    path_len: usize,
) -> UnicodeString {
    let Some(path) = byte_slice(path, path_len) else {
        return null_string();
    };
    read_file(path)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_parse_cpu_mask_rust(
    mask: *const u8,
    mask_len: usize,
    boolmask: *mut bool,
    boolmask_len: usize,
) -> c_int {
    if (mask.is_null() && mask_len != 0) || (boolmask.is_null() && boolmask_len != 0) {
        return -2;
    }
    let mask = if mask_len == 0 {
        &[]
    } else {
        slice::from_raw_parts(mask, mask_len)
    };
    let boolmask = if boolmask_len == 0 {
        &mut []
    } else {
        slice::from_raw_parts_mut(boolmask, boolmask_len)
    };
    parse_cpu_mask(mask, boolmask)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_get_line_col_rust(
    data: *const u8,
    len: usize,
    pos: usize,
) -> UnicodeString {
    let Some(data) = byte_slice(data, len) else {
        return into_ffi(b"line 1, column 1".to_vec());
    };
    get_line_col(data, pos)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_chat_tool_choice_parse_oaicompat_rust(
    value: *const c_char,
) -> c_int {
    if value.is_null() {
        return -1;
    }

    match CStr::from_ptr(value).to_bytes() {
        b"auto" => 0,
        b"required" => 1,
        b"none" => 2,
        _ => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_kv_cache_type_from_str_rust(value: *const c_char) -> c_int {
    if value.is_null() {
        return -1;
    }

    match CStr::from_ptr(value).to_bytes() {
        b"f32" => 0,
        b"f16" => 1,
        b"q4_0" => 2,
        b"q4_1" => 3,
        b"q5_0" => 6,
        b"q5_1" => 7,
        b"q8_0" => 8,
        b"iq4_nl" => 20,
        b"bf16" => 30,
        _ => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_embd_normalize_rust(
    inp: *const f32,
    out: *mut f32,
    n: c_int,
    embd_norm: c_int,
) {
    if n <= 0 || inp.is_null() || out.is_null() {
        return;
    }

    let inp = slice::from_raw_parts(inp, n as usize);
    let out = slice::from_raw_parts_mut(out, n as usize);
    embd_normalize(inp, out, embd_norm);
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_embd_similarity_cos_rust(
    embd1: *const f32,
    embd2: *const f32,
    n: c_int,
) -> f32 {
    if n <= 0 {
        return 1.0;
    }
    if embd1.is_null() || embd2.is_null() {
        return 0.0;
    }

    let embd1 = slice::from_raw_parts(embd1, n as usize);
    let embd2 = slice::from_raw_parts(embd2, n as usize);
    embd_similarity_cos(embd1, embd2)
}

#[no_mangle]
pub extern "C" fn llama_common_get_all_kv_cache_types_rust() -> *const c_char {
    c"f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1".as_ptr()
}

#[no_mangle]
pub extern "C" fn llama_common_bool_to_string_rust(value: bool) -> *const c_char {
    if value {
        c"true".as_ptr()
    } else {
        c"false".as_ptr()
    }
}

#[no_mangle]
pub extern "C" fn llama_common_default_thread_count_windows_rust(
    num_physical_cores: c_int,
    default_threads: c_int,
) -> c_int {
    default_thread_count_windows(num_physical_cores, default_threads)
}

#[no_mangle]
pub extern "C" fn llama_common_default_thread_count_rust(n_threads: u32) -> c_int {
    default_thread_count(n_threads)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_lr_opt_init_rust(
    lr0: f32,
    lr_min: f32,
    decay_epochs: f32,
    epochs: u32,
    out_decay_epochs: *mut f32,
    out_scale_epoch: *mut f32,
) {
    if out_decay_epochs.is_null() || out_scale_epoch.is_null() {
        return;
    }
    let (decay_epochs, scale_epoch) = lr_opt_init(lr0, lr_min, decay_epochs, epochs);
    *out_decay_epochs = decay_epochs;
    *out_scale_epoch = scale_epoch;
}

#[no_mangle]
pub extern "C" fn llama_common_lr_opt_get_lr_rust(
    lr0: f32,
    lr_min: f32,
    decay_epochs: f32,
    scale_epoch: f32,
    epoch: f32,
) -> f32 {
    lr_opt_get_lr(lr0, lr_min, decay_epochs, scale_epoch, epoch)
}

#[no_mangle]
pub extern "C" fn llama_common_log_level_to_verbosity_rust(level: c_int) -> c_int {
    match level {
        1 => 4,
        2 => 3,
        3 => 2,
        4 => 1,
        5 => 3,
        _ => 0,
    }
}

#[no_mangle]
pub extern "C" fn llama_common_time_us_rust() -> i64 {
    time_us()
}

#[no_mangle]
pub extern "C" fn llama_common_sampler_prob_desc_rust(left: f32, right: f32) -> bool {
    sampler_prob_desc(left, right)
}

#[no_mangle]
pub extern "C" fn llama_common_marker_is_opener_rust(c: u8) -> bool {
    marker_is_opener(c)
}

#[no_mangle]
pub extern "C" fn llama_common_marker_is_closer_rust(opener: u8, c: u8) -> bool {
    marker_is_closer(opener, c)
}

#[no_mangle]
pub extern "C" fn llama_common_content_is_always_wrapped_rust(
    mode: c_int,
    start_len: usize,
    end_len: usize,
) -> bool {
    content_is_always_wrapped(mode, start_len, end_len)
}

#[no_mangle]
pub extern "C" fn llama_common_cpu_has_hybrid_bit_rust(edx: u32) -> bool {
    cpu_has_hybrid_bit(edx)
}

#[no_mangle]
pub extern "C" fn llama_common_cpu_is_intel_atom_core_type_rust(eax: u32) -> bool {
    cpu_is_intel_atom_core_type(eax)
}

#[no_mangle]
pub extern "C" fn llama_common_any_terminal_rust(stdout_is_tty: bool, stderr_is_tty: bool) -> bool {
    stdout_is_tty || stderr_is_tty
}

#[no_mangle]
pub extern "C" fn llama_common_arg_has_env_value_rust(
    has_env: bool,
    has_neg_env_value: bool,
    has_env_value: bool,
) -> bool {
    arg_has_env_value(has_env, has_neg_env_value, has_env_value)
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_int_is_odd_rust(value: i64) -> bool {
    value % 2 != 0
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_int_is_even_rust(value: i64) -> bool {
    value % 2 == 0
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_int_abs_rust(value: i64) -> i64 {
    if value < 0 {
        value.wrapping_neg()
    } else {
        value
    }
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_float_abs_rust(value: f64) -> f64 {
    if value < 0.0 {
        -value
    } else {
        value
    }
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_is_false_rust(is_bool: bool, value: bool) -> bool {
    is_bool && !value
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_is_true_rust(is_bool: bool, value: bool) -> bool {
    is_bool && value
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_is_defined_rust(is_undefined: bool) -> bool {
    !is_undefined
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_compare_f64_rust(left: f64, right: f64, op: i32) -> bool {
    match op {
        0 => left == right,
        1 => left >= right,
        2 => left > right,
        3 => left < right,
        4 => left != right,
        5 => left <= right,
        _ => false,
    }
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_arithmetic_f64_rust(left: f64, right: f64, op: i32) -> f64 {
    match op {
        0 => left + right,
        1 => left - right,
        2 => left * right,
        3 => left / right,
        4 => left % right,
        _ => f64::NAN,
    }
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_compare_bool_rust(left: bool, right: bool, op: i32) -> bool {
    match op {
        0 => left == right,
        4 => left != right,
        _ => false,
    }
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_bool_not_rust(value: bool) -> bool {
    !value
}

#[no_mangle]
pub extern "C" fn llama_common_jinja_membership_result_rust(member: bool, is_not_in: bool) -> bool {
    if is_not_in {
        !member
    } else {
        member
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_jinja_compare_bytes_rust(
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
    op: i32,
) -> bool {
    let Some(left) = byte_slice(left, left_len) else {
        return false;
    };
    let Some(right) = byte_slice(right, right_len) else {
        return false;
    };

    match op {
        0 => left == right,
        1 => left >= right,
        2 => left > right,
        3 => left < right,
        4 => left != right,
        _ => false,
    }
}

#[no_mangle]
pub extern "C" fn llama_common_ring_buffer_is_empty_rust(size: usize) -> bool {
    size == 0
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_model_endpoint_rust(
    model_endpoint: *const u8,
    model_endpoint_len: usize,
    hf_endpoint: *const u8,
    hf_endpoint_len: usize,
) -> UnicodeString {
    let model_endpoint = if model_endpoint.is_null() {
        None
    } else {
        byte_slice(model_endpoint, model_endpoint_len)
    };
    let hf_endpoint = if hf_endpoint.is_null() {
        None
    } else {
        byte_slice(hf_endpoint, hf_endpoint_len)
    };
    model_endpoint_string(model_endpoint, hf_endpoint)
}

#[no_mangle]
pub extern "C" fn llama_common_is_http_status_ok_rust(status: c_int) -> bool {
    (200..400).contains(&status)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_read_etag_rust(data: *const u8, len: usize) -> UnicodeString {
    let Some(path) = byte_slice(data, len) else {
        return into_ffi(Vec::new());
    };
    read_etag(path)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_get_cached_ref_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    let Some(path) = byte_slice(data, len) else {
        return into_ffi(Vec::new());
    };
    get_cached_ref(path)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_safe_write_file_rust(
    path: *const u8,
    path_len: usize,
    data: *const u8,
    data_len: usize,
) -> bool {
    let Some(path) = byte_slice(path, path_len) else {
        return false;
    };
    let Some(data) = byte_slice(data, data_len) else {
        return false;
    };
    safe_write_file(path, data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_fs_is_directory_rust(
    path: *const u8,
    path_len: usize,
) -> bool {
    let Some(path) = byte_slice(path, path_len) else {
        return false;
    };
    fs_is_directory(path)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_is_lfm2_template_rust(data: *const u8, len: usize) -> bool {
    let Some(input) = byte_slice(data, len) else {
        return false;
    };
    is_lfm2_template(input)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_specialized_chat_template_rust(
    data: *const u8,
    len: usize,
) -> c_int {
    let Some(src) = byte_slice(data, len) else {
        return 0;
    };

    if contains_subslice(src, b"[SYSTEM_PROMPT]")
        && contains_subslice(src, b"[TOOL_CALLS]")
        && contains_subslice(src, b"[ARGS]")
        && !contains_subslice(src, b"[CALL_ID]")
    {
        return 1;
    }
    if contains_subslice(src, b"<|channel|>") {
        return 2;
    }
    if contains_subslice(src, b">>>all") && contains_subslice(src, b">>>${recipient}") {
        return 3;
    }
    if contains_subslice(src, b"<|tool_calls_section_begin|>")
        && contains_subslice(src, b"<|tool_call_begin|>")
    {
        return 4;
    }
    if is_lfm2_template(src) {
        return 5;
    }
    if contains_subslice(src, b"List of tools: [")
        && !contains_subslice(src, b"<|tool_list_start|>")
    {
        return 6;
    }
    if contains_subslice(src, b"<|role_sep|>")
        && contains_subslice(src, b"<|message_sep|>")
        && !contains_subslice(src, b"<|function_call|>")
    {
        return 7;
    }
    if contains_subslice(src, b"dsml_token")
        && contains_subslice(src, b"function_calls")
        && contains_subslice(src, b"DSML")
    {
        return 8;
    }
    if contains_subslice(src, b"'<|tool_call>call:'") {
        return 9;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_is_gemma4_modern_template_rust(
    data: *const u8,
    len: usize,
) -> bool {
    let Some(src) = byte_slice(data, len) else {
        return false;
    };
    contains_subslice(src, b"{#- OpenAI Chat Completions:")
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_needs_gpt_oss_channel_template_patch_rust(
    data: *const u8,
    len: usize,
) -> bool {
    let Some(src) = byte_slice(data, len) else {
        return false;
    };
    contains_subslice(src, b"<|channel|>") && contains_subslice(src, b"in message.content or")
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_needs_mistral_tool_calls_template_patch_rust(
    data: *const u8,
    len: usize,
) -> bool {
    let Some(src) = byte_slice(data, len) else {
        return false;
    };
    contains_subslice(src, b"[TOOL_CALLS]")
        && contains_subslice(src, b"if (message['content'] is none or")
}

#[no_mangle]
pub extern "C" fn llama_common_grammar_should_apply_rust(
    has_grammar: bool,
    has_reasoning_budget: bool,
    grammar_lazy: bool,
    reasoning_state: c_int,
    idle_state: c_int,
    done_state: c_int,
) -> bool {
    if !has_grammar {
        return false;
    }
    if !has_reasoning_budget || !grammar_lazy {
        return true;
    }
    reasoning_state == idle_state || reasoning_state == done_state
}

#[no_mangle]
pub extern "C" fn llama_common_grammar_is_empty_rust(
    grammar_type: c_int,
    grammar_len: usize,
) -> bool {
    grammar_type == 0 || grammar_len == 0
}

#[no_mangle]
pub extern "C" fn llama_common_grammar_needs_prefill_rust(grammar_type: c_int) -> bool {
    grammar_type == 2 || grammar_type == 3
}

#[no_mangle]
pub extern "C" fn llama_common_has_logit_bias_rust(logit_bias_len: usize) -> bool {
    logit_bias_len != 0
}

#[no_mangle]
pub extern "C" fn llama_common_speculative_has_draft_rust(
    path_len: usize,
    hf_repo_len: usize,
) -> bool {
    path_len != 0 || hf_repo_len != 0
}

#[no_mangle]
pub extern "C" fn llama_common_peg_parse_result_fail_rust(result_type: c_int) -> bool {
    result_type == 0
}

#[no_mangle]
pub extern "C" fn llama_common_peg_parse_result_need_more_input_rust(result_type: c_int) -> bool {
    result_type == 2
}

#[no_mangle]
pub extern "C" fn llama_common_peg_parse_result_success_rust(result_type: c_int) -> bool {
    result_type == 1
}

#[no_mangle]
pub extern "C" fn llama_common_peg_parse_flags_is_lenient_rust(flags: c_int) -> bool {
    flags & 1 != 0
}

#[no_mangle]
pub extern "C" fn llama_common_peg_parse_flags_is_debug_rust(flags: c_int) -> bool {
    flags & 2 != 0
}

#[no_mangle]
pub extern "C" fn llama_common_peg_char_range_contains_rust(
    start: u32,
    end: u32,
    codepoint: u32,
) -> bool {
    codepoint >= start && codepoint <= end
}

#[no_mangle]
pub extern "C" fn llama_common_chat_msg_is_empty_rust(
    content_len: usize,
    content_parts_len: usize,
    tool_calls_len: usize,
    reasoning_content_len: usize,
    tool_name_len: usize,
    tool_call_id_len: usize,
) -> bool {
    content_len == 0
        && content_parts_len == 0
        && tool_calls_len == 0
        && reasoning_content_len == 0
        && tool_name_len == 0
        && tool_call_id_len == 0
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_gguf_filename_is_model_rust(
    data: *const u8,
    len: usize,
) -> bool {
    if data.is_null() && len != 0 {
        return false;
    }
    let data = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(data, len)
    };
    gguf_filename_is_model(data)
}

#[no_mangle]
pub extern "C" fn llama_common_is_hex_digit_rust(c: u8) -> bool {
    is_hex_digit(c)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_parse_hex_escape_rust(
    data: *const u8,
    len: usize,
    pos: usize,
    hex_count: c_int,
    out_value: *mut u32,
) -> usize {
    if data.is_null() && len != 0 {
        return 0;
    }
    if out_value.is_null() {
        return 0;
    }
    let data = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(data, len)
    };
    let Some((value, consumed)) = parse_hex_escape(data, pos, hex_count) else {
        return 0;
    };
    *out_value = value;
    consumed
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_parse_char_class_char_rust(
    data: *const u8,
    len: usize,
    pos: usize,
    out_value: *mut u32,
) -> usize {
    if data.is_null() && len != 0 {
        return 0;
    }
    if out_value.is_null() {
        return 0;
    }
    let data = if len == 0 {
        &[]
    } else {
        slice::from_raw_parts(data, len)
    };
    let Some((value, consumed)) = parse_char_class_char(data, pos) else {
        return 0;
    };
    *out_value = value;
    consumed
}

#[no_mangle]
pub extern "C" fn llama_common_chat_format_name_rust(format: c_int) -> *const c_char {
    match format {
        0 => c"Content-only".as_ptr(),
        1 => c"peg-simple".as_ptr(),
        2 => c"peg-native".as_ptr(),
        3 => c"peg-gemma4".as_ptr(),
        _ => std::ptr::null(),
    }
}

#[no_mangle]
pub extern "C" fn llama_common_reasoning_format_name_rust(format: c_int) -> *const c_char {
    match format {
        0 => c"none".as_ptr(),
        1 => c"auto".as_ptr(),
        2 => c"deepseek-legacy".as_ptr(),
        3 => c"deepseek".as_ptr(),
        _ => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_reasoning_format_from_name_rust(
    format: *const c_char,
) -> c_int {
    if format.is_null() {
        return -1;
    }
    match CStr::from_ptr(format).to_bytes() {
        b"none" => 0,
        b"auto" => 1,
        b"deepseek-legacy" => 2,
        b"deepseek" => 3,
        _ => -1,
    }
}

#[no_mangle]
pub extern "C" fn llama_common_sampler_type_to_chr_rust(sampler_type: c_int) -> c_char {
    match sampler_type {
        1 => b'd' as c_char,
        2 => b'k' as c_char,
        3 => b'p' as c_char,
        4 => b'm' as c_char,
        6 => b'y' as c_char,
        7 => b't' as c_char,
        8 => b'x' as c_char,
        9 => b'i' as c_char,
        10 => b'e' as c_char,
        11 => b's' as c_char,
        12 => b'a' as c_char,
        _ => b'?' as c_char,
    }
}

#[no_mangle]
pub extern "C" fn llama_common_sampler_type_to_str_rust(sampler_type: c_int) -> *const c_char {
    match sampler_type {
        1 => c"dry".as_ptr(),
        2 => c"top_k".as_ptr(),
        3 => c"top_p".as_ptr(),
        4 => c"min_p".as_ptr(),
        6 => c"typ_p".as_ptr(),
        7 => c"temperature".as_ptr(),
        8 => c"xtc".as_ptr(),
        9 => c"infill".as_ptr(),
        10 => c"penalties".as_ptr(),
        11 => c"top_n_sigma".as_ptr(),
        12 => c"adaptive_p".as_ptr(),
        _ => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_sampler_type_from_name_rust(
    name: *const c_char,
    allow_alt_names: bool,
) -> c_int {
    if name.is_null() {
        return -1;
    }
    sampler_type_from_name(CStr::from_ptr(name).to_bytes(), allow_alt_names)
}

#[no_mangle]
pub extern "C" fn llama_common_sampler_type_from_chr_rust(name: c_char) -> c_int {
    sampler_type_from_chr(name as u8)
}

#[no_mangle]
pub extern "C" fn llama_common_speculative_type_to_str_rust(
    speculative_type: c_int,
) -> *const c_char {
    match speculative_type {
        0 => c"none".as_ptr(),
        1 => c"draft".as_ptr(),
        2 => c"eagle3".as_ptr(),
        3 => c"ngram_simple".as_ptr(),
        4 => c"ngram_map_k".as_ptr(),
        5 => c"ngram_map_k4v".as_ptr(),
        6 => c"ngram_mod".as_ptr(),
        7 => c"ngram_cache".as_ptr(),
        _ => std::ptr::null(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_speculative_type_from_name_rust(
    name: *const c_char,
) -> c_int {
    if name.is_null() {
        return -1;
    }
    match CStr::from_ptr(name).to_bytes() {
        b"none" => 0,
        b"draft" => 1,
        b"eagle3" => 2,
        b"ngram_simple" => 3,
        b"ngram_map_k" => 4,
        b"ngram_map_k4v" => 5,
        b"ngram_mod" => 6,
        b"ngram_cache" => 7,
        _ => -1,
    }
}

#[no_mangle]
pub extern "C" fn llama_common_peg_parse_result_type_name_rust(
    result_type: c_int,
) -> *const c_char {
    match result_type {
        0 => c"fail".as_ptr(),
        1 => c"success".as_ptr(),
        2 => c"need_more_input".as_ptr(),
        _ => c"unknown".as_ptr(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_peg_rule_name_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    let Some(data) = byte_slice(data, len) else {
        return into_ffi(Vec::new());
    };
    peg_rule_name(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_gbnf_format_literal_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    let Some(data) = byte_slice(data, len) else {
        return into_ffi(b"\"\"".to_vec());
    };
    gbnf_format_literal(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_gbnf_build_repetition_rust(
    item_rule: *const u8,
    item_rule_len: usize,
    min_items: c_int,
    max_items: c_int,
    separator_rule: *const u8,
    separator_rule_len: usize,
) -> UnicodeString {
    let Some(item_rule) = byte_slice(item_rule, item_rule_len) else {
        return into_ffi(Vec::new());
    };
    let Some(separator_rule) = byte_slice(separator_rule, separator_rule_len) else {
        return into_ffi(Vec::new());
    };
    gbnf_build_repetition(item_rule, min_items, max_items, separator_rule)
}

#[no_mangle]
pub extern "C" fn llama_common_gbnf_escape_char_class_rust(c: u32) -> UnicodeString {
    gbnf_escape_char_class(c)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_gbnf_is_reserved_name_rust(
    data: *const u8,
    len: usize,
) -> bool {
    byte_slice(data, len)
        .map(gbnf_is_reserved_name)
        .unwrap_or(false)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_gbnf_ref_name_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    let Some(data) = byte_slice(data, len) else {
        return into_ffi(b"ref".to_vec());
    };
    gbnf_ref_name(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_hf_folder_name_to_repo_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    let Some(data) = byte_slice(data, len) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    hf_folder_name_to_repo(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_hf_repo_to_folder_name_rust(
    data: *const u8,
    len: usize,
) -> UnicodeString {
    let Some(data) = byte_slice(data, len) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    hf_repo_to_folder_name(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_hf_make_old_cache_filename_rust(
    owner: *const u8,
    owner_len: usize,
    repo: *const u8,
    repo_len: usize,
    filename: *const u8,
    filename_len: usize,
) -> UnicodeString {
    let Some((owner, repo)) = byte_pair(owner, owner_len, repo, repo_len) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    let Some(filename) = byte_slice(filename, filename_len) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    hf_make_old_cache_filename(owner, repo, filename)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_json_brace_depth_rust(data: *const u8, len: usize) -> c_int {
    let Some(data) = byte_slice(data, len) else {
        return 0;
    };
    json_brace_depth(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_hf_is_valid_repo_id_rust(
    data: *const u8,
    len: usize,
) -> bool {
    byte_slice(data, len)
        .map(hf_is_valid_repo_id)
        .unwrap_or(false)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_hf_is_valid_token_rust(data: *const u8, len: usize) -> bool {
    byte_slice(data, len)
        .map(hf_is_valid_token)
        .unwrap_or(false)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_hf_is_valid_commit_rust(data: *const u8, len: usize) -> bool {
    byte_slice(data, len)
        .map(|data| is_hex_string(data, 40))
        .unwrap_or(false)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_hf_is_valid_oid_rust(data: *const u8, len: usize) -> bool {
    byte_slice(data, len)
        .map(|data| is_hex_string(data, 40) || is_hex_string(data, 64))
        .unwrap_or(false)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_hf_parse_manifest_name_rust(
    data: *const u8,
    len: usize,
) -> StringList {
    let Some(data) = byte_slice(data, len) else {
        return StringList {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    hf_parse_manifest_name(data)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_gguf_split_info_rust(
    data: *const u8,
    len: usize,
    extract_tag: bool,
) -> GgufSplitInfo {
    let Some(data) = byte_slice(data, len) else {
        return empty_gguf_split_info();
    };
    gguf_split_info(data, extract_tag)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_gguf_extract_quant_bits_rust(
    data: *const u8,
    len: usize,
) -> c_int {
    let Some(data) = byte_slice(data, len) else {
        return 0;
    };
    gguf_extract_quant_bits(data)
}

unsafe fn matches_cstr(value: *const c_char, expected: &[&[u8]]) -> bool {
    if value.is_null() {
        return false;
    }
    let value = CStr::from_ptr(value).to_bytes();
    expected.iter().any(|item| value == *item)
}

fn is_truthy_bytes(value: &[u8]) -> bool {
    matches!(value, b"on" | b"enabled" | b"true" | b"1")
}

unsafe fn ascii_case_equal(a: *const c_char, b: *const c_char) -> bool {
    if a.is_null() || b.is_null() {
        return false;
    }

    let a = CStr::from_ptr(a).to_bytes();
    let b = CStr::from_ptr(b).to_bytes();
    if a.len() != b.len() {
        return false;
    }

    a.iter()
        .zip(b.iter())
        .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
}

fn parse_utf8_codepoint(input: &[u8], offset: usize) -> Utf8ParseResult {
    if offset >= input.len() {
        return result(INCOMPLETE, 0, 0);
    }

    let first = input[offset];
    if (first & 0x80) == 0 {
        return result(SUCCESS, first as u32, 1);
    }

    if (first & 0x40) == 0 {
        return result(INVALID, 0, 0);
    }

    if (first & 0x20) == 0 {
        if offset + 1 >= input.len() {
            return result(INCOMPLETE, 0, 0);
        }
        if (input[offset + 1] & 0xC0) != 0x80 {
            return result(INVALID, 0, 0);
        }
        let codepoint = (((first & 0x1F) as u32) << 6) | ((input[offset + 1] & 0x3F) as u32);
        return result(SUCCESS, codepoint, 2);
    }

    if (first & 0x10) == 0 {
        if offset + 2 >= input.len() {
            return result(INCOMPLETE, 0, 0);
        }
        if (input[offset + 1] & 0xC0) != 0x80 || (input[offset + 2] & 0xC0) != 0x80 {
            return result(INVALID, 0, 0);
        }
        let codepoint = (((first & 0x0F) as u32) << 12)
            | (((input[offset + 1] & 0x3F) as u32) << 6)
            | ((input[offset + 2] & 0x3F) as u32);
        return result(SUCCESS, codepoint, 3);
    }

    if (first & 0x08) == 0 {
        if offset + 3 >= input.len() {
            return result(INCOMPLETE, 0, 0);
        }
        if (input[offset + 1] & 0xC0) != 0x80
            || (input[offset + 2] & 0xC0) != 0x80
            || (input[offset + 3] & 0xC0) != 0x80
        {
            return result(INVALID, 0, 0);
        }
        let codepoint = (((first & 0x07) as u32) << 18)
            | (((input[offset + 1] & 0x3F) as u32) << 12)
            | (((input[offset + 2] & 0x3F) as u32) << 6)
            | ((input[offset + 3] & 0x3F) as u32);
        return result(SUCCESS, codepoint, 4);
    }

    result(INVALID, 0, 0)
}

fn utf8_is_complete(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    for i in 1..=bytes.len().min(4) {
        let c = bytes[bytes.len() - i];
        if (c & 0xC0) != 0x80 {
            let expected = if c >= 0xF0 {
                4
            } else if c >= 0xE0 {
                3
            } else if c >= 0xC0 {
                2
            } else {
                1
            };
            return i >= expected;
        }
    }
    false
}

fn unicode_cpt_to_utf8(cpt: u32) -> Option<Vec<u8>> {
    let mut result = Vec::with_capacity(4);

    if cpt <= 0x7F {
        result.push(cpt as u8);
    } else if cpt <= 0x7FF {
        result.push(0xC0 | ((cpt >> 6) & 0x1F) as u8);
        result.push(0x80 | (cpt & 0x3F) as u8);
    } else if cpt <= 0xFFFF {
        result.push(0xE0 | ((cpt >> 12) & 0x0F) as u8);
        result.push(0x80 | ((cpt >> 6) & 0x3F) as u8);
        result.push(0x80 | (cpt & 0x3F) as u8);
    } else if cpt <= 0x10FFFF {
        result.push(0xF0 | ((cpt >> 18) & 0x07) as u8);
        result.push(0x80 | ((cpt >> 12) & 0x3F) as u8);
        result.push(0x80 | ((cpt >> 6) & 0x3F) as u8);
        result.push(0x80 | (cpt & 0x3F) as u8);
    } else {
        return None;
    }

    Some(result)
}

fn unicode_cpts_to_utf8(cpts: &[u32]) -> UnicodeString {
    let mut result = Vec::with_capacity(cpts.len());
    for &cpt in cpts {
        let Some(bytes) = unicode_cpt_to_utf8(cpt) else {
            return UnicodeString {
                data: std::ptr::null_mut(),
                len: 0,
            };
        };
        result.extend_from_slice(&bytes);
    }
    into_ffi(result)
}

fn http_show_masked_url(scheme: &[u8], has_user: bool, host: &[u8], path: &[u8]) -> UnicodeString {
    let mut result = Vec::with_capacity(
        scheme.len() + 3 + usize::from(has_user) * b"****:****@".len() + host.len() + path.len(),
    );
    result.extend_from_slice(scheme);
    result.extend_from_slice(b"://");
    if has_user {
        result.extend_from_slice(b"****:****@");
    }
    result.extend_from_slice(host);
    result.extend_from_slice(path);
    into_ffi(result)
}

fn peak_source(source: &[u8], pos: usize, max_peak_chars: usize) -> UnicodeString {
    if source.is_empty() {
        return into_ffi(b"(no source available)".to_vec());
    }

    let start = pos.saturating_sub(max_peak_chars);
    let end = source.len().min(pos.saturating_add(max_peak_chars));
    let snippet = if start <= end && start <= source.len() {
        &source[start..end]
    } else {
        &[]
    };

    let mut output = Vec::with_capacity(snippet.len() + 16 + pos.saturating_sub(start));
    output.extend_from_slice(b"...");
    for &c in snippet {
        if c == b'\n' {
            output.extend_from_slice("↵".as_bytes());
        } else {
            output.push(c);
        }
    }
    output.extend_from_slice(b"...\n");
    output.resize(output.len() + pos.saturating_sub(start) + 3, b' ');
    output.push(b'^');
    into_ffi(output)
}

fn fmt_error_with_source(tag: &[u8], msg: &[u8], source: &[u8], pos: usize) -> UnicodeString {
    let peak = peak_source(source, pos, 40);
    let peak = unsafe { Vec::from_raw_parts(peak.data, peak.len, peak.len) };

    let mut result = Vec::with_capacity(tag.len() + msg.len() + peak.len() + 3);
    result.extend_from_slice(tag);
    result.extend_from_slice(b": ");
    result.extend_from_slice(msg);
    result.push(b'\n');
    result.extend_from_slice(&peak);
    into_ffi(result)
}

fn string_repeat(input: &[u8], n: usize) -> UnicodeString {
    if input.is_empty() || n == 0 {
        return into_ffi(Vec::new());
    }

    let Some(capacity) = input.len().checked_mul(n) else {
        return UnicodeString {
            data: std::ptr::null_mut(),
            len: 0,
        };
    };
    let mut result = Vec::with_capacity(capacity);
    for _ in 0..n {
        result.extend_from_slice(input);
    }
    into_ffi(result)
}

fn int_vec_to_string(values: &[i32]) -> UnicodeString {
    let mut result = String::from("[ ");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            result.push_str(", ");
        }
        result.push_str(&value.to_string());
    }
    result.push_str(" ]");
    into_ffi(result.into_bytes())
}

fn string_trim(input: &[u8], mode: c_int) -> UnicodeString {
    let mut start = 0;
    let mut end = input.len();

    if mode == 0 || mode == 1 {
        while start < end && input[start].is_ascii_whitespace() {
            start += 1;
        }
    }

    if mode == 0 || mode == 2 {
        while end > start && input[end - 1].is_ascii_whitespace() {
            end -= 1;
        }
    } else if mode == 3 {
        while end > start && input[end - 1] == b'\n' {
            end -= 1;
        }
    }

    into_ffi(input[start..end].to_vec())
}

fn byte_pair<'a>(
    left: *const u8,
    left_len: usize,
    right: *const u8,
    right_len: usize,
) -> Option<(&'a [u8], &'a [u8])> {
    if (left.is_null() && left_len != 0) || (right.is_null() && right_len != 0) {
        return None;
    }
    let left = if left_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(left, left_len) }
    };
    let right = if right_len == 0 {
        &[]
    } else {
        unsafe { slice::from_raw_parts(right, right_len) }
    };
    Some((left, right))
}

fn byte_slice<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    if data.is_null() && len != 0 {
        return None;
    }
    if len == 0 {
        Some(&[])
    } else {
        Some(unsafe { slice::from_raw_parts(data, len) })
    }
}

fn read_etag(path: &[u8]) -> UnicodeString {
    let Ok(path) = std::str::from_utf8(path) else {
        return into_ffi(Vec::new());
    };
    let etag_path = format!("{path}.etag");
    let Ok(contents) = std::fs::read_to_string(etag_path) else {
        return into_ffi(Vec::new());
    };
    let line = contents.lines().next().unwrap_or("");
    into_ffi(line.as_bytes().to_vec())
}

fn get_cached_ref(path: &[u8]) -> UnicodeString {
    let Ok(path) = std::str::from_utf8(path) else {
        return into_ffi(Vec::new());
    };
    let refs_path = std::path::Path::new(path).join("refs");
    let Ok(entries) = std::fs::read_dir(refs_path) else {
        return into_ffi(Vec::new());
    };

    let mut fallback: Option<String> = None;
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }
        let Ok(contents) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let commit = contents.lines().next().unwrap_or("");
        if commit.is_empty() || !is_hex_string(commit.as_bytes(), 40) {
            continue;
        }
        if entry.file_name() == "main" {
            return into_ffi(commit.as_bytes().to_vec());
        }
        if fallback.is_none() {
            fallback = Some(commit.to_string());
        }
    }

    into_ffi(fallback.unwrap_or_default().into_bytes())
}

fn safe_write_file(path: &[u8], data: &[u8]) -> bool {
    let Ok(path) = std::str::from_utf8(path) else {
        return false;
    };
    let path = std::path::Path::new(path);
    let tmp_path = std::path::PathBuf::from(format!("{}.tmp", path.display()));

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && std::fs::create_dir_all(parent).is_err() {
            let _ = std::fs::remove_file(&tmp_path);
            return false;
        }
    }

    if std::fs::write(&tmp_path, data).is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return false;
    }

    if std::fs::rename(&tmp_path, path).is_err() {
        let _ = std::fs::remove_file(&tmp_path);
        return false;
    }

    true
}

fn fs_is_directory(path: &[u8]) -> bool {
    let Ok(path) = std::str::from_utf8(path) else {
        return false;
    };
    std::path::Path::new(path).is_dir()
}

fn model_endpoint_string(
    model_endpoint: Option<&[u8]>,
    hf_endpoint: Option<&[u8]>,
) -> UnicodeString {
    let endpoint = model_endpoint.or(hf_endpoint);
    let Some(endpoint) = endpoint else {
        return into_ffi(b"https://huggingface.co/".to_vec());
    };

    if endpoint.is_empty() {
        return into_ffi(b"/".to_vec());
    }

    let mut out = endpoint.to_vec();
    if !out.ends_with(b"/") {
        out.push(b'/');
    }
    into_ffi(out)
}

fn parse_bool_arg(neg_args: &[StringView], key: &[u8], value: &[u8]) -> UnicodeString {
    for neg_arg in neg_args {
        let Some(neg_arg) = byte_slice(neg_arg.data, neg_arg.len) else {
            continue;
        };
        let stripped = strip_leading_dashes_bytes(neg_arg);
        if stripped == key {
            return if is_truthy_bytes(value) {
                into_ffi(b"false".to_vec())
            } else {
                into_ffi(b"true".to_vec())
            };
        }
    }
    into_ffi(value.to_vec())
}

fn read_file(path: &[u8]) -> UnicodeString {
    let Ok(path) = std::str::from_utf8(path) else {
        return null_string();
    };
    match std::fs::read(path) {
        Ok(data) => into_ffi(data),
        Err(_) => null_string(),
    }
}

fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right.iter())
        .take_while(|(left, right)| left == right)
        .count()
}

fn common_suffix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .rev()
        .zip(right.iter().rev())
        .take_while(|(left, right)| left == right)
        .count()
}

fn string_replace_all(input: &[u8], search: &[u8], replace: &[u8]) -> UnicodeString {
    if search.is_empty() {
        return into_ffi(input.to_vec());
    }

    let mut result = Vec::with_capacity(input.len());
    let mut pos = 0;
    while let Some(found) = find_subslice(&input[pos..], search) {
        let absolute = pos + found;
        result.extend_from_slice(&input[pos..absolute]);
        result.extend_from_slice(replace);
        pos = absolute + search.len();
    }
    result.extend_from_slice(&input[pos..]);
    into_ffi(result)
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    find_subslice(haystack, needle).is_some()
}

fn is_lfm2_template(src: &[u8]) -> bool {
    contains_subslice(src, b"<|tool_list_start|>") && contains_subslice(src, b"<|tool_list_end|>")
}

fn regex_escape(input: &[u8]) -> UnicodeString {
    let mut result = Vec::with_capacity(input.len());
    for &byte in input {
        if matches!(
            byte,
            b'.' | b'^'
                | b'$'
                | b'|'
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b'?'
                | b'['
                | b']'
                | b'{'
                | b'}'
                | b'\\'
        ) {
            result.push(b'\\');
        }
        result.push(byte);
    }
    into_ffi(result)
}

fn string_join(values: &[StringView], separator: &[u8]) -> UnicodeString {
    let mut result = Vec::new();
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            result.extend_from_slice(separator);
        }
        if value.data.is_null() && value.len != 0 {
            return UnicodeString {
                data: std::ptr::null_mut(),
                len: 0,
            };
        }
        if value.len != 0 {
            let bytes = unsafe { slice::from_raw_parts(value.data, value.len) };
            result.extend_from_slice(bytes);
        }
    }
    into_ffi(result)
}

fn string_split(input: &[u8], delimiter: &[u8]) -> StringList {
    if delimiter.is_empty() {
        return into_string_list(vec![input.to_vec()]);
    }

    let mut parts = Vec::new();
    let mut start = 0;
    while let Some(found) = find_subslice(&input[start..], delimiter) {
        let end = start + found;
        parts.push(input[start..end].to_vec());
        start = end + delimiter.len();
    }
    parts.push(input[start..].to_vec());
    into_string_list(parts)
}

fn parse_csv_row(input: &[u8]) -> StringList {
    let mut fields = Vec::new();
    let mut field = Vec::new();
    let mut in_quotes = false;
    let mut idx = 0;

    while idx < input.len() {
        let ch = input[idx];
        if ch == b'"' {
            if !in_quotes {
                if field.is_empty() {
                    in_quotes = true;
                } else {
                    field.push(b'"');
                }
            } else if idx + 1 < input.len() && input[idx + 1] == b'"' {
                field.push(b'"');
                idx += 1;
            } else {
                in_quotes = false;
            }
        } else if ch == b',' {
            if in_quotes {
                field.push(b',');
            } else {
                fields.push(std::mem::take(&mut field));
            }
        } else {
            field.push(ch);
        }
        idx += 1;
    }

    fields.push(field);
    into_string_list(fields)
}

fn trim_trailing_space_view(input: &[u8], max: c_int) -> StringView {
    let mut end = input.len();
    let mut count = 0_i32;
    while end > 0 && input[end - 1].is_ascii_whitespace() {
        if max != -1 && count >= max {
            break;
        }
        end -= 1;
        count += 1;
    }
    StringView {
        data: input.as_ptr(),
        len: end,
    }
}

fn trim_leading_space_view(input: &[u8], max: c_int) -> StringView {
    let mut start = 0_usize;
    let mut count = 0_i32;
    while start < input.len() && input[start].is_ascii_whitespace() {
        if max != -1 && count >= max {
            break;
        }
        start += 1;
        count += 1;
    }
    StringView {
        data: unsafe { input.as_ptr().add(start) },
        len: input.len() - start,
    }
}

fn string_process_escapes(input: &[u8]) -> UnicodeString {
    let mut output = Vec::with_capacity(input.len());
    let mut idx = 0;
    while idx < input.len() {
        if input[idx] == b'\\' && idx + 1 < input.len() {
            idx += 1;
            match input[idx] {
                b'n' => output.push(b'\n'),
                b'r' => output.push(b'\r'),
                b't' => output.push(b'\t'),
                b'\'' => output.push(b'\''),
                b'"' => output.push(b'"'),
                b'\\' => output.push(b'\\'),
                b'x' => {
                    if idx + 2 < input.len() {
                        if let Some(value) = parse_hex_byte(input[idx + 1], input[idx + 2]) {
                            idx += 2;
                            output.push(value);
                        } else {
                            output.push(b'\\');
                            output.push(input[idx]);
                        }
                    } else {
                        output.push(b'\\');
                        output.push(input[idx]);
                    }
                }
                other => {
                    output.push(b'\\');
                    output.push(other);
                }
            }
        } else {
            output.push(input[idx]);
        }
        idx += 1;
    }
    into_ffi(output)
}

fn clean_file_name(input: &[u8]) -> UnicodeString {
    let mut output = input.to_vec();
    for byte in &mut output {
        if *byte == b'/' || *byte == b'\\' {
            *byte = b'_';
        }
    }
    into_ffi(output)
}

fn string_diff(last: &[u8], current: &[u8]) -> (UnicodeString, c_int) {
    if last.is_empty() {
        return (into_ffi(current.to_vec()), 0);
    }
    if !current.starts_with(last) {
        if last.starts_with(current) {
            return (into_ffi(Vec::new()), 0);
        }
        return (
            UnicodeString {
                data: std::ptr::null_mut(),
                len: 0,
            },
            -1,
        );
    }
    (into_ffi(current[last.len()..].to_vec()), 0)
}

fn c_string_bytes(input: &[u8]) -> &[u8] {
    match input.iter().position(|&byte| byte == 0) {
        Some(end) => &input[..end],
        None => input,
    }
}

fn glob_match(pattern: &[u8], str: &[u8]) -> bool {
    if pattern.is_empty() {
        return str.is_empty();
    }

    if pattern.len() >= 2 && pattern[0] == b'*' && pattern[1] == b'*' {
        let rest = &pattern[2..];
        if glob_match(rest, str) {
            return true;
        }
        if !str.is_empty() {
            return glob_match(pattern, &str[1..]);
        }
        return false;
    }

    if pattern[0] == b'*' {
        let rest = &pattern[1..];
        let mut idx = 0;
        while idx < str.len() && str[idx] != b'/' {
            if glob_match(rest, &str[idx..]) {
                return true;
            }
            idx += 1;
        }
        return glob_match(rest, &str[idx..]);
    }

    if pattern[0] == b'?' && !str.is_empty() && str[0] != b'/' {
        return glob_match(&pattern[1..], &str[1..]);
    }

    if pattern[0] == b'[' {
        let mut class_end = 1;
        if class_end < pattern.len() && (pattern[class_end] == b']' || pattern[class_end] == b'-') {
            class_end += 1;
        }
        while class_end < pattern.len() && pattern[class_end] != b']' {
            class_end += 1;
        }
        if class_end < pattern.len() && pattern[class_end] == b']' {
            if str.is_empty() {
                return false;
            }
            return glob_class_match(str[0], &pattern[1..class_end])
                && glob_match(&pattern[class_end + 1..], &str[1..]);
        }
        if !str.is_empty() && str[0] == b'[' {
            return glob_match(&pattern[1..], &str[1..]);
        }
        return false;
    }

    if !str.is_empty() && pattern[0] == str[0] {
        return glob_match(&pattern[1..], &str[1..]);
    }

    false
}

fn glob_class_match(c: u8, class: &[u8]) -> bool {
    let mut idx = 0;
    let mut negated = false;

    if idx < class.len() && class[idx] == b'!' {
        negated = true;
        idx += 1;
    }

    if idx < class.len() && (class[idx] == b']' || class[idx] == b'-') {
        if class[idx] == c {
            return !negated;
        }
        idx += 1;
    }

    let mut matched = false;
    while idx < class.len() {
        if idx + 2 < class.len() && class[idx + 1] == b'-' && class[idx + 2] != b']' {
            if c >= class[idx] && c <= class[idx + 2] {
                matched = true;
                break;
            }
            idx += 3;
        } else {
            if class[idx] == c {
                matched = true;
                break;
            }
            idx += 1;
        }
    }

    if negated {
        !matched
    } else {
        matched
    }
}

fn fs_validate_filename(filename: &[u8], allow_subdirs: bool) -> bool {
    if filename.is_empty() || filename.len() > 255 {
        return false;
    }

    let mut offset = 0;
    while offset < filename.len() {
        let parsed = parse_utf8_codepoint(filename, offset);
        if parsed.status != SUCCESS {
            return false;
        }

        let c = parsed.codepoint;
        if (parsed.bytes_consumed == 2 && c < 0x80)
            || (parsed.bytes_consumed == 3 && c < 0x800)
            || (parsed.bytes_consumed == 4 && c < 0x10000)
        {
            return false;
        }

        if c <= 0x1F
            || c == 0x7F
            || (0x80..=0x9F).contains(&c)
            || c == 0xFF0E
            || c == 0x2215
            || c == 0x2216
            || (0xD800..=0xDFFF).contains(&c)
            || c > 0x10FFFF
            || c == 0xFFFD
            || c == 0xFEFF
            || [b':', b'*', b'?', b'"', b'<', b'>', b'|']
                .iter()
                .any(|&invalid| c == invalid as u32)
        {
            return false;
        }
        if !allow_subdirs && (c == b'/' as u32 || c == b'\\' as u32) {
            return false;
        }

        offset += parsed.bytes_consumed;
    }

    if filename.first() == Some(&b' ')
        || filename.last() == Some(&b' ')
        || filename.last() == Some(&b'.')
    {
        return false;
    }
    if find_subslice(filename, b"..").is_some() {
        return false;
    }
    if filename == b"." {
        return false;
    }

    true
}

fn string_lstrip_chars(input: &[u8], chars: &[u8]) -> UnicodeString {
    let start = input
        .iter()
        .position(|byte| !chars.contains(byte))
        .unwrap_or(input.len());
    into_ffi(input[start..].to_vec())
}

fn string_rstrip_chars(input: &[u8], chars: &[u8]) -> UnicodeString {
    let end = input
        .iter()
        .rposition(|byte| !chars.contains(byte))
        .map(|idx| idx + 1)
        .unwrap_or(0);
    into_ffi(input[..end].to_vec())
}

fn break_str_into_lines(input: &[u8], max_char_per_line: usize) -> StringList {
    if input.is_empty() {
        return into_string_list(Vec::new());
    }

    let mut result = Vec::new();
    for line in input.split(|&byte| byte == b'\n') {
        add_wrapped_line(line, max_char_per_line, &mut result);
    }

    if input.last() == Some(&b'\n') {
        result.pop();
    }

    into_string_list(result)
}

fn add_wrapped_line(line: &[u8], max_char_per_line: usize, result: &mut Vec<Vec<u8>>) {
    if line.len() <= max_char_per_line {
        result.push(line.to_vec());
        return;
    }

    let mut current_line = Vec::new();
    for word in line
        .split(u8::is_ascii_whitespace)
        .filter(|word| !word.is_empty())
    {
        let separator_len = usize::from(!current_line.is_empty());
        if current_line.len() + separator_len + word.len() > max_char_per_line {
            if !current_line.is_empty() {
                result.push(std::mem::take(&mut current_line));
            }
            current_line.extend_from_slice(word);
        } else {
            if !current_line.is_empty() {
                current_line.push(b' ');
            }
            current_line.extend_from_slice(word);
        }
    }
    if !current_line.is_empty() {
        result.push(current_line);
    }
}

fn rm_leading_dashes(input: &[u8]) -> UnicodeString {
    into_ffi(strip_leading_dashes_bytes(input).to_vec())
}

fn strip_leading_dashes_bytes(input: &[u8]) -> &[u8] {
    let start = input.iter().take_while(|&&byte| byte == b'-').count();
    &input[start..]
}

fn until_common_prefix(full: &[u8], left: &[u8], right: &[u8]) -> UnicodeString {
    let prefix_len = common_prefix_len(left, right);
    if prefix_len == 0 {
        return into_ffi(Vec::new());
    }

    match find_subslice(full, &left[..prefix_len]) {
        Some(pos) => into_ffi(full[..pos].to_vec()),
        None => into_ffi(Vec::new()),
    }
}

fn after_common_suffix(full: &[u8], left: &[u8], right: &[u8]) -> UnicodeString {
    let suffix_len = common_suffix_len(left, right);
    if suffix_len == 0 {
        return into_ffi(Vec::new());
    }

    let suffix = &left[left.len() - suffix_len..];
    match rfind_subslice(full, suffix) {
        Some(pos) => into_ffi(full[pos + suffix_len..].to_vec()),
        None => into_ffi(Vec::new()),
    }
}

fn rfind_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(haystack.len());
    }
    haystack
        .windows(needle.len())
        .rposition(|window| window == needle)
}

fn json_ensure_ascii_preserving_format(input: &[u8]) -> UnicodeString {
    let mut output = Vec::with_capacity(input.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut pos = 0;

    while pos < input.len() {
        let ch = input[pos];
        if !in_string {
            output.push(ch);
            if ch == b'"' {
                in_string = true;
            }
            pos += 1;
            continue;
        }

        if escaped {
            output.push(ch);
            escaped = false;
            pos += 1;
            continue;
        }

        if ch == b'\\' {
            output.push(ch);
            escaped = true;
            pos += 1;
            continue;
        }

        if ch == b'"' {
            output.push(ch);
            in_string = false;
            pos += 1;
            continue;
        }

        if ch < 0x80 {
            output.push(ch);
            pos += 1;
            continue;
        }

        let parsed = parse_utf8_codepoint(input, pos);
        if parsed.status != SUCCESS {
            output.extend_from_slice(b"\\ufffd");
            pos += 1;
            continue;
        }

        append_codepoint_as_ascii_json_escape(&mut output, parsed.codepoint);
        pos += parsed.bytes_consumed;
    }

    into_ffi(output)
}

fn append_codepoint_as_ascii_json_escape(output: &mut Vec<u8>, mut codepoint: u32) {
    if codepoint <= 0xFFFF {
        output.extend_from_slice(format!("\\u{codepoint:04x}").as_bytes());
        return;
    }

    codepoint -= 0x10000;
    let high = 0xD800 + ((codepoint >> 10) & 0x3FF);
    let low = 0xDC00 + (codepoint & 0x3FF);
    output.extend_from_slice(format!("\\u{high:04x}\\u{low:04x}").as_bytes());
}

fn normalize_quotes_to_json(input: &[u8]) -> UnicodeString {
    let mut result = Vec::with_capacity(input.len() + 16);
    let mut in_single_quoted = false;
    let mut in_double_quoted = false;
    let mut idx = 0_usize;

    while idx < input.len() {
        let c = input[idx];

        if c == b'\\' && idx + 1 < input.len() {
            let next = input[idx + 1];

            if in_single_quoted {
                if next == b'\'' {
                    result.push(b'\'');
                    idx += 2;
                    continue;
                }
                if next == b'"' {
                    result.extend_from_slice(b"\\\"");
                    idx += 2;
                    continue;
                }
                result.push(c);
                result.push(next);
                idx += 2;
                continue;
            }

            if in_double_quoted {
                result.push(c);
                result.push(next);
                idx += 2;
                continue;
            }

            result.push(c);
            idx += 1;
            continue;
        }

        if c == b'"' {
            if in_single_quoted {
                result.extend_from_slice(b"\\\"");
            } else {
                in_double_quoted = !in_double_quoted;
                result.push(c);
            }
        } else if c == b'\'' {
            if in_double_quoted {
                result.push(c);
            } else if in_single_quoted {
                in_single_quoted = false;
                result.push(b'"');
            } else {
                in_single_quoted = true;
                result.push(b'"');
            }
        } else {
            result.push(c);
        }

        idx += 1;
    }

    into_ffi(result)
}

fn parse_cpu_mask(mask: &[u8], boolmask: &mut [bool]) -> c_int {
    let start = if mask.len() >= 2 && &mask[..2] == b"0x" {
        2
    } else {
        0
    };
    let num_digits = (mask.len() - start).min(128);
    let end = start + num_digits;

    for (offset, idx) in (start..end).enumerate() {
        let Some(id) = hex_value(mask[idx]) else {
            return idx as c_int;
        };
        let n = num_digits * 4 - 1 - offset * 4;
        if n < boolmask.len() {
            boolmask[n] |= (id & 8) != 0;
        }
        if n >= 1 && n - 1 < boolmask.len() {
            boolmask[n - 1] |= (id & 4) != 0;
        }
        if n >= 2 && n - 2 < boolmask.len() {
            boolmask[n - 2] |= (id & 2) != 0;
        }
        if n >= 3 && n - 3 < boolmask.len() {
            boolmask[n - 3] |= (id & 1) != 0;
        }
    }

    -1
}

fn get_line_col(source: &[u8], pos: usize) -> UnicodeString {
    let mut line = 1_usize;
    let mut col = 1_usize;
    for &c in source.iter().take(pos) {
        if c == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    into_ffi(format!("line {line}, column {col}").into_bytes())
}

fn lr_opt_init(lr0: f32, lr_min: f32, mut decay_epochs: f32, epochs: u32) -> (f32, f32) {
    let mut scale_epoch = 0.0;
    if lr_min > 0.0 && lr_min < lr0 {
        let nhalf = (lr0 / lr_min).ln() / 2.0_f32.ln();
        let mut e = epochs as f32;
        if decay_epochs > 0.0 && decay_epochs < e {
            e = decay_epochs;
        } else {
            decay_epochs = e;
        }
        scale_epoch = nhalf / e;
    }
    (decay_epochs, scale_epoch)
}

fn lr_opt_get_lr(lr0: f32, lr_min: f32, decay_epochs: f32, scale_epoch: f32, epoch: f32) -> f32 {
    if lr_min <= 0.0 {
        lr0
    } else if epoch >= decay_epochs {
        lr_min
    } else {
        lr0 * 0.5_f32.powf(epoch * scale_epoch)
    }
}

fn default_thread_count_windows(num_physical_cores: c_int, default_threads: c_int) -> c_int {
    if num_physical_cores > 0 {
        num_physical_cores
    } else {
        default_threads
    }
}

fn default_thread_count(n_threads: u32) -> c_int {
    if n_threads == 0 {
        4
    } else if n_threads <= 4 {
        n_threads as c_int
    } else {
        (n_threads / 2) as c_int
    }
}

fn time_us() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_micros().min(i64::MAX as u128) as i64,
        Err(err) => -(err.duration().as_micros().min(i64::MAX as u128) as i64),
    }
}

fn sampler_prob_desc(left: f32, right: f32) -> bool {
    left > right
}

fn marker_is_opener(c: u8) -> bool {
    c == b'<' || c == b'['
}

fn marker_is_closer(opener: u8, c: u8) -> bool {
    (opener == b'<' && c == b'>') || (opener == b'[' && c == b']')
}

fn content_is_always_wrapped(mode: c_int, start_len: usize, end_len: usize) -> bool {
    mode == 1 && start_len != 0 && end_len != 0
}

fn cpu_has_hybrid_bit(edx: u32) -> bool {
    edx & (1u32 << 15) != 0
}

fn cpu_is_intel_atom_core_type(eax: u32) -> bool {
    const INTEL_ATOM: u32 = 0x20;
    ((eax & 0xff00_0000) >> 24) == INTEL_ATOM
}

fn arg_has_env_value(has_env: bool, has_neg_env_value: bool, has_env_value: bool) -> bool {
    has_neg_env_value || (has_env && has_env_value)
}

fn gguf_filename_is_model(filepath: &[u8]) -> bool {
    if !filepath.ends_with(b".gguf") {
        return false;
    }

    let filename = match filepath.iter().rposition(|&byte| byte == b'/') {
        Some(pos) => &filepath[pos + 1..],
        None => filepath,
    };

    find_subslice(filename, b"mmproj").is_none() && find_subslice(filename, b"imatrix").is_none()
}

fn peg_rule_name(name: &[u8]) -> UnicodeString {
    let mut result = Vec::with_capacity(name.len());
    let mut in_invalid_run = false;

    for &c in name {
        if c.is_ascii_alphanumeric() || c == b'-' {
            result.push(c);
            in_invalid_run = false;
        } else if !in_invalid_run {
            result.push(b'-');
            in_invalid_run = true;
        }
    }

    into_ffi(result)
}

fn gbnf_format_literal(literal: &[u8]) -> UnicodeString {
    let mut result = Vec::with_capacity(literal.len() + 2);
    result.push(b'"');
    for &c in literal {
        match c {
            b'\r' => result.extend_from_slice(b"\\r"),
            b'\n' => result.extend_from_slice(b"\\n"),
            b'"' => result.extend_from_slice(b"\\\""),
            b'\\' => result.extend_from_slice(b"\\\\"),
            _ => result.push(c),
        }
    }
    result.push(b'"');
    into_ffi(result)
}

fn gbnf_build_repetition(
    item_rule: &[u8],
    min_items: c_int,
    max_items: c_int,
    separator_rule: &[u8],
) -> UnicodeString {
    const INT_MAX: c_int = c_int::MAX;
    let has_max = max_items != INT_MAX;

    if max_items == 0 {
        return into_ffi(Vec::new());
    }
    if min_items == 0 && max_items == 1 {
        let mut result = Vec::with_capacity(item_rule.len() + 1);
        result.extend_from_slice(item_rule);
        result.push(b'?');
        return into_ffi(result);
    }

    if separator_rule.is_empty() {
        let mut result = Vec::new();
        result.extend_from_slice(item_rule);
        if min_items == 1 && !has_max {
            result.push(b'+');
        } else if min_items == 0 && !has_max {
            result.push(b'*');
        } else {
            result.push(b'{');
            result.extend_from_slice(min_items.to_string().as_bytes());
            result.push(b',');
            if has_max {
                result.extend_from_slice(max_items.to_string().as_bytes());
            }
            result.push(b'}');
        }
        return into_ffi(result);
    }

    let mut inner = Vec::with_capacity(separator_rule.len() + item_rule.len() + 3);
    inner.push(b'(');
    inner.extend_from_slice(separator_rule);
    inner.push(b' ');
    inner.extend_from_slice(item_rule);
    inner.push(b')');
    let inner_min = if min_items == 0 { 0 } else { min_items - 1 };
    let inner_max = if has_max { max_items - 1 } else { max_items };
    let repeated = gbnf_build_repetition(&inner, inner_min, inner_max, &[]);
    let repeated = unsafe { Vec::from_raw_parts(repeated.data, repeated.len, repeated.len) };

    let mut result = Vec::with_capacity(item_rule.len() + repeated.len() + 6);
    result.extend_from_slice(item_rule);
    result.push(b' ');
    result.extend_from_slice(&repeated);
    if min_items == 0 {
        result.insert(0, b'(');
        result.push(b')');
        result.push(b'?');
    }
    into_ffi(result)
}

fn gbnf_escape_char_class(c: u32) -> UnicodeString {
    match c {
        45 | 93 | 91 | 92 => into_ffi(vec![b'\\', c as u8]),
        10 => into_ffi(b"\\n".to_vec()),
        9 => into_ffi(b"\\t".to_vec()),
        13 => into_ffi(b"\\r".to_vec()),
        0x20..=0x7E => into_ffi(vec![c as u8]),
        0x00..=0xFF => into_ffi(format!("\\x{c:02X}").into_bytes()),
        0x0100..=0xFFFF => into_ffi(format!("\\u{c:04X}").into_bytes()),
        _ => into_ffi(format!("\\U{c:08X}").into_bytes()),
    }
}

fn gbnf_is_reserved_name(name: &[u8]) -> bool {
    matches!(
        name,
        b"root"
            | b"boolean"
            | b"decimal-part"
            | b"integral-part"
            | b"number"
            | b"integer"
            | b"value"
            | b"object"
            | b"array"
            | b"uuid"
            | b"char"
            | b"string"
            | b"null"
            | b"date"
            | b"time"
            | b"date-time"
            | b"date-string"
            | b"time-string"
            | b"date-time-string"
    )
}

fn gbnf_ref_name(reference: &[u8]) -> UnicodeString {
    let fragment = match reference.iter().position(|&c| c == b'#') {
        Some(pos) => &reference[pos + 1..],
        None => reference,
    };

    let mut result = b"ref".to_vec();
    let mut in_invalid_run = false;
    for &c in fragment {
        if c.is_ascii_alphanumeric() || c == b'-' {
            result.push(c);
            in_invalid_run = false;
        } else if !in_invalid_run {
            result.push(b'-');
            in_invalid_run = true;
        }
    }
    into_ffi(result)
}

fn hf_folder_name_to_repo(folder: &[u8]) -> UnicodeString {
    const PREFIX: &[u8] = b"models--";
    if !folder.starts_with(PREFIX) {
        return into_ffi(Vec::new());
    }
    string_replace_all(&folder[PREFIX.len()..], b"--", b"/")
}

fn hf_repo_to_folder_name(repo_id: &[u8]) -> UnicodeString {
    let mut result = b"models--".to_vec();
    result.extend_from_slice(repo_id);
    let replaced = string_replace_all(&result, b"/", b"--");
    replaced
}

fn hf_make_old_cache_filename(owner: &[u8], repo: &[u8], filename: &[u8]) -> UnicodeString {
    let mut result = Vec::with_capacity(owner.len() + repo.len() + filename.len() + 2);
    result.extend_from_slice(owner);
    result.push(b'_');
    result.extend_from_slice(repo);
    result.push(b'_');
    result.extend_from_slice(filename);
    string_replace_all(&result, b"/", b"_")
}

fn json_brace_depth(input: &[u8]) -> c_int {
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escaped = false;

    for &c in input {
        if escaped {
            escaped = false;
            continue;
        }
        if c == b'\\' && in_string {
            escaped = true;
            continue;
        }
        if c == b'"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if c == b'{' {
                depth += 1;
            } else if c == b'}' {
                depth -= 1;
            }
        }
    }
    depth
}

fn is_hex_string(input: &[u8], expected_len: usize) -> bool {
    input.len() == expected_len && input.iter().all(|&c| is_hex_digit(c))
}

fn is_alphanum(c: u8) -> bool {
    c.is_ascii_alphanumeric()
}

fn hf_is_special_char(c: u8) -> bool {
    matches!(c, b'/' | b'.' | b'-')
}

fn hf_is_valid_repo_id(repo_id: &[u8]) -> bool {
    if repo_id.is_empty() || repo_id.len() > 256 {
        return false;
    }
    let mut slash = 0;
    let mut special = true;

    for &c in repo_id {
        if is_alphanum(c) || c == b'_' {
            special = false;
        } else if hf_is_special_char(c) {
            if special {
                return false;
            }
            slash += usize::from(c == b'/');
            special = true;
        } else {
            return false;
        }
    }
    !special && slash == 1
}

fn hf_is_valid_token(token: &[u8]) -> bool {
    token.len() >= 37
        && token.len() <= 256
        && token.starts_with(b"hf_")
        && token[3..].iter().all(|&c| is_alphanum(c))
}

fn hf_parse_manifest_name(filename: &[u8]) -> StringList {
    const PREFIX: &[u8] = b"manifest=";
    const SUFFIX: &[u8] = b".json";
    if !filename.starts_with(PREFIX) || !filename.ends_with(SUFFIX) {
        return into_string_list(Vec::new());
    }
    let rest = &filename[PREFIX.len()..];
    let Some(owner_end) = rest.iter().position(|&c| c == b'=') else {
        return into_string_list(Vec::new());
    };
    if owner_end == 0 {
        return into_string_list(Vec::new());
    }
    let owner = &rest[..owner_end];
    let rest = &rest[owner_end + 1..];
    let Some(repo_end) = rest.iter().position(|&c| c == b'=') else {
        return into_string_list(Vec::new());
    };
    if repo_end == 0 {
        return into_string_list(Vec::new());
    }
    if repo_end + 1 >= rest.len() {
        return into_string_list(Vec::new());
    }
    into_string_list(vec![owner.to_vec(), rest[..repo_end].to_vec()])
}

fn empty_gguf_split_info() -> GgufSplitInfo {
    GgufSplitInfo {
        prefix: into_ffi(Vec::new()),
        tag: into_ffi(Vec::new()),
        index: 0,
        count: 0,
    }
}

fn gguf_split_info(path: &[u8], extract_tag: bool) -> GgufSplitInfo {
    let Some(mut prefix) = path.strip_suffix(b".gguf").map(Vec::from) else {
        return empty_gguf_split_info();
    };

    let mut index = 1;
    let mut count = 1;
    if let Some((split_prefix, split_index, split_count)) = gguf_split_suffix(&prefix) {
        prefix = split_prefix;
        index = split_index;
        count = split_count;
    }

    let tag = if extract_tag {
        gguf_extract_tag(&prefix)
    } else {
        Vec::new()
    };

    GgufSplitInfo {
        prefix: into_ffi(prefix),
        tag: into_ffi(tag),
        index,
        count,
    }
}

fn gguf_split_suffix(prefix: &[u8]) -> Option<(Vec<u8>, c_int, c_int)> {
    const MID: &[u8] = b"-of-";
    if prefix.len() < 16 {
        return None;
    }
    let count_start = prefix.len() - 5;
    let mid_start = count_start.checked_sub(MID.len())?;
    let index_start = mid_start.checked_sub(5)?;
    if index_start == 0 || prefix.get(index_start.wrapping_sub(1)) != Some(&b'-') {
        return None;
    }
    if &prefix[mid_start..count_start] != MID {
        return None;
    }
    let index = parse_fixed_decimal(&prefix[index_start..mid_start])?;
    let count = parse_fixed_decimal(&prefix[count_start..])?;
    Some((prefix[..index_start - 1].to_vec(), index, count))
}

fn parse_fixed_decimal(input: &[u8]) -> Option<c_int> {
    if input.is_empty() || !input.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut value = 0_i32;
    for &c in input {
        value = value.checked_mul(10)?.checked_add((c - b'0') as i32)?;
    }
    Some(value)
}

fn gguf_extract_tag(prefix: &[u8]) -> Vec<u8> {
    let Some(pos) = prefix.iter().rposition(|&c| c == b'-' || c == b'.') else {
        return Vec::new();
    };
    let tag = &prefix[pos + 1..];
    if tag.is_empty() || !tag.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'_') {
        return Vec::new();
    }
    tag.iter().map(u8::to_ascii_uppercase).collect()
}

fn gguf_extract_quant_bits(filename: &[u8]) -> c_int {
    let split = gguf_split_info(filename, true);
    if split.tag.data.is_null() {
        unsafe { llama_common_unicode_string_free(split.prefix) };
        return 0;
    }

    let tag = unsafe { Vec::from_raw_parts(split.tag.data, split.tag.len, split.tag.len) };
    unsafe { llama_common_unicode_string_free(split.prefix) };

    let Some(pos) = tag.iter().position(u8::is_ascii_digit) else {
        return 0;
    };
    let mut value = 0_i32;
    for &c in &tag[pos..] {
        if !c.is_ascii_digit() {
            break;
        }
        let Some(next) = value
            .checked_mul(10)
            .and_then(|v| v.checked_add((c - b'0') as i32))
        else {
            return 0;
        };
        value = next;
    }
    value
}

fn embd_normalize(inp: &[f32], out: &mut [f32], embd_norm: c_int) {
    let mut sum = 0.0_f64;

    match embd_norm {
        -1 => sum = 1.0,
        0 => {
            for &value in inp {
                sum = sum.max((value as f64).abs());
            }
            sum /= 32760.0;
        }
        2 => {
            for &value in inp {
                let value = value as f64;
                sum += value * value;
            }
            sum = sum.sqrt();
        }
        norm => {
            for &value in inp {
                sum += (value as f64).abs().powf(norm as f64);
            }
            sum = sum.powf(1.0 / norm as f64);
        }
    }

    let norm = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    for (dst, src) in out.iter_mut().zip(inp.iter()) {
        *dst = *src * norm;
    }
}

fn embd_similarity_cos(embd1: &[f32], embd2: &[f32]) -> f32 {
    let mut sum = 0.0_f64;
    let mut sum1 = 0.0_f64;
    let mut sum2 = 0.0_f64;

    for (&left, &right) in embd1.iter().zip(embd2.iter()) {
        let left = left as f64;
        let right = right as f64;
        sum += left * right;
        sum1 += left * left;
        sum2 += right * right;
    }

    if sum1 == 0.0 || sum2 == 0.0 {
        if sum1 == 0.0 && sum2 == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (sum / (sum1.sqrt() * sum2.sqrt())) as f32
    }
}

fn parse_hex_byte(high: u8, low: u8) -> Option<u8> {
    Some(hex_value(high)? << 4 | hex_value(low)?)
}

fn hex_value(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn is_hex_digit(c: u8) -> bool {
    c.is_ascii_hexdigit()
}

fn parse_hex_escape(input: &[u8], pos: usize, hex_count: c_int) -> Option<(u32, usize)> {
    if hex_count < 0 {
        return None;
    }
    let hex_count = hex_count as usize;
    if pos.checked_add(hex_count)? > input.len() {
        return None;
    }

    let mut value = 0_u32;
    for &c in &input[pos..pos + hex_count] {
        let digit = hex_value(c)?;
        value = (value << 4) + digit as u32;
    }
    Some((value, hex_count))
}

fn parse_char_class_char(input: &[u8], pos: usize) -> Option<(u32, usize)> {
    let current = *input.get(pos)?;
    if current == b'\\' && pos + 1 < input.len() {
        return match input[pos + 1] {
            b'x' => Some(parse_hex_escape(input, pos + 2, 2).unwrap_or((b'x' as u32, 0))).map(
                |(value, consumed)| {
                    if consumed > 0 {
                        (value, 2 + consumed)
                    } else {
                        (value, 2)
                    }
                },
            ),
            b'u' => Some(parse_hex_escape(input, pos + 2, 4).unwrap_or((b'u' as u32, 0))).map(
                |(value, consumed)| {
                    if consumed > 0 {
                        (value, 2 + consumed)
                    } else {
                        (value, 2)
                    }
                },
            ),
            b'U' => Some(parse_hex_escape(input, pos + 2, 8).unwrap_or((b'U' as u32, 0))).map(
                |(value, consumed)| {
                    if consumed > 0 {
                        (value, 2 + consumed)
                    } else {
                        (value, 2)
                    }
                },
            ),
            b'n' => Some((b'\n' as u32, 2)),
            b't' => Some((b'\t' as u32, 2)),
            b'r' => Some((b'\r' as u32, 2)),
            b'\\' => Some((b'\\' as u32, 2)),
            b']' => Some((b']' as u32, 2)),
            b'[' => Some((b'[' as u32, 2)),
            other => Some((other as u32, 2)),
        };
    }

    Some((current as u32, 1))
}

fn sampler_type_from_name(name: &[u8], allow_alt_names: bool) -> c_int {
    match name {
        b"dry" => 1,
        b"top_k" => 2,
        b"top_p" => 3,
        b"min_p" => 4,
        b"typ_p" => 6,
        b"temperature" => 7,
        b"xtc" => 8,
        b"infill" => 9,
        b"penalties" => 10,
        b"top_n_sigma" => 11,
        b"adaptive_p" => 12,
        b"top-k" if allow_alt_names => 2,
        b"top-p" if allow_alt_names => 3,
        b"nucleus" if allow_alt_names => 3,
        b"min-p" if allow_alt_names => 4,
        b"typical-p" if allow_alt_names => 6,
        b"typical" if allow_alt_names => 6,
        b"typ-p" if allow_alt_names => 6,
        b"typ" if allow_alt_names => 6,
        b"temp" if allow_alt_names => 7,
        b"top-n-sigma" if allow_alt_names => 11,
        b"adaptive-p" if allow_alt_names => 12,
        _ => -1,
    }
}

fn sampler_type_from_chr(name: u8) -> c_int {
    match name {
        b'd' => 1,
        b'k' => 2,
        b'p' => 3,
        b'm' => 4,
        b'y' => 6,
        b't' => 7,
        b'x' => 8,
        b'i' => 9,
        b'e' => 10,
        b's' => 11,
        b'a' => 12,
        _ => -1,
    }
}

fn into_string_list(parts: Vec<Vec<u8>>) -> StringList {
    let mut strings = parts.into_iter().map(into_ffi).collect::<Vec<_>>();
    let len = strings.len();
    let data = strings.as_mut_ptr();
    std::mem::forget(strings);
    StringList { data, len }
}

fn result(status: i32, codepoint: u32, bytes_consumed: usize) -> Utf8ParseResult {
    Utf8ParseResult {
        codepoint,
        bytes_consumed,
        status,
    }
}

fn into_ffi(mut bytes: Vec<u8>) -> UnicodeString {
    let len = bytes.len();
    let data = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    UnicodeString { data, len }
}

fn null_string() -> UnicodeString {
    UnicodeString {
        data: std::ptr::null_mut(),
        len: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_codepoints_like_common_cpp() {
        let parsed = parse_utf8_codepoint("A".as_bytes(), 0);
        assert_eq!(parsed.status, SUCCESS);
        assert_eq!(parsed.codepoint, b'A' as u32);
        assert_eq!(parsed.bytes_consumed, 1);

        let parsed = parse_utf8_codepoint(b"\xE2\x80\x9C", 0);
        assert_eq!(parsed.status, SUCCESS);
        assert_eq!(parsed.codepoint, 0x201C);
        assert_eq!(parsed.bytes_consumed, 3);

        assert_eq!(parse_utf8_codepoint(&[0x80], 0).status, INVALID);
        assert_eq!(parse_utf8_codepoint(&[0xF0, 0x9F], 0).status, INCOMPLETE);
    }

    #[test]
    fn utf8_completion_matches_streaming_cases() {
        assert!(utf8_is_complete(b""));
        assert!(utf8_is_complete(b"hello"));
        assert!(utf8_is_complete(b"hello\xC3\xA9"));
        assert!(!utf8_is_complete(b"hello\xC3"));
        assert!(!utf8_is_complete(&[0x80]));
    }

    #[test]
    fn encodes_codepoints() {
        assert_eq!(unicode_cpt_to_utf8(0x41).unwrap(), b"A");
        assert_eq!(unicode_cpt_to_utf8(0x00A0).unwrap(), b"\xC2\xA0");
        assert_eq!(unicode_cpt_to_utf8(0x201C).unwrap(), b"\xE2\x80\x9C");
        assert_eq!(unicode_cpt_to_utf8(0x1F600).unwrap(), b"\xF0\x9F\x98\x80");
        assert!(unicode_cpt_to_utf8(0x11_0000).is_none());
    }

    #[test]
    fn encodes_codepoint_vectors() {
        let cpts = [0x41, 0x00A0, 0x201C, 0x1F600];
        let encoded = unsafe { llama_common_unicode_cpts_to_utf8_rust(cpts.as_ptr(), cpts.len()) };
        unsafe {
            assert_eq!(
                std::slice::from_raw_parts(encoded.data, encoded.len),
                b"A\xC2\xA0\xE2\x80\x9C\xF0\x9F\x98\x80"
            );
            llama_common_unicode_string_free(encoded);

            let invalid = [0x41, 0x11_0000];
            let encoded = llama_common_unicode_cpts_to_utf8_rust(invalid.as_ptr(), invalid.len());
            assert!(encoded.data.is_null());
            assert_eq!(encoded.len, 0);

            let null = llama_common_unicode_cpts_to_utf8_rust(std::ptr::null(), 1);
            assert!(null.data.is_null());
            assert_eq!(null.len, 0);
        }
    }

    #[test]
    fn maps_jinja_token_types() {
        unsafe {
            assert_eq!(
                CStr::from_ptr(llama_common_jinja_token_type_to_string_rust(0)).to_bytes(),
                b"eof"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_jinja_token_type_to_string_rust(4)).to_bytes(),
                b"identifier"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_jinja_token_type_to_string_rust(20)).to_bytes(),
                b"call_operator"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_jinja_token_type_to_string_rust(25)).to_bytes(),
                b"comment"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_jinja_token_type_to_string_rust(99)).to_bytes(),
                b"unknown"
            );
        }
    }

    #[test]
    fn classifies_jinja_word_and_integer_chars() {
        assert!(llama_common_jinja_is_word_rust(b'a'));
        assert!(llama_common_jinja_is_word_rust(b'Z'));
        assert!(llama_common_jinja_is_word_rust(b'5'));
        assert!(llama_common_jinja_is_word_rust(b'_'));
        assert!(!llama_common_jinja_is_word_rust(b'-'));
        assert!(!llama_common_jinja_is_word_rust(0xE9));

        assert!(llama_common_jinja_is_integer_rust(b'0'));
        assert!(llama_common_jinja_is_integer_rust(b'9'));
        assert!(!llama_common_jinja_is_integer_rust(b'a'));
        assert!(!llama_common_jinja_is_integer_rust(b'_'));
    }

    #[test]
    fn masks_http_urls() {
        unsafe {
            let with_user = llama_common_http_show_masked_url_rust(
                b"https".as_ptr(),
                5,
                true,
                b"example.com".as_ptr(),
                11,
                b"/models/a.gguf".as_ptr(),
                14,
            );
            assert_eq!(
                std::slice::from_raw_parts(with_user.data, with_user.len),
                b"https://****:****@example.com/models/a.gguf"
            );
            llama_common_unicode_string_free(with_user);

            let anonymous = llama_common_http_show_masked_url_rust(
                b"http".as_ptr(),
                4,
                false,
                b"localhost".as_ptr(),
                9,
                b"/".as_ptr(),
                1,
            );
            assert_eq!(
                std::slice::from_raw_parts(anonymous.data, anonymous.len),
                b"http://localhost/"
            );
            llama_common_unicode_string_free(anonymous);
        }
    }

    #[test]
    fn parses_hex_escapes() {
        unsafe {
            assert!(llama_common_is_hex_digit_rust(b'0'));
            assert!(llama_common_is_hex_digit_rust(b'f'));
            assert!(llama_common_is_hex_digit_rust(b'F'));
            assert!(!llama_common_is_hex_digit_rust(b'g'));

            let mut value = 0_u32;
            assert_eq!(
                llama_common_parse_hex_escape_rust(b"xx00afzz".as_ptr(), 8, 2, 4, &mut value),
                4
            );
            assert_eq!(value, 0x00af);

            value = 123;
            assert_eq!(
                llama_common_parse_hex_escape_rust(b"12xz".as_ptr(), 4, 0, 4, &mut value),
                0
            );
            assert_eq!(value, 123);
            assert_eq!(
                llama_common_parse_hex_escape_rust(b"12".as_ptr(), 2, 0, 4, &mut value),
                0
            );
            assert_eq!(
                llama_common_parse_hex_escape_rust(std::ptr::null(), 1, 0, 1, &mut value),
                0
            );
        }
    }

    #[test]
    fn parses_character_class_characters() {
        unsafe {
            let mut value = 0_u32;
            assert_eq!(
                llama_common_parse_char_class_char_rust(b"a".as_ptr(), 1, 0, &mut value),
                1
            );
            assert_eq!(value, b'a' as u32);

            assert_eq!(
                llama_common_parse_char_class_char_rust(br"\n".as_ptr(), 2, 0, &mut value),
                2
            );
            assert_eq!(value, b'\n' as u32);

            assert_eq!(
                llama_common_parse_char_class_char_rust(br"\x7f".as_ptr(), 4, 0, &mut value),
                4
            );
            assert_eq!(value, 0x7f);

            assert_eq!(
                llama_common_parse_char_class_char_rust(br"\u20ac".as_ptr(), 6, 0, &mut value),
                6
            );
            assert_eq!(value, 0x20ac);

            assert_eq!(
                llama_common_parse_char_class_char_rust(br"\xzz".as_ptr(), 4, 0, &mut value),
                2
            );
            assert_eq!(value, b'x' as u32);

            assert_eq!(
                llama_common_parse_char_class_char_rust(std::ptr::null(), 1, 0, &mut value),
                0
            );
        }
    }

    #[test]
    fn maps_chat_and_reasoning_formats() {
        unsafe {
            assert_eq!(
                CStr::from_ptr(llama_common_chat_format_name_rust(0)).to_bytes(),
                b"Content-only"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_chat_format_name_rust(1)).to_bytes(),
                b"peg-simple"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_chat_format_name_rust(2)).to_bytes(),
                b"peg-native"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_chat_format_name_rust(3)).to_bytes(),
                b"peg-gemma4"
            );
            assert!(llama_common_chat_format_name_rust(99).is_null());

            assert_eq!(
                CStr::from_ptr(llama_common_reasoning_format_name_rust(0)).to_bytes(),
                b"none"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_reasoning_format_name_rust(1)).to_bytes(),
                b"auto"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_reasoning_format_name_rust(2)).to_bytes(),
                b"deepseek-legacy"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_reasoning_format_name_rust(3)).to_bytes(),
                b"deepseek"
            );
            assert!(llama_common_reasoning_format_name_rust(-1).is_null());

            assert_eq!(
                llama_common_reasoning_format_from_name_rust(c"none".as_ptr()),
                0
            );
            assert_eq!(
                llama_common_reasoning_format_from_name_rust(c"auto".as_ptr()),
                1
            );
            assert_eq!(
                llama_common_reasoning_format_from_name_rust(c"deepseek-legacy".as_ptr()),
                2
            );
            assert_eq!(
                llama_common_reasoning_format_from_name_rust(c"deepseek".as_ptr()),
                3
            );
            assert_eq!(
                llama_common_reasoning_format_from_name_rust(c"missing".as_ptr()),
                -1
            );
            assert_eq!(
                llama_common_reasoning_format_from_name_rust(std::ptr::null()),
                -1
            );
        }
    }

    #[test]
    fn maps_sampler_types() {
        unsafe {
            assert_eq!(llama_common_sampler_type_to_chr_rust(1) as u8, b'd');
            assert_eq!(llama_common_sampler_type_to_chr_rust(2) as u8, b'k');
            assert_eq!(llama_common_sampler_type_to_chr_rust(3) as u8, b'p');
            assert_eq!(llama_common_sampler_type_to_chr_rust(4) as u8, b'm');
            assert_eq!(llama_common_sampler_type_to_chr_rust(6) as u8, b'y');
            assert_eq!(llama_common_sampler_type_to_chr_rust(7) as u8, b't');
            assert_eq!(llama_common_sampler_type_to_chr_rust(8) as u8, b'x');
            assert_eq!(llama_common_sampler_type_to_chr_rust(9) as u8, b'i');
            assert_eq!(llama_common_sampler_type_to_chr_rust(10) as u8, b'e');
            assert_eq!(llama_common_sampler_type_to_chr_rust(11) as u8, b's');
            assert_eq!(llama_common_sampler_type_to_chr_rust(12) as u8, b'a');
            assert_eq!(llama_common_sampler_type_to_chr_rust(99) as u8, b'?');

            assert_eq!(
                CStr::from_ptr(llama_common_sampler_type_to_str_rust(1)).to_bytes(),
                b"dry"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_sampler_type_to_str_rust(11)).to_bytes(),
                b"top_n_sigma"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_sampler_type_to_str_rust(12)).to_bytes(),
                b"adaptive_p"
            );
            assert!(llama_common_sampler_type_to_str_rust(0).is_null());
        }
    }

    #[test]
    fn parses_sampler_types_from_names_and_chars() {
        unsafe {
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"dry".as_ptr(), false),
                1
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"top_k".as_ptr(), false),
                2
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"top_p".as_ptr(), false),
                3
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"min_p".as_ptr(), false),
                4
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"typ_p".as_ptr(), false),
                6
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"temperature".as_ptr(), false),
                7
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"top-n-sigma".as_ptr(), false),
                -1
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"top-n-sigma".as_ptr(), true),
                11
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"nucleus".as_ptr(), true),
                3
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"typ".as_ptr(), true),
                6
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(c"missing".as_ptr(), true),
                -1
            );
            assert_eq!(
                llama_common_sampler_type_from_name_rust(std::ptr::null(), true),
                -1
            );

            assert_eq!(llama_common_sampler_type_from_chr_rust(b'd' as c_char), 1);
            assert_eq!(llama_common_sampler_type_from_chr_rust(b's' as c_char), 11);
            assert_eq!(llama_common_sampler_type_from_chr_rust(b'?' as c_char), -1);
        }
    }

    #[test]
    fn maps_speculative_types() {
        unsafe {
            let expected: &[(i32, &[u8])] = &[
                (0, b"none"),
                (1, b"draft"),
                (2, b"eagle3"),
                (3, b"ngram_simple"),
                (4, b"ngram_map_k"),
                (5, b"ngram_map_k4v"),
                (6, b"ngram_mod"),
                (7, b"ngram_cache"),
            ];
            for &(value, name) in expected {
                assert_eq!(
                    CStr::from_ptr(llama_common_speculative_type_to_str_rust(value)).to_bytes(),
                    name
                );
                let c_name = std::ffi::CString::new(name).unwrap();
                assert_eq!(
                    llama_common_speculative_type_from_name_rust(c_name.as_ptr()),
                    value
                );
            }
            assert!(llama_common_speculative_type_to_str_rust(8).is_null());
            assert_eq!(
                llama_common_speculative_type_from_name_rust(c"missing".as_ptr()),
                -1
            );
            assert_eq!(
                llama_common_speculative_type_from_name_rust(std::ptr::null()),
                -1
            );
        }
    }

    #[test]
    fn maps_peg_parse_result_type_names() {
        unsafe {
            assert_eq!(
                CStr::from_ptr(llama_common_peg_parse_result_type_name_rust(0)).to_bytes(),
                b"fail"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_peg_parse_result_type_name_rust(1)).to_bytes(),
                b"success"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_peg_parse_result_type_name_rust(2)).to_bytes(),
                b"need_more_input"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_peg_parse_result_type_name_rust(99)).to_bytes(),
                b"unknown"
            );
        }
    }

    #[test]
    fn formats_peg_rule_names_and_literals() {
        unsafe {
            let rule = llama_common_peg_rule_name_rust(b"tool name!*x".as_ptr(), 12);
            assert_eq!(
                std::slice::from_raw_parts(rule.data, rule.len),
                b"tool-name-x"
            );
            llama_common_unicode_string_free(rule);

            let leading = llama_common_peg_rule_name_rust(b"@root".as_ptr(), 5);
            assert_eq!(
                std::slice::from_raw_parts(leading.data, leading.len),
                b"-root"
            );
            llama_common_unicode_string_free(leading);

            let literal_input = b"a\r\n\"\\b";
            let literal =
                llama_common_gbnf_format_literal_rust(literal_input.as_ptr(), literal_input.len());
            assert_eq!(
                std::slice::from_raw_parts(literal.data, literal.len),
                b"\"a\\r\\n\\\"\\\\b\""
            );
            llama_common_unicode_string_free(literal);

            let null_literal = llama_common_gbnf_format_literal_rust(std::ptr::null(), 1);
            assert_eq!(
                std::slice::from_raw_parts(null_literal.data, null_literal.len),
                b"\"\""
            );
            llama_common_unicode_string_free(null_literal);
        }
    }

    #[test]
    fn builds_gbnf_repetition() {
        unsafe {
            let optional =
                llama_common_gbnf_build_repetition_rust(b"item".as_ptr(), 4, 0, 1, b"".as_ptr(), 0);
            assert_eq!(
                std::slice::from_raw_parts(optional.data, optional.len),
                b"item?"
            );
            llama_common_unicode_string_free(optional);

            let unbounded = llama_common_gbnf_build_repetition_rust(
                b"item".as_ptr(),
                4,
                1,
                c_int::MAX,
                b"".as_ptr(),
                0,
            );
            assert_eq!(
                std::slice::from_raw_parts(unbounded.data, unbounded.len),
                b"item+"
            );
            llama_common_unicode_string_free(unbounded);

            let ranged =
                llama_common_gbnf_build_repetition_rust(b"item".as_ptr(), 4, 2, 5, b"".as_ptr(), 0);
            assert_eq!(
                std::slice::from_raw_parts(ranged.data, ranged.len),
                b"item{2,5}"
            );
            llama_common_unicode_string_free(ranged);

            let separated = llama_common_gbnf_build_repetition_rust(
                b"value".as_ptr(),
                5,
                0,
                3,
                b"\",\" space".as_ptr(),
                9,
            );
            assert_eq!(
                std::slice::from_raw_parts(separated.data, separated.len),
                b"(value (\",\" space value){0,2})?"
            );
            llama_common_unicode_string_free(separated);

            let none =
                llama_common_gbnf_build_repetition_rust(b"item".as_ptr(), 4, 0, 0, b"".as_ptr(), 0);
            assert_eq!(none.len, 0);
            llama_common_unicode_string_free(none);
        }
    }

    #[test]
    fn escapes_gbnf_char_classes() {
        unsafe {
            for (c, expected) in [
                (b'-' as u32, br"\-".as_slice()),
                (b']' as u32, br"\]".as_slice()),
                (b'[' as u32, br"\[".as_slice()),
                (b'\\' as u32, br"\\".as_slice()),
                (b'\n' as u32, br"\n".as_slice()),
                (b'\t' as u32, br"\t".as_slice()),
                (b'\r' as u32, br"\r".as_slice()),
                (b'A' as u32, b"A".as_slice()),
                (0x1f, br"\x1F".as_slice()),
                (0x1234, br"\u1234".as_slice()),
                (0x1f600, br"\U0001F600".as_slice()),
            ] {
                let escaped = llama_common_gbnf_escape_char_class_rust(c);
                assert_eq!(
                    std::slice::from_raw_parts(escaped.data, escaped.len),
                    expected
                );
                llama_common_unicode_string_free(escaped);
            }
        }
    }

    #[test]
    fn detects_gbnf_reserved_names() {
        unsafe {
            for name in [
                b"root".as_slice(),
                b"boolean".as_slice(),
                b"decimal-part".as_slice(),
                b"date-time-string".as_slice(),
                b"uuid".as_slice(),
            ] {
                assert!(llama_common_gbnf_is_reserved_name_rust(
                    name.as_ptr(),
                    name.len()
                ));
            }

            for name in [
                b"Root".as_slice(),
                b"user".as_slice(),
                b"date_time".as_slice(),
            ] {
                assert!(!llama_common_gbnf_is_reserved_name_rust(
                    name.as_ptr(),
                    name.len()
                ));
            }

            assert!(!llama_common_gbnf_is_reserved_name_rust(
                std::ptr::null(),
                1
            ));
        }
    }

    #[test]
    fn formats_gbnf_ref_names() {
        unsafe {
            let input = b"#/$defs/user.name";
            let ref_name = llama_common_gbnf_ref_name_rust(input.as_ptr(), input.len());
            assert_eq!(
                std::slice::from_raw_parts(ref_name.data, ref_name.len),
                b"ref-defs-user-name"
            );
            llama_common_unicode_string_free(ref_name);

            let input = b"https://x/y#foo_bar";
            let external = llama_common_gbnf_ref_name_rust(input.as_ptr(), input.len());
            assert_eq!(
                std::slice::from_raw_parts(external.data, external.len),
                b"reffoo-bar"
            );
            llama_common_unicode_string_free(external);

            let input = b"plain/ref";
            let no_hash = llama_common_gbnf_ref_name_rust(input.as_ptr(), input.len());
            assert_eq!(
                std::slice::from_raw_parts(no_hash.data, no_hash.len),
                b"refplain-ref"
            );
            llama_common_unicode_string_free(no_hash);

            let null = llama_common_gbnf_ref_name_rust(std::ptr::null(), 1);
            assert_eq!(std::slice::from_raw_parts(null.data, null.len), b"ref");
            llama_common_unicode_string_free(null);
        }
    }

    #[test]
    fn maps_hf_cache_repo_folder_names() {
        unsafe {
            let repo =
                llama_common_hf_folder_name_to_repo_rust(b"models--owner--repo-name".as_ptr(), 24);
            assert_eq!(
                std::slice::from_raw_parts(repo.data, repo.len),
                b"owner/repo-name"
            );
            llama_common_unicode_string_free(repo);

            let missing_input = b"datasets--owner--repo";
            let missing = llama_common_hf_folder_name_to_repo_rust(
                missing_input.as_ptr(),
                missing_input.len(),
            );
            assert_eq!(missing.len, 0);
            llama_common_unicode_string_free(missing);

            let folder = llama_common_hf_repo_to_folder_name_rust(b"owner/repo/name".as_ptr(), 15);
            assert_eq!(
                std::slice::from_raw_parts(folder.data, folder.len),
                b"models--owner--repo--name"
            );
            llama_common_unicode_string_free(folder);

            let old = llama_common_hf_make_old_cache_filename_rust(
                b"owner".as_ptr(),
                5,
                b"repo".as_ptr(),
                4,
                b"sub/dir/model.gguf".as_ptr(),
                18,
            );
            assert_eq!(
                std::slice::from_raw_parts(old.data, old.len),
                b"owner_repo_sub_dir_model.gguf"
            );
            llama_common_unicode_string_free(old);
        }
    }

    #[test]
    fn counts_json_brace_depth() {
        unsafe {
            let balanced = br#"{"a": {"b": 1}, "c": "ignore { this }"}"#;
            assert_eq!(
                llama_common_json_brace_depth_rust(balanced.as_ptr(), balanced.len()),
                0
            );

            let open = br#"{"a": {"text": "escaped \" } still string"}"#;
            assert_eq!(
                llama_common_json_brace_depth_rust(open.as_ptr(), open.len()),
                1
            );

            let extra_close = br#"{"a": 1}}"#;
            assert_eq!(
                llama_common_json_brace_depth_rust(extra_close.as_ptr(), extra_close.len()),
                -1
            );

            assert_eq!(llama_common_json_brace_depth_rust(std::ptr::null(), 1), 0);
        }
    }

    #[test]
    fn validates_hf_identifiers() {
        unsafe {
            assert!(llama_common_hf_is_valid_repo_id_rust(
                b"owner/repo".as_ptr(),
                10
            ));
            let repo_with_specials = b"own_er/re.po-1";
            assert!(llama_common_hf_is_valid_repo_id_rust(
                repo_with_specials.as_ptr(),
                repo_with_specials.len()
            ));
            assert!(!llama_common_hf_is_valid_repo_id_rust(b"owner".as_ptr(), 5));
            assert!(!llama_common_hf_is_valid_repo_id_rust(
                b"/owner/repo".as_ptr(),
                11
            ));
            assert!(!llama_common_hf_is_valid_repo_id_rust(
                b"owner//repo".as_ptr(),
                11
            ));
            assert!(!llama_common_hf_is_valid_repo_id_rust(
                b"owner/repo/extra".as_ptr(),
                16
            ));
            assert!(!llama_common_hf_is_valid_repo_id_rust(std::ptr::null(), 1));

            let valid_token = b"hf_abcdefghijklmnopqrstuvwxyzABCDEFGH";
            assert!(llama_common_hf_is_valid_token_rust(
                valid_token.as_ptr(),
                valid_token.len()
            ));
            assert!(!llama_common_hf_is_valid_token_rust(
                b"hf_short".as_ptr(),
                8
            ));
            assert!(!llama_common_hf_is_valid_token_rust(
                b"hf_abcdefghijklmnopqrstuvwxyzABCDEFG!".as_ptr(),
                37
            ));

            let sha40 = [b'a'; 40];
            let sha64 = [b'F'; 64];
            assert!(llama_common_hf_is_valid_commit_rust(
                sha40.as_ptr(),
                sha40.len()
            ));
            assert!(llama_common_hf_is_valid_oid_rust(
                sha40.as_ptr(),
                sha40.len()
            ));
            assert!(llama_common_hf_is_valid_oid_rust(
                sha64.as_ptr(),
                sha64.len()
            ));
            assert!(!llama_common_hf_is_valid_commit_rust(
                sha64.as_ptr(),
                sha64.len()
            ));
            assert!(!llama_common_hf_is_valid_oid_rust(b"not-hex".as_ptr(), 7));
        }
    }

    #[test]
    fn parses_hf_manifest_names() {
        unsafe {
            let valid = b"manifest=owner=repo=model.gguf.json";
            let parsed = llama_common_hf_parse_manifest_name_rust(valid.as_ptr(), valid.len());
            assert_eq!(parsed.len, 2);
            assert_eq!(
                std::slice::from_raw_parts((*parsed.data.add(0)).data, (*parsed.data.add(0)).len),
                b"owner"
            );
            assert_eq!(
                std::slice::from_raw_parts((*parsed.data.add(1)).data, (*parsed.data.add(1)).len),
                b"repo"
            );
            llama_common_string_list_free(parsed);

            let bad_prefix_input = b"other=owner=repo=file.json";
            let bad_prefix = llama_common_hf_parse_manifest_name_rust(
                bad_prefix_input.as_ptr(),
                bad_prefix_input.len(),
            );
            assert_eq!(bad_prefix.len, 0);
            llama_common_string_list_free(bad_prefix);

            let missing_repo_input = b"manifest=owner==file.json";
            let missing_repo = llama_common_hf_parse_manifest_name_rust(
                missing_repo_input.as_ptr(),
                missing_repo_input.len(),
            );
            assert_eq!(missing_repo.len, 0);
            llama_common_string_list_free(missing_repo);

            let bad_suffix_input = b"manifest=owner=repo=file.txt";
            let bad_suffix = llama_common_hf_parse_manifest_name_rust(
                bad_suffix_input.as_ptr(),
                bad_suffix_input.len(),
            );
            assert_eq!(bad_suffix.len, 0);
            llama_common_string_list_free(bad_suffix);

            let null = llama_common_hf_parse_manifest_name_rust(std::ptr::null(), 1);
            assert!(null.data.is_null());
        }
    }

    #[test]
    fn parses_gguf_split_info() {
        unsafe {
            let split_input = b"folder/model-Q4_K_M-00002-of-00008.gguf";
            let split =
                llama_common_gguf_split_info_rust(split_input.as_ptr(), split_input.len(), true);
            assert_eq!(
                std::slice::from_raw_parts(split.prefix.data, split.prefix.len),
                b"folder/model-Q4_K_M"
            );
            assert_eq!(
                std::slice::from_raw_parts(split.tag.data, split.tag.len),
                b"Q4_K_M"
            );
            assert_eq!(split.index, 2);
            assert_eq!(split.count, 8);
            llama_common_unicode_string_free(split.prefix);
            llama_common_unicode_string_free(split.tag);

            let unsplit_input = b"model.f16.gguf";
            let unsplit = llama_common_gguf_split_info_rust(
                unsplit_input.as_ptr(),
                unsplit_input.len(),
                true,
            );
            assert_eq!(
                std::slice::from_raw_parts(unsplit.prefix.data, unsplit.prefix.len),
                b"model.f16"
            );
            assert_eq!(
                std::slice::from_raw_parts(unsplit.tag.data, unsplit.tag.len),
                b"F16"
            );
            assert_eq!(unsplit.index, 1);
            assert_eq!(unsplit.count, 1);
            llama_common_unicode_string_free(unsplit.prefix);
            llama_common_unicode_string_free(unsplit.tag);

            let no_tag_input = b"model-q8_0.gguf";
            let no_tag =
                llama_common_gguf_split_info_rust(no_tag_input.as_ptr(), no_tag_input.len(), false);
            assert_eq!(
                std::slice::from_raw_parts(no_tag.tag.data, no_tag.tag.len),
                b""
            );
            llama_common_unicode_string_free(no_tag.prefix);
            llama_common_unicode_string_free(no_tag.tag);

            let invalid = llama_common_gguf_split_info_rust(b"model.bin".as_ptr(), 9, true);
            assert_eq!(invalid.prefix.len, 0);
            assert_eq!(invalid.tag.len, 0);
            assert_eq!(invalid.index, 0);
            assert_eq!(invalid.count, 0);
            llama_common_unicode_string_free(invalid.prefix);
            llama_common_unicode_string_free(invalid.tag);
        }
    }

    #[test]
    fn extracts_gguf_quant_bits() {
        unsafe {
            let q4 = b"folder/model-Q4_K_M-00002-of-00008.gguf";
            assert_eq!(
                llama_common_gguf_extract_quant_bits_rust(q4.as_ptr(), q4.len()),
                4
            );

            let f16 = b"model.f16.gguf";
            assert_eq!(
                llama_common_gguf_extract_quant_bits_rust(f16.as_ptr(), f16.len()),
                16
            );

            let nvfp4 = b"model-NVFP4.gguf";
            assert_eq!(
                llama_common_gguf_extract_quant_bits_rust(nvfp4.as_ptr(), nvfp4.len()),
                4
            );

            let no_tag = b"model.gguf";
            assert_eq!(
                llama_common_gguf_extract_quant_bits_rust(no_tag.as_ptr(), no_tag.len()),
                0
            );

            assert_eq!(
                llama_common_gguf_extract_quant_bits_rust(std::ptr::null(), 1),
                0
            );
        }
    }

    #[test]
    fn compares_ascii_case_insensitively() {
        unsafe {
            assert_eq!(
                llama_common_ascii_case_equal(c"adamw".as_ptr(), c"ADAMW".as_ptr()),
                1
            );
            assert_eq!(
                llama_common_ascii_case_equal(c"sgd".as_ptr(), c"SgD".as_ptr()),
                1
            );
            assert_eq!(
                llama_common_ascii_case_equal(c"sgd".as_ptr(), c"sgdx".as_ptr()),
                0
            );
            assert_eq!(
                llama_common_ascii_case_equal(std::ptr::null(), c"sgd".as_ptr()),
                0
            );
        }
    }

    #[test]
    fn maps_optimizer_names() {
        unsafe {
            assert_eq!(llama_common_opt_get_optimizer_rust(c"adamw".as_ptr()), 0);
            assert_eq!(llama_common_opt_get_optimizer_rust(c"ADAMW".as_ptr()), 0);
            assert_eq!(llama_common_opt_get_optimizer_rust(c"sgd".as_ptr()), 1);
            assert_eq!(llama_common_opt_get_optimizer_rust(c"SgD".as_ptr()), 1);
            assert_eq!(llama_common_opt_get_optimizer_rust(c"missing".as_ptr()), 2);
            assert_eq!(llama_common_opt_get_optimizer_rust(std::ptr::null()), 2);
        }
    }

    #[test]
    fn classifies_arg_boolean_values() {
        unsafe {
            assert_eq!(llama_common_arg_is_truthy_rust(c"on".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_truthy_rust(c"enabled".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_truthy_rust(c"true".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_truthy_rust(c"1".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_truthy_rust(c"yes".as_ptr()), 0);

            assert_eq!(llama_common_arg_is_falsey_rust(c"off".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_falsey_rust(c"disabled".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_falsey_rust(c"false".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_falsey_rust(c"0".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_falsey_rust(c"no".as_ptr()), 0);

            assert_eq!(llama_common_arg_is_autoy_rust(c"auto".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_autoy_rust(c"-1".as_ptr()), 1);
            assert_eq!(llama_common_arg_is_autoy_rust(c"automatic".as_ptr()), 0);
            assert_eq!(llama_common_arg_is_autoy_rust(std::ptr::null()), 0);
        }
    }

    #[test]
    fn parses_bool_values() {
        unsafe {
            assert_eq!(llama_common_parse_bool_value_rust(c"true".as_ptr()), 1);
            assert_eq!(llama_common_parse_bool_value_rust(c"1".as_ptr()), 1);
            assert_eq!(llama_common_parse_bool_value_rust(c"false".as_ptr()), 0);
            assert_eq!(llama_common_parse_bool_value_rust(c"0".as_ptr()), 0);
            assert_eq!(llama_common_parse_bool_value_rust(c"maybe".as_ptr()), -1);
            assert_eq!(llama_common_parse_bool_value_rust(std::ptr::null()), -1);
        }
    }

    #[test]
    fn parses_negated_bool_args() {
        unsafe {
            let neg = [StringView {
                data: b"--no-cache".as_ptr(),
                len: "--no-cache".len(),
            }];

            let flipped_false = llama_common_parse_bool_arg_rust(
                neg.as_ptr(),
                neg.len(),
                b"no-cache".as_ptr(),
                "no-cache".len(),
                b"true".as_ptr(),
                "true".len(),
            );
            assert_eq!(
                std::slice::from_raw_parts(flipped_false.data, flipped_false.len),
                b"false"
            );
            llama_common_unicode_string_free(flipped_false);

            let flipped_true = llama_common_parse_bool_arg_rust(
                neg.as_ptr(),
                neg.len(),
                b"no-cache".as_ptr(),
                "no-cache".len(),
                b"0".as_ptr(),
                "0".len(),
            );
            assert_eq!(
                std::slice::from_raw_parts(flipped_true.data, flipped_true.len),
                b"true"
            );
            llama_common_unicode_string_free(flipped_true);

            let unchanged = llama_common_parse_bool_arg_rust(
                neg.as_ptr(),
                neg.len(),
                b"cache".as_ptr(),
                "cache".len(),
                b"auto".as_ptr(),
                "auto".len(),
            );
            assert_eq!(
                std::slice::from_raw_parts(unchanged.data, unchanged.len),
                b"auto"
            );
            llama_common_unicode_string_free(unchanged);
        }
    }

    #[test]
    fn parses_openai_compatible_tool_choice() {
        unsafe {
            assert_eq!(
                llama_common_chat_tool_choice_parse_oaicompat_rust(c"auto".as_ptr()),
                0
            );
            assert_eq!(
                llama_common_chat_tool_choice_parse_oaicompat_rust(c"required".as_ptr()),
                1
            );
            assert_eq!(
                llama_common_chat_tool_choice_parse_oaicompat_rust(c"none".as_ptr()),
                2
            );
            assert_eq!(
                llama_common_chat_tool_choice_parse_oaicompat_rust(c"AUTO".as_ptr()),
                -1
            );
            assert_eq!(
                llama_common_chat_tool_choice_parse_oaicompat_rust(std::ptr::null()),
                -1
            );
        }
    }

    #[test]
    fn formats_line_and_column() {
        unsafe {
            let source = b"one\ntwo\nthree";
            let start = llama_common_get_line_col_rust(source.as_ptr(), source.len(), 0);
            assert_eq!(
                std::slice::from_raw_parts(start.data, start.len),
                b"line 1, column 1"
            );
            llama_common_unicode_string_free(start);

            let second_line = llama_common_get_line_col_rust(source.as_ptr(), source.len(), 5);
            assert_eq!(
                std::slice::from_raw_parts(second_line.data, second_line.len),
                b"line 2, column 2"
            );
            llama_common_unicode_string_free(second_line);

            let past_end = llama_common_get_line_col_rust(source.as_ptr(), source.len(), 99);
            assert_eq!(
                std::slice::from_raw_parts(past_end.data, past_end.len),
                b"line 3, column 6"
            );
            llama_common_unicode_string_free(past_end);

            let null = llama_common_get_line_col_rust(std::ptr::null(), 1, 10);
            assert_eq!(
                std::slice::from_raw_parts(null.data, null.len),
                b"line 1, column 1"
            );
            llama_common_unicode_string_free(null);
        }
    }

    #[test]
    fn parses_cpu_hex_masks() {
        unsafe {
            let mut mask = [false; 16];
            assert_eq!(
                llama_common_parse_cpu_mask_rust(b"0x9".as_ptr(), 3, mask.as_mut_ptr(), mask.len()),
                -1
            );
            assert!(mask[0]);
            assert!(!mask[1]);
            assert!(!mask[2]);
            assert!(mask[3]);

            let mut mask = [false; 16];
            assert_eq!(
                llama_common_parse_cpu_mask_rust(b"3f".as_ptr(), 2, mask.as_mut_ptr(), mask.len()),
                -1
            );
            assert_eq!(
                &mask[..8],
                &[true, true, true, true, true, true, false, false]
            );

            assert_eq!(
                llama_common_parse_cpu_mask_rust(b"0xG".as_ptr(), 3, mask.as_mut_ptr(), mask.len()),
                2
            );
        }
    }

    #[test]
    fn cpu_hex_mask_parser_preserves_empty_and_truncation_behavior() {
        unsafe {
            let mut mask = [false; 520];
            assert_eq!(
                llama_common_parse_cpu_mask_rust(b"0x".as_ptr(), 2, mask.as_mut_ptr(), mask.len()),
                -1
            );
            assert!(mask.iter().all(|value| !*value));

            let mut long = vec![b'0'; 129];
            long[128] = b'G';
            assert_eq!(
                llama_common_parse_cpu_mask_rust(
                    long.as_ptr(),
                    long.len(),
                    mask.as_mut_ptr(),
                    mask.len()
                ),
                -1
            );
        }
    }

    #[test]
    fn maps_kv_cache_types() {
        unsafe {
            assert_eq!(llama_common_kv_cache_type_from_str_rust(c"f32".as_ptr()), 0);
            assert_eq!(llama_common_kv_cache_type_from_str_rust(c"f16".as_ptr()), 1);
            assert_eq!(
                llama_common_kv_cache_type_from_str_rust(c"bf16".as_ptr()),
                30
            );
            assert_eq!(
                llama_common_kv_cache_type_from_str_rust(c"q8_0".as_ptr()),
                8
            );
            assert_eq!(
                llama_common_kv_cache_type_from_str_rust(c"q4_0".as_ptr()),
                2
            );
            assert_eq!(
                llama_common_kv_cache_type_from_str_rust(c"q4_1".as_ptr()),
                3
            );
            assert_eq!(
                llama_common_kv_cache_type_from_str_rust(c"iq4_nl".as_ptr()),
                20
            );
            assert_eq!(
                llama_common_kv_cache_type_from_str_rust(c"q5_0".as_ptr()),
                6
            );
            assert_eq!(
                llama_common_kv_cache_type_from_str_rust(c"q5_1".as_ptr()),
                7
            );
            assert_eq!(
                llama_common_kv_cache_type_from_str_rust(c"missing".as_ptr()),
                -1
            );
            assert_eq!(
                llama_common_kv_cache_type_from_str_rust(std::ptr::null()),
                -1
            );
        }
    }

    #[test]
    fn formats_kv_cache_type_list() {
        unsafe {
            assert_eq!(
                CStr::from_ptr(llama_common_get_all_kv_cache_types_rust()).to_bytes(),
                b"f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1"
            );
        }
    }

    #[test]
    fn normalizes_embeddings() {
        unsafe {
            let input = [3.0_f32, 4.0];
            let mut output = [0.0_f32; 2];
            llama_common_embd_normalize_rust(
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len() as c_int,
                2,
            );
            assert!((output[0] - 0.6).abs() < 1e-6);
            assert!((output[1] - 0.8).abs() < 1e-6);

            llama_common_embd_normalize_rust(
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len() as c_int,
                -1,
            );
            assert_eq!(output, input);

            let input = [32760.0_f32, -16380.0];
            llama_common_embd_normalize_rust(
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len() as c_int,
                0,
            );
            assert!((output[0] - 32760.0).abs() < 1e-3);
            assert!((output[1] + 16380.0).abs() < 1e-3);
        }
    }

    #[test]
    fn computes_embedding_cosine_similarity() {
        unsafe {
            let a = [1.0_f32, 0.0];
            let b = [0.0_f32, 1.0];
            assert_eq!(
                llama_common_embd_similarity_cos_rust(a.as_ptr(), b.as_ptr(), 2),
                0.0
            );

            let c = [2.0_f32, 0.0];
            assert_eq!(
                llama_common_embd_similarity_cos_rust(a.as_ptr(), c.as_ptr(), 2),
                1.0
            );

            let z = [0.0_f32, 0.0];
            assert_eq!(
                llama_common_embd_similarity_cos_rust(z.as_ptr(), z.as_ptr(), 2),
                1.0
            );
            assert_eq!(
                llama_common_embd_similarity_cos_rust(a.as_ptr(), z.as_ptr(), 2),
                0.0
            );
        }
    }

    #[test]
    fn initializes_learning_rate_decay() {
        unsafe {
            let mut decay_epochs = -1.0_f32;
            let mut scale_epoch = 0.0_f32;
            llama_common_lr_opt_init_rust(
                1.0e-3,
                1.0e-5,
                -1.0,
                4,
                &mut decay_epochs,
                &mut scale_epoch,
            );
            assert_eq!(decay_epochs, 4.0);
            assert!((scale_epoch - 1.660_964).abs() < 1e-5);

            llama_common_lr_opt_init_rust(
                1.0e-3,
                1.0e-5,
                2.0,
                4,
                &mut decay_epochs,
                &mut scale_epoch,
            );
            assert_eq!(decay_epochs, 2.0);
            assert!((scale_epoch - 3.321_928).abs() < 1e-5);

            decay_epochs = -1.0;
            scale_epoch = 7.0;
            llama_common_lr_opt_init_rust(
                1.0e-3,
                -1.0,
                decay_epochs,
                4,
                &mut decay_epochs,
                &mut scale_epoch,
            );
            assert_eq!(decay_epochs, -1.0);
            assert_eq!(scale_epoch, 0.0);
        }
    }

    #[test]
    fn computes_learning_rate_schedule() {
        assert_eq!(
            llama_common_lr_opt_get_lr_rust(1.0e-3, -1.0, -1.0, 0.0, 99.0),
            1.0e-3
        );
        assert_eq!(
            llama_common_lr_opt_get_lr_rust(1.0e-3, 1.0e-5, 2.0, 3.321_928, 2.0),
            1.0e-5
        );
        let lr = llama_common_lr_opt_get_lr_rust(1.0e-3, 1.0e-5, 2.0, 3.321_928, 1.0);
        assert!((lr - 1.0e-4).abs() < 1e-8);
    }

    #[test]
    fn maps_log_levels_to_verbosity() {
        assert_eq!(llama_common_log_level_to_verbosity_rust(1), 4);
        assert_eq!(llama_common_log_level_to_verbosity_rust(2), 3);
        assert_eq!(llama_common_log_level_to_verbosity_rust(3), 2);
        assert_eq!(llama_common_log_level_to_verbosity_rust(4), 1);
        assert_eq!(llama_common_log_level_to_verbosity_rust(5), 3);
        assert_eq!(llama_common_log_level_to_verbosity_rust(0), 0);
        assert_eq!(llama_common_log_level_to_verbosity_rust(99), 0);
    }

    #[test]
    fn selects_default_thread_counts() {
        assert_eq!(llama_common_default_thread_count_windows_rust(12, 4), 12);
        assert_eq!(llama_common_default_thread_count_windows_rust(0, 4), 4);
        assert_eq!(llama_common_default_thread_count_windows_rust(-1, 6), 6);

        assert_eq!(llama_common_default_thread_count_rust(0), 4);
        assert_eq!(llama_common_default_thread_count_rust(1), 1);
        assert_eq!(llama_common_default_thread_count_rust(4), 4);
        assert_eq!(llama_common_default_thread_count_rust(6), 3);
        assert_eq!(llama_common_default_thread_count_rust(16), 8);
    }

    #[test]
    fn compares_sampler_probabilities_descending() {
        assert!(llama_common_sampler_prob_desc_rust(0.75, 0.25));
        assert!(!llama_common_sampler_prob_desc_rust(0.25, 0.75));
        assert!(!llama_common_sampler_prob_desc_rust(0.5, 0.5));
    }

    #[test]
    fn classifies_marker_delimiters() {
        assert!(llama_common_marker_is_opener_rust(b'<'));
        assert!(llama_common_marker_is_opener_rust(b'['));
        assert!(!llama_common_marker_is_opener_rust(b'{'));

        assert!(llama_common_marker_is_closer_rust(b'<', b'>'));
        assert!(llama_common_marker_is_closer_rust(b'[', b']'));
        assert!(!llama_common_marker_is_closer_rust(b'<', b']'));
        assert!(!llama_common_marker_is_closer_rust(b'[', b'>'));
    }

    #[test]
    fn checks_content_always_wrapped_state() {
        assert!(llama_common_content_is_always_wrapped_rust(1, 1, 1));
        assert!(!llama_common_content_is_always_wrapped_rust(0, 1, 1));
        assert!(!llama_common_content_is_always_wrapped_rust(2, 1, 1));
        assert!(!llama_common_content_is_always_wrapped_rust(1, 0, 1));
        assert!(!llama_common_content_is_always_wrapped_rust(1, 1, 0));
    }

    #[test]
    fn checks_platform_runtime_predicates() {
        assert!(llama_common_cpu_has_hybrid_bit_rust(1 << 15));
        assert!(!llama_common_cpu_has_hybrid_bit_rust(1 << 14));

        assert!(llama_common_cpu_is_intel_atom_core_type_rust(0x20 << 24));
        assert!(!llama_common_cpu_is_intel_atom_core_type_rust(0x40 << 24));

        assert!(llama_common_any_terminal_rust(true, false));
        assert!(llama_common_any_terminal_rust(false, true));
        assert!(!llama_common_any_terminal_rust(false, false));

        assert!(llama_common_arg_has_env_value_rust(true, true, false));
        assert!(llama_common_arg_has_env_value_rust(true, false, true));
        assert!(!llama_common_arg_has_env_value_rust(true, false, false));
        assert!(!llama_common_arg_has_env_value_rust(false, false, true));
    }

    #[test]
    fn checks_jinja_numeric_predicates() {
        assert!(llama_common_jinja_int_is_odd_rust(3));
        assert!(llama_common_jinja_int_is_odd_rust(-3));
        assert!(!llama_common_jinja_int_is_odd_rust(4));

        assert!(llama_common_jinja_int_is_even_rust(4));
        assert!(llama_common_jinja_int_is_even_rust(-4));
        assert!(!llama_common_jinja_int_is_even_rust(3));

        assert_eq!(llama_common_jinja_int_abs_rust(5), 5);
        assert_eq!(llama_common_jinja_int_abs_rust(-5), 5);
        assert_eq!(llama_common_jinja_int_abs_rust(i64::MIN), i64::MIN);

        assert_eq!(llama_common_jinja_float_abs_rust(2.5), 2.5);
        assert_eq!(llama_common_jinja_float_abs_rust(-2.5), 2.5);
        assert_eq!(
            llama_common_jinja_float_abs_rust(-0.0).to_bits(),
            (-0.0f64).to_bits()
        );
    }

    #[test]
    fn checks_jinja_boolean_state_predicates() {
        assert!(llama_common_jinja_is_false_rust(true, false));
        assert!(!llama_common_jinja_is_false_rust(true, true));
        assert!(!llama_common_jinja_is_false_rust(false, false));

        assert!(llama_common_jinja_is_true_rust(true, true));
        assert!(!llama_common_jinja_is_true_rust(true, false));
        assert!(!llama_common_jinja_is_true_rust(false, true));

        assert!(llama_common_jinja_is_defined_rust(false));
        assert!(!llama_common_jinja_is_defined_rust(true));
    }

    #[test]
    fn compares_jinja_scalar_values_by_compare_op() {
        assert!(llama_common_jinja_compare_f64_rust(2.0, 2.0, 0));
        assert!(llama_common_jinja_compare_f64_rust(3.0, 2.0, 1));
        assert!(llama_common_jinja_compare_f64_rust(3.0, 2.0, 2));
        assert!(llama_common_jinja_compare_f64_rust(1.0, 2.0, 3));
        assert!(llama_common_jinja_compare_f64_rust(1.0, 2.0, 4));
        assert!(llama_common_jinja_compare_f64_rust(1.0, 2.0, 5));
        assert!(!llama_common_jinja_compare_f64_rust(1.0, 2.0, 99));

        assert_eq!(llama_common_jinja_arithmetic_f64_rust(2.0, 3.0, 0), 5.0);
        assert_eq!(llama_common_jinja_arithmetic_f64_rust(2.0, 3.0, 1), -1.0);
        assert_eq!(llama_common_jinja_arithmetic_f64_rust(2.0, 3.0, 2), 6.0);
        assert_eq!(llama_common_jinja_arithmetic_f64_rust(6.0, 3.0, 3), 2.0);
        assert_eq!(llama_common_jinja_arithmetic_f64_rust(7.0, 3.0, 4), 1.0);
        assert!(llama_common_jinja_arithmetic_f64_rust(1.0, 2.0, 99).is_nan());

        assert!(llama_common_jinja_compare_bool_rust(true, true, 0));
        assert!(llama_common_jinja_compare_bool_rust(true, false, 4));
        assert!(!llama_common_jinja_compare_bool_rust(true, false, 1));

        assert!(llama_common_jinja_bool_not_rust(false));
        assert!(!llama_common_jinja_bool_not_rust(true));
        assert!(llama_common_jinja_membership_result_rust(true, false));
        assert!(!llama_common_jinja_membership_result_rust(false, false));
        assert!(!llama_common_jinja_membership_result_rust(true, true));
        assert!(llama_common_jinja_membership_result_rust(false, true));

        unsafe {
            assert!(llama_common_jinja_compare_bytes_rust(
                b"abc".as_ptr(),
                3,
                b"abc".as_ptr(),
                3,
                0
            ));
            assert!(llama_common_jinja_compare_bytes_rust(
                b"abd".as_ptr(),
                3,
                b"abc".as_ptr(),
                3,
                1
            ));
            assert!(llama_common_jinja_compare_bytes_rust(
                b"abd".as_ptr(),
                3,
                b"abc".as_ptr(),
                3,
                2
            ));
            assert!(llama_common_jinja_compare_bytes_rust(
                b"abc".as_ptr(),
                3,
                b"abd".as_ptr(),
                3,
                3
            ));
            assert!(llama_common_jinja_compare_bytes_rust(
                b"abc".as_ptr(),
                3,
                b"abd".as_ptr(),
                3,
                4
            ));
            assert!(!llama_common_jinja_compare_bytes_rust(
                std::ptr::null(),
                1,
                b"abd".as_ptr(),
                3,
                0
            ));
        }
    }

    #[test]
    fn checks_ring_buffer_empty_predicate() {
        assert!(llama_common_ring_buffer_is_empty_rust(0));
        assert!(!llama_common_ring_buffer_is_empty_rust(1));
    }

    #[test]
    fn returns_microsecond_epoch_time() {
        let first = llama_common_time_us_rust();
        let second = llama_common_time_us_rust();
        assert!(first > 0);
        assert!(second >= first);
    }

    #[test]
    fn normalizes_model_endpoints() {
        unsafe {
            let default =
                llama_common_model_endpoint_rust(std::ptr::null(), 0, std::ptr::null(), 0);
            assert_eq!(
                std::slice::from_raw_parts(default.data, default.len),
                b"https://huggingface.co/"
            );
            llama_common_unicode_string_free(default);

            let hf = llama_common_model_endpoint_rust(
                std::ptr::null(),
                0,
                b"https://mirror.example".as_ptr(),
                "https://mirror.example".len(),
            );
            assert_eq!(
                std::slice::from_raw_parts(hf.data, hf.len),
                b"https://mirror.example/"
            );
            llama_common_unicode_string_free(hf);

            let model = llama_common_model_endpoint_rust(
                b"https://models.example/".as_ptr(),
                "https://models.example/".len(),
                b"https://mirror.example".as_ptr(),
                "https://mirror.example".len(),
            );
            assert_eq!(
                std::slice::from_raw_parts(model.data, model.len),
                b"https://models.example/"
            );
            llama_common_unicode_string_free(model);
        }
    }

    #[test]
    fn classifies_http_success_statuses() {
        assert!(!llama_common_is_http_status_ok_rust(199));
        assert!(llama_common_is_http_status_ok_rust(200));
        assert!(llama_common_is_http_status_ok_rust(302));
        assert!(llama_common_is_http_status_ok_rust(399));
        assert!(!llama_common_is_http_status_ok_rust(400));
        assert!(!llama_common_is_http_status_ok_rust(500));
    }

    #[test]
    fn reads_first_etag_line() {
        unsafe {
            let base = std::env::temp_dir().join(format!(
                "llama-common-etag-test-{}-{}",
                std::process::id(),
                llama_common_time_us_rust()
            ));
            let etag = base.with_extension("etag");
            std::fs::write(&etag, "first\nsecond\n").unwrap();

            let base = base.to_string_lossy();
            let value = llama_common_read_etag_rust(base.as_bytes().as_ptr(), base.len());
            assert_eq!(
                std::str::from_utf8(std::slice::from_raw_parts(value.data, value.len)).unwrap(),
                "first"
            );
            llama_common_unicode_string_free(value);

            let missing = format!("{base}.missing");
            let value = llama_common_read_etag_rust(missing.as_bytes().as_ptr(), missing.len());
            assert_eq!(value.len, 0);
            llama_common_unicode_string_free(value);

            let value = llama_common_read_etag_rust(std::ptr::null(), 1);
            assert_eq!(value.len, 0);
            llama_common_unicode_string_free(value);

            let _ = std::fs::remove_file(etag);
        }
    }

    #[test]
    fn reads_cached_ref_from_refs_directory() {
        unsafe {
            let repo = std::env::temp_dir().join(format!(
                "llama-common-ref-test-{}-{}",
                std::process::id(),
                llama_common_time_us_rust()
            ));
            let refs = repo.join("refs");
            std::fs::create_dir_all(&refs).unwrap();
            std::fs::write(
                refs.join("dev"),
                "0123456789abcdef0123456789abcdef01234567\n",
            )
            .unwrap();

            let repo_text = repo.to_string_lossy();
            let value =
                llama_common_get_cached_ref_rust(repo_text.as_bytes().as_ptr(), repo_text.len());
            assert_eq!(
                std::str::from_utf8(std::slice::from_raw_parts(value.data, value.len)).unwrap(),
                "0123456789abcdef0123456789abcdef01234567"
            );
            llama_common_unicode_string_free(value);

            std::fs::write(
                refs.join("main"),
                "ffffffffffffffffffffffffffffffffffffffff\n",
            )
            .unwrap();
            std::fs::write(refs.join("bad"), "not-a-commit\n").unwrap();
            let value =
                llama_common_get_cached_ref_rust(repo_text.as_bytes().as_ptr(), repo_text.len());
            assert_eq!(
                std::str::from_utf8(std::slice::from_raw_parts(value.data, value.len)).unwrap(),
                "ffffffffffffffffffffffffffffffffffffffff"
            );
            llama_common_unicode_string_free(value);

            let missing = format!("{repo_text}.missing");
            let value =
                llama_common_get_cached_ref_rust(missing.as_bytes().as_ptr(), missing.len());
            assert_eq!(value.len, 0);
            llama_common_unicode_string_free(value);

            let _ = std::fs::remove_dir_all(repo);
        }
    }

    #[test]
    fn safely_writes_file_via_temp_path() {
        unsafe {
            let root = std::env::temp_dir().join(format!(
                "llama-common-write-test-{}-{}",
                std::process::id(),
                llama_common_time_us_rust()
            ));
            let target = root.join("nested").join("file.txt");
            let target_text = target.to_string_lossy();
            assert!(llama_common_safe_write_file_rust(
                target_text.as_bytes().as_ptr(),
                target_text.len(),
                b"payload".as_ptr(),
                7,
            ));
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "payload");
            assert!(!target.with_extension("txt.tmp").exists());

            assert!(!llama_common_safe_write_file_rust(
                std::ptr::null(),
                1,
                b"payload".as_ptr(),
                7,
            ));
            assert!(!llama_common_safe_write_file_rust(
                target_text.as_bytes().as_ptr(),
                target_text.len(),
                std::ptr::null(),
                1,
            ));

            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn detects_filesystem_directories() {
        unsafe {
            let root = std::env::temp_dir().join(format!(
                "llama-common-is-dir-test-{}-{}",
                std::process::id(),
                llama_common_time_us_rust()
            ));
            let nested = root.join("nested");
            std::fs::create_dir_all(&nested).unwrap();
            let file = root.join("file.txt");
            std::fs::write(&file, "payload").unwrap();

            let nested_text = nested.to_string_lossy();
            assert!(llama_common_fs_is_directory_rust(
                nested_text.as_bytes().as_ptr(),
                nested_text.len()
            ));

            let file_text = file.to_string_lossy();
            assert!(!llama_common_fs_is_directory_rust(
                file_text.as_bytes().as_ptr(),
                file_text.len()
            ));

            let missing = root.join("missing");
            let missing_text = missing.to_string_lossy();
            assert!(!llama_common_fs_is_directory_rust(
                missing_text.as_bytes().as_ptr(),
                missing_text.len()
            ));
            assert!(!llama_common_fs_is_directory_rust(std::ptr::null(), 1));

            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn reads_files_as_bytes() {
        unsafe {
            let root = std::env::temp_dir().join(format!(
                "llama-common-read-file-test-{}-{}",
                std::process::id(),
                llama_common_time_us_rust()
            ));
            std::fs::create_dir_all(&root).unwrap();
            let file = root.join("input.bin");
            std::fs::write(&file, b"line\n\xff").unwrap();

            let file_text = file.to_string_lossy();
            let content =
                llama_common_read_file_rust(file_text.as_bytes().as_ptr(), file_text.len());
            assert_eq!(
                std::slice::from_raw_parts(content.data, content.len),
                b"line\n\xff"
            );
            llama_common_unicode_string_free(content);

            let missing = root.join("missing.txt");
            let missing_text = missing.to_string_lossy();
            let content =
                llama_common_read_file_rust(missing_text.as_bytes().as_ptr(), missing_text.len());
            assert!(content.data.is_null());
            llama_common_unicode_string_free(content);

            let content = llama_common_read_file_rust(std::ptr::null(), 1);
            assert!(content.data.is_null());
            llama_common_unicode_string_free(content);

            let _ = std::fs::remove_dir_all(root);
        }
    }

    #[test]
    fn detects_lfm2_templates() {
        unsafe {
            let src = b"prefix <|tool_list_start|>[]<|tool_list_end|> suffix";
            assert!(llama_common_is_lfm2_template_rust(src.as_ptr(), src.len()));

            let missing_end = b"<|tool_list_start|>[]";
            assert!(!llama_common_is_lfm2_template_rust(
                missing_end.as_ptr(),
                missing_end.len()
            ));
            assert!(!llama_common_is_lfm2_template_rust(std::ptr::null(), 1));
        }
    }

    #[test]
    fn classifies_specialized_chat_templates() {
        unsafe {
            let cases: &[(&[u8], c_int)] = &[
                (b"[SYSTEM_PROMPT] [TOOL_CALLS] [ARGS]", 1),
                (b"<|channel|>analysis", 2),
                (b">>>all\n>>>${recipient}", 3),
                (b"<|tool_calls_section_begin|><|tool_call_begin|>", 4),
                (b"<|tool_list_start|>[]<|tool_list_end|>", 5),
                (b"List of tools: []", 6),
                (b"<|role_sep|><|message_sep|>", 7),
                (b"dsml_token function_calls DSML", 8),
                (b"{{ '<|tool_call>call:' }}", 9),
                (b"plain template", 0),
            ];

            for &(src, expected) in cases {
                assert_eq!(
                    llama_common_specialized_chat_template_rust(src.as_ptr(), src.len()),
                    expected
                );
            }

            let mistral_small = b"[SYSTEM_PROMPT] [TOOL_CALLS] [ARGS] [CALL_ID]";
            assert_eq!(
                llama_common_specialized_chat_template_rust(
                    mistral_small.as_ptr(),
                    mistral_small.len()
                ),
                0
            );
            let lfm25_with_lfm2_marker = b"List of tools: [] <|tool_list_start|>";
            assert_eq!(
                llama_common_specialized_chat_template_rust(
                    lfm25_with_lfm2_marker.as_ptr(),
                    lfm25_with_lfm2_marker.len()
                ),
                0
            );
            let gigachat_function_call = b"<|role_sep|><|message_sep|><|function_call|>";
            assert_eq!(
                llama_common_specialized_chat_template_rust(
                    gigachat_function_call.as_ptr(),
                    gigachat_function_call.len()
                ),
                0
            );
            assert_eq!(
                llama_common_specialized_chat_template_rust(std::ptr::null(), 1),
                0
            );
        }
    }

    #[test]
    fn detects_modern_gemma4_templates() {
        unsafe {
            let modern = b"{#- OpenAI Chat Completions: tool support";
            assert!(llama_common_is_gemma4_modern_template_rust(
                modern.as_ptr(),
                modern.len()
            ));
            let old = b"{{ '<|tool_call>call:' }}";
            assert!(!llama_common_is_gemma4_modern_template_rust(
                old.as_ptr(),
                old.len()
            ));
            assert!(!llama_common_is_gemma4_modern_template_rust(
                std::ptr::null(),
                1
            ));
        }
    }

    #[test]
    fn detects_chat_template_workaround_guards() {
        unsafe {
            let gpt_oss =
                b"<|channel|> {% if '<|channel|>analysis<|message|>' in message.content or";
            assert!(llama_common_needs_gpt_oss_channel_template_patch_rust(
                gpt_oss.as_ptr(),
                gpt_oss.len()
            ));
            assert!(!llama_common_needs_gpt_oss_channel_template_patch_rust(
                b"<|channel|>".as_ptr(),
                b"<|channel|>".len()
            ));

            let mistral =
                b"[TOOL_CALLS] {% if (message['content'] is none or message['content'] == '') %}";
            assert!(llama_common_needs_mistral_tool_calls_template_patch_rust(
                mistral.as_ptr(),
                mistral.len()
            ));
            assert!(!llama_common_needs_mistral_tool_calls_template_patch_rust(
                b"[TOOL_CALLS]".as_ptr(),
                b"[TOOL_CALLS]".len()
            ));
            assert!(!llama_common_needs_gpt_oss_channel_template_patch_rust(
                std::ptr::null(),
                1
            ));
            assert!(!llama_common_needs_mistral_tool_calls_template_patch_rust(
                std::ptr::null(),
                1
            ));
        }
    }

    #[test]
    fn decides_when_grammar_should_apply() {
        assert!(!llama_common_grammar_should_apply_rust(
            false, false, false, 0, 0, 2
        ));
        assert!(llama_common_grammar_should_apply_rust(
            true, false, false, 1, 0, 2
        ));
        assert!(llama_common_grammar_should_apply_rust(
            true, true, false, 1, 0, 2
        ));
        assert!(llama_common_grammar_should_apply_rust(
            true, true, true, 0, 0, 2
        ));
        assert!(llama_common_grammar_should_apply_rust(
            true, true, true, 2, 0, 2
        ));
        assert!(!llama_common_grammar_should_apply_rust(
            true, true, true, 1, 0, 2
        ));

        assert!(llama_common_grammar_is_empty_rust(0, 12));
        assert!(llama_common_grammar_is_empty_rust(1, 0));
        assert!(!llama_common_grammar_is_empty_rust(1, 12));
        assert!(!llama_common_grammar_needs_prefill_rust(0));
        assert!(!llama_common_grammar_needs_prefill_rust(1));
        assert!(llama_common_grammar_needs_prefill_rust(2));
        assert!(llama_common_grammar_needs_prefill_rust(3));
    }

    #[test]
    fn checks_common_parameter_presence_flags() {
        assert!(!llama_common_has_logit_bias_rust(0));
        assert!(llama_common_has_logit_bias_rust(1));

        assert!(!llama_common_speculative_has_draft_rust(0, 0));
        assert!(llama_common_speculative_has_draft_rust(1, 0));
        assert!(llama_common_speculative_has_draft_rust(0, 1));
        assert!(llama_common_speculative_has_draft_rust(3, 5));
    }

    #[test]
    fn checks_peg_parse_flags_and_ranges() {
        assert!(llama_common_peg_parse_result_fail_rust(0));
        assert!(!llama_common_peg_parse_result_fail_rust(1));
        assert!(llama_common_peg_parse_result_success_rust(1));
        assert!(!llama_common_peg_parse_result_success_rust(2));
        assert!(llama_common_peg_parse_result_need_more_input_rust(2));
        assert!(!llama_common_peg_parse_result_need_more_input_rust(0));

        assert!(!llama_common_peg_parse_flags_is_lenient_rust(0));
        assert!(llama_common_peg_parse_flags_is_lenient_rust(1));
        assert!(llama_common_peg_parse_flags_is_lenient_rust(3));
        assert!(!llama_common_peg_parse_flags_is_debug_rust(1));
        assert!(llama_common_peg_parse_flags_is_debug_rust(2));
        assert!(llama_common_peg_parse_flags_is_debug_rust(3));

        assert!(llama_common_peg_char_range_contains_rust(
            b'a' as u32,
            b'z' as u32,
            b'm' as u32
        ));
        assert!(llama_common_peg_char_range_contains_rust(
            b'a' as u32,
            b'z' as u32,
            b'a' as u32
        ));
        assert!(llama_common_peg_char_range_contains_rust(
            b'a' as u32,
            b'z' as u32,
            b'z' as u32
        ));
        assert!(!llama_common_peg_char_range_contains_rust(
            b'a' as u32,
            b'z' as u32,
            b'A' as u32
        ));
    }

    #[test]
    fn checks_chat_message_empty_state() {
        assert!(llama_common_chat_msg_is_empty_rust(0, 0, 0, 0, 0, 0));
        assert!(!llama_common_chat_msg_is_empty_rust(1, 0, 0, 0, 0, 0));
        assert!(!llama_common_chat_msg_is_empty_rust(0, 1, 0, 0, 0, 0));
        assert!(!llama_common_chat_msg_is_empty_rust(0, 0, 1, 0, 0, 0));
        assert!(!llama_common_chat_msg_is_empty_rust(0, 0, 0, 1, 0, 0));
        assert!(!llama_common_chat_msg_is_empty_rust(0, 0, 0, 0, 1, 0));
        assert!(!llama_common_chat_msg_is_empty_rust(0, 0, 0, 0, 0, 1));
    }

    #[test]
    fn detects_gguf_model_filenames() {
        unsafe {
            let model = b"Q4_K_M/model.gguf";
            assert!(llama_common_gguf_filename_is_model_rust(
                model.as_ptr(),
                model.len()
            ));
            let parent_with_mmproj = b"folder/mmproj_parent/model.gguf";
            assert!(llama_common_gguf_filename_is_model_rust(
                parent_with_mmproj.as_ptr(),
                parent_with_mmproj.len()
            ));
            let not_gguf = b"model.bin";
            assert!(!llama_common_gguf_filename_is_model_rust(
                not_gguf.as_ptr(),
                not_gguf.len()
            ));
            let mmproj = b"mmproj-model.gguf";
            assert!(!llama_common_gguf_filename_is_model_rust(
                mmproj.as_ptr(),
                mmproj.len()
            ));
            let imatrix = b"imatrix-model.gguf";
            assert!(!llama_common_gguf_filename_is_model_rust(
                imatrix.as_ptr(),
                imatrix.len()
            ));
            assert!(!llama_common_gguf_filename_is_model_rust(
                std::ptr::null(),
                1
            ));
        }
    }

    #[test]
    fn formats_bool_strings() {
        unsafe {
            assert_eq!(
                CStr::from_ptr(llama_common_bool_to_string_rust(true)).to_bytes(),
                b"true"
            );
            assert_eq!(
                CStr::from_ptr(llama_common_bool_to_string_rust(false)).to_bytes(),
                b"false"
            );
        }
    }

    #[test]
    fn formats_source_peeks() {
        unsafe {
            let source = b"abcdef";
            let peek = llama_common_peak_source_rust(source.as_ptr(), source.len(), 3, 2);
            assert_eq!(
                std::slice::from_raw_parts(peek.data, peek.len),
                b"...bcde...\n     ^"
            );
            llama_common_unicode_string_free(peek);

            let multiline = b"ab\ncd";
            let peek = llama_common_peak_source_rust(multiline.as_ptr(), multiline.len(), 2, 4);
            assert_eq!(
                std::slice::from_raw_parts(peek.data, peek.len),
                "...ab↵cd...\n     ^".as_bytes()
            );
            llama_common_unicode_string_free(peek);

            let empty = llama_common_peak_source_rust(b"".as_ptr(), 0, 0, 40);
            assert_eq!(
                std::slice::from_raw_parts(empty.data, empty.len),
                b"(no source available)"
            );
            llama_common_unicode_string_free(empty);
        }
    }

    #[test]
    fn formats_errors_with_source() {
        unsafe {
            let source = b"abcdef";
            let formatted = llama_common_fmt_error_with_source_rust(
                b"lexer".as_ptr(),
                5,
                b"bad token".as_ptr(),
                9,
                source.as_ptr(),
                source.len(),
                3,
            );
            assert_eq!(
                std::slice::from_raw_parts(formatted.data, formatted.len),
                b"lexer: bad token\n...abcdef...\n      ^"
            );
            llama_common_unicode_string_free(formatted);

            let null = llama_common_fmt_error_with_source_rust(
                std::ptr::null(),
                1,
                b"bad".as_ptr(),
                3,
                source.as_ptr(),
                source.len(),
                0,
            );
            assert_eq!(null.len, 0);
            llama_common_unicode_string_free(null);
        }
    }

    #[test]
    fn repeats_strings() {
        unsafe {
            let repeated = llama_common_string_repeat_rust(b"ab".as_ptr(), 2, 3);
            assert!(!repeated.data.is_null());
            assert_eq!(
                std::slice::from_raw_parts(repeated.data, repeated.len),
                b"ababab"
            );
            llama_common_unicode_string_free(repeated);

            let empty = llama_common_string_repeat_rust(b"ab".as_ptr(), 2, 0);
            assert_eq!(empty.len, 0);
            llama_common_unicode_string_free(empty);

            let null = llama_common_string_repeat_rust(std::ptr::null(), 1, 2);
            assert!(null.data.is_null());
            assert_eq!(null.len, 0);
        }
    }

    #[test]
    fn formats_int_vectors() {
        unsafe {
            let values = [1, -2, 300];
            let formatted = llama_common_int_vec_to_string_rust(values.as_ptr(), values.len());
            assert!(!formatted.data.is_null());
            assert_eq!(
                std::slice::from_raw_parts(formatted.data, formatted.len),
                b"[ 1, -2, 300 ]"
            );
            llama_common_unicode_string_free(formatted);

            let empty = llama_common_int_vec_to_string_rust(values.as_ptr(), 0);
            assert_eq!(std::slice::from_raw_parts(empty.data, empty.len), b"[  ]");
            llama_common_unicode_string_free(empty);

            let null = llama_common_int_vec_to_string_rust(std::ptr::null(), 1);
            assert!(null.data.is_null());
            assert_eq!(null.len, 0);
        }
    }

    #[test]
    fn trims_strings() {
        unsafe {
            let both = llama_common_string_trim_rust(b" \tvalue\n".as_ptr(), 8, 0);
            assert_eq!(std::slice::from_raw_parts(both.data, both.len), b"value");
            llama_common_unicode_string_free(both);

            let leading = llama_common_string_trim_rust(b" \tvalue\n".as_ptr(), 8, 1);
            assert_eq!(
                std::slice::from_raw_parts(leading.data, leading.len),
                b"value\n"
            );
            llama_common_unicode_string_free(leading);

            let trailing = llama_common_string_trim_rust(b" \tvalue\n".as_ptr(), 8, 2);
            assert_eq!(
                std::slice::from_raw_parts(trailing.data, trailing.len),
                b" \tvalue"
            );
            llama_common_unicode_string_free(trailing);

            let newlines = llama_common_string_trim_rust(b"value\n\n ".as_ptr(), 8, 3);
            assert_eq!(
                std::slice::from_raw_parts(newlines.data, newlines.len),
                b"value\n\n "
            );
            llama_common_unicode_string_free(newlines);

            let all_ws = llama_common_string_trim_rust(b" \n\t".as_ptr(), 3, 0);
            assert_eq!(all_ws.len, 0);
            llama_common_unicode_string_free(all_ws);
        }
    }

    #[test]
    fn strips_custom_character_sets() {
        unsafe {
            let left = llama_common_string_lstrip_chars_rust(
                b" \t\r\nvalue \n".as_ptr(),
                11,
                c" \t\r\n".as_ptr(),
            );
            assert_eq!(std::slice::from_raw_parts(left.data, left.len), b"value \n");
            llama_common_unicode_string_free(left);

            let right = llama_common_string_rstrip_chars_rust(
                b" \tvalue\r\n".as_ptr(),
                9,
                c" \t\r\n".as_ptr(),
            );
            assert_eq!(
                std::slice::from_raw_parts(right.data, right.len),
                b" \tvalue"
            );
            llama_common_unicode_string_free(right);

            let all =
                llama_common_string_lstrip_chars_rust(b"\r\n\t".as_ptr(), 3, c" \t\r\n".as_ptr());
            assert_eq!(all.len, 0);
            llama_common_unicode_string_free(all);

            let unchanged =
                llama_common_string_rstrip_chars_rust(b"value".as_ptr(), 5, c" \t\r\n".as_ptr());
            assert_eq!(
                std::slice::from_raw_parts(unchanged.data, unchanged.len),
                b"value"
            );
            llama_common_unicode_string_free(unchanged);

            let null = llama_common_string_lstrip_chars_rust(std::ptr::null(), 1, c" ".as_ptr());
            assert!(null.data.is_null());
        }
    }

    #[test]
    fn breaks_strings_into_wrapped_lines() {
        unsafe {
            let wrapped =
                llama_common_break_str_into_lines_rust(b"alpha beta gamma\ndelta".as_ptr(), 22, 10);
            assert_eq!(wrapped.len, 3);
            assert_eq!(
                std::slice::from_raw_parts((*wrapped.data.add(0)).data, (*wrapped.data.add(0)).len),
                b"alpha beta"
            );
            assert_eq!(
                std::slice::from_raw_parts((*wrapped.data.add(1)).data, (*wrapped.data.add(1)).len),
                b"gamma"
            );
            assert_eq!(
                std::slice::from_raw_parts((*wrapped.data.add(2)).data, (*wrapped.data.add(2)).len),
                b"delta"
            );
            llama_common_string_list_free(wrapped);
        }
    }

    #[test]
    fn breaks_strings_like_getline_edge_cases() {
        unsafe {
            let empty = llama_common_break_str_into_lines_rust(b"".as_ptr(), 0, 10);
            assert_eq!(empty.len, 0);
            llama_common_string_list_free(empty);

            let newline = llama_common_break_str_into_lines_rust(b"\n".as_ptr(), 1, 10);
            assert_eq!(newline.len, 1);
            assert_eq!(
                std::slice::from_raw_parts((*newline.data).data, (*newline.data).len),
                b""
            );
            llama_common_string_list_free(newline);

            let trailing = llama_common_break_str_into_lines_rust(b"a\n".as_ptr(), 2, 10);
            assert_eq!(trailing.len, 1);
            assert_eq!(
                std::slice::from_raw_parts((*trailing.data).data, (*trailing.data).len),
                b"a"
            );
            llama_common_string_list_free(trailing);

            let long_word =
                llama_common_break_str_into_lines_rust(b"alphabet soup".as_ptr(), 13, 3);
            assert_eq!(long_word.len, 2);
            assert_eq!(
                std::slice::from_raw_parts(
                    (*long_word.data.add(0)).data,
                    (*long_word.data.add(0)).len
                ),
                b"alphabet"
            );
            assert_eq!(
                std::slice::from_raw_parts(
                    (*long_word.data.add(1)).data,
                    (*long_word.data.add(1)).len
                ),
                b"soup"
            );
            llama_common_string_list_free(long_word);
        }
    }

    #[test]
    fn removes_leading_dashes() {
        unsafe {
            let long = llama_common_rm_leading_dashes_rust(b"--threads".as_ptr(), 9);
            assert_eq!(std::slice::from_raw_parts(long.data, long.len), b"threads");
            llama_common_unicode_string_free(long);

            let short = llama_common_rm_leading_dashes_rust(b"-t".as_ptr(), 2);
            assert_eq!(std::slice::from_raw_parts(short.data, short.len), b"t");
            llama_common_unicode_string_free(short);

            let unchanged = llama_common_rm_leading_dashes_rust(b"LLAMA_ARG_THREADS".as_ptr(), 17);
            assert_eq!(
                std::slice::from_raw_parts(unchanged.data, unchanged.len),
                b"LLAMA_ARG_THREADS"
            );
            llama_common_unicode_string_free(unchanged);

            let only_dashes = llama_common_rm_leading_dashes_rust(b"---".as_ptr(), 3);
            assert_eq!(only_dashes.len, 0);
            llama_common_unicode_string_free(only_dashes);
        }
    }

    #[test]
    fn extracts_around_common_prefixes_and_suffixes() {
        unsafe {
            let before = llama_common_until_common_prefix_rust(
                b"before {\"first\": value".as_ptr(),
                23,
                b"{\"first\":".as_ptr(),
                9,
                b"{\"second\":".as_ptr(),
                10,
            );
            assert_eq!(
                std::slice::from_raw_parts(before.data, before.len),
                b"before "
            );
            llama_common_unicode_string_free(before);

            let after = llama_common_after_common_suffix_rust(
                b"value \"XXXX\"} after".as_ptr(),
                19,
                b"\"XXXX\"}".as_ptr(),
                7,
                b"\"YYYY\"}".as_ptr(),
                7,
            );
            assert_eq!(std::slice::from_raw_parts(after.data, after.len), b" after");
            llama_common_unicode_string_free(after);
        }
    }

    #[test]
    fn common_prefix_suffix_extractors_return_empty_without_shared_anchor() {
        unsafe {
            let no_prefix = llama_common_until_common_prefix_rust(
                b"full".as_ptr(),
                4,
                b"left".as_ptr(),
                4,
                b"right".as_ptr(),
                5,
            );
            assert_eq!(no_prefix.len, 0);
            llama_common_unicode_string_free(no_prefix);

            let missing_prefix = llama_common_until_common_prefix_rust(
                b"full".as_ptr(),
                4,
                b"same-left".as_ptr(),
                9,
                b"same-right".as_ptr(),
                10,
            );
            assert_eq!(missing_prefix.len, 0);
            llama_common_unicode_string_free(missing_prefix);

            let no_suffix = llama_common_after_common_suffix_rust(
                b"full".as_ptr(),
                4,
                b"left".as_ptr(),
                4,
                b"right".as_ptr(),
                5,
            );
            assert_eq!(no_suffix.len, 0);
            llama_common_unicode_string_free(no_suffix);

            let missing_suffix = llama_common_after_common_suffix_rust(
                b"full".as_ptr(),
                4,
                b"left-same".as_ptr(),
                9,
                b"right-same".as_ptr(),
                10,
            );
            assert_eq!(missing_suffix.len, 0);
            llama_common_unicode_string_free(missing_suffix);
        }
    }

    #[test]
    fn ensures_json_ascii_inside_strings() {
        unsafe {
            let input = "\"é 😀\"";
            let escaped =
                llama_common_json_ensure_ascii_preserving_format_rust(input.as_ptr(), input.len());
            assert_eq!(
                std::slice::from_raw_parts(escaped.data, escaped.len),
                br#""\u00e9 \ud83d\ude00""#
            );
            llama_common_unicode_string_free(escaped);

            let ascii_input = br#"{"x":"already \" escaped"}"#;
            let ascii = llama_common_json_ensure_ascii_preserving_format_rust(
                ascii_input.as_ptr(),
                ascii_input.len(),
            );
            assert_eq!(
                std::slice::from_raw_parts(ascii.data, ascii.len),
                br#"{"x":"already \" escaped"}"#
            );
            llama_common_unicode_string_free(ascii);
        }
    }

    #[test]
    fn json_ascii_formatter_preserves_format_and_replaces_invalid_string_bytes() {
        unsafe {
            let invalid = [b'"', 0xC3, b'(', b'"'];
            let escaped = llama_common_json_ensure_ascii_preserving_format_rust(
                invalid.as_ptr(),
                invalid.len(),
            );
            assert_eq!(
                std::slice::from_raw_parts(escaped.data, escaped.len),
                br#""\ufffd(""#
            );
            llama_common_unicode_string_free(escaped);

            let outside = "é \"é\"";
            let escaped = llama_common_json_ensure_ascii_preserving_format_rust(
                outside.as_ptr(),
                outside.len(),
            );
            let mut expected = "é ".as_bytes().to_vec();
            expected.extend_from_slice(br#""\u00e9""#);
            assert_eq!(
                std::slice::from_raw_parts(escaped.data, escaped.len),
                expected.as_slice()
            );
            llama_common_unicode_string_free(escaped);
        }
    }

    #[test]
    fn normalizes_python_style_quotes_to_json() {
        unsafe {
            let input = br#"{'key': 'value'}"#;
            let normalized =
                llama_common_normalize_quotes_to_json_rust(input.as_ptr(), input.len());
            assert_eq!(
                std::slice::from_raw_parts(normalized.data, normalized.len),
                br#"{"key": "value"}"#
            );
            llama_common_unicode_string_free(normalized);

            let code = br#"{'code': 'print(\'hello\')'}"#;
            let normalized = llama_common_normalize_quotes_to_json_rust(code.as_ptr(), code.len());
            assert_eq!(
                std::slice::from_raw_parts(normalized.data, normalized.len),
                br#"{"code": "print('hello')"}"#
            );
            llama_common_unicode_string_free(normalized);

            let mixed = br#"{'msg': 'He said "hi"', "keep": "it's ok"}"#;
            let normalized =
                llama_common_normalize_quotes_to_json_rust(mixed.as_ptr(), mixed.len());
            assert_eq!(
                std::slice::from_raw_parts(normalized.data, normalized.len),
                br#"{"msg": "He said \"hi\"", "keep": "it's ok"}"#
            );
            llama_common_unicode_string_free(normalized);

            let null = llama_common_normalize_quotes_to_json_rust(std::ptr::null(), 1);
            assert!(null.data.is_null());
            assert_eq!(null.len, 0);
        }
    }

    #[test]
    fn computes_common_prefix_and_suffix_lengths() {
        unsafe {
            assert_eq!(
                llama_common_prefix_len_rust(b"abcdef".as_ptr(), 6, b"abcXYZ".as_ptr(), 6),
                3
            );
            assert_eq!(
                llama_common_suffix_len_rust(b"XYZdef".as_ptr(), 6, b"abcdef".as_ptr(), 6),
                3
            );
            assert_eq!(
                llama_common_prefix_len_rust(b"abc".as_ptr(), 3, b"xyz".as_ptr(), 3),
                0
            );
            assert_eq!(
                llama_common_suffix_len_rust(b"abc".as_ptr(), 3, b"xyz".as_ptr(), 3),
                0
            );
            assert_eq!(
                llama_common_prefix_len_rust(std::ptr::null(), 1, b"abc".as_ptr(), 3),
                0
            );
        }
    }

    #[test]
    fn matches_simple_globs() {
        unsafe {
            assert!(llama_common_glob_match_rust(
                b"*.gguf".as_ptr(),
                6,
                b"model.gguf".as_ptr(),
                10
            ));
            assert!(!llama_common_glob_match_rust(
                b"*.gguf".as_ptr(),
                6,
                b"dir/model.gguf".as_ptr(),
                14
            ));
            assert!(llama_common_glob_match_rust(
                b"**/*.gguf".as_ptr(),
                9,
                b"dir/model.gguf".as_ptr(),
                14
            ));
            assert!(llama_common_glob_match_rust(
                b"file-?.txt".as_ptr(),
                10,
                b"file-a.txt".as_ptr(),
                10
            ));
            assert!(!llama_common_glob_match_rust(
                b"file-?.txt".as_ptr(),
                10,
                b"file-/.txt".as_ptr(),
                10
            ));
        }
    }

    #[test]
    fn matches_glob_character_classes() {
        unsafe {
            assert!(llama_common_glob_match_rust(
                b"model-[0-9].gguf".as_ptr(),
                16,
                b"model-7.gguf".as_ptr(),
                12
            ));
            assert!(!llama_common_glob_match_rust(
                b"model-[!0-9].gguf".as_ptr(),
                17,
                b"model-7.gguf".as_ptr(),
                12
            ));
            assert!(llama_common_glob_match_rust(
                b"model-[!0-9].gguf".as_ptr(),
                17,
                b"model-a.gguf".as_ptr(),
                12
            ));
            assert!(llama_common_glob_match_rust(
                b"file-[]].txt".as_ptr(),
                12,
                b"file-].txt".as_ptr(),
                10
            ));
            assert!(llama_common_glob_match_rust(
                b"file-[-].txt".as_ptr(),
                12,
                b"file--.txt".as_ptr(),
                10
            ));
            assert!(llama_common_glob_match_rust(
                b"literal[".as_ptr(),
                8,
                b"literal[".as_ptr(),
                8
            ));
            assert!(!llama_common_glob_match_rust(
                b"literal[".as_ptr(),
                8,
                b"literalx".as_ptr(),
                8
            ));
        }
    }

    #[test]
    fn validates_filenames() {
        unsafe {
            assert!(llama_common_fs_validate_filename_rust(
                b"model.gguf".as_ptr(),
                10,
                false
            ));
            assert!(llama_common_fs_validate_filename_rust(
                "unicode-é.gguf".as_ptr(),
                14,
                false
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                b"".as_ptr(),
                0,
                false
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                b" leading.gguf".as_ptr(),
                13,
                false
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                b"trailing.gguf ".as_ptr(),
                14,
                false
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                b"trailing.".as_ptr(),
                9,
                false
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                b"bad:name.gguf".as_ptr(),
                13,
                false
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                b"bad..name.gguf".as_ptr(),
                14,
                false
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                b".".as_ptr(),
                1,
                false
            ));
        }
    }

    #[test]
    fn validates_filename_subdir_and_utf8_rules() {
        unsafe {
            assert!(!llama_common_fs_validate_filename_rust(
                b"dir/model.gguf".as_ptr(),
                14,
                false
            ));
            assert!(llama_common_fs_validate_filename_rust(
                b"dir/model.gguf".as_ptr(),
                14,
                true
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                b"bad\\name.gguf".as_ptr(),
                13,
                false
            ));
            assert!(llama_common_fs_validate_filename_rust(
                b"bad\\name.gguf".as_ptr(),
                13,
                true
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                b"bad\nname.gguf".as_ptr(),
                13,
                false
            ));
            let overlong_nul = [0xC0, 0x80];
            assert!(!llama_common_fs_validate_filename_rust(
                overlong_nul.as_ptr(),
                2,
                false
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                "bad\u{ff0e}name".as_ptr(),
                10,
                false
            ));
            assert!(!llama_common_fs_validate_filename_rust(
                std::ptr::null(),
                1,
                false
            ));
        }
    }

    #[test]
    fn replaces_all_substrings() {
        unsafe {
            let replaced = llama_common_string_replace_all_rust(
                b"a/b/c".as_ptr(),
                5,
                b"/".as_ptr(),
                1,
                b"--".as_ptr(),
                2,
            );
            assert_eq!(
                std::slice::from_raw_parts(replaced.data, replaced.len),
                b"a--b--c"
            );
            llama_common_unicode_string_free(replaced);

            let unchanged = llama_common_string_replace_all_rust(
                b"abc".as_ptr(),
                3,
                b"".as_ptr(),
                0,
                b"x".as_ptr(),
                1,
            );
            assert_eq!(
                std::slice::from_raw_parts(unchanged.data, unchanged.len),
                b"abc"
            );
            llama_common_unicode_string_free(unchanged);

            let null = llama_common_string_replace_all_rust(
                std::ptr::null(),
                1,
                b"a".as_ptr(),
                1,
                b"b".as_ptr(),
                1,
            );
            assert!(null.data.is_null());
        }
    }

    #[test]
    fn escapes_regex_special_characters() {
        unsafe {
            let escaped = llama_common_regex_escape_rust(b"a.b[c]\\d+".as_ptr(), 9);
            assert_eq!(
                std::slice::from_raw_parts(escaped.data, escaped.len),
                br"a\.b\[c\]\\d\+"
            );
            llama_common_unicode_string_free(escaped);

            let plain = llama_common_regex_escape_rust(b"abc".as_ptr(), 3);
            assert_eq!(std::slice::from_raw_parts(plain.data, plain.len), b"abc");
            llama_common_unicode_string_free(plain);
        }
    }

    #[test]
    fn joins_string_views() {
        unsafe {
            let values = [
                StringView {
                    data: b"one".as_ptr(),
                    len: 3,
                },
                StringView {
                    data: b"two".as_ptr(),
                    len: 3,
                },
                StringView {
                    data: b"three".as_ptr(),
                    len: 5,
                },
            ];
            let joined =
                llama_common_string_join_rust(values.as_ptr(), values.len(), b", ".as_ptr(), 2);
            assert_eq!(
                std::slice::from_raw_parts(joined.data, joined.len),
                b"one, two, three"
            );
            llama_common_unicode_string_free(joined);

            let empty = llama_common_string_join_rust(values.as_ptr(), 0, b", ".as_ptr(), 2);
            assert_eq!(empty.len, 0);
            llama_common_unicode_string_free(empty);

            let bad = [StringView {
                data: std::ptr::null(),
                len: 1,
            }];
            let joined = llama_common_string_join_rust(bad.as_ptr(), bad.len(), b", ".as_ptr(), 2);
            assert!(joined.data.is_null());
        }
    }

    #[test]
    fn splits_strings() {
        unsafe {
            let split = llama_common_string_split_rust(b"a/b/".as_ptr(), 4, b"/".as_ptr(), 1);
            assert_eq!(split.len, 3);
            let items = std::slice::from_raw_parts(split.data, split.len);
            assert_eq!(
                std::slice::from_raw_parts(items[0].data, items[0].len),
                b"a"
            );
            assert_eq!(
                std::slice::from_raw_parts(items[1].data, items[1].len),
                b"b"
            );
            assert_eq!(items[2].len, 0);
            llama_common_string_list_free(split);

            let no_delim = llama_common_string_split_rust(b"abc".as_ptr(), 3, b"/".as_ptr(), 1);
            assert_eq!(no_delim.len, 1);
            let items = std::slice::from_raw_parts(no_delim.data, no_delim.len);
            assert_eq!(
                std::slice::from_raw_parts(items[0].data, items[0].len),
                b"abc"
            );
            llama_common_string_list_free(no_delim);

            let empty_delim = llama_common_string_split_rust(b"abc".as_ptr(), 3, b"".as_ptr(), 0);
            assert_eq!(empty_delim.len, 1);
            llama_common_string_list_free(empty_delim);
        }
    }

    #[test]
    fn parses_csv_rows() {
        unsafe {
            let input = br#"value1,"value, with, commas","value with ""escaped"" quotes",value4"#;
            let parsed = llama_common_parse_csv_row_rust(input.as_ptr(), input.len());
            assert_eq!(parsed.len, 4);
            let items = std::slice::from_raw_parts(parsed.data, parsed.len);
            assert_eq!(
                std::slice::from_raw_parts(items[0].data, items[0].len),
                b"value1"
            );
            assert_eq!(
                std::slice::from_raw_parts(items[1].data, items[1].len),
                b"value, with, commas"
            );
            assert_eq!(
                std::slice::from_raw_parts(items[2].data, items[2].len),
                br#"value with "escaped" quotes"#
            );
            assert_eq!(
                std::slice::from_raw_parts(items[3].data, items[3].len),
                b"value4"
            );
            llama_common_string_list_free(parsed);

            let trailing = llama_common_parse_csv_row_rust(b"a,".as_ptr(), 2);
            assert_eq!(trailing.len, 2);
            let items = std::slice::from_raw_parts(trailing.data, trailing.len);
            assert_eq!(
                std::slice::from_raw_parts(items[0].data, items[0].len),
                b"a"
            );
            assert_eq!(items[1].len, 0);
            llama_common_string_list_free(trailing);
        }
    }

    #[test]
    fn trims_space_views() {
        unsafe {
            let input = b" \tvalue \n";

            let leading =
                llama_common_trim_leading_space_view_rust(input.as_ptr(), input.len(), -1);
            assert_eq!(
                std::slice::from_raw_parts(leading.data, leading.len),
                b"value \n"
            );

            let leading_limited =
                llama_common_trim_leading_space_view_rust(input.as_ptr(), input.len(), 1);
            assert_eq!(
                std::slice::from_raw_parts(leading_limited.data, leading_limited.len),
                b"\tvalue \n"
            );

            let trailing =
                llama_common_trim_trailing_space_view_rust(input.as_ptr(), input.len(), -1);
            assert_eq!(
                std::slice::from_raw_parts(trailing.data, trailing.len),
                b" \tvalue"
            );

            let trailing_limited =
                llama_common_trim_trailing_space_view_rust(input.as_ptr(), input.len(), 1);
            assert_eq!(
                std::slice::from_raw_parts(trailing_limited.data, trailing_limited.len),
                b" \tvalue "
            );

            let null = llama_common_trim_leading_space_view_rust(std::ptr::null(), 1, -1);
            assert!(null.data.is_null());
            assert_eq!(null.len, 0);
        }
    }

    #[test]
    fn processes_escape_sequences() {
        unsafe {
            let input = br#"a\nb\t\x41\\\'\""#;
            let processed = llama_common_string_process_escapes_rust(input.as_ptr(), input.len());
            assert_eq!(
                std::slice::from_raw_parts(processed.data, processed.len),
                &[b'a', b'\n', b'b', b'\t', b'A', b'\\', b'\'', b'"']
            );
            llama_common_unicode_string_free(processed);

            let invalid_input = br"\xZZ\q";
            let invalid = llama_common_string_process_escapes_rust(
                invalid_input.as_ptr(),
                invalid_input.len(),
            );
            assert_eq!(
                std::slice::from_raw_parts(invalid.data, invalid.len),
                br"\xZZ\q"
            );
            llama_common_unicode_string_free(invalid);
        }
    }

    #[test]
    fn cleans_file_names() {
        unsafe {
            let cleaned = llama_common_clean_file_name_rust(br"org\repo/model".as_ptr(), 14);
            assert_eq!(
                std::slice::from_raw_parts(cleaned.data, cleaned.len),
                b"org_repo_model"
            );
            llama_common_unicode_string_free(cleaned);

            let unchanged = llama_common_clean_file_name_rust(b"model.gguf".as_ptr(), 10);
            assert_eq!(
                std::slice::from_raw_parts(unchanged.data, unchanged.len),
                b"model.gguf"
            );
            llama_common_unicode_string_free(unchanged);
        }
    }

    #[test]
    fn computes_string_diffs() {
        unsafe {
            let mut status = -99;
            let diff = llama_common_string_diff_rust(
                b"hello".as_ptr(),
                5,
                b"hello world".as_ptr(),
                11,
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(std::slice::from_raw_parts(diff.data, diff.len), b" world");
            llama_common_unicode_string_free(diff);

            let partial = llama_common_string_diff_rust(
                b"hello world".as_ptr(),
                11,
                b"hello".as_ptr(),
                5,
                &mut status,
            );
            assert_eq!(status, 0);
            assert_eq!(partial.len, 0);
            llama_common_unicode_string_free(partial);

            let invalid =
                llama_common_string_diff_rust(b"abc".as_ptr(), 3, b"xyz".as_ptr(), 3, &mut status);
            assert_eq!(status, -1);
            assert!(invalid.data.is_null());
        }
    }

    #[test]
    fn checks_string_prefixes_and_suffixes() {
        unsafe {
            assert!(llama_common_string_starts_with_rust(
                b"abcdef".as_ptr(),
                6,
                b"abc".as_ptr(),
                3
            ));
            assert!(!llama_common_string_starts_with_rust(
                b"abcdef".as_ptr(),
                6,
                b"abd".as_ptr(),
                3
            ));
            assert!(llama_common_string_starts_with_rust(
                b"abcdef".as_ptr(),
                6,
                b"".as_ptr(),
                0
            ));
            assert!(!llama_common_string_starts_with_rust(
                std::ptr::null(),
                1,
                b"a".as_ptr(),
                1
            ));

            assert!(llama_common_string_ends_with_rust(
                b"abcdef".as_ptr(),
                6,
                b"def".as_ptr(),
                3
            ));
            assert!(!llama_common_string_ends_with_rust(
                b"abcdef".as_ptr(),
                6,
                b"cef".as_ptr(),
                3
            ));
            assert!(llama_common_string_ends_with_rust(
                b"abcdef".as_ptr(),
                6,
                b"".as_ptr(),
                0
            ));
            assert!(!llama_common_string_ends_with_rust(
                b"abc".as_ptr(),
                3,
                std::ptr::null(),
                1
            ));
            assert!(llama_common_bytes_equal_rust(
                b"same".as_ptr(),
                4,
                b"same".as_ptr(),
                4
            ));
            assert!(!llama_common_bytes_equal_rust(
                b"same".as_ptr(),
                4,
                b"diff".as_ptr(),
                4
            ));
            assert!(llama_common_bytes_equal_rust(
                b"".as_ptr(),
                0,
                std::ptr::null(),
                0
            ));
            assert!(!llama_common_bytes_equal_rust(
                std::ptr::null(),
                1,
                b"a".as_ptr(),
                1
            ));

            assert_eq!(
                llama_common_string_find_partial_stop_rust(
                    b"hello wor".as_ptr(),
                    9,
                    b"world".as_ptr(),
                    5,
                ),
                6
            );
            assert_eq!(
                llama_common_string_find_partial_stop_rust(
                    b"hello world".as_ptr(),
                    11,
                    b"world".as_ptr(),
                    5,
                ),
                6
            );
            assert_eq!(
                llama_common_string_find_partial_stop_rust(
                    b"hello".as_ptr(),
                    5,
                    b"world".as_ptr(),
                    5,
                ),
                usize::MAX
            );
            assert_eq!(
                llama_common_string_find_partial_stop_rust(
                    std::ptr::null(),
                    1,
                    b"world".as_ptr(),
                    5,
                ),
                usize::MAX
            );

            assert_eq!(
                llama_common_string_remove_suffix_len_rust(
                    b"abcdef".as_ptr(),
                    6,
                    b"def".as_ptr(),
                    3,
                ),
                3
            );
            assert_eq!(
                llama_common_string_remove_suffix_len_rust(
                    b"abcdef".as_ptr(),
                    6,
                    b"xyz".as_ptr(),
                    3,
                ),
                usize::MAX
            );
            assert_eq!(
                llama_common_string_remove_suffix_len_rust(b"abcdef".as_ptr(), 6, b"".as_ptr(), 0,),
                6
            );
            assert_eq!(
                llama_common_string_remove_suffix_len_rust(std::ptr::null(), 1, b"def".as_ptr(), 3,),
                usize::MAX
            );
        }
    }
}
