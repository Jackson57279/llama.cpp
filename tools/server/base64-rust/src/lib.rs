use std::collections::HashSet;
use std::slice;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[repr(C)]
pub struct StringView {
    data: *const u8,
    len: usize,
}

#[repr(C)]
pub struct SseEventView {
    event: StringView,
    data: StringView,
}

#[repr(C)]
pub struct AnthropicSseEventView {
    event: StringView,
    data: StringView,
    has_event: u8,
}

#[repr(C)]
pub struct ResultTimingsView {
    cache_n: i32,
    prompt_n: i32,
    prompt_ms: f64,
    prompt_per_token_ms: f64,
    prompt_per_second: f64,
    predicted_n: i32,
    predicted_ms: f64,
    predicted_per_token_ms: f64,
    predicted_per_second: f64,
    draft_n: i32,
    draft_n_accepted: i32,
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BOUNDARY_CHARS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
static BOUNDARY_COUNTER: AtomicU64 = AtomicU64::new(0);

const SERVER_TASK_TYPE_COMPLETION: i32 = 0;
const SERVER_TASK_TYPE_EMBEDDING: i32 = 1;
const SERVER_TASK_TYPE_RERANK: i32 = 2;
const SERVER_TASK_TYPE_INFILL: i32 = 3;
const SERVER_TASK_TYPE_CANCEL: i32 = 4;

const TASK_RESPONSE_TYPE_NONE: i32 = 0;
const TASK_RESPONSE_TYPE_OAI_CHAT: i32 = 1;
const TASK_RESPONSE_TYPE_OAI_CMPL: i32 = 2;
const TASK_RESPONSE_TYPE_OAI_RESP: i32 = 3;
const TASK_RESPONSE_TYPE_OAI_EMBD: i32 = 5;
const TASK_RESPONSE_TYPE_ANTHROPIC: i32 = 6;

const SERVER_MODEL_STATUS_UNLOADED: i32 = 0;
const SERVER_MODEL_STATUS_LOADING: i32 = 1;
const SERVER_MODEL_STATUS_LOADED: i32 = 2;
const SERVER_MODEL_STATUS_SLEEPING: i32 = 3;

const SLOT_STATE_IDLE: i32 = 0;

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
pub unsafe extern "C" fn llama_server_base64_decode(
    data: *const u8,
    len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let Some(input) = ffi_bytes(data, len) else {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null_mut();
    };

    let mut output = decode(input).into_boxed_slice();
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

pub fn decode(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() / 4 * 3);
    let mut char_array_4 = [0u8; 4];
    let mut i = 0usize;
    let mut pos = 0usize;

    while pos < input.len() && input[pos] != b'=' && is_base64_byte(input[pos]) {
        char_array_4[i] = input[pos];
        i += 1;
        pos += 1;

        if i == 4 {
            for value in &mut char_array_4 {
                *value = decode_base64_index(*value);
            }
            output.extend_from_slice(&decode_base64_quad(char_array_4));
            i = 0;
        }
    }

    if i != 0 {
        for item in char_array_4.iter_mut().skip(i) {
            *item = 0;
        }
        for value in &mut char_array_4 {
            *value = decode_base64_index(*value);
        }

        let bytes = decode_base64_quad(char_array_4);
        output.extend_from_slice(&bytes[..i - 1]);
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
pub unsafe extern "C" fn llama_server_model_status_from_string_rust(
    data: *const u8,
    len: usize,
) -> i32 {
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
pub extern "C" fn llama_server_model_status_is_ready_rust(status: i32) -> bool {
    model_status_is_ready(status)
}

#[no_mangle]
pub extern "C" fn llama_server_model_status_is_running_rust(status: i32) -> bool {
    model_status_is_running(status)
}

#[no_mangle]
pub extern "C" fn llama_server_model_status_is_failed_rust(status: i32, exit_code: i32) -> bool {
    model_status_is_failed(status, exit_code)
}

#[no_mangle]
pub extern "C" fn llama_server_model_status_is_unloaded_rust(status: i32) -> bool {
    model_status_is_unloaded(status)
}

#[no_mangle]
pub extern "C" fn llama_server_model_status_is_loading_rust(status: i32) -> bool {
    model_status_is_loading(status)
}

#[no_mangle]
pub extern "C" fn llama_server_model_status_is_sleeping_rust(status: i32) -> bool {
    model_status_is_sleeping(status)
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
pub unsafe extern "C" fn llama_server_should_strip_proxy_header_rust(
    data: *const u8,
    len: usize,
) -> bool {
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
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
pub unsafe extern "C" fn llama_server_sanitize_multipart_field_rust(
    data: *const u8,
    len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let Some(input) = ffi_bytes(data, len) else {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null_mut();
    };

    let mut output = sanitize_multipart_field(input).into_boxed_slice();
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

#[no_mangle]
pub extern "C" fn llama_server_is_base64_char_rust(value: u8) -> bool {
    is_base64_byte(value)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_normalize_anthropic_billing_header_rust(
    data: *mut u8,
    len: usize,
) -> i32 {
    let Some(system_text) = ffi_bytes_mut(data, len) else {
        return 0;
    };
    normalize_anthropic_billing_header(system_text)
}

#[no_mangle]
pub extern "C" fn llama_server_probability_logarithm_rust(value: f32) -> f32 {
    if value == 0.0 {
        f32::MIN
    } else {
        value.ln()
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_validate_utf8_prefix_len_rust(
    data: *const u8,
    len: usize,
) -> usize {
    let Some(text) = ffi_bytes(data, len) else {
        return 0;
    };
    validate_utf8_prefix_len(text)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_is_valid_utf8_rust(data: *const u8, len: usize) -> bool {
    let Some(text) = ffi_bytes(data, len) else {
        return false;
    };
    is_valid_utf8(text)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_is_autoload_rust(
    default_value: bool,
    data: *const u8,
    len: usize,
) -> bool {
    let Some(value) = ffi_bytes(data, len) else {
        return default_value;
    };
    if value.is_empty() {
        default_value
    } else {
        value == b"true" || value == b"1"
    }
}

#[no_mangle]
pub extern "C" fn llama_server_stop_type_to_string_rust(stop_type: i32) -> *const std::ffi::c_char {
    match stop_type {
        1 => c"eos".as_ptr(),
        2 => c"word".as_ptr(),
        3 => c"limit".as_ptr(),
        _ => c"none".as_ptr(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_copy_string_bytes_rust(
    data: *const u8,
    len: usize,
    out: *mut u8,
    out_len: usize,
) -> usize {
    let Some(input) = ffi_bytes(data, len) else {
        return 0;
    };
    let Some(output) = ffi_bytes_mut(out, out_len) else {
        return 0;
    };
    let copy_len = input.len().min(output.len());
    output[..copy_len].copy_from_slice(&input[..copy_len]);
    copy_len
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_lora_all_alora_rust(
    scales: *const f32,
    is_alora: *const u8,
    len: usize,
) -> bool {
    let Some(scales) = ffi_slice(scales, len) else {
        return false;
    };
    let Some(is_alora) = ffi_slice(is_alora, len) else {
        return false;
    };
    lora_all_alora(scales, is_alora)
}

#[no_mangle]
pub extern "C" fn llama_server_lora_should_clear_cache_rust(
    current_has_enabled: bool,
    current_all_alora: bool,
    next_all_alora: bool,
) -> bool {
    lora_should_clear_cache(current_has_enabled, current_all_alora, next_all_alora)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_lora_are_equal_rust(
    left_scales: *const f32,
    left_ptrs: *const usize,
    left_len: usize,
    right_scales: *const f32,
    right_ptrs: *const usize,
    right_len: usize,
) -> bool {
    let Some(left_scales) = ffi_slice(left_scales, left_len) else {
        return false;
    };
    let Some(left_ptrs) = ffi_slice(left_ptrs, left_len) else {
        return false;
    };
    let Some(right_scales) = ffi_slice(right_scales, right_len) else {
        return false;
    };
    let Some(right_ptrs) = ffi_slice(right_ptrs, right_len) else {
        return false;
    };
    lora_are_equal(left_scales, left_ptrs, right_scales, right_ptrs)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_lora_enabled_ids_rust(
    scales: *const f32,
    len: usize,
    out: *mut usize,
    out_len: usize,
) -> usize {
    let Some(scales) = ffi_slice(scales, len) else {
        return 0;
    };
    let Some(out) = ffi_slice_mut(out, out_len) else {
        return 0;
    };
    lora_enabled_ids(scales, out)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_format_token_output_rust(
    data: *const u8,
    len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let Some(input) = ffi_bytes(data, len) else {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null_mut();
    };

    let mut output = format_token_output(input).into_boxed_slice();
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
pub unsafe extern "C" fn llama_server_format_oai_sse_rust(
    items: *const StringView,
    len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let Some(items) = ffi_slice(items, len) else {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null_mut();
    };

    let mut output = format_oai_sse(items).into_boxed_slice();
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
pub unsafe extern "C" fn llama_server_format_oai_resp_sse_rust(
    items: *const SseEventView,
    len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let Some(items) = ffi_slice(items, len) else {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null_mut();
    };

    let mut output = format_oai_resp_sse(items).into_boxed_slice();
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
pub unsafe extern "C" fn llama_server_format_anthropic_sse_rust(
    items: *const AnthropicSseEventView,
    len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let Some(items) = ffi_slice(items, len) else {
        if !out_len.is_null() {
            unsafe {
                *out_len = 0;
            }
        }
        return std::ptr::null_mut();
    };

    let mut output = format_anthropic_sse(items).into_boxed_slice();
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
pub unsafe extern "C" fn llama_server_token_common_prefix_rust(
    left: *const i32,
    left_len: usize,
    right: *const i32,
    right_len: usize,
) -> usize {
    let Some(left) = ffi_slice(left, left_len) else {
        return 0;
    };
    let Some(right) = ffi_slice(right, right_len) else {
        return 0;
    };
    token_common_prefix(left, right)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_filter_text_tokens_rust(
    input: *const i32,
    input_len: usize,
    out: *mut i32,
    out_len: usize,
) -> usize {
    let Some(input) = ffi_slice(input, input_len) else {
        return 0;
    };
    let Some(out) = ffi_slice_mut(out, out_len) else {
        return 0;
    };
    filter_text_tokens(input, out)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_tokens_debug_string_rust(
    tokens: *const i32,
    tokens_len: usize,
    media_indices: *const usize,
    media_indices_len: usize,
    out_len: *mut usize,
) -> *mut u8 {
    let Some(tokens) = ffi_slice(tokens, tokens_len) else {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return std::ptr::null_mut();
    };
    let Some(media_indices) = ffi_slice(media_indices, media_indices_len) else {
        if !out_len.is_null() {
            *out_len = 0;
        }
        return std::ptr::null_mut();
    };

    let mut output = tokens_debug_string(tokens, media_indices).into_boxed_slice();
    let ptr = output.as_mut_ptr();
    let len = output.len();
    std::mem::forget(output);

    if !out_len.is_null() {
        *out_len = len;
    }
    ptr
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_tokens_valid_range_rust(
    tokens: *const i32,
    tokens_len: usize,
    n_vocab: i32,
) -> bool {
    let Some(tokens) = ffi_slice(tokens, tokens_len) else {
        return false;
    };
    tokens_valid_range(tokens, n_vocab)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_tokens_pos_next_rust(
    token_len: usize,
    n_tokens: i64,
    media_indices: *const usize,
    media_n_pos: *const i32,
    media_n_tokens: *const usize,
    media_len: usize,
) -> i32 {
    let Some(media_indices) = ffi_slice(media_indices, media_len) else {
        return 0;
    };
    let Some(media_n_pos) = ffi_slice(media_n_pos, media_len) else {
        return 0;
    };
    let Some(media_n_tokens) = ffi_slice(media_n_tokens, media_len) else {
        return 0;
    };
    tokens_pos_next(
        token_len,
        n_tokens,
        media_indices,
        media_n_pos,
        media_n_tokens,
    )
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_tokens_size_up_to_pos_rust(
    token_len: usize,
    max_pos: i32,
    media_indices: *const usize,
    media_n_pos: *const i32,
    media_n_tokens: *const usize,
    media_len: usize,
) -> usize {
    let Some(media_indices) = ffi_slice(media_indices, media_len) else {
        return 0;
    };
    let Some(media_n_pos) = ffi_slice(media_n_pos, media_len) else {
        return 0;
    };
    let Some(media_n_tokens) = ffi_slice(media_n_tokens, media_len) else {
        return 0;
    };
    tokens_size_up_to_pos(
        token_len,
        max_pos,
        media_indices,
        media_n_pos,
        media_n_tokens,
    )
}

#[no_mangle]
pub extern "C" fn llama_server_task_need_embd_rust(task_type: i32) -> bool {
    task_need_embd(task_type)
}

#[no_mangle]
pub extern "C" fn llama_server_task_need_logits_rust(task_type: i32) -> bool {
    task_need_logits(task_type)
}

#[no_mangle]
pub extern "C" fn llama_server_task_need_sampling_rust(task_type: i32) -> bool {
    task_need_sampling(task_type)
}

#[no_mangle]
pub extern "C" fn llama_server_task_is_cancel_rust(task_type: i32) -> bool {
    task_is_cancel(task_type)
}

#[no_mangle]
pub extern "C" fn llama_server_task_is_completion_rust(task_type: i32) -> bool {
    task_is_completion(task_type)
}

#[no_mangle]
pub extern "C" fn llama_server_task_is_embedding_rust(task_type: i32) -> bool {
    task_is_embedding(task_type)
}

#[no_mangle]
pub extern "C" fn llama_server_task_is_rerank_rust(task_type: i32) -> bool {
    task_is_rerank(task_type)
}

#[no_mangle]
pub extern "C" fn llama_server_task_is_completion_or_infill_rust(task_type: i32) -> bool {
    task_is_completion_or_infill(task_type)
}

#[no_mangle]
pub extern "C" fn llama_server_response_type_is_oai_embd_rust(response_type: i32) -> bool {
    response_type_is_oai_embd(response_type)
}

#[no_mangle]
pub extern "C" fn llama_server_response_type_is_non_native_rust(response_type: i32) -> bool {
    response_type_is_non_native(response_type)
}

#[no_mangle]
pub extern "C" fn llama_server_response_type_is_oai_chat_or_cmpl_rust(response_type: i32) -> bool {
    response_type_is_oai_chat_or_cmpl(response_type)
}

#[no_mangle]
pub extern "C" fn llama_server_response_type_is_anthropic_rust(response_type: i32) -> bool {
    response_type_is_anthropic(response_type)
}

#[no_mangle]
pub extern "C" fn llama_server_response_type_is_oai_resp_rust(response_type: i32) -> bool {
    response_type_is_oai_resp(response_type)
}

#[no_mangle]
pub extern "C" fn llama_server_response_type_stream_done_is_empty_rust(response_type: i32) -> bool {
    response_type_stream_done_is_empty(response_type)
}

#[no_mangle]
pub extern "C" fn llama_server_task_is_parent_rust(child_task_count: usize) -> bool {
    task_is_parent(child_task_count)
}

#[no_mangle]
pub extern "C" fn llama_server_task_is_child_rust(id_parent: i32) -> bool {
    task_is_child(id_parent)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_unique_task_ids_rust(
    ids: *const i32,
    ids_len: usize,
    out: *mut i32,
    out_len: usize,
) -> usize {
    let Some(ids) = ffi_slice(ids, ids_len) else {
        return 0;
    };
    let Some(out) = ffi_slice_mut(out, out_len) else {
        return 0;
    };
    unique_task_ids(ids, out)
}

#[no_mangle]
pub extern "C" fn llama_server_slot_is_processing_rust(slot_state: i32) -> bool {
    slot_is_processing(slot_state)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_slot_has_budget_rust(
    task_n_predict: i32,
    global_n_predict: i32,
    n_decoded: i32,
    out_n_remaining: *mut i32,
) -> bool {
    let Some(out_n_remaining) = out_n_remaining.as_mut() else {
        return false;
    };
    let (has_budget, n_remaining) = slot_has_budget(task_n_predict, global_n_predict, n_decoded);
    *out_n_remaining = n_remaining;
    has_budget
}

#[no_mangle]
pub extern "C" fn llama_server_slot_can_split_rust(
    need_embd: bool,
    has_memory: bool,
    pooling_is_last: bool,
) -> bool {
    slot_can_split(need_embd, has_memory, pooling_is_last)
}

#[no_mangle]
pub extern "C" fn llama_server_slot_n_draft_max_rust(
    has_spec: bool,
    n_draft_min: i32,
    n_draft_max: i32,
    n_ctx: i32,
    prompt_tokens: i32,
    n_remaining: i32,
) -> i32 {
    slot_n_draft_max(
        has_spec,
        n_draft_min,
        n_draft_max,
        n_ctx,
        prompt_tokens,
        n_remaining,
    )
}

#[no_mangle]
pub extern "C" fn llama_server_slot_can_batch_with_rust(
    task_type: i32,
    other_task_type: i32,
    lora_equal: bool,
) -> bool {
    slot_can_batch_with(task_type, other_task_type, lora_equal)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_prompt_size_rust(
    data_len: usize,
    checkpoint_sizes: *const usize,
    checkpoint_count: usize,
) -> usize {
    let Some(checkpoint_sizes) = ffi_slice(checkpoint_sizes, checkpoint_count) else {
        return data_len;
    };
    prompt_size(data_len, checkpoint_sizes)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_sum_usize_rust(values: *const usize, len: usize) -> usize {
    let Some(values) = ffi_slice(values, len) else {
        return 0;
    };
    sum_usize(values)
}

#[no_mangle]
pub extern "C" fn llama_server_tokens_len_rust(token_count: usize) -> usize {
    tokens_len(token_count)
}

#[no_mangle]
pub extern "C" fn llama_server_tokens_is_empty_rust(token_count: usize) -> bool {
    tokens_is_empty(token_count)
}

#[no_mangle]
pub extern "C" fn llama_server_tokens_len_i32_rust(token_count: usize) -> i32 {
    tokens_len_i32(token_count)
}

#[no_mangle]
pub extern "C" fn llama_server_json_field_is_array_rust(has_key: bool, is_array: bool) -> bool {
    json_field_matches_type(has_key, is_array)
}

#[no_mangle]
pub extern "C" fn llama_server_json_field_is_string_rust(has_key: bool, is_string: bool) -> bool {
    json_field_matches_type(has_key, is_string)
}

#[no_mangle]
pub extern "C" fn llama_server_queue_should_process_rust(running: bool, queue_empty: bool) -> bool {
    queue_should_process(running, queue_empty)
}

#[no_mangle]
pub extern "C" fn llama_server_queue_sleep_wait_done_rust(
    running: bool,
    req_stop_sleeping: bool,
) -> bool {
    queue_sleep_wait_done(running, req_stop_sleeping)
}

#[no_mangle]
pub extern "C" fn llama_server_queue_task_wait_done_rust(running: bool, queue_empty: bool) -> bool {
    queue_task_wait_done(running, queue_empty)
}

#[no_mangle]
pub extern "C" fn llama_server_queue_should_sleep_rust(
    idle_sleep_ms: i64,
    now_ms: i64,
    time_last_task_ms: i64,
) -> bool {
    queue_should_sleep(idle_sleep_ms, now_ms, time_last_task_ms)
}

#[no_mangle]
pub extern "C" fn llama_server_response_reader_has_next_rust(
    cancelled: bool,
    received_count: usize,
    task_count: usize,
) -> bool {
    response_reader_has_next(cancelled, received_count, task_count)
}

#[no_mangle]
pub extern "C" fn llama_server_slot_timings_rust(
    cache_n: i32,
    prompt_n: i32,
    prompt_ms: f64,
    predicted_n: i32,
    predicted_ms: f64,
    draft_n: i32,
    draft_n_accepted: i32,
) -> ResultTimingsView {
    slot_timings(
        cache_n,
        prompt_n,
        prompt_ms,
        predicted_n,
        predicted_ms,
        draft_n,
        draft_n_accepted,
    )
}

#[no_mangle]
pub extern "C" fn llama_server_score_desc_rust(left: f64, right: f64) -> bool {
    score_desc(left, right)
}

#[no_mangle]
pub extern "C" fn llama_server_logit_desc_rust(left: f32, right: f32) -> bool {
    logit_desc(left, right)
}

#[no_mangle]
pub extern "C" fn llama_server_line_start_desc_rust(left: i32, right: i32) -> bool {
    line_start_desc(left, right)
}

#[no_mangle]
pub extern "C" fn llama_server_id_matches_rust(left: i32, right: i32) -> bool {
    id_matches(left, right)
}

#[no_mangle]
pub extern "C" fn llama_server_http_res_is_stream_rust(has_next: bool) -> bool {
    http_res_is_stream(has_next)
}

#[no_mangle]
pub extern "C" fn llama_server_checkpoint_before_threshold_rust(
    pos_min: i32,
    pos_min_threshold: i32,
) -> bool {
    checkpoint_before_threshold(pos_min, pos_min_threshold)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_json_flags_all_numbers_rust(
    flags: *const u8,
    len: usize,
) -> bool {
    let Some(flags) = ffi_slice(flags, len) else {
        return false;
    };
    json_flags_all_numbers(flags)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_json_flags_mixed_numbers_strings_rust(
    flags: *const u8,
    len: usize,
) -> bool {
    let Some(flags) = ffi_slice(flags, len) else {
        return false;
    };
    json_flags_mixed_numbers_strings(flags)
}

#[no_mangle]
pub unsafe extern "C" fn llama_server_json_flags_contains_number_rust(
    flags: *const u8,
    len: usize,
) -> bool {
    let Some(flags) = ffi_slice(flags, len) else {
        return false;
    };
    json_flags_contains_number(flags)
}

fn ffi_bytes<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if data.is_null() {
        return None;
    }
    Some(unsafe { slice::from_raw_parts(data, len) })
}

fn ffi_bytes_mut<'a>(data: *mut u8, len: usize) -> Option<&'a mut [u8]> {
    if len == 0 {
        return Some(unsafe {
            slice::from_raw_parts_mut(std::ptr::NonNull::<u8>::dangling().as_ptr(), 0)
        });
    }
    if data.is_null() {
        return None;
    }
    Some(unsafe { slice::from_raw_parts_mut(data, len) })
}

fn ffi_slice<'a, T>(data: *const T, len: usize) -> Option<&'a [T]> {
    if len == 0 {
        return Some(&[]);
    }
    if data.is_null() {
        return None;
    }
    Some(unsafe { slice::from_raw_parts(data, len) })
}

fn ffi_slice_mut<'a, T>(data: *mut T, len: usize) -> Option<&'a mut [T]> {
    if len == 0 {
        return Some(unsafe {
            slice::from_raw_parts_mut(std::ptr::NonNull::<T>::dangling().as_ptr(), 0)
        });
    }
    if data.is_null() {
        return None;
    }
    Some(unsafe { slice::from_raw_parts_mut(data, len) })
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

fn is_base64_byte(value: u8) -> bool {
    value.is_ascii_alphanumeric() || value == b'+' || value == b'/'
}

fn decode_base64_index(value: u8) -> u8 {
    ALPHABET
        .iter()
        .position(|candidate| *candidate == value)
        .unwrap_or(usize::MAX) as u8
}

fn decode_base64_quad(values: [u8; 4]) -> [u8; 3] {
    [
        (values[0] << 2).wrapping_add((values[1] & 0x30) >> 4),
        ((values[1] & 0x0f) << 4).wrapping_add((values[2] & 0x3c) >> 2),
        ((values[2] & 0x03) << 6).wrapping_add(values[3]),
    ]
}

fn model_status_is_ready(status: i32) -> bool {
    status == SERVER_MODEL_STATUS_LOADED
}

fn model_status_is_running(status: i32) -> bool {
    matches!(
        status,
        SERVER_MODEL_STATUS_LOADED | SERVER_MODEL_STATUS_LOADING | SERVER_MODEL_STATUS_SLEEPING
    )
}

fn model_status_is_failed(status: i32, exit_code: i32) -> bool {
    status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0
}

fn model_status_is_unloaded(status: i32) -> bool {
    status == SERVER_MODEL_STATUS_UNLOADED
}

fn model_status_is_loading(status: i32) -> bool {
    status == SERVER_MODEL_STATUS_LOADING
}

fn model_status_is_sleeping(status: i32) -> bool {
    status == SERVER_MODEL_STATUS_SLEEPING
}

fn sanitize_multipart_field(input: &[u8]) -> Vec<u8> {
    input
        .iter()
        .copied()
        .filter(|c| *c != b'\n' && *c != b'\r' && *c != b'"')
        .collect()
}

fn normalize_anthropic_billing_header(system_text: &mut [u8]) -> i32 {
    const PREFIX: &[u8] = b"x-anthropic-billing-header:";
    const CCH: &[u8] = b"cch=";
    const CCH_LENGTH: usize = 5;

    if !system_text.starts_with(PREFIX) {
        return 0;
    }

    let search = &system_text[PREFIX.len()..];
    let Some(index_cch) = find_subslice(search, CCH).map(|index| index + PREFIX.len()) else {
        return 0;
    };

    let index_replace = index_cch + CCH.len();
    if index_replace + CCH_LENGTH < system_text.len()
        && system_text[index_replace + CCH_LENGTH] == b';'
    {
        system_text[index_replace..index_replace + CCH_LENGTH].fill(b'f');
        1
    } else {
        -1
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn validate_utf8_prefix_len(text: &[u8]) -> usize {
    let len = text.len();
    if len == 0 {
        return 0;
    }

    for i in 1..=4.min(len) {
        let c = text[len - i];
        if (c & 0xe0) == 0xc0 {
            if i < 2 {
                return len - i;
            }
        } else if (c & 0xf0) == 0xe0 {
            if i < 3 {
                return len - i;
            }
        } else if (c & 0xf8) == 0xf0 && i < 4 {
            return len - i;
        }
    }
    len
}

fn is_valid_utf8(text: &[u8]) -> bool {
    let mut pos = 0usize;
    while pos < text.len() {
        let byte = text[pos];
        if byte <= 0x7f {
            pos += 1;
        } else if (byte & 0xe0) == 0xc0 {
            if pos + 1 >= text.len() || (text[pos + 1] & 0xc0) != 0x80 {
                return false;
            }
            pos += 2;
        } else if (byte & 0xf0) == 0xe0 {
            if pos + 2 >= text.len()
                || (text[pos + 1] & 0xc0) != 0x80
                || (text[pos + 2] & 0xc0) != 0x80
            {
                return false;
            }
            pos += 3;
        } else if (byte & 0xf8) == 0xf0 {
            if pos + 3 >= text.len()
                || (text[pos + 1] & 0xc0) != 0x80
                || (text[pos + 2] & 0xc0) != 0x80
                || (text[pos + 3] & 0xc0) != 0x80
            {
                return false;
            }
            pos += 4;
        } else {
            return false;
        }
    }
    true
}

fn lora_all_alora(scales: &[f32], is_alora: &[u8]) -> bool {
    let mut found_alora = false;
    for (&scale, &is_alora) in scales.iter().zip(is_alora.iter()) {
        if scale != 0.0 {
            if is_alora == 0 {
                return false;
            }
            found_alora = true;
        }
    }
    found_alora
}

fn lora_should_clear_cache(
    current_has_enabled: bool,
    current_all_alora: bool,
    next_all_alora: bool,
) -> bool {
    (current_has_enabled && !current_all_alora) || !next_all_alora
}

fn lora_are_equal(
    left_scales: &[f32],
    left_ptrs: &[usize],
    right_scales: &[f32],
    right_ptrs: &[usize],
) -> bool {
    left_scales.len() == right_scales.len()
        && left_ptrs.len() == right_ptrs.len()
        && left_scales
            .iter()
            .zip(right_scales)
            .all(|(left, right)| left == right)
        && left_ptrs
            .iter()
            .zip(right_ptrs)
            .all(|(left, right)| left == right)
}

fn lora_enabled_ids(scales: &[f32], out: &mut [usize]) -> usize {
    let mut count = 0usize;
    for (idx, scale) in scales.iter().enumerate() {
        if *scale > 0.0 {
            if count < out.len() {
                out[count] = idx;
            }
            count += 1;
        }
    }
    count
}

fn format_token_output(input: &[u8]) -> Vec<u8> {
    if input.len() == 1 && (input[0] & 0x80) == 0x80 {
        format!("byte: \\x{:x}", input[0]).into_bytes()
    } else {
        input.to_vec()
    }
}

fn format_oai_sse(items: &[StringView]) -> Vec<u8> {
    let mut out = Vec::new();
    for item in items {
        let Some(bytes) = ffi_bytes(item.data, item.len) else {
            continue;
        };
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(bytes);
        out.extend_from_slice(b"\n\n");
    }
    out
}

fn format_oai_resp_sse(items: &[SseEventView]) -> Vec<u8> {
    let mut out = Vec::new();
    for item in items {
        let Some(event) = ffi_bytes(item.event.data, item.event.len) else {
            continue;
        };
        let Some(data) = ffi_bytes(item.data.data, item.data.len) else {
            continue;
        };
        out.extend_from_slice(b"event: ");
        out.extend_from_slice(event);
        out.extend_from_slice(b"\ndata: ");
        out.extend_from_slice(data);
        out.extend_from_slice(b"\n\n");
    }
    out
}

fn format_anthropic_sse(items: &[AnthropicSseEventView]) -> Vec<u8> {
    let mut out = Vec::new();
    for item in items {
        let Some(data) = ffi_bytes(item.data.data, item.data.len) else {
            continue;
        };
        if item.has_event != 0 {
            let Some(event) = ffi_bytes(item.event.data, item.event.len) else {
                continue;
            };
            out.extend_from_slice(b"event: ");
            out.extend_from_slice(event);
            out.extend_from_slice(b"\n");
        }
        out.extend_from_slice(b"data: ");
        out.extend_from_slice(data);
        out.extend_from_slice(b"\n\n");
    }
    out
}

fn token_common_prefix(left: &[i32], right: &[i32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn filter_text_tokens(input: &[i32], out: &mut [i32]) -> usize {
    let mut count = 0usize;
    for token in input.iter().copied() {
        if token != -1 {
            if count < out.len() {
                out[count] = token;
            }
            count += 1;
        }
    }
    count
}

fn tokens_debug_string(tokens: &[i32], media_indices: &[usize]) -> Vec<u8> {
    let mut out = String::from("tokens: ");
    for (idx, token) in tokens.iter().enumerate() {
        out.push_str("idx:");
        out.push_str(&idx.to_string());
        out.push(' ');
        if *token == -1 {
            out.push_str("<embd> ");
        } else {
            out.push_str(&token.to_string());
            out.push(' ');
        }
    }
    out.push('\n');
    out.push_str("image idx: ");
    for idx in media_indices {
        out.push_str(&idx.to_string());
        out.push_str(", ");
    }
    out.into_bytes()
}

fn tokens_valid_range(tokens: &[i32], n_vocab: i32) -> bool {
    tokens
        .iter()
        .all(|token| *token == -1 || (*token >= 0 && *token < n_vocab))
}

fn tokens_pos_next(
    token_len: usize,
    n_tokens: i64,
    media_indices: &[usize],
    media_n_pos: &[i32],
    media_n_tokens: &[usize],
) -> i32 {
    if n_tokens < 0 {
        let media_delta: i64 = media_n_pos
            .iter()
            .copied()
            .zip(media_n_tokens.iter().copied())
            .map(|(n_pos, n_tok)| i64::from(n_pos) - n_tok as i64)
            .sum();
        return (token_len as i64 + media_delta) as i32;
    }

    let mut idx = 0usize;
    let mut media_idx = 0usize;
    let mut pos = 0i32;
    let target = n_tokens as usize;

    while idx < target {
        while media_idx < media_indices.len() && media_indices[media_idx] < idx {
            media_idx += 1;
        }
        if media_idx < media_indices.len() && media_indices[media_idx] == idx {
            pos += media_n_pos[media_idx];
            idx += media_n_tokens[media_idx];
        } else {
            pos += 1;
            idx += 1;
        }
    }

    pos
}

fn tokens_size_up_to_pos(
    token_len: usize,
    max_pos: i32,
    media_indices: &[usize],
    media_n_pos: &[i32],
    media_n_tokens: &[usize],
) -> usize {
    if media_indices.is_empty() {
        return if max_pos < 0 {
            token_len
        } else {
            (max_pos as usize).min(token_len)
        };
    }

    let mut idx = 0usize;
    let mut media_idx = 0usize;
    let mut pos = 0i32;

    while idx < token_len {
        while media_idx < media_indices.len() && media_indices[media_idx] < idx {
            media_idx += 1;
        }
        if media_idx < media_indices.len() && media_indices[media_idx] == idx {
            pos += media_n_pos[media_idx];
            idx += media_n_tokens[media_idx];
        } else {
            pos += 1;
            idx += 1;
        }

        if pos >= max_pos {
            break;
        }
    }

    idx
}

fn task_need_embd(task_type: i32) -> bool {
    matches!(
        task_type,
        SERVER_TASK_TYPE_EMBEDDING | SERVER_TASK_TYPE_RERANK
    )
}

fn task_need_logits(task_type: i32) -> bool {
    matches!(
        task_type,
        SERVER_TASK_TYPE_COMPLETION | SERVER_TASK_TYPE_INFILL
    )
}

fn task_need_sampling(task_type: i32) -> bool {
    matches!(
        task_type,
        SERVER_TASK_TYPE_COMPLETION | SERVER_TASK_TYPE_INFILL
    )
}

fn task_is_cancel(task_type: i32) -> bool {
    task_type == SERVER_TASK_TYPE_CANCEL
}

fn task_is_completion(task_type: i32) -> bool {
    task_type == SERVER_TASK_TYPE_COMPLETION
}

fn task_is_embedding(task_type: i32) -> bool {
    task_type == SERVER_TASK_TYPE_EMBEDDING
}

fn task_is_rerank(task_type: i32) -> bool {
    task_type == SERVER_TASK_TYPE_RERANK
}

fn task_is_completion_or_infill(task_type: i32) -> bool {
    matches!(
        task_type,
        SERVER_TASK_TYPE_COMPLETION | SERVER_TASK_TYPE_INFILL
    )
}

fn response_type_is_oai_embd(response_type: i32) -> bool {
    response_type == TASK_RESPONSE_TYPE_OAI_EMBD
}

fn response_type_is_non_native(response_type: i32) -> bool {
    response_type != TASK_RESPONSE_TYPE_NONE
}

fn response_type_is_oai_chat_or_cmpl(response_type: i32) -> bool {
    matches!(
        response_type,
        TASK_RESPONSE_TYPE_OAI_CHAT | TASK_RESPONSE_TYPE_OAI_CMPL
    )
}

fn response_type_is_anthropic(response_type: i32) -> bool {
    response_type == TASK_RESPONSE_TYPE_ANTHROPIC
}

fn response_type_is_oai_resp(response_type: i32) -> bool {
    response_type == TASK_RESPONSE_TYPE_OAI_RESP
}

fn response_type_stream_done_is_empty(response_type: i32) -> bool {
    matches!(
        response_type,
        TASK_RESPONSE_TYPE_NONE | TASK_RESPONSE_TYPE_OAI_RESP | TASK_RESPONSE_TYPE_ANTHROPIC
    )
}

fn task_is_parent(child_task_count: usize) -> bool {
    child_task_count > 0
}

fn task_is_child(id_parent: i32) -> bool {
    id_parent != -1
}

fn unique_task_ids(ids: &[i32], out: &mut [i32]) -> usize {
    let mut seen = HashSet::with_capacity(ids.len());
    let mut count = 0usize;
    for id in ids.iter().copied() {
        if seen.insert(id) {
            if count < out.len() {
                out[count] = id;
            }
            count += 1;
        }
    }
    count
}

fn slot_is_processing(slot_state: i32) -> bool {
    slot_state != SLOT_STATE_IDLE
}

fn slot_has_budget(task_n_predict: i32, global_n_predict: i32, n_decoded: i32) -> (bool, i32) {
    if task_n_predict == -1 && global_n_predict == -1 {
        return (true, -1);
    }

    let n_remaining = if task_n_predict != -1 {
        task_n_predict - n_decoded
    } else {
        global_n_predict - n_decoded
    };
    (n_remaining > 0, n_remaining)
}

fn slot_can_split(need_embd: bool, has_memory: bool, pooling_is_last: bool) -> bool {
    !need_embd || (has_memory && pooling_is_last)
}

fn slot_n_draft_max(
    has_spec: bool,
    n_draft_min: i32,
    n_draft_max: i32,
    n_ctx: i32,
    prompt_tokens: i32,
    n_remaining: i32,
) -> i32 {
    if !has_spec {
        return 0;
    }

    let mut n_draft_max = n_draft_max.min(n_ctx - prompt_tokens - 2);
    if n_remaining > 0 {
        n_draft_max = n_draft_max.min(n_remaining - 1);
    }
    if n_draft_max < n_draft_min {
        0
    } else {
        n_draft_max
    }
}

fn slot_can_batch_with(task_type: i32, other_task_type: i32, lora_equal: bool) -> bool {
    task_type == other_task_type && lora_equal
}

fn prompt_size(data_len: usize, checkpoint_sizes: &[usize]) -> usize {
    data_len.saturating_add(sum_usize(checkpoint_sizes))
}

fn sum_usize(values: &[usize]) -> usize {
    values.iter().copied().fold(0, usize::saturating_add)
}

fn tokens_len(token_count: usize) -> usize {
    token_count
}

fn tokens_is_empty(token_count: usize) -> bool {
    token_count == 0
}

fn tokens_len_i32(token_count: usize) -> i32 {
    token_count as i32
}

fn json_field_matches_type(has_key: bool, type_matches: bool) -> bool {
    has_key && type_matches
}

fn queue_should_process(running: bool, queue_empty: bool) -> bool {
    !running || !queue_empty
}

fn queue_sleep_wait_done(running: bool, req_stop_sleeping: bool) -> bool {
    !running || req_stop_sleeping
}

fn queue_task_wait_done(running: bool, queue_empty: bool) -> bool {
    !queue_empty || !running
}

fn queue_should_sleep(idle_sleep_ms: i64, now_ms: i64, time_last_task_ms: i64) -> bool {
    idle_sleep_ms >= 0 && now_ms.saturating_sub(time_last_task_ms) >= idle_sleep_ms
}

fn response_reader_has_next(cancelled: bool, received_count: usize, task_count: usize) -> bool {
    !cancelled && received_count < task_count
}

fn score_desc(left: f64, right: f64) -> bool {
    left > right
}

fn logit_desc(left: f32, right: f32) -> bool {
    left > right
}

fn line_start_desc(left: i32, right: i32) -> bool {
    left > right
}

fn id_matches(left: i32, right: i32) -> bool {
    left == right
}

fn http_res_is_stream(has_next: bool) -> bool {
    has_next
}

fn checkpoint_before_threshold(pos_min: i32, pos_min_threshold: i32) -> bool {
    pos_min < pos_min_threshold || pos_min == 0
}

fn slot_timings(
    cache_n: i32,
    prompt_n: i32,
    prompt_ms: f64,
    predicted_n: i32,
    predicted_ms: f64,
    draft_n: i32,
    draft_n_accepted: i32,
) -> ResultTimingsView {
    ResultTimingsView {
        cache_n,
        prompt_n,
        prompt_ms,
        prompt_per_token_ms: prompt_ms / f64::from(prompt_n),
        prompt_per_second: 1e3 / prompt_ms * f64::from(prompt_n),
        predicted_n,
        predicted_ms,
        predicted_per_token_ms: predicted_ms / f64::from(predicted_n),
        predicted_per_second: 1e3 / predicted_ms * f64::from(predicted_n),
        draft_n: if draft_n > 0 { draft_n } else { 0 },
        draft_n_accepted: if draft_n > 0 { draft_n_accepted } else { 0 },
    }
}

fn json_flags_all_numbers(flags: &[u8]) -> bool {
    flags.iter().all(|flag| *flag == 1)
}

fn json_flags_mixed_numbers_strings(flags: &[u8]) -> bool {
    let mut seen_number = false;
    let mut seen_string = false;
    for flag in flags {
        seen_number |= *flag == 1;
        seen_string |= *flag == 2;
        if seen_number && seen_string {
            return true;
        }
    }
    false
}

fn json_flags_contains_number(flags: &[u8]) -> bool {
    flags.iter().any(|flag| *flag == 1)
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
    fn decodes_standard_padding_cases() {
        assert_eq!(decode(b""), b"");
        assert_eq!(decode(b"Zg=="), b"f");
        assert_eq!(decode(b"Zm8="), b"fo");
        assert_eq!(decode(b"Zm9v"), b"foo");
        assert_eq!(decode(b"aGVsbG8gd29ybGQ="), b"hello world");
    }

    #[test]
    fn decoder_stops_at_first_padding_or_invalid_byte() {
        assert_eq!(decode(b"Zm9v===="), b"foo");
        assert_eq!(decode(b"Zm9v!ignored"), b"foo");
        assert_eq!(decode(b"Z"), b"");
        assert_eq!(decode(b"Zm"), b"f");
        assert_eq!(decode(b"Zm9"), b"fo");
    }

    #[test]
    fn ffi_decode_round_trip_memory() {
        unsafe {
            let mut out_len = 0usize;
            let ptr = llama_server_base64_decode(b"YWJj".as_ptr(), 4, &mut out_len);
            assert!(!ptr.is_null());
            let output = slice::from_raw_parts(ptr, out_len).to_vec();
            assert_eq!(output, b"abc");
            llama_server_base64_free(ptr, out_len);

            let ptr = llama_server_base64_decode(std::ptr::null(), 1, &mut out_len);
            assert!(ptr.is_null());
            assert_eq!(out_len, 0);
        }
    }

    #[test]
    fn copies_string_bytes_for_cpp_vector_wrapper() {
        unsafe {
            let mut output = vec![0u8; 3];
            let copied = llama_server_copy_string_bytes_rust(
                b"hello".as_ptr(),
                5,
                output.as_mut_ptr(),
                output.len(),
            );
            assert_eq!(copied, 3);
            assert_eq!(output, b"hel");

            let mut full = vec![0u8; 5];
            let copied = llama_server_copy_string_bytes_rust(
                b"hello".as_ptr(),
                5,
                full.as_mut_ptr(),
                full.len(),
            );
            assert_eq!(copied, 5);
            assert_eq!(full, b"hello");

            assert_eq!(
                llama_server_copy_string_bytes_rust(
                    std::ptr::null(),
                    1,
                    full.as_mut_ptr(),
                    full.len()
                ),
                0
            );
            assert_eq!(
                llama_server_copy_string_bytes_rust(b"hello".as_ptr(), 5, std::ptr::null_mut(), 1),
                0
            );
        }
    }

    #[test]
    fn maps_model_statuses() {
        unsafe {
            assert_eq!(
                llama_server_model_status_from_string_rust(b"unloaded".as_ptr(), 8),
                SERVER_MODEL_STATUS_UNLOADED
            );
            assert_eq!(
                llama_server_model_status_from_string_rust(b"loading".as_ptr(), 7),
                SERVER_MODEL_STATUS_LOADING
            );
            assert_eq!(
                llama_server_model_status_from_string_rust(b"loaded".as_ptr(), 6),
                SERVER_MODEL_STATUS_LOADED
            );
            assert_eq!(
                llama_server_model_status_from_string_rust(b"sleeping".as_ptr(), 8),
                SERVER_MODEL_STATUS_SLEEPING
            );
            assert_eq!(
                llama_server_model_status_from_string_rust(b"bad".as_ptr(), 3),
                -1
            );
        }

        assert!(llama_server_model_status_is_ready_rust(
            SERVER_MODEL_STATUS_LOADED
        ));
        assert!(!llama_server_model_status_is_ready_rust(
            SERVER_MODEL_STATUS_LOADING
        ));

        assert!(!llama_server_model_status_is_running_rust(
            SERVER_MODEL_STATUS_UNLOADED
        ));
        assert!(llama_server_model_status_is_running_rust(
            SERVER_MODEL_STATUS_LOADING
        ));
        assert!(llama_server_model_status_is_running_rust(
            SERVER_MODEL_STATUS_LOADED
        ));
        assert!(llama_server_model_status_is_running_rust(
            SERVER_MODEL_STATUS_SLEEPING
        ));

        assert!(llama_server_model_status_is_failed_rust(
            SERVER_MODEL_STATUS_UNLOADED,
            1
        ));
        assert!(!llama_server_model_status_is_failed_rust(
            SERVER_MODEL_STATUS_UNLOADED,
            0
        ));
        assert!(!llama_server_model_status_is_failed_rust(
            SERVER_MODEL_STATUS_LOADING,
            1
        ));

        assert!(llama_server_model_status_is_unloaded_rust(
            SERVER_MODEL_STATUS_UNLOADED
        ));
        assert!(!llama_server_model_status_is_unloaded_rust(
            SERVER_MODEL_STATUS_LOADED
        ));
        assert!(llama_server_model_status_is_loading_rust(
            SERVER_MODEL_STATUS_LOADING
        ));
        assert!(!llama_server_model_status_is_loading_rust(
            SERVER_MODEL_STATUS_SLEEPING
        ));
        assert!(llama_server_model_status_is_sleeping_rust(
            SERVER_MODEL_STATUS_SLEEPING
        ));
        assert!(!llama_server_model_status_is_sleeping_rust(
            SERVER_MODEL_STATUS_UNLOADED
        ));
    }

    #[test]
    fn compares_and_filters_header_names() {
        unsafe {
            assert!(llama_server_header_name_is_rust(
                b"Content-Type".as_ptr(),
                12,
                b"content-type\0".as_ptr().cast()
            ));
            assert!(llama_server_should_strip_proxy_header_rust(
                b"Server".as_ptr(),
                6
            ));
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
    fn sanitizes_multipart_fields() {
        assert_eq!(sanitize_multipart_field(b"plain"), b"plain");
        assert_eq!(sanitize_multipart_field(b"a\nb\rc\"d"), b"abcd");

        unsafe {
            let mut len = 0usize;
            let ptr =
                llama_server_sanitize_multipart_field_rust(b"a\nb\rc\"d".as_ptr(), 7, &mut len);
            assert!(!ptr.is_null());
            assert_eq!(slice::from_raw_parts(ptr, len), b"abcd");
            llama_server_base64_free(ptr, len);

            let ptr = llama_server_sanitize_multipart_field_rust(std::ptr::null(), 1, &mut len);
            assert!(ptr.is_null());
            assert_eq!(len, 0);
        }
    }

    #[test]
    fn computes_fnv1a_hashes() {
        unsafe {
            assert_eq!(
                llama_server_fnv_hash_rust(b"".as_ptr(), 0),
                14695981039346656037
            );
            assert_eq!(
                llama_server_fnv_hash_rust(b"hello".as_ptr(), 5),
                11831194018420276491
            );
            assert_eq!(llama_server_fnv_hash_rust(std::ptr::null(), 1), 0);
        }
    }

    #[test]
    fn classifies_base64_characters() {
        assert!(llama_server_is_base64_char_rust(b'A'));
        assert!(llama_server_is_base64_char_rust(b'z'));
        assert!(llama_server_is_base64_char_rust(b'0'));
        assert!(llama_server_is_base64_char_rust(b'+'));
        assert!(llama_server_is_base64_char_rust(b'/'));
        assert!(!llama_server_is_base64_char_rust(b'='));
        assert!(!llama_server_is_base64_char_rust(b'-'));
    }

    #[test]
    fn normalizes_anthropic_billing_header_cch() {
        let mut header =
            b"x-anthropic-billing-header: cc_version=2.1; cch=a5145;You are Claude".to_vec();
        let status = unsafe {
            llama_server_normalize_anthropic_billing_header_rust(header.as_mut_ptr(), header.len())
        };
        assert_eq!(status, 1);
        assert_eq!(
            header,
            b"x-anthropic-billing-header: cc_version=2.1; cch=fffff;You are Claude"
        );

        let mut other = b"system prompt cch=a5145;".to_vec();
        let status = unsafe {
            llama_server_normalize_anthropic_billing_header_rust(other.as_mut_ptr(), other.len())
        };
        assert_eq!(status, 0);
        assert_eq!(other, b"system prompt cch=a5145;");

        let mut malformed = b"x-anthropic-billing-header: cch=a5145 no semicolon".to_vec();
        let status = unsafe {
            llama_server_normalize_anthropic_billing_header_rust(
                malformed.as_mut_ptr(),
                malformed.len(),
            )
        };
        assert_eq!(status, -1);
        assert_eq!(
            malformed,
            b"x-anthropic-billing-header: cch=a5145 no semicolon"
        );

        let status = unsafe {
            llama_server_normalize_anthropic_billing_header_rust(std::ptr::null_mut(), 1)
        };
        assert_eq!(status, 0);
    }

    #[test]
    fn computes_probability_logarithms_without_negative_infinity() {
        assert_eq!(llama_server_probability_logarithm_rust(0.0), f32::MIN);
        assert_eq!(llama_server_probability_logarithm_rust(1.0), 0.0);
        let half = llama_server_probability_logarithm_rust(0.5);
        assert!((half - 0.5_f32.ln()).abs() < 0.000001);
    }

    #[test]
    fn validates_utf8_prefix_lengths_like_server_helper() {
        unsafe {
            let incomplete_two = [0xc3u8];
            let incomplete_three = [b'a', 0xe2, 0x82];
            assert_eq!(
                llama_server_validate_utf8_prefix_len_rust(b"hello".as_ptr(), 5),
                5
            );
            assert_eq!(
                llama_server_validate_utf8_prefix_len_rust("é".as_bytes().as_ptr(), 2),
                2
            );
            assert_eq!(
                llama_server_validate_utf8_prefix_len_rust(
                    incomplete_two.as_ptr(),
                    incomplete_two.len()
                ),
                0
            );
            assert_eq!(
                llama_server_validate_utf8_prefix_len_rust(
                    incomplete_three.as_ptr(),
                    incomplete_three.len()
                ),
                1
            );
            assert_eq!(
                llama_server_validate_utf8_prefix_len_rust(std::ptr::null(), 1),
                0
            );
        }
    }

    #[test]
    fn validates_utf8_like_server_helper() {
        unsafe {
            let invalid_continuation = [0x80u8];
            let incomplete_two = [0xc3u8];
            let incomplete_three = [0xe2u8, 0x82];
            assert!(llama_server_is_valid_utf8_rust(b"hello".as_ptr(), 5));
            assert!(llama_server_is_valid_utf8_rust("é".as_bytes().as_ptr(), 2));
            assert!(llama_server_is_valid_utf8_rust("😀".as_bytes().as_ptr(), 4));
            assert!(!llama_server_is_valid_utf8_rust(
                invalid_continuation.as_ptr(),
                invalid_continuation.len()
            ));
            assert!(!llama_server_is_valid_utf8_rust(
                incomplete_two.as_ptr(),
                incomplete_two.len()
            ));
            assert!(!llama_server_is_valid_utf8_rust(
                incomplete_three.as_ptr(),
                incomplete_three.len()
            ));
            assert!(!llama_server_is_valid_utf8_rust(std::ptr::null(), 1));
        }
    }

    #[test]
    fn parses_autoload_query_values() {
        unsafe {
            assert!(llama_server_is_autoload_rust(true, b"".as_ptr(), 0));
            assert!(!llama_server_is_autoload_rust(false, b"".as_ptr(), 0));
            assert!(llama_server_is_autoload_rust(false, b"true".as_ptr(), 4));
            assert!(llama_server_is_autoload_rust(false, b"1".as_ptr(), 1));
            assert!(!llama_server_is_autoload_rust(true, b"false".as_ptr(), 5));
            assert!(!llama_server_is_autoload_rust(true, b"0".as_ptr(), 1));
            assert!(llama_server_is_autoload_rust(true, std::ptr::null(), 1));
            assert!(!llama_server_is_autoload_rust(false, std::ptr::null(), 1));
        }
    }

    #[test]
    fn evaluates_lora_cache_helpers() {
        unsafe {
            let scales = [0.0, 1.0, -0.5];
            let all_alora = [0u8, 1, 1];
            assert!(llama_server_lora_all_alora_rust(
                scales.as_ptr(),
                all_alora.as_ptr(),
                scales.len()
            ));

            let mixed_alora = [0u8, 1, 0];
            assert!(!llama_server_lora_all_alora_rust(
                scales.as_ptr(),
                mixed_alora.as_ptr(),
                scales.len()
            ));

            let disabled = [0.0, 0.0];
            let flags = [1u8, 1];
            assert!(!llama_server_lora_all_alora_rust(
                disabled.as_ptr(),
                flags.as_ptr(),
                disabled.len()
            ));
            assert!(!llama_server_lora_all_alora_rust(
                std::ptr::null(),
                flags.as_ptr(),
                1
            ));

            assert!(!llama_server_lora_should_clear_cache_rust(
                false, false, true
            ));
            assert!(!llama_server_lora_should_clear_cache_rust(true, true, true));
            assert!(llama_server_lora_should_clear_cache_rust(true, false, true));
            assert!(llama_server_lora_should_clear_cache_rust(
                false, false, false
            ));

            let left_scales = [1.0, 0.0];
            let left_ptrs = [7usize, 11usize];
            let right_scales = [1.0, 0.0];
            let right_ptrs = [7usize, 11usize];
            assert!(llama_server_lora_are_equal_rust(
                left_scales.as_ptr(),
                left_ptrs.as_ptr(),
                left_scales.len(),
                right_scales.as_ptr(),
                right_ptrs.as_ptr(),
                right_scales.len()
            ));

            let different_ptrs = [7usize, 12usize];
            assert!(!llama_server_lora_are_equal_rust(
                left_scales.as_ptr(),
                left_ptrs.as_ptr(),
                left_scales.len(),
                right_scales.as_ptr(),
                different_ptrs.as_ptr(),
                right_scales.len()
            ));

            let scales = [-1.0, 0.0, 0.25, 2.0];
            let mut out = [usize::MAX; 4];
            let count = llama_server_lora_enabled_ids_rust(
                scales.as_ptr(),
                scales.len(),
                out.as_mut_ptr(),
                out.len(),
            );
            assert_eq!(count, 2);
            assert_eq!(&out[..2], &[2, 3]);

            let mut short = [usize::MAX; 1];
            let count = llama_server_lora_enabled_ids_rust(
                scales.as_ptr(),
                scales.len(),
                short.as_mut_ptr(),
                short.len(),
            );
            assert_eq!(count, 2);
            assert_eq!(short, [2]);
            assert_eq!(
                llama_server_lora_enabled_ids_rust(
                    std::ptr::null(),
                    1,
                    out.as_mut_ptr(),
                    out.len()
                ),
                0
            );
        }
    }

    #[test]
    fn evaluates_json_array_type_flags() {
        unsafe {
            let numbers = [1u8, 1, 1];
            assert!(llama_server_json_flags_all_numbers_rust(
                numbers.as_ptr(),
                numbers.len()
            ));
            assert!(llama_server_json_flags_contains_number_rust(
                numbers.as_ptr(),
                numbers.len()
            ));
            assert!(!llama_server_json_flags_mixed_numbers_strings_rust(
                numbers.as_ptr(),
                numbers.len()
            ));

            let mixed = [2u8, 0, 1];
            assert!(!llama_server_json_flags_all_numbers_rust(
                mixed.as_ptr(),
                mixed.len()
            ));
            assert!(llama_server_json_flags_mixed_numbers_strings_rust(
                mixed.as_ptr(),
                mixed.len()
            ));
            assert!(llama_server_json_flags_contains_number_rust(
                mixed.as_ptr(),
                mixed.len()
            ));

            let empty: [u8; 0] = [];
            assert!(llama_server_json_flags_all_numbers_rust(
                empty.as_ptr(),
                empty.len()
            ));
            assert!(!llama_server_json_flags_contains_number_rust(
                empty.as_ptr(),
                empty.len()
            ));
            assert!(!llama_server_json_flags_all_numbers_rust(
                std::ptr::null(),
                1
            ));
        }
    }

    #[test]
    fn formats_token_output_bytes() {
        unsafe {
            let mut len = 0usize;
            let plain = llama_server_format_token_output_rust(b"a".as_ptr(), 1, &mut len);
            assert_eq!(std::slice::from_raw_parts(plain, len), b"a");
            llama_server_base64_free(plain, len);

            let high = [0x80u8];
            let formatted =
                llama_server_format_token_output_rust(high.as_ptr(), high.len(), &mut len);
            assert_eq!(std::slice::from_raw_parts(formatted, len), b"byte: \\x80");
            llama_server_base64_free(formatted, len);

            let multi = [0xc3u8, 0xa9];
            let unchanged =
                llama_server_format_token_output_rust(multi.as_ptr(), multi.len(), &mut len);
            assert_eq!(std::slice::from_raw_parts(unchanged, len), &multi);
            llama_server_base64_free(unchanged, len);

            let null = llama_server_format_token_output_rust(std::ptr::null(), 1, &mut len);
            assert!(null.is_null());
            assert_eq!(len, 0);
        }
    }

    #[test]
    fn formats_openai_sse_payloads() {
        unsafe {
            let items = [
                StringView {
                    data: br#"{"a":1}"#.as_ptr(),
                    len: br#"{"a":1}"#.len(),
                },
                StringView {
                    data: br#""done""#.as_ptr(),
                    len: br#""done""#.len(),
                },
            ];
            let mut len = 0usize;
            let ptr = llama_server_format_oai_sse_rust(items.as_ptr(), items.len(), &mut len);
            assert_eq!(
                std::slice::from_raw_parts(ptr, len),
                br#"data: {"a":1}

data: "done"

"#
            );
            llama_server_base64_free(ptr, len);

            let ptr = llama_server_format_oai_sse_rust(std::ptr::null(), 1, &mut len);
            assert!(ptr.is_null());
            assert_eq!(len, 0);
        }
    }

    #[test]
    fn formats_openai_response_sse_payloads() {
        unsafe {
            let events = [
                SseEventView {
                    event: StringView {
                        data: b"response.created".as_ptr(),
                        len: "response.created".len(),
                    },
                    data: StringView {
                        data: br#"{"id":"r1"}"#.as_ptr(),
                        len: br#"{"id":"r1"}"#.len(),
                    },
                },
                SseEventView {
                    event: StringView {
                        data: b"response.done".as_ptr(),
                        len: "response.done".len(),
                    },
                    data: StringView {
                        data: br#"{"ok":true}"#.as_ptr(),
                        len: br#"{"ok":true}"#.len(),
                    },
                },
            ];
            let mut len = 0usize;
            let ptr =
                llama_server_format_oai_resp_sse_rust(events.as_ptr(), events.len(), &mut len);
            assert_eq!(
                std::slice::from_raw_parts(ptr, len),
                br#"event: response.created
data: {"id":"r1"}

event: response.done
data: {"ok":true}

"#
            );
            llama_server_base64_free(ptr, len);

            let ptr = llama_server_format_oai_resp_sse_rust(std::ptr::null(), 1, &mut len);
            assert!(ptr.is_null());
            assert_eq!(len, 0);
        }
    }

    #[test]
    fn formats_anthropic_sse_payloads() {
        unsafe {
            let events = [
                AnthropicSseEventView {
                    event: StringView {
                        data: b"message_start".as_ptr(),
                        len: "message_start".len(),
                    },
                    data: StringView {
                        data: br#"{"type":"message_start"}"#.as_ptr(),
                        len: br#"{"type":"message_start"}"#.len(),
                    },
                    has_event: 1,
                },
                AnthropicSseEventView {
                    event: StringView {
                        data: std::ptr::null(),
                        len: 0,
                    },
                    data: StringView {
                        data: br#"{"delta":"x"}"#.as_ptr(),
                        len: br#"{"delta":"x"}"#.len(),
                    },
                    has_event: 0,
                },
            ];
            let mut len = 0usize;
            let ptr =
                llama_server_format_anthropic_sse_rust(events.as_ptr(), events.len(), &mut len);
            assert_eq!(
                std::slice::from_raw_parts(ptr, len),
                br#"event: message_start
data: {"type":"message_start"}

data: {"delta":"x"}

"#
            );
            llama_server_base64_free(ptr, len);

            let ptr = llama_server_format_anthropic_sse_rust(std::ptr::null(), 1, &mut len);
            assert!(ptr.is_null());
            assert_eq!(len, 0);
        }
    }

    #[test]
    fn finds_token_common_prefixes() {
        unsafe {
            let left = [1, 2, 3, 4];
            let right = [1, 2, 9];
            assert_eq!(
                llama_server_token_common_prefix_rust(
                    left.as_ptr(),
                    left.len(),
                    right.as_ptr(),
                    right.len()
                ),
                2
            );
            assert_eq!(
                llama_server_token_common_prefix_rust(
                    left.as_ptr(),
                    left.len(),
                    left.as_ptr(),
                    left.len()
                ),
                left.len()
            );
            assert_eq!(
                llama_server_token_common_prefix_rust(
                    std::ptr::null(),
                    1,
                    right.as_ptr(),
                    right.len()
                ),
                0
            );
        }
    }

    #[test]
    fn filters_media_tokens_from_text_tokens() {
        unsafe {
            let input = [11, -1, 12, -1, 13];
            let mut out = [0; 5];
            let count = llama_server_filter_text_tokens_rust(
                input.as_ptr(),
                input.len(),
                out.as_mut_ptr(),
                out.len(),
            );
            assert_eq!(count, 3);
            assert_eq!(&out[..3], &[11, 12, 13]);

            let mut short = [0; 2];
            let count = llama_server_filter_text_tokens_rust(
                input.as_ptr(),
                input.len(),
                short.as_mut_ptr(),
                short.len(),
            );
            assert_eq!(count, 3);
            assert_eq!(short, [11, 12]);
            assert_eq!(
                llama_server_filter_text_tokens_rust(
                    std::ptr::null(),
                    1,
                    out.as_mut_ptr(),
                    out.len()
                ),
                0
            );
        }
    }

    #[test]
    fn formats_server_token_debug_strings() {
        unsafe {
            let tokens = [11, -1, 12];
            let media = [1usize];
            let mut len = 0usize;
            let ptr = llama_server_tokens_debug_string_rust(
                tokens.as_ptr(),
                tokens.len(),
                media.as_ptr(),
                media.len(),
                &mut len,
            );
            assert_eq!(
                std::slice::from_raw_parts(ptr, len),
                b"tokens: idx:0 11 idx:1 <embd> idx:2 12 \nimage idx: 1, "
            );
            llama_server_base64_free(ptr, len);

            let ptr = llama_server_tokens_debug_string_rust(
                std::ptr::null(),
                1,
                media.as_ptr(),
                media.len(),
                &mut len,
            );
            assert!(ptr.is_null());
            assert_eq!(len, 0);
        }
    }

    #[test]
    fn validates_token_ranges_ignoring_media_placeholders() {
        unsafe {
            let tokens = [0, 9, -1, 3];
            assert!(llama_server_tokens_valid_range_rust(
                tokens.as_ptr(),
                tokens.len(),
                10
            ));

            let negative = [0, -2, 3];
            assert!(!llama_server_tokens_valid_range_rust(
                negative.as_ptr(),
                negative.len(),
                10
            ));

            let too_large = [0, 10, 3];
            assert!(!llama_server_tokens_valid_range_rust(
                too_large.as_ptr(),
                too_large.len(),
                10
            ));
            assert!(!llama_server_tokens_valid_range_rust(
                std::ptr::null(),
                1,
                10
            ));
        }
    }

    #[test]
    fn maps_token_indices_to_positions_with_media_chunks() {
        unsafe {
            assert_eq!(
                llama_server_tokens_pos_next_rust(
                    5,
                    -1,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                ),
                5
            );
            assert_eq!(
                llama_server_tokens_size_up_to_pos_rust(
                    5,
                    3,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                ),
                3
            );
            assert_eq!(
                llama_server_tokens_size_up_to_pos_rust(
                    5,
                    -1,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                ),
                5
            );

            let media_indices = [5usize, 8];
            let media_n_pos = [2i32, 2];
            let media_n_tokens = [3usize, 3];
            assert_eq!(
                llama_server_tokens_pos_next_rust(
                    11,
                    -1,
                    media_indices.as_ptr(),
                    media_n_pos.as_ptr(),
                    media_n_tokens.as_ptr(),
                    media_indices.len(),
                ),
                9
            );
            assert_eq!(
                llama_server_tokens_pos_next_rust(
                    11,
                    8,
                    media_indices.as_ptr(),
                    media_n_pos.as_ptr(),
                    media_n_tokens.as_ptr(),
                    media_indices.len(),
                ),
                7
            );
            assert_eq!(
                llama_server_tokens_size_up_to_pos_rust(
                    11,
                    7,
                    media_indices.as_ptr(),
                    media_n_pos.as_ptr(),
                    media_n_tokens.as_ptr(),
                    media_indices.len(),
                ),
                8
            );
            assert_eq!(
                llama_server_tokens_pos_next_rust(
                    11,
                    8,
                    std::ptr::null(),
                    media_n_pos.as_ptr(),
                    media_n_tokens.as_ptr(),
                    media_indices.len(),
                ),
                0
            );
        }
    }

    #[test]
    fn classifies_task_runtime_needs_by_server_enum_value() {
        assert!(llama_server_task_need_embd_rust(SERVER_TASK_TYPE_EMBEDDING));
        assert!(llama_server_task_need_embd_rust(SERVER_TASK_TYPE_RERANK));
        assert!(!llama_server_task_need_embd_rust(
            SERVER_TASK_TYPE_COMPLETION
        ));
        assert!(!llama_server_task_need_embd_rust(SERVER_TASK_TYPE_INFILL));

        assert!(llama_server_task_need_logits_rust(
            SERVER_TASK_TYPE_COMPLETION
        ));
        assert!(llama_server_task_need_logits_rust(SERVER_TASK_TYPE_INFILL));
        assert!(!llama_server_task_need_logits_rust(
            SERVER_TASK_TYPE_EMBEDDING
        ));
        assert!(!llama_server_task_need_logits_rust(SERVER_TASK_TYPE_RERANK));

        assert!(llama_server_task_need_sampling_rust(
            SERVER_TASK_TYPE_COMPLETION
        ));
        assert!(llama_server_task_need_sampling_rust(
            SERVER_TASK_TYPE_INFILL
        ));
        assert!(!llama_server_task_need_sampling_rust(
            SERVER_TASK_TYPE_EMBEDDING
        ));
        assert!(!llama_server_task_need_sampling_rust(
            SERVER_TASK_TYPE_RERANK
        ));

        assert!(!llama_server_task_need_embd_rust(-1));
        assert!(!llama_server_task_need_logits_rust(99));
        assert!(!llama_server_task_need_sampling_rust(99));

        assert!(llama_server_task_is_cancel_rust(SERVER_TASK_TYPE_CANCEL));
        assert!(!llama_server_task_is_cancel_rust(
            SERVER_TASK_TYPE_COMPLETION
        ));
        assert!(llama_server_task_is_completion_rust(
            SERVER_TASK_TYPE_COMPLETION
        ));
        assert!(!llama_server_task_is_completion_rust(
            SERVER_TASK_TYPE_INFILL
        ));
        assert!(llama_server_task_is_embedding_rust(
            SERVER_TASK_TYPE_EMBEDDING
        ));
        assert!(!llama_server_task_is_embedding_rust(
            SERVER_TASK_TYPE_RERANK
        ));
        assert!(llama_server_task_is_rerank_rust(SERVER_TASK_TYPE_RERANK));
        assert!(!llama_server_task_is_rerank_rust(
            SERVER_TASK_TYPE_EMBEDDING
        ));
        assert!(llama_server_task_is_completion_or_infill_rust(
            SERVER_TASK_TYPE_COMPLETION
        ));
        assert!(llama_server_task_is_completion_or_infill_rust(
            SERVER_TASK_TYPE_INFILL
        ));
        assert!(!llama_server_task_is_completion_or_infill_rust(
            SERVER_TASK_TYPE_CANCEL
        ));

        assert!(llama_server_response_type_is_oai_embd_rust(
            TASK_RESPONSE_TYPE_OAI_EMBD
        ));
        assert!(!llama_server_response_type_is_oai_embd_rust(0));
        assert!(!llama_server_response_type_is_oai_embd_rust(1));
        assert!(!llama_server_response_type_is_oai_embd_rust(6));

        assert!(!llama_server_response_type_is_non_native_rust(
            TASK_RESPONSE_TYPE_NONE
        ));
        assert!(llama_server_response_type_is_non_native_rust(
            TASK_RESPONSE_TYPE_OAI_CHAT
        ));

        assert!(llama_server_response_type_is_oai_chat_or_cmpl_rust(
            TASK_RESPONSE_TYPE_OAI_CHAT
        ));
        assert!(llama_server_response_type_is_oai_chat_or_cmpl_rust(
            TASK_RESPONSE_TYPE_OAI_CMPL
        ));
        assert!(!llama_server_response_type_is_oai_chat_or_cmpl_rust(
            TASK_RESPONSE_TYPE_OAI_RESP
        ));

        assert!(llama_server_response_type_is_anthropic_rust(
            TASK_RESPONSE_TYPE_ANTHROPIC
        ));
        assert!(!llama_server_response_type_is_anthropic_rust(
            TASK_RESPONSE_TYPE_OAI_RESP
        ));
        assert!(llama_server_response_type_is_oai_resp_rust(
            TASK_RESPONSE_TYPE_OAI_RESP
        ));
        assert!(!llama_server_response_type_is_oai_resp_rust(
            TASK_RESPONSE_TYPE_ANTHROPIC
        ));

        assert!(llama_server_response_type_stream_done_is_empty_rust(
            TASK_RESPONSE_TYPE_NONE
        ));
        assert!(llama_server_response_type_stream_done_is_empty_rust(
            TASK_RESPONSE_TYPE_OAI_RESP
        ));
        assert!(llama_server_response_type_stream_done_is_empty_rust(
            TASK_RESPONSE_TYPE_ANTHROPIC
        ));
        assert!(!llama_server_response_type_stream_done_is_empty_rust(
            TASK_RESPONSE_TYPE_OAI_CHAT
        ));

        assert!(!llama_server_task_is_parent_rust(0));
        assert!(llama_server_task_is_parent_rust(1));
        assert!(!llama_server_task_is_child_rust(-1));
        assert!(llama_server_task_is_child_rust(0));

        assert!(!llama_server_slot_is_processing_rust(SLOT_STATE_IDLE));
        assert!(llama_server_slot_is_processing_rust(1));
        assert!(llama_server_slot_is_processing_rust(5));

        unsafe {
            let ids = [7, 9, 7, -1, 9, 11];
            let mut out = [0; 6];
            let count = llama_server_unique_task_ids_rust(
                ids.as_ptr(),
                ids.len(),
                out.as_mut_ptr(),
                out.len(),
            );
            assert_eq!(count, 4);
            assert_eq!(&out[..count], &[7, 9, -1, 11]);

            let mut short_out = [0; 2];
            let count = llama_server_unique_task_ids_rust(
                ids.as_ptr(),
                ids.len(),
                short_out.as_mut_ptr(),
                short_out.len(),
            );
            assert_eq!(count, 4);
            assert_eq!(short_out, [7, 9]);
            assert_eq!(
                llama_server_unique_task_ids_rust(std::ptr::null(), 1, out.as_mut_ptr(), out.len()),
                0
            );
        }
    }

    #[test]
    fn computes_slot_budget_and_remaining_tokens() {
        unsafe {
            let mut n_remaining = 123;
            assert!(llama_server_slot_has_budget_rust(
                -1,
                -1,
                99,
                &mut n_remaining
            ));
            assert_eq!(n_remaining, -1);

            assert!(llama_server_slot_has_budget_rust(
                5,
                -1,
                2,
                &mut n_remaining
            ));
            assert_eq!(n_remaining, 3);

            assert!(!llama_server_slot_has_budget_rust(
                5,
                -1,
                5,
                &mut n_remaining
            ));
            assert_eq!(n_remaining, 0);

            assert!(llama_server_slot_has_budget_rust(
                -1,
                4,
                3,
                &mut n_remaining
            ));
            assert_eq!(n_remaining, 1);

            assert!(!llama_server_slot_has_budget_rust(
                5,
                -1,
                2,
                std::ptr::null_mut()
            ));
        }
    }

    #[test]
    fn classifies_slot_split_policy() {
        assert!(llama_server_slot_can_split_rust(false, false, false));
        assert!(llama_server_slot_can_split_rust(false, true, false));
        assert!(llama_server_slot_can_split_rust(true, true, true));
        assert!(!llama_server_slot_can_split_rust(true, false, true));
        assert!(!llama_server_slot_can_split_rust(true, true, false));
    }

    #[test]
    fn computes_slot_draft_limits() {
        assert_eq!(
            llama_server_slot_n_draft_max_rust(false, 1, 8, 32, 4, -1),
            0
        );
        assert_eq!(llama_server_slot_n_draft_max_rust(true, 1, 8, 32, 4, -1), 8);
        assert_eq!(llama_server_slot_n_draft_max_rust(true, 1, 8, 10, 4, -1), 4);
        assert_eq!(llama_server_slot_n_draft_max_rust(true, 1, 8, 32, 4, 3), 2);
        assert_eq!(llama_server_slot_n_draft_max_rust(true, 3, 8, 32, 4, 3), 0);
    }

    #[test]
    fn classifies_slot_batch_compatibility() {
        assert!(llama_server_slot_can_batch_with_rust(
            SERVER_TASK_TYPE_COMPLETION,
            SERVER_TASK_TYPE_COMPLETION,
            true
        ));
        assert!(!llama_server_slot_can_batch_with_rust(
            SERVER_TASK_TYPE_COMPLETION,
            SERVER_TASK_TYPE_INFILL,
            true
        ));
        assert!(!llama_server_slot_can_batch_with_rust(
            SERVER_TASK_TYPE_COMPLETION,
            SERVER_TASK_TYPE_COMPLETION,
            false
        ));
    }

    #[test]
    fn sums_prompt_checkpoint_sizes() {
        unsafe {
            let checkpoint_sizes = [3usize, 5, 7];
            assert_eq!(
                llama_server_prompt_size_rust(
                    11,
                    checkpoint_sizes.as_ptr(),
                    checkpoint_sizes.len()
                ),
                26
            );
            assert_eq!(llama_server_prompt_size_rust(11, std::ptr::null(), 0), 11);
            assert_eq!(llama_server_prompt_size_rust(11, std::ptr::null(), 1), 11);
            assert_eq!(
                llama_server_prompt_size_rust(usize::MAX - 1, checkpoint_sizes.as_ptr(), 1),
                usize::MAX
            );
        }
    }

    #[test]
    fn sums_usize_slices() {
        unsafe {
            let values = [2usize, 4, 8];
            assert_eq!(
                llama_server_sum_usize_rust(values.as_ptr(), values.len()),
                14
            );
            assert_eq!(llama_server_sum_usize_rust(std::ptr::null(), 0), 0);
            assert_eq!(llama_server_sum_usize_rust(std::ptr::null(), 1), 0);

            let overflowing = [usize::MAX - 1, 2];
            assert_eq!(
                llama_server_sum_usize_rust(overflowing.as_ptr(), overflowing.len()),
                usize::MAX
            );
        }
    }

    #[test]
    fn reports_token_collection_lengths() {
        assert_eq!(llama_server_tokens_len_rust(0), 0);
        assert_eq!(llama_server_tokens_len_rust(3), 3);
        assert!(llama_server_tokens_is_empty_rust(0));
        assert!(!llama_server_tokens_is_empty_rust(1));
        assert_eq!(llama_server_tokens_len_i32_rust(17), 17);
        assert_eq!(
            llama_server_tokens_len_i32_rust((i32::MAX as usize) + 1),
            i32::MIN
        );
    }

    #[test]
    fn checks_json_field_type_presence() {
        assert!(llama_server_json_field_is_array_rust(true, true));
        assert!(!llama_server_json_field_is_array_rust(true, false));
        assert!(!llama_server_json_field_is_array_rust(false, true));
        assert!(llama_server_json_field_is_string_rust(true, true));
        assert!(!llama_server_json_field_is_string_rust(true, false));
        assert!(!llama_server_json_field_is_string_rust(false, true));
    }

    #[test]
    fn checks_queue_wait_predicates() {
        assert!(!llama_server_queue_should_process_rust(true, true));
        assert!(llama_server_queue_should_process_rust(true, false));
        assert!(llama_server_queue_should_process_rust(false, true));

        assert!(!llama_server_queue_sleep_wait_done_rust(true, false));
        assert!(llama_server_queue_sleep_wait_done_rust(true, true));
        assert!(llama_server_queue_sleep_wait_done_rust(false, false));

        assert!(!llama_server_queue_task_wait_done_rust(true, true));
        assert!(llama_server_queue_task_wait_done_rust(true, false));
        assert!(llama_server_queue_task_wait_done_rust(false, true));
    }

    #[test]
    fn checks_queue_sleep_idle_threshold() {
        assert!(!llama_server_queue_should_sleep_rust(-1, 200, 100));
        assert!(!llama_server_queue_should_sleep_rust(150, 200, 100));
        assert!(llama_server_queue_should_sleep_rust(100, 200, 100));
        assert!(llama_server_queue_should_sleep_rust(0, 100, 100));
        assert!(llama_server_queue_should_sleep_rust(10, i64::MAX, i64::MIN));
    }

    #[test]
    fn checks_descending_comparators() {
        assert!(llama_server_score_desc_rust(0.75, 0.25));
        assert!(!llama_server_score_desc_rust(0.25, 0.75));
        assert!(!llama_server_score_desc_rust(0.5, 0.5));

        assert!(llama_server_logit_desc_rust(2.0, -1.0));
        assert!(!llama_server_logit_desc_rust(-1.0, 2.0));
        assert!(!llama_server_logit_desc_rust(1.0, 1.0));

        assert!(llama_server_line_start_desc_rust(30, 10));
        assert!(!llama_server_line_start_desc_rust(10, 30));
        assert!(!llama_server_line_start_desc_rust(10, 10));
    }

    #[test]
    fn checks_id_matching_predicate() {
        assert!(llama_server_id_matches_rust(42, 42));
        assert!(!llama_server_id_matches_rust(42, -42));
    }

    #[test]
    fn checks_http_stream_and_checkpoint_predicates() {
        assert!(llama_server_http_res_is_stream_rust(true));
        assert!(!llama_server_http_res_is_stream_rust(false));

        assert!(llama_server_checkpoint_before_threshold_rust(4, 8));
        assert!(llama_server_checkpoint_before_threshold_rust(0, 0));
        assert!(llama_server_checkpoint_before_threshold_rust(0, -1));
        assert!(!llama_server_checkpoint_before_threshold_rust(8, 8));
        assert!(!llama_server_checkpoint_before_threshold_rust(9, 8));
    }

    #[test]
    fn checks_response_reader_pending_state() {
        assert!(llama_server_response_reader_has_next_rust(false, 0, 1));
        assert!(llama_server_response_reader_has_next_rust(false, 1, 2));
        assert!(!llama_server_response_reader_has_next_rust(true, 0, 1));
        assert!(!llama_server_response_reader_has_next_rust(false, 1, 1));
        assert!(!llama_server_response_reader_has_next_rust(false, 2, 1));
    }

    #[test]
    fn computes_slot_timings() {
        let timings = llama_server_slot_timings_rust(4, 8, 20.0, 10, 25.0, 3, 2);
        assert_eq!(timings.cache_n, 4);
        assert_eq!(timings.prompt_n, 8);
        assert_eq!(timings.prompt_ms, 20.0);
        assert_eq!(timings.prompt_per_token_ms, 2.5);
        assert_eq!(timings.prompt_per_second, 400.0);
        assert_eq!(timings.predicted_n, 10);
        assert_eq!(timings.predicted_ms, 25.0);
        assert_eq!(timings.predicted_per_token_ms, 2.5);
        assert_eq!(timings.predicted_per_second, 400.0);
        assert_eq!(timings.draft_n, 3);
        assert_eq!(timings.draft_n_accepted, 2);

        let no_draft = llama_server_slot_timings_rust(0, 1, 2.0, 1, 4.0, 0, 99);
        assert_eq!(no_draft.draft_n, 0);
        assert_eq!(no_draft.draft_n_accepted, 0);
    }

    #[test]
    fn maps_stop_types_to_strings() {
        unsafe {
            assert_eq!(
                std::ffi::CStr::from_ptr(llama_server_stop_type_to_string_rust(0)).to_bytes(),
                b"none"
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(llama_server_stop_type_to_string_rust(1)).to_bytes(),
                b"eos"
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(llama_server_stop_type_to_string_rust(2)).to_bytes(),
                b"word"
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(llama_server_stop_type_to_string_rust(3)).to_bytes(),
                b"limit"
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(llama_server_stop_type_to_string_rust(99)).to_bytes(),
                b"none"
            );
        }
    }
}
