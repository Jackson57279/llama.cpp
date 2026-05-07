pub const LLAMA_MAX_SEQ: usize = 256;

#[repr(C)]
pub struct LlamaByteBuffer {
    data: *mut u8,
    len: usize,
}

#[no_mangle]
pub extern "C" fn llama_max_parallel_sequences() -> usize {
    LLAMA_MAX_SEQ
}

#[no_mangle]
pub extern "C" fn llama_is_power_of_2_rust(n: i32) -> bool {
    (n & (n - 1)) == 0
}

#[no_mangle]
pub extern "C" fn llama_is_digit_char_rust(c: u8) -> bool {
    c.is_ascii_digit()
}

#[no_mangle]
pub extern "C" fn llama_is_word_char_rust(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'-'
}

#[no_mangle]
pub extern "C" fn llama_model_saver_supports_arch_rust(arch: i32) -> bool {
    !matches!(
        arch,
        29 | 32 | 33 | 39 | 47 | 48 | 58 | 61 | 72 | 73 | 82 | 97 | 113 | 121 | 122
    )
}

#[no_mangle]
pub extern "C" fn llama_llm_arch_is_recurrent_rust(arch: i32) -> bool {
    matches!(arch, 52 | 53 | 83 | 84 | 85 | 86)
}

#[no_mangle]
pub extern "C" fn llama_llm_arch_is_hybrid_rust(arch: i32) -> bool {
    matches!(
        arch,
        54 | 55 | 38 | 89 | 105 | 106 | 78 | 79 | 29 | 125 | 32 | 33
    )
}

#[no_mangle]
pub extern "C" fn llama_llm_arch_is_diffusion_rust(arch: i32) -> bool {
    matches!(arch, 107 | 109 | 110 | 116)
}

#[no_mangle]
pub extern "C" fn llama_llm_arch_supports_sm_tensor_rust(arch: i32) -> bool {
    !matches!(
        arch,
        6 | 10
            | 38
            | 44
            | 48
            | 52
            | 53
            | 54
            | 55
            | 61
            | 62
            | 66
            | 71
            | 72
            | 73
            | 78
            | 79
            | 89
            | 105
            | 106
            | 114
            | 119
            | 125
    )
}

#[no_mangle]
pub unsafe extern "C" fn llama_tensor_name_match_token_embd_rust(tensor_name: *const u8) -> bool {
    let Some(name) = (unsafe { cstr_bytes(tensor_name) }) else {
        return false;
    };
    tensor_name_match_token_embd(name)
}

#[no_mangle]
pub unsafe extern "C" fn llama_tensor_name_match_output_weight_rust(
    tensor_name: *const u8,
) -> bool {
    let Some(name) = (unsafe { cstr_bytes(tensor_name) }) else {
        return false;
    };
    name == b"output.weight"
}

#[no_mangle]
pub unsafe extern "C" fn llama_bytes_ends_with_rust(
    data: *const u8,
    len: usize,
    suffix: *const u8,
    suffix_len: usize,
) -> bool {
    let Some(data) = ffi_bytes(data, len) else {
        return false;
    };
    let Some(suffix) = ffi_bytes(suffix, suffix_len) else {
        return false;
    };
    data.ends_with(suffix)
}

#[no_mangle]
pub extern "C" fn llama_seq_pos_or_none_rust(is_empty: bool, value: i32) -> i32 {
    if is_empty {
        -1
    } else {
        value
    }
}

#[no_mangle]
pub extern "C" fn llama_adapter_invocation_token_count_rust(
    is_null: bool,
    token_count: usize,
) -> u64 {
    if is_null {
        0
    } else {
        token_count as u64
    }
}

#[no_mangle]
pub extern "C" fn llama_memory_recurrent_n_rs_rust(
    is_full: bool,
    size: u32,
    partial_n: u32,
) -> u32 {
    if is_full {
        size
    } else {
        partial_n
    }
}

#[no_mangle]
pub extern "C" fn llama_memory_recurrent_head_rust(is_full: bool, head: u32) -> u32 {
    if is_full {
        0
    } else {
        head
    }
}

#[no_mangle]
pub extern "C" fn llama_memory_recurrent_rs_z_rust(is_full: bool, rs_z: i32) -> i32 {
    if is_full {
        0
    } else {
        rs_z
    }
}

#[no_mangle]
pub extern "C" fn llama_model_n_gpu_layers_rust(param_n_gpu_layers: i32, n_layer: u32) -> u32 {
    if param_n_gpu_layers >= 0 {
        param_n_gpu_layers as u32
    } else {
        n_layer.saturating_add(1)
    }
}

#[no_mangle]
pub extern "C" fn llama_model_rope_freq_rust(
    is_swa: bool,
    swa_value: f32,
    default_value: f32,
) -> f32 {
    if is_swa {
        swa_value
    } else {
        default_value
    }
}

#[no_mangle]
pub extern "C" fn llama_model_mrope_rope_type_rust(
    use_mrope: bool,
    mrope_type: i32,
    fallback_type: i32,
) -> i32 {
    if use_mrope {
        mrope_type
    } else {
        fallback_type
    }
}

#[no_mangle]
pub extern "C" fn llama_yarn_get_mscale_rust(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

#[no_mangle]
pub extern "C" fn llama_detokenize_result_rust(total: i32, text_len_max: i32) -> i32 {
    if total <= text_len_max {
        total
    } else {
        -total
    }
}

#[no_mangle]
pub extern "C" fn llama_status_code_from_bool_rust(success: bool) -> i32 {
    if success {
        0
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn llama_file_has_direct_io_rust(fd: i32, alignment: usize) -> bool {
    fd != -1 && alignment > 1
}

#[no_mangle]
pub extern "C" fn llama_quantize_use_more_bits_rust(i_layer: i32, n_layers: i32) -> bool {
    i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2
}

#[no_mangle]
pub extern "C" fn llama_token_attr_has_rust(attr: i32, mask: i32) -> bool {
    attr & mask != 0
}

#[no_mangle]
pub extern "C" fn llama_vocab_is_eog_rust(token_id: i32, eog_count: usize) -> bool {
    token_id != -1 && eog_count > 0
}

#[no_mangle]
pub extern "C" fn llama_graph_max_nodes_linear_arch_rust(n_tokens: u32, n_tensors: usize) -> u32 {
    n_tokens
        .saturating_mul(40)
        .max((n_tensors as u32).saturating_mul(32))
}

#[no_mangle]
pub extern "C" fn llama_ring_buffer_is_empty_rust(size: usize) -> bool {
    size == 0
}

#[no_mangle]
pub extern "C" fn llama_token_logit_desc_rust(left: f32, right: f32) -> bool {
    left > right
}

#[no_mangle]
pub extern "C" fn llama_spm_bigram_after_rust(
    left_score: f32,
    right_score: f32,
    left_index: i32,
    right_index: i32,
) -> bool {
    left_score < right_score || (left_score == right_score && left_index > right_index)
}

#[no_mangle]
pub extern "C" fn llama_bpe_bigram_after_rust(
    left_rank: i32,
    right_rank: i32,
    left_index: i32,
    right_index: i32,
) -> bool {
    left_rank > right_rank || (left_rank == right_rank && left_index > right_index)
}

#[no_mangle]
pub extern "C" fn llama_shifted_score_asc_rust(left: f32, right: f32) -> bool {
    left < right
}

#[no_mangle]
pub extern "C" fn llama_mirostat_surprise_exceeds_mu_rust(probability: f32, mu: f32) -> bool {
    -probability.log2() > mu
}

#[no_mangle]
pub unsafe extern "C" fn llama_tensor_split_all_zero_rust(values: *const f32, len: usize) -> bool {
    if len == 0 {
        return true;
    }
    if values.is_null() {
        return true;
    }
    let values = unsafe { core::slice::from_raw_parts(values, len) };
    values.iter().all(|&value| value == 0.0)
}

#[no_mangle]
pub extern "C" fn llama_memory_has_enough_cells_rust(used_cells: u32, required_seqs: u32) -> bool {
    used_cells >= required_seqs
}

#[no_mangle]
pub extern "C" fn llama_hybrid_attn_nemotron_filter_rust(is_recurrent: bool, n_ff: u32) -> bool {
    !is_recurrent && n_ff == 0
}

#[no_mangle]
pub extern "C" fn llama_hybrid_recr_nemotron_filter_rust(is_recurrent: bool, n_ff: u32) -> bool {
    is_recurrent && n_ff == 0
}

#[no_mangle]
pub unsafe extern "C" fn llama_cstr_less_rust(left: *const u8, right: *const u8) -> bool {
    let Some(left) = cstr_bytes(left) else {
        return !right.is_null();
    };
    let Some(right) = cstr_bytes(right) else {
        return false;
    };
    left < right
}

#[no_mangle]
pub extern "C" fn llama_tensor_weight_before_rust(
    left_idx: u16,
    left_offs: usize,
    right_idx: u16,
    right_offs: usize,
) -> bool {
    if left_idx == right_idx {
        left_offs < right_offs
    } else {
        left_idx < right_idx
    }
}

#[no_mangle]
pub extern "C" fn llama_token_text_len_desc_rust(left_len: usize, right_len: usize) -> bool {
    left_len > right_len
}

#[no_mangle]
pub extern "C" fn llama_grammar_match_polarity_rust(found: bool, is_positive: bool) -> bool {
    found == is_positive
}

#[no_mangle]
pub extern "C" fn llama_grammar_token_match_rust(element_value: u32, token: i32) -> bool {
    element_value == token as u32
}

#[no_mangle]
pub extern "C" fn llama_grammar_token_not_match_rust(element_value: u32, token: i32) -> bool {
    element_value != token as u32
}

#[no_mangle]
pub unsafe extern "C" fn llama_trim_ascii_whitespace_bounds_rust(
    data: *const u8,
    len: usize,
    start_out: *mut usize,
    end_out: *mut usize,
) {
    let Some(bytes) = ffi_bytes(data, len) else {
        write_usize(start_out, 0);
        write_usize(end_out, 0);
        return;
    };

    let mut start = 0usize;
    let mut end = bytes.len();
    while start < end && bytes[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && bytes[end - 1].is_ascii_whitespace() {
        end -= 1;
    }

    write_usize(start_out, start);
    write_usize(end_out, end);
}

#[no_mangle]
pub extern "C" fn llama_grammar_type_is_char_element_rust(element_type: i32) -> bool {
    matches!(element_type, 3 | 4 | 5 | 6 | 7)
}

#[no_mangle]
pub extern "C" fn llama_grammar_type_is_end_of_sequence_rust(element_type: i32) -> bool {
    matches!(element_type, 0 | 1)
}

#[no_mangle]
pub extern "C" fn llama_tensor_category_is_attn_v_rust(category: i32) -> bool {
    matches!(category, 2 | 4 | 5)
}

#[no_mangle]
pub extern "C" fn llama_model_ftype_name_base_rust(ftype: i32) -> *const core::ffi::c_char {
    match ftype {
        0 => c"all F32".as_ptr(),
        1 => c"F16".as_ptr(),
        2 => c"Q4_0".as_ptr(),
        3 => c"Q4_1".as_ptr(),
        7 => c"Q8_0".as_ptr(),
        8 => c"Q5_0".as_ptr(),
        9 => c"Q5_1".as_ptr(),
        10 => c"Q2_K - Medium".as_ptr(),
        11 => c"Q3_K - Small".as_ptr(),
        12 => c"Q3_K - Medium".as_ptr(),
        13 => c"Q3_K - Large".as_ptr(),
        14 => c"Q4_K - Small".as_ptr(),
        15 => c"Q4_K - Medium".as_ptr(),
        16 => c"Q5_K - Small".as_ptr(),
        17 => c"Q5_K - Medium".as_ptr(),
        18 => c"Q6_K".as_ptr(),
        19 => c"IQ2_XXS - 2.0625 bpw".as_ptr(),
        20 => c"IQ2_XS - 2.3125 bpw".as_ptr(),
        21 => c"Q2_K - Small".as_ptr(),
        22 => c"IQ3_XS - 3.3 bpw".as_ptr(),
        23 => c"IQ3_XXS - 3.0625 bpw".as_ptr(),
        24 => c"IQ1_S - 1.5625 bpw".as_ptr(),
        25 => c"IQ4_NL - 4.5 bpw".as_ptr(),
        26 => c"IQ3_S - 3.4375 bpw".as_ptr(),
        27 => c"IQ3_S mix - 3.66 bpw".as_ptr(),
        28 => c"IQ2_S - 2.5 bpw".as_ptr(),
        29 => c"IQ2_M - 2.7 bpw".as_ptr(),
        30 => c"IQ4_XS - 4.25 bpw".as_ptr(),
        31 => c"IQ1_M - 1.75 bpw".as_ptr(),
        32 => c"BF16".as_ptr(),
        36 => c"TQ1_0 - 1.69 bpw ternary".as_ptr(),
        37 => c"TQ2_0 - 2.06 bpw ternary".as_ptr(),
        38 => c"MXFP4 MoE".as_ptr(),
        39 => c"NVFP4".as_ptr(),
        40 => c"Q1_0".as_ptr(),
        _ => c"unknown, may not work".as_ptr(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_unescape_rwkv_token_rust(
    data: *const u8,
    len: usize,
) -> LlamaByteBuffer {
    let Some(input) = ffi_bytes(data, len) else {
        return LlamaByteBuffer {
            data: core::ptr::null_mut(),
            len: 0,
        };
    };
    into_byte_buffer(unescape_rwkv_token(input))
}

#[no_mangle]
pub unsafe extern "C" fn llama_tensor_requires_imatrix_rust(
    tensor_name: *const u8,
    dst_type: i32,
    ftype: i32,
) -> bool {
    let Some(name) = (unsafe { cstr_bytes(tensor_name) }) else {
        return false;
    };
    if tensor_name_match_token_embd(name) || name == b"output.weight" {
        return false;
    }

    match dst_type {
        18 | 16 | 17 | 22 | 29 | 19 => true,
        10 => ftype == 21,
        _ => false,
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_tensor_get_category_rust(tensor_name: *const u8) -> i32 {
    let Some(name) = (unsafe { cstr_bytes(tensor_name) }) else {
        return 11;
    };
    tensor_get_category(name)
}

#[no_mangle]
pub extern "C" fn llama_can_reuse_kq_mask_rust(
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    n_kv: i64,
    n_tokens: u32,
    n_stream: u32,
) -> bool {
    if n_stream == 0 {
        return false;
    }
    ne0 == n_kv && ne1 == i64::from(n_tokens / n_stream) && ne2 == 1 && ne3 == i64::from(n_stream)
}

#[no_mangle]
pub extern "C" fn llama_ftype_get_default_type_rust(ftype: i32) -> i32 {
    match ftype {
        2 => 2,
        3 => 3,
        8 => 6,
        9 => 7,
        7 => 8,
        1 => 1,
        32 => 30,
        0 => 0,
        40 => 41,
        38 => 39,
        21 | 10 => 10,
        22 => 21,
        11 | 12 | 13 => 11,
        14 | 15 => 12,
        16 | 17 => 13,
        18 => 14,
        36 => 34,
        37 => 35,
        19 => 16,
        20 => 17,
        28 => 17,
        29 => 22,
        23 => 18,
        24 => 19,
        31 => 29,
        25 => 20,
        30 => 23,
        26 | 27 => 21,
        _ => 42,
    }
}

#[no_mangle]
pub unsafe extern "C" fn llama_gguf_data_to_str_rust(
    gguf_type: i32,
    data: *const u8,
    index: usize,
) -> LlamaByteBuffer {
    if data.is_null() {
        return into_byte_buffer(format!("unknown type {gguf_type}").into_bytes());
    }

    let text = unsafe {
        match gguf_type {
            0 => (*data.cast::<u8>().add(index)).to_string(),
            1 => (*data.cast::<i8>().add(index)).to_string(),
            2 => (*data.cast::<u16>().add(index)).to_string(),
            3 => (*data.cast::<i16>().add(index)).to_string(),
            4 => (*data.cast::<u32>().add(index)).to_string(),
            5 => (*data.cast::<i32>().add(index)).to_string(),
            6 => format!("{:.6}", *data.cast::<f32>().add(index)),
            7 => {
                if *data.cast::<i8>().add(index) != 0 {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            10 => (*data.cast::<u64>().add(index)).to_string(),
            11 => (*data.cast::<i64>().add(index)).to_string(),
            12 => format!("{:.6}", *data.cast::<f64>().add(index)),
            _ => format!("unknown type {gguf_type}"),
        }
    };

    into_byte_buffer(text.into_bytes())
}

#[no_mangle]
pub unsafe extern "C" fn llama_format_tensor_shape_rust(
    data: *const i64,
    len: usize,
) -> LlamaByteBuffer {
    if data.is_null() && len != 0 {
        return into_byte_buffer(Vec::new());
    }
    let dims = unsafe { core::slice::from_raw_parts(data, len) };
    into_byte_buffer(format_tensor_shape(dims).into_bytes())
}

#[no_mangle]
pub unsafe extern "C" fn llama_byte_buffer_free_rust(buffer: LlamaByteBuffer) {
    if !buffer.data.is_null() {
        unsafe {
            drop(Vec::from_raw_parts(buffer.data, buffer.len, buffer.len));
        }
    }
}

fn into_byte_buffer(bytes: Vec<u8>) -> LlamaByteBuffer {
    let mut bytes = bytes.into_boxed_slice();
    let data = bytes.as_mut_ptr();
    let len = bytes.len();
    core::mem::forget(bytes);
    LlamaByteBuffer { data, len }
}

fn unescape_rwkv_token(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    let mut escaping = false;
    let mut hex_remaining = 0u8;
    let mut hex_acc = 0u8;

    for &c in input {
        if hex_remaining != 0 {
            let value = if c >= b'a' { c - b'a' + 10 } else { c - b'0' };
            hex_acc = (hex_acc << 4).wrapping_add(value);
            hex_remaining -= 1;
            if hex_remaining == 0 {
                output.push(hex_acc);
                hex_acc = 0;
            }
            continue;
        }

        if escaping {
            if c == b't' {
                output.push(b'\t');
            } else if c == b'n' {
                output.push(b'\n');
            } else if c == b'r' {
                output.push(b'\r');
            } else if c == b'x' {
                hex_remaining = 2;
            } else {
                output.push(c);
            }
            escaping = false;
            continue;
        }

        if c == b'\\' {
            escaping = true;
            continue;
        }

        output.push(c);
    }

    output
}

fn tensor_name_match_token_embd(name: &[u8]) -> bool {
    name == b"token_embd.weight" || name == b"per_layer_token_embd.weight"
}

fn tensor_get_category(name: &[u8]) -> i32 {
    if name == b"output.weight" {
        return 10;
    }
    if tensor_name_match_token_embd(name) {
        return 0;
    }
    if contains_bytes(name, b"attn_qkv.weight") {
        return 4;
    }
    if contains_bytes(name, b"attn_kv_b.weight") {
        return 5;
    }
    if contains_bytes(name, b"attn_v.weight") {
        return 2;
    }
    if contains_bytes(name, b"attn_k.weight") {
        return 3;
    }
    if contains_bytes(name, b"attn_q.weight") {
        return 1;
    }
    if contains_bytes(name, b"attn_output.weight") {
        return 6;
    }
    if contains_bytes(name, b"ffn_up") {
        return 7;
    }
    if contains_bytes(name, b"ffn_gate") {
        return 8;
    }
    if contains_bytes(name, b"ffn_down") {
        return 9;
    }
    11
}

fn format_tensor_shape(dims: &[i64]) -> String {
    let mut out = String::new();
    for (index, dim) in dims.iter().enumerate() {
        if index != 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{dim:6}"));
    }
    out
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    needle.is_empty()
        || haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

fn ffi_bytes<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if data.is_null() {
        return None;
    }
    Some(unsafe { core::slice::from_raw_parts(data, len) })
}

fn write_usize(out: *mut usize, value: usize) {
    if !out.is_null() {
        unsafe {
            *out = value;
        }
    }
}

unsafe fn cstr_bytes<'a>(ptr: *const u8) -> Option<&'a [u8]> {
    if ptr.is_null() {
        return None;
    }
    let mut len = 0usize;
    while unsafe { *ptr.add(len) } != 0 {
        len += 1;
    }
    Some(unsafe { core::slice::from_raw_parts(ptr, len) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_max_sequence_constant() {
        assert_eq!(llama_max_parallel_sequences(), 256);
    }

    #[test]
    fn matches_existing_power_of_two_helper() {
        assert!(llama_is_power_of_2_rust(0));
        assert!(llama_is_power_of_2_rust(1));
        assert!(llama_is_power_of_2_rust(2));
        assert!(!llama_is_power_of_2_rust(3));
        assert!(llama_is_power_of_2_rust(1024));
        assert!(!llama_is_power_of_2_rust(1025));
    }

    #[test]
    fn classifies_grammar_parser_chars() {
        assert!(llama_is_digit_char_rust(b'0'));
        assert!(llama_is_digit_char_rust(b'9'));
        assert!(!llama_is_digit_char_rust(b'a'));

        assert!(llama_is_word_char_rust(b'a'));
        assert!(llama_is_word_char_rust(b'Z'));
        assert!(llama_is_word_char_rust(b'5'));
        assert!(llama_is_word_char_rust(b'-'));
        assert!(!llama_is_word_char_rust(b'_'));
        assert!(!llama_is_word_char_rust(b':'));
    }

    #[test]
    fn checks_model_saver_arch_allowlist() {
        assert!(llama_model_saver_supports_arch_rust(0));
        assert!(llama_model_saver_supports_arch_rust(1));
        assert!(llama_model_saver_supports_arch_rust(126));
        for arch in [
            29, 32, 33, 39, 47, 48, 58, 61, 72, 73, 82, 97, 113, 121, 122,
        ] {
            assert!(!llama_model_saver_supports_arch_rust(arch));
        }
    }

    #[test]
    fn classifies_llm_arch_families() {
        for arch in [52, 53, 83, 84, 85, 86] {
            assert!(llama_llm_arch_is_recurrent_rust(arch));
        }
        assert!(!llama_llm_arch_is_recurrent_rust(1));

        for arch in [54, 55, 38, 89, 105, 106, 78, 79, 29, 125, 32, 33] {
            assert!(llama_llm_arch_is_hybrid_rust(arch));
        }
        assert!(!llama_llm_arch_is_hybrid_rust(1));

        for arch in [107, 109, 110, 116] {
            assert!(llama_llm_arch_is_diffusion_rust(arch));
        }
        assert!(!llama_llm_arch_is_diffusion_rust(1));

        for arch in [
            6, 10, 38, 44, 48, 52, 53, 54, 55, 61, 62, 66, 71, 72, 73, 78, 79, 89, 105, 106, 114,
            119, 125,
        ] {
            assert!(!llama_llm_arch_supports_sm_tensor_rust(arch));
        }
        assert!(llama_llm_arch_supports_sm_tensor_rust(1));
    }

    #[test]
    fn matches_tensor_names() {
        unsafe {
            assert!(llama_tensor_name_match_token_embd_rust(
                b"token_embd.weight\0".as_ptr()
            ));
            assert!(llama_tensor_name_match_token_embd_rust(
                b"per_layer_token_embd.weight\0".as_ptr()
            ));
            assert!(!llama_tensor_name_match_token_embd_rust(
                b"output.weight\0".as_ptr()
            ));
            assert!(llama_tensor_name_match_output_weight_rust(
                b"output.weight\0".as_ptr()
            ));
            assert!(!llama_tensor_name_match_output_weight_rust(
                b"token_embd.weight\0".as_ptr()
            ));
            assert!(!llama_tensor_name_match_output_weight_rust(
                core::ptr::null()
            ));

            assert!(llama_bytes_ends_with_rust(
                b"blk.0.lora_a".as_ptr(),
                12,
                b".lora_a".as_ptr(),
                7
            ));
            assert!(llama_bytes_ends_with_rust(
                b"blk.0.lora_a".as_ptr(),
                12,
                b"".as_ptr(),
                0
            ));
            assert!(!llama_bytes_ends_with_rust(
                b"blk.0.lora_a".as_ptr(),
                12,
                b".lora_b".as_ptr(),
                7
            ));
            assert!(!llama_bytes_ends_with_rust(
                core::ptr::null(),
                1,
                b".lora_a".as_ptr(),
                7
            ));
        }
    }

    #[test]
    fn applies_llama_container_fallback_policies() {
        assert_eq!(llama_seq_pos_or_none_rust(true, 42), -1);
        assert_eq!(llama_seq_pos_or_none_rust(false, 42), 42);
        assert_eq!(llama_seq_pos_or_none_rust(false, -7), -7);

        assert_eq!(llama_adapter_invocation_token_count_rust(true, 12), 0);
        assert_eq!(llama_adapter_invocation_token_count_rust(false, 12), 12);
    }

    #[test]
    fn applies_recurrent_memory_full_cache_fallbacks() {
        assert_eq!(llama_memory_recurrent_n_rs_rust(true, 128, 7), 128);
        assert_eq!(llama_memory_recurrent_n_rs_rust(false, 128, 7), 7);
        assert_eq!(llama_memory_recurrent_head_rust(true, 11), 0);
        assert_eq!(llama_memory_recurrent_head_rust(false, 11), 11);
        assert_eq!(llama_memory_recurrent_rs_z_rust(true, -3), 0);
        assert_eq!(llama_memory_recurrent_rs_z_rust(false, -3), -3);
    }

    #[test]
    fn selects_model_parameter_fallbacks() {
        assert_eq!(llama_model_n_gpu_layers_rust(5, 32), 5);
        assert_eq!(llama_model_n_gpu_layers_rust(0, 32), 0);
        assert_eq!(llama_model_n_gpu_layers_rust(-1, 32), 33);
        assert_eq!(llama_model_n_gpu_layers_rust(-1, u32::MAX), u32::MAX);

        assert_eq!(llama_model_rope_freq_rust(true, 100.0, 200.0), 100.0);
        assert_eq!(llama_model_rope_freq_rust(false, 100.0, 200.0), 200.0);

        assert_eq!(llama_model_mrope_rope_type_rust(true, 3, 1), 3);
        assert_eq!(llama_model_mrope_rope_type_rust(false, 3, 1), 1);
    }

    #[test]
    fn computes_yarn_mscale_like_cpp_formula() {
        assert_eq!(llama_yarn_get_mscale_rust(1.0, 4.0), 1.0);
        assert_eq!(llama_yarn_get_mscale_rust(0.5, 4.0), 1.0);

        let scale = 4.0f32;
        let mscale = 2.5f32;
        let expected = 0.1 * mscale * scale.ln() + 1.0;
        assert!((llama_yarn_get_mscale_rust(scale, mscale) - expected).abs() < 1e-6);
    }

    #[test]
    fn shapes_status_and_detokenize_results() {
        assert_eq!(llama_detokenize_result_rust(0, 0), 0);
        assert_eq!(llama_detokenize_result_rust(8, 8), 8);
        assert_eq!(llama_detokenize_result_rust(9, 8), -9);
        assert_eq!(llama_detokenize_result_rust(-1, 0), -1);

        assert_eq!(llama_status_code_from_bool_rust(true), 0);
        assert_eq!(llama_status_code_from_bool_rust(false), -1);
    }

    #[test]
    fn checks_file_and_quantization_selectors() {
        assert!(!llama_file_has_direct_io_rust(-1, 4096));
        assert!(!llama_file_has_direct_io_rust(3, 1));
        assert!(llama_file_has_direct_io_rust(3, 4096));

        assert!(llama_quantize_use_more_bits_rust(0, 32));
        assert!(llama_quantize_use_more_bits_rust(28, 32));
        assert!(llama_quantize_use_more_bits_rust(6, 32));
        assert!(!llama_quantize_use_more_bits_rust(5, 32));
    }

    #[test]
    fn checks_vocab_token_attribute_helpers() {
        assert!(llama_token_attr_has_rust(0b1010, 0b0010));
        assert!(llama_token_attr_has_rust(0b1010, 0b1000));
        assert!(!llama_token_attr_has_rust(0b1010, 0b0100));

        assert!(!llama_vocab_is_eog_rust(-1, 1));
        assert!(!llama_vocab_is_eog_rust(7, 0));
        assert!(llama_vocab_is_eog_rust(7, 1));
    }

    #[test]
    fn computes_linear_arch_graph_node_reserve() {
        assert_eq!(llama_graph_max_nodes_linear_arch_rust(10, 3), 400);
        assert_eq!(llama_graph_max_nodes_linear_arch_rust(1, 100), 3200);
        assert_eq!(
            llama_graph_max_nodes_linear_arch_rust(u32::MAX, 1),
            u32::MAX
        );
    }

    #[test]
    fn checks_sampler_ring_and_logit_helpers() {
        assert!(llama_ring_buffer_is_empty_rust(0));
        assert!(!llama_ring_buffer_is_empty_rust(1));

        assert!(llama_token_logit_desc_rust(2.0, -1.0));
        assert!(!llama_token_logit_desc_rust(-1.0, 2.0));
        assert!(!llama_token_logit_desc_rust(1.0, 1.0));
    }

    #[test]
    fn checks_tokenizer_bigram_priority_helpers() {
        assert!(llama_spm_bigram_after_rust(0.1, 0.2, 1, 0));
        assert!(!llama_spm_bigram_after_rust(0.2, 0.1, 1, 0));
        assert!(llama_spm_bigram_after_rust(0.2, 0.2, 3, 2));
        assert!(!llama_spm_bigram_after_rust(0.2, 0.2, 2, 3));

        assert!(llama_bpe_bigram_after_rust(20, 10, 1, 0));
        assert!(!llama_bpe_bigram_after_rust(10, 20, 1, 0));
        assert!(llama_bpe_bigram_after_rust(10, 10, 3, 2));
        assert!(!llama_bpe_bigram_after_rust(10, 10, 2, 3));
    }

    #[test]
    fn checks_sampler_probability_cutoff_helpers() {
        assert!(llama_shifted_score_asc_rust(0.1, 0.2));
        assert!(!llama_shifted_score_asc_rust(0.2, 0.1));
        assert!(!llama_shifted_score_asc_rust(0.2, 0.2));

        assert!(llama_mirostat_surprise_exceeds_mu_rust(0.125, 2.0));
        assert!(!llama_mirostat_surprise_exceeds_mu_rust(0.25, 2.0));
    }

    #[test]
    fn checks_tensor_split_and_memory_predicates() {
        unsafe {
            assert!(llama_tensor_split_all_zero_rust(core::ptr::null(), 0));
            assert!(llama_tensor_split_all_zero_rust(core::ptr::null(), 4));

            let zeros = [0.0f32, 0.0, 0.0];
            assert!(llama_tensor_split_all_zero_rust(
                zeros.as_ptr(),
                zeros.len()
            ));

            let non_zero = [0.0f32, 0.5, 0.0];
            assert!(!llama_tensor_split_all_zero_rust(
                non_zero.as_ptr(),
                non_zero.len()
            ));
        }

        assert!(llama_memory_has_enough_cells_rust(4, 4));
        assert!(llama_memory_has_enough_cells_rust(5, 4));
        assert!(!llama_memory_has_enough_cells_rust(3, 4));
    }

    #[test]
    fn checks_nemotron_hybrid_layer_filters() {
        assert!(llama_hybrid_attn_nemotron_filter_rust(false, 0));
        assert!(!llama_hybrid_attn_nemotron_filter_rust(true, 0));
        assert!(!llama_hybrid_attn_nemotron_filter_rust(false, 1));

        assert!(llama_hybrid_recr_nemotron_filter_rust(true, 0));
        assert!(!llama_hybrid_recr_nemotron_filter_rust(false, 0));
        assert!(!llama_hybrid_recr_nemotron_filter_rust(true, 1));
    }

    #[test]
    fn checks_core_sort_comparators() {
        unsafe {
            assert!(llama_cstr_less_rust(b"cpu\0".as_ptr(), b"gpu\0".as_ptr()));
            assert!(!llama_cstr_less_rust(b"gpu\0".as_ptr(), b"cpu\0".as_ptr()));
            assert!(!llama_cstr_less_rust(b"cpu\0".as_ptr(), b"cpu\0".as_ptr()));
            assert!(llama_cstr_less_rust(core::ptr::null(), b"cpu\0".as_ptr()));
            assert!(!llama_cstr_less_rust(b"cpu\0".as_ptr(), core::ptr::null()));
        }

        assert!(llama_tensor_weight_before_rust(1, 10, 2, 0));
        assert!(!llama_tensor_weight_before_rust(2, 0, 1, 10));
        assert!(llama_tensor_weight_before_rust(1, 10, 1, 20));
        assert!(!llama_tensor_weight_before_rust(1, 20, 1, 10));

        assert!(llama_token_text_len_desc_rust(10, 3));
        assert!(!llama_token_text_len_desc_rust(3, 10));
        assert!(!llama_token_text_len_desc_rust(3, 3));
    }

    #[test]
    fn checks_grammar_match_predicates() {
        assert!(llama_grammar_match_polarity_rust(true, true));
        assert!(llama_grammar_match_polarity_rust(false, false));
        assert!(!llama_grammar_match_polarity_rust(true, false));
        assert!(!llama_grammar_match_polarity_rust(false, true));

        assert!(llama_grammar_token_match_rust(7, 7));
        assert!(!llama_grammar_token_match_rust(7, 8));
        assert!(llama_grammar_token_not_match_rust(7, 8));
        assert!(!llama_grammar_token_not_match_rust(7, 7));
    }

    #[test]
    fn trims_ascii_whitespace_bounds() {
        unsafe {
            let mut start = usize::MAX;
            let mut end = usize::MAX;
            llama_trim_ascii_whitespace_bounds_rust(
                b" \t\nhello \r\n".as_ptr(),
                11,
                &mut start,
                &mut end,
            );
            assert_eq!((start, end), (3, 8));

            llama_trim_ascii_whitespace_bounds_rust(b"hello".as_ptr(), 5, &mut start, &mut end);
            assert_eq!((start, end), (0, 5));

            llama_trim_ascii_whitespace_bounds_rust(b" \t".as_ptr(), 2, &mut start, &mut end);
            assert_eq!((start, end), (2, 2));

            llama_trim_ascii_whitespace_bounds_rust(core::ptr::null(), 1, &mut start, &mut end);
            assert_eq!((start, end), (0, 0));
        }
    }

    #[test]
    fn classifies_grammar_element_types() {
        assert!(!llama_grammar_type_is_char_element_rust(0));
        assert!(!llama_grammar_type_is_char_element_rust(1));
        assert!(!llama_grammar_type_is_char_element_rust(2));
        for element_type in 3..=7 {
            assert!(llama_grammar_type_is_char_element_rust(element_type));
        }
        assert!(!llama_grammar_type_is_char_element_rust(8));

        assert!(llama_grammar_type_is_end_of_sequence_rust(0));
        assert!(llama_grammar_type_is_end_of_sequence_rust(1));
        assert!(!llama_grammar_type_is_end_of_sequence_rust(2));
        assert!(!llama_grammar_type_is_end_of_sequence_rust(3));
    }

    #[test]
    fn classifies_attention_value_tensor_categories() {
        assert!(!llama_tensor_category_is_attn_v_rust(0));
        assert!(!llama_tensor_category_is_attn_v_rust(1));
        assert!(llama_tensor_category_is_attn_v_rust(2));
        assert!(!llama_tensor_category_is_attn_v_rust(3));
        assert!(llama_tensor_category_is_attn_v_rust(4));
        assert!(llama_tensor_category_is_attn_v_rust(5));
        assert!(!llama_tensor_category_is_attn_v_rust(6));
        assert!(!llama_tensor_category_is_attn_v_rust(11));
    }

    #[test]
    fn maps_model_ftype_names() {
        unsafe {
            let name = |ftype| {
                core::ffi::CStr::from_ptr(llama_model_ftype_name_base_rust(ftype))
                    .to_str()
                    .unwrap()
            };
            assert_eq!(name(0), "all F32");
            assert_eq!(name(1), "F16");
            assert_eq!(name(10), "Q2_K - Medium");
            assert_eq!(name(21), "Q2_K - Small");
            assert_eq!(name(32), "BF16");
            assert_eq!(name(38), "MXFP4 MoE");
            assert_eq!(name(39), "NVFP4");
            assert_eq!(name(40), "Q1_0");
            assert_eq!(name(999), "unknown, may not work");
        }
    }

    #[test]
    fn unescapes_rwkv_tokens_like_cpp_parser() {
        assert_eq!(unescape_rwkv_token(b"plain"), b"plain");
        assert_eq!(unescape_rwkv_token(br"a\tb\nc\rd"), b"a\tb\nc\rd");
        assert_eq!(unescape_rwkv_token(br"\x41\xff"), vec![0x41, 0xff]);
        assert_eq!(unescape_rwkv_token(br"\q"), b"q");
        assert_eq!(unescape_rwkv_token(br"trailing\"), b"trailing");
        assert_eq!(unescape_rwkv_token(br"\x4"), Vec::<u8>::new());
    }

    #[test]
    fn unescape_rwkv_token_ffi_allocates_and_frees() {
        unsafe {
            let buffer = llama_unescape_rwkv_token_rust(br"a\x20b".as_ptr(), 6);
            assert!(!buffer.data.is_null());
            assert_eq!(core::slice::from_raw_parts(buffer.data, buffer.len), b"a b");
            llama_byte_buffer_free_rust(buffer);

            let buffer = llama_unescape_rwkv_token_rust(core::ptr::null(), 1);
            assert!(buffer.data.is_null());
            assert_eq!(buffer.len, 0);
        }
    }

    #[test]
    fn checks_imatrix_requirement_rules() {
        unsafe {
            assert!(!llama_tensor_requires_imatrix_rust(
                b"token_embd.weight\0".as_ptr(),
                18,
                0,
            ));
            assert!(!llama_tensor_requires_imatrix_rust(
                b"output.weight\0".as_ptr(),
                16,
                0,
            ));
            assert!(llama_tensor_requires_imatrix_rust(
                b"blk.0.attn_v.weight\0".as_ptr(),
                18,
                0,
            ));
            assert!(llama_tensor_requires_imatrix_rust(
                b"blk.0.attn_v.weight\0".as_ptr(),
                29,
                0,
            ));
            assert!(!llama_tensor_requires_imatrix_rust(
                b"blk.0.attn_v.weight\0".as_ptr(),
                10,
                10,
            ));
            assert!(llama_tensor_requires_imatrix_rust(
                b"blk.0.attn_v.weight\0".as_ptr(),
                10,
                21,
            ));
            assert!(!llama_tensor_requires_imatrix_rust(
                b"blk.0.attn_v.weight\0".as_ptr(),
                0,
                21,
            ));
            assert!(!llama_tensor_requires_imatrix_rust(
                core::ptr::null(),
                18,
                0
            ));
        }
    }

    #[test]
    fn categorizes_tensor_names_for_quantization() {
        unsafe {
            assert_eq!(
                llama_tensor_get_category_rust(b"token_embd.weight\0".as_ptr()),
                0
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"per_layer_token_embd.weight\0".as_ptr()),
                0
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.attn_q.weight\0".as_ptr()),
                1
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.attn_v.weight\0".as_ptr()),
                2
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.attn_k.weight\0".as_ptr()),
                3
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.attn_qkv.weight\0".as_ptr()),
                4
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.attn_kv_b.weight\0".as_ptr()),
                5
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.attn_output.weight\0".as_ptr()),
                6
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.ffn_up.weight\0".as_ptr()),
                7
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.ffn_gate.weight\0".as_ptr()),
                8
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.ffn_down.weight\0".as_ptr()),
                9
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"output.weight\0".as_ptr()),
                10
            );
            assert_eq!(
                llama_tensor_get_category_rust(b"blk.0.norm.weight\0".as_ptr()),
                11
            );
            assert_eq!(llama_tensor_get_category_rust(core::ptr::null()), 11);
        }
    }

    #[test]
    fn checks_kq_mask_reuse_shape() {
        assert!(llama_can_reuse_kq_mask_rust(128, 16, 1, 2, 128, 32, 2));
        assert!(llama_can_reuse_kq_mask_rust(128, 32, 1, 1, 128, 32, 1));
        assert!(!llama_can_reuse_kq_mask_rust(127, 16, 1, 2, 128, 32, 2));
        assert!(!llama_can_reuse_kq_mask_rust(128, 15, 1, 2, 128, 32, 2));
        assert!(!llama_can_reuse_kq_mask_rust(128, 16, 2, 2, 128, 32, 2));
        assert!(!llama_can_reuse_kq_mask_rust(128, 16, 1, 3, 128, 32, 2));
        assert!(!llama_can_reuse_kq_mask_rust(128, 16, 1, 2, 128, 32, 0));
    }

    #[test]
    fn maps_ftype_to_default_ggml_type() {
        assert_eq!(llama_ftype_get_default_type_rust(0), 0);
        assert_eq!(llama_ftype_get_default_type_rust(1), 1);
        assert_eq!(llama_ftype_get_default_type_rust(2), 2);
        assert_eq!(llama_ftype_get_default_type_rust(7), 8);
        assert_eq!(llama_ftype_get_default_type_rust(21), 10);
        assert_eq!(llama_ftype_get_default_type_rust(22), 21);
        assert_eq!(llama_ftype_get_default_type_rust(28), 17);
        assert_eq!(llama_ftype_get_default_type_rust(29), 22);
        assert_eq!(llama_ftype_get_default_type_rust(32), 30);
        assert_eq!(llama_ftype_get_default_type_rust(36), 34);
        assert_eq!(llama_ftype_get_default_type_rust(38), 39);
        assert_eq!(llama_ftype_get_default_type_rust(40), 41);
        assert_eq!(llama_ftype_get_default_type_rust(999), 42);
    }

    #[test]
    fn formats_gguf_scalar_data_like_cpp_to_string() {
        unsafe {
            let as_string = |buffer: LlamaByteBuffer| {
                let text = String::from_utf8(
                    core::slice::from_raw_parts(buffer.data, buffer.len).to_vec(),
                )
                .unwrap();
                llama_byte_buffer_free_rust(buffer);
                text
            };

            let u8_values = [7u8, 9];
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(0, u8_values.as_ptr(), 1)),
                "9"
            );
            let i8_values = [-3i8];
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(1, i8_values.as_ptr().cast(), 0)),
                "-3"
            );
            let u16_values = [65535u16];
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(
                    2,
                    u16_values.as_ptr().cast(),
                    0
                )),
                "65535"
            );
            let i32_values = [-12345i32];
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(
                    5,
                    i32_values.as_ptr().cast(),
                    0
                )),
                "-12345"
            );
            let f32_values = [1.25f32];
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(
                    6,
                    f32_values.as_ptr().cast(),
                    0
                )),
                "1.250000"
            );
            let bool_values = [0i8, 1];
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(
                    7,
                    bool_values.as_ptr().cast(),
                    0
                )),
                "false"
            );
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(
                    7,
                    bool_values.as_ptr().cast(),
                    1
                )),
                "true"
            );
            let u64_values = [18_446_744_073_709_551_615u64];
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(
                    10,
                    u64_values.as_ptr().cast(),
                    0
                )),
                "18446744073709551615"
            );
            let f64_values = [-2.5f64];
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(
                    12,
                    f64_values.as_ptr().cast(),
                    0
                )),
                "-2.500000"
            );
            assert_eq!(
                as_string(llama_gguf_data_to_str_rust(99, core::ptr::null(), 0)),
                "unknown type 99"
            );
        }
    }

    #[test]
    fn formats_tensor_shapes_like_cpp_snprintf_width() {
        assert_eq!(format_tensor_shape(&[1]), "     1");
        assert_eq!(
            format_tensor_shape(&[1, 22, 333, 4444]),
            "     1,     22,    333,   4444"
        );
        assert_eq!(format_tensor_shape(&[-1, 0]), "    -1,      0");

        unsafe {
            let dims = [1i64, 2, 3, 4];
            let buffer = llama_format_tensor_shape_rust(dims.as_ptr(), dims.len());
            let text =
                String::from_utf8(core::slice::from_raw_parts(buffer.data, buffer.len).to_vec())
                    .unwrap();
            llama_byte_buffer_free_rust(buffer);
            assert_eq!(text, "     1,      2,      3,      4");
        }
    }
}
