#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{c_char, c_float, c_int, c_void};

pub const LLAMA_TOKEN_NULL: llama_token = -1;
pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFF_FFFF;

pub type llama_pos = i32;
pub type llama_token = i32;
pub type llama_seq_id = i32;
pub type ggml_backend_dev_t = *mut c_void;
pub type ggml_backend_reg_t = *mut c_void;
pub type ggml_backend_buffer_type_t = *mut c_void;
pub type ggml_type = c_int;
pub type llama_memory_t = *mut c_void;
pub type ggml_log_level = c_int;
pub type llama_pooling_type = c_int;
pub type llama_vocab_type = c_int;
pub type ggml_opt_dataset_t = *mut c_void;
pub type ggml_opt_result_t = *mut c_void;
pub type ggml_opt_context_t = *mut c_void;
pub type ggml_opt_optimizer_type = c_int;

#[repr(C)]
pub struct llama_vocab {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_tensor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ggml_context {
    _private: [u8; 0],
}

#[repr(C)]
pub struct gguf_context {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ggml_opt_adamw_params {
    pub alpha: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub wd: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ggml_opt_sgd_params {
    pub alpha: f32,
    pub wd: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ggml_opt_optimizer_params {
    pub adamw: ggml_opt_adamw_params,
    pub sgd: ggml_opt_sgd_params,
}

pub type ggml_opt_get_optimizer_params =
    Option<unsafe extern "C" fn(userdata: *mut c_void) -> ggml_opt_optimizer_params>;
pub type llama_opt_param_filter =
    Option<unsafe extern "C" fn(tensor: *const ggml_tensor, userdata: *mut c_void) -> bool>;
pub type ggml_opt_epoch_callback = Option<
    unsafe extern "C" fn(
        opt_ctx: ggml_opt_context_t,
        dataset: ggml_opt_dataset_t,
        result: ggml_opt_result_t,
        ibatch: i64,
        ibatch_max: i64,
        train: bool,
    ),
>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_opt_params {
    pub n_ctx_train: u32,
    pub param_filter: llama_opt_param_filter,
    pub param_filter_ud: *mut c_void,
    pub get_opt_pars: ggml_opt_get_optimizer_params,
    pub get_opt_pars_ud: *mut c_void,
    pub optimizer_type: ggml_opt_optimizer_type,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ggml_init_params {
    pub mem_size: usize,
    pub mem_buffer: *mut c_void,
    pub no_alloc: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct gguf_init_params {
    pub no_alloc: bool,
    pub ctx: *mut *mut ggml_context,
}

#[repr(C)]
pub struct llama_model_tensor_buft_override {
    pub pattern: *const c_char,
    pub buft: ggml_backend_buffer_type_t,
}

#[repr(C)]
pub struct llama_model_kv_override {
    _private: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler_seq_config {
    pub seq_id: llama_seq_id,
    pub sampler: *mut llama_sampler,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_token_data {
    pub id: llama_token,
    pub logit: f32,
    pub p: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_token_data_array {
    pub data: *mut llama_token_data,
    pub size: usize,
    pub selected: i64,
    pub sorted: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_model_params {
    pub devices: *mut ggml_backend_dev_t,
    pub tensor_buft_overrides: *const llama_model_tensor_buft_override,
    pub n_gpu_layers: i32,
    pub split_mode: c_int,
    pub main_gpu: i32,
    pub tensor_split: *const c_float,
    pub progress_callback: Option<unsafe extern "C" fn(c_float, *mut c_void) -> bool>,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const llama_model_kv_override,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: Option<unsafe extern "C" fn(*mut ggml_tensor, bool, *mut c_void) -> bool>,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: ggml_type,
    pub type_v: ggml_type,
    pub abort_callback: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut llama_sampler_seq_config,
    pub n_samplers: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut f32,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct llama_chat_message {
    pub role: *const c_char,
    pub content: *const c_char,
}

extern "C" {
    pub fn ggml_backend_load_all();
    pub fn ggml_backend_dev_count() -> usize;
    pub fn ggml_backend_dev_get(index: usize) -> ggml_backend_dev_t;
    pub fn ggml_backend_dev_type(dev: ggml_backend_dev_t) -> c_int;
    pub fn ggml_backend_dev_by_name(name: *const c_char) -> ggml_backend_dev_t;
    pub fn ggml_backend_dev_by_type(typ: c_int) -> ggml_backend_dev_t;
    pub fn ggml_backend_dev_name(dev: ggml_backend_dev_t) -> *const c_char;
    pub fn ggml_backend_dev_description(dev: ggml_backend_dev_t) -> *const c_char;
    pub fn ggml_backend_dev_memory(dev: ggml_backend_dev_t, free: *mut usize, total: *mut usize);
    pub fn ggml_backend_reg_by_name(name: *const c_char) -> ggml_backend_reg_t;
    pub fn ggml_backend_reg_get_proc_address(
        reg: ggml_backend_reg_t,
        name: *const c_char,
    ) -> *mut c_void;
    pub fn ggml_tensor_overhead() -> usize;
    pub fn ggml_init(params: ggml_init_params) -> *mut ggml_context;
    pub fn ggml_free(ctx: *mut ggml_context);
    pub fn ggml_new_tensor_1d(ctx: *mut ggml_context, typ: ggml_type, ne0: i64)
        -> *mut ggml_tensor;
    pub fn ggml_new_tensor_2d(
        ctx: *mut ggml_context,
        typ: ggml_type,
        ne0: i64,
        ne1: i64,
    ) -> *mut ggml_tensor;
    pub fn ggml_get_data(tensor: *const ggml_tensor) -> *mut c_void;
    pub fn ggml_get_data_f32(tensor: *const ggml_tensor) -> *mut f32;
    pub fn ggml_set_name(tensor: *mut ggml_tensor, name: *const c_char) -> *mut ggml_tensor;
    pub fn ggml_time_us() -> i64;
    pub fn llama_log_set(
        log_callback: Option<unsafe extern "C" fn(ggml_log_level, *const c_char, *mut c_void)>,
        user_data: *mut c_void,
    );

    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;
    pub fn llama_backend_init();
    pub fn llama_backend_free();
    pub fn llama_numa_init(numa: c_int);
    pub fn llama_time_us() -> i64;
    pub fn llama_max_parallel_sequences() -> usize;

    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);
    pub fn llama_model_save_to_file(model: *const llama_model, path_model: *const c_char);
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_get_model(ctx: *const llama_context) -> *const llama_model;
    pub fn llama_get_memory(ctx: *const llama_context) -> llama_memory_t;
    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;
    pub fn llama_n_seq_max(ctx: *const llama_context) -> u32;
    pub fn llama_n_ubatch(ctx: *const llama_context) -> u32;
    pub fn llama_set_n_threads(
        ctx: *mut llama_context,
        n_threads: i32,
        n_threads_batch: i32,
    );
    pub fn llama_pooling_type(ctx: *const llama_context) -> llama_pooling_type;

    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_model_n_embd_out(model: *const llama_model) -> i32;
    pub fn llama_model_n_ctx_train(model: *const llama_model) -> i32;
    pub fn llama_model_chat_template(
        model: *const llama_model,
        name: *const c_char,
    ) -> *const c_char;
    pub fn llama_model_has_encoder(model: *const llama_model) -> bool;
    pub fn llama_model_has_decoder(model: *const llama_model) -> bool;
    pub fn llama_model_is_diffusion(model: *const llama_model) -> bool;
    pub fn llama_model_meta_val_str(
        model: *const llama_model,
        key: *const c_char,
        buf: *mut c_char,
        buf_size: usize,
    ) -> i32;
    pub fn llama_model_n_cls_out(model: *const llama_model) -> u32;
    pub fn llama_model_cls_label(model: *const llama_model, i: u32) -> *const c_char;
    pub fn llama_model_decoder_start_token(model: *const llama_model) -> llama_token;

    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eot(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_mask(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_sep(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_get_text(vocab: *const llama_vocab, token: llama_token) -> *const c_char;
    pub fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> bool;
    pub fn llama_vocab_get_add_eos(vocab: *const llama_vocab) -> bool;
    pub fn llama_vocab_get_add_sep(vocab: *const llama_vocab) -> bool;
    pub fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;
    pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;
    pub fn llama_vocab_type(vocab: *const llama_vocab) -> llama_vocab_type;
    pub fn llama_memory_seq_pos_max(mem: llama_memory_t, seq_id: llama_seq_id) -> llama_pos;
    pub fn llama_memory_clear(mem: llama_memory_t, data: bool);
    pub fn llama_memory_seq_rm(
        mem: llama_memory_t,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> bool;
    pub fn llama_memory_seq_cp(
        mem: llama_memory_t,
        seq_id_src: llama_seq_id,
        seq_id_dst: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    );
    pub fn llama_memory_seq_keep(mem: llama_memory_t, seq_id: llama_seq_id);
    pub fn llama_memory_seq_add(
        mem: llama_memory_t,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
    );
    pub fn llama_memory_seq_div(
        mem: llama_memory_t,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        d: i32,
    );

    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
    pub fn llama_detokenize(
        vocab: *const llama_vocab,
        tokens: *const llama_token,
        n_tokens: i32,
        text: *mut c_char,
        text_len_max: i32,
        remove_special: bool,
        unparse_special: bool,
    ) -> i32;
    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    pub fn llama_batch_get_one(tokens: *mut llama_token, n_tokens: i32) -> llama_batch;
    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
    pub fn llama_batch_free(batch: llama_batch);
    pub fn llama_encode(ctx: *mut llama_context, batch: llama_batch) -> i32;
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;
    pub fn llama_synchronize(ctx: *mut llama_context);
    pub fn llama_set_causal_attn(ctx: *mut llama_context, causal_attn: bool);
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *const f32;
    pub fn llama_get_embeddings(ctx: *mut llama_context) -> *mut f32;
    pub fn llama_get_embeddings_ith(ctx: *mut llama_context, i: i32) -> *mut f32;
    pub fn llama_get_embeddings_seq(ctx: *mut llama_context, seq_id: llama_seq_id) -> *mut f32;
    pub fn llama_state_seq_get_size(ctx: *mut llama_context, seq_id: llama_seq_id) -> usize;
    pub fn llama_state_seq_get_data(
        ctx: *mut llama_context,
        dst: *mut u8,
        size: usize,
        seq_id: llama_seq_id,
    ) -> usize;
    pub fn llama_state_seq_set_data(
        ctx: *mut llama_context,
        src: *const u8,
        size: usize,
        dest_seq_id: llama_seq_id,
    ) -> usize;
    pub fn llama_state_load_file(
        ctx: *mut llama_context,
        path_session: *const c_char,
        tokens_out: *mut llama_token,
        n_token_capacity: usize,
        n_token_count_out: *mut usize,
    ) -> bool;
    pub fn llama_state_save_file(
        ctx: *mut llama_context,
        path_session: *const c_char,
        tokens: *const llama_token,
        n_token_count: usize,
    ) -> bool;

    pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, sampler: *mut llama_sampler);
    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_min_p(p: f32, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_temp(t: f32) -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_apply(smpl: *mut llama_sampler, cur_p: *mut llama_token_data_array);
    pub fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;
    pub fn llama_sampler_accept(smpl: *mut llama_sampler, token: llama_token);
    pub fn llama_sampler_reset(smpl: *mut llama_sampler);
    pub fn llama_sampler_free(smpl: *mut llama_sampler);

    pub fn llama_perf_sampler_print(chain: *const llama_sampler);
    pub fn llama_perf_context_print(ctx: *const llama_context);

    pub fn ggml_opt_dataset_init(
        type_data: ggml_type,
        type_label: ggml_type,
        ne_datapoint: i64,
        ne_label: i64,
        ndata: i64,
        ndata_shard: i64,
    ) -> ggml_opt_dataset_t;
    pub fn ggml_opt_dataset_free(dataset: ggml_opt_dataset_t);
    pub fn ggml_opt_dataset_ndata(dataset: ggml_opt_dataset_t) -> i64;
    pub fn ggml_opt_dataset_data(dataset: ggml_opt_dataset_t) -> *mut ggml_tensor;
    pub fn ggml_opt_dataset_labels(dataset: ggml_opt_dataset_t) -> *mut ggml_tensor;
    pub fn ggml_opt_get_default_optimizer_params(
        userdata: *mut c_void,
    ) -> ggml_opt_optimizer_params;
    pub fn ggml_opt_optimizer_name(optimizer: ggml_opt_optimizer_type) -> *const c_char;
    pub fn ggml_opt_result_init() -> ggml_opt_result_t;
    pub fn ggml_opt_result_free(result: ggml_opt_result_t);
    pub fn ggml_opt_result_reset(result: ggml_opt_result_t);
    pub fn ggml_opt_epoch_callback_progress_bar(
        opt_ctx: ggml_opt_context_t,
        dataset: ggml_opt_dataset_t,
        result: ggml_opt_result_t,
        ibatch: i64,
        ibatch_max: i64,
        train: bool,
    );
    pub fn llama_opt_param_filter_all(tensor: *const ggml_tensor, userdata: *mut c_void) -> bool;
    pub fn llama_opt_init(
        lctx: *mut llama_context,
        model: *mut llama_model,
        lopt_params: llama_opt_params,
    );
    pub fn llama_opt_epoch(
        lctx: *mut llama_context,
        dataset: ggml_opt_dataset_t,
        result_train: ggml_opt_result_t,
        result_eval: ggml_opt_result_t,
        idata_split: i64,
        callback_train: ggml_opt_epoch_callback,
        callback_eval: ggml_opt_epoch_callback,
    );

    pub fn llama_chat_apply_template(
        tmpl: *const c_char,
        chat: *const llama_chat_message,
        n_msg: usize,
        add_ass: bool,
        buf: *mut c_char,
        length: i32,
    ) -> i32;

    pub fn gguf_init_empty() -> *mut gguf_context;
    pub fn gguf_init_from_file(fname: *const c_char, params: gguf_init_params)
        -> *mut gguf_context;
    pub fn gguf_free(ctx: *mut gguf_context);
    pub fn gguf_get_data_offset(ctx: *const gguf_context) -> usize;
    pub fn gguf_find_key(ctx: *const gguf_context, key: *const c_char) -> i64;
    pub fn gguf_get_val_u16(ctx: *const gguf_context, key_id: i64) -> u16;
    pub fn gguf_get_val_str(ctx: *const gguf_context, key_id: i64) -> *const c_char;
    pub fn gguf_get_n_tensors(ctx: *const gguf_context) -> i64;
    pub fn gguf_get_tensor_name(ctx: *const gguf_context, tensor_id: i64) -> *const c_char;
    pub fn gguf_find_tensor(ctx: *const gguf_context, name: *const c_char) -> i64;
    pub fn gguf_get_tensor_offset(ctx: *const gguf_context, tensor_id: i64) -> usize;
    pub fn gguf_get_tensor_size(ctx: *const gguf_context, tensor_id: i64) -> usize;
    pub fn gguf_set_kv(ctx: *mut gguf_context, src: *const gguf_context);
    pub fn gguf_set_val_u16(ctx: *mut gguf_context, key: *const c_char, val: u16);
    pub fn gguf_set_val_i32(ctx: *mut gguf_context, key: *const c_char, val: i32);
    pub fn gguf_set_val_str(ctx: *mut gguf_context, key: *const c_char, val: *const c_char);
    pub fn gguf_add_tensor(ctx: *mut gguf_context, tensor: *const ggml_tensor);
    pub fn gguf_get_meta_size(ctx: *const gguf_context) -> usize;
    pub fn gguf_get_meta_data(ctx: *const gguf_context, data: *mut c_void);
    pub fn gguf_write_to_file(
        ctx: *const gguf_context,
        fname: *const c_char,
        only_meta: bool,
    ) -> bool;

    pub fn ggml_get_tensor(ctx: *mut ggml_context, name: *const c_char) -> *mut ggml_tensor;
    pub fn ggml_nbytes(tensor: *const ggml_tensor) -> usize;

    pub fn llama_split_path(
        split_path: *mut c_char,
        maxlen: usize,
        path_prefix: *const c_char,
        split_no: i32,
        split_count: i32,
    ) -> i32;
    pub fn llama_split_prefix(
        split_prefix: *mut c_char,
        maxlen: usize,
        split_path: *const c_char,
        split_no: i32,
        split_count: i32,
    ) -> i32;
}
