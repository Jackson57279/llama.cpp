#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_float, c_int, c_void, CString};
use std::io::Read;
use std::process::ExitCode;

use llama_tokenize_rust::{parse_args, process_escapes, usage, ParseResult, TokenizeParams};

type llama_token = i32;
type ggml_log_level = c_int;
type ggml_type = c_int;

#[repr(C)]
struct ggml_tensor {
    _private: [u8; 0],
}

#[repr(C)]
struct llama_model {
    _private: [u8; 0],
}

#[repr(C)]
struct llama_context {
    _private: [u8; 0],
}

#[repr(C)]
struct llama_vocab {
    _private: [u8; 0],
}

type ggml_backend_dev_t = *mut c_void;
type ggml_backend_buffer_type_t = *mut c_void;

#[repr(C)]
struct llama_model_tensor_buft_override {
    pattern: *const c_char,
    buft: ggml_backend_buffer_type_t,
}

#[repr(C)]
struct llama_model_kv_override {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_model_params {
    devices: *mut ggml_backend_dev_t,
    tensor_buft_overrides: *const llama_model_tensor_buft_override,
    n_gpu_layers: i32,
    split_mode: c_int,
    main_gpu: i32,
    tensor_split: *const c_float,
    progress_callback: Option<unsafe extern "C" fn(c_float, *mut c_void) -> bool>,
    progress_callback_user_data: *mut c_void,
    kv_overrides: *const llama_model_kv_override,
    vocab_only: bool,
    use_mmap: bool,
    use_direct_io: bool,
    use_mlock: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_sampler_seq_config {
    seq_id: i32,
    sampler: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct llama_context_params {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_threads: i32,
    n_threads_batch: i32,
    rope_scaling_type: c_int,
    pooling_type: c_int,
    attention_type: c_int,
    flash_attn_type: c_int,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: Option<unsafe extern "C" fn(*mut ggml_tensor, bool, *mut c_void) -> bool>,
    cb_eval_user_data: *mut c_void,
    type_k: ggml_type,
    type_v: ggml_type,
    abort_callback: Option<unsafe extern "C" fn(*mut c_void) -> bool>,
    abort_callback_data: *mut c_void,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: *mut llama_sampler_seq_config,
    n_samplers: usize,
}

extern "C" {
    fn llama_log_set(
        log_callback: Option<unsafe extern "C" fn(ggml_log_level, *const c_char, *mut c_void)>,
        user_data: *mut c_void,
    );
    fn llama_backend_init();
    fn llama_backend_free();
    fn llama_model_default_params() -> llama_model_params;
    fn llama_context_default_params() -> llama_context_params;
    fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    fn llama_model_free(model: *mut llama_model);
    fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    fn llama_free(ctx: *mut llama_context);
    fn llama_vocab_get_add_bos(vocab: *const llama_vocab) -> bool;
    fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
    fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;
}

unsafe extern "C" fn llama_log_callback_null(
    _level: ggml_log_level,
    _text: *const c_char,
    _user_data: *mut c_void,
) {
}

fn main() -> ExitCode {
    let argv = std::env::args().collect::<Vec<_>>();
    match run(argv) {
        Ok(()) => ExitCode::SUCCESS,
        Err(AppError::Usage(argv0)) => {
            print!("{}", usage(&argv0));
            ExitCode::FAILURE
        }
        Err(AppError::Message(message)) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

fn run(argv: Vec<String>) -> Result<(), AppError> {
    let argv0 = argv
        .first()
        .cloned()
        .unwrap_or_else(|| "llama-tokenize".to_string());
    let params = match parse_args(&argv) {
        Ok(ParseResult::Help) => {
            print!("{}", usage(&argv0));
            return Ok(());
        }
        Ok(ParseResult::Params(params)) => params,
        Err(err) if err == "usage" => return Err(AppError::Usage(argv0)),
        Err(err) => return Err(AppError::Message(err)),
    };

    tokenize(params)
}

fn tokenize(params: TokenizeParams) -> Result<(), AppError> {
    let mut prompt = if let Some(path) = &params.prompt_path {
        std::fs::read_to_string(path).map_err(|err| {
            AppError::Message(format!(
                "read_prompt_from_file: could not read the entire file '{path}': {err}"
            ))
        })?
    } else if let Some(prompt) = &params.prompt_arg {
        prompt.clone()
    } else {
        String::new()
    };

    unsafe {
        if params.disable_logging {
            llama_log_set(Some(llama_log_callback_null), std::ptr::null_mut());
        }

        llama_backend_init();
        let mut model_params = llama_model_default_params();
        model_params.vocab_only = true;
        let c_model_path = CString::new(params.model_path.clone())
            .map_err(|err| AppError::Message(format!("invalid model path: {err}")))?;
        let model = llama_model_load_from_file(c_model_path.as_ptr(), model_params);
        if model.is_null() {
            return Err(AppError::Message(format!(
                "Error: could not load model from file '{}'.",
                params.model_path
            )));
        }

        let vocab = llama_model_get_vocab(model);
        let ctx = llama_init_from_model(model, llama_context_default_params());
        if ctx.is_null() {
            llama_model_free(model);
            return Err(AppError::Message(
                "Error: could not create context.".to_string(),
            ));
        }

        if params.stdin_set {
            std::io::stdin().read_to_string(&mut prompt).map_err(|_| {
                AppError::Message("Error: could not read the entire standard input.".to_string())
            })?;
        }

        let add_bos = llama_vocab_get_add_bos(vocab) && !params.no_bos;
        let parse_special = !params.no_parse_special;
        let prompt_bytes = if params.no_escape {
            prompt.into_bytes()
        } else {
            process_escapes(&prompt)
        };

        let tokens = common_tokenize(vocab, &prompt_bytes, add_bos, parse_special)?;
        if params.printing_ids {
            print!("[");
        }

        for (i, token) in tokens.iter().enumerate() {
            if params.printing_ids {
                if i > 0 {
                    print!(", ");
                }
                print!("{token}");
            } else {
                let piece = common_token_to_piece(vocab, *token, true)?;
                print!("{token:6} -> '");
                print!("{}", String::from_utf8_lossy(&piece));
                println!("'");
            }
        }

        if params.printing_ids {
            println!("]");
        }
        if params.show_token_count {
            println!("Total number of tokens: {}", tokens.len());
        }

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    }

    Ok(())
}

unsafe fn common_tokenize(
    vocab: *const llama_vocab,
    text: &[u8],
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<llama_token>, AppError> {
    let mut tokens = vec![0; text.len() + 2 * add_special as usize];
    let mut n_tokens = llama_tokenize(
        vocab,
        text.as_ptr().cast(),
        text.len() as i32,
        tokens.as_mut_ptr(),
        tokens.len() as i32,
        add_special,
        parse_special,
    );
    if n_tokens == i32::MIN {
        return Err(AppError::Message(
            "Tokenization failed: input text too large, tokenization result exceeds int32_t limit"
                .to_string(),
        ));
    }
    if n_tokens < 0 {
        tokens.resize((-n_tokens) as usize, 0);
        n_tokens = llama_tokenize(
            vocab,
            text.as_ptr().cast(),
            text.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            add_special,
            parse_special,
        );
    }
    tokens.truncate(n_tokens as usize);
    Ok(tokens)
}

unsafe fn common_token_to_piece(
    vocab: *const llama_vocab,
    token: llama_token,
    special: bool,
) -> Result<Vec<u8>, AppError> {
    let mut piece = vec![0_i8; 32];
    let mut n_chars = llama_token_to_piece(
        vocab,
        token,
        piece.as_mut_ptr(),
        piece.len() as i32,
        0,
        special,
    );
    if n_chars < 0 {
        piece.resize((-n_chars) as usize, 0);
        n_chars = llama_token_to_piece(
            vocab,
            token,
            piece.as_mut_ptr(),
            piece.len() as i32,
            0,
            special,
        );
    }
    if n_chars < 0 {
        return Err(AppError::Message("llama_token_to_piece failed".to_string()));
    }
    piece.truncate(n_chars as usize);
    Ok(piece.into_iter().map(|ch| ch as u8).collect())
}

enum AppError {
    Usage(String),
    Message(String),
}
