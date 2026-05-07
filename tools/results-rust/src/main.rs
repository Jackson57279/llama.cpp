use llama_results_rust::{nmse, parse_args, Args};
use llama_simple_rust::ffi;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::{env, ptr, slice};

const GGML_TYPE_F32: ffi::ggml_type = 0;
const GGML_TYPE_I32: ffi::ggml_type = 26;

struct Model(*mut ffi::llama_model);

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::llama_model_free(self.0);
            }
        }
    }
}

struct Context(*mut ffi::llama_context);

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::llama_free(self.0);
            }
        }
    }
}

struct Batch(ffi::llama_batch);

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_batch_free(self.0);
        }
    }
}

struct GgmlContext(*mut ffi::ggml_context);

impl Drop for GgmlContext {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::ggml_free(self.0);
            }
        }
    }
}

struct GgufContext(*mut ffi::gguf_context);

impl Drop for GgufContext {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::gguf_free(self.0);
            }
        }
    }
}

fn tokenize(vocab: *const ffi::llama_vocab, prompt: &str) -> Result<Vec<ffi::llama_token>, String> {
    let prompt_c =
        CString::new(prompt).map_err(|_| "prompt contains an interior NUL byte".to_string())?;
    let n = unsafe {
        -ffi::llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            prompt.len() as i32,
            ptr::null_mut(),
            0,
            true,
            false,
        )
    };
    if n <= 0 {
        return Err("failed to size prompt tokenization".to_string());
    }
    let mut tokens = vec![0_i32; n as usize];
    let check = unsafe {
        ffi::llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            prompt.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            true,
            false,
        )
    };
    if check < 0 {
        return Err("failed to tokenize prompt".to_string());
    }
    tokens.truncate(check as usize);
    Ok(tokens)
}

fn batch_add(
    batch: &mut ffi::llama_batch,
    token: ffi::llama_token,
    pos: ffi::llama_pos,
    logits: bool,
) {
    let idx = batch.n_tokens as isize;
    unsafe {
        *batch.token.offset(idx) = token;
        *batch.pos.offset(idx) = pos;
        *batch.n_seq_id.offset(idx) = 1;
        **batch.seq_id.offset(idx) = 0;
        *batch.logits.offset(idx) = i8::from(logits);
    }
    batch.n_tokens += 1;
}

fn load_model_and_context(args: &Args) -> Result<(Model, Context, Vec<ffi::llama_token>), String> {
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;
    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;
    let model =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model.0.is_null() {
        return Err("unable to load model".to_string());
    }

    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        return Err("failed to get model vocabulary".to_string());
    }
    let tokens = tokenize(vocab, &args.prompt)?;
    if tokens.is_empty() {
        return Err("prompt produced no tokens".to_string());
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = tokens.len().max(1) as u32;
    ctx_params.n_batch = tokens.len() as u32;
    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create context".to_string());
    }

    Ok((model, ctx, tokens))
}

fn get_logits(
    model: *mut ffi::llama_model,
    ctx: *mut ffi::llama_context,
    tokens: &[ffi::llama_token],
) -> Result<Vec<f32>, String> {
    let vocab = unsafe { ffi::llama_model_get_vocab(model) };
    let n_vocab = unsafe { ffi::llama_vocab_n_tokens(vocab) } as usize;
    let mut batch = Batch(unsafe { ffi::llama_batch_init(tokens.len() as i32, 0, 1) });
    for (pos, &token) in tokens.iter().enumerate() {
        batch_add(&mut batch.0, token, pos as i32, true);
    }
    if unsafe { ffi::llama_decode(ctx, batch.0) } != 0 {
        return Err("failed to decode batch".to_string());
    }

    let mut logits = Vec::with_capacity(tokens.len() * n_vocab);
    for i in 0..tokens.len() {
        let ptr = unsafe { ffi::llama_get_logits_ith(ctx, i as i32) };
        if ptr.is_null() {
            return Err("failed to get logits".to_string());
        }
        logits.extend_from_slice(unsafe { slice::from_raw_parts(ptr, n_vocab) });
    }
    Ok(logits)
}

fn read_tensor_data(
    path: &str,
    gguf: *mut ffi::gguf_context,
    name: &str,
) -> Result<Vec<u8>, String> {
    let name_c = CString::new(name).unwrap();
    let tid = unsafe { ffi::gguf_find_tensor(gguf, name_c.as_ptr()) };
    if tid < 0 {
        return Err(format!("missing tensor '{name}'"));
    }
    let size = unsafe { ffi::gguf_get_tensor_size(gguf, tid) };
    let offset =
        unsafe { ffi::gguf_get_data_offset(gguf) + ffi::gguf_get_tensor_offset(gguf, tid) };
    let mut file = File::open(path).map_err(|err| err.to_string())?;
    file.seek(SeekFrom::Start(offset as u64))
        .map_err(|err| err.to_string())?;
    let mut data = vec![0_u8; size];
    file.read_exact(&mut data).map_err(|err| err.to_string())?;
    Ok(data)
}

fn check_results(args: &Args, tokens: &[ffi::llama_token], logits: &[f32]) -> Result<i32, String> {
    let path_c = CString::new(args.output_path.as_str())
        .map_err(|_| "output path contains an interior NUL byte".to_string())?;
    let gguf = GgufContext(unsafe {
        ffi::gguf_init_from_file(
            path_c.as_ptr(),
            ffi::gguf_init_params {
                no_alloc: true,
                ctx: ptr::null_mut(),
            },
        )
    });
    if gguf.0.is_null() {
        return Err("failed to open results file".to_string());
    }

    let path_key = CString::new("path_model").unwrap();
    let key_id = unsafe { ffi::gguf_find_key(gguf.0, path_key.as_ptr()) };
    if key_id < 0 {
        return Err("missing path_model key".to_string());
    }
    let path_model_disk =
        unsafe { CStr::from_ptr(ffi::gguf_get_val_str(gguf.0, key_id)) }.to_string_lossy();
    if path_model_disk != args.model_path {
        return Err(format!(
            "model path mismatch: file has '{path_model_disk}', current is '{}'",
            args.model_path
        ));
    }

    let token_data = read_tensor_data(&args.output_path, gguf.0, "tokens")?;
    let tokens_disk = token_data
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    if tokens_disk != tokens {
        return Err("token tensor mismatch".to_string());
    }

    let logits_data = read_tensor_data(&args.output_path, gguf.0, "logits")?;
    let logits_disk = logits_data
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    if logits_disk.len() != logits.len() {
        return Err("logits tensor size mismatch".to_string());
    }
    let nmse_val = nmse(&logits_disk, logits);
    eprintln!("NMSE={nmse_val:.3e}");
    if nmse_val > 1e-6 {
        println!("\x1b[1;31mFAIL\x1b[0m");
        Ok(1)
    } else {
        println!("\x1b[1;32mOK\x1b[0m");
        Ok(0)
    }
}

fn write_results(args: &Args, tokens: &[ffi::llama_token], logits: &[f32]) -> Result<(), String> {
    let mem_size = tokens.len() * std::mem::size_of::<ffi::llama_token>()
        + logits.len() * std::mem::size_of::<f32>()
        + unsafe { ffi::ggml_tensor_overhead() } * 2;
    let ggml = GgmlContext(unsafe {
        ffi::ggml_init(ffi::ggml_init_params {
            mem_size,
            mem_buffer: ptr::null_mut(),
            no_alloc: false,
        })
    });
    if ggml.0.is_null() {
        return Err("failed to create ggml context".to_string());
    }
    let gguf = GgufContext(unsafe { ffi::gguf_init_empty() });
    if gguf.0.is_null() {
        return Err("failed to create gguf context".to_string());
    }

    let key = CString::new("path_model").unwrap();
    let value = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;
    unsafe {
        ffi::gguf_set_val_str(gguf.0, key.as_ptr(), value.as_ptr());
    }

    let tokens_name = CString::new("tokens").unwrap();
    let t_tokens = unsafe { ffi::ggml_new_tensor_1d(ggml.0, GGML_TYPE_I32, tokens.len() as i64) };
    if t_tokens.is_null() {
        return Err("failed to create tokens tensor".to_string());
    }
    unsafe {
        ffi::ggml_set_name(t_tokens, tokens_name.as_ptr());
        let dst = ffi::ggml_get_data(t_tokens) as *mut ffi::llama_token;
        ptr::copy_nonoverlapping(tokens.as_ptr(), dst, tokens.len());
        ffi::gguf_add_tensor(gguf.0, t_tokens);
    }

    let logits_name = CString::new("logits").unwrap();
    let n_vocab = logits.len() / tokens.len();
    let t_logits = unsafe {
        ffi::ggml_new_tensor_2d(ggml.0, GGML_TYPE_F32, tokens.len() as i64, n_vocab as i64)
    };
    if t_logits.is_null() {
        return Err("failed to create logits tensor".to_string());
    }
    unsafe {
        ffi::ggml_set_name(t_logits, logits_name.as_ptr());
        let dst = ffi::ggml_get_data_f32(t_logits);
        ptr::copy_nonoverlapping(logits.as_ptr(), dst, logits.len());
        ffi::gguf_add_tensor(gguf.0, t_logits);
    }

    let output_c = CString::new(args.output_path.as_str())
        .map_err(|_| "output path contains an interior NUL byte".to_string())?;
    if !unsafe { ffi::gguf_write_to_file(gguf.0, output_c.as_ptr(), false) } {
        return Err("failed to write results file".to_string());
    }
    eprintln!("writing results to {}...", args.output_path);
    Ok(())
}

fn run(args: Args) -> Result<i32, String> {
    unsafe {
        ffi::llama_backend_init();
        ffi::llama_numa_init(0);
    }

    let (model, ctx, tokens) = load_model_and_context(&args)?;
    let logits = get_logits(model.0, ctx.0, &tokens)?;

    if args.check {
        check_results(&args, &tokens, &logits)
    } else {
        write_results(&args, &tokens, &logits)?;
        Ok(0)
    }
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-results".to_string());
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{program}: error: {err}");
            eprintln!("usage: {program} --model model.gguf --output results.gguf --prompt PROMPT [--check]");
            std::process::exit(1);
        }
    };

    let status = match run(args) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("{program}: error: {err}");
            1
        }
    };
    unsafe {
        ffi::llama_backend_free();
    }
    std::process::exit(status);
}
