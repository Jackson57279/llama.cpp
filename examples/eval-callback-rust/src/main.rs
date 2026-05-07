use llama_eval_callback_rust::{parse_args, Args};
use llama_simple_rust::ffi;
use std::env;
use std::ffi::{c_void, CString};
use std::ptr;

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

unsafe extern "C" fn eval_callback(
    _tensor: *mut ffi::ggml_tensor,
    _ask: bool,
    _user_data: *mut c_void,
) -> bool {
    true
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf --prompt hello -ngl 0");
    eprintln!();
}

fn tokenize(
    vocab: *const ffi::llama_vocab,
    prompt: &str,
    add_special: bool,
) -> Result<Vec<ffi::llama_token>, String> {
    let prompt_c =
        CString::new(prompt).map_err(|_| "prompt contains an interior NUL byte".to_string())?;
    let n_prompt = unsafe {
        -ffi::llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            prompt.len() as i32,
            ptr::null_mut(),
            0,
            add_special,
            true,
        )
    };

    if n_prompt <= 0 {
        return Err("failed to size prompt tokenization".to_string());
    }

    let mut tokens = vec![0_i32; n_prompt as usize];
    let n_tokenized = unsafe {
        ffi::llama_tokenize(
            vocab,
            prompt_c.as_ptr(),
            prompt.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            add_special,
            true,
        )
    };

    if n_tokenized < 0 {
        return Err("failed to tokenize the prompt".to_string());
    }

    tokens.truncate(n_tokenized as usize);
    Ok(tokens)
}

fn run(args: Args) -> Result<(), String> {
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;

    unsafe {
        ffi::llama_backend_init();
        ffi::llama_numa_init(0);
    }

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

    let add_bos = unsafe { ffi::llama_vocab_get_add_bos(vocab) };
    let mut tokens = tokenize(vocab, &args.prompt, add_bos)?;
    if tokens.is_empty() {
        return Err("there are no input tokens to process".to_string());
    }

    eprintln!("number of input tokens = {}", tokens.len());
    for token in &tokens {
        eprintln!("  {token}");
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = tokens.len().max(1) as u32;
    ctx_params.n_batch = tokens.len() as u32;
    ctx_params.cb_eval = Some(eval_callback);
    ctx_params.cb_eval_user_data = ptr::null_mut();
    ctx_params.no_perf = false;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create the llama_context".to_string());
    }

    let batch = unsafe { ffi::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
    if unsafe { ffi::llama_decode(ctx.0, batch) } != 0 {
        return Err("failed to eval".to_string());
    }

    eprintln!();
    unsafe {
        ffi::llama_perf_context_print(ctx.0);
    }

    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv
        .next()
        .unwrap_or_else(|| "llama-eval-callback".to_string());

    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{program}: error: {err}");
            print_usage(&program);
            std::process::exit(1);
        }
    };

    let status = match run(args) {
        Ok(()) => 0,
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
