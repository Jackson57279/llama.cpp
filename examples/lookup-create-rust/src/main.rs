use llama_lookup_create_rust::{build_static_ngram_cache, parse_args, Args};
use llama_lookup_merge::save_cache;
use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
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
            true,
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
            true,
        )
    };
    if check < 0 {
        return Err("failed to tokenize prompt".to_string());
    }
    tokens.truncate(check as usize);
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

    let tokens = tokenize(vocab, &args.prompt)?;
    eprintln!("tokenization done");
    let cache = build_static_ngram_cache(&tokens);
    eprintln!("hashing done, writing file to {}", args.cache_path);
    save_cache(&cache, &args.cache_path).map_err(|err| err.to_string())?;
    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv
        .next()
        .unwrap_or_else(|| "llama-lookup-create".to_string());
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{program}: error: {err}");
            eprintln!("usage: {program} -m model.gguf -p PROMPT -lcs cache.bin");
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
