use llama_simple_rust::ffi;
use llama_simple_rust::{parse_args, Args};
use std::env;
use std::ffi::CString;
use std::io::{self, Write};
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

struct Sampler(*mut ffi::llama_sampler);

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::llama_sampler_free(self.0);
            }
        }
    }
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]");
    eprintln!();
}

fn token_to_piece(
    vocab: *const ffi::llama_vocab,
    token: ffi::llama_token,
) -> Result<String, String> {
    let mut buf = vec![0_i8; 128];
    let n = unsafe {
        ffi::llama_token_to_piece(vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
    };

    if n < 0 {
        return Err("failed to convert token to piece".to_string());
    }

    let bytes = buf[..n as usize]
        .iter()
        .map(|&c| c as u8)
        .collect::<Vec<_>>();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn run(args: Args) -> Result<(), String> {
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;
    let prompt = CString::new(args.prompt.as_str())
        .map_err(|_| "prompt contains an interior NUL byte".to_string())?;

    unsafe {
        ffi::ggml_backend_load_all();
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

    let n_prompt = unsafe {
        -ffi::llama_tokenize(
            vocab,
            prompt.as_ptr(),
            args.prompt.len() as i32,
            ptr::null_mut(),
            0,
            true,
            true,
        )
    };

    if n_prompt <= 0 {
        return Err("failed to size prompt tokenization".to_string());
    }

    let mut prompt_tokens = vec![0_i32; n_prompt as usize];
    let n_tokenized = unsafe {
        ffi::llama_tokenize(
            vocab,
            prompt.as_ptr(),
            args.prompt.len() as i32,
            prompt_tokens.as_mut_ptr(),
            prompt_tokens.len() as i32,
            true,
            true,
        )
    };

    if n_tokenized < 0 {
        return Err("failed to tokenize the prompt".to_string());
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = (n_prompt + args.n_predict - 1).max(1) as u32;
    ctx_params.n_batch = n_prompt as u32;
    ctx_params.no_perf = false;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create the llama_context".to_string());
    }

    let mut sampler_params = unsafe { ffi::llama_sampler_chain_default_params() };
    sampler_params.no_perf = false;
    let sampler = Sampler(unsafe { ffi::llama_sampler_chain_init(sampler_params) });
    if sampler.0.is_null() {
        return Err("failed to create sampler chain".to_string());
    }

    let greedy = unsafe { ffi::llama_sampler_init_greedy() };
    if greedy.is_null() {
        return Err("failed to create greedy sampler".to_string());
    }
    unsafe {
        ffi::llama_sampler_chain_add(sampler.0, greedy);
    }

    for &id in &prompt_tokens {
        print!("{}", token_to_piece(vocab, id)?);
    }
    io::stdout().flush().map_err(|err| err.to_string())?;

    let mut batch =
        unsafe { ffi::llama_batch_get_one(prompt_tokens.as_mut_ptr(), prompt_tokens.len() as i32) };
    let mut decoder_start_token: ffi::llama_token;

    if unsafe { ffi::llama_model_has_encoder(model.0) } {
        if unsafe { ffi::llama_encode(ctx.0, batch) } != 0 {
            return Err("failed to eval encoder".to_string());
        }

        decoder_start_token = unsafe { ffi::llama_model_decoder_start_token(model.0) };
        if decoder_start_token == ffi::LLAMA_TOKEN_NULL {
            decoder_start_token = unsafe { ffi::llama_vocab_bos(vocab) };
        }

        batch = unsafe { ffi::llama_batch_get_one(&mut decoder_start_token, 1) };
    }

    let t_main_start = unsafe { ffi::ggml_time_us() };
    let mut n_decode = 0;
    let mut new_token_id = 0_i32;
    let mut n_pos = 0;

    while n_pos + batch.n_tokens < n_prompt + args.n_predict {
        let ret = unsafe { ffi::llama_decode(ctx.0, batch) };
        if ret != 0 {
            return Err(format!("failed to eval, return code {ret}"));
        }

        n_pos += batch.n_tokens;
        new_token_id = unsafe { ffi::llama_sampler_sample(sampler.0, ctx.0, -1) };

        if unsafe { ffi::llama_vocab_is_eog(vocab, new_token_id) } {
            break;
        }

        print!("{}", token_to_piece(vocab, new_token_id)?);
        io::stdout().flush().map_err(|err| err.to_string())?;

        batch = unsafe { ffi::llama_batch_get_one(&mut new_token_id, 1) };
        n_decode += 1;
    }

    println!();

    let t_main_end = unsafe { ffi::ggml_time_us() };
    let elapsed = (t_main_end - t_main_start) as f32 / 1_000_000.0;
    eprintln!(
        "main: decoded {n_decode} tokens in {elapsed:.2} s, speed: {:.2} t/s",
        n_decode as f32 / elapsed.max(f32::EPSILON)
    );
    eprintln!();
    unsafe {
        ffi::llama_perf_sampler_print(sampler.0);
        ffi::llama_perf_context_print(ctx.0);
    }
    eprintln!();

    let _ = new_token_id;

    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv
        .next()
        .unwrap_or_else(|| "llama-simple-rust".to_string());

    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{program}: error: {err}");
            print_usage(&program);
            std::process::exit(1);
        }
    };

    if let Err(err) = run(args) {
        eprintln!("{program}: error: {err}");
        std::process::exit(1);
    }
}
