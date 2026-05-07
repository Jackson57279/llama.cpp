use llama_debug_rust::{normalize_embedding, parse_args, Args};
use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::ptr;
use std::slice;

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

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf -p \"Hello my name is\" --save-logits");
    eprintln!("    {program} -m model.gguf --embedding -p \"Hello\" --save-logits");
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

fn token_to_piece(
    vocab: *const ffi::llama_vocab,
    token: ffi::llama_token,
) -> Result<String, String> {
    let mut buf = vec![0_i8; 128];
    let n = unsafe {
        ffi::llama_token_to_piece(vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
    };
    if n < 0 {
        return Err(format!("failed to convert token {token} to piece"));
    }
    let bytes = buf[..n as usize]
        .iter()
        .map(|&c| c as u8)
        .collect::<Vec<_>>();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn print_tokenized_prompt(
    vocab: *const ffi::llama_vocab,
    tokens: &[ffi::llama_token],
    prompt: &str,
) -> Result<(), String> {
    println!("Model add_bos: {}", unsafe {
        ffi::llama_vocab_get_add_bos(vocab)
    });
    println!("Input prompt: \"{prompt}\"");
    println!("Token ids ({}):", tokens.len());
    for &token in tokens {
        print!("{}({token}) ", token_to_piece(vocab, token)?);
    }
    println!();
    Ok(())
}

fn save_output_data(
    data: &[f32],
    tokens: &[ffi::llama_token],
    prompt: &str,
    model_path: &str,
    output_dir: &str,
    embedding: bool,
) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|err| err.to_string())?;
    let model_name = Path::new(model_path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    let type_suffix = if embedding { "-embeddings" } else { "" };
    let base_path = Path::new(output_dir).join(format!("llamacpp-{model_name}{type_suffix}"));

    let bin_path = base_path.with_extension("bin");
    let mut bin_file = BufWriter::new(File::create(&bin_path).map_err(|err| err.to_string())?);
    for value in data {
        bin_file
            .write_all(&value.to_ne_bytes())
            .map_err(|err| err.to_string())?;
    }
    println!("Data saved to {}", bin_path.display());

    let txt_path = base_path.with_extension("txt");
    let mut txt_file = BufWriter::new(File::create(&txt_path).map_err(|err| err.to_string())?);
    for (i, value) in data.iter().enumerate() {
        writeln!(txt_file, "{i}: {value}").map_err(|err| err.to_string())?;
    }
    println!("Data saved to {}", txt_path.display());

    let prompt_path = Path::new(&format!("{}-prompt.txt", base_path.display())).to_path_buf();
    let mut prompt_file =
        BufWriter::new(File::create(&prompt_path).map_err(|err| err.to_string())?);
    writeln!(prompt_file, "prompt: {prompt}").map_err(|err| err.to_string())?;
    writeln!(prompt_file, "n_tokens: {}", tokens.len()).map_err(|err| err.to_string())?;
    write!(prompt_file, "token ids: ").map_err(|err| err.to_string())?;
    for (i, token) in tokens.iter().enumerate() {
        if i > 0 {
            write!(prompt_file, ", ").map_err(|err| err.to_string())?;
        }
        write!(prompt_file, "{token}").map_err(|err| err.to_string())?;
    }
    writeln!(prompt_file).map_err(|err| err.to_string())?;
    println!("Prompt saved to {}", prompt_path.display());

    let tokens_path = Path::new(&format!("{}-tokens.bin", base_path.display())).to_path_buf();
    let mut tokens_file =
        BufWriter::new(File::create(&tokens_path).map_err(|err| err.to_string())?);
    for token in tokens {
        tokens_file
            .write_all(&token.to_ne_bytes())
            .map_err(|err| err.to_string())?;
    }
    println!("Tokens saved to {}", tokens_path.display());

    Ok(())
}

fn output_data(
    ctx: *mut ffi::llama_context,
    model: *mut ffi::llama_model,
    vocab: *const ffi::llama_vocab,
    tokens: &[ffi::llama_token],
    args: &Args,
) -> Result<Vec<f32>, String> {
    if args.embedding {
        let n_embd = unsafe { ffi::llama_model_n_embd_out(model) };
        if n_embd <= 0 {
            return Err("model returned invalid embedding size".to_string());
        }
        let pooling = !matches!(unsafe { ffi::llama_pooling_type(ctx) }, -1 | 0);
        let n_embd_count = if pooling { 1 } else { tokens.len() };
        let n_floats = n_embd as usize * n_embd_count;
        let embd = if pooling {
            unsafe { ffi::llama_get_embeddings_seq(ctx, 0) }
        } else {
            unsafe { ffi::llama_get_embeddings(ctx) }
        };
        if embd.is_null() {
            return Err("failed to get embeddings from the model".to_string());
        }
        let raw = unsafe { slice::from_raw_parts(embd, n_floats) };
        if args.embd_normalize >= 0 {
            let mut normalized = Vec::with_capacity(n_floats);
            for chunk in raw.chunks(n_embd as usize) {
                normalized.extend(normalize_embedding(chunk, args.embd_normalize));
            }
            Ok(normalized)
        } else {
            Ok(raw.to_vec())
        }
    } else {
        let logits = unsafe { ffi::llama_get_logits_ith(ctx, tokens.len() as i32 - 1) };
        if logits.is_null() {
            return Err("failed to get logits from the model".to_string());
        }
        let n_logits = unsafe { ffi::llama_vocab_n_tokens(vocab) };
        Ok(unsafe { slice::from_raw_parts(logits, n_logits as usize) }.to_vec())
    }
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

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = tokens.len().max(1) as u32;
    ctx_params.n_batch = tokens.len() as u32;
    ctx_params.embeddings = args.embedding;
    ctx_params.no_perf = false;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create the llama_context".to_string());
    }

    let batch = unsafe { ffi::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
    if unsafe { ffi::llama_decode(ctx.0, batch) } != 0 {
        return Err("failed to eval".to_string());
    }

    print_tokenized_prompt(vocab, &tokens, &args.prompt)?;

    if args.save_logits {
        let data = output_data(ctx.0, model.0, vocab, &tokens, &args)?;
        save_output_data(
            &data,
            &tokens,
            &args.prompt,
            &args.model_path,
            &args.logits_output_dir,
            args.embedding,
        )?;
    }

    println!();
    unsafe {
        ffi::llama_perf_context_print(ctx.0);
    }

    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-debug".to_string());
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
