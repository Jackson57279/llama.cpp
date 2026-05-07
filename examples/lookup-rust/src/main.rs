use llama_lookup_merge::{
    draft_tokens, load_cache, merge_cache, save_cache, update_cache, NgramCache, LLAMA_NGRAM_MAX,
    LLAMA_NGRAM_MIN,
};
use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::io::{self, Write};
use std::path::Path;
use std::ptr;

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub n_predict: i32,
    pub n_draft: usize,
    pub static_cache: String,
    pub dynamic_cache: String,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_gpu_layers: i32,
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub seed: u32,
    pub use_color: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: "Hello my name is".to_string(),
            n_predict: 32,
            n_draft: 16,
            static_cache: String::new(),
            dynamic_cache: String::new(),
            n_ctx: 512,
            n_batch: 512,
            n_gpu_layers: 99,
            top_k: 40,
            top_p: 0.95,
            temp: 0.8,
            seed: ffi::LLAMA_DEFAULT_SEED,
            use_color: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    MissingModel,
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidFloat(String, String),
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m/--model model.gguf"),
            ParseError::MissingValue(flag) => write!(f, "missing value for {flag}"),
            ParseError::InvalidInteger(flag, value) => {
                write!(f, "invalid integer for {flag}: {value}")
            }
            ParseError::InvalidFloat(flag, value) => write!(f, "invalid float for {flag}: {value}"),
            ParseError::InvalidValue(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ParseError {}

pub fn parse_args<I, S>(args: I) -> Result<Args, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut parsed = Args::default();
    let mut iter = args.into_iter().map(Into::into);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-m" | "--model" => parsed.model_path = take(&mut iter, &arg)?,
            "-p" | "--prompt" => parsed.prompt = take(&mut iter, &arg)?,
            "-f" | "--file" | "--prompt-file" => {
                let path = take(&mut iter, &arg)?;
                parsed.prompt = std::fs::read_to_string(&path).map_err(|err| {
                    ParseError::InvalidValue(format!("failed to read prompt file {path}: {err}"))
                })?;
            }
            "-n" | "--n-predict" => parsed.n_predict = parse_i32(&mut iter, &arg)?,
            "--spec-draft-n-max" => parsed.n_draft = parse_usize(&mut iter, &arg)?,
            "-lcs" | "--lookup-cache-static" => parsed.static_cache = take(&mut iter, &arg)?,
            "-lcd" | "--lookup-cache-dynamic" => parsed.dynamic_cache = take(&mut iter, &arg)?,
            "-c" | "--ctx-size" => parsed.n_ctx = parse_u32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_u32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "--top-k" => parsed.top_k = parse_i32(&mut iter, &arg)?,
            "--top-p" => parsed.top_p = parse_f32(&mut iter, &arg)?,
            "--temp" => parsed.temp = parse_f32(&mut iter, &arg)?,
            "-s" | "--seed" => parsed.seed = parse_u32(&mut iter, &arg)?,
            "--color" => parsed.use_color = true,
            "--no-color" => parsed.use_color = false,
            "-h" | "--help" => return Err(ParseError::MissingModel),
            other => {
                let mut prompt = vec![other.to_string()];
                prompt.extend(iter);
                parsed.prompt = prompt.join(" ");
                break;
            }
        }
    }

    if parsed.model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }
    if parsed.n_ctx == 0 || parsed.n_batch == 0 {
        return Err(ParseError::InvalidValue(
            "context and batch sizes must be positive".to_string(),
        ));
    }
    Ok(parsed)
}

fn take<I>(iter: &mut I, flag: &str) -> Result<String, ParseError>
where
    I: Iterator<Item = String>,
{
    iter.next()
        .ok_or_else(|| ParseError::MissingValue(flag.to_string()))
}

fn parse_i32<I>(iter: &mut I, flag: &str) -> Result<i32, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidInteger(flag.to_string(), value))
}

fn parse_u32<I>(iter: &mut I, flag: &str) -> Result<u32, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidInteger(flag.to_string(), value))
}

fn parse_usize<I>(iter: &mut I, flag: &str) -> Result<usize, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidInteger(flag.to_string(), value))
}

fn parse_f32<I>(iter: &mut I, flag: &str) -> Result<f32, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidFloat(flag.to_string(), value))
}

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

struct Backend;

impl Backend {
    fn init() -> Self {
        unsafe {
            ffi::ggml_backend_load_all();
            ffi::llama_backend_init();
            ffi::llama_numa_init(0);
        }
        Self
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            ffi::llama_backend_free();
        }
    }
}

fn sampler_init(args: &Args) -> Sampler {
    unsafe {
        let chain = ffi::llama_sampler_chain_init(ffi::llama_sampler_chain_default_params());
        ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_top_k(args.top_k));
        ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_top_p(args.top_p, 1));
        ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_temp(args.temp));
        ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_dist(args.seed));
        Sampler(chain)
    }
}

fn tokenize(
    vocab: *const ffi::llama_vocab,
    text: &str,
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<ffi::llama_token>, String> {
    let text_c =
        CString::new(text).map_err(|_| "prompt contains an interior NUL byte".to_string())?;
    let n_tokens = unsafe {
        -ffi::llama_tokenize(
            vocab,
            text_c.as_ptr(),
            text.len() as i32,
            ptr::null_mut(),
            0,
            add_special,
            parse_special,
        )
    };
    if n_tokens <= 0 {
        return Err("failed to size tokenization".to_string());
    }
    let mut tokens = vec![0_i32; n_tokens as usize];
    let n = unsafe {
        ffi::llama_tokenize(
            vocab,
            text_c.as_ptr(),
            text.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            add_special,
            parse_special,
        )
    };
    if n < 0 {
        return Err("failed to tokenize prompt".to_string());
    }
    tokens.truncate(n as usize);
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
        return Err("failed to convert token to piece".to_string());
    }
    let bytes = buf[..n as usize]
        .iter()
        .map(|&c| c as u8)
        .collect::<Vec<_>>();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn batch_clear(batch: &mut ffi::llama_batch) {
    batch.n_tokens = 0;
}

unsafe fn batch_add(
    batch: &mut ffi::llama_batch,
    token: ffi::llama_token,
    pos: ffi::llama_pos,
    seq_ids: &[ffi::llama_seq_id],
    logits: bool,
) {
    let i = batch.n_tokens as isize;
    *batch.token.offset(i) = token;
    *batch.pos.offset(i) = pos;
    *batch.n_seq_id.offset(i) = seq_ids.len() as i32;
    let seq_slot = *batch.seq_id.offset(i);
    for (j, seq_id) in seq_ids.iter().enumerate() {
        *seq_slot.add(j) = *seq_id;
    }
    *batch.logits.offset(i) = if logits { 1 } else { 0 };
    batch.n_tokens += 1;
}

fn decode_prompt(
    ctx: *mut ffi::llama_context,
    tokens: &mut [ffi::llama_token],
) -> Result<(), String> {
    if tokens.len() > 1 {
        let batch =
            unsafe { ffi::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32 - 1) };
        if unsafe { ffi::llama_decode(ctx, batch) } != 0 {
            return Err("failed to decode prompt prefix".to_string());
        }
    }
    let last = tokens
        .last_mut()
        .ok_or_else(|| "prompt produced no tokens".to_string())?;
    let batch = unsafe { ffi::llama_batch_get_one(last as *mut ffi::llama_token, 1) };
    if unsafe { ffi::llama_decode(ctx, batch) } != 0 {
        return Err("failed to decode final prompt token".to_string());
    }
    Ok(())
}

fn load_cache_optional(path: &str, required: bool) -> Result<NgramCache, String> {
    if path.is_empty() {
        return Ok(NgramCache::new());
    }
    match load_cache(path) {
        Ok(cache) => Ok(cache),
        Err(err) if !required && err.kind() == io::ErrorKind::NotFound => Ok(NgramCache::new()),
        Err(err) => Err(format!("failed to load lookup cache {path}: {err}")),
    }
}

fn maybe_save_cache(cache: &NgramCache, path: &str) -> Result<(), String> {
    if path.is_empty() {
        return Ok(());
    }
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(format!("directory does not exist: {}", parent.display()));
        }
    }
    save_cache(cache, path).map_err(|err| format!("failed to save lookup cache {path}: {err}"))
}

fn run(args: Args) -> Result<(), String> {
    let _backend = Backend::init();
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;

    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;
    let model =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model.0.is_null() {
        return Err("unable to load model".to_string());
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = args.n_batch;
    ctx_params.n_ubatch = args.n_batch;
    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create llama_context".to_string());
    }

    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        return Err("failed to get model vocabulary".to_string());
    }

    let mut input = tokenize(vocab, &args.prompt, true, true)?;
    let mut ngram_cache_context = NgramCache::new();
    let t_start_draft = unsafe { ffi::ggml_time_us() };
    update_cache(
        &mut ngram_cache_context,
        LLAMA_NGRAM_MIN,
        LLAMA_NGRAM_MAX,
        &input,
        input.len(),
    );
    let ngram_cache_static = load_cache_optional(&args.static_cache, true)?;
    let mut ngram_cache_dynamic = load_cache_optional(&args.dynamic_cache, false)?;
    let t_draft_flat_us = unsafe { ffi::ggml_time_us() } - t_start_draft;
    let mut t_draft_us = 0_i64;

    let max_tokens_list_size = unsafe { ffi::llama_n_ctx(ctx.0) as usize }.saturating_sub(4);
    if input.len() > max_tokens_list_size {
        return Err(format!(
            "prompt too long ({} tokens, max {max_tokens_list_size})",
            input.len()
        ));
    }

    println!();
    for token in &input {
        print!("{}", token_to_piece(vocab, *token)?);
    }
    io::stdout().flush().map_err(|err| err.to_string())?;

    let n_input = input.len();
    let t_enc_start = unsafe { ffi::ggml_time_us() };
    decode_prompt(ctx.0, &mut input)?;
    let t_enc_end = unsafe { ffi::ggml_time_us() };

    let mut n_predict = 0_i32;
    let mut n_drafted = 0_i32;
    let mut n_accept = 0_i32;
    let mut n_past = input.len() as i32;
    let mut has_eos = false;
    let sampler = sampler_init(&args);
    let mut draft = Vec::new();
    let mut batch_tgt =
        Batch(unsafe { ffi::llama_batch_init(ffi::llama_n_ctx(ctx.0) as i32, 0, 1) });
    let t_dec_start = unsafe { ffi::ggml_time_us() };

    loop {
        let mut i_dft = 0usize;
        loop {
            let id = unsafe { ffi::llama_sampler_sample(sampler.0, ctx.0, i_dft as i32) };
            unsafe {
                ffi::llama_sampler_accept(sampler.0, id);
            }
            let token_str = token_to_piece(vocab, id)?;
            if !args.use_color {
                print!("{token_str}");
            }
            if unsafe { ffi::llama_vocab_is_eog(vocab, id) } {
                has_eos = true;
            }
            n_predict += 1;

            if i_dft < draft.len() && id == draft[i_dft] {
                n_accept += 1;
                n_past += 1;
                i_dft += 1;
                input.push(id);
                let t_start = unsafe { ffi::ggml_time_us() };
                update_cache(
                    &mut ngram_cache_context,
                    LLAMA_NGRAM_MIN,
                    LLAMA_NGRAM_MAX,
                    &input,
                    1,
                );
                t_draft_us += unsafe { ffi::ggml_time_us() } - t_start;
                if args.use_color {
                    print!("\x1b[34m{token_str}\x1b[0m");
                }
                continue;
            }

            if args.use_color {
                print!("{token_str}");
            }
            io::stdout().flush().map_err(|err| err.to_string())?;
            draft.clear();
            draft.push(id);
            input.push(id);
            let t_start = unsafe { ffi::ggml_time_us() };
            update_cache(
                &mut ngram_cache_context,
                LLAMA_NGRAM_MIN,
                LLAMA_NGRAM_MAX,
                &input,
                1,
            );
            t_draft_us += unsafe { ffi::ggml_time_us() } - t_start;
            break;
        }

        if (args.n_predict > 0 && n_predict > args.n_predict) || has_eos {
            break;
        }

        unsafe {
            ffi::llama_memory_seq_rm(ffi::llama_get_memory(ctx.0), 0, n_past, -1);
        }
        batch_clear(&mut batch_tgt.0);
        unsafe {
            batch_add(&mut batch_tgt.0, draft[0], n_past, &[0], true);
        }
        let t_start = unsafe { ffi::ggml_time_us() };
        draft_tokens(
            &input,
            &mut draft,
            args.n_draft,
            LLAMA_NGRAM_MIN,
            LLAMA_NGRAM_MAX,
            &ngram_cache_context,
            &ngram_cache_dynamic,
            &ngram_cache_static,
        );
        for (i, token) in draft.iter().enumerate().skip(1) {
            unsafe {
                batch_add(&mut batch_tgt.0, *token, n_past + i as i32, &[0], true);
            }
        }
        t_draft_us += unsafe { ffi::ggml_time_us() } - t_start;
        n_drafted += draft.len().saturating_sub(1) as i32;

        if unsafe { ffi::llama_decode(ctx.0, batch_tgt.0) } != 0 {
            return Err("failed to decode drafted batch".to_string());
        }
        n_past += 1;
        draft.remove(0);
    }

    let t_dec_end = unsafe { ffi::ggml_time_us() };
    merge_cache(&mut ngram_cache_dynamic, ngram_cache_context);
    maybe_save_cache(&ngram_cache_dynamic, &args.dynamic_cache)?;

    println!("\n");
    eprintln!(
        "encoded {n_input:4} tokens in {:8.3} seconds, speed: {:8.3} t/s",
        (t_enc_end - t_enc_start) as f64 / 1e6,
        input.len() as f64 / ((t_enc_end - t_enc_start) as f64 / 1e6)
    );
    eprintln!(
        "decoded {n_predict:4} tokens in {:8.3} seconds, speed: {:8.3} t/s",
        (t_dec_end - t_dec_start) as f64 / 1e6,
        n_predict as f64 / ((t_dec_end - t_dec_start) as f64 / 1e6)
    );
    eprintln!("\nn_draft      = {}", args.n_draft);
    eprintln!("n_predict    = {n_predict}");
    eprintln!("n_drafted    = {n_drafted}");
    eprintln!("t_draft_flat = {:.2} ms", t_draft_flat_us as f64 * 1e-3);
    if n_drafted > 0 && t_draft_us > 0 {
        eprintln!(
            "t_draft      = {:.2} ms, {:.2} us per token, {:.2} tokens per second",
            t_draft_us as f64 * 1e-3,
            t_draft_us as f64 / n_drafted as f64,
            n_drafted as f64 / (1e-6 * t_draft_us as f64)
        );
        eprintln!(
            "accept       = {:.3}%",
            100.0 * n_accept as f64 / n_drafted as f64
        );
    } else {
        eprintln!("t_draft      = 0.00 ms");
        eprintln!("accept       = 0.000%");
    }
    eprintln!("n_accept     = {n_accept}\n");
    unsafe {
        ffi::llama_perf_sampler_print(sampler.0);
        ffi::llama_perf_context_print(ctx.0);
    }
    Ok(())
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf -p \"Hello\" -lcs static.bin -lcd dynamic.bin");
    eprintln!();
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-lookup".to_string());
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

#[cfg(test)]
mod tests {
    use super::*;
    use llama_lookup_merge::{ngram_from_slice, NgramCache};
    use std::collections::HashMap;

    #[test]
    fn parses_lookup_options() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-p",
            "hello",
            "-n",
            "8",
            "--spec-draft-n-max",
            "3",
            "-lcs",
            "static.bin",
            "-lcd",
            "dynamic.bin",
            "-c",
            "256",
            "-b",
            "128",
            "-ngl",
            "0",
            "--top-k",
            "10",
            "--top-p",
            "0.8",
            "--temp",
            "0.5",
            "-s",
            "42",
            "--color",
        ])
        .unwrap();
        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.n_predict, 8);
        assert_eq!(args.n_draft, 3);
        assert_eq!(args.static_cache, "static.bin");
        assert_eq!(args.dynamic_cache, "dynamic.bin");
        assert_eq!(args.n_ctx, 256);
        assert_eq!(args.n_batch, 128);
        assert_eq!(args.n_gpu_layers, 0);
        assert_eq!(args.top_k, 10);
        assert_eq!(args.top_p, 0.8);
        assert_eq!(args.temp, 0.5);
        assert_eq!(args.seed, 42);
        assert!(args.use_color);
    }

    #[test]
    fn rejects_missing_model() {
        assert_eq!(
            parse_args(["-p", "hello"]).unwrap_err(),
            ParseError::MissingModel
        );
    }

    #[test]
    fn dynamic_cache_missing_is_allowed() {
        let cache = load_cache_optional("/tmp/llama-lookup-rust-missing-cache.bin", false).unwrap();
        assert!(cache.is_empty());
    }

    #[test]
    fn draft_uses_existing_lookup_cache() {
        let mut context = NgramCache::new();
        context.insert(ngram_from_slice(&[10]), HashMap::from([(20, 2)]));
        let mut draft = vec![10];
        draft_tokens(
            &[1, 10],
            &mut draft,
            1,
            LLAMA_NGRAM_MIN,
            LLAMA_NGRAM_MAX,
            &context,
            &NgramCache::new(),
            &NgramCache::new(),
        );
        assert_eq!(draft, vec![10, 20]);
    }
}
