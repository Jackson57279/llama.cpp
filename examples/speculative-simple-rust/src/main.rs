use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::io::{self, Write};
use std::ptr;

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub model_path: String,
    pub draft_model_path: String,
    pub prompt: String,
    pub n_predict: i32,
    pub n_draft: usize,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_gpu_layers: i32,
    pub draft_n_gpu_layers: i32,
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
            draft_model_path: String::new(),
            prompt: "Hello my name is".to_string(),
            n_predict: 32,
            n_draft: 16,
            n_ctx: 512,
            n_batch: 512,
            n_gpu_layers: 99,
            draft_n_gpu_layers: -1,
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
    MissingDraftModel,
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidFloat(String, String),
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m/--model model.gguf"),
            ParseError::MissingDraftModel => {
                write!(f, "missing required -md/--model-draft draft.gguf")
            }
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
            "-md" | "--model-draft" | "--spec-draft-model" => {
                parsed.draft_model_path = take(&mut iter, &arg)?
            }
            "-p" | "--prompt" => parsed.prompt = take(&mut iter, &arg)?,
            "-f" | "--file" | "--prompt-file" => {
                let path = take(&mut iter, &arg)?;
                parsed.prompt = std::fs::read_to_string(&path).map_err(|err| {
                    ParseError::InvalidValue(format!("failed to read prompt file {path}: {err}"))
                })?;
            }
            "-n" | "--n-predict" => parsed.n_predict = parse_i32(&mut iter, &arg)?,
            "--spec-draft-n-max" => parsed.n_draft = parse_usize(&mut iter, &arg)?,
            "-c" | "--ctx-size" => parsed.n_ctx = parse_u32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_u32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "-ngld" | "--gpu-layers-draft" | "--n-gpu-layers-draft" => {
                parsed.draft_n_gpu_layers = parse_i32(&mut iter, &arg)?
            }
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
    if parsed.draft_model_path.is_empty() {
        return Err(ParseError::MissingDraftModel);
    }
    if parsed.n_predict < -1 {
        return Err(ParseError::InvalidValue(
            "--n-predict must be >= -1".to_string(),
        ));
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

fn decode_batch_get_one(
    ctx: *mut ffi::llama_context,
    tokens: &mut [ffi::llama_token],
) -> Result<(), String> {
    if tokens.is_empty() {
        return Ok(());
    }
    let batch = unsafe { ffi::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
    if unsafe { ffi::llama_decode(ctx, batch) } != 0 {
        return Err("llama_decode failed".to_string());
    }
    Ok(())
}

fn rebuild_draft_context(
    ctx: *mut ffi::llama_context,
    tokens: &[ffi::llama_token],
) -> Result<(), String> {
    unsafe {
        ffi::llama_memory_clear(ffi::llama_get_memory(ctx), true);
    }
    let mut working = tokens.to_vec();
    decode_batch_get_one(ctx, &mut working)
}

fn draft_tokens_from_model(
    ctx: *mut ffi::llama_context,
    sampler: *mut ffi::llama_sampler,
    n_draft: usize,
) -> Result<Vec<ffi::llama_token>, String> {
    let mut draft = Vec::new();
    for _ in 0..n_draft {
        let token = unsafe { ffi::llama_sampler_sample(sampler, ctx, -1) };
        unsafe {
            ffi::llama_sampler_accept(sampler, token);
        }
        draft.push(token);
        let mut one = [token];
        decode_batch_get_one(ctx, &mut one)?;
    }
    Ok(draft)
}

fn run(args: Args) -> Result<(), String> {
    let _backend = Backend::init();
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;
    let draft_model_path = CString::new(args.draft_model_path.as_str())
        .map_err(|_| "draft model path contains an interior NUL byte".to_string())?;

    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;
    let model_tgt =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model_tgt.0.is_null() {
        return Err("unable to load target model".to_string());
    }

    let mut draft_params = unsafe { ffi::llama_model_default_params() };
    draft_params.n_gpu_layers = args.draft_n_gpu_layers;
    let model_dft =
        Model(unsafe { ffi::llama_model_load_from_file(draft_model_path.as_ptr(), draft_params) });
    if model_dft.0.is_null() {
        return Err("unable to load draft model".to_string());
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = args.n_batch;
    ctx_params.n_ubatch = args.n_batch;
    let ctx_tgt = Context(unsafe { ffi::llama_init_from_model(model_tgt.0, ctx_params) });
    if ctx_tgt.0.is_null() {
        return Err("failed to create target context".to_string());
    }

    let mut ctx_dft_params = unsafe { ffi::llama_context_default_params() };
    ctx_dft_params.n_ctx = args.n_ctx;
    ctx_dft_params.n_batch = args.n_batch;
    ctx_dft_params.n_ubatch = args.n_batch;
    let ctx_dft = Context(unsafe { ffi::llama_init_from_model(model_dft.0, ctx_dft_params) });
    if ctx_dft.0.is_null() {
        return Err("failed to create draft context".to_string());
    }

    let vocab = unsafe { ffi::llama_model_get_vocab(model_tgt.0) };
    if vocab.is_null() {
        return Err("failed to get target vocabulary".to_string());
    }

    let input = tokenize(vocab, &args.prompt, true, true)?;
    if input.is_empty() {
        return Err("prompt produced no tokens".to_string());
    }
    if unsafe { ffi::llama_n_ctx(ctx_tgt.0) as usize } < input.len() {
        return Err(format!(
            "the prompt exceeds the context size ({} tokens, ctx {})",
            input.len(),
            unsafe { ffi::llama_n_ctx(ctx_tgt.0) }
        ));
    }

    println!();
    for token in &input {
        print!("{}", token_to_piece(vocab, *token)?);
    }
    io::stdout().flush().map_err(|err| err.to_string())?;

    let n_input = input.len();
    let t_enc_start = unsafe { ffi::ggml_time_us() };
    let sampler_tgt = sampler_init(&args);
    let sampler_dft = sampler_init(&args);
    let mut prompt_tgt = input[..input.len() - 1].to_vec();
    if !prompt_tgt.is_empty() {
        decode_batch_get_one(ctx_tgt.0, &mut prompt_tgt)?;
    }
    let mut id_last = *input.last().unwrap();
    let mut n_past = prompt_tgt.len() as i32;
    let mut batch_tgt = Batch(unsafe { ffi::llama_batch_init(args.n_batch as i32, 0, 1) });
    let t_enc_end = unsafe { ffi::ggml_time_us() };

    let mut n_predict = 0_i32;
    let mut n_drafted = 0_i32;
    let mut n_accept = 0_i32;
    let mut has_eos = false;
    let t_dec_start = unsafe { ffi::ggml_time_us() };

    loop {
        let mut draft_prefix = prompt_tgt.clone();
        draft_prefix.push(id_last);
        rebuild_draft_context(ctx_dft.0, &draft_prefix)?;
        let draft = draft_tokens_from_model(ctx_dft.0, sampler_dft.0, args.n_draft)?;
        n_drafted += draft.len() as i32;

        batch_clear(&mut batch_tgt.0);
        unsafe {
            batch_add(&mut batch_tgt.0, id_last, n_past, &[0], true);
            for (i, token) in draft.iter().enumerate() {
                batch_add(&mut batch_tgt.0, *token, n_past + 1 + i as i32, &[0], true);
            }
        }
        if unsafe { ffi::llama_decode(ctx_tgt.0, batch_tgt.0) } != 0 {
            return Err("failed to decode target speculative batch".to_string());
        }

        let mut ids = Vec::new();
        for i in 0..=draft.len() {
            let sampled = unsafe { ffi::llama_sampler_sample(sampler_tgt.0, ctx_tgt.0, i as i32) };
            unsafe {
                ffi::llama_sampler_accept(sampler_tgt.0, sampled);
            }
            ids.push(sampled);
            if i >= draft.len() || sampled != draft[i] {
                break;
            }
        }

        let accepted_draft = ids.len().saturating_sub(1);
        n_accept += accepted_draft as i32;
        n_predict += ids.len() as i32;
        n_past += ids.len() as i32;

        for (i, token) in ids.iter().enumerate() {
            prompt_tgt.push(id_last);
            id_last = *token;
            if unsafe { ffi::llama_vocab_is_eog(vocab, id_last) } {
                has_eos = true;
                break;
            }
            let token_str = token_to_piece(vocab, id_last)?;
            if args.use_color && i + 1 < ids.len() {
                print!("\x1b[36m{token_str}\x1b[37m");
            } else {
                print!("{token_str}");
            }
        }
        io::stdout().flush().map_err(|err| err.to_string())?;

        unsafe {
            ffi::llama_memory_seq_rm(ffi::llama_get_memory(ctx_tgt.0), 0, n_past, -1);
        }

        if (args.n_predict >= 0 && n_predict > args.n_predict) || has_eos {
            break;
        }
    }

    let t_dec_end = unsafe { ffi::ggml_time_us() };
    println!("\n");
    eprintln!(
        "encoded {n_input:4} tokens in {:8.3} seconds, speed: {:8.3} t/s",
        (t_enc_end - t_enc_start) as f64 / 1e6,
        n_input as f64 / ((t_enc_end - t_enc_start) as f64 / 1e6)
    );
    eprintln!(
        "decoded {n_predict:4} tokens in {:8.3} seconds, speed: {:8.3} t/s",
        (t_dec_end - t_dec_start) as f64 / 1e6,
        n_predict as f64 / ((t_dec_end - t_dec_start) as f64 / 1e6)
    );
    eprintln!("\nn_draft   = {}", args.n_draft);
    eprintln!("n_predict = {n_predict}");
    eprintln!("n_drafted = {n_drafted}");
    eprintln!("n_accept  = {n_accept}");
    if n_drafted > 0 {
        eprintln!(
            "accept    = {:.3}%",
            100.0 * n_accept as f64 / n_drafted as f64
        );
    } else {
        eprintln!("accept    = 0.000%");
    }
    eprintln!("\ndraft:\n");
    unsafe {
        ffi::llama_perf_sampler_print(sampler_dft.0);
    }
    eprintln!("\ntarget:\n");
    unsafe {
        ffi::llama_perf_sampler_print(sampler_tgt.0);
        ffi::llama_perf_context_print(ctx_tgt.0);
    }
    Ok(())
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m target.gguf -md draft.gguf -p \"Hello\"");
    eprintln!();
}

fn main() {
    let mut argv = env::args();
    let program = argv
        .next()
        .unwrap_or_else(|| "llama-speculative-simple".to_string());
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

    #[test]
    fn parses_speculative_simple_options() {
        let args = parse_args([
            "-m",
            "target.gguf",
            "-md",
            "draft.gguf",
            "-p",
            "hello",
            "-n",
            "8",
            "--spec-draft-n-max",
            "3",
            "-c",
            "256",
            "-b",
            "128",
            "-ngl",
            "0",
            "-ngld",
            "1",
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
        assert_eq!(args.model_path, "target.gguf");
        assert_eq!(args.draft_model_path, "draft.gguf");
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.n_predict, 8);
        assert_eq!(args.n_draft, 3);
        assert_eq!(args.n_ctx, 256);
        assert_eq!(args.n_batch, 128);
        assert_eq!(args.n_gpu_layers, 0);
        assert_eq!(args.draft_n_gpu_layers, 1);
        assert_eq!(args.top_k, 10);
        assert_eq!(args.top_p, 0.8);
        assert_eq!(args.temp, 0.5);
        assert_eq!(args.seed, 42);
        assert!(args.use_color);
    }

    #[test]
    fn rejects_missing_draft_model() {
        assert_eq!(
            parse_args(["-m", "target.gguf"]).unwrap_err(),
            ParseError::MissingDraftModel
        );
    }

    #[test]
    fn rejects_bad_predict_count() {
        assert_eq!(
            parse_args(["-m", "target.gguf", "-md", "draft.gguf", "-n", "-2"]).unwrap_err(),
            ParseError::InvalidValue("--n-predict must be >= -1".to_string())
        );
    }
}
