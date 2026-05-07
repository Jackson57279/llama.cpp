use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::io::{self, Write};
use std::ptr;

const STATE_FILE: &str = "dump_state.bin";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub n_predict: i32,
    pub n_batch: i32,
    pub n_parallel: i32,
    pub n_gpu_layers: i32,
    pub seed: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingModel,
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m model.gguf"),
            ParseError::MissingValue(flag) => write!(f, "missing value for {flag}"),
            ParseError::InvalidInteger(flag, value) => {
                write!(f, "invalid integer for {flag}: {value}")
            }
            ParseError::InvalidValue(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ParseError {}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: "The quick brown fox".to_string(),
            n_predict: 16,
            n_batch: 512,
            n_parallel: 1,
            n_gpu_layers: 99,
            seed: 1234,
        }
    }
}

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
            "-n" | "--n-predict" => parsed.n_predict = parse_i32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_i32(&mut iter, &arg)?,
            "-np" | "--parallel" => parsed.n_parallel = parse_i32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "-s" | "--seed" => parsed.seed = parse_u32(&mut iter, &arg)?,
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
    if parsed.n_predict < 0 {
        parsed.n_predict = 16;
    }
    if parsed.n_predict == 0 {
        return Err(ParseError::InvalidValue(
            "--n-predict must be non-zero".to_string(),
        ));
    }
    if parsed.n_batch <= 0 {
        return Err(ParseError::InvalidValue(
            "--batch-size must be positive".to_string(),
        ));
    }
    if parsed.n_parallel <= 0 {
        return Err(ParseError::InvalidValue(
            "--parallel must be positive".to_string(),
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

pub fn should_enable_unified_kv(n_parallel: i32) -> bool {
    n_parallel == 1
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

fn tokenize(vocab: *const ffi::llama_vocab, text: &str) -> Result<Vec<ffi::llama_token>, String> {
    let text_c =
        CString::new(text).map_err(|_| "prompt contains an interior NUL byte".to_string())?;
    let n_tokens = unsafe {
        -ffi::llama_tokenize(
            vocab,
            text_c.as_ptr(),
            text.len() as i32,
            ptr::null_mut(),
            0,
            true,
            true,
        )
    };
    if n_tokens <= 0 {
        return Err("failed to size prompt tokenization".to_string());
    }

    let mut tokens = vec![0_i32; n_tokens as usize];
    let n = unsafe {
        ffi::llama_tokenize(
            vocab,
            text_c.as_ptr(),
            text.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            true,
            true,
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
    seq_id: ffi::llama_seq_id,
    logits: bool,
) {
    let i = batch.n_tokens as isize;
    *batch.token.offset(i) = token;
    *batch.pos.offset(i) = pos;
    *batch.n_seq_id.offset(i) = 1;
    let seq_slot = *batch.seq_id.offset(i);
    *seq_slot = seq_id;
    *batch.logits.offset(i) = i8::from(logits);
    batch.n_tokens += 1;
}

fn replay_last_token(
    ctx: *mut ffi::llama_context,
    last_token: ffi::llama_token,
    pos: ffi::llama_pos,
) -> Result<(), String> {
    let mut token = last_token;
    let mut batch = unsafe { ffi::llama_batch_get_one(&mut token, 1) };
    batch.pos = &pos as *const i32 as *mut i32;
    if unsafe { ffi::llama_decode(ctx, batch) } != 0 {
        return Err("failed to replay last token".to_string());
    }
    Ok(())
}

fn prompt_batch_decode(
    ctx: *mut ffi::llama_context,
    tokens: &mut [ffi::llama_token],
    n_past: &mut i32,
    n_batch: i32,
    state_file: &CString,
) -> Result<(), String> {
    if tokens.is_empty() {
        return Ok(());
    }
    if tokens.len() > 1 {
        let n_before_last = tokens.len() - 1;
        if tokens.len() as i32 > n_batch {
            return Err("prompt token count exceeds batch size".to_string());
        }

        let batch = unsafe { ffi::llama_batch_get_one(tokens.as_mut_ptr(), n_before_last as i32) };
        if unsafe { ffi::llama_decode(ctx, batch) } != 0 {
            return Err("failed to evaluate prompt before last token".to_string());
        }
        *n_past += n_before_last as i32;

        if !unsafe {
            ffi::llama_state_save_file(ctx, state_file.as_ptr(), tokens.as_ptr(), n_before_last)
        } {
            return Err("failed to save state file".to_string());
        }
        eprintln!(
            "saved session before last token to {}, n_tokens = {n_before_last}",
            state_file.to_string_lossy()
        );

        replay_last_token(ctx, *tokens.last().unwrap(), *n_past)?;
        *n_past += 1;
    } else {
        let batch = unsafe { ffi::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
        if unsafe { ffi::llama_decode(ctx, batch) } != 0 {
            return Err("failed to evaluate prompt".to_string());
        }
        *n_past += tokens.len() as i32;
    }
    Ok(())
}

fn make_sampler(seed: u32) -> Result<Sampler, String> {
    let params = unsafe { ffi::llama_sampler_chain_default_params() };
    let sampler = Sampler(unsafe { ffi::llama_sampler_chain_init(params) });
    if sampler.0.is_null() {
        return Err("failed to create sampler".to_string());
    }
    unsafe {
        ffi::llama_sampler_chain_add(sampler.0, ffi::llama_sampler_init_dist(seed));
    }
    Ok(sampler)
}

fn make_context(
    model: *mut ffi::llama_model,
    args: &Args,
    n_seq_max: u32,
) -> Result<Context, String> {
    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_batch = args.n_batch as u32;
    ctx_params.n_ubatch = args.n_batch as u32;
    ctx_params.n_seq_max = n_seq_max;
    ctx_params.kv_unified = should_enable_unified_kv(args.n_parallel);
    ctx_params.no_perf = false;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create context".to_string());
    }
    Ok(ctx)
}

fn generate(
    ctx: *mut ffi::llama_context,
    vocab: *const ffi::llama_vocab,
    sampler: *mut ffi::llama_sampler,
    batch: &mut ffi::llama_batch,
    n_past: &mut i32,
    seq_id: ffi::llama_seq_id,
    n_predict: i32,
) -> Result<String, String> {
    let mut result = String::new();
    for _ in 0..n_predict {
        let next_token = unsafe { ffi::llama_sampler_sample(sampler, ctx, -1) };
        let piece = token_to_piece(vocab, next_token)?;
        print!("{piece}");
        io::stdout().flush().map_err(|err| err.to_string())?;
        result.push_str(&piece);

        batch_clear(batch);
        unsafe {
            batch_add(batch, next_token, *n_past, seq_id, true);
        }
        if unsafe { ffi::llama_decode(ctx, *batch) } != 0 {
            return Err("failed to evaluate generated token".to_string());
        }
        *n_past += 1;
    }
    Ok(result)
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf -p \"The quick brown fox\" -n 16 --seed 1234");
    eprintln!();
}

fn run(args: Args) -> Result<(), String> {
    if should_enable_unified_kv(args.n_parallel) {
        println!("main: n_parallel == 1, enabling unified kv cache");
    }

    let _backend = Backend::init();
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;
    let state_file = CString::new(STATE_FILE).unwrap();

    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;

    let model =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model.0.is_null() {
        return Err("unable to load model".to_string());
    }
    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        return Err("failed to get vocabulary".to_string());
    }

    let ctx = make_context(model.0, &args, args.n_parallel as u32)?;
    let sampler = make_sampler(args.seed)?;
    let mut tokens = tokenize(vocab, &args.prompt)?;
    let mut n_past = 0;

    prompt_batch_decode(ctx.0, &mut tokens, &mut n_past, args.n_batch, &state_file)?;

    println!("\nfirst run: {}", args.prompt);
    let mut batch = Batch(unsafe { ffi::llama_batch_init(1, 0, 1) });
    let result0 = generate(
        ctx.0,
        vocab,
        sampler.0,
        &mut batch.0,
        &mut n_past,
        0,
        args.n_predict,
    )?;
    println!("\n");

    let ctx2 = make_context(model.0, &args, args.n_parallel as u32)?;
    let sampler2 = make_sampler(args.seed)?;
    println!("\nsecond run: {}", args.prompt);

    let mut unused_tokens = vec![0_i32; tokens.len()];
    let mut n_token_count_out = 0usize;
    if !unsafe {
        ffi::llama_state_load_file(
            ctx2.0,
            state_file.as_ptr(),
            unused_tokens.as_mut_ptr(),
            unused_tokens.len(),
            &mut n_token_count_out,
        )
    } {
        return Err("failed to load state".to_string());
    }
    eprintln!("main: loaded state with {n_token_count_out} tokens");

    let mut n_past2 = n_token_count_out as i32;
    replay_last_token(ctx2.0, *tokens.last().unwrap(), n_past2)?;
    n_past2 += 1;
    let result1 = generate(
        ctx2.0,
        vocab,
        sampler2.0,
        &mut batch.0,
        &mut n_past2,
        0,
        args.n_predict,
    )?;
    println!("\n");

    if result0 != result1 {
        return Err("the 2 generations are different".to_string());
    }

    let ctx3 = make_context(model.0, &args, 2)?;
    let sampler3 = make_sampler(args.seed)?;
    println!("\nsingle seq run: {}", args.prompt);

    n_token_count_out = 0;
    if !unsafe {
        ffi::llama_state_load_file(
            ctx3.0,
            state_file.as_ptr(),
            unused_tokens.as_mut_ptr(),
            unused_tokens.len(),
            &mut n_token_count_out,
        )
    } {
        return Err("failed to load state into third context".to_string());
    }
    eprintln!("main: loaded state with {n_token_count_out} tokens");

    let mut n_past3 = n_token_count_out as i32;
    replay_last_token(ctx3.0, *tokens.last().unwrap(), n_past3)?;
    n_past3 += 1;

    let seq_size = unsafe { ffi::llama_state_seq_get_size(ctx3.0, 0) };
    let mut seq_store = vec![0u8; seq_size];
    let ncopy =
        unsafe { ffi::llama_state_seq_get_data(ctx3.0, seq_store.as_mut_ptr(), seq_size, 0) };
    if ncopy != seq_store.len() {
        return Err(format!(
            "seq copy data length {ncopy} does not match expected length {}",
            seq_store.len()
        ));
    }
    eprintln!("main: seq 0 copied, {ncopy} bytes");

    unsafe {
        ffi::llama_memory_clear(ffi::llama_get_memory(ctx3.0), true);
    }
    eprintln!("main: kv cache cleared");

    let nset =
        unsafe { ffi::llama_state_seq_set_data(ctx3.0, seq_store.as_ptr(), seq_store.len(), 1) };
    if nset != seq_store.len() {
        return Err(format!(
            "seq set data length {nset} does not match expected length {}",
            seq_store.len()
        ));
    }
    eprintln!("main: seq 1 restored, {nset} bytes");

    let result2 = generate(
        ctx3.0,
        vocab,
        sampler3.0,
        &mut batch.0,
        &mut n_past3,
        1,
        args.n_predict,
    )?;
    println!();

    if result0 != result2 {
        return Err("the seq restore generation is different".to_string());
    }

    eprintln!("\nmain: success");
    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv
        .next()
        .unwrap_or_else(|| "llama-save-load-state".to_string());

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
    fn parses_defaults() {
        let args = parse_args(["-m", "model.gguf"]).unwrap();
        assert_eq!(args.prompt, "The quick brown fox");
        assert_eq!(args.seed, 1234);
        assert_eq!(args.n_predict, 16);
        assert_eq!(args.n_parallel, 1);
    }

    #[test]
    fn parses_overrides_and_positional_prompt() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-n",
            "4",
            "-b",
            "64",
            "-np",
            "2",
            "-ngl",
            "0",
            "--seed",
            "42",
            "custom",
            "prompt",
        ])
        .unwrap();
        assert_eq!(args.prompt, "custom prompt");
        assert_eq!(args.n_predict, 4);
        assert_eq!(args.n_batch, 64);
        assert_eq!(args.n_parallel, 2);
        assert_eq!(args.n_gpu_layers, 0);
        assert_eq!(args.seed, 42);
    }

    #[test]
    fn negative_predict_uses_cpp_default() {
        let args = parse_args(["-m", "model.gguf", "-n", "-1"]).unwrap();
        assert_eq!(args.n_predict, 16);
    }

    #[test]
    fn unified_kv_matches_cpp_condition() {
        assert!(should_enable_unified_kv(1));
        assert!(!should_enable_unified_kv(2));
    }
}
