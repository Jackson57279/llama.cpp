use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::io::{self, Write};
use std::ptr;

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub n_predict: i32,
    pub n_parallel: i32,
    pub n_gpu_layers: i32,
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub seed: u32,
    pub backend_sampling: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    MissingModel,
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidFloat(String, String),
    InvalidParallel(i32),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m model.gguf"),
            ParseError::MissingValue(flag) => write!(f, "missing value for {flag}"),
            ParseError::InvalidInteger(flag, value) => {
                write!(f, "invalid integer for {flag}: {value}")
            }
            ParseError::InvalidFloat(flag, value) => write!(f, "invalid float for {flag}: {value}"),
            ParseError::InvalidParallel(value) => {
                write!(f, "-np/--parallel must be positive, got {value}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: "Hello my name is".to_string(),
            n_predict: 32,
            n_parallel: 1,
            n_gpu_layers: 99,
            top_k: 40,
            top_p: 0.95,
            temp: 0.8,
            seed: ffi::LLAMA_DEFAULT_SEED,
            backend_sampling: false,
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
            "-np" | "--parallel" => parsed.n_parallel = parse_i32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "--top-k" => parsed.top_k = parse_i32(&mut iter, &arg)?,
            "--top-p" => parsed.top_p = parse_f32(&mut iter, &arg)?,
            "--temp" => parsed.temp = parse_f32(&mut iter, &arg)?,
            "-s" | "--seed" => parsed.seed = parse_u32(&mut iter, &arg)?,
            "--backend-sampling" => parsed.backend_sampling = true,
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
    if parsed.n_parallel <= 0 {
        return Err(ParseError::InvalidParallel(parsed.n_parallel));
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

fn tokenize(
    vocab: *const ffi::llama_vocab,
    prompt: &str,
    prompt_c: &CString,
) -> Result<Vec<ffi::llama_token>, String> {
    let n_prompt = unsafe {
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
            true,
            true,
        )
    };
    if n_tokenized < 0 {
        return Err("failed to tokenize prompt".to_string());
    }
    tokens.truncate(n_tokenized as usize);
    Ok(tokens)
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
    *batch.logits.offset(i) = i8::from(logits);
    batch.n_tokens += 1;
}

fn make_sampler(args: &Args) -> Result<Sampler, String> {
    let mut params = unsafe { ffi::llama_sampler_chain_default_params() };
    params.no_perf = false;
    let sampler = Sampler(unsafe { ffi::llama_sampler_chain_init(params) });
    if sampler.0.is_null() {
        return Err("failed to create sampler chain".to_string());
    }

    unsafe {
        ffi::llama_sampler_chain_add(sampler.0, ffi::llama_sampler_init_top_k(args.top_k));
        ffi::llama_sampler_chain_add(sampler.0, ffi::llama_sampler_init_top_p(args.top_p, 1));
        ffi::llama_sampler_chain_add(sampler.0, ffi::llama_sampler_init_temp(args.temp));
        ffi::llama_sampler_chain_add(sampler.0, ffi::llama_sampler_init_dist(args.seed));
    }

    Ok(sampler)
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf -p \"Hello my name is\" -n 32 -np 4");
    eprintln!();
}

fn run(args: Args) -> Result<(), String> {
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;
    let prompt_c = CString::new(args.prompt.as_str())
        .map_err(|_| "prompt contains an interior NUL byte".to_string())?;

    let _backend = Backend::init();

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

    let tokens = tokenize(vocab, &args.prompt, &prompt_c)?;
    let n_parallel = args.n_parallel;
    let n_predict = args.n_predict.max(tokens.len() as i32);
    let n_kv_req = tokens.len() as i32 + (n_predict - tokens.len() as i32) * n_parallel;

    let mut sampler_storage = Vec::with_capacity(n_parallel as usize);
    let mut sampler_configs = Vec::with_capacity(n_parallel as usize);
    for i in 0..n_parallel {
        let sampler = make_sampler(&args)?;
        sampler_configs.push(ffi::llama_sampler_seq_config {
            seq_id: i,
            sampler: sampler.0,
        });
        sampler_storage.push(sampler);
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = n_kv_req.max(1) as u32;
    ctx_params.n_batch = n_predict.max(n_parallel) as u32;
    ctx_params.no_perf = false;
    if args.backend_sampling {
        ctx_params.samplers = sampler_configs.as_mut_ptr();
        ctx_params.n_samplers = sampler_configs.len();
    }

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create the llama_context".to_string());
    }

    let n_ctx = unsafe { ffi::llama_n_ctx(ctx.0) as i32 };
    eprintln!(
        "main: n_predict = {n_predict}, n_ctx = {n_ctx}, n_batch = {}, n_parallel = {n_parallel}, n_kv_req = {n_kv_req}",
        ctx_params.n_batch
    );
    if n_kv_req > n_ctx {
        return Err(format!(
            "n_kv_req ({n_kv_req}) > n_ctx; reduce n_parallel or increase n_ctx"
        ));
    }

    for &id in &tokens {
        print!("{}", token_to_piece(vocab, id)?);
    }
    io::stdout().flush().map_err(|err| err.to_string())?;

    let batch_capacity = tokens.len().max(n_parallel as usize) as i32;
    let mut batch = Batch(unsafe { ffi::llama_batch_init(batch_capacity, 0, n_parallel) });
    let seq_ids = (0..n_parallel).collect::<Vec<_>>();

    for (i, token) in tokens.iter().enumerate() {
        unsafe {
            batch_add(&mut batch.0, *token, i as ffi::llama_pos, &seq_ids, false);
        }
    }

    let mut decoder_start_token: ffi::llama_token;
    if unsafe { ffi::llama_model_has_encoder(model.0) } {
        if unsafe { ffi::llama_encode(ctx.0, batch.0) } != 0 {
            return Err("failed to eval encoder".to_string());
        }

        decoder_start_token = unsafe { ffi::llama_model_decoder_start_token(model.0) };
        if decoder_start_token == ffi::LLAMA_TOKEN_NULL {
            decoder_start_token = unsafe { ffi::llama_vocab_bos(vocab) };
        }

        batch_clear(&mut batch.0);
        unsafe {
            batch_add(&mut batch.0, decoder_start_token, 0, &seq_ids, false);
        }
    }

    if batch.0.n_tokens == 0 {
        return Err("empty prompt batch".to_string());
    }
    unsafe {
        *batch.0.logits.offset(batch.0.n_tokens as isize - 1) = 1;
    }

    if unsafe { ffi::llama_decode(ctx.0, batch.0) } != 0 {
        return Err("llama_decode failed".to_string());
    }

    if n_parallel > 1 {
        eprintln!("\nmain: generating {n_parallel} sequences ...");
    }

    let mut streams = vec![String::new(); n_parallel as usize];
    let mut i_batch = vec![batch.0.n_tokens - 1; n_parallel as usize];
    let mut n_cur = batch.0.n_tokens;
    let mut n_decode = 0;
    let t_main_start = unsafe { ffi::ggml_time_us() };

    while n_cur <= n_predict {
        batch_clear(&mut batch.0);

        for i in 0..n_parallel as usize {
            if i_batch[i] < 0 {
                continue;
            }

            let new_token_id =
                unsafe { ffi::llama_sampler_sample(sampler_configs[i].sampler, ctx.0, i_batch[i]) };

            if unsafe { ffi::llama_vocab_is_eog(vocab, new_token_id) } || n_cur == n_predict {
                i_batch[i] = -1;
                println!();
                if n_parallel > 1 {
                    eprintln!("main: stream {i} finished at n_cur = {n_cur}");
                }
                continue;
            }

            let piece = token_to_piece(vocab, new_token_id)?;
            if n_parallel == 1 {
                print!("{piece}");
                io::stdout().flush().map_err(|err| err.to_string())?;
            }
            streams[i].push_str(&piece);
            i_batch[i] = batch.0.n_tokens;

            unsafe {
                batch_add(
                    &mut batch.0,
                    new_token_id,
                    n_cur,
                    &[i as ffi::llama_seq_id],
                    true,
                );
            }
            n_decode += 1;
        }

        if batch.0.n_tokens == 0 {
            break;
        }

        n_cur += 1;

        if unsafe { ffi::llama_decode(ctx.0, batch.0) } != 0 {
            return Err("failed to eval".to_string());
        }
    }

    if n_parallel > 1 {
        println!();
        for (i, stream) in streams.iter().enumerate() {
            println!("sequence {i}:\n\n{}{stream}\n", args.prompt);
        }
    }

    let t_main_end = unsafe { ffi::ggml_time_us() };
    let elapsed = (t_main_end - t_main_start) as f32 / 1_000_000.0;
    eprintln!(
        "main: decoded {n_decode} tokens in {elapsed:.2} s, speed: {:.2} t/s",
        n_decode as f32 / elapsed.max(f32::EPSILON)
    );
    eprintln!();
    unsafe {
        ffi::llama_perf_sampler_print(sampler_configs[0].sampler);
        ffi::llama_perf_context_print(ctx.0);
    }
    eprintln!();

    drop(sampler_storage);
    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-batched".to_string());

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
        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "Hello my name is");
        assert_eq!(args.n_predict, 32);
        assert_eq!(args.n_parallel, 1);
    }

    #[test]
    fn parses_generation_options() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-p",
            "Hi",
            "-n",
            "8",
            "-np",
            "4",
            "-ngl",
            "0",
            "--top-k",
            "16",
            "--top-p",
            "0.7",
            "--temp",
            "0.2",
            "--seed",
            "42",
            "--backend-sampling",
        ])
        .unwrap();
        assert_eq!(args.prompt, "Hi");
        assert_eq!(args.n_predict, 8);
        assert_eq!(args.n_parallel, 4);
        assert_eq!(args.n_gpu_layers, 0);
        assert_eq!(args.top_k, 16);
        assert_eq!(args.top_p, 0.7);
        assert_eq!(args.temp, 0.2);
        assert_eq!(args.seed, 42);
        assert!(args.backend_sampling);
    }

    #[test]
    fn accepts_positional_prompt_tail() {
        let args = parse_args(["-m", "model.gguf", "hello", "there"]).unwrap();
        assert_eq!(args.prompt, "hello there");
    }

    #[test]
    fn rejects_missing_model() {
        assert_eq!(
            parse_args(["-n", "8"]).unwrap_err(),
            ParseError::MissingModel
        );
    }

    #[test]
    fn rejects_invalid_parallel() {
        assert_eq!(
            parse_args(["-m", "model.gguf", "-np", "0"]).unwrap_err(),
            ParseError::InvalidParallel(0)
        );
    }
}
