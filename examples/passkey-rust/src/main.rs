use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::io::{self, Write};
use std::ptr;

const PROMPT_PREFIX: &str = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.";
const PROMPT_SUFFIX: &str = " What is the pass key? The pass key is";
const JUNK_SENTENCE: &str =
    " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub n_junk: i32,
    pub n_keep: i32,
    pub n_grp: i32,
    pub i_pos: i32,
    pub n_batch: i32,
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
            n_junk: 250,
            n_keep: 32,
            n_grp: 1,
            i_pos: -1,
            n_batch: 512,
            n_gpu_layers: 99,
            seed: 1,
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
            "--junk" => parsed.n_junk = parse_i32(&mut iter, &arg)?,
            "--keep" => parsed.n_keep = parse_i32(&mut iter, &arg)?,
            "--grp-attn-n" => parsed.n_grp = parse_i32(&mut iter, &arg)?,
            "--pos" => parsed.i_pos = parse_i32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_i32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "-s" | "--seed" => parsed.seed = parse_u32(&mut iter, &arg)?,
            "-h" | "--help" => return Err(ParseError::MissingModel),
            _ => {}
        }
    }

    if parsed.model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }
    validate_args(&parsed)?;
    Ok(parsed)
}

fn validate_args(args: &Args) -> Result<(), ParseError> {
    if args.n_junk <= 0 {
        return Err(ParseError::InvalidValue(
            "--junk must be positive".to_string(),
        ));
    }
    if args.n_keep < 0 {
        return Err(ParseError::InvalidValue(
            "--keep must be non-negative".to_string(),
        ));
    }
    if args.n_grp <= 0 {
        return Err(ParseError::InvalidValue(
            "--grp-attn-n must be positive".to_string(),
        ));
    }
    if args.n_batch <= 0 {
        return Err(ParseError::InvalidValue(
            "--batch-size must be positive".to_string(),
        ));
    }
    if args.n_batch % args.n_grp != 0 {
        return Err(ParseError::InvalidValue(
            "--batch-size must be divisible by --grp-attn-n".to_string(),
        ));
    }
    if args.i_pos >= args.n_junk {
        return Err(ParseError::InvalidValue(
            "--pos must be less than --junk".to_string(),
        ));
    }
    Ok(())
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

fn lcg_next(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

pub fn build_prompt(n_junk: i32, i_pos: i32, seed: u32) -> (String, i32, i32) {
    let mut rng = seed;
    let pos = if i_pos < 0 {
        (lcg_next(&mut rng) % n_junk as u32) as i32
    } else {
        i_pos
    };
    let passkey = (lcg_next(&mut rng) % 50_000 + 1) as i32;

    let mut prompt = String::from(PROMPT_PREFIX);
    for i in 0..n_junk {
        if i == pos {
            prompt.push_str(&format!(
                " The pass key is {passkey}. Remember it. {passkey} is the pass key."
            ));
        }
        prompt.push_str(JUNK_SENTENCE);
    }
    prompt.push_str(PROMPT_SUFFIX);

    (prompt, passkey, pos)
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

fn tokenize(
    vocab: *const ffi::llama_vocab,
    text: &str,
    add_special: bool,
) -> Result<Vec<ffi::llama_token>, String> {
    let text_c =
        CString::new(text).map_err(|_| "text contains an interior NUL byte".to_string())?;
    let n_tokens = unsafe {
        -ffi::llama_tokenize(
            vocab,
            text_c.as_ptr(),
            text.len() as i32,
            ptr::null_mut(),
            0,
            add_special,
            true,
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

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!(
        "    {program} -m model.gguf --junk 250 --pos 90 --keep 32 --grp-attn-n 2 [--seed 1234]"
    );
    eprintln!();
}

fn run(args: Args) -> Result<(), String> {
    let _backend = Backend::init();
    let (prompt, passkey, i_pos) = build_prompt(args.n_junk, args.i_pos, args.seed);

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

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx =
        (unsafe { ffi::llama_model_n_ctx_train(model.0) } * args.n_grp + args.n_keep).max(1) as u32;
    ctx_params.n_batch = args.n_batch as u32;
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
    unsafe {
        ffi::llama_sampler_chain_add(sampler.0, ffi::llama_sampler_init_greedy());
    }

    let tokens = tokenize(vocab, &prompt, true)?;
    let n_tokens_prefix = tokenize(vocab, PROMPT_PREFIX, true)?.len() as i32;
    let n_tokens_all = tokens.len() as i32;
    let n_predict = 16;
    let n_len = n_tokens_all + n_predict;
    let n_ctx = unsafe { ffi::llama_n_ctx(ctx.0) as i32 } - args.n_keep;
    let n_kv_req = unsafe { ffi::llama_n_ctx(ctx.0) as i32 };
    let n_batch = args.n_batch;
    let n_batch_grp = args.n_batch / args.n_grp;

    eprintln!(
        "main: n_len = {n_len}, n_ctx = {n_ctx}, n_kv_req = {n_kv_req}, n_grp = {}, n_batch = {n_batch}, n_junk = {}, i_pos = {i_pos}",
        args.n_grp, args.n_junk
    );
    eprintln!();
    eprintln!("prefix tokens: {n_tokens_prefix}");
    eprintln!("prompt tokens: {n_tokens_all}");

    let mut batch = Batch(unsafe { ffi::llama_batch_init(args.n_batch, 0, 1) });
    let mut n_past = 0;
    let mem = unsafe { ffi::llama_get_memory(ctx.0) };

    let mut i = 0;
    while i < n_ctx {
        if i > 0 && args.n_grp > 1 {
            let ib = i / n_batch - 1;
            let bd = n_batch_grp * (args.n_grp - 1);
            unsafe {
                ffi::llama_memory_seq_add(mem, 0, n_past - n_batch, n_past, ib * bd);
                ffi::llama_memory_seq_div(
                    mem,
                    0,
                    n_past - n_batch + ib * bd,
                    n_past + ib * bd,
                    args.n_grp,
                );
                n_past = ffi::llama_memory_seq_pos_max(mem, 0) + 1;
            }
        }

        batch_clear(&mut batch.0);
        for j in 0..n_batch {
            if i + j >= n_tokens_all {
                break;
            }
            unsafe {
                batch_add(&mut batch.0, tokens[(i + j) as usize], n_past, 0, false);
            }
            n_past += 1;
        }

        if i + n_batch >= n_tokens_all {
            unsafe {
                *batch.0.logits.offset(batch.0.n_tokens as isize - 1) = 1;
            }
        }

        if unsafe { ffi::llama_decode(ctx.0, batch.0) } != 0 {
            return Err("llama_decode failed".to_string());
        }

        eprintln!(
            "main: processed: [{:6}, {:6})",
            i,
            (i + n_batch).min(n_tokens_all)
        );
        if i + n_batch >= n_tokens_all {
            break;
        }
        i += n_batch;
    }

    let mut i = n_ctx;
    while i < n_tokens_all {
        let n_discard = n_batch;
        eprintln!("main: shifting KV cache with {n_discard}");
        unsafe {
            ffi::llama_memory_seq_rm(mem, 0, args.n_keep, args.n_keep + n_discard);
            ffi::llama_memory_seq_add(mem, 0, args.n_keep + n_discard, n_ctx, -n_discard);
            n_past = ffi::llama_memory_seq_pos_max(mem, 0) + 1;
        }

        batch_clear(&mut batch.0);
        for j in 0..n_batch {
            if i + j >= n_tokens_all {
                break;
            }
            unsafe {
                batch_add(&mut batch.0, tokens[(i + j) as usize], n_past, 0, false);
            }
            n_past += 1;
        }
        if i + n_batch >= n_tokens_all {
            unsafe {
                *batch.0.logits.offset(batch.0.n_tokens as isize - 1) = 1;
            }
        }
        if unsafe { ffi::llama_decode(ctx.0, batch.0) } != 0 {
            return Err("llama_decode failed".to_string());
        }
        eprintln!(
            "main: processed: [{:6}, {:6})",
            i,
            (i + n_batch).min(n_tokens_all)
        );
        i += n_batch;
    }

    let n_discard = n_past - n_ctx + n_predict;
    if n_discard > 0 {
        eprintln!("main: shifting KV cache with {n_discard} to free space for the answer");
        unsafe {
            ffi::llama_memory_seq_rm(mem, 0, args.n_keep, args.n_keep + n_discard);
            ffi::llama_memory_seq_add(mem, 0, args.n_keep + n_discard, n_ctx, -n_discard);
            n_past = ffi::llama_memory_seq_pos_max(mem, 0) + 1;
        }
    }

    eprintln!();
    eprintln!(
        "main: passkey = {passkey}, inserted at position {i_pos} / {} (token pos: ~{})",
        args.n_junk,
        (i_pos * n_tokens_all) / args.n_junk
    );
    eprintln!();

    let mut n_cur = n_tokens_all;
    let mut n_decode = 0;

    print!("{PROMPT_SUFFIX}");
    io::stdout().flush().map_err(|err| err.to_string())?;

    let t_main_start = unsafe { ffi::ggml_time_us() };
    while n_cur <= n_len {
        let new_token_id =
            unsafe { ffi::llama_sampler_sample(sampler.0, ctx.0, batch.0.n_tokens - 1) };
        if unsafe { ffi::llama_vocab_is_eog(vocab, new_token_id) } || n_cur == n_len {
            println!();
            break;
        }

        print!("{}", token_to_piece(vocab, new_token_id)?);
        io::stdout().flush().map_err(|err| err.to_string())?;
        n_decode += 1;

        batch_clear(&mut batch.0);
        unsafe {
            batch_add(&mut batch.0, new_token_id, n_past, 0, true);
        }
        n_past += 1;
        n_cur += 1;

        if unsafe { ffi::llama_decode(ctx.0, batch.0) } != 0 {
            return Err("failed to eval".to_string());
        }
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
        ffi::llama_perf_context_print(ctx.0);
    }
    eprintln!();

    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-passkey".to_string());

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
        assert_eq!(args.n_junk, 250);
        assert_eq!(args.n_keep, 32);
        assert_eq!(args.n_grp, 1);
        assert_eq!(args.i_pos, -1);
        assert_eq!(args.n_batch, 512);
    }

    #[test]
    fn parses_passkey_options() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "--junk",
            "20",
            "--pos",
            "9",
            "--keep",
            "16",
            "--grp-attn-n",
            "2",
            "-b",
            "64",
            "--seed",
            "1234",
            "-ngl",
            "0",
        ])
        .unwrap();
        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.n_junk, 20);
        assert_eq!(args.i_pos, 9);
        assert_eq!(args.n_keep, 16);
        assert_eq!(args.n_grp, 2);
        assert_eq!(args.n_batch, 64);
        assert_eq!(args.seed, 1234);
        assert_eq!(args.n_gpu_layers, 0);
    }

    #[test]
    fn validates_group_divides_batch() {
        assert_eq!(
            parse_args(["-m", "model.gguf", "--grp-attn-n", "3", "-b", "64"]).unwrap_err(),
            ParseError::InvalidValue("--batch-size must be divisible by --grp-attn-n".to_string())
        );
    }

    #[test]
    fn builds_prompt_with_inserted_passkey() {
        let (prompt, passkey, pos) = build_prompt(3, 1, 7);
        assert_eq!(pos, 1);
        assert!(prompt.starts_with(PROMPT_PREFIX));
        assert!(prompt.ends_with(PROMPT_SUFFIX));
        assert!(prompt.contains(&format!("The pass key is {passkey}.")));
        assert_eq!(prompt.matches(JUNK_SENTENCE).count(), 3);
    }

    #[test]
    fn seeded_random_position_is_stable() {
        let (_, passkey_a, pos_a) = build_prompt(250, -1, 1234);
        let (_, passkey_b, pos_b) = build_prompt(250, -1, 1234);
        assert_eq!((passkey_a, pos_a), (passkey_b, pos_b));
    }
}
