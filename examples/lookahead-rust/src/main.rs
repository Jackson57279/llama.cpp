use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::io::{self, Write};
use std::ptr;

const W: usize = 15;
const N: usize = 5;
const G: usize = 15;

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub n_predict: i32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_gpu_layers: i32,
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub seed: u32,
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

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: "Hello my name is".to_string(),
            n_predict: 32,
            n_ctx: 512,
            n_batch: 512,
            n_gpu_layers: 99,
            top_k: 40,
            top_p: 0.95,
            temp: 0.8,
            seed: ffi::LLAMA_DEFAULT_SEED,
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
            "-c" | "--ctx-size" => parsed.n_ctx = parse_u32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_u32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "--top-k" => parsed.top_k = parse_i32(&mut iter, &arg)?,
            "--top-p" => parsed.top_p = parse_f32(&mut iter, &arg)?,
            "--temp" => parsed.temp = parse_f32(&mut iter, &arg)?,
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

fn parse_f32<I>(iter: &mut I, flag: &str) -> Result<f32, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidFloat(flag.to_string(), value))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramData {
    pub active: bool,
    pub seq_id: ffi::llama_seq_id,
    pub i_batch: Vec<i32>,
    pub tokens: Vec<ffi::llama_token>,
}

impl Default for NgramData {
    fn default() -> Self {
        Self {
            active: false,
            seq_id: -1,
            i_batch: Vec::new(),
            tokens: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramContainer {
    pub n_total: usize,
    pub n_vocab: usize,
    pub n: usize,
    pub g: usize,
    pub cnt: Vec<usize>,
    pub head: Vec<usize>,
    pub tokens: Vec<ffi::llama_token>,
}

impl NgramContainer {
    pub fn new(n_vocab: usize, n: usize, g: usize) -> Self {
        Self {
            n_total: 0,
            n_vocab,
            n,
            g,
            cnt: vec![0; n_vocab],
            head: vec![0; n_vocab],
            tokens: vec![0; n_vocab * g * (n - 1)],
        }
    }

    pub fn observed_for(&self, token: ffi::llama_token) -> usize {
        if token < 0 || token as usize >= self.n_vocab {
            0
        } else {
            self.cnt[token as usize]
        }
    }

    pub fn get(&self, token: ffi::llama_token, g: usize, j: usize) -> ffi::llama_token {
        let token = token as usize;
        self.tokens[token * (self.n - 1) * self.g + g * (self.n - 1) + j]
    }

    pub fn insert_unique(&mut self, first: ffi::llama_token, ngram: &[ffi::llama_token]) -> bool {
        if first < 0 || first as usize >= self.n_vocab || ngram.len() != self.n - 1 {
            return false;
        }
        let first = first as usize;
        for k in 0..self.cnt[first] {
            let idx = first * (self.n - 1) * self.g + k * (self.n - 1);
            if self.tokens[idx..idx + self.n - 1] == *ngram {
                return false;
            }
        }

        let head = self.head[first];
        let idx = first * (self.n - 1) * self.g + head * (self.n - 1);
        self.tokens[idx..idx + self.n - 1].copy_from_slice(ngram);
        self.cnt[first] = self.g.min(self.cnt[first] + 1);
        self.head[first] = (head + 1) % self.g;
        self.n_total += 1;
        true
    }
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

fn sample_accept(
    sampler: *mut ffi::llama_sampler,
    ctx: *mut ffi::llama_context,
    idx: i32,
) -> ffi::llama_token {
    unsafe {
        let token = ffi::llama_sampler_sample(sampler, ctx, idx);
        ffi::llama_sampler_accept(sampler, token);
        token
    }
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

fn decode_tokens(
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
    ctx_params.n_seq_max = (W + G + 1) as u32;
    ctx_params.kv_unified = true;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create llama_context".to_string());
    }

    let mem = unsafe { ffi::llama_get_memory(ctx.0) };
    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        return Err("failed to get model vocabulary".to_string());
    }

    let mut inp = tokenize(vocab, &args.prompt, true, true)?;
    let mut all = inp.clone();
    let max_tokens_list_size = unsafe { ffi::llama_n_ctx(ctx.0) as usize }.saturating_sub(4);
    if inp.len() > max_tokens_list_size {
        return Err(format!(
            "prompt too long ({} tokens, max {max_tokens_list_size})",
            inp.len()
        ));
    }
    if inp.is_empty() {
        return Err("prompt produced no tokens".to_string());
    }

    println!();
    for token in &inp {
        print!("{}", token_to_piece(vocab, *token)?);
    }
    io::stdout().flush().map_err(|err| err.to_string())?;

    let n_input = inp.len();
    let t_enc_start = unsafe { ffi::ggml_time_us() };
    decode_tokens(ctx.0, &mut inp)?;
    for s in 1..(W + G + 1) {
        unsafe {
            ffi::llama_memory_seq_cp(mem, 0, s as i32, -1, -1);
        }
    }
    let t_enc_end = unsafe { ffi::ggml_time_us() };

    let mut n_predict = 0;
    let mut n_accept = 0;
    let mut n_past = inp.len() as i32;
    let mut has_eos = false;
    let sampler = sampler_init(&args);
    let mut batch = Batch(unsafe {
        ffi::llama_batch_init(ffi::llama_n_ctx(ctx.0) as i32, 0, (W + G + 1) as i32)
    });

    let mut ngrams_cur = vec![NgramData::default(); G];
    let mut tokens_j_prev = vec![0; W];
    let n_vocab = unsafe { ffi::llama_vocab_n_tokens(vocab) as usize };
    let mut tokens_j = vec![vec![0; W]; N - 1];
    for row in tokens_j.iter_mut().take(N - 1) {
        for (i, value) in row.iter_mut().enumerate().take(W) {
            *value = (100 + i) as i32;
            if n_vocab > 0 && *value as usize >= n_vocab {
                *value = (n_vocab - 1) as i32;
            }
        }
    }

    let seq_id_all = (0..(W + G + 1)).map(|i| i as i32).collect::<Vec<_>>();
    let mut ngrams_observed = NgramContainer::new(n_vocab, N, G);
    let t_dec_start = unsafe { ffi::ggml_time_us() };

    let mut id = sample_accept(sampler.0, ctx.0, 0);
    print!("{}", token_to_piece(vocab, id)?);
    io::stdout().flush().map_err(|err| err.to_string())?;

    loop {
        batch_clear(&mut batch.0);
        unsafe {
            batch_add(&mut batch.0, id, n_past, &seq_id_all, true);
        }

        let g_cur = ngrams_observed.observed_for(id);
        ngrams_cur.resize(g_cur, NgramData::default());
        for (g, item) in ngrams_cur.iter_mut().enumerate().take(g_cur) {
            item.active = true;
            item.tokens = vec![0; N];
            item.i_batch = vec![0; N];
            item.seq_id = (W + 1 + g) as i32;
            item.i_batch[0] = 0;
            item.tokens[0] = id;
        }
        for j in 0..N - 1 {
            for (g, item) in ngrams_cur.iter_mut().enumerate().take(g_cur) {
                let token = ngrams_observed.get(id, g, j);
                item.tokens[j + 1] = token;
                item.i_batch[j + 1] = batch.0.n_tokens;
                unsafe {
                    batch_add(
                        &mut batch.0,
                        token,
                        n_past + j as i32 + 1,
                        &[(W + 1 + g) as i32],
                        true,
                    );
                }
            }
        }

        for i in 1..W {
            let seq_id_look = ((i + 1)..=W).map(|s| s as i32).collect::<Vec<_>>();
            unsafe {
                batch_add(
                    &mut batch.0,
                    tokens_j[0][i],
                    n_past + i as i32,
                    &seq_id_look,
                    false,
                );
            }
        }

        for j in 1..N - 1 {
            for i in 0..W {
                unsafe {
                    batch_add(
                        &mut batch.0,
                        tokens_j[j][i],
                        n_past + j as i32 + i as i32,
                        &[(i + 1) as i32],
                        j == N - 2,
                    );
                }
            }
        }

        if unsafe { ffi::llama_decode(ctx.0, batch.0) } != 0 {
            return Err("llama_decode failed - increase KV cache size".to_string());
        }

        let mut seq_id_best = 0;
        for v in 0..N {
            let mut i_batch = 0;
            if v > 0 {
                for item in &ngrams_cur {
                    if item.active {
                        i_batch = item.i_batch[v];
                        seq_id_best = item.seq_id;
                        n_accept += 1;
                        break;
                    }
                }
                if i_batch == 0 {
                    break;
                }
            }

            id = sample_accept(sampler.0, ctx.0, i_batch);
            let token_str = token_to_piece(vocab, id)?;
            if v == 0 {
                print!("{token_str}");
            } else {
                print!("\x1b[0;96m{token_str}\x1b[0m");
            }
            io::stdout().flush().map_err(|err| err.to_string())?;
            if unsafe { ffi::llama_vocab_is_eog(vocab, id) } {
                has_eos = true;
            }
            all.push(id);
            n_predict += 1;
            n_past += 1;

            if (args.n_predict >= 0 && n_predict > args.n_predict) || has_eos {
                break;
            }

            for item in &mut ngrams_cur {
                if item.active {
                    item.active = v != N - 1 && id == item.tokens[v + 1];
                }
            }

            tokens_j_prev.copy_from_slice(&tokens_j[0]);
            for j in 0..N - 2 {
                tokens_j[j] = tokens_j[j + 1].clone();
            }
            if v == 0 {
                let base = ngrams_cur.len() * (N - 1) + W * (N - 2);
                for i in 0..W {
                    tokens_j[N - 2][i] =
                        unsafe { ffi::llama_sampler_sample(sampler.0, ctx.0, (base + i) as i32) };
                }
            } else {
                tokens_j[N - 2] = tokens_j[0].clone();
            }

            if v == 0 {
                for f in 0..W {
                    let first = tokens_j_prev[f];
                    let ngram = (0..N - 1).map(|j| tokens_j[j][f]).collect::<Vec<_>>();
                    ngrams_observed.insert_unique(first, &ngram);
                }
            }
        }

        if (args.n_predict >= 0 && n_predict > args.n_predict) || has_eos {
            break;
        }

        unsafe {
            ffi::llama_memory_seq_rm(mem, -1, n_past, -1);
            if seq_id_best != 0 {
                ffi::llama_memory_seq_keep(mem, seq_id_best);
                ffi::llama_memory_seq_cp(mem, seq_id_best, 0, -1, -1);
                ffi::llama_memory_seq_rm(mem, seq_id_best, -1, -1);
                for s in 1..(W + G + 1) {
                    ffi::llama_memory_seq_cp(mem, 0, s as i32, -1, -1);
                }
            }
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
    eprintln!("\nW = {W:2}\nN = {N:2}\nG = {G:2}\n");
    eprintln!("n_predict = {n_predict}");
    eprintln!("n_accept  = {n_accept}\n");
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
    eprintln!("    {program} -m model.gguf -p \"Hello my name is\"");
    eprintln!();
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-lookahead".to_string());
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
    fn parses_lookahead_options() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-p",
            "hello",
            "-n",
            "8",
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
        ])
        .unwrap();
        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.n_predict, 8);
        assert_eq!(args.n_ctx, 256);
        assert_eq!(args.n_batch, 128);
        assert_eq!(args.n_gpu_layers, 0);
        assert_eq!(args.top_k, 10);
        assert_eq!(args.top_p, 0.8);
        assert_eq!(args.temp, 0.5);
        assert_eq!(args.seed, 42);
    }

    #[test]
    fn rejects_missing_model() {
        assert_eq!(
            parse_args(["-p", "hello"]).unwrap_err(),
            ParseError::MissingModel
        );
    }

    #[test]
    fn ngram_container_keeps_unique_ring_entries() {
        let mut container = NgramContainer::new(8, 3, 2);
        assert!(container.insert_unique(2, &[3, 4]));
        assert!(!container.insert_unique(2, &[3, 4]));
        assert!(container.insert_unique(2, &[4, 5]));
        assert!(container.insert_unique(2, &[5, 6]));
        assert_eq!(container.cnt[2], 2);
        assert_eq!(container.n_total, 3);
        assert_eq!(container.observed_for(2), 2);
    }
}
