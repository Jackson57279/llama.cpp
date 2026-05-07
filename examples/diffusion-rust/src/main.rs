use llama_simple_rust::ffi;
use std::env;
use std::ffi::{CStr, CString};
use std::io::{self, Write};
use std::ptr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionAlgorithm {
    Origin = 0,
    EntropyBased = 1,
    MarginBased = 2,
    Random = 3,
    ConfidenceBased = 4,
}

impl TryFrom<i32> for DiffusionAlgorithm {
    type Error = ParseError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Origin),
            1 => Ok(Self::EntropyBased),
            2 => Ok(Self::MarginBased),
            3 => Ok(Self::Random),
            4 => Ok(Self::ConfidenceBased),
            _ => Err(ParseError::InvalidValue(format!(
                "diffusion algorithm must be in 0..=4, got {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferSchedule {
    TimestepBased,
    BlockBased,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub system_prompt: String,
    pub use_chat_template: bool,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub n_gpu_layers: i32,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub no_perf: bool,
    pub flash_attn_type: i32,
    pub cache_type_k: ffi::ggml_type,
    pub cache_type_v: ffi::ggml_type,
    pub seed: u32,
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub steps: i32,
    pub eps: f32,
    pub block_length: i32,
    pub algorithm: DiffusionAlgorithm,
    pub visual_mode: bool,
    pub add_gumbel_noise: bool,
    pub cfg_scale: f32,
    pub alg_temp: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: String::new(),
            system_prompt: String::new(),
            use_chat_template: false,
            n_ctx: 512,
            n_batch: 512,
            n_ubatch: 512,
            n_threads: 4,
            n_threads_batch: 4,
            n_gpu_layers: 99,
            use_mmap: true,
            use_mlock: false,
            check_tensors: false,
            no_perf: false,
            flash_attn_type: 0,
            cache_type_k: 0,
            cache_type_v: 0,
            seed: ffi::LLAMA_DEFAULT_SEED,
            top_k: 40,
            top_p: 0.95,
            temp: 0.0,
            steps: 64,
            eps: 1e-3,
            block_length: 0,
            algorithm: DiffusionAlgorithm::ConfidenceBased,
            visual_mode: false,
            add_gumbel_noise: false,
            cfg_scale: 0.0,
            alg_temp: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    MissingModel,
    MissingPrompt,
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidFloat(String, String),
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingModel => write!(f, "missing required -m/--model model.gguf"),
            Self::MissingPrompt => write!(f, "missing prompt; pass -p/--prompt or trailing text"),
            Self::MissingValue(flag) => write!(f, "missing value for {flag}"),
            Self::InvalidInteger(flag, value) => write!(f, "invalid integer for {flag}: {value}"),
            Self::InvalidFloat(flag, value) => write!(f, "invalid float for {flag}: {value}"),
            Self::InvalidValue(message) => write!(f, "{message}"),
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
            "--system-prompt" => parsed.system_prompt = take(&mut iter, &arg)?,
            "--chat-template" | "--conversation" => parsed.use_chat_template = true,
            "--no-chat-template" => parsed.use_chat_template = false,
            "-c" | "--ctx-size" => parsed.n_ctx = parse_u32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_u32(&mut iter, &arg)?,
            "-ub" | "--ubatch-size" => parsed.n_ubatch = parse_u32(&mut iter, &arg)?,
            "-t" | "--threads" => parsed.n_threads = parse_i32(&mut iter, &arg)?,
            "-tb" | "--threads-batch" => parsed.n_threads_batch = parse_i32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "--no-mmap" => parsed.use_mmap = false,
            "--mlock" => parsed.use_mlock = true,
            "--check-tensors" => parsed.check_tensors = true,
            "--no-perf" => parsed.no_perf = true,
            "-fa" | "--flash-attn" => parsed.flash_attn_type = 1,
            "--no-flash-attn" => parsed.flash_attn_type = 0,
            "-s" | "--seed" => parsed.seed = parse_u32(&mut iter, &arg)?,
            "--top-k" => parsed.top_k = parse_i32(&mut iter, &arg)?,
            "--top-p" => parsed.top_p = parse_f32(&mut iter, &arg)?,
            "--temp" => parsed.temp = parse_f32(&mut iter, &arg)?,
            "--diffusion-steps" | "--steps" => parsed.steps = parse_i32(&mut iter, &arg)?,
            "--diffusion-eps" | "--eps" => {
                parsed.eps = parse_f32(&mut iter, &arg)?;
                parsed.block_length = 0;
            }
            "--diffusion-block-length" | "--block-length" => {
                parsed.block_length = parse_i32(&mut iter, &arg)?;
                parsed.eps = 0.0;
            }
            "--diffusion-algorithm" | "--algorithm" => {
                parsed.algorithm = DiffusionAlgorithm::try_from(parse_i32(&mut iter, &arg)?)?
            }
            "--diffusion-visual" | "--visual" => parsed.visual_mode = true,
            "--diffusion-gumbel" | "--gumbel" => parsed.add_gumbel_noise = true,
            "--cfg-scale" => parsed.cfg_scale = parse_f32(&mut iter, &arg)?,
            "--alg-temp" => parsed.alg_temp = parse_f32(&mut iter, &arg)?,
            "-h" | "--help" => return Err(ParseError::MissingModel),
            other if other.starts_with('-') => {
                return Err(ParseError::InvalidValue(format!(
                    "unknown argument: {other}"
                )));
            }
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
    if parsed.prompt.is_empty() {
        return Err(ParseError::MissingPrompt);
    }
    if parsed.n_ctx == 0 || parsed.n_batch == 0 || parsed.n_ubatch == 0 {
        return Err(ParseError::InvalidValue(
            "context and batch sizes must be positive".to_string(),
        ));
    }
    if parsed.steps <= 0 {
        return Err(ParseError::InvalidValue(
            "--steps must be positive".to_string(),
        ));
    }
    if (parsed.eps == 0.0) == (parsed.block_length == 0) {
        return Err(ParseError::InvalidValue(
            "set exactly one of --eps or --block-length".to_string(),
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

#[derive(Clone)]
struct SmallRng(u64);

impl SmallRng {
    fn new(seed: u32) -> Self {
        Self((seed as u64).wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64 + 1.0)) as f32
    }
}

fn calculate_confidence(
    cur_p: &[ffi::llama_token_data],
    selected: usize,
    algorithm: DiffusionAlgorithm,
    rng: &mut SmallRng,
) -> f32 {
    match algorithm {
        DiffusionAlgorithm::ConfidenceBased | DiffusionAlgorithm::Origin => cur_p[selected].p,
        DiffusionAlgorithm::EntropyBased => -cur_p
            .iter()
            .map(|candidate| {
                let p = candidate.p.max(0.0);
                p * (p + 1e-10).ln()
            })
            .sum::<f32>(),
        DiffusionAlgorithm::MarginBased => {
            if cur_p.len() > 1 {
                cur_p[0].p - cur_p[1].p
            } else {
                cur_p[0].p
            }
        }
        DiffusionAlgorithm::Random => rng.next_f32(),
    }
}

pub fn calculate_transfer_count(
    step: i32,
    total_steps: i32,
    remaining_masked: i32,
    schedule: TransferSchedule,
    eps: f32,
    num_transfer_tokens: &[i32],
) -> i32 {
    match schedule {
        TransferSchedule::TimestepBased => {
            let t = 1.0 - step as f32 / total_steps as f32 * (1.0 - eps);
            let s = 1.0 - (step + 1) as f32 / total_steps as f32 * (1.0 - eps);
            let p_transfer = if step < total_steps - 1 {
                1.0 - s / t
            } else {
                1.0
            };
            (remaining_masked as f32 * p_transfer) as i32
        }
        TransferSchedule::BlockBased => num_transfer_tokens
            .get(step as usize)
            .copied()
            .unwrap_or_else(|| remaining_masked / (total_steps - step).max(1)),
    }
}

pub fn get_num_transfer_tokens(mask_count: i32, steps: i32) -> Vec<i32> {
    let base = mask_count / steps;
    let remainder = mask_count % steps;
    (0..steps)
        .map(|i| base + if i < remainder { 1 } else { 0 })
        .collect()
}

fn gumbel_adjust(logit: f32, temperature: f32, rng: &mut SmallRng) -> f32 {
    if temperature == 0.0 {
        return logit;
    }
    let noise = rng.next_f32().max(1e-20) as f64;
    let gumbel_noise = (-noise.ln()).powf(temperature as f64);
    (f64::from(logit).exp() / gumbel_noise) as f32
}

fn fill_candidates_from_logits(
    candidates: &mut [ffi::llama_token_data],
    ctx: *mut ffi::llama_context,
    cond_logits: Option<&[f32]>,
    n_vocab: usize,
    pos: usize,
    shift_logits: bool,
    add_noise: bool,
    temperature: f32,
    rng: &mut SmallRng,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual_pos = if shift_logits && pos > 0 {
        pos - 1
    } else {
        pos
    };
    let src = if let Some(cond_logits) = cond_logits {
        &cond_logits[actual_pos * n_vocab..(actual_pos + 1) * n_vocab]
    } else {
        let ptr = unsafe { ffi::llama_get_logits_ith(ctx, actual_pos as i32) };
        if ptr.is_null() {
            return Err(format!("failed to get logits for position {actual_pos}").into());
        }
        unsafe { std::slice::from_raw_parts(ptr, n_vocab) }
    };

    for (token_id, candidate) in candidates.iter_mut().enumerate() {
        let mut logit = src[token_id];
        if add_noise && temperature > 0.0 {
            logit = gumbel_adjust(logit, temperature, rng);
        }
        candidate.id = token_id as i32;
        candidate.logit = logit;
        candidate.p = 0.0;
    }
    Ok(())
}

fn sampler_init(args: &Args) -> Sampler {
    unsafe {
        let chain = ffi::llama_sampler_chain_init(ffi::llama_sampler_chain_default_params());
        if args.top_k > 0 {
            ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_top_k(args.top_k));
        }
        if args.top_p < 1.0 {
            ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_top_p(args.top_p, 1));
        }
        if args.temp > 0.0 {
            ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_temp(args.temp));
        }
        ffi::llama_sampler_chain_add(chain, ffi::llama_sampler_init_dist(args.seed));
        Sampler(chain)
    }
}

fn dist_sampler_init(seed: u32) -> Sampler {
    unsafe { Sampler(ffi::llama_sampler_init_dist(seed)) }
}

fn token_to_piece(vocab: *const ffi::llama_vocab, token: ffi::llama_token) -> String {
    let mut buf = vec![0_i8; 256];
    let n = unsafe { ffi::llama_token_to_piece(vocab, token, buf.as_mut_ptr(), 256, 0, false) };
    if n <= 0 {
        String::new()
    } else {
        String::from_utf8_lossy(unsafe {
            std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), n as usize)
        })
        .into_owned()
    }
}

fn detokenize(vocab: *const ffi::llama_vocab, tokens: &[ffi::llama_token]) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    let mut cap = tokens.len().saturating_mul(16).max(128);
    loop {
        let mut buf = vec![0_i8; cap];
        let n = unsafe {
            ffi::llama_detokenize(
                vocab,
                tokens.as_ptr(),
                tokens.len() as i32,
                buf.as_mut_ptr(),
                cap as i32,
                false,
                false,
            )
        };
        if n >= 0 && (n as usize) < cap {
            return String::from_utf8_lossy(unsafe {
                std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), n as usize)
            })
            .into_owned();
        }
        cap *= 2;
    }
}

fn tokenize(
    vocab: *const ffi::llama_vocab,
    text: &str,
) -> Result<Vec<ffi::llama_token>, Box<dyn std::error::Error>> {
    let text_c = CString::new(text)?;
    let mut cap = text.len().saturating_add(8).max(32);
    loop {
        let mut tokens = vec![0; cap];
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
        if n >= 0 {
            tokens.truncate(n as usize);
            return Ok(tokens);
        }
        cap = (-n) as usize;
    }
}

fn format_input_text(
    args: &Args,
    model: *const ffi::llama_model,
) -> Result<String, Box<dyn std::error::Error>> {
    if !args.use_chat_template {
        return Ok(args.prompt.clone());
    }
    let tmpl = unsafe { ffi::llama_model_chat_template(model, ptr::null()) };
    if tmpl.is_null() {
        return Ok(args.prompt.clone());
    }
    let user_role = CString::new("user")?;
    let user_content = CString::new(args.prompt.as_str())?;
    let system_role = CString::new("system")?;
    let system_content = CString::new(args.system_prompt.as_str())?;
    let mut messages = Vec::new();
    if !args.system_prompt.is_empty() {
        messages.push(ffi::llama_chat_message {
            role: system_role.as_ptr(),
            content: system_content.as_ptr(),
        });
    }
    messages.push(ffi::llama_chat_message {
        role: user_role.as_ptr(),
        content: user_content.as_ptr(),
    });

    let mut cap = 1024;
    loop {
        let mut buf = vec![0_i8; cap];
        let n = unsafe {
            ffi::llama_chat_apply_template(
                tmpl,
                messages.as_ptr(),
                messages.len(),
                true,
                buf.as_mut_ptr(),
                cap as i32,
            )
        };
        if n < 0 {
            return Err("failed to apply chat template".into());
        }
        if n as usize <= cap {
            return Ok(String::from_utf8_lossy(unsafe {
                std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), n as usize)
            })
            .into_owned());
        }
        cap = n as usize + 1;
    }
}

fn callback_progress(
    step: i32,
    total_steps: i32,
    tokens: &[ffi::llama_token],
    n_input: usize,
    vocab: *const ffi::llama_vocab,
    mask_token: ffi::llama_token,
    visual_mode: bool,
) {
    let progress_percent = if total_steps > 0 {
        step * 100 / total_steps
    } else {
        100
    };
    let progress_bars = if total_steps > 0 {
        step * 50 / total_steps
    } else {
        50
    };
    eprint!(
        "\rdiffusion step: {step}/{total_steps} [{}{}] {progress_percent}%",
        "=".repeat(progress_bars as usize),
        " ".repeat((50 - progress_bars).max(0) as usize)
    );
    if visual_mode {
        eprint!("\x1b[2J\x1b[H");
        let mut text = String::from(" ");
        for &token in &tokens[n_input..] {
            if token == mask_token {
                text.push(' ');
            } else {
                text.push_str(&token_to_piece(vocab, token));
            }
        }
        eprintln!("\n{text}");
    }
    let _ = io::stderr().flush();
}

fn diffusion_generate(
    ctx: *mut ffi::llama_context,
    input_tokens: &[ffi::llama_token],
    args: &Args,
    shift_logits: bool,
    mask_token_id: ffi::llama_token,
    vocab: *const ffi::llama_vocab,
) -> Result<Vec<ffi::llama_token>, Box<dyn std::error::Error>> {
    if ctx.is_null() || input_tokens.is_empty() || args.n_ubatch as usize <= input_tokens.len() {
        return Err("invalid diffusion generation inputs".into());
    }

    unsafe {
        ffi::llama_set_causal_attn(ctx, false);
    }

    let max_length = args.n_ubatch as usize;
    let n_vocab = unsafe { ffi::llama_vocab_n_tokens(vocab) as usize };
    let mut output_tokens = vec![mask_token_id; max_length];
    output_tokens[..input_tokens.len()].copy_from_slice(input_tokens);
    let mut rng = SmallRng::new(args.seed);
    let sampler = sampler_init(args);
    let dist_sampler = dist_sampler_init(args.seed);
    let mut batch = Batch(unsafe { ffi::llama_batch_init(max_length as i32, 0, 1) });
    batch.0.n_tokens = max_length as i32;

    let mut candidates = vec![
        ffi::llama_token_data {
            id: 0,
            logit: 0.0,
            p: 0.0,
        };
        n_vocab
    ];
    let mut conf_candidates = Vec::with_capacity(max_length);
    let mut cond_logits_buffer = if args.cfg_scale > 0.0 {
        vec![0.0; n_vocab * max_length]
    } else {
        Vec::new()
    };
    let mut uncond_tokens = if args.cfg_scale > 0.0 {
        vec![mask_token_id; max_length]
    } else {
        Vec::new()
    };

    let schedule = if args.block_length > 0 {
        TransferSchedule::BlockBased
    } else {
        TransferSchedule::TimestepBased
    };
    let mut num_blocks = 1;
    let mut steps_per_block = args.steps;
    if schedule == TransferSchedule::BlockBased {
        if max_length % args.block_length as usize != 0
            || args.steps % (max_length as i32 / args.block_length) != 0
        {
            return Err("block length must divide length and steps".into());
        }
        num_blocks = max_length as i32 / args.block_length;
        steps_per_block = args.steps / num_blocks;
    }

    let start = unsafe { ffi::ggml_time_us() };
    let mut sampling_time = 0_i64;

    for block_num in 0..num_blocks {
        let block_start = if schedule == TransferSchedule::BlockBased {
            input_tokens.len() + block_num as usize * args.block_length as usize
        } else {
            0
        };
        let block_end = if schedule == TransferSchedule::BlockBased {
            (input_tokens.len() + (block_num as usize + 1) * args.block_length as usize)
                .min(max_length)
        } else {
            max_length
        };
        let num_transfer_tokens = if schedule == TransferSchedule::BlockBased {
            let block_mask_count = output_tokens[block_start..block_end]
                .iter()
                .filter(|&&token| token == mask_token_id)
                .count() as i32;
            get_num_transfer_tokens(block_mask_count, steps_per_block)
        } else {
            Vec::new()
        };

        for step in 0..steps_per_block {
            let global_step = block_num * steps_per_block + step;
            callback_progress(
                global_step,
                args.steps,
                &output_tokens,
                input_tokens.len(),
                vocab,
                mask_token_id,
                args.visual_mode,
            );

            for i in 0..max_length {
                unsafe {
                    *batch.0.token.add(i) = output_tokens[i];
                    *batch.0.pos.add(i) = i as i32;
                    *batch.0.n_seq_id.add(i) = 1;
                    *(*batch.0.seq_id.add(i)) = 0;
                    *batch.0.logits.add(i) = 1;
                }
            }

            if args.cfg_scale > 0.0 {
                let ret = unsafe { ffi::llama_decode(ctx, batch.0) };
                if ret != 0 {
                    return Err(format!("failed to generate conditional logits: {ret}").into());
                }
                for pos in 0..max_length {
                    let src = unsafe { ffi::llama_get_logits_ith(ctx, pos as i32) };
                    if src.is_null() {
                        return Err("failed to get conditional logits".into());
                    }
                    unsafe {
                        ptr::copy_nonoverlapping(
                            src,
                            cond_logits_buffer[pos * n_vocab..].as_mut_ptr(),
                            n_vocab,
                        );
                    }
                }

                uncond_tokens.copy_from_slice(&output_tokens);
                for token in &mut uncond_tokens[..input_tokens.len()] {
                    *token = mask_token_id;
                }
                for (i, &token) in uncond_tokens.iter().enumerate() {
                    unsafe {
                        *batch.0.token.add(i) = token;
                    }
                }
                let ret = unsafe { ffi::llama_decode(ctx, batch.0) };
                if ret != 0 {
                    return Err(format!("failed to generate unconditional logits: {ret}").into());
                }
                for pos in 0..max_length {
                    let src = unsafe { ffi::llama_get_logits_ith(ctx, pos as i32) };
                    if src.is_null() {
                        return Err("failed to get unconditional logits".into());
                    }
                    let dst = &mut cond_logits_buffer[pos * n_vocab..(pos + 1) * n_vocab];
                    for (i, dst_logit) in dst.iter_mut().enumerate() {
                        let uncond = unsafe { *src.add(i) };
                        *dst_logit = uncond + (args.cfg_scale + 1.0) * (*dst_logit - uncond);
                    }
                }
            } else {
                let ret = unsafe { ffi::llama_decode(ctx, batch.0) };
                if ret != 0 {
                    return Err(format!("failed to decode at step {global_step}: {ret}").into());
                }
            }

            let sampling_start = unsafe { ffi::ggml_time_us() };
            let mut mask_positions = Vec::new();
            for (i, &token) in output_tokens.iter().enumerate() {
                if token == mask_token_id
                    && (schedule != TransferSchedule::BlockBased
                        || (i >= block_start && i < block_end))
                {
                    mask_positions.push(i);
                }
            }
            if mask_positions.is_empty() {
                break;
            }

            let cond_logits = if args.cfg_scale > 0.0 {
                Some(cond_logits_buffer.as_slice())
            } else {
                None
            };

            if args.algorithm == DiffusionAlgorithm::Origin {
                let transfer_count = calculate_transfer_count(
                    step,
                    steps_per_block,
                    mask_positions.len() as i32,
                    schedule,
                    args.eps,
                    &num_transfer_tokens,
                );
                let p_transfer = transfer_count as f32 / mask_positions.len() as f32;
                for pos in mask_positions {
                    if rng.next_f32() < p_transfer {
                        fill_candidates_from_logits(
                            &mut candidates,
                            ctx,
                            cond_logits,
                            n_vocab,
                            pos,
                            shift_logits,
                            args.add_gumbel_noise,
                            args.temp,
                            &mut rng,
                        )?;
                        let mut cur_p = ffi::llama_token_data_array {
                            data: candidates.as_mut_ptr(),
                            size: candidates.len(),
                            selected: -1,
                            sorted: false,
                        };
                        unsafe {
                            ffi::llama_sampler_apply(sampler.0, &mut cur_p);
                        }
                        output_tokens[pos] = candidates[cur_p.selected as usize].id;
                    }
                }
            } else {
                let mut confidences = Vec::with_capacity(mask_positions.len());
                let mut sampled_tokens = vec![0; mask_positions.len()];
                for (mask_idx, &pos) in mask_positions.iter().enumerate() {
                    fill_candidates_from_logits(
                        &mut candidates,
                        ctx,
                        cond_logits,
                        n_vocab,
                        pos,
                        shift_logits,
                        args.add_gumbel_noise,
                        args.temp,
                        &mut rng,
                    )?;
                    let mut cur_p = ffi::llama_token_data_array {
                        data: candidates.as_mut_ptr(),
                        size: candidates.len(),
                        selected: -1,
                        sorted: false,
                    };
                    unsafe {
                        ffi::llama_sampler_apply(sampler.0, &mut cur_p);
                    }
                    let selected = cur_p.selected as usize;
                    let sampled_token = candidates[selected].id;
                    let conf =
                        calculate_confidence(&candidates, selected, args.algorithm, &mut rng);
                    sampled_tokens[mask_idx] = sampled_token;
                    confidences.push((conf, mask_idx));
                }

                let transfer_count = calculate_transfer_count(
                    step,
                    steps_per_block,
                    mask_positions.len() as i32,
                    schedule,
                    args.eps,
                    &num_transfer_tokens,
                )
                .max(0) as usize;
                let n_take = transfer_count.min(confidences.len());
                if args.alg_temp == 0.0 {
                    confidences.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                    for &(_, mask_idx) in confidences.iter().take(n_take) {
                        output_tokens[mask_positions[mask_idx]] = sampled_tokens[mask_idx];
                    }
                } else {
                    conf_candidates.clear();
                    for (i, (conf, _)) in confidences.iter().enumerate() {
                        conf_candidates.push(ffi::llama_token_data {
                            id: i as i32,
                            logit: *conf / args.alg_temp,
                            p: 0.0,
                        });
                    }
                    for _ in 0..n_take {
                        let mut conf_array = ffi::llama_token_data_array {
                            data: conf_candidates.as_mut_ptr(),
                            size: conf_candidates.len(),
                            selected: -1,
                            sorted: false,
                        };
                        unsafe {
                            ffi::llama_sampler_apply(dist_sampler.0, &mut conf_array);
                        }
                        let selected_idx = conf_array.selected as usize;
                        let mask_idx = selected_idx;
                        output_tokens[mask_positions[mask_idx]] = sampled_tokens[mask_idx];
                        conf_candidates[selected_idx].p = 0.0;
                    }
                }
            }
            sampling_time += unsafe { ffi::ggml_time_us() } - sampling_start;
        }
    }

    let total_time = unsafe { ffi::ggml_time_us() } - start;
    eprintln!(
        "\ntotal time: {:.2}ms, time per step: {:.2}ms, sampling time per step: {:.2}ms",
        total_time as f64 / 1000.0,
        total_time as f64 / 1000.0 / args.steps as f64,
        sampling_time as f64 / 1000.0 / args.steps as f64
    );

    Ok(output_tokens)
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let _backend = Backend::init();
    let model_path = CString::new(args.model_path.as_str())?;
    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;
    model_params.use_mmap = args.use_mmap;
    model_params.use_mlock = args.use_mlock;
    model_params.check_tensors = args.check_tensors;

    let model =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model.0.is_null() {
        return Err(format!("failed to load model '{}'", args.model_path).into());
    }
    if !unsafe { ffi::llama_model_is_diffusion(model.0) } {
        return Err("unsupported model for diffusion".into());
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = args.n_batch;
    ctx_params.n_ubatch = args.n_ubatch;
    ctx_params.no_perf = args.no_perf;
    ctx_params.flash_attn_type = args.flash_attn_type;
    ctx_params.type_k = args.cache_type_k;
    ctx_params.type_v = args.cache_type_v;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create context".into());
    }
    unsafe {
        ffi::llama_set_n_threads(ctx.0, args.n_threads, args.n_threads_batch);
    }

    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    let formatted_prompt = format_input_text(&args, model.0)?;
    let input_tokens = tokenize(vocab, &formatted_prompt)?;
    if input_tokens.len() as u32 >= unsafe { ffi::llama_n_ctx(ctx.0) } {
        return Err(format!(
            "input too long ({} tokens), max context is {}",
            input_tokens.len(),
            unsafe { ffi::llama_n_ctx(ctx.0) }
        )
        .into());
    }

    let mask_token = unsafe { ffi::llama_vocab_mask(vocab) };
    if mask_token == ffi::LLAMA_TOKEN_NULL {
        return Err("model vocabulary does not define a mask token".into());
    }

    let mut shift_buf = vec![0_i8; 8];
    let key = CString::new("diffusion.shift_logits")?;
    let shift_logits = if unsafe {
        ffi::llama_model_meta_val_str(
            model.0,
            key.as_ptr(),
            shift_buf.as_mut_ptr(),
            shift_buf.len(),
        )
    } >= 0
    {
        unsafe { CStr::from_ptr(shift_buf.as_ptr()) }.to_string_lossy() == "true"
    } else {
        true
    };

    let mut output_tokens =
        diffusion_generate(ctx.0, &input_tokens, &args, shift_logits, mask_token, vocab)?;
    if args.visual_mode {
        eprint!("\x1b[2J\x1b[H");
    }
    output_tokens.drain(0..input_tokens.len());
    println!("\n{}", detokenize(vocab, &output_tokens));
    Ok(())
}

fn main() {
    let args = match parse_args(env::args().skip(1)) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    };
    if let Err(err) = run(args) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_diffusion_flags() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-p",
            "hello",
            "--steps",
            "12",
            "--block-length",
            "16",
            "--algorithm",
            "2",
            "--visual",
            "--gumbel",
            "--cfg-scale",
            "1.5",
        ])
        .unwrap();
        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.steps, 12);
        assert_eq!(args.eps, 0.0);
        assert_eq!(args.block_length, 16);
        assert_eq!(args.algorithm, DiffusionAlgorithm::MarginBased);
        assert!(args.visual_mode);
        assert!(args.add_gumbel_noise);
        assert_eq!(args.cfg_scale, 1.5);
    }

    #[test]
    fn trailing_text_becomes_prompt() {
        let args = parse_args(["-m", "model.gguf", "write", "text"]).unwrap();
        assert_eq!(args.prompt, "write text");
    }

    #[test]
    fn requires_exactly_one_schedule() {
        let err = parse_args(["-m", "model.gguf", "-p", "x", "--eps", "0"]).unwrap_err();
        assert!(matches!(err, ParseError::InvalidValue(_)));
    }

    #[test]
    fn distributes_block_transfer_tokens() {
        assert_eq!(get_num_transfer_tokens(10, 4), vec![3, 3, 2, 2]);
    }

    #[test]
    fn timestep_transfer_finishes_on_last_step() {
        assert_eq!(
            calculate_transfer_count(3, 4, 7, TransferSchedule::TimestepBased, 0.001, &[]),
            7
        );
    }

    #[test]
    fn margin_confidence_uses_first_two_candidates() {
        let mut rng = SmallRng::new(1);
        let candidates = [
            ffi::llama_token_data {
                id: 1,
                logit: 0.0,
                p: 0.7,
            },
            ffi::llama_token_data {
                id: 2,
                logit: 0.0,
                p: 0.2,
            },
        ];
        assert!(
            (calculate_confidence(&candidates, 0, DiffusionAlgorithm::MarginBased, &mut rng) - 0.5)
                .abs()
                < 1e-6
        );
    }
}
