use llama_simple_rust::ffi;
use std::env;
use std::ffi::{CStr, CString};
use std::ptr;

const LLAMA_POOLING_TYPE_NONE: i32 = 0;
const LLAMA_POOLING_TYPE_RANK: i32 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub embd_sep: String,
    pub cls_sep: String,
    pub embd_out: String,
    pub embd_normalize: i32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_parallel: u32,
    pub n_gpu_layers: i32,
    pub verbose_prompt: bool,
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
            ParseError::MissingModel => write!(f, "missing required -m/--model model.gguf"),
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
            prompt: String::new(),
            embd_sep: "\n".to_string(),
            cls_sep: "\t".to_string(),
            embd_out: String::new(),
            embd_normalize: 2,
            n_ctx: 512,
            n_batch: 512,
            n_parallel: 1,
            n_gpu_layers: 99,
            verbose_prompt: false,
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
            "--embd-sep" => parsed.embd_sep = take(&mut iter, &arg)?,
            "--cls-sep" => parsed.cls_sep = take(&mut iter, &arg)?,
            "--embd-output-format" | "--embd-out" => parsed.embd_out = take(&mut iter, &arg)?,
            "--embd-normalize" => parsed.embd_normalize = parse_i32(&mut iter, &arg)?,
            "-c" | "--ctx-size" => parsed.n_ctx = parse_u32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_u32(&mut iter, &arg)?,
            "-np" | "--parallel" => parsed.n_parallel = parse_u32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "--verbose-prompt" => parsed.verbose_prompt = true,
            "-h" | "--help" => return Err(ParseError::MissingModel),
            _ => {
                if !arg.starts_with('-') {
                    if !parsed.prompt.is_empty() {
                        parsed.prompt.push(' ');
                    }
                    parsed.prompt.push_str(&arg);
                }
            }
        }
    }

    if parsed.model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }
    if parsed.prompt.is_empty() {
        parsed.prompt = "The quick brown fox jumps over the lazy dog".to_string();
    }
    if parsed.embd_sep.is_empty() {
        return Err(ParseError::InvalidValue(
            "--embd-sep must not be empty".to_string(),
        ));
    }
    if parsed.cls_sep.is_empty() {
        return Err(ParseError::InvalidValue(
            "--cls-sep must not be empty".to_string(),
        ));
    }
    if !matches!(
        parsed.embd_out.as_str(),
        "" | "json" | "json+" | "array" | "raw"
    ) {
        return Err(ParseError::InvalidValue(format!(
            "unsupported embedding output format: {}",
            parsed.embd_out
        )));
    }
    if parsed.n_parallel == 0 {
        return Err(ParseError::InvalidValue(
            "--parallel must be positive".to_string(),
        ));
    }
    if parsed.n_batch < parsed.n_ctx {
        parsed.n_batch = parsed.n_ctx;
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

pub fn split_lines(text: &str, separator: &str) -> Vec<String> {
    text.split(separator).map(str::to_string).collect()
}

pub fn normalize_embedding(input: &[f32], mode: i32) -> Vec<f32> {
    match mode {
        0 => input.to_vec(),
        -1 => {
            let sum = input.iter().map(|x| f64::from(*x)).sum::<f64>();
            let mean = if input.is_empty() {
                0.0
            } else {
                sum / input.len() as f64
            };
            let variance = input
                .iter()
                .map(|x| {
                    let d = f64::from(*x) - mean;
                    d * d
                })
                .sum::<f64>()
                / input.len().max(1) as f64;
            let scale = if variance > 0.0 {
                1.0 / variance.sqrt()
            } else {
                0.0
            };
            input
                .iter()
                .map(|x| ((f64::from(*x) - mean) * scale) as f32)
                .collect()
        }
        _ => {
            let sum = input
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>()
                .sqrt();
            let norm = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            input
                .iter()
                .map(|x| (f64::from(*x) * norm) as f32)
                .collect()
        }
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = f64::from(*x);
        let y = f64::from(*y);
        sum += x * y;
        sum_a += x * x;
        sum_b += y * y;
    }
    if sum_a == 0.0 || sum_b == 0.0 {
        return if sum_a == 0.0 && sum_b == 0.0 {
            1.0
        } else {
            0.0
        };
    }
    (sum / (sum_a.sqrt() * sum_b.sqrt())) as f32
}

fn replace_all(mut text: String, from: &str, to: &str) -> String {
    text = text.replace(from, to);
    text
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

fn cstr_to_string(ptr: *const i8) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

fn tokenize(
    vocab: *const ffi::llama_vocab,
    text: &str,
    add_special: bool,
    parse_special: bool,
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
        return Err("failed to tokenize text".to_string());
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

unsafe fn batch_add_seq(
    batch: &mut ffi::llama_batch,
    tokens: &[ffi::llama_token],
    seq_id: ffi::llama_seq_id,
) {
    for (pos, token) in tokens.iter().enumerate() {
        let i = batch.n_tokens as isize;
        *batch.token.offset(i) = *token;
        *batch.pos.offset(i) = pos as ffi::llama_pos;
        *batch.n_seq_id.offset(i) = 1;
        let seq_slot = *batch.seq_id.offset(i);
        *seq_slot = seq_id;
        *batch.logits.offset(i) = 1;
        batch.n_tokens += 1;
    }
}

fn batch_decode(
    ctx: *mut ffi::llama_context,
    batch: &mut ffi::llama_batch,
    output: &mut [f32],
    n_embd_out: usize,
    embd_norm: i32,
) -> Result<(), String> {
    let pooling_type = unsafe { ffi::llama_pooling_type(ctx) };
    unsafe {
        ffi::llama_memory_clear(ffi::llama_get_memory(ctx), true);
        eprintln!(
            "batch_decode: n_tokens = {}, n_seq = {}",
            batch.n_tokens,
            ffi::llama_n_seq_max(ctx)
        );
        if ffi::llama_decode(ctx, *batch) < 0 {
            return Err("failed to process batch".to_string());
        }

        for i in 0..batch.n_tokens {
            if *batch.logits.offset(i as isize) == 0 {
                continue;
            }

            let (emb, emb_pos) = if pooling_type == LLAMA_POOLING_TYPE_NONE {
                (ffi::llama_get_embeddings_ith(ctx, i), i as usize)
            } else {
                let seq_id = **batch.seq_id.offset(i as isize);
                (ffi::llama_get_embeddings_seq(ctx, seq_id), seq_id as usize)
            };

            if emb.is_null() {
                return Err(format!("failed to get embeddings for batch item {i}"));
            }
            let start = emb_pos * n_embd_out;
            if start + n_embd_out > output.len() {
                return Err("embedding output buffer is too small".to_string());
            }
            let raw = std::slice::from_raw_parts(emb, n_embd_out);
            let normalized = normalize_embedding(raw, embd_norm);
            output[start..start + n_embd_out].copy_from_slice(&normalized);
        }
    }
    Ok(())
}

fn build_rank_prompt(
    prompt: &str,
    cls_sep: &str,
    rerank_template: Option<&str>,
    added_eos_token: &str,
    added_sep_token: &str,
) -> String {
    let pairs = split_lines(prompt, cls_sep);
    if let (Some(template), Some(query), Some(document)) =
        (rerank_template, pairs.first(), pairs.get(1))
    {
        return replace_all(
            replace_all(template.to_string(), "{query}", query),
            "{document}",
            document,
        );
    }
    let mut final_prompt = String::new();
    for (i, pair) in pairs.iter().enumerate() {
        final_prompt.push_str(pair);
        if i + 1 != pairs.len() {
            final_prompt.push_str(added_eos_token);
            final_prompt.push_str(added_sep_token);
        }
    }
    final_prompt
}

fn prepare_inputs(
    args: &Args,
    vocab: *const ffi::llama_vocab,
    model: *const ffi::llama_model,
    pooling_type: i32,
) -> Result<(Vec<String>, Vec<Vec<ffi::llama_token>>), String> {
    let prompts = split_lines(&args.prompt, &args.embd_sep);
    let sep = unsafe { ffi::llama_vocab_sep(vocab) };
    let eos = unsafe { ffi::llama_vocab_eos(vocab) };
    let added_sep_token = if unsafe { ffi::llama_vocab_get_add_sep(vocab) } {
        cstr_to_string(unsafe { ffi::llama_vocab_get_text(vocab, sep) })
    } else {
        String::new()
    };
    let added_eos_token = if unsafe { ffi::llama_vocab_get_add_eos(vocab) } {
        cstr_to_string(unsafe { ffi::llama_vocab_get_text(vocab, eos) })
    } else {
        String::new()
    };
    let rerank_name = CString::new("rerank").unwrap();
    let rerank_ptr = unsafe { ffi::llama_model_chat_template(model, rerank_name.as_ptr()) };
    let rerank_template = if rerank_ptr.is_null() {
        None
    } else {
        Some(cstr_to_string(rerank_ptr))
    };

    let mut inputs = Vec::with_capacity(prompts.len());
    for prompt in &prompts {
        let text = if pooling_type == LLAMA_POOLING_TYPE_RANK && prompt.contains(&args.cls_sep) {
            build_rank_prompt(
                prompt,
                &args.cls_sep,
                rerank_template.as_deref(),
                &added_eos_token,
                &added_sep_token,
            )
        } else {
            prompt.clone()
        };
        inputs.push(tokenize(vocab, &text, true, true)?);
    }
    Ok((prompts, inputs))
}

fn print_json(
    embeddings: &[f32],
    n_embd_count: usize,
    n_embd_out: usize,
    n_prompts: usize,
    embd_out: &str,
    embd_normalize: i32,
) {
    let not_array = embd_out != "array";
    if not_array {
        println!("{{\n  \"object\": \"list\",\n  \"data\": [");
    } else {
        print!("[");
    }

    for j in 0..n_embd_count {
        if not_array {
            print!("    {{\n      \"object\": \"embedding\",\n      \"index\": {j},\n      \"embedding\": ");
        }
        print!("[");
        for i in 0..n_embd_out {
            let value = embeddings[j * n_embd_out + i];
            if embd_normalize == 0 {
                print!("{value:.0}");
            } else {
                print!("{value:.7}");
            }
            if i + 1 < n_embd_out {
                print!(",");
            }
        }
        if not_array {
            print!("]\n    }}");
        } else {
            print!("]");
        }
        if j + 1 < n_embd_count {
            if not_array {
                println!(",");
            } else {
                print!(",");
            }
        }
    }

    if not_array {
        print!("\n  ]");
    } else {
        println!("]");
    }

    if embd_out == "json+" && n_prompts > 1 {
        println!(",\n  \"cosineSimilarity\": [");
        for i in 0..n_embd_count {
            print!("    [");
            for j in 0..n_embd_count {
                let sim = cosine_similarity(
                    &embeddings[i * n_embd_out..(i + 1) * n_embd_out],
                    &embeddings[j * n_embd_out..(j + 1) * n_embd_out],
                );
                print!("{sim:6.2}");
                if j + 1 < n_embd_count {
                    print!(", ");
                }
            }
            print!(" ]");
            if i + 1 < n_embd_count {
                println!(",");
            }
        }
        print!("\n  ]");
    }

    if not_array {
        println!("\n}}");
    }
}

fn print_raw_embeddings(
    embeddings: &[f32],
    n_embd_count: usize,
    n_embd_out: usize,
    model: *const ffi::llama_model,
    pooling_type: i32,
    embd_normalize: i32,
) {
    let n_cls_out = unsafe { ffi::llama_model_n_cls_out(model) as usize };
    let cols = if pooling_type == LLAMA_POOLING_TYPE_RANK {
        n_embd_out.min(n_cls_out)
    } else {
        n_embd_out
    };

    for j in 0..n_embd_count {
        for i in 0..cols {
            let value = embeddings[j * n_embd_out + i];
            if embd_normalize == 0 {
                print!("{value:.0}");
            } else {
                print!("{value:.7}");
            }
            if i + 1 < cols {
                print!(" ");
            }
        }
        println!();
    }
}

fn print_human_embeddings(
    embeddings: &[f32],
    prompts: &[String],
    n_embd_count: usize,
    n_embd_out: usize,
    model: *const ffi::llama_model,
    pooling_type: i32,
    embd_normalize: i32,
) {
    println!();
    if pooling_type == LLAMA_POOLING_TYPE_NONE {
        for j in 0..n_embd_count {
            print!("embedding {j}: ");
            for i in 0..3.min(n_embd_out) {
                print_value(embeddings[j * n_embd_out + i], embd_normalize);
                print!(" ");
            }
            print!(" ... ");
            for i in n_embd_out.saturating_sub(3)..n_embd_out {
                print_value(embeddings[j * n_embd_out + i], embd_normalize);
                print!(" ");
            }
            println!();
        }
    } else if pooling_type == LLAMA_POOLING_TYPE_RANK {
        let n_cls_out = unsafe { ffi::llama_model_n_cls_out(model) };
        let labels = (0..n_cls_out)
            .map(|i| {
                let label = cstr_to_string(unsafe { ffi::llama_model_cls_label(model, i) });
                if label.is_empty() {
                    i.to_string()
                } else {
                    label
                }
            })
            .collect::<Vec<_>>();
        for j in 0..n_embd_count {
            for i in 0..n_cls_out as usize {
                if n_cls_out == 1 {
                    println!("rerank score {j}: {:8.3}", embeddings[j * n_embd_out]);
                } else {
                    println!(
                        "rerank score {j}: {:8.3} [{}]",
                        embeddings[j * n_embd_out + i],
                        labels[i]
                    );
                }
            }
        }
    } else {
        for (j, _) in prompts.iter().enumerate() {
            print!("embedding {j}: ");
            let cols = if prompts.len() > 1 {
                16.min(n_embd_out)
            } else {
                n_embd_out
            };
            for i in 0..cols {
                print_value(embeddings[j * n_embd_out + i], embd_normalize);
                print!(" ");
            }
            println!();
        }

        if prompts.len() > 1 {
            println!("\ncosine similarity matrix:\n");
            for prompt in prompts {
                print!("{:.6} ", prompt);
            }
            println!();
            for i in 0..prompts.len() {
                for j in 0..prompts.len() {
                    let sim = cosine_similarity(
                        &embeddings[i * n_embd_out..(i + 1) * n_embd_out],
                        &embeddings[j * n_embd_out..(j + 1) * n_embd_out],
                    );
                    print!("{sim:6.2} ");
                }
                println!("{:.10}", prompts[i]);
            }
        }
    }
}

fn print_value(value: f32, embd_normalize: i32) {
    if embd_normalize == 0 {
        print!("{value:6.0}");
    } else {
        print!("{value:9.6}");
    }
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

    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        return Err("failed to get model vocabulary".to_string());
    }

    let max_parallel = unsafe { ffi::llama_max_parallel_sequences() as u32 };
    let mut n_parallel = args.n_parallel;
    let kv_unified = if n_parallel == 1 {
        eprintln!("n_parallel == 1 -> unified KV cache is enabled");
        n_parallel = max_parallel;
        true
    } else {
        false
    };

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = args.n_batch;
    ctx_params.n_ubatch = args.n_batch;
    ctx_params.n_seq_max = n_parallel;
    ctx_params.embeddings = true;
    ctx_params.kv_unified = kv_unified;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create llama_context".to_string());
    }

    if unsafe { ffi::llama_model_has_encoder(model.0) && ffi::llama_model_has_decoder(model.0) } {
        return Err("computing embeddings in encoder-decoder models is not supported".to_string());
    }

    let n_ctx_train = unsafe { ffi::llama_model_n_ctx_train(model.0) };
    let n_ctx = unsafe { ffi::llama_n_ctx(ctx.0) as i32 };
    if n_ctx > n_ctx_train {
        eprintln!(
            "warning: model was trained on only {n_ctx_train} context tokens ({n_ctx} specified)"
        );
    }

    let pooling_type = unsafe { ffi::llama_pooling_type(ctx.0) };
    let (prompts, inputs) = prepare_inputs(&args, vocab, model.0, pooling_type)?;
    let n_batch = args.n_batch as usize;
    for (i, input) in inputs.iter().enumerate() {
        if input.len() > n_batch {
            return Err(format!(
                "number of tokens in input line ({}) exceeds batch size ({n_batch}), increase batch size and re-run",
                input.len()
            ));
        }
        let sep = unsafe { ffi::llama_vocab_sep(vocab) };
        let eos = unsafe { ffi::llama_vocab_eos(vocab) };
        if input.last().copied() != Some(sep) && input.last().copied() != Some(eos) {
            eprintln!("warning: last token in prompt {i} is not SEP or EOS");
        }
        if args.verbose_prompt {
            eprintln!("prompt {i}: '{}'", prompts[i]);
            eprintln!("number of tokens in prompt = {}", input.len());
            for token in input {
                eprintln!("{token:6} -> '{}'", token_to_piece(vocab, *token)?);
            }
        }
    }

    let n_embd_count = if pooling_type == LLAMA_POOLING_TYPE_NONE {
        inputs.iter().map(Vec::len).sum()
    } else {
        prompts.len()
    };
    let n_embd_out = unsafe { ffi::llama_model_n_embd_out(model.0) as usize };
    let mut embeddings = vec![0.0f32; n_embd_count * n_embd_out];
    let mut batch = Batch(unsafe { ffi::llama_batch_init(n_batch as i32, 0, 1) });

    let mut e = 0usize;
    let mut s = 0usize;
    let n_seq_max = unsafe { ffi::llama_n_seq_max(ctx.0) as usize };
    for input in &inputs {
        if batch.0.n_tokens as usize + input.len() > n_batch || s >= n_seq_max {
            let next_e = e + if pooling_type == LLAMA_POOLING_TYPE_NONE {
                batch.0.n_tokens as usize
            } else {
                s
            };
            batch_decode(
                ctx.0,
                &mut batch.0,
                &mut embeddings[e * n_embd_out..next_e * n_embd_out],
                n_embd_out,
                args.embd_normalize,
            )?;
            e = next_e;
            s = 0;
            batch_clear(&mut batch.0);
        }
        unsafe {
            batch_add_seq(&mut batch.0, input, s as i32);
        }
        s += 1;
    }

    if s > 0 {
        let next_e = e + if pooling_type == LLAMA_POOLING_TYPE_NONE {
            batch.0.n_tokens as usize
        } else {
            s
        };
        batch_decode(
            ctx.0,
            &mut batch.0,
            &mut embeddings[e * n_embd_out..next_e * n_embd_out],
            n_embd_out,
            args.embd_normalize,
        )?;
    }

    match args.embd_out.as_str() {
        "" => print_human_embeddings(
            &embeddings,
            &prompts,
            n_embd_count,
            n_embd_out,
            model.0,
            pooling_type,
            args.embd_normalize,
        ),
        "json" | "json+" | "array" => print_json(
            &embeddings,
            n_embd_count,
            n_embd_out,
            prompts.len(),
            &args.embd_out,
            args.embd_normalize,
        ),
        "raw" => print_raw_embeddings(
            &embeddings,
            n_embd_count,
            n_embd_out,
            model.0,
            pooling_type,
            args.embd_normalize,
        ),
        _ => unreachable!(),
    }

    println!();
    unsafe {
        ffi::llama_perf_context_print(ctx.0);
    }
    Ok(())
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m ./models/bge-base-en-v1.5-f16.gguf -p \"hello world\"");
    eprintln!();
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-embedding".to_string());
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
    fn parses_embedding_options() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-p",
            "a\nb",
            "--embd-sep",
            "\n",
            "--cls-sep",
            " || ",
            "--embd-out",
            "json+",
            "--embd-normalize",
            "0",
            "-c",
            "256",
            "-b",
            "128",
            "-np",
            "4",
            "-ngl",
            "0",
            "--verbose-prompt",
        ])
        .unwrap();
        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "a\nb");
        assert_eq!(args.cls_sep, " || ");
        assert_eq!(args.embd_out, "json+");
        assert_eq!(args.embd_normalize, 0);
        assert_eq!(args.n_ctx, 256);
        assert_eq!(args.n_batch, 256);
        assert_eq!(args.n_parallel, 4);
        assert_eq!(args.n_gpu_layers, 0);
        assert!(args.verbose_prompt);
    }

    #[test]
    fn rejects_bad_output_format() {
        assert_eq!(
            parse_args(["--model", "model.gguf", "--embd-out", "yaml"]).unwrap_err(),
            ParseError::InvalidValue("unsupported embedding output format: yaml".to_string())
        );
    }

    #[test]
    fn splits_on_custom_separator() {
        assert_eq!(
            split_lines("alpha || beta || gamma", " || "),
            vec!["alpha", "beta", "gamma"]
        );
    }

    #[test]
    fn normalizes_embeddings() {
        let normalized = normalize_embedding(&[3.0, 4.0], 2);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
        assert_eq!(normalize_embedding(&[3.0, 4.0], 0), vec![3.0, 4.0]);
    }

    #[test]
    fn builds_rank_prompt_from_template_or_tokens() {
        let templated = build_rank_prompt(
            "question\tanswer",
            "\t",
            Some("Q: {query}\nD: {document}"),
            "</s>",
            "[SEP]",
        );
        assert_eq!(templated, "Q: question\nD: answer");
        let joined = build_rank_prompt("question\tanswer", "\t", None, "</s>", "[SEP]");
        assert_eq!(joined, "question</s>[SEP]answer");
    }
}
