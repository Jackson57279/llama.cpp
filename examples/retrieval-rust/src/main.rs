use llama_simple_rust::ffi;
use std::cmp::Ordering;
use std::env;
use std::ffi::CString;
use std::fs;
use std::io::{self, Write};
use std::ptr;

const LLAMA_POOLING_TYPE_NONE: i32 = 0;

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub model_path: String,
    pub context_files: Vec<String>,
    pub chunk_size: usize,
    pub chunk_separator: String,
    pub top_k: i32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_gpu_layers: i32,
    pub verbose_prompt: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingModel,
    MissingContextFile,
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m/--model model.gguf"),
            ParseError::MissingContextFile => write!(f, "--context-file must be specified"),
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
            context_files: Vec::new(),
            chunk_size: 64,
            chunk_separator: "\n".to_string(),
            top_k: 3,
            n_ctx: 512,
            n_batch: 512,
            n_gpu_layers: 99,
            verbose_prompt: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub filename: String,
    pub filepos: usize,
    pub textdata: String,
    pub tokens: Vec<ffi::llama_token>,
    pub embedding: Vec<f32>,
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
            "--context-file" => parsed.context_files.push(take(&mut iter, &arg)?),
            "--chunk-size" => parsed.chunk_size = parse_usize(&mut iter, &arg)?,
            "--chunk-separator" => parsed.chunk_separator = take(&mut iter, &arg)?,
            "--top-k" => parsed.top_k = parse_i32(&mut iter, &arg)?,
            "-c" | "--ctx-size" => parsed.n_ctx = parse_u32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_u32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "--verbose-prompt" => parsed.verbose_prompt = true,
            "-h" | "--help" => return Err(ParseError::MissingModel),
            _ => {}
        }
    }

    if parsed.model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }
    if parsed.context_files.is_empty() {
        return Err(ParseError::MissingContextFile);
    }
    if parsed.chunk_size == 0 {
        return Err(ParseError::InvalidValue(
            "--chunk-size must be positive".to_string(),
        ));
    }
    if parsed.chunk_separator.is_empty() {
        return Err(ParseError::InvalidValue(
            "--chunk-separator must not be empty".to_string(),
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

fn parse_usize<I>(iter: &mut I, flag: &str) -> Result<usize, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidInteger(flag.to_string(), value))
}

pub fn chunk_text(filename: &str, text: &str, chunk_size: usize, separator: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut filepos = 0usize;

    for piece in text.split_inclusive(separator) {
        current.push_str(piece);
        if current.len() > chunk_size {
            filepos += push_chunk(&mut chunks, filename, filepos, &mut current);
        }
    }

    if !current.is_empty() {
        if chunks.is_empty() {
            push_chunk(&mut chunks, filename, filepos, &mut current);
        } else {
            chunks.last_mut().unwrap().textdata.push_str(&current);
        }
    }

    chunks
}

fn push_chunk(
    chunks: &mut Vec<Chunk>,
    filename: &str,
    filepos: usize,
    current: &mut String,
) -> usize {
    let textdata = std::mem::take(current);
    let len = textdata.len();
    chunks.push(Chunk {
        filename: filename.to_string(),
        filepos,
        textdata,
        tokens: Vec::new(),
        embedding: Vec::new(),
    });
    len
}

fn chunk_file(filename: &str, chunk_size: usize, separator: &str) -> Result<Vec<Chunk>, String> {
    let text = fs::read_to_string(filename)
        .map_err(|err| format!("could not open/read file {filename}: {err}"))?;
    Ok(chunk_text(filename, &text, chunk_size, separator))
}

pub fn normalize_l2(input: &[f32]) -> Vec<f32> {
    let sum = input
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    let norm = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    input.iter().map(|value| value * norm).collect()
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

pub fn top_k_similarities(chunks: &[Chunk], query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let mut similarities = chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| (index, cosine_similarity(&chunk.embedding, query)))
        .collect::<Vec<_>>();
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    similarities.truncate(top_k.min(similarities.len()));
    similarities
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

fn batch_process(
    ctx: *mut ffi::llama_context,
    batch: &mut ffi::llama_batch,
    output: &mut [f32],
    n_embd: usize,
) -> Result<(), String> {
    unsafe {
        ffi::llama_memory_clear(ffi::llama_get_memory(ctx), false);
        eprintln!("batch_process: n_tokens = {}", batch.n_tokens);
        if ffi::llama_decode(ctx, *batch) < 0 {
            return Err("failed to process batch".to_string());
        }

        for i in 0..batch.n_tokens {
            if *batch.logits.offset(i as isize) == 0 {
                continue;
            }
            let seq_id = **batch.seq_id.offset(i as isize);
            let mut emb = ffi::llama_get_embeddings_seq(ctx, seq_id);
            if emb.is_null() {
                emb = ffi::llama_get_embeddings_ith(ctx, i);
                if emb.is_null() {
                    return Err(format!("failed to get embeddings for token {i}"));
                }
            }

            let raw = std::slice::from_raw_parts(emb, n_embd);
            let normalized = normalize_l2(raw);
            let start = seq_id as usize * n_embd;
            output[start..start + n_embd].copy_from_slice(&normalized);
        }
    }
    Ok(())
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} --model ./models/bge-base-en-v1.5-f16.gguf --top-k 3 --context-file README.md --context-file License --chunk-size 100 --chunk-separator .");
    eprintln!();
}

fn run(args: Args) -> Result<(), String> {
    let mut chunks = Vec::new();
    for context_file in &args.context_files {
        eprintln!("{context_file}");
        chunks.extend(chunk_file(
            context_file,
            args.chunk_size,
            &args.chunk_separator,
        )?);
    }
    eprintln!("Number of chunks: {}", chunks.len());
    if chunks.is_empty() {
        return Err("no chunks were produced from context files".to_string());
    }

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

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = args.n_batch;
    ctx_params.n_ubatch = args.n_batch;
    ctx_params.embeddings = true;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create the llama_context".to_string());
    }

    if unsafe { ffi::llama_pooling_type(ctx.0) } == LLAMA_POOLING_TYPE_NONE {
        return Err("pooling type NONE not supported".to_string());
    }

    let n_ctx_train = unsafe { ffi::llama_model_n_ctx_train(model.0) };
    let n_ctx = unsafe { ffi::llama_n_ctx(ctx.0) as i32 };
    if n_ctx > n_ctx_train {
        eprintln!(
            "warning: model was trained on only {n_ctx_train} context tokens ({n_ctx} specified)"
        );
    }

    let n_batch = args.n_batch as usize;
    for (chunk_id, chunk) in chunks.iter_mut().enumerate() {
        let mut tokens = tokenize(vocab, &chunk.textdata, true, false)?;
        if tokens.len() > n_batch {
            return Err(format!(
                "chunk size ({}) exceeds batch size ({n_batch}), increase batch size and re-run",
                tokens.len()
            ));
        }
        let eos = unsafe { ffi::llama_vocab_eos(vocab) };
        if eos >= 0 && tokens.last().copied() != Some(eos) {
            tokens.push(eos);
        }
        if args.verbose_prompt {
            eprintln!("prompt {chunk_id}: '{}'", chunk.textdata);
            eprintln!("number of tokens in prompt = {}", tokens.len());
            for token in &tokens {
                eprintln!("{token:6} -> '{}'", token_to_piece(vocab, *token)?);
            }
        }
        chunk.tokens = tokens;
    }

    let n_chunks = chunks.len();
    let mut batch = Batch(unsafe { ffi::llama_batch_init(n_batch as i32, 0, 1) });
    let n_embd = unsafe { ffi::llama_model_n_embd_out(model.0) as usize };
    let mut embeddings = vec![0.0f32; n_chunks * n_embd];

    let mut processed = 0usize;
    let mut seq_count = 0usize;
    for chunk_index in 0..n_chunks {
        let n_toks = chunks[chunk_index].tokens.len();
        if batch.0.n_tokens as usize + n_toks > n_batch
            || seq_count >= unsafe { ffi::llama_n_seq_max(ctx.0) as usize }
        {
            let out = &mut embeddings[processed * n_embd..(processed + seq_count) * n_embd];
            batch_process(ctx.0, &mut batch.0, out, n_embd)?;
            batch_clear(&mut batch.0);
            processed += seq_count;
            seq_count = 0;
        }

        unsafe {
            batch_add_seq(&mut batch.0, &chunks[chunk_index].tokens, seq_count as i32);
        }
        seq_count += 1;
    }

    if seq_count > 0 {
        let out = &mut embeddings[processed * n_embd..(processed + seq_count) * n_embd];
        batch_process(ctx.0, &mut batch.0, out, n_embd)?;
    }

    for (i, chunk) in chunks.iter_mut().enumerate() {
        chunk.embedding = embeddings[i * n_embd..(i + 1) * n_embd].to_vec();
        chunk.tokens.clear();
    }

    let mut query_batch = Batch(unsafe { ffi::llama_batch_init(n_batch as i32, 0, 1) });
    let mut query = String::new();
    loop {
        print!("Enter query: ");
        io::stdout().flush().map_err(|err| err.to_string())?;
        query.clear();
        if io::stdin()
            .read_line(&mut query)
            .map_err(|err| err.to_string())?
            == 0
        {
            break;
        }

        let query_tokens = tokenize(vocab, query.trim_end(), true, true)?;
        unsafe {
            batch_add_seq(&mut query_batch.0, &query_tokens, 0);
        }

        let mut query_emb = vec![0.0f32; n_embd];
        batch_process(ctx.0, &mut query_batch.0, &mut query_emb, n_embd)?;
        batch_clear(&mut query_batch.0);

        println!("Top {} similar chunks:", args.top_k);
        for (index, sim) in top_k_similarities(&chunks, &query_emb, args.top_k.max(0) as usize) {
            let chunk = &chunks[index];
            println!("filename: {}", chunk.filename);
            println!("filepos: {}", chunk.filepos);
            println!("similarity: {sim}");
            println!("textdata:\n{}", chunk.textdata);
            println!("--------------------");
        }
    }

    unsafe {
        ffi::llama_perf_context_print(ctx.0);
    }
    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-retrieval".to_string());

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
    fn parses_retrieval_options() {
        let args = parse_args([
            "--model",
            "model.gguf",
            "--context-file",
            "README.md",
            "--context-file",
            "LICENSE",
            "--chunk-size",
            "100",
            "--chunk-separator",
            ".",
            "--top-k",
            "5",
            "-c",
            "256",
            "-b",
            "128",
            "-ngl",
            "0",
            "--verbose-prompt",
        ])
        .unwrap();
        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.context_files, vec!["README.md", "LICENSE"]);
        assert_eq!(args.chunk_size, 100);
        assert_eq!(args.chunk_separator, ".");
        assert_eq!(args.top_k, 5);
        assert_eq!(args.n_ctx, 256);
        assert_eq!(args.n_batch, 256);
        assert_eq!(args.n_gpu_layers, 0);
        assert!(args.verbose_prompt);
    }

    #[test]
    fn rejects_missing_context_file() {
        assert_eq!(
            parse_args(["--model", "model.gguf"]).unwrap_err(),
            ParseError::MissingContextFile
        );
    }

    #[test]
    fn chunks_text_on_separator_after_min_size() {
        let chunks = chunk_text("doc.txt", "aa.bb.cc.dd.", 5, ".");
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].textdata, "aa.bb.");
        assert_eq!(chunks[0].filepos, 0);
        assert_eq!(chunks[1].textdata, "cc.dd.");
        assert_eq!(chunks[1].filepos, 6);
    }

    #[test]
    fn normalizes_and_compares_cosine() {
        let normalized = normalize_l2(&[3.0, 4.0]);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn ranks_top_similar_chunks() {
        let chunks = vec![
            Chunk {
                filename: "a".to_string(),
                filepos: 0,
                textdata: "a".to_string(),
                tokens: Vec::new(),
                embedding: vec![1.0, 0.0],
            },
            Chunk {
                filename: "b".to_string(),
                filepos: 0,
                textdata: "b".to_string(),
                tokens: Vec::new(),
                embedding: vec![0.0, 1.0],
            },
        ];
        let ranked = top_k_similarities(&chunks, &[0.9, 0.1], 1);
        assert_eq!(ranked[0].0, 0);
    }
}
