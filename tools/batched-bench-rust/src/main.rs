use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub model_path: String,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_gpu_layers: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub flash_attn_type: i32,
    pub kv_unified: bool,
    pub is_pp_shared: bool,
    pub is_tg_separate: bool,
    pub output_jsonl: bool,
    pub n_pp: Vec<i32>,
    pub n_tg: Vec<i32>,
    pub n_pl: Vec<i32>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            n_ctx: 2048,
            n_batch: 2048,
            n_ubatch: 512,
            n_gpu_layers: 99,
            n_threads: 4,
            n_threads_batch: 4,
            flash_attn_type: 0,
            kv_unified: false,
            is_pp_shared: false,
            is_tg_separate: false,
            output_jsonl: false,
            n_pp: vec![128],
            n_tg: vec![128],
            n_pl: vec![1],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    MissingModel,
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidList(String, String),
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingModel => write!(f, "missing required -m/--model model.gguf"),
            Self::MissingValue(flag) => write!(f, "missing value for {flag}"),
            Self::InvalidInteger(flag, value) => write!(f, "invalid integer for {flag}: {value}"),
            Self::InvalidList(flag, value) => write!(f, "invalid comma list for {flag}: {value}"),
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
            "-c" | "--ctx-size" => parsed.n_ctx = parse_u32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_u32(&mut iter, &arg)?,
            "-ub" | "--ubatch-size" => parsed.n_ubatch = parse_u32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "-t" | "--threads" => parsed.n_threads = parse_i32(&mut iter, &arg)?,
            "-tb" | "--threads-batch" => parsed.n_threads_batch = parse_i32(&mut iter, &arg)?,
            "-fa" | "--flash-attn" => parsed.flash_attn_type = 1,
            "--no-flash-attn" => parsed.flash_attn_type = 0,
            "--kv-unified" => parsed.kv_unified = true,
            "--no-kv-unified" => parsed.kv_unified = false,
            "-pps" => parsed.is_pp_shared = true,
            "-tgs" => parsed.is_tg_separate = true,
            "-npp" => parsed.n_pp = parse_list(&mut iter, &arg)?,
            "-ntg" => parsed.n_tg = parse_list(&mut iter, &arg)?,
            "-npl" => parsed.n_pl = parse_list(&mut iter, &arg)?,
            "--output-format" => {
                let value = take(&mut iter, &arg)?;
                parsed.output_jsonl = match value.as_str() {
                    "jsonl" => true,
                    "md" => false,
                    _ => {
                        return Err(ParseError::InvalidValue(format!(
                            "--output-format must be md or jsonl, got {value}"
                        )))
                    }
                };
            }
            "--jsonl" => parsed.output_jsonl = true,
            "-h" | "--help" => return Err(ParseError::MissingModel),
            other => {
                return Err(ParseError::InvalidValue(format!(
                    "unknown argument: {other}"
                )))
            }
        }
    }

    if parsed.model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }
    if parsed.n_ctx == 0 || parsed.n_batch == 0 || parsed.n_ubatch == 0 {
        return Err(ParseError::InvalidValue(
            "context and batch sizes must be positive".to_string(),
        ));
    }
    for (name, values) in [
        ("-npp", &parsed.n_pp),
        ("-ntg", &parsed.n_tg),
        ("-npl", &parsed.n_pl),
    ] {
        if values.is_empty() || values.iter().any(|&v| v <= 0) {
            return Err(ParseError::InvalidValue(format!(
                "{name} must contain positive integers"
            )));
        }
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

fn parse_list<I>(iter: &mut I, flag: &str) -> Result<Vec<i32>, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .split(',')
        .map(|part| {
            part.parse::<i32>()
                .map_err(|_| ParseError::InvalidList(flag.to_string(), value.clone()))
        })
        .collect()
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

#[derive(Clone)]
struct SmallRng(u64);

impl SmallRng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_token(&mut self, n_vocab: i32) -> ffi::llama_token {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as i32).rem_euclid(n_vocab)
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
    *batch.logits.offset(i) = i8::from(logits);
    batch.n_tokens += 1;
}

fn decode_helper(
    ctx: *mut ffi::llama_context,
    batch: &ffi::llama_batch,
    n_batch: i32,
    synchronize: bool,
) -> Result<(), String> {
    let mut i = 0;
    while i < batch.n_tokens {
        let n_tokens = n_batch.min(batch.n_tokens - i);
        let offset = i as isize;
        let batch_view = ffi::llama_batch {
            n_tokens,
            token: unsafe { batch.token.offset(offset) },
            embd: std::ptr::null_mut(),
            pos: unsafe { batch.pos.offset(offset) },
            n_seq_id: unsafe { batch.n_seq_id.offset(offset) },
            seq_id: unsafe { batch.seq_id.offset(offset) },
            logits: unsafe { batch.logits.offset(offset) },
        };
        let ret = unsafe { ffi::llama_decode(ctx, batch_view) };
        if ret != 0 {
            return Err(format!(
                "failed to decode the batch, n_batch = {n_batch}, ret = {ret}"
            ));
        }
        if synchronize {
            unsafe {
                ffi::llama_synchronize(ctx);
            }
        }
        i += n_batch;
    }
    Ok(())
}

fn required_context(args: &Args, pp: i32, tg: i32, pl: i32) -> i32 {
    if args.is_pp_shared {
        (if args.kv_unified { pp } else { pl * pp }) + pl * tg
    } else {
        pl * (pp + tg)
    }
}

fn run(args: Args) -> Result<(), String> {
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;
    let _backend = Backend::init();

    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;

    let model =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model.0.is_null() {
        return Err("unable to load model".to_string());
    }

    let max_pl = args.n_pl.iter().copied().max().unwrap_or(1).max(1);
    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = args.n_batch;
    ctx_params.n_ubatch = args.n_ubatch;
    ctx_params.n_seq_max = max_pl as u32;
    ctx_params.n_threads = args.n_threads;
    ctx_params.n_threads_batch = args.n_threads_batch;
    ctx_params.flash_attn_type = args.flash_attn_type;
    ctx_params.kv_unified = args.kv_unified;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create the llama_context".to_string());
    }

    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    let n_vocab = unsafe { ffi::llama_vocab_n_tokens(vocab) };
    let mem = unsafe { ffi::llama_get_memory(ctx.0) };
    let n_kv_max = unsafe { ffi::llama_n_ctx(ctx.0) as i32 };
    let mut batch = Batch(unsafe { ffi::llama_batch_init(n_kv_max, 0, max_pl) });
    let mut rng = SmallRng::new(0xC0FFEE);

    for i in 0..16 {
        unsafe {
            batch_add(&mut batch.0, rng.next_token(n_vocab), i, &[0], false);
        }
    }
    decode_helper(ctx.0, &batch.0, args.n_batch as i32, true)?;

    if !args.output_jsonl {
        println!();
        println!(
            "main: n_kv_max = {n_kv_max}, n_batch = {}, n_ubatch = {}, flash_attn = {}, is_pp_shared = {}, is_tg_separate = {}, n_gpu_layers = {}, n_threads = {}, n_threads_batch = {}",
            args.n_batch,
            args.n_ubatch,
            args.flash_attn_type,
            i32::from(args.is_pp_shared),
            i32::from(args.is_tg_separate),
            args.n_gpu_layers,
            ctx_params.n_threads,
            ctx_params.n_threads_batch
        );
        println!();
        println!(
            "|{:>6} | {:>6} | {:>4} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} |",
            "PP", "TG", "B", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "T s", "S t/s"
        );
        println!("|------|--------|------|--------|----------|----------|----------|----------|----------|----------|");
    }

    for &pp in &args.n_pp {
        for &tg in &args.n_tg {
            for &pl in &args.n_pl {
                let n_ctx_req = required_context(&args, pp, tg, pl);
                if n_ctx_req > n_kv_max {
                    continue;
                }

                batch_clear(&mut batch.0);
                let prompt_copies = if args.is_pp_shared { 1 } else { pl };
                for j in 0..prompt_copies {
                    for i in 0..pp {
                        unsafe {
                            batch_add(&mut batch.0, rng.next_token(n_vocab), i, &[j], i == pp - 1);
                        }
                    }
                }

                unsafe {
                    ffi::llama_memory_clear(mem, false);
                }
                let t_pp_start = unsafe { ffi::ggml_time_us() };
                decode_helper(ctx.0, &batch.0, args.n_batch as i32, false)?;
                unsafe {
                    ffi::llama_synchronize(ctx.0);
                }
                let t_pp_end = unsafe { ffi::ggml_time_us() };

                if args.is_pp_shared {
                    for i in 1..pl {
                        unsafe {
                            ffi::llama_memory_seq_cp(mem, 0, i, -1, -1);
                        }
                    }
                    if !args.kv_unified {
                        batch_clear(&mut batch.0);
                        unsafe {
                            batch_add(&mut batch.0, rng.next_token(n_vocab), pp, &[0], true);
                        }
                        decode_helper(ctx.0, &batch.0, args.n_batch as i32, true)?;
                        unsafe {
                            ffi::llama_memory_seq_rm(mem, 0, pp, -1);
                        }
                    }
                }

                let t_tg_start = unsafe { ffi::ggml_time_us() };
                if args.is_tg_separate {
                    for j in 0..pl {
                        for i in 0..tg {
                            batch_clear(&mut batch.0);
                            unsafe {
                                batch_add(
                                    &mut batch.0,
                                    rng.next_token(n_vocab),
                                    pp + i,
                                    &[j],
                                    true,
                                );
                            }
                            decode_helper(ctx.0, &batch.0, args.n_batch as i32, true)?;
                        }
                    }
                } else {
                    for i in 0..tg {
                        batch_clear(&mut batch.0);
                        for j in 0..pl {
                            unsafe {
                                batch_add(
                                    &mut batch.0,
                                    rng.next_token(n_vocab),
                                    pp + i,
                                    &[j],
                                    true,
                                );
                            }
                        }
                        decode_helper(ctx.0, &batch.0, args.n_batch as i32, true)?;
                    }
                }
                let t_tg_end = unsafe { ffi::ggml_time_us() };

                let t_pp = (t_pp_end - t_pp_start) as f32 / 1_000_000.0;
                let t_tg = (t_tg_end - t_tg_start) as f32 / 1_000_000.0;
                let t = t_pp + t_tg;
                let speed_pp = if args.is_pp_shared {
                    pp as f32
                } else {
                    (pl * pp) as f32
                } / t_pp.max(f32::EPSILON);
                let speed_tg = (pl * tg) as f32 / t_tg.max(f32::EPSILON);
                let speed = ((if args.is_pp_shared { pp } else { pl * pp }) + pl * tg) as f32
                    / t.max(f32::EPSILON);

                if args.output_jsonl {
                    println!(
                        "{{\"n_kv_max\": {n_kv_max}, \"n_batch\": {}, \"n_ubatch\": {}, \"flash_attn\": {}, \"is_pp_shared\": {}, \"n_gpu_layers\": {}, \"n_threads\": {}, \"n_threads_batch\": {}, \"pp\": {pp}, \"tg\": {tg}, \"pl\": {pl}, \"n_kv\": {n_ctx_req}, \"t_pp\": {t_pp}, \"speed_pp\": {speed_pp}, \"t_tg\": {t_tg}, \"speed_tg\": {speed_tg}, \"t\": {t}, \"speed\": {speed}}}",
                        args.n_batch,
                        args.n_ubatch,
                        args.flash_attn_type,
                        i32::from(args.is_pp_shared),
                        args.n_gpu_layers,
                        ctx_params.n_threads,
                        ctx_params.n_threads_batch
                    );
                } else {
                    println!(
                        "|{pp:6} | {tg:6} | {pl:4} | {n_ctx_req:6} | {t_pp:8.3} | {speed_pp:8.2} | {t_tg:8.3} | {speed_tg:8.2} | {t:8.3} | {speed:8.2} |"
                    );
                }
            }
        }
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
    eprintln!("    {program} -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]");
    eprintln!();
}

fn main() {
    let mut argv = env::args();
    let program = argv
        .next()
        .unwrap_or_else(|| "llama-batched-bench".to_string());

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
    fn parses_benchmark_lists() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-npp",
            "128,256",
            "-ntg",
            "16,32",
            "-npl",
            "1,4",
            "-pps",
            "-tgs",
            "--output-format",
            "jsonl",
        ])
        .unwrap();
        assert_eq!(args.n_pp, vec![128, 256]);
        assert_eq!(args.n_tg, vec![16, 32]);
        assert_eq!(args.n_pl, vec![1, 4]);
        assert!(args.is_pp_shared);
        assert!(args.is_tg_separate);
        assert!(args.output_jsonl);
    }

    #[test]
    fn rejects_invalid_list() {
        assert_eq!(
            parse_args(["-m", "model.gguf", "-npp", "1,nope"]).unwrap_err(),
            ParseError::InvalidList("-npp".to_string(), "1,nope".to_string())
        );
    }

    #[test]
    fn computes_required_context_for_shared_prompt() {
        let mut args = Args::default();
        args.is_pp_shared = true;
        args.kv_unified = false;
        assert_eq!(required_context(&args, 10, 3, 4), 52);
        args.kv_unified = true;
        assert_eq!(required_context(&args, 10, 3, 4), 22);
    }

    #[test]
    fn rejects_missing_model() {
        assert_eq!(
            parse_args(["-npp", "1"]).unwrap_err(),
            ParseError::MissingModel
        );
    }
}
