use llama_lookup_merge::{
    draft_tokens, load_cache, merge_cache, update_cache, NgramCache, LLAMA_NGRAM_MAX,
    LLAMA_NGRAM_MIN,
};
use llama_lookup_stats_rust::{compute_acceptance, parse_args, Args};
use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::time::Instant;

struct Backend;

impl Backend {
    fn init() -> Self {
        unsafe {
            ffi::llama_backend_init();
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

fn tokenize(
    vocab: *const ffi::llama_vocab,
    text: &[u8],
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<ffi::llama_token>, String> {
    let mut tokens = vec![0_i32; text.len() + 2 * usize::from(add_special)];
    let n = unsafe {
        ffi::llama_tokenize(
            vocab,
            text.as_ptr() as *const i8,
            text.len() as i32,
            tokens.as_mut_ptr(),
            tokens.len() as i32,
            add_special,
            parse_special,
        )
    };
    let n = if n < 0 {
        tokens.resize((-n) as usize, 0);
        let check = unsafe {
            ffi::llama_tokenize(
                vocab,
                text.as_ptr() as *const i8,
                text.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
                parse_special,
            )
        };
        if check != -n {
            return Err("tokenization size changed between calls".to_string());
        }
        -n
    } else {
        n
    };
    tokens.truncate(n as usize);
    Ok(tokens)
}

fn load_model(args: &Args) -> Result<(Model, Context), String> {
    let path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;
    let mut mparams = unsafe { ffi::llama_model_default_params() };
    mparams.n_gpu_layers = args.n_gpu_layers;
    let model = Model(unsafe { ffi::llama_model_load_from_file(path.as_ptr(), mparams) });
    if model.0.is_null() {
        return Err(format!("failed to load model '{}'", args.model_path));
    }
    let cparams = unsafe { ffi::llama_context_default_params() };
    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, cparams) });
    if ctx.0.is_null() {
        return Err("failed to create context".to_string());
    }
    Ok((model, ctx))
}

fn run(args: Args) -> Result<i32, String> {
    let _backend = Backend::init();
    let (model, ctx) = load_model(&args)?;
    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        return Err("failed to get vocab".to_string());
    }
    let input = tokenize(vocab, args.prompt.as_bytes(), true, true)?;

    let mut context_cache = NgramCache::new();
    let mut dynamic_cache = NgramCache::new();
    let mut static_cache = NgramCache::new();

    let flat_start = Instant::now();
    if let Some(path) = &args.lookup_cache_static {
        static_cache = load_cache(path)
            .map_err(|err| format!("failed to open static lookup cache: {path}: {err}"))?;
    }
    if let Some(path) = &args.lookup_cache_dynamic {
        if let Ok(cache) = load_cache(path) {
            dynamic_cache = cache;
        }
    }
    let t_draft_flat = flat_start.elapsed();

    let n_input = input.len();
    let n_ctx = unsafe { ffi::llama_n_ctx(ctx.0) as usize };
    let mut n_drafted = 0_i32;
    let mut n_accept = 0_i32;
    let mut t_draft = std::time::Duration::ZERO;
    let start = Instant::now();

    let mut i_start = 0;
    while i_start + n_ctx < n_input {
        let input_slice = &input[i_start..i_start + n_ctx];
        let mut pseudo_output = vec![input_slice[0]];

        while pseudo_output.len() < n_ctx {
            let mut draft = vec![*pseudo_output.last().unwrap()];
            let draft_start = Instant::now();
            draft_tokens(
                &pseudo_output,
                &mut draft,
                args.n_draft,
                LLAMA_NGRAM_MIN,
                LLAMA_NGRAM_MAX,
                &context_cache,
                &dynamic_cache,
                &static_cache,
            );
            t_draft += draft_start.elapsed();
            n_drafted += (draft.len() - 1) as i32;

            for &drafted in draft.iter().skip(1) {
                if pseudo_output.len() >= n_ctx {
                    break;
                }
                let ground_truth = input_slice[pseudo_output.len()];
                if ground_truth != drafted {
                    break;
                }
                n_accept += 1;
                pseudo_output.push(ground_truth);
                let update_start = Instant::now();
                update_cache(
                    &mut context_cache,
                    LLAMA_NGRAM_MIN,
                    LLAMA_NGRAM_MAX,
                    &pseudo_output,
                    1,
                );
                t_draft += update_start.elapsed();
            }

            if pseudo_output.len() < n_ctx {
                pseudo_output.push(input_slice[pseudo_output.len()]);
                let update_start = Instant::now();
                update_cache(
                    &mut context_cache,
                    LLAMA_NGRAM_MIN,
                    LLAMA_NGRAM_MAX,
                    &pseudo_output,
                    1,
                );
                t_draft += update_start.elapsed();
            }
        }

        if i_start > 0 && i_start / 100_000 != i_start.saturating_sub(n_ctx) / 100_000 {
            let elapsed_ms = start.elapsed().as_millis() as usize;
            let eta_ms = (n_input - i_start) * elapsed_ms / i_start;
            eprintln!(
                "lookup-stats: {i_start}/{n_input} done, ETA: {:02}:{:02}",
                eta_ms / 60_000,
                (eta_ms % 60_000) / 1000
            );
        }

        merge_cache(&mut dynamic_cache, std::mem::take(&mut context_cache));
        i_start += n_ctx;
    }

    println!();
    println!("n_draft      = {}", args.n_draft);
    println!("n_predict    = {}", n_input - n_input % n_ctx);
    println!("n_drafted    = {n_drafted}");
    println!(
        "t_draft_flat = {:.2} ms",
        t_draft_flat.as_secs_f64() * 1000.0
    );
    let draft_us = t_draft.as_secs_f64() * 1_000_000.0;
    if n_drafted > 0 && draft_us > 0.0 {
        println!(
            "t_draft      = {:.2} ms, {:.2} us per token, {:.2} tokens per second",
            draft_us * 1e-3,
            draft_us / f64::from(n_drafted),
            f64::from(n_drafted) / (draft_us * 1e-6)
        );
    } else {
        println!(
            "t_draft      = {:.2} ms, n/a us per token, n/a tokens per second",
            draft_us * 1e-3
        );
    }
    println!("n_accept     = {n_accept}");
    if let Some(accept) = compute_acceptance(n_accept, n_drafted) {
        println!("accept       = {accept:.3}%");
    } else {
        println!("accept       = n/a");
    }
    println!();

    Ok(0)
}

fn main() {
    let mut argv = env::args();
    let program = argv
        .next()
        .unwrap_or_else(|| "llama-lookup-stats".to_string());
    let args = match parse_args(argv) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("Usage: {program} -m <model> [-p prompt] [-lcs cache] [-lcd cache] [--spec-draft-n-max N]");
            eprintln!("{program}: {err}");
            std::process::exit(1);
        }
    };

    match run(args) {
        Ok(code) => std::process::exit(code),
        Err(err) => {
            eprintln!("{program}: error: {err}");
            std::process::exit(1);
        }
    }
}
