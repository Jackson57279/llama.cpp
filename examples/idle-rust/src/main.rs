use llama_idle_rust::{parse_args, Args};
use llama_simple_rust::ffi;
use std::env;
use std::ffi::CString;
use std::thread;
use std::time::Duration;

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

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf [-ngl n_gpu_layers]");
    eprintln!();
}

fn run(args: Args) -> Result<(), String> {
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;

    unsafe {
        ffi::llama_backend_init();
        ffi::llama_numa_init(0);
    }

    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;

    let model =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model.0.is_null() {
        return Err("unable to load model".to_string());
    }

    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    let mut prompt_tokens = [unsafe { ffi::llama_vocab_bos(vocab) }];

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.no_perf = false;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create the llama_context".to_string());
    }

    let batch = unsafe { ffi::llama_batch_get_one(prompt_tokens.as_mut_ptr(), 1) };
    let memory = unsafe { ffi::llama_get_memory(ctx.0) };

    unsafe {
        ffi::llama_decode(ctx.0, batch);
        ffi::llama_memory_clear(memory, true);
        ffi::llama_synchronize(ctx.0);
    }

    const N_ITERS: i32 = 3;
    let mut t_pause_ms = 0_i64;
    while t_pause_ms <= 4000 {
        let mut t_sum_us = 0.0_f64;
        let mut t_sum2_us = 0.0_f64;

        for _ in 0..N_ITERS {
            thread::sleep(Duration::from_millis(t_pause_ms as u64));

            let t_start_us = unsafe { ffi::llama_time_us() };
            unsafe {
                ffi::llama_decode(ctx.0, batch);
                ffi::llama_synchronize(ctx.0);
            }
            let t_end_us = unsafe { ffi::llama_time_us() };

            let t_cur_us = (t_end_us - t_start_us) as f64;
            println!("  - decode time: {:8.2} ms", t_cur_us / 1000.0);

            t_sum_us += t_cur_us;
            t_sum2_us += t_cur_us * t_cur_us;

            unsafe {
                ffi::llama_memory_clear(memory, true);
                ffi::llama_synchronize(ctx.0);
            }
        }

        let t_avg_us = t_sum_us / f64::from(N_ITERS);
        let t_dev_us = ((t_sum2_us / f64::from(N_ITERS - 1))
            - (t_avg_us * t_avg_us * f64::from(N_ITERS)) / f64::from(N_ITERS - 1))
        .sqrt();

        println!(
            "iters: {:4}, pause: {:5} ms, avg decode time: {:8.2} +/- {:4.2} ms",
            N_ITERS,
            t_pause_ms,
            t_avg_us / 1000.0,
            t_dev_us / 1000.0
        );

        t_pause_ms += 800;
    }

    unsafe {
        ffi::llama_backend_free();
    }

    Ok(())
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-idle".to_string());

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
