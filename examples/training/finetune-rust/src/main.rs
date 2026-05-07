use llama_simple_rust::ffi;
use std::env;
use std::ffi::{c_void, CStr, CString};
use std::ptr;

const GGML_TYPE_F32: ffi::ggml_type = 0;
const GGML_TYPE_I32: ffi::ggml_type = 26;
const OPTIMIZER_ADAMW: ffi::ggml_opt_optimizer_type = 0;
const OPTIMIZER_SGD: ffi::ggml_opt_optimizer_type = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct LrOpt {
    pub lr0: f32,
    pub lr_min: f32,
    pub decay_epochs: f32,
    pub scale_epoch: f32,
    pub wd: f32,
    pub epochs: u32,
    pub epoch: u32,
}

impl Default for LrOpt {
    fn default() -> Self {
        Self {
            lr0: 1e-5,
            lr_min: -1.0,
            decay_epochs: -1.0,
            scale_epoch: 0.0,
            wd: 0.0,
            epochs: 2,
            epoch: 0,
        }
    }
}

impl LrOpt {
    pub fn init(&mut self) {
        if self.lr_min > 0.0 && self.lr_min < self.lr0 {
            let nhalf = (self.lr0 / self.lr_min).ln() / 2.0_f32.ln();
            let mut epochs = self.epochs as f32;
            if self.decay_epochs > 0.0 && self.decay_epochs < epochs {
                epochs = self.decay_epochs;
            } else {
                self.decay_epochs = epochs;
            }
            self.scale_epoch = nhalf / epochs;
        }
    }

    pub fn get_lr(&self, epoch: f32) -> f32 {
        if self.lr_min <= 0.0 {
            self.lr0
        } else if epoch >= self.decay_epochs {
            self.lr_min
        } else {
            self.lr0 * 0.5_f32.powf(epoch * self.scale_epoch)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub out_file: String,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_gpu_layers: i32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub lr: LrOpt,
    pub optimizer: ffi::ggml_opt_optimizer_type,
    pub val_split: f32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: String::new(),
            out_file: String::new(),
            n_ctx: 512,
            n_batch: 512,
            n_ubatch: 512,
            n_gpu_layers: 99,
            n_threads: 0,
            n_threads_batch: 0,
            lr: LrOpt::default(),
            optimizer: OPTIMIZER_ADAMW,
            val_split: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    MissingModel,
    MissingOutput,
    MissingPrompt,
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidFloat(String, String),
    InvalidValue(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m/--model model.gguf"),
            ParseError::MissingOutput => write!(f, "missing required -o/--output output.gguf"),
            ParseError::MissingPrompt => write!(f, "missing required -p/--prompt text"),
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
            "-o" | "--output" => parsed.out_file = take(&mut iter, &arg)?,
            "-c" | "--ctx-size" => parsed.n_ctx = parse_u32(&mut iter, &arg)?,
            "-b" | "--batch-size" => parsed.n_batch = parse_u32(&mut iter, &arg)?,
            "-ub" | "--ubatch-size" => parsed.n_ubatch = parse_u32(&mut iter, &arg)?,
            "-ngl" | "--gpu-layers" => parsed.n_gpu_layers = parse_i32(&mut iter, &arg)?,
            "-t" | "--threads" => parsed.n_threads = parse_i32(&mut iter, &arg)?,
            "-tb" | "--threads-batch" => parsed.n_threads_batch = parse_i32(&mut iter, &arg)?,
            "-lr" | "--learning-rate" => parsed.lr.lr0 = parse_f32(&mut iter, &arg)?,
            "-lr-min" | "--learning-rate-min" => parsed.lr.lr_min = parse_f32(&mut iter, &arg)?,
            "-lr-decay" | "--learning-rate-decay-epochs" => {
                parsed.lr.decay_epochs = parse_f32(&mut iter, &arg)?
            }
            "-wd" | "--weight-decay" => parsed.lr.wd = parse_f32(&mut iter, &arg)?,
            "--val-split" => parsed.val_split = parse_f32(&mut iter, &arg)?,
            "--epochs" => parsed.lr.epochs = parse_u32(&mut iter, &arg)?,
            "-opt" | "--optimizer" => {
                parsed.optimizer = parse_optimizer(&take(&mut iter, &arg)?)?;
            }
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
    if parsed.out_file.is_empty() {
        return Err(ParseError::MissingOutput);
    }
    if parsed.prompt.is_empty() {
        return Err(ParseError::MissingPrompt);
    }
    if parsed.n_ctx == 0 || parsed.n_batch == 0 || parsed.n_ubatch == 0 {
        return Err(ParseError::InvalidValue(
            "context, batch, and ubatch sizes must be positive".to_string(),
        ));
    }
    if parsed.lr.epochs == 0 {
        return Err(ParseError::InvalidValue(
            "--epochs must be positive".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&parsed.val_split) {
        return Err(ParseError::InvalidValue(
            "--val-split must be in [0.0, 1.0)".to_string(),
        ));
    }
    parsed.lr.init();
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

fn parse_optimizer(name: &str) -> Result<ffi::ggml_opt_optimizer_type, ParseError> {
    match name.to_ascii_lowercase().as_str() {
        "adamw" => Ok(OPTIMIZER_ADAMW),
        "sgd" => Ok(OPTIMIZER_SGD),
        _ => Err(ParseError::InvalidValue(
            "invalid --optimizer, valid options: adamw, sgd".to_string(),
        )),
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

struct Dataset(ffi::ggml_opt_dataset_t);

impl Drop for Dataset {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::ggml_opt_dataset_free(self.0);
            }
        }
    }
}

struct OptResult(ffi::ggml_opt_result_t);

impl Drop for OptResult {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::ggml_opt_result_free(self.0);
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
            false,
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
            true,
            false,
        )
    };
    if n < 0 {
        return Err("failed to tokenize prompt".to_string());
    }
    tokens.truncate(n as usize);
    Ok(tokens)
}

fn create_dataset(
    ctx: *mut ffi::llama_context,
    tokens: &[ffi::llama_token],
) -> Result<Dataset, String> {
    let ne_datapoint = unsafe { ffi::llama_n_ctx(ctx) as i64 };
    let stride = ne_datapoint / 2;
    if stride <= 0 {
        return Err("context size is too small for training dataset".to_string());
    }
    if tokens.len() as i64 <= ne_datapoint + 1 {
        return Err(format!(
            "prompt is too short for finetuning dataset: {} tokens, need more than {}",
            tokens.len(),
            ne_datapoint + 1
        ));
    }
    let ndata = (tokens.len() as i64 - ne_datapoint - 1) / stride;
    if ndata <= 0 {
        return Err("prompt did not produce any finetuning datapoints".to_string());
    }
    let dataset = Dataset(unsafe {
        ffi::ggml_opt_dataset_init(
            GGML_TYPE_I32,
            GGML_TYPE_I32,
            ne_datapoint,
            ne_datapoint,
            ndata,
            1,
        )
    });
    if dataset.0.is_null() {
        return Err("failed to create optimizer dataset".to_string());
    }

    unsafe {
        let data_ptr = ffi::ggml_get_data(ffi::ggml_opt_dataset_data(dataset.0)) as *mut i32;
        let labels_ptr = ffi::ggml_get_data(ffi::ggml_opt_dataset_labels(dataset.0)) as *mut i32;
        if data_ptr.is_null() || labels_ptr.is_null() {
            return Err("failed to access optimizer dataset tensors".to_string());
        }
        for idata in 0..ndata as usize {
            let src = tokens.as_ptr().add(idata * stride as usize);
            ptr::copy_nonoverlapping(
                src,
                data_ptr.add(idata * ne_datapoint as usize),
                ne_datapoint as usize,
            );
            ptr::copy_nonoverlapping(
                src.add(1),
                labels_ptr.add(idata * ne_datapoint as usize),
                ne_datapoint as usize,
            );
        }
    }
    Ok(dataset)
}

unsafe extern "C" fn lr_params_callback(userdata: *mut c_void) -> ffi::ggml_opt_optimizer_params {
    let mut result = ffi::ggml_opt_get_default_optimizer_params(ptr::null_mut());
    let lr = &*(userdata as *const LrOpt);
    let value = lr.get_lr(lr.epoch as f32);
    eprintln!("epoch {:.2} lr={:.2e}", lr.epoch, value);
    result.adamw.alpha = value;
    result.sgd.alpha = value;
    result.adamw.wd = lr.wd;
    result.sgd.wd = lr.wd;
    result
}

fn optimizer_name(optimizer: ffi::ggml_opt_optimizer_type) -> String {
    let ptr = unsafe { ffi::ggml_opt_optimizer_name(optimizer) };
    if ptr.is_null() {
        optimizer.to_string()
    } else {
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }
}

fn run(mut args: Args) -> Result<(), String> {
    let _backend = Backend::init();
    let model_path = CString::new(args.model_path.as_str())
        .map_err(|_| "model path contains an interior NUL byte".to_string())?;
    let out_file = CString::new(args.out_file.as_str())
        .map_err(|_| "output path contains an interior NUL byte".to_string())?;

    let mut model_params = unsafe { ffi::llama_model_default_params() };
    model_params.n_gpu_layers = args.n_gpu_layers;
    model_params.use_mmap = false;

    let model =
        Model(unsafe { ffi::llama_model_load_from_file(model_path.as_ptr(), model_params) });
    if model.0.is_null() {
        return Err("unable to load model".to_string());
    }

    let mut ctx_params = unsafe { ffi::llama_context_default_params() };
    ctx_params.n_ctx = args.n_ctx;
    ctx_params.n_batch = args.n_batch;
    ctx_params.n_ubatch = args.n_ubatch;
    ctx_params.n_threads = args.n_threads;
    ctx_params.n_threads_batch = args.n_threads_batch;
    ctx_params.type_k = GGML_TYPE_F32;
    ctx_params.type_v = GGML_TYPE_F32;

    let ctx = Context(unsafe { ffi::llama_init_from_model(model.0, ctx_params) });
    if ctx.0.is_null() {
        return Err("failed to create llama_context".to_string());
    }

    let vocab = unsafe { ffi::llama_model_get_vocab(model.0) };
    if vocab.is_null() {
        return Err("failed to get model vocabulary".to_string());
    }
    let tokens = tokenize(vocab, &args.prompt)?;
    let dataset = create_dataset(ctx.0, &tokens)?;

    eprintln!(
        "-optimizer {} -lr0 {:.2e} -wd {:.2e} -lr-min {:.2e} -min-epochs {:.2e} -epochs {} -period {:.2e} -val {:.2e}",
        optimizer_name(args.optimizer),
        args.lr.lr0,
        args.lr.wd,
        args.lr.lr_min,
        args.lr.decay_epochs,
        args.lr.epochs,
        args.n_batch as f32 / args.n_ubatch as f32,
        args.val_split
    );

    let lopt_params = ffi::llama_opt_params {
        n_ctx_train: 0,
        param_filter: Some(ffi::llama_opt_param_filter_all),
        param_filter_ud: ptr::null_mut(),
        get_opt_pars: Some(lr_params_callback),
        get_opt_pars_ud: (&mut args.lr as *mut LrOpt).cast::<c_void>(),
        optimizer_type: args.optimizer,
    };
    unsafe {
        ffi::llama_opt_init(ctx.0, model.0, lopt_params);
    }

    let ndata = unsafe { ffi::ggml_opt_dataset_ndata(dataset.0) };
    let idata_split = (ndata as f32 * (1.0 - args.val_split)) as i64;
    let result_train = OptResult(unsafe { ffi::ggml_opt_result_init() });
    let result_eval = OptResult(unsafe { ffi::ggml_opt_result_init() });
    if result_train.0.is_null() || result_eval.0.is_null() {
        return Err("failed to create optimizer results".to_string());
    }

    for epoch in 0..args.lr.epochs {
        args.lr.epoch = epoch;
        unsafe {
            ffi::llama_opt_epoch(
                ctx.0,
                dataset.0,
                result_train.0,
                result_eval.0,
                idata_split,
                Some(ffi::ggml_opt_epoch_callback_progress_bar),
                Some(ffi::ggml_opt_epoch_callback_progress_bar),
            );
            eprintln!();
            ffi::ggml_opt_result_reset(result_train.0);
            ffi::ggml_opt_result_reset(result_eval.0);
        }
    }

    unsafe {
        ffi::llama_model_save_to_file(model.0, out_file.as_ptr());
    }
    Ok(())
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("example usage:");
    eprintln!();
    eprintln!("    {program} -m model.gguf -p \"training text\" -o finetuned.gguf");
    eprintln!();
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "llama-finetune".to_string());
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
    fn parses_finetune_options() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-p",
            "training data",
            "-o",
            "out.gguf",
            "-c",
            "256",
            "-b",
            "128",
            "-ub",
            "64",
            "-ngl",
            "0",
            "-t",
            "2",
            "-tb",
            "4",
            "-lr",
            "0.01",
            "-lr-min",
            "0.001",
            "-lr-decay",
            "3",
            "-wd",
            "0.0001",
            "--val-split",
            "0.2",
            "--epochs",
            "4",
            "-opt",
            "sgd",
        ])
        .unwrap();
        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "training data");
        assert_eq!(args.out_file, "out.gguf");
        assert_eq!(args.n_ctx, 256);
        assert_eq!(args.n_batch, 128);
        assert_eq!(args.n_ubatch, 64);
        assert_eq!(args.n_gpu_layers, 0);
        assert_eq!(args.n_threads, 2);
        assert_eq!(args.n_threads_batch, 4);
        assert_eq!(args.lr.lr0, 0.01);
        assert_eq!(args.lr.lr_min, 0.001);
        assert_eq!(args.lr.decay_epochs, 3.0);
        assert_eq!(args.lr.wd, 0.0001);
        assert_eq!(args.val_split, 0.2);
        assert_eq!(args.lr.epochs, 4);
        assert_eq!(args.optimizer, OPTIMIZER_SGD);
    }

    #[test]
    fn rejects_missing_output() {
        assert_eq!(
            parse_args(["-m", "model.gguf", "-p", "text"]).unwrap_err(),
            ParseError::MissingOutput
        );
    }

    #[test]
    fn rejects_unknown_optimizer() {
        assert_eq!(
            parse_args([
                "-m",
                "model.gguf",
                "-p",
                "text",
                "-o",
                "out.gguf",
                "-opt",
                "rmsprop"
            ])
            .unwrap_err(),
            ParseError::InvalidValue("invalid --optimizer, valid options: adamw, sgd".to_string())
        );
    }

    #[test]
    fn learning_rate_decays_like_common_helper() {
        let mut lr = LrOpt {
            lr0: 1e-2,
            lr_min: 1e-3,
            epochs: 4,
            ..LrOpt::default()
        };
        lr.init();
        assert!((lr.get_lr(0.0) - 1e-2).abs() < 1e-8);
        assert!((lr.get_lr(4.0) - 1e-3).abs() < 1e-8);
    }
}
