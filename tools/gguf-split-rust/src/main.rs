use llama_simple_rust::ffi;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::{env, ptr};

const GGUF_DEFAULT_ALIGNMENT: usize = 32;
const LLM_KV_SPLIT_NO: &str = "split.no";
const LLM_KV_SPLIT_COUNT: &str = "split.count";
const LLM_KV_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitOperation {
    Split,
    Merge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitMode {
    Tensor,
    Size,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitParams {
    pub operation: SplitOperation,
    pub mode: SplitMode,
    pub n_bytes_split: usize,
    pub n_split_tensors: i32,
    pub input: String,
    pub output: String,
    pub no_tensor_first_split: bool,
    pub dry_run: bool,
}

impl Default for SplitParams {
    fn default() -> Self {
        Self {
            operation: SplitOperation::Split,
            mode: SplitMode::Tensor,
            n_bytes_split: 0,
            n_split_tensors: 128,
            input: String::new(),
            output: String::new(),
            no_tensor_first_split: false,
            dry_run: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingValue(String),
    UnknownArgument(String),
    ConflictingOperation,
    ConflictingMode,
    BadArguments,
    InvalidInteger(String, String),
    InvalidSize(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingValue(flag) => write!(f, "missing value for {flag}"),
            Self::UnknownArgument(arg) => write!(f, "unknown argument: {arg}"),
            Self::ConflictingOperation => write!(
                f,
                "either --split or --merge can be specified, but not both"
            ),
            Self::ConflictingMode => write!(
                f,
                "either --split-max-tensors or --split-max-size can be specified, but not both"
            ),
            Self::BadArguments => write!(f, "bad arguments"),
            Self::InvalidInteger(flag, value) => write!(f, "invalid integer for {flag}: {value}"),
            Self::InvalidSize(value) => write!(f, "invalid split size: {value}"),
        }
    }
}

impl std::error::Error for ParseError {}

pub fn split_str_to_n_bytes(value: &str) -> Result<usize, ParseError> {
    let (number, multiplier) = match value.as_bytes().last().copied() {
        Some(b'M') => (&value[..value.len() - 1], 1_000_000_usize),
        Some(b'G') => (&value[..value.len() - 1], 1_000_000_000_usize),
        _ => return Err(ParseError::InvalidSize(value.to_string())),
    };
    let n = number
        .parse::<usize>()
        .map_err(|_| ParseError::InvalidSize(value.to_string()))?;
    if n == 0 {
        return Err(ParseError::InvalidSize(value.to_string()));
    }
    n.checked_mul(multiplier)
        .ok_or_else(|| ParseError::InvalidSize(value.to_string()))
}

pub fn parse_args<I, S>(args: I) -> Result<SplitParams, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut params = SplitParams::default();
    let mut operation_seen = false;
    let mut mode_seen = false;
    let mut positionals = Vec::new();
    let mut iter = args.into_iter().map(Into::into).peekable();

    while let Some(raw_arg) = iter.next() {
        let arg = raw_arg.replace('_', "-");
        match arg.as_str() {
            "--dry-run" => params.dry_run = true,
            "--no-tensor-first-split" => params.no_tensor_first_split = true,
            "--merge" => {
                if operation_seen && params.operation != SplitOperation::Merge {
                    return Err(ParseError::ConflictingOperation);
                }
                operation_seen = true;
                params.operation = SplitOperation::Merge;
            }
            "--split" => {
                if operation_seen && params.operation != SplitOperation::Split {
                    return Err(ParseError::ConflictingOperation);
                }
                operation_seen = true;
                params.operation = SplitOperation::Split;
            }
            "--split-max-tensors" => {
                if mode_seen && params.mode != SplitMode::Tensor {
                    return Err(ParseError::ConflictingMode);
                }
                mode_seen = true;
                params.mode = SplitMode::Tensor;
                let value = take(&mut iter, "--split-max-tensors")?;
                params.n_split_tensors = value.parse().map_err(|_| {
                    ParseError::InvalidInteger("--split-max-tensors".to_string(), value)
                })?;
            }
            "--split-max-size" => {
                if mode_seen && params.mode != SplitMode::Size {
                    return Err(ParseError::ConflictingMode);
                }
                mode_seen = true;
                params.mode = SplitMode::Size;
                let value = take(&mut iter, "--split-max-size")?;
                params.n_bytes_split = split_str_to_n_bytes(&value)?;
            }
            "-h" | "--help" | "--version" => return Err(ParseError::UnknownArgument(arg)),
            other if other.starts_with("--") => {
                return Err(ParseError::UnknownArgument(other.to_string()))
            }
            _ => {
                positionals.push(raw_arg);
                positionals.extend(iter);
                break;
            }
        }
    }

    if positionals.len() != 2 {
        return Err(ParseError::BadArguments);
    }
    params.input = positionals.remove(0);
    params.output = positionals.remove(0);
    Ok(params)
}

fn take<I>(iter: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, ParseError>
where
    I: Iterator<Item = String>,
{
    iter.next()
        .ok_or_else(|| ParseError::MissingValue(flag.to_string()))
}

fn pad(n: usize, alignment: usize) -> usize {
    (n + alignment - 1) & !(alignment - 1)
}

struct GgmlContext(*mut ffi::ggml_context);

impl Drop for GgmlContext {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::ggml_free(self.0);
            }
        }
    }
}

struct GgufContext(*mut ffi::gguf_context);

impl Drop for GgufContext {
    fn drop(&mut self) {
        unsafe {
            if !self.0.is_null() {
                ffi::gguf_free(self.0);
            }
        }
    }
}

struct LoadedGguf {
    gguf: GgufContext,
    meta: GgmlContext,
}

fn load_gguf(path: &str) -> Result<LoadedGguf, String> {
    let path_c = CString::new(path).map_err(|_| "path contains NUL byte".to_string())?;
    let mut ctx_meta: *mut ffi::ggml_context = ptr::null_mut();
    let params = ffi::gguf_init_params {
        no_alloc: true,
        ctx: &mut ctx_meta,
    };
    let ctx_gguf = unsafe { ffi::gguf_init_from_file(path_c.as_ptr(), params) };
    if ctx_gguf.is_null() {
        return Err(format!("failed to load input GGUF from {path}"));
    }
    Ok(LoadedGguf {
        gguf: GgufContext(ctx_gguf),
        meta: GgmlContext(ctx_meta),
    })
}

fn cstr(ptr: *const std::ffi::c_char) -> Result<String, String> {
    if ptr.is_null() {
        Err("unexpected null string".to_string())
    } else {
        Ok(unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned())
    }
}

fn cstring(value: &str) -> Result<CString, String> {
    CString::new(value).map_err(|_| format!("string contains NUL byte: {value:?}"))
}

fn key(name: &str) -> CString {
    CString::new(name).unwrap()
}

fn tensor_name(ctx: *const ffi::gguf_context, i: i64) -> Result<String, String> {
    cstr(unsafe { ffi::gguf_get_tensor_name(ctx, i) })
}

fn tensor_by_name(
    ctx_meta: *mut ffi::ggml_context,
    name: &str,
) -> Result<*mut ffi::ggml_tensor, String> {
    let name_c = cstring(name)?;
    let tensor = unsafe { ffi::ggml_get_tensor(ctx_meta, name_c.as_ptr()) };
    if tensor.is_null() {
        Err(format!("failed to find tensor metadata for {name}"))
    } else {
        Ok(tensor)
    }
}

fn split_path(prefix: &str, split_no: i32, split_count: i32) -> Result<String, String> {
    let prefix_c = cstring(prefix)?;
    let mut buf = vec![0_i8; 4096];
    let ret = unsafe {
        ffi::llama_split_path(
            buf.as_mut_ptr(),
            buf.len(),
            prefix_c.as_ptr(),
            split_no,
            split_count,
        )
    };
    if ret <= 0 {
        return Err(format!("failed to construct split path for {prefix}"));
    }
    cstr(buf.as_ptr())
}

fn split_prefix(path: &str, split_no: i32, split_count: i32) -> Result<String, String> {
    let path_c = cstring(path)?;
    let mut buf = vec![0_i8; 4096];
    let ret = unsafe {
        ffi::llama_split_prefix(
            buf.as_mut_ptr(),
            buf.len(),
            path_c.as_ptr(),
            split_no,
            split_count,
        )
    };
    if ret == 0 {
        return Err(format!(
            "unexpected input file name: {path} i_split={split_no} n_split={split_count}"
        ));
    }
    cstr(buf.as_ptr())
}

fn copy_range(
    input: &mut File,
    output: &mut File,
    offset: u64,
    len: usize,
    buf: &mut Vec<u8>,
) -> Result<(), String> {
    if buf.len() < len {
        buf.resize(len, 0);
    }
    input
        .seek(SeekFrom::Start(offset))
        .map_err(|err| err.to_string())?;
    input
        .read_exact(&mut buf[..len])
        .map_err(|err| err.to_string())?;
    output.write_all(&buf[..len]).map_err(|err| err.to_string())
}

fn write_zeros(output: &mut File, n: usize) -> Result<(), String> {
    const ZEROS: [u8; 4096] = [0; 4096];
    let mut remaining = n;
    while remaining > 0 {
        let chunk = remaining.min(ZEROS.len());
        output
            .write_all(&ZEROS[..chunk])
            .map_err(|err| err.to_string())?;
        remaining -= chunk;
    }
    Ok(())
}

fn should_split(params: &SplitParams, i_tensor: i64, n_tensors: i64, next_size: usize) -> bool {
    match params.mode {
        SplitMode::Size => next_size > params.n_bytes_split,
        SplitMode::Tensor => {
            i_tensor > 0
                && i_tensor < n_tensors
                && i_tensor % i64::from(params.n_split_tensors) == 0
        }
    }
}

fn plan_splits(params: &SplitParams, loaded: &LoadedGguf) -> Result<Vec<GgufContext>, String> {
    let n_tensors = unsafe { ffi::gguf_get_n_tensors(loaded.gguf.0) };
    let mut ctx_outs = Vec::new();
    let mut curr = unsafe { ffi::gguf_init_empty() };
    let mut i_split = 0_i32;

    unsafe {
        ffi::gguf_set_kv(curr, loaded.gguf.0);
        ffi::gguf_set_val_u16(curr, key(LLM_KV_SPLIT_NO).as_ptr(), i_split as u16);
        ffi::gguf_set_val_u16(curr, key(LLM_KV_SPLIT_COUNT).as_ptr(), 0);
        ffi::gguf_set_val_i32(
            curr,
            key(LLM_KV_SPLIT_TENSORS_COUNT).as_ptr(),
            n_tensors as i32,
        );
    }

    let mut current_size = 0_usize;
    if params.no_tensor_first_split {
        ctx_outs.push(GgufContext(curr));
        i_split += 1;
        curr = unsafe { ffi::gguf_init_empty() };
        unsafe {
            ffi::gguf_set_val_u16(curr, key(LLM_KV_SPLIT_NO).as_ptr(), i_split as u16);
            ffi::gguf_set_val_u16(curr, key(LLM_KV_SPLIT_COUNT).as_ptr(), 0);
            ffi::gguf_set_val_i32(
                curr,
                key(LLM_KV_SPLIT_TENSORS_COUNT).as_ptr(),
                n_tensors as i32,
            );
        }
    }

    for i in 0..n_tensors {
        let name = tensor_name(loaded.gguf.0, i)?;
        let tensor = tensor_by_name(loaded.meta.0, &name)?;
        let n_bytes = unsafe { ffi::ggml_nbytes(tensor) };
        let next_size = current_size + pad(n_bytes, GGUF_DEFAULT_ALIGNMENT);
        if should_split(params, i, n_tensors, next_size) {
            if unsafe { ffi::gguf_get_n_tensors(curr) } == 0 {
                return Err(
                    "one of splits has 0 tensors; size or tensor limit is too small".to_string(),
                );
            }
            ctx_outs.push(GgufContext(curr));
            i_split += 1;
            curr = unsafe { ffi::gguf_init_empty() };
            unsafe {
                ffi::gguf_set_val_u16(curr, key(LLM_KV_SPLIT_NO).as_ptr(), i_split as u16);
                ffi::gguf_set_val_u16(curr, key(LLM_KV_SPLIT_COUNT).as_ptr(), 0);
                ffi::gguf_set_val_i32(
                    curr,
                    key(LLM_KV_SPLIT_TENSORS_COUNT).as_ptr(),
                    n_tensors as i32,
                );
            }
            current_size = pad(n_bytes, GGUF_DEFAULT_ALIGNMENT);
        } else {
            current_size = next_size;
        }
        unsafe {
            ffi::gguf_add_tensor(curr, tensor);
        }
    }

    ctx_outs.push(GgufContext(curr));
    let n_split = ctx_outs.len() as u16;
    for ctx in &ctx_outs {
        unsafe {
            ffi::gguf_set_val_u16(ctx.0, key(LLM_KV_SPLIT_COUNT).as_ptr(), n_split);
        }
    }
    Ok(ctx_outs)
}

fn print_split_info(
    ctx_outs: &[GgufContext],
    ctx_meta: *mut ffi::ggml_context,
) -> Result<(), String> {
    println!("n_split: {}", ctx_outs.len());
    for (i_split, ctx_out) in ctx_outs.iter().enumerate() {
        let mut total_size = unsafe { ffi::gguf_get_meta_size(ctx_out.0) };
        let n_tensors = unsafe { ffi::gguf_get_n_tensors(ctx_out.0) };
        for i in 0..n_tensors {
            let name = tensor_name(ctx_out.0, i)?;
            let tensor = tensor_by_name(ctx_meta, &name)?;
            total_size += unsafe { ffi::ggml_nbytes(tensor) };
        }
        println!(
            "split {:05}: n_tensors = {}, total_size = {}M",
            i_split + 1,
            n_tensors,
            total_size / 1_000_000
        );
    }
    Ok(())
}

fn write_splits(
    params: &SplitParams,
    loaded: &LoadedGguf,
    ctx_outs: &[GgufContext],
) -> Result<(), String> {
    let mut input = File::open(&params.input)
        .map_err(|err| format!("failed to open {}: {err}", params.input))?;
    let mut read_buf = Vec::new();
    let n_split = ctx_outs.len() as i32;
    for (i_split, ctx_out) in ctx_outs.iter().enumerate() {
        let path = split_path(&params.output, i_split as i32, n_split)?;
        print!("Writing file {path} ... ");
        std::io::stdout().flush().map_err(|err| err.to_string())?;
        let mut output =
            File::create(&path).map_err(|err| format!("failed to create {path}: {err}"))?;
        let meta_size = unsafe { ffi::gguf_get_meta_size(ctx_out.0) };
        let mut meta = vec![0_u8; meta_size];
        unsafe {
            ffi::gguf_get_meta_data(ctx_out.0, meta.as_mut_ptr().cast());
        }
        output.write_all(&meta).map_err(|err| err.to_string())?;

        let n_tensors = unsafe { ffi::gguf_get_n_tensors(ctx_out.0) };
        for i in 0..n_tensors {
            let name = tensor_name(ctx_out.0, i)?;
            let tensor = tensor_by_name(loaded.meta.0, &name)?;
            let n_bytes = unsafe { ffi::ggml_nbytes(tensor) };
            let tensor_id =
                unsafe { ffi::gguf_find_tensor(loaded.gguf.0, cstring(&name)?.as_ptr()) };
            let offset = unsafe {
                ffi::gguf_get_data_offset(loaded.gguf.0)
                    + ffi::gguf_get_tensor_offset(loaded.gguf.0, tensor_id)
            };
            copy_range(
                &mut input,
                &mut output,
                offset as u64,
                n_bytes,
                &mut read_buf,
            )?;
            write_zeros(&mut output, pad(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes)?;
        }
        println!("done");
    }
    Ok(())
}

fn gguf_split(params: &SplitParams) -> Result<(), String> {
    let loaded = load_gguf(&params.input)?;
    let ctx_outs = plan_splits(params, &loaded)?;
    print_split_info(&ctx_outs, loaded.meta.0)?;
    if !params.dry_run {
        write_splits(params, &loaded, &ctx_outs)?;
    }
    println!(
        "gguf_split: {} gguf split written with a total of {} tensors.",
        ctx_outs.len(),
        unsafe { ffi::gguf_get_n_tensors(loaded.gguf.0) }
    );
    Ok(())
}

fn gguf_merge(params: &SplitParams) -> Result<(), String> {
    eprintln!("gguf_merge: {} -> {}", params.input, params.output);
    if !params.dry_run && std::path::Path::new(&params.output).exists() {
        return Err(format!("output file {} already exists", params.output));
    }

    let ctx_out = GgufContext(unsafe { ffi::gguf_init_empty() });
    let mut loaded_parts = Vec::new();
    let mut n_split = 1_i32;
    let mut total_tensors = 0_i64;
    let mut split_path_current = params.input.clone();
    let mut prefix = String::new();

    for i_split in 0..n_split {
        if i_split > 0 {
            split_path_current = split_path(&prefix, i_split, n_split)?;
        }
        eprint!("gguf_merge: reading metadata {split_path_current} ...");
        let loaded = load_gguf(&split_path_current)?;
        if i_split == 0 {
            let split_count_key = key(LLM_KV_SPLIT_COUNT);
            let key_id = unsafe { ffi::gguf_find_key(loaded.gguf.0, split_count_key.as_ptr()) };
            if key_id < 0 {
                return Err(format!(
                    "input file does not contain {LLM_KV_SPLIT_COUNT} metadata"
                ));
            }
            n_split = unsafe { ffi::gguf_get_val_u16(loaded.gguf.0, key_id) } as i32;
            if n_split < 1 {
                return Err(format!(
                    "input file does not contain a valid split count {n_split}"
                ));
            }
            prefix = split_prefix(&split_path_current, i_split, n_split)?;
            unsafe {
                ffi::gguf_set_val_u16(loaded.gguf.0, split_count_key.as_ptr(), 0);
                ffi::gguf_set_kv(ctx_out.0, loaded.gguf.0);
            }
        }
        let n_tensors = unsafe { ffi::gguf_get_n_tensors(loaded.gguf.0) };
        for i in 0..n_tensors {
            let name = tensor_name(loaded.gguf.0, i)?;
            let tensor = tensor_by_name(loaded.meta.0, &name)?;
            unsafe {
                ffi::gguf_add_tensor(ctx_out.0, tensor);
            }
        }
        total_tensors += n_tensors;
        loaded_parts.push(loaded);
        eprintln!("done");
    }

    let mut output = if params.dry_run {
        None
    } else {
        let mut file = File::create(&params.output)
            .map_err(|err| format!("failed to create {}: {err}", params.output))?;
        write_zeros(&mut file, unsafe { ffi::gguf_get_meta_size(ctx_out.0) })?;
        Some(file)
    };

    let mut read_buf = Vec::new();
    for i_split in 0..n_split {
        let path = split_path(&prefix, i_split, n_split)?;
        eprint!("gguf_merge: writing tensors {path} ...");
        let mut input = File::open(&path).map_err(|err| format!("failed to open {path}: {err}"))?;
        let loaded = &loaded_parts[i_split as usize];
        let n_tensors = unsafe { ffi::gguf_get_n_tensors(loaded.gguf.0) };
        for i in 0..n_tensors {
            let name = tensor_name(loaded.gguf.0, i)?;
            let tensor = tensor_by_name(loaded.meta.0, &name)?;
            let n_bytes = unsafe { ffi::ggml_nbytes(tensor) };
            let offset = unsafe {
                ffi::gguf_get_data_offset(loaded.gguf.0)
                    + ffi::gguf_get_tensor_offset(loaded.gguf.0, i)
            };
            if let Some(output) = output.as_mut() {
                copy_range(&mut input, output, offset as u64, n_bytes, &mut read_buf)?;
                write_zeros(output, pad(n_bytes, GGUF_DEFAULT_ALIGNMENT) - n_bytes)?;
            }
        }
        eprintln!("done");
    }

    if let Some(mut output) = output {
        output
            .seek(SeekFrom::Start(0))
            .map_err(|err| err.to_string())?;
        let meta_size = unsafe { ffi::gguf_get_meta_size(ctx_out.0) };
        let mut meta = vec![0_u8; meta_size];
        unsafe {
            ffi::gguf_get_meta_data(ctx_out.0, meta.as_mut_ptr().cast());
        }
        output.write_all(&meta).map_err(|err| err.to_string())?;
    }

    eprintln!(
        "gguf_merge: {} merged from {} split with {} tensors.",
        params.output, n_split, total_tensors
    );
    Ok(())
}

fn print_usage(program: &str) {
    eprintln!();
    eprintln!("usage: {program} [options] GGUF_IN GGUF_OUT");
    eprintln!();
    eprintln!("Apply a GGUF operation on IN to OUT.");
    eprintln!();
    eprintln!("options:");
    eprintln!("  -h, --help              show this help message and exit");
    eprintln!("  --split                 split GGUF to multiple GGUF (enabled by default)");
    eprintln!("  --merge                 merge multiple GGUF to a single GGUF");
    eprintln!("  --split-max-tensors     max tensors in each split (default: 128)");
    eprintln!("  --split-max-size N(M|G) max size per split");
    eprintln!(
        "  --no-tensor-first-split do not add tensors to the first split (disabled by default)"
    );
    eprintln!("  --dry-run               only print out a split plan and exit, without writing any new files");
    eprintln!();
}

fn main() {
    let mut argv = env::args();
    let program = argv
        .next()
        .unwrap_or_else(|| "llama-gguf-split".to_string());
    let args = argv.collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage(&program);
        return;
    }
    if args.iter().any(|arg| arg == "--version") {
        eprintln!("version info unavailable in Rust port");
        return;
    }
    let params = match parse_args(args) {
        Ok(params) => params,
        Err(err) => {
            eprintln!("error: {err}");
            print_usage(&program);
            std::process::exit(1);
        }
    };
    let result = match params.operation {
        SplitOperation::Split => gguf_split(&params),
        SplitOperation::Merge => gguf_merge(&params),
    };
    if let Err(err) = result {
        eprintln!("{program}: error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_split_size_units() {
        assert_eq!(split_str_to_n_bytes("128M").unwrap(), 128_000_000);
        assert_eq!(split_str_to_n_bytes("4G").unwrap(), 4_000_000_000);
        assert!(split_str_to_n_bytes("0M").is_err());
        assert!(split_str_to_n_bytes("12K").is_err());
    }

    #[test]
    fn parses_default_split_args() {
        let params = parse_args(["--split-max-tensors", "3", "in.gguf", "out.gguf"]).unwrap();
        assert_eq!(params.operation, SplitOperation::Split);
        assert_eq!(params.mode, SplitMode::Tensor);
        assert_eq!(params.n_split_tensors, 3);
        assert_eq!(params.input, "in.gguf");
        assert_eq!(params.output, "out.gguf");
    }

    #[test]
    fn parses_merge_and_dry_run() {
        let params = parse_args(["--merge", "--dry-run", "part.gguf", "merged.gguf"]).unwrap();
        assert_eq!(params.operation, SplitOperation::Merge);
        assert!(params.dry_run);
    }

    #[test]
    fn rejects_conflicting_mode() {
        assert_eq!(
            parse_args([
                "--split-max-tensors",
                "2",
                "--split-max-size",
                "1G",
                "a",
                "b"
            ])
            .unwrap_err(),
            ParseError::ConflictingMode
        );
    }

    #[test]
    fn split_decision_matches_tensor_mode() {
        let params = SplitParams {
            n_split_tensors: 2,
            ..SplitParams::default()
        };
        assert!(!should_split(&params, 0, 5, 10));
        assert!(should_split(&params, 2, 5, 10));
        assert!(!should_split(&params, 5, 5, 10));
    }

    #[test]
    fn pads_to_alignment() {
        assert_eq!(pad(32, GGUF_DEFAULT_ALIGNMENT), 32);
        assert_eq!(pad(33, GGUF_DEFAULT_ALIGNMENT), 64);
    }
}
