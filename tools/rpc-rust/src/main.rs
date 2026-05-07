use std::env;
use std::ffi::{CStr, CString};
use std::path::PathBuf;

const GGML_BACKEND_DEVICE_TYPE_CPU: i32 = 0;

mod ffi {
    #![allow(non_camel_case_types)]

    use std::ffi::{c_char, c_int, c_void};

    pub type ggml_backend_dev_t = *mut c_void;
    pub type ggml_backend_reg_t = *mut c_void;

    extern "C" {
        pub fn ggml_backend_load_all();
        pub fn ggml_backend_dev_count() -> usize;
        pub fn ggml_backend_dev_get(index: usize) -> ggml_backend_dev_t;
        pub fn ggml_backend_dev_type(dev: ggml_backend_dev_t) -> c_int;
        pub fn ggml_backend_dev_by_name(name: *const c_char) -> ggml_backend_dev_t;
        pub fn ggml_backend_dev_by_type(typ: c_int) -> ggml_backend_dev_t;
        pub fn ggml_backend_dev_name(dev: ggml_backend_dev_t) -> *const c_char;
        pub fn ggml_backend_dev_description(dev: ggml_backend_dev_t) -> *const c_char;
        pub fn ggml_backend_dev_memory(
            dev: ggml_backend_dev_t,
            free: *mut usize,
            total: *mut usize,
        );
        pub fn ggml_backend_reg_by_name(name: *const c_char) -> ggml_backend_reg_t;
        pub fn ggml_backend_reg_get_proc_address(
            reg: ggml_backend_reg_t,
            name: *const c_char,
        ) -> *mut c_void;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RpcServerParams {
    pub host: String,
    pub port: u16,
    pub use_cache: bool,
    pub n_threads: usize,
    pub devices: Vec<String>,
}

impl Default for RpcServerParams {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 50052,
            use_cache: false,
            n_threads: std::thread::available_parallelism()
                .map(|n| (n.get() / 2).max(1))
                .unwrap_or(1),
            devices: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    MissingValue(String),
    InvalidInteger(String, String),
    InvalidPort(i32),
    InvalidThreads(isize),
    UnknownArgument(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingValue(flag) => write!(f, "missing value for {flag}"),
            Self::InvalidInteger(flag, value) => write!(f, "invalid integer for {flag}: {value}"),
            Self::InvalidPort(value) => write!(f, "invalid port: {value}"),
            Self::InvalidThreads(value) => write!(f, "invalid number of threads: {value}"),
            Self::UnknownArgument(arg) => write!(f, "unknown argument: {arg}"),
        }
    }
}

impl std::error::Error for ParseError {}

pub fn parse_args<I, S>(args: I) -> Result<RpcServerParams, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut params = RpcServerParams::default();
    let mut iter = args.into_iter().map(Into::into);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-H" | "--host" => params.host = take(&mut iter, &arg)?,
            "-t" | "--threads" => {
                let value = parse_isize(&mut iter, &arg)?;
                if value <= 0 {
                    return Err(ParseError::InvalidThreads(value));
                }
                params.n_threads = value as usize;
            }
            "-d" | "--device" => {
                let value = take(&mut iter, &arg)?;
                params.devices.extend(
                    value
                        .split(|ch| ch == ',' || ch == '/')
                        .filter(|part| !part.is_empty())
                        .map(ToOwned::to_owned),
                );
            }
            "-p" | "--port" => {
                let value = parse_i32(&mut iter, &arg)?;
                if !(1..=65535).contains(&value) {
                    return Err(ParseError::InvalidPort(value));
                }
                params.port = value as u16;
            }
            "-c" | "--cache" => params.use_cache = true,
            "-h" | "--help" => return Err(ParseError::UnknownArgument(arg)),
            other => return Err(ParseError::UnknownArgument(other.to_string())),
        }
    }

    Ok(params)
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

fn parse_isize<I>(iter: &mut I, flag: &str) -> Result<isize, ParseError>
where
    I: Iterator<Item = String>,
{
    let value = take(iter, flag)?;
    value
        .parse()
        .map_err(|_| ParseError::InvalidInteger(flag.to_string(), value))
}

fn cstr(ptr: *const std::ffi::c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

pub fn cache_directory(
    env_llama_cache: Option<&str>,
    env_xdg: Option<&str>,
    env_home: Option<&str>,
) -> Option<PathBuf> {
    let base = if let Some(path) = env_llama_cache {
        PathBuf::from(path)
    } else if cfg!(target_os = "macos") {
        PathBuf::from(env_home?).join("Library/Caches/llama.cpp")
    } else if cfg!(target_os = "windows") {
        PathBuf::from(env::var("LOCALAPPDATA").ok()?).join("llama.cpp")
    } else if let Some(path) = env_xdg {
        PathBuf::from(path).join("llama.cpp")
    } else {
        PathBuf::from(env_home?).join(".cache/llama.cpp")
    };
    Some(base.join("rpc"))
}

fn runtime_cache_directory() -> Result<PathBuf, String> {
    cache_directory(
        env::var("LLAMA_CACHE").ok().as_deref(),
        env::var("XDG_CACHE_HOME").ok().as_deref(),
        env::var("HOME").ok().as_deref(),
    )
    .ok_or_else(|| "failed to find cache directory".to_string())
}

fn available_devices() {
    eprintln!("available devices:");
    for i in 0..unsafe { ffi::ggml_backend_dev_count() } {
        let dev = unsafe { ffi::ggml_backend_dev_get(i) };
        let mut free = 0_usize;
        let mut total = 0_usize;
        unsafe {
            ffi::ggml_backend_dev_memory(dev, &mut free, &mut total);
        }
        println!(
            "  {}: {} ({} MiB, {} MiB free)",
            cstr(unsafe { ffi::ggml_backend_dev_name(dev) }),
            cstr(unsafe { ffi::ggml_backend_dev_description(dev) }),
            total / 1024 / 1024,
            free / 1024 / 1024
        );
    }
}

fn get_devices(params: &RpcServerParams) -> Result<Vec<ffi::ggml_backend_dev_t>, String> {
    let mut devices = Vec::new();
    for device in &params.devices {
        let device_c =
            CString::new(device.as_str()).map_err(|_| "device name contains NUL byte")?;
        let dev = unsafe { ffi::ggml_backend_dev_by_name(device_c.as_ptr()) };
        if dev.is_null() {
            eprintln!("error: unknown device: {device}");
            available_devices();
            return Err(format!("unknown device: {device}"));
        }
        devices.push(dev);
    }

    if devices.is_empty() {
        for i in 0..unsafe { ffi::ggml_backend_dev_count() } {
            let dev = unsafe { ffi::ggml_backend_dev_get(i) };
            if unsafe { ffi::ggml_backend_dev_type(dev) } != GGML_BACKEND_DEVICE_TYPE_CPU {
                devices.push(dev);
            }
        }
    }

    if devices.is_empty() {
        let dev = unsafe { ffi::ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) };
        if !dev.is_null() {
            devices.push(dev);
        }
    }

    Ok(devices)
}

type StartServerFn = unsafe extern "C" fn(
    endpoint: *const std::ffi::c_char,
    cache_dir: *const std::ffi::c_char,
    n_threads: usize,
    n_devices: usize,
    devices: *mut ffi::ggml_backend_dev_t,
);

fn run(params: RpcServerParams) -> Result<(), String> {
    unsafe {
        ffi::ggml_backend_load_all();
    }

    if params.host != "127.0.0.1" {
        eprintln!();
        eprintln!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        eprintln!("WARNING: Host ('{}') is != '127.0.0.1'", params.host);
        eprintln!("         Never expose the RPC server to an open network!");
        eprintln!("         This is an experimental feature and is not secure!");
        eprintln!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        eprintln!();
    }

    let mut devices = get_devices(&params)?;
    if devices.is_empty() {
        return Err("No devices found".to_string());
    }

    let endpoint = CString::new(format!("{}:{}", params.host, params.port))
        .map_err(|_| "endpoint contains NUL byte".to_string())?;
    let cache_dir = if params.use_cache {
        let dir = runtime_cache_directory()?;
        std::fs::create_dir_all(&dir)
            .map_err(|err| format!("failed to create cache directory {}: {err}", dir.display()))?;
        Some(
            CString::new(dir.to_string_lossy().as_bytes())
                .map_err(|_| "cache directory contains NUL byte")?,
        )
    } else {
        None
    };

    let rpc = CString::new("RPC").unwrap();
    let reg = unsafe { ffi::ggml_backend_reg_by_name(rpc.as_ptr()) };
    if reg.is_null() {
        return Err("Failed to find RPC backend".to_string());
    }
    let symbol = CString::new("ggml_backend_rpc_start_server").unwrap();
    let proc = unsafe { ffi::ggml_backend_reg_get_proc_address(reg, symbol.as_ptr()) };
    if proc.is_null() {
        return Err("Failed to obtain RPC backend start server function".to_string());
    }
    let start_server: StartServerFn = unsafe { std::mem::transmute(proc) };
    unsafe {
        start_server(
            endpoint.as_ptr(),
            cache_dir.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            params.n_threads,
            devices.len(),
            devices.as_mut_ptr(),
        );
    }
    Ok(())
}

fn print_usage(program: &str) {
    let defaults = RpcServerParams::default();
    eprintln!("Usage: {program} [options]\n");
    eprintln!("options:");
    eprintln!("  -h, --help                       show this help message and exit");
    eprintln!(
        "  -t, --threads N                  number of threads for the CPU device (default: {})",
        defaults.n_threads
    );
    eprintln!("  -d, --device <dev1,dev2,...>     comma-separated list of devices");
    eprintln!(
        "  -H, --host HOST                  host to bind to (default: {})",
        defaults.host
    );
    eprintln!(
        "  -p, --port PORT                  port to bind to (default: {})",
        defaults.port
    );
    eprintln!("  -c, --cache                      enable local file cache");
    eprintln!();
}

fn main() {
    let mut argv = env::args();
    let program = argv.next().unwrap_or_else(|| "rpc-server".to_string());
    let args = argv.collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_usage(&program);
        return;
    }
    let params = match parse_args(args) {
        Ok(params) => params,
        Err(err) => {
            eprintln!("{program}: error: {err}");
            print_usage(&program);
            std::process::exit(1);
        }
    };
    if let Err(err) = run(params) {
        eprintln!("{program}: error: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rpc_options() {
        let params = parse_args([
            "-H",
            "0.0.0.0",
            "-p",
            "6000",
            "-t",
            "3",
            "-d",
            "CUDA0/CPU",
            "-c",
        ])
        .unwrap();
        assert_eq!(params.host, "0.0.0.0");
        assert_eq!(params.port, 6000);
        assert_eq!(params.n_threads, 3);
        assert_eq!(params.devices, vec!["CUDA0", "CPU"]);
        assert!(params.use_cache);
    }

    #[test]
    fn rejects_invalid_port() {
        assert_eq!(
            parse_args(["--port", "70000"]).unwrap_err(),
            ParseError::InvalidPort(70000)
        );
    }

    #[test]
    fn rejects_invalid_threads() {
        assert_eq!(
            parse_args(["--threads", "0"]).unwrap_err(),
            ParseError::InvalidThreads(0)
        );
    }

    #[test]
    fn builds_default_linux_cache_directory() {
        let dir = cache_directory(None, None, Some("/home/me")).unwrap();
        assert_eq!(dir, PathBuf::from("/home/me/.cache/llama.cpp/rpc"));
    }

    #[test]
    fn llama_cache_takes_precedence() {
        let dir =
            cache_directory(Some("/tmp/llama-cache"), Some("/xdg"), Some("/home/me")).unwrap();
        assert_eq!(dir, PathBuf::from("/tmp/llama-cache/rpc"));
    }
}
