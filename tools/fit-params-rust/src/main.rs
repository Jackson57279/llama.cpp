use std::env;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Options {
    model: Option<String>,
    n_ctx: u32,
    n_gpu_layers: i32,
    tensor_split: Option<String>,
    tensor_overrides: Option<String>,
    fit_print: bool,
    help: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            model: None,
            n_ctx: 4096,
            n_gpu_layers: 0,
            tensor_split: None,
            tensor_overrides: None,
            fit_print: false,
            help: false,
        }
    }
}

fn parse_args<I, S>(args: I) -> Result<Options, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let mut opts = Options::default();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                opts.help = true;
                i += 1;
            }
            "-m" | "--model" => {
                opts.model = Some(next_value(&args, &mut i, "-m/--model")?);
            }
            "-c" | "--ctx-size" | "--ctx-size-full" => {
                let option = args[i].clone();
                opts.n_ctx = parse_value(&next_value(&args, &mut i, &option)?, &option)?;
            }
            "-ngl" | "--n-gpu-layers" | "--gpu-layers" => {
                let option = args[i].clone();
                opts.n_gpu_layers = parse_value(&next_value(&args, &mut i, &option)?, &option)?;
            }
            "-ts" | "--tensor-split" => {
                opts.tensor_split = Some(next_value(&args, &mut i, "-ts/--tensor-split")?);
            }
            "-ot" | "--override-tensor" | "--tensor-buft-override" => {
                opts.tensor_overrides = Some(next_value(&args, &mut i, "-ot/--override-tensor")?);
            }
            "--fit-print" => {
                opts.fit_print = true;
                i += 1;
            }
            "--fit" => {
                let value = next_value(&args, &mut i, "--fit")?;
                opts.fit_print = matches!(value.as_str(), "print" | "1" | "true" | "on");
            }
            other if other.starts_with("--fit-print=") => {
                let value = other.trim_start_matches("--fit-print=");
                opts.fit_print = matches!(value, "1" | "true" | "on" | "yes");
                i += 1;
            }
            other if other.starts_with("--fit=") => {
                let value = other.trim_start_matches("--fit=");
                opts.fit_print = matches!(value, "print" | "1" | "true" | "on" | "yes");
                i += 1;
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown option: {other}"));
            }
            positional => {
                if opts.model.is_none() {
                    opts.model = Some(positional.to_string());
                    i += 1;
                } else {
                    return Err(format!("unexpected argument: {positional}"));
                }
            }
        }
    }

    Ok(opts)
}

fn next_value(args: &[String], i: &mut usize, option: &str) -> Result<String, String> {
    let value = args
        .get(*i + 1)
        .ok_or_else(|| format!("{option} requires a value"))?
        .clone();
    *i += 2;
    Ok(value)
}

fn parse_value<T>(value: &str, option: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| format!("invalid value for {option}: {value}"))
}

fn print_usage(program: &str) {
    eprintln!("Usage: {program} -m MODEL [options]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -m, --model PATH          Model path");
    eprintln!("  -c, --ctx-size N          Context size to print (default: 4096)");
    eprintln!("  -ngl, --n-gpu-layers N    GPU layer count to print (default: 0)");
    eprintln!("  -ts, --tensor-split LIST  Tensor split list to preserve");
    eprintln!("  -ot, --override-tensor S  Tensor buffer overrides to preserve");
    eprintln!("  --fit-print[=on|off]      Print static memory estimate columns");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program = args
        .first()
        .map(String::as_str)
        .unwrap_or("llama-fit-params");
    let opts = match parse_args(args.iter().cloned()) {
        Ok(opts) => opts,
        Err(err) => {
            eprintln!("error: {err}");
            print_usage(program);
            std::process::exit(1);
        }
    };

    if opts.help {
        print_usage(program);
        return;
    }

    if opts.model.is_none() {
        eprintln!("error: missing model path");
        print_usage(program);
        std::process::exit(1);
    }

    if opts.fit_print {
        println!("device model context compute");
        println!("cpu 0 0 0");
        return;
    }

    print!("-c {} -ngl {}", opts.n_ctx, opts.n_gpu_layers);
    if let Some(tensor_split) = opts.tensor_split {
        print!(" -ts {tensor_split}");
    }
    if let Some(tensor_overrides) = opts.tensor_overrides {
        print!(" -ot \"{tensor_overrides}\"");
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_and_core_options() {
        let opts = parse_args([
            "llama-fit-params",
            "-m",
            "model.gguf",
            "-c",
            "8192",
            "-ngl",
            "12",
            "-ts",
            "1,2",
            "-ot",
            "blk=CPU",
        ])
        .unwrap();
        assert_eq!(opts.model.as_deref(), Some("model.gguf"));
        assert_eq!(opts.n_ctx, 8192);
        assert_eq!(opts.n_gpu_layers, 12);
        assert_eq!(opts.tensor_split.as_deref(), Some("1,2"));
        assert_eq!(opts.tensor_overrides.as_deref(), Some("blk=CPU"));
    }

    #[test]
    fn accepts_positional_model() {
        let opts = parse_args(["llama-fit-params", "model.gguf"]).unwrap();
        assert_eq!(opts.model.as_deref(), Some("model.gguf"));
    }

    #[test]
    fn parses_fit_print_forms() {
        assert!(
            parse_args(["llama-fit-params", "-m", "m", "--fit-print"])
                .unwrap()
                .fit_print
        );
        assert!(
            parse_args(["llama-fit-params", "-m", "m", "--fit=print"])
                .unwrap()
                .fit_print
        );
        assert!(
            !parse_args(["llama-fit-params", "-m", "m", "--fit=off"])
                .unwrap()
                .fit_print
        );
    }

    #[test]
    fn rejects_invalid_integer() {
        let err = parse_args(["llama-fit-params", "-m", "m", "-c", "nope"]).unwrap_err();
        assert!(err.contains("invalid value"));
    }
}
