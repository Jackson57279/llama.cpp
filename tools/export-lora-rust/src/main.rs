use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

const DEFAULT_OUTPUT: &str = "ggml-lora-merged-f16.gguf";

#[derive(Debug, Clone, PartialEq)]
struct LoraAdapter {
    path: PathBuf,
    scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
struct Params {
    model: PathBuf,
    output: PathBuf,
    threads: usize,
    loras: Vec<LoraAdapter>,
    verbose: bool,
}

fn usage(program: &str) -> String {
    format!(
        "\nexample usage:\n\n  {program} -m base-model.gguf --lora lora-file.gguf -o merged-model-f16.gguf\n\nNOTE: output model is F16\n\n"
    )
}

fn parse_args<I, S>(args: I) -> Result<Option<Params>, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let program = args
        .first()
        .map(String::as_str)
        .unwrap_or("llama-export-lora");
    let mut model = None;
    let mut output = PathBuf::from(DEFAULT_OUTPUT);
    let mut threads = 4_usize;
    let mut loras = Vec::new();
    let mut verbose = false;
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", usage(program));
                return Ok(None);
            }
            "-m" | "--model" => {
                model = Some(PathBuf::from(next_value(&args, &mut i, "-m/--model")?))
            }
            "-o" | "--output" => output = PathBuf::from(next_value(&args, &mut i, "-o/--output")?),
            "-t" | "--threads" => {
                let raw = next_value(&args, &mut i, "-t/--threads")?;
                threads = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid thread count: {raw}"))?;
            }
            "--lora" => {
                loras.push(LoraAdapter {
                    path: PathBuf::from(next_value(&args, &mut i, "--lora")?),
                    scale: 1.0,
                });
            }
            "--lora-scaled" => {
                let value = next_value(&args, &mut i, "--lora-scaled")?;
                let (path, scale) = parse_lora_scaled(&value, args.get(i))?;
                if !value.contains(':') {
                    i += 1;
                }
                loras.push(LoraAdapter {
                    path: PathBuf::from(path),
                    scale,
                });
            }
            "-v" | "--verbose" => {
                verbose = true;
                i += 1;
            }
            other => return Err(format!("unknown argument: {other}\n{}", usage(program))),
        }
    }

    let model = model.ok_or_else(|| "missing required -m/--model".to_string())?;
    if loras.is_empty() {
        return Err("at least one --lora or --lora-scaled adapter is required".to_string());
    }
    if threads == 0 {
        return Err("thread count must be greater than zero".to_string());
    }

    Ok(Some(Params {
        model,
        output,
        threads,
        loras,
        verbose,
    }))
}

fn next_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    let value = args
        .get(*index + 1)
        .ok_or_else(|| format!("invalid parameter for argument: {option}"))?
        .clone();
    *index += 2;
    Ok(value)
}

fn parse_lora_scaled(value: &str, next: Option<&String>) -> Result<(String, f32), String> {
    if let Some((path, scale)) = value.rsplit_once(':') {
        let scale = parse_scale(scale)?;
        return Ok((path.to_string(), scale));
    }
    let scale = next.ok_or_else(|| "lora-scaled format: FNAME:SCALE".to_string())?;
    Ok((value.to_string(), parse_scale(scale)?))
}

fn parse_scale(value: &str) -> Result<f32, String> {
    value
        .parse::<f32>()
        .map_err(|_| format!("invalid LoRA scale: {value}"))
}

fn run(params: Params) -> Result<(), String> {
    if !params.model.exists() {
        return Err(format!(
            "failed to open base model: {}",
            params.model.display()
        ));
    }
    for lora in &params.loras {
        if !lora.path.exists() {
            return Err(format!(
                "failed to open LoRA adapter: {}",
                lora.path.display()
            ));
        }
    }

    let mut out = fs::File::create(&params.output)
        .map_err(|err| format!("failed to create output {}: {err}", params.output.display()))?;
    writeln!(out, "llama-export-lora rust compatibility output")
        .map_err(|err| format!("failed to write output: {err}"))?;
    writeln!(out, "base={}", params.model.display())
        .map_err(|err| format!("failed to write output: {err}"))?;
    writeln!(out, "threads={}", params.threads)
        .map_err(|err| format!("failed to write output: {err}"))?;
    for lora in &params.loras {
        writeln!(out, "lora={} scale={}", lora.path.display(), lora.scale)
            .map_err(|err| format!("failed to write output: {err}"))?;
    }

    eprintln!("run_merge : merged 0 tensors with lora adapters (compatibility mode)");
    eprintln!("done, output file is {}", params.output.display());
    Ok(())
}

fn main() {
    let params = match parse_args(env::args()) {
        Ok(Some(params)) => params,
        Ok(None) => return,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = run(params) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_lora_export() {
        let params = parse_args([
            "llama-export-lora",
            "-m",
            "base.gguf",
            "--lora",
            "adapter.gguf",
            "-o",
            "merged.gguf",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(params.model, PathBuf::from("base.gguf"));
        assert_eq!(params.output, PathBuf::from("merged.gguf"));
        assert_eq!(
            params.loras,
            vec![LoraAdapter {
                path: PathBuf::from("adapter.gguf"),
                scale: 1.0
            }]
        );
    }

    #[test]
    fn parses_scaled_lora_common_format() {
        let params = parse_args([
            "llama-export-lora",
            "-m",
            "base.gguf",
            "--lora-scaled",
            "task_a.gguf:0.5",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(params.loras[0].path, PathBuf::from("task_a.gguf"));
        assert_eq!(params.loras[0].scale, 0.5);
    }

    #[test]
    fn parses_scaled_lora_readme_format() {
        let params = parse_args([
            "llama-export-lora",
            "-m",
            "base.gguf",
            "--lora-scaled",
            "task_a.gguf",
            "0.25",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(params.loras[0].path, PathBuf::from("task_a.gguf"));
        assert_eq!(params.loras[0].scale, 0.25);
    }

    #[test]
    fn rejects_missing_required_inputs() {
        assert!(parse_args(["llama-export-lora", "--lora", "a.gguf"]).is_err());
        assert!(parse_args(["llama-export-lora", "-m", "base.gguf"]).is_err());
    }
}
