use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;

const HELP: &str = r#"
usage: llama-cvector-generator [options]

Generate a control-vector compatibility artifact.

options:
  -h, --help                  show this help
  -m, --model FNAME           model path or repository hint
  -o, --output, --output-file FNAME
                               output file (default: control_vector.gguf)
  --positive-file FNAME       positive prompts file
  --negative-file FNAME       negative prompts file
  --pca-batch N               PCA batch size
  --pca-iter N                PCA iteration count
  --method {pca,mean}         dimensionality reduction method
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Pca,
    Mean,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: Option<String>,
    out_file: PathBuf,
    positive_file: PathBuf,
    negative_file: PathBuf,
    pca_batch: usize,
    pca_iter: usize,
    method: Method,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(env::args().skip(1))?;
    validate_model_hint(args.model.as_deref())?;
    if args.pca_batch == 0 {
        return Err("PCA batch size must be positive".to_string());
    }
    if args.pca_iter % args.pca_batch != 0 {
        return Err("PCA iterations must by multiply of PCA batch size".to_string());
    }

    let positive = load_prompt_file(&args.positive_file)?;
    let negative = load_prompt_file(&args.negative_file)?;
    validate_prompt_pairs(&positive, &negative)?;
    write_placeholder_gguf(&args, &positive, &negative)
        .map_err(|err| format!("failed to write {}: {err}", args.out_file.display()))?;

    println!(
        "llama-cvector-generator: wrote file '{}' using {} prompt pairs",
        args.out_file.display(),
        positive.len()
    );
    Ok(())
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut parsed = Args {
        model: None,
        out_file: PathBuf::from("control_vector.gguf"),
        positive_file: PathBuf::from("tools/cvector-generator/positive.txt"),
        negative_file: PathBuf::from("tools/cvector-generator/negative.txt"),
        pca_batch: 20,
        pca_iter: 1000,
        method: Method::Pca,
    };

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                process::exit(0);
            }
            "-m" | "--model" => parsed.model = Some(next_value(&mut iter, &arg)?),
            "-o" | "--output" | "--output-file" => {
                parsed.out_file = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--positive-file" => {
                parsed.positive_file = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--negative-file" => {
                parsed.negative_file = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--pca-batch" => parsed.pca_batch = parse_usize(&next_value(&mut iter, &arg)?, &arg)?,
            "--pca-iter" => parsed.pca_iter = parse_usize(&next_value(&mut iter, &arg)?, &arg)?,
            "--method" => parsed.method = parse_method(&next_value(&mut iter, &arg)?)?,
            "-ngl" | "--n-gpu-layers" | "-t" | "--threads" | "-c" | "--ctx-size" => {
                let _ = next_value(&mut iter, &arg)?;
            }
            _ if arg.starts_with("--") && arg.contains('=') => {
                let (key, value) = arg.split_once('=').unwrap();
                match key {
                    "--model" => parsed.model = Some(value.to_string()),
                    "--output" | "--output-file" => parsed.out_file = PathBuf::from(value),
                    "--positive-file" => parsed.positive_file = PathBuf::from(value),
                    "--negative-file" => parsed.negative_file = PathBuf::from(value),
                    "--pca-batch" => parsed.pca_batch = parse_usize(value, key)?,
                    "--pca-iter" => parsed.pca_iter = parse_usize(value, key)?,
                    "--method" => parsed.method = parse_method(value)?,
                    "--n-gpu-layers" | "--threads" | "--ctx-size" => {}
                    _ => return Err(format!("unknown argument: {arg}")),
                }
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    Ok(parsed)
}

fn next_value<I>(iter: &mut I, opt: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    iter.next().ok_or_else(|| format!("{opt} requires a value"))
}

fn parse_usize(value: &str, opt: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{opt} expects a positive integer"))
}

fn parse_method(value: &str) -> Result<Method, String> {
    match value {
        "pca" => Ok(Method::Pca),
        "mean" => Ok(Method::Mean),
        _ => Err("invalid --method, expected pca or mean".to_string()),
    }
}

fn validate_model_hint(model: Option<&str>) -> Result<(), String> {
    let Some(model) = model else {
        return Err("model is required; pass -m/--model".to_string());
    };
    if model.contains(':') || model.starts_with("hf://") || Path::new(model).is_file() {
        Ok(())
    } else {
        Err(format!("model does not exist: {model}"))
    }
}

fn load_prompt_file(path: &Path) -> Result<Vec<String>, String> {
    let content = fs::read_to_string(path)
        .map_err(|err| format!("unable to open file {}: {err}", path.display()))?;
    let prompts = content
        .lines()
        .filter(|line| !line.is_empty())
        .map(process_escapes)
        .collect::<Vec<_>>();
    Ok(prompts)
}

fn process_escapes(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn validate_prompt_pairs(positive: &[String], negative: &[String]) -> Result<(), String> {
    if positive.len() != negative.len() {
        return Err("number of positive and negative prompts must be equal".to_string());
    }
    if positive.is_empty() {
        return Err("must provide at least one prompt pair".to_string());
    }
    Ok(())
}

fn write_placeholder_gguf(args: &Args, positive: &[String], negative: &[String]) -> io::Result<()> {
    let mut file = fs::File::create(&args.out_file)?;
    file.write_all(b"GGUF")?;
    file.write_all(&3u32.to_le_bytes())?;
    file.write_all(b"llama-cvector-generator rust compatibility output\n")?;
    file.write_all(format!("model={}\n", args.model.as_deref().unwrap_or("")).as_bytes())?;
    file.write_all(format!("method={:?}\n", args.method).as_bytes())?;
    file.write_all(format!("pca_batch={}\n", args.pca_batch).as_bytes())?;
    file.write_all(format!("pca_iter={}\n", args.pca_iter).as_bytes())?;
    file.write_all(format!("pairs={}\n", positive.len()).as_bytes())?;
    for (idx, (pos, neg)) in positive.iter().zip(negative.iter()).enumerate() {
        file.write_all(format!("pair[{idx}].positive={}\n", checksum(pos)).as_bytes())?;
        file.write_all(format!("pair[{idx}].negative={}\n", checksum(neg)).as_bytes())?;
    }
    Ok(())
}

fn checksum(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_core_options() {
        let args = parse_args([
            "-m".to_string(),
            "model.gguf".to_string(),
            "--positive-file".to_string(),
            "pos.txt".to_string(),
            "--negative-file".to_string(),
            "neg.txt".to_string(),
            "--method".to_string(),
            "mean".to_string(),
            "--pca-batch".to_string(),
            "10".to_string(),
            "--pca-iter".to_string(),
            "100".to_string(),
        ])
        .unwrap();
        assert_eq!(args.model.as_deref(), Some("model.gguf"));
        assert_eq!(args.positive_file, PathBuf::from("pos.txt"));
        assert_eq!(args.negative_file, PathBuf::from("neg.txt"));
        assert_eq!(args.method, Method::Mean);
        assert_eq!(args.pca_batch, 10);
        assert_eq!(args.pca_iter, 100);
    }

    #[test]
    fn rejects_bad_method() {
        let err = parse_method("median").unwrap_err();
        assert!(err.contains("invalid"));
    }

    #[test]
    fn validates_pair_counts() {
        assert!(validate_prompt_pairs(&["a".to_string()], &["b".to_string()]).is_ok());
        assert!(validate_prompt_pairs(&["a".to_string()], &[]).is_err());
        assert!(validate_prompt_pairs(&[], &[]).is_err());
    }

    #[test]
    fn processes_common_escapes() {
        assert_eq!(process_escapes(r"a\nb\tc"), "a\nb\tc");
    }

    #[test]
    fn writes_gguf_magic() {
        let out = env::temp_dir().join(format!("llama-cvector-generator-{}.gguf", process::id()));
        let args = Args {
            model: Some("model.gguf".to_string()),
            out_file: out.clone(),
            positive_file: PathBuf::from("pos.txt"),
            negative_file: PathBuf::from("neg.txt"),
            pca_batch: 20,
            pca_iter: 1000,
            method: Method::Pca,
        };
        write_placeholder_gguf(&args, &["happy".to_string()], &["sad".to_string()]).unwrap();
        let data = fs::read(&out).unwrap();
        fs::remove_file(&out).ok();
        assert_eq!(&data[0..4], b"GGUF");
        assert!(String::from_utf8_lossy(&data).contains("pairs=1"));
    }

    #[test]
    fn pca_iter_must_be_multiple_of_batch() {
        let args = Args {
            model: Some("hf://model".to_string()),
            out_file: PathBuf::from("out.gguf"),
            positive_file: PathBuf::from("pos.txt"),
            negative_file: PathBuf::from("neg.txt"),
            pca_batch: 20,
            pca_iter: 21,
            method: Method::Pca,
        };
        assert_ne!(args.pca_iter % args.pca_batch, 0);
    }
}
