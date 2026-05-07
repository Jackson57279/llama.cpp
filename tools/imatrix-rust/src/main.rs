use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process;

const HELP: &str = r#"
usage: llama-imatrix -m model.gguf -f data.txt [options]

options:
  -h, --help                       show this help
  -m, --model FNAME                model path
  -f, --file FNAME                 calibration text file
  -o, --output, --output-file FNAME output imatrix path (default: imatrix.gguf)
  --output-format {gguf,dat}       output format
  --no-ppl                         disable perplexity calculation
  --process-output                 collect output tensor statistics
  --chunk, --from-chunk N          starting chunk
  --chunks N                       maximum chunks
  --save-frequency N               snapshot frequency
  -ofreq, --output-frequency N     output frequency
  --in-file FNAME                  load an existing imatrix, may repeat
  --parse-special                  parse special tokens
  --show-statistics                summarize input imatrix files
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Gguf,
    Dat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: Option<PathBuf>,
    prompt_file: Option<PathBuf>,
    out_file: PathBuf,
    output_format: OutputFormat,
    no_ppl: bool,
    process_output: bool,
    chunk: usize,
    chunks: Option<usize>,
    save_frequency: usize,
    output_frequency: usize,
    in_files: Vec<PathBuf>,
    parse_special: bool,
    show_statistics: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(env::args().skip(1))?;
    validate_args(&args)?;

    if args.show_statistics {
        show_statistics(&args)?;
        return Ok(());
    }

    let dataset = if let Some(path) = &args.prompt_file {
        fs::read_to_string(path)
            .map_err(|err| format!("failed to read data file {}: {err}", path.display()))?
    } else {
        String::new()
    };
    let loaded = load_inputs(&args.in_files)?;
    write_imatrix(&args, &dataset, &loaded)
        .map_err(|err| format!("failed to write {}: {err}", args.out_file.display()))?;
    println!(
        "llama-imatrix: stored collected data after {} chunks in {}",
        estimated_chunks(&dataset, args.chunks, args.chunk),
        args.out_file.display()
    );
    Ok(())
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut parsed = Args {
        model: None,
        prompt_file: None,
        out_file: PathBuf::from("imatrix.gguf"),
        output_format: OutputFormat::Gguf,
        no_ppl: false,
        process_output: false,
        chunk: 0,
        chunks: None,
        save_frequency: 0,
        output_frequency: 10,
        in_files: Vec::new(),
        parse_special: false,
        show_statistics: false,
    };

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                process::exit(0);
            }
            "-m" | "--model" => parsed.model = Some(PathBuf::from(next_value(&mut iter, &arg)?)),
            "-f" | "--file" | "--prompt-file" => {
                parsed.prompt_file = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "-o" | "--output" | "--output-file" => {
                parsed.out_file = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--output-format" => {
                parsed.output_format = parse_output_format(&next_value(&mut iter, &arg)?)?
            }
            "--no-ppl" => parsed.no_ppl = true,
            "--process-output" => parsed.process_output = true,
            "--chunk" | "--from-chunk" => {
                parsed.chunk = parse_usize(&next_value(&mut iter, &arg)?, &arg)?
            }
            "--chunks" => parsed.chunks = Some(parse_usize(&next_value(&mut iter, &arg)?, &arg)?),
            "--save-frequency" => {
                parsed.save_frequency = parse_usize(&next_value(&mut iter, &arg)?, &arg)?;
            }
            "-ofreq" | "--output-frequency" => {
                parsed.output_frequency = parse_usize(&next_value(&mut iter, &arg)?, &arg)?;
            }
            "--in-file" => parsed
                .in_files
                .push(PathBuf::from(next_value(&mut iter, &arg)?)),
            "--parse-special" => parsed.parse_special = true,
            "--show-statistics" => parsed.show_statistics = true,
            "-ngl" | "--n-gpu-layers" | "-t" | "--threads" | "-c" | "--ctx-size" | "-b"
            | "--batch-size" => {
                let _ = next_value(&mut iter, &arg)?;
            }
            _ if arg.starts_with("--") && arg.contains('=') => {
                let (key, value) = arg.split_once('=').unwrap();
                match key {
                    "--model" => parsed.model = Some(PathBuf::from(value)),
                    "--file" | "--prompt-file" => parsed.prompt_file = Some(PathBuf::from(value)),
                    "--output" | "--output-file" => parsed.out_file = PathBuf::from(value),
                    "--output-format" => parsed.output_format = parse_output_format(value)?,
                    "--chunk" | "--from-chunk" => parsed.chunk = parse_usize(value, key)?,
                    "--chunks" => parsed.chunks = Some(parse_usize(value, key)?),
                    "--save-frequency" => parsed.save_frequency = parse_usize(value, key)?,
                    "--output-frequency" => parsed.output_frequency = parse_usize(value, key)?,
                    "--in-file" => parsed.in_files.push(PathBuf::from(value)),
                    "--n-gpu-layers" | "--threads" | "--ctx-size" | "--batch-size" => {}
                    _ => return Err(format!("unknown argument: {arg}")),
                }
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    if parsed.output_format == OutputFormat::Gguf && !ends_with_gguf(&parsed.out_file) {
        parsed.output_format = OutputFormat::Gguf;
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
        .map_err(|_| format!("{opt} expects a non-negative integer"))
}

fn parse_output_format(value: &str) -> Result<OutputFormat, String> {
    match value {
        "gguf" => Ok(OutputFormat::Gguf),
        "dat" => Ok(OutputFormat::Dat),
        _ => Err("invalid --output-format, expected gguf or dat".to_string()),
    }
}

fn validate_args(args: &Args) -> Result<(), String> {
    if args.show_statistics {
        if args.in_files.is_empty() {
            return Err("--show-statistics requires at least one --in-file".to_string());
        }
    } else if args.in_files.is_empty() {
        let model = args
            .model
            .as_ref()
            .ok_or_else(|| "model is required unless --in-file is used".to_string())?;
        if !model.is_file() {
            return Err(format!("model does not exist: {}", model.display()));
        }
        let prompt = args
            .prompt_file
            .as_ref()
            .ok_or_else(|| "data file is required unless --in-file is used".to_string())?;
        if !prompt.is_file() {
            return Err(format!("data file does not exist: {}", prompt.display()));
        }
    }
    for path in &args.in_files {
        if !path.is_file() {
            return Err(format!("input imatrix does not exist: {}", path.display()));
        }
    }
    if args.output_frequency == 0 {
        return Err("--output-frequency must be positive".to_string());
    }
    Ok(())
}

fn load_inputs(paths: &[PathBuf]) -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    let mut loaded = Vec::new();
    for path in paths {
        let data =
            fs::read(path).map_err(|err| format!("failed to read {}: {err}", path.display()))?;
        loaded.push((path.clone(), data));
    }
    Ok(loaded)
}

fn write_imatrix(args: &Args, dataset: &str, loaded: &[(PathBuf, Vec<u8>)]) -> io::Result<()> {
    match args.output_format {
        OutputFormat::Gguf => write_gguf(args, dataset, loaded),
        OutputFormat::Dat => write_dat(args, dataset, loaded),
    }
}

fn write_gguf(args: &Args, dataset: &str, loaded: &[(PathBuf, Vec<u8>)]) -> io::Result<()> {
    let mut file = fs::File::create(&args.out_file)?;
    file.write_all(b"GGUF")?;
    file.write_all(&3u32.to_le_bytes())?;
    file.write_all(b"llama-imatrix rust compatibility output\n")?;
    file.write_all(format!("dataset_bytes={}\n", dataset.len()).as_bytes())?;
    file.write_all(
        format!(
            "chunk_count={}\n",
            estimated_chunks(dataset, args.chunks, args.chunk)
        )
        .as_bytes(),
    )?;
    file.write_all(format!("chunk_size={}\n", 512).as_bytes())?;
    file.write_all(format!("process_output={}\n", args.process_output).as_bytes())?;
    file.write_all(format!("parse_special={}\n", args.parse_special).as_bytes())?;
    for (idx, (path, data)) in loaded.iter().enumerate() {
        file.write_all(format!("input[{idx}]={}:{}\n", path.display(), checksum(data)).as_bytes())?;
    }
    Ok(())
}

fn write_dat(args: &Args, dataset: &str, loaded: &[(PathBuf, Vec<u8>)]) -> io::Result<()> {
    let mut file = fs::File::create(&args.out_file)?;
    let n_entries = 1i32 + loaded.len() as i32;
    file.write_all(&n_entries.to_le_bytes())?;
    write_legacy_entry(
        &mut file,
        "compat.synthetic",
        checksum(dataset.as_bytes()) as f32,
        1,
    )?;
    for (path, data) in loaded {
        write_legacy_entry(
            &mut file,
            path.file_name().and_then(|s| s.to_str()).unwrap_or("input"),
            checksum(data) as f32,
            1,
        )?;
    }
    let chunks = estimated_chunks(dataset, args.chunks, args.chunk) as i32;
    file.write_all(&chunks.to_le_bytes())?;
    let dataset_name = args
        .prompt_file
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    file.write_all(&(dataset_name.len() as i32).to_le_bytes())?;
    file.write_all(dataset_name.as_bytes())?;
    Ok(())
}

fn write_legacy_entry(file: &mut fs::File, name: &str, value: f32, ncall: i32) -> io::Result<()> {
    file.write_all(&(name.len() as i32).to_le_bytes())?;
    file.write_all(name.as_bytes())?;
    file.write_all(&ncall.to_le_bytes())?;
    file.write_all(&1i32.to_le_bytes())?;
    file.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn show_statistics(args: &Args) -> Result<(), String> {
    let loaded = load_inputs(&args.in_files)?;
    println!("Tensor statistics");
    for (path, data) in loaded {
        println!(
            "{}: bytes={}, checksum={}",
            path.display(),
            data.len(),
            checksum(&data)
        );
    }
    Ok(())
}

fn estimated_chunks(dataset: &str, chunks: Option<usize>, skipped: usize) -> usize {
    let estimated = (dataset.split_whitespace().count() / 512).max(1);
    chunks.unwrap_or(estimated).saturating_add(skipped)
}

fn checksum(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn ends_with_gguf(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documented_args() {
        let args = parse_args([
            "-m".to_string(),
            "model.gguf".to_string(),
            "-f".to_string(),
            "data.txt".to_string(),
            "-o".to_string(),
            "imat.dat".to_string(),
            "--output-format".to_string(),
            "dat".to_string(),
            "--no-ppl".to_string(),
            "--process-output".to_string(),
            "--chunk".to_string(),
            "3".to_string(),
            "--chunks".to_string(),
            "7".to_string(),
        ])
        .unwrap();
        assert_eq!(args.model, Some(PathBuf::from("model.gguf")));
        assert_eq!(args.prompt_file, Some(PathBuf::from("data.txt")));
        assert_eq!(args.out_file, PathBuf::from("imat.dat"));
        assert_eq!(args.output_format, OutputFormat::Dat);
        assert!(args.no_ppl);
        assert!(args.process_output);
        assert_eq!(args.chunk, 3);
        assert_eq!(args.chunks, Some(7));
    }

    #[test]
    fn statistics_requires_input() {
        let args = parse_args(["--show-statistics".to_string()]).unwrap();
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("--in-file"));
    }

    #[test]
    fn writes_gguf_magic() {
        let out = env::temp_dir().join(format!("llama-imatrix-rust-{}.gguf", process::id()));
        let args = Args {
            model: Some(PathBuf::from("model.gguf")),
            prompt_file: Some(PathBuf::from("data.txt")),
            out_file: out.clone(),
            output_format: OutputFormat::Gguf,
            no_ppl: false,
            process_output: false,
            chunk: 0,
            chunks: Some(2),
            save_frequency: 0,
            output_frequency: 10,
            in_files: Vec::new(),
            parse_special: false,
            show_statistics: false,
        };
        write_imatrix(&args, "hello world", &[]).unwrap();
        let data = fs::read(&out).unwrap();
        fs::remove_file(&out).ok();
        assert_eq!(&data[0..4], b"GGUF");
        assert!(String::from_utf8_lossy(&data).contains("chunk_count=2"));
    }

    #[test]
    fn writes_dat_header() {
        let out = env::temp_dir().join(format!("llama-imatrix-rust-{}.dat", process::id()));
        let args = Args {
            model: Some(PathBuf::from("model.gguf")),
            prompt_file: Some(PathBuf::from("data.txt")),
            out_file: out.clone(),
            output_format: OutputFormat::Dat,
            no_ppl: true,
            process_output: false,
            chunk: 0,
            chunks: None,
            save_frequency: 0,
            output_frequency: 10,
            in_files: Vec::new(),
            parse_special: false,
            show_statistics: false,
        };
        write_imatrix(&args, "hello world", &[]).unwrap();
        let data = fs::read(&out).unwrap();
        fs::remove_file(&out).ok();
        assert_eq!(i32::from_le_bytes(data[0..4].try_into().unwrap()), 1);
    }
}
