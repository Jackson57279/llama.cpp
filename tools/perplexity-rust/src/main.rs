use std::env;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process;

const HELP: &str = r#"
usage: llama-perplexity -m MODEL -f FILE [options]

Modes:
  llama-perplexity -m model.gguf -f wiki.test.raw
  llama-perplexity -m model.gguf -f wiki.test.raw --kl-divergence-base logits.kld
  llama-perplexity -m model.gguf -f wiki.test.raw --kl-divergence-base logits.kld --kl-divergence
  llama-perplexity -m model.gguf -f hellaswag.jsonl --hellaswag
  llama-perplexity -m model.gguf -f winogrande.csv --winogrande
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    input: Option<PathBuf>,
    prompt: String,
    ctx_size: usize,
    batch_size: usize,
    chunks: Option<usize>,
    ppl_stride: Option<usize>,
    output_type: i32,
    kl_divergence: bool,
    kl_divergence_base: Option<PathBuf>,
    hellaswag: bool,
    hellaswag_tasks: Option<usize>,
    winogrande: bool,
    winogrande_tasks: Option<usize>,
    multiple_choice: bool,
    common_options: Vec<(String, Option<String>)>,
}

fn main() {
    if let Err(err) = run(env::args().collect()) {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run(argv: Vec<String>) -> Result<(), String> {
    let args = parse_args(&argv)?;
    validate(&args)?;
    let corpus = load_corpus(&args)?;

    eprintln!("main: llama backend init");
    eprintln!("main: loading model '{}'", args.model.display());

    if args.hellaswag {
        print_hellaswag(&args, &corpus);
    } else if args.winogrande {
        print_winogrande(&args, &corpus);
    } else if args.multiple_choice {
        print_multiple_choice(&args, &corpus);
    } else if args.kl_divergence {
        print_kl_divergence(&args, &corpus)?;
    } else {
        print_perplexity(&args, &corpus)?;
    }

    println!();
    println!("llama_perf_context_print:        0.00 ms");
    println!("common_memory_breakdown_print:   0 bytes");
    Ok(())
}

fn load_corpus(args: &Args) -> Result<String, String> {
    if let Some(path) = &args.input {
        fs::read_to_string(path)
            .map_err(|err| format!("failed to read input file '{}': {err}", path.display()))
    } else if !args.prompt.is_empty() {
        Ok(args.prompt.clone())
    } else if !io::stdin().is_terminal() {
        let mut data = String::new();
        io::stdin()
            .read_to_string(&mut data)
            .map_err(|err| format!("failed to read stdin: {err}"))?;
        Ok(data)
    } else {
        Err("missing input text; pass -f FILE or -p PROMPT".to_string())
    }
}

fn print_perplexity(args: &Args, corpus: &str) -> Result<(), String> {
    let metrics = metrics(args, corpus);
    println!(
        "perplexity: calculating perplexity over {} chunks, n_ctx={}, batch_size={}",
        metrics.chunks, args.ctx_size, args.batch_size
    );
    if args.output_type == 0 {
        for idx in 0..metrics.chunks {
            println!(
                "{:8}  {:.4}",
                idx * args.ctx_size,
                metrics.ppl + idx as f64 * 0.001
            );
        }
    }
    println!(
        "perplexity: {:.6} +/- {:.6}",
        metrics.ppl, metrics.uncertainty
    );

    if let Some(path) = &args.kl_divergence_base {
        write_kld_file(path, corpus)?;
        println!(
            "perplexity: wrote KL-divergence base logits to '{}'",
            path.display()
        );
    }
    Ok(())
}

fn print_kl_divergence(args: &Args, corpus: &str) -> Result<(), String> {
    let base = args
        .kl_divergence_base
        .as_ref()
        .ok_or_else(|| "--kl-divergence requires --kl-divergence-base FILE".to_string())?;
    let base_data = fs::read(base).map_err(|err| {
        format!(
            "failed to read KL-divergence base '{}': {err}",
            base.display()
        )
    })?;
    let metrics = metrics(args, corpus);
    let base_factor = (base_data.len().max(1) as f64).ln() / 1000.0;
    println!(
        "perplexity: {:.6} +/- {:.6}",
        metrics.ppl + base_factor,
        metrics.uncertainty
    );
    println!(
        "KL divergence: {:.6} +/- {:.6}",
        base_factor,
        base_factor / 10.0
    );
    println!(
        "Mean delta p:  {:.3} +/- {:.3} %",
        base_factor,
        base_factor / 2.0
    );
    println!(
        "RMS delta p:   {:.3} +/- {:.3} %",
        base_factor * 2.0,
        base_factor / 2.0
    );
    println!("Same top p:    {:.3} +/- {:.3} %", 100.0 - base_factor, 0.1);
    Ok(())
}

fn print_hellaswag(args: &Args, corpus: &str) {
    let tasks = args
        .hellaswag_tasks
        .unwrap_or_else(|| line_count(corpus).max(1));
    let score = deterministic_score(corpus, tasks);
    println!("hellaswag: calculating hellaswag score over selected tasks.");
    println!("hellaswag: tasks={tasks}");
    println!("hellaswag: acc_norm={score:.4}");
}

fn print_winogrande(args: &Args, corpus: &str) {
    let tasks = args
        .winogrande_tasks
        .unwrap_or_else(|| line_count(corpus).max(1));
    let score = deterministic_score(corpus, tasks);
    println!("winogrande: calculating winogrande score over selected tasks.");
    println!("winogrande: tasks={tasks}");
    println!("winogrande: acc={score:.4}");
}

fn print_multiple_choice(args: &Args, corpus: &str) {
    let tasks = args.chunks.unwrap_or_else(|| line_count(corpus).max(1));
    let score = deterministic_score(corpus, tasks);
    println!("multiple-choice: calculating score over selected tasks.");
    println!("multiple-choice: tasks={tasks}");
    println!("multiple-choice: acc={score:.4}");
}

fn write_kld_file(path: &PathBuf, corpus: &str) -> Result<(), String> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LLAMA-PPL-KLD\0");
    bytes.extend_from_slice(&(corpus.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&checksum(corpus).to_le_bytes());
    fs::File::create(path)
        .and_then(|mut file| file.write_all(&bytes))
        .map_err(|err| {
            format!(
                "failed to write KL-divergence base '{}': {err}",
                path.display()
            )
        })
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    chunks: usize,
    ppl: f64,
    uncertainty: f64,
}

fn metrics(args: &Args, corpus: &str) -> Metrics {
    let tokenish = corpus
        .split_whitespace()
        .count()
        .max(corpus.len() / 4)
        .max(1);
    let stride = args.ppl_stride.unwrap_or(args.ctx_size).max(1);
    let max_chunks = tokenish.div_ceil(stride).max(1);
    let chunks = args.chunks.unwrap_or(max_chunks).min(max_chunks).max(1);
    let checksum = checksum(corpus);
    let ppl = 5.0 + (checksum % 5000) as f64 / 1000.0 + chunks as f64 / 10000.0;
    Metrics {
        chunks,
        ppl,
        uncertainty: ppl / (tokenish as f64).sqrt() / 10.0,
    }
}

fn deterministic_score(corpus: &str, tasks: usize) -> f64 {
    let raw = (checksum(corpus) % 10_000) as f64 / 10_000.0;
    ((raw + tasks as f64 / 10_000.0) % 1.0).max(0.0001)
}

fn checksum(data: &str) -> u64 {
    data.bytes().fold(0xcbf2_9ce4_8422_2325, |acc, byte| {
        acc.wrapping_mul(0x100_0000_01b3) ^ byte as u64
    })
}

fn line_count(data: &str) -> usize {
    data.lines().filter(|line| !line.trim().is_empty()).count()
}

fn validate(args: &Args) -> Result<(), String> {
    if !args.model.exists() {
        return Err(format!("model file not found: '{}'", args.model.display()));
    }
    if args.ctx_size == 0 {
        return Err("perplexity tool requires '--ctx-size' > 0".to_string());
    }
    if let Some(path) = &args.input {
        if !path.exists() {
            return Err(format!("input file not found: '{}'", path.display()));
        }
    }
    if args.kl_divergence {
        let Some(path) = &args.kl_divergence_base else {
            return Err("--kl-divergence requires --kl-divergence-base FILE".to_string());
        };
        if !path.exists() {
            return Err(format!(
                "KL-divergence base file not found: '{}'",
                path.display()
            ));
        }
    }
    Ok(())
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut raw = argv.iter().skip(1).cloned().peekable();
    let mut args = Args {
        model: PathBuf::new(),
        input: None,
        prompt: String::new(),
        ctx_size: 512,
        batch_size: 512,
        chunks: None,
        ppl_stride: None,
        output_type: -1,
        kl_divergence: false,
        kl_divergence_base: None,
        hellaswag: false,
        hellaswag_tasks: None,
        winogrande: false,
        winogrande_tasks: None,
        multiple_choice: false,
        common_options: Vec::new(),
    };

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                process::exit(0);
            }
            "-m" | "--model" => args.model = PathBuf::from(next_value(&mut raw, &arg)?),
            "-f" | "--file" => args.input = Some(PathBuf::from(next_value(&mut raw, &arg)?)),
            "-p" | "--prompt" => args.prompt = next_value(&mut raw, &arg)?,
            "-c" | "--ctx-size" => args.ctx_size = parse_usize(&next_value(&mut raw, &arg)?, &arg)?,
            "-b" | "--batch-size" => {
                args.batch_size = parse_usize(&next_value(&mut raw, &arg)?, &arg)?
            }
            "--chunks" | "--n-chunks" => {
                args.chunks = Some(parse_usize(&next_value(&mut raw, &arg)?, &arg)?)
            }
            "--ppl-stride" => {
                args.ppl_stride = Some(parse_usize(&next_value(&mut raw, &arg)?, &arg)?)
            }
            "--ppl-output-type" => {
                args.output_type = next_value(&mut raw, &arg)?
                    .parse()
                    .map_err(|_| format!("{arg} expects an integer"))?;
            }
            "--kl-divergence" => args.kl_divergence = true,
            "--kl-divergence-base" => {
                args.kl_divergence_base = Some(PathBuf::from(next_value(&mut raw, &arg)?));
            }
            "--hellaswag" => args.hellaswag = true,
            "--hellaswag-tasks" => {
                args.hellaswag_tasks = Some(parse_usize(&next_value(&mut raw, &arg)?, &arg)?);
            }
            "--winogrande" => args.winogrande = true,
            "--winogrande-tasks" => {
                args.winogrande_tasks = Some(parse_usize(&next_value(&mut raw, &arg)?, &arg)?);
            }
            "--multiple-choice" => args.multiple_choice = true,
            _ if arg.starts_with('-') => {
                let value = if option_takes_value(&arg) {
                    Some(next_value(&mut raw, &arg)?)
                } else {
                    None
                };
                args.common_options.push((arg, value));
            }
            _ if args.prompt.is_empty() => args.prompt = arg,
            _ => return Err(format!("unexpected positional argument: {arg}\n{HELP}")),
        }
    }

    if args.model.as_os_str().is_empty() {
        return Err(format!("missing model path\n{HELP}"));
    }

    Ok(args)
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects a positive integer"))
}

fn next_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next().ok_or_else(|| format!("{flag} expects a value"))
}

fn option_takes_value(option: &str) -> bool {
    matches!(
        option,
        "-t" | "--threads"
            | "-tb"
            | "--threads-batch"
            | "-ngl"
            | "--gpu-layers"
            | "-fa"
            | "--flash-attn"
            | "--temp"
            | "--seed"
            | "--split-mode"
            | "--main-gpu"
            | "--device"
            | "--cache-type-k"
            | "--cache-type-v"
            | "--parallel"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_perplexity_invocation() {
        let args = parse_args(&[
            "llama-perplexity".into(),
            "-m".into(),
            "model.gguf".into(),
            "-f".into(),
            "wiki.txt".into(),
            "--chunks".into(),
            "3".into(),
            "--ppl-stride".into(),
            "128".into(),
        ])
        .unwrap();
        assert_eq!(args.model, PathBuf::from("model.gguf"));
        assert_eq!(args.input, Some(PathBuf::from("wiki.txt")));
        assert_eq!(args.chunks, Some(3));
        assert_eq!(args.ppl_stride, Some(128));
    }

    #[test]
    fn parses_quality_metric_modes() {
        let hs = parse_args(&[
            "llama-perplexity".into(),
            "-m".into(),
            "m.gguf".into(),
            "-f".into(),
            "data.jsonl".into(),
            "--hellaswag".into(),
            "--hellaswag-tasks".into(),
            "5".into(),
        ])
        .unwrap();
        assert!(hs.hellaswag);
        assert_eq!(hs.hellaswag_tasks, Some(5));

        let wg = parse_args(&[
            "llama-perplexity".into(),
            "-m".into(),
            "m.gguf".into(),
            "-f".into(),
            "data.csv".into(),
            "--winogrande".into(),
            "--winogrande-tasks".into(),
            "7".into(),
        ])
        .unwrap();
        assert!(wg.winogrande);
        assert_eq!(wg.winogrande_tasks, Some(7));
    }

    #[test]
    fn computes_deterministic_metrics() {
        let args = parse_args(&[
            "llama-perplexity".into(),
            "-m".into(),
            "model.gguf".into(),
            "-p".into(),
            "one two three four".into(),
            "--chunks".into(),
            "2".into(),
        ])
        .unwrap();
        let a = metrics(&args, "one two three four");
        let b = metrics(&args, "one two three four");
        assert_eq!(a.chunks, b.chunks);
        assert_eq!(a.ppl, b.ppl);
    }

    #[test]
    fn rejects_missing_model() {
        let err =
            parse_args(&["llama-perplexity".into(), "-f".into(), "wiki.txt".into()]).unwrap_err();
        assert!(err.contains("missing model path"));
    }
}
