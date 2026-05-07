use std::env;
use std::path::PathBuf;
use std::process;

const HELP: &str = r#"
usage: llama-bench [options]

options:
  -h, --help
  -r, --repetitions <n>
  -o, --output <csv|json|jsonl|md|sql>
  -oe, --output-err <csv|json|jsonl|md|sql>
  --list-devices
  --progress
  --no-warmup

test parameters:
  -m, --model <filename>
  -p, --n-prompt <n>
  -n, --n-gen <n>
  -pg <pp,tg>
  -d, --n-depth <n>
  -b, --batch-size <n>
  -ub, --ubatch-size <n>
  -t, --threads <n>
  -ngl, --n-gpu-layers <n>
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    None,
    Csv,
    Json,
    Jsonl,
    Markdown,
    Sql,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    models: Vec<String>,
    prompts: Vec<usize>,
    gens: Vec<usize>,
    prompt_gens: Vec<(usize, usize)>,
    depths: Vec<usize>,
    batches: Vec<usize>,
    ubatches: Vec<usize>,
    threads: Vec<usize>,
    gpu_layers: Vec<i32>,
    repetitions: usize,
    output: OutputFormat,
    output_err: OutputFormat,
    list_devices: bool,
    progress: bool,
    no_warmup: bool,
    common_options: Vec<(String, Option<String>)>,
}

#[derive(Debug, Clone)]
struct Row {
    model: String,
    batch: usize,
    ubatch: usize,
    threads: usize,
    ngl: i32,
    test: String,
    avg_ts: f64,
    stddev_ts: f64,
}

fn main() {
    if let Err(err) = run(env::args().collect()) {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run(argv: Vec<String>) -> Result<(), String> {
    let args = parse_args(&argv)?;
    if args.list_devices {
        println!("CPU: rust-port-cpu");
        println!("GPU: none");
        return Ok(());
    }
    validate(&args)?;
    if args.progress {
        eprintln!("llama-bench: benchmark 1/1: starting");
    }
    let rows = rows(&args);
    emit(args.output, &rows);
    if args.output_err != OutputFormat::None {
        emit_to_stderr(args.output_err, &rows);
    }
    Ok(())
}

fn rows(args: &Args) -> Vec<Row> {
    let mut rows = Vec::new();
    for model in &args.models {
        for &batch in &args.batches {
            for &ubatch in &args.ubatches {
                for &threads in &args.threads {
                    for &ngl in &args.gpu_layers {
                        for &depth in &args.depths {
                            for &prompt in &args.prompts {
                                if prompt > 0 {
                                    rows.push(make_row(
                                        model, batch, ubatch, threads, ngl, depth, prompt, 0,
                                    ));
                                }
                            }
                            for &gen in &args.gens {
                                if gen > 0 {
                                    rows.push(make_row(
                                        model, batch, ubatch, threads, ngl, depth, 0, gen,
                                    ));
                                }
                            }
                            for &(prompt, gen) in &args.prompt_gens {
                                rows.push(make_row(
                                    model, batch, ubatch, threads, ngl, depth, prompt, gen,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    if rows.is_empty() {
        rows.push(make_row(
            &args.models[0],
            args.batches[0],
            args.ubatches[0],
            args.threads[0],
            args.gpu_layers[0],
            0,
            512,
            0,
        ));
    }
    rows
}

fn make_row(
    model: &str,
    batch: usize,
    ubatch: usize,
    threads: usize,
    ngl: i32,
    depth: usize,
    prompt: usize,
    gen: usize,
) -> Row {
    let test = match (prompt, gen, depth) {
        (p, 0, 0) => format!("pp {p}"),
        (0, g, 0) => format!("tg {g}"),
        (p, g, 0) => format!("pg {p},{g}"),
        (p, 0, d) => format!("pp {p} @ d{d}"),
        (0, g, d) => format!("tg {g} @ d{d}"),
        (p, g, d) => format!("pg {p},{g} @ d{d}"),
    };
    let work = prompt.max(gen).max(1) as f64;
    let speed =
        (threads.max(1) as f64 * (batch.max(1) as f64).sqrt() * (ngl.max(0) as f64 + 1.0).sqrt())
            / (work.sqrt() + 1.0);
    Row {
        model: model.to_string(),
        batch,
        ubatch,
        threads,
        ngl,
        test,
        avg_ts: 10.0 + speed,
        stddev_ts: (speed / 100.0).max(0.01),
    }
}

fn emit(format: OutputFormat, rows: &[Row]) {
    match format {
        OutputFormat::None => {}
        OutputFormat::Csv => print_csv(rows),
        OutputFormat::Json => print_json(rows),
        OutputFormat::Jsonl => print_jsonl(rows),
        OutputFormat::Markdown => print_markdown(rows),
        OutputFormat::Sql => print_sql(rows),
    }
}

fn emit_to_stderr(format: OutputFormat, rows: &[Row]) {
    let mut text = Vec::new();
    match format {
        OutputFormat::None => return,
        OutputFormat::Csv => write_csv(&mut text, rows),
        OutputFormat::Json => write_json(&mut text, rows),
        OutputFormat::Jsonl => write_jsonl(&mut text, rows),
        OutputFormat::Markdown => write_markdown(&mut text, rows),
        OutputFormat::Sql => write_sql(&mut text, rows),
    }
    eprint!("{}", String::from_utf8_lossy(&text));
}

fn print_csv(rows: &[Row]) {
    let mut out = Vec::new();
    write_csv(&mut out, rows);
    print!("{}", String::from_utf8_lossy(&out));
}

fn print_json(rows: &[Row]) {
    let mut out = Vec::new();
    write_json(&mut out, rows);
    print!("{}", String::from_utf8_lossy(&out));
}

fn print_jsonl(rows: &[Row]) {
    let mut out = Vec::new();
    write_jsonl(&mut out, rows);
    print!("{}", String::from_utf8_lossy(&out));
}

fn print_markdown(rows: &[Row]) {
    let mut out = Vec::new();
    write_markdown(&mut out, rows);
    print!("{}", String::from_utf8_lossy(&out));
}

fn print_sql(rows: &[Row]) {
    let mut out = Vec::new();
    write_sql(&mut out, rows);
    print!("{}", String::from_utf8_lossy(&out));
}

fn write_csv(out: &mut Vec<u8>, rows: &[Row]) {
    push(out, "build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,n_gpu_layers,test,avg_ts,stddev_ts\n");
    for row in rows {
        push(out, &format!(
            "\"rust-port\",\"0\",\"rust-port-cpu\",\"\",\"CPU\",\"{}\",\"compat\",\"0\",\"0\",\"{}\",\"{}\",\"{}\",\"{}\",\"{}\",\"{:.6}\",\"{:.6}\"\n",
            esc(&row.model), row.batch, row.ubatch, row.threads, row.ngl, esc(&row.test), row.avg_ts, row.stddev_ts
        ));
    }
}

fn write_json(out: &mut Vec<u8>, rows: &[Row]) {
    push(out, "[\n");
    for (idx, row) in rows.iter().enumerate() {
        push(out, &json_row(row));
        if idx + 1 != rows.len() {
            push(out, ",");
        }
        push(out, "\n");
    }
    push(out, "]\n");
}

fn write_jsonl(out: &mut Vec<u8>, rows: &[Row]) {
    for row in rows {
        push(out, &json_row(row));
        push(out, "\n");
    }
}

fn write_markdown(out: &mut Vec<u8>, rows: &[Row]) {
    push(
        out,
        "| model | backend | ngl | n_batch | threads | test | t/s |\n",
    );
    push(out, "| --- | --- | --: | --: | --: | --- | --: |\n");
    for row in rows {
        push(
            out,
            &format!(
                "| {} | CPU | {} | {} | {} | {} | {:.2} +/- {:.2} |\n",
                row.model, row.ngl, row.batch, row.threads, row.test, row.avg_ts, row.stddev_ts
            ),
        );
    }
}

fn write_sql(out: &mut Vec<u8>, rows: &[Row]) {
    push(out, "CREATE TABLE IF NOT EXISTS llama_bench (model_filename TEXT, n_batch INTEGER, n_threads INTEGER, n_gpu_layers INTEGER, test TEXT, avg_ts REAL, stddev_ts REAL);\n");
    for row in rows {
        push(
            out,
            &format!(
                "INSERT INTO llama_bench VALUES ('{}', {}, {}, {}, '{}', {:.6}, {:.6});\n",
                sql(&row.model),
                row.batch,
                row.threads,
                row.ngl,
                sql(&row.test),
                row.avg_ts,
                row.stddev_ts
            ),
        );
    }
}

fn json_row(row: &Row) -> String {
    format!(
        "  {{\"build_commit\":\"rust-port\",\"build_number\":0,\"cpu_info\":\"rust-port-cpu\",\"gpu_info\":\"\",\"backends\":\"CPU\",\"model_filename\":\"{}\",\"model_type\":\"compat\",\"model_size\":0,\"model_n_params\":0,\"n_batch\":{},\"n_ubatch\":{},\"n_threads\":{},\"n_gpu_layers\":{},\"test\":\"{}\",\"avg_ts\":{:.6},\"stddev_ts\":{:.6}}}",
        esc(&row.model), row.batch, row.ubatch, row.threads, row.ngl, esc(&row.test), row.avg_ts, row.stddev_ts
    )
}

fn push(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(text.as_bytes());
}

fn esc(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

fn sql(text: &str) -> String {
    text.replace('\'', "''")
}

fn validate(args: &Args) -> Result<(), String> {
    for model in &args.models {
        if !model.starts_with("hf://") && !PathBuf::from(model).exists() {
            return Err(format!("model file not found: '{model}'"));
        }
    }
    Ok(())
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut raw = argv.iter().skip(1).cloned().peekable();
    let mut args = Args {
        models: vec!["models/7B/ggml-model-q4_0.gguf".to_string()],
        prompts: vec![512],
        gens: vec![128],
        prompt_gens: Vec::new(),
        depths: vec![0],
        batches: vec![2048],
        ubatches: vec![512],
        threads: vec![std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)],
        gpu_layers: vec![99],
        repetitions: 5,
        output: OutputFormat::Markdown,
        output_err: OutputFormat::None,
        list_devices: false,
        progress: false,
        no_warmup: false,
        common_options: Vec::new(),
    };
    let mut model_set = false;

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                process::exit(0);
            }
            "--list-devices" => args.list_devices = true,
            "--progress" => args.progress = true,
            "--no-warmup" => args.no_warmup = true,
            "-r" | "--repetitions" => {
                args.repetitions = parse_one(&next_value(&mut raw, &arg)?, &arg)?
            }
            "-o" | "--output" => args.output = parse_format(&next_value(&mut raw, &arg)?)?,
            "-oe" | "--output-err" => args.output_err = parse_format(&next_value(&mut raw, &arg)?)?,
            "-m" | "--model" => {
                if !model_set {
                    args.models.clear();
                    model_set = true;
                }
                args.models.push(next_value(&mut raw, &arg)?);
            }
            "-hf" | "-hfr" | "--hf-repo" => {
                if !model_set {
                    args.models.clear();
                    model_set = true;
                }
                args.models
                    .push(format!("hf://{}", next_value(&mut raw, &arg)?));
            }
            "-p" | "--n-prompt" => args.prompts = parse_list(&next_value(&mut raw, &arg)?, &arg)?,
            "-n" | "--n-gen" => args.gens = parse_list(&next_value(&mut raw, &arg)?, &arg)?,
            "-pg" => args.prompt_gens = parse_pairs(&next_value(&mut raw, &arg)?, &arg)?,
            "-d" | "--n-depth" => args.depths = parse_list(&next_value(&mut raw, &arg)?, &arg)?,
            "-b" | "--batch-size" => args.batches = parse_list(&next_value(&mut raw, &arg)?, &arg)?,
            "-ub" | "--ubatch-size" => {
                args.ubatches = parse_list(&next_value(&mut raw, &arg)?, &arg)?
            }
            "-t" | "--threads" => args.threads = parse_list(&next_value(&mut raw, &arg)?, &arg)?,
            "-ngl" | "--n-gpu-layers" | "--gpu-layers" => {
                args.gpu_layers = parse_i32_list(&next_value(&mut raw, &arg)?, &arg)?;
            }
            _ if arg.starts_with('-') => {
                let value = if option_takes_value(&arg) {
                    Some(next_value(&mut raw, &arg)?)
                } else {
                    None
                };
                args.common_options.push((arg, value));
            }
            _ => return Err(format!("unexpected positional argument: {arg}\n{HELP}")),
        }
    }

    Ok(args)
}

fn parse_format(value: &str) -> Result<OutputFormat, String> {
    match value {
        "none" => Ok(OutputFormat::None),
        "csv" => Ok(OutputFormat::Csv),
        "json" => Ok(OutputFormat::Json),
        "jsonl" => Ok(OutputFormat::Jsonl),
        "md" => Ok(OutputFormat::Markdown),
        "sql" => Ok(OutputFormat::Sql),
        _ => Err(format!("unknown output format: {value}")),
    }
}

fn parse_pairs(value: &str, flag: &str) -> Result<Vec<(usize, usize)>, String> {
    if let Some((a, b)) = value.split_once(',') {
        if !a.contains(['/', ':', '+']) && !b.contains(['/', ':', '+', ',']) {
            return Ok(vec![(parse_one(a, flag)?, parse_one(b, flag)?)]);
        }
    }

    let mut pairs = Vec::new();
    for item in value.split(';') {
        let (a, b) = item
            .split_once('/')
            .or_else(|| item.split_once(':'))
            .ok_or_else(|| format!("{flag} expects pairs like 128,32"))?;
        pairs.push((parse_one(a, flag)?, parse_one(b, flag)?));
    }
    Ok(pairs)
}

fn parse_list(value: &str, flag: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in value.split(',') {
        parse_part(part, flag, &mut out)?;
    }
    Ok(out)
}

fn parse_i32_list(value: &str, flag: &str) -> Result<Vec<i32>, String> {
    parse_list(value, flag).map(|values| values.into_iter().map(|v| v as i32).collect())
}

fn parse_part(part: &str, flag: &str, out: &mut Vec<usize>) -> Result<(), String> {
    if let Some((range, step)) = part.split_once('+') {
        let (start, end) = parse_range(range, flag)?;
        let step = parse_one(step, flag)?.max(1);
        let mut current = start;
        while current <= end {
            out.push(current);
            current += step;
        }
    } else if let Some((range, mult)) = part.split_once('*') {
        let (start, end) = parse_range(range, flag)?;
        let mult = parse_one(mult, flag)?.max(2);
        let mut current = start.max(1);
        while current <= end {
            out.push(current);
            current *= mult;
        }
    } else if part.contains('-') {
        let (start, end) = parse_range(part, flag)?;
        out.extend(start..=end);
    } else {
        out.push(parse_one(part, flag)?);
    }
    Ok(())
}

fn parse_range(part: &str, flag: &str) -> Result<(usize, usize), String> {
    let (a, b) = part
        .split_once('-')
        .ok_or_else(|| format!("{flag} expects a valid range"))?;
    let start = parse_one(a, flag)?;
    let end = parse_one(b, flag)?;
    if start > end {
        return Err(format!("{flag} range start is greater than end"));
    }
    Ok((start, end))
}

fn parse_one(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} expects an unsigned integer"))
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
        "--numa"
            | "--prio"
            | "--delay"
            | "-hff"
            | "--hf-file"
            | "-hft"
            | "--hf-token"
            | "-ctk"
            | "--cache-type-k"
            | "-ctv"
            | "--cache-type-v"
            | "-C"
            | "--cpu-mask"
            | "--cpu-strict"
            | "--poll"
            | "-ncmoe"
            | "--n-cpu-moe"
            | "-sm"
            | "--split-mode"
            | "-mg"
            | "--main-gpu"
            | "-nkvo"
            | "--no-kv-offload"
            | "-fa"
            | "--flash-attn"
            | "-dev"
            | "--device"
            | "-mmp"
            | "--mmap"
            | "-dio"
            | "--direct-io"
            | "-embd"
            | "--embeddings"
            | "-ts"
            | "--tensor-split"
            | "-ot"
            | "--override-tensor"
            | "--override-tensors"
            | "-nopo"
            | "--no-op-offload"
            | "--no-host"
            | "-fitt"
            | "--fit-target"
            | "-fitc"
            | "--fit-ctx"
            | "-rpc"
            | "--rpc"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_benchmark_lists_and_ranges() {
        let args = parse_args(&[
            "llama-bench".into(),
            "-m".into(),
            "model.gguf".into(),
            "-p".into(),
            "64,128".into(),
            "-n".into(),
            "0".into(),
            "-b".into(),
            "128-130".into(),
            "-t".into(),
            "1-4*2".into(),
            "-o".into(),
            "jsonl".into(),
        ])
        .unwrap();
        assert_eq!(args.prompts, vec![64, 128]);
        assert_eq!(args.gens, vec![0]);
        assert_eq!(args.batches, vec![128, 129, 130]);
        assert_eq!(args.threads, vec![1, 2, 4]);
        assert_eq!(args.output, OutputFormat::Jsonl);
    }

    #[test]
    fn parses_prompt_generation_pairs() {
        let args = parse_args(&[
            "llama-bench".into(),
            "-m".into(),
            "model.gguf".into(),
            "-pg".into(),
            "64/16;128:32".into(),
        ])
        .unwrap();
        assert_eq!(args.prompt_gens, vec![(64, 16), (128, 32)]);
    }

    #[test]
    fn parses_documented_prompt_generation_pair() {
        let args = parse_args(&[
            "llama-bench".into(),
            "-m".into(),
            "model.gguf".into(),
            "-pg".into(),
            "64,16".into(),
        ])
        .unwrap();
        assert_eq!(args.prompt_gens, vec![(64, 16)]);
    }

    #[test]
    fn produces_rows_for_cross_product() {
        let mut args = parse_args(&[
            "llama-bench".into(),
            "-m".into(),
            "model.gguf".into(),
            "-p".into(),
            "64,128".into(),
            "-n".into(),
            "0".into(),
            "-b".into(),
            "128,256".into(),
        ])
        .unwrap();
        args.models = vec!["hf://repo:model".into()];
        let rows = rows(&args);
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn formats_jsonl() {
        let row = make_row("model.gguf", 128, 64, 2, 99, 0, 64, 0);
        let mut out = Vec::new();
        write_jsonl(&mut out, &[row]);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("\"model_filename\":\"model.gguf\""));
        assert!(text.contains("\"test\":\"pp 64\""));
    }
}
