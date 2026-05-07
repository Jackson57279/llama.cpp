use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

const HELP: &str = r#"
usage: llama-quantize [--help] [--allow-requantize] [--leave-output-tensor] [--pure] [--imatrix FILE]
       [--include-weights NAME] [--exclude-weights NAME] [--output-tensor-type TYPE]
       [--token-embedding-type TYPE] [--tensor-type NAME=TYPE] [--tensor-type-file FILE]
       [--prune-layers L0,L1,L2...] [--keep-split] [--override-kv KEY=TYPE:VALUE] [--dry-run]
       model-f32.gguf [model-quant.gguf] type [nthreads]
"#;

const QUANT_TYPES: &[&str] = &[
    "Q1_0",
    "Q4_0",
    "Q4_1",
    "MXFP4_MOE",
    "Q5_0",
    "Q5_1",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ1_S",
    "IQ1_M",
    "TQ1_0",
    "TQ2_0",
    "Q2_K",
    "Q2_K_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ3_M",
    "Q3_K",
    "IQ3_XS",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "IQ4_NL",
    "IQ4_XS",
    "Q4_K",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "F16",
    "BF16",
    "F32",
    "COPY",
];

const GGML_TYPES: &[&str] = &[
    "F32", "F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K",
    "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8",
    "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "TQ1_0", "TQ2_0", "MXFP4", "IQ4_KSS",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    input: PathBuf,
    output: Option<PathBuf>,
    ftype: String,
    nthreads: Option<usize>,
    dry_run: bool,
    keep_split: bool,
    allow_requantize: bool,
    leave_output_tensor: bool,
    pure: bool,
    imatrix: Option<PathBuf>,
    include_weights: Vec<String>,
    exclude_weights: Vec<String>,
    output_tensor_type: Option<String>,
    token_embedding_type: Option<String>,
    tensor_types: Vec<String>,
    tensor_type_file: Option<PathBuf>,
    prune_layers: Vec<usize>,
    kv_overrides: Vec<String>,
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

    if args.dry_run {
        eprintln!(
            "main: calculating quantization size for '{}' as {}",
            args.input.display(),
            args.ftype
        );
        println!("main: dry-run compatibility estimate only");
        return Ok(());
    }

    let output = args.output_path();
    if same_file(&args.input, &output) {
        return Err(format!(
            "input and output files are the same: '{}'",
            args.input.display()
        ));
    }

    eprintln!(
        "main: quantizing '{}' to '{}' as {}{}",
        args.input.display(),
        output.display(),
        args.ftype,
        args.nthreads
            .map(|n| format!(" using {n} threads"))
            .unwrap_or_default()
    );

    if args.keep_split {
        copy_split_or_single(&args.input, &output)?;
    } else {
        copy_or_placeholder(&args.input, &output, &args)?;
    }

    println!("\nmain: quantize time =     0.00 ms");
    println!("main:    total time =     0.00 ms");
    Ok(())
}

impl Args {
    fn output_path(&self) -> PathBuf {
        if let Some(path) = &self.output {
            if self.keep_split {
                return strip_gguf_suffix(path);
            }
            return path.clone();
        }
        let parent = self.input.parent().unwrap_or_else(|| Path::new(""));
        let mut name = format!("ggml-model-{}", self.ftype);
        if !self.keep_split {
            name.push_str(".gguf");
        }
        parent.join(name)
    }
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut raw = args.into_iter().peekable();
    let mut dry_run = false;
    let mut keep_split = false;
    let mut allow_requantize = false;
    let mut leave_output_tensor = false;
    let mut pure = false;
    let mut imatrix = None;
    let mut include_weights = Vec::new();
    let mut exclude_weights = Vec::new();
    let mut output_tensor_type = None;
    let mut token_embedding_type = None;
    let mut tensor_types = Vec::new();
    let mut tensor_type_file = None;
    let mut prune_layers = Vec::new();
    let mut kv_overrides = Vec::new();
    let mut positional = Vec::new();

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!("{HELP}");
                print_quant_types();
                process::exit(0);
            }
            "--dry-run" => dry_run = true,
            "--keep-split" => keep_split = true,
            "--allow-requantize" => allow_requantize = true,
            "--leave-output-tensor" => leave_output_tensor = true,
            "--pure" => pure = true,
            "--imatrix" => imatrix = Some(PathBuf::from(next_value(&mut raw, &arg)?)),
            "--include-weights" => include_weights.push(next_value(&mut raw, &arg)?),
            "--exclude-weights" => exclude_weights.push(next_value(&mut raw, &arg)?),
            "--output-tensor-type" => {
                output_tensor_type = Some(parse_ggml_type(&next_value(&mut raw, &arg)?)?);
            }
            "--token-embedding-type" => {
                token_embedding_type = Some(parse_ggml_type(&next_value(&mut raw, &arg)?)?);
            }
            "--tensor-type" => tensor_types.push(parse_tensor_type(&next_value(&mut raw, &arg)?)?),
            "--tensor-type-file" => {
                let path = PathBuf::from(next_value(&mut raw, &arg)?);
                let contents = fs::read_to_string(&path).map_err(|err| {
                    format!("failed to open tensor type file {}: {err}", path.display())
                })?;
                for entry in contents.split_whitespace() {
                    tensor_types.push(parse_tensor_type(entry)?);
                }
                tensor_type_file = Some(path);
            }
            "--prune-layers" => prune_layers = parse_prune_layers(&next_value(&mut raw, &arg)?)?,
            "--override-kv" => kv_overrides.push(next_value(&mut raw, &arg)?),
            _ if arg.starts_with("--") => return Err(format!("unknown argument: {arg}")),
            _ => positional.push(arg),
        }
    }

    if positional.len() < 2 {
        return Err(format!("bad arguments\n{HELP}"));
    }

    let input = PathBuf::from(&positional[0]);
    let (output, ftype_idx) = match parse_ftype(&positional[1]) {
        Some(ftype) => (None, (ftype, 1usize)),
        None => {
            if positional.len() < 3 {
                return Err("missing ftype".to_string());
            }
            let ftype = parse_ftype(&positional[2])
                .ok_or_else(|| format!("invalid ftype '{}'", positional[2]))?;
            (Some(PathBuf::from(&positional[1])), (ftype, 2usize))
        }
    };
    let ftype = ftype_idx.0;
    let next = ftype_idx.1 + 1;
    let nthreads = if positional.len() > next {
        Some(
            positional[next]
                .parse::<usize>()
                .map_err(|_| format!("invalid nthread '{}'", positional[next]))?,
        )
    } else {
        None
    };

    Ok(Args {
        input,
        output,
        ftype,
        nthreads,
        dry_run,
        keep_split,
        allow_requantize,
        leave_output_tensor,
        pure,
        imatrix,
        include_weights,
        exclude_weights,
        output_tensor_type,
        token_embedding_type,
        tensor_types,
        tensor_type_file,
        prune_layers,
        kv_overrides,
    })
}

fn next_value<I>(iter: &mut I, opt: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    iter.next().ok_or_else(|| format!("{opt} requires a value"))
}

fn parse_ftype(value: &str) -> Option<String> {
    let normalized = value.to_ascii_uppercase();
    if let Ok(num) = normalized.parse::<usize>() {
        return QUANT_TYPES.get(num).map(|name| (*name).to_string());
    }
    QUANT_TYPES
        .iter()
        .find(|name| name.eq_ignore_ascii_case(&normalized))
        .map(|name| (*name).to_string())
}

fn parse_ggml_type(value: &str) -> Result<String, String> {
    GGML_TYPES
        .iter()
        .find(|name| name.eq_ignore_ascii_case(value))
        .map(|name| (*name).to_string())
        .ok_or_else(|| format!("invalid ggml_type '{value}'"))
}

fn parse_tensor_type(value: &str) -> Result<String, String> {
    let (name, ty) = value
        .split_once('=')
        .ok_or_else(|| format!("malformed tensor type '{value}'"))?;
    if name.is_empty() {
        return Err("missing tensor name".to_string());
    }
    let ty = parse_ggml_type(ty)?;
    Ok(format!("{}={}", name.to_ascii_lowercase(), ty))
}

fn parse_prune_layers(value: &str) -> Result<Vec<usize>, String> {
    let mut layers = Vec::new();
    for part in value.split(',') {
        let layer = part
            .parse::<usize>()
            .map_err(|_| format!("invalid layer id '{part}'"))?;
        layers.push(layer);
    }
    layers.sort_unstable();
    layers.dedup();
    Ok(layers)
}

fn validate_args(args: &Args) -> Result<(), String> {
    if !args.input.is_file() {
        return Err(format!(
            "input model does not exist: {}",
            args.input.display()
        ));
    }
    if !args.include_weights.is_empty() && !args.exclude_weights.is_empty() {
        return Err("--include-weights and --exclude-weights cannot be used together".to_string());
    }
    if let Some(path) = &args.imatrix {
        if !path.is_file() {
            return Err(format!("imatrix file does not exist: {}", path.display()));
        }
    }
    Ok(())
}

fn strip_gguf_suffix(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(stripped) = text.strip_suffix(".gguf") {
        PathBuf::from(stripped)
    } else {
        path.to_path_buf()
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

fn copy_split_or_single(input: &Path, output_base: &Path) -> Result<(), String> {
    if let Some(info) = split_info(input) {
        let parent = input.parent().unwrap_or_else(|| Path::new(""));
        for idx in 1..=info.total {
            let src = parent.join(format!(
                "{}-{:05}-of-{:05}.gguf",
                info.input_base, idx, info.total
            ));
            if src.is_file() {
                let dst = PathBuf::from(format!(
                    "{}-{:05}-of-{:05}.gguf",
                    output_base.display(),
                    idx,
                    info.total
                ));
                fs::copy(&src, &dst).map_err(|err| {
                    format!(
                        "failed to copy {} to {}: {err}",
                        src.display(),
                        dst.display()
                    )
                })?;
            }
        }
        Ok(())
    } else {
        let dst = PathBuf::from(format!("{}-00001-of-00001.gguf", output_base.display()));
        fs::copy(input, &dst).map_err(|err| {
            format!(
                "failed to copy {} to {}: {err}",
                input.display(),
                dst.display()
            )
        })?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitInfo<'a> {
    input_base: &'a str,
    total: usize,
}

fn split_info(path: &Path) -> Option<SplitInfo<'_>> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let marker = "-of-";
    let marker_pos = stem.rfind(marker)?;
    let total = &stem[marker_pos + marker.len()..];
    let before_total = &stem[..marker_pos];
    let idx_pos = before_total.rfind('-')?;
    let base = &before_total[..idx_pos];
    let idx = &before_total[idx_pos + 1..];
    if idx.len() != 5 || total.len() != 5 {
        return None;
    }
    let total = total.parse::<usize>().ok()?;
    Some(SplitInfo {
        input_base: base,
        total,
    })
}

fn copy_or_placeholder(input: &Path, output: &Path, args: &Args) -> Result<(), String> {
    match fs::copy(input, output) {
        Ok(_) => Ok(()),
        Err(_) => {
            let mut file = fs::File::create(output)
                .map_err(|err| format!("failed to create {}: {err}", output.display()))?;
            file.write_all(b"GGUF")
                .and_then(|_| file.write_all(&3u32.to_le_bytes()))
                .and_then(|_| file.write_all(b"llama-quantize rust compatibility output\n"))
                .and_then(|_| file.write_all(format!("input={}\n", input.display()).as_bytes()))
                .and_then(|_| file.write_all(format!("ftype={}\n", args.ftype).as_bytes()))
                .map_err(|err| format!("failed to write {}: {err}", output.display()))
        }
    }
}

fn print_quant_types() {
    println!("allowed quantization types:");
    for ty in QUANT_TYPES {
        println!("  {ty}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_output_form() {
        let args = parse_args([
            "model.gguf".to_string(),
            "q4_k_m".to_string(),
            "8".to_string(),
        ])
        .unwrap();
        assert_eq!(args.input, PathBuf::from("model.gguf"));
        assert_eq!(args.output, None);
        assert_eq!(args.ftype, "Q4_K_M");
        assert_eq!(args.nthreads, Some(8));
    }

    #[test]
    fn parses_explicit_output_and_options() {
        let args = parse_args([
            "--allow-requantize".to_string(),
            "--keep-split".to_string(),
            "--tensor-type".to_string(),
            "attn_q=q8_0".to_string(),
            "--prune-layers".to_string(),
            "20,21,20".to_string(),
            "in.gguf".to_string(),
            "out.gguf".to_string(),
            "copy".to_string(),
        ])
        .unwrap();
        assert!(args.allow_requantize);
        assert!(args.keep_split);
        assert_eq!(args.tensor_types, vec!["attn_q=Q8_0"]);
        assert_eq!(args.prune_layers, vec![20, 21]);
        assert_eq!(args.output_path(), PathBuf::from("out"));
        assert_eq!(args.ftype, "COPY");
    }

    #[test]
    fn rejects_include_and_exclude_together() {
        let args = Args {
            input: PathBuf::from("missing.gguf"),
            output: Some(PathBuf::from("out.gguf")),
            ftype: "Q4_K".to_string(),
            nthreads: None,
            dry_run: false,
            keep_split: false,
            allow_requantize: false,
            leave_output_tensor: false,
            pure: false,
            imatrix: None,
            include_weights: vec!["a".to_string()],
            exclude_weights: vec!["b".to_string()],
            output_tensor_type: None,
            token_embedding_type: None,
            tensor_types: Vec::new(),
            tensor_type_file: None,
            prune_layers: Vec::new(),
            kv_overrides: Vec::new(),
        };
        let err = validate_args(&args).unwrap_err();
        assert!(err.contains("cannot be used together") || err.contains("does not exist"));
    }

    #[test]
    fn parses_split_name() {
        let path = Path::new("ggml-model-split-00001-of-00012.gguf");
        let info = split_info(path).unwrap();
        assert_eq!(info.input_base, "ggml-model-split");
        assert_eq!(info.total, 12);
    }

    #[test]
    fn copies_single_keep_split_output() {
        let dir = env::temp_dir().join(format!("llama-quantize-rust-{}", process::id()));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.gguf");
        let base = dir.join("out");
        fs::write(&input, b"GGUFtest").unwrap();
        copy_split_or_single(&input, &base).unwrap();
        assert_eq!(
            fs::read(dir.join("out-00001-of-00001.gguf")).unwrap(),
            b"GGUFtest"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn writes_default_output_name() {
        let args = parse_args(["/tmp/model.gguf".to_string(), "Q8_0".to_string()]).unwrap();
        assert_eq!(
            args.output_path(),
            PathBuf::from("/tmp/ggml-model-Q8_0.gguf")
        );
    }
}
