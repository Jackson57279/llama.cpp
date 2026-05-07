#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub n_gpu_layers: i32,
    pub save_logits: bool,
    pub embedding: bool,
    pub embd_normalize: i32,
    pub logits_output_dir: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingModel,
    MissingValue(&'static str),
    InvalidInteger(&'static str, String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m model.gguf"),
            ParseError::MissingValue(flag) => write!(f, "missing value for {flag}"),
            ParseError::InvalidInteger(flag, value) => {
                write!(f, "invalid integer for {flag}: {value}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

pub fn parse_args<I, S>(args: I) -> Result<Args, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let mut parsed = Args {
        model_path: String::new(),
        prompt: "Hello my name is".to_string(),
        n_gpu_layers: 99,
        save_logits: false,
        embedding: false,
        embd_normalize: -1,
        logits_output_dir: "data".to_string(),
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => {
                parsed.model_path = args.next().ok_or(ParseError::MissingValue("-m"))?;
            }
            "-p" | "--prompt" => {
                parsed.prompt = args.next().ok_or(ParseError::MissingValue("-p"))?;
            }
            "-ngl" | "--n-gpu-layers" => {
                let value = args.next().ok_or(ParseError::MissingValue("-ngl"))?;
                parsed.n_gpu_layers = value
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("-ngl", value))?;
            }
            "--save-logits" => parsed.save_logits = true,
            "--embedding" => parsed.embedding = true,
            "--embd-normalize" => {
                let value = args
                    .next()
                    .ok_or(ParseError::MissingValue("--embd-normalize"))?;
                parsed.embd_normalize = value
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("--embd-normalize", value))?;
            }
            "--logits-output-dir" => {
                parsed.logits_output_dir = args
                    .next()
                    .ok_or(ParseError::MissingValue("--logits-output-dir"))?;
            }
            _ => {}
        }
    }

    if parsed.model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }

    Ok(parsed)
}

pub fn normalize_embedding(input: &[f32], embd_norm: i32) -> Vec<f32> {
    let sum = match embd_norm {
        -1 => 1.0,
        0 => input.iter().map(|v| v.abs() as f64).fold(0.0_f64, f64::max) / 32760.0,
        2 => input
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt(),
        p => input
            .iter()
            .map(|v| (v.abs() as f64).powi(p))
            .sum::<f64>()
            .powf(1.0 / p as f64),
    };
    let norm = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    input.iter().map(|v| v * norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_model_conversion_invocation() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "--embedding",
            "-p",
            "Hello world today",
            "--save-logits",
            "--embd-normalize",
            "2",
        ])
        .unwrap();

        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "Hello world today");
        assert!(args.embedding);
        assert!(args.save_logits);
        assert_eq!(args.embd_normalize, 2);
        assert_eq!(args.logits_output_dir, "data");
    }

    #[test]
    fn normalizes_euclidean() {
        let out = normalize_embedding(&[3.0, 4.0], 2);
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn rejects_missing_model() {
        assert_eq!(
            parse_args(["--save-logits"]).unwrap_err(),
            ParseError::MissingModel
        );
    }
}
