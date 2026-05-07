#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub output_path: String,
    pub prompt: String,
    pub check: bool,
    pub n_gpu_layers: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingModel,
    MissingOutput,
    MissingValue(&'static str),
    InvalidInteger(&'static str, String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required --model model.gguf"),
            ParseError::MissingOutput => write!(f, "missing required --output results.gguf"),
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
        output_path: String::new(),
        prompt: String::new(),
        check: false,
        n_gpu_layers: 99,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => {
                parsed.model_path = args.next().ok_or(ParseError::MissingValue("--model"))?;
            }
            "-o" | "--output" | "--output-file" => {
                parsed.output_path = args.next().ok_or(ParseError::MissingValue("--output"))?;
            }
            "-p" | "--prompt" => {
                parsed.prompt = args.next().ok_or(ParseError::MissingValue("--prompt"))?;
            }
            "--check" => parsed.check = true,
            "-ngl" | "--n-gpu-layers" => {
                let value = args.next().ok_or(ParseError::MissingValue("-ngl"))?;
                parsed.n_gpu_layers = value
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("-ngl", value))?;
            }
            _ => {}
        }
    }

    if parsed.model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }
    if parsed.output_path.is_empty() {
        return Err(ParseError::MissingOutput);
    }
    Ok(parsed)
}

pub fn nmse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut mse_a_b = 0.0;
    let mut mse_a_0 = 0.0;
    for (&a_i, &b_i) in a.iter().zip(b) {
        let a_i = a_i as f64;
        let b_i = b_i as f64;
        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }
    mse_a_b / mse_a_0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_readme_invocation() {
        let args = parse_args([
            "--model",
            "model.gguf",
            "--output",
            "results.gguf",
            "--prompt",
            "People die when they are killed.",
            "--check",
        ])
        .unwrap();

        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.output_path, "results.gguf");
        assert_eq!(args.prompt, "People die when they are killed.");
        assert!(args.check);
    }

    #[test]
    fn computes_nmse() {
        let value = nmse(&[1.0, 2.0], &[1.0, 1.0]);
        assert!((value - 0.2).abs() < 1e-12);
    }

    #[test]
    fn rejects_missing_output() {
        assert_eq!(
            parse_args(["--model", "model.gguf"]).unwrap_err(),
            ParseError::MissingOutput
        );
    }
}
