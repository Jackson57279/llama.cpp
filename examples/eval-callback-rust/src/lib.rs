#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub n_gpu_layers: i32,
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
    let mut model_path = String::new();
    let mut prompt = "hello".to_string();
    let mut n_gpu_layers = 99;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => {
                model_path = args.next().ok_or(ParseError::MissingValue("-m"))?;
            }
            "-p" | "--prompt" => {
                prompt = args.next().ok_or(ParseError::MissingValue("--prompt"))?;
            }
            "-ngl" | "--n-gpu-layers" => {
                let value = args.next().ok_or(ParseError::MissingValue("-ngl"))?;
                n_gpu_layers = value
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("-ngl", value))?;
            }
            "--seed" | "-s" => {
                args.next().ok_or(ParseError::MissingValue("--seed"))?;
            }
            _ => {}
        }
    }

    if model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }

    Ok(Args {
        model_path,
        prompt,
        n_gpu_layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ctest_invocation() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "--prompt",
            "hello",
            "--seed",
            "42",
            "-ngl",
            "0",
        ])
        .unwrap();

        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.n_gpu_layers, 0);
    }

    #[test]
    fn rejects_missing_model() {
        assert_eq!(
            parse_args(["--prompt", "hello"]).unwrap_err(),
            ParseError::MissingModel
        );
    }

    #[test]
    fn rejects_invalid_gpu_layer_count() {
        assert_eq!(
            parse_args(["-m", "model.gguf", "-ngl", "gpu"]).unwrap_err(),
            ParseError::InvalidInteger("-ngl", "gpu".to_string())
        );
    }
}
