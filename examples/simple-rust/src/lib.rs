pub mod ffi;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub n_gpu_layers: i32,
    pub n_predict: i32,
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
    let mut args = args.into_iter().map(Into::into).peekable();
    let mut model_path = String::new();
    let mut prompt_parts = Vec::new();
    let mut n_gpu_layers = 99;
    let mut n_predict = 32;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" => {
                model_path = args.next().ok_or(ParseError::MissingValue("-m"))?;
            }
            "-n" => {
                let value = args.next().ok_or(ParseError::MissingValue("-n"))?;
                n_predict = value
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("-n", value))?;
            }
            "-ngl" => {
                let value = args.next().ok_or(ParseError::MissingValue("-ngl"))?;
                n_gpu_layers = value
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("-ngl", value))?;
            }
            _ => {
                prompt_parts.push(arg);
                prompt_parts.extend(args);
                break;
            }
        }
    }

    if model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }

    let prompt = if prompt_parts.is_empty() {
        "Hello my name is".to_string()
    } else {
        prompt_parts.join(" ")
    };

    Ok(Args {
        model_path,
        prompt,
        n_gpu_layers,
        n_predict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_model_with_defaults() {
        let args = parse_args(["-m", "model.gguf"]).unwrap();

        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "Hello my name is");
        assert_eq!(args.n_gpu_layers, 99);
        assert_eq!(args.n_predict, 32);
    }

    #[test]
    fn parses_generation_options_and_prompt() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-n",
            "8",
            "-ngl",
            "0",
            "a",
            "short",
            "prompt",
        ])
        .unwrap();

        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "a short prompt");
        assert_eq!(args.n_gpu_layers, 0);
        assert_eq!(args.n_predict, 8);
    }

    #[test]
    fn rejects_missing_model() {
        assert_eq!(
            parse_args(["-n", "8"]).unwrap_err(),
            ParseError::MissingModel
        );
    }

    #[test]
    fn rejects_missing_option_value() {
        assert_eq!(
            parse_args(["-m", "model.gguf", "-n"]).unwrap_err(),
            ParseError::MissingValue("-n")
        );
    }

    #[test]
    fn rejects_invalid_integer() {
        assert_eq!(
            parse_args(["-m", "model.gguf", "-ngl", "gpu"]).unwrap_err(),
            ParseError::InvalidInteger("-ngl", "gpu".to_string())
        );
    }
}
