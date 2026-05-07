#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub n_ctx: i32,
    pub n_gpu_layers: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingModel,
    MissingValue(&'static str),
    InvalidInteger(&'static str, String),
    UnexpectedArgument(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m model.gguf"),
            ParseError::MissingValue(flag) => write!(f, "missing value for {flag}"),
            ParseError::InvalidInteger(flag, value) => {
                write!(f, "invalid integer for {flag}: {value}")
            }
            ParseError::UnexpectedArgument(value) => write!(f, "unexpected argument: {value}"),
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
    let mut n_ctx = 2048;
    let mut n_gpu_layers = 99;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" => {
                model_path = args.next().ok_or(ParseError::MissingValue("-m"))?;
            }
            "-c" => {
                let value = args.next().ok_or(ParseError::MissingValue("-c"))?;
                n_ctx = value
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("-c", value))?;
            }
            "-ngl" => {
                let value = args.next().ok_or(ParseError::MissingValue("-ngl"))?;
                n_gpu_layers = value
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("-ngl", value))?;
            }
            _ => return Err(ParseError::UnexpectedArgument(arg)),
        }
    }

    if model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }

    Ok(Args {
        model_path,
        n_ctx,
        n_gpu_layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_defaults() {
        let args = parse_args(["-m", "model.gguf"]).unwrap();

        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.n_ctx, 2048);
        assert_eq!(args.n_gpu_layers, 99);
    }

    #[test]
    fn parses_options() {
        let args = parse_args(["-m", "model.gguf", "-c", "512", "-ngl", "0"]).unwrap();

        assert_eq!(args.n_ctx, 512);
        assert_eq!(args.n_gpu_layers, 0);
    }

    #[test]
    fn rejects_missing_model() {
        assert_eq!(
            parse_args(["-c", "512"]).unwrap_err(),
            ParseError::MissingModel
        );
    }

    #[test]
    fn rejects_unexpected_argument() {
        assert_eq!(
            parse_args(["-m", "model.gguf", "prompt"]).unwrap_err(),
            ParseError::UnexpectedArgument("prompt".to_string())
        );
    }
}
