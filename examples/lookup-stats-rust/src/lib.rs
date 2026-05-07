#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub lookup_cache_static: Option<String>,
    pub lookup_cache_dynamic: Option<String>,
    pub n_gpu_layers: i32,
    pub n_draft: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            prompt: String::new(),
            lookup_cache_static: None,
            lookup_cache_dynamic: None,
            n_gpu_layers: 99,
            n_draft: 16,
        }
    }
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
    let mut parsed = Args::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => parsed.model_path = value(&mut args, "-m")?,
            "-p" | "--prompt" => parsed.prompt = value(&mut args, "-p")?,
            "-lcs" | "--lookup-cache-static" => {
                parsed.lookup_cache_static = Some(value(&mut args, "-lcs")?);
            }
            "-lcd" | "--lookup-cache-dynamic" => {
                parsed.lookup_cache_dynamic = Some(value(&mut args, "-lcd")?);
            }
            "-ngl" | "--n-gpu-layers" => {
                let raw = value(&mut args, "-ngl")?;
                parsed.n_gpu_layers = raw
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("-ngl", raw))?;
            }
            "--spec-draft-n-max" => {
                let raw = value(&mut args, "--spec-draft-n-max")?;
                parsed.n_draft = raw
                    .parse()
                    .map_err(|_| ParseError::InvalidInteger("--spec-draft-n-max", raw))?;
            }
            _ => {}
        }
    }

    if parsed.model_path.is_empty() {
        return Err(ParseError::MissingModel);
    }
    Ok(parsed)
}

fn value<I>(args: &mut I, flag: &'static str) -> Result<String, ParseError>
where
    I: Iterator<Item = String>,
{
    args.next().ok_or(ParseError::MissingValue(flag))
}

pub fn compute_acceptance(n_accept: i32, n_drafted: i32) -> Option<f32> {
    if n_drafted <= 0 {
        None
    } else {
        Some(100.0 * n_accept as f32 / n_drafted as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lookup_stats_options() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-p",
            "hello",
            "-lcs",
            "static.bin",
            "-lcd",
            "dynamic.bin",
            "-ngl",
            "0",
            "--spec-draft-n-max",
            "8",
        ])
        .unwrap();

        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.lookup_cache_static, Some("static.bin".to_string()));
        assert_eq!(args.lookup_cache_dynamic, Some("dynamic.bin".to_string()));
        assert_eq!(args.n_gpu_layers, 0);
        assert_eq!(args.n_draft, 8);
    }

    #[test]
    fn rejects_missing_model() {
        assert_eq!(
            parse_args(["-p", "hello"]).unwrap_err(),
            ParseError::MissingModel
        );
    }

    #[test]
    fn avoids_division_by_zero_for_acceptance() {
        assert_eq!(compute_acceptance(0, 0), None);
        assert_eq!(compute_acceptance(1, 4), Some(25.0));
    }
}
