use llama_lookup_merge::{LlamaToken, Ngram, NgramCache};

const LLAMA_TOKEN_NULL: LlamaToken = -1;
const LLAMA_NGRAM_STATIC: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub model_path: String,
    pub prompt: String,
    pub cache_path: String,
    pub n_gpu_layers: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingModel,
    MissingCachePath,
    MissingValue(&'static str),
    InvalidInteger(&'static str, String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::MissingModel => write!(f, "missing required -m model.gguf"),
            ParseError::MissingCachePath => {
                write!(f, "missing required --lookup-cache-static path")
            }
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
        prompt: String::new(),
        cache_path: String::new(),
        n_gpu_layers: 99,
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-m" | "--model" => {
                parsed.model_path = args.next().ok_or(ParseError::MissingValue("-m"))?;
            }
            "-p" | "--prompt" => {
                parsed.prompt = args.next().ok_or(ParseError::MissingValue("-p"))?;
            }
            "-lcs" | "--lookup-cache-static" => {
                parsed.cache_path = args.next().ok_or(ParseError::MissingValue("-lcs"))?;
            }
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
    if parsed.cache_path.is_empty() {
        return Err(ParseError::MissingCachePath);
    }
    Ok(parsed)
}

pub fn build_static_ngram_cache(tokens: &[LlamaToken]) -> NgramCache {
    let mut cache = NgramCache::new();
    for i in LLAMA_NGRAM_STATIC..tokens.len() {
        let ngram = Ngram {
            tokens: [
                tokens[i - 2],
                tokens[i - 1],
                LLAMA_TOKEN_NULL,
                LLAMA_TOKEN_NULL,
            ],
        };
        let token = tokens[i];
        *cache.entry(ngram).or_default().entry(token).or_default() += 1;
    }
    cache
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lookup_create_args() {
        let args = parse_args([
            "-m",
            "model.gguf",
            "-p",
            "hello",
            "-lcs",
            "cache.bin",
            "-ngl",
            "0",
        ])
        .unwrap();

        assert_eq!(args.model_path, "model.gguf");
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.cache_path, "cache.bin");
        assert_eq!(args.n_gpu_layers, 0);
    }

    #[test]
    fn builds_static_two_gram_cache() {
        let cache = build_static_ngram_cache(&[1, 2, 3, 1, 2, 4]);

        assert_eq!(
            cache[&Ngram {
                tokens: [1, 2, -1, -1]
            }][&3],
            1
        );
        assert_eq!(
            cache[&Ngram {
                tokens: [1, 2, -1, -1]
            }][&4],
            1
        );
        assert_eq!(
            cache[&Ngram {
                tokens: [2, 3, -1, -1]
            }][&1],
            1
        );
    }

    #[test]
    fn rejects_missing_cache_path() {
        assert_eq!(
            parse_args(["-m", "model.gguf"]).unwrap_err(),
            ParseError::MissingCachePath
        );
    }
}
