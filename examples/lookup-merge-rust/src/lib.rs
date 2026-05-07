use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, ErrorKind, Read, Write};
use std::path::Path;

pub const LLAMA_NGRAM_MAX: usize = 4;
pub const LLAMA_NGRAM_MIN: usize = 1;
pub const LLAMA_NGRAM_STATIC: usize = 2;
pub const LLAMA_TOKEN_NULL: LlamaToken = -1;

pub type LlamaToken = i32;
pub type TokenCount = i32;
pub type NgramCachePart = HashMap<LlamaToken, TokenCount>;
pub type NgramCache = HashMap<Ngram, NgramCachePart>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ngram {
    pub tokens: [LlamaToken; LLAMA_NGRAM_MAX],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    Help,
    MissingPaths,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Help => write!(f, "help requested"),
            ParseError::MissingPaths => write!(f, "missing input or output cache path"),
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub inputs: Vec<String>,
    pub output: String,
}

pub fn parse_args<I, S>(args: I) -> Result<Args, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let paths: Vec<String> = args.into_iter().map(Into::into).collect();
    if paths.iter().any(|arg| arg == "-h" || arg == "--help") {
        return Err(ParseError::Help);
    }
    if paths.len() < 2 {
        return Err(ParseError::MissingPaths);
    }

    let output = paths.last().expect("checked len").clone();
    Ok(Args {
        inputs: paths[..paths.len() - 1].to_vec(),
        output,
    })
}

pub fn merge_cache(target: &mut NgramCache, add: NgramCache) {
    for (ngram, part) in add {
        let merged_part = target.entry(ngram).or_default();
        for (token, count) in part {
            *merged_part.entry(token).or_default() += count;
        }
    }
}

pub fn update_cache(
    cache: &mut NgramCache,
    ngram_min: usize,
    ngram_max: usize,
    input: &[LlamaToken],
    nnew: usize,
) {
    let input_size = input.len();
    for ngram_size in ngram_min..=ngram_max {
        let i_start = input_size.saturating_sub(nnew).max(ngram_size);
        for i in i_start..input_size {
            let ngram = ngram_from_slice(&input[i - ngram_size..i]);
            let token = input[i];
            *cache.entry(ngram).or_default().entry(token).or_default() += 1;
        }
    }
}

fn token_from_combined(input: &[LlamaToken], draft: &[LlamaToken], index: usize) -> LlamaToken {
    if index < input.len() {
        input[index]
    } else {
        draft[1 + index - input.len()]
    }
}

const DRAFT_MIN_SAMPLE_SIZE_LAX: [i32; LLAMA_NGRAM_MAX] = [2, 2, 1, 1];
const DRAFT_MIN_PERCENT_LAX: [i32; LLAMA_NGRAM_MAX] = [66, 50, 50, 50];
const DRAFT_MIN_SAMPLE_SIZE_STRICT: [i32; LLAMA_NGRAM_MAX] = [4, 3, 2, 2];
const DRAFT_MIN_PERCENT_STRICT: [i32; LLAMA_NGRAM_MAX] = [75, 66, 66, 66];

pub fn draft_tokens(
    input: &[LlamaToken],
    draft: &mut Vec<LlamaToken>,
    n_draft: usize,
    ngram_min: usize,
    ngram_max: usize,
    context: &NgramCache,
    dynamic: &NgramCache,
    static_cache: &NgramCache,
) {
    assert_eq!(draft.len(), 1);
    if input.len() < LLAMA_NGRAM_STATIC {
        return;
    }

    while draft.len() - 1 < n_draft {
        let ngram_start_static = input.len() - LLAMA_NGRAM_STATIC + draft.len() - 1;
        let mut ngram_static = Ngram {
            tokens: [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX],
        };
        for j in ngram_start_static..ngram_start_static + LLAMA_NGRAM_STATIC {
            ngram_static.tokens[j - ngram_start_static] = token_from_combined(input, draft, j);
        }
        let part_static = static_cache.get(&ngram_static).cloned().unwrap_or_default();

        let mut ngrams_cd = Vec::new();
        for ngram_size in ngram_min..=ngram_max {
            if ngram_size > input.len() + draft.len() - 1 {
                continue;
            }
            let ngram_start_cd = input.len() - ngram_size + draft.len() - 1;
            let mut ngram_cd = Ngram {
                tokens: [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX],
            };
            for j in ngram_start_cd..ngram_start_cd + ngram_size {
                ngram_cd.tokens[j - ngram_start_cd] = token_from_combined(input, draft, j);
            }
            ngrams_cd.push(ngram_cd);
        }

        let drafted = try_draft_primary(
            context,
            &ngrams_cd,
            &part_static,
            &DRAFT_MIN_SAMPLE_SIZE_LAX,
            &DRAFT_MIN_PERCENT_LAX,
        )
        .or_else(|| {
            try_draft_primary(
                dynamic,
                &ngrams_cd,
                &part_static,
                &DRAFT_MIN_SAMPLE_SIZE_STRICT,
                &DRAFT_MIN_PERCENT_STRICT,
            )
        })
        .or_else(|| try_draft_static(static_cache, ngram_static));

        let Some(token) = drafted else {
            break;
        };
        draft.push(token);
    }
}

fn try_draft_static(cache: &NgramCache, ngram: Ngram) -> Option<LlamaToken> {
    let part = cache.get(&ngram)?;
    let mut max_count = 0;
    let mut sum_count = 0;
    let mut max_token = LLAMA_TOKEN_NULL;

    for (&token, &count) in part {
        if count > max_count {
            max_token = token;
            max_count = count;
        }
        sum_count += count;
    }

    if sum_count < DRAFT_MIN_SAMPLE_SIZE_LAX[LLAMA_NGRAM_STATIC - 1] {
        return None;
    }
    if 100 * max_count < DRAFT_MIN_PERCENT_LAX[LLAMA_NGRAM_STATIC - 1] * sum_count {
        return None;
    }
    Some(max_token)
}

fn try_draft_primary(
    cache: &NgramCache,
    ngrams: &[Ngram],
    part_static: &NgramCachePart,
    min_sample_size: &[i32; LLAMA_NGRAM_MAX],
    min_percent: &[i32; LLAMA_NGRAM_MAX],
) -> Option<LlamaToken> {
    for (i, ngram) in ngrams.iter().enumerate().rev() {
        let Some(part_primary) = cache.get(ngram) else {
            continue;
        };

        let mut max_count_primary = 0;
        let mut max_count_static = 0;
        let mut sum_count_primary = 0;
        let mut max_token = LLAMA_TOKEN_NULL;

        for (&token, &count_primary) in part_primary {
            let count_static = part_static.get(&token).map_or(1, |count| 100 * *count);
            if count_primary * count_static > max_count_primary * max_count_static {
                max_token = token;
                max_count_primary = count_primary;
                max_count_static = count_static;
            }
            sum_count_primary += count_primary;
        }

        if sum_count_primary < min_sample_size[i] {
            continue;
        }
        if 100 * max_count_primary < min_percent[i] * sum_count_primary {
            continue;
        }
        return Some(max_token);
    }

    None
}

pub fn ngram_from_slice(slice: &[LlamaToken]) -> Ngram {
    let mut tokens = [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX];
    for (dst, src) in tokens.iter_mut().zip(slice.iter().copied()) {
        *dst = src;
    }
    Ngram { tokens }
}

pub fn load_cache(path: impl AsRef<Path>) -> io::Result<NgramCache> {
    let mut file = File::open(path)?;
    let mut cache = NgramCache::new();

    loop {
        let Some(ngram) = read_ngram(&mut file)? else {
            break;
        };

        let ntokens = read_i32(&mut file, "missing token count")?;
        if ntokens <= 0 {
            return Err(invalid_data("token count must be positive"));
        }

        let mut part = NgramCachePart::new();
        for _ in 0..ntokens {
            let token = read_i32(&mut file, "missing token")?;
            let count = read_i32(&mut file, "missing count")?;
            if count <= 0 {
                return Err(invalid_data("count must be positive"));
            }
            part.insert(token, count);
        }
        cache.insert(ngram, part);
    }

    Ok(cache)
}

pub fn save_cache(cache: &NgramCache, path: impl AsRef<Path>) -> io::Result<()> {
    let mut file = File::create(path)?;
    for (ngram, part) in cache {
        if part.is_empty() {
            return Err(invalid_data("token count map must not be empty"));
        }

        for token in ngram.tokens {
            file.write_all(&token.to_ne_bytes())?;
        }

        let ntokens =
            i32::try_from(part.len()).map_err(|_| invalid_data("token count map is too large"))?;
        file.write_all(&ntokens.to_ne_bytes())?;

        for (token, count) in part {
            if *count <= 0 {
                return Err(invalid_data("count must be positive"));
            }
            file.write_all(&token.to_ne_bytes())?;
            file.write_all(&count.to_ne_bytes())?;
        }
    }

    Ok(())
}

fn read_ngram(mut reader: impl Read) -> io::Result<Option<Ngram>> {
    let mut bytes = [0_u8; LLAMA_NGRAM_MAX * std::mem::size_of::<i32>()];
    let mut read = 0;
    while read < bytes.len() {
        match reader.read(&mut bytes[read..]) {
            Ok(0) if read == 0 => return Ok(None),
            Ok(0) => return Err(invalid_data("partial ngram record")),
            Ok(n) => read += n,
            Err(err) => return Err(err),
        }
    }

    let mut tokens = [0; LLAMA_NGRAM_MAX];
    for (token, chunk) in tokens.iter_mut().zip(bytes.chunks_exact(4)) {
        *token = i32::from_ne_bytes(chunk.try_into().expect("chunk size"));
    }
    Ok(Some(Ngram { tokens }))
}

fn read_i32(mut reader: impl Read, missing: &'static str) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(|err| {
        if err.kind() == ErrorKind::UnexpectedEof {
            invalid_data(missing)
        } else {
            err
        }
    })?;
    Ok(i32::from_ne_bytes(bytes))
}

fn invalid_data(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;
    use std::process;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

    fn temp_file(name: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!(
            "llama-lookup-merge-{name}-{}-{}.bin",
            process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn sample_cache() -> NgramCache {
        let mut cache = NgramCache::new();
        cache.insert(
            Ngram {
                tokens: [10, 20, -1, -1],
            },
            HashMap::from([(30, 2), (40, 3)]),
        );
        cache.insert(
            Ngram {
                tokens: [5, 6, 7, 8],
            },
            HashMap::from([(9, 1)]),
        );
        cache
    }

    #[test]
    fn parses_input_and_output_paths() {
        let args = parse_args(["a.bin", "b.bin", "out.bin"]).unwrap();

        assert_eq!(args.inputs, ["a.bin", "b.bin"]);
        assert_eq!(args.output, "out.bin");
    }

    #[test]
    fn rejects_missing_paths() {
        assert_eq!(
            parse_args(["only-input.bin"]).unwrap_err(),
            ParseError::MissingPaths
        );
    }

    #[test]
    fn detects_help_flag() {
        assert_eq!(parse_args(["--help"]).unwrap_err(), ParseError::Help);
    }

    #[test]
    fn round_trips_cache_file() {
        let path = temp_file("round-trip");
        let cache = sample_cache();

        save_cache(&cache, &path).unwrap();
        let loaded = load_cache(&path).unwrap();
        fs::remove_file(path).unwrap();

        assert_eq!(loaded, cache);
    }

    #[test]
    fn merges_counts_and_new_parts() {
        let ngram = Ngram {
            tokens: [10, 20, -1, -1],
        };
        let mut target = HashMap::from([(ngram, HashMap::from([(30, 2), (40, 3)]))]);
        let add = HashMap::from([
            (ngram, HashMap::from([(30, 5), (50, 7)])),
            (
                Ngram {
                    tokens: [1, 2, 3, -1],
                },
                HashMap::from([(4, 9)]),
            ),
        ]);

        merge_cache(&mut target, add);

        assert_eq!(target[&ngram][&30], 7);
        assert_eq!(target[&ngram][&40], 3);
        assert_eq!(target[&ngram][&50], 7);
        assert_eq!(
            target[&Ngram {
                tokens: [1, 2, 3, -1]
            }][&4],
            9
        );
    }

    #[test]
    fn updates_multiple_ngram_sizes_for_new_tail() {
        let mut cache = NgramCache::new();
        let tokens = [10, 20, 30, 40];

        update_cache(&mut cache, 1, 2, &tokens, tokens.len());

        assert_eq!(cache[&ngram_from_slice(&[10])][&20], 1);
        assert_eq!(cache[&ngram_from_slice(&[20, 30])][&40], 1);
    }

    #[test]
    fn drafts_from_context_cache_before_static_cache() {
        let mut context = NgramCache::new();
        context.insert(ngram_from_slice(&[2]), HashMap::from([(3, 2)]));
        let mut static_cache = NgramCache::new();
        static_cache.insert(ngram_from_slice(&[1, 2]), HashMap::from([(4, 2)]));
        let mut draft = vec![2];

        draft_tokens(
            &[1, 2],
            &mut draft,
            1,
            LLAMA_NGRAM_MIN,
            LLAMA_NGRAM_MAX,
            &context,
            &NgramCache::new(),
            &static_cache,
        );

        assert_eq!(draft, vec![2, 3]);
    }

    #[test]
    fn rejects_partial_file() {
        let path = temp_file("partial");
        fs::write(&path, [1_u8, 2, 3]).unwrap();
        let err = load_cache(&path).unwrap_err();
        fs::remove_file(path).unwrap();

        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }
}
