use std::collections::HashMap;
use std::ffi::CStr;
use std::fs::File;
use std::io::{self, ErrorKind, Read};
use std::os::raw::{c_char, c_void};

const LLAMA_NGRAM_MAX: usize = 4;
const LLAMA_NGRAM_STATIC: usize = 2;
const LLAMA_TOKEN_NULL: LlamaToken = -1;

type LlamaToken = i32;
type TokenCount = i32;
type NgramCachePart = HashMap<LlamaToken, TokenCount>;
type NgramCache = HashMap<Ngram, NgramCachePart>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Ngram {
    tokens: [LlamaToken; LLAMA_NGRAM_MAX],
}

#[no_mangle]
pub extern "C" fn common_ngram_cache_rust_new() -> *mut c_void {
    Box::into_raw(Box::new(NgramCache::new())) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_cache_rust_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr as *mut NgramCache));
    }
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_cache_rust_load(path: *const c_char) -> *mut c_void {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let Ok(path) = CStr::from_ptr(path).to_str() else {
        return std::ptr::null_mut();
    };
    match load_cache(path) {
        Ok(cache) => Box::into_raw(Box::new(cache)) as *mut c_void,
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_cache_rust_update(
    cache: *mut c_void,
    ngram_min: usize,
    ngram_max: usize,
    input: *const LlamaToken,
    input_len: usize,
    nnew: usize,
) {
    if cache.is_null() || input.is_null() {
        return;
    }
    let input = std::slice::from_raw_parts(input, input_len);
    update_cache(&mut *(cache as *mut NgramCache), ngram_min, ngram_max, input, nnew);
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_cache_rust_draft(
    context: *const c_void,
    dynamic: *const c_void,
    static_cache: *const c_void,
    input: *const LlamaToken,
    input_len: usize,
    n_draft: usize,
    ngram_min: usize,
    ngram_max: usize,
    out: *mut LlamaToken,
    out_len: usize,
) -> usize {
    if context.is_null()
        || dynamic.is_null()
        || static_cache.is_null()
        || input.is_null()
        || out.is_null()
        || input_len == 0
    {
        return 0;
    }

    let input = std::slice::from_raw_parts(input, input_len);
    let mut draft = vec![input[input_len - 1]];
    draft_tokens(
        input,
        &mut draft,
        n_draft.min(out_len),
        ngram_min,
        ngram_max,
        &*(context as *const NgramCache),
        &*(dynamic as *const NgramCache),
        &*(static_cache as *const NgramCache),
    );

    let drafted = draft.len().saturating_sub(1).min(out_len);
    std::ptr::copy_nonoverlapping(draft[1..].as_ptr(), out, drafted);
    drafted
}

fn update_cache(
    cache: &mut NgramCache,
    ngram_min: usize,
    ngram_max: usize,
    input: &[LlamaToken],
    nnew: usize,
) {
    let input_size = input.len();
    for ngram_size in ngram_min..=ngram_max.min(LLAMA_NGRAM_MAX) {
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

fn draft_tokens(
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
        for ngram_size in ngram_min..=ngram_max.min(LLAMA_NGRAM_MAX) {
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

fn ngram_from_slice(slice: &[LlamaToken]) -> Ngram {
    let mut tokens = [LLAMA_TOKEN_NULL; LLAMA_NGRAM_MAX];
    for (dst, src) in tokens.iter_mut().zip(slice.iter().copied()) {
        *dst = src;
    }
    Ngram { tokens }
}

fn load_cache(path: &str) -> io::Result<NgramCache> {
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

    #[test]
    fn updates_and_drafts_from_context_cache() {
        let mut context = NgramCache::new();
        let dynamic = NgramCache::new();
        let mut static_cache = NgramCache::new();
        let input = [1, 2, 3, 1, 2, 3, 1, 2];

        update_cache(&mut context, 1, 4, &input, input.len());
        static_cache.insert(
            Ngram { tokens: [1, 2, -1, -1] },
            HashMap::from([(3, 2)]),
        );

        let mut draft = vec![2];
        draft_tokens(&input, &mut draft, 1, 1, 4, &context, &dynamic, &static_cache);
        assert_eq!(draft, vec![2, 3]);
    }
}
