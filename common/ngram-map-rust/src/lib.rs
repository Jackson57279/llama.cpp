use std::ffi::c_void;
use std::ptr;

const LCG_FACTOR: u32 = 2_654_435_761;
const MAX_VALUES: usize = 4;
const HASH_MAP_SIZE: usize = 262_144;
const MAX_VALUE_COUNT: u16 = 16_380;

type LlamaToken = i32;

#[derive(Clone, Copy)]
struct MapValue {
    value_idx: usize,
    value_num: u16,
    n_accepted: i16,
}

impl Default for MapValue {
    fn default() -> Self {
        Self {
            value_idx: 0,
            value_num: 0,
            n_accepted: -1,
        }
    }
}

#[derive(Clone)]
struct MapKey {
    key_idx: usize,
    stat_idx: usize,
    key_num: u16,
    values: [MapValue; MAX_VALUES],
}

pub struct NgramMap {
    size_key: u16,
    size_value: u16,
    key_only: bool,
    keys: Vec<MapKey>,
    min_hits: u16,
    size_last_begin: usize,
    last_draft_created: bool,
    last_draft_key_idx: usize,
    last_draft_value_idx: u16,
    idx_last_check: usize,
    key_map: Vec<u32>,
    key_map_last_idx: u32,
}

impl NgramMap {
    fn new(size_key: u16, size_value: u16, key_only: bool, min_hits: u16) -> Self {
        Self {
            size_key,
            size_value,
            key_only,
            keys: Vec::new(),
            min_hits,
            size_last_begin: 0,
            last_draft_created: false,
            last_draft_key_idx: 0,
            last_draft_value_idx: 0,
            idx_last_check: 0,
            key_map: vec![0; HASH_MAP_SIZE],
            key_map_last_idx: 0,
        }
    }
}

#[no_mangle]
pub extern "C" fn common_ngram_map_rust_new(
    size_key: u16,
    size_value: u16,
    key_only: bool,
    min_hits: u16,
) -> *mut c_void {
    Box::into_raw(Box::new(NgramMap::new(
        size_key, size_value, key_only, min_hits,
    ))) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_map_rust_free(ptr: *mut c_void) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr as *mut NgramMap));
    }
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_map_rust_size_value(ptr: *const c_void) -> u16 {
    if ptr.is_null() {
        return 0;
    }
    (*(ptr as *const NgramMap)).size_value
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_map_rust_begin(
    ptr: *mut c_void,
    tokens: *const LlamaToken,
    tokens_len: usize,
) {
    if ptr.is_null() || tokens.is_null() {
        return;
    }
    let map = &mut *(ptr as *mut NgramMap);
    let tokens = std::slice::from_raw_parts(tokens, tokens_len);
    map_begin(map, tokens);
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_map_rust_draft(
    ptr: *mut c_void,
    input: *const LlamaToken,
    input_len: usize,
    sampled: LlamaToken,
    out: *mut LlamaToken,
    out_len: usize,
) -> usize {
    if ptr.is_null() || input.is_null() || out.is_null() {
        return 0;
    }
    let map = &mut *(ptr as *mut NgramMap);
    let input = std::slice::from_raw_parts(input, input_len);
    let mut draft = Vec::new();
    map_draft(map, input, sampled, &mut draft);
    let copied = draft.len().min(out_len);
    ptr::copy_nonoverlapping(draft.as_ptr(), out, copied);
    copied
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_map_rust_accept(ptr: *mut c_void, n_accepted: u16) {
    if ptr.is_null() {
        return;
    }
    map_accept(&mut *(ptr as *mut NgramMap), n_accepted);
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_simple_rust_draft(
    size_ngram: u16,
    size_mgram: u16,
    tokens: *const LlamaToken,
    tokens_len: usize,
    sampled: LlamaToken,
    out: *mut LlamaToken,
    out_len: usize,
) -> usize {
    if tokens.is_null() || out.is_null() {
        return 0;
    }
    let tokens = std::slice::from_raw_parts(tokens, tokens_len);
    let draft = simple_draft(size_ngram as usize, size_mgram as usize, tokens, sampled);
    let copied = draft.len().min(out_len);
    ptr::copy_nonoverlapping(draft.as_ptr(), out, copied);
    copied
}

fn hash(tokens: &[LlamaToken], start: usize, len: usize) -> u32 {
    let mut hash = 0_u32;
    for i in 0..len {
        hash = hash
            .wrapping_mul(LCG_FACTOR)
            .wrapping_add(tokens[start + i] as u32);
    }
    hash
}

fn simple_draft(
    size_ngram: usize,
    size_mgram: usize,
    tokens: &[LlamaToken],
    sampled: LlamaToken,
) -> Vec<LlamaToken> {
    let cur_len = tokens.len();
    if cur_len <= size_ngram + size_mgram + 1 {
        return Vec::new();
    }

    let mut pattern = Vec::with_capacity(size_ngram);
    for j in cur_len - size_ngram + 1..cur_len {
        pattern.push(tokens[j]);
    }
    pattern.push(sampled);

    let mut match_pos = 0;
    for j in (1..cur_len - size_ngram).rev() {
        if pattern.iter().enumerate().all(|(k, token)| tokens[j + k] == *token) {
            match_pos = j;
            break;
        }
    }
    if match_pos == 0 {
        return Vec::new();
    }

    let copy_max = size_mgram.min(cur_len - (match_pos + size_ngram));
    if copy_max < size_ngram {
        return Vec::new();
    }
    tokens[match_pos + size_ngram..match_pos + size_ngram + copy_max].to_vec()
}

fn map_begin(map: &mut NgramMap, tokens: &[LlamaToken]) {
    let size_begin = tokens.len();

    if !map.key_map.is_empty() && size_begin < map.idx_last_check {
        for value in &mut map.key_map {
            if *value as usize >= map.size_last_begin {
                *value = 0;
            }
        }
        map.key_map_last_idx = map.size_last_begin.saturating_sub(1) as u32;
    }

    if size_begin < map.idx_last_check && !map.keys.is_empty() {
        for i in (0..map.keys.len()).rev() {
            if map.keys[i].key_idx >= map.size_last_begin {
                map.keys.remove(i);
                continue;
            }
            if map.key_only {
                continue;
            }
            for j in (0..MAX_VALUES).rev() {
                if map.keys[i].values[j].value_idx >= map.size_last_begin {
                    for k in j..MAX_VALUES - 1 {
                        map.keys[i].values[k] = map.keys[i].values[k + 1];
                    }
                    map.keys[i].values[MAX_VALUES - 1] = MapValue::default();
                }
            }
            if map.keys[i].values[0].value_idx == 0 {
                map.keys.remove(i);
            }
        }
    }

    map.idx_last_check = size_begin;
    map.size_last_begin = size_begin;
}

fn map_draft(
    map: &mut NgramMap,
    input: &[LlamaToken],
    sampled: LlamaToken,
    draft: &mut Vec<LlamaToken>,
) {
    map.last_draft_created = false;
    map.last_draft_key_idx = 0;
    map.last_draft_value_idx = 0;

    let cur_len = input.len();
    let n = map.size_key as usize;
    let m = map.size_value as usize;
    if cur_len < 2 * n + m || cur_len >= u32::MAX as usize || map.idx_last_check > cur_len {
        return;
    }
    map.idx_last_check = cur_len;

    let mut key_tokens = Vec::with_capacity(n);
    for j in cur_len - n + 1..cur_len {
        key_tokens.push(input[j]);
    }
    key_tokens.push(sampled);

    let mut match_pos = 0;
    if map.size_last_begin > cur_len {
        return;
    }
    if !map.key_map.is_empty() {
        let idx_hash = hash(&key_tokens, 0, n) as usize % map.key_map.len();
        let idx_key = map.key_map[idx_hash] as usize;
        if idx_key != 0
            && idx_key < cur_len - n - m - 1
            && (0..n).all(|k| input[idx_key + k] == key_tokens[k])
        {
            match_pos = idx_key;
        }
    }

    if match_pos == 0 && map.size_last_begin > n + m + 1 {
        let upper = map.size_last_begin - n - m - 1;
        let lower = map.key_map_last_idx as usize;
        for j in (lower + 1..=upper).rev() {
            if (0..n).all(|k| input[j + k] == key_tokens[k]) {
                match_pos = j;
                break;
            }
        }
    }
    if match_pos == 0 {
        let upper = cur_len - n - m - 1;
        let lower = map.size_last_begin.max(map.key_map_last_idx as usize);
        if upper > lower {
            for j in (lower + 1..=upper).rev() {
                if (0..n).all(|k| input[j + k] == key_tokens[k]) {
                    match_pos = j;
                    break;
                }
            }
        }
    }

    if !map.key_map.is_empty() {
        if map.size_last_begin > n + m + 1 {
            let upper = map.size_last_begin - n - m - 1;
            let lower = map.key_map_last_idx as usize;
            for j in (lower + 1..=upper).rev() {
                let idx_hash = hash(input, j, n) as usize % map.key_map.len();
                if map.key_map[idx_hash] == 0 {
                    map.key_map[idx_hash] = j as u32;
                }
            }
        }
        let upper = cur_len - n - m - 1;
        let lower = map.size_last_begin.max(map.key_map_last_idx as usize);
        if upper > lower {
            for j in (lower + 1..=upper).rev() {
                let idx_hash = hash(input, j, n) as usize % map.key_map.len();
                if map.key_map[idx_hash] == 0 {
                    map.key_map[idx_hash] = j as u32;
                }
            }
        }
        map.key_map_last_idx = map.key_map_last_idx.max((cur_len - n - m - 1) as u32);
    }

    if match_pos == 0 {
        return;
    }

    let key_offset = map
        .keys
        .iter()
        .position(|key| (0..n).all(|j| input[key.key_idx + j] == key_tokens[j]));
    let key_offset = match key_offset {
        Some(idx) => idx,
        None => {
            let key = MapKey {
                key_idx: match_pos,
                stat_idx: 0,
                key_num: 0,
                values: [MapValue {
                    n_accepted: m as i16,
                    ..MapValue::default()
                }; MAX_VALUES],
            };
            map.keys.push(key);
            map.keys.len() - 1
        }
    };

    let curr_key = &mut map.keys[key_offset];
    curr_key.key_num = curr_key.key_num.saturating_add(1).min(MAX_VALUE_COUNT);

    if map.key_only {
        let n_draft_tokens = m.min(curr_key.values[0].n_accepted.max(0) as usize);
        draft.extend_from_slice(&input[match_pos + n..match_pos + n + n_draft_tokens]);
        map.last_draft_created = true;
        map.last_draft_key_idx = key_offset;
        map.last_draft_value_idx = 0;
        return;
    }

    if curr_key.key_num < map.min_hits {
        return;
    }

    for i in curr_key.stat_idx..=match_pos {
        if !(0..n).all(|k| input[i + k] == key_tokens[k]) {
            continue;
        }
        let idx_begin_value_key = i + n;
        let mut idx_value = None;
        for v in 0..MAX_VALUES {
            let idx_begin_value_v = curr_key.values[v].value_idx;
            if idx_begin_value_v == 0 {
                curr_key.values[v].value_idx = idx_begin_value_key;
                curr_key.values[v].value_num = 0;
                curr_key.values[v].n_accepted = m as i16;
                idx_value = Some(v);
                break;
            }
            if (0..m).all(|j| input[idx_begin_value_key + j] == input[idx_begin_value_v + j]) {
                idx_value = Some(v);
                break;
            }
        }
        if let Some(v) = idx_value {
            curr_key.values[v].value_num =
                curr_key.values[v].value_num.saturating_add(1).min(MAX_VALUE_COUNT);
        }
    }
    curr_key.stat_idx = match_pos;

    let mut max_occur = 0;
    let mut slot_max = 0;
    for v in 0..MAX_VALUES {
        if curr_key.values[v].value_num > max_occur {
            max_occur = curr_key.values[v].value_num;
            slot_max = v;
        }
    }
    let mut sum_occur = 0_u32;
    for v in 0..MAX_VALUES {
        if v != slot_max {
            sum_occur += curr_key.values[v].value_num as u32;
        }
    }
    if sum_occur > 0 && (max_occur as u32) < 2 * sum_occur {
        return;
    }

    let n_draft_tokens = m.min(curr_key.values[slot_max].n_accepted.max(0) as usize);
    draft.extend_from_slice(&input[match_pos + n..match_pos + n + n_draft_tokens]);
    map.last_draft_created = true;
    map.last_draft_key_idx = key_offset;
    map.last_draft_value_idx = slot_max as u16;
}

fn map_accept(map: &mut NgramMap, n_accepted: u16) {
    if !map.last_draft_created {
        return;
    }
    let key_idx = map.last_draft_key_idx;
    let val_idx = map.last_draft_value_idx as usize;
    if let Some(key) = map.keys.get_mut(key_idx) {
        key.values[val_idx].n_accepted = n_accepted as i16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_draft_repeats_prior_suffix() {
        let tokens = [1, 2, 3, 4, 9, 2, 3];
        assert_eq!(simple_draft(2, 2, &tokens, 4), vec![9, 2]);
    }

    #[test]
    fn key_only_map_drafts_after_repeated_key() {
        let mut map = NgramMap::new(2, 2, true, 1);
        let tokens = [1, 2, 7, 8, 3, 4, 7];
        map_begin(&mut map, &tokens);
        let mut draft = Vec::new();
        map_draft(&mut map, &tokens, 8, &mut draft);
        assert_eq!(draft, vec![3, 4]);
    }
}
