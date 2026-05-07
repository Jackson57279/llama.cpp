use std::slice;

pub const EMPTY: i32 = -1;
const HASH_MULTIPLIER: usize = 6_364_136_223_846_793_005;

pub struct NgramMod {
    n: usize,
    used: usize,
    entries: Vec<i32>,
}

impl NgramMod {
    fn new(n: u16, size: usize) -> Self {
        Self {
            n: n as usize,
            used: 0,
            entries: vec![EMPTY; size],
        }
    }

    fn idx(&self, tokens: &[i32]) -> usize {
        let mut result = 0_usize;
        for token in &tokens[..self.n] {
            result = result
                .wrapping_mul(HASH_MULTIPLIER)
                .wrapping_add(*token as usize);
        }
        result % self.entries.len()
    }

    fn add(&mut self, tokens: &[i32]) {
        let idx = self.idx(tokens);
        if self.entries[idx] == EMPTY {
            self.used += 1;
        }
        self.entries[idx] = tokens[self.n];
    }

    fn get(&self, tokens: &[i32]) -> i32 {
        self.entries[self.idx(tokens)]
    }

    fn reset(&mut self) {
        self.entries.fill(EMPTY);
        self.used = 0;
    }
}

#[no_mangle]
pub extern "C" fn common_ngram_mod_rust_new(n: u16, size: usize) -> *mut NgramMod {
    Box::into_raw(Box::new(NgramMod::new(n, size)))
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_mod_rust_free(ptr: *mut NgramMod) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_mod_rust_idx(
    ptr: *const NgramMod,
    tokens: *const i32,
) -> usize {
    let state = &*ptr;
    state.idx(slice::from_raw_parts(tokens, state.n + 1))
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_mod_rust_add(ptr: *mut NgramMod, tokens: *const i32) {
    let state = &mut *ptr;
    state.add(slice::from_raw_parts(tokens, state.n + 1));
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_mod_rust_get(
    ptr: *const NgramMod,
    tokens: *const i32,
) -> i32 {
    let state = &*ptr;
    state.get(slice::from_raw_parts(tokens, state.n + 1))
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_mod_rust_reset(ptr: *mut NgramMod) {
    (&mut *ptr).reset();
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_mod_rust_get_n(ptr: *const NgramMod) -> usize {
    (&*ptr).n
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_mod_rust_get_used(ptr: *const NgramMod) -> usize {
    (&*ptr).used
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_mod_rust_size(ptr: *const NgramMod) -> usize {
    (&*ptr).entries.len()
}

#[no_mangle]
pub unsafe extern "C" fn common_ngram_mod_rust_size_bytes(ptr: *const NgramMod) -> usize {
    (&*ptr).entries.len() * std::mem::size_of::<i32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_gets_and_resets() {
        let mut ngram = NgramMod::new(2, 1024);
        let tokens = [11, 22, 33];

        assert_eq!(ngram.get(&tokens), EMPTY);
        ngram.add(&tokens);
        assert_eq!(ngram.get(&tokens), 33);
        assert_eq!(ngram.used, 1);

        ngram.reset();
        assert_eq!(ngram.get(&tokens), EMPTY);
        assert_eq!(ngram.used, 0);
    }

    #[test]
    fn replacing_same_slot_does_not_increase_used() {
        let mut ngram = NgramMod::new(1, 16);
        ngram.add(&[7, 8]);
        ngram.add(&[7, 9]);

        assert_eq!(ngram.get(&[7, 0]), 9);
        assert_eq!(ngram.used, 1);
    }
}
