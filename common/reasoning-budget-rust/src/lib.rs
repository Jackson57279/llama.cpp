use std::ffi::{c_char, c_void};
use std::ptr;
use std::slice;

type LlamaToken = i32;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningBudgetState {
    Idle = 0,
    Counting = 1,
    Forcing = 2,
    WaitingUtf8 = 3,
    Done = 4,
}

#[repr(C)]
pub struct LlamaVocab {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LlamaSampler {
    iface: *mut LlamaSamplerI,
    ctx: *mut c_void,
}

#[repr(C)]
pub struct LlamaTokenData {
    id: LlamaToken,
    logit: f32,
    p: f32,
}

#[repr(C)]
pub struct LlamaTokenDataArray {
    data: *mut LlamaTokenData,
    size: usize,
    selected: i64,
    sorted: bool,
}

#[repr(C)]
pub struct LlamaSamplerData {
    logits: *mut c_void,
    probs: *mut c_void,
    sampled: *mut c_void,
    candidates: *mut c_void,
}

type NameFn = unsafe extern "C" fn(*const LlamaSampler) -> *const c_char;
type AcceptFn = unsafe extern "C" fn(*mut LlamaSampler, LlamaToken);
type ApplyFn = unsafe extern "C" fn(*mut LlamaSampler, *mut LlamaTokenDataArray);
type ResetFn = unsafe extern "C" fn(*mut LlamaSampler);
type CloneFn = unsafe extern "C" fn(*const LlamaSampler) -> *mut LlamaSampler;
type FreeFn = unsafe extern "C" fn(*mut LlamaSampler);
type BackendInitFn = unsafe extern "C" fn(*mut LlamaSampler, *mut c_void) -> bool;
type BackendAcceptFn = unsafe extern "C" fn(*mut LlamaSampler, *mut c_void, *mut c_void, *mut c_void);
type BackendApplyFn =
    unsafe extern "C" fn(*mut LlamaSampler, *mut c_void, *mut c_void, *mut LlamaSamplerData);
type BackendSetInputFn = unsafe extern "C" fn(*mut LlamaSampler);

#[repr(C)]
pub struct LlamaSamplerI {
    name: Option<NameFn>,
    accept: Option<AcceptFn>,
    apply: Option<ApplyFn>,
    reset: Option<ResetFn>,
    clone: Option<CloneFn>,
    free: Option<FreeFn>,
    backend_init: Option<BackendInitFn>,
    backend_accept: Option<BackendAcceptFn>,
    backend_apply: Option<BackendApplyFn>,
    backend_set_input: Option<BackendSetInputFn>,
}

extern "C" {
    fn llama_sampler_init(iface: *mut LlamaSamplerI, ctx: *mut c_void) -> *mut LlamaSampler;
    fn llama_token_to_piece(
        vocab: *const LlamaVocab,
        token: LlamaToken,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;
}

#[derive(Clone)]
struct TokenMatcher {
    tokens: Vec<LlamaToken>,
    pos: usize,
}

impl TokenMatcher {
    fn advance(&mut self, token: LlamaToken) -> bool {
        if self.tokens.is_empty() {
            return false;
        }

        if token == self.tokens[self.pos] {
            self.pos += 1;
            if self.pos >= self.tokens.len() {
                self.pos = 0;
                return true;
            }
        } else {
            self.pos = 0;
            if token == self.tokens[0] {
                self.pos = 1;
            }
        }
        false
    }

    fn reset(&mut self) {
        self.pos = 0;
    }
}

struct ReasoningBudgetCtx {
    vocab: *const LlamaVocab,
    start_matcher: TokenMatcher,
    end_matcher: TokenMatcher,
    forced_tokens: Vec<LlamaToken>,
    budget: i32,
    remaining: i32,
    state: ReasoningBudgetState,
    force_pos: usize,
}

const NAME: &[u8] = b"reasoning-budget\0";

#[no_mangle]
pub unsafe extern "C" fn llama_common_reasoning_budget_init_rust(
    vocab: *const LlamaVocab,
    start_tokens: *const LlamaToken,
    start_len: usize,
    end_tokens: *const LlamaToken,
    end_len: usize,
    forced_tokens: *const LlamaToken,
    forced_len: usize,
    budget: i32,
    initial_state: ReasoningBudgetState,
) -> *mut LlamaSampler {
    let start = copy_tokens(start_tokens, start_len);
    let end = copy_tokens(end_tokens, end_len);
    let forced = copy_tokens(forced_tokens, forced_len);
    init_state(vocab, start, end, forced, budget, initial_state)
}

#[no_mangle]
pub unsafe extern "C" fn llama_common_reasoning_budget_get_state_rust(
    smpl: *const LlamaSampler,
) -> ReasoningBudgetState {
    if smpl.is_null() {
        return ReasoningBudgetState::Idle;
    }
    let ctx = (*smpl).ctx as *const ReasoningBudgetCtx;
    if ctx.is_null() {
        return ReasoningBudgetState::Idle;
    }
    (*ctx).state
}

unsafe fn copy_tokens(ptr: *const LlamaToken, len: usize) -> Vec<LlamaToken> {
    if ptr.is_null() || len == 0 {
        Vec::new()
    } else {
        slice::from_raw_parts(ptr, len).to_vec()
    }
}

unsafe fn init_state(
    vocab: *const LlamaVocab,
    start_tokens: Vec<LlamaToken>,
    end_tokens: Vec<LlamaToken>,
    forced_tokens: Vec<LlamaToken>,
    budget: i32,
    mut initial_state: ReasoningBudgetState,
) -> *mut LlamaSampler {
    if initial_state == ReasoningBudgetState::Counting && budget <= 0 {
        initial_state = ReasoningBudgetState::Forcing;
    }

    let ctx = Box::new(ReasoningBudgetCtx {
        vocab,
        start_matcher: TokenMatcher {
            tokens: start_tokens,
            pos: 0,
        },
        end_matcher: TokenMatcher {
            tokens: end_tokens,
            pos: 0,
        },
        forced_tokens,
        budget,
        remaining: budget,
        state: initial_state,
        force_pos: 0,
    });

    llama_sampler_init(
        ptr::addr_of_mut!(LLAMA_REASONING_BUDGET_I),
        Box::into_raw(ctx) as *mut c_void,
    )
}

unsafe extern "C" fn reasoning_budget_name(_smpl: *const LlamaSampler) -> *const c_char {
    NAME.as_ptr() as *const c_char
}

unsafe extern "C" fn reasoning_budget_accept(smpl: *mut LlamaSampler, token: LlamaToken) {
    let ctx = &mut *((*smpl).ctx as *mut ReasoningBudgetCtx);

    match ctx.state {
        ReasoningBudgetState::Idle => {
            if ctx.start_matcher.advance(token) {
                ctx.state = ReasoningBudgetState::Counting;
                ctx.remaining = ctx.budget;
                if ctx.remaining <= 0 {
                    ctx.state = ReasoningBudgetState::Forcing;
                    ctx.force_pos = 0;
                }
            }
        }
        ReasoningBudgetState::Counting | ReasoningBudgetState::WaitingUtf8 => {
            if ctx.end_matcher.advance(token) {
                ctx.state = ReasoningBudgetState::Done;
                return;
            }

            let utf8_complete = token_piece_is_utf8_complete(ctx.vocab, token);

            if ctx.state == ReasoningBudgetState::WaitingUtf8 {
                if utf8_complete {
                    ctx.state = ReasoningBudgetState::Forcing;
                    ctx.force_pos = 0;
                    ctx.end_matcher.reset();
                }
            } else {
                ctx.remaining -= 1;
                if ctx.remaining <= 0 {
                    if utf8_complete {
                        ctx.state = ReasoningBudgetState::Forcing;
                        ctx.force_pos = 0;
                        ctx.end_matcher.reset();
                    } else {
                        ctx.state = ReasoningBudgetState::WaitingUtf8;
                        ctx.end_matcher.reset();
                    }
                }
            }
        }
        ReasoningBudgetState::Forcing => {
            ctx.force_pos += 1;
            if ctx.force_pos >= ctx.forced_tokens.len() {
                ctx.state = ReasoningBudgetState::Done;
            }
        }
        ReasoningBudgetState::Done => {
            if ctx.start_matcher.advance(token) {
                ctx.state = ReasoningBudgetState::Counting;
                ctx.remaining = ctx.budget;
                ctx.end_matcher.reset();

                if ctx.remaining <= 0 {
                    ctx.state = ReasoningBudgetState::Forcing;
                    ctx.force_pos = 0;
                }
            }
        }
    }
}

unsafe extern "C" fn reasoning_budget_apply(
    smpl: *mut LlamaSampler,
    cur_p: *mut LlamaTokenDataArray,
) {
    let ctx = &mut *((*smpl).ctx as *mut ReasoningBudgetCtx);
    if ctx.state != ReasoningBudgetState::Forcing || ctx.force_pos >= ctx.forced_tokens.len() {
        return;
    }

    let forced = ctx.forced_tokens[ctx.force_pos];
    let cur = &mut *cur_p;
    let data = slice::from_raw_parts_mut(cur.data, cur.size);
    for item in data {
        if item.id != forced {
            item.logit = f32::NEG_INFINITY;
        }
    }
}

unsafe extern "C" fn reasoning_budget_reset(smpl: *mut LlamaSampler) {
    let ctx = &mut *((*smpl).ctx as *mut ReasoningBudgetCtx);
    ctx.state = ReasoningBudgetState::Idle;
    ctx.remaining = ctx.budget;
    ctx.start_matcher.reset();
    ctx.end_matcher.reset();
    ctx.force_pos = 0;
}

unsafe extern "C" fn reasoning_budget_clone(smpl: *const LlamaSampler) -> *mut LlamaSampler {
    let ctx = &*((*smpl).ctx as *const ReasoningBudgetCtx);
    init_state(
        ctx.vocab,
        ctx.start_matcher.tokens.clone(),
        ctx.end_matcher.tokens.clone(),
        ctx.forced_tokens.clone(),
        ctx.budget,
        ctx.state,
    )
}

unsafe extern "C" fn reasoning_budget_free(smpl: *mut LlamaSampler) {
    let ctx = (*smpl).ctx as *mut ReasoningBudgetCtx;
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

unsafe fn token_piece_is_utf8_complete(vocab: *const LlamaVocab, token: LlamaToken) -> bool {
    if vocab.is_null() {
        return true;
    }

    let mut buf = vec![0u8; 32];
    let n = llama_token_to_piece(
        vocab,
        token,
        buf.as_mut_ptr() as *mut c_char,
        buf.len() as i32,
        0,
        false,
    );
    let len = if n < 0 {
        let needed = (-n) as usize;
        buf.resize(needed, 0);
        let check = llama_token_to_piece(
            vocab,
            token,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as i32,
            0,
            false,
        );
        if check != -n {
            return true;
        }
        needed
    } else {
        n as usize
    };

    common_utf8_is_complete(&buf[..len])
}

fn common_utf8_is_complete(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return true;
    }
    for i in 1..=bytes.len().min(4) {
        let c = bytes[bytes.len() - i];
        if (c & 0xC0) != 0x80 {
            let expected = if c >= 0xF0 {
                4
            } else if c >= 0xE0 {
                3
            } else if c >= 0xC0 {
                2
            } else {
                1
            };
            return i >= expected;
        }
    }
    false
}

static mut LLAMA_REASONING_BUDGET_I: LlamaSamplerI = LlamaSamplerI {
    name: Some(reasoning_budget_name),
    accept: Some(reasoning_budget_accept),
    apply: Some(reasoning_budget_apply),
    reset: Some(reasoning_budget_reset),
    clone: Some(reasoning_budget_clone),
    free: Some(reasoning_budget_free),
    backend_init: None,
    backend_accept: None,
    backend_apply: None,
    backend_set_input: None,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_handles_overlap() {
        let mut matcher = TokenMatcher {
            tokens: vec![1, 2, 1],
            pos: 0,
        };
        assert!(!matcher.advance(1));
        assert!(!matcher.advance(2));
        assert!(matcher.advance(1));
        assert!(!matcher.advance(2));
        assert!(!matcher.advance(1));
        assert_eq!(matcher.pos, 1);
    }

    #[test]
    fn utf8_completion_matches_common_cases() {
        assert!(common_utf8_is_complete(b"hello"));
        assert!(common_utf8_is_complete(b"abc\xC3\xA9"));
        assert!(!common_utf8_is_complete(&[0xC2]));
        assert!(!common_utf8_is_complete(&[0xE2, 0x80]));
        assert!(!common_utf8_is_complete(&[0x80]));
    }
}
