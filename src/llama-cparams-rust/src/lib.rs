#![cfg_attr(not(test), no_std)]

pub const LLAMA_MAX_SEQ: usize = 256;

#[no_mangle]
pub extern "C" fn llama_max_parallel_sequences() -> usize {
    LLAMA_MAX_SEQ
}

#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_max_sequence_constant() {
        assert_eq!(llama_max_parallel_sequences(), 256);
    }
}
