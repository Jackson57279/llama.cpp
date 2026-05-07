#![allow(unexpected_cfgs)]

#[cfg(target_arch = "riscv64")]
use std::ffi::{c_int, c_long, c_void};

#[cfg(target_arch = "riscv64")]
#[repr(C)]
struct RiscvHwprobe {
    key: i64,
    value: u64,
}

#[cfg(target_arch = "riscv64")]
const NR_RISCV_HWPROBE: c_long = 258;
#[cfg(target_arch = "riscv64")]
const RISCV_HWPROBE_KEY_IMA_EXT_0: i64 = 4;
#[cfg(target_arch = "riscv64")]
const RISCV_HWPROBE_IMA_V: u64 = 1 << 2;

#[cfg(target_arch = "riscv64")]
extern "C" {
    fn syscall(num: c_long, ...) -> c_long;
}

#[cfg(target_arch = "riscv64")]
fn has_rvv() -> bool {
    let mut probe = RiscvHwprobe {
        key: RISCV_HWPROBE_KEY_IMA_EXT_0,
        value: 0,
    };

    let ret = unsafe {
        syscall(
            NR_RISCV_HWPROBE,
            &mut probe as *mut RiscvHwprobe,
            1_usize,
            0_usize,
            std::ptr::null::<c_void>(),
            0_usize,
        )
    } as c_int;

    ret == 0 && (probe.value & RISCV_HWPROBE_IMA_V) != 0
}

#[cfg(not(target_arch = "riscv64"))]
fn has_rvv() -> bool {
    false
}

fn score_for_features(has_rvv: bool) -> i32 {
    let mut score = 1_i32;

    if cfg!(ggml_use_rvv) {
        if !has_rvv {
            return 0;
        }
        score += 1 << 1;
    }

    score
}

#[no_mangle]
pub extern "C" fn ggml_backend_score() -> i32 {
    score_for_features(has_rvv())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_score_without_cfg_requirement() {
        assert_eq!(score_for_features(false), 1);
        assert_eq!(score_for_features(true), 1);
    }

    #[cfg(target_arch = "riscv64")]
    #[test]
    fn hwprobe_is_callable() {
        let _ = has_rvv();
    }
}
