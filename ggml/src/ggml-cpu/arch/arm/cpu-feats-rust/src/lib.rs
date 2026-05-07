#![allow(unexpected_cfgs)]

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
use std::ffi::c_ulong;
#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
use std::ffi::{c_char, c_int, c_void};

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const AT_HWCAP: c_ulong = 16;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const AT_HWCAP2: c_ulong = 26;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const HWCAP_FPHP: c_ulong = 1 << 9;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const HWCAP_ASIMDDP: c_ulong = 1 << 20;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const HWCAP_SVE: c_ulong = 1 << 22;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const HWCAP2_SVE2: c_ulong = 1 << 1;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const HWCAP2_I8MM: c_ulong = 1 << 13;
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const HWCAP2_SME: c_ulong = 1 << 23;

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
extern "C" {
    fn getauxval(key: c_ulong) -> c_ulong;
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
extern "C" {
    fn sysctlbyname(
        name: *const c_char,
        oldp: *mut c_void,
        oldlenp: *mut usize,
        newp: *mut c_void,
        newlen: usize,
    ) -> c_int;
}

#[derive(Clone, Copy, Default)]
struct ArmFeatures {
    has_dotprod: bool,
    has_fp16_va: bool,
    has_sve: bool,
    has_sve2: bool,
    has_i8mm: bool,
    has_sme: bool,
}

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
fn detect_features() -> ArmFeatures {
    let hwcap = unsafe { getauxval(AT_HWCAP) };
    let hwcap2 = unsafe { getauxval(AT_HWCAP2) };
    ArmFeatures {
        has_dotprod: (hwcap & HWCAP_ASIMDDP) != 0,
        has_fp16_va: (hwcap & HWCAP_FPHP) != 0,
        has_sve: (hwcap & HWCAP_SVE) != 0,
        has_sve2: (hwcap2 & HWCAP2_SVE2) != 0,
        has_i8mm: (hwcap2 & HWCAP2_I8MM) != 0,
        has_sme: (hwcap2 & HWCAP2_SME) != 0,
    }
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
fn apple_sysctl_bool(name: *const c_char) -> bool {
    let mut oldp = 0_i32;
    let mut size = std::mem::size_of_val(&oldp);
    let ret = unsafe {
        sysctlbyname(
            name,
            &mut oldp as *mut i32 as *mut c_void,
            &mut size as *mut usize,
            std::ptr::null_mut(),
            0,
        )
    };
    ret == 0 && oldp != 0
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
fn detect_features() -> ArmFeatures {
    ArmFeatures {
        has_dotprod: apple_sysctl_bool(c"hw.optional.arm.FEAT_DotProd".as_ptr()),
        has_fp16_va: false,
        has_sve: false,
        has_sve2: false,
        has_i8mm: apple_sysctl_bool(c"hw.optional.arm.FEAT_I8MM".as_ptr()),
        has_sme: apple_sysctl_bool(c"hw.optional.arm.FEAT_SME".as_ptr()),
    }
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_os = "linux"),
    all(target_arch = "aarch64", target_vendor = "apple")
)))]
#[allow(dead_code)]
fn detect_features() -> ArmFeatures {
    ArmFeatures::default()
}

fn score_for_features(features: ArmFeatures) -> i32 {
    let mut score = 1_i32;

    if cfg!(ggml_use_dotprod) {
        if !features.has_dotprod {
            return 0;
        }
        score += 1 << 1;
    }
    if cfg!(ggml_use_fp16_va) {
        if !features.has_fp16_va {
            return 0;
        }
        score += 1 << 2;
    }
    if cfg!(ggml_use_sve) {
        if !features.has_sve {
            return 0;
        }
        score += 1 << 3;
    }
    if cfg!(ggml_use_matmul_int8) {
        if !features.has_i8mm {
            return 0;
        }
        score += 1 << 4;
    }
    if cfg!(ggml_use_sve2) {
        if !features.has_sve2 {
            return 0;
        }
        score += 1 << 5;
    }
    if cfg!(ggml_use_sme) {
        if !features.has_sme {
            return 0;
        }
        score += 1 << 6;
    }

    score
}

#[cfg(target_arch = "aarch64")]
#[no_mangle]
pub extern "C" fn ggml_backend_score() -> i32 {
    score_for_features(detect_features())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_score_without_cfg_requirements() {
        assert_eq!(score_for_features(ArmFeatures::default()), 1);
        assert_eq!(
            score_for_features(ArmFeatures {
                has_dotprod: true,
                has_fp16_va: true,
                has_sve: true,
                has_sve2: true,
                has_i8mm: true,
                has_sme: true,
            }),
            1
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn feature_probe_is_callable() {
        let _ = detect_features();
    }
}
