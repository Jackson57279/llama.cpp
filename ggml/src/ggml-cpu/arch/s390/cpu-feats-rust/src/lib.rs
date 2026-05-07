#![allow(unexpected_cfgs)]

#[cfg(target_arch = "s390x")]
use std::ffi::c_ulong;

#[cfg(target_arch = "s390x")]
const AT_HWCAP: c_ulong = 16;
#[cfg(target_arch = "s390x")]
const HWCAP_VXRS_EXT2: c_ulong = 1 << 15;
#[cfg(target_arch = "s390x")]
const HWCAP_NNPA: c_ulong = 1 << 20;

#[cfg(target_arch = "s390x")]
extern "C" {
    fn getauxval(key: c_ulong) -> c_ulong;
}

#[derive(Clone, Copy, Default)]
struct S390Features {
    has_vxe2: bool,
    has_nnpa: bool,
}

#[cfg(target_arch = "s390x")]
fn detect_features() -> S390Features {
    let hwcap = unsafe { getauxval(AT_HWCAP) };
    S390Features {
        has_vxe2: (hwcap & HWCAP_VXRS_EXT2) != 0,
        has_nnpa: (hwcap & HWCAP_NNPA) != 0,
    }
}

#[cfg(not(target_arch = "s390x"))]
fn detect_features() -> S390Features {
    S390Features::default()
}

fn score_for_features(features: S390Features) -> i32 {
    let mut score = 1_i32;

    if cfg!(ggml_use_vxe2) {
        if !features.has_vxe2 {
            return 0;
        }
        score += 1 << 1;
    }

    if cfg!(ggml_use_nnpa) {
        if !features.has_nnpa {
            return 0;
        }
        score += 1 << 2;
    }

    score
}

#[no_mangle]
pub extern "C" fn ggml_backend_score() -> i32 {
    score_for_features(detect_features())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_score_without_cfg_requirements() {
        assert_eq!(score_for_features(S390Features::default()), 1);
        assert_eq!(
            score_for_features(S390Features {
                has_vxe2: true,
                has_nnpa: true,
            }),
            1
        );
    }

    #[cfg(target_arch = "s390x")]
    #[test]
    fn getauxval_is_callable() {
        let _ = detect_features();
    }
}
