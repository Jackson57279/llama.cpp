#![allow(unexpected_cfgs)]

#[cfg(target_arch = "powerpc64")]
use std::ffi::{c_ulong, CStr};

#[cfg(target_arch = "powerpc64")]
const AT_PLATFORM: c_ulong = 15;

#[cfg(target_arch = "powerpc64")]
extern "C" {
    fn getauxval(key: c_ulong) -> c_ulong;
}

#[derive(Clone, Copy, Default)]
struct PowerpcFeatures {
    power_version: i32,
    has_vsx: bool,
}

fn parse_power_version(platform: &str) -> i32 {
    if !platform.starts_with("power") {
        return -1;
    }

    let digits_start = platform
        .char_indices()
        .rev()
        .find_map(|(idx, ch)| if ch.is_ascii_digit() { Some(idx) } else { None });
    let Some(mut start) = digits_start else {
        return -1;
    };

    for (idx, ch) in platform[..start].char_indices().rev() {
        if ch.is_ascii_digit() {
            start = idx;
        } else {
            break;
        }
    }

    platform[start..].parse::<i32>().unwrap_or(-1)
}

#[cfg(target_arch = "powerpc64")]
fn detect_features() -> PowerpcFeatures {
    let platform_ptr = unsafe { getauxval(AT_PLATFORM) } as *const std::ffi::c_char;
    let power_version = if platform_ptr.is_null() {
        -1
    } else {
        let platform = unsafe { CStr::from_ptr(platform_ptr) };
        platform
            .to_str()
            .map(parse_power_version)
            .unwrap_or(-1)
    };

    PowerpcFeatures {
        power_version,
        has_vsx: power_version >= 9,
    }
}

#[cfg(not(target_arch = "powerpc64"))]
fn detect_features() -> PowerpcFeatures {
    PowerpcFeatures {
        power_version: -1,
        has_vsx: false,
    }
}

fn score_for_features(features: PowerpcFeatures) -> i32 {
    let mut score = 1_i32;

    if cfg!(ggml_use_power7) {
        if features.power_version < 7 {
            return 0;
        }
        score += 1 << 1;
    }
    if cfg!(ggml_use_power8) {
        if features.power_version < 8 {
            return 0;
        }
        score += 1 << 2;
    }
    if cfg!(ggml_use_power9) {
        if features.power_version < 9 {
            return 0;
        }
        score += 1 << 3;
    }
    if cfg!(ggml_use_power10) {
        if features.power_version < 10 {
            return 0;
        }
        score += 1 << 4;
    }
    if cfg!(ggml_use_power11) {
        if features.power_version < 11 {
            return 0;
        }
        score += 1 << 5;
    }
    if cfg!(ggml_use_vsx) {
        if !features.has_vsx {
            return 0;
        }
        score += 1 << 6;
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
    fn parses_power_platform_suffix() {
        assert_eq!(parse_power_version("power9"), 9);
        assert_eq!(parse_power_version("power10"), 10);
        assert_eq!(parse_power_version("powerpc"), -1);
        assert_eq!(parse_power_version("POWER10"), -1);
        assert_eq!(parse_power_version("unknown10"), -1);
    }

    #[test]
    fn baseline_score_without_cfg_requirements() {
        assert_eq!(
            score_for_features(PowerpcFeatures {
                power_version: -1,
                has_vsx: false,
            }),
            1
        );
    }

    #[cfg(target_arch = "powerpc64")]
    #[test]
    fn getauxval_is_callable() {
        let _ = detect_features();
    }
}
