#![allow(unexpected_cfgs)]

#[cfg(target_arch = "x86")]
use std::arch::x86::__cpuid_count;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__cpuid_count;

#[derive(Default)]
struct CpuId {
    f_1_ecx: u32,
    f_7_ebx: u32,
    f_7_ecx: u32,
    f_7_edx: u32,
    f_7_1_eax: u32,
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl CpuId {
    fn new() -> Self {
        let leaf0 = __cpuid_count(0, 0);
        let n_ids = leaf0.eax;
        let mut this = Self::default();

        if n_ids >= 1 {
            let leaf1 = __cpuid_count(1, 0);
            this.f_1_ecx = leaf1.ecx;
        }

        if n_ids >= 7 {
            let leaf7 = __cpuid_count(7, 0);
            this.f_7_ebx = leaf7.ebx;
            this.f_7_ecx = leaf7.ecx;
            this.f_7_edx = leaf7.edx;
            this.f_7_1_eax = __cpuid_count(7, 1).eax;
        }

        this
    }

    fn bit(reg: u32, bit: u32) -> bool {
        (reg & (1_u32 << bit)) != 0
    }

    fn fma(&self) -> bool {
        Self::bit(self.f_1_ecx, 12)
    }

    fn f16c(&self) -> bool {
        Self::bit(self.f_1_ecx, 29)
    }

    fn sse42(&self) -> bool {
        Self::bit(self.f_1_ecx, 20)
    }

    fn bmi2(&self) -> bool {
        Self::bit(self.f_7_ebx, 8)
    }

    fn avx(&self) -> bool {
        Self::bit(self.f_1_ecx, 28)
    }

    fn avx2(&self) -> bool {
        Self::bit(self.f_7_ebx, 5)
    }

    fn avx_vnni(&self) -> bool {
        Self::bit(self.f_7_1_eax, 4)
    }

    fn avx512f(&self) -> bool {
        Self::bit(self.f_7_ebx, 16)
    }

    fn avx512dq(&self) -> bool {
        Self::bit(self.f_7_ebx, 17)
    }

    fn avx512cd(&self) -> bool {
        Self::bit(self.f_7_ebx, 28)
    }

    fn avx512bw(&self) -> bool {
        Self::bit(self.f_7_ebx, 30)
    }

    fn avx512vl(&self) -> bool {
        Self::bit(self.f_7_ebx, 31)
    }

    fn avx512_vbmi(&self) -> bool {
        Self::bit(self.f_7_ecx, 1)
    }

    fn avx512_vnni(&self) -> bool {
        Self::bit(self.f_7_ecx, 11)
    }

    fn avx512_bf16(&self) -> bool {
        Self::bit(self.f_7_1_eax, 5)
    }

    fn amx_int8(&self) -> bool {
        Self::bit(self.f_7_edx, 25)
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
impl CpuId {
    fn new() -> Self {
        Self::default()
    }

    fn fma(&self) -> bool { false }
    fn f16c(&self) -> bool { false }
    fn sse42(&self) -> bool { false }
    fn bmi2(&self) -> bool { false }
    fn avx(&self) -> bool { false }
    fn avx2(&self) -> bool { false }
    fn avx_vnni(&self) -> bool { false }
    fn avx512f(&self) -> bool { false }
    fn avx512dq(&self) -> bool { false }
    fn avx512cd(&self) -> bool { false }
    fn avx512bw(&self) -> bool { false }
    fn avx512vl(&self) -> bool { false }
    fn avx512_vbmi(&self) -> bool { false }
    fn avx512_vnni(&self) -> bool { false }
    fn avx512_bf16(&self) -> bool { false }
    fn amx_int8(&self) -> bool { false }
}

fn score_for_cpu(cpu: &CpuId) -> i32 {
    let mut score = 1_i32;

    if cfg!(ggml_fma) {
        if !cpu.fma() {
            return 0;
        }
        score += 1;
    }
    if cfg!(ggml_f16c) {
        if !cpu.f16c() {
            return 0;
        }
        score += 1 << 1;
    }
    if cfg!(ggml_sse42) {
        if !cpu.sse42() {
            return 0;
        }
        score += 1 << 2;
    }
    if cfg!(ggml_bmi2) {
        if !cpu.bmi2() {
            return 0;
        }
        score += 1 << 3;
    }
    if cfg!(ggml_avx) {
        if !cpu.avx() {
            return 0;
        }
        score += 1 << 4;
    }
    if cfg!(ggml_avx2) {
        if !cpu.avx2() {
            return 0;
        }
        score += 1 << 5;
    }
    if cfg!(ggml_avx_vnni) {
        if !cpu.avx_vnni() {
            return 0;
        }
        score += 1 << 6;
    }
    if cfg!(ggml_avx512) {
        if !cpu.avx512f() || !cpu.avx512cd() || !cpu.avx512vl() || !cpu.avx512dq() || !cpu.avx512bw() {
            return 0;
        }
        score += 1 << 7;
    }
    if cfg!(ggml_avx512_vbmi) {
        if !cpu.avx512_vbmi() {
            return 0;
        }
        score += 1 << 8;
    }
    if cfg!(ggml_avx512_bf16) {
        if !cpu.avx512_bf16() {
            return 0;
        }
        score += 1 << 9;
    }
    if cfg!(ggml_avx512_vnni) {
        if !cpu.avx512_vnni() {
            return 0;
        }
        score += 1 << 10;
    }
    if cfg!(ggml_amx_int8) {
        if !cpu.amx_int8() {
            return 0;
        }
        score += 1 << 11;
    }

    score
}

#[no_mangle]
pub extern "C" fn ggml_backend_score() -> i32 {
    score_for_cpu(&CpuId::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_is_at_least_baseline_without_cfg_requirements() {
        assert!(score_for_cpu(&CpuId::default()) >= 1);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn reads_vendor_flags_on_x86() {
        let cpu = CpuId::new();
        assert!(cpu.f_1_ecx != 0 || cpu.f_7_ebx != 0 || cpu.f_7_ecx != 0 || cpu.f_7_edx != 0);
    }
}
