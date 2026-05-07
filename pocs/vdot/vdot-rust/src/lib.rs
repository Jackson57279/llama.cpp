pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK8_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    pub d: u16,
    pub qs: [u8; QK4_0 / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_1 {
    pub d: u16,
    pub m: u16,
    pub qs: [u8; QK4_1 / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: u16,
    pub qs: [i8; QK8_0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_1 {
    pub d: u16,
    pub s: u16,
    pub qs: [i8; QK8_0],
}

pub struct Rng {
    state: u64,
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    pub fn gaussian(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let u1 = 1.0 - (self.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        let u2 = (self.next_u32() as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let phi = std::f64::consts::TAU * u2;
        self.spare = Some((r * phi.sin()) as f32);
        (r * phi.cos()) as f32
    }
}

pub fn fill_random_gaussian(values: &mut [f32], rng: &mut Rng, mean: f32) {
    for value in values {
        *value = mean + rng.gaussian();
    }
}

pub fn dot_q4_0_f32(blocks: &[BlockQ4_0], y: &[f32]) -> f64 {
    const VALUES: [f32; 16] = [
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    ];
    let mut sum = 0.0_f64;
    for (block, y) in blocks.iter().zip(y.chunks_exact(QK4_0)) {
        let mut s = 0.0_f32;
        for (j, byte) in block.qs.iter().enumerate() {
            s += y[j] * VALUES[(byte & 0x0f) as usize];
            s += y[j + QK4_0 / 2] * VALUES[(byte >> 4) as usize];
        }
        sum += f64::from(s * f16_to_f32(block.d));
    }
    sum
}

pub fn dot_q4_1_f32(blocks: &[BlockQ4_1], y: &[f32]) -> f64 {
    let mut sum = 0.0_f64;
    for (block, y) in blocks.iter().zip(y.chunks_exact(QK4_1)) {
        let mut s = 0.0_f32;
        let mut s1 = 0.0_f32;
        for (j, byte) in block.qs.iter().enumerate() {
            s += y[j] * f32::from(byte & 0x0f);
            s += y[j + QK4_1 / 2] * f32::from(byte >> 4);
            s1 += y[j] + y[j + QK4_1 / 2];
        }
        sum += f64::from(s * f16_to_f32(block.d) + s1 * f16_to_f32(block.m));
    }
    sum
}

pub fn quantize_row_q8_0_reference(x: &[f32], y: &mut [BlockQ8_0]) {
    assert_eq!(x.len() % QK8_0, 0);
    assert!(y.len() >= x.len() / QK8_0);
    for (block, values) in y.iter_mut().zip(x.chunks_exact(QK8_0)) {
        let amax = values
            .iter()
            .fold(0.0_f32, |acc, value| acc.max(value.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        block.d = f32_to_f16(d);
        for (q, value) in block.qs.iter_mut().zip(values) {
            *q = (value * id).round() as i8;
        }
    }
}

pub fn dot_q4_0_q8_0(n: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let nb = n / QK8_0;
    let mut sumf = 0.0_f32;
    for i in 0..nb {
        let mut sumi = 0_i32;
        for j in 0..QK8_0 / 2 {
            let v0 = x[i].qs[j];
            let i0 = i32::from(v0 & 0x0f) - 8;
            let i1 = i32::from(v0 >> 4) - 8;
            let i2 = i32::from(y[i].qs[j]);
            let i3 = i32::from(y[i].qs[j + QK8_0 / 2]);
            sumi += i0 * i2 + i1 * i3;
        }
        sumf += f16_to_f32(x[i].d) * f16_to_f32(y[i].d) * sumi as f32;
    }
    sumf
}

pub fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exp = (value & 0x7c00) >> 10;
    let frac = (value & 0x03ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut frac = frac;
            let mut exp = -14_i32;
            while (frac & 0x0400) == 0 {
                frac <<= 1;
                exp -= 1;
            }
            frac &= 0x03ff;
            sign | (((exp + 127) as u32) << 23) | (frac << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | (((i32::from(exp) - 15 + 127) as u32) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 255 {
        return sign | if frac == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        sign | 0x7c00
    } else if half_exp <= 0 {
        if half_exp < -10 {
            sign
        } else {
            let mant = frac | 0x0080_0000;
            let shift = 14 - half_exp;
            let mut half = (mant >> shift) as u16;
            if ((mant >> (shift - 1)) & 1) != 0 {
                half = half.wrapping_add(1);
            }
            sign | half
        }
    } else {
        let mut half = sign | ((half_exp as u16) << 10) | ((frac >> 13) as u16);
        if ((frac >> 12) & 1) != 0 {
            half = half.wrapping_add(1);
        }
        half
    }
}

#[derive(Default)]
pub struct TimingStats {
    pub sum: f64,
    pub sum2: f64,
    pub max: f64,
}

impl TimingStats {
    pub fn add(&mut self, value: f64) {
        self.sum += value;
        self.sum2 += value * value;
        self.max = self.max.max(value);
    }

    pub fn mean_stddev(&self, n: f64) -> (f64, f64) {
        let mean = self.sum / n;
        let variance = (self.sum2 / n - mean * mean).max(0.0);
        (mean, variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_c_layout() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(std::mem::size_of::<BlockQ8_1>(), 36);
    }

    #[test]
    fn scalar_q4_q8_dot_matches_manual_terms() {
        let mut x = [BlockQ4_0 {
            d: f32_to_f16(2.0),
            qs: [0; QK4_0 / 2],
        }];
        x[0].qs[0] = 0x98;
        let mut y = [BlockQ8_0 {
            d: f32_to_f16(0.5),
            qs: [0; QK8_0],
        }];
        y[0].qs[0] = 2;
        y[0].qs[QK8_0 / 2] = 3;
        assert_eq!(dot_q4_0_q8_0(QK8_0, &x, &y), 3.0);
    }

    #[test]
    fn scalar_q4_float_dot_uses_split_half_block_layout() {
        let mut x = [BlockQ4_0 {
            d: f32_to_f16(2.0),
            qs: [0; QK4_0 / 2],
        }];
        x[0].qs[0] = 0x70;
        let mut y = [0.0_f32; QK4_0];
        y[0] = 1.0;
        y[QK4_0 / 2] = 1.0;

        assert_eq!(dot_q4_0_f32(&x, &y), -18.0);
    }

    #[test]
    fn half_round_trips_representative_values() {
        for value in [0.0_f32, 0.5, 1.0, 2.0, 127.0] {
            assert_eq!(f16_to_f32(f32_to_f16(value)), value);
        }
    }
}
