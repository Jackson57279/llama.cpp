pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK8_0: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    pub d: f32,
    pub qs: [u8; QK4_0 / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_1 {
    pub d: f32,
    pub m: f32,
    pub qs: [u8; QK4_1 / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: f32,
    pub s: f32,
    pub qs: [i8; QK8_0],
}

pub struct Rng(u32);

impl Rng {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

pub fn fill_q4_0(blocks: &mut [BlockQ4_0], rng: &mut Rng) {
    for block in blocks {
        block.d = 1.0;
        for q in &mut block.qs {
            let v1 = (rng.next_u32() >> 28) as u8;
            let v2 = (rng.next_u32() >> 28) as u8;
            *q = v1 | (v2 << 4);
        }
    }
}

pub fn fill_q4_1(blocks: &mut [BlockQ4_1], rng: &mut Rng) {
    for block in blocks {
        block.d = 1.0;
        block.m = 1.0;
        for q in &mut block.qs {
            let v1 = (rng.next_u32() >> 28) as u8;
            let v2 = (rng.next_u32() >> 28) as u8;
            *q = v1 | (v2 << 4);
        }
    }
}

pub fn fill_q8_0(blocks: &mut [BlockQ8_0], rng: &mut Rng) {
    for block in blocks {
        block.d = 1.0;
        let mut sum = 0_i32;
        for q in &mut block.qs {
            *q = ((rng.next_u32() >> 24) as i32 - 128) as i8;
            sum += i32::from(*q);
        }
        block.s = block.d * sum as f32;
    }
}

pub fn simple_dot_q4_0(x: &BlockQ4_0, y: &BlockQ8_0) -> f32 {
    let mut s1 = 0_i32;
    for i in (0..QK4_0 / 2).step_by(2) {
        let v1 = i32::from(x.qs[i] & 0x0f);
        let v2 = i32::from(x.qs[i] >> 4);
        let v3 = i32::from(x.qs[i + 1] & 0x0f);
        let v4 = i32::from(x.qs[i + 1] >> 4);
        let j = 2 * i;
        s1 += v1 * i32::from(y.qs[j])
            + v2 * i32::from(y.qs[j + 1])
            + v3 * i32::from(y.qs[j + 2])
            + v4 * i32::from(y.qs[j + 3]);
    }
    y.d * x.d * s1 as f32 - 8.0 * x.d * y.s
}

pub fn simple_dot_q4_1(x: &BlockQ4_1, y: &BlockQ8_0) -> f32 {
    let mut s1 = 0_i32;
    for i in (0..QK4_1 / 2).step_by(2) {
        let v1 = i32::from(x.qs[i] & 0x0f);
        let v2 = i32::from(x.qs[i] >> 4);
        let v3 = i32::from(x.qs[i + 1] & 0x0f);
        let v4 = i32::from(x.qs[i + 1] >> 4);
        let j = 2 * i;
        s1 += v1 * i32::from(y.qs[j])
            + v2 * i32::from(y.qs[j + 1])
            + v3 * i32::from(y.qs[j + 2])
            + v4 * i32::from(y.qs[j + 3]);
    }
    y.d * x.d * s1 as f32 + y.s * x.m
}

#[derive(Default)]
pub struct Stat {
    pub sum: f64,
    pub sumt: f64,
    pub sumt2: f64,
    pub maxt: f64,
    pub nloop: i32,
}

impl Stat {
    pub fn add_result(&mut self, s: f64, t: f64) {
        self.sum += s;
        self.sumt += t;
        self.sumt2 += t * t;
        self.maxt = self.maxt.max(t);
        self.nloop += 1;
    }

    pub fn report(&self, title: &str) {
        if self.nloop < 1 {
            println!("report_result({title}): no result");
            return;
        }

        let t = self.sumt / f64::from(self.nloop);
        let mut dt = self.sumt2 / f64::from(self.nloop) - t * t;
        if dt > 0.0 {
            dt = dt.sqrt();
        }

        println!("============ {title}");
        println!("<dot> = {}", self.sum / f64::from(self.nloop));
        println!("<time> = {t} +/- {dt} us. Max. time = {} us.", self.maxt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sizes_match_c_layout() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 4 + QK4_0 / 2);
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 8 + QK4_1 / 2);
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 8 + QK8_0);
    }

    #[test]
    fn simple_q4_1_dot_includes_min_term() {
        let x = BlockQ4_1 {
            d: 1.0,
            m: 1.0,
            qs: [0; QK4_1 / 2],
        };
        let y = BlockQ8_0 {
            d: 1.0,
            s: 32.0,
            qs: [1; QK8_0],
        };

        assert_eq!(simple_dot_q4_1(&x, &y), 32.0);
    }
}
