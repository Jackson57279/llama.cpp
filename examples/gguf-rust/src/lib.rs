pub const GGML_MAX_DIMS: usize = 4;

#[derive(Clone)]
pub struct DeterministicRng {
    state: u32,
}

impl DeterministicRng {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn next_bounded(&mut self, bound: i64) -> i64 {
        self.state = self.state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((self.state / 65_536) % bound as u32) as i64
    }
}

pub fn tensor_shape(rng: &mut DeterministicRng) -> ([i64; GGML_MAX_DIMS], i32) {
    let mut ne = [1_i64; GGML_MAX_DIMS];
    let n_dims = rng.next_bounded(GGML_MAX_DIMS as i64) as i32 + 1;
    for dim in ne.iter_mut().take(n_dims as usize) {
        *dim = rng.next_bounded(10) + 1;
    }
    (ne, n_dims)
}

pub fn usage(program: &str) -> String {
    format!("usage: {program} data.gguf r|w [n]\nr: read data.gguf file\nw: write data.gguf file\nn: no check of tensor data\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_shapes_keep_valid_dimensions() {
        let mut rng = DeterministicRng::new(123456);
        for _ in 0..32 {
            let (ne, n_dims) = tensor_shape(&mut rng);
            assert!((1..=4).contains(&n_dims));
            for dim in ne.iter().take(n_dims as usize) {
                assert!((1..=10).contains(dim));
            }
            for dim in ne.iter().skip(n_dims as usize) {
                assert_eq!(*dim, 1);
            }
        }
    }

    #[test]
    fn usage_contains_modes() {
        let text = usage("llama-gguf");
        assert!(text.contains("r: read"));
        assert!(text.contains("w: write"));
    }
}
