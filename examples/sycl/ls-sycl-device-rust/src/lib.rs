pub fn binary_name() -> &'static str {
    "llama-ls-sycl-device"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_binary_name() {
        assert_eq!(binary_name(), "llama-ls-sycl-device");
    }
}
