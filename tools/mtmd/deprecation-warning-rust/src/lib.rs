pub fn program_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

pub fn warning_message(path: &str) -> String {
    let filename = program_name(path);

    format!(
        "\nWARNING: The binary '{filename}' is deprecated.\nPlease use 'llama-mtmd-cli' instead.\n\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_unix_program_name() {
        assert_eq!(program_name("/tmp/llama-llava-cli"), "llama-llava-cli");
    }

    #[test]
    fn extracts_windows_program_name() {
        assert_eq!(
            program_name(r"C:\tools\llama-qwen2vl-cli"),
            "llama-qwen2vl-cli"
        );
    }

    #[test]
    fn renders_warning() {
        let message = warning_message("/usr/bin/llama-gemma3-cli");

        assert!(message.contains("The binary 'llama-gemma3-cli' is deprecated."));
        assert!(message.contains("Please use 'llama-mtmd-cli' instead."));
    }
}
