pub fn program_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

pub fn replacement_name(filename: &str) -> String {
    if filename == "main" {
        "llama-cli".to_string()
    } else {
        format!("llama-{filename}")
    }
}

pub fn warning_message(path: &str) -> String {
    let filename = program_name(path);
    let replacement = replacement_name(filename);

    format!(
        "\nWARNING: The binary '{filename}' is deprecated.\n Please use '{replacement}' instead.\n See https://github.com/ggml-org/llama.cpp/tree/master/examples/deprecation-warning/README.md for more information.\n\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_unix_program_name() {
        assert_eq!(program_name("/tmp/server"), "server");
    }

    #[test]
    fn extracts_windows_program_name() {
        assert_eq!(program_name(r"C:\tools\quantize"), "quantize");
    }

    #[test]
    fn maps_main_to_cli() {
        assert_eq!(replacement_name("main"), "llama-cli");
    }

    #[test]
    fn prefixes_other_names() {
        assert_eq!(replacement_name("server"), "llama-server");
    }

    #[test]
    fn renders_warning() {
        let message = warning_message("/usr/bin/main");

        assert!(message.contains("The binary 'main' is deprecated."));
        assert!(message.contains("Please use 'llama-cli' instead."));
    }
}
