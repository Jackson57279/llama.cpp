#[derive(Debug, Default, PartialEq, Eq)]
pub struct TokenizeParams {
    pub printing_ids: bool,
    pub no_bos: bool,
    pub no_escape: bool,
    pub no_parse_special: bool,
    pub disable_logging: bool,
    pub show_token_count: bool,
    pub model_path: String,
    pub prompt_path: Option<String>,
    pub prompt_arg: Option<String>,
    pub stdin_set: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ParseResult {
    Help,
    Params(TokenizeParams),
}

pub fn usage(argv0: &str) -> String {
    format!(
        "usage: {argv0} [options]\n\n{}",
        concat!(
            "The tokenize program tokenizes a prompt using a given model,\n",
            "and prints the resulting tokens to standard output.\n\n",
            "It needs a model file, a prompt, and optionally other flags\n",
            "to control the behavior of the tokenizer.\n\n",
            "    The possible options are:\n",
            "\n",
            "    -h, --help                           print this help and exit\n",
            "    -m MODEL_PATH, --model MODEL_PATH    path to model.\n",
            "    --ids                                if given, only print numerical token IDs, and not token strings.\n",
            "                                         The output format looks like [1, 2, 3], i.e. parseable by Python.\n",
            "    -f PROMPT_FNAME, --file PROMPT_FNAME read prompt from a file.\n",
            "    -p PROMPT, --prompt PROMPT           read prompt from the argument.\n",
            "    --stdin                              read prompt from standard input.\n",
            "    --no-bos                             do not ever add a BOS token to the prompt, even if normally the model uses a BOS token.\n",
            "    --no-escape                          do not escape input (such as \\n, \\t, etc.).\n",
            "    --no-parse-special                   do not parse control tokens.\n",
            "    --log-disable                        disable logs. Makes stderr quiet when loading the model.\n",
            "    --show-count                         print the total number of tokens.\n",
        )
    )
}

pub fn parse_args(args: &[String]) -> Result<ParseResult, String> {
    if args.len() <= 1 {
        return Err("usage".to_string());
    }

    let mut params = TokenizeParams::default();
    let mut model_path_set = false;
    let mut prompt_path_set = false;
    let mut prompt_set = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => return Ok(ParseResult::Help),
            "--ids" => params.printing_ids = true,
            "-m" | "--model" => {
                if model_path_set {
                    return Err("Error: -m or --model specified multiple times.".to_string());
                }
                i += 1;
                params.model_path = args
                    .get(i)
                    .ok_or_else(|| "Error: --model requires an argument.".to_string())?
                    .clone();
                model_path_set = true;
            }
            "--no-bos" => params.no_bos = true,
            "--no-escape" => params.no_escape = true,
            "--no-parse-special" => params.no_parse_special = true,
            "-p" | "--prompt" => {
                if prompt_set {
                    return Err("Error: -p or --prompt specified multiple times.".to_string());
                }
                i += 1;
                params.prompt_arg = Some(
                    args.get(i)
                        .ok_or_else(|| "Error: --prompt requires an argument.".to_string())?
                        .clone(),
                );
                prompt_set = true;
            }
            "-f" | "--file" => {
                if prompt_path_set {
                    return Err("Error: -f or --file specified multiple times.".to_string());
                }
                i += 1;
                params.prompt_path = Some(
                    args.get(i)
                        .ok_or_else(|| "Error: --file requires an argument.".to_string())?
                        .clone(),
                );
                prompt_path_set = true;
            }
            "--stdin" => params.stdin_set = true,
            "--log-disable" => params.disable_logging = true,
            "--show-count" => params.show_token_count = true,
            other => return Err(format!("Error: unknown option '{other}'")),
        }
        i += 1;
    }

    if !model_path_set {
        return Err("Error: must specify --model.".to_string());
    }

    let prompts_set = prompt_path_set as u8 + prompt_set as u8 + params.stdin_set as u8;
    if prompts_set > 1 {
        return Err("Error: --stdin, --file and --prompt are mutually exclusive.".to_string());
    }
    if prompts_set == 0 {
        return Err("Error: must specify one of: --stdin, --file or --prompt.".to_string());
    }

    Ok(ParseResult::Params(params))
}

pub fn process_escapes(input: &str) -> Vec<u8> {
    let bytes = input.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 1;
            match bytes[i] {
                b'n' => output.push(b'\n'),
                b'r' => output.push(b'\r'),
                b't' => output.push(b'\t'),
                b'\'' => output.push(b'\''),
                b'"' => output.push(b'"'),
                b'\\' => output.push(b'\\'),
                b'x' if i + 2 < bytes.len() => {
                    let hi = hex_value(bytes[i + 1]);
                    let lo = hex_value(bytes[i + 2]);
                    if let (Some(hi), Some(lo)) = (hi, lo) {
                        output.push((hi << 4) | lo);
                        i += 2;
                    } else {
                        output.push(b'\\');
                        output.push(bytes[i]);
                    }
                }
                other => {
                    output.push(b'\\');
                    output.push(other);
                }
            }
        } else {
            output.push(bytes[i]);
        }
        i += 1;
    }
    output
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn parses_required_options() {
        let parsed = parse_args(&argv(&["llama-tokenize", "-m", "m.gguf", "-p", "hi"])).unwrap();
        assert_eq!(
            parsed,
            ParseResult::Params(TokenizeParams {
                model_path: "m.gguf".to_string(),
                prompt_arg: Some("hi".to_string()),
                ..TokenizeParams::default()
            })
        );
    }

    #[test]
    fn rejects_missing_prompt() {
        assert_eq!(
            parse_args(&argv(&["llama-tokenize", "-m", "m.gguf"])).unwrap_err(),
            "Error: must specify one of: --stdin, --file or --prompt."
        );
    }

    #[test]
    fn rejects_conflicting_prompts() {
        assert_eq!(
            parse_args(&argv(&[
                "llama-tokenize",
                "-m",
                "m.gguf",
                "-p",
                "hi",
                "--stdin"
            ]))
            .unwrap_err(),
            "Error: --stdin, --file and --prompt are mutually exclusive."
        );
    }

    #[test]
    fn processes_escapes_like_common_helper() {
        assert_eq!(process_escapes(r"a\n\t\\\x41\q"), b"a\n\t\\A\\q");
    }
}
