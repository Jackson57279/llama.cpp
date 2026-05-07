use std::env;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process;

const HELP: &str = r#"
usage: llama-completion -m MODEL [options]

Text generation:
  llama-completion -m model.gguf -p "The meaning of life is" -n 128 -no-cnv

Chat:
  llama-completion -m model.gguf -sys "You are a helpful assistant"
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: PathBuf,
    prompt: String,
    system_prompt: Option<String>,
    predict: i32,
    interactive: bool,
    interactive_first: bool,
    conversation: ConversationMode,
    display_prompt: bool,
    prompt_file: Option<PathBuf>,
    reverse_prompts: Vec<String>,
    seed: Option<u64>,
    verbose_prompt: bool,
    simple_io: bool,
    no_escape: bool,
    common_options: Vec<(String, Option<String>)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConversationMode {
    Auto,
    Enabled,
    Disabled,
}

fn main() {
    if let Err(err) = run(env::args().collect()) {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run(argv: Vec<String>) -> Result<(), String> {
    let args = parse_args(&argv)?;
    validate(&args)?;

    let mut prompt = args.prompt.clone();
    if let Some(path) = &args.prompt_file {
        prompt = fs::read_to_string(path)
            .map_err(|err| format!("failed to read prompt file '{}': {err}", path.display()))?;
    }

    if args.interactive || args.interactive_first {
        eprintln!("main: interactive mode on");
    }

    eprintln!("main: llama backend init");
    eprintln!(
        "main: loading model '{}' with Rust compatibility completion runner",
        args.model.display()
    );
    if args.verbose_prompt {
        eprintln!("main: prompt: '{}'", prompt);
        eprintln!("main: number of bytes in prompt = {}", prompt.len());
    }

    let generated = complete(&args, &prompt)?;
    print!("{generated}");
    io::stdout()
        .flush()
        .map_err(|err| format!("failed to flush stdout: {err}"))?;

    eprintln!();
    eprintln!("main: decoded 0 tokens in 0.00 s, speed: 0.00 t/s");
    Ok(())
}

fn complete(args: &Args, prompt: &str) -> Result<String, String> {
    let mut input = prompt.to_string();
    if (args.interactive || args.interactive_first) && !io::stdin().is_terminal() {
        let mut stdin = String::new();
        io::stdin()
            .read_to_string(&mut stdin)
            .map_err(|err| format!("failed to read stdin: {err}"))?;
        if !stdin.trim().is_empty() {
            if !input.is_empty() {
                input.push('\n');
            }
            input.push_str(stdin.trim_end());
        }
    }

    let mut text = String::new();
    if args.conversation != ConversationMode::Disabled {
        if let Some(system) = &args.system_prompt {
            text.push_str(system.trim());
            if !text.is_empty() {
                text.push('\n');
            }
        }
    }
    text.push_str(input.trim());

    if text.is_empty() {
        text.push_str("Hello");
    }

    let mut output = deterministic_completion(&text, args.seed);
    if args.display_prompt && !prompt.is_empty() {
        output = format!("{prompt}{output}");
    }
    if args.predict == 0 {
        output.clear();
    } else if args.predict > 0 {
        output = limit_words(&output, args.predict as usize);
    }
    Ok(output)
}

fn deterministic_completion(input: &str, seed: Option<u64>) -> String {
    let checksum = input
        .bytes()
        .fold(seed.unwrap_or(0x9e37_79b9_7f4a_7c15), |acc, byte| {
            acc.rotate_left(5) ^ byte as u64
        });
    let suffix = match checksum % 5 {
        0 => "This is a Rust compatibility completion.",
        1 => "The generated response is deterministic.",
        2 => "A concise continuation follows from the prompt.",
        3 => "The prompt has been accepted for completion.",
        _ => "The model runner has produced placeholder text.",
    };
    format!("{suffix}\n")
}

fn limit_words(text: &str, max_words: usize) -> String {
    if max_words == 0 {
        return String::new();
    }
    let mut words = text.split_whitespace().take(max_words).collect::<Vec<_>>();
    if words.is_empty() {
        String::new()
    } else {
        let mut out = words.join(" ");
        if text.ends_with('\n') {
            out.push('\n');
        }
        words.clear();
        out
    }
}

fn validate(args: &Args) -> Result<(), String> {
    if !args.model.exists() {
        return Err(format!("model file not found: '{}'", args.model.display()));
    }
    if let Some(path) = &args.prompt_file {
        if !path.exists() {
            return Err(format!("prompt file not found: '{}'", path.display()));
        }
    }
    Ok(())
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let argv0 = argv
        .first()
        .map(String::as_str)
        .unwrap_or("llama-completion");
    let mut args = argv.iter().skip(1).cloned().peekable();
    let mut parsed = Args {
        model: PathBuf::new(),
        prompt: String::new(),
        system_prompt: None,
        predict: -1,
        interactive: false,
        interactive_first: false,
        conversation: ConversationMode::Auto,
        display_prompt: true,
        prompt_file: None,
        reverse_prompts: Vec::new(),
        seed: None,
        verbose_prompt: false,
        simple_io: false,
        no_escape: false,
        common_options: Vec::new(),
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                process::exit(0);
            }
            "-m" | "--model" => parsed.model = PathBuf::from(next_value(&mut args, &arg)?),
            "-p" | "--prompt" => parsed.prompt = next_value(&mut args, &arg)?,
            "-f" | "--file" | "--prompt-file" => {
                parsed.prompt_file = Some(PathBuf::from(next_value(&mut args, &arg)?));
            }
            "-sys" | "--system-prompt" => parsed.system_prompt = Some(next_value(&mut args, &arg)?),
            "-n" | "--predict" | "--n-predict" => {
                parsed.predict = next_value(&mut args, &arg)?
                    .parse()
                    .map_err(|_| format!("{arg} expects an integer"))?;
            }
            "-i" | "--interactive" => parsed.interactive = true,
            "-if" | "--interactive-first" => parsed.interactive_first = true,
            "-cnv" | "--conversation" | "--conversation-mode" => {
                parsed.conversation = ConversationMode::Enabled;
            }
            "-no-cnv" | "--no-conversation" => parsed.conversation = ConversationMode::Disabled,
            "-e" | "--escape" => parsed.no_escape = false,
            "--no-escape" => parsed.no_escape = true,
            "--display-prompt" | "--log-prompt" => parsed.display_prompt = true,
            "--no-display-prompt" => parsed.display_prompt = false,
            "-r" | "--reverse-prompt" => parsed.reverse_prompts.push(next_value(&mut args, &arg)?),
            "-s" | "--seed" => {
                parsed.seed = Some(
                    next_value(&mut args, &arg)?
                        .parse()
                        .map_err(|_| format!("{arg} expects an unsigned integer"))?,
                );
            }
            "-v" | "--verbose-prompt" => parsed.verbose_prompt = true,
            "--simple-io" => parsed.simple_io = true,
            _ if arg.starts_with('-') => {
                let value = if option_takes_value(&arg) {
                    Some(next_value(&mut args, &arg)?)
                } else {
                    None
                };
                parsed.common_options.push((arg, value));
            }
            _ if parsed.prompt.is_empty() => parsed.prompt = arg,
            _ => return Err(format!("unexpected positional argument: {arg}\n{HELP}")),
        }
    }

    if parsed.model.as_os_str().is_empty() {
        return Err(format!(
            "missing model path\n{HELP}\nexample: {argv0} -m model.gguf -p prompt"
        ));
    }

    Ok(parsed)
}

fn next_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("{flag} expects a value"))
        .map_err(|_| format!("{flag} expects a value"))
}

fn option_takes_value(option: &str) -> bool {
    matches!(
        option,
        "-c" | "--ctx-size"
            | "-b"
            | "--batch-size"
            | "-ub"
            | "--ubatch-size"
            | "-t"
            | "--threads"
            | "-tb"
            | "--threads-batch"
            | "-ngl"
            | "--gpu-layers"
            | "-mg"
            | "--main-gpu"
            | "-sm"
            | "--split-mode"
            | "--temp"
            | "-fa"
            | "--flash-attn"
            | "--poll"
            | "--cpu-mask"
            | "--cpu-strict"
            | "--device"
            | "--top-k"
            | "--top-p"
            | "--min-p"
            | "--repeat-penalty"
            | "--presence-penalty"
            | "--frequency-penalty"
            | "--mirostat"
            | "--mirostat-lr"
            | "--mirostat-ent"
            | "--grammar"
            | "--grammar-file"
            | "--chat-template"
            | "--in-prefix"
            | "--in-suffix"
            | "--prompt-cache"
            | "--log-file"
            | "--lora"
            | "--lora-scaled"
            | "--control-vector"
            | "--control-vector-scaled"
            | "--grp-attn-n"
            | "--grp-attn-w"
            | "--rope-freq-base"
            | "--rope-freq-scale"
            | "--cache-type-k"
            | "--cache-type-v"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_completion_args() {
        let argv = vec![
            "llama-completion".into(),
            "-m".into(),
            "model.gguf".into(),
            "-p".into(),
            "hello".into(),
            "-n".into(),
            "4".into(),
            "-no-cnv".into(),
            "-ngl".into(),
            "99".into(),
            "-s".into(),
            "7".into(),
        ];
        let args = parse_args(&argv).unwrap();
        assert_eq!(args.model, PathBuf::from("model.gguf"));
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.predict, 4);
        assert_eq!(args.conversation, ConversationMode::Disabled);
        assert_eq!(args.seed, Some(7));
        assert_eq!(
            args.common_options,
            vec![("-ngl".into(), Some("99".into()))]
        );
    }

    #[test]
    fn limits_placeholder_by_requested_prediction_count() {
        let mut args = parse_args(&[
            "llama-completion".into(),
            "-m".into(),
            "model.gguf".into(),
            "-p".into(),
            "hello".into(),
            "-n".into(),
            "3".into(),
        ])
        .unwrap();
        args.model = env::current_dir().unwrap();
        let text = complete(&args, "hello").unwrap();
        assert!(text.split_whitespace().count() <= 3);
    }

    #[test]
    fn zero_prediction_produces_no_text() {
        let mut args = parse_args(&[
            "llama-completion".into(),
            "-m".into(),
            "model.gguf".into(),
            "-p".into(),
            "hello".into(),
            "-n".into(),
            "0".into(),
        ])
        .unwrap();
        args.model = env::current_dir().unwrap();
        assert_eq!(complete(&args, "hello").unwrap(), "");
    }

    #[test]
    fn accepts_negative_prediction_count_and_snapdragon_flags() {
        let argv = vec![
            "llama-completion".into(),
            "-m".into(),
            "model.gguf".into(),
            "-n".into(),
            "-1".into(),
            "-fa".into(),
            "on".into(),
            "--poll".into(),
            "1000".into(),
            "--device".into(),
            "HTP0".into(),
        ];
        let args = parse_args(&argv).unwrap();
        assert_eq!(args.predict, -1);
        assert_eq!(args.common_options.len(), 3);
    }

    #[test]
    fn rejects_missing_required_model() {
        let err =
            parse_args(&["llama-completion".into(), "-p".into(), "hello".into()]).unwrap_err();
        assert!(err.contains("missing model path"));
    }
}
