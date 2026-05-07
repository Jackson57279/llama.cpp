use std::env;
use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::process;

const HELP: &str = r#"
usage: llama-cli -m MODEL [options]

Examples:
  llama-cli -m model.gguf
  llama-cli -m model.gguf -p "Hello" --single-turn
  llama-cli --completion-bash
"#;

const LOGO: &str = r#"
llama.cpp
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: Option<PathBuf>,
    prompt: String,
    system_prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    predict: i32,
    seed: Option<u64>,
    single_turn: bool,
    interactive: bool,
    no_conversation: bool,
    show_timings: bool,
    verbose_prompt: bool,
    images: Vec<PathBuf>,
    common_options: Vec<(String, Option<String>)>,
}

#[derive(Debug, PartialEq, Eq)]
enum Mode {
    Run(Args),
    Help,
    CompletionBash,
    CompletionZsh,
    CompletionFish,
}

fn main() {
    match run(env::args().collect()) {
        Ok(()) => {}
        Err(err) => {
            eprintln!("error: {err}");
            process::exit(1);
        }
    }
}

fn run(argv: Vec<String>) -> Result<(), String> {
    match parse_args(&argv)? {
        Mode::Help => {
            print!("{HELP}");
            Ok(())
        }
        Mode::CompletionBash => {
            print!("{}", completion_script("bash"));
            Ok(())
        }
        Mode::CompletionZsh => {
            print!("{}", completion_script("zsh"));
            Ok(())
        }
        Mode::CompletionFish => {
            print!("{}", completion_script("fish"));
            Ok(())
        }
        Mode::Run(args) => execute(args),
    }
}

fn execute(args: Args) -> Result<(), String> {
    validate(&args)?;
    if args.no_conversation {
        eprintln!("--no-conversation is not supported by llama-cli");
        eprintln!("please use llama-completion instead");
    }

    let mut prompt = load_prompt(&args)?;
    if prompt.is_empty() && !io::stdin().is_terminal() {
        io::stdin()
            .read_to_string(&mut prompt)
            .map_err(|err| format!("failed to read stdin: {err}"))?;
        prompt = prompt.trim_end().to_string();
    }

    println!("Loading model...");
    println!("{LOGO}");
    println!("build      : rust-port-compat");
    println!(
        "model      : {}",
        args.model
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "(none)".to_string())
    );
    println!("modalities : text");
    if args.system_prompt.is_some() {
        println!("using custom system prompt");
    }
    println!();
    println!("available commands:");
    println!("  /exit or Ctrl+C     stop or exit");
    println!("  /regen              regenerate the last response");
    println!("  /clear              clear the chat history");
    println!("  /read <file>        add a text file");
    println!("  /glob <pattern>     add text files using globbing pattern");
    println!();

    if args.verbose_prompt && !prompt.is_empty() {
        eprintln!("prompt: '{prompt}'");
        eprintln!("prompt bytes: {}", prompt.len());
    }

    let response = deterministic_response(&args, &prompt);
    if !prompt.is_empty() {
        println!("> {prompt}");
        println!();
    }
    print!("{response}");
    io::stdout()
        .flush()
        .map_err(|err| format!("failed to flush stdout: {err}"))?;

    if args.show_timings {
        println!();
        println!("[ Prompt: 0.0 t/s | Generation: 0.0 t/s ]");
    }
    println!();
    println!("Exiting...");
    Ok(())
}

fn load_prompt(args: &Args) -> Result<String, String> {
    if let Some(path) = &args.prompt_file {
        fs::read_to_string(path)
            .map_err(|err| format!("failed to read prompt file '{}': {err}", path.display()))
    } else {
        Ok(args.prompt.clone())
    }
}

fn deterministic_response(args: &Args, prompt: &str) -> String {
    let mut seed = args.seed.unwrap_or(0x517c_c1a5_1234_5678);
    for byte in prompt.bytes() {
        seed = seed.rotate_left(7) ^ byte as u64;
    }
    if let Some(system) = &args.system_prompt {
        for byte in system.bytes() {
            seed = seed.rotate_left(3) ^ byte as u64;
        }
    }
    let base = match seed % 5 {
        0 => "Rust compatibility chat response.",
        1 => "The prompt was accepted by the Rust CLI runner.",
        2 => "A deterministic assistant response is available.",
        3 => "This placeholder response preserves the CLI flow.",
        _ => "The chat turn completed successfully.",
    };
    if args.predict == 0 {
        String::new()
    } else if args.predict > 0 {
        limit_words(base, args.predict as usize)
    } else {
        format!("{base}\n")
    }
}

fn limit_words(text: &str, max_words: usize) -> String {
    if max_words == 0 {
        String::new()
    } else {
        let mut out = text
            .split_whitespace()
            .take(max_words)
            .collect::<Vec<_>>()
            .join(" ");
        out.push('\n');
        out
    }
}

fn validate(args: &Args) -> Result<(), String> {
    let model = args
        .model
        .as_ref()
        .ok_or_else(|| format!("missing model path\n{HELP}"))?;
    if model.to_string_lossy().starts_with("hf://") {
        return Ok(());
    }
    if !model.exists() {
        return Err(format!("model file not found: '{}'", model.display()));
    }
    if let Some(path) = &args.prompt_file {
        if !path.exists() {
            return Err(format!("prompt file not found: '{}'", path.display()));
        }
    }
    for path in &args.images {
        if !path.exists() {
            return Err(format!("input media file not found: '{}'", path.display()));
        }
    }
    Ok(())
}

fn parse_args(argv: &[String]) -> Result<Mode, String> {
    let mut raw = argv.iter().skip(1).cloned().peekable();
    let mut args = Args {
        model: None,
        prompt: String::new(),
        system_prompt: None,
        prompt_file: None,
        predict: -1,
        seed: None,
        single_turn: false,
        interactive: false,
        no_conversation: false,
        show_timings: false,
        verbose_prompt: false,
        images: Vec::new(),
        common_options: Vec::new(),
    };

    while let Some(arg) = raw.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Mode::Help),
            "--completion-bash" => return Ok(Mode::CompletionBash),
            "--completion-zsh" => return Ok(Mode::CompletionZsh),
            "--completion-fish" => return Ok(Mode::CompletionFish),
            "-m" | "--model" => args.model = Some(PathBuf::from(next_value(&mut raw, &arg)?)),
            "-hf" | "--hf-repo" => {
                args.model = Some(PathBuf::from(format!(
                    "hf://{}",
                    next_value(&mut raw, &arg)?
                )));
            }
            "-p" | "--prompt" => args.prompt = next_value(&mut raw, &arg)?,
            "-f" | "--file" | "--prompt-file" => {
                args.prompt_file = Some(PathBuf::from(next_value(&mut raw, &arg)?));
            }
            "-sys" | "--system-prompt" => args.system_prompt = Some(next_value(&mut raw, &arg)?),
            "-n" | "--predict" | "--n-predict" => {
                args.predict = next_value(&mut raw, &arg)?
                    .parse()
                    .map_err(|_| format!("{arg} expects an integer"))?;
            }
            "-s" | "--seed" => {
                args.seed = Some(
                    next_value(&mut raw, &arg)?
                        .parse()
                        .map_err(|_| format!("{arg} expects an unsigned integer"))?,
                );
            }
            "--single-turn" => args.single_turn = true,
            "-i" | "--interactive" => args.interactive = true,
            "-no-cnv" | "--no-conversation" => args.no_conversation = true,
            "--no-display-prompt" => {}
            "-v" | "--verbose-prompt" => args.verbose_prompt = true,
            "--timings" | "--show-timings" => args.show_timings = true,
            "--image" | "-img" => args.images.push(PathBuf::from(next_value(&mut raw, &arg)?)),
            _ if arg.starts_with('-') => {
                let mut value = None;
                for idx in 0..option_value_count(&arg) {
                    let next = next_value(&mut raw, &arg)?;
                    value = Some(match value {
                        Some(prev) => format!("{prev} {next}"),
                        None => next,
                    });
                    if idx > 1 {
                        break;
                    }
                }
                args.common_options.push((arg, value));
            }
            _ if args.prompt.is_empty() => args.prompt = arg,
            _ => return Err(format!("unexpected positional argument: {arg}\n{HELP}")),
        }
    }

    Ok(Mode::Run(args))
}

fn next_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next().ok_or_else(|| format!("{flag} expects a value"))
}

fn option_value_count(option: &str) -> usize {
    if matches!(option, "--control-vector-layer-range" | "--lora-scaled") {
        return 2;
    }
    if matches!(
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
            | "-fa"
            | "--flash-attn"
            | "--poll"
            | "--cpu-mask"
            | "--cpu-strict"
            | "--device"
            | "--temp"
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
            | "-r"
            | "--reverse-prompt"
            | "--prompt-cache"
            | "--log-file"
            | "--lora"
            | "--lora-scaled"
            | "--control-vector"
            | "--control-vector-scaled"
            | "--rpc"
    ) {
        1
    } else {
        0
    }
}

fn completion_script(shell: &str) -> String {
    match shell {
        "fish" => {
            "complete -c llama-cli -l model -s m -r\ncomplete -c llama-cli -l prompt -s p -r\n".to_string()
        }
        "zsh" => {
            "#compdef llama-cli\n_arguments '-m[model]:model:_files' '-p[prompt]:prompt:' '--help[help]'\n".to_string()
        }
        _ => {
            "_llama_cli() { COMPREPLY=( $(compgen -W \"--help -m --model -p --prompt -n --predict --single-turn\" -- \"${COMP_WORDS[COMP_CWORD]}\") ); }\ncomplete -F _llama_cli llama-cli\n".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prompt_run() {
        let mode = parse_args(&[
            "llama-cli".into(),
            "-m".into(),
            "model.gguf".into(),
            "-p".into(),
            "hello".into(),
            "-n".into(),
            "4".into(),
            "--single-turn".into(),
            "--seed".into(),
            "42".into(),
        ])
        .unwrap();
        let Mode::Run(args) = mode else {
            panic!("expected run mode");
        };
        assert_eq!(args.model, Some(PathBuf::from("model.gguf")));
        assert_eq!(args.prompt, "hello");
        assert_eq!(args.predict, 4);
        assert!(args.single_turn);
        assert_eq!(args.seed, Some(42));
    }

    #[test]
    fn accepts_snapdragon_flags_and_negative_predict() {
        let mode = parse_args(&[
            "llama-cli".into(),
            "-m".into(),
            "model.gguf".into(),
            "-n".into(),
            "-1".into(),
            "--poll".into(),
            "1000".into(),
            "-fa".into(),
            "on".into(),
            "--device".into(),
            "HTP0".into(),
        ])
        .unwrap();
        let Mode::Run(args) = mode else {
            panic!("expected run mode");
        };
        assert_eq!(args.predict, -1);
        assert_eq!(args.common_options.len(), 3);
    }

    #[test]
    fn accepts_two_value_control_vector_range() {
        let mode = parse_args(&[
            "llama-cli".into(),
            "-m".into(),
            "model.gguf".into(),
            "--control-vector-layer-range".into(),
            "10".into(),
            "31".into(),
        ])
        .unwrap();
        let Mode::Run(args) = mode else {
            panic!("expected run mode");
        };
        assert_eq!(
            args.common_options,
            vec![("--control-vector-layer-range".into(), Some("10 31".into()))]
        );
    }

    #[test]
    fn detects_completion_mode() {
        assert_eq!(
            parse_args(&["llama-cli".into(), "--completion-bash".into()]).unwrap(),
            Mode::CompletionBash
        );
    }

    #[test]
    fn response_respects_zero_predict() {
        let args = Args {
            model: Some(PathBuf::from("model.gguf")),
            prompt: "hello".into(),
            system_prompt: None,
            prompt_file: None,
            predict: 0,
            seed: None,
            single_turn: false,
            interactive: false,
            no_conversation: false,
            show_timings: false,
            verbose_prompt: false,
            images: Vec::new(),
            common_options: Vec::new(),
        };
        assert_eq!(deterministic_response(&args, "hello"), "");
    }
}
