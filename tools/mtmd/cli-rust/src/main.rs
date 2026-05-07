use std::env;
use std::io::{self, BufRead, Write};

#[derive(Debug, Clone, PartialEq, Eq)]
struct Params {
    model: Option<String>,
    mmproj: Option<String>,
    hf_repo: Option<String>,
    prompt: Option<String>,
    system_prompt: Option<String>,
    images: Vec<String>,
    audio: Vec<String>,
    n_predict: Option<i32>,
    chat_template: Option<String>,
    mmproj_use_gpu: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            model: None,
            mmproj: None,
            hf_repo: None,
            prompt: None,
            system_prompt: None,
            images: Vec::new(),
            audio: Vec::new(),
            n_predict: None,
            chat_template: None,
            mmproj_use_gpu: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Action {
    SingleTurn,
    Chat,
}

fn usage(program: &str) -> String {
    format!(
        "Experimental CLI for multimodal\n\n\
Usage: {program} [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n\
  -m and --mmproj are required\n\
  -hf user/repo can replace both -m and --mmproj in most cases\n\
  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n\
  to disable using GPU for mmproj model, add --no-mmproj-offload\n"
    )
}

fn parse_args<I, S>(args: I) -> Result<Option<Params>, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let program = args.first().map(String::as_str).unwrap_or("llama-mtmd-cli");
    let mut params = Params::default();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", usage(program));
                return Ok(None);
            }
            "-m" | "--model" => params.model = Some(next_value(&args, &mut i, "-m/--model")?),
            "--mmproj" => params.mmproj = Some(next_value(&args, &mut i, "--mmproj")?),
            "-hf" | "--hf-repo" => {
                params.hf_repo = Some(next_value(&args, &mut i, "-hf/--hf-repo")?)
            }
            "-p" | "--prompt" => params.prompt = Some(next_value(&args, &mut i, "-p/--prompt")?),
            "--system-prompt" => {
                params.system_prompt = Some(next_value(&args, &mut i, "--system-prompt")?);
            }
            "--image" => params.images.push(next_value(&args, &mut i, "--image")?),
            "--audio" => params.audio.push(next_value(&args, &mut i, "--audio")?),
            "-n" | "--predict" => {
                let raw = next_value(&args, &mut i, "-n/--predict")?;
                params.n_predict = Some(
                    raw.parse::<i32>()
                        .map_err(|_| format!("invalid value for -n/--predict: {raw}"))?,
                );
            }
            "--chat-template" => {
                params.chat_template = Some(next_value(&args, &mut i, "--chat-template")?);
            }
            "--no-mmproj-offload" => {
                params.mmproj_use_gpu = false;
                i += 1;
            }
            other if other.starts_with('-') => {
                if option_takes_value(other) {
                    let _ = next_value(&args, &mut i, other)?;
                } else {
                    i += 1;
                }
            }
            _ => i += 1,
        }
    }

    validate_params(&params, program)?;
    Ok(Some(params))
}

fn option_takes_value(option: &str) -> bool {
    matches!(
        option,
        "-c" | "--ctx-size"
            | "-b"
            | "--batch-size"
            | "--temp"
            | "--top-p"
            | "--top-k"
            | "--repeat-penalty"
            | "--threads"
            | "--seed"
            | "--flash-attn"
            | "--jinja"
    )
}

fn next_value(args: &[String], i: &mut usize, option: &str) -> Result<String, String> {
    let value = args
        .get(*i + 1)
        .ok_or_else(|| format!("error: invalid parameter for argument: {option}"))?
        .clone();
    *i += 2;
    Ok(value)
}

fn validate_params(params: &Params, program: &str) -> Result<(), String> {
    if params.hf_repo.is_none() && params.mmproj.as_deref().unwrap_or("").is_empty() {
        return Err(format!(
            "ERR: Missing --mmproj argument\n{}",
            usage(program)
        ));
    }
    Ok(())
}

fn decide_action(params: &Params) -> Action {
    if params
        .prompt
        .as_deref()
        .is_some_and(|prompt| !prompt.is_empty())
        && !params.images.is_empty()
    {
        Action::SingleTurn
    } else {
        Action::Chat
    }
}

fn add_markers_if_needed(prompt: &str, images: &[String]) -> String {
    const MARKER: &str = "<__media__>";
    if images.is_empty() || prompt.contains(MARKER) {
        prompt.to_string()
    } else {
        format!("{}{}", MARKER.repeat(images.len()), prompt)
    }
}

fn run_chat_loop<R: BufRead, W: Write>(reader: R, mut writer: W) -> io::Result<()> {
    writeln!(writer, " Running in chat mode, available commands:")?;
    writeln!(writer, "   /image <path>    load an image")?;
    writeln!(writer, "   /audio <path>    load an audio")?;
    writeln!(writer, "   /clear           clear the chat history")?;
    writeln!(writer, "   /quit or /exit   exit the program")?;

    let mut content = String::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match line {
            "/quit" | "/exit" => break,
            "/clear" => {
                content.clear();
                writeln!(writer, "Chat history cleared")?;
            }
            "/image" | "/audio" => {
                writeln!(writer, "ERR: Missing media filename")?;
            }
            _ if line.starts_with("/image ") => {
                content.push_str("<__media__>");
                writeln!(writer, "{} image loaded", &line[7..])?;
            }
            _ if line.starts_with("/audio ") => {
                content.push_str("<__media__>");
                writeln!(writer, "{} audio loaded", &line[7..])?;
            }
            _ => {
                content.push_str(line);
                writeln!(
                    writer,
                    "assistant: multimodal generation is unavailable in the Rust compatibility CLI"
                )?;
                content.clear();
            }
        }
    }
    Ok(())
}

fn run(params: Params) -> Result<(), String> {
    eprintln!("WARN: This is an experimental CLI for testing multimodal capability.");
    eprintln!("      For normal use cases, please use the standard llama-cli");
    eprintln!(
        "llama-mtmd-cli: compatibility mode, model={}, mmproj={}, hf={}",
        params.model.as_deref().unwrap_or(""),
        params.mmproj.as_deref().unwrap_or(""),
        params.hf_repo.as_deref().unwrap_or("")
    );

    match decide_action(&params) {
        Action::SingleTurn => {
            let prompt =
                add_markers_if_needed(params.prompt.as_deref().unwrap_or(""), &params.images);
            eprintln!(
                "single-turn request: images={}, audio={}, prompt={}",
                params.images.len(),
                params.audio.len(),
                prompt
            );
        }
        Action::Chat => {
            let stdin = io::stdin();
            run_chat_loop(stdin.lock(), io::stdout()).map_err(|err| err.to_string())?;
        }
    }
    Ok(())
}

fn main() {
    let params = match parse_args(env::args()) {
        Ok(Some(params)) => params,
        Ok(None) => return,
        Err(err) => {
            eprint!("{err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = run(params) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_turn_invocation() {
        let params = parse_args([
            "llama-mtmd-cli",
            "-m",
            "model.gguf",
            "--mmproj",
            "mmproj.gguf",
            "--image",
            "image.png",
            "-p",
            "describe",
            "-n",
            "32",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(decide_action(&params), Action::SingleTurn);
        assert_eq!(params.images, vec!["image.png"]);
        assert_eq!(params.n_predict, Some(32));
    }

    #[test]
    fn hf_repo_can_replace_mmproj() {
        let params = parse_args(["llama-mtmd-cli", "-hf", "ggml-org/gemma-3-4b-it-GGUF"])
            .unwrap()
            .unwrap();
        assert_eq!(
            params.hf_repo.as_deref(),
            Some("ggml-org/gemma-3-4b-it-GGUF")
        );
        assert_eq!(decide_action(&params), Action::Chat);
    }

    #[test]
    fn rejects_missing_mmproj_without_hf() {
        assert!(parse_args(["llama-mtmd-cli", "-m", "model.gguf"]).is_err());
    }

    #[test]
    fn inserts_media_markers_for_single_turn_images() {
        assert_eq!(
            add_markers_if_needed("hello", &["a.png".to_string(), "b.png".to_string()]),
            "<__media__><__media__>hello"
        );
        assert_eq!(
            add_markers_if_needed("<__media__>hello", &["a.png".to_string()]),
            "<__media__>hello"
        );
    }

    #[test]
    fn chat_loop_handles_commands() {
        let mut out = Vec::new();
        run_chat_loop(
            io::Cursor::new("/image img.png\nhello\n/clear\n/quit\n"),
            &mut out,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("img.png image loaded"));
        assert!(text.contains("Chat history cleared"));
    }
}
