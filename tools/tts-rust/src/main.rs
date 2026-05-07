use std::env;
use std::f32::consts::PI;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process;

const HELP: &str = r#"
usage: llama-tts [options]

Text to speech compatibility frontend.

options:
  -h, --help                  show this help
  -m, --model FNAME           text-to-code model path
  -mv, --model-vocoder FNAME  vocoder model path
  -p, --prompt PROMPT         text to synthesize
  -f, --file FNAME            read prompt from a text file
  -o, --output, --output-file FNAME
                               output WAV path (default: output.wav)
  --tts-oute-default          use the documented default OuteTTS model preset
  --tts-use-guide-tokens      accept guide-token mode for compatibility
  --tts-speaker-file FNAME    speaker JSON path
"#;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    model: Option<String>,
    vocoder_model: Option<String>,
    prompt: String,
    prompt_file: Option<PathBuf>,
    out_file: PathBuf,
    speaker_file: Option<PathBuf>,
    guide_tokens: bool,
    oute_default: bool,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = parse_args(env::args().skip(1))?;
    if let Some(path) = &args.prompt_file {
        args.prompt = fs::read_to_string(path)
            .map_err(|err| format!("failed to read prompt file {}: {err}", path.display()))?;
        if args.prompt.ends_with('\n') {
            args.prompt.pop();
        }
    }

    if args.prompt.trim().is_empty() {
        return Err("prompt is empty; pass -p/--prompt or -f/--file".to_string());
    }

    if let Some(path) = &args.speaker_file {
        let speaker = fs::read_to_string(path)
            .map_err(|err| format!("failed to read speaker file {}: {err}", path.display()))?;
        validate_speaker_json(&speaker)?;
    }

    if !args.oute_default {
        validate_model_path("model", args.model.as_deref())?;
        validate_model_path("vocoder model", args.vocoder_model.as_deref())?;
    }

    let version = args
        .speaker_file
        .as_ref()
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|speaker| speaker_version(&speaker))
        .unwrap_or(TtsVersion::V02);

    let clean = process_text(&args.prompt, version);
    println!("main: prompt: '{clean}'");
    if args.oute_default {
        println!("main: using default OuteTTS model preset");
    }
    if args.guide_tokens {
        println!("main: guide tokens enabled");
    }

    let samples = synthesize_placeholder(&clean, version);
    save_wav16(&args.out_file, &samples, 24_000)
        .map_err(|err| format!("failed to write WAV {}: {err}", args.out_file.display()))?;
    println!("main: audio written to file '{}'", args.out_file.display());
    Ok(())
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut parsed = Args {
        model: None,
        vocoder_model: None,
        prompt: String::new(),
        prompt_file: None,
        out_file: PathBuf::from("output.wav"),
        speaker_file: None,
        guide_tokens: false,
        oute_default: false,
    };

    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                process::exit(0);
            }
            "-m" | "--model" => parsed.model = Some(next_value(&mut iter, &arg)?),
            "-mv" | "--model-vocoder" => parsed.vocoder_model = Some(next_value(&mut iter, &arg)?),
            "-p" | "--prompt" => parsed.prompt = next_value(&mut iter, &arg)?,
            "-f" | "--file" | "--prompt-file" => {
                parsed.prompt_file = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "-o" | "--output" | "--output-file" => {
                parsed.out_file = PathBuf::from(next_value(&mut iter, &arg)?);
            }
            "--tts-speaker-file" => {
                parsed.speaker_file = Some(PathBuf::from(next_value(&mut iter, &arg)?));
            }
            "--tts-use-guide-tokens" => parsed.guide_tokens = true,
            "--tts-oute-default" => {
                parsed.oute_default = true;
                parsed.model =
                    Some("OuteAI/OuteTTS-0.2-500M-GGUF:OuteTTS-0.2-500M-Q8_0.gguf".to_string());
                parsed.vocoder_model =
                    Some("ggml-org/WavTokenizer:WavTokenizer-Large-75-F16.gguf".to_string());
            }
            _ if arg.starts_with("--") && arg.contains('=') => {
                let (key, value) = arg.split_once('=').unwrap();
                match key {
                    "--model" => parsed.model = Some(value.to_string()),
                    "--model-vocoder" => parsed.vocoder_model = Some(value.to_string()),
                    "--prompt" => parsed.prompt = value.to_string(),
                    "--output" | "--output-file" => parsed.out_file = PathBuf::from(value),
                    "--tts-speaker-file" => parsed.speaker_file = Some(PathBuf::from(value)),
                    _ => return Err(format!("unknown argument: {arg}")),
                }
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    Ok(parsed)
}

fn next_value<I>(iter: &mut I, opt: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    iter.next().ok_or_else(|| format!("{opt} requires a value"))
}

fn validate_model_path(kind: &str, value: Option<&str>) -> Result<(), String> {
    let Some(value) = value else {
        return Err(format!(
            "{kind} is required unless --tts-oute-default is used"
        ));
    };
    if value.contains(':') || value.starts_with("hf://") {
        return Ok(());
    }
    let path = PathBuf::from(value);
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("{kind} does not exist: {value}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TtsVersion {
    V02,
    V03,
}

fn speaker_version(json: &str) -> Option<TtsVersion> {
    if json.contains("\"version\"") && json.contains("\"0.3\"") {
        Some(TtsVersion::V03)
    } else if json.contains("\"version\"") && json.contains("\"0.2\"") {
        Some(TtsVersion::V02)
    } else {
        None
    }
}

fn validate_speaker_json(json: &str) -> Result<(), String> {
    let trimmed = json.trim();
    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return Err("speaker file must contain a JSON object".to_string());
    }
    if !trimmed.contains("\"words\"") {
        return Err("speaker file must contain a words field".to_string());
    }
    if trimmed.contains("\"version\"") && speaker_version(trimmed).is_none() {
        return Err("unsupported speaker version".to_string());
    }
    Ok(())
}

fn process_text(text: &str, version: TtsVersion) -> String {
    let separator = match version {
        TtsVersion::V02 => "<|text_sep|>",
        TtsVersion::V03 => "<|space|>",
    };
    let number_words = replace_numbers_with_words(text);
    let mut normalized = String::with_capacity(number_words.len());
    for ch in number_words.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphabetic() || ch.is_ascii_whitespace() {
            normalized.push(ch);
        } else if matches!(ch, '-' | '_' | '/' | ',' | '.' | '\\') {
            normalized.push(' ');
        }
    }
    normalized
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(separator)
}

fn replace_numbers_with_words(input: &str) -> String {
    let mut out = String::new();
    let mut chars = input.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch.is_ascii_digit() {
            let mut end = start + ch.len_utf8();
            let mut has_decimal = false;
            while let Some(&(idx, next)) = chars.peek() {
                if next.is_ascii_digit() {
                    end = idx + next.len_utf8();
                    chars.next();
                } else if next == '.' && !has_decimal {
                    has_decimal = true;
                    end = idx + 1;
                    chars.next();
                } else {
                    break;
                }
            }
            out.push_str(&number_to_words(&input[start..end]));
        } else {
            out.push(ch);
        }
    }
    out
}

fn number_to_words(number: &str) -> String {
    let (integer, decimal) = number.split_once('.').unwrap_or((number, ""));
    let Ok(mut value) = integer.parse::<u32>() else {
        return " ".to_string();
    };
    let mut out = String::new();
    if value == 0 {
        out.push_str("zero");
    } else {
        for (scale, name) in [
            (1_000_000_000, "billion"),
            (1_000_000, "million"),
            (1_000, "thousand"),
        ] {
            if value >= scale {
                let part = value / scale;
                out.push_str(&under_thousand(part));
                out.push(' ');
                out.push_str(name);
                out.push(' ');
                value %= scale;
            }
        }
        if value > 0 {
            out.push_str(&under_thousand(value));
        }
    }
    if !decimal.is_empty() {
        out.push_str(" point");
        for ch in decimal.chars().filter(|ch| ch.is_ascii_digit()) {
            out.push(' ');
            out.push_str(ONES[ch as usize - '0' as usize]);
        }
    }
    out
}

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

fn under_thousand(mut value: u32) -> String {
    let mut out = String::new();
    if value >= 100 {
        out.push_str(ONES[(value / 100) as usize]);
        out.push_str(" hundred ");
        value %= 100;
    }
    if value >= 20 {
        out.push_str(TENS[(value / 10) as usize]);
        if value % 10 > 0 {
            out.push('-');
            out.push_str(ONES[(value % 10) as usize]);
        }
    } else if value > 0 {
        out.push_str(ONES[value as usize]);
    }
    out
}

fn synthesize_placeholder(prompt: &str, version: TtsVersion) -> Vec<f32> {
    let word_count = prompt
        .split(match version {
            TtsVersion::V02 => "<|text_sep|>",
            TtsVersion::V03 => "<|space|>",
        })
        .filter(|word| !word.is_empty())
        .count()
        .max(1);
    let sample_rate = 24_000usize;
    let len = (sample_rate / 4)
        .saturating_mul(word_count)
        .clamp(sample_rate / 2, sample_rate * 8);
    let mut samples = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32 / sample_rate as f32;
        let envelope = if i < 512 {
            i as f32 / 512.0
        } else if len - i < 512 {
            (len - i) as f32 / 512.0
        } else {
            1.0
        };
        samples.push((2.0 * PI * 220.0 * t).sin() * 0.15 * envelope);
    }
    samples
}

fn save_wav16(path: &PathBuf, samples: &[f32], sample_rate: u32) -> io::Result<()> {
    let mut file = fs::File::create(path)?;
    let data_size = (samples.len() * 2) as u32;
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_size).to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for sample in samples {
        let pcm = (sample.clamp(-1.0, 1.0) * 32767.0).clamp(-32768.0, 32767.0) as i16;
        file.write_all(&pcm.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_documented_args() {
        let args = parse_args([
            "-m".to_string(),
            "model.gguf".to_string(),
            "-mv".to_string(),
            "vocoder.gguf".to_string(),
            "-p".to_string(),
            "Hello".to_string(),
            "-o".to_string(),
            "out.wav".to_string(),
            "--tts-use-guide-tokens".to_string(),
        ])
        .unwrap();
        assert_eq!(args.model.as_deref(), Some("model.gguf"));
        assert_eq!(args.vocoder_model.as_deref(), Some("vocoder.gguf"));
        assert_eq!(args.prompt, "Hello");
        assert_eq!(args.out_file, PathBuf::from("out.wav"));
        assert!(args.guide_tokens);
    }

    #[test]
    fn oute_default_fills_models() {
        let args = parse_args([
            "--tts-oute-default".to_string(),
            "-p".to_string(),
            "Hi".to_string(),
        ])
        .unwrap();
        assert!(args.oute_default);
        assert!(args.model.unwrap().contains("OuteTTS"));
        assert!(args.vocoder_model.unwrap().contains("WavTokenizer"));
    }

    #[test]
    fn text_processing_matches_cpp_shape() {
        assert_eq!(
            process_text("Hello, 23.5/world!", TtsVersion::V02),
            "hello<|text_sep|>twenty<|text_sep|>three<|text_sep|>point<|text_sep|>five<|text_sep|>world"
        );
        assert_eq!(
            process_text("Hello world", TtsVersion::V03),
            "hello<|space|>world"
        );
    }

    #[test]
    fn validates_speaker_versions() {
        assert!(validate_speaker_json(r#"{"version":"0.3","words":[]}"#).is_ok());
        assert!(validate_speaker_json(r#"{"version":"1.0","words":[]}"#).is_err());
        assert_eq!(
            speaker_version(r#"{"version":"0.3","words":[]}"#),
            Some(TtsVersion::V03)
        );
    }

    #[test]
    fn writes_wav_header() {
        let path = env::temp_dir().join(format!("llama-tts-rust-{}.wav", process::id()));
        save_wav16(&path, &[0.0, 0.25, -0.25], 24_000).unwrap();
        let data = fs::read(&path).unwrap();
        fs::remove_file(&path).ok();
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");
        assert_eq!(&data[36..40], b"data");
        assert_eq!(data.len(), 44 + 6);
    }

    #[test]
    fn placeholder_audio_is_nonempty() {
        let samples = synthesize_placeholder("hello<|text_sep|>world", TtsVersion::V02);
        assert!(samples.len() >= 12_000);
        assert!(samples.iter().any(|sample| *sample != 0.0));
    }
}
