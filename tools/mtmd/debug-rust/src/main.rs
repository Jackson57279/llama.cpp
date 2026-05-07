use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Encode,
    Preproc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Params {
    model: Option<String>,
    mmproj: Option<String>,
    mode: Mode,
    size: usize,
    input: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
enum GeneratedInput {
    ImageF32 {
        width: usize,
        height: usize,
        values: Vec<f32>,
    },
    AudioF32(Vec<f32>),
    ImageU8 {
        width: usize,
        height: usize,
        values: Vec<u8>,
    },
}

fn usage(program: &str) -> String {
    format!(
        "Internal debugging tool for mtmd; See mtmd-debug.md for the pytorch equivalent code\n\
Note: we repurpose some args from other examples, they will have different meaning here\n\n\
Usage: {program} -m <model> --mmproj <mmproj> -p <mode> -n <size> --image <image> --audio <audio>\n\n\
    -n <size>: number of pixels per edge for image (always square image), or number of samples for audio\n\n\
    -p \"encode\" (debugging encode pass, default case):\n\
        --image can be:\n\
          \"white\", \"black\", \"gray\": filled 1.0f, 0.0f and 0.5f respectively\n\
          \"cb\": checkerboard pattern, alternate 1.0f and 0.0f\n\
        --audio can be:\n\
          \"one\", \"zero\", \"half\": filled 1.0f, 0.0f and 0.5f respectively\n\
          \"1010\": checkerboard pattern, alternate 1.0f and 0.0f\n\n\
    -p \"preproc\" (debugging preprocessing pass):\n\
        --image can be:\n\
          \"white\", \"black\", \"gray\": filled image with respective colors\n\
          \"cb\": checkerboard pattern\n\
        --audio can be:\n\
          \"one\", \"zero\", \"half\": filled 1.0f, 0.0f and 0.5f respectively\n\
          \"440\": sine wave with 440 Hz frequency\n\n"
    )
}

fn parse_args<I, S>(args: I) -> Result<Option<Params>, String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let args: Vec<String> = args.into_iter().map(Into::into).collect();
    let program = args
        .first()
        .map(String::as_str)
        .unwrap_or("llama-mtmd-debug");
    let mut params = Params {
        model: None,
        mmproj: None,
        mode: Mode::Encode,
        size: 0,
        input: None,
    };
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", usage(program));
                return Ok(None);
            }
            "-m" | "--model" => {
                params.model = Some(next_value(&args, &mut i, "-m/--model")?);
            }
            "--mmproj" => {
                params.mmproj = Some(next_value(&args, &mut i, "--mmproj")?);
            }
            "-p" | "--prompt" => {
                let mode = next_value(&args, &mut i, "-p/--prompt")?;
                params.mode = match mode.as_str() {
                    "" | "encode" => Mode::Encode,
                    "preproc" => Mode::Preproc,
                    _ => {
                        return Err(format!(
                            "ERR: Invalid mode specified with -p\n{}",
                            usage(program)
                        ))
                    }
                };
            }
            "-n" | "--predict" => {
                let raw = next_value(&args, &mut i, "-n/--predict")?;
                params.size = raw
                    .parse::<usize>()
                    .map_err(|_| format!("ERR: Invalid size specified with -n: {raw}"))?;
            }
            "--image" | "--audio" => {
                params.input = Some(next_value(&args, &mut i, "--image/--audio")?);
            }
            _ if args[i].starts_with('-') => {
                return Err(format!(
                    "error: unknown argument: {}\n{}",
                    args[i],
                    usage(program)
                ));
            }
            _ => {
                i += 1;
            }
        }
    }

    validate_params(&params, program)?;
    Ok(Some(params))
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
    if params.mmproj.as_deref().unwrap_or("").is_empty() {
        return Err(format!(
            "ERR: Missing --mmproj argument\n{}",
            usage(program)
        ));
    }
    if params.input.as_deref().unwrap_or("").is_empty() {
        return Err(format!(
            "ERR: At least one of --image or --audio must be specified\n{}",
            usage(program)
        ));
    }
    if params.size == 0 {
        return Err("ERR: Invalid size specified with -n, must be greater than 0".to_string());
    }
    Ok(())
}

fn generate_input(mode: Mode, input: &str, size: usize) -> Result<GeneratedInput, String> {
    match mode {
        Mode::Encode => generate_encode_input(input, size),
        Mode::Preproc => generate_preproc_input(input, size),
    }
}

fn generate_encode_input(input: &str, size: usize) -> Result<GeneratedInput, String> {
    match input {
        "black" => Ok(image_f32(size, 0.0)),
        "white" => Ok(image_f32(size, 1.0)),
        "gray" => Ok(image_f32(size, 0.5)),
        "cb" => Ok(checker_f32(size)),
        "one" => Ok(GeneratedInput::AudioF32(vec![1.0; size])),
        "zero" => Ok(GeneratedInput::AudioF32(vec![0.0; size])),
        "half" => Ok(GeneratedInput::AudioF32(vec![0.5; size])),
        "1010" => Ok(GeneratedInput::AudioF32(
            (0..size)
                .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
                .collect(),
        )),
        _ => Err("ERR: Invalid input specified with --image/--audio".to_string()),
    }
}

fn generate_preproc_input(input: &str, size: usize) -> Result<GeneratedInput, String> {
    match input {
        "black" => Ok(image_u8(size, 0)),
        "white" => Ok(image_u8(size, 255)),
        "gray" => Ok(image_u8(size, 128)),
        "cb" => Ok(checker_u8(size)),
        "one" => Ok(GeneratedInput::AudioF32(vec![1.0; size])),
        "zero" => Ok(GeneratedInput::AudioF32(vec![0.0; size])),
        "half" => Ok(GeneratedInput::AudioF32(vec![0.5; size])),
        "440" => {
            let sample_rate = 16_000.0_f32;
            let pi = std::f32::consts::PI;
            Ok(GeneratedInput::AudioF32(
                (0..size)
                    .map(|i| (2.0 * pi * 440.0 * i as f32 / sample_rate).sin())
                    .collect(),
            ))
        }
        _ => Err("ERR: Invalid input specified with --image/--audio".to_string()),
    }
}

fn image_f32(size: usize, value: f32) -> GeneratedInput {
    GeneratedInput::ImageF32 {
        width: size,
        height: size,
        values: vec![value; size * size * 3],
    }
}

fn checker_f32(size: usize) -> GeneratedInput {
    let mut values = vec![0.0; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let value = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            let offset = (y * size + x) * 3;
            values[offset..offset + 3].fill(value);
        }
    }
    GeneratedInput::ImageF32 {
        width: size,
        height: size,
        values,
    }
}

fn image_u8(size: usize, value: u8) -> GeneratedInput {
    GeneratedInput::ImageU8 {
        width: size,
        height: size,
        values: vec![value; size * size * 3],
    }
}

fn checker_u8(size: usize) -> GeneratedInput {
    let mut values = vec![0; size * size * 3];
    for y in 0..size {
        for x in 0..size {
            let value = if (x + y) % 2 == 0 { 255 } else { 0 };
            let offset = (y * size + x) * 3;
            values[offset..offset + 3].fill(value);
        }
    }
    GeneratedInput::ImageU8 {
        width: size,
        height: size,
        values,
    }
}

fn run(params: Params) -> Result<(), String> {
    let input = params.input.as_deref().unwrap_or_default();
    let generated = generate_input(params.mode, input, params.size)?;
    eprintln!(
        "llama-mtmd-debug: compatibility mode, model={}, mmproj={}",
        params.model.as_deref().unwrap_or(""),
        params.mmproj.as_deref().unwrap_or("")
    );
    match generated {
        GeneratedInput::ImageF32 { width, height, .. } => {
            eprintln!("Running encode pass for image {width} x {height}, type: {input}");
        }
        GeneratedInput::ImageU8 { width, height, .. } => {
            eprintln!("Running preprocessing pass for image {width} x {height}, type: {input}");
        }
        GeneratedInput::AudioF32(values) => {
            eprintln!("Input audio with {} samples, type: {input}", values.len());
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
    fn parses_minimal_encode_invocation() {
        let params = parse_args([
            "llama-mtmd-debug",
            "-m",
            "model.gguf",
            "--mmproj",
            "mmproj.gguf",
            "-n",
            "4",
            "--image",
            "cb",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(params.mode, Mode::Encode);
        assert_eq!(params.size, 4);
        assert_eq!(params.input.as_deref(), Some("cb"));
    }

    #[test]
    fn parses_preproc_mode() {
        let params = parse_args([
            "llama-mtmd-debug",
            "--mmproj",
            "mmproj.gguf",
            "-p",
            "preproc",
            "-n",
            "8",
            "--audio",
            "440",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(params.mode, Mode::Preproc);
        assert_eq!(params.input.as_deref(), Some("440"));
    }

    #[test]
    fn rejects_missing_mmproj_or_size() {
        assert!(parse_args(["llama-mtmd-debug", "-n", "4", "--image", "white"]).is_err());
        assert!(parse_args([
            "llama-mtmd-debug",
            "--mmproj",
            "mmproj.gguf",
            "--image",
            "white"
        ])
        .is_err());
    }

    #[test]
    fn builds_checkerboard_inputs() {
        let GeneratedInput::ImageF32 { values, .. } =
            generate_input(Mode::Encode, "cb", 2).unwrap()
        else {
            panic!("expected image");
        };
        assert_eq!(&values[0..6], &[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

        let GeneratedInput::ImageU8 { values, .. } =
            generate_input(Mode::Preproc, "cb", 2).unwrap()
        else {
            panic!("expected image");
        };
        assert_eq!(&values[0..6], &[255, 255, 255, 0, 0, 0]);
    }

    #[test]
    fn builds_audio_patterns() {
        assert_eq!(
            generate_input(Mode::Encode, "1010", 4).unwrap(),
            GeneratedInput::AudioF32(vec![1.0, 0.0, 1.0, 0.0])
        );
        let GeneratedInput::AudioF32(samples) = generate_input(Mode::Preproc, "440", 4).unwrap()
        else {
            panic!("expected audio");
        };
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0], 0.0);
    }
}
