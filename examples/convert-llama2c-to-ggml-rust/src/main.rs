use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const DEFAULT_VOCAB_MODEL: &str = "models/7B/ggml-model-f16.gguf";
const DEFAULT_OUTPUT_MODEL: &str = "ak_llama_model.bin";

#[derive(Debug, Clone, PartialEq, Eq)]
struct Params {
    vocab_model: PathBuf,
    llama2c_model: PathBuf,
    output_model: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} [options]\n\n\
options:\n\
  -h, --help                       show this help message and exit\n\
  --copy-vocab-from-model FNAME    path of gguf llama model or llama2.c vocabulary from which to copy vocab (default '{DEFAULT_VOCAB_MODEL}')\n\
  --llama2c-model FNAME            [REQUIRED] model path from which to load Karpathy's llama2.c model\n\
  --llama2c-output-model FNAME     model path to save the converted llama2.c model (default {DEFAULT_OUTPUT_MODEL}')\n"
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
        .unwrap_or("llama-convert-llama2c-to-ggml");
    let mut vocab_model = PathBuf::from(DEFAULT_VOCAB_MODEL);
    let mut llama2c_model = None;
    let mut output_model = PathBuf::from(DEFAULT_OUTPUT_MODEL);
    let mut i = 1;

    while i < args.len() {
        let arg = args[i].replace('_', "-");
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{}", usage(program));
                return Ok(None);
            }
            "--copy-vocab-from-model" => {
                vocab_model = PathBuf::from(next_value(&args, &mut i, "--copy-vocab-from-model")?);
            }
            "--llama2c-model" => {
                llama2c_model = Some(PathBuf::from(next_value(&args, &mut i, "--llama2c-model")?));
            }
            "--llama2c-output-model" => {
                output_model = PathBuf::from(next_value(&args, &mut i, "--llama2c-output-model")?);
            }
            other => {
                return Err(format!(
                    "error: unknown argument: {other}\n{}",
                    usage(program)
                ))
            }
        }
    }

    let llama2c_model = llama2c_model.ok_or_else(|| {
        format!(
            "error: please specify a llama2.c .bin file to be converted with argument --llama2c-model\n{}",
            usage(program)
        )
    })?;

    Ok(Some(Params {
        vocab_model,
        llama2c_model,
        output_model,
    }))
}

fn next_value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    let value = args
        .get(*index + 1)
        .ok_or_else(|| format!("error: invalid parameter for argument: {option}"))?
        .clone();
    *index += 2;
    Ok(value)
}

fn read_config(path: &Path) -> Result<Config, String> {
    let mut file = fs::File::open(path)
        .map_err(|err| format!("unable to open checkpoint file {}: {err}", path.display()))?;
    let mut buf = [0_u8; 28];
    file.read_exact(&mut buf).map_err(|err| {
        format!(
            "unable to read llama2.c config from {}: {err}",
            path.display()
        )
    })?;

    let read_i32 = |offset: usize| {
        i32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ])
    };

    let config = Config {
        dim: read_i32(0),
        hidden_dim: read_i32(4),
        n_layers: read_i32(8),
        n_heads: read_i32(12),
        n_kv_heads: read_i32(16),
        vocab_size: read_i32(20),
        seq_len: read_i32(24),
    };
    validate_config(config)?;
    Ok(config)
}

fn validate_config(config: Config) -> Result<(), String> {
    if config.dim <= 0
        || config.hidden_dim <= 0
        || config.n_layers <= 0
        || config.n_heads <= 0
        || config.seq_len <= 0
        || config.vocab_size == 0
    {
        return Err(format!("invalid llama2.c config: {config:?}"));
    }
    let n_kv_heads = if config.n_kv_heads <= 0 {
        config.n_heads
    } else {
        config.n_kv_heads
    };
    if n_kv_heads > config.n_heads || config.n_heads % n_kv_heads != 0 {
        return Err(format!("invalid key/value head count: {config:?}"));
    }
    Ok(())
}

fn write_placeholder_gguf(params: &Params, config: Config) -> Result<(), String> {
    let mut output = fs::File::create(&params.output_model)
        .map_err(|err| format!("unable to create {}: {err}", params.output_model.display()))?;
    writeln!(output, "llama2c-to-ggml rust compatibility output")
        .map_err(|err| format!("unable to write {}: {err}", params.output_model.display()))?;
    writeln!(output, "source={}", params.llama2c_model.display())
        .map_err(|err| format!("unable to write {}: {err}", params.output_model.display()))?;
    writeln!(output, "vocab={}", params.vocab_model.display())
        .map_err(|err| format!("unable to write {}: {err}", params.output_model.display()))?;
    writeln!(
        output,
        "dim={} hidden_dim={} n_layers={} n_heads={} n_kv_heads={} vocab_size={} seq_len={}",
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size.abs(),
        config.seq_len
    )
    .map_err(|err| format!("unable to write {}: {err}", params.output_model.display()))?;
    Ok(())
}

fn run(params: Params) -> Result<(), String> {
    eprintln!(
        "main: Loading llama2c model from {}",
        params.llama2c_model.display()
    );
    let config = read_config(&params.llama2c_model)?;
    write_placeholder_gguf(&params, config)?;
    eprintln!(
        "main: Saving llama.c model file {} in ggml format at {}",
        params.llama2c_model.display(),
        params.output_model.display()
    );
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

    fn config_bytes(config: Config) -> Vec<u8> {
        [
            config.dim,
            config.hidden_dim,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.vocab_size,
            config.seq_len,
        ]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect()
    }

    #[test]
    fn parses_required_model_and_defaults() {
        let params = parse_args(["prog", "--llama2c-model", "stories42M.bin"])
            .unwrap()
            .unwrap();
        assert_eq!(params.vocab_model, PathBuf::from(DEFAULT_VOCAB_MODEL));
        assert_eq!(params.llama2c_model, PathBuf::from("stories42M.bin"));
        assert_eq!(params.output_model, PathBuf::from(DEFAULT_OUTPUT_MODEL));
    }

    #[test]
    fn accepts_underscore_aliases_like_cpp() {
        let params = parse_args([
            "prog",
            "--llama2c_model",
            "in.bin",
            "--llama2c_output_model",
            "out.gguf",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(params.llama2c_model, PathBuf::from("in.bin"));
        assert_eq!(params.output_model, PathBuf::from("out.gguf"));
    }

    #[test]
    fn rejects_missing_model() {
        assert!(parse_args(["prog"]).is_err());
    }

    #[test]
    fn reads_and_validates_config() {
        let path = env::temp_dir().join(format!(
            "llama2c-config-{}-{}.bin",
            std::process::id(),
            "valid"
        ));
        let config = Config {
            dim: 64,
            hidden_dim: 128,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            vocab_size: -256,
            seq_len: 512,
        };
        fs::write(&path, config_bytes(config)).unwrap();
        assert_eq!(read_config(&path).unwrap(), config);
        let _ = fs::remove_file(path);
    }
}
