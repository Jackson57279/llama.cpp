use std::env;
use std::error::Error;
use std::process::ExitCode;

use llama_lookup_merge::{load_cache, merge_cache, parse_args, save_cache, ParseError};

fn print_usage(argv0: &str) {
    eprintln!("Merges multiple lookup cache files into a single one.");
    eprintln!("Usage: {argv0} [--help] lookup_part_1.bin lookup_part_2.bin ... lookup_merged.bin");
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(AppError::Help(argv0)) => {
            print_usage(&argv0);
            ExitCode::SUCCESS
        }
        Err(AppError::Usage(argv0)) => {
            print_usage(&argv0);
            ExitCode::FAILURE
        }
        Err(AppError::Other(err)) => {
            eprintln!("lookup-merge: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), AppError> {
    let mut raw_args = env::args();
    let argv0 = raw_args
        .next()
        .unwrap_or_else(|| "llama-lookup-merge".to_string());

    let args = parse_args(raw_args).map_err(|err| match err {
        ParseError::Help => AppError::Help(argv0.clone()),
        ParseError::MissingPaths => AppError::Usage(argv0.clone()),
    })?;

    eprintln!("lookup-merge: loading file {}", args.inputs[0]);
    let mut merged = load_cache(&args.inputs[0])?;

    for input in args.inputs.iter().skip(1) {
        eprintln!("lookup-merge: loading file {input}");
        let cache = load_cache(input)?;
        merge_cache(&mut merged, cache);
    }

    eprintln!("lookup-merge: saving file {}", args.output);
    save_cache(&merged, args.output)?;
    Ok(())
}

enum AppError {
    Help(String),
    Usage(String),
    Other(Box<dyn Error>),
}

impl<E> From<E> for AppError
where
    E: Error + 'static,
{
    fn from(value: E) -> Self {
        AppError::Other(Box::new(value))
    }
}
