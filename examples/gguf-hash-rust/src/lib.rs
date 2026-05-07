use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub const HASH_TYPE_SHA256: &str = "sha256";
pub const HASH_TYPE_SHA1: &str = "sha1";
pub const HASH_TYPE_XXH64: &str = "xxh64";
pub const HASH_TYPE_UUID: &str = "uuid";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum HashExitCode {
    Success = 0,
    Failure = 1,
    Mismatch = 2,
    ManifestMissingEntry = 3,
    ManifestUnknownHash = 4,
    ManifestFileError = 5,
}

impl fmt::Display for HashExitCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HashExitCode::Success => write!(f, "Success"),
            HashExitCode::Failure => write!(f, "Failure"),
            HashExitCode::Mismatch => write!(f, "Mismatch"),
            HashExitCode::ManifestMissingEntry => write!(f, "Manifest Missing Entry"),
            HashExitCode::ManifestUnknownHash => write!(f, "Manifest Unknown Hash"),
            HashExitCode::ManifestFileError => write!(f, "Manifest File Error"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestResult {
    NotFound,
    Mismatch,
    Ok,
}

impl fmt::Display for ManifestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ManifestResult::NotFound => write!(f, "Not Found"),
            ManifestResult::Mismatch => write!(f, "Mismatch"),
            ManifestResult::Ok => write!(f, "Ok"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HashParams {
    pub input: String,
    pub xxh64: bool,
    pub sha1: bool,
    pub sha256: bool,
    pub uuid: bool,
    pub no_layer: bool,
    pub manifest_is_usable: bool,
    pub manifest_file: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManifestCheck {
    pub xxh64: bool,
    pub sha1: bool,
    pub sha256: bool,
    pub uuid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    Help,
    MissingValue(String),
    UnknownArgument(String),
    BadArguments,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Help => write!(f, "help requested"),
            ParseError::MissingValue(arg) => {
                write!(f, "error: invalid parameter for argument:{arg}")
            }
            ParseError::UnknownArgument(arg) => write!(f, "error: unknown argument: {arg}"),
            ParseError::BadArguments => write!(f, "error: bad arguments"),
        }
    }
}

impl std::error::Error for ParseError {}

pub fn parse_args<I, S>(args: I) -> Result<HashParams, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut params = HashParams::default();
    let mut args = args.into_iter().map(Into::into).peekable();

    while let Some(arg) = args.peek().cloned() {
        if !arg.starts_with('-') {
            break;
        }
        let arg = args.next().expect("peeked");
        let normalized = if arg.starts_with("--") {
            arg.replace('_', "-")
        } else {
            arg
        };

        match normalized.as_str() {
            "-h" | "--help" => return Err(ParseError::Help),
            "--xxh64" => params.xxh64 = true,
            "--sha1" => params.sha1 = true,
            "--uuid" => params.uuid = true,
            "--sha256" => params.sha256 = true,
            "--all" => {
                params.sha256 = true;
                params.sha1 = true;
                params.xxh64 = true;
            }
            "--no-layer" => params.no_layer = true,
            "-c" | "--check" => {
                params.manifest_file = args
                    .next()
                    .ok_or_else(|| ParseError::MissingValue(normalized.clone()))?;
            }
            _ => return Err(ParseError::UnknownArgument(normalized)),
        }
    }

    params.input = args.next().ok_or(ParseError::BadArguments)?;
    Ok(params)
}

pub fn manifest_type(path: &str) -> io::Result<ManifestCheck> {
    let file = File::open(path)?;
    let mut check = ManifestCheck::default();
    for line in BufReader::new(file).lines() {
        let line = line?;
        match line.split_whitespace().next() {
            Some(HASH_TYPE_SHA256) => check.sha256 = true,
            Some(HASH_TYPE_SHA1) => check.sha1 = true,
            Some(HASH_TYPE_XXH64) => check.xxh64 = true,
            Some(HASH_TYPE_UUID) => check.uuid = true,
            _ => {}
        }
    }
    Ok(check)
}

pub fn manifest_verify(
    path: &str,
    hash_type: &str,
    hash: &str,
    tensor_name: &str,
) -> io::Result<ManifestResult> {
    if path.is_empty() {
        return Ok(ManifestResult::NotFound);
    }

    let file = File::open(path)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        let Some(file_hash_type) = parts.next() else {
            continue;
        };
        let Some(file_hash) = parts.next() else {
            continue;
        };
        let Some(file_tensor_name) = parts.next() else {
            continue;
        };

        if file_hash_type == hash_type && file_tensor_name == tensor_name {
            return Ok(if file_hash == hash {
                ManifestResult::Ok
            } else {
                ManifestResult::Mismatch
            });
        }
    }

    Ok(ManifestResult::NotFound)
}

pub fn generate_uuidv5(sha1_digest: &[u8; 20]) -> [u8; 16] {
    let mut uuid = [0_u8; 16];
    uuid.copy_from_slice(&sha1_digest[..16]);
    uuid[6] &= !(0xF << 4);
    uuid[6] |= 5 << 4;
    uuid[8] &= !(0xC << 4);
    uuid[8] |= 0x8 << 4;
    uuid
}

pub fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(&mut out, "{byte:02x}").expect("write to String");
    }
    out
}

pub fn hex_u64_be(value: u64) -> String {
    format!("{value:016x}")
}

pub fn format_uuid(uuid: &[u8; 16]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        uuid[0],
        uuid[1],
        uuid[2],
        uuid[3],
        uuid[4],
        uuid[5],
        uuid[6],
        uuid[7],
        uuid[8],
        uuid[9],
        uuid[10],
        uuid[11],
        uuid[12],
        uuid[13],
        uuid[14],
        uuid[15],
    )
}

pub fn known_hash_types(check: &ManifestCheck) -> HashSet<&'static str> {
    let mut out = HashSet::new();
    if check.sha256 {
        out.insert(HASH_TYPE_SHA256);
    }
    if check.sha1 {
        out.insert(HASH_TYPE_SHA1);
    }
    if check.xxh64 {
        out.insert(HASH_TYPE_XXH64);
    }
    if check.uuid {
        out.insert(HASH_TYPE_UUID);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_input() {
        let params = parse_args(["model.gguf"]).unwrap();
        assert_eq!(params.input, "model.gguf");
        assert!(!params.xxh64);
    }

    #[test]
    fn parses_hash_options_and_manifest() {
        let params = parse_args(["--all", "--no-layer", "--check", "manifest", "model"]).unwrap();
        assert_eq!(params.input, "model");
        assert_eq!(params.manifest_file, "manifest");
        assert!(params.xxh64);
        assert!(params.sha1);
        assert!(params.sha256);
        assert!(params.no_layer);
    }

    #[test]
    fn supports_short_check_and_help() {
        let params = parse_args(["-c", "manifest", "--uuid", "model"]).unwrap();
        assert_eq!(params.manifest_file, "manifest");
        assert!(params.uuid);
        assert_eq!(parse_args(["--help"]).unwrap_err(), ParseError::Help);
    }

    #[test]
    fn uuidv5_sets_version_and_variant() {
        let digest = [0xff_u8; 20];
        let uuid = generate_uuidv5(&digest);
        assert_eq!(uuid[6] >> 4, 5);
        assert_eq!(uuid[8] >> 6, 2);
    }

    #[test]
    fn formats_hex_and_uuid() {
        assert_eq!(hex_lower(&[0xab, 0x00, 0x12]), "ab0012");
        assert_eq!(hex_u64_be(0x1234), "0000000000001234");
        assert_eq!(
            format_uuid(&[0, 1, 2, 3, 4, 5, 0x56, 7, 0x89, 9, 10, 11, 12, 13, 14, 15]),
            "00010203-0405-5607-8909-0a0b0c0d0e0f"
        );
    }
}
