use std::error::Error;
use std::ffi::{c_char, CStr, CString};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::process::ExitCode;

use sha1::{Digest as Sha1Digest, Sha1};
use sha2::Sha256;
use xxhash_rust::xxh64::{xxh64, Xxh64};

use llama_gguf_hash_rust::{
    format_uuid, generate_uuidv5, hex_lower, hex_u64_be, manifest_type, manifest_verify,
    parse_args, HashExitCode, HashParams, ManifestResult, ParseError, HASH_TYPE_SHA1,
    HASH_TYPE_SHA256, HASH_TYPE_UUID, HASH_TYPE_XXH64,
};

const UUID_NAMESPACE_LLAMA_CPP: [u8; 16] = [
    0xef, 0x00, 0x12, 0x06, 0xda, 0xdc, 0x5f, 0x6d, 0xa1, 0x5f, 0x33, 0x59, 0xe5, 0x77, 0xd4, 0xe5,
];

#[repr(C)]
struct gguf_context {
    _private: [u8; 0],
}

#[repr(C)]
struct ggml_context {
    _private: [u8; 0],
}

#[repr(C)]
struct gguf_init_params {
    no_alloc: bool,
    ctx: *mut *mut ggml_context,
}

extern "C" {
    fn gguf_init_from_file(fname: *const c_char, params: gguf_init_params) -> *mut gguf_context;
    fn gguf_free(ctx: *mut gguf_context);
    fn gguf_get_data_offset(ctx: *const gguf_context) -> usize;
    fn gguf_get_n_tensors(ctx: *const gguf_context) -> i64;
    fn gguf_get_tensor_offset(ctx: *const gguf_context, tensor_id: i64) -> usize;
    fn gguf_get_tensor_name(ctx: *const gguf_context, tensor_id: i64) -> *const c_char;
    fn gguf_get_tensor_size(ctx: *const gguf_context, tensor_id: i64) -> usize;
}

fn print_usage(executable: &str) {
    println!();
    println!("usage: {executable} [options] GGUF_IN");
    println!();
    println!("Hash a GGUF file");
    println!();
    println!("options:");
    println!("  -h, --help              show this help message and exit");
    println!("      --xxh64             use xxh64 hash");
    println!("      --sha1              use sha1 hash");
    println!("      --sha256            use sha256 hash");
    println!("      --all               use all hash");
    println!("      --no-layer          exclude per layer hash");
    println!("      --uuid              generate UUIDv5 ID");
    println!("  -c, --check <manifest>  verify against a manifest");
    println!();
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(AppError::Help(argv0)) => {
            print_usage(&argv0);
            ExitCode::SUCCESS
        }
        Err(AppError::Usage(argv0, err)) => {
            eprintln!("{err}");
            print_usage(&argv0);
            ExitCode::FAILURE
        }
        Err(AppError::Exit(code, message)) => {
            if let Some(message) = message {
                print!("{message}");
            }
            ExitCode::from(code as u8)
        }
        Err(AppError::Other(err)) => {
            eprintln!("llama-gguf-hash: {err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<HashExitCode, AppError> {
    let mut raw_args = std::env::args();
    let argv0 = raw_args
        .next()
        .unwrap_or_else(|| "llama-gguf-hash".to_string());
    let mut params = parse_args(raw_args).map_err(|err| match err {
        ParseError::Help => AppError::Help(argv0.clone()),
        err => AppError::Usage(argv0.clone(), err),
    })?;

    if !params.manifest_file.is_empty() {
        let manifest_check = manifest_type(&params.manifest_file).map_err(|_| {
            AppError::Exit(
                HashExitCode::ManifestFileError,
                Some(format!(
                    "ERROR cannot open manifest {}",
                    params.manifest_file
                )),
            )
        })?;

        if !manifest_check.sha256
            && !manifest_check.sha1
            && !manifest_check.xxh64
            && !manifest_check.uuid
        {
            return Err(AppError::Exit(
                HashExitCode::ManifestUnknownHash,
                Some(format!(
                    "ERROR manifest does not have any known hash format in {}",
                    params.manifest_file
                )),
            ));
        }

        print!("manifest  {}", params.manifest_file);
        if manifest_check.sha256 {
            print!("  sha256");
        }
        if manifest_check.sha1 {
            print!("  sha1");
        }
        if manifest_check.xxh64 {
            print!("  xxh64");
        }
        if manifest_check.uuid {
            print!("  uuid");
        }
        println!();

        if !params.xxh64 && !params.sha1 && !params.uuid && !params.sha256 {
            if manifest_check.sha256 {
                params.sha256 = true;
            } else if manifest_check.sha1 {
                params.sha1 = true;
            } else if manifest_check.xxh64 {
                params.xxh64 = true;
            } else if manifest_check.uuid {
                params.uuid = true;
            }
        }

        params.manifest_is_usable = true;
    }

    if !params.xxh64 && !params.sha1 && !params.uuid && !params.sha256 {
        params.xxh64 = true;
    }

    let exit_code = gguf_hash(&params)?;
    if params.manifest_is_usable {
        println!(
            "\nVerification results for {} - {}",
            params.manifest_file, exit_code
        );
    }
    Ok(exit_code)
}

fn gguf_hash(params: &HashParams) -> Result<HashExitCode, AppError> {
    let fname = params.input.clone();
    let c_fname = CString::new(fname.clone())?;
    let ctx = unsafe {
        gguf_init_from_file(
            c_fname.as_ptr(),
            gguf_init_params {
                no_alloc: true,
                ctx: std::ptr::null_mut(),
            },
        )
    };
    if ctx.is_null() {
        return Err(std::io::Error::other(format!("failed to open GGUF file {fname}")).into());
    }

    let result = unsafe { hash_loaded_gguf(params, ctx) };
    unsafe {
        gguf_free(ctx);
    }
    result
}

unsafe fn hash_loaded_gguf(
    params: &HashParams,
    ctx: *mut gguf_context,
) -> Result<HashExitCode, AppError> {
    let mut file = File::open(&params.input)?;
    let data_offset = gguf_get_data_offset(ctx) as u64;
    let n_tensors = gguf_get_n_tensors(ctx);

    let mut xxh64_model = params.xxh64.then(|| Xxh64::new(0));
    let mut sha1_model = params.sha1.then(Sha1::new);
    let mut sha256_model = params.sha256.then(Sha256::new);
    let mut sha1_uuid = params.uuid.then(|| {
        let mut hasher = Sha1::new();
        hasher.update(UUID_NAMESPACE_LLAMA_CPP);
        hasher
    });

    let mut tensor_layer_in_manifest = false;
    let mut model_in_manifest = false;
    let mut tensor_layer_has_mismatch = false;
    let mut model_has_mismatch = false;

    for i in 0..n_tensors {
        let name = CStr::from_ptr(gguf_get_tensor_name(ctx, i))
            .to_string_lossy()
            .into_owned();
        let size = gguf_get_tensor_size(ctx, i);
        let offset = data_offset + gguf_get_tensor_offset(ctx, i) as u64;
        let mut data = vec![0_u8; size];
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(&mut data)?;

        let tensor_layer_name = format!("{}:{name}", params.input);

        if params.xxh64 {
            if !params.no_layer {
                let hash = hex_u64_be(xxh64(&data, 0));
                update_manifest_state(
                    params,
                    HASH_TYPE_XXH64,
                    &hash,
                    &tensor_layer_name,
                    &mut tensor_layer_in_manifest,
                    &mut tensor_layer_has_mismatch,
                )?;
            }
            if let Some(hasher) = xxh64_model.as_mut() {
                hasher.update(&data);
            }
        }

        if params.sha1 {
            if !params.no_layer {
                let hash = hex_lower(&Sha1::digest(&data));
                update_manifest_state(
                    params,
                    HASH_TYPE_SHA1,
                    &hash,
                    &tensor_layer_name,
                    &mut tensor_layer_in_manifest,
                    &mut tensor_layer_has_mismatch,
                )?;
            }
            if let Some(hasher) = sha1_model.as_mut() {
                hasher.update(&data);
            }
        }

        if params.sha256 {
            if !params.no_layer {
                let hash = hex_lower(&Sha256::digest(&data));
                update_manifest_state(
                    params,
                    HASH_TYPE_SHA256,
                    &hash,
                    &tensor_layer_name,
                    &mut tensor_layer_in_manifest,
                    &mut tensor_layer_has_mismatch,
                )?;
            }
            if let Some(hasher) = sha256_model.as_mut() {
                hasher.update(&data);
            }
        }

        if let Some(hasher) = sha1_uuid.as_mut() {
            hasher.update(&data);
        }
    }

    if let Some(hasher) = xxh64_model {
        let hash = hex_u64_be(hasher.digest());
        update_manifest_state(
            params,
            HASH_TYPE_XXH64,
            &hash,
            &params.input,
            &mut model_in_manifest,
            &mut model_has_mismatch,
        )?;
    }

    if let Some(hasher) = sha1_model {
        let hash = hex_lower(&hasher.finalize());
        update_manifest_state(
            params,
            HASH_TYPE_SHA1,
            &hash,
            &params.input,
            &mut model_in_manifest,
            &mut model_has_mismatch,
        )?;
    }

    if let Some(hasher) = sha256_model {
        let hash = hex_lower(&hasher.finalize());
        update_manifest_state(
            params,
            HASH_TYPE_SHA256,
            &hash,
            &params.input,
            &mut model_in_manifest,
            &mut model_has_mismatch,
        )?;
    }

    if let Some(hasher) = sha1_uuid {
        let digest = hasher.finalize();
        let mut digest_array = [0_u8; 20];
        digest_array.copy_from_slice(&digest);
        let uuid = format_uuid(&generate_uuidv5(&digest_array));
        let verify_type = if params.manifest_is_usable {
            // Preserve the historical C++ behavior for verification lookup.
            HASH_TYPE_SHA256
        } else {
            HASH_TYPE_UUID
        };
        update_manifest_state(
            params,
            verify_type,
            &uuid,
            &params.input,
            &mut model_in_manifest,
            &mut model_has_mismatch,
        )?;
    }

    if params.manifest_is_usable {
        if !model_in_manifest {
            if !tensor_layer_in_manifest {
                return Ok(HashExitCode::ManifestMissingEntry);
            }
            if tensor_layer_has_mismatch {
                return Ok(HashExitCode::Failure);
            }
            return Ok(HashExitCode::Success);
        }

        if tensor_layer_in_manifest && tensor_layer_has_mismatch {
            return Ok(HashExitCode::Failure);
        }
        if model_has_mismatch {
            return Ok(HashExitCode::Failure);
        }
    }

    Ok(HashExitCode::Success)
}

fn update_manifest_state(
    params: &HashParams,
    hash_type: &str,
    hash: &str,
    name: &str,
    in_manifest: &mut bool,
    has_mismatch: &mut bool,
) -> Result<(), AppError> {
    if params.manifest_is_usable {
        let result = manifest_verify(&params.manifest_file, hash_type, hash, name)?;
        match result {
            ManifestResult::NotFound => {}
            ManifestResult::Mismatch => {
                *in_manifest = true;
                *has_mismatch = true;
            }
            ManifestResult::Ok => *in_manifest = true,
        }
        println!("{hash_type:<8}  {hash}  {name}  -  {result}");
    } else {
        let print_type = if hash_type == HASH_TYPE_SHA256 && hash.len() == 36 {
            HASH_TYPE_UUID
        } else {
            hash_type
        };
        println!("{print_type:<8}  {hash}  {name}");
    }
    Ok(())
}

enum AppError {
    Help(String),
    Usage(String, ParseError),
    Exit(HashExitCode, Option<String>),
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
