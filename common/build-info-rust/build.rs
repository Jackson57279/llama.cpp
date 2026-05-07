use std::env;

fn main() {
    let build_number = env::var("LLAMA_BUILD_NUMBER").unwrap_or_else(|_| "0".to_string());
    let commit = env::var("LLAMA_BUILD_COMMIT").unwrap_or_else(|_| "unknown".to_string());
    let compiler = env::var("BUILD_COMPILER").unwrap_or_else(|_| "unknown".to_string());
    let target = env::var("BUILD_TARGET").unwrap_or_else(|_| "unknown".to_string());
    let build_info = format!("b{build_number}-{commit}");

    println!("cargo:rustc-env=LLAMA_BUILD_NUMBER={build_number}");
    println!("cargo:rustc-env=LLAMA_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=BUILD_COMPILER={compiler}");
    println!("cargo:rustc-env=BUILD_TARGET={target}");
    println!("cargo:rustc-env=LLAMA_BUILD_INFO={build_info}");
}
