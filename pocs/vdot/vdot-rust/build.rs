use std::env;
use std::path::PathBuf;

fn main() {
    if let Ok(lib_dir) = env::var("LLAMA_CPP_LIB_DIR") {
        emit_link(&lib_dir);
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.join("../../..");
    let candidates = [
        root.join("build/bin"),
        root.join("build/ggml/src"),
        root.join("build"),
    ];

    for candidate in candidates {
        if candidate.join("libggml-cpu.so").exists()
            || candidate.join("libggml-cpu.dylib").exists()
            || candidate.join("ggml-cpu.dll").exists()
            || candidate.join("ggml-cpu.lib").exists()
            || candidate.join("libggml-cpu.a").exists()
        {
            emit_link(&candidate.display().to_string());
            return;
        }
    }

    println!("cargo:warning=libggml-cpu was not found; build llama.cpp before building llama-vdot");
}

fn emit_link(lib_dir: &str) {
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=ggml-cpu");
    println!("cargo:rustc-link-lib=ggml-base");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
