use std::env;
use std::path::PathBuf;

fn main() {
    if let Ok(lib_dir) = env::var("LLAMA_CPP_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=llama");
        println!("cargo:rustc-link-lib=ggml");
        println!("cargo:rustc-link-lib=ggml-base");
        emit_rpath(&lib_dir);
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.join("../..");
    let candidates = [
        root.join("build/bin"),
        root.join("build/src"),
        root.join("build"),
    ];

    for candidate in candidates {
        if candidate.join("libllama.so").exists()
            || candidate.join("libllama.dylib").exists()
            || candidate.join("llama.dll").exists()
            || candidate.join("llama.lib").exists()
            || candidate.join("libllama.a").exists()
        {
            println!("cargo:rustc-link-search=native={}", candidate.display());
            println!("cargo:rustc-link-lib=llama");
            println!("cargo:rustc-link-lib=ggml");
            println!("cargo:rustc-link-lib=ggml-base");
            emit_rpath(&candidate.display().to_string());
            return;
        }
    }

    println!("cargo:warning=libllama was not found; set LLAMA_CPP_LIB_DIR or build llama.cpp before building the binary");
}

fn emit_rpath(lib_dir: &str) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
