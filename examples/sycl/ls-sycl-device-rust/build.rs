use std::env;
use std::path::PathBuf;

fn main() {
    if let Ok(lib_dir) = env::var("LLAMA_CPP_LIB_DIR") {
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=ggml-sycl");
        emit_rpath(&lib_dir);
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.join("../../..");
    let candidates = [
        root.join("build/bin"),
        root.join("build/ggml/src/ggml-sycl"),
        root.join("build"),
    ];

    for candidate in candidates {
        if candidate.join("libggml-sycl.so").exists()
            || candidate.join("libggml-sycl.dylib").exists()
            || candidate.join("ggml-sycl.dll").exists()
            || candidate.join("ggml-sycl.lib").exists()
            || candidate.join("libggml-sycl.a").exists()
        {
            println!("cargo:rustc-link-search=native={}", candidate.display());
            println!("cargo:rustc-link-lib=ggml-sycl");
            emit_rpath(&candidate.display().to_string());
            return;
        }
    }

    println!("cargo:warning=libggml-sycl was not found; this binary requires a GGML_SYCL build");
}

fn emit_rpath(lib_dir: &str) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
