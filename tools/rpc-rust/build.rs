use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_LIB_DIR");

    if let Ok(lib_dir) = env::var("LLAMA_CPP_LIB_DIR") {
        emit_link(&lib_dir);
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
        if candidate.join("libggml.so").exists()
            || candidate.join("libggml.dylib").exists()
            || candidate.join("ggml.dll").exists()
            || candidate.join("ggml.lib").exists()
            || candidate.join("libggml.a").exists()
        {
            emit_link(&candidate.display().to_string());
            return;
        }
    }

    println!("cargo:warning=libggml was not found; set LLAMA_CPP_LIB_DIR or build llama.cpp before building rpc-server");
}

fn emit_link(lib_dir: &str) {
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=ggml");
    println!("cargo:rustc-link-lib=ggml-base");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
