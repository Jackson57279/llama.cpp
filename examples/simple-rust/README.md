# simple-rust

Rust port of `examples/simple/simple.cpp` using the llama.cpp C API.

This is intentionally a standalone Cargo example, not part of the default CMake build.
Build llama.cpp first, then point Cargo at the produced `libllama` directory if it is not under `build/bin`, `build/src`, or `build`.

```sh
cmake -B build
cmake --build build --target llama

cargo test --manifest-path examples/simple-rust/Cargo.toml --lib
LLAMA_CPP_LIB_DIR="$PWD/build/bin" cargo run --manifest-path examples/simple-rust/Cargo.toml --bin llama-simple -- -m model.gguf -n 32 "Hello my name is"
```
