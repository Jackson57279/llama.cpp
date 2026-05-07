set(TARGET llama-gguf-hash)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(GGUF_HASH_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-hash-rust/Cargo.toml)
set(GGUF_HASH_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/gguf-hash)
set(GGUF_HASH_RUST_BIN ${GGUF_HASH_RUST_TARGET_DIR}/debug/llama-gguf-hash${CMAKE_EXECUTABLE_SUFFIX})
set(GGUF_HASH_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${GGUF_HASH_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env GGML_LIB_DIR=$<TARGET_FILE_DIR:ggml>
            ${CARGO_EXECUTABLE} build --manifest-path ${GGUF_HASH_RUST_MANIFEST} --target-dir ${GGUF_HASH_RUST_TARGET_DIR} --bin llama-gguf-hash
    COMMAND ${CMAKE_COMMAND} -E copy ${GGUF_HASH_RUST_BIN} ${GGUF_HASH_RUST_OUTPUT}
    DEPENDS
        ggml
        ${GGUF_HASH_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-hash-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-hash-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-hash-rust/src/main.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${GGUF_HASH_RUST_OUTPUT})
install(PROGRAMS ${GGUF_HASH_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
