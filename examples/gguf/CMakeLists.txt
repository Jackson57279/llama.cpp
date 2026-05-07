set(TARGET llama-gguf)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(GGUF_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-rust/Cargo.toml)
set(GGUF_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/gguf)
set(GGUF_RUST_BIN ${GGUF_RUST_TARGET_DIR}/debug/llama-gguf${CMAKE_EXECUTABLE_SUFFIX})
set(GGUF_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${GGUF_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env GGML_LIB_DIR=$<TARGET_FILE_DIR:ggml>
            ${CARGO_EXECUTABLE} build --manifest-path ${GGUF_RUST_MANIFEST} --target-dir ${GGUF_RUST_TARGET_DIR} --bin llama-gguf
    COMMAND ${CMAKE_COMMAND} -E copy ${GGUF_RUST_BIN} ${GGUF_RUST_OUTPUT}
    DEPENDS
        ggml
        ${GGUF_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-rust/src/main.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${GGUF_RUST_OUTPUT})
install(PROGRAMS ${GGUF_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
