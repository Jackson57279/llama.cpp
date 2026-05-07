set(TARGET llama-gguf-split)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(GGUF_SPLIT_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-split-rust/Cargo.toml)
set(GGUF_SPLIT_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/gguf-split)
set(GGUF_SPLIT_RUST_BIN ${GGUF_SPLIT_RUST_TARGET_DIR}/debug/llama-gguf-split${CMAKE_EXECUTABLE_SUFFIX})
set(GGUF_SPLIT_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${GGUF_SPLIT_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${GGUF_SPLIT_RUST_MANIFEST} --target-dir ${GGUF_SPLIT_RUST_TARGET_DIR} --bin llama-gguf-split
    COMMAND ${CMAKE_COMMAND} -E copy ${GGUF_SPLIT_RUST_BIN} ${GGUF_SPLIT_RUST_OUTPUT}
    DEPENDS
        llama
        ${GGUF_SPLIT_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-split-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../gguf-split-rust/src/main.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${GGUF_SPLIT_RUST_OUTPUT})

if(LLAMA_TOOLS_INSTALL)
    install(PROGRAMS ${GGUF_SPLIT_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
