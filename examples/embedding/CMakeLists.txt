set(TARGET llama-embedding)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(EMBEDDING_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../embedding-rust/Cargo.toml)
set(EMBEDDING_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/embedding)
set(EMBEDDING_RUST_BIN ${EMBEDDING_RUST_TARGET_DIR}/debug/llama-embedding${CMAKE_EXECUTABLE_SUFFIX})
set(EMBEDDING_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${EMBEDDING_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${EMBEDDING_RUST_MANIFEST} --target-dir ${EMBEDDING_RUST_TARGET_DIR} --bin llama-embedding
    COMMAND ${CMAKE_COMMAND} -E copy ${EMBEDDING_RUST_BIN} ${EMBEDDING_RUST_OUTPUT}
    DEPENDS
        llama
        ${EMBEDDING_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../embedding-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../embedding-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${EMBEDDING_RUST_OUTPUT})
install(PROGRAMS ${EMBEDDING_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
