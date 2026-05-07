set(TARGET llama-retrieval)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(RETRIEVAL_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../retrieval-rust/Cargo.toml)
set(RETRIEVAL_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/retrieval)
set(RETRIEVAL_RUST_BIN ${RETRIEVAL_RUST_TARGET_DIR}/debug/llama-retrieval${CMAKE_EXECUTABLE_SUFFIX})
set(RETRIEVAL_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${RETRIEVAL_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${RETRIEVAL_RUST_MANIFEST} --target-dir ${RETRIEVAL_RUST_TARGET_DIR} --bin llama-retrieval
    COMMAND ${CMAKE_COMMAND} -E copy ${RETRIEVAL_RUST_BIN} ${RETRIEVAL_RUST_OUTPUT}
    DEPENDS
        llama
        ${RETRIEVAL_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../retrieval-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../retrieval-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${RETRIEVAL_RUST_OUTPUT})
install(PROGRAMS ${RETRIEVAL_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
