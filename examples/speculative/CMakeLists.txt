set(TARGET llama-speculative)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(SPECULATIVE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../speculative-rust/Cargo.toml)
set(SPECULATIVE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/speculative)
set(SPECULATIVE_RUST_BIN ${SPECULATIVE_RUST_TARGET_DIR}/debug/llama-speculative${CMAKE_EXECUTABLE_SUFFIX})
set(SPECULATIVE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${SPECULATIVE_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${SPECULATIVE_RUST_MANIFEST} --target-dir ${SPECULATIVE_RUST_TARGET_DIR} --bin llama-speculative
    COMMAND ${CMAKE_COMMAND} -E copy ${SPECULATIVE_RUST_BIN} ${SPECULATIVE_RUST_OUTPUT}
    DEPENDS
        llama
        ${SPECULATIVE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../speculative-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../speculative-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../speculative-simple-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${SPECULATIVE_RUST_OUTPUT})
install(PROGRAMS ${SPECULATIVE_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
