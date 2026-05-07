set(TARGET llama-debug)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(DEBUG_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../debug-rust/Cargo.toml)
set(DEBUG_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/debug-example)
set(DEBUG_RUST_BIN ${DEBUG_RUST_TARGET_DIR}/debug/llama-debug${CMAKE_EXECUTABLE_SUFFIX})
set(DEBUG_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${DEBUG_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${DEBUG_RUST_MANIFEST} --target-dir ${DEBUG_RUST_TARGET_DIR} --bin llama-debug
    COMMAND ${CMAKE_COMMAND} -E copy ${DEBUG_RUST_BIN} ${DEBUG_RUST_OUTPUT}
    DEPENDS
        llama
        ${DEBUG_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../debug-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../debug-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../debug-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${DEBUG_RUST_OUTPUT})
install(PROGRAMS ${DEBUG_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
