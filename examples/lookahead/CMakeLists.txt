set(TARGET llama-lookahead)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(LOOKAHEAD_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../lookahead-rust/Cargo.toml)
set(LOOKAHEAD_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/lookahead)
set(LOOKAHEAD_RUST_BIN ${LOOKAHEAD_RUST_TARGET_DIR}/debug/llama-lookahead${CMAKE_EXECUTABLE_SUFFIX})
set(LOOKAHEAD_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${LOOKAHEAD_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${LOOKAHEAD_RUST_MANIFEST} --target-dir ${LOOKAHEAD_RUST_TARGET_DIR} --bin llama-lookahead
    COMMAND ${CMAKE_COMMAND} -E copy ${LOOKAHEAD_RUST_BIN} ${LOOKAHEAD_RUST_OUTPUT}
    DEPENDS
        llama
        ${LOOKAHEAD_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookahead-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookahead-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${LOOKAHEAD_RUST_OUTPUT})
install(PROGRAMS ${LOOKAHEAD_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
