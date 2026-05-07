set(TARGET llama-batched)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(BATCHED_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../batched-rust/Cargo.toml)
set(BATCHED_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/batched)
set(BATCHED_RUST_BIN ${BATCHED_RUST_TARGET_DIR}/debug/llama-batched${CMAKE_EXECUTABLE_SUFFIX})
set(BATCHED_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${BATCHED_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${BATCHED_RUST_MANIFEST} --target-dir ${BATCHED_RUST_TARGET_DIR} --bin llama-batched
    COMMAND ${CMAKE_COMMAND} -E copy ${BATCHED_RUST_BIN} ${BATCHED_RUST_OUTPUT}
    DEPENDS
        llama
        ${BATCHED_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../batched-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../batched-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${BATCHED_RUST_OUTPUT})
install(PROGRAMS ${BATCHED_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
