set(TARGET llama-parallel)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(PARALLEL_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../parallel-rust/Cargo.toml)
set(PARALLEL_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/parallel)
set(PARALLEL_RUST_BIN ${PARALLEL_RUST_TARGET_DIR}/debug/llama-parallel${CMAKE_EXECUTABLE_SUFFIX})
set(PARALLEL_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${PARALLEL_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${PARALLEL_RUST_MANIFEST} --target-dir ${PARALLEL_RUST_TARGET_DIR} --bin llama-parallel
    COMMAND ${CMAKE_COMMAND} -E copy ${PARALLEL_RUST_BIN} ${PARALLEL_RUST_OUTPUT}
    DEPENDS
        llama
        ${PARALLEL_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../parallel-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../parallel-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${PARALLEL_RUST_OUTPUT})
install(PROGRAMS ${PARALLEL_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
