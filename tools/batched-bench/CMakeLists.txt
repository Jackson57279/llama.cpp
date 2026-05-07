set(TARGET llama-batched-bench)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(BATCHED_BENCH_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../batched-bench-rust/Cargo.toml)
set(BATCHED_BENCH_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/batched-bench)
set(BATCHED_BENCH_RUST_BIN ${BATCHED_BENCH_RUST_TARGET_DIR}/debug/llama-batched-bench${CMAKE_EXECUTABLE_SUFFIX})
set(BATCHED_BENCH_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${BATCHED_BENCH_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${BATCHED_BENCH_RUST_MANIFEST} --target-dir ${BATCHED_BENCH_RUST_TARGET_DIR} --bin llama-batched-bench
    COMMAND ${CMAKE_COMMAND} -E copy ${BATCHED_BENCH_RUST_BIN} ${BATCHED_BENCH_RUST_OUTPUT}
    DEPENDS
        llama
        ${BATCHED_BENCH_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../batched-bench-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../batched-bench-rust/src/main.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${BATCHED_BENCH_RUST_OUTPUT})

if(LLAMA_TOOLS_INSTALL)
    install(PROGRAMS ${BATCHED_BENCH_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
