set(TARGET llama-results)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(RESULTS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../results-rust/Cargo.toml)
set(RESULTS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/results)
set(RESULTS_RUST_BIN ${RESULTS_RUST_TARGET_DIR}/debug/llama-results${CMAKE_EXECUTABLE_SUFFIX})
set(RESULTS_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${RESULTS_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${RESULTS_RUST_MANIFEST} --target-dir ${RESULTS_RUST_TARGET_DIR} --bin llama-results
    COMMAND ${CMAKE_COMMAND} -E copy ${RESULTS_RUST_BIN} ${RESULTS_RUST_OUTPUT}
    DEPENDS
        llama
        ${RESULTS_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../results-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../results-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../results-rust/src/main.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${RESULTS_RUST_OUTPUT})

if(LLAMA_TOOLS_INSTALL)
    install(PROGRAMS ${RESULTS_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
