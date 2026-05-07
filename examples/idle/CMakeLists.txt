set(TARGET llama-idle)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(IDLE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../idle-rust/Cargo.toml)
set(IDLE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/idle)
set(IDLE_RUST_BIN ${IDLE_RUST_TARGET_DIR}/debug/llama-idle${CMAKE_EXECUTABLE_SUFFIX})
set(IDLE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${IDLE_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${IDLE_RUST_MANIFEST} --target-dir ${IDLE_RUST_TARGET_DIR} --bin llama-idle
    COMMAND ${CMAKE_COMMAND} -E copy ${IDLE_RUST_BIN} ${IDLE_RUST_OUTPUT}
    DEPENDS
        llama
        ${IDLE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../idle-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../idle-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${IDLE_RUST_OUTPUT})
install(PROGRAMS ${IDLE_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
