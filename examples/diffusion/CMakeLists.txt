set(TARGET llama-diffusion-cli)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(DIFFUSION_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../diffusion-rust/Cargo.toml)
set(DIFFUSION_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/diffusion)
set(DIFFUSION_RUST_BIN ${DIFFUSION_RUST_TARGET_DIR}/debug/llama-diffusion-cli${CMAKE_EXECUTABLE_SUFFIX})
set(DIFFUSION_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${DIFFUSION_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${DIFFUSION_RUST_MANIFEST} --target-dir ${DIFFUSION_RUST_TARGET_DIR} --bin llama-diffusion-cli
    COMMAND ${CMAKE_COMMAND} -E copy ${DIFFUSION_RUST_BIN} ${DIFFUSION_RUST_OUTPUT}
    DEPENDS
        llama
        ${DIFFUSION_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../diffusion-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../diffusion-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${DIFFUSION_RUST_OUTPUT})
install(PROGRAMS ${DIFFUSION_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
