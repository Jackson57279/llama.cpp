#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

set(TARGET llama-ls-sycl-device)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(SYCL_LS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/ls-sycl-device-rust/Cargo.toml)
set(SYCL_LS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/ls-sycl-device)
set(SYCL_LS_RUST_BIN ${SYCL_LS_RUST_TARGET_DIR}/debug/llama-ls-sycl-device${CMAKE_EXECUTABLE_SUFFIX})
set(SYCL_LS_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${SYCL_LS_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
            ${CARGO_EXECUTABLE} build --manifest-path ${SYCL_LS_RUST_MANIFEST} --target-dir ${SYCL_LS_RUST_TARGET_DIR} --bin llama-ls-sycl-device
    COMMAND ${CMAKE_COMMAND} -E copy ${SYCL_LS_RUST_BIN} ${SYCL_LS_RUST_OUTPUT}
    DEPENDS
        ggml-sycl
        ${SYCL_LS_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/ls-sycl-device-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/ls-sycl-device-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/ls-sycl-device-rust/src/main.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${SYCL_LS_RUST_OUTPUT})
install(PROGRAMS ${SYCL_LS_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
