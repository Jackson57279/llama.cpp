cmake_minimum_required(VERSION 3.12)
project(llama-simple-cmake-pkg)

set(TARGET llama-simple-cmake-pkg)

find_package(Llama REQUIRED)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(SIMPLE_RUST_MANIFEST ${CMAKE_CURRENT_LIST_DIR}/../simple-rust/Cargo.toml)
set(SIMPLE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/simple-cmake-pkg)
set(SIMPLE_RUST_BIN ${SIMPLE_RUST_TARGET_DIR}/debug/llama-simple${CMAKE_EXECUTABLE_SUFFIX})
set(SIMPLE_RUST_OUTPUT ${CMAKE_BINARY_DIR}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${SIMPLE_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${SIMPLE_RUST_MANIFEST} --target-dir ${SIMPLE_RUST_TARGET_DIR} --bin llama-simple
    COMMAND ${CMAKE_COMMAND} -E copy ${SIMPLE_RUST_BIN} ${SIMPLE_RUST_OUTPUT}
    DEPENDS
        ${SIMPLE_RUST_MANIFEST}
        ${CMAKE_CURRENT_LIST_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_LIST_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_LIST_DIR}/../simple-rust/src/lib.rs
        ${CMAKE_CURRENT_LIST_DIR}/../simple-rust/src/main.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${SIMPLE_RUST_OUTPUT})
include(GNUInstallDirs)
install(PROGRAMS ${SIMPLE_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
