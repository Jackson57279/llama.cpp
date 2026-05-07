set(TARGET llama-passkey)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(PASSKEY_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../passkey-rust/Cargo.toml)
set(PASSKEY_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/passkey)
set(PASSKEY_RUST_BIN ${PASSKEY_RUST_TARGET_DIR}/debug/llama-passkey${CMAKE_EXECUTABLE_SUFFIX})
set(PASSKEY_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${PASSKEY_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${PASSKEY_RUST_MANIFEST} --target-dir ${PASSKEY_RUST_TARGET_DIR} --bin llama-passkey
    COMMAND ${CMAKE_COMMAND} -E copy ${PASSKEY_RUST_BIN} ${PASSKEY_RUST_OUTPUT}
    DEPENDS
        llama
        ${PASSKEY_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../passkey-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../passkey-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${PASSKEY_RUST_OUTPUT})
install(PROGRAMS ${PASSKEY_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
