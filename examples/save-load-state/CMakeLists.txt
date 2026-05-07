set(TARGET llama-save-load-state)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(SAVE_LOAD_STATE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../save-load-state-rust/Cargo.toml)
set(SAVE_LOAD_STATE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/save-load-state)
set(SAVE_LOAD_STATE_RUST_BIN ${SAVE_LOAD_STATE_RUST_TARGET_DIR}/debug/llama-save-load-state${CMAKE_EXECUTABLE_SUFFIX})
set(SAVE_LOAD_STATE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${SAVE_LOAD_STATE_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${SAVE_LOAD_STATE_RUST_MANIFEST} --target-dir ${SAVE_LOAD_STATE_RUST_TARGET_DIR} --bin llama-save-load-state
    COMMAND ${CMAKE_COMMAND} -E copy ${SAVE_LOAD_STATE_RUST_BIN} ${SAVE_LOAD_STATE_RUST_OUTPUT}
    DEPENDS
        llama
        ${SAVE_LOAD_STATE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../save-load-state-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../save-load-state-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${SAVE_LOAD_STATE_RUST_OUTPUT})
install(PROGRAMS ${SAVE_LOAD_STATE_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
