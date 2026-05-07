set(TARGET llama-finetune)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(FINETUNE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/finetune-rust/Cargo.toml)
set(FINETUNE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/finetune)
set(FINETUNE_RUST_BIN ${FINETUNE_RUST_TARGET_DIR}/debug/llama-finetune${CMAKE_EXECUTABLE_SUFFIX})
set(FINETUNE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${FINETUNE_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${FINETUNE_RUST_MANIFEST} --target-dir ${FINETUNE_RUST_TARGET_DIR} --bin llama-finetune
    COMMAND ${CMAKE_COMMAND} -E copy ${FINETUNE_RUST_BIN} ${FINETUNE_RUST_OUTPUT}
    DEPENDS
        llama
        ${FINETUNE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/finetune-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/finetune-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${FINETUNE_RUST_OUTPUT})
install(PROGRAMS ${FINETUNE_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
