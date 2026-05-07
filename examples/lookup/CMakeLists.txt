set(TARGET llama-lookup)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(LOOKUP_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-rust/Cargo.toml)
set(LOOKUP_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/lookup)
set(LOOKUP_RUST_BIN ${LOOKUP_RUST_TARGET_DIR}/debug/llama-lookup${CMAKE_EXECUTABLE_SUFFIX})
set(LOOKUP_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${LOOKUP_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${LOOKUP_RUST_MANIFEST} --target-dir ${LOOKUP_RUST_TARGET_DIR} --bin llama-lookup
    COMMAND ${CMAKE_COMMAND} -E copy ${LOOKUP_RUST_BIN} ${LOOKUP_RUST_OUTPUT}
    DEPENDS
        llama
        ${LOOKUP_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-merge-rust/src/lib.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${LOOKUP_RUST_OUTPUT})
install(PROGRAMS ${LOOKUP_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})

set(TARGET llama-lookup-create)
set(LOOKUP_CREATE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-create-rust/Cargo.toml)
set(LOOKUP_CREATE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/lookup-create)
set(LOOKUP_CREATE_RUST_BIN ${LOOKUP_CREATE_RUST_TARGET_DIR}/debug/llama-lookup-create${CMAKE_EXECUTABLE_SUFFIX})
set(LOOKUP_CREATE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${LOOKUP_CREATE_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${LOOKUP_CREATE_RUST_MANIFEST} --target-dir ${LOOKUP_CREATE_RUST_TARGET_DIR} --bin llama-lookup-create
    COMMAND ${CMAKE_COMMAND} -E copy ${LOOKUP_CREATE_RUST_BIN} ${LOOKUP_CREATE_RUST_OUTPUT}
    DEPENDS
        llama
        ${LOOKUP_CREATE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-create-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-create-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-create-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-merge-rust/src/lib.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${LOOKUP_CREATE_RUST_OUTPUT})
install(PROGRAMS ${LOOKUP_CREATE_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})

set(TARGET llama-lookup-merge)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(LOOKUP_MERGE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-merge-rust/Cargo.toml)
set(LOOKUP_MERGE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/lookup-merge)
set(LOOKUP_MERGE_RUST_BIN ${LOOKUP_MERGE_RUST_TARGET_DIR}/debug/llama-lookup-merge${CMAKE_EXECUTABLE_SUFFIX})
set(LOOKUP_MERGE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${LOOKUP_MERGE_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${LOOKUP_MERGE_RUST_MANIFEST} --target-dir ${LOOKUP_MERGE_RUST_TARGET_DIR} --bin llama-lookup-merge
    COMMAND ${CMAKE_COMMAND} -E copy ${LOOKUP_MERGE_RUST_BIN} ${LOOKUP_MERGE_RUST_OUTPUT}
    DEPENDS
        ${LOOKUP_MERGE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-merge-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-merge-rust/src/main.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${LOOKUP_MERGE_RUST_OUTPUT})
install(PROGRAMS ${LOOKUP_MERGE_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})

set(TARGET llama-lookup-stats)
set(LOOKUP_STATS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-stats-rust/Cargo.toml)
set(LOOKUP_STATS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/lookup-stats)
set(LOOKUP_STATS_RUST_BIN ${LOOKUP_STATS_RUST_TARGET_DIR}/debug/llama-lookup-stats${CMAKE_EXECUTABLE_SUFFIX})
set(LOOKUP_STATS_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${LOOKUP_STATS_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${LOOKUP_STATS_RUST_MANIFEST} --target-dir ${LOOKUP_STATS_RUST_TARGET_DIR} --bin llama-lookup-stats
    COMMAND ${CMAKE_COMMAND} -E copy ${LOOKUP_STATS_RUST_BIN} ${LOOKUP_STATS_RUST_OUTPUT}
    DEPENDS
        llama
        ${LOOKUP_STATS_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-stats-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-stats-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-stats-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../lookup-merge-rust/src/lib.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${LOOKUP_STATS_RUST_OUTPUT})
install(PROGRAMS ${LOOKUP_STATS_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
