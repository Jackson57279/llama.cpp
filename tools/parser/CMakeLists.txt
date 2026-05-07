if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # this tool is disabled on Windows when building with shared libraries because it uses internal functions not exported with LLAMA_API
    set(TARGET llama-debug-template-parser)
    find_program(CARGO_EXECUTABLE cargo REQUIRED)

    set(DEBUG_TEMPLATE_PARSER_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../parser-rust/Cargo.toml)
    set(DEBUG_TEMPLATE_PARSER_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/parser-tools)
    set(DEBUG_TEMPLATE_PARSER_RUST_BIN ${DEBUG_TEMPLATE_PARSER_RUST_TARGET_DIR}/debug/llama-debug-template-parser${CMAKE_EXECUTABLE_SUFFIX})
    set(DEBUG_TEMPLATE_PARSER_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${DEBUG_TEMPLATE_PARSER_RUST_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
                ${CARGO_EXECUTABLE} build --manifest-path ${DEBUG_TEMPLATE_PARSER_RUST_MANIFEST} --target-dir ${DEBUG_TEMPLATE_PARSER_RUST_TARGET_DIR} --bin llama-debug-template-parser
        COMMAND ${CMAKE_COMMAND} -E copy ${DEBUG_TEMPLATE_PARSER_RUST_BIN} ${DEBUG_TEMPLATE_PARSER_RUST_OUTPUT}
        DEPENDS
            llama
            ${DEBUG_TEMPLATE_PARSER_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/../parser-rust/build.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/../parser-rust/src/lib.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/../parser-rust/src/bin/debug_template_parser.rs
            ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
            ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
            ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
    )

    add_custom_target(${TARGET} ALL DEPENDS ${DEBUG_TEMPLATE_PARSER_RUST_OUTPUT})

    if(LLAMA_TOOLS_INSTALL)
        install(PROGRAMS ${DEBUG_TEMPLATE_PARSER_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()

set(TARGET llama-template-analysis)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(TEMPLATE_ANALYSIS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../parser-rust/Cargo.toml)
set(TEMPLATE_ANALYSIS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/parser-tools)
set(TEMPLATE_ANALYSIS_RUST_BIN ${TEMPLATE_ANALYSIS_RUST_TARGET_DIR}/debug/llama-template-analysis${CMAKE_EXECUTABLE_SUFFIX})
set(TEMPLATE_ANALYSIS_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEMPLATE_ANALYSIS_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEMPLATE_ANALYSIS_RUST_MANIFEST} --target-dir ${TEMPLATE_ANALYSIS_RUST_TARGET_DIR} --bin llama-template-analysis
    COMMAND ${CMAKE_COMMAND} -E copy ${TEMPLATE_ANALYSIS_RUST_BIN} ${TEMPLATE_ANALYSIS_RUST_OUTPUT}
    DEPENDS
        llama
        ${TEMPLATE_ANALYSIS_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../parser-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../parser-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../parser-rust/src/bin/template_analysis.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${TEMPLATE_ANALYSIS_RUST_OUTPUT})

if(LLAMA_TOOLS_INSTALL)
    install(PROGRAMS ${TEMPLATE_ANALYSIS_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
