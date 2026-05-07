llama_add_compile_flags()

add_test(
    NAME test-no-c-cpp-extension-files
    COMMAND ${CMAKE_COMMAND}
        -DPROJECT_ROOT=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/no-c-cpp-files.cmake
)

find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(LLAMA_TEST_GET_MODEL_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/get-model-rust/Cargo.toml)
set(LLAMA_TEST_GET_MODEL_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-get-model)
set(LLAMA_TEST_GET_MODEL_RUST_LIB ${LLAMA_TEST_GET_MODEL_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_test_get_model_rust${CMAKE_STATIC_LIBRARY_SUFFIX})
set(LLAMA_SIMPLE_TOKENIZE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/peg-parser/simple-tokenize-rust/Cargo.toml)
set(LLAMA_SIMPLE_TOKENIZE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/simple-tokenize)
set(LLAMA_SIMPLE_TOKENIZE_RUST_LIB ${LLAMA_SIMPLE_TOKENIZE_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_simple_tokenize_rust${CMAKE_STATIC_LIBRARY_SUFFIX})

add_custom_command(
    OUTPUT ${LLAMA_TEST_GET_MODEL_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${LLAMA_TEST_GET_MODEL_RUST_MANIFEST} --target-dir ${LLAMA_TEST_GET_MODEL_RUST_TARGET_DIR}
    DEPENDS
        ${LLAMA_TEST_GET_MODEL_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/get-model-rust/src/lib.rs
)
add_custom_target(llama-test-get-model-rust-build DEPENDS ${LLAMA_TEST_GET_MODEL_RUST_LIB})
add_library(llama-test-get-model-rust STATIC IMPORTED GLOBAL)
set_target_properties(llama-test-get-model-rust PROPERTIES IMPORTED_LOCATION ${LLAMA_TEST_GET_MODEL_RUST_LIB})
add_dependencies(llama-test-get-model-rust llama-test-get-model-rust-build)

add_custom_command(
    OUTPUT ${LLAMA_SIMPLE_TOKENIZE_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${LLAMA_SIMPLE_TOKENIZE_RUST_MANIFEST} --target-dir ${LLAMA_SIMPLE_TOKENIZE_RUST_TARGET_DIR}
    DEPENDS
        ${LLAMA_SIMPLE_TOKENIZE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/peg-parser/simple-tokenize-rust/src/lib.rs
)
add_custom_target(llama-simple-tokenize-rust-build DEPENDS ${LLAMA_SIMPLE_TOKENIZE_RUST_LIB})
add_library(llama-simple-tokenize-rust STATIC IMPORTED GLOBAL)
set_target_properties(llama-simple-tokenize-rust PROPERTIES IMPORTED_LOCATION ${LLAMA_SIMPLE_TOKENIZE_RUST_LIB})
add_dependencies(llama-simple-tokenize-rust llama-simple-tokenize-rust-build)

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common llama-test-get-model-rust)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
set(TEST_TOKENIZER_0_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-0-rust/Cargo.toml)
set(TEST_TOKENIZER_0_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-tokenizer-0)
set(TEST_TOKENIZER_0_RUST_BIN ${TEST_TOKENIZER_0_RUST_TARGET_DIR}/debug/test-tokenizer-0${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_TOKENIZER_0_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-tokenizer-0${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_TOKENIZER_0_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_TOKENIZER_0_RUST_MANIFEST} --target-dir ${TEST_TOKENIZER_0_RUST_TARGET_DIR} --bin test-tokenizer-0
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_TOKENIZER_0_RUST_BIN} ${TEST_TOKENIZER_0_RUST_OUTPUT}
    DEPENDS
        llama
        ${TEST_TOKENIZER_0_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-0-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-0-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-0-rust/src/main.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)
add_custom_target(test-tokenizer-0 ALL DEPENDS ${TEST_TOKENIZER_0_RUST_OUTPUT})

llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test_cmd(${TEST_TOKENIZER_0_RUST_OUTPUT} NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    set(TEST_GRAMMAR_LLGUIDANCE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-grammar-llguidance-rust/Cargo.toml)
    set(TEST_GRAMMAR_LLGUIDANCE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-grammar-llguidance)
    set(TEST_GRAMMAR_LLGUIDANCE_RUST_BIN ${TEST_GRAMMAR_LLGUIDANCE_RUST_TARGET_DIR}/debug/test-grammar-llguidance${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_GRAMMAR_LLGUIDANCE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-grammar-llguidance${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_GRAMMAR_LLGUIDANCE_RUST_OUTPUT}
        COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_GRAMMAR_LLGUIDANCE_RUST_MANIFEST} --target-dir ${TEST_GRAMMAR_LLGUIDANCE_RUST_TARGET_DIR} --bin test-grammar-llguidance
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_GRAMMAR_LLGUIDANCE_RUST_BIN} ${TEST_GRAMMAR_LLGUIDANCE_RUST_OUTPUT}
        DEPENDS
            ${TEST_GRAMMAR_LLGUIDANCE_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-grammar-llguidance-rust/src/main.rs
    )
    add_custom_target(test-grammar-llguidance ALL DEPENDS ${TEST_GRAMMAR_LLGUIDANCE_RUST_OUTPUT})
    add_test(NAME test-grammar-llguidance WORKING_DIRECTORY . COMMAND ${TEST_GRAMMAR_LLGUIDANCE_RUST_OUTPUT} ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
    set_property(TEST test-grammar-llguidance PROPERTY LABELS "main")
endif ()

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    set(TEST_SAMPLING_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-sampling-rust/Cargo.toml)
    set(TEST_SAMPLING_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-sampling)
    set(TEST_SAMPLING_RUST_BIN ${TEST_SAMPLING_RUST_TARGET_DIR}/debug/test-sampling${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_SAMPLING_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-sampling${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_SAMPLING_RUST_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
                ${CARGO_EXECUTABLE} build --manifest-path ${TEST_SAMPLING_RUST_MANIFEST} --target-dir ${TEST_SAMPLING_RUST_TARGET_DIR} --bin test-sampling
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_SAMPLING_RUST_BIN} ${TEST_SAMPLING_RUST_OUTPUT}
        DEPENDS
            llama
            ${TEST_SAMPLING_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-sampling-rust/build.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-sampling-rust/src/lib.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-sampling-rust/src/main.rs
    )
    add_custom_target(test-sampling ALL DEPENDS ${TEST_SAMPLING_RUST_OUTPUT})
    add_test(NAME test-sampling WORKING_DIRECTORY . COMMAND ${TEST_SAMPLING_RUST_OUTPUT})
    set_property(TEST test-sampling PROPERTY LABELS "main")
    set(TEST_GRAMMAR_PARSER_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-grammar-parser-rust/Cargo.toml)
    set(TEST_GRAMMAR_PARSER_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-grammar-parser)
    set(TEST_GRAMMAR_PARSER_RUST_BIN ${TEST_GRAMMAR_PARSER_RUST_TARGET_DIR}/debug/test-grammar-parser${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_GRAMMAR_PARSER_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-grammar-parser${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_GRAMMAR_PARSER_RUST_OUTPUT}
        COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_GRAMMAR_PARSER_RUST_MANIFEST} --target-dir ${TEST_GRAMMAR_PARSER_RUST_TARGET_DIR} --bin test-grammar-parser
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_GRAMMAR_PARSER_RUST_BIN} ${TEST_GRAMMAR_PARSER_RUST_OUTPUT}
        DEPENDS
            ${TEST_GRAMMAR_PARSER_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-grammar-parser-rust/src/main.rs
    )
    add_custom_target(test-grammar-parser ALL DEPENDS ${TEST_GRAMMAR_PARSER_RUST_OUTPUT})
    add_test(NAME test-grammar-parser WORKING_DIRECTORY . COMMAND ${TEST_GRAMMAR_PARSER_RUST_OUTPUT})
    set_property(TEST test-grammar-parser PROPERTY LABELS "main")
    set(TEST_LLAMA_GRAMMAR_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-llama-grammar-rust/Cargo.toml)
    set(TEST_LLAMA_GRAMMAR_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-llama-grammar)
    set(TEST_LLAMA_GRAMMAR_RUST_BIN ${TEST_LLAMA_GRAMMAR_RUST_TARGET_DIR}/debug/test-llama-grammar${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_LLAMA_GRAMMAR_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-llama-grammar${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_LLAMA_GRAMMAR_RUST_OUTPUT}
        COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_LLAMA_GRAMMAR_RUST_MANIFEST} --target-dir ${TEST_LLAMA_GRAMMAR_RUST_TARGET_DIR} --bin test-llama-grammar
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_LLAMA_GRAMMAR_RUST_BIN} ${TEST_LLAMA_GRAMMAR_RUST_OUTPUT}
        DEPENDS
            ${TEST_LLAMA_GRAMMAR_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-llama-grammar-rust/src/main.rs
    )
    add_custom_target(test-llama-grammar ALL DEPENDS ${TEST_LLAMA_GRAMMAR_RUST_OUTPUT})
    add_test(NAME test-llama-grammar WORKING_DIRECTORY . COMMAND ${TEST_LLAMA_GRAMMAR_RUST_OUTPUT})
    set_property(TEST test-llama-grammar PROPERTY LABELS "main")
    set(TEST_GRAMMAR_INTEGRATION_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-grammar-integration-rust/Cargo.toml)
    set(TEST_GRAMMAR_INTEGRATION_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-grammar-integration)
    set(TEST_GRAMMAR_INTEGRATION_RUST_BIN ${TEST_GRAMMAR_INTEGRATION_RUST_TARGET_DIR}/debug/test-grammar-integration${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_GRAMMAR_INTEGRATION_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-grammar-integration${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_GRAMMAR_INTEGRATION_RUST_OUTPUT}
        COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_GRAMMAR_INTEGRATION_RUST_MANIFEST} --target-dir ${TEST_GRAMMAR_INTEGRATION_RUST_TARGET_DIR} --bin test-grammar-integration
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_GRAMMAR_INTEGRATION_RUST_BIN} ${TEST_GRAMMAR_INTEGRATION_RUST_OUTPUT}
        DEPENDS
            ${TEST_GRAMMAR_INTEGRATION_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-grammar-integration-rust/src/main.rs
    )
    add_custom_target(test-grammar-integration ALL DEPENDS ${TEST_GRAMMAR_INTEGRATION_RUST_OUTPUT})
    add_test(NAME test-grammar-integration WORKING_DIRECTORY . COMMAND ${TEST_GRAMMAR_INTEGRATION_RUST_OUTPUT})
    set_property(TEST test-grammar-integration PROPERTY LABELS "main")
    set(TEST_CHAT_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-chat-rust/Cargo.toml)
    set(TEST_CHAT_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-chat)
    set(TEST_CHAT_RUST_BIN ${TEST_CHAT_RUST_TARGET_DIR}/debug/test-chat${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_CHAT_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-chat${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_CHAT_RUST_OUTPUT}
        COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_CHAT_RUST_MANIFEST} --target-dir ${TEST_CHAT_RUST_TARGET_DIR} --bin test-chat
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_CHAT_RUST_BIN} ${TEST_CHAT_RUST_OUTPUT}
        DEPENDS
            ${TEST_CHAT_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-chat-rust/src/main.rs
    )
    add_custom_target(test-chat ALL DEPENDS ${TEST_CHAT_RUST_OUTPUT})
    add_test(NAME test-chat WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} COMMAND ${TEST_CHAT_RUST_OUTPUT})
    set_property(TEST test-chat PROPERTY LABELS "main")
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        set(TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-json-schema-to-grammar-rust/Cargo.toml)
        set(TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-json-schema-to-grammar)
        set(TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_BIN ${TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_TARGET_DIR}/debug/test-json-schema-to-grammar${CMAKE_EXECUTABLE_SUFFIX})
        set(TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-json-schema-to-grammar${CMAKE_EXECUTABLE_SUFFIX})

        add_custom_command(
            OUTPUT ${TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_OUTPUT}
            COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_MANIFEST} --target-dir ${TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_TARGET_DIR} --bin test-json-schema-to-grammar
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_BIN} ${TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_OUTPUT}
            DEPENDS
                ${TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_MANIFEST}
                ${CMAKE_CURRENT_SOURCE_DIR}/test-json-schema-to-grammar-rust/src/main.rs
        )
        add_custom_target(test-json-schema-to-grammar ALL DEPENDS ${TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_OUTPUT})
        add_test(NAME test-json-schema-to-grammar WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} COMMAND ${TEST_JSON_SCHEMA_TO_GRAMMAR_RUST_OUTPUT})
        set_property(TEST test-json-schema-to-grammar PROPERTY LABELS "main")
    endif()

    if (NOT GGML_BACKEND_DL)
        set(TEST_QUANTIZE_STATS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-stats-rust/Cargo.toml)
        set(TEST_QUANTIZE_STATS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-quantize-stats)
        set(TEST_QUANTIZE_STATS_RUST_BIN ${TEST_QUANTIZE_STATS_RUST_TARGET_DIR}/debug/test-quantize-stats${CMAKE_EXECUTABLE_SUFFIX})
        set(TEST_QUANTIZE_STATS_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-quantize-stats${CMAKE_EXECUTABLE_SUFFIX})

        add_custom_command(
            OUTPUT ${TEST_QUANTIZE_STATS_RUST_OUTPUT}
            COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_QUANTIZE_STATS_RUST_MANIFEST} --target-dir ${TEST_QUANTIZE_STATS_RUST_TARGET_DIR} --bin test-quantize-stats
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_QUANTIZE_STATS_RUST_BIN} ${TEST_QUANTIZE_STATS_RUST_OUTPUT}
            DEPENDS
                ${TEST_QUANTIZE_STATS_RUST_MANIFEST}
                ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-stats-rust/src/main.rs
        )
        add_custom_target(test-quantize-stats ALL DEPENDS ${TEST_QUANTIZE_STATS_RUST_OUTPUT})
    endif()

    # build test-tokenizer-1-bpe target once and add many tests
    set(TEST_TOKENIZER_1_BPE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-1-bpe-rust/Cargo.toml)
    set(TEST_TOKENIZER_1_BPE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-tokenizer-1-bpe)
    set(TEST_TOKENIZER_1_BPE_RUST_BIN ${TEST_TOKENIZER_1_BPE_RUST_TARGET_DIR}/debug/test-tokenizer-1-bpe${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_TOKENIZER_1_BPE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-tokenizer-1-bpe${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_TOKENIZER_1_BPE_RUST_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
                ${CARGO_EXECUTABLE} build --manifest-path ${TEST_TOKENIZER_1_BPE_RUST_MANIFEST} --target-dir ${TEST_TOKENIZER_1_BPE_RUST_TARGET_DIR} --bin test-tokenizer-1-bpe
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_TOKENIZER_1_BPE_RUST_BIN} ${TEST_TOKENIZER_1_BPE_RUST_OUTPUT}
        DEPENDS
            llama
            ${TEST_TOKENIZER_1_BPE_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-1-bpe-rust/build.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-1-bpe-rust/src/lib.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-1-bpe-rust/src/main.rs
            ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
            ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
            ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
    )
    add_custom_target(test-tokenizer-1-bpe ALL DEPENDS ${TEST_TOKENIZER_1_BPE_RUST_OUTPUT})

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    set(TEST_TOKENIZER_1_SPM_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-1-spm-rust/Cargo.toml)
    set(TEST_TOKENIZER_1_SPM_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-tokenizer-1-spm)
    set(TEST_TOKENIZER_1_SPM_RUST_BIN ${TEST_TOKENIZER_1_SPM_RUST_TARGET_DIR}/debug/test-tokenizer-1-spm${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_TOKENIZER_1_SPM_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-tokenizer-1-spm${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_TOKENIZER_1_SPM_RUST_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
                ${CARGO_EXECUTABLE} build --manifest-path ${TEST_TOKENIZER_1_SPM_RUST_MANIFEST} --target-dir ${TEST_TOKENIZER_1_SPM_RUST_TARGET_DIR} --bin test-tokenizer-1-spm
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_TOKENIZER_1_SPM_RUST_BIN} ${TEST_TOKENIZER_1_SPM_RUST_OUTPUT}
        DEPENDS
            llama
            ${TEST_TOKENIZER_1_SPM_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-1-spm-rust/build.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-1-spm-rust/src/lib.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizer-1-spm-rust/src/main.rs
            ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
            ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
            ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
    )
    add_custom_target(test-tokenizer-1-spm ALL DEPENDS ${TEST_TOKENIZER_1_SPM_RUST_OUTPUT})

    add_test(NAME test-tokenizer-1-llama-spm WORKING_DIRECTORY . COMMAND ${TEST_TOKENIZER_1_SPM_RUST_OUTPUT} ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    set_property(TEST test-tokenizer-1-llama-spm PROPERTY LABELS "main")
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # TODO: slow; run with `cargo test -p llama-test-double-float -- --ignored`

    set(TEST_LLAMA_ARCHS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-llama-archs-rust/Cargo.toml)
    set(TEST_LLAMA_ARCHS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-llama-archs)
    set(TEST_LLAMA_ARCHS_RUST_BIN ${TEST_LLAMA_ARCHS_RUST_TARGET_DIR}/debug/test-llama-archs${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_LLAMA_ARCHS_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-llama-archs${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_LLAMA_ARCHS_RUST_OUTPUT}
        COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_LLAMA_ARCHS_RUST_MANIFEST} --target-dir ${TEST_LLAMA_ARCHS_RUST_TARGET_DIR} --bin test-llama-archs
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_LLAMA_ARCHS_RUST_BIN} ${TEST_LLAMA_ARCHS_RUST_OUTPUT}
        DEPENDS
            ${TEST_LLAMA_ARCHS_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-llama-archs-rust/src/main.rs
    )
    add_custom_target(test-llama-archs ALL DEPENDS ${TEST_LLAMA_ARCHS_RUST_OUTPUT})
    add_test(NAME test-llama-archs WORKING_DIRECTORY . COMMAND ${TEST_LLAMA_ARCHS_RUST_OUTPUT})
    set_property(TEST test-llama-archs PROPERTY LABELS "main")
endif()

set(TEST_CHAT_PEG_PARSER_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-chat-peg-parser-rust/Cargo.toml)
set(TEST_CHAT_PEG_PARSER_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-chat-peg-parser)
set(TEST_CHAT_PEG_PARSER_RUST_BIN ${TEST_CHAT_PEG_PARSER_RUST_TARGET_DIR}/debug/test-chat-peg-parser${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_CHAT_PEG_PARSER_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-chat-peg-parser${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_CHAT_PEG_PARSER_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_CHAT_PEG_PARSER_RUST_MANIFEST} --target-dir ${TEST_CHAT_PEG_PARSER_RUST_TARGET_DIR} --bin test-chat-peg-parser
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_CHAT_PEG_PARSER_RUST_BIN} ${TEST_CHAT_PEG_PARSER_RUST_OUTPUT}
    DEPENDS
        ${TEST_CHAT_PEG_PARSER_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-chat-peg-parser-rust/src/main.rs
)
add_custom_target(test-chat-peg-parser ALL DEPENDS ${TEST_CHAT_PEG_PARSER_RUST_OUTPUT})
add_test(NAME test-chat-peg-parser WORKING_DIRECTORY . COMMAND ${TEST_CHAT_PEG_PARSER_RUST_OUTPUT})
set_property(TEST test-chat-peg-parser PROPERTY LABELS "main")
set(TEST_JINJA_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-jinja-rust/Cargo.toml)
set(TEST_JINJA_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-jinja)
set(TEST_JINJA_RUST_BIN ${TEST_JINJA_RUST_TARGET_DIR}/debug/test-jinja${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_JINJA_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-jinja${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_JINJA_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_JINJA_RUST_MANIFEST} --target-dir ${TEST_JINJA_RUST_TARGET_DIR} --bin test-jinja
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_JINJA_RUST_BIN} ${TEST_JINJA_RUST_OUTPUT}
    DEPENDS
        ${TEST_JINJA_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-jinja-rust/src/main.rs
)
add_custom_target(test-jinja ALL DEPENDS ${TEST_JINJA_RUST_OUTPUT})
add_test(NAME test-jinja WORKING_DIRECTORY . COMMAND ${TEST_JINJA_RUST_OUTPUT})
set_property(TEST test-jinja PROPERTY LABELS "main")
add_test(NAME test-jinja-py WORKING_DIRECTORY . COMMAND ${TEST_JINJA_RUST_OUTPUT} -py)
set_property(TEST test-jinja-py PROPERTY LABELS "python")
set(TEST_CHAT_AUTO_PARSER_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-chat-auto-parser-rust/Cargo.toml)
set(TEST_CHAT_AUTO_PARSER_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-chat-auto-parser)
set(TEST_CHAT_AUTO_PARSER_RUST_BIN ${TEST_CHAT_AUTO_PARSER_RUST_TARGET_DIR}/debug/test-chat-auto-parser${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_CHAT_AUTO_PARSER_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-chat-auto-parser${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_CHAT_AUTO_PARSER_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_CHAT_AUTO_PARSER_RUST_MANIFEST} --target-dir ${TEST_CHAT_AUTO_PARSER_RUST_TARGET_DIR} --bin test-chat-auto-parser
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_CHAT_AUTO_PARSER_RUST_BIN} ${TEST_CHAT_AUTO_PARSER_RUST_OUTPUT}
    DEPENDS
        ${TEST_CHAT_AUTO_PARSER_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-chat-auto-parser-rust/src/main.rs
)
add_custom_target(test-chat-auto-parser ALL DEPENDS ${TEST_CHAT_AUTO_PARSER_RUST_OUTPUT})
add_test(NAME test-chat-auto-parser WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} COMMAND ${TEST_CHAT_AUTO_PARSER_RUST_OUTPUT})
set_property(TEST test-chat-auto-parser PROPERTY LABELS "main")
set(TEST_CHAT_TEMPLATE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-chat-template-rust/Cargo.toml)
set(TEST_CHAT_TEMPLATE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-chat-template)
set(TEST_CHAT_TEMPLATE_RUST_BIN ${TEST_CHAT_TEMPLATE_RUST_TARGET_DIR}/debug/test-chat-template${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_CHAT_TEMPLATE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-chat-template${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_CHAT_TEMPLATE_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_CHAT_TEMPLATE_RUST_MANIFEST} --target-dir ${TEST_CHAT_TEMPLATE_RUST_TARGET_DIR} --bin test-chat-template
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_CHAT_TEMPLATE_RUST_BIN} ${TEST_CHAT_TEMPLATE_RUST_OUTPUT}
    DEPENDS
        ${TEST_CHAT_TEMPLATE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-chat-template-rust/src/main.rs
)
add_custom_target(test-chat-template ALL DEPENDS ${TEST_CHAT_TEMPLATE_RUST_OUTPUT})
add_test(NAME test-chat-template WORKING_DIRECTORY . COMMAND ${TEST_CHAT_TEMPLATE_RUST_OUTPUT})
set_property(TEST test-chat-template PROPERTY LABELS "main")
set(TEST_JSON_PARTIAL_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-json-partial-rust/Cargo.toml)
set(TEST_JSON_PARTIAL_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-json-partial)
set(TEST_JSON_PARTIAL_RUST_BIN ${TEST_JSON_PARTIAL_RUST_TARGET_DIR}/debug/test-json-partial${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_JSON_PARTIAL_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-json-partial${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_JSON_PARTIAL_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_JSON_PARTIAL_RUST_MANIFEST} --target-dir ${TEST_JSON_PARTIAL_RUST_TARGET_DIR} --bin test-json-partial
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_JSON_PARTIAL_RUST_BIN} ${TEST_JSON_PARTIAL_RUST_OUTPUT}
    DEPENDS
        ${TEST_JSON_PARTIAL_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-json-partial-rust/src/main.rs
)
add_custom_target(test-json-partial ALL DEPENDS ${TEST_JSON_PARTIAL_RUST_OUTPUT})
add_test(NAME test-json-partial WORKING_DIRECTORY . COMMAND ${TEST_JSON_PARTIAL_RUST_OUTPUT})
set_property(TEST test-json-partial PROPERTY LABELS "main")
find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(TEST_LOG_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-log-rust/Cargo.toml)
set(TEST_LOG_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-log)
set(TEST_LOG_RUST_BIN ${TEST_LOG_RUST_TARGET_DIR}/debug/test-log${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_LOG_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-log${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_GBNF_VALIDATOR_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-gbnf-validator-rust/Cargo.toml)
set(TEST_GBNF_VALIDATOR_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-gbnf-validator)
set(TEST_GBNF_VALIDATOR_RUST_BIN ${TEST_GBNF_VALIDATOR_RUST_TARGET_DIR}/debug/test-gbnf-validator${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_GBNF_VALIDATOR_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-gbnf-validator${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_BARRIER_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-barrier-rust/Cargo.toml)
set(TEST_BARRIER_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-barrier)
set(TEST_BARRIER_RUST_BIN ${TEST_BARRIER_RUST_TARGET_DIR}/debug/test-barrier${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_BARRIER_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-barrier${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_ROPE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-rope-rust/Cargo.toml)
set(TEST_ROPE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-rope)
set(TEST_ROPE_RUST_BIN ${TEST_ROPE_RUST_TARGET_DIR}/debug/test-rope${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_ROPE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-rope${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_REASONING_BUDGET_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-reasoning-budget-rust/Cargo.toml)
set(TEST_REASONING_BUDGET_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-reasoning-budget)
set(TEST_REASONING_BUDGET_RUST_BIN ${TEST_REASONING_BUDGET_RUST_TARGET_DIR}/debug/test-reasoning-budget${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_REASONING_BUDGET_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-reasoning-budget${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_PEG_PARSER_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-peg-parser-rust/Cargo.toml)
set(TEST_PEG_PARSER_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-peg-parser)
set(TEST_PEG_PARSER_RUST_LIB ${TEST_PEG_PARSER_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_test_peg_parser_rust${CMAKE_STATIC_LIBRARY_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_LOG_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_COMMON_LIB_DIR=$<TARGET_FILE_DIR:llama-common>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_LOG_RUST_MANIFEST} --target-dir ${TEST_LOG_RUST_TARGET_DIR} --bin test-log
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_LOG_RUST_BIN} ${TEST_LOG_RUST_OUTPUT}
    DEPENDS
        llama-common
        ${TEST_LOG_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-log-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-log-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-log-rust/src/main.rs
)
add_custom_target(test-log ALL DEPENDS ${TEST_LOG_RUST_OUTPUT})
add_test(NAME test-log WORKING_DIRECTORY . COMMAND ${TEST_LOG_RUST_OUTPUT})
set_property(TEST test-log PROPERTY LABELS "main")

add_custom_command(
    OUTPUT ${TEST_GBNF_VALIDATOR_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_GBNF_VALIDATOR_RUST_MANIFEST} --target-dir ${TEST_GBNF_VALIDATOR_RUST_TARGET_DIR} --bin test-gbnf-validator
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_GBNF_VALIDATOR_RUST_BIN} ${TEST_GBNF_VALIDATOR_RUST_OUTPUT}
    DEPENDS
        llama
        ${TEST_GBNF_VALIDATOR_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-gbnf-validator-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-gbnf-validator-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-gbnf-validator-rust/src/main.rs
)
add_custom_target(test-gbnf-validator ALL DEPENDS ${TEST_GBNF_VALIDATOR_RUST_OUTPUT})

add_custom_command(
    OUTPUT ${TEST_BARRIER_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:ggml>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_BARRIER_RUST_MANIFEST} --target-dir ${TEST_BARRIER_RUST_TARGET_DIR} --bin test-barrier
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_BARRIER_RUST_BIN} ${TEST_BARRIER_RUST_OUTPUT}
    DEPENDS
        ggml
        ggml-cpu
        ${TEST_BARRIER_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-barrier-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-barrier-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-barrier-rust/src/main.rs
)
add_custom_target(test-barrier ALL DEPENDS ${TEST_BARRIER_RUST_OUTPUT})

add_custom_command(
    OUTPUT ${TEST_ROPE_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:ggml>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_ROPE_RUST_MANIFEST} --target-dir ${TEST_ROPE_RUST_TARGET_DIR} --bin test-rope
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_ROPE_RUST_BIN} ${TEST_ROPE_RUST_OUTPUT}
    DEPENDS
        ggml
        ggml-cpu
        ${TEST_ROPE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-rope-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-rope-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-rope-rust/src/main.rs
)
add_custom_target(test-rope ALL DEPENDS ${TEST_ROPE_RUST_OUTPUT})

add_custom_command(
    OUTPUT ${TEST_REASONING_BUDGET_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_COMMON_LIB_DIR=$<TARGET_FILE_DIR:llama-common>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_REASONING_BUDGET_RUST_MANIFEST} --target-dir ${TEST_REASONING_BUDGET_RUST_TARGET_DIR} --bin test-reasoning-budget
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_REASONING_BUDGET_RUST_BIN} ${TEST_REASONING_BUDGET_RUST_OUTPUT}
    DEPENDS
        llama-common
        ${TEST_REASONING_BUDGET_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-reasoning-budget-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-reasoning-budget-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-reasoning-budget-rust/src/main.rs
)
add_custom_target(test-reasoning-budget ALL DEPENDS ${TEST_REASONING_BUDGET_RUST_OUTPUT})
add_test(NAME test-reasoning-budget WORKING_DIRECTORY . COMMAND ${TEST_REASONING_BUDGET_RUST_OUTPUT})
set_property(TEST test-reasoning-budget PROPERTY LABELS "main")

set(TEST_PEG_PARSER_RUST_BIN ${TEST_PEG_PARSER_RUST_TARGET_DIR}/debug/test-peg-parser${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_PEG_PARSER_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-peg-parser${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_PEG_PARSER_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_PEG_PARSER_RUST_MANIFEST} --target-dir ${TEST_PEG_PARSER_RUST_TARGET_DIR} --bin test-peg-parser
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_PEG_PARSER_RUST_BIN} ${TEST_PEG_PARSER_RUST_OUTPUT}
    DEPENDS
        ${TEST_PEG_PARSER_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-peg-parser-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-peg-parser-rust/src/main.rs
)
add_custom_target(test-peg-parser ALL DEPENDS ${TEST_PEG_PARSER_RUST_OUTPUT})
add_test(NAME test-peg-parser WORKING_DIRECTORY . COMMAND ${TEST_PEG_PARSER_RUST_OUTPUT})
set_property(TEST test-peg-parser PROPERTY LABELS "main")
set(TEST_REGEX_PARTIAL_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-regex-partial-rust/Cargo.toml)
set(TEST_REGEX_PARTIAL_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-regex-partial)
set(TEST_REGEX_PARTIAL_RUST_BIN ${TEST_REGEX_PARTIAL_RUST_TARGET_DIR}/debug/test-regex-partial${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_REGEX_PARTIAL_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-regex-partial${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_REGEX_PARTIAL_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_REGEX_PARTIAL_RUST_MANIFEST} --target-dir ${TEST_REGEX_PARTIAL_RUST_TARGET_DIR} --bin test-regex-partial
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_REGEX_PARTIAL_RUST_BIN} ${TEST_REGEX_PARTIAL_RUST_OUTPUT}
    DEPENDS
        ${TEST_REGEX_PARTIAL_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-regex-partial-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-regex-partial-rust/src/main.rs
        ${PROJECT_SOURCE_DIR}/common/regex-partial-rust/Cargo.toml
        ${PROJECT_SOURCE_DIR}/common/regex-partial-rust/src/lib.rs
)
add_custom_target(test-regex-partial ALL DEPENDS ${TEST_REGEX_PARTIAL_RUST_OUTPUT})
add_test(NAME test-regex-partial WORKING_DIRECTORY . COMMAND ${TEST_REGEX_PARTIAL_RUST_OUTPUT})
set_property(TEST test-regex-partial PROPERTY LABELS "main")

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

set(TEST_THREAD_SAFETY_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-thread-safety-rust/Cargo.toml)
set(TEST_THREAD_SAFETY_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-thread-safety)
set(TEST_THREAD_SAFETY_RUST_BIN ${TEST_THREAD_SAFETY_RUST_TARGET_DIR}/debug/test-thread-safety${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_THREAD_SAFETY_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-thread-safety${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_THREAD_SAFETY_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_THREAD_SAFETY_RUST_MANIFEST} --target-dir ${TEST_THREAD_SAFETY_RUST_TARGET_DIR} --bin test-thread-safety
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_THREAD_SAFETY_RUST_BIN} ${TEST_THREAD_SAFETY_RUST_OUTPUT}
    DEPENDS
        llama
        ${TEST_THREAD_SAFETY_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-thread-safety-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-thread-safety-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-thread-safety-rust/src/main.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)
add_custom_target(test-thread-safety ALL DEPENDS ${TEST_THREAD_SAFETY_RUST_OUTPUT})
add_test(NAME test-thread-safety WORKING_DIRECTORY . COMMAND ${TEST_THREAD_SAFETY_RUST_OUTPUT} -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_property(TEST test-thread-safety PROPERTY LABELS "main")
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

set(TEST_ARG_PARSER_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-arg-parser-rust/Cargo.toml)
set(TEST_ARG_PARSER_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-arg-parser)
set(TEST_ARG_PARSER_RUST_BIN ${TEST_ARG_PARSER_RUST_TARGET_DIR}/debug/test-arg-parser${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_ARG_PARSER_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-arg-parser${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_ARG_PARSER_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_ARG_PARSER_RUST_MANIFEST} --target-dir ${TEST_ARG_PARSER_RUST_TARGET_DIR} --bin test-arg-parser
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_ARG_PARSER_RUST_BIN} ${TEST_ARG_PARSER_RUST_OUTPUT}
    DEPENDS
        ${TEST_ARG_PARSER_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-arg-parser-rust/src/main.rs
)
add_custom_target(test-arg-parser ALL DEPENDS ${TEST_ARG_PARSER_RUST_OUTPUT})
add_test(NAME test-arg-parser WORKING_DIRECTORY . COMMAND ${TEST_ARG_PARSER_RUST_OUTPUT})
set_property(TEST test-arg-parser PROPERTY LABELS "main")

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  set(TEST_OPT_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-opt-rust/Cargo.toml)
  set(TEST_OPT_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-opt)
  set(TEST_OPT_RUST_BIN ${TEST_OPT_RUST_TARGET_DIR}/debug/test-opt${CMAKE_EXECUTABLE_SUFFIX})
  set(TEST_OPT_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-opt${CMAKE_EXECUTABLE_SUFFIX})

  add_custom_command(
      OUTPUT ${TEST_OPT_RUST_OUTPUT}
      COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_OPT_RUST_MANIFEST} --target-dir ${TEST_OPT_RUST_TARGET_DIR} --bin test-opt
      COMMAND ${CMAKE_COMMAND} -E copy ${TEST_OPT_RUST_BIN} ${TEST_OPT_RUST_OUTPUT}
      DEPENDS
          ${TEST_OPT_RUST_MANIFEST}
          ${CMAKE_CURRENT_SOURCE_DIR}/test-opt-rust/src/main.rs
  )
  add_custom_target(test-opt ALL DEPENDS ${TEST_OPT_RUST_OUTPUT})
  add_test(NAME test-opt WORKING_DIRECTORY . COMMAND ${TEST_OPT_RUST_OUTPUT})
  set_property(TEST test-opt PROPERTY LABELS "main")
endif()
set(TEST_GGUF_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-gguf-rust/Cargo.toml)
set(TEST_GGUF_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-gguf)
set(TEST_GGUF_RUST_BIN ${TEST_GGUF_RUST_TARGET_DIR}/debug/test-gguf${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_GGUF_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-gguf${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_GGUF_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_GGUF_RUST_MANIFEST} --target-dir ${TEST_GGUF_RUST_TARGET_DIR} --bin test-gguf
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_GGUF_RUST_BIN} ${TEST_GGUF_RUST_OUTPUT}
    DEPENDS
        ${TEST_GGUF_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-gguf-rust/src/main.rs
)
add_custom_target(test-gguf ALL DEPENDS ${TEST_GGUF_RUST_OUTPUT})
add_test(NAME test-gguf WORKING_DIRECTORY . COMMAND ${TEST_GGUF_RUST_OUTPUT})
set_property(TEST test-gguf PROPERTY LABELS "main")
set(TEST_BACKEND_OPS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-backend-ops-rust/Cargo.toml)
set(TEST_BACKEND_OPS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-backend-ops)
set(TEST_BACKEND_OPS_RUST_BIN ${TEST_BACKEND_OPS_RUST_TARGET_DIR}/debug/test-backend-ops${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_BACKEND_OPS_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-backend-ops${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_BACKEND_OPS_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_BACKEND_OPS_RUST_MANIFEST} --target-dir ${TEST_BACKEND_OPS_RUST_TARGET_DIR} --bin test-backend-ops
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_BACKEND_OPS_RUST_BIN} ${TEST_BACKEND_OPS_RUST_OUTPUT}
    DEPENDS
        ${TEST_BACKEND_OPS_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-backend-ops-rust/src/main.rs
)
add_custom_target(test-backend-ops ALL DEPENDS ${TEST_BACKEND_OPS_RUST_OUTPUT})
add_test(NAME test-backend-ops WORKING_DIRECTORY . COMMAND ${TEST_BACKEND_OPS_RUST_OUTPUT})
set_property(TEST test-backend-ops PROPERTY LABELS "main")

find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(TEST_MODEL_LOAD_CANCEL_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-model-load-cancel-rust/Cargo.toml)
set(TEST_MODEL_LOAD_CANCEL_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-model-load-cancel)
set(TEST_MODEL_LOAD_CANCEL_RUST_BIN ${TEST_MODEL_LOAD_CANCEL_RUST_TARGET_DIR}/debug/test-model-load-cancel${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_MODEL_LOAD_CANCEL_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-model-load-cancel${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_MODEL_LOAD_CANCEL_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_MODEL_LOAD_CANCEL_RUST_MANIFEST} --target-dir ${TEST_MODEL_LOAD_CANCEL_RUST_TARGET_DIR} --bin test-model-load-cancel
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_MODEL_LOAD_CANCEL_RUST_BIN} ${TEST_MODEL_LOAD_CANCEL_RUST_OUTPUT}
    DEPENDS
        llama
        ${TEST_MODEL_LOAD_CANCEL_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-model-load-cancel-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-model-load-cancel-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-model-load-cancel-rust/src/main.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)
add_custom_target(test-model-load-cancel ALL DEPENDS ${TEST_MODEL_LOAD_CANCEL_RUST_OUTPUT})
add_test(NAME test-model-load-cancel WORKING_DIRECTORY . COMMAND ${TEST_MODEL_LOAD_CANCEL_RUST_OUTPUT})
set_property(TEST test-model-load-cancel PROPERTY LABELS "model")
find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(TEST_AUTORELEASE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-autorelease-rust/Cargo.toml)
set(TEST_AUTORELEASE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-autorelease)
set(TEST_AUTORELEASE_RUST_BIN ${TEST_AUTORELEASE_RUST_TARGET_DIR}/debug/test-autorelease${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_AUTORELEASE_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-autorelease${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_STATE_RESTORE_FRAGMENTED_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-state-restore-fragmented-rust/Cargo.toml)
set(TEST_STATE_RESTORE_FRAGMENTED_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-state-restore-fragmented)
set(TEST_STATE_RESTORE_FRAGMENTED_RUST_BIN ${TEST_STATE_RESTORE_FRAGMENTED_RUST_TARGET_DIR}/debug/test-state-restore-fragmented${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_STATE_RESTORE_FRAGMENTED_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-state-restore-fragmented${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_AUTORELEASE_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_AUTORELEASE_RUST_MANIFEST} --target-dir ${TEST_AUTORELEASE_RUST_TARGET_DIR} --bin test-autorelease
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_AUTORELEASE_RUST_BIN} ${TEST_AUTORELEASE_RUST_OUTPUT}
    DEPENDS
        llama
        ${TEST_AUTORELEASE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-autorelease-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-autorelease-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-autorelease-rust/src/main.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)
add_custom_target(test-autorelease ALL DEPENDS ${TEST_AUTORELEASE_RUST_OUTPUT})
add_test(NAME test-autorelease WORKING_DIRECTORY . COMMAND ${TEST_AUTORELEASE_RUST_OUTPUT})
set_property(TEST test-autorelease PROPERTY LABELS "model")

add_custom_command(
    OUTPUT ${TEST_STATE_RESTORE_FRAGMENTED_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_STATE_RESTORE_FRAGMENTED_RUST_MANIFEST} --target-dir ${TEST_STATE_RESTORE_FRAGMENTED_RUST_TARGET_DIR} --bin test-state-restore-fragmented
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_STATE_RESTORE_FRAGMENTED_RUST_BIN} ${TEST_STATE_RESTORE_FRAGMENTED_RUST_OUTPUT}
    DEPENDS
        llama
        ${TEST_STATE_RESTORE_FRAGMENTED_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-state-restore-fragmented-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-state-restore-fragmented-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-state-restore-fragmented-rust/src/main.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/build.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/ffi.rs
        ${PROJECT_SOURCE_DIR}/examples/simple-rust/src/lib.rs
)
add_custom_target(test-state-restore-fragmented ALL DEPENDS ${TEST_STATE_RESTORE_FRAGMENTED_RUST_OUTPUT})
add_test(NAME test-state-restore-fragmented WORKING_DIRECTORY . COMMAND ${TEST_STATE_RESTORE_FRAGMENTED_RUST_OUTPUT} -m "${MODEL_DEST}")
set_property(TEST test-state-restore-fragmented PROPERTY LABELS "model")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)
set(TEST_BACKEND_SAMPLER_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-backend-sampler-rust/Cargo.toml)
set(TEST_BACKEND_SAMPLER_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-backend-sampler)
set(TEST_BACKEND_SAMPLER_RUST_BIN ${TEST_BACKEND_SAMPLER_RUST_TARGET_DIR}/debug/test-backend-sampler${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_BACKEND_SAMPLER_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-backend-sampler${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_BACKEND_SAMPLER_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_BACKEND_SAMPLER_RUST_MANIFEST} --target-dir ${TEST_BACKEND_SAMPLER_RUST_TARGET_DIR} --bin test-backend-sampler
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_BACKEND_SAMPLER_RUST_BIN} ${TEST_BACKEND_SAMPLER_RUST_OUTPUT}
    DEPENDS
        ${TEST_BACKEND_SAMPLER_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-backend-sampler-rust/src/main.rs
)
add_custom_target(test-backend-sampler ALL DEPENDS ${TEST_BACKEND_SAMPLER_RUST_OUTPUT})
add_test(NAME test-backend-sampler WORKING_DIRECTORY . COMMAND ${TEST_BACKEND_SAMPLER_RUST_OUTPUT})
set_property(TEST test-backend-sampler PROPERTY LABELS "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    add_test(NAME test-barrier WORKING_DIRECTORY . COMMAND ${TEST_BARRIER_RUST_OUTPUT})
    set_property(TEST test-barrier PROPERTY LABELS "main")
    set(TEST_QUANTIZE_FNS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-fns-rust/Cargo.toml)
    set(TEST_QUANTIZE_FNS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-quantize-fns)
    set(TEST_QUANTIZE_FNS_RUST_BIN ${TEST_QUANTIZE_FNS_RUST_TARGET_DIR}/debug/test-quantize-fns${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_QUANTIZE_FNS_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-quantize-fns${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_QUANTIZE_FNS_RUST_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:ggml>
                ${CARGO_EXECUTABLE} build --manifest-path ${TEST_QUANTIZE_FNS_RUST_MANIFEST} --target-dir ${TEST_QUANTIZE_FNS_RUST_TARGET_DIR} --bin test-quantize-fns
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_QUANTIZE_FNS_RUST_BIN} ${TEST_QUANTIZE_FNS_RUST_OUTPUT}
        DEPENDS
            ggml
            ggml-cpu
            ${TEST_QUANTIZE_FNS_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-fns-rust/build.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-fns-rust/src/lib.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-fns-rust/src/main.rs
    )
    add_custom_target(test-quantize-fns ALL DEPENDS ${TEST_QUANTIZE_FNS_RUST_OUTPUT})
    add_test(NAME test-quantize-fns WORKING_DIRECTORY . COMMAND ${TEST_QUANTIZE_FNS_RUST_OUTPUT})
    set_property(TEST test-quantize-fns PROPERTY LABELS "main")

    set(TEST_QUANTIZE_PERF_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-perf-rust/Cargo.toml)
    set(TEST_QUANTIZE_PERF_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-quantize-perf)
    set(TEST_QUANTIZE_PERF_RUST_BIN ${TEST_QUANTIZE_PERF_RUST_TARGET_DIR}/debug/test-quantize-perf${CMAKE_EXECUTABLE_SUFFIX})
    set(TEST_QUANTIZE_PERF_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-quantize-perf${CMAKE_EXECUTABLE_SUFFIX})

    add_custom_command(
        OUTPUT ${TEST_QUANTIZE_PERF_RUST_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:ggml>
                ${CARGO_EXECUTABLE} build --manifest-path ${TEST_QUANTIZE_PERF_RUST_MANIFEST} --target-dir ${TEST_QUANTIZE_PERF_RUST_TARGET_DIR} --bin test-quantize-perf
        COMMAND ${CMAKE_COMMAND} -E copy ${TEST_QUANTIZE_PERF_RUST_BIN} ${TEST_QUANTIZE_PERF_RUST_OUTPUT}
        DEPENDS
            ggml
            ggml-cpu
            ${TEST_QUANTIZE_PERF_RUST_MANIFEST}
            ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-perf-rust/build.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-perf-rust/src/lib.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-perf-rust/src/main.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-fns-rust/Cargo.toml
            ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-fns-rust/build.rs
            ${CMAKE_CURRENT_SOURCE_DIR}/test-quantize-fns-rust/src/lib.rs
    )
    add_custom_target(test-quantize-perf ALL DEPENDS ${TEST_QUANTIZE_PERF_RUST_OUTPUT})
    add_test(NAME test-quantize-perf WORKING_DIRECTORY . COMMAND ${TEST_QUANTIZE_PERF_RUST_OUTPUT} --type q4_0 --op quantize_row_q --size 4096 -i 1)
    set_property(TEST test-quantize-perf PROPERTY LABELS "main")
    add_test(NAME test-rope WORKING_DIRECTORY . COMMAND ${TEST_ROPE_RUST_OUTPUT})
    set_property(TEST test-rope PROPERTY LABELS "main")
endif()

# libmtmd
find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(TEST_MTMD_C_API_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-mtmd-c-api-rust/Cargo.toml)
set(TEST_MTMD_C_API_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-mtmd-c-api)
set(TEST_MTMD_C_API_RUST_BIN ${TEST_MTMD_C_API_RUST_TARGET_DIR}/debug/test-mtmd-c-api${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_MTMD_C_API_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-mtmd-c-api${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_MTMD_C_API_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env MTMD_LIB_DIR=$<TARGET_FILE_DIR:mtmd>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_MTMD_C_API_RUST_MANIFEST} --target-dir ${TEST_MTMD_C_API_RUST_TARGET_DIR} --bin test-mtmd-c-api
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_MTMD_C_API_RUST_BIN} ${TEST_MTMD_C_API_RUST_OUTPUT}
    DEPENDS
        mtmd
        ${TEST_MTMD_C_API_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-mtmd-c-api-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-mtmd-c-api-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-mtmd-c-api-rust/src/main.rs
)
add_custom_target(test-mtmd-c-api ALL DEPENDS ${TEST_MTMD_C_API_RUST_OUTPUT})
add_test(NAME test-mtmd-c-api WORKING_DIRECTORY . COMMAND ${TEST_MTMD_C_API_RUST_OUTPUT})
set_property(TEST test-mtmd-c-api PROPERTY LABELS "main")

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        set(TEST_GGUF_MODEL_DATA_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-gguf-model-data-rust/Cargo.toml)
        set(TEST_GGUF_MODEL_DATA_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-gguf-model-data)
        set(TEST_GGUF_MODEL_DATA_RUST_BIN ${TEST_GGUF_MODEL_DATA_RUST_TARGET_DIR}/debug/test-gguf-model-data${CMAKE_EXECUTABLE_SUFFIX})
        set(TEST_GGUF_MODEL_DATA_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-gguf-model-data${CMAKE_EXECUTABLE_SUFFIX})

        add_custom_command(
            OUTPUT ${TEST_GGUF_MODEL_DATA_RUST_OUTPUT}
            COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_GGUF_MODEL_DATA_RUST_MANIFEST} --target-dir ${TEST_GGUF_MODEL_DATA_RUST_TARGET_DIR} --bin test-gguf-model-data
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_GGUF_MODEL_DATA_RUST_BIN} ${TEST_GGUF_MODEL_DATA_RUST_OUTPUT}
            DEPENDS
                ${TEST_GGUF_MODEL_DATA_RUST_MANIFEST}
                ${CMAKE_CURRENT_SOURCE_DIR}/test-gguf-model-data-rust/src/lib.rs
                ${CMAKE_CURRENT_SOURCE_DIR}/test-gguf-model-data-rust/src/main.rs
        )
        add_custom_target(test-gguf-model-data ALL DEPENDS ${TEST_GGUF_MODEL_DATA_RUST_OUTPUT})
        add_test(NAME test-gguf-model-data WORKING_DIRECTORY . COMMAND ${TEST_GGUF_MODEL_DATA_RUST_OUTPUT})
        set_property(TEST test-gguf-model-data PROPERTY LABELS "model")

        set(TEST_QUANT_TYPE_SELECTION_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-quant-type-selection-rust/Cargo.toml)
        set(TEST_QUANT_TYPE_SELECTION_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-quant-type-selection)
        set(TEST_QUANT_TYPE_SELECTION_RUST_BIN ${TEST_QUANT_TYPE_SELECTION_RUST_TARGET_DIR}/debug/test-quant-type-selection${CMAKE_EXECUTABLE_SUFFIX})
        set(TEST_QUANT_TYPE_SELECTION_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-quant-type-selection${CMAKE_EXECUTABLE_SUFFIX})

        add_custom_command(
            OUTPUT ${TEST_QUANT_TYPE_SELECTION_RUST_OUTPUT}
            COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${TEST_QUANT_TYPE_SELECTION_RUST_MANIFEST} --target-dir ${TEST_QUANT_TYPE_SELECTION_RUST_TARGET_DIR} --bin test-quant-type-selection
            COMMAND ${CMAKE_COMMAND} -E copy ${TEST_QUANT_TYPE_SELECTION_RUST_BIN} ${TEST_QUANT_TYPE_SELECTION_RUST_OUTPUT}
            DEPENDS
                ${TEST_QUANT_TYPE_SELECTION_RUST_MANIFEST}
                ${CMAKE_CURRENT_SOURCE_DIR}/test-quant-type-selection-rust/src/main.rs
        )
        add_custom_target(test-quant-type-selection ALL DEPENDS ${TEST_QUANT_TYPE_SELECTION_RUST_OUTPUT})
        add_test(NAME test-quant-type-selection WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMAND ${TEST_QUANT_TYPE_SELECTION_RUST_OUTPUT} --snapshot-dir ${CMAKE_CURRENT_SOURCE_DIR}/snapshots)
        set_property(TEST test-quant-type-selection PROPERTY LABELS "model")
    endif()
endif()

set(TEST_ALLOC_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/test-alloc-rust/Cargo.toml)
set(TEST_ALLOC_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/test-alloc)
set(TEST_ALLOC_RUST_BIN ${TEST_ALLOC_RUST_TARGET_DIR}/debug/test-alloc${CMAKE_EXECUTABLE_SUFFIX})
set(TEST_ALLOC_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test-alloc${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${TEST_ALLOC_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:ggml>
            ${CARGO_EXECUTABLE} build --manifest-path ${TEST_ALLOC_RUST_MANIFEST} --target-dir ${TEST_ALLOC_RUST_TARGET_DIR} --bin test-alloc
    COMMAND ${CMAKE_COMMAND} -E copy ${TEST_ALLOC_RUST_BIN} ${TEST_ALLOC_RUST_OUTPUT}
    DEPENDS
        ggml
        ggml-base
        ${TEST_ALLOC_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/test-alloc-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-alloc-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/test-alloc-rust/src/main.rs
)
add_custom_target(test-alloc ALL DEPENDS ${TEST_ALLOC_RUST_OUTPUT})
add_test(NAME test-alloc WORKING_DIRECTORY . COMMAND ${TEST_ALLOC_RUST_OUTPUT})
set_property(TEST test-alloc PROPERTY LABELS "main")

set(EXPORT_GRAPH_OPS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/export-graph-ops-rust/Cargo.toml)
set(EXPORT_GRAPH_OPS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/export-graph-ops)
set(EXPORT_GRAPH_OPS_RUST_BIN ${EXPORT_GRAPH_OPS_RUST_TARGET_DIR}/debug/export-graph-ops${CMAKE_EXECUTABLE_SUFFIX})
set(EXPORT_GRAPH_OPS_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/export-graph-ops${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${EXPORT_GRAPH_OPS_RUST_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${EXPORT_GRAPH_OPS_RUST_MANIFEST} --target-dir ${EXPORT_GRAPH_OPS_RUST_TARGET_DIR} --bin export-graph-ops
    COMMAND ${CMAKE_COMMAND} -E copy ${EXPORT_GRAPH_OPS_RUST_BIN} ${EXPORT_GRAPH_OPS_RUST_OUTPUT}
    DEPENDS
        ${EXPORT_GRAPH_OPS_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/export-graph-ops-rust/src/main.rs
)
add_custom_target(export-graph-ops ALL DEPENDS ${EXPORT_GRAPH_OPS_RUST_OUTPUT})
