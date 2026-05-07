set(TARGET llama-eval-callback)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(EVAL_CALLBACK_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/../eval-callback-rust/Cargo.toml)
set(EVAL_CALLBACK_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/eval-callback)
set(EVAL_CALLBACK_RUST_BIN ${EVAL_CALLBACK_RUST_TARGET_DIR}/debug/llama-eval-callback${CMAKE_EXECUTABLE_SUFFIX})
set(EVAL_CALLBACK_RUST_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${EVAL_CALLBACK_RUST_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E env LLAMA_CPP_LIB_DIR=$<TARGET_FILE_DIR:llama>
            ${CARGO_EXECUTABLE} build --manifest-path ${EVAL_CALLBACK_RUST_MANIFEST} --target-dir ${EVAL_CALLBACK_RUST_TARGET_DIR} --bin llama-eval-callback
    COMMAND ${CMAKE_COMMAND} -E copy ${EVAL_CALLBACK_RUST_BIN} ${EVAL_CALLBACK_RUST_OUTPUT}
    DEPENDS
        llama
        ${EVAL_CALLBACK_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/../eval-callback-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../eval-callback-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../eval-callback-rust/src/main.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/ffi.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/../simple-rust/src/lib.rs
)

add_custom_target(${TARGET} ALL DEPENDS ${EVAL_CALLBACK_RUST_OUTPUT})
install(PROGRAMS ${EVAL_CALLBACK_RUST_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})

if(LLAMA_BUILD_TESTS)
    if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
        set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
        set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
    else()
        set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
        set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
    endif()
    set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")
    set(TEST_TARGET test-eval-callback)
    add_test(NAME ${TEST_TARGET}-download-model COMMAND ${CMAKE_COMMAND}
        -DDEST=${MODEL_DEST}
        -DNAME=${MODEL_NAME}
        -DHASH=${MODEL_HASH}
        -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
    )
    set_tests_properties(${TEST_TARGET}-download-model PROPERTIES FIXTURES_SETUP ${TEST_TARGET}-download-model)
    add_test(NAME ${TEST_TARGET} COMMAND llama-eval-callback -m "${MODEL_DEST}" --prompt hello --seed 42 -ngl 0)
    set_tests_properties(${TEST_TARGET} PROPERTIES FIXTURES_REQUIRED ${TEST_TARGET}-download-model)
endif()
