# mtmd

find_package(Threads REQUIRED)

add_library(mtmd
            mtmd.cpp.inc
            mtmd.h.inc
            )
set_source_files_properties(mtmd.cpp.inc PROPERTIES LANGUAGE CXX)
set_source_files_properties(mtmd.h.inc PROPERTIES HEADER_FILE_ONLY TRUE)

set_target_properties(mtmd PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_link_libraries     (mtmd PUBLIC ggml llama)
target_link_libraries     (mtmd PRIVATE Threads::Threads)
target_include_directories(mtmd PUBLIC  .)
target_include_directories(mtmd PRIVATE ../..)
target_include_directories(mtmd PRIVATE ../../vendor)
target_compile_features   (mtmd PRIVATE cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (mtmd PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mtmd PRIVATE LLAMA_BUILD)
    target_compile_definitions(mtmd PUBLIC  LLAMA_SHARED)
endif()

set(MTMD_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtmd.h.inc
    )

set_target_properties(mtmd
    PROPERTIES
    PUBLIC_HEADER "${MTMD_PUBLIC_HEADERS}")

install(TARGETS mtmd LIBRARY PUBLIC_HEADER)

if (NOT MSVC)
    # for stb_image.h.inc and miniaudio.h.inc
    target_compile_options(mtmd PRIVATE -Wno-cast-qual)
endif()

if (ANDROID)
    # miniaudio.h.inc defines ma_android_sdk_version() without a prior prototype
    target_compile_options(mtmd PRIVATE -Wno-missing-prototypes)
endif()

if (TARGET BUILD_INFO)
    add_dependencies(mtmd        BUILD_INFO)
    add_dependencies(mtmd-helper BUILD_INFO)
endif()

# if mtmd is linked against llama-common, we throw an error
if (TARGET mtmd)
    get_target_property(libs mtmd LINK_LIBRARIES)
    if (libs AND "llama-common" IN_LIST libs)
        message(FATAL_ERROR "mtmd is designed to be a public library.\n"
                            "It must not link against llama-common")
    endif()
endif()

find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(MTMD_DEPRECATION_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/deprecation-warning-rust/Cargo.toml)
set(MTMD_DEPRECATION_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/mtmd-deprecation-warning)
set(MTMD_DEPRECATION_BIN ${MTMD_DEPRECATION_TARGET_DIR}/debug/mtmd-deprecation-warning${CMAKE_EXECUTABLE_SUFFIX})
set(MTMD_DEPRECATION_OUTPUTS
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-llava-cli${CMAKE_EXECUTABLE_SUFFIX}
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-gemma3-cli${CMAKE_EXECUTABLE_SUFFIX}
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-minicpmv-cli${CMAKE_EXECUTABLE_SUFFIX}
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-qwen2vl-cli${CMAKE_EXECUTABLE_SUFFIX}
)

add_custom_command(
    OUTPUT ${MTMD_DEPRECATION_OUTPUTS}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${MTMD_DEPRECATION_MANIFEST} --target-dir ${MTMD_DEPRECATION_TARGET_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${MTMD_DEPRECATION_BIN} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-llava-cli${CMAKE_EXECUTABLE_SUFFIX}
    COMMAND ${CMAKE_COMMAND} -E copy ${MTMD_DEPRECATION_BIN} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-gemma3-cli${CMAKE_EXECUTABLE_SUFFIX}
    COMMAND ${CMAKE_COMMAND} -E copy ${MTMD_DEPRECATION_BIN} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-minicpmv-cli${CMAKE_EXECUTABLE_SUFFIX}
    COMMAND ${CMAKE_COMMAND} -E copy ${MTMD_DEPRECATION_BIN} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-qwen2vl-cli${CMAKE_EXECUTABLE_SUFFIX}
    DEPENDS
        ${MTMD_DEPRECATION_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/deprecation-warning-rust/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/deprecation-warning-rust/src/main.rs
)

add_custom_target(llama-llava-cli    ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-llava-cli${CMAKE_EXECUTABLE_SUFFIX})
add_custom_target(llama-gemma3-cli   ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-gemma3-cli${CMAKE_EXECUTABLE_SUFFIX})
add_custom_target(llama-minicpmv-cli ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-minicpmv-cli${CMAKE_EXECUTABLE_SUFFIX})
add_custom_target(llama-qwen2vl-cli  ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-qwen2vl-cli${CMAKE_EXECUTABLE_SUFFIX})

set(TARGET llama-mtmd-cli)
set(MTMD_CLI_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/cli-rust/Cargo.toml)
set(MTMD_CLI_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/mtmd-cli)
set(MTMD_CLI_BIN ${MTMD_CLI_TARGET_DIR}/debug/llama-mtmd-cli${CMAKE_EXECUTABLE_SUFFIX})
set(MTMD_CLI_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-mtmd-cli${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${MTMD_CLI_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${MTMD_CLI_MANIFEST} --target-dir ${MTMD_CLI_TARGET_DIR} --bin llama-mtmd-cli
    COMMAND ${CMAKE_COMMAND} -E copy ${MTMD_CLI_BIN} ${MTMD_CLI_OUTPUT}
    DEPENDS
        ${MTMD_CLI_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/cli-rust/src/main.rs
)
add_custom_target(${TARGET} ALL DEPENDS ${MTMD_CLI_OUTPUT})
if(LLAMA_TOOLS_INSTALL)
    install(PROGRAMS ${MTMD_CLI_OUTPUT} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# mtmd-debug tool
set(MTMD_DEBUG_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/debug-rust/Cargo.toml)
set(MTMD_DEBUG_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/mtmd-debug)
set(MTMD_DEBUG_BIN ${MTMD_DEBUG_TARGET_DIR}/debug/llama-mtmd-debug${CMAKE_EXECUTABLE_SUFFIX})
set(MTMD_DEBUG_OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llama-mtmd-debug${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${MTMD_DEBUG_OUTPUT}
    COMMAND ${CARGO_EXECUTABLE} build --manifest-path ${MTMD_DEBUG_MANIFEST} --target-dir ${MTMD_DEBUG_TARGET_DIR} --bin llama-mtmd-debug
    COMMAND ${CMAKE_COMMAND} -E copy ${MTMD_DEBUG_BIN} ${MTMD_DEBUG_OUTPUT}
    DEPENDS
        ${MTMD_DEBUG_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/debug-rust/src/main.rs
)
add_custom_target(llama-mtmd-debug ALL DEPENDS ${MTMD_DEBUG_OUTPUT})
