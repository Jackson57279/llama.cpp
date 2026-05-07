find_package(Threads REQUIRED)

llama_add_compile_flags()
find_program(CARGO_EXECUTABLE cargo REQUIRED)

#
# llama-common-base
#

# Build info header

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    set(GIT_DIR "${PROJECT_SOURCE_DIR}/.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        string(FIND "${REAL_GIT_DIR}" "/" SLASH_POS)
        if (SLASH_POS EQUAL 0)
            set(GIT_DIR "${REAL_GIT_DIR}")
        else()
            set(GIT_DIR "${PROJECT_SOURCE_DIR}/${REAL_GIT_DIR}")
        endif()
    endif()

    if(EXISTS "${GIT_DIR}/index")
        # For build-info.cpp below
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GIT_DIR}/index")
    else()
        message(WARNING "Git index not found in git repository.")
    endif()
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
endif()

set(COMMON_BUILD_INFO_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/build-info-rust/Cargo.toml)
set(COMMON_BUILD_INFO_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/common-build-info)
set(COMMON_BUILD_INFO_RUST_LIB ${COMMON_BUILD_INFO_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_common_build_info_rust${CMAKE_STATIC_LIBRARY_SUFFIX})
set(COMMON_JINJA_STRING_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/jinja/string-rust/Cargo.toml)
set(COMMON_JINJA_STRING_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/common-jinja-string)
set(COMMON_JINJA_STRING_RUST_LIB ${COMMON_JINJA_STRING_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_common_jinja_string_rust${CMAKE_STATIC_LIBRARY_SUFFIX})

add_custom_command(
    OUTPUT ${COMMON_BUILD_INFO_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env
            "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            "LLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}"
            "LLAMA_BUILD_COMMIT=${LLAMA_BUILD_COMMIT}"
            "BUILD_COMPILER=${BUILD_COMPILER}"
            "BUILD_TARGET=${BUILD_TARGET}"
            ${CARGO_EXECUTABLE} build --manifest-path ${COMMON_BUILD_INFO_RUST_MANIFEST} --target-dir ${COMMON_BUILD_INFO_RUST_TARGET_DIR}
    DEPENDS
        ${COMMON_BUILD_INFO_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/build-info-rust/build.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/build-info-rust/src/lib.rs
)
add_custom_target(common-build-info-rust-build DEPENDS ${COMMON_BUILD_INFO_RUST_LIB})
add_library(common-build-info-rust STATIC IMPORTED GLOBAL)
set_target_properties(common-build-info-rust PROPERTIES IMPORTED_LOCATION ${COMMON_BUILD_INFO_RUST_LIB})
add_dependencies(common-build-info-rust common-build-info-rust-build)

add_custom_command(
    OUTPUT ${COMMON_JINJA_STRING_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${COMMON_JINJA_STRING_RUST_MANIFEST} --target-dir ${COMMON_JINJA_STRING_RUST_TARGET_DIR}
    DEPENDS
        ${COMMON_JINJA_STRING_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/jinja/string-rust/src/lib.rs
)
add_custom_target(common-jinja-string-rust-build DEPENDS ${COMMON_JINJA_STRING_RUST_LIB})
add_library(common-jinja-string-rust STATIC IMPORTED GLOBAL)
set_target_properties(common-jinja-string-rust PROPERTIES IMPORTED_LOCATION ${COMMON_JINJA_STRING_RUST_LIB})
add_dependencies(common-jinja-string-rust common-jinja-string-rust-build)

set(TARGET llama-common-base)
add_library(${TARGET} INTERFACE)

target_include_directories(${TARGET} INTERFACE .)
target_link_libraries(${TARGET} INTERFACE common-build-info-rust)
target_link_libraries(${TARGET} INTERFACE common-jinja-string-rust)

#
# llama-common
#

set(TARGET llama-common)

set(COMMON_NGRAM_MOD_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/ngram-mod-rust/Cargo.toml)
set(COMMON_NGRAM_MOD_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/common-ngram-mod)
set(COMMON_NGRAM_MOD_RUST_LIB ${COMMON_NGRAM_MOD_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_common_ngram_mod_rust${CMAKE_STATIC_LIBRARY_SUFFIX})
set(COMMON_NGRAM_CACHE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/ngram-cache-rust/Cargo.toml)
set(COMMON_NGRAM_CACHE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/common-ngram-cache)
set(COMMON_NGRAM_CACHE_RUST_LIB ${COMMON_NGRAM_CACHE_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_common_ngram_cache_rust${CMAKE_STATIC_LIBRARY_SUFFIX})
set(COMMON_NGRAM_MAP_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/ngram-map-rust/Cargo.toml)
set(COMMON_NGRAM_MAP_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/common-ngram-map)
set(COMMON_NGRAM_MAP_RUST_LIB ${COMMON_NGRAM_MAP_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_common_ngram_map_rust${CMAKE_STATIC_LIBRARY_SUFFIX})
set(COMMON_REASONING_BUDGET_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/reasoning-budget-rust/Cargo.toml)
set(COMMON_REASONING_BUDGET_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/common-reasoning-budget)
set(COMMON_REASONING_BUDGET_RUST_LIB ${COMMON_REASONING_BUDGET_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_common_reasoning_budget_rust${CMAKE_STATIC_LIBRARY_SUFFIX})
set(COMMON_REGEX_PARTIAL_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/regex-partial-rust/Cargo.toml)
set(COMMON_REGEX_PARTIAL_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/common-regex-partial)
set(COMMON_REGEX_PARTIAL_RUST_LIB ${COMMON_REGEX_PARTIAL_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_common_regex_partial_rust${CMAKE_STATIC_LIBRARY_SUFFIX})
set(COMMON_UNICODE_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/unicode-rust/Cargo.toml)
set(COMMON_UNICODE_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/common-unicode)
set(COMMON_UNICODE_RUST_LIB ${COMMON_UNICODE_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_common_unicode_rust${CMAKE_STATIC_LIBRARY_SUFFIX})

add_custom_command(
    OUTPUT ${COMMON_NGRAM_MOD_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env RUSTFLAGS=-Crelocation-model=pic
            ${CARGO_EXECUTABLE} build --manifest-path ${COMMON_NGRAM_MOD_RUST_MANIFEST} --target-dir ${COMMON_NGRAM_MOD_RUST_TARGET_DIR}
    DEPENDS
        ${COMMON_NGRAM_MOD_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/ngram-mod-rust/src/lib.rs
)
add_custom_target(common-ngram-mod-rust-build DEPENDS ${COMMON_NGRAM_MOD_RUST_LIB})
add_library(common-ngram-mod-rust STATIC IMPORTED GLOBAL)
set_target_properties(common-ngram-mod-rust PROPERTIES IMPORTED_LOCATION ${COMMON_NGRAM_MOD_RUST_LIB})
add_dependencies(common-ngram-mod-rust common-ngram-mod-rust-build)

add_custom_command(
    OUTPUT ${COMMON_NGRAM_CACHE_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${COMMON_NGRAM_CACHE_RUST_MANIFEST} --target-dir ${COMMON_NGRAM_CACHE_RUST_TARGET_DIR}
    DEPENDS
        ${COMMON_NGRAM_CACHE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/ngram-cache-rust/src/lib.rs
)
add_custom_target(common-ngram-cache-rust-build DEPENDS ${COMMON_NGRAM_CACHE_RUST_LIB})
add_library(common-ngram-cache-rust STATIC IMPORTED GLOBAL)
set_target_properties(common-ngram-cache-rust PROPERTIES IMPORTED_LOCATION ${COMMON_NGRAM_CACHE_RUST_LIB})
add_dependencies(common-ngram-cache-rust common-ngram-cache-rust-build)

add_custom_command(
    OUTPUT ${COMMON_NGRAM_MAP_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${COMMON_NGRAM_MAP_RUST_MANIFEST} --target-dir ${COMMON_NGRAM_MAP_RUST_TARGET_DIR}
    DEPENDS
        ${COMMON_NGRAM_MAP_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/ngram-map-rust/src/lib.rs
)
add_custom_target(common-ngram-map-rust-build DEPENDS ${COMMON_NGRAM_MAP_RUST_LIB})
add_library(common-ngram-map-rust STATIC IMPORTED GLOBAL)
set_target_properties(common-ngram-map-rust PROPERTIES IMPORTED_LOCATION ${COMMON_NGRAM_MAP_RUST_LIB})
add_dependencies(common-ngram-map-rust common-ngram-map-rust-build)

add_custom_command(
    OUTPUT ${COMMON_REASONING_BUDGET_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${COMMON_REASONING_BUDGET_RUST_MANIFEST} --target-dir ${COMMON_REASONING_BUDGET_RUST_TARGET_DIR}
    DEPENDS
        ${COMMON_REASONING_BUDGET_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/reasoning-budget-rust/src/lib.rs
)
add_custom_target(common-reasoning-budget-rust-build DEPENDS ${COMMON_REASONING_BUDGET_RUST_LIB})
add_library(common-reasoning-budget-rust STATIC IMPORTED GLOBAL)
set_target_properties(common-reasoning-budget-rust PROPERTIES IMPORTED_LOCATION ${COMMON_REASONING_BUDGET_RUST_LIB})
add_dependencies(common-reasoning-budget-rust common-reasoning-budget-rust-build)

add_custom_command(
    OUTPUT ${COMMON_REGEX_PARTIAL_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${COMMON_REGEX_PARTIAL_RUST_MANIFEST} --target-dir ${COMMON_REGEX_PARTIAL_RUST_TARGET_DIR}
    DEPENDS
        ${COMMON_REGEX_PARTIAL_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/regex-partial-rust/src/lib.rs
)
add_custom_target(common-regex-partial-rust-build DEPENDS ${COMMON_REGEX_PARTIAL_RUST_LIB})
add_library(common-regex-partial-rust STATIC IMPORTED GLOBAL)
set_target_properties(common-regex-partial-rust PROPERTIES IMPORTED_LOCATION ${COMMON_REGEX_PARTIAL_RUST_LIB})
add_dependencies(common-regex-partial-rust common-regex-partial-rust-build)

add_custom_command(
    OUTPUT ${COMMON_UNICODE_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${COMMON_UNICODE_RUST_MANIFEST} --target-dir ${COMMON_UNICODE_RUST_TARGET_DIR}
    DEPENDS
        ${COMMON_UNICODE_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/unicode-rust/src/lib.rs
)
add_custom_target(common-unicode-rust-build DEPENDS ${COMMON_UNICODE_RUST_LIB})
add_library(common-unicode-rust STATIC IMPORTED GLOBAL)
set_target_properties(common-unicode-rust PROPERTIES IMPORTED_LOCATION ${COMMON_UNICODE_RUST_LIB})
add_dependencies(common-unicode-rust common-unicode-rust-build)

add_library(${TARGET}
    common.cpp.inc
    common.h.inc
    )
set_source_files_properties(common.cpp.inc PROPERTIES LANGUAGE CXX)
set_source_files_properties(common.h.inc PROPERTIES HEADER_FILE_ONLY TRUE)

set_target_properties(${TARGET} PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(${TARGET} PUBLIC . ../vendor)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # TODO: make fine-grained exports in the future
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_libraries(${TARGET} PUBLIC  llama-common-base)
target_link_libraries(${TARGET} PRIVATE common-ngram-mod-rust)
target_link_libraries(${TARGET} PRIVATE common-ngram-cache-rust)
target_link_libraries(${TARGET} PRIVATE common-ngram-map-rust)
target_link_libraries(${TARGET} PUBLIC  common-reasoning-budget-rust)
target_link_libraries(${TARGET} PUBLIC  common-regex-partial-rust)
target_link_libraries(${TARGET} PUBLIC  common-unicode-rust)
target_link_libraries(${TARGET} PRIVATE cpp-httplib)

if (LLAMA_LLGUIDANCE)
    message(FATAL_ERROR "LLAMA_LLGUIDANCE is not available in this Rust-port build")
endif()

target_link_libraries(${TARGET} PUBLIC llama Threads::Threads)
