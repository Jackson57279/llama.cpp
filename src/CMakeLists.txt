llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp.inc")
set_source_files_properties(llama.cpp.inc ${LLAMA_MODELS_SOURCES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(../include/llama.h.inc PROPERTIES HEADER_FILE_ONLY TRUE)

find_program(CARGO_EXECUTABLE cargo REQUIRED)
set(LLAMA_CPARAMS_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/llama-cparams-rust/Cargo.toml)
set(LLAMA_CPARAMS_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/llama-cparams)
set(LLAMA_CPARAMS_RUST_LIB ${LLAMA_CPARAMS_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_cparams_rust${CMAKE_STATIC_LIBRARY_SUFFIX})
set(LLAMA_IO_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/llama-io-rust/Cargo.toml)
set(LLAMA_IO_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/llama-io)
set(LLAMA_IO_RUST_LIB ${LLAMA_IO_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_io_rust${CMAKE_STATIC_LIBRARY_SUFFIX})
set(LLAMA_MEMORY_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/llama-memory-rust/Cargo.toml)
set(LLAMA_MEMORY_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/llama-memory)
set(LLAMA_MEMORY_RUST_LIB ${LLAMA_MEMORY_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_memory_rust${CMAKE_STATIC_LIBRARY_SUFFIX})

add_custom_command(
    OUTPUT ${LLAMA_CPARAMS_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${LLAMA_CPARAMS_RUST_MANIFEST} --target-dir ${LLAMA_CPARAMS_RUST_TARGET_DIR}
    DEPENDS
        ${LLAMA_CPARAMS_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/llama-cparams-rust/src/lib.rs
)
add_custom_target(llama-cparams-rust-build DEPENDS ${LLAMA_CPARAMS_RUST_LIB})
add_library(llama-cparams-rust STATIC IMPORTED GLOBAL)
set_target_properties(llama-cparams-rust PROPERTIES IMPORTED_LOCATION ${LLAMA_CPARAMS_RUST_LIB})
add_dependencies(llama-cparams-rust llama-cparams-rust-build)

add_custom_command(
    OUTPUT ${LLAMA_IO_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${LLAMA_IO_RUST_MANIFEST} --target-dir ${LLAMA_IO_RUST_TARGET_DIR}
    DEPENDS
        ${LLAMA_IO_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/llama-io-rust/src/lib.rs
)
add_custom_target(llama-io-rust-build DEPENDS ${LLAMA_IO_RUST_LIB})
add_library(llama-io-rust STATIC IMPORTED GLOBAL)
set_target_properties(llama-io-rust PROPERTIES IMPORTED_LOCATION ${LLAMA_IO_RUST_LIB})
add_dependencies(llama-io-rust llama-io-rust-build)

add_custom_command(
    OUTPUT ${LLAMA_MEMORY_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${LLAMA_MEMORY_RUST_MANIFEST} --target-dir ${LLAMA_MEMORY_RUST_TARGET_DIR}
    DEPENDS
        ${LLAMA_MEMORY_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/llama-memory-rust/src/lib.rs
)
add_custom_target(llama-memory-rust-build DEPENDS ${LLAMA_MEMORY_RUST_LIB})
add_library(llama-memory-rust STATIC IMPORTED GLOBAL)
set_target_properties(llama-memory-rust PROPERTIES IMPORTED_LOCATION ${LLAMA_MEMORY_RUST_LIB})
add_dependencies(llama-memory-rust llama-memory-rust-build)

add_library(llama
            ../include/llama.h.inc
            llama.cpp.inc
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PRIVATE llama-cparams-rust llama-io-rust llama-memory-rust)
if (MSVC)
    target_link_options(llama PRIVATE "/INCLUDE:llama_max_parallel_sequences")
    target_link_options(llama PRIVATE "/INCLUDE:llama_memory_status_combine")
elseif(APPLE)
    target_link_options(llama PRIVATE "LINKER:-u,_llama_max_parallel_sequences")
    target_link_options(llama PRIVATE "LINKER:-u,_llama_memory_status_combine")
else()
    target_link_options(llama PRIVATE "LINKER:-u,llama_max_parallel_sequences")
    target_link_options(llama PRIVATE "LINKER:-u,llama_memory_status_combine")
endif()
target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
