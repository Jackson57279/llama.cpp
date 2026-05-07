cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

message(STATUS "Enable the VirtGPU/Virglrenderer backend library")

set_source_files_properties(backend.cpp.inc PROPERTIES LANGUAGE CXX)
ggml_add_backend_library(ggml-virtgpu-backend
                         backend.cpp.inc)

set(GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/backend-dispatched-rust/Cargo.toml)
set(GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/ggml-virtgpu-backend-dispatched)
set(GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_LIB ${GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}ggml_virtgpu_backend_dispatched_rust${CMAKE_STATIC_LIBRARY_SUFFIX})

add_custom_command(
    OUTPUT ${GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_MANIFEST} --target-dir ${GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_TARGET_DIR}
    DEPENDS
        ${GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/backend-dispatched-rust/src/lib.rs
)
add_custom_target(ggml-virtgpu-backend-dispatched-rust-build DEPENDS ${GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_LIB})
add_library(ggml-virtgpu-backend-dispatched-rust STATIC IMPORTED GLOBAL)
set_target_properties(ggml-virtgpu-backend-dispatched-rust PROPERTIES IMPORTED_LOCATION ${GGML_VIRTGPU_BACKEND_DISPATCHED_RUST_LIB})
add_dependencies(ggml-virtgpu-backend-dispatched-rust ggml-virtgpu-backend-dispatched-rust-build)
target_link_libraries(ggml-virtgpu-backend PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,ggml-virtgpu-backend-dispatched-rust>")

target_compile_options(ggml-virtgpu-backend PRIVATE -std=c++20)

# Add include directory for ggml-backend-impl-defs.inc and other core headers
target_include_directories(ggml-virtgpu-backend PRIVATE ../..)
