include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(SERVER_BASE64_RUST_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/base64-rust/Cargo.toml)
set(SERVER_BASE64_RUST_TARGET_DIR ${CMAKE_BINARY_DIR}/cargo/server-base64)
set(SERVER_BASE64_RUST_LIB ${SERVER_BASE64_RUST_TARGET_DIR}/debug/${CMAKE_STATIC_LIBRARY_PREFIX}llama_server_base64_rust${CMAKE_STATIC_LIBRARY_SUFFIX})

add_custom_command(
    OUTPUT ${SERVER_BASE64_RUST_LIB}
    COMMAND ${CMAKE_COMMAND} -E env "RUSTFLAGS=-Crelocation-model=pic -Cpanic=abort"
            ${CARGO_EXECUTABLE} build --manifest-path ${SERVER_BASE64_RUST_MANIFEST} --target-dir ${SERVER_BASE64_RUST_TARGET_DIR}
    DEPENDS
        ${SERVER_BASE64_RUST_MANIFEST}
        ${CMAKE_CURRENT_SOURCE_DIR}/base64-rust/src/lib.rs
)
add_custom_target(server-base64-rust-build DEPENDS ${SERVER_BASE64_RUST_LIB})
add_library(server-base64-rust STATIC IMPORTED GLOBAL)
set_target_properties(server-base64-rust PROPERTIES IMPORTED_LOCATION ${SERVER_BASE64_RUST_LIB})
add_dependencies(server-base64-rust server-base64-rust-build)

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-common.cpp.inc
)
set_source_files_properties(server-common.cpp.inc PROPERTIES LANGUAGE CXX)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${TARGET} PRIVATE server-base64-rust)


# llama-server executable

set(TARGET llama-server)

set(TARGET_SRCS
    server.cpp.inc
)
set_source_files_properties(server.cpp.inc PROPERTIES LANGUAGE CXX)

option(LLAMA_BUILD_WEBUI "Build the embedded Web UI" ON)

if (LLAMA_BUILD_WEBUI)
    set(PUBLIC_ASSETS
        index.html
        bundle.js
        bundle.css
        loading.html
    )

    foreach(asset ${PUBLIC_ASSETS})
        set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
        list(APPEND TARGET_SRCS ${output})
        add_custom_command(
            DEPENDS "${input}"
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
        )
        set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
    endforeach()
    add_definitions(-DLLAMA_BUILD_WEBUI)
else()
endif()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)
