cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

include(ExternalProject)

message(STATUS "Including the VirtGPU/Virglrenderer API Remoting")

# Download venus_hw.h from virglrenderer repository
ExternalProject_Add(
    venus_hw_header
    URL https://gitlab.freedesktop.org/virgl/virglrenderer/-/raw/virglrenderer-1.2.0/src/venus_hw.h
    DOWNLOAD_NO_EXTRACT YES
    DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include
    DOWNLOAD_NAME venus_hw.h
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

if (NOT GGML_VIRTGPU_BACKEND STREQUAL "ONLY")
    message(STATUS "Enable the VirtGPU/Virglrenderer API Remoting frontend library")

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DRM REQUIRED libdrm)
    if (NOT GGML_BACKEND_DL)
      # cannot simply use USE_VIRTGPU, as in the 'else()' case the
      # frontend isn't compiled
      target_compile_definitions(ggml PUBLIC "GGML_USE_VIRTGPU_FRONTEND")
    endif()

    set_source_files_properties(virtgpu.cpp.inc PROPERTIES LANGUAGE CXX)
    ggml_add_backend_library(ggml-virtgpu
                             virtgpu.cpp.inc
                             ../../include/ggml-virtgpu.h.inc)

    target_include_directories(ggml-virtgpu PUBLIC /usr/include/libdrm/)

    target_link_libraries(ggml-virtgpu PUBLIC ${DRM_LIBRARIES})
    target_include_directories(ggml-virtgpu PUBLIC ${DRM_INCLUDE_DIRS})
    target_compile_options(ggml-virtgpu PUBLIC ${DRM_CFLAGS_OTHER})

    target_include_directories(ggml-virtgpu PUBLIC ./include)
    target_include_directories(ggml-virtgpu PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    # Ensure venus_hw.h is downloaded before building ggml-virtgpu
    add_dependencies(ggml-virtgpu venus_hw_header)

    target_compile_options(ggml-virtgpu PRIVATE -std=c++20)
else()
    message(STATUS "Not building the VirtGPU/Virglrenderer API Remoting frontend library")
endif()

if (NOT GGML_VIRTGPU_BACKEND STREQUAL "OFF")
    add_subdirectory("backend")
endif()
