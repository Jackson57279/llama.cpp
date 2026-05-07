cmake_minimum_required(VERSION 3.22.2)
project(ggml-htp C CXX ASM)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

set(HTP_LIB ggml-htp-${DSP_VERSION})
set(HTP_SOURCES
    main.inc
    htp_iface_skel.c
    matmul-ops.inc
    binary-ops.inc
    unary-ops.inc
    sum-rows-ops.inc
    softmax-ops.inc
    act-ops.inc
    rope-ops.inc
    flash-attn-ops.inc
    set-rows-ops.inc
    get-rows-ops.inc
    cpy-ops.inc
    repeat-ops.inc
    argsort-ops.inc
    ssm-conv.inc
    cumsum-ops.inc
    fill-ops.inc
    diag-ops.inc
    solve-tri-ops.inc
)

set_source_files_properties(${HTP_SOURCES} PROPERTIES LANGUAGE C)

add_library(${HTP_LIB} SHARED ${HTP_SOURCES})

target_compile_definitions(${HTP_LIB} PRIVATE
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,HTP_DEBUG=1,NDEBUG=1>
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,FARF_HIGH=1,>
    FP32_QUANTIZE_GROUP_SIZE=${GGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE})

if (GGML_HEXAGON_FA_EXP2_HF)
    message(STATUS "ggml-htp: HMX_FA_USE_EXP2_HF=1 (use FP16 exp2 polynomial in FA softmax)")
    target_compile_definitions(${HTP_LIB} PRIVATE HMX_FA_USE_EXP2_HF=1)
endif()

# HMX acceleration: available on v73+ architectures
set(HTP_HMX_VERSIONS v73 v75 v79 v81)
list(FIND HTP_HMX_VERSIONS ${DSP_VERSION} _hmx_idx)

if (_hmx_idx GREATER_EQUAL 0)
    set(HTP_HMX_SOURCES
        hmx-matmul-ops.inc
        hmx-flash-attn-ops.inc
    )
    set_source_files_properties(${HTP_HMX_SOURCES} PROPERTIES LANGUAGE C)
    target_sources(${HTP_LIB} PRIVATE ${HTP_HMX_SOURCES})

    # -mhmx enables HMX instruction set (needed by files that include hmx-utils.h)
    set_source_files_properties(
        hmx-matmul-ops.inc
        hmx-flash-attn-ops.inc
        PROPERTIES COMPILE_OPTIONS "-mhmx"
    )

    target_compile_definitions(${HTP_LIB} PRIVATE HTP_HAS_HMX=1)
endif()

build_idl(htp_iface.idl ${HTP_LIB})

set_target_properties(${HTP_LIB} PROPERTIES EXPORT_COMPILE_COMMANDS ON)

install(TARGETS ${HTP_LIB})
