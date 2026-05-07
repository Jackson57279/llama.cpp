#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common-defs.inc"
#include "ggml-backend-impl-defs.inc"

#include "ggml-impl-defs.inc"
#include "ggml-cpu.h.inc"
#include "ggml-cpu-impl-defs.inc"
#include "simd-mappings-defs.inc"
#include "ggml.h.inc"
#include "ggml-backend-impl-defs.inc"
#include "ggml-cpu-impl-defs.inc"
#include "ggml-impl-defs.inc"
#include "simd-mappings-defs.inc"

#ifdef __cplusplus
#include <vector>
extern "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);
bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  public:
    virtual ~tensor_traits();
    virtual bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size)        = 0;
    virtual bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) = 0;
};

class extra_buffer_type {
  public:
    virtual ~extra_buffer_type();
    virtual bool            supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const struct ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#include <utility>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
static inline ggml_fp16_t f32_to_f16(float x) {
    return GGML_CPU_FP32_TO_FP16(x);
}

static inline float f16_to_f32(ggml_fp16_t x) {
    return GGML_CPU_FP16_TO_FP32(x);
}

static inline ggml_bf16_t f32_to_bf16(float x) {
    return GGML_FP32_TO_BF16(x);
}

static inline float bf16_to_f32(ggml_bf16_t x) {
    return GGML_BF16_TO_FP32(x);
}

static inline float i32_to_f32(int32_t x) {
    return x;
}

static inline int32_t f32_to_i32(float x) {
    return x;
}

static inline float f32_to_f32(float x) {
    return x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
struct type_conversion_table;

template <>
struct type_conversion_table<ggml_fp16_t> {
    static constexpr float (*to_f32)(ggml_fp16_t) = f16_to_f32;
    static constexpr ggml_fp16_t (*from_f32)(float) = f32_to_f16;
};

template <>
struct type_conversion_table<float> {
    static constexpr float (*to_f32)(float) = f32_to_f32;
    static constexpr float (*from_f32)(float) = f32_to_f32;
};

template <>
struct type_conversion_table<ggml_bf16_t> {
    static constexpr float (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    static constexpr ggml_bf16_t (*from_f32)(float) = f32_to_bf16;
};

template <>
struct type_conversion_table<int32_t> {
    static constexpr float (*to_f32)(int32_t) = i32_to_f32;
    static constexpr int32_t (*from_f32)(float) = f32_to_i32;
};

static std::pair<int64_t, int64_t> get_thread_range(const struct ggml_compute_params * params, const struct ggml_tensor * src0) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    const int64_t nr  = ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    return {ir0, ir1};
}

struct ggml_fa_tile_config {
    static constexpr size_t Q  = GGML_FA_TILE_Q;
    static constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif



// Rename `_generic` functions if no native implementation is available.
// This effectively selects the generic implementation.

#if defined(GGML_CPU_GENERIC)
// quants.c
#define quantize_row_q8_0_generic quantize_row_q8_0
#define quantize_row_q8_1_generic quantize_row_q8_1
#define quantize_row_q8_K_generic quantize_row_q8_K
#define ggml_vec_dot_q4_0_q8_0_generic ggml_vec_dot_q4_0_q8_0
#define ggml_vec_dot_q4_1_q8_1_generic ggml_vec_dot_q4_1_q8_1
#define ggml_vec_dot_q5_0_q8_0_generic ggml_vec_dot_q5_0_q8_0
#define ggml_vec_dot_q5_1_q8_1_generic ggml_vec_dot_q5_1_q8_1
#define ggml_vec_dot_q8_0_q8_0_generic ggml_vec_dot_q8_0_q8_0
#define ggml_vec_dot_mxfp4_q8_0_generic ggml_vec_dot_mxfp4_q8_0
#define ggml_vec_dot_nvfp4_q8_0_generic ggml_vec_dot_nvfp4_q8_0
#define ggml_vec_dot_q1_0_q8_0_generic ggml_vec_dot_q1_0_q8_0
#define ggml_vec_dot_tq1_0_q8_K_generic ggml_vec_dot_tq1_0_q8_K
#define ggml_vec_dot_tq2_0_q8_K_generic ggml_vec_dot_tq2_0_q8_K
#define ggml_vec_dot_q2_K_q8_K_generic ggml_vec_dot_q2_K_q8_K
#define ggml_vec_dot_q3_K_q8_K_generic ggml_vec_dot_q3_K_q8_K
#define ggml_vec_dot_q4_K_q8_K_generic ggml_vec_dot_q4_K_q8_K
#define ggml_vec_dot_q5_K_q8_K_generic ggml_vec_dot_q5_K_q8_K
#define ggml_vec_dot_q6_K_q8_K_generic ggml_vec_dot_q6_K_q8_K
#define ggml_vec_dot_iq2_xxs_q8_K_generic ggml_vec_dot_iq2_xxs_q8_K
#define ggml_vec_dot_iq2_xs_q8_K_generic ggml_vec_dot_iq2_xs_q8_K
#define ggml_vec_dot_iq2_s_q8_K_generic ggml_vec_dot_iq2_s_q8_K
#define ggml_vec_dot_iq3_xxs_q8_K_generic ggml_vec_dot_iq3_xxs_q8_K
#define ggml_vec_dot_iq3_s_q8_K_generic ggml_vec_dot_iq3_s_q8_K
#define ggml_vec_dot_iq1_s_q8_K_generic ggml_vec_dot_iq1_s_q8_K
#define ggml_vec_dot_iq1_m_q8_K_generic ggml_vec_dot_iq1_m_q8_K
#define ggml_vec_dot_iq4_nl_q8_0_generic ggml_vec_dot_iq4_nl_q8_0
#define ggml_vec_dot_iq4_xs_q8_K_generic ggml_vec_dot_iq4_xs_q8_K
// repack.cpp
#define ggml_quantize_mat_q8_0_4x4_generic ggml_quantize_mat_q8_0_4x4
#define ggml_quantize_mat_q8_0_4x8_generic ggml_quantize_mat_q8_0_4x8
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_quantize_mat_q8_K_4x8_generic ggml_quantize_mat_q8_K_4x8
#define ggml_gemv_q4_0_4x4_q8_0_generic ggml_gemv_q4_0_4x4_q8_0
#define ggml_gemv_q4_0_4x8_q8_0_generic ggml_gemv_q4_0_4x8_q8_0
#define ggml_gemv_q4_0_8x8_q8_0_generic ggml_gemv_q4_0_8x8_q8_0
#define ggml_gemv_q2_K_8x8_q8_K_generic ggml_gemv_q2_K_8x8_q8_K
#define ggml_gemv_q4_K_8x4_q8_K_generic ggml_gemv_q4_K_8x4_q8_K
#define ggml_gemv_q4_K_8x8_q8_K_generic ggml_gemv_q4_K_8x8_q8_K
#define ggml_gemv_q5_K_8x4_q8_K_generic ggml_gemv_q5_K_8x4_q8_K
#define ggml_gemv_q5_K_8x8_q8_K_generic ggml_gemv_q5_K_8x8_q8_K
#define ggml_gemv_q6_K_8x4_q8_K_generic ggml_gemv_q6_K_8x4_q8_K
#define ggml_gemv_q6_K_8x8_q8_K_generic ggml_gemv_q6_K_8x8_q8_K
#define ggml_gemv_iq4_nl_4x4_q8_0_generic ggml_gemv_iq4_nl_4x4_q8_0
#define ggml_gemv_iq4_nl_8x8_q8_0_generic ggml_gemv_iq4_nl_8x8_q8_0
#define ggml_gemv_mxfp4_4x4_q8_0_generic ggml_gemv_mxfp4_4x4_q8_0
#define ggml_gemv_mxfp4_8x8_q8_0_generic ggml_gemv_mxfp4_8x8_q8_0
#define ggml_gemv_q8_0_4x4_q8_0_generic ggml_gemv_q8_0_4x4_q8_0
#define ggml_gemv_q8_0_4x8_q8_0_generic ggml_gemv_q8_0_4x8_q8_0
#define ggml_gemm_q4_0_4x4_q8_0_generic ggml_gemm_q4_0_4x4_q8_0
#define ggml_gemm_q4_0_4x8_q8_0_generic ggml_gemm_q4_0_4x8_q8_0
#define ggml_gemm_q4_0_8x8_q8_0_generic ggml_gemm_q4_0_8x8_q8_0
#define ggml_gemm_q2_K_8x8_q8_K_generic ggml_gemm_q2_K_8x8_q8_K
#define ggml_gemm_q4_K_8x4_q8_K_generic ggml_gemm_q4_K_8x4_q8_K
#define ggml_gemm_q4_K_8x8_q8_K_generic ggml_gemm_q4_K_8x8_q8_K
#define ggml_gemm_q5_K_8x4_q8_K_generic ggml_gemm_q5_K_8x4_q8_K
#define ggml_gemm_q5_K_8x8_q8_K_generic ggml_gemm_q5_K_8x8_q8_K
#define ggml_gemm_q6_K_8x4_q8_K_generic ggml_gemm_q6_K_8x4_q8_K
#define ggml_gemm_q6_K_8x8_q8_K_generic ggml_gemm_q6_K_8x8_q8_K
#define ggml_gemm_iq4_nl_4x4_q8_0_generic ggml_gemm_iq4_nl_4x4_q8_0
#define ggml_gemm_iq4_nl_8x8_q8_0_generic ggml_gemm_iq4_nl_8x8_q8_0
#define ggml_gemm_mxfp4_4x4_q8_0_generic ggml_gemm_mxfp4_4x4_q8_0
#define ggml_gemm_mxfp4_8x8_q8_0_generic ggml_gemm_mxfp4_8x8_q8_0
#define ggml_gemm_q8_0_4x4_q8_0_generic ggml_gemm_q8_0_4x4_q8_0
#define ggml_gemm_q8_0_4x8_q8_0_generic ggml_gemm_q8_0_4x8_q8_0
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64)
// repack.cpp
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_quantize_mat_q8_K_4x8_generic ggml_quantize_mat_q8_K_4x8
#define ggml_gemv_iq4_nl_8x8_q8_0_generic ggml_gemv_iq4_nl_8x8_q8_0
#define ggml_gemv_mxfp4_8x8_q8_0_generic ggml_gemv_mxfp4_8x8_q8_0
#define ggml_gemv_q2_K_8x8_q8_K_generic ggml_gemv_q2_K_8x8_q8_K
#define ggml_gemm_iq4_nl_8x8_q8_0_generic ggml_gemm_iq4_nl_8x8_q8_0
#define ggml_gemm_mxfp4_8x8_q8_0_generic ggml_gemm_mxfp4_8x8_q8_0
#define ggml_gemm_q2_K_8x8_q8_K_generic ggml_gemm_q2_K_8x8_q8_K
#elif defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
// quants.c
#define ggml_vec_dot_nvfp4_q8_0_generic ggml_vec_dot_nvfp4_q8_0
// repack.cpp
#define ggml_quantize_mat_q8_0_4x4_generic ggml_quantize_mat_q8_0_4x4
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_gemv_q4_0_4x4_q8_0_generic ggml_gemv_q4_0_4x4_q8_0
#define ggml_gemv_q4_0_4x8_q8_0_generic ggml_gemv_q4_0_4x8_q8_0
#define ggml_gemv_q4_K_8x4_q8_K_generic ggml_gemv_q4_K_8x4_q8_K
#define ggml_gemv_q5_K_8x4_q8_K_generic ggml_gemv_q5_K_8x4_q8_K
#define ggml_gemv_q5_K_8x8_q8_K_generic ggml_gemv_q5_K_8x8_q8_K
#define ggml_gemv_q6_K_8x4_q8_K_generic ggml_gemv_q6_K_8x4_q8_K
#define ggml_gemv_q6_K_8x8_q8_K_generic ggml_gemv_q6_K_8x8_q8_K
#define ggml_gemv_iq4_nl_4x4_q8_0_generic ggml_gemv_iq4_nl_4x4_q8_0
#define ggml_gemv_mxfp4_4x4_q8_0_generic ggml_gemv_mxfp4_4x4_q8_0
#define ggml_gemv_q8_0_4x4_q8_0_generic ggml_gemv_q8_0_4x4_q8_0
#define ggml_gemv_q8_0_4x8_q8_0_generic ggml_gemv_q8_0_4x8_q8_0
#define ggml_gemm_q4_0_4x4_q8_0_generic ggml_gemm_q4_0_4x4_q8_0
#define ggml_gemm_q4_0_4x8_q8_0_generic ggml_gemm_q4_0_4x8_q8_0
#define ggml_gemm_q4_K_8x4_q8_K_generic ggml_gemm_q4_K_8x4_q8_K
#define ggml_gemm_q5_K_8x4_q8_K_generic ggml_gemm_q5_K_8x4_q8_K
#define ggml_gemm_q5_K_8x8_q8_K_generic ggml_gemm_q5_K_8x8_q8_K
#define ggml_gemm_q6_K_8x4_q8_K_generic ggml_gemm_q6_K_8x4_q8_K
#define ggml_gemm_q6_K_8x8_q8_K_generic ggml_gemm_q6_K_8x8_q8_K
#define ggml_gemm_iq4_nl_4x4_q8_0_generic ggml_gemm_iq4_nl_4x4_q8_0
#define ggml_gemm_mxfp4_4x4_q8_0_generic ggml_gemm_mxfp4_4x4_q8_0
#define ggml_gemm_q8_0_4x4_q8_0_generic ggml_gemm_q8_0_4x4_q8_0
#define ggml_gemm_q8_0_4x8_q8_0_generic ggml_gemm_q8_0_4x8_q8_0
#elif defined(__POWERPC__) || defined(__powerpc__)
// ref: https://github.com/ggml-org/llama.cpp/pull/14146#issuecomment-2972561679
// quants.c
#define quantize_row_q8_K_generic quantize_row_q8_K
#define ggml_vec_dot_nvfp4_q8_0_generic ggml_vec_dot_nvfp4_q8_0
#define ggml_vec_dot_q1_0_q8_0_generic ggml_vec_dot_q1_0_q8_0
#define ggml_vec_dot_tq1_0_q8_K_generic ggml_vec_dot_tq1_0_q8_K
#define ggml_vec_dot_tq2_0_q8_K_generic ggml_vec_dot_tq2_0_q8_K
#define ggml_vec_dot_iq1_m_q8_K_generic ggml_vec_dot_iq1_m_q8_K
// repack.cpp
#define ggml_quantize_mat_q8_0_4x4_generic ggml_quantize_mat_q8_0_4x4
#define ggml_quantize_mat_q8_0_4x8_generic ggml_quantize_mat_q8_0_4x8
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_quantize_mat_q8_K_4x8_generic ggml_quantize_mat_q8_K_4x8
#define ggml_gemv_q4_0_4x4_q8_0_generic ggml_gemv_q4_0_4x4_q8_0
#define ggml_gemv_q4_0_4x8_q8_0_generic ggml_gemv_q4_0_4x8_q8_0
#define ggml_gemv_q4_0_8x8_q8_0_generic ggml_gemv_q4_0_8x8_q8_0
#define ggml_gemv_q2_K_8x8_q8_K_generic ggml_gemv_q2_K_8x8_q8_K
#define ggml_gemv_q4_K_8x4_q8_K_generic ggml_gemv_q4_K_8x4_q8_K
#define ggml_gemv_q4_K_8x8_q8_K_generic ggml_gemv_q4_K_8x8_q8_K
#define ggml_gemv_q5_K_8x4_q8_K_generic ggml_gemv_q5_K_8x4_q8_K
#define ggml_gemv_q5_K_8x8_q8_K_generic ggml_gemv_q5_K_8x8_q8_K
#define ggml_gemv_q6_K_8x4_q8_K_generic ggml_gemv_q6_K_8x4_q8_K
#define ggml_gemv_q6_K_8x8_q8_K_generic ggml_gemv_q6_K_8x8_q8_K
#define ggml_gemv_iq4_nl_4x4_q8_0_generic ggml_gemv_iq4_nl_4x4_q8_0
#define ggml_gemv_iq4_nl_8x8_q8_0_generic ggml_gemv_iq4_nl_8x8_q8_0
#define ggml_gemv_mxfp4_4x4_q8_0_generic ggml_gemv_mxfp4_4x4_q8_0
#define ggml_gemv_mxfp4_8x8_q8_0_generic ggml_gemv_mxfp4_8x8_q8_0
#define ggml_gemv_q8_0_4x4_q8_0_generic ggml_gemv_q8_0_4x4_q8_0
#define ggml_gemv_q8_0_4x8_q8_0_generic ggml_gemv_q8_0_4x8_q8_0
#define ggml_gemm_q4_0_4x4_q8_0_generic ggml_gemm_q4_0_4x4_q8_0
#define ggml_gemm_q4_0_4x8_q8_0_generic ggml_gemm_q4_0_4x8_q8_0
#define ggml_gemm_q4_0_8x8_q8_0_generic ggml_gemm_q4_0_8x8_q8_0
#define ggml_gemm_q2_K_8x8_q8_K_generic ggml_gemm_q2_K_8x8_q8_K
#define ggml_gemm_q4_K_8x4_q8_K_generic ggml_gemm_q4_K_8x4_q8_K
#define ggml_gemm_q4_K_8x8_q8_K_generic ggml_gemm_q4_K_8x8_q8_K
#define ggml_gemm_q5_K_8x4_q8_K_generic ggml_gemm_q5_K_8x4_q8_K
#define ggml_gemm_q5_K_8x8_q8_K_generic ggml_gemm_q5_K_8x8_q8_K
#define ggml_gemm_q6_K_8x4_q8_K_generic ggml_gemm_q6_K_8x4_q8_K
#define ggml_gemm_q6_K_8x8_q8_K_generic ggml_gemm_q6_K_8x8_q8_K
#define ggml_gemm_iq4_nl_4x4_q8_0_generic ggml_gemm_iq4_nl_4x4_q8_0
#define ggml_gemm_iq4_nl_8x8_q8_0_generic ggml_gemm_iq4_nl_8x8_q8_0
#define ggml_gemm_mxfp4_4x4_q8_0_generic ggml_gemm_mxfp4_4x4_q8_0
#define ggml_gemm_mxfp4_8x8_q8_0_generic ggml_gemm_mxfp4_8x8_q8_0
#define ggml_gemm_q8_0_4x4_q8_0_generic ggml_gemm_q8_0_4x4_q8_0
#define ggml_gemm_q8_0_4x8_q8_0_generic ggml_gemm_q8_0_4x8_q8_0
#elif defined(__loongarch64)
// quants.c
#define quantize_row_q8_K_generic quantize_row_q8_K
#define ggml_vec_dot_tq1_0_q8_K_generic ggml_vec_dot_tq1_0_q8_K
#define ggml_vec_dot_tq2_0_q8_K_generic ggml_vec_dot_tq2_0_q8_K
#define ggml_vec_dot_iq1_m_q8_K_generic ggml_vec_dot_iq1_m_q8_K
#define ggml_vec_dot_mxfp4_q8_0_generic ggml_vec_dot_mxfp4_q8_0
#define ggml_vec_dot_nvfp4_q8_0_generic ggml_vec_dot_nvfp4_q8_0
#define ggml_vec_dot_q1_0_q8_0_generic ggml_vec_dot_q1_0_q8_0
// repack.cpp
#define ggml_quantize_mat_q8_0_4x4_generic ggml_quantize_mat_q8_0_4x4
#define ggml_quantize_mat_q8_0_4x8_generic ggml_quantize_mat_q8_0_4x8
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_quantize_mat_q8_K_4x8_generic ggml_quantize_mat_q8_K_4x8
#define ggml_gemv_q4_0_4x4_q8_0_generic ggml_gemv_q4_0_4x4_q8_0
#define ggml_gemv_q4_0_4x8_q8_0_generic ggml_gemv_q4_0_4x8_q8_0
#define ggml_gemv_q4_0_8x8_q8_0_generic ggml_gemv_q4_0_8x8_q8_0
#define ggml_gemv_q2_K_8x8_q8_K_generic ggml_gemv_q2_K_8x8_q8_K
#define ggml_gemv_q4_K_8x4_q8_K_generic ggml_gemv_q4_K_8x4_q8_K
#define ggml_gemv_q4_K_8x8_q8_K_generic ggml_gemv_q4_K_8x8_q8_K
#define ggml_gemv_q5_K_8x4_q8_K_generic ggml_gemv_q5_K_8x4_q8_K
#define ggml_gemv_q5_K_8x8_q8_K_generic ggml_gemv_q5_K_8x8_q8_K
#define ggml_gemv_q6_K_8x4_q8_K_generic ggml_gemv_q6_K_8x4_q8_K
#define ggml_gemv_q6_K_8x8_q8_K_generic ggml_gemv_q6_K_8x8_q8_K
#define ggml_gemv_iq4_nl_4x4_q8_0_generic ggml_gemv_iq4_nl_4x4_q8_0
#define ggml_gemv_iq4_nl_8x8_q8_0_generic ggml_gemv_iq4_nl_8x8_q8_0
#define ggml_gemv_mxfp4_4x4_q8_0_generic ggml_gemv_mxfp4_4x4_q8_0
#define ggml_gemv_mxfp4_8x8_q8_0_generic ggml_gemv_mxfp4_8x8_q8_0
#define ggml_gemv_q8_0_4x4_q8_0_generic ggml_gemv_q8_0_4x4_q8_0
#define ggml_gemv_q8_0_4x8_q8_0_generic ggml_gemv_q8_0_4x8_q8_0
#define ggml_gemm_q4_0_4x4_q8_0_generic ggml_gemm_q4_0_4x4_q8_0
#define ggml_gemm_q4_0_4x8_q8_0_generic ggml_gemm_q4_0_4x8_q8_0
#define ggml_gemm_q4_0_8x8_q8_0_generic ggml_gemm_q4_0_8x8_q8_0
#define ggml_gemm_q2_K_8x8_q8_K_generic ggml_gemm_q2_K_8x8_q8_K
#define ggml_gemm_q4_K_8x4_q8_K_generic ggml_gemm_q4_K_8x4_q8_K
#define ggml_gemm_q4_K_8x8_q8_K_generic ggml_gemm_q4_K_8x8_q8_K
#define ggml_gemm_q5_K_8x4_q8_K_generic ggml_gemm_q5_K_8x4_q8_K
#define ggml_gemm_q5_K_8x8_q8_K_generic ggml_gemm_q5_K_8x8_q8_K
#define ggml_gemm_q6_K_8x4_q8_K_generic ggml_gemm_q6_K_8x4_q8_K
#define ggml_gemm_q6_K_8x8_q8_K_generic ggml_gemm_q6_K_8x8_q8_K
#define ggml_gemm_iq4_nl_4x4_q8_0_generic ggml_gemm_iq4_nl_4x4_q8_0
#define ggml_gemm_iq4_nl_8x8_q8_0_generic ggml_gemm_iq4_nl_8x8_q8_0
#define ggml_gemm_mxfp4_4x4_q8_0_generic ggml_gemm_mxfp4_4x4_q8_0
#define ggml_gemm_mxfp4_8x8_q8_0_generic ggml_gemm_mxfp4_8x8_q8_0
#define ggml_gemm_q8_0_4x4_q8_0_generic ggml_gemm_q8_0_4x4_q8_0
#define ggml_gemm_q8_0_4x8_q8_0_generic ggml_gemm_q8_0_4x8_q8_0
#elif defined(__riscv)
// quants.c
#define ggml_vec_dot_nvfp4_q8_0_generic ggml_vec_dot_nvfp4_q8_0
#define ggml_vec_dot_q1_0_q8_0_generic ggml_vec_dot_q1_0_q8_0
// repack.cpp
#define ggml_quantize_mat_q8_0_4x1_generic ggml_quantize_mat_q8_0_4x1
#define ggml_quantize_mat_q8_0_4x4_generic ggml_quantize_mat_q8_0_4x4
#define ggml_quantize_mat_q8_K_4x1_generic ggml_quantize_mat_q8_K_4x1
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_quantize_mat_q8_K_4x8_generic ggml_quantize_mat_q8_K_4x8
#define ggml_gemv_q4_0_4x4_q8_0_generic ggml_gemv_q4_0_4x4_q8_0
#define ggml_gemv_q4_0_4x8_q8_0_generic ggml_gemv_q4_0_4x8_q8_0
#define ggml_gemv_q2_K_8x8_q8_K_generic ggml_gemv_q2_K_8x8_q8_K
#define ggml_gemv_q4_K_8x4_q8_K_generic ggml_gemv_q4_K_8x4_q8_K
#define ggml_gemv_q4_K_8x8_q8_K_generic ggml_gemv_q4_K_8x8_q8_K
#define ggml_gemv_q5_K_8x4_q8_K_generic ggml_gemv_q5_K_8x4_q8_K
#define ggml_gemv_q5_K_8x8_q8_K_generic ggml_gemv_q5_K_8x8_q8_K
#define ggml_gemv_q6_K_8x4_q8_K_generic ggml_gemv_q6_K_8x4_q8_K
#define ggml_gemv_q6_K_8x8_q8_K_generic ggml_gemv_q6_K_8x8_q8_K
#define ggml_gemv_iq4_nl_4x4_q8_0_generic ggml_gemv_iq4_nl_4x4_q8_0
#define ggml_gemv_iq4_nl_8x8_q8_0_generic ggml_gemv_iq4_nl_8x8_q8_0
#define ggml_gemv_mxfp4_4x4_q8_0_generic ggml_gemv_mxfp4_4x4_q8_0
#define ggml_gemv_mxfp4_8x8_q8_0_generic ggml_gemv_mxfp4_8x8_q8_0
#define ggml_gemv_q8_0_4x4_q8_0_generic ggml_gemv_q8_0_4x4_q8_0
#define ggml_gemv_q8_0_4x8_q8_0_generic ggml_gemv_q8_0_4x8_q8_0
#define ggml_gemm_q4_0_4x4_q8_0_generic ggml_gemm_q4_0_4x4_q8_0
#define ggml_gemm_q4_0_4x8_q8_0_generic ggml_gemm_q4_0_4x8_q8_0
#define ggml_gemm_q2_K_8x8_q8_K_generic ggml_gemm_q2_K_8x8_q8_K
#define ggml_gemm_q4_K_8x4_q8_K_generic ggml_gemm_q4_K_8x4_q8_K
#define ggml_gemm_q4_K_8x8_q8_K_generic ggml_gemm_q4_K_8x8_q8_K
#define ggml_gemm_q5_K_8x4_q8_K_generic ggml_gemm_q5_K_8x4_q8_K
#define ggml_gemm_q5_K_8x8_q8_K_generic ggml_gemm_q5_K_8x8_q8_K
#define ggml_gemm_q6_K_8x4_q8_K_generic ggml_gemm_q6_K_8x4_q8_K
#define ggml_gemm_q6_K_8x8_q8_K_generic ggml_gemm_q6_K_8x8_q8_K
#define ggml_gemm_iq4_nl_4x4_q8_0_generic ggml_gemm_iq4_nl_4x4_q8_0
#define ggml_gemm_iq4_nl_8x8_q8_0_generic ggml_gemm_iq4_nl_8x8_q8_0
#define ggml_gemm_mxfp4_4x4_q8_0_generic ggml_gemm_mxfp4_4x4_q8_0
#define ggml_gemm_mxfp4_8x8_q8_0_generic ggml_gemm_mxfp4_8x8_q8_0
#define ggml_gemm_q8_0_4x4_q8_0_generic ggml_gemm_q8_0_4x4_q8_0
#define ggml_gemm_q8_0_4x8_q8_0_generic ggml_gemm_q8_0_4x8_q8_0
#elif defined(__s390x__)
// quants.c
#define quantize_row_q8_K_generic quantize_row_q8_K
#define ggml_vec_dot_nvfp4_q8_0_generic ggml_vec_dot_nvfp4_q8_0
#define ggml_vec_dot_q1_0_q8_0_generic ggml_vec_dot_q1_0_q8_0
#define ggml_vec_dot_tq1_0_q8_K_generic ggml_vec_dot_tq1_0_q8_K
#define ggml_vec_dot_tq2_0_q8_K_generic ggml_vec_dot_tq2_0_q8_K
#define ggml_vec_dot_q2_K_q8_K_generic ggml_vec_dot_q2_K_q8_K
#define ggml_vec_dot_iq2_xxs_q8_K_generic ggml_vec_dot_iq2_xxs_q8_K
#define ggml_vec_dot_iq2_xs_q8_K_generic ggml_vec_dot_iq2_xs_q8_K
#define ggml_vec_dot_iq2_s_q8_K_generic ggml_vec_dot_iq2_s_q8_K
#define ggml_vec_dot_iq3_xxs_q8_K_generic ggml_vec_dot_iq3_xxs_q8_K
#define ggml_vec_dot_iq3_s_q8_K_generic ggml_vec_dot_iq3_s_q8_K
#define ggml_vec_dot_iq1_s_q8_K_generic ggml_vec_dot_iq1_s_q8_K
#define ggml_vec_dot_iq1_m_q8_K_generic ggml_vec_dot_iq1_m_q8_K
// repack.cpp
#define ggml_quantize_mat_q8_0_4x4_generic ggml_quantize_mat_q8_0_4x4
#define ggml_quantize_mat_q8_0_4x8_generic ggml_quantize_mat_q8_0_4x8
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_quantize_mat_q8_K_4x8_generic ggml_quantize_mat_q8_K_4x8
#define ggml_gemv_q4_0_4x4_q8_0_generic ggml_gemv_q4_0_4x4_q8_0
#define ggml_gemv_q4_0_4x8_q8_0_generic ggml_gemv_q4_0_4x8_q8_0
#define ggml_gemv_q4_0_8x8_q8_0_generic ggml_gemv_q4_0_8x8_q8_0
#define ggml_gemv_q2_K_8x8_q8_K_generic ggml_gemv_q2_K_8x8_q8_K
#define ggml_gemv_q4_K_8x4_q8_K_generic ggml_gemv_q4_K_8x4_q8_K
#define ggml_gemv_q4_K_8x8_q8_K_generic ggml_gemv_q4_K_8x8_q8_K
#define ggml_gemv_q5_K_8x4_q8_K_generic ggml_gemv_q5_K_8x4_q8_K
#define ggml_gemv_q5_K_8x8_q8_K_generic ggml_gemv_q5_K_8x8_q8_K
#define ggml_gemv_q6_K_8x4_q8_K_generic ggml_gemv_q6_K_8x4_q8_K
#define ggml_gemv_q6_K_8x8_q8_K_generic ggml_gemv_q6_K_8x8_q8_K
#define ggml_gemv_iq4_nl_4x4_q8_0_generic ggml_gemv_iq4_nl_4x4_q8_0
#define ggml_gemv_iq4_nl_8x8_q8_0_generic ggml_gemv_iq4_nl_8x8_q8_0
#define ggml_gemv_mxfp4_4x4_q8_0_generic ggml_gemv_mxfp4_4x4_q8_0
#define ggml_gemv_mxfp4_8x8_q8_0_generic ggml_gemv_mxfp4_8x8_q8_0
#define ggml_gemv_q8_0_4x4_q8_0_generic ggml_gemv_q8_0_4x4_q8_0
#define ggml_gemv_q8_0_4x8_q8_0_generic ggml_gemv_q8_0_4x8_q8_0
#define ggml_gemm_q4_0_4x4_q8_0_generic ggml_gemm_q4_0_4x4_q8_0
#define ggml_gemm_q4_0_4x8_q8_0_generic ggml_gemm_q4_0_4x8_q8_0
#define ggml_gemm_q4_0_8x8_q8_0_generic ggml_gemm_q4_0_8x8_q8_0
#define ggml_gemm_q2_K_8x8_q8_K_generic ggml_gemm_q2_K_8x8_q8_K
#define ggml_gemm_q4_K_8x4_q8_K_generic ggml_gemm_q4_K_8x4_q8_K
#define ggml_gemm_q4_K_8x8_q8_K_generic ggml_gemm_q4_K_8x8_q8_K
#define ggml_gemm_q5_K_8x4_q8_K_generic ggml_gemm_q5_K_8x4_q8_K
#define ggml_gemm_q5_K_8x8_q8_K_generic ggml_gemm_q5_K_8x8_q8_K
#define ggml_gemm_q6_K_8x4_q8_K_generic ggml_gemm_q6_K_8x4_q8_K
#define ggml_gemm_q6_K_8x8_q8_K_generic ggml_gemm_q6_K_8x8_q8_K
#define ggml_gemm_iq4_nl_4x4_q8_0_generic ggml_gemm_iq4_nl_4x4_q8_0
#define ggml_gemm_iq4_nl_8x8_q8_0_generic ggml_gemm_iq4_nl_8x8_q8_0
#define ggml_gemm_mxfp4_4x4_q8_0_generic ggml_gemm_mxfp4_4x4_q8_0
#define ggml_gemm_mxfp4_8x8_q8_0_generic ggml_gemm_mxfp4_8x8_q8_0
#define ggml_gemm_q8_0_4x4_q8_0_generic ggml_gemm_q8_0_4x4_q8_0
#define ggml_gemm_q8_0_4x8_q8_0_generic ggml_gemm_q8_0_4x8_q8_0
#elif defined(__wasm__)
// quants.c
#define ggml_vec_dot_q4_1_q8_1_generic ggml_vec_dot_q4_1_q8_1
#define ggml_vec_dot_tq1_0_q8_K_generic ggml_vec_dot_tq1_0_q8_K
#define ggml_vec_dot_tq2_0_q8_K_generic ggml_vec_dot_tq2_0_q8_K
#define ggml_vec_dot_iq2_xxs_q8_K_generic ggml_vec_dot_iq2_xxs_q8_K
#define ggml_vec_dot_iq2_xs_q8_K_generic ggml_vec_dot_iq2_xs_q8_K
#define ggml_vec_dot_iq2_s_q8_K_generic ggml_vec_dot_iq2_s_q8_K
#define ggml_vec_dot_iq3_xxs_q8_K_generic ggml_vec_dot_iq3_xxs_q8_K
#define ggml_vec_dot_iq3_s_q8_K_generic ggml_vec_dot_iq3_s_q8_K
#define ggml_vec_dot_iq1_s_q8_K_generic ggml_vec_dot_iq1_s_q8_K
#define ggml_vec_dot_iq1_m_q8_K_generic ggml_vec_dot_iq1_m_q8_K
#define ggml_vec_dot_iq4_nl_q8_0_generic ggml_vec_dot_iq4_nl_q8_0
#define ggml_vec_dot_iq4_xs_q8_K_generic ggml_vec_dot_iq4_xs_q8_K
#define ggml_vec_dot_mxfp4_q8_0_generic ggml_vec_dot_mxfp4_q8_0
#define ggml_vec_dot_nvfp4_q8_0_generic ggml_vec_dot_nvfp4_q8_0
#define ggml_vec_dot_q1_0_q8_0_generic ggml_vec_dot_q1_0_q8_0
// repack.cpp
#define ggml_quantize_mat_q8_0_4x4_generic ggml_quantize_mat_q8_0_4x4
#define ggml_quantize_mat_q8_0_4x8_generic ggml_quantize_mat_q8_0_4x8
#define ggml_quantize_mat_q8_K_4x4_generic ggml_quantize_mat_q8_K_4x4
#define ggml_quantize_mat_q8_K_4x8_generic ggml_quantize_mat_q8_K_4x8
#define ggml_gemv_q4_0_4x4_q8_0_generic ggml_gemv_q4_0_4x4_q8_0
#define ggml_gemv_q4_0_4x8_q8_0_generic ggml_gemv_q4_0_4x8_q8_0
#define ggml_gemv_q4_0_8x8_q8_0_generic ggml_gemv_q4_0_8x8_q8_0
#define ggml_gemv_q2_K_8x8_q8_K_generic ggml_gemv_q2_K_8x8_q8_K
#define ggml_gemv_q4_K_8x4_q8_K_generic ggml_gemv_q4_K_8x4_q8_K
#define ggml_gemv_q4_K_8x8_q8_K_generic ggml_gemv_q4_K_8x8_q8_K
#define ggml_gemv_q5_K_8x4_q8_K_generic ggml_gemv_q5_K_8x4_q8_K
#define ggml_gemv_q5_K_8x8_q8_K_generic ggml_gemv_q5_K_8x8_q8_K
#define ggml_gemv_q6_K_8x4_q8_K_generic ggml_gemv_q6_K_8x4_q8_K
#define ggml_gemv_q6_K_8x8_q8_K_generic ggml_gemv_q6_K_8x8_q8_K
#define ggml_gemv_iq4_nl_4x4_q8_0_generic ggml_gemv_iq4_nl_4x4_q8_0
#define ggml_gemv_iq4_nl_8x8_q8_0_generic ggml_gemv_iq4_nl_8x8_q8_0
#define ggml_gemv_mxfp4_4x4_q8_0_generic ggml_gemv_mxfp4_4x4_q8_0
#define ggml_gemv_mxfp4_8x8_q8_0_generic ggml_gemv_mxfp4_8x8_q8_0
#define ggml_gemv_q8_0_4x4_q8_0_generic ggml_gemv_q8_0_4x4_q8_0
#define ggml_gemv_q8_0_4x8_q8_0_generic ggml_gemv_q8_0_4x8_q8_0
#define ggml_gemm_q4_0_4x4_q8_0_generic ggml_gemm_q4_0_4x4_q8_0
#define ggml_gemm_q4_0_4x8_q8_0_generic ggml_gemm_q4_0_4x8_q8_0
#define ggml_gemm_q4_0_8x8_q8_0_generic ggml_gemm_q4_0_8x8_q8_0
#define ggml_gemm_q2_K_8x8_q8_K_generic ggml_gemm_q2_K_8x8_q8_K
#define ggml_gemm_q4_K_8x4_q8_K_generic ggml_gemm_q4_K_8x4_q8_K
#define ggml_gemm_q4_K_8x8_q8_K_generic ggml_gemm_q4_K_8x8_q8_K
#define ggml_gemm_q5_K_8x4_q8_K_generic ggml_gemm_q5_K_8x4_q8_K
#define ggml_gemm_q5_K_8x8_q8_K_generic ggml_gemm_q5_K_8x8_q8_K
#define ggml_gemm_q6_K_8x4_q8_K_generic ggml_gemm_q6_K_8x4_q8_K
#define ggml_gemm_q6_K_8x8_q8_K_generic ggml_gemm_q6_K_8x8_q8_K
#define ggml_gemm_iq4_nl_4x4_q8_0_generic ggml_gemm_iq4_nl_4x4_q8_0
#define ggml_gemm_iq4_nl_8x8_q8_0_generic ggml_gemm_iq4_nl_8x8_q8_0
#define ggml_gemm_mxfp4_4x4_q8_0_generic ggml_gemm_mxfp4_4x4_q8_0
#define ggml_gemm_mxfp4_8x8_q8_0_generic ggml_gemm_mxfp4_8x8_q8_0
#define ggml_gemm_q8_0_4x4_q8_0_generic ggml_gemm_q8_0_4x4_q8_0
#define ggml_gemm_q8_0_4x8_q8_0_generic ggml_gemm_q8_0_4x8_q8_0
#endif


#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdio>  // for GGML_ASSERT

#define GGML_COMMON_DECL_CPP
#include "ggml-common-defs.inc"

#include "ggml.h.inc"

// GGML internal header

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

// control size
static_assert(sizeof(block<4, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(sizeof(block<4, 16>) == 16 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<4,16> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");
static_assert(sizeof(block<8, 16>) == 16 * sizeof(ggml_half) + QK8_0 * 16, "wrong block<8,16> size/padding");

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q4_0x16 = block<4, 16>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;
using block_q8_0x16 = block<8, 16>;

struct block_q4_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[96];  // scales and mins, quantized with 6 bits
    uint8_t qs[1024];    // 4--bit quants
};

static_assert(sizeof(block_q4_Kx8) == sizeof(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 4, "wrong q4_K block size/padding");
struct block_q4_Kx16 {
    ggml_half d[16];      // super-block scale for quantized scales
    ggml_half dmin[16];   // super-block scale for quantized mins
    uint8_t scales[192];  // scales and mins, quantized with 6 bits
    uint8_t qs[2048];    // 4--bit quants
};

static_assert(sizeof(block_q4_Kx16) == sizeof(ggml_half) * 32 + K_SCALE_SIZE * 16 + QK_K * 8, "wrong q4_K block size/padding");
struct block_q2_Kx8 {
    ggml_half d[8];      // super-block scale for quantized scales
    ggml_half dmin[8];   // super-block scale for quantized mins
    uint8_t scales[128];  // scales and mins, quantized with 4 bits
    uint8_t qs[512];    // 2--bit quants
};

static_assert(sizeof(block_q2_Kx8) == sizeof(ggml_half) * 16 + QK_K/2 + QK_K * 2, "wrong q2_K block size/padding");
struct block_q2_Kx16 {
    ggml_half d[16];       // Super-block scale for quantized scales
    ggml_half dmin[16];    // Super-block scale for quantized mins
    uint8_t   scales[256]; // Sub-block scales (16 cols * 16 sub-blocks)
    uint8_t   qs[1024];    // Data (16 cols * 64 bytes per block)
};
static_assert(sizeof(block_q2_Kx16) == sizeof(ggml_half) * 32 + QK_K + QK_K * 4, "wrong q2_K block size/padding");

struct block_q5_Kx8 {
    ggml_half d[8];              // super-block scale for quantized scales
    ggml_half dmin[8];           // super-block scale for quantized mins
    uint8_t   scales[96];        // scales and mins, quantized with 6 bits
    uint8_t   qh[QK_K * 8 / 8];  // high bits of 5-bit quants
    uint8_t   qs[QK_K * 8 / 2];  // low bits of 5-bit quants (in groups of 4)
};

static_assert(sizeof(block_q5_Kx8) == sizeof(ggml_half) * 16 + K_SCALE_SIZE * 8 + QK_K * 5,
              "wrong q5_K block size/padding");

struct block_q6_Kx8 {
    ggml_half d[8];
    int8_t    scales[QK_K / 16 * 8];
    uint8_t   ql[QK_K / 2 * 8];  // low bits of 6-bit quants (groups of 2)
    uint8_t   qh[QK_K / 4 * 8];  // high bits of 6-bit quants (groups of 4)
};

static_assert(sizeof(block_q6_Kx8) == sizeof(ggml_half) * 8 + QK_K / 16 * 8 + 3 * QK_K / 4 * 8,
              "wrong q6_K block size/padding");

struct block_q8_Kx4 {
    float d[4];              // delta
    int8_t qs[QK_K * 4];     // quants
    int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

static_assert(sizeof(block_q8_Kx4) == sizeof(float) * 4 + QK_K * 4 + (QK_K / 4) * sizeof(int16_t), "wrong q8_K block size/padding");

struct block_iq4_nlx4 {
    ggml_half d[4];            // deltas for 4 iq4_nl blocks
    uint8_t   qs[QK4_NL * 2];  // nibbles / quants for 4 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx4) == 4 * sizeof(ggml_half) + QK4_NL * 2, "wrong iq4_nlx4 block size/padding");

struct block_iq4_nlx8 {
    ggml_half d[8];            // deltas for 8 iq4_nl blocks
    uint8_t   qs[QK4_NL * 4];  // nibbles / quants for 8 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx8) == 8 * sizeof(ggml_half) + QK4_NL * 4, "wrong iq4_nlx8 block size/padding");

struct block_iq4_nlx16 {
    ggml_half d[16];            // deltas for 16 iq4_nl blocks
    uint8_t   qs[QK4_NL * 8];  // nibbles / quants for 16 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx16) == 16 * sizeof(ggml_half) + QK4_NL * 8, "wrong iq4_nlx16 block size/padding");
struct block_mxfp4x4 {
    uint8_t e[4];
    uint8_t qs[QK_MXFP4 * 2];
};
static_assert(sizeof(block_mxfp4x4) == 4 + QK_MXFP4 * 2, "wrong mxfp4x4 block size/padding");

struct block_mxfp4x8 {
    uint8_t e[8];
    uint8_t qs[QK_MXFP4 * 4];
};
static_assert(sizeof(block_mxfp4x8) == 8 + QK_MXFP4 * 4, "wrong mxfp4x8 block size/padding");

#if defined(__cplusplus)
extern "C" {
#endif

void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x1(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_16x1_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_16x1_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#endif

// Native implementations
void ggml_quantize_mat_q8_0_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_0_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_quantize_mat_q8_K_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void ggml_gemv_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemv_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
#endif

#if defined(__cplusplus)
} // extern "C"
#endif


#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants bytes
//                         from bias offset form to pure sign form (this saves subtract
//                         operations durin unpacking)
//

extern "C" {

#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 1;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_K_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    const int blck_size_interleave = 1;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;
            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }
        for (int j = 0; j < QK_K * 4; j++) {
            int src_id = j % 4;
            int src_offset = j / 4;
            int index = ((j >> 6) << 2) + (j & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}
#endif

void ggml_quantize_mat_q8_0_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_0_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_K_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}

void ggml_quantize_mat_q8_K_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}

} // extern "C"

template <int64_t INTER_SIZE, ggml_type PARAM_TYPE>
void ggml_quantize_mat_t(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row);

template <> void ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x4(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x4(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x8(x, vy, n_per_row);
}

#if defined __riscv_zvfh
template <> void ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x1(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x1(x, vy, n_per_row);
}
#endif

template <int M, int N>
static void ggml_gemv_q6_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int blocklen          = M;
    constexpr int ncols_interleaved = N;
    const int     qk                = QK_K;
    const int     nb                = n / qk;
    const int     blocks_per_half   = 64 / blocklen;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q6_Kx8 * b_ptr = (const block_q6_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                const int base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                const int base_h = base_l + 64;

                const int scale_idx_l = base_l / 16;
                const int scale_idx_h = base_h / 16;

                const int qh_shift_l = ((base_l % 128) / 32) * 2;
                const int qh_shift_h = ((base_h % 128) / 32) * 2;

                const int qh_half_l = (base_l / 128) * 32;
                const int qh_half_h = (base_h / 128) * 32;

                for (int j = 0; j < ncols_interleaved; j++) {
                    const int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    const int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    int sumi_l = 0;
                    int sumi_h = 0;

                    for (int i = 0; i < blocklen; i++) {
                        const int ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        const int l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        const int hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        const int qh_idx_l    = qh_half_l + ((base_l + i) % 32);
                        const int qh_chunk_l  = qh_idx_l / blocklen;
                        const int qh_pos_l    = qh_idx_l % blocklen;
                        const int qh_offset_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        const int hi_2_l      = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                        const int qh_idx_h    = qh_half_h + ((base_h + i) % 32);
                        const int qh_chunk_h  = qh_idx_h / blocklen;
                        const int qh_pos_h    = qh_idx_h % blocklen;
                        const int qh_offset_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        const int hi_2_h      = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                        const int q_l = ((hi_2_l << 4) | l_4) - 32;
                        const int q_h = ((hi_2_h << 4) | hi_4) - 32;

                        const int8_t a_l = a_ptr[l].qs[base_l + i];
                        const int8_t a_h = a_ptr[l].qs[base_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

template <int M, int N>
static void ggml_gemm_q6_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int blocklen          = M;
    constexpr int ncols_interleaved = N;
    const int     qk                = QK_K;
    const int     nb                = n / qk;
    const int     blocks_per_half   = 64 / blocklen;
    const int     q8_half_stride    = 512;
    const int     q8_low_high_step  = 256;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);

    float sumf[4][8];

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q6_Kx8 * b_ptr = (const block_q6_Kx8 *) vx + (x * nb);

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    const int base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                    const int base_h = base_l + 64;

                    const int scale_idx_l = base_l / 16;
                    const int scale_idx_h = base_h / 16;

                    const int qh_shift_l = ((base_l % 128) / 32) * 2;
                    const int qh_shift_h = ((base_h % 128) / 32) * 2;

                    const int qh_half_l = (base_l / 128) * 32;
                    const int qh_half_h = (base_h / 128) * 32;

                    const int q8_base = (k / blocks_per_half) * q8_half_stride + (k % blocks_per_half) * (blocklen * 4);

                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            const int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            const int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            int sumi_l = 0;
                            int sumi_h = 0;

                            for (int i = 0; i < blocklen; i++) {
                                const int ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                const int l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                const int hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                const int qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                                const int qh_chunk_l = qh_idx_l / blocklen;
                                const int qh_pos_l   = qh_idx_l % blocklen;
                                const int qh_offset_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                const int hi_2_l = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                                const int qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                                const int qh_chunk_h = qh_idx_h / blocklen;
                                const int qh_pos_h   = qh_idx_h % blocklen;
                                const int qh_offset_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                const int hi_2_h = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                                const int q_l = ((hi_2_l << 4) | l_4) - 32;
                                const int q_h = ((hi_2_h << 4) | hi_4) - 32;

                                const int8_t q8_l = a_ptr[l].qs[q8_base + m * blocklen + i];
                                const int8_t q8_h = a_ptr[l].qs[q8_base + m * blocklen + i + q8_low_high_step];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

template <int M, int N>
static void ggml_gemv_q5_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int         blocklen          = M;
    constexpr int         ncols_interleaved = N;
    const int             qk                = QK_K;
    const int             nb                = n / qk;
    static const uint32_t kmask1            = 0x3f3f3f3f;
    static const uint32_t kmask2            = 0x0f0f0f0f;
    static const uint32_t kmask3            = 0x03030303;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float    sumf[ncols_interleaved];
    float    sum_minf[ncols_interleaved];
    uint32_t utmp[32];
    int      sumi1;
    int      sumi2;
    int      sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q5_Kx8 * b_ptr = (const block_q5_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j]     = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_SIZE, K_SCALE_SIZE);
                utmp[sb * 4 + 3]      = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2]      = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                constexpr int scale_stride = 32;
                uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                const int qh_shift = (k / (32 / blocklen)) * 2;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi  = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                        const int qh_idx      = (k * blocklen + i) % 32;
                        const int qh_chunk    = qh_idx / blocklen;
                        const int qh_pos      = qh_idx % blocklen;
                        const int b_qh_offset = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                        const uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                        const uint8_t h0     = (qh_val >> qh_shift) & 1;
                        const uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                        const int v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                        const int v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                        const int q8_offset = (k / (32 / blocklen)) * 64 + (k % (32 / blocklen)) * blocklen + i;

                        sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                        sumi2 = (v1 * a_ptr[l].qs[q8_offset + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                                   GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

template <int M, int N>
static void ggml_gemm_q5_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int         blocklen          = M;
    constexpr int         ncols_interleaved = N;
    const int             qk                = QK_K;
    const int             nb                = n / qk;
    static const uint32_t kmask1            = 0x3f3f3f3f;
    static const uint32_t kmask2            = 0x0f0f0f0f;
    static const uint32_t kmask3            = 0x03030303;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float    sumf[4][ncols_interleaved];
    float    sum_minf[4][ncols_interleaved];
    uint32_t utmp[32];
    int      sumi1;
    int      sumi2;
    int      sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q5_Kx8 * b_ptr = (const block_q5_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j]     = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_SIZE, K_SCALE_SIZE);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    constexpr int scale_stride = 32;
                    uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                    uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                    const int qh_shift = (k / (32 / blocklen)) * 2;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi  = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                                const int qh_idx   = (k * blocklen + i) % 32;
                                const int qh_chunk = qh_idx / blocklen;
                                const int qh_pos   = qh_idx % blocklen;
                                const int b_qh_offset =
                                    qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                                const uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                                const uint8_t h0     = (qh_val >> qh_shift) & 1;
                                const uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                                const int v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                                const int v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                                const int q8_offset = (k / (32 / blocklen)) * 256 +
                                                      (k % (32 / blocklen)) * 4 * blocklen + m * blocklen + i;

                                sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                                sumi2 = (v1 * a_ptr[l].qs[q8_offset + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    for (int m = 0; m < 4; m++) {
                        const int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                              GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

extern "C" {

void ggml_gemv_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[8];
    float sum_minf[8];
    int sumi1,sumi2,sumi3,sumi4;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *)vy;
    for(int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q2_Kx8 * b_ptr = (const block_q2_Kx8 *) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (4 * blocklen)); k++) {
                const uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                const uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                const uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                const uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi3 = 0;
                    sumi4 = 0;
                    sumi = 0;
                    int offset = ((k / 2) % 2) + j * 2;
                    for (int i = 0; i < blocklen; ++i){
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                        const int v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                        const int v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                        const int v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 32]);
                        sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 64]);
                        sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 96]);

                        sumi1 = sumi1 * (scales_0[offset] & 0xF);
                        sumi2 = sumi2 * (scales_1[offset] & 0xF);
                        sumi3 = sumi3 * (scales_2[offset] & 0xF);
                        sumi4 = sumi4 * (scales_3[offset] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for(int sb = 0; sb < 8; sb++) {
                const uint8_t *mins = b_ptr[l].scales + sb * 16;
                for(int j = 0; j < ncols_interleaved; j++){
                    sum_minf[j] += ((mins[j * 2] >> 4) * a_ptr[l].bsums[sb * 2] + (mins[(j * 2)+ 1] >> 4) * a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}


void ggml_gemv_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_mxfp4x4 * b_ptr = (const block_mxfp4x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_mxfp4x8 * b_ptr = (const block_mxfp4x8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q8_0_4x4_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int   sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q8_0_4x8_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int   sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// Only enable these for RISC-V.
#if defined __riscv_zvfh
void ggml_gemv_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[16];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x16 * b_ptr = (const block_q4_0x16 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;
    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);
    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);
    float sumf[16];
    float sum_minf[16];
    uint8_t scales[128];
    uint8_t mins[128];
    int sumi1;
    int sumi2;
    int sumi;
    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx16 * b_ptr = (const block_q4_Kx16 *) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
            sum_minf[j] = 0.0f;
        }
        for (int l = 0; l < nb; l++) {
            for (int i = 0; i < 128; i++) {
                scales[i] = b_ptr[l].scales[i] & 0x0F;
                mins[i] = b_ptr[l].scales[i] >> 4;
            }
            for (int i = 0; i < 64; i++) {
                scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *min = &mins[sb * 16];
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb += 2) {
                uint8_t *scales_0 = &scales[sb * 16];
                uint8_t *scales_1 = &scales[(sb + 1) * 16];
                for (int i = 0; i < QK4_0; i++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        sumi1 = 0;
                        sumi2 = 0;
                        sumi = 0;
                        const int v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[sb * 32 + i]);
                        sumi2 = (v1 * a_ptr[l].qs[sb * 32 + 32 + i]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                        sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                    }
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[16];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx16 * b_ptr = (const block_iq4_nlx16 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen          = 1;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[16];
    int   sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    assert(n % QK_K == 0);
    assert(nr == 1);
    assert(nc % 16 == 0);

    UNUSED(bs);
    UNUSED(nr);

    const int nb = n / QK_K;
    const block_q2_Kx16 * x = (const block_q2_Kx16 *)vx;
    const block_q8_K    * y = (const block_q8_K *)vy;

    // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
    const int sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,  // 0-7
        8, 12, 9, 13, 10, 14, 11, 15 // 8-15
    };

    for (int col_tile = 0; col_tile < nc; col_tile += 16) {
        const block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
        const block_q8_K    * y_ptr = y;

        float sumf[16] = {0};

        // Loop over K-blocks
        for (int k_block = 0; k_block < nb; ++k_block) {
            int32_t isum[16]  = {0};
            int32_t summs[16] = {0};

            const uint8_t * qs_rhs = x_ptr[k_block].qs;
            const uint8_t * sc_rhs = x_ptr[k_block].scales;
            const int8_t  * qs_lhs = y_ptr[k_block].qs;
            const int16_t * bs_lhs = y_ptr[k_block].bsums;

            // Iterate over sub-blocks 0..15
            for (int sb = 0; sb < 16; ++sb) {
                // Correction Term
                int16_t bsum = bs_lhs[sb];
                int scale_offset = sb_perm[sb] * 16;

                for (int col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    summs[col] += bsum * (sc_val >> 4); // Min is high 4 bits
                }

                // Main Dot Product
                // Calculate base offsets for Q2 unpacking based on SB
                int byte_base;
                if (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                else        byte_base = (sb % 2 == 0) ? 32 : 48;

                int shift = ((sb / 2) % 4) * 2;

                for (int col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

                    // Process 16 elements (l=0..15)
                    for (int l = 0; l < 16; ++l) {
                        // Q2: Interleaved by column. Byte `l` contains 4 k-values.
                        int qs_idx = (byte_base + l) * 16 + col;
                        uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                        // Q8: Linear access
                        int k = sb * 16 + l;
                        int8_t q8_val = qs_lhs[k];

                        isum[col] += q8_val * q2_val * d_sb;
                    }
                }
            }

            // Finalize K-Block
            for (int col = 0; col < 16; ++col) {
                float d_lhs = y_ptr[k_block].d;
                float d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                float dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                float d_all = d_lhs * d_rhs;
                float d_min = d_lhs * dm_rhs;

                sumf[col] += (isum[col] * d_all) - (summs[col] * d_min);
            }
        }

        for (int col = 0; col < 16; ++col) {
            s[col_tile + col] = sumf[col];
        }
    }
}
#endif

void ggml_gemm_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][8];
    float sum_minf[4][8];
    int sumi1, sumi2, sumi3, sumi4;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q2_Kx8 * b_ptr = (const block_q2_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (4 * blocklen)); k++) {

                    const uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                    const uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                    const uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                    const uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi3 = 0;
                            sumi4 = 0;
                            sumi = 0;
                            int offset = ((k / 2) % 2) + j * 2;
                            for (int i = 0; i < blocklen; ++i){
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                                const int v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                                const int v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                                const int v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 512 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 256]);
                                sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 384]);
                                sumi1 = sumi1 * (scales_0[offset] & 0xF);
                                sumi2 = sumi2 * (scales_1[offset] & 0xF);
                                sumi3 = sumi3 * (scales_2[offset] & 0xF);
                                sumi4 = sumi4 * (scales_3[offset] & 0xF);
                                sumi += sumi1 + sumi2 + sumi3 + sumi4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for(int sb = 0; sb < 8; sb++) {
                    const uint8_t *mins = b_ptr[l].scales + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) *  6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            int mins_prod = ((mins[j * 2] >> 4) * bsums[0] + (mins[(j * 2)+ 1] >> 4) * bsums[1]);
                            sum_minf[m][j] += (mins_prod) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_mxfp4x4 * b_ptr = (const block_mxfp4x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_mxfp4x8 * b_ptr = (const block_mxfp4x8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q8_0_4x4_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}



void ggml_gemm_q8_0_4x8_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

// Only enable these for RISC-V.
#if defined __riscv_zvfh
void ggml_gemm_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][16];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x16 * b_ptr = (const block_q4_0x16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][16];
    float sum_minf[4][16];
    uint8_t scales[128];
    uint8_t mins[128];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx16 * b_ptr = (const block_q4_Kx16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int i = 0; i < 128; i++) {
                    scales[i] = b_ptr[l].scales[i] & 0x0F;
                    mins[i] = b_ptr[l].scales[i] >> 4;
                }
                for (int i = 0; i < 64; i++) {
                    scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                    mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                    mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
                }

                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *min = &mins[sb * 16];
                    for(int m = 0; m < 4; m++) {
                        const int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }

                for (int sb = 0; sb < 8; sb += 2) {
                    uint8_t *scales_0 = &scales[sb * 16];
                    uint8_t *scales_1 = &scales[(sb + 1) * 16];

                    for (int i = 0; i < QK4_0; i++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi1 = 0;
                                sumi2 = 0;
                                sumi = 0;

                                const int v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[sb * 4 * 32 + i * 4 + m]);
                                sumi2 = (v1 * a_ptr[l].qs[sb * 4 * 32 + 32 * 4 + i * 4 + m]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;

                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                            }
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][16];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx16 * b_ptr = (const block_iq4_nlx16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen          = 1;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][16];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}


void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % 16 == 0);
    const int nb = n / QK_K;
    const block_q2_Kx16 * x = (const block_q2_Kx16 *)vx;
    const block_q8_Kx4  * y = (const block_q8_Kx4 *)vy;

    const int sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15
    };

    // Iterate Rows in tiles of 4
    for (int row_tile = 0; row_tile < nr; row_tile += 4) {
        // Iterate Columns in tiles of 16
        for (int col_tile = 0; col_tile < nc; col_tile += 16) {

            const block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
            const block_q8_Kx4  * y_ptr = y + (row_tile / 4) * nb;

            float sumf[4][16];
            memset(sumf, 0, sizeof(sumf));

            for (int k_block = 0; k_block < nb; ++k_block) {
                int32_t isum[4][16];
                int32_t summs[4][16];
                memset(isum, 0, sizeof(isum));
                memset(summs, 0, sizeof(summs));

                const uint8_t * qs_rhs = x_ptr[k_block].qs;
                const uint8_t * sc_rhs = x_ptr[k_block].scales;
                const int8_t  * qs_lhs = y_ptr[k_block].qs;
                const int16_t * bs_lhs = y_ptr[k_block].bsums;

                for (int sb = 0; sb < 16; ++sb) {
                    int scale_offset = sb_perm[sb] * 16;

                    int byte_base;
                    if (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                    else        byte_base = (sb % 2 == 0) ? 32 : 48;
                    int shift = ((sb / 2) % 4) * 2;

                    for (int col = 0; col < 16; ++col) {
                        uint8_t sc_val = sc_rhs[scale_offset + col];
                        int32_t d_sb = sc_val & 0xF;
                        int32_t m_sb = sc_val >> 4;

                        // Correction Term
                        for (int r = 0; r < 4; ++r) {
                            int bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][col] += bs_lhs[bsum_idx] * m_sb;
                        }

                        // Main Dot Product
                        for (int l = 0; l < 16; ++l) {
                            int qs_idx = (byte_base + l) * 16 + col;
                            uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                            // Calculate Q8 index for this specific k and row
                            int k = sb * 16 + l;
                            int q8_idx = (k / 4) * 16 + (k % 4);

                            for (int r = 0; r < 4; ++r) {
                                // Add r*4 to jump to the correct row within the 4x4 chunk
                                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                                isum[r][col] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                // Finalize K-Block
                for (int col = 0; col < 16; ++col) {
                    float d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                    float dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                    for (int r = 0; r < 4; ++r) {
                        float d_lhs = y_ptr[k_block].d[r];
                        float d_all = d_lhs * d_rhs;
                        float d_min = d_lhs * dm_rhs;
                        sumf[r][col] += (isum[r][col] * d_all) - (summs[r][col] * d_min);
                    }
                }
            }

            for (int r = 0; r < 4; ++r) {
                for (int col = 0; col < 16; ++col) {
                    s[(row_tile + r) * bs + (col_tile + col)] = sumf[r][col];
                }
            }
        }
    }
}
#endif

} // extern "C"

static block_q8_0x4 make_block_q8_0x4(block_q8_0 * in, unsigned int blck_size_interleave) {
    block_q8_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK8_0 * 4 / blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 4;
        int src_offset = (i / 4) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }
    return out;
}

static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 2 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        const uint64_t xor_mask = 0x8888888888888888ULL;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
        }
    } else if (blck_size_interleave == 4) {
        const uint32_t xor_mask = 0x88888888;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint32_t elems;
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
static block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 4 / blck_size_interleave;
    const uint64_t xor_mask = 0x8888888888888888ULL;

    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        elems ^= xor_mask;
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    return out;
}

static block_q4_0x16 make_block_q4_0x16(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x16 out;

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 8 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        const uint8_t xor_mask = 0x88;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset] ^ xor_mask;
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static block_q4_Kx8 make_block_q4_Kx8(block_q4_K * in, unsigned int blck_size_interleave) {
    block_q4_Kx8 out;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 4 / blck_size_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], blck_size_interleave);
        memcpy(&out.qs[dst_offset], &elems, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    return out;
}

static block_q4_Kx16 make_block_q4_Kx16(block_q4_K * in, unsigned int blck_size_interleave) {
    block_q4_Kx16 out;
    //Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 16; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 8 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }

        // RVV repacking.
        //
        // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
        uint8_t s[128], m[128];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) {
                s[i * 16 + j] = in[j].scales[i] & 63;
                m[i * 16 + j] = in[j].scales[i + 4] & 63;
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) {
                s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
                m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
            }
        }

        for (int i = 0; i < 128; i++) {
            out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
        }
        for (int i = 0; i < 64; i++) {
            out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static block_q2_Kx8 make_block_q2_Kx8(block_q2_K * in, unsigned int blck_size_interleave) {
    block_q2_Kx8 out;

    // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 2 / blck_size_interleave;

    // Interleave Q2_K quants by taking 8 bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
    // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
    // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
    // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
    // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K structures

    for (int i = 0; i < 128; i++) {
        // Index for selecting which q2k super block
        int src1 = (i % 16) / 2;
        // Index for selecting scale
        int src2 = ((i / 16) * 2) + (i % 2);

        out.scales[i] = in[src1].scales[src2];
    }
    return out;
}

static block_q5_Kx8 make_block_q5_Kx8(block_q5_K * in, unsigned int blck_size_interleave) {
    block_q5_Kx8 out;
    //Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 4 / blck_size_interleave;

    // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }

    // Repeat for high bits with the same chunk size, since
    // the high bits are interleaved in Q5_K and the index is
    // qh_idx = (qs_idx % 32);
    // qh_val = qh[qh_idx] >> (qs_idx / 32);
    for (int i = 0; i < end / 4; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy(&out.qh[dst_offset], &in[src_id].qh[src_offset], blck_size_interleave);
    }

    // The below logic is copied over from Q4_K
    // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
    // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q5_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
    uint8_t s[8], m[8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    return out;
}

static block_q6_Kx8 make_block_q6_Kx8(block_q6_K * in, unsigned int blck_size_interleave) {
    block_q6_Kx8  out;
    constexpr int n_blocks = 8;  // Kx8
    for (int i = 0; i < n_blocks; i++) {
        out.d[i] = in[i].d;
    }

    const int end_ls = QK_K * 4 / blck_size_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    for (int i = 0; i < end_ls; ++i) {
        int src_id     = i % n_blocks;
        int src_offset = (i / n_blocks) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elem_ls;
        memcpy(&elem_ls, &in[src_id].ql[src_offset], blck_size_interleave);
        memcpy(&out.ql[dst_offset], &elem_ls, blck_size_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    const int end_hs = end_ls / 2;
    for (int i = 0; i < end_hs; ++i) {
        int src_id     = i % n_blocks;
        int src_offset = (i / n_blocks) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elem_hs;
        memcpy(&elem_hs, &in[src_id].qh[src_offset], blck_size_interleave);
        memcpy(&out.qh[dst_offset], &elem_hs, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr int n_scales = QK_K / 16;

    for (int i = 0; i < n_blocks; i++) {
        for (int j = 0; j < n_scales; j++) {
            out.scales[j * n_blocks + i] = in[i].scales[j];
        }
    }

    return out;
}

static block_q2_Kx16 make_block_q2_Kx16(const block_q2_K * in, unsigned int blck_size_interleave) {
    block_q2_Kx16 out;
    constexpr int N_COLS = 16;

    // 1. Copy Super-Scales (d) and Super-Mins (dmin)
    for (int i = 0; i < N_COLS; i++) {
        out.d[i]    = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    // 2. Interleave Q2_K Data
    const int bytes_per_col = 64;
    const int total_bytes = N_COLS * bytes_per_col;
    const int end = total_bytes / blck_size_interleave;

    for (int i = 0; i < end; ++i) {
        int src_col_id = i % N_COLS;
        int src_offset = (i / N_COLS) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_col_id].qs[src_offset], blck_size_interleave);
    }

    // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
    int out_idx = 0;

    // Arrays define the sub-block order for each group
    const int even_low_sbs[]  = {0, 2, 4, 6};
    const int odd_low_sbs[]   = {1, 3, 5, 7};
    const int even_high_sbs[] = {8, 10, 12, 14};
    const int odd_high_sbs[]  = {9, 11, 13, 15};

    // Pack Group 1: Even-Low
    for (int sb : even_low_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 2: Odd-Low
    for (int sb : odd_low_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 3: Even-High
    for (int sb : even_high_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 4: Odd-High
    for (int sb : odd_high_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    return out;
}

static int repack_q4_0_to_q4_0_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 4;

    block_q4_0x4 * dst = (block_q4_0x4 *)t->data;
    const block_q4_0 * src = (const block_q4_0 *)data;
    block_q4_0 dst_tmp[4];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_K_to_q4_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 8 || interleave_block == 4);
    constexpr int nrows_interleaved = 8;

    block_q4_Kx8 * dst = (block_q4_Kx8*)t->data;
    const block_q4_K * src = (const block_q4_K*) data;
    block_q4_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_K_to_q4_K_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr int nrows_interleaved = 16;

    block_q4_Kx16 * dst = (block_q4_Kx16*)t->data;
    const block_q4_K * src = (const block_q4_K*) data;
    block_q4_K dst_tmp[16];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q2_K_to_q2_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q2_Kx8 * dst = (block_q2_Kx8*)t->data;
    const block_q2_K * src = (const block_q2_K*) data;
    block_q2_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q2_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q2_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q2_K_to_q2_K_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    constexpr int nrows_interleaved = 16;

    block_q2_Kx16 * dst = (block_q2_Kx16*)t->data;
    const block_q2_K * src = (const block_q2_K*) data;

    block_q2_K dst_tmp[nrows_interleaved];

    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q2_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            // This loop gathers 16 separate blocks (one from each column)
            // that correspond to the same K-dimension chunk.
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }

            *dst++ = make_block_q2_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_0_to_q4_0_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    constexpr int nrows_interleaved = 16;

    block_q4_0x16 * dst = (block_q4_0x16*)t->data;
    const block_q4_0 * src = (const block_q4_0*) data;
    block_q4_0 dst_tmp[16];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q5_K_to_q5_K_8_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q5_Kx8 *     dst = (block_q5_Kx8 *) t->data;
    const block_q5_K * src = (const block_q5_K *) data;
    block_q5_K         dst_tmp[8];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q5_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q5_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_q6_K_to_q6_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q6_Kx8 * dst = (block_q6_Kx8 *)t->data;
    const block_q6_K * src = (const block_q6_K *) data;
    block_q6_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q6_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q6_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_q4_0_to_q4_0_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q4_0x8 * dst = (block_q4_0x8*)t->data;
    const block_q4_0 * src = (const block_q4_0*) data;
    block_q4_0 dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q8_0_to_q8_0_4_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 4;

    block_q8_0x4 *     dst = (block_q8_0x4 *) t->data;
    const block_q8_0 * src = (const block_q8_0 *) data;
    block_q8_0         dst_tmp[4];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q8_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned int blck_size_interleave) {
    block_q8_0x16 out;

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK8_0 * 16 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id     = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;
            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_q8_0_to_q8_0_16_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr int nrows_interleaved = 16;

    block_q8_0x16 *     dst = (block_q8_0x16 *) t->data;
    const block_q8_0 * src = (const block_q8_0 *) data;
    block_q8_0         dst_tmp[16];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q8_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static block_iq4_nlx4 make_block_iq4_nlx4(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 2 / blck_size_interleave;

    // TODO: this branch seems wrong
    //if (blck_size_interleave == 8) {
    //    for (int i = 0; i < end; ++i) {
    //        int src_id = i % 4;
    //        int src_offset = (i / 4) * blck_size_interleave;
    //        int dst_offset = i * blck_size_interleave;

    //        // Using memcpy to avoid unaligned memory accesses
    //        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
    //    }
    //} else
    if (blck_size_interleave == 4) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 4);

    const block_iq4_nl   * src = (const block_iq4_nl   *)data;
          block_iq4_nlx4 * dst = (      block_iq4_nlx4 *)t->data;

    block_iq4_nl dst_tmp[4];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 4;
    int nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_iq4_nlx8 make_block_iq4_nlx8(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 4 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 8;
            int src_offset = (i / 8) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 8);

    const block_iq4_nl   * src = (const block_iq4_nl   *)data;
          block_iq4_nlx8 * dst = (      block_iq4_nlx8 *)t->data;

    block_iq4_nl dst_tmp[8];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 8;
    int nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_iq4_nlx16 make_block_iq4_nlx16(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx16 out;

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 8 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 1);

    const block_iq4_nl    * src = (const block_iq4_nl   *)data;
          block_iq4_nlx16 * dst = (      block_iq4_nlx16 *)t->data;

    block_iq4_nl dst_tmp[16];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 16;
    int nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_mxfp4x4 make_block_mxfp4x4(block_mxfp4 * in, unsigned int blck_size_interleave) {
    block_mxfp4x4 out;

    for (int i = 0; i < 4; i++) {
        out.e[i] = in[i].e;
    }

    const int end = QK_MXFP4 * 2 / blck_size_interleave;

    if (blck_size_interleave == 4) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_mxfp4_to_mxfp4_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 4);

    const block_mxfp4   * src = (const block_mxfp4   *)data;
          block_mxfp4x4 * dst = (      block_mxfp4x4 *)t->data;

    block_mxfp4 dst_tmp[4];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 4;
    int nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_mxfp4));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_mxfp4x8 make_block_mxfp4x8(block_mxfp4 * in, unsigned int blck_size_interleave) {
    block_mxfp4x8 out;

    for (int i = 0; i < 8; i++) {
        out.e[i] = in[i].e;
    }

    const int end = QK_MXFP4 * 4 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 8;
            int src_offset = (i / 8) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_mxfp4_to_mxfp4_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 8);

    const block_mxfp4   * src = (const block_mxfp4   *)data;
          block_mxfp4x8 * dst = (      block_mxfp4x8 *)t->data;

    block_mxfp4 dst_tmp[8];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 8;
    int nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_mxfp4));

    if (t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

namespace ggml::cpu::repack {
// repack
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(struct ggml_tensor *, const void *, size_t);

// TODO: generalise.
template <> int repack<block_q4_0, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 4, data, data_size);
}

template <> int repack<block_q4_0, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_0, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_K, 4, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_8_bl(t, 4, data, data_size);
}

template <> int repack<block_q2_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q2_K_to_q2_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q5_K, 4, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_K_to_q5_K_8_bl(t, 4, data, data_size);
}

template <> int repack<block_q5_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_K_to_q5_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q6_K, 4, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q6_K_to_q6_K_8_bl(t, 4, data, data_size);
}

template <> int repack<block_q6_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q6_K_to_q6_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_iq4_nl, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_4_bl(t, 4, data, data_size);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

template <> int repack<block_iq4_nl, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_8_bl(t, 8, data, data_size);
}

template <> int repack<block_mxfp4, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_mxfp4_to_mxfp4_4_bl(t, 4, data, data_size);
}

template <> int repack<block_mxfp4, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_mxfp4_to_mxfp4_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q8_0, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_4_bl(t, 4, data, data_size);
}

template <> int repack<block_q8_0, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_4_bl(t, 8, data, data_size);
}

#if defined __riscv_zvfh
template <> int repack<block_q4_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_16_bl(t, 1, data, data_size);
}

template <> int repack<block_q4_K, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_16_bl(t, 1, data, data_size);
}

template <> int repack<block_iq4_nl, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_16_bl(t, 1, data, data_size);
}

template <> int repack<block_q8_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_16_bl(t, 1, data, data_size);
}

template <> int repack<block_q2_K, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q2_K_to_q2_K_16_bl(t, 1, data, data_size);
}
#endif

// gemv
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
void gemv(int, float *, size_t, const void *, const void *, int, int);

template <> void gemv<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(int          n,
                                            float *      s,
                                            size_t       bs,
                                            const void * vx,
                                            const void * vy,
                                            int          nr,
                                            int          nc) {
    ggml_gemv_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#if defined __riscv_zvfh
template <> void gemv<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

// gemm
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
void gemm(int, float *, size_t, const void *, const void *, int, int);

template <> void gemm<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemm<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(int          n,
                                            float *      s,
                                            size_t       bs,
                                            const void * vx,
                                            const void * vy,
                                            int          nr,
                                            int          nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#if defined __riscv_zvfh
template <> void gemm<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(struct ggml_tensor * t, const void * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> class tensor_traits : public tensor_traits_base {

    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    size = GGML_PAD(size, sizeof(int64_t)); // + padding for next block.

                    const int64_t ne02 = op->src[0]->ne[2]; // n_as, n_expert
                    const int64_t ne12 = op->src[1]->ne[2]; // n_tokens

                    const size_t sizeof_mmid_row_mapping = sizeof(int64_t);

                    size += sizeof_mmid_row_mapping*ne02*(ne12 + 1);

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                return true;
            case GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,
                                   int64_t               src0_start,
                                   int64_t               src0_end,
                                   int64_t               src1_start,
                                   int64_t               src1_end) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const size_t src1_col_stride = ggml_row_size(PARAM_TYPE, ne10);

        GGML_ASSERT(ne03 == 1 && ne13 == 1);
        GGML_ASSERT(ne12 % ne02 == 0);
        const int64_t r2 = ne12 / ne02;

        const int64_t i12 = src1_start / ne1;
        const int64_t i11 = src1_start - i12 * ne1;

        // Determine batch index
        const int64_t i02 = i12 / r2;

        const int64_t i1 = i11;
        const int64_t i2 = i12;

        const char * src0_ptr = (const char *) src0->data + i02 * nb02;
        const char * src1_ptr = (const char *) params->wdata + (i11 + i12 * ne11) * src1_col_stride;
        char *       dst_ptr  = ((char *) dst->data + (i1 * nb1 + i2 * nb2));

        const int64_t nrows = src1_end - src1_start;
        const int64_t ncols = src0_end - src0_start;

        GGML_ASSERT(src1_ptr + src1_col_stride * nrows <= (const char *) params->wdata + params->wsize);

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        if (nrows > 3) {
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *) (dst_ptr) + src0_start, nb1 / nb0,
                                                             src0_ptr + src0_start * nb01, src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        for (int iter = nrows - (nrows % 4); iter < nrows; iter++) {
            gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *) (dst_ptr + (iter * nb1)) + src0_start,
                                                             ne01, src0_ptr + src0_start * nb01,
                                                             src1_ptr + (src1_col_stride * iter), 1 /* nrows */, ncols);
        }
    }

    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        // TODO: General batched mul mat for 4D tensors
        // Currently only supports 3D tensors
        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3 == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_n_dims(op->src[0]) == 2);
        // GGML_ASSERT(ggml_n_dims(op->src[1]) == 2);

        char *       wdata = static_cast<char *>(params->wdata);
        const size_t nbw1  = ggml_row_size(PARAM_TYPE, ne10);
        const size_t nbw2  = nbw1 * ne11;

        assert(params->wsize >= nbw2 * ne12);

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // INFO: Quantization is done in planes to avoid extra complexity in chunking.
        // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
        // the planes are broadcast.
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            char * data_ptr  = (char *) src1->data + i12 * nb12;
            char * wdata_ptr = wdata + i12 * nbw2;

            for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>((float *) (data_ptr + i11 * nb11),
                                                            (void *) (wdata_ptr + i11 * nbw1), 4, ne10);
            }

            const int64_t i11_processed = ne11 - ne11 % 4;
            for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                from_float((float *) (data_ptr + i11 * nb11), (void *) (wdata_ptr + i11 * nbw1), ne10);
            }
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        const int64_t nr0 = ggml_nrows(op->src[0]);

        int     nth_scaled  = nth * 4;
        int64_t chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;
        int64_t nchunk0     = (nr0 + chunk_size0 - 1) / chunk_size0;

        // src1 is chunked only by full planes.
        // When we flatten we need to address dimensions not multiple of the q8 INTER_SIZE
        // to route them thorugh GEMV.
        // nchunk1 = ne12 also avoids messing the chunking for models with no 3d tensors
        // to avoid affecting their performance
        int64_t nchunk1 = ne12;

        // Ensure minimum chunk size to avoid alignment issues with high thread counts
        // Minimum chunk size should be at least NB_COLS to prevent overlapping chunks after alignment
        const int64_t min_chunk_size = NB_COLS;
        if (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_size && nr0 >= min_chunk_size) {
            nchunk0 = (nr0 + min_chunk_size - 1) / min_chunk_size;
        }

        int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        // Only increase nchunk0 to nth if it won't make chunks too small
        if (nth == 1 || ((nchunk0 < nth || disable_chunking) && (nr0 + nth - 1) / nth >= min_chunk_size)) {
            nchunk0 = nth;
            dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        }

        // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
        // This prevents creating too many tiny chunks that could overlap after alignment
        const int64_t max_nchunk = (nr0 + min_chunk_size - 1) / min_chunk_size;
        nchunk0                  = MIN(nchunk0, max_nchunk);

        if (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            int64_t src0_start = dr0 * ith0;
            int64_t src0_end   = MIN(src0_start + dr0, nr0);

            // full-plane range for src1
            int64_t src1_start = ith1 * ne11;
            int64_t src1_end = (ith1 + 1) * ne11;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            // The chunk size limiting above ensures chunks are large enough to prevent overlaps
            src0_start = (src0_start % NB_COLS) ? src0_start + NB_COLS - (src0_start % NB_COLS) : src0_start;
            src0_end   = (src0_end % NB_COLS) ? src0_end + NB_COLS - (src0_end % NB_COLS) : src0_end;
            src0_end   = MIN(src0_end, ne01);

            // Make sure current plane is the last one before exiting
            if (src0_start >= src0_end) {
                current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                continue;
            }

            forward_mul_mat_one_chunk(params, dst, src0_start, src0_end, src1_start, src1_end);

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    void forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_size(src0->type));
        GGML_ASSERT(nb10 == ggml_type_size(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        const int n_ids = ids->ne[0]; // n_expert_used
        const int n_as  = ne02;       // n_expert

        const size_t nbw1 = ggml_row_size(PARAM_TYPE, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        GGML_ASSERT(params->wsize >=
                (GGML_PAD(nbw3, sizeof(int64_t)) +
                 n_as*(ne12 + 1)*sizeof(mmid_row_mapping))
                );

        auto * wdata          = (char *)params->wdata;
        auto * wdata_src1_end = (char *)wdata + GGML_PAD(nbw3, sizeof(int64_t));

        // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
        auto * matrix_row_counts = (int64_t *) (wdata_src1_end);                                        // [n_as]
        struct mmid_row_mapping * matrix_rows = (struct mmid_row_mapping *) (matrix_row_counts + n_as); // [n_as][ne12]

        // src1: float32 => param type
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((float *)((char *) src1->data + i12 * nb12 + i11 * nb11),
                           (void *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ne12 + (i1)]

        if (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute each matrix multiplication in sequence
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            const int64_t cne1 = matrix_row_counts[cur_a];

            if (cne1 == 0) {
                continue;
            }

            const auto * src0_cur = (const char *) src0->data + cur_a*nb02;

            //const int64_t nr0 = ne01; // src0 rows
            const int64_t nr1 = cne1; // src1 rows

            int64_t src0_cur_start = (ith * ne01) / nth;
            int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            src0_cur_start = (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
            src0_cur_end   = (src0_cur_end   % NB_COLS) ? src0_cur_end   + NB_COLS - (src0_cur_end   % NB_COLS) : src0_cur_end;
            if (src0_cur_end > ne01) {
                src0_cur_end = ne01;
            }

            if (src0_cur_start >= src0_cur_end) {
                return;
            }

            for (int ir1 = 0; ir1 < nr1; ir1++) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                const int id = row_mapping.i1;  // selected expert index

                const int64_t i11 = id % ne11;
                const int64_t i12 = row_mapping.i2;  // row index in src1

                const int64_t i1 = id;               // selected expert index
                const int64_t i2 = i12;              // row

                const auto * src1_col = (const char *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                    ne00, (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, ne01,
                    src0_cur + src0_cur_start * nb01, src1_col, 1, src0_cur_end - src0_cur_start);
            }
        }
#undef MMID_MATRIX_ROW
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::repack::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

}  // namespace ggml::cpu::repack

static const ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(const struct ggml_tensor * cur) {
    // instance for Q4
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;

    // instance for Q4_K
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

    // instance for Q5_K
    static const ggml::cpu::repack::tensor_traits<block_q5_K, 4, 8, GGML_TYPE_Q8_K> q5_K_8x4_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q5_K, 8, 8, GGML_TYPE_Q8_K> q5_K_8x8_q8_K;

    // instance for Q6_K
    static const ggml::cpu::repack::tensor_traits<block_q6_K, 4, 8, GGML_TYPE_Q8_K> q6_K_8x4_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q6_K, 8, 8, GGML_TYPE_Q8_K> q6_K_8x8_q8_K;

    // instance for Q2
    static const ggml::cpu::repack::tensor_traits<block_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // instance for IQ4
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;

    // instance for MXFP4
    static const ggml::cpu::repack::tensor_traits<block_mxfp4, 4, 4, GGML_TYPE_Q8_0> mxfp4_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;

    // instance for Q8_0
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_4x8_q8_0;

    // instances for RISC-V
    //
    // These implement outer-product style matrix multiplication kernels with
    // an interleave of 1.
#if defined __riscv_zvfh
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#endif

    if (cur->type == GGML_TYPE_Q4_0) {
        if (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_0_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_0_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_Q4_K) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x4_q8_K;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_Q2_K) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &q2_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q2_K_16x1_q8_K; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_Q5_K) {
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q5_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 8 == 0) {
                return &q5_K_8x4_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_Q6_K) {
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q6_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 8 == 0) {
                return &q6_K_8x4_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_IQ4_NL) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &iq4_nl_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &iq4_nl_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &iq4_nl_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_MXFP4) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &mxfp4_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &mxfp4_4x4_q8_0;
            }
        }
    } else if (cur->type == GGML_TYPE_Q8_0) {
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &q8_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &q8_0_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    }

    return nullptr;
}

static enum ggml_status ggml_backend_cpu_repack_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cpu_repack_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                       const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::repack::tensor_traits_base *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_REPACK";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_repack_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_repack_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_repack_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::repack {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->buffer &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type() &&
                ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } else if (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->buffer
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()
                && ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()) {
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        return nullptr;
    }
};
}  // namespace ggml::cpu::repack

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_repack = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_repack_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_repack_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_repack_buffer_type_get_alignment,
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
                           /* .is_host          = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::repack::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_repack;
}


// ---- x86 repack implementation ----

#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common-defs.inc"
#include "ggml-backend-impl-defs.inc"

#include "ggml-impl-defs.inc"
#include "ggml-cpu.h.inc"
#include "ggml-cpu-impl-defs.inc"
#include "simd-mappings-defs.inc"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

#if defined(__AVX__)
#if defined(__F16C__)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)     _mm512_cvtph_ps(_mm256_set_m128i(_mm_loadu_si128((const __m128i *)(y)), _mm_loadu_si128((const __m128i *)(x))))
#define GGML_F32Cx16_REPEAT_LOAD(x)  _mm512_cvtph_ps(_mm256_set_m128i(x, x))
#endif
// the  _mm256_cvt intrinsics require F16C
#define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     _mm256_cvtph_ps(_mm_shuffle_epi32(_mm_maskload_epi32((int const*)(x), loadMask), 68))
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)     _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask))
#else
#if defined(__AVX512F__)
static inline __m512 __avx512_f32cx8x2_load(ggml_fp16_t *x, ggml_fp16_t *y) {
    float tmp[16];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    for (int i = 0; i < 8; i++) {
        tmp[i + 8] = GGML_CPU_FP16_TO_FP32(y[i]);
    }

    return _mm512_loadu_ps(tmp);
}
static inline __m512 __avx512_repeat_f32cx16_load(__m128i x) {
    float tmp[16];
    uint16_t tmphalf[8];
    _mm_storeu_si128((__m128i*)tmphalf, x);

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 4] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 8] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 12] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
    }

    return _mm512_loadu_ps(tmp);
}
#endif
static inline __m256 __avx_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_repeat_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
        tmp[i + 4] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_rearranged_f32cx8_load(ggml_fp16_t *x, __m128i arrangeMask) {
    uint16_t tmphalf[8];
    float tmp[8];

    _mm_storeu_si128((__m128i*)tmphalf, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask));
    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
    }

    return _mm256_loadu_ps(tmp);
}

#define GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     __avx_repeat_f32cx8_load(x)
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)     __avx_rearranged_f32cx8_load(x, arrangeMask)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)     __avx512_f32cx8x2_load(x, y)
#define GGML_F32Cx16_REPEAT_LOAD(x)  __avx512_repeat_f32cx16_load(x)
#endif
#endif
#endif

static inline int x86_nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

#if defined(__AVX2__) || defined(__AVX512F__)
#if defined(__AVX512F__)
// add int16_t pairwise and return as 512 bit int vector, then add the accumulator
static inline __m512i sum_i16_pairs_acc_int32x16(const __m512i acc, const __m512i x) {
    const __m512i ones = _mm512_set1_epi16(1);
    return _mm512_add_epi32(acc, _mm512_madd_epi16(ones, x));
}

static inline __m512i mul_sum_us8_pairs_acc_int32x16(const __m512i acc, const __m512i ax, const __m512i sy) {
#if defined(__AVX512VNNI__)
    return _mm512_dpbusd_epi32(acc, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x16(acc, dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as 512 bit int vector，then add the accumulator
static inline __m512i mul_sum_i8_pairs_acc_int32x16(const __m512i acc, const __m512i x, const __m512i y) {
    const __m512i zero = _mm512_setzero_si512();
    // Get absolute values of x vectors
    const __m512i ax = _mm512_abs_epi8(x);
    // Sign the values of the y vectors
    __mmask64 blt0 = _mm512_movepi8_mask(x);
    const __m512i sy = _mm512_mask_sub_epi8(y, blt0, zero, y);
    return mul_sum_us8_pairs_acc_int32x16(acc, ax, sy);
}
#endif

// add int16_t pairwise and return as 256 bit int vector, then add the accumulator
static inline __m256i sum_i16_pairs_acc_int32x8(const __m256i acc, const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(ones, x));
}

static inline __m256i mul_sum_us8_pairs_acc_int32x8(const __m256i acc, const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, ax, sy);
#elif defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x8(acc, dot);
#endif
}

// Integer variant of the function defined in the ggml quantization implementation
// multiply int8_t, add results pairwise twice and return as 256 bit int vector, then add the accumulator
static inline __m256i mul_sum_i8_pairs_acc_int32x8(const __m256i acc, const __m256i x, const __m256i y) {
#if defined(__AVXVNNIINT8__)
    return _mm256_dpbssd_epi32(acc, x, y);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_acc_int32x8(acc, ax, sy);
#endif
}
#endif

void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__AVX2__) || defined(__AVX__)
    float id[4];
    __m256 srcv[4][4];
    __m256 idvec[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            // Load elements into 4 AVX vectors
            __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 32 );
            __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 8 );
            __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 16 );
            __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 24 );

            // Compute max(abs(e)) for the block
            const __m256 signBit = _mm256_set1_ps( -0.0f );
            __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

            __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
            max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
            max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
            const float maxScalar = _mm_cvtss_f32( max4 );

            // Divided by 127.f to mirror results in quantize_row_q8_0
            const float d = maxScalar  / 127.f;
            id[row_iter] = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f; //d ? 1.0f / d : 0.0f;

            // Store the scale for the individual block
            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);

            // Store the values in blocks of eight values - Aim is to use these later for block interleaving
            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;
            idvec[row_iter] = _mm256_set1_ps(id[row_iter]);
        }

        // The loop iterates four times - The aim is to get 4 corresponding chunks of eight bytes from the original weight blocks that are interleaved
        for (int j = 0; j < 4; j++) {
            // Apply the multiplier
            __m256 v0 = _mm256_mul_ps(srcv[0][j], idvec[0]);
            __m256 v1 = _mm256_mul_ps(srcv[1][j], idvec[1]);
            __m256 v2 = _mm256_mul_ps(srcv[2][j], idvec[2]);
            __m256 v3 = _mm256_mul_ps(srcv[3][j], idvec[3]);

            // Round to nearest integer
            v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
            v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
            v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
            v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

            // Convert floats to integers
            __m256i i0 = _mm256_cvtps_epi32( v0 );
            __m256i i1 = _mm256_cvtps_epi32( v1 );
            __m256i i2 = _mm256_cvtps_epi32( v2 );
            __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
            // Convert int32 to int16
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            // Convert int16 to int8
            i0 = _mm256_packs_epi16( i0, i2 );

            //  Permute and store the quantized weights in the required order after the pack instruction
            const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );

            _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
#else
            // Since we don't have in AVX some necessary functions,
            // we split the registers in half and call AVX2 analogs from SSE
            __m128i ni0 = _mm256_castsi256_si128( i0 );
            __m128i ni1 = _mm256_extractf128_si256( i0, 1);
            __m128i ni2 = _mm256_castsi256_si128( i1 );
            __m128i ni3 = _mm256_extractf128_si256( i1, 1);
            __m128i ni4 = _mm256_castsi256_si128( i2 );
            __m128i ni5 = _mm256_extractf128_si256( i2, 1);
            __m128i ni6 = _mm256_castsi256_si128( i3 );
            __m128i ni7 = _mm256_extractf128_si256( i3, 1);

            // Convert int32 to int16
            ni0 = _mm_packs_epi32( ni0, ni1 );
            ni2 = _mm_packs_epi32( ni2, ni3 );
            ni4 = _mm_packs_epi32( ni4, ni5 );
            ni6 = _mm_packs_epi32( ni6, ni7 );
            // Convert int16 to int8
            ni0 = _mm_packs_epi16( ni0, ni2 );
            ni4 = _mm_packs_epi16( ni4, ni6 );
            _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j), ni0);
            _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j + 16), ni4);
#endif
        }
    }

#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x8_generic(x, vy, k);
#endif
}

void ggml_quantize_mat_q8_K_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

#if defined(__AVX2__)
    float iscale[4];
    __m256 srcv[4][32];
    __m256 iscale_vec[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            // Load elements into 4 AVX vectors
            __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 256 );
            __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 256 + 8 );
            __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 256 + 16 );
            __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 256 + 24 );

            // Compute max(abs(e)) for the block
            const __m256 signBit = _mm256_set1_ps( -0.0f );
            __m256 abs0 = _mm256_andnot_ps( signBit, v0 );
            __m256 abs1 = _mm256_andnot_ps( signBit, v1 );
            __m256 abs2 = _mm256_andnot_ps( signBit, v2 );
            __m256 abs3 = _mm256_andnot_ps( signBit, v3 );

            __m256 maxAbs = _mm256_max_ps( abs0, abs1 );
            maxAbs = _mm256_max_ps( maxAbs, abs2 );
            maxAbs = _mm256_max_ps( maxAbs, abs3 );

            __m256 mask0 = _mm256_cmp_ps( maxAbs, v0, _CMP_EQ_OQ );
            __m256 mask1 = _mm256_cmp_ps( maxAbs, v1, _CMP_EQ_OQ );
            __m256 mask2 = _mm256_cmp_ps( maxAbs, v2, _CMP_EQ_OQ );
            __m256 mask3 = _mm256_cmp_ps( maxAbs, v3, _CMP_EQ_OQ );

            __m256 maskAbs = _mm256_or_ps(_mm256_or_ps(mask0, mask1),_mm256_or_ps(mask2, mask3));

            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;

            for (int sb = 1; sb < 8; sb++) {
                // Temporarily stores absolute quant values
                __m256 tempAbs = maxAbs;

                // Load elements into 4 AVX vectors
                __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32);
                __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32 + 8 );
                __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32 + 16 );
                __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32 + 24 );

                // Compute max(abs(e)) for the block
                __m256 abs0 = _mm256_andnot_ps( signBit, v0 );
                __m256 abs1 = _mm256_andnot_ps( signBit, v1 );
                __m256 abs2 = _mm256_andnot_ps( signBit, v2 );
                __m256 abs3 = _mm256_andnot_ps( signBit, v3 );

                maxAbs = _mm256_max_ps( maxAbs, abs0 );
                maxAbs = _mm256_max_ps( maxAbs, abs1 );
                maxAbs = _mm256_max_ps( maxAbs, abs2 );
                maxAbs = _mm256_max_ps( maxAbs, abs3 );

                __m256 mask_prev = _mm256_cmp_ps( tempAbs, maxAbs, _CMP_EQ_OQ );
                maskAbs = _mm256_and_ps( maskAbs, mask_prev );

                mask0 = _mm256_cmp_ps( maxAbs, v0, _CMP_EQ_OQ );
                mask1 = _mm256_cmp_ps( maxAbs, v1, _CMP_EQ_OQ );
                mask2 = _mm256_cmp_ps( maxAbs, v2, _CMP_EQ_OQ );
                mask3 = _mm256_cmp_ps( maxAbs, v3, _CMP_EQ_OQ );

                __m256 mask_curr = _mm256_or_ps(_mm256_or_ps(mask0, mask1),_mm256_or_ps(mask2, mask3));
                maskAbs =  _mm256_or_ps(maskAbs, mask_curr);

                srcv[row_iter][sb * 4] = v0;
                srcv[row_iter][sb * 4 + 1] = v1;
                srcv[row_iter][sb * 4 + 2] = v2;
                srcv[row_iter][sb * 4 + 3] = v3;
            }

            __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
            max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
            max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
            const float maxScalar = _mm_cvtss_f32( max4 );

            __m256 maxScalarVec = _mm256_set1_ps(maxScalar);

            __m256 mask_next = _mm256_cmp_ps( maxScalarVec, maxAbs, _CMP_EQ_OQ );
            __m256 finalMask = _mm256_and_ps(maskAbs, mask_next);

            const int mask = _mm256_movemask_ps(finalMask);
            iscale[row_iter] = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;

            if(mask) {
                iscale[row_iter] = ( maxScalar != 0.0f ) ? -127.f / maxScalar: 0.0f;
            }

            y[i].d[row_iter] = maxScalar ? 1/iscale[row_iter] : 0;
            iscale_vec[row_iter] = _mm256_set1_ps(iscale[row_iter]);
        }

        __m256i quants_interleaved[32];
        for (int j = 0; j < 32; j++) {
            // Apply the multiplier
            __m256 v0 = _mm256_mul_ps(srcv[0][j], iscale_vec[0]);
            __m256 v1 = _mm256_mul_ps(srcv[1][j], iscale_vec[1]);
            __m256 v2 = _mm256_mul_ps(srcv[2][j], iscale_vec[2]);
            __m256 v3 = _mm256_mul_ps(srcv[3][j], iscale_vec[3]);

            // Round to nearest integer
            v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
            v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
            v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
            v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

            // Convert floats to integers
            __m256i i0 = _mm256_cvtps_epi32( v0 );
            __m256i i1 = _mm256_cvtps_epi32( v1 );
            __m256i i2 = _mm256_cvtps_epi32( v2 );
            __m256i i3 = _mm256_cvtps_epi32( v3 );

            // Convert int32 to int16
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            // Convert int16 to int8
            i0 = _mm256_packs_epi16( i0, i2 );

            //  Permute and store the quantized weights in the required order after the pack instruction
            const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );

            _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
            quants_interleaved[j] = i0;
        }

        // Masks to shuffle the quants of corresponding sub blocks for rearranging quants for vectorized bsums computation
        __m256i shuffle_mask_sb2 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 15));
        shuffle_mask_sb2 = _mm256_permute2f128_si256(shuffle_mask_sb2, shuffle_mask_sb2, 0);
        __m256i shuffle_mask_sb3 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 8, 9, 14, 15));
        shuffle_mask_sb3 = _mm256_permute2f128_si256(shuffle_mask_sb3, shuffle_mask_sb3, 0);
        __m256i shuffle_mask_sb4 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13, 8, 9));
        shuffle_mask_sb4 = _mm256_permute2f128_si256(shuffle_mask_sb4, shuffle_mask_sb4, 0);

        for (int k = 0; k < 4; k++) {
            // Quants from four different sub blocks are taken
            __m256i q0 = quants_interleaved[k * 8 + 0];
            __m256i q1 = quants_interleaved[k * 8 + 1];
            __m256i q2 = quants_interleaved[k * 8 + 2];
            __m256i q3 = quants_interleaved[k * 8 + 3];
            __m256i q4 = quants_interleaved[k * 8 + 4];
            __m256i q5 = quants_interleaved[k * 8 + 5];
            __m256i q6 = quants_interleaved[k * 8 + 6];
            __m256i q7 = quants_interleaved[k * 8 + 7];


            // The below code block has the first half of different sub blocks shuffled and blended so as to process 2 values from each sub block at a time
            __m256i sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
            __m256i sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
            __m256i sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
            sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
            __m256i sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
            sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

            __m256i one = _mm256_set1_epi8(1);
            __m256i bsums_r1 = _mm256_maddubs_epi16(one, sb_h1_interleaved);

            for (int l = 0; l < 3; l++) {
                // Quants value shifted to process next two values from each sub block
                q0 = _mm256_srli_epi64(q0, 16);
                q2 = _mm256_srli_epi64(q2, 16);
                q4 = _mm256_srli_epi64(q4, 16);
                q6 = _mm256_srli_epi64(q6, 16);

                sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
                sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
                sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
                sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
                sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
                sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

                bsums_r1 = _mm256_add_epi16(bsums_r1, _mm256_maddubs_epi16(one, sb_h1_interleaved));
            }

            // The below code block has the second half of different sub blocks shuffled and blended so as to process 2 values from each sub block at a time
            __m256i sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
            __m256i sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
            __m256i sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
            sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
            __m256i sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
            sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

            __m256i bsums_r2 = _mm256_maddubs_epi16(one, sb_h2_interleaved);

            for (int l = 0; l < 3; l++) {
                // Quants value shifted to process next two values from each sub block
                q1 = _mm256_srli_epi64(q1, 16);
                q3 = _mm256_srli_epi64(q3, 16);
                q5 = _mm256_srli_epi64(q5, 16);
                q7 = _mm256_srli_epi64(q7, 16);

                sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
                sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
                sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
                sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
                sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
                sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

                bsums_r2 = _mm256_add_epi16(bsums_r2, _mm256_maddubs_epi16(one, sb_h2_interleaved));
            }

            // Overall bsums in interleaved fashion computed by adding results of both halves
            __m256i bsums_r = _mm256_add_epi16(bsums_r1, bsums_r2);
            _mm256_storeu_si256((__m256i *)(y[i].bsums + 16 * k), bsums_r);
        }
    }

#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_K_4x8_generic(x, vy, k);
#endif
}

//
// GEMV/GEMM templates
//

#if defined(__AVX2__) || defined(__AVX512F__)

// GEMV for 8x blocks of 32 4-bit quants with a single scale factor per block
template<typename block_tx8>
static void gemv_q4_b32_8x8_q8_0_lut_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, __m256i signextendlut) {
    static_assert(
            std::is_same_v<block_tx8, block_q4_0x8> ||
            std::is_same_v<block_tx8, block_iq4_nlx8> ||
            std::is_same_v<block_tx8, block_mxfp4x8>,
            "Unsupported block type");

    const int qk = QK8_0;
    const int nb = n / qk;

    UNUSED(bs);

    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // Permute mask used for easier vector processing at later stages
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    int64_t b_nb = n / 32;

    const block_tx8  * b_ptr_start = (const block_tx8  *)vx;
    const block_q8_0 * a_ptr_start = (const block_q8_0 *)vy;

    // Process Q8_0 blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_0 format
        const block_q8_0 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight blocks at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_tx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulator
            __m256 acc_row = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {
                // Load 8 blocks of 32 interleaved as 8 bytes (B0 - B7)
                const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 1);
                const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 2);
                const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 3);

                // 4-bit -> 8-bit - Sign is maintained
                const __m256i rhs_vec_0123_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_0123_0, m4b)); // B0(0-7) B1(0-7) B2(0-7) B3(0-7)
                const __m256i rhs_vec_4567_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_4567_0, m4b)); // B4(0-7) B5(0-7) B6(0-7) B7(0-7)
                const __m256i rhs_vec_0123_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_0123_1, m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)
                const __m256i rhs_vec_4567_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_4567_1, m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)

                const __m256i rhs_vec_0123_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b)); // B0(16-23) B1(16-23) B2(16-23) B3(16-23)
                const __m256i rhs_vec_4567_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b)); // B4(16-23) B5(16-23) B6(16-23) B7(16-23)
                const __m256i rhs_vec_0123_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b)); // B0(24-31) B1(24-31) B2(24-31) B3(24-31)
                const __m256i rhs_vec_4567_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b)); // B4(24-31) B5(24-31) B6(24-31) B7(24-31)

                // Load the scale values for the 8 blocks interleaved in block_tx8
                __m256 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    const __m128i changemask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
                    col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, changemask);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    // Load 8 E8M0 exponents and convert to float via LUT
                    // Rearranged to match changemask order: 0,4,1,5,2,6,3,7
                    col_scale_f32 = _mm256_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[0]));
                }

                // Load and convert to FP32 scale from block_q8_0
                const __m256 row_scale_f32 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(a_ptr[b].d));

                // Load the block values in block_q8_0 in batches of 16 bytes and replicate the same across 256 bit vector
                __m256i lhs_vec_0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)a_ptr[b].qs));
                __m256i lhs_vec_1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16)));

                lhs_vec_0 = _mm256_permute2f128_si256(lhs_vec_0, lhs_vec_0, 0); // A0 (0-15) A0(0-15)
                lhs_vec_1 = _mm256_permute2f128_si256(lhs_vec_1, lhs_vec_1, 0); // A0 (16-31) A0(16-31))

                __m256i iacc = _mm256_setzero_si256();

                // Dot product done within 32 bit lanes and accumulated in the same vector
                // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                // ...........................................................................
                // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_0 ,_mm256_shuffle_epi32(rhs_vec_4567_0, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_0, 177) ,rhs_vec_4567_0, 170), _mm256_shuffle_epi32(lhs_vec_0, 85));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_1 ,_mm256_shuffle_epi32(rhs_vec_4567_1, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_1, 177) ,rhs_vec_4567_1, 170), _mm256_shuffle_epi32(lhs_vec_0, 255));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_2 ,_mm256_shuffle_epi32(rhs_vec_4567_2, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_2, 177) ,rhs_vec_4567_2, 170), _mm256_shuffle_epi32(lhs_vec_1, 85));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_3 ,_mm256_shuffle_epi32(rhs_vec_4567_3, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_3, 177) ,rhs_vec_4567_3, 170), _mm256_shuffle_epi32(lhs_vec_1, 255));

                // Accumulated values multiplied with appropriate scales
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
            }

            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), acc_row);
        }
    }
}

// GEMM for 8x blocks of 32 4-bit quants with a single scale factor per block
template<typename block_tx8>
static void gemm_q4_b32_8x8_q8_0_lut_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, __m256i signextendlut) {
    static_assert(
            std::is_same_v<block_tx8, block_q4_0x8> ||
            std::is_same_v<block_tx8, block_iq4_nlx8> ||
            std::is_same_v<block_tx8, block_mxfp4x8>,
            "Unsupported block type");

    const int qk = QK8_0;
    const int nb = n / qk;

    const block_tx8    * b_ptr_start = (const block_tx8    *)vx;
    const block_q8_0x4 * a_ptr_start = (const block_q8_0x4 *)vy;

    int64_t b_nb = n / 32;
    int64_t y = 0;
    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m128i loadMask = _mm_blend_epi32(_mm_setzero_si128(), _mm_set1_epi32(0xFFFFFFFF), 3);
    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr%16; // Used to align nr with boundary of 16
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
    int anc = nc - nc%16; // Used to align nc with boundary of 16
                          // Mask to mask out nibbles from packed bytes expanded to 512 bit length
    const __m512i m4bexpanded = _mm512_set1_epi8(0x0F);
    // Lookup table to convert signed nibbles to signed bytes expanded to 512 bit length
    __m512i signextendlutexpanded = _mm512_inserti32x8(_mm512_castsi256_si512(signextendlut), signextendlut, 1);

    // Take group of four block_q8_0x4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_0x4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of two block_tx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_tx8 * b_ptr_0 = b_ptr_start + ((x)     * b_nb);
            const block_tx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the sixteen blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....BE,BF
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 96));

                const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs));
                const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 32));
                const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 64));
                const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 96));

                // Save the values in the following vectors in the formats B0B1B4B5B8B9BCBD, B2B3B6B7BABBBEBF for further processing and storing of values
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);

                const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                // 4-bit -> 8-bit - Sign is maintained
                const __m512i rhs_mat_014589CD_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7) B8(0-7) B9(0-7) BC(0-7) BD(0-7)
                const __m512i rhs_mat_2367ABEF_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7) BA(0-7) BB(0-7) BE(0-7) BF(0-7)

                const __m512i rhs_mat_014589CD_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15) B9(8-15) BC(8-15) BD(8-15)
                const __m512i rhs_mat_2367ABEF_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15) BB(8-15) BE(8-15) BF(8-15)

                const __m512i rhs_mat_014589CD_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23) B8(16-23) B9(16-23) BC(16-23) BD(16-23)
                const __m512i rhs_mat_2367ABEF_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23) BA(16-23) BB(16-23) BE(16-23) BF(16-23)

                const __m512i rhs_mat_014589CD_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31) B8(24-31) B9(24-31) BC(24-31) BD(24-31)
                const __m512i rhs_mat_2367ABEF_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31) BA(24-31) BB(24-31) BE(24-31) BF(24-31)

                // Shuffle pattern one - right side input
                const __m512i rhs_mat_014589CD_0_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)136); //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3) B8(0-3) B9(0-3) B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3) BD(0-3)
                const __m512i rhs_mat_2367ABEF_0_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)136); //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3) BA(0-3) BB(0-3) BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3) BF(0-3)

                const __m512i rhs_mat_014589CD_1_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)136); //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11) B8(8-11) B9(8-11) B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11) BD(8-11)
                const __m512i rhs_mat_2367ABEF_1_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)136); //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11) BA(8-11) BB(8-11) BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11) BF(8-11)

                const __m512i rhs_mat_014589CD_2_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)136); //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19) B8(16-19) B9(16-19) B8(16-19) B9(16-19) BC(16-19) BD(16-19) BC(16-19) BD(16-19)
                const __m512i rhs_mat_2367ABEF_2_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)136); //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19) BA(16-19) BB(16-19) BA(16-19) BB(16-19) BE(16-19) BF(16-19) BE(16-19) BF(16-19)

                const __m512i rhs_mat_014589CD_3_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)136); //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27) B8(24-27) B9(24-27) B8(24-27) B9(24-27) BC(24-27) BD(24-27) BC(24-27) BD(24-27)
                const __m512i rhs_mat_2367ABEF_3_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)136); //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27) BA(24-27) BB(24-27) BA(24-27) BB(24-27) BE(24-27) BF(24-27) BE(24-27) BF(24-27)

                // Shuffle pattern two - right side input

                const __m512i rhs_mat_014589CD_0_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)221); //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7) B8(4-7) B9(4-7) B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7) BD(4-7)
                const __m512i rhs_mat_2367ABEF_0_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)221); //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7) BA(4-7) BB(4-7) BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7) BF(4-7)

                const __m512i rhs_mat_014589CD_1_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)221); //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15) B8(12-15) B9(12-15) B8(12-15) B9(12-15) BC(12-15) BD(12-15) BC(12-15) BD(12-15)
                const __m512i rhs_mat_2367ABEF_1_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)221); //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15) BA(12-15) BB(12-15) BA(12-15) BB(12-15) BE(12-15) BF(12-15) BE(12-15) BF(12-15)

                const __m512i rhs_mat_014589CD_2_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)221); //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23) B8(20-23) B9(20-23) B8(20-23) B9(20-23) BC(20-23) BD(20-23) BC(20-23) BD(20-23)
                const __m512i rhs_mat_2367ABEF_2_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)221); //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23) BA(20-23) BB(20-23) BA(20-23) BB(20-23) BE(20-23) BF(20-23) BE(20-23) BF(20-23)

                const __m512i rhs_mat_014589CD_3_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)221); //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31) B8(28-31) B9(28-31) B8(28-31) B9(28-31) BC(28-31) BD(28-31) BC(28-31) BD(28-31)
                const __m512i rhs_mat_2367ABEF_3_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)221); //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31) BA(28-31) BB(28-31) BA(28-31) BB(28-31) BE(28-31) BF(28-31) BE(28-31) BF(28-31)

                // Scale values - Load the weight scale values of two block_tx8
                __m512 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    //TODO: simd-ify
                    col_scale_f32 = _mm512_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[0]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[0]));
                }

                // Process LHS in pairs of rows
                for (int rp = 0; rp < 4; rp++) {

                    // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                    __m256i lhs_mat_ymm_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
                    __m256i lhs_mat_ymm_01_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 0);
                    __m256i lhs_mat_ymm_23_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 17);
                    __m256i lhs_mat_ymm_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
                    __m256i lhs_mat_ymm_01_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 0);
                    __m256i lhs_mat_ymm_23_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 17);
                    __m256i lhs_mat_ymm_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
                    __m256i lhs_mat_ymm_01_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 0);
                    __m256i lhs_mat_ymm_23_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 17);
                    __m256i lhs_mat_ymm_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
                    __m256i lhs_mat_ymm_01_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 0);
                    __m256i lhs_mat_ymm_23_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 17);

                    __m512i lhs_mat_01_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_0), lhs_mat_ymm_01_0, 1);
                    __m512i lhs_mat_23_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_0), lhs_mat_ymm_23_0, 1);
                    __m512i lhs_mat_01_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_1), lhs_mat_ymm_01_1, 1);
                    __m512i lhs_mat_23_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_1), lhs_mat_ymm_23_1, 1);
                    __m512i lhs_mat_01_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_2), lhs_mat_ymm_01_2, 1);
                    __m512i lhs_mat_23_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_2), lhs_mat_ymm_23_2, 1);
                    __m512i lhs_mat_01_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_3), lhs_mat_ymm_01_3, 1);
                    __m512i lhs_mat_23_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_3), lhs_mat_ymm_23_3, 1);

                    // Shuffle pattern one - left side input

                    const __m512i lhs_mat_01_0_sp1 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                    const __m512i lhs_mat_23_0_sp1 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                    const __m512i lhs_mat_01_1_sp1 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                    const __m512i lhs_mat_23_1_sp1 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                    const __m512i lhs_mat_01_2_sp1 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                    const __m512i lhs_mat_23_2_sp1 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                    const __m512i lhs_mat_01_3_sp1 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                    const __m512i lhs_mat_23_3_sp1 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                    // Shuffle pattern two - left side input

                    const __m512i lhs_mat_01_0_sp2 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                    const __m512i lhs_mat_23_0_sp2 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                    const __m512i lhs_mat_01_1_sp2 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                    const __m512i lhs_mat_23_1_sp2 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                    const __m512i lhs_mat_01_2_sp2 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                    const __m512i lhs_mat_23_2_sp2 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                    const __m512i lhs_mat_01_3_sp2 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                    const __m512i lhs_mat_23_3_sp2 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    // Resembles MMLAs into 2x2 matrices in ARM Version
                    const __m512i zero = _mm512_setzero_epi32();
                    __m512i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1);
                    __m512i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1);
                    __m512i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1);
                    __m512i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1);
                    __m512i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2);
                    __m512i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2);
                    __m512i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2);
                    __m512i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m512i iacc_mat_00 = _mm512_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                    __m512i iacc_mat_01 = _mm512_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                    __m512i iacc_mat_10 = _mm512_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                    __m512i iacc_mat_11 = _mm512_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                    // Straighten out to make 4 row vectors
                    __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                    __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                    // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                    const __m128i row_scale_f16 = _mm_shuffle_epi32(_mm_maskload_epi32((int const*)(a_ptrs[rp][b].d), loadMask), 68);
                    const __m512 row_scale_f32 = GGML_F32Cx16_REPEAT_LOAD(row_scale_f16);

                    // Multiply with appropriate scales and accumulate
                    acc_rows[rp * 4]     = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),   acc_rows[rp * 4]);
                    acc_rows[rp * 4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),  acc_rows[rp * 4 + 1]);
                    acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                    acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }

    // Take a block_q8_0x4 structures at each pass of the loop and perform dot product operation
    for (; y < nr / 4; y ++) {
        const block_q8_0x4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of two block_tx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_tx8 * b_ptr_0 = b_ptr_start + ((x)     * b_nb);
            const block_tx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the sixteen blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....BE,BF
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 96));

                const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs));
                const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 32));
                const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 64));
                const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 96));

                // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);

                const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                // 4-bit -> 8-bit - Sign is maintained
                const __m512i rhs_mat_014589CD_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7) B8(0-7) B9(0-7) BC(0-7) BD(0-7)
                const __m512i rhs_mat_2367ABEF_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7) BA(0-7) BB(0-7) BE(0-7) BF(0-7)

                const __m512i rhs_mat_014589CD_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15) B9(8-15) BC(8-15) BD(8-15)
                const __m512i rhs_mat_2367ABEF_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15) BB(8-15) BE(8-15) BF(8-15)

                const __m512i rhs_mat_014589CD_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23) B8(16-23) B9(16-23) BC(16-23) BD(16-23)
                const __m512i rhs_mat_2367ABEF_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23) BA(16-23) BB(16-23) BE(16-23) BF(16-23)

                const __m512i rhs_mat_014589CD_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31) B8(24-31) B9(24-31) BC(24-31) BD(24-31)
                const __m512i rhs_mat_2367ABEF_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31) BA(24-31) BB(24-31) BE(24-31) BF(24-31)

                // Shuffle pattern one - right side input
                const __m512i rhs_mat_014589CD_0_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)136); //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3) B8(0-3) B9(0-3) B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3) BD(0-3)
                const __m512i rhs_mat_2367ABEF_0_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)136); //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3) BA(0-3) BB(0-3) BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3) BF(0-3)

                const __m512i rhs_mat_014589CD_1_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)136); //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11) B8(8-11) B9(8-11) B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11) BD(8-11)
                const __m512i rhs_mat_2367ABEF_1_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)136); //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11) BA(8-11) BB(8-11) BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11) BF(8-11)

                const __m512i rhs_mat_014589CD_2_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)136); //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19) B8(16-19) B9(16-19) B8(16-19) B9(16-19) BC(16-19) BD(16-19) BC(16-19) BD(16-19)
                const __m512i rhs_mat_2367ABEF_2_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)136); //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19) BA(16-19) BB(16-19) BA(16-19) BB(16-19) BE(16-19) BF(16-19) BE(16-19) BF(16-19)

                const __m512i rhs_mat_014589CD_3_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)136); //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27) B8(24-27) B9(24-27) B8(24-27) B9(24-27) BC(24-27) BD(24-27) BC(24-27) BD(24-27)
                const __m512i rhs_mat_2367ABEF_3_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)136); //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27) BA(24-27) BB(24-27) BA(24-27) BB(24-27) BE(24-27) BF(24-27) BE(24-27) BF(24-27)

                // Shuffle pattern two - right side input

                const __m512i rhs_mat_014589CD_0_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)221); //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7) B8(4-7) B9(4-7) B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7) BD(4-7)
                const __m512i rhs_mat_2367ABEF_0_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)221); //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7) BA(4-7) BB(4-7) BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7) BF(4-7)

                const __m512i rhs_mat_014589CD_1_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)221); //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15) B8(12-15) B9(12-15) B8(12-15) B9(12-15) BC(12-15) BD(12-15) BC(12-15) BD(12-15)
                const __m512i rhs_mat_2367ABEF_1_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)221); //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15) BA(12-15) BB(12-15) BA(12-15) BB(12-15) BE(12-15) BF(12-15) BE(12-15) BF(12-15)

                const __m512i rhs_mat_014589CD_2_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)221); //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23) B8(20-23) B9(20-23) B8(20-23) B9(20-23) BC(20-23) BD(20-23) BC(20-23) BD(20-23)
                const __m512i rhs_mat_2367ABEF_2_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)221); //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23) BA(20-23) BB(20-23) BA(20-23) BB(20-23) BE(20-23) BF(20-23) BE(20-23) BF(20-23)

                const __m512i rhs_mat_014589CD_3_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)221); //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31) B8(28-31) B9(28-31) B8(28-31) B9(28-31) BC(28-31) BD(28-31) BC(28-31) BD(28-31)
                const __m512i rhs_mat_2367ABEF_3_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)221); //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31) BA(28-31) BB(28-31) BA(28-31) BB(28-31) BE(28-31) BF(28-31) BE(28-31) BF(28-31)


                // Scale values - Load the weight scale values of two block_tx8
                __m512 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    //TODO: simd-ify
                    col_scale_f32 = _mm512_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[0]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[0]));
                }

                // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                __m256i lhs_mat_ymm_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
                __m256i lhs_mat_ymm_01_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 0);
                __m256i lhs_mat_ymm_23_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 17);
                __m256i lhs_mat_ymm_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
                __m256i lhs_mat_ymm_01_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 0);
                __m256i lhs_mat_ymm_23_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 17);
                __m256i lhs_mat_ymm_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
                __m256i lhs_mat_ymm_01_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 0);
                __m256i lhs_mat_ymm_23_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 17);
                __m256i lhs_mat_ymm_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
                __m256i lhs_mat_ymm_01_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 0);
                __m256i lhs_mat_ymm_23_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 17);

                __m512i lhs_mat_01_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_0), lhs_mat_ymm_01_0, 1);
                __m512i lhs_mat_23_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_0), lhs_mat_ymm_23_0, 1);
                __m512i lhs_mat_01_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_1), lhs_mat_ymm_01_1, 1);
                __m512i lhs_mat_23_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_1), lhs_mat_ymm_23_1, 1);
                __m512i lhs_mat_01_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_2), lhs_mat_ymm_01_2, 1);
                __m512i lhs_mat_23_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_2), lhs_mat_ymm_23_2, 1);
                __m512i lhs_mat_01_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_3), lhs_mat_ymm_01_3, 1);
                __m512i lhs_mat_23_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_3), lhs_mat_ymm_23_3, 1);

                // Shuffle pattern one - left side input

                const __m512i lhs_mat_01_0_sp1 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                const __m512i lhs_mat_23_0_sp1 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                const __m512i lhs_mat_01_1_sp1 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                const __m512i lhs_mat_23_1_sp1 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                const __m512i lhs_mat_01_2_sp1 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                const __m512i lhs_mat_23_2_sp1 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                const __m512i lhs_mat_01_3_sp1 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                const __m512i lhs_mat_23_3_sp1 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                // Shuffle pattern two - left side input

                const __m512i lhs_mat_01_0_sp2 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                const __m512i lhs_mat_23_0_sp2 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                const __m512i lhs_mat_01_1_sp2 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                const __m512i lhs_mat_23_1_sp2 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                const __m512i lhs_mat_01_2_sp2 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                const __m512i lhs_mat_23_2_sp2 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                const __m512i lhs_mat_01_3_sp2 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                const __m512i lhs_mat_23_3_sp2 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                // Resembles MMLAs into 2x2 matrices in ARM Version
                const __m512i zero = _mm512_setzero_epi32();
                __m512i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1);
                __m512i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1);
                __m512i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1);
                __m512i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1);
                __m512i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2);
                __m512i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2);
                __m512i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2);
                __m512i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2);

                // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                __m512i iacc_mat_00 = _mm512_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                __m512i iacc_mat_01 = _mm512_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                __m512i iacc_mat_10 = _mm512_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                __m512i iacc_mat_11 = _mm512_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                // Straighten out to make 4 row vectors
                __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                const __m128i row_scale_f16 = _mm_shuffle_epi32(_mm_maskload_epi32((int const*)(a_ptr[b].d), loadMask), 68);
                const __m512 row_scale_f32 = GGML_F32Cx16_REPEAT_LOAD(row_scale_f16);

                // Multiply with appropriate scales and accumulate
                acc_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),   acc_rows[0]);
                acc_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),  acc_rows[1]);
                acc_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                acc_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);
            }

            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }
    if (anc != nc) {
        xstart = anc/8;
        y = 0;
    }
#endif // __AVX512BW__ && __AVX512DQ__

    // Take group of four block_q8_0x4 structures at each pass of the loop and perform dot product operation

    for (; y < anr / 4; y += 4) {
        const block_q8_0x4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_tx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_tx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the eight blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

                // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                // 4-bit -> 8-bit - Sign is maintained
                const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_0, m4b)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_0, m4b)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7)

                const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_1, m4b)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15)
                const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_1, m4b)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15)

                const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23)

                const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31)

                // Shuffle pattern one - right side input
                const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_0, 136);  //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_0, 136);  //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

                const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_1, 136);  //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
                const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_1, 136);  //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

                const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_2, 136);  //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_2, 136);  //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

                const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_3, 136);  //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_3, 136);  //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

                // Shuffle pattern two - right side input

                const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_0, 221);  //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_0, 221);  //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

                const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_1, 221);  //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_1, 221);  //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

                const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_2, 221);  //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_2, 221);  //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

                const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_3, 221);  //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_3, 221);  //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

                // Scale values - Load the wight scale values of block_tx8
                __m256 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    col_scale_f32 = _mm256_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[0]));
                }

                // Process LHS in groups of four
                for (int rp = 0; rp < 4; rp++) {
                    // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
                    __m256i lhs_mat_01_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
                    __m256i lhs_mat_23_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
                    __m256i lhs_mat_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
                    __m256i lhs_mat_01_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
                    __m256i lhs_mat_23_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
                    __m256i lhs_mat_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
                    __m256i lhs_mat_01_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
                    __m256i lhs_mat_23_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
                    __m256i lhs_mat_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
                    __m256i lhs_mat_01_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
                    __m256i lhs_mat_23_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(lhs_mat_01_0, 160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                    const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(lhs_mat_23_0, 160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                    const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(lhs_mat_01_1, 160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                    const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(lhs_mat_23_1, 160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                    const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(lhs_mat_01_2, 160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                    const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(lhs_mat_23_2, 160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                    const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(lhs_mat_01_3, 160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                    const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(lhs_mat_23_3, 160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                    // Shuffle pattern two - left side input
                    const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(lhs_mat_01_0, 245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                    const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(lhs_mat_23_0, 245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                    const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(lhs_mat_01_1, 245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                    const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(lhs_mat_23_1, 245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                    const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(lhs_mat_01_2, 245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                    const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(lhs_mat_23_2, 245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                    const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(lhs_mat_01_3, 245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                    const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(lhs_mat_23_3, 245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    // Resembles MMLAs into 2x2 matrices in ARM Version
                    const __m256i zero = _mm256_setzero_si256();
                    __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
                    __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
                    __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
                    __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
                    __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
                    __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
                    __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
                    __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m256i iacc_mat_00 = _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                    __m256i iacc_mat_01 = _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                    __m256i iacc_mat_10 = _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                    __m256i iacc_mat_11 = _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

                    // Straighten out to make 4 row vectors
                    __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                    __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                    __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                    __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                    // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                    const __m256 row_scale_f32 = GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                    // Multiply with appropriate scales and accumulate
                    acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                    acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                    acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                    acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  255)), acc_rows[rp * 4 + 3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }

    // Take a block_q8_0x4 structures at each pass of the loop and perform dot product operation
    for (; y < nr / 4; y ++) {
        const block_q8_0x4 * a_ptr = a_ptr_start + (y * nb);

        // Load the eight blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
        for (int64_t x = xstart; x < nc / 8; x++) {
            const block_tx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the eight block_q8_0 quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

                // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                // 4-bit -> 8-bit - Sign is maintained
                const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_0, m4b));  //B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_0, m4b));  //B2(0-7) B3(0-7) B6(0-7) B7(0-7)

                const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_1, m4b));  //B0(8-15) B1(8-15) B4(8-15) B5(8-15)
                const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_1, m4b));  //B2(8-15) B3(8-15) B6(8-15) B7(8-15)

                const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b));  //B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b));  //B2(16-23) B3(16-23) B6(16-23) B7(16-23)

                const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b));  //B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b));  //B2(24-31) B3(24-31) B6(24-31) B7(24-31)

                // Shuffle pattern one - right side input
                const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_0, 136);  //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_0, 136);  //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

                const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_1, 136);  //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
                const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_1, 136);  //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

                const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_2, 136);  //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_2, 136);  //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

                const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_3, 136);  //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_3, 136);  //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

                // Shuffle pattern two - right side input

                const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_0, 221);  //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_0, 221);  //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

                const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_1, 221);  //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_1, 221);  //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

                const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_2, 221);  //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_2, 221);  //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

                const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_3, 221);  //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_3, 221);  //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

                // Scale values - Load the wight scale values of block_tx8
                __m256 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    col_scale_f32 = _mm256_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[0]));
                }

                // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                __m256i lhs_mat_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
                __m256i lhs_mat_01_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
                __m256i lhs_mat_23_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
                __m256i lhs_mat_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
                __m256i lhs_mat_01_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
                __m256i lhs_mat_23_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
                __m256i lhs_mat_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
                __m256i lhs_mat_01_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
                __m256i lhs_mat_23_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
                __m256i lhs_mat_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
                __m256i lhs_mat_01_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
                __m256i lhs_mat_23_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

                // Shuffle pattern one - left side input

                const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(lhs_mat_01_0, 160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(lhs_mat_23_0, 160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(lhs_mat_01_1, 160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(lhs_mat_23_1, 160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(lhs_mat_01_2, 160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(lhs_mat_23_2, 160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(lhs_mat_01_3, 160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(lhs_mat_23_3, 160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                // Shuffle pattern two - left side input

                const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(lhs_mat_01_0, 245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(lhs_mat_23_0, 245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(lhs_mat_01_1, 245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(lhs_mat_23_1, 245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(lhs_mat_01_2, 245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(lhs_mat_23_2, 245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(lhs_mat_01_3, 245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(lhs_mat_23_3, 245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                // Resembles MMLAs into 2x2 matrices in ARM Version
                const __m256i zero = _mm256_setzero_si256();
                __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
                __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
                __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
                __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
                __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
                __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
                __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
                __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

                // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                __m256i iacc_mat_00 = _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                __m256i iacc_mat_01 = _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                __m256i iacc_mat_10 = _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                __m256i iacc_mat_11 = _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                // Straighten out to make 4 row vectors
                __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                const __m256 row_scale_f32 = GGML_F32Cx8_REPEAT_LOAD(a_ptr[b].d, loadMask);

                // Multiply with appropriate scales and accumulate
                acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);
            }

            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }
}

#endif // defined(__AVX2__) || defined(__AVX512F__)

void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        // Lookup table to convert signed nibbles to signed bytes
        __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

        gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc, signextendlut);

        return;
    }
#endif

    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__AVX2__)
    // Lookup table to convert signed nibbles to signed bytes
    __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    // Shuffle masks to rearrange delta and scale values to multiply with appropriate scales
    __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    __m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // Mask to extract nibbles from bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    int64_t b_nb = n / QK_K;

    const block_q4_Kx8 * b_ptr_start = (const block_q4_Kx8 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;

    // Process Q8_K blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_K format
        const block_q8_K * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_row = _mm256_setzero_ps();
            __m256 acc_min_rows = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {

                // Load and convert to FP32 scale from block_q8_K
                const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

                // Load the scale values for the 8 blocks interleaved in block_q4_Kx8
                // col_scale_f32 rearranged so as to multiply with appropriate quants
                const __m256 col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                __m256i iacc_b = _mm256_setzero_si256();
                __m256i iacc_min_b = _mm256_setzero_si256();

                const __m256i q8sums = _mm256_loadu_si256((const __m256i * )(a_ptr[b].bsums));
                __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
                q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

                // Processes two sub blocks from each Q4_K in each iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // 4-bit -> 8-bit
                    // Values of the first sub block of eight block_q4_K structures for the sb loop
                    const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
                    const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
                    const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
                    const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
                    const __m256i rhs_vec_0123_02 = _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
                    const __m256i rhs_vec_4567_02 = _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
                    const __m256i rhs_vec_0123_03 = _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
                    const __m256i rhs_vec_4567_03 = _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

                    // Values of the second sub block of eight block_q4_K structures when sb = 1
                    const __m256i rhs_vec_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                    const __m256i rhs_vec_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                    const __m256i rhs_vec_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                    const __m256i rhs_vec_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                    const __m256i rhs_vec_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                    const __m256i rhs_vec_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                    const __m256i rhs_vec_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                    const __m256i rhs_vec_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q8_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                    __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                    // Scales of second sub block in the sb loop
                    __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                    __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    // Load the two sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same across 256 bit vector
                    __m256i lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
                    __m256i lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 64)));
                    __m256i lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 64)));
                    __m256i lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 64)));

                    lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                    lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                    lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                    lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                    // Dot product done within 32 bit lanes and accumulated in the same vector
                    // First done for first sub block and then for second sub block in each sb
                    // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                    // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                    // ...........................................................................
                    // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)


                    __m256i iacc_0 = _mm256_setzero_si256();
                    __m256i iacc_1 = _mm256_setzero_si256();

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00 ,_mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177) ,rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01 ,_mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177) ,rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00, 255)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_02 ,_mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177) ,rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_03 ,_mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177) ,rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01, 255)));

                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10 ,_mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177) ,rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11 ,_mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177) ,rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10, 255)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_12 ,_mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177) ,rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_13 ,_mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177) ,rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11, 255)));

                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                    // Accumulate the iacc value for one sb
                    __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

                    // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K across the vector
                    // Multiply-Add with corresponding mins of Q4_Kx8 with bsums
                    __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
                    __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                    q8s = _mm256_bsrli_epi128(q8s, 4);

                    // Accumulate for the complete block
                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                // Multiply-Add with scale values for the complete super block
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);

            }

            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
        }
    }

#else
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemv_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__)
    __m256i signextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_iq4nl));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

    gemv_q4_b32_8x8_q8_0_lut_avx<block_iq4_nlx8>(n, s, bs, vx, vy, nr, nc, signextendlut);

    return;
#endif

    ggml_gemv_iq4_nl_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__)
    __m256i signextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_mxfp4));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

    gemv_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>(n, s, bs, vx, vy, nr, nc, signextendlut);

    return;
#endif

    ggml_gemv_mxfp4_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__AVX2__)
    // Lookup table to convert signed nibbles to signed bytes
    __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    // Shuffle masks to rearrange delta values to multiply with appropriate scales
    __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    const __m256i m3b = _mm256_set1_epi8(3);
    const __m128i m4b_sse = _mm_set1_epi8(0xF);

    //Mask to get appropriate scales
    __m128i scalemask1 = _mm_set_epi8(14,14,6,6,12,12,4,4,10,10,2,2,8,8,0,0);
    __m128i scalemask2 = _mm_set_epi8(15,15,7,7,13,13,5,5,11,11,3,3,9,9,1,1);

    int64_t b_nb = n / QK_K;

    const block_q2_Kx8 * b_ptr_start = (const block_q2_Kx8 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;

    // Process Q8_K blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_K format
        const block_q8_K * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight interleaved block_q2_K structures at each pass of the loop and perform dot product operation
        for(int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_q2_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_row = _mm256_setzero_ps();
            __m256 acc_min_rows = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {

                // Load and convert to FP32 delta from block_q8_K
                const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

                // Load the delta values for the 8 blocks interleaved in block_q2_Kx8
                // col_scale_f32 rearranged so as to multiply with appropriate quants
                const __m256 col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                __m256i iacc_b = _mm256_setzero_si256();
                __m256i iacc_min_b = _mm256_setzero_si256();

                // Processes eight sub blocks from each Q2_K in each iteration
                for(int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_K for eight sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // 2-bit -> 8-bit
                    // Values of the 0th,2nd,4th,6th sub blocks of eight block_q2_K structures for the sb loop
                    const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m3b); //B00(0-7) B01(0-7) B02(0-7) B03(0-7)
                    const __m256i rhs_vec_0123_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 2), m3b); //B20(0-7) B21(0-7) B22(0-7) B23(0-7)
                    const __m256i rhs_vec_0123_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m3b); //B40(0-7) B41(0-7) B42(0-7) B43(0-7)
                    const __m256i rhs_vec_0123_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 6), m3b); //B60(0-7) B61(0-7) B62(0-7) B63(0-7)

                    const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m3b); //B04(0-7) B05(0-7) B06(0-7) B07(0-7)
                    const __m256i rhs_vec_4567_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 2), m3b); //B24(0-7) B25(0-7) B26(0-7) B27(0-7)
                    const __m256i rhs_vec_4567_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m3b); //B44(0-7) B45(0-7) B46(0-7) B47(0-7)
                    const __m256i rhs_vec_4567_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 6), m3b); //B64(0-7) B65(0-7) B66(0-7) B67(0-7)

                    const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m3b); //B00(8-15) B01(8-15) B02(8-15) B03(8-15)
                    const __m256i rhs_vec_0123_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 2), m3b); //B20(8-15) B21(8-15) B22(8-15) B23(8-15)
                    const __m256i rhs_vec_0123_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m3b); //B40(8-15) B41(8-15) B42(8-15) B43(8-15)
                    const __m256i rhs_vec_0123_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 6), m3b); //B60(8-15) B61(8-15) B62(8-15) B63(8-15)

                    const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m3b); //B04(8-15) B05(8-15) B06(8-15) B07(8-15)
                    const __m256i rhs_vec_4567_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 2), m3b); //B24(8-15) B25(8-15) B26(8-15) B27(8-15)
                    const __m256i rhs_vec_4567_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m3b); //B44(8-15) B45(8-15) B46(8-15) B47(8-15)
                    const __m256i rhs_vec_4567_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 6), m3b); //B64(8-15) B65(8-15) B66(8-15) B67(8-15)

                    // Values of the 1st,3rd,5th,7th sub blocks of eight block_q2_K structures for the sb loop
                    const __m256i rhs_vec_0123_10 = _mm256_and_si256(rhs_raw_vec_0123_2, m3b); //B10(0-7) B11(0-7) B12(0-7) B13(0-7)
                    const __m256i rhs_vec_0123_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 2), m3b); //B30(0-7) B31(0-7) B32(0-7) B33(0-7)
                    const __m256i rhs_vec_0123_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m3b); //B50(0-7) B51(0-7) B52(0-7) B53(0-7)
                    const __m256i rhs_vec_0123_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 6), m3b); //B70(0-7) B71(0-7) B72(0-7) B73(0-7)

                    const __m256i rhs_vec_4567_10 = _mm256_and_si256(rhs_raw_vec_4567_2, m3b); //B14(0-7) B15(0-7) B16(0-7) B17(0-7)
                    const __m256i rhs_vec_4567_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 2), m3b); //B34(0-7) B35(0-7) B36(0-7) B37(0-7)
                    const __m256i rhs_vec_4567_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m3b); //B54(0-7) B55(0-7) B56(0-7) B57(0-7)
                    const __m256i rhs_vec_4567_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 6), m3b); //B74(0-7) B75(0-7) B76(0-7) B77(0-7)

                    const __m256i rhs_vec_0123_11 = _mm256_and_si256(rhs_raw_vec_0123_3, m3b); //B10(8-15) B11(8-15) B12(8-15) B13(8-15)
                    const __m256i rhs_vec_0123_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 2), m3b); //B30(8-15) B31(8-15) B32(8-15) B33(8-15)
                    const __m256i rhs_vec_0123_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m3b); //B50(8-15) B51(8-15) B52(8-15) B53(8-15)
                    const __m256i rhs_vec_0123_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 6), m3b); //B70(8-15) B71(8-15) B72(8-15) B73(8-15)

                    const __m256i rhs_vec_4567_11 = _mm256_and_si256(rhs_raw_vec_4567_3, m3b); //B14(8-15) B15(8-15) B16(8-15) B17(8-15)
                    const __m256i rhs_vec_4567_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 2), m3b); //B34(8-15) B35(8-15) B36(8-15) B37(8-15)
                    const __m256i rhs_vec_4567_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m3b); //B54(8-15) B55(8-15) B56(8-15) B57(8-15)
                    const __m256i rhs_vec_4567_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 6), m3b); //B74(8-15) B75(8-15) B76(8-15) B77(8-15)

                    //Scales and Mins of corresponding sub blocks from different Q2_K structures are stored together
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    const __m128i mins_and_scales_01 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + sb * 64));
                    const __m128i mins_and_scales_23 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 48 + sb * 64));

                    // Extract scales which is lower half from mins_and_scales
                    const __m128i scales_01 = _mm_and_si128(mins_and_scales_01, m4b_sse);
                    const __m128i scales_23 = _mm_and_si128(mins_and_scales_23, m4b_sse);
                    const __m128i scales_45 = _mm_and_si128(mins_and_scales_45, m4b_sse);
                    const __m128i scales_67 = _mm_and_si128(mins_and_scales_67, m4b_sse);

                    // Extract mins which is upper half from mins_and_scales
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_01, 4), m4b_sse));
                    const __m256i mins_23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_23, 4), m4b_sse));
                    const __m256i mins_45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_45, 4), m4b_sse));
                    const __m256i mins_67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_67, 4), m4b_sse));

                    // Scales of sub blocks in the sb loop
                    // Scales of the 0th sub block from each super block
                    __m128i scales_rearrange_0 = _mm_shuffle_epi8(scales_01, scalemask1);
                    __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                    // Scales of the 1st sub block from each super block
                    __m128i scales_rearrange_1 = _mm_shuffle_epi8(scales_01, scalemask2);
                    __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                    // Scales of the 2nd sub block from each super block
                    __m128i scales_rearrange_2 = _mm_shuffle_epi8(scales_23, scalemask1);
                    __m256i scales_2 = _mm256_cvtepu8_epi16(scales_rearrange_2);

                    // Scales of the 3rd sub block from each super block
                    __m128i scales_rearrange_3 = _mm_shuffle_epi8(scales_23, scalemask2);
                    __m256i scales_3 = _mm256_cvtepu8_epi16(scales_rearrange_3);

                    // Scales of the 4th sub block from each super block
                    __m128i scales_rearrange_4 = _mm_shuffle_epi8(scales_45, scalemask1);
                    __m256i scales_4 = _mm256_cvtepu8_epi16(scales_rearrange_4);

                    // Scales of the 5th sub block from each super block
                    __m128i scales_rearrange_5 = _mm_shuffle_epi8(scales_45, scalemask2);
                    __m256i scales_5 = _mm256_cvtepu8_epi16(scales_rearrange_5);

                    // Scales of the 6th sub block from each super block
                    __m128i scales_rearrange_6 = _mm_shuffle_epi8(scales_67, scalemask1);
                    __m256i scales_6 = _mm256_cvtepu8_epi16(scales_rearrange_6);

                    // Scales of the 7th sub block from each super block
                    __m128i scales_rearrange_7 = _mm_shuffle_epi8(scales_67, scalemask2);
                    __m256i scales_7 = _mm256_cvtepu8_epi16(scales_rearrange_7);

                    // Load the sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same across 256 bit vector
                    __m256i lhs_vec_0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 128)));
                    __m256i lhs_vec_1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 128)));
                    __m256i lhs_vec_2 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 128)));
                    __m256i lhs_vec_3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 128)));
                    __m256i lhs_vec_4 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 64 + sb * 128)));
                    __m256i lhs_vec_5 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 80 + sb * 128)));
                    __m256i lhs_vec_6 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 96 + sb * 128)));
                    __m256i lhs_vec_7 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 112 + sb * 128)));

                    lhs_vec_0 = _mm256_permute2f128_si256(lhs_vec_0, lhs_vec_0, 0);
                    lhs_vec_1 = _mm256_permute2f128_si256(lhs_vec_1, lhs_vec_1, 0);
                    lhs_vec_2 = _mm256_permute2f128_si256(lhs_vec_2, lhs_vec_2, 0);
                    lhs_vec_3 = _mm256_permute2f128_si256(lhs_vec_3, lhs_vec_3, 0);
                    lhs_vec_4 = _mm256_permute2f128_si256(lhs_vec_4, lhs_vec_4, 0);
                    lhs_vec_5 = _mm256_permute2f128_si256(lhs_vec_5, lhs_vec_5, 0);
                    lhs_vec_6 = _mm256_permute2f128_si256(lhs_vec_6, lhs_vec_6, 0);
                    lhs_vec_7 = _mm256_permute2f128_si256(lhs_vec_7, lhs_vec_7, 0);

                    __m256i iacc_0 = _mm256_setzero_si256();
                    __m256i iacc_1 = _mm256_setzero_si256();
                    __m256i iacc_2 = _mm256_setzero_si256();
                    __m256i iacc_3 = _mm256_setzero_si256();
                    __m256i iacc_4 = _mm256_setzero_si256();
                    __m256i iacc_5 = _mm256_setzero_si256();
                    __m256i iacc_6 = _mm256_setzero_si256();
                    __m256i iacc_7 = _mm256_setzero_si256();

                    // Dot product done within 32 bit lanes and accumulated in the same vector
                    // First done for 0th sub block and then for seven (1st - 7th) other sub blocks processed for each sb (sb < QK_K/128 loop)                    // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                    // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                    // B0(8-11) B4(8-11) B1(8-11) B5(8-11) B2(8-11) B6(8-11) B3(8-11) B7(8-11) with A0(8-11)
                    // B0(12-15) B4(12-15) B1(12-15) B5(12-15) B2(12-15) B6(12-15) B3(12-15) B7(12-15) with A0(12-15)

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00 ,_mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177) ,rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_0, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01 ,_mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177) ,rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_0, 255)));

                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10 ,_mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177) ,rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_1, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11 ,_mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177) ,rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_1, 255)));

                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_20 ,_mm256_shuffle_epi32(rhs_vec_4567_20, 177), 170), _mm256_shuffle_epi32(lhs_vec_2, 0)));
                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_20, 177) ,rhs_vec_4567_20, 170), _mm256_shuffle_epi32(lhs_vec_2, 85)));

                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_21 ,_mm256_shuffle_epi32(rhs_vec_4567_21, 177), 170), _mm256_shuffle_epi32(lhs_vec_2, 170)));
                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_21, 177) ,rhs_vec_4567_21, 170), _mm256_shuffle_epi32(lhs_vec_2, 255)));

                    iacc_2 = _mm256_madd_epi16(iacc_2, scales_2);

                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_30 ,_mm256_shuffle_epi32(rhs_vec_4567_30, 177), 170), _mm256_shuffle_epi32(lhs_vec_3, 0)));
                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_30, 177) ,rhs_vec_4567_30, 170), _mm256_shuffle_epi32(lhs_vec_3, 85)));

                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_31 ,_mm256_shuffle_epi32(rhs_vec_4567_31, 177), 170), _mm256_shuffle_epi32(lhs_vec_3, 170)));
                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_31, 177) ,rhs_vec_4567_31, 170), _mm256_shuffle_epi32(lhs_vec_3, 255)));

                    iacc_3 = _mm256_madd_epi16(iacc_3, scales_3);

                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_40 ,_mm256_shuffle_epi32(rhs_vec_4567_40, 177), 170), _mm256_shuffle_epi32(lhs_vec_4, 0)));
                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_40, 177) ,rhs_vec_4567_40, 170), _mm256_shuffle_epi32(lhs_vec_4, 85)));

                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_41 ,_mm256_shuffle_epi32(rhs_vec_4567_41, 177), 170), _mm256_shuffle_epi32(lhs_vec_4, 170)));
                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_41, 177) ,rhs_vec_4567_41, 170), _mm256_shuffle_epi32(lhs_vec_4, 255)));

                    iacc_4 = _mm256_madd_epi16(iacc_4, scales_4);

                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_50 ,_mm256_shuffle_epi32(rhs_vec_4567_50, 177), 170), _mm256_shuffle_epi32(lhs_vec_5, 0)));
                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_50, 177) ,rhs_vec_4567_50, 170), _mm256_shuffle_epi32(lhs_vec_5, 85)));

                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_51 ,_mm256_shuffle_epi32(rhs_vec_4567_51, 177), 170), _mm256_shuffle_epi32(lhs_vec_5, 170)));
                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_51, 177) ,rhs_vec_4567_51, 170), _mm256_shuffle_epi32(lhs_vec_5, 255)));

                    iacc_5 = _mm256_madd_epi16(iacc_5, scales_5);

                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_60 ,_mm256_shuffle_epi32(rhs_vec_4567_60, 177), 170), _mm256_shuffle_epi32(lhs_vec_6, 0)));
                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_60, 177) ,rhs_vec_4567_60, 170), _mm256_shuffle_epi32(lhs_vec_6, 85)));

                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_61 ,_mm256_shuffle_epi32(rhs_vec_4567_61, 177), 170), _mm256_shuffle_epi32(lhs_vec_6, 170)));
                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_61, 177) ,rhs_vec_4567_61, 170), _mm256_shuffle_epi32(lhs_vec_6, 255)));

                    iacc_6 = _mm256_madd_epi16(iacc_6, scales_6);

                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_70 ,_mm256_shuffle_epi32(rhs_vec_4567_70, 177), 170), _mm256_shuffle_epi32(lhs_vec_7, 0)));
                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_70, 177) ,rhs_vec_4567_70, 170), _mm256_shuffle_epi32(lhs_vec_7, 85)));

                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_71 ,_mm256_shuffle_epi32(rhs_vec_4567_71, 177), 170), _mm256_shuffle_epi32(lhs_vec_7, 170)));
                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_71, 177) ,rhs_vec_4567_71, 170), _mm256_shuffle_epi32(lhs_vec_7, 255)));

                    iacc_7 = _mm256_madd_epi16(iacc_7, scales_7);

                    // Accumulate the iacc value for one sb
                    __m256i iacc_sb = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_0, iacc_1), _mm256_add_epi32(iacc_2, iacc_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_4, iacc_5), _mm256_add_epi32(iacc_6, iacc_7)));

                    __m128i q8sums = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + sb * 8));
                    __m256i q8s = _mm256_castsi128_si256(q8sums);
                    q8s= _mm256_permute2f128_si256(q8s, q8s, 0);

                    // Broadcast the bsums of the two corresponding subblocks of q8_k
                    // Multiply-Add with corresponding mins of Q2_Kx8 with bsums
                    __m256i iacc_min_sb_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(q8s, 0), mins_01);
                    __m256i iacc_min_sb_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(q8s, 85), mins_23);
                    __m256i iacc_min_sb_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(q8s, 170), mins_45);
                    __m256i iacc_min_sb_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(q8s, 255), mins_67);

                    __m256i iacc_min_sb = _mm256_add_epi32(_mm256_add_epi32(iacc_min_sb_01, iacc_min_sb_23), _mm256_add_epi32(iacc_min_sb_45,iacc_min_sb_67));

                    // Accumulate for the complete block
                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                //Multiply-Add with scale values for complete super block
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
            }
            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
        }
    }
#else

    ggml_gemv_q2_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);

#endif
}

void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        // Lookup table to convert signed nibbles to signed bytes
        __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

        gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc, signextendlut);

        return;
    }
#endif // defined(__AVX2__) || defined(__AVX512F__)

    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__AVX2__) || defined(__AVX512F__)
    const block_q4_Kx8 * b_ptr_start = (const block_q4_Kx8 * ) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 * ) vy;
    int64_t b_nb = n / QK_K;
    int64_t y = 0;

    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr % 16;; // Used to align nr with boundary of 16
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
    int anc = nc - nc % 16; // Used to align nc with boundary of 16
    // Mask to mask out nibbles from packed bytes expanded to 512 bit length
    const __m512i m4bexpanded = _mm512_set1_epi8(0x0F);
    //Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q4_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q4_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Scale values - Load the sixteen scale values from two block_q4_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q4_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //4-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)

                    const __m512i rhs_mat_014589CD_02 = _mm512_and_si512(rhs_raw_mat_014589CD_2, m4bexpanded); //B00(16-23) B01(16-23) B04(16-23) B05(16-23) B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
                    const __m512i rhs_mat_2367ABEF_02 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2, m4bexpanded); //B02(16-23) B03(16-23) B06(16-23) B07(16-23) B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
                    const __m512i rhs_mat_014589CD_03 = _mm512_and_si512(rhs_raw_mat_014589CD_3, m4bexpanded); //B00(24-31) B01(24-31) B04(24-31) B05(24-31) B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
                    const __m512i rhs_mat_2367ABEF_03 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3, m4bexpanded); //B02(24-31) B03(24-31) B06(24-31) B07(24-31) B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m4bexpanded); //B10(16-23) B11(16-23) B14(16-23) B15(16-23) B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
                    const __m512i rhs_mat_2367ABEF_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m4bexpanded); //B12(16-23) B13(16-23) B16(16-23) B17(16-23) B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
                    const __m512i rhs_mat_014589CD_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m4bexpanded); //B10(24-31) B11(24-31) B14(24-31) B15(24-31) B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
                    const __m512i rhs_mat_2367ABEF_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m4bexpanded); //B12(24-31) B13(24-31) B16(24-31) B17(24-31) B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

                    // Shuffle pattern one - right side input
                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)
                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
                    const __m512i rhs_mat_014589CD_02_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19) B08(16-19) B09(16-19) B08(16-19) B09(16-19) B0C(16-19) B0D(16-19) B0C(16-19) B0D(16-19)
                    const __m512i rhs_mat_2367ABEF_02_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19) B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19) B0E(16-19) B0F(16-19) B0E(16-19) B0F(16-19)
                    const __m512i rhs_mat_014589CD_03_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27) B08(24-27) B09(24-27) B08(24-27) B09(24-27) B0C(24-27) B0D(24-27) B0C(24-27) B0D(24-27)
                    const __m512i rhs_mat_2367ABEF_03_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27) B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27) B0E(24-27) B0F(24-27) B0E(24-27) B0F(24-27)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)
                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
                    const __m512i rhs_mat_014589CD_12_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19) B18(16-19) B19(16-19) B18(16-19) B19(16-19) B1C(16-19) B1D(16-19) B1C(16-19) B1D(16-19)
                    const __m512i rhs_mat_2367ABEF_12_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19) B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19) B1E(16-19) B1F(16-19) B1E(16-19) B1F(16-19)
                    const __m512i rhs_mat_014589CD_13_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27) B18(24-27) B19(24-27) B18(24-27) B19(24-27) B1C(24-27) B1D(24-27) B1C(24-27) B1D(24-27)
                    const __m512i rhs_mat_2367ABEF_13_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27) B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27) B1E(24-27) B1F(24-27) B1E(24-27) B1F(24-27)

                    // Shuffle pattern two - right side input
                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)
                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)
                    const __m512i rhs_mat_014589CD_02_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23) B08(20-23) B09(20-23) B08(20-23) B09(20-23) B0C(20-23) B0D(20-23) B0C(20-23) B0D(20-23)
                    const __m512i rhs_mat_2367ABEF_02_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23) B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23) B0E(20-23) B0F(20-23) B0E(20-23) B0F(20-23)
                    const __m512i rhs_mat_014589CD_03_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31) B08(28-31) B09(28-31) B08(28-31) B09(28-31) B0C(28-31) B0D(28-31) B0C(28-31) 0BD(28-31)
                    const __m512i rhs_mat_2367ABEF_03_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31) B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31) B0E(28-31) B0F(28-31) B0E(28-31) B0F(28-31)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)
                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)
                    const __m512i rhs_mat_014589CD_12_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23) B18(20-23) B19(20-23) B18(20-23) B19(20-23) B1C(20-23) B1D(20-23) B1C(20-23) B1D(20-23)
                    const __m512i rhs_mat_2367ABEF_12_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23) B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23) B1E(20-23) B1F(20-23) B1E(20-23) B1F(20-23)
                    const __m512i rhs_mat_014589CD_13_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31) B18(28-31) B19(28-31) B18(28-31) B19(28-31) B1C(28-31) B1D(28-31) B1C(28-31) B1D(28-31)
                    const __m512i rhs_mat_2367ABEF_13_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31) B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31) B1E(28-31) B1F(28-31) B1E(28-31) B1F(28-31)

                    uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
                    utmp_00[3] = ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_00 = utmp_00[1] & kmask1;
                    utmp_00[1] = (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
                    utmp_00[2] = uaux_00;
                    utmp_00[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
                    utmp_01[3] = ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_01 = utmp_01[1] & kmask1;
                    utmp_01[1] = (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
                    utmp_01[2] = uaux_01;
                    utmp_01[0] &= kmask1;

                    memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
                    utmp_10[3] = ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_10 = utmp_10[1] & kmask1;
                    utmp_10[1] = (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
                    utmp_10[2] = uaux_10;
                    utmp_10[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
                    utmp_11[3] = ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_11 = utmp_11[1] & kmask1;
                    utmp_11[1] = (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
                    utmp_11[2] = uaux_11;
                    utmp_11[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m256i mins_and_scales_0 = _mm256_set_epi32(utmp_10[3], utmp_10[2], utmp_10[1], utmp_10[0], utmp_00[3], utmp_00[2], utmp_00[1], utmp_00[0]);
                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m256i mins_and_scales_1 = _mm256_set_epi32(utmp_11[3], utmp_11[2], utmp_11[1], utmp_11[0], utmp_01[3], utmp_01[2], utmp_01[1], utmp_01[0]);
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78), _mm256_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                        __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 * sb)));
                        __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                        __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                        __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                        __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                        __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
                        __m256i lhs_mat_ymm_23_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
                        __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
                        __m256i lhs_mat_ymm_23_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
                        __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                        __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                        __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                        __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                        __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
                        __m256i lhs_mat_ymm_23_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
                        __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
                        __m256i lhs_mat_ymm_23_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

                        __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                        __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                        __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                        __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
                        __m512i lhs_mat_01_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
                        __m512i lhs_mat_23_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
                        __m512i lhs_mat_01_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
                        __m512i lhs_mat_23_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

                        __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                        __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                        __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                        __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
                        __m512i lhs_mat_01_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
                        __m512i lhs_mat_23_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
                        __m512i lhs_mat_01_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
                        __m512i lhs_mat_23_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

                        // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                        __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].bsums + 16 * sb)));
                        __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                        lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
                        __m512i lhs_bsums_hsum_0123_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01), lhs_bsums_hsum_ymm_0123_01, 1);

                        // Shuffle pattern one - left side input
                        const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                        const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                        const __m512i lhs_mat_01_02_sp1 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)160); //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                        const __m512i lhs_mat_23_02_sp1 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)160); //A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                        const __m512i lhs_mat_01_03_sp1 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)160); //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                        const __m512i lhs_mat_23_03_sp1 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)160); //A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27)

                        const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                        const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                        const __m512i lhs_mat_01_12_sp1 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)160); //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                        const __m512i lhs_mat_23_12_sp1 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)160); //A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                        const __m512i lhs_mat_01_13_sp1 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                        const __m512i lhs_mat_23_13_sp1 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)160); //A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27)

                        const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                        const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                        const __m512i lhs_mat_01_02_sp2 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                        const __m512i lhs_mat_23_02_sp2 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)245); //A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                        const __m512i lhs_mat_01_03_sp2 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                        const __m512i lhs_mat_23_03_sp2 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)245); //A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31)

                        const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                        const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                        const __m512i lhs_mat_01_12_sp2 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                        const __m512i lhs_mat_23_12_sp2 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)245); //A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                        const __m512i lhs_mat_01_13_sp2 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                        const __m512i lhs_mat_23_13_sp2 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)245); //A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
                        __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
                        __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
                        __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
                        __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
                        __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
                        __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
                        __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

                        __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
                        __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
                        __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
                        __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
                        __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
                        __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
                        __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
                        __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                        iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                        iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                        iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                        iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                        iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                        iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                        iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                        // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                        __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_0, _mm512_shuffle_epi32(iacc_mat_01_0, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_0, (_MM_PERM_ENUM)78), iacc_mat_01_0);
                        __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_0, _mm512_shuffle_epi32(iacc_mat_11_0, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_0, (_MM_PERM_ENUM)78), iacc_mat_11_0);
                        __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_1, _mm512_shuffle_epi32(iacc_mat_01_1, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_1, (_MM_PERM_ENUM)78), iacc_mat_01_1);
                        __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_1, _mm512_shuffle_epi32(iacc_mat_11_1, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(0xCCCC,_mm512_shuffle_epi32(iacc_mat_10_1, (_MM_PERM_ENUM)78), iacc_mat_11_1);

                        __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
                        __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
                        __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
                        __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                        const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                        // Multiply with appropriate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4  + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        __m512i iacc_row_min_0 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)0), mins_01);
                        __m512i iacc_row_min_1 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)85), mins_01);
                        __m512i iacc_row_min_2 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)170), mins_01);
                        __m512i iacc_row_min_3 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)255), mins_01);

                        acc_min_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);
                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

    for (; y < nr / 4; y++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q4_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q4_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Scale values - Load the sixteen scale values from two block_q4_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q4_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //4-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)

                    const __m512i rhs_mat_014589CD_02 = _mm512_and_si512(rhs_raw_mat_014589CD_2, m4bexpanded); //B00(16-23) B01(16-23) B04(16-23) B05(16-23) B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
                    const __m512i rhs_mat_2367ABEF_02 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2, m4bexpanded); //B02(16-23) B03(16-23) B06(16-23) B07(16-23) B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
                    const __m512i rhs_mat_014589CD_03 = _mm512_and_si512(rhs_raw_mat_014589CD_3, m4bexpanded); //B00(24-31) B01(24-31) B04(24-31) B05(24-31) B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
                    const __m512i rhs_mat_2367ABEF_03 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3, m4bexpanded); //B02(24-31) B03(24-31) B06(24-31) B07(24-31) B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m4bexpanded); //B10(16-23) B11(16-23) B14(16-23) B15(16-23) B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
                    const __m512i rhs_mat_2367ABEF_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m4bexpanded); //B12(16-23) B13(16-23) B16(16-23) B17(16-23) B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
                    const __m512i rhs_mat_014589CD_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m4bexpanded); //B10(24-31) B11(24-31) B14(24-31) B15(24-31) B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
                    const __m512i rhs_mat_2367ABEF_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m4bexpanded); //B12(24-31) B13(24-31) B16(24-31) B17(24-31) B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

                    // Shuffle pattern one - right side input
                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)
                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
                    const __m512i rhs_mat_014589CD_02_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19) B08(16-19) B09(16-19) B08(16-19) B09(16-19) B0C(16-19) B0D(16-19) B0C(16-19) B0D(16-19)
                    const __m512i rhs_mat_2367ABEF_02_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19) B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19) B0E(16-19) B0F(16-19) B0E(16-19) B0F(16-19)
                    const __m512i rhs_mat_014589CD_03_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27) B08(24-27) B09(24-27) B08(24-27) B09(24-27) B0C(24-27) B0D(24-27) B0C(24-27) B0D(24-27)
                    const __m512i rhs_mat_2367ABEF_03_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27) B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27) B0E(24-27) B0F(24-27) B0E(24-27) B0F(24-27)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)
                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
                    const __m512i rhs_mat_014589CD_12_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19) B18(16-19) B19(16-19) B18(16-19) B19(16-19) B1C(16-19) B1D(16-19) B1C(16-19) B1D(16-19)
                    const __m512i rhs_mat_2367ABEF_12_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19) B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19) B1E(16-19) B1F(16-19) B1E(16-19) B1F(16-19)
                    const __m512i rhs_mat_014589CD_13_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27) B18(24-27) B19(24-27) B18(24-27) B19(24-27) B1C(24-27) B1D(24-27) B1C(24-27) B1D(24-27)
                    const __m512i rhs_mat_2367ABEF_13_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27) B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27) B1E(24-27) B1F(24-27) B1E(24-27) B1F(24-27)

                    // Shuffle pattern two - right side input
                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)
                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)
                    const __m512i rhs_mat_014589CD_02_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23) B08(20-23) B09(20-23) B08(20-23) B09(20-23) B0C(20-23) B0D(20-23) B0C(20-23) B0D(20-23)
                    const __m512i rhs_mat_2367ABEF_02_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23) B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23) B0E(20-23) B0F(20-23) B0E(20-23) B0F(20-23)
                    const __m512i rhs_mat_014589CD_03_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31) B08(28-31) B09(28-31) B08(28-31) B09(28-31) B0C(28-31) B0D(28-31) B0C(28-31) 0BD(28-31)
                    const __m512i rhs_mat_2367ABEF_03_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31) B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31) B0E(28-31) B0F(28-31) B0E(28-31) B0F(28-31)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)
                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)
                    const __m512i rhs_mat_014589CD_12_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23) B18(20-23) B19(20-23) B18(20-23) B19(20-23) B1C(20-23) B1D(20-23) B1C(20-23) B1D(20-23)
                    const __m512i rhs_mat_2367ABEF_12_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23) B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23) B1E(20-23) B1F(20-23) B1E(20-23) B1F(20-23)
                    const __m512i rhs_mat_014589CD_13_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31) B18(28-31) B19(28-31) B18(28-31) B19(28-31) B1C(28-31) B1D(28-31) B1C(28-31) B1D(28-31)
                    const __m512i rhs_mat_2367ABEF_13_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31) B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31) B1E(28-31) B1F(28-31) B1E(28-31) B1F(28-31)

                    uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
                    utmp_00[3] = ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_00 = utmp_00[1] & kmask1;
                    utmp_00[1] = (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
                    utmp_00[2] = uaux_00;
                    utmp_00[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
                    utmp_01[3] = ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_01 = utmp_01[1] & kmask1;
                    utmp_01[1] = (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
                    utmp_01[2] = uaux_01;
                    utmp_01[0] &= kmask1;

                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
                    utmp_10[3] = ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_10 = utmp_10[1] & kmask1;
                    utmp_10[1] = (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
                    utmp_10[2] = uaux_10;
                    utmp_10[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
                    utmp_11[3] = ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_11 = utmp_11[1] & kmask1;
                    utmp_11[1] = (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
                    utmp_11[2] = uaux_11;
                    utmp_11[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m256i mins_and_scales_0 = _mm256_set_epi32(utmp_10[3], utmp_10[2], utmp_10[1], utmp_10[0], utmp_00[3], utmp_00[2], utmp_00[1], utmp_00[0]);
                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m256i mins_and_scales_1 = _mm256_set_epi32(utmp_11[3], utmp_11[2], utmp_11[1], utmp_11[0], utmp_01[3], utmp_01[2], utmp_01[1], utmp_01[0]);
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78), _mm256_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 * sb)));
                    __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                    __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                    __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                    __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                    __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
                    __m256i lhs_mat_ymm_23_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
                    __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
                    __m256i lhs_mat_ymm_23_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
                    __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                    __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                    __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                    __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                    __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
                    __m256i lhs_mat_ymm_23_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
                    __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
                    __m256i lhs_mat_ymm_23_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

                    //Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into a 512 bit vector
                    __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                    __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                    __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                    __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
                    __m512i lhs_mat_01_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
                    __m512i lhs_mat_23_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
                    __m512i lhs_mat_01_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
                    __m512i lhs_mat_23_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

                    __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                    __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                    __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                    __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
                    __m512i lhs_mat_01_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
                    __m512i lhs_mat_23_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
                    __m512i lhs_mat_01_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
                    __m512i lhs_mat_23_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

                    // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                    __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].bsums + 16 * sb)));
                    __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                    lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
                    __m512i lhs_bsums_hsum_0123_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01), lhs_bsums_hsum_ymm_0123_01, 1);

                    // Shuffle pattern one - left side input
                    const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                    const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                    const __m512i lhs_mat_01_02_sp1 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)160); //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                    const __m512i lhs_mat_23_02_sp1 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)160); //A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                    const __m512i lhs_mat_01_03_sp1 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)160); //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                    const __m512i lhs_mat_23_03_sp1 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)160); //A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27)

                    const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                    const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                    const __m512i lhs_mat_01_12_sp1 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)160); //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                    const __m512i lhs_mat_23_12_sp1 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)160); //A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                    const __m512i lhs_mat_01_13_sp1 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                    const __m512i lhs_mat_23_13_sp1 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)160); //A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27)

                    const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                    const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                    const __m512i lhs_mat_01_02_sp2 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                    const __m512i lhs_mat_23_02_sp2 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)245); //A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                    const __m512i lhs_mat_01_03_sp2 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                    const __m512i lhs_mat_23_03_sp2 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)245); //A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31)

                    const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                    const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                    const __m512i lhs_mat_01_12_sp2 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                    const __m512i lhs_mat_23_12_sp2 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)245); //A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                    const __m512i lhs_mat_01_13_sp2 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                    const __m512i lhs_mat_23_13_sp2 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)245); //A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
                    __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
                    __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
                    __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
                    __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
                    __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
                    __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
                    __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

                    __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
                    __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
                    __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
                    __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
                    __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
                    __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
                    __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
                    __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                    iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                    iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                    iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                    iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                    iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                    iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                    iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                    // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                    __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_0, _mm512_shuffle_epi32(iacc_mat_01_0, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_0, (_MM_PERM_ENUM)78), iacc_mat_01_0);
                    __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_0, _mm512_shuffle_epi32(iacc_mat_11_0, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_0, (_MM_PERM_ENUM)78), iacc_mat_11_0);
                    __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_1, _mm512_shuffle_epi32(iacc_mat_01_1, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_1, (_MM_PERM_ENUM)78), iacc_mat_01_1);
                    __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_1, _mm512_shuffle_epi32(iacc_mat_11_1, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(0xCCCC,_mm512_shuffle_epi32(iacc_mat_10_1, (_MM_PERM_ENUM)78), iacc_mat_11_1);

                    __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                    const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                    // Multiply with appropriate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m512i iacc_row_min_0 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)0), mins_01);
                    __m512i iacc_row_min_1 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)85), mins_01);
                    __m512i iacc_row_min_2 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)170), mins_01);
                    __m512i iacc_row_min_3 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)255), mins_01);

                    acc_min_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }
            // Store accumulated values
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
    if (anc != nc) {
        xstart = anc/8;
        y = 0;
    }
#endif // __AVX512BW__ && __AVX512DQ__

    // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {

                // Scale values - Load the eight scale values of block_q4_kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q4_kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 4-bit -> 8-bit
                    // First sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    const __m256i rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                    const __m256i rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                    const __m256i rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                    const __m256i rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                    // Second sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    const __m256i rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                    const __m256i rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                    const __m256i rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                    const __m256i rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                    const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

                    const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                    const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                    const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

                    const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                    const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)


                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                    const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

                    const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                    const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                    const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

                    const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                    const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                        __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 * sb)));
                        __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                        __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                        __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 256 * sb)));
                        __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                        __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                        __m256i lhs_mat_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 256 * sb)));
                        __m256i lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                        __m256i lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                        __m256i lhs_mat_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 256 * sb)));
                        __m256i lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                        __m256i lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                        __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 256 * sb)));
                        __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                        __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                        __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 256 * sb)));
                        __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                        __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                        __m256i lhs_mat_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 256 * sb)));
                        __m256i lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                        __m256i lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                        __m256i lhs_mat_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 256 * sb)));
                        __m256i lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                        __m256i lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                        // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                        __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].bsums + 16 * sb)));
                        __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                        lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

                        // Shuffle pattern one - left side input
                        const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                        const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);  //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);  //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                        const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);  //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                        const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);  //A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

                        const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);  //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                        const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

                        const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                        const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);  //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);  //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                        const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);  //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                        const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);  //A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

                        const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                        const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

                        // Shuffle pattern two- left side input
                        const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                        const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                        const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                        const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

                        const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                        const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

                        const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                        const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                        const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                        const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

                        const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                        const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                        __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                        __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                        __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                        __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                        __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                        __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                        __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                        __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                        __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                        __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                        __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                        __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                        __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                        __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                        __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                        iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                        iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                        iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                        iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                        iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                        iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                        iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                        // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                        __m256i iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                        __m256i iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                        __m256i iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                        __m256i iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                        __m256i iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                        __m256i iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                        __m256i iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                        __m256i iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                        __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                        __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                        __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                        __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);//GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                        // Multiply with appropriate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        __m256i iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                        __m256i iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                        __m256i iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                        __m256i iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                        acc_min_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);

                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
    for (; y < nr / 4; y++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {

                // Scale values - Load the eight scale values of block_q4_Kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q4_Kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_k for two sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 4-bit -> 8-bit
                    // First sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    const __m256i rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                    const __m256i rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                    const __m256i rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                    const __m256i rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                    // Second sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    const __m256i rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                    const __m256i rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                    const __m256i rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                    const __m256i rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                    const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

                    const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                    const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                    const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

                    const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                    const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)

                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                    const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

                    const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                    const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                    const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

                    const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                    const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures when sb = 1
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 * sb)));
                    __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                    __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 256 * sb)));
                    __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                    __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    __m256i lhs_mat_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 256 * sb)));
                    __m256i lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                    __m256i lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                    __m256i lhs_mat_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 256 * sb)));
                    __m256i lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                    __m256i lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                    __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 256 * sb)));
                    __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                    __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 256 * sb)));
                    __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                    __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    __m256i lhs_mat_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 256 * sb)));
                    __m256i lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                    __m256i lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                    __m256i lhs_mat_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 256 * sb)));
                    __m256i lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                    __m256i lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                    // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                    __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].bsums + 16 * sb)));
                    __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                    lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                    const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);  //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);  //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                    const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);  //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                    const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);  //A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

                    const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);  //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                    const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

                    const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                    const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);  //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);  //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                    const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);  //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                    const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);  //A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

                    const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                    const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

                    // Shuffle pattern two- left side input
                    const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                    const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                    const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                    const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

                    const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                    const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

                    const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                    const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                    const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                    const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

                    const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                    const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                    __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                    __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                    __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                    // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                    __m256i iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                    __m256i iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                    __m256i iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                    __m256i iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                    __m256i iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                    __m256i iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                    __m256i iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                    __m256i iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                    __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse); //GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                    // Multiply with appropriate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m256i iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                    __m256i iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                    __m256i iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                    __m256i iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

#else
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    ggml_gemm_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        __m256i signextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_iq4nl));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

        gemm_q4_b32_8x8_q8_0_lut_avx<block_iq4_nlx8>(n, s, bs, vx, vy, nr, nc, signextendlut);

        return;
    }
#endif // defined(__AVX2__) || defined(__AVX512F__)

    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        __m256i signextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_mxfp4));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

        gemm_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>(n, s, bs, vx, vy, nr, nc, signextendlut);

        return;
    }
#endif // defined(__AVX2__) || defined(__AVX512F__)

    ggml_gemm_mxfp4_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__AVX2__) || defined(__AVX512F__)
    const block_q2_Kx8 * b_ptr_start = (const block_q2_Kx8 * ) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 * ) vy;
    int64_t b_nb = n / QK_K;
    int64_t y = 0;

    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr % 16; // Used to align nr with boundary of 16

    // Mask to convert 2 bit and 4 bit values into a bytes
    const __m256i m3b = _mm256_set1_epi8(3);
    const __m128i m4b_sse = _mm_set1_epi8(0xF);

    //Mask to get appropriate scales
    __m128i scalesmask1_sse = _mm_set_epi8(14,14,12,12,10,10,8,8,6,6,4,4,2,2,0,0);
    __m128i scalesmask2_sse = _mm_set_epi8(15,15,13,13,11,11,9,9,7,7,5,5,3,3,1,1);

    __m256i scalesmask1 = _mm256_castsi128_si256(scalesmask1_sse);
    scalesmask1 = _mm256_permute2f128_si256(scalesmask1, scalesmask1, 0);
    __m256i scalesmask2 = _mm256_castsi128_si256(scalesmask2_sse);
    scalesmask2 = _mm256_permute2f128_si256(scalesmask2, scalesmask2, 0);

#if defined(__AVX512BW__) && defined(__AVX512DQ__)

    int anc = nc - nc % 16; // Used to align nc with boundary of 16

    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    // Mask to mask out nibbles from packed bytes expanded to 512 bit length
    const __m512i m3bexpanded = _mm512_set1_epi8(3);
    //Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q2_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q2_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }
            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Delta values - Load the sixteen scale values from two block_q2_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q2_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_k for eight sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //2-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0,m3bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0,m3bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1,m3bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1,m3bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)
                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(rhs_raw_mat_014589CD_2,m3bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2,m3bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(rhs_raw_mat_014589CD_3,m3bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3,m3bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_20 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 2), m3bexpanded); //B20(0-7) B21(0-7) B24(0-7) B25(0-7) B28(0-7) B29(0-7) B2C(0-7) B2D(0-7)
                    const __m512i rhs_mat_2367ABEF_20 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 2), m3bexpanded); //B22(0-7) B23(0-7) B26(0-7) B27(0-7) B2A(0-7) B2B(0-7) B2E(0-7) B2F(0-7)

                    const __m512i rhs_mat_014589CD_21 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 2), m3bexpanded); //B20(8-15) B21(8-15) B24(8-15) B25(8-15) B28(8-15) B29(8-15) B2C(8-15) B2D(8-15)
                    const __m512i rhs_mat_2367ABEF_21 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 2), m3bexpanded); //B22(8-15) B23(8-15) B26(8-15) B27(8-15) B2A(8-15) B2B(8-15) B2E(8-15) B2F(8-15)

                    const __m512i rhs_mat_014589CD_30 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 2), m3bexpanded); //B30(0-7) B31(0-7) B34(0-7) B35(0-7) B38(0-7) B39(0-7) B3C(0-7) B3D(0-7)
                    const __m512i rhs_mat_2367ABEF_30 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 2), m3bexpanded); //B32(0-7) B33(0-7) B36(0-7) B37(0-7) B3A(0-7) B3B(0-7) B3E(0-7) B3F(0-7)

                    const __m512i rhs_mat_014589CD_31 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 2), m3bexpanded); //B30(8-15) B31(8-15) B34(8-15) B35(8-15) B38(8-15) B39(8-15) B3C(8-15) B3D(8-15)
                    const __m512i rhs_mat_2367ABEF_31 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 2), m3bexpanded); //B32(8-15) B33(8-15) B36(8-15) B37(8-15) B3A(8-15) B3B(8-15) B3E(8-15) B3F(8-15)

                    const __m512i rhs_mat_014589CD_40 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m3bexpanded); //B40(0-7) B41(0-7) B44(0-7) B45(0-7) B48(0-7) B49(0-7) B4C(0-7) B4D(0-7)
                    const __m512i rhs_mat_2367ABEF_40 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m3bexpanded); //B42(0-7) B43(0-7) B46(0-7) B47(0-7) B4A(0-7) B4B(0-7) B4E(0-7) B4F(0-7)

                    const __m512i rhs_mat_014589CD_41 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m3bexpanded); //B40(8-15) B41(8-15) B44(8-15) B45(8-15) B48(8-15) B49(8-15) B4C(8-15) B4D(8-15)
                    const __m512i rhs_mat_2367ABEF_41 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m3bexpanded); //B42(8-15) B43(8-15) B46(8-15) B47(8-15) B4A(8-15) B4B(8-15) B4E(8-15) B4F(8-15)

                    const __m512i rhs_mat_014589CD_50 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m3bexpanded); //B50(0-7) B51(0-7) B54(0-7) B55(0-7) B58(0-7) B59(0-7) B5C(0-7) B5D(0-7)
                    const __m512i rhs_mat_2367ABEF_50 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m3bexpanded); //B52(0-7) B53(0-7) B56(0-7) B57(0-7) B5A(0-7) B5B(0-7) B5E(0-7) B5F(0-7)

                    const __m512i rhs_mat_014589CD_51 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m3bexpanded); //B50(8-15) B51(8-15) B54(8-15) B55(8-15) B58(8-15) B59(8-15) B5C(8-15) B5D(8-15)
                    const __m512i rhs_mat_2367ABEF_51 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m3bexpanded); //B52(8-15) B53(8-15) B56(8-15) B57(8-15) B5A(8-15) B5B(8-15) B5E(8-15) B5F(8-15)

                    const __m512i rhs_mat_014589CD_60 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 6), m3bexpanded); //B60(0-7) B61(0-7) B64(0-7) B65(0-7) B68(0-7) B69(0-7) B6C(0-7) B6D(0-7)
                    const __m512i rhs_mat_2367ABEF_60 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 6), m3bexpanded); //B62(0-7) B63(0-7) B66(0-7) B67(0-7) B6A(0-7) B6B(0-7) B6E(0-7) B6F(0-7)

                    const __m512i rhs_mat_014589CD_61 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 6), m3bexpanded); //B60(8-15) B61(8-15) B64(8-15) B65(8-15) B68(8-15) B69(8-15) B6C(8-15) B6D(8-15)
                    const __m512i rhs_mat_2367ABEF_61 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 6), m3bexpanded); //B62(8-15) B63(8-15) B66(8-15) B67(8-15) B6A(8-15) B6B(8-15) B6E(8-15) B6F(8-15)

                    const __m512i rhs_mat_014589CD_70 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 6), m3bexpanded); //B70(0-7) B71(0-7) B74(0-7) B75(0-7) B78(0-7) B79(0-7) B7C(0-7) B7D(0-7)
                    const __m512i rhs_mat_2367ABEF_70 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 6), m3bexpanded); //B72(0-7) B73(0-7) B76(0-7) B77(0-7) B7A(0-7) B7B(0-7) B7E(0-7) B7F(0-7)

                    const __m512i rhs_mat_014589CD_71 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 6), m3bexpanded); //B70(8-15) B71(8-15) B74(8-15) B75(8-15) B78(8-15) B79(8-15) B7C(8-15) B7D(8-15)
                    const __m512i rhs_mat_2367ABEF_71 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 6), m3bexpanded); //B72(8-15) B73(8-15) B76(8-15) B77(8-15) B7A(8-15) B7B(8-15) B7E(8-15) B7F(8-15)

                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)

                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)

                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)

                    const __m512i rhs_mat_014589CD_20_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_20, (_MM_PERM_ENUM)136); //B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3) B28(0-3) B29(0-3) B28(0-3) B29(0-3) B2C(0-3) B2D(0-3) B2C(0-3) B2D(0-3)
                    const __m512i rhs_mat_2367ABEF_20_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_20, (_MM_PERM_ENUM)136); //B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3) B2A(0-3) B2B(0-3) B2A(0-3) B2B(0-3) B2E(0-3) B2F(0-3) B2E(0-3) B2F(0-3)

                    const __m512i rhs_mat_014589CD_21_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_21, (_MM_PERM_ENUM)136); //B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11) B28(8-11) B29(8-11) B28(8-11) B29(8-11) B2C(8-11) B2D(8-11) B2C(8-11) B2D(8-11)
                    const __m512i rhs_mat_2367ABEF_21_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_21, (_MM_PERM_ENUM)136); //B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11) B2A(8-11) B2B(8-11) B2A(8-11) B2B(8-11) B2E(8-11) B2F(8-11) B2E(8-11) B2F(8-11)

                    const __m512i rhs_mat_014589CD_30_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_30, (_MM_PERM_ENUM)136); ///B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3) B38(0-3) B39(0-3) B38(0-3) B39(0-3) B3C(0-3) B3D(0-3) B3C(0-3) B3D(0-3)
                    const __m512i rhs_mat_2367ABEF_30_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_30, (_MM_PERM_ENUM)136); //B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3) B3A(0-3) B3B(0-3) B3A(0-3) B3B(0-3) B3E(0-3) B3F(0-3) B3E(0-3) B3F(0-3)

                    const __m512i rhs_mat_014589CD_31_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_31, (_MM_PERM_ENUM)136); //B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11) B38(8-11) B39(8-11) B38(8-11) B39(8-11) B3C(8-11) B3D(8-11) B3C(8-11) B3D(8-11)
                    const __m512i rhs_mat_2367ABEF_31_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_31, (_MM_PERM_ENUM)136); //B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11) B3A(8-11) B3B(8-11) B3A(8-11) B3B(8-11) B3E(8-11) B3F(8-11) B3E(8-11) B3F(8-11)

                    const __m512i rhs_mat_014589CD_40_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_40, (_MM_PERM_ENUM)136); //B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3) B48(0-3) B49(0-3) B48(0-3) B49(0-3) B4C(0-3) B4D(0-3) B4C(0-3) B4D(0-3)
                    const __m512i rhs_mat_2367ABEF_40_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_40, (_MM_PERM_ENUM)136); //B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3) B4A(0-3) B4B(0-3) B4A(0-3) B4B(0-3) B4E(0-3) B4F(0-3) B4E(0-3) B4F(0-3)

                    const __m512i rhs_mat_014589CD_41_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_41, (_MM_PERM_ENUM)136); //B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11) B48(8-11) B49(8-11) B48(8-11) B49(8-11) B4C(8-11) B4D(8-11) B4C(8-11) B4D(8-11)
                    const __m512i rhs_mat_2367ABEF_41_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_41, (_MM_PERM_ENUM)136); //B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11) B4A(8-11) B4B(8-11) B4A(8-11) B4B(8-11) B4E(8-11) B4F(8-11) B4E(8-11) B4F(8-11)

                    const __m512i rhs_mat_014589CD_50_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_50, (_MM_PERM_ENUM)136); //B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3) B58(0-3) B59(0-3) B58(0-3) B59(0-3) B5C(0-3) B5D(0-3) B5C(0-3) B5D(0-3)
                    const __m512i rhs_mat_2367ABEF_50_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_50, (_MM_PERM_ENUM)136); //B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3) B5A(0-3) B5B(0-3) B5A(0-3) B5B(0-3) B5E(0-3) B5F(0-3) B5E(0-3) B5F(0-3)

                    const __m512i rhs_mat_014589CD_51_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_51, (_MM_PERM_ENUM)136); //B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11) B58(8-11) B59(8-11) B58(8-11) B59(8-11) B5C(8-11) B5D(8-11) B5C(8-11) B5D(8-11)
                    const __m512i rhs_mat_2367ABEF_51_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_51, (_MM_PERM_ENUM)136); //B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11) B5A(8-11) B5B(8-11) B5A(8-11) B5B(8-11) B5E(8-11) B5F(8-11) B5E(8-11) B5F(8-11)

                    const __m512i rhs_mat_014589CD_60_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_60, (_MM_PERM_ENUM)136); //B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3) B68(0-3) B69(0-3) B68(0-3) B69(0-3) B6C(0-3) B6D(0-3) B6C(0-3) B6D(0-3)
                    const __m512i rhs_mat_2367ABEF_60_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_60, (_MM_PERM_ENUM)136); //B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3) B6A(0-3) B6B(0-3) B6A(0-3) B6B(0-3) B6E(0-3) B6F(0-3) B6E(0-3) B6F(0-3)

                    const __m512i rhs_mat_014589CD_61_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_61, (_MM_PERM_ENUM)136); //B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11) B68(8-11) B69(8-11) B68(8-11) B69(8-11) B6C(8-11) B6D(8-11) B6C(8-11) B6D(8-11)
                    const __m512i rhs_mat_2367ABEF_61_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_61, (_MM_PERM_ENUM)136); //B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11) B6A(8-11) B6B(8-11) B6A(8-11) B6B(8-11) B6E(8-11) B6F(8-11) B6E(8-11) B6F(8-11)

                    const __m512i rhs_mat_014589CD_70_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_70, (_MM_PERM_ENUM)136); //B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3) B78(0-3) B79(0-3) B78(0-3) B79(0-3) B7C(0-3) B7D(0-3) B7C(0-3) B7D(0-3)
                    const __m512i rhs_mat_2367ABEF_70_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_70, (_MM_PERM_ENUM)136); //B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3) B7A(0-3) B7B(0-3) B7A(0-3) B7B(0-3) B7E(0-3) B7F(0-3) B7E(0-3) B7F(0-3)

                    const __m512i rhs_mat_014589CD_71_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_71, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_71_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_71, (_MM_PERM_ENUM)136); //B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11) B7A(8-11) B7B(8-11) B7A(8-11) B7B(8-11) B7E(8-11) B7F(8-11) B7E(8-11) B7F(8-11)

                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)

                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)

                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)

                    const __m512i rhs_mat_014589CD_20_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_20, (_MM_PERM_ENUM)221); //B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7) B28(4-7) B29(4-7) B28(4-7) B29(4-7) B2C(4-7) B2D(4-7) B2C(4-7) B2D(4-7)
                    const __m512i rhs_mat_2367ABEF_20_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_20, (_MM_PERM_ENUM)221); //B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7) B2A(4-7) B2B(4-7) B2A(4-7) B2B(4-7) B2E(4-7) B2F(4-7) B2E(4-7) B2F(4-7)

                    const __m512i rhs_mat_014589CD_21_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_21, (_MM_PERM_ENUM)221); //B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15) B28(12-15) B29(12-15) B28(12-15) B29(12-15) B2C(12-15) B2D(12-15) B2C(12-15) B2D(12-15)
                    const __m512i rhs_mat_2367ABEF_21_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_21, (_MM_PERM_ENUM)221); //B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15) B2A(12-15) B2B(12-15) B2A(12-15) B2B(12-15) B2E(12-15) B2F(12-15) B2E(12-15) B2F(12-15)

                    const __m512i rhs_mat_014589CD_30_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_30, (_MM_PERM_ENUM)221); //B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7) B38(4-7) B39(4-7) B38(4-7) B39(4-7) B3C(4-7) B3D(4-7) B3C(4-7) B3D(4-7)
                    const __m512i rhs_mat_2367ABEF_30_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_30, (_MM_PERM_ENUM)221); //B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7) B3A(4-7) B3B(4-7) B3A(4-7) B3B(4-7) B3E(4-7) B3F(4-7) B3E(4-7) B3F(4-7)

                    const __m512i rhs_mat_014589CD_31_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_31, (_MM_PERM_ENUM)221); //B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15) B38(12-15) B39(12-15) B38(12-15) B39(12-15) B3C(12-15) B3D(12-15) B3C(12-15) B3D(12-15)
                    const __m512i rhs_mat_2367ABEF_31_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_31, (_MM_PERM_ENUM)221); //B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15) B3A(12-15) B3B(12-15) B3A(12-15) B3B(12-15) B3E(12-15) B3F(12-15) B3E(12-15) B3F(12-15)

                    const __m512i rhs_mat_014589CD_40_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_40, (_MM_PERM_ENUM)221); //B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7) B48(4-7) B49(4-7) B48(4-7) B49(4-7) B4C(4-7) B4D(4-7) B4C(4-7) B4D(4-7)
                    const __m512i rhs_mat_2367ABEF_40_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_40, (_MM_PERM_ENUM)221); //B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7) B4A(4-7) B4B(4-7) B4A(4-7) B4B(4-7) B4E(4-7) B4F(4-7) B4E(4-7) B4F(4-7)

                    const __m512i rhs_mat_014589CD_41_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_41, (_MM_PERM_ENUM)221); //B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15) B48(12-15) B49(12-15) B48(12-15) B49(12-15) B4C(12-15) B4D(12-15) B4C(12-15) B4D(12-15)
                    const __m512i rhs_mat_2367ABEF_41_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_41, (_MM_PERM_ENUM)221); //B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15) B4A(12-15) B4B(12-15) B4A(12-15) B4B(12-15) B4E(12-15) B4F(12-15) B4E(12-15) B4F(12-15)

                    const __m512i rhs_mat_014589CD_50_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_50, (_MM_PERM_ENUM)221); //B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7) B58(4-7) B59(4-7) B58(4-7) B59(4-7) B5C(4-7) B5D(4-7) B5C(4-7) B5D(4-7)
                    const __m512i rhs_mat_2367ABEF_50_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_50, (_MM_PERM_ENUM)221); //B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7) B5A(4-7) B5B(4-7) B5A(4-7) B5B(4-7) B5E(4-7) B5F(4-7) B5E(4-7) B5F(4-7)

                    const __m512i rhs_mat_014589CD_51_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_51, (_MM_PERM_ENUM)221); //B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15) B58(12-15) B59(12-15) B58(12-15) B59(12-15) B5C(12-15) B5D(12-15) B5C(12-15) B5D(12-15)
                    const __m512i rhs_mat_2367ABEF_51_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_51, (_MM_PERM_ENUM)221); //B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15) B5A(12-15) B5B(12-15) B5A(12-15) B5B(12-15) B5E(12-15) B5F(12-15) B5E(12-15) B5F(12-15)

                    const __m512i rhs_mat_014589CD_60_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_60, (_MM_PERM_ENUM)221); //B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7) B68(4-7) B69(4-7) B68(4-7) B69(4-7) B6C(4-7) B6D(4-7) B6C(4-7) B6D(4-7)
                    const __m512i rhs_mat_2367ABEF_60_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_60, (_MM_PERM_ENUM)221); //B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7) B6A(4-7) B6B(4-7) B6A(4-7) B6B(4-7) B6E(4-7) B6F(4-7) B6E(4-7) B6F(4-7)

                    const __m512i rhs_mat_014589CD_61_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_61, (_MM_PERM_ENUM)221); //B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15) B68(12-15) B69(12-15) B68(12-15) B69(12-15) B6C(12-15) B6D(12-15) B6C(12-15) B6D(12-15)
                    const __m512i rhs_mat_2367ABEF_61_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_61, (_MM_PERM_ENUM)221); //B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15) B6A(12-15) B6B(12-15) B6A(12-15) B6B(12-15) B6E(12-15) B6F(12-15) B6E(12-15) B6F(12-15)

                    const __m512i rhs_mat_014589CD_70_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_70, (_MM_PERM_ENUM)221); //B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7) B78(4-7) B79(4-7) B78(4-7) B79(4-7) B7C(4-7) B7D(4-7) B7C(4-7) B7D(4-7)
                    const __m512i rhs_mat_2367ABEF_70_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_70, (_MM_PERM_ENUM)221); //B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7) B7A(4-7) B7B(4-7) B7A(4-7) B7B(4-7) B7E(4-7) B7F(4-7) B7E(4-7) B7F(4-7)

                    const __m512i rhs_mat_014589CD_71_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_71, (_MM_PERM_ENUM)221); //B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15) B78(12-15) B79(12-15) B78(12-15) B79(12-15) B7C(12-15) B7D(12-15) B7C(12-15) B7D(12-15)
                    const __m512i rhs_mat_2367ABEF_71_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_71, (_MM_PERM_ENUM)221); //B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15) B7A(12-15) B7B(12-15) B7A(12-15) B7B(12-15) B7E(12-15) B7F(12-15) B7E(12-15) B7F(12-15)

                    //notation:superblock subblock
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    const __m128i mins_and_scales_01_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + sb * 64));
                    const __m128i mins_and_scales_23_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 48 + sb * 64));

                    const __m128i mins_and_scales_01_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + sb * 64));
                    const __m128i mins_and_scales_23_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 48 + sb * 64));

                    // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
                    const __m256i mins_and_scales_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_01_0), mins_and_scales_01_1, 1);
                    const __m256i mins_and_scales_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_23_0), mins_and_scales_23_1, 1);
                    const __m256i mins_and_scales_45 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_45_0), mins_and_scales_45_1, 1);
                    const __m256i mins_and_scales_67 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_67_0), mins_and_scales_67_1, 1);

                    // Extract scales which is lower half from mins_and_scales
                    const __m256i scales_01 = _mm256_and_si256(mins_and_scales_01, m4b);
                    const __m256i scales_23 = _mm256_and_si256(mins_and_scales_23, m4b);
                    const __m256i scales_45 = _mm256_and_si256(mins_and_scales_45, m4b);
                    const __m256i scales_67 = _mm256_and_si256(mins_and_scales_67, m4b);

                    // Extract mins which is upper half from mins_and_scales
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_01, 4), m4b));
                    const __m512i mins_23 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_23, 4), m4b));
                    const __m512i mins_45 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_45, 4), m4b));
                    const __m512i mins_67 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_67, 4), m4b));

                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_01,scalesmask1));
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_01,scalesmask2));
                    const __m512i scales_2 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_23,scalesmask1));
                    const __m512i scales_3 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_23,scalesmask2));
                    const __m512i scales_4 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_45,scalesmask1));
                    const __m512i scales_5 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_45,scalesmask2));
                    const __m512i scales_6 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_67,scalesmask1));
                    const __m512i scales_7 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_67,scalesmask2));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_2 = _mm512_shuffle_epi32(scales_2, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_2 = _mm512_shuffle_epi32(scales_2, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_3 = _mm512_shuffle_epi32(scales_3, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_3 = _mm512_shuffle_epi32(scales_3, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_4 = _mm512_shuffle_epi32(scales_4, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_4 = _mm512_shuffle_epi32(scales_4, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_5 = _mm512_shuffle_epi32(scales_5, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_5 = _mm512_shuffle_epi32(scales_5, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_6 = _mm512_shuffle_epi32(scales_6, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_6 = _mm512_shuffle_epi32(scales_6, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_7 = _mm512_shuffle_epi32(scales_7, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_7 = _mm512_shuffle_epi32(scales_7, (_MM_PERM_ENUM)238);


                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                        __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 512 * sb)));
                        __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                        __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                        __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                        __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                        __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                        __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                        __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                        __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                        __m256i lhs_mat_ymm_0123_20 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_20 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_20, lhs_mat_ymm_0123_20, 0);
                        __m256i lhs_mat_ymm_23_20 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_20, lhs_mat_ymm_0123_20, 17);
                        __m256i lhs_mat_ymm_0123_21 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_21 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_21, lhs_mat_ymm_0123_21, 0);
                        __m256i lhs_mat_ymm_23_21 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_21, lhs_mat_ymm_0123_21, 17);
                        __m256i lhs_mat_ymm_0123_30 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_30 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_30, lhs_mat_ymm_0123_30, 0);
                        __m256i lhs_mat_ymm_23_30 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_30, lhs_mat_ymm_0123_30, 17);
                        __m256i lhs_mat_ymm_0123_31 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_31 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_31, lhs_mat_ymm_0123_31, 0);
                        __m256i lhs_mat_ymm_23_31 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_31, lhs_mat_ymm_0123_31, 17);

                        __m256i lhs_mat_ymm_0123_40 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_40 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_40, lhs_mat_ymm_0123_40, 0);
                        __m256i lhs_mat_ymm_23_40 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_40, lhs_mat_ymm_0123_40, 17);
                        __m256i lhs_mat_ymm_0123_41 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 288 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_41 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_41, lhs_mat_ymm_0123_41, 0);
                        __m256i lhs_mat_ymm_23_41 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_41, lhs_mat_ymm_0123_41, 17);
                        __m256i lhs_mat_ymm_0123_50 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 320 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_50 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_50, lhs_mat_ymm_0123_50, 0);
                        __m256i lhs_mat_ymm_23_50 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_50, lhs_mat_ymm_0123_50, 17);
                        __m256i lhs_mat_ymm_0123_51 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 352 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_51 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_51, lhs_mat_ymm_0123_51, 0);
                        __m256i lhs_mat_ymm_23_51 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_51, lhs_mat_ymm_0123_51, 17);
                        __m256i lhs_mat_ymm_0123_60 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 384 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_60 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_60, lhs_mat_ymm_0123_60, 0);
                        __m256i lhs_mat_ymm_23_60 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_60, lhs_mat_ymm_0123_60, 17);
                        __m256i lhs_mat_ymm_0123_61 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 416 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_61 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_61, lhs_mat_ymm_0123_61, 0);
                        __m256i lhs_mat_ymm_23_61 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_61, lhs_mat_ymm_0123_61, 17);
                        __m256i lhs_mat_ymm_0123_70 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 448 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_70 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_70, lhs_mat_ymm_0123_70, 0);
                        __m256i lhs_mat_ymm_23_70 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_70, lhs_mat_ymm_0123_70, 17);
                        __m256i lhs_mat_ymm_0123_71 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 480 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_71 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_71, lhs_mat_ymm_0123_71, 0);
                        __m256i lhs_mat_ymm_23_71 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_71, lhs_mat_ymm_0123_71, 17);


                        __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                        __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                        __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                        __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);

                        __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                        __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                        __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                        __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);

                        __m512i lhs_mat_01_20 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_20), lhs_mat_ymm_01_20, 1);
                        __m512i lhs_mat_23_20 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_20), lhs_mat_ymm_23_20, 1);
                        __m512i lhs_mat_01_21 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_21), lhs_mat_ymm_01_21, 1);
                        __m512i lhs_mat_23_21 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_21), lhs_mat_ymm_23_21, 1);

                        __m512i lhs_mat_01_30 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_30), lhs_mat_ymm_01_30, 1);
                        __m512i lhs_mat_23_30 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_30), lhs_mat_ymm_23_30, 1);
                        __m512i lhs_mat_01_31 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_31), lhs_mat_ymm_01_31, 1);
                        __m512i lhs_mat_23_31 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_31), lhs_mat_ymm_23_31, 1);

                        __m512i lhs_mat_01_40 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_40), lhs_mat_ymm_01_40, 1);
                        __m512i lhs_mat_23_40 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_40), lhs_mat_ymm_23_40, 1);
                        __m512i lhs_mat_01_41 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_41), lhs_mat_ymm_01_41, 1);
                        __m512i lhs_mat_23_41 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_41), lhs_mat_ymm_23_41, 1);

                        __m512i lhs_mat_01_50 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_50), lhs_mat_ymm_01_50, 1);
                        __m512i lhs_mat_23_50 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_50), lhs_mat_ymm_23_50, 1);
                        __m512i lhs_mat_01_51 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_51), lhs_mat_ymm_01_51, 1);
                        __m512i lhs_mat_23_51 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_51), lhs_mat_ymm_23_51, 1);

                        __m512i lhs_mat_01_60 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_60), lhs_mat_ymm_01_60, 1);
                        __m512i lhs_mat_23_60 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_60), lhs_mat_ymm_23_60, 1);
                        __m512i lhs_mat_01_61 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_61), lhs_mat_ymm_01_61, 1);
                        __m512i lhs_mat_23_61 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_61), lhs_mat_ymm_23_61, 1);

                        __m512i lhs_mat_01_70 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_70), lhs_mat_ymm_01_70, 1);
                        __m512i lhs_mat_23_70 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_70), lhs_mat_ymm_23_70, 1);
                        __m512i lhs_mat_01_71 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_71), lhs_mat_ymm_01_71, 1);
                        __m512i lhs_mat_23_71 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_71), lhs_mat_ymm_23_71, 1);

                        // Bsums are loaded for the different Q8_K blocks
                        __m128i lhs_raw_bsums_01_0123 = _mm_loadu_si128((const __m128i *)((a_ptrs[rp][b].bsums + 32 * sb)));
                        __m128i lhs_raw_bsums_23_0123 = _mm_loadu_si128((const __m128i *)(a_ptrs[rp][b].bsums + 8 + 32 * sb));
                        __m128i lhs_raw_bsums_01_4567 = _mm_loadu_si128((const __m128i *)((a_ptrs[rp][b].bsums + 16 + 32 * sb)));
                        __m128i lhs_raw_bsums_23_4567 = _mm_loadu_si128((const __m128i *)(a_ptrs[rp][b].bsums + 24 + 32 * sb));

                        __m256i lhs_bsums_ymm_01_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_0123), lhs_raw_bsums_01_0123, 1);
                        __m512i lhs_bsums_01_0123 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_01_0123), lhs_bsums_ymm_01_0123, 1);
                        __m256i lhs_bsums_ymm_23_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_0123), lhs_raw_bsums_23_0123, 1);
                        __m512i lhs_bsums_23_0123 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_23_0123), lhs_bsums_ymm_23_0123, 1);                        __m256i lhs_bsums_ymm_01_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_4567), lhs_raw_bsums_01_4567, 1);
                        __m512i lhs_bsums_01_4567 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_01_4567), lhs_bsums_ymm_01_4567, 1);
                        __m256i lhs_bsums_ymm_23_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_4567), lhs_raw_bsums_23_4567, 1);
                        __m512i lhs_bsums_23_4567 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_23_4567), lhs_bsums_ymm_23_4567, 1);

                        // Shuffle pattern one - left side input
                        const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)

                        const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)

                        const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)

                        const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)

                        const __m512i lhs_mat_01_20_sp1 = _mm512_shuffle_epi32(lhs_mat_01_20, (_MM_PERM_ENUM)160); //A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
                        const __m512i lhs_mat_23_20_sp1 = _mm512_shuffle_epi32(lhs_mat_23_20, (_MM_PERM_ENUM)160); //A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3)

                        const __m512i lhs_mat_01_21_sp1 = _mm512_shuffle_epi32(lhs_mat_01_21, (_MM_PERM_ENUM)160); //A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                        const __m512i lhs_mat_23_21_sp1 = _mm512_shuffle_epi32(lhs_mat_23_21, (_MM_PERM_ENUM)160); //A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11)

                        const __m512i lhs_mat_01_30_sp1 = _mm512_shuffle_epi32(lhs_mat_01_30, (_MM_PERM_ENUM)160); //A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
                        const __m512i lhs_mat_23_30_sp1 = _mm512_shuffle_epi32(lhs_mat_23_30, (_MM_PERM_ENUM)160); //A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3)

                        const __m512i lhs_mat_01_31_sp1 = _mm512_shuffle_epi32(lhs_mat_01_31, (_MM_PERM_ENUM)160); //A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                        const __m512i lhs_mat_23_31_sp1 = _mm512_shuffle_epi32(lhs_mat_23_31, (_MM_PERM_ENUM)160); //A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11)

                        const __m512i lhs_mat_01_40_sp1 = _mm512_shuffle_epi32(lhs_mat_01_40, (_MM_PERM_ENUM)160); //A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
                        const __m512i lhs_mat_23_40_sp1 = _mm512_shuffle_epi32(lhs_mat_23_40, (_MM_PERM_ENUM)160); //A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3)

                        const __m512i lhs_mat_01_41_sp1 = _mm512_shuffle_epi32(lhs_mat_01_41, (_MM_PERM_ENUM)160); //A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                        const __m512i lhs_mat_23_41_sp1 = _mm512_shuffle_epi32(lhs_mat_23_41, (_MM_PERM_ENUM)160); //A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11)

                        const __m512i lhs_mat_01_50_sp1 = _mm512_shuffle_epi32(lhs_mat_01_50, (_MM_PERM_ENUM)160); //A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
                        const __m512i lhs_mat_23_50_sp1 = _mm512_shuffle_epi32(lhs_mat_23_50, (_MM_PERM_ENUM)160); //A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3)

                        const __m512i lhs_mat_01_51_sp1 = _mm512_shuffle_epi32(lhs_mat_01_51, (_MM_PERM_ENUM)160); //A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                        const __m512i lhs_mat_23_51_sp1 = _mm512_shuffle_epi32(lhs_mat_23_51, (_MM_PERM_ENUM)160); //A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11)

                        const __m512i lhs_mat_01_60_sp1 = _mm512_shuffle_epi32(lhs_mat_01_60, (_MM_PERM_ENUM)160); //A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
                        const __m512i lhs_mat_23_60_sp1 = _mm512_shuffle_epi32(lhs_mat_23_60, (_MM_PERM_ENUM)160); //A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3)

                        const __m512i lhs_mat_01_61_sp1 = _mm512_shuffle_epi32(lhs_mat_01_61, (_MM_PERM_ENUM)160); //A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                        const __m512i lhs_mat_23_61_sp1 = _mm512_shuffle_epi32(lhs_mat_23_61, (_MM_PERM_ENUM)160); //A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11)

                        const __m512i lhs_mat_01_70_sp1 = _mm512_shuffle_epi32(lhs_mat_01_70, (_MM_PERM_ENUM)160); //A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
                        const __m512i lhs_mat_23_70_sp1 = _mm512_shuffle_epi32(lhs_mat_23_70, (_MM_PERM_ENUM)160); //A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3)

                        const __m512i lhs_mat_01_71_sp1 = _mm512_shuffle_epi32(lhs_mat_01_71, (_MM_PERM_ENUM)160); //A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                        const __m512i lhs_mat_23_71_sp1 = _mm512_shuffle_epi32(lhs_mat_23_71, (_MM_PERM_ENUM)160); //A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11)

                        const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)

                        const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)

                        const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)

                        const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)

                        const __m512i lhs_mat_01_20_sp2 = _mm512_shuffle_epi32(lhs_mat_01_20, (_MM_PERM_ENUM)245); //A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
                        const __m512i lhs_mat_23_20_sp2 = _mm512_shuffle_epi32(lhs_mat_23_20, (_MM_PERM_ENUM)245); //A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7)

                        const __m512i lhs_mat_01_21_sp2 = _mm512_shuffle_epi32(lhs_mat_01_21, (_MM_PERM_ENUM)245); //A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                        const __m512i lhs_mat_23_21_sp2 = _mm512_shuffle_epi32(lhs_mat_23_21, (_MM_PERM_ENUM)245); //A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15)

                        const __m512i lhs_mat_01_30_sp2 = _mm512_shuffle_epi32(lhs_mat_01_30, (_MM_PERM_ENUM)245); //A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
                        const __m512i lhs_mat_23_30_sp2 = _mm512_shuffle_epi32(lhs_mat_23_30, (_MM_PERM_ENUM)245); //A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7)

                        const __m512i lhs_mat_01_31_sp2 = _mm512_shuffle_epi32(lhs_mat_01_31, (_MM_PERM_ENUM)245); //A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                        const __m512i lhs_mat_23_31_sp2 = _mm512_shuffle_epi32(lhs_mat_23_31, (_MM_PERM_ENUM)245); //A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15)

                        const __m512i lhs_mat_01_40_sp2 = _mm512_shuffle_epi32(lhs_mat_01_40, (_MM_PERM_ENUM)245); //A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
                        const __m512i lhs_mat_23_40_sp2 = _mm512_shuffle_epi32(lhs_mat_23_40, (_MM_PERM_ENUM)245); //A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7)

                        const __m512i lhs_mat_01_41_sp2 = _mm512_shuffle_epi32(lhs_mat_01_41, (_MM_PERM_ENUM)245); //A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                        const __m512i lhs_mat_23_41_sp2 = _mm512_shuffle_epi32(lhs_mat_23_41, (_MM_PERM_ENUM)245); //A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15)

                        const __m512i lhs_mat_01_50_sp2 = _mm512_shuffle_epi32(lhs_mat_01_50, (_MM_PERM_ENUM)245); //A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
                        const __m512i lhs_mat_23_50_sp2 = _mm512_shuffle_epi32(lhs_mat_23_50, (_MM_PERM_ENUM)245); //A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7)

                        const __m512i lhs_mat_01_51_sp2 = _mm512_shuffle_epi32(lhs_mat_01_51, (_MM_PERM_ENUM)245); //A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                        const __m512i lhs_mat_23_51_sp2 = _mm512_shuffle_epi32(lhs_mat_23_51, (_MM_PERM_ENUM)245); //A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15)

                        const __m512i lhs_mat_01_60_sp2 = _mm512_shuffle_epi32(lhs_mat_01_60, (_MM_PERM_ENUM)245); //A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
                        const __m512i lhs_mat_23_60_sp2 = _mm512_shuffle_epi32(lhs_mat_23_60, (_MM_PERM_ENUM)245); //A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7)

                        const __m512i lhs_mat_01_61_sp2 = _mm512_shuffle_epi32(lhs_mat_01_61, (_MM_PERM_ENUM)245); //A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                        const __m512i lhs_mat_23_61_sp2 = _mm512_shuffle_epi32(lhs_mat_23_61, (_MM_PERM_ENUM)245); //A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15)

                        const __m512i lhs_mat_01_70_sp2 = _mm512_shuffle_epi32(lhs_mat_01_70, (_MM_PERM_ENUM)245); //A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
                        const __m512i lhs_mat_23_70_sp2 = _mm512_shuffle_epi32(lhs_mat_23_70, (_MM_PERM_ENUM)245); //A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7)

                        const __m512i lhs_mat_01_71_sp2 = _mm512_shuffle_epi32(lhs_mat_01_71, (_MM_PERM_ENUM)245); //A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                        const __m512i lhs_mat_23_71_sp2 = _mm512_shuffle_epi32(lhs_mat_23_71, (_MM_PERM_ENUM)245); //A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1));
                        __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1));

                        __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1));
                        __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1));

                        __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1));
                        __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1));

                        __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1));
                        __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1));

                        __m512i iacc_mat_00_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp1, lhs_mat_01_20_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp1, lhs_mat_01_21_sp1));
                        __m512i iacc_mat_01_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp1, lhs_mat_01_20_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp1, lhs_mat_01_21_sp1));

                        __m512i iacc_mat_10_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp1, lhs_mat_23_20_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp1, lhs_mat_23_21_sp1));
                        __m512i iacc_mat_11_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp1, lhs_mat_23_20_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp1, lhs_mat_23_21_sp1));

                        __m512i iacc_mat_00_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp1, lhs_mat_01_30_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp1, lhs_mat_01_31_sp1));
                        __m512i iacc_mat_01_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp1, lhs_mat_01_30_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp1, lhs_mat_01_31_sp1));

                        __m512i iacc_mat_10_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp1, lhs_mat_23_30_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp1, lhs_mat_23_31_sp1));
                        __m512i iacc_mat_11_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp1, lhs_mat_23_30_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp1, lhs_mat_23_31_sp1));

                        __m512i iacc_mat_00_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp1, lhs_mat_01_40_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp1, lhs_mat_01_41_sp1));
                        __m512i iacc_mat_01_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp1, lhs_mat_01_40_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp1, lhs_mat_01_41_sp1));

                        __m512i iacc_mat_10_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp1, lhs_mat_23_40_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp1, lhs_mat_23_41_sp1));
                        __m512i iacc_mat_11_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp1, lhs_mat_23_40_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp1, lhs_mat_23_41_sp1));

                        __m512i iacc_mat_00_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp1, lhs_mat_01_50_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp1, lhs_mat_01_51_sp1));
                        __m512i iacc_mat_01_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp1, lhs_mat_01_50_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp1, lhs_mat_01_51_sp1));

                        __m512i iacc_mat_10_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp1, lhs_mat_23_50_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp1, lhs_mat_23_51_sp1));
                        __m512i iacc_mat_11_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp1, lhs_mat_23_50_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp1, lhs_mat_23_51_sp1));

                        __m512i iacc_mat_00_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp1, lhs_mat_01_60_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp1, lhs_mat_01_61_sp1));
                        __m512i iacc_mat_01_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp1, lhs_mat_01_60_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp1, lhs_mat_01_61_sp1));

                        __m512i iacc_mat_10_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp1, lhs_mat_23_60_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp1, lhs_mat_23_61_sp1));
                        __m512i iacc_mat_11_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp1, lhs_mat_23_60_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp1, lhs_mat_23_61_sp1));

                        __m512i iacc_mat_00_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp1, lhs_mat_01_70_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp1, lhs_mat_01_71_sp1));
                        __m512i iacc_mat_01_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp1, lhs_mat_01_70_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp1, lhs_mat_01_71_sp1));

                        __m512i iacc_mat_10_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp1, lhs_mat_23_70_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp1, lhs_mat_23_71_sp1));
                        __m512i iacc_mat_11_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp1, lhs_mat_23_70_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp1, lhs_mat_23_71_sp1));


                        __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2));
                        __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2));

                        __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2));
                        __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2));

                        __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2));
                        __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2));

                        __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2));
                        __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2));

                        __m512i iacc_mat_00_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp2, lhs_mat_01_20_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp2, lhs_mat_01_21_sp2));
                        __m512i iacc_mat_01_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp2, lhs_mat_01_20_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp2, lhs_mat_01_21_sp2));

                        __m512i iacc_mat_10_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp2, lhs_mat_23_20_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp2, lhs_mat_23_21_sp2));
                        __m512i iacc_mat_11_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp2, lhs_mat_23_20_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp2, lhs_mat_23_21_sp2));

                        __m512i iacc_mat_00_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp2, lhs_mat_01_30_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp2, lhs_mat_01_31_sp2));
                        __m512i iacc_mat_01_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp2, lhs_mat_01_30_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp2, lhs_mat_01_31_sp2));

                        __m512i iacc_mat_10_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp2, lhs_mat_23_30_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp2, lhs_mat_23_31_sp2));
                        __m512i iacc_mat_11_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp2, lhs_mat_23_30_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp2, lhs_mat_23_31_sp2));

                        __m512i iacc_mat_00_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp2, lhs_mat_01_40_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp2, lhs_mat_01_41_sp2));
                        __m512i iacc_mat_01_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp2, lhs_mat_01_40_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp2, lhs_mat_01_41_sp2));

                        __m512i iacc_mat_10_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp2, lhs_mat_23_40_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp2, lhs_mat_23_41_sp2));
                        __m512i iacc_mat_11_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp2, lhs_mat_23_40_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp2, lhs_mat_23_41_sp2));

                        __m512i iacc_mat_00_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp2, lhs_mat_01_50_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp2, lhs_mat_01_51_sp2));
                        __m512i iacc_mat_01_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp2, lhs_mat_01_50_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp2, lhs_mat_01_51_sp2));

                        __m512i iacc_mat_10_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp2, lhs_mat_23_50_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp2, lhs_mat_23_51_sp2));
                        __m512i iacc_mat_11_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp2, lhs_mat_23_50_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp2, lhs_mat_23_51_sp2));

                        __m512i iacc_mat_00_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp2, lhs_mat_01_60_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp2, lhs_mat_01_61_sp2));
                        __m512i iacc_mat_01_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp2, lhs_mat_01_60_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp2, lhs_mat_01_61_sp2));

                        __m512i iacc_mat_10_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp2, lhs_mat_23_60_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp2, lhs_mat_23_61_sp2));
                        __m512i iacc_mat_11_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp2, lhs_mat_23_60_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp2, lhs_mat_23_61_sp2));

                        __m512i iacc_mat_00_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp2, lhs_mat_01_70_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp2, lhs_mat_01_71_sp2));
                        __m512i iacc_mat_01_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp2, lhs_mat_01_70_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp2, lhs_mat_01_71_sp2));

                        __m512i iacc_mat_10_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp2, lhs_mat_23_70_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp2, lhs_mat_23_71_sp2));
                        __m512i iacc_mat_11_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp2, lhs_mat_23_70_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp2, lhs_mat_23_71_sp2));

                        // Combine results from both shuffle patterns for each output block
                        __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        __m512i iacc_mat_00_2 = _mm512_add_epi16(iacc_mat_00_2_sp1, iacc_mat_00_2_sp2);
                        __m512i iacc_mat_01_2 = _mm512_add_epi16(iacc_mat_01_2_sp1, iacc_mat_01_2_sp2);
                        __m512i iacc_mat_10_2 = _mm512_add_epi16(iacc_mat_10_2_sp1, iacc_mat_10_2_sp2);
                        __m512i iacc_mat_11_2 = _mm512_add_epi16(iacc_mat_11_2_sp1, iacc_mat_11_2_sp2);

                        __m512i iacc_mat_00_3 = _mm512_add_epi16(iacc_mat_00_3_sp1, iacc_mat_00_3_sp2);
                        __m512i iacc_mat_01_3 = _mm512_add_epi16(iacc_mat_01_3_sp1, iacc_mat_01_3_sp2);
                        __m512i iacc_mat_10_3 = _mm512_add_epi16(iacc_mat_10_3_sp1, iacc_mat_10_3_sp2);
                        __m512i iacc_mat_11_3 = _mm512_add_epi16(iacc_mat_11_3_sp1, iacc_mat_11_3_sp2);

                        __m512i iacc_mat_00_4 = _mm512_add_epi16(iacc_mat_00_4_sp1, iacc_mat_00_4_sp2);
                        __m512i iacc_mat_01_4 = _mm512_add_epi16(iacc_mat_01_4_sp1, iacc_mat_01_4_sp2);
                        __m512i iacc_mat_10_4 = _mm512_add_epi16(iacc_mat_10_4_sp1, iacc_mat_10_4_sp2);
                        __m512i iacc_mat_11_4 = _mm512_add_epi16(iacc_mat_11_4_sp1, iacc_mat_11_4_sp2);

                        __m512i iacc_mat_00_5 = _mm512_add_epi16(iacc_mat_00_5_sp1, iacc_mat_00_5_sp2);
                        __m512i iacc_mat_01_5 = _mm512_add_epi16(iacc_mat_01_5_sp1, iacc_mat_01_5_sp2);
                        __m512i iacc_mat_10_5 = _mm512_add_epi16(iacc_mat_10_5_sp1, iacc_mat_10_5_sp2);
                        __m512i iacc_mat_11_5 = _mm512_add_epi16(iacc_mat_11_5_sp1, iacc_mat_11_5_sp2);

                        __m512i iacc_mat_00_6 = _mm512_add_epi16(iacc_mat_00_6_sp1, iacc_mat_00_6_sp2);
                        __m512i iacc_mat_01_6 = _mm512_add_epi16(iacc_mat_01_6_sp1, iacc_mat_01_6_sp2);
                        __m512i iacc_mat_10_6 = _mm512_add_epi16(iacc_mat_10_6_sp1, iacc_mat_10_6_sp2);
                        __m512i iacc_mat_11_6 = _mm512_add_epi16(iacc_mat_11_6_sp1, iacc_mat_11_6_sp2);

                        __m512i iacc_mat_00_7 = _mm512_add_epi16(iacc_mat_00_7_sp1, iacc_mat_00_7_sp2);
                        __m512i iacc_mat_01_7 = _mm512_add_epi16(iacc_mat_01_7_sp1, iacc_mat_01_7_sp2);
                        __m512i iacc_mat_10_7 = _mm512_add_epi16(iacc_mat_10_7_sp1, iacc_mat_10_7_sp2);
                        __m512i iacc_mat_11_7 = _mm512_add_epi16(iacc_mat_11_7_sp1, iacc_mat_11_7_sp2);

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                        iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                        iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                        iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                        iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                        iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                        iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                        iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                        iacc_mat_00_2 = _mm512_madd_epi16(iacc_mat_00_2, scale_014589CD_2);
                        iacc_mat_01_2 = _mm512_madd_epi16(iacc_mat_01_2, scale_2367ABEF_2);
                        iacc_mat_10_2 = _mm512_madd_epi16(iacc_mat_10_2, scale_014589CD_2);
                        iacc_mat_11_2 = _mm512_madd_epi16(iacc_mat_11_2, scale_2367ABEF_2);

                        iacc_mat_00_3 = _mm512_madd_epi16(iacc_mat_00_3, scale_014589CD_3);
                        iacc_mat_01_3 = _mm512_madd_epi16(iacc_mat_01_3, scale_2367ABEF_3);
                        iacc_mat_10_3 = _mm512_madd_epi16(iacc_mat_10_3, scale_014589CD_3);
                        iacc_mat_11_3 = _mm512_madd_epi16(iacc_mat_11_3, scale_2367ABEF_3);

                        iacc_mat_00_4 = _mm512_madd_epi16(iacc_mat_00_4, scale_014589CD_4);
                        iacc_mat_01_4 = _mm512_madd_epi16(iacc_mat_01_4, scale_2367ABEF_4);
                        iacc_mat_10_4 = _mm512_madd_epi16(iacc_mat_10_4, scale_014589CD_4);
                        iacc_mat_11_4 = _mm512_madd_epi16(iacc_mat_11_4, scale_2367ABEF_4);

                        iacc_mat_00_5 = _mm512_madd_epi16(iacc_mat_00_5, scale_014589CD_5);
                        iacc_mat_01_5 = _mm512_madd_epi16(iacc_mat_01_5, scale_2367ABEF_5);
                        iacc_mat_10_5 = _mm512_madd_epi16(iacc_mat_10_5, scale_014589CD_5);
                        iacc_mat_11_5 = _mm512_madd_epi16(iacc_mat_11_5, scale_2367ABEF_5);

                        iacc_mat_00_6 = _mm512_madd_epi16(iacc_mat_00_6, scale_014589CD_6);
                        iacc_mat_01_6 = _mm512_madd_epi16(iacc_mat_01_6, scale_2367ABEF_6);
                        iacc_mat_10_6 = _mm512_madd_epi16(iacc_mat_10_6, scale_014589CD_6);
                        iacc_mat_11_6 = _mm512_madd_epi16(iacc_mat_11_6, scale_2367ABEF_6);

                        iacc_mat_00_7 = _mm512_madd_epi16(iacc_mat_00_7, scale_014589CD_7);
                        iacc_mat_01_7 = _mm512_madd_epi16(iacc_mat_01_7, scale_2367ABEF_7);
                        iacc_mat_10_7 = _mm512_madd_epi16(iacc_mat_10_7, scale_014589CD_7);
                        iacc_mat_11_7 = _mm512_madd_epi16(iacc_mat_11_7, scale_2367ABEF_7);

                        __m512i iacc_mat_00 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_00_0, iacc_mat_00_1), _mm512_add_epi32(iacc_mat_00_2, iacc_mat_00_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_00_4, iacc_mat_00_5), _mm512_add_epi32(iacc_mat_00_6, iacc_mat_00_7)));
                        __m512i iacc_mat_01 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_01_0, iacc_mat_01_1), _mm512_add_epi32(iacc_mat_01_2, iacc_mat_01_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_01_4, iacc_mat_01_5), _mm512_add_epi32(iacc_mat_01_6, iacc_mat_01_7)));
                        __m512i iacc_mat_10 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_10_0, iacc_mat_10_1), _mm512_add_epi32(iacc_mat_10_2, iacc_mat_10_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_10_4, iacc_mat_10_5), _mm512_add_epi32(iacc_mat_10_6, iacc_mat_10_7)));
                        __m512i iacc_mat_11 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_11_0, iacc_mat_11_1), _mm512_add_epi32(iacc_mat_11_2, iacc_mat_11_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_11_4, iacc_mat_11_5), _mm512_add_epi32(iacc_mat_11_6, iacc_mat_11_7)));

                        // Straighten out to make 4 row vectors
                        __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                        __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                        const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                        // Multiply with appropriate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4  + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
                        __m512i iacc_row_min_0_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)0), mins_01);
                        __m512i iacc_row_min_1_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)170), mins_01);
                        __m512i iacc_row_min_2_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)0), mins_01);
                        __m512i iacc_row_min_3_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)170), mins_01);

                        __m512i iacc_row_min_0_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)85), mins_23);
                        __m512i iacc_row_min_1_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)255), mins_23);
                        __m512i iacc_row_min_2_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)85), mins_23);
                        __m512i iacc_row_min_3_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)255), mins_23);

                        __m512i iacc_row_min_0_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)0), mins_45);
                        __m512i iacc_row_min_1_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)170), mins_45);
                        __m512i iacc_row_min_2_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)0), mins_45);
                        __m512i iacc_row_min_3_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)170), mins_45);

                        __m512i iacc_row_min_0_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)85), mins_67);
                        __m512i iacc_row_min_1_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)255), mins_67);
                        __m512i iacc_row_min_2_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)85), mins_67);
                        __m512i iacc_row_min_3_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)255), mins_67);

                        __m512i iacc_row_min_0 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_0_01, iacc_row_min_0_23), _mm512_add_epi32(iacc_row_min_0_45,iacc_row_min_0_67));
                        __m512i iacc_row_min_1 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_1_01, iacc_row_min_1_23), _mm512_add_epi32(iacc_row_min_1_45,iacc_row_min_1_67));
                        __m512i iacc_row_min_2 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_2_01, iacc_row_min_2_23), _mm512_add_epi32(iacc_row_min_2_45,iacc_row_min_2_67));
                        __m512i iacc_row_min_3 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_3_01, iacc_row_min_3_23), _mm512_add_epi32(iacc_row_min_3_45,iacc_row_min_3_67));

                        acc_min_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);
                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

    for (; y < nr / 4; y ++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q2_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q2_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }
            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Delta values - Load the sixteen scale values from two block_q2_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q2_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_k for eight sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //2-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0,m3bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0,m3bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1,m3bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1,m3bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)
                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(rhs_raw_mat_014589CD_2,m3bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2,m3bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(rhs_raw_mat_014589CD_3,m3bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3,m3bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_20 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 2), m3bexpanded); //B20(0-7) B21(0-7) B24(0-7) B25(0-7) B28(0-7) B29(0-7) B2C(0-7) B2D(0-7)
                    const __m512i rhs_mat_2367ABEF_20 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 2), m3bexpanded); //B22(0-7) B23(0-7) B26(0-7) B27(0-7) B2A(0-7) B2B(0-7) B2E(0-7) B2F(0-7)

                    const __m512i rhs_mat_014589CD_21 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 2), m3bexpanded); //B20(8-15) B21(8-15) B24(8-15) B25(8-15) B28(8-15) B29(8-15) B2C(8-15) B2D(8-15)
                    const __m512i rhs_mat_2367ABEF_21 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 2), m3bexpanded); //B22(8-15) B23(8-15) B26(8-15) B27(8-15) B2A(8-15) B2B(8-15) B2E(8-15) B2F(8-15)

                    const __m512i rhs_mat_014589CD_30 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 2), m3bexpanded); //B30(0-7) B31(0-7) B34(0-7) B35(0-7) B38(0-7) B39(0-7) B3C(0-7) B3D(0-7)
                    const __m512i rhs_mat_2367ABEF_30 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 2), m3bexpanded); //B32(0-7) B33(0-7) B36(0-7) B37(0-7) B3A(0-7) B3B(0-7) B3E(0-7) B3F(0-7)

                    const __m512i rhs_mat_014589CD_31 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 2), m3bexpanded); //B30(8-15) B31(8-15) B34(8-15) B35(8-15) B38(8-15) B39(8-15) B3C(8-15) B3D(8-15)
                    const __m512i rhs_mat_2367ABEF_31 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 2), m3bexpanded); //B32(8-15) B33(8-15) B36(8-15) B37(8-15) B3A(8-15) B3B(8-15) B3E(8-15) B3F(8-15)

                    const __m512i rhs_mat_014589CD_40 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m3bexpanded); //B40(0-7) B41(0-7) B44(0-7) B45(0-7) B48(0-7) B49(0-7) B4C(0-7) B4D(0-7)
                    const __m512i rhs_mat_2367ABEF_40 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m3bexpanded); //B42(0-7) B43(0-7) B46(0-7) B47(0-7) B4A(0-7) B4B(0-7) B4E(0-7) B4F(0-7)

                    const __m512i rhs_mat_014589CD_41 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m3bexpanded); //B40(8-15) B41(8-15) B44(8-15) B45(8-15) B48(8-15) B49(8-15) B4C(8-15) B4D(8-15)
                    const __m512i rhs_mat_2367ABEF_41 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m3bexpanded); //B42(8-15) B43(8-15) B46(8-15) B47(8-15) B4A(8-15) B4B(8-15) B4E(8-15) B4F(8-15)

                    const __m512i rhs_mat_014589CD_50 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m3bexpanded); //B50(0-7) B51(0-7) B54(0-7) B55(0-7) B58(0-7) B59(0-7) B5C(0-7) B5D(0-7)
                    const __m512i rhs_mat_2367ABEF_50 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m3bexpanded); //B52(0-7) B53(0-7) B56(0-7) B57(0-7) B5A(0-7) B5B(0-7) B5E(0-7) B5F(0-7)

                    const __m512i rhs_mat_014589CD_51 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m3bexpanded); //B50(8-15) B51(8-15) B54(8-15) B55(8-15) B58(8-15) B59(8-15) B5C(8-15) B5D(8-15)
                    const __m512i rhs_mat_2367ABEF_51 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m3bexpanded); //B52(8-15) B53(8-15) B56(8-15) B57(8-15) B5A(8-15) B5B(8-15) B5E(8-15) B5F(8-15)

                    const __m512i rhs_mat_014589CD_60 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 6), m3bexpanded); //B60(0-7) B61(0-7) B64(0-7) B65(0-7) B68(0-7) B69(0-7) B6C(0-7) B6D(0-7)
                    const __m512i rhs_mat_2367ABEF_60 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 6), m3bexpanded); //B62(0-7) B63(0-7) B66(0-7) B67(0-7) B6A(0-7) B6B(0-7) B6E(0-7) B6F(0-7)

                    const __m512i rhs_mat_014589CD_61 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 6), m3bexpanded); //B60(8-15) B61(8-15) B64(8-15) B65(8-15) B68(8-15) B69(8-15) B6C(8-15) B6D(8-15)
                    const __m512i rhs_mat_2367ABEF_61 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 6), m3bexpanded); //B62(8-15) B63(8-15) B66(8-15) B67(8-15) B6A(8-15) B6B(8-15) B6E(8-15) B6F(8-15)

                    const __m512i rhs_mat_014589CD_70 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 6), m3bexpanded); //B70(0-7) B71(0-7) B74(0-7) B75(0-7) B78(0-7) B79(0-7) B7C(0-7) B7D(0-7)
                    const __m512i rhs_mat_2367ABEF_70 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 6), m3bexpanded); //B72(0-7) B73(0-7) B76(0-7) B77(0-7) B7A(0-7) B7B(0-7) B7E(0-7) B7F(0-7)

                    const __m512i rhs_mat_014589CD_71 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 6), m3bexpanded); //B70(8-15) B71(8-15) B74(8-15) B75(8-15) B78(8-15) B79(8-15) B7C(8-15) B7D(8-15)
                    const __m512i rhs_mat_2367ABEF_71 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 6), m3bexpanded); //B72(8-15) B73(8-15) B76(8-15) B77(8-15) B7A(8-15) B7B(8-15) B7E(8-15) B7F(8-15)

                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)

                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)

                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)

                    const __m512i rhs_mat_014589CD_20_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_20, (_MM_PERM_ENUM)136); //B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3) B28(0-3) B29(0-3) B28(0-3) B29(0-3) B2C(0-3) B2D(0-3) B2C(0-3) B2D(0-3)
                    const __m512i rhs_mat_2367ABEF_20_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_20, (_MM_PERM_ENUM)136); //B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3) B2A(0-3) B2B(0-3) B2A(0-3) B2B(0-3) B2E(0-3) B2F(0-3) B2E(0-3) B2F(0-3)

                    const __m512i rhs_mat_014589CD_21_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_21, (_MM_PERM_ENUM)136); //B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11) B28(8-11) B29(8-11) B28(8-11) B29(8-11) B2C(8-11) B2D(8-11) B2C(8-11) B2D(8-11)
                    const __m512i rhs_mat_2367ABEF_21_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_21, (_MM_PERM_ENUM)136); //B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11) B2A(8-11) B2B(8-11) B2A(8-11) B2B(8-11) B2E(8-11) B2F(8-11) B2E(8-11) B2F(8-11)
                    const __m512i rhs_mat_014589CD_30_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_30, (_MM_PERM_ENUM)136); ///B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3) B38(0-3) B39(0-3) B38(0-3) B39(0-3) B3C(0-3) B3D(0-3) B3C(0-3) B3D(0-3)
                    const __m512i rhs_mat_2367ABEF_30_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_30, (_MM_PERM_ENUM)136); //B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3) B3A(0-3) B3B(0-3) B3A(0-3) B3B(0-3) B3E(0-3) B3F(0-3) B3E(0-3) B3F(0-3)

                    const __m512i rhs_mat_014589CD_31_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_31, (_MM_PERM_ENUM)136); //B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11) B38(8-11) B39(8-11) B38(8-11) B39(8-11) B3C(8-11) B3D(8-11) B3C(8-11) B3D(8-11)
                    const __m512i rhs_mat_2367ABEF_31_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_31, (_MM_PERM_ENUM)136); //B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11) B3A(8-11) B3B(8-11) B3A(8-11) B3B(8-11) B3E(8-11) B3F(8-11) B3E(8-11) B3F(8-11)

                    const __m512i rhs_mat_014589CD_40_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_40, (_MM_PERM_ENUM)136); //B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3) B48(0-3) B49(0-3) B48(0-3) B49(0-3) B4C(0-3) B4D(0-3) B4C(0-3) B4D(0-3)
                    const __m512i rhs_mat_2367ABEF_40_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_40, (_MM_PERM_ENUM)136); //B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3) B4A(0-3) B4B(0-3) B4A(0-3) B4B(0-3) B4E(0-3) B4F(0-3) B4E(0-3) B4F(0-3)

                    const __m512i rhs_mat_014589CD_41_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_41, (_MM_PERM_ENUM)136); //B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11) B48(8-11) B49(8-11) B48(8-11) B49(8-11) B4C(8-11) B4D(8-11) B4C(8-11) B4D(8-11)
                    const __m512i rhs_mat_2367ABEF_41_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_41, (_MM_PERM_ENUM)136); //B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11) B4A(8-11) B4B(8-11) B4A(8-11) B4B(8-11) B4E(8-11) B4F(8-11) B4E(8-11) B4F(8-11)

                    const __m512i rhs_mat_014589CD_50_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_50, (_MM_PERM_ENUM)136); //B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3) B58(0-3) B59(0-3) B58(0-3) B59(0-3) B5C(0-3) B5D(0-3) B5C(0-3) B5D(0-3)
                    const __m512i rhs_mat_2367ABEF_50_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_50, (_MM_PERM_ENUM)136); //B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3) B5A(0-3) B5B(0-3) B5A(0-3) B5B(0-3) B5E(0-3) B5F(0-3) B5E(0-3) B5F(0-3)

                    const __m512i rhs_mat_014589CD_51_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_51, (_MM_PERM_ENUM)136); //B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11) B58(8-11) B59(8-11) B58(8-11) B59(8-11) B5C(8-11) B5D(8-11) B5C(8-11) B5D(8-11)
                    const __m512i rhs_mat_2367ABEF_51_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_51, (_MM_PERM_ENUM)136); //B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11) B5A(8-11) B5B(8-11) B5A(8-11) B5B(8-11) B5E(8-11) B5F(8-11) B5E(8-11) B5F(8-11)

                    const __m512i rhs_mat_014589CD_60_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_60, (_MM_PERM_ENUM)136); //B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3) B68(0-3) B69(0-3) B68(0-3) B69(0-3) B6C(0-3) B6D(0-3) B6C(0-3) B6D(0-3)
                    const __m512i rhs_mat_2367ABEF_60_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_60, (_MM_PERM_ENUM)136); //B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3) B6A(0-3) B6B(0-3) B6A(0-3) B6B(0-3) B6E(0-3) B6F(0-3) B6E(0-3) B6F(0-3)

                    const __m512i rhs_mat_014589CD_61_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_61, (_MM_PERM_ENUM)136); //B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11) B68(8-11) B69(8-11) B68(8-11) B69(8-11) B6C(8-11) B6D(8-11) B6C(8-11) B6D(8-11)
                    const __m512i rhs_mat_2367ABEF_61_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_61, (_MM_PERM_ENUM)136); //B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11) B6A(8-11) B6B(8-11) B6A(8-11) B6B(8-11) B6E(8-11) B6F(8-11) B6E(8-11) B6F(8-11)

                    const __m512i rhs_mat_014589CD_70_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_70, (_MM_PERM_ENUM)136); //B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3) B78(0-3) B79(0-3) B78(0-3) B79(0-3) B7C(0-3) B7D(0-3) B7C(0-3) B7D(0-3)
                    const __m512i rhs_mat_2367ABEF_70_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_70, (_MM_PERM_ENUM)136); //B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3) B7A(0-3) B7B(0-3) B7A(0-3) B7B(0-3) B7E(0-3) B7F(0-3) B7E(0-3) B7F(0-3)

                    const __m512i rhs_mat_014589CD_71_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_71, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_71_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_71, (_MM_PERM_ENUM)136); //B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11) B7A(8-11) B7B(8-11) B7A(8-11) B7B(8-11) B7E(8-11) B7F(8-11) B7E(8-11) B7F(8-11)

                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)

                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)

                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)

                    const __m512i rhs_mat_014589CD_20_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_20, (_MM_PERM_ENUM)221); //B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7) B28(4-7) B29(4-7) B28(4-7) B29(4-7) B2C(4-7) B2D(4-7) B2C(4-7) B2D(4-7)
                    const __m512i rhs_mat_2367ABEF_20_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_20, (_MM_PERM_ENUM)221); //B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7) B2A(4-7) B2B(4-7) B2A(4-7) B2B(4-7) B2E(4-7) B2F(4-7) B2E(4-7) B2F(4-7)

                    const __m512i rhs_mat_014589CD_21_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_21, (_MM_PERM_ENUM)221); //B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15) B28(12-15) B29(12-15) B28(12-15) B29(12-15) B2C(12-15) B2D(12-15) B2C(12-15) B2D(12-15)
                    const __m512i rhs_mat_2367ABEF_21_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_21, (_MM_PERM_ENUM)221); //B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15) B2A(12-15) B2B(12-15) B2A(12-15) B2B(12-15) B2E(12-15) B2F(12-15) B2E(12-15) B2F(12-15)

                    const __m512i rhs_mat_014589CD_30_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_30, (_MM_PERM_ENUM)221); //B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7) B38(4-7) B39(4-7) B38(4-7) B39(4-7) B3C(4-7) B3D(4-7) B3C(4-7) B3D(4-7)
                    const __m512i rhs_mat_2367ABEF_30_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_30, (_MM_PERM_ENUM)221); //B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7) B3A(4-7) B3B(4-7) B3A(4-7) B3B(4-7) B3E(4-7) B3F(4-7) B3E(4-7) B3F(4-7)

                    const __m512i rhs_mat_014589CD_31_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_31, (_MM_PERM_ENUM)221); //B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15) B38(12-15) B39(12-15) B38(12-15) B39(12-15) B3C(12-15) B3D(12-15) B3C(12-15) B3D(12-15)
                    const __m512i rhs_mat_2367ABEF_31_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_31, (_MM_PERM_ENUM)221); //B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15) B3A(12-15) B3B(12-15) B3A(12-15) B3B(12-15) B3E(12-15) B3F(12-15) B3E(12-15) B3F(12-15)

                    const __m512i rhs_mat_014589CD_40_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_40, (_MM_PERM_ENUM)221); //B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7) B48(4-7) B49(4-7) B48(4-7) B49(4-7) B4C(4-7) B4D(4-7) B4C(4-7) B4D(4-7)
                    const __m512i rhs_mat_2367ABEF_40_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_40, (_MM_PERM_ENUM)221); //B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7) B4A(4-7) B4B(4-7) B4A(4-7) B4B(4-7) B4E(4-7) B4F(4-7) B4E(4-7) B4F(4-7)

                    const __m512i rhs_mat_014589CD_41_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_41, (_MM_PERM_ENUM)221); //B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15) B48(12-15) B49(12-15) B48(12-15) B49(12-15) B4C(12-15) B4D(12-15) B4C(12-15) B4D(12-15)
                    const __m512i rhs_mat_2367ABEF_41_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_41, (_MM_PERM_ENUM)221); //B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15) B4A(12-15) B4B(12-15) B4A(12-15) B4B(12-15) B4E(12-15) B4F(12-15) B4E(12-15) B4F(12-15)

                    const __m512i rhs_mat_014589CD_50_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_50, (_MM_PERM_ENUM)221); //B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7) B58(4-7) B59(4-7) B58(4-7) B59(4-7) B5C(4-7) B5D(4-7) B5C(4-7) B5D(4-7)
                    const __m512i rhs_mat_2367ABEF_50_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_50, (_MM_PERM_ENUM)221); //B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7) B5A(4-7) B5B(4-7) B5A(4-7) B5B(4-7) B5E(4-7) B5F(4-7) B5E(4-7) B5F(4-7)

                    const __m512i rhs_mat_014589CD_51_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_51, (_MM_PERM_ENUM)221); //B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15) B58(12-15) B59(12-15) B58(12-15) B59(12-15) B5C(12-15) B5D(12-15) B5C(12-15) B5D(12-15)
                    const __m512i rhs_mat_2367ABEF_51_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_51, (_MM_PERM_ENUM)221); //B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15) B5A(12-15) B5B(12-15) B5A(12-15) B5B(12-15) B5E(12-15) B5F(12-15) B5E(12-15) B5F(12-15)

                    const __m512i rhs_mat_014589CD_60_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_60, (_MM_PERM_ENUM)221); //B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7) B68(4-7) B69(4-7) B68(4-7) B69(4-7) B6C(4-7) B6D(4-7) B6C(4-7) B6D(4-7)
                    const __m512i rhs_mat_2367ABEF_60_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_60, (_MM_PERM_ENUM)221); //B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7) B6A(4-7) B6B(4-7) B6A(4-7) B6B(4-7) B6E(4-7) B6F(4-7) B6E(4-7) B6F(4-7)

                    const __m512i rhs_mat_014589CD_61_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_61, (_MM_PERM_ENUM)221); //B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15) B68(12-15) B69(12-15) B68(12-15) B69(12-15) B6C(12-15) B6D(12-15) B6C(12-15) B6D(12-15)
                    const __m512i rhs_mat_2367ABEF_61_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_61, (_MM_PERM_ENUM)221); //B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15) B6A(12-15) B6B(12-15) B6A(12-15) B6B(12-15) B6E(12-15) B6F(12-15) B6E(12-15) B6F(12-15)

                    const __m512i rhs_mat_014589CD_70_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_70, (_MM_PERM_ENUM)221); //B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7) B78(4-7) B79(4-7) B78(4-7) B79(4-7) B7C(4-7) B7D(4-7) B7C(4-7) B7D(4-7)
                    const __m512i rhs_mat_2367ABEF_70_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_70, (_MM_PERM_ENUM)221); //B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7) B7A(4-7) B7B(4-7) B7A(4-7) B7B(4-7) B7E(4-7) B7F(4-7) B7E(4-7) B7F(4-7)

                    const __m512i rhs_mat_014589CD_71_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_71, (_MM_PERM_ENUM)221); //B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15) B78(12-15) B79(12-15) B78(12-15) B79(12-15) B7C(12-15) B7D(12-15) B7C(12-15) B7D(12-15)
                    const __m512i rhs_mat_2367ABEF_71_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_71, (_MM_PERM_ENUM)221); //B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15) B7A(12-15) B7B(12-15) B7A(12-15) B7B(12-15) B7E(12-15) B7F(12-15) B7E(12-15) B7F(12-15)

                    //notation:superblock subblock
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    const __m128i mins_and_scales_01_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + sb * 64));
                    const __m128i mins_and_scales_23_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 48 + sb * 64));

                    const __m128i mins_and_scales_01_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + sb * 64));
                    const __m128i mins_and_scales_23_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 48 + sb * 64));

                    // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
                    const __m256i mins_and_scales_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_01_0), mins_and_scales_01_1, 1);
                    const __m256i mins_and_scales_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_23_0), mins_and_scales_23_1, 1);
                    const __m256i mins_and_scales_45 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_45_0), mins_and_scales_45_1, 1);
                    const __m256i mins_and_scales_67 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_67_0), mins_and_scales_67_1, 1);

                    // Extract scales which is lower half from mins_and_scales
                    const __m256i scales_01 = _mm256_and_si256(mins_and_scales_01, m4b);
                    const __m256i scales_23 = _mm256_and_si256(mins_and_scales_23, m4b);
                    const __m256i scales_45 = _mm256_and_si256(mins_and_scales_45, m4b);
                    const __m256i scales_67 = _mm256_and_si256(mins_and_scales_67, m4b);

                    // Extract mins which is upper half from mins_and_scales
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_01, 4), m4b));
                    const __m512i mins_23 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_23, 4), m4b));
                    const __m512i mins_45 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_45, 4), m4b));
                    const __m512i mins_67 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_67, 4), m4b));

                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_01, scalesmask1));
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_01, scalesmask2));
                    const __m512i scales_2 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_23, scalesmask1));
                    const __m512i scales_3 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_23, scalesmask2));
                    const __m512i scales_4 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_45, scalesmask1));
                    const __m512i scales_5 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_45, scalesmask2));
                    const __m512i scales_6 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_67, scalesmask1));
                    const __m512i scales_7 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_67, scalesmask2));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_2 = _mm512_shuffle_epi32(scales_2, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_2 = _mm512_shuffle_epi32(scales_2, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_3 = _mm512_shuffle_epi32(scales_3, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_3 = _mm512_shuffle_epi32(scales_3, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_4 = _mm512_shuffle_epi32(scales_4, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_4 = _mm512_shuffle_epi32(scales_4, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_5 = _mm512_shuffle_epi32(scales_5, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_5 = _mm512_shuffle_epi32(scales_5, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_6 = _mm512_shuffle_epi32(scales_6, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_6 = _mm512_shuffle_epi32(scales_6, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_7 = _mm512_shuffle_epi32(scales_7, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_7 = _mm512_shuffle_epi32(scales_7, (_MM_PERM_ENUM)238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 512 * sb)));
                    __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                    __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                    __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                    __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                    __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                    __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                    __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                    __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                    __m256i lhs_mat_ymm_0123_20 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_20 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_20, lhs_mat_ymm_0123_20, 0);
                    __m256i lhs_mat_ymm_23_20 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_20, lhs_mat_ymm_0123_20, 17);
                    __m256i lhs_mat_ymm_0123_21 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_21 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_21, lhs_mat_ymm_0123_21, 0);
                    __m256i lhs_mat_ymm_23_21 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_21, lhs_mat_ymm_0123_21, 17);
                    __m256i lhs_mat_ymm_0123_30 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_30 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_30, lhs_mat_ymm_0123_30, 0);
                    __m256i lhs_mat_ymm_23_30 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_30, lhs_mat_ymm_0123_30, 17);
                    __m256i lhs_mat_ymm_0123_31 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_31 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_31, lhs_mat_ymm_0123_31, 0);
                    __m256i lhs_mat_ymm_23_31 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_31, lhs_mat_ymm_0123_31, 17);

                    __m256i lhs_mat_ymm_0123_40 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_40 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_40, lhs_mat_ymm_0123_40, 0);
                    __m256i lhs_mat_ymm_23_40 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_40, lhs_mat_ymm_0123_40, 17);
                    __m256i lhs_mat_ymm_0123_41 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 288 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_41 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_41, lhs_mat_ymm_0123_41, 0);
                    __m256i lhs_mat_ymm_23_41 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_41, lhs_mat_ymm_0123_41, 17);
                    __m256i lhs_mat_ymm_0123_50 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 320 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_50 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_50, lhs_mat_ymm_0123_50, 0);
                    __m256i lhs_mat_ymm_23_50 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_50, lhs_mat_ymm_0123_50, 17);
                    __m256i lhs_mat_ymm_0123_51 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 352 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_51 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_51, lhs_mat_ymm_0123_51, 0);
                    __m256i lhs_mat_ymm_23_51 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_51, lhs_mat_ymm_0123_51, 17);
                    __m256i lhs_mat_ymm_0123_60 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 384 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_60 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_60, lhs_mat_ymm_0123_60, 0);
                    __m256i lhs_mat_ymm_23_60 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_60, lhs_mat_ymm_0123_60, 17);
                    __m256i lhs_mat_ymm_0123_61 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 416 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_61 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_61, lhs_mat_ymm_0123_61, 0);
                    __m256i lhs_mat_ymm_23_61 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_61, lhs_mat_ymm_0123_61, 17);
                    __m256i lhs_mat_ymm_0123_70 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 448 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_70 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_70, lhs_mat_ymm_0123_70, 0);
                    __m256i lhs_mat_ymm_23_70 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_70, lhs_mat_ymm_0123_70, 17);
                    __m256i lhs_mat_ymm_0123_71 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 480 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_71 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_71, lhs_mat_ymm_0123_71, 0);
                    __m256i lhs_mat_ymm_23_71 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_71, lhs_mat_ymm_0123_71, 17);

                    __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                    __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                    __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                    __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);

                    __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                    __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                    __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                    __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);

                    __m512i lhs_mat_01_20 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_20), lhs_mat_ymm_01_20, 1);
                    __m512i lhs_mat_23_20 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_20), lhs_mat_ymm_23_20, 1);
                    __m512i lhs_mat_01_21 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_21), lhs_mat_ymm_01_21, 1);
                    __m512i lhs_mat_23_21 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_21), lhs_mat_ymm_23_21, 1);

                    __m512i lhs_mat_01_30 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_30), lhs_mat_ymm_01_30, 1);
                    __m512i lhs_mat_23_30 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_30), lhs_mat_ymm_23_30, 1);
                    __m512i lhs_mat_01_31 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_31), lhs_mat_ymm_01_31, 1);
                    __m512i lhs_mat_23_31 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_31), lhs_mat_ymm_23_31, 1);

                    __m512i lhs_mat_01_40 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_40), lhs_mat_ymm_01_40, 1);
                    __m512i lhs_mat_23_40 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_40), lhs_mat_ymm_23_40, 1);
                    __m512i lhs_mat_01_41 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_41), lhs_mat_ymm_01_41, 1);
                    __m512i lhs_mat_23_41 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_41), lhs_mat_ymm_23_41, 1);

                    __m512i lhs_mat_01_50 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_50), lhs_mat_ymm_01_50, 1);
                    __m512i lhs_mat_23_50 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_50), lhs_mat_ymm_23_50, 1);
                    __m512i lhs_mat_01_51 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_51), lhs_mat_ymm_01_51, 1);
                    __m512i lhs_mat_23_51 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_51), lhs_mat_ymm_23_51, 1);

                    __m512i lhs_mat_01_60 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_60), lhs_mat_ymm_01_60, 1);
                    __m512i lhs_mat_23_60 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_60), lhs_mat_ymm_23_60, 1);
                    __m512i lhs_mat_01_61 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_61), lhs_mat_ymm_01_61, 1);
                    __m512i lhs_mat_23_61 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_61), lhs_mat_ymm_23_61, 1);

                    __m512i lhs_mat_01_70 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_70), lhs_mat_ymm_01_70, 1);
                    __m512i lhs_mat_23_70 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_70), lhs_mat_ymm_23_70, 1);
                    __m512i lhs_mat_01_71 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_71), lhs_mat_ymm_01_71, 1);
                    __m512i lhs_mat_23_71 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_71), lhs_mat_ymm_23_71, 1);

                    // Bsums are loaded for the different Q8_K blocks
                    __m128i lhs_raw_bsums_01_0123 = _mm_loadu_si128((const __m128i *)((a_ptr[b].bsums + 32 * sb)));
                    __m128i lhs_raw_bsums_23_0123 = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + 8 + 32 * sb));
                    __m128i lhs_raw_bsums_01_4567 = _mm_loadu_si128((const __m128i *)((a_ptr[b].bsums + 16 + 32 * sb)));
                    __m128i lhs_raw_bsums_23_4567 = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + 24 + 32 * sb));

                    __m256i lhs_bsums_ymm_01_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_0123), lhs_raw_bsums_01_0123, 1);
                    __m512i lhs_bsums_01_0123 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_01_0123), lhs_bsums_ymm_01_0123, 1);
                    __m256i lhs_bsums_ymm_23_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_0123), lhs_raw_bsums_23_0123, 1);
                    __m512i lhs_bsums_23_0123 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_23_0123), lhs_bsums_ymm_23_0123, 1);
                    __m256i lhs_bsums_ymm_01_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_4567), lhs_raw_bsums_01_4567, 1);
                    __m512i lhs_bsums_01_4567 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_01_4567), lhs_bsums_ymm_01_4567, 1);
                    __m256i lhs_bsums_ymm_23_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_4567), lhs_raw_bsums_23_4567, 1);
                    __m512i lhs_bsums_23_4567 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_23_4567), lhs_bsums_ymm_23_4567, 1);

                    // Shuffle pattern one - left side input
                    const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)

                    const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)

                    const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)

                    const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)

                    const __m512i lhs_mat_01_20_sp1 = _mm512_shuffle_epi32(lhs_mat_01_20, (_MM_PERM_ENUM)160); //A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
                    const __m512i lhs_mat_23_20_sp1 = _mm512_shuffle_epi32(lhs_mat_23_20, (_MM_PERM_ENUM)160); //A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3)

                    const __m512i lhs_mat_01_21_sp1 = _mm512_shuffle_epi32(lhs_mat_01_21, (_MM_PERM_ENUM)160); //A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                    const __m512i lhs_mat_23_21_sp1 = _mm512_shuffle_epi32(lhs_mat_23_21, (_MM_PERM_ENUM)160); //A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11)

                    const __m512i lhs_mat_01_30_sp1 = _mm512_shuffle_epi32(lhs_mat_01_30, (_MM_PERM_ENUM)160); //A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
                    const __m512i lhs_mat_23_30_sp1 = _mm512_shuffle_epi32(lhs_mat_23_30, (_MM_PERM_ENUM)160); //A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3)

                    const __m512i lhs_mat_01_31_sp1 = _mm512_shuffle_epi32(lhs_mat_01_31, (_MM_PERM_ENUM)160); //A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                    const __m512i lhs_mat_23_31_sp1 = _mm512_shuffle_epi32(lhs_mat_23_31, (_MM_PERM_ENUM)160); //A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11)

                    const __m512i lhs_mat_01_40_sp1 = _mm512_shuffle_epi32(lhs_mat_01_40, (_MM_PERM_ENUM)160); //A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
                    const __m512i lhs_mat_23_40_sp1 = _mm512_shuffle_epi32(lhs_mat_23_40, (_MM_PERM_ENUM)160); //A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3)

                    const __m512i lhs_mat_01_41_sp1 = _mm512_shuffle_epi32(lhs_mat_01_41, (_MM_PERM_ENUM)160); //A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                    const __m512i lhs_mat_23_41_sp1 = _mm512_shuffle_epi32(lhs_mat_23_41, (_MM_PERM_ENUM)160); //A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11)

                    const __m512i lhs_mat_01_50_sp1 = _mm512_shuffle_epi32(lhs_mat_01_50, (_MM_PERM_ENUM)160); //A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
                    const __m512i lhs_mat_23_50_sp1 = _mm512_shuffle_epi32(lhs_mat_23_50, (_MM_PERM_ENUM)160); //A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3)

                    const __m512i lhs_mat_01_51_sp1 = _mm512_shuffle_epi32(lhs_mat_01_51, (_MM_PERM_ENUM)160); //A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                    const __m512i lhs_mat_23_51_sp1 = _mm512_shuffle_epi32(lhs_mat_23_51, (_MM_PERM_ENUM)160); //A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11)

                    const __m512i lhs_mat_01_60_sp1 = _mm512_shuffle_epi32(lhs_mat_01_60, (_MM_PERM_ENUM)160); //A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
                    const __m512i lhs_mat_23_60_sp1 = _mm512_shuffle_epi32(lhs_mat_23_60, (_MM_PERM_ENUM)160); //A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3)

                    const __m512i lhs_mat_01_61_sp1 = _mm512_shuffle_epi32(lhs_mat_01_61, (_MM_PERM_ENUM)160); //A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                    const __m512i lhs_mat_23_61_sp1 = _mm512_shuffle_epi32(lhs_mat_23_61, (_MM_PERM_ENUM)160); //A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11)

                    const __m512i lhs_mat_01_70_sp1 = _mm512_shuffle_epi32(lhs_mat_01_70, (_MM_PERM_ENUM)160); //A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
                    const __m512i lhs_mat_23_70_sp1 = _mm512_shuffle_epi32(lhs_mat_23_70, (_MM_PERM_ENUM)160); //A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3)

                    const __m512i lhs_mat_01_71_sp1 = _mm512_shuffle_epi32(lhs_mat_01_71, (_MM_PERM_ENUM)160); //A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                    const __m512i lhs_mat_23_71_sp1 = _mm512_shuffle_epi32(lhs_mat_23_71, (_MM_PERM_ENUM)160); //A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11)

                    const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)

                    const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)

                    const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)

                    const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)

                    const __m512i lhs_mat_01_20_sp2 = _mm512_shuffle_epi32(lhs_mat_01_20, (_MM_PERM_ENUM)245); //A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
                    const __m512i lhs_mat_23_20_sp2 = _mm512_shuffle_epi32(lhs_mat_23_20, (_MM_PERM_ENUM)245); //A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7)

                    const __m512i lhs_mat_01_21_sp2 = _mm512_shuffle_epi32(lhs_mat_01_21, (_MM_PERM_ENUM)245); //A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                    const __m512i lhs_mat_23_21_sp2 = _mm512_shuffle_epi32(lhs_mat_23_21, (_MM_PERM_ENUM)245); //A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15)

                    const __m512i lhs_mat_01_30_sp2 = _mm512_shuffle_epi32(lhs_mat_01_30, (_MM_PERM_ENUM)245); //A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
                    const __m512i lhs_mat_23_30_sp2 = _mm512_shuffle_epi32(lhs_mat_23_30, (_MM_PERM_ENUM)245); //A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7)

                    const __m512i lhs_mat_01_31_sp2 = _mm512_shuffle_epi32(lhs_mat_01_31, (_MM_PERM_ENUM)245); //A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                    const __m512i lhs_mat_23_31_sp2 = _mm512_shuffle_epi32(lhs_mat_23_31, (_MM_PERM_ENUM)245); //A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15)

                    const __m512i lhs_mat_01_40_sp2 = _mm512_shuffle_epi32(lhs_mat_01_40, (_MM_PERM_ENUM)245); //A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
                    const __m512i lhs_mat_23_40_sp2 = _mm512_shuffle_epi32(lhs_mat_23_40, (_MM_PERM_ENUM)245); //A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7)

                    const __m512i lhs_mat_01_41_sp2 = _mm512_shuffle_epi32(lhs_mat_01_41, (_MM_PERM_ENUM)245); //A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                    const __m512i lhs_mat_23_41_sp2 = _mm512_shuffle_epi32(lhs_mat_23_41, (_MM_PERM_ENUM)245); //A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15)

                    const __m512i lhs_mat_01_50_sp2 = _mm512_shuffle_epi32(lhs_mat_01_50, (_MM_PERM_ENUM)245); //A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
                    const __m512i lhs_mat_23_50_sp2 = _mm512_shuffle_epi32(lhs_mat_23_50, (_MM_PERM_ENUM)245); //A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7)

                    const __m512i lhs_mat_01_51_sp2 = _mm512_shuffle_epi32(lhs_mat_01_51, (_MM_PERM_ENUM)245); //A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                    const __m512i lhs_mat_23_51_sp2 = _mm512_shuffle_epi32(lhs_mat_23_51, (_MM_PERM_ENUM)245); //A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15)

                    const __m512i lhs_mat_01_60_sp2 = _mm512_shuffle_epi32(lhs_mat_01_60, (_MM_PERM_ENUM)245); //A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
                    const __m512i lhs_mat_23_60_sp2 = _mm512_shuffle_epi32(lhs_mat_23_60, (_MM_PERM_ENUM)245); //A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7)

                    const __m512i lhs_mat_01_61_sp2 = _mm512_shuffle_epi32(lhs_mat_01_61, (_MM_PERM_ENUM)245); //A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                    const __m512i lhs_mat_23_61_sp2 = _mm512_shuffle_epi32(lhs_mat_23_61, (_MM_PERM_ENUM)245); //A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15)

                    const __m512i lhs_mat_01_70_sp2 = _mm512_shuffle_epi32(lhs_mat_01_70, (_MM_PERM_ENUM)245); //A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
                    const __m512i lhs_mat_23_70_sp2 = _mm512_shuffle_epi32(lhs_mat_23_70, (_MM_PERM_ENUM)245); //A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7)

                    const __m512i lhs_mat_01_71_sp2 = _mm512_shuffle_epi32(lhs_mat_01_71, (_MM_PERM_ENUM)245); //A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                    const __m512i lhs_mat_23_71_sp2 = _mm512_shuffle_epi32(lhs_mat_23_71, (_MM_PERM_ENUM)245); //A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1));
                    __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1));

                    __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1));
                    __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1));

                    __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1));
                    __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1));

                    __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1));
                    __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1));

                    __m512i iacc_mat_00_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp1, lhs_mat_01_20_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp1, lhs_mat_01_21_sp1));
                    __m512i iacc_mat_01_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp1, lhs_mat_01_20_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp1, lhs_mat_01_21_sp1));

                    __m512i iacc_mat_10_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp1, lhs_mat_23_20_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp1, lhs_mat_23_21_sp1));
                    __m512i iacc_mat_11_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp1, lhs_mat_23_20_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp1, lhs_mat_23_21_sp1));

                    __m512i iacc_mat_00_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp1, lhs_mat_01_30_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp1, lhs_mat_01_31_sp1));
                    __m512i iacc_mat_01_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp1, lhs_mat_01_30_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp1, lhs_mat_01_31_sp1));

                    __m512i iacc_mat_10_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp1, lhs_mat_23_30_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp1, lhs_mat_23_31_sp1));
                    __m512i iacc_mat_11_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp1, lhs_mat_23_30_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp1, lhs_mat_23_31_sp1));

                    __m512i iacc_mat_00_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp1, lhs_mat_01_40_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp1, lhs_mat_01_41_sp1));
                    __m512i iacc_mat_01_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp1, lhs_mat_01_40_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp1, lhs_mat_01_41_sp1));

                    __m512i iacc_mat_10_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp1, lhs_mat_23_40_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp1, lhs_mat_23_41_sp1));
                    __m512i iacc_mat_11_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp1, lhs_mat_23_40_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp1, lhs_mat_23_41_sp1));

                    __m512i iacc_mat_00_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp1, lhs_mat_01_50_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp1, lhs_mat_01_51_sp1));
                    __m512i iacc_mat_01_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp1, lhs_mat_01_50_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp1, lhs_mat_01_51_sp1));

                    __m512i iacc_mat_10_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp1, lhs_mat_23_50_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp1, lhs_mat_23_51_sp1));
                    __m512i iacc_mat_11_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp1, lhs_mat_23_50_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp1, lhs_mat_23_51_sp1));

                    __m512i iacc_mat_00_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp1, lhs_mat_01_60_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp1, lhs_mat_01_61_sp1));
                    __m512i iacc_mat_01_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp1, lhs_mat_01_60_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp1, lhs_mat_01_61_sp1));

                    __m512i iacc_mat_10_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp1, lhs_mat_23_60_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp1, lhs_mat_23_61_sp1));
                    __m512i iacc_mat_11_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp1, lhs_mat_23_60_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp1, lhs_mat_23_61_sp1));

                    __m512i iacc_mat_00_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp1, lhs_mat_01_70_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp1, lhs_mat_01_71_sp1));
                    __m512i iacc_mat_01_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp1, lhs_mat_01_70_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp1, lhs_mat_01_71_sp1));

                    __m512i iacc_mat_10_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp1, lhs_mat_23_70_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp1, lhs_mat_23_71_sp1));
                    __m512i iacc_mat_11_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp1, lhs_mat_23_70_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp1, lhs_mat_23_71_sp1));


                    __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2));
                    __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2));

                    __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2));
                    __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2));

                    __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2));
                    __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2));

                    __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2));
                    __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2));

                    __m512i iacc_mat_00_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp2, lhs_mat_01_20_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp2, lhs_mat_01_21_sp2));
                    __m512i iacc_mat_01_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp2, lhs_mat_01_20_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp2, lhs_mat_01_21_sp2));

                    __m512i iacc_mat_10_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp2, lhs_mat_23_20_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp2, lhs_mat_23_21_sp2));
                    __m512i iacc_mat_11_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp2, lhs_mat_23_20_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp2, lhs_mat_23_21_sp2));

                    __m512i iacc_mat_00_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp2, lhs_mat_01_30_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp2, lhs_mat_01_31_sp2));
                    __m512i iacc_mat_01_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp2, lhs_mat_01_30_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp2, lhs_mat_01_31_sp2));

                    __m512i iacc_mat_10_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp2, lhs_mat_23_30_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp2, lhs_mat_23_31_sp2));
                    __m512i iacc_mat_11_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp2, lhs_mat_23_30_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp2, lhs_mat_23_31_sp2));

                    __m512i iacc_mat_00_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp2, lhs_mat_01_40_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp2, lhs_mat_01_41_sp2));
                    __m512i iacc_mat_01_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp2, lhs_mat_01_40_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp2, lhs_mat_01_41_sp2));

                    __m512i iacc_mat_10_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp2, lhs_mat_23_40_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp2, lhs_mat_23_41_sp2));
                    __m512i iacc_mat_11_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp2, lhs_mat_23_40_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp2, lhs_mat_23_41_sp2));

                    __m512i iacc_mat_00_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp2, lhs_mat_01_50_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp2, lhs_mat_01_51_sp2));
                    __m512i iacc_mat_01_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp2, lhs_mat_01_50_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp2, lhs_mat_01_51_sp2));

                    __m512i iacc_mat_10_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp2, lhs_mat_23_50_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp2, lhs_mat_23_51_sp2));
                    __m512i iacc_mat_11_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp2, lhs_mat_23_50_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp2, lhs_mat_23_51_sp2));

                    __m512i iacc_mat_00_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp2, lhs_mat_01_60_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp2, lhs_mat_01_61_sp2));
                    __m512i iacc_mat_01_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp2, lhs_mat_01_60_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp2, lhs_mat_01_61_sp2));

                    __m512i iacc_mat_10_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp2, lhs_mat_23_60_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp2, lhs_mat_23_61_sp2));
                    __m512i iacc_mat_11_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp2, lhs_mat_23_60_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp2, lhs_mat_23_61_sp2));

                    __m512i iacc_mat_00_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp2, lhs_mat_01_70_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp2, lhs_mat_01_71_sp2));
                    __m512i iacc_mat_01_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp2, lhs_mat_01_70_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp2, lhs_mat_01_71_sp2));

                    __m512i iacc_mat_10_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp2, lhs_mat_23_70_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp2, lhs_mat_23_71_sp2));
                    __m512i iacc_mat_11_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp2, lhs_mat_23_70_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp2, lhs_mat_23_71_sp2));

                    // Combine results from both shuffle patterns for each output block
                    __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    __m512i iacc_mat_00_2 = _mm512_add_epi16(iacc_mat_00_2_sp1, iacc_mat_00_2_sp2);
                    __m512i iacc_mat_01_2 = _mm512_add_epi16(iacc_mat_01_2_sp1, iacc_mat_01_2_sp2);
                    __m512i iacc_mat_10_2 = _mm512_add_epi16(iacc_mat_10_2_sp1, iacc_mat_10_2_sp2);
                    __m512i iacc_mat_11_2 = _mm512_add_epi16(iacc_mat_11_2_sp1, iacc_mat_11_2_sp2);

                    __m512i iacc_mat_00_3 = _mm512_add_epi16(iacc_mat_00_3_sp1, iacc_mat_00_3_sp2);
                    __m512i iacc_mat_01_3 = _mm512_add_epi16(iacc_mat_01_3_sp1, iacc_mat_01_3_sp2);
                    __m512i iacc_mat_10_3 = _mm512_add_epi16(iacc_mat_10_3_sp1, iacc_mat_10_3_sp2);
                    __m512i iacc_mat_11_3 = _mm512_add_epi16(iacc_mat_11_3_sp1, iacc_mat_11_3_sp2);

                    __m512i iacc_mat_00_4 = _mm512_add_epi16(iacc_mat_00_4_sp1, iacc_mat_00_4_sp2);
                    __m512i iacc_mat_01_4 = _mm512_add_epi16(iacc_mat_01_4_sp1, iacc_mat_01_4_sp2);
                    __m512i iacc_mat_10_4 = _mm512_add_epi16(iacc_mat_10_4_sp1, iacc_mat_10_4_sp2);
                    __m512i iacc_mat_11_4 = _mm512_add_epi16(iacc_mat_11_4_sp1, iacc_mat_11_4_sp2);

                    __m512i iacc_mat_00_5 = _mm512_add_epi16(iacc_mat_00_5_sp1, iacc_mat_00_5_sp2);
                    __m512i iacc_mat_01_5 = _mm512_add_epi16(iacc_mat_01_5_sp1, iacc_mat_01_5_sp2);
                    __m512i iacc_mat_10_5 = _mm512_add_epi16(iacc_mat_10_5_sp1, iacc_mat_10_5_sp2);
                    __m512i iacc_mat_11_5 = _mm512_add_epi16(iacc_mat_11_5_sp1, iacc_mat_11_5_sp2);

                    __m512i iacc_mat_00_6 = _mm512_add_epi16(iacc_mat_00_6_sp1, iacc_mat_00_6_sp2);
                    __m512i iacc_mat_01_6 = _mm512_add_epi16(iacc_mat_01_6_sp1, iacc_mat_01_6_sp2);
                    __m512i iacc_mat_10_6 = _mm512_add_epi16(iacc_mat_10_6_sp1, iacc_mat_10_6_sp2);
                    __m512i iacc_mat_11_6 = _mm512_add_epi16(iacc_mat_11_6_sp1, iacc_mat_11_6_sp2);

                    __m512i iacc_mat_00_7 = _mm512_add_epi16(iacc_mat_00_7_sp1, iacc_mat_00_7_sp2);
                    __m512i iacc_mat_01_7 = _mm512_add_epi16(iacc_mat_01_7_sp1, iacc_mat_01_7_sp2);
                    __m512i iacc_mat_10_7 = _mm512_add_epi16(iacc_mat_10_7_sp1, iacc_mat_10_7_sp2);
                    __m512i iacc_mat_11_7 = _mm512_add_epi16(iacc_mat_11_7_sp1, iacc_mat_11_7_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                    iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                    iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                    iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                    iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                    iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                    iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                    iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                    iacc_mat_00_2 = _mm512_madd_epi16(iacc_mat_00_2, scale_014589CD_2);
                    iacc_mat_01_2 = _mm512_madd_epi16(iacc_mat_01_2, scale_2367ABEF_2);
                    iacc_mat_10_2 = _mm512_madd_epi16(iacc_mat_10_2, scale_014589CD_2);
                    iacc_mat_11_2 = _mm512_madd_epi16(iacc_mat_11_2, scale_2367ABEF_2);

                    iacc_mat_00_3 = _mm512_madd_epi16(iacc_mat_00_3, scale_014589CD_3);
                    iacc_mat_01_3 = _mm512_madd_epi16(iacc_mat_01_3, scale_2367ABEF_3);
                    iacc_mat_10_3 = _mm512_madd_epi16(iacc_mat_10_3, scale_014589CD_3);
                    iacc_mat_11_3 = _mm512_madd_epi16(iacc_mat_11_3, scale_2367ABEF_3);

                    iacc_mat_00_4 = _mm512_madd_epi16(iacc_mat_00_4, scale_014589CD_4);
                    iacc_mat_01_4 = _mm512_madd_epi16(iacc_mat_01_4, scale_2367ABEF_4);
                    iacc_mat_10_4 = _mm512_madd_epi16(iacc_mat_10_4, scale_014589CD_4);
                    iacc_mat_11_4 = _mm512_madd_epi16(iacc_mat_11_4, scale_2367ABEF_4);

                    iacc_mat_00_5 = _mm512_madd_epi16(iacc_mat_00_5, scale_014589CD_5);
                    iacc_mat_01_5 = _mm512_madd_epi16(iacc_mat_01_5, scale_2367ABEF_5);
                    iacc_mat_10_5 = _mm512_madd_epi16(iacc_mat_10_5, scale_014589CD_5);
                    iacc_mat_11_5 = _mm512_madd_epi16(iacc_mat_11_5, scale_2367ABEF_5);

                    iacc_mat_00_6 = _mm512_madd_epi16(iacc_mat_00_6, scale_014589CD_6);
                    iacc_mat_01_6 = _mm512_madd_epi16(iacc_mat_01_6, scale_2367ABEF_6);
                    iacc_mat_10_6 = _mm512_madd_epi16(iacc_mat_10_6, scale_014589CD_6);
                    iacc_mat_11_6 = _mm512_madd_epi16(iacc_mat_11_6, scale_2367ABEF_6);

                    iacc_mat_00_7 = _mm512_madd_epi16(iacc_mat_00_7, scale_014589CD_7);
                    iacc_mat_01_7 = _mm512_madd_epi16(iacc_mat_01_7, scale_2367ABEF_7);
                    iacc_mat_10_7 = _mm512_madd_epi16(iacc_mat_10_7, scale_014589CD_7);
                    iacc_mat_11_7 = _mm512_madd_epi16(iacc_mat_11_7, scale_2367ABEF_7);

                    __m512i iacc_mat_00 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_00_0, iacc_mat_00_1), _mm512_add_epi32(iacc_mat_00_2, iacc_mat_00_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_00_4, iacc_mat_00_5), _mm512_add_epi32(iacc_mat_00_6, iacc_mat_00_7)));
                    __m512i iacc_mat_01 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_01_0, iacc_mat_01_1), _mm512_add_epi32(iacc_mat_01_2, iacc_mat_01_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_01_4, iacc_mat_01_5), _mm512_add_epi32(iacc_mat_01_6, iacc_mat_01_7)));
                    __m512i iacc_mat_10 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_10_0, iacc_mat_10_1), _mm512_add_epi32(iacc_mat_10_2, iacc_mat_10_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_10_4, iacc_mat_10_5), _mm512_add_epi32(iacc_mat_10_6, iacc_mat_10_7)));
                    __m512i iacc_mat_11 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_11_0, iacc_mat_11_1), _mm512_add_epi32(iacc_mat_11_2, iacc_mat_11_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_11_4, iacc_mat_11_5), _mm512_add_epi32(iacc_mat_11_6, iacc_mat_11_7)));

                    // Straighten out to make 4 row vectors
                    __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                    __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                    const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                    // Multiply with appropiate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
                    __m512i iacc_row_min_0_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)0), mins_01);
                    __m512i iacc_row_min_1_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)170), mins_01);
                    __m512i iacc_row_min_2_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)0), mins_01);
                    __m512i iacc_row_min_3_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)170), mins_01);

                    __m512i iacc_row_min_0_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)85), mins_23);
                    __m512i iacc_row_min_1_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)255), mins_23);
                    __m512i iacc_row_min_2_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)85), mins_23);
                    __m512i iacc_row_min_3_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)255), mins_23);

                    __m512i iacc_row_min_0_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)0), mins_45);
                    __m512i iacc_row_min_1_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)170), mins_45);
                    __m512i iacc_row_min_2_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)0), mins_45);
                    __m512i iacc_row_min_3_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)170), mins_45);

                    __m512i iacc_row_min_0_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)85), mins_67);
                    __m512i iacc_row_min_1_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)255), mins_67);
                    __m512i iacc_row_min_2_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)85), mins_67);
                    __m512i iacc_row_min_3_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)255), mins_67);

                    __m512i iacc_row_min_0 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_0_01, iacc_row_min_0_23), _mm512_add_epi32(iacc_row_min_0_45,iacc_row_min_0_67));
                    __m512i iacc_row_min_1 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_1_01, iacc_row_min_1_23), _mm512_add_epi32(iacc_row_min_1_45,iacc_row_min_1_67));
                    __m512i iacc_row_min_2 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_2_01, iacc_row_min_2_23), _mm512_add_epi32(iacc_row_min_2_45,iacc_row_min_2_67));
                    __m512i iacc_row_min_3 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_3_01, iacc_row_min_3_23), _mm512_add_epi32(iacc_row_min_3_45,iacc_row_min_3_67));

                    acc_min_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }
            // Store accumulated values
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

    if (anc != nc) {
        xstart = anc/8;
        y = 0;
    }

#endif // __AVX512BW__ && __AVX512DQ__

    // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q2_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Delta values - Load the eight scale values of block_q2_kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q2_kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_K for eight sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    //superblock    sub block   which part of sub block
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);

                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);

                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 2-bit -> 8-bit
                    // First sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m3b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m3b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m3b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m3b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    // Second sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(rhs_raw_mat_0145_2, m3b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(rhs_raw_mat_2367_2, m3b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(rhs_raw_mat_0145_3, m3b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(rhs_raw_mat_2367_3, m3b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    // Third sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 2), m3b); //B20(0-7) B21(0-7) B24(0-7) B25(0-7)
                    const __m256i rhs_mat_2367_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 2), m3b); //B22(0-7) B23(0-7) B26(0-7) B27(0-7)

                    const __m256i rhs_mat_0145_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 2), m3b); //B20(8-15) B21(8-15) B24(8-15) B25(8-15)
                    const __m256i rhs_mat_2367_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 2), m3b); //B22(8-15) B23(8-15) B26(8-15) B27(8-15)

                    // Fourth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 2), m3b); //B30(0-7) B31(0-7) B34(0-7) B35(0-7)
                    const __m256i rhs_mat_2367_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 2), m3b); //B32(0-7) B33(0-7) B36(0-7) B37(0-7)

                    const __m256i rhs_mat_0145_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 2), m3b); //B30(8-15) B31(8-15) B34(8-15) B35(8-15)
                    const __m256i rhs_mat_2367_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 2), m3b); //B32(8-15) B33(8-15) B36(8-15) B37(8-15)

                    // Fifth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m3b); //B40(0-7) B41(0-7) B44(0-7) B45(0-7)
                    const __m256i rhs_mat_2367_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m3b); //B42(0-7) B43(0-7) B46(0-7) B47(0-7)

                    const __m256i rhs_mat_0145_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m3b); //B40(8-15) B41(8-15) B44(8-15) B45(8-15)
                    const __m256i rhs_mat_2367_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m3b); //B42(8-15) B43(8-15) B46(8-15) B47(8-15)

                    // Sixth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m3b); //B50(0-7) B51(0-7) B54(0-7) B55(0-7)
                    const __m256i rhs_mat_2367_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m3b); //B52(0-7) B53(0-7) B56(0-7) B57(0-7)

                    const __m256i rhs_mat_0145_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m3b); //B50(8-15) B51(8-15) B54(8-15) B55(8-15)
                    const __m256i rhs_mat_2367_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m3b); //B52(8-15) B53(8-15) B56(8-15) B57(8-15)

                    // Seventh sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 6), m3b); //B60(0-7) B61(0-7) B64(0-7) B65(0-7)
                    const __m256i rhs_mat_2367_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 6), m3b); //B62(0-7) B63(0-7) B66(0-7) B67(0-7)

                    const __m256i rhs_mat_0145_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 6), m3b); //B60(8-15) B61(8-15) B64(8-15) B65(8-15)
                    const __m256i rhs_mat_2367_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 6), m3b); //B62(8-15) B63(8-15) B66(8-15) B67(8-15)

                    // Eighth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 6), m3b); //B70(0-7) B71(0-7) B74(0-7) B75(0-7)
                    const __m256i rhs_mat_2367_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 6), m3b); //B72(0-7) B73(0-7) B76(0-7) B77(0-7)

                    const __m256i rhs_mat_0145_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 6), m3b); //B70(8-15) B71(8-15) B74(8-15) B75(8-15)
                    const __m256i rhs_mat_2367_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 6), m3b); //B72(8-15) B73(8-15) B76(8-15) B77(8-15)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_20_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_20, 136); //B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3)
                    const __m256i rhs_mat_2367_20_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_20, 136); //B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3)

                    const __m256i rhs_mat_0145_21_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_21, 136); //B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11)
                    const __m256i rhs_mat_2367_21_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_21, 136); //B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11)

                    const __m256i rhs_mat_0145_30_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_30, 136); //B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3)
                    const __m256i rhs_mat_2367_30_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_30, 136); //B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3)

                    const __m256i rhs_mat_0145_31_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_31, 136); //B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11
                    const __m256i rhs_mat_2367_31_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_31, 136); //B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11)

                    const __m256i rhs_mat_0145_40_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_40, 136); //B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3)
                    const __m256i rhs_mat_2367_40_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_40, 136); //B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3)

                    const __m256i rhs_mat_0145_41_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_41, 136); //B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11)
                    const __m256i rhs_mat_2367_41_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_41, 136); //B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11)

                    const __m256i rhs_mat_0145_50_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_50, 136); //B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3)
                    const __m256i rhs_mat_2367_50_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_50, 136); //B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3)

                    const __m256i rhs_mat_0145_51_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_51, 136); //B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11)
                    const __m256i rhs_mat_2367_51_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_51, 136); //B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11)

                    const __m256i rhs_mat_0145_60_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_60, 136); //B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3)
                    const __m256i rhs_mat_2367_60_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_60, 136); //B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3)

                    const __m256i rhs_mat_0145_61_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_61, 136); //B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11)
                    const __m256i rhs_mat_2367_61_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_61, 136); //B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11)

                    const __m256i rhs_mat_0145_70_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_70, 136); //B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3)
                    const __m256i rhs_mat_2367_70_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_70, 136); //B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3)

                    const __m256i rhs_mat_0145_71_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_71, 136); //B70(8-11) B71(8-11) B70(8-11) B71(8-11) B74(8-11) B75(8-11) B74(8-11) B75(8-11)
                    const __m256i rhs_mat_2367_71_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_71, 136); //B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11)


                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_20_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_20, 221); //B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7)
                    const __m256i rhs_mat_2367_20_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_20, 221); //B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7)

                    const __m256i rhs_mat_0145_21_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_21, 221); //B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15)
                    const __m256i rhs_mat_2367_21_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_21, 221); //B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15)

                    const __m256i rhs_mat_0145_30_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_30, 221); //B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7)
                    const __m256i rhs_mat_2367_30_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_30, 221); //B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7)

                    const __m256i rhs_mat_0145_31_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_31, 221); //B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15)
                    const __m256i rhs_mat_2367_31_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_31, 221); //B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15)

                    const __m256i rhs_mat_0145_40_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_40, 221); //B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7)
                    const __m256i rhs_mat_2367_40_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_40, 221); //B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7)

                    const __m256i rhs_mat_0145_41_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_41, 221); //B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15)
                    const __m256i rhs_mat_2367_41_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_41, 221); //B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15)

                    const __m256i rhs_mat_0145_50_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_50, 221); //B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7)
                    const __m256i rhs_mat_2367_50_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_50, 221); //B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7)

                    const __m256i rhs_mat_0145_51_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_51, 221); //B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15)
                    const __m256i rhs_mat_2367_51_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_51, 221); //B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15)

                    const __m256i rhs_mat_0145_60_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_60, 221); //B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7)
                    const __m256i rhs_mat_2367_60_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_60, 221); //B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7)

                    const __m256i rhs_mat_0145_61_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_61, 221); //B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15)
                    const __m256i rhs_mat_2367_61_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_61, 221); //B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15)

                    const __m256i rhs_mat_0145_70_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_70, 221); //B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7)
                    const __m256i rhs_mat_2367_70_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_70, 221); //B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7)

                    const __m256i rhs_mat_0145_71_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_71, 221); //B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15)
                    const __m256i rhs_mat_2367_71_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_71, 221); //B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15)

                    //Scales and Mins of corresponding sub blocks from different Q2_K structures are stored together
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
                    const __m128i mins_and_scales_01 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + sb * 64));
                    const __m128i mins_and_scales_23 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 48 + sb * 64));

                    // Extract scales which is lower half from mins_and_scales
                    const __m128i scales_01 = _mm_and_si128(mins_and_scales_01, m4b_sse);
                    const __m128i scales_23 = _mm_and_si128(mins_and_scales_23, m4b_sse);
                    const __m128i scales_45 = _mm_and_si128(mins_and_scales_45, m4b_sse);
                    const __m128i scales_67 = _mm_and_si128(mins_and_scales_67, m4b_sse);

                    // Extract mins which is upper half from mins_and_scales
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_01, 4), m4b_sse));
                    const __m256i mins_23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_23, 4), m4b_sse));
                    const __m256i mins_45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_45, 4), m4b_sse));
                    const __m256i mins_67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_67, 4), m4b_sse));

                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask1_sse));
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask2_sse));

                    const __m256i scales_2 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask1_sse));
                    const __m256i scales_3 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask2_sse));

                    const __m256i scales_4 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask1_sse));
                    const __m256i scales_5 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask2_sse));

                    const __m256i scales_6 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask1_sse));
                    const __m256i scales_7 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask2_sse));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    const __m256i scale_0145_2 = _mm256_shuffle_epi32(scales_2, 68);
                    const __m256i scale_2367_2 = _mm256_shuffle_epi32(scales_2, 238);

                    const __m256i scale_0145_3 = _mm256_shuffle_epi32(scales_3, 68);
                    const __m256i scale_2367_3 = _mm256_shuffle_epi32(scales_3, 238);

                    const __m256i scale_0145_4 = _mm256_shuffle_epi32(scales_4, 68);
                    const __m256i scale_2367_4 = _mm256_shuffle_epi32(scales_4, 238);

                    const __m256i scale_0145_5 = _mm256_shuffle_epi32(scales_5, 68);
                    const __m256i scale_2367_5 = _mm256_shuffle_epi32(scales_5, 238);

                    const __m256i scale_0145_6 = _mm256_shuffle_epi32(scales_6, 68);
                    const __m256i scale_2367_6 = _mm256_shuffle_epi32(scales_6, 238);

                    const __m256i scale_0145_7 = _mm256_shuffle_epi32(scales_7, 68);
                    const __m256i scale_2367_7 = _mm256_shuffle_epi32(scales_7, 238);


                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                        __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 512 * sb)));
                        __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                        __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                        __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 512 * sb)));
                        __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                        __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                        __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 512 * sb)));
                        __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                        __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                        __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 512 * sb)));
                        __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                        __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                        __m256i lhs_mat_0123_20 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 512 * sb)));
                        __m256i lhs_mat_01_20 = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20, 0);
                        __m256i lhs_mat_23_20 = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20, 17);
                        __m256i lhs_mat_0123_21 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 512 * sb)));
                        __m256i lhs_mat_01_21 = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21, 0);
                        __m256i lhs_mat_23_21 = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21, 17);
                        __m256i lhs_mat_0123_30 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 512 * sb)));
                        __m256i lhs_mat_01_30 = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30, 0);
                        __m256i lhs_mat_23_30 = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30, 17);
                        __m256i lhs_mat_0123_31 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 512 * sb)));
                        __m256i lhs_mat_01_31 = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31, 0);
                        __m256i lhs_mat_23_31 = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31, 17);

                        __m256i lhs_mat_0123_40 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 + 512 * sb)));
                        __m256i lhs_mat_01_40 = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40, 0);
                        __m256i lhs_mat_23_40 = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40, 17);
                        __m256i lhs_mat_0123_41 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 288 + 512 * sb)));
                        __m256i lhs_mat_01_41 = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41, 0);
                        __m256i lhs_mat_23_41 = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41, 17);
                        __m256i lhs_mat_0123_50 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 320 + 512 * sb)));
                        __m256i lhs_mat_01_50 = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50, 0);
                        __m256i lhs_mat_23_50 = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50, 17);
                        __m256i lhs_mat_0123_51 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 352 + 512 * sb)));
                        __m256i lhs_mat_01_51 = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51, 0);
                        __m256i lhs_mat_23_51 = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51, 17);
                        __m256i lhs_mat_0123_60 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 384 + 512 * sb)));
                        __m256i lhs_mat_01_60 = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60, 0);
                        __m256i lhs_mat_23_60 = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60, 17);
                        __m256i lhs_mat_0123_61 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 416 + 512 * sb)));
                        __m256i lhs_mat_01_61 = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61, 0);
                        __m256i lhs_mat_23_61 = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61, 17);
                        __m256i lhs_mat_0123_70 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 448 + 512 * sb)));
                        __m256i lhs_mat_01_70 = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70, 0);
                        __m256i lhs_mat_23_70 = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70, 17);
                        __m256i lhs_mat_0123_71 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 480 + 512 * sb)));
                        __m256i lhs_mat_01_71 = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71, 0);
                        __m256i lhs_mat_23_71 = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71, 17);

                        // Bsums are loaded for the different Q8_K blocks
                        __m128i lhs_raw_bsums_01_0123 = _mm_loadu_si128((const __m128i *)((a_ptrs[rp][b].bsums + 32 * sb)));
                        __m128i lhs_raw_bsums_23_0123 = _mm_loadu_si128((const __m128i *)(a_ptrs[rp][b].bsums + 8 + 32 * sb));
                        __m128i lhs_raw_bsums_01_4567 = _mm_loadu_si128((const __m128i *)((a_ptrs[rp][b].bsums + 16 + 32 * sb)));
                        __m128i lhs_raw_bsums_23_4567 = _mm_loadu_si128((const __m128i *)(a_ptrs[rp][b].bsums + 24 + 32 * sb));

                        // Shuffle pattern one - left side input
                        const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                        const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160); //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                        const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                        const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160); //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                        const __m256i lhs_mat_01_20_sp1 = _mm256_shuffle_epi32(lhs_mat_01_20, 160); //A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
                        const __m256i lhs_mat_23_20_sp1 = _mm256_shuffle_epi32(lhs_mat_23_20, 160); //A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3)

                        const __m256i lhs_mat_01_21_sp1 = _mm256_shuffle_epi32(lhs_mat_01_21, 160); //A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                        const __m256i lhs_mat_23_21_sp1 = _mm256_shuffle_epi32(lhs_mat_23_21, 160); //A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11)

                        const __m256i lhs_mat_01_30_sp1 = _mm256_shuffle_epi32(lhs_mat_01_30, 160); //A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
                        const __m256i lhs_mat_23_30_sp1 = _mm256_shuffle_epi32(lhs_mat_23_30, 160); //A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3)

                        const __m256i lhs_mat_01_31_sp1 = _mm256_shuffle_epi32(lhs_mat_01_31, 160); //A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                        const __m256i lhs_mat_23_31_sp1 = _mm256_shuffle_epi32(lhs_mat_23_31, 160); //A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11)

                        const __m256i lhs_mat_01_40_sp1 = _mm256_shuffle_epi32(lhs_mat_01_40, 160); //A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
                        const __m256i lhs_mat_23_40_sp1 = _mm256_shuffle_epi32(lhs_mat_23_40, 160); //A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3)

                        const __m256i lhs_mat_01_41_sp1 = _mm256_shuffle_epi32(lhs_mat_01_41, 160); //A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                        const __m256i lhs_mat_23_41_sp1 = _mm256_shuffle_epi32(lhs_mat_23_41, 160); //A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11)

                        const __m256i lhs_mat_01_50_sp1 = _mm256_shuffle_epi32(lhs_mat_01_50, 160); //A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
                        const __m256i lhs_mat_23_50_sp1 = _mm256_shuffle_epi32(lhs_mat_23_50, 160); //A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3)

                        const __m256i lhs_mat_01_51_sp1 = _mm256_shuffle_epi32(lhs_mat_01_51, 160); //A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                        const __m256i lhs_mat_23_51_sp1 = _mm256_shuffle_epi32(lhs_mat_23_51, 160); //A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11)

                        const __m256i lhs_mat_01_60_sp1 = _mm256_shuffle_epi32(lhs_mat_01_60, 160); //A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
                        const __m256i lhs_mat_23_60_sp1 = _mm256_shuffle_epi32(lhs_mat_23_60, 160); //A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3)

                        const __m256i lhs_mat_01_61_sp1 = _mm256_shuffle_epi32(lhs_mat_01_61, 160); //A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                        const __m256i lhs_mat_23_61_sp1 = _mm256_shuffle_epi32(lhs_mat_23_61, 160); //A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11)

                        const __m256i lhs_mat_01_70_sp1 = _mm256_shuffle_epi32(lhs_mat_01_70, 160); //A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
                        const __m256i lhs_mat_23_70_sp1 = _mm256_shuffle_epi32(lhs_mat_23_70, 160); //A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3)

                        const __m256i lhs_mat_01_71_sp1 = _mm256_shuffle_epi32(lhs_mat_01_71, 160); //A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                        const __m256i lhs_mat_23_71_sp1 = _mm256_shuffle_epi32(lhs_mat_23_71, 160); //A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11)

                        // Shuffle pattern two- left side input
                        const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                        const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                        const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                        const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                        const __m256i lhs_mat_01_20_sp2 = _mm256_shuffle_epi32(lhs_mat_01_20, 245); //A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
                        const __m256i lhs_mat_23_20_sp2 = _mm256_shuffle_epi32(lhs_mat_23_20, 245); //A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7)

                        const __m256i lhs_mat_01_21_sp2 = _mm256_shuffle_epi32(lhs_mat_01_21, 245); //A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                        const __m256i lhs_mat_23_21_sp2 = _mm256_shuffle_epi32(lhs_mat_23_21, 245); //A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15)

                        const __m256i lhs_mat_01_30_sp2 = _mm256_shuffle_epi32(lhs_mat_01_30, 245); //A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
                        const __m256i lhs_mat_23_30_sp2 = _mm256_shuffle_epi32(lhs_mat_23_30, 245); //A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7)

                        const __m256i lhs_mat_01_31_sp2 = _mm256_shuffle_epi32(lhs_mat_01_31, 245); //A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                        const __m256i lhs_mat_23_31_sp2 = _mm256_shuffle_epi32(lhs_mat_23_31, 245); //A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15)

                        const __m256i lhs_mat_01_40_sp2 = _mm256_shuffle_epi32(lhs_mat_01_40, 245); //A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
                        const __m256i lhs_mat_23_40_sp2 = _mm256_shuffle_epi32(lhs_mat_23_40, 245); //A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7)

                        const __m256i lhs_mat_01_41_sp2 = _mm256_shuffle_epi32(lhs_mat_01_41, 245); //A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                        const __m256i lhs_mat_23_41_sp2 = _mm256_shuffle_epi32(lhs_mat_23_41, 245); //A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15)

                        const __m256i lhs_mat_01_50_sp2 = _mm256_shuffle_epi32(lhs_mat_01_50, 245); //A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
                        const __m256i lhs_mat_23_50_sp2 = _mm256_shuffle_epi32(lhs_mat_23_50, 245); //A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7)

                        const __m256i lhs_mat_01_51_sp2 = _mm256_shuffle_epi32(lhs_mat_01_51, 245); //A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                        const __m256i lhs_mat_23_51_sp2 = _mm256_shuffle_epi32(lhs_mat_23_51, 245); //A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15)

                        const __m256i lhs_mat_01_60_sp2 = _mm256_shuffle_epi32(lhs_mat_01_60, 245); //A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
                        const __m256i lhs_mat_23_60_sp2 = _mm256_shuffle_epi32(lhs_mat_23_60, 245); //A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7)

                        const __m256i lhs_mat_01_61_sp2 = _mm256_shuffle_epi32(lhs_mat_01_61, 245); //A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                        const __m256i lhs_mat_23_61_sp2 = _mm256_shuffle_epi32(lhs_mat_23_61, 245); //A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15)

                        const __m256i lhs_mat_01_70_sp2 = _mm256_shuffle_epi32(lhs_mat_01_70, 245); //A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
                        const __m256i lhs_mat_23_70_sp2 = _mm256_shuffle_epi32(lhs_mat_23_70, 245); //A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7)

                        const __m256i lhs_mat_01_71_sp2 = _mm256_shuffle_epi32(lhs_mat_01_71, 245); //A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                        const __m256i lhs_mat_23_71_sp2 = _mm256_shuffle_epi32(lhs_mat_23_71, 245); //A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1),_mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1));
                        __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1),_mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1));

                        __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1),_mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1));
                        __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1),_mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1));

                        __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1),_mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1));
                        __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1),_mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1));

                        __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1),_mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1));
                        __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1),_mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1));

                        __m256i iacc_mat_00_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_01_20_sp1),_mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_01_21_sp1));
                        __m256i iacc_mat_01_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_01_20_sp1),_mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_01_21_sp1));

                        __m256i iacc_mat_10_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_23_20_sp1),_mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_23_21_sp1));
                        __m256i iacc_mat_11_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_23_20_sp1),_mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_23_21_sp1));

                        __m256i iacc_mat_00_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_01_30_sp1),_mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_01_31_sp1));
                        __m256i iacc_mat_01_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_01_30_sp1),_mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_01_31_sp1));

                        __m256i iacc_mat_10_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_23_30_sp1),_mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_23_31_sp1));
                        __m256i iacc_mat_11_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_23_30_sp1),_mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_23_31_sp1));

                        __m256i iacc_mat_00_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_01_40_sp1),_mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_01_41_sp1));
                        __m256i iacc_mat_01_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_01_40_sp1),_mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_01_41_sp1));

                        __m256i iacc_mat_10_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_23_40_sp1),_mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_23_41_sp1));
                        __m256i iacc_mat_11_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_23_40_sp1),_mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_23_41_sp1));

                        __m256i iacc_mat_00_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_01_50_sp1),_mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_01_51_sp1));
                        __m256i iacc_mat_01_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_01_50_sp1),_mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_01_51_sp1));

                        __m256i iacc_mat_10_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_23_50_sp1),_mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_23_51_sp1));
                        __m256i iacc_mat_11_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_23_50_sp1),_mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_23_51_sp1));

                        __m256i iacc_mat_00_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_01_60_sp1),_mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_01_61_sp1));
                        __m256i iacc_mat_01_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_01_60_sp1),_mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_01_61_sp1));

                        __m256i iacc_mat_10_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_23_60_sp1),_mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_23_61_sp1));
                        __m256i iacc_mat_11_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_23_60_sp1),_mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_23_61_sp1));

                        __m256i iacc_mat_00_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_01_70_sp1),_mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_01_71_sp1));
                        __m256i iacc_mat_01_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_01_70_sp1),_mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_01_71_sp1));

                        __m256i iacc_mat_10_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_23_70_sp1),_mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_23_71_sp1));
                        __m256i iacc_mat_11_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_23_70_sp1),_mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_23_71_sp1));


                        __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2),_mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2));
                        __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2),_mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2));

                        __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2),_mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2));
                        __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2),_mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2));

                        __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2),_mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2));
                        __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2),_mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2));

                        __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2),_mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2));
                        __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2),_mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2));

                        __m256i iacc_mat_00_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_01_20_sp2),_mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_01_21_sp2));
                        __m256i iacc_mat_01_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_01_20_sp2),_mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_01_21_sp2));

                        __m256i iacc_mat_10_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_23_20_sp2),_mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_23_21_sp2));
                        __m256i iacc_mat_11_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_23_20_sp2),_mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_23_21_sp2));

                        __m256i iacc_mat_00_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_01_30_sp2),_mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_01_31_sp2));
                        __m256i iacc_mat_01_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_01_30_sp2),_mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_01_31_sp2));

                        __m256i iacc_mat_10_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_23_30_sp2),_mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_23_31_sp2));
                        __m256i iacc_mat_11_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_23_30_sp2),_mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_23_31_sp2));

                        __m256i iacc_mat_00_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_01_40_sp2),_mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_01_41_sp2));
                        __m256i iacc_mat_01_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_01_40_sp2),_mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_01_41_sp2));

                        __m256i iacc_mat_10_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_23_40_sp2),_mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_23_41_sp2));
                        __m256i iacc_mat_11_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_23_40_sp2),_mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_23_41_sp2));

                        __m256i iacc_mat_00_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_01_50_sp2),_mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_01_51_sp2));
                        __m256i iacc_mat_01_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_01_50_sp2),_mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_01_51_sp2));

                        __m256i iacc_mat_10_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_23_50_sp2),_mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_23_51_sp2));
                        __m256i iacc_mat_11_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_23_50_sp2),_mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_23_51_sp2));

                        __m256i iacc_mat_00_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_01_60_sp2),_mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_01_61_sp2));
                        __m256i iacc_mat_01_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_01_60_sp2),_mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_01_61_sp2));

                        __m256i iacc_mat_10_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_23_60_sp2),_mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_23_61_sp2));
                        __m256i iacc_mat_11_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_23_60_sp2),_mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_23_61_sp2));

                        __m256i iacc_mat_00_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_01_70_sp2),_mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_01_71_sp2));
                        __m256i iacc_mat_01_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_01_70_sp2),_mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_01_71_sp2));

                        __m256i iacc_mat_10_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_23_70_sp2),_mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_23_71_sp2));
                        __m256i iacc_mat_11_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_23_70_sp2),_mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_23_71_sp2));

                        // Combine results from both shuffle patterns for each output block
                        __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        __m256i iacc_mat_00_2 = _mm256_add_epi16(iacc_mat_00_2_sp1, iacc_mat_00_2_sp2);
                        __m256i iacc_mat_01_2 = _mm256_add_epi16(iacc_mat_01_2_sp1, iacc_mat_01_2_sp2);
                        __m256i iacc_mat_10_2 = _mm256_add_epi16(iacc_mat_10_2_sp1, iacc_mat_10_2_sp2);
                        __m256i iacc_mat_11_2 = _mm256_add_epi16(iacc_mat_11_2_sp1, iacc_mat_11_2_sp2);

                        __m256i iacc_mat_00_3 = _mm256_add_epi16(iacc_mat_00_3_sp1, iacc_mat_00_3_sp2);
                        __m256i iacc_mat_01_3 = _mm256_add_epi16(iacc_mat_01_3_sp1, iacc_mat_01_3_sp2);
                        __m256i iacc_mat_10_3 = _mm256_add_epi16(iacc_mat_10_3_sp1, iacc_mat_10_3_sp2);
                        __m256i iacc_mat_11_3 = _mm256_add_epi16(iacc_mat_11_3_sp1, iacc_mat_11_3_sp2);

                        __m256i iacc_mat_00_4 = _mm256_add_epi16(iacc_mat_00_4_sp1, iacc_mat_00_4_sp2);
                        __m256i iacc_mat_01_4 = _mm256_add_epi16(iacc_mat_01_4_sp1, iacc_mat_01_4_sp2);
                        __m256i iacc_mat_10_4 = _mm256_add_epi16(iacc_mat_10_4_sp1, iacc_mat_10_4_sp2);
                        __m256i iacc_mat_11_4 = _mm256_add_epi16(iacc_mat_11_4_sp1, iacc_mat_11_4_sp2);

                        __m256i iacc_mat_00_5 = _mm256_add_epi16(iacc_mat_00_5_sp1, iacc_mat_00_5_sp2);
                        __m256i iacc_mat_01_5 = _mm256_add_epi16(iacc_mat_01_5_sp1, iacc_mat_01_5_sp2);
                        __m256i iacc_mat_10_5 = _mm256_add_epi16(iacc_mat_10_5_sp1, iacc_mat_10_5_sp2);
                        __m256i iacc_mat_11_5 = _mm256_add_epi16(iacc_mat_11_5_sp1, iacc_mat_11_5_sp2);

                        __m256i iacc_mat_00_6 = _mm256_add_epi16(iacc_mat_00_6_sp1, iacc_mat_00_6_sp2);
                        __m256i iacc_mat_01_6 = _mm256_add_epi16(iacc_mat_01_6_sp1, iacc_mat_01_6_sp2);
                        __m256i iacc_mat_10_6 = _mm256_add_epi16(iacc_mat_10_6_sp1, iacc_mat_10_6_sp2);
                        __m256i iacc_mat_11_6 = _mm256_add_epi16(iacc_mat_11_6_sp1, iacc_mat_11_6_sp2);

                        __m256i iacc_mat_00_7 = _mm256_add_epi16(iacc_mat_00_7_sp1, iacc_mat_00_7_sp2);
                        __m256i iacc_mat_01_7 = _mm256_add_epi16(iacc_mat_01_7_sp1, iacc_mat_01_7_sp2);
                        __m256i iacc_mat_10_7 = _mm256_add_epi16(iacc_mat_10_7_sp1, iacc_mat_10_7_sp2);
                        __m256i iacc_mat_11_7 = _mm256_add_epi16(iacc_mat_11_7_sp1, iacc_mat_11_7_sp2);

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                        iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                        iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                        iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                        iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                        iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                        iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                        iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                        iacc_mat_00_2 = _mm256_madd_epi16(iacc_mat_00_2, scale_0145_2);
                        iacc_mat_01_2 = _mm256_madd_epi16(iacc_mat_01_2, scale_2367_2);
                        iacc_mat_10_2 = _mm256_madd_epi16(iacc_mat_10_2, scale_0145_2);
                        iacc_mat_11_2 = _mm256_madd_epi16(iacc_mat_11_2, scale_2367_2);

                        iacc_mat_00_3 = _mm256_madd_epi16(iacc_mat_00_3, scale_0145_3);
                        iacc_mat_01_3 = _mm256_madd_epi16(iacc_mat_01_3, scale_2367_3);
                        iacc_mat_10_3 = _mm256_madd_epi16(iacc_mat_10_3, scale_0145_3);
                        iacc_mat_11_3 = _mm256_madd_epi16(iacc_mat_11_3, scale_2367_3);

                        iacc_mat_00_4 = _mm256_madd_epi16(iacc_mat_00_4, scale_0145_4);
                        iacc_mat_01_4 = _mm256_madd_epi16(iacc_mat_01_4, scale_2367_4);
                        iacc_mat_10_4 = _mm256_madd_epi16(iacc_mat_10_4, scale_0145_4);
                        iacc_mat_11_4 = _mm256_madd_epi16(iacc_mat_11_4, scale_2367_4);

                        iacc_mat_00_5 = _mm256_madd_epi16(iacc_mat_00_5, scale_0145_5);
                        iacc_mat_01_5 = _mm256_madd_epi16(iacc_mat_01_5, scale_2367_5);
                        iacc_mat_10_5 = _mm256_madd_epi16(iacc_mat_10_5, scale_0145_5);
                        iacc_mat_11_5 = _mm256_madd_epi16(iacc_mat_11_5, scale_2367_5);

                        iacc_mat_00_6 = _mm256_madd_epi16(iacc_mat_00_6, scale_0145_6);
                        iacc_mat_01_6 = _mm256_madd_epi16(iacc_mat_01_6, scale_2367_6);
                        iacc_mat_10_6 = _mm256_madd_epi16(iacc_mat_10_6, scale_0145_6);
                        iacc_mat_11_6 = _mm256_madd_epi16(iacc_mat_11_6, scale_2367_6);

                        iacc_mat_00_7 = _mm256_madd_epi16(iacc_mat_00_7, scale_0145_7);
                        iacc_mat_01_7 = _mm256_madd_epi16(iacc_mat_01_7, scale_2367_7);
                        iacc_mat_10_7 = _mm256_madd_epi16(iacc_mat_10_7, scale_0145_7);
                        iacc_mat_11_7 = _mm256_madd_epi16(iacc_mat_11_7, scale_2367_7);

                        __m256i iacc_mat_00 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_0, iacc_mat_00_1), _mm256_add_epi32(iacc_mat_00_2, iacc_mat_00_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_4, iacc_mat_00_5), _mm256_add_epi32(iacc_mat_00_6, iacc_mat_00_7)));
                        __m256i iacc_mat_01 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_0, iacc_mat_01_1), _mm256_add_epi32(iacc_mat_01_2, iacc_mat_01_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_4, iacc_mat_01_5), _mm256_add_epi32(iacc_mat_01_6, iacc_mat_01_7)));
                        __m256i iacc_mat_10 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_0, iacc_mat_10_1), _mm256_add_epi32(iacc_mat_10_2, iacc_mat_10_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_4, iacc_mat_10_5), _mm256_add_epi32(iacc_mat_10_6, iacc_mat_10_7)));
                        __m256i iacc_mat_11 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_0, iacc_mat_11_1), _mm256_add_epi32(iacc_mat_11_2, iacc_mat_11_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_4, iacc_mat_11_5), _mm256_add_epi32(iacc_mat_11_6, iacc_mat_11_7)));

                        // Straighten out to make 4 row vectors
                        __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                        __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                        __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                        __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);

                        // Multiply with appropriate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        __m256i lhs_bsums_01_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_0123), lhs_raw_bsums_01_0123, 1);
                        __m256i lhs_bsums_23_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_0123), lhs_raw_bsums_23_0123, 1);
                        __m256i lhs_bsums_01_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_4567), lhs_raw_bsums_01_4567, 1);
                        __m256i lhs_bsums_23_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_4567), lhs_raw_bsums_23_4567, 1);

                       // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
                        __m256i iacc_row_min_0_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 0), mins_01);
                        __m256i iacc_row_min_1_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 170), mins_01);
                        __m256i iacc_row_min_2_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 0), mins_01);
                        __m256i iacc_row_min_3_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 170), mins_01);

                        __m256i iacc_row_min_0_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 85), mins_23);
                        __m256i iacc_row_min_1_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 255), mins_23);
                        __m256i iacc_row_min_2_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 85), mins_23);
                        __m256i iacc_row_min_3_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 255), mins_23);

                        __m256i iacc_row_min_0_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 0), mins_45);
                        __m256i iacc_row_min_1_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 170), mins_45);
                        __m256i iacc_row_min_2_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 0), mins_45);
                        __m256i iacc_row_min_3_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 170), mins_45);

                        __m256i iacc_row_min_0_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 85), mins_67);
                        __m256i iacc_row_min_1_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 255), mins_67);
                        __m256i iacc_row_min_2_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 85), mins_67);
                        __m256i iacc_row_min_3_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 255), mins_67);

                        __m256i iacc_row_min_0 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_0_01, iacc_row_min_0_23), _mm256_add_epi32(iacc_row_min_0_45,iacc_row_min_0_67));
                        __m256i iacc_row_min_1 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_1_01, iacc_row_min_1_23), _mm256_add_epi32(iacc_row_min_1_45,iacc_row_min_1_67));
                        __m256i iacc_row_min_2 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_2_01, iacc_row_min_2_23), _mm256_add_epi32(iacc_row_min_2_45,iacc_row_min_2_67));
                        __m256i iacc_row_min_3 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_3_01, iacc_row_min_3_23), _mm256_add_epi32(iacc_row_min_3_45,iacc_row_min_3_67));

                        acc_min_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);

                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));

            }
        }
    }

    for (; y < nr / 4; y ++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q2_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Delta values - Load the eight scale values of block_q2_kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q2_kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_k for eight sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    //superblock    sub block   which part of sub block
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);

                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);

                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 2-bit -> 8-bit
                    // First sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m3b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m3b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m3b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m3b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    // Second sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(rhs_raw_mat_0145_2, m3b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(rhs_raw_mat_2367_2, m3b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(rhs_raw_mat_0145_3, m3b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(rhs_raw_mat_2367_3, m3b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    // Third sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 2), m3b); //B20(0-7) B21(0-7) B24(0-7) B25(0-7)
                    const __m256i rhs_mat_2367_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 2), m3b); //B22(0-7) B23(0-7) B26(0-7) B27(0-7)

                    const __m256i rhs_mat_0145_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 2), m3b); //B20(8-15) B21(8-15) B24(8-15) B25(8-15)
                    const __m256i rhs_mat_2367_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 2), m3b); //B22(8-15) B23(8-15) B26(8-15) B27(8-15)

                    // Fourth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 2), m3b); //B30(0-7) B31(0-7) B34(0-7) B35(0-7)
                    const __m256i rhs_mat_2367_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 2), m3b); //B32(0-7) B33(0-7) B36(0-7) B37(0-7)

                    const __m256i rhs_mat_0145_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 2), m3b); //B30(8-15) B31(8-15) B34(8-15) B35(8-15)
                    const __m256i rhs_mat_2367_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 2), m3b); //B32(8-15) B33(8-15) B36(8-15) B37(8-15)

                    // Fifth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m3b); //B40(0-7) B41(0-7) B44(0-7) B45(0-7)
                    const __m256i rhs_mat_2367_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m3b); //B42(0-7) B43(0-7) B46(0-7) B47(0-7)

                    const __m256i rhs_mat_0145_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m3b); //B40(8-15) B41(8-15) B44(8-15) B45(8-15)
                    const __m256i rhs_mat_2367_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m3b); //B42(8-15) B43(8-15) B46(8-15) B47(8-15)

                    // Sixth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m3b); //B50(0-7) B51(0-7) B54(0-7) B55(0-7)
                    const __m256i rhs_mat_2367_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m3b); //B52(0-7) B53(0-7) B56(0-7) B57(0-7)

                    const __m256i rhs_mat_0145_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m3b); //B50(8-15) B51(8-15) B54(8-15) B55(8-15)
                    const __m256i rhs_mat_2367_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m3b); //B52(8-15) B53(8-15) B56(8-15) B57(8-15)

                    // Seventh sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 6), m3b); //B60(0-7) B61(0-7) B64(0-7) B65(0-7)
                    const __m256i rhs_mat_2367_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 6), m3b); //B62(0-7) B63(0-7) B66(0-7) B67(0-7)

                    const __m256i rhs_mat_0145_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 6), m3b); //B60(8-15) B61(8-15) B64(8-15) B65(8-15)
                    const __m256i rhs_mat_2367_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 6), m3b); //B62(8-15) B63(8-15) B66(8-15) B67(8-15)

                    // Eighth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 6), m3b); //B70(0-7) B71(0-7) B74(0-7) B75(0-7)
                    const __m256i rhs_mat_2367_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 6), m3b); //B72(0-7) B73(0-7) B76(0-7) B77(0-7)

                    const __m256i rhs_mat_0145_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 6), m3b); //B70(8-15) B71(8-15) B74(8-15) B75(8-15)
                    const __m256i rhs_mat_2367_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 6), m3b); //B72(8-15) B73(8-15) B76(8-15) B77(8-15)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_20_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_20, 136); //B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3)
                    const __m256i rhs_mat_2367_20_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_20, 136); //B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3)

                    const __m256i rhs_mat_0145_21_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_21, 136); //B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11)
                    const __m256i rhs_mat_2367_21_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_21, 136); //B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11)

                    const __m256i rhs_mat_0145_30_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_30, 136); //B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3)
                    const __m256i rhs_mat_2367_30_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_30, 136); //B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3)

                    const __m256i rhs_mat_0145_31_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_31, 136); //B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11
                    const __m256i rhs_mat_2367_31_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_31, 136); //B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11)

                    const __m256i rhs_mat_0145_40_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_40, 136); //B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3)
                    const __m256i rhs_mat_2367_40_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_40, 136); //B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3)

                    const __m256i rhs_mat_0145_41_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_41, 136); //B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11)
                    const __m256i rhs_mat_2367_41_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_41, 136); //B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11)

                    const __m256i rhs_mat_0145_50_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_50, 136); //B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3)
                    const __m256i rhs_mat_2367_50_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_50, 136); //B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3)

                    const __m256i rhs_mat_0145_51_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_51, 136); //B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11)
                    const __m256i rhs_mat_2367_51_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_51, 136); //B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11)

                    const __m256i rhs_mat_0145_60_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_60, 136); //B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3)
                    const __m256i rhs_mat_2367_60_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_60, 136); //B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3)

                    const __m256i rhs_mat_0145_61_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_61, 136); //B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11)
                    const __m256i rhs_mat_2367_61_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_61, 136); //B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11)

                    const __m256i rhs_mat_0145_70_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_70, 136); //B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3)
                    const __m256i rhs_mat_2367_70_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_70, 136); //B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3)

                    const __m256i rhs_mat_0145_71_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_71, 136); //B70(8-11) B71(8-11) B70(8-11) B71(8-11) B74(8-11) B75(8-11) B74(8-11) B75(8-11)
                    const __m256i rhs_mat_2367_71_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_71, 136); //B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11)


                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_20_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_20, 221); //B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7)
                    const __m256i rhs_mat_2367_20_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_20, 221); //B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7)

                    const __m256i rhs_mat_0145_21_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_21, 221); //B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15)
                    const __m256i rhs_mat_2367_21_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_21, 221); //B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15)

                    const __m256i rhs_mat_0145_30_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_30, 221); //B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7)
                    const __m256i rhs_mat_2367_30_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_30, 221); //B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7)

                    const __m256i rhs_mat_0145_31_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_31, 221); //B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15)
                    const __m256i rhs_mat_2367_31_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_31, 221); //B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15)

                    const __m256i rhs_mat_0145_40_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_40, 221); //B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7)
                    const __m256i rhs_mat_2367_40_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_40, 221); //B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7)

                    const __m256i rhs_mat_0145_41_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_41, 221); //B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15)
                    const __m256i rhs_mat_2367_41_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_41, 221); //B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15)

                    const __m256i rhs_mat_0145_50_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_50, 221); //B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7)
                    const __m256i rhs_mat_2367_50_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_50, 221); //B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7)

                    const __m256i rhs_mat_0145_51_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_51, 221); //B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15)
                    const __m256i rhs_mat_2367_51_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_51, 221); //B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15)

                    const __m256i rhs_mat_0145_60_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_60, 221); //B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7)
                    const __m256i rhs_mat_2367_60_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_60, 221); //B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7)

                    const __m256i rhs_mat_0145_61_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_61, 221); //B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15)
                    const __m256i rhs_mat_2367_61_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_61, 221); //B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15)

                    const __m256i rhs_mat_0145_70_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_70, 221); //B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7)
                    const __m256i rhs_mat_2367_70_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_70, 221); //B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7)

                    const __m256i rhs_mat_0145_71_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_71, 221); //B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15)
                    const __m256i rhs_mat_2367_71_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_71, 221); //B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15)


                    //Scales and Mins of corresponding sub blocks from different Q2_K structures are stored together
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
                    const __m128i mins_and_scales_01 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + sb * 64));
                    const __m128i mins_and_scales_23 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 48 + sb * 64));

                    // Extract scales which is lower half from mins_and_scales
                    const __m128i scales_01 = _mm_and_si128(mins_and_scales_01, m4b_sse);
                    const __m128i scales_23 = _mm_and_si128(mins_and_scales_23, m4b_sse);
                    const __m128i scales_45 = _mm_and_si128(mins_and_scales_45, m4b_sse);
                    const __m128i scales_67 = _mm_and_si128(mins_and_scales_67, m4b_sse);

                    // Extract mins which is upper half from mins_and_scales
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_01, 4), m4b_sse));
                    const __m256i mins_23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_23, 4), m4b_sse));
                    const __m256i mins_45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_45, 4), m4b_sse));
                    const __m256i mins_67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_67, 4), m4b_sse));

                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask1_sse));
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask2_sse));

                    const __m256i scales_2 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask1_sse));
                    const __m256i scales_3 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask2_sse));

                    const __m256i scales_4 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask1_sse));
                    const __m256i scales_5 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask2_sse));

                    const __m256i scales_6 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask1_sse));
                    const __m256i scales_7 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask2_sse));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    const __m256i scale_0145_2 = _mm256_shuffle_epi32(scales_2, 68);
                    const __m256i scale_2367_2 = _mm256_shuffle_epi32(scales_2, 238);

                    const __m256i scale_0145_3 = _mm256_shuffle_epi32(scales_3, 68);
                    const __m256i scale_2367_3 = _mm256_shuffle_epi32(scales_3, 238);

                    const __m256i scale_0145_4 = _mm256_shuffle_epi32(scales_4, 68);
                    const __m256i scale_2367_4 = _mm256_shuffle_epi32(scales_4, 238);

                    const __m256i scale_0145_5 = _mm256_shuffle_epi32(scales_5, 68);
                    const __m256i scale_2367_5 = _mm256_shuffle_epi32(scales_5, 238);

                    const __m256i scale_0145_6 = _mm256_shuffle_epi32(scales_6, 68);
                    const __m256i scale_2367_6 = _mm256_shuffle_epi32(scales_6, 238);

                    const __m256i scale_0145_7 = _mm256_shuffle_epi32(scales_7, 68);
                    const __m256i scale_2367_7 = _mm256_shuffle_epi32(scales_7, 238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 512 * sb)));
                    __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                    __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 512 * sb)));
                    __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                    __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 512 * sb)));
                    __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                    __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 512 * sb)));
                    __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                    __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    __m256i lhs_mat_0123_20 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 512 * sb)));
                    __m256i lhs_mat_01_20 = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20, 0);
                    __m256i lhs_mat_23_20 = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20, 17);
                    __m256i lhs_mat_0123_21 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 512 * sb)));
                    __m256i lhs_mat_01_21 = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21, 0);
                    __m256i lhs_mat_23_21 = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21, 17);
                    __m256i lhs_mat_0123_30 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 512 * sb)));
                    __m256i lhs_mat_01_30 = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30, 0);
                    __m256i lhs_mat_23_30 = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30, 17);
                    __m256i lhs_mat_0123_31 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 512 * sb)));
                    __m256i lhs_mat_01_31 = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31, 0);
                    __m256i lhs_mat_23_31 = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31, 17);

                    __m256i lhs_mat_0123_40 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 + 512 * sb)));
                    __m256i lhs_mat_01_40 = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40, 0);
                    __m256i lhs_mat_23_40 = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40, 17);
                    __m256i lhs_mat_0123_41 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 288 + 512 * sb)));
                    __m256i lhs_mat_01_41 = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41, 0);
                    __m256i lhs_mat_23_41 = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41, 17);
                    __m256i lhs_mat_0123_50 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 320 + 512 * sb)));
                    __m256i lhs_mat_01_50 = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50, 0);
                    __m256i lhs_mat_23_50 = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50, 17);
                    __m256i lhs_mat_0123_51 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 352 + 512 * sb)));
                    __m256i lhs_mat_01_51 = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51, 0);
                    __m256i lhs_mat_23_51 = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51, 17);
                    __m256i lhs_mat_0123_60 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 384 + 512 * sb)));
                    __m256i lhs_mat_01_60 = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60, 0);
                    __m256i lhs_mat_23_60 = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60, 17);
                    __m256i lhs_mat_0123_61 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 416 + 512 * sb)));
                    __m256i lhs_mat_01_61 = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61, 0);
                    __m256i lhs_mat_23_61 = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61, 17);
                    __m256i lhs_mat_0123_70 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 448 + 512 * sb)));
                    __m256i lhs_mat_01_70 = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70, 0);
                    __m256i lhs_mat_23_70 = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70, 17);
                    __m256i lhs_mat_0123_71 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 480 + 512 * sb)));
                    __m256i lhs_mat_01_71 = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71, 0);
                    __m256i lhs_mat_23_71 = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71, 17);

                    // Bsums are loaded for the different Q8_K blocks
                    __m128i lhs_raw_bsums_01_0123 = _mm_loadu_si128((const __m128i *)((a_ptr[b].bsums + 32 * sb)));
                    __m128i lhs_raw_bsums_23_0123 = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + 8 + 32 * sb));
                    __m128i lhs_raw_bsums_01_4567 = _mm_loadu_si128((const __m128i *)((a_ptr[b].bsums + 16 + 32 * sb)));
                    __m128i lhs_raw_bsums_23_4567 = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + 24 + 32 * sb));

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                    const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160); //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                    const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                    const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160); //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                    const __m256i lhs_mat_01_20_sp1 = _mm256_shuffle_epi32(lhs_mat_01_20, 160); //A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
                    const __m256i lhs_mat_23_20_sp1 = _mm256_shuffle_epi32(lhs_mat_23_20, 160); //A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3)

                    const __m256i lhs_mat_01_21_sp1 = _mm256_shuffle_epi32(lhs_mat_01_21, 160); //A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                    const __m256i lhs_mat_23_21_sp1 = _mm256_shuffle_epi32(lhs_mat_23_21, 160); //A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11)

                    const __m256i lhs_mat_01_30_sp1 = _mm256_shuffle_epi32(lhs_mat_01_30, 160); //A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
                    const __m256i lhs_mat_23_30_sp1 = _mm256_shuffle_epi32(lhs_mat_23_30, 160); //A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3)

                    const __m256i lhs_mat_01_31_sp1 = _mm256_shuffle_epi32(lhs_mat_01_31, 160); //A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                    const __m256i lhs_mat_23_31_sp1 = _mm256_shuffle_epi32(lhs_mat_23_31, 160); //A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11)

                    const __m256i lhs_mat_01_40_sp1 = _mm256_shuffle_epi32(lhs_mat_01_40, 160); //A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
                    const __m256i lhs_mat_23_40_sp1 = _mm256_shuffle_epi32(lhs_mat_23_40, 160); //A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3)

                    const __m256i lhs_mat_01_41_sp1 = _mm256_shuffle_epi32(lhs_mat_01_41, 160); //A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                    const __m256i lhs_mat_23_41_sp1 = _mm256_shuffle_epi32(lhs_mat_23_41, 160); //A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11)

                    const __m256i lhs_mat_01_50_sp1 = _mm256_shuffle_epi32(lhs_mat_01_50, 160); //A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
                    const __m256i lhs_mat_23_50_sp1 = _mm256_shuffle_epi32(lhs_mat_23_50, 160); //A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3)

                    const __m256i lhs_mat_01_51_sp1 = _mm256_shuffle_epi32(lhs_mat_01_51, 160); //A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                    const __m256i lhs_mat_23_51_sp1 = _mm256_shuffle_epi32(lhs_mat_23_51, 160); //A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11)

                    const __m256i lhs_mat_01_60_sp1 = _mm256_shuffle_epi32(lhs_mat_01_60, 160); //A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
                    const __m256i lhs_mat_23_60_sp1 = _mm256_shuffle_epi32(lhs_mat_23_60, 160); //A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3)

                    const __m256i lhs_mat_01_61_sp1 = _mm256_shuffle_epi32(lhs_mat_01_61, 160); //A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                    const __m256i lhs_mat_23_61_sp1 = _mm256_shuffle_epi32(lhs_mat_23_61, 160); //A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11)

                    const __m256i lhs_mat_01_70_sp1 = _mm256_shuffle_epi32(lhs_mat_01_70, 160); //A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
                    const __m256i lhs_mat_23_70_sp1 = _mm256_shuffle_epi32(lhs_mat_23_70, 160); //A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3)

                    const __m256i lhs_mat_01_71_sp1 = _mm256_shuffle_epi32(lhs_mat_01_71, 160); //A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                    const __m256i lhs_mat_23_71_sp1 = _mm256_shuffle_epi32(lhs_mat_23_71, 160); //A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11)

                    // Shuffle pattern two- left side input
                    const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                    const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                    const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                    const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                    const __m256i lhs_mat_01_20_sp2 = _mm256_shuffle_epi32(lhs_mat_01_20, 245); //A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
                    const __m256i lhs_mat_23_20_sp2 = _mm256_shuffle_epi32(lhs_mat_23_20, 245); //A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7)

                    const __m256i lhs_mat_01_21_sp2 = _mm256_shuffle_epi32(lhs_mat_01_21, 245); //A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                    const __m256i lhs_mat_23_21_sp2 = _mm256_shuffle_epi32(lhs_mat_23_21, 245); //A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15)

                    const __m256i lhs_mat_01_30_sp2 = _mm256_shuffle_epi32(lhs_mat_01_30, 245); //A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
                    const __m256i lhs_mat_23_30_sp2 = _mm256_shuffle_epi32(lhs_mat_23_30, 245); //A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7)

                    const __m256i lhs_mat_01_31_sp2 = _mm256_shuffle_epi32(lhs_mat_01_31, 245); //A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                    const __m256i lhs_mat_23_31_sp2 = _mm256_shuffle_epi32(lhs_mat_23_31, 245); //A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15)

                    const __m256i lhs_mat_01_40_sp2 = _mm256_shuffle_epi32(lhs_mat_01_40, 245); //A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
                    const __m256i lhs_mat_23_40_sp2 = _mm256_shuffle_epi32(lhs_mat_23_40, 245); //A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7)

                    const __m256i lhs_mat_01_41_sp2 = _mm256_shuffle_epi32(lhs_mat_01_41, 245); //A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                    const __m256i lhs_mat_23_41_sp2 = _mm256_shuffle_epi32(lhs_mat_23_41, 245); //A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15)

                    const __m256i lhs_mat_01_50_sp2 = _mm256_shuffle_epi32(lhs_mat_01_50, 245); //A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
                    const __m256i lhs_mat_23_50_sp2 = _mm256_shuffle_epi32(lhs_mat_23_50, 245); //A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7)

                    const __m256i lhs_mat_01_51_sp2 = _mm256_shuffle_epi32(lhs_mat_01_51, 245); //A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                    const __m256i lhs_mat_23_51_sp2 = _mm256_shuffle_epi32(lhs_mat_23_51, 245); //A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15)

                    const __m256i lhs_mat_01_60_sp2 = _mm256_shuffle_epi32(lhs_mat_01_60, 245); //A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
                    const __m256i lhs_mat_23_60_sp2 = _mm256_shuffle_epi32(lhs_mat_23_60, 245); //A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7)

                    const __m256i lhs_mat_01_61_sp2 = _mm256_shuffle_epi32(lhs_mat_01_61, 245); //A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                    const __m256i lhs_mat_23_61_sp2 = _mm256_shuffle_epi32(lhs_mat_23_61, 245); //A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15)

                    const __m256i lhs_mat_01_70_sp2 = _mm256_shuffle_epi32(lhs_mat_01_70, 245); //A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
                    const __m256i lhs_mat_23_70_sp2 = _mm256_shuffle_epi32(lhs_mat_23_70, 245); //A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7)

                    const __m256i lhs_mat_01_71_sp2 = _mm256_shuffle_epi32(lhs_mat_01_71, 245); //A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                    const __m256i lhs_mat_23_71_sp2 = _mm256_shuffle_epi32(lhs_mat_23_71, 245); //A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1),_mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1));
                    __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1),_mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1));

                    __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1),_mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1));
                    __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1),_mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1));

                    __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1),_mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1));
                    __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1),_mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1));

                    __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1),_mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1));
                    __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1),_mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1));

                    __m256i iacc_mat_00_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_01_20_sp1),_mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_01_21_sp1));
                    __m256i iacc_mat_01_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_01_20_sp1),_mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_01_21_sp1));

                    __m256i iacc_mat_10_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_23_20_sp1),_mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_23_21_sp1));
                    __m256i iacc_mat_11_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_23_20_sp1),_mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_23_21_sp1));

                    __m256i iacc_mat_00_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_01_30_sp1),_mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_01_31_sp1));
                    __m256i iacc_mat_01_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_01_30_sp1),_mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_01_31_sp1));

                    __m256i iacc_mat_10_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_23_30_sp1),_mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_23_31_sp1));
                    __m256i iacc_mat_11_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_23_30_sp1),_mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_23_31_sp1));

                    __m256i iacc_mat_00_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_01_40_sp1),_mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_01_41_sp1));
                    __m256i iacc_mat_01_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_01_40_sp1),_mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_01_41_sp1));

                    __m256i iacc_mat_10_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_23_40_sp1),_mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_23_41_sp1));
                    __m256i iacc_mat_11_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_23_40_sp1),_mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_23_41_sp1));

                    __m256i iacc_mat_00_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_01_50_sp1),_mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_01_51_sp1));
                    __m256i iacc_mat_01_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_01_50_sp1),_mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_01_51_sp1));

                    __m256i iacc_mat_10_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_23_50_sp1),_mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_23_51_sp1));
                    __m256i iacc_mat_11_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_23_50_sp1),_mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_23_51_sp1));

                    __m256i iacc_mat_00_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_01_60_sp1),_mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_01_61_sp1));
                    __m256i iacc_mat_01_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_01_60_sp1),_mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_01_61_sp1));

                    __m256i iacc_mat_10_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_23_60_sp1),_mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_23_61_sp1));
                    __m256i iacc_mat_11_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_23_60_sp1),_mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_23_61_sp1));

                    __m256i iacc_mat_00_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_01_70_sp1),_mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_01_71_sp1));
                    __m256i iacc_mat_01_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_01_70_sp1),_mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_01_71_sp1));

                    __m256i iacc_mat_10_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_23_70_sp1),_mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_23_71_sp1));
                    __m256i iacc_mat_11_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_23_70_sp1),_mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_23_71_sp1));


                    __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2),_mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2));
                    __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2),_mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2));

                    __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2),_mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2));
                    __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2),_mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2));

                    __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2),_mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2));
                    __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2),_mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2));

                    __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2),_mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2));
                    __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2),_mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2));

                    __m256i iacc_mat_00_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_01_20_sp2),_mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_01_21_sp2));
                    __m256i iacc_mat_01_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_01_20_sp2),_mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_01_21_sp2));

                    __m256i iacc_mat_10_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_23_20_sp2),_mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_23_21_sp2));
                    __m256i iacc_mat_11_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_23_20_sp2),_mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_23_21_sp2));

                    __m256i iacc_mat_00_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_01_30_sp2),_mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_01_31_sp2));
                    __m256i iacc_mat_01_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_01_30_sp2),_mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_01_31_sp2));

                    __m256i iacc_mat_10_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_23_30_sp2),_mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_23_31_sp2));
                    __m256i iacc_mat_11_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_23_30_sp2),_mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_23_31_sp2));

                    __m256i iacc_mat_00_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_01_40_sp2),_mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_01_41_sp2));
                    __m256i iacc_mat_01_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_01_40_sp2),_mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_01_41_sp2));

                    __m256i iacc_mat_10_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_23_40_sp2),_mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_23_41_sp2));
                    __m256i iacc_mat_11_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_23_40_sp2),_mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_23_41_sp2));

                    __m256i iacc_mat_00_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_01_50_sp2),_mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_01_51_sp2));
                    __m256i iacc_mat_01_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_01_50_sp2),_mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_01_51_sp2));

                    __m256i iacc_mat_10_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_23_50_sp2),_mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_23_51_sp2));
                    __m256i iacc_mat_11_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_23_50_sp2),_mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_23_51_sp2));

                    __m256i iacc_mat_00_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_01_60_sp2),_mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_01_61_sp2));
                    __m256i iacc_mat_01_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_01_60_sp2),_mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_01_61_sp2));

                    __m256i iacc_mat_10_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_23_60_sp2),_mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_23_61_sp2));
                    __m256i iacc_mat_11_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_23_60_sp2),_mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_23_61_sp2));

                    __m256i iacc_mat_00_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_01_70_sp2),_mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_01_71_sp2));
                    __m256i iacc_mat_01_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_01_70_sp2),_mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_01_71_sp2));

                    __m256i iacc_mat_10_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_23_70_sp2),_mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_23_71_sp2));
                    __m256i iacc_mat_11_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_23_70_sp2),_mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_23_71_sp2));

                    // Combine results from both shuffle patterns for each output block.
                    __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    __m256i iacc_mat_00_2 = _mm256_add_epi16(iacc_mat_00_2_sp1, iacc_mat_00_2_sp2);
                    __m256i iacc_mat_01_2 = _mm256_add_epi16(iacc_mat_01_2_sp1, iacc_mat_01_2_sp2);
                    __m256i iacc_mat_10_2 = _mm256_add_epi16(iacc_mat_10_2_sp1, iacc_mat_10_2_sp2);
                    __m256i iacc_mat_11_2 = _mm256_add_epi16(iacc_mat_11_2_sp1, iacc_mat_11_2_sp2);

                    __m256i iacc_mat_00_3 = _mm256_add_epi16(iacc_mat_00_3_sp1, iacc_mat_00_3_sp2);
                    __m256i iacc_mat_01_3 = _mm256_add_epi16(iacc_mat_01_3_sp1, iacc_mat_01_3_sp2);
                    __m256i iacc_mat_10_3 = _mm256_add_epi16(iacc_mat_10_3_sp1, iacc_mat_10_3_sp2);
                    __m256i iacc_mat_11_3 = _mm256_add_epi16(iacc_mat_11_3_sp1, iacc_mat_11_3_sp2);

                    __m256i iacc_mat_00_4 = _mm256_add_epi16(iacc_mat_00_4_sp1, iacc_mat_00_4_sp2);
                    __m256i iacc_mat_01_4 = _mm256_add_epi16(iacc_mat_01_4_sp1, iacc_mat_01_4_sp2);
                    __m256i iacc_mat_10_4 = _mm256_add_epi16(iacc_mat_10_4_sp1, iacc_mat_10_4_sp2);
                    __m256i iacc_mat_11_4 = _mm256_add_epi16(iacc_mat_11_4_sp1, iacc_mat_11_4_sp2);

                    __m256i iacc_mat_00_5 = _mm256_add_epi16(iacc_mat_00_5_sp1, iacc_mat_00_5_sp2);
                    __m256i iacc_mat_01_5 = _mm256_add_epi16(iacc_mat_01_5_sp1, iacc_mat_01_5_sp2);
                    __m256i iacc_mat_10_5 = _mm256_add_epi16(iacc_mat_10_5_sp1, iacc_mat_10_5_sp2);
                    __m256i iacc_mat_11_5 = _mm256_add_epi16(iacc_mat_11_5_sp1, iacc_mat_11_5_sp2);

                    __m256i iacc_mat_00_6 = _mm256_add_epi16(iacc_mat_00_6_sp1, iacc_mat_00_6_sp2);
                    __m256i iacc_mat_01_6 = _mm256_add_epi16(iacc_mat_01_6_sp1, iacc_mat_01_6_sp2);
                    __m256i iacc_mat_10_6 = _mm256_add_epi16(iacc_mat_10_6_sp1, iacc_mat_10_6_sp2);
                    __m256i iacc_mat_11_6 = _mm256_add_epi16(iacc_mat_11_6_sp1, iacc_mat_11_6_sp2);

                    __m256i iacc_mat_00_7 = _mm256_add_epi16(iacc_mat_00_7_sp1, iacc_mat_00_7_sp2);
                    __m256i iacc_mat_01_7 = _mm256_add_epi16(iacc_mat_01_7_sp1, iacc_mat_01_7_sp2);
                    __m256i iacc_mat_10_7 = _mm256_add_epi16(iacc_mat_10_7_sp1, iacc_mat_10_7_sp2);
                    __m256i iacc_mat_11_7 = _mm256_add_epi16(iacc_mat_11_7_sp1, iacc_mat_11_7_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                    iacc_mat_00_2 = _mm256_madd_epi16(iacc_mat_00_2, scale_0145_2);
                    iacc_mat_01_2 = _mm256_madd_epi16(iacc_mat_01_2, scale_2367_2);
                    iacc_mat_10_2 = _mm256_madd_epi16(iacc_mat_10_2, scale_0145_2);
                    iacc_mat_11_2 = _mm256_madd_epi16(iacc_mat_11_2, scale_2367_2);

                    iacc_mat_00_3 = _mm256_madd_epi16(iacc_mat_00_3, scale_0145_3);
                    iacc_mat_01_3 = _mm256_madd_epi16(iacc_mat_01_3, scale_2367_3);
                    iacc_mat_10_3 = _mm256_madd_epi16(iacc_mat_10_3, scale_0145_3);
                    iacc_mat_11_3 = _mm256_madd_epi16(iacc_mat_11_3, scale_2367_3);

                    iacc_mat_00_4 = _mm256_madd_epi16(iacc_mat_00_4, scale_0145_4);
                    iacc_mat_01_4 = _mm256_madd_epi16(iacc_mat_01_4, scale_2367_4);
                    iacc_mat_10_4 = _mm256_madd_epi16(iacc_mat_10_4, scale_0145_4);
                    iacc_mat_11_4 = _mm256_madd_epi16(iacc_mat_11_4, scale_2367_4);

                    iacc_mat_00_5 = _mm256_madd_epi16(iacc_mat_00_5, scale_0145_5);
                    iacc_mat_01_5 = _mm256_madd_epi16(iacc_mat_01_5, scale_2367_5);
                    iacc_mat_10_5 = _mm256_madd_epi16(iacc_mat_10_5, scale_0145_5);
                    iacc_mat_11_5 = _mm256_madd_epi16(iacc_mat_11_5, scale_2367_5);

                    iacc_mat_00_6 = _mm256_madd_epi16(iacc_mat_00_6, scale_0145_6);
                    iacc_mat_01_6 = _mm256_madd_epi16(iacc_mat_01_6, scale_2367_6);
                    iacc_mat_10_6 = _mm256_madd_epi16(iacc_mat_10_6, scale_0145_6);
                    iacc_mat_11_6 = _mm256_madd_epi16(iacc_mat_11_6, scale_2367_6);

                    iacc_mat_00_7 = _mm256_madd_epi16(iacc_mat_00_7, scale_0145_7);
                    iacc_mat_01_7 = _mm256_madd_epi16(iacc_mat_01_7, scale_2367_7);
                    iacc_mat_10_7 = _mm256_madd_epi16(iacc_mat_10_7, scale_0145_7);
                    iacc_mat_11_7 = _mm256_madd_epi16(iacc_mat_11_7, scale_2367_7);

                    __m256i iacc_mat_00 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_0, iacc_mat_00_1), _mm256_add_epi32(iacc_mat_00_2, iacc_mat_00_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_4, iacc_mat_00_5), _mm256_add_epi32(iacc_mat_00_6, iacc_mat_00_7)));
                    __m256i iacc_mat_01 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_0, iacc_mat_01_1), _mm256_add_epi32(iacc_mat_01_2, iacc_mat_01_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_4, iacc_mat_01_5), _mm256_add_epi32(iacc_mat_01_6, iacc_mat_01_7)));
                    __m256i iacc_mat_10 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_0, iacc_mat_10_1), _mm256_add_epi32(iacc_mat_10_2, iacc_mat_10_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_4, iacc_mat_10_5), _mm256_add_epi32(iacc_mat_10_6, iacc_mat_10_7)));
                    __m256i iacc_mat_11 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_0, iacc_mat_11_1), _mm256_add_epi32(iacc_mat_11_2, iacc_mat_11_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_4, iacc_mat_11_5), _mm256_add_epi32(iacc_mat_11_6, iacc_mat_11_7)));

                    // Straighten out to make 4 row vectors
                    __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                    __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                    __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                    __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);

                    // Multiply with appropriate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m256i lhs_bsums_01_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_0123), lhs_raw_bsums_01_0123, 1);
                    __m256i lhs_bsums_23_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_0123), lhs_raw_bsums_23_0123, 1);
                    __m256i lhs_bsums_01_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_4567), lhs_raw_bsums_01_4567, 1);
                    __m256i lhs_bsums_23_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_4567), lhs_raw_bsums_23_4567, 1);

                    // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
                    __m256i iacc_row_min_0_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 0), mins_01);
                    __m256i iacc_row_min_1_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 170), mins_01);
                    __m256i iacc_row_min_2_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 0), mins_01);
                    __m256i iacc_row_min_3_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 170), mins_01);

                    __m256i iacc_row_min_0_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 85), mins_23);
                    __m256i iacc_row_min_1_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 255), mins_23);
                    __m256i iacc_row_min_2_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 85), mins_23);
                    __m256i iacc_row_min_3_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 255), mins_23);

                    __m256i iacc_row_min_0_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 0), mins_45);
                    __m256i iacc_row_min_1_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 170), mins_45);
                    __m256i iacc_row_min_2_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 0), mins_45);
                    __m256i iacc_row_min_3_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 170), mins_45);

                    __m256i iacc_row_min_0_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 85), mins_67);
                    __m256i iacc_row_min_1_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 255), mins_67);
                    __m256i iacc_row_min_2_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 85), mins_67);
                    __m256i iacc_row_min_3_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 255), mins_67);

                    __m256i iacc_row_min_0 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_0_01, iacc_row_min_0_23), _mm256_add_epi32(iacc_row_min_0_45,iacc_row_min_0_67));
                    __m256i iacc_row_min_1 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_1_01, iacc_row_min_1_23), _mm256_add_epi32(iacc_row_min_1_45,iacc_row_min_1_67));
                    __m256i iacc_row_min_2 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_2_01, iacc_row_min_2_23), _mm256_add_epi32(iacc_row_min_2_45,iacc_row_min_2_67));
                    __m256i iacc_row_min_3 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_3_01, iacc_row_min_3_23), _mm256_add_epi32(iacc_row_min_3_45,iacc_row_min_3_67));

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
#else

    ggml_gemm_q2_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);


#endif
}


#undef UNUSED
