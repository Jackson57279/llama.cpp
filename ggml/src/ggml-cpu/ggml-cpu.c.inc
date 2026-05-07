#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl-defs.inc"
#include "ggml-backend.h.inc"
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

#include "ggml-cpu-impl-defs.inc"
#include "ggml-impl-defs.inc"
#define GGML_COMMON_DECL_C
#include "ggml-common-defs.inc"

#include "ggml.h.inc"

// GGML CPU internal header

#ifdef __cplusplus
extern "C" {
#endif

// Quantization
void quantize_row_q1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_nvfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_q2_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q3_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q4_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q5_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_tq1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

void quantize_row_iq4_nl (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void quantize_row_iq4_xs (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);

// Dot product
void ggml_vec_dot_q1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_mxfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_nvfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_tq1_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_tq2_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0 (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// Generic implementation
void quantize_row_q8_0_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void quantize_row_q8_1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
void quantize_row_q8_K_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void ggml_vec_dot_q1_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_1_q8_1_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_mxfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_nvfp4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_tq1_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_tq2_0_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q3_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q4_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq2_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_xxs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq3_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_s_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq1_m_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_nl_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif

// Vectorized functions for fundamental operations

#include "ggml-impl-defs.inc"
#include "simd-mappings-defs.inc"
#include "ggml.h.inc"
#include "ggml-cpu.h.inc"

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

// floating point type used to accumulate sums
typedef double ggml_float;

#define GGML_GELU_FP16
#define GGML_GELU_QUICK_FP16

#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL  2
#define GGML_VEC_MAD_UNROLL  32

#ifdef __cplusplus
extern "C" {
#endif

//
// global data
//

// precomputed gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

//
// fundamental operations
//

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);

void ggml_vec_silu_f32(const int n, float * y, const float * x);
ggml_float ggml_vec_cvar_f32(const int n, float * y, const float * x, const float mean); //it will also center y ( y = y - mean )
ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max);
ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max);

inline static void ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_set_i32(const int n, int32_t * x, const int32_t   v) { for (int i = 0; i < n; ++i) x[i] = v;    }
inline static void ggml_vec_cpy_i32(const int n, int32_t * y, const int32_t * x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }

inline static void ggml_vec_set_f16(const int n, ggml_fp16_t * x, const ggml_fp16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_bf16(const int n, ggml_bf16_t * x, const ggml_bf16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) {
    int i = 0;
#if defined(__AVX2__)
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vz = _mm256_add_ps(vx, vy);
        _mm256_storeu_ps(z + i, vz);
    }
#endif
    for (; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

inline static void ggml_vec_add_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) + GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_add1_f32(const int n, float * z, const float * x, const float   v) { for (int i = 0; i < n; ++i) z[i]  = x[i] + v;    }
inline static void ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void ggml_vec_acc1_f32(const int n, float * y, const float   v)                  { for (int i = 0; i < n; ++i) y[i] += v;           }
inline static void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
inline static void ggml_vec_sub_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) - GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void ggml_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void ggml_vec_neg_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = -x[i];       }
inline static void ggml_vec_neg_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(-GGML_CPU_FP16_TO_FP32(x[i]));
    }
}

inline static void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
inline static void ggml_vec_mul_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) * GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }
inline static void ggml_vec_div_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) / GGML_CPU_FP16_TO_FP32(y[i]));
    }
}

// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void ggml_vec_dot_f16_unroll(const int n, const int xs, float * GGML_RESTRICT s, void * GGML_RESTRICT xv, ggml_fp16_t * GGML_RESTRICT y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * GGML_RESTRICT x[GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((char *) xv + i*xs);
    }

#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)

        const int sve_register_length = svcntb() * 8;
        const int ggml_f16_epr = sve_register_length / 16; // running when 16
        const int ggml_f16_step = 8 * ggml_f16_epr; // choose 8 SVE registers

        int np = (n & ~(ggml_f16_step - 1));

        svfloat16_t sum_00 = svdup_n_f16(0.0f);
        svfloat16_t sum_01 = svdup_n_f16(0.0f);
        svfloat16_t sum_02 = svdup_n_f16(0.0f);
        svfloat16_t sum_03 = svdup_n_f16(0.0f);

        svfloat16_t sum_10 = svdup_n_f16(0.0f);
        svfloat16_t sum_11 = svdup_n_f16(0.0f);
        svfloat16_t sum_12 = svdup_n_f16(0.0f);
        svfloat16_t sum_13 = svdup_n_f16(0.0f);

        svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;

        for (int i = 0; i < np; i += ggml_f16_step) {
            ay1 = GGML_F16x_VEC_LOAD(y + i + 0 * ggml_f16_epr, 0); // 8 elements

            ax1 = GGML_F16x_VEC_LOAD(x[0] + i + 0*ggml_f16_epr, 0); // 8 elements
            sum_00 = GGML_F16x_VEC_FMA(sum_00, ax1, ay1);     // sum_00 = sum_00+ax1*ay1
            ax1 = GGML_F16x_VEC_LOAD(x[1] + i + 0*ggml_f16_epr, 0); // 8 elements
            sum_10 = GGML_F16x_VEC_FMA(sum_10, ax1, ay1);

            ay2 = GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 1); // next 8 elements

            ax2 = GGML_F16x_VEC_LOAD(x[0] + i + 1*ggml_f16_epr, 1); // next 8 elements
            sum_01 = GGML_F16x_VEC_FMA(sum_01, ax2, ay2);
            ax2 = GGML_F16x_VEC_LOAD(x[1] + i + 1*ggml_f16_epr, 1);
            sum_11 = GGML_F16x_VEC_FMA(sum_11, ax2, ay2);

            ay3 = GGML_F16x_VEC_LOAD(y + i + 2 * ggml_f16_epr, 2);

            ax3 = GGML_F16x_VEC_LOAD(x[0] + i + 2*ggml_f16_epr, 2);
            sum_02 = GGML_F16x_VEC_FMA(sum_02, ax3, ay3);
            ax3 = GGML_F16x_VEC_LOAD(x[1] + i + 2*ggml_f16_epr, 2);
            sum_12 = GGML_F16x_VEC_FMA(sum_12, ax3, ay3);

            ay4 = GGML_F16x_VEC_LOAD(y + i + 3 * ggml_f16_epr, 3);

            ax4 = GGML_F16x_VEC_LOAD(x[0] + i + 3*ggml_f16_epr, 3);
            sum_03 = GGML_F16x_VEC_FMA(sum_03, ax4, ay4);
            ax4 = GGML_F16x_VEC_LOAD(x[1] + i + 3*ggml_f16_epr, 3);
            sum_13 = GGML_F16x_VEC_FMA(sum_13, ax4, ay4);

            ay5 = GGML_F16x_VEC_LOAD(y + i + 4 * ggml_f16_epr, 4);

            ax5 = GGML_F16x_VEC_LOAD(x[0] + i + 4*ggml_f16_epr, 4);

            sum_00 = GGML_F16x_VEC_FMA(sum_00, ax5, ay5);
            ax5 = GGML_F16x_VEC_LOAD(x[1] + i + 4*ggml_f16_epr, 4);
            sum_10 = GGML_F16x_VEC_FMA(sum_10, ax5, ay5);

            ay6 = GGML_F16x_VEC_LOAD(y + i + 5 * ggml_f16_epr, 5);

            ax6 = GGML_F16x_VEC_LOAD(x[0] + i + 5*ggml_f16_epr, 5);

            sum_01 = GGML_F16x_VEC_FMA(sum_01, ax6, ay6);
            ax6 = GGML_F16x_VEC_LOAD(x[1] + i + 5*ggml_f16_epr, 5);
            sum_11 = GGML_F16x_VEC_FMA(sum_11, ax6, ay6);

            ay7 = GGML_F16x_VEC_LOAD(y + i + 6 * ggml_f16_epr, 6);

            ax7 = GGML_F16x_VEC_LOAD(x[0] + i + 6*ggml_f16_epr, 6);

            sum_02 = GGML_F16x_VEC_FMA(sum_02, ax7, ay7);
            ax7 = GGML_F16x_VEC_LOAD(x[1] + i + 6*ggml_f16_epr, 6);
            sum_12 = GGML_F16x_VEC_FMA(sum_12, ax7, ay7);

            ay8 = GGML_F16x_VEC_LOAD(y + i + 7 * ggml_f16_epr, 7);

            ax8 = GGML_F16x_VEC_LOAD(x[0] + i + 7*ggml_f16_epr, 7);

            sum_03 = GGML_F16x_VEC_FMA(sum_03, ax8, ay8);
            ax8 = GGML_F16x_VEC_LOAD(x[1] + i + 7*ggml_f16_epr, 7);
            sum_13 = GGML_F16x_VEC_FMA(sum_13, ax8, ay8);
        }

        const int np2 = (n & ~(ggml_f16_epr - 1));
        for (int k = np; k < np2; k += ggml_f16_epr) {
            svfloat16_t ry = GGML_F16x_VEC_LOAD(y + k, 0);

            svfloat16_t rx = GGML_F16x_VEC_LOAD(x[0] + k, 0);
            sum_00 = GGML_F16x_VEC_FMA(sum_00, rx, ry);
            rx = GGML_F16x_VEC_LOAD(x[1] + k, 0);
            sum_10 = GGML_F16x_VEC_FMA(sum_10, rx, ry);
        }

        if (np2 < n) {
            svbool_t pg = svwhilelt_b16(np2, n);
            svfloat16_t hx_0 = svld1_f16(pg, (const __fp16 *)(x[0] + np2));
            svfloat16_t hx_1 = svld1_f16(pg, (const __fp16 *)(x[1] + np2));
            svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(y + np2));

            sum_00 = svmad_f16_x(pg, hx_0, hy, sum_00);
            sum_10 = svmad_f16_x(pg, hx_1, hy, sum_10);
        }
        GGML_F16x_VEC_REDUCE(sumf[0], sum_00, sum_01, sum_02, sum_03);
        GGML_F16x_VEC_REDUCE(sumf[1], sum_10, sum_11, sum_12, sum_13);
        np = n;
    #elif defined(__riscv_v_intrinsic)
        #if defined(__riscv_zvfh)
            size_t vl = __riscv_vsetvlmax_e32m4();

            // initialize accumulators to all zeroes
            vfloat32m4_t vsum0_0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vsum0_1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vsum1_0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            vfloat32m4_t vsum1_1 = __riscv_vfmv_v_f_f32m4(0.0f, vl);

            // calculate step size
            const size_t epr = __riscv_vsetvlmax_e16m2();
            const size_t step = epr * 2;
            int np = (n & ~(step - 1));

            // unroll by 2 along the row dimension
            for (int i = 0; i < np; i += step) {
                vfloat16m2_t ay0 = __riscv_vle16_v_f16m2((const _Float16 *)(y + i), epr);
                vfloat16m2_t ax0_0 = __riscv_vle16_v_f16m2((const _Float16 *)(x[0] + i), epr);
                vfloat16m2_t ax1_0 = __riscv_vle16_v_f16m2((const _Float16 *)(x[1] + i), epr);
                vsum0_0 = __riscv_vfwmacc_vv_f32m4(vsum0_0, ax0_0, ay0, epr);
                vsum1_0 = __riscv_vfwmacc_vv_f32m4(vsum1_0, ax1_0, ay0, epr);

                vfloat16m2_t ay1 = __riscv_vle16_v_f16m2((const _Float16 *)(y + i + epr), epr);
                vfloat16m2_t ax0_1 = __riscv_vle16_v_f16m2((const _Float16 *)(x[0] + i + epr), epr);
                vfloat16m2_t ax1_1 = __riscv_vle16_v_f16m2((const _Float16 *)(x[1] + i + epr), epr);
                vsum0_1 = __riscv_vfwmacc_vv_f32m4(vsum0_1, ax0_1, ay1, epr);
                vsum1_1 = __riscv_vfwmacc_vv_f32m4(vsum1_1, ax1_1, ay1, epr);
            }

            vfloat32m4_t vsum0 = __riscv_vfadd_vv_f32m4(vsum0_0, vsum0_1, vl);
            vfloat32m4_t vsum1 = __riscv_vfadd_vv_f32m4(vsum1_0, vsum1_1, vl);

            // leftovers
            for (int i = np; i < n; i += vl) {
                vl = __riscv_vsetvl_e16m2(n - i);
                vfloat16m2_t ay = __riscv_vle16_v_f16m2((const _Float16 *)(y + i), vl);
                vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16 *)(x[0] + i), vl);
                vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16 *)(x[1] + i), vl);

                vsum0 = __riscv_vfwmacc_vv_f32m4(vsum0, ax0, ay, vl);
                vsum1 = __riscv_vfwmacc_vv_f32m4(vsum1, ax1, ay, vl);
            }

            // reduce
            vl = __riscv_vsetvlmax_e32m2();
            vfloat32m2_t acc0_0 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(vsum0, 0),
                                        __riscv_vget_v_f32m4_f32m2(vsum0, 1), vl);
            vl = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t acc0_1 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(acc0_0, 0),
            __riscv_vget_v_f32m2_f32m1(acc0_0, 1), vl);
            vfloat32m1_t redsum0 = __riscv_vfredusum_vs_f32m1_f32m1(
                                        acc0_1, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);

            vl = __riscv_vsetvlmax_e32m2();
            vfloat32m2_t acc1_0 = __riscv_vfadd_vv_f32m2(__riscv_vget_v_f32m4_f32m2(vsum1, 0),
                                        __riscv_vget_v_f32m4_f32m2(vsum1, 1), vl);
            vl = __riscv_vsetvlmax_e32m1();
            vfloat32m1_t acc1_1 = __riscv_vfadd_vv_f32m1(__riscv_vget_v_f32m2_f32m1(acc1_0, 0),
                                        __riscv_vget_v_f32m2_f32m1(acc1_0, 1), vl);
            vfloat32m1_t redsum1 = __riscv_vfredusum_vs_f32m1_f32m1(
                                        acc1_1, __riscv_vfmv_v_f_f32m1(0.0f, 1), vl);
            sumf[0] = __riscv_vfmv_f_s_f32m1_f32(redsum0);
            sumf[1] = __riscv_vfmv_f_s_f32m1_f32(redsum1);
            np = n;
        #else
            const int np = 0;
        #endif
    #else
        const int np = (n & ~(GGML_F16_STEP - 1));

        GGML_F16_VEC sum[GGML_VEC_DOT_UNROLL][GGML_F16_ARR] = { { GGML_F16_VEC_ZERO } };

        GGML_F16_VEC ax[GGML_F16_ARR];
        GGML_F16_VEC ay[GGML_F16_ARR];

        for (int i = 0; i < np; i += GGML_F16_STEP) {
            for (int j = 0; j < GGML_F16_ARR; j++) {
                ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

                for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
                    ax[j] = GGML_F16_VEC_LOAD(x[k] + i + j*GGML_F16_EPR, j);

                    sum[k][j] = GGML_F16_VEC_FMA(sum[k][j], ax[j], ay[j]);
                }
            }
        }

        // reduce sum0..sum3 to sum0
        for (int k = 0; k < GGML_VEC_DOT_UNROLL; ++k) {
            GGML_F16_VEC_REDUCE(sumf[k], sum[k]);
        }
    #endif
#else
    // scalar path
    const int np = 0;
#endif
    // scalar and leftovers
    for (int i = np; i < n; ++i) {
        for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[j][i])*GGML_CPU_FP16_TO_FP32(y[i]));
        }
    }

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = (float)sumf[i];
    }
}

inline static void ggml_vec_mad_f32(const int n, float * GGML_RESTRICT y, const float * GGML_RESTRICT x, const float v) {
#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)

        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 8 * ggml_f32_epr; // choose 8 SVE registers
        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
        svfloat32_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
        for (int i = 0; i < np; i += ggml_f32_step) {

            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_FMA(ay1, ax1, vx);

            GGML_F32_VEC_STORE(y + i, ay1);

            ax2 = GGML_F32_VEC_LOAD(x + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_FMA(ay2, ax2, vx);

            GGML_F32_VEC_STORE(y + i + 1*ggml_f32_epr, ay2);

            ax3 = GGML_F32_VEC_LOAD(x + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_LOAD(y + i + 2*ggml_f32_epr);
            ay3 = GGML_F32_VEC_FMA(ay3, ax3, vx);

            GGML_F32_VEC_STORE(y + i + 2*ggml_f32_epr, ay3);

            ax4 = GGML_F32_VEC_LOAD(x + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_LOAD(y + i + 3*ggml_f32_epr);
            ay4 = GGML_F32_VEC_FMA(ay4, ax4, vx);

            GGML_F32_VEC_STORE(y + i + 3*ggml_f32_epr, ay4);

            ax5 = GGML_F32_VEC_LOAD(x + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_LOAD(y + i + 4*ggml_f32_epr);
            ay5 = GGML_F32_VEC_FMA(ay5, ax5, vx);

            GGML_F32_VEC_STORE(y + i + 4*ggml_f32_epr, ay5);

            ax6 = GGML_F32_VEC_LOAD(x + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_LOAD(y + i + 5*ggml_f32_epr);
            ay6 = GGML_F32_VEC_FMA(ay6, ax6, vx);

            GGML_F32_VEC_STORE(y + i + 5*ggml_f32_epr, ay6);

            ax7 = GGML_F32_VEC_LOAD(x + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_LOAD(y + i + 6*ggml_f32_epr);
            ay7 = GGML_F32_VEC_FMA(ay7, ax7, vx);

            GGML_F32_VEC_STORE(y + i + 6*ggml_f32_epr, ay7);

            ax8 = GGML_F32_VEC_LOAD(x + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_LOAD(y + i + 7*ggml_f32_epr);
            ay8 = GGML_F32_VEC_FMA(ay8, ax8, vx);

            GGML_F32_VEC_STORE(y + i + 7*ggml_f32_epr, ay8);
        }
        // leftovers
        // Since 8 unrolls are done in above loop, leftovers lie in range [0, ggml_f32_step] which is handled in below loop
        const int np2 = (n & ~(ggml_f32_epr - 1));
        for (int i = np; i < np2; i += ggml_f32_epr) {
            ax1 = GGML_F32_VEC_LOAD(x + i);
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_FMA(ay1, ax1, vx);

            GGML_F32_VEC_STORE(y + i, ay1);
        }
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        if (np2 < n) {
            svbool_t pg =svwhilelt_b32(np2, n);
            ax1 = svld1_f32(pg, x + np2);
            ay1 = svld1_f32(pg, y + np2);
            ay1 = svmad_f32_m(pg, ax1, vx, ay1);

            svst1_f32(pg, y + np2, ay1);
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[i], avl);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(ax, v, ay, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        GGML_F32_VEC ax[GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] += x[i]*v;
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
#endif
}

inline static void ggml_vec_mad_f16(const int n, ggml_fp16_t * GGML_RESTRICT y, const ggml_fp16_t * GGML_RESTRICT x, const float v) {
#if defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
    const int sve_register_length = svcntb() * 8;
    const int ggml_f16_epr = sve_register_length / 16;
    const int ggml_f16_step = 8 * ggml_f16_epr;

    GGML_F16x_VEC vx = GGML_F16x_VEC_SET1(v);

    int np = (n & ~(ggml_f16_step - 1));

    svfloat16_t ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8;
    svfloat16_t ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8;
    for (int i = 0; i < np; i += ggml_f16_step) {
        ax1 = GGML_F16x_VEC_LOAD(x + i + 0 * ggml_f16_epr, 0);
        ay1 = GGML_F16x_VEC_LOAD(y + i + 0 * ggml_f16_epr, 0);
        ay1 = GGML_F16x_VEC_FMA(ay1, ax1, vx);

        GGML_F16x_VEC_STORE(y + i + 0 * ggml_f16_epr, ay1, 0);

        ax2 = GGML_F16x_VEC_LOAD(x + i + 1 * ggml_f16_epr, 1);
        ay2 = GGML_F16x_VEC_LOAD(y + i + 1 * ggml_f16_epr, 1);
        ay2 = GGML_F16x_VEC_FMA(ay2, ax2, vx);

        GGML_F16x_VEC_STORE(y + i + 1 * ggml_f16_epr, ay2, 1);

        ax3 = GGML_F16x_VEC_LOAD(x + i + 2 * ggml_f16_epr, 2);
        ay3 = GGML_F16x_VEC_LOAD(y + i + 2 * ggml_f16_epr, 2);
        ay3 = GGML_F16x_VEC_FMA(ay3, ax3, vx);

        GGML_F16x_VEC_STORE(y + i + 2 * ggml_f16_epr, ay3, 2);

        ax4 = GGML_F16x_VEC_LOAD(x + i + 3 * ggml_f16_epr, 3);
        ay4 = GGML_F16x_VEC_LOAD(y + i + 3 * ggml_f16_epr, 3);
        ay4 = GGML_F16x_VEC_FMA(ay4, ax4, vx);

        GGML_F16x_VEC_STORE(y + i + 3 * ggml_f16_epr, ay4, 3);

        ax5 = GGML_F16x_VEC_LOAD(x + i + 4 * ggml_f16_epr, 4);
        ay5 = GGML_F16x_VEC_LOAD(y + i + 4 * ggml_f16_epr, 4);
        ay5 = GGML_F16x_VEC_FMA(ay5, ax5, vx);

        GGML_F16x_VEC_STORE(y + i + 4 * ggml_f16_epr, ay5, 4);

        ax6 = GGML_F16x_VEC_LOAD(x + i + 5 * ggml_f16_epr, 5);
        ay6 = GGML_F16x_VEC_LOAD(y + i + 5 * ggml_f16_epr, 5);
        ay6 = GGML_F16x_VEC_FMA(ay6, ax6, vx);

        GGML_F16x_VEC_STORE(y + i + 5 * ggml_f16_epr, ay6, 5);

        ax7 = GGML_F16x_VEC_LOAD(x + i + 6 * ggml_f16_epr, 6);
        ay7 = GGML_F16x_VEC_LOAD(y + i + 6 * ggml_f16_epr, 6);
        ay7 = GGML_F16x_VEC_FMA(ay7, ax7, vx);

        GGML_F16x_VEC_STORE(y + i + 6 * ggml_f16_epr, ay7, 6);

        ax8 = GGML_F16x_VEC_LOAD(x + i + 7 * ggml_f16_epr, 7);
        ay8 = GGML_F16x_VEC_LOAD(y + i + 7 * ggml_f16_epr, 7);
        ay8 = GGML_F16x_VEC_FMA(ay8, ax8, vx);

        GGML_F16x_VEC_STORE(y + i + 7 * ggml_f16_epr, ay8, 7);
    }
    const int np2 = (n & ~(ggml_f16_epr - 1));
    for (int k = np; k < np2; k += ggml_f16_epr) {
        svfloat16_t rx = GGML_F16x_VEC_LOAD(x + k, 0);
        svfloat16_t ry = GGML_F16x_VEC_LOAD(y + k, 0);
        ry = GGML_F16x_VEC_FMA(ry, rx, vx);

        GGML_F16x_VEC_STORE(y + k, ry, 0);
    }

    if (np2 < n) {
        svbool_t pg = svwhilelt_b16(np2, n);
        svfloat16_t hx = svld1_f16(pg, (const __fp16 *)(x + np2));
        svfloat16_t hy = svld1_f16(pg, (const __fp16 *)(y + np2));
        hy = svmad_f16_x(pg, hx, vx, hy);
        svst1_f16(pg, (__fp16 *)(y + np2), hy);
    }
    np = n;
#elif defined(__riscv_v_intrinsic) // implies __riscv_v_intrinsic
    #if defined (__riscv_zvfh)
        const ggml_fp16_t s = GGML_CPU_FP32_TO_FP16(v);
        const _Float16 scale = *(const _Float16*)(&s);

        // calculate step size
        const int epr = __riscv_vsetvlmax_e16m4();
        const int step = epr * 2;
        int np = (n & ~(step - 1));

        // unroll by 2
        for (int i = 0; i < np; i += step) {
            vfloat16m4_t ax0 = __riscv_vle16_v_f16m4((const _Float16*)x + i, epr);
            vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16*)y + i, epr);
            ay0 = __riscv_vfmacc_vf_f16m4(ay0, scale, ax0, epr);
            __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, epr);
            __asm__ __volatile__ ("" ::: "memory");

            vfloat16m4_t ax1 = __riscv_vle16_v_f16m4((const _Float16*)x + i + epr, epr);
            vfloat16m4_t ay1 = __riscv_vle16_v_f16m4((const _Float16*)y + i + epr, epr);
            ay1 = __riscv_vfmacc_vf_f16m4(ay1, scale, ax1, epr);
            __riscv_vse16_v_f16m4((_Float16*)y + i + epr, ay1, epr);
            __asm__ __volatile__ ("" ::: "memory");
        }

        // leftovers
        int vl;
        for (int i = np; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m4(n - i);
            vfloat16m4_t ax0 = __riscv_vle16_v_f16m4((const _Float16*)x + i, vl);
            vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16*)y + i, vl);
            ay0 = __riscv_vfmacc_vf_f16m4(ay0, scale, ax0, vl);
            __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, vl);
        }
        np = n;
    #else
        // fall to scalar path
        const int np = 0;
    #endif
#elif defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_FMA(ay[j], ax[j], vx);

            GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
        }
    }
#else
    // scalar path
    const int np = 0;
#endif

    // scalar and leftovers
    for (int i = np; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i]) + GGML_CPU_FP16_TO_FP32(x[i])*v);
    }
}

// xs and vs are byte strides of x and v
inline static void ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float * GGML_RESTRICT y, const float * GGML_RESTRICT xv, const float * GGML_RESTRICT vv) {

    const float * GGML_RESTRICT x[GGML_VEC_MAD_UNROLL];
    const float * GGML_RESTRICT v[GGML_VEC_MAD_UNROLL];

    for (int i = 0; i < GGML_VEC_MAD_UNROLL; ++i) {
        x[i] = (const float *) ((const char *) xv + i*xs);
        v[i] = (const float *) ((const char *) vv + i*vs);
    }

#if defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        // scalar Route to scalar implementation       //TODO: Write SVE code
        for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            for (int i = 0; i < n; ++i) {
                y[i] += x[k][i]*v[k][0];
            }
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            for (int k = 0; k < GGML_VEC_MAD_UNROLL; k++) {
                vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[k][i], avl);
                ay = __riscv_vfmadd_vf_f32m8(ax, v[k][0], ay, avl);
            }
            __riscv_vse32_v_f32m8(&y[i], ay, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx[GGML_VEC_MAD_UNROLL];

        for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            vx[k] = GGML_F32_VEC_SET1(v[k][0]);
        }

        GGML_F32_VEC ax[GGML_VEC_MAD_UNROLL][GGML_F32_ARR];
        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

                for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
                    ax[k][j] = GGML_F32_VEC_LOAD(x[k] + i + j*GGML_F32_EPR);
                    ay[j] = GGML_F32_VEC_FMA(ay[j], ax[k][j], vx[k]);
                }

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
            for (int i = np; i < n; ++i) {
                y[i] += x[k][i]*v[k][0];
            }
        }
    #endif
#else
    // scalar
    for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = 0; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
#endif
}

inline static void ggml_vec_mad1_f32(const int n, float * y, const float * x, const float s, const float b) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmsa(x, 1, &s, &b, y, 1, n);
#elif defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        // scalar ; TODO: Write SVE code
        for (int i = 0; i < n; ++i) {
            y[i] = x[i]*s + b;
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ax = __riscv_vle32_v_f32m8(&x[i], avl);
            vfloat32m8_t vb = __riscv_vfmv_v_f_f32m8(b, avl);
            vfloat32m8_t ny = __riscv_vfmadd_vf_f32m8(ax, s, vb, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vs = GGML_F32_VEC_SET1(s);
        GGML_F32_VEC vb = GGML_F32_VEC_SET1(b);

        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_FMA(vb, ay[j], vs);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] = x[i]*s + b;
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = x[i]*s + b;
    }
#endif
}

//inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) { for (int i = 0; i < n; ++i) y[i] *= v;          }
inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
    #if defined(__ARM_FEATURE_SVE)
        const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
        const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
        const int ggml_f32_step = 2 * ggml_f32_epr;

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
        const int np = (n & ~(ggml_f32_step - 1));
        svfloat32_t ay1;
        svfloat32_t ay2;
        for (int i = 0; i < np; i += ggml_f32_step) {
            ay1 = GGML_F32_VEC_LOAD(y + i);
            ay1 = GGML_F32_VEC_MUL(ay1, vx);
            GGML_F32_VEC_STORE(y + i, ay1);

            ay2 = GGML_F32_VEC_LOAD(y + i + 1*ggml_f32_epr);
            ay2 = GGML_F32_VEC_MUL(ay2, vx);
            GGML_F32_VEC_STORE(y + i + 1*ggml_f32_epr, ay2);
        }
        // leftovers
        // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
        for (int i = np; i < n; i += ggml_f32_epr) {
            svbool_t pg = svwhilelt_b32(i, n);
            ay1 = svld1_f32(pg, y + i);
            ay1 = svmul_f32_m(pg, ay1, vx);
            svst1_f32(pg, y + i, ay1);
        }
    #elif defined(__riscv_v_intrinsic)
        for (int i = 0, avl; i < n; i += avl) {
            avl = __riscv_vsetvl_e32m8(n - i);
            vfloat32m8_t ay = __riscv_vle32_v_f32m8(&y[i], avl);
            vfloat32m8_t ny = __riscv_vfmul_vf_f32m8(ay, v, avl);
            __riscv_vse32_v_f32m8(&y[i], ny, avl);
        }
    #else
        const int np = (n & ~(GGML_F32_STEP - 1));

        GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

        GGML_F32_VEC ay[GGML_F32_ARR];

        for (int i = 0; i < np; i += GGML_F32_STEP) {
            for (int j = 0; j < GGML_F32_ARR; j++) {
                ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
                ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

                GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
            }
        }

        // leftovers
        for (int i = np; i < n; ++i) {
            y[i] *= v;
        }
    #endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void ggml_vec_scale_f16(const int n, ggml_fp16_t * y, const float v) {
#if defined(GGML_SIMD) && defined(__ARM_FEATURE_SVE)
    const int sve_register_length = svcntb() * 8;
    const int ggml_f16_epr = sve_register_length / 16;
    const int ggml_f16_step = 2 * ggml_f16_epr;

    GGML_F16x_VEC vx =  GGML_F16x_VEC_SET1(v);
    int np = (n & ~(ggml_f16_step - 1));
    svfloat16_t ay1, ay2;

    for (int i = 0; i < np; i += ggml_f16_step) {
        ay1 = GGML_F16x_VEC_LOAD(y + i + 0*ggml_f16_epr, 0);
        ay1 = GGML_F16x_VEC_MUL(ay1, vx);
        GGML_F16x_VEC_STORE(y + i + 0*ggml_f16_epr, ay1, 0);

        ay2 = GGML_F16x_VEC_LOAD(y + i + 1*ggml_f16_epr, 1);
        ay2 = GGML_F16x_VEC_MUL(ay2, vx);
        GGML_F16x_VEC_STORE(y + i + 1*ggml_f16_epr, ay2, 1);
    }
    // leftovers
    // maximum number of leftover elements will be less that ggmlF_16x_epr. Apply predicated svmad on available elements only
    if (np < n) {
        svbool_t pg = svwhilelt_b16(np, n);
        svfloat16_t hy = svld1_f16(pg, (__fp16 *)(y + np));
        svfloat16_t out = svmul_f16_m(pg, hy, vx);
        svst1_f16(pg, (__fp16 *)(y + np), out);
    }
    np = n;
#elif defined(__riscv_v_intrinsic)
    #if defined(__riscv_zvfh)
        const ggml_fp16_t s = GGML_CPU_FP32_TO_FP16(v);
        const _Float16 scale = *(const _Float16*)(&s);

        // calculate step size
        const int epr = __riscv_vsetvlmax_e16m4();
        const int step = epr * 2;
        int np = (n & ~(step - 1));

        // unroll by 2
        for (int i = 0; i < np; i += step) {
            vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16*)y + i, epr);
            ay0 = __riscv_vfmul_vf_f16m4(ay0, scale, epr);
            __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, epr);
            __asm__ __volatile__ ("" ::: "memory");

            vfloat16m4_t ay1 = __riscv_vle16_v_f16m4((const _Float16*)y + i + epr, epr);
            ay1 = __riscv_vfmul_vf_f16m4(ay1, scale, epr);
            __riscv_vse16_v_f16m4((_Float16*)y + i + epr, ay1, epr);
            __asm__ __volatile__ ("" ::: "memory");
        }

        // leftovers
        int vl;
        for (int i = np; i < n; i += vl) {
            vl = __riscv_vsetvl_e16m4(n - i);
            vfloat16m4_t ay0 = __riscv_vle16_v_f16m4((const _Float16*)y + i, vl);
            ay0 = __riscv_vfmul_vf_f16m4(ay0, scale, vl);
            __riscv_vse16_v_f16m4((_Float16*)y + i, ay0, vl);
        }
        np = n;
    #else
        // fall to scalar path
        const int np = 0;
    #endif
#elif defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_MUL(ay[j], vx);

            GGML_F16_VEC_STORE(y + i + j*GGML_F16_EPR, ay, j);
        }
    }
#else
    // scalar path
    const int np = 0;
#endif
    // scalar and leftovers
    for (int i = np; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i])*v);
    }
}

inline static void ggml_vec_norm_f32 (const int n, float * s, const float * x) { ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);   }
inline static void ggml_vec_sqr_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
inline static void ggml_vec_sqr_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v*v);
    }
}
inline static void ggml_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
inline static void ggml_vec_sqrt_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sqrtf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_log_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = logf(x[i]);  }
inline static void ggml_vec_log_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(logf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_sin_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sinf(x[i]);  }
inline static void ggml_vec_sin_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sinf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_cos_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = cosf(x[i]);  }
inline static void ggml_vec_cos_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(cosf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_abs_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
inline static void ggml_vec_abs_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(fabsf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_sgn_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
inline static void ggml_vec_sgn_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f));
    }
}
inline static void ggml_vec_step_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
inline static void ggml_vec_step_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16((GGML_CPU_FP16_TO_FP32(x[i]) > 0.f) ? 1.f : 0.f);
    }
}
inline static void ggml_vec_tanh_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = tanhf(x[i]);  }
inline static void ggml_vec_tanh_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(tanhf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_elu_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : expm1f(x[i]); }
inline static void ggml_vec_elu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        const float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v : expm1f(v));
    }
}
inline static void ggml_vec_relu_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }
inline static void ggml_vec_relu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v : 0.f);
    }
}
inline static void ggml_vec_leaky_relu_f32 (const int n, float * y, const float * x, const float ns) { for (int i = 0; i < n; ++i) y[i] = ((x[i] > 0.f) ? x[i] : 0.f) + ns * ((x[i] < 0.0f) ? x[i] : 0.f); }
inline static void ggml_vec_leaky_relu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float ns) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(((v > 0.f) ? v : 0.f) + ns * ((v < 0.0f) ? v : 0.f));
    }
}
inline static void ggml_vec_sigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
inline static void ggml_vec_sigmoid_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(1.f / (1.f + expf(-GGML_CPU_FP16_TO_FP32(x[i]))));
    }
}
// TODO: optimize performance
inline static void ggml_vec_hardswish_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void ggml_vec_hardswish_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v * fminf(1.0f, fmaxf(0.0f, (v + 3.0f) / 6.0f)));
    }
}
inline static void ggml_vec_hardsigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void ggml_vec_hardsigmoid_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(fminf(1.0f, fmaxf(0.0f, (GGML_CPU_FP16_TO_FP32(x[i]) + 3.0f) / 6.0f)));
    }
}
inline static void ggml_vec_exp_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = expf(x[i]); }
inline static void ggml_vec_exp_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(expf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}

static const float GELU_COEF_A     = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
static const float SQRT_2_INV      = 0.70710678118654752440084436210484f;

inline static float ggml_gelu_f32(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

inline static void ggml_vec_gelu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_table_gelu_f16[i16[i]];
    }
}

inline static void ggml_vec_gelu_erf_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float xi = GGML_CPU_FP16_TO_FP32(x[i]);
        float res = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
        y[i] = GGML_CPU_FP32_TO_FP16(res);
    }
}

#ifdef GGML_GELU_FP16
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } else if (x[i] >= 10.0f) {
            y[i] = x[i];
        } else {
            ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]);
        }
    }
}
#else
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]);
    }
}
#endif

inline static void ggml_vec_gelu_erf_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        y[i] = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
    }
}

inline static float ggml_gelu_quick_f32(float x) {
    return x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)));
}

inline static void ggml_vec_gelu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_table_gelu_quick_f16[i16[i]];
    }
}

#ifdef GGML_GELU_QUICK_FP16
inline static void ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[t]);
    }
}
#else
inline static void ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_quick_f32(x[i]);
    }
}
#endif

// Sigmoid Linear Unit (SiLU) function
inline static float ggml_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}
inline static ggml_fp16_t ggml_silu_f16(ggml_fp16_t x) {
    float v = GGML_CPU_FP16_TO_FP32(x);
    return GGML_CPU_FP32_TO_FP16(v/(1.0f + expf(-v)));
}

#if __FINITE_MATH_ONLY__
#error "some routines in ggml.c.inc require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
#error "ref: https://github.com/ggml-org/llama.cpp/pull/7154#issuecomment-2143844461"
#endif

/* Below function was borrowed from the GitHub repository:
https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/common.h.incpp */
#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
    inline static svfloat32_t exp_ps_sve(svbool_t pg, svfloat32_t src) {
        // Constants
        const svfloat32_t log2_e = svdup_n_f32(1.4426950409f);
        const svfloat32_t ln2 = svdup_n_f32(0.6931473921f);
        const svfloat32_t half_ln2_sq = svdup_n_f32(0.2413862043f);
        const svuint32_t not_mask17 = svdup_n_u32(~((1u << 17) - 1));
        const svfloat32_t one = svdup_n_f32(1.0f);
        const svfloat32_t inactive1 = svdup_n_f32(0.0f);
        const svint32_t inactive2 = svdup_n_s32(0);

        // Algorithm starts here
        svfloat32_t t0 = svmul_f32_m(pg, src, log2_e);  // y = x * log2(e)
        svfloat32_t t1 = svrintm_f32_m(inactive1, pg, t0);         // rount to int (float)
        svint32_t t2 = svcvt_s32_f32_m(inactive2, pg, t1);         // n

        t1 = svsub_f32_m(pg, t0, t1);   // a = y - floor(y)
        t1 = svadd_f32_m(pg, t1, one);  // b = a + 1

        svuint32_t t3 = svlsr_n_u32_m(pg, svreinterpret_u32_f32(t1), 17);  // v = b >> 17 (u32)
        svfloat32_t t4 = svexpa_f32(t3);                                   // c = fexpa(v)
        t4 = svscale_f32_m(pg, t4, t2);                                    // fexpa(v) * 2^(n)

        // and_(t2.d, t1.d, not_mask17.d)
        svfloat32_t t5 = svreinterpret_f32_u32(svand_u32_m(pg, svreinterpret_u32_f32(t1), not_mask17));
        t5 = svsub_f32_m(pg, t1, t5);                // z
        t0 = svmla_f32_m(pg, ln2, t5, half_ln2_sq);  // ln2 + half_ln2_sq * z
        t0 = svmla_f32_m(pg, one, t5, t0);           // 1 + (ln2 * z) + (half_ln2_sq * z * z)
        t0 = svmul_f32_m(pg, t0, t4);                // Final result

        return t0;
    }
#endif

#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)

inline static svfloat32_t ggml_v_expf(svbool_t pg, svfloat32_t x) {
    const svfloat32_t r = svdup_n_f32_x(pg, 0x1.8p23f);
    const svfloat32_t z = svmla_n_f32_x(pg, r, x, 0x1.715476p+0f);
    const svfloat32_t n = svsub_f32_x(pg, z, r);
    const svfloat32_t b = svmls_n_f32_x(pg, svmls_n_f32_x(pg, x, n, 0x1.62e4p-1f), n, 0x1.7f7d1cp-20f);
    const svuint32_t e = svlsl_n_u32_x(pg, svreinterpret_u32_f32(z), 23);
    const svfloat32_t k = svreinterpret_f32_u32(svadd_u32_x(pg, e, svreinterpret_u32_f32(svdup_n_f32_x(pg, 1))));
    const svbool_t c = svacgt_n_f32(pg, n, 126);
    const svfloat32_t u = svmul_f32_x(pg, b, b);
    const svfloat32_t j = svmla_f32_x(pg,
        svmul_n_f32_x(pg, b, 0x1.ffffecp-1f),
        svmla_f32_x(pg, svmla_f32_x(pg, svdup_n_f32_x(pg, 0x1.fffdb6p-2f), svdup_n_f32_x(pg, 0x1.555e66p-3f), b),
                        svmla_f32_x(pg, svdup_n_f32_x(pg, 0x1.573e2ep-5f), svdup_n_f32_x(pg, 0x1.0e4020p-7f), b), u), u);
    const svuint32_t d = svdup_n_u32_z(svcmple_n_f32(pg, n, 0.0), 0x82000000);
    const svfloat32_t s1 = svreinterpret_f32_u32(svadd_n_u32_x(pg, d, 0x7f000000));
    const svfloat32_t s2 = svreinterpret_f32_u32(svsub_u32_x(pg, e, d));
    return svsel_f32(svacgt_f32(pg, n, svdup_n_f32_x(pg, 192)), svmul_f32_x(pg, s1, s1),
                     svsel_f32(c, svmul_f32_x(pg, svmla_f32_x(pg, s2, s2, j), s1), svmla_f32_x(pg, k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static svfloat32_t ggml_v_silu(svbool_t pg, svfloat32_t x) {
    const svfloat32_t one = svdup_n_f32_x(pg, 1.0f);
    const svfloat32_t zero = svdup_n_f32_x(pg, 0.0f);
    const svfloat32_t neg_x = svsub_f32_x(pg, zero, x);
    const svfloat32_t exp_neg_x = ggml_v_expf(pg, neg_x);
    const svfloat32_t one_plus_exp_neg_x = svadd_f32_x(pg, one, exp_neg_x);
    return svdiv_f32_x(pg, x, one_plus_exp_neg_x);
}

#elif defined(__ARM_NEON) && defined(__aarch64__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static float32x4_t ggml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                    vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c)))
        return vfmaq_f32(k, j, k);
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static float32x4_t ggml_v_silu(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t neg_x = vsubq_f32(zero, x);
    const float32x4_t exp_neg_x = ggml_v_expf(neg_x);
    const float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
    return vdivq_f32(x, one_plus_exp_neg_x);
}

#elif defined(__AVX512F__) && defined(__AVX512DQ__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m512 ggml_v_expf(__m512 x) {
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                       _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __mmask16 d =
      _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512 u = _mm512_mul_ps(b, b);
  const __m512 j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                      _mm512_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                      _mm512_set1_ps(0x1.fffdb6p-2f))),
      u,
      _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  const __m512 res = _mm512_scalef_ps(j, n);
  if (_mm512_kortestz(d, d))
    return res;
  const __m512 zero = _mm512_setzero_ps();
  const __m512 alt = _mm512_mask_blend_ps(
      _mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  return _mm512_mask_blend_ps(d, res, alt);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m512 ggml_v_silu(__m512 x) {
    const __m512 one = _mm512_set1_ps(1);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 neg_x = _mm512_sub_ps(zero, x);
    const __m512 exp_neg_x = ggml_v_expf(neg_x);
    const __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
    return _mm512_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__AVX2__) && defined(__FMA__)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m256 ggml_v_expf(__m256 x) {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                    _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                                                   _mm256_set1_ps(0x1.573e2ep-5f)), u,
                                                   _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                                                   _mm256_set1_ps(0x1.fffdb6p-2f))),
                                   u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c)))
    return _mm256_fmadd_ps(j, k, k);
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(0x82000000u));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m256 ggml_v_silu(__m256 x) {
    const __m256 one = _mm256_set1_ps(1);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 neg_x = _mm256_sub_ps(zero, x);
    const __m256 exp_neg_x = ggml_v_expf(neg_x);
    const __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
    return _mm256_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__SSE2__) // __AVX2__ / __ARM_NEON

#if defined(__FMA__)
#define MADD128(x, y, z) _mm_fmadd_ps(x, y, z)
#define NMADD128(x, y, z) _mm_fnmadd_ps(x, y, z)
#else
#define MADD128(x, y, z) _mm_add_ps(_mm_mul_ps(x, y), z)
#define NMADD128(x, y, z) _mm_sub_ps(z, _mm_mul_ps(x, y))
#endif

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static __m128 ggml_v_expf(__m128 x) {
    const __m128 r = _mm_set1_ps(0x1.8p23f);
    const __m128 z = MADD128(x, _mm_set1_ps(0x1.715476p+0f), r);
    const __m128 n = _mm_sub_ps(z, r);
    const __m128 b =
        NMADD128(n, _mm_set1_ps(0x1.7f7d1cp-20f), NMADD128(n, _mm_set1_ps(0x1.62e4p-1f), x));
    const __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
    const __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
    const __m128i c =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
    const __m128 u = _mm_mul_ps(b, b);
    const __m128 j =
        MADD128(MADD128(MADD128(_mm_set1_ps(0x1.0e4020p-7f), b, _mm_set1_ps(0x1.573e2ep-5f)), u,
                        MADD128(_mm_set1_ps(0x1.555e66p-3f), b, _mm_set1_ps(0x1.fffdb6p-2f))),
                u, _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
    if (!_mm_movemask_epi8(c))
        return MADD128(j, k, k);
    const __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),
                                    _mm_set1_epi32(0x82000000u));
    const __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
    const __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
    const __m128i d =
        _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
    return _mm_or_ps(
        _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
        _mm_andnot_ps(_mm_castsi128_ps(d),
                      _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(MADD128(s2, j, s2), s1)),
                                _mm_andnot_ps(_mm_castsi128_ps(c), MADD128(k, j, k)))));
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static __m128 ggml_v_silu(__m128 x) {
    const __m128 one = _mm_set1_ps(1);
    const __m128 zero = _mm_setzero_ps();
    const __m128 neg_x = _mm_sub_ps(zero, x);
    const __m128 exp_neg_x = ggml_v_expf(neg_x);
    const __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
    return _mm_div_ps(x, one_plus_exp_neg_x);
}

#elif defined(__riscv_v_intrinsic)

// adapted from arm limited optimized routine
// the maximum error is 1.45358 plus 0.5 ulps
// numbers above 88.38 will flush to infinity
// numbers beneath -103.97 will flush to zero
inline static vfloat32m2_t ggml_v_expf_m2(vfloat32m2_t x, int vl) {
    const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
#ifdef __riscv_xtheadvector
    // workaround for compiler bug (gcc 14.3.0: Error: unrecognized opcode `th.vmv1r.v v2,v4')
    vfloat32m2_t z = __riscv_vfadd_vf_f32m2(r, 0.0f, vl);
    z = __riscv_vfmacc_vf_f32m2(z, 0x1.715476p+0f, x, vl);
#else
    const vfloat32m2_t z = __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
#endif
    const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
    const vfloat32m2_t b = __riscv_vfnmsac_vf_f32m2(__riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl),
                                                    0x1.7f7d1cp-20f, n, vl);
    const vuint32m2_t e = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(z), 23, vl);
    const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(e, 0x3f800000, vl)); // 1.0f
    const vbool16_t c = __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 126.0f, vl);
    const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
    const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(
        __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl),
        __riscv_vfmacc_vv_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl), 0x1.555e66p-3f, b, vl),
            __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl), 0x1.0e4020p-7f, b, vl),
            u, vl), u, vl);
    if (!__riscv_vcpop_m_b16(c, vl))
        return __riscv_vfmacc_vv_f32m2(k, j, k, vl);
    const vbool16_t  dm = __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
    const vuint32m2_t d = __riscv_vmerge_vxm_u32m2(__riscv_vmv_v_x_u32m2(0, vl), 0x82000000, dm, vl);
    const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vadd_vx_u32m2(d, 0x7f000000, vl));
    const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(__riscv_vsub_vv_u32m2(e, d, vl));
    const vfloat32m2_t r1 = __riscv_vmerge_vvm_f32m2(
        __riscv_vfmacc_vv_f32m2(k, k, j, vl),
        __riscv_vfmul_vv_f32m2(__riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl),
        c, vl);
    return __riscv_vmerge_vvm_f32m2(
        r1, __riscv_vfmul_vv_f32m2(s1, s1, vl),
        __riscv_vmfgt_vf_f32m2_b16(__riscv_vfabs_v_f32m2(n, vl), 192.0f, vl),
        vl);
}

// computes silu x/(1+exp(-x)) in single precision vector
inline static vfloat32m2_t ggml_v_silu_m2(vfloat32m2_t x, int vl) {
    const vfloat32m2_t neg_x = __riscv_vfneg_v_f32m2(x, vl);
    const vfloat32m2_t exp_neg_x = ggml_v_expf_m2(neg_x, vl);
    const vfloat32m2_t one_plus_exp_neg_x = __riscv_vfadd_vf_f32m2(exp_neg_x, 1.0f, vl);
    return __riscv_vfdiv_vv_f32m2(x, one_plus_exp_neg_x, vl);
}

#endif // __ARM_NEON / __AVX2__ / __SSE2__ / __riscv_v_intrinsic

inline static void ggml_vec_silu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_silu_f16(x[i]);
    }
}

inline static float ggml_silu_backward_f32(float x, float dy) {
    const float s = 1.0f/(1.0f + expf(-x));
    return dy*s*(1.0f + x*(1.0f - s));
}

inline static ggml_fp16_t ggml_silu_backward_f16(ggml_fp16_t x, ggml_fp16_t dy) {
    const float v = GGML_CPU_FP16_TO_FP32(x);
    const float s = 1.0f/(1.0f + expf(-v));
    return GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(dy)*s*(1.0f + v*(1.0f - s)));
}

inline static void ggml_vec_silu_backward_f32(const int n, float * dx, const float * x, const float * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = ggml_silu_backward_f32(x[i], dy[i]);
    }
}

inline static void ggml_vec_silu_backward_f16(const int n, ggml_fp16_t * dx, const ggml_fp16_t * x, const ggml_fp16_t * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = ggml_silu_backward_f16(x[i], dy[i]);
    }
}

inline static void ggml_vec_reglu_f32 (const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        y[i] = (x[i] > 0.f) ? x[i] * g[i] : 0.f;
    }
}

inline static void ggml_vec_reglu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v * GGML_CPU_FP16_TO_FP32(g[i]) : 0.f);
    }
}

#ifdef GGML_GELU_FP16
inline static void ggml_vec_geglu_f32(const int n, float * y, const float * x, const float * g) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } else if (x[i] >= 10.0f) {
            y[i] = x[i] * g[i];
        } else {
            ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]) * g[i];
        }
    }
}
#else
inline static void ggml_vec_geglu_f32(const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]) * g[i];
    }
}
#endif

inline static void ggml_vec_geglu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[i16[i]]) * v);
    }
}

void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g);

inline static void ggml_vec_swiglu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float xi = GGML_CPU_FP16_TO_FP32(x[i]);
        float gi = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((xi/(1.0f + expf(-xi))) * gi);
    }
}

inline static void ggml_vec_geglu_erf_f32(const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        y[i] = 0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * g[i];
    }
}

inline static void ggml_vec_geglu_erf_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float xi = GGML_CPU_FP16_TO_FP32(x[i]);
        float gi = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * gi);
    }
}

#ifdef GGML_GELU_QUICK_FP16
inline static void ggml_vec_geglu_quick_f32(const int n, float * y, const float * x, const float * g) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[t]) * g[i];
    }
}
#else
inline static void ggml_vec_geglu_quick_f32(const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_quick_f32(x[i]) * g[i];
    }
}
#endif

inline static void ggml_vec_geglu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[i16[i]]) * v);
    }
}

inline static void ggml_vec_sum_f32(const int n, float * s, const float * x) {
#ifndef GGML_USE_ACCELERATE
    ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (ggml_float)x[i];
    }
    *s = (float)sum;
#else
    vDSP_sve(x, 1, s, n);
#endif
}

inline static void ggml_vec_cumsum_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            y[i] = x[i];
        } else {
            y[i] = y[i - 1] + x[i];
        }
    }
}

inline static void ggml_vec_sum_f32_ggf(const int n, ggml_float * s, const float * x) {
    ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (ggml_float)x[i];
    }
    *s = sum;
}

inline static void ggml_vec_sum_f16_ggf(const int n, float * s, const ggml_fp16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += GGML_CPU_FP16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void ggml_vec_sum_bf16_ggf(const int n, float * s, const ggml_bf16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += GGML_BF16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void ggml_vec_max_f32(const int n, float * s, const float * x) {
#ifndef GGML_USE_ACCELERATE
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
#else
    vDSP_maxv(x, 1, s, n);
#endif
}

inline static void ggml_vec_norm_inv_f32(const int n, float * s, const float * x) {
    ggml_vec_norm_f32(n, s, x);
    *s = 1.f/(*s);
}

inline static void ggml_vec_argmax_f32(const int n, int * s, const float * x) {
    float max = -INFINITY;
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
        if (max == x[i]) { idx = i; }
    }
    *s = idx;
}

#ifdef __cplusplus
}
#endif

#include "ggml.h.inc"

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)
#define CACHE_LINE_SIZE std::hardware_destructive_interference_size
#else
#if defined(__POWER9_VECTOR__)
#define CACHE_LINE_SIZE 128
#elif defined(__VXE__) || defined(__VXE2__)
#define CACHE_LINE_SIZE 256
#else
#define CACHE_LINE_SIZE 64
#endif
#endif

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_dup(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add1(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_acc(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sum(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sum_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cumsum(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mean(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_argmax(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_count_equal(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_repeat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_repeat_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_concat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_silu_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rms_norm_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_group_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_l2_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_out_prod(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_scale(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_set(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cpy(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cont(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rows_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_set_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_diag(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_diag_mask_inf(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_diag_mask_zero(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_soft_max(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_soft_max_ext_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rope(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rope_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_clamp(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_transpose_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_im2col(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_im2col_back_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_im2col_3d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_3d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_transpose_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_conv_2d_dw(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pool_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pool_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pool_2d_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_upscale(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pad(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_pad_reflect_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_roll(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_arange(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_timestep_embedding(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_argsort(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_top_k(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_leaky_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_tri(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_fill(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_flash_attn_ext(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_flash_attn_back(
        const struct ggml_compute_params * params,
        const bool masked,
        struct ggml_tensor * dst);
void ggml_compute_forward_ssm_conv(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_ssm_scan(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_win_part(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_win_unpart(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_unary(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_get_rel_pos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_add_rel_pos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rwkv_wkv6(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_rwkv_wkv7(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_solve_tri(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_gla(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_gated_delta_net(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_map_custom1(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_map_custom2(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_map_custom3(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_custom(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cross_entropy_loss(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cross_entropy_loss_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_opt_step_adamw(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_opt_step_sgd(const struct ggml_compute_params * params, struct ggml_tensor * dst);
#ifdef __cplusplus
}
#endif

#include "ggml.h.inc"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
        const void * A, int64_t lda, const void * B, int64_t ldb, void * C, int64_t ldc,
        int ith, int nth, int type);
#endif

void ggml_compute_forward_add_non_quantized(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sub(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mul(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_div(const struct ggml_compute_params * params, struct ggml_tensor * dst);

void ggml_compute_forward_abs(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sgn(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_neg(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_step(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_tanh(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_elu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_hardsigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_exp(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_hardswish(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sqr(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sqrt(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sin(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_cos(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_log(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_expm1(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_softplus(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_floor(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_ceil(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_round(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_trunc(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_xielu(const struct ggml_compute_params * params, struct ggml_tensor * dst);
GGML_API void ggml_critical_section_start(void);
GGML_API void ggml_critical_section_end(void);

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

// precomputed f32 table for f16 (256 KB) (simd-mappings.h)
float ggml_table_f32_f16[1 << 16];

// precomputed f32 table for e8m0 half (1 KB) (simd-mappings.h)
float ggml_table_f32_e8m0_half[1 << 8];

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int sve_cnt;
} ggml_arm_arch_features = { 0 };
#endif

#if defined(__riscv)
struct ggml_riscv_arch_features_type {
    int rvv_vlen;
} ggml_riscv_arch_features = { 0 };
#endif

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif

typedef pthread_t ggml_thread_t;

#define GGML_THREADPOOL_N_THREADS_MASK (0xffffU)
#define GGML_THREADPOOL_N_THREADS_BITS (16)

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp32,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_fp16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q1_0] = {
        .from_float               = quantize_row_q1_0,
        .vec_dot                  = ggml_vec_dot_q1_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_MXFP4] = {
        .from_float               = quantize_row_mxfp4,
        .vec_dot                  = ggml_vec_dot_mxfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_NVFP4] = {
        .from_float               = quantize_row_nvfp4,
        .vec_dot                  = ggml_vec_dot_nvfp4_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
    },
    [GGML_TYPE_BF16] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_bf16,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_I32] = {
        .from_float               = (ggml_from_float_t) ggml_cpu_fp32_to_i32,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // updated when there is work to be done (i.e each graph) holds graph and active thread counts.
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int  abort;        // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads;   // Number of threads in the pool
    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
#ifndef GGML_USE_OPENMP
    ggml_thread_t thrd;
    int  last_graph;
    bool pending;
#endif
    bool cpumask[GGML_MAX_N_THREADS];
    struct ggml_threadpool * threadpool;
    int ith;
};

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#elif defined(__riscv)
static inline void ggml_thread_cpu_relax(void) {
    #ifdef __riscv_zihintpause
        __asm__ __volatile__ ("pause");
    #else
        /* Encoding of the pause instruction */
        __asm__ __volatile__ (".4byte 0x100000F");
    #endif
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
    #pragma omp barrier
#else
    int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

    // enter barrier (full seq-cst fence)
    int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

    if (n_barrier == (n_threads - 1)) {
        // last thread
        atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

        // exit barrier (full seq-cst fence)
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
        return;
    }

    // wait for other threads
    while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
        ggml_thread_cpu_relax();
    }

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
#endif
}

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value) {
    atomic_store_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

int ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value) {
    return atomic_fetch_add_explicit(&tp->current_chunk, value, memory_order_relaxed);
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
static void ggml_init_arm_arch_features(void) {
    ggml_arm_arch_features.sve_cnt = svcntb();
}
#else
static void ggml_init_arm_arch_features(void) {}
#endif
#endif // __ARM_ARCH

#if defined(__riscv) && defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
static void ggml_init_riscv_arch_features(void) {
    ggml_riscv_arch_features.rvv_vlen = __riscv_vlenb();
}
#else
static void ggml_init_riscv_arch_features(void) {}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_CPU_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_CPU_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_CPU_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    #endif
    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD_ID:
            {
                ggml_compute_forward_add_id(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_CUMSUM:
            {
                ggml_compute_forward_cumsum(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_SET_ROWS:
            {
                ggml_compute_forward_set_rows(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_IM2COL_3D:
            {
                ggml_compute_forward_im2col_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D:
            {
                ggml_compute_forward_conv_2d(params, tensor);
            } break;
        case GGML_OP_CONV_3D:
            {
                ggml_compute_forward_conv_3d(params, tensor);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                ggml_compute_forward_conv_2d_dw(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ROLL:
            {
                ggml_compute_forward_roll(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_TOP_K:
            {
                ggml_compute_forward_top_k(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_TRI:
            {
                ggml_compute_forward_tri(params, tensor);
            } break;
        case GGML_OP_FILL:
            {
                ggml_compute_forward_fill(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GLU:
            {
                ggml_compute_forward_glu(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                ggml_compute_forward_solve_tri(params, tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_SGD:
            {
                ggml_compute_forward_opt_step_sgd(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_RESHAPE:
            {
                // nop
            } break;
        case GGML_OP_PERMUTE:
            {
                // nop
            } break;
        case GGML_OP_VIEW:
            {
                // nop
            } break;
        case GGML_OP_TRANSPOSE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_FILL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_GATED_DELTA_NET:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_XIELU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(node)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
        case GGML_OP_CONV_3D:
        case GGML_OP_CONV_2D_DW:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ROLL:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                const int64_t n_heads = node->src[1]->ne[1];
                n_tasks = MIN(n_threads, n_heads);
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = THREAD_PRIORITY_BELOW_NORMAL;  break;
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio != GGML_SCHED_PRIO_LOW) {
        // Tell Windows that this thread should not be throttled (needs its own CPU core).
        // Newer Windows 11 versions aggressively park (offline) CPU cores and often place
        // all our threads onto the first 4 cores which results in terrible performance with
        // n_threads > 4
        #if _WIN32_WINNT >= 0x0602
        THREAD_POWER_THROTTLING_STATE t;
        ZeroMemory(&t, sizeof(t));
        t.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
        t.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
        t.StateMask   = 0;

        if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &t, sizeof(t))) {
            GGML_LOG_DEBUG("failed to disable thread power throttling %d : (%d)\n", prio, (int) GetLastError());
            return false;
        }
        #endif
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        // TODO: there seems to be no way to set lower prio on Apple platforms
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__gnu_linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      policy = SCHED_BATCH; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads;

#ifndef GGML_USE_OPENMP
    struct ggml_compute_state* workers = threadpool->workers;

    ggml_mutex_lock(&threadpool->mutex);

    threadpool->stop = true;
    threadpool->pause = false;

    ggml_cond_broadcast(&threadpool->cond);
    ggml_mutex_unlock(&threadpool->mutex);

    for (int j = 1; j < n_threads; j++) {
        int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
        GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
        UNUSED(rc);
    }

    ggml_mutex_destroy(&threadpool->mutex);
    ggml_cond_destroy(&threadpool->cond);
#endif // GGML_USE_OPENMP

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

#ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
#endif

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (!threadpool->pause) {
       ggml_threadpool_pause_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
#ifndef GGML_USE_OPENMP
    ggml_mutex_lock(&threadpool->mutex);
    if (threadpool->pause) {
       ggml_threadpool_resume_locked(threadpool);
    }
    ggml_mutex_unlock(&threadpool->mutex);
#else
    UNUSED(threadpool);
#endif
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads : GGML_DEFAULT_N_THREADS;
    }

#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
    // Emscripten without pthreads support can only use a single thread
    n_threads = 1;
#endif

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16) ||
                            // conversion between F32 and I32
                            (node->src[0]->type == GGML_TYPE_F32 && node->src[1] && node->src[1]->type == GGML_TYPE_I32) ||
                            (node->src[0]->type == GGML_TYPE_I32 && node->src[1] && node->src[1]->type == GGML_TYPE_F32)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD_ID:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_2D:
                case GGML_OP_CONV_3D:
                    {
                        cur = GGML_IM2COL_WORK_SIZE;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                        GGML_ASSERT(node->src[0]->type == GGML_TYPE_F16 || node->src[0]->type == GGML_TYPE_F32);
                        GGML_ASSERT(node->src[1]->type == GGML_TYPE_F32);

                        cur += ggml_type_size(node->src[0]->type) * ne00 * ne01 * ne02 * ne03;
                        cur += ggml_type_size(node->src[0]->type) * ne10 * ne11 * ne12;

                    } break;
                case GGML_OP_TOP_K:
                    {
                        cur += sizeof(int32_t)*node->src[0]->ne[0]*n_tasks;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t neq2 = node->src[0]->ne[2]; // number of query heads
                        const int64_t DK = node->src[1]->ne[0];
                        const int64_t DV = node->src[2]->ne[0];

                        // Tiled flash attention scratch (tile sizes defined in common.h.inc)
                        // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                        size_t prefill  = sizeof(float)*(GGML_FA_TILE_Q*DK + 2*GGML_FA_TILE_Q*GGML_FA_TILE_KV + GGML_FA_TILE_Q*DV + GGML_FA_TILE_KV*DV + GGML_FA_TILE_KV*DK)*n_tasks;

                        // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                        // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                        size_t n_chunks = n_tasks;
                        size_t decode   = sizeof(float)*(neq2*n_chunks*(2+DV) + n_tasks*(DK + 2*DV));

                        cur += MAX(prefill, decode);
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        cur = S_v * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

    set_numa_thread_affinity(state->ith);

    struct ggml_compute_params params = {
        /*.ith        =*/ state->ith,
        /*.nth        =*/ atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK,
        /*.wsize      =*/ cplan->work_size,
        /*.wdata      =*/ cplan->work_data,
        /*.threadpool =*/ tp,
        /*.use_ref    =*/ cplan->use_ref,
    };

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-start cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

        if (ggml_op_is_empty(node->op)) {
            // skip NOPs
            continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        ggml_compute_forward(&params, node);

        if (state->ith == 0 && cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
            atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
            tp->ec    = GGML_STATUS_ABORTED;
        }

        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

#ifdef GGML_USE_OPENMP
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p\n", state->ith, (const void *)cplan);
#else
    GGML_PRINT_DEBUG("thread #%d compute-done cplan %p last-graph %d\n", state->ith, (const void *)cplan, state->last_graph);
#endif

    ggml_barrier(state->threadpool);

    return 0;
}

#ifndef GGML_USE_OPENMP

// check if thread is ready to proceed (exit from polling or sleeping)
// returns true if loops should exit, sets state->pending to indicate new work
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int n_graph   = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    int n_threads = n_graph & GGML_THREADPOOL_N_THREADS_MASK;
    if (n_graph != state->last_graph) {
        state->pending    = (state->ith < n_threads);
        state->last_graph = n_graph;
        return true;
    }

    return false;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;
            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    // Update the number of active threads and the graph count
    int n_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed) >> GGML_THREADPOOL_N_THREADS_BITS;
    n_graph = ((n_graph + 1) << GGML_THREADPOOL_N_THREADS_BITS) | (n_threads & GGML_THREADPOOL_N_THREADS_MASK);

    GGML_PRINT_DEBUG("compute-kickoff: n_threads %d n_graph %d\n", n_threads, n_graph);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_store_explicit(&threadpool->n_graph, n_graph, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

#endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads        = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

#ifdef GGML_USE_OPENMP
    int32_t cpumask_iter = 0;

    // Compute CPU masks for each thread
    for (int j = 0; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);
    }
#else // GGML_USE_OPENMP
    ggml_mutex_init(&threadpool->mutex);
    ggml_cond_init(&threadpool->cond);

    // Spin the threads for all workers, and update CPU placements.
    // Place the main thread last (towards the higher numbered CPU cores).

    int32_t cpumask_iter = 0;

    for (int j = 1; j < tpp->n_threads; j++) {
        ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

        int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
        GGML_ASSERT(rc == 0);
    }

    ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

    if (!threadpool->pause) {
        // Update main thread prio and affinity at the start, otherwise we'll do it in resume
        ggml_thread_apply_priority(threadpool->prio);
        if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
            ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
        }
    }
#endif // GGML_USE_OPENMP

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

#ifdef GGML_USE_OPENMP
    if (n_threads > 1) {
        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp single
            {
                // update the number of threads from the actual number of threads that we got from OpenMP
                n_threads = omp_get_num_threads();
                atomic_store_explicit(&threadpool->n_graph, n_threads, memory_order_relaxed);
            }

            // Apply thread CPU mask and priority
            int ith = omp_get_thread_num();

            ggml_thread_apply_priority(threadpool->prio);
            if (ggml_thread_cpumask_is_valid(threadpool->workers[ith].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[ith].cpumask);
            }
            ggml_graph_compute_thread(&threadpool->workers[ith]);
        }
    } else {
        atomic_store_explicit(&threadpool->n_graph, 1, memory_order_relaxed);
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }
#else
    if (n_threads > threadpool->n_threads) {
        GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads);
        n_threads = threadpool->n_threads;
    }

    // Kick all threads to start the new graph
    ggml_graph_compute_kickoff(threadpool, n_threads);

    // This is a work thread too
    ggml_graph_compute_thread(&threadpool->workers[0]);
#endif

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}

void ggml_cpu_fp32_to_fp32(const float * x, float * y, int64_t n) {
    memcpy(y, x, n * sizeof(float));
}

void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}

void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }

#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfhmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vfloat16m2_t ax1 = __riscv_vle16_v_f16m2((const _Float16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvt_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vfloat16m2_t ax0 = __riscv_vle16_v_f16m2((const _Float16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvt_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }

#endif

    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }
}

void ggml_cpu_fp32_to_bf16(const float * x, ggml_bf16_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }
}

void ggml_cpu_fp32_to_i32(const float * x, int32_t * y, int64_t n) {
    int64_t i = 0;
    for (; i < n; ++i) {
        y[i] = x[i];
    }
}

void ggml_cpu_bf16_to_fp32(const ggml_bf16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__AVX2__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i,
                        _mm512_castsi512_ps(
                            _mm512_slli_epi32(
                                _mm512_cvtepu16_epi32(
                                    _mm256_loadu_si256(
                                        (const __m256i *)(x + i))),
                                16)));
    }
#endif
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i,
                        _mm256_castsi256_ps(
                            _mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(
                                    _mm_loadu_si128(
                                        (const __m128i *)(x + i))),
                                16)));
    }
#elif defined(__riscv_v_intrinsic) && defined(__riscv_zvfbfmin)
    // calculate step size
    const int epr = __riscv_vsetvlmax_e16m2();
    const int step = epr * 2;
    const int np = (n & ~(step - 1));

    // unroll by 2
    for (; i < np; i += step) {
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, epr);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, epr);
        __riscv_vse32_v_f32m4(y + i, ay0, epr);

        vbfloat16m2_t ax1 = __riscv_vle16_v_bf16m2((const __bf16*)x + i + epr, epr);
        vfloat32m4_t ay1 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax1, epr);
        __riscv_vse32_v_f32m4(y + i + epr, ay1, epr);
    }

    // leftovers
    int vl;
    for (i = np; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m2(n - i);
        vbfloat16m2_t ax0 = __riscv_vle16_v_bf16m2((const __bf16*)x + i, vl);
        vfloat32m4_t ay0 = __riscv_vfwcvtbf16_f_f_v_f32m4(ax0, vl);
        __riscv_vse32_v_f32m4(y + i, ay0, vl);
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }
}

int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_rvv_vlen(void) {
#if defined(__riscv) && defined(__riscv_v_intrinsic)
    return ggml_riscv_arch_features.rvv_vlen;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return 1;
#else
    return 0;
#endif
}

void ggml_cpu_init(void) {
    // needed to initialize ggml_time
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    static bool is_first_call = true;

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
                ggml_table_f32_f16[i] = f;
                ggml_table_gelu_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_CPU_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            // initialize E8M0 half table (256 entries)
            for (int i = 0; i < (1 << 8); ++i) {
                ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);

#ifdef GGML_USE_OPENMP
            //if (!getenv("OMP_WAIT_POLICY")) {
            //    // set the wait policy to active, so that OpenMP threads don't sleep
            //    setenv("OMP_WAIT_POLICY", "active", 0)
            //}

            if (!getenv("KMP_BLOCKTIME")) {
                // set the time to wait before sleeping a thread
                // this is less aggressive than setting the wait policy to active, but should achieve similar results in most cases
#ifdef _WIN32
                _putenv_s("KMP_BLOCKTIME", "200"); // 200ms
#else
                setenv("KMP_BLOCKTIME", "200", 0); // 200ms
#endif
            }
#endif
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(__riscv)
        ggml_init_riscv_arch_features();
#endif

        is_first_call = false;
    }

    ggml_critical_section_end();
}
