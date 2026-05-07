#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <assert.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <math.h>
#include <string.h>

#include "hex-dma-defs.inc"
#include "hvx-utils-defs.inc"
#include <HAP_farf.h>

#include <stdbool.h>
#include <stdint.h>

#include "hex-utils-defs.inc"
#include "hvx-types-defs.inc"

static void hvx_vec_dump_f16_n(char * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    const uint32_t n0 = n / 16;
    const uint32_t n1 = n % 16;
    int            i  = 0;
    for (; i < n0; i++) {
        hex_dump_f16_line(pref, u.fp16 + (16 * i), 16);
    }
    if (n1) {
        hex_dump_f16_line(pref, u.fp16 + (16 * i), n1);
    }
}

static void hvx_vec_dump_f16(char * pref, HVX_Vector v) {
    hvx_vec_dump_f16_n(pref, v, 64);
}

static void hvx_vec_dump_f32_n(char * pref, HVX_Vector v, uint32_t n) {
    HVX_VectorAlias u = { .v = v };

    const uint32_t n0 = n / 16;
    const uint32_t n1 = n % 16;
    int            i  = 0;
    for (; i < n0; i++) {
        hex_dump_f32_line(pref, u.fp32 + (16 * i), 16);
    }
    if (n1) {
        hex_dump_f32_line(pref, u.fp32 + (16 * i), n1);
    }
}

static void hvx_vec_dump_f32_hmt(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        float      d[32];
    } u = { .v = v };

    FARF(HIGH, "%s: %.6f %.6f %.6f %.6f ...  %.6f %.6f %.6f %.6f ... %.6f %.6f %.6f %.6f\n", pref, u.d[0], u.d[1],
         u.d[2], u.d[3], u.d[12], u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

static void hvx_vec_dump_f32(char * pref, HVX_Vector v) {
    hvx_vec_dump_f32_n(pref, v, 32);
}

static void hvx_vec_dump_int32(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    for (int i = 0; i < 32 / 16; i++) {
        hex_dump_int32_line(pref, u.d + (16 * i), 16);
    }
}

static void hvx_vec_dump_int32_hmt(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        int32_t    d[32];
    } u = { .v = v };

    FARF(HIGH, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[12],
         u.d[13], u.d[14], u.d[15], u.d[28], u.d[29], u.d[30], u.d[31]);
}

static void hvx_vec_dump_int8_hmt(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    FARF(HIGH, "%s: %d %d %d %d ... %d %d %d %d ... %d %d %d %d\n", pref, u.d[0], u.d[1], u.d[2], u.d[3], u.d[60],
         u.d[61], u.d[62], u.d[63], u.d[124], u.d[125], u.d[126], u.d[127]);
}

static void hvx_vec_dump_int8(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        int8_t     d[128];
    } u = { .v = v };

    for (int i = 0; i < 128 / 16; i++) {
        hex_dump_int8_line(pref, u.d + (16 * i), 16);
    }
}

static void hvx_vec_dump_uint8(char * pref, HVX_Vector v) {
    union {
        HVX_Vector v;
        uint8_t    d[128];
    } u = { .v = v };

    for (int i = 0; i < 128 / 16; i++) {
        hex_dump_uint8_line(pref, u.d + (16 * i), 16);
    }
}

static bool hvx_vec_eq(HVX_Vector v0, HVX_Vector v1, size_t n) {
    typedef union {
        HVX_Vector v;
        int8_t     d[128];
    } U;

    U u0 = { .v = v0 };
    U u1 = { .v = v1 };

    for (int i = 0; i < n; i++) {
        if (u0.d[i] != u1.d[i]) {
            return false;
        }
    }

    return true;
}


#define GGML_COMMON_DECL_C
#include "ggml-common-defs.inc"
#include "htp-ctx-defs.inc"
#include "htp-ops-defs.inc"
#include "htp-ops-defs.inc"
// HMX operation entry-point declarations.
// Ported from htp-ops-lib/include/dsp/ops.h (renamed, benchmark kernels removed). (https://github.com/haozixu/htp-ops-lib)

#ifndef HMX_OPS_H
#define HMX_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "htp-ops-defs.inc"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float        *dst;
    const float  *activation;
    const __fp16 *permuted_weight;
    int           m;
    int           k;
    int           n;
    int           act_stride;
    int           weight_stride;
    int           dst_stride;
    int           ne02;
    int           ne03;
    int           ne12;
    int           ne13;
    size_t        src0_nb2;
    size_t        src0_nb3;
    size_t        src1_nb2;
    size_t        src1_nb3;
    size_t        dst_nb2;
    size_t        dst_nb3;
} hmx_matmul_w16a32_batched_params_t;

// HMX matrix multiplication — tile-permuted FP16 weights, FP32 activation/output
// act_stride: activation row stride in elements (= k for contiguous, or
//             nb[1]/sizeof(float) for permuted tensors like attention Q).
// weight_stride: weight row stride in elements (= k for compact weights, or
//                nb[1]/sizeof(__fp16) for permuted KV-cache views used by QK).
int hmx_mat_mul_permuted_w16a32(struct htp_context *ctx,
                                float *restrict dst,
                                const float *activation,
                                const __fp16 *permuted_weight,
                                int m, int k, int n,
                                int act_stride,
                                int weight_stride);

// Batched F16 wrapper over hmx_mat_mul_permuted_w16a32.
// Batch semantics match ggml_mul_mat(): src0 broadcasts to src1 in dims 2/3.
int hmx_mat_mul_permuted_w16a32_batched(struct htp_context *ctx,
                                        const hmx_matmul_w16a32_batched_params_t *params);

// HMX matrix multiplication — tile-permuted quantised weights (Q4_0/Q8_0/IQ4_NL)
int hmx_mat_mul_permuted_qk_0_d16a32(struct htp_context *ctx,
                                      float *restrict dst,
                                      const float *activation,
                                      const uint8_t *permuted_weight,
                                      int m, int k, int n,
                                      int weight_type);

// HMX flash attention
int hmx_flash_attn_ext(struct htp_ops_context * octx);

#ifdef __cplusplus
}
#endif

#endif // HMX_OPS_H


// Must be multiple of 32
#define FLASH_ATTN_BLOCK_SIZE (32 * 2)

// This is a bit of a hack because the compiler is strugling to properly inline
// the default hvx_vec_f32_to_f16 with output into the local array.
static __attribute__((noinline)) void hvx_vec_f32_to_f16_a(void *ptr, HVX_Vector v0, HVX_Vector v1)
{
    *(HVX_Vector *) ptr = hvx_vec_f32_to_f16(v0, v1);
}

// Dot product of two F16 vectors, accumulating to float
static inline void hvx_dot_f16_f16_aa(float * restrict r, const void * restrict x, const void * restrict y, unsigned int n, float s) {
    const HVX_Vector * restrict vx = (const HVX_Vector * restrict) x; // fp16
    const HVX_Vector * restrict vy = (const HVX_Vector * restrict) y; // fp16

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_VectorPair rsum_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    #pragma unroll(4)
    for (i = 0; i < nvec; i++) {
        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, vx[i], vy[i]);
    }

    if (nloe) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector y_hf = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector x_hf = Q6_V_vand_QV(bmask, vx[i]);

        rsum_p = hvx_vec_mpyacc_f32_f16(rsum_p, x_hf, y_hf);
    }

    HVX_Vector rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum_p), Q6_V_hi_W(rsum_p)));
    rsum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(hvx_vec_splat_f32(s), hvx_vec_reduce_sum_f32(rsum)));
    hvx_vec_store_u(r, 4, rsum);
}

static inline HVX_Vector hvx_dot_f16_f16_aa_rx4(const void * restrict y,
                                                const uint8_t * restrict x,
                                                const size_t stride_x,
                                                const size_t nvec,
                                                const size_t nloe) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector * restrict) x;                   // fp16
    const HVX_Vector * restrict vx1 = (const HVX_Vector * restrict) (x + stride_x);      // fp16
    const HVX_Vector * restrict vx2 = (const HVX_Vector * restrict) (x + stride_x * 2);  // fp16
    const HVX_Vector * restrict vx3 = (const HVX_Vector * restrict) (x + stride_x * 3);  // fp16
    const HVX_Vector * restrict vy  = (const HVX_Vector * restrict) y;                   // fp16

    HVX_VectorPair rsum0_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum1_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum2_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));
    HVX_VectorPair rsum3_p = Q6_W_vcombine_VV(Q6_V_vsplat_R(0), Q6_V_vsplat_R(0));

    uint32_t i = 0;

    for (i = 0; i < nvec; i++) {
        HVX_Vector y_hf  = vy[i];
        HVX_Vector x0_hf = vx0[i];
        HVX_Vector x1_hf = vx1[i];
        HVX_Vector x2_hf = vx2[i];
        HVX_Vector x3_hf = vx3[i];

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    if (nloe) {
        // Load x (fp16) and zero-out unused elements
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 2);
        HVX_Vector     y_hf  = Q6_V_vand_QV(bmask, vy[i]);
        HVX_Vector     x0_hf = Q6_V_vand_QV(bmask, vx0[i]);
        HVX_Vector     x1_hf = Q6_V_vand_QV(bmask, vx1[i]);
        HVX_Vector     x2_hf = Q6_V_vand_QV(bmask, vx2[i]);
        HVX_Vector     x3_hf = Q6_V_vand_QV(bmask, vx3[i]);

        rsum0_p = hvx_vec_mpyacc_f32_f16(rsum0_p, x0_hf, y_hf);
        rsum1_p = hvx_vec_mpyacc_f32_f16(rsum1_p, x1_hf, y_hf);
        rsum2_p = hvx_vec_mpyacc_f32_f16(rsum2_p, x2_hf, y_hf);
        rsum3_p = hvx_vec_mpyacc_f32_f16(rsum3_p, x3_hf, y_hf);
    }

    HVX_Vector rsum0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum0_p), Q6_V_hi_W(rsum0_p)));
    HVX_Vector rsum1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum1_p), Q6_V_hi_W(rsum1_p)));
    HVX_Vector rsum2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum2_p), Q6_V_hi_W(rsum2_p)));
    HVX_Vector rsum3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(rsum3_p), Q6_V_hi_W(rsum3_p)));

    HVX_Vector_x4 rsum0123 = { .v = { rsum0, rsum1, rsum2, rsum3 } };
    return hvx_vec_reduce_sum_f32x4(rsum0123);
}

static inline HVX_Vector hvx_dot_f16_f16_aa_rx32(const void * restrict y,
                                                 const uint8_t * restrict x,
                                                 const size_t stride_x,
                                                 const size_t n,
                                                 float        s) {

    const size_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    const size_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector   sums;  // initialize at j = 0
    const size_t stride_x_4 = stride_x * 4;
    for (uint32_t j = 0; j < VLEN_FP32; j += 4) {
        HVX_Vector     sums_x4 = hvx_dot_f16_f16_aa_rx4(y, x, stride_x, nvec, nloe);
        HVX_VectorPred pred    = Q6_Q_vsetq_R(j * SIZEOF_FP32);
        sums                   = Q6_V_vmux_QVV(pred, sums, sums_x4);
        x += stride_x_4;
    }

    sums = Q6_Vqf32_vmpy_VsfVsf(hvx_vec_splat_f32(s), sums);
    return Q6_Vsf_equals_Vqf32(sums);
}

// MAD: y (F32) += x (F16) * s (F16)
static inline void hvx_mad_f32_f16_aa(float * restrict y, const void * restrict x, const __fp16 * restrict s, int n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x;

    HVX_VectorPair * restrict vy_p = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16; // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16; // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

// MAD: y (F32) += x0 (F16) * s0 (F16) + x1 (F16) * s1 (F16)
static inline void hvx_mad_f32_f16_aa_rx2(float * restrict y, const void * restrict x0, const void * restrict x1,
                                          const __fp16 * restrict s0, const __fp16 * restrict s1, int n) {
    const HVX_Vector * restrict vx0 = (const HVX_Vector *) x0;
    const HVX_Vector * restrict vx1 = (const HVX_Vector *) x1;

    HVX_VectorPair * restrict vy_p  = (HVX_VectorPair *) y;
    HVX_Vector * restrict vy        = (HVX_Vector *) y;

    uint32_t nvec = n / VLEN_FP16;  // num full fp16 hvx vectors
    uint32_t nloe = n % VLEN_FP16;  // leftover elements

    HVX_Vector S0 = hvx_vec_splat_f16(*s0);
    HVX_Vector S1 = hvx_vec_splat_f16(*s1);

    uint32_t i = 0;

    #pragma unroll(2)
    for (i = 0; i < nvec; ++i) {
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx0[i]), S0);
        vy_p[i] = hvx_vec_mpyacc_f32_f16(vy_p[i], Q6_Vh_vshuff_Vh(vx1[i]), S1);
    }

    if (nloe) {
        HVX_VectorPair xy_p = vy_p[i];
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx0[i]), S0);
        xy_p = hvx_vec_mpyacc_f32_f16(xy_p, Q6_Vh_vshuff_Vh(vx1[i]), S1);

        HVX_Vector xy = Q6_V_lo_W(xy_p);
        i = 2 * i;  // index for vy

        if (nloe >= VLEN_FP32) {
            vy[i] = xy;
            nloe -= VLEN_FP32; ++i; xy = Q6_V_hi_W(xy_p);
        }

        if (nloe) {
            hvx_vec_store_a(&vy[i], nloe * 4, xy);
        }
    }
}

struct htp_fa_context {
    const struct htp_ops_context * octx;

    struct fastdiv_values src0_div21;
    struct fastdiv_values src0_div1;

    struct fastdiv_values broadcast_rk2;
    struct fastdiv_values broadcast_rk3;
    struct fastdiv_values broadcast_rv2;
    struct fastdiv_values broadcast_rv3;

    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t n_head_log2;
    float m0;
    float m1;

    uint32_t n_blocks;

    size_t size_q_row_padded;
    size_t size_k_row_padded;
    size_t size_v_row_padded;

    size_t size_k_block;
    size_t size_v_block;
    size_t size_m_block;

    uint32_t qrows;
    uint32_t qrows_per_thread;

    bool is_q_fp32;

    uint64_t t_start;
};

static inline void hvx_scale_vec_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, HVX_Vector vs) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);

    const HVX_Vector * restrict vsrc = (const HVX_Vector * restrict) src;
    HVX_Vector * restrict vdst       = (HVX_Vector * restrict) dst;

    const uint32_t nvec = n / VLEN_FP32;
    const uint32_t nloe = n % VLEN_FP32;

    uint32_t i = 0;
    #pragma unroll(4)
    for (; i < nvec; ++i) {
        vdst[i] = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs));
    }
    if (nloe) {
        HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);
        hvx_vec_store_a(&vdst[i], nloe * sizeof(float), Q6_Vsf_equals_Vqf32(v));
    }
}

static void flash_attn_ext_f16_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * mask  = octx->src[3];
    const struct htp_tensor * sinks = octx->src[4];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t neq0 = q->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t neq3 = q->ne[3];

    const uint32_t nek0 = k->ne[0];
    const uint32_t nek1 = k->ne[1];
    const uint32_t nek2 = k->ne[2];
    const uint32_t nek3 = k->ne[3];

    const uint32_t nev0 = v->ne[0];
    const uint32_t nev1 = v->ne[1];
    const uint32_t nev2 = v->ne[2];
    const uint32_t nev3 = v->ne[3];

    const uint32_t nbq1 = q->nb[1];
    const uint32_t nbq2 = q->nb[2];
    const uint32_t nbq3 = q->nb[3];

    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbk3 = k->nb[3];

    const uint32_t nbv1 = v->nb[1];
    const uint32_t nbv2 = v->nb[2];
    const uint32_t nbv3 = v->nb[3];

    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    // total rows in q
    const uint32_t nr = factx->qrows;
    const uint32_t dr = factx->qrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, nr);

    if (ir0 >= ir1) return;

    dma_queue * dma = octx->ctx->dma[ith];

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    const size_t size_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    const size_t size_k_row = DK * sizeof(__fp16);
    const size_t size_v_row = DV * sizeof(__fp16);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = octx->src0_spad.data + octx->src0_spad.size_per_thread * ith;
    uint8_t * spad_k = octx->src1_spad.data + octx->src1_spad.size_per_thread * ith;
    uint8_t * spad_v = octx->src2_spad.data + octx->src2_spad.size_per_thread * ith;
    uint8_t * spad_m = octx->src3_spad.data + octx->src3_spad.size_per_thread * ith;
    uint8_t * spad_a = octx->dst_spad.data  + octx->dst_spad.size_per_thread  * ith;

    const HVX_Vector logit_cap = hvx_vec_splat_f32(factx->logit_softcap);

    dma_cache m_cache;
    dma_cache_init(&m_cache, spad_m, factx->size_m_block, DMA_CACHE_MAX_SIZE);

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        const uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        const uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        const uint32_t ik3 = fastdiv(iq3, &factx->broadcast_rk3);
        const uint32_t ik2 = fastdiv(iq2, &factx->broadcast_rk2);

        const uint32_t iv3 = fastdiv(iq3, &factx->broadcast_rv3);
        const uint32_t iv2 = fastdiv(iq2, &factx->broadcast_rv2);

        // Fetch Q row
        const uint8_t * q_row_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);
        dma_queue_push(dma, dma_make_ptr(spad_q, q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);

        // FARF(HIGH, "fa %u: prefetch Q: ir %u iq1 %u iq2 %u iq3 %u q_row_ptr %p size %u : usec %u", ith, ir, iq1, iq2, iq3, q_row_ptr, size_q_row,
        //                 (unsigned)HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - factx->t_start));

        const __fp16 * mp_base = NULL;
        if (mask) {
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &factx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &factx->src3_div3);
            mp_base = (const __fp16 *) ((const uint8_t *) mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3]);
        }

        // Prefetch first two blocks
        for (uint32_t ib = 0; ib < MIN(factx->n_blocks, 2); ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // K
            const uint8_t * k_src = (const uint8_t *) k->data + (ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
            uint8_t * k_dst = spad_k + (ib % 2) * factx->size_k_block;
            dma_queue_push(dma, dma_make_ptr(k_dst, k_src), factx->size_k_row_padded, nbk1, size_k_row, current_block_size);

            // V
            const uint8_t * v_src = (const uint8_t *) v->data + (ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
            uint8_t * v_dst = spad_v + (ib % 2) * factx->size_v_block;
            dma_queue_push(dma, dma_make_ptr(v_dst, v_src), factx->size_v_row_padded, nbv1, size_v_row, current_block_size);

            // Mask
            if (mask) {
                const uint8_t * m_src = (const uint8_t *) (mp_base + ic_start);
                // Mask is 1D contiguous for this row
                dma_cache_push(dma, &m_cache, m_src, current_block_size * 2, current_block_size * 2, current_block_size * 2, 1);
            }

            // FARF(HIGH, "fa %u: prefetch KVM: ir %u ib %u iq1 %u iq2 %u iq3 %u : size_k_row %u size_v_row %u bs %u: usec %u",
            //             ith, ir, ib, iq1, iq2, iq3,
            //             size_k_row, size_v_row, current_block_size,
            //             (unsigned)HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - factx->t_start));
        }

        const uint32_t h = iq2; // head index
        const float slope = (factx->max_bias > 0.0f) ? (h < factx->n_head_log2 ? powf(factx->m0, h + 1) : powf(factx->m1, 2*(h - factx->n_head_log2) + 1)) : 1.0f;

        HVX_Vector S_vec = hvx_vec_splat_f32(0.0f);
        HVX_Vector M_vec = hvx_vec_splat_f32(-INFINITY);

        // Clear accumulator
        hvx_splat_f32_a(spad_a, 0, DV);
        float * VKQ32 = (float *) (spad_a + 0);

        uint8_t * q_ptr_vtcm = dma_queue_pop(dma).dst;
        if (factx->is_q_fp32) {
            hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);  // inplace convert f32 to f16
        }

        const HVX_Vector slope_vec = hvx_vec_splat_f16(slope);
        for (uint32_t ib = 0; ib < factx->n_blocks; ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // Wait for DMA
            uint8_t * k_base = dma_queue_pop(dma).dst; // K
            uint8_t * v_base = dma_queue_pop(dma).dst; // V
            __fp16  * m_base = mask ? dma_queue_pop(dma).dst : NULL; // M

            // FARF(HIGH, "fa %u: process: ir %u ib %u : iq1 %u iq2 %u iq3 %u q_ptr_vtcm %p : usec %u",
            //              ith, ir, ib, iq1, iq2, iq3, q_ptr_vtcm,
            //             (unsigned)HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - factx->t_start));

            // Inner loop processing the block from VTCM
            uint32_t ic = 0;

            // Process in sub-blocks of 32 (VLEN_FP32)
            HVX_Vector sb_scores[FLASH_ATTN_BLOCK_SIZE / VLEN_FP32];
            HVX_Vector v_max = hvx_vec_splat_f32(-INFINITY);
            for (uint32_t iv = 0; ic + VLEN_FP32 <= current_block_size; ic += VLEN_FP32, ++iv) {
                // 1. Compute scores
                HVX_Vector scores = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_base + ic * factx->size_k_row_padded, factx->size_k_row_padded, DK, factx->scale);

                // 2. Softcap
                if (factx->logit_softcap != 0.0f) {
                    scores = hvx_vec_tanh_f32(scores);
                    scores = Q6_Vqf32_vmpy_VsfVsf(scores, logit_cap);
                    scores = Q6_Vsf_equals_Vqf32(scores);
                }

                // 3. Mask
                if (mask) {
                    const __fp16 * mp = m_base + ic;
                    HVX_Vector m_vals_f16 = *(const HVX_UVector *) mp;
                    HVX_VectorPair m_vals_f32_pair = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(m_vals_f16), slope_vec);
                    HVX_Vector add_val = Q6_V_lo_W(m_vals_f32_pair);
                    scores = Q6_Vqf32_vadd_Vqf32Vsf(add_val, scores);
                    scores = Q6_Vsf_equals_Vqf32(scores);
                }

                sb_scores[iv] = scores;
                v_max = hvx_vec_reduce_max2_f32(scores, v_max); // All lanes have block max
            }

            {
                // 4. Online Softmax Update
                HVX_Vector M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec);
                HVX_Vector diff_vec  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(M_vec, M_new_vec));
                HVX_Vector ms_vec    = hvx_vec_exp_f32(diff_vec);
                M_vec = M_new_vec;

                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                HVX_Vector p_sum_vec = hvx_vec_splat_f32(0.0f);
                for (uint32_t ic2 = 0, iv = 0; ic2 + VLEN_FP32 <= current_block_size; ic2 += VLEN_FP32, ++iv) {
                    HVX_Vector scores = sb_scores[iv];
                    HVX_Vector scores_shifted = Q6_Vqf32_vsub_VsfVsf(scores, M_vec);
                    HVX_Vector P = hvx_vec_exp_f32(Q6_Vsf_equals_Vqf32(scores_shifted));

                    p_sum_vec = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(p_sum_vec, P));

                    // 5. Accumulate V
                    __fp16 __attribute__((aligned(VLEN))) p_arr[VLEN_FP16];
                    hvx_vec_f32_to_f16_a(p_arr, P, hvx_vec_splat_f32(0));

                    for (uint32_t j = 0; j < VLEN_FP32; j += 2) {
                        const uint32_t  cur_ic = ic2 + j;
                        const uint8_t * v_ptr  = v_base + cur_ic * factx->size_v_row_padded;
                        hvx_mad_f32_f16_aa_rx2(VKQ32, v_ptr, v_ptr + factx->size_v_row_padded, (p_arr + j), (p_arr + j + 1), DV);
                    }
                }

                p_sum_vec = hvx_vec_reduce_sum_f32(p_sum_vec);
                S_vec = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(S_vec, ms_vec)), p_sum_vec));
            }

            if (ic < current_block_size) {
                // Sync scalars for leftover/next block if needed
                float M = hvx_vec_get_f32(M_vec);
                float S = hvx_vec_get_f32(S_vec);

                // Leftover
                for (; ic < current_block_size; ++ic) {
                    float s_val;
                    const uint8_t * k_ptr = k_base + ic * factx->size_k_row_padded;
                    hvx_dot_f16_f16_aa(&s_val, q_ptr_vtcm, k_ptr, DK, factx->scale);
                    if (factx->logit_softcap != 0.0f) {
                        s_val = factx->logit_softcap * tanhf(s_val);
                    }

                    if (mask) {
                        const float m_val = m_base[ic];
                        s_val += slope * m_val;
                    }

                    const float Mold = M;
                    __fp16 vs = 1.0f;

                    if (s_val > M) {
                        M = s_val;
                        HVX_Vector diff_vec = hvx_vec_splat_f32(Mold - M);
                        HVX_Vector ms_vec   = hvx_vec_exp_f32(diff_vec);
                        hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                        float ms = hvx_vec_get_f32(ms_vec);
                        S = S * ms + vs;
                    } else {
                        HVX_Vector diff_vec = hvx_vec_splat_f32(s_val - M);
                        vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                        S += vs;
                    }

                    const uint8_t * v_ptr = v_base + ic * factx->size_v_row_padded;

                    hvx_mad_f32_f16_aa(VKQ32, v_ptr, &vs, DV);
                }

                M_vec = hvx_vec_splat_f32(M);
                S_vec = hvx_vec_splat_f32(S);
            }

            // Issue DMA for next+1 block (if exists)
            if (ib + 2 < factx->n_blocks) {
                const uint32_t next_ib = ib + 2;
                const uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                // K
                const uint8_t * k_src = (const uint8_t *) k->data + (next_ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
                dma_queue_push(dma, dma_make_ptr(k_base, k_src), factx->size_k_row_padded, nbk1, size_k_row, next_block_size);

                // V
                const uint8_t * v_src = (const uint8_t *) v->data + (next_ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
                dma_queue_push(dma, dma_make_ptr(v_base, v_src), factx->size_v_row_padded, nbv1, size_v_row, next_block_size);

                // Mask
                if (mask) {
                    const uint8_t * m_src = (const uint8_t *) (mp_base + next_ic_start);
                    dma_cache_push(dma, &m_cache, m_src, next_block_size * 2, next_block_size * 2, next_block_size * 2, 1);
                }

                // FARF(HIGH, "fa %u: prefetch KVM: ir %u ib %u : iq1 %u iq2 %u iq3 %u : size_k_row %u size_v_row %u bs %u: usec %u",
                //         ith, ir, next_ib, iq1, iq2, iq3,
                //         size_k_row, size_v_row, next_block_size,
                //         (unsigned)HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - factx->t_start));
            }
        }

        // sinks
        float M = hvx_vec_get_f32(M_vec);
        float S = hvx_vec_get_f32(S_vec);

        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            float vs = 1.0f;

            if (s > M) {
                HVX_Vector diff_vec = hvx_vec_splat_f32(M - s);
                HVX_Vector ms_vec   = hvx_vec_exp_f32(diff_vec);
                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                float ms = hvx_vec_get_f32(ms_vec);
                S = S * ms + vs;
            } else {
                HVX_Vector diff_vec = hvx_vec_splat_f32(s - M);
                vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                S += vs;
            }
        }

        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        hvx_scale_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, S_inv);

        // Store result
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // dst is permuted
        uint8_t * dst_ptr = (uint8_t *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1) * nb1;

        if (dst->type == HTP_TYPE_F32) {
            hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        } else if (dst->type == HTP_TYPE_F16) {
            hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        }
    }
}

int op_flash_attn_ext(struct htp_ops_context * octx) {
    const struct htp_tensor * q    = octx->src[0];
    const struct htp_tensor * k    = octx->src[1];
    const struct htp_tensor * v    = octx->src[2];
    const struct htp_tensor * mask = octx->src[3];
    const struct htp_tensor * dst  = octx->dst;

    // Check support
    if ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) || k->type != HTP_TYPE_F16 || v->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

#ifdef HTP_HAS_HMX
    // HMX path: prefill (neq1 >= 32), head_dim multiple of 32, F16 KV
    if (k->type == HTP_TYPE_F16 && v->type == HTP_TYPE_F16 && k->ne[0] % 32 == 0 && q->ne[1] >= 32) {
        int ret = hmx_flash_attn_ext(octx);
        if (ret == HTP_STATUS_OK) {
            return ret;
        }
        // VTCM too small or other failure -> fall through to HVX path
    }
#endif

    struct htp_fa_context factx;
    factx.octx = octx;

    factx.t_start = HAP_perf_get_qtimer_count();

    factx.src0_div21 = init_fastdiv_values(q->ne[2] * q->ne[1]);
    factx.src0_div1  = init_fastdiv_values(q->ne[1]);

    factx.broadcast_rk2 = init_fastdiv_values(q->ne[2]/k->ne[2]);
    factx.broadcast_rk3 = init_fastdiv_values(q->ne[3]/k->ne[3]);
    factx.broadcast_rv2 = init_fastdiv_values(q->ne[2]/v->ne[2]);
    factx.broadcast_rv3 = init_fastdiv_values(q->ne[3]/v->ne[3]);

    if (mask) {
        factx.src3_div2 = init_fastdiv_values(mask->ne[2]);
        factx.src3_div3 = init_fastdiv_values(mask->ne[3]);
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.size_q_row_padded = hex_round_up(q->ne[0] * (factx.is_q_fp32 ? 4 : 2), 128);
    factx.size_k_row_padded = hex_round_up(k->ne[0] * sizeof(__fp16), 128);
    factx.size_v_row_padded = hex_round_up(v->ne[0] * sizeof(__fp16), 128);

    size_t size_q_block = factx.size_q_row_padded * 1; // single row for now
    factx.size_k_block = factx.size_k_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_v_block = factx.size_v_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_m_block = hex_round_up(FLASH_ATTN_BLOCK_SIZE * sizeof(__fp16), 128);

    factx.n_blocks = (k->ne[1] + FLASH_ATTN_BLOCK_SIZE - 1) / FLASH_ATTN_BLOCK_SIZE;

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (float *) octx->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (float *) octx->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) octx->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    factx.scale = scale;
    factx.max_bias = max_bias;
    factx.logit_softcap = logit_softcap;

    uint32_t n_head = q->ne[2];
    factx.n_head_log2 = 1u << (uint32_t) floor(log2(n_head));
    factx.m0 = powf(2.0f, -(max_bias       ) / factx.n_head_log2);
    factx.m1 = powf(2.0f, -(max_bias / 2.0f) / factx.n_head_log2);

    // total rows in q
    const uint32_t neq0 = q->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t neq3 = q->ne[3];

    factx.qrows = neq1*neq2*neq3;
    factx.qrows_per_thread = (factx.qrows + octx->n_threads - 1) / octx->n_threads;

    size_t size_vkq_acc = hex_round_up(v->ne[0] * sizeof(float), 128); // VKQ32

    octx->src0_spad.size_per_thread = size_q_block * 1;
    octx->src1_spad.size_per_thread = factx.size_k_block * 2;
    octx->src2_spad.size_per_thread = factx.size_v_block * 2;
    octx->src3_spad.size_per_thread = mask ? factx.size_m_block * DMA_CACHE_MAX_SIZE : 0;
    octx->dst_spad.size_per_thread  = size_vkq_acc;

    octx->src0_spad.size = octx->src0_spad.size_per_thread * octx->n_threads;
    octx->src1_spad.size = octx->src1_spad.size_per_thread * octx->n_threads;
    octx->src2_spad.size = octx->src2_spad.size_per_thread * octx->n_threads;
    octx->src3_spad.size = octx->src3_spad.size_per_thread * octx->n_threads;
    octx->dst_spad.size  = octx->dst_spad.size_per_thread  * octx->n_threads;

    size_t total_spad = octx->src0_spad.size + octx->src1_spad.size + octx->src2_spad.size + octx->src3_spad.size + octx->dst_spad.size;

    if (octx->ctx->vtcm_size < total_spad) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.data = octx->ctx->vtcm_base;                        octx->src0_spad.src = NULL;
    octx->src1_spad.data = octx->src0_spad.data + octx->src0_spad.size; octx->src1_spad.src = NULL;
    octx->src2_spad.data = octx->src1_spad.data + octx->src1_spad.size; octx->src2_spad.src = NULL;
    octx->src3_spad.data = octx->src2_spad.data + octx->src2_spad.size; octx->src3_spad.src = NULL;
    octx->dst_spad.data  = octx->src3_spad.data + octx->src3_spad.size; octx->dst_spad.src  = NULL;

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        worker_pool_run_func(octx->ctx->worker_pool, flash_attn_ext_f16_thread, &factx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}
