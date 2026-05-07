#ifndef HVX_UTILS_H
#define HVX_UTILS_H

#include "hex-utils-defs.inc"

#include "hvx-types-defs.inc"
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base-defs.inc"

#define hvx_splat_pragma(x) _Pragma(#x)
#define hvx_splat_loop_body(dst_type, vec_store, unroll_cnt)     \
    do {                                                         \
        dst_type * restrict vdst = (dst_type *) dst;             \
                                                                 \
        uint32_t nvec = n / (128 / elem_size);                   \
        uint32_t nloe = n % (128 / elem_size);                   \
                                                                 \
        uint32_t i = 0;                                          \
                                                                 \
        hvx_splat_pragma(unroll(unroll_cnt))                     \
        for (; i < nvec; i++) {                                  \
            vdst[i] = src;                                       \
        }                                                        \
        if (nloe) {                                              \
            vec_store((void *) &vdst[i], nloe * elem_size, src); \
        }                                                        \
    } while(0)

static inline void hvx_splat_a(void * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    hvx_splat_loop_body(HVX_Vector, hvx_vec_store_a, 4);
}

static inline void hvx_splat_u(void * restrict dst, HVX_Vector src, uint32_t n, uint32_t elem_size) {
    hvx_splat_loop_body(HVX_UVector, hvx_vec_store_u, 4);
}

static inline void hvx_splat_f32_a(void * restrict dst, float v, uint32_t n) {
    hvx_splat_a(dst,  hvx_vec_splat_f32(v), n, sizeof(float));
}

static inline void hvx_splat_f32_u(void * restrict dst, float v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f32(v), n, sizeof(float));
}

static inline void hvx_splat_f16_a(void * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
}

static inline void hvx_splat_f16_u(void * restrict dst, _Float16 v, uint32_t n) {
    hvx_splat_u(dst,  hvx_vec_splat_f16(v), n, sizeof(__fp16));
}

static inline void hvx_splat_u16_a(void * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vh_vsplat_R(v), n, sizeof(uint16_t));
}

static inline void hvx_splat_u16_u(void * restrict dst, uint16_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vh_vsplat_R(v), n, sizeof(uint16_t));
}

static inline void hvx_splat_u8_a(void * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_a(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

static inline void hvx_splat_u8_u(void * restrict dst, uint8_t v, uint32_t n) {
    hvx_splat_u(dst,  Q6_Vb_vsplat_R(v), n, 1);
}

#define hvx_copy_loop_body(dst_type, src_type, vec_store)            \
    do {                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                 \
        src_type * restrict vsrc = (src_type *) src;                 \
                                                                     \
        const uint32_t epv  = 128 / elem_size;                       \
        const uint32_t nvec = n / epv;                               \
        const uint32_t nloe = n % epv;                               \
                                                                     \
        uint32_t i = 0;                                              \
                                                                     \
        _Pragma("unroll(4)")                                         \
        for (; i < nvec; i++) { vdst[i] = vsrc[i]; }                 \
        if (nloe) {                                                  \
            vec_store((void *) &vdst[i], nloe * elem_size, vsrc[i]); \
        }                                                            \
    } while(0)

// Generic copy routines
static inline void hvx_copy_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_copy_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_copy_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_copy_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n, uint32_t elem_size) {
    hvx_copy_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// copy n fp16 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(__fp16));
}

// copy n fp16 elements : source is aligned, destination is potentially unaligned
static inline void hvx_copy_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(__fp16));
}

// copy n fp32 elements : source and destination are aligned to HVX Vector (128)
static inline void hvx_copy_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_aa(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is aligned, destination is unaligned
static inline void hvx_copy_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_ua(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is unaligned, destination is aligned
static inline void hvx_copy_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_au(dst, src, n, sizeof(float));
}

// copy n fp32 elements : source is unaligned, destination unaligned
static inline void hvx_copy_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_uu(dst, src, n, sizeof(float));
}

//// fp32 -> fp16

#define hvx_copy_f16_f32_loop_body(dst_type, src_type, vec_store)                   \
    do {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        const uint32_t elem_size = sizeof(__fp16);                                  \
        const uint32_t epv  = 128 / elem_size;                                      \
        const uint32_t nvec = n / epv;                                              \
        const uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        for (; i < nvec; i++) {                                                     \
            vdst[i] = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);                 \
        }                                                                           \
        if (nloe) {                                                                 \
            HVX_Vector v = hvx_vec_f32_to_f16(vsrc[i*2+0], vsrc[i*2+1]);            \
            vec_store((void *) &vdst[i], nloe * elem_size, v);                      \
        }                                                                           \
    } while(0)

// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is aligned
static inline void hvx_copy_f16_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is aligned
static inline void hvx_copy_f16_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp32 elements into n fp16 elements : source is aligned, destination is unaligned
static inline void hvx_copy_f16_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp32 elements into n fp16 elements : source is unaligned, destination is unaligned
static inline void hvx_copy_f16_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f16_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

//// fp16 -> fp32

#define hvx_copy_f32_f16_loop_body(dst_type, src_type, vec_store)                   \
    do {                                                                            \
        dst_type * restrict vdst = (dst_type *) dst;                                \
        src_type * restrict vsrc = (src_type *) src;                                \
                                                                                    \
        const HVX_Vector one = hvx_vec_splat_f16(1.0);                              \
                                                                                    \
        const uint32_t elem_size = sizeof(__fp16);                                  \
        const uint32_t epv  = 128 / elem_size;                                      \
        const uint32_t nvec = n / epv;                                              \
              uint32_t nloe = n % epv;                                              \
                                                                                    \
        uint32_t i = 0;                                                             \
                                                                                    \
        _Pragma("unroll(4)")                                                        \
        for (i = 0; i < nvec; ++i) {                                                \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
            vdst[i*2]   = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));                        \
            vdst[i*2+1] = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));                        \
        }                                                                           \
                                                                                    \
        if (nloe) {                                                                 \
            HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(Q6_Vh_vshuff_Vh(vsrc[i]), one); \
                                                                                    \
            HVX_Vector vd = Q6_V_lo_W(p);                                           \
            i = 2 * i;                                                              \
                                                                                    \
            if (nloe >= 32) {                                                       \
                vdst[i] = Q6_Vsf_equals_Vqf32(vd);                                  \
                nloe -= 32; ++i; vd = Q6_V_hi_W(p);                                 \
            }                                                                       \
                                                                                    \
            if (nloe) {                                                             \
                vd = Q6_Vsf_equals_Vqf32(vd);                                       \
                hvx_vec_store_u(&vdst[i], nloe * sizeof(float), vd);                \
            }                                                                       \
        }                                                                           \
    } while(0)

// copy/convert n fp16 elements into n fp32 elements : source is aligned, destination is aligned
static inline void hvx_copy_f32_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : source is unaligned, destination is aligned
static inline void hvx_copy_f32_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

// copy/convert n fp16 elements into n fp32 elements : source is aligned, destination is unaligned
static inline void hvx_copy_f32_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

// copy/convert n fp16 elements into n fp32 elements : source is unaligned, destination is unaligned
static inline void hvx_copy_f32_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_copy_f32_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base-defs.inc"

#define hvx_scale_f32_loop_body(dst_type, src_type, vec_store)                       \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
                                                                                     \
        const uint32_t elem_size = sizeof(float);                                    \
        const uint32_t epv = 128 / elem_size;                                        \
        const uint32_t nvec = n / epv;                                               \
        const uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        for (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vdst[i]      = Q6_Vsf_equals_Vqf32(v);                                   \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs);                        \
            vec_store((void *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } while(0)

static inline void hvx_scale_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    assert((size_t) src % 128 == 0);
    hvx_scale_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    hvx_scale_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_f32_aa(dst, src, n, scale);
        } else {
            hvx_scale_f32_au(dst, src, n, scale);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_f32_ua(dst, src, n, scale);
        } else {
            hvx_scale_f32_uu(dst, src, n, scale);
        }
    }
}

#define hvx_scale_offset_f32_loop_body(dst_type, src_type, vec_store)                \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src_type * restrict vsrc = (src_type *) src;                                 \
                                                                                     \
        HVX_Vector vs = hvx_vec_splat_f32(scale);                                    \
        HVX_Vector vo = hvx_vec_splat_f32(offset);                                   \
                                                                                     \
        const uint32_t elem_size = sizeof(float);                                    \
        const uint32_t epv = 128 / elem_size;                                        \
        const uint32_t nvec = n / epv;                                               \
        const uint32_t nloe = n % epv;                                               \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        for (; i < nvec; ++i) {                                                      \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vdst[i] = Q6_Vsf_equals_Vqf32(v);                                        \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector v = Q6_Vqf32_vadd_Vqf32Vsf(Q6_Vqf32_vmpy_VsfVsf(vsrc[i], vs), vo); \
            vec_store((void *) &vdst[i], nloe * elem_size, Q6_Vsf_equals_Vqf32(v));  \
        }                                                                            \
    } while(0)

static inline void hvx_scale_offset_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) dst % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_scale_offset_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    assert((size_t) src % 128 == 0);
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    hvx_scale_offset_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_scale_offset_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int n, const float scale, const float offset) {
    if (((size_t) dst & 127) == 0) {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_aa(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f32_au(dst, src, n, scale, offset);
        }
    } else {
        if (((size_t) src & 127) == 0) {
            hvx_scale_offset_f32_ua(dst, src, n, scale, offset);
        } else {
            hvx_scale_offset_f32_uu(dst, src, n, scale, offset);
        }
    }
}


#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#include "hvx-base-defs.inc"
#include <stdbool.h>
#include <stdint.h>

#include "hvx-base-defs.inc"

#define IEEE_VSF_EXPLEN   (8)
#define IEEE_VSF_EXPBIAS  (127)
#define IEEE_VSF_EXPMASK  (0xFF)
#define IEEE_VSF_MANTLEN  (23)
#define IEEE_VSF_MANTMASK (0x7FFFFF)
#define IEEE_VSF_MIMPMASK (0x800000)

static inline HVX_Vector hvx_vec_truncate_f32(HVX_Vector in_vec) {
    HVX_Vector mask_mant_v  = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v  = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_zero_v = Q6_V_vzero();

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, in_vec);

    HVX_Vector expval_v = in_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    // negative exp == fractional value
    HVX_VectorPred q_negexp = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);

    HVX_Vector rshift_v = IEEE_VSF_MANTLEN - expval_v;         // fractional bits - exp shift

    HVX_Vector mant_v = in_vec & mask_mant_v;                  // obtain mantissa
    HVX_Vector vout   = Q6_Vw_vadd_VwVw(mant_v, mask_impl_v);  // add implicit 1.0

    vout = Q6_Vw_vasr_VwVw(vout, rshift_v);                    // shift to obtain truncated integer
    vout = Q6_V_vmux_QVV(q_negexp, const_zero_v, vout);        // expval<0 -> 0

    HVX_Vector neg_vout = -vout;

    vout = Q6_V_vmux_QVV(q_negative, neg_vout, vout);  // handle negatives

    return (vout);
}

static inline HVX_Vector hvx_vec_floor_f32(HVX_Vector in_vec) {
    HVX_Vector mask_mant_v    = Q6_V_vsplat_R(IEEE_VSF_MANTMASK);
    HVX_Vector mask_impl_v    = Q6_V_vsplat_R(IEEE_VSF_MIMPMASK);
    HVX_Vector const_mnlen_v  = Q6_V_vsplat_R(IEEE_VSF_MANTLEN);
    HVX_Vector const_zero_v   = Q6_V_vzero();
    HVX_Vector const_negone_v = Q6_V_vsplat_R(0xbf800000);  // -1 IEEE vsf

    HVX_VectorPred q_negative = Q6_Q_vcmp_gt_VwVw(const_zero_v, in_vec);

    HVX_Vector expval_v = in_vec >> IEEE_VSF_MANTLEN;
    expval_v &= IEEE_VSF_EXPMASK;
    expval_v -= IEEE_VSF_EXPBIAS;

    HVX_VectorPred q_negexp     = Q6_Q_vcmp_gt_VwVw(const_zero_v, expval_v);
    HVX_VectorPred q_expltmn    = Q6_Q_vcmp_gt_VwVw(const_mnlen_v, expval_v);
    HVX_VectorPred q_negexp_pos = Q6_Q_vcmp_gtand_QVwVw(q_negexp, in_vec, const_zero_v);
    HVX_VectorPred q_negexp_neg = Q6_Q_vcmp_gtand_QVwVw(q_negexp, const_zero_v, in_vec);

    // if expval < 0 (q_negexp)         // <0, floor is 0
    //    if vin > 0
    //       floor = 0
    //    if vin < 0
    //       floor = -1
    // if expval < mant_len (q_expltmn) // >0, but fraction may exist
    //    get sign (q_negative)
    //    mask >> expval                // fraction bits to mask off
    //    vout = ~(mask)                // apply mask to remove fraction
    //    if (qneg)                     // negative floor is one less (more, sign bit for neg)
    //      vout += ((impl_mask) >> expval)
    //    if (mask && vin)
    //      vout = vin
    // else                             // already an integer
    //    ;                             // no change

    // compute floor
    mask_mant_v >>= expval_v;
    HVX_Vector neg_addin_v    = mask_impl_v >> expval_v;
    HVX_Vector vout_neg_addin = Q6_Vw_vadd_VwVw(in_vec, neg_addin_v);
    HVX_Vector vout           = Q6_V_vmux_QVV(q_negative, vout_neg_addin, in_vec);

    HVX_Vector     mask_chk_v = Q6_V_vand_VV(in_vec, mask_mant_v);  // chk if bits set
    HVX_VectorPred q_integral = Q6_Q_vcmp_eq_VwVw(const_zero_v, mask_chk_v);

    HVX_Vector not_mask_v = Q6_V_vnot_V(mask_mant_v);        // frac bits to clear
    HVX_Vector vfrfloor_v = Q6_V_vand_VV(vout, not_mask_v);  // clear frac bits

    vout = in_vec;
    vout = Q6_V_vmux_QVV(q_expltmn, vfrfloor_v, vout);         // expval<mant
    vout = Q6_V_vmux_QVV(q_integral, in_vec, vout);            // integral values
    vout = Q6_V_vmux_QVV(q_negexp_pos, const_zero_v, vout);    // expval<0 x>0 -> 0
    vout = Q6_V_vmux_QVV(q_negexp_neg, const_negone_v, vout);  // expval<0 x<0 -> -1

    return vout;
}



#define EXP_COEFF_5 (0x39506967)  // 0.000198757 = 1/(7!)
#define EXP_COEFF_4 (0x3AB743CE)  // 0.0013982   = 1/(6!)
#define EXP_COEFF_3 (0x3C088908)  // 0.00833345  = 1/(5!)
#define EXP_COEFF_2 (0x3D2AA9C1)  // 0.416658    = 1/(4!)
#define EXP_COEFF_1 (0x3E2AAAAA)  // 0.16666667  = 1/(3!)
#define EXP_COEFF_0 (0x3F000000)  // 0.5         = 1/(2!)
#define EXP_LOGN2   (0x3F317218)  // ln(2)   = 0.6931471805
#define EXP_LOG2E   (0x3FB8AA3B)  // log2(e) = 1/ln(2) = 1.4426950408
#define EXP_ONE     (0x3f800000)  // 1.0
#define EXP_RANGE_R (0x42B17218)  // ln(FLT_MAX) approx = 88.7228
#define EXP_RANGE_L (0xC2B00000)  // -88.0 (approx log(FLT_MIN))

static inline HVX_Vector hvx_vec_exp_f32(HVX_Vector in_vec) {
    HVX_Vector z_qf32_v;
    HVX_Vector x_v;
    HVX_Vector x_qf32_v;
    HVX_Vector y_v;
    HVX_Vector k_v;
    HVX_Vector f_v;
    HVX_Vector epsilon_v;
    HVX_Vector log2e = Q6_V_vsplat_R(EXP_LOG2E);
    HVX_Vector logn2 = Q6_V_vsplat_R(EXP_LOGN2);
    HVX_Vector E_const;
    HVX_Vector zero_v = Q6_V_vzero();

    // exp(x) is approximated as follows:
    //   f = floor(x/ln(2)) = floor(x*log2(e))
    //   epsilon = x - f*ln(2)
    //   exp(x) = exp(epsilon+f*ln(2))
    //          = exp(epsilon)*exp(f*ln(2))
    //          = exp(epsilon)*2^f
    //
    //   Since epsilon is close to zero, it can be approximated with its Taylor series:
    //            exp(x) ~= 1+x+x^2/2!+x^3/3!+...+x^n/n!+...
    //   Preserving the first eight elements, we get:
    //            exp(x) ~= 1+x+e0*x^2+e1*x^3+e2*x^4+e3*x^5+e4*x^6+e5*x^7
    //                   =  1+x+(E0+(E1+(E2+(E3+(E4+E5*x)*x)*x)*x)*x)*x^2

    HVX_Vector temp_v = in_vec;

    // Clamp inputs to (-88.0, 88.0) to avoid overflow/underflow
    HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(in_vec, Q6_V_vsplat_R(EXP_RANGE_R));
    HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(Q6_V_vsplat_R(EXP_RANGE_L), in_vec);

    in_vec = Q6_V_vmux_QVV(pred_cap_right, Q6_V_vsplat_R(EXP_RANGE_R), temp_v);
    in_vec = Q6_V_vmux_QVV(pred_cap_left, Q6_V_vsplat_R(EXP_RANGE_L), in_vec);

    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(log2e, in_vec);
    epsilon_v = Q6_Vsf_equals_Vqf32(epsilon_v);

    //    f_v is the floating point result and k_v is the integer result
    f_v = hvx_vec_floor_f32(epsilon_v);
    k_v = hvx_vec_truncate_f32(f_v);

    x_qf32_v = Q6_Vqf32_vadd_VsfVsf(in_vec, zero_v);

    //  x = x - f_v * logn2;
    epsilon_v = Q6_Vqf32_vmpy_VsfVsf(f_v, logn2);
    x_qf32_v  = Q6_Vqf32_vsub_Vqf32Vqf32(x_qf32_v, epsilon_v);
    // normalize before every QFloat's vmpy
    x_qf32_v  = Q6_Vqf32_vadd_Vqf32Vsf(x_qf32_v, zero_v);

    x_v = Q6_Vsf_equals_Vqf32(x_qf32_v);

    // z = x * x;
    z_qf32_v = Q6_Vqf32_vmpy_Vqf32Vqf32(x_qf32_v, x_qf32_v);
    z_qf32_v = Q6_Vqf32_vadd_Vqf32Vsf(z_qf32_v, zero_v);

    // y = E4 + E5 * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_5);
    y_v     = Q6_Vqf32_vmpy_VsfVsf(E_const, x_v);
    E_const = Q6_V_vsplat_R(EXP_COEFF_4);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E3 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_3);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E2 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_2);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E1 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_1);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = E0 + y * x;
    E_const = Q6_V_vsplat_R(EXP_COEFF_0);
    y_v     = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, x_qf32_v);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, E_const);
    y_v     = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = x + y * z;
    y_v = Q6_Vqf32_vmpy_Vqf32Vqf32(y_v, z_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vqf32(y_v, x_qf32_v);
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, zero_v);

    // y = y + 1.0;
    y_v = Q6_Vqf32_vadd_Vqf32Vsf(y_v, Q6_V_vsplat_R(EXP_ONE));

    // insert exponents
    //        y = ldexpf(y, k);
    //    y_v += k_v; // qf32
    // modify exponent

    y_v = Q6_Vsf_equals_Vqf32(y_v);

    // add k_v to the exponent of y_v
    HVX_Vector y_v_exponent = Q6_Vw_vasl_VwR(y_v, 1);

    y_v_exponent = Q6_Vuw_vlsr_VuwR(y_v_exponent, IEEE_VSF_MANTLEN + 1);
    y_v_exponent = Q6_Vw_vadd_VwVw(k_v, y_v_exponent);

    // exponent cannot be negative; if overflow is detected, result is set to zero
    HVX_VectorPred qy_v_negative_exponent = Q6_Q_vcmp_gt_VwVw(zero_v, y_v_exponent);

    y_v = Q6_Vw_vaslacc_VwVwR(y_v, k_v, IEEE_VSF_MANTLEN);

    y_v = Q6_V_vmux_QVV(qy_v_negative_exponent, zero_v, y_v);

    return y_v;
}

static inline HVX_Vector hvx_vec_exp_f32_guard(HVX_Vector in_vec, HVX_Vector max_exp, HVX_Vector inf) {
    const HVX_VectorPred pred0 = Q6_Q_vcmp_gt_VsfVsf(in_vec, max_exp);

    HVX_Vector out = hvx_vec_exp_f32(in_vec);

    return Q6_V_vmux_QVV(pred0, inf, out);
}

static inline void hvx_exp_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int num_elems, bool negate) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == hex_is_aligned((void *) src, VLEN)) || (0 == hex_is_aligned((void *) dst, VLEN))) {
        unaligned_addr = 1;
    }
    // assert((0 == unaligned_addr) || (0 == num_elems_whole));
    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
    }

    HVX_Vector vec_out = Q6_V_vzero();

    static const float kInf    = INFINITY;
    static const float kMaxExp = 88.7228f;

    const HVX_Vector max_exp = hvx_vec_splat_f32(kMaxExp);
    const HVX_Vector inf     = hvx_vec_splat_f32(kInf);

    if (0 == unaligned_loop) {
        HVX_Vector * p_vec_in1 = (HVX_Vector *) src;
        HVX_Vector * p_vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            if (true == negate) {
                HVX_Vector neg_vec_in = hvx_vec_neg_f32(*p_vec_in1++);
                *p_vec_out++          = hvx_vec_exp_f32_guard(neg_vec_in, max_exp, inf);
            } else {
                *p_vec_out++ = hvx_vec_exp_f32_guard(*p_vec_in1++, max_exp, inf);
            }
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in = *(HVX_UVector *) (src + i * SIZEOF_FP32);

            if (true == negate) {
                HVX_Vector neg_vec_in                    = hvx_vec_neg_f32(in);
                *(HVX_UVector *) (dst + i * SIZEOF_FP32) = hvx_vec_exp_f32_guard(neg_vec_in, max_exp, inf);
            } else {
                *(HVX_UVector *) (dst + i * SIZEOF_FP32) = hvx_vec_exp_f32_guard(in, max_exp, inf);
            }
        }
    }

    if (left_over > 0) {
        const float * srcf = (float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in = *(HVX_UVector *) srcf;

        if (true == negate) {
            HVX_Vector neg_vec_in = hvx_vec_neg_f32(in);

            vec_out = hvx_vec_exp_f32_guard(neg_vec_in, max_exp, inf);
        } else {
            vec_out = hvx_vec_exp_f32_guard(in, max_exp, inf);
        }

        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, vec_out);
    }
}

#include <HAP_farf.h>

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base-defs.inc"

// ====================================================
// FUNCTION: 1/(x+1)     y(0) = 1,  y(0.5) = 0.6667, y(1) = 0.5
// Order:3; continuity: True; Ends forced: True
// Mode: unsigned;   Result fractional bits: 14
// Peak Error: 1.1295e-04  Rms Error: 2.8410e-05   Mean Error: 1.1370e-05
//      32769  -32706   31252  -10589
//      32590  -30635   22793   -4493
//      32066  -27505   16481   -2348
//      31205  -24054   11849   -1306

static inline HVX_Vector hvx_vec_recip_xp1_O3_unsigned(HVX_Vector vx) {
    // input is 0..0xffff representing 0.0  .. 1.0
    HVX_Vector p;
    p = Q6_Vh_vlut4_VuhPh(vx, 0xFAE6F6D4EE73D6A3ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x2E49406159097A14ull);
    p = Q6_Vh_vmps_VhVhVuhPuh_sat(p, vx, 0x5DF66B7177AB7FC2ull);
    p = Q6_Vh_vmpa_VhVhVuhPuh_sat(p, vx, 0x79E57D427F4E8001ull);
    return p;  // signed result, 14 fractional bits
}

// Find reciprocal of fp16.
// (1) first, convert to fp32, multiplying by 1.0; this is done to
//    handle denormals. Ignoring sign and zero, result should be at
//    least 5.9604645e-08 (32-bit code 0x33800000) and at most 131008 (0x47ffe000)
//    (exponent in range [103,143])
// (2) extract the mantissa into 16-bit unsigned; find reciprocal using a fitted poly
// (3) put this, along with '253-exp' (exp from (1)) together to make an qf32
// (4) convert that to fp16
// (5) put sign back in. Also, if the original value (w/o sign) was <0x81, replace
//     the result with the max value.
static inline HVX_Vector hvx_vec_inverse_f16(HVX_Vector vals) {
    HVX_Vector     em_mask  = Q6_Vh_vsplat_R(0x7FFF);
    HVX_Vector     avals    = Q6_V_vand_VV(vals, em_mask);
    HVX_VectorPred is_neg   = Q6_Q_vcmp_gt_VhVh(avals, vals);
    // is too small to 1/x ? for 'standard' fp16, this would be 0x101
    HVX_VectorPred is_small = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0x101), avals);

    HVX_VectorPair to_qf32  = Q6_Wqf32_vmpy_VhfVhf(avals, Q6_Vh_vsplat_R(0x3C00));  // *1.0
    HVX_Vector     to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(to_qf32));
    HVX_Vector     to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(to_qf32));

    // bits 22..13 contain the mantissa now (w/o hidden bit); move to bit 14..5 of a 16-bit vector
    HVX_Vector mant_u16 = Q6_Vh_vshuffo_VhVh(Q6_Vw_vasl_VwR(to_f32_1, 9), Q6_Vw_vasl_VwR(to_f32_0, 9));
    // likewise extract the upper 16 from each, containing the exponents in range 103..142
    HVX_Vector exp_u16  = Q6_Vh_vshuffo_VhVh(to_f32_1, to_f32_0);
    //Get exponent in IEEE 32-bit representation
    exp_u16             = Q6_Vuh_vlsr_VuhR(exp_u16, 7);

    // so, mant_u16 contains an unbiased mantissa in upper 10 bits of each u16 lane
    // We can consider it to be x-1.0, with 16 fractional bits, where 'x' is in range [1.0,2.0)
    // Use poly to transform to 1/x, with 14 fractional bits
    //
    HVX_Vector rm = hvx_vec_recip_xp1_O3_unsigned(mant_u16);

    HVX_Vector vcl0 = Q6_Vuh_vcl0_Vuh(rm);  //count leading zeros

    // Get mantissa for 16-bit representation
    HVX_Vector mant_recip = Q6_V_vand_VV(Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhVh(rm, vcl0), 5), Q6_Vh_vsplat_R(0x03FF));

    //Compute Reciprocal Exponent
    HVX_Vector exp_recip =
        Q6_Vh_vsub_VhVh(Q6_Vh_vsub_VhVh(Q6_Vh_vsplat_R(254), exp_u16), Q6_Vh_vsub_VhVh(vcl0, Q6_Vh_vsplat_R(1)));
    //Convert it for 16-bit representation
    exp_recip = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vsub_VhVh(exp_recip, Q6_Vh_vsplat_R(127)), Q6_Vh_vsplat_R(15));
    exp_recip = Q6_Vh_vasl_VhR(exp_recip, 10);

    //Merge exponent and mantissa for reciprocal
    HVX_Vector recip = Q6_V_vor_VV(exp_recip, mant_recip);
    // map 'small' inputs to standard largest value 0x7bff
    recip            = Q6_V_vmux_QVV(is_small, Q6_Vh_vsplat_R(0x7bff), recip);
    // add sign back
    recip            = Q6_V_vandor_VQR(recip, is_neg, 0x80008000);
    return recip;
}

static inline HVX_Vector hvx_vec_inverse_f32(HVX_Vector v_sf) {
    HVX_Vector inv_aprox_sf = Q6_V_vsplat_R(0x7EEEEBB3);
    HVX_Vector two_sf       = hvx_vec_splat_f32(2.0);

    // First approximation
    HVX_Vector i_sf = Q6_Vw_vsub_VwVw(inv_aprox_sf, v_sf);

    HVX_Vector r_qf;

    // Refine
    r_qf = Q6_Vqf32_vmpy_VsfVsf(
        i_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(i_sf, v_sf)))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));
    r_qf = Q6_Vqf32_vmpy_Vqf32Vqf32(
        r_qf, Q6_Vqf32_vsub_VsfVsf(two_sf, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(r_qf), v_sf))));

    return Q6_Vsf_equals_Vqf32(r_qf);
}

static inline HVX_Vector hvx_vec_inverse_f32_guard(HVX_Vector v_sf, HVX_Vector nan_inf_mask) {
    HVX_Vector out = hvx_vec_inverse_f32(v_sf);

    HVX_Vector     masked_out = Q6_V_vand_VV(out, nan_inf_mask);
    const HVX_VectorPred pred = Q6_Q_vcmp_eq_VwVw(nan_inf_mask, masked_out);

    return Q6_V_vmux_QVV(pred, Q6_V_vzero(), out);
}

#define hvx_inverse_f32_loop_body(dst_type, src_type, vec_store)             \
    do {                                                                     \
        dst_type * restrict vdst = (dst_type *) dst;                         \
        src_type * restrict vsrc = (src_type *) src;                         \
                                                                             \
        const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);           \
                                                                             \
        const uint32_t nvec = n / VLEN_FP32;                                 \
        const uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        for (; i < nvec; i++) {                                              \
             vdst[i] = hvx_vec_inverse_f32_guard(vsrc[i], nan_inf_mask);     \
        }                                                                    \
        if (nloe) {                                                          \
            HVX_Vector v = hvx_vec_inverse_f32_guard(vsrc[i], nan_inf_mask); \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP32, v);             \
        }                                                                    \
    } while(0)

static inline HVX_Vector hvx_vec_inverse_f16_guard(HVX_Vector v_sf, HVX_Vector nan_inf_mask) {
    HVX_Vector out = hvx_vec_inverse_f16(v_sf);

    HVX_Vector     masked_out = Q6_V_vand_VV(out, nan_inf_mask);
    const HVX_VectorPred pred = Q6_Q_vcmp_eq_VhVh(nan_inf_mask, masked_out);

    return Q6_V_vmux_QVV(pred, Q6_V_vzero(), out);
}

#define hvx_inverse_f16_loop_body(dst_type, src_type, vec_store)             \
    do {                                                                     \
        dst_type * restrict vdst = (dst_type *) dst;                         \
        src_type * restrict vsrc = (src_type *) src;                         \
                                                                             \
        const HVX_Vector nan_inf_mask = Q6_Vh_vsplat_R(0x7c00);              \
                                                                             \
        const uint32_t nvec = n / VLEN_FP16;                                 \
        const uint32_t nloe = n % VLEN_FP16;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        for (; i < nvec; i++) {                                              \
             vdst[i] = hvx_vec_inverse_f16_guard(vsrc[i], nan_inf_mask);     \
        }                                                                    \
        if (nloe) {                                                          \
            HVX_Vector v = hvx_vec_inverse_f16_guard(vsrc[i], nan_inf_mask); \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP16, v);             \
        }                                                                    \
    } while(0)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_INV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY) \
static inline void OP_NAME##_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src % 128 == 0); \
    OP_LOOP_BODY(HVX_Vector, HVX_Vector, hvx_vec_store_a); \
} \
static inline void OP_NAME##_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    OP_LOOP_BODY(HVX_Vector, HVX_UVector, hvx_vec_store_a); \
} \
static inline void OP_NAME##_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) { \
    assert((uintptr_t) src % 128 == 0); \
    OP_LOOP_BODY(HVX_UVector, HVX_Vector, hvx_vec_store_u); \
} \
static inline void OP_NAME##_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, hvx_vec_store_u); \
} \

// Dispatcher logic
#define HVX_INV_DISPATCHER(OP_NAME) \
static inline void OP_NAME(uint8_t * restrict dst, const uint8_t * restrict src, const uint32_t num_elems) { \
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) { \
        OP_NAME##_aa(dst, src, num_elems); \
    } else if (hex_is_aligned((void *) dst, 128)) { \
        OP_NAME##_au(dst, src, num_elems); \
    } else if (hex_is_aligned((void *) src, 128)) { \
        OP_NAME##_ua(dst, src, num_elems); \
    } else { \
        OP_NAME##_uu(dst, src, num_elems); \
    } \
}

DEFINE_HVX_INV_OP_VARIANTS(hvx_inverse_f32, hvx_inverse_f32_loop_body)
DEFINE_HVX_INV_OP_VARIANTS(hvx_inverse_f16, hvx_inverse_f16_loop_body)

HVX_INV_DISPATCHER(hvx_inverse_f32)
HVX_INV_DISPATCHER(hvx_inverse_f16)

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>

#include "hex-utils-defs.inc"
#include "hvx-base-defs.inc"
#include "hvx-types-defs.inc"

static inline HVX_Vector hvx_vec_reduce_sum_n_i32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // int32

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);     // rotate right
        sum   = Q6_Vw_vadd_VwVw(sum_t, sum);  // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_reduce_sum_i32(HVX_Vector in) {
    return hvx_vec_reduce_sum_n_i32(in, 32);
}

static inline HVX_Vector hvx_vec_reduce_sum_n_qf32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum), width);  // rotate right
        sum   = Q6_Vqf32_vadd_Vqf32Vsf(sum, sum_t);             // elementwise sum
        width = width << 1;
    }
    return sum;
}

static inline HVX_Vector hvx_vec_reduce_sum_qf32(HVX_Vector in) {
    return hvx_vec_reduce_sum_n_qf32(in, 32);
}

#if __HVX_ARCH__ > 75

static inline HVX_Vector hvx_vec_reduce_sum_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair sum_p01 = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair sum_p23 = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector  sum_sf01  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p01), Q6_V_hi_W(sum_p01));
    HVX_Vector  sum_sf23  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p23), Q6_V_hi_W(sum_p23));

    HVX_VectorPair sum_p0123 = Q6_W_vshuff_VVR(sum_sf23, sum_sf01, 8);
    HVX_Vector  sum_sf       = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sum_p0123), Q6_V_hi_W(sum_p0123));

    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 2));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 4));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 8));
    return sum_sf;
}

static inline HVX_Vector hvx_vec_reduce_sum_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  sum_sf  = Q6_Vsf_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 2));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 4));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 8));
    sum_sf = Q6_Vsf_vadd_VsfVsf(sum_sf, Q6_V_vror_VR(sum_sf, VLEN / 16));
    return sum_sf;
}

static inline HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);       // rotate right
        sum   = Q6_Vsf_vadd_VsfVsf(sum, sum_t); // elementwise sum
        width = width << 1;
    }
    return sum;
}

#else

static inline HVX_Vector hvx_vec_reduce_sum_f32x4(HVX_Vector_x4 in) {
    HVX_VectorPair sum_p01  = Q6_W_vshuff_VVR(in.v[1], in.v[0], 4);
    HVX_VectorPair sum_p23  = Q6_W_vshuff_VVR(in.v[3], in.v[2], 4);
    HVX_Vector     sum_qf01 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p01), Q6_V_hi_W(sum_p01));
    HVX_Vector     sum_qf23 = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p23), Q6_V_hi_W(sum_p23));

    HVX_VectorPair sum_p0123 = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(sum_qf23), Q6_Vsf_equals_Vqf32(sum_qf01), 8);
    HVX_Vector     sum_qf    = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sum_p0123), Q6_V_hi_W(sum_p0123));

    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 2));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 4));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 8));
    return Q6_Vsf_equals_Vqf32(sum_qf);
}

static inline HVX_Vector hvx_vec_reduce_sum_f32x2(HVX_Vector in0, HVX_Vector in1) {
    HVX_VectorPair sump = Q6_W_vshuff_VVR(in1, in0, 4);
    HVX_Vector  sum_qf  = Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(sump), Q6_V_hi_W(sump));

    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 2));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 4));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 8));
    sum_qf = Q6_Vqf32_vadd_Vqf32Vsf(sum_qf, Q6_V_vror_VR(Q6_Vsf_equals_Vqf32(sum_qf), VLEN / 16));
    return Q6_Vsf_equals_Vqf32(sum_qf);
}

static inline HVX_Vector hvx_vec_reduce_sum_n_f32(HVX_Vector in, unsigned int n) {
    unsigned int total = n * 4;  // total vec nbytes
    unsigned int width = 4;      // fp32 nbytes

    HVX_Vector sum = in, sum_t;
    while (width < total) {
        sum_t = Q6_V_vror_VR(sum, width);                               // rotate right
        sum   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sum, sum_t));  // elementwise sum
        width = width << 1;
    }
    return sum;
}

#endif

static inline HVX_Vector hvx_vec_reduce_sum_f32(HVX_Vector in) {
    return hvx_vec_reduce_sum_n_f32(in, 32);
}

static inline HVX_Vector hvx_vec_reduce_max_f16(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp16 nbytes

    HVX_Vector _max = in, _max_t;
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max2_f16(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 2;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vhf_vmax_VhfVhf(in, _max);
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vhf_vmax_VhfVhf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max_f32(HVX_Vector in) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max = in, _max_t;
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

static inline HVX_Vector hvx_vec_reduce_max2_f32(HVX_Vector in, HVX_Vector _max) {
    unsigned total = 128;  // total vec nbytes
    unsigned width = 4;    // fp32 nbytes

    HVX_Vector _max_t;

    _max = Q6_Vsf_vmax_VsfVsf(in, _max);
    while (width < total) {
        _max_t = Q6_V_vror_VR(_max, width);         // rotate right
        _max   = Q6_Vsf_vmax_VsfVsf(_max_t, _max);  // elementwise max
        width  = width << 1;
    }

    return _max;
}

#define hvx_reduce_loop_body(src_type, init_vec, pad_vec, vec_op, reduce_op, scalar_reduce) \
    do {                                                                                    \
        src_type * restrict vsrc = (src_type *) src;                                        \
        HVX_Vector acc = init_vec;                                                          \
                                                                                            \
        const uint32_t elem_size = sizeof(float);                                           \
        const uint32_t epv  = 128 / elem_size;                                              \
        const uint32_t nvec = num_elems / epv;                                              \
        const uint32_t nloe = num_elems % epv;                                              \
                                                                                            \
        uint32_t i = 0;                                                                     \
        _Pragma("unroll(4)")                                                                \
        for (; i < nvec; i++) {                                                             \
            acc = vec_op(acc, vsrc[i]);                                                     \
        }                                                                                   \
        if (nloe) {                                                                         \
            const float * srcf = (const float *) src + i * epv;                             \
            HVX_Vector in = *(HVX_UVector *) srcf;                                          \
            HVX_Vector temp = Q6_V_valign_VVR(in, pad_vec, nloe * elem_size);               \
            acc = vec_op(acc, temp);                                                        \
        }                                                                                   \
        HVX_Vector v = reduce_op(acc);                                                      \
        return scalar_reduce(v);                                                            \
    } while(0)

#define HVX_REDUCE_MAX_OP(acc, val) Q6_Vsf_vmax_VsfVsf(acc, val)
#define HVX_REDUCE_SUM_OP(acc, val) Q6_Vqf32_vadd_VsfVsf(Q6_Vsf_equals_Vqf32(acc), val)
#define HVX_SUM_SQ_OP(acc, val) Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(val, val))
#define HVX_REDUCE_MAX_SCALAR(v) hvx_vec_get_f32(v)
#define HVX_REDUCE_SUM_SCALAR(v) hvx_vec_get_f32(Q6_Vsf_equals_Vqf32(v))

// Max variants

static inline float hvx_reduce_max_f32_a(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((const float *) src)[0]);
    assert((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

static inline float hvx_reduce_max_f32_u(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = hvx_vec_splat_f32(((const float *) src)[0]);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_MAX_OP, hvx_vec_reduce_max_f32, HVX_REDUCE_MAX_SCALAR);
}

static inline float hvx_reduce_max_f32(const uint8_t * restrict src, const int num_elems) {
    if (hex_is_aligned((void *) src, 128)) {
        return hvx_reduce_max_f32_a(src, num_elems);
    } else {
        return hvx_reduce_max_f32_u(src, num_elems);
    }
}

// Sum variants

static inline float hvx_reduce_sum_f32_a(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    assert((unsigned long) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_REDUCE_SUM_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

static inline float hvx_reduce_sum_f32_u(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_REDUCE_SUM_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

static inline float hvx_reduce_sum_f32(const uint8_t * restrict src, const int num_elems) {
    if (hex_is_aligned((void *) src, 128)) {
        return hvx_reduce_sum_f32_a(src, num_elems);
    } else {
        return hvx_reduce_sum_f32_u(src, num_elems);
    }
}

// Sum of squares variants

static inline float hvx_sum_of_squares_f32_a(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    assert((uintptr_t) src % 128 == 0);
    hvx_reduce_loop_body(HVX_Vector, init_vec, init_vec, HVX_SUM_SQ_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

static inline float hvx_sum_of_squares_f32_u(const uint8_t * restrict src, const int num_elems) {
    HVX_Vector init_vec = Q6_V_vsplat_R(0);
    hvx_reduce_loop_body(HVX_UVector, init_vec, init_vec, HVX_SUM_SQ_OP, hvx_vec_reduce_sum_qf32, HVX_REDUCE_SUM_SCALAR);
}

static inline float hvx_sum_of_squares_f32(const uint8_t * restrict src, const int num_elems) {
    if (hex_is_aligned((void *) src, 128)) {
        return hvx_sum_of_squares_f32_a(src, num_elems);
    } else {
        return hvx_sum_of_squares_f32_u(src, num_elems);
    }
}

#undef hvx_reduce_loop_body
#undef HVX_REDUCE_MAX_OP
#undef HVX_REDUCE_SUM_OP
#undef HVX_REDUCE_MAX_SCALAR
#undef HVX_REDUCE_SUM_SCALAR
#undef HVX_SUM_SQ_OP

#include "hvx-base-defs.inc"

#define FAST_SIGMOID_LOG2F (0x3fb8aa3b)  // 1.442695022
#define FAST_SIGMOID_C1    (0x3d009076)  // 0.03138777
#define FAST_SIGMOID_C2    (0x3e8d74bd)  // 0.276281267
#define FAST_SIGMOID_C3    (0x3f000000)  // 0.5

static inline HVX_Vector hvx_vec_fast_sigmoid_f32(HVX_Vector v) {
    v = Q6_Vqf32_vmpy_VsfVsf(v, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));
    v = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v), Q6_V_vsplat_R(FAST_SIGMOID_C3));

    HVX_Vector in_int = hvx_vec_truncate_f32(Q6_Vsf_equals_Vqf32(v));
    HVX_Vector x      = Q6_Vqf32_vsub_Vqf32Vsf(v, Q6_Vsf_equals_Vw(in_int));
    HVX_Vector xx     = Q6_Vqf32_vmpy_Vqf32Vqf32(x, x);

    HVX_Vector v1 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(xx), Q6_V_vsplat_R(FAST_SIGMOID_C2));
    v1            = Q6_Vqf32_vadd_Vqf32Vsf(v1, Q6_V_vsplat_R(FAST_SIGMOID_LOG2F));

    HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(x), Q6_V_vsplat_R(FAST_SIGMOID_C1));
    v2            = Q6_Vqf32_vmpy_Vqf32Vqf32(v2, xx);
    v2            = Q6_Vqf32_vadd_Vqf32Vqf32(v2, x);

    HVX_Vector v3          = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v2, v1));
    HVX_Vector v3_exponent = Q6_Vw_vasl_VwR(v3, 1);
    v3_exponent            = Q6_Vuw_vlsr_VuwR(v3_exponent, 24);
    v3_exponent            = Q6_Vw_vadd_VwVw(in_int, v3_exponent);
    v3                     = Q6_Vw_vaslacc_VwVwR(v3, in_int, 24);

    HVX_Vector v4 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(v2, v1));
    HVX_Vector v5 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(v3, v4));

    HVX_Vector res = hvx_vec_inverse_f32(v5);
    res            = Q6_Vqf32_vmpy_VsfVsf(v3, res);

    return Q6_Vsf_equals_Vqf32(res);
}

static inline HVX_Vector hvx_vec_fast_sigmoid_f32_guard(HVX_Vector v,
                                                         HVX_Vector one,
                                                         HVX_Vector max_exp,
                                                         HVX_Vector min_exp) {
    const HVX_VectorPred pred_max = Q6_Q_vcmp_gt_VsfVsf(max_exp, v);
    const HVX_VectorPred pred_min = Q6_Q_vcmp_gt_VsfVsf(v, min_exp);

    HVX_Vector out = hvx_vec_fast_sigmoid_f32(v);
    out            = Q6_V_vmux_QVV(pred_max, out, one);
    return Q6_V_vmux_QVV(pred_min, out, Q6_V_vzero());
}

static inline HVX_Vector hvx_vec_tanh_f32(HVX_Vector x) {
    // tanh(x) = 2 * sigmoid(2x) - 1
    HVX_Vector two = hvx_vec_splat_f32(2.0f);
    HVX_Vector one = hvx_vec_splat_f32(1.0f);
    HVX_Vector x2  = Q6_Vqf32_vmpy_VsfVsf(x, two);

    HVX_Vector max_exp = hvx_vec_splat_f32(87.f);
    HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);

    HVX_Vector sig2x = hvx_vec_fast_sigmoid_f32_guard(Q6_Vsf_equals_Vqf32(x2), one, max_exp, min_exp);

    HVX_Vector res = Q6_Vqf32_vmpy_VsfVsf(sig2x, two);
    res = Q6_Vqf32_vsub_Vqf32Vsf(res, one);
    return Q6_Vsf_equals_Vqf32(res);
}

#define hvx_sigmoid_loop_body(dst_type, src_type, vec_store)    \
    do {                                                        \
        dst_type * restrict vdst = (dst_type *) dst;            \
        src_type * restrict vsrc = (src_type *) src;            \
                                                                \
        const HVX_Vector one     = hvx_vec_splat_f32(1.f);      \
        const HVX_Vector max_exp = hvx_vec_splat_f32(87.f);     \
        const HVX_Vector min_exp = hvx_vec_splat_f32(-87.f);    \
                                                                \
        const uint32_t epv  = 128 / sizeof(float);              \
        const uint32_t nvec = n / epv;                          \
        const uint32_t nloe = n % epv;                          \
                                                                \
        uint32_t i = 0;                                         \
                                                                \
        _Pragma("unroll(4)")                                    \
        for (; i < nvec; i++) {                                 \
             vdst[i] = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp); \
        }                                                       \
        if (nloe) {                                             \
             HVX_Vector tmp = hvx_vec_fast_sigmoid_f32_guard(vsrc[i], one, max_exp, min_exp); \
             vec_store((void *) &vdst[i], nloe * sizeof(float), tmp); \
        }                                                       \
    } while(0)

#define hvx_tanh_loop_body(dst_type, src_type, vec_store)       \
    do {                                                        \
        dst_type * restrict vdst = (dst_type *) dst;            \
        src_type * restrict vsrc = (src_type *) src;            \
                                                                \
        const uint32_t epv  = 128 / sizeof(float);              \
        const uint32_t nvec = n / epv;                          \
        const uint32_t nloe = n % epv;                          \
                                                                \
        uint32_t i = 0;                                         \
                                                                \
        _Pragma("unroll(4)")                                    \
        for (; i < nvec; i++) {                                 \
             vdst[i] = hvx_vec_tanh_f32(vsrc[i]);               \
        }                                                       \
        if (nloe) {                                             \
             HVX_Vector tmp = hvx_vec_tanh_f32(vsrc[i]);        \
             vec_store((void *) &vdst[i], nloe * sizeof(float), tmp); \
        }                                                       \
    } while(0)

static inline void hvx_sigmoid_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_sigmoid_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_sigmoid_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_sigmoid_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_sigmoid_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_sigmoid_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_sigmoid_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_tanh_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_tanh_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

#include <stdbool.h>
#include <stdint.h>

#include "hex-utils-defs.inc"

#include "hvx-base-defs.inc"

#define RSQRT_CONST        0x5f3759df  // Constant for fast inverse square root calculation
#define RSQRT_ONE_HALF     0x3f000000  // 0.5
#define RSQRT_THREE_HALVES 0x3fc00000  // 1.5

#if __HVX_ARCH__ < 79
#define HVX_OP_MUL(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#else
#define HVX_OP_MUL(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#endif

static inline HVX_Vector hvx_vec_rsqrt_f32(HVX_Vector in_vec) {
    //Algorithm :
    //  x2 = input*0.5
    //  y  = * (long *) &input
    //  y  = 0x5f3759df - (y>>1)
    //  y  = y*(threehalfs - x2*y*y)

    HVX_Vector rsqrtconst = Q6_V_vsplat_R(RSQRT_CONST);
    HVX_Vector onehalf    = Q6_V_vsplat_R(RSQRT_ONE_HALF);
    HVX_Vector threehalfs = Q6_V_vsplat_R(RSQRT_THREE_HALVES);

    HVX_Vector x2, y, ypower2, temp;

    x2 = Q6_Vqf32_vmpy_VsfVsf(in_vec, onehalf);
    x2 = Q6_Vqf32_vadd_Vqf32Vsf(x2, Q6_V_vzero());

    y = Q6_Vw_vasr_VwR(in_vec, 1);
    y = Q6_Vw_vsub_VwVw(rsqrtconst, y);

    // 1st iteration
    ypower2 = Q6_Vqf32_vmpy_VsfVsf(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_VsfVsf(y, Q6_Vsf_equals_Vqf32(temp));

    // 2nd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    // 3rd iteration
    y       = Q6_Vqf32_vadd_Vqf32Vsf(temp, Q6_V_vzero());
    ypower2 = Q6_Vqf32_vmpy_Vqf32Vqf32(y, y);
    ypower2 = Q6_Vqf32_vadd_Vqf32Vsf(ypower2, Q6_V_vzero());
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(x2, ypower2);
    temp    = Q6_Vqf32_vsub_VsfVsf(threehalfs, Q6_Vsf_equals_Vqf32(temp));
    temp    = Q6_Vqf32_vmpy_Vqf32Vqf32(y, temp);

    return Q6_Vsf_equals_Vqf32(temp);
}

// Compute sqrt(x) as x*inv_sqrt(x)
#define hvx_sqrt_f32_loop_body(dst_type, src_type, vec_store)                \
    do {                                                                     \
        dst_type * restrict vdst = (dst_type *) dst;                         \
        src_type * restrict vsrc = (src_type *) src;                         \
                                                                             \
        const uint32_t nvec = n / VLEN_FP32;                                 \
        const uint32_t nloe = n % VLEN_FP32;                                 \
                                                                             \
        uint32_t i = 0;                                                      \
                                                                             \
        _Pragma("unroll(4)")                                                 \
        for (; i < nvec; i++) {                                              \
            HVX_Vector inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_Vector sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vdst[i] = sqrt_res;                                              \
        }                                                                    \
        if (nloe) {                                                          \
            HVX_Vector inv_sqrt = hvx_vec_rsqrt_f32(vsrc[i]);                \
            HVX_Vector sqrt_res = HVX_OP_MUL(inv_sqrt, vsrc[i]);             \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP32, sqrt_res);      \
        }                                                                    \
    } while(0)

static inline void hvx_sqrt_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_sqrt_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_sqrt_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_sqrt_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_sqrt_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_sqrt_f32(uint8_t * restrict dst, const uint8_t * restrict src, const int num_elems) {
    if ((unsigned long) dst % 128 == 0) {
        if ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f32_aa(dst, src, num_elems);
        } else {
            hvx_sqrt_f32_au(dst, src, num_elems);
        }
    } else {
        if ((unsigned long) src % 128 == 0) {
            hvx_sqrt_f32_ua(dst, src, num_elems);
        } else {
            hvx_sqrt_f32_uu(dst, src, num_elems);
        }
    }
}


#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include "hvx-base-defs.inc"
#include "hex-utils-defs.inc"

//
// Binary operations (add, mul, sub)
//

#define UNUSED(x) (void)(x)

#define hvx_arith_loop_body(dst_type, src0_type, src1_type, elem_size, vec_store, vec_op) \
    do {                                                                       \
        dst_type * restrict vdst  = (dst_type *) dst;                          \
        src0_type * restrict vsrc0 = (src0_type *) src0;                       \
        src1_type * restrict vsrc1 = (src1_type *) src1;                       \
                                                                               \
        const uint32_t epv  = 128 / (elem_size);                               \
        const uint32_t nvec = n / epv;                                         \
        const uint32_t nloe = n % epv;                                         \
                                                                               \
        uint32_t i = 0;                                                        \
                                                                               \
        _Pragma("unroll(4)")                                                   \
        for (; i < nvec; i++) {                                                \
            vdst[i] = vec_op(vsrc0[i], vsrc1[i]);                              \
        }                                                                      \
        if (nloe) {                                                            \
            HVX_Vector v = vec_op(vsrc0[i], vsrc1[i]);                         \
            vec_store((void *) &vdst[i], nloe * (elem_size), v);               \
        }                                                                      \
    } while(0)

#if __HVX_ARCH__ < 79

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b))
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b))
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))

#else

#define HVX_OP_ADD_F32(a, b) Q6_Vsf_vadd_VsfVsf(a, b)
#define HVX_OP_SUB_F32(a, b) Q6_Vsf_vsub_VsfVsf(a, b)
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)

#endif

#define HVX_OP_ADD_F16(a, b) hvx_vec_add_f16_f16(a, b)
#define HVX_OP_SUB_F16(a, b) hvx_vec_sub_f16_f16(a, b)
#define HVX_OP_MUL_F16(a, b) hvx_vec_mul_f16_f16(a, b)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_BINARY_OP_VARIANTS(OP_NAME, OP_MACRO, ELEM_TYPE) \
static inline void OP_NAME##_aaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src0 % 128 == 0); \
    assert((uintptr_t) src1 % 128 == 0); \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_aau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src0 % 128 == 0); \
    hvx_arith_loop_body(HVX_Vector, HVX_Vector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_aua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src1 % 128 == 0); \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_auu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    hvx_arith_loop_body(HVX_Vector, HVX_UVector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_uaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src0 % 128 == 0); \
    assert((uintptr_t) src1 % 128 == 0); \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \
static inline void OP_NAME##_uau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src0 % 128 == 0); \
    hvx_arith_loop_body(HVX_UVector, HVX_Vector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \
static inline void OP_NAME##_uua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src1 % 128 == 0); \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \
static inline void OP_NAME##_uuu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    hvx_arith_loop_body(HVX_UVector, HVX_UVector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \

DEFINE_HVX_BINARY_OP_VARIANTS(hvx_add_f32, HVX_OP_ADD_F32, float)
DEFINE_HVX_BINARY_OP_VARIANTS(hvx_sub_f32, HVX_OP_SUB_F32, float)
DEFINE_HVX_BINARY_OP_VARIANTS(hvx_mul_f32, HVX_OP_MUL_F32, float)

DEFINE_HVX_BINARY_OP_VARIANTS(hvx_add_f16, HVX_OP_ADD_F16, _Float16)
DEFINE_HVX_BINARY_OP_VARIANTS(hvx_sub_f16, HVX_OP_SUB_F16, _Float16)
DEFINE_HVX_BINARY_OP_VARIANTS(hvx_mul_f16, HVX_OP_MUL_F16, _Float16)

// Dispatcher logic
#define HVX_BINARY_DISPATCHER(OP_NAME) \
static inline void OP_NAME(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint32_t num_elems) { \
    if (hex_is_aligned((void *) dst, 128)) { \
        if (hex_is_aligned((void *) src0, 128)) { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_aau(dst, src0, src1, num_elems); \
        } else { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_aua(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_auu(dst, src0, src1, num_elems); \
        } \
    } else { \
        if (hex_is_aligned((void *) src0, 128)) { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_uau(dst, src0, src1, num_elems); \
        } else { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_uua(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_uuu(dst, src0, src1, num_elems); \
        } \
    } \
}

HVX_BINARY_DISPATCHER(hvx_add_f32)
HVX_BINARY_DISPATCHER(hvx_sub_f32)
HVX_BINARY_DISPATCHER(hvx_mul_f32)

HVX_BINARY_DISPATCHER(hvx_add_f16)
HVX_BINARY_DISPATCHER(hvx_sub_f16)
HVX_BINARY_DISPATCHER(hvx_mul_f16)

// Mul-Mul Optimized
static inline void hvx_mul_mul_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint8_t * restrict src2, const uint32_t num_elems) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src0 % 128 == 0);
    assert((unsigned long) src1 % 128 == 0);
    assert((unsigned long) src2 % 128 == 0);

    HVX_Vector * restrict vdst  = (HVX_Vector *) dst;
    HVX_Vector * restrict vsrc0 = (HVX_Vector *) src0;
    HVX_Vector * restrict vsrc1 = (HVX_Vector *) src1;
    HVX_Vector * restrict vsrc2 = (HVX_Vector *) src2;

    const uint32_t elem_size = sizeof(float);
    const uint32_t epv  = 128 / elem_size;
    const uint32_t nvec = num_elems / epv;
    const uint32_t nloe = num_elems % epv;

    uint32_t i = 0;

    _Pragma("unroll(4)")
    for (; i < nvec; i++) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        vdst[i] = HVX_OP_MUL(v1, vsrc2[i]);
    }

    if (nloe) {
        HVX_Vector v1 = HVX_OP_MUL_F32(vsrc0[i], vsrc1[i]);
        HVX_Vector v2 = HVX_OP_MUL_F32(v1, vsrc2[i]);
        hvx_vec_store_a((void *) &vdst[i], nloe * elem_size, v2);
    }
}

// Scalar Operations

#define hvx_scalar_loop_body(dst_type, src_type, elem_size, vec_store, scalar_op_macro)   \
    do {                                                                       \
        dst_type * restrict vdst = (dst_type *) dst;                           \
        src_type * restrict vsrc = (src_type *) src;                           \
                                                                               \
        const uint32_t epv  = 128 / (elem_size);                               \
        const uint32_t nvec = n / epv;                                         \
        const uint32_t nloe = n % epv;                                         \
                                                                               \
        uint32_t i = 0;                                                        \
                                                                               \
        _Pragma("unroll(4)")                                                   \
        for (; i < nvec; i++) {                                                \
            HVX_Vector v = vsrc[i];                                            \
            vdst[i] = scalar_op_macro(v);                                      \
        }                                                                      \
        if (nloe) {                                                            \
            HVX_Vector v = vsrc[i];                                            \
            v = scalar_op_macro(v);                                            \
            vec_store((void *) &vdst[i], nloe * (elem_size), v);               \
        }                                                                      \
    } while(0)

#define HVX_OP_ADD_SCALAR_F32(v) \
    ({ \
        const HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VwVw(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F32(v, val_vec); \
        Q6_V_vmux_QVV(pred_inf, inf, out); \
    })

#define HVX_OP_MUL_SCALAR_F32(v) HVX_OP_MUL_F32(v, val_vec)
#define HVX_OP_SUB_SCALAR_F32(v) HVX_OP_SUB_F32(v, val_vec)

#define HVX_OP_ADD_SCALAR_F16(v) \
    ({ \
        const HVX_VectorPred pred_inf = Q6_Q_vcmp_eq_VhVh(inf, v); \
        HVX_Vector out = HVX_OP_ADD_F16(v, val_vec); \
        Q6_V_vmux_QVV(pred_inf, inf, out); \
    })

#define HVX_OP_MUL_SCALAR_F16(v) HVX_OP_MUL_F16(v, val_vec)
#define HVX_OP_SUB_SCALAR_F16(v) HVX_OP_SUB_F16(v, val_vec)

// Scalar Variants

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(OP_NAME, OP_MACRO, SPLAT_MACRO, ELEM_TYPE) \
static inline void OP_NAME##_aa(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, uint32_t n) { \
    const HVX_Vector val_vec = SPLAT_MACRO(val); \
    const HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf); \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src % 128 == 0); \
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_au(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, uint32_t n) { \
    const HVX_Vector val_vec = SPLAT_MACRO(val); \
    const HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf); \
    assert((uintptr_t) dst % 128 == 0); \
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_a, OP_MACRO); \
} \
static inline void OP_NAME##_ua(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, uint32_t n) { \
    const HVX_Vector val_vec = SPLAT_MACRO(val); \
    const HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf); \
    assert((uintptr_t) src % 128 == 0); \
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \
static inline void OP_NAME##_uu(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, uint32_t n) { \
    const HVX_Vector val_vec = SPLAT_MACRO(val); \
    const HVX_Vector inf = SPLAT_MACRO((ELEM_TYPE)INFINITY); UNUSED(inf); \
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, sizeof(ELEM_TYPE), hvx_vec_store_u, OP_MACRO); \
} \

DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_add_scalar_f32, HVX_OP_ADD_SCALAR_F32, hvx_vec_splat_f32, float)
DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_sub_scalar_f32, HVX_OP_SUB_SCALAR_F32, hvx_vec_splat_f32, float)
DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_mul_scalar_f32, HVX_OP_MUL_SCALAR_F32, hvx_vec_splat_f32, float)

DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_add_scalar_f16, HVX_OP_ADD_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_sub_scalar_f16, HVX_OP_SUB_SCALAR_F16, hvx_vec_splat_f16, _Float16)
DEFINE_HVX_BINARY_SCALAR_OP_VARIANTS(hvx_mul_scalar_f16, HVX_OP_MUL_SCALAR_F16, hvx_vec_splat_f16, _Float16)

// Dispatcher logic
#define HVX_BINARY_SCALAR_DISPATCHER(OP_NAME, ELEM_TYPE) \
static inline void OP_NAME(uint8_t * restrict dst, const uint8_t * restrict src, const ELEM_TYPE val, const uint32_t num_elems) { \
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) { \
        OP_NAME##_aa(dst, src, val, num_elems); \
    } else if (hex_is_aligned((void *) dst, 128)) { \
        OP_NAME##_au(dst, src, val, num_elems); \
    } else if (hex_is_aligned((void *) src, 128)) { \
        OP_NAME##_ua(dst, src, val, num_elems); \
    } else { \
        OP_NAME##_uu(dst, src, val, num_elems); \
    } \
}

HVX_BINARY_SCALAR_DISPATCHER(hvx_add_scalar_f32, float)
HVX_BINARY_SCALAR_DISPATCHER(hvx_sub_scalar_f32, float)
HVX_BINARY_SCALAR_DISPATCHER(hvx_mul_scalar_f32, float)

HVX_BINARY_SCALAR_DISPATCHER(hvx_add_scalar_f16, _Float16)
HVX_BINARY_SCALAR_DISPATCHER(hvx_sub_scalar_f16, _Float16)
HVX_BINARY_SCALAR_DISPATCHER(hvx_mul_scalar_f16, _Float16)

// MIN Scalar variants

#define HVX_OP_MIN_SCALAR(v) Q6_Vsf_vmin_VsfVsf(val_vec, v)

static inline void hvx_min_scalar_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, sizeof(float), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, sizeof(float), hvx_vec_store_a, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, sizeof(float), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const float val, uint32_t n) {
    const HVX_Vector val_vec = hvx_vec_splat_f32(val);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, sizeof(float), hvx_vec_store_u, HVX_OP_MIN_SCALAR);
}

static inline void hvx_min_scalar_f32(uint8_t * restrict dst, const uint8_t * restrict src, const float val, const int num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) {
        hvx_min_scalar_f32_aa(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) dst, 128)) {
        hvx_min_scalar_f32_au(dst, src, val, num_elems);
    } else if (hex_is_aligned((void *) src, 128)) {
        hvx_min_scalar_f32_ua(dst, src, val, num_elems);
    } else {
        hvx_min_scalar_f32_uu(dst, src, val, num_elems);
    }
}

// CLAMP Scalar variants

#define HVX_OP_CLAMP_SCALAR(v) \
    ({ \
        HVX_VectorPred pred_cap_right = Q6_Q_vcmp_gt_VsfVsf(v, max_vec); \
        HVX_VectorPred pred_cap_left  = Q6_Q_vcmp_gt_VsfVsf(min_vec, v); \
        HVX_Vector tmp = Q6_V_vmux_QVV(pred_cap_right, max_vec, v); \
        Q6_V_vmux_QVV(pred_cap_left, min_vec, tmp); \
    })

static inline void hvx_clamp_scalar_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_Vector, sizeof(float), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    assert((unsigned long) dst % 128 == 0);
    hvx_scalar_loop_body(HVX_Vector, HVX_UVector, sizeof(float), hvx_vec_store_a, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    assert((unsigned long) src % 128 == 0);
    hvx_scalar_loop_body(HVX_UVector, HVX_Vector, sizeof(float), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, uint32_t n) {
    const HVX_Vector min_vec = hvx_vec_splat_f32(min);
    const HVX_Vector max_vec = hvx_vec_splat_f32(max);
    hvx_scalar_loop_body(HVX_UVector, HVX_UVector, sizeof(float), hvx_vec_store_u, HVX_OP_CLAMP_SCALAR);
}

static inline void hvx_clamp_scalar_f32(uint8_t * restrict dst, const uint8_t * restrict src, const float min, const float max, const int num_elems) {
    if (hex_is_aligned((void *) dst, 128) && hex_is_aligned((void *) src, 128)) {
        hvx_clamp_scalar_f32_aa(dst, src, min, max, num_elems);
    } else if (hex_is_aligned((void *) dst, 128)) {
        hvx_clamp_scalar_f32_au(dst, src, min, max, num_elems);
    } else if (hex_is_aligned((void *) src, 128)) {
        hvx_clamp_scalar_f32_ua(dst, src, min, max, num_elems);
    } else {
        hvx_clamp_scalar_f32_uu(dst, src, min, max, num_elems);
    }
}

//
// Square
//

#define hvx_sqr_f32_loop_body(dst_type, src_type, vec_store)           \
    do {                                                                   \
        dst_type * restrict vdst  = (dst_type *) dst;                      \
        src_type * restrict vsrc = (src_type *) src;                       \
                                                                           \
        const uint32_t elem_size = sizeof(float);                          \
        const uint32_t epv  = 128 / elem_size;                             \
        const uint32_t nvec = n / epv;                                     \
        const uint32_t nloe = n % epv;                                     \
                                                                           \
        uint32_t i = 0;                                                    \
                                                                           \
        _Pragma("unroll(4)")                                               \
        for (; i < nvec; i++) {                                            \
            vdst[i] = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                        \
        }                                                                  \
        if (nloe) {                                                        \
            HVX_Vector v = HVX_OP_MUL_F32(vsrc[i], vsrc[i]);                   \
            vec_store((void *) &vdst[i], nloe * elem_size, v);             \
        }                                                                  \
    } while(0)

static inline void hvx_sqr_f32_aa(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    assert((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}

static inline void hvx_sqr_f32_au(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) dst % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}

static inline void hvx_sqr_f32_ua(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    assert((unsigned long) src % 128 == 0);
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}

static inline void hvx_sqr_f32_uu(uint8_t * restrict dst, const uint8_t * restrict src, uint32_t n) {
    hvx_sqr_f32_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

static inline void hvx_sqr_f32(uint8_t * restrict dst, const uint8_t * restrict src, const uint32_t num_elems) {
    if (hex_is_aligned((void *) dst, 128)) {
        if (hex_is_aligned((void *) src, 128)) {
            hvx_sqr_f32_aa(dst, src, num_elems);
        } else {
            hvx_sqr_f32_au(dst, src, num_elems);
        }
    } else {
        if (hex_is_aligned((void *) src, 128)) {
            hvx_sqr_f32_ua(dst, src, num_elems);
        } else {
            hvx_sqr_f32_uu(dst, src, num_elems);
        }
    }
}

#undef HVX_OP_ADD_F32
#undef HVX_OP_SUB_F32
#undef HVX_OP_MUL_F32
#undef HVX_OP_ADD_F16
#undef HVX_OP_SUB_F16
#undef HVX_OP_MUL_F16
#undef hvx_arith_loop_body
#undef HVX_OP_ADD_SCALAR_F32
#undef HVX_OP_SUB_SCALAR_F32
#undef HVX_OP_MUL_SCALAR_F32
#undef HVX_OP_ADD_SCALAR_F16
#undef HVX_OP_SUB_SCALAR_F16
#undef HVX_OP_MUL_SCALAR_F16
#undef hvx_scalar_loop_body
#undef HVX_OP_MIN_SCALAR
#undef HVX_OP_CLAMP_SCALAR
#undef DEFINE_HVX_BINARY_OP_VARIANTS
#undef HVX_BINARY_DISPATCHER
#undef UNUSED

#include <HAP_farf.h>

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "hvx-base-defs.inc"
#include "hex-utils-defs.inc"

#if __HVX_ARCH__ < 79
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b))
#define HVX_OP_MUL_F16(a, b) Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(a, b))
#else
#define HVX_OP_MUL_F32(a, b) Q6_Vsf_vmpy_VsfVsf(a, b)
#define HVX_OP_MUL_F16(a, b) Q6_Vhf_vmpy_VhfVhf(a, b)
#endif

// Compute div by scaler in f32. Requires first by expanding fp32 to fp16 and converting the result back to fp32.
static inline HVX_Vector hvx_div_mul_f16_const_using_f32(HVX_Vector vec1_hf, HVX_Vector vec2_sf_const, HVX_Vector vec_hf_one_1_0) {
#if __HVX_ARCH__ < 79
    HVX_VectorPair src_to_f32 = Q6_Wqf32_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_Vector src_to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(src_to_f32));
    HVX_Vector src_to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(src_to_f32));
#else
    HVX_VectorPair src_to_f32 = Q6_Wsf_vmpy_VhfVhf(vec1_hf, vec_hf_one_1_0);
    HVX_Vector src_to_f32_0 = Q6_V_lo_W(src_to_f32);
    HVX_Vector src_to_f32_1 = Q6_V_hi_W(src_to_f32);
#endif

    HVX_Vector div_f32_0 = HVX_OP_MUL_F32(src_to_f32_0, vec2_sf_const);
    HVX_Vector div_f32_1 = HVX_OP_MUL_F32(src_to_f32_1, vec2_sf_const);

#if __HVX_ARCH__ < 79
    HVX_Vector res = hvx_vec_f32_to_f16(div_f32_0, div_f32_1);
#else
    HVX_Vector res = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#endif
    return res;
}

// Variant for <v79: Use pre-computed f16 reciprocal constant
static inline HVX_Vector hvx_div_mul_f16_const_using_f16(HVX_Vector vec1_hf, HVX_Vector const_inv_hf) {
    // Multiply by pre-computed f16 reciprocal constant
    return HVX_OP_MUL_F16(vec1_hf, const_inv_hf);
}

#define hvx_div_scaler_f16_loop_body(dst_type, src_type, vec_store)                                    \
    do {                                                                                               \
        dst_type * restrict vdst = (dst_type *) dst;                                                   \
        src_type * restrict vsrc = (src_type *) src;                                                   \
                                                                                                       \
        HVX_Vector hf_one = Q6_Vh_vsplat_R(0x3C00);                                                    \
                                                                                                       \
        const uint32_t nvec = n / VLEN_FP16;                                                           \
        const uint32_t nloe = n % VLEN_FP16;                                                           \
                                                                                                       \
        uint32_t i = 0;                                                                                \
                                                                                                       \
        _Pragma("unroll(4)")                                                                           \
        for (; i < nvec; i++) {                                                                        \
            HVX_Vector res;                                                                            \
            if (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } else {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vdst[i] = res;                                                                             \
        }                                                                                              \
        if (nloe) {                                                                                    \
            HVX_Vector res;                                                                            \
            if (__HVX_ARCH__ < 79) {                                                                   \
                res = hvx_div_mul_f16_const_using_f16(vsrc[i], val_vec_f16);                           \
            } else {                                                                                   \
                res = hvx_div_mul_f16_const_using_f32(vsrc[i], val_vec_f32, hf_one);                   \
            }                                                                                          \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP16, res);                                     \
        }                                                                                              \
    } while(0)

static inline void hvx_div_scalar_f16_aa(uint8_t * restrict dst, const uint8_t * restrict src, const _Float16 val, uint32_t n) {
    const HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((float)val));
    const HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    assert((uintptr_t) dst % 128 == 0);
    assert((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_Vector, HVX_Vector, hvx_vec_store_a);
}
static inline void hvx_div_scalar_f16_au(uint8_t * restrict dst, const uint8_t * restrict src, const _Float16 val, uint32_t n) {
    const HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((float)val));
    const HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    assert((uintptr_t) dst % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_Vector, HVX_UVector, hvx_vec_store_a);
}
static inline void hvx_div_scalar_f16_ua(uint8_t * restrict dst, const uint8_t * restrict src, const _Float16 val, uint32_t n) {
    const HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((float)val));
    const HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    assert((uintptr_t) src % 128 == 0);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_Vector, hvx_vec_store_u);
}
static inline void hvx_div_scalar_f16_uu(uint8_t * restrict dst, const uint8_t * restrict src, const _Float16 val, uint32_t n) {
    const HVX_Vector val_vec_f32 = hvx_vec_splat_f32(1.0f/((float)val));
    const HVX_Vector val_vec_f16 = hvx_vec_splat_f16(1.0f / val);
    hvx_div_scaler_f16_loop_body(HVX_UVector, HVX_UVector, hvx_vec_store_u);
}

// Compute div by using hvx_vec_inverse_f32_guard. Requires first by exapnding fp32 to fp16 and convert the result back to fp32.
static inline HVX_Vector hvx_vec_div_f16_using_f32(HVX_Vector vec1, HVX_Vector vec2, HVX_Vector f32_nan_inf_mask, HVX_Vector vec_hf_one_1_0) {
#if __HVX_ARCH__ < 79
    // Convert first input to fp32
    HVX_VectorPair vec1_to_f32   = Q6_Wqf32_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec1_to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec1_to_f32));
    HVX_Vector     vec1_to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec1_to_f32));

    // Convert second input to fp32
    HVX_VectorPair vec2_to_f32   = Q6_Wqf32_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec2_to_f32_0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(vec2_to_f32));
    HVX_Vector     vec2_to_f32_1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(vec2_to_f32));
#else
    // Convert first input to fp32
    HVX_VectorPair vec1_to_f32   = Q6_Wsf_vmpy_VhfVhf(vec1, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec1_to_f32_0 = Q6_V_lo_W(vec1_to_f32);
    HVX_Vector     vec1_to_f32_1 = Q6_V_hi_W(vec1_to_f32);

    // Convert second input to fp32
    HVX_VectorPair vec2_to_f32   = Q6_Wsf_vmpy_VhfVhf(vec2, vec_hf_one_1_0);  // *1.0
    HVX_Vector     vec2_to_f32_0 = Q6_V_lo_W(vec2_to_f32);
    HVX_Vector     vec2_to_f32_1 = Q6_V_hi_W(vec2_to_f32);
#endif

    // Inverse second input in fp32
    HVX_Vector     vec2_inv_f32_0 = hvx_vec_inverse_f32_guard(vec2_to_f32_0, f32_nan_inf_mask);
    HVX_Vector     vec2_inv_f32_1 = hvx_vec_inverse_f32_guard(vec2_to_f32_1, f32_nan_inf_mask);

    // Multiply first input by inverse of second, in fp32
    HVX_Vector     div_f32_0 = HVX_OP_MUL_F32(vec1_to_f32_0, vec2_inv_f32_0);
    HVX_Vector     div_f32_1 = HVX_OP_MUL_F32(vec1_to_f32_1, vec2_inv_f32_1);

    // Convert back to fp16
#if __HVX_ARCH__ < 79
    HVX_Vector     recip = hvx_vec_f32_to_f16(div_f32_0, div_f32_1);
#else
    HVX_Vector     recip = Q6_Vhf_vcvt_VsfVsf(div_f32_0, div_f32_1);
#endif

    return recip;
}

// Hybrid approach: f16 reciprocal for <v79, f32 precision for >=v79
static inline HVX_Vector hvx_vec_hybrid_div_f16(HVX_Vector vec1, HVX_Vector vec2, HVX_Vector f32_nan_inf_mask, HVX_Vector f16_nan_inf_mask, HVX_Vector vec_hf_one_1_0) {
#if __HVX_ARCH__ < 79
    // For older architectures, use f16 reciprocal to avoid NaN/-inf issues
    HVX_Vector vec2_inv = hvx_vec_inverse_f16_guard(vec2, f16_nan_inf_mask);
    return HVX_OP_MUL_F16(vec1, vec2_inv);
#else
    return hvx_vec_div_f16_using_f32(vec1, vec2, f32_nan_inf_mask, vec_hf_one_1_0);
#endif
}

#define hvx_div_f16_loop_body(dst_type, src0_type, src1_type, vec_store)                  \
    do {                                                                                  \
        dst_type * restrict vdst = (dst_type *) dst;                                      \
        src0_type * restrict vsrc0 = (src0_type *) src0;                                  \
        src1_type * restrict vsrc1 = (src1_type *) src1;                                  \
                                                                                          \
        const HVX_Vector f32_nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                    \
        const HVX_Vector f16_nan_inf_mask = Q6_Vh_vsplat_R(0x7c00);                       \
        const HVX_Vector hf_one = Q6_Vh_vsplat_R(0x3C00);                                 \
                                                                                          \
        const uint32_t nvec = n / VLEN_FP16;                                              \
        const uint32_t nloe = n % VLEN_FP16;                                              \
                                                                                          \
        uint32_t i = 0;                                                                   \
                                                                                          \
        _Pragma("unroll(4)")                                                              \
        for (; i < nvec; i++) {                                                           \
            HVX_Vector res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_mask, f16_nan_inf_mask,   \
                                                    hf_one);                              \
            vdst[i] = res;                                                                \
        }                                                                                 \
        if (nloe) {                                                                       \
            HVX_Vector res = hvx_vec_hybrid_div_f16(vsrc0[i], vsrc1[i],                   \
                                                    f32_nan_inf_mask, f16_nan_inf_mask,   \
                                                    hf_one);                              \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP16, res);                        \
        }                                                                                 \
    } while(0)

#define hvx_div_f32_loop_body(dst_type, src0_type, src1_type, vec_store)             \
    do {                                                                             \
        dst_type * restrict vdst = (dst_type *) dst;                                 \
        src0_type * restrict vsrc0 = (src0_type *) src0;                             \
        src1_type * restrict vsrc1 = (src1_type *) src1;                             \
                                                                                     \
        const HVX_Vector nan_inf_mask = Q6_V_vsplat_R(0x7f800000);                   \
                                                                                     \
        const uint32_t nvec = n / VLEN_FP32;                                         \
        const uint32_t nloe = n % VLEN_FP32;                                         \
                                                                                     \
        uint32_t i = 0;                                                              \
                                                                                     \
        _Pragma("unroll(4)")                                                         \
        for (; i < nvec; i++) {                                                      \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vdst[i] = res;                                                           \
        }                                                                            \
        if (nloe) {                                                                  \
            HVX_Vector inv_src1 = hvx_vec_inverse_f32_guard(vsrc1[i], nan_inf_mask); \
            HVX_Vector res = HVX_OP_MUL_F32(vsrc0[i], inv_src1);                     \
            vec_store((void *) &vdst[i], nloe * SIZEOF_FP32, res);                   \
        }                                                                            \
    } while(0)

// Generic macro to define alignment permutations for an op
#define DEFINE_HVX_DIV_OP_VARIANTS(OP_NAME, OP_LOOP_BODY) \
static inline void OP_NAME##_aaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src0 % 128 == 0); \
    assert((uintptr_t) src1 % 128 == 0); \
    OP_LOOP_BODY(HVX_Vector, HVX_Vector, HVX_Vector, hvx_vec_store_a); \
} \
static inline void OP_NAME##_aau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src0 % 128 == 0); \
    OP_LOOP_BODY(HVX_Vector, HVX_Vector, HVX_UVector, hvx_vec_store_a); \
} \
static inline void OP_NAME##_aua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    assert((uintptr_t) src1 % 128 == 0); \
    OP_LOOP_BODY(HVX_Vector, HVX_UVector, HVX_Vector, hvx_vec_store_a); \
} \
static inline void OP_NAME##_auu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) dst % 128 == 0); \
    OP_LOOP_BODY(HVX_Vector, HVX_UVector, HVX_UVector, hvx_vec_store_a); \
} \
static inline void OP_NAME##_uaa(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src0 % 128 == 0); \
    assert((uintptr_t) src1 % 128 == 0); \
    OP_LOOP_BODY(HVX_UVector, HVX_Vector, HVX_Vector, hvx_vec_store_u); \
} \
static inline void OP_NAME##_uau(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src0 % 128 == 0); \
    OP_LOOP_BODY(HVX_UVector, HVX_Vector, HVX_UVector, hvx_vec_store_u); \
} \
static inline void OP_NAME##_uua(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    assert((uintptr_t) src1 % 128 == 0); \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_Vector, hvx_vec_store_u); \
} \
static inline void OP_NAME##_uuu(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, uint32_t n) { \
    OP_LOOP_BODY(HVX_UVector, HVX_UVector, HVX_UVector, hvx_vec_store_u); \
} \

// Dispatcher logic
#define HVX_DIV_DISPATCHER(OP_NAME) \
static inline void OP_NAME(uint8_t * restrict dst, const uint8_t * restrict src0, const uint8_t * restrict src1, const uint32_t num_elems) { \
    if (hex_is_aligned((void *) dst, 128)) { \
        if (hex_is_aligned((void *) src0, 128)) { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_aaa(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_aau(dst, src0, src1, num_elems); \
        } else { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_aua(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_auu(dst, src0, src1, num_elems); \
        } \
    } else { \
        if (hex_is_aligned((void *) src0, 128)) { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_uaa(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_uau(dst, src0, src1, num_elems); \
        } else { \
            if (hex_is_aligned((void *) src1, 128)) OP_NAME##_uua(dst, src0, src1, num_elems); \
            else                                    OP_NAME##_uuu(dst, src0, src1, num_elems); \
        } \
    } \
}

DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f32, hvx_div_f32_loop_body)
DEFINE_HVX_DIV_OP_VARIANTS(hvx_div_f16, hvx_div_f16_loop_body)

HVX_DIV_DISPATCHER(hvx_div_f32)
HVX_DIV_DISPATCHER(hvx_div_f16)

#undef HVX_OP_MUL_F32
#undef HVX_OP_MUL_F16


#include "hvx-base-defs.inc"

#endif /* HVX_UTILS_H */
