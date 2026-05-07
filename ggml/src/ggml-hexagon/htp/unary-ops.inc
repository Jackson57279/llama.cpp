#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#include "hex-dma-defs.inc"
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

#include "hvx-utils-defs.inc"

#define GGML_COMMON_DECL_C
#include "ggml-common-defs.inc"
#include "htp-ctx-defs.inc"
#include "htp-ops-defs.inc"
#include "htp-ops-defs.inc"

struct htp_unary_context {
    struct htp_ops_context * octx;

    // Precomputed values
    const uint8_t *           data_src0;
    uint8_t *                 data_dst;

    size_t                    src0_data_row_size;   // actual data bytes per row
    size_t                    dst_data_row_size;    // actual data bytes per row

    size_t                    src0_row_size_aligned;
    size_t                    dst_row_size_aligned;

    size_t                    src0_spad_half_size;
    size_t                    dst_spad_half_size;

    uint32_t                  block;
    uint32_t                  src0_nrows;
    uint32_t                  src0_nrows_per_thread;
    uint32_t                  nc;
};

// Convert flat row index to DDR byte offset using the tensor's actual strides.
// ir = i1 + ne1*(i2 + ne2*i3)  =>  offset = i1*nb1 + i2*nb2 + i3*nb3
static inline size_t unary_row_offset(uint32_t ir,
                                      uint32_t ne1, uint32_t ne2,
                                      size_t nb1, size_t nb2, size_t nb3) {
    const uint32_t i1 = ir % ne1;
    const uint32_t i2 = (ir / ne1) % ne2;
    const uint32_t i3 = ir / (ne1 * ne2);
    return i1 * nb1 + i2 * nb2 + i3 * nb3;
}
// Safe DMA block size from row `ir`: clamp to the tighter dim-1 slice
// boundary of src and dst so the nb1 stride stays valid for all rows.
static inline uint32_t unary_block_size(uint32_t ir,
                                        uint32_t end_row,
                                        uint32_t block,
                                        bool src_contig,
                                        bool dst_contig,
                                        uint32_t src_ne1,
                                        uint32_t dst_ne1) {
    uint32_t limit = MIN(block, end_row - ir);

    if (!src_contig) {
        const uint32_t src_slice_end = (ir / src_ne1 + 1) * src_ne1;
        limit = MIN(limit, src_slice_end - ir);
    }

    if (!dst_contig) {
        const uint32_t dst_slice_end = (ir / dst_ne1 + 1) * dst_ne1;
        limit = MIN(limit, dst_slice_end - ir);
    }

    return limit;
}

#define htp_unary_preamble            \
    const uint32_t ne00 = src->ne[0]; \
    const uint32_t ne01 = src->ne[1]; \
    const uint32_t ne02 = src->ne[2]; \
    const uint32_t ne03 = src->ne[3]; \
                                      \
    const uint32_t ne0 = dst->ne[0];  \
    const uint32_t ne1 = dst->ne[1];  \
    const uint32_t ne2 = dst->ne[2];  \
    const uint32_t ne3 = dst->ne[3];  \
                                      \
    const uint32_t nb00 = src->nb[0]; \
    const uint32_t nb01 = src->nb[1]; \
    const uint32_t nb02 = src->nb[2]; \
    const uint32_t nb03 = src->nb[3]; \
                                      \
    const uint32_t nb0 = dst->nb[0];  \
    const uint32_t nb1 = dst->nb[1];  \
    const uint32_t nb2 = dst->nb[2];  \
    const uint32_t nb3 = dst->nb[3];

static void hvx_fast_rms_norm_f32(const uint8_t * restrict src,
                                  uint8_t * restrict dst,
                                  uint8_t * restrict pad,
                                  const int num_elems,
                                  float     epsilon) {
    (void)pad;

    const HVX_Vector * restrict v_src = (HVX_Vector *) src;
    HVX_Vector * restrict v_dst       = (HVX_Vector *) dst;

    const int nvec = num_elems / VLEN_FP32;    // number of full vectors
    const int nloe = num_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_Vector sum_v = Q6_V_vsplat_R(0x00000000);
    HVX_Vector epsilon_v = hvx_vec_splat_f32(epsilon);

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {
        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        sum_v = Q6_Vqf32_vadd_Vqf32Vqf32(sum_v, v2);
    }

    // Reduce HVX sum
    sum_v = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(sum_v));

    HVX_Vector t_v            = hvx_vec_splat_f32((float) num_elems);
    HVX_Vector denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_Vector mean_v         = Q6_Vqf32_vmpy_VsfVsf(sum_v, denom_v);
    HVX_Vector mean_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(mean_v, epsilon_v);

    // Scale full vectors
    HVX_Vector scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(mean_epsilon_v));

    #pragma unroll(4)
    for (int i = 0; i < nvec; i++) {
        HVX_Vector v1 = v_src[i];
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    if (nloe > 0) {

        HVX_VectorPred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_Vector v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_Vector v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_Vector result = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, result);
    }
}

static void scale_f32(const float * restrict src,
                      float * restrict dst,
                      uint8_t * restrict spad,
                      const uint32_t num_rows,
                      const uint32_t row_elems,
                      const size_t   row_size,
                      int32_t *      op_params) {
    float scale = 0.f;
    float bias  = 0.f;
    memcpy(&scale, &op_params[0], sizeof(float));
    memcpy(&bias,  &op_params[1], sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_scale_offset_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, row_elems, scale, bias);
    }
}

static void rms_norm_f32(const float * restrict src,
                         float * restrict dst,
                         uint8_t * restrict spad,
                         const uint32_t num_rows,
                         const uint32_t row_elems,
                         const size_t   row_size,
                         int32_t *      op_params) {
    float epsilon = 0.f;
    memcpy(&epsilon, op_params, sizeof(float));

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_fast_rms_norm_f32((const uint8_t *) src_local, (uint8_t *) dst_local, spad, row_elems, epsilon);
    }
}

static void sqr_f32(const float * restrict src,
                    float * restrict dst,
                    uint8_t * restrict spad,
                    const uint32_t num_rows,
                    const uint32_t row_elems,
                    const size_t   row_size,
                    int32_t *      op_params) {

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_sqr_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, row_elems);
    }
}

static void sqrt_f32(const float * restrict src,
                     float * restrict dst,
                     uint8_t * restrict spad,
                     const uint32_t num_rows,
                     const uint32_t row_elems,
                     const size_t   row_size,
                     int32_t *      op_params) {

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_sqrt_f32_aa((uint8_t *) dst_local, (const uint8_t *) src_local, row_elems);
    }
}

static void neg_f32(const float * restrict src,
                    float * restrict dst,
                    uint8_t * restrict spad,
                    const uint32_t num_rows,
                    const uint32_t row_elems,
                    const size_t   row_size,
                    int32_t *      op_params) {

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_scale_f32_aa(dst_local, src_local, row_elems, -1.0f);
    }
}

static void exp_f32(const float * restrict src,
                    float * restrict dst,
                    uint8_t * restrict spad,
                    const uint32_t num_rows,
                    const uint32_t row_elems,
                    const size_t   row_size,
                    int32_t *      op_params) {

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_exp_f32(dst_local, src_local, row_elems, false);
    }
}

static void sigmoid_f32(const float * restrict src,
                        float * restrict dst,
                        uint8_t * restrict spad,
                        const uint32_t num_rows,
                        const uint32_t row_elems,
                        const size_t   row_size,
                        int32_t *      op_params) {

    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const uint8_t * restrict src_local = (const uint8_t *)src + (ir * row_size);
        uint8_t * restrict dst_local       = (uint8_t *)dst + (ir * row_size);

        hvx_sigmoid_f32_aa(dst_local, src_local, row_elems);
    }
}

static void softplus_f32(const float * restrict src,
                         float * restrict dst,
                         uint8_t * restrict spad,
                         const uint32_t num_rows,
                         const uint32_t row_elems,
                         const size_t   row_size,
                         int32_t *      op_params) {
    // softplus(x) = log(1 + exp(x))
    // Match CPU reference: ggml_compute_softplus_f32() in ggml-impl-defs.inc
    for (uint32_t ir = 0; ir < num_rows; ir++) {
        const float * restrict src_f = (const float *)((const uint8_t *)src + (ir * row_size));
        float * restrict dst_f       = (float *)((uint8_t *)dst + (ir * row_size));

        for (uint32_t i = 0; i < row_elems; i++) {
            float x = src_f[i];
            // For x > 20: softplus(x) ≈ x (avoids exp overflow)
            dst_f[i] = (x > 20.0f) ? x : logf(1.0f + expf(x));
        }
    }
}

static void unary_job_f32_per_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_unary_context * uctx = (const struct htp_unary_context *) data;
    struct htp_ops_context * octx = uctx->octx;
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    htp_unary_preamble;

    int                       htp_op = octx->op;
    int32_t *                 op_params = octx->op_params;
    uint32_t                  src0_nrows_per_thread = uctx->src0_nrows_per_thread;

    const size_t src0_data_row_size = uctx->src0_data_row_size;
    const size_t dst_data_row_size  = uctx->dst_data_row_size;

    const size_t src0_row_size_aligned = uctx->src0_row_size_aligned;
    const size_t dst_row_size_aligned  = uctx->dst_row_size_aligned;

    const uint32_t src0_nrows = uctx->src0_nrows;
    const uint32_t src0_start_row = src0_nrows_per_thread * ith;
    const uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, src0_nrows);

    // no work for this thread
    if (src0_start_row >= src0_end_row) {
        return;
    }

    uint64_t t1, t2;
    t1 = HAP_perf_get_qtimer_count();

    const uint8_t * restrict data_src = uctx->data_src0;
    uint8_t * restrict       data_dst = uctx->data_dst;

    uint8_t * src0_spad_data = octx->src0_spad.data + (ith * octx->src0_spad.size_per_thread);
    uint8_t * dst_spad_data  = octx->dst_spad.data  + (ith * octx->dst_spad.size_per_thread);

    size_t src0_spad_half_size = uctx->src0_spad_half_size;
    size_t dst_spad_half_size  = uctx->dst_spad_half_size;

    // Non-contiguous tensors have gaps at dim-2/3 boundaries that a single-stride
    // 2D DMA descriptor cannot span. Clamp BLOCK to ne1 (one dim-1 slice) so every
    // transfer stays within a nb1-uniform region. Skipped for contiguous tensors.
    const bool src0_contig = (nb02 == (size_t)ne01 * nb01) &&
                             (nb03 == (size_t)ne02 * nb02);
    const bool dst_contig  = (nb2  == (size_t)ne1  * nb1)  &&
                             (nb3  == (size_t)ne2  * nb2);
    const uint32_t src0_max_block = src0_contig ? uctx->block : MIN((uint32_t)uctx->block, ne01);
    const uint32_t dst_max_block  = dst_contig  ? uctx->block : MIN((uint32_t)uctx->block, ne1);
    const uint32_t BLOCK = MIN(src0_max_block, dst_max_block);
    if (BLOCK == 0) {
        FARF(ERROR, "unary-f32 : current VTCM reservation %zu is too small for even 1 row per thread, needed at least %zu\n",
             octx->src0_spad.size_per_thread, src0_row_size_aligned);
        return;
    }

    dma_queue * dma_queue = octx->ctx->dma[ith];

    for (uint32_t ir = src0_start_row, spad_idx = 0; ir < src0_end_row && spad_idx < 2; spad_idx++) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01, ne1);

        // Dummy DMA transation for sequencing (interleaving dst,src,dst,...)
        dma_queue_push(dma_queue,
            dma_make_ptr(data_dst, dst_spad_data + (spad_idx * dst_spad_half_size)),
            nb1, dst_row_size_aligned, dst_data_row_size, 0);

        const size_t src0_off = unary_row_offset(ir, ne01, ne02, nb01, nb02, nb03);
        dma_queue_push(dma_queue,
            dma_make_ptr(src0_spad_data + (spad_idx * src0_spad_half_size), data_src + src0_off),
            src0_row_size_aligned, nb01, src0_data_row_size, block_size);
        ir += block_size;
    }

    for (uint32_t ir = src0_start_row; ir < src0_end_row; ) {
        const uint32_t block_size = unary_block_size(ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01, ne1);

        float * dst_spad  = (float *) dma_queue_pop(dma_queue).src;
        float * src0_spad = (float *) dma_queue_pop(dma_queue).dst;

        // Process block in VTCM
        switch (htp_op) {
            case HTP_OP_RMS_NORM:
                rms_norm_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_SCALE:
                scale_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_SQR:
                sqr_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_SQRT:
                sqrt_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_UNARY_NEG:
                neg_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_UNARY_EXP:
                exp_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_UNARY_SIGMOID:
                sigmoid_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            case HTP_OP_UNARY_SOFTPLUS:
                softplus_f32(src0_spad, dst_spad, NULL, block_size, ne0, src0_row_size_aligned, op_params);
                break;
            default:
                break;
        }

        const size_t dst_off = unary_row_offset(ir, ne1, ne2, nb1, nb2, nb3);
        dma_queue_push(dma_queue,
            dma_make_ptr(data_dst + dst_off, dst_spad),
            nb1, dst_row_size_aligned, dst_data_row_size, block_size);

        // prefetch N+2 loop iteration if any
        const uint32_t next_ir = ir + block_size;
        if (next_ir < src0_end_row) {
            const uint32_t next_block_size = unary_block_size(next_ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01, ne1);
            const uint32_t pref_ir = next_ir + next_block_size;
            if (pref_ir < src0_end_row) {
                const uint32_t pref_block_size = unary_block_size(pref_ir, src0_end_row, BLOCK, src0_contig, dst_contig, ne01, ne1);
                const size_t src0_pref_off = unary_row_offset(pref_ir, ne01, ne02, nb01, nb02, nb03);
            dma_queue_push(dma_queue,
                dma_make_ptr(src0_spad, data_src + src0_pref_off),
                src0_row_size_aligned, nb01, src0_data_row_size, pref_block_size);
            }
        }
        ir += block_size;
    }

    dma_queue_flush(dma_queue);

    t2 = HAP_perf_get_qtimer_count();

    FARF(HIGH, "unary-f32 %d/%d: %ux%ux%ux%u (%u:%u) -> %ux%ux%ux%u usec %u\n", ith, nth, src->ne[0],
         src->ne[1], src->ne[2], src->ne[3], src0_start_row, src0_end_row, dst->ne[0], dst->ne[1], dst->ne[2],
         dst->ne[3], (unsigned) HAP_perf_qtimer_count_to_us(t2 - t1));
}

static int execute_op_unary_f32(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    const char * op_type = NULL;

    switch (octx->op) {
        case HTP_OP_RMS_NORM:
            op_type = "rmsnorm-f32";
            break;
        case HTP_OP_SCALE:
            op_type = "scale-f32";
            break;
        case HTP_OP_SQR:
            op_type = "sqr-f32";
            break;
        case HTP_OP_SQRT:
            op_type = "sqrt-f32";
            break;
        case HTP_OP_UNARY_NEG:
            op_type = "neg-f32";
            break;
        case HTP_OP_UNARY_EXP:
            op_type = "exp-f32";
            break;
        case HTP_OP_UNARY_SIGMOID:
            op_type = "sigmoid-f32";
            break;
        case HTP_OP_UNARY_SOFTPLUS:
            op_type = "softplus-f32";
            break;

        default:
            FARF(ERROR, "Unsupported unary Op %u\n", octx->op);
            return HTP_STATUS_NO_SUPPORT;
    }

    const uint32_t src0_nrows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    const uint32_t n_threads  = MIN(octx->n_threads, src0_nrows);

    const size_t src0_data_row_size = src0->ne[0] * sizeof(float);
    const size_t dst_data_row_size  = dst->ne[0]  * sizeof(float);

    const size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, VLEN);
    const size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  VLEN);

    // VTCM scratchpads for all tensors
    // N rows per thread, padded to HVX vector size
    // Double buffering requires 2x size per buffer

    size_t spad_size_per_row   = 2 * (src0_row_size_aligned + dst_row_size_aligned);
    size_t vtcm_row_per_thread = (octx->ctx->vtcm_size)/ (n_threads * spad_size_per_row);

    // Make sure the reserved vtcm size is sufficient
    if (vtcm_row_per_thread == 0) {
        FARF(ERROR, "unary-%s : current VTCM reservation %zu is too small, needed %zu\n", op_type, octx->ctx->vtcm_size,
             spad_size_per_row * n_threads);
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    octx->src0_spad.size_per_thread = src0_row_size_aligned * vtcm_row_per_thread * 2;
    octx->dst_spad.size_per_thread  = dst_row_size_aligned * vtcm_row_per_thread * 2;

    octx->src0_spad.size = n_threads * octx->src0_spad.size_per_thread;
    octx->dst_spad.size  = n_threads * octx->dst_spad.size_per_thread;

    octx->src0_spad.data = octx->ctx->vtcm_base;
    octx->dst_spad.data  = octx->src0_spad.data + octx->src0_spad.size;

    FARF(HIGH, "%s: (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-spad-size %u src1-spad-size %u dst-spad-size %u\n", op_type,
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
         octx->src0_spad.size, octx->src1_spad.size, octx->dst_spad.size);

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        struct htp_unary_context uctx = {
            .octx                  = octx,
            .src0_nrows_per_thread = (src0_nrows + n_threads - 1) / n_threads,
            .src0_nrows            = src0_nrows,

            .data_src0             = (const uint8_t *)src0->data,
            .data_dst              = (uint8_t *)dst->data,

            .src0_data_row_size    = src0_data_row_size,
            .dst_data_row_size     = dst_data_row_size,

            .src0_row_size_aligned = src0_row_size_aligned,
            .dst_row_size_aligned  = dst_row_size_aligned,

            .src0_spad_half_size   = octx->src0_spad.size_per_thread / 2,
            .dst_spad_half_size    = octx->dst_spad.size_per_thread / 2,

            .block                 = (octx->src0_spad.size_per_thread / 2) / src0_row_size_aligned,
            .nc                    = src0->ne[0],
        };

        worker_pool_run_func(octx->ctx->worker_pool, unary_job_f32_per_thread, &uctx, n_threads);
    }

    return err;
}

int op_unary(struct htp_ops_context * octx) {
    int err = HTP_STATUS_OK;

    switch (octx->src[0]->type) {
        case HTP_TYPE_F32:
            err = execute_op_unary_f32(octx);
            break;

        default:
            err = HTP_STATUS_NO_SUPPORT;
            break;
    }

    return err;
}
