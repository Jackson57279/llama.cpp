#pragma once

// GGML internal header

#include "ggml.h.inc"
#include "gguf.h.inc"

#include <assert.h>
#include <math.h>
#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#if defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void ggml_print_backtrace(void);

uint64_t ggml_graph_next_uid(void);
bool ggml_op_is_empty_rust(int op);
bool ggml_is_view_op_rust(int op);
size_t ggml_bitset_size_rust(size_t n);
bool ggml_are_same_layout_rust(
        int a_type,
        int b_type,
        const int64_t * a_ne,
        const int64_t * b_ne,
        const size_t * a_nb,
        const size_t * b_nb,
        size_t n_dims);
int32_t ggml_get_op_params_i32_rust(const int32_t * op_params, uint32_t i);
float ggml_get_op_params_f32_rust(const int32_t * op_params, uint32_t i);
void ggml_set_op_params_i32_rust(int32_t * op_params, uint32_t i, int32_t value);
void ggml_set_op_params_f32_rust(int32_t * op_params, uint32_t i, float value);
size_t ggml_aligned_offset_rust(const void * buffer, size_t offset, size_t alignment);
int ggml_get_node_buffer_id_rust(const int * node_buffer_ids, int index);
bool ggml_buffer_address_less_rust(int a_chunk, size_t a_offset, int b_chunk, size_t b_offset);
bool ggml_isinf_fp16_rust(uint16_t value);
bool ggml_isnan_fp16_rust(uint16_t value);
bool ggml_is_invalid_e8m0_rust(uint8_t value);
int ggml_float_validation_kind_rust(float value);
int ggml_int_pair_compare_rust(const void * left, const void * right);
int ggml_float_pair_compare_rust(const void * left, const void * right);
size_t ggml_graph_nbytes_rust(
        size_t size,
        bool grads,
        size_t hash_size,
        size_t cgraph_size,
        size_t tensor_ptr_size,
        size_t int32_size,
        size_t bitset_size,
        size_t bitset_t_size);
int ggml_iq2_find_best_neighbour_rust(
        const uint16_t * neighbours,
        const uint64_t * grid,
        const float * xval,
        const float * weight,
        float scale,
        int8_t * out_l);
int ggml_iq3_find_best_neighbour_rust(
        const uint16_t * neighbours,
        const uint32_t * grid,
        const float * xval,
        const float * weight,
        float scale,
        int8_t * out_l);
int ggml_iq1_find_best_neighbour2_rust(
        const uint16_t * neighbours,
        const uint64_t * grid,
        const float * xval,
        const float * weight,
        float scale,
        const float * xg,
        int8_t * out_l,
        int ngrid);
int64_t ggml_calc_conv_output_size_rust(int64_t ins, int64_t ks, int s, int p, int d);
int64_t ggml_calc_conv_transpose_1d_output_size_rust(int64_t ins, int64_t ks, int s, int p, int d);
int64_t ggml_calc_conv_transpose_output_size_rust(int64_t ins, int64_t ks, int s, int p);
int64_t ggml_calc_pool_output_size_rust(int64_t ins, int ks, int s, float p);
float ggml_rope_yarn_corr_dim_rust(int n_dims, int n_ctx_orig, float n_rot, float base);
int ggml_nearest_int_rust(float fval);
int ggml_iq2_data_index_rust(int type);
int ggml_iq2_grid_size_rust(int type);
int ggml_iq3_data_index_rust(int grid_size);
int ggml_best_index_int8_rust(int n, const int8_t * val, float x);
int ggml_best_index_mxfp4_rust(float x, float e);
int ggml_up32_rust(int n);
int ggml_up_rust(int n, int m);
float ggml_compute_softplus_f32_rust(float input);
bool ggml_bitset_get_rust(const uint32_t * bitset, size_t i);
void ggml_bitset_set_rust(uint32_t * bitset, size_t i);
void ggml_bitset_clear_rust(uint32_t * bitset, size_t i);
size_t ggml_hash_rust(const void * ptr);
bool ggml_impl_is_view_rust(const void * view_src);
bool ggml_hash_contains_rust(const void * hash_set, void * key);
size_t ggml_hash_find_rust(const void * hash_set, const void * key);
size_t ggml_hash_insert_rust(void * hash_set, void * key);
size_t ggml_hash_find_or_insert_rust(void * hash_set, void * key);
int32_t ggml_node_get_use_count_rust(const void * node, const void * hash_set, const int32_t * use_counts);
bool ggml_node_has_n_uses_rust(int32_t use_count, int32_t n_uses, const void * view_src, int flags, int output_flag);
float ggml_e8m0_to_fp32_rust(uint8_t x);
float ggml_e8m0_to_fp32_half_rust(uint8_t x);
float ggml_ue4m3_to_fp32_rust(uint8_t x);
uint8_t ggml_fp32_to_ue4m3_rust(float x);

#ifndef MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// required for mmap as gguf only guarantees 32-byte alignment
#define TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef __cplusplus
    #ifndef static_assert
        #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, msg) _Static_assert(cond, msg)
        #else
            #define static_assert(cond, msg) struct global_scope_noop_trick
        #endif
    #endif
#endif

#define ggml_up32(n) ggml_up32_rust(n)

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

#define ggml_up(n, m) ggml_up_rust(n, m)

// TODO: move to ggml.h.inc? (won't be able to inline)
static bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    return ggml_are_same_layout_rust((int) a->type, (int) b->type, a->ne, b->ne, a->nb, b->nb, GGML_MAX_DIMS);
}

#define ggml_op_is_empty(op) ggml_op_is_empty_rust((int) (op))

#define ggml_impl_is_view(t) ggml_impl_is_view_rust((t)->view_src)

#define ggml_compute_softplus_f32(input) ggml_compute_softplus_f32_rust(input)
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API void ggml_log_internal        (enum ggml_log_level level, const char * format, ...);
GGML_API void ggml_log_callback_default(enum ggml_log_level level, const char * text, void * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

static void ggml_set_op_params(struct ggml_tensor * tensor, const void * params, size_t params_size) {
    GGML_ASSERT(tensor != NULL); // silence -Warray-bounds warnings
    assert(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

static int32_t ggml_get_op_params_i32(const struct ggml_tensor * tensor, uint32_t i) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
    return ggml_get_op_params_i32_rust(tensor->op_params, i);
}

static float ggml_get_op_params_f32(const struct ggml_tensor * tensor, uint32_t i) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(float));
    return ggml_get_op_params_f32_rust(tensor->op_params, i);
}

static void ggml_set_op_params_i32(struct ggml_tensor * tensor, uint32_t i, int32_t value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
    ggml_set_op_params_i32_rust(tensor->op_params, i, value);
}

static void ggml_set_op_params_f32(struct ggml_tensor * tensor, uint32_t i, float value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(float));
    ggml_set_op_params_f32_rust(tensor->op_params, i, value);
}

struct ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    int                n_tasks;
    void             * userdata;
};

struct ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    int                 n_tasks;
    void              * userdata;
};

struct ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    int               n_tasks;
    void            * userdata;
};

struct ggml_custom_op_params {
    ggml_custom_op_t fun;
    int              n_tasks;
    void           * userdata;
};

// bitset

typedef uint32_t ggml_bitset_t;

static_assert(sizeof(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (sizeof(ggml_bitset_t)*8 - 1)

#define ggml_bitset_size(n) ggml_bitset_size_rust(n)

#define ggml_bitset_get(bitset, i) ggml_bitset_get_rust(bitset, i)

#define ggml_bitset_set(bitset, i) ggml_bitset_set_rust(bitset, i)

#define ggml_bitset_clear(bitset, i) ggml_bitset_clear_rust(bitset, i)

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

struct ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    struct ggml_tensor ** keys; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

struct ggml_hash_set ggml_hash_set_new(size_t size);
void                 ggml_hash_set_free(struct ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
void ggml_hash_set_reset(struct ggml_hash_set * hash_set);

// hash function for ggml_tensor
#define ggml_hash(p) ggml_hash_rust(p)
#define ggml_hash_contains(hash_set, key) ggml_hash_contains_rust(hash_set, key)
#define ggml_hash_find(hash_set, key) ggml_hash_find_rust(hash_set, key)
#define ggml_hash_insert(hash_set, key) ggml_hash_insert_rust(hash_set, key)
#define ggml_hash_find_or_insert(hash_set, key) ggml_hash_find_or_insert_rust(hash_set, key)

// computation graph

enum ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
    GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
    GGML_CGRAPH_EVAL_ORDER_COUNT
};

struct ggml_cgraph {
    int size;    // maximum number of nodes/leafs/grads/grad_accs
    int n_nodes; // number of nodes currently in use
    int n_leafs; // number of leafs currently in use

    struct ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    struct ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    struct ggml_tensor ** grad_accs; // accumulators for node gradients
    struct ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    struct ggml_hash_set visited_hash_set;

    enum ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph, int i0, int i1);

// allocator implementation: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(enum ggml_op op);


// Memory allocation

GGML_API void * ggml_aligned_malloc(size_t size);
GGML_API void ggml_aligned_free(void * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#define ggml_e8m0_to_fp32(x) ggml_e8m0_to_fp32_rust(x)

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
#define ggml_e8m0_to_fp32_half(x) ggml_e8m0_to_fp32_half_rust(x)

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
#define ggml_ue4m3_to_fp32(x) ggml_ue4m3_to_fp32_rust(x)

#define ggml_fp32_to_ue4m3(x) ggml_fp32_to_ue4m3_rust(x)

/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */
static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float s) {
    ggml_bf16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        return h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

static inline int32_t ggml_node_get_use_count(const struct ggml_cgraph * cgraph, int node_idx) {
    const struct ggml_tensor * node = cgraph->nodes[node_idx];
    return ggml_node_get_use_count_rust(node, &cgraph->visited_hash_set, cgraph->use_counts);
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
static inline bool ggml_node_has_n_uses(const struct ggml_cgraph * cgraph, int node_idx, int32_t n_uses) {
    const struct ggml_tensor * node = cgraph->nodes[node_idx];
    return ggml_node_has_n_uses_rust(
            ggml_node_get_use_count(cgraph, node_idx),
            n_uses,
            node->view_src,
            node->flags,
            GGML_TENSOR_FLAG_OUTPUT);
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
static inline bool ggml_can_fuse_ext(const struct ggml_cgraph * cgraph, const int * node_idxs, const enum ggml_op * ops, int num_ops) {
    for (int i = 0; i < num_ops; ++i) {
        if (node_idxs[i] >= cgraph->n_nodes) {
            return false;
        }

        struct ggml_tensor * node = cgraph->nodes[node_idxs[i]];
        if (node->op != ops[i]) {
            return false;
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
        if (i < num_ops - 1 && !ggml_node_has_n_uses(cgraph, node_idxs[i], 1)) {
            return false;
        }
        if (i > 0) {
            struct ggml_tensor * prev = cgraph->nodes[node_idxs[i - 1]];
            if (node->src[0] != prev && node->src[1] != prev) {
                return false;
            }
            if (!ggml_are_same_shape(node, prev)) {
                return false;
            }
        }
    }
    return true;
}

// same as above, for sequential indices starting at node_idx
static inline bool ggml_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, const enum ggml_op * ops, int num_ops) {
    assert(num_ops < 32);

    if (node_idx + num_ops > cgraph->n_nodes) {
        return false;
    }

    int idxs[32];
    for (int i = 0; i < num_ops; ++i) {
        idxs[i] = node_idx + i;
    }

    return ggml_can_fuse_ext(cgraph, idxs, ops, num_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(const struct ggml_cgraph * cgraph,
                                         const int *                node_idxs,
                                         int                        count,
                                         const enum ggml_op *       ops,
                                         const int *                outputs,
                                         int                        num_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
static inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph * cgraph,
                                          int                        node_idx,
                                          int                        count,
                                          const enum ggml_op *       ops,
                                          const int *                outputs,
                                          int                        num_outputs) {
    GGML_ASSERT(count < 32);
    if (node_idx + count > cgraph->n_nodes) {
        return false;
    }

    int idxs[32];

    for (int i = 0; i < count; ++i) {
        idxs[i] = node_idx + i;
    }

    return ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, outputs, num_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <array>
#include <initializer_list>
#include <vector>

// nicer C++ syntax for ggml_can_fuse
inline bool ggml_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    return ggml_can_fuse(cgraph, node_idx, ops.begin(), (int)ops.size());
}

inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph *          cgraph,
                                   int                                 start_idx,
                                   std::initializer_list<enum ggml_op> ops,
                                   std::initializer_list<int>          outputs = {}) {
    return ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
inline bool ggml_check_edges(const struct ggml_cgraph *                cgraph,
                             int                                       start_idx,
                             std::initializer_list<std::array<int, 3>> edges) {
    for (const auto & edge : edges) {
        int dst_node = edge[0];
        int src_idx  = edge[1];
        int src_node = edge[2];
        if (cgraph->nodes[start_idx + dst_node]->src[src_idx] != cgraph->nodes[start_idx + src_node]) {
            return false;
        }
    }
    return true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(enum gguf_type type);
GGML_API void gguf_write_to_buf(const struct gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
