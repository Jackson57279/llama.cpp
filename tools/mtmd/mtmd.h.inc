#ifndef MTMD_H
#define MTMD_H

#include "ggml.h.inc"
#include "llama.h.inc"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <string>
#include <vector>
#include <cinttypes>
#include <memory>
#endif

/**
 * libmtmd: A library for multimodal support in llama.cpp.
 *
 * WARNING: This API is experimental and subject to many BREAKING CHANGES.
 *          Issues related to API usage may receive lower priority support.
 *
 * For the usage, see an example in mtmd-cli.cpp
 *
 * For contributors:
 * - Make sure the C API is aligned with the libllama C API (as in llama.h.inc)
 * - Do not include model name (e.g., qwen, gemma) in the API, use generic terms instead
 * - Keep the API minimal, do not expose internal details unless necessary
 *
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MTMD_API __declspec(dllexport)
#        else
#            define MTMD_API __declspec(dllimport)
#        endif
#    else
#        define MTMD_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MTMD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
    MTMD_INPUT_CHUNK_TYPE_AUDIO,
};

// opaque types
struct mtmd_context;
struct mtmd_bitmap;
struct mtmd_image_tokens;
struct mtmd_input_chunk;
struct mtmd_input_chunks;

struct mtmd_input_text {
    const char * text;
    bool add_special;
    bool parse_special;
};

//
// C API
//

typedef struct mtmd_context      mtmd_context;
typedef struct mtmd_bitmap       mtmd_bitmap;
typedef struct mtmd_image_tokens mtmd_image_tokens;
typedef struct mtmd_input_chunk  mtmd_input_chunk;
typedef struct mtmd_input_chunks mtmd_input_chunks;
typedef struct mtmd_input_text   mtmd_input_text;

struct mtmd_context_params {
    bool use_gpu;
    bool print_timings;
    int n_threads;
    const char * image_marker; // deprecated, use media_marker instead
    const char * media_marker;
    enum llama_flash_attn_type flash_attn_type;
    bool warmup; // whether to run a warmup encode pass after initialization

    // limit number of image tokens, only for vision models with dynamic resolution
    int image_min_tokens; // minimum number of tokens for image input (default: read from metadata)
    int image_max_tokens; // maximum number of tokens for image input (default: read from metadata)

    // callback function passed over to mtmd proper
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};

MTMD_API const char * mtmd_default_marker(void);

MTMD_API struct mtmd_context_params mtmd_context_params_default(void);

// initialize the mtmd context
// return nullptr on failure
MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
                                            const struct llama_model * text_model,
                                            const struct mtmd_context_params ctx_params);

MTMD_API void mtmd_free(mtmd_context * ctx);

// whether we need to set non-causal mask before llama_decode
// if chunk is nullptr, we assume the default case where chunk is an image chunk
MTMD_API bool mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk);

// whether the current model use M-RoPE for llama_decode
MTMD_API bool mtmd_decode_use_mrope(const mtmd_context * ctx);

// whether the current model supports vision input
MTMD_API bool mtmd_support_vision(const mtmd_context * ctx);

// whether the current model supports audio input
MTMD_API bool mtmd_support_audio(const mtmd_context * ctx);

// get audio sample rate in Hz, for example 16000 for Whisper
// return -1 if audio is not supported
MTMD_API int mtmd_get_audio_sample_rate(const mtmd_context * ctx);

// mtmd_bitmap
//
// if bitmap is image:
//     length of data must be nx * ny * 3
//     the data is in RGBRGBRGB... format
// if bitmap is audio:
//     length of data must be n_samples * sizeof(float)
//     the data is in float format (PCM F32)
MTMD_API mtmd_bitmap *         mtmd_bitmap_init           (uint32_t nx, uint32_t ny, const unsigned char * data);
MTMD_API mtmd_bitmap *         mtmd_bitmap_init_from_audio(size_t n_samples,         const float         * data);
MTMD_API uint32_t              mtmd_bitmap_get_nx     (const mtmd_bitmap * bitmap);
MTMD_API uint32_t              mtmd_bitmap_get_ny     (const mtmd_bitmap * bitmap);
MTMD_API const unsigned char * mtmd_bitmap_get_data   (const mtmd_bitmap * bitmap);
MTMD_API size_t                mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap);
MTMD_API bool                  mtmd_bitmap_is_audio   (const mtmd_bitmap * bitmap);
MTMD_API void                  mtmd_bitmap_free       (mtmd_bitmap * bitmap);
// bitmap ID is optional, but useful for KV cache tracking
// these getters/setters are dedicated functions, so you can for example calculate the hash of the image based on mtmd_bitmap_get_data()
MTMD_API const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap);
MTMD_API void         mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id);


// mtmd_input_chunks
//
// this is simply a list of mtmd_input_chunk
// the elements can only be populated via mtmd_tokenize()
MTMD_API mtmd_input_chunks *      mtmd_input_chunks_init(void);
MTMD_API size_t                   mtmd_input_chunks_size(const mtmd_input_chunks * chunks);
MTMD_API const mtmd_input_chunk * mtmd_input_chunks_get (const mtmd_input_chunks * chunks, size_t idx);
MTMD_API void                     mtmd_input_chunks_free(mtmd_input_chunks * chunks);

// mtmd_input_chunk
//
// the instance will be constructed via mtmd_tokenize()
// it will be freed along with mtmd_input_chunks
MTMD_API enum mtmd_input_chunk_type mtmd_input_chunk_get_type        (const mtmd_input_chunk * chunk);
MTMD_API const llama_token *        mtmd_input_chunk_get_tokens_text (const mtmd_input_chunk * chunk, size_t * n_tokens_output);
MTMD_API const mtmd_image_tokens *  mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk);
MTMD_API size_t                     mtmd_input_chunk_get_n_tokens    (const mtmd_input_chunk * chunk);
// returns nullptr for ID on text chunk
MTMD_API const char *               mtmd_input_chunk_get_id          (const mtmd_input_chunk * chunk);
// number of temporal positions (equals to max(t,h,w) for M-RoPE; equals to n_tokens otherwise)
MTMD_API llama_pos                  mtmd_input_chunk_get_n_pos       (const mtmd_input_chunk * chunk);

// in case you want to use custom logic to handle the chunk (i.e. KV cache management)
// you can move the chunk ownership to your own code by copying it
// remember to free the chunk when you are done with it
MTMD_API mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk);
MTMD_API void               mtmd_input_chunk_free(mtmd_input_chunk * chunk);


// mtmd_image_tokens
//
// the instance will be constructed via mtmd_tokenize()
// it will be freed along with mtmd_input_chunk
MTMD_API size_t       mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens); // TODO: deprecate
MTMD_API const char * mtmd_image_tokens_get_id      (const mtmd_image_tokens * image_tokens); // TODO: deprecate
// number of temporal positions (equals to max(t,h,w) for M-RoPE; equals to n_tokens otherwise)
MTMD_API llama_pos    mtmd_image_tokens_get_n_pos   (const mtmd_image_tokens * image_tokens); // TODO: deprecate

DEPRECATED(MTMD_API size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens),
           "use mtmd_image_tokens_get_decoder_pos() instead");
DEPRECATED(MTMD_API size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens),
           "use mtmd_image_tokens_get_decoder_pos() instead");

struct mtmd_decoder_pos {
    uint32_t t;
    uint32_t x;
    uint32_t y;
    uint32_t z; // unused for now, reserved for future use
};
// get position for decoder attention, to be used by M-RoPE models
// i is the index of the embedding token, ranging from 0 to mtmd_image_tokens_get_n_tokens() - 1
// pos_0 is the absolute position of the first token
// return relative position (for example, embedding 0 will have position (0, 0, 0); remember to adjust it to the current absolute position)
MTMD_API struct mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i);

// tokenize an input text prompt and a list of bitmaps (images/audio)
// the prompt must have the input image marker (default: "<__media__>") in it
// the default marker is defined by mtmd_default_marker()
// the marker will be replaced with the image/audio chunk
// for example:
//   "here is an image: <__media__>\ndescribe it in detail."
//   this will gives 3 chunks:
//   1. "here is an image: <start_of_image>"
//   2. (image/audio tokens)
//   3. "<end_of_image>\ndescribe it in detail."
// number of bitmaps must be equal to the number of markers in the prompt
// this function is thread-safe (shared ctx)
// return values:
//   0 on success
//   1 on number of bitmaps not matching the number of markers
//   2 on image preprocessing error
MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
                               mtmd_input_chunks * output,
                               const mtmd_input_text * text,
                               const mtmd_bitmap ** bitmaps,
                               size_t n_bitmaps);

// returns 0 on success
// TODO: deprecate
MTMD_API int32_t mtmd_encode(mtmd_context * ctx,
                             const mtmd_image_tokens * image_tokens);

// returns 0 on success
MTMD_API int32_t mtmd_encode_chunk(mtmd_context * ctx,
                                   const mtmd_input_chunk * chunk);

// get output embeddings from the last encode pass
// the reading size (in bytes) is equal to:
// llama_model_n_embd_inp(model) * mtmd_input_chunk_get_n_tokens(chunk) * sizeof(float)
MTMD_API float * mtmd_get_output_embd(mtmd_context * ctx);

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
MTMD_API void mtmd_log_set(ggml_log_callback log_callback, void * user_data);

/////////////////////////////////////////

//
// libmtmd helper functions
//
// Please note that these helpers are not guaranteed to be stable.
// BREAKING CHANGES are expected.
//

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
// Note: this also call mtmd_log_set() internally
MTMD_API void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data);

// helper function to construct a mtmd_bitmap from a file
// it calls mtmd_helper_bitmap_init_from_buf() internally
// returns nullptr on failure
// this function is thread-safe
MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname);

// helper function to construct a mtmd_bitmap from a buffer containing a file
// supported formats:
//     image: formats supported by stb_image: jpg, png, bmp, gif, etc.
//     audio: formats supported by miniaudio: wav, mp3, flac
// note: audio files will be auto-detected based on magic bytes
// returns nullptr on failure
// this function is thread-safe
MTMD_API mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len);

// helper to count the total number of tokens from a list of chunks, useful to keep track of KV cache
MTMD_API size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks);

// helper to count the total position of tokens from a list of chunks, useful to keep track of n_past
// normally, n_pos is equal to n_tokens, but for M-RoPE it is different
MTMD_API llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks);

// helper to get the list of relative positions corresponding to the embedding tokens, to be used by M-RoPE
// out_pos must have length == mtmd_helper_get_n_tokens(image)
MTMD_API void mtmd_helper_image_get_decoder_pos(const mtmd_image_tokens * image, llama_pos pos_0, struct mtmd_decoder_pos * out_pos);

// helper function that automatically:
// 1. run llama_decode() on text chunks
// 2. run mtmd_encode() on image chunks, then mtmd_get_output_embd() and then llama_decode()
// if any of the mtmd_encode() or llama_decode() calls return non-zero, stop and forward the error
// otherwise, returns 0 on success
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                         struct llama_context * lctx,
                                         const mtmd_input_chunks * chunks,
                                         llama_pos n_past,
                                         llama_seq_id seq_id,
                                         int32_t n_batch,
                                         bool logits_last,
                                         llama_pos * new_n_past);

// works like mtmd_helper_eval_chunks(), but only for a single chunk
// this function is NOT thread-safe
MTMD_API int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
                                               struct llama_context * lctx,
                                               const mtmd_input_chunk * chunk,
                                               llama_pos n_past,
                                               llama_seq_id seq_id,
                                               int32_t n_batch,
                                               bool logits_last,
                                               llama_pos * new_n_past);

// helper function to decode an image whose embeddings have already been calculated
// this helper will handle batching and pre/post decoding setup (for ex. gemma 3 requires non-causal attention)
// ret 0 on success, -1 on chunk not being a valid image chunk, 1 on decode failure
MTMD_API int32_t mtmd_helper_decode_image_chunk(mtmd_context * ctx,
                                                struct llama_context * lctx,
                                                const mtmd_input_chunk * chunk,
                                                float * encoded_embd,
                                                llama_pos n_past,
                                                llama_seq_id seq_id,
                                                int32_t n_batch,
                                                llama_pos * new_n_past);

// test function, to be used in test-mtmd-c-api.c
MTMD_API mtmd_input_chunks * mtmd_test_create_input_chunks(void);

#ifdef __cplusplus
} // extern "C"
#endif

//
// C++ wrappers
//

#ifdef __cplusplus

namespace mtmd {

struct mtmd_context_deleter {
    void operator()(mtmd_context * val) { mtmd_free(val); }
};
using context_ptr = std::unique_ptr<mtmd_context, mtmd_context_deleter>;

struct mtmd_bitmap_deleter {
    void operator()(mtmd_bitmap * val) { mtmd_bitmap_free(val); }
};
using bitmap_ptr = std::unique_ptr<mtmd_bitmap, mtmd_bitmap_deleter>;

struct mtmd_input_chunks_deleter {
    void operator()(mtmd_input_chunks * val) { mtmd_input_chunks_free(val); }
};
using input_chunks_ptr = std::unique_ptr<mtmd_input_chunks, mtmd_input_chunks_deleter>;

struct mtmd_input_chunk_deleter {
    void operator()(mtmd_input_chunk * val) { mtmd_input_chunk_free(val); }
};
using input_chunk_ptr = std::unique_ptr<mtmd_input_chunk, mtmd_input_chunk_deleter>;

struct bitmap {
    bitmap_ptr ptr;
    bitmap() : ptr(nullptr) {}
    bitmap(mtmd_bitmap * bitmap) : ptr(bitmap) {}
    bitmap(bitmap && other) noexcept : ptr(std::move(other.ptr)) {}
    bitmap(uint32_t nx, uint32_t ny, const unsigned char * data) {
        ptr.reset(mtmd_bitmap_init(nx, ny, data));
    }
    ~bitmap() = default;
    uint32_t nx() const { return mtmd_bitmap_get_nx(ptr.get()); }
    uint32_t ny() const { return mtmd_bitmap_get_ny(ptr.get()); }
    const unsigned char * data() const { return mtmd_bitmap_get_data(ptr.get()); }
    size_t n_bytes() const { return mtmd_bitmap_get_n_bytes(ptr.get()); }
    std::string id() const { return mtmd_bitmap_get_id(ptr.get()); }
    void set_id(const char * id) const { mtmd_bitmap_set_id(ptr.get(), id); }
};

struct bitmaps {
    std::vector<bitmap> entries;
    ~bitmaps() = default;
    // return list of pointers to mtmd_bitmap
    // example:
    //   auto bitmaps_c_ptr = bitmaps.c_ptr();
    //   int32_t res = mtmd_tokenize(... bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    std::vector<const mtmd_bitmap *> c_ptr() {
        std::vector<const mtmd_bitmap *> res(entries.size());
        for (size_t i = 0; i < entries.size(); i++) {
            res[i] = entries[i].ptr.get();
        }
        return res;
    }
};

struct input_chunks {
    input_chunks_ptr ptr;
    input_chunks() = default;
    input_chunks(mtmd_input_chunks * chunks) : ptr(chunks) {}
    ~input_chunks() = default;
    size_t size() const { return mtmd_input_chunks_size(ptr.get()); }
    const mtmd_input_chunk * operator[](size_t idx) const {
        return mtmd_input_chunks_get(ptr.get(), idx);
    }
};

} // namespace mtmd

#ifdef MTMD_INTERNAL_HEADER
#include "gguf.h.inc"

#include <array>
#include <cmath>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string_view>
#include <unordered_set>

// Internal declarations for the CLIP graph and mtmd implementation files.

struct clip_ctx;

struct clip_image_size {
    int width;
    int height;
};

struct clip_image_f32;
struct clip_image_u8_batch;
struct clip_image_f32_batch;

enum clip_modality {
    CLIP_MODALITY_VISION,
    CLIP_MODALITY_AUDIO,
};

enum clip_flash_attn_type {
    CLIP_FLASH_ATTN_TYPE_AUTO     = -1,
    CLIP_FLASH_ATTN_TYPE_DISABLED = 0,
    CLIP_FLASH_ATTN_TYPE_ENABLED  = 1,
};

struct clip_context_params {
    bool use_gpu;
    enum clip_flash_attn_type flash_attn_type;
    int image_min_tokens;
    int image_max_tokens;
    bool warmup;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};

struct clip_init_result {
    struct clip_ctx * ctx_v; // vision context
    struct clip_ctx * ctx_a; // audio context
};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params);

void clip_free(struct clip_ctx * ctx);

size_t clip_embd_nbytes(const struct clip_ctx * ctx);
size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_w, int img_h);

int32_t clip_get_image_size (const struct clip_ctx * ctx);
int32_t clip_get_patch_size (const struct clip_ctx * ctx);
int32_t clip_get_hidden_size(const struct clip_ctx * ctx);

// TODO: should be enum, not string
const char * clip_patch_merge_type(const struct clip_ctx * ctx);

int clip_n_output_tokens(const struct clip_ctx * ctx, struct clip_image_f32 * img);

// for M-RoPE, this will be the number of token positions in X and Y directions
// for other models, X will be the total number of tokens and Y will be 1
int clip_n_output_tokens_x(const struct clip_ctx * ctx, struct clip_image_f32 * img);
int clip_n_output_tokens_y(const struct clip_ctx * ctx, struct clip_image_f32 * img);

// this should be equal to the embedding dimension of the text model
int clip_n_mmproj_embd(const struct clip_ctx * ctx);

struct clip_image_size      * clip_image_size_init(void);
struct clip_image_u8        * clip_image_u8_init (void);
struct clip_image_f32       * clip_image_f32_init(void);
struct clip_image_f32_batch * clip_image_f32_batch_init(void); // only used by libllava

// nx, ny are the output image dimensions
unsigned char * clip_image_u8_get_data(struct clip_image_u8 * img, uint32_t * nx, uint32_t * ny);

void clip_image_size_free (struct clip_image_size * img_size);
void clip_image_u8_free (struct clip_image_u8  * img);
void clip_image_f32_free(struct clip_image_f32 * img);
void clip_image_u8_batch_free (struct clip_image_u8_batch  * batch);
void clip_image_f32_batch_free(struct clip_image_f32_batch * batch);

// use for accessing underlay data of clip_image_f32_batch
size_t clip_image_f32_batch_n_images(const struct clip_image_f32_batch * batch); // equivalent to batch->size()
size_t clip_image_f32_batch_nx(const struct clip_image_f32_batch * batch, int idx); // equivalent to batch[idx]->nx
size_t clip_image_f32_batch_ny(const struct clip_image_f32_batch * batch, int idx); // equivalent to batch[idx]->ny
struct clip_image_f32 * clip_image_f32_get_img(const struct clip_image_f32_batch * batch, int idx); // equivalent to batch[idx]->data

/**
 * Build image from pixels decoded by other libraries instead of stb_image.h.inc for better performance.
 * The memory layout is RGBRGBRGB..., input buffer length must be 3*nx*ny bytes
 */
void clip_build_img_from_pixels(const unsigned char * rgb_pixels, int nx, int ny, struct clip_image_u8 * img);

struct ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx);

bool clip_image_encode      (struct clip_ctx * ctx, int n_threads, struct clip_image_f32 * img, float * vec);
bool clip_image_batch_encode(struct clip_ctx * ctx, int n_threads, const struct clip_image_f32_batch * imgs, float * vec);

int clip_is_minicpmv(const struct clip_ctx * ctx);
bool clip_is_glm(const struct clip_ctx * ctx);
bool clip_is_llava(const struct clip_ctx * ctx);
// note for contributor: this clip_is_(model) pattern is deprecated
//                       do NOT add new functions like this

bool clip_encode_float_image (struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec);

// use by audio input
void clip_image_f32_batch_add_mel(struct clip_image_f32_batch * batch, int n_mel, int n_frames, float * mel);

bool clip_has_vision_encoder(const struct clip_ctx * ctx);
bool clip_has_audio_encoder(const struct clip_ctx * ctx);
bool clip_has_whisper_encoder(const struct clip_ctx * ctx);

#define KEY_FTYPE               "general.file_type"
#define KEY_NAME                "general.name"
#define KEY_DESCRIPTION         "general.description"
#define KEY_PROJ_TYPE           "clip.projector_type"
#define KEY_HAS_AUDIO_ENC       "clip.has_audio_encoder"
#define KEY_HAS_VISION_ENC      "clip.has_vision_encoder"
#define KEY_USE_GELU            "clip.use_gelu"
#define KEY_USE_SILU            "clip.use_silu"

#define KEY_N_EMBD              "clip.%s.embedding_length"
#define KEY_N_FF                "clip.%s.feed_forward_length"
#define KEY_N_BLOCK             "clip.%s.block_count"
#define KEY_PROJ_DIM            "clip.%s.projection_dim"
#define KEY_N_HEAD              "clip.%s.attention.head_count"
#define KEY_LAYER_NORM_EPS      "clip.%s.attention.layer_norm_epsilon"

// vision-specific
#define KEY_VISION_PROJ_TYPE    "clip.vision.projector_type" // for models with mixed modalities
#define KEY_IMAGE_SIZE          "clip.vision.image_size"
#define KEY_IMAGE_MIN_PIXELS    "clip.vision.image_min_pixels"
#define KEY_IMAGE_MAX_PIXELS    "clip.vision.image_max_pixels"
#define KEY_PREPROC_MIN_TILES   "clip.vision.preproc_min_tiles"
#define KEY_PREPROC_MAX_TILES   "clip.vision.preproc_max_tiles"
#define KEY_PREPROC_IMAGE_SIZE  "clip.vision.preproc_image_size"
#define KEY_PATCH_SIZE          "clip.vision.patch_size"
#define KEY_IMAGE_MEAN          "clip.vision.image_mean"
#define KEY_IMAGE_STD           "clip.vision.image_std"
#define KEY_FEATURE_LAYER       "clip.vision.feature_layer"
#define KEY_PROJ_SCALE_FACTOR   "clip.vision.projector.scale_factor"
#define KEY_SPATIAL_MERGE_SIZE  "clip.vision.spatial_merge_size"
#define KEY_IS_DEEPSTACK_LAYERS "clip.vision.is_deepstack_layers"

#define KEY_MM_PATCH_MERGE_TYPE    "clip.vision.mm_patch_merge_type"
#define KEY_IMAGE_GRID_PINPOINTS   "clip.vision.image_grid_pinpoints"
#define KEY_WIN_ATTN_PATTERN       "clip.vision.n_wa_pattern"
#define KEY_WIN_ATTN_LAYER_INDEXES "clip.vision.wa_layer_indexes"
#define KEY_ATTN_WINDOW_SIZE       "clip.vision.window_size"
#define KEY_MINICPMV_VERSION       "clip.minicpmv_version"
#define KEY_MINICPMV_QUERY_NUM     "clip.minicpmv_query_num"
#define KEY_SAM_N_HEAD             "clip.vision.sam.head_count"
#define KEY_SAM_N_BLOCK            "clip.vision.sam.block_count"
#define KEY_SAM_N_EMBD             "clip.vision.sam.embedding_length"
// audio-specific
#define KEY_AUDIO_PROJ_TYPE     "clip.audio.projector_type" // for models with mixed modalities
#define KEY_A_NUM_MEL_BINS      "clip.audio.num_mel_bins"
#define KEY_A_PROJ_STACK_FACTOR "clip.audio.projector.stack_factor"


//
// tensor name constants
//

#define TN_POS_EMBD        "%s.position_embd.weight"
#define TN_CLASS_EMBD      "v.class_embd"
#define TN_PATCH_EMBD      "v.patch_embd.weight"  // not rename tensor with ".0" postfix for backward compat
#define TN_PATCH_EMBD_1    "v.patch_embd.weight.1"
#define TN_PATCH_BIAS      "v.patch_embd.bias"
#define TN_NORM_EMBD       "v.norm_embd.%s"
#define TN_ATTN_QKV        "%s.blk.%d.attn_qkv.%s"
#define TN_ATTN_K          "%s.blk.%d.attn_k.%s"
#define TN_ATTN_Q          "%s.blk.%d.attn_q.%s"
#define TN_ATTN_V          "%s.blk.%d.attn_v.%s"
#define TN_ATTN_OUTPUT     "%s.blk.%d.attn_out.%s"
#define TN_ATTN_K_NORM     "%s.blk.%d.attn_k_norm.%s"
#define TN_ATTN_Q_NORM     "%s.blk.%d.attn_q_norm.%s"
#define TN_FFN_DOWN        "%s.blk.%d.ffn_down.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_FFN_UP          "%s.blk.%d.ffn_up.%s"
#define TN_FFN_GATE        "%s.blk.%d.ffn_gate.%s"
#define TN_LN_1            "%s.blk.%d.ln1.%s" // layer norm
#define TN_LN_2            "%s.blk.%d.ln2.%s" // layer norm
#define TN_LS_1            "%s.blk.%d.ls1.%s"         // layer scale
#define TN_LS_2            "%s.blk.%d.ls2.%s"         // layer scale
#define TN_LS_OUT          "%s.blk.%d.out_scale.%s"      // layer out scale (gemma4)
#define TN_ATTN_POST_NORM  "%s.blk.%d.attn_post_norm.%s" // post-attn norm (gemma4)
#define TN_FFN_POST_NORM   "%s.blk.%d.ffn_post_norm.%s"  // post-FFN norm (gemma4)
#define TN_LN_PRE          "%s.pre_ln.%s"
#define TN_LN_POST         "%s.post_ln.%s"
#define TN_LLAVA_PROJ      "mm.%d.%s"
#define TN_MM_UP           "mm.up.%s"
#define TN_MM_GATE         "mm.gate.%s"
#define TN_MM_DOWN         "mm.down.%s"
#define TN_MM_POST_NORM    "mm.post_norm.%s"
#define TN_MVLM_PROJ_MLP   "mm.model.mlp.%d.%s"
#define TN_MVLM_PROJ_BLOCK "mm.model.mb_block.%d.block.%d.%s"
#define TN_MVLM_PROJ_PEG   "mm.model.peg.%d.%s"
#define TN_IMAGE_NEWLINE   "v.image_newline"
#define TN_IMAGE_SEPERATOR "v.view_seperator"
#define TN_MM_INP_NORM     "mm.input_norm.weight"
#define TN_MM_INP_NORM_B   "mm.input_norm.bias"
#define TN_MM_INP_PROJ     "mm.input_projection.weight" // gemma3
#define TN_MM_SOFT_EMB_N   "mm.soft_emb_norm.weight"    // gemma3
#define TN_MM_PROJECTOR    "mm.model.fc.%s"             // idefics3, deepseekocr
#define TN_MM_PATCH_MERGER "mm.patch_merger.%s"         // mistral small 3.1, glm4v
#define TN_TOK_IMG_BREAK   "v.token_embd.img_break"     // pixtral
#define TN_TOK_GLM_BOI     "adapter.boi"                // glm-edge (these embeddings are not in text model)
#define TN_TOK_GLM_EOI     "adapter.eoi"                // glm-edge (these embeddings are not in text model)
#define TN_DEEPSTACK_NORM  "v.deepstack.%d.norm.%s"     // qwen3vl deepstack
#define TN_DEEPSTACK_FC1   "v.deepstack.%d.fc1.%s"      // qwen3vl deepstack
#define TN_DEEPSTACK_FC2   "v.deepstack.%d.fc2.%s"      // qwen3vl deepstack

// mimicpmv
#define TN_MINICPMV_POS_EMBD_K "resampler.pos_embed_k"
#define TN_MINICPMV_QUERY      "resampler.query"
#define TN_MINICPMV_PROJ       "resampler.proj.weight"
#define TN_MINICPMV_KV_PROJ    "resampler.kv.weight"
#define TN_MINICPMV_ATTN       "resampler.attn.%s.%s"
#define TN_MINICPMV_LN         "resampler.ln_%s.%s"

#define TN_GLM_ADAPER_CONV      "adapter.conv.%s"
#define TN_GLM_ADAPTER_LINEAR   "adapter.linear.linear.%s"
#define TN_GLM_ADAPTER_NORM_1   "adapter.linear.norm1.%s"
#define TN_GLM_ADAPTER_D_H_2_4H "adapter.linear.dense_h_to_4h.%s"
#define TN_GLM_ADAPTER_GATE     "adapter.linear.gate.%s"
#define TN_GLM_ADAPTER_D_4H_2_H "adapter.linear.dense_4h_to_h.%s"

// ultravox
#define TN_CONV1D       "a.conv1d.%d.%s"
#define TN_CONV2D       "a.conv2d.%d.%s"
#define TN_CONV_OUT     "a.conv_out.%s"
#define TN_MM_AUDIO_MLP "mm.a.mlp.%d.%s"
#define TN_MM_AUDIO_FC  "mm.a.fc.%s" // fully connected layer
#define TN_MM_NORM_PRE  "mm.a.norm_pre.%s"
#define TN_MM_NORM_MID  "mm.a.norm_mid.%s"

// cogvlm
#define TN_MM_POST_FC_NORM "mm.post_fc_norm.%s"
#define TN_MM_H_TO_4H      "mm.up.%s"
#define TN_MM_GATE         "mm.gate.%s"
#define TN_MM_4H_TO_H      "mm.down.%s"
#define TN_TOK_BOI         "v.boi"
#define TN_TOK_EOI         "v.eoi"

// hunyuanocr / hunyuanvl (shared GGUF tensor names)
#define TN_MM_PRE_NORM     "mm.pre_norm.%s"
#define TN_TOK_IMG_BEGIN   "mm.image_begin"
#define TN_TOK_IMG_END     "mm.image_end"

// deepseek-ocr
#define TN_SAM_POS_EMBD   "v.sam.pos_embd.%s"
#define TN_SAM_PATCH_EMBD "v.sam.patch_embd.%s"
#define TN_SAM_PRE_NORM   "v.sam.blk.%d.pre_ln.%s"
#define TN_SAM_POST_NORM  "v.sam.blk.%d.post_ln.%s"
#define TN_SAM_ATTN_POS_H "v.sam.blk.%d.attn.pos_h.%s"
#define TN_SAM_ATTN_POS_W "v.sam.blk.%d.attn.pos_w.%s"
#define TN_SAM_ATTN_QKV   "v.sam.blk.%d.attn.qkv.%s"
#define TN_SAM_ATTN_OUT   "v.sam.blk.%d.attn.out.%s"
#define TN_SAM_FFN_UP     "v.sam.blk.%d.mlp.lin1.%s"
#define TN_SAM_FFN_DOWN   "v.sam.blk.%d.mlp.lin2.%s"
#define TN_SAM_NECK       "v.sam.neck.%d.%s"
#define TN_SAM_NET        "v.sam.net_%d.%s"
// (conformer) lfm2
#define TN_PRE_ENCODE_OUT  "a.pre_encode.out.%s"
#define TN_FFN_NORM        "%s.blk.%d.ffn_norm.%s"
#define TN_FFN_NORM_1      "%s.blk.%d.ffn_norm_1.%s"
#define TN_FFN_UP_1        "%s.blk.%d.ffn_up_1.%s"
#define TN_FFN_DOWN_1      "%s.blk.%d.ffn_down_1.%s"
#define TN_POS_BIAS_U      "%s.blk.%d.pos_bias_u"
#define TN_POS_BIAS_V      "%s.blk.%d.pos_bias_v"
#define TN_NORM_CONV       "%s.blk.%d.norm_conv.%s"
#define TN_LINEAR_POS      "%s.blk.%d.linear_pos.%s"
#define TN_CONV_DW         "%s.blk.%d.conv_dw.%s"
#define TN_CONV_NORM       "%s.blk.%d.conv_norm.%s"
#define TN_CONV_PW1        "%s.blk.%d.conv_pw1.%s"
#define TN_CONV_PW2        "%s.blk.%d.conv_pw2.%s"

// gemma4 audio conformer
#define TN_A_MM_INP_PROJ     "mm.a.input_projection.%s"
#define TN_A_MM_SOFT_EMB_N   "mm.a.soft_emb_norm.%s"
#define TN_A_INP_PROJ        "a.input_projection.%s"
#define TN_A_CONV1D          "a.conv1d.%d.%s"
#define TN_A_CONV1D_NORM     "a.conv1d.%d.norm.%s"
#define TN_A_OUT_PROJ        "a.pre_encode.out.%s"
#define TN_A_ATTN_PRE_NORM   "%s.blk.%d.attn_pre_norm.%s"
#define TN_A_ATTN_POST_NORM  "%s.blk.%d.attn_post_norm.%s"
#define TN_A_ATTN_K_REL      "%s.blk.%d.attn_k_rel.%s"
#define TN_A_PER_DIM_SCALE   "%s.blk.%d.per_dim_scale.%s"
#define TN_A_PER_DIM_K_SCALE "%s.blk.%d.per_dim_k_scale.%s"
#define TN_A_FFN_POST_NORM   "%s.blk.%d.ffn_post_norm.%s"
#define TN_A_FFN_POST_NORM_1 "%s.blk.%d.ffn_post_norm_1.%s"

// mobilenetv5 (gemma3n) definitions
#define TN_MNV5_STEM_CONV        "v.conv_stem.conv.weight"
#define TN_MNV5_STEM_BIAS        "v.conv_stem.conv.bias"
#define TN_MNV5_STEM_BN          "v.conv_stem.bn.weight"

// Stage 0 Block (Edge Residual)
#define TN_MNV5_BLK_S0_EXP_W     "v.blk.%d.%d.conv_exp.weight"
#define TN_MNV5_BLK_S0_BN1_W     "v.blk.%d.%d.bn1.weight"
#define TN_MNV5_BLK_S0_PWL_W     "v.blk.%d.%d.conv_pwl.weight"
#define TN_MNV5_BLK_S0_BN2_W     "v.blk.%d.%d.bn2.weight"

// Stage 1+ Block (Universal Inverted Residual)
#define TN_MNV5_BLK_DW_START_W   "v.blk.%d.%d.dw_start.conv.weight"
#define TN_MNV5_BLK_DW_START_BN  "v.blk.%d.%d.dw_start.bn.weight"
#define TN_MNV5_BLK_DW_MID_W     "v.blk.%d.%d.dw_mid.conv.weight"
#define TN_MNV5_BLK_DW_MID_BN    "v.blk.%d.%d.dw_mid.bn.weight"
#define TN_MNV5_BLK_PW_EXP_W     "v.blk.%d.%d.pw_exp.conv.weight"
#define TN_MNV5_BLK_PW_EXP_BN    "v.blk.%d.%d.pw_exp.bn.weight"
#define TN_MNV5_BLK_PW_PROJ_W    "v.blk.%d.%d.pw_proj.conv.weight"
#define TN_MNV5_BLK_PW_PROJ_BN   "v.blk.%d.%d.pw_proj.bn.weight"
#define TN_MNV5_BLK_LAYER_SCALE  "v.blk.%d.%d.layer_scale.gamma"

// Attention Components
#define TN_MNV5_ATTN_Q_W         "v.blk.%d.%d.attn.query.proj.weight"
#define TN_MNV5_ATTN_K_W         "v.blk.%d.%d.attn.key.proj.weight"
#define TN_MNV5_ATTN_V_W         "v.blk.%d.%d.attn.value.proj.weight"
#define TN_MNV5_ATTN_O_W         "v.blk.%d.%d.attn.output.proj.weight"
#define TN_MNV5_ATTN_K_DW        "v.blk.%d.%d.attn.key.down_conv.weight"
#define TN_MNV5_ATTN_K_NORM      "v.blk.%d.%d.attn.key.norm.weight"
#define TN_MNV5_ATTN_V_DW        "v.blk.%d.%d.attn.value.down_conv.weight"
#define TN_MNV5_ATTN_V_NORM      "v.blk.%d.%d.attn.value.norm.weight"
#define TN_MNV5_ATTN_NORM        "v.blk.%d.%d.norm.weight" // Block norm used in attn blocks

// MSFA
#define TN_MNV5_MSFA_FFN_EXP_W   "v.msfa.ffn.pw_exp.conv.weight"
#define TN_MNV5_MSFA_FFN_EXP_BN  "v.msfa.ffn.pw_exp.bn.weight"
#define TN_MNV5_MSFA_FFN_PROJ_W  "v.msfa.ffn.pw_proj.conv.weight"
#define TN_MNV5_MSFA_FFN_PROJ_BN "v.msfa.ffn.pw_proj.bn.weight"
#define TN_MNV5_MSFA_NORM        "v.msfa.norm.weight"

// gemma4
#define TN_STD_BIAS              "v.std_bias"
#define TN_STD_SCALE             "v.std_scale"

// yasa2
#define TN_YASA_PATCH_LN_W       "v.patch_ln.weight"
#define TN_YASA_PATCH_LN_B       "v.patch_ln.bias"
#define TN_YASA_BACKBONE_LN_W    "v.backbone_ln.weight"
#define TN_YASA_BACKBONE_LN_B    "v.backbone_ln.bias"
#define TN_YASA_POS_EMBD         "v.vision_pos_embed"
#define TN_YASA_STAGE_DOWN_LN    "v.stage.%d.down.ln.%s"
#define TN_YASA_STAGE_DOWN_CONV  "v.stage.%d.down.conv.%s"
#define TN_YASA_STAGE_BLK        "v.stage.%d.blk.%d.%s.%s"

// align x to upper multiple of n
#define CLIP_ALIGN(x, n) ((((x) + (n) - 1) / (n)) * (n))

// forward declaration
// TODO: improve this later
struct clip_ctx;

enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_MLP_NORM,
    PROJECTOR_TYPE_LDP,
    PROJECTOR_TYPE_LDPV2,
    PROJECTOR_TYPE_MINICPMV,
    PROJECTOR_TYPE_GLM_EDGE,
    PROJECTOR_TYPE_QWEN2VL,
    PROJECTOR_TYPE_QWEN3VL,
    PROJECTOR_TYPE_STEP3VL,
    PROJECTOR_TYPE_GEMMA3,
    PROJECTOR_TYPE_GEMMA3NV,
    PROJECTOR_TYPE_GEMMA3NA,
    PROJECTOR_TYPE_GEMMA4V,
    PROJECTOR_TYPE_GEMMA4A,
    PROJECTOR_TYPE_PHI4,
    PROJECTOR_TYPE_IDEFICS3,
    PROJECTOR_TYPE_PIXTRAL,
    PROJECTOR_TYPE_QWEN25VL,
    PROJECTOR_TYPE_ULTRAVOX,
    PROJECTOR_TYPE_INTERNVL,
    PROJECTOR_TYPE_LLAMA4,
    PROJECTOR_TYPE_QWEN2A,
    PROJECTOR_TYPE_QWEN3A,
    PROJECTOR_TYPE_GLMA,
    PROJECTOR_TYPE_QWEN25O, // will be replaced by QWEN2A or QWEN25VL depending on clip_ctx
    PROJECTOR_TYPE_VOXTRAL,
    PROJECTOR_TYPE_MERALION,
    PROJECTOR_TYPE_MUSIC_FLAMINGO,
    PROJECTOR_TYPE_LFM2,
    PROJECTOR_TYPE_KIMIVL,
    PROJECTOR_TYPE_PADDLEOCR,
    PROJECTOR_TYPE_LIGHTONOCR,
    PROJECTOR_TYPE_COGVLM,
    PROJECTOR_TYPE_JANUS_PRO,
    PROJECTOR_TYPE_DOTS_OCR,
    PROJECTOR_TYPE_DEEPSEEKOCR,
    PROJECTOR_TYPE_LFM2A,
    PROJECTOR_TYPE_GLM4V,
    PROJECTOR_TYPE_YOUTUVL,
    PROJECTOR_TYPE_YASA2,
    PROJECTOR_TYPE_KIMIK25,
    PROJECTOR_TYPE_NEMOTRON_V2_VL,
    PROJECTOR_TYPE_HUNYUANOCR,
    PROJECTOR_TYPE_HUNYUANVL,
    PROJECTOR_TYPE_UNKNOWN,
};

static std::map<projector_type, std::string> PROJECTOR_TYPE_NAMES = {
    { PROJECTOR_TYPE_MLP,       "mlp" },
    { PROJECTOR_TYPE_LDP,       "ldp" },
    { PROJECTOR_TYPE_LDPV2,     "ldpv2"},
    { PROJECTOR_TYPE_MINICPMV,  "resampler"},
    { PROJECTOR_TYPE_GLM_EDGE,  "adapter"},
    { PROJECTOR_TYPE_QWEN2VL,   "qwen2vl_merger"},
    { PROJECTOR_TYPE_QWEN25VL,  "qwen2.5vl_merger"},
    { PROJECTOR_TYPE_QWEN3VL,   "qwen3vl_merger"},
    { PROJECTOR_TYPE_STEP3VL,   "step3vl"},
    { PROJECTOR_TYPE_GEMMA3,    "gemma3"},
    { PROJECTOR_TYPE_GEMMA3NV,  "gemma3nv"},
    { PROJECTOR_TYPE_GEMMA3NA,  "gemma3na"},
    { PROJECTOR_TYPE_GEMMA4V,   "gemma4v"},
    { PROJECTOR_TYPE_GEMMA4A,   "gemma4a"},
    { PROJECTOR_TYPE_PHI4,      "phi4"},
    { PROJECTOR_TYPE_IDEFICS3,  "idefics3"},
    { PROJECTOR_TYPE_PIXTRAL,   "pixtral"},
    { PROJECTOR_TYPE_ULTRAVOX,  "ultravox"},
    { PROJECTOR_TYPE_INTERNVL,  "internvl"},
    { PROJECTOR_TYPE_LLAMA4,    "llama4"},
    { PROJECTOR_TYPE_QWEN2A,    "qwen2a"},
    { PROJECTOR_TYPE_QWEN3A,    "qwen3a"},
    { PROJECTOR_TYPE_GLMA,      "glma"},
    { PROJECTOR_TYPE_QWEN25O,   "qwen2.5o"},
    { PROJECTOR_TYPE_VOXTRAL,   "voxtral"},
    { PROJECTOR_TYPE_MERALION,  "meralion"},
    { PROJECTOR_TYPE_MUSIC_FLAMINGO, "musicflamingo"},
    { PROJECTOR_TYPE_LFM2,      "lfm2"},
    { PROJECTOR_TYPE_KIMIVL,    "kimivl"},
    { PROJECTOR_TYPE_PADDLEOCR, "paddleocr"},
    { PROJECTOR_TYPE_LIGHTONOCR,"lightonocr"},
    { PROJECTOR_TYPE_COGVLM,    "cogvlm"},
    { PROJECTOR_TYPE_JANUS_PRO, "janus_pro"},
    { PROJECTOR_TYPE_DOTS_OCR,  "dots_ocr"},
    { PROJECTOR_TYPE_DEEPSEEKOCR,"deepseekocr"},
    { PROJECTOR_TYPE_LFM2A,     "lfm2a"},
    { PROJECTOR_TYPE_GLM4V,     "glm4v"},
    { PROJECTOR_TYPE_YOUTUVL,   "youtuvl"},
    { PROJECTOR_TYPE_YASA2,     "yasa2"},
    { PROJECTOR_TYPE_KIMIK25,   "kimik25"},
    { PROJECTOR_TYPE_NEMOTRON_V2_VL, "nemotron_v2_vl"},
    { PROJECTOR_TYPE_HUNYUANOCR, "hunyuanocr"},
    { PROJECTOR_TYPE_HUNYUANVL,  "hunyuanvl"},
};

static projector_type clip_projector_type_from_string(const std::string & str) {
    for (const auto & pair : PROJECTOR_TYPE_NAMES) {
        if (pair.second == str) {
            return pair.first;
        }
    }
    return PROJECTOR_TYPE_UNKNOWN;
}

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// For images, buf.size() == nx*ny*3
//     Memory layout: RGBRGBRGB...
// For audio, only one channel is used, buf.size() == nx*ny
//     nx will be n_frames and ny will be n_mel
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

//
// logging
//

static void clip_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

struct clip_logger_state {
    ggml_log_callback log_callback;
    void * log_callback_user_data;
};

extern struct clip_logger_state g_logger_state;

static void clip_log_internal_v(enum ggml_log_level level, const char * format, va_list args) {
    if (format == NULL) {
        return;
    }
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = (char *) calloc(len + 1, sizeof(char));
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        free(buffer2);
    }
    va_end(args_copy);
}

static void clip_log_internal(enum ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    clip_log_internal_v(level, format, args);
    va_end(args);
}

#define LOG_INF(...) clip_log_internal(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) clip_log_internal(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) clip_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_DBG(...) clip_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_CNT(...) clip_log_internal(GGML_LOG_LEVEL_CONT,  __VA_ARGS__)

//
// cpp wrappers
//

// wrapper for clip_image_size
struct clip_image_size_deleter {
    void operator()(clip_image_size * val) { clip_image_size_free(val); }
};
typedef std::unique_ptr<clip_image_size, clip_image_size_deleter> clip_image_size_ptr;

// wrapper for clip_image_u8
struct clip_image_u8_deleter {
    void operator()(clip_image_u8 * val) { clip_image_u8_free(val); }
};
typedef std::unique_ptr<clip_image_u8, clip_image_u8_deleter> clip_image_u8_ptr;

// wrapper for clip_image_f32
struct clip_image_f32_deleter {
    void operator()(clip_image_f32 * val) { clip_image_f32_free(val); }
};
typedef std::unique_ptr<clip_image_f32, clip_image_f32_deleter> clip_image_f32_ptr;

struct clip_image_u8_batch {
    std::vector<clip_image_u8_ptr> entries;
};

struct clip_image_f32_batch {
    std::vector<clip_image_f32_ptr> entries;
    bool is_audio = false;

    // for llava-uhd style models, we need to know the grid size
    // note: entries.size() == grid_x * grid_y + 1 (one overview image)
    int grid_x = 0;
    int grid_y = 0;

    clip_image_f32_batch clone() const {
        clip_image_f32_batch new_batch{
            /* entries  */ {},
            /* is_audio */ is_audio,
            /* grid_x   */ grid_x,
            /* grid_y   */ grid_y,
        };
        new_batch.entries.reserve(entries.size());
        for (const auto & entry : entries) {
            new_batch.entries.emplace_back(new clip_image_f32(*entry));
        }
        return new_batch;
    }
};

//
// common utils
//

static std::string string_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

static void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

// split string by a `std::string delim` instead of `char delim`
static std::vector<std::string> string_split_str(std::string s, const std::string & delimiter) {
    std::vector<std::string> tokens;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);
    return tokens;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

// remove when moving to c++20
inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

//
// gguf utils
//

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const int8_t *)data)[i] != 0 ? "true" : "false";
        default:                return string_format("unknown type %d", type);
    }
}

static std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        string_replace_all(val, "\\", "\\\\");
                        string_replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

enum ffn_op_type {
    FFN_GELU,
    FFN_GELU_ERF,
    FFN_SILU,
    FFN_GELU_QUICK,
    FFN_RELU_SQR,
};

enum norm_type {
    NORM_TYPE_NORMAL,
    NORM_TYPE_RMS,
};

enum patch_merge_type {
    PATCH_MERGE_FLAT,
    PATCH_MERGE_SPATIAL_UNPAD,
};

enum resize_algo {
    RESIZE_ALGO_BILINEAR, // stretch to target resolution
    RESIZE_ALGO_BICUBIC, // center-crop when aspect ratio doesn't match
    RESIZE_ALGO_BICUBIC_PILLOW,
    // RESIZE_ALGO_LANCZOS, // TODO
};

struct clip_hparams {
    int32_t image_size = 0;
    int32_t patch_size = 0;
    int32_t n_embd = 0;
    int32_t n_ff = 0;
    int32_t projection_dim = 0;
    int32_t n_head = 0;
    int32_t n_layer = 0;
    // idefics3
    int32_t n_merge = 0; // number of patch merges **per-side**

    // for preprocessor
    int32_t image_longest_edge = 0;
    int32_t image_min_pixels = -1;
    int32_t image_max_pixels = -1;
    resize_algo image_resize_algo = RESIZE_ALGO_BICUBIC;
    bool image_resize_pad = true; // if false, center-crop will be applied when resizing
    std::array<uint8_t, 3> image_pad_color = {0, 0, 0};

    // (preprocessor) for llava-uhd style models
    std::vector<clip_image_size> image_res_candidates;
    int32_t preproc_min_tiles = 0;
    int32_t preproc_max_tiles = 0;
    resize_algo image_resize_algo_rf = RESIZE_ALGO_BICUBIC;
    resize_algo image_resize_algo_ov = RESIZE_ALGO_BILINEAR;
    bool image_pad_rf = true;  // if true, refined image will be padded (e.g. llava-1.6)
    bool image_pad_ov = false; // if true, overview image will be padded (e.g. llava-1.6)
    std::array<uint8_t, 3> image_pad_color_rf = {0, 0, 0}; // padding color for refined image
    std::array<uint8_t, 3> image_pad_color_ov = {0, 0, 0}; // padding color for overview image

    float image_mean[3];
    float image_std[3];

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM every time they load the model
    int32_t warmup_image_size = 0;
    int32_t warmup_audio_size = 3000;

    ffn_op_type ffn_op = FFN_GELU;

    patch_merge_type mm_patch_merge_type = PATCH_MERGE_FLAT;

    float eps = 1e-6;
    float rope_theta = 0.0;
    std::unordered_set<int32_t> vision_feature_layer;
    int32_t attn_window_size = 0;
    int32_t n_wa_pattern = 0;
    std::unordered_set<int32_t> wa_layer_indexes; // explicit layer indexes that use full attention (for irregular patterns like YoutuVL)

    // deepseek-ocr (sam)
    int32_t sam_n_layer = 0;
    int32_t sam_n_head  = 0;
    int32_t sam_n_embd  = 0;

    // audio
    int32_t n_mel_bins = 0; // whisper preprocessor
    int32_t proj_stack_factor = 0; // ultravox

    // audio-to-mel preprocessor params
    int32_t audio_chunk_len   = -1; // in seconds
    int32_t audio_sample_rate = -1;
    int32_t audio_n_fft       = -1;
    int32_t audio_window_len  = -1;
    int32_t audio_hop_len     = -1;

    // legacy
    bool has_llava_projector = false;
    int minicpmv_version = 0;
    int32_t minicpmv_query_num = 0;         // MiniCPM-V query number

    // custom value provided by user, can be undefined if not set
    int32_t custom_image_min_tokens = -1;
    int32_t custom_image_max_tokens = -1;

    void set_limit_image_tokens(int n_tokens_min, int n_tokens_max) {
        const int cur_merge = n_merge == 0 ? 1 : n_merge;
        const int patch_area = patch_size * patch_size * cur_merge * cur_merge;
        image_min_pixels = (custom_image_min_tokens > 0 ? custom_image_min_tokens : n_tokens_min) * patch_area;
        image_max_pixels = (custom_image_max_tokens > 0 ? custom_image_max_tokens : n_tokens_max) * patch_area;
        warmup_image_size = static_cast<int>(std::sqrt(image_max_pixels));
    }

    void set_warmup_n_tokens(int n_tokens) {
        int n_tok_per_side = static_cast<int>(std::sqrt(n_tokens));
        GGML_ASSERT(n_tok_per_side * n_tok_per_side == n_tokens && "n_tokens must be n*n");
        const int cur_merge = n_merge == 0 ? 1 : n_merge;
        warmup_image_size = n_tok_per_side * patch_size * cur_merge;
        // TODO: support warmup size for custom token numbers
    }
    // sam vit deepseek-ocr
    std::vector<int32_t> global_attn_indices() const {
        return {  2,  5,  8, 11 };
    }
    bool is_global_attn(int32_t layer) const {
        const auto indices = global_attn_indices();

        for (const auto & idx : indices) {
            if (layer == idx) {
                return true;
            }
        }

        return false;
    }
};

struct clip_layer {
    // layernorm 1 (or layer input norm, or pre-attention norm)
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;

    // attention
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * qkv_w = nullptr;
    ggml_tensor * qkv_b = nullptr;

    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;

    ggml_tensor * k_norm = nullptr;
    ggml_tensor * q_norm = nullptr;

    ggml_tensor * attn_post_norm_w = nullptr;

    ggml_tensor * ff_up_w = nullptr;
    ggml_tensor * ff_up_b = nullptr;
    ggml_tensor * ff_gate_w = nullptr;
    ggml_tensor * ff_gate_b = nullptr;
    ggml_tensor * ff_down_w = nullptr;
    ggml_tensor * ff_down_b = nullptr;

    // layernorm 2 (or pre-FFN norm)
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;

    ggml_tensor * ff_post_norm_w = nullptr;

    // layer scale (no bias)
    ggml_tensor * ls_1_w   = nullptr;
    ggml_tensor * ls_2_w   = nullptr;
    ggml_tensor * ls_out_w = nullptr; // gemma4

    // qwen3vl deepstack merger
    ggml_tensor * deepstack_norm_w = nullptr;
    ggml_tensor * deepstack_norm_b = nullptr;
    ggml_tensor * deepstack_fc1_w = nullptr;
    ggml_tensor * deepstack_fc1_b = nullptr;
    ggml_tensor * deepstack_fc2_w = nullptr;
    ggml_tensor * deepstack_fc2_b = nullptr;

    // sam rel_pos
    ggml_tensor * rel_pos_w = nullptr;
    ggml_tensor * rel_pos_h = nullptr;
    // lfm2
    ggml_tensor * ff_norm_w     = nullptr;
    ggml_tensor * ff_norm_b     = nullptr;
    ggml_tensor * ff_norm_1_w   = nullptr;
    ggml_tensor * ff_norm_1_b   = nullptr;
    ggml_tensor * ff_up_1_w     = nullptr;
    ggml_tensor * ff_up_1_b     = nullptr;
    ggml_tensor * ff_down_1_w   = nullptr;
    ggml_tensor * ff_down_1_b   = nullptr;
    ggml_tensor * pos_bias_u    = nullptr;
    ggml_tensor * pos_bias_v    = nullptr;
    ggml_tensor * norm_conv_w   = nullptr;
    ggml_tensor * norm_conv_b   = nullptr;
    ggml_tensor * linear_pos_w  = nullptr;

    ggml_tensor * conv_norm_w   = nullptr;
    ggml_tensor * conv_norm_b   = nullptr;
    ggml_tensor * conv_dw_w     = nullptr;
    ggml_tensor * conv_dw_b     = nullptr;
    ggml_tensor * conv_pw1_w    = nullptr;
    ggml_tensor * conv_pw1_b    = nullptr;
    ggml_tensor * conv_pw2_w    = nullptr;
    ggml_tensor * conv_pw2_b    = nullptr;

    // gemma4 audio conformer per-layer
    ggml_tensor * attn_pre_norm_w   = nullptr;
    ggml_tensor * attn_k_rel_w      = nullptr;
    ggml_tensor * per_dim_scale_w   = nullptr;
    ggml_tensor * per_dim_k_scale_w = nullptr;
    ggml_tensor * ff_post_norm_1_w  = nullptr;

    bool has_deepstack() const {
        return deepstack_fc1_w != nullptr;
    }
};

// Expanded MobileNetV5 block structure for Gemma3n vision encoder
struct mobilenetv5_block {
    // Stage 0 (Edge Residual)
    ggml_tensor * s0_conv_exp_w = nullptr;
    ggml_tensor * s0_bn1_w      = nullptr;
    ggml_tensor * s0_conv_pwl_w = nullptr;
    ggml_tensor * s0_bn2_w      = nullptr;

    // Stage 1+ (Universal Inverted Residual)
    ggml_tensor * dw_start_w    = nullptr;
    ggml_tensor * dw_start_bn_w = nullptr;

    ggml_tensor * pw_exp_w      = nullptr;
    ggml_tensor * pw_exp_bn_w   = nullptr;

    ggml_tensor * dw_mid_w      = nullptr;
    ggml_tensor * dw_mid_bn_w   = nullptr;

    ggml_tensor * pw_proj_w     = nullptr;
    ggml_tensor * pw_proj_bn_w  = nullptr;

    ggml_tensor * layer_scale_w = nullptr;

    // Attention (MQA) components
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_o_w = nullptr;

    // Optional downsampling/norm in attention
    ggml_tensor * attn_k_dw_w   = nullptr;
    ggml_tensor * attn_k_norm_w = nullptr;
    ggml_tensor * attn_v_dw_w   = nullptr;
    ggml_tensor * attn_v_norm_w = nullptr;

    // Block norm (often present in attention blocks)
    ggml_tensor * attn_norm_w   = nullptr;
};

struct yasa2_block {
    ggml_tensor * dw_w  = nullptr;
    ggml_tensor * dw_b  = nullptr;
    ggml_tensor * ln_w  = nullptr;
    ggml_tensor * ln_b  = nullptr;
    ggml_tensor * pw1_w = nullptr;
    ggml_tensor * pw1_b = nullptr;
    ggml_tensor * grn_w = nullptr;
    ggml_tensor * grn_b = nullptr;
    ggml_tensor * pw2_w = nullptr;
    ggml_tensor * pw2_b = nullptr;
};

struct yasa2_stage {
    ggml_tensor * down_ln_w   = nullptr;
    ggml_tensor * down_ln_b   = nullptr;
    ggml_tensor * down_conv_w = nullptr;
    ggml_tensor * down_conv_b = nullptr;
    std::vector<yasa2_block> blocks;
};

struct clip_model {
    clip_modality modality = CLIP_MODALITY_VISION;
    projector_type proj_type = PROJECTOR_TYPE_MLP;
    clip_hparams hparams;

    // embeddings
    ggml_tensor * class_embedding = nullptr;
    ggml_tensor * patch_embeddings_0 = nullptr;
    ggml_tensor * patch_embeddings_1 = nullptr;  // second Conv2D kernel when we decouple Conv3D along temporal dimension (Qwen2VL)
    ggml_tensor * patch_bias = nullptr;
    ggml_tensor * position_embeddings = nullptr;
    ggml_tensor * norm_embd_w = nullptr;
    ggml_tensor * norm_embd_b = nullptr;

    ggml_tensor * pre_ln_w = nullptr;
    ggml_tensor * pre_ln_b = nullptr;

    std::vector<clip_layer> layers;

    int32_t n_deepstack_layers = 0; // used by Qwen3-VL, calculated from clip_layer

    ggml_tensor * post_ln_w;
    ggml_tensor * post_ln_b;

    ggml_tensor * mm_fc_w;
    ggml_tensor * mm_fc_b;
    ggml_tensor * mm_ffn_up_w = nullptr;
    ggml_tensor * mm_ffn_up_b = nullptr;
    ggml_tensor * mm_ffn_gate_w = nullptr;
    ggml_tensor * mm_ffn_gate_b = nullptr;
    ggml_tensor * mm_ffn_down_w = nullptr;
    ggml_tensor * mm_ffn_down_b = nullptr;
    ggml_tensor * mm_post_norm_w = nullptr;
    ggml_tensor * mm_post_norm_b = nullptr;

    // LLaVA projection
    ggml_tensor * mm_input_norm_w = nullptr;
    ggml_tensor * mm_input_norm_b = nullptr;
    ggml_tensor * mm_0_w = nullptr;
    ggml_tensor * mm_0_b = nullptr;
    ggml_tensor * mm_2_w = nullptr;
    ggml_tensor * mm_2_b = nullptr;

    ggml_tensor * image_newline = nullptr;
    ggml_tensor * view_seperator = nullptr;


    // Yi type models with mlp+normalization projection
    ggml_tensor * mm_1_w = nullptr; // Yi type models have 0, 1, 3, 4
    ggml_tensor * mm_1_b = nullptr;
    ggml_tensor * mm_3_w = nullptr;
    ggml_tensor * mm_3_b = nullptr;
    ggml_tensor * mm_4_w = nullptr;
    ggml_tensor * mm_4_b = nullptr;

    // GLMV-Edge projection
    ggml_tensor * mm_model_adapter_conv_w = nullptr;
    ggml_tensor * mm_model_adapter_conv_b = nullptr;

    // MobileVLM projection
    ggml_tensor * mm_model_mlp_1_w = nullptr;
    ggml_tensor * mm_model_mlp_1_b = nullptr;
    ggml_tensor * mm_model_mlp_3_w = nullptr;
    ggml_tensor * mm_model_mlp_3_b = nullptr;
    ggml_tensor * mm_model_block_1_block_0_0_w = nullptr;
    ggml_tensor * mm_model_block_1_block_0_1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_0_1_b = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc1_b = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc2_w = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc2_b = nullptr;
    ggml_tensor * mm_model_block_1_block_2_0_w = nullptr;
    ggml_tensor * mm_model_block_1_block_2_1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_2_1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_0_0_w = nullptr;
    ggml_tensor * mm_model_block_2_block_0_1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_0_1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc2_w = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc2_b = nullptr;
    ggml_tensor * mm_model_block_2_block_2_0_w = nullptr;
    ggml_tensor * mm_model_block_2_block_2_1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_2_1_b = nullptr;

    // MobileVLM_V2 projection
    ggml_tensor * mm_model_mlp_0_w = nullptr;
    ggml_tensor * mm_model_mlp_0_b = nullptr;
    ggml_tensor * mm_model_mlp_2_w = nullptr;
    ggml_tensor * mm_model_mlp_2_b = nullptr;
    ggml_tensor * mm_model_peg_0_w = nullptr;
    ggml_tensor * mm_model_peg_0_b = nullptr;

    // MINICPMV projection
    ggml_tensor * mm_model_pos_embed_k = nullptr;
    ggml_tensor * mm_model_query = nullptr;
    ggml_tensor * mm_model_proj   = nullptr;
    ggml_tensor * mm_model_proj_b = nullptr;
    ggml_tensor * mm_model_kv_proj = nullptr;
    ggml_tensor * mm_model_attn_q_w = nullptr;
    ggml_tensor * mm_model_attn_q_b = nullptr;
    ggml_tensor * mm_model_attn_k_w = nullptr;
    ggml_tensor * mm_model_attn_k_b = nullptr;
    ggml_tensor * mm_model_attn_v_w = nullptr;
    ggml_tensor * mm_model_attn_v_b = nullptr;
    ggml_tensor * mm_model_attn_o_w = nullptr;
    ggml_tensor * mm_model_attn_o_b = nullptr;
    ggml_tensor * mm_model_ln_q_w = nullptr;
    ggml_tensor * mm_model_ln_q_b = nullptr;
    ggml_tensor * mm_model_ln_kv_w = nullptr;
    ggml_tensor * mm_model_ln_kv_b = nullptr;
    ggml_tensor * mm_model_ln_post_w = nullptr;
    ggml_tensor * mm_model_ln_post_b = nullptr;

    // gemma3
    ggml_tensor * mm_input_proj_w = nullptr;
    ggml_tensor * mm_soft_emb_norm_w = nullptr;

    // mobilenetv5 for gemma3n
    std::vector<mobilenetv5_block> mobilenet_blocks;
    std::vector<int> mobilenet_stage_ends;
    ggml_tensor * mobilenet_stem_conv_w = nullptr;
    ggml_tensor * mobilenet_stem_conv_b = nullptr;
    ggml_tensor * mobilenet_stem_norm_w = nullptr;
    ggml_tensor * mm_post_proj_norm_w = nullptr;

    // Multi-Scale Fusion Adapter (MSFA) components
    ggml_tensor * msfa_concat_conv_w = nullptr;
    ggml_tensor * msfa_concat_norm_w = nullptr;
    ggml_tensor * msfa_ffn_expand_w = nullptr;
    ggml_tensor * msfa_ffn_project_w = nullptr;
    ggml_tensor * msfa_ffn_expand_bn = nullptr;
    ggml_tensor * msfa_ffn_project_bn = nullptr;

    // yasa2
    ggml_tensor * yasa_patch_w = nullptr;
    ggml_tensor * yasa_patch_b = nullptr;
    ggml_tensor * yasa_patch_ln_w = nullptr;
    ggml_tensor * yasa_patch_ln_b = nullptr;
    ggml_tensor * yasa_backbone_ln_w = nullptr;
    ggml_tensor * yasa_backbone_ln_b = nullptr;
    ggml_tensor * yasa_vision_pos_embed = nullptr;
    std::vector<yasa2_stage> yasa_stages;

    // pixtral, glm4v
    ggml_tensor * token_embd_img_break = nullptr;
    ggml_tensor * mm_patch_merger_w = nullptr;
    ggml_tensor * mm_patch_merger_b = nullptr;

    // ultravox / whisper encoder
    ggml_tensor * conv1d_1_w = nullptr;
    ggml_tensor * conv1d_1_b = nullptr;
    ggml_tensor * conv1d_2_w = nullptr;
    ggml_tensor * conv1d_2_b = nullptr;
    ggml_tensor * conv_out_w = nullptr;
    ggml_tensor * conv_out_b = nullptr;
    ggml_tensor * mm_norm_pre_w = nullptr;
    ggml_tensor * mm_norm_pre_b = nullptr;
    ggml_tensor * mm_norm_mid_w = nullptr;

    // qwen3a
    ggml_tensor * conv2d_1_w = nullptr;
    ggml_tensor * conv2d_1_b = nullptr;
    ggml_tensor * conv2d_2_w = nullptr;
    ggml_tensor * conv2d_2_b = nullptr;
    ggml_tensor * conv2d_3_w = nullptr;
    ggml_tensor * conv2d_3_b = nullptr;

    // cogvlm
    ggml_tensor * mm_post_fc_norm_w = nullptr;
    ggml_tensor * mm_post_fc_norm_b = nullptr;
    ggml_tensor * mm_h_to_4h_w = nullptr;
    ggml_tensor * mm_gate_w = nullptr;
    ggml_tensor * mm_4h_to_h_w = nullptr;
    ggml_tensor * mm_boi = nullptr;
    ggml_tensor * mm_eoi = nullptr;

    // hunyuanocr perceiver
    ggml_tensor * mm_pre_norm_w  = nullptr;
    ggml_tensor * mm_img_begin   = nullptr;
    ggml_tensor * mm_img_end     = nullptr;

    // deepseek ocr sam
    ggml_tensor * patch_embed_proj_w = nullptr;
    ggml_tensor * patch_embed_proj_b = nullptr;
    ggml_tensor * pos_embed          = nullptr;

    ggml_tensor * neck_0_w;
    ggml_tensor * neck_1_w;
    ggml_tensor * neck_1_b;
    ggml_tensor * neck_2_w;
    ggml_tensor * neck_3_w;
    ggml_tensor * neck_3_b;
    ggml_tensor * net_2;
    ggml_tensor * net_3;

    int32_t n_sam_layers = 12; // used by deepseek-ocr sam encoder

    std::vector<clip_layer> sam_layers;
    // lfm2 audio
    std::array<ggml_tensor *, 7> pre_encode_conv_X_w = {nullptr};
    std::array<ggml_tensor *, 7> pre_encode_conv_X_b = {nullptr};
    ggml_tensor * pre_encode_out_w = nullptr;
    ggml_tensor * pre_encode_out_b = nullptr;

    // gemma4
    ggml_tensor * std_bias = nullptr;
    ggml_tensor * std_scale = nullptr;
    // Gemma4ClippableLinear
    struct clamp_info {
        float inp_max;
        float inp_min;
        float out_max;
        float out_min;
    };
    std::map<std::string, clamp_info> clamp_info_map;

    // gemma4 audio conformer
    std::array<ggml_tensor *, 2> sscp_conv_w = {nullptr};
    std::array<ggml_tensor *, 2> sscp_conv_b = {nullptr};
    std::array<ggml_tensor *, 2> sscp_norm_w = {nullptr};
    ggml_tensor * sscp_inp_proj_w = nullptr;
    ggml_tensor * sscp_inp_proj_b = nullptr;
    ggml_tensor * audio_out_proj_w = nullptr;
    ggml_tensor * audio_out_proj_b = nullptr;

    bool audio_has_avgpool() const {
        return proj_type == PROJECTOR_TYPE_QWEN2A
            || proj_type == PROJECTOR_TYPE_VOXTRAL
            || proj_type == PROJECTOR_TYPE_MUSIC_FLAMINGO;
    }

    bool audio_has_stack_frames() const {
        return proj_type == PROJECTOR_TYPE_ULTRAVOX
            || proj_type == PROJECTOR_TYPE_VOXTRAL
            || proj_type == PROJECTOR_TYPE_MERALION;
    }
};

const clip_hparams * clip_get_hparams(const struct clip_ctx * ctx);

//
// debugging
//

static void print_tensor_shape(ggml_tensor * t) {
    printf("%s.shape = [", t->name);
    for (int i = 0; i < ggml_n_dims(t); ++i) {
        printf("%" PRId64, t->ne[i]);
        if (i < ggml_n_dims(t) - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

static void print_tensor_data(ggml_tensor * t, uint8_t * data, int64_t n) {
    ggml_type type = t->type;
    int64_t * ne = t->ne;
    size_t * nb = t->nb;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        printf("%s.data: [\n", t->name);
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                printf("     ..., \n");
                i2 = ne[2] - n;
            }
            printf("     [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    printf("      ..., \n");
                    i1 = ne[1] - n;
                }
                printf("      [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        printf("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    printf("%8.4f", v);
                    if (i0 < ne[0] - 1) printf(", ");
                }
                printf("],\n");
            }
            printf("     ],\n");
        }
        printf("    ]\n");
    }
}

//
// API used internally with mtmd
//

projector_type clip_get_projector_type(const struct clip_ctx * ctx);
void clip_set_debug_output_embeddings(struct clip_ctx * ctx, bool debug);
#endif // MTMD_INTERNAL_HEADER

#endif

#endif
