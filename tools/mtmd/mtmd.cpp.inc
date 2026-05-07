#define MTMD_INTERNAL_HEADER
#include "mtmd.h.inc"

#include "llama.h.inc"

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

// base class, models must inherit from this class
struct mtmd_image_preprocessor {
    const clip_hparams & hparams;

    mtmd_image_preprocessor(const clip_ctx * ctx): hparams(*clip_get_hparams(ctx)) {}

    virtual ~mtmd_image_preprocessor() = default;
    virtual bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) = 0;

    void img_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst, const float mean[3], const float std[3]);
    void img_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst);
};

struct mtmd_image_preprocessor_llava_uhd : mtmd_image_preprocessor {
    mtmd_image_preprocessor_llava_uhd(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;

    struct slice_coordinates {
        int x;
        int y;
        clip_image_size size;
    };

    struct slice_instructions {
        clip_image_size overview_size;
        clip_image_size refined_size;
        clip_image_size grid_size;
        std::vector<slice_coordinates> slices;
    };

    virtual slice_instructions get_slice_instructions(const clip_image_size & original_size);

    std::vector<clip_image_u8_ptr> slice_image(const clip_image_u8 & img, const slice_instructions & inst, bool overview_first = true);

private:
    clip_image_size get_best_resize(const clip_image_size & original_size, int scale_resolution, int patch_size, bool allow_upscale = false);
    clip_image_size resize_maintain_aspect_ratio(const clip_image_size & orig, const clip_image_size & target_max);
    clip_image_size select_best_resolution(const clip_image_size & original_size, const std::vector<clip_image_size> & possible_resolutions);
    int ensure_divide(int length, int patch_size);
    clip_image_size get_refine_size(const clip_image_size & original_size, const clip_image_size & grid, int scale_resolution, int patch_size, bool allow_upscale = false);
    clip_image_size get_best_grid(const int max_slice_nums, const int multiple, const float log_ratio);
};

struct mtmd_image_preprocessor_fixed_size : mtmd_image_preprocessor {
    mtmd_image_preprocessor_fixed_size(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

struct mtmd_image_preprocessor_dyn_size : mtmd_image_preprocessor {
    mtmd_image_preprocessor_dyn_size(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

struct mtmd_image_preprocessor_longest_edge : mtmd_image_preprocessor {
    mtmd_image_preprocessor_longest_edge(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

struct mtmd_image_preprocessor_lfm2 : mtmd_image_preprocessor_llava_uhd {
    static constexpr int   min_tiles            = 2;
    static constexpr int   max_tiles            = 10;
    static constexpr float max_pixels_tolerance = 2.0f;
    static constexpr int   tile_size            = 512;

    using mtmd_image_preprocessor_llava_uhd::mtmd_image_preprocessor_llava_uhd;
    slice_instructions get_slice_instructions(const clip_image_size & original_size) override;

private:
    clip_image_size find_closest_aspect_ratio(
            float aspect_ratio,
            const std::vector<clip_image_size> & target_ratios,
            int width, int height);
    std::vector<clip_image_size> get_target_ratios();
    clip_image_size get_grid_layout(int height, int width);
};

struct mtmd_image_preprocessor_idefics3 : mtmd_image_preprocessor_llava_uhd {
    mtmd_image_preprocessor_idefics3(const clip_ctx * ctx) : mtmd_image_preprocessor_llava_uhd(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

struct mtmd_image_preprocessor_internvl : mtmd_image_preprocessor_llava_uhd {
    mtmd_image_preprocessor_internvl(const clip_ctx * ctx) : mtmd_image_preprocessor_llava_uhd(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

struct mtmd_image_preprocessor_deepseekocr : mtmd_image_preprocessor {
    mtmd_image_preprocessor_deepseekocr(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

struct mtmd_image_preprocessor_step3vl : mtmd_image_preprocessor_llava_uhd {
    mtmd_image_preprocessor_step3vl(const clip_ctx * ctx) : mtmd_image_preprocessor_llava_uhd(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
    static slice_instructions build_slice_instructions(const clip_hparams & params, const clip_image_size & prepared_size);

private:
    static constexpr int   default_image_longest_edge = 3024;
    static constexpr int   default_image_crop_size    = 504;
    static constexpr float small_aspect_ratio_limit   = 1.5f;
    static constexpr float wide_aspect_ratio_limit    = 4.0f;
    static constexpr float crop_rounding_threshold    = 0.2f;

    void img_u8_resize_bilinear_to_f32(
            const clip_image_u8 & src,
            clip_image_f32 & dst,
            int target_width,
            int target_height,
            const float mean[3],
            const float std[3]);
    static int get_image_longest_edge(const clip_hparams & params);
    static int determine_window_size(const clip_hparams & params, int longer, int shorter);
    static int calc_crop_extent(int length, int window_size);
    static std::vector<int> calc_grid(int length, int window_size);
    static clip_image_u8 prepare_image(const clip_image_u8 & img, const clip_hparams & params);
    static clip_image_u8 crop_with_black_padding(const clip_image_u8 & image, int x, int y, int w, int h);
};

struct mtmd_image_preprocessor_youtuvl : mtmd_image_preprocessor {
    mtmd_image_preprocessor_youtuvl(const clip_ctx * ctx) : mtmd_image_preprocessor(ctx) {}
    bool preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) override;
};

struct mtmd_audio_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct mtmd_audio_mel_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

// cache for audio processing, each processor instance owns its own cache
struct mtmd_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;

    std::vector<float> hann_window;

    mtmd_audio_mel_filters filters;

    void fill_sin_cos_table(uint32_t n);

    void fill_hann_window(uint32_t length, bool periodic);

    // Build mel filterbank matrix [n_mel x n_fft_bins] at runtime.
    // n_fft_bins must be (N_fft / 2 + 1). Example: if N_fft=512 -> n_fft_bins=257.
    void fill_mel_filterbank_matrix(int   n_mel,
                                    int   n_fft,
                                    int   sample_rate,
                                    float fmin             = 0.0f,
                                    float fmax             = -1.0f,
                                    bool  slaney_area_norm = true,
                                    float scale            = 1.0f,
                                    bool  use_htk          = false
    );
};

struct mtmd_audio_preprocessor {
    const clip_hparams & hparams;

    mtmd_audio_preprocessor(const clip_ctx * ctx): hparams(*clip_get_hparams(ctx)) {}

    virtual ~mtmd_audio_preprocessor() = default;
    virtual void initialize() = 0; // NOT thread-safe
    virtual bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) = 0;
};

struct mtmd_audio_preprocessor_whisper : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_whisper(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_conformer : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_conformer(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_preprocessor_gemma4a : mtmd_audio_preprocessor {
    mtmd_audio_preprocessor_gemma4a(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    void initialize() override;
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;

  private:
    mtmd_audio_cache cache;
};

struct mtmd_audio_streaming_istft {
    mtmd_audio_streaming_istft(int n_fft, int hop_length);

    void reset();
    std::vector<float> process_frame(const float * frame_spectrum);
    std::vector<float> flush();

  private:
    int n_fft;
    int hop_length;
    int n_fft_bins;

    mtmd_audio_cache cache;

    std::vector<float> overlap_buffer;
    std::vector<float> window_sum_buffer;
    int                padding_to_remove;

    std::vector<float> ifft_in;
    std::vector<float> ifft_out;
};

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void mtmd_debug_encode_image(mtmd_context * ctx, const std::vector<std::vector<float>> & image);
void mtmd_debug_encode_audio(mtmd_context * ctx, const std::vector<float> & input);
void mtmd_debug_preprocess_image(mtmd_context * ctx, const std::vector<uint8_t> & rgb_values, int nx, int ny);
void mtmd_debug_preprocess_audio(mtmd_context * ctx, const std::vector<float> & pcm_samples);

// represents raw image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3
struct mtmd_bitmap {
    uint32_t nx;
    uint32_t ny;
    std::vector<unsigned char> data;
    std::string id; // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
    bool is_audio = false; // true if the bitmap is audio
};

// position indexing for decoder model
enum mtmd_pos_type {
    MTMD_POS_TYPE_NORMAL,    // number of positions equals to number of tokens
    MTMD_POS_TYPE_MROPE,     // qwen-vl mrope style, each image takes max(t,h,w) position indexes
    MTMD_POS_TYPE_HUNYUANVL, // HunyuanVL mrope + BOI/EOI/newline layout with XD-RoPE dim-3
};

struct mtmd_image_tokens {
    uint32_t nx; // number of tokens in x direction
    uint32_t ny; // number of tokens in y direction
    mtmd_pos_type pos = MTMD_POS_TYPE_NORMAL;
    uint32_t image_idx = 0; // 0-based position of this image among image chunks in the prompt(used by pos == MTMD_POS_TYPE_HUNYUANVL)
    uint32_t n_tokens() const {
        if (pos == MTMD_POS_TYPE_HUNYUANVL) {
            // [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
            return (nx + 1) * ny + 2;
        }
        return nx * ny;
    }
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    mtmd_image_tokens clone() {
        return mtmd_image_tokens{
            nx,
            ny,
            pos,
            image_idx,
            batch_f32.clone(),
            id
        };
    }
};
using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens>;

struct mtmd_audio_tokens {
    uint32_t n_tokens; // number of tokens
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    mtmd_audio_tokens clone() {
        return mtmd_audio_tokens{
            n_tokens,
            batch_f32.clone(),
            id
        };
    }
};
using mtmd_audio_tokens_ptr = std::unique_ptr<mtmd_audio_tokens>;

struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens_text;
    mtmd_image_tokens_ptr tokens_image;
    mtmd_audio_tokens_ptr tokens_audio;
};

struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
};

// slice template, used by some llava-uhd models to correctly place the special tokens around image embeddings
// models not having it (llava-1.6) will process embeddings without any special tokens in-between
enum mtmd_slice_tmpl {
    MTMD_SLICE_TMPL_NONE,
    MTMD_SLICE_TMPL_MINICPMV_2_5,
    MTMD_SLICE_TMPL_MINICPMV_2_6,
    MTMD_SLICE_TMPL_LLAMA4,
    MTMD_SLICE_TMPL_IDEFICS3,
    MTMD_SLICE_TMPL_LFM2,
    MTMD_SLICE_TMPL_STEP3VL,
};

const char * mtmd_default_marker() {
    return "<__media__>";
}

static clip_flash_attn_type mtmd_get_clip_flash_attn_type(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:     return CLIP_FLASH_ATTN_TYPE_AUTO;
        case LLAMA_FLASH_ATTN_TYPE_DISABLED: return CLIP_FLASH_ATTN_TYPE_DISABLED;
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:  return CLIP_FLASH_ATTN_TYPE_ENABLED;
    }
    return CLIP_FLASH_ATTN_TYPE_AUTO;
}

mtmd_context_params mtmd_context_params_default() {
    mtmd_context_params params {
        /* use_gpu           */ true,
        /* print_timings     */ true,
        /* n_threads         */ 4,
        /* image_marker      */ nullptr,
        /* media_marker      */ mtmd_default_marker(),
        /* flash_attn_type   */ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /* warmup            */ true,
        /* image_min_tokens  */ -1,
        /* image_max_tokens  */ -1,
        /* cb_eval           */ nullptr,
        /* cb_eval_user_data */ nullptr,
    };
    return params;
}

struct mtmd_context {
    struct clip_ctx * ctx_v; // vision
    struct clip_ctx * ctx_a; // audio
    const struct llama_model * text_model;
    std::vector<float> image_embd_v; // image embedding vector

    bool print_timings;
    int n_threads;
    std::string media_marker;
    const int n_embd_text;
    mtmd_pos_type pos_type;

    // these are not token, but strings used to mark the beginning and end of image/audio embeddings
    std::string img_beg;
    std::string img_end;
    std::string aud_beg;
    std::string aud_end;

    // for llava-uhd style models, we need special tokens in-between slices
    // minicpmv calls them "slices", llama 4 calls them "tiles"
    mtmd_slice_tmpl slice_tmpl = MTMD_SLICE_TMPL_NONE;
    std::vector<llama_token> tok_ov_img_start;  // overview image
    std::vector<llama_token> tok_ov_img_end;    // overview image
    std::vector<llama_token> tok_slices_start;  // start of all slices
    std::vector<llama_token> tok_slices_end;    // end of all slices
    std::vector<llama_token> tok_sli_img_start; // single slice start
    std::vector<llama_token> tok_sli_img_end;   // single slice end
    std::vector<llama_token> tok_sli_img_mid;   // between 2 slices
    std::vector<llama_token> tok_row_end;       // end of row
    bool tok_row_end_trail = false;
    bool ov_img_first      = false;

    // string template for slice image delimiters with row/col (idefics3)
    std::string sli_img_start_tmpl;

    std::unique_ptr<mtmd_audio_preprocessor> audio_preproc;
    std::unique_ptr<mtmd_image_preprocessor> image_preproc;

    // TODO @ngxson : add timings

    mtmd_context(const char * mmproj_fname,
                   const llama_model * text_model,
                   const mtmd_context_params & ctx_params) :
        text_model   (text_model),
        print_timings(ctx_params.print_timings),
        n_threads    (ctx_params.n_threads),
        media_marker (ctx_params.media_marker),
        n_embd_text  (llama_model_n_embd_inp(text_model))
    {
        if (ctx_params.image_marker != nullptr) {
            throw std::runtime_error("custom image_marker is not supported anymore, use media_marker instead");
        }

        if (media_marker.empty()) {
            throw std::runtime_error("media_marker must not be empty");
        }

        auto decoder_rope_type = llama_model_rope_type(text_model);
        switch (decoder_rope_type) {
            case LLAMA_ROPE_TYPE_NONE:
            case LLAMA_ROPE_TYPE_NORM:
            case LLAMA_ROPE_TYPE_NEOX:
                {
                    pos_type = MTMD_POS_TYPE_NORMAL;
                } break;
            case LLAMA_ROPE_TYPE_MROPE:
            case LLAMA_ROPE_TYPE_IMROPE:
                {
                    pos_type = MTMD_POS_TYPE_MROPE;
                } break;
            default:
                throw std::runtime_error(string_format("unsupported decoder rope type: %d\n", decoder_rope_type));
        }

        clip_context_params ctx_clip_params {
            /* use_gpu           */ ctx_params.use_gpu,
            /* flash_attn_type   */ mtmd_get_clip_flash_attn_type(ctx_params.flash_attn_type),
            /* image_min_tokens  */ ctx_params.image_min_tokens,
            /* image_max_tokens  */ ctx_params.image_max_tokens,
            /* warmup            */ ctx_params.warmup,
            /* cb_eval           */ ctx_params.cb_eval,
            /* cb_eval_user_data */ ctx_params.cb_eval_user_data,
        };

        auto res = clip_init(mmproj_fname, ctx_clip_params);
        ctx_v = res.ctx_v;
        ctx_a = res.ctx_a;
        if (!ctx_v && !ctx_a) {
            throw std::runtime_error(string_format("Failed to load CLIP model from %s\n", mmproj_fname));
        }

        // if both vision and audio mmproj are present, we need to validate their n_embd
        if (ctx_v && ctx_a) {
            int n_embd_v = clip_n_mmproj_embd(ctx_v);
            int n_embd_a = clip_n_mmproj_embd(ctx_a);
            if (n_embd_v != n_embd_a) {
                throw std::runtime_error(string_format(
                    "mismatch between vision and audio mmproj (n_embd_v = %d, n_embd_a = %d)\n",
                    n_embd_v, n_embd_a));
            }
        }

        // since we already validate n_embd of vision and audio mmproj,
        // we can safely assume that they are the same
        int n_embd_clip = clip_n_mmproj_embd(ctx_v ? ctx_v : ctx_a);
        if (n_embd_text != n_embd_clip) {
            throw std::runtime_error(string_format(
                "mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n"
                "hint: you may be using wrong mmproj\n",
                n_embd_text, n_embd_clip));
        }
        if (ctx_v) {
            init_vision();
        }
        if (ctx_a) {
            init_audio();
        }
    }

    void init_vision() {
        GGML_ASSERT(ctx_v != nullptr);
        image_preproc.reset();

        projector_type proj = clip_get_projector_type(ctx_v);

        switch (proj) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
            case PROJECTOR_TYPE_LDP:
            case PROJECTOR_TYPE_LDPV2:
            case PROJECTOR_TYPE_COGVLM:
            case PROJECTOR_TYPE_JANUS_PRO:
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    bool has_pinpoints = !clip_get_hparams(ctx_v)->image_res_candidates.empty();
                    if (has_pinpoints) {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                    } else {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                    }
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    int minicpmv_version = clip_is_minicpmv(ctx_v);
                    if (minicpmv_version == 2) {
                        // minicpmv 2.5 format:
                        // <image> (overview) </image><slice><image> (slice) </image><image> (slice) </image>\n ... </slice>
                        slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_5;
                        tok_ov_img_start  = {lookup_token("<image>")};
                        tok_ov_img_end    = {lookup_token("</image>")};
                        tok_slices_start  = {lookup_token("<slice>")};
                        tok_slices_end    = {lookup_token("</slice>")};
                        tok_sli_img_start = tok_ov_img_start;
                        tok_sli_img_end   = tok_ov_img_end;
                        tok_row_end       = {lookup_token("\n")};
                        tok_row_end_trail = false; // no trailing end-of-row token
                        ov_img_first      = true;
                    } else if (minicpmv_version == 3 || minicpmv_version == 4 || minicpmv_version == 5 || minicpmv_version == 6 || minicpmv_version == 100045) {
                        // minicpmv 2.6 format:
                        // <image> (overview) </image><slice> (slice) </slice><slice> (slice) </slice>\n ...
                        slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_6;
                        tok_ov_img_start  = {lookup_token("<image>")};
                        tok_ov_img_end    = {lookup_token("</image>")};
                        tok_sli_img_start = {lookup_token("<slice>")};
                        tok_sli_img_end   = {lookup_token("</slice>")};
                        tok_row_end       = {lookup_token("\n")};
                        tok_row_end_trail = false; // no trailing end-of-row token
                        ov_img_first      = true;

                    } else if (minicpmv_version != 0) {
                        throw std::runtime_error(string_format("unsupported minicpmv version: %d\n", minicpmv_version));
                    }
                    image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
            case PROJECTOR_TYPE_QWEN3VL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_YOUTUVL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_youtuvl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_YASA2:
                {
                    img_beg = "<image>";
                    img_end = "</image>";
                    // Currently only supprots single-tile preprocessing: any input is downscaled
                    // to one image_size x image_size tile (64 output tokens via 8x8 adaptive avg
                    // pool).
                    // However, the model itself supports llava-uhd multi-tile tiling for high-res
                    // images. This will be implemented in a future PR (dispatch on has_pinpoints
                    // - see LDP/COGVLM branch above) and emit image_grid_pinpoints in the conversion
                    // script.
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GEMMA3:
            case PROJECTOR_TYPE_GEMMA3NV:
                {
                    // <start_of_image> ... (image embeddings) ... <end_of_image>
                    img_beg = "<start_of_image>";
                    img_end = "<end_of_image>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    // https://github.com/huggingface/transformers/blob/a42ba80fa520c784c8f11a973ca9034e5f859b79/src/transformers/models/idefics3/processing_idefics3.py#L192-L215
                    slice_tmpl         = MTMD_SLICE_TMPL_IDEFICS3;
                    tok_ov_img_start   = {lookup_token("\n\n"), lookup_token("<fake_token_around_image>"), lookup_token("<global-img>")};
                    tok_ov_img_end     = {lookup_token("<fake_token_around_image>")};
                    tok_row_end        = {lookup_token("\n")};
                    sli_img_start_tmpl = "<fake_token_around_image><row_%d_col_%d>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_idefics3>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    // https://github.com/huggingface/transformers/blob/1cd110c6cb6a6237614130c470e9a902dbc1a4bd/docs/source/en/model_doc/pixtral.md
                    img_end = "[IMG_END]";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PHI4:
                {
                    // Phi-4 uses media marker insertion only. Keep image boundary text empty.
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    // (more details in mtmd_context constructor)
                    img_beg = "<|image_start|>";
                    img_end = "<|image_end|>";
                    LOG_WRN("%s: llama 4 vision is known to have degraded quality:\n"
                            "    https://github.com/ggml-org/llama.cpp/pull/13282\n", __func__);
                    image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                } break;
            case PROJECTOR_TYPE_STEP3VL:
                {
                    // Step3 format:
                    //   <patch_start> (patch) <patch_end> [<patch_newline>]
                    //   ... (all patch rows)
                    //   <im_start> (overview) <im_end>
                    slice_tmpl        = MTMD_SLICE_TMPL_STEP3VL;
                    tok_ov_img_start  = {lookup_token("<im_start>")};
                    tok_ov_img_end    = {lookup_token("<im_end>")};
                    tok_sli_img_start = {lookup_token("<patch_start>")};
                    tok_sli_img_end   = {lookup_token("<patch_end>")};
                    tok_row_end       = {lookup_token("<patch_newline>")};
                    tok_row_end_trail = false;
                    ov_img_first      = false; // patches first, overview last
                    image_preproc = std::make_unique<mtmd_image_preprocessor_step3vl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    // <img> ... (image embeddings) ... </img>
                    img_beg = "<img>";
                    img_end = "</img>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_internvl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_KIMIVL:
                {
                    // <|media_start|> ... (image embeddings) ... <|media_end|>
                    img_beg = "<|media_start|>";
                    img_end = "<|media_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_KIMIK25:
                {
                    // <|media_begin|> ... (image embeddings) ... <|media_end|>
                    img_beg = "<|media_begin|>";
                    img_end = "<|media_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LIGHTONOCR:
                {
                    // <|im_start|> ... (image embeddings) ... <|im_end|>
                    img_beg = "<|im_start|>";
                    img_end = "<|im_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_longest_edge>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DOTS_OCR:
                {
                    // <|img|> ... (image embeddings) ... <|endofimg|>
                    img_beg = "<|img|>";
                    img_end = "<|endofimg|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                {
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LFM2:
                {
                    // multi-tile:
                    //   <|image_start|>
                    //     <|img_row_1_col_1|> (tile) <|img_row_1_col_2|> (tile) ...
                    //     <|img_thumbnail|> (thumbnail)
                    //   <|image_end|>
                    // single-tile:
                    //   <|image_start|> (image) <|image_end|>
                    img_beg            = "<|image_start|>";
                    img_end            = "<|image_end|>";
                    slice_tmpl         = MTMD_SLICE_TMPL_LFM2;
                    sli_img_start_tmpl = "<|img_row_%d_col_%d|>";
                    tok_ov_img_start   = {lookup_token("<|img_thumbnail|>")};
                    ov_img_first       = false;
                    image_preproc = std::make_unique<mtmd_image_preprocessor_lfm2>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GLM4V:
                {
                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                    img_beg = "<|begin_of_image|>";
                    img_end = "<|end_of_image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PADDLEOCR:
                {
                    // <|IMAGE_START|> ... (image embeddings) ... <|IMAGE_END|>
                    img_beg = "<|IMAGE_START|>";
                    img_end = "<|IMAGE_END|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GEMMA4V:
                {
                    // <|image> ... (image embeddings) ... <image|>
                    img_beg = "<|image>";
                    img_end = "<image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DEEPSEEKOCR:
                {
                    img_end = "\n"; // prevent empty batch on llama-server
                    image_preproc = std::make_unique<mtmd_image_preprocessor_deepseekocr>(ctx_v);
                } break;
            case PROJECTOR_TYPE_HUNYUANOCR:
            case PROJECTOR_TYPE_HUNYUANVL:
                {
                    // note: these use fullwidth ｜ (U+FF5C) and ▁ (U+2581) to match the tokenizer vocabulary
                    img_beg = "<｜hy_place▁holder▁no▁100｜>";
                    img_end = "<｜hy_place▁holder▁no▁101｜>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            default:
                throw std::runtime_error(string_format("%s: unexpected vision projector type %d\n", __func__, proj));
        }

        GGML_ASSERT(image_preproc != nullptr);
    }

    void init_audio() {
        GGML_ASSERT(ctx_a != nullptr);
        audio_preproc.reset();

        projector_type proj = clip_get_projector_type(ctx_a);

        LOG_WRN("%s: audio input is in experimental stage and may have reduced quality:\n"
                "    https://github.com/ggml-org/llama.cpp/discussions/13759\n", __func__);

        // set preprocessor
        switch (proj) {
            case PROJECTOR_TYPE_QWEN2A:
            case PROJECTOR_TYPE_QWEN3A:
            case PROJECTOR_TYPE_QWEN25O:
                {
                    // <|audio_bos|> ... (embeddings) ... <|audio_eos|>
                    aud_beg = "<|audio_bos|>";
                    aud_end = "<|audio_eos|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
                {
                    // [BEGIN_AUDIO] ... (embeddings) ...
                    aud_beg = "[BEGIN_AUDIO]";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                {
                    // <sound> ... (embeddings) ...
                    aud_beg = "<sound>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
            case PROJECTOR_TYPE_GLMA:
            case PROJECTOR_TYPE_MERALION:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_LFM2A:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_conformer>(ctx_a);
                } break;
            case PROJECTOR_TYPE_GEMMA4A:
                {
                    aud_beg = "<|audio>";
                    aud_end = "<audio|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_gemma4a>(ctx_a);
                } break;
            default:
                throw std::runtime_error(string_format("%s: unexpected audio projector type %d\n", __func__, proj));
        }

        // initialize audio preprocessor
        GGML_ASSERT(audio_preproc != nullptr);
        audio_preproc->initialize();
    }

    // get clip ctx based on chunk type
    clip_ctx * get_clip_ctx(const mtmd_input_chunk * chunk) const {
        if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            return ctx_v;
        } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            return ctx_a;
        }
        GGML_ABORT("unknown chunk type");
    }

    projector_type proj_type_v() const {
        return ctx_v ? clip_get_projector_type(ctx_v) : PROJECTOR_TYPE_UNKNOWN;
    }

    projector_type proj_type_a() const {
        return ctx_a ? clip_get_projector_type(ctx_a) : PROJECTOR_TYPE_UNKNOWN;
    }

    ~mtmd_context() {
        clip_free(ctx_a);
        clip_free(ctx_v);
    }

private:
    llama_token lookup_token(const std::string & token_text) {
        const llama_vocab * vocab = llama_model_get_vocab(text_model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_vocab; i++) {
            if (token_to_piece(vocab, i, true) == token_text) {
                return i;
            }
        }
        return LLAMA_TOKEN_NULL;
    }

    std::string token_to_piece(const llama_vocab * vocab, llama_token token, bool special) {
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        } else {
            piece.resize(n_chars);
        }
        return piece;
    }
};

mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
        const struct llama_model * text_model,
        const struct mtmd_context_params ctx_params) {
    try {
        return new mtmd_context(mmproj_fname, text_model, ctx_params);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return nullptr;
    }
}

void mtmd_free(mtmd_context * ctx) {
    delete ctx;
}

struct mtmd_tokenizer {
    mtmd_context * ctx;
    std::vector<const mtmd_bitmap *> bitmaps;

    std::string input_text;
    bool add_special;
    bool parse_special;
    const llama_vocab * vocab;

    mtmd_input_chunks cur;
    uint32_t n_images_added = 0; // 0-based index assigned to the next image chunk

    mtmd_tokenizer(mtmd_context * ctx,
            const mtmd_input_text * text,
            const mtmd_bitmap ** bitmaps,
            size_t n_bitmaps) : ctx(ctx), bitmaps(bitmaps, bitmaps + n_bitmaps) {
        add_special   = text->add_special;
        parse_special = text->parse_special;
        input_text    = text->text;
        vocab         = llama_model_get_vocab(ctx->text_model);
    }

    int32_t tokenize(mtmd_input_chunks * output) {
        cur.entries.clear();
        std::vector<std::string> parts = split_text(input_text, ctx->media_marker);
        size_t i_bm = 0; // index of the current bitmap
        for (auto & part : parts) {
            if (part == ctx->media_marker) {
                // this is a marker, we should add the next bitmap
                if (i_bm >= bitmaps.size()) {
                    LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                            __func__, bitmaps.size(), parts.size() - 1);
                    return 1;
                }
                const mtmd_bitmap * bitmap = bitmaps[i_bm++];
                int32_t res = add_media(bitmap);
                if (res != 0) {
                    return res;
                }
            } else {
                // this is a text part, we should add it as text
                add_text(part, parse_special);
            }
        }

        if (add_special && llama_vocab_get_add_bos(vocab)) {
            // if first chunk is text, we add BOS token to first text chunk
            // otherwise, create a new text chunk with BOS token
            if (!cur.entries.empty() && cur.entries[0].type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                // add BOS token to the beginning of first text chunk
                cur.entries[0].tokens_text.insert(cur.entries[0].tokens_text.begin(), llama_vocab_bos(vocab));
            } else {
                // create a new text chunk with BOS token at the beginning
                mtmd_input_chunk bos_chunk{
                    MTMD_INPUT_CHUNK_TYPE_TEXT,
                    {llama_vocab_bos(vocab)},
                    nullptr, // image tokens
                    nullptr, // audio tokens
                };
                cur.entries.insert(cur.entries.begin(), std::move(bos_chunk));
            }
        }

        if (add_special && llama_vocab_get_add_eos(vocab)) {
            // if last chunk is text, we add EOS token to it
            add_text({llama_vocab_eos(vocab)});
        }

        if (i_bm != bitmaps.size()) {
            LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                    __func__, bitmaps.size(), parts.size() - 1);
            return 1;
        }

        *output = std::move(cur);

        return 0;
    }

    void add_text(const std::string & txt, bool parse_special) {
        LOG_DBG("%s: %s\n", __func__, txt.c_str());
        auto tokens = mtmd_tokenize_text_internal(vocab, txt, /* add_special */ false, parse_special);
        add_text(tokens);
    }

    void add_text(const std::vector<llama_token> & tokens) {
        if (tokens.empty()) {
            return;
        }
        // if last entry is also a text chunk, add tokens to it instead of creating new chunk
        if (!cur.entries.empty() && cur.entries.back().type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            cur.entries.back().tokens_text.insert(
                                            cur.entries.back().tokens_text.end(),
                                            tokens.begin(),
                                            tokens.end());
        } else {
            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_TEXT,
                tokens,
                nullptr, // image tokens
                nullptr, // audio tokens
            };
            cur.entries.emplace_back(std::move(chunk));
        }
    }

    int32_t add_media(const mtmd_bitmap * bitmap) {
        if (!bitmap->is_audio) {
            // handle image

            if (!ctx->ctx_v) {
                LOG_ERR("%s: error: model does not support vision input\n", __func__);
                return 2;
            }

            if (!ctx->img_beg.empty()) {
                add_text(ctx->img_beg, true); // add image begin token
            }

            // sanity check
            GGML_ASSERT(bitmap->nx > 0 && bitmap->ny > 0);
            GGML_ASSERT(bitmap->data.size() == (size_t)bitmap->nx * bitmap->ny * 3);
            GGML_ASSERT(ctx->image_preproc != nullptr);

            // convert mtmd_bitmap to clip_image_u8
            clip_image_u8_ptr img_u8(clip_image_u8_init());
            img_u8->nx = bitmap->nx;
            img_u8->ny = bitmap->ny;
            img_u8->buf.resize(bitmap->data.size());
            std::memcpy(img_u8->buf.data(), bitmap->data.data(), img_u8->nx * img_u8->ny * 3);

            // preprocess image
            clip_image_f32_batch batch_f32;
            bool ok = ctx->image_preproc->preprocess(*img_u8, batch_f32);
            if (!ok) {
                LOG_ERR("Unable to preprocess image\n");
                return 2;
            }

            // handle llava-uhd style preprocessing
            const bool has_tiling_grid = batch_f32.grid_x > 0 && batch_f32.grid_y > 0;
            if (
                ctx->slice_tmpl == MTMD_SLICE_TMPL_MINICPMV_2_5
                || ctx->slice_tmpl == MTMD_SLICE_TMPL_MINICPMV_2_6
                || ctx->slice_tmpl == MTMD_SLICE_TMPL_LLAMA4
                || ctx->slice_tmpl == MTMD_SLICE_TMPL_IDEFICS3
                || ctx->slice_tmpl == MTMD_SLICE_TMPL_STEP3VL
                || (ctx->slice_tmpl == MTMD_SLICE_TMPL_LFM2 && has_tiling_grid)
            ) {
                const int n_col = batch_f32.grid_x;
                const int n_row = batch_f32.grid_y;
                // split batch into chunks of single images
                // NOTE: batch_f32 will be invalidated after this call
                auto chunks = split_batch_to_chunk(std::move(batch_f32), bitmap->id);
                GGML_ASSERT(chunks.size() > 0);

                auto ov_chunk = std::move(chunks.front());
                chunks.erase(chunks.begin());

                // add overview image (first)
                if (ctx->ov_img_first) {
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }

                // add slices (or tiles)
                if (!chunks.empty()) {
                    GGML_ASSERT((int)chunks.size() == n_row * n_col);
                    add_text(ctx->tok_slices_start);
                    for (int y = 0; y < n_row; y++) {
                        for (int x = 0; x < n_col; x++) {
                            const bool is_last_in_row = (x == n_col - 1);
                            if (!ctx->tok_sli_img_start.empty()) {
                                add_text(ctx->tok_sli_img_start);
                            } else if (!ctx->sli_img_start_tmpl.empty()) {
                                // If using a template to preceed a slice image
                                const size_t sz = std::snprintf(nullptr, 0, ctx->sli_img_start_tmpl.c_str(), y+1, x+1) + 1;
                                std::unique_ptr<char[]> buf(new char[sz]);
                                std::snprintf(buf.get(), sz, ctx->sli_img_start_tmpl.c_str(), y+1, x+1);
                                add_text(std::string(buf.get(), buf.get() + sz - 1), true);
                            }
                            cur.entries.emplace_back(std::move(chunks[y * n_col + x]));
                            add_text(ctx->tok_sli_img_end);
                            if (!is_last_in_row) {
                                add_text(ctx->tok_sli_img_mid);
                            }
                        }
                        if ((y != n_row - 1 || ctx->tok_row_end_trail)) {
                            add_text(ctx->tok_row_end);
                        }
                    }
                    add_text(ctx->tok_slices_end);
                }

                // add overview image (last)
                if (!ctx->ov_img_first) {
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }

            } else {
                size_t n_tokens = 0;
                for (const auto & entry : batch_f32.entries) {
                    n_tokens += clip_n_output_tokens(ctx->ctx_v, entry.get());
                }

                mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
                if (mtmd_decode_use_mrope(ctx)) {
                    // for Qwen2VL, we need this information for M-RoPE decoding positions
                    image_tokens->nx = clip_n_output_tokens_x(ctx->ctx_v, batch_f32.entries[0].get());
                    image_tokens->ny = clip_n_output_tokens_y(ctx->ctx_v, batch_f32.entries[0].get());
                } else {
                    // other models, we only need the total number of tokens
                    image_tokens->nx = n_tokens;
                    image_tokens->ny = 1;
                }
                image_tokens->pos = ctx->pos_type;
                // HunyuanVL wraps the image grid with BOI/EOI and adds one newline per row,
                // and uses XD-RoPE (dim-3 = image index). Override the position type so that
                // n_tokens() and mtmd_image_tokens_get_decoder_pos pick the HunyuanVL layout.
                if (ctx->proj_type_v() == PROJECTOR_TYPE_HUNYUANVL) {
                    image_tokens->pos       = MTMD_POS_TYPE_HUNYUANVL;
                    image_tokens->image_idx = n_images_added;
                    GGML_ASSERT(n_tokens == (size_t)image_tokens->n_tokens());
                }
                image_tokens->batch_f32 = std::move(batch_f32);
                image_tokens->id = bitmap->id; // optional

                LOG_DBG("image_tokens->nx = %d\n", image_tokens->nx);
                LOG_DBG("image_tokens->ny = %d\n", image_tokens->ny);
                LOG_DBG("batch_f32 size = %d\n", (int)image_tokens->batch_f32.entries.size());

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    {}, // text tokens
                    std::move(image_tokens),
                    nullptr, // audio tokens
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->img_end.empty()) {
                add_text(ctx->img_end, true); // add image end token
            }

            // advance image-chunk counter so the next image gets the next XD-RoPE dim-3 slot
            n_images_added++;

        } else {
            // handle audio

            if (!ctx->ctx_a) {
                LOG_ERR("%s: error: model does not support audio input\n", __func__);
                return 2;
            }

            if (bitmap->data.size() == 0) {
                LOG_ERR("%s: error: empty audio data\n", __func__);
                return 2;
            }

            if (!ctx->aud_beg.empty()) {
                add_text(ctx->aud_beg, true); // add audio begin token
            }

            // sanity check
            GGML_ASSERT(ctx->audio_preproc != nullptr);
            GGML_ASSERT(bitmap->data.size() > sizeof(float));
            GGML_ASSERT(bitmap->data.size() % sizeof(float) == 0);

            // preprocess audio
            std::vector<mtmd_audio_mel> mel_spec_chunks;
            const float * samples = (const float *)bitmap->data.data();
            size_t n_samples = bitmap->data.size() / sizeof(float);
            bool ok = ctx->audio_preproc->preprocess(samples, n_samples, mel_spec_chunks);
            if (!ok) {
                LOG_ERR("Unable to preprocess audio\n");
                return 2;
            }

            // consider each mel_spec as a separate audio chunk
            // TODO: maybe support batching, but this may come with memory cost
            for (auto & mel_spec : mel_spec_chunks) {
                clip_image_f32_ptr mel_f32(clip_image_f32_init());
                mel_f32->nx  = mel_spec.n_len;
                mel_f32->ny  = mel_spec.n_mel;
                mel_f32->buf = std::move(mel_spec.data);
                size_t n_tokens = clip_n_output_tokens(ctx->ctx_a, mel_f32.get());

                clip_image_f32_batch batch_f32;
                batch_f32.is_audio = true;
                batch_f32.entries.push_back(std::move(mel_f32));

                mtmd_audio_tokens_ptr audio_tokens(new mtmd_audio_tokens);
                audio_tokens->n_tokens = n_tokens;
                audio_tokens->batch_f32 = std::move(batch_f32);
                audio_tokens->id = bitmap->id; // optional

                LOG_DBG("audio_tokens->n_tokens = %d\n", audio_tokens->n_tokens);

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    {}, // text tokens
                    nullptr, // image tokens
                    std::move(audio_tokens),
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->aud_end.empty()) {
                add_text(ctx->aud_end, true); // add audio end token
            }
        }

        return 0;
    }

    std::vector<mtmd_input_chunk> split_batch_to_chunk(clip_image_f32_batch && batch_f32, const std::string & id) {
        std::vector<mtmd_input_chunk> chunks;

        for (auto & entry : batch_f32.entries) {
            mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
            image_tokens->nx = clip_n_output_tokens(ctx->ctx_v, entry.get());
            image_tokens->ny = 1;
            image_tokens->batch_f32.entries.push_back(std::move(entry));
            image_tokens->id = id;

            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_IMAGE,
                {}, // text tokens
                std::move(image_tokens),
                nullptr, // audio tokens
            };
            chunks.emplace_back(std::move(chunk));
        }

        return chunks;
    }

    // for example: "a <__media__> b <__media__> c" --> "a", "<__media__>", "b", "<__media__>", "c"
    static std::vector<std::string> split_text(const std::string & input, const std::string & delimiter) {
        std::vector<std::string> result;
        if (input.empty()) {
            return result;
        }
        size_t start = 0;
        size_t pos = 0;
        while ((pos = input.find(delimiter, start)) != std::string::npos) {
            if (pos > start) {
                result.push_back(input.substr(start, pos - start));
            }
            result.push_back(delimiter);
            start = pos + delimiter.length();
        }
        if (start < input.length()) {
            result.push_back(input.substr(start));
        }
        return result;
    }

    // copied from common_tokenize
    static std::vector<llama_token> mtmd_tokenize_text_internal(
        const struct llama_vocab * vocab,
               const std::string & text,
                            bool   add_special,
                            bool   parse_special) {
        // upper limit for the number of tokens
        int n_tokens = text.length() + 2 * add_special;
        std::vector<llama_token> result(n_tokens);
        n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        if (n_tokens < 0) {
            result.resize(-n_tokens);
            int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
            GGML_ASSERT(check == -n_tokens);
        } else {
            result.resize(n_tokens);
        }
        return result;
    }
};

int32_t mtmd_tokenize(mtmd_context * ctx,
            mtmd_input_chunks * output,
            const mtmd_input_text * text,
            const mtmd_bitmap ** bitmaps,
            size_t n_bitmaps) {
    mtmd_tokenizer tokenizer(ctx, text, bitmaps, n_bitmaps);
    return tokenizer.tokenize(output);
}

int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_WRN("mtmd_encode_chunk has no effect for text chunks\n");
        return 0;
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        if (!ctx->ctx_v) {
            LOG_ERR("%s: model does not support vision input\n", __func__);
            return 1;
        }
        return mtmd_encode(ctx, chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        if (!ctx->ctx_a) {
            LOG_ERR("%s: model does not support audio input\n", __func__);
            return 1;
        }
        int n_mmproj_embd = ctx->n_embd_text;
        ctx->image_embd_v.resize(chunk->tokens_audio->n_tokens * n_mmproj_embd);
        bool ok = clip_image_batch_encode(
            ctx->ctx_a,
            ctx->n_threads,
            &chunk->tokens_audio->batch_f32,
            ctx->image_embd_v.data());
        return ok ? 0 : 1;
    }

    LOG_ERR("%s: unknown chunk type %d\n", __func__, (int)chunk->type);
    return 1;
}

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    clip_ctx * ctx_clip = ctx->ctx_v;
    if (!ctx_clip) {
        LOG_ERR("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", __func__);
        return 1;
    }
    auto proj_type = clip_get_projector_type(ctx_clip);
    int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);
    ctx->image_embd_v.resize(image_tokens->n_tokens() * n_mmproj_embd);
    bool ok = false;

    if (clip_is_llava(ctx_clip)
        || clip_is_minicpmv(ctx_clip)
        || clip_is_glm(ctx_clip)
        || proj_type == PROJECTOR_TYPE_INTERNVL) {
        // TODO @ngxson : llava does not support batched encoding ; this should be fixed inside clip_image_batch_encode()
        const auto & entries = image_tokens->batch_f32.entries;
        for (size_t i = 0; i < entries.size(); i++) {
            int n_tokens_per_image = clip_n_output_tokens(ctx_clip, entries[i].get());
            ok = clip_image_encode(
                ctx_clip,
                ctx->n_threads,
                entries[i].get(),
                ctx->image_embd_v.data() + i*n_mmproj_embd*n_tokens_per_image);
        }
    } else {
        ok = clip_image_batch_encode(
            ctx_clip,
            ctx->n_threads,
            &image_tokens->batch_f32,
            ctx->image_embd_v.data());
    }

    return ok ? 0 : 1;
}

float * mtmd_get_output_embd(mtmd_context * ctx) {
    return ctx->image_embd_v.data();
}

bool mtmd_decode_use_non_causal(const mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    auto proj_type = ctx->proj_type_v();
    if (chunk && chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        proj_type = ctx->proj_type_a();
    }
    switch (proj_type) {
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA4V:
            return true;
        default:
            return false;
    }
}

bool mtmd_decode_use_mrope(const mtmd_context * ctx) {
    return ctx->pos_type == MTMD_POS_TYPE_MROPE;
}

bool mtmd_support_vision(const mtmd_context * ctx) {
    return ctx->ctx_v != nullptr;
}

bool mtmd_support_audio(const mtmd_context * ctx) {
    return ctx->ctx_a != nullptr;
}

int mtmd_get_audio_sample_rate(const mtmd_context * ctx) {
    if (!ctx->ctx_a) {
        return -1;
    }
    return clip_get_hparams(ctx->ctx_a)->audio_sample_rate;
}

//
// public API functions
//

// mtmd_bitmap

mtmd_bitmap * mtmd_bitmap_init(uint32_t nx,
                               uint32_t ny,
                               const unsigned char * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap;
    bitmap->nx = nx;
    bitmap->ny = ny;
    size_t data_size = (size_t)nx * ny * 3;
    bitmap->data.resize(data_size);
    std::memcpy(bitmap->data.data(), data, data_size);
    return bitmap;
}

mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples,
                                          const float * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap;
    bitmap->nx = n_samples;
    bitmap->ny = 1;
    bitmap->is_audio = true;
    size_t data_size = n_samples * sizeof(float);
    bitmap->data.resize(data_size);
    std::memcpy(bitmap->data.data(), data, data_size);
    return bitmap;
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap * bitmap) {
    return bitmap->nx;
}

uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap * bitmap) {
    return bitmap->ny;
}

const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap * bitmap) {
    return bitmap->data.data();
}

size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap) {
    return bitmap->data.size();
}

bool mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap) {
    return bitmap->is_audio;
}

const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap) {
    return bitmap->id.c_str();
}

void mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id) {
    if (id) {
        bitmap->id = std::string(id);
    } else {
        bitmap->id.clear();
    }
}

void mtmd_bitmap_free(mtmd_bitmap * bitmap) {
    if (bitmap) {
        delete bitmap;
    }
}

// mtmd_input_chunks

mtmd_input_chunks * mtmd_input_chunks_init() {
    return new mtmd_input_chunks;
}

size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks) {
    return chunks->entries.size();
}

const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx) {
    if (idx >= chunks->entries.size()) {
        return nullptr;
    }
    return &chunks->entries[idx];
}

void mtmd_input_chunks_free(mtmd_input_chunks * chunks) {
    if (chunks) {
        delete chunks;
    }
}

// mtmd_input_chunk

enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk) {
    return chunk->type;
}

const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        *n_tokens_output = chunk->tokens_text.size();
        return chunk->tokens_text.data();
    }
    *n_tokens_output = 0;
    return nullptr;
}

const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return chunk->tokens_image.get();
    }
    return nullptr;
}

size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_tokens(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return mtmd_image_tokens_get_n_pos(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

const char * mtmd_input_chunk_get_id(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        return chunk->tokens_image->id.c_str();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->id.c_str();
    }
    return nullptr;
}

mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk) {
    mtmd_input_chunk * copy = new mtmd_input_chunk{
        chunk->type,
        chunk->tokens_text,
        nullptr,
        nullptr,
    };
    if (chunk->tokens_image) {
        // copy the image tokens
        copy->tokens_image = mtmd_image_tokens_ptr(new mtmd_image_tokens());
        *copy->tokens_image = chunk->tokens_image->clone();
    }
    if (chunk->tokens_audio) {
        // copy the audio tokens
        copy->tokens_audio = mtmd_audio_tokens_ptr(new mtmd_audio_tokens());
        *copy->tokens_audio = chunk->tokens_audio->clone();
    }
    return copy;
}

void mtmd_input_chunk_free(mtmd_input_chunk * chunk) {
    if (chunk) {
        delete chunk;
    }
}

// mtmd_image_tokens

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens) {
    return image_tokens->n_tokens();
}

size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens) {
    return image_tokens->nx;
}

size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens) {
    return image_tokens->ny;
}

mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    mtmd_decoder_pos pos;
    switch (image_tokens->pos) {
        case MTMD_POS_TYPE_MROPE:
            {
                pos.t = pos_0;
                pos.x = pos_0 + (i % image_tokens->nx);
                pos.y = pos_0 + (i / image_tokens->nx);
                pos.z = 0; // unused for now
            } break;
        case MTMD_POS_TYPE_NORMAL:
            {
                pos.t = pos_0 + i;
                pos.x = pos_0 + i;
                pos.y = pos_0 + i;
                pos.z = pos_0 + i;
            } break;
        case MTMD_POS_TYPE_HUNYUANVL:
            {
                // HunyuanVL layout: [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
                // Total = 1 + ny*(nx+1) + 1. BOI and EOI use sequential positions in every dim;
                // content and row-newline tokens use (row, col) with XD-RoPE dim-3 = image_idx.
                const uint32_t nx      = image_tokens->nx;
                const uint32_t n_total = image_tokens->n_tokens();
                if (i == 0) {
                    // BOI
                    pos.t = pos_0 + i;
                    pos.x = pos_0 + i;
                    pos.y = pos_0 + i;
                    pos.z = pos_0 + i;
                } else if (i == n_total - 1) {
                    // EOI
                    pos.t = pos_0 + i;
                    pos.x = pos_0 + i;
                    pos.y = pos_0 + i;
                    pos.z = pos_0 + i;
                } else {
                    // content token at (row, col), or the trailing newline of a row (col == nx)
                    //   section 0 = sequential, section 1 = w(col), section 2 = h(row), section 3 = image_count.
                    // set_position_mrope_2d writes .y -> section 1 and .x -> section 2
                    const uint32_t offset = (uint32_t)i - 1;
                    const uint32_t row    = offset / (nx + 1);
                    const uint32_t col    = offset % (nx + 1);
                    pos.t = pos_0 + i;
                    pos.x = row;
                    pos.y = col;
                    pos.z = image_tokens->image_idx;
                }
            } break;
        default:
            GGML_ABORT("invalid position type");
    }
    return pos;
}

const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens) {
    return image_tokens->id.c_str();
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    switch (image_tokens->pos) {
        case MTMD_POS_TYPE_MROPE:
            return std::max(image_tokens->nx, image_tokens->ny);
        case MTMD_POS_TYPE_NORMAL:
            return image_tokens->n_tokens();
        case MTMD_POS_TYPE_HUNYUANVL:
            // HunyuanVL: the sequential (dim-0) position advances by the full token count
            // (includes BOI/EOI and row newline tokens), not by max(nx, ny)
            return image_tokens->n_tokens();
        default:
            GGML_ABORT("invalid position type");
    }
}

// test function

mtmd_input_chunks * mtmd_test_create_input_chunks() {
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        return nullptr;
    }

    // create a text chunk
    std::vector<llama_token> tokens_text = { 1, 2, 3, 4, 5 };
    mtmd_input_chunk chunk_text{
        MTMD_INPUT_CHUNK_TYPE_TEXT,
        std::move(tokens_text),
        nullptr, // image tokens
        nullptr, // audio tokens
    };
    chunks->entries.emplace_back(std::move(chunk_text));

    // create an image chunk
    mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
    image_tokens->nx = 4;
    image_tokens->ny = 4;
    image_tokens->batch_f32.entries.resize(16);
    image_tokens->id = "image_1";
    mtmd_input_chunk chunk_image{
        MTMD_INPUT_CHUNK_TYPE_IMAGE,
        {}, // text tokens
        std::move(image_tokens),
        nullptr, // audio tokens
    };
    chunks->entries.emplace_back(std::move(chunk_image));

    return chunks;
}

void mtmd_log_set(ggml_log_callback log_callback, void * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : clip_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

//
// Debugging API (NOT intended for public use)
//

static void mtmd_debug_encode_impl(mtmd_context * ctx, clip_ctx * ctx_clip, clip_image_f32 & image) {
    clip_set_debug_output_embeddings(ctx_clip, true);
    int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);
    int n_tokens = clip_n_output_tokens(ctx_clip, &image);
    std::vector<float> embd_output(n_tokens * n_mmproj_embd, 0.0f);
    bool ok = clip_image_encode(
        ctx_clip,
        ctx->n_threads,
        &image,
        embd_output.data());
    if (!ok) {
        LOG_ERR("%s: failed to encode image\n", __func__);
    }
}

void mtmd_debug_encode_image(mtmd_context * ctx, const std::vector<std::vector<float>> & image) {
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    clip_image_f32 inp_image;
    inp_image.nx = image.size();
    inp_image.ny = inp_image.nx;
    inp_image.buf.reserve(inp_image.nx * inp_image.ny);
    for (const auto & row : image) {
        inp_image.buf.insert(inp_image.buf.end(), row.begin(), row.end());
    }
    LOG_INF("%s: created input image with nx=%d, ny=%d\n", __func__, inp_image.nx, inp_image.ny);
    mtmd_debug_encode_impl(ctx, ctx->ctx_v, inp_image);
}

void mtmd_debug_encode_audio(mtmd_context * ctx, const std::vector<float> & input) {
    if (!ctx->ctx_a) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    int n_mel = clip_get_hparams(ctx->ctx_a)->n_mel_bins;
    clip_image_f32 inp_audio;
    inp_audio.nx = input.size();
    inp_audio.ny = n_mel;
    inp_audio.buf.resize(input.size() * n_mel);
    for (size_t i = 0; i < input.size(); i++) {
        for (int j = 0; j < n_mel; j++) {
            inp_audio.buf[j * inp_audio.nx + i] = input[i];
        }
    }
    LOG_INF("%s: created input audio with nx=%d, ny=%d\n", __func__, inp_audio.nx, inp_audio.ny);
    mtmd_debug_encode_impl(ctx, ctx->ctx_a, inp_audio);
}

void mtmd_debug_preprocess_image(mtmd_context * ctx, const std::vector<uint8_t> & rgb_values, int nx, int ny) {
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    clip_image_u8 img_u8;
    img_u8.nx = nx;
    img_u8.ny = ny;
    img_u8.buf = rgb_values;
    clip_image_f32_batch batch_f32;
    GGML_ASSERT(ctx->image_preproc != nullptr);
    bool ok = ctx->image_preproc->preprocess(img_u8, batch_f32);
    if (!ok) {
        LOG_ERR("%s: failed to preprocess image\n", __func__);
        return;
    }
    LOG_INF("%s: preprocessed image to batch_f32 with %d entries\n", __func__, (int)batch_f32.entries.size());
    for (size_t i = 0; i < batch_f32.entries.size(); i++) {
        LOG_INF("%s: entry %zu has nx=%d, ny=%d\n", __func__, i, batch_f32.entries[i]->nx, batch_f32.entries[i]->ny);
        // TODO: better way to dump entry content?
    }
}

void mtmd_debug_preprocess_audio(mtmd_context * ctx, const std::vector<float> & samples) {
    if (!ctx->ctx_a) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    std::vector<mtmd_audio_mel> mel_spec_chunks;
    bool ok = ctx->audio_preproc->preprocess(samples.data(), samples.size(), mel_spec_chunks);
    if (!ok) {
        LOG_ERR("%s: failed to preprocess audio\n", __func__);
        return;
    }
    LOG_INF("%s: preprocessed audio to %zu mel spec chunks\n", __func__, mel_spec_chunks.size());
    for (size_t i = 0; i < mel_spec_chunks.size(); i++) {
        LOG_INF("%s: mel spec chunk %zu has n_len=%d, n_mel=%d\n", __func__, i, mel_spec_chunks[i].n_len, mel_spec_chunks[i].n_mel);

        // dump mel entries: data is stored as [n_mel][n_len] (mel-major)
        const auto & mel = mel_spec_chunks[i];
        for (int m = 0; m < mel.n_mel; m++) {
            for (int t = 0; t < mel.n_len; t++) {
                LOG_INF("mel[%zu][m=%d][t=%d] = %f\n", i, m, t, mel.data[m * mel.n_len + t]);
            }
        }
    }
}


// ---- CLIP graph implementation ----

#define MTMD_INTERNAL_HEADER
#include "mtmd.h.inc"

#include "ggml.h.inc"
#include "ggml-alloc.h.inc"
#include "ggml-backend.h.inc"
#include "gguf.h.inc"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <cinttypes>
#include <limits>
#include <array>
#include <functional>
#include <memory>
#include <float.h>

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) { ggml_free(ctx); }
};

struct gguf_context_deleter {
    void operator()(gguf_context * ctx) { gguf_free(ctx); }
};

struct ggml_backend_buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) { ggml_backend_buffer_free(buffer); }
};

struct ggml_backend_sched_deleter {
    void operator()(ggml_backend_sched_t sched) { ggml_backend_sched_free(sched); }
};

using ggml_context_ptr        = std::unique_ptr<ggml_context, ggml_context_deleter>;
using gguf_context_ptr        = std::unique_ptr<gguf_context, gguf_context_deleter>;
using ggml_backend_buffer_ptr = std::unique_ptr<ggml_backend_buffer, ggml_backend_buffer_deleter>;
using ggml_backend_sched_ptr  = std::unique_ptr<ggml_backend_sched, ggml_backend_sched_deleter>;

#define DEFAULT_INTERPOLATION_MODE (GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS)

struct clip_graph {
    const clip_model & model;
    const clip_hparams & hparams;
    projector_type proj_type;

    // we only support single image per batch
    const clip_image_f32 & img;

    const int patch_size;
    const int n_patches_x;
    const int n_patches_y;
    const int n_patches;
    const int n_embd;
    const int n_head;
    const int d_head;
    const int n_layer;
    const int n_mmproj_embd;
    const float eps;
    float kq_scale; // TODO: maybe move this to hparams
    const clip_flash_attn_type flash_attn_type;

    ggml_context_ptr ctx0_ptr;
    ggml_context * ctx0;
    ggml_cgraph * gf;

    clip_graph(clip_ctx * ctx, const clip_image_f32 & img);

    virtual ~clip_graph() = default;
    virtual ggml_cgraph * build() = 0;

    // wrapper around ggml_mul_mat, allow hooking (e.g. LoRA, clamping) depending on the model
    // tensor w should be the weight matrix, and tensor x should be the input
    virtual ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const;
    // TODO: build_mm(w, b, x) to support bias

    //
    // utility functions
    //
    void cb(ggml_tensor * cur0, const char * name, int il) const;

    // siglip2 naflex
    ggml_tensor * resize_position_embeddings(uint32_t interpolation_mode = DEFAULT_INTERPOLATION_MODE);

    // build vision transformer (ViT) cgraph
    // this function should cover most of the models
    // if your model has specific features, you should probably duplicate this function
    ggml_tensor * build_vit(
                ggml_tensor * inp,
                int64_t n_pos,
                norm_type norm_t,
                ffn_op_type ffn_t,
                ggml_tensor * learned_pos_embd,
                std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> add_pos);

    // build the input after conv2d (inp_raw --> patches)
    // returns tensor with shape [n_embd, n_patches]
    ggml_tensor * build_inp();

    ggml_tensor * build_inp_raw(int channels = 3);

    ggml_tensor * build_norm(
            ggml_tensor * cur,
            ggml_tensor * mw,
            ggml_tensor * mb,
            norm_type type,
            float norm_eps,
            int il) const;

    ggml_tensor * build_ffn(
            ggml_tensor * cur,
            ggml_tensor * up,
            ggml_tensor * up_b,
            ggml_tensor * gate,
            ggml_tensor * gate_b,
            ggml_tensor * down,
            ggml_tensor * down_b,
            ffn_op_type type_op,
            int il) const;

    ggml_tensor * build_attn(
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * kq_mask,
            float kq_scale,
            int il) const;

    // implementation of the 2D RoPE without adding a new op in ggml
    // this is not efficient (use double the memory), but works on all backends
    // TODO: there was a more efficient which relies on ggml_view and ggml_rope_ext_inplace, but the rope inplace does not work well with non-contiguous tensors ; we should fix that and revert back to the original implementation in https://github.com/ggml-org/llama.cpp/pull/13065
    ggml_tensor * build_rope_2d(
        ggml_context * ctx0,
        ggml_tensor * cur,
        ggml_tensor * pos_a, // first half
        ggml_tensor * pos_b, // second half
        const float freq_base,
        const bool interleave_freq
    );

    // aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
    // support dynamic resolution
    ggml_tensor * build_patch_merge_permute(ggml_tensor * cur, int scale_factor);

    // Generic function to stack frames for audio processing
    // Abstracts out the StackAudioFrames logic used by ultravox
    ggml_tensor * build_stack(ggml_tensor * cur, int32_t stack_factor, int32_t n_embed);
};

struct clip_graph_siglip : clip_graph {
    clip_graph_siglip(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_gemma4v : clip_graph {
    clip_graph_gemma4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
};

struct clip_graph_pixtral : clip_graph {
    clip_graph_pixtral(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen2vl : clip_graph {
    clip_graph_qwen2vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen3vl : clip_graph {
    clip_graph_qwen3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_step3vl : clip_graph {
    clip_graph_step3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_youtuvl : clip_graph {
    clip_graph_youtuvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_yasa2 : clip_graph {
    clip_graph_yasa2(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * layer_norm_channels(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b, float eps = 1e-6f);
    ggml_tensor * convnext_grn(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b);
};

struct clip_graph_minicpmv : clip_graph {
    clip_graph_minicpmv(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_internvl : clip_graph {
    clip_graph_internvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_nemotron_v2_vl : clip_graph {
    clip_graph_nemotron_v2_vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llama4 : clip_graph {
    clip_graph_llama4(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_kimivl : clip_graph {
    clip_graph_kimivl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_paddleocr : clip_graph {
    clip_graph_paddleocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_dotsocr : clip_graph {
    clip_graph_dotsocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_cogvlm : clip_graph {
    clip_graph_cogvlm(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llava : clip_graph {
    clip_graph_llava(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_whisper_enc : clip_graph {
    clip_graph_whisper_enc(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_deepseekocr : clip_graph {
    clip_graph_deepseekocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_conformer : clip_graph {
    clip_graph_conformer(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_gemma4a : clip_graph {
    clip_graph_gemma4a(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const override;
};

struct clip_graph_glm4v : clip_graph {
    clip_graph_glm4v(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_hunyuanocr : clip_graph {
    clip_graph_hunyuanocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_mobilenetv5 : clip_graph {
    clip_graph_mobilenetv5(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * rms_norm_2d(
        ggml_tensor * inp,
        ggml_tensor * weight,
        float eps = 1e-6f);

    ggml_tensor* pad_same_2d(
        ggml_tensor* inp,
        int kernel_h,
        int kernel_w,
        int stride_h,
        int stride_w,
        int dilation_h = 1,
        int dilation_w = 1);

    ggml_tensor * build_edge_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_inverted_residual(
        ggml_tensor * inp,
        const mobilenetv5_block & block,
        int stride);

    ggml_tensor * build_mobilenet_attn(
        ggml_tensor * inp,
        const mobilenetv5_block & block);
};

struct clip_graph_qwen3a : clip_graph {
    clip_graph_qwen3a(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_kimik25 : clip_graph {
    clip_graph_kimik25(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * resize_position_embeddings_3d(uint32_t interpolation_mode);
};

struct clip_logger_state g_logger_state = {clip_log_callback_default, NULL};

//#define CLIP_DEBUG_FUNCTIONS

ggml_cgraph * clip_graph_dotsocr::build() {
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    // note: similar to PaddleOCR
    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, 10000, 1, 0, 1, 32, 1);
    };

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_RMS,
                            hparams.ffn_op,
                            nullptr,
                            add_pos);

    cb(cur, "vit_out", -1);

    // dots.ocr patch merger + projector
    {
        GGML_ASSERT(hparams.n_merge > 0);
        cur = build_norm(cur, model.mm_input_norm_w, model.mm_input_norm_b, NORM_TYPE_NORMAL, 1e-6, -1);
        cur = build_patch_merge_permute(cur, hparams.n_merge);
        cb(cur, "after_patch_merger", -1);
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr, // no gate
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF, -1); // nn.GELU() defaults to exact erf-based GELU
        cb(cur, "after_projector", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_nemotron_v2_vl::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_registers = model.class_embedding->ne[1];
    const int n_pos = n_patches + n_registers;

    ggml_tensor * inp = build_inp();

    // add position embeddings (pre-downsampled during GGUF conversion for fixed 512x512 input)
    inp = ggml_add(ctx0, inp, model.position_embeddings);
    cb(inp, "inp_pos", -1);

    inp = ggml_concat(ctx0, model.class_embedding, inp, 1);

    ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_NORMAL, hparams.ffn_op, nullptr, nullptr);

    cur = ggml_view_2d(ctx0, cur,
        n_embd, n_patches,
        ggml_row_size(cur->type, n_embd),
        n_registers * ggml_row_size(cur->type, n_embd));

    cur = build_patch_merge_permute(cur, model.hparams.n_merge);

    {
        cur = build_norm(cur, model.mm_0_w, nullptr, NORM_TYPE_RMS, 1e-6, -1);
        cur = build_ffn(cur, model.mm_1_w, nullptr, nullptr, nullptr, model.mm_3_w, nullptr, FFN_RELU_SQR, -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_paddleocr::build() {
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, 10000, 1, 0, 1, 32, 1);
    };

    ggml_tensor * learned_pos_embd = resize_position_embeddings();
    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        // mlp_AR paddleocr projector
        float proj_norm_eps = 1e-5;
        cur = build_norm(cur,
                    model.mm_input_norm_w, model.mm_input_norm_b,
                    NORM_TYPE_NORMAL, proj_norm_eps, -1);

        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);
        cur = build_ffn(cur,
                    model.mm_1_w, model.mm_1_b,
                    nullptr, nullptr,
                    model.mm_2_w, model.mm_2_b,
                    hparams.ffn_op, -1);
        cb(cur, "mlp_out", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_cogvlm::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_pos = n_patches + 1; // +1 for [CLS]

    ggml_tensor * inp = build_inp();
    inp = ggml_concat(ctx0, inp, model.class_embedding, 1);

    inp = ggml_add(ctx0, inp, model.position_embeddings);
    cb(inp, "inp_pos", -1);

    ggml_tensor * inpL = inp;

    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL;

        cur = build_mm(layer.qkv_w, cur);
        cur = ggml_add(ctx0, cur, layer.qkv_b);

        ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, d_head*sizeof(float),
            cur->nb[1], 0);
        ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, d_head*sizeof(float),
            cur->nb[1], n_embd * sizeof(float));
        ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos, d_head*sizeof(float),
            cur->nb[1], 2 * n_embd * sizeof(float));

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        cur = build_attn(layer.o_w, layer.o_b,
            Qcur, Kcur, Vcur, nullptr, kq_scale, il);
        cb(cur, "attn_out", il);

        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "attn_post_norm", il);

        cur = ggml_add(ctx0, cur, inpL);
        inpL = cur;

        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "layer_out", il);
        inpL = cur;
    }

    ggml_tensor * cur = ggml_view_2d(ctx0, inpL,
        n_embd, n_patches,
        ggml_row_size(inpL->type, n_embd), 0);

    cur = build_mm(model.mm_model_proj, cur);
    cur = build_norm(cur, model.mm_post_fc_norm_w, model.mm_post_fc_norm_b, NORM_TYPE_NORMAL, 1e-5, -1);
    cur = ggml_gelu_inplace(ctx0, cur);

    ggml_tensor * h_to_4h = build_mm(model.mm_h_to_4h_w, cur);
    ggml_tensor * gate = build_mm(model.mm_gate_w, cur);
    gate = ggml_swiglu_split(ctx0, gate, h_to_4h);
    cur = build_mm(model.mm_4h_to_h_w, gate);

    cur = ggml_concat(ctx0, model.mm_boi, cur, 1);
    cur = ggml_concat(ctx0, cur, model.mm_eoi, 1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_hunyuanocr::build() {
    const int merge = hparams.n_merge;
    const int pw    = n_patches_x;
    const int ph    = n_patches_y;

    ggml_tensor * pos_embd = nullptr;
    if (proj_type == PROJECTOR_TYPE_HUNYUANVL && model.position_embeddings) {
        pos_embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ph * pw);
        ggml_set_name(pos_embd, "hunyuanvl_pos_embd");
        ggml_set_input(pos_embd);
    } else {
        pos_embd = resize_position_embeddings(GGML_SCALE_MODE_BILINEAR);
    }

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(inp, n_patches, NORM_TYPE_NORMAL, hparams.ffn_op, pos_embd, nullptr);

    cur = build_norm(cur, model.mm_pre_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, pw, ph);
    cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
    cur = ggml_cont(ctx0, cur);

    cur = ggml_conv_2d(ctx0, model.mm_0_w, cur, merge, merge, 0, 0, 1, 1);
    if (model.mm_0_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.mm_0_b, 1, 1, model.mm_0_b->ne[0]));
    }
    cur = ggml_gelu(ctx0, cur);
    cur = ggml_conv_2d(ctx0, model.mm_1_w, cur, 1, 1, 0, 0, 1, 1);
    if (model.mm_1_b) {
        cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.mm_1_b, 1, 1, model.mm_1_b->ne[0]));
    }

    const int ow   = pw / merge;
    const int oh   = ph / merge;
    const int idim = (int)cur->ne[2];

    ggml_tensor * nl = ggml_reshape_4d(ctx0, model.image_newline, 1, 1, idim, 1);
    nl = ggml_repeat_4d(ctx0, nl, 1, oh, idim, 1);
    cur = ggml_concat(ctx0, cur, nl, 0);

    cur = ggml_permute(ctx0, cur, 1, 2, 0, 3);
    cur = ggml_cont_2d(ctx0, cur, idim, (ow + 1) * oh);

    cur = build_mm(model.mm_model_proj, cur);
    if (model.mm_model_proj_b) {
        cur = ggml_add(ctx0, cur, model.mm_model_proj_b);
    }

    cur = ggml_concat(ctx0, ggml_reshape_2d(ctx0, model.mm_img_begin, model.mm_img_begin->ne[0], 1), cur, 1);
    cur = ggml_concat(ctx0, cur, ggml_reshape_2d(ctx0, model.mm_img_end, model.mm_img_end->ne[0], 1), 1);

    cur = build_norm(cur, model.mm_post_norm_w, nullptr, NORM_TYPE_RMS, eps, -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_cgraph * clip_graph_kimivl::build() {
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * learned_pos_embd = resize_position_embeddings();

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        int proj_inp_dim = cur->ne[0];
        cur = ggml_view_2d(ctx0, cur,
            n_embd, cur->ne[1] * scale_factor * scale_factor,
            ggml_row_size(cur->type, n_embd), 0);
        cur = ggml_norm(ctx0, cur, 1e-5);
        cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        cur = ggml_view_2d(ctx0, cur,
            proj_inp_dim, cur->ne[1] / scale_factor / scale_factor,
            ggml_row_size(cur->type, proj_inp_dim), 0);
        cb(cur, "proj_inp_normed", -1);

        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);
        cb(cur, "proj_out", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_qwen3a::build() {
    ggml_tensor * inp = build_inp_raw(1);

    {
        inp = ggml_conv_2d(ctx0, model.conv2d_1_w, inp, 2, 2, 1, 1, 1, 1);
        inp = ggml_add(ctx0, inp, model.conv2d_1_b);
        inp = ggml_gelu_erf(ctx0, inp);

        inp = ggml_conv_2d(ctx0, model.conv2d_2_w, inp, 2, 2, 1, 1, 1, 1);
        inp = ggml_add(ctx0, inp, model.conv2d_2_b);
        inp = ggml_gelu_erf(ctx0, inp);

        inp = ggml_conv_2d(ctx0, model.conv2d_3_w, inp, 2, 2, 1, 1, 1, 1);
        inp = ggml_add(ctx0, inp, model.conv2d_3_b);
        inp = ggml_gelu_erf(ctx0, inp);

        cb(inp, "after_conv_blocks", -1);

        const int64_t n_pos_after_conv = inp->ne[0];
        const int64_t n_mel_after_conv = inp->ne[1];

        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 0, 2, 3, 1));
        inp = ggml_reshape_2d(ctx0, inp, n_pos_after_conv, n_mel_after_conv * inp->ne[3]);
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

        inp = ggml_mul_mat(ctx0, model.conv_out_w, inp);
        if (model.conv_out_b) {
            inp = ggml_add(ctx0, inp, model.conv_out_b);
        }
        cb(inp, "after_conv_out", -1);
    }

    auto n_pos = inp->ne[1];

    ggml_tensor * pos_embd_selected = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], n_pos,
        model.position_embeddings->nb[1], 0
    );
    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            pos_embd_selected,
                            nullptr);

    cb(cur, "after_transformer", -1);

    cur = build_ffn(cur,
        model.mm_1_w, model.mm_1_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF,
        -1);

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_internvl::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_pos = n_patches + 1;
    ggml_tensor * inp = build_inp();

    inp = ggml_concat(ctx0, inp, model.class_embedding, 1);

    norm_type norm_t = (hparams.n_embd == 3200 && hparams.n_layer == 45)
        ? NORM_TYPE_RMS
        : NORM_TYPE_NORMAL;

    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            norm_t,
                            hparams.ffn_op,
                            model.position_embeddings,
                            nullptr);

    cur = ggml_view_2d(ctx0, cur,
        n_embd, n_patches,
        ggml_row_size(cur->type, n_embd), 0);

    {
        const int scale_factor = model.hparams.n_merge;
        const int bsz    = 1;
        const int height = n_patches_y;
        const int width  = n_patches_x;
        GGML_ASSERT(scale_factor > 0);
        cur = ggml_reshape_4d(ctx0, cur, n_embd * scale_factor, height / scale_factor, width, bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_4d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            height / scale_factor,
            width / scale_factor,
            bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            cur->ne[1] * cur->ne[2]);
    }

    {
        cur = build_norm(cur, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-5, -1);
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_3_w, model.mm_3_b,
            FFN_GELU,
            -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_pixtral::build() {
    const int n_merge = hparams.n_merge;

    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_h, pos_w, hparams.rope_theta, true);
    };

    ggml_tensor * inp = build_inp();
    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_RMS,
                            hparams.ffn_op,
                            nullptr,
                            add_pos);

    if (model.mm_patch_merger_w) {
        GGML_ASSERT(hparams.n_merge > 0);

        cur = ggml_mul(ctx0, ggml_rms_norm(ctx0, cur, eps), model.mm_input_norm_w);

        cur = ggml_reshape_3d(ctx0, cur, n_embd, n_patches_x, n_patches_y);
        cur = ggml_permute(ctx0, cur, 2, 0, 1, 3);
        cur = ggml_cont(ctx0, cur);

        ggml_tensor * kernel = ggml_view_3d(ctx0, cur, n_merge, n_merge, cur->ne[2], 0, 0, 0);
        cur = ggml_im2col(ctx0, kernel, cur, n_merge, n_merge, 0, 0, 1, 1, true, inp->type);

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
        cur = build_mm(model.mm_patch_merger_w, cur);
    }

    {
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);
    }

    if (model.token_embd_img_break) {
        const int p_y             = n_merge > 0 ? n_patches_y / n_merge : n_patches_y;
        const int p_x             = n_merge > 0 ? n_patches_x / n_merge : n_patches_x;
        const int p_total         = p_x * p_y;
        const int n_embd_text     = cur->ne[0];
        const int n_tokens_output = p_total + p_y - 1;

        ggml_tensor * tmp = ggml_reshape_3d(ctx0, cur, n_embd_text, p_x, p_y);
        ggml_tensor * tok = ggml_new_tensor_3d(ctx0, tmp->type, n_embd_text, 1, p_y);
        tok = ggml_scale(ctx0, tok, 0.0);
        tok = ggml_add(ctx0, tok, model.token_embd_img_break);
        tmp = ggml_concat(ctx0, tmp, tok, 1);
        cur = ggml_view_2d(ctx0, tmp,
            n_embd_text, n_tokens_output,
            ggml_row_size(tmp->type, n_embd_text), 0);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_siglip::build() {
    ggml_tensor * inp = build_inp();

    ggml_tensor * learned_pos_embd = model.position_embeddings;
    if (proj_type == PROJECTOR_TYPE_LFM2 || proj_type == PROJECTOR_TYPE_PHI4) {
        learned_pos_embd = resize_position_embeddings();
    }

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            nullptr);

    if (proj_type == PROJECTOR_TYPE_GEMMA3) {
        const int batch_size = 1;
        GGML_ASSERT(n_patches_x == n_patches_y);
        const int patches_per_image = n_patches_x;
        const int kernel_size = hparams.n_merge;

        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont_4d(ctx0, cur, patches_per_image, patches_per_image, n_embd, batch_size);

        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        cur = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[0], n_embd, batch_size);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);

        cur = ggml_mul_mat(ctx0,
            ggml_cont(ctx0, ggml_transpose(ctx0, model.mm_input_proj_w)),
            cur);

    } else if (proj_type == PROJECTOR_TYPE_IDEFICS3) {
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);
        cur = build_mm(model.mm_fc_w, cur);

    } else if (proj_type == PROJECTOR_TYPE_LFM2) {
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        if (model.mm_input_norm_w) {
            cur = ggml_norm(ctx0, cur, 1e-5);
            cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        }

        if (model.mm_input_norm_b) {
            cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        }

        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_JANUS_PRO) {
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_1_w, model.mm_1_b,
            hparams.ffn_op,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_PHI4) {
        cur = build_ffn(cur,
            model.mm_0_w, model.mm_0_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

    } else {
        GGML_ABORT("SigLIP: Unsupported projector type");
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_step3vl::build() {
    GGML_ASSERT(model.class_embedding == nullptr);
    GGML_ASSERT(model.patch_embeddings_0 != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    norm_type norm_t = NORM_TYPE_NORMAL;

    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * inp = build_inp();
    ggml_tensor * learned_pos_embd = resize_position_embeddings();

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };

    auto add_spatial_bias = [&](ggml_tensor * cur, ggml_tensor * bias) {
        if (bias == nullptr) {
            return cur;
        }

        const int64_t width    = cur->ne[0];
        const int64_t height   = cur->ne[1];
        const int64_t channels = cur->ne[2];

        cur = ggml_reshape_2d(ctx0, cur, width * height, channels);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_add(ctx0, cur, bias);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_reshape_3d(ctx0, cur, width, height, channels);

        return cur;
    };

    ggml_tensor * cur = build_vit(
        inp,
        n_patches,
        norm_t,
        hparams.ffn_op,
        learned_pos_embd,
        add_pos);
    cb(cur, "vit_out", -1);

    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont_3d(ctx0, cur, n_patches_x, n_patches_y, n_embd);

    cur = ggml_conv_2d(ctx0, model.mm_0_w, cur, 2, 2, 1, 1, 1, 1);
    cur = add_spatial_bias(cur, model.mm_0_b);
    cb(cur, "downsample_0", -1);

    cur = ggml_conv_2d(ctx0, model.mm_1_w, cur, 2, 2, 1, 1, 1, 1);
    cur = add_spatial_bias(cur, model.mm_1_b);
    cb(cur, "downsample_1", -1);

    {
        const int64_t w = cur->ne[0];
        const int64_t h = cur->ne[1];
        cur = ggml_reshape_3d(ctx0, cur, w * h, cur->ne[2], cur->ne[3]);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    }
    cb(cur, "downsample_flatten", -1);

    cur = ggml_mul_mat(ctx0, model.mm_model_proj, cur);
    cb(cur, "projector_out", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_tensor * clip_graph_kimik25::resize_position_embeddings_3d(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny / patch_size;
    const int width        = img.nx / patch_size;
    const uint32_t mode    = interpolation_mode;

    GGML_ASSERT(pos_embd);

    const int64_t stored_c = pos_embd->ne[0];
    const int64_t orig_w = pos_embd->ne[1];
    const int64_t orig_h = pos_embd->ne[2];

    GGML_ASSERT(stored_c == n_embd);

    if (height == (int)orig_h && width == (int)orig_w) {
        return ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    }

    pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = ggml_interpolate(ctx0, pos_embd, height, width, n_embd, 1, mode);
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    return pos_embd;
}

ggml_cgraph * clip_graph_kimik25::build() {
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * learned_pos_embd = resize_position_embeddings_3d(GGML_SCALE_MODE_BICUBIC);

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        cur = build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
        return cur;
    };

    ggml_tensor * inp = build_inp();
    inp = ggml_add(ctx0, inp, learned_pos_embd);

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            nullptr,
                            add_pos);

    cb(cur, "vit_out", -1);

    {
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        int proj_inp_dim = cur->ne[0];
        int n_merged_patches = cur->ne[1];
        cur = ggml_view_2d(ctx0, cur,
            n_embd, n_merged_patches * scale_factor * scale_factor,
            ggml_row_size(cur->type, n_embd), 0);
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.mm_input_norm_w);
        cur = ggml_add(ctx0, cur, model.mm_input_norm_b);
        cur = ggml_view_2d(ctx0, cur,
            proj_inp_dim, n_merged_patches,
            ggml_row_size(cur->type, proj_inp_dim), 0);
        cb(cur, "proj_inp_normed", -1);

        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU,
            -1);

        cb(cur, "proj_out", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_minicpmv::build() {
    GGML_ASSERT(model.class_embedding == nullptr);
    const int n_pos       = n_patches;
    const int n_embd_proj = n_mmproj_embd;

    ggml_tensor * omega = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_embd_proj / 4);
    ggml_set_name(omega, "omega");
    ggml_set_input(omega);

    ggml_tensor * pos_h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_pos);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);
    ggml_tensor * pos_w = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_pos);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, model.position_embeddings, positions);

    ggml_tensor * inp = build_inp();
    ggml_tensor * embeddings = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            nullptr);

    ggml_tensor * q = model.mm_model_query;
    ggml_tensor * v = build_mm(model.mm_model_kv_proj, embeddings);

    q = build_norm(q, model.mm_model_ln_q_w,  model.mm_model_ln_q_b,  NORM_TYPE_NORMAL, eps, -1);
    v = build_norm(v, model.mm_model_ln_kv_w, model.mm_model_ln_kv_b, NORM_TYPE_NORMAL, eps, -1);

    ggml_tensor * pos_embed = nullptr;
    {
        ggml_tensor * omega_b = ggml_repeat_4d(ctx0, omega, omega->ne[0], n_pos, 1, 1);
        ggml_tensor * theta_x = ggml_mul(ctx0, omega_b, pos_w);
        ggml_tensor * theta_y = ggml_mul(ctx0, omega_b, pos_h);
        ggml_tensor * pos_embd_x = ggml_concat(
            ctx0,
            ggml_sin(ctx0, theta_x),
            ggml_cos(ctx0, theta_x),
            0
        );
        ggml_tensor * pos_embd_y = ggml_concat(
            ctx0,
            ggml_sin(ctx0, theta_y),
            ggml_cos(ctx0, theta_y),
            0
        );
        pos_embed = ggml_concat(ctx0, pos_embd_x, pos_embd_y, 0);
    }

    ggml_tensor * k = ggml_add(ctx0, v, pos_embed);

    {
        const int d_head = 128;
        int n_head = n_embd_proj/d_head;
        int num_query = hparams.minicpmv_query_num;
        ggml_tensor * Q = ggml_add(ctx0,
            build_mm(model.mm_model_attn_q_w, q),
            model.mm_model_attn_q_b);
        ggml_tensor * K = ggml_add(ctx0,
            build_mm(model.mm_model_attn_k_w, k),
            model.mm_model_attn_k_b);
        ggml_tensor * V = ggml_add(ctx0,
            build_mm(model.mm_model_attn_v_w, v),
            model.mm_model_attn_v_b);

        Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, num_query);
        K = ggml_reshape_3d(ctx0, K, d_head, n_head, n_pos);
        V = ggml_reshape_3d(ctx0, V, d_head, n_head, n_pos);

        cb(Q, "resampler_Q", -1);
        cb(K, "resampler_K", -1);
        cb(V, "resampler_V", -1);

        float resampler_kq_scale = 1.0f/ sqrtf(float(d_head));
        embeddings = build_attn(
            model.mm_model_attn_o_w,
            model.mm_model_attn_o_b,
            Q, K, V, nullptr, resampler_kq_scale, -1);
        cb(embeddings, "resampler_attn_out", -1);
    }

    embeddings = build_norm(embeddings, model.mm_model_ln_post_w, model.mm_model_ln_post_b, NORM_TYPE_NORMAL, eps, -1);
    embeddings = build_mm(model.mm_model_proj, embeddings);

    ggml_build_forward_expand(gf, embeddings);

    return gf;
}

ggml_cgraph * clip_graph_llama4::build() {
    GGML_ASSERT(model.class_embedding != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);

    const int n_pos = n_patches + 1; // +1 for [CLS]

    // 2D input positions
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * inp = build_inp_raw();

    // Llama4UnfoldConvolution
    {
        ggml_tensor * kernel = ggml_reshape_4d(ctx0, model.patch_embeddings_0,
                                                patch_size, patch_size, 3, n_embd);
        inp = ggml_im2col(ctx0, kernel, inp, patch_size, patch_size, 0, 0, 1, 1, true, inp->type);
        inp = build_mm(model.patch_embeddings_0, inp);
        inp = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);
        cb(inp, "patch_conv", -1);
    }

    // add CLS token
    inp = ggml_concat(ctx0, inp, model.class_embedding, 1);

    // build ViT with 2D position embeddings
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        // first half is X axis and second half is Y axis
        // ref: https://github.com/huggingface/transformers/blob/40a493c7ed4f19f08eadb0639cf26d49bfa5e180/src/transformers/models/llama4/modeling_llama4.py#L1312
        // ref: https://github.com/Blaizzy/mlx-vlm/blob/a57156aa87b33cca6e5ee6cfc14dd4ef8f611be6/mlx_vlm/models/llama4/vision.py#L441
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };
    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            model.position_embeddings,
                            add_pos);

    // remove CLS token
    cur = ggml_view_2d(ctx0, cur,
        n_embd, n_patches,
        ggml_row_size(cur->type, n_embd), 0);

    // pixel shuffle
    // based on Llama4VisionPixelShuffleMLP
    // https://github.com/huggingface/transformers/blob/2932f318a20d9e54cc7aea052e040164d85de7d6/src/transformers/models/llama4/modeling_llama4.py#L1151
    {
        const int scale_factor = model.hparams.n_merge;
        const int bsz = 1; // batch size, always 1 for now since we don't support batching
        GGML_ASSERT(scale_factor > 0);
        GGML_ASSERT(n_patches_x == n_patches_y); // llama4 only supports square images
        cur = ggml_reshape_4d(ctx0, cur,
            n_embd * scale_factor,
            n_patches_x / scale_factor,
            n_patches_y,
            bsz);
        cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        cur = ggml_cont_4d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            n_patches_x / scale_factor,
            n_patches_y / scale_factor,
            bsz);
        //cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
        // flatten to 2D
        cur = ggml_cont_2d(ctx0, cur,
            n_embd * scale_factor * scale_factor,
            n_patches / scale_factor / scale_factor);
        cb(cur, "pixel_shuffle", -1);
    }

    // based on Llama4VisionMLP2 (always uses GELU activation, no bias)
    {
        cur = build_mm(model.mm_model_mlp_1_w, cur);
        cur = ggml_gelu(ctx0, cur);
        cur = build_mm(model.mm_model_mlp_2_w, cur);
        cur = ggml_gelu(ctx0, cur);
        cb(cur, "adapter_mlp", -1);
    }

    // Llama4MultiModalProjector
    cur = build_mm(model.mm_model_proj, cur);
    cb(cur, "projected", -1);

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_glm4v::build() {
    GGML_ASSERT(model.patch_bias != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size = 1;

    norm_type norm_t = NORM_TYPE_RMS;

    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};
    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches * 4);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    GGML_ASSERT(img.nx % (patch_size * 2) == 0);
    GGML_ASSERT(img.ny % (patch_size * 2) == 0);

    // second conv dimension
    {
        auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_add(ctx0, inp, inp_1);

        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // add patch bias
    inp = ggml_add(ctx0, inp, model.patch_bias);
    cb(inp, "patch_bias", -1);

    // pos-conv norm
    inp = build_norm(inp, model.norm_embd_w, model.norm_embd_b, norm_t, eps, -1);

    ggml_tensor * learned_pos_embd = nullptr;
    // Note: GLM-OCR does not have learned position embeddings
    if (model.position_embeddings != nullptr) {
        learned_pos_embd = resize_position_embeddings(GGML_SCALE_MODE_BICUBIC);
        learned_pos_embd = ggml_cont_4d(
            ctx0, learned_pos_embd,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        learned_pos_embd = ggml_reshape_4d(
            ctx0, learned_pos_embd,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
        learned_pos_embd = ggml_cont_3d(
            ctx0, learned_pos_embd,
            n_embd, n_patches_x * n_patches_y, batch_size);
        cb(learned_pos_embd, "learned_pos_embd", -1);
    }

    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, hparams.rope_theta, 1, 0, 1, 32, 1);
    };

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            norm_t,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);

    cb(cur, "vit_out", -1);
    // cb(ggml_sum(ctx0, cur), "vit_out_sum", -1);

    // GLM4V projector
    // ref: https://github.com/huggingface/transformers/blob/40dc11cd3eb4126652aa41ef8272525affd4a636/src/transformers/models/glm4v/modeling_glm4v.py#L116-L130

    // patch merger (downsample)
    {
        int n_merge = hparams.n_merge;
        GGML_ASSERT(n_merge > 0);

        int n_token_out = n_patches / n_merge / n_merge;
        cur = ggml_reshape_4d(ctx0, cur, n_embd, n_merge, n_merge, n_token_out);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3)); // [n_merge, n_merge, n_embd, n_token_out]
        cur = ggml_conv_2d(ctx0, model.mm_patch_merger_w, cur, n_merge, n_merge, 0, 0, 1, 1);
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[2], n_token_out); // [n_embd_out, n_token_out]

        cur = ggml_add(ctx0, cur, model.mm_patch_merger_b);
    }

    // FC projector
    {
        cur = build_mm(model.mm_fc_w, cur);
        // default LayerNorm (post_projection_norm)
        cur = build_norm(cur, model.mm_post_norm_w, model.mm_post_norm_b, NORM_TYPE_NORMAL, 1e-5, -1);
        cur = ggml_gelu_erf(ctx0, cur);
        cb(cur, "after_fc_proj", -1);
    }

    // FFN projector
    {
        cur = build_ffn(cur,
            model.mm_ffn_up_w, model.mm_ffn_up_b,
            model.mm_ffn_gate_w, model.mm_ffn_gate_b,
            model.mm_ffn_down_w, model.mm_ffn_down_b,
            hparams.ffn_op, -1);
        cb(cur, "after_ffn_proj", -1);
        // cb(ggml_sum(ctx0, cur), "merged_sum", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_gemma4v::build() {
    ggml_tensor * inp_raw = build_inp_raw();

    // patches = 2 * (patches - 0.5)
    // equivalent to: patches * 2 - 1
    inp_raw = ggml_scale_bias(ctx0, inp_raw, 2.0f, -1.0f);
    ggml_set_name(inp_raw, "inp_raw_scaled");

    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    ggml_set_name(inp, "inp");
    // note: no patch bias

    ggml_tensor * pos_x = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_x, "pos_x");
    ggml_set_input(pos_x);

    ggml_tensor * pos_y = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_y, "pos_y");
    ggml_set_input(pos_y);

    {
        const int64_t pos_size = model.position_embeddings->ne[1];
        const size_t  nb1      = ggml_row_size(model.position_embeddings->type, n_embd);

        // positional embeddings are stored as lookup tables (one for x, one for y)
        ggml_tensor * tbl_x = ggml_view_2d(ctx0, model.position_embeddings,
                                             n_embd, pos_size, nb1, 0);
        ggml_tensor * tbl_y = ggml_view_2d(ctx0, model.position_embeddings,
                                             n_embd, pos_size, nb1, pos_size * nb1);

        // ggml_get_rows: [n_embd, n_patches]
        ggml_tensor * emb_x = ggml_get_rows(ctx0, tbl_x, pos_x);
        ggml_tensor * emb_y = ggml_get_rows(ctx0, tbl_y, pos_y);

        inp = ggml_add(ctx0, inp, emb_x);
        inp = ggml_add(ctx0, inp, emb_y);
        cb(inp, "pos_embd", -1);
    }

    // similar to build_rope_2d, but use neox ordering
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        const int64_t n_dim  = cur->ne[0];
        const int64_t n_head = cur->ne[1];
        const int64_t n_pos  = cur->ne[2];

        // first half
        ggml_tensor * first;
        {
            first = ggml_view_3d(ctx0, cur,
                n_dim/2, n_head, n_pos,
                cur->nb[1],
                cur->nb[2],
                0);
            first = ggml_rope_ext(
                ctx0,
                first,
                pos_x,      // positions
                nullptr,    // freq factors
                n_dim/2,    // n_dims
                GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta,
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f
            );
        }

        // second half
        ggml_tensor * second;
        {
            second = ggml_view_3d(ctx0, cur,
                n_dim/2, n_head, n_pos,
                cur->nb[1],
                cur->nb[2],
                n_dim/2 * ggml_element_size(cur));
            second = ggml_rope_ext(
                ctx0,
                second,
                pos_y,      // positions
                nullptr,    // freq factors
                n_dim/2,    // n_dims
                GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta,
                1.0f, 0.0f, 1.0f, 0.0f, 0.0f
            );
        }

        cur = ggml_concat(ctx0, first, second, 0);
        return cur;
    };

    kq_scale = 1.0f;
    ggml_tensor * cur = build_vit(
                        inp, n_patches,
                        NORM_TYPE_RMS,
                        hparams.ffn_op,
                        nullptr, // pos embd is already handled above
                        add_pos);

    // Gemma4VisionPooler
    {
        const int kernel_size = hparams.n_merge;
        GGML_ASSERT(kernel_size > 0);

        // [n_embd, n_patches] -> [n_patches_x, n_patches_y, n_embd, 1]
        cur = ggml_cont_4d(ctx0, ggml_transpose(ctx0, cur), n_patches_x, n_patches_y, n_embd, 1);
        cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG,
                           kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        const int out_x = n_patches_x / kernel_size;
        const int out_y = n_patches_y / kernel_size;
        // [out_x, out_y, n_embd, 1] -> [n_embd, out_x * out_y]
        cur = ggml_reshape_3d(ctx0, cur, out_x * out_y, n_embd, 1);
        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cur = ggml_scale(ctx0, cur, sqrtf((float)n_embd));
        cb(cur, "pooled", -1);
    }

    // hidden_states = (hidden_states - self.std_bias) * self.std_scale
    if (model.std_bias && model.std_scale) {
        cur = ggml_sub(ctx0, cur, model.std_bias);
        cur = ggml_mul(ctx0, cur, model.std_scale);
        cb(cur, "std_scaled", -1);
    }

    // Gemma4MultimodalEmbedder
    cur = build_mm(model.mm_input_proj_w, cur);
    cb(cur, "projected", -1);

    // embedding_post_projection_norm
    cur = ggml_rms_norm(ctx0, cur, hparams.eps);
    cb(cur, "projected_normed", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_tensor * clip_graph_gemma4v::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    // Gemma4ClippableLinear

    auto it = model.clamp_info_map.find(w->name);
    if (it == model.clamp_info_map.end()) {
        return ggml_mul_mat(ctx0, w, x);
    } else {
        const auto & clamp_info = it->second;
        ggml_tensor * clamped = ggml_clamp(ctx0, x, clamp_info.inp_min, clamp_info.inp_max);
        ggml_tensor * out = ggml_mul_mat(ctx0, w, clamped);
        out = ggml_clamp(ctx0, out, clamp_info.out_min, clamp_info.out_max);
        return out;
    }
}

ggml_cgraph * clip_graph_whisper_enc::build() {
    const int n_frames = img.nx;
    const int n_pos    = n_frames / 2;
    GGML_ASSERT(model.position_embeddings->ne[1] >= n_pos);

    ggml_tensor * inp = build_inp_raw(1);

    // conv1d block
    {
        // convolution + gelu
        ggml_tensor * cur = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_1_b);

        cur = ggml_gelu_erf(ctx0, cur);

        cur = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_2_b);

        cur = ggml_gelu_erf(ctx0, cur);
        // transpose
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cb(inp, "after_conv1d", -1);
    }

    // sanity check (only check one layer, but it should be the same for all)
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
    GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
    GGML_ASSERT(model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].v_b);
    GGML_ASSERT(!model.layers[0].k_b); // no bias for k

    ggml_tensor * pos_embd_selected = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], n_pos,
        model.position_embeddings->nb[1], 0
    );
    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            pos_embd_selected,
                            nullptr);

    cb(cur, "after_transformer", -1);

    if (model.audio_has_stack_frames()) {
        // StackAudioFrames
        // https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/blob/main/ultravox_model.py
        cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
        cb(cur, "after_stacked", -1);
    }

    if (proj_type == PROJECTOR_TYPE_ULTRAVOX) {
        // UltravoxProjector
        // pre-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);

        // ffn in
        cur = build_mm(model.mm_1_w, cur);

        // swiglu
        // see SwiGLU in ultravox_model.py, the second half passed through is silu, not the first half
        cur = ggml_swiglu_swapped(ctx0, cur);

        // mid-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_mid_w);

        // ffn out
        cur = build_mm(model.mm_2_w, cur);

    } else if (proj_type == PROJECTOR_TYPE_QWEN2A) {
        // projector
        cur = build_mm(model.mm_fc_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_fc_b);

    } else if (proj_type == PROJECTOR_TYPE_VOXTRAL) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_MUSIC_FLAMINGO) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_MERALION) {
        // stack (above) -> ln -> linear0+silu -> GLU -> out
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
        cur = ggml_add(ctx0, cur, model.mm_norm_pre_b);

        cur = ggml_mul_mat(ctx0, model.mm_0_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_0_b);
        cur = ggml_silu(ctx0, cur);

        ggml_tensor * gate = ggml_mul_mat(ctx0, model.mm_1_w, cur);
        gate = ggml_add(ctx0, gate, model.mm_1_b);
        gate = ggml_silu(ctx0, gate);

        ggml_tensor * pool = ggml_mul_mat(ctx0, model.mm_2_w, cur);
        pool = ggml_add(ctx0, pool, model.mm_2_b);

        cur = ggml_mul(ctx0, gate, pool);

        cur = ggml_mul_mat(ctx0, model.mm_3_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_3_b);

    } else if (proj_type == PROJECTOR_TYPE_GLMA) {
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
            cur = ggml_add(ctx0, cur, model.mm_norm_pre_b);
            cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
            cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_2_w, model.mm_2_b, hparams.ffn_op, 0);
            cur = ggml_concat(ctx0, model.mm_boi, cur, 1);
            cur = ggml_concat(ctx0, cur, model.mm_eoi, 1);
    } else {
        GGML_ABORT("%s: unknown projector type", __func__);
    }

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_youtuvl::build() {
    GGML_ASSERT(model.class_embedding == nullptr);
    const int batch_size       = 1;
    const bool use_window_attn = !hparams.wa_layer_indexes.empty();
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4;
    const int m = 2;
    const int Wp = n_patches_x;
    const int Hp = n_patches_y;
    const int Hm = Hp / m;
    const int Wm = Wp / m;
    norm_type norm_t = NORM_TYPE_NORMAL;

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp = build_inp_raw();

    // change conv3d to linear
    // reshape and permute to get patches, permute from (patch_size, m, Wm, patch_size, m, Hm, C) to (C, patch_size, patch_size, m, m, Wm, Hm)
    {
        inp = ggml_reshape_4d(
            ctx0, inp,
            Wm * m * patch_size, m * patch_size, Hm, 3);
        inp = ggml_permute(ctx0, inp, 1, 2, 3, 0);
        inp = ggml_cont_4d(
            ctx0, inp,
            m * patch_size * 3, Wm, m * patch_size, Hm);

        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_4d(
            ctx0, inp,
            m * patch_size * 3, patch_size, m, Hm * Wm);

        inp = ggml_permute(ctx0, inp, 1, 0, 2, 3);
        inp = ggml_cont_4d(
            ctx0, inp,
            patch_size, 3, patch_size, Hm * Wm * m * m);

        inp = ggml_permute(ctx0, inp, 2, 0, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            3*patch_size* patch_size,  Hm * Wm * m * m, 1);
    }
    inp = build_mm(model.patch_embeddings_0, inp);

    if (model.patch_bias) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
    }

    inp = ggml_reshape_2d(ctx0, inp, n_embd, n_patches);

    ggml_tensor * inpL           = inp;
    ggml_tensor * window_mask    = nullptr;
    ggml_tensor * window_idx     = nullptr;
    ggml_tensor * inv_window_idx = nullptr;

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }
    if (use_window_attn) {
        inv_window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(inv_window_idx, "inv_window_idx");
        ggml_set_input(inv_window_idx);
        // mask for window attention
        window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
        ggml_set_name(window_mask, "window_mask");
        ggml_set_input(window_mask);

        // if flash attn is used, we need to pad the mask and cast to f16
        if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
            window_mask = ggml_cast(ctx0, window_mask, GGML_TYPE_F16);
        }

        // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        GGML_ASSERT(batch_size == 1);
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_patches_x * n_patches_y * batch_size / 4);
        inpL = ggml_get_rows(ctx0, inpL, inv_window_idx);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const bool full_attn = use_window_attn ? hparams.wa_layer_indexes.count(il) > 0 : true;

        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        // self-attention
        {
            ggml_tensor * Qcur = ggml_add(ctx0,
                build_mm(layer.q_w, cur), layer.q_b);
            ggml_tensor * Kcur = ggml_add(ctx0,
                build_mm(layer.k_w, cur), layer.k_b);
            ggml_tensor * Vcur = ggml_add(ctx0,
                build_mm(layer.v_w, cur), layer.v_b);

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

            Qcur = ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
        }
        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            nullptr, nullptr,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);

        inpL = cur;
    }

    ggml_tensor * embeddings = inpL;
    if (use_window_attn) {
        const int spatial_merge_unit = 4;
        window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / spatial_merge_unit);
        ggml_set_name(window_idx, "window_idx");
        ggml_set_input(window_idx);
        GGML_ASSERT(batch_size == 1);
        embeddings = ggml_reshape_2d(ctx0, embeddings, n_embd * spatial_merge_unit, n_patches / spatial_merge_unit);
        embeddings = ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd, n_patches, batch_size);
        cb(embeddings, "window_order_restored", -1);
    }

    // post-layernorm (part of Siglip2VisionTransformer, applied after encoder)
    if (model.post_ln_w) {
        embeddings = build_norm(embeddings, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    }

    // Now apply merger (VLPatchMerger):
    // 1. Apply RMS norm (ln_q in VLPatchMerger)
    embeddings = build_norm(embeddings, model.mm_input_norm_w, nullptr, NORM_TYPE_RMS, 1e-6, -1);
    cb(embeddings, "merger_normed", -1);

    // 2. First reshape for spatial merge (merge 2x2 patches)
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);
    cb(embeddings, "merger_reshaped", -1);

    embeddings = build_ffn(embeddings,
                    model.mm_0_w, model.mm_0_b,
                    nullptr, nullptr,
                    model.mm_1_w, model.mm_1_b,
                    FFN_GELU,
                    -1);
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}

ggml_cgraph * clip_graph_qwen2vl::build() {
    GGML_ASSERT(model.patch_bias == nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size       = 1;
    const bool use_window_attn = hparams.n_wa_pattern > 0;
    const int n_wa_pattern     = hparams.n_wa_pattern;
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    norm_type norm_t = proj_type == PROJECTOR_TYPE_QWEN25VL
        ? NORM_TYPE_RMS // qwen 2.5 vl
        : NORM_TYPE_NORMAL; // qwen 2 vl

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    GGML_ASSERT(img.nx % (patch_size * 2) == 0);
    GGML_ASSERT(img.ny % (patch_size * 2) == 0);

    // second conv dimension
    {
        auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_add(ctx0, inp, inp_1);

        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    ggml_tensor * inpL           = inp;
    ggml_tensor * window_mask    = nullptr;
    ggml_tensor * window_idx     = nullptr;
    ggml_tensor * inv_window_idx = nullptr;

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }

    if (use_window_attn) {
        // handle window attention inputs
        inv_window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(inv_window_idx, "inv_window_idx");
        ggml_set_input(inv_window_idx);
        // mask for window attention
        window_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
        ggml_set_name(window_mask, "window_mask");
        ggml_set_input(window_mask);

        // if flash attn is used, we need to pad the mask and cast to f16
        if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
            window_mask = ggml_cast(ctx0, window_mask, GGML_TYPE_F16);
        }

        // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        GGML_ASSERT(batch_size == 1);
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * 4, n_patches_x * n_patches_y * batch_size / 4);
        inpL = ggml_get_rows(ctx0, inpL, inv_window_idx);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        const auto & layer = model.layers[il];
        const bool full_attn = use_window_attn ? (il + 1) % n_wa_pattern == 0 : true;

        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        // self-attention
        {
            ggml_tensor * Qcur = ggml_add(ctx0,
                build_mm(layer.q_w, cur), layer.q_b);
            ggml_tensor * Kcur = ggml_add(ctx0,
                build_mm(layer.k_w, cur), layer.k_b);
            ggml_tensor * Vcur = ggml_add(ctx0,
                build_mm(layer.v_w, cur), layer.v_b);

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // apply M-RoPE
            Qcur = ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            ggml_tensor * attn_mask = full_attn ? nullptr : window_mask;

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, attn_mask, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    }

    // multimodal projection
    ggml_tensor * embeddings = inpL;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);
    embeddings = build_ffn(embeddings,
                        model.mm_0_w, model.mm_0_b,
                        nullptr, nullptr,
                        model.mm_1_w, model.mm_1_b,
                        FFN_GELU,
                        -1);

    if (use_window_attn) {
        window_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos / 4);
        ggml_set_name(window_idx, "window_idx");
        ggml_set_input(window_idx);

        // embeddings shape: [n_embd, n_patches_x * n_patches_y, batch_size]
        GGML_ASSERT(batch_size == 1);
        embeddings = ggml_reshape_2d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4);
        embeddings = ggml_get_rows(ctx0, embeddings, window_idx);
        embeddings = ggml_reshape_3d(ctx0, embeddings, hparams.projection_dim, n_patches_x * n_patches_y / 4, batch_size);
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}

ggml_cgraph * clip_graph_qwen3vl::build() {
    GGML_ASSERT(model.patch_bias != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size       = 1;
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    norm_type norm_t = NORM_TYPE_NORMAL;

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    GGML_ASSERT(img.nx % (patch_size * 2) == 0);
    GGML_ASSERT(img.ny % (patch_size * 2) == 0);

    // second conv dimension
    {
        auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_add(ctx0, inp, inp_1);

        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // add patch bias
    if (model.patch_bias != nullptr) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }

    // calculate absolute position embedding and apply
    ggml_tensor * learned_pos_embd = resize_position_embeddings();
    learned_pos_embd = ggml_cont_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
    learned_pos_embd = ggml_reshape_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
    learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = ggml_cont_3d(
        ctx0, learned_pos_embd,
        n_embd, n_patches_x * n_patches_y, batch_size);
    inp = ggml_add(ctx0, inp, learned_pos_embd);
    cb(inp, "inp_pos_emb", -1);

    ggml_tensor * inpL = inp;

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }

    // deepstack features (stack along the feature dimension), [n_embd * len(deepstack_layers), n_patches_x * n_patches_y, batch_size]
    ggml_tensor * deepstack_features = nullptr;
    const int merge_factor = hparams.n_merge > 0 ? hparams.n_merge * hparams.n_merge : 4; // default 2x2=4 for qwen3vl

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];

        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "ln1", il);

        // self-attention
        {
            cur = build_mm(layer.qkv_w, cur);
            cur = ggml_add(ctx0, cur, layer.qkv_b);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ 0);

            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, n_embd));

            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, 2 * n_embd));

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            // apply M-RoPE
            Qcur = ggml_rope_multi(
                ctx0, Qcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);
            Kcur = ggml_rope_multi(
                ctx0, Kcur, positions, nullptr,
                d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION, 32768, 10000, 1, 0, 1, 32, 1);

            cb(Qcur, "Qcur_rope", il);
            cb(Kcur, "Kcur_rope", il);

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        if (layer.has_deepstack()) {
            ggml_tensor * feat = ggml_reshape_3d(ctx0, cur, n_embd * merge_factor, n_pos / merge_factor, batch_size);
            feat = build_norm(feat, layer.deepstack_norm_w, layer.deepstack_norm_b, norm_t, eps, il);
            feat = build_ffn(feat,
                layer.deepstack_fc1_w, layer.deepstack_fc1_b,
                nullptr, nullptr,
                layer.deepstack_fc2_w, layer.deepstack_fc2_b,
                ffn_op_type::FFN_GELU, il);

            if(!deepstack_features) {
                deepstack_features = feat;
            } else {
                // concat along the feature dimension
                deepstack_features = ggml_concat(ctx0, deepstack_features, feat, 0);
            }
        }

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    }

    // multimodal projection
    ggml_tensor * embeddings = inpL;
    embeddings = ggml_reshape_3d(ctx0, embeddings, n_embd * 4, n_pos / 4, batch_size);

    embeddings = build_ffn(embeddings,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_1_w, model.mm_1_b,
        ffn_op_type::FFN_GELU, -1);

    if (deepstack_features) {
        embeddings = ggml_concat(ctx0, embeddings, deepstack_features, 0);
    } // concat along the feature dimension

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}

ggml_cgraph * clip_graph_conformer::build() {
    const int n_frames   = img.nx;
    const int n_pos      = n_frames / 2;
    const int n_pos_embd = (((((n_frames + 1) / 2) + 1) / 2 + 1) / 2) * 2 - 1;
    GGML_ASSERT(model.position_embeddings->ne[1] >= n_pos);

    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, n_pos_embd);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);
    ggml_build_forward_expand(gf, pos_emb);

    ggml_tensor * inp = build_inp_raw(1);

    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    // pre encode, conv subsampling
    {
        // layer.0 - conv2d
        cur = ggml_conv_2d(ctx0, model.pre_encode_conv_X_w[0], cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[0]);
        cb(cur, "conformer.pre_encode.conv.{}", 0);

        // layer.1 - relu
        cur = ggml_relu_inplace(ctx0, cur);

        // layer.2 conv2d dw
        cur = ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[2], cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[2]);
        cb(cur, "conformer.pre_encode.conv.{}", 2);

        // layer.3 conv2d
        cur = ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[3], cur, 1, 1, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[3]);
        cb(cur, "conformer.pre_encode.conv.{}", 3);

        // layer.4 - relu
        cur = ggml_relu_inplace(ctx0, cur);

        // layer.5 conv2d dw
        cur = ggml_conv_2d_dw_direct(ctx0, model.pre_encode_conv_X_w[5], cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[5]);
        cb(cur, "conformer.pre_encode.conv.{}", 5);

        // layer.6 conv2d
        cur = ggml_conv_2d_direct(ctx0, model.pre_encode_conv_X_w[6], cur, 1, 1, 0, 0, 1, 1);
        cur = ggml_add(ctx0, cur, model.pre_encode_conv_X_b[6]);
        cb(cur, "conformer.pre_encode.conv.{}", 6);

        // layer.7 - relu
        cur = ggml_relu_inplace(ctx0, cur);

        // flatten channel and frequency axis
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);

        // calculate out
        cur = build_mm(model.pre_encode_out_w, cur);
        cur = ggml_add(ctx0, cur, model.pre_encode_out_b);
        cb(cur, "conformer.pre_encode.out", -1);
    }

    // pos_emb
    cb(pos_emb, "pos_emb", -1);

    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];

        auto * residual = cur;

        cb(cur, "layer.in", il);

        // feed_forward1
        cur = build_norm(cur, layer.ff_norm_w, layer.ff_norm_b, NORM_TYPE_NORMAL, 1e-5, il);
        cb(cur, "conformer.layers.{}.norm_feed_forward1", il);

        cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, nullptr, nullptr, layer.ff_down_w, layer.ff_down_b, FFN_SILU,
                        il);
        cb(cur, "conformer.layers.{}.feed_forward1.linear2", il);

        const auto fc_factor = 0.5f;
        residual             = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, fc_factor));

        // self-attention
        {
            cur = build_norm(residual, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, 1e-5, il);
            cb(cur, "conformer.layers.{}.norm_self_att", il);

            ggml_tensor * Qcur     = build_mm(layer.q_w, cur);
            Qcur                   = ggml_add(ctx0, Qcur, layer.q_b);
            Qcur                   = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, Qcur->ne[1]);
            ggml_tensor * Q_bias_u = ggml_add(ctx0, Qcur, layer.pos_bias_u);
            Q_bias_u               = ggml_permute(ctx0, Q_bias_u, 0, 2, 1, 3);
            ggml_tensor * Q_bias_v = ggml_add(ctx0, Qcur, layer.pos_bias_v);
            Q_bias_v               = ggml_permute(ctx0, Q_bias_v, 0, 2, 1, 3);

            // TODO @ngxson : some cont can/should be removed when ggml_mul_mat support these cases
            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            Kcur               = ggml_add(ctx0, Kcur, layer.k_b);
            Kcur               = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, Kcur->ne[1]);
            Kcur               = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            ggml_tensor * Vcur = build_mm(layer.v_w, cur);
            Vcur               = ggml_add(ctx0, Vcur, layer.v_b);
            Vcur               = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, Vcur->ne[1]);
            Vcur               = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 1, 2, 0, 3));

            // build_attn won't fit due to matrix_ac and matrix_bd separation
            ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, Q_bias_u, Kcur);
            matrix_ac               = ggml_cont(ctx0, ggml_permute(ctx0, matrix_ac, 1, 0, 2, 3));
            cb(matrix_ac, "conformer.layers.{}.self_attn.id3", il);

            auto * p = build_mm(layer.linear_pos_w, pos_emb);
            cb(p, "conformer.layers.{}.self_attn.linear_pos", il);
            p = ggml_reshape_3d(ctx0, p, d_head, n_head, p->ne[1]);
            p = ggml_permute(ctx0, p, 0, 2, 1, 3);

            auto * matrix_bd = ggml_mul_mat(ctx0, Q_bias_v, p);
            matrix_bd        = ggml_cont(ctx0, ggml_permute(ctx0, matrix_bd, 1, 0, 2, 3));

            // rel shift
            {
                const auto pos_len = matrix_bd->ne[0];
                const auto q_len   = matrix_bd->ne[1];
                const auto h       = matrix_bd->ne[2];
                matrix_bd          = ggml_pad(ctx0, matrix_bd, 1, 0, 0, 0);
                matrix_bd          = ggml_roll(ctx0, matrix_bd, 1, 0, 0, 0);
                matrix_bd          = ggml_reshape_3d(ctx0, matrix_bd, q_len, pos_len + 1, h);
                matrix_bd          = ggml_view_3d(ctx0, matrix_bd, q_len, pos_len, h, matrix_bd->nb[1],
                                                        matrix_bd->nb[2], matrix_bd->nb[0] * q_len);
                matrix_bd          = ggml_cont_3d(ctx0, matrix_bd, pos_len, q_len, h);
            }

            matrix_bd     = ggml_view_3d(ctx0, matrix_bd, matrix_ac->ne[0], matrix_bd->ne[1],
                                               matrix_bd->ne[2], matrix_bd->nb[1], matrix_bd->nb[2], 0);
            auto * scores = ggml_add(ctx0, matrix_ac, matrix_bd);
            scores        = ggml_scale(ctx0, scores, 1.0f / std::sqrt(d_head));
            cb(scores, "conformer.layers.{}.self_attn.id0", il);

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);
            ggml_tensor * x    = ggml_mul_mat(ctx0, attn, Vcur);
            x                  = ggml_permute(ctx0, x, 2, 0, 1, 3);
            x                  = ggml_cont_2d(ctx0, x, x->ne[0] * x->ne[1], x->ne[2]);

            ggml_tensor * out = build_mm(layer.o_w, x);
            out               = ggml_add(ctx0, out, layer.o_b);
            cb(out, "conformer.layers.{}.self_attn.linear_out", il);

            cur = out;
        }

        residual = ggml_add(ctx0, residual, cur);
        cur      = build_norm(residual, layer.norm_conv_w, layer.norm_conv_b, NORM_TYPE_NORMAL, 1e-5, il);
        cb(cur, "conformer.layers.{}.norm_conv", il);

        // conv
        {
            auto * x = cur;
            x = build_mm(layer.conv_pw1_w, x);
            x = ggml_add(ctx0, x, layer.conv_pw1_b);
            cb(x, "conformer.layers.{}.conv.pointwise_conv1", il);

            // ggml_glu doesn't support sigmoid
            // TODO @ngxson : support this ops in ggml
            {
                int64_t       d    = x->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0]));
                x                  = ggml_mul(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x                  = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }

            // use ggml_ssm_conv for f32 precision
            x = ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = ggml_roll(ctx0, x, 4, 0, 0, 0);
            x = ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            x = ggml_add(ctx0, x, layer.conv_dw_b);

            x = ggml_add(ctx0, ggml_mul(ctx0, x, layer.conv_norm_w), layer.conv_norm_b);
            x = ggml_silu(ctx0, x);

            // pointwise_conv2
            x = build_mm(layer.conv_pw2_w, x);
            x = ggml_add(ctx0, x, layer.conv_pw2_b);

            cur = x;
        }

        residual = ggml_add(ctx0, residual, cur);

        cur = build_norm(residual, layer.ff_norm_1_w, layer.ff_norm_1_b, NORM_TYPE_NORMAL, 1e-5, il);
        cb(cur, "conformer.layers.{}.norm_feed_forward2", il);

        cur = build_ffn(cur, layer.ff_up_1_w, layer.ff_up_1_b, nullptr, nullptr, layer.ff_down_1_w, layer.ff_down_1_b,
                        FFN_SILU, il);  // TODO(tarek): read activation for ffn from hparams
        cb(cur, "conformer.layers.{}.feed_forward2.linear2", il);

        residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, fc_factor));
        cb(residual, "conformer.layers.{}.conv.id", il);

        cur = build_norm(residual, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, 1e-5, il);
        cb(cur, "conformer.layers.{}.norm_out", il);
    }

    // audio adapter
    cur = build_norm(cur, model.mm_0_w, model.mm_0_b, NORM_TYPE_NORMAL, 1e-5, -1);
    cb(cur, "audio_adapter.model.{}", 0);
    cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_3_w, model.mm_3_b, FFN_GELU_ERF, -1);

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

ggml_cgraph * clip_graph_gemma4a::build() {
    const float res_weight = 0.5f;
    const float norm_eps   = 1e-6f;

    // 1. Input
    ggml_tensor * inp = build_inp_raw(1);
    auto * cur = ggml_cont(ctx0, ggml_transpose(ctx0, inp));

    // 2. Subsampling Conv2D (symmetric padding=1, matching PyTorch)
    {
        for (int i = 0; i < 2; i++) {
            cur = ggml_conv_2d(ctx0, model.sscp_conv_w[i], cur, 2, 2, 1, 1, 1, 1);
            if (model.sscp_conv_b[i]) {
                cur = ggml_add(ctx0, cur, model.sscp_conv_b[i]);
            }
            // nn.LayerNorm(channels): permute ch to ne[0], normalize, permute back
            if (model.sscp_norm_w[i]) {
                cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
                cur = ggml_norm(ctx0, cur, norm_eps);
                cur = ggml_mul(ctx0, cur, model.sscp_norm_w[i]);
                cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
            }
            cur = ggml_relu(ctx0, cur);
        }
        // Flatten [freq, time, ch, 1] -> [ch*freq, time]
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2]);
        if (model.sscp_inp_proj_w) {
            cur = build_mm(model.sscp_inp_proj_w, cur);
            if (model.sscp_inp_proj_b) {
                cur = ggml_add(ctx0, cur, model.sscp_inp_proj_b);
            }
        }
    }

    const int64_t n_pos = cur->ne[1];

    // Chunked local attention parameters
    const int64_t C  = 12;                              // chunk_size
    const int64_t P  = 12;                              // max_past_horizon (context_left - 1)
    const int64_t S  = C + P;                           // context_size = 24
    const int64_t R  = P + 1;                           // RPE positions = 13
    const int64_t B  = (n_pos + C - 1) / C;            // num_blocks
    const int64_t Np = B * C;                           // padded sequence length
    const int64_t pad_seq = Np - n_pos;

    // Input tensors: blocked RPE and blocked attention mask
    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_head * d_head, R);
    ggml_set_name(pos_emb, "pos_emb");
    ggml_set_input(pos_emb);

    ggml_tensor * kq_mask = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, S, C, B);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    // 3. Conformer Blocks
    for (int il = 0; il < hparams.n_layer; il++) {
        const auto & layer = model.layers[il];
        auto * residual = cur;

        // FFN 1 (half-step)
        if (layer.ff_norm_w && layer.ff_up_w && layer.ff_down_w) {
            cur = build_norm(cur, layer.ff_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_w, nullptr, nullptr, nullptr,
                layer.ff_down_w, nullptr, FFN_SILU, il);
            if (layer.ff_post_norm_w) {
                cur = build_norm(cur, layer.ff_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        // Chunked local self-attention with RPE
        if (layer.q_w && layer.k_w && layer.v_w && layer.o_w) {
            const float q_scale = (1.0f / sqrtf((float)d_head)) / logf(2.0f);
            const float k_scale = logf(1.0f + expf(1.0f)) / logf(2.0f);
            const float softcap = 50.0f;

            ggml_tensor * attn_norm_w = layer.attn_pre_norm_w ? layer.attn_pre_norm_w : layer.ln_1_w;
            cur = attn_norm_w
                ? build_norm(residual, attn_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il)
                : residual;

            ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            ggml_tensor * Vcur = build_mm(layer.v_w, cur);

            // [n_embd, n_pos] -> [D, H, N]
            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            // Q/K scaling
            Qcur = ggml_scale(ctx0, Qcur, q_scale);
            if (layer.per_dim_scale_w) {
                Qcur = ggml_mul(ctx0, Qcur, ggml_reshape_3d(ctx0, layer.per_dim_scale_w, d_head, 1, 1));
            }
            Kcur = ggml_scale(ctx0, Kcur, k_scale);
            if (layer.per_dim_k_scale_w) {
                Kcur = ggml_mul(ctx0, Kcur, ggml_reshape_3d(ctx0, layer.per_dim_k_scale_w, d_head, 1, 1));
            }

            // Q blocking: [D, H, N] -> pad to Np -> reshape [D, H, C, B]
            // ggml permute: ne[ax_i] = src->ne[i], so (0,3,1,2) sends H->3, C->1, B->2
            Qcur = ggml_pad(ctx0, Qcur, 0, 0, pad_seq, 0);          // [D, H, Np]
            Qcur = ggml_reshape_4d(ctx0, Qcur, d_head, n_head, C, B); // [D, H, C, B]
            Qcur = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 0, 3, 1, 2)); // [D, C, B, H]

            // K/V block context extraction via overlapping view:
            // Pad to S*B elements, roll right by P to create left-padding,
            // then view with stride C in the block dimension (overlapping windows).
            auto extract_blocks = [&](ggml_tensor * t) -> ggml_tensor * {
                // [D, H, N] -> pad to S*B -> roll right by P -> cont (materialize)
                const int64_t pad_kv = S * B - n_pos;
                t = ggml_pad(ctx0, t, 0, 0, pad_kv, 0);     // [D, H, S*B]
                t = ggml_roll(ctx0, t, 0, 0, P, 0);          // left-pad by P
                t = ggml_cont(ctx0, t);                       // materialize roll (removes view offset)
                // Overlapping view: stride for B dim is C positions, not S
                // ne = [D, H, S, B], data_size = D*H*S*B*sizeof = source_nbytes (exact fit)
                // nb1=D*sizeof, nb2=D*H*sizeof, nb3=C*D*H*sizeof (overlap: C < S)
                t = ggml_view_4d(ctx0, t, d_head, n_head, S, B,
                    t->nb[1], t->nb[2], C * t->nb[2], 0);
                t = ggml_cont(ctx0, t);                       // materialize overlapping windows
                return t;
            };

            ggml_tensor * Kblk = extract_blocks(Kcur);
            // [D, H, S, B] -> [D, S, B, H] via permute(0,3,1,2)
            Kblk = ggml_cont(ctx0, ggml_permute(ctx0, Kblk, 0, 3, 1, 2));

            ggml_tensor * Vblk = extract_blocks(Vcur);
            // [D, H, S, B] -> [S, D, B, H] via permute(1,3,0,2)
            Vblk = ggml_cont(ctx0, ggml_permute(ctx0, Vblk, 1, 3, 0, 2));

            // Content attention: Q @ K^T
            // Kblk=[D,S,B,H], Qcur=[D,C,B,H] -> mul_mat contracts on D -> [S,C,B,H]
            ggml_tensor * matrix_ac = ggml_mul_mat(ctx0, Kblk, Qcur);

            // Relative position attention
            if (layer.attn_k_rel_w) {
                // RPE: [n_embd, R] -> project -> [D, H, R] -> [D, R, H]
                auto * p = ggml_mul_mat(ctx0, layer.attn_k_rel_w, pos_emb);
                p = ggml_reshape_3d(ctx0, p, d_head, n_head, R);
                p = ggml_cont(ctx0, ggml_permute(ctx0, p, 0, 2, 1, 3)); // [D, R, H]

                // Q_flat @ RPE^T: [D, C*B, H] @ [D, R, H] -> [R, C*B, H]
                auto * Q_flat = ggml_reshape_3d(ctx0, Qcur, d_head, C * B, n_head);
                auto * matrix_bd = ggml_mul_mat(ctx0, p, Q_flat);       // [R, C*B, H]
                matrix_bd = ggml_reshape_4d(ctx0, matrix_bd, R, C, B, n_head); // [R, C, B, H]

                // Blocked relative shift (appendix B of Transformer-XL)
                {
                    matrix_bd = ggml_pad(ctx0, matrix_bd, S + 1 - R, 0, 0, 0); // [S+1, C, B, H]
                    matrix_bd = ggml_reshape_3d(ctx0, matrix_bd, (S + 1) * C, B, n_head);
                    matrix_bd = ggml_view_3d(ctx0, matrix_bd,
                        C * S, B, n_head,
                        matrix_bd->nb[1], matrix_bd->nb[2], 0);
                    matrix_bd = ggml_cont(ctx0, matrix_bd);              // [C*S, B, H]
                    matrix_bd = ggml_reshape_4d(ctx0, matrix_bd, S, C, B, n_head); // [S, C, B, H]
                }

                matrix_ac = ggml_add(ctx0, matrix_ac, matrix_bd);
            }

            auto * scores = matrix_ac; // [S, C, B, H]

            // Softcap
            scores = ggml_scale(ctx0, scores, 1.0f / softcap);
            scores = ggml_tanh(ctx0, scores);
            scores = ggml_scale(ctx0, scores, softcap);

            // Blocked attention mask: [S, C, B] broadcasts over H
            scores = ggml_add(ctx0, scores, kq_mask);

            ggml_tensor * attn = ggml_soft_max(ctx0, scores);

            // attn @ V: [S,C,B,H] @ [S,D,B,H] -> [D,C,B,H]
            ggml_tensor * x = ggml_mul_mat(ctx0, Vblk, attn);

            // [D,C,B,H] -> [D,H,C,B] via permute(0,2,3,1) -> flatten -> trim
            x = ggml_cont(ctx0, ggml_permute(ctx0, x, 0, 2, 3, 1));
            x = ggml_cont_2d(ctx0, x, d_head * n_head, C * B);
            if (pad_seq > 0) {
                x = ggml_view_2d(ctx0, x, d_head * n_head, n_pos, x->nb[1], 0);
                x = ggml_cont(ctx0, x);
            }

            x = build_mm(layer.o_w, x);
            if (layer.o_b) { x = ggml_add(ctx0, x, layer.o_b); }

            if (layer.attn_post_norm_w) {
                x = build_norm(x, layer.attn_post_norm_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, x);
        }

        // Convolution Module
        if (layer.norm_conv_w && layer.conv_pw1_w && layer.conv_dw_w && layer.conv_pw2_w) {
            cur = build_norm(residual, layer.norm_conv_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            auto * x = build_mm(layer.conv_pw1_w, cur);

            // GLU
            {
                int64_t d = x->ne[0] / 2;
                ggml_tensor * gate = ggml_sigmoid(ctx0,
                    ggml_cont(ctx0, ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], d * x->nb[0])));
                x = ggml_mul(ctx0,
                    ggml_view_2d(ctx0, x, d, x->ne[1], x->nb[1], 0), gate);
                x = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            }

            // Causal depthwise Conv1D via ggml_ssm_conv (pad+roll for left-only padding).
            x = ggml_pad(ctx0, x, 4, 0, 0, 0);
            x = ggml_roll(ctx0, x, 4, 0, 0, 0);
            x = ggml_ssm_conv(ctx0, x, layer.conv_dw_w);
            if (layer.conv_dw_b) {
                x = ggml_add(ctx0, x, layer.conv_dw_b);
            }

            if (layer.conv_norm_w) {
                x = ggml_rms_norm(ctx0, x, norm_eps);
                x = ggml_mul(ctx0, x, layer.conv_norm_w);
            }
            x = ggml_silu(ctx0, x);
            x = build_mm(layer.conv_pw2_w, x);
            residual = ggml_add(ctx0, residual, x);
        }

        // FFN 2 (half-step)
        if (layer.ff_norm_1_w && layer.ff_up_1_w && layer.ff_down_1_w) {
            cur = build_norm(residual, layer.ff_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            cur = build_ffn(cur,
                layer.ff_up_1_w, nullptr, nullptr, nullptr,
                layer.ff_down_1_w, nullptr, FFN_SILU, il);
            if (layer.ff_post_norm_1_w) {
                cur = build_norm(cur, layer.ff_post_norm_1_w, nullptr, NORM_TYPE_RMS, norm_eps, il);
            }
            residual = ggml_add(ctx0, residual, ggml_scale(ctx0, cur, res_weight));
        }

        // Layer output norm
        cur = layer.ln_2_w
            ? build_norm(residual, layer.ln_2_w, nullptr, NORM_TYPE_RMS, norm_eps, il)
            : residual;

    }

    // 4. Output Projection
    if (model.audio_out_proj_w) {
        cur = build_mm(model.audio_out_proj_w, cur);
        if (model.audio_out_proj_b) {
            cur = ggml_add(ctx0, cur, model.audio_out_proj_b);
        }
    }

    // 5. Audio Multimodal Embedder
    cur = ggml_rms_norm(ctx0, cur, norm_eps);
    if (model.mm_soft_emb_norm_w) {
        cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
    }
    if (model.mm_input_proj_w) {
        cur = build_mm(model.mm_input_proj_w, cur);
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}

ggml_tensor * clip_graph_gemma4a::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    auto it = model.clamp_info_map.find(w->name);
    if (it == model.clamp_info_map.end()) {
        return ggml_mul_mat(ctx0, w, x);
    }
    const auto & ci = it->second;
    ggml_tensor * clamped = ggml_clamp(ctx0, x, ci.inp_min, ci.inp_max);
    ggml_tensor * out = ggml_mul_mat(ctx0, w, clamped);
    return ggml_clamp(ctx0, out, ci.out_min, ci.out_max);
}

static ggml_tensor * add_channel_bias(
        ggml_context * ctx0,
        ggml_tensor * x_whcb,
        ggml_tensor * b_c) {
    if (!b_c) {
        return x_whcb;
    }
    ggml_tensor * b4 = ggml_reshape_4d(ctx0, b_c, 1, 1, b_c->ne[0], 1);
    return ggml_add(ctx0, x_whcb, b4);
}

static ggml_tensor * mul_channel_weight(
        ggml_context * ctx0,
        ggml_tensor * x_whcb,
        ggml_tensor * w_c) {
    if (!w_c) {
        return x_whcb;
    }
    ggml_tensor * w4 = ggml_reshape_4d(ctx0, w_c, 1, 1, w_c->ne[0], 1);
    return ggml_mul(ctx0, x_whcb, w4);
}

ggml_tensor * clip_graph_yasa2::layer_norm_channels(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b, float eps) {
    // Match HF ConvNextLayerNorm(channels_first):
    // u = mean_c(x), s = mean_c((x-u)^2), x = (x-u)/sqrt(s+eps)
    // cast back to input dtype before affine.
    ggml_tensor * cur = ggml_permute(ctx0, inp, 2, 1, 0, 3); // [W,H,C,B] -> [C,H,W,B]
    cur = ggml_cont(ctx0, cur);

    ggml_tensor * u = ggml_mean(ctx0, cur);                 // [1,H,W,B]
    ggml_tensor * xm = ggml_sub(ctx0, cur, u);              // [C,H,W,B]

    ggml_tensor * s = ggml_mul(ctx0, xm, xm);               // [C,H,W,B]
    s = ggml_mean(ctx0, s);                                 // [1,H,W,B]
    s = ggml_clamp(ctx0, s, eps, 1e30f);                    // avoid div-by-zero in no-alloc warmup
    s = ggml_sqrt(ctx0, s);                                 // [1,H,W,B]

    ggml_tensor * xhat = ggml_div(ctx0, xm, s);             // [C,H,W,B]
    xhat = ggml_permute(ctx0, xhat, 2, 1, 0, 3);            // [W,H,C,B]
    xhat = ggml_cont(ctx0, xhat);
    xhat = mul_channel_weight(ctx0, xhat, w);
    xhat = add_channel_bias(ctx0, xhat, b);
    return xhat;
}

ggml_tensor * clip_graph_yasa2::convnext_grn(ggml_tensor * inp, ggml_tensor * w, ggml_tensor * b) {
    // Exact ConvNeXtV2 GRN:
    // Gx = ||x||_2 over spatial dims (W,H), Nx = Gx / (mean_c(Gx) + eps)
    // y  = w * (x * Nx) + b + x
    const int64_t wdim = inp->ne[0];
    const int64_t hdim = inp->ne[1];
    const int64_t cdim = inp->ne[2];
    const int64_t bdim = inp->ne[3];

    // Keep GRN math in fp32 for stability; fp16/bf16 accumulation can drift.
    ggml_tensor * sq = ggml_mul(ctx0, inp, inp);
    ggml_tensor * sq_flat = ggml_reshape_4d(ctx0, sq, wdim * hdim, cdim, 1, bdim);   // [WH,C,1,B]
    ggml_tensor * gx = ggml_sum_rows(ctx0, sq_flat);                                   // [1,C,1,B]
    gx = ggml_sqrt(ctx0, gx);                                                           // [1,C,1,B]

    ggml_tensor * gx_ch_first = ggml_permute(ctx0, gx, 1, 0, 2, 3);                    // [C,1,1,B]
    gx_ch_first = ggml_cont(ctx0, gx_ch_first);
    ggml_tensor * gx_mean = ggml_mean(ctx0, gx_ch_first);                              // [1,1,1,B]

    gx_mean = ggml_clamp(ctx0, gx_mean, 1e-6f, 1e30f);                                  // approx +eps, warmup-safe
    ggml_tensor * nx = ggml_div(ctx0, gx, gx_mean);                                    // [1,C,1,B]
    nx = ggml_permute(ctx0, nx, 0, 2, 1, 3);                                            // [1,1,C,B]
    nx = ggml_cont(ctx0, nx);

    ggml_tensor * xnx = ggml_mul(ctx0, inp, nx);
    xnx = mul_channel_weight(ctx0, xnx, w);
    xnx = add_channel_bias(ctx0, xnx, b);
    return ggml_add(ctx0, inp, xnx);
}

ggml_cgraph * clip_graph_yasa2::build() {
    ggml_tensor * cur = build_inp_raw();

    // Patch embedding Conv2d(kernel=4, stride=4)
    cur = ggml_conv_2d(ctx0, model.yasa_patch_w, cur, patch_size, patch_size, 0, 0, 1, 1);
    cur = add_channel_bias(ctx0, cur, model.yasa_patch_b);
    ggml_set_name(cur, "yasa2_patch_conv_out");
    cb(cur, "yasa2_patch_conv_out", -1);
    cur = layer_norm_channels(cur, model.yasa_patch_ln_w, model.yasa_patch_ln_b, eps);
    ggml_set_name(cur, "yasa2_patch_ln_out");
    cb(cur, "yasa2_patch_ln_out", -1);

    // ConvNeXt stages
    for (size_t s = 0; s < model.yasa_stages.size(); ++s) {
        const auto & stage = model.yasa_stages[s];

        if (stage.down_conv_w) {
            cur = layer_norm_channels(cur, stage.down_ln_w, stage.down_ln_b, eps);
            cur = ggml_conv_2d(ctx0, stage.down_conv_w, cur, 2, 2, 0, 0, 1, 1);
            cur = add_channel_bias(ctx0, cur, stage.down_conv_b);
            ggml_format_name(cur, "yasa2_stage%zu_down_out", s);
        }

        for (size_t bi = 0; bi < stage.blocks.size(); ++bi) {
            const auto & blk = stage.blocks[bi];
            ggml_tensor * res = cur;

            ggml_tensor * x = ggml_conv_2d_dw(ctx0, blk.dw_w, cur, 1, 1, 3, 3, 1, 1);
            x = add_channel_bias(ctx0, x, blk.dw_b);
            x = layer_norm_channels(x, blk.ln_w, blk.ln_b, eps);

            // pwconv1/pwconv2 are HF Linear layers over channels; implement via matmul on tokens.
            const int64_t w = x->ne[0];
            const int64_t h = x->ne[1];
            const int64_t b = x->ne[3];

            ggml_tensor * tok = ggml_reshape_3d(ctx0, x, w * h, x->ne[2], b); // [T,C,B]
            tok = ggml_permute(ctx0, tok, 1, 0, 2, 3);                        // [C,T,B]
            tok = ggml_cont(ctx0, tok);

            tok = ggml_mul_mat(ctx0, blk.pw1_w, tok);                         // [4C,T,B]
            if (blk.pw1_b) {
                ggml_tensor * b1 = ggml_reshape_3d(ctx0, blk.pw1_b, blk.pw1_b->ne[0], 1, 1); // [4C,1,1]
                tok = ggml_add(ctx0, tok, b1);
            }
            x = ggml_permute(ctx0, tok, 1, 0, 2, 3);                         // [T,4C,B]
            x = ggml_cont(ctx0, x);
            x = ggml_reshape_4d(ctx0, x, w, h, tok->ne[0], b);               // [W,H,4C,B]
            x = ggml_gelu_erf(ctx0, x);
            x = convnext_grn(x, blk.grn_w, blk.grn_b);

            tok = ggml_reshape_3d(ctx0, x, w * h, x->ne[2], b);              // [T,4C,B]
            tok = ggml_permute(ctx0, tok, 1, 0, 2, 3);                       // [4C,T,B]
            tok = ggml_cont(ctx0, tok);

            tok = ggml_mul_mat(ctx0, blk.pw2_w, tok);                        // [C,T,B]
            if (blk.pw2_b) {
                ggml_tensor * b2 = ggml_reshape_3d(ctx0, blk.pw2_b, blk.pw2_b->ne[0], 1, 1); // [C,1,1]
                tok = ggml_add(ctx0, tok, b2);
            }
            x = ggml_permute(ctx0, tok, 1, 0, 2, 3);                         // [T,C,B]
            x = ggml_cont(ctx0, x);
            x = ggml_reshape_4d(ctx0, x, w, h, tok->ne[0], b);               // [W,H,C,B]

            cur = ggml_add(ctx0, res, x);
            ggml_format_name(cur, "yasa2_stage%zu_blk%zu_out", s, bi);
        }
    }

    // HF path adds vision position embeddings BEFORE adaptive pooling.
    const int64_t pre_w = cur->ne[0];
    const int64_t pre_h = cur->ne[1];
    ggml_tensor * tokens_pre = ggml_reshape_3d(ctx0, cur, pre_w * pre_h, cur->ne[2], cur->ne[3]); // [T,C,B]
    tokens_pre = ggml_permute(ctx0, tokens_pre, 1, 0, 2, 3); // [C,T,B]
    tokens_pre = ggml_cont(ctx0, tokens_pre);
    if (model.yasa_vision_pos_embed && tokens_pre->ne[1] == model.yasa_vision_pos_embed->ne[1]) {
        const int64_t n_ch = model.yasa_vision_pos_embed->ne[0];
        const int64_t n_tokens = model.yasa_vision_pos_embed->ne[1];
        ggml_tensor * pos = ggml_reshape_3d(ctx0, model.yasa_vision_pos_embed, (int) n_ch, (int) n_tokens, 1);
        tokens_pre = ggml_add(ctx0, tokens_pre, pos);
    }
    cur = ggml_permute(ctx0, tokens_pre, 1, 0, 2, 3); // [T,C,B]
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_4d(ctx0, cur, pre_w, pre_h, cur->ne[1], cur->ne[2]); // [W,H,C,B]

    // AdaptiveAvgPool2d target is 8x8 for real inputs, but warmup can use tiny images.
    const int pooled_w = std::min(8, (int) cur->ne[0]);
    const int pooled_h = std::min(8, (int) cur->ne[1]);
    const int kw = std::max(1, (int) cur->ne[0] / pooled_w);
    const int kh = std::max(1, (int) cur->ne[1] / pooled_h);
    cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, kw, kh, kw, kh, 0, 0);

    // [W,H,C,B] -> [C,T,B]
    ggml_tensor * tokens = ggml_reshape_3d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2], cur->ne[3]);
    tokens = ggml_permute(ctx0, tokens, 1, 0, 2, 3);
    tokens = ggml_cont(ctx0, tokens);
    cb(tokens, "yasa2_tokens", -1);

    GGML_ASSERT(model.mm_0_w && model.mm_2_w);
    ggml_tensor * embeddings = build_ffn(
        tokens,
        model.mm_0_w, model.mm_0_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF,
        -1);
    cb(embeddings, "yasa2_emb", -1);

    ggml_build_forward_expand(gf, embeddings);
    return gf;
}

// Implementation based on approach suggested by Acly
// See: https://github.com/ggml-org/llama.cpp/pull/17383#issuecomment-3554227091
static ggml_tensor * window_partition(ggml_context * ctx0, ggml_tensor * x, const int window) {
    auto [c, w, h, b] = x->ne;
    // same as
    // x = ggml_win_part(m, x, window);
    // x = ggml_reshape_3d(m, x, c, window * window, x->ne[3]);

    const int64_t px  = (window - w % window) % window;
    const int64_t py  = (window - h % window) % window;
    const int64_t npw = (w + px) / window;
    const int64_t nph = (h + py) / window;

    ggml_tensor * cur = x;
    if (px > 0 || py > 0) {
        cur = ggml_pad(ctx0, cur, 0, static_cast<int>(px), static_cast<int>(py), 0);
    }
    cur = ggml_reshape_4d(ctx0, cur, c * window, npw, window, nph * b);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_4d(ctx0, cur, c, window, window, npw * nph * b);
    return cur;
}

// Implementation based on approach suggested by Acly
// See: https://github.com/ggml-org/llama.cpp/pull/17383#issuecomment-3554227091
static ggml_tensor * window_unpartition(ggml_context * ctx0,
                                        ggml_tensor *  x,
                                        const int      w,
                                        const int      h,
                                        const int      window) {
    const int64_t c = x->ne[0];
    // same as
    // x = ggml_reshape_4d(m, x, c, window, window, x->ne[2]);
    // x = ggml_win_unpart(m, x, w, h, window);

    const int64_t px  = (window - w % window) % window;
    const int64_t py  = (window - h % window) % window;
    const int64_t npw = (w + px) / window;
    const int64_t nph = (h + py) / window;

    const int64_t b = x->ne[3] / (npw * nph);
    ggml_tensor * cur = x;
    cur = ggml_reshape_4d(ctx0, cur, c * window, window, npw, nph * b);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 0, 2, 1, 3));
    cur = ggml_reshape_4d(ctx0, cur, c, w + px, h + py, b);
    cur = ggml_view_4d(ctx0, cur, cur->ne[0], w, h, cur->ne[3], cur->nb[1], cur->nb[2], cur->nb[3], 0);
    cur = ggml_cont(ctx0, cur);
    return cur;
}

static ggml_tensor * get_rel_pos(ggml_context * ctx0,
                                 ggml_tensor *  rel_pos,  // [L, C]
                                 ggml_tensor *  indices,  // [q_size, k_size]
                                 const int      q_size,
                                 const int      k_size) {
    const int64_t C = rel_pos->ne[0];  // channels
    const int64_t L = rel_pos->ne[1];  // length

    GGML_ASSERT(indices != nullptr);
    GGML_ASSERT(indices->type == GGML_TYPE_I32);
    GGML_ASSERT(indices->ne[0] == k_size);
    GGML_ASSERT(indices->ne[1] == q_size);

    const auto    max_rel_dist = 2 * std::max(q_size, k_size) - 1;
    ggml_tensor * cur          = rel_pos;

    if (max_rel_dist != L) {
        // Linear interpolation
        const int64_t ne0 = cur->ne[0];
        const int64_t ne1 = cur->ne[1];
        const int64_t ne2 = cur->ne[2];
        const int64_t ne3 = cur->ne[3];

        cur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3)), ne1, 1, ne0 * ne2 * ne3);
        cur = ggml_reshape_4d(
            ctx0, ggml_interpolate(ctx0, cur, max_rel_dist, 1, ne0 * ne2 * ne3, 1, GGML_SCALE_MODE_BILINEAR),
            max_rel_dist, ne0, ne2, ne3);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    }

    // Flatten indices to 1D for ggml_get_rows
    const int qk = q_size * k_size;

    cur = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, cur, ggml_reshape_1d(ctx0, indices, qk)), C, k_size, q_size);

    return cur;  // [C, k_size, q_size]
}

ggml_cgraph * clip_graph_deepseekocr::build() {
    // patch embedding
    ggml_tensor * inp_raw = build_inp_raw();

    ggml_tensor * sam_out;
    // Building SAM
    {
        const int n_embd  = hparams.sam_n_embd;
        const int n_layer = hparams.sam_n_layer;
        const int n_heads = hparams.sam_n_head;
        const int d_heads = n_embd / n_heads;
        const int window  = hparams.attn_window_size;

        ggml_tensor * inpL;

        inpL = ggml_conv_2d_sk_p0(ctx0, model.patch_embed_proj_w, inp_raw);
        inpL = ggml_add(ctx0, inpL, ggml_reshape_3d(ctx0, model.patch_embed_proj_b, 1, 1, n_embd));
        inpL = ggml_cont(ctx0, ggml_permute(ctx0, inpL, 1, 2, 0, 3));

        ggml_tensor * rel_pos_indices_local;
        ggml_tensor * rel_pos_indices_global;

        rel_pos_indices_local  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, window, window);
        rel_pos_indices_global = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, inpL->ne[1], inpL->ne[2]);
        ggml_set_name(rel_pos_indices_local, "rel_pos_indices_local");
        ggml_set_name(rel_pos_indices_global, "rel_pos_indices_global");
        ggml_set_input(rel_pos_indices_local);
        ggml_set_input(rel_pos_indices_global);

        ggml_tensor * cur;
        const auto    tgt_size = inpL->ne[1];
        const auto    str_size = model.pos_embed->ne[1];

        if (str_size != tgt_size) {
            ggml_tensor * old_pos_embed = nullptr;
            old_pos_embed               = ggml_cont(ctx0, ggml_permute(ctx0, model.pos_embed, 2, 0, 1, 3));
            ggml_tensor * new_pos_embed =
                ggml_interpolate(ctx0, old_pos_embed, tgt_size, tgt_size, n_embd, 1, GGML_SCALE_MODE_BICUBIC);
            new_pos_embed = ggml_cont(ctx0, ggml_permute(ctx0, new_pos_embed, 1, 2, 0, 3));
            cur           = ggml_add(ctx0, inpL, new_pos_embed);
        } else {
            cur = ggml_add(ctx0, inpL, model.pos_embed);
        }

        // loop over layers
        for (int il = 0; il < n_layer; il++) {
            auto &        layer    = model.sam_layers[il];
            ggml_tensor * shortcut = cur;

            // layernorm1
            cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);

            const int64_t w0 = cur->ne[1];
            const int64_t h0 = cur->ne[2];

            ggml_tensor * indices;

            if (hparams.is_global_attn(il)) {
                indices = rel_pos_indices_global;
            } else {
                // local attention layer - apply window partition
                cur     = window_partition(ctx0, cur, window);
                indices = rel_pos_indices_local;
            }

            const int64_t W = cur->ne[1];
            const int64_t H = cur->ne[2];
            // self-attention
            {
                const int B = cur->ne[3];

                cur = ggml_mul_mat(ctx0, layer.qkv_w, cur);
                cur = ggml_add(ctx0, cur, layer.qkv_b);
                cur = ggml_cont(ctx0, cur);  // Ensure tensor is contiguous before reshape
                cur = ggml_reshape_4d(ctx0, cur, n_embd, 3, W * H, B);

                ggml_tensor * Q;
                ggml_tensor * K;
                ggml_tensor * V;

                Q = ggml_view_3d(ctx0, cur, n_embd, W * H, B, cur->nb[2], cur->nb[3], 0 * cur->nb[1]);
                Q = ggml_reshape_4d(ctx0, ggml_cont(ctx0, Q), d_heads, n_heads, W * H, B);

                K = ggml_view_3d(ctx0, cur, n_embd, W * H, B, cur->nb[2], cur->nb[3], 1 * cur->nb[1]);
                K = ggml_reshape_4d(ctx0, ggml_cont(ctx0, K), d_heads, n_heads, W * H, B);

                V = ggml_view_3d(ctx0, cur, n_embd, W * H, B, cur->nb[2], cur->nb[3], 2 * cur->nb[1]);
                V = ggml_reshape_4d(ctx0, ggml_cont(ctx0, V), d_heads, n_heads, W * H, B);

                ggml_tensor * mask;
                ggml_tensor * rw;
                ggml_tensor * rh;
                ggml_tensor * qr;

                rw = get_rel_pos(ctx0, layer.rel_pos_w, indices, W, W);  // [W, W, C]
                rh = get_rel_pos(ctx0, layer.rel_pos_h, indices, H, H);  // [H, H, C]
                qr = ggml_permute(ctx0, Q, 0, 2, 1, 3);
                qr = ggml_reshape_4d(ctx0, ggml_cont(ctx0, qr), d_heads, W, H, B * n_heads);

                rw   = ggml_mul_mat(ctx0, rw,
                                    ggml_cont(ctx0, ggml_permute(ctx0, qr, 0, 2, 1, 3)));  // [B*n_heads, W, H, W]
                rw   = ggml_cont(ctx0, ggml_permute(ctx0, rw, 0, 2, 1, 3));                // [B*n_heads, H, W, W]
                rw   = ggml_reshape_4d(ctx0, rw, W, 1, W * H, n_heads * B);
                rw   = ggml_repeat_4d(ctx0, rw, W, H, W * H, n_heads * B);
                rh   = ggml_mul_mat(ctx0, rh, qr);  // [B*n_heads, H, W, H]
                rh   = ggml_reshape_4d(ctx0, rh, 1, H, W * H, n_heads * B);
                mask = ggml_add(ctx0, rw, rh);      // [B*n_heads, H*W, H, W]
                mask = ggml_reshape_4d(ctx0, mask, W * H, W * H, n_heads, B);
                mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);

                const float scale = 1.0f / sqrtf(static_cast<float>(d_heads));

                cur = build_attn(layer.o_w, layer.o_b, Q, K, V, mask, scale,
                                 il);  // [B, H*W, n_embd]
                cur = ggml_reshape_4d(ctx0, ggml_cont(ctx0, cur), n_embd, W, H, B);
            }

            if (hparams.is_global_attn(il) == false) {
                // local attention layer - reverse window partition
                cur = window_unpartition(ctx0, cur, w0, h0, window);
            }

            // re-add the layer input, e.g., residual
            cur = ggml_add(ctx0, cur, shortcut);

            ggml_tensor * inpFF = cur;

            // layernorm2
            cur = build_norm(inpFF, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);

            // ffn
            cur = build_ffn(cur, layer.ff_up_w, layer.ff_up_b, nullptr, nullptr, layer.ff_down_w, layer.ff_down_b,
                            hparams.ffn_op, il);

            // residual 2
            cur = ggml_add(ctx0, cur, inpFF);
            cb(cur, "sam_layer_out", il);
        }

        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));

        cur = ggml_conv_2d(ctx0, model.neck_0_w, cur, 1, 1, 0, 0, 1, 1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
        cur = build_norm(cur, model.neck_1_w, model.neck_1_b, NORM_TYPE_NORMAL, hparams.eps, -1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));

        cur = ggml_conv_2d(ctx0, model.neck_2_w, cur, 1, 1, 1, 1, 1, 1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 2, 0, 3));
        cur = build_norm(cur, model.neck_3_w, model.neck_3_b, NORM_TYPE_NORMAL, hparams.eps, -1);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));

        cur = ggml_conv_2d(ctx0, model.net_2, cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_conv_2d(ctx0, model.net_3, cur, 2, 2, 1, 1, 1, 1);
        cb(cur, "sam_output", -1);

        ggml_build_forward_expand(gf, cur);
        sam_out = cur;
    }

    ggml_tensor * clip_out;
    // Building DS-OCR CLIP
    {
        ggml_tensor * inp;

        inp = ggml_cpy(ctx0, sam_out, ggml_dup_tensor(ctx0, sam_out));
        inp = ggml_reshape_2d(ctx0, inp, inp->ne[0] * inp->ne[1], inp->ne[2]);
        inp = ggml_cont(ctx0, ggml_permute(ctx0, inp, 1, 0, 2, 3));

        ggml_tensor * new_pos_embd =
            ggml_cpy(ctx0, model.position_embeddings, ggml_dup_tensor(ctx0, model.position_embeddings));

        int        n_pos    = new_pos_embd->ne[1];  // +1 for [CLS]
        const auto tgt_size = static_cast<int>(std::sqrt(inp->ne[1]));
        const auto src_size = static_cast<int>(std::sqrt(n_pos - 1));

        if (tgt_size != src_size) {
            ggml_tensor * old_pos_embd;
            ggml_tensor * cls_tok;

            old_pos_embd = ggml_view_2d(ctx0, new_pos_embd, new_pos_embd->ne[0], src_size * src_size,
                                        ggml_row_size(new_pos_embd->type, new_pos_embd->ne[0]), 0);
            cls_tok      = ggml_view_2d(ctx0, new_pos_embd, new_pos_embd->ne[0], 1,
                                        ggml_row_size(new_pos_embd->type, new_pos_embd->ne[0]), src_size * src_size);
            new_pos_embd = ggml_interpolate(ctx0, old_pos_embd, tgt_size, tgt_size, new_pos_embd->ne[0], 1,
                                            GGML_SCALE_MODE_BICUBIC);
            new_pos_embd = ggml_reshape_3d(ctx0, new_pos_embd, n_embd, tgt_size * tgt_size, 1);
            new_pos_embd = ggml_concat(ctx0, new_pos_embd, cls_tok, 1);
            n_pos        = tgt_size * tgt_size + 1;
        }

        // add CLS token
        inp = ggml_concat(ctx0, model.class_embedding, inp, 1);

        // for selecting learned pos embd, used by ViT
        ggml_tensor * positions        = ggml_cast(ctx0, ggml_arange(ctx0, 0, n_pos, 1), GGML_TYPE_I32);
        ggml_tensor * learned_pos_embd = ggml_get_rows(ctx0, new_pos_embd, positions);

        ggml_tensor * cur = build_vit(inp, n_pos, NORM_TYPE_NORMAL, FFN_GELU_QUICK, learned_pos_embd, nullptr);

        ggml_build_forward_expand(gf, cur);
        clip_out = cur;
    }

    const int clip_n_patches = sam_out->ne[0] * sam_out->ne[1];

    sam_out  = ggml_cont(ctx0, ggml_permute(ctx0, sam_out, 1, 2, 0, 3));
    sam_out  = ggml_reshape_2d(ctx0, sam_out, sam_out->ne[0], clip_n_patches);
    clip_out = ggml_view_2d(ctx0, clip_out, n_embd, clip_n_patches, clip_out->nb[1], clip_out->nb[1]);

    ggml_tensor * cur;
    cur = ggml_concat(ctx0, clip_out, sam_out, 0);
    cur = ggml_reshape_2d(ctx0, cur, 2 * n_embd, clip_n_patches);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model.mm_fc_w, cur);
    cur = ggml_add(ctx0, cur, model.mm_fc_b);

    const auto h     = static_cast<int>(std::sqrt(static_cast<float>(cur->ne[1])));
    const auto w     = h;
    const auto n_dim = cur->ne[0];

    ggml_tensor * imgnl;
    ggml_tensor * vs;

    imgnl = ggml_repeat_4d(ctx0, model.image_newline, n_dim, 1, h, 1);
    vs    = ggml_reshape_2d(ctx0, model.view_seperator, n_dim, 1);  // (n_dim, 1)
    cur   = ggml_reshape_3d(ctx0, cur, n_dim, w, h);
    cur   = ggml_reshape_2d(ctx0, ggml_concat(ctx0, cur, imgnl, 1), n_dim, (w + 1) * h);
    cur   = ggml_concat(ctx0, cur, vs, 1);  // (n_dim, h*(w+1) + 1)

    cb(cur, "dsocr_output", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}

// this graph is used by llava, granite and glm
// due to having embedding_stack (used by granite), we cannot reuse build_vit
ggml_cgraph * clip_graph_llava::build() {
    const int batch_size = 1;
    const int n_pos = n_patches + (model.class_embedding ? 1 : 0);

    GGML_ASSERT(n_patches_x == n_patches_y && "only square images supported");

    // Calculate the deepest feature layer based on hparams and projector type
    int max_feature_layer = n_layer;
    {
        // Get the index of the second to last layer; this is the default for models that have a llava projector
        int il_last = hparams.n_layer - 1;
        int deepest_feature_layer = -1;

        if (proj_type == PROJECTOR_TYPE_MINICPMV || proj_type == PROJECTOR_TYPE_GLM_EDGE) {
            il_last += 1;
        }

        // If we set explicit vision feature layers, only go up to the deepest one
        // NOTE: only used by granite-vision models for now
        for (const auto & feature_layer : hparams.vision_feature_layer) {
            if (feature_layer > deepest_feature_layer) {
                deepest_feature_layer = feature_layer;
            }
        }
        max_feature_layer = deepest_feature_layer < 0 ? il_last : deepest_feature_layer;
    }

    ggml_tensor * inp = build_inp();

    // concat class_embeddings and patch_embeddings
    if (model.class_embedding) {
        inp = ggml_concat(ctx0, inp, model.class_embedding, 1);
    }

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    inp = ggml_add(ctx0, inp, ggml_get_rows(ctx0, model.position_embeddings, positions));

    ggml_tensor * inpL = inp;

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    std::vector<ggml_tensor *> embedding_stack;
    const auto & vision_feature_layer = hparams.vision_feature_layer;

    // loop over layers
    for (int il = 0; il < max_feature_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // If this is an embedding feature layer, save the output.
        // NOTE: 0 index here refers to the input to the encoder.
        if (vision_feature_layer.find(il) != vision_feature_layer.end()) {
            embedding_stack.push_back(cur);
        }

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "layer_inp_normed", il);

        // self-attention
        {
            ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            if (layer.q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.q_b);
            }

            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            if (layer.k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.k_b);
            }

            ggml_tensor * Vcur = build_mm(layer.v_w, cur);
            if (layer.v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.v_b);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
    }

    ggml_tensor * embeddings = inpL;

    // process vision feature layers (used by granite)
    {
        // final layer is a vision feature layer
        if (vision_feature_layer.find(max_feature_layer) != vision_feature_layer.end()) {
            embedding_stack.push_back(inpL);
        }

        // If feature layers are explicitly set, stack them (if we have multiple)
        if (!embedding_stack.empty()) {
            embeddings = embedding_stack[0];
            for (size_t i = 1; i < embedding_stack.size(); i++) {
                embeddings = ggml_concat(ctx0, embeddings, embedding_stack[i], 0);
            }
        }
    }

    // llava projector (also used by granite)
    if (hparams.has_llava_projector) {
        embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

        ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
        ggml_set_name(patches, "patches");
        ggml_set_input(patches);

        // shape [1, 576, 1024]
        // ne is whcn, ne = [1024, 576, 1, 1]
        embeddings = ggml_get_rows(ctx0, embeddings, patches);

        // print_tensor_info(embeddings, "embeddings");

        // llava projector
        if (proj_type == PROJECTOR_TYPE_MLP) {
            embeddings = build_mm(model.mm_0_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

            embeddings = ggml_gelu(ctx0, embeddings);
            if (model.mm_2_w) {
                embeddings = build_mm(model.mm_2_w, embeddings);
                embeddings = ggml_add(ctx0, embeddings, model.mm_2_b);
            }
        }
        else if (proj_type == PROJECTOR_TYPE_MLP_NORM) {
            embeddings = build_mm(model.mm_0_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);
            // ggml_tensor_printf(embeddings, "mm_0_w",0,true,false);
            // First LayerNorm
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_1_w),
                                model.mm_1_b);

            // GELU activation
            embeddings = ggml_gelu(ctx0, embeddings);

            // Second linear layer
            embeddings = build_mm(model.mm_3_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_3_b);

            // Second LayerNorm
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_4_w),
                                model.mm_4_b);
        }
        else if (proj_type == PROJECTOR_TYPE_LDP) {
            // MobileVLM projector
            int n_patch = 24;
            ggml_tensor * mlp_1 = build_mm(model.mm_model_mlp_1_w, embeddings);
            mlp_1 = ggml_add(ctx0, mlp_1, model.mm_model_mlp_1_b);
            mlp_1 = ggml_gelu(ctx0, mlp_1);
            ggml_tensor * mlp_3 = build_mm(model.mm_model_mlp_3_w, mlp_1);
            mlp_3 = ggml_add(ctx0, mlp_3, model.mm_model_mlp_3_b);
            // mlp_3 shape = [1, 576, 2048], ne = [2048, 576, 1, 1]

            // block 1
            ggml_tensor * block_1 = nullptr;
            {
                // transpose from [1, 576, 2048] --> [1, 2048, 576] --> [1, 2048, 24, 24]
                mlp_3 = ggml_permute(ctx0, mlp_3, 1, 0, 2, 3);
                mlp_3 = ggml_cont_4d(ctx0, mlp_3, n_patch, n_patch, mlp_3->ne[1], mlp_3->ne[2]);
                // stride = 1, padding = 1, bias is nullptr
                block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_1_block_0_0_w, mlp_3, 1, 1, 1, 1, 1, 1);

                // layer norm
                // // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_0_1_w), model.mm_model_block_1_block_0_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));

                // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                // hardswish
                ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                // pointwise conv
                block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                block_1 = build_mm(model.mm_model_block_1_block_1_fc1_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc1_b);
                block_1 = ggml_relu(ctx0, block_1);
                block_1 = build_mm(model.mm_model_block_1_block_1_fc2_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc2_b);
                block_1 = ggml_hardsigmoid(ctx0, block_1);
                // block_1_hw shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1], block_1 shape = [1, 2048], ne = [2048, 1, 1, 1]
                block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                int w = block_1->ne[0], h = block_1->ne[1];
                block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));

                // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                block_1 = build_mm(model.mm_model_block_1_block_2_0_w, block_1);
                block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);

                // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_2_1_w), model.mm_model_block_1_block_2_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // block1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                // residual
                block_1 = ggml_add(ctx0, mlp_3, block_1);
            }

            // block_2
            {
                // stride = 2
                block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_2_block_0_0_w, block_1, 2, 2, 1, 1, 1, 1);

                // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                // layer norm
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_0_1_w), model.mm_model_block_2_block_0_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                // hardswish
                ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                // not sure the parameters is right for globalAvgPooling
                block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                // pointwise conv
                block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                block_1 = build_mm(model.mm_model_block_2_block_1_fc1_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc1_b);
                block_1 = ggml_relu(ctx0, block_1);
                block_1 = build_mm(model.mm_model_block_2_block_1_fc2_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc2_b);
                block_1 = ggml_hardsigmoid(ctx0, block_1);

                // block_1_hw shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1], block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                int w = block_1->ne[0], h = block_1->ne[1];
                block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));
                // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                block_1 = build_mm(model.mm_model_block_2_block_2_0_w, block_1);
                block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);


                // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_2_1_w), model.mm_model_block_2_block_2_1_b);
                block_1 = ggml_reshape_3d(ctx0, block_1, block_1->ne[0], block_1->ne[1] * block_1->ne[2], block_1->ne[3]);
                // block_1 shape = [1, 144, 2048], ne = [2048, 144, 1]
            }
            embeddings = block_1;
        }
        else if (proj_type == PROJECTOR_TYPE_LDPV2)
        {
            int n_patch = 24;
            ggml_tensor * mlp_0 = build_mm(model.mm_model_mlp_0_w, embeddings);
            mlp_0 = ggml_add(ctx0, mlp_0, model.mm_model_mlp_0_b);
            mlp_0 = ggml_gelu(ctx0, mlp_0);
            ggml_tensor * mlp_2 = build_mm(model.mm_model_mlp_2_w, mlp_0);
            mlp_2 = ggml_add(ctx0, mlp_2, model.mm_model_mlp_2_b);
            // mlp_2 ne = [2048, 576, 1, 1]
            // // AVG Pool Layer 2*2, strides = 2
            mlp_2 = ggml_permute(ctx0, mlp_2, 1, 0, 2, 3);
            // mlp_2 ne = [576, 2048, 1, 1]
            mlp_2 = ggml_cont_4d(ctx0, mlp_2, n_patch, n_patch, mlp_2->ne[1], mlp_2->ne[2]);
            // mlp_2 ne [24, 24, 2048, 1]
            mlp_2 = ggml_pool_2d(ctx0, mlp_2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
            // weight ne = [3, 3, 2048, 1]
            ggml_tensor * peg_0 = ggml_conv_2d_dw(ctx0, model.mm_model_peg_0_w, mlp_2, 1, 1, 1, 1, 1, 1);
            peg_0 = ggml_cont(ctx0, ggml_permute(ctx0, peg_0, 1, 2, 0, 3));
            peg_0 = ggml_add(ctx0, peg_0, model.mm_model_peg_0_b);
            mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 2, 0, 3));
            peg_0 = ggml_add(ctx0, peg_0, mlp_2);
            peg_0 = ggml_reshape_3d(ctx0, peg_0, peg_0->ne[0], peg_0->ne[1] * peg_0->ne[2], peg_0->ne[3]);
            embeddings = peg_0;
        }
        else {
            GGML_ABORT("fatal error");
        }
    }

    // glm projector
    else if (proj_type == PROJECTOR_TYPE_GLM_EDGE) {
        size_t gridsz = (size_t)sqrt(embeddings->ne[1]);
        embeddings = ggml_permute(ctx0,embeddings,1,0,2,3);
        embeddings = ggml_cont_3d(ctx0, embeddings, gridsz, gridsz, embeddings->ne[1]);
        embeddings = ggml_conv_2d(ctx0, model.mm_model_adapter_conv_w, embeddings, 2, 2, 0, 0, 1, 1);
        embeddings = ggml_reshape_3d(ctx0, embeddings,embeddings->ne[0]*embeddings->ne[1] , embeddings->ne[2], batch_size);
        embeddings = ggml_cont(ctx0, ggml_permute(ctx0,embeddings, 1, 0, 2, 3));
        embeddings = ggml_add(ctx0, embeddings, model.mm_model_adapter_conv_b);
        // GLU
        {
            embeddings = build_mm(model.mm_model_mlp_0_w, embeddings);
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_model_ln_q_w), model.mm_model_ln_q_b);
            embeddings = ggml_gelu_inplace(ctx0, embeddings);
            ggml_tensor * x = embeddings;
            embeddings = build_mm(model.mm_model_mlp_2_w, embeddings);
            x = build_mm(model.mm_model_mlp_1_w,x);
            embeddings = ggml_swiglu_split(ctx0, embeddings, x);
            embeddings = build_mm(model.mm_model_mlp_3_w, embeddings);
        }
        // arrangement of BOI/EOI token embeddings
        // note: these embeddings are not present in text model, hence we cannot process them as text tokens
        // see: https://huggingface.co/THUDM/glm-edge-v-2b/blob/main/siglip.py#L53
        {
            embeddings = ggml_concat(ctx0, model.mm_boi, embeddings, 1); // BOI
            embeddings = ggml_concat(ctx0, embeddings, model.mm_eoi, 1); // EOI
        }
    }

    else {
        GGML_ABORT("llava: unknown projector type");
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}

ggml_tensor * clip_graph_mobilenetv5::rms_norm_2d(ggml_tensor * inp, ggml_tensor * weight, float eps) {
    // inp: [W, H, C, B]

    ggml_tensor * cur = ggml_permute(ctx0, inp, 2, 1, 0, 3);
    cur = ggml_cont(ctx0, cur);
    cur = ggml_rms_norm(ctx0, cur, eps);

    if (weight) {
        cur = ggml_mul(ctx0, cur, weight);
    }

    cur = ggml_permute(ctx0, cur, 2, 1, 0, 3);
    cur = ggml_cont(ctx0, cur);

    return cur;
}

ggml_tensor* clip_graph_mobilenetv5::pad_same_2d(ggml_tensor* inp, int kernel_h, int kernel_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
    const int64_t ih = inp->ne[1];  // height
    const int64_t iw = inp->ne[0];  // width

    // Calculate output size (ceil division)
    const int64_t oh = (ih + stride_h - 1) / stride_h;
    const int64_t ow = (iw + stride_w - 1) / stride_w;

    // Calculate padding needed
    const int64_t pad_h = std::max((int64_t)0, (oh - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - ih);
    const int64_t pad_w = std::max((int64_t)0, (ow - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - iw);

    // Split padding asymmetrically
    const int pad_h_top = pad_h / 2;
    const int pad_h_bottom = pad_h - pad_h_top;
    const int pad_w_left = pad_w / 2;
    const int pad_w_right = pad_w - pad_w_left;

    // Apply padding if needed
    // ggml_pad_ext: (ctx, tensor, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
    // For [W, H, C, B]: p0=width, p1=height, p2=channels, p3=batch
    if (pad_h > 0 || pad_w > 0) {
        inp = ggml_pad_ext(ctx0, inp,
            pad_w_left, pad_w_right,     // width padding (dim 0)
            pad_h_top, pad_h_bottom,      // height padding (dim 1)
            0, 0,                         // no channel padding (dim 2)
            0, 0);                        // no batch padding (dim 3)
    }

    return inp;
}

ggml_tensor * clip_graph_mobilenetv5::build_edge_residual(ggml_tensor * inp, const mobilenetv5_block & block, int stride) {
    ggml_tensor * cur = inp;

    // 1. Expansion Conv (3x3)
    if (stride == 2) {
        // Case: Downsampling (Block 0)
        // Replicates Conv2dSame(kernel=3, stride=2)
        cur = pad_same_2d(cur, 3, 3, stride, stride);
        cur = ggml_conv_2d_direct(ctx0, block.s0_conv_exp_w, cur, stride, stride, 0, 0, 1, 1);
    } else {
        // Case: Normal 3x3 Block (Block 1, 2)
        // Replicates Conv2d(kernel=3, stride=1, padding=1)
        cur = ggml_conv_2d_direct(ctx0, block.s0_conv_exp_w, cur, stride, stride, 1, 1, 1, 1);
    }

    // BN + Activation
    if (block.s0_bn1_w) cur = rms_norm_2d(cur, block.s0_bn1_w);
    cur = ggml_gelu(ctx0, cur);

    // 2. Pointwise Linear Conv (1x1)
    // 1x1 Convs usually have padding=0 and stride=1
    cur = ggml_conv_2d_direct(ctx0, block.s0_conv_pwl_w, cur, 1, 1, 0, 0, 1, 1);
    if (block.s0_bn2_w) cur = rms_norm_2d(cur, block.s0_bn2_w);

    // 3. Residual Connection
    // Only apply residual if spatial dimensions and channels match (stride 1)
    if (stride == 1 && inp->ne[2] == cur->ne[2] && inp->ne[0] == cur->ne[0]) {
        cur = ggml_add(ctx0, cur, inp);
    }

    return cur;
}

ggml_tensor * clip_graph_mobilenetv5::build_inverted_residual(ggml_tensor * inp, const mobilenetv5_block & block, int stride) {
    ggml_tensor * cur = inp;

    // 1. Depthwise Start (Optional)
    // NOTE: dw_start always has stride=1 (no downsampling here)
    if (block.dw_start_w) {
        int k = block.dw_start_w->ne[0]; // 3 or 5
        int p = k / 2;
        cur = ggml_conv_2d_dw(ctx0, block.dw_start_w, cur, 1, 1, p, p, 1, 1);
        if (block.dw_start_bn_w) cur = rms_norm_2d(cur, block.dw_start_bn_w);
    }

    // 2. Pointwise Expansion (1x1)
    if (block.pw_exp_w) {
        // Standard 1x1 conv, pad=0, stride=1
        cur = ggml_conv_2d_direct(ctx0, block.pw_exp_w, cur, 1, 1, 0, 0, 1, 1);
        if (block.pw_exp_bn_w) cur = rms_norm_2d(cur, block.pw_exp_bn_w);
        cur = ggml_gelu(ctx0, cur);
    }

    // 3. Depthwise Mid (Optional)
    // NOTE: dw_mid is where downsampling happens (stride=2 for first block of stage)
    if (block.dw_mid_w) {
        int k = block.dw_mid_w->ne[0]; // 3 or 5

        if (stride > 1) {
            // Case: Stride 2 (Downsample) -> Use Asymmetric "Same" Padding
            cur = pad_same_2d(cur, k, k, stride, stride);
            cur = ggml_conv_2d_dw(ctx0, block.dw_mid_w, cur, stride, stride, 0, 0, 1, 1); // pad=0
        } else {
            // Case: Stride 1 -> Use Standard Symmetric Padding
            int p = k / 2;
            cur = ggml_conv_2d_dw(ctx0, block.dw_mid_w, cur, stride, stride, p, p, 1, 1);
        }

        if (block.dw_mid_bn_w) cur = rms_norm_2d(cur, block.dw_mid_bn_w);
        cur = ggml_gelu(ctx0, cur);
    }

    // 4. Pointwise Projection (1x1)
    if (block.pw_proj_w) {
        cur = ggml_conv_2d_direct(ctx0, block.pw_proj_w, cur, 1, 1, 0, 0, 1, 1);
        if (block.pw_proj_bn_w) cur = rms_norm_2d(cur, block.pw_proj_bn_w);
    }

    // Apply Layer Scaling if present
    if (block.layer_scale_w) {
        cur = ggml_mul(ctx0, cur, block.layer_scale_w);
    }

    // 5. Residual Connection
    bool same_spatial = (inp->ne[0] == cur->ne[0]) && (inp->ne[1] == cur->ne[1]);
    bool same_channel = (inp->ne[2] == cur->ne[2]);
    if (same_spatial && same_channel) {
        cur = ggml_add(ctx0, cur, inp);
    }

    return cur;
}

ggml_tensor * clip_graph_mobilenetv5::build_mobilenet_attn(ggml_tensor * inp, const mobilenetv5_block & block) {
    ggml_tensor * cur = inp;

    // Norm
    if (block.attn_norm_w) {
        cur = rms_norm_2d(cur, block.attn_norm_w, 1e-6f);
    }

    // 1. Q Calculation
    ggml_tensor * q = ggml_conv_2d_direct(ctx0, block.attn_q_w, cur, 1, 1, 0, 0, 1, 1);

    // 2. K Calculation (Downsampled)
    // Uses Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640)
    ggml_tensor * k_inp = cur;
    if (block.attn_k_dw_w) {
        int k_size = block.attn_k_dw_w->ne[0];  // Usually 3
        k_inp = pad_same_2d(cur, k_size, k_size, 2, 2);  // Apply SAME padding
        k_inp = ggml_conv_2d_dw(ctx0, block.attn_k_dw_w, k_inp, 2, 2, 0, 0, 1, 1);  // padding=0
        if (block.attn_k_norm_w) {
            k_inp = rms_norm_2d(k_inp, block.attn_k_norm_w, 1e-6f);
        }
    }
    ggml_tensor * k = ggml_conv_2d_direct(ctx0, block.attn_k_w, k_inp, 1, 1, 0, 0, 1, 1);

    // 3. V Calculation (Downsampled)
    // Uses Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640)
    ggml_tensor * v_inp = cur;
    if (block.attn_v_dw_w) {
        int v_size = block.attn_v_dw_w->ne[0];  // Usually 3
        v_inp = pad_same_2d(cur, v_size, v_size, 2, 2);  // Apply SAME padding
        v_inp = ggml_conv_2d_dw(ctx0, block.attn_v_dw_w, v_inp, 2, 2, 0, 0, 1, 1);  // padding=0
        if (block.attn_v_norm_w) {
            v_inp = rms_norm_2d(v_inp, block.attn_v_norm_w, 1e-6f);
        }
    }
    ggml_tensor * v = ggml_conv_2d_direct(ctx0, block.attn_v_w, v_inp, 1, 1, 0, 0, 1, 1);

    const int W = cur->ne[0]; const int H = cur->ne[1]; const int B = cur->ne[3];
    const int D = k->ne[2]; // Head dimension
    const int n_head = q->ne[2] / D;
    const int N = W * H;

    // Process Q: [W, H, D*n_head, B] -> [D, N, n_head, B]
    q = ggml_reshape_3d(ctx0, q, N, D*n_head, B);
    q = ggml_reshape_4d(ctx0, q, N, D, n_head, B);
    q = ggml_permute(ctx0, q, 1, 0, 2, 3); // [D, N, n_head, B]
    q = ggml_cont(ctx0, q);

    const int Wk = k->ne[0]; const int Hk = k->ne[1];
    const int M = Wk * Hk;

    // Process K: [Wk, Hk, D, B] -> [D, M, 1, B]
    k = ggml_reshape_3d(ctx0, k, M, D, B);
    k = ggml_reshape_4d(ctx0, k, M, D, 1, B);
    k = ggml_permute(ctx0, k, 1, 0, 2, 3); // [D, M, 1, B]
    k = ggml_cont(ctx0, k);

    // Process V: [Wk, Hk, D, B] -> [M, D, 1, B]
    v = ggml_reshape_3d(ctx0, v, M, D, B);
    v = ggml_reshape_4d(ctx0, v, M, D, 1, B);
    v = ggml_cont(ctx0, v); // [M, D, 1, B]

    // Multi-Query Attention
    float scale = 1.0f / sqrtf((float)D);

    // Step 1: Compute Q @ K.T
    ggml_tensor * scores = ggml_mul_mat(ctx0, k, q);

    scores = ggml_scale(ctx0, scores, scale);

    scores = ggml_soft_max(ctx0, scores);

    ggml_tensor * kqv = ggml_mul_mat(ctx0, v, scores);

    kqv = ggml_permute(ctx0, kqv, 1, 0, 2, 3);
    kqv = ggml_cont(ctx0, kqv);

    kqv = ggml_reshape_3d(ctx0, kqv, N, D * n_head, B);
    kqv = ggml_reshape_4d(ctx0, kqv, W, H, D * n_head, B);
    kqv = ggml_cont(ctx0, kqv);

    // Output projection
    cur = ggml_conv_2d_direct(ctx0, block.attn_o_w, kqv, 1, 1, 0, 0, 1, 1);

    // Residual & Layer Scale
    if (inp->ne[0] == cur->ne[0] && inp->ne[2] == cur->ne[2]) {
        if (block.layer_scale_w) {
            cur = ggml_mul(ctx0, cur, block.layer_scale_w);
        }
        cur = ggml_add(ctx0, cur, inp);
    }

    return cur;
}

ggml_cgraph * clip_graph_mobilenetv5::build() {
    ggml_tensor * inp = build_inp_raw();

    // 1. Stem - Conv2dSame(3, 64, kernel_size=(3, 3), stride=(2, 2))
    ggml_tensor * cur = pad_same_2d(inp, 3, 3, 2, 2);  // Apply SAME padding

    cur = ggml_conv_2d_direct(ctx0, model.mobilenet_stem_conv_w, cur, 2, 2, 0, 0, 1, 1);  // padding=0
    if (model.mobilenet_stem_conv_b) {
        cur = ggml_add(ctx0, cur, model.mobilenet_stem_conv_b);
    }
    if (model.mobilenet_stem_norm_w) cur = rms_norm_2d(cur, model.mobilenet_stem_norm_w);
    cur = ggml_gelu(ctx0, cur);

    // 2. Blocks
    std::vector<ggml_tensor*> intermediate_features;
    const int total_blocks = model.mobilenet_blocks.size();

    auto is_stage_start = [&](int i) {
        if (i == 0) return true;
        for (int end_idx : model.mobilenet_stage_ends) {
            if (i == end_idx + 1) return true;
        }
        return false;
    };

    auto is_fusion_point = [&](int i) {
        if (model.mobilenet_stage_ends.size() >= 4) {
                if (i == model.mobilenet_stage_ends[2]) return true; // End of Stage 2
                if (i == model.mobilenet_stage_ends[3]) return true; // End of Stage 3
        } else {
            if (i == total_blocks - 1) return true;
        }
        return false;
    };

    for (int i = 0; i < total_blocks; i++) {
        const auto & block = model.mobilenet_blocks[i];
        int stride = is_stage_start(i) ? 2 : 1;

        if (block.s0_conv_exp_w)      cur = build_edge_residual(cur, block, stride);
        else if (block.attn_q_w)      cur = build_mobilenet_attn(cur, block);
        else                          cur = build_inverted_residual(cur, block, stride);

        if (is_fusion_point(i)) {

            intermediate_features.push_back(cur);
        }
    }

    // 3. Multi-Scale Fusion Adapter (MSFA)
    if (!intermediate_features.empty()) {

        // A. Reference Resolution: PyTorch implementation uses inputs[0]
        // We assume intermediate_features[0] is the "High Resolution" target.
        // In MobileNet designs, this is typically the feature map with the smallest stride (e.g. 32x32).
        ggml_tensor* target_feat = intermediate_features[0];
        int high_res_w = target_feat->ne[0];
        int high_res_h = target_feat->ne[1];

        std::vector<ggml_tensor*> resized_feats;

        // B. Resize inputs to match inputs[0] (High Resolution)
        for (auto feat : intermediate_features) {
            int feat_w = feat->ne[0];
            int feat_h = feat->ne[1];

            // PyTorch: if feat_size < high_resolution: interpolate
            if (feat_w < high_res_w || feat_h < high_res_h) {
                // Calculate scale factor.
                // Note: PyTorch 'nearest' works on arbitrary float scales.
                // ggml_upscale generally takes integer factors or target sizes depending on helper.
                // Assuming standard power-of-2 scaling (e.g. 16 -> 32 means scale=2).
                int scale_w = high_res_w / feat_w;
                // int scale_h = high_res_h / feat_h;

                // Safety check for non-integer scaling if strictly replicating
                GGML_ASSERT(high_res_w % feat_w == 0);

                // Upsample (Nearest Neighbor)
                // 2 is the scale factor
                feat = ggml_upscale(ctx0, feat, scale_w, ggml_scale_mode::GGML_SCALE_MODE_NEAREST);
            }
            resized_feats.push_back(feat);
        }

        // C. Concatenate at High Resolution (Channel Dim = 2 in ggml)
        cur = resized_feats[0];
        for (size_t k = 1; k < resized_feats.size(); ++k) {
            cur = ggml_concat(ctx0, cur, resized_feats[k], 2);
        }

        // D. FFN (UniversalInvertedResidual)
        // Structure: Expand Conv -> Norm -> GELU -> Project Conv -> Norm

        // 1. Expansion
        if (model.msfa_ffn_expand_w) {
            // 1x1 Conv
            cur = ggml_conv_2d_direct(ctx0, model.msfa_ffn_expand_w, cur, 1, 1, 0, 0, 1, 1);

            if (model.msfa_ffn_expand_bn) {
                cur = rms_norm_2d(cur, model.msfa_ffn_expand_bn);
            }

            cur = ggml_gelu(ctx0, cur);

        }

        // 2. Projection (No DW because kernel_size=0)
        if (model.msfa_ffn_project_w) {
            // 1x1 Conv
            cur = ggml_conv_2d_direct(ctx0, model.msfa_ffn_project_w, cur, 1, 1, 0, 0, 1, 1);

            // UniversalInvertedResidual typically has a norm after projection
            if (model.msfa_ffn_project_bn) {
                cur = rms_norm_2d(cur, model.msfa_ffn_project_bn);
            }

        }

        // E. Final Downsample to Target Resolution (Output Resolution)
        // PyTorch: matches self.output_resolution (e.g. 16x16)
        const int target_out_res = 16;
        int current_w = cur->ne[0];

        if (current_w > target_out_res) {
            int s = current_w / target_out_res;

            GGML_ASSERT(current_w % target_out_res == 0);

            // Avg Pool: Kernel=s, Stride=s
            cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, s, s, s, s, 0, 0);

        }

        // F. Final Norm
        if (model.msfa_concat_norm_w) {
            cur = rms_norm_2d(cur, model.msfa_concat_norm_w);

        }
    }

    // 4. Gemma 3n Multimodal Projection (Embedder)
    // Input: 'cur' is [Width, Height, Channels, Batch]
    int W = cur->ne[0];
    int H = cur->ne[1];
    int C = cur->ne[2];
    int B = cur->ne[3];

    GGML_ASSERT(C == hparams.n_embd);

    // 1. Permute and Flatten to [Channels, Tokens, Batch]
    // PyTorch expects (Batch, Seq, Hidden), GGML usually processes (Hidden, Seq, Batch)
    cur = ggml_permute(ctx0, cur, 2, 1, 0, 3); // -> [C, H, W, B]
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3); // -> [C, W, H, B]
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_3d(ctx0, cur, C, W*H, B);
    cur = ggml_cont(ctx0, cur);

    // 2. FEATURE SCALING
    // PyTorch: vision_outputs *= self.config.vision_config.hidden_size**0.5
    const float scale_factor = sqrtf((float)C);
    cur = ggml_scale(ctx0, cur, scale_factor);

    // 3. SOFT EMBEDDING NORM
    // PyTorch: self._norm(x) * self.weight
    // We must normalize regardless, then multiply if weight exists.
    {
        const float eps = 1e-6f; // Gemma3n uses 1e-6
        cur = ggml_rms_norm(ctx0, cur, eps);

        if (model.mm_soft_emb_norm_w) {
            // Weight shape is (2048,) -> Element-wise broadcast multiply
            cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
        }

    }

    // 4. PROJECTION
    // PyTorch: embedding_projection = nn.Linear(vision_hidden, text_hidden, bias=False)
    // Weight stored as [out_features, in_features] = [text_hidden_size, vision_hidden_size]
    if (model.mm_input_proj_w) {
        cur = build_mm(model.mm_input_proj_w, cur);
    }

    // 5. POST PROJECTION NORM
    // PyTorch: embedding_post_projection_norm = Gemma3nRMSNorm(..., with_scale=False)
    // with_scale=False means weight is registered as buffer with value 1.0
    // So output = rms_norm(x) * 1.0 = rms_norm(x), magnitude ~1
    {
        const float eps = 1e-6f;
        cur = ggml_rms_norm(ctx0, cur, eps);

        if (model.mm_post_proj_norm_w) {
            // If weight is loaded, multiply (should be ~1.0 anyway)
            cur = ggml_mul(ctx0, cur, model.mm_post_proj_norm_w);
        }
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}

#ifdef CLIP_DEBUG_FUNCTIONS
static void clip_image_write_image_to_ppm(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    // PPM header: P6 format, width, height, and max color value
    file << "P6\n" << img.nx << " " << img.ny << "\n255\n";

    // Write pixel data
    for (size_t i = 0; i < img.buf.size(); i += 3) {
        // PPM expects binary data in RGB format, which matches our image buffer
        file.write(reinterpret_cast<const char*>(&img.buf[i]), 3);
    }

    file.close();
}

static void clip_image_save_to_bmp(const clip_image_u8& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
        return;
    }

    int fileSize = 54 + 3 * img.nx * img.ny; // File header + info header + pixel data
    int bytesPerPixel = 3;
    int widthInBytes = img.nx * bytesPerPixel;
    int paddingAmount = (4 - (widthInBytes % 4)) % 4;
    int stride = widthInBytes + paddingAmount;

    // Bitmap file header
    unsigned char fileHeader[14] = {
        'B','M',     // Signature
        0,0,0,0,    // Image file size in bytes
        0,0,0,0,    // Reserved
        54,0,0,0    // Start of pixel array
    };

    // Total file size
    fileSize = 54 + (stride * img.ny);
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);

    // Bitmap information header (BITMAPINFOHEADER)
    unsigned char infoHeader[40] = {
        40,0,0,0,   // Size of this header (40 bytes)
        0,0,0,0,    // Image width
        0,0,0,0,    // Image height
        1,0,        // Number of color planes
        24,0,       // Bits per pixel
        0,0,0,0,    // No compression
        0,0,0,0,    // Image size (can be 0 for no compression)
        0,0,0,0,    // X pixels per meter (not specified)
        0,0,0,0,    // Y pixels per meter (not specified)
        0,0,0,0,    // Total colors (color table not used)
        0,0,0,0     // Important colors (all are important)
    };

    // Width and height in the information header
    infoHeader[4] = (unsigned char)(img.nx);
    infoHeader[5] = (unsigned char)(img.nx >> 8);
    infoHeader[6] = (unsigned char)(img.nx >> 16);
    infoHeader[7] = (unsigned char)(img.nx >> 24);
    infoHeader[8] = (unsigned char)(img.ny);
    infoHeader[9] = (unsigned char)(img.ny >> 8);
    infoHeader[10] = (unsigned char)(img.ny >> 16);
    infoHeader[11] = (unsigned char)(img.ny >> 24);

    // Write file headers
    file.write(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char*>(infoHeader), sizeof(infoHeader));

    // Pixel data
    std::vector<unsigned char> padding(3, 0); // Max padding size to be added to each row
    for (int y = img.ny - 1; y >= 0; --y) { // BMP files are stored bottom-to-top
        for (int x = 0; x < img.nx; ++x) {
            // Each pixel
            size_t pixelIndex = (y * img.nx + x) * 3;
            unsigned char pixel[3] = {
                img.buf[pixelIndex + 2], // BMP stores pixels in BGR format
                img.buf[pixelIndex + 1],
                img.buf[pixelIndex]
            };
            file.write(reinterpret_cast<char*>(pixel), 3);
        }
        // Write padding for the row
        file.write(reinterpret_cast<char*>(padding.data()), paddingAmount);
    }

    file.close();
}

// debug function to convert f32 to u8
static void clip_image_convert_f32_to_u8(const clip_image_f32& src, clip_image_u8& dst) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(3 * src.nx * src.ny);
    for (size_t i = 0; i < src.buf.size(); ++i) {
        dst.buf[i] = static_cast<uint8_t>(std::min(std::max(int(src.buf[i] * 255.0f), 0), 255));
    }
}
#endif


struct clip_ctx {
    clip_model model;

    gguf_context_ptr ctx_gguf;
    ggml_context_ptr ctx_data;

    std::vector<uint8_t> buf_compute_meta;

    std::vector<ggml_backend_t> backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_buffer_ptr buf;


    int max_nodes = 8192;
    ggml_backend_sched_ptr sched;
    clip_flash_attn_type flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    bool is_allocated = false;

    bool debug_output_embeddings = false;

    clip_ctx(clip_context_params & ctx_params) {
        flash_attn_type = ctx_params.flash_attn_type;
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        if (ctx_params.use_gpu) {
            auto backend_name = std::getenv("MTMD_BACKEND_DEVICE");
            if (backend_name != nullptr) {
                backend = ggml_backend_init_by_name(backend_name, nullptr);
                if (!backend) {
                    LOG_WRN("%s: Warning: Failed to initialize \"%s\" backend, falling back to default GPU backend\n", __func__, backend_name);
                }
            }
            if (!backend) {
                backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
                backend = backend ? backend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
            }
        }

        if (backend) {
            LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(backend));
            backend_ptrs.push_back(backend);
            backend_buft.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("%s: CLIP using CPU backend\n", __func__);
        }

        if (ctx_params.image_min_tokens > 0) {
            model.hparams.custom_image_min_tokens = ctx_params.image_min_tokens;
        }
        if (ctx_params.image_max_tokens > 0) {
            model.hparams.custom_image_max_tokens = ctx_params.image_max_tokens;
        }

        backend_ptrs.push_back(backend_cpu);
        backend_buft.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(
            ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), 8192, false, true)
        );

        if (ctx_params.cb_eval != nullptr) {
            ggml_backend_sched_set_eval_callback(sched.get(), ctx_params.cb_eval, ctx_params.cb_eval_user_data);
        }

        debug_output_embeddings = std::getenv("MTMD_DEBUG_EMBEDDINGS") != nullptr;
    }

    ~clip_ctx() {
        ggml_backend_free(backend);
        if (backend != backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }

    // this function is added so that we don't change too much of the existing code
    projector_type proj_type() const {
        return model.proj_type;
    }
};

//
// clip_graph
//

clip_graph::clip_graph(clip_ctx * ctx, const clip_image_f32 & img) :
        model(ctx->model),
        hparams(model.hparams),
        proj_type(ctx->proj_type()),
        img(img),
        patch_size(hparams.patch_size),
        n_patches_x(img.nx / patch_size),
        n_patches_y(img.ny / patch_size),
        n_patches(n_patches_x * n_patches_y),
        n_embd(hparams.n_embd),
        n_head(hparams.n_head),
        d_head(n_embd / n_head),
        n_layer(hparams.n_layer),
        n_mmproj_embd(clip_n_mmproj_embd(ctx)),
        eps(hparams.eps),
        kq_scale(1.0f / sqrtf((float)d_head)),
        flash_attn_type(ctx->flash_attn_type) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ctx0_ptr.reset(ggml_init(params));
    ctx0 = ctx0_ptr.get();
    gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
}

ggml_tensor * clip_graph::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    return ggml_mul_mat(ctx0, w, x);
}

void clip_graph::cb(ggml_tensor * cur, const char * name, int il) const {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }
}

// siglip2 naflex
ggml_tensor * clip_graph::resize_position_embeddings(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny / patch_size;
    const int width        = img.nx / patch_size;
    const uint32_t mode    = interpolation_mode;
    const int n_per_side   = (int)std::sqrt(pos_embd->ne[1]);

    GGML_ASSERT(pos_embd);

    if (height == n_per_side && width == n_per_side) {
        return pos_embd;
    }

    pos_embd = ggml_reshape_3d(ctx0, pos_embd, n_embd, n_per_side, n_per_side);  // -> (n_embd, n_per_side, n_per_side)
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 0, 1, 3);                         // -> (n_per_side, n_per_side, n_embd)
    pos_embd = ggml_interpolate(ctx0, pos_embd, width, height, n_embd, 1, mode); // -> (width, height, n_embd)
    pos_embd = ggml_permute(ctx0, pos_embd, 1, 2, 0, 3);                         // -> (n_embd, width, height)
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);             // -> (n_embd, width * height)

    return pos_embd;
}

// build vision transformer (ViT) cgraph
// this function should cover most of the models
// if your model has specific features, you should probably duplicate this function
ggml_tensor * clip_graph::build_vit(
            ggml_tensor * inp,
            int64_t n_pos,
            norm_type norm_t,
            ffn_op_type ffn_t,
            ggml_tensor * learned_pos_embd,
            std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> add_pos
        ) {
    if (learned_pos_embd) {
        inp = ggml_add(ctx0, inp, learned_pos_embd);
        cb(inp, "pos_embed", -1);
    }

    ggml_tensor * inpL = inp;

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(cur, "layer_inp_normed", il);

        // self-attention
        {
            ggml_tensor * Qcur = nullptr;
            ggml_tensor * Kcur = nullptr;
            ggml_tensor * Vcur = nullptr;
            if (layer.qkv_w != nullptr) {
                // fused qkv
                cur = build_mm(layer.qkv_w, cur);
                if (layer.qkv_b != nullptr) {
                    cur = ggml_add(ctx0, cur, layer.qkv_b);
                }

                Qcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ 0);

                Kcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, n_embd));

                Vcur = ggml_view_3d(ctx0, cur, d_head, n_head, n_pos,
                    /* nb1    */ ggml_row_size(cur->type, d_head),
                    /* nb2    */ cur->nb[1],
                    /* offset */ ggml_row_size(cur->type, 2 * n_embd));

                if (layer.q_norm) {
                    GGML_ASSERT(layer.q_norm->ne[0] == Qcur->ne[0]);
                    Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                    cb(Qcur, "Qcur_norm", il);
                }

                if (layer.k_norm) {
                    GGML_ASSERT(layer.k_norm->ne[0] == Kcur->ne[0]);
                    Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                    cb(Kcur, "Kcur_norm", il);
                }

            } else {
                // separate q, k, v
                Qcur = build_mm(layer.q_w, cur);
                if (layer.q_b) {
                    Qcur = ggml_add(ctx0, Qcur, layer.q_b);
                }

                Kcur = build_mm(layer.k_w, cur);
                if (layer.k_b) {
                    Kcur = ggml_add(ctx0, Kcur, layer.k_b);
                }

                Vcur = build_mm(layer.v_w, cur);
                if (layer.v_b) {
                    Vcur = ggml_add(ctx0, Vcur, layer.v_b);
                }

                // if true, norm must be applied after reshaping to (d_head, n_head, n_pos)
                bool norm_per_head = layer.q_norm && layer.q_norm->ne[0] == d_head;

                if (!norm_per_head) {
                    if (layer.q_norm) {
                        Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                        cb(Qcur, "Qcur_norm", il);
                    }
                    if (layer.k_norm) {
                        Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                        cb(Kcur, "Kcur_norm", il);
                    }
                }

                Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
                Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
                Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

                if (norm_per_head) {
                    if (layer.q_norm) {
                        Qcur = build_norm(Qcur, layer.q_norm, NULL, norm_t, eps, il);
                        cb(Qcur, "Qcur_norm_per_head", il);
                    }
                    if (layer.k_norm) {
                        Kcur = build_norm(Kcur, layer.k_norm, NULL, norm_t, eps, il);
                        cb(Kcur, "Kcur_norm_per_head", il);
                    }
                }
            }

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (add_pos) {
                Qcur = add_pos(Qcur, layer);
                Kcur = add_pos(Kcur, layer);
                cb(Qcur, "Qcur_pos", il);
                cb(Kcur, "Kcur_pos", il);
            }

            if (proj_type == PROJECTOR_TYPE_GEMMA4V) {
                Vcur = ggml_rms_norm(ctx0, Vcur, eps);
                cb(Vcur, "Vcur_normed", il);
            }

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (layer.ls_1_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_1_w);
            cb(cur, "attn_out_scaled", il);
        }

        if (layer.attn_post_norm_w) {
            cur = build_norm(cur, layer.attn_post_norm_w, nullptr, norm_t, eps, il);
            cb(cur, "attn_post_normed", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2 (pre-ffn norm)
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            ffn_t, il);

        cb(cur, "ffn_out", il);

        if (layer.ff_post_norm_w) {
            cur = build_norm(cur, layer.ff_post_norm_w, nullptr, norm_t, eps, il);
            cb(cur, "ffn_post_normed", il);
        }

        if (layer.ls_2_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_2_w);
            cb(cur, "ffn_out_scaled", il);
        }

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        if (layer.ls_out_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_out_w);
            cb(cur, "layer_out_scaled", il);
        }

        inpL = cur;
    }

    if (model.audio_has_avgpool()) {
        ggml_tensor * cur = inpL;
        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont(ctx0, cur);
        cur = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, 2, 2, 0);
        cur = ggml_transpose(ctx0, cur);
        cur = ggml_cont(ctx0, cur);
        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, norm_t, eps, -1);
    }
    return inpL;
}

// build the input after conv2d (inp_raw --> patches)
// returns tensor with shape [n_embd, n_patches]
ggml_tensor * clip_graph::build_inp() {
    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_2d(ctx0, inp, n_patches, n_embd);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    if (model.patch_bias) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }
    return inp;
}

ggml_tensor * clip_graph::build_inp_raw(int channels) {
    ggml_tensor * inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, img.nx, img.ny, channels);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    return inp_raw;
}

ggml_tensor * clip_graph::build_norm(
        ggml_tensor * cur,
        ggml_tensor * mw,
        ggml_tensor * mb,
        norm_type type,
        float norm_eps,
        int il) const {

    cur = type == NORM_TYPE_RMS
        ? ggml_rms_norm(ctx0, cur, norm_eps)
        : ggml_norm(ctx0, cur, norm_eps);

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        cb(cur, "norm_w", il);
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
        cb(cur, "norm_b", il);
    }

    return cur;
}

ggml_tensor * clip_graph::build_ffn(
        ggml_tensor * cur,
        ggml_tensor * up,
        ggml_tensor * up_b,
        ggml_tensor * gate,
        ggml_tensor * gate_b,
        ggml_tensor * down,
        ggml_tensor * down_b,
        ffn_op_type type_op,
        int il) const {

    ggml_tensor * tmp = up ? build_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (gate) {
        cur = build_mm(gate, cur);
        cb(cur, "ffn_gate", il);

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }
    } else {
        cur = tmp;
    }

    // we only support parallel ffn for now
    switch (type_op) {
        case FFN_SILU:
            if (gate) {
                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case FFN_GELU:
            if (gate) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
            } break;
        case FFN_GELU_ERF:
            if (gate) {
                cur = ggml_geglu_erf_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_erf", il);
            } else {
                cur = ggml_gelu_erf(ctx0, cur);
                cb(cur, "ffn_gelu_erf", il);
            } break;
        case FFN_GELU_QUICK:
            if (gate) {
                cur = ggml_geglu_quick_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu_quick", il);
            } else {
                cur = ggml_gelu_quick(ctx0, cur);
                cb(cur, "ffn_gelu_quick", il);
            } break;
        case FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_relu_sqr", il);
            } break;
    }

    if (down) {
        cur = build_mm(down, cur);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    return cur;
}

ggml_tensor * clip_graph::build_attn(
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float kq_scale,
        int il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    //cb(q, "q", il);

    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    //cb(k, "k", il);

    ggml_tensor * cur;

    if (flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);

        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

    } else {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v = ggml_cont(ctx0, v);

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        // F32 may not needed for vision encoders?
        // ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }

    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_mm(wo, cur);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

// implementation of the 2D RoPE without adding a new op in ggml
// this is not efficient (use double the memory), but works on all backends
// TODO: there was a more efficient which relies on ggml_view and ggml_rope_ext_inplace, but the rope inplace does not work well with non-contiguous tensors ; we should fix that and revert back to the original implementation in https://github.com/ggml-org/llama.cpp/pull/13065
ggml_tensor * clip_graph::build_rope_2d(
    ggml_context * ctx0,
    ggml_tensor * cur,
    ggml_tensor * pos_a, // first half
    ggml_tensor * pos_b, // second half
    const float freq_base,
    const bool interleave_freq
) {
    const int64_t n_dim  = cur->ne[0];
    const int64_t n_head = cur->ne[1];
    const int64_t n_pos  = cur->ne[2];

    // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
    // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
    // first half of cur will use 1e-0, 1e-2 (even)
    // second half of cur will use 1e-1, 1e-3 (odd)
    // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
    //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
    // then for the second half, we use freq_scale to shift the inv_freq
    //  ^ why? replace (2i) with (2i+1) in the above equation
    const float freq_scale_odd = interleave_freq
                                ? std::pow(freq_base, (float)-2/n_dim)
                                : 1.0;

    // first half
    ggml_tensor * first;
    {
        first = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            cur->nb[1],
            cur->nb[2],
            0);
        first = ggml_rope_ext(
            ctx0,
            first,
            pos_a,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    // second half
    ggml_tensor * second;
    {
        second = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            cur->nb[1],
            cur->nb[2],
            n_dim/2 * ggml_element_size(cur));
        second = ggml_rope_ext(
            ctx0,
            second,
            pos_b,      // positions
            nullptr,    // freq factors
            n_dim/2,    // n_dims
            0, 0, freq_base,
            freq_scale_odd,
            0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    cur = ggml_concat(ctx0, first, second, 0);
    return cur;
}

// Generic function to stack frames for audio processing
// Abstracts out the StackAudioFrames logic used by ultravox
ggml_tensor * clip_graph::build_stack(ggml_tensor * cur, int32_t stack_factor, int32_t n_embed) {
    if (stack_factor <= 1) {
        return cur;
    }

    int64_t total_elements = ggml_nelements(cur);
    int64_t stride = n_embed * stack_factor;

    // Calculate padded length
    int64_t padded_len = GGML_PAD(total_elements, stride);
    int64_t pad = padded_len - total_elements;

    if (pad > 0) {
        // Pad the tensor to make it divisible by stride
        cur = ggml_view_1d(ctx0, cur, total_elements, 0);
        cur = ggml_pad(ctx0, cur, pad, 0, 0, 0);
    }

    // Reshape to [stride, padded_len / stride]
    cur = ggml_view_2d(ctx0, cur, stride, padded_len / stride,
                        ggml_row_size(cur->type, stride), 0);
    return cur;
}

// aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
// support dynamic resolution
ggml_tensor * clip_graph::build_patch_merge_permute(ggml_tensor * cur, int scale_factor) {
    GGML_ASSERT(scale_factor > 1);

    const int n_embd = cur->ne[0];
    int width  = img.nx / patch_size;
    int height = img.ny / patch_size;

    // pad width and height to factor
    const int64_t pad_width  = CLIP_ALIGN(width,  scale_factor) - width;
    const int64_t pad_height = CLIP_ALIGN(height, scale_factor) - height;
    cur = ggml_reshape_3d(ctx0, cur, n_embd, width, height);
    if (pad_width || pad_height) {
        cur     = ggml_pad(ctx0, cur, 0, pad_width, pad_height, 0);
        width  += pad_width;
        height += pad_height;
    }

    // unshuffle h
    cur = ggml_reshape_3d(ctx0, cur, n_embd * scale_factor, width / scale_factor, height);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    // unshuffle w
    cur = ggml_cont_3d(ctx0, cur, n_embd * scale_factor * scale_factor, height / scale_factor, width / scale_factor);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);

    cur = ggml_cont_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    cb(cur, "pixel_shuffle", -1);

    return cur;
}

static ggml_cgraph * clip_image_build_graph(clip_ctx * ctx, const clip_image_f32_batch & imgs) {
    GGML_ASSERT(imgs.entries.size() == 1 && "n_batch > 1 is not supported");

    const clip_image_f32 & img = *imgs.entries[0];
    std::unique_ptr<clip_graph> builder;

    switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_PHI4:
            {
                builder = std::make_unique<clip_graph_siglip>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA3NV:
            {
                builder = std::make_unique<clip_graph_mobilenetv5>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA4V:
            {
                builder = std::make_unique<clip_graph_gemma4v>(ctx, img);
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                builder = std::make_unique<clip_graph_pixtral>(ctx, img);
            } break;
        case PROJECTOR_TYPE_DOTS_OCR:
            {
                builder = std::make_unique<clip_graph_dotsocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
            {
                builder = std::make_unique<clip_graph_qwen2vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN3VL:
            {
                builder = std::make_unique<clip_graph_qwen3vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_STEP3VL:
            {
                builder = std::make_unique<clip_graph_step3vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                builder = std::make_unique<clip_graph_minicpmv>(ctx, img);
            } break;
        case PROJECTOR_TYPE_INTERNVL:
            {
                builder = std::make_unique<clip_graph_internvl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
            {
                builder = std::make_unique<clip_graph_nemotron_v2_vl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                builder = std::make_unique<clip_graph_llama4>(ctx, img);
            } break;
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            {
                builder = std::make_unique<clip_graph_whisper_enc>(ctx, img);
            } break;
        case PROJECTOR_TYPE_KIMIVL:
            {
                builder = std::make_unique<clip_graph_kimivl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_PADDLEOCR:
            {
                builder = std::make_unique<clip_graph_paddleocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_KIMIK25:
            {
                builder = std::make_unique<clip_graph_kimik25>(ctx, img);
            } break;
        case PROJECTOR_TYPE_COGVLM:
            {
                builder = std::make_unique<clip_graph_cogvlm>(ctx, img);
            } break;
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
            {
                builder = std::make_unique<clip_graph_hunyuanocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                builder = std::make_unique<clip_graph_llava>(ctx, img);
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR:
            {
                builder = std::make_unique<clip_graph_deepseekocr>(ctx, img);
            } break;
        case PROJECTOR_TYPE_LFM2A:
            {
                builder = std::make_unique<clip_graph_conformer>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GEMMA4A:
            {
                builder = std::make_unique<clip_graph_gemma4a>(ctx, img);
            } break;
        case PROJECTOR_TYPE_GLM4V:
            {
                builder = std::make_unique<clip_graph_glm4v>(ctx, img);
            } break;
        case PROJECTOR_TYPE_QWEN3A:
            {
                builder = std::make_unique<clip_graph_qwen3a>(ctx, img);
            } break;
        case PROJECTOR_TYPE_YOUTUVL:
            {
                builder = std::make_unique<clip_graph_youtuvl>(ctx, img);
            } break;
        case PROJECTOR_TYPE_YASA2:
            {
                builder = std::make_unique<clip_graph_yasa2>(ctx, img);
            } break;
        default:
            GGML_ABORT("missing cgraph builder");
    }

    return builder->build();
}

//
// clip_model_loader
//

struct clip_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    std::string fname;

    size_t model_size = 0; // in bytes

    bool has_vision = false;
    bool has_audio  = false;

    // TODO @ngxson : we should not pass clip_ctx here, it should be clip_model
    clip_model_loader(const char * fname) : fname(fname) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(string_format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname));
        }

        ctx_meta.reset(meta);

        const int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

        // print gguf info
        {
            std::string name;
            get_string(KEY_NAME, name, false);
            std::string description;
            get_string(KEY_DESCRIPTION, description, false);
            LOG_INF("%s: model name:   %s\n",  __func__, name.c_str());
            LOG_INF("%s: description:  %s\n",  __func__, description.c_str());
            LOG_INF("%s: GGUF version: %d\n",  __func__, gguf_get_version(ctx_gguf.get()));
            LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf.get()));
            LOG_INF("%s: n_tensors:    %d\n",  __func__, n_tensors);
            LOG_INF("%s: n_kv:         %d\n",  __func__, (int)gguf_get_n_kv(ctx_gguf.get()));
            LOG_INF("\n");
        }

        // modalities
        {
            get_bool(KEY_HAS_VISION_ENC, has_vision, false);
            get_bool(KEY_HAS_AUDIO_ENC,  has_audio,  false);

            if (has_vision) {
                LOG_INF("%s: has vision encoder\n", __func__);
            }
            if (has_audio) {
                LOG_INF("%s: has audio encoder\n", __func__);
            }
        }

        // tensors
        {
            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t offset = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf.get(), i);
                ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                    __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams(clip_model & model, clip_modality modality) {
        auto & hparams = model.hparams;
        std::string log_ffn_op; // for logging

        // sanity check
        if (modality == CLIP_MODALITY_VISION) {
            GGML_ASSERT(has_vision);
        } else if (modality == CLIP_MODALITY_AUDIO) {
            GGML_ASSERT(has_audio);
        }
        model.modality = modality;


        // projector type
        std::string proj_type;
        {
            // default key
            get_string(KEY_PROJ_TYPE, proj_type, false);

            // for models with mixed modalities
            if (proj_type.empty()) {
                if (modality == CLIP_MODALITY_VISION) {
                    get_string(KEY_VISION_PROJ_TYPE, proj_type, false);
                } else if (modality == CLIP_MODALITY_AUDIO) {
                    get_string(KEY_AUDIO_PROJ_TYPE, proj_type, false);
                } else {
                    GGML_ABORT("unknown modality");
                }
            }

            model.proj_type = clip_projector_type_from_string(proj_type);

            if (model.proj_type == PROJECTOR_TYPE_UNKNOWN) {
                throw std::runtime_error(string_format("%s: unknown projector type: %s\n", __func__, proj_type.c_str()));
            }

            // correct arch for multimodal models (legacy method)
            if (model.proj_type == PROJECTOR_TYPE_QWEN25O) {
                model.proj_type = modality == CLIP_MODALITY_VISION
                                    ? PROJECTOR_TYPE_QWEN25VL
                                    : PROJECTOR_TYPE_QWEN2A;
            }
        }

        const bool is_vision = model.modality == CLIP_MODALITY_VISION;
        const bool is_audio  = model.modality == CLIP_MODALITY_AUDIO;

        // other hparams
        {
            const char * prefix = is_vision ? "vision" : "audio";
            get_u32(string_format(KEY_N_EMBD,         prefix), hparams.n_embd);
            get_u32(string_format(KEY_N_HEAD,         prefix), hparams.n_head);
            get_u32(string_format(KEY_N_FF,           prefix), hparams.n_ff);
            get_u32(string_format(KEY_N_BLOCK,        prefix), hparams.n_layer);
            get_u32(string_format(KEY_PROJ_DIM,       prefix), hparams.projection_dim);
            get_f32(string_format(KEY_LAYER_NORM_EPS, prefix), hparams.eps);

            if (is_vision) {
                get_u32(KEY_IMAGE_SIZE, hparams.image_size);
                get_u32(KEY_PATCH_SIZE, hparams.patch_size);
                get_i32(KEY_MINICPMV_VERSION, hparams.minicpmv_version, false); // legacy
                get_u32(KEY_MINICPMV_QUERY_NUM, hparams.minicpmv_query_num, false);
                if (hparams.minicpmv_query_num == 0) {
                    // Fallback to hardcoded values for legacy models
                    if (hparams.minicpmv_version == 3) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 4) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 5) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 6) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 100045) {
                        hparams.minicpmv_query_num = 64;
                    } else {
                        hparams.minicpmv_query_num = 96;
                    }
                }
            } else if (is_audio) {
                get_u32(KEY_A_NUM_MEL_BINS, hparams.n_mel_bins);
                // some hparams are unused, but still need to set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;

            } else {
                GGML_ASSERT(false && "unknown modality");
            }

            // for pinpoints, we need to convert it into a list of resolution candidates
            {
                std::vector<int> pinpoints;
                get_arr_int(KEY_IMAGE_GRID_PINPOINTS, pinpoints, false);
                if (!pinpoints.empty()) {
                    for (size_t i = 0; i < pinpoints.size(); i += 2) {
                        hparams.image_res_candidates.push_back({
                            pinpoints[i],
                            pinpoints[i+1],
                        });
                    }
                }
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;

            {
                bool use_gelu = false;
                bool use_silu = false;
                get_bool(KEY_USE_GELU, use_gelu, false);
                get_bool(KEY_USE_SILU, use_silu, false);
                if (use_gelu && use_silu) {
                    throw std::runtime_error(string_format("%s: both use_gelu and use_silu are set to true\n", __func__));
                }
                if (use_gelu) {
                    hparams.ffn_op = FFN_GELU;
                    log_ffn_op = "gelu";
                } else if (use_silu) {
                    hparams.ffn_op = FFN_SILU;
                    log_ffn_op = "silu";
                } else {
                    hparams.ffn_op = FFN_GELU_QUICK;
                    log_ffn_op = "gelu_quick";
                }
            }

            {
                std::string mm_patch_merge_type;
                get_string(KEY_MM_PATCH_MERGE_TYPE, mm_patch_merge_type, false);
                if (mm_patch_merge_type == "spatial_unpad") {
                    hparams.mm_patch_merge_type = PATCH_MERGE_SPATIAL_UNPAD;
                }
            }

            if (is_vision) {
                int idx_mean = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_MEAN);
                int idx_std  = gguf_find_key(ctx_gguf.get(), KEY_IMAGE_STD);
                GGML_ASSERT(idx_mean >= 0 && "image_mean not found");
                GGML_ASSERT(idx_std >= 0  && "image_std not found");
                const float * mean_data = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_mean);
                const float * std_data  = (const float *) gguf_get_arr_data(ctx_gguf.get(), idx_std);
                for (int i = 0; i < 3; ++i) {
                    hparams.image_mean[i] = mean_data[i];
                    hparams.image_std[i]  = std_data[i];
                }
            }

            // Load the vision feature layer indices if they are explicitly provided;
            // if multiple vision feature layers are present, the values will be concatenated
            // to form the final visual features.
            // NOTE: gguf conversions should standardize the values of the vision feature layer to
            // be non-negative, since we use -1 to mark values as unset here.
            std::vector<int> vision_feature_layer;
            get_arr_int(KEY_FEATURE_LAYER, vision_feature_layer, false);
            // convert std::vector to std::unordered_set
            for (auto & layer : vision_feature_layer) {
                hparams.vision_feature_layer.insert(layer);
            }

            // model-specific params
            switch (model.proj_type) {
                case PROJECTOR_TYPE_MLP:
                case PROJECTOR_TYPE_MLP_NORM:
                case PROJECTOR_TYPE_LDP:
                case PROJECTOR_TYPE_LDPV2:
                case PROJECTOR_TYPE_COGVLM:
                    {
                        hparams.has_llava_projector = model.proj_type != PROJECTOR_TYPE_COGVLM;
                        hparams.image_pad_color     = {122, 116, 104};
                        if (!hparams.image_res_candidates.empty()) {
                            hparams.image_resize_pad  = true;
                            hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        } else {
                            // llava-1.6 default params
                            hparams.image_pad_ov         = false;
                            hparams.image_pad_rf         = true;
                            hparams.image_pad_color_rf   = {122, 116, 104};
                            hparams.image_resize_algo_rf = RESIZE_ALGO_BICUBIC;
                            hparams.image_resize_algo_ov = RESIZE_ALGO_BILINEAR;
                        }
                    } break;
                case PROJECTOR_TYPE_GLM_EDGE:
                    {
                        hparams.image_resize_pad  = true;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                    } break;
                case PROJECTOR_TYPE_MINICPMV:
                    {
                        // use default llava-uhd preprocessing params
                        if (hparams.minicpmv_version == 0) {
                            hparams.minicpmv_version = 2; // default to 2 if not set
                        }
                    } break;
                case PROJECTOR_TYPE_INTERNVL:
                    {
                        // use default llava-uhd preprocessing params
                        // older version of internvl doesn't have min/max tiles, we need to provide default values for them to avoid issues
                        hparams.preproc_min_tiles = 1;
                        hparams.preproc_max_tiles = 12;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        get_u32(KEY_PREPROC_MIN_TILES, hparams.preproc_min_tiles, false);
                        get_u32(KEY_PREPROC_MAX_TILES, hparams.preproc_max_tiles, false);
                        GGML_ASSERT(hparams.preproc_min_tiles <= hparams.preproc_max_tiles && hparams.preproc_max_tiles < INT32_MAX);
                        set_internvl_dhr_res_candidates(model);
                    } break;
                case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                    {
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;
                case PROJECTOR_TYPE_IDEFICS3:
                    {
                        // use default llava-uhd preprocessing params
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                    } break;
                case PROJECTOR_TYPE_LFM2:
                    {
                        hparams.image_resize_algo    = RESIZE_ALGO_BILINEAR;
                        hparams.image_resize_algo_rf = RESIZE_ALGO_BILINEAR;
                        hparams.image_resize_algo_ov = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // ref: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/processor_config.json
                        hparams.set_limit_image_tokens(64, 256);
                    } break;
                case PROJECTOR_TYPE_PHI4:
                    {
                        hparams.n_merge = 1;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(16*16);
                    } break;
                case PROJECTOR_TYPE_PIXTRAL:
                    {
                        // ref: https://huggingface.co/mistral-community/pixtral-12b/blob/main/preprocessor_config.json
                        // TODO: verify the image_min_tokens
                        hparams.n_merge = 1; // the original pixtral does not use patch merging
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.set_limit_image_tokens(8, 1024);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_LIGHTONOCR:
                    {
                        hparams.n_merge = 1;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.image_longest_edge = hparams.image_size;
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_DOTS_OCR:
                    {
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_KIMIVL:
                    {
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // TODO: check kimivl preprocessor for exact values
                        hparams.set_limit_image_tokens(8, 1024);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_KIMIK25:
                    {
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);

                        int min_pixels = 0, max_pixels = 0;
                        get_u32(KEY_IMAGE_MIN_PIXELS, min_pixels, false);
                        get_u32(KEY_IMAGE_MAX_PIXELS, max_pixels, false);
                        if (min_pixels > 0 && max_pixels > 0) {
                            hparams.image_min_pixels = min_pixels;
                            hparams.image_max_pixels = max_pixels;
                            hparams.warmup_image_size = static_cast<int>(std::sqrt(max_pixels));
                        } else {
                            hparams.set_limit_image_tokens(2, 4096);
                        }
                    } break;
                case PROJECTOR_TYPE_GEMMA3:
                    {
                        // default value (used by all model sizes in gemma 3 family)
                        // number of patches for each **side** is reduced by a factor of 4
                        hparams.n_merge = 4;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        // test model (tinygemma3) has a different value, we optionally read it
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;

                case PROJECTOR_TYPE_GEMMA4V:
                    {
                        hparams.rope_theta = 100.0f;
                        hparams.n_merge = 3; // pooling_kernel_size
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // @ngxson : the model performs quite poor with small images, we need to bump minimum image tokens to 40 to avoid that
                        hparams.set_limit_image_tokens(252, 280);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;

                case PROJECTOR_TYPE_GEMMA3NV:
                    {
                        // Gemma3n uses MobileNetV5 which produces 256 tokens (16x16)
                        // Similar configuration to Gemma3
                        hparams.n_merge = 1;  // MobileNetV5 handles resizing internally
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;
                case PROJECTOR_TYPE_QWEN2VL:
                case PROJECTOR_TYPE_QWEN25VL:
                case PROJECTOR_TYPE_QWEN3VL:
                    {
                        hparams.n_merge = 2; // default value for Qwen 2 and 2.5
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_WIN_ATTN_PATTERN, hparams.n_wa_pattern, model.proj_type == PROJECTOR_TYPE_QWEN25VL); // only 2.5 requires it
                        // ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
                        hparams.set_limit_image_tokens(8, 4096);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                        const int warn_min_pixels = 1024 * hparams.n_merge * hparams.n_merge * hparams.patch_size * hparams.patch_size;
                        if (hparams.image_min_pixels < warn_min_pixels) {
                            LOG_WRN("%s: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks\n", __func__);
                            LOG_WRN("%s: if you encounter problems with accuracy, try adding --image-min-tokens 1024\n", __func__);
                            LOG_WRN("%s: more info: https://github.com/ggml-org/llama.cpp/issues/16842\n\n", __func__);
                        }
                    } break;
                case PROJECTOR_TYPE_STEP3VL:
                    {
                        hparams.n_merge = 4; // two stride-2 downsamplers after patching
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                        if (hparams.image_longest_edge == 0) {
                            hparams.image_longest_edge = 3024;
                        }
                        hparams.warmup_image_size = hparams.image_size;
                    } break;
                case PROJECTOR_TYPE_YOUTUVL:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        hparams.image_resize_pad  = false;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_ATTN_WINDOW_SIZE, hparams.attn_window_size, true);
                        std::vector<int> wa_layer_indexes_vec;
                        get_arr_int(KEY_WIN_ATTN_LAYER_INDEXES, wa_layer_indexes_vec, true);
                        for (auto & layer : wa_layer_indexes_vec) {
                            hparams.wa_layer_indexes.insert(layer);
                        }
                        // support max_height * max_width = 8000 * 8000. 8000/16/2 = 250 image tokens
                        hparams.set_limit_image_tokens(1, 62500);
                        hparams.set_warmup_n_tokens(16*16); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_YASA2:
                    {
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf";
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;

                        // reka model performs better when using resize_bicubic, which stretches
                        // the image to fit fixed square size
                        hparams.image_resize_pad = false;
                    } break;
                case PROJECTOR_TYPE_GLM4V:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.n_merge = 2; // default value for GLM4-V
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.set_limit_image_tokens(8, 4096);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_LLAMA4:
                    {
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        set_llava_uhd_res_candidates(model, 3);
                    } break;
                case PROJECTOR_TYPE_ULTRAVOX:
                case PROJECTOR_TYPE_QWEN2A:
                case PROJECTOR_TYPE_QWEN3A:
                case PROJECTOR_TYPE_GLMA:
                case PROJECTOR_TYPE_VOXTRAL:
                case PROJECTOR_TYPE_MERALION:
                case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                    {
                        bool require_stack = model.proj_type == PROJECTOR_TYPE_ULTRAVOX ||
                                             model.proj_type == PROJECTOR_TYPE_VOXTRAL ||
                                             model.proj_type == PROJECTOR_TYPE_MERALION ||
                                             model.proj_type == PROJECTOR_TYPE_GLMA;
                        get_u32(KEY_A_PROJ_STACK_FACTOR, hparams.proj_stack_factor, require_stack);
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf"; // temporary solution for logging

                        // audio preprocessing params
                        hparams.audio_chunk_len    = 30; // in seconds
                        hparams.audio_sample_rate  = 16000;
                        hparams.audio_n_fft        = 400;
                        hparams.audio_window_len   = 400;
                        hparams.audio_hop_len      = 160;
                    } break;
                case PROJECTOR_TYPE_PADDLEOCR:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);

                        hparams.set_warmup_n_tokens(28*28); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_DEEPSEEKOCR:
                    {
                        hparams.patch_size = 16;
                        hparams.image_size = 1024;
                        hparams.warmup_image_size = 1024;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC_PILLOW;
                        hparams.image_pad_color[0] = hparams.image_mean[0];
                        hparams.image_pad_color[1] = hparams.image_mean[1];
                        hparams.image_pad_color[2] = hparams.image_mean[2];

                        get_u32(KEY_SAM_N_BLOCK, hparams.sam_n_layer, true);
                        get_u32(KEY_SAM_N_HEAD, hparams.sam_n_head, true);
                        get_u32(KEY_SAM_N_EMBD, hparams.sam_n_embd, true);
                        get_u32(KEY_ATTN_WINDOW_SIZE, hparams.attn_window_size, true);
                     } break;
                case PROJECTOR_TYPE_HUNYUANOCR:
                    {
                        hparams.n_merge = 2;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(28*28);
                    } break;
                case PROJECTOR_TYPE_HUNYUANVL:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC_PILLOW;
                        hparams.image_resize_pad = false;
                        hparams.ffn_op = FFN_GELU;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.set_limit_image_tokens(256, 16384);
                        hparams.set_warmup_n_tokens(32*32);
                    } break;
                case PROJECTOR_TYPE_LFM2A:
                    {
                        // audio preprocessing params
                        hparams.audio_chunk_len        = 1; // in seconds
                        hparams.audio_sample_rate      = 16000;
                        hparams.audio_n_fft            = 512;
                        hparams.audio_window_len       = 400;
                        hparams.audio_hop_len          = 160;
                    } break;
                case PROJECTOR_TYPE_GEMMA4A:
                    {
                        // Gemma4 feature_extraction_gemma4.py:
                        // frame_length_ms=20 -> 320 samples, n_fft=512, hop=10ms -> 160
                        hparams.audio_chunk_len        = 0;  // no fixed-length padding
                        hparams.audio_sample_rate      = 16000;
                        hparams.audio_n_fft            = 512;
                        hparams.audio_window_len       = 320;  // 20ms frame (NOT 25ms/400)
                        hparams.audio_hop_len          = 160;
                    } break;
                case PROJECTOR_TYPE_JANUS_PRO:
                    {
                        hparams.image_pad_color   = {127, 127, 127};
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                    } break;
                default:
                    throw std::runtime_error(string_format("%s: unknown vision projector type %s\n", __func__, proj_type.c_str()));
            }

            // sanity check
            {
                if (hparams.image_size < 0) {
                    // note: some models having hparams.image_size == 0, which means the image size is dynamic
                    throw std::runtime_error(string_format("%s: image_size (%d) cannot be negative\n", __func__, hparams.image_size));
                }
                if (hparams.patch_size <= 0) {
                    throw std::runtime_error(string_format("%s: patch_size (%d) must be greater than 0\n", __func__, hparams.patch_size));
                }
                if (hparams.n_embd <= 0) {
                    throw std::runtime_error(string_format("%s: n_embd (%d) must be greater than 0\n", __func__, hparams.n_embd));
                }
                if (hparams.image_max_pixels < hparams.image_min_pixels) {
                    throw std::runtime_error(string_format("%s: image_max_pixels (%d) is less than image_min_pixels (%d)\n", __func__, hparams.image_max_pixels, hparams.image_min_pixels));
                }
            }

            LOG_INF("%s: projector:          %s\n", __func__, proj_type.c_str());
            LOG_INF("%s: n_embd:             %d\n", __func__, hparams.n_embd);
            LOG_INF("%s: n_head:             %d\n", __func__, hparams.n_head);
            LOG_INF("%s: n_ff:               %d\n", __func__, hparams.n_ff);
            LOG_INF("%s: n_layer:            %d\n", __func__, hparams.n_layer);
            LOG_INF("%s: ffn_op:             %s\n", __func__, log_ffn_op.c_str());
            LOG_INF("%s: projection_dim:     %d\n", __func__, hparams.projection_dim);
            if (is_vision) {
                LOG_INF("\n--- vision hparams ---\n");
                LOG_INF("%s: image_size:         %d\n", __func__, hparams.image_size);
                LOG_INF("%s: patch_size:         %d\n", __func__, hparams.patch_size);
                LOG_INF("%s: has_llava_proj:     %d\n", __func__, hparams.has_llava_projector);
                LOG_INF("%s: minicpmv_version:   %d\n", __func__, hparams.minicpmv_version);
                LOG_INF("%s: n_merge:            %d\n", __func__, hparams.n_merge);
                LOG_INF("%s: n_wa_pattern: %d\n", __func__, hparams.n_wa_pattern);
                if (!hparams.wa_layer_indexes.empty()) {
                    LOG_INF("%s: wa_layer_indexes:  ", __func__);
                    for (auto & layer : hparams.wa_layer_indexes) {
                        LOG_INF("%d ", layer);
                    }
                    LOG_INF("\n");
                }
                if (hparams.image_min_pixels > 0) {
                    LOG_INF("%s: image_min_pixels:   %d%s\n", __func__, hparams.image_min_pixels, hparams.custom_image_min_tokens > 0 ? " (custom value)" : "");
                }
                if (hparams.image_max_pixels > 0) {
                    LOG_INF("%s: image_max_pixels:   %d%s\n", __func__, hparams.image_max_pixels, hparams.custom_image_max_tokens > 0 ? " (custom value)" : "");
                }
            } else if (is_audio) {
                LOG_INF("\n--- audio hparams ---\n");
                LOG_INF("%s: n_mel_bins:         %d\n", __func__, hparams.n_mel_bins);
                LOG_INF("%s: proj_stack_factor:  %d\n", __func__, hparams.proj_stack_factor);
                LOG_INF("%s: audio_chunk_len:    %d\n", __func__, hparams.audio_chunk_len);
                LOG_INF("%s: audio_sample_rate:  %d\n", __func__, hparams.audio_sample_rate);
                LOG_INF("%s: audio_n_fft:        %d\n", __func__, hparams.audio_n_fft);
                LOG_INF("%s: audio_window_len:   %d\n", __func__, hparams.audio_window_len);
                LOG_INF("%s: audio_hop_len:      %d\n", __func__, hparams.audio_hop_len);
            }
            LOG_INF("\n");
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors(clip_ctx & ctx_clip) {
        auto & model = ctx_clip.model;
        auto & hparams = model.hparams;
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;

        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            throw std::runtime_error(string_format("%s: failed to open %s\n", __func__, fname.c_str()));
        }

        // TODO @ngxson : support both audio and video in the future
        const char * prefix = model.modality == CLIP_MODALITY_AUDIO ? "a" : "v";

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        struct ggml_init_params params = {
            /*.mem_size =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        ctx_clip.ctx_data.reset(ggml_init(params));
        if (!ctx_clip.ctx_data) {
            throw std::runtime_error(string_format("%s: failed to init ggml context\n", __func__));
        }

        // helper function
        std::unordered_set<std::string> loaded_tensor_names;
        auto get_tensor = [&](const std::string & name, bool required = true) {
            // Each tensor should only be loaded once; duplicates indicate a bug
            if (loaded_tensor_names.count(name)) {
                throw std::runtime_error(string_format("%s: tensor already loaded: %s\n", __func__, name.c_str()));
            }
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                ggml_tensor * data_tensor = ggml_dup_tensor(ctx_clip.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                loaded_tensor_names.insert(name);
                cur = data_tensor;
            }
            return cur;
        };

        auto get_scalar = [&](const std::string & name, float default_val) {
            auto it = tensor_offset.find(name);
            if (it == tensor_offset.end()) {
                return default_val;
            }
            size_t offset = it->second;
            fin.seekg(offset, std::ios::beg);
            float value;
            fin.read(reinterpret_cast<char*>(&value), sizeof(float));
            return value;
        };

        model.class_embedding = get_tensor(TN_CLASS_EMBD, false);

        model.pre_ln_w = get_tensor(string_format(TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = get_tensor(string_format(TN_LN_PRE, prefix, "bias"),   false);

        model.post_ln_w = get_tensor(string_format(TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = get_tensor(string_format(TN_LN_POST, prefix, "bias"),   false);

        model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = get_tensor(TN_PATCH_EMBD,   false);
        model.patch_embeddings_1 = get_tensor(TN_PATCH_EMBD_1, false);

        model.norm_embd_w = get_tensor(string_format(TN_NORM_EMBD, "weight"), false);
        model.norm_embd_b = get_tensor(string_format(TN_NORM_EMBD, "bias"),   false);

        model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, prefix), false);

        if (model.proj_type == PROJECTOR_TYPE_GEMMA3NV) {
            hparams.n_layer = 0; // gemma3n does not use normal layer structure
        }

        // layers
        model.layers.resize(hparams.n_layer);
        for (int il = 0; il < hparams.n_layer; ++il) {
            auto & layer = model.layers[il];
            layer.k_w    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "weight"), false);
            layer.q_w    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "weight"), false);
            layer.v_w    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "weight"), false);
            layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "weight"));
            layer.qkv_w  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "weight"), false);
            layer.k_norm = get_tensor(string_format(TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = get_tensor(string_format(TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = get_tensor(string_format(TN_LN_1,        prefix, il, "weight"), false);
            layer.ln_2_w = get_tensor(string_format(TN_LN_2,        prefix, il, "weight"), false);
            layer.ls_1_w        = get_tensor(string_format(TN_LS_1,         prefix, il, "weight"), false); // no bias
            layer.ls_2_w        = get_tensor(string_format(TN_LS_2,         prefix, il, "weight"), false); // no bias
            layer.ls_out_w      = get_tensor(string_format(TN_LS_OUT,        prefix, il, "weight"), false); // no bias
            layer.attn_post_norm_w = get_tensor(string_format(TN_ATTN_POST_NORM, prefix, il, "weight"), false); // no bias
            layer.ff_post_norm_w   = get_tensor(string_format(TN_FFN_POST_NORM,  prefix, il, "weight"), false); // no bias

            layer.k_b    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "bias"), false);
            layer.q_b    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "bias"), false);
            layer.v_b    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "bias"), false);
            layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.qkv_b  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "bias"), false);
            layer.ln_1_b = get_tensor(string_format(TN_LN_1,        prefix, il, "bias"), false);
            layer.ln_2_b = get_tensor(string_format(TN_LN_2,        prefix, il, "bias"), false);

            // ffn
            layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "weight"));
            layer.ff_up_b   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "bias"),   false);
            layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = get_tensor(string_format(TN_FFN_GATE, prefix, il, "bias"),   false);
            layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "weight"));
            layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "bias"),   false);


            // qwen3vl deepstack layer
            layer.deepstack_norm_w = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "weight"), false);
            layer.deepstack_norm_b = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "bias"), false);
            layer.deepstack_fc1_w  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "weight"), false);
            layer.deepstack_fc1_b  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "bias"), false);
            layer.deepstack_fc2_w  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "weight"), false);
            layer.deepstack_fc2_b  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "bias"), false);
            if (layer.has_deepstack()) {
                model.n_deepstack_layers++;
            }

            // some models already exported with legacy (incorrect) naming which is quite messy, let's fix it here
            // note: Qwen model converted from the old surgery script has n_ff = 0, so we cannot use n_ff to check!
            bool is_ffn_swapped = (
                    // only old models need this fix
                    model.proj_type == PROJECTOR_TYPE_MLP
                    || model.proj_type == PROJECTOR_TYPE_MLP_NORM
                    || model.proj_type == PROJECTOR_TYPE_LDP
                    || model.proj_type == PROJECTOR_TYPE_LDPV2
                    || model.proj_type == PROJECTOR_TYPE_QWEN2VL
                    || model.proj_type == PROJECTOR_TYPE_QWEN25VL
                    || model.proj_type == PROJECTOR_TYPE_GLM_EDGE
                    || model.proj_type == PROJECTOR_TYPE_GEMMA3
                    || model.proj_type == PROJECTOR_TYPE_IDEFICS3
                    || model.proj_type == PROJECTOR_TYPE_MINICPMV
                ) && layer.ff_up_w && layer.ff_down_w && layer.ff_down_w->ne[0] == hparams.n_embd;
            if (is_ffn_swapped) {
                // swap up and down weights
                ggml_tensor * tmp = layer.ff_up_w;
                layer.ff_up_w = layer.ff_down_w;
                layer.ff_down_w = tmp;
                // swap up and down biases
                tmp = layer.ff_up_b;
                layer.ff_up_b = layer.ff_down_b;
                layer.ff_down_b = tmp;
                if (il == 0) {
                    LOG_WRN("%s: ffn up/down are swapped\n", __func__);
                }
            }
        }


        switch (model.proj_type) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
                {
                    // LLaVA projection
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"), false);
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    // Yi-type llava
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"), false);
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    // missing in Yi-type llava
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"), false);
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // Yi-type llava
                    model.mm_3_w = get_tensor(string_format(TN_LLAVA_PROJ, 3, "weight"), false);
                    model.mm_3_b = get_tensor(string_format(TN_LLAVA_PROJ, 3, "bias"), false);
                    model.mm_4_w = get_tensor(string_format(TN_LLAVA_PROJ, 4, "weight"), false);
                    model.mm_4_b = get_tensor(string_format(TN_LLAVA_PROJ, 4, "bias"), false);
                    if (model.mm_3_w) {
                        // TODO: this is a hack to support Yi-type llava
                        model.proj_type = PROJECTOR_TYPE_MLP_NORM;
                    }
                    model.image_newline = get_tensor(TN_IMAGE_NEWLINE, false);
                } break;
            case PROJECTOR_TYPE_LDP:
                {
                    // MobileVLM projection
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_model_mlp_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                    model.mm_model_block_1_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "0.weight"));
                    model.mm_model_block_1_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.weight"));
                    model.mm_model_block_1_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.bias"));
                    model.mm_model_block_1_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.weight"));
                    model.mm_model_block_1_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.bias"));
                    model.mm_model_block_1_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.weight"));
                    model.mm_model_block_1_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.bias"));
                    model.mm_model_block_1_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "0.weight"));
                    model.mm_model_block_1_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.weight"));
                    model.mm_model_block_1_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.bias"));
                    model.mm_model_block_2_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "0.weight"));
                    model.mm_model_block_2_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.weight"));
                    model.mm_model_block_2_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.bias"));
                    model.mm_model_block_2_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.weight"));
                    model.mm_model_block_2_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.bias"));
                    model.mm_model_block_2_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.weight"));
                    model.mm_model_block_2_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.bias"));
                    model.mm_model_block_2_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "0.weight"));
                    model.mm_model_block_2_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.weight"));
                    model.mm_model_block_2_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.bias"));
                } break;
            case PROJECTOR_TYPE_LDPV2:
                {
                    // MobilVLM_V2 projection
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_model_mlp_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                    model.mm_model_mlp_2_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "bias"));
                    model.mm_model_peg_0_w = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "weight"));
                    model.mm_model_peg_0_b = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "bias"));
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    // model.mm_model_pos_embed = get_tensor(new_clip->ctx_data, TN_MINICPMV_POS_EMBD);
                    model.mm_model_pos_embed_k = get_tensor(TN_MINICPMV_POS_EMBD_K);
                    model.mm_model_query = get_tensor(TN_MINICPMV_QUERY);
                    model.mm_model_proj = get_tensor(TN_MINICPMV_PROJ);
                    model.mm_model_kv_proj = get_tensor(TN_MINICPMV_KV_PROJ);
                    model.mm_model_attn_q_w = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "weight"));
                    model.mm_model_attn_k_w = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "weight"));
                    model.mm_model_attn_v_w = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "weight"));
                    model.mm_model_attn_q_b = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "bias"));
                    model.mm_model_attn_k_b = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "bias"));
                    model.mm_model_attn_v_b = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "bias"));
                    model.mm_model_attn_o_w = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "weight"));
                    model.mm_model_attn_o_b = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "bias"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_MINICPMV_LN, "q", "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_MINICPMV_LN, "q", "bias"));
                    model.mm_model_ln_kv_w = get_tensor(string_format(TN_MINICPMV_LN, "kv", "weight"));
                    model.mm_model_ln_kv_b = get_tensor(string_format(TN_MINICPMV_LN, "kv", "bias"));
                    model.mm_model_ln_post_w = get_tensor(string_format(TN_MINICPMV_LN, "post", "weight"));
                    model.mm_model_ln_post_b = get_tensor(string_format(TN_MINICPMV_LN, "post", "bias"));
                } break;
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    model.mm_model_adapter_conv_w = get_tensor(string_format(TN_GLM_ADAPER_CONV, "weight"));
                    model.mm_model_adapter_conv_b = get_tensor(string_format(TN_GLM_ADAPER_CONV, "bias"));
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_GLM_ADAPTER_LINEAR, "weight"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "bias"));
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_GLM_ADAPTER_D_H_2_4H, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_GLM_ADAPTER_GATE, "weight"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_GLM_ADAPTER_D_4H_2_H, "weight"));
                    model.mm_boi = get_tensor(string_format(TN_TOK_GLM_BOI));
                    model.mm_eoi = get_tensor(string_format(TN_TOK_GLM_EOI));
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN3VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_STEP3VL:
                {
                    model.mm_0_w     = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b     = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    model.mm_1_w     = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b     = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_model_proj = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                } break;
            case PROJECTOR_TYPE_YOUTUVL:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);        // merger.ln_q (RMS norm)
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));  // merger.mlp.0
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));  // merger.mlp.2
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_YASA2:
                {
                    // reuse tensors already loaded by the common section
                    // (TN_PATCH_EMBD and TN_PATCH_BIAS have the same tensor names)
                    GGML_ASSERT(model.patch_embeddings_0 && "yasa2 requires v.patch_embd.weight");
                    model.yasa_patch_w = model.patch_embeddings_0;
                    model.yasa_patch_b = model.patch_bias;
                    model.yasa_patch_ln_w = get_tensor(TN_YASA_PATCH_LN_W, false);
                    model.yasa_patch_ln_b = get_tensor(TN_YASA_PATCH_LN_B, false);
                    model.yasa_backbone_ln_w = get_tensor(TN_YASA_BACKBONE_LN_W, false);
                    model.yasa_backbone_ln_b = get_tensor(TN_YASA_BACKBONE_LN_B, false);
                    model.yasa_vision_pos_embed = get_tensor(TN_YASA_POS_EMBD, false);
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);

                    model.yasa_stages.clear();
                    for (int s = 0; ; ++s) {
                        yasa2_stage stage;
                        stage.down_ln_w   = get_tensor(string_format(TN_YASA_STAGE_DOWN_LN, s, "weight"), false);
                        stage.down_ln_b   = get_tensor(string_format(TN_YASA_STAGE_DOWN_LN, s, "bias"), false);
                        stage.down_conv_w = get_tensor(string_format(TN_YASA_STAGE_DOWN_CONV, s, "weight"), false);
                        stage.down_conv_b = get_tensor(string_format(TN_YASA_STAGE_DOWN_CONV, s, "bias"), false);

                        for (int bi = 0; ; ++bi) {
                            yasa2_block blk;
                            blk.dw_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "dw", "weight"), false);
                            if (!blk.dw_w) {
                                break;
                            }
                            blk.dw_b  = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "dw", "bias"), false);
                            blk.ln_w  = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "ln", "weight"), false);
                            blk.ln_b  = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "ln", "bias"), false);
                            blk.pw1_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw1", "weight"), false);
                            blk.pw1_b = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw1", "bias"), false);
                            blk.grn_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "grn", "weight"), false);
                            blk.grn_b = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "grn", "bias"), false);
                            blk.pw2_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw2", "weight"), false);
                            blk.pw2_b = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw2", "bias"), false);
                            stage.blocks.push_back(blk);
                        }

                        if (!stage.down_conv_w && stage.blocks.empty()) {
                            break;
                        }
                        model.yasa_stages.push_back(std::move(stage));
                    }
                } break;
            case PROJECTOR_TYPE_GLM4V:
                {
                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_ffn_up_w    = get_tensor(string_format(TN_MM_UP,        "weight"));
                    model.mm_ffn_up_b    = get_tensor(string_format(TN_MM_UP,        "bias"), false);
                    model.mm_ffn_gate_w  = get_tensor(string_format(TN_MM_GATE,      "weight"));
                    model.mm_ffn_gate_b  = get_tensor(string_format(TN_MM_GATE,      "bias"), false);
                    model.mm_ffn_down_w  = get_tensor(string_format(TN_MM_DOWN,      "weight"));
                    model.mm_ffn_down_b  = get_tensor(string_format(TN_MM_DOWN,      "bias"), false);
                    model.mm_post_norm_w = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                    model.mm_post_norm_b = get_tensor(string_format(TN_MM_POST_NORM, "bias"), false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"));
                    model.mm_patch_merger_b = get_tensor(string_format(TN_MM_PATCH_MERGER, "bias"));
                } break;
            case PROJECTOR_TYPE_GEMMA3:
                {
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.mm_soft_emb_norm_w = get_tensor(TN_MM_SOFT_EMB_N);
                } break;
            case PROJECTOR_TYPE_GEMMA4V:
                {
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.std_bias  = get_tensor(TN_STD_BIAS,  false);
                    model.std_scale = get_tensor(TN_STD_SCALE, false);
                    // load scalar for Gemma4ClippableLinear
                    for (auto * tensor : tensors_to_load) {
                        std::string name = tensor->name;
                        if (string_ends_with(name, ".weight")) {
                            std::string name_inp_max = name;
                            std::string name_inp_min = name;
                            std::string name_out_max = name;
                            std::string name_out_min = name;
                            string_replace_all(name_inp_max, ".weight", ".input_max");
                            string_replace_all(name_inp_min, ".weight", ".input_min");
                            string_replace_all(name_out_max, ".weight", ".output_max");
                            string_replace_all(name_out_min, ".weight", ".output_min");
                            model.clamp_info_map[name] = {
                                get_scalar(name_inp_max, FLT_MAX),
                                get_scalar(name_inp_min, -FLT_MAX),
                                get_scalar(name_out_max, FLT_MAX),
                                get_scalar(name_out_min, -FLT_MAX)
                            };
                        }
                    }
                } break;
            case PROJECTOR_TYPE_GEMMA3NV:
                {
                    model.mobilenet_stem_conv_w = get_tensor(TN_MNV5_STEM_CONV, false);
                    model.mobilenet_stem_conv_b = get_tensor(TN_MNV5_STEM_BIAS, false);
                    model.mobilenet_stem_norm_w = get_tensor(TN_MNV5_STEM_BN, false);

                    model.msfa_ffn_expand_w  = get_tensor(TN_MNV5_MSFA_FFN_EXP_W, false);
                    model.msfa_ffn_expand_bn = get_tensor(TN_MNV5_MSFA_FFN_EXP_BN, false); // Consume BN if present but likely folded
                    model.msfa_ffn_project_w = get_tensor(TN_MNV5_MSFA_FFN_PROJ_W, false);
                    model.msfa_ffn_project_bn = get_tensor(TN_MNV5_MSFA_FFN_PROJ_BN, false);

                    model.msfa_concat_norm_w = get_tensor(TN_MNV5_MSFA_NORM, false);

                    // Dynamically load blocks stage by stage
                    for (int stage = 0; stage < 4; ++stage) {
                        int blocks_found_in_stage = 0;

                        for (int blk_idx = 0; ; ++blk_idx) {
                            bool found_block = false;
                            mobilenetv5_block block;

                            // 1. Check for Edge Residual (S0)
                            block.s0_conv_exp_w = get_tensor(string_format(TN_MNV5_BLK_S0_EXP_W, stage, blk_idx), false);
                            if (block.s0_conv_exp_w) {
                                found_block = true;
                                block.s0_bn1_w      = get_tensor(string_format(TN_MNV5_BLK_S0_BN1_W, stage, blk_idx), false);
                                block.s0_conv_pwl_w = get_tensor(string_format(TN_MNV5_BLK_S0_PWL_W, stage, blk_idx), false);
                                block.s0_bn2_w      = get_tensor(string_format(TN_MNV5_BLK_S0_BN2_W, stage, blk_idx), false);
                            }
                            // 2. Check for UIR (Universal Inverted Residual)
                            else {
                                // Check for dw_start OR pw_exp (some UIR blocks skip dw_start)
                                block.dw_start_w = get_tensor(string_format(TN_MNV5_BLK_DW_START_W, stage, blk_idx), false);
                                block.pw_exp_w   = get_tensor(string_format(TN_MNV5_BLK_PW_EXP_W, stage, blk_idx), false);

                                if (block.dw_start_w || block.pw_exp_w) {
                                    found_block = true;
                                    if (block.dw_start_w) {
                                        block.dw_start_bn_w = get_tensor(string_format(TN_MNV5_BLK_DW_START_BN, stage, blk_idx), false);
                                    }
                                    if (block.pw_exp_w) {
                                        block.pw_exp_bn_w   = get_tensor(string_format(TN_MNV5_BLK_PW_EXP_BN, stage, blk_idx), false);
                                    }
                                    block.dw_mid_w      = get_tensor(string_format(TN_MNV5_BLK_DW_MID_W, stage, blk_idx), false);
                                    if (block.dw_mid_w) {
                                        block.dw_mid_bn_w   = get_tensor(string_format(TN_MNV5_BLK_DW_MID_BN, stage, blk_idx), false);
                                    }
                                    block.pw_proj_w     = get_tensor(string_format(TN_MNV5_BLK_PW_PROJ_W, stage, blk_idx), false);
                                    if (block.pw_proj_w) {
                                        block.pw_proj_bn_w  = get_tensor(string_format(TN_MNV5_BLK_PW_PROJ_BN, stage, blk_idx), false);
                                    }
                                    block.layer_scale_w = get_tensor(string_format(TN_MNV5_BLK_LAYER_SCALE, stage, blk_idx), false);
                                }
                            }

                            // 3. Check for Attention (MQA)
                            // Even if UIR/Edge check failed, this might be a pure attention block
                            ggml_tensor* attn_q_check = get_tensor(string_format(TN_MNV5_ATTN_Q_W, stage, blk_idx), false);
                            if (attn_q_check) {
                                found_block = true;
                                block.attn_q_w = attn_q_check;
                                block.attn_k_w = get_tensor(string_format(TN_MNV5_ATTN_K_W, stage, blk_idx), false);
                                block.attn_v_w = get_tensor(string_format(TN_MNV5_ATTN_V_W, stage, blk_idx), false);
                                block.attn_o_w = get_tensor(string_format(TN_MNV5_ATTN_O_W, stage, blk_idx), false);
                                block.attn_k_dw_w   = get_tensor(string_format(TN_MNV5_ATTN_K_DW, stage, blk_idx), false);
                                block.attn_k_norm_w = get_tensor(string_format(TN_MNV5_ATTN_K_NORM, stage, blk_idx), false);
                                block.attn_v_dw_w   = get_tensor(string_format(TN_MNV5_ATTN_V_DW, stage, blk_idx), false);
                                block.attn_v_norm_w = get_tensor(string_format(TN_MNV5_ATTN_V_NORM, stage, blk_idx), false);
                                block.attn_norm_w   = get_tensor(string_format(TN_MNV5_ATTN_NORM, stage, blk_idx), false);
                                // Note: Attention blocks also have layer_scale, load it if not already loaded by UIR check
                                if (!block.layer_scale_w) {
                                    block.layer_scale_w = get_tensor(string_format(TN_MNV5_BLK_LAYER_SCALE, stage, blk_idx), false);
                                }
                            }

                            if (found_block) {
                                model.mobilenet_blocks.push_back(block);
                                blocks_found_in_stage++;
                            } else {
                                // End of blocks for this stage
                                break;
                            }
                        }

                        // Track where this stage ends in the flat vector
                        if (blocks_found_in_stage > 0) {
                            model.mobilenet_stage_ends.push_back(model.mobilenet_blocks.size() - 1);
                            LOG_INF("%s: Stage %d ended at global block index %zu\n", __func__, stage, model.mobilenet_blocks.size() - 1);
                        }
                    }
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.mm_soft_emb_norm_w = get_tensor(TN_MM_SOFT_EMB_N);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    model.mm_fc_w = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                } break;
            case PROJECTOR_TYPE_LFM2:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B, false);
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_KIMIVL:
            case PROJECTOR_TYPE_PADDLEOCR:
            case PROJECTOR_TYPE_KIMIK25:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // [IMG_BREAK] token embedding
                    model.token_embd_img_break = get_tensor(TN_TOK_IMG_BREAK);
                    // for mistral small 3.1
                    model.mm_input_norm_w   = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"), false);
                } break;
            case PROJECTOR_TYPE_LIGHTONOCR:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    model.mm_input_norm_w   = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"), false);
                } break;
            case PROJECTOR_TYPE_DOTS_OCR:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);
                    // post_trunk_norm: applied after all ViT blocks, before the merger
                    model.post_ln_w = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_mid_w = get_tensor(string_format(TN_MM_NORM_MID, "weight"));
                } break;
            case PROJECTOR_TYPE_MERALION:
                {
                    // Whisper encoder conv layers
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    // MERaLiON adaptor: 4 linear layers + ln_pre
                    // linear_0 = frame compression (19200->6400) + SiLU
                    // linear_1 = gate_proj (6400->6400) for GLU
                    // linear_2 = pool_proj (6400->6400) for GLU
                    // linear_3 = out_proj  (6400->3584)
                    model.mm_0_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "bias"));
                    // ln_speech (LayerNorm before adaptor)
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_pre_b = get_tensor(string_format(TN_MM_NORM_PRE, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN2A:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_fc_w = get_tensor(string_format(TN_MM_AUDIO_FC, "weight"));
                    model.mm_fc_b = get_tensor(string_format(TN_MM_AUDIO_FC, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN3A:
                {
                    model.conv2d_1_w = get_tensor(string_format(TN_CONV2D, 1, "weight"));
                    model.conv2d_1_b = get_tensor(string_format(TN_CONV2D, 1, "bias"));
                    model.conv2d_2_w = get_tensor(string_format(TN_CONV2D, 2, "weight"));
                    model.conv2d_2_b = get_tensor(string_format(TN_CONV2D, 2, "bias"));
                    model.conv2d_3_w = get_tensor(string_format(TN_CONV2D, 3, "weight"));
                    model.conv2d_3_b = get_tensor(string_format(TN_CONV2D, 3, "bias"));
                    model.conv_out_w = get_tensor(string_format(TN_CONV_OUT, "weight")); // no bias
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                } break;
            case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                } break;
            case PROJECTOR_TYPE_GLMA:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_pre_b = get_tensor(string_format(TN_MM_NORM_PRE, "bias"));
                    model.mm_boi = get_tensor(string_format(TN_TOK_BOI));
                    model.mm_eoi = get_tensor(string_format(TN_TOK_EOI));
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    model.mm_model_proj    = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_COGVLM:
                {
                    model.mm_model_proj     = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_post_fc_norm_w = get_tensor(string_format(TN_MM_POST_FC_NORM, "weight"));
                    model.mm_post_fc_norm_b = get_tensor(string_format(TN_MM_POST_FC_NORM, "bias"));
                    model.mm_h_to_4h_w      = get_tensor(string_format(TN_MM_H_TO_4H,      "weight"));
                    model.mm_gate_w         = get_tensor(string_format(TN_MM_GATE,         "weight"));
                    model.mm_4h_to_h_w      = get_tensor(string_format(TN_MM_4H_TO_H,      "weight"));
                    model.mm_boi            = get_tensor(TN_TOK_BOI);
                    model.mm_eoi            = get_tensor(TN_TOK_EOI);
                } break;
            case PROJECTOR_TYPE_HUNYUANOCR:
            case PROJECTOR_TYPE_HUNYUANVL:
                {
                    // proj.0 -> mm.0 (conv1), proj.2 -> mm.2 (conv2), mlp -> mm.model.fc (linear)
                    model.mm_0_w            = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b            = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w            = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b            = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    model.mm_model_proj     = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_model_proj_b   = get_tensor(string_format(TN_MM_PROJECTOR, "bias"));
                    model.mm_pre_norm_w     = get_tensor(string_format(TN_MM_PRE_NORM, "weight"));
                    model.mm_post_norm_w    = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                    model.mm_img_begin      = get_tensor(TN_TOK_IMG_BEGIN);
                    model.mm_img_end        = get_tensor(TN_TOK_IMG_END);
                    model.image_newline     = get_tensor(TN_IMAGE_NEWLINE);
                    model.view_seperator    = get_tensor(TN_IMAGE_SEPERATOR, false);
                } break;
            case PROJECTOR_TYPE_JANUS_PRO:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                } break;
            case PROJECTOR_TYPE_PHI4:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_DEEPSEEKOCR:
                {
                    model.pos_embed          = get_tensor(string_format(TN_SAM_POS_EMBD,   "weight"));
                    model.patch_embed_proj_w = get_tensor(string_format(TN_SAM_PATCH_EMBD, "weight"));
                    model.patch_embed_proj_b = get_tensor(string_format(TN_SAM_PATCH_EMBD, "bias"));
                    model.sam_layers.resize(model.n_sam_layers);
                    for (int il = 0; il < model.n_sam_layers; ++il) {
                        auto & layer    = model.sam_layers[il];
                        layer.qkv_w     = get_tensor(string_format(TN_SAM_ATTN_QKV, il, "weight"));
                        layer.qkv_b     = get_tensor(string_format(TN_SAM_ATTN_QKV, il, "bias"));
                        layer.o_w       = get_tensor(string_format(TN_SAM_ATTN_OUT, il, "weight"));
                        layer.o_b       = get_tensor(string_format(TN_SAM_ATTN_OUT, il, "bias"));
                        layer.ln_1_w    = get_tensor(string_format(TN_SAM_PRE_NORM, il, "weight"));
                        layer.ln_1_b    = get_tensor(string_format(TN_SAM_PRE_NORM, il, "bias"));
                        layer.ln_2_w    = get_tensor(string_format(TN_SAM_POST_NORM, il, "weight"));
                        layer.ln_2_b    = get_tensor(string_format(TN_SAM_POST_NORM, il, "bias"));
                        layer.rel_pos_h = get_tensor(string_format(TN_SAM_ATTN_POS_H, il, "weight"));
                        layer.rel_pos_w = get_tensor(string_format(TN_SAM_ATTN_POS_W, il, "weight"));
                        layer.ff_up_w   = get_tensor(string_format(TN_SAM_FFN_UP, il, "weight"));
                        layer.ff_up_b   = get_tensor(string_format(TN_SAM_FFN_UP, il, "bias"));
                        layer.ff_down_w = get_tensor(string_format(TN_SAM_FFN_DOWN, il, "weight"));
                        layer.ff_down_b = get_tensor(string_format(TN_SAM_FFN_DOWN, il, "bias"));
                    }
                    model.neck_0_w       = get_tensor(string_format(TN_SAM_NECK, 0, "weight"));
                    model.neck_1_b       = get_tensor(string_format(TN_SAM_NECK, 1, "bias"));
                    model.neck_1_w       = get_tensor(string_format(TN_SAM_NECK, 1, "weight"));
                    model.neck_2_w       = get_tensor(string_format(TN_SAM_NECK, 2, "weight"));
                    model.neck_3_b       = get_tensor(string_format(TN_SAM_NECK, 3, "bias"));
                    model.neck_3_w       = get_tensor(string_format(TN_SAM_NECK, 3, "weight"));
                    model.net_2          = get_tensor(string_format(TN_SAM_NET, 2, "weight"));
                    model.net_3          = get_tensor(string_format(TN_SAM_NET, 3, "weight"));
                    model.image_newline  = get_tensor(TN_IMAGE_NEWLINE);
                    model.view_seperator = get_tensor(TN_IMAGE_SEPERATOR);
                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_fc_b        = get_tensor(string_format(TN_MM_PROJECTOR, "bias"));
                 } break;
            case PROJECTOR_TYPE_GEMMA4A:
                {
                    for (int i = 0; i < 2; i++) {
                        model.sscp_conv_w[i] = get_tensor(string_format(TN_A_CONV1D, i, "weight"));
                        model.sscp_conv_b[i] = get_tensor(string_format(TN_A_CONV1D, i, "bias"), false);
                        model.sscp_norm_w[i] = get_tensor(string_format(TN_A_CONV1D_NORM, i, "weight"), false);
                    }
                    model.sscp_inp_proj_w = get_tensor(string_format(TN_A_INP_PROJ, "weight"));
                    model.sscp_inp_proj_b = get_tensor(string_format(TN_A_INP_PROJ, "bias"), false);
                    model.audio_out_proj_w = get_tensor(string_format(TN_A_OUT_PROJ, "weight"), false);
                    model.audio_out_proj_b = get_tensor(string_format(TN_A_OUT_PROJ, "bias"), false);
                    // audio multimodal embedder (mm.a.* namespace, not mm.*)
                    model.mm_soft_emb_norm_w = get_tensor(string_format(TN_A_MM_SOFT_EMB_N, "weight"), false);
                    model.mm_input_proj_w    = get_tensor(string_format(TN_A_MM_INP_PROJ, "weight"), false);

                    // Per-layer tensors NOT loaded by the generic loop above
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];

                        // Gemma4 audio conformer-specific tensors
                        layer.ff_norm_w        = get_tensor(string_format(TN_FFN_NORM, prefix, il, "weight"));
                        layer.attn_pre_norm_w  = get_tensor(string_format(TN_A_ATTN_PRE_NORM, prefix, il, "weight"), false);
                        layer.per_dim_scale_w  = get_tensor(string_format(TN_A_PER_DIM_SCALE, prefix, il, "weight"), false);
                        layer.per_dim_k_scale_w = get_tensor(string_format(TN_A_PER_DIM_K_SCALE, prefix, il, "weight"), false);
                        layer.attn_k_rel_w     = get_tensor(string_format(TN_A_ATTN_K_REL, prefix, il, "weight"), false);

                        // Convolution module
                        // Note: conv_norm / norm_conv are swapped in GGUF due to
                        // upstream tensor_mapping.py, so we load them in reverse order
                        layer.norm_conv_w  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "weight"), false);
                        layer.norm_conv_b  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "bias"), false);
                        layer.conv_pw1_w   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "weight"));
                        layer.conv_pw1_b   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "bias"), false);
                        layer.conv_dw_w    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "weight"));
                        layer.conv_dw_b    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "bias"), false);
                        layer.conv_norm_w  = get_tensor(string_format(TN_NORM_CONV, prefix, il, "weight"), false);
                        layer.conv_norm_b  = get_tensor(string_format(TN_NORM_CONV, prefix, il, "bias"), false);
                        layer.conv_pw2_w   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "weight"));
                        layer.conv_pw2_b   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "bias"), false);

                        // FFN2 (second half-step)
                        layer.ff_norm_1_w      = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "weight"));
                        layer.ff_up_1_w        = get_tensor(string_format(TN_FFN_UP_1, prefix, il, "weight"));
                        layer.ff_up_1_b        = get_tensor(string_format(TN_FFN_UP_1, prefix, il, "bias"), false);
                        layer.ff_down_1_w      = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "weight"));
                        layer.ff_down_1_b      = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "bias"), false);
                        layer.ff_post_norm_1_w = get_tensor(string_format(TN_A_FFN_POST_NORM_1, prefix, il, "weight"), false);
                    }

                    // Load clamp info for ClippableLinear AFTER all tensors are loaded
                    for (auto * tensor : tensors_to_load) {
                        std::string name = tensor->name;
                        if (string_ends_with(name, ".weight")) {
                            std::string name_inp_max = name;
                            std::string name_inp_min = name;
                            std::string name_out_max = name;
                            std::string name_out_min = name;
                            string_replace_all(name_inp_max, ".weight", ".input_max");
                            string_replace_all(name_inp_min, ".weight", ".input_min");
                            string_replace_all(name_out_max, ".weight", ".output_max");
                            string_replace_all(name_out_min, ".weight", ".output_min");
                            model.clamp_info_map[name] = {
                                get_scalar(name_inp_max, FLT_MAX),
                                get_scalar(name_inp_min, -FLT_MAX),
                                get_scalar(name_out_max, FLT_MAX),
                                get_scalar(name_out_min, -FLT_MAX)
                            };
                        }
                    }
                } break;
            case PROJECTOR_TYPE_LFM2A:
                {
                    for (int i : {0, 2, 3, 5, 6}) {
                        model.pre_encode_conv_X_w[i] = get_tensor(string_format(TN_CONV1D, i, "weight"));
                        model.pre_encode_conv_X_b[i] = get_tensor(string_format(TN_CONV1D, i, "bias"));
                    }
                    model.pre_encode_out_w    = get_tensor(string_format(TN_PRE_ENCODE_OUT, "weight"));
                    model.pre_encode_out_b    = get_tensor(string_format(TN_PRE_ENCODE_OUT, "bias"));

                    model.mm_0_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "bias"));

                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];

                        layer.ff_norm_w   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "weight"));
                        layer.ff_norm_b   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "bias"));
                        layer.ff_norm_1_w = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "weight"));
                        layer.ff_norm_1_b = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "bias"));
                        layer.ff_up_1_w   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "weight"));
                        layer.ff_up_1_b   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "bias"));
                        layer.ff_down_1_w = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "weight"));
                        layer.ff_down_1_b = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "bias"));

                        layer.pos_bias_u = get_tensor(string_format(TN_POS_BIAS_U, prefix, il));
                        layer.pos_bias_v = get_tensor(string_format(TN_POS_BIAS_V, prefix, il));

                        layer.norm_conv_w = get_tensor(string_format(TN_NORM_CONV, prefix, il, "weight"));
                        layer.norm_conv_b = get_tensor(string_format(TN_NORM_CONV, prefix, il, "bias"));

                        layer.linear_pos_w = get_tensor(string_format(TN_LINEAR_POS, prefix, il, "weight"));

                        layer.conv_norm_w  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "weight"));
                        layer.conv_norm_b  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "bias"));
                        layer.conv_dw_w    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "weight"));
                        layer.conv_dw_b    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "bias"));
                        layer.conv_pw1_w   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "weight"));
                        layer.conv_pw1_b   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "bias"));
                        layer.conv_pw2_w   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "weight"));
                        layer.conv_pw2_b   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "bias"));
                    }
                } break;
            default:
                GGML_ASSERT(false && "unknown projector type");
        }

        // load data
        {
            std::vector<uint8_t> read_buf;

            // alloc memory and offload data
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx_clip.backend);
            ctx_clip.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_clip.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(ctx_clip.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            for (auto & t : tensors_to_load) {
                ggml_tensor * cur = ggml_get_tensor(ctx_clip.ctx_data.get(), t->name);
                GGML_ASSERT(cur && "tensor not found in ctx_data");
                auto it_off = tensor_offset.find(t->name);
                GGML_ASSERT(it_off != tensor_offset.end() && "no offset for tensor");
                const size_t offset = it_off->second;
                fin.seekg(offset, std::ios::beg);
                if (!fin) {
                    throw std::runtime_error(string_format("%s: failed to seek for tensor %s\n", __func__, t->name));
                }
                size_t num_bytes = ggml_nbytes(cur);
                if (ggml_backend_buft_is_host(buft)) {
                    // for the CPU and Metal backend, we can read directly into the tensor
                    fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
                } else {
                    // read into a temporary buffer first, then copy to device memory
                    read_buf.resize(num_bytes);
                    fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                    ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
                }
            }
            fin.close();

            LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(), fname.c_str());
        }

    }

    struct support_info_op {
        ggml_tensor * op;

        // true if the op runs on the accelerated ctx_clip.backend
        bool is_accel = true;
    };

    struct support_info_graph {
        // whether the clip_ctx.backend supports flash attention
        bool fattn = true;
        ggml_tensor * fattn_op = nullptr; // for debugging

        std::vector<support_info_op> ops;
    };

    static void warmup(clip_ctx & ctx_clip) {
        // create a fake batch
        const auto & hparams = ctx_clip.model.hparams;
        clip_image_f32_batch batch;
        clip_image_f32_ptr img(clip_image_f32_init());
        if (ctx_clip.model.modality == CLIP_MODALITY_VISION) {
            img->nx = hparams.warmup_image_size;
            img->ny = hparams.warmup_image_size;
            LOG_INF("%s: warmup with image size = %d x %d\n", __func__, img->nx, img->ny);
        } else {
            img->nx = hparams.warmup_audio_size;
            img->ny = hparams.n_mel_bins;
            LOG_INF("%s: warmup with audio size = %d\n", __func__, img->nx);
        }
        batch.entries.push_back(std::move(img));
        warmup(ctx_clip, batch);
    }

    static void warmup(clip_ctx & ctx_clip, const clip_image_f32_batch & batch) {
        support_info_graph info;

        if (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_AUTO) {
            // try to enable flash attention to see if it's supported
            ctx_clip.flash_attn_type = CLIP_FLASH_ATTN_TYPE_ENABLED;
            info = alloc_compute_meta(ctx_clip, batch);
            if (!info.fattn && info.fattn_op) {
                auto op = info.fattn_op;
                LOG_WRN("%s: *****************************************************************\n", __func__);
                LOG_WRN("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", __func__, ggml_backend_name(ctx_clip.backend));
                LOG_WRN("%s: op params: \n", __func__);
                static auto print_shape = [](const char * fn, const char * name, ggml_tensor * t) {
                    LOG_WRN("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", fn,
                            name, ggml_type_name(t->type),
                            t->ne[0], t->ne[1], t->ne[2], t->ne[3],
                            t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
                };
                print_shape(__func__, " dst", op);
                print_shape(__func__, "src0", op->src[0]);
                print_shape(__func__, "src1", op->src[1]);
                print_shape(__func__, "src2", op->src[2]);
                LOG_WRN("%s: please report this on github as an issue\n", __func__);
                LOG_WRN("%s: *****************************************************************\n", __func__);
                ctx_clip.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
                alloc_compute_meta(ctx_clip, batch);
            }
        } else {
            info = alloc_compute_meta(ctx_clip, batch);
            if (!info.fattn && ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) {
                LOG_WRN("%s: flash attention is not supported by the current backend; falling back to CPU (performance will be degraded)\n", __func__);
            }
        }

        ctx_clip.is_allocated = true; // mark buffers as allocated

        LOG_INF("%s: flash attention is %s\n", __func__,
            (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");

        // print ops that are not supported by the GPU backend (if there is one)
        if (ctx_clip.backend && ctx_clip.backend != ctx_clip.backend_cpu) {
            std::vector<support_info_op> unsupported_ops;
            for (const auto & op : info.ops) {
                if (!op.is_accel) {
                    unsupported_ops.push_back(op);
                }
            }
            if (!unsupported_ops.empty()) {
                LOG_WRN("%s: *****************************************************************\n", __func__);
                LOG_WRN("%s: WARNING: the CLIP graph uses unsupported operators by the backend\n", __func__);
                LOG_WRN("%s:          the performance will be suboptimal                      \n", __func__);
                LOG_WRN("%s:          list of unsupported ops (backend=%s):\n", __func__, ggml_backend_name(ctx_clip.backend));
                for (const auto & op : unsupported_ops) {
                    LOG_WRN("%s: %16s: type = %s, ne = [%d %d %d %d]\n", __func__,
                            ggml_op_name(op.op->op),
                            ggml_type_name(op.op->type),
                            op.op->ne[0], op.op->ne[1], op.op->ne[2], op.op->ne[3]);
                }
                LOG_WRN("%s: flash attention is %s\n", __func__,
                    (ctx_clip.flash_attn_type == CLIP_FLASH_ATTN_TYPE_ENABLED) ? "enabled" : "disabled");
                LOG_WRN("%s: please report this on github as an issue\n", __func__);
                LOG_WRN("%s: ref: https://github.com/ggml-org/llama.cpp/pull/16837#issuecomment-3461676118\n", __func__);
                LOG_WRN("%s: *****************************************************************\n", __func__);
            }
        }
    }

    static support_info_graph alloc_compute_meta(clip_ctx & ctx_clip, const clip_image_f32_batch & batch) {
        ctx_clip.buf_compute_meta.resize(ctx_clip.max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());

        ggml_cgraph * gf = clip_image_build_graph(&ctx_clip, batch);
        ggml_backend_sched_reserve(ctx_clip.sched.get(), gf);

        for (size_t i = 0; i < ctx_clip.backend_ptrs.size(); ++i) {
            ggml_backend_t backend = ctx_clip.backend_ptrs[i];
            ggml_backend_buffer_type_t buft = ctx_clip.backend_buft[i];
            size_t size = ggml_backend_sched_get_buffer_size(ctx_clip.sched.get(), backend);
            if (size > 1) {
                LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
            }
        }

        const int n_splits = ggml_backend_sched_get_n_splits(ctx_clip.sched.get());
        const int n_nodes  = ggml_graph_n_nodes(gf);

        LOG_INF("%s: graph splits = %d, nodes = %d\n", __func__,  n_splits, n_nodes);

        support_info_graph res {
            /*.fattn    = */ true,
            /*.fattn_op = */ nullptr,
            /*.ops      = */ {},
        };

        // check op support
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            res.ops.push_back({node, true});
            if (!ggml_backend_supports_op(ctx_clip.backend, node)) {
                res.ops.back().is_accel = false;
                if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                    res.fattn    = false;
                    res.fattn_op = node;
                }
            }
        }

        return res;
    }

    void get_bool(const std::string & key, bool & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_i32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_i32(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_u32(ctx_gguf.get(), i);
    }

    void get_f32(const std::string & key, float & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_string(const std::string & key, std::string & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }

    void get_arr_int(const std::string & key, std::vector<int> & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        int n = gguf_get_arr_n(ctx_gguf.get(), i);
        output.resize(n);
        const int32_t * values = (const int32_t *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (int i = 0; i < n; ++i) {
            output[i] = values[i];
        }
    }

    static void set_llava_uhd_res_candidates(clip_model & model, const int max_patches_per_side) {
        auto & hparams = model.hparams;
        for (int x = 1; x <= max_patches_per_side; x++) {
            for (int y = 1; y <= max_patches_per_side; y++) {
                if (x == 1 && y == 1) {
                    continue; // skip the first point
                }
                hparams.image_res_candidates.push_back(clip_image_size{
                    x*hparams.image_size,
                    y*hparams.image_size,
                });
            }
        }
    }

    static void set_internvl_dhr_res_candidates(clip_model & model) {
        auto & hparams = model.hparams;
        int min_num = hparams.preproc_min_tiles;
        int max_num = hparams.preproc_max_tiles;
        if (min_num < 1) {
           return; // avoid  divide by 0
        }
        for (int a = min_num; a <= max_num; ++a) {
            int b_lo = (min_num + a - 1) / a;
            int b_hi = max_num / a;
            b_lo = std::max(b_lo, min_num);
            b_hi = std::min(b_hi, max_num);
            for (int b = b_lo; b <= b_hi; ++b) {
                hparams.image_res_candidates.push_back(clip_image_size {
                    a*hparams.image_size,
                    b*hparams.image_size,
                });
            }
        }
    }
};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params) {
    clip_ctx * ctx_vision = nullptr;
    clip_ctx * ctx_audio = nullptr;

    try {
        clip_model_loader loader(fname);
        bool skip_audio = false;

        if (loader.has_vision) {
            ctx_vision = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_vision->model, CLIP_MODALITY_VISION);
            loader.load_tensors(*ctx_vision);
            if (ctx_params.warmup) {
                loader.warmup(*ctx_vision);
            }

            // TODO: we don't support audio for Gemma 3N, but GGUF contains audio tensors
            // we can remove this check when we implement audio support for Gemma 3N
            skip_audio = ctx_vision->model.proj_type == PROJECTOR_TYPE_GEMMA3NV;
        }

        if (loader.has_audio && !skip_audio) {
            ctx_audio = new clip_ctx(ctx_params);
            loader.load_hparams(ctx_audio->model, CLIP_MODALITY_AUDIO);
            loader.load_tensors(*ctx_audio);
            if (ctx_params.warmup) {
                loader.warmup(*ctx_audio);
            }
        }

    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to load model '%s': %s\n", __func__, fname, e.what());

        delete ctx_vision;
        delete ctx_audio;

        return {nullptr, nullptr};
    }

    return {ctx_vision, ctx_audio};
}

struct clip_image_size * clip_image_size_init() {
    struct clip_image_size * load_image_size = new struct clip_image_size();
    load_image_size->width = 448;
    load_image_size->height = 448;
    return load_image_size;
}

struct clip_image_u8 * clip_image_u8_init() {
    return new clip_image_u8();
}

struct clip_image_f32 * clip_image_f32_init() {
    return new clip_image_f32();
}

struct clip_image_f32_batch * clip_image_f32_batch_init() {
    return new clip_image_f32_batch();
}

unsigned char * clip_image_u8_get_data(struct clip_image_u8 * img, uint32_t * nx, uint32_t * ny) {
    if (nx) *nx = img->nx;
    if (ny) *ny = img->ny;
    return img->buf.data();
}

void clip_image_size_free(struct clip_image_size * load_image_size) {
    if (load_image_size == nullptr) {
        return;
    }
    delete load_image_size;
}
void clip_image_u8_free(struct clip_image_u8  * img) { delete img; }
void clip_image_f32_free(struct clip_image_f32 * img) { delete img; }
void clip_image_u8_batch_free(struct clip_image_u8_batch * batch) { delete batch; }
void clip_image_f32_batch_free(struct clip_image_f32_batch * batch) { delete batch; }

size_t clip_image_f32_batch_n_images(const struct clip_image_f32_batch * batch) {
    return batch->entries.size();
}

size_t clip_image_f32_batch_nx(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->nx;
}

size_t clip_image_f32_batch_ny(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return 0;
    }
    return batch->entries[idx]->ny;
}

clip_image_f32 * clip_image_f32_get_img(const struct clip_image_f32_batch * batch, int idx) {
    if (idx < 0 || idx >= (int)batch->entries.size()) {
        LOG_ERR("%s: invalid index %d\n", __func__, idx);
        return nullptr;
    }
    return batch->entries[idx].get();
}

void clip_build_img_from_pixels(const unsigned char * rgb_pixels, int nx, int ny, clip_image_u8 * img) {
    img->nx = nx;
    img->ny = ny;
    img->buf.resize(3 * nx * ny);
    memcpy(img->buf.data(), rgb_pixels, img->buf.size());
}

ggml_tensor * clip_get_newline_tensor(const struct clip_ctx * ctx) {
    return ctx->model.image_newline;
}

void clip_free(clip_ctx * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

// deprecated
size_t clip_embd_nbytes(const struct clip_ctx * ctx) {
    const int32_t nx = ctx->model.hparams.image_size;
    const int32_t ny = ctx->model.hparams.image_size;
    return clip_embd_nbytes_by_img(ctx, nx, ny);
}

size_t clip_embd_nbytes_by_img(const struct clip_ctx * ctx, int img_w, int img_h) {
    clip_image_f32 img;
    img.nx = img_w;
    img.ny = img_h;
    return clip_n_output_tokens(ctx, &img) * clip_n_mmproj_embd(ctx) * sizeof(float);
}

int32_t clip_get_image_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.image_size;
}

int32_t clip_get_patch_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.patch_size;
}

int32_t clip_get_hidden_size(const struct clip_ctx * ctx) {
    return ctx->model.hparams.n_embd;
}

const char * clip_patch_merge_type(const struct clip_ctx * ctx) {
    return ctx->model.hparams.mm_patch_merge_type == PATCH_MERGE_SPATIAL_UNPAD ? "spatial_unpad" : "flat";
}

int clip_n_output_tokens_x(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    const int n_total = clip_n_output_tokens(ctx, img);
    const auto & proj = ctx->proj_type();
    switch (proj) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
        case PROJECTOR_TYPE_YOUTUVL:
            return (img->nx / params.patch_size) / 2;
        case PROJECTOR_TYPE_STEP3VL:
            return img->nx / (params.patch_size * params.n_merge);
        default:
            break;
    }
    return n_total;
}

int clip_n_output_tokens_y(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;
    const auto & proj = ctx->proj_type();
    switch (proj) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
        case PROJECTOR_TYPE_YOUTUVL:
            return (img->ny / params.patch_size) / 2;
        case PROJECTOR_TYPE_STEP3VL:
            return img->ny / (params.patch_size * params.n_merge);
        default:
            break;
    }
    return 1;
}

int clip_n_output_tokens(const struct clip_ctx * ctx, struct clip_image_f32 * img) {
    const auto & params = ctx->model.hparams;

    // for models with fixed size image, the input image is already pre-processed and resized to square
    int patch_size = params.patch_size;
    int n_patches = (img->nx / patch_size) * (img->ny / patch_size);

    projector_type proj = ctx->proj_type();

    switch (proj) {
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_PHI4:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_YASA2:
            {
                n_patches = 64; // adaptive average pooling to 8x8 tokens
            } break;
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                n_patches /= 4;
                if (ctx->model.mm_boi) {
                    n_patches += 2; // for BOI and EOI token embeddings
                }
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                // Use actual config value if available, otherwise fall back to hardcoded values
                if (params.minicpmv_query_num > 0) {
                    n_patches = params.minicpmv_query_num;
                } else {
                    // Fallback to hardcoded values for legacy models
                    if (params.minicpmv_version == 2) {
                        n_patches = 96;
                    } else if (params.minicpmv_version == 3) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 4) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 5) {
                        // MiniCPM-V 4.0
                        n_patches = 64;
                    } else if (params.minicpmv_version == 6) {
                        // MiniCPM-V 4.5
                        n_patches = 64;
                    } else if (params.minicpmv_version == 100045) {
                        // MiniCPM-o 4.5
                        n_patches = 64;
                    } else {
                        GGML_ABORT("Unknown minicpmv version");
                    }
                }
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_YOUTUVL:
            {
                // dynamic size (2 conv, so double patch size)
                int x_patch = img->nx / (params.patch_size * 2);
                int y_patch = img->ny / (params.patch_size * 2);
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_STEP3VL:
            {
                int x_patch = img->nx / (params.patch_size * params.n_merge);
                int y_patch = img->ny / (params.patch_size * params.n_merge);
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA4V:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
        case PROJECTOR_TYPE_LLAMA4:
            {
                // both X and Y are downscaled by the scale factor
                int scale_factor = ctx->model.hparams.n_merge;
                n_patches /= (scale_factor * scale_factor);
            } break;
        case PROJECTOR_TYPE_GEMMA3NV:
            {
                // MobileNetV5 MSFA adapter always outputs fixed 16x16 resolution
                // regardless of input size (see architecture description)
                n_patches = ctx->model.hparams.image_size / ctx->model.hparams.patch_size;
            } break;
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_KIMIK25:
            {
                // dynamic size
                int out_patch_size = params.patch_size * ctx->model.hparams.n_merge;
                int x_patch = CLIP_ALIGN(img->nx, out_patch_size) / out_patch_size;
                int y_patch = CLIP_ALIGN(img->ny, out_patch_size) / out_patch_size;
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_DOTS_OCR:
            {
                // dynamic size
                int n_merge = ctx->model.hparams.n_merge;
                int stride = n_merge * n_merge;
                n_patches = CLIP_ALIGN(n_patches, stride) / stride;
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                // dynamic size
                int n_merge = ctx->model.hparams.n_merge;
                int n_patches_x = img->nx / patch_size / (n_merge > 0 ? n_merge : 1);
                int n_patches_y = img->ny / patch_size / (n_merge > 0 ? n_merge : 1);
                if (ctx->model.token_embd_img_break) {
                    n_patches = n_patches_y * n_patches_x + n_patches_y - 1; // + one [IMG_BREAK] per row, except the last row
                } else {
                    n_patches = n_patches_y * n_patches_x;
                }
            } break;
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            {
                n_patches = img->nx;

                const int proj_stack_factor = ctx->model.hparams.proj_stack_factor;
                if (ctx->model.audio_has_stack_frames()) {
                    GGML_ASSERT(proj_stack_factor > 0);
                    const int n_len = CLIP_ALIGN(n_patches, proj_stack_factor);
                    n_patches = n_len / proj_stack_factor;
                }

                // whisper downscales input token by half after conv1d
                n_patches /= 2;

                if (ctx->model.audio_has_avgpool()) {
                    // divide by 2 because of nn.AvgPool1d(2, stride=2)
                    n_patches /= 2;
                }
            } break;
        case PROJECTOR_TYPE_QWEN3A:
            {
                // 3x stride-2 conv2d: each step is floor((n-1)/2)+1
                int n = img->nx;
                n = (n - 1) / 2 + 1;
                n = (n - 1) / 2 + 1;
                n = (n - 1) / 2 + 1;
                n_patches = n;
            } break;
        case PROJECTOR_TYPE_GLMA:
            {
                n_patches = img->nx;
                // whisper downscales input token by half after conv1d
                n_patches /= 2;
                // reshape by merge_factor
                n_patches /= ctx->model.hparams.proj_stack_factor;
                // for BOI and EOI token embeddings
                n_patches += 2;
            } break;
        case PROJECTOR_TYPE_COGVLM:
            {
                n_patches += 2; // for BOI and EOI token embeddings
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR:
        {
            // SAM encoder applies two stride-2 convolutions (net_2 and net_3)
            // which reduces spatial dimensions by 4x in each direction (16x total)
            // E.g., 64x64 -> 16x16 patches
            n_patches /= 16;

            // build_global_local_features adds image newlines and view separator
            // Formula: h*(w+1) + 1 where h = w = sqrt(n_patches)
            int h = static_cast<int>(std::sqrt(static_cast<float>(n_patches)));
            n_patches = h * (h + 1) + 1;
        } break;
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
            {
                int merge = ctx->model.hparams.n_merge;
                int ow = (img->nx / patch_size) / merge;
                int oh = (img->ny / patch_size) / merge;
                n_patches = (ow + 1) * oh + 2;
            } break;
        case PROJECTOR_TYPE_LFM2A:
            {
                n_patches = ((((img->nx + 1) / 2) + 1) / 2 + 1) / 2;
            } break;
        case PROJECTOR_TYPE_GEMMA4A:
            {
                // Two Conv2D stride-2: O = floor((I + 2p - k) / s) + 1, p=1, k=3, s=2
                // O = floor((I - 1) / 2) + 1
                int n = img->nx;
                for (int i = 0; i < 2; i++) {
                    n = (n - 1) / 2 + 1;
                }
                n_patches = n;
            } break;
        default:
            GGML_ABORT("unsupported projector type");
    }

    return n_patches;
}

bool clip_image_encode(struct clip_ctx * ctx, const int n_threads, clip_image_f32 * img, float * vec) {
    clip_image_f32_batch imgs;
    clip_image_f32_ptr img_copy(clip_image_f32_init());
    *img_copy = *img;
    imgs.entries.push_back(std::move(img_copy));

    return clip_image_batch_encode(ctx, n_threads, &imgs, vec);
}

bool clip_image_batch_encode(clip_ctx * ctx, const int n_threads, const clip_image_f32_batch * imgs_c_ptr, float * vec) {
    const clip_image_f32_batch & imgs = *imgs_c_ptr;
    int batch_size = imgs.entries.size();

    // TODO @ngxson : implement batch size > 1 as a loop
    //                we don't need true batching support because the cgraph will gonna be big anyway
    if (batch_size != 1) {
        return false; // only support batch size of 1
    }

    // if buffers are not allocated, we need to do a warmup run to allocate them
    if (!ctx->is_allocated) {
        clip_model_loader::warmup(*ctx, *imgs_c_ptr);
    }

    // build the inference graph
    ggml_backend_sched_reset(ctx->sched.get());
    ggml_cgraph * gf = clip_image_build_graph(ctx, imgs);
    ggml_backend_sched_alloc_graph(ctx->sched.get(), gf);

    // set inputs
    const auto & model   = ctx->model;
    const auto & hparams = model.hparams;

    const int image_size_width  = imgs.entries[0]->nx;
    const int image_size_height = imgs.entries[0]->ny;

    const int patch_size    = hparams.patch_size;
    const int num_patches   = ((image_size_width / patch_size) * (image_size_height / patch_size));
    const int n_pos = num_patches + (model.class_embedding ? 1 : 0);
    const int pos_w = image_size_width  / patch_size;
    const int pos_h = image_size_height / patch_size;


    auto get_inp_tensor = [&gf](const char * name) {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (inp == nullptr) {
            GGML_ABORT("Failed to get tensor %s", name);
        }
        if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            GGML_ABORT("Tensor %s is not an input tensor", name);
        }
        return inp;
    };

    auto set_input_f32 = [&get_inp_tensor](const char * name, std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    auto set_input_i32 = [&get_inp_tensor](const char * name, std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    // set input pixel values
    if (!imgs.is_audio) {
        size_t nelem = 0;
        for (const auto & img : imgs.entries) {
            nelem += img->nx * img->ny * 3;
        }
        std::vector<float> inp_raw(nelem);

        // layout of data (note: the channel dim is unrolled to better visualize the layout):
        //
        // ┌──W──┐
        // │     H │  channel = R
        // ├─────┤ │
        // │     H │  channel = G
        // ├─────┤ │
        // │     H │  channel = B
        // └─────┘ │
        //   ──────┘ x B

        for (size_t i = 0; i < imgs.entries.size(); i++) {
            const int nx = imgs.entries[i]->nx;
            const int ny = imgs.entries[i]->ny;
            const int n = nx * ny;

            for (int b = 0; b < batch_size; b++) {
                float * batch_entry = inp_raw.data() + b * (3*n);
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        size_t base_src = 3*(y * nx + x); // idx of the first channel
                        size_t base_dst =    y * nx + x;  // idx of the first channel
                        batch_entry[      base_dst] = imgs.entries[b]->buf[base_src    ];
                        batch_entry[1*n + base_dst] = imgs.entries[b]->buf[base_src + 1];
                        batch_entry[2*n + base_dst] = imgs.entries[b]->buf[base_src + 2];
                    }
                }
            }
        }
        set_input_f32("inp_raw", inp_raw);

    } else {
        // audio input
        GGML_ASSERT(imgs.entries.size() == 1);
        const auto & mel_inp = imgs.entries[0];
        const int n_step = mel_inp->nx;
        const int n_mel  = mel_inp->ny;
        std::vector<float> inp_raw(n_step * n_mel);
        std::memcpy(inp_raw.data(), mel_inp->buf.data(), n_step * n_mel * sizeof(float));
        set_input_f32("inp_raw", inp_raw);
    }

    // set input per projector
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_MINICPMV:
            {
                // inspired from siglip:
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit
                //    -> https://huggingface.co/HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit/blob/d66538faeba44480d0bfaa42145eef26f9423199/modeling_siglip.py#L316
                std::vector<int32_t> positions(pos_h * pos_w);
                int bucket_coords_h[1024];
                int bucket_coords_w[1024];
                for (int i = 0; i < pos_h; i++){
                    bucket_coords_h[i] = std::floor(70.0*i/pos_h);
                }
                for (int i = 0; i < pos_w; i++){
                    bucket_coords_w[i] = std::floor(70.0*i/pos_w);
                }
                for (int i = 0, id = 0; i < pos_h; i++){
                    for (int j = 0; j < pos_w; j++){
                        positions[id++] = bucket_coords_h[i]*70 + bucket_coords_w[j];
                    }
                }
                set_input_i32("positions", positions);

                // inputs for resampler projector
                // set the 2D positions (using float for sinusoidal embedding)
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<float> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = static_cast<float>(i / n_patches_per_col);
                }
                set_input_f32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = static_cast<float>(i % n_patches_per_col);
                }
                set_input_f32("pos_w", pos_data);
                // base frequency omega
                const float base_freq   = 10000.0f;
                const int   n_embd_proj = clip_n_mmproj_embd(ctx);
                std::vector<float> omega(n_embd_proj / 4);
                for (int i = 0; i < n_embd_proj / 4; ++i) {
                    omega[i] = 1.0f / std::pow(base_freq, static_cast<float>(i) / (n_embd_proj / 4));
                }
                set_input_f32("omega", omega);
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
            {
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size;
                const int ph = image_size_height / patch_size;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                for (int y = 0; y < ph; y += merge_ratio) {
                    for (int x = 0; x < pw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                positions[                  ptr] = y + dy;
                                positions[    num_patches + ptr] = x + dx;
                                positions[2 * num_patches + ptr] = y + dy;
                                positions[3 * num_patches + ptr] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_STEP3VL:
            {
                std::vector<int32_t> pos_data(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / pos_w;
                }
                set_input_i32("pos_h", pos_data);
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % pos_w;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_PADDLEOCR:
            {
                const int merge_ratio = hparams.n_merge;
                const int pw = image_size_width  / patch_size;
                const int ph = image_size_height / patch_size;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;
                // NOTE: same as Qwen-VL, but x and y are swapped
                for (int y = 0; y < ph; y += merge_ratio) {
                    for (int dy = 0; dy < 2; dy++) {
                        for (int x = 0; x < pw; x += merge_ratio) {
                            for (int dx = 0; dx < 2; dx++) {
                                positions[                  ptr] = y + dy;
                                positions[    num_patches + ptr] = x + dx;
                                positions[2 * num_patches + ptr] = y + dy;
                                positions[3 * num_patches + ptr] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_DOTS_OCR:
            {
                const int pw = image_size_width / patch_size;
                const int ph = image_size_height / patch_size;
                const int n_pos = ph * pw;
                std::vector<int> positions(n_pos * 4);
                int ptr = 0;

                // flat layout: [h, w, h, w] for each patch
                // patches are in raster order (matching conv2d output)
                for (int y = 0; y < ph; y++) {
                    for (int x = 0; x < pw; x++) {
                        positions[          ptr] = y;
                        positions[  n_pos + ptr] = x;
                        positions[2*n_pos + ptr] = y;
                        positions[3*n_pos + ptr] = x;
                        ptr++;
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_YOUTUVL:
            {
                // pw * ph = number of tokens output by ViT after apply patch merger
                // ipw * ipw = number of vision token been processed inside ViT
                const bool use_window_attn = ctx->model.proj_type == PROJECTOR_TYPE_QWEN25VL ? hparams.n_wa_pattern > 0 : !hparams.wa_layer_indexes.empty();
                const int merge_ratio = 2;
                const int pw  = image_size_width  / patch_size / merge_ratio;
                const int ph  = image_size_height / patch_size / merge_ratio;
                const int ipw = image_size_width  / patch_size;
                const int iph = image_size_height / patch_size;

                std::vector<int> idx    (ph * pw);
                std::vector<int> inv_idx(ph * pw);

                if (use_window_attn) {
                    const int attn_window_size = hparams.attn_window_size > 0 ? hparams.attn_window_size : 112;
                    const int grid_window = attn_window_size / patch_size / merge_ratio;
                    int dst = 0;
                    // [num_vision_tokens, num_vision_tokens] attention mask tensor
                    std::vector<float> mask(pow(ipw * iph, 2), std::numeric_limits<float>::lowest());
                    int mask_row = 0;

                    for (int y = 0; y < ph; y += grid_window) {
                        for (int x = 0; x < pw; x += grid_window) {
                            const int win_h = std::min(grid_window, ph - y);
                            const int win_w = std::min(grid_window, pw - x);
                            const int dst_0 = dst;
                            // group all tokens belong to the same window togather (to a continue range)
                            for (int dy = 0; dy < win_h; dy++) {
                                for (int dx = 0; dx < win_w; dx++) {
                                    const int src = (y + dy) * pw + (x + dx);
                                    GGML_ASSERT(src < (int)idx.size());
                                    GGML_ASSERT(dst < (int)inv_idx.size());
                                    idx    [src] = dst;
                                    inv_idx[dst] = src;
                                    dst++;
                                }
                            }

                            for (int r=0; r < win_h * win_w * merge_ratio * merge_ratio; r++) {
                                int row_offset = mask_row * (ipw * iph);
                                std::fill(
                                    mask.begin() + row_offset + (dst_0 * merge_ratio * merge_ratio),
                                    mask.begin() + row_offset + (dst   * merge_ratio * merge_ratio),
                                    0.0);
                                mask_row++;
                            }
                        }
                    }

                    set_input_i32("window_idx",     idx);
                    set_input_i32("inv_window_idx", inv_idx);
                    set_input_f32("window_mask",    mask);
                } else {
                    for (int i = 0; i < ph * pw; i++) {
                        idx[i] = i;
                    }
                }

                const int mpow = merge_ratio * merge_ratio;
                std::vector<int> positions(n_pos * 4);

                int ptr = 0;
                for (int y = 0; y < iph; y += merge_ratio) {
                    for (int x = 0; x < ipw; x += merge_ratio) {
                        for (int dy = 0; dy < 2; dy++) {
                            for (int dx = 0; dx < 2; dx++) {
                                auto remap = idx[ptr / mpow];
                                remap = (remap * mpow) + (ptr % mpow);

                                positions[                  remap] = y + dy;
                                positions[    num_patches + remap] = x + dx;
                                positions[2 * num_patches + remap] = y + dy;
                                positions[3 * num_patches + remap] = x + dx;
                                ptr++;
                            }
                        }
                    }
                }

                set_input_i32("positions", positions);
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_KIMIK25:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(n_pos);
                // dimension H
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i / n_patches_per_col;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < n_pos; i++) {
                    pos_data[i] = i % n_patches_per_col;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_GLM_EDGE:
        {
            // llava and other models
            std::vector<int32_t> positions(n_pos);
            for (int i = 0; i < n_pos; i++) {
                positions[i] = i;
            }
            set_input_i32("positions", positions);
        } break;
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
            {
                // llava and other models
                std::vector<int32_t> positions(n_pos);
                for (int i = 0; i < n_pos; i++) {
                    positions[i] = i;
                }
                set_input_i32("positions", positions);

                // The patches vector is used to get rows to index into the embeds with;
                // we should skip dim 0 only if we have CLS to avoid going out of bounds
                // when retrieving the rows.
                int patch_offset = model.class_embedding ? 1 : 0;
                std::vector<int32_t> patches(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    patches[i] = i + patch_offset;
                }
                set_input_i32("patches", patches);
            } break;
        case PROJECTOR_TYPE_GEMMA4V:
            {
                // set (col, row) patch positions for learned positional embedding
                const int n_cols = image_size_width  / patch_size;
                std::vector<int> pos_x(num_patches), pos_y(num_patches);
                for (int i = 0; i < num_patches; i++) {
                    pos_x[i] = i % n_cols;
                    pos_y[i] = i / n_cols;
                }
                set_input_i32("pos_x", pos_x);
                set_input_i32("pos_y", pos_y);
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR:
            {
                GGML_ASSERT(pos_w == pos_h);

                const int window = hparams.attn_window_size;
                const int pos = pos_w;
                std::vector<int32_t> rel_pos_indices_local(window * window);
                std::vector<int32_t> rel_pos_indices_global(pos * pos);

                for (int q = 0; q < window; q++) {
                    for (int k = 0; k < window; k++) {
                        rel_pos_indices_local[q * window + k] = q - k + window - 1;
                    }
                }

                for (int q = 0; q < pos; q++) {
                    for (int k = 0; k < pos; k++) {
                        rel_pos_indices_global[q * pos + k] = q - k + pos - 1;
                    }
                }

                set_input_i32("rel_pos_indices_local", rel_pos_indices_local);
                set_input_i32("rel_pos_indices_global", rel_pos_indices_global);
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA3NV:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_QWEN3A:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_PHI4:
        case PROJECTOR_TYPE_COGVLM:
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_YASA2:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_HUNYUANVL:
            {
                // Compute the HunyuanVL 2D position embedding on CPU (with the
                // custom sf=(target+0.1)/n_grid bilinear sampling that the
                // reference implementation uses) and upload it to the graph
                // input declared in clip_graph_hunyuanocr::build().
                GGML_ASSERT(model.position_embeddings != nullptr);
                ggml_tensor * src_t   = model.position_embeddings;
                const int64_t n_embd  = src_t->ne[0];
                const int64_t n_pos   = src_t->ne[1];            // = n_grid * n_grid
                const int     n_grid  = (int)std::lround(std::sqrt((double)n_pos));
                GGML_ASSERT((int64_t)n_grid * n_grid == n_pos);
                const int     out_w   = pos_w;                    // pw
                const int     out_h   = pos_h;                    // ph

                // Pull weight to host.
                std::vector<float> src(n_embd * n_pos);
                ggml_backend_tensor_get(src_t, src.data(), 0, ggml_nbytes(src_t));

                // Output layout matches ggml_new_tensor_2d(F32, n_embd, out_h*out_w):
                //   ne[0] = n_embd (fastest), ne[1] = out_h*out_w
                //   dst[(y*out_w + x) * n_embd + c]
                std::vector<float> dst((size_t)n_embd * out_h * out_w);

                const float sx = (float)(out_w + 0.1f) / (float)n_grid;
                const float sy = (float)(out_h + 0.1f) / (float)n_grid;

                for (int y = 0; y < out_h; ++y) {
                    // Match ggml_compute_forward_upscale_f32 pixel-center
                    // convention (align_corners=False): src_y = (y+0.5)/sy - 0.5.
                    const float fy = ((float)y + 0.5f) / sy - 0.5f;
                    int y0 = (int)std::floor(fy);
                    int y1 = y0 + 1;
                    y0 = std::clamp(y0, 0, n_grid - 1);
                    y1 = std::clamp(y1, 0, n_grid - 1);
                    float wy1 = std::clamp(fy - (float)y0, 0.0f, 1.0f);
                    const float wy0 = 1.0f - wy1;
                    for (int x = 0; x < out_w; ++x) {
                        const float fx = ((float)x + 0.5f) / sx - 0.5f;
                        int x0 = (int)std::floor(fx);
                        int x1 = x0 + 1;
                        x0 = std::clamp(x0, 0, n_grid - 1);
                        x1 = std::clamp(x1, 0, n_grid - 1);
                        float wx1 = std::clamp(fx - (float)x0, 0.0f, 1.0f);
                        const float wx0 = 1.0f - wx1;

                        const float w00 = wy0 * wx0;
                        const float w01 = wy0 * wx1;
                        const float w10 = wy1 * wx0;
                        const float w11 = wy1 * wx1;

                        const float * s00 = &src[((size_t)y0 * n_grid + x0) * n_embd];
                        const float * s01 = &src[((size_t)y0 * n_grid + x1) * n_embd];
                        const float * s10 = &src[((size_t)y1 * n_grid + x0) * n_embd];
                        const float * s11 = &src[((size_t)y1 * n_grid + x1) * n_embd];
                        float * d         = &dst[((size_t)y * out_w + x) * n_embd];
                        for (int c = 0; c < n_embd; ++c) {
                            d[c] = w00 * s00[c] + w01 * s01[c] + w10 * s10[c] + w11 * s11[c];
                        }
                    }
                }

                set_input_f32("hunyuanvl_pos_embd", dst);
            } break;
        case PROJECTOR_TYPE_LLAMA4:
            {
                // set the 2D positions
                int n_patches_per_col = image_size_width / patch_size;
                std::vector<int> pos_data(num_patches + 1, 0); // +1 for the [CLS] token
                // last pos is always kept 0, it's for CLS
                // dimension H
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i / n_patches_per_col) + 1;
                }
                set_input_i32("pos_h", pos_data);
                // dimension W
                for (int i = 0; i < num_patches; i++) {
                    pos_data[i] = (i % n_patches_per_col) + 1;
                }
                set_input_i32("pos_w", pos_data);
            } break;
        case PROJECTOR_TYPE_GEMMA4A:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                const auto & img0 = imgs.entries.front();
                // Compute n_pos matching SSCP output: two stride-2 convs
                int n_pos = img0->nx;
                for (int i = 0; i < 2; i++) { n_pos = (n_pos - 1) / 2 + 1; }

                // Chunked local attention: blocked causal mask and RPE
                const int chunk_size   = 12;
                const int max_past     = 12;
                const int context_size = chunk_size + max_past;
                const int num_blocks   = (n_pos + chunk_size - 1) / chunk_size;

                // Blocked causal attention mask: [context_size, chunk_size, num_blocks]
                {
                    std::vector<float> mask(context_size * chunk_size * num_blocks, -1e9f);
                    for (int b = 0; b < num_blocks; b++) {
                        for (int q = 0; q < chunk_size; q++) {
                            int gq = b * chunk_size + q;
                            for (int k = 0; k < context_size; k++) {
                                int gk = b * chunk_size - max_past + k;
                                if (gq < n_pos && gk >= 0 && gk < n_pos && gk <= gq && (gq - gk) < max_past) {
                                    mask[k + q * context_size + b * context_size * chunk_size] = 0.0f;
                                }
                            }
                        }
                    }
                    set_input_f32("kq_mask", mask);
                }

                // Sinusoidal RPE: 13 positions [12, 11, ..., 0]
                {
                    const int n_embd = ctx->model.hparams.n_embd;
                    const int num_timescales = n_embd / 2;
                    const float log_timescale_increment = logf(10000.0f) / std::max(num_timescales - 1, 1);
                    const int rpe_len = max_past + 1;
                    std::vector<float> pos_emb(n_embd * rpe_len, 0.0f);
                    for (int p = 0; p < rpe_len; p++) {
                        float position = (float)(max_past - p);
                        for (int i = 0; i < num_timescales; i++) {
                            float inv_ts = expf(-(float)i * log_timescale_increment);
                            float scaled = position * inv_ts;
                            pos_emb[p * n_embd + i]                 = sinf(scaled);
                            pos_emb[p * n_embd + i + num_timescales] = cosf(scaled);
                        }
                    }
                    set_input_f32("pos_emb", pos_emb);
                }
            } break;
        case PROJECTOR_TYPE_LFM2A:
            {
                GGML_ASSERT(imgs.entries.size() == 1);
                const auto n_frames = clip_n_output_tokens(ctx, imgs.entries.front().get());

                auto d_model = 512;
                auto seq_len = n_frames * 2 - 1;
                std::vector<float> pos_emb(d_model*seq_len);
                std::vector<double> inv_freq(d_model / 2);
                for (size_t i = 0; i < inv_freq.size(); ++i) {
                    inv_freq[i] = std::exp(-(std::log(10000.0) / (float)d_model) * (2.0f * (float)(i)));
                }
                for (int64_t pos = 0; pos < seq_len; ++pos) {
                    for (size_t i = 0; i < inv_freq.size(); ++i) {
                        const float ang = (n_frames - pos - 1) * inv_freq[i];
                        pos_emb[pos*d_model + 2*i + 0] = sinf(ang);  // even
                        pos_emb[pos*d_model + 2*i + 1] = cosf(ang);  // odd
                    }
                }
                set_input_f32("pos_emb", pos_emb);
            } break;
        default:
            GGML_ABORT("Unknown projector type");
    }

    // ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(ctx->backend_cpu, n_threads);
        }
    }

    auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    // the last node is the embedding tensor
    ggml_tensor * embeddings = ggml_graph_node(gf, -1);

    // sanity check (only support batch size of 1 for now)
    const int n_tokens_out = embeddings->ne[1];
    const int expected_n_tokens_out = clip_n_output_tokens(ctx, imgs.entries[0].get());
    if (n_tokens_out != expected_n_tokens_out) {
        LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, expected_n_tokens_out, n_tokens_out);
        GGML_ABORT("Invalid number of output tokens");
    }

    // copy the embeddings to the location passed by the user
    if (vec != nullptr) {
        ggml_backend_tensor_get(embeddings, vec, 0, ggml_nbytes(embeddings));
    }

    // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set
    if (ctx->debug_output_embeddings) {
        const int64_t n_embd = embeddings->ne[0];
        const int64_t n_tokens = embeddings->ne[1];
        std::vector<float> emb_data(n_embd * n_tokens);
        ggml_backend_tensor_get(embeddings, emb_data.data(), 0, ggml_nbytes(embeddings));

        LOG_INF("\n=== MTMD_DEBUG_EMBEDDINGS ===\n");
        LOG_INF("Shape: [%lld, %lld]\n", (long long)n_embd, (long long)n_tokens);

        // Print first few values of first token
        LOG_INF("Token 0 (first 16 values): ");
        for (int i = 0; i < std::min((int64_t)16, n_embd); i++) {
            LOG_INF("%.6f ", emb_data[i]);
        }
        LOG_INF("\n");

        // Print last few values of first token
        if (n_embd > 16) {
            LOG_INF("Token 0 (last 16 values):  ");
            for (int64_t i = n_embd - 16; i < n_embd; i++) {
                LOG_INF("%.6f ", emb_data[i]);
            }
            LOG_INF("\n");
        }

        // Compute and print statistics
        float sum = 0.0f, sum_sq = 0.0f, min_val = emb_data[0], max_val = emb_data[0];
        for (size_t i = 0; i < emb_data.size(); i++) {
            sum += emb_data[i];
            sum_sq += emb_data[i] * emb_data[i];
            min_val = std::min(min_val, emb_data[i]);
            max_val = std::max(max_val, emb_data[i]);
        }
        float mean = sum / emb_data.size();
        float variance = (sum_sq / emb_data.size()) - (mean * mean);
        LOG_INF("Stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, sum=%.6f\n",
                mean, sqrtf(variance), min_val, max_val, sum);
        LOG_INF("=== END MTMD_DEBUG_EMBEDDINGS ===\n\n");
    }

    return true;
}

int clip_n_mmproj_embd(const struct clip_ctx * ctx) {
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_LDP:
            return ctx->model.mm_model_block_1_block_2_1_b->ne[0];
        case PROJECTOR_TYPE_LDPV2:
            return ctx->model.mm_model_peg_0_b->ne[0];
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_PHI4:
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
        case PROJECTOR_TYPE_DOTS_OCR:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_MLP_NORM:
            return ctx->model.mm_3_b->ne[0];
        case PROJECTOR_TYPE_MINICPMV:
            return ctx->model.mm_model_proj->ne[0];
        case PROJECTOR_TYPE_GLM_EDGE:
            return ctx->model.mm_model_mlp_3_w->ne[1];
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_YOUTUVL:
            return ctx->model.mm_1_b->ne[0];
        case PROJECTOR_TYPE_QWEN3VL:
            // main path + deepstack paths
            return ctx->model.mm_1_b->ne[0] * (1 + ctx->model.n_deepstack_layers);
        case PROJECTOR_TYPE_STEP3VL:
            return ctx->model.mm_model_proj->ne[1];
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA3NV:
            return ctx->model.mm_input_proj_w->ne[0];
        case PROJECTOR_TYPE_GEMMA4V:
            return ctx->model.mm_input_proj_w->ne[1];
        case PROJECTOR_TYPE_IDEFICS3:
            return ctx->model.mm_fc_w->ne[1];
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_MERALION:
            return ctx->model.mm_3_w->ne[1]; // out_proj output dim
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
            return ctx->model.mm_3_w->ne[1];
        case PROJECTOR_TYPE_LLAMA4:
            return ctx->model.mm_model_proj->ne[1];
        case PROJECTOR_TYPE_QWEN2A:
            return ctx->model.mm_fc_w->ne[1];
        case PROJECTOR_TYPE_QWEN3A:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_KIMIK25:
        case PROJECTOR_TYPE_YASA2:
            return ctx->model.mm_2_w->ne[1];
        case PROJECTOR_TYPE_HUNYUANOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
            return ctx->model.mm_model_proj->ne[1];
        case PROJECTOR_TYPE_COGVLM:
            return ctx->model.mm_4h_to_h_w->ne[1];
        case PROJECTOR_TYPE_DEEPSEEKOCR:
            return ctx->model.mm_fc_w->ne[1];
        case PROJECTOR_TYPE_LFM2A:
            return ctx->model.position_embeddings->ne[0];
        case PROJECTOR_TYPE_GEMMA4A:
            return ctx->model.hparams.projection_dim;
        case PROJECTOR_TYPE_GLM4V:
            return ctx->model.mm_ffn_down_w->ne[1];
        default:
            GGML_ABORT("Unknown projector type");
    }
}

int clip_is_minicpmv(const struct clip_ctx * ctx) {
    // TODO: remove this function
    if (ctx->proj_type() == PROJECTOR_TYPE_MINICPMV) {
        return ctx->model.hparams.minicpmv_version;
    }
    return 0;
}

bool clip_is_glm(const struct clip_ctx * ctx) {
    // TODO: remove this function
    return ctx->proj_type() == PROJECTOR_TYPE_GLM_EDGE;
}

bool clip_is_llava(const struct clip_ctx * ctx) {
    return ctx->model.hparams.has_llava_projector;
}

bool clip_has_vision_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_VISION;
}

bool clip_has_audio_encoder(const struct clip_ctx * ctx) {
    return ctx->model.modality == CLIP_MODALITY_AUDIO;
}

bool clip_has_whisper_encoder(const struct clip_ctx * ctx) {
    switch (ctx->proj_type()) {
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_QWEN3A:
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            return true;
        default:
            return false;
    }
}

bool clip_encode_float_image (struct clip_ctx * ctx, int n_threads, float * img, int h, int w, float * vec) {
    clip_image_f32 clip_img;
    clip_img.buf.resize(h * w * 3);
    for (int i = 0; i < h*w*3; i++)
    {
        clip_img.buf[i] = img[i];
    }
    clip_img.nx = w;
    clip_img.ny = h;
    clip_image_encode(ctx, n_threads, &clip_img, vec);
    return true;
}

//
// API used internally with mtmd
//

projector_type clip_get_projector_type(const struct clip_ctx * ctx) {
    return ctx->proj_type();
}

void clip_image_f32_batch_add_mel(struct clip_image_f32_batch * batch, int n_mel, int n_frames, float * mel) {
    clip_image_f32 * audio = new clip_image_f32;
    audio->nx = n_frames;
    audio->ny = n_mel;
    audio->buf.resize(n_frames * n_mel);
    std::memcpy(audio->buf.data(), mel, n_frames * n_mel * sizeof(float));

    batch->entries.push_back(clip_image_f32_ptr(audio));
    batch->is_audio = true;
}

const clip_hparams * clip_get_hparams(const struct clip_ctx * ctx) {
    return &ctx->model.hparams;
}

//
// API for debugging
//

void clip_set_debug_output_embeddings(clip_ctx * ctx, bool enable) {
    ctx->debug_output_embeddings = enable;
}


// ---- mtmd-audio implementation ----

// some of the code here is copied from whisper.cpp

constexpr bool DEBUG = false;

void mtmd_audio_cache::fill_sin_cos_table(uint32_t n) {
    sin_vals.resize(n);
    cos_vals.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        double theta = (2 * M_PI * i) / n;
        sin_vals[i]  = sinf(theta);
        cos_vals[i]  = cosf(theta);
    }
}

void mtmd_audio_cache::fill_hann_window(uint32_t length, bool periodic) {
    hann_window.resize(length);
    int offset = periodic ? 0 : -1;
    for (uint32_t i = 0; i < length; i++) {
        hann_window[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
    }
}

void mtmd_audio_cache::fill_mel_filterbank_matrix(int   n_mel,
                                                  int   n_fft,
                                                  int   sample_rate,
                                                  float fmin,
                                                  float fmax,
                                                  bool  slaney_area_norm,
                                                  float scale,
                                                  bool  use_htk) {
    GGML_ASSERT(n_mel > 0 && n_fft > 1);
    if (fmax <= 0.0f) {
        fmax = 0.5f * sample_rate;
    }

    std::function<double(double)> hz_to_mel;
    std::function<double(double)> mel_to_hz;

    if (use_htk) {
        hz_to_mel = [](const double f_hz) -> double {
            return 2595.0 * log10(1.0 + f_hz / 700.0);
        };
        mel_to_hz = [](const double m) -> double {
            return 700.0 * (pow(10.0, m / 2595.0) - 1.0);
        };
    } else {
        // Slaney scale (matches librosa default)
        const double min_log_hz  = 1000.0;
        const double lin_slope   = 3 / 200.;
        const double min_log_mel = min_log_hz * lin_slope;
        const double log_step    = log(6.4) / 27.0;
        hz_to_mel = [min_log_hz, lin_slope, log_step, min_log_mel](const double f_hz) -> double {
            return (f_hz < min_log_hz) ? f_hz * lin_slope : min_log_mel + log(f_hz / min_log_hz) / log_step;
        };
        mel_to_hz = [min_log_hz, lin_slope, log_step, min_log_mel](const double m) -> double {
            return (m < min_log_mel) ? m / lin_slope : min_log_hz * exp((m - min_log_mel) * log_step);
        };
    }

    // infer N_fft from n_fft_bins
    const double bin_hz_step = double(sample_rate) / double(n_fft);

    // mel grid: n_mel + 2 edges
    const double        m_lo = hz_to_mel(fmin);
    const double        m_hi = hz_to_mel(fmax);
    std::vector<double> mel_pts(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) {
        mel_pts[i] = m_lo + (m_hi - m_lo) * (double(i) / (n_mel + 1));
    }

    // convert to Hz
    std::vector<double> hz_pts(n_mel + 2);
    for (int i = 0; i < n_mel + 2; ++i) {
        hz_pts[i] = mel_to_hz(mel_pts[i]);
    }

    const int n_fft_bins = n_fft / 2 + 1;

    // filterbank
    std::vector<float> out(n_mel * n_fft_bins, 0);
    for (int m = 0; m < n_mel; ++m) {
        const double f_left   = hz_pts[m];
        const double f_center = hz_pts[m + 1];
        const double f_right  = hz_pts[m + 2];

        const double denom_l = std::max(1e-30, f_center - f_left);
        const double denom_r = std::max(1e-30, f_right - f_center);
        const double enorm   = slaney_area_norm ? (2.0 / std::max(1e-30, f_right - f_left)) : 1.0;

        for (int k = 0; k < n_fft_bins; ++k) {
            const double f = k * bin_hz_step;
            double       w = 0.0;
            if (f >= f_left && f <= f_center) {
                w = (f - f_left) / denom_l;
            } else if (f > f_center && f <= f_right) {
                w = (f_right - f) / denom_r;
            }
            out[size_t(m) * size_t(n_fft_bins) + size_t(k)] = float(w * enorm * scale);
        }
    }

    filters.n_mel = n_mel;
    filters.n_fft = n_fft;
    filters.data  = std::move(out);

    if (DEBUG) {  // debug
        for (size_t i = 0; i < filters.data.size(); ++i) {
            if (filters.data[i] != 0.0f) {
                printf("filters[%zu] = %f\n", i, filters.data[i] * 1000.0f);
            }
        }
    }
}

// Unified DFT implementation for both forward and inverse transforms
// Template parameters:
//   Inverse: false = DFT with exp(-2πi·k·n/N), no scaling
//            true  = IDFT with exp(+2πi·k·n/N), scales by 1/N
//   RealInput: true = input is real-valued (stride 1), avoids imaginary computations
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void dft_impl(const mtmd_audio_cache & cache, const float * in, int N, float * out) {
    const int n_sin_cos_vals = cache.sin_vals.size();
    const int sin_cos_step   = n_sin_cos_vals / N;

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    const float     scale = Inverse ? (1.0f / N) : 1.0f;

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            int   idx     = (k * n * sin_cos_step) % n_sin_cos_vals;
            float cos_val = cache.cos_vals[idx];
            float sin_val = cache.sin_vals[idx];

            if constexpr (RealInput) {
                // Real input: in_im = 0, simplifies to:
                // re += in_re * cos_val
                // im += sign * in_re * sin_val
                float in_re = in[n];
                re += in_re * cos_val;
                im += sign * in_re * sin_val;
            } else {
                float in_re = in[n * 2 + 0];
                float in_im = in[n * 2 + 1];
                // (a + bi) * (cos + sign*i*sin) = (a*cos - sign*b*sin) + (sign*a*sin + b*cos)i
                re += in_re * cos_val - sign * in_im * sin_val;
                im += sign * in_re * sin_val + in_im * cos_val;
            }
        }

        out[k * 2 + 0] = re * scale;
        out[k * 2 + 1] = im * scale;
    }
}

// Cooley-Tukey FFT/IFFT unified implementation
// Template parameters:
//   Inverse: false = FFT with exp(-2πi·k/N), no scaling
//            true  = IFFT with exp(+2πi·k/N), scales by 0.5 at each level
//   RealInput: true = input is real-valued (stride 1)
//              false = input is complex-valued (interleaved real/imag, stride 2)
template <bool Inverse, bool RealInput>
static void fft_impl(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    GGML_ASSERT(N > 0);
    const int n_sin_cos_vals = cache.sin_vals.size();

    if (N == 1) {
        out[0] = in[0];
        if constexpr (RealInput) {
            out[1] = 0.0f;
        } else {
            out[1] = in[1];
        }
        return;
    }

    const int half_N = N / 2;
    if (N - half_N * 2 == 1) {
        // Odd N: fall back to DFT
        dft_impl<Inverse, RealInput>(cache, in, N, out);
        return;
    }

    // Split into even and odd
    if constexpr (RealInput) {
        // Real input: stride is 1, copy only real values
        float * even = in + N;
        for (int i = 0; i < half_N; ++i) {
            even[i] = in[2 * i];
        }
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, true>(cache, even, half_N, even_fft);

        float * odd = even;
        for (int i = 0; i < half_N; ++i) {
            odd[i] = in[2 * i + 1];
        }
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, true>(cache, odd, half_N, odd_fft);
    } else {
        // Complex input: stride is 2, copy complex pairs
        float * even = in + N * 2;
        for (int i = 0; i < half_N; ++i) {
            even[i * 2 + 0] = in[2 * i * 2 + 0];
            even[i * 2 + 1] = in[2 * i * 2 + 1];
        }
        float * even_fft = out + 2 * N;
        fft_impl<Inverse, false>(cache, even, half_N, even_fft);

        float * odd = even;
        for (int i = 0; i < half_N; ++i) {
            odd[i * 2 + 0] = in[(2 * i + 1) * 2 + 0];
            odd[i * 2 + 1] = in[(2 * i + 1) * 2 + 1];
        }
        float * odd_fft = even_fft + N;
        fft_impl<Inverse, false>(cache, odd, half_N, odd_fft);
    }

    float * even_fft = out + 2 * N;
    float * odd_fft  = even_fft + N;

    const int sin_cos_step = n_sin_cos_vals / N;

    constexpr float sign  = Inverse ? 1.0f : -1.0f;
    constexpr float scale = Inverse ? 0.5f : 1.0f;

    for (int k = 0; k < half_N; k++) {
        int   idx = k * sin_cos_step;  // t = 2*M_PI*k/N
        float re  = cache.cos_vals[idx];
        float im  = sign * cache.sin_vals[idx];

        float re_odd = odd_fft[2 * k + 0];
        float im_odd = odd_fft[2 * k + 1];

        out[2 * k + 0] = scale * (even_fft[2 * k + 0] + re * re_odd - im * im_odd);
        out[2 * k + 1] = scale * (even_fft[2 * k + 1] + re * im_odd + im * re_odd);

        out[2 * (k + half_N) + 0] = scale * (even_fft[2 * k + 0] - re * re_odd + im * im_odd);
        out[2 * (k + half_N) + 1] = scale * (even_fft[2 * k + 1] - re * im_odd - im * re_odd);
    }
}

// Forward FFT for real input (used by mel spectrogram)
static void fft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    fft_impl<false, true>(cache, in, N, out);
}

// Inverse FFT for complex input
static void ifft(const mtmd_audio_cache & cache, float * in, int N, float * out) {
    fft_impl<true, false>(cache, in, N, out);
}

struct filter_params {
    int32_t n_mel;
    int32_t n_fft_bins;
    int32_t hann_window_size;
    int32_t hop_length;
    int32_t sample_rate;
    bool    no_padding      = false;
    bool    center_padding  = false;
    float   preemph         = 0.f;
    bool    use_natural_log = false;
    bool    norm_per_feature = false;
    bool    use_magnitude   = false;  // |X| instead of |X|^2
    float   mel_floor       = 5.960464477539063e-08f;
};

static void log_mel_spectrogram_worker_thread(int                        ith,
                                              const float *              hann,
                                              const std::vector<float> & samples,
                                              int                        n_samples,
                                              int                        frame_size,
                                              int                        frame_step,
                                              int                        n_threads,
                                              const filter_params &      params,
                                              const mtmd_audio_cache &   cache,
                                              mtmd_audio_mel &           out) {
    std::vector<float> fft_in(frame_size * 2, 0.0);
    std::vector<float> fft_out(frame_size * 2 * 2 * 2);

    int n_fft_bins = params.n_fft_bins;
    int i = ith;

    const auto & filters = cache.filters;

    // make sure n_fft == 1 + (WHISPER_N_FFT / 2), bin_0 to bin_nyquist
    GGML_ASSERT(n_fft_bins == 1 + (frame_size / 2));
    GGML_ASSERT(cache.sin_vals.size() == cache.cos_vals.size());
    // calculate FFT only when fft_in are not all zero
    for (; i < std::min(n_samples / frame_step + 1, out.n_len); i += n_threads) {
        const int offset = i * frame_step;

        // apply Hann window (~10% faster)
        for (int j = 0; j < std::min(frame_size, n_samples - offset); j++) {
            fft_in[j] = hann[j] * samples[offset + j];
        }

        // fill the rest with zeros
        if (n_samples - offset < frame_size) {
            std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);
        }

        // FFT
        fft(cache, fft_in.data(), frame_size, fft_out.data());

        // Calculate modulus^2 (power) or modulus (magnitude)
        for (int j = 0; j < n_fft_bins; j++) {
            float power = (fft_out[2 * j + 0] * fft_out[2 * j + 0] + fft_out[2 * j + 1] * fft_out[2 * j + 1]);
            fft_out[j] = params.use_magnitude ? sqrtf(power) : power;
        }

        // mel spectrogram
        for (int j = 0; j < out.n_mel; j++) {
            double sum = 0.0;
            // unroll loop (suggested by GH user @lunixbochs)
            int k = 0;
            for (k = 0; k < n_fft_bins - 3; k += 4) {
                size_t idx = size_t(j) * size_t(n_fft_bins) + size_t(k);
                sum +=
                        fft_out[k + 0] * filters.data[idx + 0] +
                        fft_out[k + 1] * filters.data[idx + 1] +
                        fft_out[k + 2] * filters.data[idx + 2] +
                        fft_out[k + 3] * filters.data[idx + 3];
            }
            // handle n_fft remainder
            for (; k < n_fft_bins; k++) {
                sum += fft_out[k] * filters.data[j * n_fft_bins + k];
            }
            sum = std::max(sum, (double)params.mel_floor);
            sum = params.use_natural_log
                ? log(sum)
                : log10(sum);
            out.data[j * out.n_len + i] = sum;
        }
    }

    // Otherwise fft_out are all zero
    double sum = params.use_natural_log ? log(1e-10) : log10(1e-10);
    for (; i < out.n_len; i += n_threads) {
        for (int j = 0; j < out.n_mel; j++) {
            out.data[j * out.n_len + i] = sum;
        }
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L110-L157
static bool log_mel_spectrogram(
        const float * samples,
        const int     n_samples_in,
        const int     n_threads,
        const filter_params & params,
        const mtmd_audio_cache & cache,
        mtmd_audio_mel & out) {
    //const int64_t t_start_us = ggml_time_us();

    out.n_len_org = n_samples_in;
    int n_samples = n_samples_in;

    // Hann window
    const float * hann       = cache.hann_window.data();
    const int     frame_size = (params.n_fft_bins - 1) * 2;
    const int     frame_step = params.hop_length;

    // Padding
    std::vector<float> samples_padded;
    if (params.no_padding) {
        // no padding, use samples as-is
        samples_padded = std::vector<float>(samples, samples + n_samples);
        samples = samples_padded.data();
        n_samples = samples_padded.size();
    } else if (params.center_padding) {
        const auto pad_amount = frame_size / 2;
        samples_padded = std::vector<float>(n_samples + 2 * pad_amount, 0);
        std::copy(samples, samples + n_samples, samples_padded.data() + pad_amount);
        samples = samples_padded.data();
        n_samples = samples_padded.size();
    } else {
        // existing padding logic
        int64_t stage_1_pad = params.sample_rate * 30;
        int64_t stage_2_pad = frame_size / 2;
        samples_padded.resize(n_samples + stage_1_pad + stage_2_pad * 2);
        std::copy(samples, samples + n_samples, samples_padded.begin() + stage_2_pad);
        // pad 30 seconds of zeros at the end of audio (480,000 samples) + reflective pad 200 samples at the end of audio
        std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);
        // reflective pad 200 samples at the beginning of audio
        if (n_samples < stage_2_pad + 1) {
            // TODO: Handle short audio differently or return error
            return false;
        }
        std::reverse_copy(samples + 1, samples + 1 + stage_2_pad, samples_padded.begin());
    }

    // preemphasis
    if (params.preemph) {
        const int   pad_amount = frame_size / 2;
        const float preemph = 0.97f;
        float       prev = samples_padded[pad_amount];
        for (int i = pad_amount + 1; i + pad_amount < n_samples; ++i) {
            float cur = samples_padded[i];
            samples_padded[i] = cur - preemph * prev;
            prev = cur;
        }
    }

    // pad hann window if it's smaller than frame_size
    // TODO: probably unnecessary here? (or better doing it in g_cache?)
    std::vector<float> hann_window_padded;
    if (params.hann_window_size < frame_size) {
        hann_window_padded.resize(frame_size);
        const int padding = (frame_size - params.hann_window_size) / 2;
        std::copy(hann, hann + params.hann_window_size, &hann_window_padded[padding]);
        hann = hann_window_padded.data();
    }


    GGML_ASSERT(params.n_fft_bins > 0);
    GGML_ASSERT(params.hop_length > 0);
    out.n_mel = params.n_mel;
    out.n_len = (n_samples - frame_size) / frame_step + 1;
    // TODO: handle these checks better
    if (out.n_mel > 0 && (unsigned long)out.n_len > SIZE_MAX / out.n_mel) {
        LOG_ERR("%s: size overflow\n", __func__);
        return false;
    }
    if (n_samples < frame_size) {
        LOG_ERR("%s: not enough samples after padding\n", __func__);
        return false;
    }
    out.data.resize(out.n_mel * out.n_len);

    {
        std::vector<std::thread> workers(n_threads - 1);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw] =
                std::thread(log_mel_spectrogram_worker_thread, iw + 1, hann, std::cref(samples_padded), n_samples,
                            frame_size, frame_step, n_threads, std::cref(params), std::cref(cache), std::ref(out));
        }

        // main thread
        log_mel_spectrogram_worker_thread(0, hann, samples_padded, n_samples, frame_size, frame_step, n_threads, params,
                                          cache, out);
        for (int iw = 0; iw < n_threads - 1; ++iw) {
            workers[iw].join();
        }
    }

    const int effective_n_len = n_samples_in / frame_step;
    if (params.norm_per_feature) {
        GGML_ASSERT(effective_n_len > 1);
        for (int i = 0; i < out.n_mel; i++) {
            double mean = 0;
            for (int j = 0; j < effective_n_len; ++j) {
                mean += out.data[i * out.n_len + j];
            }
            mean /= effective_n_len;

            double var = 0.0;
            for (int j = 0; j < effective_n_len; ++j) {
                const double value = out.data[i * out.n_len + j] - mean;
                var += value * value;
            }
            var /= effective_n_len - 1;  // unbiased
            const double mstd = std::sqrt(var + 1e-5);

            for (int j = 0; j < effective_n_len; ++j) {
                auto &value = out.data[i * out.n_len + j];
                value        = (value - mean) / mstd;
            }

            // pad the rest with zeros
            for (int j = effective_n_len; j < out.n_len; ++j) {
                out.data[i * out.n_len + j] = 0.0;
            }
        }
    } else if (!params.no_padding) {
        // Whisper-style clamping and normalization (NOT used by Gemma4)
        double mmax = -1e20;
        for (int i = 0; i < out.n_mel*out.n_len; i++) {
            if (out.data[i] > mmax) {
                mmax = out.data[i];
            }
        }

        mmax -= 8.0;

        for (int i = 0; i < out.n_mel*out.n_len; i++) {
            if (out.data[i] < mmax) {
                out.data[i] = mmax;
            }
            out.data[i] = (out.data[i] + 4.0)/4.0;
        }
    }

    // Dump log_mel_spectrogram
    if (DEBUG) {
        std::ofstream outFile("log_mel_spectrogram.json");
        outFile << "[";
        for (uint64_t i = 0; i < out.data.size() - 1; i++) {
            outFile << out.data[i] << ", ";
        }
        outFile << out.data[out.data.size() - 1] << "]";
        outFile.close();
    }

    return true;
}

//
// mtmd_audio_preprocessor_whisper
//

void mtmd_audio_preprocessor_whisper::initialize() {
    cache.fill_sin_cos_table(hparams.audio_n_fft);
    cache.fill_hann_window(hparams.audio_window_len, true);
    cache.fill_mel_filterbank_matrix(hparams.n_mel_bins, hparams.audio_n_fft, hparams.audio_sample_rate);
}

bool mtmd_audio_preprocessor_whisper::preprocess(const float *                 samples,
                                                 size_t                        n_samples,
                                                 std::vector<mtmd_audio_mel> & output) {
    if (n_samples == 0) {
        // empty audio
        return false;
    }

    std::vector<float> smpl;
    // if input is too short, pad with zeros
    // this is to avoid potential issues with stage1/2 padding in log_mel_spectrogram
    // TODO: maybe handle this better
    size_t min_samples = (size_t) hparams.audio_sample_rate * (hparams.audio_chunk_len + 1);  // +1 second margin
    if (n_samples < min_samples) {
        smpl.resize(min_samples, 0.0f);
        std::memcpy(smpl.data(), samples, n_samples * sizeof(float));
        samples   = smpl.data();
        n_samples = smpl.size();
    }

    filter_params params;
    params.n_mel            = hparams.n_mel_bins;
    params.n_fft_bins       = 1 + (hparams.audio_n_fft / 2);
    params.hann_window_size = hparams.audio_window_len;
    params.hop_length       = hparams.audio_hop_len;
    params.sample_rate      = hparams.audio_sample_rate;
    params.center_padding   = false;
    params.preemph          = 0.0f;  // disabled
    params.use_natural_log  = false;
    params.norm_per_feature = false;

    // make sure the cache is initialized
    GGML_ASSERT(!cache.sin_vals.empty());
    GGML_ASSERT(!cache.cos_vals.empty());
    GGML_ASSERT(!cache.filters.data.empty());

    mtmd_audio_mel out_full;
    bool           ok = log_mel_spectrogram(samples, n_samples,
                                            4,  // n_threads
                                            params, cache, out_full);
    if (!ok) {
        return false;
    }

    // because the CLIP graph only accepts 3000 frames each, we need to split the mel
    // we always expect the mel to have 3000 silent frames at the end
    if (DEBUG) {
        printf("output: n_mel = %d, n_len = %d\n", out_full.n_mel, out_full.n_len);
    }
    const size_t frames_per_chunk = 3000;
    GGML_ASSERT((size_t) out_full.n_len > frames_per_chunk);
    for (size_t off = 0; off < (size_t) out_full.n_len; off += frames_per_chunk) {
        int n_len = std::min(frames_per_chunk, (size_t) out_full.n_len - off);
        if ((size_t) n_len < frames_per_chunk) {
            break;  // last incomplete chunk will always be a padded chunk, safe to ignore
        }

        mtmd_audio_mel out_chunk;
        out_chunk.n_len     = n_len;
        out_chunk.n_mel     = out_full.n_mel;
        out_chunk.n_len_org = out_full.n_mel;  // unused
        out_chunk.data.reserve(out_chunk.n_mel * out_chunk.n_len);

        for (int i = 0; i < out_full.n_mel; i++) {
            auto src = out_full.data.begin() + i * out_full.n_len + off;
            out_chunk.data.insert(out_chunk.data.end(), src, src + frames_per_chunk);
        }

        output.push_back(std::move(out_chunk));
    }

    return true;
}

//
// mtmd_audio_preprocessor_conformer
//

void mtmd_audio_preprocessor_conformer::initialize() {
    cache.fill_sin_cos_table(hparams.audio_n_fft);
    cache.fill_hann_window(hparams.audio_window_len, true);
    cache.fill_mel_filterbank_matrix(hparams.n_mel_bins, hparams.audio_n_fft, hparams.audio_sample_rate);
}

bool mtmd_audio_preprocessor_conformer::preprocess(const float *                 samples,
                                                   size_t                        n_samples,
                                                   std::vector<mtmd_audio_mel> & output) {
    // empty audio
    if (n_samples == 0) {
        return false;
    }

    filter_params params;
    params.n_mel            = hparams.n_mel_bins;
    params.n_fft_bins       = 1 + (hparams.audio_n_fft / 2);
    params.hann_window_size = hparams.audio_window_len;
    params.hop_length       = hparams.audio_hop_len;
    params.sample_rate      = hparams.audio_sample_rate;
    params.center_padding   = true;
    params.preemph          = 0.97f;
    params.use_natural_log  = true;
    params.norm_per_feature = true;

    // make sure the cache is initialized
    GGML_ASSERT(!cache.sin_vals.empty());
    GGML_ASSERT(!cache.cos_vals.empty());
    GGML_ASSERT(!cache.filters.data.empty());

    mtmd_audio_mel out_full;
    bool           ok = log_mel_spectrogram(samples, n_samples,
                                            4,  // n_threads
                                            params, cache, out_full);
    if (!ok) {
        return false;
    }

    output.push_back(std::move(out_full));
    return true;
}

//
// mtmd_audio_preprocessor_gemma4a
//

void mtmd_audio_preprocessor_gemma4a::initialize() {
    cache.fill_sin_cos_table(hparams.audio_n_fft);

    // Standard periodic Hann window, zero-padded to FFT size
    cache.hann_window.assign(hparams.audio_n_fft, 0.0f);
    for (uint32_t i = 0; i < (uint32_t)hparams.audio_window_len; i++) {
        cache.hann_window[i] = 0.5f - 0.5f * cosf((2.0f * (float)M_PI * i) / hparams.audio_window_len);
    }

    // HTK mel scale, no Slaney area normalization
    cache.fill_mel_filterbank_matrix(
        hparams.n_mel_bins, hparams.audio_n_fft, hparams.audio_sample_rate,
        0.0f, hparams.audio_sample_rate / 2.0f,
        /*slaney_area_norm=*/ false,
        /*scale=*/ 1.0f,
        /*use_htk=*/ true
    );
}

bool mtmd_audio_preprocessor_gemma4a::preprocess(const float *                 samples,
                                                  size_t                        n_samples,
                                                  std::vector<mtmd_audio_mel> & output) {
    if (n_samples == 0) {
        return false;
    }

    GGML_ASSERT(!cache.sin_vals.empty());
    GGML_ASSERT(!cache.cos_vals.empty());
    GGML_ASSERT(!cache.filters.data.empty());

    filter_params params;
    params.n_mel            = hparams.n_mel_bins;
    params.n_fft_bins       = 1 + (hparams.audio_n_fft / 2);
    params.hann_window_size = hparams.audio_n_fft; // window is zero-padded to FFT size
    params.hop_length       = hparams.audio_hop_len;
    params.sample_rate      = hparams.audio_sample_rate;
    params.no_padding       = true;
    params.center_padding   = false;
    params.preemph          = 0.0f;
    params.use_natural_log  = true;
    params.use_magnitude    = true;
    params.mel_floor        = 0.001f;
    params.norm_per_feature = false;

    // Split into 30-second chunks (model context limit, ~750 tokens each)
    const size_t chunk_samples = 30 * hparams.audio_sample_rate;
    for (size_t off = 0; off < n_samples; off += chunk_samples) {
        const float * chunk_ptr = samples + off;
        size_t chunk_len = std::min(chunk_samples, n_samples - off);

        // Semicausal left-padding + right-padding to match PyTorch frame count
        const int pad_left = hparams.audio_window_len / 2;
        const int fft_size = hparams.audio_n_fft;
        const int hop = hparams.audio_hop_len;
        const int n_with_left = (int)chunk_len + pad_left;
        // PyTorch: unfold(size=frame_length+1, step=hop) on semicausal-padded waveform
        const int pt_frames = (n_with_left - (hparams.audio_window_len + 1)) / hop + 1;
        const int n_padded_needed = (pt_frames - 1) * hop + fft_size;
        const int total_pad = std::max((int)(n_padded_needed - (int)chunk_len), pad_left);
        std::vector<float> padded_samples(total_pad + chunk_len, 0.0f);
        std::copy(chunk_ptr, chunk_ptr + chunk_len, padded_samples.data() + pad_left);

        mtmd_audio_mel out_chunk;
        bool ok = log_mel_spectrogram(padded_samples.data(), padded_samples.size(), 4, params, cache, out_chunk);
        if (!ok) {
            return false;
        }

        // Trim to PyTorch frame count
        out_chunk.n_len = std::min(out_chunk.n_len, pt_frames);

        output.push_back(std::move(out_chunk));
    }

    return true;
}

//
// mtmd_audio_streaming_istft implementation
//

mtmd_audio_streaming_istft::mtmd_audio_streaming_istft(int n_fft, int hop_length) :
    n_fft(n_fft),
    hop_length(hop_length),
    n_fft_bins(n_fft / 2 + 1),
    overlap_buffer(n_fft, 0.0f),
    window_sum_buffer(n_fft, 0.0f),
    padding_to_remove((n_fft - hop_length) / 2),
    ifft_in(n_fft * 2 * 4, 0.0f),  // extra space for recursive IFFT
    ifft_out(n_fft * 2 * 4, 0.0f) {
    GGML_ASSERT(n_fft > 0 && hop_length > 0 && hop_length <= n_fft);
    cache.fill_sin_cos_table(n_fft);
    cache.fill_hann_window(n_fft, true);
}

void mtmd_audio_streaming_istft::reset() {
    std::fill(overlap_buffer.begin(), overlap_buffer.end(), 0.0f);
    std::fill(window_sum_buffer.begin(), window_sum_buffer.end(), 0.0f);
    padding_to_remove = (n_fft - hop_length) / 2;
}

std::vector<float> mtmd_audio_streaming_istft::process_frame(const float * frame_spectrum) {
    std::vector<float> output(hop_length);

    // copy frequencies
    for (int j = 0; j < n_fft_bins; j++) {
        ifft_in[j * 2 + 0] = frame_spectrum[j * 2 + 0];
        ifft_in[j * 2 + 1] = frame_spectrum[j * 2 + 1];
    }

    // mirror negative frequencies
    for (int j = 1; j < n_fft_bins - 1; j++) {
        int mirror_idx              = n_fft - j;
        ifft_in[mirror_idx * 2 + 0] = ifft_in[j * 2 + 0];
        ifft_in[mirror_idx * 2 + 1] = -ifft_in[j * 2 + 1];  // conjugate
    }

    ifft(cache, ifft_in.data(), n_fft, ifft_out.data());

    // update window sum and overlap buffer
    for (int j = 0; j < n_fft; j++) {
        window_sum_buffer[j] += cache.hann_window[j] * cache.hann_window[j];
        overlap_buffer[j] += ifft_out[j * 2] * cache.hann_window[j];
    }

    // extract hop_length samples with normalization
    for (int i = 0; i < hop_length; i++) {
        if (window_sum_buffer[i] > 1e-8f) {
            output[i] = overlap_buffer[i] / window_sum_buffer[i];
        } else {
            output[i] = overlap_buffer[i];
        }
    }

    // shift buffers left by hop_length
    std::copy(overlap_buffer.begin() + hop_length, overlap_buffer.end(), overlap_buffer.begin());
    std::fill(overlap_buffer.end() - hop_length, overlap_buffer.end(), 0.0f);

    std::copy(window_sum_buffer.begin() + hop_length, window_sum_buffer.end(), window_sum_buffer.begin());
    std::fill(window_sum_buffer.end() - hop_length, window_sum_buffer.end(), 0.0f);

    // Remove padding if needed
    int to_remove = std::min(padding_to_remove, (int) output.size());
    padding_to_remove -= to_remove;
    output.erase(output.begin(), output.begin() + to_remove);

    return output;
}

std::vector<float> mtmd_audio_streaming_istft::flush() {
    std::vector<float> output;

    // Extract remaining samples from overlap buffer
    // Continue until we've extracted all meaningful samples
    int remaining = n_fft - hop_length;
    while (remaining > 0) {
        int chunk_size = std::min(remaining, hop_length);

        for (int i = 0; i < chunk_size; i++) {
            float sample;
            if (window_sum_buffer[i] > 1e-8f) {
                sample = overlap_buffer[i] / window_sum_buffer[i];
            } else {
                sample = overlap_buffer[i];
            }
            output.push_back(sample);
        }

        // Shift buffers
        std::copy(overlap_buffer.begin() + chunk_size, overlap_buffer.end(), overlap_buffer.begin());
        std::fill(overlap_buffer.end() - chunk_size, overlap_buffer.end(), 0.0f);

        std::copy(window_sum_buffer.begin() + chunk_size, window_sum_buffer.end(), window_sum_buffer.begin());
        std::fill(window_sum_buffer.end() - chunk_size, window_sum_buffer.end(), 0.0f);

        remaining -= chunk_size;
    }

    return output;
}


// ---- mtmd-image implementation ----

//
// base implementation
//

void mtmd_image_preprocessor::img_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst, const float mean[3], const float std[3]) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    // TODO @ngxson : seems like this could be done more efficiently on cgraph
    for (size_t i = 0; i < src.buf.size(); ++i) {
        int c = i % 3; // rgb
        dst.buf[i] = (static_cast<float>(src.buf[i]) / 255.0f - mean[c]) / std[c];
    }
}

void mtmd_image_preprocessor::img_u8_to_f32(const clip_image_u8 & src, clip_image_f32 & dst) {
    dst.nx = src.nx;
    dst.ny = src.ny;
    dst.buf.resize(src.buf.size());

    for (size_t i = 0; i < src.buf.size(); ++i) {
        dst.buf[i] = static_cast<float>(src.buf[i]);
    }
}

// set of tools to manipulate images
// in the future, we can have HW acceleration by allowing this struct to access 3rd party lib like imagick or opencv
struct img_tool {
    static void resize(
            const clip_image_u8 & src,
            clip_image_u8 & dst,
            const clip_image_size & target_resolution,
            resize_algo algo,
            bool add_padding = true, // TODO: define the behavior for add_padding = false
            std::array<uint8_t, 3> pad_color = {0, 0, 0}) {
        dst.nx = target_resolution.width;
        dst.ny = target_resolution.height;
        dst.buf.resize(3 * dst.nx * dst.ny);

        if (dst.nx == src.nx && dst.ny == src.ny) {
            // no resize needed, simple copy
            dst.buf = src.buf;
            return;
        }

        if (!add_padding) {
            // direct resize
            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, dst, target_resolution.width, target_resolution.height);
                    break;
                case RESIZE_ALGO_BICUBIC_PILLOW:
                    resize_bicubic_pillow(src, dst, target_resolution.width, target_resolution.height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }
        } else {
            // resize with padding
            clip_image_u8 resized_image;
            float scale_w = static_cast<float>(target_resolution.width) / src.nx;
            float scale_h = static_cast<float>(target_resolution.height) / src.ny;
            float scale = std::min(scale_w, scale_h);
            int new_width  = std::min(static_cast<int>(std::ceil(src.nx * scale)), target_resolution.width);
            int new_height = std::min(static_cast<int>(std::ceil(src.ny * scale)), target_resolution.height);

            switch (algo) {
                case RESIZE_ALGO_BILINEAR:
                    resize_bilinear(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_BICUBIC:
                    resize_bicubic(src, resized_image, new_width, new_height);
                    break;
                case RESIZE_ALGO_BICUBIC_PILLOW:
                    resize_bicubic_pillow(src, resized_image, new_width, new_height);
                    break;
                default:
                    throw std::runtime_error("Unsupported resize algorithm");
            }

            // fill dst with pad_color
            fill(dst, pad_color);

            int offset_x = (target_resolution.width  - new_width)  / 2;
            int offset_y = (target_resolution.height - new_height) / 2;

            composite(dst, resized_image, offset_x, offset_y);
        }
    }

    static void crop(const clip_image_u8 & image, clip_image_u8 & dst, int x, int y, int w, int h) {
        GGML_ASSERT(x >= 0 && y >= 0 && w > 0 && h > 0);
        GGML_ASSERT(x + w <= image.nx && y + h <= image.ny);
        dst.nx = w;
        dst.ny = h;
        dst.buf.resize(3 * w * h);

        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                int src_idx = 3 * ((y + i)*image.nx + (x + j));
                int dst_idx = 3 * (i*w + j);
                dst.buf[dst_idx]     = image.buf[src_idx];
                dst.buf[dst_idx + 1] = image.buf[src_idx + 1];
                dst.buf[dst_idx + 2] = image.buf[src_idx + 2];
            }
        }
    }

    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will be aligned to the nearest multiple of align_size
    // if H or W size is larger than longest_edge, it will be resized to longest_edge
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const int align_size, const int longest_edge) {
        GGML_ASSERT(align_size > 0);
        if (inp_size.width <= 0 || inp_size.height <= 0 || longest_edge <= 0) {
            return {0, 0};
        }

        float scale = std::min(static_cast<float>(longest_edge) / inp_size.width,
                               static_cast<float>(longest_edge) / inp_size.height);

        float target_width_f  = static_cast<float>(inp_size.width)  * scale;
        float target_height_f = static_cast<float>(inp_size.height) * scale;

        auto ceil_by_factor = [f = align_size](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
        int aligned_width  = ceil_by_factor(target_width_f);
        int aligned_height = ceil_by_factor(target_height_f);

        return {aligned_width, aligned_height};
    }

    // calculate the size of the **resized** image, while preserving the aspect ratio
    // the calculated size will have min_pixels <= W*H <= max_pixels
    // this is referred as "smart_resize" in transformers code
    static clip_image_size calc_size_preserved_ratio(const clip_image_size & inp_size, const int align_size, const int min_pixels, const int max_pixels) {
        GGML_ASSERT(align_size > 0);
        const int width  = inp_size.width;
        const int height = inp_size.height;

        auto round_by_factor = [f = align_size](float x) { return static_cast<int>(std::round(x / static_cast<float>(f))) * f; };
        auto ceil_by_factor  = [f = align_size](float x) { return static_cast<int>(std::ceil(x / static_cast<float>(f))) * f; };
        auto floor_by_factor = [f = align_size](float x) { return static_cast<int>(std::floor(x / static_cast<float>(f))) * f; };

        // always align up first
        int h_bar = std::max(align_size, round_by_factor(height));
        int w_bar = std::max(align_size, round_by_factor(width));

        if (h_bar * w_bar > max_pixels) {
            const auto beta = std::sqrt(static_cast<float>(height * width) / max_pixels);
            h_bar = std::max(align_size, floor_by_factor(height / beta));
            w_bar = std::max(align_size, floor_by_factor(width  / beta));
        } else if (h_bar * w_bar < min_pixels) {
            const auto beta = std::sqrt(static_cast<float>(min_pixels) / (height * width));
            h_bar = ceil_by_factor(height * beta);
            w_bar = ceil_by_factor(width * beta);
        }

        return {w_bar, h_bar};
    }

    // draw src image into dst image at offset (offset_x, offset_y)
    static void composite(clip_image_u8 & dst, const clip_image_u8 & src, int offset_x, int offset_y) {
        for (int y = 0; y < src.ny; ++y) {
            for (int x = 0; x < src.nx; ++x) {
                int dx = x + offset_x;
                int dy = y + offset_y;
                // skip pixels that would be out of bounds in the destination
                if (dx < 0 || dy < 0 || dx >= dst.nx || dy >= dst.ny) {
                    continue;
                }
                size_t dst_idx = 3 * (static_cast<size_t>(dy) * dst.nx + static_cast<size_t>(dx));
                size_t src_idx = 3 * (static_cast<size_t>(y) * src.nx + static_cast<size_t>(x));
                dst.buf[dst_idx + 0] = src.buf[src_idx + 0];
                dst.buf[dst_idx + 1] = src.buf[src_idx + 1];
                dst.buf[dst_idx + 2] = src.buf[src_idx + 2];
            }
        }
    }

    // fill the image with a solid color
    static void fill(clip_image_u8 & img, const std::array<uint8_t, 3> & color) {
        for (size_t i = 0; i < img.buf.size(); i += 3) {
            img.buf[i]     = color[0];
            img.buf[i + 1] = color[1];
            img.buf[i + 2] = color[2];
        }
    }

private:
    // Bilinear resize function
    static void resize_bilinear(const clip_image_u8 & src, clip_image_u8 & dst, int target_width, int target_height) {
        if (src.nx == 0 || src.ny == 0) { dst.nx = dst.ny = 0; dst.buf.clear(); return; }
        if (target_width  <= 0) target_width  = 1;
        if (target_height <= 0) target_height = 1;

        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float x_ratio = target_width  > 1 ? static_cast<float>(src.nx - 1) / (target_width  - 1) : 0.0f;
        float y_ratio = target_height > 1 ? static_cast<float>(src.ny - 1) / (target_height - 1) : 0.0f;

        for (int y = 0; y < target_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                float px = x * x_ratio;
                float py = y * y_ratio;

                int x0 = std::min(static_cast<int>(px), src.nx - 1);
                int y0 = std::min(static_cast<int>(py), src.ny - 1);
                int x1 = std::min(x0 + 1, src.nx - 1);
                int y1 = std::min(y0 + 1, src.ny - 1);

                float xf = px - x0;
                float yf = py - y0;

                for (int c = 0; c < 3; ++c) {
                    float top    = lerp(static_cast<float>(src.buf[3 * (y0 * src.nx + x0) + c]),
                                        static_cast<float>(src.buf[3 * (y0 * src.nx + x1) + c]),
                                        xf);
                    float bottom = lerp(static_cast<float>(src.buf[3 * (y1 * src.nx + x0) + c]),
                                        static_cast<float>(src.buf[3 * (y1 * src.nx + x1) + c]),
                                        xf);
                    dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(lerp(top, bottom, yf));
                }
            }
        }
    }

    // Bicubic resize function
    // part of image will be cropped if the aspect ratio is different
    static bool resize_bicubic(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        const int nx = img.nx;
        const int ny = img.ny;

        dst.nx = target_width;
        dst.ny = target_height;
        dst.buf.resize(3 * target_width * target_height);

        float Cc;
        float C[5] = {};
        float d0, d2, d3, a0, a1, a2, a3;
        int i, j, k, jj;
        int x, y;
        float dx, dy;
        float tx, ty;

        tx = (float)nx / (float)target_width;
        ty = (float)ny / (float)target_height;

        // Bicubic interpolation; adapted from ViT.cpp, inspired from :
        //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
        //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

        for (i = 0; i < target_height; i++) {
            for (j = 0; j < target_width; j++) {
                x = (int)(tx * j);
                y = (int)(ty * i);

                dx = tx * j - x;
                dy = ty * i - y;

                for (k = 0; k < 3; k++) {
                    for (jj = 0; jj <= 3; jj++) {
                        d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                        a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                        C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                        d0 = C[0] - C[1];
                        d2 = C[2] - C[1];
                        d3 = C[3] - C[1];
                        a0 = C[1];
                        a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                        a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                        a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                        Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                        const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                        dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                    }
                }
            }
        }

        return true;
    }

    // Bicubic resize function using Pillow's ImagingResample algorithm
    // Adapted from https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
    //
    // Key Difference with resize_bicubic:
    // 1. Uses separable filtering: horizontal pass followed by vertical pass
    // 2. Pre-computes normalized filter coefficients for each output pixel
    // 3. Applies convolution using fixed-point integer arithmetic for performance
    static bool resize_bicubic_pillow(const clip_image_u8 & img, clip_image_u8 & dst, int target_width, int target_height) {
        // Fixed-point precision: 22 bits = 32 (int32_t) - 8 (uint8_t pixels) - 2 (headroom for accumulation)
        // This allows encoding fractional weights as integers: weight * 2^22
        const int PRECISION_BITS = 32 - 8 - 2;

        // Bicubic filter function with a = -0.5 (Note that GGML/PyTorch takes a = -0.75)
        // Returns filter weight for distance x from pixel center
        // Support: [-2, 2], meaning the filter influences pixels within 2 units of distance
        auto bicubic_filter = [](double x) -> double {
            constexpr double a = -0.5;
            if (x < 0.0) {
                x = -x;
            }
            if (x < 1.0) {
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
            }
            if (x < 2.0) {
                return (((x - 5) * x + 8) * x - 4) * a;
            }
            return 0.0;  // Zero outside [-2, 2]
        };

        // Filter support radius: bicubic extends 2 pixels in each direction
        constexpr double filter_support = 2.0;

        // Clipping function for 8-bit values
        auto clip8 = [](int val) -> uint8_t {
            if (val < 0) return 0;
            if (val > 255) return 255;
            return static_cast<uint8_t>(val);
        };

        // Precompute filter coefficients for ONE dimension (horizontal or vertical)
        //
        // Parameters:
        //   inSize  - Number of pixels in input dimension (e.g., src_width or src_height)
        //   outSize - Number of pixels in output dimension (e.g., target_width or target_height)
        //   bounds  - [OUTPUT] Array of size outSize*2 storing input pixel ranges:
        //             bounds[xx*2+0] = first input pixel index for output pixel xx (xmin)
        //             bounds[xx*2+1] = number of input pixels for output pixel xx (xcnt)
        //   weights - [OUTPUT] Array of size outSize*ksize storing fixed-point filter weights:
        //             kk[xx*ksize + x] = weight for input pixel x contributing to output pixel xx
        //
        // Returns: kernel size (ksize) - number of input pixels that contribute to each output pixel
        auto precompute_weights = [&](int inSize, int outSize,
                                     std::vector<int> & bounds, std::vector<int32_t> & weights) -> int {
            GGML_ASSERT(inSize > 0 && outSize > 0);
            double support, scale, filterscale;
            double center, ww, ss;
            int xx, x, ksize, xmin, xmax, xcnt;

            // Calculate scaling factor: ratio of input range to output size
            filterscale = scale = (double)inSize / outSize;
            // For upsampling (scale < 1), keep filterscale = 1 to maintain filter sharpness
            // For downsampling (scale > 1), widen filter to prevent aliasing
            if (filterscale < 1.0) {
                filterscale = 1.0;
            }

            // Determine filter support radius and kernel size
            support = filter_support * filterscale;  // Widen filter when downsampling
            ksize = static_cast<int>(std::ceil(support)) * 2 + 1;  // Total pixels in kernel

            std::vector<double> pre_weights(outSize * ksize);  // Temporary weights
            bounds.resize(outSize * 2);

            // For each output pixel, compute its filter coefficients
            for (xx = 0; xx < outSize; xx++) {
                // Calculate the center position in input space (pixel-center convention: +0.5)
                center = (xx + 0.5) * scale;
                ww = 0.0;  // Sum of weights for normalization
                ss = 1.0 / filterscale;  // Scale factor for filter function

                // Determine the range of input pixels that contribute to this output pixel
                xmin = static_cast<int>(center - support + 0.5);
                if (xmin < 0) {
                    xmin = 0;
                }

                xmax = static_cast<int>(center + support + 0.5);
                if (xmax > inSize) {
                    xmax = inSize;
                }

                xcnt = xmax - xmin;

                // Compute filter weights for each contributing input pixel
                for (x = 0; x < xcnt; x++) {
                    // Distance from input pixel center to output pixel center in input space
                    double w = bicubic_filter((x + xmin - center + 0.5) * ss);
                    pre_weights[xx * ksize + x] = w;
                    ww += w;  // Accumulate for normalization
                }

                // Normalize weights to sum to 1.0 (preserves brightness)
                for (x = 0; x < xcnt; x++) {
                    if (ww != 0.0) {
                        pre_weights[xx * ksize + x] /= ww;
                    }
                }

                // Zero-pad remaining kernel positions
                for (; x < ksize; x++) {
                    pre_weights[xx * ksize + x] = 0;
                }

                // Store input pixel range for this output pixel
                bounds[xx * 2 + 0] = xmin;
                bounds[xx * 2 + 1] = xcnt;
            }

            // Convert floating-point coefficients to fixed-point integers
            // Formula: int32 = round(float * 2^PRECISION_BITS)
            weights.resize(outSize * ksize);
            for (int i = 0; i < outSize * ksize; i++) {
                if (pre_weights[i] < 0) {
                    weights[i] = static_cast<int32_t>(-0.5 + pre_weights[i] * (1 << PRECISION_BITS));
                } else {
                    weights[i] = static_cast<int32_t>(0.5 + pre_weights[i] * (1 << PRECISION_BITS));
                }
            }

            return ksize;
        };

        // Horizontal resampling pass
        // Resizes width from imIn.nx to imOut.nx, preserving height
        auto resample_horizontal = [&](const clip_image_u8 & imIn, clip_image_u8 & imOut,
                                       int ksize, const std::vector<int> & bounds, const std::vector<int32_t> & weights) {
            imOut.ny = imIn.ny;
            imOut.buf.resize(3 * imOut.nx * imOut.ny);

            // Process each row independently
            for (int yy = 0; yy < imOut.ny; yy++) {
                // For each output pixel in this row
                for (int xx = 0; xx < imOut.nx; xx++) {
                    // Get the range of input pixels and filter coefficients
                    int xmin = bounds[xx * 2 + 0];  // First input pixel index
                    int xcnt = bounds[xx * 2 + 1];  // Number of input pixels

                    // Initialize accumulators for RGB channels with rounding bias (0.5 in fixed-point)
                    int32_t ss0 = 1 << (PRECISION_BITS - 1);
                    int32_t ss1 = 1 << (PRECISION_BITS - 1);
                    int32_t ss2 = 1 << (PRECISION_BITS - 1);

                    // Convolve: sum weighted input pixels
                    for (int x = 0; x < xcnt; x++) {
                        int src_idx = ((yy * imIn.nx) + (x + xmin)) * 3;
                        ss0 += static_cast<uint8_t>(imIn.buf[src_idx + 0]) * weights[xx * ksize + x];  // R channel
                        ss1 += static_cast<uint8_t>(imIn.buf[src_idx + 1]) * weights[xx * ksize + x];  // G channel
                        ss2 += static_cast<uint8_t>(imIn.buf[src_idx + 2]) * weights[xx * ksize + x];  // B channel
                    }

                    // Convert back from fixed-point (divide by 2^PRECISION_BITS) and clamp to [0,255]
                    int dst_idx = (yy * imOut.nx + xx) * 3;
                    imOut.buf[dst_idx + 0] = clip8(ss0 >> PRECISION_BITS);
                    imOut.buf[dst_idx + 1] = clip8(ss1 >> PRECISION_BITS);
                    imOut.buf[dst_idx + 2] = clip8(ss2 >> PRECISION_BITS);
                }
            }
        };

        // Vertical resampling pass
        // Resizes height from imIn.ny to imOut.ny, preserving width
        auto resample_vertical = [&](const clip_image_u8 & imIn, clip_image_u8 & imOut,
                                     int ksize, const std::vector<int> & bounds, const std::vector<int32_t> & weight) {
            imOut.nx = imIn.nx;
            imOut.buf.resize(3 * imOut.nx * imOut.ny);

            // For each output row
            for (int yy = 0; yy < imOut.ny; yy++) {
                // Get the range of input rows and filter coefficients
                int ymin = bounds[yy * 2 + 0];  // First input row index
                int ycnt = bounds[yy * 2 + 1];  // Number of input rows

                // Process each column in this output row
                for (int xx = 0; xx < imOut.nx; xx++) {
                    // Initialize accumulators for RGB channels with rounding bias
                    int32_t ss0 = 1 << (PRECISION_BITS - 1);
                    int32_t ss1 = 1 << (PRECISION_BITS - 1);
                    int32_t ss2 = 1 << (PRECISION_BITS - 1);

                    // Convolve: sum weighted input pixels vertically
                    for (int y = 0; y < ycnt; y++) {
                        int src_idx = ((y + ymin) * imIn.nx + xx) * 3;
                        ss0 += static_cast<uint8_t>(imIn.buf[src_idx + 0]) * weight[yy * ksize + y];  // R channel
                        ss1 += static_cast<uint8_t>(imIn.buf[src_idx + 1]) * weight[yy * ksize + y];  // G channel
                        ss2 += static_cast<uint8_t>(imIn.buf[src_idx + 2]) * weight[yy * ksize + y];  // B channel
                    }

                    // Convert back from fixed-point and clamp to [0,255]
                    int dst_idx = (yy * imOut.nx + xx) * 3;
                    imOut.buf[dst_idx + 0] = clip8(ss0 >> PRECISION_BITS);
                    imOut.buf[dst_idx + 1] = clip8(ss1 >> PRECISION_BITS);
                    imOut.buf[dst_idx + 2] = clip8(ss2 >> PRECISION_BITS);
                }
            }
        };

        // Main resampling logic using separable two-pass approach
        const int src_width = img.nx;
        const int src_height = img.ny;

        dst.nx = target_width;
        dst.ny = target_height;

        bool need_horizontal = (target_width != src_width);
        bool need_vertical = (target_height != src_height);

        // Precompute filter coefficients for both dimensions
        std::vector<int> bounds_horiz, bounds_vert;
        std::vector<int32_t> weights_horiz, weights_vert;
        int ksize_horiz = 0, ksize_vert = 0;

        if (need_horizontal) {
            ksize_horiz = precompute_weights(src_width, target_width, bounds_horiz, weights_horiz);
        }

        if (need_vertical) {
            ksize_vert = precompute_weights(src_height, target_height, bounds_vert, weights_vert);
        }

        // Perform two-pass resampling
        if (need_horizontal && need_vertical) {
            // Both horizontal and vertical
            clip_image_u8 temp;
            temp.nx = target_width;
            resample_horizontal(img, temp, ksize_horiz, bounds_horiz, weights_horiz);
            resample_vertical(temp, dst, ksize_vert, bounds_vert, weights_vert);
        } else if (need_horizontal) {
            // Only horizontal
            resample_horizontal(img, dst, ksize_horiz, bounds_horiz, weights_horiz);
        } else if (need_vertical) {
            // Only vertical
            resample_vertical(img, dst, ksize_vert, bounds_vert, weights_vert);
        } else {
            // No resizing needed - direct copy
            dst.buf = img.buf;
        }

        return true;
    }

    static inline int clip(int x, int lower, int upper) {
        return std::max(lower, std::min(x, upper));
    }

    // Linear interpolation between two points
    static inline float lerp(float s, float e, float t) {
        return s + (e - s) * t;
    }
};


//
// mtmd_image_preprocessor_llava_uhd
//

bool mtmd_image_preprocessor_llava_uhd::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    const clip_image_size original_size{img.nx, img.ny};
    auto const inst = get_slice_instructions(original_size);
    std::vector<clip_image_u8_ptr> imgs = slice_image(img, inst);

    for (size_t i = 0; i < imgs.size(); ++i) {
        // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
        clip_image_f32_ptr res(clip_image_f32_init());
        img_u8_to_f32(*imgs[i], *res, hparams.image_mean, hparams.image_std);
        output.entries.push_back(std::move(res));
    }

    output.grid_x = inst.grid_size.width;
    output.grid_y = inst.grid_size.height;
    return true;
}

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_llava_uhd::get_slice_instructions(const clip_image_size & original_size) {
    mtmd_image_preprocessor_llava_uhd::slice_instructions res;
    const int patch_size      = hparams.patch_size;
    const int slice_size      = hparams.image_size;
    const int original_width  = original_size.width;
    const int original_height = original_size.height;

    const bool has_slices    = original_size.width > slice_size || original_size.height > slice_size;
    const bool has_pinpoints = !hparams.image_res_candidates.empty();

    if (!has_slices) {
        // skip slicing logic
        res.overview_size = clip_image_size{slice_size, slice_size};
        res.refined_size  = clip_image_size{0, 0};
        res.grid_size     = clip_image_size{0, 0};

        return res;
    }

    if (has_pinpoints) {
        // has pinpoints, use them to calculate the grid size (e.g. llava-1.6)
        auto refine_size = select_best_resolution(
            original_size,
            hparams.image_res_candidates);
        res.overview_size         = clip_image_size{slice_size, slice_size};
        res.refined_size          = refine_size;
        res.grid_size             = clip_image_size{0, 0};

        LOG_DBG("%s: using pinpoints for slicing\n", __func__);
        LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d\n",
                __func__, original_width, original_height,
                res.overview_size.width, res.overview_size.height,
                res.refined_size.width,  res.refined_size.height);

        for (int y = 0; y < refine_size.height; y += slice_size) {
            for (int x = 0; x < refine_size.width; x += slice_size) {
                slice_coordinates slice;
                slice.x = x;
                slice.y = y;
                slice.size.width  = std::min(slice_size, refine_size.width  - x);
                slice.size.height = std::min(slice_size, refine_size.height - y);
                res.slices.push_back(slice);
                LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                        __func__, (int)res.slices.size() - 1,
                        slice.x, slice.y, slice.size.width, slice.size.height);
            }
        }

        res.grid_size.height = refine_size.height / slice_size;
        res.grid_size.width  = refine_size.width  / slice_size;
        LOG_DBG("%s: grid size: %d x %d\n", __func__, res.grid_size.width, res.grid_size.height);

        return res;
    }

    // no pinpoints, dynamically calculate the grid size (e.g. minicpmv)

    auto best_size    = get_best_resize(original_size, slice_size, patch_size, !has_slices);
    res.overview_size = best_size;

    {
        const int max_slice_nums = 9; // TODO: this is only used by minicpmv, maybe remove it
        const float log_ratio = log((float)original_width / original_height);
        const float ratio = (float)original_width * original_height / (slice_size * slice_size);
        const int multiple = fmin(ceil(ratio), max_slice_nums);

        auto best_grid   = get_best_grid(max_slice_nums, multiple, log_ratio);
        auto refine_size = get_refine_size(original_size, best_grid, slice_size, patch_size, true);
        res.grid_size    = best_grid;
        res.refined_size = refine_size;

        LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n",
                __func__, original_width, original_height,
                res.overview_size.width, res.overview_size.height,
                res.refined_size.width, res.refined_size.height,
                res.grid_size.width, res.grid_size.height);

        int width  = refine_size.width;
        int height = refine_size.height;
        int grid_x = int(width  / best_grid.width);
        int grid_y = int(height / best_grid.height);
        for (int patches_y = 0,                    ic = 0;
                patches_y < refine_size.height && ic < best_grid.height;
                patches_y += grid_y,              ic += 1) {
            for (int patches_x = 0,                   jc = 0;
                    patches_x < refine_size.width && jc < best_grid.width;
                    patches_x += grid_x,             jc += 1) {
                slice_coordinates slice;
                slice.x = patches_x;
                slice.y = patches_y;
                slice.size.width  = grid_x;
                slice.size.height = grid_y;
                res.slices.push_back(slice);
                LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n",
                        __func__, (int)res.slices.size() - 1,
                        slice.x, slice.y, slice.size.width, slice.size.height);
            }
        }
    }

    return res;
}

std::vector<clip_image_u8_ptr> mtmd_image_preprocessor_llava_uhd::slice_image(const clip_image_u8 & img, const mtmd_image_preprocessor_llava_uhd::slice_instructions & inst, bool overview_first) {
    std::vector<clip_image_u8_ptr> output;

    // resize to overview size
    clip_image_u8_ptr resized_img(clip_image_u8_init());
    img_tool::resize(img, *resized_img, inst.overview_size, hparams.image_resize_algo_ov,
                        hparams.image_pad_ov, hparams.image_pad_color_ov);
    if (overview_first) {
        output.push_back(std::move(resized_img));
    }

    if (inst.slices.empty()) {
        // no slices, just return the resized image
        if (!overview_first) {
            output.push_back(std::move(resized_img));
        }
        return output;
    }

    // resize to refined size
    clip_image_u8_ptr refined_img(clip_image_u8_init());
    img_tool::resize(img, *refined_img, inst.refined_size, hparams.image_resize_algo_rf,
                        hparams.image_pad_rf, hparams.image_pad_color_rf);

    // create slices
    for (const auto & slice : inst.slices) {
        int x = slice.x;
        int y = slice.y;
        int w = slice.size.width;
        int h = slice.size.height;

        clip_image_u8_ptr img_slice(clip_image_u8_init());
        img_tool::crop(*refined_img, *img_slice, x, y, w, h);
        output.push_back(std::move(img_slice));
    }

    if (!overview_first) {
        output.push_back(std::move(resized_img));
    }

    return output;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::get_best_resize(const clip_image_size & original_size, int scale_resolution, int patch_size, bool allow_upscale) {
    int width  = original_size.width;
    int height = original_size.height;
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
        float r = static_cast<float>(width) / height;
        height  = static_cast<int>(scale_resolution / std::sqrt(r));
        width   = static_cast<int>(height * r);
    }
    clip_image_size res;
    res.width  = ensure_divide(width,  patch_size);
    res.height = ensure_divide(height, patch_size);
    return res;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::resize_maintain_aspect_ratio(const clip_image_size & orig, const clip_image_size & target_max) {
    float scale_width  = static_cast<float>(target_max.width)  / orig.width;
    float scale_height = static_cast<float>(target_max.height) / orig.height;
    float scale = std::min(scale_width, scale_height);
    return clip_image_size{
        static_cast<int>(orig.width  * scale),
        static_cast<int>(orig.height * scale),
    };
}

clip_image_size mtmd_image_preprocessor_llava_uhd::select_best_resolution(const clip_image_size & original_size, const std::vector<clip_image_size> & possible_resolutions) {
    clip_image_size best_fit;
    int min_wasted_area = std::numeric_limits<int>::max();
    int max_effective_resolution = 0;

    for (const clip_image_size & candidate : possible_resolutions) {
        auto target_size = resize_maintain_aspect_ratio(original_size, candidate);
        int effective_resolution = std::min(
            target_size.width * target_size.height,
            original_size.width * original_size.height);
        int wasted_area = (candidate.width * candidate.height) - effective_resolution;

        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_area < min_wasted_area)) {
            max_effective_resolution = effective_resolution;
            min_wasted_area = wasted_area;
            best_fit = candidate;
        }

        LOG_DBG("%s: candidate: %d x %d, target: %d x %d, wasted: %d, effective: %d\n", __func__, candidate.width, candidate.height, target_size.width, target_size.height, wasted_area, effective_resolution);
    }

    return best_fit;
}

int mtmd_image_preprocessor_llava_uhd::ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

clip_image_size mtmd_image_preprocessor_llava_uhd::get_refine_size(const clip_image_size & original_size, const clip_image_size & grid, int scale_resolution, int patch_size, bool allow_upscale) {
    int width  = original_size.width;
    int height = original_size.height;
    int grid_x = grid.width;
    int grid_y = grid.height;

    int refine_width  = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    clip_image_size grid_size;
    grid_size.width  = refine_width  / grid_x;
    grid_size.height = refine_height / grid_y;

    auto best_grid_size  = get_best_resize(grid_size, scale_resolution, patch_size, allow_upscale);
    int best_grid_width  = best_grid_size.width;
    int best_grid_height = best_grid_size.height;

    clip_image_size refine_size;
    refine_size.width  = best_grid_width  * grid_x;
    refine_size.height = best_grid_height * grid_y;
    return refine_size;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::get_best_grid(const int max_slice_nums, const int multiple, const float log_ratio) {
    std::vector<int> candidate_split_grids_nums;
    for (int i : {multiple - 1, multiple, multiple + 1}) {
        if (i == 1 || i > max_slice_nums) {
            continue;
        }
        candidate_split_grids_nums.push_back(i);
    }

    std::vector<clip_image_size> candidate_grids;
    for (int split_grids_nums : candidate_split_grids_nums) {
        int m = 1;
        while (m <= split_grids_nums) {
            if (split_grids_nums % m == 0) {
                candidate_grids.push_back(clip_image_size{m, split_grids_nums / m});
            }
            ++m;
        }
    }

    clip_image_size best_grid{1, 1};
    float min_error = std::numeric_limits<float>::infinity();
    for (const auto& grid : candidate_grids) {
        float error = std::abs(log_ratio - std::log(1.0 * grid.width / grid.height));
        if (error < min_error) {
            best_grid = grid;
            min_error = error;
        }
    }
    return best_grid;
}

//
// mtmd_image_preprocessor_fixed_size
//

bool mtmd_image_preprocessor_fixed_size::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    clip_image_u8 resized_image;
    int sz = hparams.image_size;
    img_tool::resize(img, resized_image, {sz, sz},
                        hparams.image_resize_algo,
                        hparams.image_resize_pad,
                        hparams.image_pad_color);
    clip_image_f32_ptr img_f32(clip_image_f32_init());
    img_u8_to_f32(resized_image, *img_f32, hparams.image_mean, hparams.image_std);
    output.entries.push_back(std::move(img_f32));
    return true;
}

//
// mtmd_image_preprocessor_dyn_size
//

bool mtmd_image_preprocessor_dyn_size::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    GGML_ASSERT(hparams.image_min_pixels > 0 && hparams.image_max_pixels > 0);
    clip_image_u8 resized_image;
    const clip_image_size original_size{img.nx, img.ny};
    // the original pixtral model doesn't have n_merge
    const int cur_merge = hparams.n_merge == 0 ? 1 : hparams.n_merge;
    const clip_image_size target_size = img_tool::calc_size_preserved_ratio(
        original_size,
        hparams.patch_size * cur_merge,
        hparams.image_min_pixels,
        hparams.image_max_pixels);
    img_tool::resize(img, resized_image, target_size,
                        hparams.image_resize_algo,
                        hparams.image_resize_pad,
                        hparams.image_pad_color);
    clip_image_f32_ptr img_f32(clip_image_f32_init());
    img_u8_to_f32(resized_image, *img_f32, hparams.image_mean, hparams.image_std);
    output.entries.push_back(std::move(img_f32));
    return true;
}

//
// mtmd_image_preprocessor_longest_edge
//

bool mtmd_image_preprocessor_longest_edge::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    GGML_ASSERT(hparams.image_longest_edge > 0);
    clip_image_u8 resized_image;
    const clip_image_size original_size{img.nx, img.ny};
    // the original pixtral model doesn't have n_merge
    const int cur_merge = hparams.n_merge == 0 ? 1 : hparams.n_merge;
    const clip_image_size target_size = img_tool::calc_size_preserved_ratio(
        original_size,
        hparams.patch_size * cur_merge,
        hparams.image_longest_edge);
    img_tool::resize(img, resized_image, target_size,
                        hparams.image_resize_algo,
                        hparams.image_resize_pad,
                        hparams.image_pad_color);
    clip_image_f32_ptr img_f32(clip_image_f32_init());
    img_u8_to_f32(resized_image, *img_f32, hparams.image_mean, hparams.image_std);
    output.entries.push_back(std::move(img_f32));
    return true;
}

//
// mtmd_image_preprocessor_lfm2
//

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_lfm2::get_slice_instructions(const clip_image_size & original_size) {
    mtmd_image_preprocessor_llava_uhd::slice_instructions inst;
    const int align_size = hparams.patch_size * hparams.n_merge;
    inst.overview_size = img_tool::calc_size_preserved_ratio(
                            original_size, align_size,
                            hparams.image_min_pixels, hparams.image_max_pixels);
    // tile if either dimension exceeds tile_size with tolerance
    const bool needs_tiling = original_size.width > tile_size * max_pixels_tolerance || original_size.height > tile_size * max_pixels_tolerance;

    if (!needs_tiling) {
        inst.refined_size = clip_image_size{0, 0};
        inst.grid_size    = clip_image_size{0, 0};
        return inst;
    }

    const clip_image_size grid = get_grid_layout(original_size.height, original_size.width);

    inst.grid_size    = grid;
    inst.refined_size = clip_image_size{tile_size * grid.width, tile_size * grid.height};

    LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n",
            __func__,
            original_size.width, original_size.height,
            inst.overview_size.width, inst.overview_size.height,
            inst.refined_size.width, inst.refined_size.height,
            grid.width, grid.height);

    for (int row = 0; row < grid.height; row++) {
        for (int col = 0; col < grid.width; col++) {
            mtmd_image_preprocessor_llava_uhd::slice_coordinates slice;
            slice.x    = col * tile_size;
            slice.y    = row * tile_size;
            slice.size = clip_image_size{tile_size, tile_size};
            inst.slices.push_back(slice);
            LOG_DBG("%s: slice %d: x=%d, y=%d, size=%d x %d\n",
                    __func__, (int)inst.slices.size() - 1,
                    slice.x, slice.y, slice.size.width, slice.size.height);
        }
    }

    return inst;
}

clip_image_size mtmd_image_preprocessor_lfm2::find_closest_aspect_ratio(
        float aspect_ratio,
        const std::vector<clip_image_size> & target_ratios,
        int width, int height) {
    float best_ratio_diff = std::numeric_limits<float>::max();
    clip_image_size best_ratio = {1, 1};
    const float area = static_cast<float>(width * height);

    for (const auto & ratio : target_ratios) {
        const float target_aspect_ratio = static_cast<float>(ratio.width) / ratio.height;
        const float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            const float target_area = static_cast<float>(tile_size * tile_size * ratio.width * ratio.height);
            if (area > 0.5f * target_area) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

std::vector<clip_image_size> mtmd_image_preprocessor_lfm2::get_target_ratios() {
    std::vector<clip_image_size> ratios;
    for (int n = min_tiles; n <= max_tiles; n++) {
        for (int w = 1; w <= n; w++) {
            for (int h = 1; h <= n; h++) {
                if (w * h >= min_tiles && w * h <= max_tiles) {
                    bool found = false;
                    for (const auto & r : ratios) {
                        if (r.width == w && r.height == h) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        ratios.push_back({w, h});
                    }
                }
            }
        }
    }
    std::sort(ratios.begin(), ratios.end(), [](const clip_image_size & a, const clip_image_size & b) {
        return a.width * a.height < b.width * b.height;
    });
    return ratios;
}

clip_image_size mtmd_image_preprocessor_lfm2::get_grid_layout(int height, int width) {
    const float aspect_ratio = static_cast<float>(width) / height;
    const auto ratios = get_target_ratios();
    return find_closest_aspect_ratio(aspect_ratio, ratios, width, height);
}

//
// mtmd_image_preprocessor_idefics3
//

bool mtmd_image_preprocessor_idefics3::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    // The refined size has two steps:
    // 1. Resize w/ aspect-ratio preserving such that the longer side is
    //      the preprocessor longest size
    // 2. Resize w/out preserving aspect ratio such that both sides are
    //      multiples of image_size (always rounding up)
    //
    // CITE: https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py#L737
    const clip_image_size original_size{img.nx, img.ny};
    const clip_image_size refined_size = img_tool::calc_size_preserved_ratio(
        original_size, hparams.image_size, hparams.image_longest_edge);
    // LOG_INF("%s: original size: %d x %d, refined size: %d x %d\n",
    //         __func__, original_size.width, original_size.height,
    //         refined_size.width, refined_size.height);

    mtmd_image_preprocessor_llava_uhd::slice_instructions instructions;
    instructions.overview_size = clip_image_size{hparams.image_size, hparams.image_size};
    instructions.refined_size = refined_size;
    instructions.grid_size = clip_image_size{
        static_cast<int>(std::ceil(static_cast<float>(refined_size.width) / hparams.image_size)),
        static_cast<int>(std::ceil(static_cast<float>(refined_size.height) / hparams.image_size)),
    };
    for (int y = 0; y < refined_size.height; y += hparams.image_size) {
        for (int x = 0; x < refined_size.width; x += hparams.image_size) {
            // LOG_INF("%s: adding slice at x=%d, y=%d\n", __func__, x, y);
            instructions.slices.push_back(mtmd_image_preprocessor_llava_uhd::slice_coordinates{
                /* x    */x,
                /* y    */y,
                /* size */clip_image_size{
                    std::min(hparams.image_size, refined_size.width - x),
                    std::min(hparams.image_size, refined_size.height - y)
                }
            });
        }
    }
    auto imgs = slice_image(img, instructions);

    // cast and normalize to f32
    for (size_t i = 0; i < imgs.size(); ++i) {
        // clip_image_save_to_bmp(*imgs[i], "slice_" + std::to_string(i) + ".bmp");
        clip_image_f32_ptr res(clip_image_f32_init());
        img_u8_to_f32(*imgs[i], *res, hparams.image_mean, hparams.image_std);
        output.entries.push_back(std::move(res));
    }

    output.grid_x = instructions.grid_size.width;
    output.grid_y = instructions.grid_size.height;
    return true;
}

//
// mtmd_image_preprocessor_internvl
//

bool mtmd_image_preprocessor_internvl::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    GGML_ASSERT(!hparams.image_res_candidates.empty());
    const clip_image_size original_size{img.nx, img.ny};
    auto const inst = get_slice_instructions(original_size);
    std::vector<clip_image_u8_ptr> imgs = slice_image(img, inst, false);

    for (size_t i = 0; i < imgs.size(); ++i) {
        clip_image_f32_ptr res(clip_image_f32_init());
        img_u8_to_f32(*imgs[i], *res, hparams.image_mean, hparams.image_std);
        output.entries.push_back(std::move(res));
    }
    return true;
}

//
// mtmd_image_preprocessor_deepseekocr
//

bool mtmd_image_preprocessor_deepseekocr::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    const std::vector native_resolutions = {
        /*512 tiny , 640 small, */ 1024 /* base */, 1280 /* large */
    };
    // original image size
    const clip_image_size original_size{img.nx, img.ny};
    const int orig_w = original_size.width;
    const int orig_h = original_size.height;
    const int orig_area = orig_h * orig_w;

    size_t mode_i = 0;
    int min_diff = orig_area;

    for (size_t i = 0; i < native_resolutions.size(); i++) {
        int r = native_resolutions[i];
        if (std::abs(orig_area - r * r) < min_diff) {
            mode_i = i;
            min_diff = std::abs(orig_area - r * r);
        }
    }

    /* Native Resolution (Base/Large) */
    const int image_size = native_resolutions[mode_i];

    // scaled and padded image
    clip_image_u8_ptr scaled_img(clip_image_u8_init());
    img_tool::resize(img, *scaled_img, clip_image_size{image_size, image_size}, hparams.image_resize_algo);

    clip_image_f32_ptr res(clip_image_f32_init());
    img_u8_to_f32(*scaled_img, *res, hparams.image_mean, hparams.image_std);
    output.entries.push_back(std::move(res));

    output.grid_x = 1;
    output.grid_y = 1;
    return true;
}

//
// mtmd_image_preprocessor_step3vl
//

void mtmd_image_preprocessor_step3vl::img_u8_resize_bilinear_to_f32(
        const clip_image_u8 & src,
        clip_image_f32 & dst,
        int target_width,
        int target_height,
        const float mean[3],
        const float std[3]) {
    if (src.nx == target_width && src.ny == target_height) {
        img_u8_to_f32(src, dst, mean, std);
        return;
    }

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    const float scale_x = static_cast<float>(src.nx) / target_width;
    const float scale_y = static_cast<float>(src.ny) / target_height;

    for (int y = 0; y < target_height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0_floor = static_cast<int>(std::floor(src_y));
        const int y0 = std::max(0, std::min(y0_floor, src.ny - 1));
        const int y1 = std::max(0, std::min(y0_floor + 1, src.ny - 1));
        const float ly = src_y - y0_floor;

        for (int x = 0; x < target_width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0_floor = static_cast<int>(std::floor(src_x));
            const int x0 = std::max(0, std::min(x0_floor, src.nx - 1));
            const int x1 = std::max(0, std::min(x0_floor + 1, src.nx - 1));
            const float lx = src_x - x0_floor;

            const size_t idx00 = 3 * (y0 * src.nx + x0);
            const size_t idx01 = 3 * (y0 * src.nx + x1);
            const size_t idx10 = 3 * (y1 * src.nx + x0);
            const size_t idx11 = 3 * (y1 * src.nx + x1);
            const size_t idx_dst = 3 * (y * target_width + x);

            for (int c = 0; c < 3; ++c) {
                const float v00 = (static_cast<float>(src.buf[idx00 + c]) / 255.0f - mean[c]) / std[c];
                const float v01 = (static_cast<float>(src.buf[idx01 + c]) / 255.0f - mean[c]) / std[c];
                const float v10 = (static_cast<float>(src.buf[idx10 + c]) / 255.0f - mean[c]) / std[c];
                const float v11 = (static_cast<float>(src.buf[idx11 + c]) / 255.0f - mean[c]) / std[c];

                const float top = v00 + (v01 - v00) * lx;
                const float bot = v10 + (v11 - v10) * lx;
                dst.buf[idx_dst + c] = top + (bot - top) * ly;
            }
        }
    }
}

int mtmd_image_preprocessor_step3vl::get_image_longest_edge(const clip_hparams & params) {
    return params.image_longest_edge > 0 ? params.image_longest_edge : default_image_longest_edge;
}

int mtmd_image_preprocessor_step3vl::determine_window_size(const clip_hparams & params, int longer, int shorter) {
    const int image_size = params.image_size;
    const int crop_size  = default_image_crop_size;
    const float aspect_ratio = static_cast<float>(longer) / shorter;

    if (longer <= image_size) {
        return aspect_ratio > small_aspect_ratio_limit ? shorter : 0;
    }

    return aspect_ratio > wide_aspect_ratio_limit ? std::min(shorter, crop_size) : crop_size;
}

int mtmd_image_preprocessor_step3vl::calc_crop_extent(int length, int window_size) {
    const float ratio = static_cast<float>(length) / window_size;
    if (ratio < 1.0f) {
        return length;
    }

    const float decimal = ratio - std::floor(ratio);
    const int rounded = decimal > crop_rounding_threshold
        ? static_cast<int>(std::floor(ratio)) + 1
        : static_cast<int>(std::floor(ratio));
    return window_size * rounded;
}

std::vector<int> mtmd_image_preprocessor_step3vl::calc_grid(int length, int window_size) {
    const int n = length <= window_size
        ? 1
        : static_cast<int>(std::ceil(static_cast<float>(length - window_size) / window_size + 1.0f));
    std::vector<int> starts(n);

    for (int i = 0; i < n; ++i) {
        starts[i] = window_size * i;
    }

    if (n > 1 && starts.back() + window_size > length) {
        starts.back() = length - window_size;
    }

    return starts;
}

clip_image_u8 mtmd_image_preprocessor_step3vl::prepare_image(const clip_image_u8 & img, const clip_hparams & params) {
    clip_image_u8 resized = img;
    const float aspect_ratio = img.ny > 0 ? static_cast<float>(img.nx) / img.ny : 1.0f;
    if (std::min(img.nx, img.ny) < 32 &&
        (aspect_ratio > wide_aspect_ratio_limit ||
         aspect_ratio < 1.0f / wide_aspect_ratio_limit)) {
        const int square_size = std::max(img.nx, img.ny);
        clip_image_u8 padded;
        padded.nx = square_size;
        padded.ny = square_size;
        padded.buf.resize(3 * square_size * square_size);
        img_tool::fill(padded, {0, 0, 0});
        img_tool::composite(padded, img, 0, 0);
        resized = std::move(padded);
    }

    const int max_image_size = get_image_longest_edge(params);
    if (std::max(resized.nx, resized.ny) > max_image_size) {
        const float scale = static_cast<float>(max_image_size) / std::max(resized.nx, resized.ny);
        const clip_image_size new_size = {
            std::max(1, static_cast<int>(std::floor(resized.nx * scale))),
            std::max(1, static_cast<int>(std::floor(resized.ny * scale))),
        };
        clip_image_u8 scaled;
        img_tool::resize(resized, scaled, new_size, RESIZE_ALGO_BILINEAR, false);
        resized = std::move(scaled);
    }

    return resized;
}

clip_image_u8 mtmd_image_preprocessor_step3vl::crop_with_black_padding(const clip_image_u8 & image, int x, int y, int w, int h) {
    clip_image_u8 dst;
    dst.nx = w;
    dst.ny = h;
    dst.buf.resize(3 * w * h, 0);

    const int src_x0 = std::max(0, x);
    const int src_y0 = std::max(0, y);
    const int src_x1 = std::min(image.nx, x + w);
    const int src_y1 = std::min(image.ny, y + h);

    if (src_x0 >= src_x1 || src_y0 >= src_y1) {
        return dst;
    }

    const int dst_x0 = src_x0 - x;
    const int dst_y0 = src_y0 - y;

    for (int yy = 0; yy < src_y1 - src_y0; ++yy) {
        for (int xx = 0; xx < src_x1 - src_x0; ++xx) {
            const int src_idx = 3 * ((src_y0 + yy) * image.nx + (src_x0 + xx));
            const int dst_idx = 3 * ((dst_y0 + yy) * w + (dst_x0 + xx));
            dst.buf[dst_idx + 0] = image.buf[src_idx + 0];
            dst.buf[dst_idx + 1] = image.buf[src_idx + 1];
            dst.buf[dst_idx + 2] = image.buf[src_idx + 2];
        }
    }

    return dst;
}

mtmd_image_preprocessor_step3vl::slice_instructions mtmd_image_preprocessor_step3vl::build_slice_instructions(
        const clip_hparams & params,
        const clip_image_size & prepared_size) {
    slice_instructions instructions;
    instructions.overview_size = prepared_size;

    const int window_size = determine_window_size(
        params,
        std::max(prepared_size.width, prepared_size.height),
        std::min(prepared_size.width, prepared_size.height));
    if (window_size <= 0) {
        instructions.refined_size = clip_image_size{0, 0};
        instructions.grid_size    = clip_image_size{0, 0};
        return instructions;
    }

    const int crop_width  = calc_crop_extent(prepared_size.width,  window_size);
    const int crop_height = calc_crop_extent(prepared_size.height, window_size);
    instructions.refined_size = clip_image_size{crop_width, crop_height};

    const auto xs = calc_grid(crop_width,  window_size);
    const auto ys = calc_grid(crop_height, window_size);
    instructions.grid_size = clip_image_size{
        static_cast<int>(xs.size()),
        static_cast<int>(ys.size()),
    };

    for (int y : ys) {
        for (int x : xs) {
            instructions.slices.push_back(slice_coordinates{
                /* x    */ x,
                /* y    */ y,
                /* size */ clip_image_size{window_size, window_size},
            });
        }
    }

    return instructions;
}

bool mtmd_image_preprocessor_step3vl::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    clip_image_u8 prepared = prepare_image(img, hparams);
    const auto instructions = build_slice_instructions(hparams, {prepared.nx, prepared.ny});

    clip_image_f32_ptr overview_f32(clip_image_f32_init());
    img_u8_resize_bilinear_to_f32(
        prepared,
        *overview_f32,
        hparams.image_size,
        hparams.image_size,
        hparams.image_mean,
        hparams.image_std);
    output.entries.push_back(std::move(overview_f32));

    if (instructions.slices.empty()) {
        output.grid_x = 0;
        output.grid_y = 0;
        return true;
    }

    clip_image_u8 img_for_crop = prepared;
    if (instructions.refined_size.width != prepared.nx || instructions.refined_size.height != prepared.ny) {
        clip_image_u8 refined;
        img_tool::resize(prepared, refined, instructions.refined_size, RESIZE_ALGO_BILINEAR, false);
        img_for_crop = std::move(refined);
    }

    const int crop_size = default_image_crop_size;
    for (const auto & slice : instructions.slices) {
        // If the requested patch extends past the source image, pad the out-of-bounds area with black.
        clip_image_u8 patch = crop_with_black_padding(img_for_crop, slice.x, slice.y, slice.size.width, slice.size.height);

        clip_image_f32_ptr patch_f32(clip_image_f32_init());
        img_u8_resize_bilinear_to_f32(
            patch,
            *patch_f32,
            crop_size,
            crop_size,
            hparams.image_mean,
            hparams.image_std);
        output.entries.push_back(std::move(patch_f32));
    }

    output.grid_x = instructions.grid_size.width;
    output.grid_y = instructions.grid_size.height;

    return true;
}

//
// mtmd_image_preprocessor_youtuvl
//

bool mtmd_image_preprocessor_youtuvl::preprocess(const clip_image_u8 & img, clip_image_f32_batch & output) {
    const int patch_size = hparams.patch_size;   // typically 16
    const int merge_size = hparams.n_merge;      // typically 2
    const int align_size = patch_size * merge_size;  // 32

    const int max_num_patches = hparams.image_max_pixels > 0 ?
        hparams.image_max_pixels / (patch_size * patch_size) : 256;

    // Linear search for optimal scale to fit within max_num_patches
    float scale = 1.0f;
    int target_height = img.ny;
    int target_width  = img.nx;

    auto get_scaled_image_size = [align_size](float scale, int size) -> int {
        float scaled_size = size * scale;
        // Round up to nearest multiple of align_size
        int aligned = static_cast<int>(std::ceil(scaled_size / align_size)) * align_size;
        // Ensure at least one patch
        return std::max(align_size, aligned);
    };

    // Linear search with 0.02 step size
    while (scale > 0.0f) {
        target_height = get_scaled_image_size(scale, img.ny);
        target_width  = get_scaled_image_size(scale, img.nx);

        int num_patches_h = target_height / patch_size;
        int num_patches_w = target_width / patch_size;
        int num_patches = num_patches_h * num_patches_w;

        if (num_patches > max_num_patches) {
            scale -= 0.02f;
        } else {
            break;
        }
    }

    clip_image_size new_size = {target_width, target_height};

    // Resize the image
    clip_image_u8 resized;
    img_tool::resize(img, resized, new_size, hparams.image_resize_algo, hparams.image_resize_pad);

    // Normalize to float32
    clip_image_f32_ptr img_f32(clip_image_f32_init());
    img_u8_to_f32(resized, *img_f32, hparams.image_mean, hparams.image_std);
    // Add to results
    output.entries.push_back(std::move(img_f32));
    return true;
}


// ---- mtmd helper implementation ----

//#define MTMD_AUDIO_DEBUG

#define MINIAUDIO_IMPLEMENTATION
#ifndef MTMD_AUDIO_DEBUG
#   define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h.inc"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h.inc"


//
// internal logging functions
//

struct mtmd_helper_logger {
    ggml_log_callback default_callback = [](ggml_log_level level, const char * text, void * user_data) {
        (void) level;
        (void) user_data;
        fputs(text, stderr);
        fflush(stderr);
    };

    ggml_log_callback log_callback = default_callback;
    void * log_callback_user_data;

    void log_v(enum ggml_log_level level, const char * format, va_list args) {
        if (format == NULL) {
            return;
        }
        va_list args_copy;
        va_copy(args_copy, args);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            log_callback(level, buffer, log_callback_user_data);
        } else {
            char * buffer2 = (char *) calloc(len + 1, sizeof(char));
            vsnprintf(buffer2, len + 1, format, args_copy);
            buffer2[len] = 0;
            log_callback(level, buffer2, log_callback_user_data);
            free(buffer2);
        }
        va_end(args_copy);
    }

    void log(enum ggml_log_level level, const char * format, ...) {
        va_list args;
        va_start(args, format);
        log_v(level, format, args);
        va_end(args);
    }
} g_logger;

#undef LOG_INF
#undef LOG_WRN
#undef LOG_ERR

#define LOG_INF(...) g_logger.log(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) g_logger.log(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) g_logger.log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data) {
    if (log_callback == nullptr) {
        log_callback = g_logger.default_callback;
    }
    g_logger.log_callback = log_callback;
    g_logger.log_callback_user_data = user_data;
    mtmd_log_set(log_callback, user_data);
}

//
// helper functions
//

size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks) {
    size_t n_tokens = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_tokens += mtmd_input_chunk_get_n_tokens(chunk);
    }
    return n_tokens;
}

llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks) {
    llama_pos n_pos = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        auto chunk = mtmd_input_chunks_get(chunks, i);
        n_pos += mtmd_input_chunk_get_n_pos(chunk);
    }
    return n_pos;
}

void mtmd_helper_image_get_decoder_pos(const mtmd_image_tokens * chunks, llama_pos pos_0, mtmd_decoder_pos * out_pos) {
    size_t n_tokens = mtmd_image_tokens_get_n_tokens(chunks);
    for (size_t i = 0; i < n_tokens; i++) {
        out_pos[i] = mtmd_image_tokens_get_decoder_pos(chunks, pos_0, i);
    }
}

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    std::vector<llama_pos>      pos;
    std::vector<llama_pos>      pos_view; // used by mrope
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd) : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd) {
        GGML_ASSERT(n_tokens > 0 && n_pos_per_embd > 0 && n_mmproj_embd > 0);
        pos     .resize(n_tokens * n_pos_per_embd);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
    }

    void set_position_normal(llama_pos pos_0, llama_seq_id seq_id) {
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for image
    void set_position_mrope_2d(const std::vector<mtmd_decoder_pos> & rel_pos, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        GGML_ASSERT(!rel_pos.empty() && (int32_t)rel_pos.size() == batch.n_tokens);
        seq_id_0[0] = seq_id;
        for (int32_t i = 0; i < batch.n_tokens; i++) {
            pos[i                     ] = rel_pos[i].t;
            pos[i + batch.n_tokens    ] = rel_pos[i].y;
            pos[i + batch.n_tokens * 2] = rel_pos[i].x;
            pos[i + batch.n_tokens * 3] = rel_pos[i].z;
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    // M-RoPE for audio
    void set_position_mrope_1d(llama_pos pos_0, llama_seq_id seq_id) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id_0[0] = seq_id;
        for (int i = 0; i < batch.n_tokens; i++) {
            pos[i                     ] = pos_0 + i;
            pos[i + batch.n_tokens    ] = pos_0 + i;
            pos[i + batch.n_tokens * 2] = pos_0 + i;
            pos[i + batch.n_tokens * 3] = pos_0 + i;
        }
        for (int i = 0; i < batch.n_tokens; i++) {
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }

    llama_batch get_view(int offset, int n_tokens) {
        GGML_ASSERT(offset >= 0 && n_tokens > 0 && offset + n_tokens <= batch.n_tokens);
        llama_pos * pos_ptr;
        pos_view.clear();
        pos_view.reserve(n_tokens * n_pos_per_embd);
        if (n_pos_per_embd > 1) {
            // mrope
            // for example, with layout of src: 1234...1234...1234...1234...
            //       offset 2 will give us dst: 34...34...34...34...
            for (int i = 0; i < n_pos_per_embd; i++) {
                // assume n_tokens is less than or equal to batch.n_tokens
                // batch.n_tokens is number of **total** tokens
                // n_tokens is number of viewed token
                size_t src_idx = i * batch.n_tokens + offset;
                pos_view.insert(pos_view.end(),
                    pos.data() + src_idx,
                    pos.data() + src_idx + n_tokens);
            }
            pos_ptr = pos_view.data();
        } else {
            // normal
            pos_ptr = pos.data() + offset;
        }
        return {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ batch.embd     + offset * n_mmproj_embd,
            /*pos            =*/ pos_ptr,
            /*n_seq_id       =*/ batch.n_seq_id + offset,
            /*seq_id         =*/ batch.seq_id   + offset,
            /*logits         =*/ batch.logits   + offset,
        };
    }
};

// Helper function for decoding an image whose embeddings have already been calculated
int32_t mtmd_helper_decode_image_chunk(
        mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        float * encoded_embd,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        llama_pos * new_n_past) {
    GGML_ASSERT(n_batch > 0);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);
    const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
        return -1;
    }

    const llama_model * model = llama_get_model(lctx);
    int n_mmproj_embd = llama_model_n_embd_inp(model);
    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;

    int32_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
    int32_t i_batch = 0;
    int32_t n_img_batches = (n_tokens + n_batch - 1) / n_batch;
    decode_embd_batch batch_embd(encoded_embd, n_tokens, n_pos_per_embd, n_mmproj_embd);

    if (mtmd_decode_use_mrope(ctx)) {
        if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            const auto image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            if (!image_tokens) {
                LOG_ERR("failed to decode chunk: image tokens are null\n");
                return -1;
            }
            const auto n_tokens = mtmd_image_tokens_get_n_tokens(image_tokens);
            std::vector<mtmd_decoder_pos> rel_pos(n_tokens);
            mtmd_helper_image_get_decoder_pos(image_tokens, n_past, rel_pos.data());
            batch_embd.set_position_mrope_2d(rel_pos, seq_id);
        } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            batch_embd.set_position_mrope_1d(n_past, seq_id);
        } else {
            GGML_ABORT("invalid chunk type for M-RoPE");
        }
    } else {
        batch_embd.set_position_normal(n_past, seq_id);
    }

    const bool use_non_causal = mtmd_decode_use_non_causal(ctx, chunk);
    if (use_non_causal) {
        llama_set_causal_attn(lctx, false);
        // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
    }

    while (i_batch < n_img_batches) { // split into batches
        int pos_offset = i_batch*n_batch;
        int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);
        llama_batch batch_embd_view = batch_embd.get_view(pos_offset, n_tokens_batch);

        LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, i_batch+1, n_img_batches, n_tokens_batch);

        int64_t t1 = ggml_time_ms();
        int32_t ret = llama_decode(lctx, batch_embd_view);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_set_causal_attn(lctx, true); // restore causal attn
            return ret;
        }

        LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, i_batch+1, n_img_batches, ggml_time_ms() - t1);

        i_batch++;
    }

    n_past += mtmd_input_chunk_get_n_pos(chunk);
    *new_n_past = n_past;

    if (use_non_causal) {
        llama_set_causal_attn(lctx, true);
    }
    return 0;
}

int32_t mtmd_helper_eval_chunk_single(mtmd_context * ctx,
        struct llama_context * lctx,
        const mtmd_input_chunk * chunk,
        llama_pos n_past,
        llama_seq_id seq_id,
        int32_t n_batch,
        bool logits_last,
        llama_pos * new_n_past) {
    GGML_ASSERT(n_batch > 0);
    int32_t ret;
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
        size_t i = 0;
        while (i < n_tokens) { // split into batches
            text_batch.n_tokens = 0; // clear the batch
            for (; i < n_tokens && text_batch.n_tokens < n_batch; i++) {
                int32_t j = text_batch.n_tokens;
                text_batch.token   [j]    = tokens[i];
                text_batch.pos     [j]    = n_past++;
                text_batch.n_seq_id[j]    = 1;
                text_batch.seq_id  [j][0] = seq_id;
                text_batch.logits  [j]    = false;

                text_batch.n_tokens++;
            }
            bool is_last_token = (i == n_tokens);
            if (logits_last && is_last_token) {
                text_batch.logits[text_batch.n_tokens - 1] = true;
            }
            ret = llama_decode(lctx, text_batch);
            if (ret != 0) {
                LOG_ERR("failed to decode text\n");
                llama_batch_free(text_batch);
                return ret;
            }
            *new_n_past += text_batch.n_tokens;
        }

    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        const char * name = chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
        int64_t t0 = ggml_time_ms();

        LOG_INF("encoding %s slice...\n", name);

        ret = mtmd_encode_chunk(ctx, chunk);
        if (ret != 0) {
            LOG_ERR("failed to encode %s slice\n", name);
            llama_batch_free(text_batch);
            return ret;
        }

        LOG_INF("%s slice encoded in %" PRId64 " ms\n", name, ggml_time_ms() - t0);

        float * embd = mtmd_get_output_embd(ctx);
        ret = mtmd_helper_decode_image_chunk(ctx, lctx, chunk, embd, n_past, seq_id, n_batch, new_n_past);
        if (ret != 0) {
            LOG_ERR("failed to decode %s\n", name);
            llama_batch_free(text_batch);
            return ret;
        }
    } else {
        GGML_ABORT("chunk type not supported");
    }

    llama_batch_free(text_batch);
    return 0;
}

int32_t mtmd_helper_eval_chunks(mtmd_context * ctx,
                                struct llama_context * lctx,
                                const mtmd_input_chunks * chunks,
                                llama_pos n_past,
                                llama_seq_id seq_id,
                                int32_t n_batch,
                                bool logits_last,
                                llama_pos * new_n_past) {
    size_t n_chunks = mtmd_input_chunks_size(chunks);
    if (n_chunks == 0) {
        LOG_WRN("no chunks to eval\n");
        return 0;
    }

    for (size_t i = 0; i < n_chunks; i++) {
        bool chunk_logits_last = (i == n_chunks - 1) && logits_last;
        auto chunk = mtmd_input_chunks_get(chunks, i);

        int32_t res = mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id, n_batch, chunk_logits_last, &n_past);
        if (res != 0) {
            LOG_ERR("failed to eval chunk %zu\n", i);
            return res;
        }
        *new_n_past = n_past;
    }

    return 0;
}

namespace audio_helpers {

static bool is_audio_file(const char * buf, size_t len) {
    if (len < 12) {
        return false;
    }

    // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
    // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
    bool is_wav = memcmp(buf, "RIFF", 4) == 0 && memcmp(buf + 8, "WAVE", 4) == 0;
    bool is_mp3 = len >= 3 && (
        memcmp(buf, "ID3", 3) == 0 ||
        // Check for MPEG sync word (simplified check)
        ((unsigned char)buf[0] == 0xFF && ((unsigned char)buf[1] & 0xE0) == 0xE0)
    );
    bool is_flac = memcmp(buf, "fLaC", 4) == 0;

    return is_wav || is_mp3 || is_flac;
}

// returns true if the buffer is a valid audio file
static bool decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sampler_rate, std::vector<float> & pcmf32_mono) {
    ma_result result;
    const int channels = 1;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, target_sampler_rate);
    ma_decoder decoder;

    result = ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        return false;
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

    pcmf32_mono.resize(frame_count);
    result = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        return false;
    }

#ifdef MTMD_AUDIO_DEBUG
    // save audio to wav file
    ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, target_sampler_rate);
    ma_encoder encoder;
    ma_encoder_init_file("output.wav", &config, &encoder);
    ma_encoder_write_pcm_frames(&encoder, pcmf32_mono.data(), pcmf32_mono.size(), &frames_read);
    ma_encoder_uninit(&encoder);
#endif

    ma_decoder_uninit(&decoder);
    return true;
}

} // namespace audio_helpers

mtmd_bitmap * mtmd_helper_bitmap_init_from_buf(mtmd_context * ctx, const unsigned char * buf, size_t len) {
    if (audio_helpers::is_audio_file((const char *)buf, len)) {
        std::vector<float> pcmf32;
        const int sample_rate = mtmd_get_audio_sample_rate(ctx);
        if (sample_rate < 0) {
            LOG_ERR("This model does not support audio input\n");
            return nullptr;
        }
        if (!audio_helpers::decode_audio_from_buf(buf, len, sample_rate, pcmf32)) {
            LOG_ERR("Unable to read WAV audio file from buffer\n");
            return nullptr;
        }
        return mtmd_bitmap_init_from_audio(pcmf32.size(), pcmf32.data());
    }

    // otherwise, we assume it's an image
    mtmd_bitmap * result = nullptr;
    {
        int nx, ny, nc;
        auto * data = stbi_load_from_memory(buf, len, &nx, &ny, &nc, 3);
        if (!data) {
            LOG_ERR("%s: failed to decode image bytes\n", __func__);
            return nullptr;
        }
        result = mtmd_bitmap_init(nx, ny, data);
        stbi_image_free(data);
    }
    return result;
}

mtmd_bitmap * mtmd_helper_bitmap_init_from_file(mtmd_context * ctx, const char * fname) {
    std::vector<unsigned char> buf;
    FILE * f = fopen(fname, "rb");
    if (!f) {
        LOG_ERR("Unable to open file %s: %s\n", fname, strerror(errno));
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size < 0) {
        LOG_ERR("Failed to get file size of %s\n", fname);
        fclose(f);
        return nullptr;
    }
    buf.resize(file_size);

    size_t n_read = fread(buf.data(), 1, file_size, f);
    fclose(f);
    if (n_read != (size_t)file_size) {
        LOG_ERR("Failed to read entire file %s", fname);
        return nullptr;
    }

    return mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size());
}


#undef LOG_INF
#undef LOG_WRN
#undef LOG_ERR
