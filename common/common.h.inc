// Various helper functions and utilities

#pragma once

#include "ggml-opt.h.inc"
#include "ggml.h.inc"
#include "llama.h.inc"

#include <nlohmann/json_fwd.hpp.inc>
#include <nlohmann/json.hpp.inc>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <map>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

struct utf8_parse_result {
    uint32_t codepoint;      // Decoded codepoint (only valid if status == SUCCESS)
    size_t bytes_consumed;   // How many bytes this codepoint uses (1-4)
    enum status { SUCCESS, INCOMPLETE, INVALID } status;

    utf8_parse_result(enum status s, uint32_t cp = 0, size_t bytes = 0)
        : codepoint(cp), bytes_consumed(bytes), status(s) {}
};

extern "C" {
struct llama_common_utf8_parse_result {
    uint32_t codepoint;
    size_t   bytes_consumed;
    int      status;
};

struct llama_common_unicode_string {
    uint8_t * data;
    size_t    len;
};

size_t llama_common_utf8_sequence_length(uint8_t first_byte);
bool llama_common_utf8_is_complete(const uint8_t * data, size_t len);
llama_common_utf8_parse_result llama_common_parse_utf8_codepoint(const uint8_t * data, size_t len, size_t offset);
llama_common_unicode_string llama_common_unicode_cpt_to_utf8(uint32_t cpt);
llama_common_unicode_string llama_common_unicode_cpts_to_utf8_rust(const uint32_t * data, size_t len);
void llama_common_unicode_string_free(llama_common_unicode_string value);
const char * llama_common_jinja_token_type_to_string_rust(int type);
bool llama_common_jinja_is_word_rust(uint8_t c);
bool llama_common_jinja_is_integer_rust(uint8_t c);
llama_common_unicode_string llama_common_http_show_masked_url_rust(
        const uint8_t * scheme, size_t scheme_len,
        bool has_user,
        const uint8_t * host, size_t host_len,
        const uint8_t * path, size_t path_len);
llama_common_unicode_string llama_common_peak_source_rust(
        const uint8_t * source, size_t source_len,
        size_t pos,
        size_t max_peak_chars);
llama_common_unicode_string llama_common_fmt_error_with_source_rust(
        const uint8_t * tag, size_t tag_len,
        const uint8_t * msg, size_t msg_len,
        const uint8_t * source, size_t source_len,
        size_t pos);
}

// Determine the expected length of a UTF-8 sequence from its first byte.
// Returns 0 for invalid first bytes.
static inline size_t common_utf8_sequence_length(unsigned char first_byte) {
    return llama_common_utf8_sequence_length(first_byte);
}

// Check if a string ends with a complete UTF-8 sequence.
static inline bool common_utf8_is_complete(const std::string & s) {
    return llama_common_utf8_is_complete(reinterpret_cast<const uint8_t *>(s.data()), s.size());
}

// Parse a single UTF-8 codepoint from input.
static inline utf8_parse_result common_parse_utf8_codepoint(std::string_view input, size_t offset) {
    llama_common_utf8_parse_result result = llama_common_parse_utf8_codepoint(
            reinterpret_cast<const uint8_t *>(input.data()), input.size(), offset);
    return utf8_parse_result(static_cast<enum utf8_parse_result::status>(result.status), result.codepoint, result.bytes_consumed);
}

static inline std::string common_unicode_cpt_to_utf8(uint32_t cpt) {
    llama_common_unicode_string result = llama_common_unicode_cpt_to_utf8(cpt);
    if (result.data == nullptr) {
        throw std::invalid_argument("invalid codepoint");
    }
    std::string value(reinterpret_cast<const char *>(result.data), result.len);
    llama_common_unicode_string_free(result);
    return value;
}

static inline std::string common_unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    llama_common_unicode_string result = llama_common_unicode_cpts_to_utf8_rust(cps.data(), cps.size());
    if (result.data == nullptr) {
        throw std::invalid_argument("invalid codepoint");
    }
    std::string value(reinterpret_cast<const char *>(result.data), result.len);
    llama_common_unicode_string_free(result);
    return value;
}

struct common_params_model;

using common_header      = std::pair<std::string, std::string>;
using common_header_list = std::vector<common_header>;

namespace hf_cache {

struct hf_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string final_path;
    std::string oid;
    std::string repo_id;
    size_t size = 0;
};

using hf_files = std::vector<hf_file>;

hf_files get_repo_files(
    const std::string & repo_id,
    const std::string & token
);

hf_files get_cached_files(const std::string & repo_id = {});
std::string finalize_file(const hf_file & file);
void migrate_old_cache_to_hf_cache(const std::string & token, bool offline = false);

} // namespace hf_cache

struct common_download_progress {
    std::string url;
    size_t downloaded = 0;
    size_t total      = 0;
    bool cached       = false;
};

class common_download_callback {
public:
    virtual ~common_download_callback() = default;
    virtual void on_start(const common_download_progress & p) = 0;
    virtual void on_update(const common_download_progress & p) = 0;
    virtual void on_done(const common_download_progress & p, bool ok) = 0;
    virtual bool is_cancelled() const { return false; }
};

struct common_remote_params {
    common_header_list headers;
    long timeout  = 0;
    long max_size = 0;
};

std::pair<long, std::vector<char>> common_remote_get_content(const std::string & url, const common_remote_params & params);
std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag);

struct common_cached_model_info {
    std::string repo;
    std::string tag;
    std::string to_string() const {
        return repo + ":" + tag;
    }
};

struct common_download_opts {
    std::string bearer_token;
    common_header_list headers;
    bool offline = false;
    common_download_callback * callback = nullptr;
};

struct common_download_model_result {
    std::string model_path;
    std::string mmproj_path;
};

common_download_model_result common_download_model(
    const common_params_model & model,
    const common_download_opts & opts = {},
    bool download_mmproj = false
);

std::vector<common_cached_model_info> common_list_cached_models();
int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const common_download_opts & opts = {},
                                bool skip_etag = false);
std::string common_docker_resolve_model(const std::string & docker);

#define LOG_CLR_TO_EOL  "\033[K\r"
#define LOG_COL_DEFAULT "\033[0m"
#define LOG_COL_BOLD    "\033[1m"
#define LOG_COL_RED     "\033[31m"
#define LOG_COL_GREEN   "\033[32m"
#define LOG_COL_YELLOW  "\033[33m"
#define LOG_COL_BLUE    "\033[34m"
#define LOG_COL_MAGENTA "\033[35m"
#define LOG_COL_CYAN    "\033[36m"
#define LOG_COL_WHITE   "\033[37m"

#ifndef __GNUC__
#    define LOG_ATTRIBUTE_FORMAT(...)
#elif defined(__MINGW32__) && !defined(__clang__)
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#    define LOG_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif

#define LOG_LEVEL_DEBUG  4
#define LOG_LEVEL_INFO   3
#define LOG_LEVEL_WARN   2
#define LOG_LEVEL_ERROR  1
#define LOG_LEVEL_OUTPUT 0

#define LOG_DEFAULT_DEBUG LOG_LEVEL_DEBUG
#define LOG_DEFAULT_LLAMA LOG_LEVEL_INFO

enum log_colors {
    LOG_COLORS_AUTO     = -1,
    LOG_COLORS_DISABLED = 0,
    LOG_COLORS_ENABLED  = 1,
};

int  common_log_get_verbosity_thold(void);
void common_log_set_verbosity_thold(int verbosity);

void common_log_default_callback(enum ggml_log_level level, const char * text, void * user_data);

struct common_log;

struct common_log * common_log_init();
struct common_log * common_log_main();

void common_log_pause (struct common_log * log);
void common_log_resume(struct common_log * log);
void common_log_free  (struct common_log * log);

LOG_ATTRIBUTE_FORMAT(3, 4)
void common_log_add(struct common_log * log, enum ggml_log_level level, const char * fmt, ...);

void common_log_set_file      (struct common_log * log, const char * file);
void common_log_set_colors    (struct common_log * log, log_colors colors);
void common_log_set_prefix    (struct common_log * log, bool prefix);
void common_log_set_timestamps(struct common_log * log, bool timestamps);
void common_log_flush         (struct common_log * log);

#define LOG_TMPL(level, verbosity, ...) \
    do { \
        if ((verbosity) <= common_log_get_verbosity_thold()) { \
            common_log_add(common_log_main(), (level), __VA_ARGS__); \
        } \
    } while (0)

#define LOG(...)             LOG_TMPL(GGML_LOG_LEVEL_NONE, LOG_LEVEL_OUTPUT, __VA_ARGS__)
#define LOGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_NONE, verbosity,        __VA_ARGS__)

#define LOG_DBG(...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, LOG_LEVEL_DEBUG,  __VA_ARGS__)
#define LOG_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  LOG_LEVEL_INFO,   __VA_ARGS__)
#define LOG_WRN(...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  LOG_LEVEL_WARN,   __VA_ARGS__)
#define LOG_ERR(...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, LOG_LEVEL_ERROR,  __VA_ARGS__)
#define LOG_CNT(...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  LOG_LEVEL_INFO,   __VA_ARGS__)

#define LOG_INFV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_INFO,  verbosity, __VA_ARGS__)
#define LOG_WRNV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_WARN,  verbosity, __VA_ARGS__)
#define LOG_ERRV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_ERROR, verbosity, __VA_ARGS__)
#define LOG_DBGV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_DEBUG, verbosity, __VA_ARGS__)
#define LOG_CNTV(verbosity, ...) LOG_TMPL(GGML_LOG_LEVEL_CONT,  verbosity, __VA_ARGS__)

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0,
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1,
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2,
};

enum common_params_fit_status common_fit_params(
                               const char   * path_model,
                struct llama_model_params   * mparams,
                struct llama_context_params * cparams,
                                      float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
                                     size_t * margins,
                                   uint32_t   n_ctx_min,
                        enum ggml_log_level   log_level);

void common_fit_print(
                               const char   * path_model,
                struct llama_model_params   * mparams,
                struct llama_context_params * cparams);

void common_memory_breakdown_print(const struct llama_context * ctx);

std::string json_schema_to_grammar(const nlohmann::ordered_json & schema,
                                   bool force_gbnf = false);

class common_schema_converter;

class common_schema_info {
    std::unique_ptr<common_schema_converter> impl_;

  public:
    common_schema_info();
    ~common_schema_info();

    common_schema_info(const common_schema_info &) = delete;
    common_schema_info & operator=(const common_schema_info &) = delete;
    common_schema_info(common_schema_info &&) noexcept;
    common_schema_info & operator=(common_schema_info &&) noexcept;

    void resolve_refs(nlohmann::ordered_json & schema);
    bool resolves_to_string(const nlohmann::ordered_json & schema);
};

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)> add_rule;
    std::function<std::string(const std::string &, const nlohmann::ordered_json &)> add_schema;
    std::function<void(nlohmann::ordered_json &)> resolve_refs;
};

struct common_grammar_options {
    bool dotall = false;
};

std::string gbnf_format_literal(const std::string & literal);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});

struct llama_model_deleter {
    void operator()(llama_model * model) { llama_model_free(model); }
};

struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};

struct llama_sampler_deleter {
    void operator()(llama_sampler * sampler) { llama_sampler_free(sampler); }
};

struct common_http_url {
    std::string scheme;
    std::string user;
    std::string password;
    std::string host;
    int port;
    std::string path;
};

static inline common_http_url common_http_parse_url(const std::string & url) {
    common_http_url parts;
    auto scheme_end = url.find("://");

    if (scheme_end == std::string::npos) {
        throw std::runtime_error("invalid URL: no scheme");
    }
    parts.scheme = url.substr(0, scheme_end);

    if (parts.scheme != "http" && parts.scheme != "https") {
        throw std::runtime_error("unsupported URL scheme: " + parts.scheme);
    }

    auto rest = url.substr(scheme_end + 3);
    auto at_pos = rest.find('@');

    if (at_pos != std::string::npos) {
        auto auth = rest.substr(0, at_pos);
        auto colon_pos = auth.find(':');
        if (colon_pos != std::string::npos) {
            parts.user = auth.substr(0, colon_pos);
            parts.password = auth.substr(colon_pos + 1);
        } else {
            parts.user = auth;
        }
        rest = rest.substr(at_pos + 1);
    }

    auto slash_pos = rest.find('/');

    if (slash_pos != std::string::npos) {
        parts.host = rest.substr(0, slash_pos);
        parts.path = rest.substr(slash_pos);
    } else {
        parts.host = rest;
        parts.path = "/";
    }

    auto colon_pos = parts.host.find(':');

    if (colon_pos != std::string::npos) {
        parts.port = std::stoi(parts.host.substr(colon_pos + 1));
        parts.host = parts.host.substr(0, colon_pos);
    } else if (parts.scheme == "http") {
        parts.port = 80;
    } else if (parts.scheme == "https") {
        parts.port = 443;
    } else {
        throw std::runtime_error("unsupported URL scheme: " + parts.scheme);
    }

    return parts;
}

static inline std::string common_http_show_masked_url(const common_http_url & parts) {
    llama_common_unicode_string value = llama_common_http_show_masked_url_rust(
            reinterpret_cast<const uint8_t *>(parts.scheme.data()), parts.scheme.size(),
            !parts.user.empty(),
            reinterpret_cast<const uint8_t *>(parts.host.data()), parts.host.size(),
            reinterpret_cast<const uint8_t *>(parts.path.data()), parts.path.size());
    std::string result(reinterpret_cast<const char *>(value.data), value.len);
    llama_common_unicode_string_free(value);
    return result;
}

struct llama_adapter_lora_deleter {
    void operator()(llama_adapter_lora * adapter) { llama_adapter_lora_free(adapter); }
};

using llama_model_ptr        = std::unique_ptr<llama_model, llama_model_deleter>;
using llama_context_ptr      = std::unique_ptr<llama_context, llama_context_deleter>;
using llama_sampler_ptr      = std::unique_ptr<llama_sampler, llama_sampler_deleter>;
using llama_adapter_lora_ptr = std::unique_ptr<llama_adapter_lora, llama_adapter_lora_deleter>;

struct common_time_meas {
    common_time_meas(int64_t & t_acc, bool disable = false);
    ~common_time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

struct common_adapter_lora_info {
    std::string path;
    float scale;

    std::string task_name;
    std::string prompt_prefix;

    struct llama_adapter_lora * ptr;
};

using llama_tokens = std::vector<llama_token>;

struct common_control_vector_load_info;

//
// CPU utils
//

struct common_cpu_params {
    int      n_threads                   = -1;
    bool     cpumask[GGML_MAX_N_THREADS] = {false}; // CPU affinity mask.
    bool     mask_valid                  = false;   // Default: any CPU
    enum ggml_sched_priority  priority   = GGML_SCHED_PRIO_NORMAL;  // Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
    bool     strict_cpu                  = false;   // Use strict CPU placement
    uint32_t poll                        = 50;      // Polling (busywait) level (0 - no polling, 100 - mostly polling)
};

int32_t common_cpu_get_num_physical_cores();
int32_t common_cpu_get_num_math();

//
// Common params
//

enum llama_example {
    LLAMA_EXAMPLE_BATCHED,
    LLAMA_EXAMPLE_DEBUG,
    LLAMA_EXAMPLE_COMMON,
    LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_COMPLETION,
    LLAMA_EXAMPLE_CLI,
    LLAMA_EXAMPLE_EMBEDDING,
    LLAMA_EXAMPLE_PERPLEXITY,
    LLAMA_EXAMPLE_RETRIEVAL,
    LLAMA_EXAMPLE_PASSKEY,
    LLAMA_EXAMPLE_IMATRIX,
    LLAMA_EXAMPLE_BENCH,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CVECTOR_GENERATOR,
    LLAMA_EXAMPLE_EXPORT_LORA,
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_LOOKUP,
    LLAMA_EXAMPLE_PARALLEL,
    LLAMA_EXAMPLE_TTS,
    LLAMA_EXAMPLE_DIFFUSION,
    LLAMA_EXAMPLE_FINETUNE,
    LLAMA_EXAMPLE_FIT_PARAMS,
    LLAMA_EXAMPLE_RESULTS,
    LLAMA_EXAMPLE_EXPORT_GRAPH_OPS,

    LLAMA_EXAMPLE_COUNT,
};

enum common_sampler_type {
    COMMON_SAMPLER_TYPE_NONE        = 0,
    COMMON_SAMPLER_TYPE_DRY         = 1,
    COMMON_SAMPLER_TYPE_TOP_K       = 2,
    COMMON_SAMPLER_TYPE_TOP_P       = 3,
    COMMON_SAMPLER_TYPE_MIN_P       = 4,
  //COMMON_SAMPLER_TYPE_TFS_Z       = 5,
    COMMON_SAMPLER_TYPE_TYPICAL_P   = 6,
    COMMON_SAMPLER_TYPE_TEMPERATURE = 7,
    COMMON_SAMPLER_TYPE_XTC         = 8,
    COMMON_SAMPLER_TYPE_INFILL      = 9,
    COMMON_SAMPLER_TYPE_PENALTIES   = 10,
    COMMON_SAMPLER_TYPE_TOP_N_SIGMA = 11,
    COMMON_SAMPLER_TYPE_ADAPTIVE_P  = 12,
};

// dimensionality reduction methods, used by cvector-generator
enum dimre_method {
    DIMRE_METHOD_PCA,
    DIMRE_METHOD_MEAN,
};

enum common_conversation_mode {
    COMMON_CONVERSATION_MODE_DISABLED = 0,
    COMMON_CONVERSATION_MODE_ENABLED  = 1,
    COMMON_CONVERSATION_MODE_AUTO     = 2,
};

enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    llama_token token = LLAMA_TOKEN_NULL;
};

enum common_params_sampling_config : uint64_t {
    COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS        = 1 << 0,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_K           = 1 << 1,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_P           = 1 << 2,
    COMMON_PARAMS_SAMPLING_CONFIG_MIN_P           = 1 << 3,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY = 1 << 4,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD   = 1 << 5,
    COMMON_PARAMS_SAMPLING_CONFIG_TEMP            = 1 << 6,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N  = 1 << 7,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT  = 1 << 8,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT        = 1 << 9,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU    = 1 << 10,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA    = 1 << 11,
};

enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,          // no speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT,         // draft model
    COMMON_SPECULATIVE_TYPE_EAGLE3,        // eagle draft model
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,  // simple self-speculative decoding
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,   // self-speculative decoding with n-gram keys only
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, // self-speculative decoding with n-gram keys and 4 m-gram values
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // self-speculative decoding with 3-level n-gram cache
    COMMON_SPECULATIVE_TYPE_COUNT          // number of types, unknown type
};

// Grammar type enumeration
enum common_grammar_type {
    COMMON_GRAMMAR_TYPE_NONE,           // no grammar set
    COMMON_GRAMMAR_TYPE_USER,           // user-provided GBNF (--grammar / "grammar" API field)
    COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,  // auto-generated from JSON schema (--json-schema / "json_schema" API field)
    COMMON_GRAMMAR_TYPE_TOOL_CALLS,     // auto-generated by chat template parser for function calling
};

// Grammar variant struct with type and grammar string
struct common_grammar {
    common_grammar_type type = COMMON_GRAMMAR_TYPE_NONE;
    std::string grammar;

    // Default constructor - no grammar
    common_grammar() = default;

    // Constructor with type and grammar string
    common_grammar(common_grammar_type t, std::string g) : type(t), grammar(std::move(g)) {
        GGML_ASSERT(type != COMMON_GRAMMAR_TYPE_NONE || !grammar.empty());
    }

    // Check if a grammar is set
    bool empty() const { return type == COMMON_GRAMMAR_TYPE_NONE || grammar.empty(); }
};

// Returns the raw grammar string, or empty string if no grammar is set.
inline const std::string & common_grammar_value(const common_grammar & g) {
    return g.grammar;
}

// Returns true when the generation_prompt should be prefilled into the grammar sampler.
// Only output-format and tool-call grammars need prefill; user-supplied grammars must not be prefilled.
inline bool common_grammar_needs_prefill(const common_grammar & g) {
    return g.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT
        || g.type == COMMON_GRAMMAR_TYPE_TOOL_CALLS;
}

// sampling parameters
struct common_params_sampling {
    uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler

    int32_t n_prev             = 64;     // number of previous tokens to remember
    int32_t n_probs            = 0;      // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep           = 0;      // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t top_k              = 40;     // <= 0 to use vocab size
    float   top_p              = 0.95f;  // 1.0 = disabled
    float   min_p              = 0.05f;  // 0.0 = disabled
    float   xtc_probability    = 0.00f;  // 0.0 = disabled
    float   xtc_threshold      = 0.10f;  // > 0.5 disables XTC
    float   typ_p              = 1.00f;  // typical_p, 1.0 = disabled
    float   temp               = 0.80f;  // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   dynatemp_range     = 0.00f;  // 0.0 = disabled
    float   dynatemp_exponent  = 1.00f;  // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penalty_last_n     = 64;     // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penalty_repeat     = 1.00f;  // 1.0 = disabled
    float   penalty_freq       = 0.00f;  // 0.0 = disabled
    float   penalty_present    = 0.00f;  // 0.0 = disabled
    float   dry_multiplier     = 0.0f;   // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    float   dry_base           = 1.75f;  // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    int32_t dry_allowed_length = 2;      // tokens extending repetitions beyond this receive penalty
    int32_t dry_penalty_last_n = -1;     // how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
    float   adaptive_target    = -1.0f;  // select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    float   adaptive_decay     = 0.90f;  // EMA decay for adaptation; history ≈ 1/(1-decay) tokens (0.0 - 0.99)
    int32_t mirostat           = 0;      // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   top_n_sigma        = -1.00f; // -1.0 = disabled
    float   mirostat_tau       = 5.00f;  // target entropy
    float   mirostat_eta       = 0.10f;  // learning rate
    bool    ignore_eos         = false;
    bool    no_perf            = false;  // disable performance metrics
    bool    timing_per_token   = false;

    uint64_t user_sampling_config = 0; // bitfield to track user-specified samplers

    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};     // default sequence breakers for DRY

    std::vector<enum common_sampler_type> samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };

    common_grammar              grammar;      // optional grammar constraint (user / output-format / tool-calls)
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers; // optional triggers (for lazy grammars)
    std::set<llama_token>               preserved_tokens;

    std::vector<llama_logit_bias> logit_bias;     // logit biases to apply
    std::vector<llama_logit_bias> logit_bias_eog; // pre-calculated logit biases for EOG tokens

    // The assistant generation prompt already prefilled into the prompt.
    // Fed to the grammar sampler (to advance past pre-existing tokens) and used
    // to determine the reasoning budget sampler's initial state.
    // Only applied when the grammar is of output-format or tool-calls type.
    std::string generation_prompt;

    // reasoning budget sampler parameters
    // these are populated by the server/CLI based on chat template params
    int32_t                  reasoning_budget_tokens   = -1;   // -1 = disabled, >= 0 = token budget
    std::vector<llama_token> reasoning_budget_start;           // start tag token sequence
    std::vector<llama_token> reasoning_budget_end;             // end tag token sequence
    std::vector<llama_token> reasoning_budget_forced;          // forced sequence (message + end tag)
    std::string              reasoning_budget_message;         // message injected before end tag when budget exhausted

    bool backend_sampling = false;

    bool has_logit_bias() const {
        return !logit_bias.empty();
    }

    // print the parameters into a string
    std::string print() const;
};

struct common_params_model {
    std::string path        = ""; // model local path                                       // NOLINT
    std::string url         = ""; // model url to download                                  // NOLINT
    std::string hf_repo     = ""; // HF repo                                                // NOLINT
    std::string hf_file     = ""; // HF file                                                // NOLINT
    std::string docker_repo = ""; // Docker repo                                            // NOLINT
    std::string name        = ""; // in format <user>/<model>[:<tag>] (tag is optional)     // NOLINT
};

struct common_ngram_mod;

// draft-model-based speculative decoding parameters
struct common_params_speculative_draft {
    int32_t n_max = 16; // maximum number of tokens to draft during speculative decoding
    int32_t n_min = 0;  // minimum number of draft tokens to use for speculative decoding

    float p_split = 0.1f;  // speculative decoding split probability
    float p_min   = 0.75f; // minimum speculative decoding probability (greedy)

    common_params_model mparams;

    llama_model * model = nullptr; // a llama_model that can be shared by multiple speculative contexts

    llama_context_params cparams; // these are the parameters for the draft llama_context

    int32_t n_ctx        = 0;  // draft context size
    int32_t n_gpu_layers = -1; // number of layers to store in VRAM for the draft model (-1 - use default)

    ggml_type cache_type_k = GGML_TYPE_F16; // KV cache data type for the K
    ggml_type cache_type_v = GGML_TYPE_F16; // KV cache data type for the V

    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    std::vector<ggml_backend_dev_t> devices; // devices to use for offloading

    std::vector<std::pair<std::string, std::string>> replacements; // main to speculative model replacements
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
};

struct common_params_speculative_ngram_mod {
    int32_t n_match = 24;

    int32_t n_max = 64;
    int32_t n_min = 48;

    // shared instance of the ngram container for all speculative decoding contexts
    std::shared_ptr<common_ngram_mod> obj;
};

struct common_params_speculative_ngram_map {
    uint16_t size_n   = 12; // ngram size for lookup
    uint16_t size_m   = 48; // mgram size for speculative tokens
    uint16_t min_hits = 1;  // minimum hits at ngram/mgram lookup for mgram to be proposed
};

struct common_params_speculative_ngram_cache {
    std::string lookup_cache_static;  // path of static ngram cache file for lookup decoding
    std::string lookup_cache_dynamic; // path of dynamic ngram cache file for lookup decoding
};

struct common_params_speculative {
    // TODO: become a vector in order to support "chains of speculators"
    common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;

    common_params_speculative_draft draft;

    common_params_speculative_ngram_mod ngram_mod;
    common_params_speculative_ngram_map ngram_simple;
    common_params_speculative_ngram_map ngram_map_k;
    common_params_speculative_ngram_map ngram_map_k4v;

    common_params_speculative_ngram_cache ngram_cache;

    bool has_dft() const {
        return !draft.mparams.path.empty() || !draft.mparams.hf_repo.empty();
    }
};

struct common_speculative;

std::string common_speculative_type_name_str();
enum common_speculative_type common_speculative_type_from_name(const std::string & name);
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params);
int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params);

void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_params_vocoder {
    struct common_params_model model;

    std::string speaker_file; // speaker file path

    bool use_guide_tokens = false; // enable guide tokens to improve TTS accuracy
};

struct common_params_diffusion {
    int32_t steps         = 128;
    bool    visual_mode   = false;

    float   eps           = 0;        // epsilon for timesteps
    int32_t block_length  = 0;        // block length for generation

    int32_t algorithm     = 4;        // default algorithm: low-confidence
    float   alg_temp      = 0.0f;     // algorithm temperature

    float   cfg_scale     = 0;        // classifier-free guidance scale
    bool    add_gumbel_noise = false; // add gumbel noise to the logits if temp > 0.0
};

// reasoning API response format (not to be confused as chat template's reasoning format)
// only used by server
enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_AUTO,            // Same as deepseek, using `message.reasoning_content`
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY, // Extract thinking tag contents and return as `message.reasoning_content`, or leave inline in <think> tags in stream mode
    COMMON_REASONING_FORMAT_DEEPSEEK,        // Extract thinking tag contents and return as `message.reasoning_content`, including in streaming deltas.
    // do not extend this enum unless you absolutely have to
    // in most cases, use COMMON_REASONING_FORMAT_AUTO
    // see: https://github.com/ggml-org/llama.cpp/pull/15408
};


struct lr_opt {
    float    lr0          = 1e-5; // learning rate at first epoch
    float    lr_min       = -1;
    float    decay_epochs = -1;   // if >0, the learning rate starts at lr0 and decays to lr_min after this many epochs
    float    scale_epoch  = 0;
    float    wd           = 0;
    unsigned epochs       = 2;

    unsigned epoch; // set by optimizer outer (epochs) loop
    // learning rate decay - constant LR per epoch only for now
    float get_lr(float e) const;
    float get_lr() const { return get_lr(epoch); }
    // must call after arg parse, before get_lr
    void init();
};

struct ggml_opt_optimizer_params common_opt_lr_pars(void * userdata);

struct common_params {
    int32_t n_predict             =    -1; // max. number of new tokens to predict, -1 == no limit
    int32_t n_ctx                 =     0; // context size, 0 == context the model was trained with
    int32_t n_batch               =  2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_ubatch              =   512; // physical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                =     0; // number of tokens to keep from initial prompt
    int32_t n_chunks              =    -1; // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel            =     1; // number of parallel sequences to decode
    int32_t n_sequences           =     1; // number of sequences to decode
    int32_t grp_attn_n            =     1; // group-attention factor
    int32_t grp_attn_w            =   512; // group-attention width
    int32_t n_print               =    -1; // print token count every n tokens (-1 = disabled)
    float   rope_freq_base        =  0.0f; // RoPE base frequency
    float   rope_freq_scale       =  0.0f; // RoPE frequency scaling factor
    float   yarn_ext_factor       = -1.0f; // YaRN extrapolation mix factor
    float   yarn_attn_factor      = -1.0f; // YaRN magnitude scaling factor
    float   yarn_beta_fast        = -1.0f; // YaRN low correction dim
    float   yarn_beta_slow        = -1.0f; // YaRN high correction dim
    int32_t yarn_orig_ctx         =     0; // YaRN original context length

    // offload params
    std::vector<ggml_backend_dev_t> devices; // devices to use for offloading

    int32_t n_gpu_layers       = -1;    // number of layers to store in VRAM, -1 is auto, <= -2 is all
    int32_t main_gpu           = 0;     // the GPU that is used for scratch and small tensors
    float   tensor_split[128]  = {0};   // how split tensors should be distributed across GPUs
    bool    fit_params         = true;  // whether to fit unset model/context parameters to free device memory
    bool    fit_params_print   = false; // print the estimated required memory to run the model
    int32_t fit_params_min_ctx = 4096;  // minimum context size to set when trying to reduce memory use

    // margin per device in bytes for fitting parameters to free memory:
    std::vector<size_t> fit_params_target = std::vector<size_t>(llama_max_devices(), 1024 * 1024*1024);

    enum llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER; // how to split the model across GPUs

    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data                 = nullptr;

    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

    enum llama_rope_scaling_type rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    enum llama_pooling_type      pooling_type      = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling type for embeddings
    enum llama_attention_type    attention_type    = LLAMA_ATTENTION_TYPE_UNSPECIFIED; // attention type for embeddings
    enum llama_flash_attn_type   flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO; // whether to use Flash Attention

    struct common_params_sampling    sampling;
    struct common_params_speculative speculative;
    struct common_params_vocoder     vocoder;
    struct common_params_diffusion   diffusion;

    struct common_params_model model;

    std::set<std::string> model_alias;     // model aliases                                                 // NOLINT
    std::set<std::string> model_tags;      // model tags (informational, not used for routing)              // NOLINT
    std::string hf_token             = ""; // HF token                                                      // NOLINT
    std::string prompt               = "";                                                                  // NOLINT
    std::string system_prompt        = "";                                                                  // NOLINT
    std::string prompt_file          = ""; // store the external prompt file name                           // NOLINT
    std::string path_prompt_cache    = ""; // path to file for saving/loading prompt eval state             // NOLINT
    std::string input_prefix         = ""; // string to prefix user inputs with                             // NOLINT
    std::string input_suffix         = ""; // string to suffix user inputs with                             // NOLINT
    std::string logits_file          = ""; // file for saving *all* logits                                  // NOLINT

    // llama-debug specific options
    std::string logits_output_dir = "data"; // directory for saving logits output files                     // NOLINT
    bool        save_logits       = false;  // whether to save logits to files                              // NOLINT
    std::vector<std::string> tensor_filter; // filter tensor names for debug output (regex)                 // NOLINT

    std::vector<std::string> in_files;   // all input files
    std::vector<std::string> antiprompt; // strings upon which more user input is prompted (a.k.a. reverse prompts)
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

    bool lora_init_without_apply = false; // only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_adapter_lora_apply)
    std::vector<common_adapter_lora_info> lora_adapters; // lora adapter path with user defined scale

    std::vector<common_control_vector_load_info> control_vectors; // control vector with user defined scale

    int32_t verbosity                  = 3;  // LOG_LEVEL_INFO
    int32_t control_vector_layer_start = -1; // layer range for control vector
    int32_t control_vector_layer_end   = -1; // layer range for control vector
    bool    offline                    = false;

    int32_t ppl_stride      = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int32_t ppl_output_type = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                     //                                       (which is more convenient to use for plotting)
                                     //
    bool   hellaswag        = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks  = 400;   // number of tasks to use when computing the HellaSwag score

    bool   winogrande       = false; // compute Winogrande score over random tasks from datafile supplied in prompt
    size_t winogrande_tasks = 0;     // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

    bool   multiple_choice  = false;  // compute TruthfulQA score over random tasks from datafile supplied in prompt
    size_t multiple_choice_tasks = 0; // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

    bool   kl_divergence    = false; // compute KL divergence

    bool check             = false; // check rather than generate results for llama-results

    bool usage             = false; // print usage
    bool completion        = false; // print source-able completion script
    bool use_color         = false; // use color to distinguish generations and inputs
    bool special           = false; // enable special token output
    bool interactive       = false; // interactive mode
    bool interactive_first = false; // wait for user input immediately
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool escape            = true;  // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles
    bool cont_batching     = true;  // insert new sequences for decoding on-the-fly
    bool no_perf           = false; // disable performance metrics
    bool show_timings      = true;  // show timing information on CLI
    bool ctx_shift         = false; // context shift on infinite text generation
    bool swa_full          = false; // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
    bool kv_unified        = false; // enable unified KV cache

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool use_mmap          = true;  // enable mmap to use filesystem cache
    bool use_direct_io     = false; // read from disk without buffering
    bool use_mlock         = false; // use mlock to keep model in memory
    bool verbose_prompt    = false; // print prompt tokens before generation
    bool display_prompt    = true;  // print prompt before generation
    bool no_kv_offload     = false; // disable KV offloading
    bool warmup            = true;  // warmup run
    bool check_tensors     = false; // validate tensor data
    bool no_op_offload     = false; // globally disable offload host tensor operations to device
    bool no_extra_bufts    = false; // disable extra buffer types (used for weight repacking)
    bool no_host           = false; // bypass host buffer allowing extra buffers to be used

    bool single_turn       = false; // single turn chat conversation

    ggml_type cache_type_k = GGML_TYPE_F16; // KV cache data type for the K
    ggml_type cache_type_v = GGML_TYPE_F16; // KV cache data type for the V

    common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_AUTO;

    // multimodal models (see tools/mtmd)
    struct common_params_model mmproj;
    bool mmproj_use_gpu = true;     // use GPU for multimodal model
    bool no_mmproj = false;         // explicitly disable multimodal model
    std::vector<std::string> image; // path to image file(s)
    int image_min_tokens = -1;
    int image_max_tokens = -1;

    // finetune
    struct lr_opt lr;
    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    float val_split = 0.05f; // fraction of the data used for the validation set

    // embedding
    bool embedding         = false; // get only sentence embedding
    int32_t embd_normalize = 2;     // normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
    std::string embd_out   = "";    // empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
    std::string embd_sep   = "\n";  // separator of embeddings
    std::string cls_sep    = "\t";  // separator of classification sequences

    // server params
    int32_t port                = 8080;          // server listens on this network port
    bool    reuse_port          = false;         // allow multiple sockets to bind to the same port
    int32_t timeout_read        = 600;           // http read timeout in seconds
    int32_t timeout_write       = timeout_read;  // http write timeout in seconds
    int32_t n_threads_http      = -1;    // number of threads to process HTTP requests (TODO: support threadpool)
    int32_t n_cache_reuse       = 0;     // min chunk size to reuse from the cache via KV shifting
    bool    cache_prompt        = true;  // whether to enable prompt caching
    bool    cache_idle_slots    = true;  // save and clear idle slots upon starting a new task
    int32_t n_ctx_checkpoints   = 32;    // max number of context checkpoints per slot
    int32_t checkpoint_every_nt = 8192;  // make a checkpoint every n tokens during prefill
    int32_t cache_ram_mib       = 8192;  // -1 = no limit, 0 - disable, 1 = 1 MiB, etc.

    std::string hostname      = "127.0.0.1";
    std::string public_path   = "";                                                                         // NOLINT
    std::string api_prefix    = "";                                                                         // NOLINT
    std::string chat_template = "";                                                                         // NOLINT
    bool use_jinja = true;                                                                                  // NOLINT
    bool enable_chat_template = true;
    bool force_pure_content_parser = false;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    int enable_reasoning = -1; // -1 = auto, 0 = disable, 1 = enable
    bool prefill_assistant = true; // if true, any trailing assistant message will be prefilled into the response
    int sleep_idle_seconds = -1;   // if >0, server will sleep after this many seconds of idle time

    std::vector<std::string> api_keys;

    std::string ssl_file_key  = "";                                                                         // NOLINT
    std::string ssl_file_cert = "";                                                                         // NOLINT

    std::map<std::string, std::string> default_template_kwargs;

    // webui configs
    bool webui = true;
    bool webui_mcp_proxy = false;
    std::string webui_config_json;

    // "advanced" endpoints are disabled by default for better security
    bool endpoint_slots   = true;
    bool endpoint_props   = false; // only control POST requests, not GET
    bool endpoint_metrics = false;

    // enable built-in tools
    std::vector<std::string> server_tools;

    // router server configs
    std::string models_dir    = ""; // directory containing models for the router server
    std::string models_preset = ""; // directory containing model presets for the router server
    int models_max = 4;             // maximum number of models to load simultaneously
    bool models_autoload = true;    // automatically load models when requested via the router server

    bool log_json = false;

    std::string slot_save_path;
    std::string media_path; // path to directory for loading media files

    float slot_prompt_similarity = 0.1f;

    // batched-bench params
    bool is_pp_shared   = false;
    bool is_tg_separate = false;

    std::vector<int32_t> n_pp;
    std::vector<int32_t> n_tg;
    std::vector<int32_t> n_pl;

    // retrieval params
    std::vector<std::string> context_files; // context files to embed

    int32_t chunk_size = 64; // chunk size for context embedding

    std::string chunk_separator = "\n"; // chunk separator for context embedding

    // passkey params
    int32_t n_junk = 250; // number of times to repeat the junk text
    int32_t i_pos  = -1;  // position of the passkey in the junk text

    // imatrix params
    int32_t n_out_freq  = 10; // output the imatrix every n_out_freq iterations
    int32_t n_save_freq =  0; // save the imatrix every n_save_freq iterations
    int32_t i_chunk     =  0; // start processing from this chunk
    int8_t  imat_dat    =  0; // whether the legacy imatrix.dat format should be output (gguf <= 0 < dat)

    bool process_output  = false; // collect data for the output tensor
    bool compute_ppl     = true;  // whether to compute perplexity
    bool show_statistics = false; // show imatrix statistics per tensor
    bool parse_special   = false; // whether to parse special tokens during imatrix tokenization

    // cvector-generator params
    int n_pca_batch = 100;
    int n_pca_iterations = 1000;
    dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    std::string cvector_positive_file = "tools/cvector-generator/positive.txt";
    std::string cvector_negative_file = "tools/cvector-generator/negative.txt";

    bool spm_infill = false; // suffix/prefix/middle pattern for infill

    // batched-bench params
    bool batched_bench_output_jsonl = false;

    // common params
    std::string out_file; // output filename for all example programs
    // optional callback for model loading progress and cancellation:
    // called with a progress value between 0.0 and 1.0.
    // return false from callback to abort model loading or true to continue
    llama_progress_callback load_progress_callback = NULL;
    void *                  load_progress_callback_user_data = NULL;
    bool no_alloc = false; // Don't allocate model buffers
};

// call once at the start of a program if it uses libcommon
// initializes the logging system and prints info about the build
void common_init();

std::string common_params_get_system_info(const common_params & params);

bool parse_cpu_range(const std::string & range, bool(&boolmask)[GGML_MAX_N_THREADS]);
bool parse_cpu_mask(const std::string & mask, bool(&boolmask)[GGML_MAX_N_THREADS]);
void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model = nullptr);
bool set_process_priority(enum ggml_sched_priority prio);

//
// String utils
//

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
std::string string_format(const char * fmt, ...);

std::string string_strip(const std::string & str);
std::string string_get_sortable_timestamp();

std::string string_join(const std::vector<std::string> & values, const std::string & separator);
std::vector<std::string> string_split(const std::string & str, const std::string & delimiter);
std::string string_repeat(const std::string & str, size_t n);

void string_replace_all(std::string & s, const std::string & search, const std::string & replace);

std::string regex_escape(const std::string & s);

template<class T>
static std::vector<T> string_split(const std::string & str, char delim) {
    static_assert(!std::is_same<T, std::string>::value, "Please use the specialized version for std::string");
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        token_stream >> value;
        values.push_back(value);
    }
    return values;
}

template<>
inline std::vector<std::string> string_split<std::string>(const std::string & str, char delim)
{
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t delim_pos = str.find(delim);
    while (delim_pos != std::string::npos) {
        std::string part = str.substr(begin_pos, delim_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = delim_pos + 1;
        delim_pos = str.find(delim, begin_pos);
    }
    parts.emplace_back(str.substr(begin_pos));
    return parts;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, char prefix) {
    return !str.empty() && str.front() == prefix;
}

// remove when moving to c++20
inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool string_remove_suffix(std::string & str, std::string_view suffix) {
    if (string_ends_with(str, suffix)) {
        str.resize(str.size() - suffix.size());
        return true;
    }
    return false;
}

inline size_t string_find_partial_stop(std::string_view str, std::string_view stop) {
    if (!str.empty() && !stop.empty()) {
        const size_t max_len = std::min(str.size(), stop.size());
        const char last_char = str.back();
        for (size_t len = max_len; len > 0; --len) {
            if (stop[len - 1] == last_char) {
                if (string_ends_with(str, stop.substr(0, len))) {
                    return str.size() - len;
                }
            }
        }
    }
    return std::string::npos;
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides);
void string_process_escapes(std::string & input);

std::string string_from(bool value);
std::string string_from(const std::vector<int> & values);
std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens);
std::string string_from(const struct llama_context * ctx, const struct llama_batch & batch);

bool glob_match(const std::string & pattern, const std::string & str);

//
// Filesystem utils
//

bool fs_validate_filename(const std::string & filename, bool allow_subdirs = false);
bool fs_create_directory_with_parents(const std::string & path);
bool fs_is_directory(const std::string & path);

std::string fs_get_cache_directory();
std::string fs_get_cache_file(const std::string & filename);

struct common_file_info {
    std::string path;
    std::string name;
    size_t      size = 0; // in bytes
    bool        is_dir = false;
};
std::vector<common_file_info> fs_list(const std::string & path, bool include_directories);

//
// TTY utils
//

// Auto-detect if colors can be enabled based on terminal and environment
bool tty_can_use_colors();

//
// Model utils
//

struct common_sampler;

// note: defines the model, context, samplers, ets. lifetimes
struct common_init_result {
    common_init_result(common_params & params);
    ~common_init_result();

    llama_model * model();
    llama_context * context();

    common_sampler * sampler(llama_seq_id seq_id);
    void reset_samplers();

    std::vector<llama_adapter_lora_ptr> & lora();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_init_result_ptr = std::unique_ptr<common_init_result>;

common_init_result_ptr common_init_from_params(common_params & params);

struct llama_model_params     common_model_params_to_llama  (      common_params & params);
struct llama_context_params   common_context_params_to_llama(const common_params & params);
struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params);

// clear LoRA adapters from context, then apply new list of adapters
void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora);

// model endpoint from env
std::string common_get_model_endpoint();

//
// Context utils
//

enum common_context_seq_rm_type {
    COMMON_CONTEXT_SEQ_RM_TYPE_NO   = 0, // seq_rm not supported (e.g. no memory module)
    COMMON_CONTEXT_SEQ_RM_TYPE_PART = 1, // can seq_rm partial sequences
    COMMON_CONTEXT_SEQ_RM_TYPE_FULL = 2, // can seq_rm full sequences only
};

// check if the llama_context can remove sequences
// note: clears the memory of the context
common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx);


//
// Batch utils
//

void common_batch_clear(struct llama_batch & batch);

void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

// decodes a single batch of tokens for a prompt and manages session tokens
//
// Note: We save state before the last token so that we can replay it to ensure
// compatibility with all memory types. Recurrent/hybrid models cannot remove
// tokens from memory, so this approach works across all model architectures.
bool common_prompt_batch_decode(
              struct llama_context * ctx,
    const std::vector<llama_token> & embd,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state);

// replays the last token after loading state to regenerate logits
// used after loading session state to ensure the sampling context has valid logits
bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
std::string common_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token,
                       bool          special = true);

std::string common_token_to_piece(
          const struct llama_vocab * vocab,
                       llama_token   token,
                       bool          special = true);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// optionally renders special/control tokens
std::string common_detokenize(
            const struct llama_context * ctx,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

std::string common_detokenize(
              const struct llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

//
// Embedding utils
//

// TODO: replace embd_norm with an enum
void common_embd_normalize(const float * inp, float * out, int n, int embd_norm);

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n);

//
// Control vector utils
//

struct common_control_vector_data {
    int n_embd;

    // stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
    std::vector<float> data;
};

struct common_control_vector_load_info {
    float strength;

    std::string fname;
};

// Load control vectors, scale each by strength, and add them together.
// On error, returns {-1, empty}
common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos);

//
// Split utils
//

namespace {

const char * const LLM_KV_SPLIT_NO            = "split.no";
const char * const LLM_KV_SPLIT_COUNT         = "split.count";
const char * const LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count";

}

//
// MoE utils
//

const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

inline std::string llm_ffn_exps_block_regex(int idx) {
    return string_format("blk\\.%d%s", idx, LLM_FFN_EXPS_REGEX);
}

inline llama_model_tensor_buft_override llm_ffn_exps_cpu_override() {
    return { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() };
}

//
// training utils
//

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride);

// "adamw" or "sgd" (case insensitive)
enum ggml_opt_optimizer_type common_opt_get_optimizer(const char *);

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(const struct llama_model * model, struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;

// pseudo-env variable to identify preset-only arguments
#define COMMON_ARG_PRESET_LOAD_ON_STARTUP "__PRESET_LOAD_ON_STARTUP"
#define COMMON_ARG_PRESET_STOP_TIMEOUT    "__PRESET_STOP_TIMEOUT"

//
// CLI argument parsing
//

struct common_arg {
    std::set<enum llama_example> examples = {LLAMA_EXAMPLE_COMMON};
    std::set<enum llama_example> excludes = {};
    std::vector<const char *> args;
    std::vector<const char *> args_neg;  // for negated args like --no-xxx
    const char * value_hint   = nullptr; // help text or example for arg value
    const char * value_hint_2 = nullptr; // for second arg value
    const char * env          = nullptr;
    std::string help;
    bool is_sampling = false; // is current arg a sampling param?
    bool is_spec = false; // is current arg a speculative decoding param?
    bool is_preset_only = false; // is current arg preset-only (not treated as CLI arg)
    void (*handler_void)   (common_params & params) = nullptr;
    void (*handler_string) (common_params & params, const std::string &) = nullptr;
    void (*handler_str_str)(common_params & params, const std::string &, const std::string &) = nullptr;
    void (*handler_int)    (common_params & params, int) = nullptr;
    void (*handler_bool)   (common_params & params, bool) = nullptr;

    common_arg() = default;

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &)
    ) : args(args), value_hint(value_hint), help(help), handler_string(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, int)
    ) : args(args), value_hint(value_hint), help(help), handler_int(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::string & help,
        void (*handler)(common_params & params)
    ) : args(args), help(help), handler_void(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::initializer_list<const char *> & args_neg,
        const std::string & help,
        void (*handler)(common_params & params, bool)
    ) : args(args), args_neg(args_neg), help(help), handler_bool(handler) {}

    // support 2 values for arg
    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const char * value_hint_2,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &, const std::string &)
    ) : args(args), value_hint(value_hint), value_hint_2(value_hint_2), help(help), handler_str_str(handler) {}

    common_arg & set_examples(std::initializer_list<enum llama_example> examples);
    common_arg & set_excludes(std::initializer_list<enum llama_example> excludes);
    common_arg & set_env(const char * env);
    common_arg & set_sampling();
    common_arg & set_spec();
    common_arg & set_preset_only();
    bool in_example(enum llama_example ex);
    bool is_exclude(enum llama_example ex);
    bool get_value_from_env(std::string & output) const;
    bool has_value_from_env() const;
    std::string to_string() const;

    // for using as key in std::map
    bool operator<(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) < 0;
    }
    bool operator==(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) == 0;
    }

    // get all args and env vars (including negated args/env)
    std::vector<std::string> get_args() const;
    std::vector<std::string> get_env() const;
};

namespace common_arg_utils {
    bool is_truthy(const std::string & value);
    bool is_falsey(const std::string & value);
    bool is_autoy(const std::string & value);
}

struct common_params_context {
    enum llama_example ex = LLAMA_EXAMPLE_COMMON;
    common_params & params;
    std::vector<common_arg> options;
    void(*print_usage)(int, char **) = nullptr;
    common_params_context(common_params & params) : params(params) {}
};

constexpr const char * COMMON_PRESET_DEFAULT_NAME = "default";

struct common_preset_context;

struct common_preset {
    std::string name;
    std::map<common_arg, std::string> options;

    std::vector<std::string> to_args(const std::string & bin_path = "") const;
    std::string to_ini() const;

    void set_option(const common_preset_context & ctx, const std::string & env, const std::string & value);
    void unset_option(const std::string & env);
    bool get_option(const std::string & env, std::string & value) const;
    void merge(const common_preset & other);
    void apply_to_params(common_params & params) const;
};

using common_presets = std::map<std::string, common_preset>;

struct common_preset_context {
    common_params default_params;
    common_params_context ctx_params;
    std::map<std::string, common_arg> key_to_opt;

    bool filter_allowed_keys = false;
    std::set<std::string> allowed_keys;

    common_preset_context(llama_example ex, bool only_remote_allowed = false);

    common_presets load_from_ini(const std::string & path, common_preset & global) const;
    common_presets load_from_cache() const;
    common_presets load_from_models_dir(const std::string & models_dir) const;
    common_preset load_from_args(int argc, char ** argv) const;
    common_presets cascade(const common_presets & base, const common_presets & added) const;
    common_presets cascade(const common_preset & base, const common_presets & presets) const;
};

// parse input arguments from CLI
// if one argument has invalid value, it will automatically display usage of the specific argument (and not the full usage message)
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

// parse input arguments from CLI into a map
bool common_params_to_map(int argc, char ** argv, llama_example ex, std::map<common_arg, std::string> & out_map);

// populate preset-only arguments
// these arguments are not treated as command line arguments
// see: https://github.com/ggml-org/llama.cpp/issues/18163
void common_params_add_preset_options(std::vector<common_arg> & args);

// initialize argument parser context - used by test-arg-parser and preset
common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

#include <cctype>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace jinja {

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

// Note: this is a simple hasher, not cryptographically secure, just for hash table usage
struct hasher {
    static constexpr auto size_t_digits = sizeof(size_t) * 8;
    static constexpr size_t prime = size_t_digits == 64 ? 0x100000001b3 : 0x01000193;
    static constexpr size_t seed = size_t_digits == 64 ? 0xcbf29ce484222325 : 0x811c9dc5;
    static constexpr auto block_size = sizeof(size_t); // in bytes; allowing the compiler to vectorize the computation

    static_assert(size_t_digits == 64 || size_t_digits == 32);
    static_assert(block_size == 8 || block_size == 4);

    uint8_t buffer[block_size];
    size_t idx = 0; // current index in buffer
    size_t state = seed;

    hasher() = default;
    hasher(const std::type_info & type_inf) noexcept {
        const auto type_hash = type_inf.hash_code();
        update(&type_hash, sizeof(type_hash));
    }

    // Properties:
    //   - update is not associative: update(a).update(b) != update(b).update(a)
    //   - update(a ~ b) == update(a).update(b) with ~ as concatenation operator --> useful for streaming
    //   - update("", 0) --> state unchanged with empty input
    hasher& update(void const * bytes, size_t len) noexcept {
        const uint8_t * c = static_cast<uint8_t const *>(bytes);
        if (len == 0) {
            return *this;
        }
        size_t processed = 0;

        // first, fill the existing buffer if it's partial
        if (idx > 0) {
            size_t to_fill = block_size - idx;
            if (to_fill > len) {
                to_fill = len;
            }
            std::memcpy(buffer + idx, c, to_fill);
            idx += to_fill;
            processed += to_fill;
            if (idx == block_size) {
                update_block(buffer);
                idx = 0;
            }
        }

        // process full blocks from the remaining input
        for (; processed + block_size <= len; processed += block_size) {
            update_block(c + processed);
        }

        // buffer any remaining bytes
        size_t remaining = len - processed;
        if (remaining > 0) {
            std::memcpy(buffer, c + processed, remaining);
            idx = remaining;
        }
        return *this;
    }

    // convenience function for testing only
    hasher& update(const std::string & s) noexcept {
        return update(s.data(), s.size());
    }

    // finalize and get the hash value
    // note: after calling digest, the hasher state is modified, do not call update() again
    size_t digest() noexcept {
        // if there are remaining bytes in buffer, fill the rest with zeros and process
        if (idx > 0) {
            for (size_t i = idx; i < block_size; ++i) {
                buffer[i] = 0;
            }
            update_block(buffer);
            idx = 0;
        }

        return state;
    }

private:
    // IMPORTANT: block must have at least block_size bytes
    void update_block(const uint8_t * block) noexcept {
        size_t blk = static_cast<uint32_t>(block[0])
                    | (static_cast<uint32_t>(block[1]) << 8)
                    | (static_cast<uint32_t>(block[2]) << 16)
                    | (static_cast<uint32_t>(block[3]) << 24);
        if constexpr (block_size == 8) {
            blk = blk | (static_cast<uint64_t>(block[4]) << 32)
                      | (static_cast<uint64_t>(block[5]) << 40)
                      | (static_cast<uint64_t>(block[6]) << 48)
                      | (static_cast<uint64_t>(block[7]) << 56);
        }
        state ^= blk;
        state *= prime;
    }
};

struct token {
    enum type {
        eof, // end of source
        text, // The text between Jinja statements or expressions

        numeric_literal, // e.g., 123, 1.0
        string_literal, // 'string'
        identifier, // Variables, functions, statements, booleans, etc.
        equals, // =
        open_paren, // (
        close_paren, // )
        open_statement, // {%
        close_statement, // %}
        open_expression, // {{
        close_expression, // }}
        open_square_bracket, // [
        close_square_bracket, // ]
        open_curly_bracket, // {
        close_curly_bracket, // }
        comma, // ,
        dot, // .
        colon, // :
        pipe, // |

        call_operator, // ()
        additive_binary_operator, // + - ~
        multiplicative_binary_operator, // * / %
        comparison_binary_operator, // < > <= >= == !=
        unary_operator, // ! - +
        comment, // {# ... #}
    };
    type t;
    std::string value;
    size_t pos;
};

struct lexer_result {
    std::vector<token> tokens;
    std::string source;
};

struct lexer {
    const std::map<char, char> escape_chars = {
        {'n', '\n'},
        {'t', '\t'},
        {'r', '\r'},
        {'b', '\b'},
        {'f', '\f'},
        {'v', '\v'},
        {'\\', '\\'},
        {'\'', '\''},
        {'\"', '\"'},
    };

    const std::vector<std::pair<std::string, token::type>> ordered_mapping_table = {
        // Trimmed control sequences
        {"{%-", token::open_statement},
        {"-%}", token::close_statement},
        {"{{-", token::open_expression},
        {"-}}", token::close_expression},
        // Control sequences
        {"{%", token::open_statement},
        {"%}", token::close_statement},
        {"{{", token::open_expression},
        {"}}", token::close_expression},
        // Single character tokens
        {"(", token::open_paren},
        {")", token::close_paren},
        {"{", token::open_curly_bracket},
        {"}", token::close_curly_bracket},
        {"[", token::open_square_bracket},
        {"]", token::close_square_bracket},
        {",", token::comma},
        {".", token::dot},
        {":", token::colon},
        {"|", token::pipe},
        // Comparison operators
        {"<=", token::comparison_binary_operator},
        {">=", token::comparison_binary_operator},
        {"==", token::comparison_binary_operator},
        {"!=", token::comparison_binary_operator},
        {"<", token::comparison_binary_operator},
        {">", token::comparison_binary_operator},
        // Arithmetic operators
        {"+", token::additive_binary_operator},
        {"-", token::additive_binary_operator},
        {"~", token::additive_binary_operator},
        {"*", token::multiplicative_binary_operator},
        {"/", token::multiplicative_binary_operator},
        {"%", token::multiplicative_binary_operator},
        // Assignment operator
        {"=", token::equals},
    };

    // tokenize the source string into a list of tokens
    // may throw lexer_exception on error
    lexer_result tokenize(const std::string & source);
};

struct lexer_exception : public std::runtime_error {
    lexer_exception(const std::string & msg, const std::string & source, size_t pos)
        : std::runtime_error([&]() {
            const char * tag = "lexer";
            llama_common_unicode_string result = llama_common_fmt_error_with_source_rust(
                    reinterpret_cast<const uint8_t *>(tag), std::strlen(tag),
                    reinterpret_cast<const uint8_t *>(msg.data()), msg.size(),
                    reinterpret_cast<const uint8_t *>(source.data()), source.size(), pos);
            std::string value(reinterpret_cast<const char *>(result.data), result.len);
            llama_common_unicode_string_free(result);
            return value;
        }()) {}
};

} // namespace jinja

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

namespace jinja {

extern "C" {
struct llama_jinja_string {
    uint8_t * data;
    size_t    len;
};

void llama_jinja_string_free(llama_jinja_string value);
bool llama_jinja_string_is_uppercase(const uint8_t * data, size_t len);
bool llama_jinja_string_is_lowercase(const uint8_t * data, size_t len);
llama_jinja_string llama_jinja_string_uppercase(const uint8_t * data, size_t len);
llama_jinja_string llama_jinja_string_lowercase(const uint8_t * data, size_t len);
llama_jinja_string llama_jinja_string_capitalize(const uint8_t * data, size_t len);
llama_jinja_string llama_jinja_string_titlecase(const uint8_t * data, size_t len);
llama_jinja_string llama_jinja_string_strip(
        const uint8_t * data, size_t len,
        bool left, bool right,
        const uint8_t * chars, size_t chars_len, bool has_chars);
}

static inline std::string jinja_string_from_rust(llama_jinja_string value) {
    std::string result(reinterpret_cast<const char *>(value.data), value.len);
    llama_jinja_string_free(value);
    return result;
}

// allow differentiate between user input strings and template strings
// transformations should handle this information as follows:
// - one-to-one (e.g., uppercase, lowercase): preserve is_input flag
// - one-to-many (e.g., strip): if input string is marked as is_input, all resulting parts should be marked as is_input
// - many-to-one (e.g., concat): if ALL input parts are marked as is_input, resulting part should be marked as is_input
struct string_part {
    bool is_input = false; // may skip parsing special tokens if true
    std::string val;

    bool is_uppercase() const {
        return llama_jinja_string_is_uppercase(reinterpret_cast<const uint8_t *>(val.data()), val.size());
    }
    bool is_lowercase() const {
        return llama_jinja_string_is_lowercase(reinterpret_cast<const uint8_t *>(val.data()), val.size());
    }
};

struct string {
    std::vector<string_part> parts;
    string() = default;
    string(const std::string & v, bool user_input = false) {
        parts.push_back({user_input, v});
    }
    string(int v) {
        parts.push_back({false, std::to_string(v)});
    }
    string(double v) {
        parts.push_back({false, std::to_string(v)});
    }

    // mark all parts as user input
    void mark_input() {
        for (auto & part : parts) {
            part.is_input = true;
        }
    }

    std::string str() const {
        if (parts.size() == 1) {
            return parts[0].val;
        }
        std::ostringstream oss;
        for (const auto & part : parts) {
            oss << part.val;
        }
        return oss.str();
    }
    size_t length() const {
        size_t len = 0;
        for (const auto & part : parts) {
            len += part.val.length();
        }
        return len;
    }
    void hash_update(hasher & hash) const noexcept {
        for (const auto & part : parts) {
            hash.update(part.val.data(), part.val.length());
        }
    }
    bool all_parts_are_input() const {
        for (const auto & part : parts) {
            if (!part.is_input) {
                return false;
            }
        }
        return true;
    }
    bool is_uppercase() const {
        for (const auto & part : parts) {
            if (!part.is_uppercase()) {
                return false;
            }
        }
        return true;
    }
    bool is_lowercase() const {
        for (const auto & part : parts) {
            if (!part.is_lowercase()) {
                return false;
            }
        }
        return true;
    }

    // mark this string as input if other has ALL parts as input
    void mark_input_based_on(const string & other) {
        if (other.all_parts_are_input()) {
            for (auto & part : parts) {
                part.is_input = true;
            }
        }
    }

    string append(const string & other) {
        for (const auto & part : other.parts) {
            parts.push_back(part);
        }
        return *this;
    }

    // in-place transformations

    string uppercase() {
        for (auto & part : parts) {
            part.val = jinja_string_from_rust(llama_jinja_string_uppercase(reinterpret_cast<const uint8_t *>(part.val.data()), part.val.size()));
        }
        return *this;
    }
    string lowercase() {
        for (auto & part : parts) {
            part.val = jinja_string_from_rust(llama_jinja_string_lowercase(reinterpret_cast<const uint8_t *>(part.val.data()), part.val.size()));
        }
        return *this;
    }
    string capitalize() {
        for (auto & part : parts) {
            part.val = jinja_string_from_rust(llama_jinja_string_capitalize(reinterpret_cast<const uint8_t *>(part.val.data()), part.val.size()));
        }
        return *this;
    }
    string titlecase() {
        for (auto & part : parts) {
            part.val = jinja_string_from_rust(llama_jinja_string_titlecase(reinterpret_cast<const uint8_t *>(part.val.data()), part.val.size()));
        }
        return *this;
    }
    string strip(bool left, bool right, std::optional<const std::string_view> chars = std::nullopt) {
        if (parts.empty()) {
            return *this;
        }
        auto strip_part = [&](std::string & value, bool left_part, bool right_part) {
            value = jinja_string_from_rust(llama_jinja_string_strip(
                    reinterpret_cast<const uint8_t *>(value.data()), value.size(),
                    left_part, right_part,
                    chars ? reinterpret_cast<const uint8_t *>((*chars).data()) : nullptr,
                    chars ? (*chars).size() : 0,
                    chars.has_value()));
        };
        if (left) {
            for (size_t i = 0; i < parts.size(); ++i) {
                strip_part(parts[i].val, true, false);
                if (parts[i].val.empty()) {
                    parts.erase(parts.begin() + i);
                    --i;
                    continue;
                }
                break;
            }
        }
        if (right) {
            for (size_t i = parts.size(); i-- > 0;) {
                strip_part(parts[i].val, false, true);
                if (parts[i].val.empty()) {
                    parts.erase(parts.begin() + i);
                    continue;
                }
                break;
            }
        }
        return *this;
    }
};

struct value_t;
using value = std::shared_ptr<value_t>;


// Helper to check the type of a value
template<typename T>
struct extract_pointee {
    using type = T;
};
template<typename U>
struct extract_pointee<std::shared_ptr<U>> {
    using type = U;
};
template<typename T>
bool is_val(const value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr.get()) != nullptr;
}
template<typename T>
bool is_val(const value_t * ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr) != nullptr;
}
template<typename T, typename... Args>
std::shared_ptr<typename extract_pointee<T>::type> mk_val(Args&&... args) {
    using PointeeType = typename extract_pointee<T>::type;
    return std::make_shared<PointeeType>(std::forward<Args>(args)...);
}
template<typename T>
const typename extract_pointee<T>::type * cast_val(const value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<const PointeeType*>(ptr.get());
}
template<typename T>
typename extract_pointee<T>::type * cast_val(value & ptr) {
    using PointeeType = typename extract_pointee<T>::type;
    return dynamic_cast<PointeeType*>(ptr.get());
}
// End Helper


struct context; // forward declaration


// for converting from JSON to jinja values
// example input JSON:
// {
//   "messages": [
//     {"role": "user", "content": "Hello!"},
//     {"role": "assistant", "content": "Hi there!"}
//   ],
//   "bos_token": "<s>",
//   "eos_token": "</s>",
// }
//
// to mark strings as user input, wrap them in a special object:
// {
//   "messages": [
//     {
//       "role": "user",
//       "content": {"__input__": "Hello!"}  // this string is user input
//     },
//     ...
//   ],
// }
//
// marking input can be useful for tracking data provenance
// and preventing template injection attacks
//
// Note: T_JSON can be nlohmann::ordered_json
template<typename T_JSON>
void global_from_json(context & ctx, const T_JSON & json_obj, bool mark_input);

//
// base value type
//

struct func_args; // function argument values

using func_hptr = value(const func_args &);
using func_handler = std::function<func_hptr>;
using func_builtins = std::map<std::string, func_handler>;

enum value_compare_op { eq, ge, gt, lt, ne };
bool value_compare(const value & a, const value & b, value_compare_op op);

struct value_t {
    int64_t val_int;
    double val_flt;
    string val_str;

    std::vector<value> val_arr;
    std::vector<std::pair<value, value>> val_obj;

    func_handler val_func;

    // only used if ctx.is_get_stats = true
    struct stats_t {
        bool used = false;
        // ops can be builtin calls or operators: "array_access", "object_access"
        std::set<std::string> ops;
        // utility to recursively mark value and its children as used
        static void mark_used(value & val, bool deep = false);
    } stats;

    value_t() = default;
    value_t(const value_t &) = default;
    virtual ~value_t() = default;

    // Note: only for debugging and error reporting purposes
    virtual std::string type() const { return ""; }

    virtual int64_t as_int() const { throw_type_error("is not an int value"); }
    virtual double as_float() const { throw_type_error("is not a float value"); }
    virtual string as_string() const { throw_type_error("is not a string value"); }
    virtual bool as_bool() const { throw_type_error("is not a bool value"); }
    virtual const std::vector<value> & as_array() const { throw_type_error("is not an array value"); }
    virtual const std::vector<std::pair<value, value>> & as_ordered_object() const { throw_type_error("is not an object value"); }
    virtual value invoke(const func_args &) const { throw_type_error("is not a function value"); }
    virtual bool is_none() const { return false; }
    virtual bool is_undefined() const { return false; }
    virtual const func_builtins & get_builtins() const { throw_type_error("has no builtins"); }

    virtual bool has_key(const value &) { throw_type_error("is not an object value"); }
    virtual void insert(const value & /* key */, const value & /* val */) { throw_type_error("is not an object value"); }
    virtual value & at(const value & /* key */, value & /* default_val */) { throw_type_error("is not an object value"); }
    virtual value & at(const value & /* key */) { throw_type_error("is not an object value"); }
    virtual value & at(const std::string & /* key */, value & /* default_val */) { throw_type_error("is not an object value"); }
    virtual value & at(const std::string & /* key */) { throw_type_error("is not an object value"); }
    virtual value & at(int64_t /* idx */, value & /* default_val */) { throw_type_error("is not an array value"); }
    virtual value & at(int64_t /* idx */) { throw_type_error("is not an array value"); }

    virtual bool is_numeric() const { return false; }
    virtual bool is_hashable() const { return false; }
    virtual bool is_immutable() const { return true; }
    virtual hasher unique_hash() const noexcept = 0;
    // TODO: C++20 <=> operator
    // NOTE: We are treating == as equivalent (for normal comparisons) and != as strict nonequal (for strict (is) comparisons)
    virtual bool operator==(const value_t & other) const { return equivalent(other); }
    virtual bool operator!=(const value_t & other) const { return nonequal(other); }

    // Note: only for debugging purposes
    virtual std::string as_repr() const { return as_string().str(); }

private:
    [[noreturn]] void throw_type_error(const char* expected) const {
        throw std::runtime_error(type() + " " + expected);
    }

protected:
    virtual bool equivalent(const value_t &) const = 0;
    virtual bool nonequal(const value_t & other) const { return !equivalent(other); }
};

//
// utils
//

const func_builtins & global_builtins();

std::string value_to_json(const value & val, int indent = -1, const std::string_view item_sep = ", ", const std::string_view key_sep = ": ");

// Note: only used for debugging purposes
std::string value_to_string_repr(const value & val);

struct not_implemented_exception : public std::runtime_error {
    not_implemented_exception(const std::string & msg) : std::runtime_error("NotImplemented: " + msg) {}
};

struct value_hasher {
    size_t operator()(const value & val) const noexcept {
        return val->unique_hash().digest();
    }
};

struct value_equivalence {
    bool operator()(const value & lhs, const value & rhs) const {
        return *lhs == *rhs;
    }
    bool operator()(const std::pair<value, value> & lhs, const std::pair<value, value> & rhs) const {
        return *(lhs.first) == *(rhs.first) && *(lhs.second) == *(rhs.second);
    }
};

struct value_equality {
    bool operator()(const value & lhs, const value & rhs) const {
        return !(*lhs != *rhs);
    }
};

//
// primitive value types
//

struct value_int_t : public value_t {
    value_int_t(int64_t v) {
        val_int = v;
        val_flt = static_cast<double>(v);
        if (static_cast<int64_t>(val_flt) != v) {
            val_flt = v < 0 ? -INFINITY : INFINITY;
        }
    }
    virtual std::string type() const override { return "Integer"; }
    virtual int64_t as_int() const override { return val_int; }
    virtual double as_float() const override { return val_flt; }
    virtual string as_string() const override { return std::to_string(val_int); }
    virtual bool as_bool() const override {
        return val_int != 0;
    }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_numeric() const override { return true; }
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        return hasher(typeid(*this))
            .update(&val_int, sizeof(val_int))
            .update(&val_flt, sizeof(val_flt));
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return other.is_numeric() && val_int == other.val_int && val_flt == other.val_flt;
    }
    virtual bool nonequal(const value_t & other) const override {
        return !(typeid(*this) == typeid(other) && val_int == other.val_int);
    }
};
using value_int = std::shared_ptr<value_int_t>;


struct value_float_t : public value_t {
    value val;
    value_float_t(double v) {
        val_flt = v;
        val_int = std::isfinite(v) ? static_cast<int64_t>(v) : 0;
        val = mk_val<value_int>(val_int);
    }
    virtual std::string type() const override { return "Float"; }
    virtual double as_float() const override { return val_flt; }
    virtual int64_t as_int() const override { return val_int; }
    virtual string as_string() const override {
        std::string out = std::to_string(val_flt);
        out.erase(out.find_last_not_of('0') + 1, std::string::npos); // remove trailing zeros
        if (out.back() == '.') out.push_back('0'); // leave one zero if no decimals
        return out;
    }
    virtual bool as_bool() const override {
        return val_flt != 0.0;
    }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_numeric() const override { return true; }
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        if (static_cast<double>(val_int) == val_flt) {
            return val->unique_hash();
        } else {
            return hasher(typeid(*this))
                .update(&val_int, sizeof(val_int))
                .update(&val_flt, sizeof(val_flt));
        }
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return other.is_numeric() && val_int == other.val_int && val_flt == other.val_flt;
    }
    virtual bool nonequal(const value_t & other) const override {
        return !(typeid(*this) == typeid(other) && val_flt == other.val_flt);
    }
};
using value_float = std::shared_ptr<value_float_t>;


struct value_string_t : public value_t {
    value_string_t() { val_str = string(); }
    value_string_t(const std::string & v) { val_str = string(v); }
    value_string_t(const string & v) { val_str = v; }
    virtual std::string type() const override { return "String"; }
    virtual string as_string() const override { return val_str; }
    virtual std::string as_repr() const override {
        std::ostringstream ss;
        for (const auto & part : val_str.parts) {
            ss << (part.is_input ? "INPUT: " : "TMPL:  ") << part.val << "\n";
        }
        return ss.str();
    }
    virtual bool as_bool() const override {
        return val_str.length() > 0;
    }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        const auto type_hash = typeid(*this).hash_code();
        auto hash = hasher();
        hash.update(&type_hash, sizeof(type_hash));
        val_str.hash_update(hash);
        return hash;
    }
    void mark_input() {
        val_str.mark_input();
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return typeid(*this) == typeid(other) && val_str.str() == other.val_str.str();
    }
};
using value_string = std::shared_ptr<value_string_t>;


struct value_bool_t : public value_t {
    value val;
    value_bool_t(bool v) {
        val_int = static_cast<int64_t>(v);
        val_flt = static_cast<double>(v);
        val = mk_val<value_int>(val_int);
    }
    virtual std::string type() const override { return "Boolean"; }
    virtual int64_t as_int() const override { return val_int; }
    virtual bool as_bool() const override { return val_int; }
    virtual string as_string() const override { return std::string(val_int ? "True" : "False"); }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_numeric() const override { return true; }
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        return val->unique_hash();
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return other.is_numeric() && val_int == other.val_int && val_flt == other.val_flt;
    }
    virtual bool nonequal(const value_t & other) const override {
        return !(typeid(*this) == typeid(other) && val_int == other.val_int);
    }
};
using value_bool = std::shared_ptr<value_bool_t>;


struct value_array_t : public value_t {
    value_array_t() = default;
    value_array_t(value & v) {
        val_arr = v->val_arr;
    }
    value_array_t(std::vector<value> && arr) {
        val_arr = arr;
    }
    value_array_t(const std::vector<value> & arr) {
        val_arr = arr;
    }
    void reverse() {
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
        std::reverse(val_arr.begin(), val_arr.end());
    }
    void push_back(const value & val) {
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
        val_arr.push_back(val);
    }
    void push_back(value && val) {
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
        val_arr.push_back(std::move(val));
    }
    value pop_at(int64_t index) {
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
        if (index < 0) {
            index = static_cast<int64_t>(val_arr.size()) + index;
        }
        if (index < 0 || index >= static_cast<int64_t>(val_arr.size())) {
            throw std::runtime_error("Index " + std::to_string(index) + " out of bounds for array of size " + std::to_string(val_arr.size()));
        }
        value val = val_arr.at(static_cast<size_t>(index));
        val_arr.erase(val_arr.begin() + index);
        return val;
    }
    virtual std::string type() const override { return "Array"; }
    virtual bool is_immutable() const override { return false; }
    virtual const std::vector<value> & as_array() const override { return val_arr; }
    virtual string as_string() const override {
        const bool immutable = is_immutable();
        std::ostringstream ss;
        ss << (immutable ? "(" : "[");
        for (size_t i = 0; i < val_arr.size(); i++) {
            if (i > 0) ss << ", ";
            value val = val_arr.at(i);
            ss << value_to_string_repr(val);
        }
        if (immutable && val_arr.size() == 1) {
            ss << ",";
        }
        ss << (immutable ? ")" : "]");
        return ss.str();
    }
    virtual bool as_bool() const override {
        return !val_arr.empty();
    }
    virtual value & at(int64_t index, value & default_val) override {
        if (index < 0) {
            index += val_arr.size();
        }
        if (index < 0 || static_cast<size_t>(index) >= val_arr.size()) {
            return default_val;
        }
        return val_arr[index];
    }
    virtual value & at(int64_t index) override {
        if (index < 0) {
            index += val_arr.size();
        }
        if (index < 0 || static_cast<size_t>(index) >= val_arr.size()) {
            throw std::runtime_error("Index " + std::to_string(index) + " out of bounds for array of size " + std::to_string(val_arr.size()));
        }
        return val_arr[index];
    }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_hashable() const override {
        if (std::all_of(val_arr.begin(), val_arr.end(), [&](auto & val) -> bool {
            return val->is_immutable() && val->is_hashable();
        })) {
            return true;
        }
        return false;
    }
    virtual hasher unique_hash() const noexcept override {
        auto hash = hasher(typeid(*this));
        for (const auto & val : val_arr) {
            // must use digest to prevent problems from "concatenation" property of hasher
            // for ex. hash of [ "ab", "c" ] should be different from [ "a", "bc" ]
            const size_t val_hash = val->unique_hash().digest();
            hash.update(&val_hash, sizeof(size_t));
        }
        return hash;
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return typeid(*this) == typeid(other) && is_hashable() && other.is_hashable() && std::equal(val_arr.begin(), val_arr.end(), other.val_arr.begin(), other.val_arr.end(), value_equivalence());
    }
};
using value_array = std::shared_ptr<value_array_t>;


struct value_tuple_t : public value_array_t {
    value_tuple_t(value & v) {
        val_arr = v->val_arr;
    }
    value_tuple_t(std::vector<value> && arr) {
        val_arr = arr;
    }
    value_tuple_t(const std::vector<value> & arr) {
        val_arr = arr;
    }
    value_tuple_t(const std::pair<value, value> & pair) {
        val_arr.push_back(pair.first);
        val_arr.push_back(pair.second);
    }
    virtual std::string type() const override { return "Tuple"; }
    virtual bool is_immutable() const override { return true; }
};
using value_tuple = std::shared_ptr<value_tuple_t>;


struct value_object_t : public value_t {
    std::unordered_map<value, value, value_hasher, value_equivalence> unordered;
    bool has_builtins = true; // context and loop objects do not have builtins
    value_object_t() = default;
    value_object_t(value & v) {
        val_obj = v->val_obj;
        for (const auto & pair : val_obj) {
            unordered[pair.first] = pair.second;
        }
    }
    value_object_t(const std::map<value, value> & obj) {
        for (const auto & pair : obj) {
            insert(pair.first, pair.second);
        }
    }
    value_object_t(const std::vector<std::pair<value, value>> & obj) {
        for (const auto & pair : obj) {
            insert(pair.first, pair.second);
        }
    }
    void insert(const std::string & key, const value & val) {
        insert(mk_val<value_string>(key), val);
    }
    virtual std::string type() const override { return "Object"; }
    virtual bool is_immutable() const override { return false; }
    virtual const std::vector<std::pair<value, value>> & as_ordered_object() const override { return val_obj; }
    virtual string as_string() const override {
        std::ostringstream ss;
        ss << "{";
        for (size_t i = 0; i < val_obj.size(); i++) {
            if (i > 0) ss << ", ";
            auto & [key, val] = val_obj.at(i);
            ss << value_to_string_repr(key) << ": " << value_to_string_repr(val);
        }
        ss << "}";
        return ss.str();
    }
    virtual bool as_bool() const override {
        return !unordered.empty();
    }
    virtual bool has_key(const value & key) override {
        if (!key->is_immutable() || !key->is_hashable()) {
            throw std::runtime_error("Object key of unhashable type: " + key->type());
        }
        return unordered.find(key) != unordered.end();
    }
    virtual void insert(const value & key, const value & val) override {
        bool replaced = false;
        if (is_immutable()) {
            throw std::runtime_error("Attempting to modify immutable type");
        }
        if (has_key(key)) {
            // if key exists, replace value in ordered list instead of appending
            for (auto & pair : val_obj) {
                if (*(pair.first) == *key) {
                    pair.second = val;
                    replaced = true;
                    break;
                }
            }
        }
        unordered[key] = val;
        if (!replaced) {
            val_obj.push_back({key, val});
        }
    }
    virtual value & at(const value & key, value & default_val) override {
        if (!has_key(key)) {
            return default_val;
        }
        return unordered.at(key);
    }
    virtual value & at(const value & key) override {
        if (!has_key(key)) {
            throw std::runtime_error("Key '" + key->as_string().str() + "' not found in value of type " + type());
        }
        return unordered.at(key);
    }
    virtual value & at(const std::string & key, value & default_val) override {
        value key_val = mk_val<value_string>(key);
        return at(key_val, default_val);
    }
    virtual value & at(const std::string & key) override {
        value key_val = mk_val<value_string>(key);
        return at(key_val);
    }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_hashable() const override {
        if (std::all_of(val_obj.begin(), val_obj.end(), [&](auto & pair) -> bool {
            const auto & val = pair.second;
            return val->is_immutable() && val->is_hashable();
        })) {
            return true;
        }
        return false;
    }
    virtual hasher unique_hash() const noexcept override {
        auto hash = hasher(typeid(*this));
        for (const auto & [key, val] : val_obj) {
            // must use digest to prevent problems from "concatenation" property of hasher
            // for ex. hash of key="ab", value="c" should be different from key="a", value="bc"
            const size_t key_hash = key->unique_hash().digest();
            const size_t val_hash = val->unique_hash().digest();
            hash.update(&key_hash, sizeof(key_hash));
            hash.update(&val_hash, sizeof(val_hash));
        }
        return hash;
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return typeid(*this) == typeid(other) && is_hashable() && other.is_hashable() && std::equal(val_obj.begin(), val_obj.end(), other.val_obj.begin(), other.val_obj.end(), value_equivalence());
    }
};
using value_object = std::shared_ptr<value_object_t>;

//
// none and undefined types
//

struct value_none_t : public value_t {
    virtual std::string type() const override { return "None"; }
    virtual bool is_none() const override { return true; }
    virtual bool as_bool() const override { return false; }
    virtual string as_string() const override { return string(type()); }
    virtual std::string as_repr() const override { return type(); }
    virtual const func_builtins & get_builtins() const override;
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        return hasher(typeid(*this));
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return typeid(*this) == typeid(other);
    }
};
using value_none = std::shared_ptr<value_none_t>;

struct value_undefined_t : public value_t {
    std::string hint; // for debugging, to indicate where undefined came from
    value_undefined_t(const std::string & h = "") : hint(h) {}
    virtual std::string type() const override { return hint.empty() ? "Undefined" : "Undefined (hint: '" + hint + "')"; }
    virtual bool is_undefined() const override { return true; }
    virtual bool as_bool() const override { return false; }
    virtual std::string as_repr() const override { return type(); }
    virtual const func_builtins & get_builtins() const override;
    virtual hasher unique_hash() const noexcept override {
        return hasher(typeid(*this));
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        return is_undefined() == other.is_undefined();
    }
};
using value_undefined = std::shared_ptr<value_undefined_t>;

//
// function type
//

struct func_args {
public:
    std::string func_name; // for error messages
    context & ctx;
    func_args(context & ctx) : ctx(ctx) {}
    value get_kwarg(const std::string & key, value default_val) const;
    value get_kwarg_or_pos(const std::string & key, size_t pos) const;
    value get_pos(size_t pos) const;
    value get_pos(size_t pos, value default_val) const;
    const std::vector<value> & get_args() const;
    size_t count() const { return args.size(); }
    void push_back(const value & val);
    void push_front(const value & val);
    void ensure_count(size_t min, size_t max = 999) const {
        size_t n = args.size();
        if (n < min || n > max) {
            throw std::runtime_error("Function '" + func_name + "' expected between " + std::to_string(min) + " and " + std::to_string(max) + " arguments, got " + std::to_string(n));
        }
    }
    template<typename T> void ensure_val(const value & ptr) const {
        if (!is_val<T>(ptr)) {
            throw std::runtime_error("Function '" + func_name + "' expected value of type " + std::string(typeid(T).name()) + ", got " + ptr->type());
        }
    }
    void ensure_count(bool require0, bool require1, bool require2, bool require3) const {
        static auto bool_to_int = [](bool b) { return b ? 1 : 0; };
        size_t required = bool_to_int(require0) + bool_to_int(require1) + bool_to_int(require2) + bool_to_int(require3);
        ensure_count(required);
    }
    template<typename T0> void ensure_vals(bool required0 = true) const {
        ensure_count(required0, false, false, false);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
    }
    template<typename T0, typename T1> void ensure_vals(bool required0 = true, bool required1 = true) const {
        ensure_count(required0, required1, false, false);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
        if (required1 && args.size() > 1) ensure_val<T1>(args[1]);
    }
    template<typename T0, typename T1, typename T2> void ensure_vals(bool required0 = true, bool required1 = true, bool required2 = true) const {
        ensure_count(required0, required1, required2, false);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
        if (required1 && args.size() > 1) ensure_val<T1>(args[1]);
        if (required2 && args.size() > 2) ensure_val<T2>(args[2]);
    }
    template<typename T0, typename T1, typename T2, typename T3> void ensure_vals(bool required0 = true, bool required1 = true, bool required2 = true, bool required3 = true) const {
        ensure_count(required0, required1, required2, required3);
        if (required0 && args.size() > 0) ensure_val<T0>(args[0]);
        if (required1 && args.size() > 1) ensure_val<T1>(args[1]);
        if (required2 && args.size() > 2) ensure_val<T2>(args[2]);
        if (required3 && args.size() > 3) ensure_val<T3>(args[3]);
    }
private:
    std::vector<value> args;
};

struct value_func_t : public value_t {
    std::string name;
    value arg0; // bound "this" argument, if any
    value_func_t(const std::string & name, const func_handler & func) : name(name) {
        val_func = func;
    }
    value_func_t(const std::string & name, const func_handler & func, const value & arg_this) : name(name), arg0(arg_this) {
        val_func = func;
    }
    virtual value invoke(const func_args & args) const override {
        func_args new_args(args); // copy
        new_args.func_name = name;
        if (arg0) {
            new_args.push_front(arg0);
        }
        return val_func(new_args);
    }
    virtual std::string type() const override { return "Function"; }
    virtual std::string as_repr() const override { return type() + "<" + name + ">(" + (arg0 ? arg0->as_repr() : "") + ")"; }
    virtual bool is_hashable() const override { return false; }
    virtual hasher unique_hash() const noexcept override {
        // Note: this is unused for now, we don't support function as object keys
        // use function pointer as unique identifier
        const auto target = val_func.target<func_hptr>();
        return hasher(typeid(*this)).update(&target, sizeof(target));
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        // Note: this is unused for now, we don't support function as object keys
        // compare function pointers
        // (val_func == other.val_func does not work as std::function::operator== is only used for nullptr check)
        const auto target_this  = this->val_func.target<func_hptr>();
        const auto target_other = other.val_func.target<func_hptr>();
        return typeid(*this) == typeid(other) && target_this == target_other;
    }
};
using value_func = std::shared_ptr<value_func_t>;

// special value for kwarg
struct value_kwarg_t : public value_t {
    std::string key;
    value val;
    value_kwarg_t(const std::string & k, const value & v) : key(k), val(v) {}
    virtual std::string type() const override { return "KwArg"; }
    virtual std::string as_repr() const override { return type(); }
    virtual bool is_hashable() const override { return true; }
    virtual hasher unique_hash() const noexcept override {
        const auto type_hash = typeid(*this).hash_code();
        auto hash = val->unique_hash();
        hash.update(&type_hash, sizeof(type_hash))
            .update(key.data(), key.size());
        return hash;
    }
protected:
    virtual bool equivalent(const value_t & other) const override {
        const value_kwarg_t & other_val = static_cast<const value_kwarg_t &>(other);
        return typeid(*this) == typeid(other) && key == other_val.key && val == other_val.val;
    }
};
using value_kwarg = std::shared_ptr<value_kwarg_t>;


} // namespace jinja

#include <cassert>
#include <ctime>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define JJ_DEBUG(msg, ...)  do { if (g_jinja_debug) printf("%s:%-3d : " msg "\n", FILENAME, __LINE__, __VA_ARGS__); } while (0)

extern bool g_jinja_debug;

namespace jinja {

struct statement;
using statement_ptr = std::unique_ptr<statement>;
using statements = std::vector<statement_ptr>;

// Helpers for dynamic casting and type checking
template<typename T>
struct extract_pointee_unique {
    using type = T;
};
template<typename U>
struct extract_pointee_unique<std::unique_ptr<U>> {
    using type = U;
};
template<typename T>
bool is_stmt(const statement_ptr & ptr) {
    return dynamic_cast<const T*>(ptr.get()) != nullptr;
}
template<typename T>
T * cast_stmt(statement_ptr & ptr) {
    return dynamic_cast<T*>(ptr.get());
}
template<typename T>
const T * cast_stmt(const statement_ptr & ptr) {
    return dynamic_cast<const T*>(ptr.get());
}
// End Helpers


// not thread-safe
void enable_debug(bool enable);

struct context {
    std::shared_ptr<std::string> src; // for debugging; use shared_ptr to avoid copying on scope creation
    std::time_t current_time; // for functions that need current time

    bool is_get_stats = false; // whether to collect stats

    // src is optional, used for error reporting
    context(std::string src = "") : src(std::make_shared<std::string>(std::move(src))) {
        env = mk_val<value_object>();
        env->has_builtins = false; // context object has no builtins
        env->insert("true",  mk_val<value_bool>(true));
        env->insert("True",  mk_val<value_bool>(true));
        env->insert("false", mk_val<value_bool>(false));
        env->insert("False", mk_val<value_bool>(false));
        env->insert("none",  mk_val<value_none>());
        env->insert("None",  mk_val<value_none>());
        current_time = std::time(nullptr);
    }
    ~context() = default;

    context(const context & parent) : context() {
        // inherit variables (for example, when entering a new scope)
        auto & pvar = parent.env->as_ordered_object();
        for (const auto & pair : pvar) {
            set_val(pair.first, pair.second);
        }
        current_time = parent.current_time;
        is_get_stats = parent.is_get_stats;
        src = parent.src;
    }

    value get_val(const std::string & name) {
        value default_val = mk_val<value_undefined>(name);
        return env->at(name, default_val);
    }

    void set_val(const std::string & name, const value & val) {
        env->insert(name, val);
    }

    void set_val(const value & name, const value & val) {
        env->insert(name, val);
    }

    void print_vars() const {
        printf("Context Variables:\n%s\n", value_to_json(env, 2).c_str());
    }

private:
    value_object env;
};

/**
 * Base class for all nodes in the AST.
 */
struct statement {
    size_t pos; // position in source, for debugging
    virtual ~statement() = default;
    virtual std::string type() const { return "Statement"; }

    // execute_impl must be overridden by derived classes
    virtual value execute_impl(context &) { throw_exec_error(); }
    // execute is the public method to execute a statement with error handling
    value execute(context &);

private:
    [[noreturn]] void throw_exec_error() const {
        throw std::runtime_error("cannot exec " + type());
    }
};

// Type Checking Utilities

template<typename T>
static void chk_type(const statement_ptr & ptr) {
    if (!ptr) return; // Allow null for optional fields
    assert(dynamic_cast<T *>(ptr.get()) != nullptr);
}

template<typename T, typename U>
static void chk_type(const statement_ptr & ptr) {
    if (!ptr) return;
    assert(dynamic_cast<T *>(ptr.get()) != nullptr || dynamic_cast<U *>(ptr.get()) != nullptr);
}

// Base Types

/**
 * Expressions will result in a value at runtime (unlike statements).
 */
struct expression : public statement {
    std::string type() const override { return "Expression"; }
};

// Statements

struct program : public statement {
    statements body;

    program() = default;
    explicit program(statements && body) : body(std::move(body)) {}
    std::string type() const override { return "Program"; }
    [[noreturn]] value execute_impl(context &) override {
        throw std::runtime_error("Cannot execute program directly, use jinja::runtime instead");
    }
};

struct if_statement : public statement {
    statement_ptr test;
    statements body;
    statements alternate;

    if_statement(statement_ptr && test, statements && body, statements && alternate)
        : test(std::move(test)), body(std::move(body)), alternate(std::move(alternate)) {
        chk_type<expression>(this->test);
    }

    std::string type() const override { return "If"; }
    value execute_impl(context & ctx) override;
};

struct identifier;
struct tuple_literal;

/**
 * Loop over each item in a sequence
 * https://jinja.palletsprojects.com/en/3.0.x/templates/#for
 */
struct for_statement : public statement {
    statement_ptr loopvar; // Identifier | TupleLiteral
    statement_ptr iterable;
    statements body;
    statements default_block; // if no iteration took place

    for_statement(statement_ptr && loopvar, statement_ptr && iterable, statements && body, statements && default_block)
        : loopvar(std::move(loopvar)), iterable(std::move(iterable)),
          body(std::move(body)), default_block(std::move(default_block)) {
        chk_type<identifier, tuple_literal>(this->loopvar);
        chk_type<expression>(this->iterable);
    }

    std::string type() const override { return "For"; }
    value execute_impl(context & ctx) override;
};

struct break_statement : public statement {
    std::string type() const override { return "Break"; }

    struct signal : public std::exception {
        const char* what() const noexcept override {
            return "Break statement executed";
        }
    };

    [[noreturn]] value execute_impl(context &) override {
        throw break_statement::signal();
    }
};

struct continue_statement : public statement {
    std::string type() const override { return "Continue"; }

    struct signal : public std::exception {
        const char* what() const noexcept override {
            return "Continue statement executed";
        }
    };

    [[noreturn]] value execute_impl(context &) override {
        throw continue_statement::signal();
    }
};

// do nothing
struct noop_statement : public statement {
    std::string type() const override { return "Noop"; }
    value execute_impl(context &) override {
        return mk_val<value_undefined>();
    }
};

struct set_statement : public statement {
    statement_ptr assignee;
    statement_ptr val;
    statements body;

    set_statement(statement_ptr && assignee, statement_ptr && value, statements && body)
        : assignee(std::move(assignee)), val(std::move(value)), body(std::move(body)) {
        chk_type<expression>(this->assignee);
        chk_type<expression>(this->val);
    }

    std::string type() const override { return "Set"; }
    value execute_impl(context & ctx) override;
};

struct macro_statement : public statement {
    statement_ptr name;
    statements args;
    statements body;

    macro_statement(statement_ptr && name, statements && args, statements && body)
        : name(std::move(name)), args(std::move(args)), body(std::move(body)) {
        chk_type<identifier>(this->name);
        for (const auto& arg : this->args) chk_type<expression>(arg);
    }

    std::string type() const override { return "Macro"; }
    value execute_impl(context & ctx) override;
};

struct comment_statement : public statement {
    std::string val;
    explicit comment_statement(const std::string & v) : val(v) {}
    std::string type() const override { return "Comment"; }
    value execute_impl(context &) override {
        return mk_val<value_undefined>();
    }
};

// Expressions

// Represents an omitted expression in a computed member, e.g. `a[]`.
struct blank_expression : public expression {
    std::string type() const override { return "BlankExpression"; }
    value execute_impl(context &) override {
        return mk_val<value_undefined>();
    }
};

struct member_expression : public expression {
    statement_ptr object;
    statement_ptr property;
    bool computed; // true if obj[expr] and false if obj.prop

    member_expression(statement_ptr && object, statement_ptr && property, bool computed)
        : object(std::move(object)), property(std::move(property)), computed(computed) {
        chk_type<expression>(this->object);
        chk_type<expression>(this->property);
    }
    std::string type() const override { return "MemberExpression"; }
    value execute_impl(context & ctx) override;
};

struct call_expression : public expression {
    statement_ptr callee;
    statements args;

    call_expression(statement_ptr && callee, statements && args)
        : callee(std::move(callee)), args(std::move(args)) {
        chk_type<expression>(this->callee);
        for (const auto& arg : this->args) chk_type<expression>(arg);
    }
    std::string type() const override { return "CallExpression"; }
    value execute_impl(context & ctx) override;
};

/**
 * Represents a user-defined variable or symbol in the template.
 */
struct identifier : public expression {
    std::string val;
    explicit identifier(const std::string & val) : val(val) {}
    std::string type() const override { return "Identifier"; }
    value execute_impl(context & ctx) override;
};

// Literals

struct integer_literal : public expression {
    int64_t val;
    explicit integer_literal(int64_t val) : val(val) {}
    std::string type() const override { return "IntegerLiteral"; }
    value execute_impl(context &) override {
        return mk_val<value_int>(val);
    }
};

struct float_literal : public expression {
    double val;
    explicit float_literal(double val) : val(val) {}
    std::string type() const override { return "FloatLiteral"; }
    value execute_impl(context &) override {
        return mk_val<value_float>(val);
    }
};

struct string_literal : public expression {
    std::string val;
    explicit string_literal(const std::string & val) : val(val) {}
    std::string type() const override { return "StringLiteral"; }
    value execute_impl(context &) override {
        return mk_val<value_string>(val);
    }
};

struct array_literal : public expression {
    statements val;
    explicit array_literal(statements && val) : val(std::move(val)) {
        for (const auto& item : this->val) chk_type<expression>(item);
    }
    std::string type() const override { return "ArrayLiteral"; }
    value execute_impl(context & ctx) override {
        auto arr = mk_val<value_array>();
        for (const auto & item_stmt : val) {
            arr->push_back(item_stmt->execute(ctx));
        }
        return arr;
    }
};

struct tuple_literal : public expression {
    statements val;
    explicit tuple_literal(statements && val) : val(std::move(val)) {
        for (const auto& item : this->val) chk_type<expression>(item);
    }
    std::string type() const override { return "TupleLiteral"; }
    value execute_impl(context & ctx) override {
        auto arr = mk_val<value_array>();
        for (const auto & item_stmt : val) {
            arr->push_back(item_stmt->execute(ctx));
        }
        return mk_val<value_tuple>(std::move(arr->as_array()));
    }
};

struct object_literal : public expression {
    std::vector<std::pair<statement_ptr, statement_ptr>> val;
    explicit object_literal(std::vector<std::pair<statement_ptr, statement_ptr>> && val)
        : val(std::move(val)) {
        for (const auto & pair : this->val) {
            chk_type<expression>(pair.first);
            chk_type<expression>(pair.second);
        }
    }
    std::string type() const override { return "ObjectLiteral"; }
    value execute_impl(context & ctx) override;
};

// Complex Expressions

/**
 * An operation with two sides, separated by an operator.
 * Note: Either side can be a Complex Expression, with order
 * of operations being determined by the operator.
 */
struct binary_expression : public expression {
    token op;
    statement_ptr left;
    statement_ptr right;

    binary_expression(token op, statement_ptr && left, statement_ptr && right)
        : op(std::move(op)), left(std::move(left)), right(std::move(right)) {
        chk_type<expression>(this->left);
        chk_type<expression>(this->right);
    }
    std::string type() const override { return "BinaryExpression"; }
    value execute_impl(context & ctx) override;
};

/**
 * An operation with two sides, separated by the | operator.
 * Operator precedence: https://github.com/pallets/jinja/issues/379#issuecomment-168076202
 */
struct filter_expression : public expression {
    // either an expression or a value is allowed
    statement_ptr operand;
    value_string val; // will be set by filter_statement

    statement_ptr filter;

    filter_expression(statement_ptr && operand, statement_ptr && filter)
        : operand(std::move(operand)), filter(std::move(filter)) {
        chk_type<expression>(this->operand);
        chk_type<identifier, call_expression>(this->filter);
    }

    filter_expression(value_string && val, statement_ptr && filter)
        : val(std::move(val)), filter(std::move(filter)) {
        chk_type<identifier, call_expression>(this->filter);
    }

    std::string type() const override { return "FilterExpression"; }
    value execute_impl(context & ctx) override;
};

struct filter_statement : public statement {
    statement_ptr filter;
    statements body;

    filter_statement(statement_ptr && filter, statements && body)
        : filter(std::move(filter)), body(std::move(body)) {
        chk_type<identifier, call_expression>(this->filter);
    }
    std::string type() const override { return "FilterStatement"; }
    value execute_impl(context & ctx) override;
};

/**
 * An operation which filters a sequence of objects by applying a test to each object,
 * and only selecting the objects with the test succeeding.
 *
 * It may also be used as a shortcut for a ternary operator.
 */
struct select_expression : public expression {
    statement_ptr lhs;
    statement_ptr test;

    select_expression(statement_ptr && lhs, statement_ptr && test)
        : lhs(std::move(lhs)), test(std::move(test)) {
        chk_type<expression>(this->lhs);
        chk_type<expression>(this->test);
    }
    std::string type() const override { return "SelectExpression"; }
    value execute_impl(context & ctx) override {
        auto predicate = test->execute_impl(ctx);
        if (!predicate->as_bool()) {
            return mk_val<value_undefined>();
        }
        return lhs->execute_impl(ctx);
    }
};

/**
 * An operation with two sides, separated by the "is" operator.
 * NOTE: "value is something" translates to function call "test_is_something(value)"
 */
struct test_expression : public expression {
    statement_ptr operand;
    bool negate;
    statement_ptr test;

    test_expression(statement_ptr && operand, bool negate, statement_ptr && test)
        : operand(std::move(operand)), negate(negate), test(std::move(test)) {
        chk_type<expression>(this->operand);
        chk_type<identifier, call_expression>(this->test);
    }
    std::string type() const override { return "TestExpression"; }
    value execute_impl(context & ctx) override;
};

/**
 * An operation with one side (operator on the left).
 */
struct unary_expression : public expression {
    token op;
    statement_ptr argument;

    unary_expression(token op, statement_ptr && argument)
        : op(std::move(op)), argument(std::move(argument)) {
        chk_type<expression>(this->argument);
    }
    std::string type() const override { return "UnaryExpression"; }
    value execute_impl(context & ctx) override;
};

struct slice_expression : public expression {
    statement_ptr start_expr;
    statement_ptr stop_expr;
    statement_ptr step_expr;

    slice_expression(statement_ptr && start_expr, statement_ptr && stop_expr, statement_ptr && step_expr)
        : start_expr(std::move(start_expr)), stop_expr(std::move(stop_expr)), step_expr(std::move(step_expr)) {
        chk_type<expression>(this->start_expr);
        chk_type<expression>(this->stop_expr);
        chk_type<expression>(this->step_expr);
    }
    std::string type() const override { return "SliceExpression"; }
    [[noreturn]] value execute_impl(context &) override {
        throw std::runtime_error("must be handled by MemberExpression");
    }
};

struct keyword_argument_expression : public expression {
    statement_ptr key;
    statement_ptr val;

    keyword_argument_expression(statement_ptr && key, statement_ptr && val)
        : key(std::move(key)), val(std::move(val)) {
        chk_type<identifier>(this->key);
        chk_type<expression>(this->val);
    }
    std::string type() const override { return "KeywordArgumentExpression"; }
    value execute_impl(context & ctx) override;
};

struct spread_expression : public expression {
    statement_ptr argument;
    explicit spread_expression(statement_ptr && argument) : argument(std::move(argument)) {
        chk_type<expression>(this->argument);
    }
    std::string type() const override { return "SpreadExpression"; }
};

struct call_statement : public statement {
    statement_ptr call;
    statements caller_args;
    statements body;

    call_statement(statement_ptr && call, statements && caller_args, statements && body)
        : call(std::move(call)), caller_args(std::move(caller_args)), body(std::move(body)) {
        chk_type<call_expression>(this->call);
        for (const auto & arg : this->caller_args) chk_type<expression>(arg);
    }
    std::string type() const override { return "CallStatement"; }
};

struct ternary_expression : public expression {
    statement_ptr condition;
    statement_ptr true_expr;
    statement_ptr false_expr;

    ternary_expression(statement_ptr && condition, statement_ptr && true_expr, statement_ptr && false_expr)
        : condition(std::move(condition)), true_expr(std::move(true_expr)), false_expr(std::move(false_expr)) {
        chk_type<expression>(this->condition);
        chk_type<expression>(this->true_expr);
        chk_type<expression>(this->false_expr);
    }
    std::string type() const override { return "Ternary"; }
    value execute_impl(context & ctx) override {
        value cond_val = condition->execute(ctx);
        if (cond_val->as_bool()) {
            return true_expr->execute(ctx);
        } else {
            return false_expr->execute(ctx);
        }
    }
};

struct raised_exception : public std::exception {
    std::string message;
    raised_exception(const std::string & msg) : message(msg) {}
    const char* what() const noexcept override {
        return message.c_str();
    }
};

// Used to rethrow exceptions with modified messages
struct rethrown_exception : public std::exception {
    std::string message;
    rethrown_exception(const std::string & msg) : message(msg) {}
    const char* what() const noexcept override {
        return message.c_str();
    }
};

//////////////////////

static void gather_string_parts_recursive(const value & val, value_string & parts) {
    // TODO: probably allow print value_none as "None" string? currently this breaks some templates
    if (is_val<value_string>(val)) {
        const auto & str_val = cast_val<value_string>(val)->val_str;
        parts->val_str.append(str_val);
    } else if (is_val<value_int>(val) || is_val<value_float>(val) || is_val<value_bool>(val)) {
        std::string str_val = val->as_string().str();
        parts->val_str.append(str_val);
    } else if (is_val<value_array>(val)) {
        auto items = cast_val<value_array>(val)->as_array();
        for (const auto & item : items) {
            gather_string_parts_recursive(item, parts);
        }
    }
}

static std::string render_string_parts(const value_string & parts) {
    std::ostringstream oss;
    for (const auto & part : parts->val_str.parts) {
        oss << part.val;
    }
    return oss.str();
}

struct runtime {
    context & ctx;
    explicit runtime(context & ctx) : ctx(ctx) {}

    value_array execute(const program & prog) {
        value_array results = mk_val<value_array>();
        for (const auto & stmt : prog.body) {
            value res = stmt->execute(ctx);
            results->push_back(std::move(res));
        }
        return results;
    }

    static value_string gather_string_parts(const value & val) {
        value_string parts = mk_val<value_string>();
        gather_string_parts_recursive(val, parts);
        // join consecutive parts with the same type
        auto & p = parts->val_str.parts;
        for (size_t i = 1; i < p.size(); ) {
            if (p[i].is_input == p[i - 1].is_input) {
                p[i - 1].val += p[i].val;
                p.erase(p.begin() + i);
            } else {
                i++;
            }
        }
        return parts;
    }
};

} // namespace jinja

#include <nlohmann/json_fwd.hpp.inc>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <string_view>
#include <functional>
#include <vector>
#include <variant>

struct common_grammar_builder;

class common_peg_parser_builder;

using common_peg_parser_id = size_t;
constexpr common_peg_parser_id COMMON_PEG_INVALID_PARSER_ID = static_cast<common_peg_parser_id>(-1);

using common_peg_ast_id = size_t;
constexpr common_peg_ast_id COMMON_PEG_INVALID_AST_ID = static_cast<common_peg_ast_id>(-1);

// Lightweight wrapper around common_peg_parser_id for convenience
class common_peg_parser {
    common_peg_parser_id id_;
    common_peg_parser_builder & builder_;

  public:
    common_peg_parser(const common_peg_parser & other) : id_(other.id_), builder_(other.builder_) {}
    common_peg_parser(common_peg_parser_id id, common_peg_parser_builder & builder) : id_(id), builder_(builder) {}

    common_peg_parser & operator=(const common_peg_parser & other);
    common_peg_parser & operator+=(const common_peg_parser & other);
    common_peg_parser & operator|=(const common_peg_parser & other);

    operator common_peg_parser_id() const { return id_; }
    common_peg_parser_id id() const { return id_; }

    common_peg_parser_builder & builder() const { return builder_; }

    // Creates a sequence
    common_peg_parser operator+(const common_peg_parser & other) const;

    // Creates a sequence separated by spaces.
    common_peg_parser operator<<(const common_peg_parser & other) const;

    // Creates a choice
    common_peg_parser operator|(const common_peg_parser & other) const;

    common_peg_parser operator+(const char * str) const;
    common_peg_parser operator+(const std::string & str) const;
    common_peg_parser operator<<(const char * str) const;
    common_peg_parser operator<<(const std::string & str) const;
    common_peg_parser operator|(const char * str) const;
    common_peg_parser operator|(const std::string & str) const;
};

common_peg_parser operator+(const char * str, const common_peg_parser & p);
common_peg_parser operator+(const std::string & str, const common_peg_parser & p);
common_peg_parser operator<<(const char * str, const common_peg_parser & p);
common_peg_parser operator<<(const std::string & str, const common_peg_parser & p);
common_peg_parser operator|(const char * str, const common_peg_parser & p);
common_peg_parser operator|(const std::string & str, const common_peg_parser & p);

enum common_peg_parse_result_type {
    COMMON_PEG_PARSE_RESULT_FAIL            = 0,
    COMMON_PEG_PARSE_RESULT_SUCCESS         = 1,
    COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT = 2,
};

const char * common_peg_parse_result_type_name(common_peg_parse_result_type type);

struct common_peg_ast_node {
    common_peg_ast_id id;
    std::string rule;
    std::string tag;
    size_t start;
    size_t end;
    std::string_view text;
    std::vector<common_peg_ast_id> children;

    bool is_partial = false;
};

struct common_peg_parse_result;

using common_peg_ast_visitor = std::function<void(const common_peg_ast_node & node)>;

class common_peg_ast_arena {
    std::vector<common_peg_ast_node> nodes_;
  public:
    common_peg_ast_id add_node(
        const std::string & rule,
        const std::string & tag,
        size_t start,
        size_t end,
        std::string_view text,
        std::vector<common_peg_ast_id> children,
        bool is_partial = false
    ) {
        common_peg_ast_id id = nodes_.size();
        nodes_.push_back({id, rule, tag, start, end, text, std::move(children), is_partial});
        return id;
    }

    const common_peg_ast_node & get(common_peg_ast_id id) const { return nodes_.at(id); }

    common_peg_ast_id find_by_tag(const common_peg_ast_node & parent, const std::string & tag, int max_depth = 3) const;
    common_peg_ast_id find_by_rule(const common_peg_ast_node & parent, const std::string & tag, int max_depth = 3) const;

    size_t size() const { return nodes_.size(); }

    void clear() { nodes_.clear(); }

    void visit(common_peg_ast_id id, const common_peg_ast_visitor & visitor) const;
    void visit(const common_peg_parse_result & result, const common_peg_ast_visitor & visitor) const;

    std::string dump();
};

struct common_peg_parse_result {
    common_peg_parse_result_type type = COMMON_PEG_PARSE_RESULT_FAIL;
    size_t start = 0;
    size_t end = 0;

    std::vector<common_peg_ast_id> nodes;

    common_peg_parse_result() = default;

    common_peg_parse_result(common_peg_parse_result_type type, size_t start)
        : type(type), start(start), end(start) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end)
        : type(type), start(start), end(end) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end, std::vector<common_peg_ast_id> nodes)
        : type(type), start(start), end(end), nodes(std::move(nodes)) {}

    bool fail() const { return type == COMMON_PEG_PARSE_RESULT_FAIL; }
    bool need_more_input() const { return type == COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT; }
    bool success() const { return type == COMMON_PEG_PARSE_RESULT_SUCCESS; }
};

enum common_peg_parse_flags {
    COMMON_PEG_PARSE_FLAG_NONE    = 0,
    COMMON_PEG_PARSE_FLAG_LENIENT = 1 << 0,
    COMMON_PEG_PARSE_FLAG_DEBUG   = 1 << 1,
};

inline common_peg_parse_flags operator|(common_peg_parse_flags a, common_peg_parse_flags b) {
    return static_cast<common_peg_parse_flags>(int(a) | int(b));
}

inline common_peg_parse_flags & operator|=(common_peg_parse_flags & a, common_peg_parse_flags b) {
    return a = a | b;
}

inline common_peg_parse_flags operator&(common_peg_parse_flags a, common_peg_parse_flags b) {
    return static_cast<common_peg_parse_flags>(int(a) & int(b));
}

inline common_peg_parse_flags operator~(common_peg_parse_flags a) {
    return static_cast<common_peg_parse_flags>(~int(a));
}

struct common_peg_parse_context {
    std::string input;
    common_peg_parse_flags flags;
    common_peg_ast_arena ast;

    int parse_depth;

    common_peg_parse_context(common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE)
        : flags(flags), parse_depth(0) {}

    common_peg_parse_context(const std::string & input, common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE)
        : input(input), flags(flags), parse_depth(0) {}

    bool is_lenient() const { return flags & COMMON_PEG_PARSE_FLAG_LENIENT; }
    bool is_debug() const { return flags & COMMON_PEG_PARSE_FLAG_DEBUG; }
};

class common_peg_arena;

// Parser variants
struct common_peg_epsilon_parser {};

struct common_peg_start_parser {};

struct common_peg_end_parser {};

struct common_peg_literal_parser {
    std::string literal;
};

struct common_peg_sequence_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_choice_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_repetition_parser {
    common_peg_parser_id child;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_peg_and_parser {
    common_peg_parser_id child;
};

struct common_peg_not_parser {
    common_peg_parser_id child;
};

struct common_peg_any_parser {};

struct common_peg_space_parser {};

struct common_peg_chars_parser {
    struct char_range {
        uint32_t start;
        uint32_t end;
        bool contains(uint32_t codepoint) const { return codepoint >= start && codepoint <= end; }
    };

    std::string pattern;
    std::vector<char_range> ranges;
    bool negated;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_peg_string_parser {
    char delimiter;
};

struct common_peg_until_parser {
    std::vector<std::string> delimiters;
};

struct common_peg_schema_parser {
    common_peg_parser_id child;
    std::string name;
    std::shared_ptr<nlohmann::ordered_json> schema;

    // Indicates if the GBNF should accept a raw string that matches the schema.
    bool raw;
};

struct common_peg_rule_parser {
    std::string name;
    common_peg_parser_id child;
    bool trigger;
};

struct common_peg_ref_parser {
    std::string name;
};

struct common_peg_atomic_parser {
    common_peg_parser_id child;
};

struct common_peg_tag_parser {
    common_peg_parser_id child;
    std::string tag;
};

struct common_peg_gbnf_parser {
    common_peg_parser_id child;
    std::string grammar;
};

// Variant holding all parser types
using common_peg_parser_variant = std::variant<
    common_peg_epsilon_parser,
    common_peg_start_parser,
    common_peg_end_parser,
    common_peg_literal_parser,
    common_peg_sequence_parser,
    common_peg_choice_parser,
    common_peg_repetition_parser,
    common_peg_and_parser,
    common_peg_not_parser,
    common_peg_any_parser,
    common_peg_space_parser,
    common_peg_chars_parser,
    common_peg_string_parser,
    common_peg_until_parser,
    common_peg_schema_parser,
    common_peg_rule_parser,
    common_peg_ref_parser,
    common_peg_atomic_parser,
    common_peg_tag_parser,
    common_peg_gbnf_parser
>;

class common_peg_arena {
    std::vector<common_peg_parser_variant> parsers_;
    std::unordered_map<std::string, common_peg_parser_id> rules_;
    common_peg_parser_id root_ = COMMON_PEG_INVALID_PARSER_ID;

  public:
    const common_peg_parser_variant & get(common_peg_parser_id id) const { return parsers_.at(id); }
    common_peg_parser_variant & get(common_peg_parser_id id) { return parsers_.at(id); }

    size_t size() const { return parsers_.size(); }
    bool empty() const { return parsers_.empty(); }

    common_peg_parser_id get_rule(const std::string & name) const;
    bool has_rule(const std::string & name) const { return rules_.find(name) != rules_.end(); }

    common_peg_parser_id root() const { return root_; }
    void set_root(common_peg_parser_id id) { root_ = id; }

    common_peg_parse_result parse(common_peg_parse_context & ctx, size_t start = 0) const;
    common_peg_parse_result parse(common_peg_parser_id id, common_peg_parse_context & ctx, size_t start) const;

    void resolve_refs();

    void build_grammar(const common_grammar_builder & builder, bool lazy = false) const;

    std::string dump(common_peg_parser_id id) const;

    nlohmann::json to_json() const;
    static common_peg_arena from_json(const nlohmann::json & j);

    std::string save() const;
    void load(const std::string & data);

    friend class common_peg_parser_builder;

  private:
    std::string dump_impl(common_peg_parser_id id, std::unordered_set<common_peg_parser_id> & visited) const;

    common_peg_parser_id add_parser(common_peg_parser_variant parser);
    void add_rule(const std::string & name, common_peg_parser_id id);

    common_peg_parser_id resolve_ref(common_peg_parser_id id);
};

class common_peg_parser_builder {
    common_peg_arena arena_;

    common_peg_parser wrap(common_peg_parser_id id) { return common_peg_parser(id, *this); }
    common_peg_parser add(const common_peg_parser_variant & p) { return wrap(arena_.add_parser(p)); }

  public:
    common_peg_parser_builder();

    // Match nothing, always succeed.
    //   S -> ε
    common_peg_parser eps() { return add(common_peg_epsilon_parser{}); }

    // Matches the start of the input.
    //   S -> ^
    common_peg_parser start() { return add(common_peg_start_parser{}); }

    // Matches the end of the input.
    //   S -> $
    common_peg_parser end() { return add(common_peg_end_parser{}); }

    // Matches an exact literal string.
    //   S -> "hello"
    common_peg_parser literal(const std::string & literal) { return add(common_peg_literal_parser{literal}); }

    // Matches a sequence of parsers in order, all must succeed.
    //   S -> A B C
    common_peg_parser sequence() { return add(common_peg_sequence_parser{}); }
    common_peg_parser sequence(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser sequence(const std::vector<common_peg_parser> & parsers);
    common_peg_parser sequence(std::initializer_list<common_peg_parser> parsers);

    // Matches the first parser that succeeds from a list of alternatives.
    //   S -> A | B | C
    common_peg_parser choice() { return add(common_peg_choice_parser{}); }
    common_peg_parser choice(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser choice(const std::vector<common_peg_parser> & parsers);
    common_peg_parser choice(std::initializer_list<common_peg_parser> parsers);

    // Matches one or more repetitions of a parser.
    //   S -> A+
    common_peg_parser one_or_more(const common_peg_parser & p) { return repeat(p, 1, -1); }

    // Matches zero or more repetitions of a parser, always succeeds.
    //   S -> A*
    common_peg_parser zero_or_more(const common_peg_parser & p) { return repeat(p, 0, -1); }

    // Matches zero or one occurrence of a parser, always succeeds.
    //   S -> A?
    common_peg_parser optional(const common_peg_parser & p) { return repeat(p, 0, 1); }

    // Positive lookahead: succeeds if child parser succeeds, consumes no input.
    //   S -> &A
    common_peg_parser peek(const common_peg_parser & p) { return add(common_peg_and_parser{p}); }

    // Negative lookahead: succeeds if child parser fails, consumes no input.
    //   S -> !A
    common_peg_parser negate(const common_peg_parser & p) { return add(common_peg_not_parser{p}); }

    // Matches any single character.
    //   S -> .
    common_peg_parser any() { return add(common_peg_any_parser{}); }

    // Matches between min and max repetitions of characters from a character class.
    //   S -> [a-z]{m,n}
    //
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser chars(const std::string & classes, int min = 1, int max = -1);

    // Creates a lightweight reference to a named rule (resolved during build()).
    // Use this for forward references in recursive grammars.
    //   expr_ref -> expr
    common_peg_parser ref(const std::string & name) { return add(common_peg_ref_parser{name}); }

    // Matches zero or more whitespace characters (space, tab, newline).
    //   S -> [ \t\n]*
    common_peg_parser space() { return add(common_peg_space_parser{}); }

    // Matches all characters until a delimiter is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_peg_parser until(const std::string & delimiter) { return add(common_peg_until_parser{{delimiter}}); }

    // Matches all characters until one of the delimiters in the list is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_peg_parser until_one_of(const std::vector<std::string> & delimiters) { return add(common_peg_until_parser{delimiters}); }

    // Matches everything
    //   S -> .*
    common_peg_parser rest() { return until_one_of({}); }

    // Matches between min and max repetitions of a parser (inclusive).
    //   S -> A{m,n}
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser repeat(const common_peg_parser & p, int min, int max) { return add(common_peg_repetition_parser{p, min,max}); }

    // Matches exactly n repetitions of a parser.
    //   S -> A{n}
    common_peg_parser repeat(const common_peg_parser & p, int n) { return repeat(p, n, n); }

    // Matches a double-quoted string: '"' content '"' space
    common_peg_parser double_quoted_string();

    // Matches a single-quoted string: "'" content "'" space
    common_peg_parser single_quoted_string();

    // Matches a string that accepts both double-quoted and single-quoted styles.
    common_peg_parser quoted_string();

    // Matches string content without the surrounding delimiter.
    common_peg_parser string_content(char delimiter);

    // Creates a complete JSON parser supporting objects, arrays, strings, numbers, booleans, and null.
    //   value -> object | array | string | number | true | false | null
    common_peg_parser json();
    common_peg_parser json_object();
    common_peg_parser json_string();
    common_peg_parser json_array();
    common_peg_parser json_number();
    common_peg_parser json_bool();
    common_peg_parser json_null();

    // Matches a JSON object member with a key and associated parser as the
    // value.
    common_peg_parser json_member(const std::string & key, const common_peg_parser & p);

    // Creates a complete Python format parser supporting dicts, arrays, strings, numbers, booleans, and None.
    // Differs from JSON: uses True/False/None, accepts both single and double-quoted strings.
    //   value -> dict | array | string | number | True | False | None
    common_peg_parser python_value();
    common_peg_parser python_dict();
    common_peg_parser python_string();
    common_peg_parser python_array();
    common_peg_parser python_number();
    common_peg_parser python_bool();
    common_peg_parser python_null();

    // A marker, i.e. text delimited by a pair of <> or []
    common_peg_parser marker();

    // Wraps a parser with JSON schema metadata for grammar generation.
    // Used internally to convert JSON schemas to GBNF grammar rules.
    common_peg_parser schema(const common_peg_parser & p, const std::string & name, const nlohmann::ordered_json & schema, bool raw = false);

    // Creates a named rule, stores it in the grammar, and returns a ref.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", json_obj | json_arr | ...)
    common_peg_parser rule(const std::string & name, const common_peg_parser & p, bool trigger = false);

    // Creates a named rule using a builder function, and returns a ref.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", [&]() { return json_object() | json_array() | ... })
    common_peg_parser rule(const std::string & name, const std::function<common_peg_parser()> & builder, bool trigger = false);

    // Creates a trigger rule. When generating a lazy grammar from the parser,
    // only trigger rules and descendents are emitted.
    common_peg_parser trigger_rule(const std::string & name, const common_peg_parser & p) { return rule(name, p, true); }
    common_peg_parser trigger_rule(const std::string & name, const std::function<common_peg_parser()> & builder) { return rule(name, builder, true); }

    // Creates an atomic parser. Atomic parsers do not create an AST node if
    // the child results in a partial parse, i.e. NEEDS_MORE_INPUT. This is
    // intended for situations where partial output is undesirable.
    common_peg_parser atomic(const common_peg_parser & p) { return add(common_peg_atomic_parser{p}); }

    // Tags create nodes in the generated AST for semantic purposes.
    // Unlike rules, you can tag multiple nodes with the same tag.
    common_peg_parser tag(const std::string & tag, const common_peg_parser & p) { return add(common_peg_tag_parser{p.id(), tag}); }

    // Wraps a child parser but emits a custom GBNF grammar string instead of
    // the child's grammar. Parsing delegates entirely to the child.
    common_peg_parser gbnf(const common_peg_parser & p, const std::string & grammar) { return add(common_peg_gbnf_parser{p, grammar}); }

    void set_root(const common_peg_parser & p);

    common_peg_arena build();
};

// Helper function for building parsers
common_peg_arena build_peg_parser(const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn);

// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

using json = nlohmann::ordered_json;

namespace jinja {
program parse_from_tokens(const lexer_result & lexer_res);

struct caps {
    bool supports_tools = true;
    bool supports_tool_calls = true;
    bool supports_system_role = true;
    bool supports_parallel_tool_calls = true;
    bool supports_preserve_reasoning = false; // support assistant message with reasoning_content

    // one of the 2 content capabilities must be true
    bool supports_string_content = true;
    bool supports_typed_content = false;

    bool supports_object_arguments = false;

    // for reporting on server
    std::map<std::string, bool> to_map() const;

    // for debugging
    std::string to_string() const;
};

caps caps_get(program & prog);
}

using chat_template_caps = jinja::caps;

struct common_chat_templates;

namespace autoparser {
struct generation_params;
}  // namespace autoparser

struct common_chat_tool_call {
    std::string name;
    std::string arguments;
    std::string id;

    bool operator==(const common_chat_tool_call & other) const {
        return name == other.name && arguments == other.arguments && id == other.id;
    }
};

struct common_chat_msg_content_part {
    std::string type;
    std::string text;

    // TODO @ngxson : no known chat templates support reasoning_content in content parts yet
    //                this can be useful for models with interleaved thinking (like Kimi-K2)
    //                if you see any templates explicitly support this, please ping me
    // std::string reasoning_content;

    bool operator==(const common_chat_msg_content_part & other) const {
        return type == other.type && text == other.text;
    }
};

struct common_chat_template {
    jinja::program prog;
    std::string bos_tok;
    std::string eos_tok;
    std::string src;
    chat_template_caps caps;

    common_chat_template(const std::string & src, const std::string & bos_token, const std::string & eos_token) {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(src);
        this->prog = jinja::parse_from_tokens(lexer_res);

        this->src = lexer_res.source;
        this->bos_tok = bos_token;
        this->eos_tok = eos_token;

        this->caps = jinja::caps_get(prog);
        // LOG_INF("%s: caps:\n%s\n", __func__, this->caps.to_string().c_str());
    }

    const std::string & source() const { return src; }
    const std::string & bos_token() const { return bos_tok; }
    const std::string & eos_token() const { return eos_tok; }

    chat_template_caps original_caps() const {
        return caps;
    }
};

struct common_chat_msg {
    std::string                               role;
    std::string                               content;
    std::vector<common_chat_msg_content_part> content_parts;
    std::vector<common_chat_tool_call>        tool_calls;
    std::string                               reasoning_content;
    std::string                               tool_name;
    std::string                               tool_call_id;

    nlohmann::ordered_json to_json_oaicompat(bool concat_typed_text = false) const;

    bool empty() const {
        return content.empty() && content_parts.empty() && tool_calls.empty() && reasoning_content.empty() &&
               tool_name.empty() && tool_call_id.empty();
    }

    void set_tool_call_ids(std::vector<std::string> &           ids_cache,
                           const std::function<std::string()> & gen_tool_call_id) {
        for (auto i = 0u; i < tool_calls.size(); i++) {
            if (ids_cache.size() <= i) {
                auto id = tool_calls[i].id;
                if (id.empty()) {
                    id = gen_tool_call_id();
                }
                ids_cache.push_back(id);
            }
            tool_calls[i].id = ids_cache[i];
        }
    }

    bool operator==(const common_chat_msg & other) const {
        return role == other.role && content == other.content && content_parts == other.content_parts &&
               tool_calls == other.tool_calls && reasoning_content == other.reasoning_content &&
               tool_name == other.tool_name && tool_call_id == other.tool_call_id;
    }

    bool operator!=(const common_chat_msg & other) const { return !(*this == other); }
};

struct common_chat_msg_diff {
    std::string           reasoning_content_delta;
    std::string           content_delta;
    size_t                tool_call_index = std::string::npos;
    common_chat_tool_call tool_call_delta;

    static std::vector<common_chat_msg_diff> compute_diffs(const common_chat_msg & msg_prv,
                                                           const common_chat_msg & msg_new);

    bool operator==(const common_chat_msg_diff & other) const {
        return content_delta == other.content_delta && tool_call_index == other.tool_call_index &&
               tool_call_delta == other.tool_call_delta;
    }
};

struct common_chat_tool {
    std::string name;
    std::string description;
    std::string parameters;
};

enum common_chat_tool_choice {
    COMMON_CHAT_TOOL_CHOICE_AUTO,
    COMMON_CHAT_TOOL_CHOICE_REQUIRED,
    COMMON_CHAT_TOOL_CHOICE_NONE,
};

enum common_chat_format {
    COMMON_CHAT_FORMAT_CONTENT_ONLY,

    // These are intended to be parsed by the PEG parser
    COMMON_CHAT_FORMAT_PEG_SIMPLE,
    COMMON_CHAT_FORMAT_PEG_NATIVE,
    COMMON_CHAT_FORMAT_PEG_GEMMA4,

    COMMON_CHAT_FORMAT_COUNT,  // Not a format, just the # formats
};

struct common_chat_templates_inputs {
    std::vector<common_chat_msg>          messages;
    std::string                           grammar;
    std::string                           json_schema;
    bool                                  add_generation_prompt = true;
    bool                                  use_jinja             = true;
    // Parameters below only supported when use_jinja is true
    std::vector<common_chat_tool>         tools;
    common_chat_tool_choice               tool_choice         = COMMON_CHAT_TOOL_CHOICE_AUTO;
    bool                                  parallel_tool_calls = false;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_NONE; // TODO: refactor this to "bool enable_thinking"
    bool                                  enable_thinking     = true;
    std::chrono::system_clock::time_point now                 = std::chrono::system_clock::now();
    std::map<std::string, std::string>    chat_template_kwargs;
    bool                                  add_bos = false;
    bool                                  add_eos = false;
    bool                                  force_pure_content = false;
};

struct common_chat_params {
    common_chat_format                  format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    std::string                         prompt;
    std::string                         grammar;
    bool                                grammar_lazy         = false;
    std::string                         generation_prompt;
    bool                                supports_thinking    = false;
    std::string                         thinking_start_tag;  // e.g., "<think>"
    std::string                         thinking_end_tag;    // e.g., "</think>"
    std::vector<common_grammar_trigger> grammar_triggers;
    std::vector<std::string>            preserved_tokens;
    std::vector<std::string>            additional_stops;
    std::string                         parser;
};

// per-message parsing syntax
// should be derived from common_chat_params
struct common_chat_parser_params {
    common_chat_format      format               = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format     = COMMON_REASONING_FORMAT_NONE; // TODO: refactor this to "bool parse_reasoning"
    // Whether reasoning_content should be inlined in the content (e.g. for reasoning_format=deepseek in stream mode)
    bool                    reasoning_in_content = false;
    std::string             generation_prompt;
    bool                    parse_tool_calls     = true;
    bool                    debug                = false;  // Enable debug output for PEG parser
    common_peg_arena        parser               = {};
    common_chat_parser_params() = default;
    common_chat_parser_params(const common_chat_params & chat_params) {
        format  = chat_params.format;
        generation_prompt = chat_params.generation_prompt;
    }
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool common_chat_verify_template(const std::string & tmpl, bool use_jinja);

void common_chat_templates_free(struct common_chat_templates * tmpls);

struct common_chat_templates_deleter {
    void operator()(common_chat_templates * tmpls) { common_chat_templates_free(tmpls); }
};

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override = "",
                                                     const std::string &        eos_token_override = "");

bool        common_chat_templates_was_explicit(const struct common_chat_templates * tmpls);
std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant = "");

struct common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                                      const struct common_chat_templates_inputs & inputs);

// Format single message, while taking into account the position of that message in chat history
std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass,
                                      bool                                 use_jinja);

// Returns an example of formatted chat
std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       bool                                       use_jinja,
                                       const std::map<std::string, std::string> & chat_template_kwargs);

const char *    common_chat_format_name(common_chat_format format);
common_chat_msg common_chat_parse(const std::string & input, bool is_partial, const common_chat_parser_params & params);
common_chat_msg common_chat_peg_parse(const common_peg_arena & src_parser, const std::string & input, bool is_partial, const common_chat_parser_params & params);

// used by arg and server
const char *            common_reasoning_format_name(common_reasoning_format format);
common_reasoning_format common_reasoning_format_from_name(const std::string & format);

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice);

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates);

// Parses a JSON array of messages in OpenAI's chat completion API format.
std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const nlohmann::ordered_json & messages);

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const nlohmann::ordered_json & tools);

// DEPRECATED: only used in tests
nlohmann::ordered_json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text = false);

nlohmann::ordered_json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools);

// get template caps, useful for reporting to server /props endpoint
std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates);

std::string common_chat_template_direct_apply(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs);

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &          tmpl,
        const std::string &                   src,
        autoparser::generation_params & params);

// specialized per-task preset
struct common_chat_prompt_preset {
    std::string system;
    std::string user;
};

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates);

class common_chat_peg_mapper {
  public:
    common_chat_msg & result;

    common_chat_peg_mapper(common_chat_msg & msg) : result(msg) {}

    virtual ~common_chat_peg_mapper() = default;

    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
    virtual void map(const common_peg_ast_node & node);

  protected:
    virtual std::string normalize_container_value(const std::string & input);

  private:
    std::optional<common_chat_tool_call> pending_tool_call;
    common_chat_tool_call *              current_tool          = nullptr;
    int                                  arg_count             = 0;
    bool                                 closing_quote_pending = false;
    std::string                          args_buffer;

    std::string & args_target();
};

class common_chat_peg_gemma4_mapper : public common_chat_peg_mapper {
  public:
    common_chat_peg_gemma4_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}
    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);

  private:
    void visit(const common_peg_ast_arena & arena, common_peg_ast_id id);
};

struct content_structure;
struct tool_call_structure;

class common_chat_peg_builder : public common_peg_parser_builder {
  public:
    static constexpr const char * REASONING_BLOCK = "reasoning-block";
    static constexpr const char * REASONING       = "reasoning";
    static constexpr const char * CONTENT         = "content";

    static constexpr const char * TOOL           = "tool";
    static constexpr const char * TOOL_OPEN      = "tool-open";
    static constexpr const char * TOOL_CLOSE     = "tool-close";
    static constexpr const char * TOOL_ID        = "tool-id";
    static constexpr const char * TOOL_NAME      = "tool-name";
    static constexpr const char * TOOL_ARGS      = "tool-args";
    static constexpr const char * TOOL_ARG       = "tool-arg";
    static constexpr const char * TOOL_ARG_OPEN  = "tool-arg-open";
    static constexpr const char * TOOL_ARG_CLOSE = "tool-arg-close";
    static constexpr const char * TOOL_ARG_NAME         = "tool-arg-name";
    static constexpr const char * TOOL_ARG_VALUE        = "tool-arg-value";
    static constexpr const char * TOOL_ARG_STRING_VALUE = "tool-arg-string-value";

    common_peg_parser reasoning_block(const common_peg_parser & p) { return tag(REASONING_BLOCK, p); }
    common_peg_parser reasoning(const common_peg_parser & p) { return tag(REASONING, p); }
    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }

    common_peg_parser tag_with_safe_content(const std::string &       tag_name,
                                            const std::string &       marker,
                                            const common_peg_parser & p);

    common_peg_parser tool(const common_peg_parser & p) { return tag(TOOL, p); }
    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag(TOOL_OPEN, p)); }
    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag(TOOL_CLOSE, p)); }
    common_peg_parser tool_id(const common_peg_parser & p) { return atomic(tag(TOOL_ID, p)); }
    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag(TOOL_NAME, p)); }
    common_peg_parser tool_args(const common_peg_parser & p) { return tag(TOOL_ARGS, p); }
    common_peg_parser tool_arg(const common_peg_parser & p) { return tag(TOOL_ARG, p); }
    common_peg_parser tool_arg_open(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_OPEN, p)); }
    common_peg_parser tool_arg_close(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_CLOSE, p)); }
    common_peg_parser tool_arg_name(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_NAME, p)); }
    common_peg_parser tool_arg_value(const common_peg_parser & p) { return tag(TOOL_ARG_VALUE, p); }
    common_peg_parser tool_arg_string_value(const common_peg_parser & p) { return tag(TOOL_ARG_STRING_VALUE, p); }
    common_peg_parser tool_arg_json_value(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_VALUE, p)); }

    common_peg_parser prefix(const std::string & s, const std::string & delimiter = {});

    common_peg_parser standard_json_tools(const std::string &              section_start,
                                          const std::string &              section_end,
                                          const nlohmann::ordered_json &   tools,
                                          bool                             parallel_tool_calls,
                                          bool                             force_tool_calls,
                                          const std::string &              name_key = "",
                                          const std::string &              args_key = "",
                                          bool                             array_wrapped = false,
                                          bool                             function_is_key = false,
                                          const std::string &              call_id_key = "",
                                          const std::string &              gen_call_id_key = "",
                                          const std::vector<std::string> & parameters_order = {});

    common_peg_parser standard_constructed_tools(const std::map<std::string, std::string> & markers,
                                                 const nlohmann::ordered_json &             tools,
                                                 bool                                       parallel_tool_calls,
                                                 bool                                       force_tool_calls);

    common_peg_parser python_style_tool_calls(const nlohmann::ordered_json & tools,
                                              bool                           parallel_tool_calls);

  private:
    common_peg_parser build_json_tools_function_is_key(const nlohmann::ordered_json & tools,
                                                       const std::string &            args_key,
                                                       const std::string &            effective_args_key,
                                                       const std::string &            call_id_key,
                                                       const std::string &            gen_call_id_key);

    common_peg_parser build_json_tools_nested_keys(const nlohmann::ordered_json & tools,
                                                   const std::string &            effective_name_key,
                                                   const std::string &            effective_args_key,
                                                   const std::string &            call_id_key,
                                                   const std::string &            gen_call_id_key);

    common_peg_parser build_json_tools_flat_keys(const nlohmann::ordered_json &   tools,
                                                 const std::string &              effective_name_key,
                                                 const std::string &              effective_args_key,
                                                 const std::string &              call_id_key,
                                                 const std::string &              gen_call_id_key,
                                                 const std::vector<std::string> & parameters_order);
};

inline common_peg_arena build_chat_peg_parser(
        const std::function<common_peg_parser(common_chat_peg_builder & builder)> & fn) {
    common_chat_peg_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

class tag_based_peg_mapper {
  public:
    std::map<std::string, std::string> tags;

    void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
};

struct tagged_parse_result {
    common_peg_parse_result            result;
    std::map<std::string, std::string> tags;
};

struct tagged_peg_parser {
    common_peg_arena arena;
    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE;

    tagged_peg_parser & withDebug() {
        flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
        return *this;
    }

    tagged_peg_parser & withoutDebug() {
        flags = flags & ~COMMON_PEG_PARSE_FLAG_DEBUG;
        return *this;
    }

    tagged_parse_result parse_and_extract(const std::string & input, common_peg_parse_flags extra_flags = COMMON_PEG_PARSE_FLAG_NONE) const;
    tagged_parse_result parse_anywhere_and_extract(const std::string & input) const;
};

tagged_peg_parser build_tagged_peg_parser(
    const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn);

// ============================================================================
// Parameters for template application (low-level, used by diff analysis)
// ============================================================================
struct template_params {
    json                messages;
    json                tools;
    bool                add_generation_prompt = false;
    bool                enable_thinking       = true;
    std::optional<json> extra_context         = std::nullopt;
};

struct diff_split {
    std::string prefix;
    std::string suffix;
    std::string left;
    std::string right;

    bool operator==(struct diff_split & other) const {
        return prefix == other.prefix && suffix == other.suffix && left == other.left && right == other.right;
    }
};

// Result of compare_variants containing diff and original outputs
struct compare_variants_result {
    diff_split  diff;
    std::string output_A;
    std::string output_B;
};

std::string trim_whitespace(const std::string & str);
std::string trim_leading_whitespace(const std::string & str);
std::string trim_trailing_whitespace(const std::string & str);
std::string trim_trailing_newlines(const std::string & str);

diff_split calculate_diff_split(const std::string & left, const std::string & right);
std::string until_common_prefix(const std::string & full, const std::string & left, const std::string & right);
std::string after_common_suffix(const std::string & full, const std::string & left, const std::string & right);

namespace autoparser {

std::string apply_template(const common_chat_template & tmpl, const template_params & params);

std::optional<compare_variants_result> compare_variants(
    const common_chat_template &                   tmpl,
    const template_params &                        params_A,
    const std::function<void(template_params &)> & params_modifier);

// ============================================================================
// High-level params for parser generation
// ============================================================================

struct generation_params {
    json                                  messages;
    json                                  tools;
    common_chat_tool_choice               tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
    json                                  json_schema;
    bool                                  parallel_tool_calls = true;
    common_reasoning_format               reasoning_format    = COMMON_REASONING_FORMAT_AUTO;
    bool                                  stream              = true;
    std::string                           grammar;
    bool                                  add_generation_prompt = false;
    bool                                  enable_thinking       = true;
    std::chrono::system_clock::time_point now                   = std::chrono::system_clock::now();
    std::string                           generation_prompt;
    json                                  extra_context;
    bool                                  add_bos       = false;
    bool                                  add_eos       = false;
    bool                                  is_inference  = true;
    bool                                  add_inference = false;
    bool                                  mark_input    = true;  // whether to mark input strings in the jinja context
};

// ============================================================================
// Analysis Result Enums
// ============================================================================

// Reasoning handling mode (derived from R1-R3 comparisons)
enum class reasoning_mode {
    NONE,           // No reasoning markers detected
    TAG_BASED,      // Tag-based: <think>...</think> (start can be empty for delimiter-style)
    TOOLS_ONLY      // Only reason on tool calls, not on normal content
};

inline std::ostream & operator<<(std::ostream & os, const reasoning_mode & mode) {
    switch (mode) {
        case reasoning_mode::NONE:
            return os << "NONE";
        case reasoning_mode::TAG_BASED:
            return os << "TAG_BASED";
        case reasoning_mode::TOOLS_ONLY:
            return os << "TOOLS_ONLY";
        default:
            return os << "UNKNOWN";
    }
}

// Content wrapping mode (derived from C1 comparison)
enum class content_mode {
    PLAIN,                   // No content markers
    ALWAYS_WRAPPED,          // Content always wrapped with markers
    WRAPPED_WITH_REASONING,  // Content wrapped only when reasoning present
};

inline std::ostream & operator<<(std::ostream & os, const content_mode & mode) {
    switch (mode) {
        case content_mode::PLAIN:
            return os << "PLAIN";
        case content_mode::ALWAYS_WRAPPED:
            return os << "ALWAYS_WRAPPED";
        case content_mode::WRAPPED_WITH_REASONING:
            return os << "WRAPPED_WITH_REASONING";
        default:
            return os << "UNKNOWN";
    }
}

// Call ID position in tool calls (for non-JSON formats)
enum class call_id_position {
    NONE,                   // No call ID support detected
    PRE_FUNC_NAME,          // Call ID before function name: [CALL_ID]id[FUNC]name{args}
    BETWEEN_FUNC_AND_ARGS,  // Call ID between function and args: [FUNC]name[CALL_ID]id{args}
    POST_ARGS,              // Call ID after arguments: [FUNC]name{args}[CALL_ID]id
};

inline std::ostream & operator<<(std::ostream & os, const call_id_position & pos) {
    switch (pos) {
        case call_id_position::NONE:
            return os << "NONE";
        case call_id_position::PRE_FUNC_NAME:
            return os << "PRE_FUNC_NAME";
        case call_id_position::BETWEEN_FUNC_AND_ARGS:
            return os << "BETWEEN_FUNC_AND_ARGS";
        case call_id_position::POST_ARGS:
            return os << "POST_ARGS";
        default:
            return os << "UNKNOWN";
    }
}

// Tool call format classification (derived from T1-T5, A1-A3 comparisons)
enum class tool_format {
    NONE,             // No tool support detected
    JSON_NATIVE,      // Pure JSON: {"name": "X", "arguments": {...}}
    TAG_WITH_JSON,    // Tag-based with JSON args: <function=X>{...}</function>
    TAG_WITH_TAGGED,  // Tag-based with tagged args: <param=key>value</param>
};

inline std::ostream & operator<<(std::ostream & os, const tool_format & format) {
    switch (format) {
        case tool_format::NONE:
            return os << "NONE";
        case tool_format::JSON_NATIVE:
            return os << "JSON_NATIVE";
        case tool_format::TAG_WITH_JSON:
            return os << "TAG_WITH_JSON";
        case tool_format::TAG_WITH_TAGGED:
            return os << "TAG_WITH_TAGGED";
        default:
            return os << "UNKNOWN";
    }
}

// ============================================================================
// Sub-structs for tool analysis
// ============================================================================

struct tool_format_analysis {
    tool_format mode = tool_format::NONE;

    std::string section_start;   // e.g., "<tool_call>", "[TOOL_CALLS]", ""
    std::string section_end;     // e.g., "</tool_call>", ""
    std::string per_call_start;  // e.g., "<|tool_call_begin|>", "" (for multi-call templates)
    std::string per_call_end;    // e.g., "<|tool_call_end|>", ""

    bool fun_name_is_key = false;       // In JSON format function name is JSON key, i.e. { "<funname>": { ... arguments ... } }
    bool tools_array_wrapped = false;   // Tool calls wrapped in JSON array [...]

    std::string              function_field = "function";
    std::string              name_field     = "name";
    std::string              args_field     = "arguments";
    std::string              id_field;
    std::string              gen_id_field;
    std::vector<std::string> parameter_order;
};

struct tool_function_analysis {
    std::string name_prefix;  // e.g., "<function=", "\"name\": \"", "functions."
    std::string name_suffix;  // e.g., ">", "\"", ":0"
    std::string close;        // e.g., "</function>", "" (for tag-based)
};

struct tool_arguments_analysis {
    std::string start;          // e.g., "<|tool_call_argument_begin|>", "<args>"
    std::string end;            // e.g., "<|tool_call_argument_end|>", "</args>"
    std::string name_prefix;   // e.g., "<param=", "<arg_key>", "\""
    std::string name_suffix;   // e.g., ">", "</arg_key>", "\":"
    std::string value_prefix;  // e.g., "", "<arg_value>", ""
    std::string value_suffix;  // e.g., "</param>", "</arg_value>", ""
    std::string separator;     // e.g., "", "\n", ","
};

struct tool_id_analysis {
    call_id_position pos = call_id_position::NONE;

    std::string prefix;  // e.g., "[CALL_ID]" (marker before call ID value)
    std::string suffix;  // e.g., "" (marker after call ID value, before next section)
};

// ============================================================================
// Parser build context (shared interface for build_parser methods)
// ============================================================================

struct analyze_content;
struct analyze_reasoning;

struct parser_build_context {
    common_chat_peg_builder & p;
    const generation_params &         inputs;
    common_peg_parser                 reasoning_parser;
    bool                              extracting_reasoning = false;
    const analyze_reasoning *         reasoning            = nullptr;
    const analyze_content *           content              = nullptr;

    parser_build_context(common_chat_peg_builder & p, const generation_params & inputs);
};

// ============================================================================
// Base class for analyzers with parser building
// ============================================================================

struct analyze_base {
    virtual ~analyze_base() = default;
    virtual common_peg_parser build_parser(parser_build_context & ctx) const = 0;

  protected:
    const common_chat_template * tmpl = nullptr;

    analyze_base() = default;
    explicit analyze_base(const common_chat_template & tmpl) : tmpl(&tmpl) {}
};

// ============================================================================
// Reasoning analyzer
// ============================================================================

struct analyze_reasoning : analyze_base {
    reasoning_mode mode = reasoning_mode::NONE;

    std::string start;  // e.g., "<think>", "[THINK]", "<|START_THINKING|>", ""
    std::string end;    // e.g., "</think>", "[BEGIN FINAL RESPONSE]", "<|END_THINKING|>"

    analyze_reasoning() = default;
    analyze_reasoning(const common_chat_template & tmpl, bool supports_tools);
    analyze_reasoning(std::string start_, std::string end_) : start(std::move(start_)), end(std::move(end_)) {}

    common_peg_parser build_parser(parser_build_context & ctx) const override;

  private:
    // Look for reasoning markers in rendered content
    void compare_reasoning_presence();

    // Compare generation prompt with enable_thinking=true vs false
    void compare_thinking_enabled();

    // Check if reasoning is always possible or only in tool calls
    void compare_reasoning_scope();
};

// ============================================================================
// Content analyzer
// ============================================================================

struct analyze_content : analyze_base {
    content_mode mode = content_mode::PLAIN;

    std::string start;  // e.g., "<response>", ">>>all\n", ""
    std::string end;    // e.g., "</response>", ""

    bool requires_nonnull_content = false;

    analyze_content() = default;
    analyze_content(const common_chat_template & tmpl, const analyze_reasoning & reasoning);

    common_peg_parser build_parser(parser_build_context & ctx) const override;

    bool is_always_wrapped() const;
    common_peg_parser build_optional_wrapped(parser_build_context & ctx) const;
};

// ============================================================================
// Tool analyzer
// ============================================================================

struct analyze_tools : analyze_base {
    tool_format_analysis    format;
    tool_function_analysis  function;
    tool_arguments_analysis arguments;
    tool_id_analysis        call_id;

    analyze_tools() = default;
    analyze_tools(const common_chat_template & tmpl,
                  const jinja::caps &          caps,
                  const analyze_reasoning &    reasoning);

    common_peg_parser build_parser(parser_build_context & ctx) const override;

  private:
    // Extract tool calling 'haystack' for further analysis and delegate further analysis based on format
    void analyze_tool_calls(const analyze_reasoning & reasoning, bool supports_parallel_tool_calls);

    // Analyze format based on position of function and argument name in needle
    void analyze_tool_call_format(const std::string &       haystack,
                                  const std::string &       fun_name_needle,
                                  const std::string &       arg_name_needle,
                                  const analyze_reasoning & reasoning,
                                  bool                      supports_parallel_tool_calls);

    // Analyze specifics of JSON native format (entire tool call is a JSON object)
    void analyze_tool_call_format_json_native(const std::string & clean_haystack,
                                              const std::string & fun_name_needle,
                                              const std::string & arg_name_needle);

    // Check if parallel calls in JSON native format array wrapped or tag wrapped
    void analyze_json_native_parallel_calls();

    // Analyze specifics of non-JSON native format (tags for function name or for function name and arguments)
    void analyze_tool_call_format_non_json(const std::string & clean_haystack,
                                           const std::string & fun_name_needle);

    // Check for and extract specific per-call markers for non-native-JSON templates with parallel call support
    void check_per_call_markers();

    // Extract function name markers
    void extract_function_markers();

    // Delegates to separate functions for: separator analysis, argument name analysis, argument value analysis
    void analyze_arguments();

    // Extract argument name markers
    void extract_argument_name_markers();

    // Extract argument value markers
    void extract_argument_value_markers();

    // Extract argument separator, if specified (eg. <arg=foo>...</arg><sep><arg=bar>...</arg>)
    void extract_argument_separator();

    // Extract argument wrapper markers, if present (eg. '<args><arg=foo>...</arg><arg=bar>...</arg></args>')
    void extract_args_markers();

    // Extract call ID markers, if present
    void extract_call_id_markers();

    // Per-format tool parser builders
    common_peg_parser build_tool_parser_json_native(parser_build_context & ctx) const;
    common_peg_parser build_tool_parser_tag_json(parser_build_context & ctx) const;
    common_peg_parser build_tool_parser_tag_tagged(parser_build_context & ctx) const;

    // Shared helper: builds func_parser from open+call_id+args, handling atomic wrapping and close.
    // atomic_peek: if present, used as the peek expression in the third atomicity branch.
    common_peg_parser build_func_parser(common_chat_peg_builder & p, const std::string & name,
                                        const common_peg_parser & call_id_section, bool have_call_id,
                                        const common_peg_parser & args,
                                        std::optional<common_peg_parser> atomic_peek) const;
};

// ============================================================================
// Main autoparser class
// ============================================================================

struct autoparser {
    jinja::caps          jinja_caps;
    analyze_reasoning    reasoning;
    analyze_content      content;
    analyze_tools        tools;
    bool                 analysis_complete = false;

    // Preserved tokens for tokenizer (union of all non-empty markers)
    std::vector<std::string> preserved_tokens;

    autoparser() = default;

    // Run full differential analysis on a template
    void analyze_template(const common_chat_template & tmpl);

    // Build the PEG parser for this template
    common_peg_arena build_parser(const generation_params & inputs) const;

  private:
    // Collect tokens from entire analysis to preserve
    void collect_preserved_tokens();
};

// ============================================================================
// Parser generator
// ============================================================================

class peg_generator {
  public:
    static common_chat_params generate_parser(const common_chat_template &    tmpl,
                                              const struct generation_params & inputs);

    static common_chat_params generate_parser(const common_chat_template &    tmpl,
                                              const struct generation_params & inputs,
                                              const autoparser &              autoparser);
};

}  // namespace autoparser

enum segment_type { TEXT, MARKER };

inline std::ostream & operator<<(std::ostream & os, const segment_type & type) {
    switch (type) {
        case segment_type::TEXT:
            return os << "TEXT";
        case segment_type::MARKER:
            return os << "MARKER";
        default:
            return os << "UNKNOWN";
    }
}

struct segment {
    segment_type type;
    std::string  value;

    segment(segment_type type, std::string value) : type(type), value(std::move(value)) {}

    bool operator==(const segment & other) const {
        return type == other.type && value == other.value;
    }

    bool operator!=(const segment & other) const {
        return !(*this == other);
    }
};

std::vector<segment> segmentize_markers(const std::string & text);
std::vector<segment> prune_whitespace_segments(const std::vector<segment> & segments);
