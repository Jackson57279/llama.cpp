#include "common.h.inc"
#include "llama.h.inc"
#include "mtmd.h.inc"
#include "common.h.inc"

#include "common.h.inc"
#include "llama.h.inc"
#include "mtmd.h.inc"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp.inc>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <cinttypes>

using json = nlohmann::ordered_json;

struct server_rust_string_view {
    const uint8_t * data;
    size_t len;
};

struct server_rust_sse_event_view {
    server_rust_string_view event;
    server_rust_string_view data;
};

struct server_rust_anthropic_sse_event_view {
    server_rust_string_view event;
    server_rust_string_view data;
    uint8_t has_event;
};

struct server_rust_result_timings_view {
    int32_t cache_n;
    int32_t prompt_n;
    double prompt_ms;
    double prompt_per_token_ms;
    double prompt_per_second;
    int32_t predicted_n;
    double predicted_ms;
    double predicted_per_token_ms;
    double predicted_per_second;
    int32_t draft_n;
    int32_t draft_n_accepted;
};

extern "C" bool llama_server_lora_all_alora_rust(const float * scales, const uint8_t * is_alora, size_t len);
extern "C" bool llama_server_lora_should_clear_cache_rust(
        bool current_has_enabled,
        bool current_all_alora,
        bool next_all_alora);
extern "C" bool llama_server_lora_are_equal_rust(
        const float * left_scales, const uintptr_t * left_ptrs, size_t left_len,
        const float * right_scales, const uintptr_t * right_ptrs, size_t right_len);
extern "C" size_t llama_server_lora_enabled_ids_rust(
        const float * scales, size_t len,
        size_t * out, size_t out_len);
extern "C" bool llama_server_json_flags_all_numbers_rust(const uint8_t * flags, size_t len);
extern "C" bool llama_server_json_flags_mixed_numbers_strings_rust(const uint8_t * flags, size_t len);
extern "C" bool llama_server_json_flags_contains_number_rust(const uint8_t * flags, size_t len);
extern "C" uint8_t * llama_server_format_token_output_rust(
        const uint8_t * data, size_t len,
        size_t * out_len);
extern "C" uint8_t * llama_server_format_oai_sse_rust(
        const server_rust_string_view * items, size_t len,
        size_t * out_len);
extern "C" uint8_t * llama_server_format_oai_resp_sse_rust(
        const server_rust_sse_event_view * items, size_t len,
        size_t * out_len);
extern "C" uint8_t * llama_server_format_anthropic_sse_rust(
        const server_rust_anthropic_sse_event_view * items, size_t len,
        size_t * out_len);
extern "C" size_t llama_server_token_common_prefix_rust(
        const llama_token * left, size_t left_len,
        const llama_token * right, size_t right_len);
extern "C" size_t llama_server_filter_text_tokens_rust(
        const llama_token * input, size_t input_len,
        llama_token * out, size_t out_len);
extern "C" uint8_t * llama_server_tokens_debug_string_rust(
        const llama_token * tokens, size_t tokens_len,
        const size_t * media_indices, size_t media_indices_len,
        size_t * out_len);
extern "C" bool llama_server_tokens_valid_range_rust(
        const llama_token * tokens, size_t tokens_len,
        int32_t n_vocab);
extern "C" int32_t llama_server_tokens_pos_next_rust(
        size_t token_len,
        int64_t n_tokens,
        const size_t * media_indices,
        const int32_t * media_n_pos,
        const size_t * media_n_tokens,
        size_t media_len);
extern "C" size_t llama_server_tokens_size_up_to_pos_rust(
        size_t token_len,
        int32_t max_pos,
        const size_t * media_indices,
        const int32_t * media_n_pos,
        const size_t * media_n_tokens,
        size_t media_len);
extern "C" bool llama_server_task_need_embd_rust(int32_t task_type);
extern "C" bool llama_server_task_need_logits_rust(int32_t task_type);
extern "C" bool llama_server_task_need_sampling_rust(int32_t task_type);
extern "C" bool llama_server_task_is_cancel_rust(int32_t task_type);
extern "C" bool llama_server_task_is_completion_rust(int32_t task_type);
extern "C" bool llama_server_task_is_embedding_rust(int32_t task_type);
extern "C" bool llama_server_task_is_rerank_rust(int32_t task_type);
extern "C" bool llama_server_task_is_completion_or_infill_rust(int32_t task_type);
extern "C" bool llama_server_response_type_is_oai_embd_rust(int32_t response_type);
extern "C" bool llama_server_response_type_is_non_native_rust(int32_t response_type);
extern "C" bool llama_server_response_type_is_oai_chat_or_cmpl_rust(int32_t response_type);
extern "C" bool llama_server_response_type_is_anthropic_rust(int32_t response_type);
extern "C" bool llama_server_response_type_is_oai_resp_rust(int32_t response_type);
extern "C" bool llama_server_response_type_stream_done_is_empty_rust(int32_t response_type);
extern "C" bool llama_server_task_is_parent_rust(size_t child_task_count);
extern "C" bool llama_server_task_is_child_rust(int32_t id_parent);
extern "C" size_t llama_server_unique_task_ids_rust(
        const int32_t * ids, size_t ids_len,
        int32_t * out, size_t out_len);
extern "C" bool llama_server_slot_is_processing_rust(int32_t slot_state);
extern "C" bool llama_server_slot_has_budget_rust(
        int32_t task_n_predict,
        int32_t global_n_predict,
        int32_t n_decoded,
        int32_t * out_n_remaining);
extern "C" bool llama_server_slot_can_split_rust(
        bool need_embd,
        bool has_memory,
        bool pooling_is_last);
extern "C" int32_t llama_server_slot_n_draft_max_rust(
        bool has_spec,
        int32_t n_draft_min,
        int32_t n_draft_max,
        int32_t n_ctx,
        int32_t prompt_tokens,
        int32_t n_remaining);
extern "C" bool llama_server_slot_can_batch_with_rust(
        int32_t task_type,
        int32_t other_task_type,
        bool lora_equal);
extern "C" size_t llama_server_prompt_size_rust(
        size_t data_len,
        const size_t * checkpoint_sizes,
        size_t checkpoint_count);
extern "C" size_t llama_server_sum_usize_rust(const size_t * values, size_t len);
extern "C" size_t llama_server_tokens_len_rust(size_t token_count);
extern "C" bool llama_server_tokens_is_empty_rust(size_t token_count);
extern "C" int32_t llama_server_tokens_len_i32_rust(size_t token_count);
extern "C" bool llama_server_json_field_is_array_rust(bool has_key, bool is_array);
extern "C" bool llama_server_json_field_is_string_rust(bool has_key, bool is_string);
extern "C" bool llama_server_queue_should_process_rust(bool running, bool queue_empty);
extern "C" bool llama_server_queue_sleep_wait_done_rust(bool running, bool req_stop_sleeping);
extern "C" bool llama_server_queue_task_wait_done_rust(bool running, bool queue_empty);
extern "C" bool llama_server_queue_should_sleep_rust(
        int64_t idle_sleep_ms,
        int64_t now_ms,
        int64_t time_last_task_ms);
extern "C" bool llama_server_response_reader_has_next_rust(
        bool cancelled,
        size_t received_count,
        size_t task_count);
extern "C" server_rust_result_timings_view llama_server_slot_timings_rust(
        int32_t cache_n,
        int32_t prompt_n,
        double prompt_ms,
        int32_t predicted_n,
        double predicted_ms,
        int32_t draft_n,
        int32_t draft_n_accepted);
extern "C" bool llama_server_score_desc_rust(double left, double right);
extern "C" bool llama_server_logit_desc_rust(float left, float right);
extern "C" bool llama_server_line_start_desc_rust(int32_t left, int32_t right);
extern "C" bool llama_server_id_matches_rust(int32_t left, int32_t right);
extern "C" bool llama_server_http_res_is_stream_rust(bool has_next);
extern "C" bool llama_server_checkpoint_before_threshold_rust(int32_t pos_min, int32_t pos_min_threshold);
extern "C" bool llama_server_model_status_is_ready_rust(int32_t status);
extern "C" bool llama_server_model_status_is_running_rust(int32_t status);
extern "C" bool llama_server_model_status_is_failed_rust(int32_t status, int32_t exit_code);

#define SLT_INF(slot, fmt, ...) LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT(""                                 fmt,                                                                __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...) LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...) LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)
#define SLT_DBG(slot, fmt, ...) LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).id, ((slot).task ? (slot).task->id : -1), __VA_ARGS__)

#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

using raw_buffer = std::vector<uint8_t>;

static std::vector<uint8_t> json_array_type_flags(const json & data) {
    std::vector<uint8_t> flags;
    if (!data.is_array()) {
        return flags;
    }
    flags.reserve(data.size());
    for (const auto & e : data) {
        flags.push_back(e.is_number_integer() ? 1 : (e.is_string() ? 2 : 0));
    }
    return flags;
}

// generator-like API for HTTP response generation
// this object response with one of the 2 modes:
// 1) normal response: `data` contains the full response body
// 2) streaming response: each call to next(output) generates the next chunk
//    when next(output) returns false, no more data after the current chunk
//    note: some chunks can be empty, in which case no data is sent for that chunk
struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string data;
    std::map<std::string, std::string> headers;

    // TODO: move this to a virtual function once we have proper polymorphism support
    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return llama_server_http_res_is_stream_rust(next != nullptr);
    }

    virtual ~server_http_res() = default;
};

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_http_res_ptr = std::unique_ptr<server_http_res>;

struct uploaded_file {
    raw_buffer data;
    std::string filename;
    std::string content_type;
};

struct server_http_req {
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> headers; // used by MCP proxy
    std::string path;
    std::string query_string; // query parameters string (e.g. "action=save")
    std::string body;
    std::map<std::string, uploaded_file> files; // used for file uploads (form data)
    const std::function<bool()> & should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        return def;
    }
};

struct server_http_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread; // server thread
    std::atomic<bool> is_ready = false;

    std::string path_prefix;
    std::string hostname;
    int port;

    server_http_context();
    ~server_http_context();

    bool init(const common_params & params);
    bool start();
    void stop() const;

    // note: the handler should never throw exceptions
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;

    void get(const std::string & path, const handler_t & handler) const;
    void post(const std::string & path, const handler_t & handler) const;

    // for debugging
    std::string listening_address;
};

/**
 * state diagram:
 *
 * UNLOADED ──► LOADING ──► LOADED ◄──── SLEEPING
 *  ▲            │            │               ▲
 *  └───failed───┘            │               │
 *  ▲                         └──sleeping─────┘
 *  └────────unloaded─────────┘
 */
enum server_model_status {
    // TODO: also add downloading state when the logic is added
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED,
    SERVER_MODEL_STATUS_SLEEPING
};

struct server_model_meta {
    common_preset preset;
    std::string name;
    std::set<std::string> aliases; // additional names that resolve to this model
    std::set<std::string> tags;    // informational tags, not used for routing
    int port = 0;
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
    int64_t last_used = 0; // for LRU unloading
    std::vector<std::string> args; // args passed to the model instance, will be populated by render_args()
    int exit_code = 0; // exit code of the model instance process (only valid if status == FAILED)
    int stop_timeout = 0; // seconds to wait before force-killing the model instance during shutdown

    bool is_ready() const {
        return llama_server_model_status_is_ready_rust(status);
    }

    bool is_running() const {
        return llama_server_model_status_is_running_rust(status);
    }

    bool is_failed() const {
        return llama_server_model_status_is_failed_rust(status, exit_code);
    }

    void update_args(common_preset_context & ctx_presets, std::string bin_path);
};

struct subprocess_s;

struct server_models {
private:
    struct instance_t {
        std::shared_ptr<subprocess_s> subproc; // shared between main thread and monitoring thread
        std::thread th;
        server_model_meta meta;
        FILE * stdin_file = nullptr;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    // for stopping models
    std::condition_variable cv_stop;
    std::set<std::string> stopping_models;

    common_preset_context ctx_preset;

    common_params base_params;
    std::string bin_path;
    std::vector<std::string> base_env;
    common_preset base_preset; // base preset from llama-server CLI args

    void update_meta(const std::string & name, const server_model_meta & meta);

    // unload least recently used models if the limit is reached
    void unload_lru();

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta && meta);

public:
    server_models(const common_params & params, int argc, char ** argv);

    void load_models();

    // check if a model instance exists (thread-safe)
    bool has_model(const std::string & name);

    // return a copy of model metadata (thread-safe)
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata (thread-safe)
    std::vector<server_model_meta> get_all_meta();

    // load and unload model instances
    // these functions are thread-safe
    void load(const std::string & name);
    void unload(const std::string & name);
    void unload_all();

    // update the status of a model instance (thread-safe)
    void update_status(const std::string & name, server_model_status status, int exit_code);

    // wait until the model instance is fully loaded (thread-safe)
    // return when the model no longer in "loading" state
    void wait_until_loading_finished(const std::string & name);

    // ensure the model is in ready state (thread-safe)
    // return false if model is ready
    // otherwise, load the model and blocking wait until it's ready, then return true (meta may need to be refreshed)
    bool ensure_model_ready(const std::string & name);

    // proxy an HTTP request to the model instance
    server_http_res_ptr proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used);

    // return true if the current process is a child server instance
    static bool is_child_server();

    // notify the router server that a model instance is ready
    // return the monitoring thread (to be joined by the caller)
    static std::thread setup_child_server(const std::function<void(int)> & shutdown_handler);

    // notify the router server that the sleeping state has changed
    static void notify_router_sleeping_state(bool sleeping);
};

struct server_models_routes {
    common_params params;
    json webui_settings = json::object();
    server_models models;
    server_models_routes(const common_params & params, int argc, char ** argv)
            : params(params), models(params, argc, argv) {
        if (!this->params.webui_config_json.empty()) {
            try {
                webui_settings = json::parse(this->params.webui_config_json);
            } catch (const std::exception & e) {
                LOG_ERR("%s: failed to parse webui config: %s\n", __func__, e.what());
                throw;
            }
        }
        init_routes();
    }

    void init_routes();
    // handlers using lambda function, so that they can capture `this` without `std::bind`
    server_http_context::handler_t get_router_props;
    server_http_context::handler_t proxy_get;
    server_http_context::handler_t proxy_post;
    server_http_context::handler_t get_router_models;
    server_http_context::handler_t post_router_models_load;
    server_http_context::handler_t post_router_models_unload;
};

/**
 * A simple HTTP proxy that forwards requests to another server
 * and relays the responses back.
 */
struct server_http_proxy : server_http_res {
    std::function<void()> cleanup = nullptr;
public:
    server_http_proxy(const std::string & method,
                      const std::string & scheme,
                      const std::string & host,
                      int port,
                      const std::string & path,
                      const std::map<std::string, std::string> & headers,
                      const std::string & body,
                      const std::map<std::string, uploaded_file> & files,
                      const std::function<bool()> should_stop,
                      int32_t timeout_read,
                      int32_t timeout_write
                      );
    ~server_http_proxy() {
        if (cleanup) {
            cleanup();
        }
    }
private:
    std::thread thread;
    struct msg_t {
        std::map<std::string, std::string> headers;
        int status = 0;
        std::string data;
        std::string content_type;
    };
};

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    // Fallback null to default value
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key);
        } catch (NLOHMANN_JSON_NAMESPACE::detail::type_error const & err) {
            LOG_WRN("Wrong type supplied for parameter '%s'. Expected '%s', using default value: %s\n", key.c_str(), json(default_value).type_name(), err.what());
            return default_value;
        }
    } else {
        return default_value;
    }
}

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
    ERROR_TYPE_INVALID_REQUEST,
    ERROR_TYPE_AUTHENTICATION,
    ERROR_TYPE_SERVER,
    ERROR_TYPE_NOT_FOUND,
    ERROR_TYPE_PERMISSION,
    ERROR_TYPE_UNAVAILABLE, // custom error
    ERROR_TYPE_NOT_SUPPORTED, // custom error
    ERROR_TYPE_EXCEED_CONTEXT_SIZE, // custom error
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
    common_grammar_trigger value;

    server_grammar_trigger() = default;
    server_grammar_trigger(const common_grammar_trigger & value) : value(value) {}
    server_grammar_trigger(const json & in) {
        value.type = (common_grammar_trigger_type) in.at("type").get<int>();
        value.value = in.at("value").get<std::string>();
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            value.token = (llama_token) in.at("token").get<int>();
        }
    }

    json to_json() const {
        json out {
            {"type", (int) value.type},
            {"value", value.value},
        };
        if (value.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
            out["token"] = (int) value.token;
        }
        return out;
    }
};

json format_error_response(const std::string & message, const enum error_type type);

//
// random string / id
//

std::string random_string();
std::string gen_chatcmplid();
std::string gen_tool_call_id();

// get a random marker; note: each time the server restarts, the marker will be different
const char * get_media_marker();

//
// lora utils
//

// check whether the given lora set has only aloras activated (empty => false)
bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras);

// if the two sets of loras are different, they require a cache clear unless the
// change is only from aloras to aloras.
bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next);

std::map<int, float> parse_lora_request(const json & data);

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2);

// get the ids of all enabled loras
std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras);

//
// server_tokens
//

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
    bool has_mtmd = false;

private: // disallow accessing these members directly, risking out-of-sync

    // map a **start** index in tokens to the image chunk
    // note: the order need to be in-sync with tokens
    std::map<size_t, mtmd::input_chunk_ptr> map_idx_to_media;

    // list of tokens
    //   if the token is LLAMA_TOKEN_NULL, it indicates that this position is occupied by media chunk
    //   otherwise, it is a normal text token
    // note: a non-text chunk can occupy multiple tokens (aka memory cells) in the token list
    // note(2): for M-RoPE, an image can occupy different number of pos; do not assume 1-to-1 mapping tokens <-> pos
    llama_tokens tokens;

    // for ex. with input of 5 text tokens and 2 images (each image occupies 3 tokens and 2 pos):
    //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1] [img1]
    // idx  0   1   2   3   4   5      6      7      8      9      10
    // pos  0   1   2   3   4   5      5      5      7      7      7
    // map_idx_to_media will contain: {5, img0}, {8, img1}

public:
    server_tokens() = default;
    ~server_tokens() = default;

    // Prevent copying
    // TODO: server_tokens should be copyable - remove this:
    server_tokens(const server_tokens&) = delete;
    server_tokens& operator=(const server_tokens&) = delete;

    // Allow moving (usually implicitly generated if members are movable)
    server_tokens(server_tokens&&) = default;
    server_tokens& operator=(server_tokens&&) = default;

    // Allow accessing elements using [] operator
    llama_token operator[](size_t index) { return tokens[index]; }
    const llama_token& operator[](size_t index) const { return tokens[index]; }

    server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd);
    server_tokens(const llama_tokens & tokens, bool has_mtmd);

    // for debugging
    std::string str() const;

    // the next position after n_tokens. if n_tokens < 0, return the next position after all tokens.
    llama_pos pos_next(int64_t n_tokens = -1) const;

    // number of tokens with position < max_pos
    size_t size_up_to_pos(llama_pos max_pos) const;

    const mtmd::input_chunk_ptr & find_chunk(size_t idx) const;

    void push_back(llama_token tok);

    // will create a copy of the chunk if it contains non-text data
    void push_back(const mtmd_input_chunk * chunk);

    // appends server tokens, updates the media map. copies media chunks.
    void push_back(server_tokens & tokens);

    // for compatibility with context shift and prompt truncation
    void insert(const llama_tokens & inp_tokens);

    // for compatibility with speculative decoding, ctx shift, slot save/load
    const llama_tokens & get_tokens() const;

    llama_tokens get_text_tokens() const;

    // for compatibility with speculative decoding
    void set_token(llama_pos pos, llama_token id);

    size_t size() const { return llama_server_tokens_len_rust(tokens.size()); }

    bool empty() const { return llama_server_tokens_is_empty_rust(tokens.size()); }

    void clear() {
        map_idx_to_media.clear();
        tokens.clear();
    }

    void keep_first(size_t n);

    std::string detokenize(const llama_context * ctx, bool special) const;

    size_t get_common_prefix(const server_tokens & b) const;

    // make sure all text tokens are within the vocab range
    bool validate(const struct llama_context * ctx) const;

    // encode and decode the image chunk
    int32_t process_chunk(
                llama_context * ctx,
                mtmd_context * mctx,
                size_t idx,
                llama_pos pos,
                int32_t seq_id,
                size_t & n_tokens_out) const;

    server_tokens clone() const;
};


//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data);

// is array having BOTH numbers & strings?
bool json_is_array_of_mixed_numbers_strings(const json & data);

// does array have any individual integers/tokens?
bool json_is_array_and_contains_numbers(const json & data);

// get value by path(key1 / key2)
json json_get_nested_values(const std::vector<std::string> & paths, const json & js);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special);

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t validate_utf8(const std::string& text);

// process mtmd prompt, return the server_tokens containing both text tokens and media chunks
server_tokens process_mtmd_prompt(mtmd_context * mctx, std::string prompt, std::vector<raw_buffer> files);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> tokenize_input_prompts(
                                        const llama_vocab * vocab,
                                        mtmd_context * mctx,
                                        const json & json_prompt,
                                        bool add_special,
                                        bool parse_special);

//
// OAI utils
//

// global server parameters for chat formatting / parsing
struct server_chat_params {
    bool use_jinja;
    bool prefill_assistant;
    common_reasoning_format reasoning_format;
    std::map<std::string, std::string> chat_template_kwargs; // mapping key --> json value
    common_chat_templates_ptr tmpls;
    bool allow_image;
    bool allow_audio;
    bool enable_thinking = true;
    int  reasoning_budget = -1;
    std::string reasoning_budget_message;
    std::string media_path;
    bool force_pure_content = false;
};

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body);

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files);

    // TODO: keep with the task helpers in server-common.cpp.inc
json format_embeddings_response_oaicompat(
    const json & request,
    const std::string & model_name,
    const json & embeddings,
    bool use_base64 = false);

    // TODO: keep with the task helpers in server-common.cpp.inc
json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n);

//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx);

std::string safe_json_to_str(const json & data);

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens);
std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens);

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token);

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data);

std::string format_oai_resp_sse(const json & data);

// format Anthropic-style SSE with event types
std::string format_anthropic_sse(const json & data);

bool is_valid_utf8(const std::string & str);

//
// formatting output responses
    // TODO: keep these with the task helpers in server-common.cpp.inc
//

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt);

// format rerank task: [BOS]query[EOS][SEP]doc[EOS].
server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc);

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <list>
#include <map>
#include <mutex>
#include <vector>


enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = true;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    std::string generated_text; // append new chunks of generated text here
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(const common_chat_parser_params & chat_parser_params)
        : chat_parser_params(chat_parser_params)
        , oai_resp_id("resp_" + random_string())
        , oai_resp_reasoning_id("rs_" + random_string())
        , oai_resp_message_id("msg_" + random_string()) {}

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return llama_server_tokens_len_i32_rust(tokens.size());
    }

    bool need_embd() const {
        return llama_server_task_need_embd_rust(type);
    }

    bool need_logits() const {
        return llama_server_task_need_logits_rust(type);
    }

    bool need_sampling() const {
        return llama_server_task_need_sampling_rust(type);
    }

    static task_params params_from_json_cmpl(
        const llama_vocab * vocab,
        const common_params & params_base,
        const int n_ctx_slot,
        const std::vector<llama_logit_bias> & logit_bias_eog,
        const json & data);

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::vector<int32_t> raw_ids;
        raw_ids.reserve(tasks.size());
        for (const auto & task : tasks) {
            raw_ids.push_back(task.id);
            for (const auto & child : task.child_tasks) {
                raw_ids.push_back(child.id);
            }
        }
        std::vector<int32_t> unique_ids(raw_ids.size());
        const size_t n_unique = llama_server_unique_task_ids_rust(
                raw_ids.data(), raw_ids.size(),
                unique_ids.data(), unique_ids.size());
        return std::unordered_set<int>(unique_ids.begin(), unique_ids.begin() + n_unique);
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    task_result_state create_state() const {
        return task_result_state(params.chat_parser_params);
    }

    bool is_parent() const {
        return llama_server_task_is_parent_rust(child_tasks.size());
    }

    bool is_child() const {
        return llama_server_task_is_child_rust(id_parent);
    }
};

struct result_timings {
    int32_t cache_n = -1;

    int32_t prompt_n = -1;
    double prompt_ms = 0.0;
    double prompt_per_token_ms = 0.0;
    double prompt_per_second = 0.0;

    int32_t predicted_n = -1;
    double predicted_ms = 0.0;
    double predicted_per_token_ms = 0.0;
    double predicted_per_second = 0.0;

    // Optional speculative metrics - only included when > 0
    int32_t draft_n = 0;
    int32_t draft_n_accepted = 0;

    json to_json() const;
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    std::string content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    result_timings timings;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    completion_token_output prob_output;
    result_timings timings;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

struct server_task_result_metrics : server_task_result {
    int n_idle_slots;
    int n_processing_slots;
    int n_tasks_deferred;
    int64_t t_start;

    // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt_checkpoint {
    llama_pos pos_min;
    llama_pos pos_max;

    int64_t n_tokens;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }

    bool empty() const {
        return data.empty();
    }

    void clear() {
        pos_min = 0;
        pos_max = 0;
        n_tokens = 0;
        data.clear();
    }
};

struct server_prompt {
    server_tokens tokens;

    std::vector<uint8_t> data;

    std::list<server_prompt_checkpoint> checkpoints;

    size_t size() const {
        std::vector<size_t> checkpoint_sizes;
        checkpoint_sizes.reserve(checkpoints.size());
        for (const auto & checkpoint : checkpoints) {
            checkpoint_sizes.push_back(checkpoint.size());
        }
        return llama_server_prompt_size_rust(data.size(), checkpoint_sizes.data(), checkpoint_sizes.size());
    }

    int n_tokens() const {
        return llama_server_tokens_len_i32_rust(tokens.size());
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            data,
            checkpoints
        };
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt * alloc(const server_prompt & prompt, size_t state_size);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx, int32_t id_slot);

    void update();
};

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue {
private:
    int id = 0;
    bool running  = false;
    bool sleeping = false;
    bool req_stop_sleeping = false;
    int64_t time_last_task = 0;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task &&)> callback_new_task;
    std::function<void(void)>           callback_update_slots;
    std::function<void(bool)>           callback_sleeping_state;

public:
    // Add a new task to the end of the queue
    int post(server_task && task, bool front = false);

    // multi-task version of post()
    int post(std::vector<server_task> && tasks, bool front = false);

    // Add a new task, but defer until one slot is available
    void defer(server_task && task);

    // Get the next id for creating a new task
    int get_new_id();

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
    void pop_deferred_task(int id_slot);

    // if sleeping, request exiting sleep state and wait until it is done
    // returns immediately if not sleeping
    void wait_until_no_sleep();

    bool is_sleeping() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return sleeping;
    }

    // end the start_loop routine
    void terminate();

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     *
     * Sleeping procedure (disabled if idle_sleep_ms < 0):
     * - If there is no task after idle_sleep_ms, enter sleeping state
     * - Call callback_sleeping_state(true)
     * - Wait until req_stop_sleeping is set to true
     * - Call callback_sleeping_state(false)
     * - Exit sleeping state
     */
    void start_loop(int64_t idle_sleep_ms = -1);

    // for metrics
    size_t queue_tasks_deferred_size() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return queue_tasks_deferred.size();
    }

    //
    // Functions below are not thread-safe, must only be used before start_loop() is called
    //

    // Register function to process a new task
    void on_new_task(std::function<void(server_task &&)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Register callback for sleeping state change; multiple callbacks are allowed
    // note: when entering sleeping state, the callback is called AFTER sleeping is set to true
    //       when leaving sleeping state, the callback is called BEFORE sleeping is set to false
    void on_sleeping_state(std::function<void(bool)> callback) {
        if (callback_sleeping_state) {
            auto prev_callback = std::move(callback_sleeping_state);
            callback_sleeping_state = [prev_callback, callback](bool sleeping) {
                prev_callback(sleeping);
                callback(sleeping);
            };
        } else {
            callback_sleeping_state = std::move(callback);
        }
    }

private:
    void cleanup_pending_task(int id_target);
};

// struct for managing server responses
// in most cases, use server_response_reader to retrieve results
struct server_response {
private:
    bool running = true;

    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::vector<server_task_result_ptr> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

public:
    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task);

    void add_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    // remove multiple tasks from waiting list
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks);

    // same as recv(), but have timeout in seconds
    // if timeout is reached, nullptr is returned
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);

    // single-task version of recv()
    server_task_result_ptr recv(int id_task);

    // Send a new result to a waiting id_task
    void send(server_task_result_ptr && result);

    // terminate the waiting loop
    void terminate();
};

// utility class to make working with server_queue and server_response easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_queue & queue_tasks;
    server_response & queue_results;
    size_t received_count = 0;
    bool cancelled = false;
    int polling_interval_seconds;

    // tracking generation state and partial tool calls
    // only used by streaming completions
    std::vector<task_result_state> states;

    // should_stop function will be called each polling_interval_seconds
    server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds)
        : queue_tasks(queue_tasks), queue_results(queue_results), polling_interval_seconds(polling_interval_seconds) {}
    ~server_response_reader() {
        stop();
    }

    int get_new_id() {
        return queue_tasks.get_new_id();
    }

    // if front = true, the task will be posted to the front of the queue (high priority)
    void post_task(server_task && task, bool front = false);
    void post_tasks(std::vector<server_task> && tasks, bool front = false);
    bool has_next() const;

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop);

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };
    // aggregate multiple results
    batch_response wait_for_all(const std::function<bool()> & should_stop);

    void stop();
};

struct server_context_impl; // private implementation

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    json json_webui_settings;
    int slot_n_ctx;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
};

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // register a callback to be called when sleeping state changes
    // must be set before load_model() is called
    void on_sleeping_changed(std::function<void(bool)> callback);
};

// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;
private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    const server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);
};


#include <cstdint>
#include <random>
#include <sstream>
#include <fstream>

extern "C" {
uint8_t * llama_server_base64_encode(const uint8_t * data, size_t len, size_t * out_len);
uint8_t * llama_server_base64_decode(const uint8_t * data, size_t len, size_t * out_len);
void      llama_server_base64_free(uint8_t * data, size_t len);
uint64_t  llama_server_fnv_hash_rust(const uint8_t * data, size_t len);
int       llama_server_normalize_anthropic_billing_header_rust(uint8_t * data, size_t len);
float     llama_server_probability_logarithm_rust(float value);
size_t    llama_server_validate_utf8_prefix_len_rust(const uint8_t * data, size_t len);
bool      llama_server_is_valid_utf8_rust(const uint8_t * data, size_t len);
const char * llama_server_stop_type_to_string_rust(int stop_type);
size_t    llama_server_copy_string_bytes_rust(const uint8_t * data, size_t len, uint8_t * out, size_t out_len);
}

json format_error_response(const std::string & message, const enum error_type type) {
    std::string type_str;
    int code = 500;
    switch (type) {
        case ERROR_TYPE_INVALID_REQUEST:
            type_str = "invalid_request_error";
            code = 400;
            break;
        case ERROR_TYPE_AUTHENTICATION:
            type_str = "authentication_error";
            code = 401;
            break;
        case ERROR_TYPE_NOT_FOUND:
            type_str = "not_found_error";
            code = 404;
            break;
        case ERROR_TYPE_SERVER:
            type_str = "server_error";
            code = 500;
            break;
        case ERROR_TYPE_PERMISSION:
            type_str = "permission_error";
            code = 403;
            break;
        case ERROR_TYPE_NOT_SUPPORTED:
            type_str = "not_supported_error";
            code = 501;
            break;
        case ERROR_TYPE_UNAVAILABLE:
            type_str = "unavailable_error";
            code = 503;
            break;
        case ERROR_TYPE_EXCEED_CONTEXT_SIZE:
            type_str = "exceed_context_size_error";
            code = 400;
            break;
    }
    return json {
        {"code", code},
        {"message", message},
        {"type", type_str},
    };
}

//
// random string / id
//

std::string random_string() {
    static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

    std::random_device rd;
    std::mt19937 generator(rd());

    std::string result(32, ' ');

    for (int i = 0; i < 32; ++i) {
        result[i] = str[generator() % str.size()];
    }

    return result;
}

std::string gen_chatcmplid() {
    return "chatcmpl-" + random_string();
}

std::string gen_tool_call_id() {
    return random_string();
}

const char * get_media_marker() {
    static const std::string marker = []() {
        // allow user to pin a reproducible marker via env var
        const char * env = getenv("LLAMA_MEDIA_MARKER");
        if (env && env[0] != '\0') {
            return std::string(env);
        }
        return std::string("<__media_") + random_string() + "__>";
    }();
    return marker.c_str();
}

//
// lora utils
//

bool lora_all_alora(const std::vector<common_adapter_lora_info> & loras) {
    std::vector<float> scales;
    std::vector<uint8_t> is_alora;
    scales.reserve(loras.size());
    is_alora.reserve(loras.size());
    for (const auto & lora : loras) {
        scales.push_back(lora.scale);
        is_alora.push_back(llama_adapter_get_alora_n_invocation_tokens(lora.ptr) != 0 ? 1 : 0);
    }
    return llama_server_lora_all_alora_rust(scales.data(), is_alora.data(), loras.size());
}

bool lora_should_clear_cache(
        const std::vector<common_adapter_lora_info> & current,
        const std::vector<common_adapter_lora_info> & next) {

    // This should always be called after determining that the two sets are
    // _not_ equal. This assert is therefore some slightly wasted work and
    // should be safe to remove as long as this method is called correctly.
    GGML_ASSERT(!are_lora_equal(current, next));

    return llama_server_lora_should_clear_cache_rust(
            !lora_get_enabled_ids(current).empty(),
            lora_all_alora(current),
            lora_all_alora(next));
}

std::map<int, float> parse_lora_request(const json & data) {
    std::map<int, float> lora;

    // set value
    for (const auto & entry : data) {
        int id      = json_value(entry, "id", -1);
        float scale = json_value(entry, "scale", 0.0f);
        lora[id] = scale;
    }

    return lora;
}

bool are_lora_equal(
        const std::vector<common_adapter_lora_info> & l1,
        const std::vector<common_adapter_lora_info> & l2) {
    std::vector<float> left_scales;
    std::vector<float> right_scales;
    std::vector<uintptr_t> left_ptrs;
    std::vector<uintptr_t> right_ptrs;
    left_scales.reserve(l1.size());
    right_scales.reserve(l2.size());
    left_ptrs.reserve(l1.size());
    right_ptrs.reserve(l2.size());
    for (const auto & lora : l1) {
        left_scales.push_back(lora.scale);
        left_ptrs.push_back((uintptr_t) lora.ptr);
    }
    for (const auto & lora : l2) {
        right_scales.push_back(lora.scale);
        right_ptrs.push_back((uintptr_t) lora.ptr);
    }
    // we don't check lora.path to reduce the time complexity
    return llama_server_lora_are_equal_rust(
            left_scales.data(), left_ptrs.data(), left_scales.size(),
            right_scales.data(), right_ptrs.data(), right_scales.size());
}

std::vector<size_t> lora_get_enabled_ids(const std::vector<common_adapter_lora_info> & loras) {
    std::vector<float> scales;
    std::vector<size_t> enabled_ids;
    scales.reserve(loras.size());
    enabled_ids.resize(loras.size());
    for (const auto & lora : loras) {
        scales.push_back(lora.scale);
    }
    const size_t n_enabled = llama_server_lora_enabled_ids_rust(
            scales.data(), scales.size(),
            enabled_ids.data(), enabled_ids.size());
    enabled_ids.resize(n_enabled);
    return enabled_ids;
}

static inline raw_buffer base64_decode(const std::string & encoded_string) {
    size_t out_len = 0;
    uint8_t * data = llama_server_base64_decode(
            reinterpret_cast<const uint8_t *>(encoded_string.data()), encoded_string.size(), &out_len);
    if (data == nullptr) {
        return {};
    }
    raw_buffer ret(data, data + out_len);
    llama_server_base64_free(data, out_len);
    return ret;
}

//
// server_tokens implementation
//

server_tokens::server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd) : has_mtmd(has_mtmd) {
    for (size_t i = 0; i < mtmd_chunks.size(); ++i) {
        push_back(mtmd_chunks[i]);
    }
}

server_tokens::server_tokens(const llama_tokens & tokens, bool has_mtmd) : has_mtmd(has_mtmd), tokens(tokens) {
}

llama_pos server_tokens::pos_next(int64_t n_tokens) const {
    if (!has_mtmd) {
        return llama_server_tokens_pos_next_rust(tokens.size(), n_tokens, nullptr, nullptr, nullptr, 0);
    }

    GGML_ASSERT(n_tokens <= (int64_t)tokens.size());

    std::vector<size_t> media_indices;
    std::vector<int32_t> media_n_pos;
    std::vector<size_t> media_n_tokens;
    media_indices.reserve(map_idx_to_media.size());
    media_n_pos.reserve(map_idx_to_media.size());
    media_n_tokens.reserve(map_idx_to_media.size());
    for (const auto & it : map_idx_to_media) {
        const auto & chunk = it.second;
        media_indices.push_back(it.first);
        media_n_pos.push_back(mtmd_input_chunk_get_n_pos(chunk.get()));
        media_n_tokens.push_back(mtmd_input_chunk_get_n_tokens(chunk.get()));
    }

    return llama_server_tokens_pos_next_rust(
            tokens.size(),
            n_tokens,
            media_indices.data(),
            media_n_pos.data(),
            media_n_tokens.data(),
            media_indices.size());
}

size_t server_tokens::size_up_to_pos(llama_pos max_pos) const {
    if (!has_mtmd) {
        return llama_server_tokens_size_up_to_pos_rust(tokens.size(), max_pos, nullptr, nullptr, nullptr, 0);
    }

    std::vector<size_t> media_indices;
    std::vector<int32_t> media_n_pos;
    std::vector<size_t> media_n_tokens;
    media_indices.reserve(map_idx_to_media.size());
    media_n_pos.reserve(map_idx_to_media.size());
    media_n_tokens.reserve(map_idx_to_media.size());
    for (const auto & it : map_idx_to_media) {
        const auto & chunk = it.second;
        media_indices.push_back(it.first);
        media_n_pos.push_back(mtmd_input_chunk_get_n_pos(chunk.get()));
        media_n_tokens.push_back(mtmd_input_chunk_get_n_tokens(chunk.get()));
    }

    return llama_server_tokens_size_up_to_pos_rust(
            tokens.size(),
            max_pos,
            media_indices.data(),
            media_n_pos.data(),
            media_n_tokens.data(),
            media_indices.size());
}

std::string server_tokens::str() const {
    std::vector<size_t> media_indices;
    media_indices.reserve(map_idx_to_media.size());
    for (const auto & it : map_idx_to_media) {
        media_indices.push_back(it.first);
    }
    size_t out_len = 0;
    uint8_t * out = llama_server_tokens_debug_string_rust(
            tokens.data(), tokens.size(),
            media_indices.data(), media_indices.size(),
            &out_len);
    if (out == nullptr) {
        return "";
    }
    std::string result((const char *) out, out_len);
    llama_server_base64_free(out, out_len);
    return result;
}

const mtmd::input_chunk_ptr & server_tokens::find_chunk(size_t idx) const {
    auto it = map_idx_to_media.find(idx);
    if (it != map_idx_to_media.end()) {
        return it->second;
    }
    throw std::runtime_error("Chunk not found");
}

void server_tokens::push_back(llama_token tok) {
    if (tok == LLAMA_TOKEN_NULL) {
        throw std::runtime_error("Invalid token");
    }
    tokens.emplace_back(tok);
}

void server_tokens::push_back(const mtmd_input_chunk * chunk) {
    auto type = mtmd_input_chunk_get_type(chunk);
    if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        GGML_ASSERT(has_mtmd);
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        size_t start_idx = tokens.size();
        for (size_t i = 0; i < n_tokens; ++i) {
            tokens.emplace_back(LLAMA_TOKEN_NULL);
        }
        mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
        map_idx_to_media[start_idx] = std::move(new_chunk);
    } else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n_tokens;
        const auto * text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
        for (size_t i = 0; i < n_tokens; ++i) {
            push_back(text_tokens[i]);
        }
    } else {
        GGML_ABORT("Invalid chunk type");
    }
}

void server_tokens::push_back(server_tokens & tokens) {
    size_t start_idx = size();
    for (size_t i = 0; i < tokens.size(); i++) {
        push_back(tokens[i]);
    }
    if (tokens.has_mtmd) {
        // Assert if we are copying MTMD chunks to a server_tokens that does not have mtmd.
        // We could also just check, but this will prevent silently dropping MTMD data.
        GGML_ASSERT(has_mtmd);
        for (auto it = tokens.map_idx_to_media.begin(); it != tokens.map_idx_to_media.end(); ) {
            auto * chunk = tokens.map_idx_to_media[it->first].get();
            mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
            map_idx_to_media[start_idx + it->first] = std::move(new_chunk);
        }
    }
}

void server_tokens::insert(const llama_tokens & inp_tokens) {
    tokens.insert(tokens.end(), inp_tokens.begin(), inp_tokens.end());
}

const llama_tokens & server_tokens::get_tokens() const {
    GGML_ASSERT(!has_mtmd);
    return tokens;
}

llama_tokens server_tokens::get_text_tokens() const {
    llama_tokens res(tokens.size());
    const size_t n_text = llama_server_filter_text_tokens_rust(
            tokens.data(), tokens.size(),
            res.data(), res.size());
    res.resize(n_text);
    return res;
}

void server_tokens::set_token(llama_pos pos, llama_token id) {
    GGML_ASSERT(!has_mtmd); // only allow this if mtmd is disabled
    tokens[pos] = id;
}

void server_tokens::keep_first(size_t n) {
    GGML_ASSERT(n <= tokens.size());
    if (has_mtmd) {
        if (n == tokens.size()) {
            return; // nothing to do
        }
        // we throw an error if we try to remove a token in the middle of an image
        // for ex. with input of 5 text tokens and 2 images:
        //    [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
        // n  1   2   3   4   5   6      7      8      9      10
        // allowed to resize      ^                    ^
        // disallowed to resize          ^      ^             ^
        if (n > 0) {
            // make sure we never remove tokens in the middle of an image
            // note that the case where we keep a full image at the end is allowed:
            //   tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] != LLAMA_TOKEN_NULL
            if (tokens[n - 1] == LLAMA_TOKEN_NULL && tokens[n] == LLAMA_TOKEN_NULL) {
                find_chunk(n - 1); // will throw an error if the token is not begin-of-chunk
            }
        }
        // remove all image chunks that are not used anymore
        for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ) {
            size_t idx = it->first;
            if (idx >= n) {
                it = map_idx_to_media.erase(it);
            } else {
                ++it;
            }
        }
    }
    tokens.resize(n);
}

std::string server_tokens::detokenize(const llama_context * ctx, bool special) const {
    llama_tokens text_tokens = get_text_tokens();
    return common_detokenize(ctx, text_tokens, special);
}

size_t server_tokens::get_common_prefix(const server_tokens & b) const {
    const size_t max_idx = std::min(tokens.size(), b.tokens.size());

    if (!has_mtmd) {
        return llama_server_token_common_prefix_rust(
                tokens.data(), tokens.size(),
                b.tokens.data(), b.tokens.size());
    }

    for (size_t i = 0; i < max_idx; ++i) {
        const llama_token ai =   tokens[i];
        const llama_token bi = b.tokens[i];

        if (ai == LLAMA_TOKEN_NULL && bi == LLAMA_TOKEN_NULL) {
            const auto & a_chunk =   find_chunk(i);
            const auto & b_chunk = b.find_chunk(i);

            GGML_ASSERT(a_chunk && b_chunk);

            const std::string id_ai = mtmd_input_chunk_get_id(a_chunk.get());
            const std::string id_bi = mtmd_input_chunk_get_id(b_chunk.get());

            const size_t n_tok_a = mtmd_input_chunk_get_n_tokens(a_chunk.get());
            const size_t n_tok_b = mtmd_input_chunk_get_n_tokens(b_chunk.get());

            if (id_ai == id_bi && n_tok_a == n_tok_b) {
                GGML_ASSERT(n_tok_a > 0 && "Invalid media chunk"); // should never happen
                i += n_tok_a - 1; // will be +1 by the for loop
                continue;
            }

            return i;
        }

        if (ai == bi) {
            continue;
        }

        return i;
    }

    return max_idx; // all tokens are equal
}

bool server_tokens::validate(const struct llama_context * ctx) const {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto & t = tokens[i];
        if (t == LLAMA_TOKEN_NULL) {
            try {
                const auto & chunk = find_chunk(i);
                size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk.get());
                i += n_tokens - 1; // will be +1 by the for loop
            } catch (const std::exception & e) {
                return false;
            }
        }
    }
    return llama_server_tokens_valid_range_rust(tokens.data(), tokens.size(), n_vocab);
}

int32_t server_tokens::process_chunk(
            llama_context * ctx,
            mtmd_context * mctx,
            size_t idx,
            llama_pos pos,
            int32_t seq_id,
            size_t & n_tokens_out) const {
    const auto & chunk = find_chunk(idx);
    const char * name = mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_IMAGE
                        ? "image" : "audio";
    SRV_INF("processing %s...\n", name);
    int32_t n_batch = llama_n_batch(ctx);
    int64_t t0 = ggml_time_ms();
    llama_pos new_n_past; // unused for now
    int32_t result = mtmd_helper_eval_chunk_single(mctx, ctx,
        chunk.get(),
        pos,
        seq_id,
        n_batch,
        true, // logits last
        &new_n_past);
    SRV_INF("%s processed in %" PRId64 " ms\n", name, ggml_time_ms() - t0);
    if (result != 0) {
        LOG_ERR("mtmd_helper_eval failed with status %d", result);
        n_tokens_out = 0;
        return result;
    }
    n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
    return 0;
}

server_tokens server_tokens::clone() const {
    server_tokens res;
    res.has_mtmd = has_mtmd;
    res.tokens   = tokens;
    for (auto it = map_idx_to_media.begin(); it != map_idx_to_media.end(); ++it) {
        size_t idx = it->first;
        const mtmd::input_chunk_ptr & chunk = it->second;
        res.map_idx_to_media[idx] = mtmd::input_chunk_ptr(mtmd_input_chunk_copy(chunk.get()));
    }
    return res;
}

//
// tokenizer and input processing utils
//

bool json_is_array_of_numbers(const json & data) {
    if (!data.is_array()) {
        return false;
    }
    const auto flags = json_array_type_flags(data);
    return llama_server_json_flags_all_numbers_rust(flags.data(), flags.size());
}

bool json_is_array_of_mixed_numbers_strings(const json & data) {
    if (!data.is_array()) {
        return false;
    }
    const auto flags = json_array_type_flags(data);
    return llama_server_json_flags_mixed_numbers_strings_rust(flags.data(), flags.size());
}

bool json_is_array_and_contains_numbers(const json & data) {
    if (!data.is_array()) {
        return false;
    }
    const auto flags = json_array_type_flags(data);
    return llama_server_json_flags_contains_number_rust(flags.data(), flags.size());
}

json json_get_nested_values(const std::vector<std::string> & paths, const json & js) {
    json result = json::object();

    for (const std::string & path : paths) {
        json current = js;
        const auto keys = string_split<std::string>(path, /*separator*/ '/');
        bool valid_path = true;
        for (const std::string & k : keys) {
            if (valid_path && current.is_object() && current.contains(k)) {
                current = current[k];
            } else {
                valid_path = false;
            }
        }
        if (valid_path) {
            result[path] = current;
        }
    }
    return result;
}

llama_tokens tokenize_mixed(const llama_vocab * vocab, const json & json_prompt, bool add_special, bool parse_special) {
    // If `add_bos` is true, we only add BOS, when json_prompt is a string,
    // or the first element of the json_prompt array is a string.
    llama_tokens prompt_tokens;

    if (json_prompt.is_array()) {
        bool first = true;
        for (const auto & p : json_prompt) {
            if (p.is_string()) {
                auto s = p.template get<std::string>();

                llama_tokens p;
                if (first) {
                    p = common_tokenize(vocab, s, add_special, parse_special);
                    first = false;
                } else {
                    p = common_tokenize(vocab, s, false, parse_special);
                }

                prompt_tokens.insert(prompt_tokens.end(), p.begin(), p.end());
            } else {
                if (first) {
                    first = false;
                }

                prompt_tokens.push_back(p.template get<llama_token>());
            }
        }
    } else {
        auto s = json_prompt.template get<std::string>();
        prompt_tokens = common_tokenize(vocab, s, add_special, parse_special);
    }

    return prompt_tokens;
}

size_t validate_utf8(const std::string& text) {
    return llama_server_validate_utf8_prefix_len_rust((const uint8_t *) text.data(), text.size());
}

server_tokens process_mtmd_prompt(mtmd_context * mctx, std::string prompt, std::vector<raw_buffer> files) {
    mtmd::bitmaps bitmaps;
    for (auto & file : files) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(mctx, file.data(), file.size()));
        if (!bmp.ptr) {
            throw std::runtime_error("Failed to load image or audio file");
        }
        // calculate bitmap hash (for KV caching)
        std::string hash = std::to_string(llama_server_fnv_hash_rust(bmp.data(), bmp.n_bytes()));
        bmp.set_id(hash.c_str());
        bitmaps.entries.push_back(std::move(bmp));
    }
    // process prompt
    std::vector<server_tokens> inputs;
    // multimodal
    mtmd_input_text inp_txt = {
        prompt.c_str(),
        /* add_special */   true,
        /* parse_special */ true,
    };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    int32_t tokenized = mtmd_tokenize(mctx,
                                      chunks.ptr.get(),
                                      &inp_txt,
                                      bitmaps_c_ptr.data(),
                                      bitmaps_c_ptr.size());
    if (tokenized != 0) {
        throw std::runtime_error("Failed to tokenize prompt");
    }
    auto result = server_tokens(chunks, true);
    return result;
}

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 */
static server_tokens tokenize_input_subprompt(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special) {
    constexpr char JSON_STRING_PROMPT_KEY[] = "prompt_string";
    constexpr char JSON_MTMD_DATA_KEY[] = "multimodal_data";
    const bool has_mtmd = mctx != nullptr;
    if (json_prompt.is_string() || json_is_array_of_mixed_numbers_strings(json_prompt)) {
        // string or mixed
        llama_tokens tmp = tokenize_mixed(vocab, json_prompt, add_special, parse_special);
        return server_tokens(tmp, false);
    } else if (json_is_array_of_numbers(json_prompt)) {
        // array of tokens
        llama_tokens tmp = json_prompt.get<llama_tokens>();
        return server_tokens(tmp, false);
    } else if (json_prompt.contains(JSON_STRING_PROMPT_KEY)) {
        // JSON object with prompt key.
        if (json_prompt.contains(JSON_MTMD_DATA_KEY)) {
            if (!has_mtmd)
                throw std::runtime_error("Multimodal data provided, but model does not support multimodal requests.");

            // JSON object with prompt and multimodal key.
            std::vector<raw_buffer> files;
            for (const auto & entry : json_prompt.at(JSON_MTMD_DATA_KEY)) {
                files.push_back(base64_decode(entry));
            }
            return process_mtmd_prompt(mctx, json_prompt.at(JSON_STRING_PROMPT_KEY), files);
        } else {
            // Not multimodal, but contains a subobject.
            llama_tokens tmp = tokenize_mixed(vocab, json_prompt.at(JSON_STRING_PROMPT_KEY), add_special, parse_special);
            return server_tokens(tmp, false);
        }
   } else {
       throw std::runtime_error("\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of mixed strings & tokens.");
   }
}

std::vector<server_tokens> tokenize_input_prompts(const llama_vocab * vocab, mtmd_context * mctx, const json & json_prompt, bool add_special, bool parse_special) {
    std::vector<server_tokens> result;
    if (json_prompt.is_array() && !json_is_array_and_contains_numbers(json_prompt)) {
        result.reserve(json_prompt.size());
        for (const auto & p : json_prompt) {
            result.push_back(tokenize_input_subprompt(vocab, mctx, p,add_special, parse_special));
        }
    } else {
        result.push_back(tokenize_input_subprompt(vocab, mctx, json_prompt, add_special, parse_special));
    }
    if (result.empty()) {
        throw std::runtime_error("\"prompt\" must not be empty");
    }
    return result;
}

//
// OAI utils
//

// used by /completions endpoint
json oaicompat_completion_params_parse(const json & body) {
    json llama_params;

    if (!body.contains("prompt")) {
        throw std::runtime_error("\"prompt\" is required");
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    // Handle "echo" field
    if (json_value(body, "echo", false)) {
        throw std::runtime_error("Only no echo is supported");
    }

    // Params supported by OAI but unsupported by llama.cpp
    static const std::vector<std::string> unsupported_params { "best_of", "suffix" };
    for (const auto & param : unsupported_params) {
        if (body.contains(param)) {
            throw std::runtime_error("Unsupported param: " + param);
        }
    }

    // Copy remaining properties to llama_params
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

// media_path always end with '/', see common argument parsing
static void handle_media(
        std::vector<raw_buffer> & out_files,
        json & media_obj,
        const std::string & media_path) {
    std::string url = json_value(media_obj, "url", std::string());
    if (string_starts_with(url, "http")) {
        // download remote image
        // TODO @ngxson : maybe make these params configurable
        common_remote_params params;
        params.max_size = 1024 * 1024 * 10; // 10MB
        params.timeout  = 10; // seconds
        SRV_INF("downloading image from '%s'\n", url.c_str());
        auto res = common_remote_get_content(url, params);
        if (200 <= res.first && res.first < 300) {
            SRV_INF("downloaded %zu bytes\n", res.second.size());
            raw_buffer data;
            data.insert(data.end(), res.second.begin(), res.second.end());
            out_files.push_back(data);
        } else {
            throw std::runtime_error("Failed to download image");
        }

    } else if (string_starts_with(url, "file://")) {
        if (media_path.empty()) {
            throw std::invalid_argument("file:// URLs are not allowed unless --media-path is specified");
        }
        // load local image file
        std::string file_path = url.substr(7); // remove "file://"
        raw_buffer data;
        if (!fs_validate_filename(file_path, true)) {
            throw std::invalid_argument("file path is not allowed: " + file_path);
        }
        SRV_INF("loading image from local file '%s'\n", (media_path + file_path).c_str());
        std::ifstream file(media_path + file_path, std::ios::binary);
        if (!file) {
            throw std::invalid_argument("file does not exist or cannot be opened: " + file_path);
        }
        data.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        out_files.push_back(data);

    } else {
        // try to decode base64 image
        std::vector<std::string> parts = string_split<std::string>(url, /*separator*/ ',');
        if (parts.size() != 2) {
            throw std::runtime_error("Invalid url value");
        } else if (!string_starts_with(parts[0], "data:image/")) {
            throw std::runtime_error("Invalid url format: " + parts[0]);
        } else if (!string_ends_with(parts[0], "base64")) {
            throw std::runtime_error("url must be base64 encoded");
        } else {
            auto base64_data = parts[1];
            auto decoded_data = base64_decode(base64_data);
            out_files.push_back(decoded_data);
        }
    }
}

// used by /chat/completions endpoint
json oaicompat_chat_params_parse(
    json & body, /* openai api json semantics */
    const server_chat_params & opt,
    std::vector<raw_buffer> & out_files)
{
    json llama_params;

    auto tools = json_value(body, "tools", json());
    auto has_tools = tools.is_array() && !tools.empty();
    auto stream = json_value(body, "stream", false);
    auto tool_choice = json_value(body, "tool_choice", std::string("auto"));

    if (!opt.use_jinja) {
        if (has_tools) {
            throw std::runtime_error("tools param requires --jinja flag");
        }
        if (tool_choice != "auto") {
            throw std::runtime_error("tool_choice param requires --jinja flag");
        }
    }

    // Handle "stop" field
    if (body.contains("stop") && body.at("stop").is_string()) {
        llama_params["stop"] = json::array({body.at("stop").get<std::string>()});
    } else {
        llama_params["stop"] = json_value(body, "stop", json::array());
    }

    auto json_schema = json_value(body, "json_schema", json());
    auto grammar = json_value(body, "grammar", std::string());
    if (!json_schema.is_null() && !grammar.empty()) {
        throw std::runtime_error("Cannot use both json_schema and grammar");
    }

    // Handle "response_format" field
    if (body.contains("response_format")) {
        json response_format      = json_value(body, "response_format", json::object());
        std::string response_type = json_value(response_format, "type", std::string());
        if (response_type == "json_object") {
            if (response_format.contains("schema") || json_schema.empty()) {
                json_schema = json_value(response_format, "schema", json::object());
            }
        } else if (response_type == "json_schema") {
            auto schema_wrapper = json_value(response_format, "json_schema", json::object());
            json_schema = json_value(schema_wrapper, "schema", json::object());
        } else if (!response_type.empty() && response_type != "text") {
            throw std::invalid_argument("response_format type must be one of \"text\" or \"json_object\", but got: " + response_type);
        }
    }

    // get input files
    if (!body.contains("messages")) {
        throw std::invalid_argument("'messages' is required");
    }
    json & messages = body.at("messages");
    if (!messages.is_array()) {
        throw std::invalid_argument("Expected 'messages' to be an array");
    }
    for (auto & msg : messages) {
        std::string role = json_value(msg, "role", std::string());
        if (role != "assistant" && !msg.contains("content")) {
            throw std::invalid_argument("All non-assistant messages must contain 'content'");
        }
        if (role == "assistant") {
            if (!msg.contains("content") && !msg.contains("tool_calls")) {
                throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
            }
            if (!msg.contains("content")) {
                continue; // avoid errors with no content
            }
        }
        json & content = msg.at("content");
        if (content.is_string() || content.is_null()) {
            continue;
        }

        if (!content.is_array()) {
            throw std::invalid_argument("Expected 'content' to be a string or an array");
        }

        for (auto & p : content) {
            std::string type      = json_value(p, "type", std::string());
            if (type == "image_url") {
                if (!opt.allow_image) {
                    throw std::runtime_error("image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json image_url = json_value(p, "image_url", json::object());
                handle_media(out_files, image_url, opt.media_path);

                p["type"] = "media_marker";
                p["text"] = get_media_marker();
                p.erase("image_url");

            } else if (type == "input_audio") {
                if (!opt.allow_audio) {
                    throw std::runtime_error("audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
                }

                json input_audio   = json_value(p, "input_audio", json::object());
                std::string data   = json_value(input_audio, "data", std::string());
                std::string format = json_value(input_audio, "format", std::string());
                // while we also support flac, we don't allow it here so we matches the OAI spec
                if (format != "wav" && format != "mp3") {
                    throw std::invalid_argument("input_audio.format must be either 'wav' or 'mp3'");
                }
                auto decoded_data = base64_decode(data); // expected to be base64 encoded
                out_files.push_back(decoded_data);

                // TODO: add audio_url support by reusing handle_media()

                p["type"] = "media_marker";
                p["text"] = get_media_marker();
                p.erase("input_audio");

            } else if (type != "text") {
                throw std::invalid_argument("unsupported content[].type");
            }
        }
    }

    auto caps = common_chat_templates_get_caps(opt.tmpls.get());

    common_chat_templates_inputs inputs;
    inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
    inputs.tools                 = common_chat_tools_parse_oaicompat(tools);
    inputs.tool_choice           = common_chat_tool_choice_parse_oaicompat(tool_choice);
    inputs.json_schema           = json_schema.is_null() ? "" : json_schema.dump();
    inputs.grammar               = grammar;
    inputs.use_jinja             = opt.use_jinja;
    inputs.parallel_tool_calls   = json_value(body, "parallel_tool_calls", caps["supports_parallel_tool_calls"]);
    inputs.add_generation_prompt = json_value(body, "add_generation_prompt", true);
    inputs.reasoning_format      = opt.reasoning_format;
    if (body.contains("reasoning_format")) {
        inputs.reasoning_format = common_reasoning_format_from_name(body.at("reasoning_format").get<std::string>());
    }
    inputs.enable_thinking       = opt.enable_thinking;
    if (!inputs.tools.empty() && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
        if (body.contains("grammar")) {
            throw std::invalid_argument("Cannot use custom grammar constraints with tools.");
        }
        llama_params["parse_tool_calls"] = true;
    }

    // merge the template args provided from command line with the args provided in the user request
    auto chat_template_kwargs_object = json_value(body, "chat_template_kwargs", json::object());
    inputs.chat_template_kwargs = opt.chat_template_kwargs;
    for (const auto & item : chat_template_kwargs_object.items()) {
        inputs.chat_template_kwargs[item.key()] = item.value().dump();
    }

    // parse the "enable_thinking" kwarg to override the default value
    auto enable_thinking_kwarg = json_value(inputs.chat_template_kwargs, "enable_thinking", std::string(""));
    if (enable_thinking_kwarg == "true") {
        inputs.enable_thinking = true;
    } else if (enable_thinking_kwarg == "false") {
        inputs.enable_thinking = false;
    } else if (!enable_thinking_kwarg.empty() && enable_thinking_kwarg[0] == '"') {
        throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
    }

    // if the assistant message appears at the end of list, we do not add end-of-turn token
    // for ex. this can be useful to modify the reasoning process in reasoning models
    bool prefill_assistant_message = !inputs.messages.empty() && inputs.messages.back().role == "assistant" && opt.prefill_assistant;
    common_chat_msg last_message;
    if (prefill_assistant_message) {
        last_message = inputs.messages.back();
        inputs.messages.pop_back();

        /* sanity check, max one assistant message at the end of the list */
        if (!inputs.messages.empty() && inputs.messages.back().role == "assistant"){
            throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
        }

        /* TODO: test this properly */
        inputs.reasoning_format = COMMON_REASONING_FORMAT_NONE;

        if ( inputs.enable_thinking ) {
            throw std::invalid_argument("Assistant response prefill is incompatible with enable_thinking.");
        }

        inputs.add_generation_prompt = true;
    }
    inputs.force_pure_content = opt.force_pure_content;

    // Apply chat template to the list of messages
    auto chat_params = common_chat_templates_apply(opt.tmpls.get(), inputs);

    /* Append assistant prefilled message */
    if (prefill_assistant_message) {
        if (!last_message.content_parts.empty()) {
            for (auto & p : last_message.content_parts) {
                chat_params.prompt += p.text;
            }
        } else {
            chat_params.prompt += last_message.content;
        }
    }

    llama_params["chat_format"] = static_cast<int>(chat_params.format);
    llama_params["prompt"]      = chat_params.prompt;
    if (!chat_params.grammar.empty()) {
        llama_params["grammar"]      = chat_params.grammar;
        llama_params["grammar_type"] = std::string("tool_calls");
    }
    llama_params["grammar_lazy"] = chat_params.grammar_lazy;
    auto grammar_triggers        = json::array();
    for (const auto & trigger : chat_params.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }
    llama_params["grammar_triggers"]  = grammar_triggers;
    llama_params["preserved_tokens"]  = chat_params.preserved_tokens;
    llama_params["generation_prompt"] = chat_params.generation_prompt;
    for (const auto & stop : chat_params.additional_stops) {
        llama_params["stop"].push_back(stop);
    }
    if (!chat_params.parser.empty()) {
        llama_params["chat_parser"] = chat_params.parser;
    }

    // Reasoning budget: pass parameters through to sampling layer
    {
        int reasoning_budget = opt.reasoning_budget;
        if (reasoning_budget == -1 && body.contains("thinking_budget_tokens")) {
            reasoning_budget = json_value(body, "thinking_budget_tokens", -1);
        }

        if (!chat_params.thinking_end_tag.empty()) {
            llama_params["reasoning_budget_tokens"] = reasoning_budget;
            llama_params["reasoning_budget_start_tag"] = chat_params.thinking_start_tag;
            llama_params["reasoning_budget_end_tag"] = chat_params.thinking_end_tag;
            llama_params["reasoning_budget_message"] = opt.reasoning_budget_message;
        }
    }

    // Handle "logprobs" field
    // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
    if (json_value(body, "logprobs", false)) {
        if (has_tools && stream) {
            throw std::invalid_argument("logprobs is not supported with tools + stream");
        }
        llama_params["n_probs"] = json_value(body, "top_logprobs", 20);
    } else if (body.contains("top_logprobs") && !body.at("top_logprobs").is_null()) {
        throw std::invalid_argument("top_logprobs requires logprobs to be set to true");
    }

    // Copy remaining properties to llama_params
    // This allows user to use llama.cpp-specific params like "mirostat", ... via OAI endpoint.
    // See "launch_slot_with_task()" for a complete list of params supported by llama.cpp
    for (const auto & item : body.items()) {
        // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
        if (!llama_params.contains(item.key()) || item.key() == "n_predict") {
            llama_params[item.key()] = item.value();
        }
    }

    return llama_params;
}

json format_embeddings_response_oaicompat(
        const json & request,
        const std::string & model_name,
        const json & embeddings,
        bool use_base64) {
    json data = json::array();
    int32_t n_tokens = 0;
    int i = 0;
    for (const auto & elem : embeddings) {
        json embedding_obj;

        if (use_base64) {
            const auto& vec = json_value(elem, "embedding", json::array()).get<std::vector<float>>();
            const char* data_ptr = reinterpret_cast<const char*>(vec.data());
            size_t data_size = vec.size() * sizeof(float);
            size_t encoded_size = 0;
            uint8_t * encoded = llama_server_base64_encode(
                    reinterpret_cast<const uint8_t *>(data_ptr), data_size, &encoded_size);
            if (encoded == nullptr && encoded_size != 0) {
                throw std::runtime_error("failed to encode base64");
            }
            std::string encoded_embedding(reinterpret_cast<const char *>(encoded), encoded_size);
            llama_server_base64_free(encoded, encoded_size);
            embedding_obj = {
                {"embedding", encoded_embedding},
                {"index", i++},
                {"object", "embedding"},
                {"encoding_format", "base64"}
            };
        } else {
            embedding_obj = {
                {"embedding", json_value(elem, "embedding", json::array())},
                {"index", i++},
                {"object", "embedding"}
            };
        }
        data.push_back(embedding_obj);

        n_tokens += json_value(elem, "tokens_evaluated", 0);
    }

    json res = json {
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json {
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"data", data}
    };

    return res;
}

json format_response_rerank(
        const json & request,
        const std::string & model_name,
        const json & ranks,
        bool is_tei_format,
        std::vector<std::string> & texts,
        int top_n) {
    int32_t n_tokens = 0;
    bool return_text = is_tei_format && json_value(request, "return_text", false);
    std::vector<json> elements; // Temporary vector to hold unsorted elements
    std::string score_label = is_tei_format ? "score" : "relevance_score";
    for (const auto & rank : ranks) {
        int index = json_value(rank, "index", 0);
        json elem = json{
            {"index", index},
            {score_label, json_value(rank, "score", 0.0)},
        };
        n_tokens += json_value(rank, "tokens_evaluated", 0);
        if (return_text) {
            elem["text"] = std::move(texts[index]);
        }
        elements.push_back(elem);
    }

    std::sort(elements.begin(), elements.end(), [score_label](const json& a, const json& b) {
        return llama_server_score_desc_rust(
            json_value(a, score_label, 0.0),
            json_value(b, score_label, 0.0));
    });

    elements.resize(std::min(top_n, (int)elements.size()));
    json results = elements;

    if (is_tei_format) return results;

    json res = json{
        {"model", json_value(request, "model", model_name)},
        {"object", "list"},
        {"usage", json{
            {"prompt_tokens", n_tokens},
            {"total_tokens", n_tokens}
        }},
        {"results", results}
    };

    return res;
}


//
// other utils
//

std::vector<llama_token_data> get_token_probabilities(llama_context * ctx, int idx) {
    std::vector<llama_token_data> cur;

    const auto * logits = llama_get_logits_ith(ctx, idx);
    const llama_token * sampled_ids = llama_get_sampled_candidates_ith(ctx, idx);

    const int n_logits = llama_get_sampled_logits_count_ith(ctx, idx);

    cur.resize(n_logits);
    if (sampled_ids) {
        for (int i = 0; i < n_logits; i++) {
            cur[i] = llama_token_data{sampled_ids[i], logits[i], 0.0f};
        }
    } else {
        for (llama_token token_id = 0; token_id < n_logits; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }
    }

    // sort tokens by logits
    std::sort(cur.begin(), cur.end(), [](const llama_token_data & a, const llama_token_data & b) {
        return llama_server_logit_desc_rust(a.logit, b.logit);
    });

    // apply softmax
    float max_l = cur[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < cur.size(); ++i) {
        float p = expf(cur[i].logit - max_l);
        cur[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < cur.size(); ++i) {
        cur[i].p /= cum_sum;
    }

    return cur;
}

std::string safe_json_to_str(const json & data) {
    return data.dump(-1, ' ', false, json::error_handler_t::replace);
}

// TODO: reuse llama_detokenize
template <class Iter>
static std::string tokens_to_str(const llama_vocab * ctx, Iter begin, Iter end) {
    std::string ret;
    for (; begin != end; ++begin) {
        ret += common_token_to_piece(ctx, *begin);
    }

    return ret;
}

std::string tokens_to_str(llama_context * ctx, const llama_tokens & tokens) {
    auto model = llama_get_model(ctx);
    return tokens_to_str(llama_model_get_vocab(model), tokens.begin(), tokens.end());
}

std::string tokens_to_str(const llama_vocab * vocab, const llama_tokens & tokens) {
    return tokens_to_str(vocab, tokens.begin(), tokens.end());
}

// format incomplete utf-8 multibyte character for output
std::string tokens_to_output_formatted_string(const llama_context * ctx, const llama_token token) {
    std::string out = token == LLAMA_TOKEN_NULL ? "" : common_token_to_piece(ctx, token);
    size_t out_len = 0;
    uint8_t * data = llama_server_format_token_output_rust((const uint8_t *) out.data(), out.size(), &out_len);
    if (data == nullptr) {
        return "";
    }
    std::string result((const char *) data, out_len);
    llama_server_base64_free(data, out_len);
    return result;
}

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string format_oai_sse(const json & data) {
    std::vector<std::string> serialized;
    if (data.is_array()) {
        serialized.reserve(data.size());
        for (const auto & item : data) {
            serialized.push_back(safe_json_to_str(item));
        }
    } else {
        serialized.push_back(safe_json_to_str(data));
    }

    std::vector<server_rust_string_view> views;
    views.reserve(serialized.size());
    for (const auto & item : serialized) {
        views.push_back({ (const uint8_t *) item.data(), item.size() });
    }

    size_t out_len = 0;
    uint8_t * out = llama_server_format_oai_sse_rust(views.data(), views.size(), &out_len);
    if (out == nullptr) {
        return "";
    }
    std::string result((const char *) out, out_len);
    llama_server_base64_free(out, out_len);
    return result;
}

std::string format_oai_resp_sse(const json & data) {
    std::vector<std::string> events;
    std::vector<std::string> payloads;
    auto collect = [&events, &payloads](const json & event_obj) {
        events.push_back(event_obj.at("event").get<std::string>());
        payloads.push_back(safe_json_to_str(event_obj.at("data")));
    };
    if (data.is_array()) {
        events.reserve(data.size());
        payloads.reserve(data.size());
        for (const auto & item : data) {
            collect(item);
        }
    } else {
        collect(data);
    }

    std::vector<server_rust_sse_event_view> views;
    views.reserve(events.size());
    for (size_t i = 0; i < events.size(); ++i) {
        views.push_back({
                { (const uint8_t *) events[i].data(), events[i].size() },
                { (const uint8_t *) payloads[i].data(), payloads[i].size() }
        });
    }

    size_t out_len = 0;
    uint8_t * out = llama_server_format_oai_resp_sse_rust(views.data(), views.size(), &out_len);
    if (out == nullptr) {
        return "";
    }
    std::string result((const char *) out, out_len);
    llama_server_base64_free(out, out_len);
    return result;
}

std::string format_anthropic_sse(const json & data) {
    std::vector<std::string> events;
    std::vector<std::string> payloads;
    std::vector<uint8_t> has_events;
    auto collect = [&events, &payloads, &has_events](const json & event_obj) {
        if (event_obj.contains("event") && event_obj.contains("data")) {
            events.push_back(event_obj.at("event").get<std::string>());
            payloads.push_back(safe_json_to_str(event_obj.at("data")));
            has_events.push_back(1);
        } else {
            events.push_back("");
            payloads.push_back(safe_json_to_str(event_obj));
            has_events.push_back(0);
        }
    };

    if (data.is_array()) {
        events.reserve(data.size());
        payloads.reserve(data.size());
        has_events.reserve(data.size());
        for (const auto & event : data) {
            collect(event);
        }
    } else {
        collect(data);
    }

    std::vector<server_rust_anthropic_sse_event_view> views;
    views.reserve(events.size());
    for (size_t i = 0; i < events.size(); ++i) {
        views.push_back({
                { (const uint8_t *) events[i].data(), events[i].size() },
                { (const uint8_t *) payloads[i].data(), payloads[i].size() },
                has_events[i],
        });
    }

    size_t out_len = 0;
    uint8_t * out = llama_server_format_anthropic_sse_rust(views.data(), views.size(), &out_len);
    if (out == nullptr) {
        return "";
    }
    std::string result((const char *) out, out_len);
    llama_server_base64_free(out, out_len);
    return result;
}

bool is_valid_utf8(const std::string & str) {
    return llama_server_is_valid_utf8_rust((const uint8_t *) str.data(), str.size());
}

llama_tokens format_prompt_infill(
        const llama_vocab * vocab,
        const json & input_prefix,
        const json & input_suffix,
        const json & input_extra,
        const int n_batch,
        const int n_predict,
        const int n_ctx,
        const bool spm_infill,
        const llama_tokens & tokens_prompt
    ) {
    // TODO: optimize this block by reducing memory allocations and movement

    // use FIM repo-level pattern:
    // ref: https://arxiv.org/pdf/2409.12186
    //
    // [FIM_REP]myproject
    // [FIM_SEP]filename0
    // extra chunk 0
    // [FIM_SEP]filename1
    // extra chunk 1
    // ...
    // [FIM_SEP]filename
    // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
    //
    llama_tokens extra_tokens;
    extra_tokens.reserve(n_ctx);

    auto tokens_prefix = tokenize_mixed(vocab, input_prefix, false, false);
    auto tokens_suffix = tokenize_mixed(vocab, input_suffix, false, false);

    if (llama_vocab_fim_rep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: make project name an input
        static const auto k_fim_repo = common_tokenize(vocab, "myproject\n", false, false);

        extra_tokens.push_back(llama_vocab_fim_rep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_repo.begin(), k_fim_repo.end());
    }
    for (const auto & chunk : input_extra) {
        // { "text": string, "filename": string }
        const std::string text     = json_value(chunk, "text",     std::string());
        const std::string filename = json_value(chunk, "filename", std::string("tmp"));

        if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
            const auto k_fim_file = common_tokenize(vocab, filename + "\n", false, false);

            extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
            extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
        } else {
            // chunk separator in binary form to avoid confusing the AI
            static const char k_chunk_prefix_str[] = {0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70, 0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00};
            static const auto k_chunk_prefix_tokens = common_tokenize(vocab, k_chunk_prefix_str, false, false);

            extra_tokens.insert(extra_tokens.end(), k_chunk_prefix_tokens.begin(), k_chunk_prefix_tokens.end());
        }

        const auto chunk_tokens = common_tokenize(vocab, text, false, false);
        extra_tokens.insert(extra_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
    }

    if (llama_vocab_fim_sep(vocab) != LLAMA_TOKEN_NULL) {
        // TODO: current filename
        static const auto k_fim_file = common_tokenize(vocab, "filename\n", false, false);

        extra_tokens.insert(extra_tokens.end(), llama_vocab_fim_sep(vocab));
        extra_tokens.insert(extra_tokens.end(), k_fim_file.begin(), k_fim_file.end());
    }

    // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
    const int n_prefix_take = std::min<int>(tokens_prefix.size(),                3*(n_batch/4));
    const int n_suffix_take = std::min<int>(tokens_suffix.size(), std::max<int>(0, (n_batch/4) - (2 + tokens_prompt.size())));

    SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", n_prefix_take, n_suffix_take, (n_prefix_take + n_suffix_take));

    // fill the rest of the context with extra chunks
    const int n_extra_take = std::min<int>(std::max<int>(0, n_ctx - (n_batch) - 2*n_predict), extra_tokens.size());

    tokens_prefix.erase(tokens_prefix.begin(), tokens_prefix.begin() + tokens_prefix.size() - n_prefix_take);
    tokens_suffix.resize(n_suffix_take);

    tokens_prefix.insert(tokens_prefix.begin(), llama_vocab_fim_pre(vocab));
    tokens_prefix.insert(tokens_prefix.end(),   tokens_prompt.begin(), tokens_prompt.end());
    tokens_suffix.insert(tokens_suffix.begin(), llama_vocab_fim_suf(vocab));

    auto embd_inp = spm_infill ? tokens_suffix : tokens_prefix;
    auto embd_end = spm_infill ? tokens_prefix : tokens_suffix;

    if (llama_vocab_get_add_bos(vocab)) {
        embd_inp.insert(embd_inp.begin(), llama_vocab_bos(vocab));
    }

    SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", n_ctx, n_extra_take, (int) extra_tokens.size());

    // put the extra context before the FIM prefix
    embd_inp.insert(embd_inp.begin(), extra_tokens.end() - n_extra_take, extra_tokens.end());

    embd_inp.insert(embd_inp.end(), embd_end.begin(), embd_end.end());
    embd_inp.push_back(llama_vocab_fim_mid(vocab));

    return embd_inp;
}

server_tokens format_prompt_rerank(
        const struct llama_model * model,
        const struct llama_vocab * vocab,
        mtmd_context * mctx,
        const std::string & query,
        const std::string & doc) {
    server_tokens result = {};

    const char * rerank_prompt = llama_model_chat_template(model, "rerank");

    if (rerank_prompt != nullptr) {
        std::string prompt = rerank_prompt;
        string_replace_all(prompt, "{query}"   , query);
        string_replace_all(prompt, "{document}", doc  );
        server_tokens tokens = tokenize_input_subprompt(vocab, mctx, prompt, false, true);
        result.push_back(tokens);
    } else {
        // Get EOS token - use SEP token as fallback if EOS is not available
        server_tokens query_tokens = tokenize_input_subprompt(vocab, mctx, query, false, false);
        server_tokens doc_tokens   = tokenize_input_subprompt(vocab, mctx, doc,   false, false);
        llama_token eos_token = llama_vocab_eos(vocab);
        if (eos_token == LLAMA_TOKEN_NULL) {
            eos_token = llama_vocab_sep(vocab);
        }

        if (llama_vocab_get_add_bos(vocab)) {
            result.push_back(llama_vocab_bos(vocab));
        }
        result.push_back(query_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
        if (llama_vocab_get_add_sep(vocab)) {
            result.push_back(llama_vocab_sep(vocab));
        }
        result.push_back(doc_tokens);
        if (llama_vocab_get_add_eos(vocab)) {
            result.push_back(eos_token);
        }
    }

    return result;
}


// ---- server chat conversion implementation ----

#include "common.h.inc"

#include <sstream>

json server_chat_convert_responses_to_chatcmpl(const json & body);
json server_chat_convert_anthropic_to_oai(const json & body);
json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);

json server_chat_convert_responses_to_chatcmpl(const json & response_body) {
    if (!response_body.contains("input")) {
        throw std::invalid_argument("'input' is required");
    }
    if (!json_value(response_body, "previous_response_id", std::string{}).empty()) {
        throw std::invalid_argument("llama.cpp does not support 'previous_response_id'.");
    }

    const json input_value = response_body.at("input");
    json chatcmpl_body = response_body;
    chatcmpl_body.erase("input");
    std::vector<json> chatcmpl_messages;

    if (response_body.contains("instructions")) {
        chatcmpl_messages.push_back({
            {"role",    "system"},
            {"content", json_value(response_body, "instructions", std::string())},
        });
        chatcmpl_body.erase("instructions");
    }

    if (input_value.is_string()) {
        // #responses_create-input-text_input
        chatcmpl_messages.push_back({
            {"role",    "user"},
            {"content", input_value},
        });
    } else if (input_value.is_array()) {
        // #responses_create-input-input_item_list

        static auto exists_and_is_array = [](const json & j, const char * key) -> bool {
            const bool has_key = j.contains(key);
            return llama_server_json_field_is_array_rust(has_key, has_key && j.at(key).is_array());
        };
        static auto exists_and_is_string = [](const json & j, const char * key) -> bool {
            const bool has_key = j.contains(key);
            return llama_server_json_field_is_string_rust(has_key, has_key && j.at(key).is_string());
        };

        for (json item : input_value) {
            bool merge_prev = !chatcmpl_messages.empty() && chatcmpl_messages.back().value("role", "") == "assistant";

            if (exists_and_is_string(item, "content")) {
                // #responses_create-input-input_item_list-input_message-content-text_input
                // Only "Input message" contains item["content"]::string
                // After converting item["content"]::string to item["content"]::array,
                // we can treat "Input message" as sum of "Item-Input message" and "Item-Output message"
                item["content"] = json::array({
                    json {
                        {"text", item.at("content")},
                        {"type", "input_text"}
                    }
                });
            }

            if (exists_and_is_array(item, "content") &&
                exists_and_is_string(item, "role") &&
                (item.at("role") == "user" ||
                    item.at("role") == "system" ||
                    item.at("role") == "developer")
            ) {
                // #responses_create-input-input_item_list-item-input_message
                std::vector<json> chatcmpl_content;

                for (const json & input_item : item.at("content")) {
                    const std::string type = json_value(input_item, "type", std::string());

                    if (type == "input_text") {
                        if (!input_item.contains("text")) {
                            throw std::invalid_argument("'Input text' requires 'text'");
                        }
                        chatcmpl_content.push_back({
                            {"text", input_item.at("text")},
                            {"type", "text"},
                        });
                    } else if (type == "input_image") {
                        // While `detail` is marked as required,
                        // it has default value("auto") and can be omitted.

                        if (!input_item.contains("image_url")) {
                            throw std::invalid_argument("'image_url' is required");
                        }
                        chatcmpl_content.push_back({
                            {"image_url", json {
                                {"url", input_item.at("image_url")}
                            }},
                            {"type", "image_url"},
                        });
                    } else if (type == "input_file") {
                        throw std::invalid_argument("'input_file' is not supported by llamacpp at this moment");
                    } else {
                        throw std::invalid_argument("'type' must be one of 'input_text', 'input_image', or 'input_file'");
                    }
                }

                if (item.contains("type")) {
                    item.erase("type");
                }
                if (item.contains("status")) {
                    item.erase("status");
                }
                item["content"] = chatcmpl_content;

                chatcmpl_messages.push_back(item);
            } else if (exists_and_is_string(item, "role") &&
                item.at("role") == "assistant" &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "message"
            ) {
                // #responses_create-input-input_item_list-item-output_message
                auto chatcmpl_content = json::array();

                // Handle both string content and array content
                if (item.contains("content") && item.at("content").is_string()) {
                    // String content - convert to text content part
                    chatcmpl_content.push_back({
                        {"text", item.at("content")},
                        {"type", "text"},
                    });
                } else if (exists_and_is_array(item, "content")) {
                    // Array content - process each item
                    for (const auto & output_text : item.at("content")) {
                        const std::string type = json_value(output_text, "type", std::string());
                        if (type == "output_text" || type == "input_text") {
                            // Accept both output_text and input_text (string content gets converted to input_text)
                            if (!exists_and_is_string(output_text, "text")) {
                                throw std::invalid_argument("'Output text' requires 'text'");
                            }
                            chatcmpl_content.push_back({
                                {"text", output_text.at("text")},
                                {"type", "text"},
                            });
                        } else if (type == "refusal") {
                            if (!exists_and_is_string(output_text, "refusal")) {
                                throw std::invalid_argument("'Refusal' requires 'refusal'");
                            }
                            chatcmpl_content.push_back({
                                {"refusal", output_text.at("refusal")},
                                {"type", "refusal"},
                            });
                        } else {
                            throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
                        }
                    }
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "content")) {
                        prev_msg["content"] = json::array();
                    }
                    auto & prev_content = prev_msg["content"];
                    prev_content.insert(prev_content.end(), chatcmpl_content.begin(), chatcmpl_content.end());
                } else {
                    item.erase("status");
                    item.erase("type");
                    item["content"] = chatcmpl_content;
                    chatcmpl_messages.push_back(item);
                }
            } else if (exists_and_is_string(item, "arguments") &&
                exists_and_is_string(item, "call_id") &&
                exists_and_is_string(item, "name") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call
                json tool_call = {
                    {"function", json {
                        {"arguments", item.at("arguments")},
                        {"name",      item.at("name")},
                    }},
                    {"id",   item.at("call_id")},
                    {"type", "function"},
                };

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    if (!exists_and_is_array(prev_msg, "tool_calls")) {
                        prev_msg["tool_calls"] = json::array();
                    }
                    prev_msg["tool_calls"].push_back(tool_call);
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role",       "assistant"},
                        {"tool_calls", json::array({tool_call})}
                    });
                }
            } else if (exists_and_is_string(item, "call_id") &&
                (exists_and_is_string(item, "output") || exists_and_is_array(item, "output")) &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "function_call_output"
            ) {
                // #responses_create-input-input_item_list-item-function_tool_call_output
                if (item.at("output").is_string()) {
                    chatcmpl_messages.push_back(json {
                        {"content",      item.at("output")},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                } else {
                    json chatcmpl_outputs = item.at("output");
                    for (json & chatcmpl_output : chatcmpl_outputs) {
                        if (!chatcmpl_output.contains("type") || chatcmpl_output.at("type") != "input_text") {
                            throw std::invalid_argument("Output of tool call should be 'Input text'");
                        }
                        chatcmpl_output["type"] = "text";
                    }
                    chatcmpl_messages.push_back(json {
                        {"content",      chatcmpl_outputs},
                        {"role",         "tool"},
                        {"tool_call_id", item.at("call_id")},
                    });
                }
            } else if (exists_and_is_array(item, "summary") &&
                exists_and_is_string(item, "type") &&
                item.at("type") == "reasoning") {
                // #responses_create-input-input_item_list-item-reasoning

                if (!exists_and_is_array(item, "content")) {
                    throw std::invalid_argument("item['content'] is not an array");
                }
                if (item.at("content").empty()) {
                    throw std::invalid_argument("item['content'] is empty");
                }
                if (!exists_and_is_string(item.at("content")[0], "text")) {
                    throw std::invalid_argument("item['content']['text'] is not a string");
                }

                if (merge_prev) {
                    auto & prev_msg = chatcmpl_messages.back();
                    prev_msg["reasoning_content"] = item.at("content")[0].at("text");
                } else {
                    chatcmpl_messages.push_back(json {
                        {"role", "assistant"},
                        {"content", json::array()},
                        {"reasoning_content", item.at("content")[0].at("text")},
                    });
                }
            } else {
                throw std::invalid_argument("Cannot determine type of 'item'");
            }
        }
    } else {
        throw std::invalid_argument("'input' must be a string or array of objects");
    }

    chatcmpl_body["messages"] = chatcmpl_messages;

    if (response_body.contains("tools")) {
        if (!response_body.at("tools").is_array()) {
            throw std::invalid_argument("'tools' must be an array of objects");
        }
        std::vector<json> chatcmpl_tools;
        for (json resp_tool : response_body.at("tools")) {
            json chatcmpl_tool;

            if (json_value(resp_tool, "type", std::string()) != "function") {
                throw std::invalid_argument("'type' of tool must be 'function'");
            }
            resp_tool.erase("type");
            chatcmpl_tool["type"] = "function";

            if (!resp_tool.contains("strict")) {
                resp_tool["strict"] = true;
            }
            chatcmpl_tool["function"] = resp_tool;
            chatcmpl_tools.push_back(chatcmpl_tool);
        }
        chatcmpl_body.erase("tools");
        chatcmpl_body["tools"] = chatcmpl_tools;
    }

    if (response_body.contains("max_output_tokens")) {
        chatcmpl_body.erase("max_output_tokens");
        chatcmpl_body["max_tokens"] = response_body["max_output_tokens"];
    }

    return chatcmpl_body;
}

// Edits the cch section of an "x-anthropic-billing-header" system prompt.
// Does nothing to any other prompt.
//
// This is a claude message with a "cch=ef01a" attribute that breaks prefix caching.
// The cch stamp is a whitebox end-to-end integrity hint. It's not meaningful as a
// system prompt data, particularly to llama.cpp, but its presence means the prefix
// cache will not get past it: It changes on each request.
//
// Reference: https://github.com/ggml-org/llama.cpp/pull/21793
// Example header:
// ```
// x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code, Anthropic's official CLI for Claude.
//                                                                            ^^^^^
// ```
static void normalize_anthropic_billing_header(std::string & system_text) {
    int result = llama_server_normalize_anthropic_billing_header_rust(
            reinterpret_cast<uint8_t *>(system_text.data()), system_text.size());
    if (result < 0) {
        LOG_ERR("anthropic string not as expected: %s", system_text.c_str());
    }
}

json server_chat_convert_anthropic_to_oai(const json & body) {
    json oai_body;

    // Convert system prompt
    json oai_messages = json::array();
    auto system_param = json_value(body, "system", json());
    if (!system_param.is_null()) {
        std::string system_content;

        if (system_param.is_string()) {
            system_content = system_param.get<std::string>();
            normalize_anthropic_billing_header(system_content);
        } else if (system_param.is_array()) {
            for (const auto & block : system_param) {
                if (json_value(block, "type", std::string()) == "text") {
                    auto system_text = json_value(block, "text", std::string());
                    normalize_anthropic_billing_header(system_text);
                    system_content += system_text;
                }
            }
        }

        oai_messages.push_back({
            {"role", "system"},
            {"content", system_content}
        });
    }

    // Convert messages
    if (!body.contains("messages")) {
        throw std::runtime_error("'messages' is required");
    }
    const json & messages = body.at("messages");
    if (messages.is_array()) {
        for (const auto & msg : messages) {
            std::string role = json_value(msg, "role", std::string());

            if (!msg.contains("content")) {
                if (role == "assistant") {
                    continue;
                }
                oai_messages.push_back(msg);
                continue;
            }

            const json & content = msg.at("content");

            if (content.is_string()) {
                oai_messages.push_back(msg);
                continue;
            }

            if (!content.is_array()) {
                oai_messages.push_back(msg);
                continue;
            }

            json tool_calls = json::array();
            json converted_content = json::array();
            json tool_results = json::array();
            std::string reasoning_content;
            bool has_tool_calls = false;

            for (const auto & block : content) {
                std::string type = json_value(block, "type", std::string());

                if (type == "text") {
                    converted_content.push_back(block);
                } else if (type == "thinking") {
                    reasoning_content += json_value(block, "thinking", std::string());
                } else if (type == "image") {
                    json source = json_value(block, "source", json::object());
                    std::string source_type = json_value(source, "type", std::string());

                    if (source_type == "base64") {
                        std::string media_type = json_value(source, "media_type", std::string("image/jpeg"));
                        std::string data = json_value(source, "data", std::string());
                        std::ostringstream ss;
                        ss << "data:" << media_type << ";base64," << data;

                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", ss.str()}
                            }}
                        });
                    } else if (source_type == "url") {
                        std::string url = json_value(source, "url", std::string());
                        converted_content.push_back({
                            {"type", "image_url"},
                            {"image_url", {
                                {"url", url}
                            }}
                        });
                    }
                } else if (type == "tool_use") {
                    tool_calls.push_back({
                        {"id", json_value(block, "id", std::string())},
                        {"type", "function"},
                        {"function", {
                            {"name", json_value(block, "name", std::string())},
                            {"arguments", json_value(block, "input", json::object()).dump()}
                        }}
                    });
                    has_tool_calls = true;
                } else if (type == "tool_result") {
                    std::string tool_use_id = json_value(block, "tool_use_id", std::string());

                    auto result_content = json_value(block, "content", json());
                    std::string result_text;
                    if (result_content.is_string()) {
                        result_text = result_content.get<std::string>();
                    } else if (result_content.is_array()) {
                        for (const auto & c : result_content) {
                            if (json_value(c, "type", std::string()) == "text") {
                                result_text += json_value(c, "text", std::string());
                            }
                        }
                    }

                    tool_results.push_back({
                        {"role", "tool"},
                        {"tool_call_id", tool_use_id},
                        {"content", result_text}
                    });
                }
            }

            if (!converted_content.empty() || has_tool_calls || !reasoning_content.empty()) {
                json new_msg = {{"role", role}};
                if (!converted_content.empty()) {
                    new_msg["content"] = converted_content;
                } else if (has_tool_calls || !reasoning_content.empty()) {
                    new_msg["content"] = "";
                }
                if (!tool_calls.empty()) {
                    new_msg["tool_calls"] = tool_calls;
                }
                if (!reasoning_content.empty()) {
                    new_msg["reasoning_content"] = reasoning_content;
                }
                oai_messages.push_back(new_msg);
            }

            for (const auto & tool_msg : tool_results) {
                oai_messages.push_back(tool_msg);
            }
        }
    }

    oai_body["messages"] = oai_messages;

    // Convert tools
    if (body.contains("tools")) {
        const json & tools = body.at("tools");
        if (tools.is_array()) {
            json oai_tools = json::array();
            for (const auto & tool : tools) {
                oai_tools.push_back({
                    {"type", "function"},
                    {"function", {
                        {"name", json_value(tool, "name", std::string())},
                        {"description", json_value(tool, "description", std::string())},
                        {"parameters", tool.contains("input_schema") ? tool.at("input_schema") : json::object()}
                    }}
                });
            }
            oai_body["tools"] = oai_tools;
        }
    }

    // Convert tool_choice
    if (body.contains("tool_choice")) {
        const json & tc = body.at("tool_choice");
        if (tc.is_object()) {
            std::string type = json_value(tc, "type", std::string());
            if (type == "auto") {
                oai_body["tool_choice"] = "auto";
            } else if (type == "any" || type == "tool") {
                oai_body["tool_choice"] = "required";
            }
        }
    }

    // Convert stop_sequences to stop
    if (body.contains("stop_sequences")) {
        oai_body["stop"] = body.at("stop_sequences");
    }

    // Handle max_tokens (required in Anthropic, but we're permissive)
    if (body.contains("max_tokens")) {
        oai_body["max_tokens"] = body.at("max_tokens");
    } else {
        oai_body["max_tokens"] = 4096;
    }

    // Pass through common params
    for (const auto & key : {"temperature", "top_p", "top_k", "stream", "chat_template_kwargs"}) {
        if (body.contains(key)) {
            oai_body[key] = body.at(key);
        }
    }

    // Handle Anthropic-specific thinking param
    if (body.contains("thinking")) {
        json thinking = json_value(body, "thinking", json::object());
        std::string thinking_type = json_value(thinking, "type", std::string());
        if (thinking_type == "enabled") {
            int budget_tokens = json_value(thinking, "budget_tokens", 10000);
            oai_body["thinking_budget_tokens"] = budget_tokens;
        }
    }

    // Handle Anthropic-specific metadata param
    if (body.contains("metadata")) {
        json metadata = json_value(body, "metadata", json::object());
        std::string user_id = json_value(metadata, "user_id", std::string());
        if (!user_id.empty()) {
            oai_body["__metadata_user_id"] = user_id;
        }
    }

    return oai_body;
}

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff) {
    json delta = json::object();
    if (!diff.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
        delta["content"] = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
        json tool_call;
        tool_call["index"] = diff.tool_call_index;
        if (!diff.tool_call_delta.id.empty()) {
            tool_call["id"]   = diff.tool_call_delta.id;
            tool_call["type"] = "function";
        }
        if (!diff.tool_call_delta.name.empty() || !diff.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!diff.tool_call_delta.name.empty()) {
                function["name"] = diff.tool_call_delta.name;
            }
            if (!diff.tool_call_delta.arguments.empty()) {
                function["arguments"] = diff.tool_call_delta.arguments;
            }
            tool_call["function"] = function;
        }
        delta["tool_calls"] = json::array({ tool_call });
    }
    return delta;
}

json convert_transcriptions_to_chatcmpl(
        const json & inp_body,
        const common_chat_templates * tmpls,
        const std::map<std::string, uploaded_file> & in_files,
        std::vector<raw_buffer> & out_files) {
    // TODO @ngxson : this function may need to be improved in the future
    // handle input files
    out_files.clear();
    auto it = in_files.find("file");
    if (it != in_files.end()) {
        out_files.push_back(it->second.data);
    } else {
        throw std::invalid_argument("No input file found for transcription");
    }

    // handle input data
    std::string prompt          = json_value(inp_body, "prompt", std::string());
    std::string language        = json_value(inp_body, "language", std::string());
    std::string response_format = json_value(inp_body, "response_format", std::string("json"));
    if (response_format != "json") {
        throw std::invalid_argument("Only 'json' response_format is supported for transcription");
    }
    const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls);
    if (prompt.empty()) {
        prompt = preset.user;
    }
    if (!language.empty()) {
        prompt += string_format(" (language: %s)", language.c_str());
    }
    prompt += get_media_marker();

    json messages = json::array();
    if (!preset.system.empty()) {
        messages.push_back({{"role", "system"}, {"content", preset.system}});
    }
    messages.push_back({{"role", "user"}, {"content", prompt}});

    json chatcmpl_body = inp_body; // copy all fields
    chatcmpl_body["messages"] = messages;

    // because input from form-data, everything is string, we need to correct the types here
    std::string stream = json_value(inp_body, "stream", std::string("false"));
    chatcmpl_body["stream"] = stream == "true";

    if (inp_body.contains("max_tokens")) {
        std::string inp = inp_body["max_tokens"].get<std::string>();
        chatcmpl_body["max_tokens"] = std::stoul(inp);
    }

    if (inp_body.contains("temperature")) {
        std::string inp = inp_body["temperature"].get<std::string>();
        chatcmpl_body["temperature"] = std::stof(inp);
    }

    return chatcmpl_body;
}


// ---- server task implementation ----


#include "common.h.inc"
#include "common.h.inc"
#include "llama.h.inc"

using json = nlohmann::ordered_json;

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);

extern "C" const char * llama_build_info(void);

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.type",          common_speculative_type_to_str(speculative.type)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.type",          common_speculative_type_to_str(speculative.type)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// server_task
//

task_params server_task::params_from_json_cmpl(
        const llama_vocab * vocab,
        const common_params & params_base,
        const int n_ctx_slot,
        const std::vector<llama_logit_bias> & logit_bias_eog,
        const json & data) {
    task_params params;

    // Sampling parameter defaults are loaded from the global server context (but individual requests can still them)
    task_params defaults;
    defaults.sampling      = params_base.sampling;
    defaults.speculative   = params_base.speculative;
    defaults.n_keep        = params_base.n_keep;
    defaults.n_predict     = params_base.n_predict;
    defaults.n_cache_reuse = params_base.n_cache_reuse;
    defaults.cache_prompt  = params_base.cache_prompt;
    defaults.antiprompt    = params_base.antiprompt;

    // enabling this will output extra debug information in the HTTP responses from the server
    params.verbose           = params_base.verbosity > 9;
    params.timings_per_token = json_value(data, "timings_per_token", false);

    params.stream           = json_value(data,       "stream",             false);
    auto stream_opt         = json_value(data,       "stream_options",     json::object());
    params.include_usage    = json_value(stream_opt, "include_usage",      false);
    params.cache_prompt     = json_value(data,       "cache_prompt",       defaults.cache_prompt);
    params.return_tokens    = json_value(data,       "return_tokens",      false);
    params.return_progress  = json_value(data,       "return_progress",    false);
    auto max_tokens         = json_value(data,       "max_tokens",         defaults.n_predict);
    params.n_predict        = json_value(data,       "n_predict",          json_value(data, "max_completion_tokens", max_tokens));
    params.n_indent         = json_value(data,       "n_indent",           defaults.n_indent);
    params.n_keep           = json_value(data,       "n_keep",             defaults.n_keep);
    params.n_discard        = json_value(data,       "n_discard",          defaults.n_discard);
    params.n_discard        = std::max(0, params.n_discard);
    params.n_cmpl           = json_value(data,       "n_cmpl",             json_value(data, "n", 1));
    params.n_cache_reuse    = json_value(data,       "n_cache_reuse",      defaults.n_cache_reuse);
    //params.t_max_prompt_ms  = json_value(data,       "t_max_prompt_ms",    defaults.t_max_prompt_ms); // TODO: implement
    params.t_max_predict_ms = json_value(data,       "t_max_predict_ms",   defaults.t_max_predict_ms);
    params.response_fields  = json_value(data,       "response_fields",    std::vector<std::string>());

    params.sampling.top_k              = json_value(data, "top_k",               defaults.sampling.top_k);
    params.sampling.top_p              = json_value(data, "top_p",               defaults.sampling.top_p);
    params.sampling.min_p              = json_value(data, "min_p",               defaults.sampling.min_p);
    params.sampling.top_n_sigma        = json_value(data, "top_n_sigma",         defaults.sampling.top_n_sigma);
    params.sampling.xtc_probability    = json_value(data, "xtc_probability",     defaults.sampling.xtc_probability);
    params.sampling.xtc_threshold      = json_value(data, "xtc_threshold",       defaults.sampling.xtc_threshold);
    params.sampling.typ_p              = json_value(data, "typical_p",           defaults.sampling.typ_p);
    params.sampling.temp               = json_value(data, "temperature",         defaults.sampling.temp);
    params.sampling.dynatemp_range     = json_value(data, "dynatemp_range",      defaults.sampling.dynatemp_range);
    params.sampling.dynatemp_exponent  = json_value(data, "dynatemp_exponent",   defaults.sampling.dynatemp_exponent);
    params.sampling.penalty_last_n     = json_value(data, "repeat_last_n",       defaults.sampling.penalty_last_n);
    params.sampling.penalty_repeat     = json_value(data, "repeat_penalty",      defaults.sampling.penalty_repeat);
    params.sampling.penalty_freq       = json_value(data, "frequency_penalty",   defaults.sampling.penalty_freq);
    params.sampling.penalty_present    = json_value(data, "presence_penalty",    defaults.sampling.penalty_present);
    params.sampling.dry_multiplier     = json_value(data, "dry_multiplier",      defaults.sampling.dry_multiplier);
    params.sampling.dry_base           = json_value(data, "dry_base",            defaults.sampling.dry_base);
    params.sampling.dry_allowed_length = json_value(data, "dry_allowed_length",  defaults.sampling.dry_allowed_length);
    params.sampling.dry_penalty_last_n = json_value(data, "dry_penalty_last_n",  defaults.sampling.dry_penalty_last_n);
    params.sampling.mirostat           = json_value(data, "mirostat",            defaults.sampling.mirostat);
    params.sampling.mirostat_tau       = json_value(data, "mirostat_tau",        defaults.sampling.mirostat_tau);
    params.sampling.mirostat_eta       = json_value(data, "mirostat_eta",        defaults.sampling.mirostat_eta);
    params.sampling.adaptive_target    = json_value(data, "adaptive_target",     defaults.sampling.adaptive_target);
    params.sampling.adaptive_decay     = json_value(data, "adaptive_decay",      defaults.sampling.adaptive_decay);
    params.sampling.seed               = json_value(data, "seed",                defaults.sampling.seed);
    params.sampling.n_probs            = json_value(data, "n_probs",             defaults.sampling.n_probs);
    params.sampling.min_keep           = json_value(data, "min_keep",            defaults.sampling.min_keep);
    params.sampling.backend_sampling   = json_value(data, "backend_sampling",    defaults.sampling.backend_sampling);
    params.post_sampling_probs         = json_value(data, "post_sampling_probs", defaults.post_sampling_probs);

    params.speculative = defaults.speculative;

    // TODO: for now, be able to adjust only the draft-model based speculative parameters
    params.speculative.draft.n_min = json_value(data, "speculative.n_min", defaults.speculative.draft.n_min);
    params.speculative.draft.n_max = json_value(data, "speculative.n_max", defaults.speculative.draft.n_max);
    params.speculative.draft.p_min = json_value(data, "speculative.p_min", defaults.speculative.draft.p_min);

    params.speculative.draft.n_min = std::min(params.speculative.draft.n_max, params.speculative.draft.n_min);
    params.speculative.draft.n_min = std::max(params.speculative.draft.n_min, 0);
    params.speculative.draft.n_max = std::max(params.speculative.draft.n_max, 0);

#if 0
    // for debugging and research purposes
    params.speculative.type = common_speculative_type_from_name(json_value(data, "speculative.type", common_speculative_type_to_str(defaults.speculative.type)));

    params.speculative.ngram_size_n     = json_value(data, "speculative.ngram_size_n", defaults.speculative.ngram_size_n);
    params.speculative.ngram_size_m     = json_value(data, "speculative.ngram_size_m", defaults.speculative.ngram_size_m);
    params.speculative.ngram_min_hits   = json_value(data, "speculative.ngram_m_hits", defaults.speculative.ngram_min_hits);

    params.speculative.ngram_size_n     = std::max(std::min(1, (int) params.speculative.ngram_size_n),     1024);
    params.speculative.ngram_size_m     = std::max(std::min(1, (int) params.speculative.ngram_size_m),     1024);
    params.speculative.ngram_min_hits   = std::max(std::min(1, (int) params.speculative.ngram_min_hits),   1024);
#endif

    // Use OpenAI API logprobs only if n_probs wasn't provided
    if (data.contains("logprobs") && params.sampling.n_probs == defaults.sampling.n_probs){
        params.sampling.n_probs = json_value(data, "logprobs", defaults.sampling.n_probs);
    }

    if (data.contains("lora")) {
        if (data.at("lora").is_array()) {
            params.lora = parse_lora_request(data.at("lora"));
        } else {
            throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
        }
    } else {
        params.lora = {};
    }

    // TODO: add more sanity checks for the input parameters

    if (params.sampling.penalty_last_n < -1) {
        throw std::runtime_error("Error: repeat_last_n must be >= -1");
    }

    if (params.sampling.dry_penalty_last_n < -1) {
        throw std::runtime_error("Error: dry_penalty_last_n must be >= -1");
    }

    if (params.sampling.penalty_last_n == -1) {
        // note: should be the slot's context and not the full context, but it's ok
        params.sampling.penalty_last_n = n_ctx_slot;
    }

    if (params.sampling.dry_penalty_last_n == -1) {
        params.sampling.dry_penalty_last_n = n_ctx_slot;
    }

    if (params.sampling.dry_base < 1.0f) {
        params.sampling.dry_base = defaults.sampling.dry_base;
    }

    // sequence breakers for DRY
    {
        // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
        // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39

        if (data.contains("dry_sequence_breakers")) {
            params.sampling.dry_sequence_breakers = json_value(data, "dry_sequence_breakers", std::vector<std::string>());
            if (params.sampling.dry_sequence_breakers.empty()) {
                throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
            }
        }
    }

    // process "json_schema" and "grammar"
    if (data.contains("json_schema") && !data.contains("grammar")) {
        try {
            auto schema                  = json_value(data, "json_schema", json::object());
            SRV_DBG("JSON schema: %s\n", schema.dump(2).c_str());
            std::string grammar_str      = json_schema_to_grammar(schema);
            SRV_DBG("Converted grammar: %s\n", grammar_str.c_str());
            params.sampling.grammar      = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, std::move(grammar_str)};
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("\"json_schema\": ") + e.what());
        }
    } else {
        params.sampling.grammar = defaults.sampling.grammar;

        std::string grammar_str = json_value(data, "grammar", std::string());
        if (!grammar_str.empty()) {
            // grammar_type key is set by the server when converting chat template grammars
            std::string grammar_type = json_value(data, "grammar_type", std::string());
            if (grammar_type == "tool_calls") {
                params.sampling.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, std::move(grammar_str)};
            } else {
                // explicit grammar from the user (API field "grammar")
                params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, std::move(grammar_str)};
            }
            SRV_DBG("Grammar (%s): %s\n", grammar_type.c_str(), common_grammar_value(params.sampling.grammar).c_str());
        }
        params.sampling.grammar_lazy = json_value(data, "grammar_lazy", defaults.sampling.grammar_lazy);
        SRV_DBG("Grammar lazy: %s\n", params.sampling.grammar_lazy ? "true" : "false");
    }

    {
        auto it = data.find("chat_format");
        if (it != data.end()) {
            params.chat_parser_params.format = static_cast<common_chat_format>(it->get<int>());
            SRV_INF("Chat format: %s\n", common_chat_format_name(params.chat_parser_params.format));
        } else {
            params.chat_parser_params.format = defaults.chat_parser_params.format;
        }
        common_reasoning_format reasoning_format = params_base.reasoning_format;
        if (data.contains("reasoning_format")) {
            reasoning_format = common_reasoning_format_from_name(data.at("reasoning_format").get<std::string>());
        }
        params.chat_parser_params.reasoning_format = reasoning_format;
        params.chat_parser_params.reasoning_in_content = params.stream && (reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
        params.chat_parser_params.generation_prompt = json_value(data, "generation_prompt", std::string());
        params.sampling.generation_prompt = params.chat_parser_params.generation_prompt;
        SRV_DBG("Generation prompt: '%s'\n", params.chat_parser_params.generation_prompt.c_str());
        params.chat_parser_params.parse_tool_calls = json_value(data, "parse_tool_calls", false);
        if (data.contains("chat_parser")) {
            params.chat_parser_params.parser.load(data.at("chat_parser").get<std::string>());
        }
    }

    {
        const auto preserved_tokens = data.find("preserved_tokens");
        if (preserved_tokens != data.end()) {
            for (const auto & t : *preserved_tokens) {
                auto ids = common_tokenize(vocab, t.get<std::string>(), /* add_special= */ false, /* parse_special= */ true);
                if (ids.size() == 1) {
                    SRV_DBG("Preserved token: %d\n", ids[0]);
                    params.sampling.preserved_tokens.insert(ids[0]);
                } else {
                    // This may happen when using a tool call style meant for a model with special tokens to preserve on a model without said tokens.
                    SRV_DBG("Not preserved because more than 1 token: %s\n", t.get<std::string>().c_str());
                }
            }
        }
        const auto grammar_triggers = data.find("grammar_triggers");
        if (grammar_triggers != data.end()) {
            for (const auto & t : *grammar_triggers) {
                server_grammar_trigger ct(t);
                if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
                    const auto & word = ct.value.value;
                    auto ids = common_tokenize(vocab, word, /* add_special= */ false, /* parse_special= */ true);
                    if (ids.size() == 1) {
                        auto token = ids[0];
                        if (std::find(params.sampling.preserved_tokens.begin(), params.sampling.preserved_tokens.end(), (llama_token) token) == params.sampling.preserved_tokens.end()) {
                            throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                        }
                        SRV_DBG("Grammar trigger token: %d (`%s`)\n", token, word.c_str());
                        common_grammar_trigger trigger;
                        trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                        trigger.value = word;
                        trigger.token = token;
                        params.sampling.grammar_triggers.push_back(std::move(trigger));
                    } else {
                        SRV_DBG("Grammar trigger word: `%s`\n", word.c_str());
                        params.sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
                    }
                } else {
                    if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN) {
                        SRV_DBG("Grammar trigger pattern: `%s`\n", ct.value.value.c_str());
                    } else if (ct.value.type == COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL) {
                        SRV_DBG("Grammar trigger pattern full: `%s`\n", ct.value.value.c_str());
                    } else {
                        throw std::runtime_error("Unknown grammar trigger type");
                    }
                    params.sampling.grammar_triggers.emplace_back(std::move(ct.value));
                }
            }
        }
        if (params.sampling.grammar_lazy && params.sampling.grammar_triggers.empty()) {
            throw std::runtime_error("Error: no triggers set for lazy grammar!");
        }
    }

    // Parse reasoning budget sampler parameters
    {
        const int32_t budget = json_value(data, "reasoning_budget_tokens", (int32_t) -1);
        const auto start_tag = json_value(data, "reasoning_budget_start_tag", std::string());
        const auto end_tag   = json_value(data, "reasoning_budget_end_tag", std::string());
        const auto message   = json_value(data, "reasoning_budget_message", std::string());
        params.sampling.reasoning_budget_tokens = budget;

        if (!start_tag.empty()) {
            params.sampling.reasoning_budget_start = common_tokenize(vocab, start_tag, false, true);
        }
        if (!end_tag.empty()) {
            params.sampling.reasoning_budget_end = common_tokenize(vocab, end_tag, false, true);
            params.sampling.reasoning_budget_forced = common_tokenize(vocab, message + end_tag, false, true);

            SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu toks, forced=%zu toks\n",
                budget, params.sampling.generation_prompt.c_str(),
                params.sampling.reasoning_budget_start.size(),
                params.sampling.reasoning_budget_end.size(),
                params.sampling.reasoning_budget_forced.size());
        }
    }

    {
        params.sampling.logit_bias.clear();

        const auto & logit_bias = data.find("logit_bias");
        if (logit_bias != data.end() && logit_bias->is_array()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto & el : *logit_bias) {
                // TODO: we may want to throw errors here, in case "el" is incorrect
                if (el.is_array() && el.size() == 2) {
                    float bias;
                    if (el[1].is_number()) {
                        bias = el[1].get<float>();
                    } else if (el[1].is_boolean() && !el[1].get<bool>()) {
                        bias = -INFINITY;
                    } else {
                        continue;
                    }

                    if (el[0].is_number_integer()) {
                        llama_token tok = el[0].get<llama_token>();
                        if (tok >= 0 && tok < n_vocab) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    } else if (el[0].is_string()) {
                        auto toks = common_tokenize(vocab, el[0].get<std::string>(), false);
                        for (auto tok : toks) {
                            params.sampling.logit_bias.push_back({tok, bias});
                        }
                    }
                }
            }
        } else if (logit_bias != data.end() && logit_bias->is_object()) {
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (const auto & el : logit_bias->items()) {
                float bias;
                const auto & key = el.key();
                const auto & value = el.value();
                if (value.is_number()) {
                    bias = value.get<float>();
                } else if (value.is_boolean() && !value.get<bool>()) {
                    bias = -INFINITY;
                } else {
                    continue;
                }

                char *end;
                llama_token tok = strtol(key.c_str(), &end, 10);
                if (*end == 0) {
                    if (tok >= 0 && tok < n_vocab) {
                        params.sampling.logit_bias.push_back({tok, bias});
                    }
                } else {
                    auto toks = common_tokenize(vocab, key, false);
                    for (auto tok : toks) {
                        params.sampling.logit_bias.push_back({tok, bias});
                    }
                }
            }
        }

        params.sampling.ignore_eos = json_value(data, "ignore_eos", params_base.sampling.ignore_eos);
        if (params.sampling.ignore_eos) {
            params.sampling.logit_bias.insert(
                    params.sampling.logit_bias.end(),
                    logit_bias_eog.begin(), logit_bias_eog.end());
        }
    }

    {
        params.antiprompt.clear();

        const auto & stop = data.find("stop");
        if (stop != data.end() && stop->is_array()) {
            for (const auto & word : *stop) {
                if (!word.empty()) {
                    params.antiprompt.push_back(word);
                }
            }
        }
        // set reverse prompt from cli args if not set in the request
        if (params.antiprompt.empty()) {
            params.antiprompt = defaults.antiprompt;
        }
    }

    {
        const auto samplers = data.find("samplers");
        if (samplers != data.end()) {
            if (samplers->is_array()) {
                params.sampling.samplers = common_sampler_types_from_names(*samplers, false);
            } else if (samplers->is_string()){
                params.sampling.samplers = common_sampler_types_from_chars(samplers->get<std::string>());
            }
        } else {
            params.sampling.samplers = defaults.sampling.samplers;
        }
    }

    if (params.n_cmpl > params_base.n_parallel) {
        throw std::runtime_error("n_cmpl cannot be greater than the number of slots, please increase -np");
    }

    return params;
}

//
// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    return llama_server_stop_type_to_string_rust((int) type);
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    return llama_server_probability_logarithm_rust(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes(str.size());
    const size_t copied = llama_server_copy_string_bytes_rust(
            reinterpret_cast<const uint8_t *>(str.data()), str.size(), bytes.data(), bytes.size());
    bytes.resize(copied);
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "fc_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (n_decoded == 1) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"arguments", ""},
                        {"call_id",   "fc_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }
    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return llama_server_response_type_is_oai_embd_rust(res_type)
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },

        { "slots",                           slots_data },
    };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    std::vector<size_t> sizes;
    sizes.reserve(states.size());

    for (const auto & state : states) {
        sizes.push_back(state.size());
    }

    return llama_server_sum_usize_rust(sizes.data(), sizes.size());
}

size_t server_prompt_cache::n_tokens() const {
    std::vector<size_t> counts;
    counts.reserve(states.size());

    for (const auto & state : states) {
        counts.push_back(state.n_tokens());
    }

    return llama_server_sum_usize_rust(counts.data(), counts.size());
}

server_prompt * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_WRN("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // next, remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->tokens.size()) {
            SRV_WRN(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<uint8_t> state_data;

    // check if we can allocate enough memory for the new state
    try {
        state_data.resize(state_size);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    auto & cur = states.emplace_back();
    cur = {
        /*.tokens      =*/ prompt.tokens.clone(),
        /*.data        =*/ std::move(state_data),
        /*.checkpoints =*/ prompt.checkpoints,
    };

    return &cur;
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float sim_best    = float(lcp_best) / tokens_new.size();

    SRV_WRN(" - looking for better prompt, base f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->tokens.size();
        const float sim_cur    = float(lcp_cur) / tokens_new.size();

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && sim_best < sim_cur) {
            f_keep_best = f_keep_cur;
            sim_best    = sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_WRN(" - found better prompt with f_keep = %.3f, sim = %.3f\n", f_keep_best, sim_best);

        const size_t size = it_best->data.size();
        const size_t n = llama_state_seq_set_data_ext(ctx, it_best->data.data(), size, id_slot, 0);
        if (n != size) {
            SRV_WRN("failed to restore state with size %zu\n", size);

            return false;
        }

        it_best->data.clear();
        it_best->data.shrink_to_fit();

        prompt = std::move(*it_best);

        states.erase(it_best);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        // always keep at least one state, regardless of the limits
        while (states.size() > 1 && size() > limit_size) {
            if (states.empty()) {
                break;
            }

            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (states.size() > 1 && n_tokens() > limit_tokens_cur) {
            if (states.empty()) {
                break;
            }

            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    SRV_WRN(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_WRN("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.n_tokens(), state.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}


// ---- server queue implementation ----


#include "common.h.inc"

#include <chrono>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

//
// server_queue
//

int server_queue::post(server_task && task, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(task.id != -1);
    // if this is cancel task make sure to clean up pending tasks
    if (llama_server_task_is_cancel_rust(task.type)) {
        cleanup_pending_task(task.id_target);
    }
    const int task_id = task.id;
    QUE_DBG("new task, id = %d, front = %d\n", task_id, front);
    if (front) {
        queue_tasks.push_front(std::move(task));
    } else {
        queue_tasks.push_back(std::move(task));
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
    return task_id;
}

int server_queue::post(std::vector<server_task> && tasks, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    for (auto & task : tasks) {
        if (task.id == -1) {
            task.id = id++;
        }
        // if this is cancel task make sure to clean up pending tasks
        if (llama_server_task_is_cancel_rust(task.type)) {
            cleanup_pending_task(task.id_target);
        }
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int) tasks.size(), front);
        if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
    return 0;
}

void server_queue::defer(server_task && task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    QUE_DBG("defer task, id = %d\n", task.id);
    queue_tasks_deferred.push_back(std::move(task));
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

int server_queue::get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    return new_id;
}

void server_queue::pop_deferred_task(int id_slot) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!queue_tasks_deferred.empty()) {
        // try to find a task that uses the specified slot
        bool found = false;
        for (auto it = queue_tasks_deferred.begin(); it != queue_tasks_deferred.end(); ++it) {
            if (it->id_slot == id_slot) {
                QUE_DBG("pop deferred task (use slot %d), id_task = %d\n", id_slot, it->id);
                queue_tasks.emplace_front(std::move(*it));
                queue_tasks_deferred.erase(it);
                found = true;
                break;
            }
        }
        // if not tasks found using the slot, just pop the first deferred task (default behavior)
        if (!found) {
            QUE_DBG("pop deferred task, id_task = %d\n", queue_tasks_deferred.front().id);
            queue_tasks.emplace_front(std::move(queue_tasks_deferred.front()));
            queue_tasks_deferred.pop_front();
        }
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

void server_queue::wait_until_no_sleep() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!sleeping) {
        return;
    } else {
        if (!req_stop_sleeping) {
            QUE_DBG("%s", "requesting to stop sleeping\n");
            req_stop_sleeping = true;
            condition_tasks.notify_one(); // only main thread is waiting on this
        }
        QUE_DBG("%s", "waiting until no sleep\n");
        condition_tasks.wait(lock, [&]{
            return !sleeping;
        });
    }
}

void server_queue::terminate() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    running = false;
    condition_tasks.notify_all();
}

void server_queue::start_loop(int64_t idle_sleep_ms) {
    running = true;
    time_last_task = ggml_time_ms();

    constexpr auto max_wait_time = std::chrono::seconds(1);
    auto should_sleep = [&]() -> bool {
        // caller must hold mutex_tasks
        if (idle_sleep_ms < 0) {
            return false;
        }
        int64_t now = ggml_time_ms();
        return llama_server_queue_should_sleep_rust(idle_sleep_ms, now, time_last_task);
    };

    while (true) {
        QUE_DBG("%s", "processing new tasks\n");

        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (!running) {
                QUE_DBG("%s", "terminate\n");
                return;
            }
            if (queue_tasks.empty()) {
                lock.unlock();
                break;
            }
            server_task task = std::move(queue_tasks.front());
            queue_tasks.pop_front();
            lock.unlock();

            QUE_DBG("processing task, id = %d\n", task.id);
            callback_new_task(std::move(task));
        }
        // all tasks in the current loop is processed, slots data is now ready
        QUE_DBG("%s", "update slots\n");

        // this will run the main inference process for all slots
        callback_update_slots();
        {
            // update_slots() may take a while to finish, we need to make sure it's not counted as idle
            std::unique_lock<std::mutex> lock(mutex_tasks);
            time_last_task = ggml_time_ms();
        }

        QUE_DBG("%s", "waiting for new tasks\n");
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (llama_server_queue_should_process_rust(running, queue_tasks.empty())) {
                break; // go back to process new tasks or terminate
            }

            // no tasks, check for sleeping state
            if (should_sleep()) {
                QUE_INF("%s", "entering sleeping state\n");
                sleeping = true;
                callback_sleeping_state(true);
                req_stop_sleeping = false;
                // wait until we are requested to exit sleeping state
                condition_tasks.wait(lock, [&]{
                    return llama_server_queue_sleep_wait_done_rust(running, req_stop_sleeping);
                });
                if (!running) { // may changed during sleep
                    break; // terminate
                }
                QUE_INF("%s", "exiting sleeping state\n");
                req_stop_sleeping = false;
                callback_sleeping_state(false);
                sleeping = false;
                time_last_task = ggml_time_ms();
                condition_tasks.notify_all(); // notify wait_until_no_sleep()
                break; // process new tasks
            } else {
                // wait for new tasks or timeout for checking sleeping condition
                bool res = condition_tasks.wait_for(lock, max_wait_time, [&]{
                    return llama_server_queue_task_wait_done_rust(running, queue_tasks.empty());
                });
                if (res) {
                    break; // new task arrived or terminate
                }
                // otherwise, loop again to check sleeping condition
            }
        }
    }
}

void server_queue::cleanup_pending_task(int id_target) {
    // no need lock because this is called exclusively by post()
    auto rm_func = [id_target](const server_task & task) {
        return llama_server_id_matches_rust(task.id, id_target);
    };
    queue_tasks.erase(
        std::remove_if(queue_tasks.begin(),          queue_tasks.end(),          rm_func),
        queue_tasks.end());
    queue_tasks_deferred.erase(
        std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), rm_func),
        queue_tasks_deferred.end());
}

//
// server_response
//

void server_response::add_waiting_task_id(int id_task) {
    RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void server_response::add_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.insert(id_task);
    }
}

void server_response::remove_waiting_task_id(int id_task) {
    RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
    // make sure to clean up all pending results
    queue_results.erase(
        std::remove_if(queue_results.begin(), queue_results.end(), [id_task](const server_task_result_ptr & res) {
            return llama_server_id_matches_rust(res->id, id_task);
        }),
        queue_results.end());
}

void server_response::remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }
}

server_task_result_ptr server_response::recv(const std::unordered_set<int> & id_tasks) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&]{
            if (!running) {
                RES_DBG("%s : queue result stop\n", "recv");
                std::terminate(); // we cannot return here since the caller is HTTP code
            }
            return !queue_results.empty();
        });

        for (size_t i = 0; i < queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (int i = 0; i < (int) queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_results.wait_for(lock, std::chrono::seconds(timeout));
        if (!running) {
            RES_DBG("%s : queue result stop\n", __func__);
            std::terminate(); // we cannot return here since the caller is HTTP code
        }
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv(int id_task) {
    std::unordered_set<int> id_tasks = {id_task};
    return recv(id_tasks);
}

void server_response::send(server_task_result_ptr && result) {
    RES_DBG("sending result for task id = %d\n", result->id);

    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto & id_task : waiting_task_ids) {
        if (result->id == id_task) {
            RES_DBG("task id = %d pushed to result queue\n", result->id);

            queue_results.emplace_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}

void server_response::terminate() {
    running = false;
    condition_results.notify_all();
}

//
// server_response_reader
//

void server_response_reader::post_task(server_task && task, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_task() can only be called once per reader");
    GGML_ASSERT(!task.is_parent() && "not supported, use post_tasks() instead");
    task.index = 0;
    id_tasks.insert(task.id);
    states.push_back(task.create_state());
    queue_results.add_waiting_task_id(task.id);
    queue_tasks.post(std::move(task), front);
}

void server_response_reader::post_tasks(std::vector<server_task> && tasks, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_tasks() can only be called once per reader");
    id_tasks = server_task::get_list_id(tasks);
    states.reserve(tasks.size());
    size_t index = 0;
    for (auto & task : tasks) {
        task.index = index++;
        states.push_back(task.create_state());
        // for child tasks
        for (auto & child_task : task.child_tasks) {
            child_task.index = index++;
            states.push_back(child_task.create_state());
        }
    }
    GGML_ASSERT(states.size() == id_tasks.size());
    queue_results.add_waiting_task_ids(id_tasks);
    queue_tasks.post(std::move(tasks), front);
}

bool server_response_reader::has_next() const {
    return llama_server_response_reader_has_next_rust(cancelled, received_count, id_tasks.size());
}

// return nullptr if should_stop() is true before receiving a result
// note: if one error is received, it will stop further processing and return error result
server_task_result_ptr server_response_reader::next(const std::function<bool()> & should_stop) {
    while (true) {
        server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, polling_interval_seconds);
        if (result == nullptr) {
            // timeout, check stop condition
            if (should_stop()) {
                SRV_DBG("%s", "stopping wait for next result due to should_stop condition\n");
                return nullptr;
            }
        } else {
            if (result->is_error()) {
                stop(); // cancel remaining tasks
                SRV_DBG("%s", "received error result, stopping further processing\n");
                return result;
            }
            if (!states.empty()) {
                // update the generation state if needed
                const size_t idx = result->index;
                GGML_ASSERT(idx < states.size());
                result->update(states[idx]);
            }
            if (result->is_stop()) {
                received_count++;
            }
            return result;
        }
    }

    // should not reach here
}

server_response_reader::batch_response server_response_reader::wait_for_all(const std::function<bool()> & should_stop) {
    batch_response batch_res;
    batch_res.results.clear();
    batch_res.results.resize(id_tasks.size());
    while (has_next()) {
        auto res = next(should_stop);
        if (res == nullptr) {
            batch_res.is_terminated = true;
            return batch_res;
        }
        if (res->is_error()) {
            batch_res.error = std::move(res);
            return batch_res;
        }
        const size_t idx = res->index;
        GGML_ASSERT(idx < batch_res.results.size() && "index out of range");
        GGML_ASSERT(batch_res.results[idx] == nullptr && "duplicate result received");
        batch_res.results[idx] = std::move(res);
    }
    return batch_res;
}

void server_response_reader::stop() {
    queue_results.remove_waiting_task_ids(id_tasks);
    if (has_next() && !cancelled) {
        // if tasks is not finished yet, cancel them
        cancelled = true;
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto & id_task : id_tasks) {
            SRV_WRN("cancel task, id_task = %d\n", id_task);
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id_task;
            queue_results.remove_waiting_task_id(id_task);
            cancel_tasks.push_back(std::move(task));
        }
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(std::move(cancel_tasks), true);
    } else {
        SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
    }
}


// ---- server context implementation ----



#include "common.h.inc"
#include "llama.h.inc"
#include "mtmd.h.inc"

#include <algorithm>
#include <cstddef>

json server_chat_convert_responses_to_chatcmpl(const json & body);
json server_chat_convert_anthropic_to_oai(const json & body);
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <utility>

extern "C" const char * llama_build_info(void);

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

static server_prompt_checkpoint server_get_checkpoint(llama_context * ctx, int id, int64_t n_tokens, llama_pos pos_min = -1, llama_pos pos_max = -1) {
    if (pos_min == -1) {
        pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), id);
    }
    if (pos_max == -1) {
        pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), id);
    }

    const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    auto cur = server_prompt_checkpoint {
        /*.pos_min  = */ pos_min,
        /*.pos_max  = */ pos_max,
        /*.n_tokens = */ n_tokens,
        /*.data     = */ std::vector<uint8_t>(checkpoint_size),
    };

    const size_t n = llama_state_seq_get_data_ext(ctx, cur.data.data(), checkpoint_size, id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (n != checkpoint_size) {
        GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
    }

    return cur;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

struct server_slot {
    int id;

    llama_context * ctx = nullptr;

    common_context_seq_rm_type ctx_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    // multimodal
    mtmd_context * mctx = nullptr;

    // speculative decoding
    llama_tokens spec_draft;
    std::vector<int32_t> spec_i_batch;
    server_prompt_checkpoint spec_ckpt;
    common_speculative_ptr spec;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    void prompt_save(server_prompt_cache & prompt_cache) const {
        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size = llama_state_seq_get_size_ext(ctx, id, 0);

        SRV_WRN(" - saving prompt with length %d, total state size = %.3f MiB\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size);
        if (cur == nullptr) {
            return;
        }

        llama_state_seq_get_data_ext(ctx, cur->data.data(), cur_size, id, 0);
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_INF(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        llama_memory_seq_rm(llama_get_memory(ctx), id, -1, -1);
        prompt.tokens.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0; // number of sent text character

    int64_t t_start_process_prompt;
    int64_t t_start_generation;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_INF(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    bool can_split() const {
        GGML_ASSERT(task);

        return llama_server_slot_can_split_rust(
                task->need_embd(),
                llama_get_memory(ctx) != nullptr,
                llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return llama_server_slot_can_batch_with_rust(
                task->type,
                other_slot.task->type,
                are_lora_equal(lora, other_slot.lora));
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        return llama_server_slot_has_budget_rust(
                task->params.n_predict,
                global_params.n_predict,
                n_decoded,
                &n_remaining);
    }

    bool is_processing() const {
        return llama_server_slot_is_processing_rust(state);
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        const int n_draft_min = common_speculative_n_min(spec.get(), task->params.speculative);
        const int n_draft_max = llama_server_slot_n_draft_max_rust(
                true,
                n_draft_min,
                common_speculative_n_max(spec.get(), task->params.speculative),
                n_ctx,
                prompt.n_tokens(),
                n_remaining);

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    void update_batch(llama_batch & batch) {
        const int n_draft_max = get_n_draft_max();
        if (n_draft_max > 0) {
            GGML_ASSERT(can_speculate());

            // generate draft tokens in speculative decoding mode
            // TODO: rework to have a single draft llama_context shared across all slots [TAG_SERVER_SPEC_REWORK]
            //       perform the speculative drafting for all sequences at the same time in a single batch
            const llama_tokens & tokens = prompt.tokens.get_text_tokens();

            const auto & params_spec = task->params.speculative;

            if (!spec_draft.empty()) {
                // we have a previous (partial) draft to reuse
                if (ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                    GGML_ASSERT(!spec_ckpt.empty());
                }
            } else {
                GGML_ASSERT(spec_i_batch.empty());

                // generate a new draft
                spec_draft = common_speculative_draft(spec.get(), params_spec, tokens, sampled);
                n_draft_total += spec_draft.size();

                if (spec_draft.size() > (size_t) n_draft_max) {
                    SLT_WRN(*this, "draft size %d exceeds max %d, truncating\n", (int) spec_draft.size(), n_draft_max);
                    spec_draft.resize(n_draft_max);
                }

                if (!spec_draft.empty() && ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
                    const auto n_tokens = prompt.tokens.size();

                    spec_ckpt = server_get_checkpoint(ctx, this->id, n_tokens);

                    SLT_DBG(*this, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %zu, size = %.3f MiB)\n",
                            spec_ckpt.pos_min, spec_ckpt.pos_max, n_tokens, (float) spec_ckpt.data.size() / 1024 / 1024);
                }
            }

            GGML_ASSERT(spec_draft.size() <= (size_t) n_draft_max);
        }

        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.n_tokens;

            common_batch_add(batch, sampled, prompt.tokens.pos_next(), { this->id }, true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.n_tokens);
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.n_tokens + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            common_batch_add(batch, sampled, pos0++, { this->id }, true);
            for (auto token : spec_draft) {
                common_batch_add(batch, token, pos0++, { this->id }, true);
            }
        }

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        const server_rust_result_timings_view rust = llama_server_slot_timings_rust(
                n_prompt_tokens_cache,
                n_prompt_tokens_processed,
                t_prompt_processing,
                n_decoded,
                t_token_generation,
                n_draft_total,
                n_draft_accepted);

        result_timings timings;
        timings.cache_n                = rust.cache_n;
        timings.prompt_n               = rust.prompt_n;
        timings.prompt_ms              = rust.prompt_ms;
        timings.prompt_per_token_ms    = rust.prompt_per_token_ms;
        timings.prompt_per_second      = rust.prompt_per_second;
        timings.predicted_n            = rust.predicted_n;
        timings.predicted_ms           = rust.predicted_ms;
        timings.predicted_per_token_ms = rust.predicted_per_token_ms;
        timings.predicted_per_second   = rust.predicted_per_second;
        timings.draft_n                = rust.draft_n;
        timings.draft_n_accepted       = rust.draft_n_accepted;

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "\n"
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n"
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second,
                t_token_generation, n_decoded, t_gen, n_gen_second,
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        if (n_draft_total > 0) {
            const float draft_ratio = (float) n_draft_accepted / n_draft_total;
            SLT_CNT(*this,
                    "draft acceptance rate = %0.5f (%5d accepted / %5d generated)\n",
                    draft_ratio, n_draft_accepted, n_draft_total
            );
        }

        common_speculative_print_stats(spec.get());
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        llama_memory_seq_rm(llama_get_memory(ctx), other.id,     -1, -1);
        llama_memory_seq_cp(llama_get_memory(ctx), id, other.id, -1, -1);

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model = nullptr;
    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx = nullptr;

    llama_batch batch {};

    llama_model_ptr model_dft;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_webui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    void destroy() {
        llama_init.reset();

        ctx = nullptr;
        model = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        for (server_slot & slot : slots) {
            if (slot.can_speculate()) {
                slot.spec.reset();
            }
        }

        llama_batch_free(batch);
    }

    void slot_save_and_clear(server_slot & slot) {
        if (slot.prompt.n_tokens() == 0) {
            return;
        }
        SLT_INF(slot, "%s", "saving idle slot to prompt cache\n");
        SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
        slot.prompt_save(*prompt_cache);
        slot.prompt_clear(false);
        prompt_cache->update();
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        bool is_resume = sleeping;

        SRV_INF("loading model '%s'\n", params.model.path.c_str());

        params_base = params;

        llama_init = common_init_from_params(params_base);

        model = llama_init->model();
        ctx   = llama_init->context();

        if (model == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model);

        n_ctx = llama_n_ctx(ctx);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (params_base.speculative.has_dft()) {
            // TODO speculative: move to the common speculative helpers?
            const auto & params_spec = params_base.speculative.draft;

            SRV_INF("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            auto params_dft = params_base;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx) : params_spec.n_ctx;
            params_dft.n_batch      = llama_n_ctx_seq(ctx);
            params_dft.devices      = params_spec.devices;
            params_dft.model        = params_spec.mparams;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            params_base.speculative.draft.model = model_dft.get();
            params_base.speculative.draft.cparams = common_context_params_to_llama(params_dft);
        }

        std::string & mmproj_path = params_base.mmproj.path;
        if (!mmproj_path.empty()) {
            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mtmd_context_params mparams = mtmd_context_params_default();

            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.media_marker     = get_media_marker();

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        // setup slots
        SRV_INF("initializing slots, n_slots = %d\n", params_base.n_parallel);

        const int n_ctx_train = llama_model_n_ctx_train(model);

        int n_ctx_slot = llama_n_ctx_seq(ctx);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        const auto ctx_seq_rm_type = common_context_can_seq_rm(ctx);
        if (ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_WRN("%s", "speculative decoding will use checkpoints\n");
        }

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id    = i;
            slot.ctx   = ctx;
            slot.n_ctx = n_ctx_slot;

            slot.ctx_seq_rm_type = ctx_seq_rm_type;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            // try speculative decoding
            if (ctx_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
                slot.spec.reset(common_speculative_init(params_base.speculative, slot.ctx));

                if (slot.spec) {
                    SLT_INF(slot, "%s", "speculative decoding context initialized\n");
                }
            }

            SLT_INF(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx);
            batch = llama_batch_init(std::max(n_batch, params_base.n_parallel), 0, 1);
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_WRN("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_WRN("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_WRN("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_WRN("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_WRN("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.name.empty()) {
            model_name = params_base.model.name;
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx != nullptr);
        GGML_ASSERT(model != nullptr);
        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (!params_base.kv_unified) {
                SRV_WRN("%s: --cache-idle-slots requires --kv-unified, disabling\n", __func__);
                params_base.cache_idle_slots = false;
            } else if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s: --cache-idle-slots requires --cache-ram, disabling\n", __func__);
                params_base.cache_idle_slots = false;
            } else {
                SRV_INF("%s: idle slots will be saved to prompt cache and cleared upon starting a new task\n", __func__);
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        // populate webui settings
        {
            if (!params_base.webui_config_json.empty()) {
                try {
                    json_webui_settings = json::parse(params_base.webui_config_json);
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse webui config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model, params_base.chat_template);

                LOG_INF("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_INF("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // find the slot that has at least n% prompt similarity
        if (ret == nullptr && slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        sim_best, slot_prompt_similarity, f_keep);

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);

                update_cache = true;
            }
        }

        if (ret) {
            const auto & tokens = ret->prompt.tokens;

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && llama_server_task_is_completion_rust(task.type);

            if (update_cache) {
                SRV_WRN("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                // don't save the slot's state if its context is empty
                if (tokens.size() > 0) {
                    ret->prompt_save(*prompt_cache);
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_WRN("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_INF(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_INF(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_logits = task.params.sampling.n_probs > 0;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate() && common_speculative_n_max(slot.spec.get(), task.params.speculative) > 0);

            // TODO: getting post/pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx, slot.id, nullptr);
            }

            SLT_INF(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
        } else {
            slot.smpl.reset();
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            // TODO: optimize this with min-p optimization
            std::vector<llama_token_data> cur = get_token_probabilities(ctx, idx);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(ctx, i);
            } else {
                embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_INF("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.data.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        const auto & cur = slot.prompt.checkpoints.emplace_back(server_get_checkpoint(ctx, slot.id, slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max));

        SLT_WRN(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.data.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_slot = task.id_slot;
                    const int id_task = task.id;

                    server_slot * slot = id_slot != -1 ? get_slot_by_id(id_slot) : get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & s : slots) {
                            if (!s.is_processing()) {
                                slot_save_and_clear(s);
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_INF("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void update_slots() {
        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_INF("%s", "all slots are idle\n");

                return;
            }
        }

        {
            SRV_DBG("%s", "posting NEXT_RESPONSE\n");

            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = queue_tasks.get_new_id();
            queue_tasks.post(std::move(task));
        }

        // apply context-shift if needed
        // TODO: simplify and improve
        for (server_slot & slot : slots) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    continue;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                const int n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                llama_memory_seq_rm (llama_get_memory(ctx), slot.id, n_keep            , n_keep + n_discard);
                llama_memory_seq_add(llama_get_memory(ctx), slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        }

        // start populating the batch for this iteration
        common_batch_clear(batch);

        // track if given slot can be batched with slots already in the batch
        server_slot * slot_batched = nullptr;

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        // first, add sampled tokens from any ongoing sequences
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                continue;
            }

            slot.update_batch(batch);
        }

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx);
        int32_t n_ubatch = llama_n_ubatch(ctx);

        float  alora_scale       = -1.0f;
        size_t alora_disabled_id = 0;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.n_tokens == 0) {
            for (auto & slot : slots) {
                if (!slot.is_processing()) {
                    continue;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    continue;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    continue;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.n_tokens;

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_INF(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            continue;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                continue;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                continue;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_INF(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            llama_memory_seq_rm (llama_get_memory(ctx), slot.id, head_p, head_c);
                                            llama_memory_seq_add(llama_get_memory(ctx), slot.id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa);

                            if (n_past > 0 && n_past < slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    SLT_WRN(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d, n_swa = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min, n_swa);

                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&, func_name = __func__](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            LOG_INF("slot %12.*s: id %2d | task %d | Checking checkpoint with [%d, %d] against %d...\n", 12,
                                                func_name, (slot).id, ((slot).task ? (slot).task->id : -1), cur.pos_min, cur.pos_max, pos_min_thold);
                                            return llama_server_checkpoint_before_threshold_rust(cur.pos_min, pos_min_thold);
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        const size_t checkpoint_size = it->data.size();
                                        const size_t n = llama_state_seq_set_data_ext(ctx, it->data.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        if (n != checkpoint_size) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, (float) checkpoint_size / 1024 / 1024);
                                            do_reset = true;
                                            //printf("[DEBUG] `do_reset` was set to `true` after failing to restore a checkpoint");
                                        } else {
                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                            SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) checkpoint_size / 1024 / 1024);
                                        }
                                    }

                                    if (do_reset) {
                                        SLT_WRN(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_WRN(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.data.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // send initial 0% progress update if needed
                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream && slot.task->params.return_progress) {
                            send_partial_response(slot, {}, true);
                        }
                    }

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.n_tokens + slot.task->n_tokens() > n_batch) {
                            continue;
                        }
                    }

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_INF(slot, "n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    if (!llama_memory_seq_rm(llama_get_memory(ctx), slot.id, p0, -1)) {
                        SLT_WRN(slot, "failed to truncate tokens with position >= %d - clearing the memory\n", p0);

                        slot.prompt_clear(true);

                        // there is no common part left
                        slot.n_prompt_tokens_cache = 0;
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && llama_server_task_is_completion_rust(slot.task->type);

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    do_checkpoint = do_checkpoint && (
                            (slot.ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) ||
                            (n_swa > 0));

                    bool has_mtmd = false;

                    // check if we should process the image
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL) {
                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = input_tokens.process_chunk(ctx, mctx, slot.prompt.n_tokens(), slot.prompt.tokens.pos_next(), slot.id, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(slot.prompt.n_tokens());
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.n_tokens < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output
                        common_batch_add(batch,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            { slot.id },
                            slot.task->need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.n_tokens - n_tokens_prev;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.n_tokens > 0);

                        // extract the logits only for the last token
                        batch.logits[batch.n_tokens - 1] = true;

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.n_tokens - 1;

                        slot.init_sampler();
                        SLT_INF(slot, "prompt processing done, n_tokens = %d, batch.n_tokens = %d\n", slot.prompt.n_tokens(), batch.n_tokens);
                    } else {
                        if (slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch) {
                            // near the end of the prompt
                            do_checkpoint = do_checkpoint && true;
                        } else {
                            // only do non-end checkpoints if the "checkpoint every n tokens" option is set
                            do_checkpoint = do_checkpoint && params_base.checkpoint_every_nt > 0;

                            if (do_checkpoint) {
                                llama_pos last_checkpoint = 0;
                                if (!slot.prompt.checkpoints.empty()) {
                                    last_checkpoint = slot.prompt.checkpoints.back().n_tokens;
                                }

                                do_checkpoint = do_checkpoint && slot.prompt.n_tokens() - batch.n_tokens - last_checkpoint >= params_base.checkpoint_every_nt;

                                if (do_checkpoint) {
                                    SLT_INF(slot, "%d tokens since last checkpoint at %d, creating new checkpoint during processing at position %d\n", params_base.checkpoint_every_nt, last_checkpoint, slot.prompt.n_tokens());
                                }
                            }
                        }

                        SLT_INF(slot, "prompt processing progress, n_tokens = %d, batch.n_tokens = %d, progress = %f\n", slot.prompt.n_tokens(), batch.n_tokens, (float) slot.prompt.n_tokens() / slot.task->n_tokens());
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx), slot.id);

                    // no need for empty or small checkpoints
                    do_checkpoint = do_checkpoint && (pos_min >= 0 && slot.prompt.n_tokens() >= 64);

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || slot.prompt.n_tokens() - n_tokens_cur > slot.prompt.checkpoints.back().n_tokens + 64);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }

                if (batch.n_tokens >= n_batch) {
                    break;
                }
            }
        }

        SRV_DBG("decoding batch, n_tokens = %d\n", batch.n_tokens);

        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx, slot_batched->task->need_embd());
        }

        if (batch.n_tokens == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }
        } else {
            n_empty_consecutive = 0;
        }

        int32_t i_next = 0;

        // process the created batch of tokens
        for (int32_t i = 0; i < batch.n_tokens; i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.n_tokens - i);

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);

            metrics.on_decoded(slots);

            if (ret != 0) {
                {
                    std::string err;

                    if (n_batch == 1 && ret == 1) {
                        // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                        //       need to remove the tokens from the current batch too
                        err = "Context size has been exceeded.";
                    }

                    if (ret == -1) {
                        err = "Invalid input batch.";
                    }

                    if (ret < -1) {
                        // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                        err = "Compute error.";
                    }

                    // TODO: handle ret == 2 (abort) when we start aborting

                    if (!err.empty()) {
                        SRV_ERR("%s i = %d, n_batch = %d, ret = %d\n", err.c_str(), i, n_batch, ret);

                        for (auto & slot : slots) {
                            if (slot.is_processing()) {
                                send_error(slot, err);
                                slot.release();

                                // note: it's complicated to keep track of how much of the current batch has been
                                //       processed before the error occurred, so we simply clear the entire context
                                slot.prompt_clear(false);
                            }
                        }

                        break;
                    }
                }

                // retry with half the batch size to try to find a free slot in the KV cache
                if (!try_clear_idle_slots()) {
                    n_batch /= 2;
                }

                SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, i = %d, n_batch = %d, ret = %d\n", i, n_batch, ret);

                continue; // continue loop of n_batch
            }

            // move the head of the batch forward with the number of tokens we just processed
            i_next = i + n_tokens;

            // on successful decode, restore the original batch size
            n_batch = llama_n_batch(ctx);

            // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
            for (auto & slot : slots) {
                if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                    std::vector<server_slot *> children;
                    for (auto & other : slots) {
                        if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                            children.push_back(&other);
                        }
                    }

                    // all children slots should already launched by launch_slots_with_parent_task()
                    // copy state to the child slots
                    for (auto & child : children) {
                        SLT_INF(slot, " - copying state to child %d\n", child->id);

                        GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                        slot.copy_state_to(*child);
                        child->state = SLOT_STATE_DONE_PROMPT;
                    }
                }
            }

            for (auto & slot : slots) {
                // optionally send prompt processing progress
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (slot.task->params.stream && slot.task->params.return_progress) {
                        send_partial_response(slot, {}, true);
                    }
                }

                if (slot.i_batch < (int) i || slot.i_batch >= (int) (i + n_tokens)) {
                    continue; // continue loop of slots
                }

                if (slot.state == SLOT_STATE_DONE_PROMPT) {
                    if (llama_server_task_is_embedding_rust(slot.task->type)) {
                        // prompt evaluated for embedding
                        send_embedding(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    if (llama_server_task_is_rerank_rust(slot.task->type)) {
                        send_rerank(slot, batch_view);
                        slot.release();
                        slot.i_batch = -1;
                        continue; // continue loop of slots
                    }

                    GGML_ASSERT(slot.task->need_sampling());

                    // prompt evaluated for next-token prediction
                    slot.state = SLOT_STATE_GENERATING;

                    if (slot.can_speculate()) {
                        common_speculative_begin(slot.spec.get(), slot.prompt.tokens.get_text_tokens());
                    }
                } else if (slot.state != SLOT_STATE_GENERATING) {
                    continue; // continue loop of slots
                }

                if (slot.can_speculate() && !slot.spec_draft.empty()) {
                    continue; // sample using speculative decoding
                }

                const int tok_idx = slot.i_batch - i;

                llama_token id = common_sampler_sample(slot.smpl.get(), slot.ctx, tok_idx);

                slot.i_batch = -1;

                common_sampler_accept(slot.smpl.get(), id, true);

                // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
                const int64_t t_current = ggml_time_us();

                slot.n_decoded += 1;

                if (slot.n_decoded == 1) {
                    slot.t_start_generation = t_current;
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);
                }

                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                completion_token_output result;
                result.tok          = id;
                result.text_to_send = common_token_to_piece(slot.ctx, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

                if (slot.task->params.sampling.n_probs > 0) {
                    populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
                }

                if (!process_token(result, slot)) {
                    // release slot because of stop condition
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    continue;
                }
            }

            // speculative decoding - main model sample and accept
            for (auto & slot : slots) {
                if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                    continue;
                }

                // save the original draft size
                const size_t n_draft = slot.spec_draft.size();

                GGML_ASSERT(n_draft > 0);

                // verify and try to accept the draft
                {
                    const bool use_ckpt = slot.ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                    // only save the sampler sampler state if we use checkpoints
                    common_sampler_ptr smpl_save;
                    if (use_ckpt) {
                        smpl_save.reset(common_sampler_clone(slot.smpl.get()));
                    }

                    GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                    auto accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx, slot.spec_i_batch, slot.spec_draft);
                    slot.spec_i_batch.clear();

                    GGML_ASSERT(accepted.size() >= 1);

                    // check for partial draft acceptance
                    if (accepted.size() < slot.spec_draft.size() + 1) {
                        if (use_ckpt) {
                            if (trace > 0) {
                                SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                            }

                            // partial acceptance is not supported by the context -> truncate the draft and restore the state
                            slot.spec_draft = std::move(accepted);

                            const auto & ckpt = slot.spec_ckpt;

                            SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n",
                                    ckpt.pos_min, ckpt.pos_max, ckpt.size());

                            const size_t n = llama_state_seq_set_data_ext(slot.ctx, ckpt.data.data(), ckpt.size(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                            if (n != ckpt.size()) {
                                GGML_ABORT("%s: failed to restore context checkpoint (pos_min=%d, pos_max=%d, size=%zu, get_data_ext->%zu, set_data_ext->%zu",
                                        __func__, ckpt.pos_min, ckpt.pos_max, ckpt.size(), ckpt.size(), n);
                            }

                            llama_memory_seq_rm(llama_get_memory(slot.ctx), slot.id, ckpt.pos_max + 1, -1);

                            slot.prompt.tokens.keep_first(ckpt.n_tokens);
                            slot.smpl = std::move(smpl_save);

                            continue;
                        }
                    }

                    if (trace > 0) {
                        SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                    }

                    common_speculative_accept(slot.spec.get(), accepted.size() - 1);

                    slot.spec_draft = std::move(accepted);
                }

                const int64_t t_current = ggml_time_us();

                const auto ids = std::move(slot.spec_draft);

                slot.n_decoded += ids.size();
                slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

                // update how many tokens out of those tested were accepted
                slot.n_draft_accepted += ids.size() - 1;

                // add accepted tokens to the prompt
                slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
                slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

                slot.sampled = ids.back(); // last accepted token
                SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

                llama_memory_seq_rm(llama_get_memory(slot.ctx), slot.id, slot.prompt.tokens.pos_next(), -1);

                for (size_t i = 0; i < ids.size(); ++i) {
                    completion_token_output result;

                    result.tok          = ids[i];
                    result.text_to_send = common_token_to_piece(slot.ctx, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f; // set later

                    // TODO: set result.probs

                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();

                        break;
                    }
                }

                SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
            }
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx, eos_id, true) : "";

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* json_webui_settings    */ impl->json_webui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* pooling_type           */ llama_pooling_type(impl->ctx),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model),
        /* model_n_params         */ llama_model_n_params(impl->model),
        /* model_size             */ llama_model_size(impl->model),
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::on_sleeping_changed(std::function<void(bool)> callback) {
    impl->queue_tasks.on_sleeping_state(std::move(callback));
}


//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(llama_server_task_is_completion_or_infill_rust(type));

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        // process prompt
        std::vector<server_tokens> inputs;

        if (llama_server_response_type_is_non_native_rust(res_type) && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_task::params_from_json_cmpl(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);
            task.id_slot = json_value(data, "id_slot", -1);

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (llama_server_response_type_is_oai_chat_or_cmpl_rust(res_type)) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (llama_server_response_type_is_anthropic_rust(res_type)) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (llama_server_response_type_is_oai_resp_rust(res_type)) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, &req](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (llama_server_response_type_is_anthropic_rust(res_type)) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            try {
                if (req.should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    output = llama_server_response_type_stream_done_is_empty_rust(res_type) ? "" : "data: [DONE]\n\n";
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                auto result = rd.next(req.should_stop);
                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(req.should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (llama_server_response_type_is_anthropic_rust(res_type)) {
                        output = format_anthropic_sse(res_json);
                    } else if (llama_server_response_type_is_oai_resp_rust(res_type)) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "webui",                       params.webui },
            { "webui_settings",              meta->json_webui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);
        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                {
                    {"id",       meta->model_name},
                    {"aliases",  meta->model_aliases},
                    {"tags",     meta->model_tags},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     {
                        {"vocab_type",  meta->model_vocab_type},
                        {"n_vocab",     meta->model_vocab_n_tokens},
                        {"n_ctx_train", meta->model_n_ctx_train},
                        {"n_embd",      meta->model_n_embd_inp},
                        {"n_params",    meta->model_n_params},
                        {"size",        meta->model_size},
                    }},
                },
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (llama_server_response_type_is_non_native_rust(res_type) && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = 2; // default to Euclidean/L2 norm
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = llama_server_response_type_is_oai_embd_rust(res_type)
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}


// ---- server tools implementation ----


#include <sheredom/subprocess.h.inc>

#include <memory>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <cstring>
#include <climits>
#include <vector>

struct server_tool {
    std::string name;
    std::string display_name;
    bool permission_write = false;

    virtual ~server_tool() = default;
    virtual json get_definition() = 0;
    virtual json invoke(json params) = 0;

    json to_json();
};

struct server_tools {
    std::vector<std::unique_ptr<server_tool>> tools;

    void setup(const std::vector<std::string> & enabled_tools);
    json invoke(const std::string & name, const json & params);

    server_http_context::handler_t handle_get;
    server_http_context::handler_t handle_post;
};

namespace fs = std::filesystem;

//
// internal helpers
//

static std::vector<char *> to_cstr_vec(const std::vector<std::string> & v) {
    std::vector<char *> r;
    r.reserve(v.size() + 1);
    for (const auto & s : v) {
        r.push_back(const_cast<char *>(s.c_str()));
    }
    r.push_back(nullptr);
    return r;
}

struct run_proc_result {
    std::string output;
    int  exit_code = -1;
    bool timed_out = false;
};

static run_proc_result run_process(
        const std::vector<std::string> & args,
        size_t max_output,
        int timeout_secs) {
    run_proc_result res;

    subprocess_s proc;
    auto argv = to_cstr_vec(args);

    int options = subprocess_option_no_window
                | subprocess_option_combined_stdout_stderr
                | subprocess_option_inherit_environment
                | subprocess_option_search_user_path;

    if (subprocess_create(argv.data(), options, &proc) != 0) {
        res.output = "failed to spawn process";
        return res;
    }

    std::atomic<bool> done{false};
    std::atomic<bool> timed_out{false};

    std::thread timeout_thread([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_secs);
        while (!done.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                timed_out.store(true);
                subprocess_terminate(&proc);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    FILE * f = subprocess_stdout(&proc);
    std::string output;
    bool truncated = false;
    if (f) {
        char buf[4096];
        while (fgets(buf, sizeof(buf), f) != nullptr) {
            if (!truncated) {
                size_t len = strlen(buf);
                if (output.size() + len <= max_output) {
                    output.append(buf, len);
                } else {
                    output.append(buf, max_output - output.size());
                    truncated = true;
                }
            }
        }
    }

    done.store(true);
    if (timeout_thread.joinable()) {
        timeout_thread.join();
    }

    subprocess_join(&proc, &res.exit_code);
    subprocess_destroy(&proc);

    res.output    = output;
    res.timed_out = timed_out.load();
    if (truncated) {
        res.output += "\n[output truncated]";
    }
    return res;
}

json server_tool::to_json() {
    return {
        {"display_name", display_name},
        {"tool", name},
        {"type", "builtin"},
        {"permissions", json{
            {"write", permission_write}
        }},
        {"definition", get_definition()},
    };
}

//
// read_file: read a file with optional line range and line-number prefix
//

static constexpr size_t SERVER_TOOL_READ_FILE_MAX_SIZE = 16 * 1024; // 16 KB

struct server_tool_read_file : server_tool {
    server_tool_read_file() {
        name = "read_file";
        display_name = "Read file";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Read the contents of a file. Optionally specify a 1-based line range. "
                                "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192 ...\")."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",       {{"type", "string"},  {"description", "Path to the file"}}},
                        {"start_line", {{"type", "integer"}, {"description", "First line to read, 1-based (default: 1)"}}},
                        {"end_line",   {{"type", "integer"}, {"description", "Last line to read, 1-based inclusive (default: end of file)"}}},
                        {"append_loc", {{"type", "boolean"}, {"description", "Prefix each line with its line number"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path  = params.at("path").get<std::string>();
        int  start_line   = json_value(params, "start_line", 1);
        int  end_line     = json_value(params, "end_line",  -1); // -1 = no limit
        bool append_loc   = json_value(params, "append_loc", false);

        std::error_code ec;
        uintmax_t file_size = fs::file_size(path, ec);
        if (ec) {
            return {{"error", "cannot stat file: " + ec.message()}};
        }
        if (file_size > SERVER_TOOL_READ_FILE_MAX_SIZE && end_line == -1) {
            return {{"error", string_format(
                "file too large (%zu bytes, max %zu). Use start_line/end_line to read a portion.",
                (size_t)file_size, SERVER_TOOL_READ_FILE_MAX_SIZE)}};
        }

        std::ifstream f(path);
        if (!f) {
            return {{"error", "failed to open file: " + path}};
        }

        std::string result;
        std::string line;
        int lineno = 0;

        while (std::getline(f, line)) {
            lineno++;
            if (lineno < start_line) continue;
            if (end_line != -1 && lineno > end_line) break;

            std::string out_line;
            if (append_loc) {
                out_line = std::to_string(lineno) + "\u2192 " + line + "\n";
            } else {
                out_line = line + "\n";
            }

            if (result.size() + out_line.size() > SERVER_TOOL_READ_FILE_MAX_SIZE) {
                result += "[output truncated]";
                break;
            }
            result += out_line;
        }

        return {{"plain_text_response", result}};
    }
};

//
// file_glob_search: find files matching a glob pattern under a base directory
//

static constexpr size_t SERVER_TOOL_FILE_SEARCH_MAX_RESULTS = 100;

struct server_tool_file_glob_search : server_tool {
    server_tool_file_glob_search() {
        name = "file_glob_search";
        display_name = "File search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Recursively search for files matching a glob pattern under a directory."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Base directory to search in"}}},
                        {"include", {{"type", "string"}, {"description", "Glob pattern for files to include (e.g. \"**/*.cpp\"). Default: **"}}},
                        {"exclude", {{"type", "string"}, {"description", "Glob pattern for files to exclude"}}},
                    }},
                    {"required", json::array({"path"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string base    = params.at("path").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));

        std::ostringstream output_text;
        size_t count = 0;

        std::error_code ec;
        for (const auto & entry : fs::recursive_directory_iterator(base,
                fs::directory_options::skip_permission_denied, ec)) {
            if (!entry.is_regular_file()) continue;

            std::string rel = fs::relative(entry.path(), base, ec).string();
            if (ec) continue;
            std::replace(rel.begin(), rel.end(), '\\', '/');

            if (!glob_match(include, rel)) continue;
            if (!exclude.empty() && glob_match(exclude, rel)) continue;

            output_text << entry.path().string() << "\n";
            if (++count >= SERVER_TOOL_FILE_SEARCH_MAX_RESULTS) {
                break;
            }
        }

        output_text << "\n---\nTotal matches: " << count << "\n";

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// grep_search: search for a regex pattern in files
//

static constexpr size_t SERVER_TOOL_GREP_SEARCH_MAX_RESULTS = 100;

struct server_tool_grep_search : server_tool {
    server_tool_grep_search() {
        name = "grep_search";
        display_name = "Grep search";
        permission_write = false;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Search for a regex pattern in files under a path. Returns matching lines."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",                {{"type", "string"},  {"description", "File or directory to search in"}}},
                        {"pattern",             {{"type", "string"},  {"description", "Regular expression pattern to search for"}}},
                        {"include",             {{"type", "string"},  {"description", "Glob pattern to filter files (default: **)"}}},
                        {"exclude",             {{"type", "string"},  {"description", "Glob pattern to exclude files"}}},
                        {"return_line_numbers", {{"type", "boolean"}, {"description", "If true, include line numbers in results"}}},
                    }},
                    {"required", json::array({"path", "pattern"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string pat_str = params.at("pattern").get<std::string>();
        std::string include = json_value(params, "include", std::string("**"));
        std::string exclude = json_value(params, "exclude", std::string(""));
        bool show_lineno    = json_value(params, "return_line_numbers", false);

        std::regex pattern;
        try {
            pattern = std::regex(pat_str);
        } catch (const std::regex_error & e) {
            return {{"error", std::string("invalid regex: ") + e.what()}};
        }

        std::ostringstream output_text;
        size_t total = 0;

        auto search_file = [&](const fs::path & fpath) {
            std::ifstream f(fpath);
            if (!f) return;
            std::string line;
            int lineno = 0;
            while (std::getline(f, line) && total < SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) {
                lineno++;
                if (std::regex_search(line, pattern)) {
                    output_text << fpath.string() << ":";
                    if (show_lineno) {
                        output_text << lineno << ":";
                    }
                    output_text << line << "\n";
                    total++;
                }
            }
        };

        std::error_code ec;
        if (fs::is_regular_file(path, ec)) {
            search_file(path);
        } else if (fs::is_directory(path, ec)) {
            for (const auto & entry : fs::recursive_directory_iterator(path,
                    fs::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) continue;
                if (total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS) break;

                std::string rel = fs::relative(entry.path(), path, ec).string();
                if (ec) continue;
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(include, rel)) continue;
                if (!exclude.empty() && glob_match(exclude, rel)) continue;

                search_file(entry.path());
            }
        } else {
            return {{"error", "path does not exist: " + path}};
        }

        output_text << "\n\n---\nTotal matches: " << total << "\n";

        return {{"plain_text_response", output_text.str()}};
    }
};

//
// exec_shell_command: run an arbitrary shell command
//

static constexpr size_t SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE = 16 * 1024; // 16 KB
static constexpr int    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT     = 60;        // seconds

struct server_tool_exec_shell_command : server_tool {
    server_tool_exec_shell_command() {
        name = "exec_shell_command";
        display_name = "Execute shell command";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Execute a shell command and return its output (stdout and stderr combined)."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"command",         {{"type", "string"},  {"description", "Shell command to execute"}}},
                        {"timeout",         {{"type", "integer"}, {"description", string_format("Timeout in seconds (default 10, max %d)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT)}}},
                        {"max_output_size", {{"type", "integer"}, {"description", string_format("Maximum output size in bytes (default %zu)", SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE)}}},
                    }},
                    {"required", json::array({"command"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string command   = params.at("command").get<std::string>();
        int    timeout        = json_value(params, "timeout",         10);
        size_t max_output     = (size_t) json_value(params, "max_output_size", (int) SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

        timeout    = std::min(timeout,    SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT);
        max_output = std::min(max_output, SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE);

#ifdef _WIN32
        std::vector<std::string> args = {"cmd", "/c", command};
#else
        std::vector<std::string> args = {"sh", "-c", command};
#endif

        auto res = run_process(args, max_output, timeout);

        std::string text_output = res.output;
        text_output += string_format("\n[exit code: %d]", res.exit_code);
        if (res.timed_out) {
            text_output += " [exit due to timed out]";
        }

        return {{"plain_text_response", text_output}};
    }
};

//
// write_file: create or overwrite a file
//

struct server_tool_write_file : server_tool {
    server_tool_write_file() {
        name = "write_file";
        display_name = "Write file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Write content to a file, creating it (including parent directories) if it does not exist. May use with edit_file for more complex edits."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path of the file to write"}}},
                        {"content", {{"type", "string"}, {"description", "Content to write"}}},
                    }},
                    {"required", json::array({"path", "content"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path    = params.at("path").get<std::string>();
        std::string content = params.at("content").get<std::string>();

        std::error_code ec;
        fs::path fpath(path);
        if (fpath.has_parent_path()) {
            fs::create_directories(fpath.parent_path(), ec);
            if (ec) {
                return {{"error", "failed to create directories: " + ec.message()}};
            }
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        f << content;
        if (!f) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file written successfully"}, {"path", path}, {"bytes", content.size()}};
    }
};

//
// edit_file: edit file content via line-based changes
//

struct server_tool_edit_file : server_tool {
    server_tool_edit_file() {
        name = "edit_file";
        display_name = "Edit file";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description",
                    "Edit a file by applying a list of line-based changes. "
                    "Each change targets a 1-based inclusive line range and has a mode: "
                    "\"replace\" (replace lines with content), "
                    "\"delete\" (remove lines, content must be empty string), "
                    "\"append\" (insert content after line_end). "
                    "Set line_start to -1 to target the end of file (line_end is ignored in that case). "
                    "Changes must not overlap. They are applied in reverse line order automatically."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"path",    {{"type", "string"}, {"description", "Path to the file to edit"}}},
                        {"changes", {
                            {"type", "array"},
                            {"description", "List of changes to apply"},
                            {"items", {
                                {"type", "object"},
                                {"properties", {
                                    {"mode",       {{"type", "string"},  {"description", "\"replace\", \"delete\", or \"append\""}}},
                                    {"line_start", {{"type", "integer"}, {"description", "First line of the range (1-based); use -1 for end of file"}}},
                                    {"line_end",   {{"type", "integer"}, {"description", "Last line of the range (1-based, inclusive); ignored when line_start is -1"}}},
                                    {"content",    {{"type", "string"},  {"description", "Content to insert; must be empty string for delete mode"}}},
                                }},
                                {"required", json::array({"mode", "line_start", "line_end", "content"})},
                            }},
                        }},
                    }},
                    {"required", json::array({"path", "changes"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string path = params.at("path").get<std::string>();
        const json & changes = params.at("changes");

        if (!changes.is_array()) {
            return {{"error", "\"changes\" must be an array"}};
        }

        // read file into lines
        std::ifstream fin(path);
        if (!fin) {
            return {{"error", "failed to open file: " + path}};
        }
        std::vector<std::string> lines;
        {
            std::string line;
            while (std::getline(fin, line)) {
                lines.push_back(line);
            }
        }
        fin.close();

        // validate and collect changes, then sort descending by line_start
        struct change_entry {
            std::string mode;
            int line_start; // 1-based
            int line_end;   // 1-based inclusive
            std::string content;
        };
        std::vector<change_entry> entries;
        entries.reserve(changes.size());

        for (const auto & ch : changes) {
            change_entry e;
            e.mode       = ch.at("mode").get<std::string>();
            e.line_start = ch.at("line_start").get<int>();
            e.line_end   = ch.at("line_end").get<int>();
            e.content    = ch.at("content").get<std::string>();

            if (e.mode != "replace" && e.mode != "delete" && e.mode != "append") {
                return {{"error", "invalid mode \"" + e.mode + "\"; must be replace, delete, or append"}};
            }
            if (e.mode == "delete" && !e.content.empty()) {
                return {{"error", "content must be empty string for delete mode"}};
            }
            int n = (int) lines.size();
            if (e.line_start == -1) {
                // -1 means end of file; line_end is ignored — normalize to point past last line
                e.line_start = n + 1;
                e.line_end   = n + 1;
            } else {
                if (e.line_start < 1 || e.line_end < e.line_start) {
                    return {{"error", string_format("invalid line range [%d, %d]", e.line_start, e.line_end)}};
                }
                if (e.line_end > n) {
                    return {{"error", string_format("line_end %d exceeds file length %d", e.line_end, n)}};
                }
            }
            entries.push_back(std::move(e));
        }

        // sort descending so earlier-indexed changes don't shift later ones
        std::sort(entries.begin(), entries.end(), [](const change_entry & a, const change_entry & b) {
            return llama_server_line_start_desc_rust(a.line_start, b.line_start);
        });

        // apply changes (0-based indices internally)
        for (const auto & e : entries) {
            int idx_start = e.line_start - 1; // 0-based
            int idx_end   = e.line_end   - 1; // 0-based inclusive

            // split content into lines (preserve trailing newline awareness)
            std::vector<std::string> new_lines;
            if (!e.content.empty()) {
                std::istringstream ss(e.content);
                std::string ln;
                while (std::getline(ss, ln)) {
                    new_lines.push_back(ln);
                }
                // if content ends with \n, getline consumed it — no extra empty line needed
                // if content does NOT end with \n, last line is still captured correctly
            }

            if (e.mode == "replace") {
                // erase [idx_start, idx_end] and insert new_lines
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
                lines.insert(lines.begin() + idx_start, new_lines.begin(), new_lines.end());
            } else if (e.mode == "delete") {
                lines.erase(lines.begin() + idx_start, lines.begin() + idx_end + 1);
            } else { // append
                // idx_end + 1 may equal lines.size() when line_start == -1 (end of file)
                lines.insert(lines.begin() + idx_end + 1, new_lines.begin(), new_lines.end());
            }
        }

        // write file back
        std::ofstream fout(path, std::ios::binary);
        if (!fout) {
            return {{"error", "failed to open file for writing: " + path}};
        }
        for (size_t i = 0; i < lines.size(); i++) {
            fout << lines[i];
            if (i + 1 < lines.size()) {
                fout << "\n";
            }
        }
        if (!lines.empty()) {
            fout << "\n";
        }
        if (!fout) {
            return {{"error", "failed to write file: " + path}};
        }

        return {{"result", "file edited successfully"}, {"path", path}, {"lines", (int) lines.size()}};
    }
};

//
// apply_diff: apply a unified diff via git apply
//

struct server_tool_apply_diff : server_tool {
    server_tool_apply_diff() {
        name = "apply_diff";
        display_name = "Apply diff";
        permission_write = true;
    }

    json get_definition() override {
        return {
            {"type", "function"},
            {"function", {
                {"name", name},
                {"description", "Apply a unified diff to edit one or more files using git apply. Use this instead of edit_file when the changes are complex."},
                {"parameters", {
                    {"type", "object"},
                    {"properties", {
                        {"diff", {{"type", "string"}, {"description", "Unified diff content in git diff format"}}},
                    }},
                    {"required", json::array({"diff"})},
                }},
            }},
        };
    }

    json invoke(json params) override {
        std::string diff = params.at("diff").get<std::string>();

        // write diff to a temporary file
        static std::atomic<int> counter{0};
        std::string tmp_path = (fs::temp_directory_path() /
            ("llama_patch_" + std::to_string(++counter) + ".patch")).string();

        {
            std::ofstream f(tmp_path, std::ios::binary);
            if (!f) {
                return {{"error", "failed to create temp patch file"}};
            }
            f << diff;
        }

        auto res = run_process({"git", "apply", tmp_path}, 4096, 10);

        std::error_code ec;
        fs::remove(tmp_path, ec);

        if (res.exit_code != 0) {
            return {{"error", "git apply failed (exit " + std::to_string(res.exit_code) + "): " + res.output}};
        }
        return {{"result", "patch applied successfully"}};
    }
};

//
// public API
//

static std::vector<std::unique_ptr<server_tool>> build_tools() {
    std::vector<std::unique_ptr<server_tool>> tools;
    tools.push_back(std::make_unique<server_tool_read_file>());
    tools.push_back(std::make_unique<server_tool_file_glob_search>());
    tools.push_back(std::make_unique<server_tool_grep_search>());
    tools.push_back(std::make_unique<server_tool_exec_shell_command>());
    tools.push_back(std::make_unique<server_tool_write_file>());
    tools.push_back(std::make_unique<server_tool_edit_file>());
    tools.push_back(std::make_unique<server_tool_apply_diff>());
    return tools;
}

void server_tools::setup(const std::vector<std::string> & enabled_tools) {
    if (!enabled_tools.empty()) {
        std::unordered_set<std::string> enabled_set(enabled_tools.begin(), enabled_tools.end());
        auto all_tools = build_tools();

        tools.clear();
        for (auto & t : all_tools) {
            if (enabled_set.count(t->name) > 0 || enabled_set.count("all") > 0) {
                tools.push_back(std::move(t));
            }
        }
    }

    handle_get = [this](const server_http_req &) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json result = json::array();
            for (const auto & t : tools) {
                result.push_back(t->to_json());
            }
            res->data = safe_json_to_str(result);
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };

    handle_post = [this](const server_http_req & req) -> server_http_res_ptr {
        auto res = std::make_unique<server_http_res>();
        try {
            json body = json::parse(req.body);
            std::string tool_name = body.at("tool").get<std::string>();
            json params = body.value("params", json::object());
            json result = invoke(tool_name, params);
            res->data   = safe_json_to_str(result);
        } catch (const json::exception & e) {
            res->status = 400;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        } catch (const std::exception & e) {
            SRV_ERR("got exception: %s\n", e.what());
            res->status = 500;
            res->data   = safe_json_to_str(format_error_response(e.what(), ERROR_TYPE_SERVER));
        }
        return res;
    };
}

json server_tools::invoke(const std::string & name, const json & params) {
    for (auto & t : tools) {
        if (t->name == name) {
            return t->invoke(params);
        }
    }
    return {{"error", "unknown tool: " + name}};
}
