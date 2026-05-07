#include "ggml.h.inc"
#include "gguf.h.inc"

#include "common.h.inc"
#include "llama.h.inc"
#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

struct common_rust_string_view {
    const uint8_t * data;
    size_t len;
};

struct common_rust_string_list {
    llama_common_unicode_string * data;
    size_t len;
};

struct common_rust_gguf_split_info {
    llama_common_unicode_string prefix;
    llama_common_unicode_string tag;
    int index;
    int count;
};

extern "C" {
int llama_build_number(void);
const char * llama_commit(void);
const char * llama_compiler(void);
const char * llama_build_target(void);
llama_common_unicode_string llama_common_string_repeat_rust(const uint8_t * data, size_t len, size_t n);
llama_common_unicode_string llama_common_int_vec_to_string_rust(const int32_t * data, size_t len);
llama_common_unicode_string llama_common_string_trim_rust(const uint8_t * data, size_t len, int mode);
size_t llama_common_prefix_len_rust(const uint8_t * left, size_t left_len, const uint8_t * right, size_t right_len);
size_t llama_common_suffix_len_rust(const uint8_t * left, size_t left_len, const uint8_t * right, size_t right_len);
llama_common_unicode_string llama_common_string_replace_all_rust(
        const uint8_t * data, size_t len,
        const uint8_t * search, size_t search_len,
        const uint8_t * replace, size_t replace_len);
llama_common_unicode_string llama_common_regex_escape_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_string_join_rust(
        const common_rust_string_view * values, size_t values_len,
        const uint8_t * separator, size_t separator_len);
common_rust_string_list llama_common_string_split_rust(
        const uint8_t * data, size_t len,
        const uint8_t * delimiter, size_t delimiter_len);
common_rust_string_list llama_common_parse_csv_row_rust(const uint8_t * data, size_t len);
common_rust_string_view llama_common_trim_trailing_space_view_rust(const uint8_t * data, size_t len, int max);
common_rust_string_view llama_common_trim_leading_space_view_rust(const uint8_t * data, size_t len, int max);
llama_common_unicode_string llama_common_string_process_escapes_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_clean_file_name_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_string_diff_rust(
        const uint8_t * last, size_t last_len,
        const uint8_t * current, size_t current_len,
        int * status);
bool llama_common_string_starts_with_rust(
        const uint8_t * data, size_t len,
        const uint8_t * prefix, size_t prefix_len);
bool llama_common_string_ends_with_rust(
        const uint8_t * data, size_t len,
        const uint8_t * suffix, size_t suffix_len);
bool llama_common_glob_match_rust(
        const uint8_t * pattern, size_t pattern_len,
        const uint8_t * str, size_t str_len);
bool llama_common_fs_validate_filename_rust(const uint8_t * filename, size_t filename_len, bool allow_subdirs);
llama_common_unicode_string llama_common_string_lstrip_chars_rust(
        const uint8_t * data, size_t len,
        const char * chars);
llama_common_unicode_string llama_common_string_rstrip_chars_rust(
        const uint8_t * data, size_t len,
        const char * chars);
common_rust_string_list llama_common_break_str_into_lines_rust(
        const uint8_t * data, size_t len,
        size_t max_char_per_line);
llama_common_unicode_string llama_common_rm_leading_dashes_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_until_common_prefix_rust(
        const uint8_t * full, size_t full_len,
        const uint8_t * left, size_t left_len,
        const uint8_t * right, size_t right_len);
llama_common_unicode_string llama_common_after_common_suffix_rust(
        const uint8_t * full, size_t full_len,
        const uint8_t * left, size_t left_len,
        const uint8_t * right, size_t right_len);
llama_common_unicode_string llama_common_json_ensure_ascii_preserving_format_rust(
        const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_normalize_quotes_to_json_rust(const uint8_t * data, size_t len);
void llama_common_string_list_free(common_rust_string_list value);
int llama_common_opt_get_optimizer_rust(const char * name);
int llama_common_arg_is_truthy_rust(const char * value);
int llama_common_arg_is_falsey_rust(const char * value);
int llama_common_arg_is_autoy_rust(const char * value);
int llama_common_parse_bool_value_rust(const char * value);
llama_common_unicode_string llama_common_parse_bool_arg_rust(
        const common_rust_string_view * neg_args, size_t neg_args_len,
        const uint8_t * key, size_t key_len,
        const uint8_t * value, size_t value_len);
llama_common_unicode_string llama_common_read_file_rust(const uint8_t * path, size_t path_len);
int llama_common_parse_cpu_mask_rust(const uint8_t * mask, size_t mask_len, bool * boolmask, size_t boolmask_len);
int llama_common_chat_tool_choice_parse_oaicompat_rust(const char * value);
int llama_common_kv_cache_type_from_str_rust(const char * value);
llama_common_unicode_string llama_common_get_line_col_rust(const uint8_t * data, size_t len, size_t pos);
void llama_common_embd_normalize_rust(const float * inp, float * out, int n, int embd_norm);
float llama_common_embd_similarity_cos_rust(const float * embd1, const float * embd2, int n);
const char * llama_common_get_all_kv_cache_types_rust();
const char * llama_common_bool_to_string_rust(bool value);
int llama_common_default_thread_count_windows_rust(int num_physical_cores, int default_threads);
int llama_common_default_thread_count_rust(unsigned int n_threads);
void llama_common_lr_opt_init_rust(
        float lr0, float lr_min, float decay_epochs, unsigned epochs,
        float * out_decay_epochs, float * out_scale_epoch);
float llama_common_lr_opt_get_lr_rust(float lr0, float lr_min, float decay_epochs, float scale_epoch, float epoch);
int llama_common_log_level_to_verbosity_rust(int level);
int64_t llama_common_time_us_rust();
bool llama_common_sampler_prob_desc_rust(float left, float right);
bool llama_common_marker_is_opener_rust(uint8_t c);
bool llama_common_marker_is_closer_rust(uint8_t opener, uint8_t c);
bool llama_common_content_is_always_wrapped_rust(int mode, size_t start_len, size_t end_len);
bool llama_common_cpu_has_hybrid_bit_rust(uint32_t edx);
bool llama_common_cpu_is_intel_atom_core_type_rust(uint32_t eax);
bool llama_common_any_terminal_rust(bool stdout_is_tty, bool stderr_is_tty);
bool llama_common_arg_has_env_value_rust(bool has_env, bool has_neg_env_value, bool has_env_value);
bool llama_common_jinja_int_is_odd_rust(int64_t value);
bool llama_common_jinja_int_is_even_rust(int64_t value);
int64_t llama_common_jinja_int_abs_rust(int64_t value);
double llama_common_jinja_float_abs_rust(double value);
bool llama_common_jinja_is_false_rust(bool is_bool, bool value);
bool llama_common_jinja_is_true_rust(bool is_bool, bool value);
bool llama_common_jinja_is_defined_rust(bool is_undefined);
bool llama_common_jinja_compare_f64_rust(double left, double right, int op);
bool llama_common_jinja_compare_bool_rust(bool left, bool right, int op);
bool llama_common_jinja_bool_not_rust(bool value);
bool llama_common_jinja_membership_result_rust(bool member, bool is_not_in);
bool llama_common_jinja_compare_bytes_rust(
        const uint8_t * left, size_t left_len,
        const uint8_t * right, size_t right_len,
        int op);
bool llama_common_ring_buffer_is_empty_rust(size_t size);
bool llama_common_is_http_status_ok_rust(int status);
llama_common_unicode_string llama_common_model_endpoint_rust(
        const uint8_t * model_endpoint, size_t model_endpoint_len,
        const uint8_t * hf_endpoint, size_t hf_endpoint_len);
llama_common_unicode_string llama_common_read_etag_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_get_cached_ref_rust(const uint8_t * data, size_t len);
bool llama_common_safe_write_file_rust(
        const uint8_t * path, size_t path_len,
        const uint8_t * data, size_t data_len);
bool llama_common_fs_is_directory_rust(const uint8_t * path, size_t path_len);
bool llama_common_is_lfm2_template_rust(const uint8_t * data, size_t len);
int llama_common_specialized_chat_template_rust(const uint8_t * data, size_t len);
bool llama_common_is_gemma4_modern_template_rust(const uint8_t * data, size_t len);
bool llama_common_needs_gpt_oss_channel_template_patch_rust(const uint8_t * data, size_t len);
bool llama_common_needs_mistral_tool_calls_template_patch_rust(const uint8_t * data, size_t len);
bool llama_common_grammar_should_apply_rust(
        bool has_grammar,
        bool has_reasoning_budget,
        bool grammar_lazy,
        int reasoning_state,
        int idle_state,
        int done_state);
bool llama_common_gguf_filename_is_model_rust(const uint8_t * data, size_t len);
bool llama_common_is_hex_digit_rust(uint8_t c);
size_t llama_common_parse_hex_escape_rust(
        const uint8_t * data, size_t len,
        size_t pos, int hex_count,
        uint32_t * out_value);
size_t llama_common_parse_char_class_char_rust(
        const uint8_t * data, size_t len,
        size_t pos,
        uint32_t * out_value);
const char * llama_common_chat_format_name_rust(int format);
const char * llama_common_reasoning_format_name_rust(int format);
int llama_common_reasoning_format_from_name_rust(const char * format);
char llama_common_sampler_type_to_chr_rust(int type);
const char * llama_common_sampler_type_to_str_rust(int type);
int llama_common_sampler_type_from_name_rust(const char * name, bool allow_alt_names);
int llama_common_sampler_type_from_chr_rust(char name);
const char * llama_common_speculative_type_to_str_rust(int type);
int llama_common_speculative_type_from_name_rust(const char * name);
const char * llama_common_peg_parse_result_type_name_rust(int type);
llama_common_unicode_string llama_common_peg_rule_name_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_gbnf_format_literal_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_gbnf_build_repetition_rust(
        const uint8_t * item_rule, size_t item_rule_len,
        int min_items, int max_items,
        const uint8_t * separator_rule, size_t separator_rule_len);
llama_common_unicode_string llama_common_gbnf_escape_char_class_rust(uint32_t c);
bool llama_common_gbnf_is_reserved_name_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_gbnf_ref_name_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_hf_folder_name_to_repo_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_hf_repo_to_folder_name_rust(const uint8_t * data, size_t len);
llama_common_unicode_string llama_common_hf_make_old_cache_filename_rust(
        const uint8_t * owner, size_t owner_len,
        const uint8_t * repo, size_t repo_len,
        const uint8_t * filename, size_t filename_len);
int llama_common_json_brace_depth_rust(const uint8_t * data, size_t len);
bool llama_common_hf_is_valid_repo_id_rust(const uint8_t * data, size_t len);
bool llama_common_hf_is_valid_token_rust(const uint8_t * data, size_t len);
bool llama_common_hf_is_valid_commit_rust(const uint8_t * data, size_t len);
bool llama_common_hf_is_valid_oid_rust(const uint8_t * data, size_t len);
common_rust_string_list llama_common_hf_parse_manifest_name_rust(const uint8_t * data, size_t len);
common_rust_gguf_split_info llama_common_gguf_split_info_rust(const uint8_t * data, size_t len, bool extract_tag);
int llama_common_gguf_extract_quant_bits_rust(const uint8_t * data, size_t len);
}

static std::string common_rust_string_to_std(llama_common_unicode_string value) {
    if (!value.data) {
        return "";
    }
    std::string result((const char *) value.data, value.len);
    llama_common_unicode_string_free(value);
    return result;
}

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/types.h>
#include <pwd.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

common_time_meas::~common_time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

//
// CPU utils
//

int32_t common_cpu_get_num_physical_cores() {
#ifdef __linux__
    // enumerate the set of thread siblings, num entries is num cores
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu=0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings("/sys/devices/system/cpu/cpu"
            + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) {
            break; // no more cpus
        }
        std::string line;
        if (std::getline(thread_siblings, line)) {
            siblings.insert(line);
        }
    }
    if (!siblings.empty()) {
        return static_cast<int32_t>(siblings.size());
    }
#elif defined(__APPLE__) && defined(__MACH__)
    int32_t num_physical_cores;
    size_t len = sizeof(num_physical_cores);
    int result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
    result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
    if (result == 0) {
        return num_physical_cores;
    }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    unsigned int n_threads_win = std::thread::hardware_concurrency();
    unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

    DWORD buffer_size = 0;
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return default_threads;
        }
    }

    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &buffer_size)) {
        return default_threads;
    }

    int32_t num_physical_cores = 0;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    while (buffer_size > 0) {
        if (info->Relationship == RelationProcessorCore) {
            num_physical_cores += info->Processor.GroupCount;
        }
        buffer_size -= info->Size;
        info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char*>(info) + info->Size);
    }

    return llama_common_default_thread_count_windows_rust(num_physical_cores, default_threads);
#endif
    unsigned int n_threads = std::thread::hardware_concurrency();
    return llama_common_default_thread_count_rust(n_threads);
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void cpuid(unsigned leaf, unsigned subleaf,
                  unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx) {
    __asm__("movq\t%%rbx,%%rsi\n\t"
            "cpuid\n\t"
            "xchgq\t%%rbx,%%rsi"
            : "=a"(*eax), "=S"(*ebx), "=c"(*ecx), "=d"(*edx)
            : "0"(leaf), "2"(subleaf));
}

static int pin_cpu(int cpu) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
}

static bool is_hybrid_cpu(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return llama_common_cpu_has_hybrid_bit_rust(edx);
}

static bool is_running_on_efficiency_core(void) {
    unsigned eax, ebx, ecx, edx;
    cpuid(0x1a, 0, &eax, &ebx, &ecx, &edx);
    return llama_common_cpu_is_intel_atom_core_type_rust(eax);
}

static int cpu_count_math_cpus(int n_cpu) {
    int result = 0;
    for (int cpu = 0; cpu < n_cpu; ++cpu) {
        if (pin_cpu(cpu)) {
            return -1;
        }
        if (is_running_on_efficiency_core()) {
            continue; // efficiency cores harm lockstep threading
        }
        ++cpu; // hyperthreading isn't useful for linear algebra
        ++result;
    }
    return result;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t common_cpu_get_num_math() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
    int n_cpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpu < 1) {
        return common_cpu_get_num_physical_cores();
    }
    if (is_hybrid_cpu()) {
        cpu_set_t affinity;
        if (!pthread_getaffinity_np(pthread_self(), sizeof(affinity), &affinity)) {
            int result = cpu_count_math_cpus(n_cpu);
            pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
            if (result > 0) {
                return result;
            }
        }
    }
#endif
    return common_cpu_get_num_physical_cores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    DWORD p = NORMAL_PRIORITY_CLASS;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p = BELOW_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_NORMAL:   p = NORMAL_PRIORITY_CLASS;       break;
        case GGML_SCHED_PRIO_MEDIUM:   p = ABOVE_NORMAL_PRIORITY_CLASS; break;
        case GGML_SCHED_PRIO_HIGH:     p = HIGH_PRIORITY_CLASS;         break;
        case GGML_SCHED_PRIO_REALTIME: p = REALTIME_PRIORITY_CLASS;     break;
    }

    if (!SetPriorityClass(GetCurrentProcess(), p)) {
        LOG_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool set_process_priority(enum ggml_sched_priority prio) {
    if (prio == GGML_SCHED_PRIO_NORMAL) {
        return true;
    }

    int p = 0;
    switch (prio) {
        case GGML_SCHED_PRIO_LOW:      p =  5;  break;
        case GGML_SCHED_PRIO_NORMAL:   p =  0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   p = -5;  break;
        case GGML_SCHED_PRIO_HIGH:     p = -10; break;
        case GGML_SCHED_PRIO_REALTIME: p = -20; break;
    }

    if (setpriority(PRIO_PROCESS, 0, p) != 0) {
        LOG_WRN("failed to set process priority %d : %s (%d)\n", prio, strerror(errno), errno);
        return false;
    }
    return true;
}

#endif

//
// CLI argument parsing
//


void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model) {
    int32_t n_set = 0;

    if (cpuparams.n_threads < 0) {
        // Assuming everything about cpuparams is invalid
        if (role_model != nullptr) {
            cpuparams = *role_model;
        } else {
            cpuparams.n_threads = common_cpu_get_num_math();
        }
    }

    for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (cpuparams.cpumask[i]) {
            n_set++;
        }
    }

    if (n_set && n_set < cpuparams.n_threads) {
        // Not enough set bits, may experience performance issues.
        LOG_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", n_set, cpuparams.n_threads);
    }
}

bool parse_cpu_range(const std::string & range, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    size_t dash_loc = range.find('-');
    if (dash_loc == std::string::npos) {
        LOG_ERR("Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
        return false;
    }

    size_t start_i;
    size_t end_i;

    if (dash_loc == 0) {
        start_i = 0;
    } else {
        start_i = std::stoull(range.substr(0, dash_loc));
        if (start_i >= GGML_MAX_N_THREADS) {
            LOG_ERR("Start index out of bounds!\n");
            return false;
        }
    }

    if (dash_loc == range.length() - 1) {
        end_i = GGML_MAX_N_THREADS - 1;
    } else {
        end_i = std::stoull(range.substr(dash_loc + 1));
        if (end_i >= GGML_MAX_N_THREADS) {
            LOG_ERR("End index out of bounds!\n");
            return false;
        }
    }

    for (size_t i = start_i; i <= end_i; i++) {
        boolmask[i] = true;
    }

    return true;
}

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    const int status = llama_common_parse_cpu_mask_rust(
            (const uint8_t *) mask.data(), mask.size(), boolmask, GGML_MAX_N_THREADS);
    if (status >= 0) {
        LOG_ERR("Invalid hex character '%c' at position %d\n", mask.at(status), status);
        return false;
    }
    return true;
}

void common_init() {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    llama_log_set(common_log_default_callback, NULL);

#ifdef NDEBUG
    const char * build_type = "";
#else
    const char * build_type = " (debug)";
#endif

    LOG_DBG("build: %d (%s) with %s for %s%s\n", llama_build_number(), llama_commit(), llama_compiler(), llama_build_target(), build_type);
}

std::string common_params_get_system_info(const common_params & params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << params.cpuparams.n_threads;
    if (params.cpuparams_batch.n_threads != -1) {
        os << " (n_threads_batch = " << params.cpuparams_batch.n_threads << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    return os.str();
}

//
// String utils
//

std::string string_format(const char * fmt, ...) {
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
    return std::string(buf.data(), size);
}

std::string string_strip(const std::string & str) {
    llama_common_unicode_string trimmed = llama_common_string_trim_rust((const uint8_t *) str.data(), str.size(), 0);
    if (!trimmed.data) {
        return "";
    }
    std::string result((const char *) trimmed.data, trimmed.len);
    llama_common_unicode_string_free(trimmed);
    return result;
}

std::string string_get_sortable_timestamp() {
    using clock = std::chrono::system_clock;

    const clock::time_point current_time = clock::now();
    const time_t as_time_t = clock::to_time_t(current_time);
    char timestamp_no_ns[100];
    std::strftime(timestamp_no_ns, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&as_time_t));

    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        current_time.time_since_epoch() % 1000000000).count();
    char timestamp_ns[11];
    snprintf(timestamp_ns, 11, "%09" PRId64, ns);

    return std::string(timestamp_no_ns) + "." + std::string(timestamp_ns);
}

void string_replace_all(std::string & s, const std::string & search, const std::string & replace) {
    llama_common_unicode_string replaced = llama_common_string_replace_all_rust(
        (const uint8_t *) s.data(), s.size(),
        (const uint8_t *) search.data(), search.size(),
        (const uint8_t *) replace.data(), replace.size());
    if (!replaced.data) {
        return;
    }
    s.assign((const char *) replaced.data, replaced.len);
    llama_common_unicode_string_free(replaced);
}

std::string regex_escape(const std::string & s) {
    llama_common_unicode_string escaped = llama_common_regex_escape_rust((const uint8_t *) s.data(), s.size());
    if (!escaped.data) {
        return "";
    }
    std::string result((const char *) escaped.data, escaped.len);
    llama_common_unicode_string_free(escaped);
    return result;
}

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::vector<common_rust_string_view> views;
    views.reserve(values.size());
    for (const std::string & value : values) {
        views.push_back({ (const uint8_t *) value.data(), value.size() });
    }
    llama_common_unicode_string joined = llama_common_string_join_rust(
        views.data(), views.size(),
        (const uint8_t *) separator.data(), separator.size());
    if (!joined.data) {
        return "";
    }
    std::string result((const char *) joined.data, joined.len);
    llama_common_unicode_string_free(joined);
    return result;
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    common_rust_string_list split = llama_common_string_split_rust(
        (const uint8_t *) str.data(), str.size(),
        (const uint8_t *) delimiter.data(), delimiter.size());
    std::vector<std::string> parts;
    if (!split.data) {
        return parts;
    }
    parts.reserve(split.len);
    for (size_t i = 0; i < split.len; ++i) {
        const llama_common_unicode_string & item = split.data[i];
        parts.emplace_back((const char *) item.data, item.len);
    }
    llama_common_string_list_free(split);
    return parts;
}

std::string string_repeat(const std::string & str, size_t n) {
    llama_common_unicode_string repeated = llama_common_string_repeat_rust((const uint8_t *) str.data(), str.size(), n);
    if (!repeated.data) {
        return "";
    }
    std::string result((const char *) repeated.data, repeated.len);
    llama_common_unicode_string_free(repeated);
    return result;
}

std::string string_from(bool value) {
    return llama_common_bool_to_string_rust(value);
}

std::string string_from(const std::vector<int> & values) {
    llama_common_unicode_string formatted = llama_common_int_vec_to_string_rust(values.data(), values.size());
    if (!formatted.data) {
        return "";
    }
    std::string result((const char *) formatted.data, formatted.len);
    llama_common_unicode_string_free(formatted);
    return result;
}

std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (const auto & token : tokens) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, token);

        buf << "'" << detokenized << "'"
            << ":" << std::to_string(token);
    }

    buf << " ]";

    return buf.str();
}

std::string string_from(const struct llama_context * ctx, const struct llama_batch & batch) {
    std::stringstream buf;

    buf << "[ ";

    bool first = true;
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (!first) {
            buf << ", ";
        } else {
            first = false;
        }

        auto detokenized = common_token_to_piece(ctx, batch.token[i]);

        buf << "\n"          << std::to_string(i)
            << ", token '"   << detokenized << "'"
            << ", pos "      << std::to_string(batch.pos[i])
            << ", n_seq_id " << std::to_string(batch.n_seq_id[i])
            << ", seq_id "   << std::to_string(batch.seq_id[i][0])
            << ", logits "   << std::to_string(batch.logits[i]);
    }

    buf << " ]";

    return buf.str();
}

void string_process_escapes(std::string & input) {
    llama_common_unicode_string processed = llama_common_string_process_escapes_rust((const uint8_t *) input.data(), input.size());
    if (!processed.data) {
        return;
    }
    input.assign((const char *) processed.data, processed.len);
    llama_common_unicode_string_free(processed);
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides) {
    const char * sep = strchr(data, '=');
    if (sep == nullptr || sep - data >= 128) {
        LOG_ERR("%s: malformed KV override '%s'\n", __func__, data);
        return false;
    }
    llama_model_kv_override kvo;
    std::strncpy(kvo.key, data, sep - data);
    kvo.key[sep - data] = 0;
    sep++;
    if (strncmp(sep, "int:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        kvo.val_i64 = std::atol(sep);
    } else if (strncmp(sep, "float:", 6) == 0) {
        sep += 6;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
        kvo.val_f64 = std::atof(sep);
    } else if (strncmp(sep, "bool:", 5) == 0) {
        sep += 5;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_BOOL;
        if (std::strcmp(sep, "true") == 0) {
            kvo.val_bool = true;
        } else if (std::strcmp(sep, "false") == 0) {
            kvo.val_bool = false;
        } else {
            LOG_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
            return false;
        }
    } else if (strncmp(sep, "str:", 4) == 0) {
        sep += 4;
        kvo.tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        if (strlen(sep) > 127) {
            LOG_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
            return false;
        }
        strncpy(kvo.val_str, sep, 127);
        kvo.val_str[127] = '\0';
    } else {
        LOG_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
        return false;
    }
    overrides.emplace_back(std::move(kvo));
    return true;
}

bool glob_match(const std::string & pattern, const std::string & str) {
    return llama_common_glob_match_rust(
            (const uint8_t *) pattern.data(), pattern.size(),
            (const uint8_t *) str.data(), str.size());
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fs_validate_filename(const std::string & filename, bool allow_subdirs) {
    return llama_common_fs_validate_filename_rust((const uint8_t *) filename.data(), filename.size(), allow_subdirs);
}

#include <iostream>


#ifdef _WIN32
static std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}
#endif

// returns true if successful, false otherwise
bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_wstring(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);

        pos_slash += 1;

        // skip the drive letter, in some systems it can return an access denied error
        if (subpath.length() == 2 && subpath[1] == ':') {
            continue;
        }

        const bool success = CreateDirectoryW(subpath.c_str(), NULL);

        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

bool fs_is_directory(const std::string & path) {
    return llama_common_fs_is_directory_rust((const uint8_t *) path.data(), path.size());
}

std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || \
        defined(__OpenBSD__) || defined(__NetBSD__)
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else if (std::getenv("HOME")) {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        } else {
#if defined(__linux__)
            /* no $HOME is defined, fallback to getpwuid */
            struct passwd *pw = getpwuid(getuid());
            if ((!pw) || (!pw->pw_dir)) {
                throw std::runtime_error("Failed to find $HOME directory");
            }

            cache_directory = std::string(pw->pw_dir) + std::string("/.cache/");
#else /* defined(__linux__) */
            throw std::runtime_error("Failed to find $HOME directory");
#endif /* defined(__linux__) */
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#elif defined(__EMSCRIPTEN__)
        GGML_ABORT("not implemented on this platform");
#else
#  error Unknown architecture
#endif
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

std::string fs_get_cache_file(const std::string & filename) {
    GGML_ASSERT(filename.find(DIRECTORY_SEPARATOR) == std::string::npos);
    std::string cache_directory = fs_get_cache_directory();
    const bool success = fs_create_directory_with_parents(cache_directory);
    if (!success) {
        throw std::runtime_error("failed to create cache directory: " + cache_directory);
    }
    return cache_directory + filename;
}

std::vector<common_file_info> fs_list(const std::string & path, bool include_directories) {
    std::vector<common_file_info> files;
    if (path.empty()) return files;

    std::filesystem::path dir(path);
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return files;
    }

    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        try {
            // Only include regular files (skip directories)
            const auto & p = entry.path();
            if (std::filesystem::is_regular_file(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.is_dir = false;
                try {
                    info.size = static_cast<size_t>(std::filesystem::file_size(p));
                } catch (const std::filesystem::filesystem_error &) {
                    info.size = 0;
                }
                files.push_back(std::move(info));
            } else if (include_directories && std::filesystem::is_directory(p)) {
                common_file_info info;
                info.path   = p.string();
                info.name   = p.filename().string();
                info.size   = 0; // Directories have no size
                info.is_dir = true;
                files.push_back(std::move(info));
            }
        } catch (const std::filesystem::filesystem_error &) {
            // skip entries we cannot inspect
            continue;
        }
    }

    return files;
}

//
// TTY utils
//

bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    bool stdout_is_tty = isatty(fileno(stdout));
    bool stderr_is_tty = isatty(fileno(stderr));

    return llama_common_any_terminal_rust(stdout_is_tty, stderr_is_tty);
}

//
// Model utils
//

// TODO: move to common/sampling
static void common_init_sampler_from_model(
    const llama_model * model,
    common_params_sampling & sparams) {

    const uint64_t config = sparams.user_sampling_config;

    auto get_int32 = [&](const char * key, int32_t & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[64] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            int32_t v = strtol(buf, &end, 10);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    auto get_float = [&](const char * key, float & dst, uint64_t user_config) {
        if (config & user_config) {
            return;
        }

        char buf[128] = {0};
        if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
            char * end = nullptr;
            float v = strtof(buf, &end);
            if (end && end != buf) {
                dst = v;
            }
        }
    };

    // Sampling sequence
    if (!(config & common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS)) {
        char buf[512] = {0};
        if (llama_model_meta_val_str(model, llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE), buf, sizeof(buf)) > 0) {
            const std::vector<std::string> sampler_names = string_split<std::string>(std::string(buf), ';');
            if (!sampler_names.empty()) {
                sparams.samplers = common_sampler_types_from_names(sampler_names, true);
            }
        }
    }

    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K),           sparams.top_k,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P),           sparams.top_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P),           sparams.min_p,           common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), sparams.xtc_probability, common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD),   sparams.xtc_threshold,   common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_TEMP),            sparams.temp,            common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N),  sparams.penalty_last_n,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT),  sparams.penalty_repeat,  common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT);
    get_int32(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT),        sparams.mirostat,        common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU),    sparams.mirostat_tau,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU);
    get_float(llama_model_meta_key_str(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA),    sparams.mirostat_eta,    common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA);
}

struct common_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top

    llama_model_ptr   model;
    llama_context_ptr context;

    std::vector<llama_adapter_lora_ptr> lora;

    std::vector<common_sampler_ptr> samplers;
    std::vector<llama_sampler_seq_config> samplers_seq_config;
};

common_init_result::common_init_result(common_params & params) :
    pimpl(new impl{}) {
    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    if (params.fit_params) {
        LOG_INF("%s: fitting params to device memory, for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only occurs with -fit on\n", __func__);
        common_fit_params(params.model.path.c_str(), &mparams, &cparams,
            params.tensor_split,
            params.tensor_buft_overrides.data(),
            params.fit_params_target.data(),
            params.fit_params_min_ctx,
            params.verbosity >= 4 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
    }

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == NULL) {
        return;
    }

    pimpl->model.reset(model);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // load and optionally apply lora adapters
    for (auto & la : params.lora_adapters) {
        llama_adapter_lora_ptr lora;
        lora.reset(llama_adapter_lora_init(model, la.path.c_str()));
        if (lora == nullptr) {
            LOG_ERR("%s: failed to load lora adapter '%s'\n", __func__, la.path.c_str());
            pimpl->model.reset(model);
            return;
        }

        char buf[1024];
        la.ptr = lora.get();
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.task_name", buf, sizeof(buf));
        la.task_name = buf;
        llama_adapter_meta_val_str(la.ptr, "adapter.lora.prompt_prefix", buf, sizeof(buf));
        la.prompt_prefix = buf;
        pimpl->lora.emplace_back(std::move(lora)); // copy to list of loaded adapters
    }

    // updates params.sampling
    // TODO: fix naming
    common_init_sampler_from_model(model, params.sampling);

    if (params.sampling.ignore_eos && llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        LOG_WRN("%s: warning: vocab does not have an EOS token, ignoring --ignore-eos\n", __func__);
        params.sampling.ignore_eos = false;
    }

    // initialize once
    for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            LOG_INF("%s: added %s logit bias = %f\n", __func__, common_token_to_piece(vocab, i).c_str(), -INFINITY);
            params.sampling.logit_bias_eog.push_back({i, -INFINITY});
        }
    }

    if (params.sampling.ignore_eos) {
        // add EOG biases to the active set of logit biases
        params.sampling.logit_bias.insert(
                params.sampling.logit_bias.end(),
                params.sampling.logit_bias_eog.begin(), params.sampling.logit_bias_eog.end());
    }

    //if (params.sampling.penalty_last_n == -1) {
    //    LOG_INF("%s: setting penalty_last_n to ctx_size = %d\n", __func__, llama_n_ctx(lctx));
    //    params.sampling.penalty_last_n = llama_n_ctx(lctx);
    //}

    //if (params.sampling.dry_penalty_last_n == -1) {
    //    LOG_INF("%s: setting dry_penalty_last_n to ctx_size = %d\n", __func__, llama_n_ctx(lctx));
    //    params.sampling.dry_penalty_last_n = llama_n_ctx(lctx);
    //}

    // init the backend samplers as part of the context creation
    pimpl->samplers.resize(cparams.n_seq_max);
    pimpl->samplers_seq_config.resize(cparams.n_seq_max);

    for (int i = 0; i < (int) cparams.n_seq_max; ++i) {
        pimpl->samplers[i].reset(common_sampler_init(model, params.sampling));
        pimpl->samplers_seq_config[i] = { i, common_sampler_get(pimpl->samplers[i].get()) };
    }

    if (params.sampling.backend_sampling) {
        cparams.samplers   = pimpl->samplers_seq_config.data();
        cparams.n_samplers = pimpl->samplers_seq_config.size();
    }

    llama_context * lctx = llama_init_from_model(model, cparams);
    if (lctx == NULL) {
        LOG_ERR("%s: failed to create context with model '%s'\n", __func__, params.model.path.c_str());
        return;
    }

    pimpl->context.reset(lctx);
}

llama_model * common_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_init_result::context() {
    return pimpl->context.get();
}

common_sampler * common_init_result::sampler(llama_seq_id seq_id) {
    if (seq_id < 0 || seq_id >= (int) pimpl->samplers.size()) {
        return nullptr;
    }
    return pimpl->samplers[seq_id].get();
}

void common_init_result::reset_samplers() {
    for (int i = 0; i < (int) pimpl->samplers.size(); ++i) {
        llama_sampler_reset(common_sampler_get(pimpl->samplers[i].get()));
    }
}

std::vector<llama_adapter_lora_ptr> & common_init_result::lora() {
    return pimpl->lora;
}

common_init_result_ptr common_init_from_params(common_params & params) {
    common_init_result_ptr res(new common_init_result(params));

    llama_model * model = res->model();
    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return res;
    }

    llama_context * lctx = res->context();
    if (lctx == NULL) {
        LOG_ERR("%s: failed to create context with model '%s'\n", __func__, params.model.path.c_str());
        return res;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (params.ctx_shift && !llama_memory_can_shift(llama_get_memory(lctx))) {
        LOG_WRN("%s: KV cache shifting is not supported for this context, disabling KV cache shifting\n", __func__);
        params.ctx_shift = false;
    }

    if (!params.control_vectors.empty()) {
        if (params.control_vector_layer_start <= 0) params.control_vector_layer_start = 1;
        if (params.control_vector_layer_end   <= 0) params.control_vector_layer_end   = llama_model_n_layer(model);

        const auto cvec = common_control_vector_load(params.control_vectors);
        if (cvec.n_embd == -1) {
            return res;
        }

        int err = llama_set_adapter_cvec(
                lctx,
                cvec.data.data(),
                cvec.data.size(),
                cvec.n_embd,
                params.control_vector_layer_start,
                params.control_vector_layer_end);
        if (err) {
            return res;
        }
    }

    if (llama_pooling_type(lctx) == LLAMA_POOLING_TYPE_RANK) {
        bool ok = true;

        if (llama_vocab_bos(vocab) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: vocab does not have a  BOS token, reranking will not work\n", __func__);
            ok = false;
        }

        bool has_eos = llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL;
        bool has_sep = llama_vocab_sep(vocab) != LLAMA_TOKEN_NULL;
        bool has_rerank_prompt = llama_model_chat_template(model, "rerank") != NULL;

        if (!has_eos && !has_sep && !has_rerank_prompt) {
            LOG_WRN("%s: warning: vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n", __func__);
            ok = false;
        } else if (!has_eos) {
            LOG_WRN("%s: warning: vocab does not have an EOS token, using SEP token as fallback\n", __func__);
        }

        if (!ok) {
            return res;
        }
    }

    if (!params.lora_init_without_apply) {
        common_set_adapter_lora(lctx, params.lora_adapters);
    }

    if (params.warmup) {
        LOG_WRN("%s: warming up the model with an empty run - please wait ... (--no-warmup to disable)\n", __func__);

        llama_set_warmup(lctx, true);

        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);

        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            llama_encode(lctx, llama_batch_get_one(tmp.data(), tmp.size()));
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params.n_batch)));
        }
        llama_memory_clear(llama_get_memory(lctx), true);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);
        llama_set_warmup(lctx, false);

        // reset samplers to reset RNG state after warmup to the seeded state
        res->reset_samplers();
    }

    return res;
}

common_init_result::~common_init_result() = default;

std::string common_get_model_endpoint() {
    const char * model_endpoint_env = getenv("MODEL_ENDPOINT");
    // We still respect the use of environment-variable "HF_ENDPOINT" for backward-compatibility.
    const char * hf_endpoint_env = getenv("HF_ENDPOINT");
    return common_rust_string_to_std(llama_common_model_endpoint_rust(
            model_endpoint_env ? (const uint8_t *) model_endpoint_env : nullptr,
            model_endpoint_env ? strlen(model_endpoint_env) : 0,
            hf_endpoint_env ? (const uint8_t *) hf_endpoint_env : nullptr,
            hf_endpoint_env ? strlen(hf_endpoint_env) : 0));
}

common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx) {
    auto * mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        return COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    }

    common_context_seq_rm_type res = COMMON_CONTEXT_SEQ_RM_TYPE_PART;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
        res = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx);

    return res;
}

void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora) {
    std::vector<llama_adapter_lora *> loras;
    std::vector<float> scales;

    for (auto & la: lora) {
        loras.push_back(la.ptr);
        scales.push_back(la.scale);
    }

    llama_set_adapters_lora(ctx, loras.data(), loras.size(), scales.data());
}

struct llama_model_params common_model_params_to_llama(common_params & params) {
    auto mparams = llama_model_default_params();

    if (!params.devices.empty()) {
        mparams.devices = params.devices.data();
    }

    mparams.n_gpu_layers    = params.n_gpu_layers;
    mparams.main_gpu        = params.main_gpu;
    mparams.split_mode      = params.split_mode;
    mparams.tensor_split    = params.tensor_split;
    mparams.use_mmap        = params.use_mmap;
    mparams.use_direct_io   = params.use_direct_io;
    mparams.use_mlock       = params.use_mlock;
    mparams.check_tensors   = params.check_tensors;
    mparams.use_extra_bufts = !params.no_extra_bufts;
    mparams.no_host         = params.no_host;

    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    } else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    if (params.tensor_buft_overrides.empty()) {
        mparams.tensor_buft_overrides = NULL;
    } else {
        GGML_ASSERT(params.tensor_buft_overrides.back().pattern == nullptr && "Tensor buffer overrides not terminated with empty pattern");
        mparams.tensor_buft_overrides = params.tensor_buft_overrides.data();
    }

    mparams.progress_callback           = params.load_progress_callback;
    mparams.progress_callback_user_data = params.load_progress_callback_user_data;
    mparams.no_alloc                    = params.no_alloc;

    return mparams;
}

struct llama_context_params common_context_params_to_llama(const common_params & params) {
    auto cparams = llama_context_default_params();

    cparams.n_ctx             = params.n_ctx;
    cparams.n_seq_max         = params.n_parallel;
    cparams.n_batch           = params.n_batch;
    cparams.n_ubatch          = params.n_ubatch;
    cparams.n_threads         = params.cpuparams.n_threads;
    cparams.n_threads_batch   = params.cpuparams_batch.n_threads == -1 ?
                                params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.embeddings        = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base    = params.rope_freq_base;
    cparams.rope_freq_scale   = params.rope_freq_scale;
    cparams.yarn_ext_factor   = params.yarn_ext_factor;
    cparams.yarn_attn_factor  = params.yarn_attn_factor;
    cparams.yarn_beta_fast    = params.yarn_beta_fast;
    cparams.yarn_beta_slow    = params.yarn_beta_slow;
    cparams.yarn_orig_ctx     = params.yarn_orig_ctx;
    cparams.pooling_type      = params.pooling_type;
    cparams.attention_type    = params.attention_type;
    cparams.flash_attn_type   = params.flash_attn_type;
    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;
    cparams.offload_kqv       = !params.no_kv_offload;
    cparams.no_perf           = params.no_perf;
    cparams.op_offload        = !params.no_op_offload;
    cparams.swa_full          = params.swa_full;
    cparams.kv_unified        = params.kv_unified;

    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;

    return cparams;
}

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params) {
    struct ggml_threadpool_params tpp;

    ggml_threadpool_params_init(&tpp, params.n_threads); // setup the defaults

    if (params.mask_valid) {
        std::memcpy(&tpp.cpumask, &params.cpumask, GGML_MAX_N_THREADS);
    }

    tpp.prio       = params.priority;
    tpp.poll       = params.poll;
    tpp.strict_cpu = params.strict_cpu;

    return tpp;
}

//
// Batch utils
//

void common_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");

    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

//
// Vocab utils
//

std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_tokenize(vocab, text, add_special, parse_special);
}

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string common_token_to_piece(const struct llama_context * ctx, llama_token token, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_token_to_piece(vocab, token, special);
}

std::string common_token_to_piece(const struct llama_vocab * vocab, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

std::string common_detokenize(const struct llama_context * ctx, const std::vector<llama_token> & tokens, bool special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return common_detokenize(vocab, tokens, special);
}

std::string common_detokenize(const struct llama_vocab * vocab, const std::vector<llama_token> & tokens, bool special) {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = llama_detokenize(vocab, tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

//
// Embedding utils
//

void common_embd_normalize(const float * inp, float * out, int n, int embd_norm) {
    llama_common_embd_normalize_rust(inp, out, n, embd_norm);
}

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n){
    return llama_common_embd_similarity_cos_rust(embd1, embd2, n);
}

//
// Control vector utils
//

static common_control_vector_data common_control_vector_load_one(const common_control_vector_load_info & load_info) {
    common_control_vector_data result = { -1, {} };

    ggml_context * ctx = nullptr;
    struct gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ false,
        /* .ctx      = */ &ctx,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(load_info.fname.c_str(), meta_gguf_params);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to load control vector file from %s\n", __func__, load_info.fname.c_str());
        return result;
    }

    int32_t n_tensors = gguf_get_n_tensors(ctx_gguf);
    if (n_tensors == 0) {
        LOG_WRN("%s: no direction tensors found in %s\n", __func__, load_info.fname.c_str());
    }

    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf, i);

        int layer_idx = -1;

        // split on '.'
        size_t dotpos = name.find('.');
        if (dotpos != std::string::npos && name.substr(0, dotpos) == "direction") {
            try {
                layer_idx = std::stoi(name.substr(dotpos + 1));
            } catch (...) {
                layer_idx = -1;
            }
        }
        if (layer_idx < 0) {
            LOG_ERR("%s: invalid/unparsable direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        } else if (layer_idx == 0) {
            LOG_ERR("%s: invalid (zero) direction tensor layer index in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        struct ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
        if (tensor->type != GGML_TYPE_F32) {
            LOG_ERR("%s: invalid (non-F32) direction tensor type in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }
        if (ggml_n_dims(tensor) != 1) {
            LOG_ERR("%s: invalid (non-1D) direction tensor shape in %s\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result.n_embd = ggml_nelements(tensor);
        } else if (ggml_nelements(tensor) != result.n_embd) {
            LOG_ERR("%s: direction tensor in %s does not match previous dimensions\n", __func__, load_info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        // extend if necessary - do not store data for layer 0 (it's not used)
        result.data.resize(std::max(result.data.size(), static_cast<size_t>(result.n_embd * layer_idx)), 0.0f);

        const float * src = (const float *) tensor->data;
        float * dst = result.data.data() + result.n_embd * (layer_idx - 1);  // layer 1 at [0]
        for (int j = 0; j < result.n_embd; j++) {
            dst[j] += src[j] * load_info.strength;  // allows multiple directions for same layer in same file
        }

    }

    if (result.n_embd == -1) {
        LOG_WRN("%s: skipping %s due to invalid direction tensors\n", __func__, load_info.fname.c_str());
        result.data.clear();
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx);

    return result;
}

common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos) {
    common_control_vector_data result = { -1, {} };

    for (const auto & info : load_infos) {
        auto cur = common_control_vector_load_one(info);

        if (cur.n_embd == -1) {
            result.n_embd = -1;
            break;
        }
        if (result.n_embd != -1 && result.n_embd != cur.n_embd) {
            LOG_ERR("%s: control vectors in %s does not match previous dimensions\n", __func__, info.fname.c_str());
            result.n_embd = -1;
            break;
        }

        if (result.n_embd == -1) {
            result = std::move(cur);
        } else {
            result.data.resize(std::max(result.data.size(), cur.data.size()), 0.0f);  // extend if necessary
            for (size_t i = 0; i < cur.data.size(); i++) {
                result.data[i] += cur.data[i];
            }
        }
    }

    if (result.n_embd == -1) {
        LOG_ERR("%s: no valid control vector files passed\n", __func__);
        result.data.clear();
    }

    return result;
}

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride) {
    const int64_t ne_datapoint = llama_n_ctx(ctx);
    const int64_t ndata        = (tokens.size() - ne_datapoint - 1) / stride;
    ggml_opt_dataset_t result = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, ne_datapoint, ne_datapoint, ndata, /*ndata_shard =*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data(result)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(result)->data;

    for (int64_t idata = 0; idata < ndata; ++idata) {
        memcpy(data   + idata*ne_datapoint, tokens.data() + idata*stride + 0, ne_datapoint*sizeof(llama_token));
        memcpy(labels + idata*ne_datapoint, tokens.data() + idata*stride + 1, ne_datapoint*sizeof(llama_token));
    }

    return result;
}

ggml_opt_optimizer_params common_opt_lr_pars(void * userdata) {
    ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
    const lr_opt &            d      = *(lr_opt *) userdata;
    result.adamw.alpha = result.sgd.alpha = d.get_lr(d.epoch);
    result.sgd.wd = result.adamw.wd = d.wd;
    return result;
}

enum ggml_opt_optimizer_type common_opt_get_optimizer(const char * n) {
    return (enum ggml_opt_optimizer_type) llama_common_opt_get_optimizer_rust(n);
}

void lr_opt::init() {
    llama_common_lr_opt_init_rust(lr0, lr_min, decay_epochs, epochs, &decay_epochs, &scale_epoch);
}

float lr_opt::get_lr(float epoch) const {
    float r = llama_common_lr_opt_get_lr_rust(lr0, lr_min, decay_epochs, scale_epoch, epoch);
    LOG_INF("epoch %.2g lr=%.2g\n", epoch, r);
    return r;
}

bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos) {
    llama_batch batch = llama_batch_get_one(&last_token, 1);
    batch.pos = &pos;
    if (llama_decode(ctx, batch)) {
        LOG_ERR("%s: failed to replay last token\n", __func__);
        return false;
    }
    return true;
}

bool common_prompt_batch_decode(
              struct llama_context * ctx,
    const std::vector<llama_token> & tokens,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state) {
    const int n_eval = tokens.size();
    if (n_eval == 0) {
        return true;
    }

    if (save_state && n_eval > 1) {
        const int n_tokens_before_last = n_eval - 1;

        GGML_ASSERT(n_eval <= n_batch);

        // Decode all but the last token so we can save the memory state before decoding the last token.
        // This is done so we can restore the session state later and replay the last token.
        // Memory implementations in recurrent/hybrid models don't support removing tokens from their
        // memory, so we can't just remove the last token from the memory and replay the last token which
        // is the reason for this logic.
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token*>(tokens.data()), n_tokens_before_last))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        n_past += n_tokens_before_last;

        llama_state_save_file(ctx, state_path.data(), tokens.data(), n_tokens_before_last);
        LOG_INF("saved session before last token to %s, n_tokens = %d\n", state_path.data(), n_tokens_before_last);

        llama_token last_token = tokens.back();
        llama_batch batch = llama_batch_get_one(&last_token, 1);
        int32_t pos = n_past;
        batch.pos = &pos;

        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s : failed to eval last token\n", __func__);
            return false;
        }
        n_past++;
    } else {
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token*>(tokens.data()), n_eval))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return false;
        }
        n_past += n_eval;
    }

    return true;
}

#include "common.h.inc"
#include "common.h.inc"
// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp.inc>

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <fstream>
#include <list>
#include <regex>
#include <set>
#include <string>
#include <thread> // for hardware_concurrency
#include <vector>

extern "C" {
int llama_build_number(void);
const char * llama_commit(void);
const char * llama_compiler(void);
const char * llama_build_target(void);
}

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#   if !defined(PATH_MAX)
#   define PATH_MAX MAX_PATH
#   endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

#define LLAMA_MAX_URL_LENGTH 2084 // Maximum URL Length in Chrome: 2083

extern const char * LICENSES[];

using json = nlohmann::ordered_json;
using namespace common_arg_utils;

static std::initializer_list<enum llama_example> mmproj_examples = {
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CLI,
};

static std::string read_file(const std::string & fname) {
    llama_common_unicode_string content =
            llama_common_read_file_rust((const uint8_t *) fname.data(), fname.size());
    if (!content.data) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
    }
    std::string result((const char *) content.data, content.len);
    llama_common_unicode_string_free(content);
    return result;
}

static const std::vector<common_arg> & get_common_arg_defs() {
    static const std::vector<common_arg> options = [] {
        common_params params;
        auto ctx = common_params_parser_init(params, LLAMA_EXAMPLE_SERVER, nullptr);
        return ctx.options;
    }();
    return options;
}

common_arg & common_arg::set_examples(std::initializer_list<enum llama_example> examples) {
    this->examples = examples;
    return *this;
}

common_arg & common_arg::set_excludes(std::initializer_list<enum llama_example> excludes) {
    this->excludes = excludes;
    return *this;
}

common_arg & common_arg::set_env(const char * env) {
    help = help + "\n(env: " + env + ")";
    this->env = env;
    return *this;
}

common_arg & common_arg::set_sampling() {
    is_sampling = true;
    return *this;
}

common_arg & common_arg::set_spec() {
    is_spec = true;
    return *this;
}

common_arg & common_arg::set_preset_only() {
    is_preset_only = true;
    return *this;
}

bool common_arg::in_example(enum llama_example ex) {
    return examples.find(ex) != examples.end();
}

bool common_arg::is_exclude(enum llama_example ex) {
    return excludes.find(ex) != excludes.end();
}

bool common_arg::get_value_from_env(std::string & output) const {
    if (env == nullptr) return false;
    if (!args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        char * neg_value = std::getenv(neg_env.c_str());
        if (neg_value) {
            output = "0"; // falsey
            return true;
        }
    }
    char * value = std::getenv(env);
    if (value) {
        output = value;
        return true;
    }
    return false;
}

bool common_arg::has_value_from_env() const {
    bool has_neg_env_value = false;
    if (env != nullptr && !args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        has_neg_env_value = std::getenv(neg_env.c_str()) != nullptr;
    }
    return llama_common_arg_has_env_value_rust(
            env != nullptr,
            has_neg_env_value,
            env != nullptr && std::getenv(env) != nullptr);
}

static std::vector<std::string> break_str_into_lines(std::string input, size_t max_char_per_line) {
    common_rust_string_list split =
        llama_common_break_str_into_lines_rust((const uint8_t *) input.data(), input.size(), max_char_per_line);
    std::vector<std::string> result;
    if (!split.data) {
        return result;
    }
    result.reserve(split.len);
    for (size_t i = 0; i < split.len; ++i) {
        const llama_common_unicode_string & item = split.data[i];
        result.emplace_back((const char *) item.data, item.len);
    }
    llama_common_string_list_free(split);
    return result;
}

std::string common_arg::to_string() const {
    // params for printing to console
    const static int n_leading_spaces = 40;
    const static int n_char_per_line_help = 70; // TODO: detect this based on current console
    std::string leading_spaces(n_leading_spaces, ' ');

    std::ostringstream ss;
    auto all_args = get_args(); // also contains args_neg
    for (const auto & arg : all_args) {
        if (arg == all_args.front()) {
            if (all_args.size() == 1) {
                ss << arg;
            } else {
                // first arg is usually abbreviation, we need padding to make it more beautiful
                auto tmp = std::string(arg) + ", ";
                auto spaces = std::string(std::max(0, 7 - (int)tmp.size()), ' ');
                ss << tmp << spaces;
            }
        } else {
            ss << arg << (arg != all_args.back() ? ", " : "");
        }
    }
    if (value_hint) ss << " " << value_hint;
    if (value_hint_2) ss << " " << value_hint_2;
    if (ss.tellp() > n_leading_spaces - 3) {
        // current line is too long, add new line
        ss << "\n" << leading_spaces;
    } else {
        // padding between arg and help, same line
        ss << std::string(leading_spaces.size() - ss.tellp(), ' ');
    }
    const auto help_lines = break_str_into_lines(help, n_char_per_line_help);
    for (const auto & line : help_lines) {
        ss << (&line == &help_lines.front() ? "" : leading_spaces) << line << "\n";
    }
    return ss.str();
}

std::vector<std::string> common_arg::get_args() const {
    std::vector<std::string> result;
    for (const auto & arg : args) {
        result.push_back(std::string(arg));
    }
    for (const auto & arg : args_neg) {
        result.push_back(std::string(arg));
    }
    return result;
}

std::vector<std::string> common_arg::get_env() const {
    std::vector<std::string> result;
    if (env) {
        result.push_back(std::string(env));
    }
    if (!args_neg.empty() && env) {
        // for compatibility, we need to add LLAMA_ARG_NO_ variant
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        result.push_back(neg_env);
    }
    return result;
}

//
// utils
//

// Helper function to parse tensor buffer override strings
static void parse_tensor_buffer_overrides(const std::string & value, std::vector<llama_model_tensor_buft_override> & overrides) {
    std::map<std::string, ggml_backend_buffer_type_t> buft_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            buft_list[ggml_backend_buft_name(buft)] = buft;
        }
    }

    for (const auto & override : string_split<std::string>(value, ',')) {
        std::string::size_type pos = override.find('=');
        if (pos == std::string::npos) {
            throw std::invalid_argument("invalid value");
        }
        std::string tensor_name = override.substr(0, pos);
        std::string buffer_type = override.substr(pos + 1);

        if (buft_list.find(buffer_type) == buft_list.end()) {
            printf("Available buffer types:\n");
            for (const auto & it : buft_list) {
                printf("  %s\n", ggml_backend_buft_name(it.second));
            }
            throw std::invalid_argument("unknown buffer type");
        }
        // keep strings alive and avoid leaking memory by storing them in a static vector
        static std::list<std::string> buft_overrides;
        buft_overrides.push_back(tensor_name);
        overrides.push_back({buft_overrides.back().c_str(), buft_list.at(buffer_type)});
    }
}

static bool common_params_handle_remote_preset(common_params & params, llama_example ex) {
    GGML_ASSERT(!params.model.hf_repo.empty());

    // the returned hf_repo is without tag
    auto [hf_repo, hf_tag] = common_download_split_repo_tag(params.model.hf_repo);

    // "latest" tag (default if not specified) is translated to "default" preset
    if (hf_tag == "latest") {
        hf_tag = "default";
    }

    std::string model_endpoint = common_get_model_endpoint();
    auto preset_url = model_endpoint + hf_repo + "/resolve/main/preset.ini";

    // prepare local path for caching
    const std::string preset_repo = hf_repo + "_preset.ini";
    auto preset_fname = common_rust_string_to_std(
            llama_common_clean_file_name_rust((const uint8_t *) preset_repo.data(), preset_repo.size()));
    auto preset_path = fs_get_cache_file(preset_fname);
    common_download_opts opts;
    opts.bearer_token = params.hf_token;
    opts.offline = params.offline;
    const int status = common_download_file_single(preset_url, preset_path, opts);
    const bool has_preset = status >= 200 && status < 400;

    // remote preset is optional, so we don't error out if not found
    if (has_preset) {
        LOG_INF("applying remote preset from %s\n", preset_url.c_str());
        common_preset_context ctx(ex, /* only_remote_allowed */ true);
        common_preset global;
        auto remote_presets = ctx.load_from_ini(preset_path, global);
        remote_presets = ctx.cascade(global, remote_presets);
        if (remote_presets.find(hf_tag) != remote_presets.end()) {
            common_preset preset = remote_presets.at(hf_tag);
            LOG_INF("\n%s", preset.to_ini().c_str()); // to_ini already added trailing newline
            preset.apply_to_params(params);
        } else {
            throw std::runtime_error("Remote preset.ini does not contain [" + std::string(hf_tag) + "] section");
        }
    } else {
        LOG_INF("%s", "no remote preset found, skipping\n");
    }

    return has_preset;
}

struct handle_model_result {
    bool found_mmproj = false;
    common_params_model mmproj;
};

static handle_model_result common_params_handle_model(struct common_params_model & model,
                                                      const std::string          & bearer_token,
                                                      bool                         offline) {
    handle_model_result result;

    if (!model.docker_repo.empty()) {
        model.path = common_docker_resolve_model(model.docker_repo);
        model.name = model.docker_repo;
    } else if (!model.hf_repo.empty()) {
        // If -m was used with -hf, treat the model "path" as the hf_file to download
        if (model.hf_file.empty() && !model.path.empty()) {
            model.hf_file = model.path;
            model.path = "";
        }
        common_download_opts opts;
        opts.bearer_token = bearer_token;
        opts.offline = offline;
        auto download_result = common_download_model(model, opts, true);

        if (download_result.model_path.empty()) {
            LOG_ERR("error: failed to download model from Hugging Face\n");
            exit(1);
        }

        model.name = model.hf_repo;
        model.path = download_result.model_path;

        if (!download_result.mmproj_path.empty()) {
            result.found_mmproj = true;
            result.mmproj.path  = download_result.mmproj_path;
        }
    } else if (!model.url.empty()) {
        if (model.path.empty()) {
            auto f = string_split<std::string>(model.url, '#').front();
            f = string_split<std::string>(f, '?').front();
            model.path = fs_get_cache_file(string_split<std::string>(f, '/').back());
        }

        common_download_opts opts;
        opts.bearer_token = bearer_token;
        opts.offline = offline;
        auto download_result = common_download_model(model, opts);
        if (download_result.model_path.empty()) {
            LOG_ERR("error: failed to download model from %s\n", model.url.c_str());
            exit(1);
        }
    }

    return result;
}

const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const std::string & s) {
    const int parsed = llama_common_kv_cache_type_from_str_rust(s.c_str());
    if (parsed >= 0) {
        return (ggml_type) parsed;
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

//
// CLI argument parsing functions
//

static bool common_params_parse_ex(int argc, char ** argv, common_params_context & ctx_arg) {
    common_params & params = ctx_arg.params;

    // setup log directly from params.verbosity: see tools/cli/cli.cpp
    common_log_set_verbosity_thold(params.verbosity);

    std::unordered_map<std::string, std::pair<common_arg *, bool>> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = {&opt, /* is_positive */ true};
        }
        for (const auto & arg : opt.args_neg) {
            arg_to_options[arg] = {&opt, /* is_positive */ false};
        }
    }

    // handle environment variables
    for (auto & opt : ctx_arg.options) {
        std::string value;
        if (opt.get_value_from_env(value)) {
            try {
                if (opt.handler_void && is_truthy(value)) {
                    opt.handler_void(params);
                }
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(value));
                }
                if (opt.handler_bool) {
                    const int parsed = llama_common_parse_bool_value_rust(value.c_str());
                    if (parsed < 0) {
                        throw std::invalid_argument("invalid boolean value");
                    }
                    opt.handler_bool(params, parsed != 0);
                }
                if (opt.handler_string) {
                    opt.handler_string(params, value);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(string_format(
                    "error while handling environment variable \"%s\": %s\n\n", opt.env, e.what()));
            }
        }
    }

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    auto parse_cli_args = [&]() {
        std::set<std::string> seen_args;

        for (int i = 1; i < argc; i++) {
            const std::string arg_prefix = "--";

            std::string arg = argv[i];
            if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
                std::replace(arg.begin(), arg.end(), '_', '-');
            }
            if (arg_to_options.find(arg) == arg_to_options.end()) {
                throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
            }
            if (!seen_args.insert(arg).second) {
                LOG_WRN("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\n", arg.c_str());
            }
            auto & tmp = arg_to_options[arg];
            auto opt = *tmp.first;
            bool is_positive = tmp.second;
            if (opt.has_value_from_env()) {
                fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n", opt.env, arg.c_str());
            }
            try {
                if (opt.handler_void) {
                    opt.handler_void(params);
                    continue;
                }
                if (opt.handler_bool) {
                    opt.handler_bool(params, is_positive);
                    continue;
                }

                // arg with single value
                check_arg(i);
                std::string val = argv[++i];
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(val));
                    continue;
                }
                if (opt.handler_string) {
                    opt.handler_string(params, val);
                    continue;
                }

                // arg with 2 values
                check_arg(i);
                std::string val2 = argv[++i];
                if (opt.handler_str_str) {
                    opt.handler_str_str(params, val, val2);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(string_format(
                    "error while handling argument \"%s\": %s\n\n"
                    "usage:\n%s\n\nto show complete usage, run with -h",
                    arg.c_str(), e.what(), opt.to_string().c_str()));
            }
        }
    };

    // parse the first time to get -hf option (used for remote preset)
    parse_cli_args();

    // TODO: Remove later
    try {
        hf_cache::migrate_old_cache_to_hf_cache(params.hf_token, params.offline);
    } catch (const std::exception & e) {
        LOG_WRN("HF cache migration failed: %s\n", e.what());
    }
    // export_graph_ops loads only metadata
    const bool skip_model_download = ctx_arg.ex == LLAMA_EXAMPLE_EXPORT_GRAPH_OPS;

    // maybe handle remote preset
    if (!params.model.hf_repo.empty() && !skip_model_download) {
        std::string cli_hf_repo = params.model.hf_repo;
        bool has_preset = common_params_handle_remote_preset(params, ctx_arg.ex);

        // special case: if hf_repo explicitly set by preset, we need to preserve it (ignore CLI value)
        // this is useful when we have one HF repo pointing to other HF repos (one model - multiple GGUFs)
        std::string preset_hf_repo = params.model.hf_repo;
        bool preset_has_hf_repo = preset_hf_repo != cli_hf_repo;

        if (has_preset) {
            // re-parse CLI args to override preset values
            parse_cli_args();
        }

        // preserve hf_repo from preset if needed
        if (preset_has_hf_repo) {
            params.model.hf_repo = preset_hf_repo;
        }
    }

    postprocess_cpu_params(params.cpuparams,       nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    postprocess_cpu_params(params.speculative.draft.cpuparams,       &params.cpuparams);
    postprocess_cpu_params(params.speculative.draft.cpuparams_batch, &params.cpuparams_batch);

    if (params.prompt_cache_all && (params.interactive || params.interactive_first)) {
        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    // handle model and download
    if (!skip_model_download) {
        auto res = common_params_handle_model(params.model, params.hf_token, params.offline);
        if (params.no_mmproj) {
            params.mmproj = {};
        } else if (res.found_mmproj && params.mmproj.path.empty() && params.mmproj.url.empty()) {
            // optionally, handle mmproj model when -hf is specified
            params.mmproj = res.mmproj;
        }
        // only download mmproj if the current example is using it
        for (const auto & ex : mmproj_examples) {
            if (ctx_arg.ex == ex) {
                common_params_handle_model(params.mmproj,    params.hf_token, params.offline);
                break;
            }
        }
        common_params_handle_model(params.speculative.draft.mparams, params.hf_token, params.offline);
        common_params_handle_model(params.vocoder.model,             params.hf_token, params.offline);
    }

    // model is required (except for server)
    // TODO @ngxson : maybe show a list of available models in CLI in this case
    if (params.model.path.empty() && ctx_arg.ex != LLAMA_EXAMPLE_SERVER && !skip_model_download && !params.usage && !params.completion) {
        throw std::invalid_argument("error: --model is required\n");
    }

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
        for (auto & seq_breaker : params.sampling.dry_sequence_breakers) {
            string_process_escapes(seq_breaker);
        }
        for (auto & pair : params.speculative.draft.replacements) {
            string_process_escapes(pair.first);
            string_process_escapes(pair.second);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    // pad tensor_buft_overrides for llama_params_fit:
    const size_t ntbo = llama_max_tensor_buft_overrides();
    while (params.tensor_buft_overrides.size() < ntbo) {
        params.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (!params.speculative.draft.tensor_buft_overrides.empty()) {
        params.speculative.draft.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (!params.chat_template.empty() && !common_chat_verify_template(params.chat_template, params.use_jinja)) {
        throw std::runtime_error(string_format(
            "error: the supplied chat template is not supported: %s%s\n",
            params.chat_template.c_str(),
            params.use_jinja ? "" : "\nnote: llama.cpp was started without --jinja, we only support commonly used templates"
        ));
    }

    return true;
}

static void common_params_print_usage(common_params_context & ctx_arg) {
    auto print_options = [](std::vector<common_arg *> & options) {
        for (common_arg * opt : options) {
            printf("%s", opt->to_string().c_str());
        }
    };

    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sampling_options;
    std::vector<common_arg *> spec_options;
    std::vector<common_arg *> specific_options;
    for (auto & opt : ctx_arg.options) {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.is_sampling) {
            sampling_options.push_back(&opt);
        } else if (opt.is_spec) {
            spec_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }
    printf("----- common params -----\n\n");
    print_options(common_options);
    printf("\n\n----- sampling params -----\n\n");
    print_options(sampling_options);
    printf("\n\n----- speculative params -----\n\n");
    print_options(spec_options);
    // TODO: maybe convert enum llama_example to string
    printf("\n\n----- example-specific params -----\n\n");
    print_options(specific_options);
}

static void common_params_print_completion(common_params_context & ctx_arg) {
    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sampling_options;
    std::vector<common_arg *> spec_options;
    std::vector<common_arg *> specific_options;

    for (auto & opt : ctx_arg.options) {
        if (opt.is_sampling) {
            sampling_options.push_back(&opt);
        } else if (opt.is_spec) {
            spec_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }

    printf("_llama_completions() {\n");
    printf("    local cur prev opts\n");
    printf("    COMPREPLY=()\n");
    printf("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
    printf("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n\n");

    printf("    opts=\"");
    auto print_options = [](const std::vector<common_arg *> & options) {
        for (const common_arg * opt : options) {
            for (const char * arg : opt->args) {
                printf("%s ", arg);
            }
        }
    };

    print_options(common_options);
    print_options(sampling_options);
    print_options(spec_options);
    print_options(specific_options);
    printf("\"\n\n");

    printf("    case \"$prev\" in\n");
    printf("        --model|-m)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.gguf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        --grammar-file)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.gbnf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        --chat-template-file)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.jinja' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        *)\n");
    printf("            COMPREPLY=( $(compgen -W \"${opts}\" -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("    esac\n");
    printf("}\n\n");

    std::set<std::string> executables = {
        "llama-batched",
        "llama-batched-bench",
        "llama-bench",
        "llama-cli",
        "llama-completion",
        "llama-convert-llama2c-to-ggml",
        "llama-cvector-generator",
        "llama-debug",
        "llama-diffusion-cli",
        "llama-embedding",
        "llama-eval-callback",
        "llama-export-lora",
        "llama-finetune",
        "llama-fit-params",
        "llama-gemma3-cli",
        "llama-gen-docs",
        "llama-gguf",
        "llama-gguf-hash",
        "llama-gguf-split",
        "llama-idle",
        "llama-imatrix",
        "llama-llava-cli",
        "llama-lookahead",
        "llama-lookup",
        "llama-lookup-create",
        "llama-lookup-merge",
        "llama-lookup-stats",
        "llama-minicpmv-cli",
        "llama-mtmd-cli",
        "llama-parallel",
        "llama-passkey",
        "llama-perplexity",
        "llama-q8dot",
        "llama-quantize",
        "llama-qwen2vl-cli",
        "llama-retrieval",
        "llama-save-load-state",
        "llama-server",
        "llama-simple",
        "llama-simple-chat",
        "llama-speculative",
        "llama-speculative-simple",
        "llama-tokenize",
        "llama-tts",
        "llama-vdot"
    };

    for (const auto& exe : executables) {
        printf("complete -F _llama_completions %s\n", exe.c_str());
    }
}

static std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

static void add_rpc_devices(const std::string & servers) {
    auto rpc_servers = string_split<std::string>(servers, ',');
    if (rpc_servers.empty()) {
        throw std::invalid_argument("no RPC servers specified");
    }
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        throw std::invalid_argument("failed to find RPC backend");
    }
    typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char * endpoint);
    ggml_backend_rpc_add_server_t ggml_backend_rpc_add_server_fn = (ggml_backend_rpc_add_server_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!ggml_backend_rpc_add_server_fn) {
        throw std::invalid_argument("failed to find RPC add server function");
    }
    for (const auto & server : rpc_servers) {
        auto reg = ggml_backend_rpc_add_server_fn(server.c_str());
        ggml_backend_register(reg);
    }
}

bool common_params_to_map(int argc, char ** argv, llama_example ex, std::map<common_arg, std::string> & out_map) {
    common_params dummy_params;
    common_params_context ctx_arg = common_params_parser_init(dummy_params, ex, nullptr);

    std::unordered_map<std::string, common_arg *> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = &opt;
        }
        for (const auto & arg : opt.args_neg) {
            arg_to_options[arg] = &opt;
        }
    }

    // TODO @ngxson : find a way to deduplicate this code

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    std::set<std::string> seen_args;

    for (int i = 1; i < argc; i++) {
        const std::string arg_prefix = "--";

        std::string arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (arg_to_options.find(arg) == arg_to_options.end()) {
            throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
        }
        if (!seen_args.insert(arg).second) {
            LOG_WRN("DEPRECATED: argument '%s' specified multiple times, use comma-separated values instead (only last value will be used)\n", arg.c_str());
        }
        auto opt = *arg_to_options[arg];
        std::string val;
        if (opt.value_hint == nullptr && opt.value_hint_2 == nullptr) {
            // bool arg (need to reverse the meaning for negative args)
            bool is_neg = std::find(opt.args_neg.begin(), opt.args_neg.end(), arg) != opt.args_neg.end();
            val = is_neg ? "0" : "1";
        }
        if (opt.value_hint != nullptr) {
            // arg with single value
            check_arg(i);
            val = argv[++i];
        }
        if (opt.value_hint_2 != nullptr) {
            // TODO: support arg with 2 values
            throw std::invalid_argument("error: argument with 2 values is not yet supported\n");
        }
        out_map[opt] = val;
    }

    return true;
}

bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    auto ctx_arg = common_params_parser_init(params, ex, print_usage);
    const common_params params_org = ctx_arg.params; // the example can modify the default params

    try {
        if (!common_params_parse_ex(argc, argv, ctx_arg)) {
            ctx_arg.params = params_org;
            return false;
        }
        if (ctx_arg.params.usage) {
            common_params_print_usage(ctx_arg);
            if (ctx_arg.print_usage) {
                ctx_arg.print_usage(argc, argv);
            }
            exit(0);
        }
        if (ctx_arg.params.completion) {
            common_params_print_completion(ctx_arg);
            exit(0);
        }
        params.lr.init();
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        ctx_arg.params = params_org;
        return false;
    } catch (std::exception & ex) {
        fprintf(stderr, "%s\n", ex.what());
        exit(1); // for other exceptions, we exit with status code 1
    }

    return true;
}

static std::string list_builtin_chat_templates() {
    std::vector<const char *> supported_tmpl;
    int32_t res = llama_chat_builtin_templates(nullptr, 0);
    supported_tmpl.resize(res);
    res = llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size());
    std::ostringstream msg;
    for (auto & tmpl : supported_tmpl) {
        msg << tmpl << (&tmpl == &supported_tmpl.back() ? "" : ", ");
    }
    return msg.str();
}

bool common_arg_utils::is_truthy(const std::string & value) {
    return llama_common_arg_is_truthy_rust(value.c_str()) != 0;
}

bool common_arg_utils::is_falsey(const std::string & value) {
    return llama_common_arg_is_falsey_rust(value.c_str()) != 0;
}

bool common_arg_utils::is_autoy(const std::string & value) {
    return llama_common_arg_is_autoy_rust(value.c_str()) != 0;
}

// Simple CSV parser that handles quoted fields and escaped quotes
// example:
//    input:  value1,"value, with, commas","value with ""escaped"" quotes",value4
//    output: [value1] [value, with, commas] [value with "escaped" quotes] [value4]
static std::vector<std::string> parse_csv_row(const std::string& input) {
    common_rust_string_list parsed = llama_common_parse_csv_row_rust((const uint8_t *) input.data(), input.size());
    std::vector<std::string> fields;
    if (!parsed.data) {
        return fields;
    }
    fields.reserve(parsed.len);
    for (size_t i = 0; i < parsed.len; ++i) {
        const llama_common_unicode_string & item = parsed.data[i];
        fields.emplace_back((const char *) item.data, item.len);
    }
    llama_common_string_list_free(parsed);
    return fields;
}

common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    // per-example default params
    // we define here to make sure it's included in llama-gen-docs
    if (ex == LLAMA_EXAMPLE_COMPLETION) {
        params.use_jinja = false;   // disable jinja by default

    } else if (ex == LLAMA_EXAMPLE_MTMD) {
        params.use_jinja = false;   // disable jinja by default
        params.sampling.temp = 0.2; // lower temp by default for better quality

    } else if (ex == LLAMA_EXAMPLE_SERVER) {
        params.n_parallel = -1;     // auto by default
    }

    params.use_color = tty_can_use_colors();

    // load dynamic backends
    ggml_backend_load_all();

    common_params_context ctx_arg(params);
    ctx_arg.print_usage = print_usage;
    ctx_arg.ex          = ex;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto & sampler : params.sampling.samplers) {
        sampler_type_chars += common_sampler_type_to_chr(sampler);
        sampler_type_names += common_sampler_type_to_str(sampler) + ";";
    }
    if (!sampler_type_names.empty()) {
        sampler_type_names.pop_back(); // remove last semicolon
    }


    /**
     * filter options by example
     * rules:
     * - all examples inherit options from LLAMA_EXAMPLE_COMMON
     * - if LLAMA_EXAMPLE_* is set (other than COMMON), we only show the option in the corresponding example
     * - if both {LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_*,} are set, we will prioritize the LLAMA_EXAMPLE_* matching current example
     */
    auto add_opt = [&](common_arg arg) {
        if ((arg.in_example(ex) || arg.in_example(LLAMA_EXAMPLE_COMMON)) && !arg.is_exclude(ex)) {
            ctx_arg.options.push_back(std::move(arg));
        }
    };


    add_opt(common_arg(
        {"-h", "--help", "--usage"},
        "print usage and exit",
        [](common_params & params) {
            params.usage = true;
        }
    ));
    add_opt(common_arg(
        {"--version"},
        "show version and build info",
        [](common_params &) {
            fprintf(stderr, "version: %d (%s)\n", llama_build_number(), llama_commit());
            fprintf(stderr, "built with %s for %s\n", llama_compiler(), llama_build_target());
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"--license"},
        "show source code license and dependencies",
        [](common_params &) {
            for (int i = 0; LICENSES[i]; ++i) {
                printf("%s\n", LICENSES[i]);
            }
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"-cl", "--cache-list"},
        "show list of models in cache",
        [](common_params &) {
            auto models = common_list_cached_models();
            printf("number of models in cache: %zu\n", models.size());
            for (size_t i = 0; i < models.size(); i++) {
                printf("%4zu. %s\n", i + 1, models[i].to_string().c_str());
            }
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"--completion-bash"},
        "print source-able bash completion script for llama.cpp",
        [](common_params & params) {
            params.completion = true;
        }
    ));
    add_opt(common_arg(
        {"--verbose-prompt"},
        string_format("print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false"),
        [](common_params & params) {
            params.verbose_prompt = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--display-prompt"},
        {"--no-display-prompt"},
        string_format("whether to print prompt at generation (default: %s)", params.display_prompt ? "true" : "false"),
        [](common_params & params, bool value) {
            params.display_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-co", "--color"}, "[on|off|auto]",
        "Colorize output to distinguish prompt and user input from generations ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.use_color = true;
            } else if (is_falsey(value)) {
                params.use_color = false;
            } else if (is_autoy(value)) {
                params.use_color = tty_can_use_colors();
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --color: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP}));
    add_opt(common_arg(
        {"-t", "--threads"}, "N",
        string_format("number of CPU threads to use during generation (default: %d)", params.cpuparams.n_threads),
        [](common_params & params, int value) {
            params.cpuparams.n_threads = value;
            if (params.cpuparams.n_threads <= 0) {
                params.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_env("LLAMA_ARG_THREADS"));
    add_opt(common_arg(
        {"-tb", "--threads-batch"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads)",
        [](common_params & params, int value) {
            params.cpuparams_batch.n_threads = value;
            if (params.cpuparams_batch.n_threads <= 0) {
                params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ));
    add_opt(common_arg(
        {"-C", "--cpu-mask"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
        [](common_params & params, const std::string & mask) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Cr", "--cpu-range"}, "lo-hi",
        "range of CPUs for affinity. Complements --cpu-mask",
        [](common_params & params, const std::string & range) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict"}, "<0|1>",
        string_format("use strict CPU placement (default: %u)\n", (unsigned) params.cpuparams.strict_cpu),
        [](common_params & params, const std::string & value) {
            params.cpuparams.strict_cpu = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"--prio"}, "N",
        string_format("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\n", params.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < GGML_SCHED_PRIO_LOW || prio > GGML_SCHED_PRIO_REALTIME) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll"}, "<0...100>",
        string_format("use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) params.cpuparams.poll),
        [](common_params & params, const std::string & value) {
            params.cpuparams.poll = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"-Cb", "--cpu-mask-batch"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Crb", "--cpu-range-batch"}, "lo-hi",
        "ranges of CPUs for affinity. Complements --cpu-mask-batch",
        [](common_params & params, const std::string & range) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict-batch"}, "<0|1>",
        "use strict CPU placement (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.cpuparams_batch.strict_cpu = value;
        }
    ));
    add_opt(common_arg(
        {"--prio-batch"}, "N",
        string_format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll-batch"}, "<0|1>",
        "use polling to wait for work (default: same as --poll)",
        [](common_params & params, int value) {
            params.cpuparams_batch.poll = value;
        }
    ));
    add_opt(common_arg(
        {"-lcs", "--lookup-cache-static"}, "FNAME",
        "path to static lookup cache to use for lookup decoding (not updated by generation)",
        [](common_params & params, const std::string & value) {
            params.speculative.ngram_cache.lookup_cache_static = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-lcd", "--lookup-cache-dynamic"}, "FNAME",
        "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
        [](common_params & params, const std::string & value) {
            params.speculative.ngram_cache.lookup_cache_dynamic = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-c", "--ctx-size"}, "N",
        string_format("size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx),
        [](common_params & params, int value) {
            params.n_ctx = value;
            if (value == 0) {
                // disable context reduction in llama_params_fit if the user explicitly requests the full context size:
                params.fit_params_min_ctx = UINT32_MAX;
            }
        }
    ).set_env("LLAMA_ARG_CTX_SIZE"));
    add_opt(common_arg(
        {"-n", "--predict", "--n-predict"}, "N",
        string_format(
            ex == LLAMA_EXAMPLE_COMPLETION
                ? "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)"
                : "number of tokens to predict (default: %d, -1 = infinity)",
            params.n_predict),
        [](common_params & params, int value) {
            params.n_predict = value;
        }
    ).set_env("LLAMA_ARG_N_PREDICT"));
    add_opt(common_arg(
        {"-b", "--batch-size"}, "N",
        string_format("logical maximum batch size (default: %d)", params.n_batch),
        [](common_params & params, int value) {
            params.n_batch = value;
        }
    ).set_env("LLAMA_ARG_BATCH"));
    add_opt(common_arg(
        {"-ub", "--ubatch-size"}, "N",
        string_format("physical maximum batch size (default: %d)", params.n_ubatch),
        [](common_params & params, int value) {
            params.n_ubatch = value;
        }
    ).set_env("LLAMA_ARG_UBATCH"));
    add_opt(common_arg(
        {"--keep"}, "N",
        string_format("number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep),
        [](common_params & params, int value) {
            params.n_keep = value;
        }
    ));
    add_opt(common_arg(
        {"--swa-full"},
        string_format("use full-size SWA cache (default: %s)\n"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)", params.swa_full ? "true" : "false"),
        [](common_params & params) {
            params.swa_full = true;
        }
    ).set_env("LLAMA_ARG_SWA_FULL"));
    add_opt(common_arg(
        {"-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"}, "N",
        string_format("max number of context checkpoints to create per slot (default: %d)"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/15293)", params.n_ctx_checkpoints),
        [](common_params & params, int value) {
            params.n_ctx_checkpoints = value;
        }
    ).set_env("LLAMA_ARG_CTX_CHECKPOINTS").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-cpent", "--checkpoint-every-n-tokens"}, "N",
        string_format("create a checkpoint every n tokens during prefill (processing), -1 to disable (default: %d)", params.checkpoint_every_nt),
        [](common_params & params, int value) {
            params.checkpoint_every_nt = value;
        }
    ).set_env("LLAMA_ARG_CHECKPOINT_EVERY_NT").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-cram", "--cache-ram"}, "N",
        string_format("set the maximum cache size in MiB (default: %d, -1 - no limit, 0 - disable)"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/16391)", params.cache_ram_mib),
        [](common_params & params, int value) {
            params.cache_ram_mib = value;
        }
    ).set_env("LLAMA_ARG_CACHE_RAM").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-kvu", "--kv-unified"},
        {"-no-kvu", "--no-kv-unified"},
        "use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)",
        [](common_params & params, bool value) {
            params.kv_unified = value;
        }
    ).set_env("LLAMA_ARG_KV_UNIFIED").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_BATCHED, LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"--cache-idle-slots"},
        {"--no-cache-idle-slots"},
        "save and clear idle slots on new task (default: enabled, requires unified KV and cache-ram)",
        [](common_params & params, bool value) {
            params.cache_idle_slots = value;
        }
    ).set_env("LLAMA_ARG_CACHE_IDLE_SLOTS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--context-shift"},
        {"--no-context-shift"},
        string_format("whether to use context shift on infinite text generation (default: %s)", params.ctx_shift ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.ctx_shift = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY}).set_env("LLAMA_ARG_CONTEXT_SHIFT"));
    add_opt(common_arg(
        {"--chunks"}, "N",
        string_format("max number of chunks to process (default: %d, -1 = all)", params.n_chunks),
        [](common_params & params, int value) {
            params.n_chunks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg({ "-fa", "--flash-attn" }, "[on|off|auto]",
                       string_format("set Flash Attention use ('on', 'off', or 'auto', default: '%s')",
                                     llama_flash_attn_type_name(params.flash_attn_type)),
                       [](common_params & params, const std::string & value) {
                           if (is_truthy(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                           } else if (is_falsey(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                           } else if (is_autoy(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
                           } else {
                               throw std::runtime_error(
                                   string_format("error: unknown value for --flash-attn: '%s'\n", value.c_str()));
                           }
                       }).set_env("LLAMA_ARG_FLASH_ATTN"));
    add_opt(common_arg(
        {"-p", "--prompt"}, "PROMPT",
        "prompt to start generation with; for system message, use -sys",
        [](common_params & params, const std::string & value) {
            params.prompt = value;
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sys", "--system-prompt"}, "PROMPT",
        "system prompt to use with model (if applicable, depending on chat template)",
        [](common_params & params, const std::string & value) {
            params.system_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DIFFUSION, LLAMA_EXAMPLE_MTMD}));
    add_opt(common_arg(
        {"--perf"},
        {"--no-perf"},
        string_format("whether to enable internal libllama performance timings (default: %s)", params.no_perf ? "true" : "false"),
        [](common_params & params, bool value) {
            params.no_perf = !value;
            params.sampling.no_perf = !value;
        }
    ).set_env("LLAMA_ARG_PERF"));
    add_opt(common_arg(
        {"--show-timings"},
        {"--no-show-timings"},
        string_format("whether to show timing information after each response (default: %s)", params.show_timings ? "true" : "false"),
        [](common_params & params, bool value) {
            params.show_timings = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SHOW_TIMINGS"));
    add_opt(common_arg(
        {"-f", "--file"}, "FNAME",
        "a file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            params.prompt = read_file(value);
            // store the external file name in params
            params.prompt_file = value;
            if (!params.prompt.empty() && params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sysf", "--system-prompt-file"}, "FNAME",
        "a file containing the system prompt (default: none)",
        [](common_params & params, const std::string & value) {
            params.system_prompt = read_file(value);
            if (!params.system_prompt.empty() && params.system_prompt.back() == '\n') {
                params.system_prompt.pop_back();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DIFFUSION}));
    add_opt(common_arg(
        {"--in-file"}, "FNAME",
        "an input file (use comma-separated values to specify multiple files)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                std::ifstream file(item);
                if (!file) {
                    throw std::runtime_error(string_format("error: failed to open file '%s'\n", item.c_str()));
                }
                params.in_files.push_back(item);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-bf", "--binary-file"}, "FNAME",
        "binary file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::ostringstream ss;
            ss << file.rdbuf();
            params.prompt = ss.str();
            fprintf(stderr, "Read %zu bytes from binary file %s\n", params.prompt.size(), value.c_str());
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-e", "--escape"},
        {"--no-escape"},
        string_format("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", params.escape ? "true" : "false"),
        [](common_params & params, bool value) {
            params.escape = value;
        }
    ));
    add_opt(common_arg(
        {"-ptc", "--print-token-count"}, "N",
        string_format("print token count every N tokens (default: %d)", params.n_print),
        [](common_params & params, int value) {
            params.n_print = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache"}, "FNAME",
        "file to cache prompt state for faster startup (default: none)",
        [](common_params & params, const std::string & value) {
            params.path_prompt_cache = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache-all"},
        "if specified, saves user input and generations to cache as well\n",
        [](common_params & params) {
            params.prompt_cache_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache-ro"},
        "if specified, uses the prompt cache but does not update it",
        [](common_params & params) {
            params.prompt_cache_ro = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-r", "--reverse-prompt"}, "PROMPT",
        "halt generation at PROMPT, return control in interactive mode\n",
        [](common_params & params, const std::string & value) {
            params.antiprompt.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sp", "--special"},
        string_format("special tokens output enabled (default: %s)", params.special ? "true" : "false"),
        [](common_params & params) {
            params.special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-cnv", "--conversation"},
        {"-no-cnv", "--no-conversation"},
        "whether to run in conversation mode:\n"
        "- does not print special tokens and suffix/prefix\n"
        "- interactive mode is also enabled\n"
        "(default: auto enabled if chat template is available)",
        [](common_params & params, bool value) {
            params.conversation_mode = value ? COMMON_CONVERSATION_MODE_ENABLED : COMMON_CONVERSATION_MODE_DISABLED;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-st", "--single-turn"},
        "run conversation for a single turn only, then exit when done\n"
        "will not be interactive if first turn is predefined with --prompt\n"
        "(default: false)",
        [](common_params & params) {
            params.single_turn = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-i", "--interactive"},
        string_format("run in interactive mode (default: %s)", params.interactive ? "true" : "false"),
        [](common_params & params) {
            params.interactive = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-if", "--interactive-first"},
        string_format("run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false"),
        [](common_params & params) {
            params.interactive_first = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-mli", "--multiline-input"},
        "allows you to write or paste multiple lines without ending each in '\\'",
        [](common_params & params) {
            params.multiline_input = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--in-prefix-bos"},
        "prefix BOS to user inputs, preceding the `--in-prefix` string",
        [](common_params & params) {
            params.input_prefix_bos = true;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--in-prefix"}, "STRING",
        "string to prefix user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_prefix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--in-suffix"}, "STRING",
        "string to suffix after user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_suffix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--warmup"},
        {"--no-warmup"},
        string_format("whether to perform warmup with an empty run (default: %s)", params.warmup ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.warmup = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--spm-infill"},
        string_format(
            "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)",
            params.spm_infill ? "enabled" : "disabled"
        ),
        [](common_params & params) {
            params.spm_infill = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--samplers"}, "SAMPLERS",
        string_format("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)", sampler_type_names.c_str()),
        [](common_params & params, const std::string & value) {
            const auto sampler_names = string_split<std::string>(value, ';');
            params.sampling.samplers = common_sampler_types_from_names(sampler_names, true);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-s", "--seed"}, "SEED",
        string_format("RNG seed (default: %d, use random seed for %d)", params.sampling.seed, LLAMA_DEFAULT_SEED),
        [](common_params & params, const std::string & value) {
            params.sampling.seed = std::stoul(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--sampler-seq", "--sampling-seq"}, "SEQUENCE",
        string_format("simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str()),
        [](common_params & params, const std::string & value) {
            params.sampling.samplers = common_sampler_types_from_chars(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--ignore-eos"},
        "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
        [](common_params & params) {
            params.sampling.ignore_eos = true;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--temp", "--temperature"}, "N",
        string_format("temperature (default: %.2f)", (double)params.sampling.temp),
        [](common_params & params, const std::string & value) {
            params.sampling.temp = std::stof(value);
            params.sampling.temp = std::max(params.sampling.temp, 0.0f);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--top-k"}, "N",
        string_format("top-k sampling (default: %d, 0 = disabled)", params.sampling.top_k),
        [](common_params & params, int value) {
            params.sampling.top_k = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
        }
    ).set_sampling().set_env("LLAMA_ARG_TOP_K"));
    add_opt(common_arg(
        {"--top-p"}, "N",
        string_format("top-p sampling (default: %.2f, 1.0 = disabled)", (double)params.sampling.top_p),
        [](common_params & params, const std::string & value) {
            params.sampling.top_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--min-p"}, "N",
        string_format("min-p sampling (default: %.2f, 0.0 = disabled)", (double)params.sampling.min_p),
        [](common_params & params, const std::string & value) {
            params.sampling.min_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--top-nsigma", "--top-n-sigma"}, "N",
        string_format("top-n-sigma sampling (default: %.2f, -1.0 = disabled)", params.sampling.top_n_sigma),
        [](common_params & params, const std::string & value) {
            params.sampling.top_n_sigma = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--xtc-probability"}, "N",
        string_format("xtc probability (default: %.2f, 0.0 = disabled)", (double)params.sampling.xtc_probability),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_probability = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--xtc-threshold"}, "N",
        string_format("xtc threshold (default: %.2f, 1.0 = disabled)", (double)params.sampling.xtc_threshold),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_threshold = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--typical", "--typical-p"}, "N",
        string_format("locally typical sampling, parameter p (default: %.2f, 1.0 = disabled)", (double)params.sampling.typ_p),
        [](common_params & params, const std::string & value) {
            params.sampling.typ_p = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-last-n"}, "N",
        string_format("last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", params.sampling.penalty_last_n),
        [](common_params & params, int value) {
            if (value < -1) {
                throw std::runtime_error(string_format("error: invalid repeat-last-n = %d\n", value));
            }
            params.sampling.penalty_last_n = value;
            params.sampling.n_prev = std::max(params.sampling.n_prev, params.sampling.penalty_last_n);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--repeat-penalty"}, "N",
        string_format("penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)", (double)params.sampling.penalty_repeat),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_repeat = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--presence-penalty"}, "N",
        string_format("repeat alpha presence penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty_present),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_present = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--frequency-penalty"}, "N",
        string_format("repeat alpha frequency penalty (default: %.2f, 0.0 = disabled)", (double)params.sampling.penalty_freq),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_freq = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-multiplier"}, "N",
        string_format("set DRY sampling multiplier (default: %.2f, 0.0 = disabled)", (double)params.sampling.dry_multiplier),
        [](common_params & params, const std::string & value) {
            params.sampling.dry_multiplier = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-base"}, "N",
        string_format("set DRY sampling base value (default: %.2f)", (double)params.sampling.dry_base),
        [](common_params & params, const std::string & value) {
            float potential_base = std::stof(value);
            if (potential_base >= 1.0f)
            {
                params.sampling.dry_base = potential_base;
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-allowed-length"}, "N",
        string_format("set allowed length for DRY sampling (default: %d)", params.sampling.dry_allowed_length),
        [](common_params & params, int value) {
            params.sampling.dry_allowed_length = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-penalty-last-n"}, "N",
        string_format("set DRY penalty for the last n tokens (default: %d, 0 = disable, -1 = context size)", params.sampling.dry_penalty_last_n),
        [](common_params & params, int value) {
            if (value < -1) {
                throw std::runtime_error(string_format("error: invalid dry-penalty-last-n = %d\n", value));
            }
            params.sampling.dry_penalty_last_n = value;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dry-sequence-breaker"}, "STRING",
        string_format("add sequence breaker for DRY sampling, clearing out default breakers (%s) in the process; use \"none\" to not use any sequence breakers\n",
            params.sampling.dry_sequence_breakers.empty() ? "none" :
            std::accumulate(std::next(params.sampling.dry_sequence_breakers.begin()),
                params.sampling.dry_sequence_breakers.end(),
                std::string("'") + (params.sampling.dry_sequence_breakers[0] == "\n" ? "\\n" : params.sampling.dry_sequence_breakers[0]) + "'",
                [](const std::string& a, const std::string& b) {
                    std::string formatted_b = (b == "\n") ? "\\n" : b;
                    return a + ", '" + formatted_b + "'";
                }).c_str()),
        [](common_params & params, const std::string & value) {
            static bool defaults_cleared = false;

            if (!defaults_cleared) {
                params.sampling.dry_sequence_breakers.clear();
                defaults_cleared = true;
            }

            if (value == "none") {
                params.sampling.dry_sequence_breakers.clear();
            } else {
                params.sampling.dry_sequence_breakers.emplace_back(value);
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-target"}, "N",
        string_format("adaptive-p: select tokens near this probability (valid range 0.0 "
                      "to 1.0; negative = disabled) (default: %.2f)\n"
                      "[(more info)](https://github.com/ggml-org/llama.cpp/pull/17927)",
                      (double)params.sampling.adaptive_target),
        [](common_params & params, const std::string & value) {
            params.sampling.adaptive_target = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--adaptive-decay"}, "N",
        string_format("adaptive-p: decay rate for target adaptation over time. lower values "
                      "are more reactive, higher values are more stable.\n"
                      "(valid range 0.0 to 0.99) (default: %.2f)",
                      (double)params.sampling.adaptive_decay),
        [](common_params & params, const std::string & value) {
            params.sampling.adaptive_decay = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dynatemp-range"}, "N",
        string_format("dynamic temperature range (default: %.2f, 0.0 = disabled)", (double)params.sampling.dynatemp_range),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_range = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--dynatemp-exp"}, "N",
        string_format("dynamic temperature exponent (default: %.2f)", (double)params.sampling.dynatemp_exponent),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_exponent = std::stof(value);
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat"}, "N",
        string_format("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n"
        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sampling.mirostat),
        [](common_params & params, int value) {
            params.sampling.mirostat = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat-lr"}, "N",
        string_format("Mirostat learning rate, parameter eta (default: %.2f)", (double)params.sampling.mirostat_eta),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_eta = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--mirostat-ent"}, "N",
        string_format("Mirostat target entropy, parameter tau (default: %.2f)", (double)params.sampling.mirostat_tau),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_tau = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU;
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
        "modifies the likelihood of token appearing in the completion,\n"
        "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
        "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
        [](common_params & params, const std::string & value) {
            std::stringstream ss(value);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                    params.sampling.logit_bias.push_back({key, bias});
                } else {
                    throw std::invalid_argument("invalid input format");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument("invalid input format");
            }
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--grammar"}, "GRAMMAR",
        "BNF-like grammar to constrain generations (see samples in grammars/ dir)",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, value};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"--grammar-file"}, "FNAME",
        "file to read grammar from",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_USER, read_file(value)};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-j", "--json-schema"}, "SCHEMA",
        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\nFor schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(json::parse(value))};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-jf", "--json-schema-file"}, "FILE",
        "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\nFor schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string schema;
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(schema)
            );
            params.sampling.grammar = {COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, json_schema_to_grammar(json::parse(schema))};
        }
    ).set_sampling());
    add_opt(common_arg(
        {"-bs", "--backend-sampling"},
        "enable backend sampling (experimental) (default: disabled)",
        [](common_params & params) {
            params.sampling.backend_sampling = true;
        }
    ).set_sampling().set_env("LLAMA_ARG_BACKEND_SAMPLING"));
    add_opt(common_arg(
        {"--pooling"}, "{none,mean,cls,last,rank}",
        "pooling type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
            else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
            else if (value == "cls")  { params.pooling_type = LLAMA_POOLING_TYPE_CLS;  }
            else if (value == "last") { params.pooling_type = LLAMA_POOLING_TYPE_LAST; }
            else if (value == "rank") { params.pooling_type = LLAMA_POOLING_TYPE_RANK; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_DEBUG}).set_env("LLAMA_ARG_POOLING"));
    add_opt(common_arg(
        {"--attention"}, "{causal,non-causal}",
        "attention type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "causal") { params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; }
            else if (value == "non-causal") { params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--rope-scaling"}, "{none,linear,yarn}",
        "RoPE frequency scaling method, defaults to linear unless specified by the model",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
            else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_ROPE_SCALING_TYPE"));
    add_opt(common_arg(
        {"--rope-scale"}, "N",
        "RoPE context scaling factor, expands context by a factor of N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = 1.0f / std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_SCALE"));
    add_opt(common_arg(
        {"--rope-freq-base"}, "N",
        "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
        [](common_params & params, const std::string & value) {
            params.rope_freq_base = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_BASE"));
    add_opt(common_arg(
        {"--rope-freq-scale"}, "N",
        "RoPE frequency scaling factor, expands context by a factor of 1/N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_SCALE"));
    add_opt(common_arg(
        {"--yarn-orig-ctx"}, "N",
        string_format("YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx),
        [](common_params & params, int value) {
            params.yarn_orig_ctx = value;
        }
    ).set_env("LLAMA_ARG_YARN_ORIG_CTX"));
    add_opt(common_arg(
        {"--yarn-ext-factor"}, "N",
        string_format("YaRN: extrapolation mix factor (default: %.2f, 0.0 = full interpolation)", (double)params.yarn_ext_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_ext_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_EXT_FACTOR"));
    add_opt(common_arg(
        {"--yarn-attn-factor"}, "N",
        string_format("YaRN: scale sqrt(t) or attention magnitude (default: %.2f)", (double)params.yarn_attn_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_attn_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_ATTN_FACTOR"));
    add_opt(common_arg(
        {"--yarn-beta-slow"}, "N",
        string_format("YaRN: high correction dim or alpha (default: %.2f)", (double)params.yarn_beta_slow),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_slow = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_SLOW"));
    add_opt(common_arg(
        {"--yarn-beta-fast"}, "N",
        string_format("YaRN: low correction dim or beta (default: %.2f)", (double)params.yarn_beta_fast),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_fast = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_FAST"));
    add_opt(common_arg(
        {"-gan", "--grp-attn-n"}, "N",
        string_format("group-attention factor (default: %d)", params.grp_attn_n),
        [](common_params & params, int value) {
            params.grp_attn_n = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_N").set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-gaw", "--grp-attn-w"}, "N",
        string_format("group-attention width (default: %d)", params.grp_attn_w),
        [](common_params & params, int value) {
            params.grp_attn_w = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_W").set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-kvo", "--kv-offload"},
        {"-nkvo", "--no-kv-offload"},
        string_format("whether to enable KV cache offloading (default: %s)", params.no_kv_offload ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_kv_offload = !value;
        }
    ).set_env("LLAMA_ARG_KV_OFFLOAD"));
    add_opt(common_arg(
        {"--repack"},
        {"-nr", "--no-repack"},
        string_format("whether to enable weight repacking (default: %s)", params.no_extra_bufts ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_extra_bufts = !value;
        }
    ).set_env("LLAMA_ARG_REPACK"));
    add_opt(common_arg(
        {"--no-host"},
        "bypass host buffer allowing extra buffers to be used",
        [](common_params & params) {
            params.no_host = true;
        }
    ).set_env("LLAMA_ARG_NO_HOST"));
    add_opt(common_arg(
        {"-ctk", "--cache-type-k"}, "TYPE",
        string_format(
            "KV cache data type for K\n"
            "allowed values: %s\n"
            "(default: %s)",
            llama_common_get_all_kv_cache_types_rust(),
            ggml_type_name(params.cache_type_k)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_k = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_K"));
    add_opt(common_arg(
        {"-ctv", "--cache-type-v"}, "TYPE",
        string_format(
            "KV cache data type for V\n"
            "allowed values: %s\n"
            "(default: %s)",
            llama_common_get_all_kv_cache_types_rust(),
            ggml_type_name(params.cache_type_v)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_v = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_V"));
    add_opt(common_arg(
        {"--hellaswag"},
        "compute HellaSwag score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.hellaswag = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--hellaswag-tasks"}, "N",
        string_format("number of tasks to use when computing the HellaSwag score (default: %zu)", params.hellaswag_tasks),
        [](common_params & params, int value) {
            params.hellaswag_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande"},
        "compute Winogrande score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.winogrande = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande-tasks"}, "N",
        string_format("number of tasks to use when computing the Winogrande score (default: %zu)", params.winogrande_tasks),
        [](common_params & params, int value) {
            params.winogrande_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice"},
        "compute multiple choice score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.multiple_choice = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice-tasks"}, "N",
        string_format("number of tasks to use when computing the multiple choice score (default: %zu)", params.multiple_choice_tasks),
        [](common_params & params, int value) {
            params.multiple_choice_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--kl-divergence"},
        "computes KL-divergence to logits provided via --kl-divergence-base",
        [](common_params & params) {
            params.kl_divergence = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--save-all-logits", "--kl-divergence-base"}, "FNAME",
        "set logits file",
        [](common_params & params, const std::string & value) {
            params.logits_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--ppl-stride"}, "N",
        string_format("stride for perplexity calculation (default: %d)", params.ppl_stride),
        [](common_params & params, int value) {
            params.ppl_stride = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--ppl-output-type"}, "<0|1>",
        string_format("output type for perplexity calculation (default: %d)", params.ppl_output_type),
        [](common_params & params, int value) {
            params.ppl_output_type = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"-dt", "--defrag-thold"}, "N",
        string_format("KV cache defragmentation threshold (DEPRECATED)"),
        [](common_params & params, const std::string & value) {
            GGML_UNUSED(params);
            GGML_UNUSED(value);
            LOG_WRN("DEPRECATED: --defrag-thold is deprecated and no longer necessary to specify\n");
        }
    ).set_env("LLAMA_ARG_DEFRAG_THOLD"));
    if (ex == LLAMA_EXAMPLE_SERVER) {
        // this is to make sure this option appears in the server-specific section of the help message
        add_opt(common_arg(
            {"-np", "--parallel"}, "N",
            string_format("number of server slots (default: %d, -1 = auto)", params.n_parallel),
            [](common_params & params, int value) {
                if (value == 0) {
                    throw std::invalid_argument("error: invalid value for n_parallel\n");
                }
                params.n_parallel = value;
            }
        ).set_env("LLAMA_ARG_N_PARALLEL").set_examples({LLAMA_EXAMPLE_SERVER}));
    } else {
        add_opt(common_arg(
            {"-np", "--parallel"}, "N",
            string_format("number of parallel sequences to decode (default: %d)", params.n_parallel),
            [](common_params & params, int value) {
                params.n_parallel = value;
            }
        ).set_env("LLAMA_ARG_N_PARALLEL"));
    }
    add_opt(common_arg(
        {"-ns", "--sequences"}, "N",
        string_format("number of sequences to decode (default: %d)", params.n_sequences),
        [](common_params & params, int value) {
            params.n_sequences = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-cb", "--cont-batching"},
        {"-nocb", "--no-cont-batching"},
        string_format("whether to enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cont_batching = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONT_BATCHING"));
    add_opt(common_arg(
        {"-mm", "--mmproj"}, "FILE",
        "path to a multimodal projector file. see tools/mtmd/README.md\n"
        "note: if -hf is used, this argument can be omitted",
        [](common_params & params, const std::string & value) {
            params.mmproj.path = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ"));
    add_opt(common_arg(
        {"-mmu", "--mmproj-url"}, "URL",
        "URL to a multimodal projector file. see tools/mtmd/README.md",
        [](common_params & params, const std::string & value) {
            params.mmproj.url = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_URL"));
    add_opt(common_arg(
        {"--mmproj-auto"},
        {"--no-mmproj", "--no-mmproj-auto"},
        string_format("whether to use multimodal projector file (if available), useful when using -hf (default: %s)", params.no_mmproj ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_mmproj = !value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_AUTO"));
    add_opt(common_arg(
        {"--mmproj-offload"},
        {"--no-mmproj-offload"},
        string_format("whether to enable GPU offloading for multimodal projector (default: %s)", params.mmproj_use_gpu ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.mmproj_use_gpu = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_OFFLOAD"));
    add_opt(common_arg(
        {"--image", "--audio"}, "FILE",
        "path to an image or audio file. use with multimodal models, use comma-separated values for multiple files\n",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.image.emplace_back(item);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--image-min-tokens"}, "N",
        "minimum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
        [](common_params & params, int value) {
            params.image_min_tokens = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_IMAGE_MIN_TOKENS"));
    add_opt(common_arg(
        {"--image-max-tokens"}, "N",
        "maximum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
        [](common_params & params, int value) {
            params.image_max_tokens = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_IMAGE_MAX_TOKENS"));
    if (llama_supports_rpc()) {
        add_opt(common_arg(
            {"--rpc"}, "SERVERS",
            "comma separated list of RPC servers (host:port)",
            [](common_params & params, const std::string & value) {
                add_rpc_devices(value);
                GGML_UNUSED(params);
            }
        ).set_env("LLAMA_ARG_RPC"));
    }
    add_opt(common_arg(
        {"--mlock"},
        "force system to keep model in RAM rather than swapping or compressing",
        [](common_params & params) {
            params.use_mlock = true;
        }
    ).set_env("LLAMA_ARG_MLOCK"));
    add_opt(common_arg(
        {"--mmap"},
        {"--no-mmap"},
        string_format("whether to memory-map model. (if mmap disabled, slower load but may reduce pageouts if not using mlock) (default: %s)", params.use_mmap ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.use_mmap = value;
        }
    ).set_env("LLAMA_ARG_MMAP"));
    add_opt(common_arg(
        {"-dio", "--direct-io"},
        {"-ndio", "--no-direct-io"},
        string_format("use DirectIO if available. (default: %s)", params.use_direct_io ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.use_direct_io = value;
        }
    ).set_env("LLAMA_ARG_DIO"));
    add_opt(common_arg(
        {"--numa"}, "TYPE",
        "attempt optimizations that help on some NUMA systems\n"
        "- distribute: spread execution evenly over all nodes\n"
        "- isolate: only spawn threads on CPUs on the node that execution started on\n"
        "- numactl: use the CPU map provided by numactl\n"
        "if run without this previously, it is recommended to drop the system page cache before using this\n"
        "see https://github.com/ggml-org/llama.cpp/issues/1437",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
            else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_NUMA"));
    add_opt(common_arg(
        {"-dev", "--device"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading (none = don't offload)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            params.devices = parse_device_list(value);
        }
    ).set_env("LLAMA_ARG_DEVICE"));
    add_opt(common_arg(
        {"--list-devices"},
        "print list of available devices and exit",
        [](common_params &) {
            std::vector<ggml_backend_dev_t> devices;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                    devices.push_back(dev);
                }
            }
            printf("Available devices:\n");
            for (auto * dev : devices) {
                size_t free, total;
                ggml_backend_dev_memory(dev, &free, &total);
                printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
            }
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"-ot", "--override-tensor"}, "<tensor name pattern>=<buffer type>,...",
        "override tensor buffer type", [](common_params & params, const std::string & value) {
            parse_tensor_buffer_overrides(value, params.tensor_buft_overrides);
        }
    ).set_env("LLAMA_ARG_OVERRIDE_TENSOR"));
    add_opt(common_arg(
        {"-cmoe", "--cpu-moe"},
        "keep all Mixture of Experts (MoE) weights in the CPU",
        [](common_params & params) {
            params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        }
    ).set_env("LLAMA_ARG_CPU_MOE"));
    add_opt(common_arg(
        {"-ncmoe", "--n-cpu-moe"}, "N",
        "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            for (int i = 0; i < value; ++i) {
                // keep strings alive and avoid leaking memory by storing them in a static vector
                static std::list<std::string> buft_overrides;
                buft_overrides.push_back(llm_ffn_exps_block_regex(i));
                params.tensor_buft_overrides.push_back({buft_overrides.back().c_str(), ggml_backend_cpu_buffer_type()});
            }
        }
    ).set_env("LLAMA_ARG_N_CPU_MOE"));
    GGML_ASSERT(params.n_gpu_layers < 0); // string_format would need to be extended for a default >= 0
    add_opt(common_arg(
        {"-ngl", "--gpu-layers", "--n-gpu-layers"}, "N",
        string_format("max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)", params.n_gpu_layers == -1 ? "auto" : "all"),
        [](common_params & params, const std::string & value) {
            if (value == "auto") {
                params.n_gpu_layers = -1;
            } else if (value == "all") {
                params.n_gpu_layers = -2;
            } else {
                params.n_gpu_layers = std::stoi(value);
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_env("LLAMA_ARG_N_GPU_LAYERS"));
    add_opt(common_arg(
        {"-sm", "--split-mode"}, "{none,layer,row,tensor}",
        "how to split the model across multiple GPUs, one of:\n"
        "- none: use one GPU only\n"
        "- layer (default): split layers and KV across GPUs (pipelined)\n"
        "- row: split weight across GPUs by rows (parallelized)\n"
        "- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)",
        [](common_params & params, const std::string & value) {
            if (value == "none") {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            } else if (value == "layer") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            } else if (value == "row") {
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else if (value == "tensor") {
                params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
            } else {
                throw std::invalid_argument("invalid value");
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_SPLIT_MODE"));
    add_opt(common_arg(
        {"-ts", "--tensor-split"}, "N0,N1,N2,...",
        "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    string_format("got %zu input configs, but system only has %zu devices", split_arg.size(), llama_max_devices())
                );
            }
            for (size_t i = 0; i < llama_max_devices(); ++i) {
                if (i < split_arg.size()) {
                    params.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params.tensor_split[i] = 0.0f;
                }
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_TENSOR_SPLIT"));
    add_opt(common_arg(
        {"-mg", "--main-gpu"}, "INDEX",
        string_format("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu),
        [](common_params & params, int value) {
            params.main_gpu = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_MAIN_GPU"));
    add_opt(common_arg(
        { "-fit", "--fit" }, "[on|off]",
        string_format("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '%s')", params.fit_params ? "on" : "off"),
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.fit_params = true;
            } else if (is_falsey(value)) {
                params.fit_params = false;
            } else {
                throw std::runtime_error(
                    string_format("error: unknown value for --fit: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_ARG_FIT"));
    add_opt(common_arg(
        { "-fitp", "--fit-print" }, "[on|off]",
        string_format("print the estimated required memory ('on' or 'off', default: '%s')", params.fit_params_print ? "on" : "off"),
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.fit_params_print = true;
            } else if (is_falsey(value)) {
                params.fit_params_print = false;
            } else {
                throw std::runtime_error(
                    string_format("error: unknown value for --fit-print: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_FIT_PARAMS}).set_env("LLAMA_ARG_FIT_ESTIMATE"));
    add_opt(common_arg(
        { "-fitt", "--fit-target" }, "MiB0,MiB1,MiB2,...",
        string_format("target margin per device for --fit, comma-separated list of values, "
            "single value is broadcast across all devices, default: %zu", params.fit_params_target[0]/(1024*1024)),
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    string_format("got %zu input configs, but system only has %zu devices", split_arg.size(), llama_max_devices())
                );
            }
            if (split_arg.size() == 1) {
                std::fill(params.fit_params_target.begin(), params.fit_params_target.end(), std::stoull(split_arg[0]) * 1024*1024);
                return;
            }
            for (size_t i = 0; i < split_arg.size(); i++) {
                params.fit_params_target[i] = std::stoull(split_arg[i]) * 1024*1024;
            }
        }
    ).set_env("LLAMA_ARG_FIT_TARGET"));
    add_opt(common_arg(
        { "-fitc", "--fit-ctx" }, "N",
        string_format("minimum ctx size that can be set by --fit option, default: %" PRIu32, params.fit_params_min_ctx),
        [](common_params & params, int value) {
            params.fit_params_min_ctx = value;
        }
    ).set_env("LLAMA_ARG_FIT_CTX"));
    add_opt(common_arg(
        {"--check-tensors"},
        string_format("check model tensor data for invalid values (default: %s)", params.check_tensors ? "true" : "false"),
        [](common_params & params) {
            params.check_tensors = true;
        }
    ));
    add_opt(common_arg(
        {"--override-kv"}, "KEY=TYPE:VALUE,...",
        "advanced option to override model metadata by key. to specify multiple overrides, either use comma-separated values.\n"
        "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false,tokenizer.ggml.add_eos_token=bool:false",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                if (!string_parse_kv_override(item.c_str(), params.kv_overrides)) {
                    throw std::runtime_error(string_format("error: Invalid type for KV override: %s\n", item.c_str()));
                }
            }
        }
    ));
    add_opt(common_arg(
        {"--op-offload"},
        {"--no-op-offload"},
        string_format("whether to offload host tensor operations to device (default: %s)", params.no_op_offload ? "false" : "true"),
        [](common_params & params, bool value) {
            params.no_op_offload = !value;
        }
    ));
    add_opt(common_arg(
        {"--lora"}, "FNAME",
        "path to LoRA adapter (use comma-separated values to load multiple adapters)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.lora_adapters.push_back({ item, 1.0, "", "", nullptr });
            }
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--lora-scaled"}, "FNAME:SCALE,...",
        "path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\n"
        "note: use comma-separated values",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                auto parts = string_split<std::string>(item, ':');
                if (parts.size() != 2) {
                    throw std::invalid_argument("lora-scaled format: FNAME:SCALE");
                }
                params.lora_adapters.push_back({ parts[0], std::stof(parts[1]), "", "", nullptr });
            }
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--control-vector"}, "FNAME",
        "add a control vector\nnote: use comma-separated values to add multiple control vectors",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                params.control_vectors.push_back({ 1.0f, item, });
            }
        }
    ));
    add_opt(common_arg(
        {"--control-vector-scaled"}, "FNAME:SCALE,...",
        "add a control vector with user defined scaling SCALE\n"
        "note: use comma-separated values (format: FNAME:SCALE,...)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                auto parts = string_split<std::string>(item, ':');
                if (parts.size() != 2) {
                    throw std::invalid_argument("control-vector-scaled format: FNAME:SCALE");
                }
                params.control_vectors.push_back({ std::stof(parts[1]), parts[0] });
            }
        }
    ));
    add_opt(common_arg(
        {"--control-vector-layer-range"}, "START", "END",
        "layer range to apply the control vector(s) to, start and end inclusive",
        [](common_params & params, const std::string & start, const std::string & end) {
            params.control_vector_layer_start = std::stoi(start);
            params.control_vector_layer_end = std::stoi(end);
        }
    ));
    add_opt(common_arg(
        {"-a", "--alias"}, "STRING",
        "set model name aliases, comma-separated (to be used by API)",
        [](common_params & params, const std::string & value) {
            for (auto & alias : string_split<std::string>(value, ',')) {
                alias = string_strip(alias);
                if (!alias.empty()) {
                    params.model_alias.insert(alias);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ALIAS"));
    add_opt(common_arg(
        {"--tags"}, "STRING",
        "set model tags, comma-separated (informational, not used for routing)",
        [](common_params & params, const std::string & value) {
            for (auto & tag : string_split<std::string>(value, ',')) {
                tag = string_strip(tag);
                if (!tag.empty()) {
                    params.model_tags.insert(tag);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TAGS"));
    add_opt(common_arg(
        {"-m", "--model"}, "FNAME",
        ex == LLAMA_EXAMPLE_EXPORT_LORA
            ? "model path from which to load base model"
            : "model path to load",
        [](common_params & params, const std::string & value) {
            params.model.path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}).set_env("LLAMA_ARG_MODEL"));
    add_opt(common_arg(
        {"-mu", "--model-url"}, "MODEL_URL",
        "model download url (default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.url = value;
        }
    ).set_env("LLAMA_ARG_MODEL_URL"));
    add_opt(common_arg(
        { "-dr", "--docker-repo" }, "[<repo>/]<model>[:quant]",
        "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\n"
        "example: gemma3\n"
        "(default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.docker_repo = value;
        }
    ).set_env("LLAMA_ARG_DOCKER_REPO"));
    add_opt(common_arg(
        {"-hf", "-hfr", "--hf-repo"}, "<user>/<model>[:quant]",
        "Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.\n"
        "mmproj is also downloaded automatically if available. to disable, add --no-mmproj\n"
        "example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M\n"
        "(default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.hf_repo = value;
        }
    ).set_env("LLAMA_ARG_HF_REPO"));
    add_opt(common_arg(
        {"-hff", "--hf-file"}, "FILE",
        "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.hf_file = value;
        }
    ).set_env("LLAMA_ARG_HF_FILE"));
    add_opt(common_arg(
        {"-hfv", "-hfrv", "--hf-repo-v"}, "<user>/<model>[:quant]",
        "Hugging Face model repository for the vocoder model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.vocoder.model.hf_repo = value;
        }
    ).set_env("LLAMA_ARG_HF_REPO_V"));
    add_opt(common_arg(
        {"-hffv", "--hf-file-v"}, "FILE",
        "Hugging Face model file for the vocoder model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.vocoder.model.hf_file = value;
        }
    ).set_env("LLAMA_ARG_HF_FILE_V"));
    add_opt(common_arg(
        {"-hft", "--hf-token"}, "TOKEN",
        "Hugging Face access token (default: value from HF_TOKEN environment variable)",
        [](common_params & params, const std::string & value) {
            params.hf_token = value;
        }
    ).set_env("HF_TOKEN"));
    add_opt(common_arg(
        {"--context-file"}, "FNAME",
        "file to load context from (use comma-separated values to specify multiple files)",
        [](common_params & params, const std::string & value) {
            for (const auto & item : parse_csv_row(value)) {
                std::ifstream file(item, std::ios::binary);
                if (!file) {
                    throw std::runtime_error(string_format("error: failed to open file '%s'\n", item.c_str()));
                }
                params.context_files.push_back(item);
            }
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-size"}, "N",
        string_format("minimum length of embedded text chunks (default: %d)", params.chunk_size),
        [](common_params & params, int value) {
            params.chunk_size = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-separator"}, "STRING",
        string_format("separator between chunks (default: '%s')", params.chunk_separator.c_str()),
        [](common_params & params, const std::string & value) {
            params.chunk_separator = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--junk"}, "N",
        string_format("number of times to repeat the junk text (default: %d)", params.n_junk),
        [](common_params & params, int value) {
            params.n_junk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"--pos"}, "N",
        string_format("position of the passkey in the junk text (default: %d)", params.i_pos),
        [](common_params & params, int value) {
            params.i_pos = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-o", "--output", "--output-file"}, "FNAME",
        string_format("output file (default: '%s')", params.out_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.out_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_CVECTOR_GENERATOR, LLAMA_EXAMPLE_EXPORT_LORA, LLAMA_EXAMPLE_TTS, LLAMA_EXAMPLE_FINETUNE,
                    LLAMA_EXAMPLE_RESULTS, LLAMA_EXAMPLE_EXPORT_GRAPH_OPS}));
    add_opt(common_arg(
        {"-ofreq", "--output-frequency"}, "N",
        string_format("output the imatrix every N iterations (default: %d)", params.n_out_freq),
        [](common_params & params, int value) {
            params.n_out_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--output-format"}, "{gguf,dat}",
        string_format("output format for imatrix file (default: %s)", params.imat_dat > 0 ? "dat" : "gguf"),
        [](common_params & params, const std::string & value) {
            /**/ if (value == "gguf") { params.imat_dat = -1; }
            else if (value == "dat")  { params.imat_dat = 1;  }
            else { throw std::invalid_argument("invalid output format"); }
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--save-frequency"}, "N",
        string_format("save an imatrix copy every N iterations (default: %d)", params.n_save_freq),
        [](common_params & params, int value) {
            params.n_save_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--process-output"},
        string_format("collect data for the output tensor (default: %s)", params.process_output ? "true" : "false"),
        [](common_params & params) {
            params.process_output = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--ppl"},
        {"--no-ppl"},
        string_format("whether to compute perplexity (default: %s)", params.compute_ppl ? "true" : "false"),
        [](common_params & params, bool value) {
            params.compute_ppl = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--chunk", "--from-chunk"}, "N",
        string_format("start processing the input from chunk N (default: %d)", params.i_chunk),
        [](common_params & params, int value) {
            params.i_chunk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--show-statistics"},
        string_format("show imatrix statistics and then exit (default: %s)", params.show_statistics ? "true" : "false"),
        [](common_params & params) {
            params.show_statistics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--parse-special"},
        string_format("parse special tokens (chat, tool, etc) (default: %s)", params.parse_special ? "true" : "false"),
        [](common_params & params) {
            params.parse_special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-pps"},
        string_format("is the prompt shared across parallel sequences (default: %s)", params.is_pp_shared ? "true" : "false"),
        [](common_params & params) {
            params.is_pp_shared = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-tgs"},
        string_format("is the text generation separated across the different sequences (default: %s)", params.is_tg_separate ? "true" : "false"),
        [](common_params & params) {
            params.is_tg_separate = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-npp"}, "n0,n1,...",
        "number of prompt tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pp.insert(params.n_pp.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-ntg"}, "n0,n1,...",
        "number of text generation tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_tg.insert(params.n_tg.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-npl"}, "n0,n1,...",
        "number of parallel prompts",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pl.insert(params.n_pl.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"--embd-normalize"}, "N",
        string_format("normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", params.embd_normalize),
        [](common_params & params, int value) {
            params.embd_normalize = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--embd-output-format"}, "FORMAT",
        "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix, \"raw\" = plain whitespace-delimited output (one embedding per line)",
        [](common_params & params, const std::string & value) {
            params.embd_out = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--embd-separator"}, "STRING",
        "separator of embeddings (default \\n) for example \"<#sep#>\"",
        [](common_params & params, const std::string & value) {
            params.embd_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--cls-separator"}, "STRING",
        "separator of classification sequences (default \\t) for example \"<#seq#>\"",
        [](common_params & params, const std::string & value) {
            params.cls_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--host"}, "HOST",
        string_format("ip address to listen, or bind to an UNIX socket if the address ends with .sock (default: %s)", params.hostname.c_str()),
        [](common_params & params, const std::string & value) {
            params.hostname = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_HOST"));
    add_opt(common_arg(
        {"--port"}, "PORT",
        string_format("port to listen (default: %d)", params.port),
        [](common_params & params, int value) {
            params.port = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PORT"));
    add_opt(common_arg(
        {"--reuse-port"},
        string_format("allow multiple sockets to bind to the same port (default: %s)", params.reuse_port ? "enabled" : "disabled"),
        [](common_params & params) {
            params.reuse_port = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_REUSE_PORT"));
    add_opt(common_arg(
        {"--path"}, "PATH",
        string_format("path to serve static files from (default: %s)", params.public_path.c_str()),
        [](common_params & params, const std::string & value) {
            params.public_path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_STATIC_PATH"));
    add_opt(common_arg(
        {"--api-prefix"}, "PREFIX",
        string_format("prefix path the server serves from, without the trailing slash (default: %s)", params.api_prefix.c_str()),
        [](common_params & params, const std::string & value) {
            params.api_prefix = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_API_PREFIX"));
    add_opt(common_arg(
        {"--webui-config"}, "JSON",
        "JSON that provides default WebUI settings (overrides WebUI defaults)",
        [](common_params & params, const std::string & value) {
            params.webui_config_json = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_WEBUI_CONFIG"));
    add_opt(common_arg(
        {"--webui-config-file"}, "PATH",
        "JSON file that provides default WebUI settings (overrides WebUI defaults)",
        [](common_params & params, const std::string & value) {
            params.webui_config_json = read_file(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_WEBUI_CONFIG_FILE"));
    add_opt(common_arg(
        {"--webui-mcp-proxy"},
        {"--no-webui-mcp-proxy"},
        string_format("experimental: whether to enable MCP CORS proxy - do not enable in untrusted environments (default: %s)", params.webui_mcp_proxy ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.webui_mcp_proxy = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_WEBUI_MCP_PROXY"));
    add_opt(common_arg(
        {"--tools"}, "TOOL1,TOOL2,...",
        "experimental: whether to enable built-in tools for AI agents - do not enable in untrusted environments (default: no tools)\n"
        "specify \"all\" to enable all tools\n"
        "available tools: read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, apply_diff",
        [](common_params & params, const std::string & value) {
            params.server_tools = parse_csv_row(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TOOLS"));
    add_opt(common_arg(
        {"--webui"},
        {"--no-webui"},
        string_format("whether to enable the Web UI (default: %s)", params.webui ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.webui = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_WEBUI"));
    add_opt(common_arg(
        {"--embedding", "--embeddings"},
        string_format("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", params.embedding ? "enabled" : "disabled"),
        [](common_params & params) {
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_DEBUG}).set_env("LLAMA_ARG_EMBEDDINGS"));
    add_opt(common_arg(
        {"--rerank", "--reranking"},
        string_format("enable reranking endpoint on server (default: %s)", "disabled"),
        [](common_params & params) {
            params.embedding = true;
            params.pooling_type = LLAMA_POOLING_TYPE_RANK;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_RERANKING"));
    add_opt(common_arg(
        {"--api-key"}, "KEY",
        "API key to use for authentication, multiple keys can be provided as a comma-separated list (default: none)",
        [](common_params & params, const std::string & value) {
            for (const auto & key : parse_csv_row(value)) {
                if (!key.empty()) {
                    params.api_keys.push_back(key);
                }
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_API_KEY"));
    add_opt(common_arg(
        {"--api-key-file"}, "FNAME",
        "path to file containing API keys (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream key_file(value);
            if (!key_file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string key;
            while (std::getline(key_file, key)) {
                if (!key.empty()) {
                    params.api_keys.push_back(key);
                }
            }
            key_file.close();
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--ssl-key-file"}, "FNAME",
        "path to file a PEM-encoded SSL private key",
        [](common_params & params, const std::string & value) {
            params.ssl_file_key = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_KEY_FILE"));
    add_opt(common_arg(
        {"--ssl-cert-file"}, "FNAME",
        "path to file a PEM-encoded SSL certificate",
        [](common_params & params, const std::string & value) {
            params.ssl_file_cert = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_CERT_FILE"));
    add_opt(common_arg(
        {"--chat-template-kwargs"}, "STRING",
        "sets additional params for the json template parser, must be a valid json object string, e.g. '{\"key1\":\"value1\",\"key2\":\"value2\"}'",
        [](common_params & params, const std::string & value) {
            auto parsed = json::parse(value);
            for (const auto & item : parsed.items()) {
                if (item.key() == "enable_thinking") {
                    LOG_WRN("Setting 'enable_thinking' via --chat-template-kwargs is deprecated. "
                            "Use --reasoning on / --reasoning off instead.\n");
                }
                params.default_template_kwargs[item.key()] = item.value().dump();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_CHAT_TEMPLATE_KWARGS"));
    add_opt(common_arg(
        {"-to", "--timeout"}, "N",
        string_format("server read/write timeout in seconds (default: %d)", params.timeout_read),
        [](common_params & params, int value) {
            params.timeout_read  = value;
            params.timeout_write = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TIMEOUT"));
    add_opt(common_arg(
        {"--threads-http"}, "N",
        string_format("number of threads used to process HTTP requests (default: %d)", params.n_threads_http),
        [](common_params & params, int value) {
            params.n_threads_http = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_THREADS_HTTP"));
    add_opt(common_arg(
        {"--cache-prompt"},
        {"--no-cache-prompt"},
        string_format("whether to enable prompt caching (default: %s)", params.cache_prompt ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cache_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_PROMPT"));
    add_opt(common_arg(
        {"--cache-reuse"}, "N",
        string_format(
            "min chunk size to attempt reusing from the cache via KV shifting, requires prompt caching to be enabled (default: %d)\n"
            "[(card)](https://ggml.ai/f0.png)", params.n_cache_reuse
        ),
        [](common_params & params, int value) {
            params.n_cache_reuse = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_REUSE"));
    add_opt(common_arg(
        {"--metrics"},
        string_format("enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_metrics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_METRICS"));
    add_opt(common_arg(
        {"--props"},
        string_format("enable changing global properties via POST /props (default: %s)", params.endpoint_props ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_props = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_PROPS"));
    add_opt(common_arg(
        {"--slots"},
        {"--no-slots"},
        string_format("expose slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.endpoint_slots = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_SLOTS"));
    add_opt(common_arg(
        {"--slot-save-path"}, "PATH",
        "path to save slot kv cache (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.slot_save_path = value;
            if (!fs_is_directory(params.slot_save_path)) {
                throw std::invalid_argument("not a directory: " + value);
            }
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.slot_save_path.empty() && params.slot_save_path[params.slot_save_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.slot_save_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--media-path"}, "PATH",
        "directory for loading local media files; files can be accessed via file:// URLs using relative paths (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.media_path = value;
            if (!fs_is_directory(params.media_path)) {
                throw std::invalid_argument("not a directory: " + value);
            }
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.media_path.empty() && params.media_path[params.media_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.media_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--models-dir"}, "PATH",
        "directory containing models for the router server (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.models_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_DIR"));
    add_opt(common_arg(
        {"--models-preset"}, "PATH",
        "path to INI file containing model presets for the router server (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.models_preset = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_PRESET"));
    add_opt(common_arg(
        {"--models-max"}, "N",
        string_format("for router server, maximum number of models to load simultaneously (default: %d, 0 = unlimited)", params.models_max),
        [](common_params & params, int value) {
            params.models_max = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_MAX"));
    add_opt(common_arg(
        {"--models-autoload"},
        {"--no-models-autoload"},
        string_format("for router server, whether to automatically load models (default: %s)", params.models_autoload ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.models_autoload = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_AUTOLOAD"));
    add_opt(common_arg(
        {"--jinja"},
        {"--no-jinja"},
        string_format("whether to use jinja template engine for chat (default: %s)", params.use_jinja ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.use_jinja = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_MTMD}).set_env("LLAMA_ARG_JINJA"));
    add_opt(common_arg(
        {"--reasoning-format"}, "FORMAT",
        "controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:\n"
        "- none: leaves thoughts unparsed in `message.content`\n"
        "- deepseek: puts thoughts in `message.reasoning_content`\n"
        "- deepseek-legacy: keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`\n"
        "(default: auto)",
        [](common_params & params, const std::string & value) {
            params.reasoning_format = common_reasoning_format_from_name(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK"));
    add_opt(common_arg(
        {"-rea", "--reasoning"}, "[on|off|auto]",
        "Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))",
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.enable_reasoning = 1;
                params.default_template_kwargs["enable_thinking"] = "true";
            } else if (is_falsey(value)) {
                params.enable_reasoning = 0;
                params.default_template_kwargs["enable_thinking"] = "false";
            } else if (is_autoy(value)) {
                params.enable_reasoning = -1;
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --reasoning: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_REASONING"));
    add_opt(common_arg(
        {"--reasoning-budget"}, "N",
        "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)",
        [](common_params & params, int value) {
            if (value < -1) { throw std::invalid_argument("invalid value"); }
            params.sampling.reasoning_budget_tokens = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK_BUDGET"));
    add_opt(common_arg(
        {"--reasoning-budget-message"}, "MESSAGE",
        "message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)",
        [](common_params & params, const std::string & value) {
            params.sampling.reasoning_budget_message = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK_BUDGET_MESSAGE"));
    add_opt(common_arg(
        {"--chat-template"}, "JINJA_TEMPLATE",
        string_format(
            "set custom jinja chat template (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD}).set_env("LLAMA_ARG_CHAT_TEMPLATE"));
    add_opt(common_arg(
        {"--chat-template-file"}, "JINJA_TEMPLATE_FILE",
        string_format(
            "set custom jinja chat template file (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = read_file(value);
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CHAT_TEMPLATE_FILE"));
    add_opt(common_arg(
        {"--skip-chat-parsing"},
        {"--no-skip-chat-parsing"},
        string_format(
            "force a pure content parser, even if a Jinja template is specified; model will output everything "
            "in the content section, including any reasoning and/or tool calls (default: disabled)"
        ),
        [](common_params & params, bool value) {
            params.force_pure_content_parser = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SKIP_CHAT_PARSING"));
    add_opt(common_arg(
        {"--prefill-assistant"},
        {"--no-prefill-assistant"},
        string_format(
            "whether to prefill the assistant's response if the last message is an assistant message (default: prefill enabled)\n"
            "when this flag is set, if the last message is an assistant message then it will be treated as a full message and not prefilled\n"
        ),
        [](common_params & params, bool value) {
            params.prefill_assistant = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PREFILL_ASSISTANT"));
    add_opt(common_arg(
        {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
        string_format("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity),
        [](common_params & params, const std::string & value) {
            params.slot_prompt_similarity = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--lora-init-without-apply"},
        string_format("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora_init_without_apply ? "enabled" : "disabled"),
        [](common_params & params) {
            params.lora_init_without_apply = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--sleep-idle-seconds"}, "SECONDS",
        string_format("number of seconds of idleness after which the server will sleep (default: %d; -1 = disabled)", params.sleep_idle_seconds),
        [](common_params & params, int value) {
            if (value == 0 || value < -1) {
                throw std::invalid_argument("invalid value: cannot be 0 or less than -1");
            }
            params.sleep_idle_seconds = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--simple-io"},
        "use basic IO for better compatibility in subprocesses and limited consoles",
        [](common_params & params) {
            params.simple_io = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--positive-file"}, "FNAME",
        string_format("positive prompts file, one prompt per line (default: '%s')", params.cvector_positive_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_positive_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--negative-file"}, "FNAME",
        string_format("negative prompts file, one prompt per line (default: '%s')", params.cvector_negative_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_negative_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-batch"}, "N",
        string_format("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", params.n_pca_batch),
        [](common_params & params, int value) {
            params.n_pca_batch = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-iter"}, "N",
        string_format("number of iterations used for PCA (default: %d)", params.n_pca_iterations),
        [](common_params & params, int value) {
            params.n_pca_iterations = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--method"}, "{pca, mean}",
        "dimensionality reduction method to be used (default: pca)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "pca") { params.cvector_dimre_method = DIMRE_METHOD_PCA; }
            else if (value == "mean") { params.cvector_dimre_method = DIMRE_METHOD_MEAN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--output-format"}, "{md,jsonl}",
        "output format for batched-bench results (default: md)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "jsonl") { params.batched_bench_output_jsonl = true; }
            else if (value == "md") { params.batched_bench_output_jsonl = false; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"--log-disable"},
        "Log disable",
        [](common_params &) {
            common_log_pause(common_log_main());
        }
    ));
    add_opt(common_arg(
        {"--log-file"}, "FNAME",
        "Log to file",
        [](common_params &, const std::string & value) {
            common_log_set_file(common_log_main(), value.c_str());
        }
    ).set_env("LLAMA_LOG_FILE"));
    add_opt(common_arg(
        {"--log-colors"}, "[on|off|auto]",
        "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params &, const std::string & value) {
            if (is_truthy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_ENABLED);
            } else if (is_falsey(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_DISABLED);
            } else if (is_autoy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_AUTO);
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --log-colors: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_LOG_COLORS"));
    add_opt(common_arg(
        {"-v", "--verbose", "--log-verbose"},
        "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
        [](common_params & params) {
            params.verbosity = INT_MAX;
            common_log_set_verbosity_thold(INT_MAX);
        }
    ));
    add_opt(common_arg(
        {"--offline"},
        "Offline mode: forces use of cache, prevents network access",
        [](common_params & params) {
            params.offline = true;
        }
    ).set_env("LLAMA_OFFLINE"));
    add_opt(common_arg(
        {"-lv", "--verbosity", "--log-verbosity"}, "N",
        string_format("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n"
            " - 0: generic output\n"
            " - 1: error\n"
            " - 2: warning\n"
            " - 3: info\n"
            " - 4: debug\n"
            "(default: %d)\n", params.verbosity),
        [](common_params & params, int value) {
            params.verbosity = value;
            common_log_set_verbosity_thold(value);
        }
    ).set_env("LLAMA_LOG_VERBOSITY"));
    add_opt(common_arg(
        {"--log-prefix"},
        "Enable prefix in log messages",
        [](common_params &) {
            common_log_set_prefix(common_log_main(), true);
        }
    ).set_env("LLAMA_LOG_PREFIX"));
    add_opt(common_arg(
        {"--log-timestamps"},
        "Enable timestamps in log messages",
        [](common_params &) {
            common_log_set_timestamps(common_log_main(), true);
        }
    ).set_env("LLAMA_LOG_TIMESTAMPS"));

    //
    // speculative parameters
    //

    add_opt(common_arg(
        {"--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"}, "<user>/<model>[:quant]",
        "Same as --hf-repo, but for the draft model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.draft.mparams.hf_repo = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_HF_REPO"));
    add_opt(common_arg(
        {"--spec-draft-threads", "-td", "--threads-draft"}, "N",
        "number of threads to use during generation (default: same as --threads)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams.n_threads = value;
            if (params.speculative.draft.cpuparams.n_threads <= 0) {
                params.speculative.draft.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-threads-batch", "-tbd", "--threads-batch-draft"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads-draft)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams_batch.n_threads = value;
            if (params.speculative.draft.cpuparams_batch.n_threads <= 0) {
                params.speculative.draft.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-mask", "-Cd", "--cpu-mask-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.draft.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.draft.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-range", "-Crd", "--cpu-range-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
        [](common_params & params, const std::string & range) {
            params.speculative.draft.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.draft.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-strict", "--cpu-strict-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams.strict_cpu = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-prio", "--prio-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.draft.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.draft.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-poll", "--poll-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: same as --poll])",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams.poll = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-mask-batch", "-Cbd", "--cpu-mask-batch-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.draft.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.draft.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-range-batch", "-Crbd", "--cpu-range-batch-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)",
        [](common_params & params, const std::string & range) {
            params.speculative.draft.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.draft.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--spec-draft-cpu-strict-batch", "--cpu-strict-batch-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: --cpu-strict-draft)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams_batch.strict_cpu = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-prio-batch", "--prio-batch-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.draft.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.draft.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-poll-batch", "--poll-batch-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: --poll-draft)",
        [](common_params & params, int value) {
            params.speculative.draft.cpuparams_batch.poll = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"}, "TYPE",
        string_format(
            "KV cache data type for K for the draft model\n"
            "allowed values: %s\n"
            "(default: %s)",
            llama_common_get_all_kv_cache_types_rust(),
            ggml_type_name(params.speculative.draft.cache_type_k)
        ),
        [](common_params & params, const std::string & value) {
            params.speculative.draft.cache_type_k = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K"));
    add_opt(common_arg(
        {"--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"}, "TYPE",
        string_format(
            "KV cache data type for V for the draft model\n"
            "allowed values: %s\n"
            "(default: %s)",
            llama_common_get_all_kv_cache_types_rust(),
            ggml_type_name(params.speculative.draft.cache_type_v)
        ),
        [](common_params & params, const std::string & value) {
            params.speculative.draft.cache_type_v = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V"));
    add_opt(common_arg(
        {"--spec-draft-override-tensor", "-otd", "--override-tensor-draft"}, "<tensor name pattern>=<buffer type>,...",
        "override tensor buffer type for draft model", [](common_params & params, const std::string & value) {
            parse_tensor_buffer_overrides(value, params.speculative.draft.tensor_buft_overrides);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-draft-cpu-moe", "-cmoed", "--cpu-moe-draft"},
        "keep all Mixture of Experts (MoE) weights in the CPU for the draft model",
        [](common_params & params) {
            params.speculative.draft.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_CPU_MOE"));
    add_opt(common_arg(
        {"--spec-draft-n-cpu-moe", "--spec-draft-ncmoe", "-ncmoed", "--n-cpu-moe-draft"}, "N",
        "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            for (int i = 0; i < value; ++i) {
                static std::list<std::string> buft_overrides_draft;
                buft_overrides_draft.push_back(llm_ffn_exps_block_regex(i));
                params.speculative.draft.tensor_buft_overrides.push_back({buft_overrides_draft.back().c_str(), ggml_backend_cpu_buffer_type()});
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE"));

    add_opt(common_arg(
        {"--spec-draft-n-max"}, "N",
        string_format("number of tokens to draft for speculative decoding (default: %d)", params.speculative.draft.n_max),
        [](common_params & params, int value) {
            params.speculative.draft.n_max = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_MAX"));
    add_opt(common_arg(
        {"--spec-draft-n-min"}, "N",
        string_format("minimum number of draft tokens to use for speculative decoding (default: %d)", params.speculative.draft.n_min),
        [](common_params & params, int value) {
            params.speculative.draft.n_min = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_N_MIN"));

    add_opt(common_arg(
        {"--spec-draft-p-split", "--draft-p-split"}, "P",
        string_format("speculative decoding split probability (default: %.2f)", (double)params.speculative.draft.p_split),
        [](common_params & params, const std::string & value) {
            params.speculative.draft.p_split = std::stof(value);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_P_SPLIT"));
    add_opt(common_arg(
        {"--spec-draft-p-min", "--draft-p-min"}, "P",
        string_format("minimum speculative decoding probability (greedy) (default: %.2f)", (double)params.speculative.draft.p_min),
        [](common_params & params, const std::string & value) {
            params.speculative.draft.p_min = std::stof(value);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_P_MIN"));
    add_opt(common_arg(
        {"--spec-draft-ctx-size", "-cd", "--ctx-size-draft"}, "N",
        string_format("size of the prompt context for the draft model (default: %d, 0 = loaded from model)", params.speculative.draft.n_ctx),
        [](common_params & params, int value) {
            params.speculative.draft.n_ctx = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_CTX_SIZE"));
    add_opt(common_arg(
        {"--spec-draft-device", "-devd", "--device-draft"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading the draft model (none = don't offload)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            params.speculative.draft.devices = parse_device_list(value);
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    GGML_ASSERT(params.speculative.draft.n_gpu_layers < 0); // string_format would need to be extended for a default >= 0
    add_opt(common_arg(
        {"--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
        string_format("max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)",
            params.speculative.draft.n_gpu_layers == -1 ? "auto" : "all"),
        [](common_params & params, const std::string & value) {
            if (value == "auto") {
                params.speculative.draft.n_gpu_layers = -1;
            } else if (value == "all") {
                params.speculative.draft.n_gpu_layers = -2;
            } else {
                params.speculative.draft.n_gpu_layers = std::stoi(value);
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers-draft option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_N_GPU_LAYERS_DRAFT"));
    add_opt(common_arg(
        {"--spec-draft-model", "-md", "--model-draft"}, "FNAME",
        "draft model for speculative decoding (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.draft.mparams.path = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_DRAFT_MODEL"));
    add_opt(common_arg(
        {"--spec-draft-replace", "--spec-replace"}, "TARGET", "DRAFT",
        "translate the string in TARGET into DRAFT if the draft model and main model are not compatible",
        [](common_params & params, const std::string & tgt, const std::string & dft) {
            params.speculative.draft.replacements.push_back({ tgt, dft });
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-type"}, "[none|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]",
        string_format("type of speculative decoding to use when no draft model is provided (default: %s)\n",
            common_speculative_type_to_str(params.speculative.type).c_str()),
        [](common_params & params, const std::string & value) {
            if (value == "none") {
                params.speculative.type = COMMON_SPECULATIVE_TYPE_NONE;
            } else if (value == "ngram-cache") {
                params.speculative.type = COMMON_SPECULATIVE_TYPE_NGRAM_CACHE;
            } else if (value == "ngram-simple") {
                params.speculative.type = COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE;
            } else if (value == "ngram-map-k") {
                params.speculative.type = COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
            } else if (value == "ngram-map-k4v") {
                params.speculative.type = COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V;
            } else if (value == "ngram-mod") {
                params.speculative.type = COMMON_SPECULATIVE_TYPE_NGRAM_MOD;
            } else {
                throw std::invalid_argument("unknown speculative decoding type without draft model");
            }
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_TYPE"));
    add_opt(common_arg(
        {"--spec-ngram-mod-n-min"}, "N",
        string_format("minimum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", params.speculative.ngram_mod.n_min),
        [](common_params & params, int value) {
            if (value < 0 || value > 1024) {
                throw std::invalid_argument("ngram n-min must be between 0 and 1024 inclusive");
            }
            params.speculative.ngram_mod.n_min = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-mod-n-max"}, "N",
        string_format("maximum number of ngram tokens to use for ngram-based speculative decoding (default: %d)", params.speculative.ngram_mod.n_max),
        [](common_params & params, int value) {
            if (value < 0 || value > 1024) {
                throw std::invalid_argument("ngram n-max must be between 0 and 1024 inclusive");
            }
            params.speculative.ngram_mod.n_max = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-mod-n-match"}, "N",
        string_format("ngram-mod lookup length (default: %d)", params.speculative.ngram_mod.n_match),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_mod.n_match = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-ngram-simple-size-n"}, "N",
        string_format("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram_simple.size_n),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_simple.size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-simple-size-m"}, "N",
        string_format("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram_simple.size_m),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_simple.size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-simple-min-hits"}, "N",
        string_format("minimum hits for ngram-simple speculative decoding (default: %d)", params.speculative.ngram_simple.min_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("ngram min hits must be at least 1");
            }
            params.speculative.ngram_simple.min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-ngram-map-k-size-n"}, "N",
        string_format("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram_map_k.size_n),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_map_k.size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k-size-m"}, "N",
        string_format("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram_map_k.size_m),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_map_k.size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k-min-hits"}, "N",
        string_format("minimum hits for ngram-map-k speculative decoding (default: %d)", params.speculative.ngram_map_k.min_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("ngram min hits must be at least 1");
            }
            params.speculative.ngram_map_k.min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-ngram-map-k4v-size-n"}, "N",
        string_format("ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: %d)", params.speculative.ngram_map_k4v.size_n),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_map_k4v.size_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k4v-size-m"}, "N",
        string_format("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: %d)", params.speculative.ngram_map_k4v.size_m),
        [](common_params & params, int value) {
            if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
            }
            params.speculative.ngram_map_k4v.size_m = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--spec-ngram-map-k4v-min-hits"}, "N",
        string_format("minimum hits for ngram-map-k4v speculative decoding (default: %d)", params.speculative.ngram_map_k4v.min_hits),
        [](common_params & params, int value) {
            if (value < 1) {
                throw std::invalid_argument("ngram min hits must be at least 1");
            }
            params.speculative.ngram_map_k4v.min_hits = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    //
    // removed params
    //

    add_opt(common_arg(
        {"--draft", "--draft-n", "--draft-max"}, "N",
        "the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max",
        [](common_params & /*params*/, int /*value*/) {
            throw std::invalid_argument("the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_DRAFT_MAX"));
    add_opt(common_arg(
        {"--draft-min", "--draft-n-min"}, "N",
        "the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min",
        [](common_params & /*params*/, int /*value*/) {
            throw std::invalid_argument("the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_DRAFT_MIN"));
    add_opt(common_arg(
        {"--spec-ngram-size-n"}, "N",
        "the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match",
        [](common_params & /*params*/, int /*value*/) {
            throw std::invalid_argument("the argument has been removed. use the respective --spec-ngram-*-size-n");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--spec-ngram-size-m"}, "N",
        "the argument has been removed. use the respective --spec-ngram-*-size-m",
        [](common_params & /*params*/, int /*value*/) {
            throw std::invalid_argument("the argument has been removed. use the respective --spec-ngram-*-size-m");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--spec-ngram-min-hits"}, "N",
        "the argument has been removed. use the respective --spec-ngram-*-min-hits",
        [](common_params & /*params*/, int /*value*/) {
            throw std::invalid_argument("the argument has been removed. use the respective --spec-ngram-*-min-hits");
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SERVER}));

    //
    // TTS params
    //

    add_opt(common_arg(
        {"-mv", "--model-vocoder"}, "FNAME",
        "vocoder model for audio generation (default: unused)",
        [](common_params & params, const std::string & value) {
            params.vocoder.model.path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS, LLAMA_EXAMPLE_SERVER}));
     add_opt(common_arg(
        {"--tts-use-guide-tokens"},
        "Use guide tokens to improve TTS word recall",
        [](common_params & params) {
            params.vocoder.use_guide_tokens = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--tts-speaker-file"}, "FNAME",
        "speaker file path for audio generation",
        [](common_params & params, const std::string & value) {
            params.vocoder.speaker_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS}));

    //
    // diffusion params
    //

    add_opt(common_arg(
        {"--diffusion-steps"}, "N",
        string_format("number of diffusion steps (default: %d)", params.diffusion.steps),
        [](common_params & params, int value) { params.diffusion.steps = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-visual"},
        string_format("enable visual diffusion mode (show progressive generation) (default: %s)", params.diffusion.visual_mode ? "true" : "false"),
        [](common_params & params) { params.diffusion.visual_mode = true; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eps"}, "F",
        string_format("epsilon for timesteps (default: %.6f)", (double) params.diffusion.eps),
        [](common_params & params, const std::string & value) { params.diffusion.eps = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-algorithm"}, "N",
        string_format("diffusion algorithm: 0=ORIGIN, 1=ENTROPY_BASED, 2=MARGIN_BASED, 3=RANDOM, 4=LOW_CONFIDENCE (default: %d)", params.diffusion.algorithm),
        [](common_params & params, int value) { params.diffusion.algorithm = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-alg-temp"}, "F",
        string_format("dream algorithm temperature (default: %.3f)", (double) params.diffusion.alg_temp),
        [](common_params & params, const std::string & value) { params.diffusion.alg_temp = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-block-length"}, "N",
        string_format("llada block length for generation (default: %d)", params.diffusion.block_length),
        [](common_params & params, int value) { params.diffusion.block_length = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-cfg-scale"}, "F",
        string_format("llada classifier-free guidance scale (default: %.3f)", (double) params.diffusion.cfg_scale),
        [](common_params & params, const std::string & value) { params.diffusion.cfg_scale = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-add-gumbel-noise"}, "F",
        string_format("add gumbel noise to the logits if temp > 0.0 (default: %s)", params.diffusion.add_gumbel_noise ? "true" : "false"),
        [](common_params & params, const std::string & value) { params.diffusion.add_gumbel_noise = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        { "-lr", "--learning-rate" }, "ALPHA",
        string_format("adamw or sgd optimizer alpha (default: %.2g); note: sgd alpha recommended ~10x (no momentum)", (double) params.lr.lr0),
        [](common_params & params, const std::string & value) { params.lr.lr0 = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg({ "-lr-min", "--learning-rate-min" }, "ALPHA",
        string_format("(if >0) final learning rate after decay (if -decay-epochs is set, default=%.2g)",
            (double) params.lr.lr_min),
        [](common_params & params, const std::string & value) { params.lr.lr_min = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-decay-epochs", "--learning-rate-decay-epochs"}, "ALPHA",
        string_format("(if >0) decay learning rate to -lr-min after this many epochs (exponential decay, default=%.2g)", (double) params.lr.decay_epochs),
        [](common_params & params, const std::string & value) { params.lr.decay_epochs = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-wd", "--weight-decay"}, "WD",
        string_format("adamw or sgd optimizer weight decay (0 is off; recommend very small e.g. 1e-9) (default: %.2g).", (double) params.lr.wd),
        [](common_params & params, const std::string & value) { params.lr.wd = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-val-split", "--val-split"}, "FRACTION",
        string_format("fraction of data to use as validation set for training (default: %.2g).", (double) params.val_split),
        [](common_params & params, const std::string & value) { params.val_split = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-epochs", "--epochs"}, "N",
        string_format("optimizer max # of epochs (default: %d)", params.lr.epochs),
        [](common_params & params, int epochs) { params.lr.epochs = epochs; }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-opt", "--optimizer"}, "sgd|adamw", "adamw or sgd",
        [](common_params & params, const std::string & name) {
            params.optimizer = common_opt_get_optimizer(name.c_str());
            if (params.optimizer == GGML_OPT_OPTIMIZER_TYPE_COUNT) {
                throw std::invalid_argument("invalid --optimizer, valid options: adamw, sgd");
            }
        }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"--check"},
        string_format("check rather than generate results (default: %s)", params.check ? "true" : "false"),
        [](common_params & params) {
            params.check = true;
        }
    ).set_examples({LLAMA_EXAMPLE_RESULTS}));
    add_opt(common_arg(
        {"--save-logits"},
        string_format("save final logits to files for verification (default: %s)", params.save_logits ? "true" : "false"),
        [](common_params & params) {
            params.save_logits = true;
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--logits-output-dir"}, "PATH",
        string_format("directory for saving logits output files (default: %s)", params.logits_output_dir.c_str()),
        [](common_params & params, const std::string & value) {
            params.logits_output_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));
    add_opt(common_arg(
        {"--tensor-filter"}, "REGEX",
        "filter tensor names for debug output (regex pattern, can be specified multiple times)",
        [](common_params & params, const std::string & value) {
            params.tensor_filter.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_DEBUG}));

    // presets
    add_opt(common_arg(
        {"--tts-oute-default"},
        string_format("use default OuteTTS models (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "OuteAI/OuteTTS-0.2-500M-GGUF";
            params.model.hf_file = "OuteTTS-0.2-500M-Q8_0.gguf";
            params.vocoder.model.hf_repo = "ggml-org/WavTokenizer";
            params.vocoder.model.hf_file = "WavTokenizer-Large-75-F16.gguf";
        }
    ).set_examples({LLAMA_EXAMPLE_TTS}));

    add_opt(common_arg(
        {"--embd-gemma-default"},
        string_format("use default EmbeddingGemma model (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF";
            params.model.hf_file = "embeddinggemma-300M-qat-Q4_0.gguf";
            params.port = 8011;
            params.n_ubatch = 2048;
            params.n_batch = 2048;
            params.n_parallel = 32;
            params.n_ctx = 2048*params.n_parallel;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-1.5b-default"},
        string_format("use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-1.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-3b-default"},
        string_format("use default Qwen 2.5 Coder 3B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-3b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-default"},
        string_format("use default Qwen 2.5 Coder 7B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-spec"},
        string_format("use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-14b-spec"},
        string_format("use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-14b-q8_0.gguf";
            params.speculative.draft.mparams.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.draft.mparams.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-30b-default"},
        string_format("use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF";
            params.model.hf_file = "qwen3-coder-30b-a3b-instruct-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--gpt-oss-20b-default"},
        string_format("use gpt-oss-20b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-20b-GGUF";
            params.model.hf_file = "gpt-oss-20b-mxfp4.gguf";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
            //params.default_template_kwargs["reasoning_effort"] = "\"high\"";
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--gpt-oss-120b-default"},
        string_format("use gpt-oss-120b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-120b-GGUF";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
            //params.default_template_kwargs["reasoning_effort"] = "\"high\"";
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-4b-default"},
        string_format("use Gemma 3 4B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-4b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-12b-default"},
        string_format("use Gemma 3 12B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-12b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--spec-default"},
        string_format("enable default speculative decoding config"),
        [](common_params & params) {
            params.speculative.type = COMMON_SPECULATIVE_TYPE_NGRAM_MOD;
            params.speculative.ngram_mod.n_match = 24;
            params.speculative.ngram_mod.n_min = 48;
            params.speculative.ngram_mod.n_max = 64;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    return ctx_arg;
}

void common_params_add_preset_options(std::vector<common_arg> & args) {
    // arguments below won't be treated as CLI args, only preset options
    args.push_back(common_arg(
        {"load-on-startup"}, "NAME",
        "in server router mode, autoload this model on startup",
        [](common_params &, const std::string &) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_LOAD_ON_STARTUP).set_preset_only());

    args.push_back(common_arg(
        {"stop-timeout"}, "SECONDS",
        "in server router mode, force-kill model instance after this many seconds of graceful shutdown",
        [](common_params &, int) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_STOP_TIMEOUT).set_preset_only());

    // args.push_back(common_arg(
    //     {"pin"},
    //     "in server router mode, do not unload this model if models_max is exceeded",
    //     [](common_params &) { /* unused */ }
    // ).set_preset_only());
}

#include "common.h.inc"
#include "common.h.inc"
#include "nlohmann/json.hpp.inc"

#include <stdexcept>
#include <string>

using json = nlohmann::ordered_json;

// Helper to iterate over tools/functions
static void autoparser_foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            continue;
        }
        fn(tool);
    }
}

namespace autoparser {

parser_build_context::parser_build_context(common_chat_peg_builder & p, const generation_params & inputs) :
    p(p),
    inputs(inputs),
    reasoning_parser(p.eps()) {}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs) {
    // Run differential analysis to extract template structure
    struct autoparser autoparser;
    autoparser.analyze_template(tmpl);
    return generate_parser(tmpl, inputs, autoparser);
}

common_chat_params peg_generator::generate_parser(const common_chat_template &    tmpl,
                                                  const struct generation_params & inputs,
                                                  const autoparser &              autoparser) {
    // Create the result structure
    common_chat_params data;
    data.prompt           = common_chat_template_direct_apply(tmpl, inputs);
    data.format           = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = autoparser.preserved_tokens;

    auto parser = autoparser.build_parser(inputs);
    data.parser = parser.save();

    // Build grammar if tools are present
    bool has_tools =
        autoparser.tools.format.mode != tool_format::NONE && inputs.tools.is_array() && !inputs.tools.empty();
    std::string trigger_marker = !autoparser.tools.format.section_start.empty() ? autoparser.tools.format.section_start :
                                                                                  autoparser.tools.format.per_call_start;

    bool has_response_format = !inputs.json_schema.empty() && inputs.json_schema.is_object();
    bool include_grammar = has_response_format || (has_tools &&
            ((inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO && !trigger_marker.empty()) ||
              inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));

    if (include_grammar) {
        data.grammar_lazy = !has_response_format && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            autoparser_foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Set grammar triggers based on tool section markers (fall back to per-call markers)
        if (data.grammar_lazy) {
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, trigger_marker }
            };
        }
    }

    return data;
}

common_peg_arena autoparser::build_parser(const generation_params & inputs) const {
    if (!analysis_complete) {
        throw std::invalid_argument("Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)");
    }
    return build_chat_peg_parser([&](common_chat_peg_builder & p) {
        parser_build_context ctx(p, inputs);
        bool                 extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

        ctx.extracting_reasoning = extract_reasoning && reasoning.mode != reasoning_mode::NONE;
        ctx.content              = &content;
        ctx.reasoning            = &reasoning;

        // Build reasoning parser
        ctx.reasoning_parser = reasoning.build_parser(ctx);

        auto parser = p.eps();

        bool has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
        bool has_response_format = inputs.json_schema.is_object() && !inputs.json_schema.empty();
        bool pure_content        = reasoning.mode == reasoning_mode::NONE;

        if (has_response_format) {
            auto response_format = p.rule("response-format", p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));
            parser = ctx.reasoning_parser + p.space() + p.choice({
                p.literal("```json") + p.space() + response_format + p.space() + p.literal("```"),
                response_format
            }) + p.end();
            pure_content = false;
        } else if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && jinja_caps.supports_tool_calls) {
            parser = tools.build_parser(ctx);
            pure_content = false;
        } else {
            parser = content.build_parser(ctx);
        }
        return pure_content ? p.prefix(inputs.generation_prompt, reasoning.start) + parser : p.prefix(inputs.generation_prompt, reasoning.start) << parser;
    });
}

common_peg_parser analyze_reasoning::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (!ctx.extracting_reasoning) {
        return p.eps();
    }

    if (mode == reasoning_mode::TAG_BASED || mode == reasoning_mode::TOOLS_ONLY) {
        if (!end.empty()) {
            if (!start.empty()) {
                // Standard tag-based: optional(<think>reasoning</think>)
                return p.optional(start + p.reasoning(p.until(end)) + end + p.space());
            }
            // Delimiter-style (empty start)
            return p.optional(p.reasoning(p.until(end)) + end + p.space());
        }
    }

    return p.eps();
}

common_peg_parser analyze_content::build_parser(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        if (ctx.extracting_reasoning) {
            return ctx.reasoning_parser + start + p.content(p.until(end)) + end + p.end();
        }
        return p.content(p.until(start)) + start + p.content(p.until(end)) + end + p.end();
    }
    return ctx.reasoning_parser + p.content(p.rest()) + p.end();
}

common_peg_parser analyze_content::build_optional_wrapped(parser_build_context & ctx) const {
    auto & p = ctx.p;

    if (is_always_wrapped()) {
        return p.optional(start + p.content(p.until(end)) + end);
    }
    return p.eps();
}

common_peg_parser analyze_tools::build_parser(parser_build_context & ctx) const {
    switch (format.mode) {
        case tool_format::JSON_NATIVE:
            return build_tool_parser_json_native(ctx);
        case tool_format::TAG_WITH_JSON:
            return build_tool_parser_tag_json(ctx);
        case tool_format::TAG_WITH_TAGGED:
            return build_tool_parser_tag_tagged(ctx);
        default:
            LOG_ERR("[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work properly. "
                "Check for a fixed template for your model in the models/templates directory of your llama.cpp installation or "
                "report an issue at https://github.com/ggml-org/llama.cpp/issues\n");
            return ctx.p.eps();
    }
}

common_peg_parser analyze_tools::build_tool_parser_json_native(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;
    bool         force_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    // Build effective field names with dot notation if function_field is set
    std::string name_field = format.name_field;
    std::string args_field = format.args_field;

    if (!format.function_field.empty() && format.function_field != "function" &&
        name_field.find('.') == std::string::npos) {
        name_field = format.function_field + "." + name_field;
        args_field = format.function_field + "." + args_field;
    }

    auto tools_parser = p.eps();
    if (format.section_start.empty() && !format.per_call_start.empty()) {
        auto single_tool_parser = p.standard_json_tools(
            format.per_call_start, format.per_call_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order);
        tools_parser = p.trigger_rule("tool-calls", p.one_or_more(single_tool_parser + p.space()));
    } else {
        tools_parser = p.standard_json_tools(
            format.section_start, format.section_end, inputs.tools, inputs.parallel_tool_calls,
            inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED, name_field, args_field, format.tools_array_wrapped,
            format.fun_name_is_key, format.id_field, format.gen_id_field, format.parameter_order);
    }

    // Handle content wrappers if present
    if (ctx.content && ctx.content->is_always_wrapped()) {
        auto wrapped_content = ctx.content->build_optional_wrapped(ctx);
        return ctx.reasoning_parser + wrapped_content + tools_parser + p.end();
    }

    std::string tool_start = "{";
    if (!format.section_start.empty()) {
        tool_start = format.section_start;
    } else if (!format.per_call_start.empty()) {
        tool_start = format.per_call_start;
    }

    return ctx.reasoning_parser + (force_tools ? p.eps() : p.optional(p.content(p.until(tool_start)))) + tools_parser +
           p.end();
}

common_peg_parser analyze_tools::build_func_parser(common_chat_peg_builder & p, const std::string & name,
                                                    const common_peg_parser & call_id_section, bool have_call_id,
                                                    const common_peg_parser & args,
                                                    std::optional<common_peg_parser> atomic_peek) const {
    auto              open           = p.tool_open(function.name_prefix + p.tool_name(p.literal(name)) + function.name_suffix);
    bool              matched_atomic = false;
    common_peg_parser func_parser    = p.eps();

    if (!function.name_suffix.empty()) {
        func_parser    = open + call_id_section + p.space() + args;
        matched_atomic = true;
    } else if (have_call_id) {
        func_parser    = p.atomic(open + call_id_section) + p.space() + args;
        matched_atomic = true;
    } else if (atomic_peek.has_value()) {
        func_parser    = p.atomic(open + call_id_section + p.space() + *atomic_peek) + args;
        matched_atomic = true;
    } else {
        func_parser = open + call_id_section + p.space() + args;
    }

    if (!function.close.empty()) {
        func_parser = func_parser + p.space() + p.tool_close(p.literal(function.close));
    } else if (!format.per_call_end.empty()) {
        // When there's no func_close but there is a per_call_end marker, use peek() to ensure
        // we only emit tool_close when we can actually see the closing marker. This prevents
        // premature closing during partial parsing when we've seen e.g. "</" which could be
        // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
        func_parser = func_parser + p.tool_close(p.peek(p.literal(format.per_call_end)));
    } else {
        func_parser = func_parser + p.tool_close(p.space());  // force this to process tool closing callbacks in mapper
    }
    if (!matched_atomic) {
        func_parser = p.atomic(func_parser);
    }
    return func_parser;
}

common_peg_parser analyze_tools::build_tool_parser_tag_json(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;
    bool         force_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_choice = p.choice();

    autoparser_foreach_function(inputs.tools, [&](const json & tool) {
        const auto & func   = tool.at("function");
        std::string  name   = func.at("name");
        const auto & schema = func.contains("parameters") ? func.at("parameters") : json::object();

        // Build call_id parser based on position (if supported)
        bool have_call_id = false;
        common_peg_parser call_id_section = p.eps();
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix))) + call_id.suffix;
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
            have_call_id = true;
        }
        auto args_parser = p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema));
        if (!arguments.start.empty()) {
            args_parser = p.literal(arguments.start) + args_parser;
        }
        if (!arguments.end.empty()) {
            args_parser = args_parser + p.literal(arguments.end);
        }

        auto atomic_peek = !arguments.start.empty() ? std::optional(p.peek(p.literal(arguments.start))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_parser, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + tool_choice + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call));
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call);
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end)));
        }
    } else {
        std::string separator = ", ";  // Default
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + format.section_end);
        } else {
            tool_calls = p.trigger_rule("tool-call", format.section_start + tool_choice + format.section_end);
        }
    }

    if (!require_calls) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + (force_tools ? p.eps() : p.optional(p.content(content_before_tools))) + tool_calls +
           p.end();
}

common_peg_parser analyze_tools::build_tool_parser_tag_tagged(parser_build_context & ctx) const {
    auto &       p           = ctx.p;
    const auto & inputs      = ctx.inputs;
    bool         force_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    auto until_suffix = p.rule("until-suffix", p.until(arguments.value_suffix));

    common_peg_parser tool_choice = p.choice();

    autoparser_foreach_function(inputs.tools, [&](const json & tool) {
        const auto &          func       = tool.at("function");
        std::string           name       = func.at("name");
        auto                  params     = func.contains("parameters") ? func.at("parameters") : json::object();
        const auto &          properties = params.contains("properties") ? params.at("properties") : json::object();

        std::set<std::string> required;
        if (params.contains("required")) {
            params.at("required").get_to(required);
        }

        auto schema_info = common_schema_info();
        schema_info.resolve_refs(params);

        // Build parser for each argument, separating required and optional
        std::vector<common_peg_parser> required_parsers;
        std::vector<common_peg_parser> optional_parsers;
        for (const auto & [param_name, param_schema] : properties.items()) {
            bool is_required = required.find(param_name) != required.end();

            auto arg =
                p.tool_arg(p.tool_arg_open(arguments.name_prefix + p.tool_arg_name(p.literal(param_name)) +
                                           arguments.name_suffix) +
                           arguments.value_prefix +
                           (schema_info.resolves_to_string(param_schema) ?
                                p.tool_arg_string_value(p.schema(until_suffix,
                                                                 "tool-" + name + "-arg-" + param_name + "-schema",
                                                                 param_schema, true)) :
                                p.tool_arg_json_value(p.schema(
                                    p.json(), "tool-" + name + "-arg-" + param_name + "-schema", param_schema, false)) +
                                    p.space()) +
                           p.tool_arg_close(p.literal(arguments.value_suffix)));

            auto named_arg = p.rule("tool-" + name + "-arg-" + param_name, arg);
            if (is_required) {
                required_parsers.push_back(named_arg);
            } else {
                optional_parsers.push_back(named_arg);
            }
        }

        // Build required arg sequence in definition order
        common_peg_parser args_seq = p.eps();
        for (size_t i = 0; i < required_parsers.size(); i++) {
            if (i > 0) {
                args_seq = args_seq + p.space();
            }
            args_seq = args_seq + required_parsers[i];
        }

        // Build optional args with flexible ordering
        if (!optional_parsers.empty()) {
            common_peg_parser any_opt = p.choice();
            for (const auto & opt : optional_parsers) {
                any_opt |= opt;
            }
            args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
        }

        if (!arguments.start.empty()) {
            args_seq = p.literal(arguments.start) + args_seq;
        }
        if (!arguments.end.empty()) {
            args_seq = args_seq + p.literal(arguments.end);
        }

        // Build call_id parser based on position (if supported)
        common_peg_parser call_id_section = p.eps();
        bool have_call_id = false;
        if (call_id.pos == call_id_position::BETWEEN_FUNC_AND_ARGS && !call_id.prefix.empty() &&
            (!call_id.suffix.empty() || !arguments.start.empty())) {
            have_call_id = true;
            if (!call_id.suffix.empty()) {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(call_id.suffix)) + call_id.suffix);
            } else {
                call_id_section = p.optional(call_id.prefix + p.tool_id(p.until(arguments.start)));
            }
        }

        // Only peek for an arg tag when there are required args that must follow.
        // When all args are optional, the model may emit no arg tags at all (#20650).
        auto atomic_peek = (!arguments.name_prefix.empty() && !required_parsers.empty()) ?
            std::optional(p.peek(p.literal(arguments.name_prefix))) : std::nullopt;
        auto func_parser = build_func_parser(p, name, call_id_section, have_call_id, args_seq, atomic_peek);
        tool_choice |= p.rule("tool-" + name, func_parser);
    });

    auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    common_peg_parser tool_calls = p.eps();

    if (!format.per_call_start.empty()) {
        auto wrapped_call = format.per_call_start + p.space() + tool_choice + p.space() + format.per_call_end;
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.zero_or_more(p.space() + wrapped_call) + p.space());
        } else {
            tool_calls = p.trigger_rule("tool-call", wrapped_call + p.space());
        }
        if (!format.section_start.empty()) {
            tool_calls = p.trigger_rule("tool-calls",
                                        p.literal(format.section_start) + p.space() + tool_calls + p.space() +
                                            (format.section_end.empty() ? p.end() : p.literal(format.section_end) + p.space()));
        }
    } else {
        std::string separator = ", ";  // Default

        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call", format.section_start + p.space() + tool_choice +
                                                         p.zero_or_more(separator + tool_choice) + p.space() +
                                                         format.section_end);
        } else {
            tool_calls = p.trigger_rule(
                "tool-call", format.section_start + p.space() + tool_choice + p.space() + format.section_end);
        }
    }

    if (!require_tools) {
        tool_calls = p.optional(tool_calls);
    }

    std::string trigger_marker       = !format.section_start.empty() ? format.section_start : format.per_call_start;
    auto        content_before_tools = trigger_marker.empty() ? p.eps() : p.until(trigger_marker);
    return ctx.reasoning_parser + (force_tools ? p.eps() : p.optional(p.content(content_before_tools))) + tool_calls +
           p.end();
}

}  // namespace autoparser

#include "common.h.inc"
#include "nlohmann/json.hpp.inc"

#include <cctype>
#include <numeric>

using json = nlohmann::ordered_json;

std::string trim_whitespace(const std::string & str) {
    llama_common_unicode_string trimmed = llama_common_string_trim_rust((const uint8_t *) str.data(), str.size(), 0);
    if (!trimmed.data) {
        return "";
    }
    std::string result((const char *) trimmed.data, trimmed.len);
    llama_common_unicode_string_free(trimmed);
    return result;
}

std::string trim_leading_whitespace(const std::string & str) {
    llama_common_unicode_string trimmed = llama_common_string_trim_rust((const uint8_t *) str.data(), str.size(), 1);
    if (!trimmed.data) {
        return "";
    }
    std::string result((const char *) trimmed.data, trimmed.len);
    llama_common_unicode_string_free(trimmed);
    return result;
}

std::string trim_trailing_whitespace(const std::string & str) {
    llama_common_unicode_string trimmed = llama_common_string_trim_rust((const uint8_t *) str.data(), str.size(), 2);
    if (!trimmed.data) {
        return "";
    }
    std::string result((const char *) trimmed.data, trimmed.len);
    llama_common_unicode_string_free(trimmed);
    return result;
}

std::string trim_trailing_newlines(const std::string & str) {
    llama_common_unicode_string trimmed = llama_common_string_trim_rust((const uint8_t *) str.data(), str.size(), 3);
    if (!trimmed.data) {
        return "";
    }
    std::string result((const char *) trimmed.data, trimmed.len);
    llama_common_unicode_string_free(trimmed);
    return result;
}

diff_split calculate_diff_split(const std::string & left, const std::string & right) {
    diff_split result;

    auto left_seg = segmentize_markers(left);
    auto right_seg = segmentize_markers(right);

    if (left_seg.empty()) {
        result.right = right;
        return result;
    }
    if (right_seg.empty()) {
        result.left = left;
        return result;
    }

    auto left_start = left_seg.begin();
    auto left_end = --left_seg.end();
    auto right_start = right_seg.begin();
    auto right_end = --right_seg.end();

    auto test = [&] () {
        return left_start != left_end && right_start != right_end;
    };

    bool left_fully_consumed = false;
    bool right_fully_consumed = false;

    while (test()) {
        bool advanced = false;
        if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_start++;
            advanced = true;
        }
        if (*left_end == *right_end) {
            result.suffix = left_end->value + result.suffix;
            if (left_start != left_end) {
                left_end--;
            } else {
                left_fully_consumed = true;
            }
            if (right_start != right_end) {
                right_end--;
            } else {
                right_fully_consumed = true;
            }
            advanced = true;
        }
        if (!advanced) {
            break;
        }
    }

    if (left_start == left_end && right_start != right_end) {
        if (*left_start == *right_end) {
            result.suffix = right_end->value + result.suffix;
            right_end--;
            left_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(right_start->value);
            right_start++;
            left_fully_consumed = true;
        }
    } else if (right_start == right_end && left_start != left_end) {
        if (*left_end == *right_start) {
            result.suffix = left_end->value + result.suffix;
            left_end--;
            right_fully_consumed = true;
        } else if (*left_start == *right_start) {
            result.prefix.append(left_start->value);
            left_start++;
            right_fully_consumed = true;
        }
    } else if (left_start == left_end && right_start == right_end && *left_start == *right_start && left_start->type == segment_type::MARKER) {
        result.prefix.append(right_start->value);
        left_fully_consumed = true;
        right_fully_consumed = true;
    }

    auto eat_segment = [](std::string str, const segment & seg) -> std::string { return std::move(str) + seg.value; };

    bool can_have_text_suffix = left_end->type == segment_type::TEXT && right_end->type == segment_type::TEXT;
    bool can_have_text_prefix = right_start->type == segment_type::TEXT && left_start->type == segment_type::TEXT;

    std::string remainder_left = std::accumulate(left_start, left_fully_consumed ? left_end : ++left_end, std::string(), eat_segment);
    std::string remainder_right = std::accumulate(right_start, right_fully_consumed ? right_end : ++right_end, std::string(), eat_segment);

    size_t suffix_len = can_have_text_suffix ? llama_common_suffix_len_rust(
        (const uint8_t *) remainder_left.data(), remainder_left.size(),
        (const uint8_t *) remainder_right.data(), remainder_right.size()) : 0;
    // avoid overlaps between prefix and suffix
    std::string prefix_left  = remainder_left.substr(0, remainder_left.size() - suffix_len);
    std::string prefix_right = remainder_right.substr(0, remainder_right.size() - suffix_len);
    size_t prefix_len = can_have_text_prefix ? llama_common_prefix_len_rust(
        (const uint8_t *) prefix_left.data(), prefix_left.size(),
        (const uint8_t *) prefix_right.data(), prefix_right.size()) : 0;

    result.prefix.append(remainder_left.substr(0, prefix_len));
    result.suffix = remainder_left.substr(remainder_left.length() - suffix_len, suffix_len) + result.suffix;
    result.left = remainder_left.substr(prefix_len, remainder_left.length() - prefix_len - suffix_len);
    result.right = remainder_right.substr(prefix_len, remainder_right.length() - prefix_len - suffix_len);

    if (result.left == "" && result.right == "") {
        // degenerate case, no diff
        result.prefix = left;
        result.suffix = "";
        // pick prefix = all as representation
    }

    // When left has no unique content (result.left is empty), left is entirely
    // shared with right. The simultaneous prefix/suffix segment matching can
    // incorrectly consume trailing segments of left as suffix when those same
    // segments also appear at the end of right (e.g. "\n" at the end of both
    // the shared content and the generation prompt). This rotates the diff.
    // Fix: if left is a prefix of right, enforce that directly.
    if (result.left.empty() && !result.right.empty() &&
            left.size() <= right.size() &&
            right.substr(0, left.size()) == left) {
        result.prefix = left;
        result.suffix = "";
        result.right  = right.substr(left.size());
    }

    return result;
}

// Returns the prefix of `full` up until the first occurrence of the common prefix of `left` and `right`
std::string until_common_prefix(const std::string & full, const std::string & left, const std::string & right) {
    llama_common_unicode_string result = llama_common_until_common_prefix_rust(
            (const uint8_t *) full.data(), full.size(),
            (const uint8_t *) left.data(), left.size(),
            (const uint8_t *) right.data(), right.size());
    return common_rust_string_to_std(result);
}

// Returns the suffix of `full` after the last occurrence of the common suffix of `left` and `right`
std::string after_common_suffix(const std::string & full, const std::string & left, const std::string & right) {
    llama_common_unicode_string result = llama_common_after_common_suffix_rust(
            (const uint8_t *) full.data(), full.size(),
            (const uint8_t *) left.data(), left.size(),
            (const uint8_t *) right.data(), right.size());
    return common_rust_string_to_std(result);
}

// TODO: segmentize will treat a JSON array inside tags as a tag: <calls>[{ "fun": { ... } }]</calls> will be three markers
// not too worried about that because it hasn't turned out as a problem anywhere, but noting here in case it will
// Might have to put some restrictions on tag contents as well (like "no { }")
std::vector<segment> segmentize_markers(const std::string & text) {
    std::vector<segment> retval;
    bool in_marker = false;
    char marker_opener = '\0';

    auto is_marker_opener = [](char c) -> bool { return llama_common_marker_is_opener_rust((uint8_t) c); };
    auto is_marker_closer = [](char op, char c) -> bool { return llama_common_marker_is_closer_rust((uint8_t) op, (uint8_t) c); };

    size_t last_border = 0;

    for (size_t cur_pos = 0; cur_pos < text.length(); cur_pos++) {
        if (!in_marker && is_marker_opener(text[cur_pos])) {
            if (last_border < cur_pos) {
                retval.push_back(segment(segment_type::TEXT, text.substr(last_border, cur_pos - last_border)));
            }
            last_border = cur_pos;
            in_marker = true;
            marker_opener = text[cur_pos];
        } else if (in_marker && is_marker_closer(marker_opener, text[cur_pos])) {
            // no need to check because last_border will always be smaller
                retval.push_back(segment(segment_type::MARKER, text.substr(last_border, cur_pos - last_border + 1)));
            last_border = cur_pos + 1;
            in_marker = false;
            marker_opener = '\0';
        }
    }
    if (last_border < text.length()) {
            retval.push_back(segment(segment_type::TEXT, text.substr(last_border)));
    }
    return retval;
}

std::vector<segment> prune_whitespace_segments(const std::vector<segment> & segments) {
    std::vector<segment> result;
    for (const auto & seg : segments) {
        if (!trim_whitespace(seg.value).empty()) {
            result.push_back(seg);
        }
    }
    return result;
}

namespace autoparser {

std::string apply_template(const common_chat_template & tmpl, const template_params & params) {
    generation_params tmpl_params;
    tmpl_params.messages              = params.messages;
    tmpl_params.tools                 = params.tools;
    tmpl_params.add_generation_prompt = params.add_generation_prompt;
    tmpl_params.enable_thinking       = params.enable_thinking;

    if (params.extra_context) {
        tmpl_params.extra_context = *params.extra_context;
    }
    tmpl_params.extra_context["enable_thinking"] = params.enable_thinking;

    try {
        return common_chat_template_direct_apply(tmpl, tmpl_params);
    } catch (const std::exception & e) {
        LOG_DBG("Template application failed: %s\n", e.what());
        return "";
    }
}

std::optional<compare_variants_result> compare_variants(
    const common_chat_template &                   tmpl,
    const template_params &                        params_A,
    const std::function<void(template_params &)> & params_modifier) {
    // Create variant B by copying A
    template_params params_B = params_A;

    // Apply modifier to create variant B
    if (params_modifier) {
        params_modifier(params_B);
    }

    // Apply template to both variants
    std::string output_A = apply_template(tmpl, params_A);
    std::string output_B = apply_template(tmpl, params_B);

    // Check for template application failures
    if (output_A.empty() || output_B.empty()) {
        return std::nullopt;
    }

    // Calculate diff and return result with both outputs
    compare_variants_result result;
    result.diff     = calculate_diff_split(output_A, output_B);
    result.output_A = output_A;
    result.output_B = output_B;

    return result;
}

}  // namespace autoparser

#include "common.h.inc"
#include "common.h.inc"
#include "nlohmann/json.hpp.inc"

#include <algorithm>

#define ANSI_RESET  "\033[0m"
#define ANSI_PURPLE "\033[1m\x1b[38;5;126m"
#define ANSI_ORANGE "\033[1m\x1b[38;5;214m"
#define ANSI_RED    "\033[1m\x1b[38;5;196m"

using json = nlohmann::ordered_json;

namespace autoparser {

static const std::string FUN_FIRST = "FFF_FIRST_FUN_F";
static const std::string FUN_SECOND = "SSS_SECOND_FUN_S";
static const std::string ARG_FIRST = "AA_ARG_FST_AA";
static const std::string ARG_SECOND = "BB_ARG_SND_BB";
static const std::string USER_MSG = "U_USER_MSG Hello END_U";
static const std::string ASSISTANT_MSG = "A_ASST_MSG I can help END_A";
static const std::string THINKING_CONTENT = "REASON_PART I am thinking END_R";
static const std::string CALL_ID_001 = "call00001";
static const std::string CALL_ID_002 = "call00002";
static const std::string CALL_ID_999 = "call99999";

static std::vector<std::function<void(const common_chat_template & tmpl, autoparser &)>> workarounds(
    { // Old reasoning Qwen templates - they don't really display reasoning content, but we still want to
      // support reasoning on them
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("content.split('</think>')") != std::string::npos &&
              tmpl.src.find("reasoning_content") == std::string::npos &&
              tmpl.src.find("<SPECIAL_12>") == std::string::npos &&
              analysis.reasoning.mode == reasoning_mode::NONE) {
              analysis.reasoning.mode  = reasoning_mode::TAG_BASED;
              analysis.reasoning.start = "<think>";
              analysis.reasoning.end   = "</think>";
              analysis.preserved_tokens.push_back("<think>");
              analysis.preserved_tokens.push_back("</think>");
              LOG_DBG(ANSI_ORANGE "[Patch: old Qwen/Deepseek thinking template]\n" ANSI_RESET);
          }
      },
      // Granite 3.3, with separate reasoning and content markers
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("Write your thoughts between <think></think> and write your response between "
                            "<response></response>") != std::string::npos) {
              analysis.reasoning.mode  = reasoning_mode::TAG_BASED;
              analysis.reasoning.start = "<think>";
              analysis.reasoning.end   = "</think>";
              analysis.preserved_tokens.push_back("<think>");
              analysis.preserved_tokens.push_back("</think>");
              analysis.content.mode  = content_mode::WRAPPED_WITH_REASONING;
              analysis.content.start = "<response>";
              analysis.content.end   = "</response>";
              analysis.preserved_tokens.push_back("<response>");
              analysis.preserved_tokens.push_back("</response>");
              LOG_DBG(ANSI_ORANGE "[Patch: Granite 3.3]\n" ANSI_RESET);
          }
      },
      // Cohere Command R+ - content wrapped in <|CHATBOT_TOKEN|>...<|END_OF_TURN_TOKEN|>
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("<|CHATBOT_TOKEN|>") != std::string::npos &&
              tmpl.src.find("<|END_OF_TURN_TOKEN|>") != std::string::npos && analysis.content.start.empty()) {
              analysis.content.mode  = content_mode::ALWAYS_WRAPPED;
              analysis.content.start = "<|CHATBOT_TOKEN|>";
              analysis.content.end   = "<|END_OF_TURN_TOKEN|>";
              analysis.preserved_tokens.push_back("<|CHATBOT_TOKEN|>");
              analysis.preserved_tokens.push_back("<|END_OF_TURN_TOKEN|>");
              LOG_DBG(ANSI_ORANGE "[Patch: Cohere Command R+]\n" ANSI_RESET);
          }
      },
      // Functionary - no tool call section delimiter
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find("set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", "
                            "\"code_interpreter\") | list | length > 0") != std::string::npos) {
              analysis.content.mode                = content_mode::PLAIN;
              analysis.content.end                 = "";
              analysis.tools.function.name_prefix  = "";
              analysis.tools.format.section_start  = "";
              analysis.tools.format.section_end    = "";
              analysis.tools.format.per_call_start = "<function=";
              analysis.tools.format.per_call_end   = "</function>";
              analysis.tools.function.close        = "";
              analysis.preserved_tokens.clear();
              analysis.preserved_tokens.push_back("<|eot_id|>");
              analysis.preserved_tokens.push_back("<|eom_id|>");
              analysis.preserved_tokens.push_back("<function=");
              analysis.preserved_tokens.push_back(">");
              analysis.preserved_tokens.push_back("</function>");
              LOG_DBG(ANSI_ORANGE "[Patch: Functionary 3.1]\n" ANSI_RESET);
          }
      },
      // DeepSeek-R1-Distill-Qwen
      [](const common_chat_template & tmpl, autoparser & analysis) -> void {
          if (tmpl.src.find(
                  "{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'") !=
              std::string::npos) {
              analysis.tools.format.section_start  = "<｜tool▁calls▁begin｜>";
              analysis.tools.format.section_end    = "<｜tool▁calls▁end｜>";
              analysis.tools.format.per_call_start = "<｜tool▁call▁begin｜>function";
              analysis.tools.function.name_prefix  = "<｜tool▁sep｜>";
              analysis.tools.format.per_call_end   = "<｜tool▁call▁end｜>";
              analysis.tools.function.close        = "```";
              LOG_DBG(ANSI_ORANGE "[Patch: DeepSeek-R1-Distill-Qwen]\n" ANSI_RESET);
          }
      }
    });

// Common JSON structures
static json params_schema = {
    { "type",       "object"                                                           },
    { "properties",
     { { ARG_FIRST, { { "type", "string" }, { "description", "First argument" } } },
        { ARG_SECOND, { { "type", "string" }, { "description", "Second argument" } } } } },
    { "required",   json::array({})                                                    }
};

static json tools = json::array({
    { { "type", "function" },
     { "function",
        json{ { "name", FUN_FIRST }, { "description", "Test function foo" }, { "parameters", params_schema } } } },
    { { "type", "function" },
     { "function",
        json{ { "name", FUN_SECOND }, { "description", "Test function bar" }, { "parameters", params_schema } } } }
});

static json user_msg = json{
    { "role",    "user"  },
    { "content", USER_MSG }
};

static json build_tool_call(const std::string & name, const json & args, const std::string & id = CALL_ID_001) {
    return json{
        { "id",       id                                              },
        { "type",     "function"                                      },
        { "function", json{ { "name", name }, { "arguments", args } } }
    };
}

static json first_tool_call_zero_args         = build_tool_call(FUN_FIRST, json::object(), CALL_ID_001);
static json first_tool_call_one_arg           = build_tool_call(FUN_FIRST, {{ ARG_FIRST, "XXXX" }}, CALL_ID_001);
static json first_tool_call_one_arg_other_val = build_tool_call(FUN_FIRST, {{ ARG_FIRST, "YYYY" }}, CALL_ID_001);
static json first_tool_call_other_arg         = build_tool_call(FUN_FIRST, {{ ARG_SECOND, "YYYY" }}, CALL_ID_001);

static json first_tool_call =
    build_tool_call(FUN_FIRST, json{{ ARG_FIRST,  "XXXX" }, { ARG_SECOND, "YYYY" }}, CALL_ID_001);
static json second_tool_call =
    build_tool_call(FUN_SECOND, json{ { ARG_FIRST,  "XXXX" }, { ARG_SECOND, "YYYY" }}, CALL_ID_002);
static json first_tool_call_alt_id =
    build_tool_call(FUN_FIRST, json{{ ARG_FIRST,  "XXXX" }, { ARG_SECOND, "YYYY" }}, CALL_ID_999);

template <typename T>
static std::string mode_to_str(T mode) {
    std::ostringstream os;
    os << mode;
    return os.str();
}

void autoparser::analyze_template(const common_chat_template & tmpl) {
    jinja_caps = tmpl.original_caps();
    reasoning = analyze_reasoning(tmpl, jinja_caps.supports_tool_calls);
    content = analyze_content(tmpl, reasoning);
    tools = analyze_tools(jinja_caps.supports_tool_calls ? analyze_tools(tmpl, jinja_caps, reasoning) : analyze_tools());
    collect_preserved_tokens();

    for (auto & workaround : workarounds) {
        workaround(tmpl, *this);
    }

    LOG_DBG("\n--- Reasoning & Content Structure ---\n");
    LOG_DBG("reasoning_mode: %s\n", mode_to_str(reasoning.mode).c_str());
    LOG_DBG("reasoning_start: '%s'\n", reasoning.start.c_str());
    LOG_DBG("reasoning_end: '%s'\n", reasoning.end.c_str());
    LOG_DBG("content_mode: %s\n", mode_to_str(content.mode).c_str());
    LOG_DBG("content_start: '%s'\n", content.start.c_str());
    LOG_DBG("content_end: '%s'\n", content.end.c_str());

    LOG_DBG("\n--- Tool Call Structure ---\n");
    LOG_DBG("tool_mode: %s\n", mode_to_str(tools.format.mode).c_str());
    LOG_DBG("supports_tools: %s\n", jinja_caps.supports_tools ? "true" : "false");
    LOG_DBG("supports_parallel_calls: %s\n", jinja_caps.supports_parallel_tool_calls ? "true" : "false");
    LOG_DBG("tool_section_start: '%s'\n", tools.format.section_start.c_str());
    LOG_DBG("tool_section_end: '%s'\n", tools.format.section_end.c_str());
    LOG_DBG("per_call_start: '%s'\n", tools.format.per_call_start.c_str());
    LOG_DBG("per_call_end: '%s'\n", tools.format.per_call_end.c_str());
    LOG_DBG("func_name_prefix: '%s'\n", tools.function.name_prefix.c_str());
    LOG_DBG("func_name_suffix: '%s'\n", tools.function.name_suffix.c_str());
    LOG_DBG("func_close: '%s'\n", tools.function.close.c_str());
    LOG_DBG("call_id_prefix: '%s'\n", tools.call_id.prefix.c_str());
    LOG_DBG("call_id_suffix: '%s'\n", tools.call_id.suffix.c_str());
    LOG_DBG("call_id_pos: '%s'\n", mode_to_str(tools.call_id.pos).c_str());
    LOG_DBG("args_start: '%s'\n", tools.arguments.start.c_str());
    LOG_DBG("args_end: '%s'\n", tools.arguments.end.c_str());
    LOG_DBG("arg_name_prefix: '%s'\n", tools.arguments.name_prefix.c_str());
    LOG_DBG("arg_name_suffix: '%s'\n", tools.arguments.name_suffix.c_str());
    LOG_DBG("arg_value_prefix: '%s'\n", tools.arguments.value_prefix.c_str());
    LOG_DBG("arg_value_suffix: '%s'\n", tools.arguments.value_suffix.c_str());
    LOG_DBG("name_field: '%s'\n", tools.format.name_field.c_str());
    LOG_DBG("args_field: '%s'\n", tools.format.args_field.c_str());
    LOG_DBG("id_field: '%s'\n", tools.format.id_field.c_str());
    LOG_DBG("gen_id_field: '%s'\n", tools.format.gen_id_field.c_str());
    LOG_DBG("parameter_order: '%s'\n", std::accumulate(tools.format.parameter_order.begin(), tools.format.parameter_order.end(),
        std::string(""), [] (const std::string & a, const std::string & b) { return a.empty() ? b : a + ", " + b; }
        ).c_str());

    LOG_DBG(ANSI_PURPLE "=== Differential analysis complete ===\n" ANSI_RESET);
    analysis_complete = true;
}

void autoparser::collect_preserved_tokens() {
    auto add_token = [this](const std::string & org_token) {
        std::string token = trim_whitespace(org_token);
        if (!token.empty()) {
            // Avoid duplicates
            if (std::find(preserved_tokens.begin(), preserved_tokens.end(), token) == preserved_tokens.end()) {
                preserved_tokens.push_back(token);
            }
        }
    };

    add_token(reasoning.start);
    add_token(reasoning.end);
    add_token(content.start);
    add_token(content.end);
    add_token(tools.format.section_start);
    add_token(tools.format.section_end);
    add_token(tools.format.per_call_start);
    add_token(tools.format.per_call_end);
    add_token(tools.function.name_prefix);
    add_token(tools.function.name_suffix);
    add_token(tools.function.close);
    add_token(tools.arguments.start);
    add_token(tools.arguments.end);
    add_token(tools.arguments.name_prefix);
    add_token(tools.arguments.name_suffix);
    add_token(tools.arguments.separator);
    add_token(tools.arguments.value_prefix);
    add_token(tools.arguments.value_suffix);
    add_token(tools.call_id.prefix);
    add_token(tools.call_id.suffix);
}

analyze_reasoning::analyze_reasoning(const common_chat_template & tmpl, bool supports_tools)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_PURPLE "=== Starting differential analysis ===\n" ANSI_RESET);
    LOG_DBG(ANSI_ORANGE "Phase 1: Reasoning analysis\n" ANSI_RESET);

    compare_reasoning_presence();
    compare_thinking_enabled();
    if (supports_tools) {
        compare_reasoning_scope();
    }
}

void analyze_reasoning::compare_reasoning_presence() {
    json user_msg = json{
        { "role",    "user"  },
        { "content", USER_MSG }
    };

    json assistant_no_reasoning = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG }
    };

    json assistant_with_reasoning = json{
        { "role",              "assistant"                },
        { "content",           ASSISTANT_MSG              },
        { "reasoning_content", THINKING_CONTENT           }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_reasoning });
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_reasoning }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed, skipping reasoning detection\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    const std::string reasoning_content = THINKING_CONTENT;

    if (!diff.right.empty() && diff.right.find(reasoning_content) != std::string::npos) {
        auto parser_delimiter = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.literal(reasoning_content) + p.space() + p.optional(p.tag("post", (p.marker() + p.space())) + p.rest());
        });
        auto parser_wrapped = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.marker() + p.space()) + p.literal(reasoning_content) + p.tag("post", (p.space() + p.marker() + p.space())) + p.rest();
        });
        // try the more aggressive parse first, if it fails, fall back to the delimiter one
        auto result = parser_wrapped.parse_anywhere_and_extract(comparison->output_B);
        if (!result.result.success()) {
            result = parser_delimiter.parse_anywhere_and_extract(comparison->output_B);
        }
        if (result.result.success()) {
            if (!result.tags["pre"].empty() && !result.tags["post"].empty()) {
                mode = reasoning_mode::TAG_BASED;
                start = result.tags["pre"];
                end   = result.tags["post"];
            } else if (!result.tags["post"].empty()) {
                mode = reasoning_mode::TAG_BASED;
                end = result.tags["post"];
            }
        }
    }
}

void analyze_reasoning::compare_thinking_enabled() {
    json user_msg = json{
        { "role",    "user"  },
        { "content", USER_MSG }
    };

    template_params params;
    params.messages              = json::array({ user_msg });
    params.add_generation_prompt = true;
    params.enable_thinking       = false;

    auto comparison = compare_variants(*tmpl, params, [&](template_params & p) { p.enable_thinking = true; });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET , __func__);
        return;
    }

    const auto & diff = comparison->diff;

    std::string left_trimmed = trim_whitespace(diff.left);
    std::string right_trimmed = trim_whitespace(diff.right);

    if (left_trimmed.empty() && !diff.right.empty()) {
        if (!right_trimmed.empty() && string_ends_with(comparison->output_B, right_trimmed)) {
            if (start.empty()) {
                start = trim_leading_whitespace(diff.right);
                mode  = reasoning_mode::TAG_BASED;
            }
        }
    } else if (right_trimmed.empty() && !diff.left.empty()) {
        if (!left_trimmed.empty() && string_ends_with(comparison->output_A, left_trimmed)) {
            if (end.empty()) {
                auto seg = prune_whitespace_segments(segmentize_markers(comparison->output_A));
                if (seg.size() >= 2 && seg[seg.size() - 1].value == left_trimmed && seg[seg.size() - 2].type == segment_type::MARKER) {
                    start = seg[seg.size() - 2].value;
                }
                end = trim_trailing_whitespace(diff.left);
                mode = reasoning_mode::TAG_BASED;
            }
        }
    } else if (!left_trimmed.empty() && !right_trimmed.empty()) {
        // Full-output diff is noisy (e.g., SmolLM3 changes the system message when enable_thinking flips).
        // Try to find reasoning markers by tail-anchoring:
        // one output's generation prompt tail may appear in the other with extra reasoning markers appended.
        const auto & output_A = comparison->output_A;
        const auto & output_B = comparison->output_B;
        const size_t anchor_len = 64;

        for (int dir = 0; dir < 2; dir++) {
            const auto & base     = dir == 0 ? output_B : output_A;
            const auto & extended = dir == 0 ? output_A : output_B;

            size_t len = std::min(base.size(), anchor_len);
            std::string anchor = base.substr(base.size() - len);
            auto pos = extended.rfind(anchor);
            if (pos == std::string::npos || pos + len >= extended.size()) {
                continue;
            }

            std::string extra = trim_whitespace(extended.substr(pos + len));
            if (extra.empty()) {
                continue;
            }

            auto seg = prune_whitespace_segments(segmentize_markers(extra));
            if (seg.size() == 2 && seg[0].type == segment_type::MARKER && seg[1].type == segment_type::MARKER) {
                if (start.empty()) {
                    start = seg[0].value;
                }
                if (end.empty()) {
                    end   = seg[1].value;
                }
                mode = reasoning_mode::TAG_BASED;
                break;
            }
        }
    }

    if (mode == reasoning_mode::NONE && start.empty() && !end.empty()) {
        mode = reasoning_mode::TAG_BASED;
    }
}

void analyze_reasoning::compare_reasoning_scope() {
    json assistant_reasoning_content = json{
        { "role",              "assistant"      },
        { "content",           ASSISTANT_MSG    },
        { "reasoning_content", THINKING_CONTENT }
    };

    json assistant_reasoning_tools = json{
        { "role",              "assistant"                                                                  },
        { "content",           nullptr                                                                      },
        { "reasoning_content", THINKING_CONTENT                                                             },
        { "tool_calls",
         json::array({ build_tool_call(FUN_FIRST, json{ { ARG_FIRST, "VVVV" }, { ARG_SECOND, "XXXX" } }) }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_reasoning_content });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_reasoning_tools }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    std::string reasoning_content = THINKING_CONTENT;

    // Check if reasoning only appears in variant B (with tools)
    bool reasoning_in_A = comparison->output_A.find(reasoning_content) != std::string::npos;
    bool reasoning_in_B = comparison->output_B.find(reasoning_content) != std::string::npos;

    if (!reasoning_in_A && reasoning_in_B) {
        mode = reasoning_mode::TOOLS_ONLY;
        LOG_DBG(ANSI_ORANGE "%s: Detected TOOLS_ONLY reasoning mode\n" ANSI_RESET, __func__);

        auto parser_wrapped = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.marker() + p.space()) + p.literal(reasoning_content) + p.space() + p.tag("post", (p.marker() + p.space()));
        });
        auto result = parser_wrapped.parse_anywhere_and_extract(comparison->output_B);
        if (result.result.success()) {
            start = result.tags["pre"];
            end = trim_trailing_whitespace(result.tags["post"]);
        } else {
            auto parser_delimiter = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                return p.literal(reasoning_content) + p.space() + p.optional(p.tag("post", (p.marker() + p.space())));
            });
            result = parser_delimiter.parse_anywhere_and_extract(comparison->output_B);
            if (result.result.success()) {
                end = trim_trailing_whitespace(result.tags["post"]);
            } else {
                LOG_DBG(ANSI_ORANGE "%s: Unable to extract reasoning markers, falling back to reasoning = NONE\n" ANSI_RESET, __func__);
                mode = reasoning_mode::NONE;
            }
        }
    }
}

analyze_content::analyze_content(const common_chat_template & tmpl, const analyze_reasoning & reasoning)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_ORANGE "Phase 2: Content analysis\n" ANSI_RESET);

    json assistant_content_only = json{
        { "role",    "assistant"     },
        { "content", ASSISTANT_MSG   }
    };

    json assistant_with_tools = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ build_tool_call("test_func", json{ { "arg1", "value1" } }) }) }
    };

    json assistant_with_reasoning = json{
        { "role",              "assistant"      },
        { "content",           ""               },
        { "reasoning_content", THINKING_CONTENT }
    };

    template_params params_content_only;
    params_content_only.messages              = json::array({ user_msg, assistant_content_only });
    params_content_only.add_generation_prompt = false;
    params_content_only.enable_thinking       = true;
    params_content_only.tools                 = tools;

    auto comparison_with_tools = compare_variants(tmpl, params_content_only, [&](template_params & p) {
        p.messages = json::array({ user_msg, assistant_with_tools });
    });

    auto comparison_with_reasoning = compare_variants(tmpl, params_content_only, [&](template_params & p) {
        p.messages = json::array({ user_msg, assistant_with_reasoning });
    });

    if (!comparison_with_tools || !comparison_with_reasoning) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff_tools     = comparison_with_tools->diff;
    const auto & diff_reasoning = comparison_with_reasoning->diff;

    std::string response = ASSISTANT_MSG;

    bool found_plain_content = false;
    if (trim_whitespace(diff_tools.left) == response) {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.space() + diff_reasoning.left + p.space() + p.optional(p.marker()) + p.space() + p.end();
        });
        if (parser.parse_and_extract(diff_reasoning.left).result.success()) {
            // We only have the content text in the diff (possibly with a stray EOG marker), so no markers
            mode = content_mode::PLAIN;
            found_plain_content = true;
        } else if (reasoning.mode != reasoning_mode::NONE && !reasoning.end.empty()) {
            auto post_reasoning_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.literal(reasoning.end) + p.space() + p.literal(response);
            });
            if (post_reasoning_parser.parse_anywhere_and_extract(diff_reasoning.left).result.success()) {
                mode = content_mode::PLAIN;
                found_plain_content = true;
            }
        }
    }
    if (!found_plain_content) {
        std::string rdiff = diff_reasoning.left;
        if (!reasoning.end.empty() && rdiff.find(reasoning.end) != std::string::npos) {
            rdiff = rdiff.substr(rdiff.find(reasoning.end) + reasoning.end.length());
        }
        // Take the more promising diff
        std::string pure_content = rdiff.length() > diff_tools.left.length() ? rdiff : diff_tools.left;
        auto parser_wrapped = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.marker() + p.space()) + p.literal(response) + p.space() + p.tag("post", (p.marker() + p.space())) + p.rest();
        });
        auto result = parser_wrapped.parse_anywhere_and_extract(pure_content);
        start = result.tags["pre"];
        end = result.tags["post"];
        // TODO: WRAPPED_WITH_REASONING
    }

    // Determine content mode
    if (!start.empty() || !end.empty()) {
        mode = content_mode::ALWAYS_WRAPPED;
        // TODO: END_DELIMITED content mode - delimited at end but not at start?
    }
}

bool analyze_content::is_always_wrapped() const {
    return llama_common_content_is_always_wrapped_rust((int) mode, start.size(), end.size());
}

analyze_tools::analyze_tools(const common_chat_template & tmpl,
                             const jinja::caps &          caps,
                             const analyze_reasoning &    reasoning)
    : analyze_base(tmpl) {
    LOG_DBG(ANSI_ORANGE "Phase 3: Tool call analysis\n" ANSI_RESET);

    analyze_tool_calls(reasoning, caps.supports_parallel_tool_calls);

    if (format.mode != tool_format::NONE && format.mode != tool_format::JSON_NATIVE) {
        if (caps.supports_parallel_tool_calls) {
            check_per_call_markers();
        }
        LOG_DBG(ANSI_ORANGE "Phase 3a: Function call analysis\n" ANSI_RESET);
        extract_function_markers();
        LOG_DBG(ANSI_ORANGE "Phase 3b: Argument analysis\n" ANSI_RESET);
        if (format.mode == tool_format::TAG_WITH_TAGGED) {
            analyze_arguments();
        }
        extract_argument_separator();
        extract_args_markers();
        LOG_DBG(ANSI_ORANGE "Phase 3c: Call id analysis\n" ANSI_RESET);
        extract_call_id_markers();
    }
}

void analyze_tools::analyze_tool_calls(const analyze_reasoning & reasoning, bool supports_parallel_tool_calls) {
    json assistant_no_tools = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG }
    };

    json assistant_with_tools = json{
        { "role",       "assistant"                      },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_tools });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_tools }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    std::string tool_section = diff.right;

    if (tool_section.empty()) {
        return;
    }

    analyze_tool_call_format(tool_section, FUN_FIRST, ARG_FIRST, reasoning, supports_parallel_tool_calls);
}

void analyze_tools::analyze_tool_call_format(const std::string &       haystack,
                                             const std::string &       fun_name_needle,
                                             const std::string &       arg_name_needle,
                                             const analyze_reasoning & reasoning,
                                             bool                      supports_parallel_tool_calls) {
    if (fun_name_needle.empty() || arg_name_needle.empty() || haystack.empty()) {
        return;
    }

    auto in_json_haystack = [&haystack](const std::string & needle) -> bool {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.choice({ p.literal("{"), p.literal(":") }) << p.choice({
                p.tag("dq", p.literal("\"") + p.literal(needle) + p.literal("\"")) });
        });
        auto result = parser.parse_anywhere_and_extract(haystack);
        return result.result.success();
    };

    auto fun_quote = in_json_haystack(fun_name_needle);
    auto arg_quote = in_json_haystack(arg_name_needle);

    if (fun_quote) {
        // no need to check further, we're in JSON land
        format.mode = tool_format::JSON_NATIVE;
    } else if (arg_quote) {
        format.mode = tool_format::TAG_WITH_JSON;
    } else {
        format.mode = tool_format::TAG_WITH_TAGGED;
    }

    // first, remove any reasoning markers
    std::string clean_haystack = haystack;
    if (!reasoning.start.empty()) {
        auto pos = haystack.find(reasoning.start);
        if (pos != std::string::npos) {
            clean_haystack = haystack.substr(0, pos) + haystack.substr(pos + reasoning.start.length());
        }
    }
    if (!reasoning.end.empty()) {
        auto pos = clean_haystack.find(reasoning.end);
        if (pos != std::string::npos) {
            clean_haystack = clean_haystack.substr(0, pos) + clean_haystack.substr(pos + reasoning.end.length());
        }
    }

    if (format.mode == tool_format::JSON_NATIVE) {
        analyze_tool_call_format_json_native(clean_haystack, fun_name_needle, arg_name_needle);
        if (supports_parallel_tool_calls) {
            analyze_json_native_parallel_calls();
        }
    } else {
        analyze_tool_call_format_non_json(clean_haystack, fun_name_needle);
    }
    // always relax whitespace requirements on ending markers since they don't influence content
    format.section_end  = trim_whitespace(format.section_end);
    format.per_call_end = trim_whitespace(format.per_call_end);
}

void analyze_tools::analyze_json_native_parallel_calls() {
    json assistant_one_tool = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_two_tools = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call, second_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_one_tool });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_tools }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    std::string & second_call = comparison->diff.right;
    if (!format.section_start.empty() && second_call.find(format.section_start) != std::string::npos) {
        format.per_call_start = format.section_start;
        format.per_call_end = format.section_end;
        format.section_start.clear();
        format.section_end.clear();
    }
}

void analyze_tools::analyze_tool_call_format_json_native(const std::string & clean_haystack,
                                                         const std::string & fun_name_needle,
                                                         const std::string & arg_name_needle) {
    // we might not have the typical OpenAI tool calling structure
    int  json_start     = clean_haystack.find_first_of('{');
    int  json_end       = clean_haystack.find_last_of('}');
    std::string cut     = clean_haystack.substr(json_start, json_end - json_start + 1);
    json call_struct    = json::parse(cut);
    auto register_field = [&](const std::string & prefix, const nlohmann::detail::iteration_proxy_value<json::iterator> & subel) {
        if (subel.value().is_string() && std::string(subel.value()).find("call0000") != std::string::npos) {
            format.id_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.value().is_string() && std::string(subel.value()) == fun_name_needle) {
            format.name_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.value().dump().find(arg_name_needle) !=
                   std::string::npos) {  // handle both string and JSON obj variants
            format.args_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        } else if (subel.key().find("id") != std::string::npos) {
            // heuristics for generated id field
            format.gen_id_field = !prefix.empty() ? prefix + "." + subel.key() : subel.key();
        }
    };
    for (const auto & el : call_struct.items()) {
        if (el.key() == fun_name_needle) {
            format.fun_name_is_key = true;
            // When function name is the key, there's no name field and args are direct
            format.name_field.clear();
            format.args_field.clear();
            // Don't register this element - the function name IS the key, not a field
        } else {
            if (el.value().is_object() &&
                el.value().dump().find(arg_name_needle) == std::string::npos) {  // not the args object
                format.function_field = el.key();
                for (const auto & subel : el.value().items()) {
                    register_field(el.key(), subel);
                }
            }
            // Register this element as a potential field
            register_field("", el);
        }
    }
    auto array_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
        return p.tag("pre", p.literal("[") + p.space()) + p.literal(cut) + p.tag("post", p.space() + p.literal("]"));
    });

    auto ar_parse_res = array_parser.parse_anywhere_and_extract(clean_haystack);
    if (ar_parse_res.result.success()) {
        format.tools_array_wrapped = true;
        json_start -= ar_parse_res.tags["pre"].length();
        json_end += ar_parse_res.tags["post"].length();
    }
    json_end++; // we want to move past the closing char for end marker extraction

    std::vector<std::pair<size_t, std::string>> located_params;
    if (!format.name_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.name_field), format.name_field });
    }
    if (!format.args_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.args_field), format.args_field });
    }
    if (!format.id_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.id_field), format.id_field });
    }
    if (!format.gen_id_field.empty()) {
        located_params.push_back({ clean_haystack.find(format.gen_id_field), format.gen_id_field });
    }
    std::sort(located_params.begin(), located_params.end());
    for (auto & pair : located_params) {
        format.parameter_order.push_back(pair.second);
    }
    // we can immediately extract tool calling markers too
    format.section_start = trim_leading_whitespace(clean_haystack.substr(0, json_start));
    format.section_end   = trim_whitespace(clean_haystack.substr(json_end));
    // When tools_array_wrapped is true, the closing bracket is part of the array structure,
    // not a separate section end marker. Clear tool_section_end to avoid duplicate brackets.
    if (format.tools_array_wrapped && format.section_end == "]") {
        format.section_end.clear();
    }
}

void analyze_tools::analyze_tool_call_format_non_json(const std::string & clean_haystack,
                                                      const std::string & fun_name_needle) {
    // first, let's find out if the function is inside a tag or standalone
    auto fun_marker_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("fun_marker", p.choice({
            p.tag("fun_pre", p.literal("<") + p.until_one_of({ ">", fun_name_needle })) + p.literal(fun_name_needle) +
                p.tag("fun_post", p.negate(p.space() + p.literal("<")) + p.until(">") + p.literal(">")) + p.space(),
            p.tag("fun_pre", p.literal("[") + p.until_one_of({ "]", fun_name_needle })) + p.literal(fun_name_needle) +
                p.tag("fun_post", p.negate(p.space() + p.literal("[") + p.until("]") + p.literal("]")) + p.space()) }));
    });
    auto fun_res = fun_marker_parser.parse_anywhere_and_extract(clean_haystack);
    std::string fun_marker = fun_name_needle;
    if (fun_res.result.success()) {
        fun_marker = fun_res.tags["fun_marker"];
    }
    // now, consume up to two markers, then treat everything up to the function marker as function name prefix
    auto per_tool_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
        return p.tag("sec_start", p.marker() + p.space()) + p.tag("call_start", p.marker() + p.space()) +
            p.tag("fun_pre", p.until(fun_marker)) + fun_marker + p.tag("rest", p.rest());
    });
    auto section_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
        return p.tag("sec_start", p.marker() + p.space()) + fun_marker + p.tag("rest", p.rest());
    });
    auto result = per_tool_parser.parse_anywhere_and_extract(clean_haystack);
    tagged_parse_result result_end;
    if (result.result.success()) {
        auto double_closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("call_end", p.marker() + p.space()) + p.tag("sec_end", p.marker() + p.space()) + p.end();
        });
        result_end = double_closer_parser.parse_anywhere_and_extract(result.tags["rest"]);
        function.name_prefix = fun_res.tags["fun_pre"] + function.name_prefix;
    } else {
        result = section_parser.parse_anywhere_and_extract(clean_haystack);
        auto single_closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("sec_end", p.marker() + p.space()) + p.end();
        });
        result_end = single_closer_parser.parse_anywhere_and_extract(result.tags["rest"]);
    }
    format.per_call_start = result.tags["call_start"];
    format.per_call_end = result_end.tags["call_end"];
    format.section_start = result.tags["sec_start"];
    format.section_end = result_end.tags["sec_end"];
}

void analyze_tools::check_per_call_markers() {
    json assistant_one_tool = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_two_tools = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call, second_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_one_tool });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto one_vs_two = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_tools }); });

    if (!one_vs_two) {
        LOG_DBG(ANSI_ORANGE "%s: Generating double tool call comparison failed\n" ANSI_RESET, __func__);
        return;
    }

    diff_split filter_common_call_part = calculate_diff_split(one_vs_two->diff.suffix, one_vs_two->diff.right);

    std::string second_tool_content = trim_leading_whitespace(filter_common_call_part.right);
    if (!format.section_start.empty() &&
        second_tool_content.find(format.section_start) == 0) {
        format.per_call_start = format.section_start;
        format.per_call_end   = format.section_end;
        format.section_start.clear();
        format.section_end.clear();
    }
}

void analyze_tools::extract_function_markers() {
    json assistant_nocall = json{
        { "role",    "assistant"   },
        { "content", ASSISTANT_MSG },
    };

    json assistant_foofoo = json{
        { "role",       "assistant"                      },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_barbar = json{
        { "role",       "assistant"                       },
        { "content",    ""                                },
        { "tool_calls", json::array({ second_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_foofoo });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_barbar }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left.find(FUN_FIRST) != std::string::npos && diff.right.find(FUN_SECOND) != std::string::npos) {
        std::string prefix_marker;
        if (!format.per_call_start.empty()) {
            prefix_marker = format.per_call_start;
        } else {
            prefix_marker = format.section_start;
        }
        if (!prefix_marker.empty() && diff.prefix.rfind(prefix_marker) != std::string::npos) {
            function.name_prefix =
                diff.prefix.substr(diff.prefix.rfind(prefix_marker) + prefix_marker.size());
        }

        // Extract name prefix/suffix from diff.left (stop at the next marker boundary)
        auto name_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
            return p.tag("pre", p.until(FUN_FIRST)) + p.literal(FUN_FIRST) +
                   p.tag("post", p.zero_or_more(p.negate(p.marker()) + p.any()));
        });
        auto name_result = name_parser.parse_and_extract(diff.left);
        if (name_result.result.success()) {
            function.name_prefix += name_result.tags["pre"];
            function.name_suffix = name_result.tags["post"];
        }

        // Extend name_suffix with content from diff.suffix before args begin
        if (format.mode == tool_format::TAG_WITH_JSON) {
            // For JSON: name_suffix extends to the first non-marker { or [, including any
            // markers along the way. Only applies if there's at least one marker after
            // the JSON content (matching the original "stop < seg_suf.size() - 1" guard).
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                auto non_json = p.marker() | (p.negate(p.literal("{")) + p.negate(p.literal("[")) + p.any());
                auto after_json = p.zero_or_more(p.negate(p.marker()) + p.any()) + p.marker();
                return p.tag("ext", p.zero_or_more(non_json)) + after_json;
            });
            auto suf_result = suffix_parser.parse_and_extract(diff.suffix);
            if (suf_result.result.success()) {
                function.name_suffix += suf_result.tags["ext"];
            }
        } else {
            // For tagged: name_suffix extends to the first marker (arg marker)
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                return p.tag("ext", p.zero_or_more(p.negate(p.marker()) + p.any()));
            });
            auto suf_result = suffix_parser.parse_and_extract(diff.suffix);
            if (suf_result.result.success()) {
                function.name_suffix += suf_result.tags["ext"];
            }
        }

        // Extract the closer (between last arg and call/section end marker)
        std::string suffix_marker;
        if (!format.per_call_end.empty()) {
            suffix_marker = format.per_call_end;
        } else {
            suffix_marker = format.section_end;
        }
        std::string closer_suffix;
        if (suffix_marker.empty()) {
            // we'll have to rely on an extra diff with no-calls version
            auto notool_comp = compare_variants(
                *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_nocall }); });
            if (notool_comp) {
                auto nt_diff  = notool_comp->diff;
                closer_suffix = nt_diff.left.substr(nt_diff.left.find("YYYY") + 4);
            }
        } else {
            closer_suffix = diff.suffix.substr(0, diff.suffix.find(suffix_marker));
        }
        if (!closer_suffix.empty()) {
            if (format.mode == tool_format::TAG_WITH_TAGGED) {
                // After last arg value, skip the closing arg marker, rest is closer
                auto closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                    return p.until("YYYY") + p.literal("YYYY") + p.space() +
                           p.marker() + p.space() +
                           p.tag("close", p.rest());
                });
                auto close_result = closer_parser.parse_and_extract(closer_suffix);
                if (close_result.result.success()) {
                    function.close = close_result.tags["close"];
                }
            } else if (format.mode == tool_format::TAG_WITH_JSON) {
                // After last arg value, find end of JSON args, rest is closer
                auto closer_parser = build_tagged_peg_parser([&](common_peg_parser_builder &p) {
                    return p.until("YYYY") + p.literal("YYYY") + p.tag("post_val", p.rest());
                });
                auto close_result = closer_parser.parse_and_extract(closer_suffix);
                if (close_result.result.success()) {
                    const auto & post = close_result.tags["post_val"];
                    size_t pos = post.find_last_of("}]");
                    if (pos != std::string::npos && pos < post.size() - 1) {
                        function.close = trim_leading_whitespace(post.substr(pos + 1));
                    }
                }
            }
        }
        function.close = trim_leading_whitespace(function.close);
    }
}

void analyze_tools::analyze_arguments() {
    extract_argument_name_markers();
    extract_argument_value_markers();
}

void analyze_tools::extract_argument_name_markers() {
    json assistant_first_arg = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    json assistant_second_arg = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_other_arg }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_first_arg });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_second_arg }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (!diff.left.empty() && !diff.right.empty()) {
        // Parse both sides to find ARG_FIRST/ARG_SECOND and extract the surrounding structure
        auto left_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.tag("pre", p.until(ARG_FIRST)) + p.literal(ARG_FIRST) +
                   p.tag("suffix", p.until_one_of({"\"", "X"}));
        });
        auto right_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.tag("pre", p.until(ARG_SECOND)) + p.literal(ARG_SECOND) +
                   p.tag("suffix", p.until_one_of({"\"", "Y"}));
        });
        auto left_result  = left_parser.parse_anywhere_and_extract(diff.left);
        auto right_result = right_parser.parse_anywhere_and_extract(diff.right);

        if (left_result.result.success() && right_result.result.success() &&
            !left_result.tags["pre"].empty() &&
            left_result.tags["pre"] == right_result.tags["pre"] &&
            left_result.tags["suffix"] == right_result.tags["suffix"]) {
            // Name is inside a structure (e.g., JSON key): prefix is the shared wrapper
            arguments.name_prefix = trim_whitespace(left_result.tags["pre"]);
            arguments.name_suffix = trim_leading_whitespace(left_result.tags["suffix"]);
        } else if (diff.left.substr(0, ARG_FIRST.length()) == ARG_FIRST && diff.right.substr(0, ARG_SECOND.length()) == ARG_SECOND) {
            // Name is directly in the diff: prefix comes from last marker in diff.prefix
            auto pre_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                auto last_marker = p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end();
                return p.zero_or_more(p.negate(last_marker) + p.any()) + p.tag("name_prefix", last_marker);
            });
            auto pre_result = pre_parser.parse_and_extract(diff.prefix);
            arguments.name_prefix = pre_result.result.success()
                ? pre_result.tags["name_prefix"] : diff.prefix;

            // Suffix extends from after ARG_FIRST to the first marker (+ optional whitespace).
            // The marker could be in diff.left itself or in diff.suffix, so we concatenate.
            std::string after_first = diff.left.substr(ARG_FIRST.length()) + diff.suffix;
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.tag("suffix", p.zero_or_more(p.negate(p.marker()) + p.any()) +
                                       p.marker() + p.space());
            });
            auto suf_result = suffix_parser.parse_anywhere_and_extract(after_first);
            if (suf_result.result.success()) {
                arguments.name_suffix = suf_result.tags["suffix"];
            }
        }
    }
}

void analyze_tools::extract_argument_value_markers() {
    json assistant_val_X = json{
        { "role",       "assistant"                              },
        { "content",    ""                                       },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    json assistant_val_Y = json{
        { "role",       "assistant"                                        },
        { "content",    ""                                                 },
        { "tool_calls", json::array({ first_tool_call_one_arg_other_val }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_val_X });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_val_Y }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left == "XXXX" && diff.right == "YYYY") {
        std::string arg_name_ending = ARG_FIRST + arguments.name_suffix;
        std::string prefix          = diff.prefix;
        if (prefix.rfind(arg_name_ending) != std::string::npos) {
            prefix = prefix.substr(prefix.rfind(arg_name_ending) + arg_name_ending.size());
        }
        if (!prefix.empty()) {
            // Find the last marker + any trailing non-marker text to end
            auto prefix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                auto last_marker = p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end();
                return p.zero_or_more(p.negate(last_marker) + p.any()) + p.tag("val_prefix", last_marker);
            });
            auto pre_result = prefix_parser.parse_and_extract(prefix);
            arguments.value_prefix = pre_result.result.success() ? pre_result.tags["val_prefix"] : prefix;
        }

        std::string value_suffix = diff.suffix;
        if (!function.close.empty()) {
            size_t func_close_pos = value_suffix.find(function.close);
            if (func_close_pos != std::string::npos) {
                value_suffix = value_suffix.substr(0, func_close_pos);
            }
        } else if (!format.per_call_end.empty() || !format.section_end.empty()) {
            std::string end_marker =
                !format.per_call_end.empty() ? format.per_call_end : format.section_end;
            size_t end_marker_pos = value_suffix.find(end_marker);
            if (end_marker_pos != std::string::npos) {
                value_suffix = value_suffix.substr(0, end_marker_pos);
            }
        }
        value_suffix = trim_leading_whitespace(value_suffix);
        if (!value_suffix.empty()) {
            arguments.value_suffix = value_suffix;
        }
    }
}

void analyze_tools::extract_argument_separator() {
    json assistant_one_arg = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    json assistant_two_args = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_one_arg });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_two_args }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (!diff.right.empty()) {
        std::string separator        = until_common_prefix(diff.right, ARG_FIRST, ARG_SECOND);
        arguments.separator = separator;
    }
}

void analyze_tools::extract_args_markers() {
    json assistant_no_args = json{
        { "role",       "assistant"},
        { "content",    ""         },
        { "tool_calls", json::array({ first_tool_call_zero_args }) }
    };

    json assistant_with_args = json{
        { "role",       "assistant"},
        { "content",    ""         },
        { "tool_calls", json::array({ first_tool_call_one_arg }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_no_args });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_with_args }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (format.mode == tool_format::JSON_NATIVE) {
        std::string prefix_marker = !format.section_start.empty() ? format.section_start : format.per_call_start;
        std::string suffix_marker = !format.section_end.empty() ? format.section_end : format.per_call_end;
        // these might happen earlier in the tools section as an example or somewhere else, so we need to find the closest ones
        size_t prefix_pos = prefix_marker.empty() ? 0 : diff.prefix.rfind(prefix_marker);
        size_t suffix_pos = suffix_marker.empty() ? diff.suffix.size() : diff.suffix.find(suffix_marker);
        if (prefix_pos == std::string::npos) {
            prefix_pos = 0;
        }
        if (suffix_pos == std::string::npos) {
            suffix_pos = diff.suffix.size();
        }
        std::string prefix_cut = diff.prefix.substr(prefix_pos + prefix_marker.size());
        std::string suffix_cut = diff.suffix.substr(0, suffix_pos);
        std::string args_start = until_common_prefix(prefix_cut, "{}", "{\"first\":");
        std::string args_end   = after_common_suffix(suffix_cut, "{}", "\"XXXX\"}");

        if (!args_start.empty() || !args_end.empty()) {
            size_t find_fun = args_start.find(FUN_FIRST);
            if (find_fun != std::string::npos) {
                args_start = args_start.substr(find_fun + FUN_FIRST.size(), args_start.size() - find_fun - FUN_FIRST.size());
            }
            size_t find_call_id = args_start.find(CALL_ID_001);
            if (find_call_id != std::string::npos) {
                args_start = args_start.substr(find_call_id + CALL_ID_001.size(), args_start.size() - find_call_id - CALL_ID_001.size());
            }
            arguments.start = args_start;
            arguments.end   = args_end;
        }
    }
}

void analyze_tools::extract_call_id_markers() {
    json assistant_id1 = json{
        { "role",       "assistant" },
        { "content",    ""                               },
        { "tool_calls", json::array({ first_tool_call }) }
    };

    json assistant_id2 = json{
        { "role",       "assistant" },
        { "content",    ""          },
        { "tool_calls", json::array({ first_tool_call_alt_id }) }
    };

    template_params params;
    params.messages              = json::array({ user_msg, assistant_id1 });
    params.tools                 = tools;
    params.add_generation_prompt = false;
    params.enable_thinking       = true;

    auto comparison = compare_variants(
        *tmpl, params, [&](template_params & p) { p.messages = json::array({ user_msg, assistant_id2 }); });

    if (!comparison) {
        LOG_DBG(ANSI_ORANGE "%s: Template application failed for call_id detection\n" ANSI_RESET, __func__);
        return;
    }

    const auto & diff = comparison->diff;

    if (diff.left.empty() && diff.right.empty()) {
        return;
    }

    std::string id_value_1 = CALL_ID_001;
    std::string id_value_2 = CALL_ID_999;

    size_t common_id_prefix_len = 0;
    for (size_t i = 0; i < std::min(id_value_1.length(), id_value_2.length()); i++) {
        if (id_value_1[i] == id_value_2[i]) {
            common_id_prefix_len++;
        } else {
            break;
        }
    }
    std::string common_id_part = id_value_1.substr(0, common_id_prefix_len);

    // Check if the function name is in the prefix (normal case: BETWEEN_FUNC_AND_ARGS or POST_ARGS)
    // or in the suffix (call_id is PRE_FUNC_NAME)
    std::string func_name           = FUN_FIRST;
    size_t      func_name_in_prefix = diff.prefix.rfind(func_name);
    size_t      func_name_in_suffix = diff.suffix.find(func_name);

    // Helper: find the last marker in a string (returns just the marker, not trailing text)
    auto find_last_marker = [](const std::string & str) -> std::string {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            auto last = p.marker() + p.zero_or_more(p.negate(p.marker()) + p.any()) + p.end();
            return p.zero_or_more(p.negate(last) + p.any()) + p.tag("m", p.marker());
        });
        auto res = parser.parse_anywhere_and_extract(str);
        return res.result.success() ? res.tags["m"] : "";
    };

    // Helper: find the first marker in a string
    auto find_first_marker = [](const std::string & str) -> std::string {
        auto parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
            return p.tag("m", p.marker());
        });
        auto res = parser.parse_anywhere_and_extract(str);
        return res.result.success() ? res.tags["m"] : "";
    };

    if (func_name_in_prefix != std::string::npos && func_name_in_suffix == std::string::npos) {
        // Function name is only in prefix - call_id is BETWEEN_FUNC_AND_ARGS or POST_ARGS
        // Check if args indicator "{" is in prefix or suffix
        size_t args_in_prefix = diff.prefix.find('{', func_name_in_prefix);
        size_t args_in_suffix = diff.suffix.find('{');

        if (args_in_suffix != std::string::npos &&
            (args_in_prefix == std::string::npos || args_in_prefix > diff.prefix.length())) {
            // Args are in suffix, so call_id is BETWEEN_FUNC_AND_ARGS
            call_id.pos = call_id_position::BETWEEN_FUNC_AND_ARGS;

            // Find call_id_prefix: marker immediately preceding common_id_part (no intervening markers)
            std::string after_func = diff.prefix.substr(func_name_in_prefix + func_name.length());
            auto id_prefix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.tag("prefix", p.marker()) +
                       p.zero_or_more(p.negate(p.marker()) + p.negate(p.literal(common_id_part)) + p.any()) +
                       p.literal(common_id_part);
            });
            auto id_res = id_prefix_parser.parse_anywhere_and_extract(after_func);
            if (id_res.result.success()) {
                call_id.prefix = id_res.tags["prefix"];
            } else {
                // Fallback: use the last marker in after_func
                call_id.prefix = find_last_marker(after_func);
            }

            // Extract call_id_suffix: the first marker in the suffix before args "{"
            auto suffix_parser = build_tagged_peg_parser([&](common_peg_parser_builder & p) {
                return p.zero_or_more(p.negate(p.marker()) + p.negate(p.literal("{")) + p.any()) +
                       p.tag("suffix", p.marker());
            });
            auto suf_res = suffix_parser.parse_anywhere_and_extract(diff.suffix);
            if (suf_res.result.success()) {
                call_id.suffix = suf_res.tags["suffix"];
            }
        } else if (args_in_prefix != std::string::npos) {
            // Args are in prefix, so call_id is POST_ARGS
            call_id.pos = call_id_position::POST_ARGS;

            // Extract last marker between args closing brace and the ID
            std::string after_args    = diff.prefix.substr(args_in_prefix);
            size_t      closing_brace = after_args.rfind('}');
            if (closing_brace != std::string::npos) {
                std::string between_args_and_id = after_args.substr(closing_brace + 1);
                call_id.prefix = find_last_marker(between_args_and_id);
            }

            // call_id_suffix: first marker in diff.suffix
            call_id.suffix = find_first_marker(diff.suffix);
        }
    } else if (func_name_in_suffix != std::string::npos && func_name_in_prefix == std::string::npos) {
        // Function name is only in suffix - call_id is PRE_FUNC_NAME
        call_id.pos = call_id_position::PRE_FUNC_NAME;

        // call_id_prefix: last marker in diff.prefix
        call_id.prefix = find_last_marker(diff.prefix);

        // call_id_suffix: first marker in the portion of diff.suffix before func_name
        std::string before_func = diff.suffix.substr(0, func_name_in_suffix);
        call_id.suffix = find_first_marker(before_func);
    }

    if (call_id.prefix == arguments.end) {
        call_id.prefix = "";
    }

    if (call_id.suffix == arguments.start) {
        call_id.suffix = "";
    }

    // When call_id is detected, per_call_end may have been incorrectly set to include
    // the call_id_suffix and sample args. Clear it if it starts with call_id_suffix.
    if (call_id.pos != call_id_position::NONE && !call_id.suffix.empty() &&
        format.per_call_end.find(call_id.suffix) == 0) {
        format.per_call_end.clear();
    }
}

}  // namespace autoparser

#include "common.h.inc"

#include "common.h.inc"
#include "ggml.h.inc"


#include "nlohmann/json.hpp.inc"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>

#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

static std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto               time       = std::chrono::system_clock::to_time_t(now);
    auto               local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    auto res = ss.str();
    return res;
}

static json safe_args_parse(const std::string & to_parse) {
    std::string stripped = to_parse;
    if (to_parse.at(0) == '"' && to_parse.at(to_parse.length() - 1) == '"') {
        stripped = to_parse.substr(1, to_parse.length() - 1);
    }
    try {
        return json::parse(stripped);
    } catch (json::exception & e) {
        return stripped;
    }
}

static std::string string_diff(const std::string & last, const std::string & current) {
    int status = -1;
    llama_common_unicode_string diff = llama_common_string_diff_rust(
        (const uint8_t *) last.data(), last.size(),
        (const uint8_t *) current.data(), current.size(),
        &status);
    if (status < 0) {
        throw std::runtime_error("Invalid diff: '" + last + "' not found at start of '" + current + "'");
    }
    if (!diff.data) {
        return "";
    }
    std::string result((const char *) diff.data, diff.len);
    llama_common_unicode_string_free(diff);
    return result;
}

json common_chat_msg::to_json_oaicompat(bool concat_typed_text) const {
    if (!content.empty() && !content_parts.empty()) {
        throw std::runtime_error("Cannot specify both content and content_parts");
    }
    json jmsg {
        {"role", role},
    };
    if (!content.empty()) {
        jmsg["content"] = content;
    } else if (!content_parts.empty()) {
        if (concat_typed_text) {
            std::string text;
            bool last_was_media_marker = false;
            // join parts with newline, do not add newline before or after media markers
            for (const auto & part : content_parts) {
                bool add_new_line = true;
                if (part.type == "text") {
                    add_new_line = !last_was_media_marker && !text.empty();
                    last_was_media_marker = false;
                } else if (part.type == "media_marker") {
                    add_new_line = false;
                    last_was_media_marker = true;
                } else {
                    LOG_WRN("Ignoring content part type: %s\n", part.type.c_str());
                    continue;
                }

                if (add_new_line) {
                    text += '\n';
                }

                text += part.text;
            }
            jmsg["content"] = text;
        } else {
            auto & parts = jmsg["content"] = json::array();
            for (const auto & part : content_parts) {
                parts.push_back({
                    {"type", part.type},
                    {"text", part.text},
                });
            }
        }
    } else {
        jmsg["content"] = "";
    }
    if (!reasoning_content.empty()) {
        jmsg["reasoning_content"] = reasoning_content;
    }
    if (!tool_name.empty()) {
        jmsg["name"] = tool_name;
    }
    if (!tool_call_id.empty()) {
        jmsg["tool_call_id"] = tool_call_id;
    }
    if (!tool_calls.empty()) {
        jmsg["tool_calls"] = json::array();
        auto & jtool_calls = jmsg["tool_calls"];
        for (const auto & tool_call : tool_calls) {
            json tc {
                {"type", "function"},
                {"function", {
                    {"name", tool_call.name},
                    {"arguments", json(tool_call.arguments)},
                }},
            };
            if (!tool_call.id.empty()) {
                tc["id"] = tool_call.id;
            }
            // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
            // We only generate a random id for the ones that don't generate one by themselves
            // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
            // {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
            jtool_calls.push_back(tc);
        }
    }

    return jmsg;
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::compute_diffs(const common_chat_msg & msg_prv,
                                                                      const common_chat_msg & msg_new) {
    std::vector<common_chat_msg_diff> diffs;
    if (msg_new.tool_calls.size() > msg_prv.tool_calls.size()) {
        diffs.reserve(msg_new.tool_calls.size() - msg_prv.tool_calls.size() + 3);
    } else {
        diffs.reserve(3);
    }

    // TODO: these can become expensive for long messages - how to optimize?
    if (msg_prv.reasoning_content != msg_new.reasoning_content) {
        auto & diff                  = diffs.emplace_back();
        diff.reasoning_content_delta = string_diff(msg_prv.reasoning_content, msg_new.reasoning_content);
    }
    if (msg_prv.content != msg_new.content) {
        auto & diff        = diffs.emplace_back();
        diff.content_delta = string_diff(msg_prv.content, msg_new.content);
    }

    if (msg_new.tool_calls.size() < msg_prv.tool_calls.size()) {
        std::string err = "Invalid diff: now finding less tool calls!\n";
        err += "  Previous (" + std::to_string(msg_prv.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_prv.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current (" + std::to_string(msg_new.tool_calls.size()) + "):\n";
        for (const auto & tc : msg_new.tool_calls) {
            err += "    - name: '" + tc.name + "', args: '" + tc.arguments + "'\n";
        }
        err += "  Current msg text content:\n" + msg_new.content + "\n";
        throw std::runtime_error(err);
    }

    if (!msg_prv.tool_calls.empty()) {
        const auto   idx  = msg_prv.tool_calls.size() - 1;
        const auto & pref = msg_prv.tool_calls[idx];
        const auto & newf = msg_new.tool_calls[idx];
        // Allow tool name to change during incremental parsing:
        // - empty -> non-empty (initial discovery)
        // - prefix -> longer string (name grows as more input is parsed)
        if (pref.name != newf.name && !pref.name.empty() && !newf.name.empty()) {
            // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
            bool is_prefix = (newf.name.rfind(pref.name, 0) == 0);
            if (!is_prefix) {
                LOG_ERR("Tool call mismatch: prev='%s' new='%s'\n", pref.name.c_str(), newf.name.c_str());
                throw std::runtime_error("Invalid diff: tool call mismatch!");
            }
        }
        const auto args_diff = string_diff(pref.arguments, newf.arguments);
        if (!args_diff.empty() || pref.id != newf.id || pref.name != newf.name) {
            auto & diff          = diffs.emplace_back();
            diff.tool_call_index = idx;
            if (pref.id != newf.id || pref.name != newf.name) {
                diff.tool_call_delta.id   = newf.id;
                diff.tool_call_delta.name = newf.name;
            }
            diff.tool_call_delta.arguments = args_diff;
        }
    }
    for (size_t idx = msg_prv.tool_calls.size(); idx < msg_new.tool_calls.size(); ++idx) {
        auto & diff          = diffs.emplace_back();
        diff.tool_call_index = idx;
        diff.tool_call_delta = msg_new.tool_calls[idx];
    }

    return diffs;
}

using chat_template_caps = jinja::caps;

struct common_chat_templates {
    bool add_bos;
    bool add_eos;
    bool has_explicit_template;  // Model had builtin template or template overridden was specified.
    std::unique_ptr<common_chat_template> template_default;  // always set (defaults to chatml)
    std::unique_ptr<common_chat_template> template_tool_use;
};

common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std::string & tool_choice) {
    const int parsed = llama_common_chat_tool_choice_parse_oaicompat_rust(tool_choice.c_str());
    if (parsed >= 0) {
        return (common_chat_tool_choice) parsed;
    }
    throw std::invalid_argument("Invalid tool_choice: " + tool_choice);
}

bool common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates) {
    common_chat_templates_inputs inputs;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    common_chat_msg msg;
    msg.role    = "user";
    msg.content = "test";
    inputs.messages = { msg };
    inputs.enable_thinking = true;
    inputs.add_generation_prompt = true;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;

    auto params = common_chat_templates_apply(chat_templates, inputs);
    return params.supports_thinking;
}

std::vector<common_chat_msg> common_chat_msgs_parse_oaicompat(const json & messages) {
    std::vector<common_chat_msg> msgs;

    try {
        if (!messages.is_array()) {
            throw std::invalid_argument("Expected 'messages' to be an array, got " + messages.dump());
        }

        for (const auto & message : messages) {
            if (!message.is_object()) {
                throw std::invalid_argument("Expected 'message' to be an object, got " + message.dump());
            }

            common_chat_msg msg;
            if (!message.contains("role")) {
                throw std::invalid_argument("Missing 'role' in message: " + message.dump());
            }
            msg.role = message.at("role");

            auto has_content    = message.contains("content");
            auto has_tool_calls = message.contains("tool_calls");
            if (has_content) {
                const auto & content = message.at("content");
                if (content.is_string()) {
                    msg.content = content;
                } else if (content.is_array()) {
                    for (const auto & part : content) {
                        if (!part.contains("type")) {
                            throw std::invalid_argument("Missing content part type: " + part.dump());
                        }
                        const auto & type = part.at("type");
                        if (type != "text" && type != "media_marker") {
                            throw std::invalid_argument("Unsupported content part type: " + type.dump());
                        }
                        common_chat_msg_content_part msg_part;
                        msg_part.type = type;
                        msg_part.text = part.at("text");
                        msg.content_parts.push_back(msg_part);
                    }
                } else if (!content.is_null()) {
                    throw std::invalid_argument("Invalid 'content' type: expected string or array, got " +
                                                content.dump() +
                                                " (ref: https://github.com/ggml-org/llama.cpp/issues/8367)");
                }
            }
            if (has_tool_calls) {
                for (const auto & tool_call : message.at("tool_calls")) {
                    common_chat_tool_call tc;
                    if (!tool_call.contains("type")) {
                        throw std::invalid_argument("Missing tool call type: " + tool_call.dump());
                    }
                    const auto & type = tool_call.at("type");
                    if (type != "function") {
                        throw std::invalid_argument("Unsupported tool call type: " + tool_call.dump());
                    }
                    if (!tool_call.contains("function")) {
                        throw std::invalid_argument("Missing tool call function: " + tool_call.dump());
                    }
                    const auto & fc = tool_call.at("function");
                    if (!fc.contains("name")) {
                        throw std::invalid_argument("Missing tool call name: " + tool_call.dump());
                    }
                    tc.name           = fc.at("name");
                    const auto & args = fc.at("arguments");
                    if (args.is_string()) {
                        tc.arguments = args;
                    } else {
                        tc.arguments = args.dump();
                    }
                    if (tool_call.contains("id")) {
                        tc.id = tool_call.at("id");
                    }
                    msg.tool_calls.push_back(tc);
                }
            }
            if (!has_content && !has_tool_calls) {
                throw std::invalid_argument(
                    "Expected 'content' or 'tool_calls' (ref: https://github.com/ggml-org/llama.cpp/issues/8367 & "
                    "https://github.com/ggml-org/llama.cpp/issues/12279)");
            }
            if (message.contains("reasoning_content")) {
                msg.reasoning_content = message.at("reasoning_content");
            }
            if (message.contains("name")) {
                msg.tool_name = message.at("name");
            }
            if (message.contains("tool_call_id")) {
                msg.tool_call_id = message.at("tool_call_id");
            }

            msgs.push_back(msg);
        }
    } catch (const std::exception & e) {
        // @ngxson : disable otherwise it's bloating the API response
        // printf("%s\n", std::string("; messages = ") + messages.dump(2));
        throw std::runtime_error("Failed to parse messages: " + std::string(e.what()));
    }

    return msgs;
}

static json render_message_to_json(const std::vector<common_chat_msg> & msgs, const jinja::caps & c) {
    if (!c.supports_string_content && !c.supports_typed_content) {
        LOG_WRN("%s: Neither string content nor typed content is supported by the template. This is unexpected and may lead to issues.\n", __func__);
    }

    bool only_string_accepted =  c.supports_string_content && !c.supports_typed_content;
    bool only_typed_accepted  = !c.supports_string_content &&  c.supports_typed_content;

    json messages = json::array();
    for (const auto & msg : msgs) {
        if (only_string_accepted) {
            json jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ true);
            messages.push_back(jmsg);
        } else if (only_typed_accepted) {
            json jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ false);
            if (jmsg.at("content").is_string()) {
                jmsg["content"] = json::array({
                    json{
                        {"type", "text"},
                        {"text", jmsg.at("content").get<std::string>()},
                    }
                });
            }
            messages.push_back(jmsg);
        } else {
            json jmsg = msg.to_json_oaicompat(/* concat_typed_text= */ false);
            messages.push_back(jmsg);
        }
    }
    return messages;
}

// DEPRECATED: only used in tests
json common_chat_msgs_to_json_oaicompat(const std::vector<common_chat_msg> & msgs, bool concat_typed_text) {
    jinja::caps c;
    c.supports_string_content = true;
    c.supports_typed_content = !concat_typed_text;
    return render_message_to_json(msgs, c);
}

json common_chat_tools_to_json_oaicompat(const std::vector<common_chat_tool> & tools) {
    if (tools.empty()) {
        return json();
    }

    auto result = json::array();
    for (const auto & tool : tools) {
        result.push_back({
            { "type",     "function" },
            { "function", {
                { "name", tool.name },
                { "description", tool.description },
                { "parameters", json::parse(tool.parameters) },
            }},
        });
    }
    return result;
}

std::vector<common_chat_tool> common_chat_tools_parse_oaicompat(const json & tools) {
    std::vector<common_chat_tool> result;

    try {
        if (!tools.is_null()) {
            if (!tools.is_array()) {
                throw std::invalid_argument("Expected 'tools' to be an array, got " + tools.dump());
            }
            for (const auto & tool : tools) {
                if (!tool.contains("type")) {
                    throw std::invalid_argument("Missing tool type: " + tool.dump());
                }
                const auto & type = tool.at("type");
                if (!type.is_string() || type != "function") {
                    throw std::invalid_argument("Unsupported tool type: " + tool.dump());
                }
                if (!tool.contains("function")) {
                    throw std::invalid_argument("Missing tool function: " + tool.dump());
                }

                const auto & function = tool.at("function");
                result.push_back({
                    /* .name = */ function.at("name"),
                    /* .description = */ function.value("description", ""),
                    /* .parameters = */ function.value("parameters", json::object()).dump(),
                });
            }
        }
    } catch (const std::exception & e) {
        throw std::runtime_error("Failed to parse tools: " + std::string(e.what()) + "; tools = " + tools.dump(2));
    }

    return result;
}

bool common_chat_verify_template(const std::string & tmpl, bool use_jinja) {
    if (use_jinja) {
        try {
            common_chat_msg msg;
            msg.role    = "user";
            msg.content = "test";

            auto tmpls = common_chat_templates_init(/* model= */ nullptr, tmpl);

            common_chat_templates_inputs inputs;
            inputs.messages = { msg };

            common_chat_templates_apply(tmpls.get(), inputs);
            return true;
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to apply template: %s\n", __func__, e.what());
            return false;
        }
    }
    llama_chat_message chat[] = {
        { "user", "test" }
    };
    const int res = llama_chat_apply_template(tmpl.c_str(), chat, 1, true, nullptr, 0);
    return res >= 0;
}

std::string common_chat_format_single(const struct common_chat_templates * tmpls,
                                      const std::vector<common_chat_msg> & past_msg,
                                      const common_chat_msg &              new_msg,
                                      bool                                 add_ass,
                                      bool                                 use_jinja) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_bos   = tmpls->add_bos;
    inputs.add_eos   = tmpls->add_eos;

    std::string fmt_past_msg;
    if (!past_msg.empty()) {
        inputs.messages              = past_msg;
        inputs.add_generation_prompt = false;
        fmt_past_msg                 = common_chat_templates_apply(tmpls, inputs).prompt;
    }
    std::ostringstream ss;
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    inputs.messages.push_back(new_msg);
    inputs.add_generation_prompt = add_ass;
    auto fmt_new_msg             = common_chat_templates_apply(tmpls, inputs).prompt;
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string common_chat_format_example(const struct common_chat_templates *       tmpls,
                                       bool                                       use_jinja,
                                       const std::map<std::string, std::string> & chat_template_kwargs) {
    common_chat_templates_inputs inputs;
    inputs.use_jinja            = use_jinja;
    inputs.add_bos              = tmpls->add_bos;
    inputs.add_eos              = tmpls->add_eos;
    inputs.chat_template_kwargs = chat_template_kwargs;
    auto add_simple_msg         = [&](auto role, auto content) {
        common_chat_msg msg;
        msg.role    = role;
        msg.content = content;
        inputs.messages.push_back(msg);
    };
    add_simple_msg("system", "You are a helpful assistant");
    add_simple_msg("user", "Hello");
    add_simple_msg("assistant", "Hi there");
    add_simple_msg("user", "How are you?");
    return common_chat_templates_apply(tmpls, inputs).prompt;
}

#undef FILENAME
#define FILENAME "jinja-lexer"

namespace jinja {

static void string_lstrip(std::string & s, const char * chars) {
    llama_common_unicode_string stripped =
        llama_common_string_lstrip_chars_rust((const uint8_t *) s.data(), s.size(), chars);
    s = common_rust_string_to_std(stripped);
}

static void string_rstrip(std::string & s, const char * chars) {
    llama_common_unicode_string stripped =
        llama_common_string_rstrip_chars_rust((const uint8_t *) s.data(), s.size(), chars);
    s = common_rust_string_to_std(stripped);
}

lexer_result lexer::tokenize(const std::string & source) {
    std::vector<token> tokens;

    // NOTE: do NOT transform the source string (i.e. preprocessing), as we need to keep
    //       the original character positions for error reporting etc.
    std::string src = source;

    if (source.empty()) {
        return {tokens, src};
    }

    // Normalize \r\n or \r to \n
    for (std::string::size_type pos = 0; (pos = src.find("\r\n", pos)) != std::string::npos; ) {
        src.erase(pos, 1);
        ++pos;
    }
    for (std::string::size_type pos = 0; (pos = src.find("\r", pos)) != std::string::npos; ) {
        src.replace(pos, 1, 1, '\n');
        ++pos;
    }

    // In the default configuration:
    //  - a single trailing newline is stripped if present
    //  - other whitespace (spaces, tabs, newlines etc.) is returned unchanged
    if (source.back() == '\n') {
        src.pop_back();
    }

    size_t pos = 0;
    size_t start_pos = 0;
    size_t curly_bracket_depth = 0;

    using pred = std::function<bool(char)>;
    auto consume_while = [&](const pred & predicate) -> std::string {
        std::string str;
        while (predicate(src[pos])) {
            // check for escape char
            if (src[pos] == '\\') {
                // consume backslash
                ++pos;
                // check for end of input
                if (pos >= src.size()) {
                    throw lexer_exception("unexpected end of input after escape character", source, pos);
                }
                // add escaped char
                char escaped_char = src[pos++];
                if (escape_chars.find(escaped_char) == escape_chars.end()) {
                    throw lexer_exception(std::string("unknown escape character \\") + escaped_char, source, pos);
                }
                char unescaped_char = escape_chars.at(escaped_char);
                str += unescaped_char;
                continue;
            }

            str += src[pos++];
            if (pos > src.size()) {
                throw lexer_exception("unexpected end of input during consume_while", source, pos);
            }
        }
        return str;
    };

    auto consume_numeric = [&]() -> std::string {
        std::string num = consume_while([](char c) {
            return llama_common_jinja_is_integer_rust((uint8_t) c);
        });
        if (pos < src.size() && src[pos] == '.' && pos + 1 < src.size() &&
                llama_common_jinja_is_integer_rust((uint8_t) src[pos + 1])) {
            ++pos; // Consume '.'
            std::string frac = consume_while([](char c) {
                return llama_common_jinja_is_integer_rust((uint8_t) c);
            });
            num += "." + frac;
        }
        return num;
    };

    auto next_pos_is = [&](std::initializer_list<char> chars, size_t n = 1) -> bool {
        if (pos + n >= src.size()) return false;
        for (char c : chars) {
            if (src[pos + n] == c) return true;
        }
        return false;
    };

    // note: default config for chat template: lstrip_blocks = true, trim_blocks = true

    // text\n[space]{block} --> text\n{block}
    bool opt_lstrip_blocks = true;

    // {block}\n[space]text --> {block}[space]text
    bool opt_trim_blocks = true;

    // options set dynamically based on current/last block
    bool is_lstrip_block = false; // example: {%-
    bool is_rstrip_block = false; // example: -%}

    while (pos < src.size()) {
        start_pos = pos;
        // JJ_DEBUG("lexer main loop at pos %zu: '%s...'", pos, src.substr(pos, 10).c_str());

        // First, consume all text that is outside of a Jinja statement or expression
        token::type last_token_type = tokens.empty()
                                            ? token::close_statement // initial state
                                            : tokens.back().t;
        if (last_token_type == token::close_statement ||
            last_token_type == token::close_expression ||
            last_token_type == token::comment) {

            bool last_block_can_rm_newline = false;
            is_rstrip_block = false;
            if (pos > 3) {
                char c0 = src[pos - 3];
                char c1 = src[pos - 2];
                char c2 = src[pos - 1];
                // strip if: -[%}#]}text
                is_rstrip_block = c0 == '-'
                                    && (c1 == '%' || c1 == '}' || c1 == '#')
                                    && c2 == '}';
                // match behavior of hf.js: exclude {{ and }} cases, regex: ([#%-]})
                last_block_can_rm_newline = (c1 == '#' || c1 == '%' || c1 == '-') && c2 == '}';
            }

            size_t start = pos;
            size_t end = start;
            while (pos < src.size() &&
                    // Keep going until we hit the next Jinja statement or expression
                    !(
                        src[pos] == '{' &&
                        next_pos_is( {'%', '{', '#'} )
                    )) {
                end = ++pos;
            }

            // equivalent to hf.js code: template.replace(/^[ \t]*({[#%-])/gm, "$1");
            if (opt_lstrip_blocks && src[pos] == '{' && next_pos_is({'%', '#', '-'})) {
                size_t current = end;
                while (current > start) {
                    char c = src[current - 1];
                    if (current == 1) {
                        end = 0; // Trim from the start of the string
                        break;
                    }
                    if (c == '\n') {
                        end = current; // Trim from the start of the line
                        break;
                    }
                    if (!std::isspace(static_cast<unsigned char>(c))) {
                        break; // Found non-whitespace before newline, keep
                    }
                    --current;
                }
            }

            std::string text = src.substr(start, end - start);

            // equivalent to hf.js code: template.replace(/([#%-]})\n/g, "$1");
            if (opt_trim_blocks && last_block_can_rm_newline) {
                if (!text.empty() && text.front() == '\n') {
                    text.erase(text.begin());
                }
            }

            if (is_rstrip_block) {
                // example: {last_block}[space]text
                // doing lstrip on text, effectively rstrip the LAST block
                // JJ_DEBUG("RSTRIP block detected, current text: '%s'", text.c_str());
                string_lstrip(text, " \t\r\n");
            }

            is_lstrip_block = src[pos] == '{' && next_pos_is({'{', '%', '#'}) && next_pos_is({'-'}, 2);
            if (is_lstrip_block) {
                // example: text[space]{current_block}
                // doing rstrip on text, effectively lstrip the CURRENT block
                // JJ_DEBUG("LSTRIP block detected, current text: '%s'", text.c_str());
                string_rstrip(text, " \t\r\n");
            }

            if (!text.empty()) {
                // JJ_DEBUG("consumed text: '%s'", text.c_str());
                tokens.push_back({token::text, text, start_pos});
                continue;
            }
        }

        // Possibly consume a comment
        // TODO: handle lstrip/rstrip for comments? (not important for now)
        if (src[pos] == '{' && next_pos_is( {'#'} )) {
            start_pos = pos;
            pos += 2; // Skip the opening {#
            std::string comment;
            while (!(src[pos] == '#' && next_pos_is( {'}'} ))) {
                if (pos + 2 >= src.size()) {
                    throw lexer_exception("missing end of comment tag", source, pos);
                }
                comment += src[pos++];
            }
            JJ_DEBUG("consumed comment: '%s'", comment.c_str());
            tokens.push_back({token::comment, comment, start_pos});
            pos += 2; // Skip the closing #}
            continue;
        }

        if (src[pos] == '-' && (
                last_token_type == token::open_expression ||
                last_token_type == token::open_statement)
        ) {
            JJ_DEBUG("lexer main loop at pos %zu: '%s...'", pos, src.substr(pos, 10).c_str());
            pos++; // consume '-' in {%- or {{-
            if (pos >= src.size()) break;
        }

        // Consume (and ignore) all whitespace inside Jinja statements or expressions
        consume_while([](char c) { return std::isspace(static_cast<unsigned char>(c)); });

        if (pos >= src.size()) break;

        char ch = src[pos];

        bool is_closing_block = ch == '-' && next_pos_is( {'%', '}'} );

        // Check for unary operators
        if (!is_closing_block && (ch == '-' || ch == '+')) {
            start_pos = pos;
            token::type last_token_type = tokens.empty() ? token::eof : tokens.back().t;
            if (last_token_type == token::text || last_token_type == token::eof) {
                throw lexer_exception(std::string("unexpected character: ") + ch, source, pos);
            }
            switch (last_token_type) {
                case token::identifier:
                case token::numeric_literal:
                case token::string_literal:
                case token::close_paren:
                case token::close_square_bracket:
                    // Part of a binary operator
                    // a - 1, 1 - 1, true - 1, "apple" - 1, (1) - 1, a[1] - 1
                    // Continue parsing normally
                    break;
                default: {
                    // Is part of a unary operator
                    // (-1), [-1], (1 + -1), not -1, -apple
                    ++pos; // Consume the operator

                    // Check for numbers following the unary operator
                    std::string num = consume_numeric();
                    std::string value = std::string(1, ch) + num;
                    token::type t = num.empty() ? token::unary_operator : token::numeric_literal;
                    // JJ_DEBUG("consumed unary operator or numeric literal: '%s'", value.c_str());
                    tokens.push_back({t, value, start_pos});
                    continue;
                }
            }
        }

        // Try to match one of the tokens in the mapping table
        bool matched = false;
        for (const auto & [seq, typ] : ordered_mapping_table) {
            start_pos = pos;
            // Inside an object literal, don't treat "}}" as expression-end
            if (seq == "}}" && curly_bracket_depth > 0) {
                continue;
            }
            if (pos + seq.size() <= src.size() && src.substr(pos, seq.size()) == seq) {
                tokens.push_back({typ, seq, start_pos});
                if (typ == token::open_expression) {
                    curly_bracket_depth = 0;
                } else if (typ == token::open_curly_bracket) {
                    ++curly_bracket_depth;
                } else if (typ == token::close_curly_bracket) {
                    --curly_bracket_depth;
                }

                pos += seq.size();
                matched = true;
                break; // continue main loop
            }
        }
        if (matched) continue; // continue main loop

        // Strings
        if (ch == '\'' || ch == '"') {
            start_pos = pos;
            ++pos; // Skip opening quote
            std::string str = consume_while([ch](char c) { return c != ch; });
            // JJ_DEBUG("consumed string literal: '%s'", str.c_str());
            tokens.push_back({token::string_literal, str, start_pos});
            ++pos; // Skip closing quote
            continue;
        }

        // Numbers
        if (llama_common_jinja_is_integer_rust((uint8_t) ch)) {
            start_pos = pos;
            std::string num = consume_numeric();
            // JJ_DEBUG("consumed numeric literal: '%s'", num.c_str());
            tokens.push_back({token::numeric_literal, num, start_pos});
            continue;
        }

        // Identifiers
        if (llama_common_jinja_is_word_rust((uint8_t) ch)) {
            start_pos = pos;
            std::string word = consume_while([](char c) {
                return llama_common_jinja_is_word_rust((uint8_t) c);
            });
            // JJ_DEBUG("consumed identifier: '%s'", word.c_str());
            tokens.push_back({token::identifier, word, start_pos});
            continue;
        }

        throw lexer_exception(std::string("unexpected character: ") + ch, source, pos);
    }

    return {std::move(tokens), src};
}

} // namespace jinja

#include "common.h.inc"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#undef FILENAME
#define FILENAME "jinja-parser"

namespace jinja {

// parse from a list of tokens into an AST (program)
program parse_from_tokens(const lexer_result & lexer_res);

struct parser_exception : public std::runtime_error {
    parser_exception(const std::string & msg, const std::string & source, size_t pos)
        : std::runtime_error([&]() {
            const char * tag = "parser";
            llama_common_unicode_string result = llama_common_fmt_error_with_source_rust(
                    reinterpret_cast<const uint8_t *>(tag), std::strlen(tag),
                    reinterpret_cast<const uint8_t *>(msg.data()), msg.size(),
                    reinterpret_cast<const uint8_t *>(source.data()), source.size(), pos);
            std::string value(reinterpret_cast<const char *>(result.data), result.len);
            llama_common_unicode_string_free(result);
            return value;
        }()) {}
};

// Helper to check type without asserting (useful for logic)
template<typename T>
static bool is_type(const statement_ptr & ptr) {
    return dynamic_cast<const T*>(ptr.get()) != nullptr;
}

class parser {
    const std::vector<token> & tokens;
    size_t current = 0;

    std::string source; // for error reporting

public:
    parser(const std::vector<token> & t, const std::string & src) : tokens(t), source(src) {}

    program parse() {
        statements body;
        while (current < tokens.size()) {
            body.push_back(parse_any());
        }
        return program(std::move(body));
    }

    // NOTE: start_pos is the token index, used for error reporting
    template<typename T, typename... Args>
    std::unique_ptr<T> mk_stmt(size_t start_pos, Args&&... args) {
        auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
        assert(start_pos < tokens.size());
        ptr->pos = tokens[start_pos].pos;
        return ptr;
    }

private:
    const token & peek(size_t offset = 0) const {
        if (current + offset >= tokens.size()) {
            static const token end_token{token::eof, "", 0};
            return end_token;
        }
        return tokens[current + offset];
    }

    const token & next() {
        if (current >= tokens.size()) {
            throw parser_exception("Parser Error: Unexpected EOF", source, tokens.empty() ? 0 : tokens.back().pos);
        }
        return tokens[current++];
    }

    token expect(token::type type, const std::string&  error) {
        const auto & t = peek();
        if (t.t != type) {
            throw parser_exception("Parser Error: " + error + " (Got " + t.value + ")", source, t.pos);
        }
        current++;
        return t;
    }

    void expect_identifier(const std::string & name) {
        const auto & t = peek();
        if (t.t != token::identifier || t.value != name) {
            throw parser_exception("Expected identifier: " + name, source, t.pos);
        }
        current++;
    }

    bool is(token::type type) const {
        return peek().t == type;
    }

    bool is_identifier(const std::string & name) const {
        return peek().t == token::identifier && peek().value == name;
    }

    bool is_statement(const std::vector<std::string> & names) const {
        if (peek(0).t != token::open_statement || peek(1).t != token::identifier) {
            return false;
        }
        std::string val = peek(1).value;
        return std::find(names.begin(), names.end(), val) != names.end();
    }

    statement_ptr parse_any() {
        size_t start_pos = current;
        switch (peek().t) {
            case token::comment:
                return mk_stmt<comment_statement>(start_pos, next().value);
            case token::text:
                return mk_stmt<string_literal>(start_pos, next().value);
            case token::open_statement:
                return parse_jinja_statement();
            case token::open_expression:
                return parse_jinja_expression();
            default:
                throw std::runtime_error("Unexpected token type");
        }
    }

    statement_ptr parse_jinja_expression() {
        // Consume {{ }} tokens
        expect(token::open_expression, "Expected {{");
        auto result = parse_expression();
        expect(token::close_expression, "Expected }}");
        return result;
    }

    statement_ptr parse_jinja_statement() {
        // Consume {% token
        expect(token::open_statement, "Expected {%");

        if (peek().t != token::identifier) {
            throw std::runtime_error("Unknown statement");
        }

        size_t start_pos = current;
        std::string name = next().value;

        statement_ptr result;
        if (name == "set") {
            result = parse_set_statement(start_pos);

        } else if (name == "if") {
            result = parse_if_statement(start_pos);
            // expect {% endif %}
            expect(token::open_statement, "Expected {%");
            expect_identifier("endif");
            expect(token::close_statement, "Expected %}");

        } else if (name == "macro") {
            result = parse_macro_statement(start_pos);
            // expect {% endmacro %}
            expect(token::open_statement, "Expected {%");
            expect_identifier("endmacro");
            expect(token::close_statement, "Expected %}");

        } else if (name == "for") {
            result = parse_for_statement(start_pos);
            // expect {% endfor %}
            expect(token::open_statement, "Expected {%");
            expect_identifier("endfor");
            expect(token::close_statement, "Expected %}");

        } else if (name == "break") {
            expect(token::close_statement, "Expected %}");
            result = mk_stmt<break_statement>(start_pos);

        } else if (name == "continue") {
            expect(token::close_statement, "Expected %}");
            result = mk_stmt<continue_statement>(start_pos);

        } else if (name == "call") {
            statements caller_args;
            // bool has_caller_args = false;
            if (is(token::open_paren)) {
                // Optional caller arguments, e.g. {% call(user) dump_users(...) %}
                caller_args = parse_args();
                // has_caller_args = true;
            }
            auto callee = parse_primary_expression();
            if (!is_type<identifier>(callee)) throw std::runtime_error("Expected identifier");

            auto call_args = parse_args();
            expect(token::close_statement, "Expected %}");

            statements body;
            while (!is_statement({"endcall"})) {
                body.push_back(parse_any());
            }

            expect(token::open_statement, "Expected {%");
            expect_identifier("endcall");
            expect(token::close_statement, "Expected %}");

            auto call_expr = mk_stmt<call_expression>(start_pos, std::move(callee), std::move(call_args));
            result = mk_stmt<call_statement>(start_pos, std::move(call_expr), std::move(caller_args), std::move(body));

        } else if (name == "filter") {
            auto filter_node = parse_primary_expression();
            if (is_type<identifier>(filter_node) && is(token::open_paren)) {
                filter_node = parse_call_expression(std::move(filter_node));
            }
            expect(token::close_statement, "Expected %}");

            statements body;
            while (!is_statement({"endfilter"})) {
                body.push_back(parse_any());
            }

            expect(token::open_statement, "Expected {%");
            expect_identifier("endfilter");
            expect(token::close_statement, "Expected %}");
            result = mk_stmt<filter_statement>(start_pos, std::move(filter_node), std::move(body));

        } else if (name == "generation" || name == "endgeneration") {
            // Ignore generation blocks (transformers-specific)
            // See https://github.com/huggingface/transformers/pull/30650 for more information.
            result = mk_stmt<noop_statement>(start_pos);
            ++current;

        } else {
            throw std::runtime_error("Unknown statement: " + name);
        }
        return result;
    }

    statement_ptr parse_set_statement(size_t start_pos) {
        // NOTE: `set` acts as both declaration statement and assignment expression
        auto left = parse_expression_sequence();
        statement_ptr value = nullptr;
        statements body;

        if (is(token::equals)) {
            ++current;
            value = parse_expression_sequence();
        } else {
            // parsing multiline set here
            expect(token::close_statement, "Expected %}");
            while (!is_statement({"endset"})) {
                body.push_back(parse_any());
            }
            expect(token::open_statement, "Expected {%");
            expect_identifier("endset");
        }
        expect(token::close_statement, "Expected %}");
        return mk_stmt<set_statement>(start_pos, std::move(left), std::move(value), std::move(body));
    }

    statement_ptr parse_if_statement(size_t start_pos) {
        auto test = parse_expression();
        expect(token::close_statement, "Expected %}");

        statements body;
        statements alternate;

        // Keep parsing 'if' body until we reach the first {% elif %} or {% else %} or {% endif %}
        while (!is_statement({"elif", "else", "endif"})) {
            body.push_back(parse_any());
        }

        if (is_statement({"elif"})) {
            size_t pos0 = current;
            ++current; // consume {%
            ++current; // consume 'elif'
            alternate.push_back(parse_if_statement(pos0)); // nested If
        } else if (is_statement({"else"})) {
            ++current; // consume {%
            ++current; // consume 'else'
            expect(token::close_statement, "Expected %}");

            // keep going until we hit {% endif %}
            while (!is_statement({"endif"})) {
                alternate.push_back(parse_any());
            }
        }
        return mk_stmt<if_statement>(start_pos, std::move(test), std::move(body), std::move(alternate));
    }

    statement_ptr parse_macro_statement(size_t start_pos) {
        auto name = parse_primary_expression();
        auto args = parse_args();
        expect(token::close_statement, "Expected %}");
        statements body;
        // Keep going until we hit {% endmacro
        while (!is_statement({"endmacro"})) {
            body.push_back(parse_any());
        }
        return mk_stmt<macro_statement>(start_pos, std::move(name), std::move(args), std::move(body));
    }

    statement_ptr parse_expression_sequence(bool primary = false) {
        size_t start_pos = current;
        statements exprs;
        exprs.push_back(primary ? parse_primary_expression() : parse_expression());
        bool is_tuple = is(token::comma);
        while (is(token::comma)) {
            ++current; // consume comma
            exprs.push_back(primary ? parse_primary_expression() : parse_expression());
        }
        return is_tuple ? mk_stmt<tuple_literal>(start_pos, std::move(exprs)) : std::move(exprs[0]);
    }

    statement_ptr parse_for_statement(size_t start_pos) {
        // e.g., `message` in `for message in messages`
        auto loop_var = parse_expression_sequence(true); // should be an identifier/tuple
        if (!is_identifier("in")) throw std::runtime_error("Expected 'in'");
        ++current; // consume 'in'

        // `messages` in `for message in messages`
        auto iterable = parse_expression();
        expect(token::close_statement, "Expected %}");

        statements body;
        statements alternate;

        // Keep going until we hit {% endfor or {% else
        while (!is_statement({"endfor", "else"})) {
            body.push_back(parse_any());
        }

        if (is_statement({"else"})) {
            ++current; // consume {%
            ++current; // consume 'else'
            expect(token::close_statement, "Expected %}");
            while (!is_statement({"endfor"})) {
                alternate.push_back(parse_any());
            }
        }
        return mk_stmt<for_statement>(
            start_pos,
            std::move(loop_var), std::move(iterable),
            std::move(body), std::move(alternate));
    }

    statement_ptr parse_expression() {
        // Choose parse function with lowest precedence
        return parse_if_expression();
    }

    statement_ptr parse_if_expression() {
        auto a = parse_logical_or_expression();
        if (is_identifier("if")) {
            // Ternary expression
            size_t start_pos = current;
            ++current; // consume 'if'
            auto test = parse_logical_or_expression();
            if (is_identifier("else")) {
                // Ternary expression with else
                size_t pos0 = current;
                ++current; // consume 'else'
                auto false_expr = parse_if_expression(); // recurse to support chained ternaries
                return mk_stmt<ternary_expression>(pos0, std::move(test), std::move(a), std::move(false_expr));
            } else {
                // Select expression on iterable
                return mk_stmt<select_expression>(start_pos, std::move(a), std::move(test));
            }
        }
        return a;
    }

    statement_ptr parse_logical_or_expression() {
        auto left = parse_logical_and_expression();
        while (is_identifier("or")) {
            size_t start_pos = current;
            token op = next();
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_logical_and_expression());
        }
        return left;
    }

    statement_ptr parse_logical_and_expression() {
        auto left = parse_logical_negation_expression();
        while (is_identifier("and")) {
            size_t start_pos = current;
            auto op = next();
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_logical_negation_expression());
        }
        return left;
    }

    statement_ptr parse_logical_negation_expression() {
        // Try parse unary operators
        if (is_identifier("not")) {
            size_t start_pos = current;
            auto op = next();
            return mk_stmt<unary_expression>(start_pos, op, parse_logical_negation_expression());
        }
        return parse_comparison_expression();
    }

    statement_ptr parse_comparison_expression() {
        // NOTE: membership has same precedence as comparison
        // e.g., ('a' in 'apple' == 'b' in 'banana') evaluates as ('a' in ('apple' == ('b' in 'banana')))
        auto left = parse_additive_expression();
        while (true) {
            token op;
            size_t start_pos = current;
            if (is_identifier("not") && peek(1).t == token::identifier && peek(1).value == "in") {
                op = {token::identifier, "not in", tokens[current].pos};
                ++current; // consume 'not'
                ++current; // consume 'in'
            } else if (is_identifier("in")) {
                op = next();
            } else if (is(token::comparison_binary_operator)) {
                op = next();
            } else break;
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_additive_expression());
        }
        return left;
    }

    statement_ptr parse_additive_expression() {
        auto left = parse_multiplicative_expression();
        while (is(token::additive_binary_operator)) {
            size_t start_pos = current;
            auto op = next();
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_multiplicative_expression());
        }
        return left;
    }

    statement_ptr parse_multiplicative_expression() {
        auto left = parse_test_expression();
        while (is(token::multiplicative_binary_operator)) {
            size_t start_pos = current;
            auto op = next();
            left = mk_stmt<binary_expression>(start_pos, op, std::move(left), parse_test_expression());
        }
        return left;
    }

    statement_ptr parse_test_expression() {
        auto operand = parse_filter_expression();
        while (is_identifier("is")) {
            size_t start_pos = current;
            ++current; // consume 'is'
            bool negate = false;
            if (is_identifier("not")) { ++current; negate = true; }
            auto test_id = parse_primary_expression();
            // FIXME: tests can also be expressed like this: if x is eq 3
            if (is(token::open_paren)) test_id = parse_call_expression(std::move(test_id));
            operand = mk_stmt<test_expression>(start_pos, std::move(operand), negate, std::move(test_id));
        }
        return operand;
    }

    statement_ptr parse_filter_expression() {
        auto operand = parse_call_member_expression();
        while (is(token::pipe)) {
            size_t start_pos = current;
            ++current; // consume pipe
            auto filter = parse_primary_expression();
            if (is(token::open_paren)) filter = parse_call_expression(std::move(filter));
            operand = mk_stmt<filter_expression>(start_pos, std::move(operand), std::move(filter));
        }
        return operand;
    }

    statement_ptr parse_call_member_expression() {
        // Handle member expressions recursively
        auto member = parse_member_expression(parse_primary_expression());
        return is(token::open_paren)
            ? parse_call_expression(std::move(member)) // foo.x()
            : std::move(member);
    }

    statement_ptr parse_call_expression(statement_ptr callee) {
        size_t start_pos = current;
        auto expr = mk_stmt<call_expression>(start_pos, std::move(callee), parse_args());
        auto member = parse_member_expression(std::move(expr)); // foo.x().y
        return is(token::open_paren)
            ? parse_call_expression(std::move(member)) // foo.x()()
            : std::move(member);
    }

    statements parse_args() {
        // comma-separated arguments list
        expect(token::open_paren, "Expected (");
        statements args;
        while (!is(token::close_paren)) {
            statement_ptr arg;
            // unpacking: *expr
            if (peek().t == token::multiplicative_binary_operator && peek().value == "*") {
                size_t start_pos = current;
                ++current; // consume *
                arg = mk_stmt<spread_expression>(start_pos, parse_expression());
            } else {
                arg = parse_expression();
                if (is(token::equals)) {
                    // keyword argument
                    // e.g., func(x = 5, y = a or b)
                    size_t start_pos = current;
                    ++current; // consume equals
                    arg = mk_stmt<keyword_argument_expression>(start_pos, std::move(arg), parse_expression());
                }
            }
            args.push_back(std::move(arg));
            if (is(token::comma)) {
                ++current; // consume comma
            }
        }
        expect(token::close_paren, "Expected )");
        return args;
    }

    statement_ptr parse_member_expression(statement_ptr object) {
        size_t start_pos = current;
        while (is(token::dot) || is(token::open_square_bracket)) {
            auto op = next();
            bool computed = op.t == token::open_square_bracket;
            statement_ptr prop;
            if (computed) {
                prop = parse_member_expression_arguments();
                expect(token::close_square_bracket, "Expected ]");
            } else {
                prop = parse_primary_expression();
            }
            object = mk_stmt<member_expression>(start_pos, std::move(object), std::move(prop), computed);
        }
        return object;
    }

    statement_ptr parse_member_expression_arguments() {
        // NOTE: This also handles slice expressions colon-separated arguments list
        // e.g., ['test'], [0], [:2], [1:], [1:2], [1:2:3]
        statements slices;
        bool is_slice = false;
        size_t start_pos = current;
        while (!is(token::close_square_bracket)) {
            if (is(token::colon)) {
                // A case where a default is used
                // e.g., [:2] will be parsed as [undefined, 2]
                slices.push_back(nullptr);
                ++current; // consume colon
                is_slice = true;
            } else {
                slices.push_back(parse_expression());
                if (is(token::colon)) {
                    ++current; // consume colon after expression, if it exists
                    is_slice = true;
                }
            }
        }
        if (is_slice) {
            statement_ptr start = slices.size() > 0 ? std::move(slices[0]) : nullptr;
            statement_ptr stop = slices.size() > 1 ? std::move(slices[1]) : nullptr;
            statement_ptr step = slices.size() > 2 ? std::move(slices[2]) : nullptr;
            return mk_stmt<slice_expression>(start_pos, std::move(start), std::move(stop), std::move(step));
        }
        if (slices.empty()) {
            return mk_stmt<blank_expression>(start_pos);
        }
        return std::move(slices[0]);
    }

    statement_ptr parse_primary_expression() {
        size_t start_pos = current;
        auto t = next();
        switch (t.t) {
            case token::numeric_literal:
                if (t.value.find('.') != std::string::npos) {
                    return mk_stmt<float_literal>(start_pos, std::stod(t.value));
                } else {
                    return mk_stmt<integer_literal>(start_pos, std::stoll(t.value));
                }
            case token::string_literal: {
                std::string val = t.value;
                while (is(token::string_literal)) {
                    val += next().value;
                }
                return mk_stmt<string_literal>(start_pos, val);
            }
            case token::identifier:
                return mk_stmt<identifier>(start_pos, t.value);
            case token::open_paren: {
                auto expr = parse_expression_sequence();
                expect(token::close_paren, "Expected )");
                return expr;
            }
            case token::open_square_bracket: {
                statements vals;
                while (!is(token::close_square_bracket)) {
                    vals.push_back(parse_expression());
                    if (is(token::comma)) ++current;
                }
                ++current;
                return mk_stmt<array_literal>(start_pos, std::move(vals));
            }
            case token::open_curly_bracket: {
                std::vector<std::pair<statement_ptr, statement_ptr>> pairs;
                while (!is(token::close_curly_bracket)) {
                    auto key = parse_expression();
                    expect(token::colon, "Expected :");
                    pairs.push_back({std::move(key), parse_expression()});
                    if (is(token::comma)) ++current;
                }
                ++current;
                return mk_stmt<object_literal>(start_pos, std::move(pairs));
            }
            default:
                throw std::runtime_error("Unexpected token: " + t.value + " of type " + std::to_string(t.t));
        }
    }
};

program parse_from_tokens(const lexer_result & lexer_res) {
    return parser(lexer_res.tokens, lexer_res.source).parse();
}

} // namespace jinja

#include "common.h.inc"

#include <string>
#include <vector>
#include <memory>
#include <cmath>

#undef FILENAME
#define FILENAME "jinja-runtime"

bool g_jinja_debug = false;

namespace jinja {

void enable_debug(bool enable) {
    g_jinja_debug = enable;
}

static value_string exec_statements(const statements & stmts, context & ctx) {
    auto result = mk_val<value_array>();
    for (const auto & stmt : stmts) {
        JJ_DEBUG("Executing statement of type %s", stmt->type().c_str());
        result->push_back(stmt->execute(ctx));
    }
    // convert to string parts
    value_string str = mk_val<value_string>();
    gather_string_parts_recursive(result, str);
    return str;
}

static void ensure_key_type_allowed(const value & val) {
    if (!val->is_hashable()) {
        throw std::runtime_error("Type: " + val->type() + " is not allowed as object key");
    }
}

// execute with error handling
value statement::execute(context & ctx) {
    try {
        return execute_impl(ctx);
    } catch (const continue_statement::signal & /* ex */) {
        throw;
    } catch (const break_statement::signal & /* ex */) {
        throw;
    } catch (const rethrown_exception & /* ex */) {
        throw;
    } catch (const not_implemented_exception & /* ex */) {
        throw;
    } catch (const std::exception & e) {
        const std::string & source = *ctx.src;
        if (source.empty()) {
            std::ostringstream oss;
            oss << "\nError executing " << type() << " at position " << pos << ": " << e.what();
            throw rethrown_exception(oss.str());
        } else {
            std::ostringstream oss;
            oss << "\n------------\n";
            oss << "While executing " << type() << " at "
                << common_rust_string_to_std(llama_common_get_line_col_rust(
                        (const uint8_t *) source.data(), source.size(), pos))
                << " in source:\n";
            llama_common_unicode_string peak = llama_common_peak_source_rust(
                    reinterpret_cast<const uint8_t *>(source.data()), source.size(), pos, 40);
            oss << std::string(reinterpret_cast<const char *>(peak.data), peak.len) << "\n";
            llama_common_unicode_string_free(peak);
            oss << "Error: " << e.what();
            // throw as another exception to avoid repeated formatting
            throw rethrown_exception(oss.str());
        }
    }
}

value identifier::execute_impl(context & ctx) {
    auto it = ctx.get_val(val);
    auto builtins = global_builtins();
    if (!it->is_undefined()) {
        if (ctx.is_get_stats) {
            value_t::stats_t::mark_used(it);
        }
        JJ_DEBUG("Identifier '%s' found, type = %s", val.c_str(), it->type().c_str());
        return it;
    } else if (builtins.find(val) != builtins.end()) {
        JJ_DEBUG("Identifier '%s' found in builtins", val.c_str());
        return mk_val<value_func>(val, builtins.at(val));
    } else {
        JJ_DEBUG("Identifier '%s' not found, returning undefined", val.c_str());
        return mk_val<value_undefined>(val);
    }
}

value object_literal::execute_impl(context & ctx) {
    auto obj = mk_val<value_object>();
    for (const auto & pair : val) {
        value key = pair.first->execute(ctx);
        value val = pair.second->execute(ctx);
        JJ_DEBUG("Object literal: setting key '%s' with value type %s", key->as_string().str().c_str(), val->type().c_str());
        obj->insert(key, val);
    }
    return obj;
}

value binary_expression::execute_impl(context & ctx) {
    value left_val = left->execute(ctx);

    // Logical operators
    if (op.value == "and") {
        JJ_DEBUG("Executing logical test: %s AND %s", left->type().c_str(), right->type().c_str());
        return left_val->as_bool() ? right->execute(ctx) : std::move(left_val);
    } else if (op.value == "or") {
        JJ_DEBUG("Executing logical test: %s OR %s", left->type().c_str(), right->type().c_str());
        return left_val->as_bool() ? std::move(left_val) : right->execute(ctx);
    }

    // Equality operators
    value right_val = right->execute(ctx);
    JJ_DEBUG("Executing binary expression %s '%s' %s", left_val->type().c_str(), op.value.c_str(), right_val->type().c_str());
    if (op.value == "==") {
        return mk_val<value_bool>(*left_val == *right_val);
    } else if (op.value == "!=") {
        return mk_val<value_bool>(llama_common_jinja_bool_not_rust(*left_val == *right_val));
    }

    auto workaround_concat_null_with_str = [&](value & res) -> bool {
        bool is_left_null  = left_val->is_none()  || left_val->is_undefined();
        bool is_right_null = right_val->is_none() || right_val->is_undefined();
        bool is_left_str   = is_val<value_string>(left_val);
        bool is_right_str  = is_val<value_string>(right_val);
        if ((is_left_null && is_right_str) || (is_right_null && is_left_str)) {
            JJ_DEBUG("%s", "Workaround: treating null/undefined as empty string for string concatenation");
            string left_str  = is_left_null  ? string() : left_val->as_string();
            string right_str = is_right_null ? string() : right_val->as_string();
            auto output = left_str.append(right_str);
            res = mk_val<value_string>(std::move(output));
            return true;
        }
        return false;
    };

    auto test_is_in = [&]() -> bool {
        func_args args(ctx);
        args.push_back(left_val);
        args.push_back(right_val);
        return global_builtins().at("test_is_in")(args)->as_bool();
    };

    // Handle undefined and null values
    if (is_val<value_undefined>(left_val) || is_val<value_undefined>(right_val)) {
        if (is_val<value_undefined>(right_val) && (op.value == "in" || op.value == "not in")) {
            // Special case: `anything in undefined` is `false` and `anything not in undefined` is `true`
            return mk_val<value_bool>(llama_common_jinja_membership_result_rust(false, op.value == "not in"));
        }
        if (op.value == "+" || op.value == "~") {
            value res = mk_val<value_undefined>();
            if (workaround_concat_null_with_str(res)) {
                return res;
            }
        }
        throw std::runtime_error("Cannot perform operation " + op.value + " on undefined values");
    } else if (is_val<value_none>(left_val) || is_val<value_none>(right_val)) {
        if (op.value == "+" || op.value == "~") {
            value res = mk_val<value_undefined>();
            if (workaround_concat_null_with_str(res)) {
                return res;
            }
        }
        throw std::runtime_error("Cannot perform operation on null values");
    }

    // Float operations
    if ((is_val<value_int>(left_val) || is_val<value_float>(left_val)) &&
        (is_val<value_int>(right_val) || is_val<value_float>(right_val))) {
        double a = left_val->as_float();
        double b = right_val->as_float();
        if (op.value == "+" || op.value == "-" || op.value == "*") {
            double res = (op.value == "+") ? a + b : (op.value == "-") ? a - b : a * b;
            JJ_DEBUG("Arithmetic operation: %f %s %f = %f", a, op.value.c_str(), b, res);
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return mk_val<value_float>(res);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(res));
            }
        } else if (op.value == "/") {
            JJ_DEBUG("Division operation: %f / %f", a, b);
            return mk_val<value_float>(a / b);
        } else if (op.value == "%") {
            double rem = std::fmod(a, b);
            JJ_DEBUG("Modulo operation: %f %% %f = %f", a, b, rem);
            bool is_float = is_val<value_float>(left_val) || is_val<value_float>(right_val);
            if (is_float) {
                return mk_val<value_float>(rem);
            } else {
                return mk_val<value_int>(static_cast<int64_t>(rem));
            }
        } else if (op.value == "<") {
            JJ_DEBUG("Comparison operation: %f < %f is %d", a, b, a < b);
            return mk_val<value_bool>(llama_common_jinja_compare_f64_rust(a, b, value_compare_op::lt));
        } else if (op.value == ">") {
            JJ_DEBUG("Comparison operation: %f > %f is %d", a, b, a > b);
            return mk_val<value_bool>(llama_common_jinja_compare_f64_rust(a, b, value_compare_op::gt));
        } else if (op.value == ">=") {
            JJ_DEBUG("Comparison operation: %f >= %f is %d", a, b, a >= b);
            return mk_val<value_bool>(llama_common_jinja_compare_f64_rust(a, b, value_compare_op::ge));
        } else if (op.value == "<=") {
            JJ_DEBUG("Comparison operation: %f <= %f is %d", a, b, a <= b);
            return mk_val<value_bool>(llama_common_jinja_compare_f64_rust(a, b, 5));
        }
    }

    // Array operations
    if (is_val<value_array>(left_val) && is_val<value_array>(right_val)) {
        if (op.value == "+") {
            auto & left_arr = left_val->as_array();
            auto & right_arr = right_val->as_array();
            auto result = mk_val<value_array>();
            for (const auto & item : left_arr) {
                result->push_back(item);
            }
            for (const auto & item : right_arr) {
                result->push_back(item);
            }
            return result;
        }
    } else if (is_val<value_array>(right_val)) {
        // case: 1 in [0, 1, 2]
        bool member = test_is_in();
        if (op.value == "in") {
            return mk_val<value_bool>(llama_common_jinja_membership_result_rust(member, false));
        } else if (op.value == "not in") {
            return mk_val<value_bool>(llama_common_jinja_membership_result_rust(member, true));
        }
    }

    // String concatenation with ~ and +
    if ((is_val<value_string>(left_val) || is_val<value_string>(right_val)) &&
            (op.value == "~" || op.value == "+")) {
        JJ_DEBUG("String concatenation with %s operator", op.value.c_str());
        auto output = left_val->as_string().append(right_val->as_string());
        auto res = mk_val<value_string>();
        res->val_str = std::move(output);
        return res;
    }

    // Python-style string repetition
    // TODO: support array/tuple repetition (e.g., [1, 2] * 3 → [1, 2, 1, 2, 1, 2])
    if (op.value == "*" &&
            ((is_val<value_string>(left_val) && is_val<value_int>(right_val)) ||
             (is_val<value_int>(left_val) && is_val<value_string>(right_val)))) {
        const auto & str = is_val<value_string>(left_val) ? left_val->as_string() : right_val->as_string();
        const int64_t repeat = is_val<value_int>(right_val) ? right_val->as_int() : left_val->as_int();
        auto res = mk_val<value_string>();
        if (repeat <= 0) {
            return res;
        }
        for (int64_t i = 0; i < repeat; ++i) {
            res->val_str = res->val_str.append(str);
        }
        return res;
    }

    // String membership
    if (is_val<value_string>(left_val) && is_val<value_string>(right_val)) {
        // case: "a" in "abc"
        bool member = test_is_in();
        if (op.value == "in") {
            return mk_val<value_bool>(llama_common_jinja_membership_result_rust(member, false));
        } else if (op.value == "not in") {
            return mk_val<value_bool>(llama_common_jinja_membership_result_rust(member, true));
        }
    }

    // Value key in object
    if (is_val<value_object>(right_val)) {
        // case: key in {key: value}
        bool member = test_is_in();
        if (op.value == "in") {
            return mk_val<value_bool>(llama_common_jinja_membership_result_rust(member, false));
        } else if (op.value == "not in") {
            return mk_val<value_bool>(llama_common_jinja_membership_result_rust(member, true));
        }
    }

    throw std::runtime_error("Unknown operator \"" + op.value + "\" between " + left_val->type() + " and " + right_val->type());
}

static value try_builtin_func(context & ctx, const std::string & name, value & input, bool undef_on_missing = false) {
    JJ_DEBUG("Trying built-in function '%s' for type %s", name.c_str(), input->type().c_str());
    if (ctx.is_get_stats) {
        value_t::stats_t::mark_used(input);
        input->stats.ops.insert(name);
    }
    auto builtins = input->get_builtins();
    auto it = builtins.find(name);
    if (it != builtins.end()) {
        JJ_DEBUG("Binding built-in '%s'", name.c_str());
        return mk_val<value_func>(name, it->second, input);
    }
    if (undef_on_missing) {
        return mk_val<value_undefined>(name);
    }
    throw std::runtime_error("Unknown (built-in) filter '" + name + "' for type " + input->type());
}

value filter_expression::execute_impl(context & ctx) {
    value input = operand ? operand->execute(ctx) : val;

    JJ_DEBUG("Applying filter to %s", input->type().c_str());

    if (is_stmt<identifier>(filter)) {
        auto filter_id = cast_stmt<identifier>(filter)->val;

        if (filter_id == "trim") {
            filter_id = "strip"; // alias
        }
        JJ_DEBUG("Applying filter '%s' to %s", filter_id.c_str(), input->type().c_str());
        // TODO: Refactor filters so this coercion can be done automatically
        if (!input->is_undefined() && !is_val<value_string>(input) && (
            filter_id == "capitalize" ||
            filter_id == "lower" ||
            filter_id == "replace" ||
            filter_id == "strip" ||
            filter_id == "title" ||
            filter_id == "upper" ||
            filter_id == "wordcount"
        )) {
            JJ_DEBUG("Coercing %s to String for '%s' filter", input->type().c_str(), filter_id.c_str());
            input = mk_val<value_string>(input->as_string());
        }
        return try_builtin_func(ctx, filter_id, input)->invoke(func_args(ctx));

    } else if (is_stmt<call_expression>(filter)) {
        auto call = cast_stmt<call_expression>(filter);
        if (!is_stmt<identifier>(call->callee)) {
            throw std::runtime_error("Filter callee must be an identifier");
        }
        auto filter_id = cast_stmt<identifier>(call->callee)->val;

        if (filter_id == "trim") {
            filter_id = "strip"; // alias
        }
        JJ_DEBUG("Applying filter '%s' with arguments to %s", filter_id.c_str(), input->type().c_str());
        func_args args(ctx);
        for (const auto & arg_expr : call->args) {
            args.push_back(arg_expr->execute(ctx));
        }

        return try_builtin_func(ctx, filter_id, input)->invoke(args);

    } else {
        throw std::runtime_error("Invalid filter expression");
    }
}

value filter_statement::execute_impl(context & ctx) {
    // eval body as string, then apply filter
    auto body_val = exec_statements(body, ctx);
    value_string parts = mk_val<value_string>();
    gather_string_parts_recursive(body_val, parts);

    JJ_DEBUG("FilterStatement: applying filter to body string of length %zu", parts->val_str.length());
    filter_expression filter_expr(std::move(parts), std::move(filter));
    value out = filter_expr.execute(ctx);

    // this node can be reused later, make sure filter is preserved
    this->filter = std::move(filter_expr.filter);
    return out;
}

value test_expression::execute_impl(context & ctx) {
    // NOTE: "value is something" translates to function call "test_is_something(value)"
    const auto & builtins = global_builtins();

    std::string test_id;
    value input = operand->execute(ctx);

    func_args args(ctx);
    args.push_back(input);

    if (is_stmt<identifier>(test)) {
        test_id = cast_stmt<identifier>(test)->val;
    } else if (is_stmt<call_expression>(test)) {
        auto call = cast_stmt<call_expression>(test);
        if (!is_stmt<identifier>(call->callee)) {
            throw std::runtime_error("Test callee must be an identifier");
        }
        test_id = cast_stmt<identifier>(call->callee)->val;

        JJ_DEBUG("Applying test '%s' with arguments to %s", test_id.c_str(), input->type().c_str());
        for (const auto & arg_expr : call->args) {
            args.push_back(arg_expr->execute(ctx));
        }

    } else {
        throw std::runtime_error("Invalid test expression");
    }

    auto it = builtins.find("test_is_" + test_id);
    JJ_DEBUG("Test expression %s '%s' %s (using function 'test_is_%s')", operand->type().c_str(), test_id.c_str(), negate ? "(negate)" : "", test_id.c_str());
    if (it == builtins.end()) {
        throw std::runtime_error("Unknown test '" + test_id + "'");
    }

    auto res = it->second(args);

    if (negate) {
        return mk_val<value_bool>(llama_common_jinja_bool_not_rust(res->as_bool()));
    } else {
        return res;
    }
}

value unary_expression::execute_impl(context & ctx) {
    value operand_val = argument->execute(ctx);
    JJ_DEBUG("Executing unary expression with operator '%s'", op.value.c_str());

    if (op.value == "not") {
        return mk_val<value_bool>(llama_common_jinja_bool_not_rust(operand_val->as_bool()));
    } else if (op.value == "-") {
        if (is_val<value_int>(operand_val)) {
            return mk_val<value_int>(-operand_val->as_int());
        } else if (is_val<value_float>(operand_val)) {
            return mk_val<value_float>(-operand_val->as_float());
        } else {
            throw std::runtime_error("Unary - operator requires numeric operand");
        }
    }

    throw std::runtime_error("Unknown unary operator '" + op.value + "'");
}

value if_statement::execute_impl(context & ctx) {
    value test_val = test->execute(ctx);

    auto out = mk_val<value_array>();
    if (test_val->as_bool()) {
        for (auto & stmt : body) {
            JJ_DEBUG("IF --> Executing THEN body, current block: %s", stmt->type().c_str());
            out->push_back(stmt->execute(ctx));
        }
    } else {
        for (auto & stmt : alternate) {
            JJ_DEBUG("IF --> Executing ELSE body, current block: %s", stmt->type().c_str());
            out->push_back(stmt->execute(ctx));
        }
    }
    // convert to string parts
    value_string str = mk_val<value_string>();
    gather_string_parts_recursive(out, str);
    return str;
}

value for_statement::execute_impl(context & ctx) {
    context scope(ctx); // new scope for loop variables

    jinja::select_expression * select_expr = cast_stmt<select_expression>(iterable);
    statement_ptr test_expr_nullptr;

    statement_ptr & iter_expr = [&]() -> statement_ptr & {
        auto tmp = cast_stmt<select_expression>(iterable);
        return tmp ? tmp->lhs : iterable;
    }();
    statement_ptr & test_expr = [&]() -> statement_ptr & {
        auto tmp = cast_stmt<select_expression>(iterable);
        return tmp ? tmp->test : test_expr_nullptr;
    }();

    JJ_DEBUG("Executing for statement, iterable type: %s", iter_expr->type().c_str());

    value iterable_val = iter_expr->execute(scope);

    // mark the variable being iterated as used for stats
    if (ctx.is_get_stats) {
        value_t::stats_t::mark_used(iterable_val);
        iterable_val->stats.ops.insert("array_access");
    }

    if (iterable_val->is_undefined()) {
        JJ_DEBUG("%s", "For loop iterable is undefined, skipping loop");
        iterable_val = mk_val<value_array>();
    }

    if (!is_val<value_array>(iterable_val) && !is_val<value_object>(iterable_val)) {
        throw std::runtime_error("Expected iterable or object type in for loop: got " + iterable_val->type());
    }

    std::vector<value> items;
    if (is_val<value_object>(iterable_val)) {
        JJ_DEBUG("%s", "For loop over object keys");
        auto & obj = iterable_val->as_ordered_object();
        for (auto & p : obj) {
            auto tuple = mk_val<value_tuple>(p);
            items.push_back(std::move(tuple));
        }
        if (ctx.is_get_stats) {
            value_t::stats_t::mark_used(iterable_val);
            iterable_val->stats.ops.insert("object_access");
        }
    } else {
        JJ_DEBUG("%s", "For loop over array items");
        auto & arr = iterable_val->as_array();
        for (const auto & item : arr) {
            items.push_back(item);
        }
        if (ctx.is_get_stats) {
            value_t::stats_t::mark_used(iterable_val);
            iterable_val->stats.ops.insert("array_access");
        }
    }

    std::vector<std::function<void(context &)>> scope_update_fns;

    std::vector<value> filtered_items;
    for (size_t i = 0; i < items.size(); ++i) {
        context loop_scope(scope);

        value current = items[i];

        std::function<void(context&)> scope_update_fn = [](context &) { /* no-op */};
        if (is_stmt<identifier>(loopvar)) {
            auto id = cast_stmt<identifier>(loopvar)->val;

            if (is_val<value_object>(iterable_val)) {
                // case example: {% for key in dict %}
                current = items[i]->as_array()[0];
                scope_update_fn = [id, &items, i](context & ctx) {
                    ctx.set_val(id, items[i]->as_array()[0]);
                };
            } else {
                // case example: {% for item in list %}
                scope_update_fn = [id, &items, i](context & ctx) {
                    ctx.set_val(id, items[i]);
                };
            }

        } else if (is_stmt<tuple_literal>(loopvar)) {
            // case example: {% for key, value in dict %}
            auto tuple = cast_stmt<tuple_literal>(loopvar);
            if (!is_val<value_array>(current)) {
                throw std::runtime_error("Cannot unpack non-iterable type: " + current->type());
            }
            auto & c_arr = current->as_array();
            if (tuple->val.size() != c_arr.size()) {
                throw std::runtime_error(std::string("Too ") + (tuple->val.size() > c_arr.size() ? "few" : "many") + " items to unpack");
            }
            scope_update_fn = [tuple, &items, i](context & ctx) {
                auto & c_arr = items[i]->as_array();
                for (size_t j = 0; j < tuple->val.size(); ++j) {
                    if (!is_stmt<identifier>(tuple->val[j])) {
                        throw std::runtime_error("Cannot unpack non-identifier type: " + tuple->val[j]->type());
                    }
                    auto id = cast_stmt<identifier>(tuple->val[j])->val;
                    ctx.set_val(id, c_arr[j]);
                }
            };

        } else {
            throw std::runtime_error("Invalid loop variable(s): " + loopvar->type());
        }

        if (select_expr && test_expr) {
            scope_update_fn(loop_scope);
            value test_val = test_expr->execute(loop_scope);
            if (!test_val->as_bool()) {
                continue;
            }
        }
        JJ_DEBUG("For loop: adding item type %s at index %zu", current->type().c_str(), i);
        filtered_items.push_back(current);
        scope_update_fns.push_back(scope_update_fn);
    }
    JJ_DEBUG("For loop: %zu items after filtering", filtered_items.size());

    auto result = mk_val<value_array>();

    bool noIteration = true;
    for (size_t i = 0; i < filtered_items.size(); i++) {
        JJ_DEBUG("For loop iteration %zu/%zu", i + 1, filtered_items.size());
        value_object loop_obj = mk_val<value_object>();
        loop_obj->has_builtins = false; // loop object has no builtins
        loop_obj->insert("index", mk_val<value_int>(i + 1));
        loop_obj->insert("index0", mk_val<value_int>(i));
        loop_obj->insert("revindex", mk_val<value_int>(filtered_items.size() - i));
        loop_obj->insert("revindex0", mk_val<value_int>(filtered_items.size() - i - 1));
        loop_obj->insert("first", mk_val<value_bool>(i == 0));
        loop_obj->insert("last", mk_val<value_bool>(i == filtered_items.size() - 1));
        loop_obj->insert("length", mk_val<value_int>(filtered_items.size()));
        loop_obj->insert("previtem", i > 0 ? filtered_items[i - 1] : mk_val<value_undefined>("previtem"));
        loop_obj->insert("nextitem", i < filtered_items.size() - 1 ? filtered_items[i + 1] : mk_val<value_undefined>("nextitem"));
        scope.set_val("loop", loop_obj);
        scope_update_fns[i](scope);
        try {
            for (auto & stmt : body) {
                value val = stmt->execute(scope);
                result->push_back(val);
            }
        } catch (const continue_statement::signal &) {
            continue;
        } catch (const break_statement::signal &) {
            break;
        }
        noIteration = false;
    }

    JJ_DEBUG("For loop complete, total iterations: %zu", filtered_items.size());
    if (noIteration) {
        for (auto & stmt : default_block) {
            value val = stmt->execute(ctx);
            result->push_back(val);
        }
    }

    // convert to string parts
    value_string str = mk_val<value_string>();
    gather_string_parts_recursive(result, str);
    return str;
}

value set_statement::execute_impl(context & ctx) {
    auto rhs = val ? val->execute(ctx) : exec_statements(body, ctx);

    if (is_stmt<identifier>(assignee)) {
        // case: {% set my_var = value %}
        auto var_name = cast_stmt<identifier>(assignee)->val;
        JJ_DEBUG("Setting global variable '%s' with value type %s", var_name.c_str(), rhs->type().c_str());
        ctx.set_val(var_name, rhs);

    } else if (is_stmt<tuple_literal>(assignee)) {
        // case: {% set a, b = value %}
        auto tuple = cast_stmt<tuple_literal>(assignee);
        if (!is_val<value_array>(rhs)) {
            throw std::runtime_error("Cannot unpack non-iterable type in set: " + rhs->type());
        }
        auto & arr = rhs->as_array();
        if (arr.size() != tuple->val.size()) {
            throw std::runtime_error(std::string("Too ") + (tuple->val.size() > arr.size() ? "few" : "many") + " items to unpack in set");
        }
        for (size_t i = 0; i < tuple->val.size(); ++i) {
            auto & elem = tuple->val[i];
            if (!is_stmt<identifier>(elem)) {
                throw std::runtime_error("Cannot unpack to non-identifier in set: " + elem->type());
            }
            auto var_name = cast_stmt<identifier>(elem)->val;
            ctx.set_val(var_name, arr[i]);
        }

    } else if (is_stmt<member_expression>(assignee)) {
        // case: {% set ns.my_var = value %}
        auto member = cast_stmt<member_expression>(assignee);
        if (member->computed) {
            throw std::runtime_error("Cannot assign to computed member");
        }
        if (!is_stmt<identifier>(member->property)) {
            throw std::runtime_error("Cannot assign to member with non-identifier property");
        }
        auto prop_name = cast_stmt<identifier>(member->property)->val;

        value object = member->object->execute(ctx);
        if (!is_val<value_object>(object)) {
            throw std::runtime_error("Cannot assign to member of non-object");
        }
        auto obj_ptr = cast_val<value_object>(object);
        JJ_DEBUG("Setting object property '%s' with value type %s", prop_name.c_str(), rhs->type().c_str());
        obj_ptr->insert(prop_name, rhs);

    } else {
        throw std::runtime_error("Invalid LHS inside assignment expression: " + assignee->type());
    }
    return mk_val<value_undefined>();
}

value macro_statement::execute_impl(context & ctx) {
    if (!is_stmt<identifier>(this->name)) {
        throw std::runtime_error("Macro name must be an identifier");
    }
    std::string name = cast_stmt<identifier>(this->name)->val;

    const func_handler func = [this, name, &ctx](const func_args & args) -> value {
        size_t expected_count = this->args.size();
        size_t input_count = args.count();

        JJ_DEBUG("Invoking macro '%s' with %zu input arguments (expected %zu)", name.c_str(), input_count, expected_count);
        context macro_ctx(ctx); // new scope for macro execution

        // bind parameters
        for (size_t i = 0; i < expected_count; ++i) {
            if (i < input_count) {
                if (is_stmt<identifier>(this->args[i])) {
                    // normal parameter
                    std::string param_name = cast_stmt<identifier>(this->args[i])->val;
                    value param_value = args.get_kwarg_or_pos(param_name, i);
                    JJ_DEBUG("  Binding parameter '%s' to argument of type %s", param_name.c_str(), param_value->type().c_str());
                    macro_ctx.set_val(param_name, param_value);
                } else if (is_stmt<keyword_argument_expression>(this->args[i])) {
                    // default argument used as normal parameter
                    auto kwarg = cast_stmt<keyword_argument_expression>(this->args[i]);
                    if (!is_stmt<identifier>(kwarg->key)) {
                        throw std::runtime_error("Keyword argument key must be an identifier in macro '" + name + "'");
                    }
                    std::string param_name = cast_stmt<identifier>(kwarg->key)->val;
                    value param_value = args.get_kwarg_or_pos(param_name, i);
                    JJ_DEBUG("  Binding parameter '%s' to argument of type %s", param_name.c_str(), param_value->type().c_str());
                    macro_ctx.set_val(param_name, param_value);
                } else {
                    throw std::runtime_error("Invalid parameter type in macro '" + name + "'");
                }
            } else {
                auto & default_arg = this->args[i];
                if (is_stmt<keyword_argument_expression>(default_arg)) {
                    auto kwarg = cast_stmt<keyword_argument_expression>(default_arg);
                    if (!is_stmt<identifier>(kwarg->key)) {
                        throw std::runtime_error("Keyword argument key must be an identifier in macro '" + name + "'");
                    }
                    std::string param_name = cast_stmt<identifier>(kwarg->key)->val;
                    JJ_DEBUG("  Binding parameter '%s' to default argument of type %s", param_name.c_str(), kwarg->val->type().c_str());
                    macro_ctx.set_val(param_name, kwarg->val->execute(ctx));
                } else {
                    throw std::runtime_error("Not enough arguments provided to macro '" + name + "'");
                }
                //std::string param_name = cast_stmt<identifier>(default_args[i])->val;
                //JJ_DEBUG("  Binding parameter '%s' to default", param_name.c_str());
                //macro_ctx.var[param_name] = default_args[i]->execute(ctx);
            }
        }

        // execute macro body
        JJ_DEBUG("Executing macro '%s' body with %zu statements", name.c_str(), this->body.size());
        auto res = exec_statements(this->body, macro_ctx);
        JJ_DEBUG("Macro '%s' execution complete, result: %s", name.c_str(), res->val_str.str().c_str());
        return res;
    };

    JJ_DEBUG("Defining macro '%s' with %zu parameters", name.c_str(), args.size());
    ctx.set_val(name, mk_val<value_func>(name, func));
    return mk_val<value_undefined>();
}

value member_expression::execute_impl(context & ctx) {
    value object = this->object->execute(ctx);

    value property;
    if (this->computed) {
        // syntax: obj[expr]
        JJ_DEBUG("Member expression, computing property type %s", this->property->type().c_str());

        int64_t arr_size = 0;
        if (is_val<value_array>(object)) {
            arr_size = object->as_array().size();
        } else if (is_val<value_string>(object)) {
            arr_size = object->as_string().length();
        }

        if (is_stmt<slice_expression>(this->property)) {
            auto s = cast_stmt<slice_expression>(this->property);
            value start_val = s->start_expr ? s->start_expr->execute(ctx) : mk_val<value_int>(0);
            value stop_val  = s->stop_expr  ? s->stop_expr->execute(ctx)  : mk_val<value_int>(arr_size);
            value step_val  = s->step_expr  ? s->step_expr->execute(ctx)  : mk_val<value_int>(1);

            // translate to function call: obj.slice(start, stop, step)
            JJ_DEBUG("Member expression is a slice: start %s, stop %s, step %s",
                     start_val->as_repr().c_str(),
                     stop_val->as_repr().c_str(),
                     step_val->as_repr().c_str());
            auto slice_func = try_builtin_func(ctx, "slice", object);
            func_args args(ctx);
            args.push_back(start_val);
            args.push_back(stop_val);
            args.push_back(step_val);
            return slice_func->invoke(args);
        } else {
            property = this->property->execute(ctx);
        }
    } else {
        // syntax: obj.prop
        if (!is_stmt<identifier>(this->property)) {
            throw std::runtime_error("Static member property must be an identifier");
        }
        property = mk_val<value_string>(cast_stmt<identifier>(this->property)->val);
        std::string prop = property->as_string().str();
        JJ_DEBUG("Member expression, object type %s, static property '%s'", object->type().c_str(), prop.c_str());

        // behavior of jinja2: obj having prop as a built-in function AND 'prop', as an object key,
        // then obj.prop returns the built-in function, not the property value.
        // while obj['prop'] returns the property value.
        // example: {"obj": {"items": 123}} -> obj.items is the built-in function, obj['items'] is 123

        value val = try_builtin_func(ctx, prop, object, true);
        if (!is_val<value_undefined>(val)) {
            return val;
        }
        // else, fallthrough to normal property access below
    }

    JJ_DEBUG("Member expression on object type %s, property type %s", object->type().c_str(), property->type().c_str());
    value val = mk_val<value_undefined>("object_property");

    if (property->is_undefined()) {
        JJ_DEBUG("%s", "Member expression property is undefined, returning undefined");
        return val;
    }

    ensure_key_type_allowed(property);

    if (is_val<value_undefined>(object)) {
        JJ_DEBUG("%s", "Accessing property on undefined object, returning undefined");
        return val;

    } else if (is_val<value_object>(object)) {
        auto key = property->as_string().str();
        val = object->at(property, val);
        if (is_val<value_undefined>(val)) {
            val = try_builtin_func(ctx, key, object, true);
        }
        JJ_DEBUG("Accessed property '%s' value, got type: %s", key.c_str(), val->type().c_str());

    } else if (is_val<value_array>(object) || is_val<value_string>(object)) {
        if (is_val<value_int>(property)) {
            int64_t index = property->as_int();
            JJ_DEBUG("Accessing %s index %d", object->type().c_str(), (int)index);
            if (is_val<value_array>(object)) {
                auto & arr = object->as_array();
                if (index < 0) {
                    index += static_cast<int64_t>(arr.size());
                }
                if (index >= 0 && index < static_cast<int64_t>(arr.size())) {
                    val = arr[index];
                }
            } else { // value_string
                auto str = object->as_string().str();
                if (index >= 0 && index < static_cast<int64_t>(str.size())) {
                    val = mk_val<value_string>(std::string(1, str[index]));
                }
            }

        } else if (is_val<value_string>(property)) {
            auto key = property->as_string().str();
            JJ_DEBUG("Accessing %s built-in '%s'", is_val<value_array>(object) ? "array" : "string", key.c_str());
            val = try_builtin_func(ctx, key, object, true);

        } else {
            throw std::runtime_error("Cannot access property with non-string/non-number: got " + property->type());
        }
    } else {
        if (!is_val<value_string>(property)) {
            throw std::runtime_error("Cannot access property with non-string: got " + property->type());
        }
        auto key = property->as_string().str();
        val = try_builtin_func(ctx, key, object, true);
    }

    if (ctx.is_get_stats && val && object && property) {
        value_t::stats_t::mark_used(val);
        value_t::stats_t::mark_used(object);
        value_t::stats_t::mark_used(property);
        if (is_val<value_int>(property)) {
            object->stats.ops.insert("array_access");
        } else if (is_val<value_string>(property)) {
            object->stats.ops.insert("object_access");
        }
    }

    return val;
}

value call_expression::execute_impl(context & ctx) {
    // gather arguments
    func_args args(ctx);
    for (auto & arg_stmt : this->args) {
        auto arg_val = arg_stmt->execute(ctx);
        JJ_DEBUG("  Argument type: %s", arg_val->type().c_str());
        args.push_back(arg_val);
    }
    // execute callee
    value callee_val = callee->execute(ctx);
    if (!is_val<value_func>(callee_val)) {
        throw std::runtime_error("Callee is not a function: got " + callee_val->type());
    }
    auto * callee_func = cast_val<value_func>(callee_val);
    JJ_DEBUG("Calling function '%s' with %zu arguments", callee_func->name.c_str(), args.count());
    return callee_func->invoke(args);
}

value keyword_argument_expression::execute_impl(context & ctx) {
    if (!is_stmt<identifier>(key)) {
        throw std::runtime_error("Keyword argument key must be identifiers");
    }

    std::string k = cast_stmt<identifier>(key)->val;
    JJ_DEBUG("Keyword argument expression key: %s, value: %s", k.c_str(), val->type().c_str());

    value v = val->execute(ctx);
    JJ_DEBUG("Keyword argument value executed, type: %s", v->type().c_str());

    return mk_val<value_kwarg>(k, v);
}

} // namespace jinja

#include "common.h.inc"
#include "common.h.inc"

// for converting from JSON to jinja values
#include <nlohmann/json.hpp.inc>

#include <sstream>
#include <string>
#include <cctype>
#include <vector>
#include <optional>
#include <algorithm>

#undef FILENAME
#define FILENAME "jinja-value"

namespace jinja {

// func_args method implementations

value func_args::get_kwarg(const std::string & key, value default_val) const {
    for (const auto & arg : args) {
        if (is_val<value_kwarg>(arg)) {
            auto * kwarg = cast_val<value_kwarg>(arg);
            if (kwarg->key == key) {
                return kwarg->val;
            }
        }
    }
    return default_val;
}

value func_args::get_kwarg_or_pos(const std::string & key, size_t pos) const {
    value val = get_kwarg(key, mk_val<value_undefined>());

    if (val->is_undefined() && pos < count() && !is_val<value_kwarg>(args[pos])) {
        return args[pos];
    }

    return val;
}

value func_args::get_pos(size_t pos) const {
    if (count() > pos) {
        return args[pos];
    }
    throw raised_exception("Function '" + func_name + "' expected at least " + std::to_string(pos + 1) + " arguments, got " + std::to_string(count()));
}

value func_args::get_pos(size_t pos, value default_val) const {
    if (count() > pos) {
        return args[pos];
    }
    return default_val;
}

void func_args::push_back(const value & val) {
    args.push_back(val);
}

void func_args::push_front(const value & val) {
    args.insert(args.begin(), val);
}

const std::vector<value> & func_args::get_args() const {
    return args;
}

/**
 * Function that mimics Python's array slicing.
 */
template<typename T>
static T slice(const T & array, int64_t start, int64_t stop, int64_t step = 1) {
    int64_t len = static_cast<int64_t>(array.size());
    int64_t direction = (step > 0) ? 1 : ((step < 0) ? -1 : 0);
    int64_t start_val = 0;
    int64_t stop_val = 0;
    if (direction >= 0) {
        start_val = start;
        if (start_val < 0) {
            start_val = std::max(len + start_val, (int64_t)0);
        } else {
            start_val = std::min(start_val, len);
        }

        stop_val = stop;
        if (stop_val < 0) {
            stop_val = std::max(len + stop_val, (int64_t)0);
        } else {
            stop_val = std::min(stop_val, len);
        }
    } else {
        start_val = len - 1;
        if (start_val < 0) {
            start_val = std::max(len + start_val, (int64_t)-1);
        } else {
            start_val = std::min(start_val, len - 1);
        }

        stop_val = -1;
        if (stop_val < -1) {
            stop_val = std::max(len + stop_val, (int64_t)-1);
        } else {
            stop_val = std::min(stop_val, len - 1);
        }
    }
    T result;
    if (direction == 0) {
        return result;
    }
    for (int64_t i = start_val; direction * i < direction * stop_val; i += step) {
        if (i >= 0 && i < len) {
            result.push_back(array[static_cast<size_t>(i)]);
        }
    }
    return result;
}

template<typename T>
static value empty_value_fn(const func_args &) {
    if constexpr (std::is_same_v<T, value_int>) {
        return mk_val<T>(0);
    } else if constexpr (std::is_same_v<T, value_float>) {
        return mk_val<T>(0.0);
    } else if constexpr (std::is_same_v<T, value_bool>) {
        return mk_val<T>(false);
    } else {
        return mk_val<T>();
    }
}
template<typename T>
static value test_type_fn(const func_args & args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.get_pos(0));
    JJ_DEBUG("test_type_fn: type=%s result=%d", typeid(T).name(), is_type ? 1 : 0);
    return mk_val<value_bool>(is_type);
}
template<typename T, typename U>
static value test_type_fn(const func_args & args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.get_pos(0)) || is_val<U>(args.get_pos(0));
    JJ_DEBUG("test_type_fn: type=%s or %s result=%d", typeid(T).name(), typeid(U).name(), is_type ? 1 : 0);
    return mk_val<value_bool>(is_type);
}
template<typename T, typename U, typename V>
static value test_type_fn(const func_args & args) {
    args.ensure_count(1);
    bool is_type = is_val<T>(args.get_pos(0)) || is_val<U>(args.get_pos(0)) || is_val<V>(args.get_pos(0));
    JJ_DEBUG("test_type_fn: type=%s, %s or %s result=%d", typeid(T).name(), typeid(U).name(), typeid(V).name(), is_type ? 1 : 0);
    return mk_val<value_bool>(is_type);
}
template<value_compare_op op>
static value test_compare_fn(const func_args & args) {
    args.ensure_count(2, 2);
    return mk_val<value_bool>(value_compare(args.get_pos(0), args.get_pos(1), op));
}

static std::string json_ensure_ascii_preserving_format(const std::string & json_str) {
    llama_common_unicode_string result =
        llama_common_json_ensure_ascii_preserving_format_rust((const uint8_t *) json_str.data(), json_str.size());
    return common_rust_string_to_std(result);
}

static value tojson(const func_args & args) {
    args.ensure_count(1, 5);
    value val_ascii      = args.get_kwarg_or_pos("ensure_ascii", 1);
    value val_indent     = args.get_kwarg_or_pos("indent",       2);
    value val_separators = args.get_kwarg_or_pos("separators",   3);
    value val_sort       = args.get_kwarg_or_pos("sort_keys",    4);
    int indent = -1;
    if (args.ctx.is_get_stats) {
        // mark as used (recursively) for stats
        auto val_input = args.get_pos(0);
        value_t::stats_t::mark_used(const_cast<value&>(val_input), true);
    }
    if (is_val<value_int>(val_indent)) {
        indent = static_cast<int>(val_indent->as_int());
    }
    if (val_sort->as_bool()) { // undefined == false
        throw not_implemented_exception("tojson sort_keys=true not implemented");
    }
    const bool ensure_ascii = val_ascii->as_bool(); // undefined == false
    auto separators = (is_val<value_array>(val_separators) ? val_separators : mk_val<value_array>())->as_array();
    std::string item_sep = separators.size() > 0 ? separators[0]->as_string().str() : (indent < 0 ? ", " : ",");
    std::string key_sep = separators.size() > 1 ? separators[1]->as_string().str() : ": ";
    std::string json_str = value_to_json(args.get_pos(0), indent, item_sep, key_sep);
    if (ensure_ascii) {
        json_str = json_ensure_ascii_preserving_format(json_str);
    }
    return mk_val<value_string>(json_str);
}

template<bool is_reject>
static value selectattr(const func_args & args) {
    args.ensure_count(2, 4);
    args.ensure_vals<value_array, value_string, value_string, value_string>(true, true, false, false);

    auto arr = args.get_pos(0)->as_array();
    auto attribute = args.get_pos(1);
    auto out = mk_val<value_array>();
    value val_default = mk_val<value_undefined>();

    if (args.count() == 2) {
        // example: array | selectattr("active")
        for (const auto & item : arr) {
            if (!is_val<value_object>(item)) {
                throw raised_exception("selectattr: item is not an object");
            }
            value attr_val = item->at(attribute, val_default);
            bool is_selected = attr_val->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;

    } else if (args.count() == 3) {
        // example: array | selectattr("equalto", "text")
        // translated to: test_is_equalto(item, "text")
        std::string test_name = args.get_pos(1)->as_string().str();
        value test_val = args.get_pos(2);
        auto & builtins = global_builtins();
        auto it = builtins.find("test_is_" + test_name);
        if (it == builtins.end()) {
            throw raised_exception("selectattr: unknown test '" + test_name + "'");
        }
        auto test_fn = it->second;
        for (const auto & item : arr) {
            func_args test_args(args.ctx);
            test_args.push_back(item); // current object
            test_args.push_back(test_val); // extra argument
            value test_result = test_fn(test_args);
            bool is_selected = test_result->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;

    } else if (args.count() == 4) {
        // example: array | selectattr("status", "equalto", "active")
        // translated to: test_is_equalto(item.status, "active")
        std::string test_name = args.get_pos(2)->as_string().str();
        auto extra_arg = args.get_pos(3);
        auto & builtins = global_builtins();
        auto it = builtins.find("test_is_" + test_name);
        if (it == builtins.end()) {
            throw raised_exception("selectattr: unknown test '" + test_name + "'");
        }
        auto test_fn = it->second;
        for (const auto & item : arr) {
            if (!is_val<value_object>(item)) {
                throw raised_exception("selectattr: item is not an object");
            }
            value attr_val = item->at(attribute, val_default);
            func_args test_args(args.ctx);
            test_args.push_back(attr_val); // attribute value
            test_args.push_back(extra_arg); // extra argument
            value test_result = test_fn(test_args);
            bool is_selected = test_result->as_bool();
            if constexpr (is_reject) is_selected = !is_selected;
            if (is_selected) out->push_back(item);
        }
        return out;
    } else {
        throw raised_exception("selectattr: invalid number of arguments");
    }

    return out;
}

static value default_value(const func_args & args) {
    args.ensure_count(2, 3);
    value val_check = args.get_kwarg_or_pos("boolean", 2);
    bool check_bool = val_check->as_bool(); // undefined == false
    bool no_value = check_bool
        ? (!args.get_pos(0)->as_bool())
        : (args.get_pos(0)->is_undefined() || args.get_pos(0)->is_none());
    return no_value ? args.get_pos(1) : args.get_pos(0);
}

const func_builtins & global_builtins() {
    static const func_builtins builtins = {
        {"raise_exception", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string msg = args.get_pos(0)->as_string().str();
            throw raised_exception("Jinja Exception: " + msg);
        }},
        {"namespace", [](const func_args & args) -> value {
            auto out = mk_val<value_object>();
            for (const auto & arg : args.get_args()) {
                if (!is_val<value_kwarg>(arg)) {
                    throw raised_exception("namespace() arguments must be kwargs");
                }
                auto kwarg = cast_val<value_kwarg>(arg);
                JJ_DEBUG("namespace: adding key '%s'", kwarg->key.c_str());
                out->insert(kwarg->key, kwarg->val);
            }
            return out;
        }},
        {"strftime_now", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            std::string format = args.get_pos(0)->as_string().str();
            // get current time
            // TODO: make sure this is the same behavior as Python's strftime
            char buf[100];
            if (std::strftime(buf, sizeof(buf), format.c_str(), std::localtime(&args.ctx.current_time))) {
                return mk_val<value_string>(std::string(buf));
            } else {
                throw raised_exception("strftime_now: failed to format time");
            }
        }},
        {"range", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            args.ensure_vals<value_int, value_int, value_int>(true, false, false);

            auto arg0 = args.get_pos(0);
            auto arg1 = args.get_pos(1, mk_val<value_undefined>());
            auto arg2 = args.get_pos(2, mk_val<value_undefined>());

            int64_t start, stop, step;
            if (args.count() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.count() == 2) {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = 1;
            } else {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = arg2->as_int();
            }

            auto out = mk_val<value_array>();
            if (step == 0) {
                throw raised_exception("range() step argument must not be zero");
            }
            if (step > 0) {
                for (int64_t i = start; i < stop; i += step) {
                    out->push_back(mk_val<value_int>(i));
                }
            } else {
                for (int64_t i = start; i > stop; i += step) {
                    out->push_back(mk_val<value_int>(i));
                }
            }
            return out;
        }},
        {"tojson", tojson},

        // tests
        {"test_is_boolean", test_type_fn<value_bool>},
        {"test_is_callable", test_type_fn<value_func>},
        {"test_is_odd", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.get_pos(0)->as_int();
            return mk_val<value_bool>(llama_common_jinja_int_is_odd_rust(val));
        }},
        {"test_is_even", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.get_pos(0)->as_int();
            return mk_val<value_bool>(llama_common_jinja_int_is_even_rust(val));
        }},
        {"test_is_false", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool is_bool = is_val<value_bool>(args.get_pos(0));
            bool val = llama_common_jinja_is_false_rust(is_bool, is_bool ? args.get_pos(0)->as_bool() : false);
            return mk_val<value_bool>(val);
        }},
        {"test_is_true", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool is_bool = is_val<value_bool>(args.get_pos(0));
            bool val = llama_common_jinja_is_true_rust(is_bool, is_bool ? args.get_pos(0)->as_bool() : false);
            return mk_val<value_bool>(val);
        }},
        {"test_is_divisibleby", [](const func_args & args) -> value {
            args.ensure_vals<value_int, value_int>();
            bool res = args.get_pos(0)->val_int % args.get_pos(1)->val_int == 0;
            return mk_val<value_bool>(res);
        }},
        {"test_is_string", test_type_fn<value_string>},
        {"test_is_integer", test_type_fn<value_int>},
        {"test_is_float", test_type_fn<value_float>},
        {"test_is_number", test_type_fn<value_int, value_float>},
        {"test_is_iterable", test_type_fn<value_array, value_string, value_undefined>},
        {"test_is_sequence", test_type_fn<value_array, value_string, value_undefined>},
        {"test_is_mapping", test_type_fn<value_object>},
        {"test_is_lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.get_pos(0)->val_str.is_lowercase());
        }},
        {"test_is_upper", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            return mk_val<value_bool>(args.get_pos(0)->val_str.is_uppercase());
        }},
        {"test_is_none", test_type_fn<value_none>},
        {"test_is_defined", [](const func_args & args) -> value {
            args.ensure_count(1);
            bool res = llama_common_jinja_is_defined_rust(args.get_pos(0)->is_undefined());
            JJ_DEBUG("test_is_defined: result=%d", res ? 1 : 0);
            return mk_val<value_bool>(res);
        }},
        {"test_is_undefined", test_type_fn<value_undefined>},
        {"test_is_eq", test_compare_fn<value_compare_op::eq>},
        {"test_is_equalto", test_compare_fn<value_compare_op::eq>},
        {"test_is_ge", test_compare_fn<value_compare_op::ge>},
        {"test_is_gt", test_compare_fn<value_compare_op::gt>},
        {"test_is_greaterthan", test_compare_fn<value_compare_op::gt>},
        {"test_is_lt", test_compare_fn<value_compare_op::lt>},
        {"test_is_lessthan", test_compare_fn<value_compare_op::lt>},
        {"test_is_ne", test_compare_fn<value_compare_op::ne>},
        {"test_is_in", [](const func_args & args) -> value {
            args.ensure_count(2);
            auto needle   = args.get_pos(0);
            auto haystack = args.get_pos(1);
            if (is_val<value_undefined>(haystack)) {
                return mk_val<value_bool>(false);
            }
            if (is_val<value_array>(haystack)) {
                for (const auto & item : haystack->as_array()) {
                    if (*needle == *item) {
                        return mk_val<value_bool>(true);
                    }
                }
                return mk_val<value_bool>(false);
            }
            if (is_val<value_string>(haystack)) {
                if (!is_val<value_string>(needle)) {
                    throw raised_exception("'in' test expects args[1] as string when args[0] is string, got args[1] as " + needle->type());
                }
                return mk_val<value_bool>(
                    haystack->as_string().str().find(needle->as_string().str()) != std::string::npos);
            }
            if (is_val<value_object>(haystack)) {
                return mk_val<value_bool>(haystack->has_key(needle));
            }
            throw raised_exception("'in' test expects iterable as first argument, got " + haystack->type());
        }},
        {"test_is_test", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            auto & builtins = global_builtins();
            std::string test_name = args.get_pos(0)->val_str.str();
            auto it = builtins.find("test_is_" + test_name);
            bool res = it != builtins.end();
            return mk_val<value_bool>(res);
        }},
        {"test_is_sameas", [](const func_args & args) -> value {
            // Check if an object points to the same memory address as another object
            (void)args;
            throw not_implemented_exception("sameas test not implemented");
        }},
        {"test_is_escaped", [](const func_args & args) -> value {
            (void)args;
            throw not_implemented_exception("escaped test not implemented");
        }},
        {"test_is_filter", [](const func_args & args) -> value {
            (void)args;
            throw not_implemented_exception("filter test not implemented");
        }},
    };
    return builtins;
}


const func_builtins & value_int_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            int64_t val = args.get_pos(0)->as_int();
            return mk_val<value_int>(llama_common_jinja_int_abs_rust(val));
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            return mk_val<value_int>(args.get_pos(0)->as_int());
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_int>();
            double val = static_cast<double>(args.get_pos(0)->as_int());
            return mk_val<value_float>(val);
        }},
        {"safe", tojson},
        {"string", tojson},
        {"tojson", tojson},
    };
    return builtins;
}


const func_builtins & value_float_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"abs", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            double val = args.get_pos(0)->as_float();
            return mk_val<value_float>(llama_common_jinja_float_abs_rust(val));
        }},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            int64_t val = static_cast<int64_t>(args.get_pos(0)->as_float());
            return mk_val<value_int>(val);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_float>();
            return mk_val<value_float>(args.get_pos(0)->as_float());
        }},
        {"safe", tojson},
        {"string", tojson},
        {"tojson", tojson},
    };
    return builtins;
}

[[noreturn]] static value string_join_not_implemented(const func_args &) {
    throw not_implemented_exception("String join builtin not implemented");
}

const func_builtins & value_string_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"upper", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().uppercase();
            return mk_val<value_string>(str);
        }},
        {"lower", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().lowercase();
            return mk_val<value_string>(str);
        }},
        {"strip", [](const func_args & args) -> value {
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("strip() first argument must be a string");
            }
            value val_chars = args.get_kwarg_or_pos("chars", 1);
            if (val_chars->is_undefined()) {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, true));
            } else {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, true, val_chars->as_string().str()));
            }
        }},
        {"rstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            value val_chars = args.get_kwarg_or_pos("chars", 1);
            if (val_chars->is_undefined()) {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(false, true));
            } else {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(false, true, val_chars->as_string().str()));
            }
        }},
        {"lstrip", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            value val_chars = args.get_kwarg_or_pos("chars", 1);
            if (val_chars->is_undefined()) {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, false));
            } else {
                return mk_val<value_string>(args.get_pos(0)->as_string().strip(true, false, val_chars->as_string().str()));
            }
        }},
        {"title", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().titlecase();
            return mk_val<value_string>(str);
        }},
        {"capitalize", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string().capitalize();
            return mk_val<value_string>(str);
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            jinja::string str = args.get_pos(0)->as_string();
            return mk_val<value_int>(str.length());
        }},
        {"startswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.get_pos(0)->as_string().str();
            std::string prefix = args.get_pos(1)->as_string().str();
            return mk_val<value_bool>(llama_common_string_starts_with_rust(
                (const uint8_t *) str.data(), str.size(),
                (const uint8_t *) prefix.data(), prefix.size()));
        }},
        {"endswith", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string>();
            std::string str = args.get_pos(0)->as_string().str();
            std::string suffix = args.get_pos(1)->as_string().str();
            return mk_val<value_bool>(llama_common_string_ends_with_rust(
                (const uint8_t *) str.data(), str.size(),
                (const uint8_t *) suffix.data(), suffix.size()));
        }},
        {"split", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("split() first argument must be a string");
            }
            std::string str = val_input->as_string().str();
            // FIXME: Support non-specified delimiter (split on consecutive (no leading or trailing) whitespace)
            std::string delim = (args.count() > 1) ? args.get_pos(1)->as_string().str() : " ";
            int64_t maxsplit = (args.count() > 2) ? args.get_pos(2)->as_int() : -1;
            auto result = mk_val<value_array>();
            size_t pos = 0;
            std::string token;
            while ((pos = str.find(delim)) != std::string::npos && maxsplit != 0) {
                token = str.substr(0, pos);
                result->push_back(mk_val<value_string>(token));
                str.erase(0, pos + delim.length());
                --maxsplit;
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.get_pos(0)->val_str);
            result->push_back(std::move(res));
            return result;
        }},
        {"rsplit", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            value val_input = args.get_pos(0);
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("rsplit() first argument must be a string");
            }
            std::string str = val_input->as_string().str();
            // FIXME: Support non-specified delimiter (split on consecutive (no leading or trailing) whitespace)
            std::string delim = (args.count() > 1) ? args.get_pos(1)->as_string().str() : " ";
            int64_t maxsplit = (args.count() > 2) ? args.get_pos(2)->as_int() : -1;
            auto result = mk_val<value_array>();
            size_t pos = 0;
            std::string token;
            while ((pos = str.rfind(delim)) != std::string::npos && maxsplit != 0) {
                token = str.substr(pos + delim.length());
                result->push_back(mk_val<value_string>(token));
                str.erase(pos);
                --maxsplit;
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.get_pos(0)->val_str);
            result->push_back(std::move(res));
            result->reverse();
            return result;
        }},
        {"replace", [](const func_args & args) -> value {
            args.ensure_vals<value_string, value_string, value_string, value_int>(true, true, true, false);
            std::string str = args.get_pos(0)->as_string().str();
            std::string old_str = args.get_pos(1)->as_string().str();
            std::string new_str = args.get_pos(2)->as_string().str();
            int64_t count = args.count() > 3 ? args.get_pos(3)->as_int() : -1;
            if (count > 0) {
                throw not_implemented_exception("String replace with count argument not implemented");
            }
            size_t pos = 0;
            while ((pos = str.find(old_str, pos)) != std::string::npos) {
                str.replace(pos, old_str.length(), new_str);
                pos += new_str.length();
            }
            auto res = mk_val<value_string>(str);
            res->val_str.mark_input_based_on(args.get_pos(0)->val_str);
            return res;
        }},
        {"int", [](const func_args & args) -> value {
            value val_input   = args.get_pos(0);
            value val_default = args.get_kwarg_or_pos("default", 1);
            value val_base    = args.get_kwarg_or_pos("base",    2);
            const int base = val_base->is_undefined() ? 10 : val_base->as_int();
            if (is_val<value_string>(val_input) == false) {
                throw raised_exception("int() first argument must be a string");
            }
            std::string str = val_input->as_string().str();
            try {
                return mk_val<value_int>(std::stoi(str, nullptr, base));
            } catch (...) {
                return mk_val<value_int>(val_default->is_undefined() ? 0 : val_default->as_int());
            }
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_string>();
            value val_default = args.get_kwarg_or_pos("default", 1);
            std::string str = args.get_pos(0)->as_string().str();
            try {
                return mk_val<value_float>(std::stod(str));
            } catch (...) {
                return mk_val<value_float>(val_default->is_undefined() ? 0.0 : val_default->as_float());
            }
        }},
        {"string", [](const func_args & args) -> value {
            // no-op
            args.ensure_vals<value_string>();
            return mk_val<value_string>(args.get_pos(0)->as_string());
        }},
        {"default", [](const func_args & args) -> value {
            value input = args.get_pos(0);
            if (!is_val<value_string>(input)) {
                throw raised_exception("default() first argument must be a string");
            }
            value default_val = mk_val<value_string>("");
            if (args.count() > 1 && !args.get_pos(1)->is_undefined()) {
                default_val = args.get_pos(1);
            }
            value boolean_val = args.get_kwarg_or_pos("boolean", 2); // undefined == false
            if (input->is_undefined() || (boolean_val->as_bool() && !input->as_bool())) {
                return default_val;
            } else {
                return input;
            }
        }},
        {"slice", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_string, value_int, value_int, value_int>(true, true, false, false);

            auto arg0 = args.get_pos(1);
            auto arg1 = args.get_pos(2, mk_val<value_undefined>());
            auto arg2 = args.get_pos(3, mk_val<value_undefined>());

            int64_t start, stop, step;
            if (args.count() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.count() == 2) {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = 1;
            } else {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = arg2->as_int();
            }
            if (step == 0) {
                throw raised_exception("slice step cannot be zero");
            }
            auto input = args.get_pos(0);
            auto sliced = slice(input->as_string().str(), start, stop, step);
            auto res = mk_val<value_string>(sliced);
            res->val_str.mark_input_based_on(input->as_string());
            return res;
        }},
        {"safe", [](const func_args & args) -> value {
            // no-op for now
            args.ensure_vals<value_string>();
            return args.get_pos(0);
        }},
        {"tojson", tojson},
        {"indent", [](const func_args &args) -> value {
            args.ensure_count(1, 4);
            value val_input  = args.get_pos(0);
            value val_width  = args.get_kwarg_or_pos("width", 1);
            const bool first = args.get_kwarg_or_pos("first", 2)->as_bool(); // undefined == false
            const bool blank = args.get_kwarg_or_pos("blank", 3)->as_bool(); // undefined == false
            if (!is_val<value_string>(val_input)) {
                throw raised_exception("indent() first argument must be a string");
            }
            std::string indent;
            if (is_val<value_int>(val_width)) {
                indent.assign(val_width->as_int(), ' ');
            } else if (is_val<value_string>(val_width)) {
                indent = val_width->as_string().str();
            } else {
                indent = "    ";
            }
            std::string indented;
            std::string input = val_input->as_string().str();
            std::istringstream iss = std::istringstream(input);
            std::string line;
            while (std::getline(iss, line)) {
                if (!indented.empty()) {
                    indented.push_back('\n');
                }
                if ((indented.empty() ? first : (!line.empty() || blank))) {
                    indented += indent;
                }
                indented += line;
            }
            if (!input.empty() && input.back() == '\n') {
                indented.push_back('\n');
                if (blank) {
                    indented += indent;
                }
            }

            auto res = mk_val<value_string>(indented);
            res->val_str.mark_input_based_on(val_input->as_string());
            return res;
        }},
        {"join", string_join_not_implemented},
    };
    return builtins;
}


const func_builtins & value_bool_t::get_builtins() const {
    static const func_handler tostring = [](const func_args & args) -> value {
        args.ensure_vals<value_bool>();
        bool val = args.get_pos(0)->as_bool();
        return mk_val<value_string>(val ? "True" : "False");
    };
    static const func_builtins builtins = {
        {"default", default_value},
        {"int", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.get_pos(0)->as_bool();
            return mk_val<value_int>(val ? 1 : 0);
        }},
        {"float", [](const func_args & args) -> value {
            args.ensure_vals<value_bool>();
            bool val = args.get_pos(0)->as_bool();
            return mk_val<value_float>(val ? 1.0 : 0.0);
        }},
        {"safe", tostring},
        {"string", tostring},
        {"tojson", tojson},
    };
    return builtins;
}

[[noreturn]] static value array_unique_not_implemented(const func_args &) {
    throw not_implemented_exception("Array unique builtin not implemented");
}

const func_builtins & value_array_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"list", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            auto result = mk_val<value_array>();
            for (const auto& v : arr) {
                result->push_back(v);
            }
            return result;
        }},
        {"first", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[0];
        }},
        {"last", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            if (arr.empty()) {
                return mk_val<value_undefined>();
            }
            return arr[arr.size() - 1];
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            const auto & arr = args.get_pos(0)->as_array();
            return mk_val<value_int>(static_cast<int64_t>(arr.size()));
        }},
        {"slice", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            args.ensure_vals<value_array, value_int, value_int, value_int>(true, true, false, false);

            auto val  = args.get_pos(0);
            auto arg0 = args.get_pos(1);
            auto arg1 = args.get_pos(2, mk_val<value_undefined>());
            auto arg2 = args.get_pos(3, mk_val<value_undefined>());

            int64_t start, stop, step;
            if (args.count() == 1) {
                start = 0;
                stop = arg0->as_int();
                step = 1;
            } else if (args.count() == 2) {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = 1;
            } else {
                start = arg0->as_int();
                stop = arg1->as_int();
                step = arg2->as_int();
            }
            if (step == 0) {
                throw raised_exception("slice step cannot be zero");
            }
            auto arr = slice(val->as_array(), start, stop, step);
            return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(arr)) : mk_val<value_array>(std::move(arr));
        }},
        {"selectattr", selectattr<false>},
        {"select", selectattr<false>},
        {"rejectattr", selectattr<true>},
        {"reject", selectattr<true>},
        {"join", [](const func_args & args) -> value {
            args.ensure_count(1, 3);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("join() first argument must be an array");
            }
            value val_delim = args.get_kwarg_or_pos("d",         1);
            value attribute = args.get_kwarg_or_pos("attribute", 2);
            const auto & arr = args.get_pos(0)->as_array();
            const bool attr_is_int = is_val<value_int>(attribute);
            if (!attribute->is_undefined() && !is_val<value_string>(attribute) && !attr_is_int) {
                throw raised_exception("join() attribute must be string or integer");
            }
            const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
            const std::string delim = val_delim->is_undefined() ? "" : val_delim->as_string().str();
            std::string result;
            for (size_t i = 0; i < arr.size(); ++i) {
                value val_arr = arr[i];
                if (!attribute->is_undefined()) {
                    if (attr_is_int && is_val<value_array>(val_arr)) {
                        val_arr = val_arr->at(attr_int);
                    } else if (!attr_is_int && is_val<value_object>(val_arr)) {
                        val_arr = val_arr->at(attribute);
                    }
                }
                if (!is_val<value_string>(val_arr) && !is_val<value_int>(val_arr) && !is_val<value_float>(val_arr)) {
                    throw raised_exception("join() can only join arrays of strings or numerics");
                }
                result += val_arr->as_string().str();
                if (i < arr.size() - 1) {
                    result += delim;
                }
            }
            return mk_val<value_string>(result);
        }},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            if (args.ctx.is_get_stats) {
                // mark as used (recursively) for stats
                auto val_input = args.get_pos(0);
                value_t::stats_t::mark_used(const_cast<value&>(val_input), true);
            }
            return mk_val<value_string>(args.get_pos(0)->as_string());
        }},
        {"tojson", tojson},
        {"map", [](const func_args & args) -> value {
            args.ensure_count(2);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("map: first argument must be an array");
            }
            if (!is_val<value_kwarg>(args.get_args().at(1))) {
                throw not_implemented_exception("map: filter-mapping not implemented");
            }
            value val       = args.get_pos(0);
            value attribute = args.get_kwarg_or_pos("attribute", 1);
            const bool attr_is_int = is_val<value_int>(attribute);
            if (!is_val<value_string>(attribute) && !attr_is_int) {
                throw raised_exception("map: attribute must be string or integer");
            }
            const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
            value default_val = args.get_kwarg("default", mk_val<value_undefined>());
            auto out = mk_val<value_array>();
            auto arr = val->as_array();
            for (const auto & item : arr) {
                value attr_val;
                if (attr_is_int) {
                    attr_val = is_val<value_array>(item) ? item->at(attr_int, default_val) : default_val;
                } else {
                    attr_val = is_val<value_object>(item) ? item->at(attribute, default_val) : default_val;
                }
                out->push_back(attr_val);
            }
            return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(out->as_array())) : out;
        }},
        {"append", [](const func_args & args) -> value {
            args.ensure_count(2);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("append: first argument must be an array");
            }
            const value_array_t * arr = cast_val<value_array>(args.get_pos(0));
            // need to use const_cast here to modify the array
            value_array_t * arr_editable = const_cast<value_array_t *>(arr);
            arr_editable->push_back(args.get_pos(1));
            return args.get_pos(0);
        }},
        {"pop", [](const func_args & args) -> value {
            args.ensure_count(1, 2);
            args.ensure_vals<value_array, value_int>(true, false);
            int64_t index = args.count() == 2 ? args.get_pos(1)->as_int() : -1;
            const value_array_t * arr = cast_val<value_array>(args.get_pos(0));
            // need to use const_cast here to modify the array
            value_array_t * arr_editable = const_cast<value_array_t *>(arr);
            return arr_editable->pop_at(index);
        }},
        {"sort", [](const func_args & args) -> value {
            args.ensure_count(1, 4);
            if (!is_val<value_array>(args.get_pos(0))) {
                throw raised_exception("sort: first argument must be an array");
            }
            value val         = args.get_pos(0);
            value val_reverse = args.get_kwarg_or_pos("reverse",        1);
            value val_case    = args.get_kwarg_or_pos("case_sensitive", 2);
            value attribute   = args.get_kwarg_or_pos("attribute",      3);
            // FIXME: sorting is currently always case sensitive
            //const bool case_sensitive = val_case->as_bool(); // undefined == false
            const bool reverse = val_reverse->as_bool(); // undefined == false
            const bool attr_is_int = is_val<value_int>(attribute);
            const int64_t attr_int = attr_is_int ? attribute->as_int() : 0;
            std::vector<value> arr = val->as_array(); // copy
            std::sort(arr.begin(), arr.end(),[&](const value & a, const value & b) {
                value val_a = a;
                value val_b = b;
                if (!attribute->is_undefined()) {
                    if (attr_is_int && is_val<value_array>(a) && is_val<value_array>(b)) {
                        val_a = a->at(attr_int);
                        val_b = b->at(attr_int);
                    } else if (!attr_is_int && is_val<value_object>(a) && is_val<value_object>(b)) {
                        val_a = a->at(attribute);
                        val_b = b->at(attribute);
                    } else {
                        throw raised_exception("sort: unsupported object attribute comparison between " + a->type() + " and " + b->type());
                    }
                }
                return value_compare(val_a, val_b, reverse ? value_compare_op::gt : value_compare_op::lt);
            });
            return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(arr)) : mk_val<value_array>(std::move(arr));
        }},
        {"reverse", [](const func_args & args) -> value {
            args.ensure_vals<value_array>();
            value val = args.get_pos(0);
            std::vector<value> arr = val->as_array(); // copy
            std::reverse(arr.begin(), arr.end());
            return is_val<value_tuple>(val) ? mk_val<value_tuple>(std::move(arr)) : mk_val<value_array>(std::move(arr));
        }},
        {"unique", array_unique_not_implemented},
    };
    return builtins;
}

[[noreturn]] static value object_join_not_implemented(const func_args &) {
    throw not_implemented_exception("object join not implemented");
}

const func_builtins & value_object_t::get_builtins() const {
    if (!has_builtins) {
        static const func_builtins no_builtins = {};
        return no_builtins;
    }

    static const func_builtins builtins = {
        // {"default", default_value}, // cause issue with gpt-oss
        {"get", [](const func_args & args) -> value {
            args.ensure_count(2, 3);
            if (!is_val<value_object>(args.get_pos(0))) {
                throw raised_exception("get: first argument must be an object");
            }
            if (!is_val<value_string>(args.get_pos(1))) {
                throw raised_exception("get: second argument must be a string (key)");
            }
            value default_val = mk_val<value_none>();
            if (args.count() == 3) {
                default_val = args.get_pos(2);
            }
            const value obj = args.get_pos(0);
            const value key = args.get_pos(1);
            return obj->at(key, default_val);
        }},
        {"keys", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.get_pos(0)->as_ordered_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                result->push_back(pair.first);
            }
            return result;
        }},
        {"values", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.get_pos(0)->as_ordered_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                result->push_back(pair.second);
            }
            return result;
        }},
        {"items", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.get_pos(0)->as_ordered_object();
            auto result = mk_val<value_array>();
            for (const auto & pair : obj) {
                auto item = mk_val<value_tuple>(pair);
                result->push_back(std::move(item));
            }
            return result;
        }},
        {"tojson", tojson},
        {"string", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            if (args.ctx.is_get_stats) {
                // mark as used (recursively) for stats
                auto val_input = args.get_pos(0);
                value_t::stats_t::mark_used(const_cast<value&>(val_input), true);
            }
            return mk_val<value_string>(args.get_pos(0)->as_string());
        }},
        {"length", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            const auto & obj = args.get_pos(0)->as_ordered_object();
            return mk_val<value_int>(static_cast<int64_t>(obj.size()));
        }},
        {"tojson", [](const func_args & args) -> value {
            args.ensure_vals<value_object>();
            // use global to_json
            return global_builtins().at("tojson")(args);
        }},
        {"dictsort", [](const func_args & args) -> value {
            value val_input   = args.get_pos(0);
            value val_case    = args.get_kwarg_or_pos("case_sensitive", 1);
            value val_by      = args.get_kwarg_or_pos("by",             2);
            value val_reverse = args.get_kwarg_or_pos("reverse",        3);
            // FIXME: sorting is currently always case sensitive
            //const bool case_sensitive = val_case->as_bool(); // undefined == false
            const bool reverse = val_reverse->as_bool(); // undefined == false
            const bool by_value = is_val<value_string>(val_by) && val_by->as_string().str() == "value" ? true : false;
            auto result = mk_val<value_object>(val_input); // copy
            std::sort(result->val_obj.begin(), result->val_obj.end(), [&](const auto & a, const auto & b) {
                if (by_value) {
                    return value_compare(a.second, b.second, reverse ? value_compare_op::gt : value_compare_op::lt);
                } else {
                    return value_compare(a.first, b.first, reverse ? value_compare_op::gt : value_compare_op::lt);
                }
            });
            return result;
        }},
        {"join", object_join_not_implemented},
    };
    return builtins;
}

const func_builtins & value_none_t::get_builtins() const {
    static const func_handler tostring = [](const func_args &) -> value {
        return mk_val<value_string>("None");
    };
    static const func_builtins builtins = {
        {"default", default_value},
        {"tojson", tojson},
        {"string", tostring},
        {"safe", tostring},
        {"items", empty_value_fn<value_array>},
        {"map", empty_value_fn<value_array>},
        {"reject", empty_value_fn<value_array>},
        {"rejectattr", empty_value_fn<value_array>},
        {"select", empty_value_fn<value_array>},
        {"selectattr", empty_value_fn<value_array>},
        {"unique", empty_value_fn<value_array>},
    };
    return builtins;
}


const func_builtins & value_undefined_t::get_builtins() const {
    static const func_builtins builtins = {
        {"default", default_value},
        {"capitalize", empty_value_fn<value_string>},
        {"first", empty_value_fn<value_undefined>},
        {"items", empty_value_fn<value_array>},
        {"join", empty_value_fn<value_string>},
        {"last", empty_value_fn<value_undefined>},
        {"length", empty_value_fn<value_int>},
        {"list", empty_value_fn<value_array>},
        {"lower", empty_value_fn<value_string>},
        {"map", empty_value_fn<value_array>},
        {"max", empty_value_fn<value_undefined>},
        {"min", empty_value_fn<value_undefined>},
        {"reject", empty_value_fn<value_array>},
        {"rejectattr", empty_value_fn<value_array>},
        {"replace", empty_value_fn<value_string>},
        {"reverse", empty_value_fn<value_array>},
        {"safe", empty_value_fn<value_string>},
        {"select", empty_value_fn<value_array>},
        {"selectattr", empty_value_fn<value_array>},
        {"sort", empty_value_fn<value_array>},
        {"string", empty_value_fn<value_string>},
        {"strip", empty_value_fn<value_string>},
        {"sum", empty_value_fn<value_int>},
        {"title", empty_value_fn<value_string>},
        {"truncate", empty_value_fn<value_string>},
        {"unique", empty_value_fn<value_array>},
        {"upper", empty_value_fn<value_string>},
        {"wordcount", empty_value_fn<value_int>},
    };
    return builtins;
}


//////////////////////////////////


static value from_json(const nlohmann::ordered_json & j, bool mark_input) {
    if (j.is_null()) {
        return mk_val<value_none>();
    } else if (j.is_boolean()) {
        return mk_val<value_bool>(j.get<bool>());
    } else if (j.is_number_integer()) {
        return mk_val<value_int>(j.get<int64_t>());
    } else if (j.is_number_float()) {
        return mk_val<value_float>(j.get<double>());
    } else if (j.is_string()) {
        auto str = mk_val<value_string>(j.get<std::string>());
        if (mark_input) {
            str->mark_input();
        }
        return str;
    } else if (j.is_array()) {
        auto arr = mk_val<value_array>();
        for (const auto & item : j) {
            arr->push_back(from_json(item, mark_input));
        }
        return arr;
    } else if (j.is_object()) {
        auto obj = mk_val<value_object>();
        for (auto it = j.begin(); it != j.end(); ++it) {
            obj->insert(it.key(), from_json(it.value(), mark_input));
        }
        return obj;
    } else {
        throw std::runtime_error("Unsupported JSON value type");
    }
}

// compare operator for value_t
bool value_compare(const value & a, const value & b, value_compare_op op) {
    auto cmp = [&]() {
        // compare numeric types
        if ((is_val<value_int>(a) || is_val<value_float>(a)) &&
            (is_val<value_int>(b) || is_val<value_float>(b))){
            try {
                if (op == value_compare_op::eq) {
                    return llama_common_jinja_compare_f64_rust(a->as_float(), b->as_float(), op);
                } else if (op == value_compare_op::ge) {
                    return llama_common_jinja_compare_f64_rust(a->as_float(), b->as_float(), op);
                } else if (op == value_compare_op::gt) {
                    return llama_common_jinja_compare_f64_rust(a->as_float(), b->as_float(), op);
                } else if (op == value_compare_op::lt) {
                    return llama_common_jinja_compare_f64_rust(a->as_float(), b->as_float(), op);
                } else if (op == value_compare_op::ne) {
                    return llama_common_jinja_compare_f64_rust(a->as_float(), b->as_float(), op);
                } else {
                    throw std::runtime_error("Unsupported comparison operator for numeric types");
                }
            } catch (...) {}
        }
        // compare string and number
        // TODO: not sure if this is the right behavior
        if ((is_val<value_string>(b) && (is_val<value_int>(a) || is_val<value_float>(a))) ||
            (is_val<value_string>(a) && (is_val<value_int>(b) || is_val<value_float>(b))) ||
            (is_val<value_string>(a) && is_val<value_string>(b))) {
            try {
                const std::string lhs = a->as_string().str();
                const std::string rhs = b->as_string().str();
                if (op == value_compare_op::eq) {
                    return llama_common_jinja_compare_bytes_rust((const uint8_t *) lhs.data(), lhs.size(), (const uint8_t *) rhs.data(), rhs.size(), op);
                } else if (op == value_compare_op::ge) {
                    return llama_common_jinja_compare_bytes_rust((const uint8_t *) lhs.data(), lhs.size(), (const uint8_t *) rhs.data(), rhs.size(), op);
                } else if (op == value_compare_op::gt) {
                    return llama_common_jinja_compare_bytes_rust((const uint8_t *) lhs.data(), lhs.size(), (const uint8_t *) rhs.data(), rhs.size(), op);
                } else if (op == value_compare_op::lt) {
                    return llama_common_jinja_compare_bytes_rust((const uint8_t *) lhs.data(), lhs.size(), (const uint8_t *) rhs.data(), rhs.size(), op);
                } else if (op == value_compare_op::ne) {
                    return llama_common_jinja_compare_bytes_rust((const uint8_t *) lhs.data(), lhs.size(), (const uint8_t *) rhs.data(), rhs.size(), op);
                } else {
                    throw std::runtime_error("Unsupported comparison operator for string/number types");
                }
            } catch (...) {}
        }
        // compare boolean simple
        if (is_val<value_bool>(a) && is_val<value_bool>(b)) {
            if (op == value_compare_op::eq) {
                return llama_common_jinja_compare_bool_rust(a->as_bool(), b->as_bool(), op);
            } else if (op == value_compare_op::ne) {
                return llama_common_jinja_compare_bool_rust(a->as_bool(), b->as_bool(), op);
            } else {
                throw std::runtime_error("Unsupported comparison operator for bool type");
            }
        }
        // compare by type
        if (a->type() != b->type()) {
            return false;
        }
        return false;
    };
    auto result = cmp();
    JJ_DEBUG("Comparing types: %s and %s result=%d", a->type().c_str(), b->type().c_str(), result);
    return result;
}

template<>
void global_from_json(context & ctx, const nlohmann::ordered_json & json_obj, bool mark_input) {
    // printf("global_from_json: %s\n" , json_obj.dump(2).c_str());
    if (json_obj.is_null() || !json_obj.is_object()) {
        throw std::runtime_error("global_from_json: input JSON value must be an object");
    }
    for (auto it = json_obj.begin(); it != json_obj.end(); ++it) {
        JJ_DEBUG("global_from_json: setting key '%s'", it.key().c_str());
        ctx.set_val(it.key(), from_json(it.value(), mark_input));
    }
}

// recursively convert value to JSON string
// TODO: avoid circular references
static void value_to_json_internal(std::ostringstream & oss, const value & val, int curr_lvl, int indent, const std::string_view item_sep, const std::string_view key_sep) {
    auto indent_str = [indent, curr_lvl]() -> std::string {
        return (indent > 0) ? std::string(curr_lvl * indent, ' ') : "";
    };
    auto newline = [indent]() -> std::string {
        return (indent >= 0) ? "\n" : "";
    };

    if (is_val<value_none>(val) || val->is_undefined()) {
        oss << "null";
    } else if (is_val<value_bool>(val)) {
        oss << (val->as_bool() ? "true" : "false");
    } else if (is_val<value_int>(val)) {
        oss << val->as_int();
    } else if (is_val<value_float>(val)) {
        oss << val->as_float();
    } else if (is_val<value_string>(val)) {
        oss << "\"";
        for (char c : val->as_string().str()) {
            switch (c) {
                case '"': oss << "\\\""; break;
                case '\\': oss << "\\\\"; break;
                case '\b': oss << "\\b"; break;
                case '\f': oss << "\\f"; break;
                case '\n': oss << "\\n"; break;
                case '\r': oss << "\\r"; break;
                case '\t': oss << "\\t"; break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[7];
                        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        oss << buf;
                    } else {
                        oss << c;
                    }
            }
        }
        oss << "\"";
    } else if (is_val<value_array>(val)) {
        const auto & arr = val->as_array();
        oss << "[";
        if (!arr.empty()) {
            oss << newline();
            for (size_t i = 0; i < arr.size(); ++i) {
                oss << indent_str() << (indent > 0 ? std::string(indent, ' ') : "");
                value_to_json_internal(oss, arr[i], curr_lvl + 1, indent, item_sep, key_sep);
                if (i < arr.size() - 1) {
                    oss << item_sep;
                }
                oss << newline();
            }
            oss << indent_str();
        }
        oss << "]";
    } else if (is_val<value_object>(val)) {
        const auto & obj = val->as_ordered_object(); // IMPORTANT: need to keep exact order
        oss << "{";
        if (!obj.empty()) {
            oss << newline();
            size_t i = 0;
            for (const auto & pair : obj) {
                oss << indent_str() << (indent > 0 ? std::string(indent, ' ') : "");
                value_to_json_internal(oss, mk_val<value_string>(pair.first->as_string().str()), curr_lvl + 1, indent, item_sep, key_sep);
                oss << key_sep;
                value_to_json_internal(oss, pair.second, curr_lvl + 1, indent, item_sep, key_sep);
                if (i < obj.size() - 1) {
                    oss << item_sep;
                }
                oss << newline();
                ++i;
            }
            oss << indent_str();
        }
        oss << "}";
    } else {
        oss << "null";
    }
}

std::string value_to_json(const value & val, int indent, const std::string_view item_sep, const std::string_view key_sep) {
    std::ostringstream oss;
    value_to_json_internal(oss, val, 0, indent, item_sep, key_sep);
    JJ_DEBUG("value_to_json: result=%s", oss.str().c_str());
    return oss.str();
}

// TODO: avoid circular references
std::string value_to_string_repr(const value & val) {
    if (is_val<value_string>(val)) {
        const std::string val_str = val->as_string().str();

        if (val_str.find('\'') != std::string::npos) {
            return value_to_json(val);
        } else {
            return "'" + val_str + "'";
        }
    } else {
        return val->as_repr();
    }
}

// stats utility
void value_t::stats_t::mark_used(value & val, bool deep) {
    val->stats.used = true;
    if (deep) {
        if (is_val<value_array>(val)) {
            for (auto & item : val->val_arr) {
                mark_used(item, deep);
            }
        } else if (is_val<value_object>(val)) {
            for (auto & pair : val->val_obj) {
                mark_used(pair.first, deep);
                mark_used(pair.second, deep);
            }
        }
    }
}

} // namespace jinja

#include "common.h.inc"

// note: the json dependency is only for defining input in a convenient way
// we can remove it in the future when we figure out a better way to define inputs using jinja::value
#include <nlohmann/json.hpp.inc>

#include <functional>
#include <sstream>

#undef FILENAME
#define FILENAME "jinja-caps"

using json = nlohmann::ordered_json;

namespace jinja {

using caps_json_fn = std::function<json()>;
using caps_analyze_fn = std::function<void(bool, value &, value &)>;

static void caps_try_execute(jinja::program & prog,
                             const caps_json_fn & messages_fn,
                             const caps_json_fn & tools_fn,
                             const caps_analyze_fn & analyze_fn) {
    context ctx;
    ctx.is_get_stats = true;
    jinja::global_from_json(ctx, json{
        {"messages", messages_fn()},
        {"tools", tools_fn()},
        {"bos_token", ""},
        {"eos_token", ""},
        {"add_generation_prompt", true}
    }, true);

    auto messages = ctx.get_val("messages");
    auto tools = ctx.get_val("tools");

    bool success = false;
    std::string result;
    try {
        jinja::runtime runtime(ctx);
        auto results = runtime.execute(prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        result = parts->as_string().str();
        success = true;
    } catch (const std::exception & e) {
        JJ_DEBUG("Exception during execution: %s", e.what());
        result = "";
        // ignore exceptions during capability analysis
    }

    analyze_fn(success, messages, tools);
}

// for debugging only
static void caps_print_stats(value & v, const std::string & path) {
    std::string ops;
    for (const auto & name : v->stats.ops) {
        ops += name + " ";
    }
    JJ_DEBUG("Value %s, type: %s %s, ops: %s",
                path.c_str(),
                v->type().c_str(),
                v->stats.used ? "(used)" : "",
                ops.c_str());
}

std::map<std::string, bool> caps::to_map() const {
    return {
        {"supports_string_content", supports_string_content},
        {"supports_typed_content", supports_typed_content},
        {"supports_tools", supports_tools},
        {"supports_tool_calls", supports_tool_calls},
        {"supports_parallel_tool_calls", supports_parallel_tool_calls},
        {"supports_system_role", supports_system_role},
        {"supports_preserve_reasoning", supports_preserve_reasoning},
        {"supports_object_arguments", supports_object_arguments},
    };
}

std::string caps::to_string() const {
    std::ostringstream ss;
    ss << "Caps(\n";
    for (const auto & [key, value] : to_map()) {
        ss << "  " << key << "=" << (value ? "true" : "false") << "\n";
    }
    ss << ")";
    return ss.str();
}

caps caps_get(jinja::program & prog) {
    caps result;

    static const auto has_op = [](value & v, const std::string & op_name) {
        return v->stats.ops.find(op_name) != v->stats.ops.end();
    };

    JJ_DEBUG("%s\n", ">>> Running capability check: typed content");

    // case: typed content support
    caps_try_execute(
        prog,
        [&]() {
            // messages
            return json::array({
                {
                    {"role", "user"},
                    {"content", "content"}
                }
            });
        },
        [&]() {
            // tools
            return json{nullptr};
        },
        [&](bool success, value & messages, value &) {
            auto & content = messages->at(0)->at("content");
            caps_print_stats(content, "messages[0].content");
            if (has_op(content, "selectattr") || has_op(content, "array_access")) {
                // accessed as an array
                result.supports_typed_content = true;
            }
            if (!success) {
                // failed to execute with content as string
                result.supports_string_content = false;
            }
        }
    );

    JJ_DEBUG("%s\n", ">>> Running capability check: system prompt");

    // case: system prompt support
    caps_try_execute(
        prog,
        [&]() {
            // messages
            return json::array({
                {
                    {"role", "system"},
                    {"content", "System message"}
                },
                {
                    {"role", "user"},
                    {"content", "User message"}
                },
            });
        },
        [&]() {
            // tools
            return json::array();
        },
        [&](bool, value & messages, value &) {
            auto & content = messages->at(0)->at("content");
            caps_print_stats(content, "messages[0].content");
            if (!content->stats.used) {
                result.supports_system_role = false;
            }
        }
    );

    JJ_DEBUG("%s\n", ">>> Running capability check: single tool with object arguments support");

    // case: tools support: single call with object arguments
    caps_try_execute(
        prog,
        [&]() {
            // messages
            return json::array({
                {
                    {"role", "user"},
                    {"content", "User message"},
                },
                {
                    {"role", "assistant"},
                    {"content", ""}, // Some templates expect content to be empty with tool calls
                    {"tool_calls", json::array({
                        {
                            {"id", "call00001"},
                            {"type", "function"},
                            {"function", {
                                {"name", "tool1"},
                                {"arguments", {
                                    {"arg", "value"}
                                }}
                            }}
                        }
                    })}
                },
                {
                    {"role", "tool"},
                    {"content", "Tool response"},
                    {"tool_call_id", "call00001"}
                },
                {
                    {"role", "assistant"},
                    {"content", "The tool response was 'tool response'"}
                },
                {
                    {"role", "user"},
                    {"content", "User message"},
                },
            });
        },
        [&]() {
            // tools
            return json::array({
                {
                    {"name", "tool"},
                    {"type", "function"},
                    {"function", {
                        {"name", "tool1"},
                        {"description", "Tool description"},
                        {"parameters", {
                            {"type", "object"},
                            {"properties", {
                                {"arg", {
                                    {"type", "string"},
                                    {"description", "Arg description"},
                                }},
                            }},
                            {"required", json::array({ "arg" })},
                        }},
                    }},
                },
            });
        },
        [&](bool success, value & messages, value & tools) {
            if (!success) {
                return; // Nothing can be inferred
            }

            auto & tool_name = tools->at(0)->at("function")->at("name");
            caps_print_stats(tool_name, "tools[0].function.name");
            caps_print_stats(tools, "tools");
            if (!tool_name->stats.used) {
                result.supports_tools = false;
            }

            auto & tool_calls = messages->at(1)->at("tool_calls");;
            caps_print_stats(tool_calls, "messages[1].tool_calls");
            if (!tool_calls->stats.used) {
                result.supports_tool_calls = false;
                return;
            }

            auto & tool_arg = tool_calls->at(0)->at("function")->at("arguments")->at("arg");
            caps_print_stats(tool_arg, "messages[1].tool_calls[0].function.arguments.arg");
            if (tool_arg->stats.used) {
                result.supports_object_arguments = true;
            }
        }
    );

    if (!result.supports_object_arguments) {
        JJ_DEBUG("%s\n", ">>> Running capability check: single tool with string arguments support");

        // case: tools support: single call with string arguments
        caps_try_execute(
            prog,
            [&]() {
                // messages
                return json::array({
                    {
                        {"role", "user"},
                        {"content", "User message"},
                    },
                    {
                        {"role", "assistant"},
                        {"content", ""}, // Some templates expect content to be empty with tool calls
                        {"tool_calls", json::array({
                            {
                                {"id", "call00001"},
                                {"type", "function"},
                                {"function", {
                                    {"name", "tool1"},
                                    {"arguments", R"({"arg": "value"})"}
                                }}
                            }
                        })}
                    },
                    {
                        {"role", "tool"},
                        {"content", "Tool response"},
                        {"tool_call_id", "call00001"}
                    },
                    {
                        {"role", "assistant"},
                        {"content", "The tool response was 'tool response'"}
                    },
                    {
                        {"role", "user"},
                        {"content", "User message"},
                    },
                });
            },
            [&]() {
                // tools
                return json::array({
                    {
                        {"name", "tool"},
                        {"type", "function"},
                        {"function", {
                            {"name", "tool1"},
                            {"description", "Tool description"},
                            {"parameters", {
                                {"type", "object"},
                                {"properties", {
                                    {"arg", {
                                        {"type", "string"},
                                        {"description", "Arg description"},
                                    }},
                                }},
                                {"required", json::array({ "arg" })},
                            }},
                        }},
                    },
                });
            },
            [&](bool success, value & messages, value & tools) {
                if (!success) {
                    result.supports_tool_calls = false;
                    result.supports_tools = false;
                    return;
                }

                auto & tool_name = tools->at(0)->at("function")->at("name");
                caps_print_stats(tool_name, "tools[0].function.name");
                caps_print_stats(tools, "tools");
                if (!tool_name->stats.used) {
                    result.supports_tools = false;
                }

                auto & tool_calls = messages->at(1)->at("tool_calls");
                caps_print_stats(tool_calls, "messages[1].tool_calls");
                if (!tool_calls->stats.used) {
                    result.supports_tool_calls = false;
                    return;
                }
            }
        );
    }

    JJ_DEBUG("%s\n", ">>> Running capability check: parallel tool support");

    // case: tools support: parallel calls
    caps_try_execute(
        prog,
        [&]() {
            json args = json(R"({"arg": "value"})");
            if (result.supports_object_arguments) {
                args = json{{"arg", "value"}};
            }

            // messages
            return json::array({
                {
                    {"role", "user"},
                    {"content", "User message"},
                },
                {
                    {"role", "assistant"},
                    {"content", ""}, // Some templates expect content to be empty with tool calls
                    {"tool_calls", json::array({
                        {
                            {"id", "call00001"},
                            {"type", "function"},
                            {"function", {
                                {"name", "tool1"},
                                {"arguments", args}
                            }}
                        },
                        {
                            {"id", "call00002"},
                            {"type", "function"},
                            {"function", {
                                {"name", "tool1"},
                                {"arguments", args}
                            }}
                        }
                    })}
                },
                {
                    {"role", "tool"},
                    {"content", "Tool response"},
                    {"tool_call_id", "call00001"}
                },
                {
                    {"role", "assistant"},
                    {"content", "The tool response was 'tool response'"}
                },
                {
                    {"role", "user"},
                    {"content", "User message"},
                },
            });
        },
        [&]() {
            // tools
            return json::array({
                {
                    {"name", "tool"},
                    {"type", "function"},
                    {"function", {
                        {"name", "tool1"},
                        {"description", "Tool description"},
                        {"parameters", {
                            {"type", "object"},
                            {"properties", {
                                {"arg", {
                                    {"type", "string"},
                                    {"description", "Arg description"},
                                }},
                            }},
                            {"required", json::array({ "arg" })},
                        }},
                    }},
                },
            });
        },
        [&](bool success, value & messages, value & /*tools*/) {
            if (!success) {
                result.supports_parallel_tool_calls = false;
                return;
            }

            auto & tool_calls = messages->at(1)->at("tool_calls");
            caps_print_stats(tool_calls, "messages[1].tool_calls");

            // check for second tool call usage
            auto & tool_call_1 = tool_calls->at(1)->at("function");
            caps_print_stats(tool_call_1, "messages[1].tool_calls[1].function");
            if (!tool_call_1->stats.used) {
                result.supports_parallel_tool_calls = false;
            }
        }
    );

    JJ_DEBUG("%s\n", ">>> Running capability check: preserve reasoning");

    // case: preserve reasoning content in chat history
    caps_try_execute(
        prog,
        [&]() {
            // messages
            return json::array({
                {
                    {"role", "user"},
                    {"content", "User message"}
                },
                {
                    {"role", "assistant"},
                    {"content", "Assistant message"},
                    {"reasoning_content", "Reasoning content"}
                },
                {
                    {"role", "user"},
                    {"content", "User message"}
                },
            });
        },
        [&]() {
            // tools
            return json::array();
        },
        [&](bool, value & messages, value &) {
            auto & content = messages->at(1)->at("reasoning_content");
            caps_print_stats(content, "messages[1].reasoning_content");
            if (content->stats.used) {
                result.supports_preserve_reasoning = true;
            }
        }
    );

    JJ_DEBUG("%s\n", result.to_string().c_str());

    return result;
}

} // namespace jinja

#include "common.h.inc"

#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    include <io.h>
#    include <windows.h>
#    define isatty _isatty
#    define fileno _fileno
#else
#    include <unistd.h>
#endif // defined(_WIN32)

int common_log_verbosity_thold = LOG_DEFAULT_LLAMA;

int common_log_get_verbosity_thold(void) {
    return common_log_verbosity_thold;
}

void common_log_set_verbosity_thold(int verbosity) {
    common_log_verbosity_thold = verbosity;
}

static int64_t t_us() {
    return llama_common_time_us_rust();
}

// colors
enum common_log_col : int {
    COMMON_LOG_COL_DEFAULT = 0,
    COMMON_LOG_COL_BOLD,
    COMMON_LOG_COL_RED,
    COMMON_LOG_COL_GREEN,
    COMMON_LOG_COL_YELLOW,
    COMMON_LOG_COL_BLUE,
    COMMON_LOG_COL_MAGENTA,
    COMMON_LOG_COL_CYAN,
    COMMON_LOG_COL_WHITE,
};

// disable colors by default
static const char* g_col[] = {
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
};

struct common_log_entry {
    enum ggml_log_level level;

    bool prefix;

    int64_t timestamp;

    std::vector<char> msg;

    // signals the worker thread to stop
    bool is_end;

    void print(FILE * file = nullptr) const {
        FILE * fcur = file;
        if (!fcur) {
            // stderr displays DBG messages only when their verbosity level is not higher than the threshold
            // these messages will still be logged to a file
            if (level == GGML_LOG_LEVEL_DEBUG && common_log_verbosity_thold < LOG_DEFAULT_DEBUG) {
                return;
            }

            fcur = stdout;

            if (level != GGML_LOG_LEVEL_NONE) {
                fcur = stderr;
            }
        }

        if (level != GGML_LOG_LEVEL_NONE && level != GGML_LOG_LEVEL_CONT && prefix) {
            if (timestamp) {
                // [M.s.ms.us]
                fprintf(fcur, "%s%d.%02d.%03d.%03d%s ",
                        g_col[COMMON_LOG_COL_BLUE],
                        (int) (timestamp / 1000000 / 60),
                        (int) (timestamp / 1000000 % 60),
                        (int) (timestamp / 1000 % 1000),
                        (int) (timestamp % 1000),
                        g_col[COMMON_LOG_COL_DEFAULT]);
            }

            switch (level) {
                case GGML_LOG_LEVEL_INFO:  fprintf(fcur, "%sI %s", g_col[COMMON_LOG_COL_GREEN],   g_col[COMMON_LOG_COL_DEFAULT]); break;
                case GGML_LOG_LEVEL_WARN:  fprintf(fcur, "%sW %s", g_col[COMMON_LOG_COL_MAGENTA], ""                        ); break;
                case GGML_LOG_LEVEL_ERROR: fprintf(fcur, "%sE %s", g_col[COMMON_LOG_COL_RED],     ""                        ); break;
                case GGML_LOG_LEVEL_DEBUG: fprintf(fcur, "%sD %s", g_col[COMMON_LOG_COL_YELLOW],  ""                        ); break;
                default:
                    break;
            }
        }

        fprintf(fcur, "%s", msg.data());

        if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_DEBUG) {
            fprintf(fcur, "%s", g_col[COMMON_LOG_COL_DEFAULT]);
        }

        fflush(fcur);
    }
};

struct common_log {
    // default capacity - will be expanded if needed
    common_log() : common_log(256) {}

    common_log(size_t capacity) {
        file = nullptr;
        prefix = false;
        timestamps = false;
        running = false;
        t_start = t_us();

        // initial message size - will be expanded if longer messages arrive
        entries.resize(capacity);
        for (auto & entry : entries) {
            entry.msg.resize(256);
        }

        head = 0;
        tail = 0;

        resume();
    }

    ~common_log() {
        pause();
        if (file) {
            fclose(file);
        }
    }

private:
    std::mutex mtx;
    std::thread thrd;
    std::condition_variable cv;

    FILE * file;

    bool prefix;
    bool timestamps;
    bool running;

    int64_t t_start;

    // ring buffer of entries
    std::vector<common_log_entry> entries;
    size_t head;
    size_t tail;

    // worker thread copies into this
    common_log_entry cur;

public:
    void add(enum ggml_log_level level, const char * fmt, va_list args) {
        std::lock_guard<std::mutex> lock(mtx);

        if (!running) {
            // discard messages while the worker thread is paused
            return;
        }

        auto & entry = entries[tail];

        {
            // cannot use args twice, so make a copy in case we need to expand the buffer
            va_list args_copy;
            va_copy(args_copy, args);

#if 1
            const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args);
            if (n >= entry.msg.size()) {
                entry.msg.resize(n + 1);
                vsnprintf(entry.msg.data(), entry.msg.size(), fmt, args_copy);
            }
#else
            // hack for bolding arguments

            std::stringstream ss;
            for (int i = 0; fmt[i] != 0; i++) {
                if (fmt[i] == '%') {
                    ss << LOG_COL_BOLD;
                    while (fmt[i] != ' ' && fmt[i] != ')' && fmt[i] != ']' && fmt[i] != 0) ss << fmt[i++];
                    ss << LOG_COL_DEFAULT;
                    if (fmt[i] == 0) break;
                }
                ss << fmt[i];
            }
            const size_t n = vsnprintf(entry.msg.data(), entry.msg.size(), ss.str().c_str(), args);
            if (n >= entry.msg.size()) {
                entry.msg.resize(n + 1);
                vsnprintf(entry.msg.data(), entry.msg.size(), ss.str().c_str(), args_copy);
            }
#endif
            va_end(args_copy);
        }

        entry.level = level;
        entry.prefix = prefix;
        entry.timestamp = 0;
        if (timestamps) {
            entry.timestamp = t_us() - t_start;
        }
        entry.is_end = false;

        tail = (tail + 1) % entries.size();
        if (tail == head) {
            // expand the buffer
            std::vector<common_log_entry> new_entries(2*entries.size());

            size_t new_tail = 0;

            do {
                new_entries[new_tail] = std::move(entries[head]);

                head     = (head     + 1) % entries.size();
                new_tail = (new_tail + 1);
            } while (head != tail);

            head = 0;
            tail = new_tail;

            for (size_t i = tail; i < new_entries.size(); i++) {
                new_entries[i].msg.resize(256);
            }

            entries = std::move(new_entries);
        }
        cv.notify_one();
    }

    void resume() {
        std::lock_guard<std::mutex> lock(mtx);

        if (running) {
            return;
        }

        running = true;

        thrd = std::thread([this]() {
            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    cv.wait(lock, [this]() { return head != tail; });
                    cur = entries[head];

                    head = (head + 1) % entries.size();
                }

                if (cur.is_end) {
                    break;
                }

                cur.print(); // stdout and stderr

                if (file) {
                    cur.print(file);
                }
            }
        });
    }

    void pause() {
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (!running) {
                return;
            }

            running = false;

            // push an entry to signal the worker thread to stop
            {
                auto & entry = entries[tail];
                entry.is_end = true;

                tail = (tail + 1) % entries.size();
            }
            cv.notify_one();
        }

        thrd.join();
    }

    void set_file(const char * path) {
        pause();

        if (file) {
            fclose(file);
        }

        if (path) {
            file = fopen(path, "w");
        } else {
            file = nullptr;
        }

        resume();
    }

    void set_colors(bool colors) {
        pause();

        if (colors) {
            g_col[COMMON_LOG_COL_DEFAULT] = LOG_COL_DEFAULT;
            g_col[COMMON_LOG_COL_BOLD]    = LOG_COL_BOLD;
            g_col[COMMON_LOG_COL_RED]     = LOG_COL_RED;
            g_col[COMMON_LOG_COL_GREEN]   = LOG_COL_GREEN;
            g_col[COMMON_LOG_COL_YELLOW]  = LOG_COL_YELLOW;
            g_col[COMMON_LOG_COL_BLUE]    = LOG_COL_BLUE;
            g_col[COMMON_LOG_COL_MAGENTA] = LOG_COL_MAGENTA;
            g_col[COMMON_LOG_COL_CYAN]    = LOG_COL_CYAN;
            g_col[COMMON_LOG_COL_WHITE]   = LOG_COL_WHITE;
        } else {
            for (size_t i = 0; i < std::size(g_col); i++) {
                g_col[i] = "";
            }
        }

        resume();
    }

    void set_prefix(bool prefix) {
        std::lock_guard<std::mutex> lock(mtx);

        this->prefix = prefix;
    }

    void set_timestamps(bool timestamps) {
        std::lock_guard<std::mutex> lock(mtx);

        this->timestamps = timestamps;
    }
};

//
// public API
//

struct common_log * common_log_init() {
    return new common_log;
}

struct common_log * common_log_main() {
    // We intentionally leak (i.e. do not delete) the logger singleton because
    // common_log destructor called at DLL teardown phase will cause hanging on Windows.
    // OS will release resources anyway so it should not be a significant issue,
    // though this design may cause logs to be lost if not flushed before the program exits.
    // Refer to https://github.com/ggml-org/llama.cpp/issues/22142 for details.
    static struct common_log * log;
    static std::once_flag    init_flag;
    std::call_once(init_flag, [&]() {
        log = new common_log;
        // Set default to auto-detect colors
        log->set_colors(tty_can_use_colors());
    });

    return log;
}

void common_log_pause(struct common_log * log) {
    log->pause();
}

void common_log_resume(struct common_log * log) {
    log->resume();
}

void common_log_free(struct common_log * log) {
    delete log;
}

void common_log_add(struct common_log * log, enum ggml_log_level level, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log->add(level, fmt, args);
    va_end(args);
}

void common_log_set_file(struct common_log * log, const char * file) {
    log->set_file(file);
}

void common_log_set_colors(struct common_log * log, log_colors colors) {
    if (colors == LOG_COLORS_AUTO) {
        log->set_colors(tty_can_use_colors());
        return;
    }

    if (colors == LOG_COLORS_DISABLED) {
        log->set_colors(false);
        return;
    }

    GGML_ASSERT(colors == LOG_COLORS_ENABLED);
    log->set_colors(true);
}

void common_log_set_prefix(struct common_log * log, bool prefix) {
    log->set_prefix(prefix);
}

void common_log_set_timestamps(struct common_log * log, bool timestamps) {
    log->set_timestamps(timestamps);
}

void common_log_flush(struct common_log * log) {
    log->pause();
    log->resume();
}

void common_log_default_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    auto verbosity = llama_common_log_level_to_verbosity_rust((int) level);
    if (verbosity <= common_log_verbosity_thold) {
        common_log_add(common_log_main(), level, "%s", text);
    }
}

#include "common.h.inc"

#include <fstream>
#include <sstream>
#include <filesystem>

// only allow a subset of args for remote presets for security reasons
// do not add more args unless absolutely necessary
// args that output to files are strictly prohibited
static std::set<std::string> get_remote_preset_whitelist(const std::map<std::string, common_arg> & key_to_opt) {
    static const std::set<std::string> allowed_options = {
        "model-url",
        "hf-repo",
        "hf-repo-draft",
        "hf-repo-v", // vocoder
        "hf-file-v", // vocoder
        "mmproj-url",
        "pooling",
        "jinja",
        "batch-size",
        "ubatch-size",
        "cache-reuse",
        "chat-template-kwargs",
        "mmap",
        // note: sampling params are automatically allowed by default
        // negated args will be added automatically if the positive arg is specified above
    };

    std::set<std::string> allowed_keys;

    for (const auto & it : key_to_opt) {
        const std::string & key = it.first;
        const common_arg & opt = it.second;
        if (allowed_options.find(key) != allowed_options.end() || opt.is_sampling) {
            allowed_keys.insert(key);
            // also add variant keys (args without leading dashes and env vars)
            for (const auto & arg : opt.get_args()) {
                allowed_keys.insert(common_rust_string_to_std(
                        llama_common_rm_leading_dashes_rust((const uint8_t *) arg.data(), arg.size())));
            }
            for (const auto & env : opt.get_env()) {
                allowed_keys.insert(env);
            }
        }
    }

    return allowed_keys;
}

std::vector<std::string> common_preset::to_args(const std::string & bin_path) const {
    std::vector<std::string> args;

    if (!bin_path.empty()) {
        args.push_back(bin_path);
    }

    for (const auto & [opt, value] : options) {
        if (opt.is_preset_only) {
            continue; // skip preset-only options (they are not CLI args)
        }

        // use the last arg as the main arg (i.e. --long-form)
        args.push_back(opt.args.back());

        // handle value(s)
        if (opt.value_hint == nullptr && opt.value_hint_2 == nullptr) {
            // flag option, no value
            if (common_arg_utils::is_falsey(value)) {
                // use negative arg if available
                if (!opt.args_neg.empty()) {
                    args.back() = opt.args_neg.back();
                } else {
                    // otherwise, skip the flag
                    // TODO: maybe throw an error instead?
                    args.pop_back();
                }
            }
        }
        if (opt.value_hint != nullptr) {
            // single value
            args.push_back(value);
        }
        if (opt.value_hint != nullptr && opt.value_hint_2 != nullptr) {
            throw std::runtime_error(string_format(
                "common_preset::to_args(): option '%s' has two values, which is not supported yet",
                opt.args.back()
            ));
        }
    }

    return args;
}

std::string common_preset::to_ini() const {
    std::ostringstream ss;

    ss << "[" << name << "]\n";
    for (const auto & [opt, value] : options) {
        auto espaced_value = value;
        string_replace_all(espaced_value, "\n", "\\\n");
        const std::string & arg = opt.args.back();
        ss << common_rust_string_to_std(
                llama_common_rm_leading_dashes_rust((const uint8_t *) arg.data(), arg.size())) << " = ";
        ss << espaced_value << "\n";
    }
    ss << "\n";

    return ss.str();
}

void common_preset::set_option(const common_preset_context & ctx, const std::string & env, const std::string & value) {
    // try if option exists, update it
    for (auto & [opt, val] : options) {
        if (opt.env && env == opt.env) {
            val = value;
            return;
        }
    }
    // if option does not exist, we need to add it
    if (ctx.key_to_opt.find(env) == ctx.key_to_opt.end()) {
        throw std::runtime_error(string_format(
            "%s: option with env '%s' not found in ctx_params",
            __func__, env.c_str()
        ));
    }
    options[ctx.key_to_opt.at(env)] = value;
}

void common_preset::unset_option(const std::string & env) {
    for (auto it = options.begin(); it != options.end(); ) {
        const common_arg & opt = it->first;
        if (opt.env && env == opt.env) {
            it = options.erase(it);
            return;
        } else {
            ++it;
        }
    }
}

bool common_preset::get_option(const std::string & env, std::string & value) const {
    for (const auto & [opt, val] : options) {
        if (opt.env && env == opt.env) {
            value = val;
            return true;
        }
    }
    return false;
}

void common_preset::merge(const common_preset & other) {
    for (const auto & [opt, val] : other.options) {
        options[opt] = val; // overwrite existing options
    }
}

void common_preset::apply_to_params(common_params & params) const {
    for (const auto & [opt, val] : options) {
        // apply each option to params
        if (opt.handler_string) {
            opt.handler_string(params, val);
        } else if (opt.handler_int) {
            opt.handler_int(params, std::stoi(val));
        } else if (opt.handler_bool) {
            opt.handler_bool(params, common_arg_utils::is_truthy(val));
        } else if (opt.handler_str_str) {
            // not supported yet
            throw std::runtime_error(string_format(
                "%s: option with two values is not supported yet",
                __func__
            ));
        } else if (opt.handler_void) {
            opt.handler_void(params);
        } else {
            GGML_ABORT("unknown handler type");
        }
    }
}

static std::map<std::string, std::map<std::string, std::string>> parse_ini_from_file(const std::string & path) {
    std::map<std::string, std::map<std::string, std::string>> parsed;

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("preset file does not exist: " + path);
    }

    std::ifstream file(path);
    if (!file.good()) {
        throw std::runtime_error("failed to open server preset file: " + path);
    }

    std::string contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    static const auto parser = build_peg_parser([](auto & p) {
        // newline ::= "\r\n" / "\n" / "\r"
        auto newline = p.rule("newline", p.literal("\r\n") | p.literal("\n") | p.literal("\r"));

        // ws ::= [ \t]*
        auto ws = p.rule("ws", p.chars("[ \t]", 0, -1));

        // comment ::= [;#] (!newline .)*
        auto comment = p.rule("comment", p.chars("[;#]", 1, 1) + p.zero_or_more(p.negate(newline) + p.any()));

        // eol ::= ws comment? (newline / EOF)
        auto eol = p.rule("eol", ws + p.optional(comment) + (newline | p.end()));

        // ident ::= [a-zA-Z_] [a-zA-Z0-9_.-]*
        auto ident = p.rule("ident", p.chars("[a-zA-Z_]", 1, 1) + p.chars("[a-zA-Z0-9_.-]", 0, -1));

        // value ::= (!eol-start .)*
        auto eol_start = p.rule("eol-start", ws + (p.chars("[;#]", 1, 1) | newline | p.end()));
        auto value = p.rule("value", p.zero_or_more(p.negate(eol_start) + p.any()));

        // header-line ::= "[" ws ident ws "]" eol
        auto header_line = p.rule("header-line", "[" + ws + p.tag("section-name", p.chars("[^]]")) + ws + "]" + eol);

        // kv-line ::= ident ws "=" ws value eol
        auto kv_line = p.rule("kv-line", p.tag("key", ident) + ws + "=" + ws + p.tag("value", value) + eol);

        // comment-line ::= ws comment (newline / EOF)
        auto comment_line = p.rule("comment-line", ws + comment + (newline | p.end()));

        // blank-line ::= ws (newline / EOF)
        auto blank_line = p.rule("blank-line", ws + (newline | p.end()));

        // line ::= header-line / kv-line / comment-line / blank-line
        auto line = p.rule("line", header_line | kv_line | comment_line | blank_line);

        // ini ::= line* EOF
        auto ini = p.rule("ini", p.zero_or_more(line) + p.end());

        return ini;
    });

    common_peg_parse_context ctx(contents);
    const auto result = parser.parse(ctx);
    if (!result.success()) {
        throw std::runtime_error("failed to parse server config file: " + path);
    }

    std::string current_section = COMMON_PRESET_DEFAULT_NAME;
    std::string current_key;

    ctx.ast.visit(result, [&](const auto & node) {
        if (node.tag == "section-name") {
            const std::string section = std::string(node.text);
            current_section = section;
            parsed[current_section] = {};
        } else if (node.tag == "key") {
            const std::string key = std::string(node.text);
            current_key = key;
        } else if (node.tag == "value" && !current_key.empty() && !current_section.empty()) {
            parsed[current_section][current_key] = std::string(node.text);
            current_key.clear();
        }
    });

    return parsed;
}

static std::map<std::string, common_arg> get_map_key_opt(common_params_context & ctx_params) {
    std::map<std::string, common_arg> mapping;
    for (const auto & opt : ctx_params.options) {
        for (const auto & env : opt.get_env()) {
            mapping[env] = opt;
        }
        for (const auto & arg : opt.get_args()) {
            mapping[common_rust_string_to_std(
                    llama_common_rm_leading_dashes_rust((const uint8_t *) arg.data(), arg.size()))] = opt;
        }
    }
    return mapping;
}

static std::string parse_bool_arg(const common_arg & arg, const std::string & key, const std::string & value) {
    std::vector<common_rust_string_view> neg_args;
    neg_args.reserve(arg.args_neg.size());
    for (const auto & neg_arg : arg.args_neg) {
        neg_args.push_back({ (const uint8_t *) neg_arg, std::strlen(neg_arg) });
    }
    return common_rust_string_to_std(llama_common_parse_bool_arg_rust(
            neg_args.data(), neg_args.size(),
            (const uint8_t *) key.data(), key.size(),
            (const uint8_t *) value.data(), value.size()));
}

common_preset_context::common_preset_context(llama_example ex, bool only_remote_allowed)
        : ctx_params(common_params_parser_init(default_params, ex)) {
    common_params_add_preset_options(ctx_params.options);
    key_to_opt = get_map_key_opt(ctx_params);

    // setup allowed keys if only_remote_allowed is true
    if (only_remote_allowed) {
        filter_allowed_keys = true;
        allowed_keys = get_remote_preset_whitelist(key_to_opt);
    }
}

common_presets common_preset_context::load_from_ini(const std::string & path, common_preset & global) const {
    common_presets out;
    auto ini_data = parse_ini_from_file(path);

    for (auto section : ini_data) {
        common_preset preset;
        if (section.first.empty()) {
            preset.name = COMMON_PRESET_DEFAULT_NAME;
        } else {
            preset.name = section.first;
        }
        LOG_DBG("loading preset: %s\n", preset.name.c_str());
        for (const auto & [key, value] : section.second) {
            if (key == "version") {
                // skip version key (reserved for future use)
                continue;
            }

            LOG_DBG("option: %s = %s\n", key.c_str(), value.c_str());
            if (filter_allowed_keys && allowed_keys.find(key) == allowed_keys.end()) {
                throw std::runtime_error(string_format(
                    "option '%s' is not allowed in remote presets",
                    key.c_str()
                ));
            }
            if (key_to_opt.find(key) != key_to_opt.end()) {
                const auto & opt = key_to_opt.at(key);
                if (!opt.args_neg.empty()) {
                    preset.options[opt] = parse_bool_arg(opt, key, value);
                } else {
                    preset.options[opt] = value;
                }
                LOG_DBG("accepted option: %s = %s\n", key.c_str(), preset.options[opt].c_str());
            } else {
                throw std::runtime_error(string_format(
                    "option '%s' not recognized in preset '%s'",
                    key.c_str(), preset.name.c_str()
                ));
            }
        }

        if (preset.name == "*") {
            // handle global preset
            global = preset;
        } else {
            out[preset.name] = preset;
        }
    }

    return out;
}

common_presets common_preset_context::load_from_cache() const {
    common_presets out;

    auto cached_models = common_list_cached_models();
    for (const auto & model : cached_models) {
        common_preset preset;
        preset.name = model.to_string();
        preset.set_option(*this, "LLAMA_ARG_HF_REPO", model.to_string());
        out[preset.name] = preset;
    }

    return out;
}

struct local_model {
    std::string name;
    std::string path;
    std::string path_mmproj;
};

common_presets common_preset_context::load_from_models_dir(const std::string & models_dir) const {
    if (!std::filesystem::exists(models_dir) || !std::filesystem::is_directory(models_dir)) {
        throw std::runtime_error(string_format("error: '%s' does not exist or is not a directory\n", models_dir.c_str()));
    }

    std::vector<local_model> models;
    auto scan_subdir = [&models](const std::string & subdir_path, const std::string & name) {
        auto files = fs_list(subdir_path, false);
        common_file_info model_file;
        common_file_info first_shard_file;
        common_file_info mmproj_file;
        for (const auto & file : files) {
            if (string_ends_with(file.name, ".gguf")) {
                if (file.name.find("mmproj") != std::string::npos) {
                    mmproj_file = file;
                } else if (file.name.find("-00001-of-") != std::string::npos) {
                    first_shard_file = file;
                } else {
                    model_file = file;
                }
            }
        }
        // single file model
        local_model model{
            /* name        */ name,
            /* path        */ first_shard_file.path.empty() ? model_file.path : first_shard_file.path,
            /* path_mmproj */ mmproj_file.path // can be empty
        };
        if (!model.path.empty()) {
            models.push_back(model);
        }
    };

    auto files = fs_list(models_dir, true);
    for (const auto & file : files) {
        if (file.is_dir) {
            scan_subdir(file.path, file.name);
        } else if (string_ends_with(file.name, ".gguf")) {
            // single file model
            std::string name = file.name;
            string_replace_all(name, ".gguf", "");
            local_model model{
                /* name        */ name,
                /* path        */ file.path,
                /* path_mmproj */ ""
            };
            models.push_back(model);
        }
    }

    // convert local models to presets
    common_presets out;
    for (const auto & model : models) {
        common_preset preset;
        preset.name = model.name;
        preset.set_option(*this, "LLAMA_ARG_MODEL", model.path);
        if (!model.path_mmproj.empty()) {
            preset.set_option(*this, "LLAMA_ARG_MMPROJ", model.path_mmproj);
        }
        out[preset.name] = preset;
    }

    return out;
}

common_preset common_preset_context::load_from_args(int argc, char ** argv) const {
    common_preset preset;
    preset.name = COMMON_PRESET_DEFAULT_NAME;

    bool ok = common_params_to_map(argc, argv, ctx_params.ex, preset.options);
    if (!ok) {
        throw std::runtime_error("failed to parse CLI arguments into preset");
    }

    return preset;
}

common_presets common_preset_context::cascade(const common_presets & base, const common_presets & added) const {
    common_presets out = base; // copy
    for (const auto & [name, preset_added] : added) {
        if (out.find(name) != out.end()) {
            // if exists, merge
            common_preset & target = out[name];
            target.merge(preset_added);
        } else {
            // otherwise, add directly
            out[name] = preset_added;
        }
    }
    return out;
}

common_presets common_preset_context::cascade(const common_preset & base, const common_presets & presets) const {
    common_presets out;
    for (const auto & [name, preset] : presets) {
        common_preset tmp = base; // copy
        tmp.name = name;
        tmp.merge(preset);
        out[name] = std::move(tmp);
    }
    return out;
}

#include "common.h.inc"

#include "../src/llama-impl.h.inc"

#include <array>
#include <cassert>
#include <stdexcept>
#include <cinttypes>
#include <set>
#include <string>
#include <vector>

// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};

class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

static std::vector<llama_device_memory_data> common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.use_mmap  = false;
    mparams_copy.use_mlock = false;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model);
    std::vector<llama_device_memory_data> ret(nd + 1);

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model, i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        size_t free;
        size_t total;
        ggml_backend_dev_memory(llama_model_get_device(model, i), &free, &total);

        // devices can return 0 bytes for free and total memory if they do not
        // have any to report. in this case, we will use the host memory as a fallback
        // fixes: https://github.com/ggml-org/llama.cpp/issues/18577
        if (free == 0 && total == 0) {
            free  = ret.back().free;
            total = ret.back().total;
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devs.push_back(llama_model_get_device(model, i));
    }

    hp_ngl         = llama_model_n_layer(model);
    hp_n_ctx_train = llama_model_n_ctx_train(model);
    hp_n_expert    = llama_model_n_expert(model);

    common_memory_breakdown_print(ctx);

    llama_free(ctx);
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);

    return ret;
}

static void common_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins_s, uint32_t n_ctx_min, enum ggml_log_level log_level) {
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        throw common_params_fit_exception("llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
    }
    constexpr int64_t MiB = 1024*1024;
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LOG_INF("%s: getting device memory data for initial parameters:\n", __func__);
    const dmds_t dmds_full = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
    const size_t nd = devs.size(); // number of devices

    std::vector<int64_t> margins; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    margins.reserve(nd);
    if (nd == 0) {
        margins.push_back(margins_s[0]);
    } else {
        for (size_t id = 0; id < nd; id++) {
            margins.push_back(margins_s[id]);
        }
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (const auto & dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_free            = 0;
    int64_t sum_projected_free  = 0;
    int64_t sum_projected_used  = 0;
    int64_t sum_projected_model = 0;
    std::vector<int64_t> projected_free_per_device;
    projected_free_per_device.reserve(nd);

    if (nd == 0) {
        sum_projected_used = dmds_full.back().mb.total();
        sum_free           = dmds_full.back().total;
        sum_projected_free = sum_free - sum_projected_used;
        LOG_INF("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (sum_projected_free >= margins[0]) {
            LOG_INF("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n",
                __func__, sum_projected_free/MiB, margins[0]/MiB);
            return;
        }
    } else {
        if (nd > 1) {
            LOG_INF("%s: projected memory use with initial parameters [MiB]:\n", __func__);
        }
        for (size_t id = 0; id < nd; id++) {
            const llama_device_memory_data & dmd = dmds_full[id];

            const int64_t projected_used = dmd.mb.total();
            const int64_t projected_free = dmd.free - projected_used;
            projected_free_per_device.push_back(projected_free);

            sum_free            += dmd.free;
            sum_projected_used  += projected_used;
            sum_projected_free  += projected_free;
            sum_projected_model += dmd.mb.model;

            if (nd > 1) {
                LOG_INF("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                    __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, projected_free/MiB, margins[id]/MiB);
            }
        }
        assert(sum_free >= 0 && sum_projected_used >= 0);
        LOG_INF("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, sum_projected_used/MiB, sum_free/MiB);
        if (nd == 1) {
            if (projected_free_per_device[0] >= margins[0]) {
                LOG_INF("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                    __func__, projected_free_per_device[0]/MiB, margins[0]/MiB);
                return;
            }
        } else {
            bool changes_needed = false;
            for (size_t id = 0; id < nd; id++) {
                if (projected_free_per_device[id] < margins[id]) {
                    changes_needed = true;
                    break;
                }
            }
            if (!changes_needed) {
                LOG_INF("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
                return;
            }
        }
    }

    // step 2: try reducing memory use by reducing the context size

    {
        int64_t global_surplus = sum_projected_free;
        if (nd == 0) {
            global_surplus -= margins[0];
        } else {
            for (size_t id = 0; id < nd; id++) {
                global_surplus -= margins[id];
            }
        }
        if (global_surplus < 0) {
            if (nd <= 1) {
                LOG_INF("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n",
                    __func__, margins[0]/MiB, -global_surplus/MiB);
            } else {
                LOG_INF(
                    "%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                    __func__, -global_surplus/MiB);
            }
            if (cparams->n_ctx == 0) {
                if (hp_nct > n_ctx_min) {
                    int64_t sum_used_target = sum_free;
                    if (nd == 0) {
                        sum_used_target -= margins[0];
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_used_target -= margins[id];
                        }
                    }
                    if (nd > 1) {
                        // for multiple devices we need to be more conservative in terms of how much context we think can fit:
                        //   - for dense models only whole layers can be assigned to devices
                        //   - for MoE models only whole tensors can be assigned to devices, which we estimate to be <= 1/3 of a layer
                        //   - on average we expect a waste of 0.5 layers/tensors per device
                        //   - use slightly more than the expected average for nd devices to be safe
                        const int64_t model_per_layer = sum_projected_model / std::min(uint32_t(mparams->n_gpu_layers), hp_ngl);
                        sum_used_target -= (nd + 1) * model_per_layer / (hp_nex == 0 ? 2 : 6);
                    }

                    int64_t sum_projected_used_min_ctx = 0;
                    cparams->n_ctx = n_ctx_min;
                    const dmds_t dmds_min_ctx = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
                    if (nd == 0) {
                        sum_projected_used_min_ctx = dmds_min_ctx.back().mb.total();
                    } else {
                        for (size_t id = 0; id < nd; id++) {
                            sum_projected_used_min_ctx += dmds_min_ctx[id].mb.total();
                        }
                    }
                    if (sum_used_target > sum_projected_used_min_ctx) {
                        // linear interpolation between minimum and maximum context size:
                        cparams->n_ctx += (hp_nct - n_ctx_min) * (sum_used_target - sum_projected_used_min_ctx)
                            / (sum_projected_used - sum_projected_used_min_ctx);
                        cparams->n_ctx = std::max(cparams->n_ctx - cparams->n_ctx % 256, n_ctx_min); // round down context for CUDA backend

                        const int64_t bytes_per_ctx = (sum_projected_used - sum_projected_used_min_ctx) / (hp_nct - n_ctx_min);
                        const int64_t memory_reduction = (hp_nct - cparams->n_ctx) * bytes_per_ctx;
                        LOG_INF("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                        if (nd <= 1) {
                            LOG_INF("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LOG_INF("%s: entire model should be fit across devices by reducing context\n", __func__);
                    } else {
                        const int64_t memory_reduction = sum_projected_used - sum_projected_used_min_ctx;
                        LOG_INF("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                            __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                    }
                } else {
                    if (n_ctx_min == UINT32_MAX) {
                        LOG_INF("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, hp_nct);
                    } else {
                        LOG_INF("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                            __func__, hp_nct, n_ctx_min);
                    }
                }
            } else {
                LOG_INF("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }
    if (nd == 0) {
        throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const common_layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|up|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|gate_up|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate_up|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        common_layer_fraction_t overflow_type = LAYER_FRACTION_MOE;

        uint32_t n_full() const {
            assert(n_layer >= n_part);
            return n_layer - n_part;
        }
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
        uint32_t il0 = hp_ngl + 1 - mparams.n_gpu_layers; // start index for tensor buft overrides

        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_full();
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw common_params_fit_exception("llama_max_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = il == il0 ? overflow_bufts[id] : ggml_backend_cpu_buffer_type();
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy);

        const dmds_t dmd_nl = common_get_device_memory_data(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        LOG_INF("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LOG_INF(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (size_t id = 0; id < nd; id++) {
            ret.push_back(dmd_nl[id].mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate_up|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LOG_INF("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        const dmds_t dmds_cpu_moe = common_get_device_memory_data(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        for (size_t id = 0; id < nd; id++) {
            global_surplus_cpu_moe += dmds_cpu_moe[id].free;
            global_surplus_cpu_moe -= int64_t(dmds_cpu_moe[id].mb.total()) + margins[id];
        }

        if (global_surplus_cpu_moe > 0) {
            LOG_INF("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LOG_INF("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margins[id]);
        LOG_INF("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the first partial layer of a device overflows to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        overflow_bufts.push_back(ggml_backend_cpu_buffer_type());
    }

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts);

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    //   - the last device has the output layer, which cannot be a partial layer
    if (hp_nex == 0) {
        LOG_INF("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LOG_INF("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    for (int id = nd - 1; id >= 0; id--) {
        uint32_t n_unassigned = hp_ngl + 1;
        for (size_t jd = id + 1; jd < nd; ++jd) {
            assert(n_unassigned >= ngl_per_device[jd].n_layer);
            n_unassigned -= ngl_per_device[jd].n_layer;
        }

        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = size_t(id) < nd - 1 ? ngl_per_device_high[id].n_layer : ngl_per_device_high[id].n_layer - 1;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);
            if (mem_high[id] > targets[id]) {
                assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                LOG_INF("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, id, delta);
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += size_t(id) == nd - 1 && ngl_per_device_test[id].n_part == 0 ?
                            step_size - 1 : step_size; // the first layer is the output layer which must always be full
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device = ngl_per_device_test;
                        mem            = mem_test;
                        LOG_INF("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LOG_INF("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device_high[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                assert(ngl_per_device_high[id].n_layer == n_unassigned);
                ngl_per_device = ngl_per_device_high;
                mem            = mem_high;
                LOG_INF("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_INF(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LOG_INF("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start && id_dense_start < nd; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = jd < nd - 1 ? ngl_per_device_high[jd].n_layer : ngl_per_device_high[jd].n_layer - 1;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
            uint32_t delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_part > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LOG_INF("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                assert(ngl_per_device_high[id].n_full() >= ngl_per_device[id].n_full());
                delta = ngl_per_device_high[id].n_full() - ngl_per_device[id].n_full();
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            mem            = mem_high;
            id_dense_start = id_dense_start_high;
            LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > (id < nd - 1 ? 0 : 1)) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_part == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            std::vector<ggml_backend_buffer_type_t> overflow_bufts_test = overflow_bufts;
            if (id < nd - 1) {
                overflow_bufts_test[id] = ggml_backend_dev_buffer_type(devs[id + 1]);
            }
            LOG_INF("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
            if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                ngl_per_device = ngl_per_device_test;
                overflow_bufts = overflow_bufts_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LOG_INF("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LOG_INF("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts_test);
                if (mem_test[id] < targets[id] && (id + 1 == nd || mem_test[id + 1] < targets[id + 1])) {
                    ngl_per_device = ngl_per_device_test;
                    overflow_bufts = overflow_bufts_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LOG_INF("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_INF(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    // print info for devices that were not changed during the conversion from dense only to full layers:
    for (size_t id = id_dense_start + 1; id < nd; id++) {
        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LOG_INF(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams);
}

enum common_params_fit_status common_fit_params(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams,
        float * tensor_split,
        llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t * margins,
        uint32_t n_ctx_min,
        ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        common_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margins, n_ctx_min, log_level);
        LOG_INF("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_INF("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
}

void common_memory_breakdown_print(const struct llama_context * ctx) {
    //const auto & devices = ctx->get_model().devices;
    const auto * model = llama_get_model(ctx);

    std::vector<ggml_backend_dev_t> devices;
    for (int i = 0; i < llama_model_n_devices(model); i++) {
        devices.push_back(llama_model_get_device(model, i));
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx);

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t dev = devices[i];
        llama_memory_breakdown_data mb = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const int64_t unaccounted = static_cast<int64_t>(total) - static_cast<int64_t>(free) - static_cast<int64_t>(self);

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / static_cast<int64_t>(MiB))});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LOG_INF(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

void common_fit_print(
        const char * path_model,
        llama_model_params * mparams,
        llama_context_params * cparams) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    auto dmd = common_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);
    GGML_ASSERT(dmd.size() == devs.size() + 1);

    for (size_t id = 0; id < devs.size(); id++) {
        printf("%s ",  ggml_backend_dev_name(devs[id]));
        printf("%zu ", dmd[id].mb.model/1024/1024);
        printf("%zu ", dmd[id].mb.context/1024/1024);
        printf("%zu ", dmd[id].mb.compute/1024/1024);
        printf("\n");
    }

    printf("Host ");
    printf("%zu ", dmd.back().mb.model/1024/1024);
    printf("%zu ", dmd.back().mb.context/1024/1024);
    printf("%zu ", dmd.back().mb.compute/1024/1024);
    printf("\n");
}

#include "common.h.inc"

#include "ggml.h.inc"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,
    REASONING_BUDGET_COUNTING,
    REASONING_BUDGET_FORCING,
    REASONING_BUDGET_WAITING_UTF8,
    REASONING_BUDGET_DONE,
};

extern "C" {
struct llama_sampler * llama_common_reasoning_budget_init_rust(
        const struct llama_vocab * vocab,
        const llama_token        * start_tokens,
        size_t                     start_len,
        const llama_token        * end_tokens,
        size_t                     end_len,
        const llama_token        * forced_tokens,
        size_t                     forced_len,
        int32_t                    budget,
        common_reasoning_budget_state initial_state);

common_reasoning_budget_state llama_common_reasoning_budget_get_state_rust(const struct llama_sampler * smpl);
}

static struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab       * vocab,
        const std::vector<llama_token> & start_tokens,
        const std::vector<llama_token> & end_tokens,
        const std::vector<llama_token> & forced_tokens,
        int32_t                          budget,
        common_reasoning_budget_state    initial_state = REASONING_BUDGET_IDLE) {
    return llama_common_reasoning_budget_init_rust(
            vocab,
            start_tokens.data(), start_tokens.size(),
            end_tokens.data(), end_tokens.size(),
            forced_tokens.data(), forced_tokens.size(),
            budget,
            initial_state);
}

static common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    return llama_common_reasoning_budget_get_state_rust(smpl);
}

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h.inc
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return llama_common_ring_buffer_is_empty_rust(sz);
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(const struct llama_model * model, struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            params.reasoning_budget_start,
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    if (params.has_logit_bias()) {
        samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), params.logit_bias.size(), params.logit_bias.data()));
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // default: sample from distribution
            samplers.push_back(llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    // if grammar is lazy, only apply when reasoning budget is not active
    const auto state = (gsmpl->rbudget && gsmpl->params.grammar_lazy) ?
            common_reasoning_budget_get_state(gsmpl->rbudget) : REASONING_BUDGET_IDLE;
    return llama_common_grammar_should_apply_rust(
            gsmpl->grmr != nullptr,
            gsmpl->rbudget != nullptr,
            gsmpl->params.grammar_lazy,
            state,
            REASONING_BUDGET_IDLE,
            REASONING_BUDGET_DONE);
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
    };
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            // TODO: simplify
            gsmpl->cur.resize(1);
            gsmpl->cur[0] = { id, 0.0f, 1.0f };
            cur_p = { gsmpl->cur.data(), gsmpl->cur.size(), 0, true };

            return id;
        }
    }

    gsmpl->set_logits(ctx, idx);

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return llama_common_sampler_prob_desc_rust(a.p, b.p);
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    return llama_common_sampler_type_to_chr_rust((int) cnstr);
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    const char * name = llama_common_sampler_type_to_str_rust((int) cnstr);
    return name ? name : "";
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        const int sampler = llama_common_sampler_type_from_name_rust(name.c_str(), allow_alt_names);
        if (sampler >= 0) {
            samplers.push_back((common_sampler_type) sampler);
        } else {
            LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name.c_str());
        }
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const int sampler = llama_common_sampler_type_from_chr_rust(c);
        if (sampler >= 0) {
            samplers.push_back((common_sampler_type) sampler);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

#include "common.h.inc"
#include "ggml.h.inc"
#include "llama.h.inc"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>
#include <unordered_map>


#define CHATML_TEMPLATE_SRC                                                               \
    "{%- for message in messages -%}\n"                                                   \
    "  {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}\n" \
    "{%- endfor -%}\n"                                                                    \
    "{%- if add_generation_prompt -%}\n"                                                  \
    "  {{- '<|im_start|>assistant\n' -}}\n"                                               \
    "{%- endif -%}"

void common_chat_templates_free(struct common_chat_templates * tmpls) {
    delete tmpls;
}

bool common_chat_templates_was_explicit(const struct common_chat_templates * tmpls) {
    return tmpls->has_explicit_template;
}

// LFM2 format detection: template uses <|tool_list_start|>[...]<|tool_list_end|> around the tool list
// and <|tool_call_start|>[...]<|tool_call_end|> around each tool call
static bool is_lfm2_template(const std::string & src) {
    return llama_common_is_lfm2_template_rust((const uint8_t *) src.data(), src.size());
}

common_chat_prompt_preset common_chat_get_asr_prompt(const common_chat_templates * chat_templates) {
    common_chat_prompt_preset asr_preset;
    asr_preset.system = "";
    asr_preset.user   = "Transcribe audio to text";

    if (chat_templates && chat_templates->template_default && is_lfm2_template(chat_templates->template_default->source())) {
        asr_preset.system = "Perform ASR.";
        asr_preset.user   = "";
    }

    return asr_preset;
}

std::string common_chat_templates_source(const struct common_chat_templates * tmpls, const std::string & variant) {
    if (!variant.empty()) {
        if (variant == "tool_use") {
            if (tmpls->template_tool_use) {
                return tmpls->template_tool_use->source();
            }
            return "";
        }
        LOG_DBG("%s: unknown template variant: %s\n", __func__, variant.c_str());
    }
    return tmpls->template_default->source();
}

common_chat_templates_ptr common_chat_templates_init(const struct llama_model * model,
                                                     const std::string &        chat_template_override,
                                                     const std::string &        bos_token_override,
                                                     const std::string &        eos_token_override) {
    std::string default_template_src;
    std::string template_tool_use_src;

    bool has_explicit_template = !chat_template_override.empty();
    if (chat_template_override.empty()) {
        GGML_ASSERT(model != nullptr);
        const auto * str = llama_model_chat_template(model, /* name */ nullptr);
        if (str) {
            default_template_src  = str;
            has_explicit_template = true;
        }
        str = llama_model_chat_template(model, /* name */ "tool_use");
        if (str) {
            template_tool_use_src = str;
            has_explicit_template = true;
        }
    } else {
        default_template_src = chat_template_override;
    }
    if (default_template_src.empty() || default_template_src == "chatml") {
        if (!template_tool_use_src.empty()) {
            default_template_src = template_tool_use_src;
        } else {
            default_template_src = CHATML_TEMPLATE_SRC;
        }
    }

    // TODO @ngxson : this is a temporary hack to prevent chat template from throwing an error
    // Ref: https://github.com/ggml-org/llama.cpp/pull/15230#issuecomment-3173959633
    if (llama_common_needs_gpt_oss_channel_template_patch_rust(
            (const uint8_t *) default_template_src.data(), default_template_src.size())) {
        string_replace_all(default_template_src,
                           "{%- if \"<|channel|>analysis<|message|>\" in message.content or "
                           "\"<|channel|>final<|message|>\" in message.content %}",
                           "{%- if false %}");
    }

    // TODO @aldehir : this is a temporary fix, pending Minja changes
    // Ref: https://github.com/ggml-org/llama.cpp/pull/17713#issuecomment-3631342664
    if (llama_common_needs_mistral_tool_calls_template_patch_rust(
            (const uint8_t *) default_template_src.data(), default_template_src.size())) {
        string_replace_all(default_template_src,
                           "{%- if (message['content'] is none or message['content'] == '' or "
                           "message['content']|length == 0) and (message['tool_calls'] is not defined or "
                           "message['tool_calls'] is none or message['tool_calls']|length == 0) %}",
                           "{%- if false %}");
    }

    std::string token_bos = bos_token_override;
    std::string token_eos = eos_token_override;
    bool        add_bos   = false;
    bool        add_eos   = false;
    if (model) {
        const auto * vocab     = llama_model_get_vocab(model);
        const auto   get_token = [&](llama_token token, const char * name, const char * jinja_variable_name) {
            if (token == LLAMA_TOKEN_NULL) {
                if (default_template_src.find(jinja_variable_name) != std::string::npos ||
                    template_tool_use_src.find(jinja_variable_name) != std::string::npos) {
                    LOG_WRN(
                        "common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't "
                          "work as intended.\n",
                        name);
                }
                return std::string();
            }
            return common_token_to_piece(vocab, token, true);
        };
        token_bos = get_token(llama_vocab_bos(vocab), "BOS", "bos_token");
        token_eos = get_token(llama_vocab_eos(vocab), "EOS", "eos_token");
        add_bos   = llama_vocab_get_add_bos(vocab);
        add_eos   = llama_vocab_get_add_eos(vocab);
    }
    common_chat_templates_ptr tmpls(new common_chat_templates());
    tmpls->has_explicit_template = has_explicit_template;
    tmpls->add_bos               = add_bos;
    tmpls->add_eos               = add_eos;
    try {
        tmpls->template_default = std::make_unique<common_chat_template>(default_template_src, token_bos, token_eos);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        LOG_ERR("%s: failed to initialize chat template\n", __func__);
        LOG_ERR("%s: please consider disabling jinja via --no-jinja, or using another chat template\n", __func__);
        throw e;
    }
    if (!template_tool_use_src.empty()) {
        try {
            tmpls->template_tool_use = std::make_unique<common_chat_template>(template_tool_use_src, token_bos, token_eos);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, e.what());
        }
    }
    return tmpls;
}

const char * common_chat_format_name(common_chat_format format) {
    const char * name = llama_common_chat_format_name_rust((int) format);
    if (name) {
        return name;
    }
    throw std::runtime_error("Unknown chat format");
}

const char * common_reasoning_format_name(common_reasoning_format format) {
    const char * name = llama_common_reasoning_format_name_rust((int) format);
    if (name) {
        return name;
    }
    throw std::runtime_error("Unknown reasoning format");
}

common_reasoning_format common_reasoning_format_from_name(const std::string & format) {
    const int parsed = llama_common_reasoning_format_from_name_rust(format.c_str());
    if (parsed >= 0) {
        return (common_reasoning_format) parsed;
    }
    throw std::runtime_error("Unknown reasoning format: " + format);
}

static void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            LOG_INF("Skipping tool without function: %s", tool.dump(2).c_str());
            continue;
        }
        fn(tool);
    }
}

static void foreach_parameter(const json &                                                         function,
                              const std::function<void(const std::string &, const json &, bool)> & fn) {
    if (!function.contains("parameters") || !function.at("parameters").is_object()) {
        return;
    }
    const auto & params = function.at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        return;
    }
    const auto &          props = params.at("properties");
    std::set<std::string> required;
    if (params.contains("required") && params.at("required").is_array()) {
        params.at("required").get_to(required);
    }
    for (const auto & [name, prop] : props.items()) {
        bool is_required = (required.find(name) != required.end());
        fn(name, prop, is_required);
    }
}

static std::string common_chat_template_direct_apply_impl(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt) {
    jinja::context ctx(tmpl.source());

    nlohmann::ordered_json inp = nlohmann::ordered_json{
        {"messages", messages_override.has_value() ? *messages_override : inputs.messages},
        {"bos_token", tmpl.bos_token()},
        {"eos_token", tmpl.eos_token()},
        {"enable_thinking", inputs.enable_thinking},
    };
    if (tools_override.has_value() || !inputs.tools.empty()) {
        inp["tools"] = tools_override.has_value() ? *tools_override : inputs.tools;
    }
    if (inputs.extra_context.is_object()) {
        // TODO: do we need to merge, or replacing is fine?
        for (const auto & [k, v] : inputs.extra_context.items()) {
            inp[k] = v;
        }
    }
    if (additional_context.has_value()) {
        // TODO: merge properly instead of overwriting (matching old behavior)
        for (const auto & [k, v] : additional_context->items()) {
            inp[k] = v;
        }
    }
    if (inputs.add_generation_prompt) {
        inp["add_generation_prompt"] = true;
    }

    jinja::global_from_json(ctx, inp, inputs.mark_input);

    // render
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(tmpl.prog);
    auto parts = jinja::runtime::gather_string_parts(results);

    std::string result = parts->as_string().str();

    // TODO: improve this later
    if (inputs.add_bos && string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(result, tmpl.eos_token())) {
        result = result.substr(0, result.size() - tmpl.eos_token().size());
    }
    return result;
}

std::string common_chat_template_direct_apply(
    const common_chat_template & tmpl,
    const autoparser::generation_params & inputs) {
    return common_chat_template_direct_apply_impl(tmpl, inputs, std::nullopt, std::nullopt, std::nullopt);
}

static common_chat_params common_chat_params_init_ministral_3(const common_chat_template &    tmpl,
                                                              const autoparser::generation_params & inputs) {
    common_chat_params data;

    // Build up messages to follow the format: https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512/blob/main/chat_template.jinja
    auto adjusted_messages = json::array();
    for (const auto & msg : inputs.messages) {
        auto role = msg.value("role", "");
        if (role != "system" && role != "assistant") {
            // Only adjust system and assistant messages. Interestingly, the system message may contain thinking.
            adjusted_messages.push_back(msg);
            continue;
        }

        auto content = json::array();

        // If message contains `reasoning_content`, add it as a block of type `thinking`
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            content.push_back({
                { "type",     "thinking"                                     },
                { "thinking", msg.at("reasoning_content").get<std::string>() },
            });
        }

        // If message contains `content`, add it as a block of type `text`
        if (msg.contains("content")) {
            if (msg.at("content").is_string()) {
                content.push_back({
                    { "type", "text"                               },
                    { "text", msg.at("content").get<std::string>() },
                });
            } else if (msg.at("content").is_array()) {
                auto blocks = msg.at("content");
                content.insert(content.end(), blocks.begin(), blocks.end());
            }
        }

        auto adjusted       = msg;
        adjusted["content"] = content;
        adjusted.erase("reasoning_content");
        adjusted_messages.push_back(adjusted);
    }

    auto has_tools            = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format  = inputs.json_schema.is_object() && !inputs.json_schema.empty();
    auto extract_reasoning    = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar      = true;

    data.supports_thinking  = true;
    data.thinking_start_tag = "[THINK]";
    data.thinking_end_tag   = "[/THINK]";
    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, /* messages_override = */ adjusted_messages);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = {
        "[THINK]",
        "[/THINK]",
        "[TOOL_CALLS]",
        "[ARGS]",
    };

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(inputs.generation_prompt, "[THINK]");
        auto reasoning =
            extract_reasoning ? p.optional("[THINK]" + p.reasoning(p.until("[/THINK]")) + "[/THINK]") : p.eps();

        // Response format parser
        if (has_response_format) {
            // Ministral wants to emit json surrounded by code fences
            return generation_prompt + (reasoning << "```json" << p.content(p.schema(p.json(), "response-format", inputs.json_schema)) << "```");
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                const auto & schema   = function.at("parameters");

                tool_choice |=
                    p.rule("tool-" + name, p.tool_open(p.tool_name(p.literal(name)) + "[ARGS]") +
                                               p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)));
            });

            auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call", p.repeat("[TOOL_CALLS]" + tool_choice, min_calls, max_calls));

            return generation_prompt + (reasoning << p.content(p.until("[TOOL_CALLS]")) << tool_calls);
        }

        // Content only parser
        include_grammar = false;
        return generation_prompt + (reasoning << p.content(p.rest()));
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]" }
        };
    }

    return data;
}

static common_chat_params common_chat_params_init_gpt_oss(const common_chat_template &    tmpl,
                                                          const autoparser::generation_params & inputs) {
    common_chat_params data;

    // Copy reasoning to the "thinking" field as expected by the gpt-oss template
    auto adjusted_messages = json::array();
    for (auto msg : inputs.messages) {
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            msg["thinking"] = msg.at("reasoning_content");
            if (msg.contains("tool_calls") && msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty()) {
                msg.erase("content");
            }
        }
        adjusted_messages.push_back(msg);
    }

    auto prompt = common_chat_template_direct_apply_impl(tmpl, inputs, /* messages_override= */ adjusted_messages);

    // Check if we need to replace the return token with end token during
    // inference and without generation prompt. For more details see:
    // https://github.com/ggml-org/llama.cpp/issues/15417
    if (inputs.is_inference && !inputs.add_generation_prompt) {
        static constexpr std::string_view return_token = "<|return|>";
        static constexpr std::string_view end_token    = "<|end|>";
        if (size_t pos = prompt.rfind(return_token); pos != std::string::npos) {
            prompt.replace(pos, return_token.length(), end_token);
        }
    }

    data.prompt            = prompt;
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;

    // These special tokens are required to parse properly, so we include them
    // even if parse_tool_calls is false.
    data.preserved_tokens = {
        "<|channel|>", "<|constrain|>", "<|message|>", "<|start|>", "<|end|>",
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start           = p.rule("start", p.literal("<|start|>assistant"));
        auto end             = p.rule("end", p.literal("<|end|>"));
        auto content         = p.rule("message-content", p.until("<|end|>"));
        auto channel         = p.literal("<|channel|>") + (p.literal("commentary") | p.literal("analysis"));
        auto constrain_type  = p.chars("[A-Za-z0-9_-]", 1, -1);

        // Occasionally, gpt-oss-20b will prefix channels with this commentary
        auto stray_commentary = p.optional(p.literal("<|channel|>commentary") + p.optional(p.literal(" to=assistant")));
        auto start_analysis = stray_commentary + p.literal("<|channel|>analysis<|message|>");

        if (extract_reasoning) {
            p.rule("analysis", start_analysis + p.reasoning(content) + end);
        } else {
            p.rule("analysis", p.content(start_analysis + content + end));
        }

        auto analysis = p.ref("analysis");
        auto preamble = p.rule("preamble", p.literal("<|channel|>commentary<|message|>") + p.content(content) + end);
        auto final_msg = p.rule("final", stray_commentary + p.literal("<|channel|>final<|message|>") + p.content(content));

        // Consume any unsolicited tool calls, e.g. builtin functions
        auto unsolicited = p.rule("unsolicited", p.atomic(p.optional(channel) + p.literal(" to=") + content + end));

        auto any = p.rule("any", preamble | analysis);

        if (has_response_format) {
            auto constraint = p.optional(p.space() + p.optional(p.literal("<|constrain|>")) + constrain_type);
            auto response_format = p.rule("response-format",
                p.literal("<|channel|>final") + constraint + p.literal("<|message|>") +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)));

            return p.zero_or_more(start + analysis) + start + response_format;
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                const auto & params   = function.at("parameters");

                auto func_name  = p.literal(" to=functions.") + p.tool_name(p.literal(name));
                auto constraint = p.optional(p.space() + p.optional(p.literal("<|constrain|>")) + constrain_type);
                auto args       = p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", params));

                // recipient in role header
                //   <|start|>assistant to=functions.NAME<|channel|>(commentary|analysis)[constraint]<|message|>ARGS
                auto tool_in_role = p.tool(p.tool_open(func_name + channel + constraint + p.literal("<|message|>")) + args);

                // recipient in channel header
                //   <|channel|>(commentary|analysis) to=functions.NAME[constraint]<|message|>ARGS
                auto tool_in_channel = p.tool(p.tool_open(channel + func_name + constraint + p.literal("<|message|>")) + args);

                tool_choice |= p.rule("tool-" + name, tool_in_role | tool_in_channel);
            });

            auto tool_call  = p.trigger_rule("tool-call", tool_choice);

            if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
                return p.zero_or_more(start + any) + start + tool_call;
            }

            return p.zero_or_more(start + any) + start + (tool_call | final_msg);
        }

        return p.zero_or_more(start + any) + start + (final_msg | unsolicited);
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "^\\s+to$" },
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "^<\\|channel\\|>(?:commentary|analysis)\\s+to=functions$" },
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|start\\|>assistant(\\s+to)" },
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "<\\|start\\|>assistant(<\\|channel\\|>(?:commentary|analysis)\\s+to)" }
        };
    }

    return data;
}

static common_chat_params common_chat_params_init_gemma4(const common_chat_template &    tmpl,
                                                         const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);

    if (inputs.add_generation_prompt && string_ends_with(data.prompt, "<turn|>\n")) {
        // This may happen if the model generates content + tool_call, the
        // template does not add the model's next turn and confuses the model
        // from emitting its proper reasoning token sequence.
        data.prompt += "<|turn>model\n";
    }

    data.format            = COMMON_CHAT_FORMAT_PEG_GEMMA4;
    data.supports_thinking  = true;
    data.thinking_start_tag = "<|channel>thought";
    data.thinking_end_tag   = "<channel|>";

    data.preserved_tokens = {
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|turn>",
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto start = p.rule("start", p.prefix(inputs.generation_prompt, "<|channel>"));

        if (extract_reasoning) {
            p.rule("thought", p.literal("<|channel>thought") + p.space() + p.reasoning(p.until("<channel|>")) + p.literal("<channel|>"));
        } else {
            p.rule("thought", p.content(p.literal("<|channel>thought") + p.space() + p.until("<channel|>") + p.literal("<channel|>")));
        }

        auto consume_empty_channels = p.gbnf(p.zero_or_more(p.literal("<|channel>") + p.negate(p.literal("thought"))), "");
        auto thought = (p.peek(p.literal("<|channel>")) + consume_empty_channels + p.ref("thought")) | p.negate(p.literal("<|channel>"));

        if (has_response_format) {
            auto response_format = p.literal("```json") <<
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) <<
                p.literal("```");
            return start + p.optional(thought) + response_format;
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // Gemma4 tool calling syntax
            // Rules should match traversal logic in gemma4_to_json()
            p.rule("gemma4-string-content", p.until("<|\"|>"));
            p.rule("gemma4-string", p.literal("<|\"|>") + p.ref("gemma4-string-content") + p.literal("<|\"|>"));
            p.rule("gemma4-bool", p.json_bool());
            p.rule("gemma4-null", p.json_null());
            p.rule("gemma4-number", p.json_number());
            p.rule("gemma4-dict-key", p.rule("gemma4-dict-key-name", p.chars("[^:}]", 1, -1)) + p.literal(":"));
            p.rule("gemma4-dict-kv", p.ref("gemma4-dict-key") + p.space() + p.ref("gemma4-value"));
            p.rule("gemma4-dict", [&]() {
                auto ws = p.space();
                auto member = p.ref("gemma4-dict-kv");
                auto members = p.sequence({member, p.zero_or_more(p.sequence({p.literal(","), ws, member}))});
                return p.sequence({
                    p.literal("{"), ws,
                    p.choice({p.literal("}"), p.sequence({members, ws, p.literal("}")})})
                });
            });
            p.rule("gemma4-array", [&]() {
                auto ws = p.space();
                auto value = p.ref("gemma4-value");
                auto elements = p.sequence({value, p.zero_or_more(p.sequence({p.literal(","), ws, value}))});
                return p.sequence({
                    p.literal("["), ws,
                    p.choice({p.literal("]"), p.sequence({elements, ws, p.literal("]")})})
                });
            });
            p.rule("gemma4-value", [&]() {
                return p.choice({
                    p.ref("gemma4-string"), p.ref("gemma4-dict"), p.ref("gemma4-array"),
                    p.ref("gemma4-number"), p.ref("gemma4-bool"), p.ref("gemma4-null")
                });
            });

            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string  name     = function.at("name");
                // TODO @aldehir : need to extend json-schema-to-grammar to produce more than JSON rules
                // const auto & params   = function.at("parameters");

                tool_choice |= p.rule("tool-" + name, p.tool(p.sequence({
                    p.tool_open(p.tool_name(p.literal(name)) + p.peek(p.literal("{"))),
                    p.tool_args(p.ref("gemma4-dict")),
                })));
            });

            auto tool_call = p.trigger_rule("tool-call", p.repeat(
                "<|tool_call>call:" + tool_choice + "<tool_call|>",
                /* min = */ inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0,
                /* max = */ inputs.parallel_tool_calls ? -1 : 1
            ));

            auto scan_to_toolcall = p.rule("scan-to-toolcall", p.until("<|tool_call>"));
            auto content = p.rule("content", p.content(p.until_one_of({"<|channel>", "<channel|>", "<|tool_call>"})));
            auto message = p.rule("message", thought + content);
            return start + p.zero_or_more(message) + scan_to_toolcall + tool_call;
        }

        // Gemma 4 may emit an extra <|channel>thought\n<channel|> at the end of the content. It may
        // also emit a single trailing <channel|> token. Consume all complete reasoning blocks and
        // then stop at the first unmatched <channel|> token.
        auto content = p.rule("content", p.content(p.until_one_of({"<|channel>", "<channel|>"})));
        auto message = p.rule("message", thought + content);
        return start + p.one_or_more(message);
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|tool_call>" },
        };
    }

    return data;
}

// Functionary v3.2 - uses recipient-based format: >>>recipient\n{content}
static common_chat_params common_chat_params_init_functionary_v3_2(const common_chat_template &    tmpl,
                                                                   const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt           = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format           = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens = {
        ">>>all",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // Functionary v3.2 format:
        // - Normal content: >>>all\n{content}
        // - Tool calls: >>>function_name\n{json_args}
        // Generation prompt ends with ">>>" so model outputs recipient immediately

        // Build content parser for >>>all\n{content}
        // When tools are present, content stops before the next ">>>" (tool call)
        // When no tools, content goes until end
        auto content_until_tool = p.literal("all\n") + p.content(p.until(">>>"));
        auto content_until_end  = p.literal("all\n") + p.content(p.rest());
        auto generation_prompt  = p.literal(inputs.generation_prompt);

        // If no tools or tool_choice is NONE, just parse content
        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            // When no tools, just match the prefix and capture everything after
            return generation_prompt + content_until_end + p.end();
        }

        // Build tool call parsers for each available function
        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto & schema   = function.at("parameters");

            // Tool format: >>>function_name\n{json_args}
            auto tool_parser = p.tool(
                p.tool_open(p.tool_name(p.literal(name)) + p.literal("\n")) +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema))
            );

            tool_choice |= p.rule("tool-" + name, tool_parser);
        });

        auto content_only = content_until_end;
        auto tools_only = p.trigger_rule("tools", p.one_or_more(tool_choice));
        auto content_and_tools = content_until_tool + tools_only;

        auto ret = p.eps();
        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            if (inputs.parallel_tool_calls) {
                ret = p.choice({ content_and_tools, tools_only }) + p.end();
            } else {
                ret = p.choice({ content_until_tool + tool_choice, tools_only }) + p.end();
            }
        } else if (inputs.parallel_tool_calls) {
            ret = p.choice({ content_and_tools, content_only, tools_only }) + p.end();
        } else {
            auto content_and_tool = content_until_tool + tool_choice;
            ret = p.choice({ content_and_tool, content_only, tool_choice }) + p.end();
        }
        return generation_prompt + ret;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // Grammar trigger for when the model starts outputting a tool call
        // (after the initial ">>>" in the generation prompt but recipient other than "all")
        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, ">>>(?!all)" }
        };
    }

    return data;
}

// Kimi K2 Thinking - uses unique tool call ID format: functions.<name>:<index>
// The ID contains both the function name and an incrementing counter
static common_chat_params common_chat_params_init_kimi_k2(const common_chat_template &    tmpl,
                                                          const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt             = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format             = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking  = true;
    data.preserved_tokens  = {
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_argument_begin|>",
        "<|tool_call_end|>",
        "<think>",
        "</think>",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    const std::string SECTION_BEGIN = "<|tool_calls_section_begin|>";
    const std::string SECTION_END   = "<|tool_calls_section_end|>";
    const std::string CALL_BEGIN    = "<|tool_call_begin|>";
    const std::string ARGS_BEGIN    = "<|tool_call_argument_begin|>";
    const std::string CALL_END      = "<|tool_call_end|>";

    const std::string THINK_START = "<think>";
    const std::string THINK_END   = "</think>";

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tag   = THINK_END;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        // Kimi K2 Thinking format:
        // - Reasoning: <think>{reasoning}</think>
        // - Content: text after reasoning
        // - Tool calls section:
        //   <|tool_calls_section_begin|>
        //   <|tool_call_begin|>functions.<name>:<index><|tool_call_argument_begin|>{json_args}<|tool_call_end|>
        //   ...
        //   <|tool_calls_section_end|>
        // The ID format is: functions.<function_name>:<counter> where counter is 0, 1, 2, ...

        // Tool call markers
        auto end = p.end();

        // Note: this model is CRAZY. It can diverge from its supposed tool calling pattern in so many ways it's not funny.
        // For example, it can call tools at the end of reasoning without closing reasoning...
        auto reasoning = extract_reasoning ? p.optional(THINK_START + p.reasoning(
            p.until_one_of({ THINK_END, "<|tool_calls_section_begin|>", "<|tool_call_begin|>" })) +
            p.optional(p.literal(THINK_END))) : p.eps();
        auto generation_prompt = p.prefix(inputs.generation_prompt, THINK_START);


        // Content only parser (no tools)
        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        // Build tool call parsers for each available function
        // The ID format is: functions.<name>:<index>
        // We need to match: functions.<name>:<digits>
        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto & schema   = function.at("parameters");

            // Match: functions.<name>:<digits>
            // Capture the full call id (functions.<name>:<digits>) using tool_id tag
            auto tool_id = p.tool_id(p.literal("functions.") + p.tool_name(p.literal(name)) + p.literal(":") + p.chars("[0-9]", 1, -1));
            auto tool_parser = p.tool(
                p.tool_open(tool_id + p.literal(ARGS_BEGIN)) +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)) +
                p.tool_close(p.optional((p.literal(CALL_END))))
            );

            tool_choice |= p.rule("tool-" + name, tool_parser);
        });

        // Tool calls section: <|tool_calls_section_begin|> tool_calls <|tool_calls_section_end|>
        auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
        auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
        // Use trigger_rule so grammar generator knows where to start generating rules
        auto tool_calls = p.rule("tool-calls",
            p.optional(p.literal(SECTION_BEGIN)) +
            p.trigger_rule("tool-call", p.repeat(CALL_BEGIN + tool_choice, min_calls, max_calls) +
                p.optional(p.literal(SECTION_END)))
        );

        auto content_before_tools = p.content(p.until_one_of({ SECTION_BEGIN, CALL_BEGIN }));

        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<|tool_call_begin|>" }
        };
    }

    return data;
}

// LFM2 format: uses <|tool_list_start|>[...]<|tool_list_end|> in system prompt
// and <|tool_call_start|>[name(arg="val")]<|tool_call_end|> for tool calls.
// - Reasoning: <think>{reasoning}</think> (optional)
// - Content: text before a tool call (optional)
// - Tool calls: Python-style, e.g. [function_name(arg1="value1", arg2="value2")]
//   Tool calls can appear multiple times (parallel tool calls supported)
static common_chat_params common_chat_params_init_lfm2(const common_chat_template &    tmpl,
                                                       const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.preserved_tokens  = {
        "<|tool_list_start|>",
        "<|tool_list_end|>",
        "<|tool_call_start|>",
        "<|tool_call_end|>",
        "<think>",
        "</think>",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    const std::string TOOL_CALL_START = "<|tool_call_start|>";
    const std::string TOOL_CALL_END   = "<|tool_call_end|>";
    const std::string THINK_START     = "<think>";
    const std::string THINK_END       = "</think>";

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tag   = THINK_END;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(inputs.generation_prompt, THINK_START);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }
        auto tool_calls = p.rule("tool-calls",
            p.trigger_rule("tool-call",
                p.literal(TOOL_CALL_START) +
                p.python_style_tool_calls(inputs.tools, inputs.parallel_tool_calls) +
                p.literal(TOOL_CALL_END)
            )
        );

        auto content = p.content(p.until(TOOL_CALL_START));

        return generation_prompt + reasoning + content + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, TOOL_CALL_START }
        };
    }
    return data;
}

// LFM2.5 format: uses plain "List of tools: [...]" in system prompt, no wrapper tokens.
// Tool calls are bare [name(arg="val")], though model may optionally emit <|tool_call_start|>.
// - Reasoning: <think>{reasoning}</think> (optional)
// - Content: text before a tool call (optional)
// - Tool calls: Python-style, e.g. [function_name(arg1="value1", arg2="value2")]
//   Tool calls can appear multiple times (parallel tool calls supported)
static common_chat_params common_chat_params_init_lfm2_5(const common_chat_template &    tmpl,
                                                         const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.preserved_tokens  = {
        "<|tool_call_start|>",
        "<|tool_call_end|>",
        "<think>",
        "</think>",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    const std::string THINK_START     = "<think>";
    const std::string THINK_END       = "</think>";

    data.thinking_start_tag = THINK_START;
    data.thinking_end_tag   = THINK_END;

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(inputs.generation_prompt, THINK_START);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_calls = p.rule("tool-calls",
            p.trigger_rule("tool-call",
                p.python_style_tool_calls(inputs.tools, inputs.parallel_tool_calls)
            )
        );

        auto content = p.content(p.until_one_of({"<|tool_call_start|>", "["}));
        auto maybe_start = p.optional(p.literal("<|tool_call_start|>"));
        return generation_prompt + reasoning + content + maybe_start + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });
        foreach_function(inputs.tools, [&](const json & tool) {
            const std::string name = tool.at("function").at("name");
            data.grammar_triggers.push_back({ COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[" + name + "(" });
        });
    }

    return data;
}

static common_chat_params common_chat_params_init_gigachat_v3(
        const common_chat_template & tmpl,
        const autoparser::generation_params & inputs) {

    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = false;
    data.preserved_tokens  = {
        "<|message_sep|>\n\n",
        "<|role_sep|>\n",
    };

    auto has_tools         = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar   = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    const auto *tool_call_start_prefix = "<|message_sep|>\n\nfunction call<|role_sep|>\n";

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto ret = p.eps();
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // Build a choice of all available tools
            auto tool_choice = p.choice();
            for (const auto & tool : inputs.tools) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                const auto & schema = function.at("parameters");

                auto tool_name = p.json_member("name", "\"" + p.tool_name(p.literal(name)) + "\"");
                auto tool_args = p.json_member("arguments", p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema)));

                auto tool_open = p.tool_open(p.literal("{") << tool_name);

                tool_choice |= p.rule("tool-" + name, tool_open << "," << tool_args << "}");
            }

            // Define the tool call structure
            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = 1; // parallel toolcalls are not supported
            auto tool_call = p.rule("tool-call", p.literal(tool_call_start_prefix) + tool_choice);
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, /* min = */ min_calls, /* max = */ max_calls));

            ret = p.content(p.until("<|message_sep|>\n\n")) << tool_calls;
        } else {
            // Content only parser
            include_grammar = false;
            ret = p.content(p.rest());
        }

        return p.literal(inputs.generation_prompt) + ret;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto schema = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            {COMMON_GRAMMAR_TRIGGER_TYPE_WORD, tool_call_start_prefix}
        };
    }
    return data;
}

static common_chat_params common_chat_params_init_deepseek_v3_2(const common_chat_template &    tmpl,
                                                                 const autoparser::generation_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.thinking_start_tag = "<think>";
    data.thinking_end_tag   = "</think>";
    data.preserved_tokens  = {
        "｜DSML｜",
        "<think>",
        "</think>",
    };

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto extract_reasoning   = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    auto include_grammar     = has_response_format || (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE);

    const std::string DSML         = "｜DSML｜";
    const std::string THINK_START  = "<think>";
    const std::string THINK_END    = "</think>";
    const std::string FC_START     = "<" + DSML + "function_calls>";
    const std::string FC_END       = "</" + DSML + "function_calls>";
    const std::string INVOKE_START = "<" + DSML + "invoke";
    const std::string INVOKE_END   = "</" + DSML + "invoke>";
    const std::string PARAM_START  = "<" + DSML + "parameter";
    const std::string PARAM_END    = "</" + DSML + "parameter>";

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.prefix(inputs.generation_prompt, THINK_START);
        auto end = p.end();

        auto reasoning = p.eps();
        if (extract_reasoning && inputs.enable_thinking) {
            reasoning = p.optional(THINK_START + p.reasoning(p.until(THINK_END)) + THINK_END);
        } else if (extract_reasoning) {
            // Thinking disabled but reasoning extraction requested: the generation prompt
            // contains an empty <think></think> pair that must still be consumed.
            reasoning = p.optional(p.literal(THINK_START) + p.until(THINK_END) + p.literal(THINK_END));
        }

        if (has_response_format) {
            auto response_format = p.rule("response-format",
                p.literal("```json") + p.space() +
                p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) +
                p.space() + p.literal("```"));
            return generation_prompt + reasoning + response_format + end;
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return generation_prompt + reasoning + p.content(p.rest()) + end;
        }

        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            auto params   = function.contains("parameters") ? function.at("parameters") : json::object();
            const auto & props    = params.contains("properties") ? params.at("properties") : json::object();

            std::set<std::string> required;
            if (params.contains("required")) {
                params.at("required").get_to(required);
            }

            auto schema_info = common_schema_info();
            schema_info.resolve_refs(params);

            std::vector<common_peg_parser> required_parsers;
            std::vector<common_peg_parser> optional_parsers;
            for (const auto & [param_name, param_schema] : props.items()) {
                bool is_required = required.find(param_name) != required.end();
                bool is_string   = schema_info.resolves_to_string(param_schema);

                auto arg = p.tool_arg(
                    p.tool_arg_open(
                        p.literal(PARAM_START + " name=\"") +
                        p.tool_arg_name(p.literal(param_name)) +
                        p.literal("\" string=\"" + std::string(is_string ? "true" : "false") + "\">")) +
                    (is_string
                         ? p.tool_arg_string_value(p.until(PARAM_END))
                         : p.tool_arg_json_value(p.schema(p.json(),
                                                          "tool-" + name + "-arg-" + param_name + "-schema",
                                                          param_schema, false))) +
                    p.tool_arg_close(p.literal(PARAM_END)));

                auto named_arg = p.rule("tool-" + name + "-arg-" + param_name, arg);
                if (is_required) {
                    required_parsers.push_back(named_arg);
                } else {
                    optional_parsers.push_back(named_arg);
                }
            }

            common_peg_parser args_seq = p.eps();
            for (size_t i = 0; i < required_parsers.size(); i++) {
                if (i > 0) {
                    args_seq = args_seq + p.space();
                }
                args_seq = args_seq + required_parsers[i];
            }

            if (!optional_parsers.empty()) {
                common_peg_parser any_opt = p.choice();
                for (const auto & opt : optional_parsers) {
                    any_opt |= opt;
                }
                args_seq = args_seq + p.repeat(p.space() + any_opt, 0, -1);
            }

            common_peg_parser invoke_body = args_seq;
            auto func_parser = p.tool(
                p.tool_open(p.literal(INVOKE_START + " name=\"") +
                            p.tool_name(p.literal(name)) + p.literal("\">\n")) +
                invoke_body + p.space() +
                p.tool_close(p.literal(INVOKE_END)));

            tool_choice |= p.rule("tool-" + name, func_parser);
        });

        auto require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

        common_peg_parser tool_calls = p.eps();
        if (inputs.parallel_tool_calls) {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice +
                p.zero_or_more(p.space() + tool_choice) + p.space() + p.literal(FC_END));
        } else {
            tool_calls = p.trigger_rule("tool-call",
                p.literal(FC_START) + p.space() + tool_choice + p.space() + p.literal(FC_END));
        }

        if (!require_tools) {
            tool_calls = p.optional(tool_calls);
        }

        auto content_before_tools = p.content(p.until(FC_START));
        return generation_prompt + reasoning + content_before_tools + tool_calls + end;
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = !(has_response_format || (has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED));
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.contains("parameters") ? function.at("parameters") : json::object();
                builder.resolve_refs(schema);
            });
            if (has_response_format) {
                auto schema = inputs.json_schema;
                builder.resolve_refs(schema);
            }
            parser.build_grammar(builder, data.grammar_lazy);
        });

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, FC_START },
        };
    }

    return data;
}

namespace workaround {

static void map_developer_role_to_system(json & messages) {
    for (auto & message : messages) {
        if (message.contains("role")) {
            if (message["role"] == "developer") {
                message["role"] = "system";
            }
        }
    }
}


// if first message is system and template does not support it, merge it with next message
static void system_message_not_supported(json & messages) {
    if (!messages.empty() && messages.front().at("role") == "system") {
        if (messages.size() > 1) {
            LOG_DBG("Merging system prompt into next message\n");
            auto & first_msg = messages.front();
            auto & second_msg = messages[1];
            second_msg["content"] = first_msg.at("content").get<std::string>()
                + "\n" + second_msg.at("content").get<std::string>();
            messages.erase(messages.begin());
        } else {
            LOG_WRN("Removing system prompt due to template not supporting system role\n");
            messages.erase(messages.begin());
        }
    }
}

static void requires_non_null_content(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls") && !message.contains("content")) {
            message["content"] = "";
        }
    }
}

// Gemma4 uses a custom tool_responses field instead of role:tool messages.
//
// This will transform a sequence of messages:
//   assistant(tool_call+) -> tool+ -> assistant(content)
//
// Into a single assistant message containing a tool_responses field:
//   assistant(content + tool_call + tool_responses)
//
// This is necessary for the Gemma4 chat template to properly format the prompt.
// See https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
struct gemma4_model_turn_builder {
    json & messages;
    size_t pos;
    json tool_calls = json::array();
    json tool_responses = json::array();
    json content;
    json reasoning_content;

    gemma4_model_turn_builder(json & msgs, size_t pos) : messages(msgs), pos(pos) {}

    void collect() {
        // Collect the first assistant message
        auto & msg = messages[pos];
        if (msg.contains("reasoning_content") && msg.at("reasoning_content").is_string()) {
            // According to the prompt formatting guide, we need to preserve reasoning_content
            // between function calls. The current chat templates do not support this, but we will do it anyway.
            reasoning_content = msg.at("reasoning_content");
        }
        for (auto & tc : msg.at("tool_calls")) {
            tool_calls.push_back(tc);
        }
        pos++;

        // Collect tool call results
        while (pos < messages.size() && messages[pos].value("role", "") == "tool") {
            collect_result(messages[pos]);
            pos++;
        }

        // Check if the next assistant message is the final message
        if (pos < messages.size() && messages[pos].value("role", "") == "assistant") {
            auto & next = messages[pos];
            if (!has_tool_calls(next) && has_content(next)) {
                content = next.at("content");
                pos++;
            }
        }
    }

    void collect_result(const json & curr) {
        json response;
        if (curr.contains("content")) {
            const auto & content = curr.at("content");
            if (content.is_string()) {
                // Try to parse the content as JSON; fall back to raw string
                try {
                    response = json::parse(content.get<std::string>());
                } catch (...) {
                    response = content;
                }
            } else {
                response = content;
            }
        }

        std::string name;

        // Match name with corresponding tool call
        size_t idx = tool_responses.size();
        if (idx < tool_calls.size()) {
            auto & tc = tool_calls[idx];
            if (tc.contains("function")) {
                name = tc.at("function").value("name", "");
            }
        }

        // Fallback to the tool call id
        if (name.empty()) {
            name = curr.value("tool_call_id", "");
        }

        tool_responses.push_back({{"name", name}, {"response", response}});
    }

    json build() {
        collect();

        json msg = {
            {"role", "assistant"},
            {"tool_calls", tool_calls},
        };
        if (!tool_responses.empty()) {
            msg["tool_responses"] = tool_responses;
        }
        if (!content.is_null()) {
            msg["content"] = content;
        }
        if (!reasoning_content.is_null()) {
            msg["reasoning_content"] = reasoning_content;
        }
        return msg;
    }

    static bool has_content(const json & msg) {
        if (!msg.contains("content") || msg.at("content").is_null()) {
            return false;
        }
        const auto & content = msg.at("content");
        if (content.is_string() && !content.get<std::string>().empty()) {
            return true;
        }
        if (content.is_array() && !content.empty()) {
            return true;
        }
        return false;
    }

    static bool has_tool_calls(const json & msg) {
        return msg.contains("tool_calls") && msg.at("tool_calls").is_array() && !msg.at("tool_calls").empty();
    }
};

static void convert_tool_responses_gemma4(json & messages) {
    json result = json::array();
    size_t i = 0;

    while (i < messages.size()) {
        auto & msg = messages[i];

        if (msg.value("role", "") != "assistant" || !msg.contains("tool_calls") ||
            !msg.at("tool_calls").is_array() || msg.at("tool_calls").empty()) {
            result.push_back(msg);
            i++;
            continue;
        }

        gemma4_model_turn_builder builder(messages, i);
        result.push_back(builder.build());
        i = builder.pos;
    }

    messages = result;
}

static void func_args_not_string(json & messages) {
    GGML_ASSERT(messages.is_array());
    for (auto & message : messages) {
        if (message.contains("tool_calls")) {
            for (auto & tool_call : message["tool_calls"]) {
                if (tool_call.contains("function") && tool_call["function"].contains("arguments")) {
                    auto & args = tool_call["function"]["arguments"];
                    if (args.is_string()) {
                        try {
                            args = json::parse(args.get<std::string>());
                        } catch (const std::exception & e) {
                            throw std::runtime_error("Failed to parse tool call arguments as JSON: " + std::string(e.what()));
                        }
                    }
                }
            }
        }
    }
}

}

static json common_chat_extra_context() {
    json ctx = json::object();
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::string datetime_str = format_time(now, "%b %d %Y");
    std::string date_str = format_time(now, "%d %b %Y");
    ctx["datetime"] = datetime_str;
    ctx["date_string"] = date_str;
    return ctx;
}

std::optional<common_chat_params> common_chat_try_specialized_template(
        const common_chat_template &          tmpl,
        const std::string &                   src,
        autoparser::generation_params & params) {
    switch (llama_common_specialized_chat_template_rust((const uint8_t *) src.data(), src.size())) {
    case 1:
        LOG_DBG("Using specialized template: Ministral/Magistral Large 3\n");
        return common_chat_params_init_ministral_3(tmpl, params);
    case 2:
        LOG_DBG("Using specialized template: GPT-OSS\n");
        return common_chat_params_init_gpt_oss(tmpl, params);
    case 3:
        LOG_DBG("Using specialized template: Functionary v3.2\n");
        return common_chat_params_init_functionary_v3_2(tmpl, params);
    case 4:
        LOG_DBG("Using specialized template: Kimi K2 Thinking\n");
        return common_chat_params_init_kimi_k2(tmpl, params);
    case 5:
        LOG_DBG("Using specialized template: LFM2\n");
        return common_chat_params_init_lfm2(tmpl, params);
    case 6:
        LOG_DBG("Using specialized template: LFM2.5\n");
        return common_chat_params_init_lfm2_5(tmpl, params);
    case 7:
        LOG_DBG("Using specialized template: GigaChatV3\n");
        return common_chat_params_init_gigachat_v3(tmpl, params);
    case 8:
        LOG_DBG("Using specialized template: DeepSeek V3.2\n");
        return common_chat_params_init_deepseek_v3_2(tmpl, params);
    case 9:
        if (!llama_common_is_gemma4_modern_template_rust((const uint8_t *) src.data(), src.size())) {
            // apply workarounds if using the older gemma4 templates
            LOG_WRN("%s: detected an outdated gemma4 chat template, applying compatibility workarounds. "
                    "Consider updating to the official template.\n", __func__);
            workaround::convert_tool_responses_gemma4(params.messages);
        }
        return common_chat_params_init_gemma4(tmpl, params);
    default:
        break;
    }

    return std::nullopt;
}

static common_chat_params common_chat_templates_apply_jinja(const struct common_chat_templates *        tmpls,
                                                            const struct common_chat_templates_inputs & inputs) {
    autoparser::generation_params params;
    params.tools = common_chat_tools_to_json_oaicompat(inputs.tools);
    const auto & tmpl =
        params.tools.is_array() && tmpls->template_tool_use ? *tmpls->template_tool_use : *tmpls->template_default;
    const auto & src        = tmpl.source();
    const auto & caps       = tmpl.original_caps();
    params.messages         = render_message_to_json(inputs.messages, tmpl.original_caps());
    params.tool_choice      = inputs.tool_choice;
    params.reasoning_format = inputs.reasoning_format;
    params.enable_thinking  = inputs.enable_thinking;
    params.grammar          = inputs.grammar;
    params.now              = inputs.now;
    params.add_bos          = tmpls->add_bos;
    params.add_eos          = tmpls->add_eos;

    if (src.find("<|channel|>") == std::string::npos) {
        // map developer to system for all models except for GPT-OSS
        workaround::map_developer_role_to_system(params.messages);
    }

    if (!tmpl.original_caps().supports_system_role) {
        workaround::system_message_not_supported(params.messages);
    }

    if (tmpl.original_caps().supports_tool_calls) {
        // some templates will require the content field in tool call messages
        // to still be non-null, this puts an empty string everywhere where the
        // content field is null
        workaround::requires_non_null_content(params.messages);
    }

    if (tmpl.original_caps().supports_object_arguments) {
        workaround::func_args_not_string(params.messages);
    }

    params.add_generation_prompt = false;
    std::string no_gen_prompt    = common_chat_template_direct_apply_impl(tmpl, params);
    params.add_generation_prompt = true;
    std::string gen_prompt       = common_chat_template_direct_apply_impl(tmpl, params);
    auto        diff             = calculate_diff_split(no_gen_prompt, gen_prompt);
    params.generation_prompt     = diff.right + diff.suffix;

    params.add_generation_prompt = inputs.add_generation_prompt;

    params.extra_context = common_chat_extra_context();
    for (auto el : inputs.chat_template_kwargs) {
        params.extra_context[el.first] = json::parse(el.second);
    }

    if (!inputs.json_schema.empty()) {
        params.json_schema = json::parse(inputs.json_schema);
    }

    params.parallel_tool_calls = inputs.parallel_tool_calls;

    if (params.tools.is_array()) {
        if (params.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE && !params.grammar.empty()) {
            throw std::runtime_error("Cannot specify grammar with tools");
        }
        if (caps.supports_tool_calls && !caps.supports_tools) {
            LOG_WRN(
                "Template supports tool calls but does not natively describe tools. The fallback behaviour used may "
                "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
        }
    }

    if (inputs.force_pure_content) {
        LOG_WRN("Forcing pure content template, will not render reasoning or tools separately.");
        // Create the result structure
        common_chat_params data;
        auto params_copy               = params;
        params_copy.reasoning_format   = COMMON_REASONING_FORMAT_NONE;
        data.prompt                    = common_chat_template_direct_apply_impl(tmpl, params_copy);
        data.format                    = COMMON_CHAT_FORMAT_PEG_NATIVE;
        data.generation_prompt         = params.generation_prompt;
        auto parser                    = build_chat_peg_parser([&params](common_chat_peg_builder &p) {
            return p.prefix(params.generation_prompt) << p.content(p.rest());
        });
        data.parser                    = parser.save();
        return data;
    }

    if (auto result = common_chat_try_specialized_template(tmpl, src, params)) {
        result->generation_prompt = params.generation_prompt;
        return *result;
    }

    try {
        LOG_DBG("%s: using differential autoparser\n", __func__);
        struct autoparser::autoparser autoparser;
        autoparser.analyze_template(tmpl);
        auto auto_params = autoparser::peg_generator::generate_parser(tmpl, params, autoparser);
        auto_params.supports_thinking = autoparser.reasoning.mode != autoparser::reasoning_mode::NONE;
        if (auto_params.supports_thinking) {
            auto_params.thinking_start_tag = autoparser.reasoning.start;
            auto_params.thinking_end_tag   = autoparser.reasoning.end;
        }
        auto_params.generation_prompt = params.generation_prompt;
        common_peg_arena arena;
        arena.load(auto_params.parser);
        LOG_DBG("%s: generated parser:\n%s\n\nparser generation prompt: %s\n", __func__, arena.dump(arena.root()).c_str(), auto_params.generation_prompt.c_str());
        return auto_params;
    } catch (const std::exception & e) {
        throw std::invalid_argument(std::string("Unable to generate parser for this template. Automatic parser generation failed: ") + e.what());
    }
}

// Legacy template route (adhoc C++ implementation of known templates), forward to llama_chat_apply_template.
static common_chat_params common_chat_templates_apply_legacy(const struct common_chat_templates *        tmpls,
                                                             const struct common_chat_templates_inputs & inputs) {
    size_t                          alloc_size = 0;
    std::vector<llama_chat_message> chat;
    std::vector<std::string>        contents;

    for (const auto & msg : inputs.messages) {
        auto content = msg.content;
        for (const auto & part : msg.content_parts) {
            if (part.type != "text" && part.type != "media_marker") {
                LOG_WRN("Ignoring non-text content part: %s\n", part.type.c_str());
                continue;
            }
            if (!content.empty()) {
                content += "\n";
                ;
            }
            content += part.text;
        }
        contents.emplace_back(std::move(content));
    }
    for (size_t i = 0; i < contents.size(); ++i) {
        const auto & msg     = inputs.messages[i];
        const auto & content = contents[i];
        chat.push_back({ msg.role.c_str(), content.c_str() });
        size_t msg_size = msg.role.size() + content.size();
        alloc_size += msg_size + (msg_size / 4);  // == msg_size * 1.25 but avoiding float ops
    }

    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    const auto & src = tmpls->template_default->source();
    int32_t      res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt,
                                                 buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        // if the custom "tmpl" is not supported, we throw an error
        // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
        throw std::runtime_error("this custom template is not supported, try using --jinja");
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(src.c_str(), chat.data(), chat.size(), inputs.add_generation_prompt, buf.data(),
                                        buf.size());
    }

    // for safety, we check the result again
    if (res < 0 || (size_t) res > buf.size()) {
        throw std::runtime_error("failed to apply chat template, try using --jinja");
    }

    common_chat_params params;
    params.prompt = std::string(buf.data(), res);
    if (!inputs.json_schema.empty()) {
        params.grammar = json_schema_to_grammar(json::parse(inputs.json_schema));
    } else {
        params.grammar = inputs.grammar;
    }
    return params;
}

common_chat_params common_chat_templates_apply(const struct common_chat_templates *        tmpls,
                                               const struct common_chat_templates_inputs & inputs) {
    GGML_ASSERT(tmpls != nullptr);
    return inputs.use_jinja ? common_chat_templates_apply_jinja(tmpls, inputs) :
                              common_chat_templates_apply_legacy(tmpls, inputs);
}

common_chat_msg common_chat_parse(const std::string &               input,
                                  bool                              is_partial,
                                  const common_chat_parser_params & params) {
    return common_chat_peg_parse(params.parser, input, is_partial, params);
}

common_chat_msg common_chat_peg_parse(const common_peg_arena &          src_parser,
                                      const std::string &               input,
                                      bool                              is_partial,
                                      const common_chat_parser_params & params) {
    const common_peg_arena & parser = src_parser.empty() ?
        build_chat_peg_parser([](common_chat_peg_builder & p) { return p.content(p.rest()) + p.end(); }) :
        src_parser;

    if (src_parser.empty()) {
        LOG_DBG("No parser definition detected, assuming pure content parser.");
    }

    const std::string effective_input = params.generation_prompt.empty()
        ? input
        : params.generation_prompt + input;

    //LOG_DBG("Parsing PEG input with format %s: %s\n", common_chat_format_name(params.format), effective_input.c_str());

    common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_LENIENT;
    if (params.debug) {
        flags |= COMMON_PEG_PARSE_FLAG_DEBUG;
    }

    common_peg_parse_context ctx(effective_input, flags);
    auto result = parser.parse(ctx);

    if (result.fail()) {
        // During partial parsing, return partial results if any AST nodes were captured
        // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
        if (is_partial && result.end > 0) {
            // Try to extract any partial results from what was successfully parsed
            common_chat_msg msg;
            msg.role = "assistant";
            std::unique_ptr<common_chat_peg_mapper> mapper;
            if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
                mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
            } else {
                mapper = std::make_unique<common_chat_peg_mapper>(msg);
            }
            mapper->from_ast(ctx.ast, result);

            if (ctx.is_debug()) {
                fprintf(stderr, "\nAST for partial parse (fail):\n%s\n", ctx.ast.dump().c_str());
                fflush(stderr);
            }
            return msg;
        }
        throw std::runtime_error(std::string("Failed to parse input at pos ") + std::to_string(result.end) + ": " +
                                 effective_input.substr(result.end));
    }

    common_chat_msg msg;
    msg.role = "assistant";

    std::unique_ptr<common_chat_peg_mapper> mapper;
    if (params.format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
        mapper = std::make_unique<common_chat_peg_gemma4_mapper>(msg);
    } else {
        mapper = std::make_unique<common_chat_peg_mapper>(msg);
    }
    mapper->from_ast(ctx.ast, result);

    if (ctx.is_debug()) {
        fprintf(stderr, "\nAST for %s parse:\n%s\n", is_partial ? "partial" : "full", ctx.ast.dump().c_str());
        fflush(stderr);
    }

    if (!is_partial) {
        LOG_DBG("Parsed message: %s\n", common_chat_msgs_to_json_oaicompat({ msg }).at(0).dump().c_str());
    }
    return msg;
}

std::map<std::string, bool> common_chat_templates_get_caps(const common_chat_templates * chat_templates) {
    GGML_ASSERT(chat_templates != nullptr);
    GGML_ASSERT(chat_templates->template_default != nullptr);
    return chat_templates->template_default->caps.to_map();
}

#include "common.h.inc"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp.inc>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <unordered_set>
#include <string>
#include <thread>
#include <vector>

#include <cpp-httplib/httplib.h.inc>

extern "C" const char * llama_build_info(void);

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#   if !defined(PATH_MAX)
#   define PATH_MAX MAX_PATH
#   endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

// isatty
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

//
// downloader
//

// validate repo name format: owner/repo
static void write_file(const std::string & fname, const std::string & content) {
    const std::string fname_tmp = fname + ".tmp";
    std::ofstream     file(fname_tmp);
    if (!file) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
    }

    try {
        file << content;
        file.close();

        // Makes write atomic
        if (rename(fname_tmp.c_str(), fname.c_str()) != 0) {
            LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, fname_tmp.c_str(), fname.c_str());
            // If rename fails, try to delete the temporary file
            if (remove(fname_tmp.c_str()) != 0) {
                LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, fname_tmp.c_str());
            }
        }
    } catch (...) {
        // If anything fails, try to delete the temporary file
        if (remove(fname_tmp.c_str()) != 0) {
            LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, fname_tmp.c_str());
        }

        throw std::runtime_error(string_format("error: failed to write file '%s'\n", fname.c_str()));
    }
}

static void write_etag(const std::string & path, const std::string & etag) {
    const std::string etag_path = path + ".etag";
    write_file(etag_path, etag);
    LOG_DBG("%s: file etag saved: %s\n", __func__, etag_path.c_str());
}

static std::string read_etag(const std::string & path) {
    return common_rust_string_to_std(
            llama_common_read_etag_rust((const uint8_t *) path.data(), path.size()));
}

static bool is_http_status_ok(int status) {
    return llama_common_is_http_status_ok_rust(status);
}

static std::pair<httplib::Client, common_http_url> common_http_client(const std::string & url) {
    common_http_url parts = common_http_parse_url(url);

    if (parts.host.empty()) {
        throw std::runtime_error("error: invalid URL format");
    }

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    if (parts.scheme == "https") {
        throw std::runtime_error(
            "HTTPS is not supported. Please rebuild with one of:\n"
            "  -DLLAMA_BUILD_BORINGSSL=ON\n"
            "  -DLLAMA_BUILD_LIBRESSL=ON\n"
            "  -DLLAMA_OPENSSL=ON (default, requires OpenSSL dev files installed)"
        );
    }
#endif

    httplib::Client cli(parts.scheme + "://" + parts.host + ":" + std::to_string(parts.port));

    if (!parts.user.empty()) {
        cli.set_basic_auth(parts.user, parts.password);
    }

    cli.set_follow_location(true);

    return { std::move(cli), std::move(parts) };
}

std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag) {
    auto parts = string_split<std::string>(hf_repo_with_tag, ':');
    std::string tag = parts.size() > 1 ? parts.back() : "";
    std::string hf_repo = parts[0];
    if (string_split<std::string>(hf_repo, '/').size() != 2) {
        throw std::invalid_argument("error: invalid HF repo format, expected <user>/<model>[:quant]\n");
    }
    return {hf_repo, tag};
}

class ProgressBar : public common_download_callback {
    static inline std::mutex mutex;
    static inline std::map<const ProgressBar *, int> lines;
    static inline int max_line = 0;

    std::string filename;
    size_t len = 0;

    static void cleanup(const ProgressBar * line) {
        lines.erase(line);
        if (lines.empty()) {
            max_line = 0;
        }
    }

    static bool is_output_a_tty() {
#if defined(_WIN32)
        return _isatty(_fileno(stdout));
#else
        return isatty(1);
#endif
    }

public:
    ProgressBar() = default;

    void on_start(const common_download_progress & p) override {
        filename = p.url;

        if (auto pos = filename.rfind('/'); pos != std::string::npos) {
            filename = filename.substr(pos + 1);
        }
        if (auto pos = filename.find('?'); pos != std::string::npos) {
            filename = filename.substr(0, pos);
        }
        for (size_t i = 0; i < filename.size(); ++i) {
            if ((filename[i] & 0xC0) != 0x80) {
                if (len++ == 39) {
                    filename.resize(i);
                    filename += "…";
                    break;
                }
            }
        }
    }

    void on_done(const common_download_progress &, bool) override {
        std::lock_guard<std::mutex> lock(mutex);
        cleanup(this);
    }

    void on_update(const common_download_progress & p) override {
        if (!p.total || !is_output_a_tty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);

        if (lines.find(this) == lines.end()) {
            lines[this] = max_line++;
            std::cout << "\n";
        }
        int lines_up = max_line - lines[this];

        size_t bar = (55 - len) * 2;
        size_t pct = (100 * p.downloaded) / p.total;
        size_t pos = (bar * p.downloaded) / p.total;

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "A";
        }
        std::cout << '\r' << "Downloading " << filename << " ";

        for (size_t i = 0; i < bar; i += 2) {
            std::cout << (i + 1 < pos ? "─" : (i < pos ? "╴" : " "));
        }
        std::cout << std::setw(4) << pct << "%\033[K";

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "B";
        }
        std::cout << '\r' << std::flush;

        if (p.downloaded == p.total) {
            cleanup(this);
        }
    }

    ProgressBar(const ProgressBar &) = delete;
    ProgressBar & operator=(const ProgressBar &) = delete;
};

static bool common_pull_file(httplib::Client & cli,
                             const std::string & resolve_path,
                             const std::string & path_tmp,
                             bool supports_ranges,
                             common_download_progress & p,
                             common_download_callback * callback) {
    std::ofstream ofs(path_tmp, std::ios::binary | std::ios::app);
    if (!ofs.is_open()) {
        LOG_ERR("%s: error opening local file for writing: %s\n", __func__, path_tmp.c_str());
        return false;
    }

    httplib::Headers headers;
    if (supports_ranges && p.downloaded > 0) {
        headers.emplace("Range", "bytes=" + std::to_string(p.downloaded) + "-");
    }

    const char * func = __func__; // avoid __func__ inside a lambda
    size_t progress_step = 0;

    auto res = cli.Get(resolve_path, headers,
        [&](const httplib::Response &response) {
            if (p.downloaded > 0 && response.status != 206) {
                LOG_WRN("%s: server did not respond with 206 Partial Content for a resume request. Status: %d\n", func, response.status);
                return false;
            }
            if (p.downloaded == 0 && response.status != 200) {
                LOG_WRN("%s: download received non-successful status code: %d\n", func, response.status);
                return false;
            }
            if (p.total == 0 && response.has_header("Content-Length")) {
                try {
                    size_t content_length = std::stoull(response.get_header_value("Content-Length"));
                    p.total = p.downloaded + content_length;
                } catch (const std::exception &e) {
                    LOG_WRN("%s: invalid Content-Length header: %s\n", func, e.what());
                }
            }
            return true;
        },
        [&](const char *data, size_t len) {
            ofs.write(data, len);
            if (!ofs) {
                LOG_ERR("%s: error writing to file: %s\n", func, path_tmp.c_str());
                return false;
            }
            p.downloaded += len;
            progress_step += len;

            if (progress_step >= p.total / 1000 || p.downloaded == p.total) {
                if (callback) {
                    callback->on_update(p);
                    if (callback->is_cancelled()) {
                        return false;
                    }
                }
                progress_step = 0;
            }
            return true;
        },
        nullptr
    );

    if (!res) {
        LOG_ERR("%s: download failed: %s (status: %d)\n",
                __func__,
                httplib::to_string(res.error()).c_str(),
                res ? res->status : -1);
        return false;
    }

    return true;
}

// download one single file from remote URL to local path
// returns status code or -1 on error
static int common_download_file_single_online(const std::string & url,
                                              const std::string & path,
                                              const common_download_opts & opts,
                                              bool skip_etag) {
    static const int max_attempts        = 3;
    static const int retry_delay_seconds = 2;

    const bool file_exists = std::filesystem::exists(path);

    if (file_exists && skip_etag) {
        LOG_DBG("%s: using cached file: %s\n", __func__, path.c_str());
        return 304; // 304 Not Modified - fake cached response
    }

    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : opts.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }
    if (!opts.bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + opts.bearer_token);
    }
    cli.set_default_headers(headers);

    std::string last_etag;
    if (file_exists) {
        last_etag = read_etag(path);
    } else {
        LOG_DBG("%s: no previous model file found %s\n", __func__, path.c_str());
    }

    auto head = cli.Head(parts.path);
    if (!head || head->status < 200 || head->status >= 300) {
        LOG_WRN("%s: HEAD failed, status: %d\n", __func__, head ? head->status : -1);
        if (file_exists) {
            LOG_INF("%s: using cached file (HEAD failed): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        return head ? head->status : -1;
    }

    std::string etag;
    if (head->has_header("ETag")) {
        etag = head->get_header_value("ETag");
    }

    common_download_progress p;
    p.url = url;
    if (head->has_header("Content-Length")) {
        try {
            p.total = std::stoull(head->get_header_value("Content-Length"));
        } catch (const std::exception& e) {
            LOG_WRN("%s: invalid Content-Length in HEAD response: %s\n", __func__, e.what());
        }
    }

    bool supports_ranges = false;
    if (head->has_header("Accept-Ranges")) {
        supports_ranges = head->get_header_value("Accept-Ranges") != "none";
    }

    if (file_exists) {
        if (etag.empty()) {
            LOG_DBG("%s: using cached file (no server etag): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        if (!last_etag.empty() && last_etag == etag) {
            LOG_DBG("%s: using cached file (same etag): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        if (remove(path.c_str()) != 0) {
            LOG_ERR("%s: unable to delete file: %s\n", __func__, path.c_str());
            return -1;
        }
    }

    { // silent
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    }

    bool success = false;
    const std::string path_temporary = path + ".downloadInProgress";
    int delay = retry_delay_seconds;

    if (opts.callback) {
        opts.callback->on_start(p);
    }

    for (int i = 0; i < max_attempts; ++i) {
        if (opts.callback && opts.callback->is_cancelled()) {
            break;
        }
        if (i) {
            LOG_WRN("%s: retrying after %d seconds...\n", __func__, delay);
            std::this_thread::sleep_for(std::chrono::seconds(delay));
            delay *= retry_delay_seconds;
        }

        size_t existing_size = 0;

        if (std::filesystem::exists(path_temporary)) {
            if (supports_ranges) {
                existing_size = std::filesystem::file_size(path_temporary);
            } else if (remove(path_temporary.c_str()) != 0) {
                LOG_ERR("%s: unable to delete file: %s\n", __func__, path_temporary.c_str());
                break;
            }
        }

        p.downloaded = existing_size;

        LOG_DBG("%s: downloading from %s to %s (etag:%s)...\n",
                __func__, common_http_show_masked_url(parts).c_str(),
                path_temporary.c_str(), etag.c_str());

        if (common_pull_file(cli, parts.path, path_temporary, supports_ranges, p, opts.callback)) {
            if (std::rename(path_temporary.c_str(), path.c_str()) != 0) {
                LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
                break;
            }
            if (!etag.empty() && !skip_etag) {
                write_etag(path, etag);
            }
            success = true;
            break;
        }
    }

    if (opts.callback) {
        opts.callback->on_done(p, success);
    }
    if (opts.callback && opts.callback->is_cancelled() &&
        std::filesystem::exists(path_temporary)) {
        if (remove(path_temporary.c_str()) != 0) {
            LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, path_temporary.c_str());
        }
    }
    if (!success) {
        LOG_ERR("%s: download failed after %d attempts\n", __func__, max_attempts);
        return -1; // max attempts reached
    }

    return head->status;
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string          & url,
                                                             const common_remote_params & params) {
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : params.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }

    if (params.timeout > 0) {
        cli.set_read_timeout(params.timeout, 0);
        cli.set_write_timeout(params.timeout, 0);
    }

    std::vector<char> buf;
    auto res = cli.Get(parts.path, headers,
        [&](const char *data, size_t len) {
            buf.insert(buf.end(), data, data + len);
            return params.max_size == 0 ||
                   buf.size() <= static_cast<size_t>(params.max_size);
        },
        nullptr
    );

    if (!res) {
        throw std::runtime_error("error: cannot make GET request");
    }

    return { res->status, std::move(buf) };
}

int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const common_download_opts & opts,
                                bool skip_etag) {
    if (!opts.offline) {
        ProgressBar tty_cb;
        common_download_opts online_opts = opts;
        if (!online_opts.callback) {
            online_opts.callback = &tty_cb;
        }
        return common_download_file_single_online(url, path, online_opts, skip_etag);
    }

    if (!std::filesystem::exists(path)) {
        LOG_ERR("%s: required file is not available in cache (offline mode): %s\n", __func__, path.c_str());
        return -1;
    }

    LOG_DBG("%s: using cached file (offline mode): %s\n", __func__, path.c_str());

    // notify the callback that the file was cached
    if (opts.callback) {
        common_download_progress p;
        p.url = url;
        p.cached = true;
        opts.callback->on_start(p);
        opts.callback->on_done(p, true);
    }

    return 304; // Not Modified - fake cached response
}

struct gguf_split_info {
    std::string prefix; // tag included
    std::string tag;
    int index;
    int count;
};

static gguf_split_info get_gguf_split_info(const std::string & path) {
    common_rust_gguf_split_info split =
        llama_common_gguf_split_info_rust((const uint8_t *) path.data(), path.size(), true);
    gguf_split_info result {
        common_rust_string_to_std(split.prefix),
        common_rust_string_to_std(split.tag),
        split.index,
        split.count,
    };
    return result;
}

static hf_cache::hf_files get_split_files(const hf_cache::hf_files & files,
                                          const hf_cache::hf_file  & file) {
    auto split = get_gguf_split_info(file.path);

    if (split.count <= 1) {
        return {file};
    }
    hf_cache::hf_files result;

    for (const auto & f : files) {
        auto split_f = get_gguf_split_info(f.path);
        if (split_f.count == split.count && split_f.prefix == split.prefix) {
            result.push_back(f);
        }
    }
    return result;
}

static hf_cache::hf_file find_best_mmproj(const hf_cache::hf_files & files,
                                          const std::string        & model) {
    hf_cache::hf_file best;
    size_t best_depth = 0;
    int best_diff = 0;
    bool found = false;

    auto model_bits = llama_common_gguf_extract_quant_bits_rust((const uint8_t *) model.data(), model.size());
    auto model_parts = string_split<std::string>(model, '/');
    auto model_dir = model_parts.end() - 1;

    for (const auto & f : files) {
        if (!string_ends_with(f.path, ".gguf") ||
            f.path.find("mmproj") == std::string::npos) {
            continue;
        }

        auto mmproj_parts = string_split<std::string>(f.path, '/');
        auto mmproj_dir = mmproj_parts.end() - 1;

        auto [_, dir] = std::mismatch(model_parts.begin(), model_dir,
                                      mmproj_parts.begin(), mmproj_dir);
        if (dir != mmproj_dir) {
            continue;
        }

        size_t depth = dir - mmproj_parts.begin();
        auto bits = llama_common_gguf_extract_quant_bits_rust((const uint8_t *) f.path.data(), f.path.size());
        auto diff = std::abs(bits - model_bits);

        if (!found || depth > best_depth || (depth == best_depth && diff < best_diff)) {
            best = f;
            best_depth = depth;
            best_diff = diff;
            found = true;
        }
    }
    return best;
}

static hf_cache::hf_file find_best_model(const hf_cache::hf_files & files,
                                         const std::string        & tag) {
    std::vector<std::string> tags;

    if (!tag.empty()) {
        tags.push_back(tag);
    } else {
        tags = {"Q4_K_M", "Q8_0"};
    }

    for (const auto & t : tags) {
        std::regex pattern(t + "[.-]", std::regex::icase);
        for (const auto & f : files) {
            if (llama_common_gguf_filename_is_model_rust((const uint8_t *) f.path.data(), f.path.size()) &&
                std::regex_search(f.path, pattern)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    // fallback to first available model only if tag is empty
    if (tag.empty()) {
        for (const auto & f : files) {
            if (llama_common_gguf_filename_is_model_rust((const uint8_t *) f.path.data(), f.path.size())) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    return {};
}

static void list_available_gguf_files(const hf_cache::hf_files & files) {
    LOG_INF("Available GGUF files:\n");
    for (const auto & f : files) {
        if (string_ends_with(f.path, ".gguf")) {
            LOG_INF(" - %s\n", f.path.c_str());
        }
    }
}

struct hf_plan {
    hf_cache::hf_file primary;
    hf_cache::hf_files model_files;
    hf_cache::hf_file mmproj;
};

static hf_plan get_hf_plan(const common_params_model  & model,
                           const common_download_opts & opts,
                           bool download_mmproj) {
    hf_plan plan;
    hf_cache::hf_files all;

    auto [repo, tag] = common_download_split_repo_tag(model.hf_repo);

    if (!opts.offline) {
        all = hf_cache::get_repo_files(repo, opts.bearer_token);
    }
    if (all.empty()) {
        all = hf_cache::get_cached_files(repo);
    }
    if (all.empty()) {
        return plan;
    }

    hf_cache::hf_file primary;

    if (!model.hf_file.empty()) {
        for (const auto & f : all) {
            if (f.path == model.hf_file) {
                primary = f;
                break;
            }
        }
        if (primary.path.empty()) {
            LOG_ERR("%s: file '%s' not found in repository\n", __func__, model.hf_file.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    } else {
        primary = find_best_model(all, tag);
        if (primary.path.empty()) {
            LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    }

    plan.primary = primary;
    plan.model_files = get_split_files(all, primary);

    if (download_mmproj) {
        plan.mmproj = find_best_mmproj(all, primary.path);
    }

    return plan;
}

struct download_task {
    std::string url;
    std::string path;
};

static std::vector<download_task> get_url_tasks(const common_params_model & model) {
    auto split = get_gguf_split_info(model.url);

    if (split.count <= 1) {
        return {{model.url, model.path}};
    }

    auto filename = split.prefix;
    if (auto pos = split.prefix.rfind('/'); pos != std::string::npos) {
        filename = split.prefix.substr(pos + 1);
    }

    auto parent_path = std::filesystem::path(model.path).parent_path();
    auto prefix_path = (parent_path / filename).string();

    std::vector<download_task> tasks;
    for (int i = 1; i <= split.count; i++) {
        auto suffix = string_format("-%05d-of-%05d.gguf", i, split.count);
        tasks.push_back({split.prefix + suffix, prefix_path + suffix});
    }
    return tasks;
}

common_download_model_result common_download_model(const common_params_model  & model,
                                                   const common_download_opts & opts,
                                                   bool download_mmproj) {
    common_download_model_result result;
    std::vector<download_task> tasks;
    hf_plan hf;

    bool is_hf = !model.hf_repo.empty();

    if (is_hf) {
        hf = get_hf_plan(model, opts, download_mmproj);
        for (const auto & f : hf.model_files) {
            tasks.push_back({f.url, f.local_path});
        }
        if (!hf.mmproj.path.empty()) {
            tasks.push_back({hf.mmproj.url, hf.mmproj.local_path});
        }
    } else if (!model.url.empty()) {
        tasks = get_url_tasks(model);
    } else {
        result.model_path = model.path;
        return result;
    }

    if (tasks.empty()) {
        return result;
    }

    std::vector<std::future<bool>> futures;
    for (const auto & task : tasks) {
        futures.push_back(std::async(std::launch::async,
            [&task, &opts, is_hf]() {
                int status = common_download_file_single(task.url, task.path, opts, is_hf);
                return is_http_status_ok(status);
            }
        ));
    }

    for (auto & f : futures) {
        if (!f.get()) {
            return {};
        }
    }

    if (is_hf) {
        for (const auto & f : hf.model_files) {
            hf_cache::finalize_file(f);
        }
        result.model_path = hf.primary.final_path;

        if (!hf.mmproj.path.empty()) {
            result.mmproj_path = hf_cache::finalize_file(hf.mmproj);
        }
    } else {
        result.model_path = model.path;
    }

    return result;
}

//
// Docker registry functions
//

static std::string common_docker_get_token(const std::string & repo) {
    std::string url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + repo + ":pull";

    common_remote_params params;
    auto                 res = common_remote_get_content(url, params);

    if (res.first != 200) {
        throw std::runtime_error("Failed to get Docker registry token, HTTP code: " + std::to_string(res.first));
    }

    std::string            response_str(res.second.begin(), res.second.end());
    nlohmann::ordered_json response = nlohmann::ordered_json::parse(response_str);

    if (!response.contains("token")) {
        throw std::runtime_error("Docker registry token response missing 'token' field");
    }

    return response["token"].get<std::string>();
}

std::string common_docker_resolve_model(const std::string & docker) {
    // Parse ai/smollm2:135M-Q4_0
    size_t      colon_pos = docker.find(':');
    std::string repo, tag;
    if (colon_pos != std::string::npos) {
        repo = docker.substr(0, colon_pos);
        tag  = docker.substr(colon_pos + 1);
    } else {
        repo = docker;
        tag  = "latest";
    }

    // ai/ is the default
    size_t      slash_pos = docker.find('/');
    if (slash_pos == std::string::npos) {
        repo.insert(0, "ai/");
    }

    LOG_INF("%s: Downloading Docker Model: %s:%s\n", __func__, repo.c_str(), tag.c_str());
    try {
        // --- helper: digest validation ---
        auto validate_oci_digest = [](const std::string & digest) -> std::string {
            // Expected: algo:hex ; start with sha256 (64 hex chars)
            // You can extend this map if supporting other algorithms in future.
            static const std::regex re("^sha256:([a-fA-F0-9]{64})$");
            std::smatch m;
            if (!std::regex_match(digest, m, re)) {
                throw std::runtime_error("Invalid OCI digest format received in manifest: " + digest);
            }
            // normalize hex to lowercase
            std::string normalized = digest;
            std::transform(normalized.begin()+7, normalized.end(), normalized.begin()+7, [](unsigned char c){
                return std::tolower(c);
            });
            return normalized;
        };

        std::string token = common_docker_get_token(repo);  // Get authentication token

        // Get manifest
        // TODO: cache the manifest response so that it appears in the model list
        const std::string    url_prefix = "https://registry-1.docker.io/v2/" + repo;
        std::string          manifest_url = url_prefix + "/manifests/" + tag;
        common_remote_params manifest_params;
        manifest_params.headers.push_back({"Authorization", "Bearer " + token});
        manifest_params.headers.push_back({"Accept",
            "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"
        });
        auto manifest_res = common_remote_get_content(manifest_url, manifest_params);
        if (manifest_res.first != 200) {
            throw std::runtime_error("Failed to get Docker manifest, HTTP code: " + std::to_string(manifest_res.first));
        }

        std::string            manifest_str(manifest_res.second.begin(), manifest_res.second.end());
        nlohmann::ordered_json manifest = nlohmann::ordered_json::parse(manifest_str);
        std::string            gguf_digest;  // Find the GGUF layer
        if (manifest.contains("layers")) {
            for (const auto & layer : manifest["layers"]) {
                if (layer.contains("mediaType")) {
                    std::string media_type = layer["mediaType"].get<std::string>();
                    if (media_type == "application/vnd.docker.ai.gguf.v3" ||
                        media_type.find("gguf") != std::string::npos) {
                        gguf_digest = layer["digest"].get<std::string>();
                        break;
                    }
                }
            }
        }

        if (gguf_digest.empty()) {
            throw std::runtime_error("No GGUF layer found in Docker manifest");
        }

        // Validate & normalize digest
        gguf_digest = validate_oci_digest(gguf_digest);
        LOG_DBG("%s: Using validated digest: %s\n", __func__, gguf_digest.c_str());

        // Prepare local filename
        std::string model_filename = repo;
        std::replace(model_filename.begin(), model_filename.end(), '/', '_');
        model_filename += "_" + tag + ".gguf";
        std::string local_path = fs_get_cache_file(model_filename);

        const std::string blob_url = url_prefix + "/blobs/" + gguf_digest;
        common_download_opts opts;
        opts.bearer_token = token;
        const int http_status = common_download_file_single(blob_url, local_path, opts);
        if (!is_http_status_ok(http_status)) {
            throw std::runtime_error("Failed to download Docker Model");
        }

        LOG_INF("%s: Downloaded Docker Model to: %s\n", __func__, local_path.c_str());
        return local_path;
    } catch (const std::exception & e) {
        LOG_ERR("%s: Docker Model download failed: %s\n", __func__, e.what());
        throw;
    }
}

std::vector<common_cached_model_info> common_list_cached_models() {
    std::unordered_set<std::string> seen;
    std::vector<common_cached_model_info> result;

    auto files = hf_cache::get_cached_files();

    for (const auto & f : files) {
        auto split = get_gguf_split_info(f.path);
        if (split.index != 1 || split.tag.empty() ||
            split.prefix.find("mmproj") != std::string::npos) {
            continue;
        }
        if (seen.insert(f.repo_id + ":" + split.tag).second) {
            result.push_back({f.repo_id, split.tag});
        }
    }

    return result;
}


#include "common.h.inc"

#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <nlohmann/json.hpp.inc>
#include <regex>
#include <stdexcept>
#include <unordered_set>

// Trick to catch missing branches
template <typename T>
inline constexpr bool is_always_false_v = false;

const char * common_peg_parse_result_type_name(common_peg_parse_result_type type) {
    return llama_common_peg_parse_result_type_name_rust((int) type);
}

// Trie for matching multiple literals.
// This is used in common_peg_until_parser and to build a GBNF exclusion grammar
struct trie {
    struct node {
        std::map<uint32_t, size_t> children;  // Use uint32_t to store Unicode codepoints
        bool is_word;
    };

    std::vector<node> nodes;

    trie(const std::vector<std::string> & words) {
      create_node(); // root node
      for (const auto & w : words) {
          insert(w);
      }
    }

    enum match_result { NO_MATCH, PARTIAL_MATCH, COMPLETE_MATCH };

    // Check if a delimiter starts at the given position
    match_result check_at(std::string_view sv, size_t start_pos) const {
        size_t current = 0; // Start at root
        size_t pos = start_pos;

        // LOG_DBG("%s: checking at pos %zu, sv='%s'\n", __func__, start_pos, std::string(sv).c_str());

        while (pos < sv.size()) {
            auto result = common_parse_utf8_codepoint(sv, pos);
            if (result.status != utf8_parse_result::SUCCESS) {
                break;
            }

            auto it = nodes[current].children.find(result.codepoint);
            if (it == nodes[current].children.end()) {
                // Can't continue matching
                return match_result{match_result::NO_MATCH};
            }

            current = it->second;
            pos += result.bytes_consumed;

            // Check if we've matched a complete word
            if (nodes[current].is_word) {
                return match_result{match_result::COMPLETE_MATCH};
            }
        }

        // Reached end of input while still in the trie (not at root)
        if (current != 0) {
            // We're in the middle of a potential match
            return match_result{match_result::PARTIAL_MATCH};
        }

        // Reached end at root (no match)
        return match_result{match_result::NO_MATCH};
    }

    struct prefix_and_next {
        std::vector<uint32_t> prefix;
        std::vector<uint32_t> next_chars;
    };

    std::vector<prefix_and_next> collect_prefix_and_next() {
        std::vector<uint32_t>        prefix;
        std::vector<prefix_and_next> result;
        collect_prefix_and_next(0, prefix, result);
        return result;
    }

  private:
    void collect_prefix_and_next(size_t index, std::vector<uint32_t> & prefix, std::vector<prefix_and_next> & out) {
        if (!nodes[index].is_word) {
            if (!nodes[index].children.empty()) {
                std::vector<uint32_t> chars;
                chars.reserve(nodes[index].children.size());
                for (const auto & p : nodes[index].children) {
                    chars.push_back(p.first);
                }
                out.emplace_back(prefix_and_next{prefix, chars});
            }
        }

        for (const auto & p : nodes[index].children) {
            uint32_t ch = p.first;
            auto child = p.second;
            prefix.push_back(ch);
            collect_prefix_and_next(child, prefix, out);
            prefix.pop_back();
        }
    }

    size_t create_node() {
        size_t index = nodes.size();
        nodes.emplace_back();
        return index;
    }

    void insert(const std::string & word) {
        size_t current = 0;
        size_t pos     = 0;
        while (pos < word.length()) {
            auto result = common_parse_utf8_codepoint(word, pos);
            if (result.status != utf8_parse_result::SUCCESS) {
                break;
            }

            uint32_t ch = result.codepoint;
            pos += result.bytes_consumed;

            auto it = nodes[current].children.find(ch);
            if (it == nodes[current].children.end()) {
                size_t child = create_node();
                nodes[current].children[ch] = child;
                current = child;
            } else {
                current = it->second;
            }
        }
        nodes[current].is_word = true;
    }
};

static std::pair<uint32_t, size_t> parse_hex_escape(const std::string & str, size_t pos, int hex_count) {
    uint32_t value = 0;
    const size_t consumed = llama_common_parse_hex_escape_rust(
            (const uint8_t *) str.data(), str.size(), pos, hex_count, &value);
    if (consumed == 0) {
        return {0, 0};
    }
    return {value, consumed};
}

static std::pair<uint32_t, size_t> parse_char_class_char(const std::string & content, size_t pos) {
    uint32_t value = 0;
    const size_t consumed = llama_common_parse_char_class_char_rust(
            (const uint8_t *) content.data(), content.size(), pos, &value);
    return {value, consumed};
}

static std::pair<std::vector<common_peg_chars_parser::char_range>, bool> parse_char_classes(const std::string & classes) {
    std::vector<common_peg_chars_parser::char_range> ranges;
    bool negated = false;

    std::string content = classes;
    if (content.front() == '[') {
        content = content.substr(1);
    }

    if (content.back() == ']') {
        content.pop_back();
    }

    // Check for negation
    if (!content.empty() && content.front() == '^') {
        negated = true;
        content = content.substr(1);
    }

    size_t i = 0;
    while (i < content.length()) {
        auto [start, start_len] = parse_char_class_char(content, i);
        i += start_len;

        if (i + 1 < content.length() && content[i] == '-') {
            // Range detected
            auto [end, end_len] = parse_char_class_char(content, i + 1);
            ranges.push_back(common_peg_chars_parser::char_range{start, end});
            i += 1 + end_len;
        } else {
            ranges.push_back(common_peg_chars_parser::char_range{start, start});
        }
    }

    return {ranges, negated};
}

common_peg_ast_id common_peg_ast_arena::find_by_tag(const common_peg_ast_node & parent, const std::string & tag, int max_depth) const {
    for (auto child_id : parent.children) {
        const auto & child = get(child_id);
        if (child.tag == tag) {
            return child_id;
        }
        if (max_depth > 1) {
            auto result = find_by_tag(child, tag, max_depth - 1);
            if (result != COMMON_PEG_INVALID_AST_ID) {
                return result;
            }
        }
    }
    return COMMON_PEG_INVALID_AST_ID;
}

common_peg_ast_id common_peg_ast_arena::find_by_rule(const common_peg_ast_node & parent, const std::string & rule, int max_depth) const {
    for (auto child_id : parent.children) {
        const auto & child = get(child_id);
        if (child.rule == rule) {
            return child_id;
        }
        if (max_depth > 1) {
            auto result = find_by_rule(child, rule, max_depth - 1);
            if (result != COMMON_PEG_INVALID_AST_ID) {
                return result;
            }
        }
    }
    return COMMON_PEG_INVALID_AST_ID;
}

void common_peg_ast_arena::visit(common_peg_ast_id id, const common_peg_ast_visitor & visitor) const {
    if (id == COMMON_PEG_INVALID_AST_ID) {
        return;
    }
    const auto & node = get(id);
    visitor(node);
    for (const auto & child : node.children) {
        visit(child, visitor);
    }
}

void common_peg_ast_arena::visit(const common_peg_parse_result & result, const common_peg_ast_visitor & visitor) const {
    for (const auto & node : result.nodes) {
        visit(node, visitor);
    }
}

struct parser_executor;

common_peg_parser_id common_peg_arena::add_parser(common_peg_parser_variant parser) {
    common_peg_parser_id id = parsers_.size();
    parsers_.push_back(std::move(parser));
    return id;
}

void common_peg_arena::add_rule(const std::string & name, common_peg_parser_id id) {
    rules_[name] = id;
}

common_peg_parser_id common_peg_arena::get_rule(const std::string & name) const {
    auto it = rules_.find(name);
    if (it == rules_.end()) {
        throw std::runtime_error("Rule not found: " + name);
    }
    return it->second;
}

struct parser_executor {
    const common_peg_arena & arena;
    common_peg_parse_context & ctx;
    size_t start_pos;

    parser_executor(const common_peg_arena & arena, common_peg_parse_context & ctx, size_t start)
        : arena(arena), ctx(ctx), start_pos(start) {}

    std::string debug_indent() const { return std::string(ctx.parse_depth * 2, ' '); }

    std::string debug_input_snippet(size_t pos, size_t len = 60) const {
        if (pos >= ctx.input.size()) {
            return "<EOF>";
        }
        auto        snippet = ctx.input.substr(pos, len);
        // Escape newlines for display
        std::string result;
        for (char c : snippet) {
            if (c == '\n') {
                result += "\\n";
            } else if (c == '\r') {
                result += "\\r";
            } else if (c == '\t') {
                result += "\\t";
            } else {
                result += c;
            }
        }
        if (pos + len < ctx.input.size()) {
            result += "...";
        }
        return result;
    }

    common_peg_parse_result operator()(const common_peg_epsilon_parser & /* p */) const {
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_start_parser & /* p */) const {
        return common_peg_parse_result(
            start_pos == 0 ? COMMON_PEG_PARSE_RESULT_SUCCESS : COMMON_PEG_PARSE_RESULT_FAIL,
            start_pos
        );
    }

    common_peg_parse_result operator()(const common_peg_end_parser & /* p */) const {
        return common_peg_parse_result(
            start_pos >= ctx.input.size() ? COMMON_PEG_PARSE_RESULT_SUCCESS : COMMON_PEG_PARSE_RESULT_FAIL,
            start_pos
        );
    }

    common_peg_parse_result operator()(const common_peg_literal_parser & p) {
        auto pos = start_pos;
        for (auto i = 0u; i < p.literal.size(); ++i) {
            if (pos >= ctx.input.size()) {
                if (!ctx.is_lenient()) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
            }
            if (ctx.input[pos] != p.literal[i]) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }
            ++pos;
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
    }

    common_peg_parse_result operator()(const common_peg_sequence_parser & p) {
        if (ctx.is_debug()) {
            LOG_DBG("%sSEQ start at %zu '%s' (%zu children)\n", debug_indent().c_str(), start_pos,
                    debug_input_snippet(start_pos).c_str(), p.children.size());
        }
        ctx.parse_depth++;

        auto pos = start_pos;
        std::vector<common_peg_ast_id> nodes;

        for (size_t i = 0; i < p.children.size(); i++) {
            const auto & child_id = p.children[i];
            if (ctx.is_debug()) {
                fprintf(stderr, "%sSEQ child %zu: %s\n", debug_indent().c_str(), i, arena.dump(child_id).c_str());
            }
            auto result = arena.parse(child_id, ctx, pos);

            if (ctx.is_debug()) {
                fprintf(stderr, "%sSEQ child %zu: %s at %zu->%zu\n", debug_indent().c_str(), i,
                        common_peg_parse_result_type_name(result.type), result.start, result.end);
            }

            if (result.fail()) {
                ctx.parse_depth--;
                if (ctx.is_debug()) {
                    fprintf(stderr, "%sSEQ -> FAIL\n", debug_indent().c_str());
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, result.end);
            }

            if (!result.nodes.empty()) {
                nodes.insert(nodes.end(), result.nodes.begin(), result.nodes.end());
            }

            if (result.need_more_input()) {
                ctx.parse_depth--;
                if (ctx.is_debug()) {
                    fprintf(stderr, "%sSEQ -> NEED_MORE\n", debug_indent().c_str());
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, result.end, std::move(nodes));
            }

            pos = result.end;
        }

        ctx.parse_depth--;
        if (ctx.is_debug()) {
            fprintf(stderr, "%sSEQ -> SUCCESS at %zu->%zu\n", debug_indent().c_str(), start_pos, pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos, std::move(nodes));
    }

    common_peg_parse_result operator()(const common_peg_choice_parser & p) {
        if (ctx.is_debug()) {
            fprintf(stderr, "%sCHOICE start at %zu '%s' (%zu options)\n", debug_indent().c_str(), start_pos,
                    debug_input_snippet(start_pos).c_str(), p.children.size());
        }
        ctx.parse_depth++;

        auto pos = start_pos;
        for (size_t i = 0; i < p.children.size(); i++) {
            const auto & child_id = p.children[i];
            if (ctx.is_debug()) {
                fprintf(stderr, "%sCHOICE option %zu: %s\n", debug_indent().c_str(), i, arena.dump(child_id).c_str());
            }
            auto result = arena.parse(child_id, ctx, pos);
            if (ctx.is_debug()) {
                fprintf(stderr, "%sCHOICE option %zu: %s\n", debug_indent().c_str(), i,
                        common_peg_parse_result_type_name(result.type));
            }
            if (!result.fail()) {
                ctx.parse_depth--;
                if (ctx.is_debug()) {
                    fprintf(stderr, "%sCHOICE -> %s (option %zu)\n", debug_indent().c_str(),
                            common_peg_parse_result_type_name(result.type), i);
                }
                return result;
            }
        }

        ctx.parse_depth--;
        if (ctx.is_debug()) {
            fprintf(stderr, "%sCHOICE -> FAIL (no options matched)\n", debug_indent().c_str());
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_repetition_parser & p) {
        if (ctx.is_debug()) {
            fprintf(stderr, "%sREPEAT start at %zu '%s' (min=%d, max=%d)\n", debug_indent().c_str(), start_pos,
                    debug_input_snippet(start_pos).c_str(), p.min_count, p.max_count);
        }
        ctx.parse_depth++;

        auto pos = start_pos;
        int match_count = 0;
        std::vector<common_peg_ast_id> nodes;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (p.max_count == -1 || match_count < p.max_count) {
            if (pos >= ctx.input.size()) {
                if (ctx.is_debug()) {
                    fprintf(stderr, "%sREPEAT: at end of input, count=%d\n", debug_indent().c_str(), match_count);
                }
                break;
            }

            auto result = arena.parse(p.child, ctx, pos);

            if (ctx.is_debug()) {
                fprintf(stderr, "%sREPEAT iter %d: %s at %zu->%zu, nodes=%zu\n", debug_indent().c_str(), match_count,
                        common_peg_parse_result_type_name(result.type), result.start, result.end, result.nodes.size());
                fprintf(stderr, "%sREPEAT CHILD: %s\n", debug_indent().c_str(), arena.dump(p.child).c_str());
            }

            if (result.success()) {
                // Prevent infinite loop on empty matches
                if (result.end == pos) {
                    if (ctx.is_debug()) {
                        fprintf(stderr, "%s  REPEAT: empty match, stopping\n", debug_indent().c_str());
                    }
                    break;
                }

                if (!result.nodes.empty()) {
                    nodes.insert(nodes.end(), result.nodes.begin(), result.nodes.end());
                }

                pos = result.end;
                match_count++;
                continue;
            }

            if (result.need_more_input()) {
                if (!result.nodes.empty()) {
                    nodes.insert(nodes.end(), result.nodes.begin(), result.nodes.end());
                }

                ctx.parse_depth--;
                if (ctx.is_debug()) {
                    fprintf(stderr, "%sREPEAT -> NEED_MORE (count=%d, nodes=%zu)\n", debug_indent().c_str(),
                            match_count, nodes.size());
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, result.end, std::move(nodes));
            }

            // Child failed - stop trying
            if (ctx.is_debug()) {
                fprintf(stderr, "%sREPEAT: child failed, stopping\n", debug_indent().c_str());
            }
            break;
        }

        // Check if we got enough matches
        if (p.min_count > 0 && match_count < p.min_count) {
            ctx.parse_depth--;
            if (pos >= ctx.input.size() && ctx.is_lenient()) {
                if (ctx.is_debug()) {
                    fprintf(stderr, "%sREPEAT -> NEED_MORE (not enough matches: %d < %d)\n", debug_indent().c_str(),
                            match_count, p.min_count);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos, std::move(nodes));
            }
            if (ctx.is_debug()) {
                fprintf(stderr, "%sREPEAT -> FAIL (not enough matches: %d < %d)\n", debug_indent().c_str(), match_count,
                        p.min_count);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, pos);
        }

        ctx.parse_depth--;
        if (ctx.is_debug()) {
            fprintf(stderr, "%sREPEAT -> SUCCESS (count=%d, nodes=%zu)\n", debug_indent().c_str(), match_count,
                    nodes.size());
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos, std::move(nodes));
    }

    common_peg_parse_result operator()(const common_peg_and_parser & p) {
        auto result = arena.parse(p.child, ctx, start_pos);
        // Pass result but don't consume input
        return common_peg_parse_result(result.type, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_not_parser & p) {
        auto result = arena.parse(p.child, ctx, start_pos);

        if (result.success()) {
            // Fail if the underlying parser matches
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
        }

        if (result.need_more_input()) {
            // Propagate - need to know what child would match before negating
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos);
        }

        // Child failed, so negation succeeds
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_any_parser & /* p */) const {
        // Parse a single UTF-8 codepoint (not just a single byte)
        auto result = common_parse_utf8_codepoint(ctx.input, start_pos);

        if (result.status == utf8_parse_result::INCOMPLETE) {
            if (!ctx.is_lenient()) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos);
        }
        if (result.status == utf8_parse_result::INVALID) {
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, start_pos + result.bytes_consumed);
    }

    common_peg_parse_result operator()(const common_peg_space_parser & /* p */) {
        auto pos = start_pos;
        while (pos < ctx.input.size()) {
            auto c = static_cast<unsigned char>(ctx.input[pos]);
            if (std::isspace(c)) {
                ++pos;
            } else {
                break;
            }
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
    }

    common_peg_parse_result operator()(const common_peg_chars_parser & p) const {
        auto pos = start_pos;
        int match_count = 0;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (p.max_count == -1 || match_count < p.max_count) {
            auto result = common_parse_utf8_codepoint(ctx.input, pos);

            if (result.status == utf8_parse_result::INCOMPLETE) {
                if (match_count >= p.min_count) {
                    // We have enough matches, succeed with what we have
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
                }
                // Not enough matches yet
                if (!ctx.is_lenient()) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
            }

            if (result.status == utf8_parse_result::INVALID) {
                // Malformed UTF-8 in input
                if (match_count >= p.min_count) {
                    // We have enough matches, succeed up to here
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
                }
                // Not enough matches, fail
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }

            // Check if this codepoint matches our character class
            bool matches = false;
            for (const auto & range : p.ranges) {
                if (range.contains(result.codepoint)) {
                    matches = true;
                    break;
                }
            }

            // If negated, invert the match result
            if (p.negated) {
                matches = !matches;
            }

            if (matches) {
                pos += result.bytes_consumed;
                ++match_count;
            } else {
                // Character doesn't match, stop matching
                break;
            }
        }

        // Check if we got enough matches
        if (match_count < p.min_count) {
            if (pos >= ctx.input.size() && ctx.is_lenient()) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, pos);
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
    }

    static common_peg_parse_result handle_escape_sequence(common_peg_parse_context & ctx, size_t start, size_t & pos, const char delimiter) {
        ++pos; // consume '\'
        if (pos >= ctx.input.size()) {
            if (!ctx.is_lenient()) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
        }

        char c = ctx.input[pos];
        if (c == delimiter || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') {
            ++pos;
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start, pos);
        } else if (c == 'u') {
            return handle_unicode_escape(ctx, start, pos);
        } else {
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
        }
    }

    static common_peg_parse_result handle_unicode_escape(common_peg_parse_context & ctx, size_t start, size_t & pos) {
        ++pos; // consume 'u'
        for (int i = 0; i < 4; ++i) {
            if (pos >= ctx.input.size()) {
                if (!ctx.is_lenient()) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
            }
            if (!llama_common_is_hex_digit_rust((uint8_t) ctx.input[pos])) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
            }
            ++pos;
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start, pos);
    }

    common_peg_parse_result operator()(const common_peg_string_parser & p) {
        auto pos = start_pos;

        // Parse string content (without quotes)
        while (pos < ctx.input.size()) {
            char c = ctx.input[pos];

            if (c == p.delimiter) {
                // Found closing delimiter - success (don't consume it)
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
            }

            if (c == '\\') {
                auto result = handle_escape_sequence(ctx, start_pos, pos, p.delimiter);
                if (!result.success()) {
                    return result;
                }
            } else {
                auto utf8_result = common_parse_utf8_codepoint(ctx.input, pos);

                if (utf8_result.status == utf8_parse_result::INCOMPLETE) {
                    if (!ctx.is_lenient()) {
                        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                    }
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
                }

                if (utf8_result.status == utf8_parse_result::INVALID) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }

                pos += utf8_result.bytes_consumed;
            }
        }

        // Reached end without finding closing quote
        if (!ctx.is_lenient()) {
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
    }

    common_peg_parse_result operator()(const common_peg_until_parser & p) const {
        trie matcher(p.delimiters);

        // Scan input and check for delimiters
        size_t pos = start_pos;
        size_t last_valid_pos = start_pos;

        while (pos < ctx.input.size()) {
            auto utf8_result = common_parse_utf8_codepoint(ctx.input, pos);

            if (utf8_result.status == utf8_parse_result::INCOMPLETE) {
                // Incomplete UTF-8 sequence
                if (!ctx.is_lenient()) {
                    // Input is complete but UTF-8 is incomplete = malformed
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }
                // Return what we have so far (before incomplete sequence)
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, last_valid_pos);
            }

            if (utf8_result.status == utf8_parse_result::INVALID) {
                // Malformed UTF-8
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }

            // Check if a delimiter starts at this position
            auto match = matcher.check_at(ctx.input, pos);

            if (match == trie::COMPLETE_MATCH) {
                // Found a complete delimiter, return everything before it
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
            }

            if (match == trie::PARTIAL_MATCH) {
                // Found a partial match extending to end of input, return everything before it
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
            }

            pos += utf8_result.bytes_consumed;
            last_valid_pos = pos;
        }

        if (last_valid_pos == ctx.input.size() && ctx.is_lenient()) {
            // Reached the end of a partial stream, there might still be more input that we need to consume.
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, last_valid_pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, last_valid_pos);
    }

    common_peg_parse_result operator()(const common_peg_schema_parser & p) {
        return arena.parse(p.child, ctx, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_rule_parser & p) {
        // Parse the child
        auto result = arena.parse(p.child, ctx, start_pos);

        if (!result.fail()) {
            std::string_view text;
            if (result.start < ctx.input.size()) {
                text = std::string_view(ctx.input).substr(result.start, result.end - result.start);
            }

            auto node_id = ctx.ast.add_node(
                p.name,
                "",
                result.start,
                result.end,
                text,
                std::move(result.nodes),
                result.need_more_input()
            );

            return common_peg_parse_result(result.type, result.start, result.end, { node_id });
        }

        return result;
    }

    common_peg_parse_result operator()(const common_peg_tag_parser & p) {
        // Parse the child
        if (ctx.is_debug()) {
            fprintf(stderr, "%sTAG: %s\n", debug_indent().c_str(), p.tag.c_str());
        }
        auto result = arena.parse(p.child, ctx, start_pos);

        if (!result.fail()) {
            std::string_view text;
            if (result.start < ctx.input.size()) {
                text = std::string_view(ctx.input).substr(result.start, result.end - result.start);
            }

            auto node_id = ctx.ast.add_node(
                "",
                p.tag,
                result.start,
                result.end,
                text,
                std::move(result.nodes),
                result.need_more_input()
            );

            return common_peg_parse_result(result.type, result.start, result.end, { node_id });
        }

        return result;
    }

    common_peg_parse_result operator()(const common_peg_ref_parser & p) {
        auto rule_id = arena.get_rule(p.name);
        return arena.parse(rule_id, ctx, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_atomic_parser & p) {
        auto result = arena.parse(p.child, ctx, start_pos);
        if (result.need_more_input()) {
            // Clear nodes so they don't propagate up.
            result.nodes.clear();
        }
        return result;
    }

    common_peg_parse_result operator()(const common_peg_gbnf_parser & p) {
        return arena.parse(p.child, ctx, start_pos);
    }
};

common_peg_parse_result common_peg_arena::parse(common_peg_parse_context & ctx, size_t start) const {
    if (root_ == COMMON_PEG_INVALID_PARSER_ID) {
        throw std::runtime_error("No root parser set");
    }
    return parse(root_, ctx, start);
}

common_peg_parse_result common_peg_arena::parse(common_peg_parser_id id, common_peg_parse_context & ctx, size_t start) const {
    // Execute parser
    const auto & parser = parsers_.at(id);
    parser_executor exec(*this, ctx, start);
    return std::visit(exec, parser);
}

common_peg_parser_id common_peg_arena::resolve_ref(common_peg_parser_id id) {
    const auto & parser = parsers_.at(id);
    if (auto ref = std::get_if<common_peg_ref_parser>(&parser)) {
        return get_rule(ref->name);
    }
    return id;
}

static void bfs_node(common_peg_ast_arena &arena, std::ostringstream & oss, const common_peg_ast_node & node, int indent) {
    for (int i = 0; i < indent; i++) {
        oss << "  ";
    }
    oss << "NODE " << node.id;
    if (!node.rule.empty()) {
        oss << " (rule " << node.rule << ")";
    }
    if (!node.tag.empty()) {
        oss << " (tag " << node.tag << ")";
    }
    oss << " ['" << node.text << "']\n";
    for (const auto child : node.children) {
        bfs_node(arena, oss, arena.get(child), indent + 1);
    }
}

std::string common_peg_ast_arena::dump() {
    std::ostringstream oss;
    for (auto & node : nodes_) {
        bfs_node(*this, oss, node, 0);
    }
    return oss.str();
}

void common_peg_arena::resolve_refs() {
    // Walk through all parsers and replace refs with their corresponding rule IDs
    for (auto & parser : parsers_) {
        std::visit([this](auto & p) {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
                for (auto & child : p.children) {
                    child = resolve_ref(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
                for (auto & child : p.children) {
                    child = resolve_ref(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_repetition_parser> ||
                                 std::is_same_v<T, common_peg_and_parser> ||
                                 std::is_same_v<T, common_peg_not_parser> ||
                                 std::is_same_v<T, common_peg_tag_parser> ||
                                 std::is_same_v<T, common_peg_atomic_parser> ||
                                 std::is_same_v<T, common_peg_gbnf_parser>) {
                p.child = resolve_ref(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
                p.child = resolve_ref(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
                p.child = resolve_ref(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                                 std::is_same_v<T, common_peg_start_parser> ||
                                 std::is_same_v<T, common_peg_end_parser> ||
                                 std::is_same_v<T, common_peg_ref_parser> ||
                                 std::is_same_v<T, common_peg_until_parser> ||
                                 std::is_same_v<T, common_peg_literal_parser> ||
                                 std::is_same_v<T, common_peg_string_parser> ||
                                 std::is_same_v<T, common_peg_chars_parser> ||
                                 std::is_same_v<T, common_peg_any_parser> ||
                                 std::is_same_v<T, common_peg_space_parser>) {
                // These rules do not have children
            } else {
                static_assert(is_always_false_v<T>);
            }
        }, parser);
    }

    // Also flatten root if it's a ref
    if (root_ != COMMON_PEG_INVALID_PARSER_ID) {
        root_ = resolve_ref(root_);
    }
}

std::string common_peg_arena::dump(common_peg_parser_id id) const {
    std::unordered_set<common_peg_parser_id> visited;
    return dump_impl(id, visited);
}

std::string common_peg_arena::dump_impl(common_peg_parser_id                       id,
                                        std::unordered_set<common_peg_parser_id> & visited) const {
    // Check for cycles
    if (visited.count(id)) {
        return "[cycle]";
    }
    visited.insert(id);

    const auto & parser = parsers_.at(id);

    return std::visit([this, &visited](const auto & p) -> std::string {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, common_peg_epsilon_parser>) {
            return "Epsilon";
        } else if constexpr (std::is_same_v<T, common_peg_start_parser>) {
            return "Start";
        } else if constexpr (std::is_same_v<T, common_peg_end_parser>) {
            return "End";
        } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
            return "Literal(" + p.literal + ")";
        } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            std::vector<std::string> parts;
            for (const auto & child : p.children) {
                parts.push_back(dump_impl(child, visited));
            }
            return "Sequence(" + string_join(parts, ", ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            std::vector<std::string> parts;
            for (const auto & child : p.children) {
                parts.push_back(dump_impl(child, visited));
            }
            return "Choice(" + string_join(parts, ", ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
            if (p.max_count == -1) {
                return "Repetition(" + dump_impl(p.child, visited) + ", " + std::to_string(p.min_count) +
                        ", unbounded)";
            }
            return "Repetition(" + dump_impl(p.child, visited) + ", " + std::to_string(p.min_count) + ", " + std::to_string(p.max_count) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_and_parser>) {
            return "And(" + dump_impl(p.child, visited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_not_parser>) {
            return "Not(" + dump_impl(p.child, visited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
            return "Atomic(" + dump_impl(p.child, visited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_gbnf_parser>) {
            return "Gbnf(" + p.grammar + ", " + dump_impl(p.child, visited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
            return "Any";
        } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
            return "Space";
        } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
            if (p.max_count == -1) {
                return "CharRepeat(" + p.pattern + ", " + std::to_string(p.min_count) + ", unbounded)";
            }
            return "CharRepeat(" + p.pattern + ", " + std::to_string(p.min_count) + ", " + std::to_string(p.max_count) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_string_parser>) {
            return "String(" + std::string(1, p.delimiter) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
            return "Until(" + string_join(p.delimiters, " | ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
            return "Schema(" + dump_impl(p.child, visited) + ", " + (p.schema ? p.schema->dump() : "null") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            return "Rule(" + p.name + ", " + dump_impl(p.child, visited) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
            return "Ref(" + p.name + ")";
        } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
            return "Tag(" + p.tag + ", " + dump(p.child) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
            return "Atomic(" + dump(p.child) + ")";
        } else {
            return "Unknown";
        }
    }, parser);
}

common_peg_parser & common_peg_parser::operator=(const common_peg_parser & other) {
    id_ = other.id_;
    return *this;
}

common_peg_parser & common_peg_parser::operator+=(const common_peg_parser & other) {
    id_ = builder_.sequence({id_, other.id_});
    return *this;
}

common_peg_parser & common_peg_parser::operator|=(const common_peg_parser & other) {
    id_ = builder_.choice({id_, other.id_});
    return *this;
}

common_peg_parser common_peg_parser::operator+(const common_peg_parser & other) const {
    return builder_.sequence({id_, other.id_});
}

common_peg_parser common_peg_parser::operator|(const common_peg_parser & other) const {
    return builder_.choice({id_, other.id_});
}

common_peg_parser common_peg_parser::operator<<(const common_peg_parser & other) const {
    return builder_.sequence({id_, builder_.space(), other.id_});
}

common_peg_parser common_peg_parser::operator+(const char * str) const {
    return *this + builder_.literal(str);
}

common_peg_parser common_peg_parser::operator+(const std::string & str) const {
    return *this + builder_.literal(str);
}

common_peg_parser common_peg_parser::operator<<(const char * str) const {
    return *this << builder_.literal(str);
}

common_peg_parser common_peg_parser::operator<<(const std::string & str) const {
    return *this << builder_.literal(str);
}

common_peg_parser common_peg_parser::operator|(const char * str) const {
    return *this | builder_.literal(str);
}

common_peg_parser common_peg_parser::operator|(const std::string & str) const {
    return *this | builder_.literal(str);
}

common_peg_parser operator+(const char * str, const common_peg_parser & p) {
    return p.builder().literal(str) + p;
}

common_peg_parser operator+(const std::string & str, const common_peg_parser & p) {
    return operator+(str.c_str(), p);
}

common_peg_parser operator<<(const char * str, const common_peg_parser & p) {
    return p.builder().literal(str) << p;
}

common_peg_parser operator<<(const std::string & str, const common_peg_parser & p) {
    return operator<<(str.c_str(), p);
}

common_peg_parser operator|(const char * str, const common_peg_parser & p) {
    return p.builder().literal(str) | p;
}

common_peg_parser operator|(const std::string & str, const common_peg_parser & p) {
    return operator|(str.c_str(), p);
}

common_peg_parser_builder::common_peg_parser_builder() {}

common_peg_parser common_peg_parser_builder::sequence(const std::vector<common_peg_parser_id> & parsers) {
    // Flatten nested sequences
    std::vector<common_peg_parser_id> flattened;
    for (const auto & p : parsers) {
        const auto & parser = arena_.get(p);
        if (auto seq = std::get_if<common_peg_sequence_parser>(&parser)) {
            flattened.insert(flattened.end(), seq->children.begin(), seq->children.end());
        } else {
            flattened.push_back(p);
        }
    }
    return wrap(arena_.add_parser(common_peg_sequence_parser{flattened}));
}

common_peg_parser common_peg_parser_builder::sequence(const std::vector<common_peg_parser> & parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return sequence(ids);
}

common_peg_parser common_peg_parser_builder::sequence(std::initializer_list<common_peg_parser> parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return sequence(ids);
}

common_peg_parser common_peg_parser_builder::choice(const std::vector<common_peg_parser_id> & parsers) {
    // Flatten nested choices
    std::vector<common_peg_parser_id> flattened;
    for (const auto & p : parsers) {
        const auto & parser = arena_.get(p);
        if (auto choice = std::get_if<common_peg_choice_parser>(&parser)) {
            flattened.insert(flattened.end(), choice->children.begin(), choice->children.end());
        } else {
            flattened.push_back(p);
        }
    }
    return wrap(arena_.add_parser(common_peg_choice_parser{flattened}));
}

common_peg_parser common_peg_parser_builder::choice(const std::vector<common_peg_parser> & parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return choice(ids);
}

common_peg_parser common_peg_parser_builder::choice(std::initializer_list<common_peg_parser> parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return choice(ids);
}

common_peg_parser common_peg_parser_builder::chars(const std::string & classes, int min, int max) {
    auto [ranges, negated] = parse_char_classes(classes);
    return wrap(arena_.add_parser(common_peg_chars_parser{classes, ranges, negated, min, max}));
}

common_peg_parser common_peg_parser_builder::schema(const common_peg_parser & p, const std::string & name, const nlohmann::ordered_json & schema, bool raw) {
    return wrap(arena_.add_parser(common_peg_schema_parser{p.id(), name, std::make_shared<nlohmann::ordered_json>(schema), raw}));
}

common_peg_parser common_peg_parser_builder::rule(const std::string & name, const common_peg_parser & p, bool trigger) {
    auto clean_name = common_rust_string_to_std(
            llama_common_peg_rule_name_rust((const uint8_t *) name.data(), name.size()));
    auto rule_id = arena_.add_parser(common_peg_rule_parser{clean_name, p.id(), trigger});
    arena_.add_rule(clean_name, rule_id);
    return ref(clean_name);
}

common_peg_parser common_peg_parser_builder::rule(const std::string & name, const std::function<common_peg_parser()> & builder_fn, bool trigger) {
    auto clean_name = common_rust_string_to_std(
            llama_common_peg_rule_name_rust((const uint8_t *) name.data(), name.size()));
    if (arena_.has_rule(clean_name)) {
        return ref(clean_name);
    }

    // Create placeholder rule to allow recursive references
    auto placeholder = any();  // Temporary placeholder
    auto placeholder_rule_id = arena_.add_parser(common_peg_rule_parser{clean_name, placeholder.id(), trigger});
    arena_.add_rule(clean_name, placeholder_rule_id);

    // Build the actual parser
    auto parser = builder_fn();

    // Replace placeholder with actual rule
    auto rule_id = arena_.add_parser(common_peg_rule_parser{clean_name, parser.id(), trigger});
    arena_.rules_[clean_name] = rule_id;

    return ref(clean_name);
}

void common_peg_parser_builder::set_root(const common_peg_parser & p) {
    arena_.set_root(p.id());
}

common_peg_arena common_peg_parser_builder::build() {
    arena_.resolve_refs();
    return std::move(arena_);
}

// String primitives

common_peg_parser common_peg_parser_builder::string_content(char delimiter) {
    return wrap(arena_.add_parser(common_peg_string_parser{delimiter}));
}

common_peg_parser common_peg_parser_builder::double_quoted_string() {
    return rule("double-quoted-string", [this]() {
        return sequence({literal("\""), string_content('"'), literal("\""), space()});
    });
}

common_peg_parser common_peg_parser_builder::single_quoted_string() {
    return rule("single-quoted-string", [this]() {
        return sequence({literal("'"), string_content('\''), literal("'"), space()});
    });
}

common_peg_parser common_peg_parser_builder::quoted_string() {
    return rule("quoted-string", [this]() {
        return choice({double_quoted_string(), single_quoted_string()});
    });
}

// JSON parsers

common_peg_parser common_peg_parser_builder::json_number() {
   return rule("json-number", [this]() {
        auto digit1_9 = chars("[1-9]", 1, 1);
        auto digits = chars("[0-9]");
        auto int_part = choice({literal("0"), sequence({digit1_9, chars("[0-9]", 0, -1)})});
        auto frac = sequence({literal("."), digits});
        auto exp = sequence({choice({literal("e"), literal("E")}), optional(chars("[+-]", 1, 1)), digits});
        // Negative lookahead: only commit the number when the next character can't extend it.
        // At EOF in partial mode, chars returns NEED_MORE → negate propagates NEED_MORE → number not committed.
        // This prevents premature commits of partial numbers (e.g. "3" when "3.14" is incoming).
        auto not_number_continuation = negate(chars("[0-9.eE+-]", 1, 1));
        return sequence({ optional(literal("-")), int_part, optional(frac), optional(exp), not_number_continuation, space() });
    });
}

common_peg_parser common_peg_parser_builder::json_string() {
    return rule("json-string", [this]() {
        return sequence({literal("\""), string_content('"'), literal("\""), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_bool() {
    return rule("json-bool", [this]() {
        return sequence({choice({literal("true"), literal("false")}), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_null() {
    return rule("json-null", [this]() {
        return sequence({literal("null"), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_object() {
    return rule("json-object", [this]() {
        auto ws = space();
        auto member = sequence({json_string(), ws, literal(":"), ws, json()});
        auto members = sequence({member, zero_or_more(sequence({ws, literal(","), ws, member}))});
        return sequence({
            literal("{"),
            ws,
            choice({
                literal("}"),
                sequence({members, ws, literal("}")})
            }),
            ws
        });
    });
}

common_peg_parser common_peg_parser_builder::json_array() {
    return rule("json-array", [this]() {
        auto ws = space();
        auto elements = sequence({json(), zero_or_more(sequence({literal(","), ws, json()}))});
        return sequence({
            literal("["),
            ws,
            choice({
                literal("]"),
                sequence({elements, ws, literal("]")})
            }),
            ws
        });
    });
}

common_peg_parser common_peg_parser_builder::json() {
    return rule("json-value", [this]() {
        return choice({
            json_object(),
            json_array(),
            json_string(),
            json_number(),
            json_bool(),
            json_null()
        });
    });
}

common_peg_parser common_peg_parser_builder::python_string() {
    return rule("python-string", [this]() {
        return choice({double_quoted_string(), single_quoted_string()});
    });
}

common_peg_parser common_peg_parser_builder::python_number() {
    return json_number();
}

common_peg_parser common_peg_parser_builder::python_bool() {
    return rule("python-bool", [this]() {
        return sequence({
            choice({literal("True"), literal("False")}),
            space()
        });
    });
}

common_peg_parser common_peg_parser_builder::python_null() {
    return rule("python-none", [this]() {
        return sequence({literal("None"), space()});
    });
}

common_peg_parser common_peg_parser_builder::python_dict() {
    return rule("python-dict", [this]() {
        auto ws = space();
        auto member = sequence({python_string(), ws, literal(":"), ws, python_value()});
        auto members = sequence({member, zero_or_more(sequence({ws, literal(","), ws, member}))});
        return sequence({
            literal("{"),
            ws,
            choice({
                literal("}"),
                sequence({members, ws, literal("}")})
            }),
            ws
        });
    });
}

common_peg_parser common_peg_parser_builder::python_array() {
    return rule("python-array", [this]() {
        auto ws = space();
        auto elements = sequence({python_value(), zero_or_more(sequence({literal(","), ws, python_value()}))});
        return sequence({
            literal("["),
            ws,
            choice({
                literal("]"),
                sequence({elements, ws, literal("]")})
            }),
            ws
        });
    });
}

common_peg_parser common_peg_parser_builder::python_value() {
    return rule("python-value", [this]() {
        return choice({
            python_dict(),
            python_array(),
            python_string(),
            python_number(),
            python_bool(),
            python_null()
        });
    });
}

common_peg_parser common_peg_parser_builder::marker() {
    auto sharp_bracket_parser = literal("<") + until(">") + literal(">");
    auto square_bracket_parser = literal("[") + until("]") + literal("]");
    return choice({ sharp_bracket_parser, square_bracket_parser });
}

common_peg_parser common_peg_parser_builder::json_member(const std::string & key, const common_peg_parser & p) {
    auto ws = space();
    return sequence({
        literal("\"" + key + "\""),
        ws,
        literal(":"),
        ws,
        p,
    });
}

static std::string gbnf_excluding_pattern(const std::vector<std::string> & strings) {
    trie matcher(strings);
    auto pieces = matcher.collect_prefix_and_next();

    std::string pattern;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i > 0) {
            pattern += " | ";
        }

        const auto & pre = pieces[i].prefix;
        const auto & chars = pieces[i].next_chars;

        std::string cls;
        cls.reserve(chars.size());
        for (uint32_t ch : chars) {
            cls += common_rust_string_to_std(llama_common_gbnf_escape_char_class_rust(ch));
        }

        if (!pre.empty()) {
            pattern += gbnf_format_literal(common_unicode_cpts_to_utf8(pre)) + " [^" + cls + "]";
        } else {
            pattern += "[^" + cls + "]";
        }
    }

    return "(" + pattern + ")*";
}

static std::unordered_set<std::string> collect_reachable_rules(
    const common_peg_arena & arena,
    const common_peg_parser_id & rule
) {
    std::unordered_set<std::string> reachable;
    std::unordered_set<std::string> visited;

    std::function<void(common_peg_parser_id)> visit = [&](common_peg_parser_id id) {
        const auto & parser = arena.get(id);

        std::visit([&](const auto & p) {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                          std::is_same_v<T, common_peg_start_parser> ||
                          std::is_same_v<T, common_peg_end_parser> ||
                          std::is_same_v<T, common_peg_until_parser> ||
                          std::is_same_v<T, common_peg_literal_parser> ||
                          std::is_same_v<T, common_peg_chars_parser> ||
                          std::is_same_v<T, common_peg_space_parser> ||
                          std::is_same_v<T, common_peg_any_parser> ||
                          std::is_same_v<T, common_peg_string_parser>) {
                // These parsers do not have any children
            } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
                for (auto child : p.children) {
                    visit(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
                for (auto child : p.children) {
                    visit(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_repetition_parser> ||
                                 std::is_same_v<T, common_peg_and_parser> ||
                                 std::is_same_v<T, common_peg_not_parser> ||
                                 std::is_same_v<T, common_peg_tag_parser> ||
                                 std::is_same_v<T, common_peg_atomic_parser> ||
                                 std::is_same_v<T, common_peg_gbnf_parser> ||
                                 std::is_same_v<T, common_peg_schema_parser>) {
                visit(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
                if (visited.find(p.name) == visited.end()) {
                    visited.insert(p.name);
                    reachable.insert(p.name);
                    visit(p.child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
                // Traverse rules so we pick up everything
                auto referenced_rule = arena.get_rule(p.name);
                visit(referenced_rule);
            } else {
                static_assert(is_always_false_v<T>);
            }
        }, parser);
    };

    visit(rule);
    return reachable;
}

// GBNF generation implementation
void common_peg_arena::build_grammar(const common_grammar_builder & builder, bool lazy) const {
    auto schema_delegates = [](const common_peg_schema_parser & s) -> bool {
        if (!s.schema) {
            return true;
        }
        if (s.raw && s.schema->contains("type")) {
            const auto & type_val = s.schema->at("type");
            if (type_val.is_string() && type_val == "string") {
                return true;
            }
            // Handle nullable types like ["string", "null"] - delegate when the
            // non-null type is string, since the tagged format uses raw text
            if (type_val.is_array()) {
                for (const auto & t : type_val) {
                    if (t.is_string() && t.get<std::string>() != "null") {
                        return t.get<std::string>() == "string";
                    }
                }
            }
        }
        // Delegate for enum schemas in raw mode - enum values are literal strings
        if (s.raw && !s.schema->contains("type") && s.schema->contains("enum")) {
            return true;
        }
        return false;
    };

    // Unwrap the parser so we can properly check if it's a sequence or choice
    auto effective_parser = [&](common_peg_parser_id id) -> const common_peg_parser_variant & {
        while (true) {
            const auto & p = parsers_.at(id);
            if (const auto * tag = std::get_if<common_peg_tag_parser>(&p)) {
                id = tag->child;
            } else if (const auto * atomic = std::get_if<common_peg_atomic_parser>(&p)) {
                id = atomic->child;
            } else if (const auto * schema = std::get_if<common_peg_schema_parser>(&p)) {
                if (schema_delegates(*schema)) {
                    id = schema->child;
                } else {
                    return p;
                }
            } else {
                return p;
            }
        }
    };

    // Generate GBNF for a parser
    std::function<std::string(common_peg_parser_id)> to_gbnf = [&](common_peg_parser_id id) -> std::string {
        const auto & parser = parsers_.at(id);

        return std::visit([&](const auto & p) -> std::string {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                          std::is_same_v<T, common_peg_start_parser> ||
                          std::is_same_v<T, common_peg_end_parser>) {
                return "";
            } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
                return gbnf_format_literal(p.literal);
            } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
                std::string s;
                for (const auto & child : p.children) {
                    auto child_gbnf = to_gbnf(child);
                    if (child_gbnf.empty()) {
                        continue;
                    }
                    if (!s.empty()) {
                        s += " ";
                    }
                    const auto & child_parser = effective_parser(child);
                    if (std::holds_alternative<common_peg_choice_parser>(child_parser) ||
                        std::holds_alternative<common_peg_sequence_parser>(child_parser)) {
                        s += "(" + child_gbnf + ")";
                    } else {
                        s += child_gbnf;
                    }
                }
                return s;
            } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
                std::string s;
                for (const auto & child : p.children) {
                    if (!s.empty()) {
                        s += " | ";
                    }
                    auto child_gbnf = to_gbnf(child);
                    const auto & child_parser = effective_parser(child);
                    if (std::holds_alternative<common_peg_choice_parser>(child_parser)) {
                        s += "(" + child_gbnf + ")";
                    } else {
                        s += child_gbnf;
                    }
                }
                return s;
            } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
                auto child_gbnf = to_gbnf(p.child);
                const auto & child_parser = effective_parser(p.child);
                if (std::holds_alternative<common_peg_choice_parser>(child_parser) ||
                    std::holds_alternative<common_peg_sequence_parser>(child_parser)) {
                    child_gbnf = "(" + child_gbnf + ")";
                }
                if (p.min_count == 0 && p.max_count == 1) {
                    return child_gbnf + "?";
                }
                if (p.min_count == 0 && p.max_count == -1) {
                    return child_gbnf + "*";
                }
                if (p.min_count == 1 && p.max_count == -1) {
                    return child_gbnf + "+";
                }
                if (p.max_count == -1) {
                    return child_gbnf + "{" + std::to_string(p.min_count) + ",}";
                }
                if (p.min_count == p.max_count) {
                    if (p.min_count == 1) {
                        return child_gbnf;
                    }
                    return child_gbnf + "{" + std::to_string(p.min_count) + "}";
                }
                return child_gbnf + "{" + std::to_string(p.min_count) + "," + std::to_string(p.max_count) + "}";
            } else if constexpr (std::is_same_v<T, common_peg_and_parser> || std::is_same_v<T, common_peg_not_parser>) {
                return "";  // Lookahead not supported in GBNF
            } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
                return ".";
            } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
                return "space";
            } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
                std::string result = p.pattern;
                if (p.min_count == 0 && p.max_count == 1) {
                    return result + "?";
                }
                if (p.min_count == 0 && p.max_count == -1) {
                    return result + "*";
                }
                if (p.min_count == 1 && p.max_count == -1) {
                    return result + "+";
                }
                if (p.max_count == -1) {
                    return result + "{" + std::to_string(p.min_count) + ",}";
                }
                if (p.min_count == p.max_count) {
                    if (p.min_count == 1) {
                        return result;
                    }
                    return result + "{" + std::to_string(p.min_count) + "}";
                }
                return result + "{" + std::to_string(p.min_count) + "," + std::to_string(p.max_count) + "}";
            } else if constexpr (std::is_same_v<T, common_peg_string_parser>) {
                const std::string delim(1, p.delimiter);
                return R"(( [^)" + delim + R"(\\] | "\\" ( [)" + delim + R"(\\/ bfnrt] | "u" [0-9a-fA-F]{4} ) )*)";
            } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
                if (p.delimiters.empty()) {
                    return ".*";
                }
                return gbnf_excluding_pattern(p.delimiters);
            } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
                if (schema_delegates(p)) {
                    return to_gbnf(p.child);
                }
                return builder.add_schema(p.name, *p.schema);
            } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
                return p.name;
            } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
                // Refs should not exist after flattening, but kept just in case
                return p.name;
            } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
                return to_gbnf(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
                return to_gbnf(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_gbnf_parser>) {
                return p.grammar;
            } else {
                static_assert(is_always_false_v<T>);
            }
        }, parser);
    };

    // Collect reachable rules
    std::unordered_set<std::string> reachable_rules;

    if (lazy) {
        // Collect rules reachable from trigger rules
        for (const auto & [name, id] : rules_) {
            const auto & parser = parsers_.at(id);
            if (auto rule = std::get_if<common_peg_rule_parser>(&parser)) {
                if (rule->trigger) {
                    // Mark trigger as reachable and visit it
                    reachable_rules.insert(name);
                    auto add_rules = collect_reachable_rules(*this, id);
                    reachable_rules.insert(add_rules.begin(), add_rules.end());
                }
            }
        }
    } else {
        // Collect rules reachable from root
        reachable_rules = collect_reachable_rules(*this, root_);
    }

    // Create GBNF rules for all reachable rules
    for (const auto & [name, rule_id] : rules_) {
        if (reachable_rules.find(name) == reachable_rules.end()) {
            continue;
        }

        const auto & parser = parsers_.at(rule_id);
        if (auto rule = std::get_if<common_peg_rule_parser>(&parser)) {
            builder.add_rule(rule->name, to_gbnf(rule->child));
        }
    }

    if (lazy) {
        // Generate root rule from trigger rules only
        std::vector<std::string> trigger_names;
        for (const auto & [name, rule_id] : rules_) {
            const auto & parser = parsers_.at(rule_id);
            if (auto rule = std::get_if<common_peg_rule_parser>(&parser)) {
                if (rule->trigger) {
                    trigger_names.push_back(rule->name);
                }
            }
        }

        // Sort for predictable order
        std::sort(trigger_names.begin(), trigger_names.end());
        builder.add_rule("root", string_join(trigger_names, " | "));
    } else if (root_ != COMMON_PEG_INVALID_PARSER_ID) {
        builder.add_rule("root", to_gbnf(root_));
    }
}

static nlohmann::json serialize_parser_variant(const common_peg_parser_variant & variant) {
    using json = nlohmann::json;

    return std::visit([](const auto & p) -> json {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, common_peg_epsilon_parser>) {
            return json{{"type", "epsilon"}};
        } else if constexpr (std::is_same_v<T, common_peg_start_parser>) {
            return json{{"type", "start"}};
        } else if constexpr (std::is_same_v<T, common_peg_end_parser>) {
            return json{{"type", "end"}};
        } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
            return json{{"type", "literal"}, {"literal", p.literal}};
        } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            return json{{"type", "sequence"}, {"children", p.children}};
        } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            return json{{"type", "choice"}, {"children", p.children}};
        } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
            return json{
                {"type", "repetition"},
                {"child", p.child},
                {"min_count", p.min_count},
                {"max_count", p.max_count}
            };
        } else if constexpr (std::is_same_v<T, common_peg_and_parser>) {
            return json{{"type", "and"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_not_parser>) {
            return json{{"type", "not"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
            return json{{"type", "any"}};
        } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
            return json{{"type", "space"}};
        } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
            json ranges = json::array();
            for (const auto & range : p.ranges) {
                ranges.push_back({{"start", range.start}, {"end", range.end}});
            }
            return json{
                {"type", "chars"},
                {"pattern", p.pattern},
                {"ranges", ranges},
                {"negated", p.negated},
                {"min_count", p.min_count},
                {"max_count", p.max_count}
            };
        } else if constexpr (std::is_same_v<T, common_peg_string_parser>) {
            return json{{"type", "string"}, {"delimiter", std::string(1, p.delimiter)}};
        } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
            return json{{"type", "until"}, {"delimiters", p.delimiters}};
        } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
            return json{
                {"type", "schema"},
                {"child", p.child},
                {"name", p.name},
                {"schema", p.schema ? *p.schema : nullptr},
                {"raw", p.raw}
            };
        } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            return json{
                {"type", "rule"},
                {"name", p.name},
                {"child", p.child},
                {"trigger", p.trigger}
            };
        } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
            return json{{"type", "ref"}, {"name", p.name}};
        } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
            return json{{"type", "atomic"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
            return json{
                {"type", "tag"},
                {"child", p.child},
                {"tag", p.tag}
            };
        } else if constexpr (std::is_same_v<T, common_peg_gbnf_parser>) {
            return json{{"type", "gbnf"}, {"child", p.child}, {"grammar", p.grammar}};
        }
    }, variant);
}

nlohmann::json common_peg_arena::to_json() const {
    auto parsers = nlohmann::json::array();
    for (const auto & parser : parsers_) {
        parsers.push_back(serialize_parser_variant(parser));
    }
    return nlohmann::json{
        {"parsers", parsers},
        {"rules", rules_},
        {"root", root_}
    };
}

static common_peg_parser_variant deserialize_parser_variant(const nlohmann::json & j) {
    if (!j.contains("type") || !j["type"].is_string()) {
        throw std::runtime_error("Parser variant JSON missing or invalid 'type' field");
    }

    std::string type = j["type"];

    if (type == "epsilon") {
        return common_peg_epsilon_parser{};
    }
    if (type == "start") {
        return common_peg_start_parser{};
    }
    if (type == "end") {
        return common_peg_end_parser{};
    }
    if (type == "literal") {
        if (!j.contains("literal") || !j["literal"].is_string()) {
            throw std::runtime_error("literal parser missing or invalid 'literal' field");
        }
        return common_peg_literal_parser{j["literal"]};
    }
    if (type == "sequence") {
        if (!j.contains("children") || !j["children"].is_array()) {
            throw std::runtime_error("sequence parser missing or invalid 'children' field");
        }
        return common_peg_sequence_parser{j["children"].get<std::vector<common_peg_parser_id>>()};
    }
    if (type == "choice") {
        if (!j.contains("children") || !j["children"].is_array()) {
            throw std::runtime_error("choice parser missing or invalid 'children' field");
        }
        return common_peg_choice_parser{j["children"].get<std::vector<common_peg_parser_id>>()};
    }
    if (type == "repetition") {
        if (!j.contains("child") || !j.contains("min_count") || !j.contains("max_count")) {
            throw std::runtime_error("repetition parser missing required fields");
        }
        return common_peg_repetition_parser{
            j["child"].get<common_peg_parser_id>(),
            j["min_count"].get<int>(),
            j["max_count"].get<int>()
        };
    }
    if (type == "and") {
        if (!j.contains("child")) {
            throw std::runtime_error("and parser missing 'child' field");
        }
        return common_peg_and_parser{j["child"].get<common_peg_parser_id>()};
    }
    if (type == "not") {
        if (!j.contains("child")) {
            throw std::runtime_error("not parser missing 'child' field");
        }
        return common_peg_not_parser{j["child"].get<common_peg_parser_id>()};
    }
    if (type == "any") {
        return common_peg_any_parser{};
    }
    if (type == "space") {
        return common_peg_space_parser{};
    }
    if (type == "chars") {
        if (!j.contains("pattern") || !j.contains("ranges") || !j.contains("negated") ||
            !j.contains("min_count") || !j.contains("max_count")) {
            throw std::runtime_error("chars parser missing required fields");
        }
        common_peg_chars_parser parser;
        parser.pattern = j["pattern"];
        parser.negated = j["negated"];
        parser.min_count = j["min_count"];
        parser.max_count = j["max_count"];
        for (const auto & range_json : j["ranges"]) {
            if (!range_json.contains("start") || !range_json.contains("end")) {
                throw std::runtime_error("char_range missing 'start' or 'end' field");
            }
            parser.ranges.push_back({
                range_json["start"].get<uint32_t>(),
                range_json["end"].get<uint32_t>()
            });
        }
        return parser;
    }
    if (type == "string") {
        if (!j.contains("delimiter")) {
            throw std::runtime_error("string parser missing delimiter field.");
        }
        std::string delimiter = j["delimiter"];
        if (delimiter.empty()) {
            throw std::runtime_error("string parser delimiter is empty.");
        }
        return common_peg_string_parser{delimiter[0]};
    }
    if (type == "until") {
        if (!j.contains("delimiters") || !j["delimiters"].is_array()) {
            throw std::runtime_error("until parser missing or invalid 'delimiters' field");
        }
        return common_peg_until_parser{j["delimiters"].get<std::vector<std::string>>()};
    }
    if (type == "schema") {
        if (!j.contains("child") || !j.contains("name") || !j.contains("schema") || !j.contains("raw")) {
            throw std::runtime_error("schema parser missing required fields");
        }
        common_peg_schema_parser parser;
        parser.child = j["child"].get<common_peg_parser_id>();
        parser.name = j["name"];
        if (!j["schema"].is_null()) {
            parser.schema = std::make_shared<nlohmann::ordered_json>(j["schema"]);
        }
        parser.raw = j["raw"].get<bool>();
        return parser;
    }
    if (type == "rule") {
        if (!j.contains("name") || !j.contains("child") || !j.contains("trigger")) {
            throw std::runtime_error("rule parser missing required fields");
        }
        return common_peg_rule_parser{
            j["name"].get<std::string>(),
            j["child"].get<common_peg_parser_id>(),
            j["trigger"].get<bool>()
        };
    }
    if (type == "ref") {
        if (!j.contains("name") || !j["name"].is_string()) {
            throw std::runtime_error("ref parser missing or invalid 'name' field");
        }
        return common_peg_ref_parser{j["name"]};
    }
    if (type == "atomic") {
        if (!j.contains("child")) {
            throw std::runtime_error("tag parser missing required fields");
        }
        return common_peg_atomic_parser{
            j["child"].get<common_peg_parser_id>(),
        };
    }
    if (type == "tag") {
        if (!j.contains("child") || !j.contains("tag")) {
            throw std::runtime_error("tag parser missing required fields");
        }
        return common_peg_tag_parser{
            j["child"].get<common_peg_parser_id>(),
            j["tag"].get<std::string>(),
        };
    }

    if (type == "gbnf") {
        if (!j.contains("child") || !j.contains("grammar")) {
            throw std::runtime_error("gbnf parser missing required fields");
        }
        return common_peg_gbnf_parser{
            j["child"].get<common_peg_parser_id>(),
            j["grammar"].get<std::string>(),
        };
    }

    throw std::runtime_error("Unknown parser type: " + type);
}

common_peg_arena common_peg_arena::from_json(const nlohmann::json & j) {
    if (!j.contains("parsers") || !j["parsers"].is_array()) {
        throw std::runtime_error("JSON missing or invalid 'parsers' array");
    }
    if (!j.contains("rules") || !j["rules"].is_object()) {
        throw std::runtime_error("JSON missing or invalid 'rules' object");
    }
    if (!j.contains("root")) {
        throw std::runtime_error("JSON missing 'root' field");
    }

    common_peg_arena arena;

    const auto & parsers_json = j["parsers"];
    arena.parsers_.reserve(parsers_json.size());
    for (const auto & parser_json : parsers_json) {
        arena.parsers_.push_back(deserialize_parser_variant(parser_json));
    }

    arena.rules_ = j["rules"].get<std::unordered_map<std::string, common_peg_parser_id>>();

    for (const auto & [name, id] : arena.rules_) {
        if (id >= arena.parsers_.size()) {
            throw std::runtime_error("Rule '" + name + "' references invalid parser ID: " + std::to_string(id));
        }
    }

    arena.root_ = j["root"].get<common_peg_parser_id>();
    if (arena.root_ != COMMON_PEG_INVALID_PARSER_ID && arena.root_ >= arena.parsers_.size()) {
        throw std::runtime_error("Root references invalid parser ID: " + std::to_string(arena.root_));
    }

    return arena;
}

std::string common_peg_arena::save() const {
    return to_json().dump();
}

void common_peg_arena::load(const std::string & data) {
    *this = from_json(nlohmann::json::parse(data));
}

common_peg_arena build_peg_parser(const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn) {
    common_peg_parser_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

#include "common.h.inc"
#include "ggml.h.inc"

#include <nlohmann/json.hpp.inc>

using ordered_json = nlohmann::ordered_json;

static std::string_view trim_trailing_space(std::string_view sv, int max = -1) {
    common_rust_string_view view =
        llama_common_trim_trailing_space_view_rust((const uint8_t *) sv.data(), sv.size(), max);
    return std::string_view((const char *) view.data, view.len);
}

static std::string_view trim_leading_space(std::string_view sv, int max = -1) {
    common_rust_string_view view =
        llama_common_trim_leading_space_view_rust((const uint8_t *) sv.data(), sv.size(), max);
    return std::string_view((const char *) view.data, view.len);
}

static std::string_view trim(std::string_view sv) {
    return trim_trailing_space(trim_leading_space(sv, 1));
}

// JSON-escape a string and return the inner content (without surrounding quotes).
static std::string escape_json_string_inner(const std::string & s) {
    std::string escaped = ordered_json(s).dump();
    if (escaped.size() >= 2 && escaped.front() == '"' && escaped.back() == '"') {
        return escaped.substr(1, escaped.size() - 2);
    }
    return escaped;
}

void tag_based_peg_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    arena.visit(result, [this](const common_peg_ast_node & node) {
        if (!node.tag.empty()) {
            tags[node.tag] = std::string(node.text);
        }
    });
}

tagged_parse_result tagged_peg_parser::parse_and_extract(const std::string & input, common_peg_parse_flags extra_flags) const {
    common_peg_parse_context ctx(input, flags | extra_flags);
    auto parse_result = arena.parse(ctx);

    tag_based_peg_mapper mapper;
    mapper.from_ast(ctx.ast, parse_result);

    return { std::move(parse_result), std::move(mapper.tags) };
}

tagged_parse_result tagged_peg_parser::parse_anywhere_and_extract(const std::string & input) const {
    if (input.empty()) {
        return parse_and_extract(input);
    }
    for (size_t i = 0; i < input.size(); i++) {
        common_peg_parse_context ctx(input, flags);
        auto parse_result = arena.parse(ctx, i);
        if (parse_result.success() || i == input.size() - 1) {
            tag_based_peg_mapper mapper;
            mapper.from_ast(ctx.ast, parse_result);
            return { std::move(parse_result), std::move(mapper.tags) };
        }
    }
    GGML_ABORT("Should not happen");
}

tagged_peg_parser build_tagged_peg_parser(
    const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn) {
    common_peg_parser_builder builder;
    builder.set_root(fn(builder));
    return { builder.build() };
}

common_peg_parser common_chat_peg_builder::tag_with_safe_content(const std::string &       tag_name,
                                                                 const std::string &       marker,
                                                                 const common_peg_parser & p) {
    if (marker.empty()) {
        return zero_or_more(choice({ p, rule(tag_name, content(any())) }));
    }
    auto content_chunk = rule(tag_name, content(negate(literal(marker)) + any() + until(marker)));
    return zero_or_more(choice({ p, content_chunk }));
}

std::string & common_chat_peg_mapper::args_target() {
    return (current_tool && !current_tool->name.empty()) ? current_tool->arguments : args_buffer;
}

std::string common_chat_peg_mapper::normalize_container_value(const std::string & input) {
    return common_rust_string_to_std(
            llama_common_normalize_quotes_to_json_rust((const uint8_t *) input.data(), input.size()));
}

void common_chat_peg_mapper::from_ast(const common_peg_ast_arena &    arena,
                                      const common_peg_parse_result & parse_result_arg) {
    arena.visit(parse_result_arg, [this](const common_peg_ast_node & node) { map(node); });
    // Flush any pending tool call that was started but never got a name
    // This happens during partial parsing when the tool call is incomplete
    if (pending_tool_call.has_value() && !pending_tool_call->name.empty()) {
        if (!args_buffer.empty()) {
            pending_tool_call->arguments = args_buffer;
        }
        if (closing_quote_pending && !pending_tool_call->arguments.empty()) {
            pending_tool_call->arguments += "\"";
        }
        result.tool_calls.push_back(pending_tool_call.value());
        pending_tool_call.reset();
    }

    // Discard whitespace-only reasoning content (e.g. from <think></think> prefill)
    if (!result.reasoning_content.empty()) {
        bool all_whitespace = true;
        for (char c : result.reasoning_content) {
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                all_whitespace = false;
                break;
            }
        }
        if (all_whitespace) {
            result.reasoning_content.clear();
        }
    }
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    // Handle reasoning/content tags
    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content   = node.tag == common_chat_peg_builder::CONTENT;

    if (is_reasoning) { // GPT OSS can have more than 1 reasoning block, so concatenate here
        result.reasoning_content += std::string(node.text);
    }

    if (is_content) {
        // Concatenate content from multiple content nodes (e.g., when reasoning markers
        // are preserved before content markers in reasoning_format=NONE mode)
        result.content += std::string(node.text);
    }

    // Handle tool-related tags (supporting both JSON and tagged formats)
    bool is_tool_open  = node.tag == common_chat_peg_builder::TOOL_OPEN;
    bool is_tool_close = node.tag == common_chat_peg_builder::TOOL_CLOSE;
    bool is_tool_name  = node.tag == common_chat_peg_builder::TOOL_NAME;
    bool is_tool_id    = node.tag == common_chat_peg_builder::TOOL_ID;
    bool is_tool_args  = node.tag == common_chat_peg_builder::TOOL_ARGS;
    bool is_arg_open   = node.tag == common_chat_peg_builder::TOOL_ARG_OPEN;
    bool is_arg_close  = node.tag == common_chat_peg_builder::TOOL_ARG_CLOSE;
    bool is_arg_name         = node.tag == common_chat_peg_builder::TOOL_ARG_NAME;
    bool is_arg_value        = node.tag == common_chat_peg_builder::TOOL_ARG_VALUE;
    bool is_arg_string_value = node.tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE;

    if (is_tool_open) {
        pending_tool_call     = common_chat_tool_call();
        current_tool          = &pending_tool_call.value();
        arg_count             = 0;
        args_buffer.clear();
        closing_quote_pending = false;
    }

    if (is_tool_id && current_tool) {
        auto text = trim_trailing_space(node.text);
        if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
            text = text.substr(1, text.size() - 2);
        }
        current_tool->id = std::string(text);
    }

    if (is_tool_name && current_tool) {
        current_tool->name = std::string(trim_trailing_space(node.text));
        // Now that we have the name, populate the arguments from the buffer
        if (!args_buffer.empty()) {
            current_tool->arguments = args_buffer;
            args_buffer.clear();
        } else if (current_tool->arguments.empty()) {
            current_tool->arguments = "{";
        }
        // Add the tool call to results so streaming can see it
        if (pending_tool_call.has_value()) {
            result.tool_calls.push_back(pending_tool_call.value());
            pending_tool_call.reset();
            current_tool = &result.tool_calls.back();
        }
    }

    if (is_tool_args && current_tool) {
        // For JSON format: arguments come as a complete JSON object
        // For tagged format: built up from individual arg_name/arg_value nodes
        auto text = trim_trailing_space(node.text);
        if (!text.empty() && text.front() == '{') {
            args_target() = std::string(text);
        }
    }

    if (is_arg_open) {
        closing_quote_pending = false;
    }

    if (is_arg_name && current_tool) {
        std::string arg_entry;
        if (arg_count > 0) {
            arg_entry = ",";
        }
        arg_entry += ordered_json(trim(node.text)).dump() + ":";
        ++arg_count;

        auto & target = args_target();
        if (target.empty()) {
            target = "{";
        }
        target += arg_entry;
    }

    if ((is_arg_value || is_arg_string_value) && current_tool) {
        std::string value_content = std::string(trim_trailing_space(trim_leading_space(node.text, 1), 1));

        std::string value_to_add;
        if (value_content.empty() && is_arg_string_value) {
            // Empty string value - arg_close will add the closing quote
            value_to_add          = "\"";
            closing_quote_pending = true;
        } else if (!value_content.empty() && is_arg_string_value) {
            // Schema declares this as string type - always treat as literal string value
            if (!closing_quote_pending) {
                value_to_add          = "\"";
                closing_quote_pending = true;
            }
            value_to_add += escape_json_string_inner(value_content);
        } else if (!value_content.empty()) {
            // For potential containers, normalize Python-style single quotes to JSON double quotes
            bool is_potential_container = value_content[0] == '[' || value_content[0] == '{';
            if (is_potential_container) {
                value_content = normalize_container_value(value_content);
            }

            // Try to parse as JSON value (number, bool, null, object, array)
            try {
                ordered_json parsed = ordered_json::parse(value_content);
                if (parsed.is_string()) {
                    // Don't add closing quote yet (added by arg_close) for monotonic streaming
                    std::string escaped = parsed.dump();
                    if (!escaped.empty() && escaped.back() == '"') {
                        escaped.pop_back();
                    }
                    value_to_add          = escaped;
                    closing_quote_pending = true;
                } else {
                    // Non-string values: use raw content to preserve whitespace for monotonicity
                    value_to_add = value_content;
                }
            } catch (...) {
                if (node.is_partial && is_potential_container) {
                    // Partial container: pass through the already-normalized content
                    value_to_add = value_content;
                } else {
                    // Not valid JSON - treat as string value
                    if (!closing_quote_pending) {
                        value_to_add          = "\"";
                        closing_quote_pending = true;
                    }
                    value_to_add += escape_json_string_inner(value_content);
                }
            }
        }

        args_target() += value_to_add;
    }

    if (is_arg_close && current_tool) {
        if (closing_quote_pending) {
            args_target() += "\"";
            closing_quote_pending = false;
        }
    }

    if (is_tool_close && current_tool) {
        // Flush buffer to arguments if tool name was never seen
        if (current_tool->name.empty() && !args_buffer.empty()) {
            current_tool->arguments = args_buffer;
            args_buffer.clear();
        }
        // Close any pending string quote
        if (closing_quote_pending) {
            current_tool->arguments += "\"";
            closing_quote_pending = false;
        }
        // Close any unclosed braces (accounts for nested objects)
        for (int d = llama_common_json_brace_depth_rust(
                    (const uint8_t *) current_tool->arguments.data(), current_tool->arguments.size());
                d > 0; d--) {
            current_tool->arguments += "}";
        }
        // Add tool call to results if named; otherwise discard
        if (pending_tool_call.has_value()) {
            if (!current_tool->name.empty()) {
                result.tool_calls.push_back(pending_tool_call.value());
            }
            pending_tool_call.reset();
        }
    }
}

common_peg_parser common_chat_peg_builder::standard_constructed_tools(
    const std::map<std::string, std::string> & markers,
    const ordered_json &                       tools,
    bool                                       parallel_tool_calls,
    bool                                       force_tool_calls) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    // Extract markers with defaults
    auto get_marker = [&markers](const std::string & key, const std::string & default_val = "") -> std::string {
        auto it = markers.find(key);
        return it != markers.end() ? it->second : default_val;
    };

    std::string section_start    = get_marker("tool_call_start_marker", "<tool_call>");
    std::string section_end      = get_marker("tool_call_end_marker", "</tool_call>");
    std::string func_opener      = get_marker("function_opener", "<function=");
    std::string func_name_suffix = get_marker("function_name_suffix", ">");
    std::string func_closer      = get_marker("function_closer", "</function>");
    std::string param_key_prefix = get_marker("parameter_key_prefix", "<param=");
    std::string param_key_suffix = get_marker("parameter_key_suffix", ">");
    std::string param_closer     = get_marker("parameter_closer", "</param>");

    // Build tool choices for tagged format
    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = function.contains("parameters") ? function.at("parameters") : ordered_json::object();

        // Build argument parsers
        auto args = eps();
        if (params.contains("properties") && !params["properties"].empty()) {
            auto arg_choice = choice();
            for (const auto & el : params["properties"].items()) {
                const std::string & prop_name = el.key();

                auto arg_name_parser =
                    choice({ literal(prop_name), literal("\"" + prop_name + "\""), literal("'" + prop_name + "'") });

                auto arg_rule = tool_arg(tool_arg_open(literal(param_key_prefix)) + tool_arg_name(arg_name_parser) +
                                         literal(param_key_suffix) + tool_arg_value(until(param_closer)) +
                                         tool_arg_close(literal(param_closer)));
                arg_choice |= arg_rule;
            }
            args = zero_or_more(arg_choice + space());
        }

        // Build function parser: <function=name>args</function>
        auto tool_parser = tool(tool_open(literal(func_opener) + tool_name(literal(name)) + literal(func_name_suffix)) +
                                space() + tool_args(args) + space() + tool_close(literal(func_closer)));

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    // Build the section with markers
    auto section =
        parallel_tool_calls ?
            trigger_rule("tool-call", literal(section_start) + space() + one_or_more(tool_choices + space()) +
                                          literal(section_end)) :
            trigger_rule("tool-call", literal(section_start) + space() + tool_choices + space() + literal(section_end));

    return force_tool_calls ? section : optional(section);
}

// Python-style tool calls: name(arg1="value1", arg2=123)
// Used only by LFM2 for now, so we don't merge it into autoparser
common_peg_parser common_chat_peg_builder::python_style_tool_calls(
    const ordered_json & tools,
    bool                 parallel_tool_calls) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = function.contains("parameters") ? function.at("parameters") : ordered_json::object();

        auto args = eps();
        if (params.contains("properties") && !params["properties"].empty()) {
            auto arg_choice = choice();
            for (const auto & el : params["properties"].items()) {
                const std::string & prop_name = el.key();
                const auto & prop_def = el.value();
                bool is_string_type = (prop_def.contains("type") && prop_def["type"] == "string");

                auto arg_name_parser = literal(prop_name);

                common_peg_parser arg_value_parser = eps();
                auto string_value_parser = choice({
                    literal("\"") + tool_arg_string_value(string_content('"')) + literal("\""),
                    literal("'") + tool_arg_string_value(string_content('\'')) + literal("'")
                });

                if (is_string_type) {
                    arg_value_parser = string_value_parser;
                } else {
                    arg_value_parser = tool_arg_value(python_value());
                }

                // Full argument: name="value" or name=value
                auto arg_rule = tool_arg(
                    tool_arg_open(eps()) +
                    tool_arg_name(arg_name_parser) +
                    literal("=") +
                    arg_value_parser +
                    tool_arg_close(eps())
                );
                arg_choice |= arg_rule;
            }

            args = arg_choice + zero_or_more("," + space() + arg_choice);
        }

        auto tool_parser = tool(tool_open(tool_name(literal(name)) + literal("(")) +
            space() + tool_args(args) + space() + tool_close(literal(")"))
        );

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    if (parallel_tool_calls) {
        return "[" + space() + tool_choices + zero_or_more("," + space() + tool_choices) + space() + "]";
    }
    return "[" + space() + tool_choices + space() + "]";
}

// Helper: Parse dot notation key into prefix and field name
static std::pair<std::string, std::string> parse_key_spec(const std::string & key) {
    auto dot_pos = key.find('.');
    if (dot_pos == std::string::npos) {
        return {"", key};  // Top-level field
    }
    return {key.substr(0, dot_pos), key.substr(dot_pos + 1)};
}

// Mode 1: function_is_key — parse {"function_name": {...}}
common_peg_parser common_chat_peg_builder::build_json_tools_function_is_key(
    const ordered_json & tools,
    const std::string &  args_key,
    const std::string &  effective_args_key,
    const std::string &  call_id_key,
    const std::string &  gen_call_id_key) {

    auto tool_choices = choice();

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = function.contains("parameters") ? function.at("parameters") : ordered_json::object();

        // Build inner object fields
        std::vector<common_peg_parser> inner_fields;

        if (!call_id_key.empty()) {
            auto id_parser = atomic(
                literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                literal("\"") + tool_id(string_content('"')) + literal("\"")
            );
            inner_fields.push_back(optional(id_parser + space() + optional(literal(",") + space())));
        }

        if (!gen_call_id_key.empty()) {
            auto gen_id_parser = atomic(
                literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(string_content('"')) + literal("\""),
                    tool_id(json_number())
                })
            );
            inner_fields.push_back(optional(gen_id_parser + space() + optional(literal(",") + space())));
        }

        // Arguments — either wrapped in args_key or parsed directly
        common_peg_parser args_parser = eps();
        if (args_key.empty()) {
            args_parser = tool_args(schema(json(), "tool-" + name + "-schema", params));
        } else {
            args_parser = literal("\"" + effective_args_key + "\"") + space() + literal(":") + space() +
                          tool_args(schema(json(), "tool-" + name + "-schema", params));
        }
        inner_fields.push_back(args_parser);

        // Build inner object parser
        common_peg_parser inner_object = eps();
        if (args_key.empty() && inner_fields.size() == 1) {
            inner_object = inner_fields[0];
        } else {
            inner_object = literal("{") + space();
            for (size_t i = 0; i < inner_fields.size(); i++) {
                inner_object = inner_object + inner_fields[i];
                if (i < inner_fields.size() - 1) {
                    inner_object = inner_object + space();
                }
            }
            inner_object = inner_object + space() + literal("}");
        }

        auto tool_parser = tool(
            tool_open(literal("{")) + space() +
            literal("\"") + tool_name(literal(name)) + literal("\"") +
            space() + literal(":") + space() +
            inner_object +
            space() + tool_close(literal("}"))
        );

        tool_choices |= rule("tool-" + name, tool_parser);
    }

    return tool_choices;
}

// Mode 2: Nested keys (dot notation like "function.name")
common_peg_parser common_chat_peg_builder::build_json_tools_nested_keys(
    const ordered_json & tools,
    const std::string &  effective_name_key,
    const std::string &  effective_args_key,
    const std::string &  call_id_key,
    const std::string &  gen_call_id_key) {

    auto tool_choices = choice();

    auto name_spec = parse_key_spec(effective_name_key);
    auto args_spec = parse_key_spec(effective_args_key);

    std::string nested_prefix     = !name_spec.first.empty() ? name_spec.first  : args_spec.first;
    std::string nested_name_field = !name_spec.first.empty() ? name_spec.second  : effective_name_key;
    std::string nested_args_field = !args_spec.first.empty() ? args_spec.second  : effective_args_key;

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = function.contains("parameters") ? function.at("parameters") : ordered_json::object();

        auto nested_name = literal("\"" + nested_name_field + "\"") + space() + literal(":") + space() +
                          atomic(literal("\"") + tool_name(literal(name)) + literal("\""));
        auto nested_args = literal("\"" + nested_args_field + "\"") + space() + literal(":") + space() +
                          tool_args(schema(json(), "tool-" + name + "-schema", params));

        auto nested_object = literal("{") + space() +
                            nested_name + space() + literal(",") + space() +
                            nested_args +
                            space() + literal("}");

        // Format: { id?, "function": {...} }
        auto tool_parser_body = tool_open(literal("{")) + space();

        if (!call_id_key.empty()) {
            auto id_spec = parse_key_spec(call_id_key);
            if (id_spec.first.empty()) {
                auto id_parser = atomic(
                    literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                    literal("\"") + tool_id(string_content('"')) + literal("\"")
                );
                tool_parser_body = tool_parser_body + optional(id_parser + space() + literal(",") + space());
            }
        }

        if (!gen_call_id_key.empty()) {
            auto gen_id_spec = parse_key_spec(gen_call_id_key);
            if (gen_id_spec.first.empty()) {
                auto gen_id_parser = atomic(
                    literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                    choice({
                        literal("\"") + tool_id(string_content('"')) + literal("\""),
                        tool_id(json_number())
                    })
                );
                tool_parser_body = tool_parser_body + optional(gen_id_parser + space() + literal(",") + space());
            }
        }

        auto nested_field = literal("\"" + nested_prefix + "\"") + space() + literal(":") + space() + nested_object;
        tool_parser_body = tool_parser_body + nested_field + space() + tool_close(literal("}"));

        tool_choices |= rule("tool-" + name, tool(tool_parser_body));
    }

    return tool_choices;
}

// Mode 3: Flat keys with optional ID fields and parameter ordering
common_peg_parser common_chat_peg_builder::build_json_tools_flat_keys(
    const ordered_json &             tools,
    const std::string &              effective_name_key,
    const std::string &              effective_args_key,
    const std::string &              call_id_key,
    const std::string &              gen_call_id_key,
    const std::vector<std::string> & parameters_order) {

    auto tool_choices    = choice();
    auto name_key_parser = literal("\"" + effective_name_key + "\"");
    auto args_key_parser = literal("\"" + effective_args_key + "\"");

    for (const auto & tool_def : tools) {
        if (!tool_def.contains("function")) {
            continue;
        }
        const auto &   function = tool_def.at("function");
        std::string    name     = function.at("name");
        ordered_json   params   = function.contains("parameters") ? function.at("parameters") : ordered_json::object();

        auto tool_name_ = name_key_parser + space() + literal(":") + space() +
                         atomic(literal("\"") + tool_name(literal(name)) + literal("\""));
        auto tool_args_ = args_key_parser + space() + literal(":") + space() +
                         tool_args(schema(json(), "tool-" + name + "-schema", params));

        // Build ID parsers if keys are provided
        common_peg_parser id_parser = eps();
        if (!call_id_key.empty()) {
            id_parser = atomic(
                literal("\"" + call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(string_content('"')) + literal("\""),
                    tool_id(json_number())
                })
            );
        }

        common_peg_parser gen_id_parser = eps();
        if (!gen_call_id_key.empty()) {
            gen_id_parser = atomic(
                literal("\"" + gen_call_id_key + "\"") + space() + literal(":") + space() +
                choice({
                    literal("\"") + tool_id(string_content('"')) + literal("\""),
                    tool_id(json_number())
                })
            );
        }

        // Create (parser, key) pairs for all fields, then sort by parameters_order
        std::vector<std::pair<common_peg_parser, std::string>> parser_pairs;
        parser_pairs.emplace_back(tool_name_, effective_name_key);
        parser_pairs.emplace_back(tool_args_, effective_args_key);
        if (!call_id_key.empty()) {
            parser_pairs.emplace_back(optional(id_parser), call_id_key);
        }
        if (!gen_call_id_key.empty()) {
            parser_pairs.emplace_back(optional(gen_id_parser), gen_call_id_key);
        }

        std::sort(parser_pairs.begin(), parser_pairs.end(),
            [&parameters_order](const auto & a, const auto & b) {
                auto pos_a = std::find(parameters_order.begin(), parameters_order.end(), a.second);
                auto pos_b = std::find(parameters_order.begin(), parameters_order.end(), b.second);
                size_t idx_a = (pos_a == parameters_order.end()) ? parameters_order.size() : std::distance(parameters_order.begin(), pos_a);
                size_t idx_b = (pos_b == parameters_order.end()) ? parameters_order.size() : std::distance(parameters_order.begin(), pos_b);
                return idx_a < idx_b;
            });

        auto ordered_body = tool_open(literal("{")) + space();
        for (size_t i = 0; i < parser_pairs.size(); i++) {
            ordered_body = ordered_body + parser_pairs[i].first;
            if (i < parser_pairs.size() - 1) {
                ordered_body = ordered_body + space() + literal(",") + space();
            }
        }
        ordered_body = ordered_body + space() + tool_close(literal("}"));

        tool_choices |= rule("tool-" + name, tool(ordered_body));
    }

    return tool_choices;
}

common_peg_parser common_chat_peg_builder::prefix(const std::string & s, const std::string & delimiter) {
    if (s.empty()) {
        return eps();
    }
    if (delimiter.empty()) {
        return literal(s);
    }
    return literal(s.substr(0, s.rfind(delimiter)));
}

common_peg_parser common_chat_peg_builder::standard_json_tools(
                                                       const std::string &              section_start,
                                                       const std::string &              section_end,
                                                       const ordered_json &             tools,
                                                       bool                             parallel_tool_calls,
                                                       bool                             force_tool_calls,
                                                       const std::string &              name_key,
                                                       const std::string &              args_key,
                                                       bool                             array_wrapped,
                                                       bool                             function_is_key,
                                                       const std::string &              call_id_key,
                                                       const std::string &              gen_call_id_key,
                                                       const std::vector<std::string> & parameters_order) {
    if (!tools.is_array() || tools.empty()) {
        return eps();
    }

    std::string effective_name_key = name_key.empty() ? "name" : name_key;
    std::string effective_args_key = args_key.empty() ? "arguments" : args_key;

    // Dispatch to the appropriate builder based on the JSON layout mode
    common_peg_parser tool_choices = eps();
    if (function_is_key) {
        tool_choices = build_json_tools_function_is_key(tools, args_key, effective_args_key, call_id_key, gen_call_id_key);
    } else {
        auto name_spec = parse_key_spec(effective_name_key);
        auto args_spec = parse_key_spec(effective_args_key);
        if (!name_spec.first.empty() || !args_spec.first.empty()) {
            tool_choices = build_json_tools_nested_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key);
        } else {
            tool_choices = build_json_tools_flat_keys(tools, effective_name_key, effective_args_key, call_id_key, gen_call_id_key, parameters_order);
        }
    }

    // Build the section with markers
    auto tool_calls = tool_choices;
    if (parallel_tool_calls) {
        tool_calls = tool_calls + zero_or_more(space() + literal(",") + space() + tool_choices);
    }

    if (array_wrapped) {
        tool_calls = literal("[") + space() + tool_calls + space() + literal("]");
    }

    auto section =
        trigger_rule("tool-call", literal(section_start) + space() + tool_calls + space() + literal(section_end));

    return force_tool_calls ? section : optional(section);
}

void common_chat_peg_gemma4_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    for (const auto & node : result.nodes) {
        visit(arena, node);
    }
}

static std::string gemma4_to_json(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.text.empty()) {
        return "";
    }

    if (node.rule == "gemma4-number" || node.rule == "gemma4-bool" || node.rule == "gemma4-null") {
        return std::string(node.text);
    }

    if (node.rule == "gemma4-string-content") {
        return escape_json_string_inner(std::string(node.text));
    }

    if (node.rule == "gemma4-string") {
        std::string result = "\"";
        if (!node.children.empty()) {
            result += gemma4_to_json(arena, node.children[0]);
            if (!node.is_partial) {
                result += "\"";
            }
        }
        return result;
    }

    if (node.rule == "gemma4-array") {
        std::string result = "[";

        bool add_comma = false;
        for (auto child_id : node.children) {
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += gemma4_to_json(arena, child_id);
        }

        if (!node.is_partial) {
            result += ']';
        }
        return result;
    }

    if (node.rule == "gemma4-dict-key-name") {
        return std::string(node.text);
    }

    if (node.rule == "gemma4-dict-key") {
        std::string result = "\"";
        if (!node.children.empty()) {
            result += escape_json_string_inner(gemma4_to_json(arena, node.children[0]));
        }
        if (!node.is_partial) {
            result += "\":";
        }
        return result;
    }

    if (node.rule == "gemma4-dict-kv") {
        std::string result;
        for (auto child_id : node.children) {
            result += gemma4_to_json(arena, child_id);
        }
        return result;
    }

    if (node.rule == "gemma4-dict") {
        std::string result = "{";

        bool add_comma = false;
        for (auto child_id : node.children) {
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += gemma4_to_json(arena, child_id);
        }

        if (!node.is_partial) {
            result += '}';
        }
        return result;
    }

    if (node.rule == "gemma4-value") {
        if (!node.children.empty()) {
            return gemma4_to_json(arena, node.children[0]);
        }
        return "";
    }

    return "";
}

void common_chat_peg_gemma4_mapper::visit(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.tag == "reasoning") {
        result.reasoning_content += std::string(node.text);
        return;
    }

    if (node.tag == "content") {
        result.content += std::string(node.text);
        return;
    }

    if (node.tag == "tool") {
        auto name_id = arena.find_by_tag(node, "tool-name");
        auto args_id = arena.find_by_tag(node, "tool-args");

        if (name_id != COMMON_PEG_INVALID_AST_ID && args_id != COMMON_PEG_INVALID_AST_ID) {
            const auto & name_node = arena.get(name_id);
            const auto & args_node = arena.get(args_id);

            if (!name_node.is_partial) {
                common_chat_tool_call call;
                call.name = std::string(name_node.text);
                if (!args_node.children.empty()) {
                    call.arguments = gemma4_to_json(arena, args_node.children[0]);
                }
                result.tool_calls.push_back(call);
            }
        }

        return;
    }

    for (auto child_id : node.children) {
        visit(arena, child_id);
    }
}

#include "common.h.inc"

#include <nlohmann/json.hpp.inc>

#include <algorithm>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::ordered_json;

static void build_min_max_int(int64_t min_value, int64_t max_value, std::stringstream & out, int decimals_left = 16, bool top_level = true) {
    auto has_min = min_value != std::numeric_limits<int64_t>::min();
    auto has_max = max_value != std::numeric_limits<int64_t>::max();

    auto digit_range = [&](char from, char to) {
        out << "[";
        if (from == to) {
            out << from;
        } else {
            out << from << "-" << to;
        }
        out << "]";
    };
    auto more_digits = [&](int min_digits, int max_digits) {
        out << "[0-9]";
        if (min_digits == max_digits && min_digits == 1) {
            return;
        }
        out << "{";
        out << min_digits;
        if (max_digits != min_digits) {
            out << ",";
            if (max_digits != std::numeric_limits<int>::max()) {
                out << max_digits;
            }
        }
        out << "}";
    };
    std::function<void(const std::string_view &, const std::string_view &)> uniform_range =
        [&](const std::string_view & from, const std::string_view & to) {
            size_t i = 0;
            while (i < from.length() && i < to.length() && from[i] == to[i]) {
                i++;
            }
            if (i > 0) {
                out << "\"" << from.substr(0, i) << "\"";
            }
            if (i < from.length() && i < to.length()) {
                if (i > 0) {
                    out << " ";
                }
                auto sub_len = from.length() - i - 1;
                if (sub_len > 0) {
                    auto from_sub = from.substr(i + 1);
                    auto to_sub = to.substr(i + 1);
                    auto sub_zeros = string_repeat("0", sub_len);
                    auto sub_nines = string_repeat("9", sub_len);

                    auto to_reached = false;
                    out << "(";
                    if (from_sub == sub_zeros) {
                        digit_range(from[i], to[i] - 1);
                        out << " ";
                        more_digits(sub_len, sub_len);
                    } else {
                        out << "[" << from[i] << "] ";
                        out << "(";
                        uniform_range(from_sub, sub_nines);
                        out << ")";
                        if (from[i] < to[i] - 1) {
                            out << " | ";
                            if (to_sub == sub_nines) {
                                digit_range(from[i] + 1, to[i]);
                                to_reached = true;
                            } else {
                                digit_range(from[i] + 1, to[i] - 1);
                            }
                            out << " ";
                            more_digits(sub_len, sub_len);
                        }
                    }
                    if (!to_reached) {
                        out << " | ";
                        digit_range(to[i], to[i]);
                        out << " ";
                        uniform_range(sub_zeros, to_sub);
                    }
                    out << ")";
                } else {
                    out << "[" << from[i] << "-" << to[i] << "]";
                }
            }
        };

    if (has_min && has_max) {
        if (min_value < 0 && max_value < 0) {
            out << "\"-\" (";
            build_min_max_int(-max_value, -min_value, out, decimals_left, /* top_level= */ true);
            out << ")";
            return;
        }

        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(0, -min_value, out, decimals_left, /* top_level= */ true);
            out << ") | ";
            min_value = 0;
        }

        auto min_s = std::to_string(min_value);
        auto max_s = std::to_string(max_value);
        auto min_digits = min_s.length();
        auto max_digits = max_s.length();

        for (auto digits = min_digits; digits < max_digits; digits++) {
            uniform_range(min_s, string_repeat("9", digits));
            min_s = "1" + string_repeat("0", digits);
            out << " | ";
        }
        uniform_range(min_s, max_s);
        return;
    }

    auto less_decimals = std::max(decimals_left - 1, 1);

    if (has_min) {
        if (min_value < 0) {
            out << "\"-\" (";
            build_min_max_int(std::numeric_limits<int64_t>::min(), -min_value, out, decimals_left, /* top_level= */ false);
            out << ") | [0] | [1-9] ";
            more_digits(0, decimals_left - 1);
        } else if (min_value == 0) {
            if (top_level) {
                out << "[0] | [1-9] ";
                more_digits(0, less_decimals);
            } else {
                more_digits(1, decimals_left);
            }
        } else if (min_value <= 9) {
            char c = '0' + min_value;
            auto range_start = top_level ? '1' : '0';
            if (c > range_start) {
                digit_range(range_start, c - 1);
                out << " ";
                more_digits(1, less_decimals);
                out << " | ";
            }
            digit_range(c, '9');
            out << " ";
            more_digits(0, less_decimals);
        } else {
            auto min_s = std::to_string(min_value);
            auto len = min_s.length();
            auto c = min_s[0];

            if (c > '1') {
                digit_range(top_level ? '1' : '0', c - 1);
                out << " ";
                more_digits(len, less_decimals);
                out << " | ";
            }
            digit_range(c, c);
            out << " (";
            build_min_max_int(std::stoll(min_s.substr(1)), std::numeric_limits<int64_t>::max(), out, less_decimals, /* top_level= */ false);
            out << ")";
            if (c < '9') {
                out << " | ";
                digit_range(c + 1, '9');
                out << " ";
                more_digits(len - 1, less_decimals);
            }
        }
        return;
    }

    if (has_max) {
        if (max_value >= 0) {
            if (top_level) {
                out << "\"-\" [1-9] ";
                more_digits(0, less_decimals);
                out << " | ";
            }
            build_min_max_int(0, max_value, out, decimals_left, /* top_level= */ true);
        } else {
            out << "\"-\" (";
            build_min_max_int(-max_value, std::numeric_limits<int64_t>::max(), out, decimals_left, /* top_level= */ false);
            out << ")";
        }
        return;
    }

    throw std::runtime_error("At least one of min_value or max_value must be set");
}

const std::string SPACE_RULE = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
    std::string content;
    std::vector<std::string> deps;
};

static std::unordered_map<std::string, BuiltinRule> PRIMITIVE_RULES = {
    {"boolean", {"(\"true\" | \"false\") space", {}}},
    {"decimal-part", {"[0-9]{1,16}", {}}},
    {"integral-part", {"[0] | [1-9] [0-9]{0,15}", {}}},
    {"number", {"(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)? space", {"integral-part", "decimal-part"}}},
    {"integer", {"(\"-\"? integral-part) space", {"integral-part"}}},
    {"value", {"object | array | string | number | boolean | null", {"object", "array", "string", "number", "boolean", "null"}}},
    {"object", {"\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? \"}\" space", {"string", "value"}}},
    {"array", {"\"[\" space ( value (\",\" space value)* )? \"]\" space", {"value"}}},
    {"uuid", {"\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} \"\\\"\" space", {}}},
    {"char",   {"[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {}}},
    {"string", {"\"\\\"\" char* \"\\\"\" space", {"char"}}},
    {"null", {"\"null\" space", {}}},
};

static std::unordered_map<std::string, BuiltinRule> STRING_FORMAT_RULES = {
    {"date", {"[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {}}},
    {"time", {"([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )", {}}},
    {"date-time", {"date \"T\" time", {"date", "time"}}},
    {"date-string", {"\"\\\"\" date \"\\\"\" space", {"date"}}},
    {"time-string", {"\"\\\"\" time \"\\\"\" space", {"time"}}},
    {"date-time-string", {"\"\\\"\" date-time \"\\\"\" space", {"date-time"}}}
};

static std::regex INVALID_RULE_CHARS_RE("[^a-zA-Z0-9-]+");

static std::unordered_set<char> NON_LITERAL_SET = {'|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?'};
static std::unordered_set<char> ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = {'^', '$', '.', '[', ']', '(', ')', '|', '{', '}', '*', '+', '?'};

std::string gbnf_format_literal(const std::string & literal) {
    return common_rust_string_to_std(llama_common_gbnf_format_literal_rust(
            (const uint8_t *) literal.data(), literal.size()));
}

class common_schema_converter {
private:
    friend class common_schema_info;
    friend std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options);
    std::function<json(const std::string &)> _fetch_json;
    bool _dotall;
    std::map<std::string, std::string> _rules;
    std::unordered_map<std::string, json> _refs;
    std::unordered_set<std::string> _refs_being_resolved;
    std::vector<std::string> _errors;
    std::vector<std::string> _warnings;

    std::string _add_rule(const std::string & name, const std::string & rule) {
        std::string esc_name = regex_replace(name, INVALID_RULE_CHARS_RE, "-");
        if (_rules.find(esc_name) == _rules.end() || _rules[esc_name] == rule) {
            _rules[esc_name] = rule;
            return esc_name;
        }
        int i = 0;
        while (_rules.find(esc_name + std::to_string(i)) != _rules.end() && _rules[esc_name + std::to_string(i)] != rule) {
            i++;
        }
        std::string key = esc_name + std::to_string(i);
        _rules[key] = rule;
        return key;
    }

    std::string _generate_union_rule(const std::string & name, const std::vector<json> & alt_schemas) {
        std::vector<std::string> rules;
        rules.reserve(alt_schemas.size());
        for (size_t i = 0; i < alt_schemas.size(); i++) {
            rules.push_back(visit(alt_schemas[i], name + (name.empty() ? "alternative-" : "-") + std::to_string(i)));
        }
        return string_join(rules, " | ");
    }

    std::string _visit_pattern(const std::string & pattern, const std::string & name) {
        if (!(pattern.front() == '^' && pattern.back() == '$')) {
            _errors.push_back("Pattern must start with '^' and end with '$'");
            return "";
        }
        std::string sub_pattern = pattern.substr(1, pattern.length() - 2);
        std::unordered_map<std::string, std::string> sub_rule_ids;

        size_t i = 0;
        size_t length = sub_pattern.length();

        using literal_or_rule = std::pair<std::string, bool>;
        auto to_rule = [&](const literal_or_rule & ls) {
            auto is_literal = ls.second;
            auto s = ls.first;
            return is_literal ? "\"" + s + "\"" : s;
        };
        std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
            size_t start = i;
            std::vector<literal_or_rule> seq;

            auto get_dot = [&]() {
                std::string rule;
                if (_dotall) {
                    rule = "[\\U00000000-\\U0010FFFF]";
                } else {
                    rule = "[^\\x0A\\x0D]";
                }
                return _add_rule("dot", rule);
            };

            // Joins the sequence, merging consecutive literals together.
            auto join_seq = [&]() {
                std::vector<literal_or_rule> ret;

                std::string literal;
                auto flush_literal = [&]() {
                    if (literal.empty()) {
                        return false;
                    }
                    ret.emplace_back(literal, true);
                    literal.clear();
                    return true;
                };

                for (const auto & item : seq) {
                    auto is_literal = item.second;
                    if (is_literal) {
                        literal += item.first;
                    } else {
                        flush_literal();
                        ret.push_back(item);
                    }
                }
                flush_literal();

                std::vector<std::string> results;
                results.reserve(ret.size());
                for (const auto & item : ret) {
                    results.push_back(to_rule(item));
                }
                return std::make_pair(string_join(results, " "), false);
            };

            while (i < length) {
                char c = sub_pattern[i];
                if (c == '.') {
                    seq.emplace_back(get_dot(), false);
                    i++;
                } else if (c == '(') {
                    i++;
                    if (i < length && sub_pattern[i] == '?') {
                        if (i + 1 < length && sub_pattern[i + 1] == ':') {
                            i += 2; // skip "?:" for non-capturing group, treat as regular group
                        } else {
                            // lookahead/lookbehind (?=, ?!, ?<=, ?<!) - not supported
                            _warnings.push_back("Unsupported pattern syntax");
                            // skip to matching ')' to avoid UB on empty seq
                            int depth = 1;
                            while (i < length && depth > 0) {
                                if (sub_pattern[i] == '\\' && i + 1 < length) {
                                    i += 2; // skip escaped character
                                } else {
                                    if (sub_pattern[i] == '(') depth++;
                                    else if (sub_pattern[i] == ')') depth--;
                                    i++;
                                }
                            }
                            continue;
                        }
                    }
                    seq.emplace_back("(" + to_rule(transform()) + ")", false);
                } else if (c == ')') {
                    i++;
                    if (start > 0 && sub_pattern[start - 1] != '(' && (start < 2 || sub_pattern[start - 2] != '?' || sub_pattern[start - 1] != ':')) {
                        _errors.push_back("Unbalanced parentheses");
                    }
                    return join_seq();
                } else if (c == '[') {
                    std::string square_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != ']') {
                        if (sub_pattern[i] == '\\') {
                            square_brackets += sub_pattern.substr(i, 2);
                            i += 2;
                        } else {
                            square_brackets += sub_pattern[i];
                            i++;
                        }
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced square brackets");
                    }
                    square_brackets += ']';
                    i++;
                    seq.emplace_back(square_brackets, false);
                } else if (c == '|') {
                    seq.emplace_back("|", false);
                    i++;
                } else if (c == '*' || c == '+' || c == '?') {
                    seq.back() = std::make_pair(to_rule(seq.back()) + c, false);
                    i++;
                } else if (c == '{') {
                    std::string curly_brackets = std::string(1, c);
                    i++;
                    while (i < length && sub_pattern[i] != '}') {
                        curly_brackets += sub_pattern[i];
                        i++;
                    }
                    if (i >= length) {
                        _errors.push_back("Unbalanced curly brackets");
                    }
                    curly_brackets += '}';
                    i++;
                    auto nums = string_split(curly_brackets.substr(1, curly_brackets.length() - 2), ",");
                    int min_times = 0;
                    int max_times = std::numeric_limits<int>::max();
                    try {
                        if (nums.size() == 1) {
                            min_times = max_times = std::stoi(nums[0]);
                        } else if (nums.size() != 2) {
                            _errors.push_back("Wrong number of values in curly brackets");
                        } else {
                            if (!nums[0].empty()) {
                                min_times = std::stoi(nums[0]);
                            }
                            if (!nums[1].empty()) {
                                max_times = std::stoi(nums[1]);
                            }
                        }
                    } catch (const std::invalid_argument & e) {
                        _errors.push_back("Invalid number in curly brackets");
                        return std::make_pair("", false);
                    }
                    auto &last = seq.back();
                    auto &sub = last.first;
                    auto sub_is_literal = last.second;

                    if (!sub_is_literal) {
                        std::string & sub_id = sub_rule_ids[sub];
                        if (sub_id.empty()) {
                            sub_id = _add_rule(name + "-" + std::to_string(sub_rule_ids.size()), sub);
                        }
                        sub = sub_id;
                    }
                    const std::string repeated = sub_is_literal ? "\"" + sub + "\"" : sub;
                    seq.back().first = common_rust_string_to_std(llama_common_gbnf_build_repetition_rust(
                            (const uint8_t *) repeated.data(), repeated.size(),
                            min_times, max_times,
                            nullptr, 0));
                    seq.back().second = false;
                } else {
                    std::string literal;
                    auto is_non_literal = [&](char c) {
                        return NON_LITERAL_SET.find(c) != NON_LITERAL_SET.end();
                    };
                    while (i < length) {
                        if (sub_pattern[i] == '\\' && i < length - 1) {
                            char next = sub_pattern[i + 1];
                            if (ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.find(next) != ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS.end()) {
                                i++;
                                literal += sub_pattern[i];
                                i++;
                            } else {
                                literal += sub_pattern.substr(i, 2);
                                i += 2;
                            }
                        } else if (sub_pattern[i] == '"') {
                            literal += "\\\"";
                            i++;
                        } else if (!is_non_literal(sub_pattern[i]) &&
                                (i == length - 1 || literal.empty() || sub_pattern[i + 1] == '.' || !is_non_literal(sub_pattern[i + 1]))) {
                            literal += sub_pattern[i];
                            i++;
                        } else {
                            break;
                        }
                    }
                    if (!literal.empty()) {
                        seq.emplace_back(literal, true);
                    }
                }
            }
            return join_seq();
        };
        return _add_rule(name, "\"\\\"\" (" + to_rule(transform()) + ") \"\\\"\" space");
    }

    /*
        Returns a rule that matches a JSON string that is none of the provided strings

        not_strings({"a"})
            -> ["] ( [a] char+ | [^"a] char* )? ["] space
        not_strings({"and", "also"})
            -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["] space
    */
    std::string _not_strings(const std::vector<std::string> & strings) {

        struct TrieNode {
            std::map<char, TrieNode> children;
            bool is_end_of_string;

            TrieNode() : is_end_of_string(false) {}

            void insert(const std::string & string) {
                auto *node = this;
                for (char c : string) {
                    node = &node->children[c];
                }
                node->is_end_of_string = true;
            }
        };

        TrieNode trie;
        for (const auto & s : strings) {
            trie.insert(s);
        }

        std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
        std::ostringstream out;
        out << "[\"] ( ";
        std::function<void(const TrieNode &)> visit = [&](const TrieNode & node) {
            std::ostringstream rejects;
            auto first = true;
            for (const auto & kv : node.children) {
                rejects << kv.first;
                if (first) {
                    first = false;
                } else {
                    out << " | ";
                }
                out << "[" << kv.first << "]";
                if (!kv.second.children.empty()) {
                    out << " (";
                    visit(kv.second);
                    out << ")";
                } else if (kv.second.is_end_of_string) {
                    out << " " << char_rule << "+";
                }
            }
            if (!node.children.empty()) {
                if (!first) {
                    out << " | ";
                }
                out << "[^\"" << rejects.str() << "] " << char_rule << "*";
            }
        };
        visit(trie);

        out << " )";
        if (!trie.is_end_of_string) {
            out << "?";
        }
        out << " [\"] space";
        return out.str();
    }

    std::string _resolve_ref(const std::string & ref) {
        std::string ref_name = common_rust_string_to_std(
            llama_common_gbnf_ref_name_rust((const uint8_t *) ref.data(), ref.size()));
        if (_rules.find(ref_name) == _rules.end() && _refs_being_resolved.find(ref) == _refs_being_resolved.end()) {
            _refs_being_resolved.insert(ref);
            json resolved = _refs[ref];
            ref_name = visit(resolved, ref_name);
            _refs_being_resolved.erase(ref);
        }
        return ref_name;
    }

    std::string _build_object_rule(
        const std::vector<std::pair<std::string, json>> & properties,
        const std::unordered_set<std::string> & required,
        const std::string & name,
        const json & additional_properties)
    {
        std::vector<std::string> required_props;
        std::vector<std::string> optional_props;
        std::unordered_map<std::string, std::string> prop_kv_rule_names;
        std::vector<std::string> prop_names;
        for (const auto & kv : properties) {
            const auto &prop_name = kv.first;
            const auto &prop_schema = kv.second;

            std::string prop_rule_name = visit(prop_schema, name + (name.empty() ? "" : "-") + prop_name);
            prop_kv_rule_names[prop_name] = _add_rule(
                name + (name.empty() ? "" : "-") + prop_name + "-kv",
                gbnf_format_literal(json(prop_name).dump()) + " space \":\" space " + prop_rule_name
            );
            if (required.find(prop_name) != required.end()) {
                required_props.push_back(prop_name);
            } else {
                optional_props.push_back(prop_name);
            }
            prop_names.push_back(prop_name);
        }
        if ((additional_properties.is_boolean() && additional_properties.get<bool>()) || additional_properties.is_object()) {
            std::string sub_name = name + (name.empty() ? "" : "-") + "additional";
            std::string value_rule =
                additional_properties.is_object() ? visit(additional_properties, sub_name + "-value")
                : _add_primitive("value", PRIMITIVE_RULES.at("value"));

            auto key_rule =
                prop_names.empty() ? _add_primitive("string", PRIMITIVE_RULES.at("string"))
                : _add_rule(sub_name + "-k", _not_strings(prop_names));
            std::string kv_rule = _add_rule(sub_name + "-kv", key_rule + " \":\" space " + value_rule);
            prop_kv_rule_names["*"] = kv_rule;
            optional_props.push_back("*");
        }

        std::string rule = "\"{\" space ";
        for (size_t i = 0; i < required_props.size(); i++) {
            if (i > 0) {
                rule += " \",\" space ";
            }
            rule += prop_kv_rule_names[required_props[i]];
        }

        if (!optional_props.empty()) {
            rule += " (";
            if (!required_props.empty()) {
                rule += " \",\" space ( ";
            }

            std::function<std::string(const std::vector<std::string> &, bool)> get_recursive_refs = [&](const std::vector<std::string> & ks, bool first_is_optional) {
                std::string res;
                if (ks.empty()) {
                    return res;
                }
                const std::string& k = ks[0];
                std::string kv_rule_name = prop_kv_rule_names[k];
                std::string comma_ref = "( \",\" space " + kv_rule_name + " )";
                if (first_is_optional) {
                    res = comma_ref + (k == "*" ? "*" : "?");
                } else {
                    res = kv_rule_name + (k == "*" ? " " + comma_ref + "*" : "");
                }
                if (ks.size() > 1) {
                    res += " " + _add_rule(
                        name + (name.empty() ? "" : "-") + k + "-rest",
                        get_recursive_refs(std::vector<std::string>(ks.begin() + 1, ks.end()), true)
                    );
                }
                return res;
            };

            for (size_t i = 0; i < optional_props.size(); i++) {
                if (i > 0) {
                    rule += " | ";
                }
                rule += get_recursive_refs(std::vector<std::string>(optional_props.begin() + i, optional_props.end()), false);
            }
            if (!required_props.empty()) {
                rule += " )";
            }
            rule += " )?";
        }

        rule += " \"}\" space";

        return rule;
    }

    std::string _add_primitive(const std::string & name, const BuiltinRule & rule) {
        auto n = _add_rule(name, rule.content);
        for (const auto & dep : rule.deps) {
            BuiltinRule dep_rule;
            auto it = PRIMITIVE_RULES.find(dep);
            if (it == PRIMITIVE_RULES.end()) {
                it = STRING_FORMAT_RULES.find(dep);
                if (it == STRING_FORMAT_RULES.end()) {
                    _errors.push_back("Rule " + dep + " not known");
                    continue;
                }
            }
            if (_rules.find(dep) == _rules.end()) {
                _add_primitive(dep, it->second);
            }
        }
        return n;
    }

public:
    common_schema_converter(
        const std::function<json(const std::string &)> & fetch_json,
        bool dotall)
          : _fetch_json(fetch_json), _dotall(dotall)
    {
        _rules["space"] = SPACE_RULE;
    }

    void resolve_refs(json & schema, const std::string & url) {
        /*
        * Resolves all $ref fields in the given schema, fetching any remote schemas,
        * replacing each $ref with absolute reference URL and populates _refs with the
        * respective referenced (sub)schema dictionaries.
        */
        std::function<void(json &)> visit_refs = [&](json & n) {
            if (n.is_array()) {
                for (auto & x : n) {
                    visit_refs(x);
                }
            } else if (n.is_object()) {
                if (n.contains("$ref")) {
                    std::string ref = n["$ref"];
                    if (_refs.find(ref) == _refs.end()) {
                        json target;
                        if (ref.find("https://") == 0) {
                            std::string base_url = ref.substr(0, ref.find('#'));
                            auto it = _refs.find(base_url);
                            if (it != _refs.end()) {
                                target = it->second;
                            } else {
                                // Fetch the referenced schema and resolve its refs
                                auto referenced = _fetch_json(ref);
                                resolve_refs(referenced, base_url);
                                _refs[base_url] = referenced;
                            }
                            if (ref.find('#') == std::string::npos || ref.substr(ref.find('#') + 1).empty()) {
                                return;
                            }
                        } else if (ref.find("#/") == 0) {
                            target = schema;
                            n["$ref"] = url + ref;
                            ref = url + ref;
                        } else {
                            _errors.push_back("Unsupported ref: " + ref);
                            return;
                        }
                        std::string pointer = ref.substr(ref.find('#') + 1);
                        std::vector<std::string> tokens = string_split(pointer, "/");
                        for (size_t i = 1; i < tokens.size(); ++i) {
                            const std::string& sel = tokens[i];
                            if (target.is_object() && target.contains(sel)) {
                                target = target[sel];
                            } else if (target.is_array()) {
                                size_t sel_index;
                                try {
                                    sel_index = std::stoull(sel);
                                } catch (const std::invalid_argument & e) {
                                    sel_index = target.size();
                                }
                                if (sel_index >= target.size()) {
                                    _errors.push_back("Error resolving ref " + ref + ": " + sel + " not in " + target.dump());
                                    return;
                                }
                                target = target[sel_index];
                            } else {
                                _errors.push_back("Error resolving ref " + ref + ": " + sel + " not in " + target.dump());
                                return;
                            }
                        }
                        _refs[ref] = target;
                    }
                } else {
                    for (const auto & kv : n.items()) {
                        visit_refs(kv.value());
                    }
                }
            }
        };

        visit_refs(schema);
    }

    static std::string _generate_constant_rule(const json & value) {
        return gbnf_format_literal(value.dump());
    }

    std::string visit(const json & schema, const std::string & name) {
        json schema_type = schema.contains("type") ? schema["type"] : json();
        std::string schema_format = schema.contains("format") ? schema["format"].get<std::string>() : "";
        std::string rule_name =
                llama_common_gbnf_is_reserved_name_rust((const uint8_t *) name.data(), name.size()) ?
                name + "-" : name.empty() ? "root" : name;

        if (schema.contains("$ref")) {
            return _add_rule(rule_name, _resolve_ref(schema["$ref"]));
        }
        if (schema.contains("oneOf") || schema.contains("anyOf")) {
            std::vector<json> alt_schemas = schema.contains("oneOf") ? schema["oneOf"].get<std::vector<json>>() : schema["anyOf"].get<std::vector<json>>();
            return _add_rule(rule_name, _generate_union_rule(name, alt_schemas));
        }
        if (schema_type.is_array()) {
            std::vector<json> schema_types;
            for (const auto & t : schema_type) {
                json schema_copy(schema);
                schema_copy["type"] = t;
                schema_types.push_back(schema_copy);
            }
            return _add_rule(rule_name, _generate_union_rule(name, schema_types));
        }
        if (schema.contains("const")) {
            return _add_rule(rule_name, _generate_constant_rule(schema["const"]) + " space");
        }
        if (schema.contains("enum")) {
            std::vector<std::string> enum_values;
            for (const auto & v : schema["enum"]) {
                enum_values.push_back(_generate_constant_rule(v));
            }
            return _add_rule(rule_name, "(" + string_join(enum_values, " | ") + ") space");
        }
        if ((schema_type.is_null() || schema_type == "object")
                && (schema.contains("properties") ||
                    (schema.contains("additionalProperties") && schema["additionalProperties"] != true))) {
            std::unordered_set<std::string> required;
            if (schema.contains("required") && schema["required"].is_array()) {
                for (const auto & item : schema["required"]) {
                    if (item.is_string()) {
                        required.insert(item.get<std::string>());
                    }
                }
            }
            std::vector<std::pair<std::string, json>> properties;
            if (schema.contains("properties")) {
                for (const auto & prop : schema["properties"].items()) {
                    properties.emplace_back(prop.key(), prop.value());
                }
            }
            return _add_rule(rule_name,
                _build_object_rule(
                    properties, required, name,
                    schema.contains("additionalProperties") ? schema["additionalProperties"] : json()));
        }
        if ((schema_type.is_null() || schema_type == "object" || schema_type == "string") && schema.contains("allOf")) {
            std::unordered_set<std::string> required;
            std::vector<std::pair<std::string, json>> properties;
            std::map<std::string, size_t> enum_values;
            const std::string& hybrid_name = name;
            std::function<void(const json &, bool)> add_component = [&](const json & comp_schema, bool is_required) {
                if (comp_schema.contains("$ref")) {
                    add_component(_refs[comp_schema["$ref"]], is_required);
                } else if (comp_schema.contains("properties")) {
                    for (const auto & prop : comp_schema["properties"].items()) {
                        properties.emplace_back(prop.key(), prop.value());
                        if (is_required) {
                            required.insert(prop.key());
                        }
                    }
                } else if (comp_schema.contains("enum")) {
                    for (const auto & v : comp_schema["enum"]) {
                        const auto rule = _generate_constant_rule(v);
                        if (enum_values.find(rule) == enum_values.end()) {
                            enum_values[rule] = 0;
                        }
                        enum_values[rule] += 1;
                    }
                } else {
                  // todo warning
                }
            };
            for (const auto & t : schema["allOf"]) {
                if (t.contains("anyOf")) {
                    for (const auto & tt : t["anyOf"]) {
                        add_component(tt, false);
                    }
                } else {
                    add_component(t, true);
                }
            }
            if (!enum_values.empty()) {
                std::vector<std::string> enum_intersection;
                for (const auto & p : enum_values) {
                    if (p.second == schema["allOf"].size()) {
                        enum_intersection.push_back(p.first);
                    }
                }
                if (!enum_intersection.empty()) {
                    return _add_rule(rule_name, "(" + string_join(enum_intersection, " | ") + ") space");
                }
            }
            return _add_rule(rule_name, _build_object_rule(properties, required, hybrid_name, json()));
        }
        if ((schema_type.is_null() || schema_type == "array") && (schema.contains("items") || schema.contains("prefixItems"))) {
            json items = schema.contains("items") ? schema["items"] : schema["prefixItems"];
            if (items.is_array()) {
                std::string rule = "\"[\" space ";
                for (size_t i = 0; i < items.size(); i++) {
                    if (i > 0) {
                        rule += " \",\" space ";
                    }
                    rule += visit(items[i], name + (name.empty() ? "" : "-") + "tuple-" + std::to_string(i));
                }
                rule += " \"]\" space";
                return _add_rule(rule_name, rule);
            }
            std::string item_rule_name = visit(items, name + (name.empty() ? "" : "-") + "item");
            int min_items = schema.contains("minItems") ? schema["minItems"].get<int>() : 0;
            json max_items_json = schema.contains("maxItems") ? schema["maxItems"] : json();
            int max_items = max_items_json.is_number_integer() ? max_items_json.get<int>() : std::numeric_limits<int>::max();

            const std::string separator_rule = "\",\" space";
            return _add_rule(rule_name, "\"[\" space " + common_rust_string_to_std(
                    llama_common_gbnf_build_repetition_rust(
                            (const uint8_t *) item_rule_name.data(), item_rule_name.size(),
                            min_items, max_items,
                            (const uint8_t *) separator_rule.data(), separator_rule.size())) + " \"]\" space");
        }
        if ((schema_type.is_null() || schema_type == "string") && schema.contains("pattern")) {
            return _visit_pattern(schema["pattern"], rule_name);
        }
        if ((schema_type.is_null() || schema_type == "string") && std::regex_match(schema_format, std::regex("^uuid[1-5]?$"))) {
            return _add_primitive(rule_name == "root" ? "root" : schema_format, PRIMITIVE_RULES.at("uuid"));
        }
        if ((schema_type.is_null() || schema_type == "string") && STRING_FORMAT_RULES.find(schema_format + "-string") != STRING_FORMAT_RULES.end()) {
            auto prim_name = schema_format + "-string";
            return _add_rule(rule_name, _add_primitive(prim_name, STRING_FORMAT_RULES.at(prim_name)));
        }
        if (schema_type == "string" && (schema.contains("minLength") || schema.contains("maxLength"))) {
            std::string char_rule = _add_primitive("char", PRIMITIVE_RULES.at("char"));
            int min_len = schema.contains("minLength") ? schema["minLength"].get<int>() : 0;
            int max_len = schema.contains("maxLength") ? schema["maxLength"].get<int>() : std::numeric_limits<int>::max();
            return _add_rule(rule_name, "\"\\\"\" " + common_rust_string_to_std(
                    llama_common_gbnf_build_repetition_rust(
                            (const uint8_t *) char_rule.data(), char_rule.size(),
                            min_len, max_len,
                            nullptr, 0)) + " \"\\\"\" space");
        }
        if (schema_type == "integer" && (schema.contains("minimum") || schema.contains("exclusiveMinimum") || schema.contains("maximum") || schema.contains("exclusiveMaximum"))) {
            int64_t min_value = std::numeric_limits<int64_t>::min();
            int64_t max_value = std::numeric_limits<int64_t>::max();
            if (schema.contains("minimum")) {
                min_value = schema["minimum"].get<int64_t>();
            } else if (schema.contains("exclusiveMinimum")) {
                min_value = schema["exclusiveMinimum"].get<int64_t>() + 1;
            }
            if (schema.contains("maximum")) {
                max_value = schema["maximum"].get<int64_t>();
            } else if (schema.contains("exclusiveMaximum")) {
                max_value = schema["exclusiveMaximum"].get<int64_t>() - 1;
            }
            std::stringstream out;
            out << "(";
            build_min_max_int(min_value, max_value, out);
            out << ") space";
            return _add_rule(rule_name, out.str());
        }
        if (schema.empty() || schema_type == "object") {
            return _add_rule(rule_name, _add_primitive("object", PRIMITIVE_RULES.at("object")));
        }
        if (schema_type.is_null() && schema.is_object()) {
            // No type constraint and no recognized structural keywords (e.g. {"description": "..."}).
            // Per JSON Schema semantics this is equivalent to {} and accepts any value.
            return _add_rule(rule_name, _add_primitive("value", PRIMITIVE_RULES.at("value")));
        }
        if (!schema_type.is_string() || PRIMITIVE_RULES.find(schema_type.get<std::string>()) == PRIMITIVE_RULES.end()) {
            _errors.push_back("Unrecognized schema: " + schema.dump());
            return "";
        }
        // TODO: support minimum, maximum, exclusiveMinimum, exclusiveMaximum at least for zero
        return _add_primitive(rule_name == "root" ? "root" : schema_type.get<std::string>(), PRIMITIVE_RULES.at(schema_type.get<std::string>()));
    }

    void check_errors() {
        if (!_errors.empty()) {
            throw std::invalid_argument("JSON schema conversion failed:\n" + string_join(_errors, "\n"));
        }
        if (!_warnings.empty()) {
            fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n", string_join(_warnings, "; ").c_str());
        }
    }

    std::string format_grammar() {
        std::stringstream ss;
        for (const auto & kv : _rules) {
            ss << kv.first << " ::= " << kv.second << '\n';
        }
        return ss.str();
    }
};

// common_schema_info implementation (pimpl)

common_schema_info::common_schema_info()
    : impl_(std::make_unique<common_schema_converter>(
        [](const std::string &) { return json(); },
        false)) {}

common_schema_info::~common_schema_info() = default;

common_schema_info::common_schema_info(common_schema_info &&) noexcept = default;
common_schema_info & common_schema_info::operator=(common_schema_info &&) noexcept = default;

void common_schema_info::resolve_refs(nlohmann::ordered_json & schema) {
    impl_->resolve_refs(schema, "");
}

// Determines if a JSON schema can resolve to a string type through any path.
// Some models emit raw string values rather than JSON-encoded strings for string parameters.
// If any branch of the schema (via oneOf, anyOf, $ref, etc.) permits a string, this returns
// true, allowing callers to handle the value as a raw string for simplicity.
bool common_schema_info::resolves_to_string(const nlohmann::ordered_json & schema) {
    std::unordered_set<std::string> visited_refs;

    std::function<bool(const json &)> check = [&](const json & s) -> bool {
        if (!s.is_object()) {
            return false;
        }

        // Handle $ref
        if (s.contains("$ref")) {
            const std::string & ref = s["$ref"];
            if (visited_refs.find(ref) != visited_refs.end()) {
                // Circular reference, assume not a string to be safe
                return false;
            }
            visited_refs.insert(ref);
            auto it = impl_->_refs.find(ref);
            if (it != impl_->_refs.end()) {
                return check(it->second);
            }
            return false;
        }

        // Check type field
        if (s.contains("type")) {
            const json & schema_type = s["type"];
            if (schema_type.is_string()) {
                if (schema_type == "string") {
                    return true;
                }
            } else if (schema_type.is_array()) {
                // Type can be an array like ["string", "null"]
                for (const auto & t : schema_type) {
                    if (t == "string") {
                        return true;
                    }
                }
            }
        }

        // Check oneOf/anyOf - if any alternative can be a string
        if (s.contains("oneOf")) {
            for (const auto & alt : s["oneOf"]) {
                if (check(alt)) {
                    return true;
                }
            }
        }
        if (s.contains("anyOf")) {
            for (const auto & alt : s["anyOf"]) {
                if (check(alt)) {
                    return true;
                }
            }
        }

        // Check allOf - all components must be compatible with string type
        if (s.contains("allOf")) {
            bool all_string = true;
            for (const auto & component : s["allOf"]) {
                if (!check(component)) {
                    all_string = false;
                    break;
                }
            }
            if (all_string) {
                return true;
            }
        }

        // Check const - if the constant value is a string
        if (s.contains("const")) {
            if (s["const"].is_string()) {
                return true;
            }
        }

        // Check enum - if any enum value is a string
        if (s.contains("enum")) {
            for (const auto & val : s["enum"]) {
                if (val.is_string()) {
                    return true;
                }
            }
        }

        // String-specific keywords imply string type
        if (s.contains("pattern") || s.contains("minLength") || s.contains("maxLength")) {
            return true;
        }

        // Check format - many formats imply string
        if (s.contains("format")) {
            const std::string & fmt = s["format"];
            if (fmt == "date" || fmt == "time" || fmt == "date-time" ||
                fmt == "uri" || fmt == "email" || fmt == "hostname" ||
                fmt == "ipv4" || fmt == "ipv6" || fmt == "uuid" ||
                fmt.find("uuid") == 0) {
                return true;
            }
        }

        return false;
    };

    return check(schema);
}

std::string json_schema_to_grammar(const json & schema, bool force_gbnf) {
#ifdef LLAMA_USE_LLGUIDANCE
    if (!force_gbnf) {
        return "%llguidance {}\nstart: %json " + schema.dump();
    }
#else
    (void)force_gbnf;
#endif // LLAMA_USE_LLGUIDANCE
    return build_grammar([&](const common_grammar_builder & callbacks) {
        auto copy = schema;
        callbacks.resolve_refs(copy);
        callbacks.add_schema("", copy);
    });
}

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options) {
    common_schema_converter converter([&](const std::string &) { return json(); }, options.dotall);
    common_grammar_builder builder {
        /* .add_rule = */ [&](const std::string & name, const std::string & rule) {
            return converter._add_rule(name, rule);
        },
        /* .add_schema = */ [&](const std::string & name, const nlohmann::ordered_json & schema) {
            return converter.visit(schema, name == "root" ? "" : name);
        },
        /* .resolve_refs = */ [&](nlohmann::ordered_json & schema) {
            converter.resolve_refs(schema, "");
        }
    };
    cb(builder);
    converter.check_errors();
    return converter.format_grammar();
}

#include "common.h.inc"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp.inc>
#include <cpp-httplib/httplib.h.inc>

#include <filesystem>
#include <fstream>
#include <atomic>
#include <regex> // migration only
#include <string>
#include <string_view>
#include <stdexcept>

namespace nl = nlohmann;

extern "C" const char * llama_build_info(void);

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define HOME_DIR "USERPROFILE"
#include <windows.h>
#else
#define HOME_DIR "HOME"
#include <unistd.h>
#include <pwd.h>
#endif

namespace hf_cache {

static std::pair<httplib::Client, common_http_url> hf_cache_http_client(const std::string & url) {
    common_http_url parts = common_http_parse_url(url);

    if (parts.host.empty()) {
        throw std::runtime_error("error: invalid URL format");
    }

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    if (parts.scheme == "https") {
        throw std::runtime_error(
            "HTTPS is not supported. Please rebuild with one of:\n"
            "  -DLLAMA_BUILD_BORINGSSL=ON\n"
            "  -DLLAMA_BUILD_LIBRESSL=ON\n"
            "  -DLLAMA_OPENSSL=ON (default, requires OpenSSL dev files installed)"
        );
    }
#endif

    httplib::Client cli(parts.scheme + "://" + parts.host + ":" + std::to_string(parts.port));

    if (!parts.user.empty()) {
        cli.set_basic_auth(parts.user, parts.password);
    }

    cli.set_follow_location(true);

    return { std::move(cli), std::move(parts) };
}

namespace fs = std::filesystem;

static fs::path get_cache_directory() {
    static const fs::path cache = []() {
        struct {
            const char * var;
            fs::path path;
        } entries[] = {
            {"LLAMA_CACHE",           fs::path()},
            {"HF_HUB_CACHE",          fs::path()},
            {"HUGGINGFACE_HUB_CACHE", fs::path()},
            {"HF_HOME",               fs::path("hub")},
            {"XDG_CACHE_HOME",        fs::path("huggingface") / "hub"},
            {HOME_DIR,                fs::path(".cache") / "huggingface" / "hub"}
        };
        for (const auto & entry : entries) {
            if (auto * p = std::getenv(entry.var); p && *p) {
                fs::path base(p);
                return entry.path.empty() ? base : base / entry.path;
            }
        }
#ifndef _WIN32
        const struct passwd * pw = getpwuid(getuid());

        if (pw && pw->pw_dir && *pw->pw_dir) {
            return fs::path(pw->pw_dir) / ".cache" / "huggingface" / "hub";
        }
#endif
        throw std::runtime_error("Failed to determine HF cache directory");
    }();

    return cache;
}

static fs::path get_repo_path(const std::string & repo_id) {
    return get_cache_directory() / common_rust_string_to_std(
            llama_common_hf_repo_to_folder_name_rust((const uint8_t *) repo_id.data(), repo_id.size()));
}

static bool is_valid_subpath(const fs::path & path, const fs::path & subpath) {
    if (subpath.is_absolute()) {
        return false; // never do a / b with b absolute
    }
    auto b = fs::absolute(path).lexically_normal();
    auto t = (b / subpath).lexically_normal();
    auto [b_end, _] = std::mismatch(b.begin(), b.end(), t.begin(), t.end());

    return b_end == b.end();
}

static void safe_write_file(const fs::path & path, const std::string & data) {
    const std::string path_str = path.string();
    if (!llama_common_safe_write_file_rust(
            (const uint8_t *) path_str.data(), path_str.size(),
            (const uint8_t *) data.data(), data.size())) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
}

static nl::json api_get(const std::string & url,
                        const std::string & token) {
    auto [cli, parts] = hf_cache_http_client(url);

    httplib::Headers headers = {
        {"User-Agent", "llama-cpp/" + std::string(llama_build_info())},
        {"Accept", "application/json"}
    };

    if (llama_common_hf_is_valid_token_rust((const uint8_t *) token.data(), token.size())) {
        headers.emplace("Authorization", "Bearer " + token);
    } else if (!token.empty()) {
        LOG_WRN("%s: invalid token, authentication disabled\n", __func__);
    }

    if (auto res = cli.Get(parts.path, headers)) {
        auto body = res->body;

        if (res->status == 200) {
            return nl::json::parse(res->body);
        }
        try {
            body = nl::json::parse(res->body)["error"].get<std::string>();
        } catch (...) { }

        throw std::runtime_error("GET failed (" + std::to_string(res->status) + "): " + body);
    } else {
        throw std::runtime_error("HTTPLIB failed: " + httplib::to_string(res.error()));
    }
}

static std::string get_repo_commit(const std::string & repo_id,
                                   const std::string & token) {
    try {
        auto endpoint = common_get_model_endpoint();
        auto json = api_get(endpoint + "api/models/" + repo_id + "/refs", token);

        if (!json.is_object() ||
            !json.contains("branches") || !json["branches"].is_array()) {
            LOG_WRN("%s: missing 'branches' for '%s'\n", __func__, repo_id.c_str());
            return {};
        }

        fs::path refs_path = get_repo_path(repo_id) / "refs";
        std::string name;
        std::string commit;

        for (const auto & branch : json["branches"]) {
            if (!branch.is_object() ||
                !branch.contains("name") || !branch["name"].is_string() ||
                !branch.contains("targetCommit") || !branch["targetCommit"].is_string()) {
                continue;
            }
            std::string _name = branch["name"].get<std::string>();
            std::string _commit = branch["targetCommit"].get<std::string>();

            if (!is_valid_subpath(refs_path, _name)) {
                LOG_WRN("%s: skip invalid branch: %s\n", __func__, _name.c_str());
                continue;
            }
            if (!llama_common_hf_is_valid_commit_rust((const uint8_t *) _commit.data(), _commit.size())) {
                LOG_WRN("%s: skip invalid commit: %s\n", __func__, _commit.c_str());
                continue;
            }

            if (_name == "main") {
                name = _name;
                commit = _commit;
                break;
            }

            if (name.empty() || commit.empty()) {
                name = _name;
                commit = _commit;
            }
        }

        if (name.empty() || commit.empty()) {
            LOG_WRN("%s: no valid branch for '%s'\n", __func__, repo_id.c_str());
            return {};
        }

        safe_write_file(refs_path / name, commit);
        return commit;

    } catch (const nl::json::exception & e) {
        LOG_ERR("%s: JSON error: %s\n", __func__, e.what());
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
    }
    return {};
}

hf_files get_repo_files(const std::string & repo_id,
                        const std::string & token) {
    if (!llama_common_hf_is_valid_repo_id_rust((const uint8_t *) repo_id.data(), repo_id.size())) {
        LOG_WRN("%s: invalid repository: %s\n", __func__, repo_id.c_str());
        return {};
    }

    std::string commit = get_repo_commit(repo_id, token);
    if (commit.empty()) {
        LOG_WRN("%s: failed to resolve commit for %s\n", __func__, repo_id.c_str());
        return {};
    }

    fs::path blobs_path = get_repo_path(repo_id) / "blobs";
    fs::path commit_path = get_repo_path(repo_id) / "snapshots" / commit;

    hf_files files;

    try {
        auto endpoint = common_get_model_endpoint();
        auto json = api_get(endpoint + "api/models/" + repo_id + "/tree/" + commit + "?recursive=true", token);

        if (!json.is_array()) {
            LOG_WRN("%s: response is not an array for '%s'\n", __func__, repo_id.c_str());
            return {};
        }

        for (const auto & item : json) {
            if (!item.is_object() ||
                !item.contains("type") || !item["type"].is_string() || item["type"] != "file" ||
                !item.contains("path") || !item["path"].is_string()) {
                continue;
            }

            hf_file file;
            file.repo_id = repo_id;
            file.path = item["path"].get<std::string>();

            if (!is_valid_subpath(commit_path, file.path)) {
                LOG_WRN("%s: skip invalid path: %s\n", __func__, file.path.c_str());
                continue;
            }

            if (item.contains("lfs") && item["lfs"].is_object()) {
                if (item["lfs"].contains("oid") && item["lfs"]["oid"].is_string()) {
                    file.oid = item["lfs"]["oid"].get<std::string>();
                }
                if (item["lfs"].contains("size") && item["lfs"]["size"].is_number()) {
                    file.size = item["lfs"]["size"].get<size_t>();
                }
            } else if (item.contains("oid") && item["oid"].is_string()) {
                file.oid = item["oid"].get<std::string>();
            }
            if (file.size == 0 && item.contains("size") && item["size"].is_number()) {
                file.size = item["size"].get<size_t>();
            }

            if (!file.oid.empty() &&
                !llama_common_hf_is_valid_oid_rust((const uint8_t *) file.oid.data(), file.oid.size())) {
                LOG_WRN("%s: skip invalid oid: %s\n", __func__, file.oid.c_str());
                continue;
            }

            file.url = endpoint + repo_id + "/resolve/" + commit + "/" + file.path;

            fs::path final_path = commit_path / file.path;
            file.final_path = final_path.string();

            if (!file.oid.empty() && !fs::exists(final_path)) {
                fs::path local_path = blobs_path / file.oid;
                file.local_path = local_path.string();
            } else {
                file.local_path = file.final_path;
            }

            files.push_back(file);
        }
    } catch (const nl::json::exception & e) {
        LOG_ERR("%s: JSON error: %s\n", __func__, e.what());
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
    }
    return files;
}

static std::string get_cached_ref(const fs::path & repo_path) {
    const std::string path = repo_path.string();
    return common_rust_string_to_std(
            llama_common_get_cached_ref_rust((const uint8_t *) path.data(), path.size()));
}

hf_files get_cached_files(const std::string & repo_id) {
    fs::path cache_dir = get_cache_directory();
    if (!fs::exists(cache_dir)) {
        return {};
    }

    if (!repo_id.empty() &&
        !llama_common_hf_is_valid_repo_id_rust((const uint8_t *) repo_id.data(), repo_id.size())) {
        LOG_WRN("%s: invalid repository: %s\n", __func__, repo_id.c_str());
        return {};
    }

    hf_files files;

    for (const auto & repo : fs::directory_iterator(cache_dir)) {
        if (!repo.is_directory()) {
            continue;
        }
        fs::path snapshots_path = repo.path() / "snapshots";

        if (!fs::exists(snapshots_path)) {
            continue;
        }
        const std::string folder = repo.path().filename().string();
        std::string _repo_id = common_rust_string_to_std(
                llama_common_hf_folder_name_to_repo_rust((const uint8_t *) folder.data(), folder.size()));

        if (!llama_common_hf_is_valid_repo_id_rust((const uint8_t *) _repo_id.data(), _repo_id.size())) {
            continue;
        }
        if (!repo_id.empty() && _repo_id != repo_id) {
            continue;
        }
        std::string commit = get_cached_ref(repo.path());
        fs::path commit_path = snapshots_path / commit;

        if (commit.empty() || !fs::is_directory(commit_path)) {
            continue;
        }
        for (const auto & entry : fs::recursive_directory_iterator(commit_path)) {
            if (!entry.is_regular_file() && !entry.is_symlink()) {
                continue;
            }
            fs::path path = entry.path().lexically_relative(commit_path);

            if (!path.empty()) {
                hf_file file;
                file.repo_id = _repo_id;
                file.path = path.generic_string();
                file.local_path = entry.path().string();
                file.final_path = file.local_path;
                files.push_back(std::move(file));
            }
        }
    }

    return files;
}

std::string finalize_file(const hf_file & file) {
    static std::atomic<bool> symlinks_disabled{false};

    std::error_code ec;
    fs::path local_path(file.local_path);
    fs::path final_path(file.final_path);

    if (local_path == final_path || fs::exists(final_path, ec)) {
        return file.final_path;
    }

    if (!fs::exists(local_path, ec)) {
        return file.final_path;
    }

    fs::create_directories(final_path.parent_path(), ec);

    if (!symlinks_disabled) {
        fs::path target = fs::relative(local_path, final_path.parent_path(), ec);
        if (!ec) {
            fs::create_symlink(target, final_path, ec);
        }
        if (!ec) {
            return file.final_path;
        }
    }

    if (!symlinks_disabled.exchange(true)) {
        LOG_WRN("%s: failed to create symlink: %s\n", __func__, ec.message().c_str());
        LOG_WRN("%s: switching to degraded mode\n", __func__);
    }

    fs::rename(local_path, final_path, ec);
    if (ec) {
        LOG_WRN("%s: failed to move file to snapshots: %s\n", __func__, ec.message().c_str());
        fs::copy(local_path, final_path, ec);
        if (ec) {
            LOG_ERR("%s: failed to copy file to snapshots: %s\n", __func__, ec.message().c_str());
        }
    }
    return file.final_path;
}

// delete everything after this line, one day

// copied from the download helpers without the tag part
struct gguf_split_info {
    std::string prefix; // tag included
    int index;
    int count;
};

static gguf_split_info hf_cache_get_gguf_split_info(const std::string & path) {
    common_rust_gguf_split_info split =
        llama_common_gguf_split_info_rust((const uint8_t *) path.data(), path.size(), false);
    gguf_split_info result {
        common_rust_string_to_std(split.prefix),
        split.index,
        split.count,
    };
    llama_common_unicode_string_free(split.tag);
    return result;
}

static std::pair<std::string, std::string> parse_manifest_name(std::string & filename) {
    common_rust_string_list parsed =
        llama_common_hf_parse_manifest_name_rust((const uint8_t *) filename.data(), filename.size());
    if (!parsed.data || parsed.len != 2) {
        llama_common_string_list_free(parsed);
        return {};
    }
    std::pair<std::string, std::string> result = {
        std::string((const char *) parsed.data[0].data, parsed.data[0].len),
        std::string((const char *) parsed.data[1].data, parsed.data[1].len),
    };
    llama_common_string_list_free(parsed);
    return result;
}

struct migrate_file {
    std::string path;
    std::string sha256;
    size_t size;
    fs::path old_path;
    fs::path etag_path;
    const hf_file * file;
};

using migrate_files = std::vector<migrate_file>;

static bool collect_file(const fs::path    & old_cache,
                         const std::string & owner,
                         const std::string & repo,
                         const std::string & path,
                         const std::string & sha256,
                         const hf_files    & files,
                         migrate_files     & to_migrate) {

    const hf_file * file = nullptr;

    for (const auto & f : files) {
        if (f.path == path) {
            file = &f;
            break;
        }
    }

    std::string old_filename = common_rust_string_to_std(llama_common_hf_make_old_cache_filename_rust(
            (const uint8_t *) owner.data(), owner.size(),
            (const uint8_t *) repo.data(), repo.size(),
            (const uint8_t *) path.data(), path.size()));
    fs::path old_path = old_cache / old_filename;
    fs::path etag_path = old_path.string() + ".etag";

    if (!fs::exists(old_path)) {
        if (file && fs::exists(file->final_path)) {
            return true;
        }
        LOG_WRN("%s: %s not found in old cache or HF cache\n", __func__, old_filename.c_str());
        return false;
    }

    if (!file) {
        LOG_WRN("%s: %s not found in current repo\n", __func__, old_filename.c_str());
        return false;
    }

    if (!sha256.empty() && !file->oid.empty() && sha256 != file->oid) {
        LOG_WRN("%s: %s is not up to date (sha256 mismatch)\n", __func__, old_filename.c_str());
        return false;
    }

    if (file->size > 0) {
        size_t size = fs::file_size(old_path);
        if (size != file->size) {
            LOG_WRN("%s: %s has wrong size %zu (expected %zu)\n", __func__, old_filename.c_str(), size, file->size);
            return false;
        }
    }

    to_migrate.push_back({path, sha256, file->size, old_path, etag_path, file});
    return true;
}

static bool collect_files(const fs::path    & old_cache,
                          const std::string & owner,
                          const std::string & repo,
                          const nl::json    & node,
                          const hf_files    & files,
                          migrate_files     & to_migrate) {

    if (!node.contains("rfilename") ||
        !node.contains("lfs")       ||
        !node["lfs"].contains("sha256")) {
        return true;
    }

    std::string path = node["rfilename"];
    std::string sha256 = node["lfs"]["sha256"];

    auto split = hf_cache_get_gguf_split_info(path);

    if (split.count <= 1) {
        return collect_file(old_cache, owner, repo, path, sha256, files, to_migrate);
    }

    std::vector<std::pair<std::string, std::string>> splits;

    for (const auto & f : files) {
        auto split_f = hf_cache_get_gguf_split_info(f.path);
        if (split_f.count == split.count && split_f.prefix == split.prefix) {
            // sadly the manifest only provides the sha256 of the first file (index == 1)
            // the rest will be verified using the size...
            std::string f_sha256 = (split_f.index == 1) ? sha256 : "";
            splits.emplace_back(f.path, f_sha256);
        }
    }

    if ((int)splits.size() != split.count) {
        LOG_WRN("%s: expected %d split files but found %d in repo\n", __func__, split.count, (int)splits.size());
        return false;
    }

    for (const auto & [f_path, f_sha256] : splits) {
        if (!collect_file(old_cache, owner, repo, f_path, f_sha256, files, to_migrate)) {
            return false;
        }
    }

    return true;
}

static bool migrate_file(const migrate_file & file) {
    std::error_code ec;

    fs::path new_path(file.file->local_path);
    fs::create_directories(new_path.parent_path(), ec);

    if (!fs::exists(new_path, ec)) {
        fs::rename(file.old_path, new_path, ec);
        if (ec) {
            fs::copy_file(file.old_path, new_path, ec);
            if (ec) {
                LOG_ERR("%s: failed to move/copy %s: %s\n", __func__, file.old_path.string().c_str(), ec.message().c_str());
                return false;
            }
        }
        fs::remove(file.old_path, ec);
    }
    fs::remove(file.etag_path, ec);

    std::string filename = finalize_file(*file.file);
    LOG_INF("%s: migrated %s -> %s\n", __func__, file.old_path.filename().string().c_str(), filename.c_str());
    return true;
}

void migrate_old_cache_to_hf_cache(const std::string & token, bool offline) {
    fs::path old_cache = fs_get_cache_directory();
    if (!fs::exists(old_cache)) {
        return;
    }

    if (offline) {
        LOG_WRN("%s: skipping migration in offline mode (will run when online)\n", __func__);
        return; // -hf is not going to work
    }

    bool warned = false;

    for (const auto & entry : fs::directory_iterator(old_cache)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        auto filename = entry.path().filename().string();
        auto [owner, repo] = parse_manifest_name(filename);

        if (owner.empty() || repo.empty()) {
            continue;
        }

        if (!warned) {
            warned = true;
            LOG_WRN("================================================================================\n"
                    "WARNING: Migrating cache to HuggingFace cache directory\n"
                    "  Old cache: %s\n"
                    "  New cache: %s\n"
                    "This one-time migration moves models previously downloaded with -hf\n"
                    "from the legacy llama.cpp cache to the standard HuggingFace cache.\n"
                    "Models downloaded with --model-url are not affected.\n"
                    "================================================================================\n",
                    old_cache.string().c_str(), get_cache_directory().string().c_str());
        }

        auto repo_id = owner + "/" + repo;
        auto files = get_repo_files(repo_id, token);

        if (files.empty()) {
            LOG_WRN("%s: could not get repo files for %s, skipping\n", __func__, repo_id.c_str());
            continue;
        }

        migrate_files to_migrate;
        bool ok = true;

        try {
            std::ifstream manifest(entry.path());
            auto json = nl::json::parse(manifest);
            for (const char * key : {"ggufFile", "mmprojFile"}) {
                if (json.contains(key)) {
                    if (!collect_files(old_cache, owner, repo, json[key], files, to_migrate)) {
                        ok = false;
                        break;
                    }
                }
            }
        } catch (const std::exception & e) {
            LOG_WRN("%s: failed to parse manifest %s: %s\n", __func__, filename.c_str(), e.what());
            continue;
        }

        if (!ok) {
            LOG_WRN("%s: migration skipped: one or more files failed validation\n", __func__);
            continue;
        }

        for (const auto & file : to_migrate) {
            if (!migrate_file(file)) {
                ok = false;
                break;
            }
        }

        if (!ok) {
            LOG_WRN("%s: migration failed: could not migrate all files\n", __func__);
            continue;
        }

        LOG_INF("%s: migration complete, deleting manifest: %s\n", __func__, entry.path().string().c_str());
        fs::remove(entry.path());
    }
}

} // namespace hf_cache

#include "common.h.inc"

#include <cctype>
#include <functional>
#include <map>
#include <string>
#include <vector>


#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5
#define LLAMA_NGRAM_MIN    1
#define LLAMA_NGRAM_MAX    4

extern "C" {
void * common_ngram_cache_rust_new();
void * common_ngram_cache_rust_load(const char * path);
void   common_ngram_cache_rust_free(void * ptr);
void   common_ngram_cache_rust_update(
        void *        ptr,
        size_t        ngram_min,
        size_t        ngram_max,
        const int32_t * input,
        size_t        input_len,
        size_t        nnew);
size_t common_ngram_cache_rust_draft(
        const void *  context,
        const void *  dynamic,
        const void *  static_cache,
        const int32_t * input,
        size_t        input_len,
        size_t        n_draft,
        size_t        ngram_min,
        size_t        ngram_max,
        int32_t *     out,
        size_t        out_len);
void * common_ngram_map_rust_new(uint16_t size_key, uint16_t size_value, bool key_only, uint16_t min_hits);
void   common_ngram_map_rust_free(void * ptr);
uint16_t common_ngram_map_rust_size_value(const void * ptr);
void   common_ngram_map_rust_begin(void * ptr, const int32_t * tokens, size_t tokens_len);
size_t common_ngram_map_rust_draft(
        void * ptr,
        const int32_t * input,
        size_t input_len,
        int32_t sampled,
        int32_t * out,
        size_t out_len);
void   common_ngram_map_rust_accept(void * ptr, uint16_t n_accepted);
size_t common_ngram_simple_rust_draft(
        uint16_t size_ngram,
        uint16_t size_mgram,
        const int32_t * tokens,
        size_t tokens_len,
        int32_t sampled,
        int32_t * out,
        size_t out_len);
}

struct common_ngram_simple_config {
    uint16_t size_ngram;
    uint16_t size_mgram;
};

static llama_tokens common_ngram_simple_draft(
        const common_ngram_simple_config & config,
        const llama_tokens & tokens,
        llama_token sampled) {
    static_assert(sizeof(llama_token) == sizeof(int32_t));
    llama_tokens out(config.size_mgram);
    const size_t drafted = common_ngram_simple_rust_draft(
            config.size_ngram,
            config.size_mgram,
            reinterpret_cast<const int32_t *>(tokens.data()),
            tokens.size(),
            sampled,
            reinterpret_cast<int32_t *>(out.data()),
            out.size());
    out.resize(drafted);
    return out;
}

struct common_ngram_map {
    common_ngram_map(uint16_t size_key, uint16_t size_value, bool key_only, uint16_t min_hits)
        : impl(common_ngram_map_rust_new(size_key, size_value, key_only, min_hits)) {}

    ~common_ngram_map() { common_ngram_map_rust_free(impl); }

    common_ngram_map(const common_ngram_map &) = delete;
    common_ngram_map & operator=(const common_ngram_map &) = delete;

    common_ngram_map(common_ngram_map && other) noexcept : impl(other.impl) {
        other.impl = nullptr;
    }

    common_ngram_map & operator=(common_ngram_map && other) noexcept {
        if (this != &other) {
            common_ngram_map_rust_free(impl);
            impl = other.impl;
            other.impl = nullptr;
        }
        return *this;
    }

    uint16_t size_value() const {
        return common_ngram_map_rust_size_value(impl);
    }

    void begin(const llama_tokens & tokens) {
        static_assert(sizeof(llama_token) == sizeof(int32_t));
        common_ngram_map_rust_begin(
                impl,
                reinterpret_cast<const int32_t *>(tokens.data()),
                tokens.size());
    }

    void draft(const llama_tokens & input, llama_token sampled, llama_tokens & out) {
        static_assert(sizeof(llama_token) == sizeof(int32_t));
        std::vector<llama_token> buf(size_value());
        const size_t drafted = common_ngram_map_rust_draft(
                impl,
                reinterpret_cast<const int32_t *>(input.data()),
                input.size(),
                sampled,
                reinterpret_cast<int32_t *>(buf.data()),
                buf.size());
        out.assign(buf.begin(), buf.begin() + drafted);
    }

    void accept(uint16_t n_accepted) {
        common_ngram_map_rust_accept(impl, n_accepted);
    }

private:
    void * impl = nullptr;
};

struct common_ngram_cache {
    common_ngram_cache() : impl(common_ngram_cache_rust_new()) {}
    explicit common_ngram_cache(void * ptr) : impl(ptr) {}
    ~common_ngram_cache() { common_ngram_cache_rust_free(impl); }

    common_ngram_cache(const common_ngram_cache &) = delete;
    common_ngram_cache & operator=(const common_ngram_cache &) = delete;

    common_ngram_cache(common_ngram_cache && other) noexcept : impl(other.impl) {
        other.impl = nullptr;
    }

    common_ngram_cache & operator=(common_ngram_cache && other) noexcept {
        if (this != &other) {
            common_ngram_cache_rust_free(impl);
            impl = other.impl;
            other.impl = nullptr;
        }
        return *this;
    }

    void * get() const { return impl; }

private:
    void * impl = nullptr;
};

static common_ngram_cache common_ngram_cache_load(const std::string & filename) {
    void * ptr = common_ngram_cache_rust_load(filename.c_str());
    if (ptr == nullptr) {
        throw std::runtime_error("Unable to open file " + filename);
    }
    return common_ngram_cache(ptr);
}

static void common_ngram_cache_update(
        common_ngram_cache & ngram_cache,
        int                  ngram_min,
        int                  ngram_max,
        std::vector<llama_token> & inp_data,
        int                  nnew,
        bool) {
    static_assert(sizeof(llama_token) == sizeof(int32_t));
    common_ngram_cache_rust_update(
            ngram_cache.get(),
            ngram_min,
            ngram_max,
            reinterpret_cast<const int32_t *>(inp_data.data()),
            inp_data.size(),
            nnew);
}

static void common_ngram_cache_draft(
        std::vector<llama_token> & inp,
        std::vector<llama_token> & draft,
        int                        n_draft,
        int                        ngram_min,
        int                        ngram_max,
        common_ngram_cache &       nc_context,
        common_ngram_cache &       nc_dynamic,
        common_ngram_cache &       nc_static) {
    GGML_ASSERT(draft.size() == 1);
    static_assert(sizeof(llama_token) == sizeof(int32_t));

    std::vector<llama_token> out(n_draft);
    const size_t drafted = common_ngram_cache_rust_draft(
            nc_context.get(),
            nc_dynamic.get(),
            nc_static.get(),
            reinterpret_cast<const int32_t *>(inp.data()),
            inp.size(),
            n_draft,
            ngram_min,
            ngram_max,
            reinterpret_cast<int32_t *>(out.data()),
            out.size());

    for (size_t i = 0; i < drafted; ++i) {
        LOG_DBG(" - draft candidate: token=%d\n", out[i]);
        draft.push_back(out[i]);
    }
}

extern "C" {
void * common_ngram_mod_rust_new(uint16_t n, size_t size);
void   common_ngram_mod_rust_free(void * ptr);
size_t common_ngram_mod_rust_idx(const void * ptr, const int32_t * tokens);
void   common_ngram_mod_rust_add(void * ptr, const int32_t * tokens);
int32_t common_ngram_mod_rust_get(const void * ptr, const int32_t * tokens);
void   common_ngram_mod_rust_reset(void * ptr);
size_t common_ngram_mod_rust_get_n(const void * ptr);
size_t common_ngram_mod_rust_get_used(const void * ptr);
size_t common_ngram_mod_rust_size(const void * ptr);
size_t common_ngram_mod_rust_size_bytes(const void * ptr);
}

struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size) : impl(common_ngram_mod_rust_new(n, size)) {}
    ~common_ngram_mod() { common_ngram_mod_rust_free(impl); }

    size_t  idx(const entry_t * tokens) const { return common_ngram_mod_rust_idx(impl, tokens); }
    void    add(const entry_t * tokens)       { common_ngram_mod_rust_add(impl, tokens); }
    entry_t get(const entry_t * tokens) const { return common_ngram_mod_rust_get(impl, tokens); }

    void reset() { common_ngram_mod_rust_reset(impl); }

    size_t get_n()    const { return common_ngram_mod_rust_get_n(impl); }
    size_t get_used() const { return common_ngram_mod_rust_get_used(impl); }

    size_t size()       const { return common_ngram_mod_rust_size(impl); }
    size_t size_bytes() const { return common_ngram_mod_rust_size_bytes(impl); }

private:
    void * impl;
};

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void accept(uint16_t n_accepted) = 0;

    virtual int32_t n_max(const common_params_speculative & params) const = 0;
    virtual int32_t n_min(const common_params_speculative & params) const = 0;
};

struct common_speculative_checkpoint {
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;

    int64_t   n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    bool use_ckpt = false;
    common_speculative_checkpoint ckpt;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements,
            bool use_ckpt)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , use_ckpt(use_ckpt)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & /*prompt*/) override {
    }

    size_t create_checkpoint(int n_tokens_prompt) {
        int slot_id = 0;
        const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx_dft, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        ckpt.pos_min  = llama_memory_seq_pos_min(llama_get_memory(ctx_dft), slot_id);
        ckpt.pos_max  = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot_id);
        ckpt.n_tokens = n_tokens_prompt;
        ckpt.data.resize(checkpoint_size);

        const size_t n = llama_state_seq_get_data_ext(ctx_dft, ckpt.data.data(), checkpoint_size, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != checkpoint_size) {
            GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
        }

        LOG_DBG("%s: pos_min = %d, pos_max = %d, size = %.3f MiB\n", __func__,
                ckpt.pos_min, ckpt.pos_max, (float) ckpt.data.size() / 1024 / 1024);
        return n;
    }

    size_t restore_checkpoint() {
        int slot_id = 0;
        LOG_DBG("%s: pos_min = %d, pos_max = %d\n", __func__, ckpt.pos_min, ckpt.pos_max);
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, ckpt.data.data(), ckpt.size(), slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != ckpt.size()) {
            GGML_ABORT("%s: failed to restore context checkpoint (pos_min=%d, pos_max=%d, size=%zu",
                        __func__, ckpt.pos_min, ckpt.pos_max, ckpt.size());
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, ckpt.pos_max + 1, -1);

        return n;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        const auto & sparams = params.draft;

        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0; // index of part to be reused in prompt_dft
        int reuse_n = 0; // length of part to be reused in prompt_dft

        const int n_ctx = llama_n_ctx(ctx_dft) - sparams.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        if (use_ckpt && i_start > 0) {
            LOG_WRN("%s: context shift is not supported with checkpoint-based contexts - skipping\n", __func__);
            return;
        }

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                   i       + cur < (int) prompt_dft.size() &&
                   prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }

            if (use_ckpt) {
                break;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, #prompt_dft = %zu, #prompt_cur = %zu\n",
                __func__, reuse_i, reuse_n, prompt_dft.size(), prompt_cur.size());
        if (use_ckpt && ckpt.n_tokens > reuse_n) {
            LOG_DBG("%s: checkpoint (n_tokens = %d) is outdated -> delete it\n", __func__, (int) ckpt.n_tokens);

            reuse_i = 0;
            reuse_n = 0;

            ckpt = {};
        }

        result.clear();
        result.reserve(sparams.n_max);

        if (reuse_n == 0 || (use_ckpt && reuse_i > 0)) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int64_t) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (sparams.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                GGML_ASSERT(!use_ckpt);

                bool is_removed = llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                if (!is_removed) {
                    LOG_ERR("%s: llama_memory_seq_rm failed, reuse_i=%d\n", __func__, reuse_i);
                    return;
                }
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                if (use_ckpt) {
                    if (ckpt.n_tokens > 0) {
                        LOG_DBG("%s: restoring checkpoint, reuse_n=%d, prompt_dft.size=%zu\n", __func__, reuse_n, prompt_dft.size());
                        restore_checkpoint();
                        reuse_n = ckpt.n_tokens;
                        prompt_dft.resize(reuse_n);
                    }
                } else {
                    const bool is_removed = llama_memory_seq_rm(mem_dft, 0, reuse_n, -1);
                    if (!is_removed) {
                        LOG_ERR("%s: llama_memory_seq_rm failed, reuse_n=%d, prompt_dft.size=%zu\n", __func__, reuse_n, prompt_dft.size());
                        return;
                    }
                    prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
                }
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());
            LOG_DBG("%s: draft prompt batch: %d tokens\n", __func__, batch.n_tokens);

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0 && ret != 1) {
                LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu\n",
                        __func__, ret, prompt_cur.size());
            }

            if (use_ckpt) {
                create_checkpoint(prompt_dft.size());
            }
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        //LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0 && ret != 1) {
            LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                    __func__, ret, prompt_cur.size(), prompt_dft.size());
        }

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < sparams.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < sparams.p_min) {
                break;
            }

            result.push_back(id);

            if (sparams.n_max <= (int) result.size()) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                        __func__, i, ret, prompt_cur.size(), prompt_dft.size());
            }

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t) sparams.n_max) {
                result.resize(sparams.n_max);
            }
        }

        if (result.size() < (size_t) sparams.n_min) {
            result.clear();
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map config;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map config)
        : common_speculative_state(type), config(std::move(config)) {}

    void begin(const llama_tokens & prompt) override {
        config.begin(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        config.draft(prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        config.accept(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_value();
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_value();
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        const auto & sparams = params.ngram_mod;

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + sparams.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < sparams.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < sparams.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);
                    }

                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.ngram_mod.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.ngram_mod.n_min;
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return n_draft;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return 0;
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states

    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map make_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h.inc to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    const char * name = llama_common_speculative_type_to_str_rust((int) type);
    return name ? name : "unknown";
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const int parsed = llama_common_speculative_type_from_name_rust(name.c_str());
    if (parsed < 0) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return (common_speculative_type) parsed;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    if (params.draft.model) {
        ctx_dft = llama_init_from_model(params.draft.model, params.draft.cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.draft.mparams.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            auto & sparams = params.ngram_mod;

            if (!sparams.obj) {
                sparams.obj = std::make_shared<common_ngram_mod>(sparams.n_match, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                        sparams.n_match, sparams.obj->size(), (float)(sparams.obj->size_bytes())/1024/1024);

                if (sparams.n_match < 16) {
                    LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                            "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, sparams.n_match);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                const bool use_ckpt = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.draft.replacements,
                    /* .use_ckpt     = */ use_ckpt
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ config.params.ngram_simple.size_n,
                    /* .size_mgram = */ config.params.ngram_simple.size_m
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type  = */ config.type,
                    /* .state = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    make_common_ngram_map(config.type, config.params.ngram_map_k)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod.obj);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod.obj));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(params.ngram_cache.lookup_cache_static, params.ngram_cache.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(std::move(state)));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls     = */ std::move(impls),
        /* .curr_impl = */ nullptr,
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        {
            const int n_min = impl->n_min(params);

            if (!result.empty() && (int) result.size() < n_min) {
                LOG_DBG("%s: ignoring small draft: %d < %d\n", __func__, (int) result.size(), n_min);
                result.clear();
            }
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // we have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_max = 0;
    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, impl->n_max(params));
    }

    return n_max;
}

int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_min = 0;
    for (const auto & impl : spec->impls) {
        n_min = std::max(n_min, impl->n_min(params));
    }

    return n_min;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
