/* Inlined openvino/utils.h */
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <utility>

#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

std::string getCurrentTime();

void dump_ov_model(std::shared_ptr<ov::Model> model);

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs);

int non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb);

template <typename T>
std::vector<int> argsort_descend(const std::vector<T>& v) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {
        return v[i1] > v[i2];
    });
    return idx;
}

template <typename T>
std::vector<T> sorted_descend(std::vector<T> v) {
    std::sort(v.begin(), v.end(), [](T a, T b) {
        return a > b;
    });
    return v;
}

template <typename T>
bool is_permuted(const std::vector<T>& strides) {
    for (size_t i = 0; i < strides.size() - 1; ++i) {
        if (strides[i] < strides[i + 1]) {
            return true;
        }
    }
    return false;
}

template <typename T>
std::vector<T> permute(const std::vector<T>& x, const std::vector<int>& perm) {
    std::vector<T> result;
    result.reserve(perm.size());
    for (int i : perm) {
        result.push_back(x[i]);
    }
    return result;
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims);
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);

OutputVector rename_outputs_with_suffix(const OutputVector& outputs, const std::string& suffix);

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t* rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight = nullptr,
                                                           bool imrope = false,
                                                           bool stateful = false);

ov::Output<ov::Node> process_view_input(const NodeContext& context, int input_index, int slice_len = 0);

namespace op {
template <typename T>
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto res = std::make_shared<T>(context.get_input(0), context.get_input(1));
    return rename_outputs_with_suffix({res}, context.get_name());
}
}  // namespace op

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


#include "ggml-openvino.h.inc"

#include "ggml-backend-impl-defs.inc"
#include "ggml-backend.h.inc"
#include "ggml-impl-defs.inc"
/* Inlined ggml-openvino-extra.h */
#include "ggml.h.inc"
#include "openvino/runtime/core.hpp"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <cstdlib>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/runtime/remote_context.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <string>

// ExtraQuantType enum - defines requantization target formats
enum class ExtraQuantType { F16, Q4_0_C, Q8_1_C, Q4_0_128, Q8_0_C, Q8_0_32 };

ov::Core & ov_singleton_core();

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context();

// Get the compile config for the current device
const ov::AnyMap & ggml_openvino_get_compile_config();

// Get the OpenCL command queue for GPU operations (returns nullptr for CPU/NPU)
cl_command_queue ggml_openvino_get_cl_queue();

// Intel USM extension function type
typedef cl_int(CL_API_CALL * clEnqueueMemFillINTEL_fn)(cl_command_queue queue,
                                                       void * dst_ptr,
                                                       const void * pattern,
                                                       size_t pattern_size,
                                                       size_t size,
                                                       cl_uint num_events_in_wait_list,
                                                       const cl_event * event_wait_list,
                                                       cl_event * event);

typedef cl_int(CL_API_CALL * clEnqueueMemcpyINTEL_fn)(cl_command_queue queue,
                                                      cl_bool blocking,
                                                      void * dst_ptr,
                                                      const void * src_ptr,
                                                      size_t size,
                                                      cl_uint num_events_in_wait_list,
                                                      const cl_event * event_wait_list,
                                                      cl_event * event);

// Get the clEnqueueMemFillINTEL function pointer (returns nullptr if not available)
clEnqueueMemFillINTEL_fn ggml_openvino_get_clEnqueueMemFillINTEL();

// Get the clEnqueueMemcpyINTEL function pointer (returns nullptr if not available)
clEnqueueMemcpyINTEL_fn ggml_openvino_get_clEnqueueMemcpyINTEL();

// =====================================================
// Global Device Configuration (singleton)
// =====================================================
// Initialized once during backend init from GGML_OPENVINO_DEVICE env var

struct ggml_openvino_device_config {
    std::string device_name = "CPU";
    bool is_npu = false;
    bool initialized = false;
    std::optional<ov::RemoteContext> remote_context;
    ov::AnyMap compile_config;
    cl_command_queue cl_queue = nullptr;

    void init();
    ~ggml_openvino_device_config();
};

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config();

// Initialize device config (call during backend init)
void ggml_openvino_init_device_config();

// Get the device name
const std::string & ggml_openvino_get_device_name();

// Check if running on NPU
bool ggml_openvino_is_npu();

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(const ggml_tensor * tensor, bool no_requant = false);

// =====================================================
// OpenVINO Tensor Extra Types
// =====================================================
// These types are stored in tensor->extra by the OpenVINO backend buffer.
// They allow:
// 1. Pre-built ov::Constant nodes for weights (avoiding memcpy during graph construction)
// 2. ov::Tensor wrappers for KV cache / compute tensors (for direct use with infer_request)

// Base class for OpenVINO tensor extra data
struct ggml_openvino_extra_base {
    enum class Type { WEIGHT, QUANTIZED_WEIGHT, TENSOR };
    Type type;
    virtual ~ggml_openvino_extra_base() = default;
protected:
    explicit ggml_openvino_extra_base(Type t) : type(t) {}
};

// Extra data for F16/F32/BF16 weight tensors - stores the pre-built weight node
struct ggml_openvino_weight_extra : public ggml_openvino_extra_base {
    ov::Tensor weights;                     // The underlying weight data tensor
    std::shared_ptr<ov::Node> weight_node;  // Pre-built OpenVINO weight node

    ggml_openvino_weight_extra(ov::Tensor w, std::shared_ptr<ov::Node> n) :
        ggml_openvino_extra_base(Type::WEIGHT),
        weights(std::move(w)),
        weight_node(std::move(n)) {}
};

// Extra data for quantized weight tensors - stores extracted weights/scales/zp and weight node
struct ggml_openvino_quantized_weight_extra : public ggml_openvino_extra_base {
    ov::Tensor weights;   // U4 or U8 extracted weights
    ov::Tensor scales;    // F16 scales
    ov::Tensor zp;        // U4 or U8 zero points (same type as weights)
    std::shared_ptr<ov::Node> weight_node;  // Pre-built OpenVINO weight subgraph

    ggml_openvino_quantized_weight_extra(ov::Tensor w, ov::Tensor s, ov::Tensor z, std::shared_ptr<ov::Node> n) :
        ggml_openvino_extra_base(Type::QUANTIZED_WEIGHT),
        weights(std::move(w)),
        scales(std::move(s)),
        zp(std::move(z)),
        weight_node(std::move(n)) {}
};

// Extra data for KV cache / compute tensors - stores ov::Tensor for infer_request
struct ggml_openvino_tensor_extra : public ggml_openvino_extra_base {
    std::shared_ptr<ov::Tensor> tensor;  // For direct use with infer_request

    explicit ggml_openvino_tensor_extra(std::shared_ptr<ov::Tensor> t)
        : ggml_openvino_extra_base(Type::TENSOR), tensor(std::move(t)) {}
};

// =====================================================
// Extracted Size Calculation for Quantized Tensors
// =====================================================
// For quantized tensors, we need extra space to store extracted weights, scales, and zero points.
// Returns the total size needed in the buffer for extracted data.

struct ggml_openvino_extracted_layout {
    size_t total_size = 0;      // Total bytes needed
    size_t weights_offset = 0;  // Offset to weights in buffer
    size_t weights_size = 0;    // Size of weights in bytes
    size_t scales_offset = 0;   // Offset to scales in buffer
    size_t scales_size = 0;     // Size of scales in bytes
    size_t zp_offset = 0;       // Offset to zero points in buffer
    size_t zp_size = 0;         // Size of zero points in bytes (U4 or U8)
    bool is_u4;                 // true for U4 weights, false for U8
    int64_t weights_per_block;  // weights per scale/zp block
    bool is_symmetric;        // true for symmetric quantization

    // Requantization info
    bool is_requant = false;                      // true if this tensor needs requantization
    std::optional<ExtraQuantType> requant_type;   // target requant type if is_requant
};

// Calculate the buffer layout for extracted quantized data
ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_tensor * tensor, bool use_bias = false);

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(const ggml_tensor * tensor, bool is_remote);

// Register an extra with the tensor's OpenVINO buffer context for proper lifetime management.
// This sets tensor->extra and tracks the extra in the buffer context for cleanup.
void ggml_openvino_buffer_register_extra(ggml_tensor * tensor, ggml_openvino_extra_base * extra);

// =====================================================
// OpenVINO Backend Context and Interface
// =====================================================
struct ggml_backend_openvino_context {
    int device = 0;
    std::string name = "OpenVINO";
    std::string description = "OpenVINO Backend Context";

    std::shared_ptr<void> runtime_context = nullptr;

    ggml_backend_openvino_context() = default;
};

#include "ggml-openvino/utils.h"
/* Inlined ggml-quants-defs.inc */
#include "ggml.h.inc"

#include <cstdint>
#include <openvino/op/constant.hpp>
#include <openvino/runtime/tensor.hpp>

void unpack_32_4(const uint8_t* data, uint8_t* dst);

void extract_q4_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr);

void extract_q4_1_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias = false);

void extract_q8_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr);

void unpack_256_4(const uint8_t* data, uint8_t* dst);

void extract_q4_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias = false);

void extract_q5_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias = false);

void extract_q6_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr);

static constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

ov::Output<ov::Node> make_int8_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size = GGML_QUANTIZATION_GROUP_SIZE,
                                       bool use_bias = false);

ov::Output<ov::Node> make_int4_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size = GGML_QUANTIZATION_GROUP_SIZE,
                                       bool use_bias = false);

// Extract quantized weights from tensor and create weight subgraph
// If weights/scales/zp are provided (non-empty), uses them as output buffers
// Otherwise allocates new ov::Tensors internally
// Returns the weight node (make_int4_weights or make_int8_weights result)
std::shared_ptr<ov::Node> extract_quantized_weights(
    const ggml_tensor * tensor,
    const void * data,  // Source data pointer (may differ from tensor->data)
    ov::Tensor & weights,
    ov::Tensor & scales,
    ov::Tensor & zp,
    bool use_bias = false);  // Use fp bias instead of quantized zero_point (for test-backend-ops)

// Requantize weights from tensor to target format, writing to provided buffers
// For F16 target, only weights buffer is used (scales/zp ignored)
// Returns the weight node
std::shared_ptr<ov::Node> requantize_to_buffers(const ggml_tensor * tensor,
                                                const void * data,  // Source data pointer
                                                ExtraQuantType requant_type,
                                                int64_t block_size,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp);

inline const char * extra_quant_type_name(ExtraQuantType t) {
    switch (t) {
    case ExtraQuantType::F16:
        return "F16";
    case ExtraQuantType::Q4_0_C:
        return "Q4_0_C";
    case ExtraQuantType::Q4_0_128:
        return "Q4_0_128";
    case ExtraQuantType::Q8_0_C:
        return "Q8_0_C";
    case ExtraQuantType::Q8_0_32:
        return "Q8_0_32";
    case ExtraQuantType::Q8_1_C:
        return "Q8_1_C";
    default:
        return "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
struct OvWeight {
    std::shared_ptr<ov::Node> weight_node;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() const { return layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    const ggml_tensor * tensor,
    const void * data,                 // Source data pointer (may differ from tensor->data)
    void * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use fp bias instead of quantized zero_point, only used in test-backend-ops

void quantize_q4_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk);
void quantize_q8_1(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk);
void quantize_q8_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk);

namespace ov {
namespace op {
namespace util {
// From <openvino>/src/common/transformations/include/transformations/utils/utils.hpp
bool get_single_value(const std::shared_ptr<ov::op::v0::Constant>& const_node,
                      float& value,
                      bool check_value_range = true);
}  // namespace util
}  // namespace op
}  // namespace ov

#include "ggml.h.inc"

#include <atomic>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/allocator.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/tensor.hpp>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

// =====================================================
// OpenVINO Buffer Implementation using ov::Tensor
// =====================================================
//
// Design: This implementation uses a hybrid approach:
// 1. For weight tensors: Store a pre-built ov::op::v0::Constant in tensor->extra
//    - This avoids the memcpy during graph construction
//    - For quantized weights, the constant is already converted to OpenVINO format
// 2. For KV cache / compute tensors: Store an ov::Tensor in tensor->extra
//    - This can be directly passed to infer_request
//    - Future: can be changed to ov::RemoteTensor for GPU/NPU
//
// This design is similar to:
// - CUDA split buffer: tensor->extra stores device pointers
// - CPU repack buffer: tensor->extra stores tensor_traits with repacked data
// =====================================================

// Buffer context that manages per-tensor allocations (no contiguous buffer for weights)
struct ggml_backend_openvino_buffer_context {
    int device;
    std::string name;
    size_t id;

    // For non-weight buffers (KV cache, compute), we still use contiguous allocation
    void * data;
    size_t size;
    bool is_remote;

    // Wrapping of the buffer
    std::shared_ptr<ov::Tensor> ov_buffer;

    // Track all extras for cleanup
    std::map<ggml_tensor *, ggml_openvino_extra_base *> tensor_extras;

    // Used for re-allocation on device for kvcache
    void * data_prev;

    ggml_backend_openvino_buffer_context(int device, size_t size, bool is_remote = false) :
        device(device),
        name(std::string(GGML_OPENVINO_NAME) + std::to_string(device)),
        id([]() {
            static std::atomic<size_t> next_id{1};
            return next_id.fetch_add(1);
        }()),
        data(nullptr),
        size(size),
        is_remote(is_remote) {
        if (size == 0) {
            return;
        }

        const auto & device_name = ggml_openvino_get_device_name();

        if (is_remote) {
            GGML_ASSERT(device_name == "GPU");
            auto remote_context = ggml_openvino_get_remote_context();
            auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
            ov::intel_gpu::ocl::USMTensor usm_tensor =
                gpu_context.create_usm_device_tensor(ov::element::u8, ov::Shape{size});
            data = usm_tensor.get();
            ov_buffer = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
        } else {
            data = ggml_aligned_malloc(size);
            GGML_ASSERT(data);
            memset(data, 0, size);
            ov_buffer = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape{size}, data);
        }

        if (data == nullptr) {
            GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, size);
            return;
        }

        if (reinterpret_cast<uintptr_t>(data) % TENSOR_ALIGNMENT != 0) {
            GGML_LOG_ERROR("%s: %s buffer is not aligned to %d bytes\n", __func__, device_name.c_str(),
                           TENSOR_ALIGNMENT);
            GGML_ABORT("fatal error");
        }
    }

    ~ggml_backend_openvino_buffer_context() {
        // Clean up all tensor extras
        // GGML_LOG_DEBUG("Deleting OpenVINO buffer context #%zu for device %d, size %zu MB\n", id, device,
        //                size / 1024 / 1024);
        for (auto & pair : tensor_extras) {
            delete pair.second;
        }
        tensor_extras.clear();
        if (!is_remote && data != nullptr) {
            ggml_aligned_free(data, size);
        }
    }
};

// Buffer type context (per-device)
struct ggml_backend_openvino_buffer_type_context {
    int device;
    std::string name;
};

// Buffer interface functions
static void ggml_backend_openvino_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_openvino_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    return ctx->data;
}

static bool is_stateful_enabled() {
    static const auto * stateful = getenv("GGML_OPENVINO_STATEFUL_EXECUTION");
    return stateful && *stateful != '\0' && strcmp(stateful, "0") != 0;
}

static enum ggml_status ggml_backend_openvino_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    // GGML_LOG_DEBUG("%s: buffer usage=%d, tensor name=%s\n", __func__, buffer->usage, tensor->name);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    // Put kvcache on device memory for GPU (NPU memory is too small even for kvcache)
    if (strncmp(tensor->name, "cache_", 6) == 0 && !ctx->is_remote && ggml_openvino_get_device_name() == "GPU" &&
        !is_stateful_enabled()) {
        GGML_ASSERT(ctx->tensor_extras.empty());
        auto device = ctx->device;
        auto size = ctx->size;
        auto * data_prev = ctx->data;
        delete ctx;
        ctx = new ggml_backend_openvino_buffer_context(device, size, true);
        buffer->context = ctx;
        tensor->data = (char *) ctx->data + ((char *) tensor->data - (char *) data_prev);
    }

    // Views share the extra from view_src
    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        if (tensor->view_src->extra != nullptr) {
            tensor->extra = tensor->view_src->extra;
        }
        return GGML_STATUS_SUCCESS;
    }

    ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (tensor->data != nullptr && !ggml_is_quantized(tensor->type)) {
        ggml_openvino_tensor_extra * extra = ggml_openvino_create_tensor_extra(tensor, ctx->is_remote);
        if (extra != nullptr) {
            auto it = ctx->tensor_extras.find(tensor);
            if (it != ctx->tensor_extras.end()) {
                delete it->second;
            }
            ctx->tensor_extras[tensor] = extra;
            tensor->extra = extra;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_openvino_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                       ggml_tensor * tensor,
                                                       uint8_t value,
                                                       size_t offset,
                                                       size_t size) {
    // GGML_LOG_DEBUG("%s: buffer usage=%d, tensor name=%s\n", __func__, buffer->usage, tensor->name);
    GGML_ASSERT(tensor != nullptr && tensor->data != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (ctx->is_remote) {
        // For remote (device) buffers, use OpenCL USM memfill
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_fill_fn = ggml_openvino_get_clEnqueueMemFillINTEL();
        if (queue != nullptr && mem_fill_fn != nullptr) {
            uint8_t pattern = value;
            cl_int err = mem_fill_fn(queue, (char *) tensor->data + offset, &pattern, sizeof(pattern), size, 0, nullptr,
                                     nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clEnqueueMemFillINTEL failed with error %d\n", __func__, err);
            }
            clFinish(queue);
        } else {
            GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemFillINTEL not available for GPU buffer\n", __func__);
        }
    } else {
        memset((char *) tensor->data + offset, value, size);
    }
}

static void ggml_backend_openvino_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor * tensor,
                                                    const void * data,
                                                    size_t offset,
                                                    size_t size) {
    // GGML_LOG_DEBUG("%s: buffer usage=%d, tensor name=%s\n", __func__, buffer->usage, tensor->name);
    GGML_ASSERT(tensor != nullptr && tensor->data != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    // Check if this is a weight buffer (usage is set BEFORE set_tensor is called, except in test-backend-ops)
    bool is_weight_buffer = (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // Full tensor set: offset=0, full size, not a view
    bool is_full_tensor_set = (offset == 0 && size == ggml_nbytes(tensor) && tensor->view_src == nullptr);
    // 2D tensor (typical weight shape)
    bool is_2d = (tensor->ne[2] == 1 && tensor->ne[3] == 1);

    if (is_weight_buffer && is_full_tensor_set && is_2d) {
        try {
            auto result = process_weight_tensor(tensor, data, tensor->data);
            result.weight_node->set_friendly_name(tensor->name);

            // const auto & layout = result.layout;
            ggml_openvino_extra_base * extra;

            // Quantized path with extracted weight/scale/zp tensors
            if (result.is_quantized()) {
                extra = new ggml_openvino_quantized_weight_extra(std::move(result.weights), std::move(result.scales),
                                                                 std::move(result.zp), result.weight_node);

                // if (layout.is_requant) {
                //     GGML_LOG_DEBUG("%s: requantized %s to %s (u%d, block_size=%ld)\n", __func__, tensor->name,
                //                    extra_quant_type_name(layout.requant_type.value()), layout.is_u4 ? 4 : 8,
                //                    layout.weights_per_block);
                // } else {
                //     int64_t n_blocks = ggml_nelements(tensor) / layout.weights_per_block;
                //     GGML_LOG_DEBUG("%s: extracted quantized weight node for %s (u%d, %zu weights, %ld blocks)\n",
                //                    __func__, tensor->name, layout.is_u4 ? 4 : 8, layout.weights_size, n_blocks);
                // }
            } else {
                // F16/F32/BF16 weight or F16-requant
                extra = new ggml_openvino_weight_extra(std::move(result.weights), result.weight_node);

                // if (layout.total_size > 0) {
                //     GGML_LOG_DEBUG("%s: requantized %s to F16\n", __func__, tensor->name);
                // } else {
                //     GGML_LOG_DEBUG("%s: created shared-memory weight node for %s\n", __func__, tensor->name);
                // }
            }

            ctx->tensor_extras[tensor] = extra;
            tensor->extra = extra;

        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: failed to process weight tensor for %s: %s\n", __func__, tensor->name, e.what());
            memcpy((char *) tensor->data + offset, data, size);
        }
    } else {
        // Non-weight tensor (KV cache, activations, etc.) - copy data. test-backend-ops also goes here
        if (ctx->is_remote) {
            cl_command_queue queue = ggml_openvino_get_cl_queue();
            auto mem_cpy_fn = ggml_openvino_get_clEnqueueMemcpyINTEL();
            if (queue != nullptr && mem_cpy_fn != nullptr) {
                cl_int err =
                    mem_cpy_fn(queue, CL_TRUE, (char *) tensor->data + offset, data, size, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL failed with error %d\n", __func__, err);
                }
            } else {
                GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemcpyINTEL not available for GPU buffer\n", __func__);
            }
        } else {
            memcpy((char *) tensor->data + offset, data, size);
        }

        ggml_openvino_tensor_extra * extra = ggml_openvino_create_tensor_extra(tensor, ctx->is_remote);
        if (extra == nullptr) {
            // GGML_LOG_ERROR("%s: failed to create tensor extra for %s\n", __func__, tensor->name);
            return;
        }

        auto it = ctx->tensor_extras.find(tensor);
        if (it != ctx->tensor_extras.end()) {
            delete it->second;
        }
        ctx->tensor_extras[tensor] = extra;
        tensor->extra = extra;
    }
}

static void ggml_backend_openvino_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    const ggml_tensor * tensor,
                                                    void * data,
                                                    size_t offset,
                                                    size_t size) {
    // GGML_LOG_DEBUG("%s: buffer usage=%d, tensor name=%s\n", __func__, buffer->usage, tensor->name);
    GGML_ASSERT(tensor != nullptr && tensor->data != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (ctx->is_remote) {
        // For remote (device) buffers, use OpenCL USM memcpy (device-to-host)
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_cpy_fn = ggml_openvino_get_clEnqueueMemcpyINTEL();
        if (queue != nullptr && mem_cpy_fn != nullptr) {
            cl_int err =
                mem_cpy_fn(queue, CL_TRUE, data, (const char *) tensor->data + offset, size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL failed with error %d\n", __func__, err);
            }
        } else {
            GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemcpyINTEL not available for GPU buffer\n", __func__);
        }
    } else {
        memcpy(data, (const char *) tensor->data + offset, size);
    }
}

static bool ggml_backend_openvino_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    const ggml_tensor * src,
                                                    ggml_tensor * dst) {
    // GGML_LOG_DEBUG("%s: src tensor name=%s, dst tensor name=%s\n", __func__, src->name, dst->name);
    GGML_ASSERT(src != nullptr && dst != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (ctx->is_remote) {
        // For remote (device) buffers, use OpenCL USM memcpy
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_cpy_fn = ggml_openvino_get_clEnqueueMemcpyINTEL();
        if (queue == nullptr || mem_cpy_fn == nullptr) {
            GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemcpyINTEL not available for GPU buffer\n", __func__);
            return false;
        }
        // Can copy from host to device
        if (ggml_backend_buffer_is_host(src->buffer)) {
            cl_int err = mem_cpy_fn(queue, CL_TRUE, dst->data, src->data, ggml_nbytes(src), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL (host-to-device) failed with error %d\n", __func__, err);
                return false;
            }
            return true;
        }
        // Can also copy from device to device if both are OpenVINO remote buffers
        if (ggml_backend_buffer_is_openvino(src->buffer)) {
            ggml_backend_openvino_buffer_context * src_ctx =
                (ggml_backend_openvino_buffer_context *) src->buffer->context;
            if (src_ctx->is_remote) {
                cl_int err =
                    mem_cpy_fn(queue, CL_TRUE, dst->data, src->data, ggml_nbytes(src), 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL (device-to-device) failed with error %d\n", __func__,
                                   err);
                    return false;
                }
                return true;
            }
        }
        return false;
    }

    // Host buffer - can copy from any host buffer
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_openvino_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    GGML_ASSERT(ctx->data != nullptr);
    if (ctx->is_remote) {
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_fill_fn = ggml_openvino_get_clEnqueueMemFillINTEL();
        if (queue != nullptr && mem_fill_fn != nullptr) {
            uint8_t pattern = value;
            cl_int err = mem_fill_fn(queue, ctx->data, &pattern, sizeof(pattern), ctx->size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_WARN("%s: clEnqueueMemFillINTEL failed with error %d\n", __func__, err);
            }
            clFinish(queue);
        } else {
            GGML_LOG_WARN("%s: no OpenCL queue or clEnqueueMemFillINTEL not available for GPU buffer clear\n",
                          __func__);
        }
    } else {
        memset(ctx->data, value, ctx->size);
    }
}

static const ggml_backend_buffer_i ggml_backend_openvino_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_openvino_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_openvino_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_openvino_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_openvino_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_openvino_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_openvino_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_openvino_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_openvino_buffer_clear,
    /* .reset           = */ NULL,
};

// Buffer type interface functions
static const char * ggml_backend_openvino_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_openvino_buffer_type_context * ctx = (ggml_backend_openvino_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_openvino_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t size) {
    ggml_backend_openvino_buffer_type_context * buft_ctx = (ggml_backend_openvino_buffer_type_context *) buft->context;

    // Create buffer context with contiguous memory allocation
    ggml_backend_openvino_buffer_context * ctx = new ggml_backend_openvino_buffer_context(buft_ctx->device, size);

    if (ctx->data == nullptr && size > 0) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_openvino_buffer_interface, ctx, size);
}

static size_t ggml_backend_openvino_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TENSOR_ALIGNMENT;
}

static size_t ggml_backend_openvino_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

static size_t ggml_backend_openvino_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               const ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    // For quantized 2D tensors (weights), we need extra space for extracted data
    if (ggml_is_quantized(tensor->type) && tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        ggml_openvino_extracted_layout layout = ggml_openvino_get_extracted_layout(tensor);
        if (layout.total_size > 0) {
            // GGML_LOG_DEBUG("%s: tensor %s needs %zu bytes (original %zu, extracted: weights=%zu scales=%zu zp=%zu)\n",
            //                __func__, tensor->name, layout.total_size, ggml_nbytes(tensor), layout.weights_size,
            //                layout.scales_size, layout.zp_size);
            return layout.total_size;
        }
    }

    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_openvino_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_openvino_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_openvino_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_openvino_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_openvino_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_openvino_buffer_type_get_alloc_size,
    /* .is_host          = */ nullptr,
};

// Get buffer type for a specific device
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_buffer_type(int device) {
    GGML_ASSERT(device >= 0 && device < ggml_backend_openvino_get_device_count());

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> buffer_types;
    static std::vector<ggml_backend_openvino_buffer_type_context> buffer_type_contexts;

    if (buffer_types.empty()) {
        int device_count = ggml_backend_openvino_get_device_count();
        buffer_types.resize(device_count);
        buffer_type_contexts.resize(device_count);

        for (int i = 0; i < device_count; i++) {
            buffer_type_contexts[i].device = i;
            buffer_type_contexts[i].name = std::string(GGML_OPENVINO_NAME) + std::to_string(i);

            buffer_types[i] = ggml_backend_buffer_type{
                /* .iface   = */ ggml_backend_openvino_buffer_type_interface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_openvino_reg(), i),
                /* .context = */ &buffer_type_contexts[i],
            };
        }
    }

    return &buffer_types[device];
}

// =====================================================
// OpenVINO Host Buffer Implementation
// =====================================================

static const char * ggml_backend_openvino_host_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_openvino_buffer_type_context * ctx = (ggml_backend_openvino_buffer_type_context *) buft->context;
    static std::string name;
    name = ctx->name + "_HOST";
    return name.c_str();
}

static bool ggml_backend_openvino_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static const ggml_backend_buffer_type_i ggml_backend_openvino_host_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_openvino_host_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_openvino_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_openvino_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_openvino_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_openvino_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_openvino_host_buffer_type_is_host,
};

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_host_buffer_type(int device) {
    GGML_ASSERT(device >= 0 && device < ggml_backend_openvino_get_device_count());

    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> buffer_types;
    static std::vector<ggml_backend_openvino_buffer_type_context> buffer_type_contexts;

    if (buffer_types.empty()) {
        int device_count = ggml_backend_openvino_get_device_count();
        buffer_types.resize(device_count);
        buffer_type_contexts.resize(device_count);

        for (int i = 0; i < device_count; i++) {
            buffer_type_contexts[i].device = i;
            buffer_type_contexts[i].name = std::string(GGML_OPENVINO_NAME) + std::to_string(i);

            buffer_types[i] = ggml_backend_buffer_type{
                /* .iface   = */ ggml_backend_openvino_host_buffer_type_interface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_openvino_reg(), i),
                /* .context = */ &buffer_type_contexts[i],
            };
        }
    }

    return &buffer_types[device];
}

bool ggml_backend_buffer_is_openvino(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_openvino_buffer_free_buffer;
}

size_t ggml_backend_openvino_buffer_get_ctx_id(ggml_backend_buffer_t buffer) {
    if (!ggml_backend_buffer_is_openvino(buffer)) {
        return 0;
    }
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    return ctx->id;
}

void ggml_openvino_buffer_register_extra(ggml_tensor * tensor, ggml_openvino_extra_base * extra) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_openvino(tensor->buffer));

    auto * ctx = static_cast<ggml_backend_openvino_buffer_context *>(tensor->buffer->context);

    auto it = ctx->tensor_extras.find(tensor);
    if (it != ctx->tensor_extras.end()) {
        delete it->second;
    }

    ctx->tensor_extras[tensor] = extra;
    tensor->extra = extra;
}

bool ggml_backend_buft_is_openvino(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_openvino_buffer_type_get_name;
}

bool ggml_backend_buft_is_openvino_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_openvino_host_buffer_type_get_name;
}

static void ggml_backend_openvino_free(ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;

    if (ctx->runtime_context) {
        auto r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        if (--r_ctx->backend_count == 0) {
            r_ctx->clear_caches();
        }
    }

    delete ctx;
    delete backend;
}

static const char * ggml_backend_openvino_get_name(ggml_backend_t backend) {
    return GGML_OPENVINO_NAME;
    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_openvino_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return ov_graph_compute(cgraph, backend);
    GGML_UNUSED(backend);
}

static const ggml_backend_i ggml_backend_openvino_interface = {
    /* .get_name                = */ ggml_backend_openvino_get_name,
    /* .free                    = */ ggml_backend_openvino_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_openvino_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

int ggml_backend_openvino_get_device_count() {
    return 1;
}

static ggml_guid_t ggml_backend_openvino_guid(void) {
    static ggml_guid guid = {0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97,
                             0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d};
    return &guid;
}

static std::shared_ptr<ov_runtime_context> get_ov_runtime_context_ptr() {
    static std::shared_ptr<ov_runtime_context> r_ctx = [] {
        auto ctx = std::make_shared<ov_runtime_context>();
        ctx->device = ggml_openvino_get_device_name();
        ctx->stateful = is_stateful_enabled() && !ggml_openvino_is_npu();
        return ctx;
    }();
    return r_ctx;
}

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_openvino_init(int device) {
    if (device < 0 || device >= ggml_backend_openvino_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_openvino_context * ctx = new ggml_backend_openvino_context;
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ctx->runtime_context = get_ov_runtime_context_ptr();
    if (ctx->runtime_context == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate runtime context\n", __func__);
        delete ctx;
        return nullptr;
    }

    std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
    r_ctx->backend_count++;

    ggml_backend_t openvino_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_openvino_guid(),
        /* .interface = */ ggml_backend_openvino_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_openvino_reg(), device),
        /* .context   = */ ctx,
    };

    return openvino_backend;
}

GGML_BACKEND_API bool ggml_backend_is_openvino(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_openvino_guid());
}

struct ggml_backend_openvino_device_context {
    int device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_openvino_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_openvino_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_openvino_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif  // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_openvino_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_openvino_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_openvino_device_get_name(dev);
    props->description = ggml_backend_openvino_device_get_description(dev);
    props->type = ggml_backend_openvino_device_get_type(dev);
    ggml_backend_openvino_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_openvino_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ggml_backend_openvino_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_openvino_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ggml_backend_openvino_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_openvino_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ggml_backend_openvino_host_buffer_type(ctx->device);
}

static bool has_view_op_input(const ggml_tensor * op) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] == nullptr) {
            break;
        }
        if (op->src[i]->op == GGML_OP_VIEW) {
            return true;
        }
    }
    return false;
}

static bool is_supported_flash_attn_pattern(const ggml_tensor * op) {
    // pattern of q,k,v should be q->op==PERMUTE, q->src[0]->op==VIEW, q->src[0]->src[0]->view_src==nullptr
    for (int i = 0; i < 3; i++) {
        const ggml_tensor * src = op->src[i];
        if (src->op != GGML_OP_PERMUTE || src->src[0] == nullptr || src->src[0]->op != GGML_OP_VIEW ||
            src->src[0]->src[0] == nullptr || src->src[0]->src[0]->view_src != nullptr) {
            return false;
        }
    }
    return true;
}

static bool is_op_unsupported_case(const ggml_tensor * op) {
    switch (op->op) {
    case GGML_OP_GET_ROWS:
    case GGML_OP_SET_ROWS: {
        if (op->ne[3] != 1) {
            return true;
        }
        break;
    }
    case GGML_OP_ADD:
    case GGML_OP_MUL: {
        if (op->src[1]->op == GGML_OP_PERMUTE) {
            return true;
        }
        for (int i = 0; i < 4; i++) {
            if (op->src[0]->ne[i] != op->src[1]->ne[i] && (op->src[0]->ne[i] != 1 && op->src[1]->ne[i] != 1)) {
                return true;
            }
        }
        break;
    }
    case GGML_OP_SOFT_MAX: {
        if (op->src[2] != nullptr) {
            // GGML_LOG_WARN("OpenVINO backend does not support SOFT_MAX with sinks\n");
            return true;
        }
        float scale = 1.0f;
        float max_bias = 0.0f;
        const auto * op_params = op->op_params;
        memcpy(&scale, (const float *) op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) op_params + 1, sizeof(float));
        if (max_bias > 0) {
            // GGML_LOG_WARN("OpenVINO backend does not support SOFT_MAX with max_bias > 0\n");
            return true;
        }
        break;
    }
    case GGML_OP_FLASH_ATTN_EXT: {
        if (op->src[4] != nullptr) {
            // GGML_LOG_WARN("OpenVINO backend does not support FLASH_ATTN_EXT with sinks\n");
            return true;
        }
        if (!is_supported_flash_attn_pattern(op)) {
            return true;
        }
        float scale = 1.0f;
        float max_bias = 0.0f;
        float logit_softcap = 0.0f;
        const auto * op_params = op->op_params;
        memcpy(&scale, (const float *) op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) op_params + 1, sizeof(float));
        memcpy(&logit_softcap, (const float *) op_params + 2, sizeof(float));
        if (max_bias > 0) {
            // GGML_LOG_WARN("OpenVINO backend does not support FLASH_ATTN_EXT with max_bias > 0\n");
            return true;
        }
        if (logit_softcap != 0) {
            // GGML_LOG_WARN("OpenVINO backend does not support FLASH_ATTN_EXT with logit_softcap != 0\n");
            return true;
        }
        break;
    }
    case GGML_OP_PERMUTE: {
        if (op->type == GGML_TYPE_BF16) {
            // err msg: [GPU] Could not find a suitable kernel for transpose
            // GGML_LOG_WARN("OpenVINO backend does not support PERMUTE with BF16 type\n");
            return true;
        }
        break;
    }
    case GGML_OP_CPY: {
        if (op->src[1] != op) {
            // GGML_LOG_WARN("OpenVINO backend only supports CPY that is a cast\n");
            return true;
        }
        break;
    }
    case GGML_OP_MUL_MAT: {
        if (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16) {
            // Has accuracy issue, try enabling this and see `test-backend-ops -o "MUL_MAT"`
            // GGML_LOG_WARN("OpenVINO backend does not support MUL_MAT with two F16 tensors\n");
            return true;
        }
        if (op->src[0]->ne[3] != op->src[1]->ne[3] && op->src[0]->ne[3] != 1 && op->src[1]->ne[3] != 1) {
            return true;
        }
        if (op->src[0]->op == GGML_OP_PERMUTE || op->src[1]->op == GGML_OP_PERMUTE) {
            return true;
        }
        if (ggml_is_quantized(op->src[0]->type) && op->src[0]->ne[1] == 1) {
            // MUL_MAT(type_a=q4_0,type_b=f32,m=1,n=2048,k=8192,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1)
            // triggers a bug in ov matmul_shape_inference.hpp
            return true;
        }
        if (op->src[0]->op == GGML_OP_VIEW && op->src[1]->op == GGML_OP_VIEW) {
            return true;
        }
        break;
    }
    case GGML_OP_ROPE: {
        const int32_t * op_params = op->op_params;
        const int n_dims = op_params[1];
        const int mode = op_params[2];
        if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_IMROPE) {
            // GGML_LOG_WARN("OpenVINO backend does not support ROPE with mode %d\n", mode);
            return true;
        }
        if (n_dims != 0.0f && n_dims != op->src[0]->ne[0]) {
            // GGML_LOG_WARN("OpenVINO backend does not support ROPE with n_dims %d != src[0]->ne[0] %ld\n", n_dims,
            //               op->src[0]->ne[0]);
            return true;
        }
        if (op->type != GGML_TYPE_F32) {
            // GGML_LOG_WARN("OpenVINO backend does not support ROPE with type %s\n", ggml_type_name(op->type));
            return true;
        }
        if (op->src[0]->op == GGML_OP_VIEW) {
            if (op->src[0]->view_src->ne[1] != op->src[0]->ne[2]) {
                // GGML_LOG_WARN(
                //     "OpenVINO backend does not support ROPE with src[0]->view_src->ne[1] %ld != src[0]->ne[2] "
                //     "%ld\n",
                //     op->src[0]->view_src->ne[1], op->src[0]->ne[2]);
                return true;
            }
        }
        if (mode == GGML_ROPE_TYPE_IMROPE &&
            (op->src[2] != 0 || ((const float *) op_params)[6] != 1 || ((const float *) op_params)[7] != 0 ||
             ((const float *) op_params)[8] != 1)) {
            // GGML_LOG_WARN("OpenVINO backend does not support IMROPE with freq_factors, freq_scale, ext_factor, and attn_factor\n");
            return true;
        }
        break;
    }
    default:
        break;
    }
    if (op->op == GGML_OP_GET_ROWS) {
        if (op->ne[0] == 256 && (op->src[0]->type == GGML_TYPE_Q4_K || op->src[0]->type == GGML_TYPE_Q5_K)) {
            // ERR = 0.000000306 > 0.000000100   GET_ROWS(type=q4_K,n=256,m=5,r=4,be1=1,be2=1,v=0)
            // ERR = 0.000000197 > 0.000000100   GET_ROWS(type=q5_K,n=256,m=5,r=4,be1=1,be2=1,v=0)
            return true;
        }
    }
    return false;
}

static bool ggml_backend_openvino_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(dev->reg != nullptr);

    static std::set<ggml_type> supported_types{GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_I64,
                                               GGML_TYPE_I32,  GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K,
                                               GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K};

    static const std::set<ggml_op> supported_ops{GGML_OP_NONE, GGML_OP_ADD, GGML_OP_MUL, GGML_OP_MUL_MAT, GGML_OP_VIEW,
                                                 /*GGML_OP_CONT,*/ GGML_OP_RESHAPE, GGML_OP_PERMUTE, GGML_OP_TRANSPOSE,
                                                 GGML_OP_GET_ROWS, GGML_OP_ROPE, GGML_OP_RMS_NORM, GGML_OP_SCALE,
                                                 // softmax is not updated due to replaced by flash_attn_ext
                                                 // GGML_OP_SOFT_MAX,
                                                 GGML_OP_SET_ROWS, GGML_OP_FLASH_ATTN_EXT, GGML_OP_CPY};
    static const std::set<ggml_unary_op> supported_unary_ops{
        GGML_UNARY_OP_GELU,
        GGML_UNARY_OP_SILU,
    };
    static const std::set<ggml_glu_op> supported_glu_ops{
        GGML_GLU_OP_SWIGLU,
        GGML_GLU_OP_GEGLU,
    };

    switch (op->op) {
    case GGML_OP_UNARY: {
        auto supported = supported_unary_ops.find(ggml_get_unary_op(op)) != supported_unary_ops.end();
        if (!supported) {
            // GGML_LOG_WARN("OpenVINO backend does not support unary op %s\n", ggml_unary_op_name(ggml_get_unary_op(op)));
            return false;
        }
        if (has_view_op_input(op)) {
            // GGML_LOG_WARN("OpenVINO backend does not support unary op %s with view input\n",
            //               ggml_unary_op_name(ggml_get_unary_op(op)));
            return false;
        }
        break;
    }
    case GGML_OP_GLU: {
        auto supported = supported_glu_ops.find(ggml_get_glu_op(op)) != supported_glu_ops.end();
        if (!supported) {
            // GGML_LOG_WARN("OpenVINO backend does not support GLU op %s\n", ggml_glu_op_name(ggml_get_glu_op(op)));
            return false;
        }
        if (has_view_op_input(op)) {
            // GGML_LOG_WARN("OpenVINO backend does not support unary op %s with view input\n",
            //               ggml_glu_op_name(ggml_get_glu_op(op)));
            return false;
        }
        if (op->src[1] == nullptr && op->src[0]->ne[0] % 2 != 0) {
            // triggers bug in ov gpu
            return false;
        }
        break;
    }
    default: {
        auto supported = supported_ops.find(op->op) != supported_ops.end();
        if (!supported) {
            // GGML_LOG_WARN("OpenVINO backend does not support op %s\n", ggml_op_name(op->op));
            return false;
        }
        static std::set<ggml_op> ops_not_support_view_input{
            GGML_OP_GET_ROWS,
            GGML_OP_RMS_NORM,
        };
        if (ops_not_support_view_input.find(op->op) != ops_not_support_view_input.end() && has_view_op_input(op)) {
            // GGML_LOG_WARN("OpenVINO backend does not support op %s with view input\n", ggml_op_name(op->op));
            return false;
        }
    }
    }

    if (supported_types.find(op->type) == supported_types.end()) {
        // GGML_LOG_WARN("OpenVINO backend does not support tensor type %s\n", ggml_type_name(op->type));
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto * src = op->src[i];
        if (src == nullptr) {
            break;
        }
        if (supported_types.find(src->type) == supported_types.end()) {
            // GGML_LOG_WARN("OpenVINO backend does not support tensor type %s\n", ggml_type_name(src->type));
            return false;
        }
        if (ggml_is_quantized(src->type) && src->ne[2] != 1) {
            // GGML_LOG_WARN("OpenVINO backend does not support 3D quantized tensors\n");
            return false;
        }
    }

    if (is_op_unsupported_case(op)) {
        return false;
    }
    return true;
}

static bool ggml_backend_openvino_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_openvino(buft) || ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_openvino_device_interface = {
    /* .get_name             = */ ggml_backend_openvino_device_get_name,
    /* .get_description      = */ ggml_backend_openvino_device_get_description,
    /* .get_memory           = */ ggml_backend_openvino_device_get_memory,
    /* .get_type             = */ ggml_backend_openvino_device_get_type,
    /* .get_props            = */ ggml_backend_openvino_device_get_props,
    /* .init_backend         = */ ggml_backend_openvino_device_init,
    /* .get_buffer_type      = */ ggml_backend_openvino_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_openvino_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_openvino_device_supports_op,
    /* .supports_buft        = */ ggml_backend_openvino_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

struct ggml_backend_openvino_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_openvino_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_OPENVINO_NAME;
    GGML_UNUSED(reg);
}

static size_t ggml_backend_openvino_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return (size_t) ggml_backend_openvino_get_device_count();
}

static ggml_backend_dev_t ggml_backend_openvino_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_openvino_reg_context * ctx = (ggml_backend_openvino_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_openvino_reg_interface = {
    /* .get_name         = */ ggml_backend_openvino_reg_get_name,
    /* .get_device_count = */ ggml_backend_openvino_reg_get_device_count,
    /* .get_device       = */ ggml_backend_openvino_reg_get_device,
    /* .get_proc_address = */ NULL,
};

static void ggml_openvino_init() {
    // Initialize device config singleton from env var
    ggml_openvino_init_device_config();
    GGML_LOG_INFO("OpenVINO: using device %s\n", ggml_openvino_get_device_name().c_str());
}

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_openvino_reg(void) {
    static ggml_backend_reg reg;

    static bool initialized = false;
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_openvino_init();

            ggml_backend_openvino_reg_context * ctx = new ggml_backend_openvino_reg_context;

            for (int i = 0; i < ggml_backend_openvino_get_device_count(); i++) {
                ggml_backend_openvino_device_context * dev_ctx = new ggml_backend_openvino_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_OPENVINO_NAME + std::to_string(i);

                dev_ctx->description = ov::get_openvino_version().description;

                ggml_backend_dev_t dev =
                    new ggml_backend_device{/* .interface = */ ggml_backend_openvino_device_interface,
                                            /* .reg       = */ &reg,
                                            /* .context   = */ dev_ctx};
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{/* .api_version = */ GGML_BACKEND_API_VERSION,
                                   /* .iface       = */ ggml_backend_openvino_reg_interface,
                                   /* .context     = */ ctx};
        }

        initialized = true;
    }

    return &reg;
}


/* Inlined ggml-openvino-extra.cpp */

#include "ggml-impl-defs.inc"
#include "ggml.h.inc"

#include <cstring>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/properties.hpp>
#include <optional>

ov::Core & ov_singleton_core() {
    static ov::Core core;
    return core;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

void ggml_openvino_device_config::init() {
    if (initialized) {
        return;
    }
    device_name = getenv("GGML_OPENVINO_DEVICE") ? getenv("GGML_OPENVINO_DEVICE") : "CPU";
    auto available_devices = ov_singleton_core().get_available_devices();
    if (std::find(available_devices.begin(), available_devices.end(), device_name) == available_devices.end()) {
        GGML_LOG_WARN("GGML OpenVINO Backend: device %s is not available, fallback to CPU\n", device_name.c_str());
        device_name = "CPU";
    }
    is_npu = (device_name == "NPU");

    auto * cache_dir = getenv("GGML_OPENVINO_CACHE_DIR");
    if (device_name == "NPU") {
        compile_config = {
            {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"   },
            {"NPU_USE_NPUW",                      "YES"   },
            {"NPUW_DEVICES",                      "NPU"   },
            {"NPUW_FOLD",                         "YES"   },
            {"NPUW_WEIGHTS_BANK",                 "shared"},
            {"NPUW_FUNCALL_FOR_ALL",              "YES"   },
            {"NPUW_FUNCALL_ASYNC",                "YES"   },
            {"NPUW_DQ",                           "YES"   },
            {"NPUW_DQ_FULL",                      "NO"    },
        };
        if (cache_dir && strlen(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
    } else if (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    }

    // Initialize remote context with queue sharing for GPU
    if (device_name == "GPU") {
        // Create OpenCL context and queue
        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to get OpenCL platform: %d\n", err);
            return;
        }

        cl_device_id cl_device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &cl_device, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to get OpenCL device: %d\n", err);
            return;
        }

        cl_context cl_ctx = clCreateContext(nullptr, 1, &cl_device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to create OpenCL context: %d\n", err);
            return;
        }

        cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device, nullptr, &err);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to create OpenCL command queue: %d\n", err);
            clReleaseContext(cl_ctx);
            return;
        }

        // Create OpenVINO remote context with queue sharing
        remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), cl_queue);

        // Release the context (queue keeps a reference)
        clReleaseContext(cl_ctx);
    } else if (device_name == "NPU") {
        // remote tensor is not used for NPU yet
        // remote_context = ov_singleton_core().get_default_context(device_name);
    }

    initialized = true;
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    if (cl_queue != nullptr) {
        clReleaseCommandQueue(cl_queue);
        cl_queue = nullptr;
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    static ggml_openvino_device_config config;
    return config;
}

// Initialize device config (call during backend init)
void ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
const std::string & ggml_openvino_get_device_name() {
    return ggml_openvino_get_device_config().device_name;
}

// Check if running on NPU
bool ggml_openvino_is_npu() {
    return ggml_openvino_get_device_config().is_npu;
}

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context() {
    return ggml_openvino_get_device_config().remote_context;
}

// Get the compile config for the current device
const ov::AnyMap & ggml_openvino_get_compile_config() {
    return ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue for GPU operations
cl_command_queue ggml_openvino_get_cl_queue() {
    return ggml_openvino_get_device_config().cl_queue;
}

// Get the clEnqueueMemFillINTEL function pointer (lazy load)
clEnqueueMemFillINTEL_fn ggml_openvino_get_clEnqueueMemFillINTEL() {
    static clEnqueueMemFillINTEL_fn fn = nullptr;
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
            fn = (clEnqueueMemFillINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemFillINTEL");
        }
    }
    return fn;
}

// Get the clEnqueueMemcpyINTEL function pointer (lazy load)
clEnqueueMemcpyINTEL_fn ggml_openvino_get_clEnqueueMemcpyINTEL() {
    static clEnqueueMemcpyINTEL_fn fn = nullptr;
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
            fn = (clEnqueueMemcpyINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemcpyINTEL");
        }
    }
    return fn;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(const ggml_tensor * tensor, bool no_requant) {
    if (no_requant) {
        return std::nullopt;
    }
    if (strncmp(tensor->name, "token_embd.weight", 17) == 0) {
        return ((ggml_openvino_is_npu() && tensor->type == GGML_TYPE_Q6_K) ? ExtraQuantType::F16 : ExtraQuantType::Q8_0_C);
    }
    if (strncmp(tensor->name, "output.weight", 13) == 0) {
        return ExtraQuantType::Q8_0_C;
    }
    if (ggml_openvino_is_npu()) {
        return ExtraQuantType::Q4_0_128;
    }
    switch (tensor->type) {
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q5_K:
        return ExtraQuantType::Q8_0_C;
    default:
        return std::nullopt;
    }
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_tensor * tensor, bool use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_symmetric = false;

    if (!ggml_is_quantized(tensor->type)) {
        return layout;
    }

    // Only handle 2D weight tensors
    if (tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        return layout;
    }

    int64_t n_elements = ggml_nelements(tensor);
    const size_t alignment = 64;  // Good for SIMD

    // Check if requantization is needed (NPU-specific)
    auto requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    if (requant_type.has_value()) {
        layout.is_requant = true;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        if (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_size = n_elements * sizeof(uint16_t);  // F16 = 2 bytes
            layout.total_size = layout.weights_size;
            layout.weights_offset = 0;
            // No scales/zp for F16
            return layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        switch (requant_type.value()) {
        case ExtraQuantType::Q4_0_128:
            layout.is_u4 = true;
            layout.weights_per_block = 128;
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q4_0_C:
            layout.is_u4 = true;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_0_32:
            layout.is_u4 = false;
            layout.weights_per_block = 32;
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_0_C:
            layout.is_u4 = false;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_1_C:
            layout.is_u4 = false;
            layout.weights_per_block = tensor->ne[0];
            break;
        default:
            layout.weights_per_block = -1;
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
            break;
        }

        if (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_blocks = n_elements / layout.weights_per_block;
            layout.scales_size = n_blocks * sizeof(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            if (layout.is_symmetric) {
                layout.zp_size = 0;
            } else {
                layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
            }

            layout.weights_offset = 0;
            layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
            layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
            layout.total_size = layout.zp_offset + layout.zp_size;
            layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));
            return layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = false;
    layout.weights_per_block = 32;
    layout.is_symmetric = false;

    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
        layout.is_u4 = true;
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q4_K:
        layout.is_u4 = true;
        break;

    case GGML_TYPE_Q8_0:
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q6_K:
        layout.weights_per_block = 16;
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q5_K:
        break;

    default:
        // Unsupported quantization type
        return layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block
    int64_t n_blocks = n_elements / layout.weights_per_block;
    layout.scales_size = n_blocks * sizeof(uint16_t);  // F16 = 2 bytes
    // For symmetric quantization, no zp needed (weights stored as signed)
    if (layout.is_symmetric) {
        layout.zp_size = 0;
    } else {
        layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_offset = 0;
    layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
    layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
    layout.total_size = layout.zp_offset + layout.zp_size;
    layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));

    return layout;
}

ggml_openvino_tensor_extra * ggml_openvino_create_tensor_extra(const ggml_tensor * tensor, bool is_remote) {
    ov::Shape shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    ov::element::Type element_type;
    switch (tensor->type) {
    case GGML_TYPE_F32:
        element_type = ov::element::f32;
        break;
    case GGML_TYPE_F16:
        element_type = ov::element::f16;
        break;
    case GGML_TYPE_BF16:
        element_type = ov::element::bf16;
        break;
    case GGML_TYPE_I32:
        element_type = ov::element::i32;
        break;
    case GGML_TYPE_I64:
        element_type = ov::element::i64;
        break;
    default:
        // GGML_LOG_WARN("%s: unsupported tensor type for ov::Tensor: %s\n", __func__, ggml_type_name(tensor->type));
        return nullptr;
    }

    const auto & device_name = ggml_openvino_get_device_name();
    auto remote_context = ggml_openvino_get_remote_context();

    std::shared_ptr<ov::Tensor> ov_tensor;
    if (is_remote) {
        GGML_ASSERT(device_name == "GPU");
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        auto usm_tensor = gpu_context.create_tensor(element_type, shape, tensor->data);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } else {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->data);
    }

    return new ggml_openvino_tensor_extra(ov_tensor);
}


/* Inlined ggml-quants.cpp */

#include "ggml-common-defs.inc"
#include "ggml-impl-defs.inc"
#include "ggml.h.inc"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/parallel.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/util/attr_types.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include <vector>

void unpack_32_4(const uint8_t * data, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    for (int j = 0; j < 16; ++j) {
        uint8_t x = (data[j] & 0x0F);
        uint8_t y = (data[j] >> 4);
        if (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

// Extracts (weight, scales, zp) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i4 (value - 8).
void extract_q4_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    const uint64_t bytes_per_block = 18;  // 2 bytes scale, 32x0.5 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
            // Pack two 4-bit zero points per byte
            if (i % 2 == 0) {
                zp[i / 2] = 8;          // Lower nibble
            } else {
                zp[i / 2] |= (8 << 4);  // Upper nibble
            }
            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
        });
    } else {
        // Symmetric: unpack as u4 then convert to i4 by subtracting 8 (XOR each nibble)
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
            // Convert u4 to i4: subtract 8 from each nibble. XOR 0x88 flips each nibble by 8.
            for (int j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

// Extracts (weight, scales, zp) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit min|32 x 4bit weights|.
void extract_q4_1_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    if (use_bias) {
        // Store bias (min) directly as f16 instead of computing u4 zero points
        auto * bias = zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            float scale = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            float min = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);  // bias = min, dequant: w*s + bias
            unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
        });
    } else {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            float scale = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            float min = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            // zp = -min / scale (bias = min, so zp = -bias/scale)
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            // Pack two 4-bit zero points per byte
            if (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;   // Lower nibble
            } else {
                zp[i / 2] |= (zp_val << 4);  // Upper nibble
            }
            unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
        });
    }
}

// Extracts (weight, scales, zp) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
// When zp_arr is empty (symmetric), weights are stored as signed i8 directly.
void extract_q8_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
            zp[i] = 128;
            for (size_t j = 0; j < weights_per_block; ++j) {
                uint8_t x = block_data[j + 2];
                x ^= 1 << 7;  // Convert int8 to uint8 by flipping sign bit
                weights[i * weights_per_block + j] = x;
            }
        });
    } else {
        // Symmetric: store original int8 values directly (no unsigned bias)
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
            // Copy int8 weights as-is (the tensor element type is i8)
            memcpy(weights + i * weights_per_block, block_data + 2, weights_per_block);
        });
    }
}

void unpack_256_4(const uint8_t * data, uint8_t * dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    for (size_t i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            uint8_t x = (data[i * 32 + j] & 0x0F);
            uint8_t y = (data[i * 32 + j] >> 4);
            if (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

void extract_q4_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 2 + 2 + 12 + 128;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds packed u4 zero points
    auto * zp_u4 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    auto * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;

        // Extract scale factors and offsets
        float scale_scales = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data)));
        float scale_mins = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        // Extract qs1 and qs2
        uint8_t * qs1 = block_data + 4;

        // Calculate scales
        float scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<float>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<float>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<float>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<float>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<float>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<float>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<float>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<float>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        // Calculate min values (bias = -min)
        float min_vals[8];
        min_vals[0] = scale_mins * static_cast<float>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<float>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<float>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<float>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<float>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<float>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<float>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<float>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Store scales and compute zero points or bias
        for (int j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            if (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } else {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                // Pack two 4-bit zero points per byte
                size_t idx = i * 8 + j;
                if (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } else {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(block_data + 16, weights + i * 128);
    });
}

void extract_q6_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    const uint64_t bytes_per_block = 128 + 64 + 16 + 2;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(n_super_block, [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            float scale_factor = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
            for (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<float>(*((int8_t *) (block_data + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = block_data;
            uint8_t * qh = block_data + 128;
            for (int64_t j = 0; j < 32; ++j) {
                weights[i * 256 + j] = (ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4);
                weights[i * 256 + j + 32] = (ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4);
                weights[i * 256 + j + 64] = (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4);
                weights[i * 256 + j + 96] = (ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4);
                weights[i * 256 + j + 128] = (ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4);
                weights[i * 256 + j + 160] = (ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4);
                weights[i * 256 + j + 192] = (ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4);
                weights[i * 256 + j + 224] = (ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4);
            }
        });
    } else {
        // Symmetric: subtract 32 from each weight to store as signed i8
        ov::parallel_for(n_super_block, [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            float scale_factor = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
            for (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<float>(*((int8_t *) (block_data + 128 + 64 + j))));
            }
            uint8_t * ql = block_data;
            uint8_t * qh = block_data + 128;
            auto * signed_weights = reinterpret_cast<int8_t *>(weights);
            for (int64_t j = 0; j < 32; ++j) {
                signed_weights[i * 256 + j] = static_cast<int8_t>((ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 32] =
                    static_cast<int8_t>((ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 64] = static_cast<int8_t>((ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 96] =
                    static_cast<int8_t>((ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 128] =
                    static_cast<int8_t>((ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 160] =
                    static_cast<int8_t>((ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 192] =
                    static_cast<int8_t>((ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 224] =
                    static_cast<int8_t>((ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4)) - 32;
            }
        });
    }
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void extract_q5_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 4 + 12 + 32 + 128;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // For bias path, zp_arr holds f16 bias values; for zp path, it holds u8 zero points
    auto * zp_u8 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    auto * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;

        const float d = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data)));
        const float min_factor = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        const uint8_t * scales_data = block_data + 4;   // 12 bytes of scales
        const uint8_t * qh = block_data + 4 + 12;       // 32 bytes of high bits
        const uint8_t * ql = block_data + 4 + 12 + 32;  // 128 bytes of low bits

        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        // Process 2 blocks in one iteration
        for (int j = 0; j < 256; j += 64) {  // 256 = QK_K, so 4 iterations of 64
            uint8_t sc;
            uint8_t m;

            // Get scale and min for first 32 elements
            get_scale_min_k4(is + 0, scales_data, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min_factor * m;

            // Get scale and min for second 32 elements
            get_scale_min_k4(is + 1, scales_data, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min_factor * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            if (use_bias) {
                // Store bias = -min directly as f16, dequant: w*s + bias
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } else {
                // zp = min / scale (since bias = -min and zp = -bias/scale)
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            // Extract weights for first 32 elements (matching deq formula exactly)
            for (int l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            // Extract weights for second 32 elements
            for (int l = 0; l < 32; ++l) {
                weights[i * 256 + j + l + 32] = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            }

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    });
}

// TODO Reorder for make_intX_weights

ov::Output<ov::Node> make_int8_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size,
                                       bool use_bias) {
    ov::Shape orig_shape = weight.get_shape();
    bool is_signed = (weight.get_element_type() == ov::element::i8);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    auto scale_shape = scales.get_shape();

    ov::Shape packed_shape = {orig_shape[0], orig_shape[1] / group_size, group_size};

    if (packed_shape[1] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + 1);
    } else {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        if (!is_signed && zp.get_size() > 0) {
            auto zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> result;
    if (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::i8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
        result = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
    } else {
        // Unsigned path
        auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u8, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);

        if (use_bias && zp.get_size() > 0) {
            // Bias path: w * s + b (zp tensor holds f16 bias values)
            auto bias_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            auto w_s =
                std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
            result = std::make_shared<ov::op::v1::Add>(w_s, bias_f16, ov::op::AutoBroadcastType::NUMPY);
        } else {
            // Zero point path: (w - zp) * s
            auto zero_point = std::make_shared<ov::op::v0::Constant>(zp);
            float zp_value;
            if (ov::op::util::get_single_value(zero_point, zp_value)) {
                zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_value});
            }
            auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
            result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        }
    }

    if (packed_shape.size() != 2) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        result = std::make_shared<ov::op::v1::Reshape>(result, final_shape, false);
    }

    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

ov::Output<ov::Node> make_int4_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size,
                                       bool use_bias) {
    ov::Shape orig_weight_shape = weight.get_shape();
    bool is_signed = (weight.get_element_type() == ov::element::i4);  // Symmetric: signed weights, no ZP

    // Expand dimensions for scales and zp/bias
    ov::Shape scale_shape = scales.get_shape();

    // Create INT4 weight tensor
    ov::Shape packed_shape = {orig_weight_shape[0], orig_weight_shape[1] / group_size, group_size};

    if (packed_shape[1] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + 1);
    } else {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        if (!is_signed && zp.get_size() > 0) {
            auto zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> result;
    if (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::i4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
        result = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
    } else {
        // Unsigned path
        auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u4, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);

        if (use_bias && zp.get_size() > 0) {
            // Bias path: w * s + b (zp tensor holds f16 bias values)
            auto bias_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            auto w_s =
                std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
            result = std::make_shared<ov::op::v1::Add>(w_s, bias_f16, ov::op::AutoBroadcastType::NUMPY);
        } else {
            // Zero point path: (w - zp) * s
            auto zero_points_node = std::make_shared<ov::op::v0::Constant>(zp);
            float zp_value;
            if (ov::op::util::get_single_value(zero_points_node, zp_value)) {
                zero_points_node = ov::op::v0::Constant::create(zero_points_node->get_element_type(), {}, {zp_value});
            }
            auto zero_points_f16 = std::make_shared<ov::op::v0::Convert>(zero_points_node, ov::element::f16);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_points_f16, ov::op::AutoBroadcastType::NUMPY);
            result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        }
    }

    if (packed_shape.size() != 2) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_weight_shape.size()},
                                                                  orig_weight_shape);
        result = std::make_shared<ov::op::v1::Reshape>(result, final_shape, false);
    }

    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

// Extract quantized weights from tensor and create weight subgraph
std::shared_ptr<ov::Node> extract_quantized_weights(const ggml_tensor * tensor,
                                                    const void * data,
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    bool use_bias) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor temp_tensor = *tensor;
    temp_tensor.data = const_cast<void *>(data);

    // Determine block size based on tensor type
    int64_t weights_per_block;
    bool is_u4;
    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q4_K:
        is_u4 = true;
        weights_per_block = 32;
        break;
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q5_K:
        is_u4 = false;
        weights_per_block = 32;
        break;
    case GGML_TYPE_Q6_K:
        is_u4 = false;
        weights_per_block = 16;
        break;
    default:
        throw std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // Extract quantized data
    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
        extract_q4_0_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q4_1:
        extract_q4_1_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q4_K:
        extract_q4_k_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q8_0:
        extract_q8_0_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q6_K:
        extract_q6_k_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q5_K:
        extract_q5_k_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    default:
        throw std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> weight_node;
    if (is_u4) {
        weight_node = make_int4_weights(weights, scales, zp, weights_per_block, use_bias);
    } else {
        weight_node = make_int8_weights(weights, scales, zp, weights_per_block, use_bias);
    }

    auto result = weight_node.get_node_shared_ptr();
    result->set_friendly_name(tensor->name);
    return result;
}

// Requantize weights to target format, writing to provided buffers
std::shared_ptr<ov::Node> requantize_to_buffers(const ggml_tensor * tensor,
                                                const void * data,
                                                ExtraQuantType requant_type,
                                                int64_t block_size,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);

    // First dequantize to F32
    std::vector<float> weights_f32(n_elements);
    ggml_get_type_traits(tensor->type)->to_float(data, weights_f32.data(), n_elements);

    // Handle F16 case - just convert and create constant
    if (requant_type == ExtraQuantType::F16) {
        ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.data(), weights.data(), n_elements);
        auto result = std::make_shared<ov::op::v0::Constant>(weights);
        result->set_friendly_name(tensor->name);
        return result;
    }

    // Requantize to target quantized format
    bool is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128);

    if (is_u4) {
        quantize_q4_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
    } else if (requant_type == ExtraQuantType::Q8_1_C) {
        quantize_q8_1(weights_f32.data(), weights, scales, zp, n_elements, block_size);
    } else {
        quantize_q8_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> weight_node;
    if (is_u4) {
        weight_node = make_int4_weights(weights, scales, zp, block_size);
    } else {
        weight_node = make_int8_weights(weights, scales, zp, block_size);
    }

    auto result = weight_node.get_node_shared_ptr();
    result->set_friendly_name(tensor->name);
    return result;
}

OvWeight process_weight_tensor(const ggml_tensor * tensor, const void * data, void * output_base_ptr, bool use_bias) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(data != nullptr);

    OvWeight result;

    // Get 2D shape for weights [rows, cols]
    ov::Shape node_shape = {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};

    // Handle F16/F32/BF16 weights
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        switch (tensor->type) {
        case GGML_TYPE_F32:
            element_type = ov::element::f32;
            break;
        case GGML_TYPE_F16:
            element_type = ov::element::f16;
            break;
        case GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            break;
        default:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        if (output_base_ptr && output_base_ptr != data) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_bytes = ggml_nbytes(tensor);
            memcpy(output_base_ptr, data, tensor_bytes);
            result.weights = ov::Tensor(element_type, node_shape, output_base_ptr);
        } else {
            result.weights = ov::Tensor(element_type, node_shape, data);
        }
        result.weight_node = std::make_shared<ov::op::v0::Constant>(result.weights);
        return result;
    }

    // Handle quantized weights
    if (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    result.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    const auto & layout = result.layout;
    if (layout.total_size == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    if (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_base_ptr = nullptr;
    }

    // F16 requant path - no separate scales/zp needed in result
    if (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        if (output_base_ptr) {
            result.weights = ov::Tensor(ov::element::f16, node_shape,
                                        static_cast<uint8_t *>(output_base_ptr) + layout.weights_offset);
        } else {
            result.weights = ov::Tensor(ov::element::f16, node_shape);
        }
        ov::Tensor dummy_scales, dummy_zp;  // Not used for F16
        result.weight_node =
            requantize_to_buffers(tensor, data, ExtraQuantType::F16, 0, result.weights, dummy_scales, dummy_zp);
        return result;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type weight_type = layout.is_symmetric ? (layout.is_u4 ? ov::element::i4 : ov::element::i8) :
                                                          (layout.is_u4 ? ov::element::u4 : ov::element::u8);
    ov::Shape scale_shape = {node_shape[0], node_shape[1] / layout.weights_per_block};

    if (output_base_ptr) {
        uint8_t * buf_base = static_cast<uint8_t *>(output_base_ptr);
        result.weights = ov::Tensor(weight_type, node_shape, buf_base + layout.weights_offset);
        result.scales = ov::Tensor(ov::element::f16, scale_shape, buf_base + layout.scales_offset);
        if (!layout.is_symmetric) {
            ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
            result.zp = ov::Tensor(zp_type, scale_shape, buf_base + layout.zp_offset);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } else {
        result.weights = ov::Tensor(weight_type, node_shape);
        result.scales = ov::Tensor(ov::element::f16, scale_shape);
        if (!layout.is_symmetric) {
            if (use_bias) {
                result.zp = ov::Tensor(ov::element::f16, scale_shape);
            } else {
                ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
                result.zp = ov::Tensor(zp_type, scale_shape);
            }
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    if (layout.is_requant && layout.requant_type.has_value()) {
        result.weight_node = requantize_to_buffers(tensor, data, layout.requant_type.value(), layout.weights_per_block,
                                                   result.weights, result.scales, result.zp);
    } else {
        result.weight_node =
            extract_quantized_weights(tensor, data, result.weights, result.scales, result.zp, use_bias);
    }

    return result;
}

void quantize_q4_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    assert(k % qk == 0);
    const int nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        for (int i = 0; i < nb; i++) {
            float amax = 0.0f;
            float max = 0.0f;
            for (int j = 0; j < qk; j++) {
                const float v = x[i * qk + j];
                if (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            const float d = max / -8;
            if (d == 0) {
                scales[i] = ov::float16(1.0f);
                if (i % 2 == 0) {
                    zp[i / 2] = 8;
                } else {
                    zp[i / 2] |= (8 << 4);
                }
                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                continue;
            }
            const float id = 1.0f / d;
            scales[i] = ov::float16(d);
            if (i % 2 == 0) {
                zp[i / 2] = 8;
            } else {
                zp[i / 2] |= (8 << 4);
            }
            for (int j = 0; j < qk / 2; ++j) {
                const float x0 = x[i * qk + 2 * j] * id;
                const float x1 = x[i * qk + 2 * j + 1] * id;
                const uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                const uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } else {
        // Symmetric: produce signed i4 values in [-8, 7]
        for (int i = 0; i < nb; i++) {
            float amax = 0.0f;
            float max = 0.0f;
            for (int j = 0; j < qk; j++) {
                const float v = x[i * qk + j];
                if (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            const float d = max / -8;
            if (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                memset(weights + i * qk / 2, 0, qk / 2);
                continue;
            }
            const float id = 1.0f / d;
            scales[i] = ov::float16(d);
            for (int j = 0; j < qk / 2; ++j) {
                const float x0 = x[i * qk + 2 * j] * id;
                const float x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (int) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (int) roundf(x1)));
                weights[i * qk / 2 + j] = (si0 & 0x0F) | ((si1 & 0x0F) << 4);
            }
        }
    }
}

void quantize_q8_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    assert(k % qk == 0);
    const int nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        for (int i = 0; i < nb; i++) {
            float amax = 0.0f;
            for (int j = 0; j < qk; j++) {
                const float v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            const float d = amax / 127.0f;
            const float id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            for (int j = 0; j < qk; ++j) {
                const float x0 = x[i * qk + j] * id;
                const int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } else {
        // Symmetric: store signed int8 values directly
        auto * signed_weights = reinterpret_cast<int8_t *>(weights);
        for (int i = 0; i < nb; i++) {
            float amax = 0.0f;
            for (int j = 0; j < qk; j++) {
                const float v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            const float d = amax / 127.0f;
            const float id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            for (int j = 0; j < qk; ++j) {
                const float x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

void quantize_q8_1(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    assert(k % qk == 0);
    const int nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto * zp = static_cast<uint8_t *>(zp_arr.data());
    for (int i = 0; i < nb; i++) {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        const float d = (max - min) / ((1 << 8) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        for (int j = 0; j < qk; ++j) {
            const float x0 = (x[i * qk + j] - min) * id;
            const uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}


/* Inlined utils.cpp */
/* Inlined utils.h */
#include "ggml-backend-impl-defs.inc"
#include "ggml.h.inc"
#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <openvino/core/partial_shape.hpp>
#include <optional>
#include <vector>

struct ModelParams {
    int ctx = -1;
    int ctx_swa = -1;
    int ctx_per_seq = -1;
    int ctx_per_seq_swa = -1;
    int n_seq = 1;
    int n_heads = -1;
    int n_heads_kv = -1;
    int head_size = -1;
    int32_t rope_params[15];
    std::vector<int> swa_layers;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(const ModelParams & other) const {
        return memcmp(rope_params, other.rope_params, sizeof(int32_t) * 15) == 0;
    }

    bool can_reuse_dynamically(const ModelParams & other) const { return same_rope_params(other); }

    bool can_reuse_statically(const ModelParams & other) const { return same_rope_params(other) && ctx == other.ctx; }

    bool kv_buffer_changed(const ModelParams & other) const { return kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

struct ComputeParams {
    int n_seq_active = 1;
    int seq_active_start = 0;
    int attention_size = -1;
    int attention_size_swa = -1;
    int input_len = -1;
    int token_len_per_seq = -1;
    int past_kv_len = -1;
    int output_len = 1;
};

class GgmlOvDecoder : public ov::frontend::ggml::GgmlDecoder {
public:
    struct NodeInfo {
        ggml_tensor * node;
        std::string node_name;
        std::string node_op_type;
        std::map<std::string, ggml_tensor *> node_inputs;
        std::vector<std::string> node_inputs_names;
        ggml_tensor * node_output;
        std::string node_output_name;
        int node_op_case = 0;
        void * data_addr;
    };
    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool is_prefill = false,
                  int prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights);

    virtual ov::Any get_attribute(const std::string & name) const override {
        return nullptr;
        GGML_UNUSED(name);
    }

    virtual ov::PartialShape get_input_shape(int node_idx, const std::string & name) const override;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string & name) const override;

    virtual ov::element::Type get_input_type(int node_idx, const std::string & name) const override;

    virtual size_t get_input_size() const override;

    virtual size_t get_input_size(int node_idx) const override;

    virtual void get_input_node(size_t input_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) const override {
        GGML_UNUSED(input_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    virtual std::vector<std::string> get_input_names(int node_idx) const override;

    virtual ov::PartialShape get_output_shape(int node_idx) const override;

    virtual ov::element::Type get_output_type(int node_idx) const override;

    virtual int32_t * get_input_op_params(int node_idx, const std::string & name) const override;

    virtual int32_t * get_output_op_params(int node_idx) const override;

    virtual std::vector<std::string> get_output_names(int node_idx) const override;

    virtual const std::string & get_op_type() const override;

    virtual const std::string & get_op_type(int node_idx) const override;

    virtual const std::string & get_op_name() const override;

    virtual const std::string & get_op_name(int node_idx) const override;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const override;

    ggml_tensor * get_input_ggml_tensor(const std::string & name) const { return m_inputs.at(name); }

    virtual int get_op_case(int node_idx) const override { return m_node_info_list[node_idx].node_op_case; }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_inputs() const override {
        return m_model_inputs;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_extra_inputs() const override {
        return m_model_extra_inputs;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Tensor>> & get_model_extra_input_values() const {
        return m_model_extra_input_values;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_weights() const override {
        return m_model_weights;
    }

    virtual std::vector<std::string> get_model_output_names() const override {
        return m_model_output_names;
    }

    const std::map<std::string, ggml_tensor *> & get_model_outputs() const { return m_model_outputs; }

    virtual int get_ctx_size() const { return m_model_params.ctx; }

    virtual int get_ctx_swa_size() const { return m_model_params.ctx_swa; }

    virtual int get_ctx_per_seq() const { return m_model_params.ctx_per_seq; }

    virtual int get_ctx_per_seq_swa() const { return m_model_params.ctx_per_seq_swa; }

    virtual int get_n_seq() const { return m_model_params.n_seq; }

    virtual int is_swa_layer(int layer) const override {
        return std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    int get_past_kv_len() const { return m_compute_params.past_kv_len; }

    int get_input_len() const { return m_compute_params.input_len; }

    virtual int32_t * get_rope_params() const override { return const_cast<int32_t *>(m_model_params.rope_params); }

    virtual std::map<std::string, std::string> get_kv_param_res_names() const override;

    virtual bool is_static() const override { return m_is_static; }

    virtual bool is_stateful() const override { return m_is_stateful; }

    ov::PartialShape get_graph_input_shape(const ggml_tensor * op, const ggml_tensor * input) const;

    static void dump_cgraph(const ggml_cgraph * cgraph, std::string & filename);

    static std::shared_ptr<ov::Node> create_weight_node(ggml_tensor * tensor, bool naive = false);

    static std::map<std::string, std::shared_ptr<ov::Node>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    const ggml_tensor * get_tensor_used_op(const ggml_tensor * tensor) const;

    const ggml_tensor * get_tensor_from_name(const std::string & name) const;

    void clear_model_weights() { m_model_weights.clear(); }

    static std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() const { return m_model_params; }

    ComputeParams get_compute_params() const { return m_compute_params; }

    void set_model_params(const ModelParams & model_params) { m_model_params = model_params; }

    void set_compute_params(const ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    int m_prefill_chunk_size = 0;

    static ov::Shape get_shape(const ggml_tensor * tensor);
    static std::vector<size_t> get_stride(const ggml_tensor * tensor);
    static ov::element::Type get_ov_type(const ggml_tensor * tensor);
    static std::string compute_op_type(const ggml_tensor * node);
    void add_extra_inputs();

    void update_io(ggml_cgraph * cgraph);

    inline static bool is_inp_tok(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    inline static bool is_inp_pos(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    inline static bool is_inp_emb(const ggml_tensor * tensor, const ggml_tensor * op) {
        return tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    inline static bool is_inp_mask(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]);
    }

    inline static bool is_rope_freqs_weight(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    inline static bool is_kvcache(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_SET_ROWS && op->src[2] == tensor;
    }

    inline static bool is_kv_idx(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    inline static bool is_output_idx(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE;
    }

    static std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) {
        if (is_inp_tok(tensor, op)) {
            return "inp_tokens";
        }
        if (is_inp_pos(tensor, op)) {
            return "inp_pos";
        }
        if (is_inp_emb(tensor, op)) {
            return "embd";
        }
        if (is_output_idx(tensor, op)) {
            return "inp_out_ids";
        }
        if (is_inp_mask(tensor, op)) {
            return std::string(tensor->name).find("swa") == std::string::npos ? "self_kq_mask" : "self_kq_mask_swa";
        }
        return tensor->name;
    }

private:
    void set_input_output();
    int compute_op_case(const ggml_tensor * node) const;
    bool node_is_used_as_src(const int node_idx);
    void compute_model_inputs();
    void compute_model_outputs();

    void validate_cgraph() const;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::Tensor>> m_model_extra_input_values;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::vector<std::string> m_model_output_names;
    std::vector<NodeInfo> m_node_info_list;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

void print_tensor_address_map(const ggml_cgraph * cgraph);

int extract_layer_from_name(const std::string & name);

#include "ggml-impl-defs.inc"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct graph_key {
    int n_nodes;
    std::string first_node_name;
    std::string last_node_name;

    graph_key(const ggml_cgraph * cgraph) : n_nodes(cgraph->n_nodes) {
        if (n_nodes > 0) {
            first_node_name = cgraph->nodes[0]->name;
            last_node_name = cgraph->nodes[n_nodes - 1]->name;
        }
    }

    bool operator==(const graph_key & other) const {
        return n_nodes == other.n_nodes && first_node_name == other.first_node_name &&
               last_node_name == other.last_node_name;
    }
};

struct graph_key_hash {
    size_t operator()(const graph_key & key) const {
        size_t h = std::hash<int>{}(key.n_nodes);
        if (key.n_nodes > 0) {
            h ^= std::hash<std::string>{}(key.first_node_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>{}(key.last_node_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct decoder_runtime_ctx {
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}
    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

struct ov_runtime_context {
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache_prefill;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_input_names_cache;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_output_names_cache;
    //TODO: Stateful is only supported for single request at a time.
    //      Simultanous stateful inference request support to be added.
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_input_name_map;
    std::atomic<int> backend_count;

    ov_runtime_context() :
        device("CPU"),
        stateful(false),
        stateful_kv_size(0),
        backend_count(0) {}

    void clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_input_names_cache.clear();
        ov_output_names_cache.clear();
    }
};

enum ggml_status ov_graph_compute(struct ggml_cgraph * cgraph, ggml_backend_t backend);

enum ggml_status ov_graph_compute_dynamic(struct ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);
enum ggml_status ov_graph_compute_static(struct ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);

size_t checksum(const void * data, size_t size);

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor);

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst);

template <typename T>
std::vector<T> pad_input(const T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_value) {
    std::vector<T> padded(padded_rows * padded_cols, pad_value);

    for (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        for (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    return padded;
}

template <typename T>
std::vector<T> pad_input(const ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_value) {
    return pad_input<T>(reinterpret_cast<const T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_value);
}

void set_zero_diagonal(std::vector<float> & matrix, size_t rows, size_t cols);

const ggml_tensor * get_inp_pos_tensor(struct ggml_cgraph * cgraph);

bool get_is_prefill(const ggml_tensor * inp_pos);

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                             const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index);

ov::Tensor create_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                   std::shared_ptr<ov::InferRequest> infer_request,
                                   int output_index,
                                   const ggml_tensor * ggml_tensor);

bool is_naive(struct ggml_cgraph * cgraph);

enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config);


#include "ggml-impl-defs.inc"
#include "ggml.h.inc"
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    FrontEnd();

    static std::shared_ptr<Model> convert(const InputModel::Ptr& model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <openvino/core/any.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include <unordered_map>
#include <vector>

// Suppress  deprecation warning for ov::Tensor::data()
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

enum ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    try {
        if (getenv("GGML_OPENVINO_DUMP_CGRAPH")) {
            std::string filename = "cgraph_ov.txt";
            GgmlOvDecoder::dump_cgraph(cgraph, filename);
        }

        const auto is_static = ggml_openvino_is_npu();

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);

        return is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } catch (const ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        return GGML_STATUS_FAILED;
    }
}

ov::Tensor create_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                   std::shared_ptr<ov::InferRequest> infer_request,
                                   int output_index,
                                   const ggml_tensor * ggml_tensor) {
    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    if (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
    } else {
        output_shape = ggml_decoder->get_shape(ggml_tensor);
    }

    ov::Tensor output_tensor(output_type, output_shape, ggml_tensor->data);
    return output_tensor;
}

enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
    auto & core = ov_singleton_core();
    const auto & config = ggml_openvino_get_compile_config();
    const auto & device = r_ctx->device;
    const auto & stateful = r_ctx->stateful;
    static auto is_static = false;

    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    graph_key key(cgraph);
    bool cache_hit;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto it = r_ctx->decoder_cache.find(key);
            cache_hit = it != r_ctx->decoder_cache.end();
            if (cache_hit) {
                entry = it->second;
            } else {
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[key] = entry;
            }
        }

        std::lock_guard<std::mutex> lock(*(entry->mutex));

        if (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            cache_hit = old_m_params.can_reuse_dynamically(m_params);
        }

        if (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            if (old_m_params.kv_buffer_changed(m_params)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(key);
            }

            if (stateful) {
                const auto * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                auto pos_shape = ggml_decoder->get_shape(inp_pos);
                if (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } else if (r_ctx->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } else {
                    auto states = infer_request->query_state();
                    for (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();
                        if (static_cast<uint32_t>(pos_data[0]) > r_ctx->stateful_kv_size) {
                            std::string state_name;
                            try {
                                state_name = r_ctx->kv_state_input_name_map.at(state.get_name());
                            } catch (...) {
                                GGML_LOG_ERROR("GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                return GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_input_tensor(ggml_decoder, state_name);
                            kv_tensor.set_shape({state_tensor_shape[0], kv_tensor.get_shape()[2],
                                                 state_tensor_shape[2], state_tensor_shape[3]});
                           state_tensor = kv_tensor;
                           state_tensor_shape = state_tensor.get_shape();
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end = {state_tensor_shape[0], static_cast<uint32_t>(pos_data[0]),
                                              state_tensor_shape[2], state_tensor_shape[3]};
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.set_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_size = pos_data[0] + 1;
                }
            }

            decoder_end_time = ggml_time_us();
            conversion_end_time = decoder_end_time;
            compile_end_time = decoder_end_time;
        } else {
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(key);
            }

            std::shared_ptr<ov::Model> model;
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static, stateful);
            decoder_end_time = ggml_time_us();

            auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
            model = ov::frontend::ggml::FrontEnd::convert(input_model);
            ggml_decoder->clear_model_weights();
            conversion_end_time = ggml_time_us();

            if (getenv("GGML_OPENVINO_DUMP_IR")) {
                char timestamped_filename[64];
                auto timestamp = (long long) ggml_time_us();
                snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
                ov::serialize(model, timestamped_filename);
            }

            ov::CompiledModel compiled_model;
            auto remote_context = ggml_openvino_get_remote_context();
            if (remote_context.has_value()) {
                compiled_model = core.compile_model(model, remote_context.value(), config);
            } else {
                compiled_model = core.compile_model(model, device, config);
            }
            compile_end_time = ggml_time_us();
            infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
            entry->ptr = ggml_decoder;

            std::vector<std::string> ov_input_names;
            std::vector<std::string> ov_output_names;
            for (const auto & ov_param : model->get_parameters()) {
                ov_input_names.push_back(ov_param->get_friendly_name());
            }
            for (const auto & ov_output : model->get_results()) {
                ov_output_names.push_back(ov_output->get_friendly_name());
            }

            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[key] = infer_request;
                r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
                r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
            }

            if (stateful) {
                const auto * inp_pos = get_inp_pos_tensor(cgraph);
                auto pos_shape = ggml_decoder->get_shape(inp_pos);
                r_ctx->stateful_kv_size = pos_shape[3];
                const auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                for (const auto& pair : kv_param_res_names) {
                    r_ctx->kv_state_input_name_map[pair.first+pair.second] = pair.first;
                }
            }
        }

        std::vector<std::string> ov_input_names;
        std::vector<std::string> ov_output_names;
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            ov_input_names = r_ctx->ov_input_names_cache[key];
            ov_output_names = r_ctx->ov_output_names_cache[key];
        }

        for (size_t i = 0; i < ov_input_names.size(); i++) {
            auto param_name = ov_input_names[i];
            auto input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        for (size_t i = 0; i < ov_output_names.size(); i++) {
            auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names[i]);
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        infer_request->infer();
        infer_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
            for (size_t i = 0; i < ov_output_names.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }

        if (getenv("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %ld ms \n", (decoder_end_time - start_time) / 1000);
            if (!cache_hit) {
                GGML_LOG_INFO("  - Graph conversion time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
                GGML_LOG_INFO("  - Graph compile time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
            }
            GGML_LOG_INFO("  - Graph inference time: %ld ms \n", (infer_end_time - compile_end_time) / 1000);
        }
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
    auto & core = ov_singleton_core();

    auto get_prefill_chunk_size = [] {
        const char * chunk_size_str = getenv("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
        if (chunk_size_str && atoi(chunk_size_str) > 0) {
            return atoi(chunk_size_str);
        }
        return 256;
    };

    static std::string device = "NPU";
    static auto is_static = true;
    static auto stateful = false;
    static auto prefill_chunk_size = get_prefill_chunk_size();
    const auto & config = ggml_openvino_get_compile_config();

    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    const auto * inp_pos = get_inp_pos_tensor(cgraph);
    const auto is_prefill = get_is_prefill(inp_pos);
    graph_key key(cgraph);
    bool cache_hit;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        auto it = r_ctx->decoder_cache.find(key);
        cache_hit = it != r_ctx->decoder_cache.end();
        if (cache_hit) {
            entry = it->second;
        } else {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[key] = entry;
        }
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));

    if (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    if (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        if (old_m_params.kv_buffer_changed(m_params)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(key) : r_ctx->infer_request_cache.at(key);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } else {
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(key);
            r_ctx->infer_request_cache_prefill.erase(key);
        }

        std::shared_ptr<ov::Model> model;
        auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

        auto ggml_decoder_prefill = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights,
                                                                    is_static, stateful, true, prefill_chunk_size);
        auto ggml_decoder_decode = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static,
                                                                   stateful, false, prefill_chunk_size);
        decoder_end_time = ggml_time_us();

        auto input_model_prefill = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder_prefill);
        auto input_model_decode = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder_decode);

        auto model_prefill = ov::frontend::ggml::FrontEnd::convert(input_model_prefill);
        ggml_decoder_prefill->clear_model_weights();
        auto model_decode = ov::frontend::ggml::FrontEnd::convert(input_model_decode);
        ggml_decoder_decode->clear_model_weights();
        conversion_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DUMP_IR")) {
            char timestamped_filename[64];
            auto timestamp = (long long) ggml_time_us();
            snprintf(timestamped_filename, sizeof(timestamped_filename), "model_prefill_%lld.xml", timestamp);
            ov::serialize(model_prefill, timestamped_filename);
            snprintf(timestamped_filename, sizeof(timestamped_filename), "model_decode_%lld.xml", timestamp);
            ov::serialize(model_decode, timestamped_filename);
        }

        ov::CompiledModel compiled_model_prefill;
        ov::CompiledModel compiled_model_decode;
        auto remote_context = ggml_openvino_get_remote_context();
        if (remote_context.has_value()) {
            compiled_model_prefill = core.compile_model(model_prefill, remote_context.value(), config);
            compiled_model_decode = core.compile_model(model_decode, remote_context.value(), config);
        } else {
            compiled_model_prefill = core.compile_model(model_prefill, device, config);
            compiled_model_decode = core.compile_model(model_decode, device, config);
        }

        auto infer_request_prefill = std::make_shared<ov::InferRequest>(compiled_model_prefill.create_infer_request());
        auto infer_request_decode = std::make_shared<ov::InferRequest>(compiled_model_decode.create_infer_request());
        compile_end_time = ggml_time_us();

        model = is_prefill ? model_prefill : model_decode;
        ggml_decoder = is_prefill ? ggml_decoder_prefill : ggml_decoder_decode;
        infer_request = is_prefill ? infer_request_prefill : infer_request_decode;
        entry->ptr = ggml_decoder;

        std::vector<std::string> ov_input_names;
        std::vector<std::string> ov_output_names;
        for (const auto & ov_param : model->get_parameters()) {
            ov_input_names.push_back(ov_param->get_friendly_name());
        }
        for (const auto & ov_output : model->get_results()) {
            ov_output_names.push_back(ov_output->get_friendly_name());
        }

        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache_prefill[key] = infer_request_prefill;
            r_ctx->infer_request_cache[key] = infer_request_decode;
            r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
            r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
        }
    }

    std::vector<std::string> ov_input_names_local;
    std::vector<std::string> ov_output_names_local;
    {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        ov_input_names_local = r_ctx->ov_input_names_cache[key];
        ov_output_names_local = r_ctx->ov_output_names_cache[key];
    }

    if (is_prefill) {
        auto inp_len = inp_pos->ne[0];
        for (int chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            for (size_t i = 0; i < ov_input_names_local.size(); i++) {
                auto param_name = ov_input_names_local[i];
                auto input_tensor = get_ov_input_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_input_tensor(i, input_tensor);

                if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                    const auto input_tensor = infer_request->get_input_tensor(i);
                    print_input_tensor_info(param_name, input_tensor);
                }
            }

            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names_local[i]);
                auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
                infer_request->set_output_tensor(i, output_tensor);
            }

            infer_request->infer();

            if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
                for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    const auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } else {
        for (size_t i = 0; i < ov_input_names_local.size(); i++) {
            auto param_name = ov_input_names_local[i];
            auto input_tensor = get_ov_input_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                const auto input_tensor = infer_request->get_input_tensor(i);
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        for (size_t i = 0; i < ov_output_names_local.size(); i++) {
            auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names_local[i]);
            auto output_tensor = create_ov_output_tensor(ggml_decoder, infer_request, i, ggml_tensor);
            infer_request->set_output_tensor(i, output_tensor);
        }

        infer_request->infer();
        infer_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
        }
    }

    if (getenv("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %ld ms \n", (decoder_end_time - start_time) / 1000);
        if (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
            GGML_LOG_INFO("  - Graph compile time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
        }
        GGML_LOG_INFO("  - Graph inference time: %ld ms \n", (infer_end_time - compile_end_time) / 1000);
    }

    return GGML_STATUS_SUCCESS;
}

bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;
    int count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    return count < naive_graph_size_threshold;
}

enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    bool naive = true;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
    if (getenv("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    } else {
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    }
    if (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), config).create_infer_request());
    } else {
        infer_request =
            std::make_shared<ov::InferRequest>(core.compile_model(model, device, config).create_infer_request());
    }

    auto ov_params = model->get_parameters();
    for (size_t i = 0; i < ov_params.size(); i++) {
        auto param_name = ov_params[i]->get_friendly_name();
        auto input_tensor = get_ov_input_tensor(decoder, param_name);
        infer_request->set_input_tensor(i, input_tensor);
    }

    auto ov_results = model->get_results();
    for (size_t i = 0; i < ov_results.size(); i++) {
        auto * ggml_tensor = decoder->get_model_outputs().at(ov_results[i]->get_friendly_name());
        auto output_tensor = create_ov_output_tensor(decoder, infer_request, i, ggml_tensor);
        infer_request->set_output_tensor(i, output_tensor);
    }

    infer_request->infer();
    return GGML_STATUS_SUCCESS;
}

namespace {
ov::Tensor convert_ggml_input_to_ov(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & name) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(name);

    if (ggml_tensor->extra != nullptr) {
        // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        if (extra_base->type != ggml_openvino_extra_base::Type::TENSOR) {
            throw std::runtime_error("ggml tensor extra is not of type TENSOR for input: " + name);
        }
        auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
        return *tensor_extra->tensor;
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    auto * input_data = ggml_tensor->data;
    ov::Shape input_shape;
    if (ggml_tensor->op == GGML_OP_VIEW) {
        // This case is added to make test-backend-ops work
        input_shape = ggml_decoder->get_shape(ggml_tensor->view_src);
    } else {
        input_shape = ggml_decoder->get_shape(ggml_tensor);
    }
    auto input_tensor = ov::Tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape, input_data);
    return input_tensor;
}
}  // namespace

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name) {
    ov::Tensor input_tensor;
    if (ggml_decoder->get_model_extra_inputs().find(param_name) != ggml_decoder->get_model_extra_inputs().end()) {
        input_tensor = *ggml_decoder->get_model_extra_input_values().at(param_name);
    } else {
        input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
    }
    return input_tensor;
}

ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                             const std::string & param_name) {
    // NPU decoding stage
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    if (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        assert(ggml_tensor->ne[0] == 1);
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        if (ggml_tensor->type == GGML_TYPE_I32) {
            *input_tensor.data<int32_t>() = *((int32_t *) ggml_tensor->data);
        } else if (ggml_tensor->type == GGML_TYPE_I64) {
            *input_tensor.data<int64_t>() = *((int64_t *) ggml_tensor->data);
        } else {
            throw std::runtime_error("Unexpected tensor type for " + param_name);
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        assert(ggml_tensor->ne[0] == 1);
        assert(inp_out_id == 0);
        *input_tensor.data<int32_t>() = inp_out_id;
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = ggml_decoder->get_ctx_size();
        std::vector<float> padded_data = pad_input<float>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index) {
    // NPU prompt processing stage
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    const size_t input_len = ggml_decoder->get_input_len();
    const size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    const size_t chunk_valid_size = std::min(chunk_size, input_len - chunk_index * chunk_size);
    const size_t chunk_pad_size = chunk_size - chunk_valid_size;

    if (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        ov::Shape input_shape = {1, 1, 1, chunk_size};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        void * input_data = (char *) ggml_tensor->data + chunk_index * chunk_size * element_size;
        std::memcpy(input_tensor.data(), input_data, chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        if (chunk_pad_size > 0) {
            if (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_value =
                    *((int32_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int32_t * output_data = input_tensor.data<int32_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_value + 1);
            } else if (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_value =
                    *((int64_t *) ggml_tensor->data + (chunk_index * chunk_size + chunk_valid_size - 1));
                int64_t * output_data = input_tensor.data<int64_t>();
                std::fill(output_data + chunk_valid_size, output_data + chunk_size, last_value + 1);
            } else {
                throw std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape input_shape = {1, 1, 1, output_len};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        if (ggml_tensor->ne[0] == 0) {
            *input_tensor.data<int32_t>() = 0;
        } else {
            auto * data_addr = input_tensor.data<int32_t>();
            for (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        float * ggml_data = (float *) ggml_tensor->data + chunk_index * chunk_size * cols;
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        std::vector<float> padded_data =
            pad_input<float>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

size_t checksum(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    return sum;
}

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        if (name.find("self_kq_mask") == std::string::npos) {
            std::cout << *(tensor.data<float>()) << std::endl;
        } else {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            auto * data = tensor.data<float>();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    float val = data[i * cols + j];
                    if (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } else {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << std::endl;
            }
        }

        break;
    }
    case ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << std::endl;
        break;
    case ov::element::i32:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    case ov::element::i64:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    default:
        break;
    }
}

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << std::endl;

    auto print_float_stats = [](const std::string & type_name, size_t size, auto get_value) {
        if (size == 0) {
            return;
        }

        float first = get_value(0);
        float min = first;
        float max = first;
        double sum = first;

        for (size_t i = 1; i < size; ++i) {
            float v = get_value(i);
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            sum += v;
        }
        double mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << std::endl;
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << std::endl;
    };

    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        const float * data = tensor.data<float>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { return data[i]; });
        break;
    }
    case ov::element::f16: {
        const ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { return static_cast<float>(data[i]); });
        break;
    }
    default:
        break;
    }
}

void set_zero_diagonal(std::vector<float> & matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = 0.0f;
    }
}

const ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            if (src == nullptr) {
                break;
            }
            if (GgmlOvDecoder::is_inp_pos(src, op)) {
                return src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    throw std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

bool get_is_prefill(const ggml_tensor * inp_pos) {
    return inp_pos->ne[0] > 1;
}

#pragma GCC diagnostic pop


/* Inlined ggml-decoder.cpp */
#include "ggml.h.inc"
#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <openvino/core/partial_shape.hpp>
#include <optional>
#include <vector>

struct ModelParams {
    int ctx = -1;
    int ctx_swa = -1;
    int ctx_per_seq = -1;
    int ctx_per_seq_swa = -1;
    int n_seq = 1;
    int n_heads = -1;
    int n_heads_kv = -1;
    int head_size = -1;
    int32_t rope_params[15];
    std::vector<int> swa_layers;

    std::vector<std::string> kv_names;
    size_t kv_buffer_ctx_id = 0;

    bool same_rope_params(const ModelParams & other) const {
        return memcmp(rope_params, other.rope_params, sizeof(int32_t) * 15) == 0;
    }

    bool can_reuse_dynamically(const ModelParams & other) const { return same_rope_params(other); }

    bool can_reuse_statically(const ModelParams & other) const { return same_rope_params(other) && ctx == other.ctx; }

    bool kv_buffer_changed(const ModelParams & other) const { return kv_buffer_ctx_id != other.kv_buffer_ctx_id; }
};

struct ComputeParams {
    int n_seq_active = 1;
    int seq_active_start = 0;
    int attention_size = -1;
    int attention_size_swa = -1;
    int input_len = -1;
    int token_len_per_seq = -1;
    int past_kv_len = -1;
    int output_len = 1;
};

class GgmlOvDecoder : public ov::frontend::ggml::GgmlDecoder {
public:
    struct NodeInfo {
        ggml_tensor * node;
        std::string node_name;
        std::string node_op_type;
        std::map<std::string, ggml_tensor *> node_inputs;
        std::vector<std::string> node_inputs_names;
        ggml_tensor * node_output;
        std::string node_output_name;
        int node_op_case = 0;
        void * data_addr;
    };
    // Graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph,
                  ModelParams & model_params,
                  ComputeParams & compute_params,
                  std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                  bool is_static,
                  bool is_stateful = false,
                  bool is_prefill = false,
                  int prefill_chunk_size = 256);

    // Naive graph decoder
    GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights);

    virtual ov::Any get_attribute(const std::string & name) const override {
        return nullptr;
        GGML_UNUSED(name);
    }

    virtual ov::PartialShape get_input_shape(int node_idx, const std::string & name) const override;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string & name) const override;

    virtual ov::element::Type get_input_type(int node_idx, const std::string & name) const override;

    virtual size_t get_input_size() const override;

    virtual size_t get_input_size(int node_idx) const override;

    virtual void get_input_node(size_t input_port_idx,
                                std::string & producer_name,
                                std::string & producer_output_port_name,
                                size_t & producer_output_port_index) const override {
        GGML_UNUSED(input_port_idx);
        GGML_UNUSED(producer_name);
        GGML_UNUSED(producer_output_port_name);
        GGML_UNUSED(producer_output_port_index);
    }

    virtual std::vector<std::string> get_input_names(int node_idx) const override;

    virtual ov::PartialShape get_output_shape(int node_idx) const override;

    virtual ov::element::Type get_output_type(int node_idx) const override;

    virtual int32_t * get_input_op_params(int node_idx, const std::string & name) const override;

    virtual int32_t * get_output_op_params(int node_idx) const override;

    virtual std::vector<std::string> get_output_names(int node_idx) const override;

    virtual const std::string & get_op_type() const override;

    virtual const std::string & get_op_type(int node_idx) const override;

    virtual const std::string & get_op_name() const override;

    virtual const std::string & get_op_name(int node_idx) const override;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const override;

    ggml_tensor * get_input_ggml_tensor(const std::string & name) const { return m_inputs.at(name); }

    virtual int get_op_case(int node_idx) const override { return m_node_info_list[node_idx].node_op_case; }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_inputs() const override {
        return m_model_inputs;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_extra_inputs() const override {
        return m_model_extra_inputs;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Tensor>> & get_model_extra_input_values() const {
        return m_model_extra_input_values;
    }

    virtual const std::map<std::string, std::shared_ptr<ov::Node>> & get_model_weights() const override {
        return m_model_weights;
    }

    virtual std::vector<std::string> get_model_output_names() const override {
        return m_model_output_names;
    }

    const std::map<std::string, ggml_tensor *> & get_model_outputs() const { return m_model_outputs; }

    virtual int get_ctx_size() const { return m_model_params.ctx; }

    virtual int get_ctx_swa_size() const { return m_model_params.ctx_swa; }

    virtual int get_ctx_per_seq() const { return m_model_params.ctx_per_seq; }

    virtual int get_ctx_per_seq_swa() const { return m_model_params.ctx_per_seq_swa; }

    virtual int get_n_seq() const { return m_model_params.n_seq; }

    virtual int is_swa_layer(int layer) const override {
        return std::find(m_model_params.swa_layers.begin(), m_model_params.swa_layers.end(), layer) !=
               m_model_params.swa_layers.end();
    }

    int get_past_kv_len() const { return m_compute_params.past_kv_len; }

    int get_input_len() const { return m_compute_params.input_len; }

    virtual int32_t * get_rope_params() const override { return const_cast<int32_t *>(m_model_params.rope_params); }

    virtual std::map<std::string, std::string> get_kv_param_res_names() const override;

    virtual bool is_static() const override { return m_is_static; }

    virtual bool is_stateful() const override { return m_is_stateful; }

    ov::PartialShape get_graph_input_shape(const ggml_tensor * op, const ggml_tensor * input) const;

    static void dump_cgraph(const ggml_cgraph * cgraph, std::string & filename);

    static std::shared_ptr<ov::Node> create_weight_node(ggml_tensor * tensor, bool naive = false);

    static std::map<std::string, std::shared_ptr<ov::Node>> create_weight_nodes(ggml_cgraph * cgraph,
                                                                                bool naive = false);

    const ggml_tensor * get_tensor_used_op(const ggml_tensor * tensor) const;

    const ggml_tensor * get_tensor_from_name(const std::string & name) const;

    void clear_model_weights() { m_model_weights.clear(); }

    static std::pair<ModelParams, ComputeParams> compute_llm_params(ggml_cgraph * cgraph, bool is_static);

    ModelParams get_model_params() const { return m_model_params; }

    ComputeParams get_compute_params() const { return m_compute_params; }

    void set_model_params(const ModelParams & model_params) { m_model_params = model_params; }

    void set_compute_params(const ComputeParams & compute_params) { m_compute_params = compute_params; }

    bool m_is_static = false;
    bool m_is_stateful = false;
    bool m_is_prefill = false;
    bool m_naive = false;
    int m_prefill_chunk_size = 0;

    static ov::Shape get_shape(const ggml_tensor * tensor);
    static std::vector<size_t> get_stride(const ggml_tensor * tensor);
    static ov::element::Type get_ov_type(const ggml_tensor * tensor);
    static std::string compute_op_type(const ggml_tensor * node);
    void add_extra_inputs();

    void update_io(ggml_cgraph * cgraph);

    inline static bool is_inp_tok(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op == GGML_OP_NONE;
    }

    inline static bool is_inp_pos(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_ROPE && tensor == op->src[1];
    }

    inline static bool is_inp_emb(const ggml_tensor * tensor, const ggml_tensor * op) {
        return tensor->op == GGML_OP_GET_ROWS && op->op == GGML_OP_RMS_NORM;
    }

    inline static bool is_inp_mask(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_CPY || (op->op == GGML_OP_FLASH_ATTN_EXT && tensor == op->src[3]);
    }

    inline static bool is_rope_freqs_weight(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_ROPE && tensor == op->src[2];
    }

    inline static bool is_kvcache(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_SET_ROWS && op->src[2] == tensor;
    }

    inline static bool is_kv_idx(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_SET_ROWS && op->src[1] == tensor;
    }

    inline static bool is_output_idx(const ggml_tensor * tensor, const ggml_tensor * op) {
        return op->op == GGML_OP_GET_ROWS && tensor == op->src[1] && op->src[0]->op != GGML_OP_NONE;
    }

    static std::string get_graph_input_ov_name(const ggml_tensor * tensor, const ggml_tensor * op) {
        if (is_inp_tok(tensor, op)) {
            return "inp_tokens";
        }
        if (is_inp_pos(tensor, op)) {
            return "inp_pos";
        }
        if (is_inp_emb(tensor, op)) {
            return "embd";
        }
        if (is_output_idx(tensor, op)) {
            return "inp_out_ids";
        }
        if (is_inp_mask(tensor, op)) {
            return std::string(tensor->name).find("swa") == std::string::npos ? "self_kq_mask" : "self_kq_mask_swa";
        }
        return tensor->name;
    }

private:
    void set_input_output();
    int compute_op_case(const ggml_tensor * node) const;
    bool node_is_used_as_src(const int node_idx);
    void compute_model_inputs();
    void compute_model_outputs();

    void validate_cgraph() const;

    ggml_cgraph * m_cgraph = nullptr;
    std::map<std::string, ggml_tensor *> m_inputs;

    std::map<std::string, std::shared_ptr<ov::Node>> m_model_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_extra_inputs;
    std::map<std::string, std::shared_ptr<ov::Tensor>> m_model_extra_input_values;
    std::map<std::string, std::shared_ptr<ov::Node>> m_model_weights;
    std::map<std::string, ggml_tensor *> m_model_outputs;
    std::vector<std::string> m_model_output_names;
    std::vector<NodeInfo> m_node_info_list;

    ModelParams m_model_params;
    ComputeParams m_compute_params;
};

void print_tensor_address_map(const ggml_cgraph * cgraph);

int extract_layer_from_name(const std::string & name);


#include "ggml-backend-impl-defs.inc"
#include "ggml-backend.h.inc"
#include "ggml-openvino.h.inc"

#include <ggml-impl-defs.inc>
#include <ggml.h.inc>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <openvino/core/dimension.hpp>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             bool is_static,
                             bool is_stateful,
                             bool is_prefill,
                             int prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(false),
    m_prefill_chunk_size(prefill_chunk_size),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    if (auto * env = getenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS"); env && std::string(env) != "0") {
#ifdef _WIN32
        _putenv_s("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS", "");
#else
        unsetenv("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS");
#endif
        print_tensor_address_map(cgraph);
    }

    validate_cgraph();

    set_input_output();
    compute_model_inputs();
    compute_model_outputs();

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }

    add_extra_inputs();
}

void GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_outputs.clear();
    m_node_info_list.clear();
    set_input_output();
    compute_model_inputs();
    compute_model_outputs();
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = true;
    set_input_output();
    compute_model_inputs();
    compute_model_outputs();
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }
}

void GgmlOvDecoder::set_input_output() {
    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto node = m_cgraph->nodes[node_n];

        NodeInfo current_node_info;
        auto node_name = std::string(node->name);
        auto node_output_name = node_name;
        auto * node_output = node;
        if (node->op == GGML_OP_SET_ROWS) {
            // SET_ROWS updates the tensor in place. For later ov op that uses the
            // the view_src of SET_ROWS, we need to make sure they get the updated tensor
            // by putting the view_src name in the tensor_map in
            // <openvino>/src/frontends/ggml/src/translate_session.cpp
            node_output_name = std::string(node->view_src->name);
            node_output = node->view_src;
        }

        current_node_info.node = node;
        current_node_info.node_name = node_name;
        current_node_info.node_output = node_output;
        current_node_info.node_output_name = node_output_name;
        current_node_info.node_op_case = 0;
        current_node_info.data_addr = node->data;

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            auto src_name = std::string(src->name);
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_graph_input_ov_name(src, node);
            }
            current_node_info.node_inputs[src_name] = src;
            current_node_info.node_inputs_names.push_back(src_name);
        }

        m_node_info_list.push_back(current_node_info);
    }
}

int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {
    int op_case = 0;
    switch (node->op) {
    case GGML_OP_RESHAPE: {
        auto * src = node->src[0];
        if (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == node->ne[0] && src->src[0]->ne[1] == node->ne[1]) {
            op_case = 4;
        } else if (node->ne[0] * node->ne[1] == src->ne[0]) {
            op_case = 1;
        } else if (src->ne[0] * src->ne[1] == node->ne[0]) {
            op_case = 2;
            if (src->ne[2] * src->ne[3] == node->ne[1]) {
                op_case = 5;
            }
        } else if (src->ne[0] * src->ne[1] == node->ne[1]) {
            op_case = 3;
        } else if (src->ne[1] * src->ne[2] == node->ne[1]) {
            op_case = 6;
        }
        break;
    }
    case GGML_OP_CONT: {
        if (node->src[0]->op == GGML_OP_PERMUTE) {
            op_case = 1;
        } else if (node->src[0]->op == GGML_OP_TRANSPOSE) {
            op_case = 2;
        } else if (node->src[0]->op == GGML_OP_VIEW) {
            op_case = 3;
        }
        break;
    }
    case GGML_OP_PERMUTE: {
        if (node->src[0]->op != GGML_OP_VIEW) {
            op_case = 1;
        } else if (node->src[0]->src[0]->op == GGML_OP_NONE) {
            // kv cache tensor
            std::string src_name(node->view_src->name);
            int layer = extract_layer_from_name(src_name);
            if (!is_swa_layer(layer)) {
                op_case = 2;
            } else {
                op_case = 3;
            }
        } else {
            // rope'ed query tensor
            op_case = 4;
        }
        break;
    }
    case GGML_OP_MUL_MAT: {
        if (node->src[0]->op == GGML_OP_CONT && node->src[0]->src[0]->op == GGML_OP_TRANSPOSE) {
            op_case = 2;
        } else if (node->src[0]->op == GGML_OP_VIEW && node->src[1]->op == GGML_OP_VIEW) {
            op_case = 3;
        }
        break;
    }
    case GGML_OP_GET_ROWS: {
        if (node->src[1]->op == GGML_OP_VIEW) {
            op_case = 2;
        }
        break;
    }
    case GGML_OP_ROPE: {
        const int mode = node->op_params[2];
        switch (mode) {
       case GGML_ROPE_TYPE_NEOX: {
            op_case = 0x00010000;
            break;
        }
       case GGML_ROPE_TYPE_IMROPE: {
            op_case = 0x00020000;
            break;
        }
        default:
            op_case = 0x00000000;
            break;
        }
        if (node->src[0]->op == GGML_OP_VIEW) {
            op_case = (op_case | 0x00000002);
        }
        break;
    }
    case GGML_OP_VIEW: {
        if (node->src[0]->op == GGML_OP_VIEW) {
            auto * src = node->src[0];
            if (ggml_nelements(node) != ggml_nelements(src)) {
                throw std::runtime_error("Unsupported VIEW case");
            }
            op_case = 2;
        }
        {
            auto * src = node->src[0];
            if ((ggml_nelements(node) != ggml_nelements(src)) && m_naive) {
                // Compare each dimension of node and src, if only one dimension differs then op_case=3
                int diff_count = 0;
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->ne[i] != src->ne[i]) {
                        diff_count++;
                    }
                }
                if (diff_count == 1) {
                    op_case = 3;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    return op_case;
}

int extract_layer_from_name(const std::string & name) {
    size_t pos1 = name.find("_l");
    assert(pos1 != std::string::npos);
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    if (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    int layer = std::stoi(layer_str);
    return layer;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, bool is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        auto * node = cgraph->nodes[i];
        std::string name = std::string(node->name);
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
            model_params.n_heads = node->src[0]->ne[2];
            model_params.n_heads_kv = node->src[1]->ne[2];
            model_params.head_size = node->src[0]->ne[0];
            compute_params.input_len = node->src[0]->ne[1];

            auto * cache_k_perm = node->src[1];
            if (cache_k_perm->op == GGML_OP_CPY) {
                cache_k_perm = cache_k_perm->src[0];
            }
            assert(cache_k_perm->op == GGML_OP_PERMUTE);
            auto * cache_k_view = cache_k_perm->src[0];
            assert(cache_k_view->op == GGML_OP_VIEW);

            auto * cache_k = cache_k_view->src[0];
            int layer = extract_layer_from_name(cache_k->name);
            auto * mask = node->src[3];
            std::string mask_name(mask->name);

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            if (mask_name.find("swa") != std::string::npos) {
                model_params.swa_layers.push_back(layer);
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } else {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, sizeof(size_t));
            compute_params.seq_active_start = offset / seq_size;
            compute_params.token_len_per_seq = node->ne[2];

            if (mask_name.find("swa") != std::string::npos) {
                compute_params.attention_size_swa = mask->ne[0];
            } else {
                compute_params.attention_size = mask->ne[0];
            }
            if (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
            break;
        }
        if (node->op == GGML_OP_ROPE) {
            memcpy(model_params.rope_params, node->op_params, sizeof(int32_t) * 15);
        }
    }
    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    // for NPU, output_len is always 1 except for llama-perplexity
    if (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    model_params.ctx_swa = model_params.ctx_per_seq_swa * model_params.n_seq;
    return {model_params, compute_params};
}

void GgmlOvDecoder::validate_cgraph() const {
    if (m_model_params.n_seq > 1 && m_is_static == true) {
        throw std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_input_shape(const ggml_tensor * op, const ggml_tensor * input) const {
    if (m_naive) {
        return input!= nullptr ? ov::PartialShape{get_shape(input)} : ov::PartialShape{get_shape(op)};
    }
    auto name = std::string(input->name);
    ov::PartialShape input_shape;

    if (is_inp_tok(input, op) || is_inp_pos(input, op)) {
        // tokens or positions
        int len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else if (is_output_idx(input, op)) {
        // output index
        input_shape = ov::PartialShape{1, 1, 1, m_is_static ? m_compute_params.output_len : -1};

    } else if (is_inp_mask(input, op)) {
        // mask
        if (m_is_static) {
            input_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } else if (m_is_stateful) {
            input_shape = ov::PartialShape{1, 1, -1, -1};
        } else {
            input_shape = ov::PartialShape{-1, 1, -1, -1};
        }

    } else if (is_kvcache(input, op)) {
        // kvcache
        input_shape = ov::PartialShape{get_shape(input)};
        if (!m_is_static) {
            // do not fix ctx size to make llama-bench work across test params
            input_shape[2] = -1;
        }
        if (is_stateful()) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size].
            assert(input_shape.size() == 4 && input_shape[0] == 1 && input_shape[1] == 1 &&
                   input_shape[2].is_dynamic() &&
                   input_shape[3] == (m_model_params.n_heads_kv * m_model_params.head_size));
            input_shape = {input_shape[0], ov::Dimension::dynamic(), m_model_params.n_heads_kv,
                           m_model_params.head_size};
        }

    } else if (is_kv_idx(input, op)) {
        // kv update index
        int len = m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1;
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else {
        input_shape = ov::PartialShape{get_shape(input)};
    }
    return input_shape;
}

void GgmlOvDecoder::add_extra_inputs() {
    // Extra inputs:
    // 1. `attention_size`, used in FLASH_ATTN where the shape of the matmul's are 256 aligned,
    //     see llama_kv_cache_unified::get_n_kv and llama_kv_cache_unified::get_padding.
    // 2. `n_seq_active` and `seq_active_start`, used in FLASH_ATTN_EXT to indicate the active sequences in the batch

    auto create_1d_input = [this](const std::string & name, int64_t value) {
        if (m_is_static) {
            auto constant =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{value});
            constant->set_friendly_name(name);
            m_model_extra_inputs[name] = constant;
        } else {
            auto param_node = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{1});
            param_node->set_friendly_name(name);
            param_node->output(0).get_tensor().set_names({name});
            m_model_extra_inputs[name] = param_node;

            auto tensor = std::make_shared<ov::Tensor>(ov::element::i64, ov::Shape{1});
            *tensor->data<int64_t>() = value;
            m_model_extra_input_values[name] = tensor;
        }
    };

    create_1d_input("attention_size", m_compute_params.attention_size);
    if (m_compute_params.attention_size_swa != -1) {
        create_1d_input("attention_size_swa", m_compute_params.attention_size_swa);
    }
    create_1d_input("n_seq_active", m_compute_params.n_seq_active);
    create_1d_input("seq_active_start", m_compute_params.seq_active_start);
    create_1d_input("seq_active_end", m_compute_params.seq_active_start + m_compute_params.n_seq_active);
    create_1d_input("token_len_per_seq", m_compute_params.token_len_per_seq);
    // create_1d_input("token_len", m_token_len_per_seq * m_n_seq_active);
}

bool GgmlOvDecoder::node_is_used_as_src(const int node_idx) {
    ggml_tensor * node = m_cgraph->nodes[node_idx];
    for (int i = node_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (other_node->src[j] == node) {
                return true;
            }
        }
    }
    return false;
}

void GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_inputs.clear();
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * node = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        if (node->op == GGML_OP_NONE && node_is_used_as_src(i)) {
            std::string node_name(node->name);
            if (m_model_weights.find(node_name) == m_model_weights.end()) {
                m_inputs[node_name] = node;
                auto param_node =
                    std::make_shared<ov::op::v0::Parameter>(get_ov_type(node), get_graph_input_shape(node, nullptr));
                param_node->set_friendly_name(node_name);
                param_node->output(0).get_tensor().set_names({node_name});
                m_model_inputs[node_name] = param_node;
            }
            continue;
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            std::string src_name = std::string(src->name);
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_graph_input_ov_name(src, node);
            }
            if (m_model_weights.find(src_name) != m_model_weights.end()) {
                continue;
            }

            bool is_intermediate_node = false;
            for (const auto & node_info : m_node_info_list) {
                if (node_info.node == src) {
                    is_intermediate_node = true;
                    break;
                }
            }
            if (is_intermediate_node) {
                continue;
            }
            if (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                continue;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                if (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            ov::PartialShape param_shape = get_graph_input_shape(node, src);
            auto param_node = std::make_shared<ov::op::v0::Parameter>(get_ov_type(src), param_shape);
            param_node->set_friendly_name(src_name);
            param_node->output(0).get_tensor().set_names({src_name});
            m_model_inputs[src_name] = param_node;
        }
    }
}

void GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto * cur_node = m_cgraph->nodes[node_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        if (cur_node->op == GGML_OP_NONE) {
            continue;
        }
        auto cur_node_use_count = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_node)];
        if (cur_node_use_count == 0) {
            // The output of SET_ROWS is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            if (cur_node != nullptr && cur_node->op == GGML_OP_SET_ROWS) {
                cur_node = cur_node->view_src;
            }
        } else {
            int input_use_count = 0;
            for (int i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * node = m_cgraph->nodes[i];
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != NULL && node->src[j] == cur_node) {
                        input_use_count++;
                    }
                }
            }
            if (input_use_count == cur_node_use_count) {
                cur_node = nullptr;
            }
        }
        if (cur_node != nullptr) {
            std::string node_output_name(cur_node->name);
            m_model_outputs[node_output_name] = cur_node;
            m_model_output_names.push_back(node_output_name);
        }
    }
}

const ggml_tensor * GgmlOvDecoder::get_tensor_used_op(const ggml_tensor * tensor) const {
    if (tensor == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] == tensor) {
                return node;
            }
        }
    }
    return nullptr;
}

const ggml_tensor * GgmlOvDecoder::get_tensor_from_name(const std::string & name) const {
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const auto * src = node->src[j];
            if (src == nullptr) {
                break;
            }
            if (std::string(src->name) == name) {
                return src;
            }
        }
    }
    return nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() const {
    std::map<std::string, std::string> kv_param_res_names;
    for (const auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    return kv_param_res_names;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, bool naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    for (int node_i = 0; node_i < n_nodes; node_i++) {
        auto * node = nodes[node_i];
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }

            std::string src_name(src->name);
            if (is_rope_freqs_weight(src, node)) {
                src_name = "rope_freqs.weight";
            }
            if (!src->view_src) {
                ggml_backend_buffer * buffer = src->buffer;
                if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(src->type)) {
                    if (model_weights.find(src_name) == model_weights.end()) {
                        auto weight_node = create_weight_node(src, naive);
                        weight_node->set_friendly_name(src_name);
                        model_weights[src_name] = weight_node;
                    }
                }
            }
        }
    }
    return model_weights;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_node(ggml_tensor * tensor, bool naive) {
    const bool is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    if (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        if (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            if (weight_extra->weight_node) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                return weight_extra->weight_node;
            }
        } else if (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            if (quant_extra->weight_node) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                return quant_extra->weight_node;
            }
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    static const std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16,
                                                     GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                                                     GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
    if (weight_types.find(tensor->type) == weight_types.end()) {
        throw std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    if (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        if (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } else {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } else {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_node->set_friendly_name(tensor->name);
    if (!is_ov_buffer) {
        return ov_weight.weight_node;
    }

    ggml_openvino_extra_base * extra;
    if (ov_weight.is_quantized()) {
        extra = new ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_node);
    } else {
        extra = new ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_node);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    return ov_weight.weight_node;
}

void GgmlOvDecoder::dump_cgraph(const ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return;
    }

    file << "=== GRAPH ===\n";

    // clang-format off
    file << "n_nodes = " << cgraph->n_nodes << "\n";
    file << " " << std::setw(3) << "nodes"
                <<  std::setw(15) << "shape"
                << std::setw(20) << "op"
                << std::setw(20) << "name"
                << std::setw(3) << "    "
                << std::setw(62) << "stride"
                << std::setw(20) << "buffer_type"
                << "\n";
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // Get buffer type name
        const char * buf_name = "none";
        ggml_backend_buffer_t buf = node->view_src ? node->view_src->buffer : node->buffer;
        if (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << ", "
             << std::setw(5) << node->ne[2] << ", "
             << std::setw(5) << node->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(node->op) << std::right << " "
             << std::left << std::setw(45) << node->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << node->nb[0] << ", "
             << std::setw(5) << node->nb[1] << ", "
             << std::setw(5) << node->nb[2] << ", "
             << std::setw(5) << node->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (auto* src = node->src[i]) {
                // Get buffer type name for source
                const char * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                if (src_buf) {
                    src_buf_name = ggml_backend_buffer_name(src_buf);
                }

                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::left << std::setw(12) << ggml_op_name(src->op) << std::right;
                file << std::left << std::setw(30) << src->name << std::right
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::right << std::setw(15) << src_buf_name << std::right
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    for (int i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * node = cgraph->leafs[i];

        // Get buffer type name for leaf
        const char * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = node->view_src ? node->view_src->buffer : node->buffer;
        if (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << "] "
             << std::setw(8) << ggml_op_name(node->op) << " "
             << std::setw(16) << ggml_get_name(node)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

void print_tensor_address_map(const ggml_cgraph * cgraph) {
    std::map<void *, std::vector<std::string>> address_map;
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto * node = cgraph->nodes[node_n];
        if (node->data) {
            auto it = address_map.find(node->data);
            if (it == address_map.end()) {
                address_map[node->data] = std::vector<std::string>();
            }
            address_map[node->data].push_back(node->name);
        }
    }
    for (const auto & pair : address_map) {
        std::cout << "Address: " << pair.first << std::endl;
        for (const auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << std::endl << std::endl;
    }
}

ov::Shape GgmlOvDecoder::get_shape(const ggml_tensor * tensor) {
    std::vector<size_t> shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    return shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(const ggml_tensor * tensor) {
    std::vector<size_t> stride;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    return stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(const ggml_tensor * tensor) {
    switch (tensor->type) {
    case GGML_TYPE_F64:
        return ov::element::f64;
    case GGML_TYPE_F32:
        return ov::element::f32;
    case GGML_TYPE_F16:
        return ov::element::f16;
    case GGML_TYPE_BF16:
        return ov::element::bf16;
    case GGML_TYPE_I8:
        return ov::element::i8;
    case GGML_TYPE_I16:
        return ov::element::i16;
    case GGML_TYPE_I32:
        return ov::element::i32;
    case GGML_TYPE_I64:
        return ov::element::i64;
    default:
        return ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_input_shape(int node_idx, const std::string & name) const {
    return ov::PartialShape(get_shape(m_node_info_list[node_idx].node_inputs.at(name)));
}

std::vector<size_t> GgmlOvDecoder::get_input_stride(int node_idx, const std::string & name) const {
    return get_stride(m_node_info_list[node_idx].node_inputs.at(name));
}

ov::element::Type GgmlOvDecoder::get_input_type(int node_idx, const std::string & name) const {
    return get_ov_type(m_node_info_list[node_idx].node_inputs.at(name));
}

size_t GgmlOvDecoder::get_input_size() const {
    return m_model_inputs.size();
}

size_t GgmlOvDecoder::get_input_size(int node_idx) const {
    return m_node_info_list[node_idx].node_inputs_names.size();
}

std::vector<std::string> GgmlOvDecoder::get_input_names(int node_idx) const {
    return m_node_info_list[node_idx].node_inputs_names;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(int node_idx) const {
    auto * ggml_tensor = m_node_info_list[node_idx].node_output;
    return ov::PartialShape(get_shape(ggml_tensor));
}

ov::element::Type GgmlOvDecoder::get_output_type(const int node_idx) const {
    return get_ov_type(m_node_info_list[node_idx].node);
}

std::vector<std::string> GgmlOvDecoder::get_output_names(int node_idx) const {
    return {m_node_info_list[node_idx].node_output_name};
}

const std::string & GgmlOvDecoder::get_op_name() const {
    static const std::string unknown_name = "UNKNOWN_OP_NAME";
    return unknown_name;
}

const std::string & GgmlOvDecoder::get_op_name(int node_idx) const {
    return m_node_info_list[node_idx].node_name;
}

int32_t * GgmlOvDecoder::get_input_op_params(int node_idx, const std::string & name) const {
    return m_node_info_list[node_idx].node_inputs.at(name)->op_params;
}

int32_t * GgmlOvDecoder::get_output_op_params(int node_idx) const {
    return m_node_info_list[node_idx].node->op_params;
}

void GgmlOvDecoder::visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const {
    for (int node_idx = 0; node_idx < m_cgraph->n_nodes; node_idx++) {
        if (m_cgraph->nodes[node_idx]->op == GGML_OP_NONE) {
            continue;
        }
        node_visitor(std::make_shared<GgmlOvDecoder>(*this), node_idx);
    }
}

std::string GgmlOvDecoder::compute_op_type(const ggml_tensor * node) {
    static const std::map<ggml_op, std::string> ops = {
        {GGML_OP_NONE,           "GGML_OP_NONE"          },
        {GGML_OP_ACC,            "GGML_OP_ACC"           },
        {GGML_OP_ADD,            "GGML_OP_ADD"           },
        {GGML_OP_ADD1,           "GGML_OP_ADD1"          },
        {GGML_OP_CONT,           "GGML_OP_CONT"          },
        {GGML_OP_DIV,            "GGML_OP_DIV"           },
        {GGML_OP_DUP,            "GGML_OP_DUP"           },
        {GGML_OP_GET_ROWS,       "GGML_OP_GET_ROWS"      },
        {GGML_OP_MUL,            "GGML_OP_MUL"           },
        {GGML_OP_MUL_MAT,        "GGML_OP_MUL_MAT"       },
        {GGML_OP_PERMUTE,        "GGML_OP_PERMUTE"       },
        {GGML_OP_RESHAPE,        "GGML_OP_RESHAPE"       },
        {GGML_OP_RMS_NORM,       "GGML_OP_RMS_NORM"      },
        {GGML_OP_ROPE,           "GGML_OP_ROPE"          },
        {GGML_OP_SCALE,          "GGML_OP_SCALE"         },
        {GGML_OP_SOFT_MAX,       "GGML_OP_SOFT_MAX"      },
        {GGML_OP_SUB,            "GGML_OP_SUB"           },
        {GGML_OP_TRANSPOSE,      "GGML_OP_TRANSPOSE"     },
        {GGML_OP_VIEW,           "GGML_OP_VIEW"          },
        {GGML_OP_SET_ROWS,       "GGML_OP_SET_ROWS"      },
        {GGML_OP_CPY,            "GGML_OP_CPY"           },
        {GGML_OP_FLASH_ATTN_EXT, "GGML_OP_FLASH_ATTN_EXT"},
    };
    static const std::map<ggml_unary_op, std::string> unary_ops = {
        {GGML_UNARY_OP_ABS,         "GGML_UNARY_OP_ABS"        },
        {GGML_UNARY_OP_SGN,         "GGML_UNARY_OP_SGN"        },
        {GGML_UNARY_OP_NEG,         "GGML_UNARY_OP_NEG"        },
        {GGML_UNARY_OP_STEP,        "GGML_UNARY_OP_STEP"       },
        {GGML_UNARY_OP_TANH,        "GGML_UNARY_OP_TANH"       },
        {GGML_UNARY_OP_ELU,         "GGML_UNARY_OP_ELU"        },
        {GGML_UNARY_OP_RELU,        "GGML_UNARY_OP_RELU"       },
        {GGML_UNARY_OP_SIGMOID,     "GGML_UNARY_OP_SIGMOID"    },
        {GGML_UNARY_OP_GELU,        "GGML_UNARY_OP_GELU"       },
        {GGML_UNARY_OP_GELU_QUICK,  "GGML_UNARY_OP_GELU_QUICK" },
        {GGML_UNARY_OP_SILU,        "GGML_UNARY_OP_SILU"       },
        {GGML_UNARY_OP_HARDSWISH,   "GGML_UNARY_OP_HARDSWISH"  },
        {GGML_UNARY_OP_HARDSIGMOID, "GGML_UNARY_OP_HARDSIGMOID"},
        {GGML_UNARY_OP_EXP,         "GGML_UNARY_OP_EXP"        },
        {GGML_UNARY_OP_COUNT,       "GGML_UNARY_OP_COUNT"      }
    };
    static const std::map<ggml_glu_op, std::string> glu_ops = {
        {GGML_GLU_OP_SWIGLU, "GGML_GLU_OP_SWIGLU"},
        {GGML_GLU_OP_GEGLU,  "GGML_GLU_OP_GEGLU" },
        {GGML_GLU_OP_REGLU,  "GGML_GLU_OP_REGLU" }
    };

    switch (node->op) {
    case GGML_OP_UNARY:
        return unary_ops.at(ggml_get_unary_op(node));
    case GGML_OP_GLU:
        return glu_ops.at(ggml_get_glu_op(node));
    default:
        return ops.at(node->op);
    }
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return unknown_op;
}

const std::string & GgmlOvDecoder::get_op_type(int node_idx) const {
    return m_node_info_list[node_idx].node_op_type;
}

const std::string & GgmlOvDecoder::get_op_type() const {
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return unknown_op;
}


/* Inlined openvino/frontend.cpp */
// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
public:
    using Ptr = std::shared_ptr<FrontEnd>;
    FrontEnd();

    static std::shared_ptr<Model> convert(const InputModel::Ptr& model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


#include <openvino/frontend/input_model.hpp>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::ggml::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgmlDecoder>& gdecoder);

    const std::shared_ptr<GgmlDecoder>& get_model_decoder() const;

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

namespace op {

#define GGML_OP_CONVERTER(op) OutputVector op(const NodeContext& context)

GGML_OP_CONVERTER(translate_add);
GGML_OP_CONVERTER(translate_cont);
GGML_OP_CONVERTER(translate_get_rows);
GGML_OP_CONVERTER(translate_mul);
GGML_OP_CONVERTER(translate_mulmat);
GGML_OP_CONVERTER(translate_permute);
GGML_OP_CONVERTER(translate_reshape);
GGML_OP_CONVERTER(translate_rms_norm);
GGML_OP_CONVERTER(translate_rope);
GGML_OP_CONVERTER(translate_scale);
GGML_OP_CONVERTER(translate_unary_silu);
GGML_OP_CONVERTER(translate_unary_gelu);
GGML_OP_CONVERTER(translate_soft_max);
GGML_OP_CONVERTER(translate_transpose);
GGML_OP_CONVERTER(translate_view);
GGML_OP_CONVERTER(translate_glu_swiglu);
GGML_OP_CONVERTER(translate_glu_geglu);
GGML_OP_CONVERTER(translate_set_rows);
GGML_OP_CONVERTER(translate_cpy);
GGML_OP_CONVERTER(translate_flash_attn_ext);

} // namespace op

std::unordered_map<std::string, CreatorFunction> get_supported_ops();

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
public:
    TranslateSession(const frontend::InputModel::Ptr& input_model,
                     const std::unordered_map<std::string, CreatorFunction>& translator_map, bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(const frontend::InputModel::Ptr& input_model);

private:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    const frontend::InputModel::Ptr m_input_model;
    const std::unordered_map<std::string, CreatorFunction>& m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr & model, bool naive) {
    auto ggml_model = std::dynamic_pointer_cast<ggml::InputModel>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    const auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);
        converted_model = translate_session.get_converted_model();
    }
    return converted_model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined openvino/input_model.cpp */
#include <openvino/frontend/input_model.hpp>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::ggml::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgmlDecoder>& gdecoder);

    const std::shared_ptr<GgmlDecoder>& get_model_decoder() const;

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov



namespace ov {
namespace frontend {
namespace ggml {

InputModel::InputModel(const std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

const std::shared_ptr<GgmlDecoder> & InputModel::get_model_decoder() const {
    return m_decoder;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined openvino/op_table.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

namespace op {

#define GGML_OP_CONVERTER(op) OutputVector op(const NodeContext& context)

GGML_OP_CONVERTER(translate_add);
GGML_OP_CONVERTER(translate_cont);
GGML_OP_CONVERTER(translate_get_rows);
GGML_OP_CONVERTER(translate_mul);
GGML_OP_CONVERTER(translate_mulmat);
GGML_OP_CONVERTER(translate_permute);
GGML_OP_CONVERTER(translate_reshape);
GGML_OP_CONVERTER(translate_rms_norm);
GGML_OP_CONVERTER(translate_rope);
GGML_OP_CONVERTER(translate_scale);
GGML_OP_CONVERTER(translate_unary_silu);
GGML_OP_CONVERTER(translate_unary_gelu);
GGML_OP_CONVERTER(translate_soft_max);
GGML_OP_CONVERTER(translate_transpose);
GGML_OP_CONVERTER(translate_view);
GGML_OP_CONVERTER(translate_glu_swiglu);
GGML_OP_CONVERTER(translate_glu_geglu);
GGML_OP_CONVERTER(translate_set_rows);
GGML_OP_CONVERTER(translate_cpy);
GGML_OP_CONVERTER(translate_flash_attn_ext);

} // namespace op

std::unordered_map<std::string, CreatorFunction> get_supported_ops();

}  // namespace ggml
}  // namespace frontend
}  // namespace ov



#include <openvino/op/add.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/subtract.hpp>

namespace ov {
namespace frontend {
namespace ggml {

std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
    using namespace ov::op;
    return {
        {"GGML_OP_ADD",            op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_ADD1",           op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_CONT",           op::translate_cont                             },
        {"GGML_OP_DIV",            op::translate_1to1_match_2_inputs<v1::Divide>  },
        {"GGML_OP_GET_ROWS",       op::translate_get_rows                         },
        {"GGML_OP_MUL",            op::translate_1to1_match_2_inputs<v1::Multiply>},
        {"GGML_OP_MUL_MAT",        op::translate_mulmat                           },
        {"GGML_OP_PERMUTE",        op::translate_permute                          },
        {"GGML_OP_RESHAPE",        op::translate_reshape                          },
        {"GGML_OP_RMS_NORM",       op::translate_rms_norm                         },
        {"GGML_OP_ROPE",           op::translate_rope                             },
        {"GGML_OP_SCALE",          op::translate_scale                            },
        {"GGML_OP_SOFT_MAX",       op::translate_soft_max                         },
        {"GGML_OP_SUB",            op::translate_1to1_match_2_inputs<v1::Subtract>},
        {"GGML_OP_TRANSPOSE",      op::translate_transpose                        },
        {"GGML_UNARY_OP_GELU",     op::translate_unary_gelu                       },
        {"GGML_UNARY_OP_SILU",     op::translate_unary_silu                       },
        {"GGML_OP_VIEW",           op::translate_view                             },
        {"GGML_GLU_OP_SWIGLU",     op::translate_glu_swiglu                       },
        {"GGML_GLU_OP_GEGLU",      op::translate_glu_geglu                        },
        {"GGML_OP_SET_ROWS",       op::translate_set_rows                         },
        {"GGML_OP_CPY",            op::translate_cpy                              },
        {"GGML_OP_FLASH_ATTN_EXT", op::translate_flash_attn_ext                   },
    };
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/cont.cpp */

#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cont(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported CONT case");

    auto src_shape = context.get_input_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();
    ov::Output<Node> res;

    if (op_case == 1) {
        // The input comes from a PERMUTE
        throw std::runtime_error("Code of this case might be outdated");
        dst_shape[1] = -1;
        res = std::make_shared<ov::op::v1::Reshape>(
            context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), false);
    } else if (op_case == 2) {
        // The input comes from a TRANSPOSE
        return {context.get_input(0)};
    } else {
        // The input comes from a VIEW
        res = process_view_input(context, 0);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/cpy.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <memory>
#include <openvino/op/convert.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cpy(const NodeContext & context) {
    auto res = std::make_shared<ov::op::v0::Convert>(context.get_input(0), context.get_output_type());
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/flash_attn_ext.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_flash_attn_ext(const NodeContext & context) {
    num_inputs_check(context, 4, 4);
    auto q_f32 = context.get_input(0);
    auto k = context.get_input(1);
    auto v = context.get_input(2);
    auto mask = context.get_input(3);

    float * params = reinterpret_cast<float *>(context.get_output_op_params());
    float scale = params[0];
    // float max_bias      = params[1];
    // float logit_softcap = params[2];

    auto q = std::make_shared<ov::op::v0::Convert>(q_f32, ov::element::f16);
    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{}, std::vector<float>{scale});

    ov::Output<ov::Node> mask_sliced, res;
    std::string mask_name = "KQ_mask_sliced";
    if (context.get_input_names()[3].find("swa") != std::string::npos) {
        mask_name = "KQ_mask_swa_sliced";
    }
    if (context.has_input(mask_name)) {
        mask_sliced = context.get_input(mask_name);
    } else {
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto token_len = get_dimensions(q, {2});
        mask_sliced = std::make_shared<ov::op::v8::Slice>(mask, zero, token_len, one, two);
    }

    if (mask_sliced.get_element_type() != ov::element::f16) {
        mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
    }

    auto tile_kv = [&](int64_t num_heads, int64_t num_heads_kv, int64_t head_size, ov::Output<Node> kv) {
        int64_t factor = num_heads / num_heads_kv;
        if (factor > 1 && num_heads_kv > 1) {
            ov::Output<ov::Node> kv_broadcast_shape, kv_unsqueezed, new_kv_shape;
            auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
            kv_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, unsqueeze_axes);

            kv_broadcast_shape = ov::op::v0::Constant::create(
                ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
            new_kv_shape =
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 0, num_heads, (int64_t) -1, head_size});

            kv = std::make_shared<ov::op::v3::Broadcast>(kv_unsqueezed, kv_broadcast_shape,
                                                         ov::op::BroadcastType::BIDIRECTIONAL);
            kv = std::make_shared<ov::op::v1::Reshape>(kv, new_kv_shape, true);
        }
        return kv;
    };

    auto q_shape = context.get_input_shape(0).to_shape();
    auto k_shape = context.get_input_shape(1).to_shape();
    k = tile_kv(q_shape[1], k_shape[1], q_shape[3], k);
    v = tile_kv(q_shape[1], k_shape[1], q_shape[3], v);

    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_sliced, scale_node, false);
    res = std::make_shared<ov::op::v1::Transpose>(sdpa,
                                                  ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3}));
    res = std::make_shared<ov::op::v0::Convert>(res, ov::element::f32);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/get_rows.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_get_rows(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();

    Output<Node> res;
    auto data = context.get_input(0);
    auto indices = context.get_input(1);

    if (op_case == 2) {
        // The input comes from a VIEW
        indices = process_view_input(context, 1);
    }

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    indices =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
    if (data.get_partial_shape().rank() == 4) {
        if (!(data.get_partial_shape()[1].is_dynamic()) && data.get_partial_shape()[1].get_length() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            data = std::make_shared<ov::op::v0::Squeeze>(data,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
        } else {
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            data =
                std::make_shared<ov::op::v0::Squeeze>(data, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
        }
    } else if (context.is_stateful() && data.get_partial_shape().rank() == 3) {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
    } else {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    }

    if (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    if (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/glu_geglu.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_glu_geglu(const NodeContext & context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    if (context.get_input_size() == 2) {
        src0 = context.get_input(0);
        src1 = context.get_input(1);
    } else {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        auto combined = context.get_input(0);
        auto combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        auto axis   = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step   = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    int32_t * params = context.get_output_op_params();
    const int32_t swapped = params[1];
    if (swapped) {
        std::swap(src0, src1);
    }

    auto gelu = std::make_shared<ov::op::v7::Gelu>(src0);
    auto res = std::make_shared<ov::op::v1::Multiply>(gelu, src1);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/glu_swiglu.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_glu_swiglu(const NodeContext & context) {
    num_inputs_check(context, 1, 2);

    ov::Output<ov::Node> src0;
    ov::Output<ov::Node> src1;
    if (context.get_input_size() == 2) {
        src0 = context.get_input(0);
        src1 = context.get_input(1);
    } else {
        // GGML splits along ne[0] (OV last axis) using floor division: nc = ne[0] / 2.
        // Both halves are nc elements; if the dimension is odd, the last element is dropped.
        // Use Slice instead of Split to handle odd dimensions correctly.
        auto combined = context.get_input(0);
        auto combined_shape = combined.get_partial_shape();
        int64_t last_dim_val = combined_shape[combined_shape.rank().get_length() - 1].get_length();
        int64_t nc = last_dim_val / 2;

        auto axis   = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto step   = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto start0 = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto stop0  = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto start1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {nc});
        auto stop1  = ov::op::v0::Constant::create(ov::element::i64, {1}, {2 * nc});

        src0 = std::make_shared<ov::op::v8::Slice>(combined, start0, stop0, step, axis);
        src1 = std::make_shared<ov::op::v8::Slice>(combined, start1, stop1, step, axis);
    }

    int32_t * params = context.get_output_op_params();
    const int32_t swapped = params[1];
    if (swapped) {
        std::swap(src0, src1);
    }

    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(src0);
    auto silu = std::make_shared<ov::op::v1::Multiply>(src0, sigmoid);
    auto res = std::make_shared<ov::op::v1::Multiply>(silu, src1);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/mulmat.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/util/op_types.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mulmat(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();

    ov::Output<Node> res;
    ov::Output<ov::Node> B = context.get_input(0);
    ov::Output<ov::Node> A = context.get_input(1);

    bool transpose_b = true;
    if (op_case == 2) {
        B = B.get_node_shared_ptr()->input_value(0);
        transpose_b = false;
    } else if (op_case == 3) {
        B = process_view_input(context, 0);
        A = process_view_input(context, 1);
    }
    if (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(context.get_input(0), context.get_input_type(1));
    }

    auto B_shape = context.get_input_shape(0).to_shape();
    auto A_shape = context.get_input_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    int64_t factor = batch_large / batch_small;
    if (factor > 1 && batch_small > 1) {
        auto batch_large_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) A_shape[3]});

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, true);
    }
    if (A_batch_larger) {
        B = Z;
    } else {
        A = Z;
    }

    res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/permute.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_permute(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3 || op_case == 4,
                                "Unsupported PERMUTE case");

    ov::Output<Node> res;
    auto src = context.get_input(0);
    auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 2, 1, 3});

    if (op_case == 1 || context.is_stateful()) {
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } else if (op_case == 4) {
        auto output_shape = context.get_output_shape().to_shape();
        auto n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        auto head_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        auto n_seq_active = context.has_input("n_seq_active") ?
                                context.get_input("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        auto new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, n_heads, head_size}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        auto reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, true);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } else {
        auto cache_shape = src.get_partial_shape();
        auto output_shape = context.get_output_shape().to_shape();
        int64_t head_size = output_shape[3];
        int64_t n_heads = output_shape[1];
        int64_t ctx_per_seq = cache_shape[2].is_static() ? cache_shape[2].get_length() : -1;
        int64_t n_seq = cache_shape[1].get_length();

        Output<Node> attention_size;
        if (!context.has_input("attention_size")) {
            attention_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } else if (op_case == 2) {
            attention_size = context.get_input("attention_size");
        } else {
            attention_size = context.get_input("attention_size_swa");
        }

        Output<Node> seq_active_start;
        Output<Node> seq_active_end;
        if (context.has_input("seq_active_start")) {
            seq_active_start = context.get_input("seq_active_start");
            seq_active_end = context.get_input("seq_active_end");
        } else {
            int64_t n_seq_active = output_shape[0];
            size_t offset = *((size_t *) context.get_input_op_params(0));
            int64_t seq_active_start_val = offset / context.get_input_stride(0)[0];
            int64_t seq_active_end_val = seq_active_start_val + n_seq_active;
            seq_active_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_start_val});
            seq_active_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_end_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size]
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        auto src_reshaped = std::make_shared<ov::op::v1::Reshape>(
            src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_per_seq, n_heads, head_size}), false);
        auto slice1 = std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
        auto slice2 = std::make_shared<ov::op::v8::Slice>(slice1, zero, attention_size, one, one);
        res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/reshape.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <stdexcept>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_reshape(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    if (context.get_input_shape(0) == context.get_output_shape()) {
        return {context.get_input(0)};
    }

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(
        op_case == 1 || op_case == 2 || op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6,
        "Unsupported RESHAPE case");

    auto output_shape = context.get_output_shape().to_shape();
    std::shared_ptr<ov::Node> new_shape_node;
    if (op_case == 1) {
        if (context.is_stateful()) {
            new_shape_node = ov::op::v0::Constant::create(
                ov::element::i64, {3},
                std::vector<int64_t>{-1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        } else {
            new_shape_node = ov::op::v0::Constant::create(
                ov::element::i64, {4},
                std::vector<int64_t>{(int64_t) output_shape[0], -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        }
    } else if (op_case == 2) {
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{(int64_t) output_shape[0], (int64_t) output_shape[1], -1, (int64_t) output_shape[3]});

    } else if (op_case == 3) {
        throw std::runtime_error("might be outdated RESHAPE case");
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64, {4}, std::vector<int64_t>{(int64_t) output_shape[0], (int64_t) output_shape[1], -1, 1});

    } else if (op_case == 4) {
        return {context.get_input(0).get_node_shared_ptr()->input_value(0)};

    } else if (op_case == 5) {
        if (context.is_stateful()) {
            std::vector<int64_t> shape_vec = {1, -1, (int64_t) context.get_output_shape().to_shape()[3]};
            new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {3}, shape_vec);
        } else {
            std::vector<int64_t> shape_vec = {1, 1, -1, (int64_t) context.get_output_shape().to_shape()[3]};
            new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {4}, shape_vec);
        }

        // // Alternative
        // auto token_len = context.get_input("token_len");
        // auto emb_size =
        //     ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) context.get_output_shape().to_shape()[3]});
        // auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        // new_shape_node = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, one, token_len, emb_size}, 0);

    } else if (op_case == 6) {
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {4}, context.get_output_shape().to_shape());
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/rms_norm.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/sqrt.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rms_norm(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input_node = context.get_input(0);
    auto square = std::make_shared<ov::op::v1::Power>(
        input_node, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f}));

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/rope.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rope(const NodeContext & context) {
    num_inputs_check(context, 2, 3);

    int op_case = context.get_op_case();

    ov::Output<Node> res;

    auto data_node = context.get_input(0).get_node_shared_ptr();
    auto output_shape = context.get_output_shape().to_shape();
    int32_t * op_params = context.get_output_op_params();
    const int mode = (op_case & 0xFFFF0000) >> 16;
    op_case = (op_case & 0x0000FFFF);

    constexpr int TYPE_NORMAL = 0;
    constexpr int TYPE_NEOX = 1;
    constexpr int TYPE_IMROPE = 2;

    Output<Node> cos_theta_node;
    Output<Node> sin_theta_node;
    if (context.has_input("rope_cos")) {
        cos_theta_node = context.get_input("rope_cos");
        sin_theta_node = context.get_input("rope_sin");
    } else {
        auto inp_pos = context.get_input(1).get_node_shared_ptr();
        std::shared_ptr<ov::Node> rope_freqs_weight;
        if (context.get_input_size() == 3) {
            rope_freqs_weight = context.get_input(2).get_node_shared_ptr();
        }
        auto sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_weight, mode == TYPE_IMROPE);
        sin_theta_node = sin_cos.first;
        cos_theta_node = sin_cos.second;
    }

    if (op_case == 2) {
        // The input comes from a VIEW
        int slice_len = output_shape[2] * output_shape[3];
        data_node = process_view_input(context, 0, slice_len).get_node_shared_ptr();
        if (context.is_stateful()) {
            auto data_shape = ov::op::v0::Constant::create(
                ov::element::i64, {3}, std::vector<int64_t>{-1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
            data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
        } else {
            auto data_shape = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
            data_node = std::make_shared<ov::op::v1::Reshape>(data_node, data_shape, false);
        }
    }

    if (mode == TYPE_NORMAL) {
        auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        Output<Node> even_slice;
        Output<Node> odd_slice;
        int32_t unsqueeze_dim = context.is_stateful() ? 3 : 4;
        even_slice = std::make_shared<ov::op::v8::Slice>(data_node, zero, end, two, neg_one);
        odd_slice = std::make_shared<ov::op::v8::Slice>(data_node, one, end, two, neg_one);

        Output<Node> first_half =
            std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v1::Multiply>(even_slice, cos_theta_node),
                                                   std::make_shared<ov::op::v1::Multiply>(odd_slice, sin_theta_node));
        Output<Node> second_half =
            std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(even_slice, sin_theta_node),
                                              std::make_shared<ov::op::v1::Multiply>(odd_slice, cos_theta_node));

        first_half = std::make_shared<ov::op::v0::Unsqueeze>(first_half,
                                                             ov::op::v0::Constant::create(ov::element::i64, {1}, {unsqueeze_dim}));
        second_half = std::make_shared<ov::op::v0::Unsqueeze>(second_half,
                                                              ov::op::v0::Constant::create(ov::element::i64, {1}, {unsqueeze_dim}));
        auto stack = std::make_shared<ov::op::v0::Concat>(OutputVector{first_half, second_half}, unsqueeze_dim);

        auto data_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4}, std::vector<int64_t>{1, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        res = std::make_shared<ov::op::v1::Reshape>(stack, data_shape, false);
    } else if (mode == TYPE_NEOX) {
        auto data_split = std::make_shared<ov::op::v1::Split>(
            data_node, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1}), 2);
        Output<Node> slice_data_node_0 = data_split->outputs()[0];
        Output<Node> slice_data_node_1 = data_split->outputs()[1];

        auto first_half_node = std::make_shared<ov::op::v1::Subtract>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, cos_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, sin_theta_node));

        auto second_half_node = std::make_shared<ov::op::v1::Add>(
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_0, sin_theta_node),
            std::make_shared<ov::op::v1::Multiply>(slice_data_node_1, cos_theta_node));

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_half_node, second_half_node}, -1);
    } else if (mode == TYPE_IMROPE) {
        int64_t n_dims = data_node->get_shape()[3];
        auto cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1,-1,1,(n_dims >> 1)});
        auto cos_reshaped = std::make_shared<ov::op::v1::Reshape>(cos_theta_node, cos_sin_shape, true);
        auto sin_reshaped = std::make_shared<ov::op::v1::Reshape>(sin_theta_node, cos_sin_shape, true);

        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto split_a = std::make_shared<ov::op::v1::Split>(data_node, split_axis, 2);
        auto x0 = split_a->output(0);
        auto x1 = split_a->output(1);
        auto mul_a = std::make_shared<ov::op::v1::Multiply>(x0, cos_reshaped);
        auto mul_b = std::make_shared<ov::op::v1::Multiply>(x1, sin_reshaped);
        auto sub = std::make_shared<ov::op::v1::Subtract>(mul_a, mul_b);

        auto mul_c = std::make_shared<ov::op::v1::Multiply>(x0, sin_reshaped);
        auto mul_d = std::make_shared<ov::op::v1::Multiply>(x1, cos_reshaped);
        auto add = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);

        res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub, add}, 3);
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/scale.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_scale(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    float scale;
    float bias;
    memcpy(&scale, (float *) context.get_output_op_params() + 0, sizeof(float));
    memcpy(&bias, (float *) context.get_output_op_params() + 1, sizeof(float));

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_input(0), scale_node);

    std::shared_ptr<ov::Node> res;
    if (bias != 0.0f) {
        auto bias_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_node);
    } else {
        res = scaled;
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/set_rows.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_set_rows(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    auto dst = context.get_input(2);

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type());

    auto dst_shape = context.get_output_shape().to_shape();

    auto ind_squeezed =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
    auto data_reshaped = std::make_shared<ov::op::v1::Reshape>(
        data,
        ov::op::v0::Constant::create(ov::element::i64, {4},
                                     {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) dst_shape[3]}),
        false);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    if (context.is_stateful()) {
        int concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_length();
        int64_t dim3 = dst.get_partial_shape()[3].get_length();
        data = std::make_shared<ov::op::v1::Reshape>(
            data, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), false);
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{dst, data}, concat_axis);
    } else {
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_reshaped, axes);
    }

    if (auto dst_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(dst.get_node_shared_ptr())) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        auto dst_shape_partial = dst_reshape->get_input_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_length(), dst_shape_partial[1].get_length(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_length() : -1,
                                          dst_shape_partial[3].get_length()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    false);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/softmax.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/softmax.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_soft_max(const NodeContext & context) {
    // TODO code is outdated
    num_inputs_check(context, 1, 2);

    auto input_node = context.get_input(0).get_node_shared_ptr();
    ov::Output<Node> res;

    float scale = 1.0f;
    float max_bias = 0.0f;
    auto * op_params = context.get_output_op_params();
    memcpy(&scale, (float *) op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) op_params + 1, sizeof(float));
    auto src0_shape = context.get_input_shape(0).get_shape();
    const uint32_t h = src0_shape[2];
    const uint32_t n_head = src0_shape[0];
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);
    const float slope =
        (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
    auto scaled_input = std::make_shared<ov::op::v1::Multiply>(input_node, scale_node);

    if (context.get_input_size() < 2) {
        res = std::make_shared<ov::op::v8::Softmax>(scaled_input, 2);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    ov::Output<ov::Node> mask_node_sliced;
    if (context.has_input("KQ_mask_sliced")) {
        mask_node_sliced = context.get_input("KQ_mask_sliced");
    } else {
        auto token_len = get_dimensions(input_node, {1});
        auto mask_node = context.get_input(1);
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        mask_node_sliced = std::make_shared<ov::op::v8::Slice>(mask_node, zero, token_len, one, one);
    }

    if (mask_node_sliced.get_element_type() != context.get_output_type()) {
        mask_node_sliced = std::make_shared<ov::op::v0::Convert>(mask_node_sliced, context.get_output_type());
    }

    Output<Node> slope_mask;
    if (slope != 1.0f) {
        auto slope_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{slope});
        slope_mask = std::make_shared<ov::op::v1::Multiply>(mask_node_sliced, slope_node);
        throw std::runtime_error("Slope != 1.0f in softmax has not been tested, verify it before use.");
    }
    slope_mask = mask_node_sliced;

    auto input_slope_mask_node = std::make_shared<ov::op::v1::Add>(scaled_input, slope_mask);

    res = std::make_shared<ov::op::v8::Softmax>(input_slope_mask_node, 2);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/transpose.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <openvino/op/transpose.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_transpose(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto res = std::make_shared<ov::op::v1::Transpose>(
        context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {4}, {0, 1, 3, 2}));
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/unary_gelu.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <openvino/core/node_output.hpp>
#include <openvino/op/gelu.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_unary_gelu(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto res = std::make_shared<ov::op::v7::Gelu>(input);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/unary_silu.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"

#include <openvino/core/node_output.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/sigmoid.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_unary_silu(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(input);
    auto res = std::make_shared<ov::op::v1::Multiply>(input, sigmoid);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined op/view.cpp */
#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include "../utils.h"
#include <openvino/op/reshape.hpp>
namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    if (context.get_op_case() == 2) {
        auto dst_shape = context.get_output_shape().to_shape();
        return rename_outputs_with_suffix({process_view_input(context, 0, dst_shape[2] * dst_shape[3])},
                                          context.get_name());
    }
    // op_case 3
    if (context.get_op_case() == 3) {
        auto input = context.get_input(0);
        auto input_ov_shape = input.get_partial_shape();

        auto input_llama_shape = context.get_input_shape(0).to_shape();

        // if the input ov shape size is different from the input llama shape size, it means the input is already reshaped and we need to reshape it back to the original shape before slicing
        if (input_ov_shape.size() != input_llama_shape.size()) {
            input = std::make_shared<ov::op::v1::Reshape>(input, ov::op::v0::Constant::create(ov::element::i64, {input_llama_shape.size()}, input_llama_shape), false);
        }

        auto dst_shape = context.get_output_shape().to_shape();

        // find the index of dst_shape that is different from input shape, and use that index to slice the input
        int slice_dim = -1;
        for (size_t i = 0; i < dst_shape.size(); ++i) {
            if (dst_shape[i] != input_llama_shape[i]) {
                slice_dim = i;
                break;
            }
        }

        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {dst_shape[slice_dim]});
        auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim});
        auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
        return {sliced};
    }
    return {context.get_input(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined pass/fuse_to_sdpa.cpp */
#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class FuseToSDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseToSDPA")
    FuseToSDPA();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/scaled_dot_product_attention.hpp>
#include <openvino/op/softmax.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/pass/pattern/op/label.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

FuseToSDPA::FuseToSDPA() {
    // Not maintained since FLASH_ATTN_EXT has replaced this pattern
    const auto m_k = ov::pass::pattern::any_input();
    const auto m_q = ov::pass::pattern::any_input();
    const auto m_qk = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_q, m_k});
    const auto m_qk_f32 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_qk});
    const auto m_scale = ov::pass::pattern::any_input();
    const auto m_scaled_qk = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_qk_f32, m_scale});
    const auto m_mask = ov::pass::pattern::any_input();
    const auto m_masked_qk = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_scaled_qk, m_mask});
    const auto m_softmax_qk = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>({m_masked_qk});
    const auto m_softmax_qk_f16 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({m_softmax_qk});
    const auto m_v = ov::pass::pattern::any_input();
    const auto m_qkv = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({m_softmax_qk_f16, m_v});

    const auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto & pattern_to_output = m.get_pattern_value_map();
        auto k = pattern_to_output[m_k];
        auto q = pattern_to_output[m_q];
        auto v = pattern_to_output[m_v];
        auto mask = pattern_to_output[m_mask];
        auto scale = pattern_to_output[m_scale];

        auto mask_f16 = register_new_node<ov::op::v0::Convert>(mask, ov::element::f16);
        auto scale_f16 = register_new_node<ov::op::v0::Convert>(scale, ov::element::f16);
        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, mask_f16, scale_f16, false);

        ov::replace_node(m.get_match_root(), sdpa);
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);

        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_qkv, "ov::frontend::ggml::pass::FuseToSDPA"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined pass/squeeze_matmul.cpp */
#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class SqueezeMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::SqueezeMatmul")
    SqueezeMatmul();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/pattern/op/label.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace opp = ov::pass::pattern;

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// For quantized models, NPUW expects the activation to be 3d in DQ(DynamicQuantization) opt, e.g. DQMatMulGQ2i
SqueezeMatmul::SqueezeMatmul() {
    auto m_act = opp::any_input();
    auto m_wei = opp::any_input();
    auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_act, m_wei});

    const auto callback = [=](ov::pass::pattern::Matcher & m) {
        const auto & pattern_map = m.get_pattern_value_map();
        auto matmul_node =
            std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(m_matmul).get_node_shared_ptr());
        auto act = pattern_map.at(m_act);
        auto wei = pattern_map.at(m_wei);
        auto act_shape = act.get_partial_shape();
        auto wei_shape = wei.get_partial_shape();
        if (act_shape.rank().is_dynamic() || wei_shape.rank().is_dynamic()) {
            return false;
        }
        if (act_shape.rank().get_length() == 4 && wei_shape.rank().get_length() == 2) {
            auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
            auto squeezed_act = std::make_shared<ov::op::v0::Squeeze>(act, axis);
            auto new_matmul = std::make_shared<ov::op::v0::MatMul>(squeezed_act, wei, matmul_node->get_transpose_a(),
                                                                   matmul_node->get_transpose_b());
            auto unsqueezed_output = std::make_shared<ov::op::v0::Unsqueeze>(new_matmul, axis);
            unsqueezed_output->set_friendly_name(matmul_node->get_friendly_name());
            ov::copy_runtime_info(matmul_node, {squeezed_act, new_matmul, unsqueezed_output});
            ov::replace_node(matmul_node, unsqueezed_output);
            return true;
        }
        return false;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(m_matmul, "ov::frontend::ggml::pass::SqueezeMatmul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined openvino/translate_session.cpp */
#include <openvino/frontend/input_model.hpp>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd;
class GgmlDecoder;
using ov::frontend::ggml::GgmlDecoder;

class InputModel : public ov::frontend::InputModel {
    friend class ::ov::frontend::ggml::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgmlDecoder>& gdecoder);

    const std::shared_ptr<GgmlDecoder>& get_model_decoder() const;

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov

#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class GgmlDecoder : public DecoderBase {
public:
    virtual ov::Any get_attribute(const std::string& name) const = 0;

    virtual PartialShape get_input_shape(int node_idx, const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(int node_idx, const std::string& name) const = 0;

    virtual element::Type get_input_type(int node_idx, const std::string& name) const = 0;

    virtual size_t get_input_size() const = 0;

    virtual size_t get_input_size(int node_idx) const = 0;

    virtual void get_input_node(size_t input_port_idx,
                                std::string& producer_name,
                                std::string& producer_output_port_name,
                                size_t& producer_output_port_index) const = 0;

    virtual std::vector<std::string> get_input_names(int node_idx) const = 0;

    virtual PartialShape get_output_shape(int node_idx) const = 0;

    virtual element::Type get_output_type(const int node_idx) const = 0;

    virtual int32_t* get_input_op_params(int node_idx, const std::string& name) const = 0;

    virtual int32_t * get_output_op_params(int node_idx) const = 0;

    virtual std::vector<std::string> get_output_names(int node_idx) const = 0;

    virtual const std::string& get_op_type() const = 0;

    virtual const std::string& get_op_type(int node_idx) const = 0;

    virtual const std::string& get_op_name() const = 0;

    virtual const std::string& get_op_name(int node_idx) const = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgmlDecoder>, int node_idx)> node_visitor) const = 0;

    virtual int get_op_case(int node_idx) const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_weights() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    virtual int32_t* get_rope_params() const = 0;

    virtual std::map<std::string, std::string> get_kv_param_res_names() const = 0;

    virtual bool is_static() const = 0;

    virtual bool is_stateful() const = 0;

    virtual int is_swa_layer(int layer) const = 0;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        return m_tensor_map->at(m_input_names[idx]);
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession {
public:
    TranslateSession(const frontend::InputModel::Ptr& input_model,
                     const std::unordered_map<std::string, CreatorFunction>& translator_map, bool naive = false);

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(const frontend::InputModel::Ptr& input_model);

private:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    const frontend::InputModel::Ptr m_input_model;
    const std::unordered_map<std::string, CreatorFunction>& m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


#include "ggml-openvino/openvino/utils.h"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/core/visibility.hpp"

#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    else
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY

namespace ov {
namespace pass {

class TRANSFORMATIONS_API MarkCompressedFloatConstants;

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/core_visibility.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/runtime_attribute.hpp>

namespace ov {

/**
 * @brief Holds weightless caching attributes of a single constant.
 *
 * WeightlessCacheAttribute class represents runtime info attribute that holds
 * the values of original size of the constant in bytes and the binary offset of the
 * constant's data in the weights file used by the weightless caching mechanism. It's
 * not copyable in case the data was changed (the original node was replaced by a new
 * one produced during the tranformation pipeline) - in that case weightless caching
 * can't be used for that constant.
 */
class OPENVINO_API WeightlessCacheAttribute : public RuntimeAttribute {
public:
    OPENVINO_RTTI("WeightlessCacheAttribute", "0", RuntimeAttribute)

    WeightlessCacheAttribute() = delete;

    WeightlessCacheAttribute(size_t original_size, size_t bin_offset, ov::element::Type original_dtype)
        : original_size(original_size),
          bin_offset(bin_offset),
          original_dtype(original_dtype) {}

    bool is_copyable() const override;

    size_t original_size;
    size_t bin_offset;
    ov::element::Type original_dtype;
};

}  // namespace ov


#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/convert_like.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/make_stateful.hpp>

namespace ov {
namespace frontend {
namespace ggml {

using namespace ov::op;

namespace {

ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    const std::shared_ptr<ov::Model> & model,
    const std::map<std::string, std::string> & kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    const auto & params = model->get_parameters();
    const auto & results = model->get_results();

    for (const auto & param_res : kv_param_res_names) {
        const auto & param_name = param_res.first;
        const auto & res_name = param_res.second;

        auto param_it = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<v0::Parameter> & node) {
            return node->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        auto res_it = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<v0::Result> & node) {
            return node->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::Result> res = *res_it;
        pairs.emplace_back(param, res);
    }
    return pairs;
}

void add_sliced_mask(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {

    auto create_sliced_mask = [&](const std::string & mask_name, const std::string & sliced_name, bool is_static) {
        if ((tensor_map.find(mask_name) != tensor_map.end()) &&
            (tensor_map.find("token_len_per_seq") != tensor_map.end())) {
            auto token_len_per_seq = tensor_map.at("token_len_per_seq").get_node_shared_ptr();
            auto mask = tensor_map.at(mask_name).get_node_shared_ptr();
            std::shared_ptr<ov::Node> mask_sliced;
            if (is_static) {
                mask_sliced = mask;
            } else if (ggml_model_decoder.is_stateful()) {
                auto zero_2d = ov::op::v0::Constant::create(ov::element::i64, {2}, {0,0});
                auto one_2d = ov::op::v0::Constant::create(ov::element::i64, {2}, {1,1});
                auto zero_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto three_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
                auto neg_one_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
                auto axes = ov::op::v0::Constant::create(ov::element::i64, {2}, {-2,-1});
                auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
                auto gather_inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, neg_one_1d, three_1d);
                auto reshaped_inp_pos = std::make_shared<ov::op::v1::Reshape>(gather_inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}), false);
                auto inp_pos_incremented = std::make_shared<ov::op::v1::Add>(reshaped_inp_pos, ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}));
                auto stop = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{token_len_per_seq, std::make_shared<v1::ConvertLike>(inp_pos_incremented, token_len_per_seq)}, 0);
                mask_sliced =
                    std::make_shared<ov::op::v8::Slice>(mask, zero_2d, stop, one_2d, axes);
                mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
                mask_sliced->set_friendly_name(sliced_name);
            } else {
                auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
                auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
                mask_sliced = std::make_shared<ov::op::v8::Slice>(mask, zero, token_len_per_seq, one, two);
                mask_sliced = std::make_shared<ov::op::v0::Convert>(mask_sliced, ov::element::f16);
                mask_sliced->set_friendly_name(sliced_name);
            }
            tensor_map.insert({sliced_name, mask_sliced->output(0)});
        }
    };

    create_sliced_mask("self_kq_mask", "KQ_mask_sliced", ggml_model_decoder.is_static());
    create_sliced_mask("self_kq_mask_swa", "KQ_mask_swa_sliced", ggml_model_decoder.is_static());
}

void add_rope_sin_cos(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    int32_t * rope_params = ggml_model_decoder.get_rope_params();
    if (tensor_map.find("inp_pos") == tensor_map.end() || rope_params == nullptr) {
        return;
    }
    auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    if (tensor_map.find("rope_freqs.weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_node_shared_ptr();
    }

    auto sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_weight);
    auto sin_theta = sin_cos.first;
    auto cos_theta = sin_cos.second;

    cos_theta.get_node_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_node_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
void preprocess(TensorMap & tensor_map, GgmlDecoder & ggml_model_decoder) {
    add_sliced_mask(tensor_map, ggml_model_decoder);
    add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(const frontend::InputModel::Ptr & input_model,
                                   const std::unordered_map<std::string, CreatorFunction> & translator_map,
                                   bool naive) :
    m_input_model(input_model),
    m_translator_map(translator_map),
    m_ov_model(nullptr),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    return m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(const frontend::InputModel::Ptr & input_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    const auto & ggml_model = std::dynamic_pointer_cast<InputModel>(input_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    for (const auto & it : ggml_model_decoder->get_model_inputs()) {
        params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second));
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto & it : ggml_model_decoder->get_model_extra_inputs()) {
        if (std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second)) {
            params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second));
        }
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto & it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    auto node_visitor = [&](std::shared_ptr<GgmlDecoder> decoder, int node_idx) {
        auto operation_type = decoder->get_op_type(node_idx);
        if (operation_type == "GGML_OP_NONE") {
            return;
        }

        ov::OutputVector converted_outputs;
        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(), "Translation for operation type ", operation_type,
                                      " is not implemented.");
        NodeContext node_context(decoder, tensor_map, node_idx, this);
        converted_outputs = it->second(node_context);

        const auto & node_output_names = decoder->get_output_names(node_idx);
        FRONT_END_OP_CONVERSION_CHECK(node_output_names.size() == converted_outputs.size(), "Number of ",
                                      operation_type, " outputs greater than number of converted outputs, which are ",
                                      node_output_names.size(), " and ", converted_outputs.size(), " respectively.");

        for (size_t i = 0; i < node_output_names.size(); ++i) {
            auto output_name = node_output_names[i];
            if (i < converted_outputs.size() && converted_outputs[i].get_node_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
    };

    if (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->visit_subgraph(node_visitor);

    for (const auto & name : ggml_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ", name);
        auto result = std::make_shared<v0::Result>(tensor_map->at(name));
        result->set_friendly_name(name);
        results.push_back(result);
    }

    ov::ParameterVector used_params;
    for (const auto & param : params) {
        if (!param->output(0).get_target_inputs().empty()) {
            used_params.push_back(param);
        }
    }
    // if (auto diff = params.size() - used_params.size()) {
    //     GGML_LOG_INFO("%zu parameters are not used in the model.", diff);
    // }
    resulting_model = std::make_shared<Model>(results, used_params);

    apply_transformations(resulting_model);

    // Set WeightlessCacheAttribute on large constants to avoid unnecessary memory copies
    // in the NPUW plugin. Without this attribute, NPUW's LazyTensor constructor
    // (lazy_tensor.cpp, op::Const::Const) will memcpy every constant "in case export
    // occurs", doubling memory usage per compile_model call.
    //
    // The bin_offset field serves as a unique key (not a real file offset) — this is
    // the same convention the GPU plugin uses for non-IR models (see
    // Plugin::set_weightless_cache_attributes in intel_gpu/src/plugin/plugin.cpp).
    // Each constant must have a distinct bin_offset, otherwise GPU's weightless cache
    // import will map multiple constants to the same data.
    //
    // Small constants (< 16 elements) are excluded since they may be introduced by
    // optimization patterns and the overhead is negligible.
    size_t offset = 0;
    for (auto & node : resulting_model->get_ordered_ops()) {
        if (auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
            cnst && cnst->get_byte_size() / cnst->get_element_type().size() >= 16) {
            auto & rt_info = cnst->get_rt_info();
            if (rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static()) == rt_info.end()) {
                rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] =
                    ov::WeightlessCacheAttribute(cnst->get_byte_size(), offset++, cnst->get_element_type());
            }
        }
    }
    return resulting_model;
}

std::shared_ptr<Model> TranslateSession::apply_transformations(std::shared_ptr<Model> model) {
    auto ggml_model_decoder = std::dynamic_pointer_cast<InputModel>(m_input_model)->get_model_decoder();
    {
        ov::pass::Manager manager;
        manager.set_per_pass_validation(true);
        manager.register_pass<ov::pass::MarkCompressedFloatConstants>();

        if (ggml_model_decoder->is_stateful()) {
            const auto kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
            const auto kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
            manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
        }

        if (ggml_model_decoder->is_static()) {
            manager.register_pass<pass::SqueezeMatmul>();
        }
        manager.run_passes(model);
        if (ggml_model_decoder->is_stateful()) {
            auto output_names = ggml_model_decoder->get_model_output_names();
            std::map<std::string, int> model_output_indexes;
            for (size_t i=0; i<output_names.size(); i++) {
                model_output_indexes.insert(std::make_pair(output_names[i], i));
            }
            ov::preprocess::PrePostProcessor ppp(model);
            for (size_t i=0; i<model->get_output_size(); i++) {
                auto output_friendly_name = model->output(i).get_node_shared_ptr()->get_friendly_name();
                auto output_id = model_output_indexes[output_friendly_name];
                auto model_output_shape = model->output(i).get_partial_shape();
                auto decoder_output_shape = ggml_model_decoder->get_output_shape(output_id);
                if (model_output_shape.rank().is_static() && decoder_output_shape.rank().is_static()
                    && model_output_shape.rank().get_length() + 1 == decoder_output_shape.rank().get_length()
                    && decoder_output_shape[0].is_static() && decoder_output_shape[0].get_length() == 1) {
                    ppp.output(i).postprocess().custom([](const ov::Output<ov::Node>& node) {
                        auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
                        return std::make_shared<ov::op::v0::Unsqueeze>(node, axes);
                    });
                }
            }
            model = ppp.build();
        }
    }
    return model;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov


/* Inlined openvino/utils.cpp */

#include "ggml-impl-defs.inc"

#include <cmath>
#include <cstddef>
#include <ctime>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

std::string getCurrentTime() {
    std::time_t now = std::time(nullptr);
    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return buf;
}

void num_inputs_check(const NodeContext & context, size_t min_inputs, size_t max_inputs) {
    auto input_size = context.get_input_size();
    FRONT_END_OP_CONVERSION_CHECK(input_size >= min_inputs, "Got less inputs than expected");
    FRONT_END_OP_CONVERSION_CHECK(input_size <= max_inputs, "Got more inputs than expected");
}

int non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    int dim = nb.size() - 1;
    size_t bytes = nb[dim];
    for (int i = dim; i > 0; i--) {
        bytes *= ne[i];
        if (bytes != nb[i - 1]) {
            return i;
        }
    }
    return 0;
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         const std::vector<int> & dims) {
    using namespace ov::op;
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node> & node, const std::vector<int> & dims) {
    return get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(node), dims);
}

OutputVector rename_outputs_with_suffix(const OutputVector & outputs, const std::string & suffix) {
    for (const auto & output : outputs) {
        auto node = output.get_node_shared_ptr();
        std::string name = node->get_friendly_name();
        name += "_";
        name += suffix;
        node->set_friendly_name(name);
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    return outputs;
}

namespace {
ov::Output<ov::Node> rope_yarn_ramp_mix(int n_dims, const float corr_dims[2], float ext_factor) {
    int half_n_dims = n_dims / 2;
    std::vector<float> dim_ids_vec(half_n_dims);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0);
    auto dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) half_n_dims}, dim_ids_vec);
    auto corr_low = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    auto corr_high = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    auto denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::Subtract>(corr_high, corr_low),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    auto ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::Subtract>(dim_ids, corr_low), denom);
    auto ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    auto one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    auto ramp_inverted = std::make_shared<ov::op::v1::Subtract>(one, ramp_clamped);
    auto ext_factor_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_factor});
    auto ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_factor_node);
    return ramp_mix;
}

float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif
    return n_dims * logf(n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(base));
}

void ggml_rope_yarn_corr_dims(int n_dims,
                              int n_ctx_orig,
                              float freq_base,
                              float beta_fast,
                              float beta_slow,
                              float dims[2]) {
    float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    float end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<float>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           bool imrope,
                                                           bool stateful) {
    if (stateful) {
        inp_pos = std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } else if (imrope) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, true);
        auto pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } else {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        auto pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    const int n_dims = rope_params[1];
    const size_t n_dims_half = n_dims >> 1;
    const int n_ctx_orig = rope_params[4];
    memcpy(&freq_base, rope_params + 5, sizeof(float));
    memcpy(&freq_scale, rope_params + 6, sizeof(float));
    memcpy(&ext_factor, rope_params + 7, sizeof(float));
    memcpy(&attn_factor, rope_params + 8, sizeof(float));
    memcpy(&beta_fast, rope_params + 9, sizeof(float));
    memcpy(&beta_slow, rope_params + 10, sizeof(float));

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    std::vector<float> factor(n_dims_half);

    Output<Node> freq_factors;

    Output<Node> theta;
    float mscale = attn_factor;
    if (imrope) {
        std::vector<int64_t> gather_indices(n_dims_half);
        for (size_t j = 0; j < n_dims_half; j++) {
            gather_indices[j] = j % 3;
            factor[j] = std::pow(theta_scale, j);
        }
        auto gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{n_dims_half}, gather_indices);
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        auto factor_const = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{n_dims_half}, factor);
        theta = std::make_shared<ov::op::v1::Multiply>(inp_pos, factor_const);
    } else {
        float corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        factor[0] = 1.0f;
        for (size_t i = 1; i < factor.size(); i++) {
            factor[i] = theta_scale * factor[i - 1];
        }
        if (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, factor.size()}, factor);
        } else {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, factor.size()}, factor);
        }
        if (rope_freqs_weight) {
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_freqs_weight);
        }

        auto theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        auto theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        if (ext_factor == 0.0f) {
            theta = theta_interp;
        } else {
            auto ramp_mix = rope_yarn_ramp_mix(n_dims, corr_dims, ext_factor);
            Output<Node> one;
            if (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } else {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            auto one_minus_ramp = std::make_shared<ov::op::v1::Subtract>(one, ramp_mix);

            theta = std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                      std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::log(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    if (!imrope) {
        auto mscale_node = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_node);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_node);
    }

    return std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_input(const NodeContext & context, int input_index, int slice_len) {
    // Only works for VIEW operations that slice at the lowest dimension
    // If the VIEW also reshape the result, `slice_len` should be provided
    auto input = context.get_input(input_index);
    auto * op_params = (size_t *) context.get_input_op_params(input_index);
    auto src1_stride = context.get_input_stride(input_index);

    int64_t split_addr = op_params[0] / src1_stride[3];
    if (slice_len == 0) {
        slice_len = context.get_input_shape(input_index)[3].get_length();
    }
    int64_t slice_end = split_addr + slice_len;

    auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {split_addr});
    auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_end});
    auto stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    auto sliced = std::make_shared<ov::op::v8::Slice>(input, begin, end, stride, axes);
    return sliced;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
