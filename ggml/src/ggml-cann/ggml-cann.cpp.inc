/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "ggml-cann.h.inc"

#include "ggml-backend-impl-defs.inc"
/**
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/* Inlined common.h.inc */
/*
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include "../ggml-impl-defs.inc"
#include "../include/ggml-cann.h.inc"
#include "../include/ggml.h.inc"

#include <acl/acl.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#define MATRIX_ROW_PADDING    512
#define GGML_CANN_MAX_STREAMS 8

/**
 * @brief Handles CANN-related errors by printing an error message and
 *        terminating the program.
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number at which the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_cann_error(const char * stmt, const char * func, const char * file, int line, const char * msg);

/**
 * @brief Checks the result of a CANN function call and invokes the error
 *        handler if the call fails.
 * @param stmt The CANN function call to check.
 * @param success The success code that indicates the call was successful.
 * @param error_fn The function to call to retrieve the error message.
 */
#define ACL_CHECK_GEN(stmt, success, error_fn)                                \
    do {                                                                      \
        int err_code = (stmt);                                                \
        if (err_code != (success)) {                                          \
            ggml_cann_error(#stmt, __func__, __FILE__, __LINE__, error_fn()); \
        }                                                                     \
    } while (0);

#define ACL_CHECK(stmt) ACL_CHECK_GEN(stmt, 0, aclGetRecentErrMsg)

/**
 * @brief Contains information about CANN devices.
 */
struct ggml_cann_device_info {
    /**
     * @brief Number of CANN devices available.
     */
    int32_t device_count;

    /**
     * @brief Information about a single CANN device.
     */
    struct cann_device_info {
        int    cc;              /**< Compute capability.                   */
        size_t smpb;            /**< Maximum shared memory per block.      */
        bool   vmm;             /**< Virtual memory support.               */
        size_t vmm_granularity; /**< Granularity of virtual memory.        */
        size_t total_vram;      /**< Total video RAM available on the device. */
    };

    cann_device_info devices[GGML_CANN_MAX_DEVICES] = {}; /**< Array of CANN device information. */
};

const ggml_cann_device_info & ggml_cann_info();

void    ggml_cann_set_device(int32_t device);

std::optional<std::string> get_env_as_lowercase(const std::string & name);
bool                       parse_bool(const std::string & value);
int                        parse_integer(const std::string & value);

/**
 * @brief Abstract base class for memory pools used by CANN.
 */
struct ggml_cann_pool {
    /**
     * @brief Virtual destructor for the memory pool.
     */
    virtual ~ggml_cann_pool() = default;

    /**
     * @brief Allocates memory from the pool.
     *
     * @param size         The size of the memory block to allocate.
     * @param actual_size  Pointer to a variable where the actual allocated size
     *                     will be stored.
     * @return             Pointer to the allocated memory block.
     */
    virtual void * alloc(size_t size, size_t * actual_size) = 0;

    /**
     * @brief Frees a previously allocated memory block.
     *
     * @param ptr   Pointer to the memory block to free.
     * @param size  Size of the memory block to free.
     * @note Note that all CANN opertors are running async. Make sure memory is
     *       still avaiable before this operator finished.
     */
    virtual void free(void * ptr, size_t size) = 0;
};

/**
 * @brief RAII wrapper for managing memory allocations from a CANN memory pool.
 */
struct ggml_cann_pool_alloc {
    ggml_cann_pool * pool        = nullptr; /**< Pointer to the memory pool. */
    void *           ptr         = nullptr; /**< Pointer to the allocated memory block. */
    size_t           actual_size = 0;       /**< Actual size of the allocated memory block. */

    /**
     * @brief Default constructor.
     */
    ggml_cann_pool_alloc() = default;

    /**
     * @brief Constructor that initializes the memory pool.
     * @param pool Reference to the memory pool.
     */
    explicit ggml_cann_pool_alloc(ggml_cann_pool & pool) : pool(&pool) {}

    /**
     * @brief Constructor that initializes the memory pool and allocates memory.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     */
    ggml_cann_pool_alloc(ggml_cann_pool & pool, size_t size) : pool(&pool) { alloc(size); }

    /**
     * @brief Destructor that frees the allocated memory block.
     */
    ~ggml_cann_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    /**
     * @brief Allocates memory from the pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    void * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = pool->alloc(size, &this->actual_size);
        return ptr;
    }

    /**
     * @brief Allocates memory from a specific memory pool.
     * @param pool Reference to the memory pool.
     * @param size Size of the memory block to allocate.
     * @return Pointer to the allocated memory block.
     */
    void * alloc(ggml_cann_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    /**
     * @brief Gets the pointer to the allocated memory block.
     * @return Pointer to the allocated memory block.
     */
    void * get() { return ptr; }

    // Deleted copy constructor
    ggml_cann_pool_alloc(const ggml_cann_pool_alloc &) = delete;

    // Deleted move constructor
    ggml_cann_pool_alloc(ggml_cann_pool_alloc &&) = delete;

    // Deleted copy assignment operator
    ggml_cann_pool_alloc & operator=(const ggml_cann_pool_alloc &) = delete;

    // Deleted move assignment operator
    ggml_cann_pool_alloc & operator=(ggml_cann_pool_alloc &&) = delete;
};

#ifdef USE_ACL_GRAPH
struct ggml_graph_node_properties {
    // dst tensor
    void *    node_address;
    ggml_type node_type;
    int64_t   ne[GGML_MAX_DIMS];
    size_t    nb[GGML_MAX_DIMS];

    // src tensor
    void *    src_address[GGML_MAX_SRC];
    ggml_type src_type[GGML_MAX_SRC];
    int64_t   src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
    size_t    src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];

    // op
    ggml_op node_op;
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

    /**
     * @brief Check if a ggml tensor node matches this property set.
     *
     * This function compares all relevant fields (address, op type, shape, source inputs, op params)
     * to determine whether the current node matches these previously recorded properties.
     *
     * @param node The current ggml tensor node.
     * @return true if all fields match (excluding GGML_OP_VIEW); false otherwise.
     */
    bool has_matching_properties(ggml_tensor * node) {
        if (node->data != this->node_address && node->op != GGML_OP_VIEW) {
            return false;
        }

        if (node->op != this->node_op) {
            return false;
        }

        if (node->type != this->node_type) {
            return false;
        }

        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            if (node->ne[i] != this->ne[i]) {
                return false;
            }
            if (node->nb[i] != this->nb[i]) {
                return false;
            }
        }

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node->src[i]) {
                if (node->src[i]->data != this->src_address[i] && node->op != GGML_OP_VIEW) {
                    return false;
                }

                if (node->src[i]->type != this->src_type[i]) {
                    return false;
                }

                for (int d = 0; d < GGML_MAX_DIMS; d++) {
                    if (node->src[i]->ne[d] != this->src_ne[i][d]) {
                        return false;
                    }
                    if (node->src[i]->nb[d] != this->src_nb[i][d]) {
                        return false;
                    }
                }
            } else {
                if (this->src_address[i] != nullptr) {
                    return false;
                }
            }
        }

        return memcmp(this->op_params, node->op_params, GGML_MAX_OP_PARAMS) == 0;
    }
};

struct ggml_cann_graph {
    ~ggml_cann_graph() {
        if (graph != nullptr) {
            ACL_CHECK(aclmdlRIDestroy(graph));
        }
    }

    aclmdlRI graph = nullptr;

    std::vector<ggml_graph_node_properties> ggml_graph_properties;

    /**
     * @brief Create a new CANN graph from a ggml computation graph.
     *
     * This function creates a new ggml_cann_graph object and fills its node properties
     * (operation type, dimensions, strides, input sources, and operation parameters)
     * based on the current ggml computation graph.
     *
     * Each node in the ggml graph is mapped to a property entry in the new CANN graph:
     * - node address
     * - operation type
     * - shape (ne) and strides (nb)
     * - source tensor addresses
     * - operation parameters
     *
     * @param cgraph The current ggml computation graph.
     * @return Pointer to the newly created ggml_cann_graph object.
     */
    static ggml_cann_graph * create_from_cgraph(ggml_cgraph * cgraph) {
        ggml_cann_graph * new_graph = new ggml_cann_graph();
        new_graph->ggml_graph_properties.resize(cgraph->n_nodes);

        for (int node_idx = 0; node_idx < cgraph->n_nodes; ++node_idx) {
            ggml_tensor * node = cgraph->nodes[node_idx];
            auto &        prop = new_graph->ggml_graph_properties[node_idx];

            prop.node_address = node->data;
            prop.node_op      = node->op;
            prop.node_type    = node->type;

            std::copy_n(node->ne, GGML_MAX_DIMS, prop.ne);
            std::copy_n(node->nb, GGML_MAX_DIMS, prop.nb);

            for (int src = 0; src < GGML_MAX_SRC; ++src) {
                if (node->src[src]) {
                    prop.src_address[src] = node->src[src]->data;
                    prop.src_type[src]    = node->src[src]->type;
                    std::copy_n(node->src[src]->ne, GGML_MAX_DIMS, prop.src_ne[src]);
                    std::copy_n(node->src[src]->nb, GGML_MAX_DIMS, prop.src_nb[src]);
                } else {
                    prop.src_address[src] = nullptr;
                    prop.src_type[src]    = GGML_TYPE_COUNT;
                    std::fill_n(prop.src_ne[src], GGML_MAX_DIMS, 0);
                    std::fill_n(prop.src_nb[src], GGML_MAX_DIMS, 0);
                }
            }

            memcpy(prop.op_params, node->op_params, GGML_MAX_OP_PARAMS);
        }

        return new_graph;
    }

    /**
     * @brief Check whether this CANN graph matches the given ggml computation graph.
     *
     * This function compares the number of nodes and each node's properties
     * (operation type, dimensions, strides, inputs, and operation parameters)
     * to determine whether this CANN graph matches the given ggml graph.
     *
     * @param cgraph The current ggml computation graph.
     * @return true if this CANN graph matches the ggml graph; false otherwise.
     */
    bool matches_cgraph(ggml_cgraph * cgraph) {
        if (this->ggml_graph_properties.size() != static_cast<size_t>(cgraph->n_nodes)) {
            return false;
        }

        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (!this->ggml_graph_properties[i].has_matching_properties(cgraph->nodes[i])) {
                return false;
            }
        }

        return true;
    }
};

/**
 * @brief LRU cache for managing ggml_cann_graph objects.
 *
 * This class maintains a list of shared_ptr to ggml_cann_graph objects
 * and enforces a maximum capacity. It provides methods to push new graphs,
 * move existing graphs to the front (most recently used), and clear the cache.
 */
struct ggml_cann_graph_lru_cache {
    size_t capacity;                         /**< Maximum number of graphs in the cache. */

    std::list<ggml_cann_graph *> cache_list; /**< List storing cached graphs as raw pointers. */

    ggml_cann_graph_lru_cache() { capacity = parse_integer(get_env_as_lowercase("GGML_CANN_GRAPH_CACHE_CAPACITY").value_or("12")); }

    /**
     * @brief Push a new graph to the front of the cache.
     * If the cache exceeds capacity, the least recently used graph is deleted.
     * @param new_node Pointer to the new ggml_cann_graph to cache.
     *        Ownership is transferred to the cache (cache will delete it).
     */
    void push(ggml_cann_graph * new_node) {
        if (cache_list.size() >= capacity) {
            ggml_cann_graph * old = cache_list.back();
            cache_list.pop_back();
            delete old;  // free the old graph
        }
        cache_list.push_front(new_node);
    }

    /**
     * @brief Clear all graphs from the cache (also frees memory).
     */
    void clear() {
        for (auto ptr : cache_list) {
            delete ptr;
        }
        cache_list.clear();
    }

    /**
     * @brief Destructor that clears the cache and frees all cached graphs.
     */
    ~ggml_cann_graph_lru_cache() { clear(); }

    /**
     * @brief Find a cached CANN graph that matches the given ggml graph and move it to front.
     *
     * This function iterates through the cached CANN graphs stored in the LRU cache and
     * compares them against the given ggml computation graph. If a matching graph is found,
     * it is promoted to the front of the LRU cache and returned. Otherwise, the function
     * returns nullptr.
     *
     * @param cgraph The current ggml computation graph.
     * @return true if found; false otherwise.
     */
    bool find_and_move_to_front(ggml_cgraph * cgraph) {
        for (auto & graph_ptr : this->cache_list) {
            if (graph_ptr->matches_cgraph(cgraph)) {
                cache_list.remove(graph_ptr);
                cache_list.push_front(graph_ptr);
                return true;
            }
        }
        return false;
    }
};
#endif  // USE_ACL_GRAPH

struct ggml_cann_rope_cache {
    ~ggml_cann_rope_cache() {
        if (theta_scale_cache) {
            ACL_CHECK(aclrtFree(theta_scale_cache));
        }
        if (sin_cache) {
            ACL_CHECK(aclrtFree(sin_cache));
        }
        if (cos_cache) {
            ACL_CHECK(aclrtFree(cos_cache));
        }
        if (position_select_index) {
            ACL_CHECK(aclrtFree(position_select_index));
        }
        if (theta_scale_exp_host) {
            free(theta_scale_exp_host);
        }
        if (position_select_index_host) {
            free(position_select_index_host);
        }
        if (yarn_ramp_cache) {
            ACL_CHECK(aclrtFree(yarn_ramp_cache));
        }
    }

    bool equal(int64_t theta_scale_length,
               int64_t position_length,
               float   ext_factor,
               float   theta_scale,
               float   freq_scale,
               float   attn_factor,
               bool    is_neox,
               bool    indep_sects,
               bool    mrope_used,
               bool    is_imrope,
               int     sections[4]) {
        return this->theta_scale_length == theta_scale_length && this->position_length == position_length &&
               this->ext_factor == ext_factor && this->theta_scale == theta_scale && this->freq_scale == freq_scale &&
               this->attn_factor == attn_factor && this->is_neox == is_neox && this->indep_sects == indep_sects &&
               this->mrope_used == mrope_used && this->is_imrope == is_imrope && this->sections[0] == sections[0] &&
               this->sections[1] == sections[1] && this->sections[2] == sections[2] && this->sections[3] == sections[3];
    }

    void set(int64_t theta_scale_length,
             int64_t position_length,
             float   ext_factor,
             float   theta_scale,
             float   freq_scale,
             float   attn_factor,
             bool    is_neox,
             bool    indep_sects,
             bool    mrope_used,
             bool    is_imrope,
             int     sections[4]) {
        this->theta_scale_length = theta_scale_length;
        this->position_length    = position_length;
        this->ext_factor         = ext_factor;
        this->theta_scale        = theta_scale;
        this->freq_scale         = freq_scale;
        this->attn_factor        = attn_factor;
        this->is_neox            = is_neox;
        this->indep_sects        = indep_sects;
        this->mrope_used         = mrope_used;
        this->is_imrope          = is_imrope;
        this->sections[0]        = sections[0];
        this->sections[1]        = sections[1];
        this->sections[2]        = sections[2];
        this->sections[3]        = sections[3];
    }

    // memory cache, prepare before inferencing.
    void *  theta_scale_cache          = nullptr;
    float * theta_scale_exp_host       = nullptr;
    int *   position_select_index_host = nullptr;
    void *  position_select_index      = nullptr;
    void *  yarn_ramp_cache            = nullptr;
    // sin/cos cache, used only to accelerate first layer on each device
    void *  sin_cache                  = nullptr;
    void *  cos_cache                  = nullptr;
    // Properties to check before reusing the sincos cache
    int64_t theta_scale_length         = 0;
    int64_t position_length            = 0;
    bool    cached                     = false;
    float   ext_factor                 = 0.0f;
    float   theta_scale                = 0.0f;
    float   freq_scale                 = 0.0f;
    float   attn_factor                = 0.0f;
    bool    is_neox                    = false;
    bool    indep_sects                = false;
    bool    mrope_used                 = false;
    int     sections[4]                = { 0, 0, 0, 0 };
    bool    is_imrope                  = false;
};

struct ggml_cann_tensor_cache {
    ~ggml_cann_tensor_cache() {
        if (cache != nullptr) {
            ACL_CHECK(aclrtFree(cache));
        }
    }

    void *  cache = nullptr;
    int64_t size  = 0;
};

/**
 * @brief Context for managing CANN backend operations.
 */
struct ggml_backend_cann_context {
    int32_t     device;               /**< Device ID. */
    std::string name;                 /**< Name of the device. */
    std::string description;          /**< Description of the device. */
    aclrtEvent  copy_event = nullptr; /**< Event for managing copy operations. */
#ifdef USE_ACL_GRAPH
    /// Cached CANN ACL graph used for executing the current ggml computation graph.
    ggml_cann_graph_lru_cache graph_lru_cache;
    bool                      acl_graph_mode = true;
#endif
    bool                   async_mode;
    // Rope Cache
    ggml_cann_rope_cache   rope_cache;
    // Constant Pool
    ggml_cann_tensor_cache rms_norm_one_tensor_cache;
    ggml_cann_tensor_cache rms_norm_zero_tensor_cache;

    aclrtStream streams[GGML_CANN_MAX_STREAMS] = { nullptr }; /**< Array of streams for the device. */

    /**
     * @brief Constructor for initializing the context with a given device.
     * @param device Device ID.
     */
    explicit ggml_backend_cann_context(int device) : device(device), name("CANN" + std::to_string(device)) {
        ggml_cann_set_device(device);
        description = aclrtGetSocName();

#ifdef USE_ACL_GRAPH
        acl_graph_mode = parse_bool(get_env_as_lowercase("GGML_CANN_ACL_GRAPH").value_or("on"));
        GGML_LOG_INFO("%s: device %d execution mode is %s (%s)\n", __func__, device, acl_graph_mode ? "GRAPH" : "EAGER",
                      acl_graph_mode ? "acl graph enabled" : "acl graph disabled");
#endif
    }

    /**
     * @brief Destructor for cleaning up resources.
     */
    ~ggml_backend_cann_context() {
        ggml_cann_set_device(device);
        if (copy_event != nullptr) {
            ACL_CHECK(aclrtDestroyEvent(copy_event));
        }
        for (int i = 0; i < GGML_CANN_MAX_STREAMS; ++i) {
            if (streams[i] != nullptr) {
                ACL_CHECK(aclrtDestroyStream(streams[i]));
            }
        }
    }

    /**
     * @brief Get or create a stream for a given index.
     * @param stream Index of the stream.
     * @return The stream corresponding to the given index.
     */
    aclrtStream stream(int stream) {
        if (streams[stream] == nullptr) {
            // If the device is not set here, destroying the stream later may cause a mismatch
            // between the thread contexts where the stream was created and destroyed.
            // However, I printed the device_id, thread_id, and stream, and they are all consistent.
            ACL_CHECK(aclrtSetDevice(device));
            ACL_CHECK(aclrtCreateStream(&streams[stream]));
        }
        return streams[stream];
    }

    /**
     * @brief Get or create the default stream (index 0).
     * @return The default stream.
     */
    aclrtStream stream() { return stream(0); }

    // TODO: each stream should have a memory pool.
    std::unique_ptr<ggml_cann_pool> mem_pool; /**< Memory pool for the device. */

    /**
     * @brief Create a new memory pool for a given device.
     * @param device Device ID.
     * @return A unique pointer to the new memory pool.
     */
    static std::unique_ptr<ggml_cann_pool> new_pool_for_device(int device);

    /**
     * @brief Get or create the memory pool for the context.
     * @return Reference to the memory pool.
     */
    ggml_cann_pool & pool() {
        if (mem_pool == nullptr) {
            mem_pool = new_pool_for_device(device);
        }
        return *mem_pool;
    }
};


#include <aclnn/aclnn_base.h>

#include <algorithm>
#include <cstring>

/**
 * @brief	Maps a ggml_type to its corresponding aclDataType.
 *
 * @details	This function takes a ggml_type as input and returns the corresponding
 *			aclDataType. It supports mapping for various ggml_types. If the input type
 *			does not match any of the predefined ggml_types, the function returns
 *          ACL_DT_UNDEFINED.
 *
 * @param	type    The ggml_type to be mapped.
 * @return	The corresponding aclDataType. If the input type is not recognized,
 *			ACL_DT_UNDEFINED is returned.
 */
aclDataType ggml_cann_type_mapping(ggml_type type);

// Deleter for acl objects.
template <typename T, aclError (*DestroyFunc)(const T *)> struct acl_deleter {
    void operator()(T * ptr) const noexcept {
        if (ptr) {
            ACL_CHECK(DestroyFunc(ptr));
        }
    }
};

using acl_tensor_ptr      = std::unique_ptr<aclTensor, acl_deleter<aclTensor, aclDestroyTensor>>;
using acl_int_array_ptr   = std::unique_ptr<aclIntArray, acl_deleter<aclIntArray, aclDestroyIntArray>>;
using acl_scalar_ptr      = std::unique_ptr<aclScalar, acl_deleter<aclScalar, aclDestroyScalar>>;
using acl_tensor_list_ptr = std::unique_ptr<aclTensorList, acl_deleter<aclTensorList, aclDestroyTensorList>>;

/**
 * @brief   Creates an ACL tensor from a ggml_tensor with optional shape.
 *
 * @details This function creates an ACL tensor based on the properties of the
 *          provided ggml_tensor. It supports customer shape by adjusting dimensions
 *          and strides accordingly. If customer shape is applied, additional
 *          dimensions and strides are calculated based on the provided parameters.
 *
 * @param   tensor      Pointer to the ggml_tensor to be converted to ACL tensor.
 * @param   ne          Pointer to an array containing dimensions. Defaults to nullptr
 *                      if no customer shape is applied.
 * @param   nb          Pointer to an array containing strides. Defaults to nullptr
 *                      if no customer shape is applied.
 * @param   dims        Number of dimensions in the tensor. Defaults to 0 if no customer
 *                      shape is applied.
 * @param   format      ACL tensor format. Defaults to ACL_FORMAT_ND.
 * @param   offset      Offset in bytes for the ACL tensor data. Defaults to 0.
 * @return  Pointer to the created ACL tensor.
 */
acl_tensor_ptr ggml_cann_create_tensor(const ggml_tensor * tensor,
                                       int64_t *           ne     = nullptr,
                                       size_t *            nb     = nullptr,
                                       int64_t             dims   = 0,
                                       aclFormat           format = ACL_FORMAT_ND,
                                       size_t              offset = 0);

/**
 * @brief   Template for creating an ACL tensor from provided parameters. typename TYPE
 *          should be size_t or float.
 *
 * @details This function creates an ACL tensor using the provided data pointer,
 *          data type, dimensions, strides, format, offset, and additional parameters.
 *          It calculates necessary dimensions and strides based on the provided ne and nb
 *          arrays, adjusting them for the ACL tensor creation. The ACL storage length
 *          is also calculated based on the provided dimensions and strides.
 *
 * @param   data_ptr    Pointer to the data buffer for the ACL tensor.
 * @param   dtype       ACL data type of the tensor.
 * @param   type_size   Size of each element in the tensor data buffer.
 * @param   ne          Pointer to an array containing tensor dimensions.
 * @param   nb          Pointer to an array containing tensor strides.
 * @param   dims        Number of dimensions of the tensor.
 * @param   format      ACL tensor format. Defaults to ACL_FORMAT_ND.
 * @param   offset      Offset in bytes for the ACL tensor data. Defaults to 0.
 * @return  Pointer to the created ACL tensor.
 */
template <typename TYPE>
acl_tensor_ptr ggml_cann_create_tensor(void *      data_ptr,
                                       aclDataType dtype,
                                       TYPE        type_size,
                                       int64_t *   ne,
                                       TYPE *      nb,
                                       int64_t     dims,
                                       aclFormat   format = ACL_FORMAT_ND,
                                       size_t      offset = 0) {
    int64_t tmp_ne[GGML_MAX_DIMS * 2];
    int64_t tmp_stride[GGML_MAX_DIMS * 2];

    memcpy(tmp_ne, ne, dims * sizeof(int64_t));
    for (int i = 0; i < dims; i++) {
        tmp_stride[i] = nb[i] / type_size;
    }

    int64_t acl_storage_len = 1;
    for (int i = 0; i < dims; i++) {
        acl_storage_len += (tmp_ne[i] - 1) * tmp_stride[i];
    }

    std::reverse(tmp_ne, tmp_ne + dims);
    std::reverse(tmp_stride, tmp_stride + dims);

    aclTensor * raw =
        aclCreateTensor(tmp_ne, dims, dtype, tmp_stride, offset / type_size, format, &acl_storage_len, 1, data_ptr);

    return acl_tensor_ptr(raw);
}

/**
 * @brief Create an ACL int array resource wrapped in a smart pointer.
 *
 * This function constructs an aclIntArray from the provided int64_t values
 * and returns it as an acl_int_array_ptr (a std::unique_ptr with a custom
 * deleter). The returned pointer owns the ACL resource and will automatically
 * destroy it via aclDestroyIntArray().
 *
 * @param value  Pointer to the int64_t elements.
 * @param size   Number of elements in value.
 *
 * @return A smart pointer managing the created ACL int array.
 */
acl_int_array_ptr ggml_cann_create_int_array(const int64_t * value, uint64_t size);

/**
 * @brief Create an ACL scalar resource wrapped in a smart pointer.
 *
 * This function constructs an aclScalar from the raw value pointer and ACL
 * data type, then returns it as an acl_scalar_ptr (a std::unique_ptr with
 * a custom deleter). The returned pointer owns the ACL scalar and will
 * automatically destroy it via aclDestroyScalar().
 *
 * @param value     Pointer to the raw scalar memory.
 * @param dataType  ACL data type of the scalar.
 *
 * @return A smart pointer managing the created ACL scalar.
 */
acl_scalar_ptr ggml_cann_create_scalar(void * value, aclDataType dataType);

/**
 * @brief Create an ACL tensor list from multiple tensor smart pointers.
 *
 * This function accepts a variadic list of acl_tensor_ptr (a unique_ptr with
 * custom deleter) and produces an aclTensorList using aclCreateTensorList().
 *
 * The lifecycle management of the tensor objects changes as follows:
 *  - aclCreateTensorList() takes ownership of the tensors
 *  - Each input smart pointer releases ownership using release()
 *  - As a result, the tensors will NOT be destroyed by unique_ptr
 *  - Instead, they will be destroyed when aclDestroyTensorList() is called
 *
 * This ensures correct ownership transfer and prevents double-free situations.
 *
 * @param acl_tensor_ptr  Variadic template parameter; each argument must be
 *                         a unique_ptr-like type supporting get() and release().
 *
 * @param tensors  Variadic list of acl_tensor_ptr objects. Ownership of
 *                         each tensor is transferred away from these smart pointers.
 *
 * @return A smart pointer (acl_tensor_list_ptr) owning the created ACL tensor list.
 *
 * @note This implementation is C++11 compatible. The ownership-release process is
 *       executed using a pack expansion inside an initializer list.
 */
template <typename... acl_tensor_ptr> acl_tensor_list_ptr ggml_cann_create_tensor_list(acl_tensor_ptr &&... tensors) {
    aclTensor *     raw_tensors[] = { tensors.get()... };
    aclTensorList * raw           = aclCreateTensorList(raw_tensors, sizeof...(tensors));
    // aclTensor will release by aclTensorList, so release ownership without
    // destroying the tensor
    int             dummy[]       = { (tensors.release(), 0)... };
    GGML_UNUSED(dummy);
    return acl_tensor_list_ptr(raw);
}

/**
 * @brief   Checks if tensors require broadcasting based on their shapes.
 *
 * @details This function determines if two ggml_tensors need to be broadcasted for
 *          element-wise operations. Broadcasting is necessary if the shapes of the
 *          tensors are not identical and no dimension in either tensor equals 1.
 *
 * @param   t0      Pointer to the first ggml_tensor.
 * @param   t1      Pointer to the second ggml_tensor.
 * @return  True if broadcasting is needed, False otherwise.
 *
 * @remarks This function iterates over the dimensions of t0 and t1. It checks if each
 *          dimension in t1 differs from t0's corresponding dimension and is not equal
 *          to 1. If such a dimension is found, broadcasting is required to align t1
 *          with t0 for element-wise operations.
 */
bool ggml_cann_need_bcast(const ggml_tensor * t0, const ggml_tensor * t1);

/**
 * @brief   Computes broadcast shapes and strides for two ggml_tensors.
 *
 * @details This function calculates the broadcast shapes and strides for two ggml_tensors,
 *          following the broadcasting rules similar to numpy. It adjusts dimensions and
 *          strides to ensure compatibility for element-wise operations where one tensor
 *          can be broadcasted to match the shape of another tensor.
 *
 * @param   src0                Pointer to the first ggml_tensor.
 * @param   src1                Pointer to the second ggml_tensor.
 * @param   bcast_ne_src0       Output array to store broadcasted dimensions for src0.
 * @param   bcast_ne_src1       Output array to store broadcasted dimensions for src1.
 * @param   bcast_nb_src0       Output array to store broadcasted strides for src0.
 * @param   bcast_nb_src1       Output array to store broadcasted strides for src1.
 * @return  Number of dimensions in the broadcasted shape.
 *
 * @pre     ggml_can_repeat(src1, src0) must return true, indicating src1 can be broadcasted
 *          to match src0.
 *
 * @remarks This function iterates over the dimensions of src0 and src1, calculating the
 *          necessary broadcast dimensions and strides. If a dimension requires broadcasting
 *          (i.e., its size in src1 is smaller than in src0), an additional dimension is
 *          added with size calculated to match src0's dimension. This adjustment ensures
 *          that src1 can be element-wise broadcasted to src0's shape.
 *
 *  How it works:
 *
 *  if dim0 has padding.
 *  a -> (2, 2) padding = 2
 *   a: [[1, 2, *, *]
 *       [2, 3, *, *]]
 *  nb = (8, 4, 2)
 *
 *  if a should bcast with b -> (2, 4)
 *  b' -> (2, 2, 2)
 *  b : [[1, 2, 3, 4, *, *]
 *       [5, 6, 7, 8, *, *]]
 *  nb = (12, 6, 1)
 *
 *  after bcast:
 *  a' -> (2, 1, 2)
 *  a': [[[1, 2], *, *]
 *       [[2, 3], *, *]]
 *  nb = (8, 4, 2, 1)
 *
 *  b' : [[[1, 2], [3, 4], *, *]
 *        [[5, 6], [7, 8], *, *]]
 *  nb = (12, 6, 2, 1)
 *  \endcode
 *
 *  dim1 in a inserted dim, should add nb for dim1,
 *  and all other nb moves to next in order.
 */
int64_t ggml_cann_get_bcast_shape(const ggml_tensor * src0,
                                  const ggml_tensor * src1,
                                  int64_t *           bcast_ne_src0,
                                  int64_t *           bcast_ne_src1,
                                  size_t *            bcast_nb_src0,
                                  size_t *            bcast_nb_src1);

// Bcast macro to avoid duplicate code.
#define BCAST_SHAPE(src0, src1)                                                                      \
    int64_t bcast_##src0##_ne[GGML_MAX_DIMS * 2];                                                    \
    int64_t bcast_##src1##_ne[GGML_MAX_DIMS * 2];                                                    \
    size_t  bcast_##src0##_nb[GGML_MAX_DIMS * 2];                                                    \
    size_t  bcast_##src1##_nb[GGML_MAX_DIMS * 2];                                                    \
    int64_t bcast_dims = ggml_cann_get_bcast_shape(src0, src1, bcast_##src0##_ne, bcast_##src1##_ne, \
                                                   bcast_##src0##_nb, bcast_##src1##_nb);

#define BCAST_PARAM(tensor) bcast_##tensor##_ne, bcast_##tensor##_nb, bcast_dims

/**
 * @brief Calculates broadcast shapes for matrix multiplication.
 *
 * @details This function computes the broadcast shapes required for matrix multiplication
 *          based on the input, weight, and destination tensor shapes. It ensures that the
 *          dimensions of weight tensors are expanded appropriately to satisfy matrix
 *          multiplication broadcast rules.
 *
 * @param input_ne      Array containing the dimensions of the input tensor.
 * @param weight_ne     Array containing the dimensions of the weight tensor.
 * @param dst_ne        Array containing the dimensions of the destination tensor.
 * @param input_nb      Array containing the strides of the input tensor.
 * @param weight_nb     Array containing the strides of the weight tensor.
 * @param dst_nb        Array containing the strides of the destination tensor.
 * @param bcast_input_ne    Output array for broadcasted input tensor dimensions.
 * @param bcast_weight_ne   Output array for broadcasted weight tensor dimensions.
 * @param bcast_dst_ne      Output array for broadcasted destination tensor dimensions.
 * @param bcast_input_nb    Output array for broadcasted input tensor strides.
 * @param bcast_weight_nb   Output array for broadcasted weight tensor strides.
 * @param bcast_dst_nb      Output array for broadcasted destination tensor strides.
 * @return The number of dimensions in the broadcasted tensors.
 *
 * @remarks This function iterates over the tensor dimensions and calculates the broadcast
 *          shapes needed for matrix multiplication. It ensures that dimensions where
 *          weight tensor requires expansion are appropriately handled to conform with
 *          broadcasting rules.
 * @note compare with ggml_cann_get_bcast_shape, mul_mat broadcast need add this new dim
 *       before cast dim.
 * @sa ggml_cann_get_bcast_shape
 */
int64_t ggml_cann_get_mulmat_bcast_shape(const int64_t * input_ne,
                                         const int64_t * weight_ne,
                                         const int64_t * dst_ne,
                                         const size_t *  input_nb,
                                         const size_t *  weight_nb,
                                         const size_t *  dst_nb,
                                         int64_t *       bcast_input_ne,
                                         int64_t *       bcast_weight_ne,
                                         int64_t *       bcast_dst_ne,
                                         size_t *        bcast_input_nb,
                                         size_t *        bcast_weight_nb,
                                         size_t *        bcast_dst_nb);

// Bcast macro to avoid duplicate code.
#define BCAST_MUL_MAT_SHAPE(input, weight, dst)                                                                  \
    int64_t bcast_##input##_ne[GGML_MAX_DIMS * 2];                                                               \
    int64_t bcast_##weight##_ne[GGML_MAX_DIMS * 2];                                                              \
    int64_t bcast_##dst##_ne[GGML_MAX_DIMS * 2];                                                                 \
    size_t  bcast_##input##_nb[GGML_MAX_DIMS * 2];                                                               \
    size_t  bcast_##weight##_nb[GGML_MAX_DIMS * 2];                                                              \
    size_t  bcast_##dst##_nb[GGML_MAX_DIMS * 2];                                                                 \
    int64_t bcast_dims = ggml_cann_get_mulmat_bcast_shape(                                                       \
        input->ne, weight->ne, dst->ne, input->nb, weight->nb, dst->nb, bcast_##input##_ne, bcast_##weight##_ne, \
        bcast_##dst##_ne, bcast_##input##_nb, bcast_##weight##_nb, bcast_##dst##_nb);

#define BCAST_MUL_MAT_PARAM(tensor) bcast_##tensor##_ne, bcast_##tensor##_nb, bcast_dims


#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_arange.h>
#include <aclnnop/aclnn_argsort.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_tril.h>
#include <aclnnop/aclnn_triu.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_gelu.h>
#include <aclnnop/aclnn_gelu_v2.h>
#include <aclnnop/aclnn_hardsigmoid.h>
#include <aclnnop/aclnn_hardswish.h>
#include <aclnnop/aclnn_leaky_relu.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_logsoftmax.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_norm.h>
#include <aclnnop/aclnn_relu.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_softplus.h>
#include <aclnnop/aclnn_swi_glu.h>
#include <aclnnop/aclnn_geglu.h>
#include <aclnnop/aclnn_slice.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_tanh.h>

#include <functional>
#include <unordered_set>

/**
 * @brief   Repeats a ggml tensor along each dimension to match the dimensions
 *          of another tensor.
 *
 * @details This function repeats the elements of a source ggml tensor along
 *          each dimension to create a destination tensor with the specified
 *          dimensions. The operation is performed using the ACL backend and
 *          executed asynchronously on the device.
 *
 * @param   ctx The CANN context used for operations.
 * @param   dst The ggml tensor representing the destination, which op is
 *              GGML_OP_REPEAT and specifies the desired dimensions.
 */
void ggml_cann_repeat(ggml_backend_cann_context & ctx, ggml_tensor * dst);

void ggml_cann_swiglu(ggml_backend_cann_context & ctx, ggml_tensor * dst);
void ggml_cann_geglu(ggml_backend_cann_context & ctx, ggml_tensor * dst, int64_t approximate);

/**
 * @brief   Applies the Leaky ReLU activation function to a tensor using the CANN
 *          backend.
 *
 * @details This function computes the Leaky ReLU activation for each element of
 *          the input tensor. The Leaky ReLU function allows a small gradient
 *          when the unit is not active (i.e., when the input is negative). The
 *          Leaky ReLU function is defined as:
 *          \f[
 *              \text{dst} = \max(0, src) + \text{negativeSlope} \cdot \min(0,
 *               src)
 *          \f]
 *          `negativeSlope` is in dst->params.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the result of the Leaky ReLU
 *            activation is stored, which op is `GGML_OP_LEAKY_RELU`
 */
void ggml_cann_leaky_relu(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief    Concatenates multiple tensors along a specified dimension using the
 *           CANN backend.
 *
 * @param ctx        The CANN context used for operations.
 * @param tensorList A pointer to the list of tensors to be concatenated.
 * @param dst        The destination tensor where the result of the
 *                   concatenation is stored. dst->op is `GGML_OP_CONCAT`.
 * @param concat_dim The dimension along which the tensors are concatenated.
 *
 * @attention tensorList length should be 2 and the dimension using for concat
 *            default to 1.
 */
void ggml_cann_concat(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Generates a sequence of evenly spaced values within a specified
 *          interval for a ggml tensor using the CANN backend.
 *
 * @details This function creates a sequence of numbers over a specified i
 *          nterval, starting from `start`, ending before `stop`, and
 *          incrementing by `step`. The sequence is stored in the destination
 *          tensor `dst`.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the generated sequence will be stored.
 *            `start`, 'stop' and 'step' are in dst->op_params and dst->op is
 *            `GGML_OP_ARANGE`.
 */
void ggml_cann_arange(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies a clamp operation to the elements of a ggml tensor using the
 *          CANN backend.
 *
 * @details This function clamps the elements of the input tensor `src` to a
 *          specified range defined by `min` and `max` values. The result is
 *          stored in the destination tensor `dst`. The operation is defined as:
 *          \f[
 *              y = \max(\min(x, max\_value), min\_value)
 *           \f]
 *          where `x` is an element of the input tensor, and `y` is the
 *          corresponding element in the output tensor.
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the clamped values will be stored.
 *            dst->op is `GGML_OP_CLAMP`, `min` and `max` value is in dst->params.
 */
void ggml_cann_clamp(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Scales the elements of a ggml tensor by a constant factor using the
 *          CANN backend.
 *
 * @details This function multiplies each element of the input tensor `src` by
 *          a scaling factor `scale`, storing the result in the destination
 *          tensor `dst`. The operation is defined as:
 *          \f[
 *             dst = src \times scale
 *          \f]
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the scaled values will be stored.
 *            dst->op is `GGML_OP_SCALE` and `scale` value is in dst->params.
 */
void ggml_cann_scale(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Sorts the elements of a ggml tensor and returns the indices that
 *          would sort the tensor using the CANN backend.
 *
 * @details This function performs an argsort operation on the input tensor
 *          `src`. It sorts the elements of `src` in either ascending or
 *          descending order, depending on the `GGML_SORT_ORDER_DESC`,
 *          and returns the indices that would sort the original tensor.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the sorted indices will be stored.
 *            dst->op is `GGML_OP_ARGSORT`.
 */
void ggml_cann_argsort(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the Layer Normalization for a ggml tensor using the CANN
 *          backend.
 *
 * @details This function applies the Layer Normalization operation on the
 *          input tensor `src` and stores the result in the destination tensor
 *          `dst`. Layer Normalization normalizes the features at each sample in
 *          a mini-batch independently. It is commonly used in neural networks
 *          to normalize the activations of a layer by adjusting and scaling
 *          the outputs.
 *          The operation is defined as:
 *          \f[
 *              \text { out }=\frac{x-\mathrm{E}[x]}{\sqrt{\text{Var}[x]+eps}}
 *          \f]
 *          `Var` defaults dst->ne[0]. `eps` is in dst->params.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 * @attention `Var` defaults to dst->ne[0].
 */
void ggml_cann_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the L2 Normalization for a ggml tensor using the CANN
 *          backend.
 *
 * @details This function applies the L2 Normalization operation on the
 *          input tensor `src` and stores the result in the destination tensor
 *          `dst`. L2 Normalization scales the input tensor such that the
 *          L2 norm along the specified dimension equals 1. This operation
 *          is commonly used in neural networks for feature normalization
 *          and vector scaling.
 *          The operation is defined as:
 *          \f[
 *              \text{out} = \frac{x}{\sqrt{\sum{x^2}}}
 *          \f]
 *          The normalization is performed along the last dimension by default.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 * @attention The normalization is performed along the last dimension of the
 *            input tensor by default.
 */
void ggml_cann_l2_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the Cross Entropy Loss for a ggml tensor using the CANN
 *          backend.
 *
 * @details This function computes the cross entropy loss between the predicted
 *          logits and target probability distributions. The operation follows
 *          the same computation pattern as the CPU implementation:
 *          1. Applies log_softmax to the logits along the class dimension
 *          2. Element-wise multiplication with target distributions
 *          3. Summation along the class dimension to get per-sample losses
 *          4. Global summation and scaling by -1/nr to get final loss
 *
 *          The computation can be expressed as:
 *          \f[
 *              \text{loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \cdot \log(\text{softmax}(x_{ij}))
 *          \f]
 *          where \f$N\f$ is the total number of samples, \f$C\f$ is the number
 *          of classes, \f$x\f$ are the logits, and \f$y\f$ are the target
 *          probability distributions.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the computed loss will be stored.
 *            This should be a scalar tensor containing the final loss value.
 *
 * @note This implementation computes cross entropy between probability
 *       distributions, not the typical classification cross entropy that
 *       expects class indices as targets. Both input tensors (src0 and src1)
 *       should have the same shape and represent probability distributions
 *       over the class dimension.
 * @note The function expects two source tensors:
 *       - dst->src[0]: Logits tensor (before softmax)
 *       - dst->src[1]: Target probability distributions tensor
 * @note The computation is performed using CANN backend operators including
 *       LogSoftmax, Mul, ReduceSum, and Muls for the final scaling.
 */
void ggml_cann_cross_entropy_loss(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief  Computes the Group Normalization for a ggml tensor using the CANN
 *         backend.
 *
 * @brief  This function applies the Group Normalization operation on the input
 *         tensor `src` and stores the result in the destination tensor `dst`.
 *         Group Normalization divides the channels into groups and normalizes
 *         the features within each group across spatial locations.
 *         It is commonly used in convolutional neural networks to improve
 *         training stability and performance.
 *         The operation is defined as:
 *         \f[
 *             \text { out }=\frac{x-\mathrm{E}[x]}{\sqrt{\text{Var}[x]+eps}}
 *         \f]
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 *            `n_groups` is in dst->params, which split C channel to `n_groups`.
 *            dst->op is `GGML_OP_GROUP_NORM`.
 *
 * @attention eps defaults to 1e-6f.
 */
void ggml_cann_group_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the accumulation of tensors using the CANN backend.
 *
 * @details This function performs an accumulation operation on two tensors.
 *          Depending on the `inplace` flag, it either updates the destination
 *          tensor `dst` in place by adding `alpha * src1` to it, or it creates
 *          a new tensor as the result of `src0 + alpha * src1` and stores it in
 *          `dst`.
 *          The operation is defined as:
 *          \f[
 *               dst = src0 + alpha \times src1
 *          \f]
 *          if `inplace` is `true`, `src0` is equal to 'dst'.
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the accumulated values will be stored.
 *            `inplace` is in dst->params, and dst->op is `GGML_OP_ACC`.
 */
void ggml_cann_acc(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the sum of elements along the last dimension of a ggml tensor
 *          using the CANN backend.
 *
 * @details This function performs a reduction sum operation along the last
 *          dimension of the input tensor `src`. The result of the sum is stored
 *          in the destination tensor `dst`.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the reduced values will be stored。
 *            dst->op is `GGML_OP_SUM_ROWS`.
 *
 * @attention `reduce_dims` defaults to 3, which means the last dimension.
 */
void ggml_cann_sum_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the sum of elements in a ggml tensor.
 *
 * @details This function performs a reduction sum operation along the last
 *          dimension of the input tensor `src`. The result of the sum is stored
 *          in the destination tensor `dst`.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the reduced values will be stored。
 *
 */

void ggml_cann_sum(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the cumulative sum of a ggml tensor along dim 0 using the
 *          CANN backend.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor. dst->op is `GGML_OP_CUMSUM`.
 */
void ggml_cann_cumsum(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes a triangular mask (tril/triu) of a square ggml tensor
 *          using the CANN backend.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor. dst->op is `GGML_OP_TRI`.
 */
void ggml_cann_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Solves a triangular linear system AX=B using the CANN backend.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor. dst->op is `GGML_OP_SOLVE_TRI`.
 */
void ggml_cann_solve_tri(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Creates a diagonal matrix from a vector using the CANN backend.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor. dst->op is `GGML_OP_DIAG`.
 */
void ggml_cann_diag(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Fills a tensor with a constant scalar value using the CANN backend.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor. dst->op is `GGML_OP_FILL`.
 */
void ggml_cann_fill(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Upsamples a ggml tensor using nearest neighbor interpolation using
 *          the CANN backend.
 *
 * @details This function performs upsampling of the input tensor `src` using
 *          nearest neighbor interpolation. The upsampling is applied to the
 *          height and width dimensions (last two dimensions) of the tensor. The
 *          result is stored in the destination tensor `dst`, which must have
 *          the appropriate dimensions for the upsampled output.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the upsampled values will be stored.
 *            dst->op is `GGML_OP_UPSCALE`.
 */
void ggml_cann_upsample_nearest2d(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Pads a ggml tensor to match the dimensions of the destination tensor
 *          using the CANN backend.
 *
 * @details This function pads the input tensor `src` so that it matches the
 *          dimensions of the destination tensor `dst`. The amount of padding
 *          is calculated based on the difference in sizes between `src` and
 *          `dst` along each dimension. The padded tensor is stored in `dst`.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor, which specifies the target dimensions for
 *            padding. dst->op is `GGML_OP_PAD`.
 */
void ggml_cann_pad(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Executes a 2D pooling operation on a ggml tensor using the CANN
 *          backend.
 *
 * @details This function dispatches the execution of a 2D pooling operation on
 *          the input tensor `dst`. The type of pooling (average or max) is
 *          determined by the `op` parameter, which is read from the operation
 *          parameters of `dst`. The function supports average pooling
 *          (`GGML_OP_POOL_AVG`) and max pooling (`GGML_OP_POOL_MAX`). If an
 *          invalid operation is encountered, the function asserts a failure.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor on which the pooling operation is to be
 *            performed. dst->op is `GGML_OP_POOL_2D`.
 */
void ggml_cann_pool2d(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Duplicates a ggml tensor using the CANN backend.
 *
 * @details This function duplicates the contents of the source tensor `src` to
 *          the destination tensor `dst`. The function supports various tensor
 *          types and configurations, including handling of extra data, type
 *          conversions, and special cases for contiguous and non-contiguous
 *          tensors.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the duplicated data will be stored.
 *            dst->op is `GGML_OP_DUP`
 *
 * @attention Only support Fp16/FP32. Not support when src and dst have
 *            different shape and dst is no-contiguous.
 * @note:     This func need to simplify.
 */
void ggml_cann_dup(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the Root Mean Square (RMS) normalization of a ggml tensor
 *          using the CANN backend.
 *
 * @details This function applies RMS normalization to the input tensor `src`
 *          and stores the result in the destination tensor `dst`. RMS
 *          normalization involves computing the root mean square of the input
 *          tensor along a specified dimension and then dividing each element of
 *          the tensor by this value, adjusted by a small epsilon value to
 *          prevent division by zero.
 *          The operation is defined as:
 *          \f[
 *               \text{RmsNorm}\left(x_i\right)=\frac{x_i}{\text{Rms}(\mathbf{x})} g_i,
 *               \quad \text { where } \text{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+e p s}
 *          \f]
 *          `eps` is in dst->op_params.
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 *            dst->op is `GGML_OP_RMS_NORM`.
 */
void ggml_cann_rms_norm(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies a diagonal mask to the tensor with a specified value.
 *
 * @details This function creates a mask tensor filled with ones, then applies
 *          an upper triangular and lower triangular operation to it based on
 *          the number of past elements specified. Afterward, it adds the masked
 *          tensor to the destination tensor in-place.
 *
 * @param ctx The backend CANN context used for operations.
 * @param dst The destination tensor where the result will be stored. dst->op is
 *            `GGML_OP_DIAG_MASK`
 * @param value The value to use for masking.
 */
void ggml_cann_diag_mask(ggml_backend_cann_context & ctx, ggml_tensor * dst, float value);

/**
 * @brief   Performs an image-to-column transformation on the input tensor.
 *
 * @details This function takes an input tensor and applies an image-to-column
 *          operation, converting spatial dimensions into column-like
 *          structures suitable for convolutional operations. It supports both
 *          half-precision (F16) and single-precision (F32) floating-point data
 *          types.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor that stores the result of the operation.
 *            dst->op is `GGML_OP_IM2COL`.
 */
void ggml_cann_im2col(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes time step embeddings using sine and cosine functions.
 *
 * @details This function calculates time step embeddings by applying sine and
 *          cosine transformations to a given input tensor, which is typically
 *          used in temporal models like diffusion models or transformers to
 *          encode time information effectively.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the result of the embedding operation
 *            will be stored. dst->op is `GGML_OP_TIMESTEP_EMBEDDING`.
 */
void ggml_cann_timestep_embedding(ggml_backend_cann_context & ctx, ggml_tensor * dst);

// @see ggml_cann_dup.
void ggml_cann_cpy(ggml_backend_cann_context & ctx, ggml_tensor * dst);

// @see ggml_cann_acc, but copies src1 into dst instead of adding.
void ggml_cann_set(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the softmax activation with optional masking.
 *
 * @details This function computes the softmax activation over the input tensor,
 *          optionally applying a mask and scaling factor. It supports both FP16
 *          and FP32 data types and can handle masking by broadcasting the mask
 *          across rows if necessary.
 *          The function performs the following steps:
 *          1. Multiplies the input tensor by a scale factor.
 *          2. Optionally casts the mask tensor to FP32 if it is in FP16 format.
 *          3. Broadcasts the mask tensor if its dimensions do not match the
 *             input tensor's dimensions.
 *          4. Adds the mask to the scaled input tensor.
 *          5. Applies the softmax activation function along the specified
 *             dimension.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the result will be stored. dst->op is
 *            `GGML_OP_SOFTMAX`.
 */
void ggml_cann_softmax(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Extracts specific rows from a tensor based on indices.
 *
 * @details This function retrieves rows from a source tensor src0 according to
 *          the indices provided in another tensor src1 and stores the result in
 *          a destination tensor (\p dst).
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the extracted rows will be stored.
 */
void ggml_cann_get_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Writes specific rows into a tensor at positions specified by indices.
 *
 * @details This function copies rows from a source tensor into a destination
 *          tensor (\p dst) at the positions indicated by the indices in another
 *          tensor.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the specified rows will be updated.
 */
void ggml_cann_set_rows(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Executes matrix multiplication for the given tensor.
 *
 * @details This function performs matrix multiplication on the source tensors
 *          associated with the destination tensor. It supports matrix
 *          multiplication F32, F16, and Q8_0.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor for storing the result of the matrix
 *            multiplication. dst->op is `GGML_OP_MUL_MAT`.
 */
void ggml_cann_mul_mat(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Applies Rotary Positional Embedding (RoPE) to the input tensor.
 *
 * @details This function implements the RoPE mechanism, which is a method to
 *          encode positional information into sequence data, particularly
 *          useful in transformer models. It supports both F32 and F16 data
 *          types.
 *
 * @param ctx The backend CANN context for executing operations.
 * @param dst The destination tensor where the RoPE-transformed data will be
 *            stored. dst->op is `GGML_OP_ROPE`.
 *
 * @note The function currently does not support cases where the n_dims is less
 *       than the input tensor's first dimension.
 * @note The function currently does not support cases where the freq_factors is
 *       not NULL.
 * @note The function currently does not support cases where the ext_factor is
 *       not equal 0.
 * @note The function currently does not support cases where the freq_scale is
 *       not equal 1.
 */
void ggml_cann_rope(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Pre-load the RoPE cache before ACL graph capture.
 *
 * This function must be called outside of graph capture to perform
 * host-to-device memory copies and device memory allocations that are
 * not allowed on a captured stream.  After pre-loading, the rope cache
 * metadata is updated so that the subsequent call to
 * aclnn_rope_cache_init (inside graph capture) skips these operations
 * and only records the on-device computations into the captured graph.
 *
 * @param ctx  CANN backend context.
 * @param dst  A ROPE destination tensor from the computation graph.
 */
void ggml_cann_rope_cache_preload(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the index of the maximum value along the specified dimension
 *          of a ggml tensor using the CANN backend.
 *
 * @details This function performs an argmax operation on the input tensor.
 *          It finds the index of the maximum value along the specified axis
 *          and stores these indices in the destination tensor `dst`. The
 *          operation is executed using the CANN backend for optimized performance.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the indices of the maximum values will
 *            be stored. dst->op is `GGML_OP_ARGMAX`.
 */
void ggml_cann_argmax(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Adds two tensors element-wise and stores the result in a destination
 * tensor.
 *
 * This function performs the operation:
 * \f[
 *    dst = acl\_src0 + alpha \times acl\_src1
 * \f]
 * where alpha is a scalar value and defaults to 1.0f.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src0 The first source tensor.
 * @param acl_src1 The second source tensor.
 * @param acl_dst The destination tensor where the result will be stored.
 */
void aclnn_add(ggml_backend_cann_context & ctx,
               aclTensor *                 acl_src0,
               aclTensor *                 acl_src1,
               aclTensor *                 acl_dst = nullptr);

/**
 * @brief Sub two tensors element-wise and stores the result in a destination
 * tensor.
 *
 * This function performs the operation:
 * \f[
 *    dst = acl\_src0 - alpha \times acl\_src1
 * \f]
 * where alpha is a scalar value and defaults to 1.0f.
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src0 The first source tensor.
 * @param acl_src1 The second source tensor.
 * @param acl_dst The destination tensor where the result will be stored.
 */
void aclnn_sub(ggml_backend_cann_context & ctx,
               aclTensor *                 acl_src0,
               aclTensor *                 acl_src1,
               aclTensor *                 acl_dst = nullptr);

/**
 * @brief Performs element-wise multiplication of two tensors and stores the
 * result in a destination tensor.
 *
 * This function performs element-wise multiplication of the tensors `acl_src`
 * and `acl_other` and stores the result in the destination tensor `acl_dst`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\text {acl_src }_i \times \text {acl_other }_i
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The first tensor for element-wise multiplication.
 * @param acl_other The second tensor for element-wise multiplication.
 * @param acl_dst The destination tensor where the result will be stored.
 */
void aclnn_mul(ggml_backend_cann_context & ctx,
               aclTensor *                 acl_src,
               aclTensor *                 acl_other,
               aclTensor *                 acl_dst = nullptr);

/**
 * @brief Matrix division, optionally in-place.
 *
 * This function division each element of the source tensor `acl_src` by the
 * tensor `acl_other` and stores the result in the destination tensor `acl_dst`.
 * If `inplace` is true, `acl_dst` will not be used and the operation is
 * performed in-place on `acl_src`. The operation is defined as: \f[
 *     \text{dst}_i = \frac{\text{acl_src}_i}{\text{acl_other}_i}
 * \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src Numerator tensor..
 * @param acl_other Denominator tensor.
 * @param acl_dst The destination tensor where the result will be stored if
 * `inplace` is false.
 * @param inplace Flag indicating whether to perform the operation in-place on
 * `acl_src`.
 */
void aclnn_div(ggml_backend_cann_context & ctx,
               aclTensor *                 acl_src,
               aclTensor *                 acl_other,
               aclTensor *                 acl_dst = nullptr);

/**
 * @brief Applies element-wise cosine function to the elements of a tensor.
 *
 * This function computes the cosine of each element in the source tensor
 * `acl_src` and stores the result in the destination tensor `acl_dst`. The
 * operation is defined as: \f[ \text {acl_dst }_i=\cos \left(\text {acl_src
 * }_i\right) \f]
 *
 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor on which the cosine function will be
 * applied.
 * @param acl_dst The destination tensor where the cosine results will be
 * stored.
 */
void aclnn_cos(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst);

/**
 * @brief Applies element-wise sine function to the elements of a tensor.
 *
 * This function computes the sine of each element in the source tensor
 `acl_src`
 * and stores the result in the destination tensor `acl_dst`.
 * The operation is defined as:
 * \f[
 *     \text {acl_dst }_i=\sin \left(\text {acl_src }_i\right)
 * \f]

 * @param ctx The context for the CANN backend operations.
 * @param acl_src The source tensor on which the sine function will be applied.
 * @param acl_dst The destination tensor where the sine results will be stored.
 */
void aclnn_sin(ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst);

/**
 * @brief Prepares broadcast-compatible ACL tensors for two input tensors and one
 * output tensor.
 *
 * This function checks whether broadcasting is needed between `src0` and `src1`.
 * If broadcasting is required, it calculates the proper shapes and creates
 * ACL tensors with broadcast parameters. Otherwise, it directly creates ACL tensors
 * based on the original tensor shapes.
 *
 * @param src0     The first input tensor (reference shape).
 * @param src1     The second input tensor (possibly broadcasted).
 * @param dst      The destination/output tensor.
 * @param acl_src0 Output pointer to the created ACL tensor corresponding to src0.
 * @param acl_src1 Output pointer to the created ACL tensor corresponding to src1.
 * @param acl_dst  Output pointer to the created ACL tensor corresponding to dst.
 */
void bcast_shape(ggml_tensor *    src0,
                 ggml_tensor *    src1,
                 ggml_tensor *    dst,
                 acl_tensor_ptr & acl_src0,
                 acl_tensor_ptr & acl_src1,
                 acl_tensor_ptr & acl_dst);

/**
 * @brief   Computes the 1D transposed convolution (deconvolution) of a ggml
 * tensor using the CANN backend.
 *
 * @details This function performs a 1D transposed convolution (also known as
 * deconvolution) operation on the input tensor. The computed result is stored
 * in the destination tensor `dst`. The operation is optimized using the CANN
 * backend for improved performance.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the transposed convolution result
 * will be stored. dst->op is `GGML_OP_CONV_TRANSPOSE_1D`.
 */
void ggml_cann_conv_transpose_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies the ELU (Exponential Linear Unit) activation to a ggml tensor
 * using the CANN backend.
 *
 * @details This function performs an element-wise ELU activation on the input
 *          tensor.
 *          The result is written to the destination tensor `dst` in-place.
 *          The ELU function is defined as:
 *
 *          \text{ELU}(x) =
 *          \begin{cases}
 *          x, & \text{if } x > 0 \\
 *          \alpha \left( \exp(x) - 1 \right), & \text{if } x \leq 0
 *          \end{cases}
 *
 *          where α (alpha) is a hyperparameter, typically set to 1.0.
 *          This operation is optimized using the CANN backend for high-performance
 *          inference or training.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the ELU-activated result will be stored.
 *            dst->op is expected to be `GGML_OP_ELU`.
 */
void ggml_cann_elu(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Computes the mean of a ggml tensor element-wise using the CANN backend.
 *
 * @details This function calculates the element-wise mean of the input tensor.
 *          The result is written to the destination tensor `dst`.
 *          The mean is computed by averaging the values across the entire tensor.
 *
 *          This operation is optimized using the CANN backend for high-performance inference or training.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the mean result will be stored.
 *            dst->op is expected to be `GGML_OP_MEAN`.
 */
void ggml_cann_mean(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies 1D reflect padding to a ggml tensor using the CANN backend.
 *
 * @details This function performs 1D reflect padding on the input tensor.
 *          The amount of padding on each side is specified by parameters stored in `dst->op_params`.
 *          The operation reflects the values at the borders of the tensor to generate the padded output.
 *
 *          This operation is optimized using the CANN backend for high-performance inference or training.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the padded result will be stored.
 *            dst->op is expected to be `GGML_OP_PAD_REFLECT_1D`.
 */
void ggml_cann_pad_reflect_1d(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Counts the number of equal elements in two ggml tensors using the CANN backend.
 *
 * @details This function performs an element-wise comparison between two input tensors,
 *          and counts the number of positions where the elements are equal. The result is
 *          stored in the destination tensor `dst` as a scalar.
 *
 *          The operation is optimized using the CANN backend, making it suitable for
 *          high-performance inference or training scenarios.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the result will be stored.
 *            dst->op is expected to be `GGML_OP_COUNT_EQUAL`.
 */
void ggml_cann_count_equal(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Applies the Step activation function to a ggml tensor using the CANN backend.
 *
 * @details This function applies a step function element-wise to the input tensor, where
 *          each element is transformed to 1.0 if it is greater than 0, and 0.0 otherwise.
 *          The result is stored in the destination tensor `dst`.
 *
 *          This operation is accelerated using the CANN backend to improve runtime performance.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the result will be stored.
 *            dst->op is expected to be `GGML_OP_STEP`.
 */
void ggml_cann_step(ggml_backend_cann_context & ctx, ggml_tensor * dst);
void ggml_cann_softplus(ggml_backend_cann_context & ctx, ggml_tensor * dst);
void ggml_cann_geglu_quick(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief   Performs the Flash Attention extended operator using the CANN backend.
 *
 * @details This function implements the memory-efficient Flash Attention algorithm
 *          for computing scaled dot-product attention with hardware acceleration.
 *          The result is stored in the destination tensor `dst`.
 *
 *          This operation is accelerated using the CANN backend to improve runtime performance.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the result will be stored.
 *            dst->op is expected to be `GGML_OP_FLASH_ATTN_EXT`.
 */
void ggml_cann_flash_attn_ext(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Forward Gated Linear Attention on the CANN backend.
 *
 * Expects dst->src[0..4] = {k, v, q, g, s} with shape conventions:
 *   k, v, q, g: [D] with outer dims T x H batched as ne[2]=T, ne[1]=H
 *   s: initial state [B, H, D, D], where B is batch and D=C/H
 * dst holds both outputs (o) and updated state; a scale factor is read from op params.
 *
 * The kernel updates per time step l: S_new = g ⊗ S_old + k ⊗ v, then computes o = (S_new^T q) * scale.
 *
 * @param ctx Backend context providing stream/allocator utilities.
 * @param dst Output tensor; src deps are k, v, q, g, s as above.
 */
void ggml_cann_gated_linear_attn(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Launches an asynchronous task using the memory allocator.
 *
 * This macro submit an asynchronous task on the specified stream.
 * The task uses memory allocated by the allocator. It is guaranteed
 * that the memory will not be accessed by other tasks until this task
 * completes, due to the sequential execution order within the same stream.
 *
 * @param OP_NAME aclnn operator name.
 * @param args Additional arguments required by the task.
 *
 * @note
 * Memory from the allocator will be "freed" immediately and can be
 * reallocated to other pointers. However, it won't be accessed by any
 * other task before this asynchronous task ends, because all tasks in the
 * same stream are executed in queue order.
 */

#    define GGML_CANN_CALL_ACLNN_OP(CTX, OP_NAME, ...)                                           \
        do {                                                                                     \
            uint64_t        workspaceSize = 0;                                                   \
            aclOpExecutor * executor;                                                            \
            void *          workspaceAddr = nullptr;                                             \
            ACL_CHECK(aclnn##OP_NAME##GetWorkspaceSize(__VA_ARGS__, &workspaceSize, &executor)); \
            /* workspace should alloced in main thread to keep malloc order when using vmm. */   \
            if (workspaceSize > 0) {                                                             \
                ggml_cann_pool_alloc workspace_allocator(CTX.pool(), workspaceSize);             \
                workspaceAddr = workspace_allocator.get();                                       \
            }                                                                                    \
            ACL_CHECK(aclnn##OP_NAME(workspaceAddr, workspaceSize, executor, CTX.stream()));     \
        } while (0)

/**
 * @brief   Performs sparse expert-based matrix multiplication using the CANN backend.
 *
 * @details This function implements a MoE-style batched matrix multiplication, where each input token
 *          is routed to one or more experts, and each expert corresponds to a specific [D, M] weight matrix
 *          in the source tensor `src0`. The routing indices are provided via the `ids` tensor.
 *
 *          For each token (from `src1`), the function selects the corresponding expert(s) as specified by `ids`,
 *          performs the matrix multiplication with the selected expert's weight submatrix (from `src0`),
 *          and stores the results in `dst`. This operation is optimized and executed on the CANN backend.
 *
 *          Dimensions:
 *              - src0: [D, M, A, 1], where A is the number of experts
 *              - src1: [D, B, N, 1], where N is batch size and B is the slot count per sample
 *              - ids : [K, N],       where K is the number of experts each token is routed to
 *              - dst : [M, K, N, 1], output tensor storing the result of expert × token multiplication
 *
 *          The function handles two main modes:
 *              - If `ne12 == 1`, a simpler per-token loop is used.
 *              - TODO: If `ne12 > 1`, grouped multiplication and memory copying is used for efficiency.
 *
 * @param ctx The CANN context used for operations.
 * @param dst The destination tensor where the expert-weighted token outputs are stored.
 *            Expected to be of shape [M, K, N, 1].
 */
void ggml_cann_mul_mat_id(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Performs fused ADD + RMS_NORM operation using the CANN backend.
 *
 * This function fuses the ADD and RMS_NORM operations into a single kernel call
 * for better performance. It first adds two input tensors (x1 + x2), then applies
 * RMS normalization to the result.
 *
 * @param ctx The context for the CANN backend operations.
 * @param dst The ADD operation node, contains the two input tensors to be added.
 * @param rms_norm_tensor The RMS_NORM operation node, contains the gamma weights
 *                        and epsilon parameter.
 */
void ggml_cann_op_add_rms_norm_fused(ggml_backend_cann_context & ctx,
                                     ggml_tensor *               add_node,
                                     ggml_tensor *               rms_norm_node);

/**
 * @brief   Check whether a tensor is a weight tensor for matrix multiplication.
 *
 * @details Checks whether the given tensor serves as weight parameters in matrix multiplication operations,
 *          typically within neural network layers. The function maintains a static set of canonical weight
 *          naming suffixes from Transformer-based architectures. Uses substring matching to identify weight
 *          tensors even with hierarchical naming patterns.
 *
 * @param tensor Pointer to the target ggml_tensor object (const-qualified).
 */
static bool is_matmul_weight(const ggml_tensor * tensor) {
    std::string                                  name = ggml_get_name(tensor);
    static const std::unordered_set<std::string> weight_suffixes{ "output.weight",      "attn_q.weight",
                                                                  "attn_k.weight",      "attn_v.weight",
                                                                  "attn_output.weight", "ffn_gate.weight",
                                                                  "ffn_up.weight",      "ffn_down.weight" };

    for (const auto & suffix : weight_suffixes) {
        if (name.find(suffix) != std::string::npos) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Applies a element-wise operation to two input tensors using the CANN
 * backend.
 *
 * This templated function takes a binary operator and applies it to two source
 * tensors
 * associated with the destination tensor. The function handles broadcasting as
 * needed.
 *
 * @tparam binary_op A callable object (e.g., lambda or function pointer) representing
 *         the binary operation to be performed. It must take three arguments:
 *         (ggml_backend_cann_context&, aclTensor*, aclTensor*, aclTensor*).
 *
 * @param ctx The CANN backend context used to manage execution and resources.
 * @param dst The destination tensor.
 */
template <auto binary_op> void ggml_cann_binary_op(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    acl_tensor_ptr acl_src0, acl_src1, acl_dst;

    // Need bcast
    bcast_shape(src0, src1, dst, acl_src0, acl_src1, acl_dst);
    binary_op(ctx, acl_src0.get(), acl_src1.get(), acl_dst.get());
}

/**
 * @brief Applies a unary operation to an input tensor using the CANN backend.
 *
 * This templated function applies a unary operator to the source tensor of `dst`
 * and stores the result in the destination tensor.
 *
 * @tparam unary_op A callable with the signature:
 *         void(ggml_backend_cann_context&, aclTensor *, aclTensor *)
 *         where the first aclTensor is the source and the second is the destination.
 * @param ctx The CANN backend context for managing resources and execution.
 * @param dst The destination tensor. Its src[0] is treated as the input tensor.
 */
template <void unary_op(ggml_backend_cann_context &, aclTensor *, aclTensor *)>
void ggml_cann_op_unary(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src = dst->src[0];

    acl_tensor_ptr acl_src = ggml_cann_create_tensor(src);
    acl_tensor_ptr acl_dst = ggml_cann_create_tensor(dst);

    unary_op(ctx, acl_src.get(), acl_dst.get());
}

/**
 * @brief Applies a unary operation to a ggml tensor using the CANN backend.
 *
 * @details This function applies a unary operation to the input tensor using
 * a user-provided lambda or callable `unary_op`. The lambda receives the
 * CANN backend context and two ACL tensors: the source and the destination.
 *
 * Internally, this function handles the conversion from GGML tensors to ACL tensors,
 * calls the provided unary op, and manages resource cleanup. The input is assumed
 * to be `dst->src[0]`, and the result is written to `dst`.
 *
 * This utility simplifies writing unary op wrappers by abstracting tensor preparation.
 *
 * @param unary_op A callable that performs the unary operation using CANN ACL APIs.
 * @param ctx The CANN context for operation execution.
 * @param dst The destination ggml_tensor where the result will be stored.
 *            The input tensor is assumed to be `dst->src[0]`.
 *
 * @see GGML_CANN_CALL_OP_UNARY
 */
void ggml_cann_op_unary(std::function<void(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                        ggml_backend_cann_context &                                                ctx,
                        ggml_tensor *                                                              dst);

void ggml_cann_ssm_conv(ggml_backend_cann_context & ctx, ggml_tensor * dst);

/**
 * @brief Applies a gated (GLU-style) unary operation using the CANN backend.
 *
 * @details This function performs a gated activation such as GEGLU or ReGLU.
 * It supports two input modes:
 *
 * 1. **Dual input mode**: `dst->src[0]` and `dst->src[1]` are both valid tensors.
 *    These are used directly as the value and gate tensors.
 *
 * 2. **Packed input mode**: Only `dst->src[0]` is valid, and it is assumed to
 *    contain a concatenation of value and gate along the first dimension. This tensor
 *    will be split into two equal halves to form the value and gate inputs.
 *
 * The function applies a user-provided unary operation (e.g., GELU) to the value tensor,
 * then multiplies the result in-place with the gate tensor:
 *
 * @code
 * dst = unary_op(value) * gate;
 * @endcode
 *
 * The `swapped` parameter (from `dst->op_params[1]`) allows flipping the
 * order of value/gate in the packed input case.
 *
 * @param unary_op A callable that performs the unary operation using CANN ACL APIs.
 *                 It receives (ctx, acl_value_tensor, acl_output_tensor).
 * @param ctx      The CANN context used for execution.
 * @param dst      The destination ggml_tensor. Source tensors are in `dst->src[0]` and optionally `src[1]`.
 *
 * @see GGML_CANN_CALL_OP_UNARY_GATED
 */
void ggml_cann_op_unary_gated(std::function<void(ggml_backend_cann_context &, aclTensor *, aclTensor *)> unary_op,
                              ggml_backend_cann_context &                                                ctx,
                              ggml_tensor *                                                              dst);

/**
 * @brief Helper macro to call a unary ACL operator via ggml_cann_op_unary.
 *
 * This macro wraps the specified ACLNN unary operator name into a lambda expression,
 * and passes it to `ggml_cann_op_unary`, which handles the common logic for executing
 * unary ops in the CANN backend.
 *
 * Internally, this macro expands to a lambda like:
 * @code
 * [](ggml_backend_cann_context& ctx, aclTensor* acl_src, aclTensor* acl_dst) {
 *     GGML_CANN_CALL_ACLNN_OP(ctx, OP_NAME, acl_src, acl_dst);
 * };
 * @endcode
 *
 * This lambda is then passed to `ggml_cann_op_unary`, which applies the operation.
 *
 * @param OP_NAME The name of the ACL unary operator to invoke via GGML_CANN_CALL_ACLNN_OP.
 *
 * @see ggml_cann_op_unary
 * @see GGML_CANN_CALL_ACLNN_OP
 */
#    define GGML_CANN_CALL_OP_UNARY(OP_NAME)                                                              \
        do {                                                                                              \
            auto lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) { \
                GGML_CANN_CALL_ACLNN_OP(ctx, OP_NAME, acl_src, acl_dst);                                  \
            };                                                                                            \
            ggml_cann_op_unary(lambda, ctx, dst);                                                         \
        } while (0)

/**
 * @brief Helper macro to call a gated unary ACL operator via ggml_cann_op_unary_gated.
 *
 * This macro wraps the specified ACLNN unary operator name into a lambda expression,
 * and passes it to `ggml_cann_op_unary_gated`, which handles the common logic for
 * executing gated unary ops in the CANN backend.
 *
 * Internally, this macro expands to a lambda like:
 * @code
 * [](ggml_backend_cann_context& ctx, aclTensor* acl_src, aclTensor* acl_dst) {
 *     GGML_CANN_CALL_ACLNN_OP(ctx, OP_NAME, acl_src, acl_dst);
 * };
 * @endcode
 *
 * This lambda is then passed to `ggml_cann_op_unary_gated`, which applies the operation.
 *
 * @param OP_NAME The name of the ACL unary operator to invoke via GGML_CANN_CALL_ACLNN_OP.
 *
 * @see ggml_cann_op_unary_gated
 * @see GGML_CANN_CALL_ACLNN_OP
 */
#    define GGML_CANN_CALL_OP_UNARY_GATED(OP_NAME)                                                        \
        do {                                                                                              \
            auto lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) { \
                GGML_CANN_CALL_ACLNN_OP(ctx, OP_NAME, acl_src, acl_dst);                                  \
            };                                                                                            \
            ggml_cann_op_unary_gated(lambda, ctx, dst);                                                   \
        } while (0)


/**
 * @brief Performs outer product operation on two ggml tensors using the CANN backend.
 *
 * @details This function computes the outer product of two input tensors (src0 and src1)
 * and stores the result in the destination tensor. The outer product operation is defined as:
 * dst[i,j,k,l] = sum_m (src0[i,m,k,l] * src1[j,m,k,l])
 *
 * The function supports multiple data types including F32, F16. For floating-point
 * types, it uses batch matrix multiplication for efficient computation.
 *
 * The implementation handles 4D tensor broadcasting and batch processing automatically.
 *
 * @param ctx The CANN backend context for operation execution and memory management.
 * @param dst The destination ggml_tensor where the outer product result will be stored.
 *            The input tensors are assumed to be `dst->src[0]` and `dst->src[1]`.
 *
 * @see GGML_CANN_CALL_ACLNN_OP for CANN operator invocation
 */
void ggml_cann_out_prod(ggml_backend_cann_context & ctx, ggml_tensor * dst);

#include "ggml-cann/common.h.inc"
#include "ggml-impl-defs.inc"
#include "ggml.h.inc"

#include <acl/acl.h>
#include <aclnnop/aclnn_trans_matmul_weight.h>
#include <stdarg.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define GGML_COMMON_DECL_C

#include "ggml-common-defs.inc"

#define GGML_CANN_NAME "CANN"

/**
 * @brief Handles CANN errors by printing an error message and aborting.
 *
 * @param stmt The statement that caused the error.
 * @param func The function in which the error occurred.
 * @param file The file in which the error occurred.
 * @param line The line number where the error occurred.
 * @param msg The error message.
 */
[[noreturn]] void ggml_cann_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int32_t id = -1;
    aclrtGetDevice(&id);

    GGML_LOG_ERROR("CANN error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CANN error");
}

// Thread-local variable to record the current device of this thread.
thread_local int g_current_cann_device = -1;

/**
 * @brief Set the CANN device to be used.
 *
 * @param device The target device ID to set.
 */
void ggml_cann_set_device(const int32_t device) {
    // int current_device = -1;
    // Note: In some CANN versions, if no device has been set yet,
    //       aclrtGetDevice(&current_device) may return 0 by default.
    // aclrtGetDevice(&current_device);

    // If the current device is already the target one, no need to switch.
    if (device == g_current_cann_device) {
        return;
    }

    // Switch to the new device.
    ACL_CHECK(aclrtSetDevice(device));

    // Update the global device record.
    g_current_cann_device = device;
}

/**
 * @brief Get the value of the specified environment variable (name) as lowercase.
 *        if not empty, return a std::string object
 */
std::optional<std::string> get_env_as_lowercase(const std::string & name) {
    const char * val = std::getenv(name.c_str());
    if (!val) {
        return std::nullopt;
    }
    std::string res = std::string(val);
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

/**
 * @brief Verify whether the environment variable is a valid value.
 */
bool parse_bool(const std::string & value) {
    static const std::unordered_set<std::string> valid_values = { "on", "1", "yes", "y", "enable", "true" };
    return valid_values.find(value) != valid_values.end();
}

/**
 * @brief Parse a string as an integer, returning 0 if invalid.
 *
 * This function attempts to convert the input string `value` to an `int`.
 * If the string is not a valid integer or is out of the `int` range,
 * it returns 0.
 *
 * @param value The string to parse.
 * @return The parsed integer, or 0 if conversion fails.
 */
int parse_integer(const std::string & value) {
    try {
        return std::stoi(value);
    } catch (...) {
        return 0;
    }
}

/**
 * @brief Initialize the CANN device information.
 *
 * This function initializes the CANN device information by obtaining the
 * device count and setting the memory allocation granularity for each device.
 *
 * @return A structure containing the device information.
 */
static ggml_cann_device_info ggml_cann_init() {
    ggml_cann_device_info info = {};

    aclError err = aclrtGetDeviceCount((uint32_t *) &info.device_count);

    if (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: failed to initialize CANN: %s\n", __func__, aclGetRecentErrMsg());
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CANN_MAX_DEVICES);

    for (int id = 0; id < info.device_count; ++id) {
        aclrtPhysicalMemProp prop = {};
        prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
        prop.allocationType       = ACL_MEM_ALLOCATION_TYPE_PINNED;
        prop.memAttr              = ACL_HBM_MEM_HUGE;
        prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = id;
        prop.reserve              = 0;
        err                       = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                                                                     &info.devices[id].vmm_granularity);
        info.devices[id].vmm      = err == ACL_SUCCESS;

        size_t free, total;
        ggml_backend_cann_get_device_memory(id, &free, &total);
        info.devices[id].total_vram = free;
    }

    // TODO: add more device info later.
    return info;
}

/**
 * @brief Retrieve the CANN device information.
 *
 * This function returns a reference to a structure containing the CANN device
 * information. The device information is initialized once and reused on
 * subsequent calls.
 *
 * @return A reference to the structure containing the device information.
 */
const ggml_cann_device_info & ggml_cann_info() {
    static ggml_cann_device_info info = ggml_cann_init();
    return info;
}

//#define DEBUG_CANN_MALLOC
/**
 * @brief A pool of CANN buffers(priority segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
struct ggml_cann_pool_buf_prio : public ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    static const size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    static const size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    static const size_t alignment = 128;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    bool disable_clean = false;

    /**
     * @brief Structure representing a CANN buffer.
     */
    struct ggml_cann_buffer {
        void *                                ptr  = nullptr;  ///< Pointer to the buffer.
        size_t                                size = 0;        ///< Size of the buffer.
        std::chrono::steady_clock::time_point last_used;       ///< Last used time.

        bool operator>(const ggml_cann_buffer & other) const { return size > other.size; }
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    std::unordered_map<void *, size_t>                                                   buffer_pool;
    std::priority_queue<ggml_cann_buffer, std::vector<ggml_cann_buffer>, std::greater<>> free_buffers;

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_buf_prio(int device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf_prio() {
        ggml_cann_set_device(device);
        for (auto & [b_ptr, b_size] : buffer_pool) {
            aclrtFree(b_ptr);
            pool_size -= b_size;
        }
        buffer_pool.clear();
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void * alloc(size_t size, size_t * actual_size) override {
        size = GGML_PAD(size, alignment);
        if (size == 0) {
            size = alignment;
        }

        void * ptr = nullptr;
        auto   now = std::chrono::steady_clock::now();

        std::vector<ggml_cann_buffer> free_buffers_rest;
        free_buffers_rest.reserve(free_buffers.size());
        while (!free_buffers.empty()) {
            auto b = free_buffers.top();
            free_buffers.pop();

            if (b.size >= size) {
                // reuse the buffer if the size is enough
                const size_t margin = b.size - size;
                if (margin <= max_reuse_margin) {
                    *actual_size = b.size;
                    ptr          = b.ptr;
#ifdef DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#endif
                    break;
                }
            }

            bool should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.last_used).count() > 100;
            if (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_size -= b.size;
                buffer_pool.erase(b.ptr);
#ifdef DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#endif
                continue;
            }
            free_buffers_rest.push_back(b);
        }
        for (ggml_cann_buffer & b : free_buffers_rest) {
            free_buffers.push(std::move(b));
        }

#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d] free pool_size = %5u MB\n\n", device,
                      (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576));
#endif
        if (ptr != nullptr) {
            return ptr;
        }

        // allocate a new buffer if no buffer can be reused
        ggml_cann_set_device(device);
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        *actual_size = size;
        pool_size += size;
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: allocate %p, "
            "pool_size = %5u MB, "
            "size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
            (uint32_t) (GGML_PAD(size, 1048576) / 1048576));
#endif
        buffer_pool.emplace(ptr, size);
        return ptr;
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void * ptr, size_t size) override {
        GGML_UNUSED(size);
        auto it = buffer_pool.find(ptr);
        if (it == buffer_pool.end()) {
            GGML_ABORT("cann pool[%d]: buffer %p not found in pool\n", device, ptr);
        }

        auto now = std::chrono::steady_clock::now();
        free_buffers.emplace(ggml_cann_buffer{ ptr, it->second, now });
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO(
            "cann pool[%d]: return   %p, "
            "pool_size = %5u MB\n",
            device, ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576));
#endif
    }
};

/**
 * @brief A pool of CANN buffers(segment buffer).
 *
 * This class manages a pool of CANN buffers for a specific device.
 */
struct ggml_cann_pool_buf : public ggml_cann_pool {
    /**
     * @brief The maximum reuse margin for a buffer.
     */
    static const size_t max_reuse_margin = 1ull << 22;  // 4MB

    /**
     * @brief The minimum free margin for a buffer.
     */
    static const size_t min_free_margin = 1ull << 20;  // 1MB

    /**
     * @brief The alignment for buffer allocation.
     */
    static const size_t alignment = 128;

    /**
     * @brief The maximum number of buffers in the pool.
     */
    static const int MAX_BUFFERS = 256;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Whether to disable clean during buffer allocation.
     */
    bool disable_clean = false;

    /**
     * @brief Structure representing a CANN buffer.
     */
    struct ggml_cann_buffer {
        void *                                ptr  = nullptr;  ///< Pointer to the buffer memory.
        size_t                                size = 0;        ///< Size of the buffer.
        bool                                  used = false;    ///< Whether the buffer is currently in use.
        std::chrono::steady_clock::time_point last_used;       ///< Last used time.
    };

    /**
     * @brief Array of CANN buffers in the pool.
     */
    ggml_cann_buffer buffer_pool[MAX_BUFFERS] = {};

    /**
     * @brief Total size of all buffers in the pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Constructor to initialize the buffer pool for a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_buf(int device) : device(device) {
        disable_clean = parse_bool(get_env_as_lowercase("GGML_CANN_DISABLE_BUF_POOL_CLEAN").value_or(""));
    }

    /**
     * @brief Destructor to free all buffers in the pool.
     */
    ~ggml_cann_pool_buf() {
        ggml_cann_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                aclrtFree(b.ptr);
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    /**
     * @brief Allocate a buffer of the given size.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void * alloc(size_t size, size_t * actual_size) override {
        size = GGML_PAD(size, alignment);
        if (size == 0) {
            size = alignment;
        }

        void * ptr = nullptr;
        auto   now = std::chrono::steady_clock::now();

        int i = 0;
        for (; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer & b = buffer_pool[i];
            if (b.ptr == nullptr) {
                break;
            }
            if (b.used) {
                continue;
            }
            if (b.size >= size) {
                // reuse the buffer if the size is enough
                const size_t margin = b.size - size;
                if (margin <= max_reuse_margin) {
                    *actual_size = b.size;
                    b.used       = true;
                    ptr          = b.ptr;
#ifdef DEBUG_CANN_MALLOC
                    GGML_LOG_INFO(
                        "cann pool[%d]: reused   %p, "
                        "pool_size = %5u MB, "
                        "size = %5u MB, "
                        "margin = %5u MB\n",
                        device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(size, 1048576) / 1048576),
                        (uint32_t) (GGML_PAD(margin, 1048576) / 1048576));
#endif
                    break;
                }
            }

            bool should_clean = !disable_clean && b.size > min_free_margin &&
                                std::chrono::duration_cast<std::chrono::milliseconds>(now - b.last_used).count() > 100;
            if (should_clean) {
                // free the buffer if the size is needed to be freed
                ACL_CHECK(aclrtFree(b.ptr));
                pool_size -= b.size;
#ifdef DEBUG_CANN_MALLOC
                GGML_LOG_INFO(
                    "cann pool[%d]: clean    %p, "
                    "pool_size = %5u MB, "
                    "size = %5u MB\n",
                    device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                    (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#endif
                b.ptr = nullptr;
            }
        }
        if (ptr != nullptr) {
            return ptr;
        }

        if (i < MAX_BUFFERS) {
            // allocate a new buffer if no buffer can be reused
            ggml_cann_buffer & b = buffer_pool[i];
            ggml_cann_set_device(device);
            ACL_CHECK(aclrtMalloc(&b.ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
            pool_size += size;
            *actual_size = size;
            b.size       = size;
            b.used       = true;
            if (i >= MAX_BUFFERS - 8) {
                GGML_LOG_WARN("cann pool[%d]: slots almost full\n", device);
            }
#ifdef DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: allocate %p, "
                "pool_size = %5u MB, "
                "size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576),
                (uint32_t) (GGML_PAD(b.size, 1048576) / 1048576));
#endif
            return b.ptr;
        }

        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }

    /**
     * @brief Free a buffer and return it to the pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void * ptr, size_t size) override {
        GGML_UNUSED(size);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cann_buffer & b = buffer_pool[i];
            if (b.ptr != ptr) {
                continue;
            }
            b.used      = false;
            b.last_used = std::chrono::steady_clock::now();
#ifdef DEBUG_CANN_MALLOC
            GGML_LOG_INFO(
                "cann pool[%d]: return   %p, "
                "pool_size = %5u MB\n",
                device, b.ptr, (uint32_t) (GGML_PAD(pool_size, 1048576) / 1048576));
#endif
            return;
        }
        GGML_ABORT("cann pool[%d]: slots full\n", device);
    }
};

/**
 * @brief A pool of CANN buffers with virtual memory.
 *
 * This class manages a pool of CANN buffers with virtual memory for a specific
 * device.
 */
struct ggml_cann_pool_vmm : public ggml_cann_pool {
    /**
     * @brief The maximum size of the virtual memory pool (32 GB).
     */
    size_t max_size;

    /**
     * @brief The device ID associated with this buffer pool.
     */
    int device;

    /**
     * @brief Pointer to the start of the virtual memory pool.
     */
    void * pool_addr = 0;

    /**
     * @brief Amount of virtual memory used in the pool.
     */
    size_t pool_used = 0;

    /**
     * @brief Total size of the virtual memory pool.
     */
    size_t pool_size = 0;

    /**
     * @brief Allocation granularity for the virtual memory pool.
     */
    size_t granularity;

    /**
     * @brief Handles for the physical memory allocated.
     */
    std::vector<aclrtDrvMemHandle> handles;

    /**
     * @brief Offsets for the mapped memory regions.
     */
    std::vector<void *> map_offsets;

    /**
     * @brief Constructor to initialize the buffer pool with virtual memory for
     * a specific device.
     *
     * @param device The device ID to associate with this buffer pool.
     */
    explicit ggml_cann_pool_vmm(int device) : device(device) {
        auto dev    = ggml_cann_info().devices[device];
        granularity = dev.vmm_granularity;
        max_size    = dev.total_vram;
    }

    /**
     * @brief Destructor to free all buffers in the virtual memory pool.
     */
    ~ggml_cann_pool_vmm() {
        if (pool_addr != 0) {
            for (auto & offset : map_offsets) {
                ACL_CHECK(aclrtUnmapMem(offset));
            }
            for (auto & handle : handles) {
                ACL_CHECK(aclrtFreePhysical(handle));
            }
            ACL_CHECK(aclrtReleaseMemAddress(pool_addr));
        }
    }

    /**
     * @brief Allocate a buffer of the given size in the virtual memory pool.
     *
     * @param size The size of the buffer to allocate.
     * @param actual_size A pointer to a variable to receive the actual size of
     * the allocated buffer.
     * @return A pointer to the allocated buffer.
     */
    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all
        // allocations are aligned for all data types
        const size_t alignment = 128;
        size                   = GGML_PAD(size, alignment);
        if (size == 0) {
            size = alignment;
        }

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size        = GGML_PAD(reserve_size, granularity);

            GGML_ASSERT(pool_size + reserve_size <= max_size);

            // allocate more physical memory
            aclrtPhysicalMemProp prop = {};
            prop.handleType           = ACL_MEM_HANDLE_TYPE_NONE;
            prop.allocationType       = ACL_MEM_ALLOCATION_TYPE_PINNED;
            prop.memAttr              = ACL_HBM_MEM_HUGE;
            prop.location.type        = ACL_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id          = device;
            prop.reserve              = 0;
            aclrtDrvMemHandle handle;
            ACL_CHECK(aclrtMallocPhysical(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                ACL_CHECK(aclrtReserveMemAddress(&pool_addr, max_size, 0, NULL, 1));
            }

            // map at the end of the pool
            ACL_CHECK(aclrtMapMem((char *) pool_addr + pool_size, reserve_size, 0, handle, 0));

            handles.push_back(handle);
            map_offsets.push_back((char *) pool_addr + pool_size);

            // add to the pool
            pool_size += reserve_size;

#ifdef DEBUG_CANN_MALLOC
            GGML_LOG_INFO("cann pool[%d]: size increased to %llu MB (reserved %llu MB)\n", device,
                          (unsigned long long) (pool_size / 1024 / 1024),
                          (unsigned long long) (reserve_size / 1024 / 1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr   = (void *) ((char *) pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size,
                      (unsigned long long) ptr);
#endif
        return ptr;
    }

    /**
     * @brief Free a buffer and return it to the virtual memory pool.
     *
     * @param ptr Pointer to the buffer to free.
     * @param size Size of the buffer to free.
     */
    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CANN_MALLOC
        GGML_LOG_INFO("cann pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size,
                      (unsigned long long) ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *) pool_addr + pool_used));
    }
};

/**
 * @brief Create a new CANN pool for a specific device.
 *
 * Factory method to create a new CANN pool object based on the device type.
 *
 * @param device The device ID for which to create the pool.
 * @return A unique pointer to the created CANN pool.
 */
std::unique_ptr<ggml_cann_pool> ggml_backend_cann_context::new_pool_for_device(int device) {
    std::string mem_pool_type = get_env_as_lowercase("GGML_CANN_MEM_POOL").value_or("");

    if (mem_pool_type == "prio") {
        GGML_LOG_INFO("%s: device %d use buffer pool with priority queue\n", __func__, device);
        return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_buf_prio(device));
    }

    if (ggml_cann_info().devices[device].vmm && mem_pool_type != "leg") {
        GGML_LOG_INFO("%s: device %d use vmm pool\n", __func__, device);
        return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_vmm(device));
    }

    GGML_LOG_INFO("%s: device %d use buffer pool\n", __func__, device);
    return std::unique_ptr<ggml_cann_pool>(new ggml_cann_pool_buf(device));
}

// cann buffer

/**
 * @brief Tracks multi-threaded write progress for a single tensor.
 *
 * When multiple threads call set_tensor on different chunks of the same tensor,
 * this tracker accumulates progress and defers post-processing (quantized format
 * transform or ND-to-NZ conversion) until all data has been written.
 */
struct TensorSetTracker {
    std::mutex mtx;                   ///< Protects concurrent access to this tracker
    size_t bytes_written = 0;         ///< Accumulated bytes written so far
    size_t total_bytes = 0;           ///< Target size (full tensor)
    std::vector<uint8_t> host_buffer; ///< Host staging buffer for quantized tensors
};

/**
 * @brief Context for managing a CANN buffer associated with a specific device.
 *
 * This structure holds information about a CANN buffer, including the device
 * ID, device pointer, and a name derived from GGML_CANN_NAME and the device ID.
 */
struct ggml_backend_cann_buffer_context {
    int32_t device;             ///< The device ID associated with this buffer context.
    void *  dev_ptr = nullptr;  ///< Pointer to the device memory allocated for the buffer.

    std::mutex tracker_mutex;   ///< Protects the trackers map
    std::unordered_map<void *, std::unique_ptr<TensorSetTracker>> trackers;

    /**
     * @brief Constructor to initialize the CANN buffer context.
     *
     * @param device The device ID associated with this buffer context.
     * @param dev_ptr Pointer to the device memory allocated for the buffer.
     */
    ggml_backend_cann_buffer_context(int32_t device, void * dev_ptr) : device(device), dev_ptr(dev_ptr) {}

    /**
     * @brief Destructor to free the device memory allocated for the buffer.
     */
    ~ggml_backend_cann_buffer_context() { ACL_CHECK(aclrtFree(dev_ptr)); }

    /**
     * @brief Get or create a tracker for the given tensor.
     */
    TensorSetTracker * get_or_create_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        auto key = tensor->data;
        auto it = trackers.find(key);
        if (it == trackers.end()) {
            auto tracker = std::make_unique<TensorSetTracker>();
            tracker->total_bytes = ggml_nbytes(tensor);
            auto * ptr = tracker.get();
            trackers[key] = std::move(tracker);
            return ptr;
        }
        return it->second.get();
    }

    /**
     * @brief Remove the tracker for the given tensor.
     */
    void remove_tracker(ggml_tensor * tensor) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        trackers.erase(tensor->data);
    }
};

// cann buffer type
/**
 * @brief Structure representing context information for a specific backend
 * buffer type.
 */
struct ggml_backend_cann_buffer_type_context {
    int32_t     device; /**< Device identifier associated with the buffer context. */
    std::string name;   /**< Name associated with the buffer context. */
};

/**
 * @brief Retrieves the name associated with a CANN buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * CANN buffer type context.
 *
 * @param buft Pointer to the buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char * ggml_backend_cann_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cann_buffer_type_context * buft_ctx = (ggml_backend_cann_buffer_type_context *) buft->context;

    return buft_ctx->name.c_str();
}

/**
 * @brief Checks if the backend buffer type is associated with the CANN backend.
 *
 * This function checks whether the provided backend buffer type is associated
 * with the CANN backend based on the comparison of its name retrieval function
 * pointer.
 *
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the buffer type is associated with the CANN
 * backend, otherwise false.
 */
static bool ggml_backend_buft_is_cann(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cann_buffer_type_name;
}

/**
 * @brief Free resources associated with a CANN buffer.
 *
 * This function frees the resources associated with a CANN buffer, including
 * its context.
 *
 * @param buffer The CANN buffer to free.
 */
static void ggml_backend_cann_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;
    delete ctx;
}

/**
 * @brief Retrieve the base pointer of a CANN buffer.
 *
 * This function returns the base pointer of a CANN buffer, which points to the
 * device memory allocated for the buffer.
 *
 * @param buffer The CANN buffer whose base pointer is to be retrieved.
 * @return A pointer to the base of the device memory allocated for the buffer.
 */
static void * ggml_backend_cann_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;
    return ctx->dev_ptr;
}

/**
 * @brief Transform quantized Q4.0 tensor data into a format suitable for CANN
 * processing.
 *
 * This function transforms quantized Q4.0 tensor data into a format suitable
 * for CANN processing. It extracts quantization values and scales from the
 * source data and prepares them in a format expected by CANN operations.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data in Q4.0 format.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
static void ggml_backend_cann_transform_q4_0(ggml_tensor * tensor, const void * src, void * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_bytes = n_elems * sizeof(uint8_t) / 2;

    uint8_t *  quant_offset = (uint8_t *) dst;
    uint16_t * scale_offset = (uint16_t *) ((char *) dst + quant_bytes);

    for (int i = 0; i < groups; i++) {
        const block_q4_0 * group = (const block_q4_0 *) ((const char *) src + i * sizeof(block_q4_0));
        *scale_offset            = group->d;
        scale_offset++;

        // 0-15
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] & 0x0F);
            (*quant_offset) |= ((group->qs[j + 1] << 4));
            quant_offset++;
        }

        // 16-31
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            (*quant_offset) = (group->qs[j] >> 4);
            (*quant_offset) |= (group->qs[j + 1] & 0xF0);
            quant_offset++;
        }
    }

    // put (uint4b_t -8) into int4b_t
    for (quant_offset = (uint8_t *) dst; quant_offset < (uint8_t *) dst + quant_bytes; quant_offset++) {
        (*quant_offset) ^= 0x88;
    }
}

/**
 * @brief Transform CANN processed data back into quantized Q4.0 format.
 *
 * This function transforms CANN processed data back into quantized Q4.0 format.
 * It reverses the transformation performed by
 * ggml_backend_cann_transform_q4_0(), converting the data back into its
 * original quantized form.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source buffer containing transformed data.
 * @param dst Pointer to the destination buffer where the Q4.0 formatted data
 * will be stored.
 */
static void ggml_backend_cann_transform_back_q4_0(const ggml_tensor * tensor, void * src, void * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK4_0;
    size_t  quant_bytes = n_elems * sizeof(uint8_t) / 2;

    uint8_t *  quant_offset = (uint8_t *) src;
    uint16_t * scale_offset = (uint16_t *) ((char *) src + quant_bytes);

    for (; quant_offset < (uint8_t *) src + quant_bytes; quant_offset++) {
        (*quant_offset) ^= 0x88;
    }
    quant_offset = (uint8_t *) src;

    for (int i = 0; i < groups; i++) {
        block_q4_0 * group = (block_q4_0 *) ((char *) dst + i * sizeof(block_q4_0));
        group->d           = *scale_offset;
        scale_offset++;

        // 0-15
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j]     = ((*quant_offset) & 0x0F);
            group->qs[j + 1] = ((*quant_offset) >> 4);
            quant_offset++;
        }

        // 16-31
        for (int j = 0; j < QK4_0 / 2; j += 2) {
            group->qs[j] |= ((*quant_offset) << 4);
            group->qs[j + 1] |= ((*quant_offset) & 0xF0);
            quant_offset++;
        }
    }
}

/**
 * @brief Transform quantized Q8.0 tensor data into a format suitable for CANN
 * processing.
 *
 * This function transforms quantized Q8.0 tensor data into a format suitable
 * for CANN processing. It extracts quantization values and scales from the
 * source data and prepares them in a format expected by CANN operations.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data in Q8.0 format.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
static void ggml_backend_cann_transform_q8_0(ggml_tensor * tensor, const void * src, void * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_bytes = n_elems * sizeof(uint8_t);

    uint8_t *  quant_offset = (uint8_t *) dst;
    uint16_t * scale_offset = (uint16_t *) ((char *) dst + quant_bytes);

    for (int i = 0; i < groups; i++) {
        const block_q8_0 * group = (const block_q8_0 *) ((const char *) src + i * sizeof(block_q8_0));
        *scale_offset            = group->d;
        scale_offset++;
        size_t group_quant_size = QK8_0 * sizeof(uint8_t);
        memcpy(quant_offset, group->qs, group_quant_size);
        quant_offset += group_quant_size;
    }
}

/**
 * @brief Transform CANN processed data back into quantized Q8.0 format.
 *
 * This function transforms CANN processed data back into quantized Q8.0 format.
 * It reverses the transformation performed by
 * ggml_backend_cann_transform_q8_0(), converting the data back into its
 * original quantized form.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source buffer containing transformed data.
 * @param dst Pointer to the destination buffer where the Q8.0 formatted data
 * will be stored.
 */
static void ggml_backend_cann_transform_back_q8_0(const ggml_tensor * tensor, const void * src, void * dst) {
    int64_t n_elems     = ggml_nelements(tensor);
    int64_t groups      = n_elems / QK8_0;
    size_t  quant_bytes = n_elems * sizeof(uint8_t);

    const uint8_t *  quant_offset = (const uint8_t *) src;
    const uint16_t * scale_offset = (const uint16_t *) ((const char *) src + quant_bytes);

    for (int i = 0; i < groups; i++) {
        block_q8_0 * group = (block_q8_0 *) ((char *) dst + i * sizeof(block_q8_0));
        group->d           = *scale_offset;
        scale_offset++;
        size_t group_quant_size = QK8_0 * sizeof(uint8_t);
        memcpy(group->qs, quant_offset, group_quant_size);
        quant_offset += group_quant_size;
    }
}

/**
 * @brief Transform tensor data based on its type for CANN processing.
 *
 * This function transforms tensor data based on its quantization type for CANN
 * processing. It dispatches the transformation based on the tensor's type to
 * specialized functions handling Q4.0 and Q8.0 formats.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data to be transformed.
 * @param dst Pointer to the destination buffer where transformed data will be
 * stored.
 */
static void ggml_backend_cann_transform(ggml_tensor * tensor, const void * src, void * dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_q4_0(tensor, src, dst);
            break;
        case GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_q8_0(tensor, src, dst);
            break;
        default:
            break;
    }
}

/**
 * @brief Transform CANN processed data back into tensor data based on its type.
 *
 * This function transforms CANN processed data back into tensor data based on
 * its quantization type for Q4.0 and Q8.0 formats. It dispatches the
 * transformation based on the tensor's type to specialized functions.
 *
 * @param tensor Pointer to the tensor information.
 * @param src Pointer to the source data containing CANN processed data.
 * @param dst Pointer to the destination buffer where transformed tensor data
 * will be stored.
 */
static void ggml_backend_cann_transform_back(const ggml_tensor * tensor, void * src, void * dst) {
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
            ggml_backend_cann_transform_back_q4_0(tensor, src, dst);
            break;
        case GGML_TYPE_Q8_0:
            ggml_backend_cann_transform_back_q8_0(tensor, src, dst);
            break;
        default:
            break;
    }
}

/**
 * @brief Check if transformation is needed for a given tensor type.
 *
 * This function checks if transformation is needed for a given tensor type
 * to prepare data for CANN processing.
 *
 * @param type The tensor type to check.
 * @return true if transformation is needed, false otherwise.
 */
static bool need_transform(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Initialize a tensor using data from a CANN buffer.
 *
 * This function initializes a tensor using data from a CANN buffer.
 * It handles special cases such as views and quantization.
 *
 * @param buffer The CANN buffer from which to initialize the tensor.
 * @param tensor Pointer to the tensor to be initialized.
 */
static enum ggml_status ggml_backend_cann_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    // TODO: cann backend doesn't support quantized yet. Just leave the code
    // here.
    if (ggml_is_quantized(tensor->type)) {
        // Initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size   = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size && tensor->view_src == nullptr) {
            size_t memset_size = padded_size - original_size;
            ACL_CHECK(aclrtMemset((char *) tensor->data + original_size, memset_size, 0, memset_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Workspace for caching NZ buffers per device.
 *
 * This struct manages a device buffer used in NZ computations. It supports
 * allocation, reallocation, and clearing of cached memory. The struct is
 * designed to be used with a global array, one per device.
 */
struct ggml_cann_nz_workspace {
    std::mutex mtx;    // Protects ptr/allocated from concurrent access
    void * ptr;        // Pointer to allocated device buffer
    size_t allocated;  // Size of currently allocated buffer in bytes

    /**
     * @brief Constructor. Initializes the workspace with no allocated memory.
     */
    ggml_cann_nz_workspace() : ptr(nullptr), allocated(0) {}

    /**
     * @brief Free cached memory and reset the workspace.
     *
     * If a buffer has been allocated, this function releases it using
     * aclrtFree and resets internal state.
     */
    void clear() {
        if (ptr) {
            ACL_CHECK(aclrtFree(ptr));
            ptr       = nullptr;
            allocated = 0;
        }
    }

    /**
     * @brief Allocate or reallocate the workspace buffer.
     *
     * If the requested size is larger than the currently allocated size,
     * the old buffer will be freed and a new buffer of the requested size
     * will be allocated on the device.
     *
     * @param new_size Size in bytes to allocate for the workspace.
     */
    void realloc(size_t new_size) {
        if (new_size > allocated) {
            clear();
            ACL_CHECK(aclrtMalloc(&ptr, new_size, ACL_MEM_MALLOC_HUGE_FIRST));
            allocated = new_size;
        }
    }

    /**
     * @brief Get the device buffer pointer.
     *
     * @return Pointer to the allocated buffer, or nullptr if not allocated.
     */
    void * get() const { return ptr; }
};

/**
 * @brief Global array of NZ workspaces, one per device.
 */
static ggml_cann_nz_workspace g_nz_workspaces[GGML_CANN_MAX_DEVICES];

/**
 * @brief Convert tensor weights to NZ format using Ascend CANN API.
 *
 * This function creates a transposed tensor descriptor and performs the
 * TransMatmulWeight operation. Converting tensor formats can significantly
 * improve performance on certain hardware.
 *
 * @param tensor Pointer to the input ggml_tensor containing the weights.
 * @param offset Byte offset within the tensor data buffer where weights start.
 * @param device device id.
 *
 * @note The workspace buffer used in this function is managed globally and reused
 *       across calls. This reduces overhead from repeated memory allocation and deallocation.
 */
static void weight_format_to_nz(ggml_tensor * tensor, int device) {
    acl_tensor_ptr weightTransposed = ggml_cann_create_tensor(tensor, tensor->ne, tensor->nb, 2, ACL_FORMAT_ND, 0);
    uint64_t       workspaceSize    = 0;
    aclOpExecutor * executor;

    // TransMatmulWeight
    ACL_CHECK(aclnnTransMatmulWeightGetWorkspaceSize(weightTransposed.get(), &workspaceSize, &executor));

    std::lock_guard<std::mutex> lock(g_nz_workspaces[device].mtx);
    // Avoid frequent malloc/free of the workspace.
    g_nz_workspaces[device].realloc(workspaceSize);

    void * g_nz_workspace = g_nz_workspaces[device].get();

    ACL_CHECK(aclnnTransMatmulWeight(g_nz_workspace, workspaceSize, executor, nullptr));
}

// TODO: need handle tensor which has paddings.
/**
 * @brief Set tensor data in a CANN buffer.
 *
 * This function sets tensor data in a CANN buffer, handling transformations
 * if needed based on the tensor's type. It supports multi-threaded calls
 * where different threads write different chunks of the same tensor.
 *
 * For quantized tensors (Q4_0/Q8_0), data is staged in a host buffer and
 * the format transform is deferred until all chunks are written.
 * For NZ weight tensors, chunks are uploaded directly but the ND-to-NZ
 * conversion is deferred until all chunks are written.
 *
 * @param buffer The CANN buffer where the tensor data will be set.
 * @param tensor Pointer to the tensor whose data will be set.
 * @param data Pointer to the source data to be copied into the tensor.
 * @param offset Offset in the source data from where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
static void ggml_backend_cann_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *         tensor,
                                                const void *          data,
                                                size_t                offset,
                                                size_t                size) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;

    ggml_cann_set_device(ctx->device);

    // Only check env once.
    static bool weight_to_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    bool is_quantized = need_transform(tensor->type);
    bool is_nz        = !is_quantized && tensor->type != GGML_TYPE_BF16 && weight_to_nz &&
                 is_matmul_weight((const ggml_tensor *) tensor);

    // Plain tensor (not quantized, not NZ): direct copy, no tracking needed
    if (!is_quantized && !is_nz) {
        ACL_CHECK(aclrtMemcpy((char *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE));
        return;
    }

    // Single-shot write (full tensor at once): handle directly without tracking overhead
    if (offset == 0 && size == ggml_nbytes(tensor)) {
        if (is_quantized) {
            void * transform_buffer = malloc(size);
            ggml_backend_cann_transform(tensor, data, transform_buffer);
            ACL_CHECK(aclrtMemcpy(tensor->data, size, transform_buffer, size, ACL_MEMCPY_HOST_TO_DEVICE));
            free(transform_buffer);
        } else {
            // NZ weight
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            ACL_CHECK(aclrtMemcpy(tensor->data, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE));
            weight_format_to_nz(tensor, ctx->device);
        }
        return;
    }

    // Chunked write: use tracker to accumulate progress and defer transform/conversion
    TensorSetTracker * tracker = ctx->get_or_create_tracker(tensor);
    std::unique_lock<std::mutex> lock(tracker->mtx);

    if (is_quantized) {
        // Stage data in host buffer; transform requires full tensor data
        if (tracker->host_buffer.empty()) {
            tracker->host_buffer.resize(tracker->total_bytes);
        }
        memcpy(tracker->host_buffer.data() + offset, data, size);
    } else {
        // NZ weight: upload chunk to device immediately, defer conversion
        ACL_CHECK(aclrtMemcpy((char *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    tracker->bytes_written += size;

    // All chunks received: perform deferred transform/conversion
    if (tracker->bytes_written >= tracker->total_bytes) {
        if (is_quantized) {
            void * transform_buffer = malloc(tracker->total_bytes);
            ggml_backend_cann_transform(tensor, tracker->host_buffer.data(), transform_buffer);
            ACL_CHECK(aclrtMemcpy(tensor->data, tracker->total_bytes, transform_buffer, tracker->total_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
            free(transform_buffer);
        }

        if (is_nz) {
            GGML_ASSERT(tensor->ne[2] == 1);
            GGML_ASSERT(tensor->ne[3] == 1);
            weight_format_to_nz(tensor, ctx->device);
        }

        // Unlock before removing tracker, as remove_tracker destroys the mutex
        lock.unlock();
        ctx->remove_tracker(tensor);
    }
}

/**
 * @brief Get tensor data from a CANN buffer.
 *
 * This function retrieves tensor data from a CANN buffer, handling
 * transformations if needed based on the tensor's type.
 *
 * @param buffer The CANN buffer from which to retrieve tensor data.
 * @param tensor Pointer to the tensor whose data will be retrieved.
 * @param data Pointer to the destination buffer where the tensor data will be
 * copied.
 * @param offset Offset in the destination buffer where to start copying.
 * @param size Size of the data to be copied, in bytes.
 */
static void ggml_backend_cann_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *   tensor,
                                                void *                data,
                                                size_t                offset,
                                                size_t                size) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;

    ggml_cann_set_device(ctx->device);

    if (!need_transform(tensor->type)) {
        ACL_CHECK(aclrtMemcpy(data, size, (char *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        void * transform_buffer = malloc(size);
        ACL_CHECK(aclrtMemcpy(transform_buffer, size, (char *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST));
        ggml_backend_cann_transform_back(tensor, transform_buffer, data);
        free(transform_buffer);
    }
}

/**
 * @brief Copy tensor data between CANN buffers if possible.
 *
 * This function copies tensor data between CANN buffers if the source and
 * destination buffers are CANN buffers and they meet the necessary conditions
 * (same device or devices can access each other).
 *
 * @param buffer The destination CANN buffer where the tensor data will be
 * copied.
 * @param src Pointer to the source tensor whose data will be copied.
 * @param dst Pointer to the destination tensor where the data will be copied.
 * @return true if the copy operation succeeded, false otherwise.
 */
static bool ggml_backend_cann_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *   src,
                                                ggml_tensor *         dst) {
    if (ggml_backend_buft_is_cann(src->buffer->buft)) {
        ggml_backend_cann_buffer_context * src_ctx = (ggml_backend_cann_buffer_context *) src->buffer->context;
        ggml_backend_cann_buffer_context * dst_ctx = (ggml_backend_cann_buffer_context *) buffer->context;

        size_t memcpy_size = ggml_nbytes(src);
        // Same device.
        if (src_ctx->device == dst_ctx->device) {
            ACL_CHECK(aclrtMemcpy((char *) dst->data, memcpy_size, (const char *) src->data, memcpy_size,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE));
            return true;
        } else {
#ifdef ASCEND_310P
            // TODO: Support 310p P2P copy
            return false;
#endif
            // Different device but can access by peer.
            int32_t canAccessPeer = 0;
            ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, src_ctx->device, dst_ctx->device));
            if (canAccessPeer) {
                ggml_cann_set_device(src_ctx->device);
                ACL_CHECK(aclrtDeviceEnablePeerAccess(dst_ctx->device, 0));
                ACL_CHECK(aclrtMemcpy((char *) dst->data, memcpy_size, (const char *) src->data, memcpy_size,
                                      ACL_MEMCPY_DEVICE_TO_DEVICE));
                return true;
            }
        }
    }
    return false;
}

/**
 * @brief Set a region of a tensor's device memory to a specified value.
 *
 * @param buffer The CANN buffer containing the tensor.
 * @param tensor Pointer to the tensor whose memory will be set.
 * @param value The value to which each byte in the region will be set.
 * @param offset Byte offset within the tensor's data to start setting.
 * @param size Number of bytes to set.
 */
static void ggml_backend_cann_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtMemset((char *) tensor->data + offset, size, value, size));
}

/**
 * @brief Clear a CANN buffer by setting all its memory to a specified value.
 *
 * This function clears a CANN buffer by setting all its memory to a specified
 * value.
 *
 * @param buffer The CANN buffer to be cleared.
 * @param value The value to which each byte in the buffer will be set.
 */
static void ggml_backend_cann_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cann_buffer_context * ctx = (ggml_backend_cann_buffer_context *) buffer->context;

    ggml_cann_set_device(ctx->device);
    ACL_CHECK(aclrtMemset(ctx->dev_ptr, buffer->size, value, buffer->size));
}

/**
 * @brief Interface for a CANN buffer in the backend.
 *
 * This structure defines function pointers to operations that can be performed
 * on a CANN buffer within the backend.
 */
static const ggml_backend_buffer_i ggml_backend_cann_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cann_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cann_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cann_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cann_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cann_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cann_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cann_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cann_buffer_clear,
    /* .reset           = */ NULL,
};

/**
 * @brief Allocates a new CANN buffer of the specified type and size.
 *
 * This function allocates a new CANN buffer on the specified device with the
 * given size.
 *
 * @param buft Pointer to the buffer type context.
 * @param size Size in bytes of the buffer to allocate.
 * @return Pointer to the allocated buffer, or nullptr if allocation fails.
 */
static ggml_backend_buffer_t ggml_backend_cann_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cann_buffer_type_context * buft_ctx = (ggml_backend_cann_buffer_type_context *) buft->context;

    ggml_cann_set_device(buft_ctx->device);

    const size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }
    void *   dev_ptr;
    aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != ACL_SUCCESS) {
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: aclrtMalloc failed: %s\n", __func__,
                       size / 1024.0 / 1024.0, buft_ctx->device, aclGetRecentErrMsg());
        return nullptr;
    }

    ggml_backend_cann_buffer_context * ctx = new ggml_backend_cann_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cann_buffer_interface, ctx, size);
}

/**
 * @brief Retrieves the memory alignment requirement for CANN buffers of this
 * type.
 *
 * This function returns the alignment requirement in bytes for memory allocated
 * by the CANN buffer type.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @return The alignment requirement in bytes (fixed at 128 bytes for CANN
 * buffers).
 */
static size_t ggml_backend_cann_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

/**
 * @brief Calculates the allocation size required for a tensor in a CANN buffer.
 *
 * Computes the total allocation size needed for storing the tensor's data in a
 * CANN buffer, considering any necessary padding or adjustments for quantized
 * types.
 *
 * @param buft Pointer to the buffer type context (unused in this
 * implementation).
 * @param tensor Pointer to the tensor for which the allocation size is
 * calculated.
 * @return The total allocation size in bytes required for the tensor in the
 * CANN buffer.
 */
static size_t ggml_backend_cann_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                           const ggml_tensor *        tensor) {
    size_t  size = ggml_nbytes(tensor);
    int64_t ne0  = tensor->ne[0];

    // Only check env once.
    static bool weight_to_nz = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));

    // last line must bigger than 32, because every single op deal at
    // least 32 bytes.
    // TODO: quantized type?
    // int64_t line_size = ne0 * ggml_element_size(tensor);
    // int64_t line_size_align_32 = (line_size + 31) & ~31;
    // size += (line_size_align_32 - line_size);
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    } else if (weight_to_nz && tensor->type != GGML_TYPE_BF16
               && is_matmul_weight((const ggml_tensor *) tensor)) {
        // NZ format weight are not support quantized yet.
        // If ND tensor transform to NZ, size may changed.
        int64_t shape[] = { tensor->ne[1], tensor->ne[0] };
        GGML_ASSERT(tensor->ne[2] == 1);
        GGML_ASSERT(tensor->ne[3] == 1);
        const aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
        size_t              new_size;
        ACL_CHECK(aclnnCalculateMatmulWeightSizeV2(acl_shape, ggml_cann_type_mapping(tensor->type), &new_size));
        ACL_CHECK(aclDestroyIntArray(acl_shape));
        size = std::max(size, new_size);
    }

    return size;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cann_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

/**
 * @brief Interface for managing CANN buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
static const ggml_backend_buffer_type_i ggml_backend_cann_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cann_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cann_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cann_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cann_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cann_buffer_type_is_host,
};

/**
 * @brief Retrieves the CANN buffer type for a specified device.
 *
 * This function initializes and returns the buffer type interface associated
 * with the given device. It ensures thread-safe access using a mutex.
 *
 * @param device The device index for which to retrieve the buffer type.
 * @return A pointer to the buffer type interface for the specified device, or
 * nullptr if the device index is out of range.
 */
ggml_backend_buffer_type_t ggml_backend_cann_buffer_type(int32_t device) {
    static std::mutex           mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cann_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cann_buffer_types[GGML_CANN_MAX_DEVICES];

    static bool ggml_backend_cann_buffer_type_initialized = false;

    if (!ggml_backend_cann_buffer_type_initialized) {
        for (int32_t i = 0; i < ggml_cann_info().device_count; i++) {
            ggml_backend_cann_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cann_buffer_type_interface,
                /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), i),
                /* .context  = */
                new ggml_backend_cann_buffer_type_context{ i, "CANN" + std::to_string(i) },
            };
        }
        ggml_backend_cann_buffer_type_initialized = true;
    }

    return &ggml_backend_cann_buffer_types[device];
}

/**
 * @brief Retrieves the name associated with a CANN host buffer type.
 *
 * This function returns the descriptive name associated with the specified
 * CANN host buffer type context.
 *
 * @param buft Pointer to the host buffer type context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char * ggml_backend_cann_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return "CANN_Host";

    GGML_UNUSED(buft);
}

/**
 * @brief Retrieves the name associated with a CANN host buffer.
 *
 * This function returns the descriptive name associated with the specified
 * CANN host buffer context.
 *
 * @param buft Pointer to the host buffer context.
 * @return Const pointer to the C-style string containing the name.
 */
static const char * ggml_backend_cann_host_buffer_name(ggml_backend_buffer_t buffer) {
    return "CANN_Host";

    GGML_UNUSED(buffer);
}

/**
 * @brief Free resources associated with a CANN host buffer.
 *
 * This function frees the resources associated with a CANN host buffer, including
 * its context.
 *
 * @param buffer The CANN host buffer to free.
 */
static void ggml_backend_cann_host_buffer_free(ggml_backend_buffer_t buffer) {
    ACL_CHECK(aclrtFreeHost(buffer->context));
}

/**
 * @brief Allocates a new CANN host buffer of the specified size.
 *
 * This function allocates a new CANN host buffer with the given size.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or nullptr if allocation fails.
 */
static void * ggml_cann_host_malloc(size_t size) {
    if (getenv("GGML_CANN_NO_PINNED") != nullptr) {
        return nullptr;
    }

    const size_t alignment = 128;
    size                   = GGML_PAD(size, alignment);
    if (size == 0) {
        size = alignment;
    }

    void *   hostPtr = nullptr;
    aclError err     = aclrtMallocHost((void **) &hostPtr, size);
    if (err != ACL_SUCCESS) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, size / 1024.0 / 1024.0,
                      aclGetRecentErrMsg());
        return nullptr;
    }
    return hostPtr;
}

/**
 * @brief Allocates a new CANN host buffer of the specified type and size.
 *
 * @param buft Pointer to the host buffer type context.
 * @param size Size in bytes of the host buffer to allocate.
 * @return Pointer to the allocated host buffer, or CPU buffer pointer if allocation fails.
 */
static ggml_backend_buffer_t ggml_backend_cann_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                             size_t                     size) {
    void * hostPtr = ggml_cann_host_malloc(size);

    if (hostPtr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(hostPtr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cann_host_buffer_free;

    return buffer;
}

/**
 * @brief Interface for managing CANN host buffer types in the GGML backend.
 *
 * Provides function pointers for allocating, querying properties, and managing
 * memory for CANN buffer types in the GGML backend.
 */
ggml_backend_buffer_type_t ggml_backend_cann_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cann_buffer_type_host = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cann_host_buffer_type_name,
                           /* .alloc_buffer     = */ ggml_backend_cann_host_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                           /* .get_max_size     = */ NULL,  // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                           /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                           },
        /* .device   = */
        ggml_backend_reg_dev_get(ggml_backend_cann_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cann_buffer_type_host;
}

/**
 * @brief Computes the forward operation for a given tensor using CANN
 * operations.
 *
 * This function selects the appropriate CANN operation based on the type of
 * operation specified in the tensor and performs the computation.
 *
 * @param ctx The CANN context containing necessary resources and
 * configurations.
 * @param dst The destination tensor where the result of the computation will be
 * stored.
 * @return true if the computation was successful; false otherwise.
 */
static bool ggml_cann_compute_forward(ggml_backend_cann_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_REPEAT:
            ggml_cann_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cann_get_rows(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cann_set_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
            ggml_cann_binary_op<aclnn_add>(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cann_binary_op<aclnn_sub>(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cann_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cann_binary_op<aclnn_mul>(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cann_binary_op<aclnn_div>(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    GGML_CANN_CALL_OP_UNARY(Abs);
                    break;
                case GGML_UNARY_OP_NEG:
                    GGML_CANN_CALL_OP_UNARY(Neg);
                    break;
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                    // aclnnGelu internally uses the erf-based approximation.
                    GGML_CANN_CALL_OP_UNARY(Gelu);
                    break;
                case GGML_UNARY_OP_SILU:
                    GGML_CANN_CALL_OP_UNARY(Silu);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    {
                        auto lambda = [](ggml_backend_cann_context & ctx, aclTensor * acl_src, aclTensor * acl_dst) {
                            GGML_CANN_CALL_ACLNN_OP(ctx, GeluV2, acl_src, 0, acl_dst);
                        };
                        ggml_cann_op_unary(lambda, ctx, dst);
                    }
                    break;
                case GGML_UNARY_OP_TANH:
                    GGML_CANN_CALL_OP_UNARY(Tanh);
                    break;
                case GGML_UNARY_OP_RELU:
                    GGML_CANN_CALL_OP_UNARY(Relu);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Sigmoid);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    GGML_CANN_CALL_OP_UNARY(Hardsigmoid);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    GGML_CANN_CALL_OP_UNARY(Hardswish);
                    break;
                case GGML_UNARY_OP_EXP:
                    GGML_CANN_CALL_OP_UNARY(Exp);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cann_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    GGML_CANN_CALL_OP_UNARY(Sign);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cann_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cann_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    GGML_CANN_CALL_OP_UNARY_GATED(Relu);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cann_geglu(ctx, dst, 0);  // approximate=0 → tanh
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cann_geglu(ctx, dst, 1);  // approximate=1 → erf
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cann_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cann_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cann_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cann_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cann_l2_norm(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cann_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cann_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cann_upsample_nearest2d(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cann_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cann_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cann_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cann_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cann_rms_norm(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cann_mul_mat(ctx, dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cann_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cann_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            GGML_ASSERT(dst->src[1] == nullptr);
            dst->src[1] = dst->src[0];
            ggml_cann_binary_op<aclnn_mul>(ctx, dst);
            break;
        case GGML_OP_SQRT:
            GGML_CANN_CALL_OP_UNARY(Sqrt);
            break;
        case GGML_OP_CLAMP:
            ggml_cann_clamp(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cann_cpy(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cann_set(ctx, dst);
            break;
        case GGML_OP_CONT:
            ggml_cann_dup(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cann_diag_mask(ctx, dst, -INFINITY);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cann_softmax(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cann_rope(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cann_im2col(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cann_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cann_sum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cann_sum_rows(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cann_argsort(ctx, dst);
            break;
        case GGML_OP_ARGMAX:
            ggml_cann_argmax(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cann_op_unary<aclnn_cos>(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cann_op_unary<aclnn_sin>(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cann_conv_transpose_1d(ctx, dst);
            break;
        case GGML_OP_LOG:
            GGML_CANN_CALL_OP_UNARY(Log);
            break;
        case GGML_OP_MEAN:
            ggml_cann_mean(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cann_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cann_count_equal(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cann_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cann_out_prod(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cann_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cann_ssm_conv(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cann_cumsum(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cann_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cann_fill(ctx, dst);
            break;
        case GGML_OP_DIAG:
            ggml_cann_diag(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cann_solve_tri(ctx, dst);
            break;
        default:
            return false;
    }

    return true;
}

// backend
/**
 * @brief Retrieves the name associated with the CANN backend.
 *
 * This function returns the name assigned to the CANN backend, which is stored
 * in the context of the provided backend structure.
 *
 * @param backend Pointer to the CANN backend structure.
 * @return A pointer to a constant string representing the backend name.
 */
static const char * ggml_backend_cann_name(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;

    return cann_ctx->name.c_str();
}

/**
 * @brief Frees resources associated with the CANN backend.
 *
 * This function releases resources associated with the CANN backend context
 * and resets the device associated with the backend to its initial state.
 *
 * @param backend Pointer to the CANN backend structure to be freed.
 */
static void ggml_backend_cann_free(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrtSynchronizeDevice());
    ACL_CHECK(aclrtResetDevice(cann_ctx->device));

    delete cann_ctx;
    delete backend;
}

/**
 * @brief Sets tensor data asynchronously in the CANN backend.
 *
 * This function asynchronously sets tensor data in the CANN backend.
 *
 * @param backend Pointer to the CANN backend structure.
 * @param tensor Pointer to the tensor structure to set data for.
 * @param data Pointer to the host data to copy to the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
static void ggml_backend_cann_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *  tensor,
                                               const void *   data,
                                               size_t         offset,
                                               size_t         size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_buffer_t       buf      = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cann_buffer_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync((char *) tensor->data + offset, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE,
                               cann_ctx->stream()));
}

/**
 * @brief Gets tensor data asynchronously in the CANN backend.
 *
 * This function asynchronously gets tensor data in the CANN backend.
 *
 * @param backend Pointer to the CANN backend structure.
 * @param tensor Pointer to the tensor structure to get data from.
 * @param data Pointer to the host data to copy from the tensor.
 * @param offset Offset in bytes within the host data.
 * @param size Size of the data to copy in bytes.
 */
static void ggml_backend_cann_get_tensor_async(ggml_backend_t      backend,
                                               const ggml_tensor * tensor,
                                               void *              data,
                                               size_t              offset,
                                               size_t              size) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_backend_buffer_t       buf      = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cann_buffer_type(cann_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(!ggml_is_quantized(tensor->type));

    ACL_CHECK(aclrtMemcpyAsync(data, size, (char *) tensor->data + offset, size, ACL_MEMCPY_DEVICE_TO_HOST,
                               cann_ctx->stream()));
}

/**
 * @brief Asynchronously copies tensor data between CANN backends.
 *
 * This function copies tensor data asynchronously between two CANN backends. It
 * checks if both tensors reside in CANN buffers and whether the devices support
 * peer-to-peer access for direct copying. If not, it returns false.
 *
 * @param backend_src Pointer to the source CANN backend structure.
 * @param backend_dst Pointer to the destination CANN backend structure.
 * @param src Pointer to the source tensor to copy data from.
 * @param dst Pointer to the destination tensor to copy data to.
 * @return true if the copy operation succeeds, false otherwise.
 */
static bool ggml_backend_cann_cpy_tensor_async(ggml_backend_t      backend_src,
                                               ggml_backend_t      backend_dst,
                                               const ggml_tensor * src,
                                               ggml_tensor *       dst) {
    GGML_ASSERT(ggml_backend_is_cann(backend_src) || ggml_backend_is_cann(backend_dst));

    GGML_ASSERT(!is_matmul_weight((const ggml_tensor *) src));

    if (!ggml_backend_buft_is_cann(src->buffer->buft) || !ggml_backend_buft_is_cann(dst->buffer->buft)) {
        return false;
    }

    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    ggml_backend_cann_context * cann_ctx_src = (ggml_backend_cann_context *) backend_src->context;
    ggml_backend_cann_context * cann_ctx_dst = (ggml_backend_cann_context *) backend_dst->context;

    size_t copy_size = ggml_nbytes(dst);
    if (copy_size == 0) {
        return true;
    }
    if (backend_src != backend_dst) {
#ifdef ASCEND_310P
        // TODO: Support 310p P2P copy
        return false;
#endif
        ggml_backend_cann_buffer_context * buf_ctx_src = (ggml_backend_cann_buffer_context *) buf_src->context;
        ggml_backend_cann_buffer_context * buf_ctx_dst = (ggml_backend_cann_buffer_context *) buf_dst->context;

        GGML_ASSERT(cann_ctx_src->device == buf_ctx_src->device);
        GGML_ASSERT(cann_ctx_dst->device == buf_ctx_dst->device);

        int32_t canAccessPeer = 0;
        ACL_CHECK(aclrtDeviceCanAccessPeer(&canAccessPeer, cann_ctx_src->device, cann_ctx_dst->device));
        if (!canAccessPeer) {
            return false;
        }

        // need open both directions for memcpyasync between devices.
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_src->device, 0));
        ggml_cann_set_device(cann_ctx_src->device);
        ACL_CHECK(aclrtDeviceEnablePeerAccess(cann_ctx_dst->device, 0));

        // wait for task_queue empty to keep task order.
        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_src->stream()));
        // record event on src stream after the copy
        // TODO: this event is not effective with acl graph mode, change to use aclrtSynchronizeStream
        // if (!cann_ctx_src->copy_event) {
        //     ACL_CHECK(aclrtCreateEventWithFlag(&cann_ctx_src->copy_event, ACL_EVENT_SYNC));
        // }
        // ACL_CHECK(aclrtRecordEvent(cann_ctx_src->copy_event, cann_ctx_src->stream()));

        // // wait on dst stream for the copy to complete
        // ggml_cann_set_device(cann_ctx_dst->device);
        // ACL_CHECK(aclrtStreamWaitEvent(cann_ctx_dst->stream(), cann_ctx_src->copy_event));
        ACL_CHECK(aclrtSynchronizeStream(cann_ctx_src->stream()));
    } else {
        // src and dst are on the same backend
        ACL_CHECK(aclrtMemcpyAsync(dst->data, copy_size, src->data, copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   cann_ctx_dst->stream()));
    }

    return true;
}

/**
 * @brief Synchronizes a CANN backend.
 *
 * This function synchronizes the specified CANN backend by waiting for all
 * operations in its associated stream to complete.
 *
 * @param backend Pointer to the CANN backend structure to synchronize.
 */
static void ggml_backend_cann_synchronize(ggml_backend_t backend) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_set_device(cann_ctx->device);
    ACL_CHECK(aclrtSynchronizeStream(cann_ctx->stream()));
}

/**
 * @brief Check if CANN backend can fuse the specified operation sequence
 *
 * This function determines whether an operation sequence starting from the specified node
 * can be fused into an optimized operation in the CANN backend. Operation fusion can reduce
 * memory access overhead and improve computational efficiency.
 *
 * @param cgraph Pointer to the computation graph
 * @param node_idx Index of the starting node in the computation graph
 * @param ops Sequence of operation types to check for fusion
 * @return true if the operations can be fused
 * @return false if the operations cannot be fused
 */
static bool ggml_cann_can_fuse(const struct ggml_cgraph *          cgraph,
                               int                                 node_idx,
                               std::initializer_list<enum ggml_op> ops) {
    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    // CANN backend supports fusing ADD + RMS_NORM operations
    if ((ops.size() == 2) && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_RMS_NORM) {
        ggml_tensor * add_node = cgraph->nodes[node_idx];
        // TODO: support broadcast for ADD + RMS_NORM
        if (add_node->src[0]->ne[0] != add_node->src[1]->ne[0] || add_node->src[0]->ne[1] != add_node->src[1]->ne[1] ||
            add_node->src[0]->ne[2] != add_node->src[1]->ne[2] || add_node->src[0]->ne[3] != add_node->src[1]->ne[3]) {
            return false;
        }
        return true;
    }

    return false;
}

/**
 * @brief Evaluate the computation graph and optionally capture or execute it using CANN graph API.
 *
 * If CANN graph execution is enabled and graph capture is required, this function begins
 * graph capture, runs the graph, ends capture, and stores the captured graph.
 *
 * Otherwise, it falls back to op-by-op execution using the CANN compute kernel dispatcher.
 *
 * @param cann_ctx                     The CANN backend context.
 * @param cgraph                       The ggml computation graph.
 * @param use_cann_graph               Whether to use CANN graph execution.
 * @param cann_graph_capture_required  Whether graph capture is needed due to graph changes.
 */
static void evaluate_and_capture_cann_graph(ggml_backend_cann_context * cann_ctx,
                                            ggml_cgraph *               cgraph,
                                            bool                        use_cann_graph,
                                            bool                        cann_graph_capture_required) {
#ifdef USE_ACL_GRAPH
    if (use_cann_graph && cann_graph_capture_required) {  // Begin CANN graph capture
        ACL_CHECK(aclmdlRICaptureBegin(cann_ctx->stream(), ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    }
#endif  // USE_ACL_GRAPH
    // Only perform the graph execution if CANN graphs are not enabled, or we are capturing the graph.
    // With the use of CANN graphs, the execution will be performed by the graph launch.
    static bool opt_fusion = parse_bool(get_env_as_lowercase("GGML_CANN_OPERATOR_FUSION").value_or(""));

    if (!use_cann_graph || cann_graph_capture_required) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            if (opt_fusion) {
                if (ggml_cann_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_RMS_NORM })) {
                    ggml_cann_op_add_rms_norm_fused(*cann_ctx, node, cgraph->nodes[i + 1]);
                    i++;
                    continue;
                }
            }

            if (ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
                node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
                continue;
            }

            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                continue;
            }

            bool ok = ggml_cann_compute_forward(*cann_ctx, node);
            if (!ok) {
                GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
            }
            GGML_ASSERT(ok);
        }
    }

#ifdef USE_ACL_GRAPH
    if (use_cann_graph) {
        GGML_ASSERT(!cann_ctx->graph_lru_cache.cache_list.empty());
        ggml_cann_graph * matched_graph = cann_ctx->graph_lru_cache.cache_list.front();

        if (cann_graph_capture_required) {  // End CANN graph capture
            ACL_CHECK(aclmdlRICaptureEnd(cann_ctx->stream(), &matched_graph->graph));
        }

        // Execute CANN graph
        ACL_CHECK(aclmdlRIExecuteAsync(matched_graph->graph, cann_ctx->stream()));
    }
#endif  // USE_ACL_GRAPH
}

/**
 * @brief Computes a computational graph using a CANN backend.
 *
 * This function computes the operations defined in the computational graph
 * using the specified CANN backend.
 *
 * @param backend Pointer to the CANN backend structure to use for computation.
 * @param cgraph Pointer to the computational graph structure containing nodes
 *               representing operations to be computed.
 * @return enum ggml_status Returns GGML_STATUS_SUCCESS if computation
 *         completes successfully, otherwise an appropriate error status.
 */
static enum ggml_status ggml_backend_cann_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ggml_cann_set_device(cann_ctx->device);
    g_nz_workspaces[cann_ctx->device].clear();

    // calculate rope cache for fist layer in current device.
    cann_ctx->rope_cache.cached = false;

    bool graph_capture_required = false;
#ifdef USE_ACL_GRAPH
    bool use_cann_graph = true;

    static bool prefill_use_graph = parse_bool(get_env_as_lowercase("GGML_CANN_PREFILL_USE_GRAPH").value_or(""));
    if (!prefill_use_graph) {
        // Do not use acl_graph for prefill.
        for (int i = 0; i < cgraph->n_nodes; i++) {
            ggml_tensor * node = cgraph->nodes[i];
            // TODO: Optimize here. Currently, we can only
            // get seq_len by FA's input.
            if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                // Q -> src[0], shape: [B, S, N, D]
                use_cann_graph = (node->src[0]->ne[1] == 1);
                break;
            }
        }
    }

    if (!cann_ctx->acl_graph_mode) {
        use_cann_graph = false;
    }

    if (use_cann_graph) {
        // If no matching graph is found, the graph needs to be recaptured.
        graph_capture_required = !cann_ctx->graph_lru_cache.find_and_move_to_front(cgraph);

        if (graph_capture_required) {
            // If no matching graph is found, add a new ACL graph.
            ggml_cann_graph * new_graph = ggml_cann_graph::create_from_cgraph(cgraph);
            cann_ctx->graph_lru_cache.push(new_graph);

            // Pre-load rope cache before graph capture.  During capture the
            // stream cannot perform host-to-device memcpy or device memory
            // malloc/free.  Running the full cache init now populates the
            // cache metadata so these branches are skipped during capture,
            // while also warming up the memory pool.
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->op == GGML_OP_ROPE) {
                    ggml_cann_rope_cache_preload(*cann_ctx, node);
                    break;
                }
            }
        }
    }
#else
    bool use_cann_graph = false;
#endif  // USE_ACL_GRAPH
    evaluate_and_capture_cann_graph(cann_ctx, cgraph, use_cann_graph, graph_capture_required);

    return GGML_STATUS_SUCCESS;
}

/**
 * @brief Checks if the CANN backend supports a specific operation.
 *
 * This function checks whether the specified operation is supported by the
 * CANN backend.
 *
 * @param backend Pointer to the CANN backend structure to check support for
 *                the operation.
 * @param op Pointer to the tensor representing the operation to check.
 * @return bool Returns true if the operation is supported by the backend,
 *              otherwise false.
 */
static bool ggml_backend_cann_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_SOFTPLUS:
                    return true;
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return true;
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
            {
                switch (op->src[0]->type) {
#ifndef ASCEND_310P
                    case GGML_TYPE_BF16:
#endif
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                        return true;
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
#ifdef ASCEND_310P
                        // Q4 && Q8 per group is not support on 310p device
                        return false;
#endif
                        // only support contiguous for quantized types.
                        return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
                    default:
                        return false;
                }
            }
        case GGML_OP_MUL_MAT_ID:
            switch (op->src[0]->type) {
                case GGML_TYPE_F16:
                case GGML_TYPE_F32:
                    return true;
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
#ifdef ASCEND_310P
                    // Q4 && Q8 per group is not support on 310p device
                    return false;
#endif
                    // only support contiguous for quantized types.
                    return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
                default:
                    return false;
            }
        // embedding
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
#ifndef ASCEND_310P
                    case GGML_TYPE_BF16:
#endif
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
            break;
        case GGML_OP_SET_ROWS:
            {
                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
#ifndef ASCEND_310P
                    case GGML_TYPE_BF16:
#endif
                        return true;
                    default:
                        return false;
                }
            }
            break;
        case GGML_OP_CPY:
            {
                ggml_tensor * src = op->src[0];
#ifdef ASCEND_310P
                if ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16)) {
                    // only support F32 and F16 on 310P.
                    return false;
                }
#else
                if ((op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_BF16) ||
                    (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16 && src->type != GGML_TYPE_BF16)) {
                    // only support F32, F16 and BF16.
                    return false;
                }
#endif
                return true;
            }
            break;
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
#ifndef ASCEND_310P
                    case GGML_TYPE_BF16:
#endif
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_ROPE:
            {
                if (op->src[0]->ne[0] > 896) {
                    return false;
                }
#ifdef ASCEND_310P
                // TODO: Support rope_dim < ne00(dim)
                if (op->src[0]->ne[0] != op->op_params[1]) {
                    return false;
                }
                if (!ggml_is_contiguous(op->src[0])) {
                    return false;
                }
#endif
                return true;
            }
        case GGML_OP_UPSCALE:
            {
                // aclnnUpsampleNearest2dGetWorkspaceSize not support
                // selfDimN[2]/outDimN[2] or selfDimC[3]/outDimC[3] not equal
                if (op->src[0]->ne[2] * op->ne[3] != op->src[0]->ne[3] * op->ne[2]) {
                    return false;
                }
                if (op->op_params[0] != GGML_SCALE_MODE_NEAREST) {
                    return false;
                }
                if (op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS) {
                    return false;
                }
                return true;
            }
        case GGML_OP_POOL_2D:
            {
                const int32_t * opts = (const int32_t *) op->op_params;
#ifdef ASCEND_310P
                enum ggml_op_pool opt = static_cast<ggml_op_pool>(opts[0]);
                if (opt == GGML_OP_POOL_MAX) {
                    return false;
                }
#endif
                const int k0 = opts[1];
                const int k1 = opts[2];
                const int p0 = opts[5];
                const int p1 = opts[6];
                // value of paddingH should be at most half of kernelH
                // value of paddingW should be at most half of kernelW
                return (p0 <= (k0 / 2)) && (p1 <= (k1 / 2));
            }
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_L2_NORM:
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_DUP:
        case GGML_OP_IM2COL:
        case GGML_OP_CONCAT:
        case GGML_OP_REPEAT:
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_SET:
        case GGML_OP_GROUP_NORM:
            return true;
        case GGML_OP_PAD:
            // TODO: add circular padding support for cann, see https://github.com/ggml-org/llama.cpp/pull/16985
            return ggml_get_op_params_i32(op, 8) == 0;
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_ARGMAX:
        case GGML_OP_COS:
        case GGML_OP_SIN:
        case GGML_OP_LOG:
        case GGML_OP_MEAN:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_COUNT_EQUAL:
        case GGML_OP_GATED_LINEAR_ATTN:
            return true;
        case GGML_OP_OUT_PROD:
            {
#ifdef ASCEND_310P
                // Ger is not supported on 310p device
                return false;
#endif
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_SCALE:
            float bias;
            memcpy(&bias, (const float *) (op->op_params) + 1, sizeof(float));
            return bias == 0.0f;  // TODO: support bias != 0.0f
        case GGML_OP_SOFT_MAX:
            // TODO: support attention sinks [TAG_ATTN_SINKS]
            if (op->src[2]) {
                return false;
            }
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            {
#ifdef ASCEND_310P
                // FA not support on 310p device
                return false;
#endif
                // derived from [ggml-cuda.cu]
                if (op->src[1]->type != GGML_TYPE_F16 || op->src[2]->type != GGML_TYPE_F16) {
                    return false;
                }
                if (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_F32 &&
                    op->src[1]->type != GGML_TYPE_BF16) {
                    return false;
                }
                if (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_BF16) {
                    return false;
                }
                // TODO: support attention sinks [TAG_ATTN_SINKS]
                if (op->src[4]) {
                    return false;
                }
                if (op->src[1]->ne[0] != op->src[2]->ne[0]) {
                    // different head sizes of K and V are not supported yet
                    return false;
                }
                float logitSoftcap = 0.0f;
                memcpy(&logitSoftcap, (const float *) (op->op_params) + 2, sizeof(float));
                if (logitSoftcap != 0.0f) {
                    return false;
                }
                return true;
            }
        case GGML_OP_SSM_CONV:
            return true;
        case GGML_OP_CUMSUM:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_TRI:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_FILL:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_DIAG:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_SOLVE_TRI:
            return op->src[0]->type == GGML_TYPE_F32;
        default:
            return false;
    }

    GGML_UNUSED(dev);
}

/**
 * @brief Records an event on the CANN backend stream.
 *
 * This function records the given event on the ACL runtime stream associated
 * with the backend context.
 *
 * @param event Pointer to the event structure to be recorded.
 */
static void ggml_backend_cann_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    ACL_CHECK(aclrtRecordEvent((aclrtEvent) event->context, cann_ctx->stream()));
}

/**
 * @brief Waits for a recorded event to complete on the CANN backend stream.
 *
 * This function makes the given backend wait for the event to complete on its
 * ACL runtime stream.
 *
 * @param backend Pointer to the backend structure.
 * @param event Pointer to the event structure that the backend needs to wait
 * for.
 */
static void ggml_backend_cann_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cann_context * cann_ctx = (ggml_backend_cann_context *) backend->context;
    if (ggml_backend_is_cann(backend)) {
        ACL_CHECK(aclrtStreamWaitEvent(cann_ctx->stream(), (aclrtEvent) event->context));
    } else {
        GGML_ABORT("fatal error");
    }
}

/**
 * @brief Structure defining the interface for the CANN backend.
 *
 * This structure contains function pointers for various operations
 * supported by the CANN backend, including name retrieval, memory
 * management, tensor operations, synchronization, and event handling.
 */
static const ggml_backend_i ggml_backend_cann_interface = {
    /* .get_name                = */ ggml_backend_cann_name,
    /* .free                    = */ ggml_backend_cann_free,
    /* .set_tensor_async        = */ ggml_backend_cann_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cann_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_cann_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cann_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cann_graph_compute,
    /* .event_record            = */ ggml_backend_cann_event_record,
    /* .event_wait              = */ ggml_backend_cann_event_wait,
    /* .graph_optimize          = */ NULL,
};

/**
 * @brief Return the hardcoded GUID for the CANN backend.
 *
 * This function returns a static GUID which uniquely identifies the CANN
 * backend.
 *
 * @return A pointer to the static GUID.
 */
static ggml_guid_t ggml_backend_cann_guid() {
    static ggml_guid guid = { 0xa1, 0x94, 0xaf, 0xac, 0xbd, 0x4f, 0x47, 0x34,
                              0xbe, 0x1a, 0x9e, 0x71, 0x1f, 0x9e, 0xed, 0x64 };
    return &guid;
}

// backend device
struct ggml_backend_cann_device_context {
    int         device;
    std::string name;
    std::string description;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cann_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cann_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_cann_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    ggml_backend_cann_get_device_memory(ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_cann_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cann_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cann_device_get_name(dev);
    props->description = ggml_backend_cann_device_get_description(dev);
    props->type        = ggml_backend_cann_device_get_type(dev);
    ggml_backend_cann_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CANN_NO_PINNED") == nullptr;

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
    };
}

static ggml_backend_t ggml_backend_cann_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    return ggml_backend_cann_init(ctx->device);
}

/**
 * @brief Checks if the CANN backend supports a specific backend buffer type.
 *
 * This function determines whether the CANN backend supports the given backend
 * buffer type by comparing the device context of the backend and buffer type.
 * It returns true if the devices are same between the backend context and
 * buffer type context.
 *
 * @param backend Pointer to the CANN backend.
 * @param buft Pointer to the backend buffer type to check.
 * @return bool Returns true if the CANN backend supports the buffer type,
 *              otherwise false.
 */
static bool ggml_backend_cann_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (ggml_backend_buft_is_cann(buft)) {
        ggml_backend_cann_device_context *      dev_ctx  = (ggml_backend_cann_device_context *) dev->context;
        ggml_backend_cann_buffer_type_context * buft_ctx = (ggml_backend_cann_buffer_type_context *) buft->context;
        return buft_ctx->device == dev_ctx->device;
    }
    return false;
}

static ggml_backend_buffer_type_t ggml_backend_cann_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * ctx = (ggml_backend_cann_device_context *) dev->context;
    return ggml_backend_cann_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cann_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cann_host_buffer_type();
}

/**
 * @brief Determines if a tensor operation should be offloaded to the CANN
 * backend.
 *
 * This function checks if a given tensor operation should be offloaded to the
 * CANN backend based on the operation type and the size of the tensor. It
 * returns true if the second dimension (ne[1]) of the tensor is greater than or
 * equal to the minimum batch size and the operation is not GGML_OP_GET_ROWS.
 *
 * @param backend Pointer to the CANN backend.
 * @param op Pointer to the tensor operation to check.
 * @return bool Returns true if the operation should be offloaded, otherwise
 * false.
 */
static bool ggml_backend_cann_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *)dev->context;

    return op->ne[1] >= dev_ctx->op_offload_min_batch_size && op->op != GGML_OP_GET_ROWS;
}

/**
 * @brief Creates a new event for the CANN backend device.
 *
 * This function initializes a new event for the CANN backend by setting the
 * device and creating an ACL runtime event. The created event is then wrapped
 * in a ggml_backend_event structure and returned.
 *
 * @param backend Pointer to the CANN backend.
 * @return ggml_backend_event_t Returns a pointer to the new event structure.
 */
static ggml_backend_event_t ggml_backend_cann_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_cann_device_context * dev_ctx = (ggml_backend_cann_device_context *) dev->context;

    ggml_cann_set_device(dev_ctx->device);

    aclrtEvent event;
    ACL_CHECK(aclrtCreateEvent(&event));

    return new ggml_backend_event{
        /* .device = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), dev_ctx->device),
        /* .context = */ event,
    };
}

/**
 * @brief Frees a CANN backend event.
 *
 * This function destroys the ACL runtime event associated with the given CANN
 * backend event and then deletes the event structure itself.
 *
 * @param event Pointer to the event structure to be freed.
 */
static void ggml_backend_cann_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtDestroyEvent((aclrtEvent) event->context));

    delete event;
    GGML_UNUSED(dev);
}

/**
 * @brief Synchronizes the given event on the CANN backend.
 *
 * This function waits for the specified event to complete on the ACL runtime.
 *
 * @param event Pointer to the event structure to be synchronized.
 */
static void ggml_backend_cann_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ACL_CHECK(aclrtSynchronizeEvent((aclrtEvent) event->context));

    GGML_UNUSED(dev);
}

static const ggml_backend_device_i ggml_backend_cann_device_interface = {
    /* .get_name                = */ ggml_backend_cann_device_get_name,
    /* .get_description         = */ ggml_backend_cann_device_get_description,
    /* .get_memory              = */ ggml_backend_cann_device_get_memory,
    /* .get_type                = */ ggml_backend_cann_device_get_type,
    /* .get_props               = */ ggml_backend_cann_device_get_props,
    /* .init_backend            = */ ggml_backend_cann_device_init,  // called for every card
    /* .get_buffer_type         = */ ggml_backend_cann_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cann_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,  // not supported for CANN
    /* .supports_op             = */ ggml_backend_cann_supports_op,
    /* .supports_buft           = */ ggml_backend_cann_supports_buft,
    /* .offload_op              = */ ggml_backend_cann_offload_op,
    /* .event_new               = */ ggml_backend_cann_device_event_new,
    /* .event_free              = */ ggml_backend_cann_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cann_device_event_synchronize,
};

// backend reg
struct ggml_backend_cann_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cann_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CANN_NAME;
}

static size_t ggml_backend_cann_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cann_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cann_reg_context * ctx = (ggml_backend_cann_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static void * ggml_backend_cann_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    // reserved for future use
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cann_reg_interface = {
    /* .get_name          = */ ggml_backend_cann_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cann_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cann_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cann_reg_get_proc_address,
};

// backend registry, called only once for cann backend
ggml_backend_reg_t ggml_backend_cann_reg() {
    static ggml_backend_reg reg;
    static bool             initialized = false;

    {
        static std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            aclInit(nullptr);
            ggml_backend_cann_reg_context * ctx = new ggml_backend_cann_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            for (int i = 0; i < ggml_cann_info().device_count; i++) {
                ggml_backend_cann_device_context * dev_ctx = new ggml_backend_cann_device_context();
                dev_ctx->description                       = aclrtGetSocName();
                dev_ctx->device                            = i;
                dev_ctx->name                              = GGML_CANN_NAME + std::to_string(i);
                dev_ctx->op_offload_min_batch_size         = min_batch_size;
                ggml_cann_set_device(i);
                ggml_backend_dev_t dev = new ggml_backend_device{ /* .iface   = */ ggml_backend_cann_device_interface,
                                                                  /* .reg     = */ &reg,
                                                                  /* .context = */ dev_ctx };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{ /* .api_version = */ GGML_BACKEND_API_VERSION,
                                    /* .iface       = */ ggml_backend_cann_reg_interface,
                                    /* .context     = */ ctx };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cann_init(int32_t device) {
    aclInit(nullptr);
    if (device < 0 || device >= ggml_backend_cann_get_device_count()) {
        GGML_LOG_ERROR("%s: error: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cann_context * ctx = new ggml_backend_cann_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return nullptr;
    }
    ggml_cann_set_device(ctx->device);
    ggml_backend_t cann_backend =
        new ggml_backend{ /* .guid      = */ ggml_backend_cann_guid(),
                          /* .interface = */ ggml_backend_cann_interface,
                          /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_cann_reg(), device),
                          /* .context   = */ ctx };

    return cann_backend;
}

bool ggml_backend_is_cann(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cann_guid());
}

int32_t ggml_backend_cann_get_device_count() {
    return ggml_cann_info().device_count;
}

void ggml_backend_cann_get_device_description(int32_t device, char * description, size_t description_size) {
    ggml_cann_set_device(device);
    const char * soc_name = aclrtGetSocName();
    snprintf(description, description_size, "%s", soc_name);
}

void ggml_backend_cann_get_device_memory(int32_t device, size_t * free, size_t * total) {
    ggml_cann_set_device(device);
    ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

GGML_BACKEND_DL_IMPL(ggml_backend_cann_reg)
