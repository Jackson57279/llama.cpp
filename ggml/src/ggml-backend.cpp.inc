// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h.inc"
#include "ggml-backend-impl-defs.inc"
#include "ggml-alloc.h.inc"
#include "ggml-impl-defs.inc"

#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif


// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_ASSERT(buft);
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (!ggml_backend_buffer_is_meta(buffer) && buffer->size == 0) {
        return NULL;
    }

    // FIXME JG: a multi_buffer has a non-zero size, according to the above comment get_base is not optional,
    //     I don't know whether the above comment is correct
    if (!buffer->iface.get_base) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set_async(backend, tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.set_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");

    if (n_copies <= 1 || backend->iface.set_tensor_2d_async == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get_async(backend, tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    backend->iface.get_tensor_2d_async(backend, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_set_2d(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_set(tensor, (const char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_get_2d(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size,
            size_t n_copies, size_t stride_tensor, size_t stride_data) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    GGML_ASSERT(buf != NULL && "tensor buffer not set");

    if (n_copies <= 1 || buf->iface.set_tensor_2d == NULL) {
        for (size_t i = 0; i < n_copies; i++) {
            ggml_backend_tensor_get(tensor, (char *) data + i*stride_data, offset + i*stride_tensor, size);
        }
        return;
    }
    if (size == 0) {
        return;
    }

    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + (n_copies-1)*stride_tensor + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor_2d(buf, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif // NDEBUG
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(device);
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

#define ggml_is_view_op(op) ggml_is_view_op_rust((int) (op))

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;

    // used for debugging graph reallocations [GGML_SCHED_DEBUG_REALLOC]
    // ref: https://github.com/ggml-org/llama.cpp/pull/17617
    int debug_realloc;
    int debug_graph_size;
    int debug_prev_graph_size;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d,c=%d:", i, ggml_op_name(node->op), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)], node->flags & GGML_TENSOR_FLAG_COMPUTE ? 1 : 0);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    graph->uid = ggml_graph_next_uid();

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            ggml_set_input(tensor_copy);
                            ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;

    // remember the actual graph_size for performing reallocation checks later [GGML_SCHED_DEBUG_REALLOC]
    sched->debug_prev_graph_size = sched->debug_graph_size;
    sched->debug_graph_size = graph_size;

    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }

    // set ids for all splits
    for (int i = 0; i < sched->n_splits; ++i) {
        sched->splits[i].graph.uid = ggml_graph_next_uid();
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif

        if (sched->debug_realloc > 0) {
            // we are interested only in situations where the graph was reallocated even though its size remained the same [GGML_SCHED_DEBUG_REALLOC]
            // example: https://github.com/ggml-org/llama.cpp/pull/17143
            const bool unexpected = !backend_ids_changed && sched->debug_prev_graph_size == sched->debug_graph_size;

            if (unexpected || sched->debug_realloc > 1) {
                GGML_ABORT("%s: unexpected graph reallocation (graph size = %d, nodes = %d, leafs = %d), debug_realloc = %d\n", __func__,
                        sched->debug_graph_size, sched->graph.n_nodes, sched->graph.n_leafs, sched->debug_realloc);
            }
        }

        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }

        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // copy the input tensors to the split backend
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                ggml_backend_tensor_copy(input, input_cpy);
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                }
            }
        }

        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;

    sched->debug_realloc = 0;
#ifdef GGML_SCHED_NO_REALLOC
    sched->debug_realloc = 1;
#endif
    const char * GGML_SCHED_DEBUG_REALLOC = getenv("GGML_SCHED_DEBUG_REALLOC");
    sched->debug_realloc = GGML_SCHED_DEBUG_REALLOC ? atoi(GGML_SCHED_DEBUG_REALLOC) : sched->debug_realloc;

    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->debug_graph_size = 0;
    sched->debug_prev_graph_size = 0;

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

void ggml_backend_sched_reserve_size(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph, size_t * sizes) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    GGML_ASSERT(sizes);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    ggml_gallocr_reserve_n_size(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids, sizes);
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer) ||
        (char *) addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
        (char *) ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    dst->flags = src->flags;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor const * const * test_nodes, size_t num_test_nodes) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (num_test_nodes != 0) {
        GGML_ASSERT(test_nodes);
        // Compute the whole graph and only test the output for specific tensors
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        bool verified = false;
        for (int i = 0; i < g1->n_nodes; i++) {
            for (size_t j = 0; j < num_test_nodes; ++j) {
                if (g1->nodes[i] == test_nodes[j]) {
                    callback(i, g1->nodes[i], g2->nodes[i], user_data);
                    verified = true;
                }
            }
        }
        GGML_ASSERT(verified);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}


// ---- backend meta implementation ----

#include "ggml.h.inc"
#include "ggml-impl-defs.inc"
#include "ggml-backend.h.inc"
#include "ggml-backend-impl-defs.inc"
#include "ggml-alloc.h.inc"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) { ggml_free(ctx); }
};

struct ggml_backend_buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) { ggml_backend_buffer_free(buffer); }
};

using ggml_context_ptr        = std::unique_ptr<ggml_context, ggml_context_deleter>;
using ggml_backend_buffer_ptr = std::unique_ptr<ggml_backend_buffer, ggml_backend_buffer_deleter>;

struct ggml_backend_meta_device;
struct ggml_backend_meta_buffer_type;
struct ggml_backend_meta_buffer;
struct ggml_backend_meta;

const char * ggml_backend_meta_split_axis_name(enum ggml_backend_meta_split_axis split_axis) {
    switch (split_axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
            return "0";
        case GGML_BACKEND_SPLIT_AXIS_1:
            return "1";
        case GGML_BACKEND_SPLIT_AXIS_2:
            return "2";
        case GGML_BACKEND_SPLIT_AXIS_3:
            return "3";
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            return "MIRRORED";
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL:
            return "PARTIAL";
        case GGML_BACKEND_SPLIT_AXIS_NONE:
            return "NONE";
        case GGML_BACKEND_SPLIT_AXIS_UNKNOWN:
            return "UNKNOWN";
        default:
            GGML_ABORT("fatal error");
    }
}

//
// meta backend device
//

struct ggml_backend_meta_device_context {
    std::vector<ggml_backend_dev_t>     simple_devs;
    ggml_backend_meta_get_split_state_t get_split_state;
    void *                              get_split_state_ud;

    std::string name;
    std::string description;

    ggml_backend_meta_device_context(
            std::vector<ggml_backend_dev_t> simple_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) :
            simple_devs(std::move(simple_devs)), get_split_state(get_split_state), get_split_state_ud(get_split_state_ud) {
        name        = std::string("Meta(");
        description = std::string("Meta(");
        for (size_t i = 0; i < simple_devs.size(); i++) {
            if (i > 0) {
                name        += ",";
                description += ",";
            }
            name        += ggml_backend_dev_name       (simple_devs[i]);
            description += ggml_backend_dev_description(simple_devs[i]);
        }
        name        += ")";
        description += ")";
    }

    bool operator<(const ggml_backend_meta_device_context & other) const {
        return std::tie(simple_devs, get_split_state, get_split_state_ud)
            < std::tie(other.simple_devs, other.get_split_state, other.get_split_state_ud);
    }
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev);

static const char * ggml_backend_meta_device_get_name(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->name.c_str();
}

static const char * ggml_backend_meta_device_get_description(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return meta_dev_ctx->description.c_str();
}

static void ggml_backend_meta_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    *free  = 0;
    *total = 0;
    for (ggml_backend_dev_t dev : meta_dev_ctx->simple_devs) {
        size_t tmp_free, tmp_total;
        ggml_backend_dev_memory(dev, &tmp_free, &tmp_total);
        *free  += tmp_free;
        *total += tmp_total;
    }
}

static enum ggml_backend_dev_type ggml_backend_meta_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_META;

    GGML_UNUSED(dev);
}

static void ggml_backend_meta_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    // TODO replace placeholders
    props->name        = ggml_backend_meta_device_get_name(dev);
    props->description = ggml_backend_meta_device_get_description(dev);
    props->type        = ggml_backend_meta_device_get_type(dev);
    props->device_id   = 0;

    ggml_backend_meta_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false, // Not implemented.
        /* .buffer_from_host_ptr  = */ false, // Not implemented.
        /* .events                = */ false, // Not implemented.
    };
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_dev_props tmp_props;
        ggml_backend_dev_get_props(simple_dev, &tmp_props);
        props->caps.async                = props->caps.async                && tmp_props.caps.async;
        props->caps.host_buffer          = props->caps.host_buffer          && tmp_props.caps.host_buffer;
        props->caps.buffer_from_host_ptr = props->caps.buffer_from_host_ptr && tmp_props.caps.buffer_from_host_ptr;
        props->caps.events               = props->caps.events               && tmp_props.caps.events;
    }
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev);

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev);

static bool ggml_backend_meta_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
    return std::all_of(meta_dev_ctx->simple_devs.begin(), meta_dev_ctx->simple_devs.end(),
        [op](ggml_backend_dev_t simple_dev) { return ggml_backend_dev_supports_op(simple_dev, op); });
}

static bool ggml_backend_meta_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    ggml_backend_dev_t dev_buft = ggml_backend_buft_get_device(buft);
    if (!ggml_backend_dev_is_meta(dev_buft)) {
        return false;
    }
    const ggml_backend_meta_device_context * meta_dev_ctx      = (const ggml_backend_meta_device_context *) dev->context;
    const ggml_backend_meta_device_context * meta_buft_dev_ctx = (const ggml_backend_meta_device_context *) dev_buft->context;
    if (meta_dev_ctx->simple_devs.size() != meta_buft_dev_ctx->simple_devs.size()) {
        return false;
    }
    for (size_t i = 0; i < meta_dev_ctx->simple_devs.size(); i++) {
        if (meta_dev_ctx->simple_devs[i] != meta_buft_dev_ctx->simple_devs[i]) {
            return false;
        }
    }
    return true;
}

static const ggml_backend_device_i ggml_backend_meta_device_iface = {
    /* .get_name             = */ ggml_backend_meta_device_get_name,
    /* .get_description      = */ ggml_backend_meta_device_get_description,
    /* .get_memory           = */ ggml_backend_meta_device_get_memory,
    /* .get_type             = */ ggml_backend_meta_device_get_type,
    /* .get_props            = */ ggml_backend_meta_device_get_props,
    /* .init_backend         = */ ggml_backend_meta_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_meta_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_meta_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_meta_device_supports_op,
    /* .supports_buft        = */ ggml_backend_meta_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static bool ggml_backend_dev_is_meta(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_meta_device_iface.get_name;
}

static size_t ggml_backend_meta_dev_n_devs(ggml_backend_dev_t meta_dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    return meta_dev_ctx->simple_devs.size();
}

static ggml_backend_dev_t ggml_backend_meta_dev_simple_dev(ggml_backend_dev_t meta_dev, size_t index) {
    GGML_ASSERT(ggml_backend_dev_is_meta(meta_dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) meta_dev->context;
    GGML_ASSERT(index < meta_dev_ctx->simple_devs.size());
    return meta_dev_ctx->simple_devs[index];
}

ggml_backend_dev_t ggml_backend_meta_device(
        ggml_backend_dev_t * devs, size_t n_devs, ggml_backend_meta_get_split_state_t get_split_state, void * get_split_state_ud) {
    GGML_ASSERT(n_devs <= GGML_BACKEND_META_MAX_DEVICES);
    // TODO: this is not thread-safe - needs to be fixed
    static std::vector<std::unique_ptr<ggml_backend_meta_device_context>>         ctxs;
    static std::map<ggml_backend_meta_device_context, struct ggml_backend_device> meta_devs;

    std::vector<ggml_backend_dev_t> simple_devs;
    simple_devs.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_devs.push_back(devs[i]);
    }
    ggml_backend_meta_device_context ctx(simple_devs, get_split_state, get_split_state_ud);

    {
        auto it = meta_devs.find(ctx);
        if (it != meta_devs.end()) {
            return &it->second;
        }
    }
    ctxs.push_back(std::make_unique<ggml_backend_meta_device_context>(ctx));

    struct ggml_backend_device meta_dev = {
        /*iface  =*/ ggml_backend_meta_device_iface,
        /*reg    =*/ nullptr,
        /*ctx    =*/ ctxs.back().get(),
    };

    auto result = meta_devs.emplace(*ctxs.back(), meta_dev);
    return &result.first->second;
}

//
// meta backend buffer type
//

struct ggml_backend_meta_buffer_type_context {
    std::vector<ggml_backend_buffer_type_t> simple_bufts;

    std::string name;

    ggml_backend_meta_buffer_type_context(std::vector<ggml_backend_buffer_type_t> simple_bufts) : simple_bufts(std::move(simple_bufts)) {
        name = "Meta(";
        for (size_t i = 0; i < simple_bufts.size(); i++) {
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_buft_name(simple_bufts[i]);
        }
        name += ")";
    }

    bool operator<(const ggml_backend_meta_buffer_type_context & other) const {
        return simple_bufts < other.simple_bufts;
    }
};

static size_t ggml_backend_meta_buft_n_bufts(ggml_backend_buffer_type_t meta_buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    return meta_buft_ctx->simple_bufts.size();
}

static const char * ggml_backend_meta_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(ggml_backend_buft_is_meta(buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) buft->context;
    return meta_buft_ctx->name.c_str();
}

static ggml_backend_buffer_type_t ggml_backend_meta_buft_simple_buft(ggml_backend_buffer_type_t meta_buft, size_t index) {
    GGML_ASSERT(ggml_backend_buft_is_meta(meta_buft));
    const ggml_backend_meta_buffer_type_context * meta_buft_ctx = (const ggml_backend_meta_buffer_type_context *) meta_buft->context;
    GGML_ASSERT(index < meta_buft_ctx->simple_bufts.size());
    return meta_buft_ctx->simple_bufts[index];
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

static size_t ggml_backend_meta_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alignment = 1;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_meta_buft_simple_buft(buft, i));
        max_alignment = std::max(max_alignment, alignment);
        GGML_ASSERT(max_alignment % alignment == 0);
    }
    return max_alignment;
}

static size_t ggml_backend_meta_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_size = SIZE_MAX;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        max_size = std::min(max_size, ggml_backend_buft_get_max_size(ggml_backend_meta_buft_simple_buft(buft, i)));
    }
    return max_size;
}

static size_t ggml_backend_meta_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    size_t max_alloc_size = 0;
    for (size_t i = 0; i < n_simple_bufts; i++) {
        const size_t alloc_size = ggml_backend_buft_get_alloc_size(ggml_backend_meta_buft_simple_buft(buft, i), tensor);
        max_alloc_size = std::max(max_alloc_size, alloc_size);
    }
    return max_alloc_size;
}

static bool ggml_backend_meta_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        if (!ggml_backend_buft_is_host(ggml_backend_meta_buft_simple_buft(buft, i))) {
            return false;
        }
    }
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_meta_buffer_type_iface = {
    /* .get_name         = */ ggml_backend_meta_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_meta_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_meta_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_meta_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_meta_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_meta_buffer_type_is_host,
};

bool ggml_backend_buft_is_meta(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_meta_buffer_type_iface.get_name;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_buffer_type(ggml_backend_dev_t dev) {
    static std::map<ggml_backend_dev_t, struct ggml_backend_buffer_type> meta_bufts;
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    {
        auto it = meta_bufts.find(dev);
        if (it != meta_bufts.end()) {
            return &it->second;
        }
    }

    const size_t n_devs = ggml_backend_meta_dev_n_devs(dev);
    std::vector<ggml_backend_buffer_type_t> simple_bufts;
    simple_bufts.reserve(n_devs);
    for (size_t i = 0; i < n_devs; i++) {
        simple_bufts.push_back(ggml_backend_dev_buffer_type(ggml_backend_meta_dev_simple_dev(dev, i)));
    }
    ggml_backend_meta_buffer_type_context * buft_ctx = new ggml_backend_meta_buffer_type_context(simple_bufts);

    struct ggml_backend_buffer_type meta_buft = {
        /*iface  =*/ ggml_backend_meta_buffer_type_iface,
        /*device =*/ dev,
        /*ctx    =*/ buft_ctx,
    };
    auto result = meta_bufts.emplace(dev, meta_buft);
    return &result.first->second;
}

static ggml_backend_buffer_type_t ggml_backend_meta_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_ASSERT(ggml_backend_dev_is_meta(dev));
    const ggml_backend_meta_device_context * meta_dev_ctx = (const ggml_backend_meta_device_context *) dev->context;

    ggml_backend_buffer_type_t host_buft = nullptr;
    for (ggml_backend_dev_t simple_dev : meta_dev_ctx->simple_devs) {
        ggml_backend_buffer_type_t simple_host_buft = ggml_backend_dev_host_buffer_type(simple_dev);
        if (simple_host_buft == nullptr) {
            return nullptr;
        }
        if (host_buft == nullptr) {
            host_buft = simple_host_buft;
        } else if (host_buft != simple_host_buft) {
            // if different simple devices have different host buffer types,
            // we cannot provide a single host buffer type for the meta device
            return nullptr;
        }
    }
    return host_buft;
}

//
// meta backend buffer
//

struct ggml_backend_meta_buffer_context {
    static constexpr size_t nbtc = GGML_TENSOR_SIZE - sizeof(ggml_tensor::padding);

    std::map<std::pair<const ggml_tensor *, bool>, std::pair<ggml_backend_meta_split_state, char[nbtc]>> split_state_cache;
    std::map<          const ggml_tensor *,        std::vector<ggml_tensor *>>                           simple_tensors;

    struct buffer_config {
        ggml_context          * ctx;
        ggml_backend_buffer_t   buf;

        buffer_config(ggml_context * ctx, ggml_backend_buffer_t buf) : ctx(ctx), buf(buf) {}
    };
    std::vector<buffer_config> buf_configs;

    int debug;

    ggml_backend_meta_buffer_context() {
        const char * GGML_META_DEBUG = getenv("GGML_META_DEBUG");
        debug = GGML_META_DEBUG ? atoi(GGML_META_DEBUG) : 0;
    }
};

static void ggml_backend_meta_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    for (auto & [ctx, buf] : buf_ctx->buf_configs) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    delete buf_ctx;
}

static size_t ggml_backend_meta_buffer_n_bufs(ggml_backend_buffer_t meta_buf) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    return buf_ctx->buf_configs.size();
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_simple_buffer(ggml_backend_buffer_t meta_buf, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) meta_buf->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());
    return buf_ctx->buf_configs[index].buf;
}

static struct ggml_tensor * ggml_backend_meta_buffer_simple_tensor(const struct ggml_tensor * tensor, size_t index) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;
    GGML_ASSERT(index < buf_ctx->buf_configs.size());

    auto it = buf_ctx->simple_tensors.find(tensor);
    if (it == buf_ctx->simple_tensors.end()) {
        return nullptr;
    }
    return it->second[index];
}

static struct ggml_backend_meta_split_state ggml_backend_meta_get_split_state(const struct ggml_tensor * tensor, bool assume_sync) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) tensor->buffer->context;

    auto split_states_equal = [&](const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) -> bool {
        if (a.axis != b.axis) {
            return false;
        }
        for (size_t j = 0; j < n_bufs; j++) {
            int64_t sum_a = 0;
            for (size_t s = 0; s < a.n_segments; s++) {
                sum_a += a.ne[s*n_bufs + j];
            }
            int64_t sum_b = 0;
            for (size_t s = 0; s < b.n_segments; s++) {
                sum_b += b.ne[s*n_bufs + j];
            }
            if (sum_a != sum_b) {
                return false;
            }
        }
        return true;
    };

    auto handle_generic = [&](const std::vector<ggml_backend_meta_split_state> & src_ss, bool scalar_only) -> ggml_backend_meta_split_state {
        ggml_backend_meta_split_state ret = {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, 1};
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                continue;
            }
            if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
                ret = src_ss[i];
            } else if (!split_states_equal(src_ss[i], ret)) {
                ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
                break;
            }
        }
        if (ret.axis == GGML_BACKEND_SPLIT_AXIS_NONE) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
        }
        if (scalar_only && ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
            ret = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
        }
        GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        return ret;
    };

    // Some ops process data on a per-row bases:
    auto handle_per_row = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0);
        return src_ss[0];
    };

    // Some ops broadcast the src1 data across src0:
    auto handle_bin_bcast = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS &&
                tensor->src[1]->ne[src_ss[0].axis] == 1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        if (src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[0].axis == src_ss[1].axis ||
           (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL)))) {
            return src_ss[0]; // GGML_OP_ADD_ID
        }
        GGML_ASSERT(tensor->src[2] == nullptr || src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_concat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        const ggml_backend_meta_split_axis concat_axis = ggml_backend_meta_split_axis(ggml_get_op_params_i32(tensor, 0));
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis >= 0 && src_ss[1].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[1].axis);
            return src_ss[1];
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(concat_axis != src_ss[0].axis);
            return src_ss[0];
        }
        if (src_ss[0].axis == src_ss[1].axis && src_ss[0].axis != concat_axis) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_mul_mat = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, 1};
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[0];
            ret.axis = GGML_BACKEND_SPLIT_AXIS_0;
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1 && src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            ggml_backend_meta_split_state ret = src_ss[1];
            ret.n_segments = 1;
            return ret;
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(split_states_equal(src_ss[0], src_ss[1]));
            return {assume_sync ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_PARTIAL, {0}, 1};
        }
        GGML_ABORT("fatal error");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
    };

    auto handle_cpy = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            int64_t ne_split_src = tensor->src[0]->ne[0];
            for (int dim = 1; dim <= src_ss[0].axis; dim++) {
                ne_split_src *= tensor->src[0]->ne[dim];
            }
            int64_t ne_split_dst = 1;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                ne_split_dst *= tensor->ne[dim];
                if (ne_split_dst == ne_split_src) {
                    return {ggml_backend_meta_split_axis(dim), {0}, 1};
                }
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_reshape = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                GGML_ASSERT(!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]));
                if (src_ss[0].axis == ggml_n_dims(tensor->src[0]) - 1) {
                    return {ggml_backend_meta_split_axis(ggml_n_dims(tensor) - 1), {0}, 1};
                }
                std::vector<int64_t> base_ne_in;
                base_ne_in.reserve(GGML_MAX_DIMS - src_ss[0].axis);
                {
                    base_ne_in.push_back(1);
                    int dim = 0;
                    for (; dim <= src_ss[0].axis; dim++) {
                        base_ne_in[0] *= tensor->src[0]->ne[dim];
                    }
                    for (; dim <= GGML_MAX_DIMS; dim++) {
                        base_ne_in.push_back(base_ne_in.back() * tensor->src[0]->ne[dim]);
                    }
                }
                int64_t base_ne_out = 1;
                for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                    const int64_t base_ne_out_next = base_ne_out *= tensor->ne[dim];
                    for (const int64_t & bni : base_ne_in) {
                        if (bni == base_ne_out_next) {
                            return {ggml_backend_meta_split_axis(dim), {0}, 1};
                        }
                    }
                    if (base_ne_out_next > base_ne_in[0]) {
                        GGML_ASSERT(dim + 1 < GGML_MAX_DIMS);
                        return {ggml_backend_meta_split_axis(dim + 1), {0}, 1};
                    }
                    base_ne_out = base_ne_out_next;
                }
                GGML_ABORT("shape mismatch for %s", ggml_op_name(tensor->op));
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
            }
        }
    };

    auto handle_view = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (ggml_is_contiguous(tensor) && ggml_is_contiguous(tensor->src[0])) {
            return handle_reshape(src_ss);
        }
        const int axis = src_ss[0].axis;
        {
            bool all_strides_the_same = true;
            for (int dim = 0; dim < GGML_MAX_DIMS; dim++) {
                if (tensor->ne[dim] == 1 && tensor->src[0]->ne[dim] == 1) {
                    continue;
                }
                if (tensor->nb[dim] != tensor->src[0]->nb[dim]) {
                    all_strides_the_same = false;
                    break;
                }
            }
            if (all_strides_the_same) {
                return src_ss[0];
            }
        }
        if (!ggml_is_permuted(tensor) && !ggml_is_permuted(tensor->src[0]) && axis >= 0 && axis < GGML_MAX_DIMS-1) {
            for (int dim = 0; dim < GGML_MAX_DIMS-1; dim++) {
                if (tensor->nb[dim+1] == tensor->src[0]->nb[axis+1]) {
                    return {ggml_backend_meta_split_axis(dim), {0}, 1};
                }
            }
            GGML_ABORT("fatal error");
        }
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED || src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
            return src_ss[0];
        }
        GGML_ABORT("view of permuted tensor not implemented");
        //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
    };

    auto handle_permute = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1:
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3: {
                return {ggml_backend_meta_split_axis(tensor->op_params[src_ss[0].axis]), {0}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
            }
        }
    };

    auto handle_transpose = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        switch (src_ss[0].axis) {
            case GGML_BACKEND_SPLIT_AXIS_0:
            case GGML_BACKEND_SPLIT_AXIS_1: {
                return {ggml_backend_meta_split_axis(int(src_ss[0].axis) ^ 1), {0}, 1};
            }
            case GGML_BACKEND_SPLIT_AXIS_2:
            case GGML_BACKEND_SPLIT_AXIS_3:
            case GGML_BACKEND_SPLIT_AXIS_MIRRORED:
            case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
                return src_ss[0];
            }
            default: {
                GGML_ABORT("fatal error");
                //return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
            }
        }
    };

    auto handle_get_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0 && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        return handle_generic(src_ss, /*scalar_only =*/ true);
    };

    auto handle_set_rows = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(split_states_equal(src_ss[0], src_ss[2]));
        return src_ss[0];
    };

    auto handle_rope = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        return src_ss[0];
    };

    auto handle_pad = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis >= 0 && src_ss[0].axis < GGML_MAX_DIMS) {
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 0] == 0);
            GGML_ASSERT(tensor->op_params[2*src_ss[0].axis + 1] == 0);
        }
        return src_ss[0];
    };

    auto handle_flash_attn_ext = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        GGML_ASSERT(                             src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(                             src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_2);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        GGML_ASSERT(tensor->src[4] == nullptr || src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_0);
        return {GGML_BACKEND_SPLIT_AXIS_1, {0}, 1};
    };

    auto handle_ssm_conv = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == src_ss[1].axis) {
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_0) {
                return {GGML_BACKEND_SPLIT_AXIS_1, {0}, 1};
            }
            if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1) {
                return {GGML_BACKEND_SPLIT_AXIS_0, {0}, 1};
            }
        }
        return handle_generic(src_ss, /*scalar_only =*/ false);
    };

    auto handle_gated_delta_net = [&](const std::vector<ggml_backend_meta_split_state> & src_ss) -> ggml_backend_meta_split_state {
        if (src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED &&
            src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED && src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
            return src_ss[0];
        }
        GGML_ASSERT(src_ss[0].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[1].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[2].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[3].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[4].axis == GGML_BACKEND_SPLIT_AXIS_1);
        GGML_ASSERT(src_ss[5].axis == GGML_BACKEND_SPLIT_AXIS_2);
        return {GGML_BACKEND_SPLIT_AXIS_0, {0}, 1};
    };

    auto calculate_split_state = [&]() -> ggml_backend_meta_split_state {
        if (ggml_nelements(tensor) == 0) {
            return {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
        }
        if (ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && tensor->view_src == nullptr) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer));
            const ggml_backend_meta_device_context * dev_ctx = (const ggml_backend_meta_device_context *) dev->context;
            ggml_backend_meta_split_state ret = dev_ctx->get_split_state(tensor, dev_ctx->get_split_state_ud);
            if (ret.axis >= 0 && ret.axis <= GGML_MAX_DIMS) {
                const int64_t granularity = ret.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor->type) : 1;
                int64_t ne_sum = 0;
                for (size_t sj = 0; sj < ret.n_segments*n_bufs; sj++) {
                    GGML_ASSERT(ret.ne[sj] % granularity == 0);
                    ne_sum += ret.ne[sj];
                }
                GGML_ASSERT(ne_sum == tensor->ne[ret.axis]);
            }
            return ret;
        }

        std::vector<ggml_backend_meta_split_state> src_ss(GGML_MAX_SRC, {GGML_BACKEND_SPLIT_AXIS_NONE, {0}, 1});
        for (size_t i = 0; i < GGML_MAX_SRC; i++) {
            if (tensor->src[i] == nullptr || tensor->src[i] == tensor) {
                src_ss[i] = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
                continue;
            }
            src_ss[i] = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
            GGML_ASSERT(src_ss[i].axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
        }

        ggml_backend_meta_split_state split_state;
        switch (tensor->op) {
            case GGML_OP_NONE: {
                split_state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, 1};
            } break;
            case GGML_OP_DUP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ADD:
            case GGML_OP_ADD_ID: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_ADD1:
            case GGML_OP_ACC: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV: {
                split_state = handle_bin_bcast(src_ss);
            } break;
            case GGML_OP_SQR:
            case GGML_OP_SQRT:
            case GGML_OP_LOG:
            case GGML_OP_SIN:
            case GGML_OP_COS: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SUM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SUM_ROWS:
            case GGML_OP_CUMSUM:
            case GGML_OP_MEAN:
            case GGML_OP_ARGMAX:
            case GGML_OP_COUNT_EQUAL: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_REPEAT:
            case GGML_OP_REPEAT_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONCAT: {
                split_state = handle_concat(src_ss);
            } break;
            case GGML_OP_SILU_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_RMS_NORM_BACK:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID: {
                split_state = handle_mul_mat(src_ss);
            } break;
            case GGML_OP_OUT_PROD: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_SET: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CPY: {
                split_state = handle_cpy(src_ss);
            } break;
            case GGML_OP_CONT:
            case GGML_OP_RESHAPE: {
                split_state = handle_reshape(src_ss);
            } break;
            case GGML_OP_VIEW: {
                split_state = handle_view(src_ss);
            } break;
            case GGML_OP_PERMUTE: {
                split_state = handle_permute(src_ss);
            } break;
            case GGML_OP_TRANSPOSE: {
                split_state = handle_transpose(src_ss);
            } break;
            case GGML_OP_GET_ROWS: {
                split_state = handle_get_rows(src_ss);
            } break;
            case GGML_OP_GET_ROWS_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SET_ROWS: {
                split_state = handle_set_rows(src_ss);
            } break;
            case GGML_OP_DIAG:
            case GGML_OP_DIAG_MASK_INF:
            case GGML_OP_DIAG_MASK_ZERO: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SOFT_MAX:
            case GGML_OP_SOFT_MAX_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_ROPE: {
                split_state = handle_rope(src_ss);
            } break;
            case GGML_OP_ROPE_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CLAMP: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_CONV_TRANSPOSE_1D:
            case GGML_OP_IM2COL:
            case GGML_OP_IM2COL_BACK:
            case GGML_OP_IM2COL_3D:
            case GGML_OP_CONV_2D:
            case GGML_OP_CONV_3D:
            case GGML_OP_CONV_2D_DW:
            case GGML_OP_CONV_TRANSPOSE_2D:
            case GGML_OP_POOL_1D:
            case GGML_OP_POOL_2D:
            case GGML_OP_POOL_2D_BACK:
            case GGML_OP_UPSCALE: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_PAD: {
                split_state = handle_pad(src_ss);
            } break;
            case GGML_OP_PAD_REFLECT_1D:
            case GGML_OP_ROLL:
            case GGML_OP_ARANGE:
            case GGML_OP_TIMESTEP_EMBEDDING: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_ARGSORT:
            case GGML_OP_TOP_K: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_LEAKY_RELU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_FILL: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_FLASH_ATTN_EXT: {
                split_state = handle_flash_attn_ext(src_ss);
            } break;
            case GGML_OP_FLASH_ATTN_BACK: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_SSM_CONV: {
                split_state = handle_ssm_conv(src_ss);
            } break;
            case GGML_OP_SSM_SCAN:
            case GGML_OP_WIN_PART:
            case GGML_OP_WIN_UNPART:
            case GGML_OP_GET_REL_POS:
            case GGML_OP_ADD_REL_POS:
            case GGML_OP_RWKV_WKV6:
            case GGML_OP_GATED_LINEAR_ATTN:
            case GGML_OP_RWKV_WKV7:
            case GGML_OP_SOLVE_TRI: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_GATED_DELTA_NET: {
                split_state = handle_gated_delta_net(src_ss);
            } break;
            case GGML_OP_UNARY: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            case GGML_OP_MAP_CUSTOM1:
            case GGML_OP_MAP_CUSTOM2:
            case GGML_OP_MAP_CUSTOM3:
            case GGML_OP_CUSTOM: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ true);
            } break;
            case GGML_OP_CROSS_ENTROPY_LOSS:
            case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
                split_state = handle_per_row(src_ss);
            } break;
            case GGML_OP_OPT_STEP_ADAMW:
            case GGML_OP_OPT_STEP_SGD:
            case GGML_OP_GLU: {
                split_state = handle_generic(src_ss, /*scalar_only =*/ false);
            } break;
            default: {
                GGML_ABORT("ggml op not implemented: %s", ggml_op_name(tensor->op));
                split_state = {GGML_BACKEND_SPLIT_AXIS_UNKNOWN, {0}, 1};
            } break;
        }
        if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
            bool first_src_split_by_axis = true;
            const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(tensor->buffer);

            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr || src_ss[i].axis < 0 || src_ss[i].axis >= GGML_MAX_DIMS) {
                    continue;
                }
                if (first_src_split_by_axis) {
                    for (size_t j = 0; j < n_bufs; j++) {
                        // Take over ratio from src:
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[s*n_bufs + j] = 0;
                        }
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            split_state.ne[j] += src_ss[i].ne[s*n_bufs + j];
                        }
                        split_state.ne[j] *= tensor->ne[split_state.axis];
                        if (split_state.ne[j] != 0 || tensor->src[i]->ne[src_ss[i].axis] != 0) {
                            GGML_ASSERT(split_state.ne[j] % tensor->src[i]->ne[src_ss[i].axis] == 0);
                            split_state.ne[j] /= tensor->src[i]->ne[src_ss[i].axis];
                        }
                    }
                } else {
                    for (size_t j = 0; j < n_bufs; j++) {
                        int64_t sum = 0;
                        for (size_t s = 0; s < src_ss[i].n_segments; s++) {
                            sum += src_ss[i].ne[s*n_bufs + j];
                        }
                        // Assert that ratio is consistent:
                        GGML_ASSERT(split_state.ne[j] * tensor->src[i]->ne[src_ss[i].axis]
                                               == sum * tensor->ne[split_state.axis]);
                    }
                }
                first_src_split_by_axis = false;
            }
            GGML_ASSERT(!first_src_split_by_axis);
        }
        return split_state;
    };

    const std::pair key = std::make_pair(tensor, assume_sync);
    auto it = buf_ctx->split_state_cache.find(key);
    if (it != buf_ctx->split_state_cache.end() && memcmp(it->second.second, (const char *) tensor, sizeof(it->second.second)) != 0) {
        buf_ctx->split_state_cache.clear();
        it = buf_ctx->split_state_cache.end();
    }

    if (it == buf_ctx->split_state_cache.end()) {
        buf_ctx->split_state_cache[key].first = calculate_split_state();
        memcpy(buf_ctx->split_state_cache[key].second, tensor, sizeof(buf_ctx->split_state_cache[key].second));
        if (buf_ctx->debug > 0) {
            std::string srcs_info;
            for (size_t i = 0; i < GGML_MAX_SRC; i++) {
                if (tensor->src[i] == nullptr) {
                    continue;
                }
                if (!srcs_info.empty()) {
                    srcs_info += ", ";
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor->src[0], true);
                const char * axis_name = ggml_backend_meta_split_axis_name(split_state.axis);
                std::string ne_info;
                for (size_t j = 0; j < n_bufs; j++) {
                    if (!ne_info.empty()) {
                        ne_info += ", ";
                    }
                    ne_info += std::to_string(split_state.ne[j]);
                }
                srcs_info += std::string(tensor->src[i]->name) + "[" + ggml_op_name(tensor->src[i]->op) + ", " + axis_name + ", {" + ne_info + "}]";
            }
            std::string ne_info;
            for (size_t j = 0; j < n_bufs; j++) {
                if (!ne_info.empty()) {
                    ne_info += ", ";
                }
                ne_info += std::to_string(buf_ctx->split_state_cache[key].first.ne[j]);
            }
            GGML_LOG_DEBUG("SPLIT_STATE: {%s} -> %s[%s, %s, {%s}]\n", srcs_info.c_str(), tensor->name, ggml_op_name(tensor->op),
                ggml_backend_meta_split_axis_name(buf_ctx->split_state_cache[key].first.axis), ne_info.c_str());
        }
    }

    ggml_backend_meta_split_state ret = buf_ctx->split_state_cache[key].first;
    GGML_ASSERT(ret.axis != GGML_BACKEND_SPLIT_AXIS_NONE);
#ifndef NDEBUG
    if (ret.axis >= 0 && ret.axis < GGML_MAX_DIMS) {
        int64_t ne_ret = 0;
        for (size_t sj = 0; sj < ret.n_segments*n_bufs; sj++) {
            ne_ret += ret.ne[sj];
        }
        assert(ne_ret == tensor->ne[int(ret.axis)]);
    }
#endif // NDEBUG
    return ret;
}

static void * ggml_backend_meta_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000000000000000; // FIXME
}

static enum ggml_status ggml_backend_meta_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(ggml_backend_buffer_is_meta(buffer));
    ggml_backend_meta_buffer_context * buf_ctx = (ggml_backend_meta_buffer_context *) buffer->context;
    const size_t n_simple_bufs = ggml_backend_meta_buffer_n_bufs(buffer);

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ true);
    GGML_ASSERT(ggml_nelements(tensor) == 0 || split_state.axis != GGML_BACKEND_SPLIT_AXIS_UNKNOWN);
    GGML_ASSERT(split_state.n_segments <= 16);

    int split_dim = split_state.axis;
    int64_t ne[GGML_MAX_DIMS];
    size_t  nb[GGML_MAX_DIMS];
    for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
        ne[k] = tensor->ne[k];
        nb[k] = tensor->nb[k];
    }

    std::vector<ggml_tensor *> simple_tensors;
    simple_tensors.reserve(n_simple_bufs);
    for (size_t j = 0; j < n_simple_bufs; j++) {
        ggml_context          * simple_ctx = buf_ctx->buf_configs[j].ctx;
        ggml_backend_buffer_t   simple_buf = buf_ctx->buf_configs[j].buf;

        if (split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
            // TODO: the following assert fails for llama-parallel even though the results are correct:
            // GGML_ASSERT(ggml_is_contiguously_allocated(tensor));
            ne[split_dim] = 0;
            for (size_t s = 0; s < split_state.n_segments; s++) {
                ne[split_dim] += split_state.ne[s*n_simple_bufs + j];
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (tensor->nb[i] > tensor->nb[split_dim]) {
                    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }

        ggml_tensor * t_ij = ggml_new_tensor(simple_ctx, tensor->type, GGML_MAX_DIMS, ne);
        t_ij->op = tensor->op;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            t_ij->nb[i] = nb[i];
        }
        t_ij->flags = tensor->flags;
        memcpy(t_ij->op_params, tensor->op_params, sizeof(tensor->op_params));
        ggml_set_name(t_ij, tensor->name);
        t_ij->buffer = simple_buf;
        t_ij->view_src = tensor->view_src;
        t_ij->view_offs = tensor->view_offs;
        if (t_ij->view_src != nullptr && ggml_backend_buffer_is_meta(t_ij->view_src->buffer)) {
            t_ij->view_src = ggml_backend_meta_buffer_simple_tensor(tensor->view_src, j);
            if (t_ij->view_offs > 0 && split_dim >= 0 && split_dim < GGML_MAX_DIMS) {
                GGML_ASSERT(tensor->ne[split_dim] != 0);
                const int split_dim_view_src = ggml_backend_meta_get_split_state(tensor->view_src, /*assume_sync =*/ true).axis;
                GGML_ASSERT(split_dim_view_src >= 0 && split_dim_view_src < GGML_MAX_DIMS);

                // The offset can be internal to the data split, in those cases the view offset should not be scaled.
                // If however, the offset is larger than the data split then it needs to be scaled proportionally.
                bool split_internal_offset = t_ij->view_offs <= tensor->view_src->nb[split_dim_view_src];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    const size_t dim_size = tensor->ne[i] * tensor->nb[i];
                    if (tensor->view_offs <= dim_size && dim_size < tensor->nb[split_dim]) {
                        split_internal_offset = true;
                        break;
                    }
                }
                if (!split_internal_offset) {
                    t_ij->view_offs = t_ij->view_offs * ne[split_dim]/tensor->ne[split_dim];
                }
            }
        }
        if (t_ij->view_src != nullptr) {
            t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
        } else if (simple_buf != nullptr) {
            t_ij->data = (char *) ggml_backend_buffer_get_base(simple_buf)
                + size_t(tensor->data) - size_t(ggml_backend_buffer_get_base(buffer));
        }
        t_ij->extra = tensor->extra;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            t_ij->src[i] = tensor->src[i];
            if (tensor->src[i] == tensor) {
                t_ij->src[i] = t_ij;
            } else if (t_ij->src[i] != nullptr && ggml_backend_buffer_is_meta(t_ij->src[i]->buffer)) {
                t_ij->src[i] = ggml_backend_meta_buffer_simple_tensor(tensor->src[i], j);
            }
        }

        simple_tensors.push_back(t_ij);
    }

    // If one of the sources has a zero-sized slice, disable the computation:
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i] == nullptr || !ggml_backend_buffer_is_meta(tensor->src[i]->buffer)) {
            continue;
        }

        const ggml_backend_meta_split_state split_state_src = ggml_backend_meta_get_split_state(tensor->src[i], /*assume_sync =*/ true);
        if (split_state_src.axis < 0 || split_state_src.axis >= GGML_MAX_DIMS) {
            continue;
        }
        for (size_t j = 0; j < n_simple_bufs; j++) {
            int64_t ne_sum = 0;
            for (size_t s = 0; s < split_state_src.n_segments; s++) {
                ne_sum += split_state_src.ne[s*n_simple_bufs + j];
            }
            if (ne_sum == 0) {
                simple_tensors[j]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
            }
        }
    }

    buf_ctx->simple_tensors[tensor] = simple_tensors;

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_meta_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    if (split_state.n_segments != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t r_start = offset / row_stride;
            const int64_t r_count = size   / row_stride;
            GGML_ASSERT(r_start + r_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                    const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                    ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                        simple_offsets[j] + r_start * simple_tensor->nb[1], nbytes,
                        r_count, simple_tensor->nb[1], tensor->nb[1]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
            GGML_ASSERT(offset_data*r_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t r_start = offset / row_stride;
        const int64_t r_count = size   / row_stride;
        GGML_ASSERT(r_start + r_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_data,
                    simple_offsets[j] + r_start * simple_tensor->nb[2], nbytes,
                    r_count, simple_tensor->nb[2], tensor->nb[2]);
                offset_data       += nbytes;
                simple_offsets[j] += nbytes;
            }
        }
        GGML_ASSERT(offset_data*r_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_set_2d(simple_tensor, (const char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, data, offset, size);
            }
        } break;
        case GGML_BACKEND_SPLIT_AXIS_PARTIAL: {
            GGML_ASSERT(tensor->type == GGML_TYPE_F32);
            const int64_t ne = ggml_nelements(tensor);
            std::vector<float> tmp;
            tmp.reserve(ne);
            for (int64_t i = 0; i < ne; i++) {
                tmp.push_back(((const float *) data)[i] / n_bufs);
            }
            for (size_t j = 0; j < n_bufs; j++) {
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                ggml_backend_tensor_set(simple_tensor, tmp.data(), offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_bufs = ggml_backend_meta_buffer_n_bufs(buffer);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);

    if (split_state.n_segments != 1) {
        GGML_ASSERT(split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS);
        GGML_ASSERT(tensor->ne[3] == 1);

        size_t offset_data = 0;
        std::vector<size_t> simple_offsets(n_bufs, 0);
        if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_0) {
            GGML_ASSERT(tensor->ne[2] == 1);

            const size_t row_stride = tensor->nb[1];
            GGML_ASSERT(offset % row_stride == 0);
            GGML_ASSERT(size   % row_stride == 0);
            const int64_t r_start = offset / row_stride;
            const int64_t r_count = size   / row_stride;
            GGML_ASSERT(r_start + r_count <= tensor->ne[1]);

            const int64_t blck_size = ggml_blck_size(tensor->type);
            for (size_t s = 0; s < split_state.n_segments; s++) {
                for (size_t j = 0; j < n_bufs; j++) {
                    const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                    GGML_ASSERT(split_state.ne[s*n_bufs + j] % blck_size == 0);
                    const size_t nbytes = split_state.ne[s*n_bufs + j]/blck_size * tensor->nb[0];
                    ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                        simple_offsets[j] + r_start * simple_tensor->nb[1], nbytes,
                        r_count, simple_tensor->nb[1], tensor->nb[1]);
                    offset_data       += nbytes;
                    simple_offsets[j] += nbytes;
                }
            }
            GGML_ASSERT(offset_data*r_count == size);
            return;
        }
        GGML_ASSERT(split_state.axis == GGML_BACKEND_SPLIT_AXIS_1);

        const size_t row_stride = tensor->nb[2];
        GGML_ASSERT(offset % row_stride == 0);
        GGML_ASSERT(size   % row_stride == 0);
        const int64_t r_start = offset / row_stride;
        const int64_t r_count = size   / row_stride;
        GGML_ASSERT(r_start + r_count <= tensor->ne[2]);

        for (size_t s = 0; s < split_state.n_segments; s++) {
            for (size_t j = 0; j < n_bufs; j++) {
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t nbytes = split_state.ne[s*n_bufs + j] * tensor->nb[1];
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_data,
                    simple_offsets[j] + r_start * simple_tensor->nb[2], nbytes,
                    r_count, simple_tensor->nb[2], tensor->nb[2]);
                offset_data       += nbytes;
                simple_offsets[j] += nbytes;
            }
        }
        GGML_ASSERT(offset_data*r_count == size);
        return;
    }

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_bufs; j++){
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                const size_t simple_offset = i_start * chunk_size_j;
                ggml_backend_tensor_get_2d(simple_tensor, (char *) data + offset_j, simple_offset, chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get(simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_clear(ggml_backend_meta_buffer_simple_buffer(buffer, i), value);
    }
}

static void ggml_backend_meta_buffer_reset(ggml_backend_buffer_t buffer) {
    const size_t n_buffers = ggml_backend_meta_buffer_n_bufs(buffer);
    for (size_t i = 0; i < n_buffers; i++) {
        ggml_backend_buffer_reset(ggml_backend_meta_buffer_simple_buffer(buffer, i));
    }
}

static const ggml_backend_buffer_i ggml_backend_meta_buffer_iface = {
    /* .free_buffer     = */ ggml_backend_meta_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_meta_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_meta_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr, // TODO implement
    /* .set_tensor      = */ ggml_backend_meta_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_meta_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_meta_buffer_clear,
    /* .reset           = */ ggml_backend_meta_buffer_reset,
};

bool ggml_backend_buffer_is_meta(ggml_backend_buffer_t buf) {
    return buf != nullptr && buf->iface.free_buffer == ggml_backend_meta_buffer_iface.free_buffer;
}

static ggml_backend_buffer_t ggml_backend_meta_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * buf_ctx = new ggml_backend_meta_buffer_context();
    size_t max_size = 0;
    buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        ggml_backend_buffer_t simple_buf = ggml_backend_buft_alloc_buffer(ggml_backend_meta_buft_simple_buft(buft, i), size);
        max_size = std::max(max_size, ggml_backend_buffer_get_size(simple_buf));
        buf_ctx->buf_configs.emplace_back(ggml_init(params), simple_buf);
    }

    return ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, buf_ctx, max_size);
}

struct ggml_backend_buffer * ggml_backend_meta_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft) {
    const size_t n_simple_bufts = ggml_backend_meta_buft_n_bufts(buft);

    ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024*1024, // FIXME
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_backend_meta_buffer_context * meta_buf_ctx = new ggml_backend_meta_buffer_context();
    meta_buf_ctx->buf_configs.reserve(n_simple_bufts);
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs.emplace_back(ggml_init(params), nullptr);
    }

    ggml_backend_buffer_t meta_buf = ggml_backend_buffer_init(buft, ggml_backend_meta_buffer_iface, meta_buf_ctx, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = meta_buf;
        ggml_backend_meta_buffer_init_tensor(meta_buf, t);
        t->data = (void *) 0x2000000000000000; // FIXME
    }
    for (size_t i = 0; i < n_simple_bufts; i++) {
        meta_buf_ctx->buf_configs[i].buf = ggml_backend_alloc_ctx_tensors_from_buft(
            meta_buf_ctx->buf_configs[i].ctx, ggml_backend_meta_buft_simple_buft(buft, i));
        meta_buf->size = std::max(meta_buf->size, ggml_backend_buffer_get_size(meta_buf_ctx->buf_configs[i].buf));
    }
    return meta_buf;
}

//
// meta backend
//

static ggml_guid_t ggml_backend_meta_guid() {
    static ggml_guid guid = {0xf1, 0x0e, 0x34, 0xcf, 0x9c, 0x6f, 0x43, 0xcb, 0x96, 0x92, 0xbe, 0x8e, 0xbb, 0x71, 0x3f, 0xda};
    return &guid;
}

struct ggml_backend_meta_context {
    struct cgraph_config {
        ggml_cgraph * cgraph_main = nullptr;
        int           offset      = 0; // Node offset vs. original graph

        std::vector<ggml_cgraph *> cgraphs_aux;
    };
    struct backend_config {
        ggml_backend_t backend;

        std::vector<cgraph_config>           cgraphs;
        std::vector<ggml_tensor *>           nodes;
        std::vector<ggml_backend_buffer_ptr> bufs;

        backend_config(ggml_backend_t backend, const size_t n_reduce_steps) : backend(backend) {
            bufs.resize(n_reduce_steps);
        }
    };
    std::string                 name;
    std::vector<backend_config> backend_configs;
    ggml_context_ptr            ctx;
    std::vector<ggml_cgraph *>  cgraphs_aux;
    std::vector<ggml_tensor *>  nodes_aux;
    size_t                      n_reduce_steps;
    int                         max_nnodes    = 0;
    size_t                      max_tmp_size  = 0;
    size_t                      max_subgraphs = 0;
    size_t                      n_subgraphs   = 0;
    uint64_t                    uid           = 0;

    void *                               comm_ctx       = nullptr;
    ggml_backend_comm_allreduce_tensor_t comm_allreduce = nullptr;

    ggml_backend_meta_context(ggml_backend_dev_t meta_dev, const char * params) {
        const size_t n_devs = ggml_backend_meta_dev_n_devs(meta_dev);
        n_reduce_steps = std::ceil(std::log2(n_devs));
        name = "Meta(";
        std::vector<ggml_backend_t> simple_backends;
        backend_configs.reserve(n_devs);
        simple_backends.reserve(n_devs);
        for (size_t i = 0; i < n_devs; i++) {
            ggml_backend_dev_t simple_dev = ggml_backend_meta_dev_simple_dev(meta_dev, i);
            if (i > 0) {
                name += ",";
            }
            name += ggml_backend_dev_name(simple_dev);
            simple_backends.push_back(ggml_backend_dev_init(simple_dev, params));
            backend_configs.emplace_back(simple_backends.back(), n_reduce_steps);
        }
        name += ")";

        if (n_devs > 1) {
            ggml_backend_comm_init_t comm_init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_init");
            if (comm_init != nullptr) {
                comm_ctx = comm_init(simple_backends.data(), simple_backends.size());
            }
        }
        if (comm_ctx != nullptr) {
            comm_allreduce = (ggml_backend_comm_allreduce_tensor_t)
                ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(
                    ggml_backend_get_device(simple_backends[0])), "ggml_backend_comm_allreduce_tensor");
            GGML_ASSERT(comm_allreduce != nullptr);
        }
    }

    ~ggml_backend_meta_context() {
        if (comm_ctx != nullptr) {
            ggml_backend_comm_free_t comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_configs[0].backend)), "ggml_backend_comm_free");
            GGML_ASSERT(comm_free != nullptr);
            comm_free(comm_ctx);
        }
        for (auto & bc : backend_configs) {
            ggml_backend_free(bc.backend);
        }
    }
};

static const char * ggml_backend_meta_get_name(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) backend->context;
    return backend_ctx->name.c_str();
}

static void ggml_backend_meta_free(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_meta(backend));
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;
    delete backend_ctx;
    delete backend;
}

static void ggml_backend_meta_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, (const char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            for (size_t j = 0; j < n_backends; j++) {
                ggml_backend_tensor_set_async(
                    ggml_backend_meta_simple_backend(backend, j), ggml_backend_meta_buffer_simple_tensor(tensor, j), data, offset, size);
            }
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(ggml_is_contiguous(tensor));

    const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(tensor, /*assume_sync =*/ false);
    GGML_ASSERT(split_state.n_segments == 1);

    switch (split_state.axis) {
        case GGML_BACKEND_SPLIT_AXIS_0:
        case GGML_BACKEND_SPLIT_AXIS_1:
        case GGML_BACKEND_SPLIT_AXIS_2: {
            // Exploit that tensors are contiguous to splice it with simple tensors as "chunks".
            const size_t chunk_size_full = tensor->nb[split_state.axis + 1];
            GGML_ASSERT(offset % chunk_size_full == 0);
            GGML_ASSERT(size   % chunk_size_full == 0);
            const int64_t i_start =  offset        /chunk_size_full;
            const int64_t i_stop  = (offset + size)/chunk_size_full;
            size_t offset_j = 0;
            for (size_t j = 0; j < n_backends; j++){
                ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, j);
                const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, j);
                const size_t chunk_size_j = simple_tensor->nb[split_state.axis + 1];
                ggml_backend_tensor_get_2d_async(simple_backend, simple_tensor, (char *) data + offset_j, offset, chunk_size_j,
                    i_stop - i_start, chunk_size_j, chunk_size_full);
                offset_j += chunk_size_j;
            }
            GGML_ASSERT(offset_j == chunk_size_full);
        } break;
        case GGML_BACKEND_SPLIT_AXIS_MIRRORED: {
            // TODO other simple backend may be better
            ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(backend, 0);
            const ggml_tensor * simple_tensor = ggml_backend_meta_buffer_simple_tensor(tensor, 0);
            ggml_backend_tensor_get_async(simple_backend, simple_tensor, data, offset, size);
        } break;
        default: {
            GGML_ABORT("fatal error");
        }
    }
}

static void ggml_backend_meta_synchronize(ggml_backend_t backend) {
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    for (size_t i = 0; i < n_backends; i++) {
        ggml_backend_synchronize(ggml_backend_meta_simple_backend(backend, i));
    }
}

static enum ggml_status ggml_backend_meta_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(cgraph->grads == nullptr);
    const size_t n_backends = ggml_backend_meta_n_backends(backend);
    ggml_backend_meta_context * backend_ctx = (ggml_backend_meta_context *) backend->context;

    // If the previous cgraph had a defined UID it can be used to skip rebuilding the subgraphs per simple backend.
    const bool needs_rebuild = (cgraph->uid == 0) || (cgraph->uid != backend_ctx->uid);

    bool max_nnodes_raised = false;
    if (cgraph->n_nodes > backend_ctx->max_nnodes) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            bcj.nodes.resize(cgraph->n_nodes);
            bcj.cgraphs.resize(cgraph->n_nodes);
        }
        backend_ctx->max_nnodes = cgraph->n_nodes;
        max_nnodes_raised = true;
        assert(needs_rebuild);
    }

    if (needs_rebuild) {
        size_t n_subgraphs  = 0;
        size_t max_tmp_size = 0;

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    // FIXME s_copy_main is on the CPU and its view seems to be incorrectly added to the graph nodes.
                    // For regular usage this doesn't matter since it's a noop but trying to call ggml_backend_meta_buffer_simple_tensor results in a crash.
                    bcj.nodes[i] = node;
                    continue;
                }
                bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j);
                GGML_ASSERT(bcj.nodes[i]);
            }
        }

        {
            // For MoE models it may make sense to delay the AllReduce in order to reduce I/O:
            auto get_i_delayed = [&](const int i) -> int {
                int id = i; // i_delayed
                int idr = i; // i_delayed return, last safe return value

                ggml_tensor * node = cgraph->nodes[id];
                int32_t n_used = ggml_node_get_use_count(cgraph, id);

                // Skip MIRRORED nodes that don't consume node
                auto skip_unrelated = [&]() {
                    while (id + 1 < cgraph->n_nodes) {
                        ggml_tensor * next = cgraph->nodes[id+1];
                        if (ggml_backend_meta_get_split_state(next, false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                            break;
                        }
                        bool safe = true;
                        for (int s = 0; s < GGML_MAX_SRC; s++) {
                            if (next->src[s] == nullptr) {
                                continue;
                            }
                            if (next->src[s] == node) {
                                safe = false;
                                break;
                            }
                            if (ggml_backend_meta_get_split_state(next->src[s], false).axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                                safe = false;
                                break;
                            }
                        }
                        if (!safe) {
                            break;
                        }
                        id++;
                    }
                };

                skip_unrelated();
                if (id + 1 >= cgraph->n_nodes) {
                    return idr;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_ADD_ID && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL &&
                            ggml_backend_meta_get_split_state(next->src[2], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    }
                }
                // Chain of MULs with MIRRORED src[1]
                while (true) {
                    skip_unrelated();
                    if (id + 1 >= cgraph->n_nodes) {
                        return idr;
                    }
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op == GGML_OP_MUL && next->src[0] == node &&
                            ggml_backend_meta_get_split_state(next->src[1], false).axis == GGML_BACKEND_SPLIT_AXIS_MIRRORED) {
                        node = next;
                        id++;
                        idr = id;
                        n_used = ggml_node_get_use_count(cgraph, id);
                    } else {
                        break;
                    }
                }

                if (n_used != node->ne[1] || id + 2*n_used-1 >= cgraph->n_nodes) {
                    return idr;
                }
                for (int32_t k = 0; k < n_used; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_VIEW || next->view_src != node || next->view_offs != k*node->nb[1] ||
                            next->ne[0] != node->ne[0] || next->ne[1] != node->ne[2] || next->nb[1] != node->nb[2] ||
                            ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id - (n_used-1)] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                for (int32_t k = 0; k < n_used - 2; k++) {
                    ggml_tensor * next = cgraph->nodes[id+1];
                    if (next->op != GGML_OP_ADD || next->src[0] != cgraph->nodes[id] ||
                            next->src[1] != cgraph->nodes[id - (n_used-2)] || ggml_node_get_use_count(cgraph, id+1) != 1) {
                        return idr;
                    }
                    id++;
                }
                idr = id;
                return idr;
            };

            int i_start = 0;
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (node->view_src != nullptr && node->view_src->op == GGML_OP_NONE && ggml_backend_buffer_is_host(node->view_src->buffer)) {
                    continue;
                }
                const ggml_backend_meta_split_state split_state = ggml_backend_meta_get_split_state(node, /*assume_sync =*/ false);
                if (split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
                    max_tmp_size = std::max(max_tmp_size, ggml_nbytes(node));
                }
                const bool new_subgraph = i + 1 == cgraph->n_nodes || split_state.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL;
                if (!new_subgraph) {
                    continue;
                }

                const int i_delayed = get_i_delayed(i);

                // If we can delay the AllReduce we need to consider the interaction with zero-sized tensor slices.
                // A backend with such a slice would normally have valid data after participating in the AllReduce with a node that has
                //     its compute flag disabled and thus gets its data zeroed out.
                // If the AllReduce is delayed then the nodes until that point also need to have their compute flag disabled.
                if (i_delayed > i) {
                    for (size_t j = 0; j < n_backends; j++) {
                        auto & bcj = backend_ctx->backend_configs[j];
                        if ((bcj.nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                            for (int ii = i + 1; ii <= i_delayed; ii++) {
                                bcj.nodes[ii]->flags &= ~GGML_TENSOR_FLAG_COMPUTE;
                            }
                        }
                    }
                }

                i = i_delayed;

                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    bcj.cgraphs[n_subgraphs].offset = i_start;
                }
                n_subgraphs++;
                i_start = i + 1;
            }
            GGML_ASSERT(i_start == cgraph->n_nodes);
        }

        backend_ctx->uid         = cgraph->uid;
        backend_ctx->n_subgraphs = n_subgraphs;

        if (max_tmp_size > backend_ctx->max_tmp_size) {
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < backend_ctx->n_reduce_steps; i++) {
                    bcj.bufs[i].reset(ggml_backend_alloc_buffer(bcj.backend, max_tmp_size));
                }
            }
            backend_ctx->max_tmp_size = max_tmp_size;
        }

        if (max_nnodes_raised || n_subgraphs > backend_ctx->max_subgraphs) {
            backend_ctx->max_subgraphs = std::max(backend_ctx->max_subgraphs, n_subgraphs);
            const size_t n_nodes_per_device = 3 * backend_ctx->n_reduce_steps; // tmp + ADD (+zeroing) graph per step and device
            const size_t n_cgraphs_per_device = 2 * backend_ctx->n_reduce_steps; // ADD ( + zeroing) graph per step and device
            const size_t mem_per_device_graphs_main = backend_ctx->max_subgraphs*ggml_graph_overhead_custom(backend_ctx->max_nnodes, cgraph->grads);
            const size_t mem_per_device_graphs_aux = n_cgraphs_per_device*backend_ctx->max_subgraphs*ggml_graph_overhead_custom(1, cgraph->grads);
            const size_t mem_per_device_nodes_aux = n_nodes_per_device*backend_ctx->max_subgraphs*ggml_tensor_overhead();
            ggml_init_params params = {
                /*.mem_size   =*/ n_backends * (mem_per_device_graphs_main + mem_per_device_graphs_aux + mem_per_device_nodes_aux),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            backend_ctx->ctx.reset(ggml_init(params));
            for (size_t j = 0; j < n_backends; j++) {
                auto & bcj = backend_ctx->backend_configs[j];
                for (size_t i = 0; i < n_subgraphs; i++) {
                    bcj.cgraphs[i].cgraph_main = ggml_new_graph_custom(backend_ctx->ctx.get(), cgraph->n_nodes, /*grads =*/ false);
                }
            }
            backend_ctx->cgraphs_aux.resize(n_backends*n_cgraphs_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->cgraphs_aux.size(); k++) {
                backend_ctx->cgraphs_aux[k] = ggml_new_graph_custom(backend_ctx->ctx.get(), 1, cgraph->grads);
            }
            backend_ctx->nodes_aux.resize(n_backends*n_nodes_per_device*backend_ctx->max_subgraphs);
            for (size_t k = 0; k < backend_ctx->nodes_aux.size(); k++) {
                backend_ctx->nodes_aux[k] = ggml_new_tensor_1d(backend_ctx->ctx.get(), GGML_TYPE_F32, 1);
            }
        }

        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            for (size_t i_graph = 0; i_graph < n_subgraphs; i_graph++) {
                ggml_cgraph * cgraph_ij = bcj.cgraphs[i_graph].cgraph_main;
                const size_t i_node_start = bcj.cgraphs[i_graph].offset;
                const size_t i_node_stop = i_graph + 1 < n_subgraphs ? bcj.cgraphs[i_graph + 1].offset : cgraph->n_nodes;
                cgraph_ij->n_nodes = i_node_stop - i_node_start;
                ggml_hash_set_reset(&cgraph_ij->visited_hash_set);
                for (size_t i_node = i_node_start; i_node < i_node_stop; i_node++) {
                    ggml_tensor * node_ij = bcj.nodes[i_node];
                    cgraph_ij->nodes[i_node - i_node_start] = node_ij;
                    const size_t hash_pos_orig = ggml_hash_find(&cgraph->visited_hash_set, cgraph->nodes[i_node]);
                    const size_t hash_pos_ij = ggml_hash_insert(&cgraph_ij->visited_hash_set, node_ij);
                    cgraph_ij->use_counts[hash_pos_ij] = cgraph->use_counts[hash_pos_orig];
                }
                cgraph_ij->uid = ggml_graph_next_uid();
            }
        }
    }

    size_t iga = 0; // i graph aux
    size_t ina = 0; // i node aux

    auto get_node_aux = [&](ggml_tensor * t) -> ggml_tensor * {
        ggml_tensor * ret = backend_ctx->nodes_aux[ina++];
        memset(ret, 0, sizeof(ggml_tensor));
        ret->op   = GGML_OP_NONE;
        ret->type = t->type;
        for (size_t k = 0; k < GGML_MAX_DIMS; k++) {
            ret->ne[k] = t->ne[k];
            ret->nb[k] = t->nb[k];
        }
        return ret;
    };
    auto set_tmp_data = [&](ggml_tensor * tensor, const size_t j, const size_t i_buf) {
        auto & bcj = backend_ctx->backend_configs[j];
        ggml_backend_buffer_ptr & buf_ptr = bcj.bufs[i_buf];
        if (!buf_ptr || ggml_backend_buffer_get_size(buf_ptr.get()) < backend_ctx->max_tmp_size) {
            buf_ptr.reset(ggml_backend_alloc_buffer(bcj.backend, backend_ctx->max_tmp_size));
        }
        tensor->buffer = buf_ptr.get();
        tensor->data   = ggml_backend_buffer_get_base(buf_ptr.get());
    };
    // FIXME usage_counts
    auto get_cgraph_aux = [&]() -> ggml_cgraph * {
        ggml_cgraph * ret = backend_ctx->cgraphs_aux[iga++];
        return ret;
    };

    // Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
    auto allreduce_fallback = [&](size_t i) -> ggml_status {
        std::vector<ggml_cgraph *> step_cgraphs(n_backends, nullptr);

        // Zero out nodes that were disabled due to having a zero-sized slice:
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_tensor * node = bcj.cgraphs[i].cgraph_main->nodes[bcj.cgraphs[i].cgraph_main->n_nodes - 1];
            if (node->flags & GGML_TENSOR_FLAG_COMPUTE) {
                continue;
            }
            ggml_tensor * node_zero = get_node_aux(node);
            node_zero->op = GGML_OP_SCALE; // FIXME 0.0f * NaN == NaN
            node_zero->src[0] = node;
            ggml_set_op_params_f32(node_zero, 0, 0.0f);
            node_zero->data = node->data;
            node_zero->flags |= GGML_TENSOR_FLAG_COMPUTE;

            step_cgraphs[j] = get_cgraph_aux();
            step_cgraphs[j]->nodes[0] = node_zero;
            step_cgraphs[j]->n_nodes = 1;
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }
        std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

        auto push_data = [&](const size_t j_src, const size_t j_dst, const size_t i_buf) {
            assert(step_cgraphs[j_dst] == nullptr);
            auto & bcj_src = backend_ctx->backend_configs[j_src];
            auto & bcj_dst = backend_ctx->backend_configs[j_dst];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            GGML_ASSERT(ggml_is_contiguous(node_src));
            GGML_ASSERT(ggml_is_contiguous(node_dst));

            ggml_tensor * node_tmp = get_node_aux(node_dst);
            set_tmp_data(node_tmp, j_dst, i_buf);

            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_tmp);

            ggml_tensor * node_red = get_node_aux(node_dst);
            node_red->view_src = node_dst->view_src == nullptr ? node_dst : node_dst->view_src;
            node_red->view_offs = node_dst->view_offs;
            node_red->op = GGML_OP_ADD;
            node_red->src[0] = node_dst;
            node_red->src[1] = node_tmp;
            node_red->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_backend_view_init(node_red);

            ggml_cgraph * cgraph_aux = get_cgraph_aux();
            cgraph_aux->nodes[0] = node_red;
            cgraph_aux->n_nodes = 1;
            step_cgraphs[j_dst] = cgraph_aux;
        };

        size_t offset_j = n_backends/2;
        while ((offset_j & (offset_j - 1)) != 0) {
            offset_j--;
        }
        const size_t offset_j_max = offset_j;
        size_t i_buf = 0;

        // If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
        for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
            const size_t j_dst = j_src - 2*offset_j_max;
            push_data(j_src, j_dst, i_buf);
            const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
            i_buf = 1;
        }

        // Butterfly reduction:
        for (; offset_j >= 1; offset_j /= 2) {
            std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                const size_t j_other = j ^ offset_j;
                if (j_other >= n_backends) {
                    continue;
                }
                push_data(j, j_other, i_buf);
            }

            for (size_t j = 0; j < 2*offset_j_max; j++) {
                if (step_cgraphs[j] == nullptr) {
                    continue;
                }
                auto & bcj = backend_ctx->backend_configs[j];
                const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
            i_buf++;
        }
        assert(i_buf == backend_ctx->n_reduce_steps);

        // If n_backends is not a power of 2, copy back the reduced tensors to the excess:
        for (size_t j = 2*offset_j_max; j < n_backends; j++) {
            auto & bcj_src = backend_ctx->backend_configs[j - 2*offset_j_max];
            auto & bcj_dst = backend_ctx->backend_configs[j];

            ggml_tensor * node_src = bcj_src.cgraphs[i].cgraph_main->nodes[bcj_src.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_tensor * node_dst = bcj_dst.cgraphs[i].cgraph_main->nodes[bcj_dst.cgraphs[i].cgraph_main->n_nodes - 1];
            ggml_backend_tensor_copy_async(bcj_src.backend, bcj_dst.backend, node_src, node_dst);
        }

        return GGML_STATUS_SUCCESS;
    };


    for (size_t i = 0; i < backend_ctx->n_subgraphs; i++) {
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, bcj.cgraphs[i].cgraph_main);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
            bool backend_allreduce_success = false;
            if (backend_ctx->comm_ctx) {
                std::vector<ggml_tensor *> nodes;
                nodes.reserve(n_backends);
                for (size_t j = 0; j < n_backends; j++) {
                    auto & bcj = backend_ctx->backend_configs[j];
                    ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
                    nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
                }
                backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
            }

            if (!backend_allreduce_success) {
                const ggml_status status = allreduce_fallback(i);
                if (status != GGML_STATUS_SUCCESS) {
                    return status;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_meta_i = {
    /* .get_name                = */ ggml_backend_meta_get_name,
    /* .free                    = */ ggml_backend_meta_free,
    /* .set_tensor_async        = */ ggml_backend_meta_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_meta_get_tensor_async,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_meta_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_meta_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

bool ggml_backend_is_meta(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_meta_i.get_name;
}

static ggml_backend_t ggml_backend_meta_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_meta_context * backend_ctx = new ggml_backend_meta_context(dev, params);

    ggml_backend_t backend = new struct ggml_backend;
    backend->guid    = ggml_backend_meta_guid();
    backend->iface   = ggml_backend_meta_i;
    backend->device  = dev;
    backend->context = backend_ctx;
    return backend;
}

size_t ggml_backend_meta_n_backends(ggml_backend_t meta_backend) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs.size();
}

ggml_backend_t ggml_backend_meta_simple_backend(ggml_backend_t meta_backend, size_t index) {
    GGML_ASSERT(ggml_backend_is_meta(meta_backend));
    const ggml_backend_meta_context * backend_ctx = (const ggml_backend_meta_context *) meta_backend->context;
    return backend_ctx->backend_configs[index].backend;
}


// ---- optimization API implementation ----

#include "ggml-opt.h.inc"

#include "ggml.h.inc"
#include "ggml-alloc.h.inc"
#include "ggml-backend.h.inc"
#include "ggml-impl-defs.inc"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <map>
#include <random>
#include <vector>

struct ggml_opt_dataset {
    struct ggml_context   * ctx    = nullptr;
    ggml_backend_buffer_t   buf    = nullptr;
    struct ggml_tensor    * data   = nullptr;
    struct ggml_tensor    * labels = nullptr;

    int64_t ndata       = -1;
    int64_t ndata_shard = -1;
    size_t  nbs_data    = -1;
    size_t  nbs_labels  = -1;

    std::vector<int64_t> permutation;
};

struct ggml_opt_context {
    ggml_backend_sched_t       backend_sched        = nullptr;
    ggml_cgraph              * allocated_graph      = nullptr;
    ggml_cgraph              * allocated_graph_copy = nullptr;
    struct ggml_context      * ctx_static           = nullptr;
    struct ggml_context      * ctx_cpu              = nullptr;
    struct ggml_context      * ctx_compute          = nullptr;
    struct ggml_context      * ctx_copy             = nullptr;
    ggml_backend_buffer_t      buf_static           = nullptr;
    ggml_backend_buffer_t      buf_cpu              = nullptr;
    std::mt19937               rng;
    enum ggml_opt_loss_type    loss_type;
    enum ggml_opt_build_type   build_type;
    enum ggml_opt_build_type   build_type_alloc;

    struct ggml_tensor * inputs  = nullptr;
    struct ggml_tensor * outputs = nullptr;
    struct ggml_tensor * labels  = nullptr;

    struct ggml_tensor * loss     = nullptr;
    struct ggml_tensor * pred     = nullptr;
    struct ggml_tensor * ncorrect = nullptr;

    struct ggml_cgraph * gf      = nullptr;
    struct ggml_cgraph * gb_grad = nullptr;
    struct ggml_cgraph * gb_opt  = nullptr;
    bool static_graphs           = false;
    bool eval_ready              = false;
    std::vector<struct ggml_tensor *> grad_accs;
    std::vector<struct ggml_tensor *> grad_m;
    std::vector<struct ggml_tensor *> grad_v;

    int64_t iter               = 1;
    int32_t opt_period         = 1;
    int32_t opt_i              = 0;
    bool    loss_per_datapoint = false;

    ggml_opt_get_optimizer_params get_opt_pars    = nullptr;
    void *                        get_opt_pars_ud = nullptr;
    struct ggml_tensor *          opt_step_params = nullptr; // Stores output of get_opt_pars.

    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

struct ggml_opt_result {
    int64_t              ndata    = 0;
    std::vector<float>   loss;
    std::vector<int32_t> pred;
    int64_t              ncorrect = 0;

    int64_t opt_period         = -1;
    bool    loss_per_datapoint = false;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(
        enum ggml_type type_data,
        enum ggml_type type_label,
        int64_t        ne_datapoint,
        int64_t        ne_label,
        int64_t        ndata,
        int64_t        ndata_shard) {
    GGML_ASSERT(ne_datapoint >  0);
    GGML_ASSERT(ne_label     >= 0);
    GGML_ASSERT(ndata        >  0);
    GGML_ASSERT(ndata_shard  >  0);

    ggml_opt_dataset_t result = new ggml_opt_dataset;
    result->ndata       = ndata;
    result->ndata_shard = ndata_shard;

    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 2*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        result->ctx = ggml_init(params);
    }

    result->data = ggml_new_tensor_2d(result->ctx, type_data, ne_datapoint, ndata);
    result->nbs_data = ggml_nbytes(result->data) * ndata_shard/ndata;

    if (ne_label > 0) {
        result->labels = ggml_new_tensor_2d(result->ctx, type_label, ne_label, ndata);
        result->nbs_labels = ggml_nbytes(result->labels) * ndata_shard/ndata;
    } else {
        result->labels = nullptr;
        result->nbs_labels = 0;
    }

    result->buf = ggml_backend_alloc_ctx_tensors_from_buft(result->ctx, ggml_backend_cpu_buffer_type());

    const int64_t nshards = ndata/ndata_shard;
    result->permutation.resize(nshards);
    for (int64_t i = 0; i < nshards; ++i) {
        result->permutation[i] = i;
    }
    return result;
}

void ggml_opt_dataset_free(ggml_opt_dataset_t dataset) {
    ggml_backend_buffer_free(dataset->buf);
    ggml_free(dataset->ctx);
    delete dataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t dataset) {
    return dataset->ndata;
}

struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset) {
    return dataset->data;
}

struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) {
    return dataset->labels;
}

void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata) {
    GGML_ASSERT(idata <= dataset->ndata);

    if (idata < 0) {
        std::shuffle(dataset->permutation.begin(), dataset->permutation.end(), opt_ctx->rng);
        return;
    }

    GGML_ASSERT(idata % dataset->ndata_shard == 0);
    const int64_t ishard_max = idata / dataset->ndata_shard;
    std::shuffle(dataset->permutation.begin(), dataset->permutation.begin() + ishard_max, opt_ctx->rng);
}

void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch) {
    GGML_ASSERT(   data_batch && ggml_is_contiguous(data_batch));
    GGML_ASSERT(!labels_batch || ggml_is_contiguous(labels_batch));
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(                   data_batch->type == dataset->data->type);
    GGML_ASSERT(!labels_batch || labels_batch->type == dataset->labels->type);

    const size_t nb_data_batch = ggml_nbytes(data_batch);
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);
    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    if (labels_batch) {
        const size_t nb_labels_batch = ggml_nbytes(labels_batch);
        GGML_ASSERT(nb_labels_batch == shards_per_batch*dataset->nbs_labels);
    }

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data = (const char *) dataset->data->data + ishard*dataset->nbs_data;
        ggml_backend_tensor_set(data_batch, ptr_data, ishard_batch*dataset->nbs_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels = (const char *) dataset->labels->data + ishard*dataset->nbs_labels;
        ggml_backend_tensor_set(labels_batch, ptr_labels, ishard_batch*dataset->nbs_labels, dataset->nbs_labels);
    }
}

void ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t dataset, void * data_batch, size_t nb_data_batch, void * labels_batch, int64_t ibatch) {
    GGML_ASSERT((labels_batch == nullptr) == (dataset->labels == nullptr));
    GGML_ASSERT(nb_data_batch % dataset->nbs_data == 0);

    const int64_t shards_per_batch = nb_data_batch / dataset->nbs_data;

    GGML_ASSERT((ibatch + 1)*shards_per_batch <= int64_t(dataset->permutation.size()));

    for (int64_t ishard_batch = 0; ishard_batch < shards_per_batch; ++ishard_batch) {
        const int64_t ishard = dataset->permutation[ibatch*shards_per_batch + ishard_batch];

        const char * ptr_data       = (const char *) dataset->data->data + ishard      *dataset->nbs_data;
        char       * ptr_data_batch = (char       *) data_batch          + ishard_batch*dataset->nbs_data;
        memcpy(ptr_data_batch, ptr_data, dataset->nbs_data);

        if (!labels_batch) {
            continue;
        }

        const char * ptr_labels       = (const char *) dataset->labels->data + ishard      *dataset->nbs_labels;
        char       * ptr_labels_batch = (char       *) labels_batch          + ishard_batch*dataset->nbs_labels;
        memcpy(ptr_labels_batch, ptr_labels, dataset->nbs_labels);
    }
}

// ====== Model / Context ======

struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata) {
    GGML_UNUSED(userdata);

    ggml_opt_optimizer_params result;

    result.adamw.alpha = 0.001f;
    result.adamw.beta1 = 0.9f;
    result.adamw.beta2 = 0.999f;
    result.adamw.eps   = 1e-8f;
    result.adamw.wd    = 0.0f;

    result.sgd.alpha   = 1e-3f;
    result.sgd.wd      = 0.0f;

    return result;
}


struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata) {
    return *((struct ggml_opt_optimizer_params *) userdata);
}

struct ggml_opt_params ggml_opt_default_params(
        ggml_backend_sched_t      backend_sched,
        enum ggml_opt_loss_type   loss_type) {
    return {
        /*backend_sched   =*/ backend_sched,
        /*ctx_compute     =*/ nullptr,
        /*inputs          =*/ nullptr,
        /*logits          =*/ nullptr,
        /*loss_type       =*/ loss_type,
        /*build_type      =*/ GGML_OPT_BUILD_TYPE_OPT,
        /*opt_period      =*/ 1,
        /*get_opt_pars    =*/ ggml_opt_get_default_optimizer_params,
        /*get_opt_pars_ud =*/ nullptr,
        /*optimizer       =*/ GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    };
}

static ggml_tensor * map_tensor(std::map<ggml_tensor *, ggml_tensor *> & tensor_map, ggml_context * ctx, ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }

    if (tensor_map.find(tensor) != tensor_map.end()) {
        return tensor_map[tensor];
    }

    ggml_tensor * new_tensor = ggml_dup_tensor(ctx, tensor);
    tensor_map[tensor] = new_tensor;

    new_tensor->op = tensor->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        new_tensor->nb[i] = tensor->nb[i];
    }
    new_tensor->flags = tensor->flags;
    memcpy(new_tensor->op_params, tensor->op_params, sizeof(tensor->op_params));
    strcpy(new_tensor->name, tensor->name);
    new_tensor->data = tensor->data;
    new_tensor->buffer = tensor->buffer;
    new_tensor->extra = tensor->extra;
    new_tensor->view_offs = tensor->view_offs;
    new_tensor->view_src = map_tensor(tensor_map, ctx, tensor->view_src);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        new_tensor->src[i] = map_tensor(tensor_map, ctx, tensor->src[i]);
    }

    return new_tensor;
}

static ggml_cgraph * dup_graph(ggml_context * ctx, ggml_cgraph * src) {
    std::map<ggml_tensor *, ggml_tensor *> tensor_map;

    ggml_cgraph * dst = ggml_new_graph_custom(ctx, src->size, /*grads =*/ true);

    for (int i = 0; i < src->n_leafs; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->leafs[i]));
    }
    GGML_ASSERT(dst->n_leafs == src->n_leafs);
    for (int i = 0; i < src->n_nodes; i++) {
        ggml_build_forward_expand(dst, map_tensor(tensor_map, ctx, src->nodes[i]));
    }
    GGML_ASSERT(dst->n_nodes == src->n_nodes);
    for (int i = 0; i < src->n_nodes; ++i) {
        const size_t igrad_src = ggml_hash_find(&src->visited_hash_set, src->nodes[i]);
        const size_t igrad_dst = ggml_hash_find(&dst->visited_hash_set, dst->nodes[i]);

        GGML_ASSERT(igrad_src != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(src->visited_hash_set.used, igrad_src));
        GGML_ASSERT(igrad_dst != GGML_HASHSET_FULL);
        GGML_ASSERT(ggml_bitset_get(dst->visited_hash_set.used, igrad_dst));

        dst->grads[igrad_dst]     = src->grads[igrad_src];
        dst->grad_accs[igrad_dst] = src->grad_accs[igrad_src];
    }

    return dst;
}

static void ggml_opt_build(ggml_opt_context_t opt_ctx) {
    GGML_ASSERT(opt_ctx->ctx_compute && "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
    GGML_ASSERT((!opt_ctx->static_graphs || opt_ctx->inputs->data) && "when using static graphs the inputs must be allocated statically");

    const enum ggml_opt_optimizer_type optimizer = opt_ctx->optimizer;

    const bool accumulate = opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD &&
        !(opt_ctx->static_graphs && opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period == 1);

    const bool need_momenta = opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT &&
        opt_ctx->optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

    ggml_set_input(opt_ctx->inputs);
    ggml_set_output(opt_ctx->outputs);

    int n_param = 0;
    for (int i = 0; i < opt_ctx->gf->n_nodes; ++i) {
        const struct ggml_tensor * node = opt_ctx->gf->nodes[i];
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            n_param++;
        }
        GGML_ASSERT(!(node->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
    }

    if (!opt_ctx->ctx_static) {
        // The static context is used for:
        //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
        //   - optimizer momenta (2 tensors per param)
        //   - labels (if using static graphs)
        //   - loss (if using static graphs, up to 5 tensors)
        //   - pred (if using static graphs)
        //   - ncorrect (if using static graphs, 2 tensors).
        constexpr size_t n_loss = 1;
        const size_t tensors_per_param = (accumulate ? 1 : 0) + (need_momenta ? 2 : 0);
        const size_t tensors_const = opt_ctx->static_graphs ? 9 : 0;
        const size_t size_meta = (n_loss + tensors_per_param*n_param + tensors_const) * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        opt_ctx->ctx_static = ggml_init(params);
    }
    GGML_ASSERT(opt_ctx->build_type <= opt_ctx->build_type_alloc);

    {
        // The cpu context is allocated statically if using static graphs, dynamically otherwise.
        // It is used for:
        //   - optimizer parameters (1 shared for all optimizer invocations)
        const size_t size_meta = 1 * ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_meta,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_cpu);
        opt_ctx->ctx_cpu = ggml_init(params);

        ggml_backend_buffer_free(opt_ctx->buf_cpu);
        opt_ctx->buf_cpu = nullptr;
    }

    struct ggml_context * ctx_results = opt_ctx->static_graphs ? opt_ctx->ctx_static : opt_ctx->ctx_compute;

    switch (opt_ctx->loss_type) {
        case GGML_OPT_LOSS_TYPE_MEAN: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_SUM: {
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->outputs);
            ggml_set_name(opt_ctx->loss, "loss_sum");
            opt_ctx->loss_per_datapoint = false;
            break;
        }
        case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_input(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_cross_entropy_loss(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_cross_entropy");
            if (opt_ctx->opt_period > 1) {
                opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, 1.0f / opt_ctx->opt_period);
                ggml_set_name(opt_ctx->loss, "loss_cross_entropy_scaled");
            }
            opt_ctx->loss_per_datapoint = true;
            break;
        }
        case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
            opt_ctx->labels = ggml_dup_tensor(ctx_results, opt_ctx->outputs);
            ggml_set_input(opt_ctx->labels);
            ggml_set_name(opt_ctx->labels, "labels");
            opt_ctx->loss = ggml_sub(ctx_results, opt_ctx->outputs, opt_ctx->labels);
            ggml_set_name(opt_ctx->loss, "loss_error");
            opt_ctx->loss = ggml_sqr(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_squared_error");
            opt_ctx->loss = ggml_sum(ctx_results, opt_ctx->loss);
            ggml_set_name(opt_ctx->loss, "loss_sum_squared_error");
            const float scale = 1.0f / (opt_ctx->opt_period * ggml_nelements(opt_ctx->outputs));
            opt_ctx->loss = ggml_scale(ctx_results, opt_ctx->loss, scale);
            ggml_set_name(opt_ctx->loss, "loss_mean_squared_error");
            opt_ctx->loss_per_datapoint = true;
            break;
        }
    }
    ggml_set_output(opt_ctx->loss);
    ggml_set_loss(opt_ctx->loss);
    ggml_build_forward_expand(opt_ctx->gf, opt_ctx->loss);

    if (opt_ctx->loss_type == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
        opt_ctx->pred = ggml_argmax(ctx_results, opt_ctx->outputs);
        ggml_set_name(opt_ctx->pred, "pred");
        ggml_set_output(opt_ctx->pred);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->pred);

        opt_ctx->ncorrect = ggml_count_equal(ctx_results, opt_ctx->pred, ggml_argmax(ctx_results, opt_ctx->labels));
        ggml_set_name(opt_ctx->ncorrect, "ncorrect");
        ggml_set_output(opt_ctx->ncorrect);
        ggml_build_forward_expand(opt_ctx->gf, opt_ctx->ncorrect);
    }

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_FORWARD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_FORWARD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        return;
    }

    if (opt_ctx->grad_accs.empty()) {
        GGML_ASSERT(opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_GRAD);

        const int n_nodes = opt_ctx->gf->n_nodes;
        opt_ctx->grad_accs.resize(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor * node = opt_ctx->gf->nodes[i];
            if ((accumulate && (node->flags & GGML_TENSOR_FLAG_PARAM)) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
                opt_ctx->grad_accs[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            } else {
                opt_ctx->grad_accs[i] = nullptr;
            }
        }

        if (need_momenta && opt_ctx->build_type_alloc >= GGML_OPT_BUILD_TYPE_OPT) {
            opt_ctx->grad_m.resize(n_nodes);
            opt_ctx->grad_v.resize(n_nodes);
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor * node = opt_ctx->gf->nodes[i];
                if (node->flags & GGML_TENSOR_FLAG_PARAM) {
                    opt_ctx->grad_m[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                    opt_ctx->grad_v[i] = ggml_new_tensor(opt_ctx->ctx_static, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                } else {
                    opt_ctx->grad_m[i] = nullptr;
                    opt_ctx->grad_v[i] = nullptr;
                }
            }
        }
    }

    // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
    opt_ctx->gb_grad = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gf, /*force_grads =*/ true);
    ggml_build_backward_expand(opt_ctx->ctx_compute, opt_ctx->gb_grad, opt_ctx->grad_accs.data());

    if (opt_ctx->buf_static) {
        if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_GRAD) {
            return;
        }
    } else if (opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_GRAD) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_grad);
    }

    GGML_ASSERT(opt_ctx->build_type_alloc == GGML_OPT_BUILD_TYPE_OPT);

    // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
    opt_ctx->gb_opt = ggml_graph_dup(opt_ctx->ctx_compute, opt_ctx->gb_grad, /*force_grads =*/ true);

    opt_ctx->opt_step_params = ggml_new_tensor_1d(opt_ctx->ctx_cpu, GGML_TYPE_F32, need_momenta ? 7 : 2);
    ggml_tensor * adamw_params = opt_ctx->opt_step_params;
    ggml_set_input(adamw_params);
    const char * optimizer_name = ggml_opt_optimizer_name(opt_ctx->optimizer);
    ggml_format_name(adamw_params, "%s_params", optimizer_name);
    for (int i = opt_ctx->gf->n_nodes-1; i >= 0; --i) {
        struct ggml_tensor * node = opt_ctx->gb_opt->nodes[i];
        struct ggml_tensor * grad = ggml_graph_get_grad(opt_ctx->gb_opt, node);

        if (grad && (node->flags & GGML_TENSOR_FLAG_PARAM)) {
            struct ggml_tensor * m = nullptr;
            struct ggml_tensor * v = nullptr;
            if (need_momenta) {
                m = opt_ctx->grad_m[i];
                v = opt_ctx->grad_v[i];
                ggml_format_name(m, "AdamW m for %s", node->name);
                ggml_format_name(v, "AdamW v for %s", node->name);
            }
            struct ggml_tensor * opt_step;
            switch (optimizer) {
                case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
                    opt_step = ggml_opt_step_adamw(opt_ctx->ctx_compute, node, grad, m, v, adamw_params);
                    break;
                case GGML_OPT_OPTIMIZER_TYPE_SGD:
                    opt_step = ggml_opt_step_sgd(opt_ctx->ctx_compute, node, grad, adamw_params);
                    break;
                default:
                    GGML_ABORT("fatal error");
            }
            ggml_format_name(opt_step, "%s step for %s", optimizer_name, node->name);
            ggml_build_forward_expand(opt_ctx->gb_opt, opt_step);
        }
    }

    if (!opt_ctx->buf_static) {
        opt_ctx->buf_static = ggml_backend_alloc_ctx_tensors(
            opt_ctx->ctx_static, ggml_backend_sched_get_backend(opt_ctx->backend_sched, 0));
        ggml_graph_reset(opt_ctx->gb_opt);
    }

    opt_ctx->buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(opt_ctx->ctx_cpu, ggml_backend_cpu_buffer_type());
}

ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params) {
    ggml_opt_context_t result = new struct ggml_opt_context;
    result->backend_sched    = params.backend_sched;
    result->ctx_compute      = params.ctx_compute;
    result->loss_type        = params.loss_type;
    result->build_type       = params.build_type;
    result->build_type_alloc = params.build_type;
    result->inputs           = params.inputs;
    result->outputs          = params.outputs;
    result->opt_period       = params.opt_period;
    result->get_opt_pars     = params.get_opt_pars;
    result->get_opt_pars_ud  = params.get_opt_pars_ud;
    result->optimizer        = params.optimizer;

    GGML_ASSERT(result->opt_period >= 1);

    result->static_graphs = result->ctx_compute;

    if (!result->static_graphs) {
        GGML_ASSERT(!result->inputs);
        GGML_ASSERT(!result->outputs);
        return result;
    }

    GGML_ASSERT(result->inputs);
    GGML_ASSERT(result->outputs);

    result->gf = ggml_new_graph_custom(result->ctx_compute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/ true); // Forward pass.
    ggml_build_forward_expand(result->gf, result->outputs);

    ggml_opt_build(result);

    return result;
}

void ggml_opt_free(ggml_opt_context_t opt_ctx) {
    if (opt_ctx == nullptr) {
        return;
    }
    ggml_backend_buffer_free(opt_ctx->buf_static);
    ggml_backend_buffer_free(opt_ctx->buf_cpu);
    ggml_free(opt_ctx->ctx_static);
    ggml_free(opt_ctx->ctx_cpu);
    ggml_free(opt_ctx->ctx_copy);
    delete opt_ctx;
}

void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer) {
    if (optimizer) {
        ggml_graph_reset(opt_ctx->gb_opt);
        opt_ctx->iter = 1;
    } else {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
}

bool ggml_opt_static_graphs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->static_graphs;
}

struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->inputs;
}

struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx) {
    return opt_ctx->outputs;
}

struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx) {
    return opt_ctx->labels;
}

struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx) {
    return opt_ctx->loss;
}

struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx) {
    return opt_ctx->pred;
}

struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) {
    return opt_ctx->ncorrect;
}

struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node) {
    return ggml_graph_get_grad_acc(opt_ctx->gb_opt, node);
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() {
    return new ggml_opt_result;
}

void ggml_opt_result_free(ggml_opt_result_t result) {
    delete result;
}

void ggml_opt_result_reset(ggml_opt_result_t result) {
    result->ndata = 0;
    result->loss.clear();
    result->pred.clear();
    result->ncorrect = 0;
}

void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata) {
    *ndata = result->ndata;
}

void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc) {
    const int64_t nbatches = result->loss.size(); // Number of physical batches.

    if (nbatches == 0) {
        *loss = 0.0;
        *unc  = NAN;
        return;
    }

    double sum         = 0.0;
    double sum_squared = 0.0;

    for (const float & loss : result->loss) {
        // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
        const float loss_scaled = result->loss_per_datapoint ? loss*result->opt_period : loss;
        sum         += loss_scaled;
        sum_squared += loss_scaled*loss_scaled;
    }

    const double mean = sum/nbatches;
    *loss = result->loss_per_datapoint ? mean : sum;

    if (!unc) {
        return;
    }

    if (nbatches < 2) {
        *unc = NAN;
        return;
    }

    const double var_sum = sum_squared/nbatches - mean*mean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
    *unc = result->loss_per_datapoint ? sqrt(var_sum / (nbatches - 1)) : sqrt(var_sum * nbatches/(nbatches - 1));
}

void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred) {
    for (size_t i = 0; i < result->pred.size(); ++i) {
        pred[i] = result->pred[i];
    }
}

void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc) {
    *accuracy = result->ncorrect >= 0 ? double(result->ncorrect) / double(result->ndata) : NAN;

    if (!unc) {
        return;
    }

    *unc = result->ncorrect >= 0 && result->ndata >= 2 ?
        sqrt((*accuracy) * (1.0 - (*accuracy)) / double(result->ndata - 1)) : NAN;
}

// ====== Computation ======

void ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        struct ggml_context * ctx_compute,
        struct ggml_cgraph  * gf,
        struct ggml_tensor  * inputs,
        struct ggml_tensor  * outputs) {
    GGML_ASSERT(!opt_ctx->static_graphs);
    opt_ctx->ctx_compute = ctx_compute;
    opt_ctx->gf          = gf;
    opt_ctx->inputs      = inputs;
    opt_ctx->outputs     = outputs;
}

void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bool backward) {
    GGML_ASSERT(!opt_ctx->eval_ready);
    if (opt_ctx->build_type == GGML_OPT_BUILD_TYPE_OPT && opt_ctx->opt_period > 1 && opt_ctx->opt_i == 0) {
        ggml_graph_reset(opt_ctx->gb_grad);
    }
    if (backward) {
        const int32_t opt_i_next = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;
        opt_ctx->build_type = opt_i_next == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
    } else {
        opt_ctx->build_type = GGML_OPT_BUILD_TYPE_FORWARD;
    }

    if (!opt_ctx->static_graphs) {
        ggml_opt_build(opt_ctx);
    }

    struct ggml_cgraph * graph = nullptr;
    switch (opt_ctx->build_type) {
        case GGML_OPT_BUILD_TYPE_FORWARD: {
            graph = opt_ctx->gf;
        } break;
        case GGML_OPT_BUILD_TYPE_GRAD: {
            graph = opt_ctx->gb_grad;
        } break;
        case GGML_OPT_BUILD_TYPE_OPT: {
            graph = opt_ctx->gb_opt;
        } break;
    }
    GGML_ASSERT(graph);

    if (opt_ctx->allocated_graph == graph) {
        opt_ctx->eval_ready = true;
        return;
    }

    ggml_backend_sched_reset(opt_ctx->backend_sched); // clear allocation of previous graph

    if (opt_ctx->static_graphs) {
        ggml_init_params params = {
            /*.mem_size   =*/ graph->size*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph->size, graph->grads),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_free(opt_ctx->ctx_copy);
        opt_ctx->ctx_copy = ggml_init(params);

        opt_ctx->allocated_graph_copy = dup_graph(opt_ctx->ctx_copy, graph);
    } else {
        opt_ctx->allocated_graph_copy = graph;
    }

    ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->allocated_graph = graph;

    opt_ctx->eval_ready = true;
}

void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result) {
    GGML_ASSERT(opt_ctx->eval_ready);
    if (opt_ctx->allocated_graph == opt_ctx->gb_opt) {
        const ggml_opt_optimizer_params & opt_pars = opt_ctx->get_opt_pars(opt_ctx->get_opt_pars_ud);

        switch (opt_ctx->optimizer) {
            case GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
                GGML_ASSERT(opt_pars.adamw.alpha > 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta1 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.beta2 <= 1.0f);
                GGML_ASSERT(opt_pars.adamw.eps >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd >= 0.0f);
                GGML_ASSERT(opt_pars.adamw.wd <= 1.0f);

                // beta1, beta2 after applying warmup
                const float beta1h = 1.0f / (1.0f - powf(opt_pars.adamw.beta1, opt_ctx->iter));
                const float beta2h = 1.0f / (1.0f - powf(opt_pars.adamw.beta2, opt_ctx->iter));

                float * adamw_par_data = ggml_get_data_f32(opt_ctx->opt_step_params);
                adamw_par_data[0] = opt_pars.adamw.alpha;
                adamw_par_data[1] = opt_pars.adamw.beta1;
                adamw_par_data[2] = opt_pars.adamw.beta2;
                adamw_par_data[3] = opt_pars.adamw.eps;
                adamw_par_data[4] = opt_pars.adamw.wd;
                adamw_par_data[5] = beta1h;
                adamw_par_data[6] = beta2h;
            } break;
            case GGML_OPT_OPTIMIZER_TYPE_SGD: {
                GGML_ASSERT(opt_pars.sgd.alpha > 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd >= 0.0f);
                GGML_ASSERT(opt_pars.sgd.wd <= 1.0f);
                float * sgd = ggml_get_data_f32(opt_ctx->opt_step_params);
                sgd[0] = opt_pars.sgd.alpha;
                sgd[1] = opt_pars.sgd.wd;
            } break;
            default:
                GGML_ABORT("fatal error");
        }
    }

    ggml_backend_sched_graph_compute(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy);
    opt_ctx->iter += opt_ctx->allocated_graph == opt_ctx->gb_opt;
    opt_ctx->opt_i = (opt_ctx->opt_i + 1) % opt_ctx->opt_period;

    if (!opt_ctx->static_graphs) {
        opt_ctx->gf                   = nullptr;
        opt_ctx->gb_grad              = nullptr;
        opt_ctx->gb_opt               = nullptr;
        opt_ctx->allocated_graph      = nullptr;
        opt_ctx->allocated_graph_copy = nullptr;
    }

    opt_ctx->eval_ready = false;

    if (!result) {
        return;
    }

    if (result->ndata == 0) {
        result->loss_per_datapoint = opt_ctx->loss_per_datapoint;
        result->opt_period         = opt_ctx->opt_period;
    } else {
        GGML_ASSERT(result->loss_per_datapoint == opt_ctx->loss_per_datapoint);
        GGML_ASSERT(result->opt_period         == opt_ctx->opt_period);
    }

    const int64_t ndata = opt_ctx->outputs->ne[1];
    GGML_ASSERT(result->ndata == ndata*int64_t(result->loss.size()) && "varying batch size not supported");
    result->ndata += ndata;

    GGML_ASSERT(ggml_is_scalar(opt_ctx->loss));
    GGML_ASSERT(opt_ctx->loss->type == GGML_TYPE_F32);
    float loss;
    ggml_backend_tensor_get(opt_ctx->loss, &loss, 0, ggml_nbytes(opt_ctx->loss));
    result->loss.push_back(loss);

    if (opt_ctx->pred) {
        GGML_ASSERT(opt_ctx->pred->type == GGML_TYPE_I32);
        std::vector<int32_t> pred(ndata);
        ggml_backend_tensor_get(opt_ctx->pred, pred.data(), 0, ggml_nbytes(opt_ctx->pred));
        result->pred.insert(result->pred.end(), pred.begin(), pred.end());
    }

    if (!opt_ctx->ncorrect || result->ncorrect < 0) {
        result->ncorrect = -1;
        return;
    }

    GGML_ASSERT(ggml_is_scalar(opt_ctx->ncorrect));
    GGML_ASSERT(opt_ctx->ncorrect->type == GGML_TYPE_I64);
    int64_t ncorrect;
    ggml_backend_tensor_get(opt_ctx->ncorrect, &ncorrect, 0, ggml_nbytes(opt_ctx->ncorrect));
    result->ncorrect += ncorrect;
}

// ====== High-Level Functions ======

void ggml_opt_epoch(
        ggml_opt_context_t      opt_ctx,
        ggml_opt_dataset_t      dataset,
        ggml_opt_result_t       result_train,
        ggml_opt_result_t       result_eval,
        int64_t                 idata_split,
        ggml_opt_epoch_callback callback_train,
        ggml_opt_epoch_callback callback_eval) {
    GGML_ASSERT(ggml_opt_static_graphs(opt_ctx) && "ggml_opt_epoch requires static graphs");
    struct ggml_tensor * inputs = ggml_opt_inputs(opt_ctx);
    struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
    struct ggml_tensor * data   = ggml_opt_dataset_data(dataset);
    GGML_ASSERT(data->ne[0] == inputs->ne[0]);

    const int64_t ndata       =   data->ne[1];
    const int64_t ndata_batch = inputs->ne[1];

    GGML_ASSERT(data->ne[1] % inputs->ne[1] == 0);
    const int64_t nbatches = ndata/ndata_batch;

    idata_split = idata_split < 0 ? ndata : idata_split;
    GGML_ASSERT(idata_split % ndata_batch == 0);
    const int64_t ibatch_split = idata_split / ndata_batch;

    int64_t ibatch = 0;
    int64_t t_loop_start = ggml_time_us();
    for (; ibatch < ibatch_split; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ true);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_train);
        if (callback_train) {
            callback_train(true, opt_ctx, dataset, result_train, ibatch+1, ibatch_split, t_loop_start);
        }
    }
    t_loop_start = ggml_time_us();
    for (; ibatch < nbatches; ++ibatch) {
        ggml_opt_alloc(opt_ctx, /*backward =*/ false);
        ggml_opt_dataset_get_batch(dataset, inputs, labels, ibatch);
        ggml_opt_eval(opt_ctx, result_eval);
        if (callback_eval) {
            callback_eval(false, opt_ctx, dataset, result_eval, ibatch+1-ibatch_split, nbatches-ibatch_split, t_loop_start);
        }
    }
}

void ggml_opt_epoch_callback_progress_bar(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    fprintf(stderr, "%s[", train ? "train: " : "val:   ");

    // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
    constexpr int64_t bar_length = 8;
    const int64_t ibatch8 = 8 * ibatch;
    for (int64_t j = 0; j < bar_length; ++j) {
        if        (ibatch_max * (8*j + 8) / bar_length < ibatch8) {
            fprintf(stderr, "\u2588"); // full block
        } else if (ibatch_max * (8*j + 7) / bar_length < ibatch8) {
            fprintf(stderr, "\u2589"); // 7/8 filled
        } else if (ibatch_max * (8*j + 6) / bar_length < ibatch8) {
            fprintf(stderr, "\u258A"); // 6/8 filled
        } else if (ibatch_max * (8*j + 5) / bar_length < ibatch8) {
            fprintf(stderr, "\u258B"); // 5/8 filled
        } else if (ibatch_max * (8*j + 4) / bar_length < ibatch8) {
            fprintf(stderr, "\u258C"); // 4/8 filled
        } else if (ibatch_max * (8*j + 3) / bar_length < ibatch8) {
            fprintf(stderr, "\u258D"); // 3/8 filled
        } else if (ibatch_max * (8*j + 2) / bar_length < ibatch8) {
            fprintf(stderr, "\u258E"); // 2/8 filled
        } else if (ibatch_max * (8*j + 1) / bar_length < ibatch8) {
            fprintf(stderr, "\u258F"); // 1/8 filled
        } else {
            fprintf(stderr, " ");
        }
    }

    const int64_t batch_size = ggml_opt_inputs(opt_ctx)->ne[1];
    const int64_t idata      = ibatch*batch_size;
    const int64_t idata_max  = ibatch_max*batch_size;

    double loss;
    double loss_unc;
    ggml_opt_result_loss(result, &loss, &loss_unc);

    double accuracy;
    double accuracy_unc;
    ggml_opt_result_accuracy(result, &accuracy, &accuracy_unc);

    const int64_t t_ibatch_us = ggml_time_us() - t_start_us;
    int64_t t_ibatch_s = t_ibatch_us / 1000000;
    const int64_t t_ibatch_h = t_ibatch_s / 3600;
    t_ibatch_s -= t_ibatch_h * 3600;
    const int64_t t_ibatch_m = t_ibatch_s / 60;
    t_ibatch_s -= t_ibatch_m * 60;

    const int64_t t_eta_us = t_ibatch_us * (ibatch_max - ibatch)/ibatch;
    int64_t t_eta_s = t_eta_us / 1000000;
    const int64_t t_eta_h = t_eta_s / 3600;
    t_eta_s -= t_eta_h * 3600;
    const int64_t t_eta_m = t_eta_s / 60;
    t_eta_s -= t_eta_m * 60;

    fprintf(stderr, "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
            "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
            idata, idata_max, loss, loss_unc, 100.0*accuracy, 100.0*accuracy_unc,
            t_ibatch_h, t_ibatch_m, t_ibatch_s, t_eta_h, t_eta_m, t_eta_s);
    if (ibatch == ibatch_max) {
        fprintf(stderr, "\n");
    }
    fflush(stderr);

    GGML_UNUSED(dataset);
}

void ggml_opt_fit(
        ggml_backend_sched_t            backend_sched,
        ggml_context                  * ctx_compute,
        ggml_tensor                   * inputs,
        ggml_tensor                   * outputs,
        ggml_opt_dataset_t              dataset,
        enum ggml_opt_loss_type         loss_type,
        enum ggml_opt_optimizer_type    optimizer,
        ggml_opt_get_optimizer_params   get_opt_pars,
        int64_t                         nepoch,
        int64_t                         nbatch_logical,
        float                           val_split,
        bool                            silent) {
    ggml_time_init();
    const int64_t t_start_us = ggml_time_us();

    const int64_t ndata           = ggml_opt_dataset_data(dataset)->ne[1];
    const int64_t nbatch_physical = inputs->ne[1];
    GGML_ASSERT(ndata          % nbatch_logical  == 0);
    GGML_ASSERT(nbatch_logical % nbatch_physical == 0);

    const int64_t opt_period       = nbatch_logical / nbatch_physical;
    const int64_t nbatches_logical = ndata / nbatch_logical;

    GGML_ASSERT(val_split >= 0.0f);
    GGML_ASSERT(val_split <  1.0f);
    const int64_t ibatch_split = int64_t(((1.0f - val_split) * nbatches_logical)) * opt_period; // train <-> val split index (physical)
    const int64_t idata_split  = ibatch_split * nbatch_physical;

    int64_t epoch = 1;

    ggml_opt_params params = ggml_opt_default_params(backend_sched, loss_type);
    params.ctx_compute     = ctx_compute;
    params.inputs          = inputs;
    params.outputs         = outputs;
    params.opt_period      = opt_period;
    params.get_opt_pars    = get_opt_pars;
    params.get_opt_pars_ud = &epoch;
    params.optimizer       = optimizer;
    ggml_opt_context_t opt_ctx = ggml_opt_init(params);

    // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
    if (nbatch_logical < ndata) {
        ggml_opt_dataset_shuffle(opt_ctx, dataset, -1); // Shuffle all data (train + validation).
    }

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_val   = ggml_opt_result_init();

    ggml_opt_epoch_callback epoch_callback = silent ? nullptr : ggml_opt_epoch_callback_progress_bar;

    for (; epoch <= nepoch; ++epoch) {
        if (nbatch_logical < idata_split) {
            ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_val);

        if (!silent) {
            fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, epoch, nepoch);
        }
        ggml_opt_epoch(opt_ctx, dataset, result_train, result_val, idata_split, epoch_callback, epoch_callback);
        if (!silent) {
            fprintf(stderr, "\n");
        }
    }

    if (!silent) {
        int64_t t_total_s = (ggml_time_us() - t_start_us) / 1000000;
        const int64_t t_total_h = t_total_s / 3600;
        t_total_s -= t_total_h * 3600;
        const int64_t t_total_m = t_total_s / 60;
        t_total_s -= t_total_m * 60;
        fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, t_total_h, t_total_m, t_total_s);
    }

    ggml_opt_free(opt_ctx);
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_val);
}

enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t c) {
    return c->optimizer;
}

GGML_API const char * ggml_opt_optimizer_name(enum ggml_opt_optimizer_type o) {
    switch (o) {
        case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
            return "adamw";
        case GGML_OPT_OPTIMIZER_TYPE_SGD:
            return "sgd";
        default:
            return "undefined";
    };
}


// ---- GGUF implementation ----

#include "ggml.h.inc"
#include "ggml-backend.h.inc"
#include "ggml-impl-defs.inc"
#include "gguf.h.inc"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#define GGUF_MAX_STRING_LENGTH  (1024*1024*1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024*1024*1024)

#ifdef _WIN32
#    define gguf_ftell _ftelli64
#    define gguf_fseek _fseeki64
#else
#    define gguf_ftell ftello
#    define gguf_fseek fseeko
#endif

template <typename T>
struct type_to_gguf_type;

template <>
struct type_to_gguf_type<uint8_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT8;
};

template <>
struct type_to_gguf_type<int8_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT8;
};

template <>
struct type_to_gguf_type<uint16_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT16;
};

template <>
struct type_to_gguf_type<int16_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT16;
};

template <>
struct type_to_gguf_type<uint32_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT32;
};

template <>
struct type_to_gguf_type<int32_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT32;
};

template <>
struct type_to_gguf_type<float> {
    static constexpr enum gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
struct type_to_gguf_type<bool> {
    static constexpr enum gguf_type value = GGUF_TYPE_BOOL;
};

template <>
struct type_to_gguf_type<std::string> {
    static constexpr enum gguf_type value = GGUF_TYPE_STRING;
};

template <>
struct type_to_gguf_type<uint64_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT64;
};

template <>
struct type_to_gguf_type<int64_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT64;
};

template <>
struct type_to_gguf_type<double> {
    static constexpr enum gguf_type value = GGUF_TYPE_FLOAT64;
};

static const std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   sizeof(uint8_t)},
    {GGUF_TYPE_INT8,    sizeof(int8_t)},
    {GGUF_TYPE_UINT16,  sizeof(uint16_t)},
    {GGUF_TYPE_INT16,   sizeof(int16_t)},
    {GGUF_TYPE_UINT32,  sizeof(uint32_t)},
    {GGUF_TYPE_INT32,   sizeof(int32_t)},
    {GGUF_TYPE_FLOAT32, sizeof(float)},
    {GGUF_TYPE_BOOL,    sizeof(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  sizeof(uint64_t)},
    {GGUF_TYPE_INT64,   sizeof(int64_t)},
    {GGUF_TYPE_FLOAT64, sizeof(double)},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

static const std::map<gguf_type, const char *> GGUF_TYPE_NAME = {
    {GGUF_TYPE_UINT8,   "u8"},
    {GGUF_TYPE_INT8,    "i8"},
    {GGUF_TYPE_UINT16,  "u16"},
    {GGUF_TYPE_INT16,   "i16"},
    {GGUF_TYPE_UINT32,  "u32"},
    {GGUF_TYPE_INT32,   "i32"},
    {GGUF_TYPE_FLOAT32, "f32"},
    {GGUF_TYPE_BOOL,    "bool"},
    {GGUF_TYPE_STRING,  "str"},
    {GGUF_TYPE_ARRAY,   "arr"},
    {GGUF_TYPE_UINT64,  "u64"},
    {GGUF_TYPE_INT64,   "i64"},
    {GGUF_TYPE_FLOAT64, "f64"},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_size(enum gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    return it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

struct gguf_kv {
    std::string key;

    bool is_array;
    enum gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(const std::string & key, const T value)
            : key(key), is_array(false), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!key.empty());
        data.resize(sizeof(T));
        memcpy(data.data(), &value, sizeof(T));
    }

    template <typename T>
    gguf_kv(const std::string & key, const std::vector<T> & value)
            : key(key), is_array(true), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!key.empty());
        data.resize(value.size()*sizeof(T));
        for (size_t i = 0; i < value.size(); ++i) {
            const T tmp = value[i];
            memcpy(data.data() + i*sizeof(T), &tmp, sizeof(T));
        }
    }

    gguf_kv(const std::string & key, const std::string & value)
            : key(key), is_array(false), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!key.empty());
        data_string.push_back(value);
    }

    gguf_kv(const std::string & key, const std::vector<std::string> & value)
            : key(key), is_array(true), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!key.empty());
        data_string = value;
    }

    const std::string & get_key() const {
        return key;
    }

    const enum gguf_type & get_type() const {
        return type;
    }

    size_t get_ne() const {
        if (type == GGUF_TYPE_STRING) {
            const size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            return ne;
        }
        const size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        const size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        return ne;
    }

    template <typename T>
    const T & get_val(const size_t i = 0) const {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        if constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            return data_string[i];
        }
        const size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        return reinterpret_cast<const T *>(data.data())[i];
    }

    void cast(const enum gguf_type new_type) {
        const size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

struct gguf_tensor_info {
    struct ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

struct gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<struct gguf_kv> kv;
    std::vector<struct gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    void * data = nullptr;
};

struct gguf_reader {
    gguf_reader(FILE * file) : file(file) {
        // read the remaining bytes once and update on each read
        nbytes_remain = file_remain(file);
    }

    // helper for remaining bytes in a file
    static uint64_t file_remain(FILE * file) {
        const int64_t cur = gguf_ftell(file);
        if (cur < 0) {
            return 0;
        }
        if (gguf_fseek(file, 0, SEEK_END) != 0) {
            gguf_fseek(file, cur, SEEK_SET);

            return 0;
        }
        const int64_t end = gguf_ftell(file);
        if (end < 0) {
            gguf_fseek(file, cur, SEEK_SET);

            return 0;
        }
        gguf_fseek(file, cur, SEEK_SET);
        return static_cast<uint64_t>(end - cur);
    }

    template <typename T>
    bool read(T & dst) const {
        const size_t size = sizeof(dst);
        if (nbytes_remain < size) {
            return false;
        }
        const size_t nread = fread(&dst, 1, size, file);
        nbytes_remain -= nread;
        return nread == size;
    }

    template <typename T>
    bool read(std::vector<T> & dst, const size_t n) const {
        if (n > GGUF_MAX_ARRAY_ELEMENTS) {
            return false;
        }
        if constexpr (std::is_same<T, std::string>::value) {
            // strings are prefixed with their length, so we need to account for that
            if (n > SIZE_MAX / sizeof(uint64_t)) {
                return false;
            }
            if (nbytes_remain < n * sizeof(uint64_t)) {
                return false;
            }
        } else {
            if (n > SIZE_MAX / sizeof(T)) {
                return false;
            }
            if (nbytes_remain < n * sizeof(T)) {
                return false;
            }
        }
        dst.resize(n);
        for (size_t i = 0; i < dst.size(); ++i) {
            if constexpr (std::is_same<T, bool>::value) {
                bool tmp;
                if (!read(tmp)) {
                    return false;
                }
                dst[i] = tmp;
            } else {
                if (!read(dst[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool read(bool & dst) const {
        int8_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = tmp != 0;
        return true;
    }

    bool read(enum ggml_type & dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = ggml_type(tmp);
        return true;
    }

    bool read(enum gguf_type & dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = gguf_type(tmp);
        return true;
    }

    bool read(std::string & dst) const {
        uint64_t size = 0;
        if (!read(size)) {
            return false;
        }
        if (size > GGUF_MAX_STRING_LENGTH) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, size, (uint64_t) GGUF_MAX_STRING_LENGTH);
            return false;
        }
        if (size > nbytes_remain) {
            GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, size, nbytes_remain);
            return false;
        }
        dst.resize(static_cast<size_t>(size));
        const size_t nread = fread(dst.data(), 1, size, file);
        nbytes_remain -= nread;
        return nread == size;
    }

    bool read(void * dst, const size_t size) const {
        if (size > nbytes_remain) {
            return false;
        }
        const size_t nread = fread(dst, 1, size, file);
        nbytes_remain -= nread;
        return nread == size;
    }

private:
    FILE * file;

    mutable uint64_t nbytes_remain;
};

struct gguf_context * gguf_init_empty(void) {
    return new gguf_context;
}

template<typename T>
bool gguf_read_emplace_helper(const struct gguf_reader & gr, std::vector<struct gguf_kv> & kv, const std::string & key, const bool is_array, const size_t n) {
    if (is_array) {
        std::vector<T> value;
        try {
            if (!gr.read(value, n)) {
                return false;
            }
        } catch (std::length_error &) {
            GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, key.c_str());
            return false;
        } catch (std::bad_alloc &) {
            GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, key.c_str());
            return false;
        }
        kv.emplace_back(key, value);
    } else {
        T value;
        if (!gr.read(value)) {
            return false;
        }
        kv.emplace_back(key, value);
    }
    return true;
}

struct gguf_context * gguf_init_from_file_ptr(FILE * file, struct gguf_init_params params) {
    if (!file) {
        return nullptr;
    }

    const struct gguf_reader gr(file);
    struct gguf_context * ctx = new gguf_context;

    bool ok = true;

    // file magic
    {
        std::vector<char> magic;
        ok = ok && gr.read(magic, 4);

        if (!ok) {
            GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }

        for (uint32_t i = 0; i < magic.size(); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                char c0 = isprint(magic[0]) ? magic[0] : '?';
                char c1 = isprint(magic[1]) ? magic[1] : '?';
                char c2 = isprint(magic[2]) ? magic[2] : '?';
                char c3 = isprint(magic[3]) ? magic[3] : '?';
                GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1, c2, c3);
                gguf_free(ctx);
                return nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    if (ok && gr.read(ctx->version)) {
        if (ok && ctx->version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx->version);
            ok = false;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        if (ok && (ctx->version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = false;
        }

        if (ok && ctx->version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = false;
        }
        if (ok && ctx->version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok && gr.read(n_tensors)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/sizeof(gguf_tensor_info))) {
            GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/sizeof(gguf_tensor_info));
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok && gr.read(n_kv)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_kv < 0 || n_kv > int64_t(SIZE_MAX/sizeof(gguf_kv))) {
            GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/sizeof(gguf_kv));
            ok = false;
        }
    } else {
        ok = false;
    }

    if (!ok) {
        GGML_LOG_ERROR("%s: failed to read header\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }

    // KV pairs
    {
        for (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string key;
            gguf_type   type     = gguf_type(-1);
            bool        is_array = false;
            uint64_t    n        = 1;

            try {
                ok = ok && gr.read(key);
            } catch (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            } catch (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            for (size_t j = 0; ok && j < ctx->kv.size(); ++j) {
                if (key == ctx->kv[j].key) {
                    GGML_LOG_ERROR("%s: duplicate key '%s' for tensors %zu and %" PRIi64 " \n", __func__, key.c_str(), j, i);
                    ok = false;
                }
            }
            if (!ok) {
                break;
            }

            ok = ok && gr.read(type);
            if (type == GGUF_TYPE_ARRAY) {
                is_array = true;
                ok = ok && gr.read(type);
                ok = ok && gr.read(n);
            }
            if (!ok) {
                break;
            }

            switch (type) {
                case GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<float>      (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<bool>       (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<double>     (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_ARRAY:
                default:
                    {
                        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(), type);
                        ok = false;
                    } break;
            }
        }

        if (!ok) {
            GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        const int alignment_idx = gguf_find_key(ctx, GGUF_KEY_GENERAL_ALIGNMENT);
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        if (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            return nullptr;
        }
    }

    // read the tensor info
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        struct gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            try {
                ok = ok && gr.read(name);
            } catch (std::length_error &) {
                GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            } catch (std::bad_alloc &) {
                GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            if (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = false;
                break;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            for (int64_t j = 0; ok && j < i; ++j) {
                if (strcmp(info.t.name, ctx->info[j].t.name) == 0) {
                    GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensors %" PRIi64 " and %" PRIi64 "\n", __func__, info.t.name, j, i);
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            break;
        }

        // tensor shape
        {
            uint32_t n_dims = 0;
            ok = ok && gr.read(n_dims);
            if (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = false;
                break;
            }
            for (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                if (j < n_dims) {
                    ok = ok && gr.read(info.t.ne[j]);
                }

                // check that all ne are non-negative
                if (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = false;
                    break;
                }
            }

            // check that the total number of elements is representable
            if (ok && ((INT64_MAX/info.t.ne[1] <= info.t.ne[0]) ||
                       (INT64_MAX/info.t.ne[2] <= info.t.ne[0]*info.t.ne[1]) ||
                       (INT64_MAX/info.t.ne[3] <= info.t.ne[0]*info.t.ne[1]*info.t.ne[2]))) {

                GGML_LOG_ERROR("%s: total number of elements in tensor '%s' with shape "
                    "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= %" PRIi64 "\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }

        // tensor type
        {
            ok = ok && gr.read(info.t.type);

            // check that tensor type is within defined range
            if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_COUNT);
                ok = false;
                break;
            }
            const size_t  type_size = ggml_type_size(info.t.type);
            const int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            if (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (int) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = false;
                break;
            }

            // check that the size of the tensor in bytes is representable
            if (ok && uint64_t(ggml_nelements(&info.t)/ggml_blck_size(info.t.type)) > SIZE_MAX/ggml_type_size(info.t.type)) {
                GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") has a size in bytes > %zu\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], SIZE_MAX);
                ok = false;
                break;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        if (!ok) {
            break;
        }

        // tensor data offset within buffer
        ok = ok && gr.read(info.offset);

        ctx->info.push_back(info);
    }

    if (!ok) {
        GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    if (gguf_fseek(file, GGML_PAD(gguf_ftell(file), ctx->alignment), SEEK_SET) != 0) {
        GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = gguf_ftell(file);

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const gguf_tensor_info & ti = ctx->info[i];
            if (ti.offset != ctx->size) {
                GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
            size_t padded_size = GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);
            if (SIZE_MAX - ctx->size < padded_size) {
                GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                    __func__, ti.t.name, ctx->size, padded_size);
                gguf_free(ctx);
                return nullptr;
            }
            ctx->size += padded_size;
        }
    }

    // load the tensor data only if requested
    if (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        size_t mem_size = 0;
        if (params.no_alloc) {
            if (n_tensors != 0 && SIZE_MAX / n_tensors < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }

            const size_t overhead = n_tensors * ggml_tensor_overhead();

            mem_size = overhead;
        } else {
            if ((n_tensors + 1) != 0 && SIZE_MAX / (n_tensors + 1) < ggml_tensor_overhead()) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }

            const size_t overhead = (n_tensors + 1) * ggml_tensor_overhead();

            if (SIZE_MAX - overhead < ctx->size) {
                GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }

            mem_size = overhead + ctx->size;
        }

        struct ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        if (*params.ctx == nullptr) {
            GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }

        struct ggml_context * ctx_data = *params.ctx;

        struct ggml_tensor * data = nullptr;

        if (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            if (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.read(data->data, ctx->size);

            if (!ok) {
                GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                return nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const struct gguf_tensor_info & info = ctx->info[i];

            struct ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            if (!ok) {
                break;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            if (!params.no_alloc) {
                cur->data = (char *) data->data + info.offset;
            }
        }

        if (!ok) {
            GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            return nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    return ctx;
}

struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    if (!file) {
        GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, fname, strerror(errno));
        return nullptr;
    }

    struct gguf_context * result = gguf_init_from_file_ptr(file, params);
    fclose(file);
    return result;
}

void gguf_free(struct gguf_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

const char * gguf_type_name(enum gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    return it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(const struct gguf_context * ctx) {
    return ctx->version;
}

size_t gguf_get_alignment(const struct gguf_context * ctx) {
    return ctx->alignment;
}

size_t gguf_get_data_offset(const struct gguf_context * ctx) {
    return ctx->offset;
}

int64_t gguf_get_n_kv(const struct gguf_context * ctx) {
    return ctx->kv.size();
}

int64_t gguf_find_key(const struct gguf_context * ctx, const char * key) {
    // return -1 if key not found
    int64_t keyfound = -1;

    const int64_t n_kv = gguf_get_n_kv(ctx);

    for (int64_t i = 0; i < n_kv; ++i) {
        if (strcmp(key, gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

const char * gguf_get_key(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].get_key().c_str();
}

enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[key_id].get_type();
}

enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].is_array);
    return ctx->kv[key_id].get_type();
}

const void * gguf_get_arr_data(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_STRING);
    return ctx->kv[key_id].data.data();
}

const char * gguf_get_arr_str(const struct gguf_context * ctx, int64_t key_id, size_t i) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_STRING);
    return ctx->kv[key_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));

    if (ctx->kv[key_id].type == GGUF_TYPE_STRING) {
        return ctx->kv[key_id].data_string.size();
    }

    const size_t type_size = gguf_type_size(ctx->kv[key_id].type);
    GGML_ASSERT(ctx->kv[key_id].data.size() % type_size == 0);
    return ctx->kv[key_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int32_t>();
}

float gguf_get_val_f32(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<float>();
}

uint64_t gguf_get_val_u64(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int64_t>();
}

double gguf_get_val_f64(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<double>();
}

bool gguf_get_val_bool(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<bool>();
}

const char * gguf_get_val_str(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<std::string>().c_str();
}

const void * gguf_get_val_data(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_STRING);
    return ctx->kv[key_id].data.data();
}

int64_t gguf_get_n_tensors(const struct gguf_context * ctx) {
    return ctx->info.size();
}

int64_t gguf_find_tensor(const struct gguf_context * ctx, const char * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors; ++i) {
        if (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            break;
        }
    }

    return tensor_id;
}

size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    return ctx->info[tensor_id].offset;
}

const char * gguf_get_tensor_name(const struct gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    return ctx->info[tensor_id].t.name;
}

enum ggml_type gguf_get_tensor_type(const struct gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    return ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(const struct gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    return ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_key(struct gguf_context * ctx, const char * key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + key_id);
    }
    return key_id;
}

template<typename T>
static void gguf_check_reserved_keys(const std::string & key, const T val) {
    if (key == GGUF_KEY_GENERAL_ALIGNMENT) {
        if constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_KEY_GENERAL_ALIGNMENT " must be power of 2");
        } else {
            GGML_UNUSED(val);
            GGML_ABORT(GGUF_KEY_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

void gguf_set_val_u8(struct gguf_context * ctx, const char * key, uint8_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i8(struct gguf_context * ctx, const char * key, int8_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u16(struct gguf_context * ctx, const char * key, uint16_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i16(struct gguf_context * ctx, const char * key, int16_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u32(struct gguf_context * ctx, const char * key, uint32_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i32(struct gguf_context * ctx, const char * key, int32_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_f32(struct gguf_context * ctx, const char * key, float val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u64(struct gguf_context * ctx, const char * key, uint64_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i64(struct gguf_context * ctx, const char * key, int64_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_f64(struct gguf_context * ctx, const char * key, double val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_str(struct gguf_context * ctx, const char * key, const char * val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, std::string(val));
}

void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, size_t n) {
    gguf_check_reserved_keys(key, data);
    gguf_remove_key(ctx, key);

    const size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    if (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(key, tmp);
    ctx->kv.back().cast(type);
}

void gguf_set_arr_str(struct gguf_context * ctx, const char * key, const char ** data, size_t n) {
    gguf_check_reserved_keys(key, data);
    gguf_remove_key(ctx, key);

    std::vector<std::string> tmp(n);
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(key, tmp);
}

// set or add KV pairs from another context
void gguf_set_kv(struct gguf_context * ctx, const struct gguf_context * src) {
    const int64_t n_kv = gguf_get_n_kv(src);
    for (int64_t i = 0; i < n_kv; ++i) {
        const struct gguf_kv & kv = src->kv[i];

        if (!kv.is_array) {
            switch (kv.get_type()) {
                case GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_key().c_str(), kv.get_val<uint8_t>());             break;
                case GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_key().c_str(), kv.get_val<int8_t>());              break;
                case GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_key().c_str(), kv.get_val<uint16_t>());            break;
                case GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_key().c_str(), kv.get_val<int16_t>());             break;
                case GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_key().c_str(), kv.get_val<uint32_t>());            break;
                case GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_key().c_str(), kv.get_val<int32_t>());             break;
                case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_key().c_str(), kv.get_val<float>());               break;
                case GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_key().c_str(), kv.get_val<uint64_t>());            break;
                case GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_key().c_str(), kv.get_val<int64_t>());             break;
                case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_key().c_str(), kv.get_val<double>());              break;
                case GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_key().c_str(), kv.get_val<bool>());                break;
                case GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_key().c_str(), kv.get_val<std::string>().c_str()); break;
                case GGUF_TYPE_ARRAY:
                default: GGML_ABORT("invalid type");
            }
            continue;
        }

        const size_t ne = kv.get_ne();

        switch (kv.get_type()) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
            case GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_key().c_str(), kv.get_type(), kv.data.data(), ne);
            } break;
            case GGUF_TYPE_STRING: {
                std::vector<const char *> tmp(ne);
                for (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_key().c_str(), tmp.data(), ne);
            } break;
            case GGUF_TYPE_ARRAY:
            default: GGML_ABORT("invalid type");
        }
    }
}

void gguf_add_tensor(
             struct gguf_context * ctx,
        const struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    if (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    struct gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type) {
    const int64_t tensor_id = gguf_find_tensor(ctx, name);
    if (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    struct ggml_tensor * tensor = &ctx->info[tensor_id].t;
    const size_t  type_size = ggml_type_size(type);
    const int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data) {
    const int64_t tensor_id = gguf_find_tensor(ctx, name);
    if (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (void *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

struct gguf_writer_base {
    size_t written_bytes {0u};

    ~gguf_writer_base(void) = default;

    // we bet on devirtualization
    virtual void write(int8_t val) = 0;
    virtual void write(const std::vector<int8_t> & val) = 0;
    virtual void write_tensor_data(const struct gguf_tensor_info & info, size_t offset_data, size_t alignment) = 0;

    template <typename T>
    void write(const T & val) {
        for (size_t i = 0; i < sizeof(val); ++i) {
            write(reinterpret_cast<const int8_t *>(&val)[i]);
        }
    }

    void write(const bool & val) {
        const int8_t val8 = val ? 1 : 0;
        write(val8);
    }

    void write(const std::string & val) {
        {
            const uint64_t n = val.length();
            write(n);
        }
        for (size_t i = 0; i < val.length(); ++i) {
            write((val.data())[i]);
        }
    }

    void write(const char * val) {
        write(std::string(val));
    }

    void write(const enum ggml_type & val) {
        write(int32_t(val));
    }

    void write(const enum gguf_type & val) {
        write(int32_t(val));
    }

    void write(const struct gguf_kv & kv) {
        const uint64_t ne = kv.get_ne();

        write(kv.get_key());

        if (kv.is_array) {
            write(GGUF_TYPE_ARRAY);
            write(kv.get_type());
            write(ne);
        } else {
            write(kv.get_type());
        }

        switch (kv.get_type()) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64: {
                write(kv.data);
            } break;
            case GGUF_TYPE_BOOL: {
                for (size_t i = 0; i < ne; ++i) {
                    write(kv.get_val<bool>(i));
                }
            } break;
            case GGUF_TYPE_STRING: {
                for (size_t i = 0; i < ne; ++i) {
                    write(kv.get_val<std::string>(i));
                }
            } break;
            case GGUF_TYPE_ARRAY:
            default: GGML_ABORT("invalid type");
        }
    }

    void write_tensor_meta(const struct gguf_tensor_info & info) {
        write(info.t.name);

        const uint32_t n_dims = ggml_n_dims(&info.t);
        write(n_dims);

        for (uint32_t j = 0; j < n_dims; ++j) {
            write(info.t.ne[j]);
        }
        write(info.t.type);
        write(info.offset);
    }

    void pad(const size_t alignment) {
        while (written_bytes % alignment != 0) {
            const int8_t zero = 0;
            write(zero);
        }
    }
};

// vector buffer based writer
struct gguf_writer_buf final : public gguf_writer_base {
    std::vector<int8_t> & buf;

    gguf_writer_buf(std::vector<int8_t> & buf) : buf(buf) {}

    using gguf_writer_base::write;

    void write(const int8_t val) override {
        buf.push_back(val);
        written_bytes++;
    }

    void write(const std::vector<int8_t> & val) override {
        buf.insert(buf.end(), val.begin(), val.end());
        written_bytes += val.size();
    }

    void write_tensor_data(const struct gguf_tensor_info & info, const size_t offset_data, const size_t alignment) override {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        const size_t offset = buf.size();
        const size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        if (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } else {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }
        written_bytes += nbytes;

        pad(alignment);
    }
};

// file based writer
struct gguf_writer_file final : public gguf_writer_base {
    FILE * file;

    gguf_writer_file(FILE* file) : file(file) {}

    using gguf_writer_base::write;

    void write(const int8_t val) override {
        const auto real_val = static_cast<uint8_t>(val);
        const auto ret = fputc(real_val, file);
        written_bytes++;
        if (ret != real_val) {
            throw std::runtime_error("unexpected fputc result '" + std::to_string(ret) + "' instead of '" + std::to_string((int)real_val) + "'");
        }
    }

    void write(const std::vector<int8_t> & val) override {
        const auto ret = fwrite(val.data(), 1, val.size(), file);
        written_bytes += val.size();
        if (ret != val.size()) {
            throw std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(ret) + "' instead of '" + std::to_string(val.size()) + "'");
        }
    }

    void write_tensor_data(const struct gguf_tensor_info & info, const size_t offset_data, const size_t alignment) override {
        GGML_ASSERT(written_bytes - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        const size_t nbytes = ggml_nbytes(&info.t);

        std::vector<int8_t> buf(nbytes);
        if (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data(), 0, nbytes);
        } else {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data(), info.t.data, nbytes);
        }
        write(buf);

        pad(alignment);
    }
};

template <typename writer_t>
static void gguf_write_out(const struct gguf_context * ctx, writer_t & gw, bool only_meta) {
    const int64_t n_kv      = gguf_get_n_kv(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.write(GGUF_MAGIC[0]);
    gw.write(GGUF_MAGIC[1]);
    gw.write(GGUF_MAGIC[2]);
    gw.write(GGUF_MAGIC[3]);
    gw.write(ctx->version);
    gw.write(n_tensors);
    gw.write(n_kv);

    // write key-value pairs
    for (int64_t i = 0; i < n_kv; ++i) {
        gw.write(ctx->kv[i]);
    }

    // write tensor info
    for (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    if (only_meta) {
        return;
    }

    const size_t offset_data = gw.written_bytes;

    // write tensor data
    for (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

void gguf_write_to_buf(const struct gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta) {
    gguf_writer_buf gw(buf);
    gguf_write_out(ctx, gw, only_meta);
}

bool gguf_write_to_file_ptr(const struct gguf_context * ctx, FILE * file, bool only_meta) {
    GGML_ASSERT(file);

    try {
        gguf_writer_file gw(file);
        gguf_write_out(ctx, gw, only_meta);
    } catch (const std::runtime_error& ex) {
        GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, ex.what());
        return false;
    }
    return true;
}

bool gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    if (!file) {
        GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        return false;
    }

    const bool success = gguf_write_to_file_ptr(ctx, file, only_meta);
    if (!success) {
        GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, fname);
    }

    fclose(file);
    return success;
}

size_t gguf_get_meta_size(const struct gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    return buf.size();
}

void gguf_get_meta_data(const struct gguf_context * ctx, void * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    memcpy(data, buf.data(), buf.size());
}
