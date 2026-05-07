#pragma once

#include "ggml.h.inc"
#include "ggml-backend.h.inc"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_virtgpu_reg();

#ifdef  __cplusplus
}
#endif
