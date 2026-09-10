#ifndef SAM3D_MODEL_H
#define SAM3D_MODEL_H
#include "sam3d.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct s3d_runtime_options s3d_runtime_options;
typedef struct s3d_body_model s3d_body_model;
typedef struct s3d_body_request s3d_body_request;
typedef struct s3d_body_result s3d_body_result;
#define S3D_BACKEND_CPU UINT32_C(1)
#define S3D_BACKEND_VULKAN UINT32_C(2)
#define S3D_BODY_BACKBONE UINT32_C(0)
#define S3D_BODY_POSE_BRANCH UINT32_C(1)
#define S3D_BODY_MHR UINT32_C(2)
#define S3D_DTYPE_F32 UINT32_C(1)
#define S3D_DTYPE_I32 UINT32_C(2)
#define S3D_CAP_BODY_POSE_BRANCH UINT64_C(1)
#define S3D_BACKBONE_F32 UINT32_C(0)
#define S3D_BACKBONE_BF16 UINT32_C(1)

/* Experimental, synchronous, one-person/no-mask Body pose branch; F32 default.
 * NOT hand-crop refinement, automatic detection or Objects inference.
 * All strings are explicit byte spans, no embedded NUL, at most 4096 bytes.
 * Setters copy inputs and preserve previous state on failure. Models copy all
 * options at load; requests/options may then be freed independently of models.
 * GGUFs and backend modules must be trusted and immutable while models live.
 * No backend fallback or implicit download. Vulkan strict F32 settings must be
 * enabled before its first initialization; see docs/API.md.
 * Same-model inference is serialized. Separate models may run concurrently.
 * Do not mutate/free requests/options during calls or free a model in use.
 * This version has no cancellation or configurable graph memory budget.
 * All other error/buffer/valid-pointer rules from sam3d.h apply here too.
 */
S3D_API s3d_status s3d_runtime_options_create(s3d_runtime_options **out,char *error,uint64_t capacity);
S3D_API void s3d_runtime_options_free(s3d_runtime_options *options);
/* Default F32. BF16 rounds only the image encoder, matching upstream's scope;
 * decoder, MHR and public result buffers remain F32. Uses the same F32 GGUFs.
 * Experimental: consult STATUS.md for the current numerical acceptance gate. */
S3D_API s3d_status s3d_runtime_options_set_backbone_precision(s3d_runtime_options *options,
    uint32_t precision,char *error,uint64_t capacity);
S3D_API s3d_status s3d_runtime_options_get_backbone_precision(const s3d_runtime_options *options,
    uint32_t *precision,char *error,uint64_t capacity);
S3D_API s3d_status s3d_body_model_get_backbone_precision(const s3d_body_model *model,
    uint32_t *precision,char *error,uint64_t capacity);
S3D_API s3d_status s3d_runtime_options_set_backend(s3d_runtime_options *options,uint32_t backend,
    const char *module,uint64_t module_bytes,uint32_t device,uint32_t threads,
    const char *expected_description,uint64_t description_bytes,char *error,uint64_t capacity);
S3D_API s3d_status s3d_runtime_options_set_body_file(s3d_runtime_options *options,uint32_t component,
    const char *path,uint64_t path_bytes,char *error,uint64_t capacity);
/* Borrowed string valid until options are changed or freed. */
S3D_API s3d_status s3d_runtime_options_get_body_file(const s3d_runtime_options *options,uint32_t component,
    const char **path,uint64_t *path_bytes,char *error,uint64_t capacity);
S3D_API s3d_status s3d_body_model_load(const s3d_runtime_options *options,s3d_body_model **out,char *error,uint64_t capacity);
S3D_API void s3d_body_model_free(s3d_body_model *model);
S3D_API s3d_status s3d_body_model_get_info(const s3d_body_model *model,const char **device_description,
    uint64_t *capabilities,char *error,uint64_t capacity);
S3D_API s3d_status s3d_body_request_create(s3d_body_request **out,char *error,uint64_t capacity);
S3D_API void s3d_body_request_free(s3d_body_request *request);
/* RGB U8, width/height [1,32766], at most 16 million pixels. Padding is allowed;
 * only actual pixels are copied. Capacity must cover the final pixel, not final
 * row padding. Getter exposes the owned tightly packed copy. */
S3D_API s3d_status s3d_body_request_set_rgb(s3d_body_request *request,const uint8_t *rgb,uint64_t bytes,
    uint32_t width,uint32_t height,uint64_t row_stride,char *error,uint64_t capacity);
S3D_API s3d_status s3d_body_request_get_rgb(const s3d_body_request *request,const uint8_t **rgb,uint64_t *bytes,
    uint32_t *width,uint32_t *height,uint64_t *row_stride,char *error,uint64_t capacity);
/* XYXY pixel box may extend off-image; intrinsics are fx,fy,cx,cy in pixels.
 * Both arrays have exactly four finite floats; focal lengths must be positive. */
S3D_API s3d_status s3d_body_request_set_geometry(s3d_body_request *request,
    const float *box,uint64_t box_count,const float *intrinsics,uint64_t intrinsics_count,char *error,uint64_t capacity);
S3D_API s3d_status s3d_body_model_infer(s3d_body_model *model,const s3d_body_request *request,
    s3d_body_result **out,char *error,uint64_t capacity);
S3D_API void s3d_body_result_free(s3d_body_result *result);
/* Results own every tensor and remain valid after model/request destruction.
 * Borrowed names/data remain valid until result_free. Dimensions are in logical
 * row-major order, including batch dimension; no GGML layout leaks into the ABI.
 * Names, shapes, units and coordinate conventions are specified in docs/API.md.
 */
S3D_API s3d_status s3d_body_result_get_count(const s3d_body_result *result,uint32_t *count,char *error,uint64_t capacity);
S3D_API s3d_status s3d_body_result_get_tensor(const s3d_body_result *result,uint32_t index,
    const char **name,uint32_t *dtype,uint32_t *rank,uint64_t *elements,const void **data,char *error,uint64_t capacity);
S3D_API s3d_status s3d_body_result_get_dimension(const s3d_body_result *result,uint32_t index,uint32_t axis,
    uint64_t *dimension,char *error,uint64_t capacity);
/* Metadata keys: schema, scope, geometry_units, coordinates,
 * joint_rotation_coordinates. Unknown keys fail; values are borrowed strings. */
S3D_API s3d_status s3d_body_result_get_metadata(const s3d_body_result *result,const char *key,uint64_t key_bytes,
    const char **value,uint64_t *value_bytes,char *error,uint64_t capacity);
#ifdef __cplusplus
}
#endif
#endif
