#ifndef SAM3D_H
#define SAM3D_H

#include <stdint.h>

#if defined(_WIN32)
# if defined(SAM3D_BUILDING_LIBRARY)
#  define S3D_API __declspec(dllexport)
# else
#  define S3D_API __declspec(dllimport)
# endif
#else
# define S3D_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t s3d_status;
#define S3D_OK UINT32_C(0)
#define S3D_INVALID_ARGUMENT UINT32_C(1)
#define S3D_OUT_OF_MEMORY UINT32_C(2)
#define S3D_INTERNAL_ERROR UINT32_C(3)

typedef struct s3d_body_crop_request s3d_body_crop_request;
typedef struct s3d_body_crop_result s3d_body_crop_result;
typedef struct s3d_body_image_result s3d_body_image_result;
typedef struct s3d_body_camera_request s3d_body_camera_request;
typedef struct s3d_body_camera_result s3d_body_camera_result;
typedef struct s3d_objects_image_request s3d_objects_image_request;
typedef struct s3d_objects_image_result s3d_objects_image_result;

/* Initial API implements Body and Objects image preparation, not model inference.
 * Every fallible call: caller-owned error buffer, success clears it, failure
 * truncates/NUL-terminates. NULL,0 discards diagnostics; NULL,nonzero is invalid.
 * Callers supply valid live handles and buffers of their declared capacities.
 * Requests copy input. Results own immutable borrowed buffers until result_free.
 * Separate requests may be used concurrently; do not mutate/free a live call.
 * No exception, public struct layout, or global last_error crosses this ABI.
 */
S3D_API uint32_t s3d_abi_version(void);
S3D_API s3d_status s3d_body_crop_request_create(s3d_body_crop_request **out,
                                              char *error, uint64_t error_capacity);
S3D_API void s3d_body_crop_request_free(s3d_body_crop_request *request);
/* Box is x1,y1,x2,y2 in original-image pixel coordinates. May extend off-image. */
S3D_API s3d_status s3d_body_crop_request_set_box(s3d_body_crop_request *request,
                                               const float *xyxy, uint64_t count,
                                               char *error, uint64_t error_capacity);
/* Defaults: body padding 1.25, prior aspect 0.75, output 512x512, rotation 0. */
S3D_API s3d_status s3d_body_crop_request_set_padding(s3d_body_crop_request *request,
                                                   float value, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_crop_request_set_prior_aspect(s3d_body_crop_request *request,
                                                        float value, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_crop_request_set_output_size(s3d_body_crop_request *request,
                                                       uint32_t width, uint32_t height,
                                                       char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_crop_request_set_rotation(s3d_body_crop_request *request,
                                                    float degrees, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_crop_compute(const s3d_body_crop_request *request,
                                       s3d_body_crop_result **out,
                                       char *error, uint64_t error_capacity);
S3D_API void s3d_body_crop_result_free(s3d_body_crop_result *result);
/* Each scale/center buffer has 2 F32 elements; affine has 6 F64 elements in
 * row-major 2x3 order, mapping original-image coordinates into crop pixels. */
S3D_API s3d_status s3d_body_crop_get_center(const s3d_body_crop_result *result,
                                          const float **data, uint64_t *count,
                                          char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_crop_get_padded_scale(const s3d_body_crop_result *result,
                                                const float **data, uint64_t *count,
                                                char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_crop_get_prior_scale(const s3d_body_crop_result *result,
                                               const float **data, uint64_t *count,
                                               char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_crop_get_scale(const s3d_body_crop_result *result,
                                         const float **data, uint64_t *count,
                                         char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_crop_get_affine(const s3d_body_crop_result *result,
                                          const double **data, uint64_t *count,
                                          char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_crop_get_size(const s3d_body_crop_result *result,
                                        uint32_t *width, uint32_t *height,
                                        char *error, uint64_t error_capacity);
/* RGB U8 input, positive row stride in bytes. Input is borrowed only for this
 * call. Uses black-border INTER_LINEAR crop, ToTensor scaling and the selected
 * Body DINOv3 ImageNet mean/std. Maximum output is 16M pixels. No image decoding.
 * Output owns interleaved cropped RGB and planar F32 CHW normalized data. */
S3D_API s3d_status s3d_body_image_prepare(const s3d_body_crop_result *crop,
                                        const uint8_t *rgb, uint64_t capacity,
                                        uint32_t width, uint32_t height, uint64_t row_stride,
                                        s3d_body_image_result **out,
                                        char *error, uint64_t error_capacity);
S3D_API void s3d_body_image_result_free(s3d_body_image_result *result);
S3D_API s3d_status s3d_body_image_get_rgb(const s3d_body_image_result *result,
                                        const uint8_t **data, uint64_t *count,
                                        char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_image_get_normalized(const s3d_body_image_result *result,
                                               const float **data, uint64_t *count,
                                               char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_image_get_size(const s3d_body_image_result *result,
                                         uint32_t *width, uint32_t *height,
                                         char *error, uint64_t error_capacity);

/* Camera conditioning only, not pose inference. Requires explicit intrinsics
 * and original-image size. Supports one axis-aligned square crop, up to 512².
 * Affine metadata is rounded to F32 like upstream prepare_batch before rays.
 * Intrinsics order: fx,fy,cx,cy, in original-image pixels; fx/fy > 0.
 * CLIFF condition defaults to image-center offsets, not principal-point offsets.
 * Invalid setters leave the previous valid request unchanged. */
S3D_API s3d_status s3d_body_camera_request_create(s3d_body_camera_request **out,
                                                 char *error, uint64_t error_capacity);
S3D_API void s3d_body_camera_request_free(s3d_body_camera_request *request);
S3D_API s3d_status s3d_body_camera_request_set_intrinsics(s3d_body_camera_request *request,
    const float *fx_fy_cx_cy, uint64_t count, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_camera_request_set_image_size(s3d_body_camera_request *request,
    uint32_t width, uint32_t height, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_camera_request_set_use_intrinsics_center(s3d_body_camera_request *request,
    uint32_t enabled, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_camera_compute(const s3d_body_camera_request *request,
    const s3d_body_crop_result *crop, s3d_body_camera_result **out,
    char *error, uint64_t error_capacity);
S3D_API void s3d_body_camera_result_free(s3d_body_camera_result *result);
/* Borrowed F32 buffers: rays [2,H,W], and CLIFF [(box_cx-center_x)/fx,
 * (box_cy-center_y)/fx, expanded_box_width/fx]. Note fx is used for ALL three
 * CLIFF components upstream; rays use fx and fy separately. */
S3D_API s3d_status s3d_body_camera_get_rays(const s3d_body_camera_result *result,
    const float **data, uint64_t *count, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_camera_get_condition(const s3d_body_camera_result *result,
    const float **data, uint64_t *count, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_body_camera_get_size(const s3d_body_camera_result *result,
    uint32_t *width, uint32_t *height, char *error, uint64_t error_capacity);

/* Original-default Objects image/mask pipeline, no pointmap or model inference.
 * Defaults: output_side=518, box_size_factor=1.0, padding_factor=0.1.
 * Valid ranges: side [1,1024], box factor [0.25,4], padding [0,1].
 * Invalid setters do not change a request. Values are explicit options, not a
 * claim that an arbitrary checkpoint uses these preprocessing defaults. */
S3D_API s3d_status s3d_objects_image_request_create(s3d_objects_image_request **out,
    char *error, uint64_t error_capacity);
S3D_API void s3d_objects_image_request_free(s3d_objects_image_request *request);
S3D_API s3d_status s3d_objects_image_request_set_output_side(s3d_objects_image_request *request,
    uint32_t side, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_objects_image_request_set_box_factor(s3d_objects_image_request *request,
    double factor, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_objects_image_request_set_padding(s3d_objects_image_request *request,
    double padding, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_objects_image_request_get_options(const s3d_objects_image_request *request,
    uint32_t *side, double *box_factor, double *padding, char *error, uint64_t error_capacity);
/* Caller-decoded RGBA U8, alpha > 0 selects the object. RGB is divided by 255,
 * cropped/padded and resized (bicubic AA); binary masks use nearest neighbor.
 * Output is NOT ImageNet-normalized or clamped to [0,1]. Empty/tiny masks fail.
 * Input dimensions <=4096, intermediate images <=16M pixels. Row stride is in
 * bytes, [4*width,4*width+4096]; final row need not include stride padding.
 * Pixels/request are borrowed only during the call; result owns its buffers. */
S3D_API s3d_status s3d_objects_image_prepare(const s3d_objects_image_request *request,
    const uint8_t *rgba, uint64_t capacity, uint32_t width, uint32_t height,
    uint64_t row_stride, s3d_objects_image_result **out, char *error, uint64_t error_capacity);
S3D_API void s3d_objects_image_result_free(s3d_objects_image_result *result);
#define S3D_OBJECTS_CROP_RGB UINT32_C(0)
#define S3D_OBJECTS_CROP_MASK UINT32_C(1)
#define S3D_OBJECTS_FULL_RGB UINT32_C(2)
#define S3D_OBJECTS_FULL_MASK UINT32_C(3)
/* Borrowed F32 CHW data: RGB [3,side,side], masks [1,side,side]. No public enum
 * or struct layout is required by FFI consumers. Unknown field IDs fail. */
S3D_API s3d_status s3d_objects_image_get_tensor(const s3d_objects_image_result *result,
    uint32_t field, const float **data, uint64_t *count, char *error, uint64_t error_capacity);
S3D_API s3d_status s3d_objects_image_get_side(const s3d_objects_image_result *result,
    uint32_t *side, char *error, uint64_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
