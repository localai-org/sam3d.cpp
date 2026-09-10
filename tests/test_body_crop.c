#include "sam3d.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main(void) {
    char error[128] = "old error";
    s3d_body_crop_request *request = NULL;
    s3d_body_crop_result *result = NULL, *second = NULL;
    const float box[] = {0, 0, 120, 240};
    const float *values = NULL;
    const double *affine = NULL;
    uint64_t count = 0;
    CHECK(s3d_body_crop_request_create(&request, error, sizeof(error)) == S3D_OK);
    CHECK(error[0] == 0);
    CHECK(s3d_body_crop_compute(request, &result, error, sizeof(error)) == S3D_INVALID_ARGUMENT);
    CHECK(result == NULL && error[0]);
    CHECK(s3d_body_crop_request_set_box(request, box, 3, error, sizeof(error)) == S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_crop_request_set_box(request, box, 4, error, sizeof(error)) == S3D_OK);
    CHECK(s3d_body_crop_request_set_padding(request, NAN, error, sizeof(error)) == S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_crop_request_set_padding(request, -1, error, sizeof(error)) == S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_crop_request_set_prior_aspect(request, 0, error, sizeof(error)) == S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_crop_request_set_output_size(request, 0, 512, error, sizeof(error)) == S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_crop_request_set_rotation(request, INFINITY, error, sizeof(error)) == S3D_INVALID_ARGUMENT);
    CHECK(s3d_body_crop_compute(request, &result, error, sizeof(error)) == S3D_OK);
    CHECK(s3d_body_crop_get_center(result, &values, &count, error, sizeof(error)) == S3D_OK);
    CHECK(count == 2 && values[0] == 60 && values[1] == 120);
    CHECK(s3d_body_crop_get_scale(result, &values, &count, error, sizeof(error)) == S3D_OK);
    CHECK(values[0] == 300 && values[1] == 300);
    CHECK(s3d_body_crop_get_affine(result, &affine, &count, error, sizeof(error)) == S3D_OK);
    CHECK(count == 6 && fabs(affine[0] * 60 + affine[1] * 120 + affine[2] - 256) < 1e-9);
    CHECK(fabs(affine[3] * 60 + affine[4] * 120 + affine[5] - 256) < 1e-9);
    CHECK(s3d_body_crop_request_set_padding(request, 0.9f, NULL, 0) == S3D_OK);
    CHECK(s3d_body_crop_compute(request, &second, NULL, 0) == S3D_OK);
    s3d_body_crop_request_free(request);
    CHECK(s3d_body_crop_get_scale(result, &values, &count, NULL, 0) == S3D_OK);
    CHECK(values[0] == 300); /* Independent immutable result survives request. */
    error[0] = 'X'; error[1] = 'Y';
    CHECK(s3d_body_crop_get_scale(NULL, &values, &count, error, 1) == S3D_INVALID_ARGUMENT);
    CHECK(error[0] == 0 && error[1] == 'Y' && values == NULL && count == 0);
    CHECK(s3d_body_crop_get_scale(result, &values, &count, NULL, 1) == S3D_INVALID_ARGUMENT);
    s3d_body_crop_result_free(result); s3d_body_crop_result_free(second);
    s3d_body_crop_result_free(NULL); s3d_body_crop_request_free(NULL);
    CHECK(s3d_body_crop_request_create(&request, NULL, 0) == S3D_OK);
    const float huge[] = {-FLT_MAX, 0, FLT_MAX, 1};
    CHECK(s3d_body_crop_request_set_box(request, huge, 4, NULL, 0) == S3D_OK);
    CHECK(s3d_body_crop_compute(request, &result, NULL, 0) == S3D_INVALID_ARGUMENT);
    CHECK(result == NULL);
    s3d_body_crop_request_free(request);
    puts("C crop API: input, ownership, error and geometry checks passed");
    return 0;
}
