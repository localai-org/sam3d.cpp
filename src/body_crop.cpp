#include "sam3d.h"
#include "error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>

struct s3d_body_crop_request {
    std::array<float, 4> box{};
    bool has_box = false;
    float padding = 1.25f;
    float prior_aspect = 0.75f;
    float rotation = 0;
    uint32_t width = 512, height = 512;
};

struct s3d_body_crop_result {
    std::array<float, 2> center{}, padded_scale{}, prior_scale{}, scale{};
    std::array<double, 6> affine{};
    uint32_t width = 0, height = 0;
};

namespace {
using s3d::boundary;
using s3d::require;
bool positive(float x) { return std::isfinite(x) && x > 0; }

std::array<float, 2> expand(std::array<float, 2> wh, float aspect) {
    // Retain F32 arithmetic and the upstream two-stage aspect expansion.
    return wh[0] > wh[1] * aspect ? std::array{wh[0], wh[0] / aspect}
                                      : std::array{wh[1] * aspect, wh[1]};
}

s3d_body_crop_result compute(const s3d_body_crop_request &r) {
    require(r.has_box, "set a valid box before computing crop geometry");
    s3d_body_crop_result result;
    result.width = r.width; result.height = r.height;
    for (int i = 0; i < 2; ++i) {
        result.center[i] = (r.box[i] + r.box[i + 2]) * 0.5f;
        result.padded_scale[i] = (r.box[i + 2] - r.box[i]) * r.padding;
        require(std::isfinite(result.center[i]) && positive(result.padded_scale[i]),
                "box center/scale overflow or degeneracy");
    }
    result.prior_scale = expand(result.padded_scale, r.prior_aspect);
    result.scale = expand(result.prior_scale, static_cast<float>(r.width) / r.height);
    for (float x : result.scale) require(positive(x), "aspect expansion overflow");
    // Upstream get_warp_matrix constructs three F32 source/destination points
    // before OpenCV solves a double-precision affine system. Preserve that
    // rounding boundary; a single idealized scale/rotation formula differs.
    using point = std::array<float, 2>;
    const double rad = static_cast<double>(r.rotation) * (std::numbers::pi / 180.0);
    const double half = static_cast<double>(result.scale[0] * -0.5f);
    const point src0 = result.center;
    const point src1{static_cast<float>(src0[0] - std::sin(rad) * half),
                     static_cast<float>(src0[1] + std::cos(rad) * half)};
    const point delta{src0[0] - src1[0], src0[1] - src1[1]};
    const point src2{src1[0] - delta[1], src1[1] + delta[0]};
    const point dst0{r.width * 0.5f, r.height * 0.5f};
    const point dst1{dst0[0], static_cast<float>(r.height * 0.5 - r.width * 0.5)};
    const point dst2{static_cast<float>(dst1[0] - (dst0[1] - dst1[1])),
                     static_cast<float>(dst1[1] + (dst0[0] - dst1[0]))};
    // Match getAffineTransform's 6x6 partial-pivot LU operation order. An
    // algebraically equivalent direct 2x2 solve changes the final bits, which
    // can cross OpenCV's fixed-point pixel-coordinate rounding boundaries.
    // Adapted from OpenCV matrix_decomp.cpp LUImpl (see third-party notices).
    const std::array<point,3> src{src0,src1,src2}, dst{dst0,dst1,dst2};
    double a[6][6]{}, b[6]{};
    for (int i=0;i<3;++i) {
        for (int axis=0;axis<2;++axis) {
            a[2*i+axis][3*axis]=src[i][0];
            a[2*i+axis][3*axis+1]=src[i][1];
            a[2*i+axis][3*axis+2]=1;
            b[2*i+axis]=dst[i][axis];
        }
    }
    for (int i=0;i<6;++i) {
        int pivot=i;
        for(int j=i+1;j<6;++j) if(std::abs(a[j][i])>std::abs(a[pivot][i])) pivot=j;
        require(std::isfinite(a[pivot][i]) && std::abs(a[pivot][i])>=100*std::numeric_limits<double>::epsilon(),
                "degenerate affine source points");
        if(pivot!=i) {
            for(int j=i;j<6;++j) std::swap(a[i][j],a[pivot][j]);
            std::swap(b[i],b[pivot]);
        }
        const double d=-1/a[i][i];
        for(int j=i+1;j<6;++j) {
            const double alpha=a[j][i]*d;
            for(int k=i+1;k<6;++k) a[j][k]+=alpha*a[i][k];
            b[j]+=alpha*b[i];
        }
    }
    for(int i=5;i>=0;--i) {
        double value=b[i];
        for(int k=i+1;k<6;++k) value-=a[i][k]*b[k];
        b[i]=value/a[i][i];
    }
    std::copy_n(b,6,result.affine.begin());
    for (double x : result.affine) require(std::isfinite(x), "affine transform overflow");
    return result;
}

template<class T, size_t N>
s3d_status get_buffer(const s3d_body_crop_result *result, const T **data, uint64_t *count,
                     char *error, uint64_t capacity,
                     std::array<T, N> s3d_body_crop_result::* member) noexcept {
    if (data) *data = nullptr;
    if (count) *count = 0;
    return boundary(error, capacity, [&] {
        require(result && data && count, "result and output pointers are required");
        *data = (result->*member).data(); *count = N;
    });
}
}

uint32_t s3d_abi_version(void) { return 1; }
s3d_status s3d_body_crop_request_create(s3d_body_crop_request **out, char *e, uint64_t n) {
    if (out) *out = nullptr;
    return boundary(e, n, [&] {
        require(out != nullptr, "output pointer is required"); *out = new s3d_body_crop_request;
    });
}
void s3d_body_crop_request_free(s3d_body_crop_request *r) { delete r; }
s3d_status s3d_body_crop_request_set_box(s3d_body_crop_request *r, const float *box,
                                       uint64_t count, char *e, uint64_t n) {
    return boundary(e, n, [&] {
        require(r && box && count == 4, "request and four box coordinates are required");
        for (int i = 0; i < 4; ++i) require(std::isfinite(box[i]), "box must be finite");
        require(box[2] > box[0] && box[3] > box[1], "box must have positive width/height");
        std::copy_n(box, 4, r->box.begin()); r->has_box = true;
    });
}
s3d_status s3d_body_crop_request_set_padding(s3d_body_crop_request *r, float x, char *e, uint64_t n) {
    return boundary(e, n, [&] {
        require(r && positive(x), "padding must be finite and positive"); r->padding = x;
    });
}
s3d_status s3d_body_crop_request_set_prior_aspect(s3d_body_crop_request *r, float x, char *e, uint64_t n) {
    return boundary(e, n, [&] {
        require(r && positive(x), "prior aspect must be finite and positive"); r->prior_aspect = x;
    });
}
s3d_status s3d_body_crop_request_set_output_size(s3d_body_crop_request *r, uint32_t w, uint32_t h,
                                               char *e, uint64_t n) {
    return boundary(e, n, [&] {
        require(r && w && h && w <= 16384 && h <= 16384, "output dimensions must be in [1,16384]");
        r->width = w; r->height = h;
    });
}
s3d_status s3d_body_crop_request_set_rotation(s3d_body_crop_request *r, float x, char *e, uint64_t n) {
    return boundary(e, n, [&] {
        require(r && std::isfinite(x) && std::abs(x) <= 360, "rotation must be finite, within +/-360 degrees");
        r->rotation = x;
    });
}
s3d_status s3d_body_crop_compute(const s3d_body_crop_request *r, s3d_body_crop_result **out,
                                char *e, uint64_t n) {
    if (out) *out = nullptr;
    return boundary(e, n, [&] {
        require(r && out, "request and output pointer are required");
        auto result = std::make_unique<s3d_body_crop_result>(compute(*r));
        *out = result.release();
    });
}
void s3d_body_crop_result_free(s3d_body_crop_result *r) { delete r; }
s3d_status s3d_body_crop_get_size(const s3d_body_crop_result *r, uint32_t *width, uint32_t *height,
                                 char *e, uint64_t n) {
    if (width) *width = 0;
    if (height) *height = 0;
    return boundary(e, n, [&] {
        require(r && width && height, "result and output pointers are required");
        *width = r->width; *height = r->height;
    });
}
#define S3D_CROP_GETTER(NAME, TYPE, MEMBER) \
 s3d_status NAME(const s3d_body_crop_result *r, const TYPE **d, uint64_t *c, char *e, uint64_t n) { \
     return get_buffer(r, d, c, e, n, &s3d_body_crop_result::MEMBER); \
 }
S3D_CROP_GETTER(s3d_body_crop_get_center, float, center)
S3D_CROP_GETTER(s3d_body_crop_get_padded_scale, float, padded_scale)
S3D_CROP_GETTER(s3d_body_crop_get_prior_scale, float, prior_scale)
S3D_CROP_GETTER(s3d_body_crop_get_scale, float, scale)
S3D_CROP_GETTER(s3d_body_crop_get_affine, double, affine)
#undef S3D_CROP_GETTER
