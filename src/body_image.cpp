// Affine sampling conventions adapted from OpenCV 4.11 imgwarp.cpp.
// See THIRD_PARTY_NOTICES.md and LICENSES/OpenCV-imgwarp.txt.
#include "error.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

struct s3d_body_image_result {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> rgb;
    std::vector<float> normalized;
};

namespace {
using s3d::boundary;
using s3d::require;
int64_t rounded(double value) {
    // OpenCV uses cvRound (nearest-even in this reference environment).
    // Refuse extremes that would overflow its signed fixed-point coordinate map.
    require(std::isfinite(value) && std::abs(value) < 1073741824.0,
            "crop coordinates exceed supported affine sampling range");
    return static_cast<int64_t>(std::nearbyint(value));
}

std::array<double, 6> inverse_affine(const double *m) {
    const double det = m[0] * m[4] - m[1] * m[3];
    require(std::isfinite(det) && det != 0, "singular affine transform");
    const double d = 1.0 / det;
    std::array<double, 6> r{m[4]*d, -m[1]*d, 0, -m[3]*d, m[0]*d, 0};
    r[2] = -r[0]*m[2] - r[1]*m[5];
    r[5] = -r[3]*m[2] - r[4]*m[5];
    return r;
}
}

s3d_status s3d_body_image_prepare(const s3d_body_crop_result *crop,
                                 const uint8_t *rgb, uint64_t capacity,
                                 uint32_t width, uint32_t height, uint64_t stride,
                                 s3d_body_image_result **out, char *e, uint64_t n) {
    if (out) *out = nullptr;
    return boundary(e, n, [&] {
        require(crop && rgb && out, "crop, image and output pointers are required");
        require(width && height && width < 32767 && height < 32767,
                "input dimensions must be in [1,32766]");
        const uint64_t row_bytes = static_cast<uint64_t>(width) * 3;
        require(stride >= row_bytes && stride <= std::numeric_limits<size_t>::max(), "invalid RGB row stride");
        require(height == 1 || stride <= (UINT64_MAX-row_bytes)/(height-1), "image span overflow");
        const uint64_t span = stride*(height-1) + row_bytes;
        require(span <= capacity && span <= std::numeric_limits<size_t>::max(), "RGB input buffer is too small");
        auto result = std::make_unique<s3d_body_image_result>();
        require(s3d_body_crop_get_size(crop, &result->width, &result->height, nullptr, 0) == S3D_OK,
                "invalid crop size");
        const uint64_t pixels = static_cast<uint64_t>(result->width)*result->height;
        require(pixels && pixels <= 16*1024*1024, "crop exceeds 16M-pixel output limit");
        const double *matrix = nullptr; uint64_t count = 0;
        require(s3d_body_crop_get_affine(crop, &matrix, &count, nullptr, 0) == S3D_OK && count == 6,
                "invalid crop affine");
        const auto m = inverse_affine(matrix);
        result->rgb.resize(pixels*3);
        result->normalized.resize(pixels*3);
        std::vector<int64_t> dx(result->width), dy(result->width);
        for (uint32_t x = 0; x < result->width; ++x) {
            dx[x] = rounded(m[0]*x*1024); dy[x] = rounded(m[3]*x*1024);
        }
        auto sample = [&](int64_t x, int64_t y, int channel) -> int {
            if (x < 0 || y < 0 || x >= width || y >= height) return 0;
            return rgb[static_cast<uint64_t>(y)*stride+static_cast<uint64_t>(x)*3+channel];
        };
        const auto *gather_flag=std::getenv("SAM3D_IMAGE_GATHER");
        const bool gather=gather_flag && std::strcmp(gather_flag,"1")==0;
        const std::array<uint8_t,3> border{};
        auto source_pixel=[&](int64_t x,int64_t y)->const uint8_t * {
            if(x<0 || y<0 || x>=width || y>=height)return border.data();
            return rgb+static_cast<uint64_t>(y)*stride+static_cast<uint64_t>(x)*3;
        };
        constexpr std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
        constexpr std::array<float, 3> stddev{0.229f, 0.224f, 0.225f};
        for (uint32_t y = 0; y < result->height; ++y) {
            const int64_t row_x = rounded((m[1]*y+m[2])*1024)+16;
            const int64_t row_y = rounded((m[4]*y+m[5])*1024)+16;
            for (uint32_t x = 0; x < result->width; ++x) {
                const int64_t map_x = (row_x+dx[x]) >> 5;
                const int64_t map_y = (row_y+dy[x]) >> 5;
                const int64_t sx = std::clamp<int64_t>(map_x >> 5, -32768, 32767);
                const int64_t sy = std::clamp<int64_t>(map_y >> 5, -32768, 32767);
                const int fx = static_cast<int>(map_x & 31), fy = static_cast<int>(map_y & 31);
                const uint64_t pixel = static_cast<uint64_t>(y)*result->width+x;
                if(gather) {
                    // Bounds and addresses are identical for all three channels.
                    // Share the four gathers, preserving fixed-point integer
                    // interpolation, zero borders and the original F32 steps.
                    const uint8_t *a,*b,*c,*d;
                    if(sx>=0 && sy>=0 && sx+1<width && sy+1<height) {
                        a=rgb+static_cast<uint64_t>(sy)*stride+static_cast<uint64_t>(sx)*3;
                        b=a+3;c=a+stride;d=c+3;
                    } else {
                        a=source_pixel(sx,sy);b=source_pixel(sx+1,sy);
                        c=source_pixel(sx,sy+1);d=source_pixel(sx+1,sy+1);
                    }
                    const int wa=(32-fx)*(32-fy),wb=fx*(32-fy),wc=(32-fx)*fy,wd=fx*fy;
                    for(int channel=0;channel<3;++channel) {
                        const int sum=a[channel]*wa+b[channel]*wb+c[channel]*wc+d[channel]*wd;
                        const auto value=static_cast<uint8_t>((sum+512)>>10);
                        result->rgb[pixel*3+channel]=value;
                        const float unit=static_cast<float>(value)/255.0f;
                        result->normalized[channel*pixels+pixel]=(unit-mean[channel])/stddev[channel];
                    }
                    continue;
                }
                for (int c = 0; c < 3; ++c) {
                    const int sum = sample(sx,sy,c)*(32-fx)*(32-fy) + sample(sx+1,sy,c)*fx*(32-fy)
                                  + sample(sx,sy+1,c)*(32-fx)*fy + sample(sx+1,sy+1,c)*fx*fy;
                    const auto value = static_cast<uint8_t>((sum+512) >> 10);
                    result->rgb[pixel*3+c] = value;
                    const float unit = static_cast<float>(value)/255.0f;
                    result->normalized[c*pixels+pixel] = (unit-mean[c])/stddev[c];
                }
            }
        }
        *out = result.release();
    });
}
void s3d_body_image_result_free(s3d_body_image_result *result) { delete result; }
s3d_status s3d_body_image_get_rgb(const s3d_body_image_result *r, const uint8_t **data, uint64_t *count,
                                 char *e, uint64_t n) {
    if (data) *data = nullptr;
    if (count) *count = 0;
    return boundary(e,n,[&] { require(r && data && count,"result and output pointers are required");
        *data=r->rgb.data(); *count=r->rgb.size(); });
}
s3d_status s3d_body_image_get_normalized(const s3d_body_image_result *r, const float **data, uint64_t *count,
                                        char *e, uint64_t n) {
    if (data) *data = nullptr;
    if (count) *count = 0;
    return boundary(e,n,[&] { require(r && data && count,"result and output pointers are required");
        *data=r->normalized.data(); *count=r->normalized.size(); });
}
s3d_status s3d_body_image_get_size(const s3d_body_image_result *r, uint32_t *w, uint32_t *h,
                                 char *e, uint64_t n) {
    if (w) *w=0;
    if (h) *h=0;
    return boundary(e,n,[&] { require(r && w && h,"result and output pointers are required");
        *w=r->width; *h=r->height; });
}
