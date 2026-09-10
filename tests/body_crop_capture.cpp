#include "sam3d.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3) { std::cerr << "usage: sam3d-crop-capture CASES.txt OUTPUT.txt\n"; return 2; }
    std::ifstream input(argv[1]); std::ofstream output(argv[2]);
    if (!input || !output) { std::cerr << "cannot open input/output\n"; return 2; }
    std::string header;
    input >> header;
    if (header != "SAM3D_CROP_CASES_V1") { std::cerr << "invalid input header\n"; return 2; }
    output << "SAM3D_CROP_RESULTS_V1\n" << std::setprecision(17);
    s3d_body_crop_request *request = nullptr;
    s3d_body_crop_result *result = nullptr;
    char error[256]{};
    auto check = [&](s3d_status status) { if (status != S3D_OK) throw std::runtime_error(error); };
    try {
        unsigned cases = 0;
        for (;;) {
            input >> std::ws;
            if (input.eof()) break;
            uint32_t id = 0, w = 0, h = 0;
            float padding = 0, prior = 0, rotation = 0, box[4]{};
            if (!(input >> id >> w >> h >> padding >> prior >> rotation >> box[0] >> box[1] >> box[2] >> box[3]))
                throw std::runtime_error("malformed input case");
            if (++cases > 10000) throw std::runtime_error("too many input cases");
            check(s3d_body_crop_request_create(&request, error, sizeof(error)));
            check(s3d_body_crop_request_set_box(request, box, 4, error, sizeof(error)));
            check(s3d_body_crop_request_set_padding(request, padding, error, sizeof(error)));
            check(s3d_body_crop_request_set_prior_aspect(request, prior, error, sizeof(error)));
            check(s3d_body_crop_request_set_output_size(request, w, h, error, sizeof(error)));
            check(s3d_body_crop_request_set_rotation(request, rotation, error, sizeof(error)));
            check(s3d_body_crop_compute(request, &result, error, sizeof(error)));
            output << id;
            for (auto getter : {s3d_body_crop_get_center, s3d_body_crop_get_padded_scale,
                                s3d_body_crop_get_prior_scale, s3d_body_crop_get_scale}) {
                const float *data = nullptr; uint64_t count = 0;
                check(getter(result, &data, &count, error, sizeof(error)));
                for (uint64_t i = 0; i < count; ++i) output << ' ' << data[i];
            }
            const double *affine = nullptr; uint64_t count = 0;
            check(s3d_body_crop_get_affine(result, &affine, &count, error, sizeof(error)));
            for (uint64_t i = 0; i < count; ++i) output << ' ' << affine[i];
            output << '\n';
            s3d_body_crop_result_free(result); result = nullptr;
            s3d_body_crop_request_free(request); request = nullptr;
        }
        if (!cases || !output) throw std::runtime_error("empty input or output error");
    } catch (const std::exception &e) {
        s3d_body_crop_result_free(result); s3d_body_crop_request_free(request);
        std::cerr << e.what() << '\n'; return 1;
    }
    return 0;
}
