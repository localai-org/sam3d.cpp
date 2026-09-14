#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: test_objects_runtime_format FIXTURE.samt\n";
        return 2;
    }
    sam3d::RawTensor tensor;
    if (!sam3d::load_raw_tensor(argv[1], tensor)) {
        std::cerr << "could not load Objects tensor fixture\n";
        return 1;
    }
    if (tensor.type != GGML_TYPE_F32 || tensor.ne != std::vector<int64_t>({64, 64, 64}) ||
        tensor.data.size() != 64u * 64u * 64u * sizeof(float)) {
        std::cerr << "unexpected Objects occupancy fixture contract\n";
        return 1;
    }
    const auto *values = reinterpret_cast<const float *>(tensor.data.data());
    if (!std::all_of(values, values + 64u * 64u * 64u,
                     [](float value) { return std::isfinite(value); })) {
        std::cerr << "non-finite Objects occupancy fixture\n";
        return 1;
    }
    return 0;
}
