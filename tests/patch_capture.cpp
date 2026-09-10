// Bounded diagnostic protocol only; not a model loader/public inference API.
#include "neural.hpp"
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
uint32_t number(const char *text) {
    uint32_t n;
    auto [end, ec] = std::from_chars(text, text+std::strlen(text), n);
    if (ec != std::errc() || *end) throw std::invalid_argument("invalid device index");
    return n;
}
void read(std::ifstream &f, void *data, size_t bytes) {
    if (!f.read(static_cast<char *>(data), bytes)) throw std::runtime_error("truncated input");
}
void write(std::ofstream &f, const std::vector<float> &values) {
    if (!f.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float)))
        throw std::runtime_error("output write failed");
}
void self_test(sam3d::neural_session &session) {
    std::vector<float> image(16), weights(4, 1), bias(1, 2);
    for (unsigned i=0; i<16; ++i) image[i] = i;
    auto taps = sam3d::patch_embed(session, {1,1,4,4,1,2}, image, weights, bias);
    if (taps.projection != std::vector<float>{10,18,42,50} ||
        taps.tokens != std::vector<float>{12,20,44,52} ||
        taps.patches != std::vector<float>{0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15})
        throw std::runtime_error("patch smoke test mismatch");
    bool rejected = false;
    try { sam3d::patch_embed(session, {1,1,4,4,1,0}, image, weights, bias); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid shape was accepted");
}
}
int main(int argc, char **argv) {
    try {
        if (argc != 6 && argc != 7 && argc != 8)
            throw std::invalid_argument("usage: patch-capture MODULE CPU|Vulkan DEVICE EXPECTED_DESCRIPTION|- INPUT OUTPUT [THREADS] (or --self-test)");
        sam3d::neural_session session(argv[1], argv[2], number(argv[3]), argc==8?number(argv[7]):1,
                                     std::string(argv[4]) == "-" ? "" : argv[4]);
        std::cerr << "backend=" << session.description() << '\n';
        if (argc == 6) {
            if (std::string(argv[5]) != "--self-test") throw std::invalid_argument("expected --self-test");
            self_test(session); return 0;
        }
        static_assert(std::endian::native == std::endian::little);
        std::ifstream in(argv[5], std::ios::binary);
        std::array<char,8> magic{}; read(in, magic.data(), magic.size());
        if (std::string(magic.data(),8) != "S3DPAT01") throw std::invalid_argument("invalid input header");
        std::array<uint32_t,6> dims{}; read(in, dims.data(), sizeof(dims));
        auto [batch,channels,height,width,outputs,patch] = dims;
        sam3d::patch_shape shape{batch,channels,height,width,outputs,patch};
        sam3d::validate_patch_shape(shape);
        std::vector<float> image(uint64_t(batch)*channels*height*width);
        std::vector<float> weights(uint64_t(outputs)*channels*patch*patch), biases(outputs);
        read(in, image.data(), image.size()*4); read(in, weights.data(), weights.size()*4);
        read(in, biases.data(), biases.size()*4);
        if (in.peek() != std::ifstream::traits_type::eof()) throw std::invalid_argument("trailing input");
        auto taps = sam3d::patch_embed(session, shape, image, weights, biases);
        std::ofstream out(argv[6], std::ios::binary);
        write(out, taps.patches); write(out, taps.projection); write(out, taps.tokens);
        out.close(); if (!out) throw std::runtime_error("output close failed");
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
