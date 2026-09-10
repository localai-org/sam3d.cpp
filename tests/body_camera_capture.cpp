#include "body_camera_case.hpp"
#include <bit>
#include <fstream>
#include <iostream>

int main(int argc,char **argv) {
    try {
        if (argc!=3) throw std::invalid_argument("usage: body-camera-capture INPUT OUTPUT");
        static_assert(std::endian::native==std::endian::little);
        std::ifstream input(argv[1],std::ios::binary);
        auto read=[&](void *data,size_t size){if (!input.read(static_cast<char *>(data),size)) throw std::runtime_error("truncated camera case");};
        std::array<char,8> magic;read(magic.data(),8);
        if (std::string(magic.data(),8)!="S3DRAY01") throw std::invalid_argument("invalid camera case magic");
        std::array<uint32_t,4> integers;std::array<float,9> floats;
        read(integers.data(),sizeof(integers));read(floats.data(),sizeof(floats));
        if (input.peek()!=std::ifstream::traits_type::eof()) throw std::invalid_argument("trailing camera case input");
        camera_case s{integers[0],integers[1],integers[2],integers[3],floats[0],
            {floats[1],floats[2],floats[3],floats[4]},{floats[5],floats[6],floats[7],floats[8]}};
        auto taps=run_camera_case(s);std::ofstream output(argv[2],std::ios::binary);
        for (const auto &[name,v]:taps)
            if (!output.write(reinterpret_cast<const char *>(v.data()),v.size()*4)) throw std::runtime_error("camera output write failed");
        output.close();if (!output) throw std::runtime_error("camera output close failed");
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
