#include "body_camera_case.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc,char **argv) {
    try {
        if (argc!=2) throw std::invalid_argument("expected original camera geometry fixture");
        std::ifstream file(argv[1]);std::string magic;unsigned cases;
        if (!(file>>magic>>cases) || magic!="S3D_CAMERA_GEOMETRY_REGRESSION_V1" || cases!=12)
            throw std::runtime_error("invalid camera geometry fixture header");
        unsigned checks=0;
        for (unsigned i=0;i<cases;++i) {
            camera_case s;
            if (!(file>>s.width>>s.height>>s.side>>s.use_center>>s.padding)) throw std::runtime_error("invalid case fields");
            for (auto &v:s.box) if (!(file>>v)) throw std::runtime_error("invalid fixture box");
            for (auto &v:s.intrinsics) if (!(file>>v)) throw std::runtime_error("invalid fixture camera");
            for (const auto &[name,values]:run_camera_case(s)) {
                std::string key;uint64_t count;
                if (!(file>>key>>count) || key!=name || count!=values.size()) throw std::runtime_error("regression tensor mismatch");
                double maximum=0,error_norm=0,reference_norm=0;
                for (float value:values) {
                    float expected;if (!(file>>expected) || !std::isfinite(expected)) throw std::runtime_error("invalid reference value");
                    double delta=double(value)-expected;maximum=std::max(maximum,std::abs(delta));
                    error_norm=std::hypot(error_norm,delta);reference_norm=std::hypot(reference_norm,double(expected));
                }
                if (maximum>1e-6 || error_norm/std::max(reference_norm,1e-12)>1e-6)
                    throw std::runtime_error("original camera geometry mismatch at "+std::to_string(i)+" "+name);
                ++checks;
            }
        }
        file>>std::ws;if (!file.eof()) throw std::runtime_error("trailing geometry regression input");
        std::cout<<"Original crop-to-camera regression: "<<checks<<" boundaries passed\n";
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
