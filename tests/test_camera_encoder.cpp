#include "camera_encoder.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        if (argc!=3) throw std::invalid_argument("expected backend module and original fixture");
        std::ifstream file(argv[2]); std::string magic; sam3d::camera_shape s;
        if (!(file>>magic) || magic!="S3D_CAMERA_REGRESSION_V1" ||
            !(file>>s.batch>>s.height>>s.width>>s.patch>>s.dim)) throw std::runtime_error("invalid camera fixture header");
        sam3d::validate_camera_shape(s);
        auto read=[&](const std::string &expected,uint64_t size) {
            std::string name; uint64_t count;
            if (!(file>>name>>count) || name!=expected || count!=size) throw std::runtime_error("camera fixture tensor mismatch");
            std::vector<float> values(size);
            for (auto &value:values) if (!(file>>value) || !std::isfinite(value)) throw std::runtime_error("invalid fixture number");
            return values;
        };
        auto features=read("features",uint64_t(s.batch)*s.dim*(s.height/s.patch)*(s.width/s.patch));
        auto rays=read("rays",uint64_t(s.batch)*2*s.height*s.width);
        sam3d::named_floats parameters;
        parameters["conv.weight"]=read("conv.weight",uint64_t(s.dim)*(s.dim+99));
        parameters["norm.weight"]=read("norm.weight",s.dim); parameters["norm.bias"]=read("norm.bias",s.dim);
        sam3d::neural_session session(argv[1],"CPU",0);
        auto output=sam3d::camera_encode(session,s,features,rays,parameters);
        if (output.size()!=8) throw std::runtime_error("missing camera taps");
        auto compact=sam3d::camera_encode(session,s,features,rays,parameters,false);
        if(compact.size()!=1 || compact.at("07.output")!=output.at("07.output"))
            throw std::runtime_error("camera diagnostic readbacks change output");
        const auto token_output=sam3d::camera_encode(session,s,features,rays,parameters,false,true).at("07.output");
        const uint64_t grid=(s.height/s.patch)*(s.width/s.patch);
        for(uint32_t b=0;b<s.batch;++b)for(uint64_t n=0;n<grid;++n)for(uint32_t d=0;d<s.dim;++d)
            if(token_output[(b*grid+n)*s.dim+d]!=output.at("07.output")[(uint64_t(b)*s.dim+d)*grid+n])
                throw std::runtime_error("camera token layout changed exact output");
        for (const auto &[name,values]:output) {
            auto reference=read(name,values.size());
            double maximum=0,error_norm=0,reference_norm=0;
            for (size_t i=0;i<values.size();++i) {
                double delta=double(values[i])-reference[i];
                maximum=std::max(maximum,std::abs(delta)); error_norm=std::hypot(error_norm,delta);
                reference_norm=std::hypot(reference_norm,double(reference[i]));
            }
            if (maximum>1e-4 || error_norm/std::max(reference_norm,1e-12)>2e-5)
                throw std::runtime_error("original camera regression failed at "+name);
        }
        file>>std::ws; if (!file.eof()) throw std::runtime_error("trailing camera fixture");
        auto reject=[](auto run) {
            bool rejected=false;
            try {run();} catch (const std::invalid_argument &) {rejected=true;}
            if (!rejected) throw std::runtime_error("invalid camera request accepted");
        };
        auto bad=s; bad.patch=0; reject([&]{sam3d::validate_camera_shape(bad);});
        bad=s; bad.height=513; reject([&]{sam3d::validate_camera_shape(bad);});
        bad=s; bad.dim=0; reject([&]{sam3d::validate_camera_shape(bad);});
        reject([&]{sam3d::camera_encode(session,s,features,{},parameters);});
        reject([&]{sam3d::camera_encode(session,s,features,rays,{});});
        auto wrong=parameters; wrong["norm.bias"].push_back(0);
        reject([&]{sam3d::camera_encode(session,s,features,rays,wrong);});
        wrong=parameters; wrong["norm.weight"][0]=std::numeric_limits<float>::quiet_NaN();
        reject([&]{sam3d::camera_encode(session,s,features,rays,wrong);});
        std::cout<<"Camera encoder: eight original-upstream boundaries and rejection checks passed\n";
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
