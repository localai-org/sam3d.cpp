#include "dino_block.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        const bool bf16=argc==4 && std::string_view(argv[3])=="--bf16";if(bf16)--argc;
        if (argc!=3) throw std::invalid_argument("expected module and fixture");
        std::ifstream file(argv[2]); std::string magic;
        if (!(file>>magic) || magic!="S3D_DINO_REGRESSION_V1") throw std::runtime_error("bad fixture header");
        sam3d::dino_shape s;
        if (!(file>>s.batch>>s.height>>s.width>>s.dim>>s.heads>>s.prefix>>s.hidden))
            throw std::runtime_error("bad fixture dimensions");
        sam3d::validate_dino_shape(s);
        auto read=[&](const std::string &expected,uint64_t size) {
            std::string name; uint64_t count;
            if (!(file>>name>>count) || name!=expected || size!=count) throw std::runtime_error("fixture tensor mismatch");
            std::vector<float> v(size);
            for (auto &x:v) if (!(file>>x) || !std::isfinite(x)) throw std::runtime_error("invalid fixture value");
            return v;
        };
        auto x=read("input",uint64_t(s.batch)*(s.height*s.width+s.prefix)*s.dim);
        sam3d::named_floats parameters;
        for (auto &[name,size]:sam3d::dino_parameter_sizes(s)) parameters[name]=read(name,size);
        sam3d::neural_session session(argv[1],"CPU",0);
        sam3d::weight_map checked=parameters;
        auto run=[&](bool capture){return bf16?sam3d::dino_block_bf16(session,s,x,checked,capture):sam3d::dino_block(session,s,x,parameters,capture);};
        auto taps=run(true);
        if (taps.size()!=22) throw std::runtime_error("missing diagnostic taps");
        auto compact=run(false);
        if (compact.size()!=1 || compact.at("21.output")!=taps.at("21.output"))
            throw std::runtime_error("output-only CPU block differs from diagnostic path");
        for (const auto &[name,v]:taps) {
            auto expected=read(name,v.size());
            double max_abs=0,error_norm=0,reference_norm=0;
            for (size_t i=0;i<v.size();++i) {
                const double error=double(v[i])-expected[i];
                max_abs=std::max(max_abs,std::abs(error));
                error_norm=std::hypot(error_norm,error);
                reference_norm=std::hypot(reference_norm,double(expected[i]));
            }
            if (max_abs>1e-4 || error_norm/std::max(reference_norm,1e-12)>2e-5)
                throw std::runtime_error("upstream regression failed at "+name);
            // BF16 operation outputs are exact for this tiny fixture; sin/cos,
            // logits and softmax are deliberately still F32 internal values.
            if(bf16 && name!="00.rope_sin" && name!="01.rope_cos" && name!="09.logits" && name!="10.probs" && max_abs!=0)
                throw std::runtime_error("BF16 operation rounding differs at "+name);
        }
        file>>std::ws;
        if (!file.eof()) throw std::runtime_error("trailing fixture data");
        auto expect_rejected=[&](const sam3d::named_floats &p) {
            bool rejected=false;
            try { sam3d::dino_block(session,s,x,p); }
            catch (const std::invalid_argument &) { rejected=true; }
            if (!rejected) throw std::runtime_error("invalid block parameters accepted");
        };
        auto bad=parameters; bad.erase("periods"); expect_rejected(bad);
        bad=parameters; bad["periods"][0]=0; expect_rejected(bad);
        bad=parameters; bad["attn.qkv.bias_mask"][s.dim]=1; expect_rejected(bad);
        bad=parameters; bad["norm1.weight"].push_back(0); expect_rejected(bad);
        auto zero_mask=parameters, zero_bias=parameters;
        std::fill(zero_mask["attn.qkv.bias_mask"].begin(),zero_mask["attn.qkv.bias_mask"].end(),0.f);
        std::fill(zero_bias["attn.qkv.bias"].begin(),zero_bias["attn.qkv.bias"].end(),0.f);
        if (sam3d::dino_block(session,s,x,zero_mask)!=sam3d::dino_block(session,s,x,zero_bias))
            throw std::runtime_error("published zero QKV mask does not suppress stored biases");
        std::cout<<"DINO block: all 22 original-upstream contract boundaries and rejection checks passed\n";
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
    return 0;
}
