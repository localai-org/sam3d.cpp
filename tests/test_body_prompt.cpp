#include "body_prompt.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        if(argc!=3 && argc!=4) throw std::invalid_argument("expected CPU backend, prompt fixture and optional --cuda-arithmetic");
        if(argc==4 && std::string(argv[3])!="--cuda-arithmetic") throw std::invalid_argument("invalid arithmetic option");
        const auto division=argc==4?sam3d::scalar_division::reciprocal_multiply:sam3d::scalar_division::direct;
        std::ifstream file(argv[2]);std::string magic;uint32_t count;
        if(!(file>>magic>>count) || magic!="S3D_PROMPT_REGRESSION_V1" || count!=3) throw std::runtime_error("invalid prompt fixture");
        auto group=[&] {
            uint32_t count;if(!(file>>count) || count>128) throw std::runtime_error("invalid tensor count");
            sam3d::named_floats values;
            for(uint32_t i=0;i<count;++i) {
                std::string name;uint64_t size;
                if(!(file>>name>>size) || size>100000 || values.contains(name)) throw std::runtime_error("invalid tensor");
                auto &v=values[name];v.resize(size);
                for(auto &x:v) if(!(file>>x) || !std::isfinite(x)) throw std::runtime_error("invalid float");
            }return values;
        };
        auto reject=[](auto fn){bool bad=false;try{fn();}catch(const std::invalid_argument&){bad=true;}
            if(!bad)throw std::runtime_error("invalid prompt input accepted");};
        sam3d::neural_session session(argv[1],"CPU",0);uint32_t checked=0;
        for(uint32_t index=0;index<count;++index) {
            sam3d::prompt_shape s;uint32_t ih,iw;
            if(!(file>>s.batch>>s.points>>s.dim>>s.joints>>s.height>>s.width>>ih>>iw)) throw std::runtime_error("invalid fixture shape");
            auto inputs=group(),parameters=group(),reference=group();auto &points=inputs.at("keypoints");
            auto run=[&]{return sam3d::body_prompt_encode(session,s,points,parameters,division);};
            auto pixel=[&]{return sam3d::body_position_pixels(session,s.batch,s.points,s.dim,ih,iw,inputs.at("pixels"),parameters.at("pe_layer.positional_encoding_gaussian_matrix"),division);};
            auto result=run();for(auto &[name,v]:pixel())result["30.pixel."+name]=std::move(v);
            if(result.size()!=27 || reference.size()!=27) throw std::runtime_error("missing prompt taps");
            auto compact=sam3d::body_prompt_encode(session,s,points,parameters,division,{},false);
            if(compact.size()!=3)throw std::runtime_error("compact prompt fields mismatch");
            for(auto &[name,v]:compact)if(v!=result.at(name))throw std::runtime_error("prompt diagnostic readbacks change output");
            for(auto &[name,v]:result) {
                auto &r=reference.at(name);if(r.size()!=v.size())throw std::runtime_error("prompt tap shape mismatch");
                double maximum=0,error=0,norm=0;
                for(size_t i=0;i<v.size();++i){double delta=double(v[i])-r[i];maximum=std::max(maximum,std::abs(delta));error=std::hypot(error,delta);norm=std::hypot(norm,double(r[i]));}
                bool exact=name.ends_with(".00.coords") || name=="21.mask";
                if((exact && maximum!=0) || maximum>1e-4 || error/std::max(norm,1e-12)>2e-5)
                    throw std::runtime_error("prompt divergence case "+std::to_string(index)+" at "+name+" max="+std::to_string(maximum));
                ++checked;
            }
            std::vector<float> dense_zero(s.dim,0.f);
            for(bool capture:{false,true}){
                const auto tokens=sam3d::body_prompt_encode(session,s,points,parameters,division,{},capture,true);
                if(tokens.contains("22.dense_nchw") || tokens.at("20.embeddings")!=result.at("20.embeddings") || tokens.at("21.mask")!=result.at("21.mask"))
                    throw std::runtime_error("prompt token layout changed sparse results");
                for(uint32_t n=0;n<s.height*s.width;++n)for(uint32_t d=0;d<s.dim;++d)
                    if(tokens.at("22.dense_tokens")[uint64_t(n)*s.dim+d]!=result.at("22.dense_nchw")[uint64_t(d)*s.height*s.width+n])
                        throw std::runtime_error("prompt token layout mismatch");
                if(capture && tokens.at("00.dense.07.encoding")!=tokens.at("22.dense_tokens"))
                    throw std::runtime_error("prompt token layout lost captured encoding");
            }
            auto hand_dense=[&]{return sam3d::body_prompt_encode(session,s,points,parameters,division,dense_zero);};auto hand=hand_dense();
            if(hand.at("20.embeddings")!=result.at("20.embeddings") || hand.at("21.mask")!=result.at("21.mask"))throw std::runtime_error("independent hand image PE changed shared sparse prompts");
            for(size_t channel=0;channel<s.dim;++channel)for(size_t k=0;k<s.height*s.width;++k)
                if(hand.at("22.dense_nchw")[channel*s.height*s.width+k]!=(channel<s.dim/2?0.f:1.f))throw std::runtime_error("hand dense Gaussian ignored/layout wrong");
            dense_zero.pop_back();reject(hand_dense);dense_zero.resize(s.dim);dense_zero[0]=std::numeric_limits<float>::infinity();reject(hand_dense);
            auto original=points;points[0]=-.001f;reject(run);points=original;
            points[1]=1.001f;reject(run);points=original;
            for(float label:{-3.f,float(s.joints),.5f,std::numeric_limits<float>::quiet_NaN()}) {points[2]=label;reject(run);points=original;}
            points.clear();reject(run);points=original;
            auto shape=s;s.dim=3;reject(run);s=shape;s.points=0;reject(run);s=shape;s.height=33;reject(run);s=shape;
            parameters["extra"]={0};reject(run);parameters.erase("extra");
            auto missing=parameters.extract(parameters.begin());reject(run);parameters.insert(std::move(missing));
            auto saved=parameters.begin()->second;parameters.begin()->second.push_back(0);reject(run);parameters.begin()->second=saved;
            parameters.begin()->second[0]=std::numeric_limits<float>::infinity();reject(run);parameters.begin()->second=saved;
            auto height=ih;ih=0;reject(pixel);ih=height;
            auto pix=inputs.at("pixels");inputs["pixels"].clear();reject(pixel);inputs["pixels"]=pix;
            inputs["pixels"][0]=std::numeric_limits<float>::quiet_NaN();reject(pixel);inputs["pixels"]=pix;
            reject([&]{sam3d::body_prompt_encode(session,s,points,parameters,static_cast<sam3d::scalar_division>(99));});
        }
        file>>std::ws;if(!file.eof())throw std::runtime_error("trailing fixture");
        std::cout<<"Body prompt: "<<checked<<" original boundaries and rejection tests passed\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
