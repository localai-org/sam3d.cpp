#include "condition_case.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

int main(int argc,char **argv) {
    try {
        if(argc!=3 && argc!=4)throw std::invalid_argument("expected CPU backend, fixture and optional --cuda-arithmetic");
        if(argc==4 && std::string(argv[3])!="--cuda-arithmetic")throw std::invalid_argument("invalid arithmetic option");
        auto division=argc==4?sam3d::scalar_division::reciprocal_multiply:sam3d::scalar_division::direct;
        std::ifstream file(argv[2]);std::string magic;uint32_t count;
        if(!(file>>magic>>count) || magic!="S3D_CONDITION_REGRESSION_V1" || count!=3)throw std::runtime_error("invalid conditioning regression");
        auto group=[&] {
            uint32_t n;if(!(file>>n) || n>256)throw std::runtime_error("invalid fixture tensor count");
            sam3d::named_floats result;
            for(uint32_t i=0;i<n;++i) {
                std::string name;uint64_t size;
                if(!(file>>name>>size) || size>100000 || result.contains(name))throw std::runtime_error("invalid tensor");
                auto &v=result[name];v.resize(size);for(auto &x:v)if(!(file>>x) || !std::isfinite(x))throw std::runtime_error("invalid fixture float");
            }return result;
        };
        auto reject=[](auto fn){bool bad=false;try{fn();}catch(const std::invalid_argument&){bad=true;}
            if(!bad)throw std::runtime_error("invalid conditioning input accepted");};
        sam3d::neural_session session(argv[1],"CPU",0);uint32_t checked=0;
        for(uint32_t i=0;i<count;++i) {
            std::array<uint32_t,16> dims;for(auto &d:dims)if(!(file>>d))throw std::runtime_error("missing dimensions");
            condition_case value(dims);auto input=group(),parameters=group(),reference=group();
            auto run=[&]{return value.run(session,input,parameters,division);};auto result=run();
            if(value.run(session,input,parameters,division,true)!=result)
                throw std::runtime_error("channels-last conditioning/decoder changed exact outputs");
            if(result.size()!=13 || reference.size()!=13)throw std::runtime_error("conditioning tap set mismatch");
            for(auto &[name,v]:result) {
                auto &r=reference.at(name);if(v.size()!=r.size())throw std::runtime_error("conditioning tap size mismatch");
                double maximum=0,error=0,norm=0;
                for(size_t j=0;j<v.size();++j){double delta=double(v[j])-r[j];maximum=std::max(maximum,std::abs(delta));error=std::hypot(error,delta);norm=std::hypot(norm,double(r[j]));}
                bool exact=name=="00.init_input" || name=="02.prev_input" || name=="11.prompt_mask";
                if((exact && maximum!=0) || maximum>1e-4 || error/std::max(norm,1e-12)>2e-5)throw std::runtime_error("conditioning divergence case "+std::to_string(i)+" at "+name);
                ++checked;
            }
            for(const std::string name:{"features","rays","cliff","keypoints"}) {
                auto saved=input.at(name);input[name].clear();reject(run);input[name]=saved;
                input[name][0]=std::numeric_limits<float>::infinity();reject(run);input[name]=saved;
            }
            const auto previous=input.contains("previous")?input.at("previous"):std::vector<float>{};
            input["previous"]={0};reject(run);input.erase("previous");if(!previous.empty())input["previous"]=previous;
            auto original=value.shape;value.shape.pose_dim=0;reject(run);value.shape=original;
            value.shape.keypoints=71;reject(run);value.shape=original;
            parameters["extra"]={0};reject(run);parameters.erase("extra");
            auto saved=parameters.at("init_pose.weight");parameters["init_pose.weight"].clear();reject(run);parameters["init_pose.weight"]=saved;
        }
        file>>std::ws;if(!file.eof())throw std::runtime_error("trailing fixture");
        std::cout<<"Body conditioning through first decoder layer: "<<checked<<" original boundaries and rejection cases passed\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
