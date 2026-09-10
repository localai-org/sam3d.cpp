#include "feedback_case.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("expected CPU module and feedback fixture");std::ifstream file(argv[2]);std::string magic;uint32_t count;
    if(!(file>>magic>>count) || magic!="S3D_FEEDBACK_REGRESSION_V1" || count!=4)throw std::runtime_error("invalid feedback fixture");
    auto group=[&]{uint32_t n;if(!(file>>n) || n>128)throw std::runtime_error("invalid tensor count");sam3d::named_floats result;
        for(uint32_t i=0;i<n;++i){std::string name;uint64_t size;if(!(file>>name>>size) || size>100000 || result.contains(name))throw std::runtime_error("invalid tensor");
            auto &v=result[name];v.resize(size);for(auto &x:v)if(!(file>>x) || !std::isfinite(x))throw std::runtime_error("invalid float");}return result;};
    auto reject=[](auto fn){bool bad=false;try{fn();}catch(const std::invalid_argument&){bad=true;}if(!bad)throw std::runtime_error("invalid feedback accepted");};
    sam3d::neural_session session(argv[1],"CPU",0);uint32_t checked=0;
    for(uint32_t i=0;i<count;++i){std::array<uint32_t,15> dims;for(auto &d:dims)if(!(file>>d))throw std::runtime_error("missing shape");auto s=feedback_dimensions(dims);
        std::vector<int32_t> idx(s.keypoints),idx3(s.keypoints3d);for(auto list:{std::span(idx),std::span(idx3)})for(auto &v:list)if(!(file>>v))throw std::runtime_error("missing index");
        auto input=group(),parameters=group(),reference=group();auto run=[&]{return run_feedback(session,s,input,parameters,idx,idx3);};auto result=run();
        if(result.size()!=reference.size())throw std::runtime_error("feedback tap set mismatch");
        auto compact=run_feedback(session,s,input,parameters,idx,idx3,false);
        auto channels_last=input;
        const uint64_t grid=uint64_t(s.height)*s.width;
        for(uint64_t b=0;b<s.batch;++b)for(uint64_t p=0;p<grid;++p)for(uint64_t c=0;c<s.context_dim;++c)
            channels_last.at("image")[(b*grid+p)*s.context_dim+c]=input.at("image")[(b*s.context_dim+c)*grid+p];
        if(run_feedback(session,s,channels_last,parameters,idx,idx3,true,true)!=result)
            throw std::runtime_error("channels-last feedback changed original boundaries");
        for(const char *name:{"02.crop_points","90.tokens","91.augment"})
            if(compact.at(name)!=result.at(name))throw std::runtime_error("feedback diagnostic readbacks change output");
        for(auto &[name,v]:result){auto &r=reference.at(name);if(v.size()!=r.size())throw std::runtime_error("feedback tap size mismatch");double maximum=0,error=0,norm=0;
            for(size_t k=0;k<v.size();++k){double delta=double(v[k])-r[k];maximum=std::max(maximum,std::abs(delta));error=std::hypot(error,delta);norm=std::hypot(norm,double(r[k]));}
            bool exact=name=="00.homogeneous" || name=="11.selected_depth" || name=="12.invalid" || (s.layer+1==s.depth && (name=="90.tokens" || name=="91.augment"));
            if((exact && maximum!=0) || maximum>(name=="01.crop_pixels"?1e-3:1e-4) || error/std::max(norm,1e-12)>2e-5)throw std::runtime_error("feedback divergence case "+std::to_string(i)+" at "+name+" max="+std::to_string(maximum));++checked;
        }
        if(s.layer+1<s.depth)for(uint32_t b=0;b<s.batch;++b)for(uint32_t k=0;k<s.keypoints;++k)if(result.at("12.invalid")[b*s.keypoints+k])
            for(uint32_t d=0;d<s.dim;++d){const uint64_t pos=(uint64_t(b)*s.tokens+s.start2d+k)*s.dim+d;
                if(result.at("91.augment")[pos]!=0 || result.at("90.tokens")[pos]!=input.at("tokens")[pos]+parameters.at("keypoint_feat_linear.bias")[d])throw std::runtime_error("invalid sample bias/masking contract changed");}
        for(auto &[name,size]:feedback_inputs(s)){auto saved=input.at(name);input[name].clear();reject(run);input[name]=saved;input[name][0]=std::numeric_limits<float>::infinity();reject(run);input[name]=saved;}
        auto saved=idx;idx[0]=-1;reject(run);idx=saved;idx[0]=s.points;reject(run);idx=saved;
        auto shape=s;s.layer=s.depth;reject(run);s=shape;s.hip_left=s.points;reject(run);s=shape;
        if(s.keypoints3d){s.start3d=s.start2d;reject(run);s=shape;}
        parameters["extra"]={0};reject(run);parameters.erase("extra");auto missing=parameters.extract(parameters.begin());reject(run);parameters.insert(std::move(missing));
    }
    file>>std::ws;if(!file.eof())throw std::runtime_error("trailing feedback fixture");std::cout<<"Body crop/keypoint feedback: "<<checked<<" original boundaries plus masking/bias and rejection checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
