#include "body_camera_head.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc,char **argv){
    try{
        if(argc!=3)throw std::invalid_argument("expected CPU backend and camera-head fixture");
        std::ifstream file(argv[2]);std::string magic;uint32_t count;
        if(!(file>>magic>>count) || magic!="S3D_CAMERA_HEAD_REGRESSION_V1" || count!=3)throw std::runtime_error("invalid camera-head regression");
        auto group=[&]{uint32_t n;if(!(file>>n) || n>64)throw std::runtime_error("invalid fixture tensor count");
            sam3d::named_floats result;for(uint32_t i=0;i<n;++i){std::string name;uint64_t size;
                if(!(file>>name>>size) || size>100000 || result.contains(name))throw std::runtime_error("invalid fixture tensor");
                auto &v=result[name];v.resize(size);for(auto &x:v)if(!(file>>x) || !std::isfinite(x))throw std::runtime_error("invalid fixture float");}
            return result;};
        auto reject=[](auto fn){bool bad=false;try{fn();}catch(const std::invalid_argument&){bad=true;}if(!bad)throw std::runtime_error("invalid camera-head input accepted");};
        sam3d::neural_session session(argv[1],"CPU",0);uint32_t checked=0;
        for(uint32_t index=0;index<count;++index){sam3d::camera_head_shape s;uint32_t initial;
            if(!(file>>s.batch>>s.dim>>s.hidden>>s.depth>>s.points>>s.intrinsics_center>>initial>>s.scale_factor))throw std::runtime_error("invalid camera-head shape");
            auto input=group(),parameters=group(),reference=group();
            auto run=[&]{return sam3d::body_camera_head(session,s,input.at("token"),input["initial"],input.at("points"),input.at("center"),input.at("box"),input.at("image_size"),input.at("intrinsics"),parameters);};
            auto result=run();if(result.size()!=2*s.depth+10 || result.size()!=reference.size())throw std::runtime_error("camera-head tap set mismatch");
            auto projection=sam3d::body_camera_project(session,s,result.at("10.pred_cam"),input.at("points"),input.at("center"),input.at("box"),input.at("image_size"),input.at("intrinsics"));
            if(projection.size()!=11)throw std::runtime_error("projection-only tap mismatch");
            for(auto &[name,v]:projection)if(v!=result.at(name))throw std::runtime_error("projection-only path changed "+name);
            for(auto &[name,v]:result){auto &r=reference.at(name);if(v.size()!=r.size())throw std::runtime_error("camera-head tap size mismatch");
                double maximum=0,error=0,norm=0;for(size_t i=0;i<v.size();++i){double delta=double(v[i])-r[i];maximum=std::max(maximum,std::abs(delta));error=std::hypot(error,delta);norm=std::hypot(norm,double(r[i]));}
                double limit=(name=="12.scaled_box" || name=="19.intrinsic_projection" || name=="20.pixels")?1e-3:1e-4;
                if((name=="13.focal" && maximum!=0) || maximum>limit || error/std::max(norm,1e-12)>2e-5)throw std::runtime_error("camera-head divergence case "+std::to_string(index)+" at "+name);
                ++checked;
            }
            for(const std::string name:{"token","points","center","box","image_size","intrinsics"}){auto saved=input.at(name);input[name].clear();reject(run);input[name]=saved;
                input[name][0]=std::numeric_limits<float>::infinity();reject(run);input[name]=saved;}
            auto saved=input["initial"];input["initial"]={0};reject(run);input["initial"]=saved;
            auto shape=s;s.depth=0;reject(run);s=shape;s.points=100001;reject(run);s=shape;s.scale_factor=0;reject(run);s=shape;
            saved=input.at("intrinsics");input["intrinsics"][8]=0;reject(run);input["intrinsics"]=saved;
            saved=input.at("box");input["box"][0]=-1;reject(run);input["box"]=saved;
            parameters["extra"]={0};reject(run);parameters.erase("extra");
            auto missing=parameters.extract(parameters.begin());reject(run);parameters.insert(std::move(missing));
        }
        file>>std::ws;if(!file.eof())throw std::runtime_error("trailing camera-head fixture");
        // Undefined projection at exact zero camera-space depth is rejected,
        // never hidden by a clamp or emitted as non-finite result buffers.
        sam3d::camera_head_shape zero{1,4,1,1,1,1.f,true};sam3d::named_floats weights;
        for(auto &[name,size]:sam3d::camera_head_parameter_sizes(zero))weights[name].resize(size,0.f);
        weights.at("proj.layers.0.bias")[0]=-1;
        std::vector<float> token(4,0),points{0,0,-1},center{0,0},box{2},size{2,2},k{1,0,0,0,1,0,0,0,1};
        reject([&]{sam3d::body_camera_head(session,zero,token,{},points,center,box,size,k,weights);});
        std::cout<<"Camera head/projection: "<<checked<<" original boundaries and rejection tests passed\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
