#include "objects_fuser.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("expected CPU module and original fusion fixture");
    sam3d::neural_session session(argv[1],"CPU",0);std::ifstream in(argv[2]);std::string tag;uint32_t count;
    if(!(in>>tag>>count) || tag!="S3D_FUSER_REGRESSION_V1" || count!=6)throw std::runtime_error("invalid fusion fixture");
    auto tensor=[&](uint64_t n){std::vector<float> v(n);for(auto &x:v)if(!(in>>x) || !std::isfinite(x))throw std::runtime_error("invalid fixture tensor");return v;};
    unsigned checks=0;
    for(uint32_t ci=0;ci<count;++ci){sam3d::fuser_shape s;uint32_t ne,ni,norm,random;
        if(!(in>>s.batch>>ne>>ni>>norm>>random>>s.projection_multiplier>>s.compression_multiplier) || !ne || ne>2 || !ni || ni>3 || norm>1 || random>1)throw std::runtime_error("invalid fixture counts");
        s.pre_norm=norm;s.random_position=random;s.embed_dims.resize(ne);for(auto &d:s.embed_dims)if(!(in>>d) || !d || d>32)throw std::runtime_error("invalid fixture width");
        for(uint32_t i=0;i<ni;++i){sam3d::fuser_input x;uint32_t drop;if(!(in>>x.embedder>>x.tokens>>x.position>>drop) || x.tokens>16 || drop>1)throw std::runtime_error("invalid fixture modality");x.forced_drop=drop;s.inputs.push_back(x);}
        auto sizes=sam3d::fuser_parameter_sizes(s);std::vector<std::vector<float>> input;for(auto x:s.inputs)input.push_back(tensor(uint64_t(s.batch)*x.tokens*s.embed_dims[x.embedder]));
        sam3d::named_floats params;for(auto &[key,n]:sizes)params[key]=tensor(n);
        auto original_input=input;auto taps=sam3d::objects_fuse(session,s,input,params);uint32_t nt;
        if(input!=original_input)throw std::runtime_error("fusion mutated caller embeddings");
        if(!(in>>nt) || nt!=taps.size())throw std::runtime_error("fusion tap count mismatch");
        for(auto &[key,v]:taps){std::string name;uint64_t n;if(!(in>>name>>n) || name!=key || n!=v.size())throw std::runtime_error("fusion tap shape mismatch");auto ref=tensor(n);double mx=0,e2=0,r2=0;
            for(size_t i=0;i<n;++i){if(!std::isfinite(v[i]))throw std::runtime_error("nonfinite fusion output");double delta=double(v[i])-ref[i];mx=std::max(mx,std::abs(delta));e2+=delta*delta;r2+=double(ref[i])*ref[i];}
            bool exact=key.ends_with(".input") && key.find(".projection.")==std::string::npos && !key.starts_with("30.compression.");
            if(mx>1e-4 || std::sqrt(e2)/std::max(std::sqrt(r2),1e-12)>2e-5 || (exact && mx!=0))throw std::runtime_error("original fusion mismatch "+key);
            ++checks;
        }
        auto compact=sam3d::objects_fuse(session,s,input,params,false);
        if(compact.size()!=1 || compact.at("90.output")!=taps.at("90.output"))throw std::runtime_error("fusion observer changed output");
        for(size_t i=0;i<s.inputs.size();++i)if(s.inputs[i].forced_drop)for(float x:taps.at("10.modality."+std::to_string(i)+".dropped"))if(x!=0)throw std::runtime_error("forced modality not dropped");
    }
    in>>std::ws;if(!in.eof())throw std::runtime_error("trailing fusion fixture");
    sam3d::fuser_shape s;s.embed_dims={16};s.inputs={{0,3,0,false}};
    auto reject=[](auto fn){bool caught=false;try{fn();}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("invalid fusion input accepted");};
    auto validate=[&]{sam3d::validate_fuser_shape(s);};validate();
    s.batch=0;reject(validate);s.batch=3;reject(validate);s.batch=1;
    s.inputs[0].embedder=1;reject(validate);s.inputs[0].embedder=0;
    s.inputs[0].tokens=0;reject(validate);s.inputs[0].tokens=4097;reject(validate);s.inputs[0].tokens=3;
    for(int32_t v:{-2,1,INT32_MAX}){s.inputs[0].position=v;reject(validate);}s.inputs[0].position=0;
    for(double v:{-1.,9.,1e-10,std::numeric_limits<double>::infinity()}){s.projection_multiplier=v;reject(validate);}s.projection_multiplier=4;
    s.embed_dims={16,24};s.inputs.push_back({1,4,1,false});s.compression_multiplier=4;reject(validate);s.inputs[1].tokens=3;reject(validate); // Original compression width mismatch.
    s.compression_multiplier=0;s.inputs={{1,3,0,false},{0,3,1,false}};reject(validate);
    s.embed_dims={16};s.inputs={{0,3,0,false}};
    sam3d::named_floats p;for(auto &[key,n]:sam3d::fuser_parameter_sizes(s))p[key]=std::vector<float>(n,1.f);
    std::vector<std::vector<float>> input{std::vector<float>(48,1)};
    auto run=[&]{sam3d::objects_fuse(session,s,input,p);};run();
    input[0].pop_back();reject(run);input[0].push_back(std::numeric_limits<float>::quiet_NaN());reject(run);input[0].back()=1;
    p.begin()->second.pop_back();reject(run);p.begin()->second.push_back(std::numeric_limits<float>::infinity());reject(run);
    std::cout<<checks<<" original fusion boundaries plus ownership/observer/drop/rejection checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
