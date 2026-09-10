#include "objects_pointpatch.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("expected backend module and original fixture");
    std::ifstream in(argv[2]);std::string magic;sam3d::pointpatch_shape s;uint32_t remap,has_mask,drop,force;
    if(!(in>>magic) || magic!="S3D_POINTPATCH_REGRESSION_V1" ||
       !(in>>s.batch>>s.height>>s.width>>s.side>>s.patch>>s.dim>>remap>>has_mask>>drop>>force))
        throw std::runtime_error("invalid fixture header");
    s.remap=sam3d::point_remapping(remap);s.dropout_enabled=drop;s.force_dropout=force;
    sam3d::validate_pointpatch_shape(s);
    auto values=[&](uint64_t n){std::vector<float> v(n);for(auto &x:v)if(!(in>>x) || !std::isfinite(x))throw std::runtime_error("invalid fixture value");return v;};
    auto xyz=values(uint64_t(s.batch)*3*s.height*s.width);
    std::vector<uint8_t> mask(has_mask?uint64_t(s.batch)*s.side*s.side:0);
    for(auto &v:mask){unsigned x;if(!(in>>x) || x>1)throw std::runtime_error("invalid fixture mask");v=uint8_t(x);}
    sam3d::named_floats parameters;
    for(auto &[key,n]:sam3d::pointpatch_parameter_sizes(s)){
        std::string name;uint64_t count;if(!(in>>name>>count) || name!=key || count!=n)throw std::runtime_error("invalid fixture parameter");
        parameters[key]=values(n);
    }
    uint32_t count;if(!(in>>count) || count!=27)throw std::runtime_error("invalid tap count");
    sam3d::named_floats expected;
    for(uint32_t i=0;i<count;++i){std::string key;uint64_t n;
        if(!(in>>key>>n) || !n || n>1000000 || expected.contains(key))throw std::runtime_error("invalid fixture tap");expected[key]=values(n);
    }
    in>>std::ws;if(!in.eof())throw std::runtime_error("trailing fixture data");
    auto compare=[](const std::vector<float> &got,const std::vector<float> &ref,const std::string &key){
        if(got.size()!=ref.size())throw std::runtime_error("extent mismatch "+key);
        double maximum=0,error2=0,ref2=0;
        for(size_t i=0;i<got.size();++i){if(!std::isfinite(got[i]))throw std::runtime_error("nonfinite output "+key);
            double delta=double(got[i])-ref[i];maximum=std::max(maximum,std::abs(delta));error2+=delta*delta;ref2+=double(ref[i])*ref[i];}
        if(maximum>1e-4 || std::sqrt(error2)/std::max(std::sqrt(ref2),1e-12)>2e-5)throw std::runtime_error("original mismatch "+key);
    };
    sam3d::neural_session session(argv[1],"CPU",0);
    sam3d::named_floats taps;
    auto observer=[&](const std::string &key,std::span<const float> v,uint64_t offset,uint64_t total){
        if(!expected.contains(key) || total!=expected.at(key).size() || taps[key].size()!=offset || v.size()>total-offset)
            throw std::runtime_error("overlapping/incomplete/unknown diagnostic tap");
        taps[key].insert(taps[key].end(),v.begin(),v.end());
    };
    auto actual=sam3d::objects_pointpatch(session,s,xyz,mask,parameters,observer,7);
    if(taps.size()!=27)throw std::runtime_error("incomplete diagnostic tap set");
    for(auto &[key,ref]:expected){compare(taps.at(key),ref,key);
        if((key=="00.resized" || key=="01.valid" || key=="02.safe" || key=="32.position") && taps.at(key)!=ref)
            throw std::runtime_error("exact tap mismatch "+key);
    }
    compare(actual,expected.at("90.output"),"returned output");
    for(uint32_t chunk:{1u,32u})compare(sam3d::objects_pointpatch(session,s,xyz,mask,parameters,{},chunk),actual,"chunk invariance");
    auto reject=[](auto run){bool caught=false;try{run();}catch(const std::invalid_argument &){caught=true;}
        if(!caught)throw std::runtime_error("invalid PointPatch input accepted");};
    auto run=[&]{sam3d::objects_pointpatch(session,s,xyz,mask,parameters);};
    auto saved=s;s.side=0;reject(run);s=saved;s.patch=3;reject(run);s=saved;s.dim=17;reject(run);s=saved;
    s.batch=3;reject(run);s=saved;s.remap=sam3d::point_remapping(99);reject(run);s=saved;s.force_dropout=true;reject(run);s=saved;
    reject([&]{sam3d::objects_pointpatch(session,s,xyz,mask,parameters,{},0);});
    reject([&]{sam3d::objects_pointpatch(session,s,xyz,mask,parameters,{},65);});
    auto original=parameters;parameters.erase("cls_token");reject(run);parameters=original;
    parameters["unknown"]={0};reject(run);parameters=original;
    parameters["cls_token"].push_back(0);reject(run);parameters=original;
    parameters["cls_token"][0]=std::numeric_limits<float>::quiet_NaN();reject(run);parameters=original;
    auto original_mask=mask;mask[0]=2;reject(run);mask=original_mask;mask.pop_back();reject(run);mask=original_mask;
    auto original_xyz=xyz;xyz.pop_back();reject(run);xyz=original_xyz;
    mask[0]=1;xyz[0]=std::numeric_limits<float>::infinity();reject(run);xyz=original_xyz;mask=original_mask;
    // Invalid point values must be removed before entering the transformer.
    mask[0]=0;xyz[0]=std::numeric_limits<float>::quiet_NaN();run();xyz=original_xyz;mask=original_mask;
    mask[0]=1;xyz[2*s.height*s.width]=-2;reject(run);
    std::cout<<"27 original PointPatch boundaries, chunk equivalence and malformed-input checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
