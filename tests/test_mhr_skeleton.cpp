#include "mhr_skeleton.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected local MHR fixture");std::ifstream file(argv[1]);std::string magic;uint32_t batch;
    if(!(file>>magic>>batch) || magic!="S3D_MHR_LOCAL_V1" || batch!=2)throw std::runtime_error("invalid MHR fixture");
    auto input=[&](const char *expected,auto &v,size_t count){std::string name;size_t n;if(!(file>>name>>n) || name!=expected || n!=count)throw std::runtime_error("invalid fixture input");v.resize(n);for(auto &x:v)if(!(file>>x))throw std::runtime_error("truncated fixture input");};
    sam3d::mhr_skeleton_data d;std::vector<float> jp;input("offsets",d.offsets,127*3);input("prerotations",d.prerotations,127*4);input("prefix",d.prefix,266*2);input("parents",d.parents,127);input("joint_parameters",jp,2*889);
    auto run=[&]{return sam3d::mhr_local_skeleton(batch,jp,d);};auto result=run();size_t count;if(!(file>>count) || count!=44 || result.f32.size()+result.f64.size()!=count)throw std::runtime_error("invalid tap count");
    std::map<std::string,bool> seen;double worst=0;
    auto check=[&](const auto &v,size_t size,const std::string &name){if(v.size()!=size)throw std::runtime_error("wrong native shape");double maximum=0,error=0,norm=0;
        for(size_t i=0;i<size;++i){double r;if(!(file>>r) || !std::isfinite(r) || !std::isfinite(v[i]))throw std::runtime_error("nonfinite or missing tap");double delta=double(v[i])-r;maximum=std::max(maximum,std::abs(delta));error=std::hypot(error,delta);norm=std::hypot(norm,r);}
        worst=std::max(worst,maximum);if(maximum>1e-4 || error/std::max(norm,1e-12)>2e-5)throw std::runtime_error("MHR local divergence at "+name+" max="+std::to_string(maximum));};
    for(size_t i=0;i<count;++i){std::string name;size_t width,size;if(!(file>>name>>width>>size) || size>100000 || seen.contains(name))throw std::runtime_error("invalid tap header");seen[name]=true;
        if(width==4)check(result.f32.at(name),size,name);else if(width==8)check(result.f64.at(name),size,name);else throw std::runtime_error("invalid tap dtype");}
    file>>std::ws;if(!file.eof())throw std::runtime_error("trailing fixture data");
    auto reject=[&]{bool bad=false;try{run();}catch(const std::invalid_argument &){bad=true;}if(!bad)throw std::runtime_error("invalid MHR input accepted");};
    auto original_jp=jp;auto original=d;
    batch=0;reject();batch=3;reject();batch=2;
    jp.pop_back();reject();jp=original_jp;jp[0]=std::numeric_limits<float>::quiet_NaN();reject();jp=original_jp;
    jp[6]=10000;reject();jp=original_jp;jp[6]=-10000;reject();jp=original_jp;
    d.offsets.pop_back();reject();d=original;d.offsets[0]=std::numeric_limits<float>::infinity();reject();d=original;
    std::fill_n(d.prerotations.begin(),4,0);reject();d=original;d.prerotations.pop_back();reject();d=original;
    for(int32_t bad:{-1,0,127}){d.prefix[0]=bad;reject();d=original;}
    d.prefix[1]=d.prefix[0];reject();d=original;d.prefix[266]=127;reject();d=original;
    // In-range but topologically wrong targets must not pass domain-only checks.
    d.prefix[266]=d.prefix[0];reject();d=original;d.prefix.pop_back();reject();d=original;
    d.parents[0]=0;reject();d=original;d.parents[1]=1;reject();d=original;d.parents.pop_back();reject();
    std::cout<<"MHR local/F64 prefix: "<<count<<" original boundaries, max_abs="<<worst<<", rejection checks passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
