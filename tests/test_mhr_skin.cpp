#include "mhr_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <bit>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
namespace {
void simd(bool on){
#ifdef _WIN32
    _putenv_s("SAM3D_SIMD_SKINNING",on?"1":"0");
#else
    setenv("SAM3D_SIMD_SKINNING",on?"1":"0",1);
#endif
}
void exact(const std::vector<float> &a,const std::vector<float> &b){
    if(a.size()!=b.size())throw std::runtime_error("SIMD skin shape mismatch");
    for(size_t i=0;i<a.size();++i)if(std::bit_cast<uint32_t>(a[i])!=std::bit_cast<uint32_t>(b[i]))throw std::runtime_error("SIMD skin is not bit-exact");
}
void simd_cases(){
    uint32_t seed=9163;auto random=[&]{seed=seed*1664525u+1013904223u;return float(int(seed>>8)-8388608)/8388608.f;};
    for(size_t trial=0;trial<96;++trial){const uint32_t batch=1+trial%2,vertices=1+trial%37;
        std::vector<float> skeleton(batch*127*8),bind(127*8),points(batch*vertices*3),weights;
        std::vector<int32_t> js,vs;
        for(auto *v:{&skeleton,&bind})for(size_t j=0;j<v->size()/8;++j){
            for(size_t k=0;k<7;++k)(*v)[j*8+k]=random()*(k<3?100.f:1.f);
            (*v)[j*8+6]+=2.f;(*v)[j*8+7]=1.25f+.5f*random();}
        for(auto &v:points)v=150.f*random();
        for(uint32_t v=0;v<vertices;++v){const size_t count=1+(trial+v)%7;
            for(size_t k=0;k<count;++k){js.push_back((trial*13+v*3+k*19)%127);vs.push_back(v);weights.push_back(1.f/float(count));}}
        // Include zero influences, shuffled vertex order, all SIMD tails and
        // repeated vertex IDs within the same SIMD group.
        for(size_t k=0;k<trial%4;++k){js.push_back(126);vs.push_back(0);weights.push_back(0);}
        for(size_t i=weights.size();i>1;--i){const size_t j=(seed=seed*1664525u+1013904223u)%i;
            std::swap(js[i-1],js[j]);std::swap(vs[i-1],vs[j]);std::swap(weights[i-1],weights[j]);}
        auto run=[&](bool taps){return sam3d::mhr_skinning(batch,vertices,skeleton,points,bind,js,weights,vs,taps);};
        simd(false);auto scalar=run(false);simd(true);auto packed=run(false),captured=run(true);
        for(auto &[name,value]:scalar){exact(value,packed.at(name));exact(value,captured.at(name));}
    }
}
}
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected MHR skin fixture");std::ifstream file(argv[1]);std::string magic;uint32_t batch,vertices,count;
    if(!(file>>magic>>batch>>vertices>>count) || magic!="S3D_MHR_SKIN_V1" || batch!=2 || vertices!=16 || count<16 || count>128)throw std::runtime_error("invalid fixture header");
    auto tensor=[&](const char *key,auto &v,size_t n){std::string name;size_t size;if(!(file>>name>>size) || name!=key || size!=n)throw std::runtime_error("invalid input shape");v.resize(n);for(auto &x:v)if(!(file>>x))throw std::runtime_error("truncated input");};
    std::vector<float> skeleton,unposed,bind,weights;std::vector<int32_t> joint_ids,vertex_ids;
    tensor("skeleton",skeleton,2*127*8);tensor("unposed",unposed,2*vertices*3);tensor("inverse_bind",bind,127*8);tensor("skin_joints",joint_ids,count);tensor("weights",weights,count);tensor("skin_vertices",vertex_ids,count);
    auto run=[&](bool capture=true){return sam3d::mhr_skinning(batch,vertices,skeleton,unposed,bind,joint_ids,weights,vertex_ids,capture);};auto result=run();size_t taps;if(!(file>>taps) || taps!=12 || result.size()!=taps)throw std::runtime_error("wrong tap count");
    simd(true);auto compact=run(false);
    if(compact.size()!=2 || compact.at("90.vertices")!=result.at("90.vertices") || compact.at("30.skin_joint_state")!=result.at("30.skin_joint_state"))throw std::runtime_error("skipping operation taps changed skinning");
    std::map<std::string,bool> seen;double worst=0;
    for(size_t i=0;i<taps;++i){std::string name;size_t size;if(!(file>>name>>size) || seen.contains(name) || !result.contains(name) || result.at(name).size()!=size)throw std::runtime_error("invalid output shape");seen[name]=true;double maximum=0,error=0,norm=0;
        for(float x:result.at(name)){double r;if(!(file>>r) || !std::isfinite(r) || !std::isfinite(x))throw std::runtime_error("nonfinite/truncated result");maximum=std::max(maximum,std::abs(x-r));error=std::hypot(error,x-r);norm=std::hypot(norm,r);}
        worst=std::max(worst,maximum);if(maximum>1e-4 || error/std::max(norm,1e-12)>2e-5)throw std::runtime_error("MHR skin divergence at "+name);}
    file>>std::ws;if(!file.eof())throw std::runtime_error("trailing fixture");
    auto reject=[&]{for(bool capture:{false,true}){bool bad=false;try{run(capture);}catch(const std::invalid_argument &){bad=true;}if(!bad)throw std::runtime_error("invalid skin input accepted");}};
    for(auto v:{&skeleton,&unposed,&bind,&weights}){auto original=*v;v->pop_back();reject();*v=original;(*v)[0]=std::numeric_limits<float>::quiet_NaN();reject();*v=original;}
    for(auto v:{&joint_ids,&vertex_ids}){auto original=*v;v->pop_back();reject();*v=original;(*v)[0]=-1;reject();*v=original;(*v)[0]=127000;reject();*v=original;}
    auto original=weights;weights[0]=-1;reject();weights=original;weights[0]=1.1f;reject();weights=original;std::fill(weights.begin(),weights.end(),0);reject();weights=original;
    original=skeleton;std::fill_n(skeleton.begin()+3,4,0.f);reject();skeleton=original;skeleton[7]=0;reject();skeleton=original;
    original=bind;std::fill_n(bind.begin()+3,4,0.f);reject();bind=original;bind[7]=-1;reject();bind=original;
    batch=0;reject();batch=3;reject();batch=2;vertices=0;reject();vertices=18440;reject();vertices=16;
    simd_cases();std::cout<<"MHR original selected-vertex skinning: "<<taps<<" boundaries, max_abs="<<worst<<", rejection checks and 96 bit-exact SIMD/tail/scatter cases passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
