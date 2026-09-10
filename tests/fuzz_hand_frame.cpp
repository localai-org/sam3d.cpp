#include "body_hand_frame.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *input,size_t size){
    if(size<32)return 0;
    size_t offset=1;auto value=[&](){float v;std::array<uint8_t,4> bytes;for(auto &x:bytes)x=input[offset++%size];std::memcpy(&v,bytes.data(),4);return v;};
    uint32_t batch=input[0]%4+1;
    auto tensor=[&](size_t n){std::vector<float> out(n);for(auto &v:out)v=value();return out;};
    auto r=tensor(batch*3),t=tensor(batch*3),w=tensor(9),wrist=tensor(3),root=tensor(3),p=tensor(batch*204),k=tensor(batch*924);
    std::array<int32_t,145> indices;for(size_t i=0;i<indices.size();++i)indices[i]=int32_t(i)+6;
    if(input[0]&4){for(auto *v:{&r,&t,&wrist,&root,&p,&k})for(auto &x:*v)x=(float(input[offset++%size])-128.f)/64.f;w={1,0,0,0,1,0,0,0,1};}
    if(input[0]&8)std::memcpy(&indices[input[1]%145],input+4,4);
    if(input[0]&16)std::memcpy(&batch,input+8,4);
    auto checked=[](const auto &v){for(float x:v)if(!std::isfinite(x))std::abort();};
    try{auto out=sam3d::body_hand_frame(batch,r,t,w,wrist,root);for(auto &[_,v]:out)checked(v);}catch(const std::invalid_argument &){}
    try{auto out=sam3d::body_hand_mask_parameters(batch,p,indices);checked(out);
        for(uint32_t b=0;b<batch;++b)for(size_t j=0;j<204;++j){bool masked=false;for(auto i:indices)masked|=i==int32_t(j);if(out[b*204+j]!=(masked?0.f:p[b*204+j]))std::abort();}
    }catch(const std::invalid_argument &){}
    try{auto out=sam3d::body_hand_mask_keypoints(batch,k);checked(out);
        for(uint32_t b=0;b<batch;++b)for(size_t j=0;j<924;++j)if(out[b*924+j]!=((j>=63 && j<126)?k[b*924+j]:0.f))std::abort();
    }catch(const std::invalid_argument &){}
    return 0;
}
