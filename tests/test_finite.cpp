#include "finite.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(){try{
    auto check=[](std::span<const float> v,bool expected){
        if(sam3d::all_finite_f32(v)!=expected)throw std::runtime_error("finite scan changed classification");
    };
    check({},true);
    constexpr std::array<uint32_t,10> bad{0x7f800000,0xff800000,0x7f800001,0xff800001,
        0x7fc00000,0xffc00000,0x7fffffff,0xffffffff,0x7fbfffff,0xffbfffff};
    constexpr std::array<uint32_t,10> good{0,0x80000000,1,0x80000001,0x007fffff,
        0x807fffff,0x00800000,0x3f800000,0x7f7fffff,0xff7fffff};
    // Exact allocation boundaries are checked by ASan. Also test offset spans
    // surrounded by NaNs: the scan must not include either guard value.
    for(size_t n=1;n<=129;++n)for(size_t offset:{0u,1u,3u}){
        std::vector<float> storage(n+offset,std::bit_cast<float>(bad[0]));
        auto v=std::span(storage).subspan(offset,n);
        for(size_t i=0;i<n;++i)v[i]=std::bit_cast<float>(good[i%good.size()]);
        check(v,true);
        for(size_t i=0;i<n;++i){const auto saved=v[i];
            for(auto bits:bad){v[i]=std::bit_cast<float>(bits);check(v,false);}
            v[i]=saved;
        }
        check(v,true);
        if(n>2){v.front()=v.back()=std::bit_cast<float>(bad[0]);check(v.subspan(1,n-2),true);}
    }
    uint32_t seed=0x51736a41;
    std::vector<float> random(1048583);
    for(auto &value:random){seed=1664525u*seed+1013904223u;value=std::bit_cast<float>(seed);}
    for(size_t size:{1u,7u,15u,32u,129u,1029u,1280u,262145u,1048583u}){
        auto span=std::span<const float>(random).first(size);
        check(span,std::all_of(span.begin(),span.end(),[](float x){return std::isfinite(x);}));
    }
    for(auto &value:random)if(!std::isfinite(value))value=0;
    check(random,true);random.back()=std::bit_cast<float>(bad.back());check(random,false);
    std::cout<<"F32 finite scan: both infinities, NaN payloads/signs, subnormals, maximum finite, offsets, tails and random patterns passed\n";
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
