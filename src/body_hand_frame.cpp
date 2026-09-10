// Copyright (c) Meta Platforms, Inc. and affiliates. SAM license.
// RoMa/SciPy rotation adaptations: NAVER Corp. / SciPy contributors,
// BSD-3-Clause; LICENSES/RoMa.txt, THIRD_PARTY_NOTICES.md.
#include "body_hand_frame.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace sam3d {
namespace {
void require(bool b,const char *s){if(!b)throw std::invalid_argument(s);}
void finite(std::span<const float> v){require(std::all_of(v.begin(),v.end(),[](float x){return std::isfinite(x);}),"nonfinite hand frame input/output");}
using quat=std::array<float,4>;
quat product(quat p,quat q){
    return {((p[3]*q[0]+q[3]*p[0])+(p[1]*q[2]-p[2]*q[1])),
            ((p[3]*q[1]+q[3]*p[1])+(p[2]*q[0]-p[0]*q[2])),
            ((p[3]*q[2]+q[3]*p[2])+(p[0]*q[1]-p[1]*q[0])),
            p[3]*q[3]-((p[0]*q[0]+p[1]*q[1])+p[2]*q[2])};
}
std::array<float,9> xyz_matrix(std::span<const float> angles){
    // RoMa constructs axis rotvec quaternions, composes Z,Y,X for extrinsic
    // xyz, normalizes, then expands all four squares. Not direct sin/cos Euler.
    std::array<quat,3> axes{};
    for(size_t i=0;i<3;++i){const float v=angles[i],half=std::sqrt(v*v)/2.f;
        const float square=half*half;
        const float sinc=half<1e-3f?(1.f-square/6.f)+(square*square)/120.f:std::sin(half)/half;
        axes[i][i]=(sinc/2.f)*v;axes[i][3]=std::cos(half);}
    auto q=product(product(axes[2],axes[1]),axes[0]);
    const float norm=std::sqrt(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]);require(norm>0 && std::isfinite(norm),"invalid hand quaternion");
    for(auto &v:q)v/=norm;
    const float x=q[0],y=q[1],z=q[2],w=q[3],x2=x*x,y2=y*y,z2=z*z,w2=w*w,xy=x*y,zw=z*w,xz=x*z,yw=y*w,yz=y*z,xw=x*w;
    return {((x2-y2)-z2)+w2,2.f*(xy-zw),2.f*(xz+yw),2.f*(xy+zw),((-x2+y2)-z2)+w2,2.f*(yz-xw),2.f*(xz-yw),2.f*(yz+xw),((-x2-y2)+z2)+w2};
}
void append(std::vector<float> &v,std::span<const float> x){v.insert(v.end(),x.begin(),x.end());}
}
named_floats body_hand_frame(uint32_t batch,std::span<const float> rotation,std::span<const float> translation,std::span<const float> world,std::span<const float> wrist,std::span<const float> root){
    require(batch>=1 && batch<=64 && rotation.size()==uint64_t(batch)*3 && translation.size()==rotation.size() && world.size()==9 && wrist.size()==3 && root.size()==3,"invalid hand frame shape");
    for(auto values:{rotation,translation,world,wrist,root})finite(values);
    named_floats out;
    for(uint32_t b=0;b<batch;++b){auto matrix=xyz_matrix(rotation.subspan(b*3,3));append(out["00.input_matrix"],matrix);std::array<float,9> composed;
        for(size_t i=0;i<3;++i)for(size_t j=0;j<3;++j)composed[i*3+j]=(matrix[i*3]*world[j]+matrix[i*3+1]*world[3+j])+matrix[i*3+2]*world[6+j];
        append(out["01.composed_matrix"],composed);
    }
    auto converted=body_matrix_rotation_xyz(batch,out.at("01.composed_matrix"));out["02.euler"]=std::move(converted.at("euler_xyz"));
    for(uint32_t b=0;b<batch;++b){auto matrix=xyz_matrix(std::span(out.at("02.euler")).subspan(b*3,3));append(out["03.output_matrix"],matrix);
        for(size_t i=0;i<3;++i){const float moved=(matrix[i*3]*(wrist[0]-root[0])+matrix[i*3+1]*(wrist[1]-root[1]))+matrix[i*3+2]*(wrist[2]-root[2]);
            out["04.translation"].push_back(-(moved+root[i])+translation[b*3+i]);}}
    for(auto &[_,v]:out)finite(v);return out;
}
std::vector<float> body_hand_mask_parameters(uint32_t batch,std::span<const float> p,std::span<const int32_t> indices){
    require(batch>=1 && batch<=64 && p.size()==uint64_t(batch)*204 && indices.size()==145,"invalid hand parameter mask shape");finite(p);
    std::array<bool,204> seen{};for(int32_t i:indices){require(i>=0 && i<204 && !seen[i],"invalid/duplicate nonhand parameter index");seen[i]=true;}
    std::vector<float> out(p.begin(),p.end());for(uint32_t b=0;b<batch;++b)for(int32_t i:indices)out[b*204+i]=0;return out;
}
std::vector<float> body_hand_mask_keypoints(uint32_t batch,std::span<const float> p){
    require(batch>=1 && batch<=64 && p.size()==uint64_t(batch)*308*3,"invalid hand keypoint shape");finite(p);std::vector<float> out(p.begin(),p.end());
    for(uint32_t b=0;b<batch;++b){std::fill_n(out.begin()+b*308*3,21*3,0);std::fill(out.begin()+(b*308+42)*3,out.begin()+(b+1)*308*3,0);}return out;
}
}
