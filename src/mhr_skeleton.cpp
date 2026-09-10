// Copyright (c) Meta Platforms, Inc. and affiliates.
// Adapted from Momentum functions embedded in the official MHR v1.0.1 asset.
// MIT: LICENSES/Momentum.txt; asset Apache-2.0: LICENSES/MHR-Apache-2.0.txt.
#include "mhr_skeleton.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>

namespace sam3d {
namespace {
constexpr size_t joints=127, prefix_count=266;
constexpr std::array<size_t,4> groups{65,56,62,83};
void require(bool ok,const char *message){if(!ok)throw std::invalid_argument(message);}
template<class T> void finite(const std::vector<T> &v){require(std::all_of(v.begin(),v.end(),[](T x){return std::isfinite(x);}),"nonfinite MHR skeleton tensor");}
template<class T> using quat=std::array<T,4>;
template<class T> using vec3=std::array<T,3>;
template<class T> using state=std::array<T,8>;
template<class T,size_t N> void append(std::vector<T> &out,const std::array<T,N> &v){out.insert(out.end(),v.begin(),v.end());}
template<class T> quat<T> multiply(quat<T> a,quat<T> b){
    return {((a[3]*b[0]+a[0]*b[3])+a[1]*b[2])-a[2]*b[1],
            ((a[3]*b[1]-a[0]*b[2])+a[1]*b[3])+a[2]*b[0],
            ((a[3]*b[2]+a[0]*b[1])-a[1]*b[0])+a[2]*b[3],
            ((a[3]*b[3]-a[0]*b[0])-a[1]*b[1])-a[2]*b[2]};
}
template<class T> vec3<T> cross(vec3<T> a,vec3<T> b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
quat<double> normalize(quat<double> q,std::vector<double> &norms,std::vector<double> &normalized){
    double n=std::sqrt(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]);norms.push_back(n);
    for(auto &v:q)v/=std::max(n,1e-12);append(normalized,q);return q;
}
state<double> compose(state<double> a,state<double> b,mhr_skeleton_taps &t,const std::string &prefix){
    auto aq=normalize({a[3],a[4],a[5],a[6]},t.f64[prefix+"norm.0"],t.f64[prefix+"normalized.0"]);
    auto bq=normalize({b[3],b[4],b[5],b[6]},t.f64[prefix+"norm.1"],t.f64[prefix+"normalized.1"]);
    vec3<double> axis{aq[0],aq[1],aq[2]},v{b[0],b[1],b[2]};auto av=cross(axis,v),aav=cross(axis,av);
    append(t.f64[prefix+"cross.0"],av);append(t.f64[prefix+"cross.1"],aav);state<double> out{};
    for(size_t k=0;k<3;++k)out[k]=a[k]+a[7]*(v[k]+(av[k]*aq[3]+aav[k])*2.);
    auto q=multiply(aq,bq);std::copy(q.begin(),q.end(),out.begin()+3);out[7]=a[7]*b[7];append(t.f64[prefix+"product.0"],out);return out;
}
void validate(const mhr_skeleton_data &d){
    require(d.offsets.size()==joints*3 && d.prerotations.size()==joints*4 && d.prefix.size()==prefix_count*2 && d.parents.size()==joints,"MHR skeleton parameter size mismatch");finite(d.offsets);finite(d.prerotations);
    require(d.parents[0]==-1,"MHR root parent mismatch");
    for(size_t j=0;j<joints;++j){
        if(j)require(d.parents[j]>=0 && size_t(d.parents[j])<j,"MHR parent topology mismatch");
        double norm=0;for(size_t k=0;k<4;++k)norm+=double(d.prerotations[j*4+k])*d.prerotations[j*4+k];
        require(norm>1e-12 && std::isfinite(norm),"MHR invalid prerotation");
    }
    // Validate the stored parallel scan, not merely index domains. All reads
    // precede writes in a group; its final chains must equal the parent tree.
    std::array<std::vector<int32_t>,joints> paths;
    for(size_t j=0;j<joints;++j)paths[j]={int32_t(j)};
    size_t offset=0;
    for(auto count:groups){auto next=paths;std::array<bool,joints> seen{};
        for(size_t i=offset;i<offset+count;++i){auto source=d.prefix[i],target=d.prefix[prefix_count+i];
            require(source>0 && size_t(source)<joints && target>=0 && size_t(target)<joints && !seen[source],"MHR invalid prefix index");seen[source]=true;
            require(d.parents[paths[source].front()]==paths[target].back(),"MHR prefix does not compose adjacent ancestor chains");
            next[source]=paths[target];next[source].insert(next[source].end(),paths[source].begin(),paths[source].end());require(next[source].size()<=joints,"MHR invalid prefix chain");
        }paths=std::move(next);offset+=count;
    }
    for(size_t j=0;j<joints;++j){std::vector<int32_t> expected;for(int32_t k=int32_t(j);k>=0;k=d.parents[k])expected.push_back(k);std::reverse(expected.begin(),expected.end());require(paths[j]==expected,"MHR incomplete prefix schedule");}
}
}
mhr_skeleton_data load_mhr_skeleton(tensor_archive &archive){
    mhr_skeleton_data d{archive.read("skeleton.offsets",joints*3*4),archive.read("skeleton.prerotations",joints*4*4),
        archive.read_i32("skeleton.prefix",prefix_count*2*4),archive.read_i32("skeleton.parents",joints*4)};validate(d);return d;
}
mhr_skeleton_taps mhr_local_skeleton(uint32_t batch,std::span<const float> jp,const mhr_skeleton_data &d){
    require(batch>=1 && batch<=2 && jp.size()==batch*joints*7,"MHR joint input shape mismatch");validate(d);
    require(std::all_of(jp.begin(),jp.end(),[](float x){return std::isfinite(x);}),"nonfinite MHR joint input");mhr_skeleton_taps t;
    auto &local=t.f32["10.local"];local.reserve(batch*joints*8);
    for(size_t b=0;b<batch;++b)for(size_t j=0;j<joints;++j){auto p=jp.subspan((b*joints+j)*7,7);
        float cy=std::cos(p[5]*.5f),sy=std::sin(p[5]*.5f),cp=std::cos(p[4]*.5f),sp=std::sin(p[4]*.5f),cr=std::cos(p[3]*.5f),sr=std::sin(p[3]*.5f);
        t.f32["02.cy"].push_back(cy);t.f32["02.sy"].push_back(sy);t.f32["02.cp"].push_back(cp);t.f32["02.sp"].push_back(sp);t.f32["02.cr"].push_back(cr);t.f32["02.sr"].push_back(sr);
        quat<float> q{(sr*cp)*cy-(cr*sp)*sy,(cr*sp)*cy+(sr*cp)*sy,(cr*cp)*sy-(sr*sp)*cy,(cr*cp)*cy+(sr*sp)*sy};append(t.f32["03.euler_quaternion"],q);
        q=multiply(quat<float>{d.prerotations[j*4],d.prerotations[j*4+1],d.prerotations[j*4+2],d.prerotations[j*4+3]},q);append(t.f32["04.local_quaternion"],q);
        vec3<float> tr{p[0]+d.offsets[j*3],p[1]+d.offsets[j*3+1],p[2]+d.offsets[j*3+2]};append(t.f32["05.local_translation"],tr);
        float scale=std::exp(p[6]*0.69314718246459961f);require(std::isfinite(scale) && scale>0,"MHR local scale overflow/underflow");t.f32["06.local_scale"].push_back(scale);
        append(local,tr);append(local,q);local.push_back(scale);
    }
    for(auto &[_,v]:t.f32)finite(v);
    std::vector<state<double>> global(batch*joints);for(size_t i=0;i<global.size();++i)std::copy_n(local.begin()+i*8,8,global[i].begin());size_t offset=0;
    for(size_t g=0;g<groups.size();++g){auto next=global;std::string prefix="19.prefix."+std::to_string(g)+".";
        for(size_t b=0;b<batch;++b)for(size_t i=offset;i<offset+groups[g];++i){size_t s=d.prefix[i],target=d.prefix[prefix_count+i];next[b*joints+s]=compose(global[b*joints+target],global[b*joints+s],t,prefix);}
        global=std::move(next);for(auto &v:global)append(t.f64["20.prefix."+std::to_string(g)],v);offset+=groups[g];
    }
    for(auto &[_,v]:t.f64)finite(v);for(auto &s:global)for(double v:s)t.f32["90.skeleton"].push_back(float(v));finite(t.f32.at("90.skeleton"));return t;
}
mhr_skeleton_taps mhr_skeleton(neural_session &session,tensor_archive &archive,uint32_t batch,std::span<const float> parameters){
    require(batch>=1 && batch<=2 && parameters.size()==size_t(batch)*204,"MHR parameter shape mismatch");
    require(std::all_of(parameters.begin(),parameters.end(),[](float x){return std::isfinite(x);}),"nonfinite MHR model input");
    auto data=load_mhr_skeleton(archive); // Validate before allocating any graph.
    auto weight=archive.pin("parameter.matrix",889*249*4);std::vector<float> padded(size_t(batch)*249,0),jp(size_t(batch)*889);
    for(size_t b=0;b<batch;++b)std::copy_n(parameters.begin()+b*204,204,padded.begin()+b*249);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({ggml_tensor_overhead()*8+ggml_graph_overhead_custom(8,false),nullptr,true}),ggml_free);if(!context)throw std::bad_alloc();
    auto w=session.parameter(context.get(),weight,249,889),x=ggml_new_tensor_2d(context.get(),GGML_TYPE_F32,249,batch);
    auto y=ggml_mul_mat(context.get(),w,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);ggml_set_output(y);
    require(ggml_backend_supports_op(session.backend(),y),"unsupported MHR parameter projection");auto graph=ggml_new_graph_custom(context.get(),8,false);ggml_build_forward_expand(graph,y);
    auto buffer=session.allocate(context.get());
    const neural_session::upload inputs[]={{x,padded.data(),padded.size()*4}};
    const neural_session::download outputs[]={{y,jp.data(),jp.size()*4}};
    if(session.compute(graph,inputs,outputs)!=GGML_STATUS_SUCCESS)throw std::runtime_error("MHR parameter projection failed");
    auto result=mhr_local_skeleton(batch,jp,data);result.f32["00.padded"]=std::move(padded);result.f32["01.joint_parameters"]=std::move(jp);return result;
}
}
