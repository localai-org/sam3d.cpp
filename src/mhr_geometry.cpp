// Copyright (c) Meta Platforms, Inc. and affiliates.
// Adapted from the original released MHR v1.0.1 forward and embedded Momentum
// quaternion/skinning functions. See LICENSES/MHR-Apache-2.0.txt and Momentum.txt.
#include "mhr_geometry.hpp"
#include "finite.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace sam3d {
namespace {
constexpr size_t vertices=18439,coordinates=vertices*3,joints=127,influences=51337;
void require(bool v,const char *s){if(!v)throw std::invalid_argument(s);}
void finite(std::span<const float> v){require(all_finite_f32(v),"nonfinite MHR geometry tensor");}
template<size_t N> void append(std::vector<float> &v,const std::array<float,N> &a){v.insert(v.end(),a.begin(),a.end());}
using quat=std::array<float,4>;using vec3=std::array<float,3>;using state=std::array<float,8>;
quat multiply(quat a,quat b){return {((a[3]*b[0]+a[0]*b[3])+a[1]*b[2])-a[2]*b[1],((a[3]*b[1]-a[0]*b[2])+a[1]*b[3])+a[2]*b[0],
    ((a[3]*b[2]+a[0]*b[1])-a[1]*b[0])+a[2]*b[3],((a[3]*b[3]-a[0]*b[0])-a[1]*b[1])-a[2]*b[2]};}
quat normalized(quat q,float *length=nullptr){float n=std::sqrt(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]);if(length)*length=n;for(auto &v:q)v/=std::max(n,1e-12f);return q;}
vec3 cross(vec3 a,vec3 b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
vec3 rotate(quat q,vec3 v){vec3 axis{q[0],q[1],q[2]};auto av=cross(axis,v),aav=cross(axis,av);for(size_t k=0;k<3;++k)v[k]+=(av[k]*q[3]+aav[k])*2.f;return v;}
state compose(state a,state b){auto aq=normalized({a[3],a[4],a[5],a[6]}),bq=normalized({b[3],b[4],b[5],b[6]});auto v=rotate(aq,{b[0],b[1],b[2]});state r{};
    for(size_t k=0;k<3;++k)r[k]=a[k]+a[7]*v[k];auto q=multiply(aq,bq);std::copy(q.begin(),q.end(),r.begin()+3);r[7]=a[7]*b[7];return r;}

// F32 matrix projection. Right-layout blendshape matrices are transposed inside
// GGML, not reinterpreted. A sparse COO projection may be materialized as a
// dense matrix here; its numerical oracle remains original sparse PyTorch.
std::vector<float> project(neural_session &session,uint32_t batch,size_t input,size_t output,
                          std::span<const float> x,const validated_weights &snapshot,bool right_layout,
                          std::vector<float> *relu=nullptr,bool resident=true){
    const auto weight=snapshot.values();
    require(x.size()==batch*input && weight.size()==input*output,"MHR projection shape mismatch");finite(x);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*24+ggml_graph_overhead_custom(24,false),nullptr,true}),ggml_free);if(!c)throw std::bad_alloc();
    auto wt=resident?session.constant(c.get(),snapshot,right_layout?output:input,right_layout?input:output):ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,right_layout?output:input,right_layout?input:output);
    auto xt=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,input,batch);
    auto w=right_layout?ggml_cont(c.get(),ggml_transpose(c.get(),wt)):wt;auto y=ggml_mul_mat(c.get(),w,xt);ggml_mul_mat_set_prec(y,GGML_PREC_F32);ggml_set_output(y);
    auto activated=relu?ggml_relu(c.get(),y):nullptr;if(activated)ggml_set_output(activated);auto graph=ggml_new_graph_custom(c.get(),24,false);ggml_build_forward_expand(graph,activated?activated:y);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported MHR graph operation");
    auto buffer=session.allocate(c.get());
    std::vector<neural_session::upload> inputs{{xt,x.data(),x.size_bytes()}};
    if(!resident)inputs.push_back({wt,weight.data(),weight.size_bytes()});
    std::vector<float> result(batch*output);
    std::vector<neural_session::download> outputs{{y,result.data(),result.size()*4}};
    if(relu){relu->resize(result.size());outputs.push_back({activated,relu->data(),relu->size()*4});}
    if(session.compute(graph,inputs,outputs)!=GGML_STATUS_SUCCESS)throw std::runtime_error("MHR projection graph failed");
    finite(result);if(relu)finite(*relu);return result;
}
}
named_floats mhr_geometry(neural_session &session,tensor_archive &archive,uint32_t batch,
                         std::span<const float> identity,std::span<const float> parameters,std::span<const float> face,bool correctives,bool skin_operation_taps){
    require(batch>=1 && batch<=2 && identity.size()==size_t(batch)*45 && parameters.size()==size_t(batch)*204 && face.size()==size_t(batch)*72,"invalid MHR geometry input shape");finite(identity);finite(parameters);finite(face);
    named_floats t;
    {auto w=archive.pin("identity.vectors",45*coordinates*4);t["01.identity_projection"]=project(session,batch,45,coordinates,identity,w,true);}
    auto base=archive.pin("identity.base",coordinates*4);auto &rest=t["02.identity_rest"];rest=t.at("01.identity_projection");for(size_t i=0;i<rest.size();++i)rest[i]+=base[i%coordinates];
    {auto skeleton=mhr_skeleton(session,archive,batch,parameters);t["10.joint_parameters"]=std::move(skeleton.f32.at("01.joint_parameters"));t["11.skeleton"]=std::move(skeleton.f32.at("90.skeleton"));}
    {auto w=archive.pin("expression.vectors",72*coordinates*4);t["03.expression"]=project(session,batch,72,coordinates,face,w,true);}
    auto &linear=t["04.linear_unposed"];linear=rest;const auto &expression=t.at("03.expression");for(size_t i=0;i<linear.size();++i)linear[i]+=expression[i];t["26.unposed"]=linear;
    if(correctives){auto &jp=t.at("10.joint_parameters"),&features=t["22.pose_features"];
        for(size_t b=0;b<batch;++b)for(size_t j=2;j<joints;++j){size_t offset=(b*joints+j)*7;vec3 cs{},sn{};for(size_t k=0;k<3;++k){cs[k]=std::cos(jp[offset+3+k]);sn[k]=std::sin(jp[offset+3+k]);}
            append(t["20.pose_cos"],cs);append(t["21.pose_sin"],sn);auto [cx,cy,cz]=cs;auto [sx,sy,sz]=sn;
            // Original first two matrix columns, then subtract identity at
            // features 0/4. Not first two rows, nor normalized quaternion input.
            append(features,std::array<float,6>{cy*cz-1.f,cy*sz,-sy,(-cx)*sz+(sx*sy)*cz,(cx*cz+(sx*sy)*sz)-1.f,sx*cy});
        }
        {auto dense=archive.pin_mhr_sparse_projection();
            t["23.pose_sparse_projection"]=project(session,batch,750,3000,features,dense,false,&t["24.pose_relu"]);
        }
        {auto w=archive.pin("pose.dense.weight",coordinates*3000*4);t["25.pose_correctives"]=project(session,batch,3000,coordinates,t.at("24.pose_relu"),w,false);}
        auto &unposed=t["26.unposed"];const auto &corrected=t.at("25.pose_correctives");for(size_t i=0;i<unposed.size();++i)unposed[i]+=corrected[i];
    }
    auto inverse_bind=archive.pin("skin.inverse_bind",joints*8*4),weights=archive.pin("skin.weights",influences*4);
    auto skin_joints=archive.read_i32("skin.joints",influences*4),skin_vertices=archive.read_i32("skin.vertices",influences*4);
    auto skin=mhr_skinning(batch,vertices,t.at("11.skeleton"),t.at("26.unposed"),inverse_bind,skin_joints,weights,skin_vertices,skin_operation_taps);t.merge(skin);
    for(auto &[_,v]:t)finite(v);return t;
}
named_floats mhr_skinning(uint32_t batch,uint32_t vertex_count,std::span<const float> skeleton,std::span<const float> unposed,
                         std::span<const float> inverse_bind,std::span<const int32_t> skin_joints,std::span<const float> weights,std::span<const int32_t> skin_vertices,bool operation_taps){
    require(batch>=1 && batch<=2 && vertex_count>=1 && vertex_count<=18439 && weights.size()>=1 && weights.size()<=51337,"invalid MHR skin shape");
    const size_t vertices=vertex_count,coordinates=vertices*3,influences=weights.size();
    require(skeleton.size()==batch*joints*8 && unposed.size()==batch*coordinates && inverse_bind.size()==joints*8 && skin_joints.size()==influences && skin_vertices.size()==influences,"MHR skin tensor shape mismatch");
    finite(skeleton);finite(unposed);finite(inverse_bind);finite(weights);
    for(size_t i=0;i<batch*joints;++i){auto s=skeleton.subspan(i*8,8);float norm=0;for(size_t k=3;k<7;++k)norm+=s[k]*s[k];require(norm>1e-12f && std::isfinite(norm) && s[7]>0,"invalid MHR global skeleton state");}
    for(size_t i=0;i<influences;++i)require(skin_joints[i]>=0 && size_t(skin_joints[i])<joints && skin_vertices[i]>=0 && size_t(skin_vertices[i])<vertices,"invalid MHR skin index");
    named_floats t;
    std::vector<double> total(vertices,0);
    for(size_t i=0;i<influences;++i){require(weights[i]>=0 && weights[i]<=1,"MHR skin weight out of range");total[skin_vertices[i]]+=weights[i];}
    for(double sum:total)require(std::abs(sum-1.)<2e-5,"MHR skin weights not normalized or vertex missing");
    auto &joint_state=t["30.skin_joint_state"];joint_state.reserve(batch*joints*8);
    for(size_t b=0;b<batch;++b)for(size_t j=0;j<joints;++j){state a{},bind{};std::copy_n(skeleton.begin()+(b*joints+j)*8,8,a.begin());std::copy_n(inverse_bind.begin()+j*8,8,bind.begin());
        float norm=0;for(size_t k=3;k<7;++k)norm+=bind[k]*bind[k];require(norm>1e-12f && std::isfinite(norm) && bind[7]>0,"invalid MHR inverse bind state");append(joint_state,compose(a,bind));
    }
    for(auto &[_,v]:t)finite(v);
    auto &output=t["90.vertices"];output.resize(batch*coordinates,0.f);
    // A joint's composed rotation is shared by all of its vertex influences.
    // Preserve BOTH upstream normalization steps, but evaluate each once per
    // joint. Keep their intermediate values for the unchanged diagnostic taps.
    struct rotation {quat first,second;float norm0,norm1;};
    std::vector<rotation> rotations(batch*joints);
    for(size_t j=0;j<rotations.size();++j){const auto s=std::span(joint_state).subspan(j*8,8);auto &r=rotations[j];
        r.first=normalized({s[3],s[4],s[5],s[6]},&r.norm0);r.second=normalized(r.first,&r.norm1);
    }
    for(size_t b=0;b<batch;++b){size_t i=0;
#if defined(__SSE2__) || defined(_M_X64)
    const auto flag=std::getenv("SAM3D_SIMD_SKINNING");
    if(!operation_taps && flag && std::strcmp(flag,"1")==0)for(;influences-i>=4;i+=4){
        // Four independent influences, with exactly the scalar expression's
        // operation order (this source disables FP contraction). Scatter in
        // original influence order: duplicate vertex IDs must not be reduced
        // in parallel or reordered. All gathers are within validated spans.
        const float *s[4],*p[4],*q[4];size_t vertex[4];
        for(size_t lane=0;lane<4;++lane){const size_t joint=b*joints+skin_joints[i+lane];
            vertex[lane]=b*vertices+skin_vertices[i+lane];s[lane]=joint_state.data()+joint*8;
            p[lane]=unposed.data()+vertex[lane]*3;q[lane]=rotations[joint].second.data();}
        auto gather=[](const float *const *v,size_t k){return _mm_set_ps(v[3][k],v[2][k],v[1][k],v[0][k]);};
        __m128 point[3],axis[3],av[3],aav[3];const auto scale=gather(s,7),qw=gather(q,3);
        for(size_t k=0;k<3;++k){point[k]=_mm_mul_ps(gather(p,k),scale);axis[k]=gather(q,k);}
        for(size_t k=0;k<3;++k){const size_t a=(k+1)%3,c=(k+2)%3;
            av[k]=_mm_sub_ps(_mm_mul_ps(axis[a],point[c]),_mm_mul_ps(axis[c],point[a]));}
        for(size_t k=0;k<3;++k){const size_t a=(k+1)%3,c=(k+2)%3;
            aav[k]=_mm_sub_ps(_mm_mul_ps(axis[a],av[c]),_mm_mul_ps(axis[c],av[a]));}
        const auto w=_mm_loadu_ps(weights.data()+i),two=_mm_set1_ps(2.f);float weighted[3][4];
        for(size_t k=0;k<3;++k){auto rotated=_mm_add_ps(point[k],_mm_mul_ps(_mm_add_ps(_mm_mul_ps(av[k],qw),aav[k]),two));
            _mm_storeu_ps(weighted[k],_mm_mul_ps(_mm_add_ps(gather(s,k),rotated),w));}
        for(size_t lane=0;lane<4;++lane)for(size_t k=0;k<3;++k)output[vertex[lane]*3+k]+=weighted[k][lane];
    }
#endif
    for(;i<influences;++i){state s{};size_t j=skin_joints[i],v=skin_vertices[i];std::copy_n(joint_state.begin()+(b*joints+j)*8,8,s.begin());
        vec3 point{};std::copy_n(unposed.begin()+(b*vertices+v)*3,3,point.begin());if(operation_taps){append(t["31.skin_selected_state"],s);append(t["32.skin_selected_points"],point);}
        const auto &r=rotations[b*joints+j];const auto q=r.second;
        if(operation_taps){t["33.skin_norm.0"].push_back(r.norm0);append(t["33.skin_normalized.0"],r.first);
            t["33.skin_norm.1"].push_back(r.norm1);append(t["33.skin_normalized.1"],r.second);}
        for(auto &p:point)p*=s[7];vec3 axis{q[0],q[1],q[2]};auto av=cross(axis,point),aav=cross(axis,av);if(operation_taps){append(t["34.skin_cross.0"],av);append(t["34.skin_cross.1"],aav);}
        vec3 transformed{},weighted{};for(size_t k=0;k<3;++k){transformed[k]=s[k]+(point[k]+(av[k]*q[3]+aav[k])*2.f);weighted[k]=transformed[k]*weights[i];output[(b*vertices+v)*3+k]+=weighted[k];}
        if(operation_taps){append(t["35.skin_transformed"],transformed);append(t["36.skin_weighted"],weighted);}
    }
    }
    for(auto &[_,v]:t)finite(v);return t;
}
}
