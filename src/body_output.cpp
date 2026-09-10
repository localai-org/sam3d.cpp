// Copyright (c) Meta Platforms, Inc. and affiliates. SAM license.
// RoMa/SciPy quaternion matrix adaptation: NAVER Corp. / SciPy contributors,
// BSD-3-Clause, LICENSES/RoMa.txt. See THIRD_PARTY_NOTICES.md.
#include "body_output.hpp"
#include "finite.hpp"
#include "body_hand_frame.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
namespace sam3d {
namespace {
void require(bool b,const char *s){if(!b)throw std::invalid_argument(s);}
void finite(std::span<const float> v){require(all_finite_f32(v),"nonfinite Body output input/result");}
}
named_floats body_map_geometry(neural_session &session,uint32_t batch,uint32_t vertices,
    std::span<const float> vertices_cm,std::span<const float> skeleton,const validated_weights &mapping,scalar_division arithmetic,bool hand){
    require(batch>=1 && batch<=2 && vertices>=1 && vertices<=18439,"invalid Body mapping dimensions");
    const size_t b=batch,v=vertices,n=v+127;
    require(vertices_cm.size()==b*v*3 && skeleton.size()==b*127*8 && mapping.size()==308*n,"Body mapping input shape mismatch");
    require(arithmetic==scalar_division::direct || arithmetic==scalar_division::reciprocal_multiply,"invalid Body mapping arithmetic");finite(vertices_cm);finite(skeleton);
    named_floats t;auto meters=[&](float x){return arithmetic==scalar_division::direct?x/100.f:x*.01f;};
    auto &meter_vertices=t["00.vertices_m"];meter_vertices.reserve(vertices_cm.size());
    for(float x:vertices_cm)meter_vertices.push_back(meters(x));
    for(size_t i=0;i<b*127;++i){auto s=skeleton.subspan(i*8,8);for(size_t k=0;k<3;++k)t["01.joints_m"].push_back(meters(s[k]));
        float x=s[3],y=s[4],z=s[5],w=s[6];for(size_t k=3;k<7;++k)t["02.quaternions"].push_back(s[k]);
        // RoMa unitquat_to_rotmat explicitly does not normalize; diagonals
        // contain all four squares, not the unit-only 1-2*(...) shortcut.
        float x2=x*x,y2=y*y,z2=z*z,w2=w*w,xy=x*y,zw=z*w,xz=x*z,yw=y*w,yz=y*z,xw=x*w;
        std::array<float,9> r{((x2-y2)-z2)+w2,2.f*(xy-zw),2.f*(xz+yw),2.f*(xy+zw),((-x2+y2)-z2)+w2,2.f*(yz-xw),2.f*(xz-yw),2.f*(yz+xw),((-x2-y2)+z2)+w2};
        t["03.joint_rotations"].insert(t["03.joint_rotations"].end(),r.begin(),r.end());
    }
    auto &combined=t["04.vertex_joints"];combined.resize(b*n*3);auto &input=t["05.mapping_input"];input.resize(n*b*3);
    for(size_t i=0;i<b;++i){std::copy_n(t.at("00.vertices_m").begin()+i*v*3,v*3,combined.begin()+i*n*3);std::copy_n(t.at("01.joints_m").begin()+i*127*3,127*3,combined.begin()+(i*n+v)*3);}
    for(size_t i=0;i<b;++i)for(size_t j=0;j<n;++j)for(size_t k=0;k<3;++k)input[j*b*3+i*3+k]=combined[(i*n+j)*3+k];
    std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*24+ggml_graph_overhead_custom(24,false),nullptr,true}),ggml_free);if(!c)throw std::bad_alloc();
    auto wt=session.parameter(c.get(),mapping,n,308),xt=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,b*3,n);
    auto x=ggml_cont(c.get(),ggml_transpose(c.get(),xt));auto y=ggml_mul_mat(c.get(),wt,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);auto output=ggml_cont(c.get(),ggml_transpose(c.get(),y));ggml_set_output(output);
    auto graph=ggml_new_graph_custom(c.get(),24,false);ggml_build_forward_expand(graph,output);
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported Body mapping operation");
    auto buffer=session.allocate(c.get());
    auto &projected=t["06.keypoints_linear"];projected.resize(308*b*3);
    const neural_session::upload inputs[]={{xt,input.data(),input.size()*4}};
    const neural_session::download outputs[]={{output,projected.data(),projected.size()*4}};
    if(session.compute(graph,inputs,outputs)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Body mapping graph failed");
    auto &all=t["07.keypoints308"];all.resize(b*308*3);auto &first=t["08.first70"];first.resize(b*70*3);
    for(size_t i=0;i<b;++i)for(size_t j=0;j<308;++j)for(size_t k=0;k<3;++k){all[(i*308+j)*3+k]=projected[(j*b+i)*3+k];if(j<70)first[(i*70+j)*3+k]=projected[(j*b+i)*3+k];}
    if(hand){all=body_hand_mask_keypoints(batch,all);for(size_t i=0;i<b;++i)std::copy_n(all.begin()+i*308*3,70*3,first.begin()+i*70*3);}
    t["90.vertices"]=t.at("00.vertices_m");t["91.joints"]=t.at("01.joints_m");t["92.keypoints"]=first;
    for(const char *name:{"90.vertices","91.joints","92.keypoints"}){auto &values=t.at(name);for(size_t i=0;i<values.size();++i)if(i%3)values[i]*=-1.f;}
    for(auto &[_,values]:t)finite(values);return t;
}
named_floats body_pose_geometry(neural_session &session,tensor_archive &archive,pose_shape s,
    std::span<const float> token,std::span<const float> initial,std::span<const int32_t> hand_indices,
    const weight_map &parameters,const validated_weights &mapping,scalar_division arithmetic,const hand_pose_config *hand){
    validate_pose_shape(s);require(arithmetic==scalar_division::direct || arithmetic==scalar_division::reciprocal_multiply,"invalid Body mapping arithmetic");
    require(mapping.size()==308*(18439+127),"full Body keypoint mapping shape mismatch");
    auto pose=body_pose(session,s,token,initial,hand_indices,parameters,hand);
    // Original MHRHead does not forward its do_pcblend flag; the released
    // TorchScript therefore uses its default apply_correctives=True.
    auto geometry=mhr_geometry(session,archive,s.batch,pose.at("24.shape"),pose.at("90.model_params"),pose.at("27.face"),true,false);
    auto output=body_map_geometry(session,s.batch,18439,geometry.at("90.vertices"),geometry.at("11.skeleton"),mapping,arithmetic,hand!=nullptr);
    named_floats result;for(auto &[name,x]:output)result["map."+name]=std::move(x);
    for(auto name:{"10.pred","24.shape","25.scale","26.hand","27.face","90.model_params"})result["pose."+std::string(name)]=pose.at(name);
    if(hand)for(auto &[name,x]:pose)result["pose."+name]=x;
    result["mhr.vertices_cm"]=std::move(geometry.at("90.vertices"));result["mhr.skeleton"]=std::move(geometry.at("11.skeleton"));
    result["pred_pose_raw"].resize(size_t(s.batch)*266);for(size_t i=0;i<s.batch;++i)std::copy_n(pose.at("10.pred").begin()+i*519,266,result["pred_pose_raw"].begin()+i*266);
    result["global_rot"]=std::move(pose.at("17.global.euler"));result["body_pose"]=std::move(pose.at("23.body.masked"));return result;
}
}
