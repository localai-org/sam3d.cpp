// Copyright (c) Meta Platforms, Inc. and affiliates. SAM license.
// Rotation quaternion/Euler adaptations: Copyright (c) 2020 NAVER Corp.;
// SciPy contributors. BSD-3-Clause; see LICENSES/RoMa.txt and NOTICE.
#include "body_pose.hpp"
#include "finite.hpp"
#include "body_hand_frame.hpp"
#include "ggml.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <stdexcept>
namespace sam3d {
namespace {
void require(bool b,const char *s){if(!b)throw std::invalid_argument(s);}
void finite(std::span<const float> v){require(all_finite_f32(v),"non-finite pose tensor");}
using vec3=std::array<float,3>;
vec3 cross(vec3 a,vec3 b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
vec3 normalize(vec3 a){float n=std::max(std::sqrt((a[0]*a[0]+a[1]*a[1])+a[2]*a[2]),1e-12f);for(auto &v:a)v/=n;return a;}
void append(std::vector<float> &v,std::span<const float> x){v.insert(v.end(),x.begin(),x.end());}
std::array<float,9> matrix(vec3 x,vec3 y,vec3 z){return {x[0],y[0],z[0],x[1],y[1],z[1],x[2],y[2],z[2]};}
// mhr_utils.batchXYZfrom6D: cross-product construction, not global Gram-Schmidt.
vec3 xyz(std::span<const float> p,named_floats &r,const std::string &prefix){
    vec3 a{p[0],p[1],p[2]},b{p[3],p[4],p[5]};auto x=normalize(a),z=normalize(cross(x,b)),y=cross(z,x);auto m=matrix(x,y,z);
    append(r[prefix+".matrix"],m);float sy=std::sqrt(m[0]*m[0]+m[3]*m[3]);r[prefix+".sy"].push_back(sy);
    float singular=sy<1e-6f?1.f:0.f;r[prefix+".singular"].push_back(singular);
    vec3 out{std::atan2(m[7],m[8])*(1.f-singular)+std::atan2(-m[5],m[4])*singular,
        std::atan2(-m[6],sy)*(1.f-singular)+std::atan2(-m[6],sy)*singular,
        std::atan2(m[3],m[0])*(1.f-singular)+(m[3]*0.f)*singular};append(r[prefix+".euler"],out);return out;
}
// RoMa rotmat_to_unitquat followed by unitquat_to_euler("ZYX"). Preserve its
// quaternion choice and singular branch; a direct asin formula differs there.
vec3 matrix_euler_zyx(const std::array<float,9> &m,named_floats &r,bool diagnostic=false){
    std::array<float,4> decision{m[0],m[4],m[8],(m[0]+m[4])+m[8]},q{};
    int choice=int(std::max_element(decision.begin(),decision.end())-decision.begin());r["15.global.choice"].push_back(float(choice));
    if(choice!=3){int i=choice,j=(i+1)%3,k=(j+1)%3;q[i]=(1.f-decision[3])+2.f*m[i*3+i];q[j]=m[j*3+i]+m[i*3+j];q[k]=m[k*3+i]+m[i*3+k];q[3]=m[k*3+j]-m[j*3+k];}
    else{q={m[7]-m[5],m[2]-m[6],m[3]-m[1],1.f+decision[3]};}
    float norm=std::sqrt(((q[0]*q[0]+q[1]*q[1])+q[2]*q[2])+q[3]*q[3]);require(norm>0 && std::isfinite(norm),"undefined global quaternion");for(auto &v:q)v/=norm;append(r["16.global.quaternion"],q);
    float aa=q[3]-q[1],bb=q[0]+q[2],cc=q[1]+q[3],dd=q[2]-q[0];
    float middle=2.f*std::atan2(std::hypot(cc,dd),std::hypot(aa,bb));const float pi=std::numbers::pi_v<float>;
    if(diagnostic){append(r["diagnostic.abcd"],std::array<float,4>{aa,bb,cc,dd});append(r["diagnostic.hypot"],std::array<float,2>{std::hypot(cc,dd),std::hypot(aa,bb)});r["diagnostic.middle"].push_back(middle);}
    bool case1=std::abs(middle)<=1e-7f,case2=std::abs(middle-pi)<=1e-7f;
    float sum=std::atan2(bb,aa),diff=std::atan2(dd,cc);vec3 out{sum+diff,middle,sum-diff};
    if(case1 || case2)out[2]=0;if(case1)out[0]=2.f*sum;if(case2)out[0]=2.f*diff;out[1]-=pi*.5f;
    for(auto &v:out){if(v<-pi)v+=2.f*pi;if(v>pi)v-=2.f*pi;}append(r["17.global.euler"],out);return out;
}
vec3 global_euler(std::span<const float> p,named_floats &r,bool diagnostic=false){
    vec3 a{p[0],p[1],p[2]},b{p[3],p[4],p[5]};auto x=normalize(a);float dot=(x[0]*b[0]+x[1]*b[1])+x[2]*b[2];
    vec3 residual;for(size_t k=0;k<3;++k)residual[k]=b[k]-dot*x[k];auto y=normalize(residual),z=cross(x,y);auto m=matrix(x,y,z);
    append(r["11.global.b1"],x);append(r["12.global.b2"],y);append(r["13.global.b3"],z);append(r["14.global.matrix"],m);
    return matrix_euler_zyx(m,r,diagnostic);
}
constexpr int indices3[23][3]={{0,2,4},{6,8,10},{12,13,14},{15,16,17},{18,19,20},{21,22,23},{24,25,26},{27,28,29},{34,35,36},{37,38,39},{44,45,46},{53,54,55},{64,65,66},{85,69,73},{86,70,79},{87,71,82},{88,72,76},{91,92,93},{112,96,100},{113,97,106},{114,98,109},{115,99,103},{130,131,132}};
constexpr int indices1[]={1,3,5,7,9,11,30,31,32,33,40,41,42,43,47,48,49,50,51,52,56,57,58,59,60,61,62,63,67,68,74,75,77,78,80,81,83,84,89,90,94,95,101,102,104,105,107,108,110,111,116,117,118,119,120,121,122,123};
std::array<float,133> body_parameters(std::span<const float> p,named_floats &r){std::array<float,133> out{};
    for(size_t i=0;i<23;++i){auto e=xyz(p.subspan(i*6,6),r,"20.body");for(size_t k=0;k<3;++k)out[indices3[i][k]]=e[k];}
    for(size_t i=0;i<58;++i){float e=std::atan2(p[138+i*2],p[139+i*2]);out[indices1[i]]=e;r["21.body.angles1"].push_back(e);}
    std::copy_n(p.begin()+254,6,out.begin()+124);append(r["22.body.unmasked"],out);std::fill(out.begin()+62,out.begin()+116,0.f);std::fill(out.begin()+130,out.end(),0.f);append(r["23.body.masked"],out);return out;
}
std::array<float,27> hand_parameters(std::span<const float> p,named_floats &r,const std::string &prefix){
    constexpr int dofs[]={3,1,1,3,1,1,3,1,1,3,1,1,2,3,1,1};std::array<float,27> out{};size_t src=0,dst=0;
    for(int n:dofs){if(n==3){auto e=xyz(p.subspan(src,6),r,prefix);std::copy(e.begin(),e.end(),out.begin()+dst);}
        else for(int k=0;k<n;++k){float e=std::atan2(p[src+2*k],p[src+2*k+1]);out[dst+k]=e;r[prefix+".angles1"].push_back(e);}src+=n*2;dst+=n;}
    append(r[prefix+".parameters"],out);return out;
}
struct graph_run {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx{ggml_init({ggml_tensor_overhead()*192+ggml_graph_overhead_custom(192,false),nullptr,true}),ggml_free};
    std::map<std::string,ggml_tensor *> taps;std::vector<std::pair<ggml_tensor *,std::span<const float>>> uploads;
    graph_run(){if(!ctx)throw std::bad_alloc();}
    ggml_context *c(){return ctx.get();}
    ggml_tensor *weight(neural_session &session,const validated_weights &v,uint64_t d,uint64_t rows){return session.parameter(c(),v,d,rows);}
    ggml_tensor *input(std::span<const float> v,uint64_t d,uint64_t rows){auto t=ggml_new_tensor_2d(c(),GGML_TYPE_F32,d,rows);uploads.emplace_back(t,v);return t;}
    ggml_tensor *tap(const std::string &name,ggml_tensor *t){ggml_set_output(t);ggml_set_name(t,name.c_str());taps[name]=t;return t;}
    ggml_tensor *mm(ggml_tensor *w,ggml_tensor *x){auto y=ggml_mul_mat(c(),w,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);return y;}
    named_floats run(neural_session &session){auto graph=ggml_new_graph_custom(c(),192,false);for(auto &[_,t]:taps)ggml_build_forward_expand(graph,t);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)require(ggml_backend_supports_op(session.backend(),ggml_graph_node(graph,i)),"unsupported pose operation");
        auto buffer=session.allocate(c());
        if(!buffer)throw std::bad_alloc();auto r=session.evaluate_f32(graph,uploads,taps);
        for(auto &[_,v]:r)finite(v);return r;}
};
std::string layer(uint32_t i,uint32_t depth){return "proj.layers."+std::to_string(i)+(i+1<depth?".0":"");}
}
void validate_pose_shape(pose_shape s){require(s.batch>=1 && s.batch<=2 && s.dim>=1 && s.dim<=1280 && s.hidden>=1 && s.hidden<=1280 && s.depth>=1 && s.depth<=3,"invalid pose shape");}
named_floats body_global_rotation(uint32_t batch,std::span<const float> rotation6d){
    require(batch>=1 && batch<=64 && rotation6d.size()==uint64_t(batch)*6,"invalid global rotation shape");finite(rotation6d);named_floats out;
    for(uint32_t b=0;b<batch;++b)global_euler(rotation6d.subspan(b*6,6),out,true);
    for(auto &[_,v]:out)finite(v);return out;
}
named_floats body_matrix_rotation_xyz(uint32_t batch,std::span<const float> matrices){
    require(batch>=1 && batch<=64 && matrices.size()==uint64_t(batch)*9,"invalid matrix rotation shape");finite(matrices);named_floats out;
    for(uint32_t b=0;b<batch;++b){std::array<float,9> m;std::copy_n(matrices.begin()+b*9,9,m.begin());auto e=matrix_euler_zyx(m,out);
        append(out["euler_xyz"],std::array<float,3>{e[2],e[1],e[0]});}
    for(auto &[_,v]:out)finite(v);return out;
}
std::vector<std::pair<std::string,uint64_t>> pose_parameter_sizes(pose_shape s){validate_pose_shape(s);std::vector<std::pair<std::string,uint64_t>> r{{"scale_mean",68},{"scale_comps",28*68},{"hand_pose_mean",54},{"hand_pose_comps",54*54}};
    uint64_t in=s.dim;for(uint32_t i=0;i<s.depth;++i){uint64_t out=i+1==s.depth?519:s.hidden;r.emplace_back(layer(i,s.depth)+".weight",in*out);r.emplace_back(layer(i,s.depth)+".bias",out);in=out;}return r;}
named_floats body_pose(neural_session &session,pose_shape s,std::span<const float> token,std::span<const float> initial,std::span<const int32_t> indices,const weight_map &parameters,const hand_pose_config *hand){
    const auto sizes=pose_parameter_sizes(s);require(token.size()==uint64_t(s.batch)*s.dim && (initial.empty() || initial.size()==uint64_t(s.batch)*519) && indices.size()==54,"pose input size mismatch");finite(token);finite(initial);
    std::array<bool,136> seen{};for(int32_t v:indices){require(v>=6 && v<136 && !seen[v],"invalid/duplicate hand parameter index");seen[v]=true;}
    require(parameters.size()==sizes.size(),"pose parameter set mismatch");for(auto &[name,size]:sizes){require(parameters.contains(name) && parameters.at(name).size()==size,"pose parameter shape mismatch");}
    graph_run g;auto x=g.input(token,s.dim,s.batch);uint64_t in=s.dim;
    for(uint32_t i=0;i<s.depth;++i){uint64_t out=i+1==s.depth?519:s.hidden;auto name=layer(i,s.depth);
        x=g.tap("00.ffn."+std::to_string(i)+".linear",ggml_add(g.c(),g.mm(g.weight(session,parameters.at(name+".weight"),in,out),x),g.weight(session,parameters.at(name+".bias"),out,1)));
        if(i+1<s.depth)x=g.tap("00.ffn."+std::to_string(i)+".relu",ggml_relu(g.c(),x));in=out;}
    if(!initial.empty())x=ggml_add(g.c(),x,g.input(initial,519,s.batch));g.tap("10.pred",x);auto result=g.run(session);
    auto slice=[&](const std::string &name,size_t offset,size_t count){auto &out=result[name];for(uint32_t b=0;b<s.batch;++b)append(out,std::span(result.at("10.pred")).subspan(b*519+offset,count));};
    slice("24.shape",266,45);slice("25.scale",311,28);slice("26.hand",339,108);slice("27.face",447,72);for(auto &v:result["27.face"])v*=0.f;
    auto &full=result["30.full_pose"];full.resize(uint64_t(s.batch)*136,0.f);result["18.global.translation"].resize(uint64_t(s.batch)*3,0.f);
    for(uint32_t b=0;b<s.batch;++b){auto pred=std::span(result.at("10.pred")).subspan(b*519,519);auto e=global_euler(pred.first(6),result);auto pose=body_parameters(pred.subspan(6,260),result);
        std::copy(e.begin(),e.end(),full.begin()+b*136+3);std::copy_n(pose.begin(),130,full.begin()+b*136+6);}
    if(hand){
        // Upstream passes the original ZYX result to the wrist path as xyz.
        // Preserve that behavior; returned global_rot remains the pre-transfer
        // value. Only the actual MHR input receives the transferred pose.
        auto frame=body_hand_frame(s.batch,result.at("17.global.euler"),result.at("18.global.translation"),hand->local_to_world,hand->wrist,hand->root);
        for(uint32_t b=0;b<s.batch;++b)for(size_t k=0;k<3;++k){
            full[b*136+k]=frame.at("04.translation")[b*3+k]*10.f;
            full[b*136+3+k]=frame.at("02.euler")[b*3+k];
        }
        for(auto &[key,v]:frame)result["hand."+key]=std::move(v);
    }
    graph_run maps;
    auto right_mm=[&](std::span<const float> values,uint64_t input,uint64_t output,const std::string &weight,const std::string &bias){
        // PCA is x @ components (unlike a Linear's x @ weight.T).
        auto w=ggml_cont(maps.c(),ggml_transpose(maps.c(),maps.weight(session,parameters.at(weight),output,input)));
        return ggml_add(maps.c(),maps.mm(w,maps.input(values,input,s.batch)),maps.weight(session,parameters.at(bias),output,1));};
    maps.tap("31.scales",right_mm(result.at("25.scale"),28,68,"scale_comps","scale_mean"));
    std::vector<float> left,right;for(uint32_t b=0;b<s.batch;++b){append(left,std::span(result.at("26.hand")).subspan(b*108,54));append(right,std::span(result.at("26.hand")).subspan(b*108+54,54));}
    maps.tap("32.left.continuous",right_mm(left,54,54,"hand_pose_comps","hand_pose_mean"));maps.tap("33.right.continuous",right_mm(right,54,54,"hand_pose_comps","hand_pose_mean"));
    for(auto &[name,v]:maps.run(session))result[name]=std::move(v);auto &model=result["90.model_params"];
    for(uint32_t b=0;b<s.batch;++b){auto lh=hand_parameters(std::span(result.at("32.left.continuous")).subspan(b*54,54),result,"40.left"),rh=hand_parameters(std::span(result.at("33.right.continuous")).subspan(b*54,54),result,"41.right");
        auto pose=std::span(full).subspan(b*136,136);std::vector<float> assembled(pose.begin(),pose.end());for(size_t i=0;i<27;++i){assembled[indices[i]]=lh[i];assembled[indices[i+27]]=rh[i];}
        append(result["42.full_pose_hands"],assembled);append(model,assembled);append(model,std::span(result.at("31.scales")).subspan(b*68,68));}
    if(hand){result["hand.05.unmasked_parameters"]=model;model=body_hand_mask_parameters(s.batch,model,hand->nonhand_indices);}
    for(auto &[_,v]:result)finite(v);return result;
}
}
