#include "body_flow.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("expected backend and decoder-norm fixture");
    auto reject=[](auto fn){bool caught=false;try{fn();}catch(const std::invalid_argument &){caught=true;}if(!caught)throw std::runtime_error("invalid flow input accepted");};
    sam3d::body_flow_shape shape{{1,4,6,2,8,8,2,70,519,70,true,true},6,2,4,16,4,2,4,2,.8f,false,true,true};
    auto parameters=sam3d::body_flow_parameter_sizes(shape);std::set<std::string> names;
    for(auto &[name,n]:parameters)if(!n || !names.insert(name).second)throw std::runtime_error("invalid flow parameter contract");
    if(!names.contains("decoder.layers.5.ln1.weight") || !names.contains("head_pose.keypoint_mapping"))throw std::runtime_error("incomplete flow schema");
    auto hand_shape=shape;hand_shape.hand_branch=true;auto hand_parameters=sam3d::body_flow_parameter_sizes(hand_shape);std::set<std::string> hand_names;
    for(auto &[name,n]:hand_parameters)if(!n || !hand_names.insert(name).second)throw std::runtime_error("invalid hand parameter schema");
    if(hand_parameters.size()!=parameters.size()+4 || !hand_names.contains("hand_pe_layer.positional_encoding_gaussian_matrix") || !hand_names.contains("head_pose_hand.local_to_world_wrist") || hand_names.contains("decoder.layers.5.ln1.weight"))throw std::runtime_error("hand schema missing independent state");
    for(const char *module:{"decoder","head_pose","head_camera","init_pose","init_camera","init_to_token_mhr","prev_to_token_mhr","ray_cond_emb","keypoint_embedding","keypoint3d_embedding","keypoint_posemb_linear","keypoint3d_posemb_linear","keypoint_feat_linear"}){
        const std::string name=std::string(module)+".weight";
        if(sam3d::body_flow_parameter_name(name,true)!=std::string(module)+"_hand.weight" || sam3d::body_flow_parameter_name(name,false)!=name)throw std::runtime_error("independent hand namespace mismatch");}
    for(const char *name:{"prompt_encoder.pe_layer.positional_encoding_gaussian_matrix","prompt_to_token.weight","hand_box_embedding.weight"})
        if(sam3d::body_flow_parameter_name(name,true)!=name || !hand_names.contains(name))throw std::runtime_error("shared hand namespace changed");
    auto s=shape;auto validate=[&]{sam3d::validate_body_flow_shape(s);};
    s.depth=0;reject(validate);s=shape;s.depth=17;reject(validate);s=shape;s.condition.pose_dim=404;reject(validate);s=shape;
    s.condition.keypoints3d=false;reject(validate);s=shape;s.condition.joints=69;reject(validate);s=shape;s.condition.keypoints=69;reject(validate);s=shape;
    s.camera_scale=std::numeric_limits<float>::quiet_NaN();reject(validate);
    std::ifstream file(argv[2]);std::string magic;uint32_t count;
    if(!(file>>magic>>count) || magic!="S3D_DECODER_NORM_V1" || count!=8)throw std::runtime_error("invalid norm fixture");
    sam3d::neural_session session(argv[1],"CPU",0);
    auto tensor=[&](size_t n){std::vector<float> v(n);for(float &x:v)if(!(file>>x) || !std::isfinite(x))throw std::runtime_error("invalid norm fixture float");return v;};
    for(uint32_t i=0;i<count;++i){uint32_t b,n,d;
        if(!(file>>b>>n>>d) || b<1 || b>2 || n<1 || n>256 || d!=8)throw std::runtime_error("invalid norm fixture shape");
        auto input=tensor(b*n*d),weight=tensor(d),bias=tensor(d),expected=tensor(b*n*d);
        auto run=[&]{return sam3d::body_decoder_norm(session,b,n,d,input,weight,bias);};auto result=run();
        double maximum=0,error=0,norm=0;for(size_t j=0;j<result.size();++j){double delta=double(result[j])-expected[j];maximum=std::max(maximum,std::abs(delta));error=std::hypot(error,delta);norm=std::hypot(norm,double(expected[j]));}
        if(maximum>1e-4 || error/std::max(norm,1e-12)>2e-5)throw std::runtime_error("original norm_final divergence");
        auto saved=input;input.pop_back();reject(run);input=saved;input[0]=std::numeric_limits<float>::infinity();reject(run);input=saved;
        weight.pop_back();reject(run);
    }
    file>>std::ws;if(!file.eof())throw std::runtime_error("trailing norm fixture");
    std::cout<<"8 original per-layer final norms and flow contract rejection tests passed; not full decoder parity by themselves\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
