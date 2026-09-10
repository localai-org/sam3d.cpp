#include "tensor_archive.hpp"
#include <set>
namespace sam3d {
// Original pinned pose-branch configuration. Keep axes in PyTorch order;
// cross-check this contract against both safe state and runtime graph sizes.
tensor_shapes body_branch_shapes(){
    constexpr uint64_t d=1024,c=1280; tensor_shapes s;
    auto linear=[&](const std::string &n,uint64_t in,uint64_t out){s[n+".weight"]={out,in};s[n+".bias"]={out};};
    auto norm=[&](const std::string &n,uint64_t width){s[n+".weight"]={width};s[n+".bias"]={width};};
    auto ffn=[&](const std::string &n,uint64_t in,uint64_t out){linear(n+".layers.0.0",in,d);linear(n+".layers.1",d,out);};
    s["init_pose.weight"]={1,519};s["init_camera.weight"]={1,3};
    linear("init_to_token_mhr",525,d);linear("prev_to_token_mhr",522,d);linear("prompt_to_token",c,d);
    for(auto n:{"keypoint_embedding","keypoint3d_embedding"})s[std::string(n)+".weight"]={70,d};
    s["hand_box_embedding.weight"]={2,d};s["ray_cond_emb.conv.weight"]={c,c+99,1,1};norm("ray_cond_emb.norm",c);
    s["prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"]={2,c/2};
    for(auto n:{"invalid_point_embed","not_a_point_embed","no_mask_embed"})s["prompt_encoder."+std::string(n)+".weight"]={1,c};
    for(unsigned i=0;i<70;++i)s["prompt_encoder.point_embeddings."+std::to_string(i)+".weight"]={1,c};
    for(unsigned i=0;i<6;++i){const auto p="decoder.layers."+std::to_string(i)+".";
        for(auto n:{"ln_pe_1","ln1","ln2_1","ln3"})norm(p+n,d);
        for(auto n:{"ln_pe_2","ln2_2"})norm(p+n,c);
        for(auto n:{"self_attn","cross_attn"}){const uint64_t kv=std::string(n)=="self_attn"?d:c;const auto a=p+n;
            linear(a+".q_proj",d,512);linear(a+".k_proj",kv,512);linear(a+".v_proj",kv,512);linear(a+".proj",512,d);}
        ffn(p+"ffn",d,d);
    }
    norm("decoder.norm_final",d);ffn("head_pose.proj",d,519);ffn("head_camera.proj",d,3);
    s["head_pose.scale_mean"]={68};s["head_pose.scale_comps"]={28,68};s["head_pose.hand_pose_mean"]={54};s["head_pose.hand_pose_comps"]={54,54};
    s["head_pose.keypoint_mapping"]={308,18566};s["head_pose.hand_joint_idxs_left"]={27};s["head_pose.hand_joint_idxs_right"]={27};s["head_pose.faces"]={36874,3};
    ffn("keypoint_posemb_linear",2,d);ffn("keypoint3d_posemb_linear",3,d);linear("keypoint_feat_linear",c,d);
    for(unsigned i=0;i<3;++i)linear("bbox_embed.layers."+std::to_string(i),d,i==2?4:d);
    linear("hand_cls_embed",d,2);return s;
}
bool body_branch_integer_tensor(const std::string &name){
    return name=="head_pose.faces" || name=="head_pose.hand_joint_idxs_left" || name=="head_pose.hand_joint_idxs_right";
}
}
