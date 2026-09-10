// Deterministic tiny GGUF contracts, not learned inference or GGUF fuzzing.
#include "tensor_archive.hpp"
#include "ggml.h"
#include "gguf.h"
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected scratch directory");
    const std::filesystem::path root=argv[1];std::filesystem::create_directories(root);
    auto reject=[](auto fn){bool bad=false;try{fn();}catch(const std::invalid_argument &){bad=true;}if(!bad)throw std::runtime_error("invalid Body branch archive accepted");};
    // GGML's writer elides trailing singleton dimensions in its descriptors.
    const sam3d::tensor_shapes shapes={{"head_pose.faces",{3}},{"head_pose.hand_joint_idxs_left",{27}},{"head_pose.hand_joint_idxs_right",{27}},{"init_pose.weight",{2}}};
    std::unique_ptr<gguf_context,decltype(&gguf_free)> g(gguf_init_empty(),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*8,nullptr,true}),ggml_free);
    if(!g || !c)throw std::bad_alloc();
    const std::map<std::string,std::string> meta={
        {"general.architecture","sam3d.body.pose_branch"},{"sam3d.component","body.pose_branch"},
        {"sam3d.source.body_repository","https://github.com/facebookresearch/sam-3d-body"},
        {"sam3d.source.body_revision","b5c765a0d89d789985e186d396315e7590887b94"},
        {"sam3d.source.checkpoint_sha256","b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"},
        {"sam3d.source.config_sha256","1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d"},
        {"sam3d.source.safetensors_sha256","4c6b3f63ce8a050f6587cf833a036bad3f68377d86cfe69d591502c1373ba0a3"},
        {"sam3d.converted.precision","F32,I32"},{"sam3d.tensor_layout","torch_contiguous; ggml_dimensions_reversed; no_data_transpose"},
        {"sam3d.decoder","depth=6;dim=1024;heads=8;head_dim=64;mlp=1024;repeat_pe=1;twoway=0;norm_eps=1e-6"},
        {"sam3d.condition","cliff_intrinsics_center;keypoints70;keypoints3d70;hand_tokens;no_mask_embedding"},
        {"sam3d.required_backbone","sam3d.body.dinov3.vith16plus"},
        {"sam3d.required_mhr_sha256","352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"},
        {"sam3d.output","body_vertices_m;keypoints70;camera;hand_boxes;not_hand_refinement"},{"sam3d.source.precision","F32,I64"}};
    for(auto &[k,v]:meta)gguf_set_val_str(g.get(),k.c_str(),v.c_str());
    gguf_set_val_u32(g.get(),"general.alignment",32);gguf_set_val_u32(g.get(),"sam3d.schema_version",1);
    int32_t faces[]={0,1,18438},left[27],right[27];float weights[]={-0.f,2.f};
    std::iota(left,left+27,95);std::iota(right,right+27,68);
    for(auto &[name,dims]:shapes){
        int64_t ne[2]={int64_t(dims.back()),1};if(dims.size()==2)ne[1]=dims[0];
        auto t=ggml_new_tensor(c.get(),sam3d::body_branch_integer_tensor(name)?GGML_TYPE_I32:GGML_TYPE_F32,dims.size(),ne);
        t->data=name.ends_with("left")?static_cast<void*>(left):name.ends_with("right")?static_cast<void*>(right):name.ends_with("faces")?static_cast<void*>(faces):static_cast<void*>(weights);
        ggml_set_name(t,name.c_str());gguf_add_tensor(g.get(),t);
    }
    const auto path=root/"branch.gguf";
    auto write=[&]{if(!gguf_write_to_file(g.get(),path.string().c_str(),false))throw std::runtime_error("fixture write failed");};
    auto load=[&]{return sam3d::tensor_archive(path,"sam3d.body.pose_branch",shapes);};write();
    {auto a=load();if(a.read_i32("head_pose.faces",12)!=std::vector<int32_t>{0,1,18438} || a.read_i32("head_pose.hand_joint_idxs_left",108).size()!=27 || a.read_i32("head_pose.hand_joint_idxs_right",108).front()!=68 || a.read("init_pose.weight",8)[1]!=2)throw std::runtime_error("typed roundtrip failed");
        reject([&]{a.read("head_pose.faces",12);});reject([&]{a.read_i32("init_pose.weight",8);});reject([&]{a.read_i32("head_pose.faces",11);});
    }
    for(auto &[key,value]:meta){gguf_set_val_str(g.get(),key.c_str(),"wrong");write();reject([&]{load();});gguf_set_val_str(g.get(),key.c_str(),value.c_str());}
    for(auto key:{"general.alignment","sam3d.schema_version"}){gguf_set_val_u32(g.get(),key,std::string(key)=="general.alignment"?64:7);write();reject([&]{load();});gguf_set_val_u32(g.get(),key,std::string(key)=="general.alignment"?32:1);}
    for(int32_t bad:{-1,18439}){faces[2]=bad;write();reject([&]{load().read_i32("head_pose.faces",12);});}faces[2]=18438;
    for(int32_t bad:{94,96,122}){left[0]=bad;write();reject([&]{load().read_i32("head_pose.hand_joint_idxs_left",108);});}left[0]=95;
    right[0]=95;write();reject([&]{load().read_i32("head_pose.hand_joint_idxs_right",108);});right[0]=68;
    weights[0]=std::numeric_limits<float>::quiet_NaN();write();reject([&]{load().read("init_pose.weight",8);});weights[0]=0;
    write();reject([&]{sam3d::tensor_archive a(path,"sam3d.body.pose_branch",sam3d::body_branch_shapes());});
    std::filesystem::resize_file(path,std::filesystem::file_size(path)-1);reject([&]{load();});
    std::cout<<"Body branch typed GGUF roundtrip and metadata, partition, bounds, finite, incomplete and truncation rejection passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
