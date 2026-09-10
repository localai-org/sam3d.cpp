#include "tensor_archive.hpp"
#include "gguf.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>

namespace sam3d {
namespace {
constexpr uint64_t max_header=4*1024*1024, max_file=64ULL*1024*1024*1024;
void require(bool b,const char *s) { if (!b) throw std::invalid_argument(s); }
uint64_t align32(uint64_t n) { require(n<=max_file,"archive size limit"); return (n+31)/32*32; }
struct reader {
    std::ifstream &file;
    uint64_t offset=0;
    std::vector<uint8_t> snapshot;
    void bytes(void *p,uint64_t n) {
        require(n<=max_header-offset,"GGUF metadata too large");
        require(bool(file.read(static_cast<char *>(p),n)),"truncated GGUF metadata"); offset+=n;
        const auto first=static_cast<const uint8_t *>(p);
        snapshot.insert(snapshot.end(),first,first+n);
    }
    template<class T> T scalar() {
        static_assert(std::endian::native==std::endian::little);
        T value; bytes(&value,sizeof(value)); return value;
    }
    std::string string(uint64_t limit=65536) {
        auto n=scalar<uint64_t>(); require(n>0 && n<=limit,"invalid GGUF string length");
        std::string s(n,'\0'); bytes(s.data(),n);
        // The current component contract uses printable ASCII identifiers/URLs.
        require(std::all_of(s.begin(),s.end(),[](unsigned char c){return c>=32 && c<=126;}),"invalid GGUF string bytes");
        return s;
    }
};
bool hash(const std::string &s,size_t size) {
    return s.size()==size && std::all_of(s.begin(),s.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
void validate_names(const std::string &value,size_t count) {
    std::set<std::string> names;size_t start=0;
    for(;;){auto end=value.find(',',start);auto name=value.substr(start,end==std::string::npos?end:end-start);
        require(!name.empty() && name.size()<=128 && names.insert(name).second,"invalid/duplicate geometry name");
        require(std::all_of(name.begin(),name.end(),[](char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='.'||c==':'||c=='-';}),"invalid geometry name bytes");
        if(end==std::string::npos)break;start=end+1;}
    require(names.size()==count,"geometry name count mismatch");
}
}

tensor_shapes mhr_lod1_shapes(){return {
    {"identity.vectors",{45,18439,3}},{"identity.base",{18439,3}},{"expression.vectors",{72,18439,3}},
    {"pose.sparse.indices",{2,53136}},{"pose.sparse.weight",{53136}},{"pose.dense.weight",{55317,3000}},
    {"skeleton.offsets",{127,3}},{"skeleton.prerotations",{127,4}},{"skeleton.prefix",{2,266}},{"skeleton.parents",{127}},
    {"skin.inverse_bind",{127,8}},{"skin.joints",{51337}},{"skin.weights",{51337}},{"skin.vertices",{51337}},
    {"mesh.faces",{36874,3}},{"mesh.texcoords",{19455,2}},{"mesh.texcoord_faces",{36874,3}},{"parameter.matrix",{889,249}}};}
bool mhr_integer_tensor(const std::string &name){
    static const std::set<std::string> names={"pose.sparse.indices","skeleton.prefix","skeleton.parents","skin.joints","skin.vertices","mesh.faces","mesh.texcoord_faces"};return names.contains(name);
}

tensor_shapes body_dino_shapes() {
    const uint64_t d=1280, h=5120;
    tensor_shapes result={{"cls_token",{1,1,d}},{"storage_tokens",{1,4,d}},
        {"mask_token",{1,d}},{"patch_embed.proj.weight",{d,3,16,16}},
        {"patch_embed.proj.bias",{d}},{"rope_embed.periods",{16}},
        {"norm.weight",{d}},{"norm.bias",{d}}};
    tensor_shapes block={{"norm1.weight",{d}},{"norm1.bias",{d}},
        {"attn.qkv.weight",{3*d,d}},{"attn.qkv.bias",{3*d}},{"attn.qkv.bias_mask",{3*d}},
        {"attn.proj.weight",{d,d}},{"attn.proj.bias",{d}},{"ls1.gamma",{d}},
        {"norm2.weight",{d}},{"norm2.bias",{d}},
        {"mlp.w1.weight",{h,d}},{"mlp.w1.bias",{h}},{"mlp.w2.weight",{h,d}},
        {"mlp.w2.bias",{h}},{"mlp.w3.weight",{d,h}},{"mlp.w3.bias",{d}},{"ls2.gamma",{d}}};
    for (int i=0;i<32;++i) for (const auto &[name,shape]:block)
        result.emplace("blocks."+std::to_string(i)+"."+name,shape);
    return result;
}

tensor_archive::tensor_archive(const std::filesystem::path &path,const std::string &architecture,
                               const tensor_shapes &expected) {
    // Reject ordinary non-files (including named pipes) before opening them.
    // Model paths must live in trusted directories and remain immutable.
    require(std::filesystem::is_regular_file(path),"GGUF component must be a regular file");
    file_.open(path,std::ios::binary);
    require(bool(file_),"cannot open GGUF component");
    file_.seekg(0,std::ios::end); const auto end=file_.tellg();
    require(end>=24 && uint64_t(end)<=max_file,"invalid GGUF file size");
    const uint64_t file_size=end; file_.seekg(0);
    reader r{file_};
    require(r.scalar<uint32_t>()==0x46554747,"not a GGUF file");
    require(r.scalar<uint32_t>()==3,"unsupported GGUF version");
    auto count=r.scalar<uint64_t>(), n_meta=r.scalar<uint64_t>();
    require(count>0 && count<=4096 && count==expected.size(),"GGUF tensor count mismatch");
    const bool mhr=architecture=="sam3d.mhr.lod1";
    const bool branch=architecture=="sam3d.body.pose_branch";
    require(mhr || branch || architecture=="sam3d.body.dinov3.vith16plus","unsupported GGUF architecture");
    require(n_meta==(mhr?16:branch?17:19),"GGUF metadata count mismatch");
    std::map<std::string,std::string> strings;
    std::map<std::string,uint32_t> integers;
    std::set<std::string> keys;
    for (uint64_t i=0;i<n_meta;++i) {
        auto key=r.string(128); require(keys.insert(key).second,"duplicate GGUF metadata");
        auto type=r.scalar<uint32_t>();
        if (type==GGUF_TYPE_STRING) strings.emplace(key,r.string());
        else if (type==GGUF_TYPE_UINT32) integers.emplace(key,r.scalar<uint32_t>());
        else throw std::invalid_argument("unsupported GGUF metadata type");
    }
    if(branch){
        const std::map<std::string,std::string> fixed={
            {"general.architecture",architecture},{"sam3d.component","body.pose_branch"},
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
        require(integers.size()==2 && integers.contains("general.alignment") && integers.at("general.alignment")==32 && integers.contains("sam3d.schema_version") && integers.at("sam3d.schema_version")==1,"unsupported Body branch alignment/schema");
        for(auto &[key,value]:fixed)require(strings.contains(key) && strings.at(key)==value,"incompatible Body branch metadata");
    }else if(!mhr){
    const std::map<std::string,std::string> fixed={
        {"general.architecture",architecture},{"sam3d.component","body.dinov3_backbone"},
        {"sam3d.source.body_repository","https://github.com/facebookresearch/sam-3d-body"},
        {"sam3d.source.body_revision","b5c765a0d89d789985e186d396315e7590887b94"},
        {"sam3d.source.dinov3_repository","https://github.com/facebookresearch/dinov3"},
        {"sam3d.source.dinov3_revision","6876159a11b4df116f30f667f8c9888617df0751"},
        {"sam3d.converted.precision","F32"},
        {"sam3d.tensor_layout","torch_contiguous; ggml_dimensions_reversed; no_data_transpose"},
        {"sam3d.preprocessing","body_rgb_crop_v1;512x512;ImageNet_mean_std"},
        {"sam3d.required_companions","none_for_backbone_features;not_a_complete_body_estimator"},
        {"sam3d.norm_epsilon","1e-5"},{"sam3d.rope_mode","axial_separate_eval_no_rescale"},
        {"sam3d.output","normalized_patch_features;decoder_and_MHR_not_included"}};
    require(integers.size()==2 && integers.contains("general.alignment") &&
            integers.at("general.alignment")==32 && integers.contains("sam3d.schema_version") &&
            integers.at("sam3d.schema_version")==1,"unsupported GGUF alignment/schema");
    for (const auto &[key,value]:fixed)
        require(strings.contains(key) && strings.at(key)==value,"incompatible GGUF component metadata");
    for (auto key:{"sam3d.source.checkpoint_sha256","sam3d.source.config_sha256","sam3d.source.safetensors_sha256"})
        require(strings.contains(key) && hash(strings.at(key),64),"invalid GGUF source hash");
    require(strings.at("sam3d.source.checkpoint_sha256")=="b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf" &&
            strings.at("sam3d.source.config_sha256")=="1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d",
            "unsupported GGUF checkpoint/config identity");
    require(strings.contains("sam3d.source.precision"),"missing source precision");
    const std::set<std::string> precisions={"F32","F16","BF16","F16,F32","BF16,F32","BF16,F16","BF16,F16,F32"};
    require(precisions.contains(strings.at("sam3d.source.precision")),"unsupported source precision");
    }else{
        const std::map<std::string,std::string> fixed={
            {"general.architecture","sam3d.mhr.lod1"},{"sam3d.component","mhr.lod1"},
            {"sam3d.source.asset_sha256","352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"},
            {"sam3d.source.release","https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip"},
            {"sam3d.source.release_sha256","e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94"},
            {"sam3d.converted.precision","F32,I32"},{"sam3d.tensor_layout","torch_contiguous; ggml_dimensions_reversed; no_data_transpose"},
            {"sam3d.kinematics","local_F32;prefix_F64;global_F32;XYZ;XYZW;log2_scale"},
            {"sam3d.units","centimeters"},{"sam3d.prefix_sizes","65,56,62,83"},
            {"sam3d.output","vertices_cm;skeleton_tx_ty_tz_qx_qy_qz_qw_scale;not_image_estimator"}};
        require(integers.size()==2 && integers.contains("general.alignment") && integers.at("general.alignment")==32 && integers.contains("sam3d.schema_version") && integers.at("sam3d.schema_version")==1,"unsupported MHR alignment/schema");
        for(auto &[key,value]:fixed)require(strings.contains(key) && strings.at(key)==value,"incompatible MHR metadata");
        require(strings.contains("sam3d.source.safetensors_sha256") && hash(strings.at("sam3d.source.safetensors_sha256"),64),"invalid MHR safe source hash");
        require(strings.contains("sam3d.joint_names") && strings.contains("sam3d.parameter_names"),"missing MHR geometry names");
        validate_names(strings.at("sam3d.joint_names"),127);validate_names(strings.at("sam3d.parameter_names"),249);
    }
    metadata_=strings;
    uint64_t next=0;
    for (uint64_t i=0;i<count;++i) {
        auto name=r.string(63); auto it=expected.find(name);
        require(it!=expected.end() && !tensors_.contains(name),"unexpected/duplicate GGUF tensor");
        auto rank=r.scalar<uint32_t>();
        require(rank>=1 && rank<=4 && rank==it->second.size(),"GGUF tensor rank mismatch");
        uint64_t elements=1;
        for (uint32_t j=0;j<rank;++j) {
            auto dimension=r.scalar<uint64_t>();
            require(dimension>0 && dimension==it->second[rank-1-j] &&
                    dimension<=(1ULL<<31)/elements,"GGUF tensor shape/size mismatch");
            elements*=dimension;
        }
        const uint32_t type=r.scalar<uint32_t>();
        require(type==uint32_t((mhr && mhr_integer_tensor(name)) || (branch && body_branch_integer_tensor(name))?GGML_TYPE_I32:GGML_TYPE_F32),"GGUF tensor dtype mismatch");
        auto offset=r.scalar<uint64_t>(); require(offset==next,"noncanonical GGUF tensor offset");
        tensors_.emplace(name,tensor_info{offset,elements*4,type}); next+=align32(elements*4);
        require(next<=max_file,"GGUF data size limit");
    }
    const auto start=align32(r.offset);
    require(start<=file_size && next==file_size-start,"truncated or trailing GGUF data");
    while (r.offset<start) require(r.scalar<uint8_t>()==0,"nonzero GGUF header padding");
    // Now the GGML parser sees only prebounded counts/types/names/ranks/offsets.
    // Parse the same bounded snapshot, not a second mutable-file metadata read.
    std::unique_ptr<gguf_context,decltype(&gguf_free)> gguf(gguf_init_from_buffer(
        r.snapshot.data(),r.snapshot.size(),{true,nullptr}),gguf_free);
    require(bool(gguf),"GGML rejected GGUF metadata");
    require(gguf_get_data_offset(gguf.get())==start && gguf_get_n_tensors(gguf.get())==int64_t(count),
            "GGML GGUF interpretation mismatch");
    for (int64_t i=0;i<int64_t(count);++i) {
        auto &t=tensors_.at(gguf_get_tensor_name(gguf.get(),i));
        require(gguf_get_tensor_size(gguf.get(),i)==t.bytes && gguf_get_tensor_offset(gguf.get(),i)==t.offset,
                "GGML tensor layout interpretation mismatch");
        t.offset+=start;
    }
}

std::vector<float> tensor_archive::read(const std::string &name,uint64_t budget) {
    const auto it=tensors_.find(name); require(it!=tensors_.end(),"unknown GGUF tensor");
    require(it->second.type==GGML_TYPE_F32,"GGUF tensor is not F32");
    require(it->second.bytes<=budget,"tensor read exceeds caller memory budget");
    std::vector<float> values(it->second.bytes/4);
    { std::lock_guard lock(mutex_); file_.clear(); file_.seekg(it->second.offset);
      require(bool(file_.read(reinterpret_cast<char *>(values.data()),it->second.bytes)),"GGUF data read failed"); }
    require(std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);}),"non-finite GGUF weights");
    if (name.ends_with("bias_mask")) {
        require(values.size()%3==0,"bad key bias mask size");
        const size_t d=values.size()/3;
        const bool all_zero=std::all_of(values.begin(),values.end(),[](float v){return v==0.f;});
        for (size_t i=0;i<values.size();++i) require(all_zero || values[i]==(i>=d&&i<2*d?0.f:1.f),"invalid key bias mask");
    }
    if (name=="rope_embed.periods")
        require(std::all_of(values.begin(),values.end(),[](float x){return x>0;}),"invalid RoPE periods");
    return values;
}

validated_weights tensor_archive::load(const std::string &name,uint64_t budget) {
    return validated_weights(read(name,budget),validated_weights::checked_tag{});
}
validated_weights tensor_archive::pin(const std::string &name,uint64_t budget) {
    std::lock_guard lock(pinned_mutex_);
    auto info=tensors_.find(name);
    require(info!=tensors_.end() && info->second.type==GGML_TYPE_F32,"unknown/non-F32 pinned tensor");
    require(info->second.bytes<=budget,"pinned tensor exceeds caller memory budget");
    if(auto it=pinned_.find(name);it!=pinned_.end())return it->second;
    require(info->second.bytes<=pinned_limit_-pinned_bytes_,"pinned weight cache exceeds memory budget");
    // read() checks both finiteness and tensor-specific semantic constraints.
    auto value=load(name,budget);
    pinned_.emplace(name,value);pinned_bytes_+=info->second.bytes;
    return value;
}

std::vector<int32_t> tensor_archive::read_i32(const std::string &name,uint64_t budget){
    auto it=tensors_.find(name);require(it!=tensors_.end() && it->second.type==GGML_TYPE_I32,"unknown/non-I32 tensor");require(it->second.bytes<=budget,"integer tensor exceeds memory budget");
    std::vector<int32_t> values(it->second.bytes/4);
    {std::lock_guard lock(mutex_);file_.clear();file_.seekg(it->second.offset);require(bool(file_.read(reinterpret_cast<char *>(values.data()),it->second.bytes)),"GGUF integer read failed");}
    const int32_t limit=name=="mesh.texcoord_faces"?19455:(name=="skin.vertices" || name=="mesh.faces" || name=="head_pose.faces"?18439:127);
    if(name=="pose.sparse.indices"){
        const size_t n=values.size()/2;int64_t previous=-1;
        for(size_t i=0;i<n;++i){require(values[i]>=0 && values[i]<3000 && values[n+i]>=0 && values[n+i]<750,"sparse index out of range");int64_t order=int64_t(values[i])*750+values[n+i];require(order>previous,"sparse indices not unique/sorted");previous=order;}
    }else if(name=="head_pose.hand_joint_idxs_left" || name=="head_pose.hand_joint_idxs_right"){
        const int32_t first=name.ends_with("left")?95:68;std::set<int32_t> seen;
        for(int32_t v:values)require(v>=first && v<first+27 && seen.insert(v).second,"invalid Body hand index partition");
        require(seen.size()==27,"incomplete Body hand index partition");
    }else if(name=="skeleton.parents"){
        require(!values.empty() && values[0]==-1,"invalid skeleton root");for(size_t i=1;i<values.size();++i)require(values[i]>=0 && uint64_t(values[i])<i,"invalid skeleton parent ordering");
    }else for(int32_t v:values)require(v>=0 && v<limit,"geometry index out of range");
    return values;
}
validated_weights tensor_archive::pin_mhr_sparse_projection(){
    std::lock_guard lock(pinned_mutex_);
    const std::string key="@native.mhr.pose_sparse_dense";
    if(auto it=pinned_.find(key);it!=pinned_.end())return it->second;
    constexpr uint64_t count=3000*750,bytes=count*4;
    require(bytes<=pinned_limit_-pinned_bytes_,"pinned sparse projection exceeds memory budget");
    auto indices=read_i32("pose.sparse.indices",2*53136*4);
    auto weights=read("pose.sparse.weight",53136*4);
    require(indices.size()==2*53136 && weights.size()==53136,"MHR sparse projection shape mismatch");
    // read_i32 already enforces bounded, sorted, unique COO indices.
    std::vector<float> dense(count,0.f);
    for(size_t i=0;i<weights.size();++i)dense[size_t(indices[i])*750+indices[weights.size()+i]]=weights[i];
    auto snapshot=validated_weights::checked(std::move(dense));
    pinned_.emplace(key,snapshot);pinned_bytes_+=bytes;return snapshot;
}
const std::string &tensor_archive::metadata(const std::string &key)const{
    auto it=metadata_.find(key);require(it!=metadata_.end(),"unknown archive metadata");return it->second;
}
}
