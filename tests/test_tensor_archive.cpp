// Format/loader tests use tiny synthetic tensors, not a valid trained model.
#include "tensor_archive.hpp"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
const sam3d::tensor_shapes shapes={{"linear.bias",{2}},{"linear.weight",{2,3}}};
void verify(const std::filesystem::path &path) {
    sam3d::tensor_archive archive(path,"sam3d.body.dinov3.vith16plus",shapes);
    if (archive.read("linear.bias",8)!=std::vector<float>{.25f,-.5f} ||
        archive.read("linear.weight",24)!=std::vector<float>{0,1,2,3,4,5})
        throw std::runtime_error("GGUF tensor layout/value mismatch");
    bool rejected=false;
    try { archive.read("linear.weight",23); } catch (const std::invalid_argument &) { rejected=true; }
    if (!rejected) throw std::runtime_error("tensor budget not enforced");
    rejected=false;
    try { archive.read("unknown",1024); } catch (const std::invalid_argument &) { rejected=true; }
    if (!rejected) throw std::runtime_error("unknown tensor not rejected");
    auto first=archive.pin("linear.weight",24),second=archive.pin("linear.weight",24);
    if(first.values().data()!=second.values().data() || first.values()[5]!=5)
        throw std::runtime_error("validated snapshot not reused");
    rejected=false;
    try { archive.pin("linear.weight",23); } catch (const std::invalid_argument &) { rejected=true; }
    if(!rejected)throw std::runtime_error("cached tensor bypassed caller budget");
}
std::vector<uint8_t> read_file(const std::filesystem::path &p) {
    std::ifstream f(p,std::ios::binary); return {std::istreambuf_iterator<char>(f),{}};
}
void write_file(const std::filesystem::path &p,const std::vector<uint8_t> &data) {
    std::ofstream f(p,std::ios::binary);
    if (!f.write(reinterpret_cast<const char *>(data.data()),data.size())) throw std::runtime_error("test file write failed");
}
}
int main(int argc,char **argv) {
    try {
        if (argc!=2 && argc!=3) throw std::invalid_argument("expected scratch directory and optional converter fixture");
        const std::filesystem::path root=argv[1]; std::filesystem::create_directories(root);
        if (argc==3) { verify(argv[2]); std::cout<<"Python converter/GGML reader round trip passed\n"; return 0; }
        bool directory_rejected=false;
        try { verify(root); } catch (const std::invalid_argument &) { directory_rejected=true; }
        if (!directory_rejected) throw std::runtime_error("directory accepted as archive");
        std::unique_ptr<gguf_context,decltype(&gguf_free)> g(gguf_init_empty(),gguf_free);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*16,nullptr,true}),ggml_free);
        if (!g || !c) throw std::bad_alloc();
        const std::map<std::string,std::string> strings={
            {"general.architecture","sam3d.body.dinov3.vith16plus"},
            {"sam3d.component","body.dinov3_backbone"},
            {"sam3d.source.body_repository","https://github.com/facebookresearch/sam-3d-body"},
            {"sam3d.source.body_revision","b5c765a0d89d789985e186d396315e7590887b94"},
            {"sam3d.source.dinov3_repository","https://github.com/facebookresearch/dinov3"},
            {"sam3d.source.dinov3_revision","6876159a11b4df116f30f667f8c9888617df0751"},
            {"sam3d.source.checkpoint_sha256","b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"},
            {"sam3d.source.config_sha256","1012fc3f39cb5e90e3f8fbadf7bded31604bfafdce0321d17a7c1a2d3f08b88d"},
            {"sam3d.source.safetensors_sha256",std::string(64,'3')},
            {"sam3d.source.precision","F32"},{"sam3d.converted.precision","F32"},
            {"sam3d.tensor_layout","torch_contiguous; ggml_dimensions_reversed; no_data_transpose"},
            {"sam3d.preprocessing","body_rgb_crop_v1;512x512;ImageNet_mean_std"},
            {"sam3d.required_companions","none_for_backbone_features;not_a_complete_body_estimator"},
            {"sam3d.norm_epsilon","1e-5"},{"sam3d.rope_mode","axial_separate_eval_no_rescale"},
            {"sam3d.output","normalized_patch_features;decoder_and_MHR_not_included"}};
        for (auto &[key,value]:strings) gguf_set_val_str(g.get(),key.c_str(),value.c_str());
        gguf_set_val_u32(g.get(),"general.alignment",32);
        gguf_set_val_u32(g.get(),"sam3d.schema_version",1);
        float bias[]={.25,-.5}, weight[]={0,1,2,3,4,5};
        auto b=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,2); b->data=bias;
        ggml_set_name(b,"linear.bias"); gguf_add_tensor(g.get(),b);
        auto w=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,3,2); w->data=weight;
        ggml_set_name(w,"linear.weight"); gguf_add_tensor(g.get(),w);
        const auto valid=root/"ggml-written.gguf";
        if (!gguf_write_to_file(g.get(),valid.string().c_str(),false)) throw std::runtime_error("GGML writer failed");
        verify(valid);
        const auto original=read_file(valid);
        auto detached=[&]{sam3d::tensor_archive archive(valid,"sam3d.body.dinov3.vith16plus",shapes);return archive.pin("linear.bias",8);}();
        if(detached.values()[0]!=.25f || detached.values()[1]!=-.5f)
            throw std::runtime_error("snapshot did not survive archive destruction");
        {
            // Snapshot survives the archive and does not change when the file
            // changes. A fresh read still validates those new bytes.
            sam3d::tensor_archive archive(valid,"sam3d.body.dinov3.vith16plus",shapes);
            auto pinned=archive.pin("linear.bias",8);
            auto corrupt=original;const uint32_t infinity=0x7f800000;
            std::memcpy(corrupt.data()+gguf_get_meta_size(g.get()),&infinity,4);write_file(valid,corrupt);
            if(archive.pin("linear.bias",8).values()[0]!=.25f)throw std::runtime_error("pinned weights changed with file");
            bool rejected=false;try {archive.read("linear.bias",8);}catch(const std::invalid_argument &){rejected=true;}
            if(!rejected)throw std::runtime_error("fresh invalid read bypassed validation");
            rejected=false;try {sam3d::tensor_archive fresh(valid,"sam3d.body.dinov3.vith16plus",shapes);fresh.pin("linear.bias",8);}catch(const std::invalid_argument &){rejected=true;}
            if(!rejected)throw std::runtime_error("invalid first pin accepted");
            write_file(valid,original);
        }
        auto reject=[&](const std::vector<uint8_t> &bytes) {
            auto path=root/"invalid.gguf"; write_file(path,bytes); bool rejected=false;
            try { verify(path); } catch (const std::invalid_argument &) { rejected=true; }
            if (!rejected) throw std::runtime_error("malformed GGUF accepted");
        };
        auto changed=original; changed.pop_back(); reject(changed);
        changed=original; changed.push_back(0); reject(changed);
        changed=original; changed[4]=2; reject(changed); // unsupported version
        changed=original; changed[8]=3; reject(changed); // tensor count
        changed=original; uint64_t enormous=UINT64_MAX;
        std::memcpy(changed.data()+24,&enormous,8); reject(changed); // huge key
        const std::string name="linear.weight";
        auto it=std::search(original.begin(),original.end(),name.begin(),name.end());
        if (it==original.end()) throw std::runtime_error("missing GGML tensor header");
        size_t rank_offset=size_t(it-original.begin())+name.size();
        changed=original; changed[rank_offset+4]=2; reject(changed); // wrong input dimension
        changed=original; changed[rank_offset+4+16]=1; reject(changed); // F16 type
        changed=original; changed[rank_offset+4+16+4]=0; reject(changed); // overlapping offset
        const auto data_start=gguf_get_meta_size(g.get());
        changed=original; const uint32_t nan=0x7fc00000;
        std::memcpy(changed.data()+data_start,&nan,4); reject(changed);
        std::cout<<"GGML archive: layout, metadata, bounds, finite values and budget tests passed\n";
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
