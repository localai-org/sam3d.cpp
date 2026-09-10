// Tiny GGML-written typed archive; not trained-model inference or parser fuzzing.
#include "tensor_archive.hpp"
#include "ggml.h"
#include "gguf.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected scratch directory");std::filesystem::path root=argv[1];std::filesystem::create_directories(root);
    auto reject=[](auto fn){bool bad=false;try{fn();}catch(const std::invalid_argument &){bad=true;}if(!bad)throw std::runtime_error("invalid MHR archive accepted");};
    const sam3d::tensor_shapes shapes={{"skin.joints",{3}},{"skin.weights",{3}}};
    std::unique_ptr<gguf_context,decltype(&gguf_free)> g(gguf_init_empty(),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({ggml_tensor_overhead()*8,nullptr,true}),ggml_free);if(!g || !c)throw std::bad_alloc();
    auto names=[](size_t count){std::string s;for(size_t i=0;i<count;++i){if(i)s+=',';s+='j'+std::to_string(i);}return s;};
    std::map<std::string,std::string> meta={
        {"general.architecture","sam3d.mhr.lod1"},{"sam3d.component","mhr.lod1"},{"sam3d.source.asset_sha256","352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"},
        {"sam3d.source.release","https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip"},{"sam3d.source.release_sha256","e4f4f205cd87c0fa106577ba1de4fc763e4eb197c924461d2ef7e6944e9d6b94"},
        {"sam3d.source.safetensors_sha256",std::string(64,'0')},{"sam3d.converted.precision","F32,I32"},{"sam3d.tensor_layout","torch_contiguous; ggml_dimensions_reversed; no_data_transpose"},
        {"sam3d.kinematics","local_F32;prefix_F64;global_F32;XYZ;XYZW;log2_scale"},{"sam3d.units","centimeters"},{"sam3d.prefix_sizes","65,56,62,83"},
        {"sam3d.joint_names",names(127)},{"sam3d.parameter_names",names(249)},{"sam3d.output","vertices_cm;skeleton_tx_ty_tz_qx_qy_qz_qw_scale;not_image_estimator"}};
    for(auto &[k,v]:meta)gguf_set_val_str(g.get(),k.c_str(),v.c_str());gguf_set_val_u32(g.get(),"general.alignment",32);gguf_set_val_u32(g.get(),"sam3d.schema_version",1);
    int32_t indices[]={0,1,126};float weights[]={.2f,.3f,.5f};
    auto i=ggml_new_tensor_1d(c.get(),GGML_TYPE_I32,3);i->data=indices;ggml_set_name(i,"skin.joints");gguf_add_tensor(g.get(),i);
    auto w=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,3);w->data=weights;ggml_set_name(w,"skin.weights");gguf_add_tensor(g.get(),w);
    auto path=root/"typed.gguf";auto write=[&]{if(!gguf_write_to_file(g.get(),path.string().c_str(),false))throw std::runtime_error("MHR fixture write failed");};write();
    {sam3d::tensor_archive a(path,"sam3d.mhr.lod1",shapes);
        if(a.read_i32("skin.joints",12)!=std::vector<int32_t>{0,1,126} || a.read("skin.weights",12)!=std::vector<float>{.2f,.3f,.5f})throw std::runtime_error("typed roundtrip failed");
        if(a.metadata("sam3d.joint_names")!=names(127))throw std::runtime_error("names mismatch");
        reject([&]{a.read("skin.joints",12);});reject([&]{a.read_i32("skin.weights",12);});reject([&]{a.read_i32("skin.joints",11);});reject([&]{a.metadata("missing");});}
    indices[2]=127;write();reject([&]{sam3d::tensor_archive a(path,"sam3d.mhr.lod1",shapes);a.read_i32("skin.joints",12);});indices[2]=126;
    for(auto key:{"sam3d.source.asset_sha256","sam3d.joint_names","sam3d.kinematics","sam3d.prefix_sizes"}){
        gguf_set_val_str(g.get(),key,"wrong");write();reject([&]{sam3d::tensor_archive a(path,"sam3d.mhr.lod1",shapes);});gguf_set_val_str(g.get(),key,meta.at(key).c_str());}
    write();std::filesystem::resize_file(path,std::filesystem::file_size(path)-1);reject([&]{sam3d::tensor_archive a(path,"sam3d.mhr.lod1",shapes);});
    {
        std::unique_ptr<gguf_context,decltype(&gguf_free)> sparse(gguf_init_empty(),gguf_free);
        for(auto &[k,v]:meta)gguf_set_val_str(sparse.get(),k.c_str(),v.c_str());
        gguf_set_val_u32(sparse.get(),"general.alignment",32);gguf_set_val_u32(sparse.get(),"sam3d.schema_version",1);
        constexpr size_t n=53136;
        std::vector<int32_t> ix(2*n);std::vector<float> values(n,.25f);
        for(size_t k=0;k<n;++k){ix[k]=k/750;ix[n+k]=k%750;}
        auto ti=ggml_new_tensor_2d(c.get(),GGML_TYPE_I32,n,2);ti->data=ix.data();ggml_set_name(ti,"pose.sparse.indices");gguf_add_tensor(sparse.get(),ti);
        auto tw=ggml_new_tensor_1d(c.get(),GGML_TYPE_F32,n);tw->data=values.data();ggml_set_name(tw,"pose.sparse.weight");gguf_add_tensor(sparse.get(),tw);
        const sam3d::tensor_shapes ss={{"pose.sparse.indices",{2,n}},{"pose.sparse.weight",{n}}};
        auto sp=root/"sparse.gguf";auto save=[&]{if(!gguf_write_to_file(sparse.get(),sp.string().c_str(),false))throw std::runtime_error("sparse fixture write failed");};save();
        sam3d::tensor_archive archive(sp,"sam3d.mhr.lod1",ss);
        auto dense=archive.pin_mhr_sparse_projection(),again=archive.pin_mhr_sparse_projection();
        if(dense.size()!=3000*750 || dense.data()!=again.data())throw std::runtime_error("sparse snapshot not reused");
        for(size_t k=0;k<dense.size();++k)if(dense[k]!=(k<n?.25f:0.f))throw std::runtime_error("sparse expansion/layout wrong");
        ix[0]=-1;save();reject([&]{sam3d::tensor_archive bad(sp,"sam3d.mhr.lod1",ss);bad.pin_mhr_sparse_projection();});ix[0]=0;
        values[0]=std::numeric_limits<float>::infinity();save();reject([&]{sam3d::tensor_archive bad(sp,"sam3d.mhr.lod1",ss);bad.pin_mhr_sparse_projection();});
        if(archive.pin_mhr_sparse_projection()[0]!=.25f)throw std::runtime_error("sparse snapshot changed with file");
    }
    std::cout<<"MHR GGUF: typed values, bounds, names, identity, precision and rejection tests passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
