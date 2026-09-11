// Composition of Meta SAM 3D Objects preprocessing, PointPatchEmbed and
// EmbedderFuser contracts; see NOTICE and SAM License.
#include "objects_point_condition.hpp"
#include <stdexcept>
namespace sam3d {
void validate_point_condition_shape(const point_condition_shape &s){
    validate_objects_preprocess_options(s.preprocessing);validate_pointpatch_shape(s.encoder);validate_fuser_shape(s.fusion);
    auto require=[](bool b){if(!b)throw std::invalid_argument("incompatible point conditioning configuration");};
    require(s.encoder.batch==1 && s.encoder.height==s.preprocessing.point_side && s.encoder.width==s.preprocessing.point_side);
    require(s.fusion.batch==1 && s.fusion.embed_dims==std::vector<uint32_t>{s.encoder.dim} && s.fields.size()==s.fusion.inputs.size());
    const uint32_t grid=s.encoder.side/s.encoder.patch;
    for(size_t i=0;i<s.fields.size();++i)require(uint32_t(s.fields[i])<=2 && s.fusion.inputs[i].embedder==0 && s.fusion.inputs[i].tokens==grid*grid);
}
point_condition_result objects_point_condition(neural_session &session,const point_condition_shape &s,
    std::span<const uint8_t> rgba,uint32_t width,uint32_t height,uint64_t stride,
    std::span<const float> xyz,uint32_t ph,uint32_t pw,const named_floats &ep,const named_floats &fp,
    const pointpatch_observer &observe,uint32_t chunk){
    validate_point_condition_shape(s);
    if(chunk<1 || chunk>64)throw std::invalid_argument("invalid point conditioning chunk size");
    point_condition_result result;
    result.prepared=objects_preprocess_pointmap(rgba,width,height,stride,xyz,ph,pw,s.preprocessing);
    if(observe)for(auto &[key,v]:result.prepared)observe("00.prepared."+key,v,0,v.size());
    const char *fields[]={"pointmap","rgb_pointmap","rgb_pointmap_unnorm"};
    for(size_t i=0;i<s.fields.size();++i){
        auto prefix="10.encoder."+std::to_string(i)+".";
        result.embeddings.push_back(objects_pointpatch(session,s.encoder,result.prepared.at(fields[uint32_t(s.fields[i])]),{},ep,
            observe?pointpatch_observer([&](const std::string &key,std::span<const float> v,uint64_t offset,uint64_t total){observe(prefix+key,v,offset,total);}):pointpatch_observer(),chunk));
    }
    auto fused=objects_fuse(session,s.fusion,result.embeddings,fp,bool(observe));
    if(observe)for(auto &[key,v]:fused)observe("20.fusion."+key,v,0,v.size());
    result.conditioning=std::move(fused.at("90.output"));return result;
}
}
