#include "body_result.hpp"
#include "error.hpp"
#include <cmath>
#include <numeric>
struct s3d_body_result {
    struct tensor {std::string name;std::vector<uint64_t> shape;std::vector<float> floats;std::vector<int32_t> integers;};
    std::vector<tensor> tensors;
};
namespace sam3d {
const std::vector<body_output_field> &body_output_fields(){
    static const std::vector<body_output_field> fields={
        {"vertices","layer.5.pose.map.90.vertices",{1,18439,3}},
        {"joints","layer.5.pose.map.91.joints",{1,127,3}},
        {"joint_rotations","layer.5.pose.map.03.joint_rotations",{1,127,3,3}},
        {"keypoints","layer.5.pose.map.92.keypoints",{1,70,3}},
        {"keypoints_pixels","layer.5.camera.20.pixels",{1,70,2}},
        {"vertices_pixels","layer.5.03.vertex_pixels",{1,18439,2}},
        {"camera_translation","layer.5.camera.15.translation",{1,3}},
        {"camera_parameters","layer.5.camera.10.pred_cam",{1,3}},
        {"pose_raw","layer.5.pose.pred_pose_raw",{1,266}},
        {"global_rotation","layer.5.pose.global_rot",{1,3}},
        {"body_pose","layer.5.pose.body_pose",{1,133}},
        {"shape","layer.5.pose.pose.24.shape",{1,45}},
        {"scale","layer.5.pose.pose.25.scale",{1,28}},
        {"hand","layer.5.pose.pose.26.hand",{1,108}},
        {"face","layer.5.pose.pose.27.face",{1,72}},
        {"mhr_model_parameters","layer.5.pose.pose.90.model_params",{1,204}},
        {"hand_boxes","branch.hand_box",{1,2,4}},
        {"hand_logits","branch.hand_logits",{1,2,2}},
        {"joint_transforms","layer.5.pose.mhr.skeleton",{1,127,8}}
    };return fields;
}
std::unique_ptr<s3d_body_result,decltype(&s3d_body_result_free)> make_body_result(
    std::map<std::string,std::vector<float>> values,std::span<const int32_t> faces){
    auto result=std::unique_ptr<s3d_body_result,decltype(&s3d_body_result_free)>(new s3d_body_result,s3d_body_result_free);
    for(auto &field:body_output_fields()){
        auto it=values.find(field.source);const auto count=std::accumulate(field.shape.begin(),field.shape.end(),uint64_t(1),std::multiplies<>());
        if(it==values.end() || it->second.size()!=count || !std::all_of(it->second.begin(),it->second.end(),[](float v){return std::isfinite(v);}))throw std::runtime_error("invalid native Body result tensor");
        result->tensors.push_back({field.name,field.shape,std::move(it->second),{}});
    }
    if(faces.size()!=36874*3 || !std::all_of(faces.begin(),faces.end(),[](int32_t v){return v>=0 && v<18439;}))throw std::runtime_error("invalid native Body topology");
    result->tensors.push_back({"faces",{36874,3},{},{faces.begin(),faces.end()}});return result;
}
}
void s3d_body_result_free(s3d_body_result *r){delete r;}
s3d_status s3d_body_result_get_count(const s3d_body_result *r,uint32_t *count,char *e,uint64_t n){
    if(count)*count=0;return s3d::boundary(e,n,[&]{s3d::require(r && count,"result and count required");*count=uint32_t(r->tensors.size());});
}
s3d_status s3d_body_result_get_tensor(const s3d_body_result *r,uint32_t i,const char **name,uint32_t *dtype,uint32_t *rank,uint64_t *elements,const void **data,char *e,uint64_t n){
    if(name)*name=nullptr;if(dtype)*dtype=0;if(rank)*rank=0;if(elements)*elements=0;if(data)*data=nullptr;
    return s3d::boundary(e,n,[&]{s3d::require(r && i<r->tensors.size() && name && dtype && rank && elements && data,"valid result, tensor index and outputs required");
        auto &t=r->tensors[i];*name=t.name.c_str();*rank=uint32_t(t.shape.size());*dtype=t.integers.empty()?S3D_DTYPE_F32:S3D_DTYPE_I32;
        *elements=t.integers.empty()?t.floats.size():t.integers.size();*data=t.integers.empty()?static_cast<const void*>(t.floats.data()):static_cast<const void*>(t.integers.data());});
}
s3d_status s3d_body_result_get_dimension(const s3d_body_result *r,uint32_t i,uint32_t axis,uint64_t *dimension,char *e,uint64_t n){
    if(dimension)*dimension=0;return s3d::boundary(e,n,[&]{s3d::require(r && i<r->tensors.size() && axis<r->tensors[i].shape.size() && dimension,"valid result tensor, axis and dimension required");*dimension=r->tensors[i].shape[axis];});
}
s3d_status s3d_body_result_get_metadata(const s3d_body_result *r,const char *key,uint64_t bytes,const char **value,uint64_t *value_bytes,char *e,uint64_t n){
    if(value)*value=nullptr;
    if(value_bytes)*value_bytes=0;
    return s3d::boundary(e,n,[&]{
        s3d::require(r && key && bytes>0 && bytes<=64 && value && value_bytes,"result, bounded metadata key and outputs required");
        static const std::map<std::string_view,const char*> metadata={
            {"schema","sam3d.body.pose_branch.v1"},
            {"scope","one_person;no_mask;body_pose_branch;no_hand_crop_refinement"},
            {"geometry_units","metres"},
            {"coordinates","vertices/joints/keypoints: MHR multiplied by diag(1,-1,-1); add camera_translation for projection"},
            {"joint_transform_coordinates","MHR global translation centimetres, quaternion XYZW, uniform scale; 8 floats per joint"},
            {"joint_rotation_coordinates","original MHR global rotation matrices; not axis-flipped or parent-local"}};
        auto it=metadata.find(std::string_view(key,size_t(bytes)));s3d::require(it!=metadata.end(),"unknown result metadata key");
        *value=it->second;*value_bytes=std::strlen(it->second);
    });
}
