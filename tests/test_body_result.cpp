#include "body_result.hpp"
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
int main(){try{
    auto require=[](bool b){if(!b)throw std::runtime_error("Body result contract failed");};
    std::map<std::string,std::vector<float>> values;std::vector<int32_t> faces(36874*3,7);
    for(auto &f:sam3d::body_output_fields())values[f.source]=std::vector<float>(std::accumulate(f.shape.begin(),f.shape.end(),uint64_t(1),std::multiplies<>()),float(values.size()+1));
    auto bad=values;bad.begin()->second[0]=std::numeric_limits<float>::quiet_NaN();bool rejected=false;
    try{sam3d::make_body_result(std::move(bad),faces);}catch(const std::runtime_error &){rejected=true;}require(rejected);
    auto result=sam3d::make_body_result(std::move(values),faces);faces.clear();values.clear();
    uint32_t count=0;require(s3d_body_result_get_count(result.get(),&count,nullptr,0)==S3D_OK && count==19);
    const char *metadata=nullptr;uint64_t metadata_bytes=0;
    require(s3d_body_result_get_metadata(result.get(),"schema",6,&metadata,&metadata_bytes,nullptr,0)==S3D_OK && std::string(metadata)=="sam3d.body.pose_branch.v1" && metadata_bytes==25);
    require(s3d_body_result_get_metadata(result.get(),"bad",3,&metadata,&metadata_bytes,nullptr,0)==S3D_INVALID_ARGUMENT && !metadata && !metadata_bytes);
    for(uint32_t i=0;i<count;++i){const char *name=nullptr;const void *data=nullptr;uint32_t type=0,rank=0;uint64_t n=0,total=1;
        require(s3d_body_result_get_tensor(result.get(),i,&name,&type,&rank,&n,&data,nullptr,0)==S3D_OK && data);
        for(uint32_t j=0;j<rank;++j){uint64_t dim=0;require(s3d_body_result_get_dimension(result.get(),i,j,&dim,nullptr,0)==S3D_OK);total*=dim;}
        require(n==total);uint64_t dim=9;require(s3d_body_result_get_dimension(result.get(),i,rank,&dim,nullptr,0)==S3D_INVALID_ARGUMENT && !dim);
        if(i<18){require(name==std::string(sam3d::body_output_fields()[i].name) && type==S3D_DTYPE_F32 && static_cast<const float*>(data)[0]==float(i+1));}
        else require(std::string(name)=="faces" && type==S3D_DTYPE_I32 && static_cast<const int32_t*>(data)[0]==7);
    }
    uint64_t dim=1;require(s3d_body_result_get_dimension(result.get(),UINT32_MAX,0,&dim,nullptr,0)==S3D_INVALID_ARGUMENT && !dim);
    std::cout<<"Owned result lifetime, all fields, dtype, shape and invalid indices pass\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
