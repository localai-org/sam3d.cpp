#include "body_result.hpp"
#include <cstdlib>
#include <cstring>
#include <numeric>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *input,size_t size){
    if(size<12)return 0;
    // A fixed immutable owned result exercises valid as well as invalid getters.
    // No GGUF, model loading, neural inference or fabricated acceptance output.
    static auto result=[] {
        std::map<std::string,std::vector<float>> values;
        for(auto &f:sam3d::body_output_fields())values[f.source]=std::vector<float>(std::accumulate(f.shape.begin(),f.shape.end(),uint64_t(1),std::multiplies<>()),1.f);
        std::vector<int32_t> faces(36874*3,0);return sam3d::make_body_result(std::move(values),faces);
    }();
    uint32_t index,axis;std::memcpy(&index,input,4);std::memcpy(&axis,input+4,4);
    if(input[8]&1)index%=19;if(input[8]&2)axis%=4;
    char error[17];uint64_t capacity=input[9]%sizeof(error);
    const char *name=nullptr;const void *data=nullptr;uint32_t type=0,rank=0;uint64_t n=0,dim=0;
    auto status=s3d_body_result_get_tensor(result.get(),index,&name,&type,&rank,&n,&data,error,capacity);
    if(index<19){if(status!=S3D_OK || !name || !data || !rank || !n)std::abort();
        if(type==S3D_DTYPE_F32){if(static_cast<const float*>(data)[n-1]!=1.f)std::abort();}
        else if(type!=S3D_DTYPE_I32 || static_cast<const int32_t*>(data)[n-1]!=0)std::abort();
    }else if(status!=S3D_INVALID_ARGUMENT || name || data || type || rank || n)std::abort();
    status=s3d_body_result_get_dimension(result.get(),index,axis,&dim,error,capacity);
    if(index<19 && axis<rank){if(status!=S3D_OK || !dim)std::abort();}
    else if(status!=S3D_INVALID_ARGUMENT || dim)std::abort();
    const char *value=nullptr;uint64_t bytes=0;
    s3d_body_result_get_metadata(result.get(),reinterpret_cast<const char*>(input),size,&value,&bytes,error,capacity);
    if(s3d_body_result_get_metadata(result.get(),"schema",6,&value,&bytes,error,capacity)!=S3D_OK || !value || !bytes)std::abort();
    return 0;
}
