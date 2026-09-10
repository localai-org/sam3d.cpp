#include "body_pipeline.hpp"
#include <iostream>
#include <numeric>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("usage: body-branch-verify GGUF|--schema");
    const auto shapes=sam3d::body_branch_shapes();
    const sam3d::body_pipeline_shape shape{{1,512,512,16,1280,20,5120,32,4},{{1,512,512,16,1280,1024,1,70,519,70,true,true},6,8,64,1024,1024,2,1024,2,1.f,true,false,true},true};
    auto required=sam3d::body_pipeline_parameter_sizes(shape);
    if(shapes.size()!=316 || required.size()+3!=shapes.size())throw std::runtime_error("branch archive/runtime schema count mismatch");
    for(auto &[name,n]:required){auto i=shapes.find(name);if(i==shapes.end() || sam3d::body_branch_integer_tensor(name) || std::accumulate(i->second.begin(),i->second.end(),uint64_t(1),std::multiplies<>())!=n)throw std::runtime_error("branch archive/runtime tensor mismatch: "+name);}
    if(std::string(argv[1])=="--schema"){
        std::cout<<"{";bool first=true;
        for(auto &[name,dims]:shapes){if(!first)std::cout<<",";first=false;std::cout<<'"'<<name<<"\":[";for(size_t i=0;i<dims.size();++i){if(i)std::cout<<",";std::cout<<dims[i];}std::cout<<"]";}
        std::cout<<"}\n";return 0;
    }
    sam3d::tensor_archive archive(argv[1],"sam3d.body.pose_branch",shapes);uint64_t elements=0;
    for(auto &[name,_]:shapes){const auto n=sam3d::body_branch_integer_tensor(name)?archive.read_i32(name,256*1024*1024).size():archive.read(name,256*1024*1024).size();elements+=n;}
    std::cout<<"Checked trained Body branch GGUF: "<<shapes.size()<<" tensors, "<<elements<<" elements; inference not tested\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
