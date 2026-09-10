#include "tensor_archive.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=2)throw std::invalid_argument("expected MHR GGUF path");auto shapes=sam3d::mhr_lod1_shapes();sam3d::tensor_archive a(argv[1],"sam3d.mhr.lod1",shapes);uint64_t elements=0;
    for(auto &[name,shape]:shapes){size_t count=0;
        if(sam3d::mhr_integer_tensor(name))count=a.read_i32(name,1024*1024*1024).size();
        else{auto v=a.read(name,1024*1024*1024);count=v.size();if(name=="skin.weights" && !std::all_of(v.begin(),v.end(),[](float x){return x>=0 && x<=1;}))throw std::invalid_argument("invalid skin weight");}
        elements+=count;std::cout<<name<<" "<<count<<'\n';}
    std::cout<<"Checked MHR GGUF: "<<a.tensor_count()<<" tensors, "<<elements<<" elements; geometry inference not tested\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
