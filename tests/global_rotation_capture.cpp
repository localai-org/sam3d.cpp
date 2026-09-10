#include "body_pose.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc,char **argv){try{
    if(argc!=3)throw std::invalid_argument("expected input/output");
    std::ifstream in(argv[1],std::ios::binary);uint32_t b=0;
    if(!in.read(reinterpret_cast<char *>(&b),4) || b<1 || b>64)throw std::invalid_argument("invalid batch");
    std::vector<float> values(b*6);if(!in.read(reinterpret_cast<char *>(values.data()),values.size()*4) || in.peek()!=std::ifstream::traits_type::eof())throw std::invalid_argument("invalid input");
    auto out=sam3d::body_global_rotation(b,values);std::ofstream file(argv[2],std::ios::binary);
    for(auto &[name,v]:out){std::cerr<<name<<' '<<v.size()<<'\n';if(!file.write(reinterpret_cast<const char *>(v.data()),v.size()*4))throw std::runtime_error("write failed");}
    file.close();if(!file)throw std::runtime_error("close failed");
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
