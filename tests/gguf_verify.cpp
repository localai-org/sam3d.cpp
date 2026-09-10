#include "tensor_archive.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc,char **argv) {
    try {
        if (argc!=2) throw std::invalid_argument("usage: gguf-verify BACKBONE.gguf | --schema");
        const auto shapes=sam3d::body_dino_shapes();
        if (std::string(argv[1])=="--schema") {
            for (const auto &[name,shape]:shapes) {
                std::cout<<name;
                for (auto dimension:shape) std::cout<<' '<<dimension;
                std::cout<<'\n';
            }
            return 0;
        }
        sam3d::tensor_archive archive(argv[1],"sam3d.body.dinov3.vith16plus",shapes);
        for (const auto &[name,shape]:shapes) archive.read(name,64*1024*1024);
        std::cout<<"Validated "<<archive.tensor_count()<<" F32 backbone tensors; not model parity\n";
    } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
